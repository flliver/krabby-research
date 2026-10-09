"""Omni phase-2 stage p2r, the statue amendment's code (``P2R_STATUE_PLAN.md`` step 1; CHARTER 12), on CPU.

Pinned here (no Isaac, no GPU):

* ``KRABBY_OMNI_STAND_OBS`` = ``0`` / ``1`` / ``clock`` (``motion_knobs.parse_stand_obs``): unset / ``0`` write
  nothing, ``1`` writes exactly what it always did, ``clock`` writes ``stand_bit_clock`` (no ``stand_obs``), arms
  the eval slot latch and is refused without the hidden clock;
* the observation term under ``stand_bit_clock``: the latched stand slot in the CURRENT step's column 15 only --
  the history buffer's column 15 stays 0 (both the per-step push and the fresh-episode replication);
* the mirror: column 15 is +1 under both planes with the param, the shipped tables without it;
* the command term: an episode reset re-derives the eval slot latch (the suite-wave carry-over);
* the three runner-side knobs -> ``agent_cfg.algorithm.anchor_coef / anchor_ckpt / stand_prior_coef`` (+
  ``stand_mask_col``) and their refusals;
* ``PPOWithExtractor``'s two masked mean losses: weights 0 -> the update is bitwise unchanged; each loss's
  gradient comes only from its own rows and never reaches the critic or the std; the anchor is exactly 0 at the
  fork; the normalisation is sum / B; the gradient-norm split runs every 50 iterations and moves nothing;
* the ladder's dump checks on the REAL S1 dump (knobs off pass, armed claims fail) and on an armed dump, and the
  narrowed post-run clock check.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
tensordict = pytest.importorskip("tensordict")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_MIRROR_PY = OMNI / "mdp/crab_hex_omni_mirror.py"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
EVAL_PY = TASKS / "crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
#: The S1 s18 training run (the head of record's source; read-only) and its graft (may not exist yet).
S1_RUN = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
GRAFT = REPO / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The PPO helpers first (a tiny real CrabHexActorCriticRMA + real PPOWithExtractor + RolloutStorage), then the
# path-loaded helpers of the sibling unit modules (their stubs are designed to coexist with any collection order).
PH = _load(UNIT / "test_ppo_construction_knobs.py", "p2r_statue__ppo_helpers")
ppo_mod = PH.ppo_mod
PPO = PH.PPO
mk = _load(MOTION_KNOBS_PY, "motion_knobs__p2r_statue_test")
OH = _load(UNIT / "test_crab_hex_omni_observations.py", "p2r_statue__obs_helpers")
MH = _load(UNIT / "test_crab_hex_omni_mirror.py", "p2r_statue__mirror_helpers")
CH = _load(UNIT / "test_crab_hex_omni_command.py", "p2r_statue__cmd_helpers")

STATUE_KNOBS = ("KRABBY_ANCHOR_W", "KRABBY_ANCHOR_CKPT", "KRABBY_STAND_PRIOR_W")


# ===================================================================== the column contract
def test_the_stand_bit_column_is_the_clock_sine_everywhere():
    assert mk.STAND_BIT_OBS_COL == 15
    assert OH.OmniObs.STAND_BIT_CLOCK_COL == 15
    assert MH.omni.STAND_BIT_CLOCK_SLOT == 15 and MH.omni.OBS_STAND_CLOCK_PARAM == "stand_bit_clock"
    # the eval harness records obs[:, 15:17] as ``clock_obs`` -- so clock_obs[:, 0] IS the raw bit
    assert "OBS_IDX_CLOCK = slice(15, 17)" in EVAL_PY.read_text()
    assert mk.STATUE_KNOBS == STATUE_KNOBS
    assert set(STATUE_KNOBS) <= set(mk.AGENT_KNOBS) and set(STATUE_KNOBS) <= set(mk.MOTION_KNOB_PARSERS)
    # not carried into evals, and not PPO-construction knobs (that tuple is pinned elsewhere)
    assert not set(STATUE_KNOBS) & set(mk.EVAL_SEMANTIC_KNOBS + mk.EVAL_STRUCTURE_KNOBS + mk.PPO_CONSTRUCTION_KNOBS)


# ===================================================================== the env-side knob (pure)
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


class _OmniObsFunc:
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}


def _cfg(clock_hide=True, ramp=None):
    """A stand-in env cfg as the clock knobs leave it (they run before the motion knobs)."""
    cfg = _Bag()
    cfg.commands = _Bag()
    cfg.commands.base_velocity = _Bag()
    cfg.commands.base_velocity.ang_vel_max = 0.3
    cfg.rewards = _Bag()
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    params = {"history_length": 10, "facing_obs_encoding": "raw", "facing_obs_clip": 0.6}
    if clock_hide:
        params["clock_hide"] = True
    if ramp is not None:
        params["clock_hide_ramp"] = ramp
    cfg.observations.policy.extreme_parkour_observations = _Term(func=_OmniObsFunc, params=params)
    return cfg


def _snap(obj):
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items() if k != "func"}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


def _dump(cfg) -> str:
    return json.dumps(_snap(cfg), sort_keys=True, default=str)


def test_parse_stand_obs():
    for off in (None, "", " ", "0"):
        assert mk.parse_stand_obs(off) is None
    assert mk.parse_stand_obs("1") == "1" and mk.parse_stand_obs(" clock ") == "clock"
    assert mk.parse_stand_obs("CLOCK") == "clock"
    for bad in ("2", "true", "clk", "bit"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_STAND_OBS"):
            mk.parse_stand_obs(bad)
    assert mk.MOTION_KNOB_PARSERS["KRABBY_OMNI_STAND_OBS"] is mk.parse_stand_obs
    assert mk.MOTION_KNOB_PARSERS["KRABBY_OMNI_STAND_OBS"](mk.KNOB_OFF_VALUES["KRABBY_OMNI_STAND_OBS"]) is None


def test_stand_obs_unset_or_0_is_byte_identical_and_1_is_unchanged():
    pristine = _dump(_cfg())
    for env in ({}, {"KRABBY_OMNI_STAND_OBS": ""}, {"KRABBY_OMNI_STAND_OBS": "0"}):
        cfg = _cfg()
        mk.apply_motion_knobs(cfg, env)
        assert _dump(cfg) == pristine, env
    # ``1``: exactly the pre-p2r write -- the appended head bit and the eval latch, nothing else
    cfg = _cfg()
    mk.apply_motion_knobs(cfg, {"KRABBY_OMNI_STAND_OBS": "1"})
    want = _cfg()
    want.observations.policy.extreme_parkour_observations.params["stand_obs"] = True
    want.commands.base_velocity.eval_slot_latch = True
    assert _dump(cfg) == _dump(want)
    assert "stand_bit_clock" not in cfg.observations.policy.extreme_parkour_observations.params
    # ``1`` never needed the hidden clock and still does not
    visible = _cfg(clock_hide=False)
    mk.apply_motion_knobs(visible, {"KRABBY_OMNI_STAND_OBS": "1"})
    assert visible.observations.policy.extreme_parkour_observations.params["stand_obs"] is True


def test_stand_obs_clock_writes_the_clock_bit_and_arms_the_latch(capsys):
    cfg = _cfg()
    mk.apply_motion_knobs(cfg, {"KRABBY_OMNI_STAND_OBS": "clock"})
    params = cfg.observations.policy.extreme_parkour_observations.params
    assert params["stand_bit_clock"] is True and "stand_obs" not in params
    assert cfg.commands.base_velocity.eval_slot_latch is True
    want = _cfg()
    want.observations.policy.extreme_parkour_observations.params["stand_bit_clock"] = True
    want.commands.base_velocity.eval_slot_latch = True
    assert _dump(cfg) == _dump(want)                                  # nothing else moved
    for bad in (_cfg(clock_hide=False), _cfg(ramp=(0, 240))):
        with pytest.raises(RuntimeError, match="KRABBY_OMNI_CLOCK_HIDE"):
            mk.apply_motion_knobs(bad, {"KRABBY_OMNI_STAND_OBS": "clock"})


# ===================================================================== the runner-side knobs (pure)
def _agent():
    agent = _Bag()
    agent.policy, agent.algorithm = _Bag(), _Bag()
    agent.algorithm.entropy_coef = 0.0025
    agent.algorithm.schedule = "adaptive"
    return agent


def test_statue_parsers():
    for var in ("KRABBY_ANCHOR_W", "KRABBY_STAND_PRIOR_W"):
        p = mk.MOTION_KNOB_PARSERS[var]
        assert p("") is None and p("0") is None and p(None) is None
        assert p("1.0") == 1.0 and p("0.25") == 0.25
        for bad in ("-0.1", "-1", "nan", "x"):
            with pytest.raises(ValueError, match=var):
                p(bad)
    p = mk.MOTION_KNOB_PARSERS["KRABBY_ANCHOR_CKPT"]
    assert p("") is None and p(" /a/b.pt ") == "/a/b.pt"


def test_statue_knobs_unset_write_nothing():
    for env in ({}, {k: "" for k in STATUE_KNOBS}, {"KRABBY_ANCHOR_W": "0", "KRABBY_STAND_PRIOR_W": "0"},
                {"KRABBY_OMNI_STAND_OBS": "clock"}):
        agent = _agent()
        before = _dump(agent)
        mk.apply_exploration_knobs(agent, env)
        assert _dump(agent) == before, env


def test_statue_knobs_reach_the_algorithm_cfg(tmp_path, capsys):
    ckpt = tmp_path / "graft.pt"
    ckpt.write_bytes(b"x")
    env = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_ANCHOR_CKPT": str(ckpt),
           "KRABBY_STAND_PRIOR_W": "1.0"}
    agent = _agent()
    mk.apply_exploration_knobs(agent, env)
    alg = vars(agent.algorithm)
    assert alg == {"entropy_coef": 0.0025, "schedule": "adaptive", "anchor_coef": 1.0,
                   "anchor_ckpt": str(ckpt), "stand_prior_coef": 1.0, "stand_mask_col": 15}
    assert vars(agent.policy) == {}
    out = capsys.readouterr().out
    for var in STATUE_KNOBS:
        assert f"[KNOB] {var}=" in out and "consumed" in out
    # one weight only: only its field (+ the mask column)
    agent = _agent()
    mk.apply_exploration_knobs(agent, {**env, "KRABBY_ANCHOR_W": "0"})
    assert "anchor_coef" not in vars(agent.algorithm) and agent.algorithm.stand_prior_coef == 1.0
    assert agent.algorithm.stand_mask_col == 15


def test_statue_knob_refusals(tmp_path):
    ckpt = tmp_path / "graft.pt"
    ckpt.write_bytes(b"x")
    base = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1", "KRABBY_ANCHOR_CKPT": str(ckpt),
            "KRABBY_STAND_PRIOR_W": "1"}
    cases = [
        ({**base, "KRABBY_OMNI_STAND_OBS": "1"}, "KRABBY_OMNI_STAND_OBS=clock"),
        ({**base, "KRABBY_OMNI_STAND_OBS": "0"}, "KRABBY_OMNI_STAND_OBS=clock"),
        ({k: v for k, v in base.items() if k != "KRABBY_OMNI_STAND_OBS"}, "KRABBY_OMNI_STAND_OBS=clock"),
        ({**base, "KRABBY_ANCHOR_CKPT": ""}, "need KRABBY_ANCHOR_CKPT"),
        ({**base, "KRABBY_ANCHOR_CKPT": "", "KRABBY_ANCHOR_W": "0"}, "need KRABBY_ANCHOR_CKPT"),
        ({**base, "KRABBY_ANCHOR_W": "0", "KRABBY_STAND_PRIOR_W": "0"}, "train nothing"),
        ({**base, "KRABBY_ANCHOR_CKPT": str(tmp_path / "missing.pt")}, "not a file"),
        ({**base, "KRABBY_ANCHOR_W": "-1"}, ">= 0"),
        ({**base, "KRABBY_STAND_PRIOR_W": "-0.5"}, ">= 0"),
    ]
    for env, match in cases:
        agent = _agent()
        before = _dump(agent)
        with pytest.raises(ValueError, match=match):
            mk.apply_exploration_knobs(agent, env)
        assert _dump(agent) == before, env                            # refused before anything is written


# ===================================================================== the observation term
def _obs_term(**params):
    env = OH._Env(OH.N, torch.zeros(OH.N, 4), torch.zeros(OH.N))
    term = OH.OmniObs(OH._ParamCfg(**params), env)
    return term, env


def _history(obs):
    return obs[:, -OH.HISTORY * OH.N_PROP:].reshape(OH.N, OH.HISTORY, OH.N_PROP)


def test_clock_bit_is_written_into_the_current_step_only(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env = _obs_term(clock_hide=True, stand_bit_clock=True)
    plain, plain_env = _obs_term(clock_hide=True)
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd_term.slot_id = torch.tensor([0, 1, 2, 3, 0, 1])                  # rows 0 and 4 stand
    plain_env.command_manager.get_term("base_velocity").slot_id = cmd_term.slot_id.clone()
    stand = cmd_term.slot_id == 0
    for step in range(3):
        if step == 1:                                                     # a fresh episode on row 2
            env.episode_length_buf[2] = 1
            plain_env.episode_length_buf[2] = 1
        else:
            env.episode_length_buf[:] = 50
            plain_env.episode_length_buf[:] = 50
        obs = term(env, None, None, "base_parkour", OH.HISTORY, clock_hide=True, stand_bit_clock=True)
        ref = plain(plain_env, None, None, "base_parkour", OH.HISTORY, clock_hide=True)
        assert obs.shape == ref.shape == (OH.N, 1149)                       # width unchanged
        assert torch.equal(obs[:, 15], stand.float())                     # the latched slot, current step
        assert torch.equal(obs[:, 16], torch.zeros(OH.N))                 # col 16 stays the hidden clock
        assert torch.equal(ref[:, 15], torch.zeros(OH.N))                 # knob off: hidden clock
        # every other column is the knob-off observation, bitwise -- the history block included
        keep = torch.ones(obs.shape[1], dtype=torch.bool)
        keep[15] = False
        assert torch.equal(obs[:, keep], ref[:, keep])
        assert torch.equal(_history(obs)[:, :, 15], torch.zeros(OH.N, OH.HISTORY))
        assert torch.equal(term._obs_history_buffer[:, :, 15], torch.zeros(OH.N, OH.HISTORY))
        assert torch.equal(term._obs_history_buffer, plain._obs_history_buffer)
    # the bit follows the latched slot
    cmd_term.slot_id = torch.tensor([1, 0, 0, 3, 1, 0])
    obs = term(env, None, None, "base_parkour", OH.HISTORY, clock_hide=True, stand_bit_clock=True)
    assert torch.equal(obs[:, 15], torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
    assert torch.equal(term._obs_history_buffer[:, :, 15], torch.zeros(OH.N, OH.HISTORY))


def test_existing_stand_obs_1_is_unchanged_by_the_clock_param(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env = _obs_term(stand_obs=True, clock_hide=True)
    assert term.stand_bit_clock is False
    env.command_manager.get_term("base_velocity").slot_id = torch.tensor([0, 1, 2, 3, 0, 1])
    obs = term(env, None, None, "base_parkour", OH.HISTORY, stand_obs=True, clock_hide=True)
    assert torch.equal(obs[:, 17], torch.tensor([1.0, 0, 0, 0, 1.0, 0]))   # the appended bit
    assert torch.equal(obs[:, 15], torch.zeros(OH.N))                        # the clock stays hidden


def test_clock_bit_refusals(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    for params, match in (({"stand_bit_clock": True}, "clock_hide"),
                          ({"stand_bit_clock": True, "clock_hide": True, "clock_hide_ramp": (0, 240)}, "no ramp"),
                          ({"stand_bit_clock": True, "clock_hide": True, "stand_obs": True}, "alternative")):
        with pytest.raises(ValueError, match=match):
            _obs_term(**params)
    term, env = _obs_term(clock_hide=True)
    with pytest.raises(ValueError, match="construction-time"):
        term(env, None, None, "base_parkour", OH.HISTORY, clock_hide=True, stand_bit_clock=True)
    term, env = _obs_term(clock_hide=True, stand_bit_clock=True)
    with pytest.raises(ValueError, match="construction-time"):
        term(env, None, None, "base_parkour", OH.HISTORY, clock_hide=True)


# ===================================================================== the mirror
class _ClockEnv(MH._Env):
    def __init__(self, *, stand_bit_clock=False):
        super().__init__("raw")
        params = self.cfg.observations.policy.extreme_parkour_observations.params
        params["clock_hide"] = True
        if stand_bit_clock:
            params["stand_bit_clock"] = True


def test_mirror_keeps_the_clock_bit_under_both_planes():
    lr, fa, lr_perm, fa_perm = MH.omni.extend_head(MH.omni.HEAD_SIGNS_LR, MH.omni.HEAD_SIGNS_FA,
                                                   stand_bit_clock=True)
    assert lr_perm is None and fa_perm is None and len(lr) == len(fa) == 17
    assert lr[15] == fa[15] == 1.0 and MH.omni.HEAD_SIGNS_LR[15] == -1.0          # the clock sine flips L/R
    assert [i for i in range(17) if lr[i] != MH.omni.HEAD_SIGNS_LR[i]] == [15]
    assert fa == MH.omni.HEAD_SIGNS_FA                                            # fore-aft was already +1
    # knob off: the shipped mirror, bitwise
    base_lr, base_fa = MH.omni.build_mirrors(MH._Env("raw"), MH.OBS_DIM)
    off_lr, off_fa = MH.omni.build_mirrors(_ClockEnv(), MH.OBS_DIM)
    for a, b in ((base_lr, off_lr), (base_fa, off_fa)):
        assert torch.equal(a.obs_perm, b.obs_perm) and torch.equal(a.obs_sign, b.obs_sign)
    on_lr, on_fa = MH.omni.build_mirrors(_ClockEnv(stand_bit_clock=True), MH.OBS_DIM)
    x = torch.randn(4, MH.OBS_DIM)
    x[:, 15] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    for m in (on_lr, on_fa):
        y = m.mirror_obs(x)
        assert torch.equal(y[:, 15], x[:, 15])
        assert torch.equal(m.mirror_obs(y), x)
    diff = (on_lr.obs_sign != base_lr.obs_sign).nonzero().reshape(-1).tolist()
    hist0 = MH.OBS_DIM - MH.HISTORY * MH.N_PROP
    # column 15 of the current head and of every history row (the history copies share the head table; their
    # column 15 is always 0, so their sign is moot) -- and nothing else
    assert diff == [15] + [hist0 + k * MH.N_PROP + 15 for k in range(MH.HISTORY)], diff
    assert torch.equal(on_lr.obs_perm, base_lr.obs_perm) and torch.equal(on_fa.obs_sign, base_fa.obs_sign)


# ===================================================================== the command latch on reset
def _fake_reset(self, env_ids=None):
    ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long)
    self._resample_command(ids)
    return {}


def _command_base():
    """The stub ``CommandTerm`` the omni term really inherits from (the FIRST loader's -- collection order)."""
    return CH.upc.UniformParkourCommand.__bases__[0]


def test_reset_rederives_the_eval_slot_latch(monkeypatch):
    """The suite-wave carry-over: a turn wave latches ``turning``; the next wave re-spawns at the pinned yaw and a
    non-turning resample holds ``psi = yaw`` = the settle face, so the hook sees no fresh target. Without the reset
    re-derivation the settle would read TURN (stand bit 0); with it, STAND."""
    monkeypatch.setattr(_command_base(), "reset", _fake_reset, raising=False)
    S_STAND, S_TURN = CH.S_STAND, CH.S_TURN
    term, robot = CH._term(CH._cfg(eval_slot_latch=True), n=4, yaw=torch.zeros(4))
    assert torch.equal(term._latch_rearm, torch.zeros(4, dtype=torch.bool))
    turn = {"speed": 0.0, "dir": 0.0, "face": math.pi / 2}
    settle = {"speed": 0.0, "dir": 0.0, "face": 0.0}
    term.schedule_hook(turn)
    robot.data.heading_w[:] = math.pi / 2                               # the turn settles: still TURN
    term.schedule_hook(turn)
    assert bool((term.slot_id == S_TURN).all()) and not bool(term.stand_mask().any())

    def _wave_reset(ids=None):
        robot.data.heading_w[:] = 0.0                                   # re-spawn at the pinned yaw
        term.reset(ids)
        term.heading_target[:] = 0.0                                    # a non-turning resample: psi = yaw

    # the legacy behaviour, reproduced by undoing the reset's two writes: the latch carries over
    _wave_reset()
    term._latch_rearm = torch.zeros(4, dtype=torch.bool)
    term._hook_turning = torch.ones(4, dtype=torch.bool)
    term.schedule_hook(settle)
    assert bool((term.slot_id == S_TURN).all()), "the carry-over this fix exists for"
    # with the reset's re-derivation: STAND (raw relay output 0 at face == yaw)
    _wave_reset()
    assert bool(term._latch_rearm.all()) and not bool(term._hook_turning.any())
    term.schedule_hook(settle)
    assert bool((term.slot_id == S_STAND).all()) and bool(term.stand_mask().all())
    assert not bool(term._latch_rearm.any())                             # consumed by the hook
    term.schedule_hook(settle)
    assert bool(term.stand_mask().all())
    # a partial reset touches only its rows
    term.schedule_hook(turn)
    robot.data.heading_w[:] = math.pi / 2
    term.schedule_hook(turn)
    assert bool((term.slot_id == S_TURN).all())
    robot.data.heading_w[:] = 0.0
    term.reset(torch.tensor([1, 3]))
    assert term._latch_rearm.tolist() == [False, True, False, True]
    assert term._hook_turning.tolist() == [True, False, True, False]
    term.heading_target[:] = 0.0
    term.schedule_hook(settle)
    assert term.stand_mask().tolist() == [False, True, False, True]
    # re-DERIVED, not merely cleared: a turn hold whose target equals the resampled facing still reads TURN
    robot.data.heading_w[:] = 0.0
    term.reset()
    term.heading_target[:] = math.pi / 2                                # no fresh target at the hook
    term.schedule_hook(turn)
    assert bool((term.slot_id == S_TURN).all()) and not bool(term.stand_mask().any())


def test_reset_without_the_latch_is_the_base_reset(monkeypatch):
    monkeypatch.setattr(_command_base(), "reset", _fake_reset, raising=False)
    a, _ = CH._term(CH._cfg(), n=8, yaw=CH._spread_yaw(8))
    b, _ = CH._term(CH._cfg(), n=8, yaw=CH._spread_yaw(8))
    torch.manual_seed(3)
    a.reset()
    torch.manual_seed(3)
    _fake_reset(b)
    assert not hasattr(a, "_latch_rearm") and not hasattr(a, "_hook_turning")
    for name in ("slot_id", "travel_dir_w", "speed_cmd", "heading_target", "vel_command_b"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


# ===================================================================== PPOWithExtractor
STAND_COL = 7                     # inside the tiny policy's proprio head (NUM_PROP = 12), like col 15 of 89
ROLL_BITS = torch.tensor([1.0, 0.0, 0.0, 1.0] + [0.0] * 12)          # 2 of 16 envs stand (0.125, like 0.12)


def _statue_obs(gen: torch.Generator) -> torch.Tensor:
    obs = torch.randn(PH.N_ENVS, PH.OBS_W, generator=gen)
    obs[:, STAND_COL] = ROLL_BITS
    return obs


def _statue_rollout(alg, gen: torch.Generator) -> None:
    with torch.inference_mode():
        obs = _statue_obs(gen)
        for _ in range(PH.N_STEPS):
            alg.act(obs, obs, hist_encoding=False)
            obs = _statue_obs(gen)
            rewards = torch.randn(PH.N_ENVS, generator=gen)
            dones = (torch.rand(PH.N_ENVS, generator=gen) < 0.15).long()
            td = tensordict.TensorDict({"policy": obs, "critic": obs}, batch_size=[PH.N_ENVS])
            alg.process_env_step(td, rewards, dones, {})
        alg.compute_returns(obs)


def _soft_kw():
    return {"policy_kw": {"noise_std_type": "soft", "init_noise_std": 0.8}}


def _anchor_ckpt(tmp_path, seed=0, **kw) -> Path:
    """The fork's own weights as the anchor (``model_state_dict`` like ``runner.save``)."""
    alg = PH._alg(seed, **_soft_kw(), **kw)
    path = tmp_path / f"anchor_s{seed}.pt"
    torch.save({"model_state_dict": alg.policy.state_dict(), "estimator_state_dict": alg.estimator.state_dict()},
               path)
    return path


def _statue_alg(tmp_path, seed=0, anchor=1.0, prior=1.0, ckpt=None, **kw):
    ckpt = ckpt or _anchor_ckpt(tmp_path, seed)
    return PH._alg(seed, **_soft_kw(), anchor_coef=anchor, stand_prior_coef=prior,
                   anchor_ckpt=str(ckpt) if (anchor or prior) else None,
                   stand_mask_col=STAND_COL if (anchor or prior) else None, **kw)


def _update(alg, seed=11, iteration=None):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                       # the rollout's action sampling draws the GLOBAL RNG
    _statue_rollout(alg, gen)
    alg.iteration = iteration
    torch.manual_seed(1234)
    return alg.update()


def test_weights_zero_leave_the_update_bitwise_unchanged():
    base = PH._alg(0, **_soft_kw())
    off = PH._alg(0, **_soft_kw(), anchor_coef=0.0, stand_prior_coef=0.0, stand_mask_col=STAND_COL)
    assert off.anchor_actor is None and off.anchor_sigma is None
    for it in (0, 1, 50):                                     # the split's cadence never matters when off
        la, lb = _update(base, iteration=it), _update(off, iteration=it)
        assert la == lb
        PH._assert_same(PH._snapshot(base), PH._snapshot(off))
        assert not {"anchor", "stand_prior", "anchor_rows", "prior_rows", "anchor_mb0",
                    "grad_norm_statue", "grad_norm_anchor", "grad_norm_rest"} & set(la)


def test_construction_refusals(tmp_path):
    ckpt = _anchor_ckpt(tmp_path)
    for kw, match in (({"anchor_coef": -1.0, "anchor_ckpt": str(ckpt), "stand_mask_col": STAND_COL}, ">= 0"),
                      ({"stand_prior_coef": float("nan"), "anchor_ckpt": str(ckpt), "stand_mask_col": STAND_COL},
                       ">= 0"),
                      ({"anchor_coef": 1.0, "stand_mask_col": STAND_COL}, "need anchor_ckpt"),
                      ({"stand_prior_coef": 1.0, "stand_mask_col": STAND_COL}, "need anchor_ckpt"),
                      ({"anchor_coef": 1.0, "anchor_ckpt": str(ckpt)}, "stand_mask_col"),
                      ({"anchor_ckpt": str(ckpt)}, "both 0")):
        with pytest.raises(ValueError, match=match):
            PH._alg(0, **_soft_kw(), **kw)
    # the runner's alg-cfg filter accepts the four keys (and still refuses a typo)
    cfg = {"anchor_coef": 1.0, "anchor_ckpt": "x", "stand_prior_coef": 1.0, "stand_mask_col": 15}
    assert PH.runner_mod.filter_ppo_alg_cfg(PPO, dict(cfg)) == cfg
    with pytest.raises(ValueError, match="anchor_cof"):
        PH.runner_mod.filter_ppo_alg_cfg(PPO, {"anchor_cof": 1.0})


def test_the_anchor_is_a_frozen_copy_with_the_checkpoints_soft_sigma(tmp_path):
    ckpt = _anchor_ckpt(tmp_path, seed=5)                    # a DIFFERENT network from the seed-0 policy
    alg = _statue_alg(tmp_path, seed=0, ckpt=ckpt)
    state = torch.load(ckpt, weights_only=False)["model_state_dict"]
    assert torch.equal(alg.anchor_sigma, PH.crab_mod.soft_std(state["std_raw"]))
    assert torch.allclose(alg.anchor_sigma, torch.full((PH.N_ACT,), 0.8), atol=1e-6)
    for name, p in alg.anchor_actor.named_parameters():
        assert torch.equal(p, state[f"actor.{name}"]) and not p.requires_grad, name
    assert not alg.anchor_actor.training
    anchor_ids = {id(p) for p in alg.anchor_actor.parameters()}
    assert not anchor_ids & {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    assert not anchor_ids & {id(p) for p in alg.policy.parameters()}
    assert not any(k.startswith("anchor") for k in alg.policy.state_dict())
    before = {k: v.clone() for k, v in alg.anchor_actor.state_dict().items()}
    _update(alg, iteration=0)
    assert all(torch.equal(v, alg.anchor_actor.state_dict()[k]) for k, v in before.items())
    # a prior-only arm needs only sigma_A: no anchor network is built
    prior_only = _statue_alg(tmp_path, seed=0, anchor=0.0, ckpt=ckpt)
    assert prior_only.anchor_actor is None and torch.equal(prior_only.anchor_sigma, alg.anchor_sigma)


def _minibatch(alg, seed=3, b=24):
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(b, PH.OBS_W, generator=gen)
    bits = torch.zeros(b)
    bits[::4] = 1.0
    obs[:, STAND_COL] = bits
    return obs, bits.bool()


def test_each_loss_takes_gradient_only_from_its_own_rows(tmp_path):
    alg = _statue_alg(tmp_path, seed=0, ckpt=_anchor_ckpt(tmp_path, seed=5))     # anchor != policy: anchor live
    obs, stand = _minibatch(alg)
    b = obs.shape[0]
    mu = alg.policy.actor(obs, False)
    leaf = mu.detach().clone().requires_grad_(True)
    out = alg._statue_losses(obs, leaf, b)
    assert torch.equal(out["bit"], stand)
    g_anchor, = torch.autograd.grad(out["anchor"], leaf, retain_graph=True)
    g_prior, = torch.autograd.grad(out["prior"], leaf)
    assert torch.equal(g_anchor[stand], torch.zeros_like(g_anchor[stand]))           # exactly 0 on stand rows
    assert bool((g_anchor[~stand].abs().sum(-1) > 0).all())
    assert torch.equal(g_prior[~stand], torch.zeros_like(g_prior[~stand]))          # exactly 0 on walking rows
    assert bool((g_prior[stand].abs().sum(-1) > 0).all())
    # the closed forms: d/dmu of sum_rows 0.5 ((mu - mu_A)/s)^2 / B = (mu - mu_A) / s^2 / B
    s2 = alg.anchor_sigma.square()
    with torch.no_grad():
        mu_a = alg.anchor_actor(obs, False)
    assert torch.allclose(g_anchor[~stand], ((leaf - mu_a) / s2 / b)[~stand], atol=1e-7)
    assert torch.allclose(g_prior[stand], (leaf / s2 / b)[stand], atol=1e-7)
    # through the network: only actor parameters, never the critic or the std
    out = alg._statue_losses(obs, mu, b)
    total = out["anchor"] + out["prior"]
    critic = list(alg.policy.critic.parameters())
    assert all(g is None for g in torch.autograd.grad(total, critic, retain_graph=True, allow_unused=True))
    assert torch.autograd.grad(total, [alg.policy.std_raw], retain_graph=True, allow_unused=True)[0] is None
    # changing the OTHER group's inputs leaves a loss and its parameter gradient unchanged
    params = [p for p in alg.policy.actor.parameters() if p.requires_grad]
    g_p = torch.autograd.grad(out["prior"], params, retain_graph=True, allow_unused=True)
    obs2 = obs.clone()
    obs2[~stand, :STAND_COL] += 3.0                         # walking rows only (the bit untouched)
    out2 = alg._statue_losses(obs2, alg.policy.actor(obs2, False), b)
    g_p2 = torch.autograd.grad(out2["prior"], params, allow_unused=True)
    assert torch.allclose(out2["prior"], out["prior"], rtol=0, atol=1e-6)
    for a, c in zip(g_p, g_p2):
        assert (a is None and c is None) or torch.allclose(a, c, atol=1e-6)
    obs3 = obs.clone()
    obs3[stand, :STAND_COL] += 3.0                          # stand rows only
    out3 = alg._statue_losses(obs3, alg.policy.actor(obs3, False), b)
    assert torch.allclose(out3["anchor"], out["anchor"], rtol=0, atol=1e-6)


def test_normalisation_is_sum_over_masked_rows_over_b(tmp_path):
    alg = _statue_alg(tmp_path, seed=0, ckpt=_anchor_ckpt(tmp_path, seed=5))
    obs, stand = _minibatch(alg, b=40)
    with torch.no_grad():
        mu = alg.policy.actor(obs, False)
        mu_a = alg.anchor_actor(obs, False)
        out = alg._statue_losses(obs, mu, 40)
    s = alg.anchor_sigma
    k_anchor = 0.5 * (((mu - mu_a) / s) ** 2).sum(-1)
    k_prior = 0.5 * ((mu / s) ** 2).sum(-1)
    assert torch.allclose(out["anchor"], k_anchor[~stand].sum() / 40, rtol=1e-6)
    assert torch.allclose(out["prior"], k_prior[stand].sum() / 40, rtol=1e-6)
    # NOT the masked mean: with 10 of 40 stand rows that would be 4x larger
    assert not torch.allclose(out["prior"], k_prior[stand].mean(), rtol=1e-3)
    # the mask is a THRESHOLD on the bit: a mirrored image's -1 is not a stand row
    obs[:, STAND_COL] = torch.where(stand, torch.tensor(1.0), torch.tensor(-1.0))
    assert torch.equal(alg._statue_losses(obs, mu, 40)["bit"], stand)
    with pytest.raises(ValueError, match="stand_mask_col"):
        alg._statue_losses(obs[:, :STAND_COL], mu, 40)


def test_the_anchor_is_exactly_zero_at_the_fork_and_the_losses_log(tmp_path):
    alg = _statue_alg(tmp_path, seed=0)                    # the anchor = the fork's own weights
    obs, _ = _minibatch(alg)
    mu = alg.policy.actor(obs, False)
    assert float(alg._statue_losses(obs, mu, obs.shape[0])["anchor"]) == 0.0
    ld = _update(alg, iteration=7)
    assert ld["anchor_mb0"] == 0.0                         # before any step of the first update
    assert math.isfinite(ld["anchor"]) and ld["anchor"] > 0.0            # after the first step it moves
    assert math.isfinite(ld["stand_prior"]) and ld["stand_prior"] > 0.0
    assert ld["anchor_rows"] + ld["prior_rows"] == pytest.approx(1.0)
    assert ld["prior_rows"] == pytest.approx(float(ROLL_BITS.mean()))   # every row of the rollout seen 2x
    assert not {"grad_norm_statue", "grad_norm_anchor", "grad_norm_rest"} & set(ld)   # iteration 7: off cadence


def test_the_anchor_is_exactly_zero_at_the_fork_under_data_augmentation(tmp_path):
    """With ``use_data_augmentation`` the policy forward runs on the 2B-row augmented tensor; the anchor forwards on
    that same tensor and slices the first B rows, so the fork value is still exact (and the bit is read off the
    ORIGINAL rows: the stand-in mirror negates the augmented copies' bit to -1)."""
    aug = {"use_data_augmentation": True, "use_mirror_loss": False, "data_augmentation_func": PH._mirror,
           "mirror_loss_coeff": 0.0, "_env": None}
    alg = _statue_alg(tmp_path, seed=0, symmetry_cfg=aug)
    ld = _update(alg, iteration=1)
    assert ld["anchor_mb0"] == 0.0 and ld["anchor"] > 0.0
    assert ld["prior_rows"] == pytest.approx(float(ROLL_BITS.mean()))


def test_the_statue_terms_change_the_update_only_through_their_weights(tmp_path):
    """Armed at the fork with the prior OFF: the anchor is 0 with a 0 gradient on minibatch 0, so that first step
    is the knob-off step; the run diverges afterwards only because the anchor then pulls."""
    ckpt = _anchor_ckpt(tmp_path, seed=0)
    base = PH._alg(0, **_soft_kw(), num_learning_epochs=1, num_mini_batches=1)
    armed = PH._alg(0, **_soft_kw(), num_learning_epochs=1, num_mini_batches=1, anchor_coef=1.0,
                    anchor_ckpt=str(ckpt), stand_mask_col=STAND_COL)
    la, lb = _update(base, iteration=3), _update(armed, iteration=3)
    assert lb["anchor_mb0"] == 0.0 and lb["anchor"] == 0.0
    PH._assert_same(PH._snapshot(base), PH._snapshot(armed))           # one minibatch: bitwise the same step
    assert {k: v for k, v in lb.items() if k not in ("anchor", "anchor_mb0", "anchor_rows", "prior_rows")} == la


def test_grad_norm_split_cadence_and_inertness(tmp_path):
    ckpt = _anchor_ckpt(tmp_path, seed=5)
    on = _statue_alg(tmp_path, seed=0, ckpt=ckpt)
    off = _statue_alg(tmp_path, seed=0, ckpt=ckpt)
    lon = _update(on, iteration=50)
    loff = _update(off, iteration=51)
    for key in ("grad_norm_statue", "grad_norm_anchor", "grad_norm_rest"):
        assert key in lon and math.isfinite(lon[key]) and lon[key] > 0.0, key
        assert key not in loff
    assert {k: v for k, v in lon.items() if not k.startswith("grad_norm_")} == loff
    PH._assert_same(PH._snapshot(on), PH._snapshot(off))                 # logging moved nothing (RNG included)
    for it, armed in ((0, True), (1, False), (49, False), (100, True), (150, True)):
        alg = _statue_alg(tmp_path, seed=0, ckpt=ckpt)
        assert ("grad_norm_rest" in _update(alg, iteration=it)) is armed, it
    # a weight that is off logs no norm of its own
    prior_only = _statue_alg(tmp_path, seed=0, anchor=0.0, ckpt=ckpt)
    ld = _update(prior_only, iteration=0)
    assert "grad_norm_statue" in ld and "grad_norm_anchor" not in ld and "anchor" not in ld
    assert "anchor_mb0" not in ld and "stand_prior" in ld
    # the split's statue norm is the weighted prior's actor gradient on minibatch 0 (independent recompute)
    alg = _statue_alg(tmp_path, seed=0, anchor=0.0, prior=2.0, ckpt=ckpt, num_learning_epochs=1,
                      num_mini_batches=1)
    gen = torch.Generator().manual_seed(11)
    _statue_rollout(alg, gen)
    obs = alg.storage.observations["policy"].reshape(-1, PH.OBS_W).clone()
    params = [p for p in alg.policy.actor.parameters() if p.requires_grad]
    mu = alg.policy.actor(obs, False)
    want = torch.autograd.grad(2.0 * alg._statue_losses(obs, mu, obs.shape[0])["prior"], params, allow_unused=True)
    want_norm = math.sqrt(sum(float(g.square().sum()) for g in want if g is not None))
    alg.iteration = 0
    torch.manual_seed(1234)
    got = alg.update()["grad_norm_statue"]
    assert got == pytest.approx(want_norm, rel=1e-5)


# ===================================================================== the ladder's dump checks
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__p2r_statue_test")


def _s1_dumps():
    env, agent = S1_RUN / "params/env.yaml", S1_RUN / "params/agent.yaml"
    if not (env.exists() and agent.exists()):
        pytest.skip("the S1 s18 training dump is not on this machine")
    return env.read_text(), agent.read_text()


def _armed_dumps(env_txt: str, agent_txt: str, ckpt: str) -> tuple[str, str]:
    """The S1 dumps as the trainer would write them with the p2r knobs on (text edits of the real files)."""
    assert "        clock_hide: true\n" in env_txt
    env_txt = env_txt.replace("        clock_hide: true\n", "        clock_hide: true\n        stand_bit_clock: true\n", 1)
    marker = "  base_velocity:\n"
    assert env_txt.count(marker) == 1
    env_txt = env_txt.replace(marker, marker + "    eval_slot_latch: true\n", 1)
    marker = "  hand_diag_vy_obs_col: 9\n"
    assert agent_txt.count(marker) == 1
    agent_txt = agent_txt.replace(marker, marker + f"  anchor_coef: 1.0\n  anchor_ckpt: {ckpt}\n"
                                  "  stand_prior_coef: 1.0\n  stand_mask_col: 15\n", 1)
    return env_txt, agent_txt


def _run_dir(root: Path, env_txt: str, agent_txt: str, log: str | None = None) -> dict:
    params = root / "params"
    params.mkdir(parents=True)
    (params / "env.yaml").write_text(env_txt)
    (params / "agent.yaml").write_text(agent_txt)
    (root / "model_1.pt").write_text("")
    rec = {"ckpt": str(root / "model_1.pt")}
    if log is not None:
        (root / "train.log").write_text(log)
        rec["log"] = str(root / "train.log")
    return rec


def test_statue_knobs_are_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in STATUE_KNOBS + ("KRABBY_OMNI_STAND_OBS",):
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
        assert parsers[var] is rc.knobs()._mk.MOTION_KNOB_PARSERS[var], var
    assert set(STATUE_KNOBS) <= rc.AGENT_DUMP_KEYS and "KRABBY_OMNI_STAND_OBS" not in rc.AGENT_DUMP_KEYS
    assert rc.STAND_BIT_HOLD_LABELS == ("stand", "settle")
    reasons = []
    rc._check_knob("KRABBY_OMNI_STAND_OBS", "clock", "t", whitelist, parsers, reasons, set())
    rc._check_knob("KRABBY_ANCHOR_W", "1.0", "t", whitelist, parsers, reasons, set())
    assert not reasons, reasons
    rc._check_knob("KRABBY_OMNI_STAND_OBS", "clk", "t", whitelist, parsers, reasons, set())
    rc._check_knob("KRABBY_STAND_PRIOR_W", "-1", "t", whitelist, parsers, reasons, set())
    assert len(reasons) == 2, reasons


def test_dump_checks_on_the_real_s1_dump(rc, tmp_path):
    env_txt, agent_txt = _s1_dumps()
    checks = rc.OVERLAY_DUMP_CHECKS
    ckpt = str(GRAFT)
    # knobs off: every off claim passes on S1, every armed claim fails
    for key, value in (("KRABBY_OMNI_STAND_OBS", "0"), ("KRABBY_ANCHOR_W", "0"), ("KRABBY_STAND_PRIOR_W", "0"),
                       ("KRABBY_ANCHOR_CKPT", "")):
        txt = agent_txt if key in rc.AGENT_DUMP_KEYS else env_txt
        assert checks[key](txt, value), (key, value)
    for key, value in (("KRABBY_OMNI_STAND_OBS", "clock"), ("KRABBY_OMNI_STAND_OBS", "1"),
                       ("KRABBY_ANCHOR_W", "1.0"), ("KRABBY_STAND_PRIOR_W", "1.0"), ("KRABBY_ANCHOR_CKPT", ckpt)):
        txt = agent_txt if key in rc.AGENT_DUMP_KEYS else env_txt
        assert not checks[key](txt, value), (key, value)
    off = {"KRABBY_OMNI_STAND_OBS": "0", "KRABBY_ANCHOR_W": "0", "KRABBY_STAND_PRIOR_W": "0", "KRABBY_ANCHOR_CKPT": ""}
    got = rc.overlay_checks(_run_dir(tmp_path / "off", env_txt, agent_txt), off)
    assert got and all(ok for _n, ok in got), got
    armed = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
             "KRABBY_ANCHOR_CKPT": ckpt}
    log = "".join(f"[KNOB] {k}={v} consumed\n" for k, v in armed.items() if k != "KRABBY_OMNI_STAND_OBS")
    log += "   Mean anchor: 0.0000\n   Mean stand_prior: 2.3000\n"
    got = dict(rc.overlay_checks(_run_dir(tmp_path / "s1_armed_claim", env_txt, agent_txt, log), armed))
    assert not any(v for k, v in got.items() if k.startswith("overlay ")), got   # S1 never had them
    # the armed dump: armed claims pass (dump + log), off claims fail
    env_on, agent_on = _armed_dumps(env_txt, agent_txt, ckpt)
    rec_on = _run_dir(tmp_path / "on", env_on, agent_on, log)
    got = rc.overlay_checks(rec_on, armed)
    assert got and all(ok for _n, ok in got), [n for n, ok in got if not ok]
    names = dict(got)
    assert "log: Mean anchor printed (the KRABBY_ANCHOR_W loss ran)" in names
    assert "log: [KNOB] KRABBY_ANCHOR_CKPT consumed" in names
    bad = [n for n, ok in rc.overlay_checks(rec_on, off) if not ok]
    assert len(bad) == len(off), bad
    # a claim of the OTHER stand bit fails on the clock dump, and the log half catches a loss that never ran
    assert not checks["KRABBY_OMNI_STAND_OBS"](env_on, "1")
    assert not checks["KRABBY_ANCHOR_W"](agent_on, "0.5")
    assert not checks["KRABBY_ANCHOR_CKPT"](agent_on, "/elsewhere/model.pt")
    rec_nolog = _run_dir(tmp_path / "nolog", env_on, agent_on, log.replace("   Mean anchor: 0.0000\n", ""))
    got = dict(rc.overlay_checks(rec_nolog, armed))
    assert got["log: Mean anchor printed (the KRABBY_ANCHOR_W loss ran)"] is False
    # without the latch / the hidden clock the clock claim fails
    no_latch = env_on.replace("    eval_slot_latch: true\n", "", 1)
    assert not checks["KRABBY_OMNI_STAND_OBS"](no_latch, "clock")


# ===================================================================== the narrowed post-run clock check
LABELS = ["settle", "stand", "fwd", "turn_l90"]


def _stamp(root: Path, episodes: list[tuple[np.ndarray, np.ndarray]], labels=LABELS) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True)
    for i, (clock, seg) in enumerate(episodes):
        np.savez_compressed(raw / f"episode_{i:02d}.npz", clock_obs=clock.astype(np.float32),
                            segment_id=seg.astype(np.int16))
    (root / "run_meta.json").write_text(json.dumps({"hold_labels": labels,
                                                    "suite": {"settle_label": "settle"}}))
    return root


def _episode(seg: list[int], lag: int = 0) -> tuple[np.ndarray, np.ndarray]:
    seg = np.asarray(seg)
    stand = np.array([1.0 if s >= 0 and LABELS[s] in ("stand", "settle") else 0.0 for s in seg])
    if lag:
        stand = np.concatenate([stand[:1].repeat(lag), stand[:-lag]])
    clock = np.zeros((seg.size, 2))
    clock[:, 0] = stand
    return clock, seg


def test_narrowed_clock_check(rc, tmp_path):
    seg = [-1] * 3 + [0] * 5 + [2] * 6 + [1] * 5 + [3] * 4
    ok, detail = rc._npz_clock_obs_stand_bit(_stamp(tmp_path / "a", [_episode(seg), _episode(seg, lag=1)]))
    assert ok, detail
    # a two-step lag is not the allowed one-step lag
    ok, detail = rc._npz_clock_obs_stand_bit(_stamp(tmp_path / "b", [_episode(seg, lag=2)]))
    assert not ok and "off the stand label" in detail
    # col 16 must stay exactly 0
    clock, s = _episode(seg)
    clock[10, 1] = 0.5
    assert not rc._npz_clock_obs_stand_bit(_stamp(tmp_path / "c", [(clock, s)]))[0]
    # col 15 must be a bit
    clock, s = _episode(seg)
    clock[4, 0] = 0.5
    assert not rc._npz_clock_obs_stand_bit(_stamp(tmp_path / "d", [(clock, s)]))[0]
    # a knob-off stamp (clock_obs == 0 everywhere) fails the narrowed check -- and passes the old one
    zero = (np.zeros((len(seg), 2)), np.asarray(seg))
    off_stamp = _stamp(tmp_path / "e", [zero])
    assert not rc._npz_clock_obs_stand_bit(off_stamp)[0]
    assert rc._npz_clock_obs_zero(off_stamp)[0]
    # and the old check fails a stand-bit stamp, which is why the narrowing exists
    assert not rc._npz_clock_obs_zero(tmp_path / "a")[0]
    # unreadable / missing rows fail
    assert not rc._npz_clock_obs_stand_bit(tmp_path / "nowhere")[0]
    # the stage selector reads the composed env
    assert rc._stand_bit_clock_stage({"KRABBY_OMNI_STAND_OBS": "clock"})
    assert not rc._stand_bit_clock_stage({"KRABBY_OMNI_STAND_OBS": "1"})
    assert not rc._stand_bit_clock_stage({})
    assert not rc._stand_bit_clock_stage({"KRABBY_OMNI_STAND_OBS": "clk"})


def test_narrowed_clock_check_on_real_s1_suite_raws_fails_as_knob_off(rc, tmp_path):
    """REFS raws (knob off) carry clock_obs == 0 on the stand holds: the narrowed check must call them wrong."""
    stamps = sorted((REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2/"
                     "seed18_X08_p2a_A15pB/suite__omni/seed001").glob("*/run_meta.json"))
    if not stamps:
        pytest.skip("the S1 suite stamp is not on this machine")
    run = stamps[0].parent
    # a small copy (the full stamp has 1000 episodes)
    dst = tmp_path / "s1"
    (dst / "raw").mkdir(parents=True)
    (dst / "run_meta.json").write_text((run / "run_meta.json").read_text())
    for f in sorted((run / "raw").glob("episode_*.npz"))[:40]:
        (dst / "raw" / f.name).write_bytes(f.read_bytes())
    ok, detail = rc._npz_clock_obs_stand_bit(dst)
    assert not ok and "off the stand label" in detail, detail
    # with the stand bit written where the labels say stand, the same raws pass
    meta = json.loads((dst / "run_meta.json").read_text())
    stand_ids = [i for i, n in enumerate(meta["hold_labels"]) if n in ("stand", "settle")]
    for f in sorted((dst / "raw").glob("episode_*.npz")):
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        bit = np.isin(data["segment_id"], stand_ids) | (data["segment_id"] < 0)
        data["clock_obs"] = data["clock_obs"].copy()
        data["clock_obs"][:, 0] = bit.astype(np.float32)
        np.savez_compressed(f, **data)
    ok, detail = rc._npz_clock_obs_stand_bit(dst)
    assert ok, detail
