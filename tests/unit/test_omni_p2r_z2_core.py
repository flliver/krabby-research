"""Omni phase-2 stage p2r Z2, the CORE code (``P2R_Z2_PLAN.md`` step 1; CHARTER 12), on CPU.

Pinned here (no Isaac, no GPU):

* the observation term under ``stand_proxy_hold`` (``KRABBY_OMNI_STAND_PROXY=hold``), on the REAL omni class over
  the REAL forward ``__call__`` and base ``reset`` (the harness of ``test_crab_hex_omni_observations.py`` /
  ``test_crab_hex_omni_clock_hide.py``): knob off -> bitwise Z1's observation, buffers never built, no log; on ->
  the held value follows the live proxy wherever the KNEE gate is full (never on a hip-only gate), stand rows read
  it in the CURRENT step only, walking rows and every history row stay raw, a reset restores -0.5, and
  ``Metrics/stand_proxy_live_frac`` is the live share of the (stand row, leg) entries since the last log;
* the knobs: ``KRABBY_OMNI_STAND_PROXY`` (env side, an eval-semantics knob carried outside ``run_phases``),
  ``KRABBY_STAND_PRIOR_TARGET`` / ``KRABBY_RESTANCE`` (runner side) and their cross rules, including an EVAL env
  that carries ``KRABBY_RESTANCE`` for the probe. NOTE(p2r Z2 cams-only, 2026-10-10): the target no longer needs
  ``KRABBY_OMNI_STAND_PROXY=hold`` (R reads cams + hip-pitch positions); the proxy stays an independent knob;
* ``PPOWithExtractor``'s re-stance prior: off -> Z1's update bitwise (and R == 0 reproduces it too); the prior's
  gradient comes only from bit-1 rows and equals the hand formula ``(mu - R) / sigma_A^2 / B``; the
  ``Loss/restance_*`` scalars; construction refusals and the print line;
* the ladder: dump checks, the training-log audit, the composition refusals, and a validator dry run of the
  candidates v14 chain ``P2__Z2_s18`` (= ``P2__Z1_s18`` verbatim + the TARGET / RESTANCE knobs; no proxy).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task/mdp"
FWD_OBS_PY = FWD_MDP / "observations.py"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
BASE_OBS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/observations.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
CAMP = OMNI / "experiments/2026-10-07_omni_phase2"
S1_RUN = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_unregistered(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The statue tests' module first: it loads the PPO helpers (a tiny real CrabHexActorCriticRMA + real
# PPOWithExtractor) and the obs / mirror / command harnesses; this file reuses them rather than copying.
SC = _load(UNIT / "test_omni_p2r_statue_core.py", "p2r_z2__statue_core")
PH = SC.PH
mk = _load(MOTION_KNOBS_PY, "motion_knobs__p2r_z2_test")
RST = _load_unregistered(RESTANCE_PY, "_t_p2r_z2_restance")
OH = _load(UNIT / "test_crab_hex_omni_observations.py", "p2r_z2__obs_helpers")

if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_linkage as linkage  # noqa: E402

_GATE_NS: dict = {}
exec(OH._module_assign(FWD_OBS_PY, "_CURRENT_SENSE_GATE_FRACTION"), _GATE_NS)
GATE_FRACTION = _GATE_NS["_CURRENT_SENSE_GATE_FRACTION"]
#: The knee rod speed at which the drive gate is exactly full.
KNEE_FULL = GATE_FRACTION * linkage.KNEE_SPEED_M_S

# ---- the omni class over the stub forward base WITH the real base reset (as test_crab_hex_omni_clock_hide.py)
_BASE_RESET_NS = {"torch": torch}
exec(compile("from __future__ import annotations\nfrom typing import Sequence\n"
             + OH._source_of(BASE_OBS_PY, cls="ExtremeParkourObservations", name="reset"),
             str(BASE_OBS_PY), "exec"), _BASE_RESET_NS)


class _StubBaseWithReset(OH._StubForwardObservations):
    reset = _BASE_RESET_NS["reset"]


_NS = dict(OH._OMNI_NS)
_NS.update(CrabHexParkourObservations=_StubBaseWithReset, linkage=linkage,
           _CURRENT_SENSE_GATE_FRACTION=GATE_FRACTION, ramp_value=lambda *a, **k: 0.0)
exec(
    compile(
        "from __future__ import annotations\n"
        + OH._module_assign(OMNI_OBS_PY, "FACING_OBS_ENCODINGS") + "\n"
        + OH._module_assign(OMNI_OBS_PY, "STAND_OBS_DIM") + "\n"
        + OH._module_assign(OMNI_OBS_PY, "LPF_OBS_DIM") + "\n\n"
        + OH._source_of(OMNI_OBS_PY, cls=None, name="_check_facing_obs_encoding") + "\n\n"
        + OH._source_of(OMNI_OBS_PY, cls=None, name="_check_omni") + "\n\n"
        + OH._source_of(OMNI_OBS_PY, cls=None, name="CrabHexOmniParkourObservations"),
        str(OMNI_OBS_PY),
        "exec",
    ),
    _NS,
)
OmniObs = _NS["CrabHexOmniParkourObservations"]
N, HISTORY, N_PROP = OH.N, OH.HISTORY, OH.N_PROP
C0 = 83
PROXY = slice(C0, C0 + 6)
HIST0 = OH.OBS_DIM - HISTORY * N_PROP
LOG_KEY = "Metrics/stand_proxy_live_frac"
Z1_PARAMS = {"clock_hide": True, "stand_bit_clock": True}
STAND_SLOTS = torch.tensor([0, 1, 2, 3, 0, 1])             # rows 0 and 4 stand


@pytest.fixture(autouse=True)
def _no_heading(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)


class _Live:
    """The live proxy the stub's ``_get_contact_fill`` returns (one controllable tensor per term)."""

    def __init__(self):
        self.value = torch.full((N, 6), -0.5)

    def __call__(self):
        return self.value.clone()


def _term(**params):
    env = OH._Env(N, torch.zeros(N, 4), torch.zeros(N))
    env.extras = {}
    term = OmniObs(OH._ParamCfg(**params), env)
    live = _Live()
    term._get_contact_fill = live
    act = env.action_manager.get_term("joint_pos")
    act.knee_rod_speed = torch.zeros(N, 6)
    act.hip_rod_speed = torch.zeros(N, 6)
    env.command_manager.get_term("base_velocity").slot_id = STAND_SLOTS.clone()
    return term, env, live


def _call(term, env, **params):
    return term(env, None, None, "base_parkour", HISTORY, **params)


def _history(obs):
    return obs[:, HIST0:].reshape(N, HISTORY, N_PROP)


# ===================================================================== the column contract
def test_columns_and_keys():
    assert OmniObs.STAND_PROXY_COL == RST.PROXY_OBS_COLS[0] == C0
    assert N_PROP - OH.NC == C0                                         # the fake layout puts the proxy there too
    assert OmniObs.STAND_PROXY_LIVE_FRAC_KEY == LOG_KEY and OmniObs.STAND_PROXY_UNKNOWN == -0.5
    assert mk.STAND_PROXY_VAR == "KRABBY_OMNI_STAND_PROXY"
    assert mk.RESTANCE_VAR == RST.RESTANCE_VAR == "KRABBY_RESTANCE"
    assert mk.RESTANCE_KNOBS == ("KRABBY_STAND_PRIOR_TARGET", "KRABBY_RESTANCE")
    assert set(mk.RESTANCE_KNOBS) <= set(mk.AGENT_KNOBS) and mk.STAND_PROXY_VAR not in mk.AGENT_KNOBS
    assert mk.STAND_PROXY_VAR in mk.EVAL_SEMANTIC_KNOBS and mk.KNOB_OFF_VALUES[mk.STAND_PROXY_VAR] == "0"
    assert mk.EVAL_CARRY_EXTERNAL == (mk.STAND_PROXY_VAR,)
    # the runner-side pair is never carried into an eval (the probe reads KRABBY_RESTANCE on its own)
    assert not set(mk.RESTANCE_KNOBS) & set(mk.EVAL_SEMANTIC_KNOBS + mk.EVAL_STRUCTURE_KNOBS
                                            + mk.PPO_CONSTRUCTION_KNOBS + mk.STATUE_KNOBS)
    # the forward proxy's knee gate is the expression the held proxy re-uses
    src = FWD_OBS_PY.read_text()
    assert "action_term.knee_rod_speed.abs()\n            / (_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)" in src


# ===================================================================== the observation term
def test_knob_off_is_bitwise_z1_with_no_buffers_and_no_log():
    plain, env_a, live_a = _term(**Z1_PARAMS)
    explicit, env_b, live_b = _term(**Z1_PARAMS, stand_proxy_hold=False)
    assert plain.stand_proxy_hold is False and plain._stand_proxy_held is None
    assert plain._stand_proxy_n_live is None and plain._stand_proxy_n_total is None
    g = torch.Generator().manual_seed(4)
    for step in range(5):
        live_a.value = live_b.value = torch.rand(N, 6, generator=g) - 0.4
        speed = torch.rand(N, 6, generator=g) * 2 * KNEE_FULL
        for env in (env_a, env_b):
            env.action_manager.get_term("joint_pos").knee_rod_speed = speed.clone()
        if step == 3:
            plain.reset(torch.tensor([0, 2]))
            explicit.reset(torch.tensor([0, 2]))
        a = _call(plain, env_a, **Z1_PARAMS)
        b = _call(explicit, env_b, **Z1_PARAMS, stand_proxy_hold=False)
        assert torch.equal(a, b)
        assert torch.equal(a[:, PROXY], live_a.value)                  # the live proxy, stand rows included
    assert env_a.extras == env_b.extras and LOG_KEY not in env_a.extras.get("log", {})   # (only the clock's log)


def test_hold_follows_the_knee_gate_on_stand_rows_only():
    term, env, live = _term(**Z1_PARAMS, stand_proxy_hold=True)
    ref, env_r, live_r = _term(**Z1_PARAMS)
    act, act_r = env.action_manager.get_term("joint_pos"), env_r.action_manager.get_term("joint_pos")
    stand = STAND_SLOTS == 0
    # step 0: every knee gate full -> held = live everywhere
    l0 = torch.rand(N, 6, generator=torch.Generator().manual_seed(1)) * 1.2 - 0.4
    live.value = live_r.value = l0
    act.knee_rod_speed = act_r.knee_rod_speed = torch.full((N, 6), KNEE_FULL * 1.5)
    obs = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    assert torch.equal(term._stand_proxy_held, l0) and torch.equal(obs[:, PROXY], l0)
    _call(ref, env_r, **Z1_PARAMS)
    # step 1: legs 0-1 parked (gate 0), legs 2-3 below full (0.999), legs 4-5 exactly full; hip gates full
    speed = torch.tensor([0.0, 0.0, 0.999 * KNEE_FULL, -0.999 * KNEE_FULL, KNEE_FULL, -KNEE_FULL]).repeat(N, 1)
    act.knee_rod_speed = act_r.knee_rod_speed = speed
    act.hip_rod_speed = act_r.hip_rod_speed = torch.full((N, 6), 10.0)   # a hip-only reading is never held
    l1 = torch.full((N, 6), -0.5)                                        # the parked rods read -0.5 live
    l1[:, 4:] = 0.7
    live.value = live_r.value = l1
    obs = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    obs_r = _call(ref, env_r, **Z1_PARAMS)
    held = torch.cat([l0[:, :4], l1[:, 4:]], dim=1)
    assert torch.equal(term._stand_proxy_held, held)                     # every row updates, stand or not
    assert torch.equal(obs[stand][:, PROXY], held[stand])                # stand rows: the held value
    assert torch.equal(obs[~stand][:, PROXY], l1[~stand])                # walking rows: live
    # nothing else moved: every other column, the returned history block and the history buffer are Z1's
    keep = torch.ones(obs.shape[1], dtype=torch.bool)
    keep[PROXY] = False
    assert torch.equal(obs[:, keep], obs_r[:, keep])
    assert torch.equal(obs[~stand], obs_r[~stand])
    assert torch.equal(term._obs_history_buffer, ref._obs_history_buffer)
    assert torch.equal(term._obs_history_buffer[:, -1, PROXY], l1)       # history = the raw live proxy
    # step 2: the history block of the returned obs carries the RAW step-1 proxy on the stand rows too
    obs2 = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    obs2_r = _call(ref, env_r, **Z1_PARAMS)
    assert torch.equal(_history(obs2)[:, -1, PROXY], l1)
    assert torch.equal(obs2[:, HIST0:], obs2_r[:, HIST0:])
    # the stand rows follow the command term's latched slot
    env.command_manager.get_term("base_velocity").slot_id = torch.tensor([1, 0, 0, 1, 1, 1])
    obs3 = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    assert torch.equal(obs3[1:3, PROXY], held[1:3]) and torch.equal(obs3[0, PROXY], l1[0])


def test_reset_restores_unknown_and_logs_the_live_share():
    term, env, live = _term(**Z1_PARAMS, stand_proxy_hold=True)
    act = env.action_manager.get_term("joint_pos")
    assert torch.equal(term._stand_proxy_held, torch.full((N, 6), -0.5))
    # a reset with nothing seen yet logs nothing (a share of nothing is not a 1.0)
    term.reset(None)
    assert LOG_KEY not in env.extras.get("log", {})
    # step A: all gates full -> 2 stand rows x 6 legs, all live
    live.value = torch.full((N, 6), 0.6)
    act.knee_rod_speed = torch.full((N, 6), 2 * KNEE_FULL)
    _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    # step B: gates closed on legs 0-2, the live value moves -> 3 of 6 legs held != live on each stand row
    live.value = torch.full((N, 6), -0.5)
    act.knee_rod_speed[:, :3] = 0.0
    obs = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    assert torch.equal(obs[0, PROXY], torch.tensor([0.6, 0.6, 0.6, -0.5, -0.5, -0.5]))
    term.reset(torch.tensor([0, 1]))
    assert env.extras["log"][LOG_KEY].item() == pytest.approx((12 + 6) / 24)
    assert int(term._stand_proxy_n_total) == 0 and int(term._stand_proxy_n_live) == 0
    # the reset rows read unknown again; the others keep their held values
    assert torch.equal(term._stand_proxy_held[0], torch.full((6,), -0.5))
    assert torch.equal(term._stand_proxy_held[1], torch.full((6,), -0.5))
    assert torch.equal(term._stand_proxy_held[4], torch.tensor([0.6, 0.6, 0.6, -0.5, -0.5, -0.5]))
    # ... so after the reset, stand row 0 (gates still closed on legs 0-2) reads -0.5 there, row 4 its 0.6
    env.extras = {}
    obs = _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    assert torch.equal(obs[0, PROXY], torch.full((6,), -0.5))
    assert torch.equal(obs[4, PROXY], torch.tensor([0.6, 0.6, 0.6, -0.5, -0.5, -0.5]))
    term.reset(None)
    assert env.extras["log"][LOG_KEY].item() == pytest.approx(9 / 12)   # row 0 all live, row 4 half
    assert torch.equal(term._stand_proxy_held, torch.full((N, 6), -0.5))
    # history rows of the reset envs are zeroed by the base reset, exactly as without the knob
    assert torch.equal(term._obs_history_buffer, torch.zeros_like(term._obs_history_buffer))


def test_refusals_and_construction_time_only():
    for params, match in (({"stand_proxy_hold": True, "clock_hide": True}, "stand_bit_clock"),
                          ({"stand_proxy_hold": True}, "stand_bit_clock"),
                          ({**Z1_PARAMS, "stand_proxy_hold": True, "lpf_obs": True}, "lpf_obs")):
        with pytest.raises(ValueError, match=match):
            _term(**params)
    term, env, _ = _term(**Z1_PARAMS)
    with pytest.raises(ValueError, match="construction-time"):
        _call(term, env, **Z1_PARAMS, stand_proxy_hold=True)
    term, env, _ = _term(**Z1_PARAMS, stand_proxy_hold=True)
    with pytest.raises(ValueError, match="construction-time"):
        _call(term, env, **Z1_PARAMS)
    import inspect
    assert inspect.signature(OmniObs.__call__).parameters["stand_proxy_hold"].default is False


# ===================================================================== the knobs (pure)
def _cfg(clock_hide=True, lpf_obs=False):
    cfg = SC._cfg(clock_hide=clock_hide)
    if lpf_obs:
        cfg.observations.policy.extreme_parkour_observations.params["lpf_obs"] = True
    return cfg


def test_parsers():
    for off in (None, "", " ", "0"):
        assert mk.parse_stand_proxy(off) is None and mk.parse_stand_prior_target(off) is None
        assert mk.parse_restance_knob(off) is None
    assert mk.parse_stand_proxy(" HOLD ") == "hold" and mk.parse_stand_prior_target("restance") == "restance"
    # (motion_knobs loads its own copy of the module: compare values, not dataclass identity)
    assert mk.parse_restance_knob("default").to_dict() == RST.RestanceParams().to_dict()
    assert mk.parse_restance_knob("lift=-0.6").lift == -0.6
    for bad in ("1", "held", "on"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_STAND_PROXY"):
            mk.parse_stand_proxy(bad)
    for bad in ("1", "zero", "R"):
        with pytest.raises(ValueError, match="KRABBY_STAND_PRIOR_TARGET"):
            mk.parse_stand_prior_target(bad)
    with pytest.raises(ValueError, match="KRABBY_RESTANCE"):
        mk.parse_restance_knob("lift=0.5")
    for var in (mk.STAND_PROXY_VAR, *mk.RESTANCE_KNOBS):
        assert var in mk.MOTION_KNOB_PARSERS
    assert mk.MOTION_KNOB_PARSERS[mk.STAND_PROXY_VAR](mk.KNOB_OFF_VALUES[mk.STAND_PROXY_VAR]) is None


def test_stand_proxy_env_knob():
    pristine = SC._dump(_cfg())
    for env in ({}, {"KRABBY_OMNI_STAND_PROXY": ""}, {"KRABBY_OMNI_STAND_PROXY": "0"},
                {"KRABBY_OMNI_STAND_PROXY": "0", "KRABBY_RESTANCE": "default"}):
        cfg = _cfg()
        mk.apply_motion_knobs(cfg, env)
        assert SC._dump(cfg) == pristine, env
    cfg = _cfg()
    mk.apply_motion_knobs(cfg, {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_PROXY": "hold"})
    want = _cfg()
    mk.apply_motion_knobs(want, {"KRABBY_OMNI_STAND_OBS": "clock"})
    want.observations.policy.extreme_parkour_observations.params["stand_proxy_hold"] = True
    assert SC._dump(cfg) == SC._dump(want)                              # Z1's cfg + exactly the one param
    for env, cfg in (({"KRABBY_OMNI_STAND_PROXY": "hold"}, _cfg()),
                     ({"KRABBY_OMNI_STAND_OBS": "1", "KRABBY_OMNI_STAND_PROXY": "hold"}, _cfg())):
        with pytest.raises(RuntimeError, match="KRABBY_OMNI_STAND_OBS=clock"):
            mk.apply_motion_knobs(cfg, env)
    with pytest.raises(RuntimeError, match="KRABBY_ACTION_LPF_OBS"):
        mk.apply_motion_knobs(_cfg(lpf_obs=True), {"KRABBY_OMNI_STAND_OBS": "clock",
                                                   "KRABBY_OMNI_STAND_PROXY": "hold"})


def _z2_env(tmp_path, **over):
    ckpt = tmp_path / "graft.pt"
    ckpt.write_bytes(b"x")
    env = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_ANCHOR_CKPT": str(ckpt),
           "KRABBY_STAND_PRIOR_W": "1.0",
           "KRABBY_STAND_PRIOR_TARGET": "restance", "KRABBY_RESTANCE": "default"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_runner_knobs_reach_the_algorithm_cfg(tmp_path, capsys):
    env = _z2_env(tmp_path)
    agent = SC._agent()
    mk.apply_exploration_knobs(agent, env)
    alg = vars(agent.algorithm)
    assert alg["stand_prior_target"] == "restance"
    assert alg["restance"] == RST.RestanceParams().to_dict()
    assert alg["stand_prior_coef"] == 1.0 and alg["stand_mask_col"] == 15
    out = capsys.readouterr().out
    assert "[KNOB] KRABBY_STAND_PRIOR_TARGET=restance consumed" in out
    assert "[KNOB] KRABBY_RESTANCE=default consumed" in out
    agent = SC._agent()
    mk.apply_exploration_knobs(agent, _z2_env(tmp_path, KRABBY_RESTANCE="lift=-0.6,up_frac=0.7"))
    assert agent.algorithm.restance == RST.RestanceParams(lift=-0.6, up_frac=0.7).to_dict()
    # NOTE(p2r Z2 cams-only, 2026-10-10): the held proxy is independent of the target -- on or off, same fields
    for proxy in ("hold", "0"):
        agent = SC._agent()
        mk.apply_exploration_knobs(agent, _z2_env(tmp_path, KRABBY_OMNI_STAND_PROXY=proxy))
        assert agent.algorithm.stand_prior_target == "restance"
        assert agent.algorithm.restance == RST.RestanceParams().to_dict()
    # Z1's env (no Z2 knob): exactly Z1's fields
    agent = SC._agent()
    mk.apply_exploration_knobs(agent, _z2_env(tmp_path, KRABBY_STAND_PRIOR_TARGET=None, KRABBY_RESTANCE=None))
    assert "stand_prior_target" not in vars(agent.algorithm) and "restance" not in vars(agent.algorithm)


def test_runner_knob_cross_rules(tmp_path):
    cases = [
        (_z2_env(tmp_path, KRABBY_STAND_PRIOR_W="0"), "KRABBY_STAND_PRIOR_W is off"),
        (_z2_env(tmp_path, KRABBY_STAND_PRIOR_W=None, KRABBY_ANCHOR_W="1.0"), "KRABBY_STAND_PRIOR_W is off"),
        (_z2_env(tmp_path, KRABBY_RESTANCE=None), "needs KRABBY_RESTANCE"),
        (_z2_env(tmp_path, KRABBY_RESTANCE="0"), "needs KRABBY_RESTANCE"),
        # KRABBY_RESTANCE without the target in a TRAINING recipe (a statue knob armed)
        (_z2_env(tmp_path, KRABBY_STAND_PRIOR_TARGET=None), "never distils"),
        (_z2_env(tmp_path, KRABBY_STAND_PRIOR_TARGET="0"), "never distils"),
        # the held proxy without the clock bit (an independent knob, still refused there)
        (_z2_env(tmp_path, KRABBY_OMNI_STAND_OBS="1", KRABBY_ANCHOR_W=None, KRABBY_ANCHOR_CKPT=None,
                 KRABBY_STAND_PRIOR_W=None, KRABBY_STAND_PRIOR_TARGET=None, KRABBY_RESTANCE=None,
                 KRABBY_OMNI_STAND_PROXY="hold"),
         "KRABBY_OMNI_STAND_OBS=clock"),
        # the statue rules still bind the target (the clock bit is the stand mask)
        (_z2_env(tmp_path, KRABBY_OMNI_STAND_OBS="0"), "KRABBY_OMNI_STAND_OBS=clock"),
        (_z2_env(tmp_path, KRABBY_RESTANCE="theta_L=0.6"), "unknown"),
        # malformed values, anywhere
        (_z2_env(tmp_path, KRABBY_RESTANCE="lift=0.2"), "lift"),
        (_z2_env(tmp_path, KRABBY_STAND_PRIOR_TARGET="zero"), "KRABBY_STAND_PRIOR_TARGET"),
    ]
    for env, match in cases:
        agent = SC._agent()
        before = SC._dump(agent)
        with pytest.raises(ValueError, match=match):
            mk.apply_exploration_knobs(agent, env)
        assert SC._dump(agent) == before, env                         # refused before anything is written


def test_an_eval_env_carrying_restance_for_the_probe_is_not_refused(capsys):
    """An eval builds its agent cfg through the same ``__post_init__``: it carries the eval-semantics knobs
    (``STAND_OBS=clock``, ``STAND_PROXY=hold``) and, for the controller-level probe, ``KRABBY_RESTANCE`` -- but
    never the statue knobs. That env writes nothing and is not refused."""
    for env in ({"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_PROXY": "hold", "KRABBY_RESTANCE": "default"},
                {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_PROXY": "hold"},
                {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "lift=-0.6"}):
        agent = SC._agent()
        before = SC._dump(agent)
        mk.apply_exploration_knobs(agent, env)
        assert SC._dump(agent) == before, env
    assert "[KNOB]" not in capsys.readouterr().out
    # ... and the env side of that eval builds the held proxy
    cfg = _cfg()
    mk.apply_motion_knobs(cfg, {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_PROXY": "hold",
                                "KRABBY_RESTANCE": "default"})
    assert cfg.observations.policy.extreme_parkour_observations.params["stand_proxy_hold"] is True
    # but a malformed probe constant is still refused at cfg build
    with pytest.raises(ValueError, match="KRABBY_RESTANCE"):
        mk.apply_exploration_knobs(SC._agent(), {"KRABBY_RESTANCE": "lift=0.3"})


def test_check_restance_composition_training_rule():
    env = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_RESTANCE": "default"}
    mk.check_restance_composition(env, training=False)
    with pytest.raises(ValueError, match="never distils"):
        mk.check_restance_composition(env, training=True)
    mk.check_restance_composition({}, training=True)


def test_runner_filter_accepts_the_new_algorithm_keys():
    cfg = {"stand_prior_coef": 1.0, "anchor_ckpt": "x", "stand_mask_col": 15, "stand_prior_target": "restance",
           "restance": RST.RestanceParams().to_dict()}
    assert PH.runner_mod.filter_ppo_alg_cfg(PH.PPO, dict(cfg)) == cfg


# ===================================================================== PPOWithExtractor
STAND_COL = SC.STAND_COL
P_DICT = RST.RestanceParams().to_dict()


def _z2_alg(tmp_path, seed=0, anchor=1.0, prior=1.0, ckpt=None, target="restance", restance=None, **kw):
    if target is not None:
        kw.update(stand_prior_target=target, restance=dict(restance or P_DICT))
    return SC._statue_alg(tmp_path, seed=seed, anchor=anchor, prior=prior, ckpt=ckpt, **kw)


def test_target_unset_is_z1_bitwise(tmp_path):
    ckpt = SC._anchor_ckpt(tmp_path, seed=5)
    z1 = SC._statue_alg(tmp_path, seed=0, ckpt=ckpt)
    off = SC._statue_alg(tmp_path, seed=0, ckpt=ckpt, stand_prior_target=None, restance=None)
    assert off.stand_prior_target is None and off.restance_params is None
    for it in (0, 1, 50):
        la, lb = SC._update(z1, iteration=it), SC._update(off, iteration=it)
        assert la == lb
        PH._assert_same(PH._snapshot(z1), PH._snapshot(off))
        assert not any(k.startswith("restance") for k in la)


def test_a_zero_target_reproduces_z1_bitwise(tmp_path, monkeypatch):
    """With R monkeypatched to 0 the re-stance path IS Z1's prior (mu - 0 is mu bitwise): the only thing the target
    changes is the target."""
    ckpt = SC._anchor_ckpt(tmp_path, seed=5)
    z1 = SC._statue_alg(tmp_path, seed=0, ckpt=ckpt)
    z2 = _z2_alg(tmp_path, seed=0, ckpt=ckpt)
    real = PH.ppo_mod.restance_module()
    fake = types.SimpleNamespace(
        restance_target=lambda obs, p: (torch.zeros(obs.shape[0], 18), torch.full((obs.shape[0], 6), 4)),
        BRANCH_NAMES=real.BRANCH_NAMES, RestanceParams=real.RestanceParams, format_restance=real.format_restance)
    monkeypatch.setattr(PH.ppo_mod, "restance_module", lambda: fake)
    la, lb = SC._update(z1, iteration=3), SC._update(z2, iteration=3)
    PH._assert_same(PH._snapshot(z1), PH._snapshot(z2))
    assert {k: v for k, v in lb.items() if not k.startswith("restance")} == la
    assert lb["restance_nonzero"] == 0.0 and lb["restance_seat"] == 1.0


def _minibatch(b=24, seed=3):
    obs, stand = SC._minibatch(None, seed=seed, b=b)
    g = torch.Generator().manual_seed(seed + 100)
    obs[:, list(RST.CAM_OBS_COLS)] = (torch.rand(b, 6, generator=g) - 0.5) * 0.8
    # NOTE(p2r Z2 cams-only, 2026-10-10): R's second input is the hip-pitch POSITION (default .. past the lift)
    obs[:, list(RST.HIP_POS_OBS_COLS)] = torch.rand(b, 6, generator=g) * (2 * RST.ROD_SCALE * RST.RestanceParams().lift)
    obs[:, list(RST.PROXY_OBS_COLS)] = torch.rand(b, 6, generator=g) * 1.5 - 0.5      # never read by R
    return obs, stand


def test_the_restance_prior_gradient_is_the_hand_formula_on_bit1_rows_only(tmp_path):
    alg = _z2_alg(tmp_path, seed=0, ckpt=SC._anchor_ckpt(tmp_path, seed=5))
    obs, stand = _minibatch()
    b = obs.shape[0]
    mu = alg.policy.actor(obs, False)
    leaf = mu.detach().clone().requires_grad_(True)
    out = alg._statue_losses(obs, leaf, b)
    target, branch = RST.restance_target(obs, RST.RestanceParams())
    assert torch.equal(out["restance_target"], target) and torch.equal(out["restance_branch"], branch)
    assert not out["restance_target"].requires_grad
    assert bool((target[stand].abs().sum(-1) > 0).all())                  # a live, non-zero target
    g_prior, = torch.autograd.grad(out["prior"], leaf, retain_graph=True)
    s2 = alg.anchor_sigma.square()
    assert torch.equal(g_prior[~stand], torch.zeros_like(g_prior[~stand]))
    assert torch.allclose(g_prior[stand], ((leaf - target) / s2 / b)[stand], atol=1e-7)
    # the value: sum over the bit-1 rows of 0.5 ((mu - R) / sigma_A)^2, divided by B
    k = 0.5 * (((leaf.detach() - target) / alg.anchor_sigma) ** 2).sum(-1)
    assert torch.allclose(out["prior"], k[stand].sum() / b, rtol=1e-6)
    # the anchor is untouched by the target
    z1 = SC._statue_alg(tmp_path, seed=0, ckpt=SC._anchor_ckpt(tmp_path, seed=5))
    assert torch.equal(alg._statue_losses(obs, leaf, b)["anchor"], z1._statue_losses(obs, leaf, b)["anchor"])
    # through the network: actor parameters only; walking rows' inputs do not move the prior or its gradient
    out = alg._statue_losses(obs, alg.policy.actor(obs, False), b)
    critic = list(alg.policy.critic.parameters())
    assert all(g is None for g in torch.autograd.grad(out["prior"], critic, retain_graph=True, allow_unused=True))
    assert torch.autograd.grad(out["prior"], [alg.policy.std_raw], retain_graph=True, allow_unused=True)[0] is None
    params = [p for p in alg.policy.actor.parameters() if p.requires_grad]
    g_p = torch.autograd.grad(out["prior"], params, allow_unused=True)
    obs2 = obs.clone()
    obs2[~stand] += 3.0
    obs2[~stand, STAND_COL] = 0.0                                          # still walking rows
    out2 = alg._statue_losses(obs2, alg.policy.actor(obs2, False), b)
    g_p2 = torch.autograd.grad(out2["prior"], params, allow_unused=True)
    assert torch.allclose(out2["prior"], out["prior"], rtol=0, atol=1e-6)
    for a, c in zip(g_p, g_p2):
        assert (a is None and c is None) or torch.allclose(a, c, atol=1e-6)


def test_r_is_read_off_the_original_rows_under_augmentation(tmp_path):
    alg = _z2_alg(tmp_path, seed=0, ckpt=SC._anchor_ckpt(tmp_path, seed=5))
    obs, stand = _minibatch(b=16)
    aug = torch.cat([obs, obs.flip(0) + 1.0])                              # 2B rows; only the first B count
    mu = alg.policy.actor(aug, False)[:16]
    out = alg._statue_losses(aug, mu, 16)
    assert torch.equal(out["bit"], stand)
    assert torch.equal(out["restance_target"], RST.restance_target(obs, RST.RestanceParams())[0])


def test_restance_scalars_are_logged_and_match_a_recompute(tmp_path, capsys):
    ckpt = SC._anchor_ckpt(tmp_path, seed=5)
    alg = _z2_alg(tmp_path, seed=0, ckpt=ckpt, num_learning_epochs=1, num_mini_batches=1)
    line = capsys.readouterr().out
    assert "[PPO] construction:" in line and "stand_prior_target=restance" in line
    assert f"restance={RST.format_restance(RST.RestanceParams())}" in line
    gen = torch.Generator().manual_seed(11)
    torch.manual_seed(11)
    SC._statue_rollout(alg, gen)
    obs = alg.storage.observations["policy"].reshape(-1, PH.OBS_W).clone()
    alg.iteration = 1
    torch.manual_seed(1234)
    ld = alg.update()
    bit = obs[:, STAND_COL] > 0.5
    target, branch = RST.restance_target(obs, RST.RestanceParams())
    assert ld["restance_nonzero"] == pytest.approx(float((target[bit].abs() > 1e-6).any(-1).float().mean()))
    shares = RST.branch_shares(branch[bit])
    for name in RST.BRANCH_NAMES:
        assert ld[f"restance_{name}"] == pytest.approx(shares[name]), name
    assert sum(ld[f"restance_{n}"] for n in RST.BRANCH_NAMES) == pytest.approx(1.0)
    assert math.isfinite(ld["stand_prior"]) and ld["stand_prior"] > 0.0
    assert ld["prior_rows"] == pytest.approx(float(SC.ROLL_BITS.mean()))
    # the default (multi-minibatch) update logs them too, and they stay finite
    alg2 = _z2_alg(tmp_path, seed=0, ckpt=ckpt)
    ld2 = SC._update(alg2, iteration=0)
    assert {f"restance_{n}" for n in RST.BRANCH_NAMES} | {"restance_nonzero"} <= set(ld2)
    assert "grad_norm_statue" in ld2


def test_ppo_construction_refusals(tmp_path):
    ckpt = SC._anchor_ckpt(tmp_path)
    base = {"anchor_ckpt": str(ckpt), "stand_mask_col": STAND_COL, **SC._soft_kw()}
    cases = [
        ({"anchor_coef": 1.0, "stand_prior_target": "restance", "restance": P_DICT}, "stand_prior_coef > 0"),
        ({"stand_prior_coef": 1.0, "stand_prior_target": "zero", "restance": P_DICT}, "stand_prior_target must"),
        ({"stand_prior_coef": 1.0, "stand_prior_target": "restance"}, "needs restance"),
        ({"stand_prior_coef": 1.0, "restance": P_DICT}, "stand_prior_target is not"),
        ({"stand_prior_coef": 1.0, "stand_prior_target": "restance", "restance": {**P_DICT, "gain": 1.0}},
         "unknown"),
        ({"stand_prior_coef": 1.0, "stand_prior_target": "restance", "restance": {**P_DICT, "lift": 0.4}}, "lift"),
    ]
    for kw, match in cases:
        with pytest.raises(ValueError, match=match):
            PH._alg(0, **base, **kw)
    with pytest.raises(ValueError, match="stand_prior_target is not"):
        PH._alg(0, **SC._soft_kw(), restance=P_DICT)


# ===================================================================== the ladder
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__p2r_z2_test")


#: NOTE(p2r Z2 cams-only, 2026-10-10): the v14 Z2 chain's two knobs (no proxy); the held proxy stays a registered,
#: independent ladder knob (PROXY_KNOB) the Z2 chain does not carry.
Z2_KNOBS = {"KRABBY_STAND_PRIOR_TARGET": "restance",  # v15: the final probe's constants
            "KRABBY_RESTANCE": "lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1"}
PROXY_KNOB = {"KRABBY_OMNI_STAND_PROXY": "hold"}
LADDER_KNOBS = {**PROXY_KNOB, **Z2_KNOBS}


def test_z2_knobs_are_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in LADDER_KNOBS:
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
        assert parsers[var] is rc.knobs()._mk.MOTION_KNOB_PARSERS[var], var
    assert set(mk.RESTANCE_KNOBS) <= rc.AGENT_DUMP_KEYS and "KRABBY_OMNI_STAND_PROXY" not in rc.AGENT_DUMP_KEYS
    assert {"stand_prior_target", "restance"} <= set(rc.STATUE_ALG_FIELDS)
    reasons = []
    for var, value in LADDER_KNOBS.items():
        rc._check_knob(var, value, "t", whitelist, parsers, reasons, set())
    assert not reasons, reasons
    for var, value in (("KRABBY_OMNI_STAND_PROXY", "on"), ("KRABBY_STAND_PRIOR_TARGET", "zero"),
                       ("KRABBY_RESTANCE", "lift=0.4"), ("KRABBY_RESTANCE", "gain=1"),
                       ("KRABBY_RESTANCE", "kappa=0.25")):                    # a removed load-sensing constant
        rc._check_knob(var, value, "t", whitelist, parsers, reasons, set())
    assert len(reasons) == 5, reasons


ENV_OFF = "observations:\n  policy:\n    extreme_parkour_observations:\n      params:\n        clock_hide: true\n"
ENV_Z1 = ENV_OFF + "        stand_bit_clock: true\n"
ENV_Z2 = ENV_Z1 + "        stand_proxy_hold: true\n"


def _agent_yaml(**alg) -> str:
    base = {"seed": 18, "algorithm": {"class_name": "PPOWithExtractor", "learning_rate": 0.0003,
                                      "stand_prior_coef": 1.0, "stand_mask_col": 15, **alg},
            "policy": {"init_noise_std": 0.8}}
    return yaml.dump(base, default_flow_style=False, sort_keys=False)       # isaaclab's dump_yaml spelling


def test_dump_checks_on_synthetic_dumps(rc):
    checks = rc.OVERLAY_DUMP_CHECKS
    proxy, target, rst = (checks[k] for k in ("KRABBY_OMNI_STAND_PROXY", "KRABBY_STAND_PRIOR_TARGET",
                                              "KRABBY_RESTANCE"))
    assert proxy(ENV_Z2, "hold") and not proxy(ENV_Z2, "0") and not proxy(ENV_Z2, "")
    assert proxy(ENV_Z1, "0") and proxy(ENV_OFF, "") and not proxy(ENV_Z1, "hold")
    assert not proxy(ENV_OFF + "        stand_proxy_hold: true\n", "hold")    # without the clock bit
    on = _agent_yaml(stand_prior_target="restance", restance=RST.RestanceParams().to_dict())
    assert "  restance:\n    lift: -0.5\n" in on                                # the format the reader expects
    off = _agent_yaml()
    assert target(on, "restance") and not target(on, "0") and target(off, "0") and not target(off, "restance")
    assert rst(on, "default") and not rst(on, "0") and rst(off, "") and not rst(off, "default")
    assert not rst(on, "lift=-0.6")                                         # a different constant
    tuned = RST.parse_restance("lift=-0.6,up_frac=0.7,down_frac=0.3")
    on2 = _agent_yaml(stand_prior_target="restance", restance=tuned.to_dict())
    assert rst(on2, RST.format_restance(tuned)) and rst(on2, "lift=-0.6, up_frac=0.7, down_frac=0.3")
    # a dump written with the load-sensing R's constants (v13) no longer matches any armed value
    old = dict(RST.RestanceParams().to_dict(), theta_L=0.35, theta_S=0.75, kappa=0.5, dmax=0.6)
    assert not rst(_agent_yaml(stand_prior_target="restance", restance=old), "default")
    missing = dict(tuned.to_dict())
    del missing["eps_air"]
    assert not rst(_agent_yaml(stand_prior_target="restance", restance=missing), RST.format_restance(tuned))
    assert not target(_agent_yaml(stand_prior_target="restance"), "restance")   # the constants must travel along
    assert rst(_agent_yaml(restance=None), "0")


def test_dump_checks_on_the_real_s1_dump(rc, tmp_path):
    env_txt, agent_txt = SC._s1_dumps()
    checks = rc.OVERLAY_DUMP_CHECKS
    for key, value in (("KRABBY_OMNI_STAND_PROXY", "0"), ("KRABBY_STAND_PRIOR_TARGET", ""), ("KRABBY_RESTANCE", "")):
        txt = agent_txt if key in rc.AGENT_DUMP_KEYS else env_txt
        assert checks[key](txt, value), key
        assert not checks[key](txt, LADDER_KNOBS[key]), key
    # the S1 dump as the trainer would write it with Z1 + the Z2 knobs on (text edits of the real files; v14:
    # no held proxy -- the env dump is Z1's)
    env_on, agent_on = SC._armed_dumps(env_txt, agent_txt, str(SC.GRAFT))
    assert checks["KRABBY_OMNI_STAND_PROXY"](env_on, "0") and not checks["KRABBY_OMNI_STAND_PROXY"](env_on, "hold")
    block = yaml.dump({"algorithm": {"stand_prior_target": "restance",
                                     "restance": RST.parse_restance(Z2_KNOBS["KRABBY_RESTANCE"]).to_dict()}},
                      default_flow_style=False, sort_keys=False).split("\n", 1)[1]
    agent_on = agent_on.replace("  stand_mask_col: 15\n", "  stand_mask_col: 15\n" + block, 1)
    for key, value in Z2_KNOBS.items():
        txt = agent_on if key in rc.AGENT_DUMP_KEYS else env_on
        assert checks[key](txt, value), key
    armed = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
             "KRABBY_ANCHOR_CKPT": str(SC.GRAFT), **Z2_KNOBS}
    log = "".join(f"[KNOB] {k}={v} consumed\n" for k, v in armed.items() if k != "KRABBY_OMNI_STAND_OBS")
    log += ("[PPO] construction: anchor_coef=1.0, stand_prior_coef=1.0, stand_prior_target=restance, "
            "restance=lift=-0.5\n   Mean anchor: 0.0000\n   Mean stand_prior: 2.3000\n"
            "   Mean restance_nonzero: 1.0000\n")
    rec = SC._run_dir(tmp_path / "on", env_on, agent_on, log)
    got = rc.overlay_checks(rec, armed)
    assert got and all(ok for _n, ok in got), [n for n, ok in got if not ok]
    names = dict(got)
    assert "log: [KNOB] KRABBY_OMNI_STAND_PROXY consumed" not in names              # not armed in v14
    for want in ("log: [KNOB] KRABBY_STAND_PRIOR_TARGET consumed",
                 "log: [KNOB] KRABBY_RESTANCE consumed", "log: [PPO] construction names stand_prior_target=restance",
                 "log: Mean restance_nonzero printed (the KRABBY_STAND_PRIOR_TARGET loss ran)"):
        assert names.get(want) is True, want
    # each log line is load-bearing
    for drop, name in (("   Mean restance_nonzero: 1.0000\n",
                        "log: Mean restance_nonzero printed (the KRABBY_STAND_PRIOR_TARGET loss ran)"),
                       ("stand_prior_target=restance, ", "log: [PPO] construction names stand_prior_target=restance"),
                       (f"[KNOB] KRABBY_RESTANCE={Z2_KNOBS['KRABBY_RESTANCE']} consumed\n",
                        "log: [KNOB] KRABBY_RESTANCE consumed")):
        rec = SC._run_dir(tmp_path / f"drop{abs(hash(drop))}", env_on, agent_on, log.replace(drop, ""))
        assert dict(rc.overlay_checks(rec, armed))[name] is False, name
    # Z1's dumps fail the Z2 claims
    env_z1, agent_z1 = SC._armed_dumps(env_txt, agent_txt, str(SC.GRAFT))
    got = dict(rc.overlay_checks(SC._run_dir(tmp_path / "z1", env_z1, agent_z1, log), armed))
    assert not any(v for k, v in got.items() if k.startswith("overlay KRABBY_STAND_PRIOR_TARGET")
                   or k.startswith("overlay KRABBY_RESTANCE"))


def test_composition_check(rc):
    stage = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_STAND_PRIOR_W": "1.0", **Z2_KNOBS}
    reasons = []
    rc._check_restance_composition(stage, "w", reasons)
    rc._check_restance_composition({}, "w", reasons)
    assert not reasons
    # NOTE(p2r Z2 cams-only, 2026-10-10): the held proxy is neither required nor refused beside the target
    rc._check_restance_composition({**stage, **PROXY_KNOB}, "w", reasons)
    assert not reasons
    for drop, match in (("KRABBY_STAND_PRIOR_TARGET", "never distils"), ("KRABBY_RESTANCE", "needs KRABBY_RESTANCE"),
                        ("KRABBY_STAND_PRIOR_W", "KRABBY_STAND_PRIOR_W is off"),
                        ("KRABBY_OMNI_STAND_OBS", "KRABBY_OMNI_STAND_OBS=clock")):
        reasons = []
        rc._check_restance_composition({k: v for k, v in stage.items() if k != drop}, "w", reasons)
        assert len(reasons) == 1 and match in reasons[0], (drop, reasons)
    reasons = []                                         # the proxy without the clock bit is still refused
    rc._check_restance_composition({**PROXY_KNOB, "KRABBY_OMNI_STAND_OBS": "0"}, "w", reasons)
    assert len(reasons) == 1 and "KRABBY_OMNI_STAND_OBS=clock" in reasons[0], reasons
    reasons = []                                         # a malformed value is _check_knob's to report, once
    rc._check_restance_composition({**stage, "KRABBY_RESTANCE": "lift=0.3"}, "w", reasons)
    assert reasons == []


# ===================================================================== candidates v14
def _launch_view(cands: dict) -> dict:
    """The file minus every placeholder-carrying chain and its references (run_arm.sh's launch view)."""
    view = copy.deepcopy(cands)
    bad = [n for n, ch in view["chains"].items() if "TBD_" in json.dumps(ch)]
    for n in bad:
        del view["chains"][n]
    view["references"] = {k: v for k, v in view["references"].items() if v["chain"] not in bad}
    return view


def test_candidates_v14_z2_chain_is_z1_verbatim_plus_the_two_knobs():
    """NOTE(p2r Z2 cams-only, 2026-10-10): v14 = v13 with P2__Z2_s18's KRABBY_OMNI_STAND_PROXY removed (the chain is
    the P2__Z1_s18 chain verbatim + KRABBY_STAND_PRIOR_TARGET + KRABBY_RESTANCE); every other chain is v13's."""
    d = yaml.safe_load(open(CAMP / "candidates.yaml"))
    v12 = yaml.safe_load(open(CAMP / "scripts/candidates_v12.yaml"))
    v13 = yaml.safe_load(open(CAMP / "scripts/candidates_v13.yaml"))
    assert (CAMP / "candidates.yaml").read_bytes() == (CAMP / "scripts/candidates_v16.yaml").read_bytes()
    assert d["chains"]["P2__Z1_s18"] == v12["chains"]["P2__Z1_s18"]           # v12's Z1 untouched
    assert d["references"]["P2__Z2_s18"] == {"chain": "P2__Z2_s18", "seed": 18}
    z1, z2 = d["chains"]["P2__Z1_s18"], d["chains"]["P2__Z2_s18"]
    assert len(z2["stages"]) == len(z1["stages"]) == 1
    s1, s2 = z1["stages"][0], z2["stages"][0]
    assert {k: v for k, v in s2.items() if k != "overlay"} == {k: v for k, v in s1.items() if k != "overlay"}
    ov = dict(s2["overlay"])
    assert "KRABBY_OMNI_STAND_PROXY" not in ov                                # the cams-only chain runs it off
    for k, v in Z2_KNOBS.items():
        assert ov.pop(k) == v, k
    assert ov == s1["overlay"] and list(ov) == list(s1["overlay"])           # verbatim, in key order
    assert list(s2["overlay"])[-2:] == list(Z2_KNOBS)                         # the new block last
    assert {k: v for k, v in z2.items() if k not in ("stages", "notes")} == \
        {k: v for k, v in z1.items() if k not in ("stages", "notes")}
    # v13's Z2 differed exactly by the proxy knob -- and v15's by the final R constants ('default' -> the probe's)
    v13_ov = dict(v13["chains"]["P2__Z2_s18"]["stages"][0]["overlay"])
    assert v13_ov.pop("KRABBY_OMNI_STAND_PROXY") == "hold" and v13_ov.pop("KRABBY_RESTANCE") == "default"
    s2_ov = dict(s2["overlay"])
    assert s2_ov.pop("KRABBY_RESTANCE") == Z2_KNOBS["KRABBY_RESTANCE"] and v13_ov == s2_ov
    # everything else in the file is v13's (and so v12's)
    rest = {k: v for k, v in d.items() if k not in ("chains", "references")}
    assert rest == {k: v for k, v in v13.items() if k not in ("chains", "references")}
    new = ("P2__Z2_s18", "P2__Z2c_s18")                                       # + v16's Z2c chain (P2R_Z2C_PLAN.md)
    assert {k: v for k, v in d["chains"].items() if k not in new} == \
        {k: v for k, v in v13["chains"].items() if k not in new} == v12["chains"]
    assert {k: v for k, v in d["references"].items() if k != "P2__Z2c_s18"} == v13["references"]


def test_candidates_v14_validator_dry_run(rc):
    d = yaml.safe_load(open(CAMP / "candidates.yaml"))
    view = _launch_view(d)
    assert "P2__Z2_s18" in view["chains"] and "P2__Z2_s18" in view["references"]
    rc.validate_candidates(view)                                              # raises SystemExit on any reason
    for drop, match in (("KRABBY_STAND_PRIOR_TARGET", "never distils"), ("KRABBY_RESTANCE", "needs KRABBY_RESTANCE")):
        bad = copy.deepcopy(view)
        del bad["chains"]["P2__Z2_s18"]["stages"][0]["overlay"][drop]
        with pytest.raises(SystemExit, match=match):
            rc.validate_candidates(bad)
    bad = copy.deepcopy(view)
    bad["chains"]["P2__Z2_s18"]["stages"][0]["overlay"]["KRABBY_RESTANCE"] = "theta_L=0.6"   # a removed constant
    with pytest.raises(SystemExit, match="KRABBY_RESTANCE"):
        rc.validate_candidates(bad)


def test_statue_log_checks_take_a_spaced_restance_value(rc, tmp_path):
    """``KRABBY_RESTANCE`` is a ``k=v`` list; a spelling with spaces still matches its consumed line."""
    env = {"KRABBY_STAND_PRIOR_W": "1.0", "KRABBY_OMNI_STAND_PROXY": "hold", "KRABBY_STAND_PRIOR_TARGET": "restance",
           "KRABBY_RESTANCE": "lift=-0.6, rho=0.4"}
    log = ("[KNOB] KRABBY_STAND_PRIOR_W=1.0 consumed\n[KNOB] KRABBY_OMNI_STAND_PROXY=hold consumed\n"
           "[KNOB] KRABBY_STAND_PRIOR_TARGET=restance consumed\n[KNOB] KRABBY_RESTANCE=lift=-0.6, rho=0.4 consumed\n"
           "[PPO] construction: stand_prior_coef=1.0, stand_prior_target=restance\n"
           "   Mean stand_prior: 1.0\n   Mean restance_nonzero: 1.0\n")
    (tmp_path / "train.log").write_text(log)
    got = dict(rc.statue_log_checks({"log": str(tmp_path / "train.log")}, env))
    assert got and all(got.values()), got
    (tmp_path / "train.log").write_text(log.replace("[KNOB] KRABBY_RESTANCE=lift=-0.6, rho=0.4 consumed\n", ""))
    assert dict(rc.statue_log_checks({"log": str(tmp_path / "train.log")}, env))[
        "log: [KNOB] KRABBY_RESTANCE consumed"] is False
