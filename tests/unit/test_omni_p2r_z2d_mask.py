"""Omni phase-2 stage p2r Z2d, the stand-row input mask (``P2R_Z2D_PLAN.md`` step 1; CHARTER 12), on CPU.

Pinned here (no Isaac, no GPU):

* the shared pure function ``crab_hex_omni_restance.stand_mask_obs``: only the given rows change, and on them exactly
  the last-action (65-82 -> 0) and proxy (83-88 -> -0.5) blocks of the current step and of the 10 REAL history frames
  (``259 + 89 j``; a narrower row, e.g. the 257-wide training-path slice, gets the current block only); every other
  column bitwise unchanged; ``rows=None`` masks every row; in place vs copy; dtype / device kept; idempotent; R's
  targets unchanged; mirror-invariant under both planes of the shipped omni mirror;
* the observation term under ``stand_mask`` (``KRABBY_OMNI_STAND_MASK=1``), on the REAL omni class over the REAL
  forward ``__call__`` (the harness of ``test_omni_p2r_z2_core.py``): knob off -> bitwise Z1's observation, no counters,
  no log; on -> the RETURNED observation equals ``stand_mask_obs(Z1's observation, stand rows)`` and the stored
  history buffer is Z1's; ``Metrics/stand_mask_frac``; construction refusals and construction-time only;
* the knob (``KRABBY_OMNI_STAND_MASK``, env side, an eval-semantics knob carried outside ``run_phases``): parser, the
  obs param, cross rules (clock bit, held proxy, lpf_obs) at both doors, and ``algorithm.bc_stand_mask`` beside BC;
* ``PPOWithExtractor(bc_stand_mask=True)``: the stored bank rows carry the mask (applied on the FULL-width rows, before
  the trim), ``bc_target`` is unchanged, refusals, the ``[PPO] construction`` line, the runner's alg-cfg filter;
* the ladder: registration, the dump check, the dry-run composition rule and the two log audits.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")
yaml = pytest.importorskip("yaml")
try:  # the training stack these modules import (rsl-rl-lib 5.x); CI pins 3.1.2 for compute/parkour, which has neither
    import rsl_rl.extensions  # noqa: E402,F401
    from rsl_rl.modules import EmpiricalNormalization  # noqa: E402,F401
except ImportError:
    pytest.skip("needs the Isaac training stack (rsl-rl-lib 5.x: rsl_rl.extensions, rsl_rl.modules.EmpiricalNormalization);"
                " CI pins rsl-rl-lib 3.1.2 for compute/parkour", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
RUN_PHASES_PY = OMNI / "experiments/tools/run_phases.py"


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


# The Z2 core module carries the obs-term harness (stub forward base with the real base reset); the Z2c BC module the
# tiny real PPO + the bank helpers. Reused, not copied.
Z2 = _load(UNIT / "test_omni_p2r_z2_core.py", "p2r_z2d__z2_core")
BC = _load(UNIT / "test_omni_p2r_z2c_bc.py", "p2r_z2d__z2c_bc")
SC, OH = Z2.SC, Z2.OH
PH = BC.PH
mk = BC.mk
RST = _load_unregistered(RESTANCE_PY, "_t_p2r_z2d_restance")

LAST_ACT = slice(65, 83)
PROXY = slice(83, 89)
MASKED = slice(65, 89)
W = 1149
FRAMES = [259 + 89 * j for j in range(10)]
LOG_KEY = "Metrics/stand_mask_frac"
Z1_PARAMS = Z2.Z1_PARAMS
N = Z2.N
STAND = Z2.STAND_SLOTS == 0                                          # rows 0 and 4 stand

# ---- the omni class re-executed with the restance module bound (the knob-on path reads it; knob off never does)
_NS = {k: v for k, v in Z2._NS.items() if k != "CrabHexOmniParkourObservations"}
_NS["restance"] = RST
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


@pytest.fixture(autouse=True)
def _no_heading(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)


def _masked_cols(width: int) -> list[int]:
    starts = [0] + (FRAMES if width >= W else [])
    return [s + c for s in starts for c in range(65, 89)]


# ===================================================================== the shared function
def test_constants_and_layout():
    assert RST.STAND_MASK_LAST_ACT_COLS == (65, 83) and RST.STAND_MASK_PROXY_COLS == (83, 89)
    assert RST.STAND_MASK_PROXY_COLS[0] == RST.PROXY_OBS_COLS[0] and RST.STAND_MASK_PROXY_COLS[1] == RST.MIN_OBS_WIDTH
    assert RST.STAND_MASK_PROXY_VALUE == -0.5 and RST.STAND_MASK_LAST_ACT_VALUE == 0.0
    assert RST.OBS_BUF_WIDTH == 89 == OH.N_PROP and RST.HISTORY_FRAMES == 10 == OH.HISTORY
    assert RST.HISTORY_OFFSET == 89 + 108 + 9 + 53 == OH.OBS_DIM - OH.HISTORY * OH.N_PROP
    assert RST.HISTORY_OFFSET + RST.HISTORY_FRAMES * RST.OBS_BUF_WIDTH == W == OH.OBS_DIM
    assert OmniObs.STAND_MASK_FRAC_KEY == LOG_KEY
    # R's columns lie outside the masked blocks (so a masked bank keeps its targets)
    r_cols = set(RST.CAM_OBS_COLS + RST.HIP_POS_OBS_COLS) | {RST.ROLL_OBS_COL, RST.PITCH_OBS_COL}
    assert not r_cols & set(_masked_cols(W)) and 15 not in _masked_cols(W)
    cols, values = RST.stand_mask_columns(W)
    assert cols == _masked_cols(W) and len(values) == 11 * 24
    assert values[:24] == [0.0] * 18 + [-0.5] * 6 and values == values[:24] * 11
    assert RST.stand_mask_columns(257)[0] == list(range(65, 89)) == RST.stand_mask_columns(89)[0]
    assert RST.stand_mask_columns(1148)[0] == list(range(65, 89))


@pytest.mark.parametrize("width", [W, 257, 89, 1148, 1160])
def test_only_the_given_rows_and_the_masked_columns_change(width):
    g = torch.Generator().manual_seed(width)
    obs = torch.randn(7, width, generator=g)
    rows = torch.tensor([True, False, False, True, False, True, False])
    ref = obs.clone()
    out = RST.stand_mask_obs(obs, rows)
    assert torch.equal(obs, ref)                                       # a copy: the input is untouched
    assert out.dtype == obs.dtype and out.device == obs.device and out.shape == obs.shape
    cols = _masked_cols(width)
    keep = torch.ones(width, dtype=torch.bool)
    keep[cols] = False
    assert torch.equal(out[~rows], obs[~rows])                         # other rows bitwise
    assert torch.equal(out[:, keep], obs[:, keep])                     # other columns bitwise
    starts = [0] + (FRAMES if width >= W else [])
    for s in starts:
        assert torch.equal(out[rows][:, s + 65:s + 83], torch.zeros(3, 18))
        assert torch.equal(out[rows][:, s + 83:s + 89], torch.full((3, 6), -0.5))
    # exact constants: no -0.0 (an input of -x times 0 would carry one)
    assert not torch.signbit(out[rows][:, [s + c for s in starts for c in range(65, 83)]]).any()
    if width < W:
        assert torch.equal(out[:, 89:], obs[:, 89:])                   # no history block below full width


def test_rows_none_masks_every_row_and_inplace_writes_the_input():
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(5, W, generator=g)
    every = RST.stand_mask_obs(obs, None)
    assert torch.equal(every, RST.stand_mask_obs(obs, torch.ones(5, dtype=torch.bool)))
    assert torch.equal(RST.stand_mask_obs(obs, torch.zeros(5, dtype=torch.bool)), obs)
    buf = obs.clone()
    got = RST.stand_mask_obs(buf, None, inplace=True)
    assert got is buf and torch.equal(buf, every)
    rows = torch.tensor([False, True, False, False, True])
    buf = obs.clone()
    got = RST.stand_mask_obs(buf, rows, inplace=True)
    assert got is buf and torch.equal(buf, RST.stand_mask_obs(obs, rows))
    # idempotent: twice == once
    once = RST.stand_mask_obs(obs, rows)
    assert torch.equal(RST.stand_mask_obs(once, rows), once)
    # dtype kept (float64)
    d = RST.stand_mask_obs(obs.double(), rows)
    assert d.dtype == torch.float64 and torch.equal(d.float(), once)


def test_refusals():
    with pytest.raises(ValueError, match=">= 89"):
        RST.stand_mask_obs(torch.zeros(2, 88))
    with pytest.raises(ValueError, match="obs must be"):
        RST.stand_mask_obs(torch.zeros(W))
    for rows in (torch.tensor([1, 0]), torch.tensor([True, False, True]), torch.ones(2, 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match="bool tensor"):
            RST.stand_mask_obs(torch.zeros(2, W), rows)


def test_r_targets_unchanged_by_the_mask():
    obs = torch.from_numpy(BC._bank_obs(200, seed=9, width=W))
    for p in (RST.RestanceParams(), BC.P_Z2C):
        t0, b0 = RST.restance_target(obs, p)
        t1, b1 = RST.restance_target(RST.stand_mask_obs(obs), p)
        assert torch.equal(t0, t1) and torch.equal(b0, b1)


def test_mirror_invariance_under_both_planes():
    lr, fa = SC.MH.omni.build_mirrors(SC._ClockEnv(stand_bit_clock=True), SC.MH.OBS_DIM)
    assert SC.MH.OBS_DIM == W
    g = torch.Generator().manual_seed(11)
    x = torch.randn(9, W, generator=g)
    rows = torch.tensor([True, False, True, True, False, False, True, False, True])
    for m in (lr, fa):
        a = m.mirror_obs(RST.stand_mask_obs(x, rows))
        b = RST.stand_mask_obs(m.mirror_obs(x), rows)
        assert torch.equal(a, b)
        # the masked columns map onto masked columns (the blocks are closed under each plane's permutation)
        cols = set(_masked_cols(W))
        assert {int(m.obs_perm[c]) for c in cols} == cols


# ===================================================================== the observation term
def _term(**params):
    env = OH._Env(N, torch.zeros(N, 4), torch.zeros(N))
    env.extras = {}
    term = OmniObs(OH._ParamCfg(**params), env)
    live = Z2._Live()
    term._get_contact_fill = live
    act = env.action_manager.get_term("joint_pos")
    act.knee_rod_speed = torch.zeros(N, 6)
    act.hip_rod_speed = torch.zeros(N, 6)
    env.command_manager.get_term("base_velocity").slot_id = Z2.STAND_SLOTS.clone()
    return term, env, live


def _call(term, env, **params):
    return term(env, None, None, "base_parkour", OH.HISTORY, **params)


def _drive(pairs, steps: int, seed: int = 4):
    """Step every ``(term, env, live, params)`` in lockstep on the same random last action + proxy; yield the obs."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        last = torch.randn(N, 18, generator=g)
        proxy = torch.rand(N, 6, generator=g) - 0.4
        outs = []
        for term, env, live, params in pairs:
            env.action_manager.get_term("joint_pos").action_history_buf[:, -1] = last
            live.value = proxy.clone()
            outs.append(_call(term, env, **params))
        yield last, proxy, outs


def test_the_stub_layout_puts_last_action_and_proxy_at_the_masked_columns():
    term, env, live = _term(**Z1_PARAMS)
    for last, proxy, (obs,) in _drive([(term, env, live, Z1_PARAMS)], 2):
        assert torch.equal(obs[:, LAST_ACT], last) and torch.equal(obs[:, PROXY], proxy)
    hist = obs[:, 259:].reshape(N, 10, 89)
    assert torch.equal(hist[:, -1, LAST_ACT], term._obs_history_buffer[:, -2, LAST_ACT])


def test_knob_off_is_bitwise_z1_with_no_counters_and_no_log():
    plain = _term(**Z1_PARAMS)
    explicit = _term(**Z1_PARAMS, stand_mask=False)
    assert plain[0].stand_mask is False and plain[0]._stand_mask_n_masked is None
    assert plain[0]._stand_mask_n_total is None
    for step, (_l, _p, (a, b)) in enumerate(_drive([(*plain, Z1_PARAMS),
                                                    (*explicit, {**Z1_PARAMS, "stand_mask": False})], 5)):
        assert torch.equal(a, b)
        if step == 3:
            plain[0].reset(torch.tensor([0, 2]))
            explicit[0].reset(torch.tensor([0, 2]))
    assert plain[1].extras == explicit[1].extras and LOG_KEY not in plain[1].extras.get("log", {})
    assert inspect.signature(OmniObs.__call__).parameters["stand_mask"].default is False


def test_mask_on_is_the_shared_function_on_z1s_observation():
    on_p = {**Z1_PARAMS, "stand_mask": True}
    on, ref = _term(**on_p), _term(**Z1_PARAMS)
    for _l, _p, (a, r) in _drive([(*on, on_p), (*ref, Z1_PARAMS)], 6):
        assert torch.equal(a, RST.stand_mask_obs(r, STAND))
        assert torch.equal(a[~STAND], r[~STAND])                      # walking rows bitwise Z1's
        assert not torch.equal(a[STAND][:, MASKED], r[STAND][:, MASKED])   # live: the raw rows differ
        for s in [0] + FRAMES:
            assert torch.equal(a[STAND][:, s + 65:s + 83], torch.zeros(2, 18))
            assert torch.equal(a[STAND][:, s + 83:s + 89], torch.full((2, 6), -0.5))
        assert torch.equal(a[:, 15], STAND.float())                   # the stand bit itself is kept
        # the stored history is never written: Z1's raw rows
        assert torch.equal(on[0]._obs_history_buffer, ref[0]._obs_history_buffer)
    # the stand rows follow the command term's latched slot
    for t in (on, ref):
        t[1].command_manager.get_term("base_velocity").slot_id = torch.tensor([1, 0, 0, 1, 1, 1])
    rows = torch.tensor([False, True, True, False, False, False])
    for _l, _p, (a, r) in _drive([(*on, on_p), (*ref, Z1_PARAMS)], 1, seed=8):
        assert torch.equal(a, RST.stand_mask_obs(r, rows))


def test_mask_frac_log():
    on_p = {**Z1_PARAMS, "stand_mask": True}
    term, env, live = _term(**on_p)
    term.reset(None)                                                   # nothing seen yet: nothing logged
    assert LOG_KEY not in env.extras.get("log", {})
    list(_drive([(term, env, live, on_p)], 3))                         # 3 steps x 6 rows, 2 stand rows each
    term.reset(torch.tensor([1]))
    assert env.extras["log"][LOG_KEY].item() == pytest.approx(6 / 18)
    assert int(term._stand_mask_n_masked) == 0 and int(term._stand_mask_n_total) == 0
    env.extras = {}
    term.reset(None)                                                   # nothing since the last log: no key
    assert LOG_KEY not in env.extras.get("log", {})
    env.command_manager.get_term("base_velocity").slot_id = torch.zeros(N, dtype=torch.long)
    list(_drive([(term, env, live, on_p)], 2))
    term.reset(None)
    assert env.extras["log"][LOG_KEY].item() == pytest.approx(1.0)


def test_obs_refusals_and_construction_time_only():
    for params, match in (({"stand_mask": True, "clock_hide": True}, "stand_bit_clock"),
                          ({"stand_mask": True}, "stand_bit_clock"),
                          ({**Z1_PARAMS, "stand_mask": True, "stand_proxy_hold": True}, "stand_proxy_hold"),
                          ({**Z1_PARAMS, "stand_mask": True, "lpf_obs": True}, "lpf_obs")):
        with pytest.raises(ValueError, match=match):
            _term(**params)
    term, env, _ = _term(**Z1_PARAMS)
    with pytest.raises(ValueError, match="construction-time"):
        _call(term, env, **Z1_PARAMS, stand_mask=True)
    term, env, _ = _term(**Z1_PARAMS, stand_mask=True)
    with pytest.raises(ValueError, match="construction-time"):
        _call(term, env, **Z1_PARAMS)


def test_the_shipped_module_imports_the_restance_module_by_package():
    src = OMNI_OBS_PY.read_text()
    assert "from parkour_tasks.crab_hex_omni_task.mdp import crab_hex_omni_restance as restance" in src
    assert "restance.stand_mask_obs(observations, rows, inplace=True)" in src
    # the restance module stays loadable by path (no ``from __future__ import annotations``: dataclasses)
    assert not re.search(r"^from __future__ import annotations", RESTANCE_PY.read_text(), re.M)


# ===================================================================== the knob
def test_knob_contract_and_parser():
    assert mk.STAND_MASK_VAR == "KRABBY_OMNI_STAND_MASK"
    assert mk.MOTION_KNOB_PARSERS[mk.STAND_MASK_VAR] is mk.parse_stand_mask
    assert "STAND_MASK_VAR" in mk.__all__ and "parse_stand_mask" in mk.__all__
    assert mk.STAND_MASK_VAR in mk.EVAL_SEMANTIC_KNOBS and mk.STAND_MASK_VAR not in mk.AGENT_KNOBS
    assert mk.EVAL_CARRY_EXTERNAL == (mk.STAND_PROXY_VAR, mk.STAND_MASK_VAR)
    assert mk.KNOB_OFF_VALUES[mk.STAND_MASK_VAR] == "0"
    assert "KRABBY_OMNI_STAND_MASK" not in RUN_PHASES_PY.read_text()  # outside the eval-locked EVAL_CARRY
    for off in (None, "", " ", "0", " 0 "):
        assert mk.parse_stand_mask(off) is None
    assert mk.parse_stand_mask("1") is True and mk.parse_stand_mask(" 1 ") is True
    for bad in ("2", "true", "on", "hold", "yes"):
        with pytest.raises(ValueError, match="KRABBY_OMNI_STAND_MASK"):
            mk.parse_stand_mask(bad)


def test_env_knob_writes_exactly_the_param():
    pristine = SC._dump(Z2._cfg())
    for env in ({}, {"KRABBY_OMNI_STAND_MASK": ""}, {"KRABBY_OMNI_STAND_MASK": "0"}):
        cfg = Z2._cfg()
        mk.apply_motion_knobs(cfg, env)
        assert SC._dump(cfg) == pristine, env
    z1 = Z2._cfg()
    mk.apply_motion_knobs(z1, {"KRABBY_OMNI_STAND_OBS": "clock"})
    off = Z2._cfg()
    mk.apply_motion_knobs(off, {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "0"})
    assert SC._dump(off) == SC._dump(z1)
    cfg = Z2._cfg()
    mk.apply_motion_knobs(cfg, {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"})
    z1.observations.policy.extreme_parkour_observations.params["stand_mask"] = True
    assert SC._dump(cfg) == SC._dump(z1)                              # Z1's cfg + exactly the one param


def test_env_knob_refusals(capsys):
    for env, cfg, match in (
            ({"KRABBY_OMNI_STAND_MASK": "1"}, Z2._cfg(), "KRABBY_OMNI_STAND_OBS=clock"),
            ({"KRABBY_OMNI_STAND_OBS": "1", "KRABBY_OMNI_STAND_MASK": "1"}, Z2._cfg(), "KRABBY_OMNI_STAND_OBS=clock"),
            ({"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_PROXY": "hold", "KRABBY_OMNI_STAND_MASK": "1"},
             Z2._cfg(), "KRABBY_OMNI_STAND_PROXY"),
            ({"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"}, Z2._cfg(lpf_obs=True),
             "KRABBY_ACTION_LPF_OBS")):
        with pytest.raises(RuntimeError, match=match):
            mk.apply_motion_knobs(cfg, env)
    capsys.readouterr()
    mk.apply_motion_knobs(Z2._cfg(), {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"})
    assert "[KNOB] KRABBY_OMNI_STAND_MASK=1 consumed" in capsys.readouterr().out


def test_composition_rule():
    ok = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"}
    for training in (True, False):
        mk.check_restance_composition(ok, training=training)
        mk.check_restance_composition({**ok, "KRABBY_OMNI_STAND_MASK": "0"}, training=training)
    for over, match in (({"KRABBY_OMNI_STAND_OBS": "0"}, "KRABBY_OMNI_STAND_OBS=clock"),
                        ({"KRABBY_OMNI_STAND_OBS": "1"}, "KRABBY_OMNI_STAND_OBS=clock"),
                        ({"KRABBY_OMNI_STAND_PROXY": "hold"}, "KRABBY_OMNI_STAND_PROXY"),
                        ({"KRABBY_ACTION_LPF_OBS": "1"}, "KRABBY_ACTION_LPF_OBS")):
        with pytest.raises(ValueError, match=match):
            mk.check_restance_composition({**ok, **over}, training=True)
    # (the lpf rule needs the mask: lpf_obs alone stays legal here)
    mk.check_restance_composition({"KRABBY_ACTION_LPF_OBS": "1"}, training=True)


def test_runner_side_bc_stand_mask(tmp_path):
    env = BC._z2c_env(tmp_path)
    plain, masked = SC._agent(), SC._agent()
    mk.apply_exploration_knobs(plain, env)
    mk.apply_exploration_knobs(masked, {**env, "KRABBY_OMNI_STAND_MASK": "1"})
    assert "bc_stand_mask" not in vars(plain.algorithm)
    assert masked.algorithm.bc_stand_mask is True
    del masked.algorithm.bc_stand_mask
    assert SC._dump(masked) == SC._dump(plain)                       # nothing else written
    off = SC._agent()
    mk.apply_exploration_knobs(off, {**env, "KRABBY_OMNI_STAND_MASK": "0"})
    assert SC._dump(off) == SC._dump(plain)
    # the mask without BC (the mask-only control eval, or a Z2 chain) writes nothing on the runner side
    z2 = {k: v for k, v in env.items() if k not in BC.BC_KNOBS}
    a, b = SC._agent(), SC._agent()
    mk.apply_exploration_knobs(a, z2)
    mk.apply_exploration_knobs(b, {**z2, "KRABBY_OMNI_STAND_MASK": "1"})
    assert SC._dump(a) == SC._dump(b)
    # the cross rules run there too, before anything is written
    agent = SC._agent()
    before = SC._dump(agent)
    with pytest.raises(ValueError, match="KRABBY_OMNI_STAND_PROXY"):
        mk.apply_exploration_knobs(agent, {**env, "KRABBY_OMNI_STAND_MASK": "1", "KRABBY_OMNI_STAND_PROXY": "hold"})
    assert SC._dump(agent) == before


# ===================================================================== the cloning bank
@pytest.fixture()
def ckpt(tmp_path):
    return SC._anchor_ckpt(tmp_path, seed=5)


def _widen_training_path(alg, cols: int = 257):
    """Pretend the actor's training-path slice is ``cols`` wide (the omni actor's 0:257) and reload the bank; the
    tiny actor's real slice (21) ends before the masked columns."""
    actor = alg.policy.actor
    actor.num_scan = cols - int(actor.num_prop) - int(actor.num_priv_explicit) - int(actor.num_priv_latent)
    alg._load_bc_bank()


def test_bc_stand_mask_masks_the_bank_rows_and_keeps_the_targets(tmp_path, ckpt, capsys):
    banks = BC._banks(tmp_path)
    full = BC._full_bank(banks)
    capsys.readouterr()
    plain = BC._z2c(tmp_path, ckpt, banks)
    line_plain = capsys.readouterr().out
    masked = BC._z2c(tmp_path, ckpt, banks, bc_stand_mask=True)
    line_masked = capsys.readouterr().out
    assert plain.bc_stand_mask is False and masked.bc_stand_mask is True
    assert "bc_stand_mask=" not in line_plain
    assert "bc_seed=5, bc_stand_mask=True" in [ln for ln in line_masked.splitlines()
                                               if ln.startswith("[PPO] construction:")][-1]
    target, _ = RST.restance_target(full, BC.P_Z2C)
    for alg in (plain, masked):
        assert torch.equal(alg.bc_target, target)
        assert torch.equal(alg.bc_obs, full[:, :BC.TRAIN_COLS])        # the tiny slice ends before column 65
    for alg in (plain, masked):
        _widen_training_path(alg)
    assert torch.equal(plain.bc_obs, full[:, :257])
    assert torch.equal(masked.bc_obs, RST.stand_mask_obs(full)[:, :257])
    assert torch.equal(masked.bc_obs[:, LAST_ACT], torch.zeros(full.shape[0], 18))
    assert torch.equal(masked.bc_obs[:, PROXY], torch.full((full.shape[0], 6), -0.5))
    keep = torch.ones(257, dtype=torch.bool)
    keep[MASKED] = False
    assert torch.equal(masked.bc_obs[:, keep], plain.bc_obs[:, keep])
    assert torch.equal(masked.bc_target, plain.bc_target) and torch.equal(masked.bc_target, target)
    # the bank and the env apply one function: a bank row masked here is what the env would feed on a stand row
    assert torch.equal(masked.bc_obs, RST.stand_mask_obs(full[:, :257]))


def test_bc_stand_mask_refusals_and_runner_filter(tmp_path, ckpt):
    with pytest.raises(ValueError, match="bc_coef is 0"):
        SC._statue_alg(tmp_path, seed=0, ckpt=ckpt, **BC._z2_kw(), bc_stand_mask=True)
    bank = [str(b) for b in BC._banks(tmp_path, sizes=(8,))]
    for bad in ("1", 1, None):
        with pytest.raises(ValueError, match="bc_stand_mask must be a bool"):
            BC._z2c(tmp_path, ckpt, bank, bc_stand_mask=bad)
    cfg = {"bc_coef": 1.0, "bc_bank": bank, "bc_batch_size": 2048, "bc_seed": 0, "bc_stand_mask": True}
    assert PH.runner_mod.filter_ppo_alg_cfg(PH.PPO, dict(cfg)) == cfg
    assert inspect.signature(PH.PPO.__init__).parameters["bc_stand_mask"].default is False


# ===================================================================== the ladder
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__p2r_z2d_test")


def test_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    var = "KRABBY_OMNI_STAND_MASK"
    assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS
    assert parsers[var] is rc.knobs()._mk.MOTION_KNOB_PARSERS[var]
    assert var not in rc.AGENT_DUMP_KEYS and "bc_stand_mask" in rc.BC_ALG_FIELDS
    reasons = []
    rc._check_knob(var, "1", "t", whitelist, parsers, reasons, set())
    rc._check_knob(var, "0", "t", whitelist, parsers, reasons, set())
    assert not reasons, reasons
    rc._check_knob(var, "on", "t", whitelist, parsers, reasons, set())
    assert len(reasons) == 1, reasons


def test_dump_check(rc):
    chk = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_STAND_MASK"]
    on = Z2.ENV_Z1 + "        stand_mask: true\n"
    assert chk(on, "1") and not chk(on, "0") and not chk(on, "")
    assert chk(Z2.ENV_Z1, "0") and chk(Z2.ENV_Z1, "") and chk(Z2.ENV_OFF, "") and not chk(Z2.ENV_Z1, "1")
    assert chk(Z2.ENV_Z1 + "        stand_mask: false\n", "0")
    assert not chk(Z2.ENV_OFF + "        stand_mask: true\n", "1")              # without the clock bit
    assert not chk(Z2.ENV_Z2 + "        stand_mask: true\n", "1")               # beside the held proxy
    # the real S1 dump: off passes, on fails
    env_txt, _agent_txt = SC._s1_dumps()
    assert chk(env_txt, "0") and not chk(env_txt, "1")


def test_dry_run_composition(rc):
    ok = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"}
    reasons = []
    rc._check_restance_composition(ok, "w", reasons)
    rc._check_restance_composition({"KRABBY_OMNI_STAND_MASK": "0"}, "w", reasons)
    assert not reasons
    for over, match in (({"KRABBY_OMNI_STAND_OBS": "0"}, "KRABBY_OMNI_STAND_OBS=clock"),
                        ({"KRABBY_OMNI_STAND_PROXY": "hold"}, "KRABBY_OMNI_STAND_PROXY"),
                        ({"KRABBY_ACTION_LPF_OBS": "1"}, "KRABBY_ACTION_LPF_OBS")):
        reasons = []
        rc._check_restance_composition({**ok, **over}, "w", reasons)
        assert len(reasons) == 1 and reasons[0].startswith("w: ") and match in reasons[0], (over, reasons)
    reasons = []                                         # a malformed value is _check_knob's to report, once
    rc._check_restance_composition({**ok, "KRABBY_OMNI_STAND_MASK": "on"}, "w", reasons)
    assert reasons == []


def test_log_audits(rc, tmp_path):
    log_path = tmp_path / "train.log"
    rec = {"log": str(log_path)}
    # the env-side consumed line (statue_log_checks covers the env-side Z2 knobs)
    env = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_OMNI_STAND_MASK": "1"}
    log_path.write_text("[KNOB] KRABBY_OMNI_STAND_MASK=1 consumed\n")
    assert dict(rc.statue_log_checks(rec, env))["log: [KNOB] KRABBY_OMNI_STAND_MASK consumed"] is True
    log_path.write_text("")
    assert dict(rc.statue_log_checks(rec, env))["log: [KNOB] KRABBY_OMNI_STAND_MASK consumed"] is False
    assert rc.statue_log_checks(rec, {**env, "KRABBY_OMNI_STAND_MASK": "0"}) == []
    # beside BC: the [PPO] construction line names bc_stand_mask=True
    name = "log: [PPO] construction names bc_stand_mask=True"
    bc_env = {**BC._z2c_env(tmp_path), "KRABBY_OMNI_STAND_MASK": "1"}
    full = BC._armed_log(bc_env).replace("bc_seed=0, ", "bc_seed=0, bc_stand_mask=True, ")
    log_path.write_text(full)
    got = rc.bc_log_checks(rec, bc_env)
    assert got and all(ok for _n, ok in got), [n for n, ok in got if not ok]
    assert dict(got)[name] is True
    log_path.write_text(full.replace("bc_stand_mask=True, ", ""))
    assert dict(rc.bc_log_checks(rec, bc_env))[name] is False
    log_path.write_text(full + "WARNING Ignoring unsupported PPO config keys for PPOWithExtractor: bc_stand_mask\n")
    assert dict(rc.bc_log_checks(rec, bc_env))["log: no BC field dropped by the runner"] is False
    # knob off: the Z2c audit, unchanged (no mask check)
    z2c_env = BC._z2c_env(tmp_path)
    log_path.write_text(BC._armed_log(z2c_env))
    got_off = rc.bc_log_checks(rec, z2c_env)
    assert name not in dict(got_off) and got_off == rc.bc_log_checks(rec, {**z2c_env, "KRABBY_OMNI_STAND_MASK": "0"})
