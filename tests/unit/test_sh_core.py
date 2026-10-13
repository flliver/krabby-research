"""Stand-height campaign (``2026-10-13_stand_height``) step 2: ``scripts/sh_core.py`` on CPU (no Isaac / Kit / GPU).

Pinned here (PLAN.md H1-H6 and the step-2 identities):

* the module is torch- and numpy-free at import (the eval launcher loads it before Kit) and writes no bytecode;
* its constants agree with the frozen facts (h_min 0.719, h_max 1.346), R's module (columns, legs, side, D3) and the
  omni observation term (yaw-rate slot 5, xy slots 8-9, norm 10, hidden clock 15);
* the posture map: q(1.064) == (0, 0) exactly (float64 and float32 1.064), interpolation == sh_kin.q, clamping;
* R_h: at 1.064 with clip 1 it is ``restance_target`` BITWISE on random obs (all five branches present); R reads no
  knee column (so the hip-only re-reference is complete); at other heights HOLD -> q and LIFT -> q + D3 lift, and plain
  R would misread the same posture; |R_h - q| <= 1 over the frozen range (residual labels);
* the limiter: rate, exact arrival, no overshoot, clamping, a 1.064 command bitwise inert, the init rules per start;
* the H6 restore on synthetic traces: engage / no-engage band, completion timing, the rod tolerance boundary, the
  settle counter reset, timeout at exactly the timeout steps, stand->stand merge, reset cuts, forced restores, and a
  zero command that never engages;
* the stand-net input: the caller's obs is never written, the estimate is taken on the caller's unmasked, unwritten
  prop, column 15 and the restore command zeroing land on the clone only; with mask 0 and a zero command it is
  sn_common's input bitwise (the identity-bridge premise);
* the router: S1 rows == clamp(S1, +-1) bitwise, stand rows == clamp(q + clamp(a_net, +-1)) (bitwise clamp(a_net) at
  1.064), hip targets beyond -1 at low heights; with the REAL S1 as the stand net, mask 0 and a zero command, every
  routed row is clamp(S1, +-1) bitwise;
* the warm start: only W[:, 15] of the actor backbone changes (norm 1.74 before), the sha is checked; with the REAL
  A-mask s17 networks a NONZERO column-15 command separates the zeroed and unzeroed nets, a zero command does not, and
  the zeroed net ignores column 15.
"""
from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1].parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
CORE_PATH = SCRIPTS / "sh_core.py"
KIN_PATH = SCRIPTS / "sh_kin.py"
PRED = TASK / "experiments/2026-10-11_stand_network/scripts"
RESTANCE_PY = TASK / "mdp/crab_hex_omni_restance.py"
OMNI_OBS_PY = TASK / "mdp/crab_hex_omni_observations.py"
RSL_MODULES = REPO / "parkour/scripts/rsl_rl/modules"
S1_DIR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
S1_CKPT = S1_DIR / "model_19994.pt"
AMASK = TASK / "policy/stand_net/model_4000.pt"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
W, P0, P1, NP = 1149, 207, 216, 75
CAMS = [18 + 2 * i for i in range(6)]
HIPS = [29 + i for i in range(6)]
DT = 0.02


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to campaign / predecessor / mdp files."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def C():
    return _load(CORE_PATH, "_t_sh_core")


@pytest.fixture(scope="module")
def K():
    return _load(KIN_PATH, "_t_sh_core_kin")


@pytest.fixture(scope="module")
def RST():
    return _load(RESTANCE_PY, "_t_sh_core_restance")


@pytest.fixture(scope="module")
def P(C):
    return C.r_params()


def bits(a: torch.Tensor) -> torch.Tensor:
    return a.contiguous().view(torch.int32) if a.dtype == torch.float32 else a.contiguous().view(torch.int64)


def bitwise_equal(a, b) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(bits(a), bits(b))


def obs_batch(n=4096, seed=0, width=W):
    """Random obs with R's branches all present: per tripod down (hip ~ 0) or up (hip ~ -0.125 rad), cams in +-0.6,
    15 % tilted rows; small noise elsewhere (non-zero last action / proxy / command columns)."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, width, generator=g) * 0.3
    o[:, 3:5] *= 0.1
    tilt = torch.rand(n, generator=g) < 0.15
    o[tilt, 3] = 0.2
    up_a = torch.rand(n, generator=g) < 0.3
    up_b = torch.rand(n, generator=g) < 0.3
    hips = torch.randn(n, 6, generator=g) * 0.012
    for leg in range(6):
        up = up_a if leg in (0, 3, 4) else up_b
        hips[:, leg] += torch.where(up, torch.full((n,), -0.125), torch.zeros(n))
    o[:, HIPS] = hips
    o[:, CAMS] = torch.rand(n, 6, generator=g) * 1.2 - 0.6
    return o


# ===================================================================================== module / constants
def test_import_is_torch_and_numpy_free_and_writes_no_bytecode():
    code = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(CORE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
        "assert 'numpy' not in sys.modules, 'numpy imported at module level'\n"
        "print(m.H_MIN, m.H_MAX)\n"
    )
    before = {p for p in SCRIPTS.iterdir()}
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["0.719", "1.346"]
    assert not (SCRIPTS / "__pycache__").exists()
    assert {p for p in SCRIPTS.iterdir()} == before


def test_frozen_constants_match_facts(C):
    fr = json.loads((CAMP / "facts.json").read_text())["frozen"]
    assert C.H_MIN == fr["h_min_m"] == 0.719
    assert C.H_MAX == fr["h_max_m"] == 1.346
    assert C.RESTORE_TIMEOUT_S == pytest.approx((1.064 - 0.719) / 0.08 + 1.0)
    assert C.steps_for(C.RESTORE_TIMEOUT_S, DT) == 266
    assert C.steps_for(C.RESTORE_SETTLE_S, DT) == 10
    assert C.H_SETTLED == 1.064 and C.HEIGHT_OBS_SCALE == 0.4 and C.HEIGHT_RATE_M_S == 0.08
    assert C.HEIGHT_COL == 15 and C.RESTORE_ZERO_COLS == (5, 8, 9, 10)
    assert (C.RAW_CLIP, C.CAM_CLIP, C.S1_CLIP, C.RESIDUAL_CLIP, C.HARNESS_CLIP) == (4.8, 1.0, 1.0, 1.0, 4.8)


def test_constants_match_r_module_and_d3(C, RST):
    assert C.SN.RESTANCE_D3 == D3
    assert RST.format_restance(C.r_params()) == D3
    assert C.HIP_POS_OBS_COLS == RST.HIP_POS_OBS_COLS
    assert C.SIDE == RST.SIDE
    assert C.R_OBS_WIDTH == RST.MIN_OBS_WIDTH
    assert (C.CAM_ACT, C.HIP_ACT, C.KNEE_ACT) == (RST.CAM_ACT, RST.HIP_ACT, RST.KNEE_ACT)
    assert C.ROD_SCALE == RST.ROD_SCALE


def _class_literals(path: Path, cls: str) -> dict:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out = {}
            for st in node.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    try:
                        out[st.targets[0].id] = ast.literal_eval(st.value)
                    except ValueError:
                        pass
            return out
    raise AssertionError(f"{cls} not found in {path}")


def test_command_and_clock_columns_match_the_omni_observation_term(C):
    lit = _class_literals(OMNI_OBS_PY, "CrabHexOmniParkourObservations")
    assert lit["YAW_RATE_OBS_SLOT"] == 5
    assert lit["CMD_NORM_OBS_SLOT"] == 10
    assert lit["OBSERVED_COMMAND_SLOTS"] == {8: 0, 9: 1, 5: 2}
    assert lit["STAND_BIT_CLOCK_COL"] == C.HEIGHT_COL
    assert set(C.RESTORE_ZERO_COLS) == {lit["YAW_RATE_OBS_SLOT"], lit["CMD_NORM_OBS_SLOT"],
                                        *lit["OBSERVED_COMMAND_SLOTS"]}


# ===================================================================================== the posture map
def test_q_anchor_is_exactly_zero(C):
    for h in (1.064, torch.tensor(1.064, dtype=torch.float64), torch.tensor([1.064] * 3, dtype=torch.float32)):
        qh, qk = C.q_map(h)
        assert torch.all(qh == 0) and torch.all(qk == 0)
        assert not torch.signbit(qh).any() and not torch.signbit(qk).any()
    q = C.q18(torch.tensor([1.064, 1.064], dtype=torch.float64))
    assert q.shape == (2, 18) and torch.all(q == 0) and not torch.signbit(q).any()
    grid, qh, qk = C.q_table()
    i0 = int((abs(grid - 1.064) < 1e-15).nonzero()[0][0])
    assert grid[i0] == 1.064 and qh[i0] == 0.0 and qk[i0] == 0.0
    assert grid[0] <= C.H_MIN and grid[-1] >= C.H_MAX


def test_q_matches_sh_kin_and_clamps(C, K):
    import numpy as np

    g = torch.Generator().manual_seed(3)
    h = C.H_MIN + (C.H_MAX - C.H_MIN) * torch.rand(2000, generator=g, dtype=torch.float64)
    qh, qk = C.q_map(h, dtype=torch.float64)
    kh, kk = K.q(h.numpy())
    assert np.abs(qh.numpy() - kh).max() < 1e-5 and np.abs(qk.numpy() - kk).max() < 1e-5
    grid, th, tk = C.q_table()                      # exact at the nodes
    nodes = torch.tensor(grid[5:-5:37], dtype=torch.float64)
    nh, nk = C.q_map(nodes, dtype=torch.float64)
    assert torch.equal(nh, torch.tensor(th[5:-5:37])) and torch.equal(nk, torch.tensor(tk[5:-5:37]))
    min_h, min_k = C.q_map(C.H_MIN, dtype=torch.float64)
    max_h, max_k = C.q_map(C.H_MAX, dtype=torch.float64)
    for below in (0.311, 0.561, 0.7):                             # ground / table spawns read below h_min
        bh, bk = C.q_map(below, dtype=torch.float64)
        assert bh == min_h and bk == min_k
    for above in (1.4, 1.5):
        ah, ak = C.q_map(above, dtype=torch.float64)
        assert ah == max_h and ak == max_k
    assert min_h == pytest.approx(-2.748, abs=2e-3) and min_k == pytest.approx(0.371, abs=2e-3)
    assert torch.equal(C.clamp_h(torch.tensor([0.3, 1.0, 1.9])),
                       torch.tensor([0.719, float(torch.tensor(1.0)), 1.346], dtype=torch.float64))
    assert torch.all(torch.diff(C.q_map(torch.linspace(0.719, 1.346, 500, dtype=torch.float64))[0]) > 0)
    assert C.q_map(1.2)[0] == pytest.approx(0.903, abs=2e-3)


def test_encode_height(C):
    assert C.encode_height(1.064).item() == 0.0 and not torch.signbit(C.encode_height(1.064))
    assert C.encode_height(torch.tensor([1.464], dtype=torch.float64)).item() == pytest.approx(1.0)
    assert C.encode_height(0.664).item() == pytest.approx(-1.0)


# ===================================================================================== R_h
def test_rh_at_walking_height_is_restance_target_bitwise(C, RST, P):
    seen = set()
    for seed in range(4):
        obs = obs_batch(4096, seed)
        ref_t, ref_b = RST.restance_target(obs, P)
        seen |= set(ref_b.unique().tolist())
        for h in (1.064, torch.tensor(1.064, dtype=torch.float64), torch.full((4096,), 1.064, dtype=torch.float64),
                  torch.full((4096,), 1.064, dtype=torch.float32)):
            t, b = C.rh_target(obs, h, P, clip=1.0)
            assert bitwise_equal(t, ref_t)
            assert torch.equal(b, ref_b)
        t, b = C.rh_target(obs, 1.064, clip=1.0)          # default params = D3
        assert bitwise_equal(t, ref_t) and torch.equal(b, ref_b)
    assert seen == {RST.BRANCH_HOLD, RST.BRANCH_WAIT, RST.BRANCH_LIFT, RST.BRANCH_HOME, RST.BRANCH_SEAT}
    # the seed constants too (R_h is a wrapper, not D3-specific)
    obs = obs_batch(2048, 9)
    rp = RST.RestanceParams()
    ref_t, ref_b = RST.restance_target(obs, rp)
    t, b = C.rh_target(obs, 1.064, rp, clip=1.0)
    assert bitwise_equal(t, ref_t) and torch.equal(b, ref_b)
    # with the raw clip at 1.064 nothing changes either (R's output is inside +-1)
    t48, _ = C.rh_target(obs, 1.064, rp)
    assert bitwise_equal(t48, ref_t)


def test_r_reads_only_tilt_cams_and_hips(RST, P):
    """Why the hip-only re-reference is complete: R reads no knee column (35-40) nor anything else of the head."""
    obs = obs_batch(2048, 11)
    ref = RST.restance_target(obs, P)
    keep = {3, 4, *CAMS, *HIPS}
    other = [c for c in range(W) if c not in keep]
    o2 = obs.clone()
    o2[:, other] = torch.randn(2048, len(other), generator=torch.Generator().manual_seed(5)) * 3.0
    got = RST.restance_target(o2, P)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_rh_at_other_heights(C, RST, P):
    side = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    for h in (0.719, 0.8, 0.9, 1.2, 1.346):
        qh, qk = (float(v) for v in C.q_map(h, dtype=torch.float64))
        n = 3
        obs = torch.zeros(n, 89)
        obs[:, HIPS] = 0.25 * qh                      # every leg exactly at the posture
        obs[1, [CAMS[i] for i in (0, 3, 4)]] = 0.5    # row 1: tripod A off its home beyond eps_step -> LIFT
        obs[2, HIPS] = 0.25 * qh - 0.125              # row 2: every leg lifted by R's lift -> SEAT (homed in air)
        t, b = C.rh_target(obs, h, P)
        # row 0: all HOLD at the posture -> rods = q, cams = trim(0)
        assert torch.all(b[0] == RST.BRANCH_HOLD)
        assert torch.allclose(t[0, 6:12], torch.full((6,), qh), atol=1e-6)
        assert torch.allclose(t[0, 12:18], side * qk, atol=1e-6)
        assert torch.all(t[0, 0:6] == 0)
        # row 1: A LIFT (q + lift, knee q + 0.5 left-convention), B HOLD at q
        a_legs, b_legs = [0, 3, 4], [1, 2, 5]
        assert torch.all(b[1, a_legs] == RST.BRANCH_LIFT) and torch.all(b[1, b_legs] == RST.BRANCH_HOLD)
        assert torch.allclose(t[1, [6 + i for i in a_legs]], torch.full((3,), qh - 0.5), atol=1e-6)
        assert torch.allclose(t[1, [12 + i for i in a_legs]], side[a_legs] * (qk + 0.5), atol=1e-6)
        assert torch.allclose(t[1, [6 + i for i in b_legs]], torch.full((3,), qh), atol=1e-6)
        assert torch.allclose(t[1, [12 + i for i in b_legs]], side[b_legs] * qk, atol=1e-6)
        # row 2: not down, homed -> SEAT -> rods back to q
        assert torch.all(b[2] == RST.BRANCH_SEAT)
        assert torch.allclose(t[2, 6:12], torch.full((6,), qh), atol=1e-6)
        # the same rows read by plain R at a low height: the posture is "not down" (re-reference matters)
        if h <= 0.9:
            _, b_plain = RST.restance_target(obs, P)
            assert torch.all(b_plain[0] == RST.BRANCH_SEAT)


def test_rh_equals_r_on_the_rereferenced_obs_plus_q(C, RST, P):
    g = torch.Generator().manual_seed(21)
    obs = obs_batch(4096, 21)
    h = C.H_MIN + (C.H_MAX - C.H_MIN) * torch.rand(4096, generator=g, dtype=torch.float64)
    q = C.q18(h)
    obs[:, HIPS] += 0.25 * q[:, 6:12]
    rel_obs = obs.clone()
    rel_obs[:, HIPS] = obs[:, HIPS] - 0.25 * q[:, 6:12]
    rel, br = RST.restance_target(rel_obs, P)
    t, b = C.rh_target(obs, h, P)
    assert torch.equal(b, br)
    assert torch.allclose(t, (rel + q).clamp(-4.8, 4.8), atol=1e-6)
    assert t[:, 6:].abs().max() <= 4.8 and t[:, :6].abs().max() <= 1.0
    lab = C.rh_label(obs, h, P)
    assert lab.label.abs().max() <= 1.0 + 1e-6
    assert torch.allclose(lab.label, rel, atol=1e-6)            # the residual label is R's relative output
    assert torch.equal(lab.target, t) and torch.equal(lab.branch, b) and torch.equal(lab.q18, q)
    assert torch.equal(C.residual_label(t, h), lab.label)
    assert float(C.label_excess(lab.label)) <= 1e-6


def test_residual_label_refuses_an_inexpressible_target(C):
    h = torch.tensor([0.9, 1.2], dtype=torch.float64)
    t = C.q18(h).clone()
    t[1, 7] += 1.5
    with pytest.raises(ValueError, match="residual clip"):
        C.residual_label(t, h)
    lab = C.residual_label(t, h, check=False)
    assert float(C.label_excess(lab)) == pytest.approx(0.5, abs=1e-5)
    t[1, 7] -= 0.6                                   # 0.9 above q: fine
    C.residual_label(t, h)


# ===================================================================================== the limiter
def test_limiter_rate_arrival_and_no_overshoot(C):
    lim = C.HeightLimiter(2, "cpu", dt=DT)
    lim.reset(None, torch.tensor([0.8, 1.3], dtype=torch.float64))
    step = 0.08 * DT
    prev = lim.state.clone()
    tgt = torch.tensor([1.2, 0.75], dtype=torch.float64)
    reached = [None, None]
    for k in range(1, 400):
        s = lim.step(tgt)
        d = (s - prev).abs()
        assert torch.all(d <= step + 1e-15)
        assert s[0] <= 1.2 and s[1] >= 0.75
        for i in range(2):
            if reached[i] is None and s[i] == tgt[i]:
                reached[i] = k
        prev = s.clone()
    assert reached == [math.ceil(0.4 / step - 1e-9), math.ceil(0.55 / step - 1e-9)]
    assert lim.state[0].item() == 1.2 and lim.state[1].item() == 0.75


def test_limiter_zero_command_is_bitwise_inert(C):
    lim = C.HeightLimiter(64, "cpu", dt=DT)
    one = torch.full((64,), 1.064, dtype=torch.float64)
    for _ in range(500):
        assert torch.equal(lim.step(1.064), one)


def test_limiter_clamps(C):
    lim = C.HeightLimiter(3, "cpu", dt=DT)
    lim.reset(None, torch.tensor([0.311, 0.9, 1.6], dtype=torch.float64))
    assert lim.state.tolist() == [0.719, 0.9, 1.346]
    for _ in range(1000):
        lim.step(torch.tensor([2.0, 0.2, 1.0], dtype=torch.float64))
    assert lim.state.tolist() == [1.346, 0.719, 1.0]


def test_controller_reset_init_rules(C):
    hc = C.HeightController(5, "cpu", dt=DT)
    meas = torch.tensor([0.311, 0.561, 0.9, 1.5, 1.0], dtype=torch.float64)
    hc.reset(None, "walk")                                       # walk: the command (default walking height)
    assert torch.equal(hc.h_lim, torch.full((5,), 1.064, dtype=torch.float64))
    hc.reset(None, "walk", h_cmd=1.2, h_meas=meas)               # walk ignores the measurement
    assert torch.all(hc.h_lim == 1.2)
    for kind in ("table", "ground", "random"):
        hc.reset(None, kind, h_meas=meas)
        assert hc.h_lim.tolist() == [0.719, 0.719, 0.9, 1.346, 1.0]
        with pytest.raises(ValueError):
            hc.reset(None, kind)
    codes = torch.tensor([0, 1, 2, 3, 0])                        # per-row kinds, START_KINDS order
    hc.reset(None, codes, h_cmd=1.1, h_meas=meas)
    assert hc.h_lim.tolist() == [1.1, 0.719, 0.9, 1.346, 1.1]
    rows = torch.tensor([True, False, False, False, False])
    hc.reset(rows, "ground", h_meas=meas)
    assert hc.h_lim.tolist() == [0.719, 0.719, 0.9, 1.346, 1.1]
    with pytest.raises(ValueError):
        hc.reset(None, "sideways")


# ===================================================================================== the H6 restore
def _obs_hips(n, hips):
    o = torch.zeros(n, 89)
    o[:, HIPS] = torch.as_tensor(hips, dtype=torch.float32).expand(n, 6)
    return o


def _drive(hc, plan):
    """plan: list of (slot bool, h_cmd float, hips (6,)) for one env; returns the per-step HeightStep list."""
    out = []
    for slot, h, hips in plan:
        out.append(hc.step(torch.tensor([bool(slot)]), h, _obs_hips(1, hips)))
    return out


def _hip_offset_for_rod_dev(C, dev_m):
    lo, hi = -0.2, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = float(C.hip_rod_dev(_obs_hips(1, [mid] * 6).double())[0, 0])
        if d < dev_m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stand_then_leave(h_stand, n_stand=200):
    home = [0.0] * 6
    return [(True, h_stand, home)] * n_stand


def test_restore_engages_and_completes_on_time(C):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "walk")
    home = [0.0] * 6
    steps = _drive(hc, _stand_then_leave(0.9))
    assert steps[-1].h_lim.item() == 0.9 and not steps[-1].restoring.item()
    walk = _drive(hc, [(False, 0.9, home)] * 200)
    assert walk[0].engaged.item() and walk[0].restoring.item() and walk[0].stand_eff.item()
    k_star = math.ceil((1.064 - 0.9) / (0.08 * DT) - 1e-9)       # limiter advances to reach 1.064
    done = [i for i, s in enumerate(walk) if s.completed.item()]
    assert done == [k_star + hc.settle_steps - 1]
    i = done[0]
    assert all(s.stand_eff.item() for s in walk[:i]) and not walk[i].stand_eff.item()
    assert walk[k_star - 1].h_lim.item() == 1.064 and walk[k_star - 2].h_lim.item() < 1.064
    assert not any(s.timed_out.item() for s in walk) and not any(s.stand_eff.item() for s in walk[i:])
    assert hc.counts() == {"engaged": 1, "completed": 1, "timed_out": 0, "merged": 0, "reset_cut": 0}


@pytest.mark.parametrize("h_stand,engages", [(1.064, False), (1.08, False), (1.048, False), (1.09, True),
                                             (1.03, True), (0.719, True), (1.346, True)])
def test_restore_engage_band(C, h_stand, engages):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "walk")
    _drive(hc, _stand_then_leave(h_stand, 400))
    # any terrain / hip state: the trigger reads only the limited command
    s = hc.step(torch.tensor([False]), None, _obs_hips(1, [-0.3, 0.1, 0.0, -0.2, 0.05, 0.0]))
    assert s.engaged.item() is engages and s.stand_eff.item() is engages


def test_restore_rod_tolerance_boundary(C):
    inside = _hip_offset_for_rod_dev(C, -0.0079)
    outside = _hip_offset_for_rod_dev(C, -0.0081)
    for off, completes in ((inside, True), (outside, False)):
        hc = C.HeightController(1, "cpu", dt=DT)
        hc.reset(None, "walk")
        _drive(hc, _stand_then_leave(1.2))
        hips = [0.0, 0.0, off, 0.0, 0.0, 0.0]                    # one leg at the boundary
        walk = _drive(hc, [(False, None, hips)] * 300)
        assert any(s.completed.item() for s in walk) is completes
        assert any(s.timed_out.item() for s in walk) is (not completes)


def test_restore_settle_counter_resets_on_a_violation(C):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "walk")
    _drive(hc, _stand_then_leave(1.1))
    k_star = math.ceil((1.1 - 1.064) / (0.08 * DT) - 1e-9)
    home, bad = [0.0] * 6, [-0.1] + [0.0] * 5
    plan = [(False, None, home)] * (k_star + 9) + [(False, None, bad)] + [(False, None, home)] * 40
    walk = _drive(hc, plan)
    done = [i for i, s in enumerate(walk) if s.completed.item()]
    # 9 good observations after arrival, one violation, then 10 more good ones
    assert done == [k_star + 9 + 10]


def test_restore_times_out_after_exactly_the_timeout_steps(C):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "walk")
    _drive(hc, _stand_then_leave(0.719, 400))
    lifted = [-0.10] * 6                                         # ~11.7 mm of rod: never within delta
    walk = _drive(hc, [(False, None, lifted)] * 400)
    to = [i for i, s in enumerate(walk) if s.timed_out.item()]
    assert to == [hc.timeout_steps] == [266]
    assert all(s.stand_eff.item() for s in walk[:266]) and not any(s.stand_eff.item() for s in walk[266:])
    assert walk[265].h_lim.item() == 1.064                      # the ramp from h_min fits inside the timeout
    assert hc.counts()["timed_out"] == 1 and hc.counts()["completed"] == 0


def test_restore_merge_continues_from_the_limiter_state(C):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "walk")
    home = [0.0] * 6
    _drive(hc, _stand_then_leave(0.8))
    walk = _drive(hc, [(False, None, home)] * 30)
    assert walk[0].engaged.item() and all(s.restoring.item() for s in walk)
    h_before = walk[-1].h_lim.item()
    s = hc.step(torch.tensor([True]), 1.2, _obs_hips(1, home))  # a stand arrives mid-restore
    assert s.merged.item() and not s.restoring.item() and s.stand_eff.item()
    assert s.h_lim.item() == pytest.approx(h_before + 0.08 * DT)
    stand = _drive(hc, [(True, 1.2, home)] * 400)
    assert stand[-1].h_lim.item() == 1.2
    s = hc.step(torch.tensor([False]), None, _obs_hips(1, home))
    assert s.engaged.item()
    assert hc.counts()["merged"] == 1 and hc.counts()["engaged"] == 2


def test_reset_cuts_a_running_restore(C):
    hc = C.HeightController(2, "cpu", dt=DT)
    hc.reset(None, "walk")
    for _ in range(200):
        hc.step(torch.tensor([True, True]), torch.tensor([0.9, 1.064], dtype=torch.float64), _obs_hips(2, [0.0] * 6))
    s = hc.step(torch.tensor([False, False]), None, _obs_hips(2, [0.0] * 6))
    assert s.restoring.tolist() == [True, False]
    cut = hc.reset(torch.tensor([True, True]), "walk")
    assert cut.tolist() == [True, False]
    assert not hc.restoring.any() and hc.counts()["reset_cut"] == 1
    s = hc.step(torch.tensor([False, False]), None, _obs_hips(2, [0.0] * 6))     # no edge after a reset
    assert not s.engaged.any() and not s.stand_eff.any()


def test_force_restore_on_a_random_height_start(C):
    hc = C.HeightController(1, "cpu", dt=DT)
    hc.reset(None, "random", h_meas=torch.tensor([0.85], dtype=torch.float64))
    hc.force_restore(torch.tensor([True]))
    walk = _drive(hc, [(False, None, [0.0] * 6)] * 300)
    assert walk[0].stand_eff.item()
    k_star = math.ceil((1.064 - 0.85) / (0.08 * DT) - 1e-9)
    assert [i for i, s in enumerate(walk) if s.completed.item()] == [k_star + hc.settle_steps - 1]


def test_zero_command_never_engages_and_stays_bitwise(C):
    n = 256
    g = torch.Generator().manual_seed(4)
    hc = C.HeightController(n, "cpu", dt=DT)
    hc.reset(None, "walk")
    slot = torch.zeros(n, dtype=torch.bool)
    one = torch.full((n,), 1.064, dtype=torch.float64)
    for t in range(600):
        flip = torch.rand(n, generator=g) < 0.02
        slot = slot ^ flip
        obs = torch.zeros(n, 89)
        obs[:, HIPS] = torch.randn(n, 6, generator=g) * 0.2
        s = hc.step(slot, 1.064, obs)
        assert torch.equal(s.h_lim, one) and torch.equal(s.stand_eff, slot) and not s.restoring.any()
        if t % 97 == 0:
            hc.reset(torch.rand(n, generator=g) < 0.1, "walk")
    assert hc.counts()["engaged"] == 0


def test_controller_refuses_a_timeout_shorter_than_the_longest_ramp(C):
    with pytest.raises(ValueError, match="longest ramp"):
        C.HeightController(1, "cpu", dt=DT, timeout_s=3.0)


# ===================================================================================== the stand-net input
class SpyEst:
    """An estimator that records its input and returns a fixed linear map of it."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(NP, P1 - P0, generator=g) * 0.1
        self.seen = []

    def inference(self, x):
        self.seen.append(x.detach().clone())
        return x @ self.w


def test_stand_net_input_never_writes_obs_and_estimates_first(C):
    obs = obs_batch(64, 2)
    obs[:, 15] = 0.7                                    # a non-zero col 15 (only to see where the estimate came from)
    obs[:, [5, 8, 9, 10]] = 0.4
    keep = obs.clone()
    h = torch.linspace(0.719, 1.346, 64, dtype=torch.float64)
    restoring = torch.arange(64) % 3 == 0
    spy, ref_spy = SpyEst(), SpyEst()
    x = C.stand_net_input(obs, spy, p0=P0, p1=P1, num_prop=NP, mask=1, h_cmd_limited=h, restoring=restoring)
    assert bitwise_equal(obs, keep)                                         # the caller's obs is untouched
    assert x.data_ptr() != obs.data_ptr()
    assert len(spy.seen) == 1 and bitwise_equal(spy.seen[0], keep[:, :NP])   # unmasked, unwritten prop
    ref = C.SN.stand_net_input(keep, ref_spy, p0=P0, p1=P1, num_prop=NP, mask=1)
    assert torch.allclose(x[:, 15], C.encode_height(h), atol=0) and x[:, 15].dtype == torch.float32
    cols = [5, 8, 9, 10]
    assert torch.all(x[restoring][:, cols] == 0)
    assert bitwise_equal(x[~restoring][:, cols], ref[~restoring][:, cols])
    other = [c for c in range(W) if c not in (15, *cols)]
    assert bitwise_equal(x[:, other], ref[:, other])                        # history col 15 etc. untouched
    assert torch.all(x[:, [259 + 89 * j + 15 for j in range(10)]] == ref[:, [259 + 89 * j + 15 for j in range(10)]])
    # the mask landed (estimate-then-mask): last action 0 / proxy -0.5 on the clone only
    assert torch.all(x[:, 65:83] == 0) and torch.all(x[:, 83:89] == -0.5)
    assert not torch.all(keep[:, 65:83] == 0)


def test_stand_net_input_zero_command_mask0_is_sn_common_bitwise(C):
    obs = obs_batch(128, 6)
    obs[:, 15] = 0.0                                    # CLOCK_HIDE + STAND_OBS=0
    est = SpyEst(1)
    a = C.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=0, h_cmd_limited=1.064)
    b = C.SN.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=0)
    assert bitwise_equal(a, b)
    a2 = C.stand_net_input(obs, est, p0=P0, p1=P1, num_prop=NP, mask=0, h_cmd_limited=1.064,
                           restoring=torch.zeros(128, dtype=torch.bool))
    assert bitwise_equal(a2, b)


# ===================================================================================== the router
def test_router_rows(C):
    g = torch.Generator().manual_seed(8)
    n = 4096
    a_net = torch.randn(n, 18, generator=g) * 2.0
    a_s1 = torch.randn(n, 18, generator=g) * 2.0
    h = C.H_MIN + (C.H_MAX - C.H_MIN) * torch.rand(n, generator=g, dtype=torch.float64)
    eff = torch.rand(n, generator=g) < 0.5
    out = C.router_actions(eff, a_net, a_s1, h)
    assert bitwise_equal(out.actions[~eff], torch.clamp(a_s1, -1.0, 1.0)[~eff])
    q = C.q18(h)
    want = q + a_net.clamp(-1, 1)
    want = torch.cat([want[:, :6].clamp(-1, 1), want[:, 6:].clamp(-4.8, 4.8)], dim=1)
    assert torch.allclose(out.actions[eff], want[eff], atol=0)
    assert torch.equal(out.q18, q) and bitwise_equal(out.a_s1, a_s1.clamp(-1, 1))
    assert torch.equal(out.actions, C.route(eff, out.a_stand, a_s1))
    # hip targets beyond -1 on low stand rows (the authority the posture offset buys)
    low = eff & (h < 0.85)
    assert out.actions[low][:, 6:12].min() < -2.0 and out.actions[:, 6:].abs().max() <= 4.8
    # at walking height the stand rows are clamp(a_net, +-1) BITWISE
    out1 = C.router_actions(eff, a_net, a_s1, 1.064)
    assert bitwise_equal(out1.actions[eff], a_net.clamp(-1, 1)[eff])
    assert bitwise_equal(C.stand_action(a_net, 1.064)[0], a_net.clamp(-1, 1))
    with pytest.raises(ValueError):
        C.route(eff.float(), a_net, a_s1)
    assert torch.equal(C.effective_stand(torch.tensor([True, False, False]), torch.tensor([False, True, False])),
                       torch.tensor([True, True, False]))


# ===================================================================================== h_meas
def test_h_meas_and_terrain_raycast(C):
    rz = torch.tensor([1.1, 0.6])
    tz = torch.tensor([[0.0] * 6, [0.1, 0.0, 0.2, 0.0, 0.1, 0.2]])
    assert C.h_meas(rz, tz).tolist() == pytest.approx([1.1, 0.5])
    fp = torch.rand(2, 6, 3)
    calls = {}

    def fake_raycast(starts, dirs, mesh):
        calls["starts"], calls["dirs"], calls["mesh"] = starts, dirs, mesh
        hits = starts.clone()
        hits[:, 2] = 0.25
        return hits, None

    z = C.terrain_z_under_feet(fp, "MESH", fake_raycast)
    assert z.shape == (2, 6) and torch.all(z == 0.25)
    assert torch.allclose(calls["starts"][:, 2], fp.reshape(-1, 3)[:, 2] + 5.0)
    assert torch.all(calls["dirs"] == torch.tensor([0.0, 0.0, -1.0])) and calls["mesh"] == "MESH"
    with pytest.raises(RuntimeError, match="no terrain mesh"):
        C.terrain_z_under_feet(fp, None, fake_raycast)


# ===================================================================================== the warm start
def test_zero_height_column_changes_only_w15(C):
    if not AMASK.is_file():
        pytest.skip("policy/stand_net/model_4000.pt is not on this machine")
    ck = torch.load(AMASK, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    orig = {k: v.clone() for k, v in sd.items()}
    new, norm = C.zero_height_column(sd)
    assert norm == pytest.approx(1.7404, abs=1e-3)
    for k in sd:
        assert torch.equal(sd[k], orig[k])                       # the source is never written
        if k != C.WARM_ZERO_KEY:
            assert torch.equal(new[k], orig[k])
    w0, w1 = orig[C.WARM_ZERO_KEY], new[C.WARM_ZERO_KEY]
    assert torch.all(w1[:, 15] == 0)
    keep = [c for c in range(w0.shape[1]) if c != 15]
    assert torch.equal(w1[:, keep], w0[:, keep])


def test_warm_start_state_checks_the_sha(C, tmp_path):
    if not AMASK.is_file():
        pytest.skip("policy/stand_net/model_4000.pt is not on this machine")
    st = C.warm_start_state()
    assert st["warm_start"]["sha256"] == C.STAND_NET_SHA256 and st["iter"] == 4000
    assert torch.all(st["model_state_dict"][C.WARM_ZERO_KEY][:, 15] == 0)
    bad = tmp_path / "x.pt"
    bad.write_bytes(AMASK.read_bytes() + b"0")
    with pytest.raises(RuntimeError, match="sha256"):
        C.warm_start_state(bad)


def rsl_modules():
    name = "_sh_test_rslmods"
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(RSL_MODULES)]
            sys.modules[name] = pkg
        crab = importlib.import_module(f"{name}.crab_actor_critic_with_encoder")
        est = importlib.import_module(f"{name}.feature_extractors.estimator")
    finally:
        sys.dont_write_bytecode = old
    return crab.CrabHexActorCriticRMA, est.DefaultEstimator


@pytest.fixture(scope="module")
def nets():
    if not (S1_CKPT.is_file() and AMASK.is_file()):
        pytest.skip("S1 model_19994.pt / the stand net is not on this machine")
    crab_cls, est_cls = rsl_modules()
    agent = yaml.safe_load((S1_DIR / "params/agent.yaml").read_text())

    def policy():
        pc = copy.deepcopy(agent["policy"])
        pc.pop("class_name")
        return crab_cls(W, 18, **pc)

    ec = copy.deepcopy(agent["estimator"])
    ec.pop("class_name")
    s1ck = torch.load(S1_CKPT, map_location="cpu", weights_only=False)
    amck = torch.load(AMASK, map_location="cpu", weights_only=False)
    s1, est, am = policy(), est_cls(**ec), policy()
    s1.load_state_dict(s1ck["model_state_dict"], strict=True)
    est.load_state_dict(s1ck["estimator_state_dict"], strict=True)
    am.load_state_dict(amck["model_state_dict"], strict=True)
    return {"s1": s1.eval(), "est": est.eval(), "am": am.eval(), "policy": policy,
            "am_est_state": amck["estimator_state_dict"], "s1_est_state": s1ck["estimator_state_dict"]}


def test_warm_start_column15_identities_on_the_real_networks(C, nets):
    zeroed = nets["policy"]()
    rec = C.load_warm_start(zeroed)
    zeroed.eval()
    assert rec["w15_norm_before"] == pytest.approx(1.7404, abs=1e-3)
    for k, v in nets["s1_est_state"].items():                   # D1: the stand net's estimator is S1's
        assert torch.equal(rec["estimator_state_dict"][k].cpu(), v.cpu())
    obs = obs_batch(256, 13)
    obs[:, 15] = 0.0
    with torch.no_grad():
        x0 = C.stand_net_input(obs, nets["est"], p0=P0, p1=P1, num_prop=NP, mask=1, h_cmd_limited=1.064)
        x1 = C.stand_net_input(obs, nets["est"], p0=P0, p1=P1, num_prop=NP, mask=1, h_cmd_limited=1.25)
        assert torch.all(x0[:, 15] == 0) and torch.all(x1[:, 15] > 0.4)
        assert torch.equal(x0[:, [c for c in range(W) if c != 15]], x1[:, [c for c in range(W) if c != 15]])
        u0 = nets["am"].act_inference(x0, hist_encoding=True)
        z0 = zeroed.act_inference(x0, hist_encoding=True)
        u1 = nets["am"].act_inference(x1, hist_encoding=True)
        z1 = zeroed.act_inference(x1, hist_encoding=True)
    assert bitwise_equal(u0, z0)                                 # zero command: equal
    assert (u1 - z1).abs().max() > 1e-3                          # non-zero command: they differ
    assert bitwise_equal(z1, z0)                                 # the zeroed net ignores column 15
    assert (u1 - u0).abs().max() > 1e-3                          # the unzeroed one does not


def test_identity_bridge_premise_on_the_real_s1(C, nets):
    """S1 as the stand net, mask 0, a zero command, the new clip path: every routed row is clamp(S1, +-1) bitwise."""
    obs = obs_batch(512, 17)
    obs[:, 15] = 0.0
    with torch.no_grad():
        h = obs.clone()
        h[:, P0:P1] = nets["est"].inference(h[:, :NP])          # the harness's in-place estimate
        a_s1 = nets["s1"].act_inference(h, hist_encoding=True)
        x = C.stand_net_input(h, nets["est"], p0=P0, p1=P1, num_prop=NP, mask=0, h_cmd_limited=1.064,
                              restoring=torch.zeros(512, dtype=torch.bool))
        a_net = nets["s1"].act_inference(x, hist_encoding=True)
    assert bitwise_equal(a_net, a_s1)
    hc = C.HeightController(512, "cpu", dt=DT)
    hc.reset(None, "walk")
    slot = torch.rand(512, generator=torch.Generator().manual_seed(1)) < 0.5
    st = hc.step(slot, 1.064, h)
    out = C.router_actions(st.stand_eff, a_net, a_s1, st.h_lim)
    assert bitwise_equal(out.actions, a_s1.clamp(-1.0, 1.0))
    assert bitwise_equal(out.actions.clamp(-C.HARNESS_CLIP, C.HARNESS_CLIP), out.actions)   # the 4.8 wrapper clip


# ===================================================================================== records
def test_records(C):
    pv = C.provenance()
    for k in ("sh_core_sha256", "sh_kin_sha256", "sn_common_sha256", "restance_sha256", "facts_json_sha256"):
        assert len(pv[k]) == 64
    assert pv["h_min"] == 0.719 and pv["h_max"] == 1.346 and pv["restore_rod_tol_m"] == 0.008
    assert pv["restance_d3"] == D3 and pv["clip"]["harness_clip_actions"] == 4.8
    rec = C.HeightController(4, "cpu", dt=DT).record()
    assert rec["timeout_steps"] == 266 and rec["settle_steps"] == 10
