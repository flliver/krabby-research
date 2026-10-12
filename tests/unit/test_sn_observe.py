"""Stand-network campaign (``2026-10-11_stand_network``) step 9: ``scripts/sn_observe_read.py`` (the closed-loop observe
pass) on SYNTHETIC raws, CPU only (no Isaac / Kit / GPU, no real stamp read).

Pinned here:

* the pure pieces: stand runs, the LIFT-onset / entry launch read (onset rule = ``sn_dagger_core.lift_onset``'s,
  inclusive 0.3-s window, censoring, first-per-stand-leg), OLS and the within-leg pooled slope;
* the statistics on a hand-built episode: |applied - R| by branch and channel group on the CLIPPED action, mu_hip on
  LIFT rows and the < -0.25 share, the per-leg HOLD cam-trim slope / intercept / fixed point in the |c| band, R's own
  slope on the same rows, counts;
* the read's self-checks refuse raws that do not mean what it assumes (routing, branch rows, c alignment, LIFT target),
  and pass on raws labelled by the REAL ``restance_target`` with the D3 constants;
* the CLI end to end on a synthetic lr tree: newest COMPLETE stamp, PENDING rows (in-progress named), pooled families,
  exit codes 0 / 2 / 3 / 1, the run_meta validity rule and the across-stamp consistency rule, nothing written but the
  two output paths.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
OBSERVE = SCRIPTS / "sn_observe_read.py"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
JOINT_NAMES = [n for leg in ("FL", "FR", "ML", "MR", "RL", "RR")
               for n in (f"{leg}_Body_Hip_RevoluteJoint", f"{leg}_Body_CamShaft_RevoluteJoint")]
JOINT_NAMES += [f"{leg}_Hip_Femur_RevoluteJoint" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")]
JOINT_NAMES += [f"{leg}_Femur_Tibia_RevoluteJoint" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")]
CAM_COLS = [1, 3, 5, 7, 9, 11]
HOLD, WAIT, LIFT, HOME, SEAT = range(5)
SIDE = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
SHA_A, SHA_B, SHA_S1 = "a" * 64, "b" * 64, "c" * 64


def _load(path: Path, name: str):
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
def OB():
    return _load(OBSERVE, "_t_sn_observe")


@pytest.fixture(scope="module")
def P(OB):
    return OB.r_params()


# ===================================================================================== synthetic raws
def r_like_target(br, c, p):
    """R's target for given branches / cams as R writes it (HOLD: cam trim; LIFT / HOME: hip lift + knee; else 0 --
    HOME / SEAT cams are 0 here, which the read never checks)."""
    br = np.asarray(br)
    c = np.asarray(c, dtype=np.float64)
    cam = np.where(br == HOLD, -np.clip(p.k_h * c, -p.trim, p.trim), 0.0)
    hip = np.where((br == LIFT) | (br == HOME), p.lift, 0.0)
    knee = -p.rho * SIDE[None, :] * hip
    return np.clip(np.concatenate([cam, hip, knee], axis=1), -1.0, 1.0)


def episode(stand, br, a, c, p, *, tgt=None, actions=None, cam_offset=0.0):
    """One raw episode (npz keys) from per-row stand-row content: ``stand`` (T,), ``br`` (T, 6), ``a`` (T, 18),
    ``c`` (T, 6); off-stand rows are zeroed / -1 as the launcher writes them."""
    stand = np.asarray(stand, dtype=bool)
    T = stand.shape[0]
    br = np.asarray(br)
    c = np.asarray(c, dtype=np.float64)
    if tgt is None:
        tgt = r_like_target(br, c, p)
    sm = stand[:, None]
    a_full = np.where(sm, a, 0.0).astype(np.float32)
    jp = np.zeros((T, 24), dtype=np.float32)
    jp[:, CAM_COLS] = (c + cam_offset).astype(np.float32)
    return {"p2r_stand_mode": stand,
            "sn_stand_action": a_full,
            "sn_s1_action": np.zeros((T, 18), dtype=np.float32),
            "sn_r_target": np.where(sm, tgt, 0.0).astype(np.float32),
            "sn_r_branch": np.where(sm, br, -1).astype(np.int8),
            "joint_pos": jp,
            "actions": a_full.copy() if actions is None else np.asarray(actions, dtype=np.float32)}


def acc_of(OB, P, eps, dt=0.02):
    acc = OB.Acc()
    window = int(round(OB.LAUNCH_WINDOW_S / dt))
    for d in eps:
        acc.add_episode(d, CAM_COLS, window, P, LIFT, HOLD)
    return acc


def hold_episode(P, T=120, seed=0, slope=-0.5, icpt=(0.01, -0.02, 0.0, 0.03, -0.01, 0.02)):
    """Every leg HOLD on every stand row (rows 10..T-1), c spread over +-[0.0, 0.3], cam action = slope c + icpt[leg],
    hip 0.1, knee 0."""
    rng = np.random.default_rng(seed)
    stand = np.zeros(T, dtype=bool)
    stand[10:] = True
    c = rng.uniform(-0.3, 0.3, size=(T, 6))
    br = np.full((T, 6), HOLD)
    a = np.zeros((T, 18))
    a[:, 0:6] = slope * c + np.asarray(icpt)[None, :]
    a[:, 6:12] = 0.1
    return episode(stand, br, a, c, P), stand, c, a


# ===================================================================================== pure pieces
def test_stand_runs(OB):
    assert OB.stand_runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
    assert OB.stand_runs([False, False]) == []
    assert OB.stand_runs([True, True, True]) == [(0, 3)]


def test_parse_row(OB):
    assert OB.parse_row("walkstop@1") == ("walkstop", 1)
    assert OB.parse_row("suite__omni@2") == ("suite", 2)
    for bad in ("field_hard@1", "walkstop", "walkstop@0", "suite@x"):
        with pytest.raises(ValueError):
            OB.parse_row(bad)


def _lift_low(n=30):
    lift = np.zeros((n, 6), dtype=bool)
    low = np.zeros((n, 6), dtype=bool)
    return lift, low


def test_launch_events_onsets_entry_censoring(OB):
    lift, low = _lift_low(30)
    lift[5:11, 0] = True
    low[8, 0] = True                     # leg 0: onset t=5, launched at latency 3
    lift[3:7, 1] = True                  # leg 1: onset t=3, never low, window inside the run -> not launched
    lift[0:3, 2] = True
    low[2, 2] = True                     # leg 2: entry LIFT (t=0) -> entry event, latency 2; no onset
    lift[20:23, 3] = True                # leg 3: onset t=20, never low, window cut by the run end -> censored
    lift[4:6, 4] = True
    lift[8:10, 4] = True
    low[9, 4] = True                     # leg 4: onsets t=4 (latency 5) and t=8 (latency 1); first = t=4
    lift[2:4, 5] = True
    low[17, 5] = True                    # leg 5: onset t=2, low exactly at t+15 -> launched (inclusive window)
    ev = OB.launch_events(lift, low, 15)
    on = ev["onset"]
    got = {(int(l), int(t)): (int(lat), bool(cen), bool(f))
           for l, t, lat, cen, f in zip(on["leg"], on["t"], on["latency"], on["censored"], on["first"])}
    assert got == {(0, 5): (3, False, True), (1, 3): (-1, False, True), (3, 20): (-1, True, True),
                   (4, 4): (5, False, True), (4, 8): (1, False, False), (5, 2): (15, False, True)}
    en = ev["entry"]
    assert en["leg"].tolist() == [2] and en["latency"].tolist() == [2] and en["censored"].tolist() == [False]


def test_launch_window_exclusive_beyond(OB):
    lift, low = _lift_low(40)
    lift[2:4, 0] = True
    low[18, 0] = True                    # t + 16: outside a 15-row window
    ev = OB.launch_events(lift, low, 15)
    assert ev["onset"]["latency"].tolist() == [-1] and ev["onset"]["censored"].tolist() == [False]


def test_launch_summary_shares(OB):
    lift, low = _lift_low(30)
    lift[5:11, 0] = True
    low[8, 0] = True
    lift[3:7, 1] = True
    lift[20:23, 3] = True
    lift[4:6, 4] = True
    lift[8:10, 4] = True
    low[4, 4] = True                     # leg 4: first onset already low (latency 0); second never low
    ev = OB.launch_events(lift, low, 15)
    s = OB.launch_summary(ev, 0.02)
    po = s["onset"]["pooled"]
    # onsets: leg0 (launched), leg1 (no), leg3 (censored), leg4 t4 (launched, latency 0), leg4 t8 (no)
    assert (po["events"], po["launched"], po["censored"]) == (5, 2, 1)
    assert po["launch_share"] == pytest.approx(2 / 4)
    assert po["already_low_at_event"] == 1
    assert po["latency_median_s"] == pytest.approx(np.median([3, 0]) * 0.02)
    ff = s["onset"]["pooled_first_per_stand_leg"]
    assert (ff["events"], ff["launched"], ff["censored"]) == (4, 2, 1)
    assert s["onset"]["per_leg"]["ML"]["events"] == 0 and s["onset"]["per_leg"]["ML"]["launch_share"] is None
    assert s["onset"]["per_leg"]["MR"]["censored"] == 1 and s["onset"]["per_leg"]["MR"]["launch_share"] is None


def test_ols_and_within(OB):
    x = np.linspace(0.03, 0.3, 50)
    f = OB.ols(x, -0.5 * x + 0.01)
    assert f["slope"] == pytest.approx(-0.5) and f["intercept"] == pytest.approx(0.01)
    assert f["r2"] == pytest.approx(1.0) and f["fixed_point"] == pytest.approx(0.02)
    assert OB.ols(x[:10], x[:10])["slope"] is None                     # under MIN_FIT_ROWS
    assert OB.ols(np.full(40, 0.1), np.arange(40.0))["slope"] is None  # constant x
    xa, xb = np.linspace(0.03, 0.3, 60), np.linspace(-0.3, -0.03, 60)
    ya, yb = -0.7 * xa + 0.17, -0.7 * xb - 0.17
    w = OB.ols_within([xa, xb], [ya, yb])
    assert w["slope"] == pytest.approx(-0.7) and w["n"] == 120
    plain = OB.ols(np.r_[xa, xb], np.r_[ya, yb])["slope"]
    assert plain > -0.3                                                 # pooling intercepts hides the slope


# ===================================================================================== statistics on built episodes
def test_hold_slope_per_leg_and_r_reference(OB, P):
    d, stand, c, a = hold_episode(P, T=400)
    acc = acc_of(OB, P, [d])
    assert OB.check_reasons(acc) == []
    out = OB.summarize(acc, 0.02, P)
    hs = out["hold_slope"]
    icpt = (0.01, -0.02, 0.0, 0.03, -0.01, 0.02)
    for i, leg in enumerate(OB.LEGS):
        f = hs["per_leg"][leg]
        m = stand & (np.abs(c[:, i]) >= 0.03) & (np.abs(c[:, i]) <= 0.3)
        assert f["n"] == int(m.sum())
        assert f["slope"] == pytest.approx(-0.5, abs=1e-4) and f["intercept"] == pytest.approx(icpt[i], abs=1e-4)
        if icpt[i]:
            assert f["fixed_point"] == pytest.approx(icpt[i] / 0.5, abs=1e-3)
        r_want = OB.ols(c[m, i], -np.clip(c[m, i], -0.1, 0.1))["slope"]
        assert f["r_slope_same_rows"] == pytest.approx(r_want, abs=1e-4)
        assert f["slope_unsat"] == pytest.approx(-0.5, abs=1e-4)
    assert hs["pooled"]["slope"] == pytest.approx(-0.5, abs=1e-4)
    assert hs["pooled"]["per_leg_slope_min"] == pytest.approx(-0.5, abs=1e-4)
    assert hs["r_slope"] == -1.0 and hs["r_clip"] == pytest.approx(0.1)
    # by_branch: HOLD only; hip |0.1 - 0| = 0.1 exactly; cam |(slope c + icpt) - (-clip(c))|
    bb = out["by_branch"]
    assert bb["hold"]["leg_rows"] == 6 * int(stand.sum()) and bb["lift"]["leg_rows"] == 0 and bb["lift"]["hip"] is None
    assert bb["hold"]["hip"]["mean_abs"] == pytest.approx(0.1, abs=1e-6)
    assert bb["hold"]["hip"]["mean_signed"] == pytest.approx(0.1, abs=1e-6)
    dc = (a[stand, 0:6] - (-np.clip(c[stand], -0.1, 0.1))).astype(np.float32).astype(np.float64)
    assert bb["hold"]["cam"]["mean_abs"] == pytest.approx(np.abs(dc).mean(), abs=1e-5)
    assert bb["hold"]["cam"]["p90_abs"] == pytest.approx(np.quantile(np.abs(dc), 0.9), abs=1e-5)
    assert out["counts"]["stand_rows"] == int(stand.sum()) and out["counts"]["stands"] == 1


def test_lift_hip_clip_and_by_branch(OB, P):
    T = 100
    stand = np.zeros(T, dtype=bool)
    stand[20:80] = True
    c = np.zeros((T, 6))
    br = np.full((T, 6), HOLD)
    br[30:40, 0] = LIFT                 # leg 0 LIFT for 10 rows, |c| 0.35, hip -0.4 (R -0.5)
    c[30:40, 0] = 0.35
    br[50:60, 1] = HOME                 # leg 1 HOME for 10 rows, hip raw -1.6 -> applied -1.0 (R -0.5)
    a = np.zeros((T, 18))
    a[30:35, 6] = -0.4
    a[35:40, 6] = -0.1                  # half the LIFT rows below -0.25
    a[50:60, 7] = -1.6
    out = OB.summarize(acc_of(OB, P, [episode(stand, br, a, c, P)]), 0.02, P)
    lh = out["lift_hip"]
    assert lh["lift"]["leg_rows"] == 10
    assert lh["lift"]["mu_hip"] == pytest.approx(-0.25) and lh["lift"]["share_below_thr"] == pytest.approx(0.5)
    assert lh["lift"]["cam_abs_median"] == pytest.approx(0.35, abs=1e-6)
    assert lh["home"]["mu_hip"] == pytest.approx(-1.0) and lh["home"]["mu_hip"] != pytest.approx(-1.6)
    bb = out["by_branch"]
    assert bb["lift"]["hip"]["mean_abs"] == pytest.approx(np.mean([0.1] * 5 + [0.4] * 5))
    assert bb["lift"]["hip"]["mean_signed"] == pytest.approx(np.mean([0.1] * 5 + [0.4] * 5))
    assert bb["home"]["hip"]["mean_abs"] == pytest.approx(0.5)          # |clip(-1.6) - (-0.5)|, the APPLIED action
    assert bb["home"]["knee"]["mean_abs"] == pytest.approx(0.5)         # knee target +0.5 on a right leg, action 0
    assert out["counts"]["preclip_out_of_range_share"]["hip"] == pytest.approx(10 / 60 / 6, abs=1e-5)
    assert out["counts"]["leg_rows_by_branch"]["lift"] == 10 and out["counts"]["lift_onsets"] == 1
    la = out["launch"]["onset"]["per_leg"]["FL"]
    assert (la["events"], la["launched"], la["already_low_at_event"]) == (1, 1, 1)


def test_multiple_stands_and_episodes_pool(OB, P):
    T = 90
    stand = np.zeros(T, dtype=bool)
    stand[10:30] = True
    stand[50:70] = True                 # two stands in one episode
    c = np.zeros((T, 6))
    br = np.full((T, 6), HOLD)
    br[10, 3] = LIFT                     # entry LIFT on the first stand row (not an onset)
    br[55:58, 3] = LIFT                  # onset in the second stand
    a = np.zeros((T, 18))
    a[56, 9] = -0.5
    e1 = episode(stand, br, a, c, P)
    e2 = episode(np.zeros(T, dtype=bool), br, a, c, P)        # no stand rows at all
    acc = acc_of(OB, P, [e1, e2])
    assert OB.check_reasons(acc) == []
    out = OB.summarize(acc, 0.02, P)
    cn = out["counts"]
    assert (cn["episodes"], cn["episodes_with_stand"], cn["stands"], cn["stand_rows"]) == (2, 1, 2, 40)
    assert cn["lift_onsets"] == 1 and cn["entry_lifts"] == 1
    on = out["launch"]["onset"]["per_leg"]["MR"]
    assert (on["events"], on["launched"]) == (1, 1) and on["latency_median_s"] == pytest.approx(0.02)
    pooled = OB.summarize(OB.Acc.pool([acc, acc_of(OB, P, [e1])]), 0.02, P)
    assert pooled["counts"]["stands"] == 4 and pooled["counts"]["lift_onsets"] == 2
    assert pooled["counts"]["episodes"] == 3 and pooled["counts"]["stand_rows"] == 80


# ===================================================================================== self-checks
def test_self_checks_refuse(OB, P):
    d, stand, c, a = hold_episode(P)
    ok = acc_of(OB, P, [d])
    assert OB.check_reasons(ok) == [] and ok.checks["c_align_rows"] > 0
    bad = dict(d)
    bad["actions"] = d["actions"].copy()
    bad["actions"][stand.nonzero()[0][3], 2] += 0.5          # S1 (not the stand net) acted on a stand row
    assert any("routed action" in r for r in OB.check_reasons(acc_of(OB, P, [bad])))
    bad = dict(d)
    bad["joint_pos"] = d["joint_pos"].copy()
    bad["joint_pos"][:, CAM_COLS] = np.roll(d["joint_pos"][:, CAM_COLS], 1, axis=0)   # c one row off
    assert any("not R's input" in r for r in OB.check_reasons(acc_of(OB, P, [bad])))
    bad = dict(d)
    bad["sn_r_branch"] = d["sn_r_branch"].copy()
    bad["sn_r_branch"][0, 0] = HOLD                         # a branch on an off-stand row
    assert any("exactly the stand rows" in r for r in OB.check_reasons(acc_of(OB, P, [bad])))
    T = 50
    st = np.ones(T, dtype=bool)
    br = np.full((T, 6), HOLD)
    br[5:8, 0] = LIFT
    tgt = r_like_target(br, np.zeros((T, 6)), P)
    tgt[5:8, 6] = -0.3                                     # R's LIFT hip target is not lift
    e = episode(st, br, np.zeros((T, 18)), np.zeros((T, 6)), P, tgt=tgt)
    assert any("LIFT leg-rows is not lift" in r for r in OB.check_reasons(acc_of(OB, P, [e])))


def test_cam_wrap(OB, P):
    d, *_ = hold_episode(P)
    d2, *_ = hold_episode(P)
    d2 = episode(d2["p2r_stand_mode"], d2["sn_r_branch"], d2["sn_stand_action"],
                 d["joint_pos"][:, CAM_COLS].astype(np.float64), P, tgt=d["sn_r_target"], cam_offset=2 * math.pi)
    acc = acc_of(OB, P, [d2])
    assert OB.check_reasons(acc) == []                      # c + 2 pi wraps back to R's c


def test_real_restance_labels_pass_and_perfect_imitation_reads_zero(OB, P):
    """Raws labelled by the REAL restance_target (D3) from synthetic obs: the read's c / branch / LIFT checks pass,
    and a stand net that executes R exactly reads |applied - R| = 0, mu_hip = lift on LIFT rows, launches every
    onset at latency 0, and its HOLD slope equals R's own slope on the same rows."""
    rst = OB.rst()
    assert rst.format_restance(P) == D3
    g = torch.Generator().manual_seed(3)
    T = 600
    obs = torch.zeros(T, 89)
    # slowly varying cams (so branches persist) and hips around default / lift, level body
    t = torch.arange(T, dtype=torch.float32)[:, None]
    phase = torch.rand(1, 6, generator=g) * 6.28
    obs[:, [18 + 2 * i for i in range(6)]] = 0.45 * torch.sin(t / 37.0 + phase)
    obs[:, 29:35] = torch.where(torch.sin(t / 23.0 + phase) > 0.6, torch.full((T, 6), -0.11), torch.zeros(T, 6))
    obs[:, 3:5] = 0.01 * torch.randn(T, 2, generator=g)
    tgt, br = rst.restance_target(obs, P)
    tgt, br = tgt.numpy(), br.numpy()
    assert {HOLD, WAIT, LIFT}.issubset(set(np.unique(br).tolist()))
    c = obs[:, [18 + 2 * i for i in range(6)]].numpy().astype(np.float64)
    stand = np.ones(T, dtype=bool)
    d = episode(stand, br, tgt, c, P, tgt=tgt)
    acc = acc_of(OB, P, [d])
    assert OB.check_reasons(acc) == []
    out = OB.summarize(acc, 0.02, P)
    for nm, blk in out["by_branch"].items():
        for gname in OB.GROUPS:
            if blk[gname] is not None:
                assert blk[gname]["mean_abs"] == pytest.approx(0.0, abs=1e-6), (nm, gname)
    assert out["lift_hip"]["lift"]["mu_hip"] == pytest.approx(P.lift)
    assert out["lift_hip"]["lift"]["share_below_thr"] == pytest.approx(1.0)
    po = out["launch"]["onset"]["pooled"]
    assert po["events"] > 0 and po["launch_share"] == pytest.approx(1.0) and po["latency_median_s"] == 0.0
    for leg in OB.LEGS:
        f = out["hold_slope"]["per_leg"][leg]
        if f["slope"] is not None:
            assert f["slope"] == pytest.approx(f["r_slope_same_rows"], abs=1e-5)


# ===================================================================================== CLI end to end
ENV = {"KRABBY_OMNI_STAND_OBS": "0", "KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft",
       "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25", "KRABBY_OMNI_FACING_OBS_CLIP": "0.6",
       "KRABBY_OMNI_STAND_HOLD": "0", "KRABBY_OMNI_STAND_PARK": "0"}


def run_meta(*, variant="stand_net", mask=1, sha=SHA_A, s1=SHA_S1, r_knob=D3, env=None, raw_series=None):
    return {"dt": 0.02, "joint_names": JOINT_NAMES, "env_vars": dict(ENV if env is None else env),
            "p2r_mode": {"variant": variant, "stand_mask": mask, "specialist": {"path": "/x/model.pt", "sha256": sha},
                         "s1_checkpoint_sha256": s1,
                         "stand_net": {"r_knob": r_knob, "stand_ckpt_kind": "sn_dagger", "trained_stand_mask": mask},
                         "raw_series": (["p2r_stand_mode", "sn_stand_action", "sn_s1_action", "sn_r_target",
                                         "sn_r_branch"] if raw_series is None else raw_series)}}


def write_stamp(lr: Path, head: str, row: str, seed: int, P, *, ts="2026-10-12_10-00-00", meta=None, complete=True,
                eps=None, drop_key=None):
    d = lr / head / row / f"seed{seed:03d}" / ts
    (d / "raw").mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps(run_meta() if meta is None else meta))
    if complete:
        (d / ("complete.json" if row.startswith("suite") else "p2_complete.json")).write_text("{}")
    for k, e in enumerate(eps if eps is not None else [hold_episode(P, seed=seed)[0]]):
        e = {kk: v for kk, v in e.items() if kk != drop_key}
        np.savez(d / "raw" / f"episode_{k:02d}.npz", **e)
    return d


def _files(root: Path) -> set:
    return {p for p in root.rglob("*") if p.is_file()}


def test_cli_partial_then_complete(OB, P, tmp_path):
    lr = tmp_path / "lr"
    head = "SN_T_s18"
    write_stamp(lr, head, "walkstop__omni", 1, P, ts="2026-10-12_09-00-00")
    write_stamp(lr, head, "walkstop__omni", 1, P, ts="2026-10-12_10-00-00")             # newest complete: read
    write_stamp(lr, head, "walkstop__omni", 2, P, ts="2026-10-12_11-00-00", complete=False)   # in progress
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    before = _files(tmp_path)
    j, m = out_dir / "o.json", out_dir / "o.md"
    rc = OB.main(["--head", head, "--lr", str(lr), "--json", str(j), "--md", str(m)])
    assert rc == OB.RC_PENDING
    assert _files(tmp_path) - before == {j, m}                                          # nothing else written
    o = json.loads(j.read_text())
    assert o["status"] == "partial"
    assert o["rows"]["walkstop@1"]["status"] == "READ"
    assert o["rows"]["walkstop@1"]["stamp"].endswith("2026-10-12_10-00-00")
    assert o["rows"]["walkstop@2"]["status"] == "PENDING"
    assert o["rows"]["walkstop@2"]["in_progress"][0].endswith("2026-10-12_11-00-00")
    assert any("in progress" in p for p in o["pending"])
    assert o["rows"]["suite@1"]["status"] == o["rows"]["suite@2"]["status"] == "PENDING"
    assert o["pooled"]["all"]["rows_included"] == ["walkstop@1"] and "suite" not in o["pooled"]
    assert o["arm"]["specialist_sha256"] == SHA_A and o["arm"]["stand_mask"] == 1
    assert o["r_knob"] == D3
    assert "Observe pass" in m.read_text() and "μ_hip LIFT" in m.read_text()
    # complete the other three rows -> exit 0, families pooled
    (lr / head / "walkstop__omni" / "seed002" / "2026-10-12_11-00-00" / "p2_complete.json").write_text("{}")
    write_stamp(lr, head, "suite__omni", 1, P)
    write_stamp(lr, head, "suite__omni", 2, P)
    rc = OB.main(["--head", head, "--lr", str(lr), "--json", str(j), "--md", str(m)])
    assert rc == OB.RC_OK
    o = json.loads(j.read_text())
    assert o["status"] == "complete" and not o["pending"] and not o["invalid"]
    assert o["pooled"]["walkstop"]["rows_included"] == ["walkstop@1", "walkstop@2"]
    assert o["pooled"]["suite"]["rows_included"] == ["suite@1", "suite@2"]
    assert o["pooled"]["all"]["counts"]["stand_rows"] == sum(o["rows"][k]["counts"]["stand_rows"]
                                                             for k in OB.ROWS_DEFAULT)
    # --rows narrows the read
    rc = OB.main(["--head", head, "--lr", str(lr), "--rows", "suite@2", "--json", str(j), "--md", str(m)])
    assert rc == OB.RC_OK and list(json.loads(j.read_text())["rows"]) == ["suite@2"]


@pytest.mark.parametrize("meta_kw, why", [
    ({"variant": "st2"}, "variant"),
    ({"mask": True}, "int 0 / 1"),
    ({"sha": "xyz"}, "malformed"),
    ({"r_knob": "default"}, "D3"),
    ({"env": {**ENV, "KRABBY_OMNI_STAND_OBS": "clock"}}, "KRABBY_OMNI_STAND_OBS"),
    ({"raw_series": ["p2r_stand_mode"]}, "raw_series"),
])
def test_cli_invalid_meta(OB, P, tmp_path, meta_kw, why):
    lr = tmp_path / "lr"
    write_stamp(lr, "H", "walkstop__omni", 1, P, meta=run_meta(**meta_kw))
    j, m = tmp_path / "o.json", tmp_path / "o.md"
    rc = OB.main(["--head", "H", "--lr", str(lr), "--rows", "walkstop@1", "--json", str(j), "--md", str(m)])
    assert rc == OB.RC_INVALID
    o = json.loads(j.read_text())
    assert o["status"] == "refused" and o["rows"]["walkstop@1"]["status"] == "INVALID"
    assert any(why in r for r in o["invalid"]["walkstop@1"]) and "all" not in o["pooled"]


def test_cli_mixed_arm_and_missing_raw_key(OB, P, tmp_path):
    lr = tmp_path / "lr"
    write_stamp(lr, "H", "walkstop__omni", 1, P, meta=run_meta(sha=SHA_A))
    write_stamp(lr, "H", "walkstop__omni", 2, P, meta=run_meta(sha=SHA_B))
    j, m = tmp_path / "o.json", tmp_path / "o.md"
    rc = OB.main(["--head", "H", "--lr", str(lr), "--rows", "walkstop@1", "walkstop@2", "--json", str(j),
                  "--md", str(m)])
    assert rc == OB.RC_INVALID
    o = json.loads(j.read_text())
    assert set(o["invalid"]) == {"walkstop@1", "walkstop@2"}
    assert all(any("mix specialist_sha256" in r for r in v) for v in o["invalid"].values())
    lr2 = tmp_path / "lr2"
    write_stamp(lr2, "H", "walkstop__omni", 1, P, drop_key="sn_r_branch")
    rc = OB.main(["--head", "H", "--lr", str(lr2), "--rows", "walkstop@1", "--json", str(j), "--md", str(m)])
    assert rc == OB.RC_INVALID
    assert any("lacks raw key" in r for r in json.loads(j.read_text())["invalid"]["walkstop@1"])


def test_cli_usage(OB, tmp_path):
    j, m = tmp_path / "o.json", tmp_path / "o.md"
    assert OB.main(["--head", "H", "--lr", str(tmp_path), "--rows", "field_hard@1", "--json", str(j),
                    "--md", str(m)]) == OB.RC_USAGE
    assert OB.main(["--head", "H", "--lr", str(tmp_path), "--json", str(tmp_path / "nope" / "o.json"),
                    "--md", str(m)]) == OB.RC_USAGE
    with pytest.raises(SystemExit):
        OB.main(["--head", "H", "--md", str(m)])                       # --json is required (no campaign default)
    assert not j.exists() and not m.exists()
