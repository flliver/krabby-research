"""Fault-injection sweep reader (2026-11-15): the pure functions of ``scripts/fault_read.py`` on CPU.

``experiments/2026-11-15_fault_injection/CHARTER.md`` §5 is the spec. Pinned here without Isaac or any stamp:

* the §5.3 verdict rules in order (FALLS before everything; TOLERATED needs lines + ΔF <= 0.03 + R; RECOVERS only for a
  transient; R report-only -> R not required) and the marginal mark (within 2 episodes of a ΔF threshold);
* the seed combination (the worse; split; FALLS on es1 alone at F >= 0.5) and the tolerated duration (monotone: the
  first failing level stops the family, whatever longer levels read);
* the R windows ([onset + 2 s, onset + 4 s) permanent, [clear + 2 s, clear + 4 s) transient), the in-hold and alive
  rules, a fall inside the window = not tracking;
* facing_closed re-anchored at onset (the 0.15 floor, exclusion, clip) and the harness window it is cut from;
* fall bookkeeping against the onset, the tracking ratio's sub-clip / short-window None.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-11-15_fault_injection"
          / "scripts/fault_read.py")


@pytest.fixture(scope="module")
def fr():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("fault_read_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fault_read_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- verdict rules
def test_verdict_rule_order(fr):
    v = fr.verdict
    # 1. FALLS beats everything, even passing lines and R
    assert v(0.11, True, 1.0, True, True) == "FALLS"
    assert v(0.10, True, 1.0, True, False) == "DEGRADES"          # 0.10 is not > 0.10, but > 0.03
    assert v(0.10, False, 0.0, True, True) == "DEGRADES"
    # 2. TOLERATED: lines, ΔF <= 0.03, R >= 0.9
    assert v(0.03, True, 0.90, True, False) == "TOLERATED"
    assert v(0.031, True, 1.0, True, False) == "DEGRADES"          # permanent: no RECOVERS rung
    assert v(0.0, False, 1.0, True, False) == "DEGRADES"           # a failed line
    assert v(0.0, True, 0.89, True, False) == "DEGRADES"           # R short
    assert v(0.0, True, None, False, False) == "TOLERATED"         # R report-only -> not required
    assert v(0.0, True, None, True, False) == "DEGRADES"           # R required but unmeasured
    # 3. RECOVERS: transient only, ΔF <= 0.05, R >= 0.9 (lines may fail)
    assert v(0.05, False, 0.95, True, True) == "RECOVERS"
    assert v(0.04, True, 0.95, True, True) == "RECOVERS"
    assert v(0.051, True, 0.95, True, True) == "DEGRADES"
    assert v(0.02, False, 0.85, True, True) == "DEGRADES"
    assert v(0.02, False, None, False, True) == "RECOVERS"
    # a negative ΔF (fewer falls than the reference) is not a failure
    assert v(-0.05, True, 1.0, True, False) == "TOLERATED"


def test_marginal_within_two_episodes(fr):
    m = fr.marginal
    assert m(0.0, 100, False) == []
    assert m(0.01, 100, False) == [0.03]                           # 1 vs 3 episodes
    assert m(0.05, 100, False) == [0.03]                           # 5 vs 3; the 0.05 rung is transient-only
    assert m(0.05, 100, True) == [0.03, 0.05]
    assert m(0.12, 100, True) == [0.1]
    assert m(0.13, 100, True) == []
    assert m(0.01, 50, False) == [0.03]                            # 0.5 vs 1.5 episodes at n = 50
    assert m(0.1, 0, True) == []


def test_worse_and_combine_seeds(fr):
    assert fr.worse("TOLERATED", "RECOVERS") == "RECOVERS"
    assert fr.worse("FALLS", "TOLERATED", "DEGRADES") == "FALLS"
    c = fr.combine_seeds({1: "TOLERATED", 2: "RECOVERS"})
    assert c == {"verdict": "RECOVERS", "split": True, "pending": False}
    assert fr.combine_seeds({1: "DEGRADES", 2: "DEGRADES"})["split"] is False
    # es2 not run: FALLS stands on es1 only when es1's worst-row F >= 0.5 (CHARTER §4)
    assert fr.combine_seeds({1: "FALLS"}, 0.6)["pending"] is False
    assert fr.combine_seeds({1: "FALLS"}, 0.3)["pending"] is True
    assert fr.combine_seeds({1: "TOLERATED"}, 0.0)["pending"] is True
    assert fr.combine_seeds({1: None, 2: None})["verdict"] is None


def test_tolerated_duration_monotone(fr):
    td = fr.tolerated_duration
    lv = [(5, "TOLERATED", False), (25, "RECOVERS", False), (100, "DEGRADES", False)]
    assert td(lv)["tolerated_steps"] == 25 and td(lv)["stopped_by"] == "DEGRADES"
    # a shorter failing level stops the family even if a longer level reads TOLERATED
    lv = [(5, "TOLERATED", False), (25, "FALLS", False), (100, "TOLERATED", False)]
    assert td(lv)["tolerated_steps"] == 5
    # order of the input does not matter
    assert td(list(reversed(lv)))["tolerated_steps"] == 5
    assert td([(5, "DEGRADES", False)])["tolerated_steps"] is None
    assert td([(2, "TOLERATED", False), (5, None, True)]) == {"tolerated_steps": 2, "tolerated_s": 2 * fr.DT,
                                                               "stopped_by": "pending"}
    every = [(d, "TOLERATED", False) for d in (2, 5, 10, 25)]
    assert td(every)["tolerated_steps"] == 25 and td(every)["stopped_by"] is None
    # monotone: lengthening the tolerated prefix never shortens the answer
    for k in range(5):
        lv = [(d, "TOLERATED" if i < k else "DEGRADES", False) for i, d in enumerate((2, 5, 10, 25))]
        got = td(lv)["tolerated_steps"]
        assert got == ((2, 5, 10, 25)[k - 1] if k else None)


# ---------------------------------------------------------------------------------------------- R windows
def test_r_window(fr):
    assert fr.r_window(400, None, True) == (500, 600)               # permanent: onset + 2 s .. + 4 s
    assert fr.r_window(400, 425, False) == (525, 625)               # transient: clear + 2 s .. + 4 s
    with pytest.raises(TypeError):
        fr.r_window(400, None, False)                               # a transient needs its clear step


def test_r_episode_rules(fr):
    T = 800
    sp = np.full(T, 0.25)
    vx = np.full(T, 0.25)
    ext = (175, 775)
    e = fr.r_episode(vx, sp, T, "schedule_complete", 500, 600, ext)
    assert e == {"inside": True, "alive": True, "ok": True}
    # window off the end of the hold: not inside (excluded, counted)
    e = fr.r_episode(vx, sp, T, "schedule_complete", 700, 800, ext)
    assert e["inside"] is False and e["ok"] is None
    # fell before the window: not alive (counted in F, not in R)
    e = fr.r_episode(vx, sp, 450, "fall", 500, 600, ext)
    assert e["alive"] is False and e["ok"] is None
    # fell inside the window: alive at its start but not tracking
    e = fr.r_episode(vx, sp, 550, "fall", 500, 600, ext)
    assert e["alive"] is True and e["ok"] is False
    # tracking tolerance 0.25
    vx2 = vx.copy()
    vx2[500:600] = 0.25 * 0.74
    assert fr.r_episode(vx2, sp, T, "schedule_complete", 500, 600, ext)["ok"] is False
    vx2[500:600] = 0.25 * 0.76
    assert fr.r_episode(vx2, sp, T, "schedule_complete", 500, 600, ext)["ok"] is True
    R, n = fr.r_share([{"inside": True, "alive": True, "ok": True}, {"inside": True, "alive": True, "ok": False},
                       {"inside": False, "alive": True, "ok": None}, {"inside": True, "alive": False, "ok": None}])
    assert (R, n) == (0.5, 2)
    assert fr.r_share([]) == (None, 0)


def test_field_translate_extents(fr):
    labels = ["stand", "fwd", "lat_l", "diag", "back", "turn_l90"]
    seg = np.concatenate([np.full(75, -1), np.full(400, 0), np.full(400, 1), np.full(400, 2), np.full(400, 3),
                          np.full(400, 4), np.full(600, 5)])
    ex = fr.field_translate_extents(seg, labels)
    assert ex == [("fwd", 475, 875), ("lat_l", 875, 1275), ("diag", 1275, 1675), ("back", 1675, 2075)]


# ---------------------------------------------------------------------------------------------- facing_closed
def test_facing_reanchored(fr):
    err = np.linspace(1.0, 0.1, 100)
    idx = np.arange(40, 100)
    v, exc = fr.facing_reanchored(err, idx)
    e0, e1 = err[40], err[99]
    assert not exc and v == pytest.approx(1.0 - e1 / e0)
    # under the floor at onset: excluded (None), never a number
    small = np.full(100, 0.1)
    assert fr.facing_reanchored(small, idx) == (None, True)
    # exactly at the floor: measured, divided by the floor
    flo = np.full(100, 0.15)
    flo[-1] = 0.0
    assert fr.facing_reanchored(flo, idx) == (pytest.approx(1.0), False)
    # error opened: clipped at -1
    opened = np.concatenate([np.full(50, 0.2), np.full(50, 3.0)])
    assert fr.facing_reanchored(opened, idx)[0] == -1.0
    # sign does not matter (|err|)
    assert fr.facing_reanchored(-err, idx)[0] == pytest.approx(v)
    assert fr.facing_reanchored(err, np.arange(0))[0] is None


def test_window_idx_matches_harness_mask(fr):
    seg = np.concatenate([np.full(75, -1), np.full(100, 0), np.full(600, 7)])
    steady = np.ones(775, bool)
    steady[175:225] = False
    idx = fr.window_idx(seg, steady, 7, 775, False, 0)
    assert idx[0] == 225 and idx[-1] == 774 and idx.size == 550
    idx = fr.window_idx(seg, steady, 7, 775, False, 400)
    assert idx[0] == 400 and idx[-1] == 774
    # a fall trims the last 25 steps (fall_exclusion_s 0.5), the facing window ignores steady
    idx = fr.window_idx(seg, None, 7, 600, True, 400)
    assert idx[0] == 400 and idx[-1] == 574
    assert fr.window_idx(seg, None, 7, 420, True, 400).size == 0


def test_fall_info(fr):
    f = fr.fall_info(775, "schedule_complete", 400)
    assert f["alive_at_onset"] and not f["post_fall"] and not f["pre_fall"] and f["exposure_steps"] == 375
    f = fr.fall_info(400, "fall", 400)                              # last step 399 < onset: pre-onset fall
    assert f["pre_fall"] and not f["alive_at_onset"] and f["exposure_steps"] == 0
    f = fr.fall_info(401, "fall", 400)                              # fell on the onset step: post-onset, F2
    assert f["post_fall"] and f["f2"] and f["alive_at_onset"]
    f = fr.fall_info(501, "fall", 400)                              # fall step 500 = onset + 100: not F2
    assert f["post_fall"] and not f["f2"]


def test_alive_at_onset_needs_the_onset_reached(fr):
    """A completed episode shorter than its onset (armed-but-never smoke) never reached it: not alive, no exposure."""
    f = fr.fall_info(775, "schedule_complete", 150000)
    assert not f["alive_at_onset"] and not f["post_fall"] and not f["pre_fall"] and f["exposure_steps"] == 0
    assert fr.fall_info(775, "schedule_complete", 774)["alive_at_onset"]           # last step 774 = onset: reached
    assert not fr.fall_info(775, "schedule_complete", 775)["alive_at_onset"]       # onset one past the last step
    # R: a window starting beyond the episode has no alive episode, whatever the termination
    vx = sp = np.ones(775)
    e = fr.r_episode(vx, sp, 775, "schedule_complete", 150000, 150100, (0, 10 ** 6))
    assert e["inside"] and not e["alive"] and e["ok"] is None
    assert fr.r_share([e]) == (None, 0)


class _StubLaunch:
    def __init__(self, n_steps):
        self._eps = [{"idx": i, "wave": 0, "env": i, "n": n, "reason": "schedule_complete"}
                     for i, n in enumerate(n_steps)]

    def episodes(self):
        return self._eps


def _s1_sidecar(fr, onset, first_active, n_envs=4):
    return {"schema": fr.fault_inject().SIDECAR_SCHEMA, "fault": {"id": "S1", "from_start": False}, "kind": "S1",
            "duration_steps": None, "num_envs": n_envs, "env_seed": 1, "onset": {"window_s": [3000.0, 3001.0]},
            "waves": [{"wave": 0, "onset": onset, "start": onset, "planned_last": None, "state_cleared": True,
                       "fell_at": [-1] * n_envs, "first_active": first_active, "last_active": first_active,
                       "probes": {}}]}


def test_sidecar_armed_but_never_counts_no_alive_env(fr):
    """Onsets (~150000 steps) beyond a 775-step episode: nobody is alive at start, and any activity is an error."""
    cell = {"id": "SMK_never", "fault": "S1:perm", "smoke": True}
    n = 4
    tab = fr.onset_table_np(3000.0, 3001.0, n, 1, 1)
    onset = tab[0].tolist()
    assert min(onset) > 100000
    L = _StubLaunch([775] * n)
    chk = fr.sidecar_checks(L, cell, _s1_sidecar(fr, onset, [-1] * n), 1, n, 1, (3000.0, 3001.0))
    assert chk["errors"] == [] and chk["report"]["n_alive_at_start"] == 0 and chk["report"]["n_active"] == 0
    # the injector reporting activity on an env whose start was never reached is a launch error
    chk = fr.sidecar_checks(L, cell, _s1_sidecar(fr, onset, [-1, onset[1], -1, -1]), 1, n, 1, (3000.0, 3001.0))
    assert any("start is beyond the episode" in e for e in chk["errors"])
    assert chk["report"]["n_active"] == 1
    # control: onsets inside the episode are alive and must activate exactly at the start
    inside = [100, 200, 300, 400]
    chk = fr.sidecar_checks(L, cell, _s1_sidecar(fr, inside, inside), 1, n, 1, (3000.0, 3001.0))
    assert chk["report"]["n_alive_at_start"] == 4
    assert not any("first_active" in e for e in chk["errors"])


def test_ratio_of(fr):
    vx, sp = np.full(200, 0.2), np.full(200, 0.25)
    assert fr.ratio_of(vx, sp, np.arange(0, 100)) == pytest.approx(0.8)
    assert fr.ratio_of(vx, sp, np.arange(0, 49)) is None            # < min_window_s
    assert fr.ratio_of(vx, np.full(200, 0.2), np.arange(0, 100)) is None   # cmd <= lin_vel_clip


def test_roll_pitch_identity_and_tilt(fr):
    q = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    r, p = fr.roll_pitch_rms(q, np.zeros(10), np.arange(10))
    assert r == 0.0 and p == 0.0
    a = 0.1                                                         # roll 0.1 rad about body x
    q = np.tile([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0], (10, 1))
    r, p = fr.roll_pitch_rms(q, np.zeros(10), np.arange(10))
    assert r == pytest.approx(0.1) and p == pytest.approx(0.0, abs=1e-12)
    # travel frame at +pi/2: the body roll is a pitch of the travel frame
    r2, p2 = fr.roll_pitch_rms(q, np.full(10, np.pi / 2), np.arange(10))
    assert r2 == pytest.approx(0.0, abs=1e-9) and abs(p2) == pytest.approx(0.1)


def test_probe_kind_labels_from_injector(fr):
    pytest.importorskip("torch")
    fi = fr.fault_inject()
    assert fi.probe_kind("a2_gain_post") == "sim"
    assert fi.probe_kind("a2_applied_torque_absmax") == "model_estimate"
    tab = fr.onset_table_np(7.5, 11.5, 100, 1, 10)
    assert tab.shape == (10, 100) and tab.min() >= 375 and tab.max() < 575


# ---------------------------------------------------------------------- physics drift (CHARTER 5.1, 2026-11-16)
def _npz(path, n=40, actions_flip=None, contact_flip=None, extra_flip=None):
    rng = np.random.default_rng(0)
    d = {"actions": rng.normal(size=(n, 3)).astype(np.float32),
         "leg_contact_force": rng.normal(size=(n, 2)).astype(np.float32),
         "root_pos_w": rng.normal(size=(n, 3)).astype(np.float32)}
    for key, step in (("actions", actions_flip), ("leg_contact_force", contact_flip), ("root_pos_w", extra_flip)):
        if step is not None:
            d[key] = d[key].copy()
            d[key][step:, 0] += np.float32(6e-5)
    np.savez(path, **d)
    return path


def _fd(fr, tmp_path, **kw):
    x = _npz(tmp_path / "x.npz")
    y = _npz(tmp_path / "y.npz", **kw)
    return fr.first_diff({"x": str(x), "y": str(y), "idx": 0})


def test_first_diff_reports_the_actions_step(fr, tmp_path):
    d = _fd(fr, tmp_path, contact_flip=30)
    assert d["step"] == 30 and d["keys"] == ["leg_contact_force"] and d["act_step"] is None
    d = _fd(fr, tmp_path, contact_flip=30, actions_flip=35)
    assert d["step"] == 30 and d["act_step"] == 35 and fr.is_physics_drift(d)
    d = _fd(fr, tmp_path, contact_flip=30, actions_flip=30)               # same step: actions are not "still equal"
    assert d["step"] == 30 and d["act_step"] == 30 and not fr.is_physics_drift(d)
    d = _fd(fr, tmp_path, contact_flip=30, actions_flip=20)               # actions first
    assert d["step"] == 20 and d["act_step"] == 20 and not fr.is_physics_drift(d)
    d = _fd(fr, tmp_path)                                                 # identical
    assert d["step"] is None and not fr.is_physics_drift(d)


def test_is_physics_drift_struct_never(fr):
    assert not fr.is_physics_drift({"step": 0, "keys": ["x"], "struct": True})
    assert fr.is_physics_drift({"step": 7, "keys": ["leg_contact_force"], "struct": False, "act_step": None})
    assert not fr.is_physics_drift({"step": 7, "keys": ["actions"], "struct": False, "act_step": 7})


class _IdLaunch:
    def __init__(self, n_envs):
        self._eps = [{"idx": i, "wave": 0, "env": i, "n": 700, "reason": "schedule_complete"} for i in range(n_envs)]

    def episodes(self):
        return self._eps


def _d(idx, step, act_step=None, keys=("leg_contact_force",)):
    return {"idx": idx, "step": step, "keys": list(keys), "struct": False, "act_step": act_step}


def test_grid_onset_horizon_from_cells_yaml(fr):
    rows = fr.load_cells()["rows"]
    assert fr.grid_onset_horizon(rows["suite"]) == 574
    assert fr.grid_onset_horizon(rows["field_light"]) == 2999
    assert fr.grid_onset_horizon(rows["ledge_down"]) == 549
    assert fr.grid_onset_horizon(rows["pits"]) == 669           # the pits grid draws onsets up to 13.4 s


def test_identity_read_drift_vs_break(fr):
    n = 6
    onsets = np.full((1, n), 400)
    L = _IdLaunch(n)
    diffs = {i: _d(i, None) for i in range(n)}
    diffs[1] = _d(1, 100)                                    # equal actions through step 100: drift
    diffs[2] = _d(2, 100, act_step=100)                      # actions at the same step: break
    diffs[3] = _d(3, 90, act_step=80)                        # actions first: break
    diffs[4] = _d(4, 450)                                    # post-onset: neither
    r = fr.identity_read(L, diffs, onsets)
    assert r["n_physics_drift"] == 1 and r["drift_idx"] == [1]
    assert r["n_breaks"] == 2 and sorted(b["idx"] for b in r["breaks"]) == [2, 3]
    assert not r["ok"]
    # drift alone is ok up to 3 envs, a launch error beyond
    for k, ok in ((3, True), (4, False)):
        dd = {i: _d(i, None) for i in range(n)}
        for i in range(k):
            dd[i] = _d(i, 50 + i)
        r = fr.identity_read(L, dd, onsets)
        assert r["n_physics_drift"] == k and r["n_breaks"] == 0 and r["ok"] is ok
    # a structural difference is always a break
    dd = {i: _d(i, None) for i in range(n)}
    dd[0] = {"idx": 0, "step": 0, "keys": ["k"], "struct": True}
    r = fr.identity_read(L, dd, onsets)
    assert r["n_breaks"] == 1 and r["n_physics_drift"] == 0


def test_identity_read_armed_never_horizon(fr):
    """Onsets far beyond the episode: the horizon caps the identity check (the pits es1 env 81 case)."""
    n = 3
    onsets = np.full((1, n), 150000)
    L = _IdLaunch(n)
    diffs = {0: _d(0, 2996), 1: _d(1, 600, act_step=600), 2: _d(2, None)}
    # uncapped: step 2996 is "pre-onset" drift, 600 a break
    r = fr.identity_read(L, diffs, onsets)
    assert r["n_physics_drift"] == 1 and r["n_breaks"] == 1
    # field_light horizon 2999: step 2996 is still inside what the grid rows need
    r = fr.identity_read(L, diffs, onsets, horizon=2999)
    assert r["n_physics_drift"] == 1 and r["n_breaks"] == 1
    # kind row horizon 669: 2996 lies beyond what the grid rows need; 600 is inside and an actions break
    r = fr.identity_read(L, diffs, onsets, horizon=669)
    assert r["n_physics_drift"] == 0 and r["n_breaks"] == 1 and r["horizon"] == 669
    # suite horizon 574: both beyond
    r = fr.identity_read(L, diffs, onsets, horizon=574)
    assert r["n_physics_drift"] == 0 and r["n_breaks"] == 0 and r["ok"]
    # a real onset earlier than the cap still wins
    r = fr.identity_read(L, {0: _d(0, 100, act_step=100), 1: _d(1, None), 2: _d(2, None)},
                         np.array([[50, 150000, 150000]]), horizon=574)
    assert r["n_breaks"] == 0                                # step 100 is post-onset (50) for env 0


def test_dose_pairing_excludes_drift_envs(fr, tmp_path):
    """Drift envs leave the paired-divergence check (n, 'before onset')."""
    # B6-less minimal path: kind A1 with no npz jobs would need the pool; check the exclusion arithmetic directly
    class _R:
        def pool(self):
            class P:
                def map(self, f, jobs, chunksize=1):
                    return []
            return P()

    class _L(_IdLaunch):
        run_meta = {}
        stamp = tmp_path

        def npz(self, i):
            return tmp_path / f"{i}.npz"

    L = _L(4)
    for e in L._eps:
        e["n"] = 700
    onsets = np.full((1, 4), 300)
    sc = {"kind": "A1", "binding": {}, "waves": [{"wave": 0, "planned_last": None}]}
    diffs = {0: _d(0, 100), 1: _d(1, 400), 2: _d(2, 410), 3: _d(3, 420)}
    r = fr.npz_dose_checks(_R(), L, {}, sc, onsets, diffs)
    assert any("diverge before the onset" in e for e in r["errors"])
    r = fr.npz_dose_checks(_R(), L, {}, sc, onsets, diffs, exclude={0})
    assert not any("diverge before the onset" in e for e in r["errors"])
    assert r["report"]["paired_divergence"]["n"] == 3 and r["report"]["paired_divergence"]["physics_drift_excluded"] == 1
