# SPDX-License-Identifier: BSD-3-Clause
"""Stage p2r Z2 (CHARTER 12 / P2R_Z2_PLAN.md step 1 "Reads"): the READS + LIVE WIRING half. CPU only -- nothing here
starts Isaac/Kit or a GPU, and nothing reads a real eval stamp (the real baselines are computed by hand, read-only).

* ``scripts/p2s_stand.py``: :func:`iter_stand_holds` over EVERY "stand" hold (walkstop's three; the first-stand
  :func:`stand_hold_index` unchanged), the per-stand Z2 estimators (:func:`stand_z2_metrics`: homed over the last
  5 s, six-down share on the steady / last-5-s window, settle time (HOMED held 0.5 s: since the 2026-10-10
  cams-only amendment six-down is not part of settle), post-settle stillness from each stand's own
  settle, re-stance lifts, body height, hip yaw) on hand-built episodes incl. settled / unsettled stands, and
  :func:`z2_head_summary` (shares, fail-closed post-settle medians, falls per 300, the per-stand breakdown);
* ``scripts/p2_read.py``: the Z2 stand gates (each clause, PENDING vs FAIL, the relative settle line; six-down
  REPORTED, never a clause -- cams-only, 2026-10-10), the premise
  gate's field rows vs HOLD (HOLD's own 2 SE) and body height vs REFS, the probe INVALID rule (env -- STAND_OBS=clock
  and STAND_PROXY OFF --, p2r_mode, raw bit), the per-seed / NOT RUN verdict logic, the Z2 INVALID widening (a
  stamp with the held proxy armed), ``adoption_grade``, the Z2 suite stamp (the driver's), the bridged
  code_sha CHAIN (a new pair appended to inertness_bridge.json links the new code to ae892cec), and ``--p2r-rst``
  end to end on a fabricated ``$OUT`` tree with real raw npz;
* ``scripts/p2_live.py``: the Z2 +200 wiring (Z1's checks, the dose band from p2r_z2_dose.json -- refused without
  it --, restance_nonzero, every branch with home > 0; the STAND_PROXY knob OFF; no stand_proxy_live_frac clause
  since the cams-only amendment), Z1 untouched, the Z2 preflight,
  the relaunch-name dispatch in ``check()``, and the branch names pinned against the restance module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = OMNI / "experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
DT = 0.02
N_FEET = 6
SC = ("c" * 64, "e" * 64, False)
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
#: NOTE(p2r Z2 cams-only, 2026-10-10): a Z2-family stamp's env -- the stand bit, the held proxy OFF (absent)
ENV_Z2 = dict(ENV4, KRABBY_OMNI_STAND_OBS="clock")
FH, FT, WS, SUITE = "field_hard__omni", "field_turn__omni", "walkstop__omni", "suite__omni"
REFS_SHA = "a" * 64
#: the 24 articulation joints in run_meta order (cams at 1, 3, ..., 11; hip yaw at 0, 2, ..., 10)
JOINT_NAMES = [n for leg in ("FL", "FR", "ML", "MR", "RL", "RR")
               for n in (f"{leg}_Body_Hip_RevoluteJoint", f"{leg}_Body_CamShaft_RevoluteJoint")] + \
              [f"{leg}_Hip_Femur_RevoluteJoint" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")] + \
              [f"{leg}_Femur_Tibia_RevoluteJoint" for leg in ("FL", "FR", "ML", "MR", "RL", "RR")]
CAM_COLS = [1, 3, 5, 7, 9, 11]
HIP_COLS = [0, 2, 4, 6, 8, 10]
_N = [0]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def P():
    return _load(SCRIPTS / "p2s_stand.py", "_t_z2_p2s_stand")


@pytest.fixture(scope="module")
def read():
    return _load(SCRIPTS / "p2_read.py", "_t_z2_read_pure")


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_z2_live")


def _read_at(out: Path, monkeypatch):
    """A fresh p2_read module with $P2_OUT at ``out`` (OUT / ARMS / LROUT are read at import)."""
    monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    return _load(SCRIPTS / "p2_read.py", f"_t_z2_read_{_N[0]}")


# ================================================================================================ synthetic episodes
def _stand_episode(seg: np.ndarray, stand_ids, *, settle_at=None, lift_foot=2, cam0=0.6, crab_failure=None,
                   root_z=1.06, march_after_settle=False) -> dict:
    """One episode over ``seg``: walking rows (every hold not in ``stand_ids``) march (alternating tripods, cams
    swinging ``cam0``); each stand hold starts with ``lift_foot`` unloaded and every cam at ``cam0`` until
    ``settle_at`` rows into the stand, then six feet down and cams at exactly 0 (``settle_at=None``: never --
    the cams stay at ``cam0`` and the foot stays up). ``march_after_settle``: after settling, foot 0 lifts for
    20 rows every 50 rows (post-settle steps)."""
    T = seg.size
    force = np.full((T, N_FEET), 600.0)
    jp = np.zeros((T, 24))
    for t in range(T):
        if seg[t] not in stand_ids:
            force[t, [0, 3, 4] if (t // 25) % 2 else [1, 2, 5]] = 0.0
            jp[t, CAM_COLS] = cam0
    for sid in stand_ids:
        rows = np.flatnonzero(seg == sid)
        k = len(rows) if settle_at is None else settle_at
        pre, post = rows[:k], rows[k:]
        force[pre, lift_foot] = 0.0
        jp[np.ix_(pre, CAM_COLS)] = cam0
        jp[np.ix_(pre, HIP_COLS)] = np.radians(10.0)
        if march_after_settle:
            for j, t in enumerate(post):
                if j % 50 >= 30:
                    force[t, 0] = 0.0
    foot_pos = np.zeros((T, N_FEET, 3))
    foot_pos[:, :, 2] = 0.03
    root = np.zeros((T, 3))
    root[:, 2] = root_z
    return {"foot_force_norm": force, "foot_pos_w": foot_pos, "foot_lin_vel_w": np.zeros((T, N_FEET, 3)),
            "terrain_z": np.zeros((T, N_FEET)), "root_pos_w": root, "root_lin_vel_b": np.zeros((T, 3)),
            "heading_err": np.zeros(T), "segment_id": seg.astype(np.int16),
            "steady_mask": np.ones(T, dtype=bool), "crab_failure": (np.zeros(T, dtype=bool) if crab_failure is None
                                                                    else crab_failure),
            "cmd_applied": np.zeros((T, 3)), "joint_pos": jp, "actions": np.zeros((T, 18))}


def _walkstop_seg(walk=50, stand=400) -> np.ndarray:
    """fwd / stand / lat_l / stand / back / stand, the walkstop schedule (shortened walks)."""
    return np.concatenate([np.full(walk if i % 2 == 0 else stand, i) for i in range(6)])


WS_LABELS = ["fwd", "stand", "lat_l", "stand", "back", "stand"]
WS_SPEEDS = [0.25, 0.0, 0.2, 0.0, 0.2, 0.0]


# ================================================================================================ p2s_stand: iteration
def _write_raw_stamp(root: Path, episodes: list, labels, speeds, *, suite=False, env=None, ckpt=REFS_SHA,
                     p2r_mode=None, comp=None, sc=SC) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    meta = {"hold_labels": labels, "hold_speeds": speeds, "dt": DT, "joint_names": JOINT_NAMES,
            "env_vars": env if env is not None else ENV_Z2, "env_vars_manifest": {}, "checkpoint_sha256": ckpt}
    if suite:
        meta["env_index_map"] = [{"env_index": i, "label": "stand", "category": "stand"} for i in range(len(episodes))]
        meta["suite"] = {"settle_label": "settle"}
    if p2r_mode is not None:
        meta["p2r_mode"] = p2r_mode
    (root / "run_meta.json").write_text(json.dumps(meta))
    stand_lab = {"stand", "settle"}
    for i, d in enumerate(episodes):
        seg = d["segment_id"]
        bit = np.array([1.0 if labels[s] in stand_lab else 0.0 for s in seg])
        np.savez(root / "raw" / f"episode_{i:02d}.npz", clock_obs=np.stack([bit, np.zeros_like(bit)], -1), **d)
    agg = {"n_episodes": len(episodes), "termination_reasons": {"fall": 0}}
    if comp is not None:
        agg["schedule_completion_rate"] = comp
    (root / "scenario_metrics.json").write_text(json.dumps({"aggregate": agg}))
    (root / "code_state.json").write_text(json.dumps(dict(zip(("code_sha", "eval_sha", "dirty"), sc))))
    (root / ("complete.json" if suite else "p2_complete.json")).write_text("{}")
    return root


def test_iter_stand_holds_reads_all_three_walkstop_stands(P, tmp_path):
    seg = _walkstop_seg()
    st = _write_raw_stamp(tmp_path / "ws", [_stand_episode(seg, (1, 3, 5), settle_at=50)] * 2, WS_LABELS, WS_SPEEDS)
    holds = list(P.iter_stand_holds(str(st)))
    assert [(h["env_index"], h["hold_idx"], h["ordinal"]) for h in holds] == \
        [(0, 1, 0), (0, 3, 1), (0, 5, 2), (1, 1, 0), (1, 3, 1), (1, 5, 2)]
    assert P.stand_hold_index(WS_LABELS) == 1                       # the first-stand index is unchanged
    assert P.stand_window_kind(json.loads((st / "run_meta.json").read_text())) == P.WINDOW_LAST


def test_joint_columns_list_and_repr(P):
    assert P.joint_columns({"joint_names": JOINT_NAMES}, P.CAM_JOINT_SUFFIX) == CAM_COLS
    assert P.joint_columns({"joint_names": repr(JOINT_NAMES)}, P.HIP_YAW_JOINT_SUFFIX) == HIP_COLS
    assert P.joint_columns({}, P.CAM_JOINT_SUFFIX) is None
    assert P.joint_columns({"joint_names": ["a", "b"]}, P.CAM_JOINT_SUFFIX) is None
    assert P.wrap_to_pi(2 * np.pi + 0.05) == pytest.approx(0.05)


# ================================================================================================ per-stand estimators
def test_settled_stand(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=50)
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    assert m["reached"] and m["settled"]
    assert m["settle_s"] == pytest.approx(50 * DT)                # homed from row 50 of the stand
    assert m["homed"] is True and m["cam_max_last_rad"] == 0.0
    assert m["six_down_share"] == 1.0 and m["window_rows"] == 250  # last 5 s
    assert m["lifts_before_settle"] == 0                           # the foot was up from row 0: no liftoff inside
    assert m["post_settle_steps_per_s"] == 0.0 and m["post_settle_taps_per_s"] == 0.0
    assert m["post_settle_ks_m_s"] == 0.0
    assert m["body_height_m"] == pytest.approx(1.06)
    assert m["hip_yaw_mean_deg"] == 0.0


def test_restance_lift_before_settle_is_counted(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=100)
    rows = np.flatnonzero(seg == 1)
    d["foot_force_norm"][rows[:20], 2] = 600.0                     # foot 2 lands, then lifts at row 20 (a re-stance)
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    assert m["lifts_before_settle"] == 1 and m["settle_s"] == pytest.approx(100 * DT)


def test_unsettled_stand_fails_closed(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=None, cam0=0.6)
    m = P.stand_z2_metrics(d, 3, DT, CAM_COLS, HIP_COLS)
    assert m["settled"] is False and m["settle_s"] is None
    assert m["homed"] is False and m["cam_max_last_rad"] == pytest.approx(0.6)
    assert m["six_down_share"] == 0.0
    assert m["post_settle_steps_per_s"] is None and m["post_settle_ks_m_s"] is None
    assert m["hip_yaw_mean_deg"] == pytest.approx(10.0)


def test_settle_needs_half_a_second_running(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=50)
    rows = np.flatnonzero(seg == 1)
    d["joint_pos"][rows[70], CAM_COLS[3]] = 0.11                   # a 1-row cam excursion 20 rows into the homed run
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    assert m["settle_s"] == pytest.approx(71 * DT)                 # the run restarts after it
    assert m["post_settle_taps_per_s"] == 0.0
    # an excursion inside the band (|cam| <= 0.10) does not restart it
    d["joint_pos"][rows[70], CAM_COLS[3]] = 0.099
    assert P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)["settle_s"] == pytest.approx(50 * DT)


def test_settle_is_homed_only_six_down_is_reported(P):
    """NOTE(p2r Z2 cams-only, 2026-10-10; P2R_Z2_PLAN "Amendment 2026-10-10"): settle = HOMED held 0.5 s -- a foot
    that never lands (the static-indeterminacy lottery) no longer blocks it; the six-down share is only reported,
    and a foot tap after the settle row counts in the post-settle stillness."""
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=50)
    rows = np.flatnonzero(seg == 1)
    d["foot_force_norm"][rows, 2] = 0.0                            # foot 2 never carries load in this stand
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    assert m["settled"] and m["settle_s"] == pytest.approx(50 * DT) and m["homed"] is True
    assert m["six_down_share"] == 0.0
    d = _stand_episode(seg, (1, 3, 5), settle_at=50)
    d["foot_force_norm"][rows[70], 0] = 0.0                        # a 1-row tap 20 rows after the settle row
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    assert m["settle_s"] == pytest.approx(50 * DT)                 # six-down is not part of settle
    assert m["post_settle_taps_per_s"] == pytest.approx(1 / (350 * DT))


def test_post_settle_steps_counted_from_each_stands_own_settle(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=50, march_after_settle=True)
    m = P.stand_z2_metrics(d, 1, DT, CAM_COLS, HIP_COLS)
    # post-settle rows 350: foot 0 lifts for 20 rows at 30..49, 80..99, ... -- after the homed settle row (50)
    assert m["settled"] and m["settle_s"] == pytest.approx(50 * DT)
    assert m["post_settle_steps_per_s"] == pytest.approx(7 / (350 * DT))
    assert m["six_down_share"] < 0.90


def test_six_down_window_steady_vs_last(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=100)
    d["steady_mask"] = np.zeros(seg.size, dtype=bool)
    rows = np.flatnonzero(seg == 1)
    d["steady_mask"][rows[:100]] = True                            # a steady window inside the unsettled part
    last = P.stand_z2_metrics(d, 1, DT, CAM_COLS, window=P.WINDOW_LAST)
    steady = P.stand_z2_metrics(d, 1, DT, CAM_COLS, window=P.WINDOW_STEADY)
    assert last["six_down_share"] == 1.0 and steady["six_down_share"] == 0.0


def test_no_joint_pos_reads_none(P):
    seg = _walkstop_seg()
    d = _stand_episode(seg, (1, 3, 5), settle_at=50)
    del d["joint_pos"]
    m = P.stand_z2_metrics(d, 1, DT, None)
    assert m["homed"] is None and m["settle_s"] is None and m["six_down_share"] == 1.0


def test_fall_in_a_stand(P):
    seg = _walkstop_seg()
    cf = np.zeros(seg.size, dtype=bool)
    cf[-1] = True                                                  # falls at the end of the third stand
    d = _stand_episode(seg, (1, 3, 5), settle_at=50, crab_failure=cf)
    assert P.stand_z2_metrics(d, 5, DT, CAM_COLS)["fall"] is True
    assert P.stand_z2_metrics(d, 3, DT, CAM_COLS)["fall"] is False


def test_median_fail_closed(P):
    assert P._median_fail_closed([0.1, 0.2, None]) == 0.2
    assert P._median_fail_closed([0.1, None, None]) is None
    assert P._median_fail_closed([0.1, 0.3, None, None]) is None   # the median averages an unsettled stand
    assert P._median_fail_closed([]) is None


# ================================================================================================ head summary
def test_z2_head_summary_three_stand_walkstop(P, tmp_path):
    seg = _walkstop_seg()
    cf = np.zeros(seg.size, dtype=bool)
    cf[-1] = True
    eps = [_stand_episode(seg, (1, 3, 5), settle_at=50), _stand_episode(seg, (1, 3, 5), settle_at=50),
           _stand_episode(seg, (1, 3, 5), settle_at=None, crab_failure=cf)]
    st = _write_raw_stamp(tmp_path / "ws", eps, WS_LABELS, WS_SPEEDS)
    s = P.z2_head_summary(str(st))
    assert s["n"] == 9 and s["n_episodes"] == 3 and s["window"] == "last5" and s["cams_read"]
    assert s["six_down_stand_share"] == pytest.approx(6 / 9, abs=1e-5)
    assert s["homed_share"] == pytest.approx(6 / 9, abs=1e-5)
    assert s["settled_share"] == pytest.approx(6 / 9, abs=1e-5)
    assert s["settle_median_s"] == pytest.approx(1.0)
    assert s["post_settle_steps_per_s"] == 0.0                    # 6 settled 0.0 + 3 unsettled: the median is 0.0
    assert s["falls"] == 1 and s["falls_per_300"] == pytest.approx(300 / 9, abs=1e-4)
    assert set(s["by_stand"]) == {"0", "1", "2"} and s["by_stand"]["2"]["falls"] == 1
    assert P.z2_head_summary(None) is None
    assert P.z2_head_summary(str(tmp_path / "nope")) is None


def test_z2_head_summary_suite_uses_the_steady_window_and_fails_closed(P, tmp_path):
    labels, speeds = ["settle", "stand", "fwd"], [0.0, 0.0, 0.25]
    seg = np.concatenate([np.full(50, 0), np.full(400, 1), np.full(50, 2)])
    eps = [_stand_episode(seg, (1,), settle_at=None) for _ in range(2)] + [_stand_episode(seg, (1,), settle_at=20)]
    st = _write_raw_stamp(tmp_path / "suite", eps, labels, speeds, suite=True)
    s = P.z2_head_summary(str(st))
    assert s["window"] == "steady" and s["n"] == 3 and "by_stand" not in s
    assert s["settled_share"] == pytest.approx(1 / 3, abs=1e-5)
    assert s["post_settle_steps_per_s"] is None                    # 2 of 3 unsettled: the fail-closed median is None
    assert s["post_settle_steps_per_s_settled"] == 0.0


def test_cli_stamp_carries_the_z2_summary(P, tmp_path):
    import subprocess
    seg = _walkstop_seg()
    st = _write_raw_stamp(tmp_path / "ws", [_stand_episode(seg, (1, 3, 5), settle_at=50)], WS_LABELS, WS_SPEEDS)
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2s_stand.py"), "--stamp", str(st), "--json", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["z2"]["n"] == 3


# ================================================================================================ p2_read: gates
def _summ(**kw):
    base = {"n": 300, "six_down_stand_share": 0.95, "homed_share": 0.95, "settled_share": 0.97,
            "settle_median_s": 2.0, "lifts_median": 3.0, "post_settle_steps_per_s": 0.1, "post_settle_taps_per_s": 0.2,
            "post_settle_ks_m_s": 0.1, "falls": 0, "falls_per_300": 0.0, "body_height_m": 1.065,
            "hip_yaw_mean_deg": 0.5, "hip_yaw_max_deg": 1.5}
    base.update(kw)
    return base


def _z2(**kw):
    return {"suite": {s: _summ() for s in (1, 2)}, "walkstop": {s: _summ(**kw) for s in (1, 2)}}


STILL = {"steps_per_s": 0.0, "clearance_m": 0.001, "off_ground_share": 0.0, "hover_episodes_per_100": 0.0,
         "taps_per_s": 0.1, "ks_m_s": 0.2, "body_height_m": 1.065, "falls": 0, "n": 100}


def test_z2_stand_gates_pass_and_each_failure(read):
    still = {1: STILL, 2: STILL}
    z1 = _z2(falls_per_300=0.0)
    probe = _z2(settle_median_s=1.5)
    g = read.z2_stand_gates(_z2(), still, z1, probe)
    assert g["verdict"] == "PASS", g["failed"]
    for kw, frag in ((dict(homed_share=0.5), "homed walkstop es1"),
                     (dict(settled_share=0.94), "settled walkstop es1"),
                     (dict(settle_median_s=2.6), "median settle walkstop es1"),
                     (dict(post_settle_steps_per_s=0.6), "post_settle_steps_per_s walkstop es1"),
                     (dict(post_settle_taps_per_s=1.1), "post_settle_taps_per_s walkstop es1"),
                     (dict(post_settle_ks_m_s=0.51), "post_settle_ks_m_s walkstop es1"),
                     (dict(post_settle_steps_per_s=None), "post_settle_steps_per_s walkstop es1"),
                     (dict(falls_per_300=2.0), "walkstop stand falls per 300 es1")):
        g = read.z2_stand_gates(_z2(**kw), still, z1, probe)
        assert g["verdict"] == "FAIL" and any(frag in f for f in g["failed"]), (kw, g["failed"])
    # the falls line is max(1, Z1's): 1 fall per 300 passes against Z1's 0; 2 pass against Z1's 3
    assert read.z2_stand_gates(_z2(falls_per_300=1.0), still, z1, probe)["verdict"] == "PASS"
    assert read.z2_stand_gates(_z2(falls_per_300=2.0), still, _z2(falls_per_300=3.0), probe)["verdict"] == "PASS"
    # NOTE(p2r Z2 cams-only, 2026-10-10): six-down is REPORTED (``reported``), never a clause -- the Z1-like 22 %
    # passes every gate, and the reported entry reads it against the 0.90 line
    z = _z2(six_down_stand_share=0.22)
    z["suite"][2] = _summ(six_down_stand_share=0.5)
    g = read.z2_stand_gates(z, still, z1, probe)
    assert g["verdict"] == "PASS" and not any("six-down" in k for k in g["clauses"])
    assert g["reported"]["six-down stands walkstop es1 (last 5 s)"] == (0.22, 0.90, False)
    assert g["reported"]["six-down stands suite es2 (steady)"] == (0.5, 0.90, False)
    assert g["reported"]["six-down stands suite es1 (steady)"][2] is True
    # STILL rides along
    g = read.z2_stand_gates(_z2(), {1: dict(STILL, taps_per_s=1.5), 2: STILL}, z1, probe)
    assert "STILL taps_per_s suite es1" in g["failed"]


def test_z2_stand_gates_pending_and_relative(read):
    still = {1: STILL, 2: STILL}
    z = _z2()
    z["walkstop"][2] = None
    g = read.z2_stand_gates(z, still, _z2(), _z2())
    assert g["verdict"] == "PENDING" and "walkstop es2 Z2 read" in g["pending"]
    g = read.z2_stand_gates(_z2(), still, None, _z2())
    assert g["verdict"] == "PENDING" and any("Z1 walkstop" in p for p in g["pending"])
    g = read.z2_stand_gates(_z2(), still, _z2(), _z2(settle_median_s=None))
    assert g["verdict"] == "PENDING" and any("relative settle" in p for p in g["pending"])
    g = read.z2_stand_gates(_z2(), still, _z2(), None, relative=False)
    assert g["verdict"] == "PASS" and not any("median settle" in k for k in g["clauses"])
    g = read.z2_stand_gates(_z2(), still, _z2(), None, seeds=(1,), relative=False)
    assert not any("es2" in k for k in g["clauses"])


def _fpt(name, stamps=(f"{FT}@1", f"{FH}@1"), ft=(0, 100, 0.99, 100), fh=(0, 100, 0.79, 100)):
    field = {}
    for k, (falls, n, rate, nc) in ((f"{FT}@1", ft), (f"{FH}@1", fh)):
        field[k] = {"stand": {"falls": falls, "n": n}, "comp": {"rate": rate, "n": nc}}
    return {"name": name, "stamps": {k: f"/x/{k}" for k in stamps}, "field": field}


def test_z2_field_vs_hold_two_hold_se(read):
    hold = _fpt("P2R_HOLD_s18")
    assert all(ok for _v, _t, ok in read.z2_field_vs_hold(_fpt("RST"), hold)["clauses"].values())
    # against HOLD's 0 falls 2 SE is 0: one fall in field_turn's stand is a fail (the plan's stop-and-ask)
    r = read.z2_field_vs_hold(_fpt("RST", ft=(1, 100, 0.99, 100)), hold)
    assert [k for k, (_v, _t, ok) in r["clauses"].items() if not ok] == [f"{FT}@1 stand-fall rate <= HOLD + 2 SE"]
    # completion: HOLD 0.79 at n=100 -> 2 SE = 0.0815; 0.71 passes, 0.70 fails
    assert read.z2_field_vs_hold(_fpt("RST", fh=(0, 100, 0.71, 100)), hold)["clauses"][
        f"{FH}@1 completion >= HOLD - 2 SE"][2] is True
    assert read.z2_field_vs_hold(_fpt("RST", fh=(0, 100, 0.70, 100)), hold)["clauses"][
        f"{FH}@1 completion >= HOLD - 2 SE"][2] is False
    # a row the probe never ran is NOT RUN, never a pass or a fail
    r = read.z2_field_vs_hold(_fpt("RST", stamps=(f"{FH}@1",)), hold)
    assert r["not_run"] == [f"{FT}@1"] and len(r["clauses"]) == 2


def test_z2_height_vs_refs(read):
    arm = {"name": "RST", "still": {1: {"body_height_m": 1.10}}, "z2": {"walkstop": {1: {"body_height_m": 1.07}}}}
    refs = {"name": "REFS", "still": {1: {"body_height_m": 1.068}}, "z2": {"walkstop": {1: {"body_height_m": 1.068}}}}
    h = read.z2_height_vs_refs(arm, refs, seeds=(1,))
    oks = {k.split(" |")[0]: ok for k, (_v, _t, ok) in h["clauses"].items()}
    assert oks == {"suite es1": False, "walkstop es1": True}          # 0.032 off vs 0.002 off
    assert read.z2_height_vs_refs(arm, {"name": "REFS"}, seeds=(1,))["pending"]


def _light(read, name, *, z2=None, still=None, stamps=None, field=None):
    stamps = stamps if stamps is not None else {"suite@1": "/s1", "suite@2": "/s2", f"{WS}@1": "/w1",
                                                f"{WS}@2": "/w2", f"{FT}@1": "/ft", f"{FH}@1": "/fh"}
    return {"name": name, "stamps": stamps, "z2": z2 or _z2(), "still": still or {1: STILL, 2: STILL},
            "field": field or _fpt(name)["field"], "missing": False}


def test_p2r_rst_gate_verdicts(read, monkeypatch):
    monkeypatch.setattr(read, "rst_invalid_reasons", lambda *a, **k: [])
    refs, hold, z1 = _light(read, "REFS"), _light(read, "HOLD"), _light(read, "Z1")
    g = read.p2r_rst_gate(_light(read, "RST"), hold, refs, z1)
    assert g["verdict"] == "PASS" and g["seeds"][1]["role"] == "tuning seed"
    assert not any("median settle" in k for k in g["seeds"][1]["clauses"])   # the probe SETS the relative line
    # the residual pass: es1 only, no field rows -> PASS ON ROWS RUN, the unrun rows listed
    es1 = _light(read, "RST", stamps={"suite@1": "/s1", f"{WS}@1": "/w1"})
    g = read.p2r_rst_gate(es1, hold, refs, z1)
    assert g["verdict"] == "PASS ON ROWS RUN" and g["seeds"][2]["verdict"] == "NOT RUN"
    assert set(g["not_run"]) == {"suite@2", f"{WS}@2", f"{FT}@1", f"{FH}@1"}
    # a held-out es2 failure is a FAIL (reported per seed)
    z = _z2()
    z["walkstop"][2] = _summ(homed_share=0.2)
    g = read.p2r_rst_gate(_light(read, "RST", z2=z), hold, refs, z1)
    assert g["verdict"] == "FAIL" and g["seeds"][1]["verdict"] == "PASS" and g["seeds"][2]["verdict"] == "FAIL"
    # half a seed is PENDING
    g = read.p2r_rst_gate(_light(read, "RST", stamps={"suite@1": "/s1", f"{WS}@1": "/w1", "suite@2": "/s2"}),
                          hold, refs, z1)
    assert g["verdict"] == "PENDING"
    # body height off REFS by 0.04 m fails
    g = read.p2r_rst_gate(_light(read, "RST", still={1: dict(STILL, body_height_m=1.105), 2: STILL}), hold, refs, z1)
    assert g["verdict"] == "FAIL" and any("body height" in f for f in g["failed"])
    # INVALID outranks everything
    monkeypatch.setattr(read, "rst_invalid_reasons", lambda *a, **k: ["bad env"])
    assert read.p2r_rst_gate(_light(read, "RST"), hold, refs, z1)["verdict"] == "INVALID EVAL"


def _meta_stamp(root: Path, env=None, p2r_mode=None, ckpt=REFS_SHA) -> str:
    root.mkdir(parents=True, exist_ok=True)
    meta = {"env_vars": ENV_Z2 if env is None else env, "env_vars_manifest": {}, "checkpoint_sha256": ckpt}
    if p2r_mode is not None:
        meta["p2r_mode"] = p2r_mode
    (root / "run_meta.json").write_text(json.dumps(meta))
    (root / "code_state.json").write_text(json.dumps(dict(zip(("code_sha", "eval_sha", "dirty"), SC))))
    return str(root)


def test_rst_invalid_reasons(read, tmp_path):
    mode = {"variant": "restance", "s1_checkpoint_sha256": REFS_SHA}
    refs_s = _meta_stamp(tmp_path / "refs", env=ENV4)
    refs = {"name": "REFS", "stamps": {"suite@1": refs_s}, "valid": {"suite@1": read.stamp_validity(refs_s)}}

    def probe(env=None, m=mode):
        s = _meta_stamp(tmp_path / f"p{_N[0]}", env=env, p2r_mode=m)
        _N[0] += 1
        return {"name": "RST", "stamps": {"suite@1": s}, "valid": {"suite@1": read.stamp_validity(s)}, "ws_env": {}}
    assert read.rst_invalid_reasons(probe(), refs, raw_bit=False) == []
    # NOTE(p2r Z2 cams-only, 2026-10-10): the held proxy must be OFF (absent / blank / 0) -- armed is INVALID
    for off in ("0", ""):
        assert read.rst_invalid_reasons(probe(env=dict(ENV_Z2, KRABBY_OMNI_STAND_PROXY=off)), refs, raw_bit=False) == []
    assert any("STAND_PROXY" in r for r in read.rst_invalid_reasons(
        probe(env=dict(ENV_Z2, KRABBY_OMNI_STAND_PROXY="hold")), refs, raw_bit=False))
    assert any("STAND_OBS" in r for r in read.rst_invalid_reasons(probe(env=ENV4), refs, raw_bit=False))
    assert any("variant 'hold'" in r for r in read.rst_invalid_reasons(probe(m=dict(mode, variant="hold")), refs,
                                                                       raw_bit=False))
    assert any("s1_checkpoint" in r for r in read.rst_invalid_reasons(
        probe(m=dict(mode, s1_checkpoint_sha256="b" * 64)), refs, raw_bit=False))
    assert any("no run_meta.p2r_mode" in r for r in read.rst_invalid_reasons(probe(m=None), refs, raw_bit=False))


def test_z2_invalid_widening(read, tmp_path, monkeypatch):
    good = _meta_stamp(tmp_path / "g")
    off0 = _meta_stamp(tmp_path / "o", env=dict(ENV_Z2, KRABBY_OMNI_STAND_PROXY="0"))
    bad = _meta_stamp(tmp_path / "b", env=dict(ENV_Z2, KRABBY_OMNI_STAND_PROXY="hold"))   # the load-sensing obs
    monkeypatch.setattr(read, "z_raw_bit_reasons", lambda pt, lag=1: ([], 1))
    assert read.z2_invalid_reasons({"name": "Z2", "stamps": {"suite@1": good, f"{WS}@1": off0}}) == []
    r = read.z2_invalid_reasons({"name": "Z2", "stamps": {"suite@1": good, f"{WS}@1": bad}})
    assert len(r) == 1 and "KRABBY_OMNI_STAND_PROXY" in r[0] and WS in r[0]
    # the ladder's verdict is overridden (as the clock rule does), PENDING is left alone
    pt = {"name": "P2__Z2_s18", "stamps": {"suite@1": bad}}
    monkeypatch.setattr(read, "p2r_z_verdicts", lambda res: {"P2__Z2_s18": {"verdict": "STILL", "why": ["ok"]}})
    v = read.p2r_z2_verdicts({"heads": {"P2__Z2_s18": pt}})["P2__Z2_s18"]
    assert v["verdict"] == "INVALID EVAL" and "ok" in v["why"]
    monkeypatch.setattr(read, "p2r_z_verdicts", lambda res: {"P2__Z2_s18": {"verdict": "PENDING", "why": ["x"]}})
    assert read.p2r_z2_verdicts({"heads": {"P2__Z2_s18": pt}})["P2__Z2_s18"]["verdict"] == "PENDING"
    # the raw bit is folded in
    monkeypatch.setattr(read, "z_raw_bit_reasons", lambda pt, lag=1: (["raw bit off"], 1))
    assert read.z2_invalid_reasons({"name": "Z2", "stamps": {"suite@1": good}}) == ["raw bit off"]


def test_p2r_z2_report_adoption_grade(read, monkeypatch):
    name = "P2__Z2_s18"
    pt = {"name": name, "stamps": {"suite@1": "/s"}, "p2s": {"suite": {1: STILL, 2: STILL}}}
    monkeypatch.setattr(read, "z2_suite_stamp", lambda nm, lrout=None: "/s")
    monkeypatch.setattr(read, "load_p2r_z", lambda *a, **k: {"heads": {name: pt}, "refs": {18: {}}, "c0r": {}})
    monkeypatch.setattr(read, "bands", lambda *a: {})
    monkeypatch.setattr(read, "p2r_z_verdicts", lambda res: {name: {"verdict": "STILL", "stand": {"outcome": "STILL"},
                                                                    "why": []}})
    monkeypatch.setattr(read, "z_raw_bit_reasons", lambda pt, lag=1: ([], 15))
    monkeypatch.setattr(read, "z2_env_reasons", lambda pt, stamps=None: [])
    monkeypatch.setattr(read, "p2s_attribution", lambda *a: {"status": "LEVER", "control": "C0R_s18"})
    monkeypatch.setattr(read, "z2_reads", lambda stamps: _z2(settle_median_s=2.0))
    monkeypatch.setattr(read, "fh400", lambda p: (0.84, 400))
    lights = {"P2R_RST_s18": _light(read, "P2R_RST_s18", z2=_z2(settle_median_s=1.5)),
              "P2__Z1_s18": _light(read, "P2__Z1_s18", z2=_z2(six_down_stand_share=0.22, homed_share=0.0)),
              "P2R_HOLD_s18": _light(read, "P2R_HOLD_s18"), "REFS_S1_s18": _light(read, "REFS_S1_s18")}
    for lp in lights.values():
        for s in (1, 2, 3, 4):
            lp["field"][f"{FH}@{s}"] = {"stand": None, "comp": {"rate": 0.83, "n": 100}}
    monkeypatch.setattr(read, "z2_light_point", lambda nm, lrout=None: lights[nm])
    out = read.p2r_z2_report(with_tb=False, env={})
    assert out["adoption_grade"] is True and out["z2_gates"]["verdict"] == "PASS"
    assert out["walls_vs_z1"]["z1"] == (0.83, 400) and out["walls_vs_z1"]["delta"] == pytest.approx(0.01)
    assert out["fidelity_to_probe"]["walkstop es1"]["settle_median_s"] == [2.0, 1.5]
    assert out["baselines"]["Z1"]["walkstop"][1]["six_down_stand_share"] == 0.22
    assert out["stopped_hip_yaw_deg"]["Z2"]["walkstop es1"]["hip_yaw_mean_deg"] == 0.5
    assert "Adoption-grade: True" in read.p2r_z2_markdown(out)
    # the relative settle line: Z2 median 2.6 > the probe's 1.5 + 1 -> not adoption-grade
    monkeypatch.setattr(read, "z2_reads", lambda stamps: _z2(settle_median_s=2.6))
    out = read.p2r_z2_report(with_tb=False, env={})
    assert out["adoption_grade"] is False and out["z2_gates"]["verdict"] == "FAIL"
    # attribution not LEVER -> not adoption-grade even with every gate
    monkeypatch.setattr(read, "z2_reads", lambda stamps: _z2(settle_median_s=2.0))
    monkeypatch.setattr(read, "p2s_attribution", lambda *a: {"status": "NOT ATTRIBUTED", "control": "C0R_s18"})
    assert read.p2r_z2_report(with_tb=False, env={})["adoption_grade"] is False


def test_z2_suite_stamp_is_the_driver_stamp(tmp_path, monkeypatch):
    """NOTE(p2r Z2 cams-only, 2026-10-10): the cams-only chain runs the held proxy OFF, so its own driver suite stamp
    is the Z2 read (as Z1's); an lr suite es1 row-eval is used only when there is no driver stamp."""
    out = tmp_path / "out"
    drv = Path(_meta_stamp(out / "driver" / "suite", env=ENV_Z2))
    (out / "arms").mkdir(parents=True)
    (out / "arms" / "P2__Z2_s18.record.json").write_text(json.dumps({"driver_stamp": str(drv)}))
    r = _read_at(out, monkeypatch)
    assert r.z2_suite_stamp("P2__Z2_s18") == str(drv)                    # no row-eval yet
    lr1 = out / "lr" / "P2__Z2_s18" / SUITE / "seed001" / "2026-10-11_00-00-00"
    _meta_stamp(lr1, env=dict(ENV_Z2, KRABBY_OMNI_STAND_PROXY="hold"))
    (lr1 / "complete.json").write_text("{}")
    assert r.z2_suite_stamp("P2__Z2_s18") == str(drv)                    # a proxy-carrying row-eval no longer wins
    lr2 = out / "lr" / "P2R_RST_s18" / SUITE / "seed001" / "2026-10-11_00-00-00"
    _meta_stamp(lr2)
    (lr2 / "complete.json").write_text("{}")
    assert r.z2_suite_stamp("P2R_RST_s18") == str(lr2)                   # no record: the lr row-eval
    assert r.z2_suite_stamp("P2__Z9_s18") is None                        # no record, no stamp


def test_bridged_reads_a_chain_of_appended_pairs(read, tmp_path, monkeypatch):
    camp = tmp_path / "camp"
    (camp / "scripts").mkdir(parents=True)
    monkeypatch.setattr(read, "HERE", camp / "scripts")
    ev = "e" * 64
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    br = {"pairs": [{"from_code_sha": a, "to_code_sha": b, "eval_sha": ev, "ok": True}]}
    (camp / "inertness_bridge.json").write_text(json.dumps(br))
    assert read._bridged((a, ev, False), (b, ev, False)) is True
    assert read._bridged((a, ev, False), (c, ev, False)) is False
    br["pairs"].append({"from_code_sha": b, "to_code_sha": c, "eval_sha": ev, "ok": True})   # the Z2 bridge
    (camp / "inertness_bridge.json").write_text(json.dumps(br))
    assert read._bridged((c, ev, False), (a, ev, False)) is True       # Z2 code vs ae892cec's REFS stamps
    assert read._bridged((a, ev, True), (c, ev, False)) is False       # a dirty tree never bridges
    assert read._bridged((a, "f" * 64, False), (c, "f" * 64, False)) is False   # another eval_sha
    br["pairs"][1]["ok"] = False
    (camp / "inertness_bridge.json").write_text(json.dumps(br))
    assert read._bridged((c, ev, False), (a, ev, False)) is False


def test_the_real_bridge_file_still_accepts_its_pair(read):
    br = json.loads((CAMP / "inertness_bridge.json").read_text())
    pr = br["pairs"][0]
    assert read._bridged((pr["from_code_sha"], pr["eval_sha"], False), (pr["to_code_sha"], pr["eval_sha"], False))


# ================================================================================================ --p2r-rst end to end
def _rst_tree(out: Path, *, probe_settle=30):
    """A fabricated $OUT for --p2r-rst (the suite stand's steady window is the whole stand here; six-down is only
    reported since the 2026-10-10 cams-only amendment)."""
    lr = out / "lr"
    mode = {"variant": "restance", "s1_checkpoint_sha256": REFS_SHA}
    seg_ws = _walkstop_seg()
    suite_labels, suite_speeds = ["settle", "stand", "fwd"], [0.0, 0.0, 0.25]
    seg_su = np.concatenate([np.full(50, 0), np.full(400, 1), np.full(50, 2)])
    ft_labels, ft_speeds = ["walk", "stand", "turn_l90"], [0.25, 0.0, 0.0]
    seg_ft = np.concatenate([np.full(50, 0), np.full(400, 1), np.full(50, 2)])
    fh_labels, fh_speeds = ["stand", "fwd"], [0.0, 0.25]
    seg_fh = np.concatenate([np.full(400, 0), np.full(50, 1)])
    ts = "2026-10-10_00-00-00"

    def put(name, row, seed, eps, labels, speeds, **kw):
        return _write_raw_stamp(lr / name / row / f"seed{seed:03d}" / ts, eps, labels, speeds, **kw)
    # REFS: marching stands (never six-down), its own env (knobs off) and no p2r_mode
    for s in (1, 2):
        put("REFS_S1_s18", SUITE, s, [_stand_episode(seg_su, (1,), settle_at=None)] * 2, suite_labels, suite_speeds,
            suite=True, env=ENV4)
        put("REFS_S1_s18", WS, s, [_stand_episode(seg_ws, (1, 3, 5), settle_at=None)] * 2, WS_LABELS, WS_SPEEDS,
            env=ENV4)
    # Z1: the falls line (walkstop)
    for s in (1, 2):
        put("P2__Z1_s18", WS, s, [_stand_episode(seg_ws, (1, 3, 5), settle_at=None)] * 2, WS_LABELS, WS_SPEEDS)
    # HOLD: the field rows
    put("P2R_HOLD_s18", FT, 1, [_stand_episode(seg_ft, (1,), settle_at=50)] * 2, ft_labels,
        ft_speeds, env=ENV4, comp=1.0)
    put("P2R_HOLD_s18", FH, 1, [_stand_episode(seg_fh, (0,), settle_at=50)] * 2, fh_labels, fh_speeds, env=ENV4,
        comp=1.0)
    # the probe: es1 suite + walkstop + both field rows (es2 not run: the residual-pass shape plus field rows)
    put("P2R_RST_s18", SUITE, 1, [_stand_episode(seg_su, (1,), settle_at=probe_settle)] * 2, suite_labels,
        suite_speeds, suite=True, p2r_mode=mode)
    put("P2R_RST_s18", WS, 1, [_stand_episode(seg_ws, (1, 3, 5), settle_at=probe_settle)] * 2, WS_LABELS, WS_SPEEDS,
        p2r_mode=mode)
    put("P2R_RST_s18", FT, 1, [_stand_episode(seg_ft, (1,), settle_at=50)] * 2, ft_labels, ft_speeds, p2r_mode=mode,
        comp=1.0)
    put("P2R_RST_s18", FH, 1, [_stand_episode(seg_fh, (0,), settle_at=50)] * 2, fh_labels, fh_speeds, p2r_mode=mode,
        comp=1.0)


def test_cli_p2r_rst_end_to_end(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _rst_tree(out)
    r = _read_at(out, monkeypatch)
    md = tmp_path / "rst.md"
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-rst", "--p2r-md", str(md)])
    rc = r.main()
    j = json.loads(capsys.readouterr().out)
    g = j["gate"]
    assert g["verdict"] == "PASS ON ROWS RUN", g.get("why")
    assert rc == 0
    assert g["seeds"]["1"]["verdict"] == "PASS" and g["seeds"]["2"]["verdict"] == "NOT RUN"
    assert j["baselines"]["REFS"]["walkstop"]["1"]["six_down_stand_share"] == 0.0
    assert j["probe"]["walkstop"]["1"]["settle_median_s"] == pytest.approx(0.6)
    text = md.read_text()
    assert "premise gate" in text and "Reported, not gated" in text and "six-down stands walkstop es1" in text


def test_cli_p2r_rst_fails_an_unsettled_probe(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _rst_tree(out, probe_settle=None)
    r = _read_at(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-rst", "P2R_RST_s18"])
    assert r.main() == 4
    g = json.loads(capsys.readouterr().out)["gate"]
    assert g["verdict"] == "FAIL"
    assert any("homed walkstop es1" in f for f in g["failed"])


def test_cli_p2r_rst_invalid_with_the_held_proxy_armed(tmp_path, monkeypatch, capsys):
    """NOTE(p2r Z2 cams-only, 2026-10-10): a probe stamp with KRABBY_OMNI_STAND_PROXY=hold ran the load-sensing
    regime's observation, not the cams-only one: INVALID."""
    out = tmp_path / "out"
    _rst_tree(out)
    meta_p = next((out / "lr" / "P2R_RST_s18" / WS / "seed001").glob("*/run_meta.json"))
    meta = json.loads(meta_p.read_text())
    meta["env_vars"] = dict(meta["env_vars"], KRABBY_OMNI_STAND_PROXY="hold")
    meta_p.write_text(json.dumps(meta))
    r = _read_at(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-rst"])
    assert r.main() == 4
    g = json.loads(capsys.readouterr().out)["gate"]
    assert g["verdict"] == "INVALID EVAL" and any("KRABBY_OMNI_STAND_PROXY" in w for w in g["why"])


# ================================================================================================ p2_live: Z2 wiring
def test_branch_names_pinned_to_the_restance_module(live):
    R = _load(RESTANCE_PY, "_t_z2_restance")
    assert live.P2R_RESTANCE_BRANCHES == R.BRANCH_NAMES
    assert live.P2R_RESTANCE == R.RESTANCE_VAR
    assert live.RESTANCE_PY == RESTANCE_PY


F0 = 14996
AUDIT_OK = [("overlay KRABBY_TERRAIN_PROMOTE=0.7:0.4 in params/env.yaml", True)]
#: NOTE(p2r Z2 cams-only, 2026-10-10): no proxy_live_min any more (the held == live clause is gone)
DOSE = {"band_lo": 3.0, "band_hi": 9.0, "rc": 0, "restance_knob": "lift=-0.5,rho=1,up_frac=0.6,down_frac=0.4,k_c=2,k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1"}  # = format_restance(defaults)


def _ov(**over):
    ov = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
          "KRABBY_ANCHOR_CKPT": "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt",
          "KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "-0.05",
          "KRABBY_STAND_PRIOR_TARGET": "restance", "KRABBY_RESTANCE": "default"}
    ov.update(over)
    return {k: v for k, v in ov.items() if v is not None}


def _ser(live, *, prior=5.0, nonzero=0.9, shares=(0.6, 0.1, 0.1, 0.05, 0.15), drop=(), n=215):
    s = {live.P2R_ANCHOR_ROWS_TAG: 0.85, live.P2R_PRIOR_ROWS_TAG: 0.15, live.P2R_LOSS_ANCHOR: 0.01,
         live.P2R_LOSS_PRIOR: prior, live.P2R_RESTANCE_NONZERO_TAG: nonzero,
         live.TAGS["reach_obst"]: 0.3, live.TAGS["clock_hidden"]: 1.0}
    s.update(dict(zip(live.P2R_RESTANCE_BRANCH_TAGS, shares)))
    ser = {t: [(F0 + i, v) for i in range(n)] for t, v in s.items() if t not in drop}
    ser[live.P2R_ANCHOR_MB0_TAG] = [(F0 + i, 0.0 if i == 0 else 0.05) for i in range(n)]
    return ser


def _facts(live):
    return {"have_env": True, "curriculum_null": True,
            "weights": dict(live.ELEMENT_WEIGHTS, reward_clock_schedule=0.001, reward_clock_swing_apex=0.001,
                            reward_action_rate=-0.05), "term_funcs": {}, "dr_events": {e: True for e in live.DR_EVENTS},
            "push_x": [-0.5, 0.5], "terrain_fns": ["omni_field_terrain"], "num_rows": 10, "dropped_ppo_keys": []}


def _wire(live, arm="Z2", ov=None, ser=None, dose=DOSE):
    return live.judge_wiring(arm, ov or _ov(), _facts(live), ser or _ser(live), F0, AUDIT_OK, None, z2_dose=dose)


def test_z2_wiring_clean_passes(live):
    v = _wire(live)
    assert v["rc"] == live.RC_PASS, v["fails"]
    lv = v["liveness"]
    assert lv["dose_band"] == [3.0, 9.0] and lv["stand_prior_first_update"] == 5.0
    assert lv["restance_branch_shares"]["home"] == pytest.approx(0.05)
    assert "stand_proxy_live_frac" not in lv and "stand_proxy_live_min" not in lv     # the clause is gone
    for name in ("P2R_PROXY_LIVE_TAG", "P2R_PROXY_LIVE_MIN", "z2_proxy_live_min"):
        assert not hasattr(live, name), name
    # the held proxy, explicitly off, passes too
    for off in ("0", ""):
        assert _wire(live, ov=_ov(KRABBY_OMNI_STAND_PROXY=off))["rc"] == live.RC_PASS


@pytest.mark.parametrize("kw,frag", [
    (dict(dose=None), "refuses to run"),
    (dict(dose={"band_lo": 3.0}), "refuses to run"),
    (dict(dose={"band_lo": 9.0, "band_hi": 3.0}), "refuses to run"),
    (dict(ser="prior=2.3"), "realised dose"),
    (dict(ser="nonzero=0.49"), "restance_nonzero"),
    (dict(ser="home=0"), "restance_home"),
    (dict(ser="drop=lift"), "not logged in the window"),
    (dict(dose={"band_lo": 3.0, "band_hi": 9.0, "rc": 2}), "refuses to run"),
    (dict(dose={"band_lo": 3.0, "band_hi": 9.0}), "refuses to run"),                # no rc recorded
    (dict(ser="rows"), "prior_rows"),
    (dict(ov=dict(KRABBY_OMNI_STAND_PROXY="hold")), "KRABBY_OMNI_STAND_PROXY"),      # cams-only: must be off
    (dict(ov=dict(KRABBY_RESTANCE="theta_L=0.6")), "KRABBY_RESTANCE"),             # a removed constant
    (dict(ov=dict(KRABBY_STAND_PRIOR_TARGET="0")), "KRABBY_STAND_PRIOR_TARGET"),
    (dict(ov=dict(KRABBY_RESTANCE="0")), "KRABBY_RESTANCE"),
    (dict(ov=dict(KRABBY_RESTANCE="lift=0.5")), "KRABBY_RESTANCE"),
])
def test_z2_wiring_each_failure(live, kw, frag):
    ser = None
    if "ser" in kw:
        k = kw.pop("ser")
        if k == "prior=2.3":
            ser = _ser(live, prior=2.3)
        elif k == "nonzero=0.49":
            ser = _ser(live, nonzero=0.49)
        elif k == "home=0":
            ser = _ser(live, shares=(0.6, 0.1, 0.1, 0.0, 0.2))
        elif k == "drop=lift":
            ser = _ser(live, drop=("Loss/restance_lift",))
        elif k == "rows":
            ser = _ser(live)
            ser[live.P2R_PRIOR_ROWS_TAG] = [(F0 + i, 0.5) for i in range(215)]
    ov = _ov(**kw.pop("ov")) if "ov" in kw else None
    v = _wire(live, ov=ov, ser=ser, **kw)
    assert v["rc"] == live.RC_FAIL and any(frag in f for f in v["fails"]), v["fails"]


def test_z1_wiring_is_unchanged(live):
    """Z1 keeps its constant [1.2, 4.6] band and never reads the Z2 tags, dose file or not."""
    ser = _ser(live, prior=2.3, drop=("Loss/restance_nonzero",) + live.P2R_RESTANCE_BRANCH_TAGS)
    ov = _ov(KRABBY_STAND_PRIOR_TARGET=None, KRABBY_RESTANCE=None)
    assert live.judge_wiring("Z1", ov, _facts(live), ser, F0, AUDIT_OK, None)["rc"] == live.RC_PASS
    assert live.judge_wiring("Z1", ov, _facts(live), ser, F0, AUDIT_OK, None, z2_dose=DOSE)["rc"] == live.RC_PASS
    assert live.judge_wiring("Z1", ov, _facts(live), _ser(live, prior=5.0), F0, AUDIT_OK, None)["rc"] == live.RC_FAIL


def test_z2_preflight(live):
    graft = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/x"]}
    em = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    ck = _ov()["KRABBY_ANCHOR_CKPT"]

    def pf(ov=None, dose=DOSE, arm="Z2"):
        return live.judge_preflight(arm, em, None, None, ov or _ov(), p2r_graft=graft, fork_ckpt=ck, z2_dose=dose)
    assert pf()["rc"] == live.RC_PASS
    assert pf(dose=None)["rc"] == live.RC_FAIL
    assert pf(ov=_ov(KRABBY_STAND_PRIOR_TARGET=None))["rc"] == live.RC_FAIL
    assert pf(ov=_ov(KRABBY_OMNI_STAND_PROXY="hold"))["rc"] == live.RC_FAIL      # cams-only: the proxy stays off
    # a v13 file (no restance_knob) or one computed for another R is refused: the band must belong to THIS chain's R
    assert pf(dose={"band_lo": 3.0, "band_hi": 9.0, "rc": 0, "proxy_live_min": 0.96})["rc"] == live.RC_FAIL
    assert pf(dose=dict(DOSE, restance_knob="lift=-0.3"))["rc"] == live.RC_FAIL
    assert pf(ov=_ov(KRABBY_ANCHOR_W="0"))["rc"] == live.RC_FAIL                 # Z1's checks still run
    assert pf(arm="Z1", dose=None, ov=_ov(KRABBY_STAND_PRIOR_TARGET=None, KRABBY_RESTANCE=None))["rc"] == live.RC_PASS


def test_check_dispatches_a_z2_relaunch_name_and_reads_the_dose_file(live, tmp_path, monkeypatch):
    chain = "P2__Z2R1_s18"
    cands = tmp_path / "candidates.yaml"
    cands.write_text(yaml.safe_dump({"chains": {chain: {"stages": [{"overlay": _ov()}]}}}))
    monkeypatch.setattr(live, "chain_overlay", lambda c, ch: _ov())
    monkeypatch.setattr(live, "read_tb", lambda run_dir, extra_tags=(): _ser(live))
    monkeypatch.setattr(live, "_gather", lambda run_dir, log: ({}, {}, None))
    monkeypatch.setattr(live, "dump_facts", lambda env, agent, log: _facts(live))

    class _RC:
        @staticmethod
        def overlay_checks(ctx, ov):
            return AUDIT_OK
    monkeypatch.setattr(live, "load_rc", lambda path=None: _RC)
    dose = tmp_path / "p2r_z2_dose.json"
    a = argparse.Namespace(candidates=cands, chain=chain, check=tmp_path, at="200", train_log=None, replay=None,
                           code_root=REPO, arms=tmp_path, source=None, z2_dose=dose)
    rc, res = live.check(a)
    assert rc == live.RC_FAIL and any("refuses to run" in f for f in res["fails"])     # no dose file yet
    dose.write_text(json.dumps(DOSE))
    rc, res = live.check(a)
    assert rc == live.RC_PASS, res.get("fails")
    assert res["liveness"]["dose_band"] == [3.0, 9.0]
    assert live.z_arm(chain) == "Z2" and live.arm_of(chain) == ""
