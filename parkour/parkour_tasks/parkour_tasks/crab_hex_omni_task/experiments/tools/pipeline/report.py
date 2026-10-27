#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 baseline-vs-final report (PLAN step 11; DESIGN 6), generated from RECORDS only (CPU; never boots Kit).

Columns: H (history: the v1 stand-ins as committed; 6300 / 9800 shown as NOT loadable), F (the forward 3a student,
the previous-generation measured "before"), T0 (S1 + SH_T17: the recorded anchors), T1 / St (T* and the selected
student of the deliverable run), and the other run (R18) in an appendix.

NEVER INVENT A NUMBER. Every value is read from a recorded stamp / record or computed on CPU from one, with its
source; a cell whose rows are not on disk yet reads ``pending (row <id>)`` (the step-11 GPU rows) or
``pending step 11`` (a run record not written yet); a cell that cannot exist is ``N/A`` WITH its reason -- never 0.
Numbers computed here that no harness / gate wrote (the omni vy / wz tracking RMS, stride ~ 6 / steps_per_m, the grid
|achieved - cmd|, the pooled drift p90, the forward 50 N set) are labelled ``derived (step 11)``; every payload number
is labelled ``sim plant, force side not enforced``; the 1 N metrics carry the misread caveat (GE/metrics.py:1271-1275).

Sections: Task 0 (per category: tripod, the 50 N rigidity set, tracking, direction error / cross track, motion
quality, world turning; the 1 N metrics; vy / wz RMS; roll / pitch + survival; actuator load at 0 kg), Task 2 (grid,
turn levels, body suite, the 16-bin sweep with its discontinuity, field rows, side22), Task 3 (the 0-270 kg curves,
max_stable, the limiting factor, actuator load per level, F's added-mass raw table, the paired 0-kg check and the heads
considered for T*), the stand K1-K6 beside SH_T17 (with the lazy-control re-anchor), the student (pairing, 90-kg WATCH,
live checks, camera use, clip + deploy export), provenance (MEASURED per-stage wall-clock / GPU-h from the queue logs,
HALT parks separated; hardware; pins and tags; every gate verdict; every HALT with its chance rate or "rate unknown";
the v1.1 -> v1.2 amendment), the grant reconciliation, and every disclosure.

The JSON is the machine record (``cells``: one entry per fill-plan cell with status filled / pending / na, its sources
and value); the .md is the human one, headline first.

    report.py --spec PL/pipeline.yaml --run F19 [--other R18] [--out PL/REPORT_baseline_vs_final]
              [--rows-log STEP11_QUEUE_LOG] [--readme PL/README.md]

``--rows-log``: the step-11 queue's sentinel log (``S11 ROW / DONE rc / SKIP``): a SKIPPED or failed row reads N/A with
its reason, and the rows' measured wall-clock is reported. Without it a missing step-11 stamp is just pending.
``--readme``: also (re)generate the campaign README (:func:`readme_md`) from the same records.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _lib():
    import importlib.util  # noqa: PLC0415

    if "tp_lib" in sys.modules:
        return sys.modules["tp_lib"]
    spec = importlib.util.spec_from_file_location("tp_lib", str(HERE / "tp_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tp_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _lib()
PENDING = "pending step 11"
NA = "N/A"
SIM = "sim plant, force side not enforced"
DERIVED = "derived (step 11)"
MISREAD = ("1 N metrics (tripod contact test, air time, tippy-tap, duty) use the scenario's 1 N contact threshold, "
           "which a lightly grazing foot clears; the gait read is the 50 N rigidity set (GE/metrics.py:1271-1275)")
FILLED, PEND, NOTAPP = "filled", "pending", "na"
#: the local annotated step-11 tag (CHARTER 2026-10-24 decision 4, DESIGN 2.4 name); never pushed
BASELINE_TAG = "omni-sim2real-baseline-2026-10-24"
#: a median over fewer than this many episodes prints its n beside it (``value (n=K)``); n = 0 prints N/A
N_FLOOR = 10
N_RULE = (f"A median computed from fewer than {N_FLOOR} episodes prints its episode count beside it as `value (n=K)`; "
          "n = 0 prints N/A")
K6_RATE_CAVEAT = "computed from the S1-era R_h values, i.e. the pre-re-anchor lines (CHARTER 2026-10-21)"
ROBOT_KG = 230.06
CHASSIS_KG = 158.757
RECONCILIATION = [
    f"Robot mass {ROBOT_KG} kg (as built), not the grant's 83 kg: 90 / 180 / 270 kg = "
    f"{90 / ROBOT_KG:.2f} / {180 / ROBOT_KG:.2f} / {270 / ROBOT_KG:.2f} x robot mass.",
    "Actuators as built: hip SLA08 2000 N @ 28 mm/s, knee YH8-523D 500 N @ 33 mm/s (not 500 N / 80 mm/s and "
    "400 N / 60 mm/s).",
    "Every payload ceiling is a SIM-PLANT number: no rod-force limit is enforced (the hip rod is ~3.4 kN static at "
    "0 kg); the force side is report-only (D8).",
]
#: the remaining grant-text vs as-run differences (TASK-4 deliverables / 4c / 4e; PLAN Budget)
RECONCILIATION_MORE = [
    "Weights location: the grant asks for the final teacher and student in `runs/`; this repo keeps the keep-set "
    "under `PL/head/` via bundle_experiment.py (the student `.pt` is cited by sha256, see Keep-set), tagged locally "
    f"`{BASELINE_TAG}` (CHARTER 2026-10-24 decision 4; never pushed; the push and the bake are the user's).",
    "4c 'every gate passed': the committed gates are HALT-as-prompt (PLAN On FAIL). The deliverable run passed "
    "every gate OR had its HALT resolved by the user (accepted, amended or re-anchored) -- each listed under "
    "Provenance -> HALTs, none silently.",
    "Dev effort: the grant's Task 4 estimate is ~3 dev days; the plan budgeted ~8-11 (PLAN Budget: the payload "
    "build and the omni depth student did not exist yet).",
    "Grant student baseline `9800` (and teacher `6300`) do not load on today's tree (see Task 0 -> H); the "
    "'before' side is the v1 stand-ins as committed plus the forward 3a student (F).",
]
CATS = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
TRANSLATE = ("fwd", "side", "back", "diag")
#: Task 0 per-category keys (run_phases.cat_summary medians of aggregate.suite.by_category; 50 N = rigidity set)
CAT_KEYS = (
    ("n", "episodes (own hold)"), ("survival", "survival (own hold)"), ("misses", "falls (own hold)"),
    ("ratio", "tracking ratio"), ("achieved_speed", "achieved speed (m/s)"),
    ("direction_error", "direction_error (rad)"), ("cross_track_ratio", "cross_track_ratio"),
    ("tripod", "tripod score (1 N contact)"),
    ("support_feet", "support_feet (50 N)"), ("steps_per_m", "steps_per_m (50 N)"),
    ("stride_m", f"stride ~ 6 / steps_per_m (m, {DERIVED})"), ("taps_per_m", "taps_per_m (50 N)"),
    ("swing_apex", "swing_apex (m, 50 N)"), ("step_rate_hz", "step_rate_hz (50 N)"),
    ("steps_per_rad", "steps_per_rad (50 N)"), ("taps_per_rad", "taps_per_rad (50 N)"),
    ("steps_per_s", "steps_per_s (50 N, stand)"),
    ("action_jitter", "action_jitter"), ("action_fft_freq", "action_fft_freq (Hz)"),
    ("joint_speed_mean", "joint speed mean (rad/s)"), ("rod_speed_mean", "rod speed mean (m/s)"),
    ("facing_closed", "facing_closed (world turning)"), ("yaw_progress", "yaw_progress (rad/s)"),
)
EP_KEYS = (
    ("duty_hold", "duty factor, own hold (1 N; valid windows only; mean of 6 feet)"),
    ("duty_episode", "duty factor, whole episode (slip_episode per foot; mean of 6 feet)"),
    ("air_p50_hold", "air time p50, own hold (s, 1 N)"), ("tippy_hold", "tippy-tap fraction, own hold (1 N)"),
    ("tippy_episode", "tippy-tap fraction, episode (1 N)"),
    ("action_rate", "action rate (actions.action_rate_mean_all)"),
    ("sign_reversals", "action sign reversals (episode total)"),
    ("roll_rms", "roll RMS (rad, episode)"), ("pitch_rms", "pitch RMS (rad, episode)"),
)
BODY_KEYS = ("net_yaw_rate", "net_yaw_ratio", "wz_ratio", "yaw_ripple", "heading_drift", "planar_drift",
             "body_ratio", "cross_body")
#: DESIGN 3.2 (p2a row, "Grid"): the 9 in-envelope cells (|v| 0 x wz {0, .08, .15}; 0.22 x {0, .08} fwd / lat;
#: 0.30 x 0 fwd / lat); the other 11 cells are reported
GRID_IN_ENVELOPE = ("g_v0_w0", "g_v0_w0p08", "g_v0_w0p15", "g_v0p22_fwd_w0", "g_v0p22_fwd_w0p08", "g_v0p22_lat_w0",
                    "g_v0p22_lat_w0p08", "g_v0p3_fwd_w0", "g_v0p3_lat_w0")
FIELD_ROWS = ("field_hard", "field_light", "ring_obst", "field_turn")
LEVELS = (0, 45, 90, 135, 180, 225, 270)
SWEEP_KEYS = ("ratio", "tripod", "steps_per_m")
#: the forward records of the previous generation (repo-relative under parkour/logs/rsl_rl; the forward 3a
#: model_24995 phase-pipeline rows -- recorded 2026-09-08/09 build: no plant key, no code_state.json, env_vars {})
FWD_RECORDED = {
    "slow": "gait_eval/phases/2026-09-07_1330_phase_pipeline/seed3_3a_003_A15pB/slow__A15pB/seed001",
    "step": "gait_eval/phases/2026-09-07_1330_phase_pipeline/seed3_3a_003_A15pB/step__A15pB/seed001",
    "obstacle": "gait_eval/phases/2026-09-07_1330_phase_pipeline/seed3_3a_003_A15pB_obst/flat_walk_slow_v2/seed001",
    "hard_band": "gait_eval/phases/2026-09-07_1330_phase_pipeline_hardband/student_3a/flat_walk_slow_v2/seed001",
}
FWD_POLICY = "policy/3a_student/model_24995.pt"
V1_STAMPS = ("teacher_2b2_forward", "student_2b2_forward", "student_flat_forward", "flat_walk_forward")
V1_PROBES = ("student_flat_probe_vy", "student_flat_probe_yaw")
GRANT_CKPTS = {"teacher 6300": "2026-05-26_21-46-37/model_6300.pt", "student 9800": "2026-05-26_22-57-01/model_9800.pt"}
GRANT_9800_README = "2026-05-26_22-57-01/README.md"
GRANT_DIR = "patina-foundation-grants/grants/Krabby-Uno/Milestone18-Krab-Sim-Tuning"
STEP11 = "step11"
COMPLETE_MARKERS = ("tp_row.json", "complete.json", "p2_complete.json")


def _s11_rows() -> dict:
    """The step-11 GPU rows (fill plan gpu_rows): id -> (NAME, SCENARIO, env seed) under ``<row_out>/step11/lr``."""
    rows = {}
    for col, name in (("ST", "T4S11_c28000"), ("T1", "T4S11_tstar")):
        for fam, sc in (("suite_body", "suite__omni_body"), ("turnlevels_body", "turnlevels__omni_body"),
                        ("grid_body", "grid__omni_body"), ("dirsweep_body", "dirsweep__omni_body")):
            for es in (1, 2):
                rows[f"{col}_{fam}_{es}"] = (name, sc, es)
    for lv in (45, 135, 180, 225, 270):
        for fam, sc in (("suite", "suite__omni"), ("fl", "field_light__omni")):
            for es in (1, 2):
                rows[f"ST_L{lv:03d}_{fam}_{es}"] = (f"T4S11_c28000_L{lv:03d}", sc, es)
    rows["F_step_pin"] = ("T4S11_fwd3a_pin", "step__A15pB", 1)
    for lv in (45, 90, 135, 180, 225, 270):
        rows[f"F_m{lv:03d}"] = (f"T4S11_fwd3a_m{lv:03d}", "slow__A15pB", 1)
    return rows


S11_ROWS = _s11_rows()
#: rows the queue runs only when the previous curve level reads stable (fill plan: excluded from the total)
S11_CONDITIONAL = {lv: f"ST_L{lv:03d}" for lv in (180, 225, 270)}
#: body-family name of a step-11 row (the row's own family) -> the step-11 id stem
BODY_FAMS = ("suite_body", "turnlevels_body", "grid_body", "dirsweep_body")


# ============================================================================== small helpers
def _P():
    return L.load_by_path(HERE / "pipeline.py", "tp_pipeline")


def _ge():
    """FWD/scripts/gait_eval/metrics.py (pure numpy), by path, no bytecode."""
    return L.load_by_path(L.FWD / "scripts/gait_eval/metrics.py", "tp_ge_metrics")


def _np():
    import numpy as np  # noqa: PLC0415

    return np


def fmt(v, nd: int = 3) -> str:
    if v is None:
        return NA
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if v != 0 and abs(v) < 0.01:
            return f"{v:.2e}" if abs(v) < 1e-4 else f"{v:.4f}"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        return f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return "/".join(fmt(x, nd) for x in v)
    return str(v)


def pair(a, b, nd: int = 3) -> str:
    return f"{fmt(a, nd)} / {fmt(b, nd)}"


def fmt_n(v, n=None, nd: int = 3) -> str:
    """:data:`N_RULE`: a median from fewer than :data:`N_FLOOR` episodes shows ``(n=K)`` beside it; n = 0 -> N/A."""
    if v is None or n == 0:
        return NA
    if isinstance(n, int) and n < N_FLOOR:
        return f"{fmt(v, nd)} (n={n})"
    return fmt(v, nd)


def _ns(d: dict, keys) -> dict:
    """{key: number of non-None values} of a dict of per-episode lists (the ``_n`` beside the medians)."""
    return {k: sum(1 for x in d.get(k) or [] if x is not None and not (isinstance(x, float) and not math.isfinite(x)))
            for k in keys}


CLIP_MARK = " ..."


def _clip(text, n: int) -> str:
    """Shorten ``text`` to about ``n`` chars WITHOUT ever cutting inside a token (a number, an id, a date): cut at
    the last clause boundary ('; ', ' -- ', ', ') in the second half of the window, else at the last whitespace, and
    mark the cut with ``CLIP_MARK``; one unbroken token is never cut. A raw ``s[:n]`` once printed
    'margin 0.0' for 'margin 0.0291' -- never slice report text. A caller that clips a table cell also prints the
    full text under the table (``_full_notes``); the JSON always keeps the full string."""
    s = "" if text is None else str(text)
    if len(s) <= n:
        return s
    head = s[: n + 1]
    for sep in ("; ", " -- ", ", "):
        i = head.rfind(sep)
        if i >= n // 2:
            return s[:i].rstrip(" ;,") + CLIP_MARK
    i = head.rfind(" ")
    if i > 0:
        return s[:i].rstrip(" ;,") + CLIP_MARK
    return s


def _full_notes(items, n: int, what: str) -> tuple:
    """``items`` = [(key, text)]: the clipped cells for a table + the bullet lines carrying the full text of every
    cell that was clipped (so the .md never loses a number to a table width)."""
    cells, notes = [], []
    for key, text in items:
        s = "" if text is None else str(text)
        c = _clip(s, n)
        cells.append(c + (" (full below)" if c != s else ""))
        if c != s:
            notes.append(f"- {key} {what} (full): {s}")
    return cells, notes


def med(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and not math.isfinite(x))]
    if not xs:
        return None
    return float(_np().median(xs))


def pct(xs, q):
    xs = [float(x) for x in xs if x is not None]
    return float(_np().percentile(xs, q)) if xs else None


def table(header: list, rows: list) -> list:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows]
    return out + [""]


def rel(p) -> str:
    return L.rel(p) if p else None


def complete(stamp) -> bool:
    d = Path(stamp) if stamp else None
    return bool(d and d.is_dir() and (d / "run_meta.json").is_file() and (d / "scenario_metrics.json").is_file()
                and any((d / m).is_file() for m in COMPLETE_MARKERS))


def newest_stamp(root: Path, name: str, scenario: str, seed: int):
    """The newest COMPLETE stamp ``<root>/<NAME>/<SCENARIO>/seed00<k>/<time>/`` (a completion marker present), else
    None (a running row's stamp has no marker yet)."""
    d = Path(root) / name / scenario / f"seed{int(seed):03d}"
    if not d.is_dir():
        return None
    for s in sorted((x for x in d.iterdir() if x.is_dir()), reverse=True):
        if complete(s):
            return s
    return None


def newest_in(d: Path):
    if not Path(d).is_dir():
        return None
    for s in sorted((x for x in Path(d).iterdir() if x.is_dir()), reverse=True):
        if (s / "scenario_metrics.json").is_file():
            return s
    return None


def jload(p):
    return json.loads(Path(p).read_text())


def run_meta(stamp) -> dict:
    return jload(Path(stamp) / "run_meta.json")


def aggregate(stamp) -> dict:
    return jload(Path(stamp) / "scenario_metrics.json")["aggregate"]


def _ep_index(path: str) -> int:
    return int(re.search(r"episode_(\d+)\.", path).group(1))


def _unreadable(v) -> list:
    if isinstance(v, dict):
        return ([str(v["unreadable"])] if "unreadable" in v else []) + [x for y in v.values() for x in _unreadable(y)]
    if isinstance(v, (list, tuple)):
        return [x for y in v for x in _unreadable(y)]
    return []


def _has_unreadable(v) -> bool:
    return bool(_unreadable(v))


def _src(path=None, *, status=None, why=None, row=None) -> dict:
    """A located source: a stamp / record path, or why there is none."""
    if path is not None and status is None:
        status = "ok"
    return {"path": str(path) if path is not None else None, "status": status or PEND, "why": why, "row": row}


def _ok(s) -> bool:
    return bool(s) and s.get("status") == "ok"


def _why(s) -> str:
    if not s:
        return PENDING
    if s.get("status") == PEND:
        return f"pending (row {s['row']})" if s.get("row") else (s.get("why") or PENDING)
    return s.get("why") or NA


# ============================================================================== stamp readers (CPU)
def read_categories(stamp) -> dict:
    """Per category of a suite-kind stamp: run_phases.cat_summary medians (+ BODY keys where written) + the own-hold
    falls (thresholds.suite_categories) + stride ~ 6 / steps_per_m (derived)."""
    a = aggregate(stamp)
    cs = L.rc().rp.cat_summary(a)
    try:
        sc = L.thresholds_mod().suite_categories(Path(stamp))
    except Exception:  # noqa: BLE001  (a non-category stamp has no own-hold miss count)
        sc = {}
    out = {}
    blocks = (a.get("suite") or {}).get("by_category") or {}
    for c, v in cs.items():
        d = {k: v.get(k) for k, _ in CAT_KEYS if k in v}
        d.update({k: v.get(k) for k in BODY_KEYS if k in v})
        d["n"], d["survival"] = v.get("n"), v.get("survival")
        d["misses"] = (sc.get(c) or {}).get("misses")
        spm = v.get("steps_per_m")
        d["stride_m"] = (6.0 / spm) if spm else None
        # the episode count behind each harness median (aggregate.suite.by_category[c][key].n; N_RULE)
        blk = blocks.get(c) or {}
        d["_n"] = {k: blk[k].get("n") for k in list(d) if isinstance(blk.get(k), dict) and "n" in blk[k]}
        if "steps_per_m" in d["_n"]:
            d["_n"]["stride_m"] = d["_n"]["steps_per_m"]
        out[c] = d
    return {"by_category": out, "trained_categories": (a.get("suite") or {}).get("trained_categories"),
            "n_episodes": a.get("n_episodes")}


def read_episode_extras(stamp) -> dict:
    """Per category, from ``metrics/episode_*.json``: duty factor (own hold, VALID windows only -- absent on a
    discarded window, so None, never 0; and whole-episode per foot), 1 N air time / tippy-tap, action rate / sign
    reversals, episode roll / pitch RMS. Medians over the category's episodes."""
    meta = run_meta(stamp)
    cat_of = {int(m["env_index"]): m.get("category") for m in meta.get("env_index_map") or []}
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(stamp) / "metrics" / "episode_*.json"))):
        ep = jload(f)
        ei = int(ep.get("env_index", _ep_index(f)))
        cat = ep.get("category") or cat_of.get(ei)
        lab = ep.get("hold")
        h = (ep.get("holds") or {}).get(lab) or {}
        a = acc[cat]
        a["n"].append(1)
        df = (h.get("tripod") or {}).get("duty_factor")
        a["duty_hold"].append(sum(df.values()) / len(df) if isinstance(df, dict) and df else None)
        pf = (ep.get("slip_episode") or {}).get("per_foot") or {}
        d2 = [v.get("duty_factor") for v in pf.values() if isinstance(v, dict) and v.get("duty_factor") is not None]
        a["duty_episode"].append(sum(d2) / len(d2) if d2 else None)
        at = h.get("air_time") or {}
        a["air_p50_hold"].append((at.get("pooled") or {}).get("p50"))
        a["tippy_hold"].append(at.get("tippy_tap_fraction"))
        a["tippy_episode"].append(ep.get("tippy_tap_fraction"))
        act = ep.get("actions") or {}
        a["action_rate"].append(act.get("action_rate_mean_all"))
        a["sign_reversals"].append(act.get("action_sign_reversals_total"))
        o = ep.get("orientation") or {}
        a["roll_rms"].append(o.get("roll_rms"))
        a["pitch_rms"].append(o.get("pitch_rms"))
    out = {}
    for cat, a in acc.items():
        out[cat] = {k: med(a[k]) for k, _ in EP_KEYS}
        out[cat]["n"] = len(a["n"])
        out[cat]["n_duty_valid"] = sum(1 for x in a["duty_hold"] if x is not None)
        out[cat]["_n"] = _ns(a, [k for k, _ in EP_KEYS])
    return out


def read_vel_rms(stamp) -> dict:
    """Omni vy / wz (and vx) tracking RMS per category (DERIVED, step 11: no harness key exists for omni). Per
    episode, over its OWN hold (``segment_id`` == the hold's index in ``hold_labels``) AND ``steady_mask``, alive steps
    only (before ``done``): RMS of root_lin_vel_b[:, 1] - cmd_applied[:, 1] (body frame both) and of
    root_ang_vel_b[:, 2] - cmd_applied[:, 2]; the category's median over episodes."""
    np = _np()
    meta = run_meta(stamp)
    idx = {lab: i for i, lab in enumerate(meta.get("hold_labels") or [])}
    emap = {int(m["env_index"]): m for m in meta.get("env_index_map") or []}
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(stamp) / "raw" / "episode_*.npz"))):
        m = emap.get(_ep_index(f))
        if m is None or m.get("label") not in idx:
            continue
        z = np.load(f)
        done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(done)
        w = (z["segment_id"][:S] == idx[m["label"]]) & z["steady_mask"][:S].astype(bool)
        if int(w.sum()) < 10:
            continue
        vb, wb, cmd = z["root_lin_vel_b"][:S][w], z["root_ang_vel_b"][:S][w], z["cmd_applied"][:S][w]
        a = acc[m.get("category")]
        a["vx_rms"].append(float(np.sqrt(np.mean((vb[:, 0] - cmd[:, 0]) ** 2))))
        a["vy_rms"].append(float(np.sqrt(np.mean((vb[:, 1] - cmd[:, 1]) ** 2))))
        a["wz_rms"].append(float(np.sqrt(np.mean((wb[:, 2] - cmd[:, 2]) ** 2))))
    return {c: {"vx_rms": med(a["vx_rms"]), "vy_rms": med(a["vy_rms"]), "wz_rms": med(a["wz_rms"]),
                "n": len(a["vy_rms"]), "_n": _ns(a, ("vx_rms", "vy_rms", "wz_rms"))} for c, a in acc.items()}


def read_orientation(stamp) -> dict:
    return L.thresholds_mod().h_meas_and_orientation(Path(stamp))


def read_actuator(stamp, level_kg: float = 0.0) -> dict:
    a = _P().PR().actuator_load(Path(stamp), level_kg)
    if not a.get("available"):
        return a
    return {"available": True, "label": SIM, "cap_share": a["cap_share"],
            "rod_force_n": a["rod_force_n"], "sum_foot_force_ratio": a["sum_foot_force_ratio"],
            "torque_note": a["torque_note"], "steps": a["steps"]}


def read_field(*stamps) -> dict:
    """thresholds.field_completion over the env-seed stamps (n-weighted pool + per env seed)."""
    return L.thresholds_mod().field_completion([Path(s) for s in stamps])


def read_side22(stamp) -> dict:
    r = _P().GR.side22(Path(stamp))
    return {"class": r["class"], "median": r["median"], "n": r["n"]}


def read_grid(stamp) -> dict:
    """grid__omni_body per cell: the harness medians + the commanded (vx, vy, wz) from run_meta + |achieved_speed -
    |v_cmd|| and |net_yaw_rate - wz_cmd| (DERIVED from the medians) + the in-envelope flag; plus the pooled p90 of the
    per-episode heading drift over the in-envelope cells (DERIVED; the aggregate stores no p90)."""
    meta = run_meta(stamp)
    cmd = {}
    for c in (meta.get("suite") or {}).get("categories") or []:
        h = (c.get("holds") or [{}])[0]
        cmd[c["name"]] = (h.get("vx"), h.get("vy"), h.get("wz"))
    cats = read_categories(stamp)["by_category"]
    out = {}
    for c, v in cats.items():
        vx, vy, wz = cmd.get(c, (None, None, None))
        vmag = math.hypot(vx or 0.0, vy or 0.0) if vx is not None else None
        vn = v.get("_n") or {}
        out[c] = {"in_envelope": c in GRID_IN_ENVELOPE, "cmd": [vx, vy, wz], "survival": v.get("survival"),
                  "_n": {**{k: vn.get(k) for k in ("achieved_speed",) + BODY_KEYS if k in vn},
                         **({"abs_speed_err": vn["achieved_speed"]} if "achieved_speed" in vn else {}),
                         **({"abs_wz_err": vn["net_yaw_rate"]} if "net_yaw_rate" in vn else {})},
                  "achieved_speed": v.get("achieved_speed"), **{k: v.get(k) for k in BODY_KEYS},
                  "abs_speed_err": (abs(v["achieved_speed"] - vmag) if v.get("achieved_speed") is not None
                                    and vmag is not None else None),
                  "abs_wz_err": (abs(v["net_yaw_rate"] - wz) if v.get("net_yaw_rate") is not None
                                 and wz is not None else None)}
    drifts = []
    for f in glob.glob(str(Path(stamp) / "metrics" / "episode_*.json")):
        ep = jload(f)
        if ep.get("category") not in GRID_IN_ENVELOPE:
            continue
        hd = (((ep.get("holds") or {}).get(ep.get("hold")) or {}).get("body") or {}).get("heading_drift")
        if hd is not None:
            drifts.append(abs(float(hd)))
    return {"cells": out, "drift_pooled_in_envelope": {"median": med(drifts), "p90": pct(drifts, 90),
                                                       "n": len(drifts), "label": DERIVED}}


def sweep_read(stamps_by_es: dict, th_doc: dict) -> dict:
    """The 16-bin direction sweep: per env seed the bins' medians of ratio / tripod / steps_per_m; per the committed
    sweep rule (thresholds.sweep_watch_lines) the value per bin is the MEAN of the env seeds' medians; per bin
    |ratio - 1| against ``p2a.watch.sweep.abs_dev.<bin>`` and the max ADJACENT-bin jump (circular) per key against
    ``p2a.watch.sweep.max_adjacent_jump.<key>``. WATCH only."""
    per_es, order, n_es = {}, None, {}
    for es, st in sorted(stamps_by_es.items()):
        bc = aggregate(st)["suite"]["by_category"]
        order = order or list(bc)
        per_es[es] = {c: {k: ((bc[c] or {}).get(k) or {}).get("median") for k in SWEEP_KEYS} for c in bc}
        n_es[es] = {c: {k: ((bc[c] or {}).get(k) or {}).get("n") for k in SWEEP_KEYS} for c in bc}
    lines = {x["id"]: x["line"] for x in th_doc.get("lines") or [] if str(x.get("id", "")).startswith("p2a.watch.sweep.")}
    if not per_es:
        return {}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    pooled = {c: {k: mean([per_es[e][c][k] for e in per_es if c in per_es[e]]) for k in SWEEP_KEYS} for c in order}
    bins = {}
    for c in order:
        r = pooled[c]["ratio"]
        dev = None if r is None else abs(r - 1.0)
        ln = lines.get(f"p2a.watch.sweep.abs_dev.{c}")
        bins[c] = {**{f"{k}_es{e}": per_es[e][c][k] for e in per_es for k in SWEEP_KEYS if c in per_es[e]},
                   **pooled[c], "abs_dev": dev, "line": ln,
                   "_n": {k: min((n_es[e][c][k] for e in n_es if c in n_es[e] and n_es[e][c][k] is not None),
                                 default=None) for k in SWEEP_KEYS},
                   "pass": (dev <= ln + 1e-12) if (dev is not None and ln is not None) else None}
    jumps = {}
    for k in SWEEP_KEYS:
        v = [pooled[c][k] for c in order]
        j = max(abs(v[i] - v[i - 1]) for i in range(len(v))) if v and all(x is not None for x in v) else None
        ln = lines.get(f"p2a.watch.sweep.max_adjacent_jump.{k}")
        jumps[k] = {"value": j, "line": ln, "pass": (j <= ln + 1e-12) if (j is not None and ln is not None) else None}
    return {"bins": bins, "max_adjacent_jump": jumps, "env_seeds": sorted(per_es), "watch": True,
            "rule": "per bin: mean of the env seeds' medians (thresholds.sweep_watch_lines); WATCH only"}


def read_forward(stamp) -> dict:
    """A forward / v1 stamp (holds run in sequence in every episode): aggregate headline + per hold from the episode
    JSONs (tracking.{vx ratio, vy rms, wz rms}, tripod score, duty (valid windows), 1 N air time p50 / tippy) +
    episode-level tippy-tap, slip, roll / pitch RMS, action rate / sign reversals."""
    a = aggregate(stamp)
    meta = run_meta(stamp)

    def m(k):
        x = a.get(k)
        return x.get("median") if isinstance(x, dict) else x

    out = {"checkpoint": meta.get("checkpoint"), "checkpoint_sha256": meta.get("checkpoint_sha256"),
           "task": meta.get("task"), "plant": meta.get("plant"), "scenario_id": meta.get("scenario_id"),
           "hold_labels": meta.get("hold_labels"), "env_vars": meta.get("env_vars"),
           "code_state": (Path(stamp) / "code_state.json").is_file(),
           "tripod_score": m("tripod_score"), "tippy_tap_fraction": m("tippy_tap_fraction"),
           "slip_ratio": m("slip_ratio"), "schedule_completion_rate": a.get("schedule_completion_rate"),
           "termination_reasons": a.get("termination_reasons"), "n_episodes": a.get("n_episodes"),
           "roll_rms": m_or((a.get("orientation") or {}).get("roll_rms")),
           "pitch_rms": m_or((a.get("orientation") or {}).get("pitch_rms")),
           "_n": {k: a[k].get("n") for k in ("tripod_score", "tippy_tap_fraction", "slip_ratio")
                  if isinstance(a.get(k), dict) and "n" in a[k]},
           "tracking_by_hold": {h: {"ratio": ((v or {}).get("ratio") or {}).get("median"),
                                    "achieved_vx": ((v or {}).get("achieved_vx") or {}).get("median"),
                                    "cmd_vx": (v or {}).get("cmd_vx_mean")}
                                for h, v in (a.get("tracking_by_hold") or {}).items()}}
    acc = defaultdict(lambda: defaultdict(list))
    ep_acc = defaultdict(list)
    for f in sorted(glob.glob(str(Path(stamp) / "metrics" / "episode_*.json"))):
        ep = jload(f)
        for lab, h in (ep.get("holds") or {}).items():
            tr = h.get("tracking") or {}
            b = acc[lab]
            vx = tr.get("vx") or {}
            b["vx_ratio"].append(vx.get("ratio"))
            cm = vx.get("cmd_mean")
            b["vx_ratio_derived"].append((vx["actual_mean"] / cm) if (vx.get("ratio") is None and cm
                                                                       and vx.get("actual_mean") is not None) else None)
            b["vy_rms"].append((tr.get("vy") or {}).get("rms"))
            b["wz_rms"].append((tr.get("wz") or {}).get("rms"))
            tri = h.get("tripod") or {}
            b["tripod"].append(tri.get("tripod_score"))
            df = tri.get("duty_factor")
            b["duty_hold"].append(sum(df.values()) / len(df) if isinstance(df, dict) and df else None)
            at = h.get("air_time") or {}
            b["air_p50"].append((at.get("pooled") or {}).get("p50"))
            b["tippy"].append(at.get("tippy_tap_fraction"))
        ep_acc["tippy_episode"].append(ep.get("tippy_tap_fraction"))
        ep_acc["slip_ratio_mean"].append(ep.get("slip_ratio_mean"))
        o = ep.get("orientation") or {}
        ep_acc["roll_rms"].append(o.get("roll_rms"))
        ep_acc["pitch_rms"].append(o.get("pitch_rms"))
        act = ep.get("actions") or {}
        ep_acc["action_rate"].append(act.get("action_rate_mean_all"))
        ep_acc["sign_reversals"].append(act.get("action_sign_reversals_total"))
    out["by_hold"] = {lab: {**{k: med(v) for k, v in b.items()}, "_n": _ns(b, list(b))} for lab, b in acc.items()}
    out["episode"] = {**{k: med(v) for k, v in ep_acc.items()}, "_n": _ns(ep_acc, list(ep_acc))}
    out["n_episode_json"] = len(ep_acc["tippy_episode"])
    return out


def m_or(x):
    return x.get("median") if isinstance(x, dict) else x


def read_forward_rigidity(stamp) -> dict:
    """The 50 N rigidity set + travel direction of a FORWARD stamp (DERIVED, step 11: forward rows are not category
    suites, so the harness wrote none): GE/metrics.py ``rigidity_metrics`` and ``travel_direction_metrics`` per hold
    window (``segment_id`` == the hold's index, alive steps), the stand hold as category ``stand`` and every moving
    hold as ``fwd``; medians over episodes. The forward holds are 10 s (the omni own hold is 12 s)."""
    np, ge = _np(), _ge()
    meta = run_meta(stamp)
    labels = meta.get("hold_labels") or []
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(stamp) / "raw" / "episode_*.npz"))):
        z = np.load(f)
        done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(done)
        dt = float(z["dt"]) if "dt" in z.files else float(meta.get("dt", 0.02))
        for i, lab in enumerate(labels):
            w = z["segment_id"][:S] == i
            if int(w.sum()) < 50:
                continue
            cat = "stand" if lab == "stand" else "fwd"
            r = ge.rigidity_metrics(cat, dt=dt, foot_force_norm=z["foot_force_norm"][:S][w],
                                    root_pos_w=z["root_pos_w"][:S][w], root_quat_w=z["root_quat_w"][:S][w],
                                    foot_pos_w=z["foot_pos_w"][:S][w], terrain_z=z["terrain_z"][:S][w]) or {}
            b = acc[lab]
            for k in ("support_feet", "steps_per_m", "taps_per_m", "swing_apex", "step_rate_hz", "steps_per_s"):
                b[k].append(r.get(k))
            if cat == "fwd":
                d = ge.travel_direction_metrics(z["root_lin_vel_b"][:S][w][:, :2], z["cmd_applied"][:S][w][:, 0]) or {}
                b["direction_error"].append(d.get("direction_error_rad"))
                b["cross_track_ratio"].append(d.get("cross_track_ratio"))
    out = {}
    for lab, b in acc.items():
        out[lab] = {k: med(v) for k, v in b.items()}
        out[lab]["n"] = len(b["support_feet"])
        spm = out[lab].get("steps_per_m")
        out[lab]["stride_m"] = (6.0 / spm) if spm else None
        out[lab]["_n"] = _ns(b, list(b))
        out[lab]["_n"]["stride_m"] = out[lab]["_n"].get("steps_per_m")
    return {"by_hold": out, "label": DERIVED, "hold_s_note": "forward holds are 10 s (omni own hold 12 s)"}


def ckpt_shapes(path) -> dict:
    """A CPU torch.load (weights only, map_location cpu) of a checkpoint: its std key and the critic input width."""
    import torch  # noqa: PLC0415

    d = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = d.get("model_state_dict") or {}
    std = next((k for k in ("std", "std_raw", "log_std") if k in sd), None)
    cw = sd.get("critic.0.weight")
    return {"std_key": std, "std_shape": list(sd[std].shape) if std else None,
            "critic_in": int(cw.shape[1]) if cw is not None else None,
            "critic_shape": list(cw.shape) if cw is not None else None, "keys": sorted(d)}


# ============================================================================== level reads (Task 3)
def level_set_read(stamps: dict, level_kg: float, th_doc: dict, trained) -> dict:
    """Stable-at-L on one level's rows (payload_read.level_reads, the a-priori ``p3a.stable90.*`` lines) + the
    report-only actuator load on its suite es1 row + the level rows' payload readback (gate_read.payload_problems)."""
    P = _P()
    lr = P.PR().level_reads({k: Path(v) for k, v in stamps.items()}, th_doc, trained_translate=trained)
    act = read_actuator(stamps["suite@1"], level_kg) if "suite@1" in stamps else {"available": False}
    vals = {x["id"]: x["value"] for x in lr["lines"]}
    probs = {}
    for k, v in stamps.items():
        meta = run_meta(v)
        armed = meta.get("payload") not in (None, {})
        p = P.GR.payload_problems(meta, level_kg if armed else None)
        if p:
            probs[k] = p
        if not armed:
            probs.setdefault("_unarmed", []).append(k)
    return {"stable": lr["stable"], "failed": lr["failed"], "skipped": lr["skipped"], "values": vals,
            "lines": lr["lines"], "actuator": act, "payload_problems": {k: v for k, v in probs.items() if k != "_unarmed"},
            "unarmed_rows": probs.get("_unarmed", []), "label": SIM}


def curve_summary(reads: dict) -> dict:
    """The curve over the 45-kg ladder (0, 45, ... 270), walked IN ORDER: max_stable = the last level of an unbroken
    run of stable reads from 0 (payload_read.max_stable's rule); the walk stops at the first unstable level OR at the
    first level not read yet -- then max_stable is None (not claimed) and ``open_from`` names the gap, so a pending
    level below a stable one never inflates the ceiling."""
    curve = {lv: r["stable"] for lv, r in reads.items() if isinstance(r, dict) and "stable" in r}
    ms, first_unstable, open_from = None, None, None
    for lv in LEVELS:
        if lv not in curve:
            open_from = lv
            break
        if not curve[lv]:
            first_unstable = lv
            break
        ms = lv
    complete_ = first_unstable is not None or (open_from is None)
    pend = sorted(lv for lv, r in reads.items() if isinstance(r, dict) and r.get("status") == PEND)
    return {"curve": {str(k): v for k, v in sorted(curve.items())},
            "max_stable": ms if complete_ else None, "stable_through": ms, "first_unstable": first_unstable,
            "open_from": None if complete_ else open_from, "pending_levels": pend,
            "rule": "0..270 kg in 45-kg steps, stop at the first unstable level (PLAN Payload)"}


def h_sag(reads: dict, lv) -> float | None:
    a = ((reads.get(0) or {}).get("values") or {}).get("p3a.stable90.h_meas.suite.es1")
    b = ((reads.get(lv) or {}).get("values") or {}).get("p3a.stable90.h_meas.suite.es1")
    return (a - b) if (a is not None and b is not None) else None


# ============================================================================== logs: wall-clock
_EV = re.compile(r"T4PIPE (\w+) (.*) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)$")


def wallclock(queue_log) -> dict:
    """MEASURED per-stage wall-clock from ``<run>/pipeline_queue.log`` (scratchpad wallclock.py logic): a stage runs
    from its first event to the next stage's first event (or PARKED / STOPPED); PARKED / STOPPED -> the next START is
    a park (a user decision wait), reported separately and never counted as stage time."""
    ev = []
    for line in Path(queue_log).read_text().splitlines():
        m = _EV.match(line.rstrip())
        if not m:
            continue
        st = re.search(r"stage=(\S+)", m.group(2))
        ev.append((datetime.fromisoformat(m.group(3)), m.group(1), st.group(1) if st else None, m.group(2)))
    stages, cur, park, parks = {}, None, None, []
    for t, kind, st, body in ev:
        if kind in ("PARKED", "STOPPED"):
            if cur is not None:
                stages[cur]["segments"][-1][1] = t
            park, cur = (t, ("STOPPED " if kind == "STOPPED" else "") + body), None
            continue
        if kind == "START":
            if park:
                parks.append({"from": park[0].isoformat(), "to": t.isoformat(),
                              "h": (t - park[0]).total_seconds() / 3600, "why": park[1]})
                park = None
            continue
        if st is None:
            continue
        if kind in ("STAGE", "LAUNCH", "ROW", "LIVE", "LEVEL", "GATE", "HALT", "BLOCK", "NOTIFY"):
            s = stages.setdefault(st, {"segments": [], "launch": None, "first_row": None, "seg_launch": []})
            if cur != st:
                if cur is not None:
                    stages[cur]["segments"][-1][1] = t
                s["segments"].append([t, t])
                s["seg_launch"].append(None)
                cur = st
            s["segments"][-1][1] = t
            if kind == "LAUNCH" and s["seg_launch"][-1] is None:
                kv = dict(x.split("=", 1) for x in body.split() if "=" in x)
                s["seg_launch"][-1] = {k: kv[k] for k in ("name", "from", "chain") if k in kv} or {"body": body}
            if kind == "LAUNCH" and s["launch"] is None:
                s["launch"] = t
            if kind == "ROW" and s["first_row"] is None:
                s["first_row"] = t
    out = {}
    for st, s in stages.items():
        h = sum((b - a).total_seconds() for a, b in s["segments"]) / 3600
        out[st] = {"active_h": h, "segments": [[a.isoformat(), b.isoformat()] for a, b in s["segments"]],
                   "segment_info": [{"from": a.isoformat(), "to": b.isoformat(), "h": (b - a).total_seconds() / 3600,
                                     "launch": la} for (a, b), la in zip(s["segments"], s["seg_launch"])],
                   "train_to_first_row_h": ((s["first_row"] - s["launch"]).total_seconds() / 3600
                                            if s["first_row"] and s["launch"] else None)}
    span = (ev[-1][0] - ev[0][0]).total_seconds() / 3600 if ev else None
    return {"stages": out, "parks": parks, "total_active_h": sum(v["active_h"] for v in out.values()),
            "total_park_h": sum(p["h"] for p in parks), "span_h": span,
            "first": ev[0][0].isoformat() if ev else None, "last": ev[-1][0].isoformat() if ev else None,
            "source": rel(queue_log), "rule": wallclock.__doc__.split("\n")[0]}


def preflight_wallclock(pl: Path) -> dict:
    out = {}
    a = pl / "preflight" / "stepA_queue.log"
    if a.is_file():
        ts = [datetime.fromisoformat(m.group(1)) for m in
              (re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[-+]\d\d:\d\d)", ln) for ln in a.read_text().splitlines()
               if ln.startswith(("T4A START", "T4A END"))) if m]
        out["A"] = {"h": (max(ts) - min(ts)).total_seconds() / 3600 if len(ts) >= 2 else None,
                    "source": rel(a), "rule": "T4A START .. T4A END"}
    c = pl / "preflight" / "stepC" / "stepC_queue.log"
    if c.is_file():
        tot = c7 = cpu = 0.0
        n = 0
        planned, first_c7_done = {}, None
        for ln in c.read_text().splitlines():
            j = re.match(r"T4C JOB (\S+) gpu_h=([0-9.]+)", ln)
            if j:
                planned[j.group(1)] = float(j.group(2))
            m = re.match(r"T4C DONE (\S+) rc=\S+ wall_s=(\d+)", ln)
            if m:
                n += 1
                tot += float(m.group(2))
                if m.group(1).startswith("C7"):
                    c7 += float(m.group(2))
                if planned.get(m.group(1)) == 0.0:
                    cpu += float(m.group(2))
                if m.group(1) == "C7_dry" and first_c7_done is None:
                    ts = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)[-+]\d\d:\d\d\s*$", ln)
                    first_c7_done = datetime.fromisoformat(ts.group(1)) if ts else None
        hang = c7_hang(pl, first_c7_done)
        out["C"] = {"h": tot / 3600, "C7_h": c7 / 3600, "n_jobs": n, "source": rel(c),
                    "rule": "sum of T4C DONE wall_s",
                    "hang": hang,
                    "h_excl_hang": (tot / 3600 - hang["h"]) if hang else None,
                    "cpu_check_jobs_h": cpu / 3600,
                    "cpu_check_note": "jobs whose T4C JOB line plans gpu_h=0.00 (CPU-only checks); included in both "
                                      "C figures, not subtracted",
                    "note": ("C7 includes the stop hang of the first dry pass (CHARTER 2026-10-18 defect 1) and its "
                             "5.6-h re-run; the hang was a hung train.py at 100 % CPU, not GPU work"
                             + (f"; measured {hang['h']:.2f} h ({hang['rule']}), excluded from the GPU-h figure"
                                if hang else "; the log does not separate it, so no GPU-h figure is claimed for C"))}
    return out


def c7_hang(pl: Path, c7_done) -> dict | None:
    """The first C7 dry pass's stop hang (CHARTER 2026-10-18 defect 1), measured from the records: its renamed queue
    log (``preflight/stepC/dryspec/dry.c7failed_*/pipeline_queue.log``) must END on the st LIVE injection (the
    injected stop that left train.py hung); from that event nothing was logged until the first ``T4C DONE C7_dry``
    (the 30-h bound). None when either record is absent or the log does not end on st."""
    logs = sorted(glob.glob(str(pl / "preflight" / "stepC" / "dryspec" / "dry.c7failed_*" / "pipeline_queue.log")))
    if not logs or c7_done is None:
        return None
    ev = [m for m in (_EV.match(ln.rstrip()) for ln in Path(logs[-1]).read_text().splitlines()) if m]
    if not ev:
        return None
    last = ev[-1]
    st = re.search(r"stage=(\S+)", last.group(2))
    if not st or st.group(1) != "st":
        return None
    t0 = datetime.fromisoformat(last.group(3))
    h = (c7_done - t0).total_seconds() / 3600
    return {"h": h, "from": t0.isoformat(), "to": c7_done.isoformat(), "last_event": f"{last.group(1)} {last.group(2)}",
            "source": rel(logs[-1]),
            "rule": "the failed first dry pass's last queue event (the st LIVE injection) -> its T4C DONE C7_dry"}


def stamps_span_h(stamps: list) -> dict | None:
    """Measured span of a hand-run row set: the first harness start (run_meta.started_utc) to the last completion
    marker's mtime (Kit boot of the first row not included)."""
    starts, ends = [], []
    for s in stamps:
        try:
            st = run_meta(s).get("started_utc")
            if st:
                starts.append(datetime.fromisoformat(st))
            ends += [datetime.fromtimestamp(os.path.getmtime(Path(s) / m), tz=timezone.utc)
                     for m in COMPLETE_MARKERS if (Path(s) / m).is_file()]
        except Exception:  # noqa: BLE001
            continue
    if not starts or not ends:
        return None
    return {"h": (max(ends) - min(starts)).total_seconds() / 3600, "from": min(starts).isoformat(),
            "to": max(ends).isoformat(), "n_rows": len(stamps),
            "rule": "first run_meta.started_utc -> last completion-marker mtime (Kit boot of the first row excluded)"}


def rows_log(path) -> dict:
    """The step-11 queue sentinels: {id: {state: running|done|skip, rc, why, start, end}} + START / END."""
    out = {"rows": {}, "start": None, "end": None, "source": str(path)}
    if not path or not Path(path).is_file():
        return out
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if len(p) < 2 or p[0] != "S11":
            continue
        t = p[-1]
        kw = dict(x.split("=", 1) for x in p[2:-1] if "=" in x)
        rid = kw.get("id")
        if p[1] == "START":
            out["start"] = t
        elif p[1] == "END":
            out["end"] = t
        elif p[1] == "ROW" and rid:
            out["rows"][rid] = {"state": "running", "start": t}
        elif p[1] == "DONE" and rid:
            r = out["rows"].setdefault(rid, {})
            r.update({"state": "done", "rc": int(kw.get("rc", "-1")), "end": t})
        elif p[1] == "SKIP" and rid:
            out["rows"][rid] = {"state": "skip", "why": kw.get("why"), "end": t}
    gpu = 0.0
    n_both = 0
    for r in out["rows"].values():
        if r.get("start") and r.get("end"):
            n_both += 1
            gpu += (datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).total_seconds() / 3600
    out["measured_h"] = gpu
    out["n_rows_measured"] = n_both
    out["n_rows_by_state"] = {s: sum(1 for r in out["rows"].values() if r.get("state") == s)
                              for s in ("done", "running", "skip")}
    out["measured_rule"] = "only rows with BOTH an S11 ROW (start) and an S11 DONE (end) sentinel"
    return out


def hardware() -> dict:
    out = {"host": socket.gethostname()}
    try:
        g = sorted(glob.glob("/proc/driver/nvidia/gpus/*/information"))
        out["gpus"] = [re.search(r"Model:\s*(.*)", Path(x).read_text()).group(1).strip() for x in g]
    except Exception:  # noqa: BLE001
        out["gpus"] = None
    try:
        out["cpu"] = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                          if ln.startswith("model name"))
        out["cpu_threads"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        out["cpu"] = None
    try:
        kb = int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1))
        out["mem_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001
        out["mem_gb"] = None
    out["note"] = "read at report time on the machine that ran both runs (one GPU, serial queue)"
    return out


def invalid_kinds(reasons) -> dict:
    """Classify a reader's INVALID reasons (sh_read / sn_calib ``invalid`` lists) without re-deciding them: which
    reference rows they compare against (``REFS_S1_s18``, ``REFS``, ...) and which identity checks differ."""
    reasons = [str(x) for x in reasons or []]
    kinds = set()
    refs = set()
    for x in reasons:
        if "s1_checkpoint_sha256" in x:
            m = re.search(r"s1_checkpoint_sha256 (\w+) != (\S+) (\w+)", x)
            kinds.add(f"p2r_mode.s1_checkpoint_sha256 {m.group(1)[:8]} != {m.group(2)} {m.group(3)[:8]}" if m
                      else "p2r_mode.s1_checkpoint_sha256")
        if "code_sha" in x:
            kinds.add("code_sha / eval_sha")
            m = re.search(r"differ from (\S+?)'s", x)
            if m:
                refs.add(m.group(1))
    return {"n": len(reasons), "kinds": sorted(kinds), "references": sorted(refs),
            "first": reasons[0] if reasons else None}


def git_ignored(paths) -> dict:
    """{path: the matching ignore rule 'file:line:pattern'} for the paths git ignores (``git check-ignore -v``)."""
    paths = [str(p) for p in paths or []]
    if not paths:
        return {}
    try:
        r = subprocess.run(["git", "-C", str(L.REPO), "check-ignore", "-v", "--", *paths], capture_output=True,
                           text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for ln in r.stdout.splitlines():
        if "\t" in ln:
            rule, p = ln.split("\t", 1)
            out[rel(p) if p.startswith("/") else p] = rule.replace(str(L.REPO) + "/", "")
    return out


def git_tags(patterns=("task4-pin-*", "omni-sim2real-baseline-*")) -> list:
    out = []
    for pat in patterns:
        try:
            r = subprocess.run(["git", "-C", str(L.REPO), "tag", "-l", pat, "--format=%(refname:short) %(objectname:short) "
                                "%(*objectname:short) %(contents:subject)"], capture_output=True, text=True, timeout=20)
            out += [ln for ln in r.stdout.splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            pass
    return out


# ============================================================================== the run records (as before)
def run_records(spec_path: Path, run: str) -> dict:
    P = _P()
    ctx = P.Ctx.open(spec_path, run)
    gates, heads = {}, {}
    for st in ctx.spec.stages(run):
        sid = st["id"]
        g = L.read_json(ctx.dir / "gates" / f"{sid}.json")
        if g:
            gates[sid] = g
        h = ctx.head(sid)
        if h:
            heads[sid] = h
    drv = L.read_json(ctx.dir / "state.json", {}) or {}
    wall = {}
    for bucket in (drv.get("keys") or {}).values():
        for k, v in bucket.items():
            if isinstance(v, dict) and v.get("wall_h") is not None:
                wall[k] = v["wall_h"]
    return {"run": run, "state": ctx.state, "gates": gates, "heads": heads, "wall_h_driver": wall,
            "halts": {s: L.read_json(L.REPO / h["record"]) for s, h in (ctx.state.get("halts") or {}).items()},
            "deliverable": L.read_json(ctx.dir / "deliverable.json"), "ctx": ctx}


def column_t1(rec: dict) -> dict:
    """T1 / St numbers from the deliverable run's gate records (reported WATCH reads included)."""
    out = {}
    ts = (rec["state"].get("tstar") or {})
    if ts:
        g = rec["gates"].get(ts.get("stage")) or {}
        out["tstar"] = {"stage": ts.get("stage"), "max_stable": ts.get("max_stable"), "curve": g.get("curve"),
                        "limiting_factor": g.get("limiting_factor"), "zero_kg": g.get("zero_kg"),
                        "label": SIM}
        out["tstar_watch"] = g.get("watch")
    p2a = rec["gates"].get("p2a") or {}
    out["p2a_watch"] = p2a.get("watch")
    sel = rec["gates"].get("sel") or {}
    out["student"] = {"selected": rec["state"].get("selected_student"), "blocks": sel.get("blocks")}
    out["stand"] = {k: (rec["gates"].get(k) or {}).get("decisive") for k in ("sn", "sh")}
    return out


def provenance(rec: dict) -> dict:
    st = rec["state"]
    verdicts = {s: (v.get("gate") or {}).get("verdict") for s, v in (st.get("stages") or {}).items()}
    accepted = [{**a, "record": (rec["halts"].get(a["stage"]) or {}).get("lines")} for a in st.get("accepted") or []]
    return {"pin": st.get("pin"), "verdicts": verdicts, "accepted_halts": accepted,
            "amendments": st.get("amendments"), "wall_h_driver_training_only": rec["wall_h_driver"],
            "wall_h_driver_total": sum(rec["wall_h_driver"].values()) if rec["wall_h_driver"] else None}


# ============================================================================== the builder
class Report:
    """Locates every source (records first, then the step-11 rows by name / scenario / seed), reads it on CPU,
    and records one cell per fill-plan cell: ``filled`` (read), ``pending`` (its row is not on disk yet) or ``na``
    (cannot exist, with the reason)."""

    def __init__(self, spec_path: Path, run: str, other: str | None = None, rows_log_path=None):
        self.P = _P()
        self.spec_path = Path(spec_path).resolve()
        self.spec = self.P.Spec(self.spec_path)
        self.pl = self.spec.dir
        self.run, self.other = run, other
        self.th_path = self.spec.thresholds_path()
        self.th_doc = L.read_json(self.th_path) or {"lines": []}
        self.row_out = L.REPO / self.spec.d.get("row_out", "parkour/outputs/omni_plan/task4")
        self.s11_root = self.row_out / STEP11 / "lr"
        self.stepc_root = self.row_out / "stepC" / "lr"
        self.body_anchor_root = self.row_out.parent / "body_cmd" / "lr"
        self.p2_lr = L.REPO / (self.spec.d.get("stand") or {}).get("p2_lr", "parkour/outputs/omni_plan/phase2/lr")
        sh_scripts = (self.spec.d.get("stand") or {}).get("sh_scripts")
        self.sh_dir = (L.OMNI / sh_scripts).parent if sh_scripts else None
        self.rows_log = rows_log(rows_log_path)
        self.cells: list = []
        self.errors: list = []
        self.memo: dict = {}

    # -- cached reads
    def get(self, fn, *args):
        key = (fn.__name__,) + tuple(str(a) for a in args)
        if key not in self.memo:
            try:
                self.memo[key] = {"ok": True, "v": fn(*args)}
            except Exception as exc:  # noqa: BLE001  (an unreadable source is reported, never a number)
                self.memo[key] = {"ok": False, "v": None, "err": f"{fn.__name__}({_clip(', '.join(map(str, args)), 160)}): "
                                                                 f"{type(exc).__name__}: {_clip(str(exc), 200)}"}
                self.errors.append(self.memo[key]["err"])
        r = self.memo[key]
        return r["v"] if r["ok"] else {"unreadable": r["err"]}

    # -- cells
    def cell(self, cid, section, metric, column, status, *, value=None, sources=(), reader=None, label=None,
             note=None, why=None, rows=()):
        if status == FILLED and _has_unreadable(value):
            status, why = NOTAPP, "read failed (see read_errors): " + _clip("; ".join(_unreadable(value)), 300)
        if status == PEND and not why:
            why = f"{PENDING} (its record is not written yet)"
        if status == NOTAPP and not why:
            why = "not obtainable (its record is absent)"
        c = {"id": cid, "section": section, "metric": metric, "column": column, "status": status,
             "sources": [rel(s) if isinstance(s, (str, Path)) and str(s).startswith("/") else s for s in sources if s],
             "reader": reader, "label": label, "note": note, "why": why, "pending_rows": list(rows), "value": value}
        self.cells.append({k: v for k, v in c.items() if v not in (None, [], ())} | {"status": status})
        return c

    @staticmethod
    def status_of(srcs: list) -> tuple:
        """(status, why, pending rows) of a cell over its sources: any pending -> pending; all N/A -> na."""
        srcs = [s if s is not None else _src(status=PEND, why=f"{PENDING} (no run record yet)") for s in srcs]
        if not srcs:
            srcs = [_src(status=PEND, why=f"{PENDING} (no run record yet)")]
        pend = [s for s in srcs if s.get("status") == PEND]
        na = [s for s in srcs if s.get("status") == NOTAPP]
        if pend:
            return PEND, "; ".join(sorted({_why(s) for s in pend})), [s["row"] for s in pend if s.get("row")]
        if srcs and len(na) == len(srcs):
            return NOTAPP, "; ".join(sorted({_why(s) for s in na})), []
        return FILLED, ("; ".join(sorted({_why(s) for s in na})) if na else None), []

    # -- sources
    def s11(self, rid: str) -> dict:
        name, sc, es = S11_ROWS[rid]
        st = newest_stamp(self.s11_root, name, sc, es)
        if st is not None:
            return _src(st, row=rid)
        lg = self.rows_log["rows"].get(rid) or {}
        if lg.get("state") == "skip":
            return _src(status=NOTAPP, why=f"not obtainable: row {rid} skipped by the step-11 queue ({lg.get('why')})",
                        row=rid)
        if lg.get("state") == "done" and lg.get("rc", 0) != 0:
            return _src(status=NOTAPP, why=f"not obtainable: row {rid} ran and failed (rc {lg.get('rc')}, no "
                                           "complete stamp)", row=rid)
        return _src(status=PEND, row=rid)

    def stepc(self, name, sc, es) -> dict:
        st = newest_stamp(self.stepc_root, name, sc, es)
        return _src(st) if st else _src(status=NOTAPP, why=f"no stepC stamp {name}/{sc}/es{es}")

    def recorded(self, path) -> dict:
        if path and complete(path):
            return _src(path)
        if path and Path(path).is_dir() and (Path(path) / "scenario_metrics.json").is_file():
            return _src(path)
        return _src(status=PEND, why=f"{PENDING} (no record {rel(path) if path else '?'})")

    def run_sources(self, run: str) -> dict:
        """T1 / St / level / body sources of one run from ITS records (stage_reads/*.rows.json, the driver stamp, the
        selection record) plus, for the deliverable run, the step-11 rows."""
        ctx = self.P.Ctx(self.spec, run)
        st = ctx.state
        ts = st.get("tstar") or {}
        tstage = ts.get("stage")
        out = {"run": run, "ctx": ctx, "tstar": ts, "tstage": tstage, "T1": {}, "St": {}, "TS": {}, "levels": {},
               "st_levels": {}, "sel": st.get("selected_student") or {}}
        if tstage:
            head = ctx.head(tstage) or {}
            rows = L.read_json(ctx.rows_path(tstage)) or {}
            out["T1"]["suite@1"] = self.recorded(head.get("driver_stamp") or ts.get("driver_stamp"))
            for k, v in rows.items():
                p = k.split("@")
                if len(p) == 2:
                    out["T1"][k] = self.recorded((v or {}).get("stamp"))
                elif len(p) == 3 and p[2].startswith("L"):
                    out["levels"].setdefault(int(p[2][1:]), {})[f"{p[0]}@{p[1]}"] = self.recorded((v or {}).get("stamp"))
        cand = out["sel"].get("cand")
        srows = L.read_json(ctx.rows_path("sel")) or {}
        for k, v in srows.items():
            who, _, row = k.partition(":")
            p = row.split("@")
            if who == cand and len(p) == 2:
                out["St"][row] = self.recorded((v or {}).get("stamp"))
            elif who == cand and len(p) == 3:
                out["st_levels"].setdefault(int(p[2][1:]), {})[f"{p[0]}@{p[1]}"] = self.recorded((v or {}).get("stamp"))
            elif who == "tstar" and len(p) == 2:
                out["TS"][row] = self.recorded((v or {}).get("stamp"))
        if cand:
            out["st_levels"].setdefault(0, {})
            for r in ("suite@1", "suite@2", "field_light@1", "field_light@2"):
                if r in out["St"]:
                    out["st_levels"][0][r] = out["St"][r]
        return out

    # ------------------------------------------------------------------ build
    def build(self) -> dict:
        P = self.P
        self.rec = run_records(self.spec_path, self.run)
        self.F = self.run_sources(self.run)
        self.R = self.run_sources(self.other) if self.other else None
        trained = self.trained(self.F)
        doc = {"schema": 2, "run": self.run, "other": self.other, "generated_at": L.now_iso(),
               "spec": rel(self.spec_path), "thresholds": {"file": rel(self.th_path), "version": self.th_doc.get("version"),
                                                         "sha256": L.sha256_file(self.th_path) if self.th_path.is_file() else None},
               "labels": {"sim": SIM, "derived": DERIVED, "misread": MISREAD, "pending": PENDING, "na": NA},
               "columns": self.columns(), "rows_log": {k: v for k, v in self.rows_log.items() if k != "rows"} |
               {"rows": self.rows_log["rows"]}}
        doc["T1"] = column_t1(self.rec)
        doc["task0"] = self.task0(trained)
        doc["task2"] = self.task2()
        doc["task3"] = self.task3(trained)
        doc["stand"] = self.stand()
        doc["student"] = self.student()
        doc["provenance"] = self.provenance_all()
        doc["reconciliation"] = self.reconciliation()
        doc["disclosures"] = self.disclosures()
        if self.R:
            doc["appendix"] = self.appendix(self.trained(self.R))
        doc["cells"] = self.cells
        doc["counts"] = {s: sum(1 for c in self.cells if c["status"] == s) for s in (FILLED, PEND, NOTAPP)}
        doc["read_errors"] = self.errors
        doc["headline"] = self.headline(doc)
        doc["F"] = {"pending": [c["id"] for c in self.cells if c["column"] == "F" and c["status"] == PEND]}
        doc["H"] = {"note": "v1 as committed; not re-runnable"}
        return doc

    def trained(self, src) -> list:
        s = (src or {}).get("T1", {}).get("suite@1") if src else None
        if _ok(s):
            tc = run_meta(s["path"]).get("trained_categories") or []
            return [c for c in tc if c in TRANSLATE]
        return list(TRANSLATE)

    def columns(self) -> dict:
        ts, sel = self.F["tstar"], self.F["sel"]
        return {
            "H": "v1 stand-ins as committed (FWD/experiments/eval/baselines/v1: teacher 21100 / student 29098 / "
                 "flat-walk 19999); the grant's 6300 / 9800 shown as NOT loadable; not re-runnable",
            "F": f"forward 3a student model_24995 (FWD/{FWD_POLICY}) on its own task (Isaac-Crab-Hex-Student-v0, "
                 "STUDENT_MDP=1) and forward rows; lateral / yaw / grid / sweep N/A (no command channel), not 0",
            "T0": "S1 s18 (X08 model_19994 == policy/6_p2a_field) + SH_T17 (policy/stand_net/model_5000.pt): the "
                  "recorded anchors (pin-build stepC stamps bitwise to REFS_S1_s18; B0b payload curve)",
            "T1": f"T* = {ts.get('stage')} {Path(ts.get('ckpt') or '?').name} (sha {str(ts.get('sha256'))[:8]}) of "
                  f"{self.run}, Walk-Play rows",
            "St": f"student {sel.get('cand')} (sha {str(sel.get('sha256'))[:8]}) of {self.run}, Student-Play "
                  "--policy-role student",
            "R18": f"the other run ({self.other}) in the appendix" if self.other else None,
        }

    # ------------------------------------------------------------------ Task 0
    def t0_sources(self) -> dict:
        S = {"suite@1": self.stepc("T4C2_S1", "suite__omni", 1), "suite@2": self.stepc("T4C2_S1", "suite__omni", 2),
             "side22@1": self.stepc("T4C2_S1", "side22_25lr__omni", 1)}
        for r in FIELD_ROWS:
            for es in (1, 2):
                S[f"{r}@{es}"] = self.stepc("T4C2_S1", f"{r}__omni", es)
        for fam, sc in (("suite_body", "suite__omni_body"), ("turnlevels_body", "turnlevels__omni_body"),
                        ("grid_body", "grid__omni_body")):
            S[f"{fam}@1"] = self.stepc("T4C2_S1", sc, 1)
            a2 = newest_stamp(self.body_anchor_root, "ANC_S1_s18", sc, 2) or newest_in(
                self.body_anchor_root / "ANC_S1_s18" / sc / "seed002")
            S[f"{fam}@2"] = _src(a2) if a2 else _src(status=NOTAPP, why=f"no recorded anchor ANC_S1_s18/{sc}/es2")
        c5 = L.read_json(self.pl / "preflight/stepC/checks/C5.json") or {}
        for es in (1, 2):
            p = ((c5.get("anchors") or {}).get(f"s18_es{es}") or {}).get("stamp")
            S[f"dirsweep_body@{es}"] = self.recorded(p) if p else _src(status=NOTAPP, why="no C5 anchor")
        return S

    def final_body(self, col: str, src: dict) -> dict:
        """The deliverable run's body-family sources: the step-11 rows (T1 = T4S11_tstar, St = T4S11_c28000)."""
        pre = {"T1": "T1", "St": "ST"}[col]
        return {f"{fam}@{es}": self.s11(f"{pre}_{fam}_{es}") for fam in BODY_FAMS for es in (1, 2)}

    def task0(self, trained) -> dict:
        sec = "Task 0"
        T0 = self.t0_sources()
        self.T0 = T0
        c2 = L.read_json(self.pl / "preflight/stepC/checks/C2.json") or {}
        c2_suite = ((c2.get("families") or {}).get("suite") or {}).get("stamp")
        cols = {"T0": T0, "T1": self.F["T1"], "St": self.F["St"]}
        out = {"columns": {}}
        notes = {"T0": f"pin-build stepC stamps (es1 {'==' if c2_suite and T0['suite@1']['path'] == c2_suite else '!='} "
                       "the C2 record's suite stamp, bitwise to REFS_S1_s18 es1)",
                 "T1": "T* at 0 kg, unarmed Walk-Play rows; suite@1 is the driver stamp",
                 "St": "Student-Play --policy-role student; T* on the same Student-Play rows is the pairing (Student section)"}
        for col, S in cols.items():
            ss = [S.get("suite@1"), S.get("suite@2")]
            status, why, rows = self.status_of(ss)
            res = {}
            for es, s in ((1, ss[0]), (2, ss[1])):
                if _ok(s):
                    p = s["path"]
                    res[es] = {"stamp": rel(p), "categories": self.get(read_categories, p),
                               "episode": self.get(read_episode_extras, p), "vel_rms": self.get(read_vel_rms, p),
                               "orientation": self.get(read_orientation, p), "actuator": self.get(read_actuator, p, 0.0)}
            out["columns"][col] = res
            srcs = [s["path"] for s in ss if _ok(s)]
            self.cell(f"t0.cat.{col}", sec, "per-category set: tripod (1 N), 50 N rigidity (support_feet, steps_per_m, "
                      "stride ~ 6/steps_per_m, taps_per_m, swing_apex, step_rate_hz), tracking ratio, direction_error, "
                      "cross_track, action jitter / fft, joint / rod speed, world turning", col, status,
                      value={es: (r["categories"] or {}).get("by_category") for es, r in res.items()}, sources=srcs,
                      reader="run_phases.cat_summary + thresholds.suite_categories", note=notes[col], why=why, rows=rows)
            self.cell(f"t0.ep.{col}", sec, "duty factor (1 N), 1 N air time + tippy-tap (misread caveat), action rate "
                      "+ sign reversals", col, status, value={es: r["episode"] for es, r in res.items()}, sources=srcs,
                      reader="metrics/episode_*.json (duty only on valid windows: None, never 0)", label=MISREAD,
                      why=why, rows=rows)
            self.cell(f"t0.rms.{col}", sec, "vy / wz tracking RMS per own hold (achieved vs commanded, body frame)", col,
                      status, value={es: r["vel_rms"] for es, r in res.items()}, sources=[f"{s}/raw" for s in srcs],
                      reader="read_vel_rms (raw npz, own hold x steady_mask)", label=DERIVED, why=why, rows=rows)
            self.cell(f"t0.orient.{col}", sec, "roll / pitch RMS, survival per category", col, status,
                      value={es: {"orientation": r["orientation"],
                                  "survival": {c: (v or {}).get("survival") for c, v in
                                               ((r["categories"] or {}).get("by_category") or {}).items()}}
                             for es, r in res.items()}, sources=srcs,
                      reader="thresholds.h_meas_and_orientation / suite_categories", why=why, rows=rows)
            self.cell(f"t0.act.{col}", sec, "torque (effort-cap share), rod force at 0 kg", col, status,
                      value={es: r["actuator"] for es, r in res.items()}, sources=srcs,
                      reader="payload_read.actuator_load(stamp, 0)", label=SIM + "; applied_torque = implicit-actuator "
                      "estimate", why=why, rows=rows)
        out["H"] = self.task0_h()
        out["F"] = self.task0_f()
        return out

    def task0_h(self) -> dict:
        sec = "Task 0"
        v1 = L.FWD / "experiments/eval/baselines/v1"
        res = {}
        for name in V1_STAMPS:
            d = v1 / name
            res[name] = self.get(read_forward, d) if (d / "scenario_metrics.json").is_file() else None
        ok = [n for n, r in res.items() if r and "unreadable" not in r]
        self.cell("t0.H.v1", sec, "tripod, 1 N tippy / air time, duty, slip, completion, per-hold tracking vx "
                  "(+ blind vy / wz RMS), action rate / reversals, roll / pitch RMS", "H",
                  FILLED if ok else NOTAPP, value=res, sources=[str(v1 / n) for n in ok],
                  reader="aggregate + metrics/episode_*.json (forward per-hold tracking block)",
                  note="v1 stand-ins as committed (21100 / 29098 / 19999); forward-only, not re-runnable (critic input "
                       "1151 vs 1149, baselines/v1/README.md); vy / wz channels are blind (not capability reads)",
                  label=MISREAD, why=None if ok else "v1 stamps missing")
        self.cell("t0.H.na", sec, "50 N rigidity set, direction_error / cross_track, torque / cap share", "H", NOTAPP,
                  why="not obtainable: the v1 record has no raw npz and no 50 N keys, and the checkpoints do not load")
        grant = {}
        old = L.FWD / "experiments/old-runs"
        today = self.F["tstar"].get("ckpt")
        ref = self.get(ckpt_shapes, today) if today and Path(today).is_file() else None
        for k, p in GRANT_CKPTS.items():
            f = old / p
            if f.is_file():
                s = self.get(ckpt_shapes, f)
                lo = (isinstance(s, dict) and isinstance(ref, dict) and "unreadable" not in s
                      and s.get("std_key") == ref.get("std_key") and s.get("critic_in") == ref.get("critic_in"))
                grant[k] = {**(s or {}), "loadable_today": bool(lo), "file": rel(f)}
        tb = None
        rd = old / GRANT_9800_README
        if rd.is_file():
            tb = next((ln.strip() for ln in rd.read_text().splitlines() if "TensorBoard" in ln and "9800" in ln), None)
        self.cell("t0.H.grant", sec, "the grant's teacher 6300 / student 9800", "H", FILLED if grant else NOTAPP,
                  value={"checkpoints": grant, "today_reference": {"ckpt": rel(today), **(ref or {})},
                         "readme_tensorboard_9800": tb},
                  sources=[str(old / p) for p in GRANT_CKPTS.values()] + [str(rd)],
                  reader="CPU torch.load(map_location='cpu', weights_only=True) + the 9800 README TensorBoard line",
                  note="NOT LOADABLE on today's tree: std key / critic input width differ from today's (pre-cam plant, "
                       "2026-05-23 USD); quote this check, not baselines/v1/README.md's era-A 'they load' sentence")
        return {"v1": res, "grant": grant, "readme_9800": tb}

    def task0_f(self) -> dict:
        sec = "Task 0"
        c1 = L.read_json(self.pl / "preflight/stepC/checks/C1.json") or {}
        pin = ((c1.get("rows") or {}).get("C1_fwd3a_cur") or {}).get("stamp")
        S = {"slow_pin": self.recorded(pin) if pin else self.stepc("T4C1_fwd3a_cur", "slow__A15pB", 1)}
        for k, p in FWD_RECORDED.items():
            d = newest_in(L.LOG_ROOT / p)
            S[f"{k}_recorded"] = _src(d) if d else _src(status=NOTAPP, why=f"no recorded stamp under {p}")
        S["step_pin"] = self.s11("F_step_pin")
        self.Fsrc = S
        res = {}
        for k in ("slow_pin", "slow_recorded", "step_recorded", "step_pin"):
            if _ok(S[k]):
                res[k] = {"stamp": rel(S[k]["path"]), "basic": self.get(read_forward, S[k]["path"]),
                          "rigidity": self.get(read_forward_rigidity, S[k]["path"])}
        step_note = ("step row: the F_step_pin re-read at the pin" if _ok(S["step_pin"]) else
                     "step row: recorded 2026-09-08 build, not re-read at the pin (no plant key, no code_state.json); "
                     f"pin re-read {_why(S['step_pin'])}")
        st_basic = FILLED if _ok(S["slow_pin"]) else PEND
        self.cell("t0.F.basic", sec, "tripod, 1 N tippy / air time, duty, completion / falls, tracking vx, roll / pitch "
                  "RMS, action rate / reversals", "F", st_basic,
                  value={k: v["basic"] for k, v in res.items()}, sources=[S[k]["path"] for k in res],
                  reader="aggregate + metrics/episode_*.json", label=MISREAD,
                  note="forward 3a model_24995 (sha e79c4a90); slow row read at the pin (T4C1_fwd3a_cur; 0.75 vs the "
                       "recorded 0.79 = the +-0.04 process-state split); " + step_note +
                       "; lateral / turn categories N/A (no command channel), not 0")
        self.cell("t0.F.rig", sec, "50 N rigidity set + direction_error / cross_track", "F", st_basic,
                  value={k: v["rigidity"] for k, v in res.items()}, sources=[f"{S[k]['path']}/raw" for k in res],
                  reader="GE/metrics.py rigidity_metrics + travel_direction_metrics per hold window", label=DERIVED,
                  note="forward rows are not category suites (no harness rigidity keys); holds are 10 s vs the omni 12 s; "
                       + step_note)
        act = self.get(read_actuator, S["slow_pin"]["path"], 0.0) if _ok(S["slow_pin"]) else None
        self.cell("t0.F.act", sec, "torque, effort-cap share, rod force at 0 kg", "F", FILLED if act else PEND,
                  value=act, sources=[S["slow_pin"]["path"]] if act else (), reader="payload_read.actuator_load(stamp, 0)",
                  label=SIM + "; same A15+B joint names")
        return {"sources": {k: v for k, v in S.items()}, "reads": res, "actuator_0kg": act}

    # ------------------------------------------------------------------ Task 2
    def task2(self) -> dict:
        sec = "Task 2"
        out = {}
        cols = {"T0": self.T0, "T1": self.final_body("T1", self.F), "St": self.final_body("St", self.F)}
        st_note = ("post hoc WATCH-class read at step 11: the student selection gate never read body / grid / turn-level "
                   "/ sweep rows (Disclosures -> selection on world rows only)")
        for fam, metric, reader in (
                ("grid_body", "command grid: 9 in-envelope cells + 11 reported, |achieved - cmd|, heading drift "
                              "median / p90", "read_grid (cat_summary BODY keys + run_meta commands)"),
                ("turnlevels_body", "turn levels: net ratio at w_min / 0.08 / 0.15 / ceiling / w_max",
                 "run_phases.cat_summary + BODY_CAT_METRIC_KEYS"),
                ("suite_body", "body suite through body_hold (per-category ratio, survival, body_ratio, cross_body)",
                 "run_phases.cat_summary + BODY_CAT_METRIC_KEYS"),
                ("dirsweep_body", "16-bin direction sweep at 0.22 m/s with discontinuity (max adjacent-bin jump)",
                 "sweep_read (thresholds sweep WATCH lines p2a.watch.sweep.*)")):
            out[fam] = {}
            for col, S in cols.items():
                ss = [S.get(f"{fam}@1"), S.get(f"{fam}@2")]
                status, why, rows = self.status_of(ss)
                okp = {es: s["path"] for es, s in ((1, ss[0]), (2, ss[1])) if _ok(s)}
                if fam == "grid_body":
                    val = {es: self.get(read_grid, p) for es, p in okp.items()}
                elif fam == "dirsweep_body":
                    val = self.get(sweep_read_paths, json.dumps(okp), self.th_path) if okp else None
                else:
                    val = {es: (self.get(read_categories, p) or {}).get("by_category") for es, p in okp.items()}
                out[fam][col] = val
                self.cell(f"t2.{fam}.{col}", sec, metric, col, status, value=val, sources=list(okp.values()),
                          reader=reader, why=why, rows=rows,
                          label=(DERIVED + " for |achieved - cmd| and the pooled drift p90") if fam == "grid_body" else
                          ("WATCH only" if fam == "dirsweep_body" else None),
                          note=(st_note if col == "St" else
                                ("es1 pin-build stepC stamp bitwise to the recorded anchor; es2 the recorded anchor "
                                 "(ANC_S1_s18 / C5 s18)" if col == "T0" else
                                 "T* on the deliverable run's p3b env through body_hold (step-11 rows)")))
        # field rows + side22
        cols = {"T0": self.T0, "T1": self.F["T1"], "St": self.F["St"]}
        out["field"], out["side22"] = {}, {}
        for col, S in cols.items():
            ss = [S.get(f"{r}@{es}") for r in FIELD_ROWS for es in (1, 2)]
            status, why, rows = self.status_of(ss)
            val = {}
            for r in FIELD_ROWS:
                ps = [S[f"{r}@{es}"]["path"] for es in (1, 2) if _ok(S.get(f"{r}@{es}"))]
                val[r] = self.get(read_field, *ps) if ps else None
            out["field"][col] = val
            self.cell(f"t2.field.{col}", sec, "field rows: field_hard / field_light / ring_obst / field_turn completion "
                      "es1-2", col, status, value=val, sources=[s["path"] for s in ss if _ok(s)],
                      reader="thresholds.field_completion", why=why, rows=rows,
                      note="Student-Play --policy-role student; the paired read is in the Student section" if col == "St"
                      else None)
            s = S.get("side22@1")
            stt, why2, rows2 = self.status_of([s])
            v = self.get(read_side22, s["path"]) if _ok(s) else None
            out["side22"][col] = v
            self.cell(f"t2.side22.{col}", sec, "side22 FAST / FAST", col, stt, value=v,
                      sources=[s["path"]] if _ok(s) else (), reader="thresholds.side22_read + side22_classify",
                      why=why2, rows=rows2)
        self.cell("t2.field_suite.na", sec, "field_suite (per-category field read)", "T1 / St", NOTAPP,
                  why="not run by choice: not in the PLAN step-11 / DESIGN 6 report list (a p2a-only row family); T0's "
                      "field_suite stamp has no pair, so it is not tabled")
        self.cell("t2.H.na", sec, "all Task 2 cells", "H", NOTAPP,
                  why="not obtainable as a capability read: the v1 vy / wz channels are blind (v1 README 'Off-axis "
                      "probes'; student_flat_probe_vy / _yaw recorded, not gated)")
        self.cell("t2.F.na", sec, "grid, turn levels, body suite, sweep, side22, omni field rows", "F", NOTAPP,
                  why="N/A, not 0: the forward student has no lateral / yaw command channel (DESIGN 6)")
        fa = {}
        for k in ("obstacle_recorded", "hard_band_recorded"):
            s = self.Fsrc.get(k)
            if _ok(s):
                a = self.get(aggregate, s["path"])
                fa[k] = {"stamp": rel(s["path"]), "completion": (a or {}).get("schedule_completion_rate"),
                         "termination_reasons": (a or {}).get("termination_reasons"), "n": (a or {}).get("n_episodes")}
        self.cell("t2.F.field", sec, "forward field analogue (recal2b2w obstacles 0.20-0.70, hard band 0.70-0.90)", "F",
                  FILLED if fa else NOTAPP, value=fa, sources=[self.Fsrc[k]["path"] for k in fa],
                  reader="aggregate.schedule_completion_rate / termination_reasons",
                  note="recorded 2026-09-08/09 build, not re-read at the pin (same checkpoint e79c4a90; run_meta has no "
                       "plant key and env_vars {}; the obstacle launcher is not recorded, so no pin row can be modelled)")
        return out

    # ------------------------------------------------------------------ Task 3
    def levels_read(self, levels: dict, trained, *, upto=None) -> dict:
        reads = {}
        for lv in sorted(levels):
            S = levels[lv]
            need = ("suite@1", "suite@2", "field_light@1", "field_light@2")
            ss = [S.get(k) for k in need]
            status, why, rows = self.status_of(ss)
            if status != FILLED or not all(_ok(s) for s in ss):
                reads[lv] = {"status": status if status != FILLED else NOTAPP, "why": why or "incomplete level rows",
                             "rows": rows}
                continue
            r = self.get(level_set_read_paths, json.dumps({k: S[k]["path"] for k in need}), lv, self.th_path,
                         ",".join(trained))
            reads[lv] = r if isinstance(r, dict) else {"unreadable": str(r)}
            reads[lv]["stamps"] = {k: rel(S[k]["path"]) for k in need}
        return reads

    def task3(self, trained) -> dict:
        sec = "Task 3 (every number 'sim plant, force side not enforced')"
        out = {"label": SIM}
        # T0: the B0b S1 curve (records: C3_S1_curve.json)
        c3 = L.read_json(self.pl / "preflight/stepC/checks/C3_S1_curve.json") or {}
        t0_levels = {int(lv): {k: self.recorded(v) for k, v in (r.get("stamps") or {}).items()}
                     for lv, r in ((c3.get("S1_curve") or {}).get("reads") or {}).items()}
        t0 = self.levels_read(t0_levels, trained)
        t0s = curve_summary(t0)
        lf0 = None
        fu = t0s.get("first_unstable")
        if fu is not None and isinstance(t0.get(fu), dict) and "failed" in t0[fu]:
            lf0 = _P().PR().limiting_factor(t0[fu], budget=None, actuator_L=t0[fu].get("actuator"),
                                            actuator_0=(t0.get(0) or {}).get("actuator"), h_sag=h_sag(t0, fu))
            lf0["note"] = "computed at step 11 on CPU (the record stores none); budget-bound not assessed (S1 is not a payload stage)"
        rec_match = (t0s["curve"] == {str(k): v for k, v in ((c3.get("S1_curve") or {}).get("curve") or {}).items()})
        out["T0"] = {"levels": t0, **t0s, "limiting_factor": lf0, "matches_record": rec_match,
                     "record": rel(self.pl / "preflight/stepC/checks/C3_S1_curve.json")}
        self.cell("t3.curve.T0", sec, "0-270 kg curve (45-kg steps, stop at first unstable), max_stable, limiting "
                  "factor, actuator load per level (cap share, rod force vs rated / self-lock, h_meas sag, sum foot "
                  "force)", "T0", FILLED if t0 else PEND, value=out["T0"],
                  sources=[self.pl / "preflight/stepC/checks/C3_S1_curve.json"],
                  reader="payload_read.level_reads / max_stable / limiting_factor / actuator_load", label=SIM,
                  note=f"B0b; recomputed curve {'==' if rec_match else '!='} the C3 record",
                  why=None if t0 else f"{PENDING} (no preflight C3_S1_curve record)")
        # T1: T*'s curve from the run's level rows
        g = (self.rec["gates"].get(self.F["tstage"]) or {}) if self.F["tstage"] else {}
        t1 = self.levels_read(self.F["levels"], trained)
        t1s = curve_summary(t1)
        out["T1"] = {"levels": t1, **t1s, "gate_curve": g.get("curve"), "gate_max_stable": g.get("max_stable"),
                     "limiting_factor": g.get("limiting_factor"),
                     "matches_gate": (t1s["curve"] == (g.get("curve") or {})) if g else None,
                     "n_p": self.spec.n_p(), "note": f"max_stable reported 'at N_p = {self.spec.n_p()} iterations per stage'"}
        self.cell("t3.curve.T1", sec, "0-270 kg curve, max_stable, limiting factor, actuator load", "T1",
                  FILLED if t1 else PEND, value=out["T1"], sources=[self.F["ctx"].dir / "gates" / f"{self.F['tstage']}.json"],
                  reader="payload_read.level_reads (recomputed) + the gate record's limiting_factor", label=SIM,
                  note="the curve stops at the first unstable level (PLAN); levels above it are not read",
                  why=None if t1 else f"{PENDING} (no T* level rows)")
        # St: level 0 = the unarmed sel rows, 90 = the sel L090 rows, 45 / 135 = step-11 rows (+ conditional 180..)
        st_levels = dict(self.F["st_levels"])
        for lv in (45, 135):
            st_levels[lv] = {f"{r}@{es}": self.s11(f"ST_L{lv:03d}_{fam}_{es}") for r, fam in
                             (("suite", "suite"), ("field_light", "fl")) for es in (1, 2)}
        st = self.levels_read(st_levels, trained)
        s11_maps = [st_levels[45], st_levels[135]]      # the step-11 stamps this cell reads (its sources)
        if isinstance(st.get(135), dict) and st[135].get("stable") is True:
            for lv in (180, 225, 270):
                ss = {f"{r}@{es}": self.s11(f"ST_L{lv:03d}_{fam}_{es}") for r, fam in
                      (("suite", "suite"), ("field_light", "fl")) for es in (1, 2)}
                s11_maps.append(ss)
                if not any(_ok(s) for s in ss.values()):
                    st[lv] = {"status": PEND, "why": f"conditional rows {S11_CONDITIONAL[lv]}_* (run only when the "
                                                     "previous level is stable; not queued)", "rows": []}
                    break
                st.update(self.levels_read({lv: ss}, trained))
                if not (isinstance(st.get(lv), dict) and st[lv].get("stable")):
                    break
        sts = curve_summary(st)
        lfs = None
        fu = sts.get("first_unstable")
        if fu is not None and isinstance(st.get(fu), dict) and "failed" in st[fu]:
            lfs = _P().PR().limiting_factor(st[fu], budget=None, actuator_L=st[fu].get("actuator"),
                                            actuator_0=(st.get(0) or {}).get("actuator"), h_sag=h_sag(st, fu))
            lfs["note"] = "no training-tail read (the student is not a payload-trained stage): budget-bound not assessed"
        pend_lv = [lv for lv, r in st.items() if isinstance(r, dict) and r.get("status") == PEND]
        out["St"] = {"levels": st, **sts, "limiting_factor": lfs,
                     "level0_note": "level 0 = the UNARMED sel rows (same checkpoint / env / rows / env seeds): armed 0:0 "
                                    "is strictly bitwise to off (preflight/stepC/checks/C1.json armed00_vs_off), the "
                                    "payload event is env-side; their run_meta shows payload None, not 0 kg (beside "
                                    "pipeline.yaml's 'curve level 0 is armed' note)"}
        rows = sorted({r for lv in pend_lv for r in (st[lv].get("rows") or [])})
        self.cell("t3.curve.St", sec, "0-270 kg curve, max_stable, limiting factor, actuator load", "St",
                  PEND if pend_lv else FILLED, value=out["St"],
                  sources=[self.F["ctx"].rows_path("sel")] + [Path(x["path"]) for m in s11_maps for x in m.values()
                                                              if _ok(x)],
                  reader="payload_read.level_reads / max_stable / actuator_load",
                  label=SIM, rows=rows, why=("; ".join(f"L{lv}: {st[lv].get('why')}" for lv in pend_lv) or None),
                  note=out["St"]["level0_note"])
        # F: the forward 3a added-mass raw table (no stable-at-L lines apply)
        fl = {0: self.Fsrc["slow_pin"]}
        for lv in (45, 90, 135, 180, 225, 270):
            fl[lv] = self.s11(f"F_m{lv:03d}")
        ftab = {}
        for lv, s in fl.items():
            if not _ok(s):
                ftab[lv] = {"status": s["status"], "why": _why(s)}
                continue
            p = s["path"]
            meta = self.get(run_meta, p) or {}
            bm = (meta.get("body_masses_kg") or [None])[0]
            a = self.get(aggregate, p) or {}
            ftab[lv] = {"stamp": rel(p), "chassis_kg": bm,
                        "mass_check": (abs(bm - (CHASSIS_KG + lv)) <= 0.01) if bm is not None else None,
                        "completion": a.get("schedule_completion_rate"),
                        "falls": (a.get("termination_reasons") or {}).get("fall"), "n": a.get("n_episodes"),
                        "tracking_ratio_by_hold": {h: ((v or {}).get("ratio") or {}).get("median")
                                                   for h, v in (a.get("tracking_by_hold") or {}).items()},
                        "orientation": self.get(read_orientation, p), "actuator": self.get(read_actuator, p, float(lv))}
        fpend = [lv for lv, v in ftab.items() if v.get("status") == PEND]
        out["F"] = {"levels": ftab, "label": "added chassis mass + proportionally scaled chassis inertia (Isaac Lab "
                    "randomize_rigid_body_mass, recompute_inertia=True), CoM unchanged (no deck-CoM composite); "
                    "a raw per-level table: the stable-at-L lines do not apply to F (no suite / field_light rows; "
                    "F already falls at 0 kg)"}
        self.cell("t3.curve.F", sec, "0-270 kg curve (raw per-level table)", "F", PEND if fpend else FILLED,
                  value=out["F"], sources=[v["stamp"] for v in ftab.values() if v.get("stamp")],
                  reader="aggregate + thresholds.h_meas_and_orientation + payload_read.actuator_load",
                  label=SIM + "; " + out["F"]["label"], rows=[f"F_m{lv:03d}" for lv in fpend],
                  why=("pending (rows " + ", ".join(f"F_m{lv:03d}" for lv in fpend) + ")") if fpend else None)
        self.cell("t3.curve.H", sec, "0-270 kg curve, max_stable, limiting factor, actuator load", "H", NOTAPP,
                  why="not obtainable: the v1 era has no payload event, and the checkpoints do not load")
        # the paired 0-kg check + the heads considered for T*
        out["tstar"] = self.tstar_block(self.F, self.rec)
        self.cell("t3.zero_kg.T1", sec, "paired 0-kg check (3f) + the payload heads considered for T*", "T1",
                  FILLED if out["tstar"].get("considered") else PEND, value=out["tstar"],
                  why=None if out["tstar"].get("considered") else f"{PENDING} (no T* record)",
                  sources=[self.F["ctx"].dir / "deliverable.json", self.F["ctx"].dir / "stage_reads" / "tstar.json"],
                  reader="gate records zero_kg (gate_read.zero_kg_check) + tstar candidates[]", label=SIM)
        return out

    def tstar_block(self, src, rec) -> dict:
        ts = src["tstar"] or {}
        ctx = src["ctx"]
        heads = {}
        for sid in ("p3a", "p3b", "p3c"):
            g = rec["gates"].get(sid) if rec else L.read_json(ctx.dir / "gates" / f"{sid}.json")
            if not g:
                continue
            z = g.get("zero_kg") or {}
            hist = L.read_json(ctx.dir / "gates" / "history" / f"{sid}.1.json")
            heads[sid] = {"verdict": g.get("verdict"), "max_stable": g.get("max_stable"), "curve": g.get("curve"),
                          "limiting_factor": (g.get("limiting_factor") or {}).get("name"),
                          "zero_kg": {"verdict": z.get("verdict"), "max_margin_used": z.get("max_margin_used"),
                                      "fails": z.get("fails"), "watch_exceeds": z.get("watch_exceeds"),
                                      "translate_statistic": z.get("translate_statistic"),
                                      "thresholds_version": z.get("thresholds_version")},
                          "zero_kg_before_regate": ({"verdict": (hist.get("zero_kg") or {}).get("verdict"),
                                                     "max_margin_used": (hist.get("zero_kg") or {}).get("max_margin_used"),
                                                     "fails": (hist.get("zero_kg") or {}).get("fails"),
                                                     "thresholds": hist.get("thresholds")} if hist else None)}
        return {"stage": ts.get("stage"), "max_stable": ts.get("max_stable"), "sha256": ts.get("sha256"),
                "ckpt": rel(ts.get("ckpt")), "candidates": ts.get("candidates"), "considered": ts.get("considered"),
                "not_trained": ts.get("not_trained"), "rule": ts.get("rule"), "retarget": ts.get("retarget"),
                "heads": heads, "label": SIM}

    # ------------------------------------------------------------------ Stand
    def k_families(self, decisive: list) -> dict:
        fam = defaultdict(lambda: {"n": 0, "pass": 0, "fails": []})
        for x in decisive or []:
            p = x["id"].split(".")
            f = p[2] if len(p) > 2 else "?"
            fam[f]["n"] += 1
            if x.get("pass"):
                fam[f]["pass"] += 1
            else:
                fam[f]["fails"].append({"id": x["id"], "value": x.get("value"), "line": x.get("line"),
                                        "rate": x.get("rate"), "rate_status": x.get("rate_status")})
        return dict(sorted(fam.items()))

    @staticmethod
    def key_clauses(gates: dict, role: str) -> dict:
        """K4 misses and K6 homed / settled shares (+ K3 STOOD) of one role in an sh_read JSON."""
        out = {}
        for K, pat in (("K4", r"^miss es\d"), ("K6", r"^(homed|settled) share bin\d es\d"), ("K3", r"^STOOD rate")):
            cl = ((gates.get(K) or {}).get(role) or {}).get("clauses") or {}
            for name, x in cl.items():
                if re.match(pat, name):
                    out[f"{K} {name.split(' >=')[0].split(' <=')[0]}"] = {"value": x.get("value"), "line": x.get("line"),
                                                                          "ok": x.get("ok")}
        return out

    def sh_block(self, run_dir: Path, gate: dict) -> dict:
        head = (gate or {}).get("head") or {}
        name = head.get("name")
        lazy = L.read_json(run_dir / "stand" / "sh_lazy_rh_read.json")
        res = {"head": name, "sha256": head.get("sha256"), "verdict": (gate or {}).get("verdict"),
               "halt_lines": (gate or {}).get("halt_lines"), "families": self.k_families((gate or {}).get("decisive")),
               "decisive_key": {x["id"].replace("stand.D2.", ""): {"value": x.get("value"), "line": x.get("line"),
                                                                    "pass": x.get("pass")}
                                for x in (gate or {}).get("decisive") or []
                                if re.search(r"\.K4\.miss|\.K6\.(homed|settled) share bin[14]", x["id"])}}
        if lazy:
            g = lazy.get("gates") or {}
            k5g = (g.get("K5") or {}).get(name) or {}
            k5i = invalid_kinds(k5g.get("invalid"))
            fam5 = res["families"].get("K5") or {}
            k5_text = None
            if k5g.get("verdict") == "INVALID":
                k5_text = ("INVALID only because the lazy re-read compares K5 against "
                           + (", ".join(k5i["references"]) or "the S1-era reference")
                           + f" ({' + '.join(k5i['kinds']) or 'identity'} mismatch; {k5i['n']} reasons in the record)"
                           + (f"; the gate's own K5 lines: {fam5.get('pass')}/{fam5.get('n')} PASS" if fam5 else ""))
            res["lazy_control"] = {
                "record": rel(run_dir / "stand" / "sh_lazy_rh_read.json"),
                "head_verdicts_reanchored": ((lazy.get("heads") or {}).get(name) or {}).get("verdicts"),
                "control_R_h": self.key_clauses(g, "R_h"), "head_reanchored": self.key_clauses(g, name),
                "control_stamps": [((v or {}).get("stamp")) for es in ((lazy.get("reads") or {}).get("R_h") or {}).values()
                                   for v in es.values()],
                "k5_invalid": {**k5i, "verdict": k5g.get("verdict"), "text": k5_text,
                               "gate_k5_lines": {"pass": fam5.get("pass"), "n": fam5.get("n")} if fam5 else None},
                "note": "the R_h control beside T* (DESIGN 3.6): a line the control fails too is walker-caused and "
                        "re-anchors on the control; K5 is read against the reference the re-read names (k5_invalid), "
                        "the gate's K5 lines are in the gate record"}
            sp = stamps_span_h([s for s in res["lazy_control"]["control_stamps"] if s])
            res["lazy_control"]["measured"] = sp
        return res

    @staticmethod
    def refs_tstar(calib: Path, head: str | None) -> dict:
        """SN's REFS(T*) row (sn_calib.json ``rows['REFS']``: T* alone on the stand rows, no stand net) beside the SN
        head's own row: per gate family the reader's verdict / expected / lands, and every clause value + line. A
        missing record or row is N/A with its reason (never a number)."""
        d = L.read_json(calib) if Path(calib).is_file() else None
        if not d:
            return {"available": False, "why": f"N/A: no SN calibration record {rel(calib)}"}
        rows = d.get("rows") or {}
        refs = rows.get("REFS")
        if not refs:
            return {"available": False, "why": f"N/A: {rel(calib)} has no rows['REFS'] (the REFS(T*) rows were not read)"}
        hd = rows.get(head) or {}
        gates = {}
        for G, v in (refs.get("gates") or {}).items():
            hv = ((hd.get("gates") or {}).get(G) or {})
            gates[G] = {"verdict": v.get("verdict"), "expected": v.get("expected"), "lands": v.get("lands"),
                        "invalid": invalid_kinds(v.get("invalid")),
                        "head_verdict": hv.get("verdict"),
                        "clauses": {c: {"refs": x.get("value"), "line": x.get("line"), "refs_ok": x.get("ok"),
                                        "head": ((hv.get("clauses") or {}).get(c) or {}).get("value")}
                                    for c, x in (v.get("clauses") or {}).items()}}
        return {"available": True, "record": rel(calib), "row": refs.get("name"), "kind": refs.get("kind"),
                "head": head, "stamps": {k: rel(v) for k, v in (refs.get("stamps") or {}).items()},
                "reported": refs.get("reported"), "gates": gates,
                "note": "REFS(T*) = T* alone on the SN rows (no stand net; KRABBY_OMNI_STAND_OBS=0): the reference the "
                        "SN lines are drawn from (G4 completion, G5 onset / body height) and the expected-FAIL "
                        "stillness baseline (G1 / G2). sn_calib's INVALID on a family is its identity check against the "
                        "S1-era calibration rows; the decisive SN read is gates/sn.json"}

    def stand(self) -> dict:
        sec = "Stand (teacher-side, 0 kg)"
        out = {}
        t17 = L.read_json(self.sh_dir / "head_SH_T17.json") if self.sh_dir else None
        k3 = L.read_json(self.sh_dir / "head_SH_T17_k3_summary.json") if self.sh_dir else None
        hw = newest_in(self.p2_lr / "T4C2_SH_T17" / "hwalkstop__omni" / "seed001")
        if t17:
            out["T0"] = {"head": "SH_T17", "verdicts": ((t17.get("heads") or {}).get("SH_T17") or {}).get("verdicts"),
                         "key": self.key_clauses(t17.get("gates") or {}, "SH_T17"),
                         "control_R_h": self.key_clauses(t17.get("gates") or {}, "R_h"),
                         "stand_sha256": ((t17.get("heads") or {}).get("SH_T17") or {}).get("stand_sha256"),
                         "k3_summary": bool(k3), "pin_build_hwalkstop": rel(hw)}
        self.cell("stand.T0", sec, "K1 tracking, K2 raise / lower, K3 table stand-up, K4 restore (<= 3/800, D9), K5 "
                  "non-regression @1.064, K6 homing + stillness", "T0", FILLED if t17 else NOTAPP, value=out.get("T0"),
                  sources=[self.sh_dir / "head_SH_T17.json", self.sh_dir / "head_SH_T17_k3_summary.json", hw] if t17 else (),
                  reader="sh_read JSON gates / heads blocks",
                  note="SH_T17 = policy/stand_net/model_5000.pt (sha 99c377a1), S1-paired; its K4 clauses read the "
                       "S1-era max(1, R_h) line (the pipeline's D9 line is <= 3/800); C2 at the pin is aggregate-equal")
        rd = self.F["ctx"].dir
        sh = self.rec["gates"].get("sh")
        out["T1"] = self.sh_block(rd, sh) if sh else None
        self.cell("stand.T1.sh", sec, "K1-K6 (+ the lazy-control re-anchor)", "T1", FILLED if sh else PEND,
                  value=out["T1"], sources=[rd / "gates/sh.json", rd / "stand/sh_read.json", rd / "stand/sh_k5_calib.json",
                                           rd / "stand/sh_lazy_rh_read.json"] if sh else (),
                  reader="gates/sh.json decisive[] + sh_lazy_rh_read (re-anchored K6)",
                  why=None if sh else f"{PENDING} (no gates/sh.json)")
        sn = self.rec["gates"].get("sn")
        out["T1_sn"] = ({"verdict": sn.get("verdict"), "head": (sn.get("head") or {}).get("name"),
                         "sha256": (sn.get("head") or {}).get("sha256"), "families": self.k_families(sn.get("decisive")),
                         "watch_lines": [{"id": x.get("id"), "value": x.get("value"), "line": x.get("line"),
                                          "pass": x.get("pass")} for x in sn.get("watch_lines") or []],
                         "watch_flags": sn.get("watch_flags")} if sn else None)
        if out["T1_sn"] is not None:
            out["T1_sn"]["refs_tstar"] = self.refs_tstar(rd / "stand" / "sn_calib.json", out["T1_sn"].get("head"))
        self.cell("stand.T1.sn", sec, "SN G1-G5 + REFS(T*)", "T1", FILLED if sn else PEND, value=out["T1_sn"],
                  sources=[rd / "gates/sn.json", rd / "stand/sn_calib.json", rd / "stand/sn_bridge.json"] if sn else (),
                  reader="gates/sn.json decisive[] + watch_lines[]", note="G1 / G2 stillness lines are WATCH since v1.1",
                  why=None if sn else f"{PENDING} (no gates/sn.json)")
        self.cell("stand.na", sec, "K1-K6", "H / F / St", NOTAPP,
                  why="H: no stand network in the v1 era; F: no forward stand network; St: the stand nets are teacher-side "
                      "(paired with T*), a depth-student stand path is a PLAN Later phase")
        return out

    # ------------------------------------------------------------------ Student
    def student(self) -> dict:
        sec = "Student"
        P = self.P
        ctx, sel = self.F["ctx"], self.F["sel"]
        cand = sel.get("cand")
        g = self.rec["gates"].get("sel") or {}
        blk = (g.get("blocks") or {}).get(cand) or {}
        out = {"cand": cand, "sha256": sel.get("sha256"), "ckpt": rel(sel.get("ckpt"))}
        comp = [x for x in blk.get("decisive") or [] if ".capability.completion." in x.get("id", "")]
        rebuilt, match = None, None
        try:
            SS = P.SS()
            TH = L.thresholds_mod()
            stu, tst = {}, {}
            for row in FIELD_ROWS:
                for es in (1, 2):
                    a, b = self.F["St"].get(f"{row}@{es}"), self.F["TS"].get(f"{row}@{es}")
                    if _ok(a) and _ok(b):
                        stu.setdefault(row, {}).update({(es, k): v for k, v in TH.field_episodes(Path(a["path"])).items()})
                        tst.setdefault(row, {}).update({(es, k): v for k, v in TH.field_episodes(Path(b["path"])).items()})
            cfg = (self.spec.stage("sel") or {}).get("completion_margin") or {}
            rebuilt = SS.completion_pairs(stu, tst, cfg) if stu else None
            if rebuilt and comp:
                rec = {x["row"]: (x.get("student"), x.get("tstar"), x.get("line")) for x in comp}
                match = all(abs(r["student"] - rec[r["row"]][0]) < 1e-9 and abs(r["tstar"] - rec[r["row"]][1]) < 1e-9
                            for r in rebuilt if r["row"] in rec and not r.get("missing"))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(_clip(f"student pairs: {exc!r}", 300))
        pre = ((ctx.state.get("stages") or {}).get("sel") or {}).get("prescreen") or {}
        ranked = [x.get("iter") for x in P.SS().rank(list(pre.values()))] if pre else None
        out["pairing"] = {"recorded": [{k: x.get(k) for k in ("row", "student", "tstar", "se_diff", "margin", "binds",
                                                               "line", "value", "pass")} for x in comp],
                          "rebuilt_from_stamps": rebuilt, "rebuilt_matches_record": match,
                          "verdict": blk.get("verdict"), "watch_flags": blk.get("watch_flags"),
                          "tracking_watch_flags": ((blk.get("watch") or {}).get("tracking") or {}).get("flags"),
                          "prescreen": pre, "ranked_iters": ranked,
                          "floors": [{k: x.get(k) for k in ("id", "value", "line", "pass")} for x in blk.get("decisive") or []
                                     if ".floor." in x.get("id", "")]}
        self.cell("st.pairing", sec, "paired completion vs T* (margin max(0.05, 2 SE_diff, split 0.0)), floors, "
                  "selection record, tracking + 0.05 WATCH", "St", FILLED if comp else PEND, value=out["pairing"],
                  sources=[ctx.dir / "gates/sel.json", ctx.rows_path("sel"), ctx.dir / "stage_reads/sel.json"],
                  reader="sel.json blocks[cand] + select_student.completion_pairs rebuilt from the sel.rows stamps "
                         "(thresholds.field_episodes) + select_student.rank on the prescreen",
                  note="disclosure (b): the tracking WATCH keeps a raw one-sided ratio row",
                  why=None if comp else f"{PENDING} (no selection record)")
        w = blk.get("watch") or {}
        lv = (w.get("level") or {})
        out["stable90"] = {"stable_at": w.get("stable_at"), "stable": lv.get("stable"), "failed": lv.get("failed")}
        self.cell("st.stable90", sec, "stable at min(90, T* max_stable) kg (WATCH)", "St", FILLED if lv else PEND,
                  value=out["stable90"], sources=[ctx.dir / "gates/sel.json"],
                  reader="sel.json watch.level (payload_read.level_reads on the c28000 L090 rows; recomputed in Task 3)",
                  label=SIM, why=None if lv else PENDING)
        live = self.live_checks()
        out["live"] = live
        self.cell("st.live", sec, "live checks (depth_actor_loss at +250, ep length / crab_failure vs T* tail)", "St",
                  FILLED if live and "unreadable" not in live else PEND, value=live,
                  sources=[live.get("run_dir")] if live else (), reader="live_watch.replay(kind=student) on the tfevents",
                  note="the values are not stored in pipeline_state (stages.st.live keeps rc / last / run_dir only); "
                       "replayed on CPU here", why=None if live else PENDING)
        c4 = L.read_json(self.pl / "preflight/stepC/checks/C4_depth.json") or {}
        out["camera"] = {"verdict": "UNVERIFIED", "c4_depth": {k: c4.get(k) for k in ("depth_completion", "zero_completion",
                                                                                        "diff", "required", "verdict")},
                         "why": "no omni row is perception-limited (S1 with a perfect scan equals a blind student); "
                                "accepted with disclosure by the user 2026-10-17"}
        self.cell("st.camera", sec, "camera use (C4.7 depth helps)", "St", FILLED if c4 else PEND, value=out["camera"],
                  why=None if c4 else f"{PENDING} (no preflight C4_depth record)",
                  sources=[self.pl / "preflight/stepC/checks/C4_depth.json", self.pl / "preflight/stepC/depthdiag.log"],
                  note="UNVERIFIED -- a disclosure, not a pass")
        clip = {p.name: {"bytes": p.stat().st_size} for p in sorted((ctx.dir / "clip").glob("*.mp4"))} \
            if (ctx.dir / "clip").is_dir() else {}
        exp_dir = Path(sel.get("ckpt") or "/nonexistent").parent / "exported_deploy"
        exp = {p.name: {"bytes": p.stat().st_size, "sha256": L.sha256_file(p)} for p in sorted(exp_dir.glob("*"))
               if p.is_file()} if exp_dir.is_dir() else {}
        clip_paths = [rel(ctx.dir / "clip" / n) for n in clip]
        ign = git_ignored([L.REPO / p for p in clip_paths]) if clip_paths else {}
        out["clip_export"] = {"clip": clip, "clip_dir": rel(ctx.dir / "clip"), "exported_deploy": exp,
                              "exported_dir": rel(exp_dir), "note": "the clip is a veto for the user, never a gate",
                              "clip_paths": clip_paths, "gitignored": ign,
                              "local_only": bool(clip_paths) and all(p in ign for p in clip_paths)}
        st_ce = FILLED if (clip and exp) else (self.status_of([self.clip_src(clip, exp)])[0])
        self.cell("st.clip", sec, "headless veto clip + JIT / ONNX deploy export", "St", st_ce, value=out["clip_export"],
                  sources=[ctx.dir / "clip", exp_dir] if (clip or exp) else (),
                  rows=[] if (clip and exp) else ["CLIP_F19_c28000"],
                  why=None if (clip and exp) else "pending (row CLIP_F19_c28000)")
        return out

    def clip_src(self, clip, exp):
        lg = self.rows_log["rows"].get("CLIP_F19_c28000") or {}
        if lg.get("state") == "done" and lg.get("rc", 0) != 0:
            return _src(status=NOTAPP, why=f"clip render failed (rc {lg.get('rc')})", row="CLIP_F19_c28000")
        return _src(status=PEND, row="CLIP_F19_c28000")

    def live_checks(self) -> dict | None:
        st = ((self.F["ctx"].state.get("stages") or {}).get("st") or {}).get("live") or {}
        rd = st.get("run_dir")
        if not rd or not Path(rd).is_dir():
            return None
        try:
            LW = self.P.LW()
            ser = LW.read_series(Path(rd))
            tt = None
            trd = (self.F["tstar"] or {}).get("run_dir")
            if trd and Path(trd).is_dir():
                ts = LW.read_series(Path(trd), (LW.T_EPLEN, LW.T_CRAB))
                tt = {"ep_len": LW.tail_mean(ts.get(LW.T_EPLEN)), "crab_failure": LW.tail_mean(ts.get(LW.T_CRAB))}
            cfg = {"student": (self.spec.d.get("live") or {}).get("student") or {}}
            rep = LW.replay("student", ser, every=50, tstar_tail=tt, cfg=cfg)
            fin = LW.check("student", ser, tstar_tail=tt, cfg=cfg)
            loss = next((r for r in fin.get("reads") or [] if r.get("rule") == "student"), {}).get("reads")
            return {"run_dir": rel(rd), "first": fin.get("first"), "last": fin.get("last"), "replay": rep,
                    "final_rc": fin.get("rc"), "final_fails": fin.get("fails"), "reads": loss, "tstar_tail": tt,
                    "superseded": [{"run_dir": rel(((x.get("records") or {}).get("live") or {}).get("run_dir")),
                                    "last": ((x.get("records") or {}).get("live") or {}).get("last"),
                                    "why": x.get("charter")}
                                   for x in ((self.F["ctx"].state.get("stages") or {}).get("st") or {}).get("superseded") or []]}
        except Exception as exc:  # noqa: BLE001
            return {"unreadable": f"{type(exc).__name__}: {_clip(str(exc), 200)}", "run_dir": rel(rd)}

    # ------------------------------------------------------------------ Provenance
    @staticmethod
    def mark_superseded(sid: str, v: dict, state: dict) -> dict:
        """A stage's wall-clock segments with the run each one LAUNCHED; a segment whose launched run is in
        ``stages[sid].superseded`` (e.g. F19 st's p3a-based distill, stopped by hand for the v1.2 retarget) is marked
        SUPERSEDED: its hours are GPU work and stay in the stage total, but its result was abandoned."""
        sup = {x.get("run_name"): x for x in ((state.get("stages") or {}).get(sid) or {}).get("superseded") or []
               if x.get("run_name")}
        info = []
        for x in v.get("segment_info") or []:
            nm = (x.get("launch") or {}).get("name")
            s = sup.get(nm)
            info.append({**x, "superseded": ({"run_name": nm, "charter": s.get("charter"), "at": s.get("at"),
                                              "status": s.get("status"),
                                              "last_iteration": ((s.get("records") or {}).get("live") or {}).get("last"),
                                              "run_dir": rel(((s.get("records") or {}).get("live") or {}).get("run_dir"))}
                                             if s else None)})
        return {**v, "segment_info": info}

    def run_provenance(self, run: str, rec: dict) -> dict:
        ctx = rec["ctx"]
        st = ctx.state
        wc = self.get(wallclock, ctx.qlog) if ctx.qlog.is_file() else None
        lazy = None
        sh = (rec["gates"].get("sh") or {})
        if sh:
            lazy = (self.sh_block(ctx.dir, sh).get("lazy_control") or {}).get("measured")
        halts = []
        for sid, h in (st.get("halts") or {}).items():
            r = rec["halts"].get(sid) or {}
            res = r.get("resolution") or {}
            kind = res.get("kind")
            label = {"accept-chance": "accepted as chance", "amend": "amended", "retry": "retried",
                     "abort": "aborted"}.get(kind, kind)
            if run == "F19" and sid == "sh":
                label = ("re-anchored on the lazy R_h control (DESIGN 3.6; recorded kind 'accept-chance' = the "
                         "pipeline's only no-re-gate continue path; not a chance acceptance)")
            rate = r.get("chance_rate_sum")
            unm = list(r.get("chance_rate_unmodelled") or [])
            th = self.P.GR.Thresholds(self.th_doc)
            degen = [x.get("id") for x in r.get("lines") or [] if th.by_id(x.get("id")) is not None
                     and self.P.GR.is_degenerate(th.by_id(x.get("id")))]
            if degen and not unm:
                unm = [{"id": i, "rate_status": self.P.GR.DEGENERATE} for i in degen]
            k6r = [x["rate"] for x in r.get("lines") or [] if ".K6." in str(x.get("id")) and x.get("rate") is not None]
            halts.append({"stage": sid, "class": r.get("class") or h.get("class"), "status": h.get("status"),
                          "chance_rate_recorded": rate,
                          "chance_rate_caveat": (f"its K6 line rates (sum {sum(k6r):.3g}) are {K6_RATE_CAVEAT}"
                                                 if (k6r and rate is not None and not unm) else None),
                          "resolution": label, "at": r.get("at"), "resolved_at": res.get("at"),
                          "why": r.get("why"), "resolution_why": res.get("why"), "charter": res.get("charter"),
                          "lines": [{k: x.get(k) for k in ("id", "value", "line", "rate", "rate_status")}
                                    for x in r.get("lines") or []],
                          "chance_rate": rate if (rate is not None and not unm and r.get("lines")) else None,
                          "chance_rate_text": (f"{rate:.3g}" if rate is not None and not unm and r.get("lines")
                                               else ("not applicable (no decisive line: a driver / guard HALT)"
                                                     if not r.get("lines") else
                                                     ("rate unknown (degenerate fixed-t line; the recorded "
                                                      f"{rate} predates v1.1)" if degen else
                                                      "rate unknown (unmodelled)"))),
                          "record": h.get("record")})
        verdicts = {}
        for sid, v in (st.get("stages") or {}).items():
            g = rec["gates"].get(sid) or {}
            verdicts[sid] = {"status": v.get("status"), "gate": (v.get("gate") or {}).get("verdict") or g.get("verdict"),
                             "thresholds": (g.get("thresholds") or {}).get("version"),
                             "watch_flags": [f.get("id") for f in g.get("watch_flags") or []]}
        if isinstance(wc, dict) and "stages" in wc:
            wc = {**wc, "stages": {sid: self.mark_superseded(sid, v, st) for sid, v in wc["stages"].items()}}
        gpu = None
        if isinstance(wc, dict) and "stages" in wc:
            gpu = {"stages_active_h": wc["total_active_h"], "lazy_control_h": (lazy or {}).get("h"),
                   "superseded_h": sum(x["h"] for v in wc["stages"].values() for x in v.get("segment_info") or []
                                       if x.get("superseded")),
                   "total": wc["total_active_h"] + ((lazy or {}).get("h") or 0.0),
                   "rule": "GPU-h = active stage wall-clock (one RTX-class GPU, serial queue, one Isaac process at a "
                           "time); parks are 0 GPU-h except a hand-run lazy R_h control inside a park (measured from "
                           "its stamps)"}
        return {"run": run, "pin": st.get("pin"), "amendments": st.get("amendments"), "verdicts": verdicts,
                "halts": halts, "wallclock": wc, "gpu_h": gpu, "lazy_control_measured": lazy,
                "driver_wall_h_training_only": rec.get("wall_h_driver"),
                "driver_wall_h_note": "the driver's state.json wall_h is training only (no rows / gates); "
                                      "pipeline_state stages[*].wall_h are null -- the queue log is the measure"}

    def provenance_all(self) -> dict:
        sec = "Provenance"
        out = {"selection_disclosure": self.selection_disclosure()}
        self.cell("prov.sel_world_rows", sec, "DISCLOSURE: student selection read world rows only", "St", FILLED,
                  value=out["selection_disclosure"], sources=[self.spec_path, self.pl / "DESIGN.md"],
                  reader="pipeline.yaml sel full_block_rows vs DESIGN 3.5 'the p2a world, body and sweep rows'")
        out[self.run] = self.run_provenance(self.run, self.rec)
        wc = out[self.run].get("wallclock")
        self.cell(f"prov.wall.{self.run}", sec, "measured wall-clock AND GPU-h per stage, parks separated", "T1",
                  FILLED if wc else PEND, value={"wallclock": wc, "gpu_h": out[self.run]["gpu_h"]},
                  sources=[self.F["ctx"].qlog, self.F["ctx"].dir / "state.json"], reader="wallclock() (queue sentinels)")
        if self.R:
            rr = run_records(self.spec_path, self.other)
            self.rec_other = rr
            out[self.other] = self.run_provenance(self.other, rr)
            self.cell(f"prov.wall.{self.other}", sec, "measured wall-clock + GPU-h per stage (appendix)", "R18",
                      FILLED if out[self.other].get("wallclock") else PEND,
                      value={"wallclock": out[self.other].get("wallclock"), "gpu_h": out[self.other]["gpu_h"]},
                      sources=[rr["ctx"].qlog], reader="wallclock()")
        out["preflight"] = preflight_wallclock(self.pl)
        self.cell("prov.wall.preflight", sec, "preflight wall-clock (Phase A + C)", "all",
                  FILLED if out["preflight"] else NOTAPP, value=out["preflight"],
                  sources=[self.pl / "preflight/stepA_queue.log", self.pl / "preflight/stepC/stepC_queue.log"],
                  reader="T4A START..END; sum of T4C DONE wall_s")
        s11 = self.rows_log
        out["step11_rows"] = {"measured_h": s11.get("measured_h") if s11["rows"] else None, "start": s11.get("start"),
                              "end": s11.get("end"), "n_rows_measured": s11.get("n_rows_measured", 0),
                              "n_rows_seen": len(s11["rows"]), "n_rows_by_state": s11.get("n_rows_by_state"),
                              "source": s11.get("source"), "rule": s11.get("measured_rule"),
                              "note": "the step-11 rows' measured wall-clock (ROW -> DONE), GPU-h, over the rows with "
                                      "both sentinels only; None without --rows-log"}
        out["gpu_h_total"] = self.gpu_total(out)
        lock = L.read_json(self.spec.lock_path()) or {}
        out["stack"] = {"hardware": hardware(), "external": lock.get("external"),
                        "lock": {"code_sha": lock.get("code_sha"), "eval_sha": lock.get("eval_sha"),
                                 "tools_sha": lock.get("tools_sha"), "written_at": lock.get("written_at"),
                                 "n_files": len(lock.get("files") or {}),
                                 "sha256": L.sha256_file(self.spec.lock_path()) if self.spec.lock_path().is_file() else None},
                        "tags": git_tags(), "pin": self.F["ctx"].state.get("pin")}
        self.cell("prov.stack", sec, "hardware + external stack + pins", "all", FILLED, value=out["stack"],
                  sources=["/proc/driver/nvidia/gpus", "/proc/cpuinfo", self.spec.lock_path(),
                           self.F["ctx"].state_path], reader="hardware() + pipeline.lock.json + git tag -l")
        self.cell(f"prov.gates.{self.run}", sec, "gate verdicts + HALTs with chance rates", "T1",
                  FILLED if out[self.run]["verdicts"] else PEND,
                  value={"verdicts": out[self.run]["verdicts"], "halts": out[self.run]["halts"],
                         "amendments": out[self.run]["amendments"]},
                  sources=[self.F["ctx"].dir / "gates", self.F["ctx"].dir / "halts", self.F["ctx"].dir / "deliverable.json"],
                  reader="pipeline_state + gates/*.json + halts/*.json",
                  note="p3a.1 accepted (rate unknown, unmodelled); st.1 a DRIVER HALT closed by the v1.2 amend + T* "
                       "retarget (an amend, not a chance acceptance); sh.1 a DESIGN 3.6 re-anchor recorded as "
                       "'accept-chance'")
        if self.R:
            self.cell(f"prov.gates.{self.other}", sec, "gate verdicts + HALTs with chance rates (appendix)", "R18", FILLED,
                      value={"verdicts": out[self.other]["verdicts"], "halts": out[self.other]["halts"]},
                      sources=[self.R["ctx"].dir / "gates", self.R["ctx"].dir / "halts"],
                      reader="pipeline_state + gates/*.json + halts/*.json",
                      note="2 GUARD re-pin pairs (p2a / sn_ref / sh_ref); p3a.1 accepted (unmodelled); sn.1 amended to "
                           "v1.1 (its recorded 0.0 rate is a degenerate fixed-t artefact); sh.1 accepted")
        sums = (self.th_doc.get("sums") or {})
        out["run_sum"] = {"recommended": (sums.get("recommended") or {}).get("run_sum"),
                          "standard": (sums.get("standard") or {}).get("run_sum"),
                          "recommended_union": (sums.get("recommended") or {}).get("run_union"),
                          "note": "the chance-HALT run sum, reported, not capped (PLAN Rule)"}
        self.cell("prov.run_sum", sec, "chance-HALT run sum", "all", FILLED if out["run_sum"]["recommended"] is not None
                  else NOTAPP, value=out["run_sum"], sources=[self.th_path], reader="thresholds sums.*.run_sum")
        out["amendment_v12"] = self.amendment_v12()
        self.cell("prov.disclosures", sec, "CHARTER disclosures", "all", FILLED,
                  value=[d["id"] for d in self.disclosures()], sources=[self.pl / "CHARTER.md"],
                  reader="the Disclosures section (each entry cites its CHARTER entry and record)")
        if self.R:
            out["r18_reproduced"] = self.r18_reproduced()
            self.cell("prov.r18_reproduced", sec, "R18 process-reproduced verdict + X stand replay", "R18", FILLED,
                      value=out["r18_reproduced"], sources=[self.R["ctx"].dir / "replay", self.R["ctx"].dir / "gates/sn_ref.json",
                                                             self.R["ctx"].dir / "gates/sh_ref.json"],
                      reader="R18/replay records (replay_read pconfig / pearly / onset / x_read)")
        out["keep_set"] = self.keep_set()
        self.cell("prov.keep_set", sec, "keep-set record: why the student checkpoint is not in head/", "St",
                  FILLED if out["keep_set"].get("student_sha256") else PEND,
                  value=out["keep_set"], sources=[self.F["sel"].get("ckpt"), L.TOOLS / "bundle_experiment.py"],
                  reader="sha256 + size + bundle_experiment.HEAD_MAX_BYTES")
        return out

    def gpu_total(self, pv: dict) -> dict:
        """ONE labelled total of the measured GPU-h: both runs (queue-active stage wall-clock + the hand-run lazy R_h
        controls), the step-11 rows with both sentinels, and the preflight (A: T4A START..END; C: the sum of T4C DONE
        wall_s MINUS the measured C7 stop hang, which was a hung train.py at 100 % CPU). Given with and without the
        hang; None terms are listed as missing, never counted as 0."""
        terms, missing = [], []

        def add(key, h, what):
            (terms if h is not None else missing).append({"term": key, "h": h, "what": what})

        for run in (self.run, self.other):
            if not run:
                continue
            g = (pv.get(run) or {}).get("gpu_h") or {}
            add(run, g.get("total"), f"queue-active stage wall-clock {fmt(g.get('stages_active_h'), 2)} h"
                + (f", incl. {fmt(g.get('superseded_h'), 2)} h of superseded segments (GPU work, result abandoned)"
                   if g.get("superseded_h") else "")
                + f" + hand-run lazy R_h control {fmt(g.get('lazy_control_h'), 2)} h")
        s = pv.get("step11_rows") or {}
        add("step-11 rows", s.get("measured_h"), f"{s.get('n_rows_measured')} rows with both S11 ROW and DONE "
            f"sentinels (of {s.get('n_rows_seen')} in the log at report time)")
        pre = pv.get("preflight") or {}
        add("preflight A", (pre.get("A") or {}).get("h"), "T4A START .. T4A END")
        c = pre.get("C") or {}
        hang = c.get("hang") or {}
        add("preflight C", c.get("h_excl_hang"),
            f"sum of T4C DONE wall_s {fmt(c.get('h'), 2)} h minus the C7 stop hang {fmt(hang.get('h'), 2)} h, "
            f"measured as {hang.get('rule')}; still incl. {fmt(c.get('cpu_check_jobs_h'), 2)} h of CPU-only check "
            "jobs, not subtracted")
        tot = sum(t["h"] for t in terms)
        return {"total_excl_hang": tot if terms else None,
                "total_incl_hang": (tot + hang["h"]) if (terms and hang.get("h") is not None) else None,
                "hang_h": hang.get("h"), "terms": terms, "missing": missing,
                "rule": "GPU-h = measured wall-clock of the one-GPU serial queue (one Isaac process at a time); "
                        "parks are 0 GPU-h; the C7 stop hang is excluded (shown separately)"}

    def selection_disclosure(self) -> dict:
        st = self.spec.stage("sel") or {}
        return {"full_block_rows": st.get("full_block_rows"), "full_block_level_rows": st.get("full_block_level_rows"),
                "design": "DESIGN 3.5: 'Full block ... the p2a world, body and sweep rows, plus the payload levels "
                          "{0, 90}'; the DESIGN header retired the student grid / sweep lines as HALT (WATCH)",
                "text": "The student selection gate read the WORLD rows only (suite, side22, field rows; + the 90-kg "
                        "level). The St body / grid / turn-level / sweep cells were never part of it: they are read "
                        "post hoc at step 11 as WATCH-class numbers (or not obtainable if a row is refused). No CHARTER "
                        "entry covers this; disclosed here."}

    def amendment_v12(self) -> dict:
        th11 = L.read_json(self.pl / "thresholds_v1_1.json") or {}

        def zline(doc):
            for x in doc.get("lines") or []:
                if x.get("id") == "p3a.capability.zero_kg_paired":
                    return {"translate_statistic": x.get("translate_statistic", "ratio_drop"), "rate_note": x.get("rate_note")}
            return None

        return {"from": {"version": th11.get("version"), "sha256": L.sha256_file(self.pl / "thresholds_v1_1.json")
                         if (self.pl / "thresholds_v1_1.json").is_file() else None, "zero_kg": zline(th11)},
                "to": {"version": self.th_doc.get("version"), "sha256": L.sha256_file(self.th_path)
                       if self.th_path.is_file() else None, "zero_kg": zline(self.th_doc)},
                "charter": "CHARTER 2026-10-22 (user: amend + switch now) / 2026-10-23 (built + verified, pin "
                           "task4-pin-2026-10-23)",
                "rule": "decisive translate statistic of the paired 0-kg check = track_err: worse_by = |r_head - 1| - "
                        "|r_p2a - 1|, failing on an increase only (margin: 99 % guard R, 3 SE_diff, X12 - X08 drift "
                        "scaled, each on |r - 1|); move_turn stays a one-sided ratio drop; the raw translate drop is "
                        "a WATCH read",
                "amendments": self.F["ctx"].state.get("amendments")}

    def r18_reproduced(self) -> dict:
        rd = self.R["ctx"].dir
        s1 = L.read_json(rd / "replay/s1.json") or {}
        pl = L.read_json(rd / "replay/s1_plive_posthoc.json") or {}
        lv = L.read_json(rd / "replay/live_posthoc.json") or {}
        pc = {}
        for sid in ("s1", "c4", "c5", "c6", "c6x", "p2a"):
            r = L.read_json(rd / "replay" / f"{sid}.json") or {}
            p = r.get("pconfig") or {}
            pc[sid] = p.get("pass", p.get("verdict"))
        x = {}
        for sid in ("sn_ref", "sh_ref"):
            g = L.read_json(rd / "gates" / f"{sid}.json") or {}
            xx = g.get("x") or {}
            x[sid] = {"gate": g.get("verdict"), "x": xx.get("verdict"), "sha_equal": xx.get("sha_equal"),
                      "loss_curve_equal": xx.get("loss_curve_equal")}
        return {"pconfig": pc, "pearly": {"pass": (s1.get("pearly") or {}).get("pass"),
                                          "bitwise": (s1.get("pearly") or {}).get("design_bitwise")},
                "plive": {"pass": pl.get("pass"), "bitwise": pl.get("design_bitwise"), "shared": pl.get("shared")},
                "live_posthoc_kills": {k: (v.get("replay") or {}).get("killed_at") for k, v in lv.items()},
                "x": x, "verdict": "P-config PASS (6 stages), P-early PASS (4-ULP amendment), P-live FAIL (iteration "
                                   "20; cuDNN history-encoder numerics) -- STOP-R accepted with disclosure; X report-only"}

    def keep_set(self) -> dict:
        ck = self.F["sel"].get("ckpt")
        size = Path(ck).stat().st_size if ck and Path(ck).is_file() else None
        sha = L.sha256_file(Path(ck)) if size else None
        guard = None
        be = L.TOOLS / "bundle_experiment.py"
        if be.is_file():
            m = re.search(r"^HEAD_MAX_BYTES\s*=\s*([0-9 *]+)$", be.read_text(), re.M)
            if m and re.fullmatch(r"[0-9 *]+", m.group(1)):
                guard = eval(m.group(1))  # noqa: S307  (digits and '*' only, checked above)
        b = L.load_yaml(self.pl / "bundle.yaml") if (self.pl / "bundle.yaml").is_file() else None
        head = (b or {}).get("head") or {}
        return {"student_ckpt": rel(ck), "student_sha256": sha, "student_bytes": size, "head_max_bytes": guard,
                "exceeds_guard": (size > guard) if (size and guard) else None,
                "recorded_sha_matches": (sha == self.F["sel"].get("sha256")) if sha else None,
                "bundle_yaml": {"head": head.get("dest"), "extra_heads": [x.get("dest") for x in head.get("extra_heads") or []],
                                "evals": [x.get("dest") for x in (b or {}).get("evals") or []],
                                "keep_globs": (b or {}).get("keep_globs")} if b else None,
                "note": "the student checkpoint stays in logs/ and is cited by sha256 (CHARTER 2026-10-24 decision 1); "
                        "head/ holds T* + SN + SH + the student's ONNX deploy export"}

    # ------------------------------------------------------------------ reconciliation / disclosures
    def reconciliation(self) -> dict:
        sec = "Provenance"
        s = self.F["T1"].get("suite@2") or self.F["T1"].get("suite@1")
        tot = None
        if _ok(s):
            bm = (self.get(run_meta, s["path"]) or {}).get("body_masses_kg")
            tot = sum(bm) if bm else None
        gd = L.REPO.parent / GRANT_DIR
        docs = {n: (gd / n).is_file() for n in ("TASK-3-ACTUATOR-DYNAMICS-AND-PAYLOAD.md",
                                                 "TASK-4-REPRODUCIBLE-CURRICULUM.md", "OVERVIEW.md")}
        val = {"robot_mass_kg_from_run_meta": tot, "robot_mass_source": rel(s["path"]) if _ok(s) else None,
               "items": RECONCILIATION + RECONCILIATION_MORE, "grant_docs_present": docs, "grant_dir": str(gd)}
        self.cell("prov.reconciliation", sec, "grant reconciliation", "all", FILLED, value=val,
                  sources=[s["path"] if _ok(s) else None] + [str(gd / n) for n, ok in docs.items() if ok],
                  reader="sum(run_meta.body_masses_kg) + RECONCILIATION")
        return val

    def disclosures(self) -> list:
        F, R = self.F["ctx"].dir, (self.R["ctx"].dir if self.R else self.pl / "R18")
        rd = lambda p: rel(p) if Path(p).exists() else None  # noqa: E731
        items = [
            ("camera_unverified", "Camera use is UNVERIFIED",
             "C4.7 'depth helps' failed (depth 0.830 vs zeroed-depth 0.815 field_hard es1-2; +0.015 < 0.069) and the "
             "diagnostic showed no omni row is perception-limited (S1 with a perfect scan equals a blind student). The "
             "student is gated on completion paired against T*; its camera use is not demonstrated.",
             "CHARTER 2026-10-17 (user: accept, disclose)", [self.pl / "preflight/stepC/checks/C4_depth.json"]),
            ("sim_plant", "Every payload number is a sim-plant number",
             "No rod-force limit is enforced (the hip rod is ~3.4 kN static at 0 kg against the 2000 N rating); every "
             "Task 3 cell is labelled 'sim plant, force side not enforced'; actuator load is report-only (D8).",
             "PLAN Payload; DESIGN header D8", []),
            ("r18_plive_fail", "R18 P-live FAILED at iteration 20 (cuDNN)",
             "R18 s1 vs X43: bitwise through step 19 (P-early PASS under the 4-ULP amendment), then diverges from "
             "iteration 20 (55 / 144 tags bitwise at +200): the cuDNN history-encoder numerics differ by build. "
             "STOP-R accepted: R18 is a statistical second sample; same build + seed is bitwise (to step 19), across "
             "builds reproduction is statistical.", "CHARTER 2026-10-19 (user decision 1)",
             [R / "replay/s1_plive_posthoc.json"]),
            ("r18_live_watch_dead", "The live watch never ran on R18 s1-p2a",
             "driver_run_dir resolved the wrong run dir, so no driver stage s1-p2a had live rules / a P-live read; "
             "post hoc the live rules would not have killed any of the six stages (R18/replay/live_posthoc.json); "
             "those walker gates' live_collapse lines passed vacuously (p2a re-gated WATCH). Fixed and re-pinned "
             "(task4-pin-2026-10-19b).", "CHARTER 2026-10-19 (finding 1, re-pin)", [R / "replay/live_posthoc.json"]),
            ("r18_p3a_accepted", "R18 p3a paired 0-kg check FAILED, accepted as chance",
             "move_turn ratio only (margin use 1.66 / 1.52; z 15.8 / 13.6), read as the known continuation drift; the "
             "chance rate is unmodelled.", "CHARTER 2026-10-20", [R / "halts/p3a.1.json"]),
            ("r18_p3b_silent_exclusion", "R18 p3b was excluded from T* silently",
             "p3b (135-kg ceiling) failed its paired 0-kg check on turn yaw_progress es2 (worse by 0.0100 vs 0.0091, "
             "use 1.10); the queue excluded it without a notification; R18's own T* record lists only p3a (T* = p3a, "
             "90 kg). Fixed for F19 (every payload head listed).", "CHARTER 2026-10-21 (decision 2)",
             [R / "gates/p3b.json", R / "stage_reads/tstar.json"]),
            ("r18_sn_amend", "R18 SN G1 stillness lines amended to WATCH (v1.1)",
             "G1 taps/s 1.042 / 1.125 vs <= 1 on suite es1 / es2 (half the stands tap >= 13 times in 12 s); the "
             "recorded chance rate 0.0 was a degenerate fixed-t artefact (now 'rate unknown'). SH's K5 / K6 stays "
             "the decisive stillness test.", "CHARTER 2026-10-21 (decision 1, re-pin v1.1)", [R / "halts/sn.1.json"]),
            ("r18_sh_accepted", "R18 SH HALT accepted with disclosure",
             "K4 restore misses 13 / 10 per 800 vs the lazy control's 7 / 3 (0 exit falls); K6 bin 0 es2 homed 0.925 "
             "vs 0.94; the K6 bin 1 / bin 4 drop is walker-caused (the R_h control beside T* fails the same lines).",
             "CHARTER 2026-10-21 15:20 (user: accept, disclose)", [R / "halts/sh.1.json", R / "stand/sh_lazy_rh_read.json"]),
            ("f19_p3a_accepted", "F19 p3a paired 0-kg check FAILED, accepted",
             "Real 0-kg losses on move_turn (0.786 -> 0.687 / 0.791 -> 0.694) and diag (0.997 -> 0.959 / 0.999 -> "
             "0.963); the fwd / back / side 'drops' are p2a's overshoot (1.05-1.07) moving to 1.0, which the "
             "one-sided ratio rule counted as worse (a gate-semantics finding). Chance rate unmodelled.",
             "CHARTER 2026-10-22 06:50 (user: accept, disclose)", [F / "halts/p3a.1.json"]),
            ("v12_rule_change", "Thresholds v1.1 -> v1.2: the 0-kg translate statistic changed mid-run",
             "By user decision (amend + switch now) the decisive translate statistic became track_err = |r_head - 1| - "
             "|r_p2a - 1| (increase fails); move_turn stays a one-sided drop; the raw drop is WATCH. F19's payload "
             "heads were re-gated: p3b PASS (use 0.66; raw back drop 0.068 / 0.071 WATCH) -> T* = p3b; p3a still "
             "FAIL (accepted). The p3a-based distill was stopped at ~it 26800 (SIGTERM, rc 134, no SIGKILL) and a "
             "fresh distill T4F19_st_a2 ran from p3b. R18 is unchanged under v1.2 (T* = p3a).",
             "CHARTER 2026-10-22 22:36 / 2026-10-23", [self.th_path, F / "gates/history/p3b.1.json",
                                                         F / "stage_reads/history/tstar.1.json"]),
            ("v12_a", "v1.2 disclosure (a)",
             "The step-8 pilot pair (PP_p3a_s18 vs S1 s18; a rate source, not a gate) reads FAIL under v1.2 on "
             "track_err back (its back overshoot grew 1.087 -> 1.162): the intended case.", "CHARTER 2026-10-23", []),
            ("v12_b", "v1.2 disclosure (b)",
             "The student full block's '+0.05 tracking' WATCH still includes a raw one-sided ratio row: an "
             "overshooting T* can raise that WATCH for a student tracking closer to 1.0 (reported only).",
             "CHARTER 2026-10-23", []),
            ("v12_c", "v1.2 disclosure (c)",
             "The re-gated p3a / p3b gate headers keep v1.1; their 0-kg sub-records say v1.2.", "CHARTER 2026-10-23",
             [F / "gates/p3a.json", F / "gates/p3b.json"]),
            ("v12_d", "v1.2 disclosure (d)",
             "R18's walker gates re-read today as WATCH (the live_collapse flag added 2026-10-19), not as recorded "
             "PASS; pre-existing, no decisive line moves.", "CHARTER 2026-10-23", []),
            ("f19_sh_reanchor", "F19 SH HALT re-anchored on the lazy R_h control",
             "K6 homed share bin 4 es1 0.935 vs 0.945 (S1-era line). The R_h control beside T* (p3b) fails the same "
             "line (0.89 / 0.845): walker-caused, so the line re-anchors on the control (DESIGN 3.6) and the head "
             "passes every K6 cell and beats its teacher (bin 4 0.935 / 0.95 vs 0.89 / 0.845; K4 2 / 3 vs 5 / 5). "
             "Continued with `continue --accept-chance sh` -- the pipeline's only no-re-gate path; NOT a chance "
             "acceptance. The user can overrule at STOP-END. (The HALT went unnoticed 23:00-23:30 after a session "
             "restart, which lengthened the park.)", "CHARTER 2026-10-23", [F / "halts/sh.1.json",
                                                                              F / "stand/sh_lazy_rh_read.json"]),
            ("level0_unarmed", "The student curve's level 0 reuses the unarmed selection stamps",
             "Same checkpoint, env, rows and env seeds; armed 0:0 is strictly bitwise to off (C1 armed00_vs_off); "
             "their run_meta shows payload None, not 0 kg.", "fill plan (step 11), CHARTER 2026-10-24",
             [self.pl / "preflight/stepC/checks/C1.json"]),
            ("sel_world_rows", "The student selection read world rows only", self.selection_disclosure()["text"],
             "step 11 (not in the CHARTER)", [self.spec_path]),
            ("f_recorded_build", "F step / obstacle / hard-band stamps are the recorded 2026-09-08/09 build",
             "No plant key, no code_state.json, env_vars {} (obstacle launcher not recorded); same checkpoint "
             "e79c4a90. Only the slow row was read at the pin (0.75 vs the recorded 0.79: the +-0.04 process-state "
             "split); the step row is re-read at the pin by F_step_pin when present.", "CHARTER 2026-10-24 decision 5",
             []),
            ("f_mass_inertia", "The F added-mass curve scales chassis inertia",
             "KRABBY_DR_MASS = Isaac Lab randomize_rigid_body_mass (startup, add, chassis body) with recompute_inertia "
             "True: chassis inertia scales with the mass ratio, CoM unchanged (no deck-CoM composite). A raw per-level "
             "table; the stable-at-L lines do not apply to F.", "CHARTER 2026-10-24 decision 5", []),
            ("np_not_converged", "N_p is the clamp, not a converged value",
             "N_p = 3000 iterations per payload stage: no pilot read settled within R of the read 500 its earlier; "
             "max_stable is reported 'at N_p iterations per stage' and p3b's limiting factor is budget_bound.",
             "CHARTER 2026-10-19 (v1 freeze)", []),
            ("student_not_replay", "A same-seed student rerun is not a replay",
             "C4.5 determinism: same-seed student distillation is not bitwise (x_rule report_only for SN / SH too).",
             "CHARTER 2026-10-17 / 2026-10-19", []),
            ("h_limits", "H limits",
             "H = the six tracked v1 stamps as committed + the 9800 README's TensorBoard numbers; forward-only, not "
             "re-runnable (critic input 1151 vs 1149); the grant's 6300 / 9800 do not load (CPU check in Task 0 -> H).",
             "CHARTER 2026-10-24 decision 5; DESIGN 6", []),
            ("f_limits", "F limits",
             "F is the forward 3a student on its own task and rows: lateral / yaw / grid / sweep / side22 / omni field "
             "cells are N/A (no command channel), not 0; its forward rows fall 21-29 / 100 at 0 kg.", "DESIGN 6", []),
        ]
        out = []
        for i, (did, title, text, charter, recs) in enumerate(items, 1):
            out.append({"n": i, "id": did, "title": title, "text": text, "charter": charter,
                        "records": [x for x in (rd(p) for p in recs) if x]})
        return out

    # ------------------------------------------------------------------ appendix (R18)
    def appendix(self, trained) -> dict:
        sec = "Appendix (R18)"
        R, rr = self.R, getattr(self, "rec_other", None) or run_records(self.spec_path, self.other)
        out = {"run": self.other, "tstar": self.tstar_block(R, rr)}
        ts, sel = out["tstar"], R["sel"]
        out["columns"] = {
            "T1": f"T* = {ts.get('stage')} {Path(ts.get('ckpt') or '?').name} (sha {str(ts.get('sha256'))[:8]}) of "
                  f"{self.other}, Walk-Play rows (the T* stage's own gate rows at 0 kg; suite@1 = the driver stamp)",
            "St": f"student {sel.get('cand')} (sha {str(sel.get('sha256'))[:8]}) of {self.other}, Student-Play "
                  "--policy-role student (its selection rows)"}
        out["columns_short"] = (f"T1 = {ts.get('stage')} on Walk-Play rows; St = {sel.get('cand')} on Student-Play "
                                "(--policy-role student)")
        cdir = R["ctx"].dir / "clip"
        clips = [rel(p) for p in sorted(cdir.glob("*.mp4"))] if cdir.is_dir() else []
        ign = git_ignored([L.REPO / p for p in clips]) if clips else {}
        out["clip"] = {"paths": clips, "gitignored": ign, "local_only": bool(clips) and all(p in ign for p in clips)}
        cols = {"T1": R["T1"], "St": R["St"]}
        out["task0"], out["field"], out["side22"] = {}, {}, {}
        for col, S in cols.items():
            res = {}
            for es in (1, 2):
                s = S.get(f"suite@{es}")
                if _ok(s):
                    res[es] = {"stamp": rel(s["path"]), "categories": self.get(read_categories, s["path"]),
                               "orientation": self.get(read_orientation, s["path"])}
            out["task0"][col] = res
            out["field"][col] = {r: (self.get(read_field, *[S[f"{r}@{es}"]["path"] for es in (1, 2) if _ok(S.get(f"{r}@{es}"))])
                                     if any(_ok(S.get(f"{r}@{es}")) for es in (1, 2)) else None) for r in FIELD_ROWS}
            s = S.get("side22@1")
            out["side22"][col] = self.get(read_side22, s["path"]) if _ok(s) else None
        curves = {}
        for sid in ("p3a", "p3b"):
            rows = L.read_json(R["ctx"].rows_path(sid)) or {}
            lv = {}
            for k, v in rows.items():
                p = k.split("@")
                if len(p) == 3:
                    lv.setdefault(int(p[2][1:]), {})[f"{p[0]}@{p[1]}"] = self.recorded((v or {}).get("stamp"))
            rd = self.levels_read(lv, trained)
            g = rr["gates"].get(sid) or {}
            curves[sid] = {"levels": rd, **curve_summary(rd), "gate_curve": g.get("curve"),
                           "gate_max_stable": g.get("max_stable"), "limiting_factor": g.get("limiting_factor"),
                           "verdict": g.get("verdict")}
        out["curves"] = curves
        out["stand"] = self.sh_block(R["ctx"].dir, rr["gates"].get("sh")) if rr["gates"].get("sh") else None
        sn = rr["gates"].get("sn") or {}
        out["sn"] = {"verdict": sn.get("verdict"), "families": self.k_families(sn.get("decisive")),
                     "watch_flags": [f.get("id") for f in sn.get("watch_flags") or []]} if sn else None
        blk = ((rr["gates"].get("sel") or {}).get("blocks") or {}).get(R["sel"].get("cand")) or {}
        out["student"] = {"cand": R["sel"].get("cand"), "sha256": R["sel"].get("sha256"),
                          "pairing": [{k: x.get(k) for k in ("row", "student", "tstar", "line", "pass")}
                                      for x in blk.get("decisive") or [] if ".capability.completion." in x.get("id", "")]}
        self.cell("app.payload", sec, "R18 appendix: payload heads considered for T* (p3a CHOSEN, p3b EXCLUDED) + the "
                  "R18 p3b curve", "R18", FILLED, value={"tstar": out["tstar"], "curves": curves},
                  sources=[R["ctx"].dir / "gates/p3b.json", R["ctx"].rows_path("p3b"), R["ctx"].dir / "gates/p3a.json",
                           R["ctx"].rows_path("p3a"), R["ctx"].dir / "stage_reads/tstar.json"],
                  reader="payload_read.level_reads + the gate records", label=SIM,
                  note="R18's own T* record lists only p3a: the p3b exclusion is disclosed (CHARTER 2026-10-21)")
        self.cell("app.columns", sec, "R18 appendix columns (T* p3a, student c27000, SN / SH_T4R18) on its gated world "
                  "rows", "R18", FILLED, value={k: out[k] for k in ("columns", "task0", "field", "side22", "stand", "sn",
                                                                     "student", "clip")},
                  sources=[R["ctx"].dir / "deliverable.json", R["ctx"].rows_path("p3a"), R["ctx"].rows_path("sel"),
                           R["ctx"].dir / "stand/sh_read.json"], reader="the T1 / St readers",
                  note="no body / grid / sweep rows on R18's T* or student")
        return out

    # ------------------------------------------------------------------ headline
    def headline(self, doc) -> dict:
        ts = self.F["tstar"] or {}
        t3 = doc["task3"]
        pair_ = (doc["student"].get("pairing") or {}).get("recorded") or []
        sh = doc["stand"].get("T1") or {}
        sn = doc["stand"].get("T1_sn") or {}
        pv = doc["provenance"]
        g1 = (pv.get(self.run) or {}).get("gpu_h") or {}
        g2 = (pv.get(self.other) or {}).get("gpu_h") or {} if self.other else {}
        pre = pv.get("preflight") or {}
        s11 = pv.get("step11_rows") or {}
        lazy = (sh.get("lazy_control") or {}).get("head_verdicts_reanchored")
        return {
            "deliverable": {"run": self.run, "tstar": ts.get("stage"), "tstar_ckpt": rel(ts.get("ckpt")),
                            "tstar_sha256": ts.get("sha256"), "student": self.F["sel"].get("cand"),
                            "student_sha256": self.F["sel"].get("sha256"),
                            "sn": (sn.get("head"), sn.get("sha256")), "sh": (sh.get("head"), sh.get("sha256")),
                            "note": "the deliverable choice is the user's at STOP-END"},
            "tstar_max_stable_kg": ts.get("max_stable"), "tstar_first_unstable_kg": t3["T1"].get("first_unstable"),
            "tstar_limiting_factor": (t3["T1"].get("limiting_factor") or {}).get("name"), "label": SIM,
            "n_p": self.spec.n_p(),
            "student_vs_tstar": [{"row": x.get("row"), "student": x.get("student"), "tstar": x.get("tstar"),
                                  "line": x.get("line"), "pass": x.get("pass")} for x in pair_],
            "student_max_stable_kg": t3["St"].get("max_stable"), "student_stable_through_kg": t3["St"].get("stable_through"),
            "student_curve_open_from_kg": t3["St"].get("open_from"), "student_curve_read": t3["St"].get("curve"),
            "student_curve_pending": [lv for lv, r in (t3["St"].get("levels") or {}).items()
                                      if isinstance(r, dict) and r.get("status") == PEND],
            "stand": {"sn": sn.get("verdict"), "sh": sh.get("verdict"), "sh_halt_lines": sh.get("halt_lines"),
                      "sh_reanchored_verdicts": lazy,
                      "sh_k5_reanchored_text": ((sh.get("lazy_control") or {}).get("k5_invalid") or {}).get("text")},
            "gpu_h": {self.run: g1.get("total"), (self.other or "other"): g2.get("total") if g2 else None,
                      "preflight_A_wall_h": (pre.get("A") or {}).get("h"), "preflight_C_wall_h": (pre.get("C") or {}).get("h"),
                      "preflight_C_excl_hang_h": (pre.get("C") or {}).get("h_excl_hang"),
                      "step11_rows": s11.get("measured_h"), "step11_rows_n": s11.get("n_rows_measured")},
            "gpu_h_total": pv.get("gpu_h_total"),
            "counts": doc["counts"]}


# ------------------------------------------------------------------ path-keyed wrappers (memo keys are strings)
def sweep_read_paths(okp_json: str, th_path) -> dict:
    okp = {int(k): v for k, v in json.loads(okp_json).items()}
    return sweep_read(okp, L.read_json(Path(th_path)) or {})


def level_set_read_paths(stamps_json: str, level_kg, th_path, trained_csv: str) -> dict:
    return level_set_read(json.loads(stamps_json), float(level_kg), L.read_json(Path(th_path)) or {},
                          [c for c in trained_csv.split(",") if c])


# ============================================================================== markdown
def _mark(doc, cid) -> str:
    """What an empty table cell shows: the cell's own status (never a number)."""
    c = next((x for x in doc["cells"] if x["id"] == cid), None)
    if not c or c["status"] == PEND:
        return "pending"
    return NA if c["status"] == NOTAPP else ""


def _pair_es(v: dict, es_key, k, doc_mark: str) -> str:
    """``es1 / es2`` of one key from {es: {key: value}}; an env seed not read yet shows the cell's mark."""
    out = []
    for es in (1, 2):
        if es not in (v or {}):
            out.append(doc_mark or "pending")
        else:
            x = (v.get(es) or {})
            x = x.get(es_key) if es_key is not None else x
            out.append(fmt_n(x.get(k), (x.get("_n") or {}).get(k)) if isinstance(x, dict) else fmt(None))
    return " / ".join(out)


def _status_line(doc, cid) -> str:
    c = next((x for x in doc["cells"] if x["id"] == cid), None)
    if not c:
        return PENDING
    if c["status"] == PEND:
        return c.get("why") or PENDING
    if c["status"] == NOTAPP:
        return f"{NA}: {c.get('why')}"
    return ""


def md(doc: dict) -> str:
    out = ["# M18 Task 4 -- baseline vs final", "",
           f"_Generated by `TP/report.py` from records only on {doc['generated_at']} (spec `{doc['spec']}`, thresholds "
           f"{doc['thresholds']['version']} `{str(doc['thresholds']['sha256'])[:8]}`). The JSON beside this file is the "
           "machine record; every cell below names its source in the JSON `cells` list._", ""]
    out += md_headline(doc)
    out += ["## Columns", ""] + table(["column", "what"], [[k, v] for k, v in doc["columns"].items() if v])
    out += ["**Labels.** `N/A` = not defined / not obtainable (reason given), never 0. `pending (row X)` = the step-11 "
            f"GPU row X is not on disk yet; `{PENDING}` = a run record is not written yet. `{DERIVED}` = computed here "
            f"on CPU from recorded stamps (no harness key). Every payload number: _{SIM}_. 1 N caveat: {MISREAD}. "
            f"**Small-n rule (every Task 0 / Task 2 table):** {N_RULE}.", ""]
    out += md_task0(doc) + md_task2(doc) + md_task3(doc) + md_stand(doc) + md_student(doc) + md_provenance(doc)
    out += ["## Grant reconciliation", ""]
    rc = doc["reconciliation"]
    out += [f"- Robot mass from run_meta (`{rc.get('robot_mass_source')}`): sum(body_masses_kg) = "
            f"{fmt(rc.get('robot_mass_kg_from_run_meta'))} kg."]
    out += [f"- {x}" for x in rc["items"]] + [""]
    out += ["## Disclosures", ""]
    for d in doc["disclosures"]:
        out.append(f"{d['n']}. **{d['title']}** ({d['charter']}). {d['text']}"
                   + (f" Records: {', '.join('`' + r + '`' for r in d['records'])}." if d["records"] else ""))
    out.append("")
    if doc.get("appendix"):
        out += md_appendix(doc)
    out += ["## Cell index", "", f"{doc['counts']}", ""]
    out += table(["id", "column", "status", "metric", "why / note"],
                 [[c["id"], c["column"], c["status"], c["metric"],
                   c.get("why") or c.get("note") or ""] for c in doc["cells"]])         # full text: never sliced
    if doc.get("read_errors"):
        out += ["### Read errors (reported, never numbers)", ""] + [f"- {e}" for e in doc["read_errors"]] + [""]
    return "\n".join(out) + "\n"


def md_headline(doc) -> list:
    h = doc["headline"]
    d = h["deliverable"]
    sv = "; ".join(f"{x['row']} {fmt(x['student'])} vs {fmt(x['tstar'])} (line {fmt(x['line'])}, "
                   f"{'PASS' if x['pass'] else 'FAIL'})" for x in h["student_vs_tstar"]) or PENDING
    st = h["stand"]
    lines = ["## Headline", "",
             f"- **Deliverable run {d['run']}** (the user confirms at STOP-END): T* = {d['tstar']} "
             f"`{d['tstar_ckpt']}` (sha `{str(d['tstar_sha256'])[:8]}`); student {d['student']} "
             f"(sha `{str(d['student_sha256'])[:8]}`); stand nets {d['sn'][0]} (`{str(d['sn'][1])[:8]}`) / "
             f"{d['sh'][0]} (`{str(d['sh'][1])[:8]}`).",
             f"- **T* max_stable {fmt(h['tstar_max_stable_kg'])} kg** (first unstable {fmt(h['tstar_first_unstable_kg'])} "
             f"kg; limiting factor {h['tstar_limiting_factor']}, at N_p = {h['n_p']} iterations per stage) -- _{SIM}_.",
             f"- **Student completion vs T*** (paired, Student-Play): {sv}. Student curve: "
             + (f"max_stable {fmt(h['student_max_stable_kg'])} kg" if h["student_max_stable_kg"] is not None else
                f"max_stable not claimed yet (level {h['student_curve_open_from_kg']} kg not read yet; levels read: "
                + ", ".join(f"{k} kg {'stable' if v else 'UNSTABLE'}" for k, v in (h["student_curve_read"] or {}).items())
                + ")")
             + (f"; levels pending: {h['student_curve_pending']}" if h["student_curve_pending"] else "") + f" -- _{SIM}_.",
             f"- **Stand:** SN {st['sn']}; SH {st['sh']} on "
             + (", ".join(x.replace("stand.D2.", "") for x in st["sh_halt_lines"] or []) or "no line")
             + " -> re-anchored on the lazy R_h control (DESIGN 3.6; walker-caused): "
             + (", ".join(f"{k} {v}" for k, v in (st["sh_reanchored_verdicts"] or {}).items() if k != "K5") or PENDING)
             + (f" (K5 reads {st['sh_k5_reanchored_text']})." if st.get("sh_k5_reanchored_text") else "."),
             md_gpu_total(h.get("gpu_h_total") or {}),
             f"- **Cells:** {h['counts'].get(FILLED, 0)} filled, {h['counts'].get(PEND, 0)} pending, "
             f"{h['counts'].get(NOTAPP, 0)} N/A.", ""]
    return lines


def md_gpu_total(t: dict) -> str:
    """The headline's one labelled total GPU-h line (every term named; the hang shown with and without)."""
    if not t or t.get("total_excl_hang") is None:
        return f"- **Total GPU-h (measured):** {PENDING} (no measured term yet)."
    s = (f"- **Total GPU-h (measured): {fmt(t['total_excl_hang'], 2)}** = "
         + " + ".join(f"{x['term']} {fmt(x['h'], 2)} ({x['what']})" for x in t["terms"]))
    if t.get("total_incl_hang") is not None:
        s += f". With the {fmt(t.get('hang_h'), 2)}-h C7 hang counted: {fmt(t['total_incl_hang'], 2)}"
    if t.get("missing"):
        s += ". Not counted (no measurement): " + ", ".join(x["term"] for x in t["missing"])
    return s + "."


def _fn(d, k: str) -> str:
    """One median of a reader dict under :data:`N_RULE` (its ``_n`` beside it)."""
    d = d or {}
    return fmt_n(d.get(k), (d.get("_n") or {}).get(k))


def _cat_rows(cols: dict, cat: str, keys, marks: dict | None = None) -> list:
    rows = []
    for k, label in keys:
        r = [label]
        for col, res in cols.items():
            r.append(_pair_es(res, None, k, (marks or {}).get(col, "pending")))
        rows.append(r)
    return rows


def md_task0(doc) -> list:
    t0 = doc["task0"]
    out = ["## Task 0 -- gait, tracking, actuator load (0 kg)", "",
           "Each cell is `es1 / es2` (suite env seeds 1 / 2). Per-category medians of the harness "
           "(`aggregate.suite.by_category`, run_phases.cat_summary); the 50 N set is the gait read.", ""]
    cols = {c: v for c, v in t0["columns"].items()}
    status = {c: _status_line(doc, f"t0.cat.{c}") for c in cols}
    for c, s in status.items():
        if s:
            out.append(f"- **{c}**: {s}")
    out.append("")
    for cat in CATS:
        out += [f"### {cat}", ""]
        cc = {col: {es: ((r.get("categories") or {}).get("by_category") or {}).get(cat) for es, r in res.items()}
              for col, res in cols.items()}
        ee = {col: {es: (r.get("episode") or {}).get(cat) for es, r in res.items()} for col, res in cols.items()}
        vv = {col: {es: (r.get("vel_rms") or {}).get(cat) for es, r in res.items()} for col, res in cols.items()}
        mk = {col: _mark(doc, f"t0.cat.{col}") for col in cols}
        rows = _cat_rows(cc, cat, CAT_KEYS, mk) + _cat_rows(ee, cat, EP_KEYS, mk) + _cat_rows(
            vv, cat, (("vy_rms", f"vy tracking RMS (m/s, {DERIVED})"), ("wz_rms", f"wz tracking RMS (rad/s, {DERIVED})")), mk)
        out += table(["metric"] + list(cols), rows)
    out += ["### Stamp-level orientation / height and actuator load at 0 kg", "", f"_{SIM}; applied torque = the "
            "implicit-actuator estimate._", ""]
    rows = []
    for col, res in cols.items():
        for es, r in sorted(res.items()):
            o, a = r.get("orientation") or {}, r.get("actuator") or {}
            cap, rod = a.get("cap_share") or {}, a.get("rod_force_n") or {}
            rows.append([col, es, fmt(o.get("h_meas_median")), fmt(o.get("roll_rms")), fmt(o.get("pitch_rms")),
                         pair(cap.get("hip"), cap.get("knee"), 4),
                         pair((rod.get("hip") or {}).get("p50"), (rod.get("hip") or {}).get("p90"), 0),
                         pair((rod.get("knee") or {}).get("p50"), (rod.get("knee") or {}).get("p90"), 0),
                         fmt(a.get("sum_foot_force_ratio"))])
    out += table(["column", "es", "h_meas median (m)", "roll RMS", "pitch RMS", "cap share hip / knee",
                  "hip rod p50 / p90 (N; rated 2000, self-lock 2800)", "knee rod p50 / p90 (N; rated 500, self-lock 4500)",
                  "sum foot force / weight"], rows)
    # H
    h = t0["H"]
    out += ["### H -- v1 stand-ins as committed (forward-only; not re-runnable)", ""]
    rows = []
    for n, r in (h.get("v1") or {}).items():
        if not r or "unreadable" in r:
            rows.append([n, PENDING] + [""] * 9)
            continue
        bh = r.get("by_hold") or {}
        e = r.get("episode") or {}
        rows.append([n, Path(str(r.get("checkpoint"))).name, _fn(r, "tripod_score"), _fn(r, "tippy_tap_fraction"),
                     _fn(r, "slip_ratio"), fmt(r.get("schedule_completion_rate")),
                     " ".join(f"{k}:{_fn(v, 'vx_ratio') if (v or {}).get('vx_ratio') is not None else _fn(v, 'vx_ratio_derived')}"
                              for k, v in bh.items()),
                     " ".join(f"{k}:{_fn(v, 'vy_rms')}/{_fn(v, 'wz_rms')}" for k, v in bh.items()),
                     " ".join(f"{k}:{_fn(v, 'duty_hold')}" for k, v in bh.items()),
                     f"{_fn(e, 'roll_rms')} / {_fn(e, 'pitch_rms')}",
                     f"{_fn(e, 'action_rate')} / {_fn(e, 'sign_reversals')}"])
    out += table(["stamp", "checkpoint", "tripod (1 N)", "tippy (1 N)", "slip", "completion",
                  f"vx ratio per hold (actual / cmd mean, {DERIVED})",
                  "vy / wz RMS per hold (blind channels)", "duty per hold (1 N)", "roll / pitch RMS",
                  "action rate / reversals"], rows)
    g = h.get("grant") or {}
    ref = None
    for c in doc["cells"]:
        if c["id"] == "t0.H.grant":
            ref = (c.get("value") or {}).get("today_reference")
    out += [f"- The grant's checkpoints, CPU `torch.load` today: " + "; ".join(
        f"{k}: std key `{v.get('std_key')}` {v.get('std_shape')}, critic.0.weight {v.get('critic_shape')} -> "
        f"{'loads' if v.get('loadable_today') else 'NOT LOADABLE'}" for k, v in g.items())
            + (f" (today's T*: `{(ref or {}).get('std_key')}`, critic input {(ref or {}).get('critic_in')})." if ref else "."),
            f"- 9800 README (TensorBoard text only): {h.get('readme_9800')}",
            f"- {NA}: the 50 N set, direction_error / cross_track and actuator load (no raw npz, no 50 N keys, the "
            "checkpoints do not load).", ""]
    # F
    f = t0["F"]
    fnote = next((c.get("note") for c in doc["cells"] if c["id"] == "t0.F.basic"), "")
    out += ["### F -- forward 3a student (model_24995) on its own rows", "", f"_{fnote}_", ""]
    rows = []
    for k, r in (f.get("reads") or {}).items():
        b = r.get("basic") or {}
        if "unreadable" in b:
            rows.append([k, b["unreadable"]] + [""] * 8)
            continue
        e = b.get("episode") or {}
        bh = b.get("by_hold") or {}
        rows.append([k, f"`{r['stamp']}`", _fn(b, "tripod_score"), _fn(b, "tippy_tap_fraction"),
                     fmt(b.get("schedule_completion_rate")), str(b.get("termination_reasons")),
                     " ".join(f"{h_}:{_fn(v, 'vx_ratio')}" for h_, v in bh.items()),
                     " ".join(f"{h_}:{_fn(v, 'duty_hold')}" for h_, v in bh.items()),
                     f"{_fn(e, 'roll_rms')} / {_fn(e, 'pitch_rms')}",
                     f"{_fn(e, 'action_rate')} / {_fn(e, 'sign_reversals')}"])
    out += table(["row", "stamp", "tripod (1 N)", "tippy (1 N)", "completion", "terminations", "vx ratio per hold",
                  "duty per hold (1 N)", "roll / pitch RMS", "action rate / reversals"], rows)
    rows = []
    for k, r in (f.get("reads") or {}).items():
        rg = (r.get("rigidity") or {}).get("by_hold") or {}
        for h_, v in rg.items():
            rows.append([k, h_, v.get("n")] + [_fn(v, x) for x in ("support_feet", "steps_per_m", "stride_m", "taps_per_m",
                                                                   "swing_apex", "step_rate_hz", "direction_error",
                                                                   "cross_track_ratio")])
    out += [f"F 50 N rigidity set + travel direction per hold window ({DERIVED}; 10-s holds):", ""]
    out += table(["row", "hold", "n", "support_feet", "steps_per_m", "stride ~6/spm", "taps_per_m", "swing_apex",
                  "step_rate_hz", "direction_error", "cross_track"], rows)
    a = f.get("actuator_0kg") or {}
    if a and a.get("available"):
        out += [f"F actuator load at 0 kg (slow row at the pin; {SIM}): cap share hip / knee "
                f"{pair((a['cap_share'] or {}).get('hip'), (a['cap_share'] or {}).get('knee'), 4)}; hip rod p50 / p90 "
                f"{pair(a['rod_force_n']['hip']['p50'], a['rod_force_n']['hip']['p90'], 0)} N; knee rod p50 / p90 "
                f"{pair(a['rod_force_n']['knee']['p50'], a['rod_force_n']['knee']['p90'], 0)} N.", ""]
    return out


def md_task2(doc) -> list:
    t2 = doc["task2"]
    out = ["## Task 2 -- omnidirectional capability", ""]
    # grid
    out += ["### Command grid (grid__omni_body; 9 in-envelope cells gated at p2a as WATCH, 11 reported)", ""]
    for col in ("T0", "T1", "St"):
        s = _status_line(doc, f"t2.grid_body.{col}")
        if s:
            out.append(f"- **{col}**: {s}")
    out.append("")
    rows = []
    cells = None
    for col in ("T0", "T1", "St"):
        v = (t2.get("grid_body") or {}).get(col) or {}
        for es, g in sorted(v.items()):
            if isinstance(g, dict) and g.get("cells"):
                cells = cells or list(g["cells"])
    for c in cells or []:
        r = [c, "yes" if c in GRID_IN_ENVELOPE else "no"]
        for col in ("T0", "T1", "St"):
            v = (t2.get("grid_body") or {}).get(col) or {}
            cv = {es: ((g or {}).get("cells") or {}).get(c) for es, g in v.items() if isinstance(g, dict)}
            mk = _mark(doc, f"t2.grid_body.{col}")
            for k in ("survival", "abs_speed_err", "abs_wz_err", "heading_drift"):
                r.append(_pair_es(cv, None, k, mk))
        rows.append(r)
    hdr = ["cell", "in env."]
    for col in ("T0", "T1", "St"):
        hdr += [f"{col} surv", f"{col} abs(v-cmd)", f"{col} abs(wz-cmd)", f"{col} drift"]
    out += [f"|achieved - cmd| from the harness medians and the pooled drift p90 are {DERIVED}.", ""] + table(hdr, rows)
    for col in ("T0", "T1", "St"):
        v = (t2.get("grid_body") or {}).get(col) or {}
        ds = [f"es{es} median {fmt((g.get('drift_pooled_in_envelope') or {}).get('median'))} / p90 "
              f"{fmt((g.get('drift_pooled_in_envelope') or {}).get('p90'))}" for es, g in sorted(v.items())
              if isinstance(g, dict) and g.get("drift_pooled_in_envelope")]
        if ds:
            out.append(f"- {col} heading drift pooled over the in-envelope cells (rad, {DERIVED}): {'; '.join(ds)} "
                       "(DESIGN's p2a reference: median <= .0100, p90 <= .0207, unvalidated, WATCH)")
    out.append("")
    # turn levels + body suite
    for fam, title, keys in (("turnlevels_body", "Turn levels (turnlevels__omni_body)",
                              (("net_yaw_ratio", "net yaw ratio"), ("wz_ratio", "wz ratio"), ("survival", "survival"))),
                             ("suite_body", "Body suite through body_hold (suite__omni_body)",
                              (("ratio", "ratio"), ("body_ratio", "body_ratio"), ("cross_body", "cross_body"),
                               ("net_yaw_ratio", "net yaw ratio"), ("survival", "survival")))):
        out += [f"### {title}", ""]
        for col in ("T0", "T1", "St"):
            s = _status_line(doc, f"t2.{fam}.{col}")
            if s:
                out.append(f"- **{col}**: {s}")
        out.append("")
        cats = []
        for col in ("T0", "T1", "St"):
            for es, v in ((t2.get(fam) or {}).get(col) or {}).items():
                for c in (v or {}):
                    if c not in cats:
                        cats.append(c)
        rows = []
        for c in cats:
            for k, lab in keys:
                r = [c, lab]
                for col in ("T0", "T1", "St"):
                    v = (t2.get(fam) or {}).get(col) or {}
                    r.append(_pair_es({es: (x or {}).get(c) for es, x in v.items()}, None, k,
                                      _mark(doc, f"t2.{fam}.{col}")))
                rows.append(r)
        out += table(["category", "metric", "T0", "T1", "St"], rows)
    # sweep
    out += ["### 16-bin direction sweep at 0.22 m/s (dirsweep__omni_body; WATCH only)", ""]
    for col in ("T0", "T1", "St"):
        s = _status_line(doc, f"t2.dirsweep_body.{col}")
        if s:
            out.append(f"- **{col}**: {s}")
    out.append("")
    sw = t2.get("dirsweep_body") or {}
    bins = []
    for col in ("T0", "T1", "St"):
        for b in ((sw.get(col) or {}).get("bins") or {}):
            if b not in bins:
                bins.append(b)
    rows = []
    for b in bins:
        r = [b]
        for col in ("T0", "T1", "St"):
            x = ((sw.get(col) or {}).get("bins") or {}).get(b) or {}
            r.append(f"{_fn(x, 'ratio')} (|r-1| {fmt(x.get('abs_dev'))} vs {fmt(x.get('line'))})" if x
                     else _mark(doc, f"t2.dirsweep_body.{col}"))
        rows.append(r)
    out += table(["bin", "T0 ratio", "T1 ratio", "St ratio"], rows)
    for col in ("T0", "T1", "St"):
        j = (sw.get(col) or {}).get("max_adjacent_jump") or {}
        if j:
            out.append(f"- {col} discontinuity (max adjacent-bin jump, circular): " + "; ".join(
                f"{k} {fmt(v.get('value'))} vs line {fmt(v.get('line'))}" for k, v in j.items()))
    out.append("")
    # field + side22
    out += ["### Field rows (completion es1 / es2; pooled) and side22", ""]
    rows = []
    for r in FIELD_ROWS:
        row = [r]
        for col in ("T0", "T1", "St"):
            v = ((t2.get("field") or {}).get(col) or {}).get(r)
            row.append(f"{pair(*((v.get('per_env_seed') or []) + [None, None])[:2], 2)} ({fmt(v.get('rate'))})"
                       if isinstance(v, dict) and "rate" in v else _mark(doc, f"t2.field.{col}"))
        rows.append(row)
    row = ["side22 class"]
    for col in ("T0", "T1", "St"):
        v = (t2.get("side22") or {}).get(col)
        row.append("/".join(v["class"]) if isinstance(v, dict) and v.get("class") else _mark(doc, f"t2.side22.{col}"))
    rows.append(row)
    out += table(["row", "T0", "T1", "St"], rows)
    out += [f"- H: {_status_line(doc, 't2.H.na') or 'filled'}", f"- F: {_status_line(doc, 't2.F.na') or 'filled'}",
            f"- field_suite: {_status_line(doc, 't2.field_suite.na')}"]
    fa = next((c.get("value") for c in doc["cells"] if c["id"] == "t2.F.field"), None) or {}
    for k, v in fa.items():
        out.append(f"- F forward field analogue {k}: completion {fmt(v.get('completion'))} (n {v.get('n')}; "
                   f"{v.get('termination_reasons')}) -- recorded 2026-09-08/09 build, not re-read at the pin.")
    out.append("")
    return out


def _level_rows(levels: dict) -> list:
    rows = []
    for lv in sorted(levels, key=lambda x: int(x)):
        r = levels[lv]
        if not isinstance(r, dict) or "stable" not in r:
            rows.append([lv, (r or {}).get("why") or (r or {}).get("unreadable") or PENDING] + [""] * 6)
            continue
        a = r.get("actuator") or {}
        cap, rod = a.get("cap_share") or {}, a.get("rod_force_n") or {}
        v = r.get("values") or {}
        rows.append([lv, "stable" if r["stable"] else "UNSTABLE: " + ", ".join(x.replace("p3a.stable90.", "")
                                                                             for x in r.get("failed") or []),
                     fmt(v.get("p3a.stable90.field_light_completion")), fmt(v.get("p3a.stable90.h_meas.suite.es1")),
                     pair(cap.get("hip"), cap.get("knee"), 4),
                     pair((rod.get("hip") or {}).get("p50"), (rod.get("hip") or {}).get("p90"), 0),
                     pair((rod.get("knee") or {}).get("p50"), (rod.get("knee") or {}).get("p90"), 0),
                     fmt(a.get("sum_foot_force_ratio"))])
    return rows


LEVEL_HDR = ["kg", "stable at L (a-priori p3a.stable90 lines)", "field_light completion", "h_meas suite es1 (m)",
             "cap share hip / knee", "hip rod p50 / p90 (N)", "knee rod p50 / p90 (N)", "sum foot force / weight"]


def md_task3(doc) -> list:
    t3 = doc["task3"]
    out = [f"## Task 3 -- payload (every number: _{SIM}_)", ""]
    for col, title in (("T0", "T0 -- S1 (B0b curve, preflight C3)"), ("T1", "T1 -- T* curve"), ("St", "St -- student curve")):
        v = t3.get(col) or {}
        ms_txt = (f"max_stable **{fmt(v.get('max_stable'))} kg**" if v.get("max_stable") is not None or
                  v.get("open_from") is None else f"max_stable **not claimed** (level {v.get('open_from')} kg not read "
                  f"yet; stable through {fmt(v.get('stable_through'))} kg)")
        lf = v.get("limiting_factor") or {}
        lf_txt = (f"limiting factor {lf.get('name')} ({lf.get('why')})" if lf else
                  "limiting factor not assessed (no unstable level read yet)")
        out += [f"### {title}", "", f"{ms_txt}, first unstable "
                f"{fmt(v.get('first_unstable'))} kg; {lf_txt}."
                + (f" Recomputed curve matches the gate record: {fmt(v.get('matches_gate'))}." if col == "T1" else "")
                + (f" Recomputed curve matches the C3 record: {fmt(v.get('matches_record'))}." if col == "T0" else ""), ""]
        if col == "St":
            out += [f"_{v.get('level0_note')}_", ""]
            s = _status_line(doc, "t3.curve.St")
            if s:
                out += [f"- {s}", ""]
        out += table(LEVEL_HDR, _level_rows(v.get("levels") or {}))
    f = t3.get("F") or {}
    out += ["### F -- forward 3a added-mass raw table", "", f"_{f.get('label')}_", ""]
    rows = []
    for lv, r in sorted((f.get("levels") or {}).items(), key=lambda x: int(x[0])):
        if r.get("status"):
            rows.append([lv, r.get("why")] + [""] * 6)
            continue
        a, o = r.get("actuator") or {}, r.get("orientation") or {}
        cap = a.get("cap_share") or {}
        rows.append([lv, f"{fmt(r.get('chassis_kg'))} ({'ok' if r.get('mass_check') else 'MISMATCH'})",
                     fmt(r.get("completion")), r.get("falls"),
                     " ".join(f"{h}:{fmt(x)}" for h, x in (r.get("tracking_ratio_by_hold") or {}).items()),
                     fmt(o.get("h_meas_median")), pair(o.get("roll_rms"), o.get("pitch_rms")),
                     pair(cap.get("hip"), cap.get("knee"), 4)])
    out += table(["+kg", "chassis kg (158.757 + L)", "completion", "falls", "vx ratio per hold", "h_meas (m)",
                  "roll / pitch RMS", "cap share hip / knee"], rows)
    out += [f"- H: {_status_line(doc, 't3.curve.H')}", ""]
    ts = t3.get("tstar") or {}
    out += ["### The paired 0-kg check and the payload heads considered for T*", "",
            f"Rule: {ts.get('rule')}", ""]
    rows = []
    cons = ts.get("considered") or []
    whys, notes = _full_notes([(x.get("stage"), x.get("why")) for x in cons], 160, "why")
    for x, why in zip(cons, whys):
        rows.append([x.get("stage"), x.get("status"), x.get("max_stable"), x.get("zero_kg"),
                     fmt(x.get("max_margin_used"), 2), "CHOSEN" if x.get("chosen") else "", why])
    out += table(["head", "status", "max_stable", "0-kg (v1.2)", "max margin use", "", "why"], rows)
    out += notes + ([""] if notes else [])
    for sid, h in (ts.get("heads") or {}).items():
        b = h.get("zero_kg_before_regate")
        if b:
            out.append(f"- {sid} before the v1.2 re-gate ({(b.get('thresholds') or {}).get('version')}): "
                       f"{b.get('verdict')} at {fmt(b.get('max_margin_used'), 2)} ({b.get('fails')}).")
        z = h.get("zero_kg") or {}
        if z.get("watch_exceeds"):
            out.append(f"- {sid} WATCH: {z.get('watch_exceeds')}")
    out.append("")
    return out


def md_sh_tables(t0: dict, t1: dict) -> list:
    """The SH family table (gate lines passed / failed, the lazy-control re-anchor per family; K5's INVALID with the
    reason the record gives) + the per-clause K3 / K4 / K6 table (SH_T17 beside S1; R_h beside T*; the head re-anchored
    on it). Shared by the deliverable run and the appendix run."""
    out, rows = [], []
    lc = t1.get("lazy_control") or {}
    k5 = lc.get("k5_invalid") or {}
    fams = sorted(set((t1.get("families") or {})) | {"K1", "K2", "K3", "K4", "K5", "K6"})
    for K in fams:
        f = (t1.get("families") or {}).get(K) or {}
        re_v = (lc.get("head_verdicts_reanchored") or {}).get(K)
        if K == "K5" and re_v == "INVALID" and k5.get("text"):
            re_v = k5["text"]
        rows.append([K, (t0.get("verdicts") or {}).get(K), f"{f.get('pass')}/{f.get('n')}" if f else PENDING,
                     "; ".join(f"{x['id'].split('.', 3)[-1]} {fmt(x['value'])} vs {fmt(x['line'])}" for x in f.get("fails") or []),
                     re_v if lc else "no lazy control read"])
    out += table(["family", "SH_T17 (T0)", f"{t1.get('head')} gate lines passed", "failed lines",
                  "re-anchored on the lazy control"], rows)
    if not lc:
        return out
    keys = sorted(set(lc.get("control_R_h") or {}) | set(lc.get("head_reanchored") or {}) | set(t0.get("key") or {}))
    rows = []
    for k in keys:
        a = (t0.get("key") or {}).get(k) or {}
        b = (lc.get("control_R_h") or {}).get(k) or {}
        c = (lc.get("head_reanchored") or {}).get(k) or {}
        rows.append([k, fmt(a.get("value")), fmt(b.get("value")), f"{fmt(c.get('value'))} vs {fmt(c.get('line'))}",
                     "PASS" if c.get("ok") else ("FAIL" if c.get("ok") is False else NA)])
    out += [f"Key K3 / K4 / K6 clauses (SH_T17 beside S1; the R_h control beside T*; {t1.get('head')} re-anchored on "
            f"it; record `{lc.get('record')}`):", ""]
    out += table(["clause", "SH_T17", "R_h beside T*", f"{t1.get('head')} re-anchored (value vs line)", "re-anchored"],
                 rows)
    m = lc.get("measured") or {}
    if m:
        out += [f"- Lazy control rows: {m.get('n_rows')} rows, measured {fmt(m.get('h'), 2)} h ({m.get('rule')}).", ""]
    return out


def md_refs_tstar(rt: dict) -> list:
    """SN's REFS(T*) numbers (sn_calib.json rows['REFS']) beside the SN head, one row per clause (es1 / es2 paired
    when the record has both); a family's identity (INVALID) reasons are printed in full under the table."""
    if not rt or not rt.get("available"):
        return [f"- SN REFS(T*): {(rt or {}).get('why') or NA + ': no REFS(T*) record'}", ""]
    out = [f"SN REFS(T*) = `{rt.get('row')}` (`{rt.get('record')}` rows['REFS']) beside {rt.get('head')}. "
           f"{rt.get('note')}.", ""]
    rows, notes = [], []
    for G, g in (rt.get("gates") or {}).items():
        inv = g.get("invalid") or {}
        verdict = (f"{g.get('verdict')} (expected {g.get('expected') or '--'}, lands {g.get('lands')}"
                   + (f"; {inv.get('n')} identity reasons, below" if inv.get("n") else "") + ")")
        if inv.get("n"):
            notes.append(f"- {G} identity reasons ({inv['n']}; {', '.join(inv.get('kinds') or []) or 'other'}), "
                         f"first: {inv.get('first')}")
        grouped = {}
        for c, x in (g.get("clauses") or {}).items():
            m = re.search(r" es([12])(?= |$)", c)
            base = (c[:m.start()] + " es*" + c[m.end():]) if m else c
            grouped.setdefault(base, []).append((int(m.group(1)) if m else 0, c, x))
        first = True
        for base, items in grouped.items():
            if len(items) == 1:
                base = items[0][1]
            items = sorted(items)

            def p(key):
                return " / ".join(fmt((x or {}).get(key)) for _, _, x in items)
            rows.append([G if first else "", verdict if first else "", base, p("refs"), p("head"), p("line")])
            first = False
    out += table(["gate", "REFS(T*) reader verdict", "clause (es1 / es2)", "REFS(T*)", "SN head", "line"], rows)
    out += notes + ([""] if notes else [])
    rep = rt.get("reported") or {}
    if rep:
        ks = sorted({k for v in rep.values() for k in (v or {})})
        out += [f"REFS(T*) reported, no line (es1 / es2): " + "; ".join(
            f"{k} {' / '.join(fmt((rep.get(e) or {}).get(k)) for e in sorted(rep))}" for k in ks), ""]
    return out


def md_stand(doc) -> list:
    s = doc["stand"]
    out = ["## Stand (teacher-side, 0 kg) -- K1-K6 beside SH_T17", ""]
    out += md_sh_tables(s.get("T0") or {}, s.get("T1") or {})
    sn = s.get("T1_sn") or {}
    out += [f"### SN ({sn.get('head')}): {sn.get('verdict')}", ""]
    out += table(["family", "lines passed"], [[k, f"{v['pass']}/{v['n']}"] for k, v in (sn.get("families") or {}).items()])
    out += [f"- WATCH lines (G1 / G2 stillness since v1.1): " + "; ".join(
        f"{x['id'].split('.', 3)[-1]} {fmt(x['value'])} vs {fmt(x['line'])}" for x in sn.get("watch_lines") or []), ""]
    out += md_refs_tstar(sn.get("refs_tstar") or ({"why": f"{PENDING} (no gates/sn.json)"} if not sn else {}))
    out += [f"- H / F / St: {_status_line(doc, 'stand.na')}", ""]
    return out


def md_student(doc) -> list:
    s = doc["student"]
    p = s.get("pairing") or {}
    out = [f"## Student ({s.get('cand')}, sha `{str(s.get('sha256'))[:8]}`)", "",
           "### Paired completion vs T* (Student-Play, same rows / env seeds; margin max(0.05, 2 SE_diff, split 0.0))", ""]
    out += table(["row", "student", "T*", "SE_diff", "margin (binds)", "line", "pass"],
                 [[x.get("row"), fmt(x.get("student")), fmt(x.get("tstar")), fmt(x.get("se_diff")),
                   f"{fmt(x.get('margin'))} ({x.get('binds')})", fmt(x.get("line")), fmt(x.get("pass"))]
                  for x in p.get("recorded") or []])
    out += md_floors(p.get("floors") or [])
    out += [f"- Rebuilt from the stamps (select_student.completion_pairs over thresholds.field_episodes): matches the "
            f"record: {fmt(p.get('rebuilt_matches_record'))}.",
            f"- Block verdict {p.get('verdict')}; WATCH flags {p.get('watch_flags')}; tracking-WATCH flags "
            f"{p.get('tracking_watch_flags')} (disclosure (b)).",
            f"- Prescreen rank (select_student.rank): {p.get('ranked_iters')}; the first full-block PASS is selected.",
            f"- Stable at {(s.get('stable90') or {}).get('stable_at')} kg (WATCH): {fmt((s.get('stable90') or {}).get('stable'))} "
            f"(failed {(s.get('stable90') or {}).get('failed')}) -- {SIM}."]
    lv = s.get("live") or {}
    if lv and "unreadable" not in lv:
        out.append(f"- Live checks (replayed on CPU, `{lv.get('run_dir')}`, its {lv.get('first')}..{lv.get('last')}): "
                   f"reads {lv.get('reads')}; T* tail {lv.get('tstar_tail')}; replay kill at "
                   f"{(lv.get('replay') or {}).get('killed_at')}; final rc {lv.get('final_rc')}.")
    else:
        out.append(f"- Live checks: {lv.get('unreadable') if lv else PENDING}")
    out.append(f"- Camera use: **{(s.get('camera') or {}).get('verdict')}** -- {(s.get('camera') or {}).get('why')} "
               f"(C4.7: {(s.get('camera') or {}).get('c4_depth')}).")
    ce = s.get("clip_export") or {}
    out.append(f"- Veto clip (never a gate): {list((ce.get('clip') or {}).keys()) or _status_line(doc, 'st.clip')} in "
               f"`{ce.get('clip_dir')}`; deploy export in `{ce.get('exported_dir')}`: "
               + ", ".join(f"{k} ({v['bytes']} B, `{v['sha256'][:8]}`)" for k, v in (ce.get("exported_deploy") or {}).items()))
    out.append(md_clip_local(ce.get("clip_paths") or [], ce.get("gitignored") or {}))
    out.append("")
    return out


def md_clip_local(paths: list, ign: dict) -> str:
    """The veto clips are local-only: which paths `git check-ignore` reports ignored, and by which rule."""
    if not paths:
        return "- Veto clip files: none on disk."
    rules = sorted({ign[p] for p in paths if p in ign})
    loc = [p for p in paths if p in ign]
    tracked = [p for p in paths if p not in ign]
    s = ("- The veto clip mp4s are **local-only** (not in git; `git check-ignore` at report time: "
         + (", ".join(f"`{r}`" for r in rules) or "no rule") + "): " + ", ".join(f"`{p}`" for p in loc) + ".")
    if tracked:
        s += " NOT ignored (would be committable): " + ", ".join(f"`{p}`" for p in tracked) + "."
    return s


def md_floors(floors: list) -> list:
    """The student's selection floor lines (sel.floor.<stat>.<cat>.es<k> + side22) as one compact table."""
    if not floors:
        return []
    by, other = {}, []
    for x in floors:
        m = re.match(r"^sel\.floor\.(survival|ratio)\.(\w+)\.es(\d)$", str(x.get("id")))
        if m:
            by.setdefault(m.group(2), {}).setdefault(m.group(1), {})[int(m.group(3))] = x
        else:
            other.append(x)
    rows = []
    for cat, st in by.items():
        r = [cat]
        for stat in ("survival", "ratio"):
            d = st.get(stat) or {}
            lines = sorted({fmt(v.get("line")) for v in d.values()})
            r += [" / ".join(fmt((d.get(e) or {}).get("value")) for e in (1, 2)), " / ".join(lines)]
        r.append("PASS" if all(v.get("pass") for d in st.values() for v in d.values()) else "FAIL")
        rows.append(r)
    return ([f"Selection floor lines ({len(floors)} lines of `sel.floor.*`; es1 / es2):", ""]
            + table(["category", "survival", "survival line (>=)", "ratio", "ratio line (>=)", "pass"], rows)
            + [f"- `{x.get('id')}`: {fmt(x.get('value'))} vs line {fmt(x.get('line'))} -- "
               f"{'PASS' if x.get('pass') else 'FAIL'}" for x in other] + ([""] if other else []))


def halt_rate_text(h: dict) -> str:
    """A HALT's chance-rate cell: the rate (or 'rate unknown' / n/a) + the K6 S1-era caveat where it applies."""
    t = h.get("chance_rate_text") or NA
    return f"{t} ({h['chance_rate_caveat']})" if h.get("chance_rate_caveat") else t


def md_provenance(doc) -> list:
    pv = doc["provenance"]
    out = ["## Provenance", "", f"- **Selection disclosure:** {pv['selection_disclosure']['text']}", ""]
    for run in (doc["run"], doc.get("other")):
        if not run or run not in pv:
            continue
        r = pv[run]
        wc = r.get("wallclock") or {}
        out += [f"### {run}: measured wall-clock and GPU-h per stage (queue sentinels; parks separated)", ""]
        rows = []
        for sid, v in (wc.get("stages") or {}).items():
            info = v.get("segment_info") or []
            split = any(x.get("superseded") for x in info)
            rows.append([sid + (" (total)" if split else ""), fmt(v["active_h"], 2), fmt(v["active_h"], 2),
                         fmt(v.get("train_to_first_row_h"), 2),
                         "; ".join(f"{a[5:16]}->{b[5:16]}" for a, b in v["segments"])])
            if split:
                for i, x in enumerate(info, 1):
                    la, sp = x.get("launch") or {}, x.get("superseded")
                    what = (f"{la.get('name')} from {la.get('from')}" if la.get("name") else "no launch in segment")
                    what += (f" -- **SUPERSEDED** (stopped by hand for the v1.2 retarget: DRIVER HALT, then the retarget "
                             f"an hour later; last logged iteration "
                             f"{sp.get('last_iteration')}; {sp.get('charter')}; GPU work, result abandoned)" if sp
                             else " -- the run's distill of record")
                    rows.append([f"  {sid} [{i}/{len(info)}] {what}", fmt(x["h"], 2), fmt(x["h"], 2), "",
                                 f"{x['from'][5:16]}->{x['to'][5:16]}"])
        out += table(["stage", "active wall-clock (h)", "GPU-h", "train -> first row (h)", "segments"], rows)
        out += [f"- Total active {fmt(wc.get('total_active_h'), 2)} h; parks {fmt(wc.get('total_park_h'), 2)} h (0 GPU-h); "
                f"span {fmt(wc.get('span_h'), 2)} h ({wc.get('first')} .. {wc.get('last')})."]
        out += [f"  - park {p['from'][5:16]} -> {p['to'][5:16]} {fmt(p['h'], 2)} h: {p['why']}" for p in wc.get("parks") or []]
        g = r.get("gpu_h") or {}
        out += [f"- GPU-h: {fmt(g.get('total'), 2)} = stages {fmt(g.get('stages_active_h'), 2)}"
                + (f" (incl. {fmt(g.get('superseded_h'), 2)} h of SUPERSEDED segments: GPU work, result abandoned)"
                   if g.get("superseded_h") else "")
                + f" + lazy control {fmt(g.get('lazy_control_h'), 2)} ({g.get('rule')}).",
                f"- {r.get('driver_wall_h_note')}", ""]
        out += [f"Gate verdicts ({run}):", ""]
        out += table(["stage", "status", "gate", "thresholds", "WATCH flags"],
                     [[k, v.get("status"), v.get("gate"), v.get("thresholds"), ", ".join(v.get("watch_flags") or [])]
                      for k, v in (r.get("verdicts") or {}).items()])
        out += [f"HALTs ({run}):", ""]
        hs = r.get("halts") or []
        fls, fl_notes = _full_notes([(h["stage"], "; ".join(f"{x['id']} {fmt(x['value'])} vs {fmt(x['line'])}"
                                                             for x in h["lines"])) for h in hs], 200, "failed lines")
        whs, wh_notes = _full_notes([(h["stage"], h.get("charter") or h.get("resolution_why") or "") for h in hs],
                                    160, "CHARTER / why")
        out += table(["stage", "class", "resolution", "chance rate", "failed lines", "CHARTER / why"],
                     [[h["stage"], h["class"], h["resolution"], halt_rate_text(h), fl, wh]
                      for h, fl, wh in zip(hs, fls, whs)])
        out += fl_notes + wh_notes + ([""] if fl_notes or wh_notes else [])
        out += [f"- Pin: {r.get('pin')}", f"- Amendments: " + "; ".join(
            f"{a.get('stage')} {a.get('kind', 'amend')} at {a.get('at')} ({a.get('charter')})" for a in r.get("amendments") or []), ""]
    pre = pv.get("preflight") or {}
    pc = pre.get("C") or {}
    hang = pc.get("hang") or {}
    s11 = pv.get("step11_rows") or {}
    bs = s11.get("n_rows_by_state") or {}
    out += ["### Preflight and step-11 wall-clock", "",
            f"- Phase A {fmt((pre.get('A') or {}).get('h'), 2)} h ({(pre.get('A') or {}).get('rule')}); Phase C "
            f"{fmt(pc.get('h'), 2)} h incl. C7 {fmt(pc.get('C7_h'), 2)} h -- {pc.get('note')}.",
            (f"- Phase C without the hang: {fmt(pc.get('h_excl_hang'), 2)} h = {fmt(pc.get('h'), 2)} - "
             f"{fmt(hang.get('h'), 2)} (hang {hang.get('from')} -> {hang.get('to')}, `{hang.get('source')}`); it still "
             f"includes {fmt(pc.get('cpu_check_jobs_h'), 2)} h of {pc.get('cpu_check_note')}." if hang else
             "- Phase C hang: not measurable from the records (no GPU-h figure without it)."),
            f"- Step-11 rows: {fmt(s11.get('measured_h'), 2)} h measured over the {s11.get('n_rows_measured')} rows "
            f"with both an S11 ROW and an S11 DONE sentinel ({s11.get('n_rows_seen')} rows in the log: "
            f"{bs.get('done', 0)} done, {bs.get('running', 0)} running, {bs.get('skip', 0)} skipped; "
            f"`{s11.get('source')}`).", ""]
    out += ["### Total GPU-h", "", md_gpu_total(pv.get("gpu_h_total") or {}), ""]
    st = pv.get("stack") or {}
    hw = st.get("hardware") or {}
    out += ["### Hardware, external stack, pins and tags", "",
            f"- Hardware: {hw.get('gpus')}; {hw.get('cpu')} ({hw.get('cpu_threads')} threads); {hw.get('mem_gb')} GB; "
            f"host {hw.get('host')} ({hw.get('note')}).",
            f"- External stack (pipeline.lock.json): {st.get('external')}",
            f"- Lock: code_sha `{str((st.get('lock') or {}).get('code_sha'))[:8]}`, eval_sha "
            f"`{str((st.get('lock') or {}).get('eval_sha'))[:8]}`, lock sha `{str((st.get('lock') or {}).get('sha256'))[:8]}` "
            f"({(st.get('lock') or {}).get('n_files')} files, written {(st.get('lock') or {}).get('written_at')}).",
            "- Tags (local, never pushed):"] + [f"  - `{t}`" for t in st.get("tags") or []] + [
            f"- Step-11 deliverable tag: `{BASELINE_TAG}` (local annotated, CHARTER 2026-10-24 decision 4; never pushed): "
            + ("present." if any(t.split()[0] == BASELINE_TAG for t in st.get("tags") or [])
               else "not created yet at report time."), ""]
    rs = pv.get("run_sum") or {}
    out += [f"- Chance-HALT run sum (recommended / standard): {fmt(rs.get('recommended'))} / {fmt(rs.get('standard'))} -- "
            f"{rs.get('note')}.", ""]
    am = pv.get("amendment_v12") or {}
    out += ["### Thresholds amendment v1.1 -> v1.2", "",
            f"- {am.get('charter')}: {(am.get('from') or {}).get('version')} (`{str((am.get('from') or {}).get('sha256'))[:8]}`, "
            f"0-kg translate statistic {((am.get('from') or {}).get('zero_kg') or {}).get('translate_statistic')}) -> "
            f"{(am.get('to') or {}).get('version')} (`{str((am.get('to') or {}).get('sha256'))[:8]}`, "
            f"{((am.get('to') or {}).get('zero_kg') or {}).get('translate_statistic')}).", f"- Rule: {am.get('rule')}.", ""]
    if pv.get("r18_reproduced"):
        r = pv["r18_reproduced"]
        out += ["### R18 process-reproduced verdict + X stand replay", "", f"- {r.get('verdict')}.",
                f"- P-config per stage {r.get('pconfig')}; P-early {r.get('pearly')}; P-live {r.get('plive')}; "
                f"post hoc live kills {r.get('live_posthoc_kills')}; X {r.get('x')}.", ""]
    ks = pv.get("keep_set") or {}
    out += ["### Keep-set", "",
            f"- Student checkpoint `{ks.get('student_ckpt')}`: sha256 `{ks.get('student_sha256')}`, "
            f"{ks.get('student_bytes')} B > the {ks.get('head_max_bytes')} B (64 MiB) bundle_experiment guard: "
            f"{fmt(ks.get('exceeds_guard'))}; recorded sha matches: {fmt(ks.get('recorded_sha_matches'))}. {ks.get('note')}.",
            f"- bundle.yaml: {ks.get('bundle_yaml')}", ""]
    return out


def md_appendix(doc) -> list:
    a = doc["appendix"]
    out = [f"## Appendix -- {a['run']} (the other run)", ""]
    cols = a.get("columns") or {}
    if cols:
        out += table(["column", "what"], [[k, v] for k, v in cols.items()])
    ts = a.get("tstar") or {}
    out += [f"- T* = {ts.get('stage')} (max_stable {ts.get('max_stable')} kg, {SIM}); candidates in its own record: "
            f"{ts.get('candidates')} (p3b's exclusion is disclosed, not in this record)."]
    for sid, c in (a.get("curves") or {}).items():
        out += [f"- {sid} curve (recomputed): {c.get('curve')} -> max_stable {c.get('max_stable')} (gate "
                f"{c.get('gate_max_stable')}, {c.get('verdict')}; limiting factor "
                f"{(c.get('limiting_factor') or {}).get('name')}); 0-kg: "
                f"{((ts.get('heads') or {}).get(sid) or {}).get('zero_kg')}"]
    out.append("")
    ctxt = a.get("columns_short") or "; ".join(f"{k} = {v}" for k, v in cols.items())
    rows = []
    for col in ("T1", "St"):
        for es, r in sorted((a.get("task0") or {}).get(col, {}).items()):
            bc = (r.get("categories") or {}).get("by_category") or {}
            rows.append([col, es] + [f"{fmt((bc.get(c) or {}).get('survival'))} / {_fn(bc.get(c), 'ratio')}" for c in CATS])
    out += [f"Task 0 survival / ratio per category (columns: {ctxt}):", ""] + table(["column", "es"] + list(CATS), rows)
    rows = []
    for r in FIELD_ROWS:
        rows.append([r] + [fmt(((a.get("field") or {}).get(col) or {}).get(r, {}).get("rate")
                               if isinstance(((a.get("field") or {}).get(col) or {}).get(r), dict) else None) for col in ("T1", "St")])
    rows.append(["side22"] + ["/".join(((a.get("side22") or {}).get(col) or {}).get("class") or []) or NA for col in ("T1", "St")])
    out += [f"Field completion (pooled es1-2) and side22 (columns: {ctxt}):", ""] + table(["row", "T1", "St"], rows)
    stn = a.get("stand") or {}
    out += [f"### {a['run']} SH {stn.get('head')}: {stn.get('verdict')} -- K1-K6 and the lazy-control re-anchor", ""]
    if (stn.get("lazy_control") or {}).get("control_R_h"):
        out += md_sh_tables((doc.get("stand") or {}).get("T0") or {}, stn)
        out += [f"- Re-anchored verdicts per family: {(stn.get('lazy_control') or {}).get('head_verdicts_reanchored')} "
                "(a line the R_h control beside T* fails too is walker-caused and re-anchors on the control; DESIGN 3.6)."]
    else:
        out += [f"- Failing gate lines {[x['id'].split('.', 3)[-1] for f in (stn.get('families') or {}).values() for x in f['fails']]}; "
                f"re-read on the lazy control (per family; the record carries no per-clause read): "
                f"{(stn.get('lazy_control') or {}).get('head_verdicts_reanchored')}."]
    out += [f"- SN: {(a.get('sn') or {}).get('verdict')} (WATCH flags {(a.get('sn') or {}).get('watch_flags')}).",
            f"- Student {(a.get('student') or {}).get('cand')}: " + "; ".join(
                f"{x['row']} {fmt(x['student'])} vs {fmt(x['tstar'])} ({'PASS' if x['pass'] else 'FAIL'})"
                for x in (a.get("student") or {}).get("pairing") or []),
            md_clip_local((a.get("clip") or {}).get("paths") or [], (a.get("clip") or {}).get("gitignored") or {}), ""]
    return out


# ============================================================================== README
def _flag_present(path: Path, flag: str) -> bool:
    """A CLI flag exists in a script: an argparse ``"--flag"`` / a bash ``--flag)`` case / a usage line."""
    if not path.is_file():
        return False
    t = path.read_text(errors="replace")
    return any(s in t for s in (f'"{flag}"', f"'{flag}'", f"{flag})", f"{flag} ", f"{flag}=", f"[{flag}"))


def readme_checks(spec) -> list:
    """Every script path and flag the README names, checked on disk (argparse / bash case source)."""
    TPd = HERE
    PK = L.REPO / "parkour"
    al = Path(_al_path()) if _al_path() else None
    checks = [
        (TPd / "pipeline.py", ["gen", "check", "lock", "run", "continue", "status", "gate", "report", "--spec", "--run",
                               "--check", "--stage", "--thresholds", "--accept-chance", "--amend", "--charter",
                               "--retry", "--abort", "--retarget-tstar", "--no-relaunch", "--why"]),
        (TPd / "report.py", ["--spec", "--run", "--other", "--out", "--rows-log", "--readme"]),
        (TPd / "gate_row.sh", ["TP_OUT", "TP_ROW_TIMEOUT", "TP_LEVEL_FILE", "TP_STRIP_KEYS", "TP_ROW_DRY"]),
        (TPd / "distill.sh", ["--name", "--checkpoint", "--env-file", "--seed", "--iterations", "--num-envs", "--task",
                              "--experiment"]),
        (TPd / "payload_read.py", ["--spec", "level", "actuator", "--stamp", "--thresholds", "--trained", "--level"]),
        (TPd / "gate_read.py", ["--spec", "--run", "--stage", "--thresholds", "--out", "--lines-only", "--stamp",
                                "--trained"]),
        (TPd / "live_watch.py", ["check", "replay", "--run-dir", "--kind", "--stage", "--source", "--tstar", "--first",
                                 "--every"]),
        (TPd / "thresholds.py", ["--inputs", "--out", "--recommended", "--self-check", "--sims", "--seed"]),
        (TPd / "bridge_read.py", ["--ref", "--new", "--mode", "--ignore", "--no-default-ignore", "--allow-new-key",
                                  "--json"]),
        (TPd / "replay_read.py", ["pconfig", "pearly", "onset", "--new", "--old", "--ref-sha", "--rule", "--ref-log",
                                  "--new-log"]),
        (TPd / "select_student.py", ["candidates", "--run-dir", "--start", "--offsets"]),
        (L.TOOLS / "launch_candidates.sh", ["--campaign-dir"]),
        (L.TOOLS / "run_candidates.py", ["--reference-chain", "--seed"]),
        (L.TOOLS / "bundle_experiment.py", ["--dry-run", "--init", "--check", "--index"]),
        (L.HARNESS, ["--manifest", "--scenario", "--env-seed", "--checkpoint", "--task", "--policy-role",
                     "--trained-categories", "--command-adapter", "--save-raw", "--no-plot", "--output-root"]),
        (PK / "scripts/rsl_rl/play.py", ["--task", "--num_envs", "--real-time"]),
        (PK / "scripts/rsl_rl/play_stream_video.py", ["--task", "--num_envs", "--video", "--video_length"]),
        (PK / "scripts/rsl_rl/cli_args.py", ["--checkpoint"]),
    ]
    if al:
        checks.append((al, ["--headless", "--enable_cameras"]))
    st = spec.d.get("stand") or {}
    for key, script, flags in (("sn_scripts", "sn_dagger.sh", ["--wait-idle"]),
                               ("sh_scripts", "sh_dagger.sh", ["--wait-idle"]),
                               ("sh_scripts", "sh_dagger.py", ["--mode", "--arm", "--seed", "--iterations", "--deadband",
                                                               "--trim-branch", "--trim-min", "--warm-start", "--warm-sha"]),
                               ("sn_scripts", "sn_dagger.py", ["--mode", "--arm", "--seed", "--iterations"]),
                               ("sn_scripts", "sn_eval_rows.sh", []), ("sh_scripts", "sh_eval_rows.sh", []),
                               ("sh_scripts", "sh_record_video.sh", ["--dry-run", "--plan", "--seed", "--terrain",
                                                                     "--prefix"]),
                               ("sh_scripts", "sh_record_video.py", ["--stand-ckpt", "--stand-mask", "--deadband",
                                                                     "--eval-run", "--allow-reference-mismatch"])):
        if st.get(key):
            checks.append((L.OMNI / st[key] / script, flags))
    if st.get("p2_run_arm"):
        checks.append((L.OMNI / st["p2_run_arm"], ["--row-eval"]))
    out = []
    for path, flags in checks:
        ok = path.is_file()
        miss = [f for f in flags if not _flag_present(path, f)] if ok else flags
        out.append({"path": rel(path), "exists": ok, "flags": flags, "missing_flags": miss})
    return out


def _al_path():
    try:
        import importlib.util  # noqa: PLC0415

        s = importlib.util.find_spec("isaaclab")
        if s and s.submodule_search_locations:
            p = Path(list(s.submodule_search_locations)[0]) / "app/app_launcher.py"
            return str(p) if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None
    return None


def log_cmds(log: Path) -> list:
    """Every ``# <time> $ <command>`` line of a TP log (the queue writes each launch verbatim), in order."""
    if not log.is_file():
        return []
    return [m.group(1) for m in (re.match(r"^# \S+ \$ (.*)$", ln) for ln in log.read_text(errors="replace").splitlines())
            if m]


def first_cmd(log: Path) -> str | None:
    c = log_cmds(log)
    return c[0] if c else None


def _from_parkour(p) -> str | None:
    """A recorded path as a command run from ``parkour/`` spells it: repo-relative -> ``$PWD/...`` (absolute at run
    time, checkout-independent in the README); an absolute path outside the checkout as recorded."""
    if not p:
        return None
    s = str(p)
    if s.startswith("/"):
        return s
    return "$PWD/" + s.removeprefix("parkour/") if s.startswith("parkour/") else "$PWD/../" + s


def readme_md(doc: dict, spec, run: str) -> str:
    #: the repo root as the README's commands spell it (they run from the repo root, or say ``cd parkour``): ``$PWD``,
    #: never this checkout's absolute root (the generated README is committed)
    R = "$PWD"
    pl_rel = L.rel(spec.dir)
    rd = spec.run_dir(run)
    h = doc["headline"]["deliverable"]
    ts = doc["task3"]["tstar"]
    tp_rel = L.rel(HERE)
    PY = L.PY
    spec_rel = L.rel(spec.path)
    tag = next((t.split()[0] for t in reversed((doc["provenance"].get("stack") or {}).get("tags") or [])
                if t.startswith("task4-pin-")), "task4-pin-<latest>")
    ext = (doc["provenance"].get("stack") or {}).get("external") or {}
    hw = (doc["provenance"].get("stack") or {}).get("hardware") or {}
    lock = (doc["provenance"].get("stack") or {}).get("lock") or {}
    logs = rd / "tp_logs"
    st_all = log_cmds(logs / "st.launcher.log")
    st_cmd = st_all[-1] if st_all else None
    sn_cmd = first_cmd(logs / "sn.train.log")
    sh_cmd = first_cmd(logs / "sh.train.log")
    row_cmd = first_cmd(logs / f"rows_{ts.get('stage')}.log") if ts.get("stage") else None
    refs_cmd = first_cmd(logs / "sn.refs.log")
    try:
        sspec = L.load_by_path(L.OMNI / "config/crab_hex/omni_student_spec.py", "tp_omni_student_spec")
        refused = list(sspec.STUDENT_RUNNER_KNOBS_REFUSED)
    except Exception:  # noqa: BLE001
        refused = None
    envf = rd / "envs" / f"{ts.get('stage')}.env"
    env = L.read_env_file(envf) if envf.is_file() else {}
    in_env_refused = [k for k in (refused or []) if k in env]
    carry = ((spec.d.get("carry_env") or {}).get("common") or {})
    req = {"KRABBY_OMNI_CLOCK_HIDE": env.get("KRABBY_OMNI_CLOCK_HIDE"), **{k: env.get(k, v) for k, v in carry.items()}}
    checks = readme_checks(spec)
    bad = [c for c in checks if not c["exists"] or c["missing_flags"]]
    sel = doc["student"]
    ce = sel.get("clip_export") or {}
    ks = doc["provenance"].get("keep_set") or {}
    lines = [
        "# M18 Task 4 -- gated from-scratch omni pipeline (campaign README)", "",
        f"_GENERATED by `{tp_rel}/report.py --readme` from the run records on {doc['generated_at']}; edit the records or "
        "the generator, not this file. Every script path and flag below was checked on disk at generation time "
        f"({len(checks) - len(bad)}/{len(checks)} scripts clean{'; see UNVERIFIED at the end' if bad else ''})._", "",
        "## 1. What it is", "",
        "The M18 Task 4 deliverable: one committed pipeline (`pipeline.yaml` + `TP/pipeline.py`) that trains the omni "
        "crab-hexapod teacher from scratch through every stage -- walker s1 -> c4 -> c5 -> c6 -> c6x -> p2a, payload "
        "p3a / p3b / p3c, T*, the 4-camera depth student (st, sel), the stand nets SN -> SH -- with every stage "
        "transition gated on harness metrics against committed thresholds (`thresholds_v1_2.json`). HALTs are "
        "prompts (accept as chance / amend / abort; CHARTER).", "",
        f"- **Deliverable run: {run}**. STOP-END is decided (CHARTER \"2026-10-24, STOP-END: user decisions\"): the "
        "user chose F19 as the deliverable, no clip veto, the bake (\"Supersede\") and no tag push yet.",
        "- **Baked 2026-10-24 (de81f98a) as the omni policy of record:** `crab_hex_omni_task/policy/manifest.yaml` "
        "current = `f19_8_p3b_payload180` (T*), student `f19_9_student` (kind distill: the ONNX deploy export + io "
        "json in the folder, the 122 MB `.pt` cited by sha256), stand net `f19_stand_net` (SH_T4F19 model_5000 + its "
        "warm-start ancestor SN_T4F19 model_4000); S1 s18, SH_T17, A-mask s17 and W are `superseded:` there (S1 s18 "
        "/ SH_T17 / A-mask s17 keep their files in policy/; W's left policy/ at the 2026-10-06 bake and live in git "
        "history). "
        "The play / eval / video commands of record are in `crab_hex_omni_task/policy/README.md`.",
        f"- **T*** = {h['tstar']} `{h['tstar_ckpt']}` (sha256 `{h['tstar_sha256']}`), max_stable "
        f"{doc['headline']['tstar_max_stable_kg']} kg ({SIM}).",
        f"- **Student** = {h['student']} `{sel.get('ckpt')}` (sha256 `{h['student_sha256']}`).",
        f"- **Stand nets** = {h['sn'][0]} (`{h['sn'][1]}`) / {h['sh'][0]} (`{h['sh'][1]}`).",
        f"- Every sha comes from `{L.rel(rd / 'deliverable.json')}`.",
        "- Records: [REPORT_baseline_vs_final.md](REPORT_baseline_vs_final.md) (+ `.json`), [CHARTER.md](CHARTER.md) "
        "(every HALT, amendment and disclosure), [PLAN.md](PLAN.md) (the approved plan; wins over DESIGN), "
        "[DESIGN.md](DESIGN.md) (annotated detail of record), `pipeline.yaml`, `pipeline.lock.json`, "
        f"`thresholds_v1_2.json` (sha `{str(doc['thresholds']['sha256'])[:8]}`).", "",
        "## 2. Reproduce at the pin", "",
        "```bash",
        f"git checkout {tag}      # from the repo root; the last task4-pin-* tag (local): the build R18 / F19 ran on, "
        "code_sha cbaf6797, eval_sha 8c8d22f0",
        "```", "",
        "- **The build of record vs the current pin.** Every task4-pin-* tag (2026-10-19 .. 2026-10-23) commits the "
        "lock R18 and F19 ran under (code_sha cbaf6797, eval_sha 8c8d22f0); check out a tag to reproduce them bit for "
        "bit. The 2026-10-25 PR prep merged origin/main (03ffa330: main's observations.py moved code_sha to 91b479cb), "
        "after which `check` reported `guard: code_sha drifted` against that pin -- intended, not a fault (CHARTER "
        "2026-10-25 PR prep). On 2026-10-26 the user accepted this milestone's version changes (\"accept the version "
        "changes and we will rerun with the new hashes as needed in future\"): the post-merge build plus "
        "documentation-only edits to pinned / hashed files (TP report.py, thresholds.py and payload_read.py; the "
        "stand campaigns' sh_core.py, sh_dagger.py, sh_table_test.py, sh_record_video.py / .sh and "
        "sn_record_video.sh; tools/run_phases.py, which is in eval_sha; scripts/rsl_rl/play_stream_video.py, which is "
        "in code_sha) are re-pinned by `pipeline.py lock` on the committed tree (a lock written on a dirty code / "
        "eval tree is not a valid pin). `pipeline.lock.json` is the current pin (its own code_sha / eval_sha / "
        "written_at); a stage re-run now runs that build and records its new shas.",
        f"- External stack (pipeline.lock.json): " + ", ".join(f"{k} {v}" for k, v in ext.items()) + ".",
        f"- Hardware: {', '.join(hw.get('gpus') or [])}; {hw.get('cpu')}; {hw.get('mem_gb')} GB; host {hw.get('host')}. "
        "One Isaac process at a time (detached queue + log-sentinel watcher).", "",
        "## 3. Pipeline commands (from the repo root)", "",
        "```bash",
        f"PY={PY}",
        f"TP={tp_rel}",
        f"SPEC={spec_rel}",
        f"$PY -I $TP/pipeline.py check --spec $SPEC --run {run}          # spec, lineage overlays, placeholders, pin guard",
        f"$PY -I $TP/pipeline.py gen --spec $SPEC --run {run} --check    # generated candidates / env files byte-stable",
        "$PY -I $TP/pipeline.py lock --spec $SPEC                         # (re)write pipeline.lock.json (a new pin)",
        f"$PY -I $TP/pipeline.py run --spec $SPEC --run {run}            # the official detached queue (resumable)",
        f"$PY -I $TP/pipeline.py status --spec $SPEC --run {run}",
        "# a HALT is a prompt -- exactly one of (each takes [--why TEXT], recorded in the HALT's resolution, and",
        "# [--no-relaunch]: by default a continue relaunches the detached queue):",
        f"$PY -I $TP/pipeline.py continue --spec $SPEC --run {run} --accept-chance STAGE",
        f"$PY -I $TP/pipeline.py continue --spec $SPEC --run {run} --amend STAGE --charter '<CHARTER entry>' [--retarget-tstar]",
        f"$PY -I $TP/pipeline.py continue --spec $SPEC --run {run} --retry STAGE",
        f"$PY -I $TP/pipeline.py continue --spec $SPEC --run {run} --abort",
        f"# CPU re-read of any gate (writes {L.rel(rd)}/gates/STAGE.reread.json):",
        f"$PY -I $TP/pipeline.py gate --spec $SPEC --run {run} --stage STAGE --thresholds {pl_rel}/thresholds_v1_2.json",
        "# this README (REPORT_baseline_vs_final.* are the step-11 records: write the REPORT to a scratch dir; pass",
        "# --rows-log, or the step-11 rows drop out of it; pipeline.py report forwards none of --other / --rows-log /",
        "# --readme and writes the default --out, i.e. over the step-11 records):",
        f"$PY -I $TP/report.py --spec $SPEC --run {run} --other R18 --rows-log {pl_rel}/step11/s11_queue_sentinels.txt "
        f"--out $(mktemp -d)/REPORT --readme {pl_rel}/README.md",
        "```", "",
        "A fresh replication needs a new `runs:` entry in `pipeline.yaml` with new seeds, then gen, lock, commit, check, run.", "",
        "**Run records and stopping.** A run's records live in `<PL>/<run>/`: `pipeline_state.json` (stage statuses, "
        "HALTs, accepted / amended stages, the pin), `pipeline_queue.log` (every launch verbatim + the `T4PIPE` "
        "sentinels), `pipeline_notify.jsonl`, `plan.json` / `candidates.yaml` / `envs/` (gen; checkout-independent: "
        "a path inside the checkout -- the training envs' `KRABBY_RSI_BANK` -- is written `$REPO/<repo-relative "
        "path>`, resolved by `tp_lib.read_env_file`; in a shell, from the repo root: `REPO=$PWD envsubst '$REPO' < "
        "FILE`), `gates/<stage>.json`, "
        "`halts/<stage>.<n>.json`, `stage_reads/`, `tp_logs/`; the gate-row stamps land under "
        "`parkour/outputs/omni_plan/task4/<run>/lr/NAME/SCENARIO/seed00<k>/<time>/` (gate_row.sh `TP_OUT`). To stop a "
        "running queue, `touch <PL>/<run>/STOP_PIPELINE`: the queue stops at its next stage boundary or before its next "
        "Isaac launch, launching nothing more (a running stage is stopped as the pipeline.py docstring's \"Stopping\" "
        "paragraph says). Nothing deletes the file -- not `continue`, not `--retarget-tstar` (which refuses while it "
        "is present unless `--no-relaunch`): remove it before `run`.", "",
        "**CPU reader tools** (`$TP/`, every one Kit-free; the queue calls them, and each runs by hand from the repo "
        "root; usage = each file's docstring):", "",
        "| tool | job | entry |", "|---|---|---|",
        f"| `thresholds.py` | every decisive gate line, its chance-HALT rate, the self-checks -> thresholds JSON | "
        f"`$PY -I $TP/thresholds.py --inputs {pl_rel}/threshold_inputs_v1_2.yaml --out OUT.json --self-check "
        "[--recommended FILE] [--sims N] [--seed S]` -- from the repo root this reproduces the committed "
        "`thresholds_v1_2.json` byte for byte |",
        "| `gate_read.py` | Gate 0 + a stage's decisive lines -> `gates/<stage>.json` (exit 0 PASS, 2 PASS + WATCH, 3 HALT, "
        "4 INVALID, 5 GUARD) | `--spec $SPEC --run RUN --stage S [--thresholds FILE] [--out FILE]`; loose stamps: "
        "`--spec $SPEC --lines-only --stage S --thresholds T.json --stamp ROW@ES=DIR ... [--trained CSV]` |",
        "| `payload_read.py` | level stability / max_stable / limiting factor; the report-only actuator load | "
        "`--spec $SPEC level --thresholds T.json --stamp ROW@ES=DIR ... [--trained CSV]`; "
        "`--spec $SPEC actuator --stamp DIR [--level KG]` |",
        "| `bridge_read.py` | is a new eval stamp the recorded one (the step-8 bridges) | "
        "`--ref STAMP --new STAMP [--mode full\\|aggregate] [--ignore KEY.PATH ...] [--no-default-ignore] "
        "[--allow-new-key K] [--json OUT]` |",
        "| `live_watch.py` | the pre-registered LIVE rules a running stage is read against | "
        "`check --run-dir DIR --kind driver\\|student [--stage S] [--source DIR] [--tstar DIR] [--first N]`; "
        "`replay ... [--every 50]` |",
        "| `replay_read.py` | R18's \"process reproduced\" reads and the X stand-replay rule | "
        "`pconfig\\|pearly\\|onset --new RUN_DIR --old RUN_DIR`; `x --new RUN_DIR --ref-sha SHA --rule "
        "bitwise\\|report_only [--ref-log F] [--new-log F]` |",
        "| `select_student.py` | the student candidates (T*'s last iteration + 1k .. 5k, nearest saved checkpoint); the "
        "prescreen / full-block reads run inside the queue's sel stage | `candidates --run-dir DIR --start ITER "
        "[--offsets 1000,...]` |",
        "| `tp_lib.py` | the shared helpers (paths, loaders, hashing, records, the Isaac-busy check) | no CLI |", "",
        "## 4. What the queue launched (verbatim, from the run's tp_logs)", "",
        "- Driver stages s1 .. p3c: `experiments/tools/launch_candidates.sh --campaign-dir <PL>/" + run +
        " --reference-chain " + run + "_<stage> --seed <seed>` (TP/pipeline.py `launch_driver`; run_candidates.py on the "
        f"generated `{L.rel(rd / 'candidates.yaml')}`).",
        f"- Student (the last launch, the deliverable's distill): `{st_cmd}`" if st_cmd else "- Student: (no st.launcher.log)",
        (f"  ({len(st_all) - 1} earlier launch(es) superseded: `{st_all[0]}`)" if len(st_all) > 1 else ""),
        f"- SN: `{sn_cmd}`" if sn_cmd else "", f"- SH: `{sh_cmd}`" if sh_cmd else "",
        f"- REFS(T*) rows: `{refs_cmd}`" if refs_cmd else "",
        f"- A gate row (T*'s first): `{row_cmd}`" if row_cmd else "",
        f"- Every gate row (from the repo root; EVERY path absolute -- gate_row.sh refuses a relative one, rc 2; "
        "`$PWD` = the repo root): "
        f"`TP_OUT={R}/parkour/outputs/omni_plan/task4/<run> TP_ROW_TIMEOUT=5400|10800 "
        f"[TP_LEVEL_FILE={R}/{pl_rel}/<run>/envs/levels/LNNN.env] [TP_STRIP_KEYS=...] bash $TP/gate_row.sh NAME "
        f"/abs/CKPT {R}/{pl_rel}/<run>/envs/<stage>[.<family>].env /abs/MANIFEST SCENARIO ES [harness args]` (e.g. "
        f"MANIFEST = `{R}/{L.rel(L.OMNI)}/experiments/eval/scenarios_omni.yaml`); the step-11 report rows use the same "
        f"form with `TP_OUT={R}/parkour/outputs/omni_plan/task4/step11`.", "",
        "## 5. What passing each gate means (4b)", "",
        "Thresholds v1.2; HALTs are prompts; the chance-HALT run sum "
        f"({fmt((doc['provenance'].get('run_sum') or {}).get('recommended'))}) is reported, not capped.", "",
        "- s1: a side-first walker formed (floors: survival >= 0.80, ratio >= 0.50, side22 not SLOW).",
        "- c4 .. c6: widening the command set cost no translation (the same floors).",
        "- c6x: the clock-free walker tracks and turns (floors; clock hidden from here on).",
        "- p2a: head-of-record capability (survival >= 0.90 / ratio >= 0.70 floors, side22 FAST/FAST, field_hard "
        "pooled es1-4 >= its three-seed line).",
        "- p3a: stable at 90 kg (the a-priori p3a.stable90 lines on suite + field_light) and no unloaded regression "
        "(the paired 0-kg check vs the run's own p2a: v1.2 track_err for the translate categories, a one-sided ratio "
        "drop for move_turn).",
        f"- p3b / p3c: measured ceilings (budget_bound at N_p {spec.n_p()}); a miss ends the stage, never the run.",
        "- T*: the trained payload head with the highest max_stable that passes its paired 0-kg check.",
        "- Student: the floors, plus paired completion >= T* - max(0.05, 2 SE_diff, split 0.0) on the WORLD rows "
        "(field_hard / field_light / ring_obst / field_turn on Student-Play). The body / grid / sweep reads in the "
        "report are post hoc. Camera use is UNVERIFIED (CHARTER 2026-10-17).",
        "- SN: G1-G5 (G1 / G2 stillness WATCH since v1.1).",
        "- SH: K1-K6, K4 <= 3/800 misses per env seed (D9); a lazy R_h control beside T* re-anchors a line it fails "
        "too (walker-caused, DESIGN 3.6).", "",
        "## 6. Play and evaluate the deliverable heads (from parkour/)", "",
        "REQUIRED carry env (Gate 0; the deliverable heads do not walk without it): "
        + ", ".join(f"`{k}={v}`" for k, v in req.items()) + ". The full composed env is "
        f"`{L.rel(envf)}` (use `{L.rel(rd / 'envs' / (str(ts.get('stage')) + '.field_hard.env'))}` for the wall field).", "",
        "```bash",
        "cd parkour",
        f"ENVF=../{L.rel(envf)}",
        f"CK_TSTAR={_from_parkour(ts.get('ckpt'))}",
        f"CK_STUDENT={_from_parkour(sel.get('ckpt'))}",
        "U=$(env | grep -oE '^KRABBY_[A-Z0-9_]+' | sed 's/^/-u /')   # drop every inherited KRABBY_* first",
        "# T* (teacher, Walk-Play):",
        f"env $U OMNI_KIT_ACCEPT_EULA=yes $(grep -E '^KRABBY_' $ENVF) {PY} scripts/rsl_rl/play.py "
        "--task Isaac-Crab-Hex-Omni-Walk-Play-v0 --num_envs 1 --real-time --checkpoint $CK_TSTAR",
        "# the student (Student-Play, 4 depth cameras); the same env minus omni_student_spec.STUDENT_RUNNER_KNOBS_REFUSED"
        + (f" (none of them is in {ENV_NAME(envf)})" if not in_env_refused else f" (strip: {' '.join(in_env_refused)})") + ":",
        f"env $U OMNI_KIT_ACCEPT_EULA=yes $(grep -E '^KRABBY_' $ENVF) {PY} scripts/rsl_rl/play.py "
        "--task Isaac-Crab-Hex-Omni-Student-Play-v0 --enable_cameras --num_envs 1 --checkpoint $CK_STUDENT",
        "```", "",
        "- play.py writes an export beside the checkpoint it plays (the student: its JIT / ONNX `exported_deploy/`; "
        "T*: `exported_teacher/`) -- a write into the run dir under logs/.",
        "- The harness play (the gate rows' eval) is the gate_row.sh line above (from the repo root, absolute paths, "
        f"`TP_OUT` set) with `{R}/{L.rel(L.OMNI)}/experiments/eval/scenarios_omni.yaml suite__omni ES "
        "--trained-categories "
        "stand,fwd,side,back,diag,turn,move_turn`; "
        "for the student add `--task Isaac-Crab-Hex-Omni-Student-Play-v0 --policy-role student` and "
        "`TP_STRIP_KEYS=\"<STUDENT_RUNNER_KNOBS_REFUSED>\"`.",
        "- Video: ONLY `scripts/rsl_rl/play_stream_video.py ... --headless --video --video_length N --enable_cameras` "
        "under `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0` (the clip command). Never "
        "`play.py --video` / gym RecordVideo (a 5-min clip buffers ~41 GB).",
        "- Stand nets: no interactive player plays the routed walker + stand pair (play.py and any JIT / ONNX export "
        "carry ONE network; policy/README.md hazard). The VIDEO recorder "
        f"`{L.rel(L.OMNI / spec.d['stand']['sh_scripts'] / 'sh_record_video.sh')}` (VIDEO ONLY) takes the walker as "
        "`-- --checkpoint <T*>` (its default is the historical S1 s18), the stand net as `--stand-ckpt <SH "
        "model_5000.pt> --stand-mask 1 --deadband 0.25` and the reference stamp as `--eval-run <the head's "
        "hraise__omni es1 stamp>`; it refuses a stand net trained beside another walker and any checkpoint / mask / "
        "deadband / code sha that differs from the stamp's. The F19 videos `sh_mix_F19_s1514_rough` (`--plan mix "
        "--seed 1514 --terrain field_medium`) and `sh_mix_F19_s1514_roughest` (`--terrain field_max`) were recorded "
        "so -- T* + SH_T4F19; the commands are in policy/README.md (\"Stand network of record\"). Since the 2026-10-26 "
        "hash acceptance moved sh_core.py's sha, every stand stamp recorded before it (SH_T4F19's included) is "
        "refused until the head's hraise__omni@1 row is re-run with the current code (sh_eval_rows.sh, walker env "
        "exported) or `--allow-reference-mismatch` is passed.",
        "- Gate-evaluate the stand nets with the spec's stand row scripts (sn_eval_rows.sh / sh_eval_rows.sh, "
        "run_arm.sh --row-eval) and the SN_WALKER_* env the queue composes (pipeline.walker_spec_env).", "",
        "## 7. Selection record", "",
        f"`{L.rel(rd / 'gates/sel.json')}` + `{L.rel(rd / 'stage_reads/sel.rows.json')}`: the prescreen ranked "
        f"{(sel.get('pairing') or {}).get('ranked_iters')}; {h['student']} was the first full-block PASS "
        f"(WATCH flags {(sel.get('pairing') or {}).get('watch_flags')}).",
        f"- Veto clip (never a gate): `{ce.get('clip_dir')}/` {sorted((ce.get('clip') or {}).keys())}.",
        md_clip_local(ce.get("clip_paths") or [], ce.get("gitignored") or {}),
        (md_clip_local(((doc.get("appendix") or {}).get("clip") or {}).get("paths") or [],
                       ((doc.get("appendix") or {}).get("clip") or {}).get("gitignored") or {})
         .replace("- The veto clip", f"- {doc.get('other')}'s veto clip", 1) if doc.get("appendix") else None),
        f"- The student checkpoint itself is recorded by sha256 `{ks.get('student_sha256')}` ({ks.get('student_bytes')} B), "
        "not copied into head/: it exceeds the 64 MiB bundle guard. head/ holds its ONNX deploy export instead "
        "(CHARTER 2026-10-24 decision 1).", "",
        "## 8. HALTs and disclosures (one line each; the CHARTER entry is the record)", ""]
    for runx in (run, doc.get("other")):
        r = (doc["provenance"].get(runx) or {}) if runx else {}
        for hh in r.get("halts") or []:
            lines.append(f"- {runx} {hh['stage']} ({hh['class']}): {hh['resolution']}; chance rate: {halt_rate_text(hh)}.")
    for d in doc["disclosures"]:
        lines.append(f"- {d['title']} ({d['charter']}).")
    lines += ["- The measured wall-clock / GPU-h table: per stage with parks separated, in REPORT_baseline_vs_final.md "
              "(Provenance).", "",
              "## 9. Keep-set and what stays the user's", "",
              f"`bundle.yaml` -> `head/` via `{L.rel(L.TOOLS / 'bundle_experiment.py')}` (--dry-run / --init / default "
              "/ --check; no --index): head = T* model, extra heads = SN / SH + the student's ONNX deploy export; the "
              "lean eval keep-set (summary.md, complete.json, tp_row.json, code_state.json); stage heads by sha256 "
              f"in `{L.rel(rd / 'deliverable.json')}`. Local annotated tag `{BASELINE_TAG}` (CHARTER 2026-10-24 "
              "decision 4; " + ("present" if any(t.split()[0] == BASELINE_TAG for t in
                                                 (doc["provenance"].get("stack") or {}).get("tags") or [])
                                else "not created yet at generation time") + "), never pushed.", "",
              "The user's decisions (STOP-END, 2026-10-24; CHARTER \"2026-10-24, STOP-END: user decisions\"): the "
              "deliverable is F19 (st.1 was an amend and sh.1 a re-anchor, not chance acceptances), no clip veto, and "
              "the bake into `crab_hex_omni_task/policy/` -- DONE in de81f98a (tools/bundle_policy.py gained `kind: "
              "distill`; tests/unit/test_policy_of_record_omni.py was rewritten; the record is policy/manifest.yaml). "
              "Still the user's: the tag push (not now; the tag stays local) and any push of the branch. 2026-10-26: "
              "the user accepted the milestone's hash changes (section 2).", "",
              "## 10. Folder map (`<PL>` = this campaign directory)", "",
              "- `pipeline.yaml` (the spec: runs, stages, rows, thresholds version) and `pipeline.lock.json` (the pin: "
              "every out-of-hash file a run depends on by sha256, code / eval / tools shas, the external stack).",
              "- `threshold_inputs*.yaml` -> `thresholds_v*.json` (`$TP/thresholds.py`; `thresholds_v1_2.json` is in "
              "force, regenerated byte for byte by the section-3 command).",
              "- `cameras.yaml`: the four depth-camera mounts of the student (v0, STOP-0 placeholder poses; read by "
              "`config/crab_hex/omni_student_spec.py`).",
              "- `R18/`, `F19/`: the two official runs' records (section 3); `dry/`: the full-graph dry pass's "
              "generated plan / candidates / envs; `preflight/`: the PLAN step-2 / step-8 preflight (Phase A / C "
              "queues, reads, STEPC_RULES.md); `m1a/`: the M1a payload-knob config-dump check record.",
              "- `rows/scenarios_pl_body.yaml`: the pipeline's own body-frame direction-sweep row (report only); "
              "`evals/`: the lean eval keep-set (F19_driver, F19_rows, step11: summary.md, complete.json, tp_row.json, "
              "code_state.json); `step11/`: the step-11 report rows (queue sentinels, envs, fill plan, NOTE.md).",
              "- `head/`: the bundled keep-set (bundle.yaml, section 9); `scripts/t4_*`: campaign-local Kit / CPU "
              "helpers of the build and preflight steps -- `t4_dump_cfgs.py` (config dumps, the M1a inertness proof) + "
              "`t4_payload_dump_check.py`, `t4_stepA_*` (Phase A queue + read), `t4_make_preflight.py` / "
              "`t4_make_stepC.py` (their static inputs), `t4_stepC_*` (the Phase C job table `t4_stepC_jobs.py`, its "
              "queue, reader, dry pass, camera / readback plays, depth diagnostic), `t4_eval_row.sh` (one eval row).",
              "- Records: CHARTER.md, PLAN.md, DESIGN.md, STOP0.md, REPORT_baseline_vs_final.md / .json, this README.",
              ""]
    if bad:
        lines += ["## UNVERIFIED (generator check)", ""] + [f"- `{c['path']}`: exists {c['exists']}, missing flags "
                                                             f"{c['missing_flags']}" for c in bad] + [""]
    return "\n".join(x for x in lines if x is not None) + "\n"


def ENV_NAME(p) -> str:  # noqa: N802
    return Path(p).name


# ============================================================================== CLI
def build(spec_path: Path, run: str, other: str | None = None, rows_log_path=None) -> dict:
    return Report(spec_path, run, other, rows_log_path).build()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--other", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows-log", type=Path, default=None, help="the step-11 queue sentinel log (S11 ROW / DONE / SKIP)")
    ap.add_argument("--readme", type=Path, default=None, help="also write the campaign README here")
    a = ap.parse_args(argv)
    doc = build(a.spec, a.run, a.other, a.rows_log)
    out = a.out or (a.spec.resolve().parent / "REPORT_baseline_vs_final")
    L.write_json_atomic(out.with_suffix(".json"), doc)
    L.write_text_atomic(out.with_suffix(".md"), md(doc))
    print(f"wrote {out.with_suffix('.md')} / .json -- cells {doc['counts']}")
    if a.readme:
        spec = _P().Spec(a.spec)
        L.write_text_atomic(a.readme, readme_md(doc, spec, a.run))
        print(f"wrote {a.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
