#!/usr/bin/env python3
"""Omni phase 2, stage p2s -- the ONE stand-estimator set (P2S_PLAN.md step 1; CPU only, read-only).

Every head and every stand hold run through this module's estimators: the suite's 12-s stand wave,
the field rows' opening 8-s stand (``field_hard__omni`` / ``field_light__omni``), ``field_turn__omni``'s
stand after the walk-out, and (once queued) the ``walkstop__omni`` stands. One definition per metric, used
everywhere, so a verdict never depends on which hold produced the number.

INPUTS. ``run_meta.json`` (``hold_labels``, ``hold_speeds``, ``dt``, and -- suite rows only --
``env_index_map``) plus ``raw/episode_<i>.npz``, per step: ``foot_force_norm`` (T,6) N (= the recorded
``net_forces_w_history`` norm, ``eval_crab_hex_gait.py:1049``), ``foot_pos_w`` / ``foot_lin_vel_w`` (T,6,3),
``terrain_z`` (T,6), ``root_pos_w`` (T,3), ``root_quat_w`` (T,4) wxyz, ``root_lin_vel_b`` (T,3),
``cmd_applied`` (T,3), ``segment_id`` (T,) int indexing ``hold_labels``, ``steady_mask`` (T,) bool (the
eval's own settle flag: the first 1.0 s of a hold, ``gait_eval.schedule``), ``heading_err`` (T,) and
``crab_failure`` (T,) bool. ``dt`` is 0.02 s throughout.

HOLD IDENTIFICATION. A "stand" hold is ``hold_labels[i] == "stand"`` (never a bare ``hold_speeds[i] == 0``
test: ``turn_l90`` / ``turn_r90`` also command zero translation but are not a stand -- risk 4 of
P2S_PLAN.md). On a ``category_suite`` row (``suite__omni``) every env in the "stand" category wave shares
one ``hold_labels`` vocabulary with every other category (``run_meta.env_index_map``, ``label == "stand"``
picks the episodes; ``hold_labels.index("stand")`` is then the stand hold's ``segment_id``, 1 on this plant).
On a schedule row (``field_hard__omni`` / ``field_light__omni`` / ``field_turn__omni`` / the future
``walkstop__omni``) every episode shares the one schedule, so ``hold_labels.index("stand")`` resolves it
directly; ``field_hard`` / ``field_light`` carry it as hold 0 (the opening stand), ``field_turn`` as hold 1
(after the 10-s walk-out). A "moving hold that directly follows a stand" (the onset read) is the next hold
in the schedule whose own ``hold_speeds`` entry is > 0 -- a stand that leads into a turn (``field_turn``) has
no such successor and is skipped by :func:`stand_onset_pairs`.

WINDOWS. ``full`` = every row of the hold (``segment_id == hold_idx``), trimmed at the tail exactly as the
harness trims ``rigidity_metrics`` / ``motion_quality_metrics``'s own-hold window
(``gait_eval/report.py:174-182``): when the EPISODE ends in a fall the last ``fall_exclusion_s`` (0.5 s,
``scenarios_omni.yaml``) of logged steps are dropped, wherever they land -- "the last half-second before a
fall is a fall, not a gait sample." A saved npz carries no ``termination_reason`` string, so the trim fires
off ``crab_failure[-1]`` (the terminal step of a fall episode is the one step ``crab_failure`` is still true
at liftoff of scoring; this is the same signal ``p2_mech.classify_fall`` keys its own ``i_fail`` off). ``steady``
= ``full`` AND ``steady_mask``. Every estimator below says which window it uses; :func:`steps_per_s` is
checked to equal ``metrics/episode_XX.json``'s ``holds.stand.rigidity.steps_per_s`` on the ``full`` window
(the harness disables ``steady`` for rigidity, ``gait_eval/report.py:175-182``) -- see ``--selfcheck``.

ESTIMATORS (per episode; "loaded" = ``foot_force_norm > MOTION_TOUCH_REF_N`` = 50 N, the eval's half-load
contact bit, NOT the tripod score's lighter ``contact_force_threshold``):
  steps/s, taps/s   :func:`gait_eval.metrics.step_tap_runs` on the ``full`` window's loaded mask, summed
                     over the six feet, / the window's own duration (``rigidity_metrics``'s own
                     ``steps_per_s`` convention, reproduced exactly).
  stance offset      the median of (foot_z - terrain_z) over LOADED samples pooled across the six feet, over
                     the ``full`` window -- ``gait_eval.metrics.swing_clearance_metrics``'s
                     ``stance_z_offset_m`` convention, and what ``rigidity_metrics`` itself calls for
                     ``swing_apex``. Every height clause below is relative to THIS, never a per-foot
                     self-reference (a foot that never lands has no self-reference at all).
  clearance           per foot, p95 - p5 of (foot_z - terrain_z) over the ``steady`` window; median over
                     feet (``p2_read.stand_clearance``'s statistic, reproduced exactly -- REF reads 0.135)
                     and, report-only, the max over feet.
  off-ground share    fraction of (foot, steady row) samples with (foot_z - terrain_z) - offset > 0.03 m.
  hover               a foot is hovering if it is BOTH unloaded (< 50 N) AND raised (> 0.03 m above the
                     offset) for > 95 % of the ``full`` hold; the episode is a hover episode if any foot
                     hovers. Force alone flags the still statue (one foot carries 0 N while grounded).
  KS, loaded count    mean over ``steady`` rows of the 3-D foot-speed sum (KS) / the loaded-foot count.
  body height         mean over ``steady`` rows of root z minus the mean ``terrain_z`` under the six feet.
  drift               mean over ``steady`` rows of the planar (x, y) ``root_lin_vel_b`` norm.
  facing error rms    sqrt(mean(heading_err ** 2)) over ``steady`` rows.
  T1/s, T2/s           numpy report-only versions of the lever's two reward kernels (P2S_PLAN.md "the
                     lever"): T1 = mean over ``steady`` rows of (sum_f clamp(F_f / 50, 0, 1) - 3); T2 = mean
                     of sum_f clamp(F_f / 50, 0, 1) * |v_f| (3-D). The replay gate's torch kernels are the
                     landed ones; these are a numpy sanity cross-check only.
  falls               ``crab_failure`` anywhere in the UNTRIMMED hold window (segment_id == hold_idx; the
                     trim above answers "was this step a gait sample", not "did a fall happen here").

Per-head summary (:func:`head_summary`) = the MEDIAN of each estimator over the hold's episodes, plus the
hover-episode count and ``n``. :func:`still_clauses` reads P2S_PLAN.md section 4's STILL rule off that
summary.

ONSET LATENCY (:func:`episode_onset_latency`, schedule rows only): for a moving hold directly following a
stand, time from the moving hold's first row until the 5-step trailing mean of ``root_lin_vel_b``'s planar
part, projected onto the hold's own commanded direction (``cmd_applied[0, :2]`` -- world rows write a true
body-frame command, the same frame ``root_lin_vel_b`` is already in), first reaches 0.5 x the commanded
speed and stays at or above it for 0.5 s running. ``None`` (censored) if it never does. The head-level
statistic is the MEDIAN over episodes (P2S_PLAN.md "left for implementation").

Z2 PER-STAND ESTIMATORS (NOTE(p2r Z2, 2026-10-10): P2R_Z2_PLAN.md step 1 "Reads"; every estimator above is
unchanged). :func:`iter_stand_holds` enumerates EVERY hold labelled "stand" of every kept episode -- walkstop
carries three per episode, and :func:`stand_hold_index` (the first one only) stays as it is for the estimators
above. Per stand (:func:`stand_z2_metrics`), on the fall-trimmed ``full`` rows of that hold:
  cams              ``joint_pos`` columns whose ``run_meta.joint_names`` end in ``_Body_CamShaft_RevoluteJoint``,
                     ``wrap_to_pi``'d (home is cam = 0; the spawn pose reads exactly 0).
  homed              max over legs AND rows of |cam| over the stand's LAST 5 s <= 0.10 rad.
  six-down share     fraction of window rows with all six ``foot_force_norm`` > 50 N; window = the ``steady``
                     window (:func:`hold_masks`) on a category_suite stamp, the last 5 s of the fall-trimmed stand
                     on every schedule row (walkstop, field rows).
  settle time        seconds from the stand's first row to the first row from which max|cam| <= 0.10 (HOMED)
                     holds for 0.5 s running (25 rows); None = never inside the stand (unsettled). NOTE(p2r Z2
                     cams-only, 2026-10-10): six-down is no longer part of settle (P2R_Z2_PLAN "Amendment
                     2026-10-10": the six-down share is REPORTED, not gated).
  post-settle        steps/s and taps/s (:func:`gait_eval.metrics.step_tap_runs` on the loaded mask) and KS (mean
                     3-D foot-speed sum) from the settle row to the stand's end; None on an unsettled stand.
  re-stance lifts    steps (:func:`gait_eval.metrics.step_tap_runs`) from the stand's first row to the settle row
                     (the whole stand when unsettled).
  hip yaw            report-only: mean / max over legs and the last 5 s of |``*_Body_Hip_RevoluteJoint``| (deg).
  body height        the estimator above (mean over ``steady`` rows of root z minus the mean terrain z).
  falls              ``crab_failure`` anywhere in the untrimmed hold (:func:`fell_in_hold`).
:func:`z2_head_summary` reads every reached stand (rows in the hold, or a fall in it): shares of stands with
six-down share >= 0.90 / homed / settled, the median settle time (settled stands), FAIL-CLOSED medians of the
post-settle steps / taps / KS over ALL reached stands (an unsettled stand ranks worst; a median that lands on one
reads None, which every clause fails), stand falls per 300 stands, the body-height and hip-yaw medians and ``n``.

CLI:
    p2s_stand.py --stamp DIR [--hold stand] [--json OUT]     one stamp, one hold label (+ the "z2" per-stand
                                                              summary when the stamp carries joint_pos)
    p2s_stand.py --baseline [--json OUT]                     the P2S_PLAN.md step-1 baseline table (markdown
                                                              to stdout; the full read tree to --json)
    p2s_stand.py --selfcheck                                 exit 0 iff the E6 statue reads STILL on every
                                                              clause AND TG reads >= 9 hover episodes
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path(__file__).resolve().parents[7]  # this checkout's root (scripts -> campaign -> experiments -> ... -> repo)
TASKS = R / "parkour/parkour_tasks/parkour_tasks"
GAIT_SCRIPTS = TASKS / "crab_hex_forward_task/scripts"
if str(GAIT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GAIT_SCRIPTS))
from gait_eval import metrics as M  # noqa: E402  (pure numpy; see module docstring)

# ================================================================================================ paths (mirrors
# p2_read.py / p2_mech.py's convention exactly, so a head name resolves the same way in every tool)
OUT = Path(os.environ.get("P2_OUT") or R / "parkour/outputs/omni_plan/phase2")
LROUT = OUT / "lr"
GE_PHASES = R / "parkour/logs/rsl_rl/gait_eval/omni_phases"
SUITE, FH, FL, FT = "suite__omni", "field_hard__omni", "field_light__omni", "field_turn__omni"

# ================================================================================================ thresholds
TOUCH_REF_N = M.MOTION_TOUCH_REF_N          #: 50 N, the eval's "loaded" line (shared with gait_eval)
MIN_STEP_STEPS = M.RIGIDITY_STEP_MIN_STEPS  #: a step is >= this many unloaded steps; shorter is a tap
OFFGROUND_TOL_M = 0.03                      #: a foot sample this far above the stance offset is "off-ground"
HOVER_FRAC = 0.95                           #: hover needs unloaded AND raised for this share of the full hold
MIN_EP_ROWS = 25                            #: p2_read.stand_clearance's own floor; too short a window to read
FALL_EXCLUSION_S = 0.5                      #: scenarios_omni.yaml's fall_exclusion_s (gait_eval/schedule.py default)
ONSET_SPEED_FRAC = 0.5                      #: onset = reaching this share of the commanded speed
ONSET_HOLD_S = 0.5                          #: ... and staying there for this long
ONSET_TRAIL_STEPS = 5                       #: the trailing-mean window the onset read is defined on

#: P2S_PLAN.md section 4's STILL clauses, against a head's :func:`head_summary` (median over episodes, plus
#: hover_episodes_per_100). Every clause is "<=" -- smaller is stiller.
STILL_THRESHOLDS = {
    "steps_per_s": 0.5,
    "clearance_m": 0.05,
    "off_ground_share": 0.10,
    "hover_episodes_per_100": 5.0,
    "taps_per_s": 1.0,
    "ks_m_s": 0.5,
}

#: NOTE(p2r Z2, 2026-10-10): the Z2 per-stand estimators' constants (P2R_Z2_PLAN.md step 1 "Reads" / step 5).
Z2_LAST_S = 5.0                              #: homed / hip yaw / schedule-row six-down window: the stand's last 5 s
Z2_HOMED_RAD = 0.10                          #: homed: every cam within 0.10 rad of home (about 1.7 deg of hip yaw)
Z2_SIXDOWN_SHARE = 0.90                      #: a stand counts as six-down at a window share >= this
Z2_SETTLE_HOLD_S = 0.5                       #: settle: homed held this long running (cams-only, 2026-10-10)
Z2_FALLS_PER = 300                           #: stand falls are reported per this many stands (walkstop: 100 x 3)
CAM_JOINT_SUFFIX = "_Body_CamShaft_RevoluteJoint"
HIP_YAW_JOINT_SUFFIX = "_Body_Hip_RevoluteJoint"
WINDOW_STEADY, WINDOW_LAST = "steady", "last5"


# ================================================================================================ stamp resolution
def stamp_complete(p) -> bool:
    """A stamp is complete when the harness wrote complete.json (category_suite rows) or run_arm.sh wrote
    p2_complete.json after a clean exit of a non-suite (8-hold schedule) row. Mirrors p2_read.py / p2_mech.py."""
    p = Path(p)
    return (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()


def newest_stamp(name: str, row: str, seed: int = 1, lrout: Path = LROUT) -> str | None:
    """The latest complete ``<lrout>/<name>/<row>/seed<seed:03d>/*/`` stamp, or None. Mirrors p2_read.py."""
    d = lrout / name / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if stamp_complete(p)]
    return str(hits[-1]) if hits else None


def newest_under(glob_dir: Path, row: str, seed: int = 1) -> str | None:
    """Newest complete ``<glob_dir>/<row>/seed<seed:03d>/*/`` stamp directly under one run's own gait-eval
    output (the driver's own suite stamp, never copied into ``lr/``: P2S_PLAN.md's C0/CL/TG/A1/F1 suite
    baselines live under ``GE_PHASES``, not ``LROUT``)."""
    d = glob_dir / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if stamp_complete(p)]
    return str(hits[-1]) if hits else None


def newest_chain_dir(parent: Path, chain_glob: str) -> Path | None:
    """The newest directory matching ``chain_glob`` under ``parent`` (a queue chain's run-id suffix is a
    launch timestamp, so "newest" is "last launched", matching p2_read.py's own stamp convention)."""
    hits = sorted(parent.glob(chain_glob))
    return hits[-1] if hits else None


def _json(p) -> dict | None:
    try:
        return json.load(open(p))
    except (OSError, ValueError, TypeError):
        return None


# ================================================================================================ hold identification
def stand_hold_index(hold_labels: list[str]) -> int | None:
    """The stand hold's ``segment_id`` -- ``hold_labels.index("stand")``, never a ``hold_speeds == 0`` test
    (``turn_l90`` / ``turn_r90`` also command zero translation; see the module docstring's risk-4 note). Every
    row this module reads (``suite__omni``, ``field_hard__omni``, ``field_light__omni``, ``field_turn__omni``)
    carries exactly one hold labelled "stand", so a single index always answers it."""
    try:
        return hold_labels.index("stand")
    except ValueError:
        return None


def stand_onset_pairs(hold_labels: list[str], hold_speeds: list[float]) -> list[tuple[int, int, str]]:
    """Every ``(stand_idx, next_idx, next_label)`` where hold ``stand_idx`` is "stand" and the very next hold
    in the schedule commands a nonzero speed (a moving hold "directly follows" the stand). A stand that leads
    into a turn (speed 0, e.g. ``field_turn``'s stand -> ``turn_l90``) has no entry -- there is no speed to
    reach onset against."""
    pairs = []
    for i, label in enumerate(hold_labels):
        if label != "stand":
            continue
        j = i + 1
        if j < len(hold_labels) and float(hold_speeds[j]) > 0.0:
            pairs.append((i, j, hold_labels[j]))
    return pairs


# ================================================================================================ episode iteration
def _raw_episodes(stamp: str, keep):
    """Every episode's ``(meta, labels, env_entry, npz)`` whose ``env_index_map`` entry ``keep()`` selects.
    Mirrors ``p2_read._raw_episodes``: a non-suite row has no ``env_index_map`` (one schedule, every env), so
    the fallback ``label: None`` entries let ``keep`` filter on other fields (here, nothing -- every episode
    of a schedule row carries the same stand hold)."""
    meta = _json(Path(stamp) / "run_meta.json") or {}
    files = {int(f.stem.split("_")[1]): f for f in (Path(stamp) / "raw").glob("episode_*.npz")}
    labels = meta.get("hold_labels") or []
    for e in meta.get("env_index_map") or [{"env_index": i, "label": None} for i in sorted(files)]:
        if keep(e) and e["env_index"] in files:
            yield meta, labels, e, np.load(files[e["env_index"]])


def iter_stand_episodes(stamp: str):
    """Every "stand"-category episode of ``stamp`` (suite: the env_index_map's "stand" wave; schedule rows:
    every episode, since there is only one schedule) as ``(labels, hold_idx, npz)``."""
    for meta, labels, e, d in _raw_episodes(stamp, lambda e: e.get("label") in (None, "stand")):
        hold_idx = stand_hold_index(labels)
        if hold_idx is None:
            continue
        yield labels, hold_idx, d


# ================================================================================================ windows
def ended_in_fall(d) -> bool:
    """Episode-level fall flag reconstructed from the npz alone (no stored ``termination_reason`` string):
    the terminal logged step is still ``crab_failure`` -- the same signal ``p2_mech.classify_fall`` keys its
    own ``i_fail`` off. A schedule-completed or time-limited episode's last step is never a failure."""
    cf = np.asarray(d["crab_failure"])
    return bool(cf[-1]) if cf.size else False


def fall_trim_upper(d, dt: float, fall_exclusion_s: float = FALL_EXCLUSION_S) -> int:
    """The index one past the last row this episode may contribute to any own-hold window -- the harness's
    own tail trim (``gait_eval/report.py:_window_mask``, ``ended_in_fall`` branch): drop the last
    ``fall_exclusion_s`` of logged steps when the episode ends in a fall, wherever in the schedule they land."""
    n = int(np.asarray(d["crab_failure"]).shape[0])
    if not ended_in_fall(d):
        return n
    return max(0, n - int(round(fall_exclusion_s / dt)))


def hold_masks(d, hold_idx: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``(full, steady)`` boolean masks of one hold: ``full`` = ``segment_id == hold_idx`` within the
    fall-trimmed tail (:func:`fall_trim_upper`); ``steady`` = ``full`` AND the eval's own ``steady_mask``."""
    seg = np.asarray(d["segment_id"])
    upper = fall_trim_upper(d, dt)
    in_bounds = np.arange(seg.shape[0]) < upper
    full = (seg == hold_idx) & in_bounds
    steady = full & np.asarray(d["steady_mask"])
    return full, steady


def fell_in_hold(d, hold_idx: int) -> bool:
    """Whether ``crab_failure`` ever fires inside this hold -- the UNTRIMMED segment mask, since the trim
    above answers "is this row a gait sample", not "did a fall happen here"."""
    seg = np.asarray(d["segment_id"])
    return bool(np.asarray(d["crab_failure"])[seg == hold_idx].any())


# ================================================================================================ per-episode metrics
def stance_offset(d, full_mask: np.ndarray) -> float | None:
    """Median of (foot_z - terrain_z) over LOADED samples pooled across the six feet, over ``full_mask`` --
    ``gait_eval.metrics.swing_clearance_metrics``'s ``stance_z_offset_m`` convention, computed on this hold's
    own window (never a per-foot self-reference: a foot that never lands has none to offer)."""
    if not full_mask.any():
        return None
    force = np.asarray(d["foot_force_norm"])[full_mask]
    height = np.asarray(d["foot_pos_w"])[full_mask, :, 2] - np.asarray(d["terrain_z"])[full_mask]
    loaded = force > TOUCH_REF_N
    vals = height[loaded]
    return float(np.median(vals)) if vals.size else None


def clearance_stats(d, steady_mask: np.ndarray) -> tuple[float | None, float | None]:
    """``(median_over_feet, max_over_feet)`` of per-foot p95 - p5 of (foot_z - terrain_z) over
    ``steady_mask`` -- ``p2_read.stand_clearance``'s per-episode statistic (reproduced exactly; the median is
    what the 5-cm STILL target and the 0.135 m REF baseline are measured on, the max report-only)."""
    if int(steady_mask.sum()) < MIN_EP_ROWS:
        return None, None
    z = np.asarray(d["foot_pos_w"])[steady_mask, :, 2] - np.asarray(d["terrain_z"])[steady_mask]
    spread = np.percentile(z, 95, axis=0) - np.percentile(z, 5, axis=0)
    return float(np.median(spread)), float(np.max(spread))


def hover_feet(d, full_mask: np.ndarray, offset: float | None) -> np.ndarray:
    """Per-foot bool (6,): unloaded (< 50 N) AND raised (> 3 cm above ``offset``) for > 95 % of ``full_mask``.
    Force alone flags the still statue in most episodes (one grounded foot can carry 0 N); clearance alone
    cannot see a foot that never lands at all -- both conditions are needed together."""
    if offset is None or not full_mask.any():
        return np.zeros(6, dtype=bool)
    force = np.asarray(d["foot_force_norm"])[full_mask]
    height = np.asarray(d["foot_pos_w"])[full_mask, :, 2] - np.asarray(d["terrain_z"])[full_mask] - offset
    unloaded_raised = (force < TOUCH_REF_N) & (height > OFFGROUND_TOL_M)
    return unloaded_raised.mean(axis=0) > HOVER_FRAC


def off_ground_share(d, steady_mask: np.ndarray, offset: float | None) -> float | None:
    """Fraction of (foot, steady row) samples with (foot_z - terrain_z) - offset > 3 cm."""
    if offset is None or not steady_mask.any():
        return None
    height = np.asarray(d["foot_pos_w"])[steady_mask, :, 2] - np.asarray(d["terrain_z"])[steady_mask] - offset
    return float((height > OFFGROUND_TOL_M).mean())


def episode_stand_metrics(d, hold_idx: int, dt: float) -> dict:
    """Every per-episode estimator of one hold (module docstring's ESTIMATORS section), as a flat dict. Any
    estimator whose window is too short or ill-defined (e.g. an episode with no stand rows at all) reads
    ``None`` -- never 0, the ``gait_eval.metrics`` convention."""
    full, steady = hold_masks(d, hold_idx, dt)
    n_full, n_steady = int(full.sum()), int(steady.sum())
    out: dict = {"n_full": n_full, "n_steady": n_steady, "fall": fell_in_hold(d, hold_idx)}

    if n_full == 0:
        out.update({k: None for k in (
            "steps_per_s", "taps_per_s", "stance_offset_m", "clearance_m", "clearance_max_m",
            "off_ground_share", "hover", "ks_m_s", "loaded_count", "body_height_m", "drift_m_s",
            "facing_err_rms", "t1_per_s", "t2_per_s",
        )})
        return out

    loaded_full = np.asarray(d["foot_force_norm"])[full] > TOUCH_REF_N
    steps, taps, _censored = M.step_tap_runs(loaded_full, min_step_steps=MIN_STEP_STEPS)
    dur_full = n_full * dt
    out["steps_per_s"] = float(len(steps) / dur_full) if dur_full > 0 else None
    out["taps_per_s"] = float(len(taps) / dur_full) if dur_full > 0 else None

    offset = stance_offset(d, full)
    out["stance_offset_m"] = offset
    clear_med, clear_max = clearance_stats(d, steady)
    out["clearance_m"], out["clearance_max_m"] = clear_med, clear_max
    out["off_ground_share"] = off_ground_share(d, steady, offset)
    out["hover"] = bool(hover_feet(d, full, offset).any())

    if n_steady == 0:
        out.update({k: None for k in (
            "ks_m_s", "loaded_count", "body_height_m", "drift_m_s", "facing_err_rms", "t1_per_s", "t2_per_s",
        )})
        return out

    force_s = np.asarray(d["foot_force_norm"])[steady]
    vel_s = np.asarray(d["foot_lin_vel_w"])[steady]
    foot_speed = np.linalg.norm(vel_s, axis=-1)                         # (n_steady, 6)
    loaded_s = force_s > TOUCH_REF_N
    s_f = np.clip(force_s / TOUCH_REF_N, 0.0, 1.0)                      # (n_steady, 6), the lever's gate variable

    out["ks_m_s"] = float(foot_speed.sum(axis=-1).mean())
    out["loaded_count"] = float(loaded_s.sum(axis=-1).mean())
    root_z = np.asarray(d["root_pos_w"])[steady, 2]
    terrain_mean = np.asarray(d["terrain_z"])[steady].mean(axis=-1)
    out["body_height_m"] = float((root_z - terrain_mean).mean())
    planar_v = np.asarray(d["root_lin_vel_b"])[steady, :2]
    out["drift_m_s"] = float(np.linalg.norm(planar_v, axis=-1).mean())
    heading = np.asarray(d["heading_err"])[steady]
    out["facing_err_rms"] = float(np.sqrt(np.mean(heading ** 2))) if heading.size else None
    out["t1_per_s"] = float((s_f.sum(axis=-1) - 3.0).mean())
    out["t2_per_s"] = float((s_f * foot_speed).sum(axis=-1).mean())
    return out


# ================================================================================================ onset latency
def episode_onset_latency(d, dt: float, move_idx: int,
                           speed_frac: float = ONSET_SPEED_FRAC, hold_s: float = ONSET_HOLD_S,
                           trail_steps: int = ONSET_TRAIL_STEPS) -> float | None:
    """Seconds from ``move_idx``'s first row until the ``trail_steps``-step trailing mean of the planar
    ``root_lin_vel_b``, projected onto the hold's own commanded direction, first reaches ``speed_frac`` of the
    commanded speed and stays at or above it for ``hold_s`` running. ``None`` (right-censored) if it never
    does inside the hold."""
    seg = np.asarray(d["segment_id"])
    idx = np.flatnonzero(seg == move_idx)
    if idx.size == 0:
        return None
    cmd0 = np.asarray(d["cmd_applied"])[idx[0], :2]
    speed_cmd = float(np.linalg.norm(cmd0))
    if speed_cmd <= 1e-9:
        return None
    unit = cmd0 / speed_cmd
    vel = np.asarray(d["root_lin_vel_b"])[idx, :2]
    along = vel @ unit                                                  # (n,)
    n = along.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(along)])
    trailing = np.empty(n)
    for t in range(n):
        lo = max(0, t - trail_steps + 1)
        trailing[t] = (cs[t + 1] - cs[lo]) / (t - lo + 1)

    threshold = speed_frac * speed_cmd
    need = max(1, int(round(hold_s / dt)))
    above = trailing >= threshold
    for t0 in range(n):
        if not above[t0]:
            continue
        window_end = min(n, t0 + need)
        if window_end - t0 < need:
            break  # not enough room left in the hold to hold it for hold_s -- censored, not a pass
        if bool(above[t0:window_end].all()):
            return float(t0 * dt)
    return None


def onset_summary(stamp: str) -> dict[str, dict]:
    """Per moving-hold-label (every ``stand -> moving hold`` pair the schedule carries): the per-episode
    onset latencies' MEDIAN and ``n`` (``None`` values -- censored episodes -- excluded from both)."""
    meta = _json(Path(stamp) / "run_meta.json") or {}
    labels, speeds, dt = meta.get("hold_labels") or [], meta.get("hold_speeds"), float(meta.get("dt", 0.02))
    if not labels or not speeds:
        return {}
    pairs = stand_onset_pairs(labels, speeds)
    if not pairs:
        return {}
    out: dict[str, dict] = {}
    for _stand_idx, move_idx, move_label in pairs:
        vals = []
        for _meta, _labels, _e, d in _raw_episodes(stamp, lambda e: True):
            v = episode_onset_latency(d, dt, move_idx)
            if v is not None:
                vals.append(v)
        out[move_label] = {"onset_median_s": round(statistics.median(vals), 4) if vals else None,
                           "n": len(vals)}
    return out


# ================================================================================================ per-head summary
def head_summary(stamp: str | None) -> dict | None:
    """The stand hold's per-head summary: the MEDIAN of every :func:`episode_stand_metrics` estimator over
    the hold's episodes, plus the hover-episode count (raw and per 100), falls, and ``n``. ``None`` if the
    stamp has no stand episodes at all (PENDING, not a crash)."""
    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    dt = float(meta.get("dt", 0.02))
    labels_seen, per_ep = None, []
    for labels, hold_idx, d in iter_stand_episodes(stamp):
        labels_seen = labels
        per_ep.append(episode_stand_metrics(d, hold_idx, dt))
    if not per_ep:
        return None

    def _med(key):
        vals = [e[key] for e in per_ep if e.get(key) is not None]
        return round(statistics.median(vals), 5) if vals else None

    n = len(per_ep)
    n_hover = sum(1 for e in per_ep if e.get("hover"))
    n_falls = sum(1 for e in per_ep if e.get("fall"))
    return {
        "stamp": stamp, "n": n, "hold_labels": labels_seen,
        "steps_per_s": _med("steps_per_s"), "taps_per_s": _med("taps_per_s"),
        "stance_offset_m": _med("stance_offset_m"),
        "clearance_m": _med("clearance_m"), "clearance_max_m": _med("clearance_max_m"),
        "off_ground_share": _med("off_ground_share"),
        "ks_m_s": _med("ks_m_s"), "loaded_count": _med("loaded_count"),
        "body_height_m": _med("body_height_m"), "drift_m_s": _med("drift_m_s"),
        "facing_err_rms": _med("facing_err_rms"),
        "t1_per_s": _med("t1_per_s"), "t2_per_s": _med("t2_per_s"),
        "hover_episodes": n_hover, "hover_episodes_per_100": round(100.0 * n_hover / n, 3) if n else None,
        "falls": n_falls,
    }


def still_clauses(summary: dict | None) -> dict[str, tuple[float | None, float, bool]]:
    """P2S_PLAN.md section 4's STILL rule, clause by clause, against one :func:`head_summary`:
    ``{clause: (value, threshold, pass)}``. A missing value (PENDING) reads ``pass = False`` -- fail closed,
    never silently true."""
    if summary is None:
        return {k: (None, v, False) for k, v in STILL_THRESHOLDS.items()}
    out = {}
    for key, threshold in STILL_THRESHOLDS.items():
        value = summary.get(key)
        out[key] = (value, threshold, bool(value is not None and value <= threshold))
    return out


# ================================================================================================ Z2 per-stand reads
# NOTE(p2r Z2, 2026-10-10): P2R_Z2_PLAN.md step 1 "Reads" -- homed / six-down / settle / post-settle / re-stance
# lifts per stand, over EVERY "stand" hold (walkstop's three included). Nothing above changes behaviour.
def wrap_to_pi(x):
    """``(x + pi) mod 2 pi - pi`` -- the cam observation's own wrapping (|value| is all any estimator reads, so
    the +-pi boundary convention does not matter)."""
    return (np.asarray(x, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def joint_columns(meta: dict, suffix: str) -> list[int] | None:
    """Indices into the raw ``joint_pos`` columns of the joints whose ``run_meta.joint_names`` end in ``suffix``
    (in articulation order). ``joint_names`` is a list on every current stamp; an older stamp that stored its repr
    is parsed. None if the names are missing or no joint matches."""
    names = meta.get("joint_names")
    if isinstance(names, str):
        try:
            names = ast.literal_eval(names)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(names, (list, tuple)):
        return None
    cols = [i for i, n in enumerate(names) if str(n).endswith(suffix)]
    return cols or None


def stand_window_kind(meta: dict) -> str:
    """The six-down window a stamp's stands are read on: ``steady`` on a category_suite stamp (it carries an
    ``env_index_map``), the last 5 s of the stand on every schedule row (walkstop, field_hard / field_light /
    field_turn)."""
    return WINDOW_STEADY if meta.get("env_index_map") else WINDOW_LAST


def iter_stand_holds(stamp: str):
    """EVERY "stand" hold of every kept episode (suite: the env_index_map's "stand" wave; schedule rows: every
    episode) as dicts ``{meta, labels, env_index, hold_idx, ordinal, d}`` -- ``ordinal`` = 0, 1, 2 for walkstop's
    three stands in schedule order. :func:`stand_hold_index` (first stand only) is left as it is."""
    for meta, labels, e, d in _raw_episodes(stamp, lambda e: e.get("label") in (None, "stand")):
        stands = [i for i, lab in enumerate(labels) if lab == "stand"]
        for k, i in enumerate(stands):
            yield {"meta": meta, "labels": labels, "env_index": e.get("env_index"), "hold_idx": i, "ordinal": k,
                   "d": d}


def _first_run_start(cond: np.ndarray, need: int) -> int | None:
    """The first index from which ``cond`` holds for ``need`` rows running, or None (a run cut off by the end of
    the window before ``need`` rows is not a settle -- censored, as :func:`episode_onset_latency`)."""
    run = 0
    for t, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= need:
            return t - need + 1
    return None


def stand_z2_metrics(d, hold_idx: int, dt: float, cam_cols: list[int] | None, hip_cols: list[int] | None = None,
                     window: str = WINDOW_LAST) -> dict:
    """One stand's Z2 estimators (module docstring, "Z2 PER-STAND ESTIMATORS"). Any estimator that cannot be read
    (no rows, no ``joint_pos`` / cam names) is None -- never 0."""
    full, steady = hold_masks(d, hold_idx, dt)
    idx = np.flatnonzero(full)
    fall = fell_in_hold(d, hold_idx)
    out: dict = {"n_full": int(idx.size), "fall": fall, "reached": bool(idx.size or fall), "window": window}
    keys = ("six_down_share", "window_rows", "homed", "cam_max_last_rad", "hip_yaw_mean_deg", "hip_yaw_max_deg",
            "settle_s", "settled", "lifts_before_settle", "post_settle_steps_per_s", "post_settle_taps_per_s",
            "post_settle_ks_m_s", "body_height_m")
    out.update({k: None for k in keys})
    if not idx.size:
        out["settled"] = False
        return out
    last_n = max(1, int(round(Z2_LAST_S / dt)))
    last = idx[-last_n:]
    force = np.asarray(d["foot_force_norm"])
    six = (force > TOUCH_REF_N).all(axis=-1)                            # (T,)
    win = np.flatnonzero(steady) if window == WINDOW_STEADY else last
    out["window_rows"] = int(win.size)
    out["six_down_share"] = float(six[win].mean()) if win.size else None

    if steady.any():
        root_z = np.asarray(d["root_pos_w"])[steady, 2]
        out["body_height_m"] = float((root_z - np.asarray(d["terrain_z"])[steady].mean(axis=-1)).mean())

    jp = np.asarray(d["joint_pos"]) if "joint_pos" in getattr(d, "files", d) else None
    cam_abs = None
    if jp is not None and cam_cols:
        cam_abs = np.abs(wrap_to_pi(jp[:, cam_cols])).max(axis=-1)      # (T,) worst leg
        out["cam_max_last_rad"] = float(cam_abs[last].max())
        out["homed"] = bool(out["cam_max_last_rad"] <= Z2_HOMED_RAD)
    if jp is not None and hip_cols:
        hy = np.degrees(np.abs(wrap_to_pi(jp[np.ix_(last, hip_cols)])))
        out["hip_yaw_mean_deg"], out["hip_yaw_max_deg"] = float(hy.mean()), float(hy.max())

    loaded = force[idx] > TOUCH_REF_N                                   # (n_full, 6)
    if cam_abs is None:                                                  # no cams: settle cannot be read
        return out
    need = max(1, int(round(Z2_SETTLE_HOLD_S / dt)))
    # NOTE(p2r Z2 cams-only, 2026-10-10): settle = HOMED held 0.5 s (six-down removed from settle; reported only)
    t0 = _first_run_start(cam_abs[idx] <= Z2_HOMED_RAD, need)
    out["settled"] = t0 is not None
    pre = loaded if t0 is None else loaded[:t0]
    out["lifts_before_settle"] = len(M.step_tap_runs(pre, min_step_steps=MIN_STEP_STEPS)[0]) if pre.shape[0] else 0
    if t0 is None:
        return out
    out["settle_s"] = float(t0 * dt)
    post = idx[t0:]
    steps, taps, _c = M.step_tap_runs(loaded[t0:], min_step_steps=MIN_STEP_STEPS)
    dur = post.size * dt
    out["post_settle_steps_per_s"] = float(len(steps) / dur)
    out["post_settle_taps_per_s"] = float(len(taps) / dur)
    out["post_settle_ks_m_s"] = float(np.linalg.norm(np.asarray(d["foot_lin_vel_w"])[post], axis=-1).sum(-1).mean())
    return out


def _median_fail_closed(vals: list) -> float | None:
    """The median of ``vals`` with every None ranked WORST (+inf): None if the median lands on one -- a clause
    read on it then fails, as an unsettled stand fails every post-settle clause."""
    if not vals:
        return None
    m = statistics.median([math.inf if v is None else float(v) for v in vals])
    return None if math.isinf(m) else round(m, 5)


def z2_head_summary(stamp: str | None, window: str | None = None) -> dict | None:
    """Every reached stand of ``stamp`` through :func:`stand_z2_metrics`, summarised (module docstring). ``window``
    overrides :func:`stand_window_kind`. None if the stamp has no stand hold at all (PENDING, not a crash)."""
    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    dt = float(meta.get("dt", 0.02))
    kind = window or stand_window_kind(meta)
    cam_cols = joint_columns(meta, CAM_JOINT_SUFFIX)
    hip_cols = joint_columns(meta, HIP_YAW_JOINT_SUFFIX)
    per, n_ep = [], set()
    for h in iter_stand_holds(stamp):
        n_ep.add(h["env_index"])
        m = stand_z2_metrics(h["d"], h["hold_idx"], dt, cam_cols, hip_cols, kind)
        if m["reached"]:
            per.append(dict(m, ordinal=h["ordinal"]))
    if not per:
        return None

    def share(pred, rows):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 5) if rows else None

    def med(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.median(vals), 5) if vals else None

    def block(rows):
        settled = [r for r in rows if r["settled"]]
        falls = sum(1 for r in rows if r["fall"])
        return {
            "n": len(rows),
            "six_down_stand_share": share(lambda r: r["six_down_share"] is not None
                                          and r["six_down_share"] >= Z2_SIXDOWN_SHARE - 1e-12, rows),
            "six_down_share_median": med("six_down_share", rows),
            "homed_share": share(lambda r: r["homed"] is True, rows),
            "settled_share": share(lambda r: r["settled"], rows),
            "settle_median_s": med("settle_s", settled),
            "lifts_median": med("lifts_before_settle", rows),
            "lifts_mean": (round(sum(r["lifts_before_settle"] for r in rows if r["lifts_before_settle"] is not None)
                                 / max(1, sum(1 for r in rows if r["lifts_before_settle"] is not None)), 5)
                           if any(r["lifts_before_settle"] is not None for r in rows) else None),
            "post_settle_steps_per_s": _median_fail_closed([r["post_settle_steps_per_s"] for r in rows]),
            "post_settle_taps_per_s": _median_fail_closed([r["post_settle_taps_per_s"] for r in rows]),
            "post_settle_ks_m_s": _median_fail_closed([r["post_settle_ks_m_s"] for r in rows]),
            "post_settle_steps_per_s_settled": med("post_settle_steps_per_s", settled),
            "post_settle_taps_per_s_settled": med("post_settle_taps_per_s", settled),
            "post_settle_ks_m_s_settled": med("post_settle_ks_m_s", settled),
            "falls": falls, "falls_per_300": round(Z2_FALLS_PER * falls / len(rows), 5),
            "body_height_m": med("body_height_m", rows),
            "hip_yaw_mean_deg": med("hip_yaw_mean_deg", rows),
            "hip_yaw_max_deg": (round(max(r["hip_yaw_max_deg"] for r in rows if r["hip_yaw_max_deg"] is not None), 5)
                                if any(r["hip_yaw_max_deg"] is not None for r in rows) else None),
            "cam_max_last_rad_median": med("cam_max_last_rad", rows),
        }
    out = {"stamp": stamp, "window": kind, "n_episodes": len(n_ep), "cams_read": bool(cam_cols)}
    out.update(block(per))
    ords = sorted({r["ordinal"] for r in per})
    if len(ords) > 1:
        out["by_stand"] = {str(k): block([r for r in per if r["ordinal"] == k]) for k in ords}
    return out


# ================================================================================================ baseline table
#: P2S_PLAN.md step 1's "C0/CL/TG/A1/F1 = newest stamp under .../seed18_X{12,13,14,06,09}_*/" mapping (verified
#: against each arm's own $OUT/arms/<name>.record.json run_dir, which names the X-id the queue assigned it).
_SUITE_X_ID = {"C0": "X12", "CL": "X13", "TG": "X14", "A1": "X06", "F1": "X09"}
_PHASE2_DIR = GE_PHASES / "2026-10-07_omni_phase2"
_E6_STATUE_STAMP = str(
    GE_PHASES / "2026-09-23_clock_deprecation/step0_E6_statue/suite__omni/seed001/2026-09-23_21-59-41"
)
_REF_NAME, _C0_NAME, _TG_NAME = "REF_S1_s18", "P2__C0_s18", "P2__TG_s18"


def _suite_stamp_for(short_name: str) -> str | None:
    x_id = _SUITE_X_ID[short_name]
    chain_dir = newest_chain_dir(_PHASE2_DIR, f"seed18_{x_id}_*")
    if chain_dir is None:
        return None
    return newest_under(chain_dir, SUITE, 1)


def baseline_table() -> dict:
    """The P2S_PLAN.md step-1 baseline: suite stands for REF (es1 + es2) and C0/CL/TG/A1/F1/E6; plus, for
    REF and C0/TG, the field_hard/field_light opening stands + onset and the field_turn stand."""
    out: dict = {"suite": {}, "field": {}}

    ref_es1 = newest_stamp(_REF_NAME, SUITE, 1, LROUT)
    ref_es2 = newest_stamp(_REF_NAME, SUITE, 2, LROUT)
    out["suite"]["REF_es1"] = head_summary(ref_es1)
    out["suite"]["REF_es2"] = head_summary(ref_es2)
    for short_name in ("C0", "CL", "TG", "A1", "F1"):
        out["suite"][short_name] = head_summary(_suite_stamp_for(short_name))
    out["suite"]["E6_statue"] = head_summary(_E6_STATUE_STAMP)

    for label, name in (("REF", _REF_NAME), ("C0", _C0_NAME), ("TG", _TG_NAME)):
        entry: dict = {}
        for row in (FH, FL):
            stamp = newest_stamp(name, row, 1, LROUT)
            entry[row] = {"stand": head_summary(stamp), "onset": onset_summary(stamp) if stamp else {}}
        ft_stamp = newest_stamp(name, FT, 1, LROUT)
        entry[FT] = {"stand": head_summary(ft_stamp)}
        out["field"][label] = entry
    return out


def _fmt(v, nd=3) -> str:
    return "." if v is None else f"{v:.{nd}f}"


def baseline_markdown(table: dict) -> str:
    cols = ("n", "steps_per_s", "taps_per_s", "clearance_m", "off_ground_share", "ks_m_s",
            "hover_episodes", "falls")
    lines = ["## suite__omni stand", "", "| head | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for head, summary in table["suite"].items():
        if summary is None:
            lines.append(f"| {head} | " + " | ".join(["PENDING"] * len(cols)) + " |")
            continue
        row = [str(summary.get(c)) if c in ("n", "hover_episodes", "falls") else _fmt(summary.get(c))
               for c in cols]
        lines.append(f"| {head} | " + " | ".join(row) + " |")
    lines += ["", "## field rows (stand + onset)", "",
             "| head | row | n | steps_per_s | clearance_m | onset_s |", "|---|---|---|---|---|---|"]
    for label, rows in table["field"].items():
        for row_name, entry in rows.items():
            s = entry.get("stand")
            onset = entry.get("onset") or {}
            onset_s = next(iter(onset.values()), {}).get("onset_median_s") if onset else None
            n = s.get("n") if s else "PENDING"
            lines.append(f"| {label} | {row_name} | {n} | {_fmt(s.get('steps_per_s')) if s else '.'} | "
                        f"{_fmt(s.get('clearance_m')) if s else '.'} | {_fmt(onset_s)} |")
    return "\n".join(lines)


# ================================================================================================ pre-freeze self-check
def _assert_matches_metrics_json(stamp: str) -> list[str]:
    """P2S_PLAN.md step 1's pre-freeze requirement: our ``steps_per_s`` on the suite stand equals the
    harness's own ``metrics/episode_XX.json`` ``holds.stand.rigidity.steps_per_s``, episode by episode.
    Returns a list of mismatch descriptions (empty = PASS)."""
    meta = _json(Path(stamp) / "run_meta.json") or {}
    dt = float(meta.get("dt", 0.02))
    labels = meta.get("hold_labels") or []
    hold_idx = stand_hold_index(labels)
    if hold_idx is None:
        return [f"{stamp}: no 'stand' hold in hold_labels"]
    mismatches = []
    for mfile in sorted((Path(stamp) / "metrics").glob("episode_*.json")):
        ep = mfile.stem.split("_")[1]
        rigidity = (_json(mfile) or {}).get("holds", {}).get("stand", {}).get("rigidity")
        if not rigidity or rigidity.get("steps_per_s") is None:
            continue
        npz_path = Path(stamp) / "raw" / f"episode_{ep}.npz"
        if not npz_path.is_file():
            continue
        d = np.load(npz_path)
        full, _steady = hold_masks(d, hold_idx, dt)
        loaded = np.asarray(d["foot_force_norm"])[full] > TOUCH_REF_N
        steps, _taps, _c = M.step_tap_runs(loaded, min_step_steps=MIN_STEP_STEPS)
        got = len(steps) / (int(full.sum()) * dt) if full.any() else None
        want = float(rigidity["steps_per_s"])
        if got is None or abs(got - want) > 1e-6:
            mismatches.append(f"{stamp} episode_{ep}: steps_per_s got {got} vs harness {want}")
    return mismatches


def selfcheck() -> int:
    """P2S_PLAN.md step 1's pre-freeze gate: the E6 statue reads STILL on every clause AND TG reads
    >= 9 hover episodes (both re-checked against the exact ``metrics/episode_XX.json`` reproduction).
    Exit 0 on PASS, nonzero otherwise, printing every failure."""
    failures = []

    mismatches = _assert_matches_metrics_json(_E6_STATUE_STAMP)
    failures.extend(mismatches)

    statue = head_summary(_E6_STATUE_STAMP)
    clauses = still_clauses(statue)
    for clause, (value, threshold, ok) in clauses.items():
        print(f"E6 statue {clause}: {value} <= {threshold} -> {'STILL' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"E6 statue fails STILL clause {clause}: {value} vs {threshold}")

    tg_stamp = _suite_stamp_for("TG")
    tg = head_summary(tg_stamp)
    n_hover = tg["hover_episodes"] if tg else None
    print(f"TG hover episodes: {n_hover} (need >= 9)")
    if tg is None or n_hover is None or n_hover < 9:
        failures.append(f"TG hover episodes {n_hover} < 9")

    if failures:
        print("\nselfcheck: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nselfcheck: PASS")
    return 0


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stamp", help="one eval stamp directory (suite__omni or a schedule row)")
    ap.add_argument("--hold", default="stand", help="hold label to read (default 'stand')")
    ap.add_argument("--json", help="write the JSON result here (default: print to stdout)")
    ap.add_argument("--baseline", action="store_true", help="the P2S_PLAN.md step-1 baseline table")
    ap.add_argument("--selfcheck", action="store_true",
                     help="exit 0 iff the E6 statue is STILL on every clause and TG has >= 9 hover episodes")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if args.baseline:
        table = baseline_table()
        if args.json:
            Path(args.json).write_text(json.dumps(table, indent=1, default=str))
        print(baseline_markdown(table))
        return 0

    if not args.stamp:
        ap.error("--stamp is required unless --baseline or --selfcheck")
    if args.hold == "stand":
        summary = head_summary(args.stamp)
        result = {"summary": summary, "still_clauses": {k: v for k, v in still_clauses(summary).items()},
                 "onset": onset_summary(args.stamp), "z2": z2_head_summary(args.stamp)}
    else:
        result = {"note": f"hold {args.hold!r} not stand -- only stand-hold estimators are implemented"}
    text = json.dumps(result, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
