#!/usr/bin/env python3
"""Omnidirectional crab-hex training pipeline driver -- omni-task sibling (2026-09-16).

Runs the ``KRABBY_OMNI_PHASE`` presets of ``crab_hex_omni_phases.py`` in order -- three phase-1
formation stages (``1a`` body +x only, ``1b`` body +y only, ``1c`` any world direction + facing;
task ``Isaac-Crab-Hex-Omni-Walk-v0``) then the teacher windows (``2a/2b/2c/2c_clockheld``, task
``Isaac-Crab-Hex-Omni-Teacher-v0``) -- each resuming the previous phase's head (``2c_clockheld``
resumes ``2b``, NOT ``2c``: see the ``--phases`` note below), with the campaign evals after every
phase (world-frame flat canary + light-ring onset on ``experiments/eval/scenarios_omni.yaml``, plus
the ring obstacle eval), a REPORT / CHANGELOG / state.json record, and stop-and-wait pauses.

No student anywhere: this task is teacher-only (no depth distillation; the camera-event bug and the
+y mount decision stay open). The training subprocess environment carries ONLY
``KRABBY_OMNI_PHASE`` and ``KRABBY_PLANT``: every knob comes from the preset (``activate_phase`` at
config import). This driver never exports ``KRABBY_PHASE`` or ``KRABBY_SIDE_PHASE`` -- doing so
would raise at config import (one task per process; see ``crab_hex_omni_phases.py::activate_phase``).
Evals never export the preset either: they get the explicit canary/terrain keys the forward and
sideways campaigns used, PLUS (2026-09-17) the :data:`EVAL_CARRY` observation/command knobs lifted
out of the preset by :func:`eval_env_for`, because without them the Play env built on class
defaults and scored the policy through an observation function it had never trained under.

    run_phases.py --campaign-dir <dir> --plant A15+B --phases 1a,1b,1c,2a,2b [--seed 3]
                  [--from-checkpoint <pt> | --from-scratch] [--iterations N] [--no-eval]
                  [--pause-after 2b]

Every gate additionally reads the stage's own TRAINING log (2026-09-18): ``track_ratio_mean``
averaged over the last 500 iterations must clear ``GATE_TRACK_RATIO_MIN`` 0.6. Tripod and survival
are not translation metrics -- the seed-3 chain passed every gate while its speed tracking fell
0.747 -> 0.433 -> 0.245 and the head ended up stepping in place at ~0.04 m/s. A from-scratch window
longer than 1000 iterations is additionally aborted at iteration 1000 (``dead_walker``) if its
800-1000 ``track_ratio_mean`` is below 0.5; ``--continue`` overrides that verdict like a gate.

``--phases`` default is the campaign of record's main chain, ``1a,1b,1c,2a,2b`` (Decision 6 of the
plan): ``2c`` and ``2c_clockheld`` are BOTH resumed from the 2b head, not from each other, so they
are always launched as two SEPARATE ``--phases 2c --from-checkpoint <2b head>`` /
``--phases 2c_clockheld --from-checkpoint <2b head>`` invocations (Part H) -- never listed together
in one ``--phases`` string, which would silently chain ``2c_clockheld`` off ``2c``'s checkpoint via
the driver's normal "resume from the previous listed phase" rule instead of off ``2b``.

Resumable: rerunning skips phases already recorded ``ok`` for the seed (``--redo`` reruns them);
the first requested phase resumes from ``--from-checkpoint`` when given, from random init when
``--from-scratch`` is given, else from the recorded head of its ``resume_from`` phase. Launch
through ``launch_phases.sh`` (systemd scope) and watch with ``heartbeat_phases.sh``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
TRAIN = PARKOUR / "scripts/rsl_rl/train.py"
PKG = HERE.parents[1]
FWD_PKG = PKG.parent / "crab_hex_forward_task"
EVAL = FWD_PKG / "scripts/eval_crab_hex_gait.py"  # shared harness: one harness serves all three tasks
OMNI_MANIFEST = PKG / "experiments/eval/scenarios_omni.yaml"
PHASES_PY = PKG / "config/crab_hex/crab_hex_omni_phases.py"
LOG_ROOT = PARKOUR / "logs/rsl_rl"
EVAL_ROOT = LOG_ROOT / "gait_eval/omni_phases"
NUM_ENVS = {"rl": "256"}          # no distill kind for this task
CANARY_KEYS = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W", "KRABBY_APEX_W")
# KRABBY_HEADING / KRABBY_HEADING_STIFFNESS dropped from the forward/sideways CANARY_KEEP list: the
# omni facing sampler replaces the heading mechanism outright and _check_omni() raises if
# KRABBY_HEADING is ever set (D2/Appendix A), so this driver must never pass it through.
CANARY_KEEP = ("KRABBY_LIN_VEL_X", "KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W", "KRABBY_CLOCK_W",
               "KRABBY_APEX_W", "KRABBY_AIRTIME_W", "KRABBY_STRIDE_W", "KRABBY_ACTION_SCALE")
# ---------------------------------------------------------------- EVAL_CARRY (2026-09-17 defect fix)
# The knobs that define what the policy OBSERVES or how its COMMAND is computed. They are carried
# from the phase's preset (``ph.phase_env``) into EVERY eval -- ``eval_morph`` (flat canary /
# light-ring onset) and ``eval_obst`` (ring) alike.
#
# WHY THIS EXISTS. ``_subenv`` strips every ``KRABBY_*`` from the parent and the driver deliberately
# never exports ``KRABBY_OMNI_PHASE`` to an eval (two presets in one process raise at config
# import). Until 2026-09-17 that meant ``activate_phase`` never ran in the eval process and
# ``OMNI_COMMON`` was never expanded there, so the Play env built on CLASS DEFAULTS while the
# canary's five reward/tracking keys were the only thing carried across. The policy was therefore
# scored through a different observation/command function than the one it trained under, and every
# gate in the campaign read that number. Carrying the preset wholesale is NOT the fix (it would
# reinstate the sampler, the reward weights, the RSI resets and the training terrain the canary is
# specifically meant to strip); carrying exactly the obs/command semantics is.
#
# Per key, what it changes and why an eval must match training:
#   KRABBY_OMNI_FACING_OBS_CLIP -- ``params["facing_obs_clip"]`` on the omni observation term
#       (obs channels 6/7, the facing error). Class default is ``None`` = the RAW error, so the
#       canary's 90-degree turn holds present 1.57 rad on a policy whose training inputs never
#       left the 0.6-rad clip. The config-dump audit found this to be the ONLY difference under
#       ``observations`` between the old eval env and training, and it is worth the whole defect:
#       on the seed-3 1c head, 100 episodes, the driver's old env scored completion 0.00 (100/100
#       falls) and the same env plus this one key scored 0.39 (the full preset scored 0.49).
#   KRABBY_OMNI_SPEED_ELLIPSE -- ``speed_ellipse`` (fore-aft, lateral, backward semi-axes). On a
#       world-frame row the harness does NOT write ``vel_command_b``: it hands the term a
#       ScheduleRow and the term's own ``_write_command`` re-derives the body command through
#       ``body_command_details(..., a, b, c)``. The ellipse therefore caps and shapes the very
#       command the eval measures tracking against.
#   KRABBY_OMNI_ANG_VEL_MAX -- ``ang_vel_max``, the yaw-rate saturation in ``heading_rate_command``
#       (same ``_write_command`` path). It sets the yaw-rate command the turn holds score, and the
#       manifest's hold-length feasibility note is written against it.
#   KRABBY_OMNI_CLOCK_TURN_R -- ``cfg.omni_clock_turn_radius``, read by ``parkour_actions`` when it
#       advances the gait clock: it is an input to the CLOCK PHASE the policy observes, not a
#       reward weight.
#   KRABBY_OMNI_YAW_RANGE -- the ring reset event's ``yaw_range`` (and the RSI re-yaw). Every row
#       in ``scenarios_omni.yaml`` is world-frame with ``spawn_yaw: first_face``, so the harness
#       pins this to ``(face0, face0)`` after the cfg is built and the carried value is inert
#       TODAY; it is carried so a body-frame or unpinned row added later spawns from the band the
#       policy trained on rather than the class default, and so ``_apply_spawn_yaw``'s structural
#       guard (a leaked forward reset event has no ``yaw_range`` param) runs in the eval too.
#
# NOT carried, deliberately: KRABBY_OMNI_SLOTS / _DIR_BAND / _FACING / _RSI_SOURCE, KRABBY_RSI_*,
# KRABBY_YAW_W, KRABBY_YAW_PROGRESS_W, KRABBY_STAND_FRAC, KRABBY_RESAMPLE_S and the
# KRABBY_FLAT_TERRAIN_* block. Those drive SAMPLING, REWARD and TERRAIN -- the canary is a minimal
# flat eval whose commands come from the manifest schedule override (the sampler never runs) and
# whose verdict must not move when a stage re-weights a reward.
#
# heading_control_stiffness is NOT here and needs no knob: ``apply_flat_walk_knobs`` sets it to 1.5
# unconditionally (crab_hex_env_cfg.py:563) in every crab-hex env, training and Play alike; the
# omni command cfg's Go2 literal 0.8 is overwritten before ``apply_omni_knobs`` runs. Only
# KRABBY_HEADING_STIFFNESS would change it and no omni preset sets it (see the CANARY_KEEP note).
EVAL_CARRY = ("KRABBY_OMNI_FACING_OBS_CLIP", "KRABBY_OMNI_SPEED_ELLIPSE", "KRABBY_OMNI_ANG_VEL_MAX",
              "KRABBY_OMNI_CLOCK_TURN_R", "KRABBY_OMNI_YAW_RANGE")
# Evals never carry RSI resets: set explicitly (not merely omitted) so a preset that later grows a
# different default, or a leaked parent value, cannot re-arm reference-state resets inside an eval.
# apply_omni_knobs / the forward knob block build events.rsi_reference_reset only for RSI_FRAC > 0.
EVAL_RSI_OFF = {"KRABBY_RSI_FRAC": "0"}
# Headless evals never render the PLAY cfg's debug markers: 100-env evals with the markers on grew to
# ~59 GB RSS and were OOM-killed (2026-09-18, seed-3 2a: all three rows died -> gate "could not be evaluated").
EVAL_DEBUG_VIS_OFF = {"KRABBY_PLAY_DEBUG_VIS": "0"}
# Ring knobs (D8's apply_omni_ring_knobs), not the forward/sideways corridor's KRABBY_CORRIDOR_HALF_WIDTH
# / KRABBY_STONE_WIDTH -- those are dead for the ring geometry.
TERRAIN_PASS = ("KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_FLAT_TERRAIN_DIFF", "KRABBY_RING_CLEAR_R", "KRABBY_RING_BLOCKS")
# Every omni process needs this, evals included: require_ring_geom() raises without it (D8/B7).
TERRAIN_GEOM_REQUIRED = {"KRABBY_FLAT_TERRAIN_GEOM": "ring"}
# KRABBY_FLAT_TERRAIN_MODE/_FLAT_FRAC are NOT optional here either, same as the sideways driver's
# own obstacle row (run_phases.py's eval_obst): apply_flat_walk_knobs only reads GEOM/DIFF (and
# dispatches the sub-terrain proportions away from all-flat) inside its "if KRABBY_FLAT_TERRAIN_MODE
# in (light, ...)" branch -- without MODE=light the earlier, unconditional block in that same knob
# loop still forces every sub-terrain except "parkour_flat" (the omni ring tiles included) to
# proportion 0.0, so the eval would silently build pure-flat terrain instead of the ring. FLAT_FRAC
# 0.0 then hands the ring sub-terrains the entire remaining share (PLAN H convention).
OBST_EVAL = {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:0.70",
             "KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0"}
SMOKE_ITERS = 2000       # soundness window (rung-v precedent)
STALL_S = 900
DEFAULT_PAUSE = "2b"     # the campaign's decision point (Part H); 2c/2c_clockheld are always separate launches
GATE_SOFT_TRIPOD = 0.30      # 1a/1b soft-gate floor (short stages: formation is not yet load-bearing)
GATE_SOFT_COMPLETION = 0.70
GATE_1C_TRIPOD = 0.40        # written formation gate (Part H), each moving hold
GATE_1C_COMPLETION = 0.70
GATE_1C_YAW_RATE = 0.05      # rad/s -- the yaw-capability gate's own floor (10x the record 0.003),
                             # read off the SIGNED yaw_progress_rate (see gate_1c_verdict)
GATE_1C_DIR_RATIO = 0.3      # per-direction tracking floor (step-16 review decision 3, 2026-09-17):
                             # median tracking_by_hold[h]["ratio"] on each of GATE_1C_DIR_HOLDS --
                             # coverage hole found in review: a forward-only walker must not pass an
                             # omnidirectional formation gate on tripod/completion alone
GATE_FACING_MAX_RAD = 0.30   # rad, median heading_error_final_abs on the turn holds (enforced from 2a)
# ------------------------------------------------- the training-log translation floor (2026-09-18)
# Every gate (soft 1a/1b and gate_1c) additionally reads the stage's OWN training log. The verified
# 1c diagnosis (ANALYSIS_1c_failure_2026-09-18.md, claims 2 and 4) is that the translation collapse
# was visible in training long before any eval -- track_ratio_mean 0.747 (1a) -> 0.433 (1b) -> 0.245
# (1c) -- while every gate passed it, because tripod and survival are not translation metrics: a
# head that steps in place with a clean rhythm scores 0.85 tripod and 1.00 survival at ~0 m/s.
# This floor is the cheapest possible cover for that hole: no GPU, no eval, one number the trainer
# already prints every iteration.
TRACK_RATIO_LABEL = "Metrics/base_parkour/track_ratio_mean"
GATE_TRACK_RATIO_MIN = 0.6   # mean over the last GATE_TRACK_RATIO_N iterations of the stage's log
GATE_TRACK_RATIO_N = 500     # measured on the seed-3 campaign logs: 1a 0.693 PASS (its 800-1000:
                             # 0.747), 1b 0.406 FAIL, 1c 0.246 FAIL -- i.e. this floor would have
                             # halted the chain at 1b, before 1c burned 2.5 GPU-h
# ------------------------------------------------- the from-scratch early abort (2026-09-18, D)
# A from-scratch window is cheap to abandon early and expensive to finish blind. User decision D
# asks for a hard stop at 1000 iterations if the walker never formed: 1a reached 0.747 over its
# 800-1000 window from scratch in exactly that budget, so 0.5 is a floor no forming walker fails.
ABORT_AT_ITERS = 1000            # iterations after which the check fires (once)
ABORT_WINDOW = 200               # iterations averaged: its 800-1000
ABORT_TRACK_RATIO_MIN = 0.5
SOFT_GATE_HOLD = {"1a": "fwd", "1b": "lat_l"}
FACING_GATE_PHASES = ("2a", "2b")
TURN_HOLDS = ("turn_l90", "turn_r90")
DIRECTION_HOLDS = ("fwd", "lat_l", "diag", "back")
GATE_1C_DIR_HOLDS = ("lat_l", "diag", "back", "move_turn")   # gate_1c's per-direction floor holds; fwd is
                                                              # already constrained by the tripod/completion criteria
STAGE_DIR_BAND = {"1a": (0.0, 0.0), "1b": (1.5707963267948966, 1.5707963267948966)}   # 1c+ = no band (full range)
STAGE_RSI_SOURCE = {"1a": "0", "1b": "1"}                                              # 1c+ = "all"
WALKING_CMD_MIN = 0.2     # documentation: the harness's lin_vel_clip -- a hold commanded at |cmd| <= this
                          # is a stand hold and gets tracking ratio n == 0, which is what walking_tripod() reads
GATE_STATUSES = {"gate_1a", "gate_1b", "gate_1c", "gate_facing"}
#: Statuses ``--continue`` may override by carrying the recorded checkpoint forward unretrained.
#: ``dead_walker`` joins the gates because it is the same KIND of verdict -- a numeric floor on a
#: head that exists -- unlike ``dead_plant`` / ``nan`` / ``infra``, which describe a run that
#: produced nothing usable.
OVERRIDABLE_STATUSES = GATE_STATUSES | {"dead_walker"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ph = _load(PHASES_PY, "crab_hex_omni_phases")

# ------------------------------------------------------------------ inlined helpers
# Verbatim structure from the sideways driver (2026-09-11), itself inlined from the forward
# campaign scripts (run_exposure.py / run_formation_arms.py); re-pointed for the omni task and
# stripped of every distillation-only path (this task has no student).
EXPO_BASE = ["reach_edge_frac", "reach_obst_frac", "field_frac_mean", "field_steps_mean",
             "goals_passed_mean", "track_ratio_mean", "heading_err_rms_mean", "block_steps_mean", "r_max_mean"]
EXPO_KEYS = EXPO_BASE + [
    "crab_failure_flat", "crab_failure_obst", "ep_steps_flat", "ep_steps_obst",
    "rsi_frac_actual", "terrain_levels", "how_far_from_start_point", "current_goal_idx"]


def series(log_path: Path, label: str) -> list[float]:
    pat = re.compile(re.escape(label) + r":\s*(-?[0-9.]+(?:e-?\d+)?)")
    out = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                pass
    return out


def tail_mean(vals: list[float], n: int = 20) -> float:
    t = vals[-n:]
    return sum(t) / len(t) if t else float("nan")


def fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"
    return str(v)


# ------------------------------------------------------------------ translation floor (pure)
def track_ratio_series(log_path: Path) -> list[float]:
    """Every ``Metrics/base_parkour/track_ratio_mean`` value in a training log, in order.

    The trainer prints one line per iteration, so index == iteration for a from-scratch run
    (a RESUMED run's list still starts at this run's first iteration, which is why the early
    abort below is from-scratch-only)."""
    return series(log_path, TRACK_RATIO_LABEL)


def track_ratio_tail(vals: list[float], n: int = GATE_TRACK_RATIO_N) -> float | None:
    """Mean of the last ``n`` values, or ``None`` when the log carried none.

    ``None`` rather than ``nan`` on purpose: the gates treat "no usable number" as a halt, and a
    silent ``nan`` compares False against every floor -- which would read as a PASS.
    """
    if not vals:
        return None
    m = tail_mean(vals, n)
    return None if m != m else m


def dead_walker(vals: list[float], at_iters: int = ABORT_AT_ITERS, window: int = ABORT_WINDOW,
                floor: float = ABORT_TRACK_RATIO_MIN) -> bool:
    """True once a FROM-SCRATCH run has reached ``at_iters`` with no walker forming.

    Reads the mean of the ``window`` iterations ending at ``at_iters`` (its 800-1000 by default),
    NOT the whole tail: the first few hundred iterations of any from-scratch run are noise, and
    including them would drag the mean down and abort a run that is forming late.

    Returns False while the run is shorter than ``at_iters`` -- the check fires once, on the way
    past, and never again (a later dip is the gates' business, not the aborter's).
    """
    if len(vals) < at_iters:
        return False
    seg = vals[at_iters - window:at_iters]
    if not seg:
        return False
    return (sum(seg) / len(seg)) < floor


def _med(x):
    if isinstance(x, dict):
        return x.get("median") if x.get("median") is not None else x.get("mean")
    return x


def exposure_from_log(tlog: Path, n: int = 20) -> dict:
    out = {k: tail_mean(series(tlog, f"Metrics/base_parkour/{k}"), n) for k in EXPO_KEYS}
    out["stand_frac_actual"] = tail_mean(series(tlog, "Metrics/base_velocity/stand_frac_actual"), n)
    out["move_frac_actual"] = tail_mean(series(tlog, "Metrics/base_velocity/move_frac_actual"), n)
    out["turn_frac_actual"] = tail_mean(series(tlog, "Metrics/base_velocity/turn_frac_actual"), n)
    out["move_turn_frac_actual"] = tail_mean(series(tlog, "Metrics/base_velocity/move_turn_frac_actual"), n)
    out["failure_all"] = tail_mean(series(tlog, "Episode_Termination/crab_failure"), n)
    out["eplen"] = tail_mean(series(tlog, "Mean episode length"), n)
    out["mean_reward"] = tail_mean(series(tlog, "Mean reward"), n)
    out["vloss"] = tail_mean(series(tlog, "Mean value_function"), 200)
    out["track"] = tail_mean(series(tlog, "Episode_Reward/track_lin_vel_xy_exp"), n)
    out["prints"] = len(series(tlog, "Mean reward"))
    return out


def expo_lines(ex: dict) -> list[str]:
    return [
        f"- exposure (ring): reach_edge {fmt(ex.get('reach_edge_frac'))} | reach_obst "
        f"{fmt(ex.get('reach_obst_frac'))} | field_frac {fmt(ex.get('field_frac_mean'))} (steps "
        f"{fmt(ex.get('field_steps_mean'))}) | goals_passed {fmt(ex.get('goals_passed_mean'))}",
        f"- terrain-promotion telemetry: track_ratio_mean {fmt(ex.get('track_ratio_mean'))} | "
        f"heading_err_rms_mean {fmt(ex.get('heading_err_rms_mean'))} | block_steps_mean "
        f"{fmt(ex.get('block_steps_mean'))} | r_max_mean {fmt(ex.get('r_max_mean'))}",
        f"- failure share flat {fmt(ex.get('crab_failure_flat'))} | obst {fmt(ex.get('crab_failure_obst'))} | "
        f"campaign-wide {fmt(ex.get('failure_all'))}",
        f"- slot fractions (logged): stand {fmt(ex.get('stand_frac_actual'))} | move {fmt(ex.get('move_frac_actual'))} | "
        f"turn {fmt(ex.get('turn_frac_actual'))} | move_turn {fmt(ex.get('move_turn_frac_actual'))} | "
        f"rsi_frac_actual {fmt(ex.get('rsi_frac_actual'))} | terrain_levels {fmt(ex.get('terrain_levels'))} | "
        f"mean ep len {fmt(ex.get('eplen'))} | prints {ex.get('prints')}",
    ]


def walking_tripod(a: dict) -> tuple[float | None, dict[str, float | None]]:
    """Per-hold tripod medians and their minimum over the WALKING holds of the schedule.

    Same rule as the sideways/forward drivers (2026-09-11 finding, carried unchanged): the harness's
    episode ``tripod_score`` is a steady-step-weighted mean over EVERY hold including the cmd-0
    stand hold with no command gate (gait_eval/report.py), so a head that truly stands at zero
    command scores low there while a head that keeps stepping through its stop does not. The gate
    therefore reads the holds the harness itself scores for tracking
    (``tracking_by_hold[h].ratio.n > 0``, i.e. ``|cmd| > lin_vel_clip``) and takes the minimum, so
    every walking direction/slot must clear the floor. On the omni schedule the walking holds are
    ``fwd, lat_l, diag, back, move_turn`` (``stand``, ``turn_l90``, ``turn_r90`` command zero speed
    and are excluded); which holds are "walking" is read from the data, never hardcoded.
    """
    by_hold = {h: (v or {}).get("median") for h, v in (a.get("tripod_score_by_hold") or {}).items()}
    track = a.get("tracking_by_hold") or {}
    walking_holds = [h for h, t in track.items() if ((((t or {}).get("ratio") or {}).get("n")) or 0) > 0]
    for h in walking_holds:
        by_hold.setdefault(h, None)          # named in the by-hold string as n/a
    meds = [by_hold[h] for h in walking_holds]
    if not meds or any(m is None for m in meds):
        return None, by_hold
    return min(meds), by_hold


def _facing_by_hold(a: dict) -> dict[str, dict]:
    """Per-hold facing medians. ``progress`` is the GATED turning quantity (the harness's signed
    ``yaw_progress_rate``, rad/s toward the command); ``net`` the signed net rotation toward the
    hold's target; ``yaw_rate_ratio`` is carried for continuity only and is NOT gated -- it is a
    ratio of magnitudes, so gait wobble reads as turning (2026-09-17, see the CHARTER Log)."""
    out = {}
    for h, v in (a.get("facing_by_hold") or {}).items():
        out[h] = {"final": _med(v.get("heading_error_final_abs")),
                  "settle": _med(v.get("settle_time_s")),
                  "progress": _med(v.get("yaw_progress_rate")),
                  "net": _med(v.get("net_yaw_rate_toward")),
                  "yaw_rate_ratio": _med(v.get("yaw_rate_ratio")),
                  "rms": _med(v.get("heading_error_rms")),
                  "max": _med(v.get("heading_error_max_abs"))}
    return out


def _speed_ratio_by_hold(a: dict, holds: tuple[str, ...] = DIRECTION_HOLDS) -> dict[str, float | None]:
    return {h: _med((v or {}).get("ratio")) for h, v in (a.get("tracking_by_hold") or {}).items() if h in holds}


PROLOGUE_S = 1.5   # gait_eval/schedule.py DEFAULTS["prologue_s"]; the schedule starts after it


def _manifest_hold_s(sid: str) -> dict[str, float]:
    """``{label: hold_s}`` for scenario ``sid`` of the omni manifest (the durations the survival
    windows are cut at). Parsed here rather than pinned so a manifest edit cannot desynchronise."""
    import yaml  # the harness's own dependency
    doc = yaml.safe_load(OMNI_MANIFEST.read_text())
    for sc in doc.get("scenarios", []):
        if sc.get("id") == sid:
            return {str(h["label"]): float(h["hold_s"]) for h in sc["schedule"]}
    raise KeyError(f"scenario {sid!r} not in {OMNI_MANIFEST}")


def survival_by_hold(episodes: list, hold_labels: list, hold_s: dict, *, prologue_s: float = PROLOGUE_S,
                     dt: float = 0.02) -> dict[str, float | None]:
    """Fraction of episodes that reached the END of each hold (``n_steps >= end step of the hold``).

    The soft gates for the short stages read this instead of ``schedule_completion_rate``: the
    ``slow__omni`` schedule issues all eight holds, so a 1a head that walks forward perfectly
    still "fails" whole-schedule completion by falling on the lateral/backward/turn holds it never
    trained (2026-09-17: 0.09 completion, 93 % survival through ``fwd``). ``None`` for a hold
    with no duration or when there are no episodes.
    """
    if not episodes or not hold_labels:
        return {str(h): None for h in hold_labels}
    out: dict[str, float | None] = {}
    t_end = float(prologue_s)
    steps = [int(e.get("n_steps", 0)) for e in episodes]
    for h in hold_labels:
        dur = hold_s.get(str(h))
        if dur is None:
            out[str(h)] = None
            continue
        t_end += float(dur)
        end_step = int(round(t_end / dt))
        out[str(h)] = sum(1 for n in steps if n >= end_step) / len(steps)
    return out


def summarize(a: dict | None) -> dict:
    if not a:
        return {}
    fc = a.get("fall_classes", {})
    poly = a.get("support_polygon", {})
    tri_walk, tri_by_hold = walking_tripod(a)
    fac_by_hold = _facing_by_hold(a)
    return {"tripod": _med(a.get("tripod_score")),
            "tripod_walking": tri_walk,       # the gated number (min over walking holds)
            "tripod_by_hold": tri_by_hold,
            "completion": a.get("schedule_completion_rate"),
            "survival_by_hold": a.get("survival_by_hold"),      # reached the END of each hold (the soft gates' number)
            "tracking": _med(a.get("tracking_ratio")),
            "speed_ratio_by_hold": _speed_ratio_by_hold(a),      # fwd/lat_l/diag/back (D12: per-direction tracking)
            "dir_ratio_by_hold": _speed_ratio_by_hold(a, GATE_1C_DIR_HOLDS),  # lat_l/diag/back/move_turn
                                                                  # (step-16 review decision 3: gate_1c's per-direction floor)
            "facing_by_hold": fac_by_hold,                       # final/settle/progress/net per hold (D12: facing)
            # The gated turning number and its cross-check, hoisted out for the gate and the REPORT.
            "yaw_progress_by_hold": {h: v.get("progress") for h, v in fac_by_hold.items()},
            "net_yaw_by_hold": {h: v.get("net") for h, v in fac_by_hold.items()},
            "facing_error_rms": _med(a.get("facing_error_rms")),
            "falls": a.get("termination_reasons", {}).get("fall", 0),
            "n": a.get("n_episodes"),
            "pitch_fwd_share": (fc.get("pitch_fwd", 0) / max(1, sum(v for k, v in fc.items() if k != "none")))
            if any(k != "none" for k in fc) else None,
            "pitch_max_abs_p50": ((a.get("orientation") or {}).get("pitch_max_abs") or {}).get("median"),
            "prefall_tip_p25": (poly.get("prefall_tip_angle_fwd_deg") or {}).get("p25"),
            "walk_tip_p50": (poly.get("walking_tip_angle_fwd_deg") or {}).get("median")}


# ------------------------------------------------------------------ campaign record
class Campaign:
    def __init__(self, cdir: Path, plant: str, seed: str):
        self.dir = cdir
        self.plant = plant
        self.seed = seed
        self.key = f"seed{seed}"
        self.state_path = cdir / "state.json"
        self.report = cdir / "REPORT.md"
        self.changelog = cdir / "CHANGELOG.md"
        self.logs = cdir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.st = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"campaign": self.dir.name, "plant": self.plant, "phase": "run", "run_no": 0, "keys": {}, "current": {}}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.st, indent=2))
        heads = {k: {p: r.get("ckpt") for p, r in recs.items() if r.get("ckpt")} for k, recs in self.st["keys"].items()}
        (self.dir / "heads.json").write_text(json.dumps(heads, indent=2))

    @property
    def recs(self) -> dict:
        return self.st["keys"].setdefault(self.key, {})

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with (self.dir / "orchestrator.log").open("a") as fh:
            fh.write(line + "\n")

    def changelog_row(self, row: str) -> None:
        with self.changelog.open("a") as fh:
            fh.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — {row}\n")

    def notify(self, msg: str) -> None:
        with self.changelog.open("a") as fh:
            fh.write(f"> NOTIFY: {msg}\n")
        with (self.dir / "notify.log").open("a") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M} {msg}\n")
        self.log(f"NOTIFY: {msg}")

    def report_block(self, block: str) -> None:
        with self.report.open("a") as fh:
            fh.write(block.rstrip("\n") + "\n\n")

    def ensure_header(self, phases: list[str], args) -> None:
        if self.report.exists():
            return
        rows = ["| phase | task / mode | iterations | resume from | notes |", "|---|---|---|---|---|"]
        for i, name in enumerate(phases):
            s = ph.PHASES[name]
            mode = f"`{s.task}`" + (f" + mode `{s.teacher_mode}`" if s.teacher_mode else "")
            resume = s.resume_from or "scratch"
            if i == 0 and getattr(args, "from_scratch", False):
                resume = "scratch (--from-scratch)"
            elif i == 0 and args.from_checkpoint:
                resume = f"`{Path(args.from_checkpoint).name}` (--from-checkpoint)"
            rows.append(f"| {name} | {mode} | {args.iterations or s.iterations} | {resume} | {s.notes} |")
        head = (f"# Omni-formation campaign `{self.dir.name}`\n\n"
                f"Plant **{self.plant}** (`{ph.plant_usd_path(self.plant) or 'assets/crab.usda (main)'}`), seed {self.seed}, "
                f"phases {', '.join(phases)}. Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_phases.py`; presets: "
                f"`crab_hex_omni_phases.py` (`KRABBY_OMNI_PHASE` / `KRABBY_PLANT`).\n\n" + "\n".join(rows))
        self.report.write_text(head + "\n\n")
        self.changelog.write_text(f"# CHANGELOG — {self.dir.name}\n")
        self.changelog_row(f"campaign start: plant {self.plant}, seed {self.seed}, phases {', '.join(phases)}"
                           + (f", iterations override {args.iterations}" if args.iterations else "")
                           + (", --from-scratch (first phase ignores its resume_from)"
                              if getattr(args, "from_scratch", False) else ""))


# ------------------------------------------------------------------ subprocess plumbing
def _subenv(env_vars: dict) -> dict:
    assert "KRABBY_PHASE" not in env_vars, "omni driver must never export KRABBY_PHASE"
    assert "KRABBY_SIDE_PHASE" not in env_vars, "omni driver must never export KRABBY_SIDE_PHASE"
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="yes", TERM="xterm")
    for k in list(env):
        if k.startswith("KRABBY_"):
            del env[k]
    env.update({k: str(v) for k, v in env_vars.items()})
    return env


def _wait_isaac_clear() -> None:
    pattern = r"isaac_venv/bin/python [^ ]*(eval_crab_hex_gait|rsl_rl/train|training_timeline_probe|crab_hex_phase_cfg_dump)\.py"
    for _ in range(60):
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        if not any(int(p) != os.getpid() for p in out):
            return
        time.sleep(10)


def _ckpt_iter(ckpt: str | None) -> int:
    return int(Path(ckpt).stem.split("_")[1]) if ckpt else 0


def latest_run_ckpt(exp_dir: Path, t0: float) -> str | None:
    runs = [p for p in exp_dir.iterdir() if p.is_dir()] if exp_dir.exists() else []
    if not runs:
        return None
    run_dir = max(runs, key=lambda p: p.stat().st_mtime)
    if run_dir.stat().st_mtime < t0:
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=_ckpt_iter)
    return str(models[-1]) if models else None


def with_plant(ev: dict, plant: str) -> dict:
    """Name the plant AND (for a non-main plant) point at its USD.

    2026-09-17: this used to set ``KRABBY_HEX_USD_PATH`` only. ``plant_usd_path`` returns ``None``
    for the campaign's own plant (``A15+B`` IS the main asset), so for this campaign it set
    NOTHING and the eval subprocess -- whose parent ``KRABBY_*`` are all stripped by ``_subenv`` --
    ran with no plant variable at all. ``_plant_guard`` still passed because the class-default USD
    happens to be ``assets/crab.usda`` = A15+B, but the harness recorded
    ``run_meta.plant_requested = "<default>"``: the eval was coincidentally on the right plant
    rather than asked for it, and a future change of the config default would have gone unnoticed.
    ``KRABBY_PLANT`` is now always set, which is also what the training subprocess passes.
    """
    ev = dict(ev)
    ev["KRABBY_PLANT"] = plant
    usd = ph.plant_usd_path(plant)
    if usd:
        ev["KRABBY_HEX_USD_PATH"] = usd
    return ev


def eval_env_for(spec, plant: str, *, keep: tuple[str, ...] = CANARY_KEYS,
                 keep_source: dict | None = None, extra: dict | None = None) -> dict:
    """The subprocess environment for ONE eval of ``spec`` -- pure, no ``os.environ``, no I/O.

    ``keep`` are the reward/tracking keys the row reproduces, read from ``keep_source``. The flat
    canary reads them from :data:`~crab_hex_omni_phases.FORMATION` on purpose -- a FIXED reward
    frame, so a 2a verdict is comparable with a 1c one even though 2a halves apex/airtime -- while
    the obstacle row reproduces the phase's own weights. ``EVAL_CARRY`` is different in kind and is
    ALWAYS read from the phase's preset: those keys decide what the policy SEES and what command it
    is scored against, so they must be the ones it trained under, whatever the row's reward frame
    (see the EVAL_CARRY block). Then :data:`EVAL_RSI_OFF`, the ring geometry every omni process
    requires, ``extra`` (the caller's row-specific terrain block), and the plant.

    Never contains a preset variable: ``KRABBY_OMNI_PHASE`` in an eval would re-expand the whole
    training preset (sampler, rewards, RSI, terrain) inside the Play env, and ``KRABBY_PHASE`` /
    ``KRABBY_SIDE_PHASE`` would raise at config import (one task per process).
    """
    pe = ph.phase_env(spec.name)
    src = pe if keep_source is None else keep_source
    ev = {k: src[k] for k in keep if k in src}
    ev.update({k: pe[k] for k in EVAL_CARRY if k in pe})
    ev.update(EVAL_RSI_OFF)
    ev.update(EVAL_DEBUG_VIS_OFF)
    ev.update(TERRAIN_GEOM_REQUIRED)
    if extra:
        ev.update(extra)
    ev = with_plant(ev, plant)
    for banned in ("KRABBY_OMNI_PHASE", "KRABBY_PHASE", "KRABBY_SIDE_PHASE"):
        ev.pop(banned, None)
    return ev


# ------------------------------------------------------------------ training
def train(c: Campaign, spec, tag: str, iters: int, resume_ckpt: str | None, soundness: bool,
          early_abort: bool = False) -> tuple[str | None, str, Path]:
    """One phase of training. Returns (ckpt | None, status, log).

    ``status`` in ``{ok, dead_plant, dead_walker, nan, infra}``.

    ``early_abort`` (2026-09-18, decision D) arms the from-scratch translation abort: at
    ``ABORT_AT_ITERS`` iterations, if the training log's ``track_ratio_mean`` over its 800-1000
    window is below ``ABORT_TRACK_RATIO_MIN``, no walker is forming and the remaining 4000-7000
    iterations would buy nothing -- kill and report ``dead_walker``. Unlike ``dead_plant`` the
    checkpoint IS returned when one exists, so the record has a head and ``--continue`` can carry
    it forward unretrained (:data:`OVERRIDABLE_STATUSES`) if the user disagrees with the abort.
    """
    t0 = time.time()
    log_path = c.logs / f"{tag}_train.log"
    cmd = [PY, str(TRAIN), "--task", spec.task, "--headless", "--num_envs", NUM_ENVS[spec.kind],
           "--seed", c.seed, "--max_iterations", str(iters)]
    if resume_ckpt:
        cmd += ["--resume", "--checkpoint", resume_ckpt]
    env = {"KRABBY_OMNI_PHASE": spec.name, "KRABBY_PLANT": c.plant}
    c.st["current"] = {"tag": tag, "phase": spec.name, "iters": iters, "kind": spec.kind, "log": str(log_path),
                       "resume": resume_ckpt, "target_final": _ckpt_iter(resume_ckpt) + iters - 1,
                       "started": datetime.now().isoformat(timespec="seconds")}
    c.save()
    c.log(f"{tag}: {spec.task}" + (f" mode {spec.teacher_mode}" if spec.teacher_mode else "") +
          f" | {iters} its | {'resume ' + Path(resume_ckpt).name if resume_ckpt else 'from scratch'} | env {env}")
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=PARKOUR, env=_subenv(env), stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(120)
            try:
                if time.time() - log_path.stat().st_mtime > STALL_S:
                    c.log(f"{tag}: no log progress for {STALL_S // 60} min — infra death")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "infra", log_path
                tail = log_path.read_text(errors="ignore")[-20000:]
                if re.search(r"Mean (reward|value_function|surrogate):\s*nan", tail, re.I) or "Traceback" in tail:
                    c.log(f"{tag}: NaN / traceback in the training log")
                    proc.kill(); proc.wait(timeout=60)
                    return None, "nan", log_path
                if soundness:
                    ep = series(log_path, "Mean episode length")
                    if len(ep) >= SMOKE_ITERS and tail_mean(ep, 100) < 1.2 * tail_mean(ep[:100], 100):
                        c.log(f"{tag}: DEAD PLANT — episode length {tail_mean(ep[:100], 100):.0f} -> "
                              f"{tail_mean(ep, 100):.0f} by {SMOKE_ITERS}")
                        proc.kill(); proc.wait(timeout=60)
                        return None, "dead_plant", log_path
                if early_abort:
                    tr = track_ratio_series(log_path)
                    if dead_walker(tr):
                        got = sum(tr[ABORT_AT_ITERS - ABORT_WINDOW:ABORT_AT_ITERS]) / ABORT_WINDOW
                        c.log(f"{tag}: DEAD WALKER — track_ratio_mean {got:.3f} over its "
                              f"{ABORT_AT_ITERS - ABORT_WINDOW}-{ABORT_AT_ITERS} (need >= "
                              f"{ABORT_TRACK_RATIO_MIN}); the walker is not forming — aborting early. "
                              f"NOTE: --continue carries the {ABORT_AT_ITERS}-iteration head into the next "
                              f"phase UNRETRAINED; to finish this window instead relaunch "
                              f"--phases {spec.name} --from-checkpoint <this head> --iterations "
                              f"{iters - ABORT_AT_ITERS} (then the chain with the same --phases list)")
                        proc.kill(); proc.wait(timeout=60)
                        # keep whatever head exists: the verdict is a floor on a head that ran,
                        # not a run that produced nothing (--continue may carry it forward).
                        return latest_run_ckpt(LOG_ROOT / spec.experiment, t0), "dead_walker", log_path
            except OSError:
                pass
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        return None, "infra", log_path
    base = _ckpt_iter(resume_ckpt)
    ckpt = latest_run_ckpt(LOG_ROOT / spec.experiment, t0)
    if ckpt is None or _ckpt_iter(ckpt) < base + iters - 100:
        c.log(f"{tag}: no checkpoint at >= {base + iters - 100} under {spec.experiment} (got {ckpt})")
        return None, "infra", log_path
    return ckpt, "ok", log_path


def train_retry(c: Campaign, spec, tag: str, iters: int, resume_ckpt: str | None, soundness: bool,
                early_abort: bool = False):
    ckpt, status, lp = train(c, spec, tag, iters, resume_ckpt, soundness, early_abort)
    if ckpt is None and status == "infra":
        c.log(f"{tag}: infra death — one retry")
        _wait_isaac_clear(); time.sleep(120)
        ckpt, status, lp = train(c, spec, f"{tag}_r2", iters, resume_ckpt, soundness, early_abort)
    return ckpt, status, lp


# ------------------------------------------------------------------ evals
def _run_meta_plant(run_dir: Path) -> str | None:
    for name in ("run_meta.json", "scenario_metrics.json"):
        f = run_dir / name
        if f.exists():
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            meta = d.get("run_meta", d) if isinstance(d, dict) else {}
            if isinstance(meta, dict) and meta.get("usd_path"):
                return str(meta["usd_path"])
    return None


def _plant_guard(run_dir: Path, plant: str, tag: str) -> None:
    want = ph.plant_usd_path(plant)
    got = _run_meta_plant(run_dir)
    if got is None:
        return
    ok = (want is None and Path(got).name in ph.MAIN_ASSET_NAMES) or (want is not None and Path(got).name == Path(want).name)
    if not ok:
        raise RuntimeError(f"{tag}: eval spawned plant {got} but {plant} requested {want or 'the main asset'} (rung-iv defect)")


def _run_eval(c: Campaign, tag: str, sid: str, cmd_tail: list[str], env: dict, out_root: Path, plant: str, timeout: int):
    lp = c.logs / f"{tag}_{sid}_eval.log"
    for attempt in range(3):
        t0 = time.time()
        with lp.open("a") as fh:
            try:
                subprocess.run([PY, str(EVAL), "--headless", *cmd_tail, "--no-plot", "--output-root", str(out_root)],
                               cwd=PARKOUR, env=_subenv(env), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        d = out_root / sid / "seed001"
        if d.exists():
            latest = max(d.iterdir(), key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime >= t0 and (latest / "scenario_metrics.json").exists():
                _plant_guard(latest, plant, tag)
                m = json.loads((latest / "scenario_metrics.json").read_text())
                a = m["aggregate"]
                # Survival THROUGH each hold (2026-09-17 gate fix): the whole-schedule completion
                # counts falls on holds a short stage never trained, so the soft gates read this.
                try:
                    a["survival_by_hold"] = survival_by_hold(
                        m.get("episodes") or [], (m.get("run_meta") or {}).get("hold_labels") or [],
                        _manifest_hold_s(sid), prologue_s=PROLOGUE_S,
                        dt=float((m.get("run_meta") or {}).get("dt") or 0.02))
                except Exception as exc:  # noqa: BLE001 - a missing number halts the gate, never the driver
                    c.log(f"{tag}/{sid}: survival_by_hold unavailable ({exc!r})")
                return latest, a
        c.log(f"{tag}/{sid}: no fresh metrics — retry {attempt + 1}")
        _wait_isaac_clear(); time.sleep(120)
    return None, None


def eval_morph(c: Campaign, spec, ckpt: str, scenario: str, tag: str) -> dict | None:
    """Omni-manifest eval (slow__omni / light__omni): the campaign's flat canary / light-ring onset.

    ``KRABBY_FLAT_TERRAIN_GEOM=ring`` is NOT optional here, unlike in the forward/sideways drivers:
    ``apply_omni_ring_knobs`` -> ``require_ring_geom`` raises in EVERY omni process whose
    environment does not carry it (a leaked corridor preset must fail loudly), and ``_subenv``
    strips every ``KRABBY_*`` from the parent environment. ``slow__omni`` deliberately carries no
    ``env:`` block (the manifest's frozen shape), so without this the flat canary -- the row all
    four gates read -- could not build its env at all. Passed in the SUBPROCESS environment, i.e.
    before Kit starts, so the manifest-shape reproducibility note does not apply.

    The reward frame is FORMATION's, fixed across phases so the canary number is comparable; the
    observation/command semantics are the PHASE's (:data:`EVAL_CARRY`), because a policy can only
    be measured through the inputs it was trained on. See :func:`eval_env_for`.
    """
    sid = f"{scenario}__omni"
    ev = eval_env_for(spec, c.plant, keep=CANARY_KEYS, keep_source=ph.FORMATION)
    latest, a = _run_eval(c, tag, sid, ["--manifest", str(OMNI_MANIFEST), "--scenario", sid, "--checkpoint", ckpt,
                                        "--save-raw"], ev, EVAL_ROOT / c.dir.name / tag, c.plant, 3600)
    if a is None:
        return None
    s = summarize(a)
    s["run_dir"] = str(latest)
    return s


def eval_obst(c: Campaign, spec, ckpt: str, tag: str) -> dict | None:
    """PLAN H obstacle eval (ring__omni on the ring geometry @ 0.20-0.70) on the plant.

    ``ring__omni`` carries no ``env:`` block in the manifest (frozen shape); this driver injects the
    ring mode/fraction/geometry/difficulty at run time (:data:`OBST_EVAL`), the same convention as
    the forward ``flat_walk_slow_v2`` / sideways ``flat_walk_slow_side`` obstacle rows. The manifest
    row's own ``task:`` is ``Isaac-Crab-Hex-Omni-Walk-Play-v0``, not the Teacher-Play task: the
    Teacher env cfg's ``__post_init__`` requires ``KRABBY_HEX_TEACHER_MODE`` to resolve
    (``_require_phase_teacher_mode``), which this driver never sets for a 2a/2b/2c head, so a
    Teacher-Play row could never build here.

    Unlike the canary this row reproduces the PHASE's own reward weights (``CANARY_KEEP`` +
    ``TERRAIN_PASS`` off the preset), but it carries the same :data:`EVAL_CARRY` observation/command
    knobs -- it was scored on class defaults too before 2026-09-17. See :func:`eval_env_for`.
    """
    ev = eval_env_for(spec, c.plant, keep=CANARY_KEEP + TERRAIN_PASS, extra=OBST_EVAL)
    latest, a = _run_eval(c, tag, "ring__omni", ["--manifest", str(OMNI_MANIFEST), "--scenario", "ring__omni",
                                                 "--checkpoint", ckpt], ev, Path(str(EVAL_ROOT / c.dir.name / tag) + "_obst"),
                          c.plant, 5400)
    if a is None:
        return None
    return {"completion": a.get("schedule_completion_rate") or 0.0, "tripod": _med(a.get("tripod_score")) or 0.0,
            "tripod_walking": walking_tripod(a)[0], "falls": (a.get("termination_reasons") or {}).get("fall", 0),
            "n": a.get("n_episodes"), "run_dir": str(latest)}


# ------------------------------------------------------------------ records
def smoke_fields(lp: Path) -> dict:
    fail = series(lp, "Episode_Termination/crab_failure")
    coll = series(lp, "Episode_Reward/reward_collision")
    ep = series(lp, "Mean episode length")
    return {"fail_2k": tail_mean(fail[:SMOKE_ITERS], 100) if fail else None,
            "fail_end": tail_mean(fail, 100) if fail else None,
            "coll_2k": tail_mean(coll[:SMOKE_ITERS], 100) if coll else None,
            "ep_len_2k": tail_mean(ep[:SMOKE_ITERS], 100) if ep else None}


def _grep_tuple(txt: str, key: str) -> tuple[float, float] | None:
    m = re.search(rf"{re.escape(key)}:\s*!!python/tuple\s*\n\s*-\s*([-\d.eE]+)\s*\n\s*-\s*([-\d.eE]+)", txt)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _dir_band_ok(txt: str, stage: str) -> bool:
    """Best-effort check that the per-stage direction band landed in the dumped cfg.

    The per-stage band lives on ``commands.base_velocity.dir_band`` (step 7's name for the
    ``KRABBY_OMNI_DIR_BAND`` knob), which is ``null`` from 1c on. ``travel_dir_range`` -- the
    world band the sampler falls back to -- is only the SECOND choice: it is present with the
    same ``(-pi, pi)`` value in every stage's dump, so reading it first would compare
    ``(-pi, pi)`` against 1a's ``(0, 0)`` and report FAIL on both short stages.
    """
    band = _grep_tuple(txt, "dir_band") or _grep_tuple(txt, "travel_dir_range")
    if band is None:
        return False
    if stage in STAGE_DIR_BAND:
        want = STAGE_DIR_BAND[stage]
        return abs(band[0] - want[0]) < 1e-3 and abs(band[1] - want[1]) < 1e-3
    return abs(band[1] - band[0]) > 1.0   # 1c+: the full (-pi, pi) range, not a degenerate 1a/1b band


def _rsi_source_ok(txt: str, stage: str) -> bool:
    m = re.search(r"rsi_reference_reset:\s*\n\s*func:.*\n\s*params:\s*\n((?:\s{6,}\S.*\n)*)", txt)
    block = m.group(1) if m else ""
    sm = re.search(r"^\s*source:\s*(\S+)", block, re.M)
    got = sm.group(1).strip("'\"") if sm else None
    want = STAGE_RSI_SOURCE.get(stage, "all")
    return got == want


def phase_checks(rec: dict, spec, c: Campaign) -> list[tuple[str, bool]]:
    """Recomputable from the record (no GPU): plant, horizon, telemetry, task, and the omni-specific
    classes/knobs (D12). No spawn-rotation check: unlike the forward/sideways corridors, the omni
    reset event applies a RANDOM world yaw every episode (D9), so there is no fixed quaternion to pin."""
    checks = []
    ckpt = rec.get("ckpt")
    if not ckpt:
        return [("training produced a checkpoint", False)]
    run_dir = Path(ckpt).parent
    envy = run_dir / "params" / "env.yaml"
    txt = envy.read_text(errors="ignore") if envy.exists() else ""
    want = ph.plant_usd_path(c.plant)
    checks.append((f"plant is {c.plant} (params/env.yaml usd path)", (Path(want).name in txt) if want else any(n in txt for n in ph.MAIN_ASSET_NAMES)))
    eps = ph.phase_env(spec.name, c.plant).get("KRABBY_EPISODE_S")
    if eps:
        checks.append((f"episode_length_s {eps} in params", re.search(rf"episode_length_s:\s*{float(eps)}", txt) is not None))
    agent_txt = (run_dir / "params" / "agent.yaml").read_text(errors="ignore") if (run_dir / "params" / "agent.yaml").exists() else ""
    checks.append((f"experiment {spec.experiment}", f"experiment_name: {spec.experiment}" in agent_txt))
    lp = Path(rec["log"])
    ltxt = lp.read_text(errors="ignore") if lp.exists() else ""
    checks.append(("exposure telemetry logged (Metrics/base_parkour/reach_obst_frac)", "reach_obst_frac" in ltxt))
    checks.append(("gait income telemetry logged (Episode_Reward/reward_clock_schedule)", "reward_clock_schedule" in ltxt))
    checks.append(("mirror-symmetry loss active (Loss/symmetry or mirror)", re.search(r"symmetry|mirror", ltxt, re.I) is not None))
    checks.append(("terrain-promotion telemetry logged (Metrics/base_parkour/track_ratio_mean)", "track_ratio_mean" in ltxt))
    # Revision 1 (2026-09-17): the added rotational progress term and its training telemetry must
    # both be live -- a silently inert term is exactly the failure the revision guards against.
    checks.append(("yaw-progress reward logged (Episode_Reward/reward_yaw_progress)", "reward_yaw_progress" in ltxt))
    checks.append(("yaw-progress telemetry logged (Metrics/base_velocity/yaw_progress)", "base_velocity/yaw_progress" in ltxt))
    checks.append(("omni parkour class (OmniParkourEvent) in params/env.yaml", "OmniParkourEvent" in txt))
    checks.append(("omni reset func (reset_root_state_ring) in params/env.yaml", "reset_root_state_ring" in txt))
    checks.append(("omni command class (CrabHexOmniCommand) in params/env.yaml", "CrabHexOmniCommand" in txt))
    checks.append(("omni obs func (CrabHexOmniParkourObservations) in params/env.yaml", "CrabHexOmniParkourObservations" in txt))
    checks.append(("ring generator installed (omni_ring_sparse/dense sub-terrain) in params/env.yaml",
                   "omni_ring_sparse" in txt or "omni_ring_dense" in txt))
    checks.append(("omni RSI bank path (rsi_bank_omni.npz) in params/env.yaml", "rsi_bank_omni.npz" in txt))
    checks.append((f"dir_band matches stage {spec.name} ({STAGE_DIR_BAND.get(spec.name, 'full range')})", _dir_band_ok(txt, spec.name)))
    checks.append((f"RSI source matches stage {spec.name} ({STAGE_RSI_SOURCE.get(spec.name, 'all')})", _rsi_source_ok(txt, spec.name)))
    return checks


def resume_label(rec: dict) -> str:
    """What the REPORT / state record says this phase resumed from.

    ``--from-scratch`` is spelled out rather than shown as a bare ``scratch``: a phase whose spec
    says ``resume_from: 1b`` but which was deliberately started cold is the single most important
    thing to be able to read back off a record months later (decision D, 2026-09-18).
    """
    if rec.get("resume"):
        return Path(rec["resume"]).name
    return "scratch (--from-scratch)" if rec.get("from_scratch") else "scratch"


def phase_lines(rec: dict, spec) -> list[str]:
    ex = rec.get("exposure") or {}
    s = (rec.get("evals") or {}).get("slow") or {}
    l = (rec.get("evals") or {}).get("light") or {}
    o = (rec.get("evals") or {}).get("ring") or {}
    ck = Path(rec["ckpt"]).parent.name + "/" + Path(rec["ckpt"]).name if rec.get("ckpt") else "—"
    lines = [f"- status **{rec.get('status')}** | checkpoint `{ck}` | resume `{resume_label(rec)}` | "
             f"{rec.get('iters')} its | wall {rec.get('wall_h', float('nan')):.2f} h"]
    sm = rec.get("smoke") or {}
    lines.append(f"- smoke fail@2k {fmt(sm.get('fail_2k'))} fail@end {fmt(sm.get('fail_end'))} coll@2k {fmt(sm.get('coll_2k'))} ep_len@2k {fmt(sm.get('ep_len_2k'))}")
    # The gated TRAINING-log translation number (2026-09-18): the one translation metric that is
    # independent of the eval env. Written by _apply_gate; recomputed here for --no-eval runs.
    tr = rec.get("track_ratio_tail")
    if tr is None and Path(rec.get("log", "/nonexistent")).exists():
        tr = track_ratio_tail(track_ratio_series(Path(rec["log"])))
    lines.append(f"- training translation floor: track_ratio_mean (last {GATE_TRACK_RATIO_N} its) "
                 f"{fmt(tr) if tr is not None else 'n/a'} (gate floor {GATE_TRACK_RATIO_MIN})")
    if rec.get("ckpt"):
        lines += expo_lines(ex)
    if rec.get("ckpt") and rec.get("evals"):
        by_hold = " / ".join(f"{h} {fmt(v)}" for h, v in (s.get("tripod_by_hold") or {}).items()) or "n/a"
        lines.append(f"- slow canary (omni manifest): walking-hold tripod {fmt(s.get('tripod_walking'))} (gated; by hold: {by_hold}; "
                     f"episode {fmt(s.get('tripod'))}) | completion {fmt(s.get('completion'))} | "
                     f"tracking {fmt(s.get('tracking'))} | falls {s.get('falls')}/{s.get('n')} | pitch-fwd share {fmt(s.get('pitch_fwd_share'))}")
        sr = s.get("speed_ratio_by_hold") or {}
        lines.append("- per-direction speed_ratio: " + " / ".join(f"{h} {fmt(sr.get(h))}" for h in DIRECTION_HOLDS))
        fac = s.get("facing_by_hold") or {}
        if fac:
            lines.append("- facing by hold (final rad/settle s/yaw-progress rad/s; progress is the gated turning "
                         "number, net = signed net rotation toward the target): " + " | ".join(
                f"{h} {fmt(v.get('final'))}/{fmt(v.get('settle'))}/{fmt(v.get('progress'))} (net {fmt(v.get('net'))})"
                for h, v in fac.items()))
        if l:
            lb = " / ".join(f"{h} {fmt(v)}" for h, v in (l.get("tripod_by_hold") or {}).items()) or "n/a"
            lines.append(f"- light ring onset (omni manifest): walking-hold tripod {fmt(l.get('tripod_walking'))} (by hold: {lb}) | "
                         f"completion {fmt(l.get('completion'))} | falls {l.get('falls')}/{l.get('n')}")
        if o:
            lines.append(f"- ring obstacle eval (0.20-0.70): completion {fmt(o.get('completion'))} | walking-hold tripod {fmt(o.get('tripod_walking'))} "
                         f"(episode {fmt(o.get('tripod'))}) | falls {o.get('falls')}/{o.get('n')}")
    for name, ok in rec.get("checks") or []:
        lines.append(f"- check {'PASS' if ok else 'FAIL'}: {name}")
    return lines


# ------------------------------------------------------------------ gates (pure; tested standalone)
def _track_ratio_clause(track_ratio: float | None, floor: float = GATE_TRACK_RATIO_MIN) -> tuple[bool, str]:
    """``(passes, text)`` for the training-log translation floor, shared by every gate.

    ``None`` (the log carried no ``track_ratio_mean`` print at all) is a FAILURE, not a skip: the
    number is the one translation metric that needs no GPU, and a gate that silently drops it is
    the 2026-09-18 coverage hole all over again.
    """
    if track_ratio is None:
        return False, f"training track_ratio_mean n/a (need >= {floor} over the last {GATE_TRACK_RATIO_N} its)"
    return track_ratio >= floor, (f"training track_ratio_mean {track_ratio:.3f} (need >= {floor} over the last "
                                  f"{GATE_TRACK_RATIO_N} its)")


def gate_soft_verdict(summary: dict, hold: str, tripod_min: float, completion_min: float, *,
                      track_ratio: float | None = None,
                      track_ratio_min: float = GATE_TRACK_RATIO_MIN) -> tuple[str, str]:
    """Soft per-stage gate for the short formation stages (1a: hold='fwd'; 1b: hold='lat_l').

    Reads the SPECIFIC hold's walking-tripod median, not the min over every walking hold: the
    ``slow__omni`` eval always issues all eight holds regardless of which slots the stage actually
    trained (1a samples body +x only; 1b body +y only), so only the stage's own hold is a fair test.
    A missing number (three eval attempts failed, or the hold was never scored) halts, same as a
    numeric failure: an ungated head must not carry on unattended.

    THIRD CRITERION (2026-09-18): the stage's own TRAINING log must show
    ``track_ratio_mean >= track_ratio_min`` averaged over its last ``GATE_TRACK_RATIO_N``
    iterations. Tripod and survival are gait/robustness metrics, not translation metrics -- the
    seed-3 1b head passed this gate at 0.360 tripod / 0.96 survival while its speed tracking had
    already fallen to 0.406, and 1c then drifted to 0.245 unwatched. ``track_ratio`` is passed in
    by :func:`_apply_gate` (via :func:`track_ratio_tail`) rather than read here, so the whole
    verdict stays pure and testable; ``None`` halts.
    """
    tri_by_hold = summary.get("tripod_by_hold") or {}
    tri = tri_by_hold.get(hold)
    # Survival THROUGH the stage's own hold, not whole-schedule completion (2026-09-17 fix: the
    # first 1a halt had 0.09 completion from falls on five untrained holds and 0.93 through fwd).
    surv_by_hold = summary.get("survival_by_hold") or {}
    comp = surv_by_hold.get(hold)
    whole = summary.get("completion")
    by_hold = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in tri_by_hold.items()) or "none"
    tr_ok, tr_str = _track_ratio_clause(track_ratio, track_ratio_min)
    if tri is None or comp is None or track_ratio is None:
        return "halt", (f"soft gate could not be evaluated on hold {hold!r} — tripod={tri!r}, survival through the "
                        f"hold={comp!r} (by hold: {by_hold}), {tr_str}; campaign HALTED (inspect the eval log, then "
                        "relaunch with --continue to override, or fix and --redo)")
    if tri < tripod_min or comp < completion_min or not tr_ok:
        return "halt", (f"soft gate FAILED on hold {hold!r} — tripod {tri:.3f} (need >= {tripod_min}; by hold: {by_hold}), "
                        f"survival through the hold {comp:.3f} (need >= {completion_min}; whole-schedule completion "
                        f"{whole!r}), {tr_str} — campaign HALTED (relaunch with --continue to override, or fix and --redo)")
    return "ok", (f"soft gate passed on hold {hold!r} — tripod {tri:.3f} (by hold: {by_hold}), survival through the hold "
                  f"{comp:.3f} (whole-schedule completion {whole!r}), {tr_str}")


def gate_1c_verdict(summary: dict, *, track_ratio: float | None = None,
                    track_ratio_min: float = GATE_TRACK_RATIO_MIN) -> tuple[str, str]:
    """The written formation gate (Part H), enforced after 1c on ``slow__omni``.

    walking-hold tripod >= ``GATE_1C_TRIPOD`` on EACH moving hold (equivalent to the MIN over
    walking holds, ``walking_tripod()``'s definition) AND completion >= ``GATE_1C_COMPLETION`` AND
    the turning-rate criterion on both turn holds: the median SIGNED yaw-progress rate
    (``facing_by_hold[h]["yaw_progress_rate"]["median"]``, hoisted into ``yaw_progress_by_hold`` by
    :func:`summarize`) >= ``GATE_1C_YAW_RATE`` rad/s, AND a per-direction tracking floor (step-16
    review decision 3, 2026-09-17): median tracking ratio (``tracking_by_hold[h]["ratio"]["median"]``,
    the harness's ``speed_ratio``, hoisted into ``dir_ratio_by_hold`` by :func:`summarize`) >=
    ``GATE_1C_DIR_RATIO`` on EACH of ``GATE_1C_DIR_HOLDS`` (``lat_l, diag, back, move_turn`` -- ``fwd``
    is already constrained by the tripod/completion criteria above). No usable number on any of the
    four criteria is a halt, same as a numeric failure.

    ``yaw_progress_rate`` is the mean of ``wz * sign(wz_cmd)`` over the hold's commanded-turn steps:
    rotation TOWARD the command, in rad/s, so wobble cancels and a wrong-way turn is negative. The
    retired proxy (``yaw_rate_ratio x ang_vel_max``) divided two magnitudes and scored 0.946 -> 0.284
    rad/s on the 2026-09-17 yaw smoke, where the body's net rotation over the same ``turn_l90`` hold
    was -0.025 rad/s against a +0.30 rad/s command: it would have PASSED a head that does not turn.

    The per-direction floor covers a different failure the tripod/completion criteria miss entirely:
    a head that walks forward only (tripod stepping through the stop on every other hold without
    actually tracking it) could otherwise pass an omnidirectional formation gate. The message names
    each failing direction hold.

    FIFTH CRITERION (2026-09-18): the training log's own ``track_ratio_mean`` over the last
    ``GATE_TRACK_RATIO_N`` iterations must clear ``track_ratio_min``. The four eval criteria above
    are all measured on the canary, and the canary was itself defective for the whole seed-3
    campaign (the facing-obs clip; ANALYSIS claim 1); the training number is independent of the
    eval env entirely, costs nothing, and on the seed-3 chain it fell 0.747 -> 0.433 -> 0.245 while
    the gates were reading tripod. ``None`` halts.
    """
    tri, tri_by_hold = summary.get("tripod_walking"), (summary.get("tripod_by_hold") or {})
    comp = summary.get("completion")
    progress = summary.get("yaw_progress_by_hold") or {}
    net = summary.get("net_yaw_by_hold") or {}
    turn_rates = {h: progress.get(h) for h in TURN_HOLDS}
    dir_ratio = {h: (summary.get("dir_ratio_by_hold") or {}).get(h) for h in GATE_1C_DIR_HOLDS}
    by_hold = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in tri_by_hold.items()) or "none"
    rate_str = ", ".join(
        f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in turn_rates.items()
    ) + " rad/s (net " + ", ".join(
        f"{h} {net.get(h):.3f}" if net.get(h) is not None else f"{h} n/a" for h in TURN_HOLDS) + ")"
    dir_str = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in dir_ratio.items())
    failing_dirs = [h for h, v in dir_ratio.items() if v is None or v < GATE_1C_DIR_RATIO]
    tr_ok, tr_str = _track_ratio_clause(track_ratio, track_ratio_min)
    if (tri is None or comp is None or track_ratio is None
            or any(v is None for v in turn_rates.values()) or any(v is None for v in dir_ratio.values())):
        return "gate_1c", (f"formation gate could not be evaluated — walking-hold tripod={tri!r} (by hold: {by_hold}), "
                           f"completion={comp!r}, yaw-progress rate {rate_str}, per-direction speed_ratio {dir_str} "
                           f"(need >= {GATE_1C_DIR_RATIO} on each of {', '.join(GATE_1C_DIR_HOLDS)}) on slow__omni, "
                           f"{tr_str}; campaign HALTED (inspect the eval log, then relaunch with --continue to "
                           "override, or fix and --redo)")
    if (tri < GATE_1C_TRIPOD or comp < GATE_1C_COMPLETION or any(v < GATE_1C_YAW_RATE for v in turn_rates.values())
            or failing_dirs or not tr_ok):
        failing_str = ", ".join(failing_dirs) if failing_dirs else "none"
        return "gate_1c", (f"formation gate FAILED — walking-hold tripod {tri:.3f} (need >= {GATE_1C_TRIPOD}; by hold: "
                           f"{by_hold}), completion {comp:.3f} (need >= {GATE_1C_COMPLETION}), yaw-progress rate "
                           f"{rate_str} (need >= {GATE_1C_YAW_RATE} rad/s on each), per-direction speed_ratio {dir_str} "
                           f"(need >= {GATE_1C_DIR_RATIO} on each of {', '.join(GATE_1C_DIR_HOLDS)}; failing: {failing_str}) "
                           f"on slow__omni, {tr_str} — campaign HALTED (relaunch with --continue to override, or fix "
                           "and --redo)")
    return "ok", (f"formation gate passed — walking-hold tripod {tri:.3f} (by hold: {by_hold}), completion {comp:.3f}, "
                 f"yaw-progress rate {rate_str}, per-direction speed_ratio {dir_str} on slow__omni, {tr_str}")


def gate_facing_verdict(summary: dict) -> tuple[str, str]:
    """Facing-precision gate, WATCHED at 1c and ENFORCED from 2a onward (Part H): median
    ``heading_error_final_abs`` <= ``GATE_FACING_MAX_RAD`` on EACH turn hold, once the facing band
    has widened past 1c's +-0.6 rad. No usable number on either turn hold is a halt."""
    facing = summary.get("facing_by_hold") or {}
    finals = {h: (facing.get(h) or {}).get("final") for h in TURN_HOLDS}
    final_str = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in finals.items())
    if any(v is None for v in finals.values()):
        return "gate_facing", (f"facing-precision gate could not be evaluated — final heading error {final_str} on "
                               "slow__omni; campaign HALTED (inspect the eval log, then relaunch with --continue "
                               "to override, or fix and --redo)")
    if max(finals.values()) > GATE_FACING_MAX_RAD:
        return "gate_facing", (f"facing-precision gate FAILED — final heading error {final_str} (need <= "
                               f"{GATE_FACING_MAX_RAD} rad on each turn hold) on slow__omni — campaign HALTED "
                               "(relaunch with --continue to override, or fix and --redo)")
    return "ok", f"facing-precision gate passed — final heading error {final_str} on slow__omni"


def _apply_gate(c: Campaign, spec, rec: dict, tag: str) -> None:
    """Runs the phase-appropriate gate (if any) against ``rec["evals"]["slow"]`` and mutates
    ``rec["status"]`` on a halt, notifying and logging exactly like the sideways ``gate_1a`` site.

    Also computes the stage's TRAINING-log translation number once (2026-09-18) and records it on
    the phase record, so the REPORT and any later re-scoring read the same value the gate did.
    """
    if rec["status"] != "ok":
        return
    slow = rec["evals"].get("slow") or {}
    lp = Path(rec["log"])
    track_ratio = track_ratio_tail(track_ratio_series(lp)) if lp.exists() else None
    rec["track_ratio_tail"] = track_ratio
    if spec.name in SOFT_GATE_HOLD:
        status, msg = gate_soft_verdict(slow, SOFT_GATE_HOLD[spec.name], GATE_SOFT_TRIPOD,
                                        GATE_SOFT_COMPLETION, track_ratio=track_ratio)
        gate_status = f"gate_{spec.name}" if status != "ok" else "ok"
    elif spec.name == "1c":
        status, msg = gate_1c_verdict(slow, track_ratio=track_ratio)
        gate_status = status
    elif spec.name in FACING_GATE_PHASES:
        status, msg = gate_facing_verdict(slow)
        gate_status = status
    else:
        return
    if status == "ok":
        c.log(f"{tag}: {msg}")
    else:
        rec["status"] = gate_status
        c.notify(f"{tag}: {msg}")


# ------------------------------------------------------------------ pause / continue
def interrupted_phase(c: Campaign) -> dict | None:
    """The phase recorded in state['current'] (a run cut short by a stop): its newest checkpoint and
    the iterations left to reach the phase's original final iteration."""
    cur = c.st.get("current") or {}
    if not cur.get("phase") or not cur.get("log"):
        return None
    spec = ph.PHASES[cur["phase"]]
    log = Path(cur["log"])
    txt = log.read_text(errors="ignore") if log.exists() else ""
    m = re.search(r"Exact experiment name requested from command line: (\S+)", txt)
    run_dir = LOG_ROOT / spec.experiment / m.group(1) if m else None
    if run_dir is None or not run_dir.exists():
        c.log(f"--continue: no run directory found for the interrupted {cur['phase']} run ({log})")
        return None
    models = sorted(run_dir.glob("model_*.pt"), key=_ckpt_iter)
    if not models:
        return None
    target = cur.get("target_final")
    if target is None:
        hdr = re.search(r"Learning iteration (\d+)/(\d+)", txt)
        target = int(hdr.group(2)) - 1 if hdr else None
    if target is None:
        return None
    last = models[-1]
    remaining = int(target) - _ckpt_iter(str(last)) + 1
    if remaining <= 0:
        return None
    started = cur.get("started")
    wall_h = 0.0
    try:
        wall_h = (log.stat().st_mtime - datetime.fromisoformat(started).timestamp()) / 3600.0 if started else 0.0
    except Exception:
        pass
    resume = cur.get("resume")
    if resume is None:          # older state records: take the loaded checkpoint from the train log
        mm = re.search(r"Loading model checkpoint from: (\S+)", txt)
        resume = mm.group(1) if mm else None
    return {"phase": cur["phase"], "run_dir": str(run_dir), "ckpt": str(last), "remaining": remaining,
            "iters": cur.get("iters", spec.iterations), "resume": resume, "log": str(log), "wall_h": wall_h}


# ------------------------------------------------------------------ per-phase run
def run_phase(c: Campaign, spec, resume_ckpt: str | None, iters: int, do_eval: bool,
              from_scratch: bool = False) -> dict:
    c.st["run_no"] += 1
    tag = f"{c.key}_{spec.name}_{c.st['run_no']:03d}_{c.plant.replace('+', 'p')}" + (f"_it{iters}" if iters != spec.iterations else "")
    t0 = time.time()
    # dead_plant soundness applies ONLY to from-scratch windows >= SMOKE_ITERS (the sideways rule,
    # step-16 review decision 2, 2026-09-17): the episode-length-growth ratio it reads is biased on
    # a RESUMED window (the episode length is already long from the checkpoint it resumes) and the
    # status is not --continue-overridable, so gating it on a resumed window risks a false halt with
    # no escape hatch. No omni window qualifies: 1a is the only from-scratch window and it is 1000
    # its (< SMOKE_ITERS); every window from 1c on resumes the previous phase's head. Collapse there
    # is covered instead by the soft gates (1a/1b) and gate_1c (1c on).
    #
    # The translation early abort (2026-09-18, decision D) is armed on the same condition MINUS
    # the length rule -- any from-scratch window long enough to reach ABORT_AT_ITERS. Under
    # decision D the 1c window itself runs from scratch for 5000-8000 its, which is exactly the
    # run that must not spend 2.5-4 GPU-h on a creeper.
    ckpt, status, lp = train_retry(
        c, spec, tag, iters, resume_ckpt, soundness=(resume_ckpt is None and iters >= SMOKE_ITERS),
        early_abort=(resume_ckpt is None and iters > ABORT_AT_ITERS),
    )
    rec = {"phase": spec.name, "tag": tag, "task": spec.task, "mode": spec.teacher_mode, "ckpt": ckpt, "status": status,
           "resume": resume_ckpt, "from_scratch": bool(from_scratch), "iters": iters, "log": str(lp),
           "wall_h": (time.time() - t0) / 3600.0,
           "env": {"KRABBY_OMNI_PHASE": spec.name, "KRABBY_PLANT": c.plant}, "evals": {}}
    rec["smoke"] = smoke_fields(lp) if lp.exists() else {}
    rec["exposure"] = exposure_from_log(lp) if lp.exists() else {}
    # An aborted head is not evaluated: the abort exists to SAVE the ~40 min of eval rows and the
    # hours that would follow. The record still carries its checkpoint and its training numbers.
    if ckpt and do_eval and status == "ok":
        rec["evals"]["slow"] = eval_morph(c, spec, ckpt, "slow", tag)
        # SHORT-STAGE handling (Part H): 1a/1b run the flat canary only (~2 min instead of ~40) --
        # no light-ring or ring row -- and are checked by their own soft gate on that one hold.
        # Driven by ``spec.eval_scenarios`` (crab_hex_omni_phases.py's EVAL_SHORT / EVAL_FULL),
        # not a hardcoded phase-name check: the presets module is the source of truth for which
        # stages are short.
        scenarios = getattr(spec, "eval_scenarios", ("slow", "light", "ring"))
        if "light" in scenarios:
            rec["evals"]["light"] = eval_morph(c, spec, ckpt, "light", tag)
        if "ring" in scenarios:
            rec["evals"]["ring"] = eval_obst(c, spec, ckpt, tag)
        _apply_gate(c, spec, rec, tag)
    rec["checks"] = phase_checks(rec, spec, c)
    return rec


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--plant", default=ph.MAIN_PLANT, choices=sorted(ph.PLANTS),
                    help="named plant (default: the main asset = A15+B; legacy_golden for pre-2026-09-09 reproductions)")
    ap.add_argument("--seed", default="3")
    ap.add_argument("--phases", default="1a,1b,1c,2a,2b",
                    help="comma-separated phase names; 2c and 2c_clockheld both resume the 2b head and are "
                         "ALWAYS launched separately with --from-checkpoint (see the module docstring) -- "
                         "never include both in one --phases list")
    ap.add_argument("--from-checkpoint", default=None, help="resume the first listed phase from this checkpoint")
    ap.add_argument("--from-scratch", action="store_true",
                    help="start the FIRST listed phase from scratch, ignoring its spec's resume_from "
                         "(decision D, 2026-09-18: a from-scratch 1c). Recorded as resume "
                         "'scratch (--from-scratch)'; mutually exclusive with --from-checkpoint. "
                         f"Arms the {ABORT_AT_ITERS}-iteration dead_walker abort on that phase.")
    ap.add_argument("--iterations", type=int, default=None, help="override every phase's iteration count (smokes)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--pause-after", default=DEFAULT_PAUSE, help="phases after which to stop and wait ('' = none)")
    ap.add_argument("--redo", action="store_true", help="rerun phases already recorded ok for this seed")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="finish the interrupted phase recorded in state['current'] from its newest checkpoint "
                         "(remaining iterations only), then carry on with the listed phases; also overrides a "
                         "gate_1a/gate_1b/gate_1c/gate_facing/dead_walker halt (uses the recorded checkpoint "
                         "without retraining)")
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in ph.PHASES]
    if unknown:
        ap.error(f"unknown phases {unknown}; known: {list(ph.PHASES)}")
    if "2c" in phases and "2c_clockheld" in phases:
        ap.error("2c and 2c_clockheld both resume the 2b head; launch them separately (see the module docstring)")
    if args.from_scratch and args.from_checkpoint:
        ap.error("--from-scratch and --from-checkpoint contradict each other: pick one")
    pause_after = {p.strip() for p in args.pause_after.split(",") if p.strip()}
    c = Campaign(Path(args.campaign_dir).resolve(), args.plant, str(args.seed))
    c.dir.mkdir(parents=True, exist_ok=True)
    c.ensure_header(phases, args)
    c.st["phase"] = "run"
    c.save()
    _wait_isaac_clear()

    prev_ckpt = args.from_checkpoint
    partial = interrupted_phase(c) if args.cont else None
    for i, name in enumerate(phases):
        spec = ph.PHASES[name]
        rec = c.recs.get(name)
        if partial and partial["phase"] == name:
            # Continue the interrupted run: resume its newest checkpoint for the remaining iterations,
            # then record the phase as a whole (original resume point, full iteration count).
            c.log(f"{name}: continuing interrupted run {Path(partial['run_dir']).name} from "
                  f"{Path(partial['ckpt']).name} ({partial['remaining']} of {partial['iters']} iterations left)")
            rec = run_phase(c, spec, partial["ckpt"], partial["remaining"], do_eval=not args.no_eval)
            rec.update({"resume": partial["resume"], "iters": partial["iters"], "continued_from": partial["ckpt"],
                        "interrupted_log": partial["log"], "wall_h": rec["wall_h"] + partial.get("wall_h", 0.0)})
            rec["checks"] = phase_checks(rec, spec, c)
            c.recs[name] = rec
            c.st["current"] = {}
            c.save()
            c.report_block(f"## Phase {name} — {spec.task} — {c.key} — {datetime.now():%Y-%m-%d %H:%M} (continued after a pause)\n\n"
                           + "\n".join(phase_lines(rec, spec)))
            c.changelog_row(f"phase {name} {rec['status']} (continued from {Path(partial['ckpt']).name}): "
                            f"{Path(rec['ckpt']).name if rec.get('ckpt') else 'no checkpoint'} ({rec['tag']})")
            if rec["status"] != "ok":
                c.st["phase"] = f"halt:{name} {rec['status']}"
                c.save(); c.notify(f"phase {name} {rec['status']} — campaign HALTED (see REPORT)"); return 1
            prev_ckpt = rec["ckpt"]
            partial = None
            if name in pause_after and i < len(phases) - 1:
                c.st["phase"] = f"await_user:{name}"
                c.save()
                c.notify(f"phase {name} done ({Path(rec['ckpt']).name}); PAUSED before {phases[i + 1]} — relaunch with the same "
                         f"--phases to continue")
                return 0
            continue
        if rec and rec.get("ckpt") and not args.redo and not (i == 0 and (args.from_checkpoint or args.from_scratch)) and (
                rec.get("status") == "ok" or (rec.get("status") in OVERRIDABLE_STATUSES and args.cont)):
            if rec.get("status") in OVERRIDABLE_STATUSES:
                gate = rec["status"]
                nxt = phases[i + 1] if i + 1 < len(phases) else "nothing"
                stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
                c.log(f"{name}: {gate} recorded but --continue overrides — using {Path(rec['ckpt']).name} unretrained")
                # make the record self-describing: a later relaunch WITHOUT --continue must not retrain
                # this phase from scratch on top of a lineage that already resumed from its head
                rec["status"] = "ok"
                rec[f"{gate}_override"] = f"{stamp} --continue (see the CHARTER Log)"
                c.recs[name] = rec
                c.save()
                c.changelog_row(f"phase {name}: {gate} overridden by --continue — {Path(rec['ckpt']).name} carried into {nxt} "
                                f"unretrained (see CHARTER); record status -> ok")
                c.report_block(f"## Phase {name} — {gate} overridden — {stamp}\n\n- `--continue`: {Path(rec['ckpt']).name} "
                               f"carried into {nxt} unretrained (CHARTER: gate section + Log); the {name} record now "
                               f"reads `status ok` with `{gate}_override`")
            else:
                c.log(f"{name}: already recorded ok ({Path(rec['ckpt']).name}) — skipping")
            prev_ckpt = rec["ckpt"]
            continue
        # --from-scratch applies to the FIRST listed phase only: the rest of the chain resumes the
        # phase before it as usual (a cold restart mid-chain is a different campaign, not a flag).
        cold = bool(args.from_scratch and i == 0)
        resume = None if cold else (prev_ckpt if (i > 0 or args.from_checkpoint) else None)
        if resume is None and spec.resume_from and not cold:
            resume = (c.recs.get(spec.resume_from) or {}).get("ckpt")
            if not resume:
                c.st["phase"] = f"halt:{name} needs the {spec.resume_from} head (pass --from-checkpoint)"
                c.save(); c.notify(c.st["phase"]); return 2
        if cold:
            c.log(f"{name}: --from-scratch — ignoring the spec's resume_from "
                  f"({spec.resume_from or 'none'}); training from random init")
        iters = args.iterations or spec.iterations
        rec = run_phase(c, spec, resume, iters, do_eval=not args.no_eval, from_scratch=cold)
        c.recs[name] = rec
        c.st["current"] = {}
        c.save()
        c.report_block(f"## Phase {name} — {spec.task}" + (f" (mode {spec.teacher_mode})" if spec.teacher_mode else "") +
                       f" — {c.key} — {datetime.now():%Y-%m-%d %H:%M}\n\n" + "\n".join(phase_lines(rec, spec)))
        c.changelog_row(f"phase {name} {rec['status']}: {Path(rec['ckpt']).name if rec.get('ckpt') else 'no checkpoint'} ({rec['tag']})")
        if rec["status"] != "ok":
            c.st["phase"] = f"halt:{name} {rec['status']}"
            c.save(); c.notify(f"phase {name} {rec['status']} — campaign HALTED (see REPORT)"); return 1
        prev_ckpt = rec["ckpt"]
        if name in pause_after and i < len(phases) - 1:
            c.st["phase"] = f"await_user:{name}"
            c.save()
            c.notify(f"phase {name} done ({Path(rec['ckpt']).name}); PAUSED before {phases[i + 1]} — relaunch with the same "
                     f"--phases to continue")
            return 0
    c.st["phase"] = "done"
    c.save()
    c.notify(f"phases {', '.join(phases)} complete for {c.key} — see REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
