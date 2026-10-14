#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``), PLAN step 2 -- the height DAgger trainer's TORCH-FREE
specification: constants, the per-episode start-mode / command schedule, the stand / episode record readers, the
step-6 premise read and the step-5 smoke list.

Imported BEFORE Kit by ``sh_dagger.py`` (and by ``sh_dagger.sh`` through ``sh_dagger.py --emit-env``): standard library
only at import (map risk L6). It loads two torch-free modules by path (no bytecode):

* the closed predecessor's spec ``2026-10-11_stand_network/scripts/sn_dagger_spec.py`` (:data:`PRED`): the process
  environment (the ``P2__S1_s18`` overlay verbatim + the four D2 / D5 changes), the fail-closed env / obs-parity /
  network checks, beta, the DART sigma table, the control share, the optimizer / reservoir numbers, the first-draw
  scoring (OPEN-7 amendment) -- REUSED, not re-stated, so the env this trainer runs is the A-mask s17 trainer's env
  plus only the campaign-local Python composition (``sh_starts.py``: start modes, the table, the command / push /
  termination wrappers), which touches no ``KRABBY_*`` variable;
* this campaign's ``sh_core.py`` (:data:`SH`): the frozen h range, the restore constants, the start kinds.

DECISIONS RECORDED HERE (PLAN H5 leaves the number to the builder; log in CHARTER section 3 before any read):

* start-mode shares (H5 "~"; re-scoped at the step-4 stop, CHARTER section 3 "narrow stand-up to tall tables"): walk
  0.50 / table 0.40 / ground 0 / random-height 0.10, one draw per episode from a private generator at the reset event
  (the ground's 15 % went to table starts, so the stand-up dose is unchanged; "ground" stays a start kind code so the
  indices do not shift -- a share of 0 is never drawn, and the smoke asserts it ABSENT);
* retracted poses on table starts: knee_folded 0.75, rod_min 0, a random fold 0.25 (the plan's 0.60 / 0.20 / 0.20
  renormalised without rod_min; the facts.json ``random_fold_family`` 9 x 9 grid, uniform over its cells; on a table
  only cells whose legs clear the floor by 2 cm at that top -- facts.json "rule_for_tables" -- with knee_folded as the
  fallback, counted);
* table tops U[top_min, top_max] = U[0.635, 0.805] (facts.json ``frozen``: top_min re-frozen 0.412 -> 0.635 at the
  step-4 stop; top_max = min(frozen, h_max - 0.264), so an h_max re-freeze drops it -- :func:`top_max_for`);
* h_cmd per stand draw (every command resample, so a stand->stand merge redraws): 1.064 with probability 0.35, else
  U[h_min, h_max]; while the table is under the robot (not yet parked) the lower bound is max(h_min, top + 0.264) and
  the 1.064 atom applies only when 1.064 >= that bound; a GROUND episode's first (forced) command is bounded below by
  max(h_min, h_spawn + 0.14625) (K3's ground rule, review 2026-10-13: a fold that spawns high still rises >= 0.10 m;
  knee_folded / rod_min spawns keep h_min);
* the FIRST command of a table / ground / random-height episode is a stand (campaign-local wrapper on
  ``_resample_command``) of ``|h_cmd - clamp(h_spawn)| / 0.08 + 4.0 s settle + 3.0 s + 0.5 s margin``; the 4.0 s
  "settle" covers the actuator slew out of the retracted pose (the predecessor's zero-action table timeline: hips at
  their default after ~3.3 s) -- the margin absorbs the analytic-vs-measured spawn height (<= 4 cm);
* walk-handover episodes init the limiter at 1.064 (the walk rows' own command; the robot spawns at the default
  pose), so a stand drawn at spawn ramps like a handover (sh_core allows any command there; this is the choice);
* "up" (pushes allowed) and the hip-contact re-arm: the chassis' lowest point >= 0.10 m above its SPAWN value
  (latched); walk and random-height starts are up from spawn; the hip-contact clause is disarmed on ground starts
  only (PLAN H5); table / ground episodes that end without ever latching "up" are counted (env_counts.json
  ``never_up_*`` / ``ended_*``);
* the table is parked (moved 20 m below the floor) at the first check after its forced first command expired, once
  the chassis' lowest point is >= 0.05 m above the table top (PLAN "park only once the belly is clear").

TRAIN-TIME READS (the step-6 premise; not a gate): per DRAWN start kind (share > 0: walk / table / random-height; the
share-0 ground kind is not read, its records -- none expected -- only counted) x executor (student / beta-R / the 10 %
R_h control slice) -- homed share and |h_meas - h_cmd| on height holds, the STOOD (>= 4 feet) share on the first stand
of table starts (its rise clause as K3's: the ROOT risen >= 0.10 m above its spawn root z on >= 95 % of the window),
falls, restore completions / timeouts (from the per-iteration counts).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
#: scripts -> 2026-10-13_stand_height -> experiments -> crab_hex_omni_task
TASK = HERE.parents[2]
REPO = HERE.parents[6]
PRED_SCRIPTS = TASK / "experiments" / "2026-10-11_stand_network" / "scripts"


def _load(path: Path, name: str):
    """A private module instance by path (not in ``sys.modules``; no bytecode)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_dagger_spec: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the predecessor's torch-free spec (env composition, schedules, first-draw scoring) -- reused, never edited
PRED = _load(PRED_SCRIPTS / "sn_dagger_spec.py", "_sh_spec_pred")
#: this campaign's core (torch-free at import)
SH = _load(HERE / "sh_core.py", "_sh_spec_core")

# ===================================================================================== reused verbatim (predecessor)
TASK_ID, PRESET, PLANT = PRED.TASK_ID, PRED.PRESET, PRED.PLANT
S1_CKPT, S1_SHA256, S1_AGENT_YAML = PRED.S1_CKPT, PRED.S1_SHA256, PRED.S1_AGENT_YAML
REFS_ENV_STAMP, CANDIDATES, OVERLAY_CHAIN = PRED.REFS_ENV_STAMP, PRED.CANDIDATES, PRED.OVERLAY_CHAIN
S1_OVERLAY, POOL, SLOTS, RESAMPLE_S = PRED.S1_OVERLAY, PRED.POOL, PRED.SLOTS, PRED.RESAMPLE_S
OVERLAY_CHANGES, DRIVER_ENV, PROCESS_ENV = PRED.OVERLAY_CHANGES, PRED.DRIVER_ENV, PRED.PROCESS_ENV
FORBIDDEN_ENV, FORBIDDEN_PREFIXES = PRED.FORBIDDEN_ENV, PRED.FORBIDDEN_PREFIXES
expected_krabby_env, train_env, env_reasons = PRED.expected_krabby_env, PRED.train_env, PRED.env_reasons
EVAL_CARRY, EVAL_CARRY_DEFAULTS, obs_parity_reasons = PRED.EVAL_CARRY, PRED.EVAL_CARRY_DEFAULTS, PRED.obs_parity_reasons
network_cfg_reasons, path_reasons, read_stands = PRED.network_cfg_reasons, PRED.path_reasons, PRED.read_stands
beta, sigma_of, SIGMA_TABLE, CONTROL_SHARE = PRED.beta, PRED.sigma_of, PRED.SIGMA_TABLE, PRED.CONTROL_SHARE
LR, STEPS_PER_ITER, MINIBATCHES, MINIBATCH = PRED.LR, PRED.STEPS_PER_ITER, PRED.MINIBATCHES, PRED.MINIBATCH
BUFFER_GB, NUM_ENVS, MAX_GRAD_NORM, CKPT_EVERY = PRED.BUFFER_GB, PRED.NUM_ENVS, PRED.MAX_GRAD_NORM, PRED.CKPT_EVERY
ONSET_SHARE, W_MAX = PRED.ONSET_SHARE, PRED.W_MAX
HOMED_EPS, TRIM_BAND, EXIT_FALL_WINDOW_S = PRED.HOMED_EPS, PRED.TRIM_BAND, PRED.EXIT_FALL_WINDOW_S
PREMISE_MIN_SCHED_S, PREMISE_RATIO = PRED.PREMISE_MIN_SCHED_S, PRED.PREMISE_RATIO
EX_STUDENT, EX_BETA_R, EX_CONTROL, EXECUTORS = PRED.EX_STUDENT, PRED.EX_BETA_R, PRED.EX_CONTROL, PRED.EXECUTORS
scored, premise_scored, homed_scored, fall_scored = PRED.scored, PRED.premise_scored, PRED.homed_scored, PRED.fall_scored
END0_SCORED = PRED.END0_SCORED

DEFAULT_RUNS = REPO / "parkour/outputs/omni_plan/stand_height/runs"
MODES = ("dagger",)
#: PLAN H4: one arm (mask 1, the adopted A-mask s17's), three seeds
ARMS = {"mask": 1}
STAND_MASK = 1

# ===================================================================================== the height channel (sh_core)
H_SETTLED, H_MIN, H_MAX = SH.H_SETTLED, SH.H_MIN, SH.H_MAX
HEIGHT_RATE_M_S = SH.HEIGHT_RATE_M_S
HARNESS_CLIP = SH.HARNESS_CLIP
FROZEN = SH.FROZEN
#: H5: table starts draw h_cmd >= top + this (the restore / reachability margin of step 1(c))
TABLE_CMD_MARGIN_M = 0.264
#: a ground episode's FIRST (forced) command >= max(h_min, h_spawn + this): K3's ground rule (sh_table_core.cmd_range,
#: 0.31 - 0.15875 - 0.005), so a fold that spawns high still rises >= 0.10 m (the "up" / hip re-arm latch)
GROUND_FIRST_CMD_RISE_M = 0.31 - 0.15875 - 0.005


def top_max_for(h_max: float, frozen_top_max: float) -> float:
    """PLAN "drop training top_max with any h_max re-freeze": the training top_max is min(the frozen value (the
    restore-clearance bound 0.805), h_max - 0.264) -- recomputed from the CURRENT h_max, so a step-4 re-freeze that
    lowers h_max in facts.json lowers the table tops with it even if top_max_train_m is not rewritten."""
    return min(float(frozen_top_max), float(h_max) - TABLE_CMD_MARGIN_M)


TOP_MIN_M = float(FROZEN["top_min_m"])
TOP_MAX_M = top_max_for(H_MAX, FROZEN["top_max_train_m"])
if not TOP_MIN_M < TOP_MAX_M:
    raise RuntimeError(f"sh_dagger_spec: table tops [{TOP_MIN_M}, {TOP_MAX_M}] empty at h_max {H_MAX}")
H_CMD_ATOM = SH.H_SETTLED
H_CMD_ATOM_P = 0.35

# ===================================================================================== start modes (H5)
START_KINDS = SH.START_KINDS                       # ("walk", "table", "ground", "random"); index = int code
SK_WALK, SK_TABLE, SK_GROUND, SK_RANDOM = 0, 1, 2, 3
#: CHARTER section 3, step-4 decision: ground starts removed (share 0, never drawn; the kind code stays so the indices
#: do not shift), their 15 % to table starts
START_SHARES = (0.50, 0.40, 0.0, 0.10)
#: retracted poses (table / ground starts): index 0 knee_folded, 1 rod_min, 2 a random fold of the facts grid; rod_min
#: share 0 (never drawn) since the step-4 decision
POSES = ("knee_folded", "rod_min", "random_fold")
POSE_SHARES = (0.75, 0.0, 0.25)
#: the start kinds / poses that are drawn (share > 0); a share-0 one is asserted ABSENT by the smoke
DRAWN_START_KINDS = tuple(k for k, p in zip(START_KINDS, START_SHARES) if p > 0)
DRAWN_POSES = tuple(k for k, p in zip(POSES, POSE_SHARES) if p > 0)
RANDOM_FOLD_GRID = 9                               #: facts.json random_fold_family: a 9 x 9 grid
RANDOM_FOLD_HIP_MAX_DEG = -30.0                    #: the family's least-raised femur (facts.json definition)
FIRST_STAND_SETTLE_S = 4.0
FIRST_STAND_HOLD_S = 3.0
FIRST_STAND_MARGIN_S = 0.5
#: "up" / hip-contact re-arm (sh_starts.post_physics): the chassis' lowest point risen this far above its spawn value;
#: train-time STOOD (and K3's): the ROOT risen this far above its spawn root z on >= 95 % of the window
UP_RISE_M = 0.10
TABLE_PARK_CLEAR_M = 0.05
#: the table: one tall kinematic cuboid per env (x along the robot's body x), its top set by z placement
TABLE_SIZE_M = (0.90, 0.70, 2.0)
TABLE_PARK_Z_M = -20.0
SPAWN_CLEARANCE_M = 0.005
TOE_FLOOR_CLEAR_M = 0.02
# train-time STOOD (K3's statistic, on the first stand's first draw)
STOOD_MIN_FEET = 4
STOOD_WINDOW_S = 3.0
STOOD_FEET_SHARE = 0.95
STOOD_DH_M = 0.04
FOOT_LOADED_N = 20.0
FOOT_FLOOR_M = 0.15
#: |h - h_cmd| is read on holds whose first draw lasts at least the ramp + this
HOLD_READ_AFTER_RAMP_S = 2.0
#: step 6 (PLAN): fall / restore-timeout stop margins
PREMISE_FALL_FACTOR, PREMISE_FALL_ADD = 2.0, 0.01


def first_stand_seconds(h_cmd: float, h_init: float, *, margin: bool = True) -> float:
    """PLAN H5: ramp + settle + 3 s (+ the draw margin) for a first stand from ``h_init`` to ``h_cmd``."""
    s = abs(float(h_cmd) - float(h_init)) / HEIGHT_RATE_M_S + FIRST_STAND_SETTLE_S + FIRST_STAND_HOLD_S
    return s + (FIRST_STAND_MARGIN_S if margin else 0.0)


def table_cmd_bounds(top: float) -> tuple[float, float, bool]:
    """``(lo, hi, atom_allowed)`` of the h_cmd draw while a table of ``top`` (m above the floor) is under the robot."""
    lo = max(H_MIN, float(top) + TABLE_CMD_MARGIN_M)
    return lo, H_MAX, bool(H_CMD_ATOM >= lo - 1e-12)


# ===================================================================================== cfg asserts (fail closed)
def _get(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    return cur


def _name(fn) -> str:
    return getattr(fn, "__name__", None) or type(fn).__name__


def cfg_reasons(env_cfg, agent_cfg) -> list[str]:
    """The predecessor's checks (``PRED.cfg_reasons``: sampler, pool / SLOTS, 40-s episodes, promotion, crab_failure
    thresholds, obs params, action delay, no RSI, runner clip 1.0 in the CONFIG, soft std) plus what THIS trainer's
    semantics rest on: the action term's raw clips (rods +-4.8, cams +-1 -- else the posture offset q would be clipped
    silently), its scales (0.25 rad / unit on the rods), the 0.02-s control step, the push event and the
    ``crab_failure`` / ``total_terminates`` terms the wrappers wrap, the omni command class (subclassed)."""
    out = list(PRED.cfg_reasons(env_cfg, agent_cfg))

    def get(path):
        try:
            return _get(env_cfg, path)
        except (AttributeError, KeyError, TypeError):
            return "<missing>"

    clip = get("actions.joint_pos.clip")
    raw = float(SH.RAW_CLIP)
    want = {".*_Body_CamShaft_RevoluteJoint": (-1.0, 1.0), ".*_Hip_Femur_RevoluteJoint": (-raw, raw),
            ".*_Femur_Tibia_RevoluteJoint": (-raw, raw)}
    if not isinstance(clip, dict) or {k: tuple(float(x) for x in v) for k, v in clip.items()} != want:
        out.append(f"actions.joint_pos.clip = {clip!r}, want {want!r} (the raw +-4.8 rod clip q rests on)")
    scale = get("actions.joint_pos.scale")
    want_s = {".*_Body_CamShaft_RevoluteJoint": math.pi, ".*_Hip_Femur_RevoluteJoint": 0.25,
              ".*_Femur_Tibia_RevoluteJoint": 0.25}
    if not isinstance(scale, dict) or {k: float(v) for k, v in scale.items()} != want_s:
        out.append(f"actions.joint_pos.scale = {scale!r}, want {want_s!r}")
    try:
        step = float(get("sim.dt")) * int(get("decimation"))
    except (TypeError, ValueError):
        step = None
    if step is None or abs(step - 0.02) > 1e-12:
        out.append(f"control step sim.dt x decimation = {step!r}, want 0.02 s")
    push = get("events.push_by_setting_velocity")
    if push == "<missing>" or push is None or _name(getattr(push, "func", None)) != "push_by_setting_velocity" \
            or getattr(push, "mode", None) != "interval":
        out.append("events.push_by_setting_velocity: want the interval push_by_setting_velocity term (wrapped)")
    cf = get("terminations.crab_failure")
    if cf == "<missing>" or cf is None or _name(getattr(cf, "func", None)) != "terminate_crab_hex_failure":
        out.append("terminations.crab_failure: want terminate_crab_hex_failure (wrapped: hip clause per env)")
    tt = get("terminations.total_terminates")
    if tt == "<missing>" or tt is None:
        out.append("terminations.total_terminates missing (roll / pitch / height / time-out cutoffs stay)")
    ct = get("commands.base_velocity.class_type")
    if getattr(ct, "__name__", None) != "CrabHexOmniCommand":
        out.append(f"commands.base_velocity.class_type = {ct!r}, want CrabHexOmniCommand (subclassed here)")
    ev = get("events")
    if ev != "<missing>" and getattr(ev, "sh_start", None) is not None:
        out.append("events.sh_start already present (compose once)")
    sc = get("scene")
    if sc != "<missing>" and getattr(sc, "sh_table", None) is not None:
        out.append("scene.sh_table already present (compose once)")
    return out


# ===================================================================================== stand records
#: the predecessor's fields + the height / start-mode fields this tracker adds (``stands.jsonl``)
RECORD_KEYS = tuple(PRED.RECORD_KEYS) + (
    "start_kind", "pose", "table_top", "first_stand", "h_cmd0", "h_init0", "h_meas_end0", "abs_dh0", "feet_end0",
    "win_n0", "win_feet4_0", "rise_min_win0", "win_rise0", "stood0", "ramp0_s", "dt")
#: one record per finished episode (``episodes.jsonl``)
EPISODE_KEYS = ("it", "env", "kind", "pose", "table_top", "length_s", "end", "cause", "run_start", "executor",
                "sigma", "first_len_s", "change_s", "park_s", "rearm_s", "up_s", "clear_at_first_check",
                "h_spawn", "spawn_root_z", "h_meas_spawn", "h_init")


def stood_scored(rec: dict) -> bool:
    """A table / ground FIRST stand that counts for the train-time STOOD share: sigma 0, first draw not timeout-cut,
    not a run-start stand, ended by reaching its first draw's end or by a fall."""
    return (bool(rec.get("first_stand")) and rec.get("start_kind") in ("table", "ground")
            and float(rec["sigma"]) == 0.0 and not bool(rec["cut0"]) and rec.get("entry_kind") != "run_start"
            and rec["end0"] in END0_SCORED)


def stood_outcome(rec: dict) -> bool:
    return rec["end0"] == "reached" and bool(rec.get("stood0"))


def hold_dh_scored(rec: dict) -> bool:
    """A hold whose |h - h_cmd| is read: sigma 0, scored on its first draw (reached, not a fall), and the first draw
    lasts at least the ramp from the stand's starting limiter state + HOLD_READ_AFTER_RAMP_S."""
    if float(rec["sigma"]) != 0.0 or rec["end0"] != "reached" or bool(rec["cut0"]) \
            or rec.get("entry_kind") == "run_start" or rec.get("abs_dh0") is None:
        return False
    return float(rec["sched_len0_s"]) >= float(rec.get("ramp0_s") or 0.0) + HOLD_READ_AFTER_RAMP_S - 1e-9


def fall_scored_set(recs: list[dict]) -> list[dict]:
    """The stands a fall share is read on: sigma 0 (as the homed / STOOD reads; DART noise is not the student's), scored
    on the first draw (``scored(r, 0)``: not timeout-cut, not a run-start stand, reached or fell)."""
    return [r for r in recs if float(r["sigma"]) == 0.0 and scored(r, 0.0)]


def _share(k: int, n: int) -> dict:
    p = (k / n) if n else None
    se = math.sqrt(p * (1 - p) / n) if n else None
    return {"k": int(k), "n": int(n), "p": p, "se": se}


def _quant(vals: list, q: float):
    if not vals:
        return None
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


RESTORE_EVENTS = ("engaged", "completed", "timed_out", "merged", "reset_cut")
#: restore counts are split by the episode's DART group: R_h's own restores time out under sigma noise (the rods
#: chase noisy targets past the 4-mm completion band), so the premise / smoke compare sigma-0 episodes only (the
#: control slice is sigma 0 by construction) -- the same rule the predecessor's homed premise used for stands
RESTORE_SIGMA = ("sigma0", "sigma_pos")
#: step-5 smoke: absolute ceiling on the R_h control slice's own sigma-0 restore-timeout share (re-review of steps
#: 2-3); with < 20 ended control restores at most one may time out
RESTORE_CTL_TIMEOUT_MAX_SHARE = 0.05


def restore_totals(log_rows: list[dict], it_lo: int = 0, it_hi: int | None = None) -> dict:
    """Sum the per-iteration restore counts (``train_log.jsonl`` ``restore`` blocks:
    ``{executor: {sigma group: {start_kind: {event: n}}}}``) over iterations ``[it_lo, it_hi)``."""
    out = {ex: {sg: {k: {e: 0 for e in RESTORE_EVENTS} for k in START_KINDS} for sg in RESTORE_SIGMA}
           for ex in EXECUTORS}
    for row in log_rows:
        it = int(row.get("it", -1))
        if it < int(it_lo) or (it_hi is not None and it >= int(it_hi)):
            continue
        for ex, d in (row.get("restore") or {}).items():
            for sg, dk in d.items():
                for k, ev in dk.items():
                    for e, n in ev.items():
                        out[ex][sg][k][e] += int(n)
    return out


def restore_pooled(rt: dict, sigma: str | None = "sigma0") -> dict:
    """``{executor: {event: n}}`` pooled over start kinds (and over both sigma groups when ``sigma`` is None)."""
    sgs = RESTORE_SIGMA if sigma is None else (sigma,)
    return {ex: {e: sum(rt[ex][sg][k][e] for sg in sgs for k in START_KINDS) for e in RESTORE_EVENTS}
            for ex in EXECUTORS}


def timeout_share(ev: dict) -> dict:
    """Restores that ended by the timeout over restores that ended by completion or timeout."""
    return _share(int(ev["timed_out"]), int(ev["timed_out"]) + int(ev["completed"]))


#: the step-6 premise's per-kind checks (a kind not drawn -- share 0 -- has no check)
PREMISE_KIND_STATS = (("walk", "homed"), ("random", "homed"), ("table", "stood"), ("ground", "stood"))


def premise_read(records: list[dict], log_rows: list[dict] | None = None, it_lo: int = 0,
                 it_hi: int | None = None) -> dict:
    """PLAN step 6 mid-run premise read (train-time, closed loop), student vs the concurrent R_h control slice (same
    iterations / envs / terrain levels), per start kind; stands attributed to the iteration they END in. STOP (ask) if
    the student is < 0.5 x R_h on the homed share (walk / random-height holds) or on the STOOD share (table / ground
    first stands), its fall share > 2 x R_h's + 0.01 (sigma-0 stands, every start kind pooled), or its restore-timeout
    share > 2 x R_h's + 0.01 (sigma-0 episodes, every start kind pooled; :data:`RESTORE_SIGMA`). A ratio with an empty side is
    None (no verdict from it). Only the DRAWN start kinds (:data:`DRAWN_START_KINDS`, share > 0) are read and checked:
    since the step-4 decision there is no ground start type (share 0); records of a share-0 kind (none expected; the
    smoke asserts the kind absent) are counted in ``absent_kinds``, never read, never a STOP."""
    sel = [r for r in records if int(r["it_end"]) >= int(it_lo) and (it_hi is None or int(r["it_end"]) < int(it_hi))]
    out: dict = {"window": [int(it_lo), None if it_hi is None else int(it_hi)], "by_kind": {},
                 "absent_kinds": {k: {"share": p, "records": sum(1 for r in sel if r.get("start_kind") == k)}
                                  for k, p in zip(START_KINDS, START_SHARES) if k not in DRAWN_START_KINDS}}
    for kind in DRAWN_START_KINDS:
        blk = {}
        for ex in EXECUTORS:
            g = [r for r in sel if r.get("start_kind") == kind and r["kind"] == ex]
            hold = [r for r in g if premise_scored(r)]
            dh = [float(r["abs_dh0"]) for r in g if hold_dh_scored(r)]
            st = [r for r in g if stood_scored(r)]
            sc = fall_scored_set(g)
            blk[ex] = {"homed": _share(sum(1 for r in hold if homed_scored(r)), len(hold)),
                       "abs_dh": {"n": len(dh), "median": _quant(dh, 0.5), "p90": _quant(dh, 0.9)},
                       "stood": _share(sum(1 for r in st if stood_outcome(r)), len(st)),
                       "fall": _share(sum(1 for r in sc if fall_scored(r)), len(sc))}
        out["by_kind"][kind] = blk
    rt = restore_totals(log_rows or [], it_lo, it_hi)
    out["restore"] = {ex: {sg: {k: dict(v, timeout_share=timeout_share(v)) for k, v in rt[ex][sg].items()}
                           for sg in RESTORE_SIGMA} for ex in EXECUTORS}
    out["restore_pooled"] = {ex: dict(v, timeout_share=timeout_share(v)) for ex, v in restore_pooled(rt).items()}
    out["restore_pooled_all_sigma"] = {ex: dict(v, timeout_share=timeout_share(v))
                                       for ex, v in restore_pooled(rt, None).items()}
    reasons, checks = [], []

    def ratio(s, c):
        return (s / c) if (s is not None and c) else None

    for kind, stat in PREMISE_KIND_STATS:
        if kind not in DRAWN_START_KINDS:
            continue
        s = out["by_kind"][kind]["student"][stat]["p"]
        c = out["by_kind"][kind]["control"][stat]["p"]
        r = ratio(s, c)
        checks.append({"check": f"{kind} {stat} student >= 0.5 x R_h", "student": s, "control": c, "ratio": r,
                       "stop": None if r is None else bool(r < PREMISE_RATIO)})
    fs = [sum(out["by_kind"][k][ex]["fall"]["k"] for k in DRAWN_START_KINDS) for ex in ("student", "control")]
    fn = [sum(out["by_kind"][k][ex]["fall"]["n"] for k in DRAWN_START_KINDS) for ex in ("student", "control")]
    pf = [(k / n) if n else None for k, n in zip(fs, fn)]
    checks.append({"check": "fall share student <= 2 x R_h + 0.01", "student": pf[0], "control": pf[1],
                   "stop": None if None in pf else bool(pf[0] > PREMISE_FALL_FACTOR * pf[1] + PREMISE_FALL_ADD)})
    ts = out["restore_pooled"]["student"]["timeout_share"]["p"]
    tc = out["restore_pooled"]["control"]["timeout_share"]["p"]
    checks.append({"check": "restore-timeout share (sigma-0 episodes) student <= 2 x R_h + 0.01", "student": ts,
                   "control": tc,
                   "stop": None if None in (ts, tc) else bool(ts > PREMISE_FALL_FACTOR * tc + PREMISE_FALL_ADD)})
    reasons = [c["check"] for c in checks if c["stop"]]
    out["checks"] = checks
    out["stop"] = bool(reasons) if any(c["stop"] is not None for c in checks) else None
    out["stop_reasons"] = reasons
    return out


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ===================================================================================== the step-5 smoke list
#: PLAN step 5 (each with the PLAN text it asserts) + the predecessor's generic checks that still apply
SMOKE_DAGGER = (
    ("start_mode_shares", "start-mode shares (walk 0.50 / table 0.40 / random-height 0.10; ground share 0 since the "
                          "step-4 decision: asserted ABSENT, 0 draws)"),
    ("retracted_pose_shares", "retracted poses: knee_folded mostly, plus random folds (rod_min share 0 since the "
                              "step-4 decision: asserted ABSENT, 0 draws)"),
    ("table_tops", "table tops U[top_min, top_max] and the realised (PhysX) top at its draw"),
    ("ground_spawns", "non-walk spawns (table / random; ground share 0: asserted ABSENT): measured spawn height = "
                      "the analytic one; below h_min clamps"),
    ("h_cmd_distribution", "h_cmd ~ U[h_min, h_max] with 35 % mass at 1.064; table starts >= top + 0.264"),
    ("limiter", "the 0.08 m/s limiter: per-step change, range, init per start kind (walk 1.064, else clamp(h_meas))"),
    ("residual_labels_within_1", "residual labels |R_h - q| <= 1"),
    ("restores_complete", "restores complete (the R_h control slice); timeouts relative to the R_h slice (left for "
                          "implementation): the beta-R slice gated, the student reported (gated at step 6)"),
    ("table_first_stand", "table first stand >= ramp + settle + 3 s with h_cmd >= top + 0.264 m"),
    ("forced_first_stand_ground_random", "random-height first stand >= ramp + settle + 3 s (ground share 0: "
                                         "asserted ABSENT)"),
    ("table_parked_at_slot_change", "table parked at the slot change, only once the belly is clear"),
    ("pushes_off_until_up", "pushes off until the robot is up"),
    ("ground_episodes_hip_rearm", "ground episodes last >= ramp + settle + 3 s with 0 hip-contact terminations "
                                  "before re-arm (ground share 0 since the step-4 decision: 0 ground episodes, 0 "
                                  "disarmed hip terminations)"),
    ("restore_rows_stand_net_inputs", "restore rows carry zeroed command columns (5, 8-10) and col 15 = h_lim"),
    ("restore_rows_in_reservoir_with_rh_labels", "restore rows enter the reservoir with R_h labels"),
    ("s1_rows_pm1", "S1 rows at +-1"),
    ("stand_hip_targets_below_minus1", "realised hip targets < -1 unit on stand rows"),
    ("harness_clip_4p8", "the wrapper clip realised at 4.8 (runtime), the config's 1.0 untouched"),
    ("forward_parity", "trainer <-> launcher forward parity (same device, same batch)"),
    ("student_deadband_realised", "the residual deadband (CHARTER 2026-10-14; --deadband d > 0 only, else not "
                                  "applied): no realised student hip / knee residual inside (0, d)"),
    ("warm_start_zeroed", "warm start = A-mask s17 (sha) with W[:, 15] zeroed"),
    ("handovers", "walk -> stand handovers and stand exits happen"),
    ("branch_shares_student_rows", "branch shares on student-visited rows"),
    ("lift_onset_after_balancing", "LIFT-onset >= 1 % after balancing"),
    ("beta_schedule", "the beta schedule function (asserted, not reached)"),
    ("dart_sigma_realized", "DART sigma realized (on the residual)"),
    ("obs_parity_env", "the trainer env carries every obs-affecting var of the eval stamp"),
    ("obs_parity_cols", "raw obs cols 15-16 (+ history copies) == 0 (the height is written on the clone only)"),
    ("s1_deterministic", "S1 runs deterministic act_inference"),
    ("buffer_projection", "the pre-allocated reservoir <= the cap"),
    ("ckpt_loads", "saved ckpt loads strictly, estimator == S1's, infos carry the height channel"),
    ("terrain_level_recorded", "every stand record carries its terrain level"),
)


def smoke_verdict(checks: list[dict]) -> dict:
    """Every listed check present exactly once with a boolean ``pass`` (missing / extra / duplicated -> FAIL)."""
    want = [n for n, _ in SMOKE_DAGGER]
    got = [c.get("name") for c in checks]
    problems = []
    missing = [n for n in want if n not in got]
    extra = [n for n in got if n not in want]
    dup = sorted({n for n in got if got.count(n) > 1})
    if missing:
        problems.append(f"missing checks {missing}")
    if extra:
        problems.append(f"unlisted checks {extra}")
    if dup:
        problems.append(f"duplicated checks {dup}")
    bad = [c.get("name") for c in checks if not isinstance(c.get("pass"), bool)]
    if bad:
        problems.append(f"checks without a boolean verdict {bad}")
    failed = [c["name"] for c in checks if c.get("pass") is False]
    ok = not problems and not failed
    return {"kind": "dagger", "verdict": "PASS" if ok else "FAIL", "problems": problems, "failed": failed,
            "plan_text": dict(SMOKE_DAGGER)}


def share_tol(p: float, n: int) -> float:
    """The smoke's share tolerance: max(0.05, 3 SE)."""
    return max(0.05, 3.0 * math.sqrt(p * (1.0 - p) / n)) if n else float("inf")
