#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the TABLE / GROUND STAND-UP test's pure pieces (the K3 cells,
PLAN.md H5 / H7 K3, step 2). ``sh_table_test.py`` runs them under Kit; ``tests/unit/test_sh_table.py`` pins them on CPU.

It extends the closed predecessor's table test (``../2026-10-11_stand_network/scripts/sn_table_core.py``, loaded BY PATH
and wrapped, never edited) from one 25-in table and one command to the K3 grid: tables at the three frozen K3 tops and
the ground, three retracted poses, commanded heights, and the stand-height controllers built from ``sh_core`` (the ONE
implementation of q(h), R_h, the limiter, h_meas, the stand-net input and the router).

TORCH- / NUMPY-FREE AT IMPORT (the launcher rule L6): the Kit script loads this file before AppLauncher. At import it
reads only the stdlib, ``sh_core`` (stdlib + the frozen facts + the predecessor's torch-free ``sn_common``) and the
predecessor's ``sn_table_core`` (torch-free at import); torch, numpy, ``sh_kin`` and the linkage load inside functions.

THE K3 CELLS (pre-registered here; PLAN H7 K3 + the audit's "pin ... K3 poses per cell"; re-scoped at the step-4 stop,
CHARTER section 3 "USER DECISION at the step-4 stop: narrow stand-up to tall tables"):

* SURFACES: ``table_0p4572`` / ``table_0p635`` / ``table_0p6985`` / ``table_0p762`` (the CHARTER-frozen gated K3 tops
  ``facts.json`` ``frozen.k3_eval_tops_m`` = 0.635 / 0.6985 / 0.762 and the reported 18-in table
  ``frozen.k3_reported_tops_m`` = 0.4572, sorted by height) and ``ground``.
* GATED vs REPORTED (:data:`K3_GATED_CELLS`): the 6 cells gated tops x gated poses (``frozen.k3_gated_poses`` =
  knee_folded / random) decide every K3 line, the R_h floor and its STOP; every other cell (the 18-in table, every
  rod_min cell, every ground cell: :data:`K3_REPORTED_CELLS`) is run, read and reported, never gated.
* POSES (every surface): ``knee_folded`` (the real retracted pose, user), ``rod_min`` (both lead screws at their minimum)
  -- both the predecessor's :func:`sn_table_core.retracted_pose` -- and ``random`` folds: per env, hip uniform on a
  0.5-deg grid in [-44.5, -30] deg, knee uniform in [max(knee at the knee-rod minimum, knee stop + 0.5 deg), the largest
  fold keeping 2 cm tibia / pad clearance to the hip plate and body (``sn_table_core.max_clear_knee_fold``)] at that hip
  (the ``facts.json`` ``c_table_ground.random_fold_family`` family; the hip grid caps the fold sweeps at 30); the same
  pose on all six legs, Body_Hip / cams 0. On a table a random fold is drawn only if every leg part stays >= 2 cm above
  the floor at spawn and no tibia point below the table-top plane reaches the table's worst jittered corner (rejection
  sampling; the facts' "rule_for_tables").
* 15 cells = 5 surfaces x 3 poses (6 gated + 9 reported), :data:`ENVS_PER_CELL` = 100 envs each (batched; one Isaac
  process can hold them all, ``--split surface`` otherwise).
* COMMANDS (per env, constant for the episode): a table cell draws ``h_cmd`` in [top + 0.31, h_max] (the CHARTER-frozen
  eval margin); a ground cell in [max(h_min, spawn height + 0.14625), h_max] -- the same rise margin the table rule gives
  (0.31 - belly 0.15875 - spawn clearance 0.005), so a ground start's command also clears the 0.10-m belly-rise clause
  by 4.6 cm, and "ground commands across the range" for knee_folded / rod_min ([h_min, h_max]). Stratified: env k of a
  cell gets the quantile ``(perm[k] + 0.5) / n`` of its range (``perm`` a seeded permutation, so the command does not
  follow the env index). Draws, jitter and commands depend only on (env seed, cell, k): a cell's envs are the same
  whatever other cells share the process.
* START: the robot at its env origin, yaw 0, zero velocity, joints at the cell's pose with matching targets (the action
  term re-syncs its rods from these angles). ONE FLOOR RULE for both kinds: the floor reference z_f = the HIGHEST
  terrain point under the robot (a downward raycast of the terrain mesh at the pose's pivot / knee / toe points of all
  six legs and the body corners / centre), so no leg part spawns inside the rough floor (+-2-3 cm). Table: the top at
  z_f + top, root z = z_f + top + 0.15875 (belly) + 0.005; one TALL kinematic cuboid per env (:data:`TABLE_HEIGHT_M`,
  the predecessor's 0.90 x 0.70 m footprint and +-3 cm / +-10 deg jitter, env k = 0 of every cell nominal), its top set
  by its z placement ("top set by z placement", PLAN H5). Ground: the table parked far below the floor
  (:data:`PARK_CENTER_Z_M`); root z = z_f + the pose's lowest-part depth + 0.005.
* EPISODE: one length for the whole K3 suite (:func:`k3_episode_s`), sized for the slowest cell: rod rise (the
  retracted rods slewing at the loaded speed to the posture q(h_init)) + the limiter ramp (h_init = clamp(spawn height)
  -> h_cmd at 0.08 m/s; ground starts bounded with h_init = h_min, the lowest a limiter can start) + settle (1 s, K2's
  allowance) + the 3-s window + 0.5 s margin, rounded up to 0.5 s (16.0 s).

THE CONTROLLERS (:data:`CONTROLLERS`; PLAN "Controls" table, K3 row and its also-reported columns; all through sh_core,
the latched stand slot from t = 0, a ``sh_core.HeightController`` per env stepped BEFORE the actions and reset at spawn
with the limiter at clamp(measured h_meas) -- the table / ground start rule):

* ``r_h`` -- R_h's target (``sh_core.rh_label``, raw clip 4.8) executed on the effective stand rows (known-good);
* ``stand_net`` -- a stand-height checkpoint on ``sh_core.stand_net_input`` (col 15 = the limited command) routed by
  ``sh_core.router_actions`` = q(h_lim) + clamp(a_net, +-1); default checkpoint = A-mask s17 with W[:, 15] zeroed (the
  "A-mask s17 + q" control); a predecessor-tagged (``infos.sn_dagger``) checkpoint is always loaded with W[:, 15] zeroed
  (its column-15 weights are leftovers), a campaign-trained one (``infos.sh_dagger``) as saved; ``--deadband d``
  (``stand_net`` ONLY; default 0 = off, bitwise the path without it; CHARTER 2026-10-14 probe): the router's residual
  deadband (``sh_core.apply_deadband``: hip / knee residual entries below d zeroed after the clamp, before + q),
  recorded in every cell's meta as ``residual_deadband`` (absent = 0); the merge refuses a label whose cells mix
  deadbands (:func:`merge_cells`);
* ``posture_only`` -- zero residual: q(h_lim) (``sh_core.stand_action`` of zeros);
* ``s1`` -- S1 alone on every row, clamp(S1, +-1) (known-bad: marches);
* ``amask_old`` -- A-mask s17 (W[:, 15] zeroed) on the OLD +-1 path: the predecessor's ``sn_common.stand_net_input``
  (no column-15 write) and clamp(a_net, +-1) with no posture (= ``sh_core.stand_action`` at 1.064, bitwise), routed on
  the latched slot (no restore on the old path).
The harness runs the wrapper clip at ``sh_core.HARNESS_CLIP`` = 4.8 (runtime ``env.clip_actions``); S1's rows are
clamped to +-1 explicitly by the router.

K3 STOOD (pre-registered, PLAN H7 K3; :func:`classify`): an env STOOD iff ALL of, over the LAST 3 s (150 steps):
  (1) the ROOT >= its realised SPAWN root z + 0.10 m (belly centre risen) on >= 95 % of the steps (>= 143 of 150;
      review 2026-10-13: the tilted lowest corner, every step, read < 0.10 on robots standing inside the 4-cm height
      tolerance at the lowest commands -- top + 0.31 leaves 0.146 m nominal rise, minus the tolerance, the max-vs-mean
      floor reference and ~1 cm per degree of tilt; envs failing ONLY this clause are reported per cell);
  (2) >= 4 footpads loaded on the floor (net contact >= 20 N, footpad centre z <= 0.15 m: the predecessor table test's
      foot rule) on >= 95 % of the steps (>= 143 of 150; post-settle load flickers) -- 5-down / 6-down reported;
  (3) |h_meas - h_cmd| <= 0.04 m on >= 95 % of the steps (h_meas = root z - mean terrain z under the six feet,
      ``sh_core.h_meas``; h_cmd = the env's raw command, which the limiter has reached by then);
and over the WHOLE episode: (4) no fall: tilt <= 0.5 rad, no ARMED hip contact (any hip link > 500 N, the env's
``crab_failure`` clause, read from the env cfg at runtime), and on a table never more than 0.05 m below the table-top
plane (fell off); (5) no termination (the env runs a time-out only, 2 s past the episode).
THE HIP-CONTACT CLAUSE PER H5: falls are recorded, not reset (the predecessor test's rule); the clause is ARMED from
step 0 on table starts and, on ground starts, from the first step on which the belly has risen >= 0.10 m above its
spawn height (then it stays armed); contacts before arming are reported (rod_min rests on its hip plates at spawn).
Reported beside STOOD: 5-down / 6-down STOOD and window shares, rise-only failures, the lowest-corner rise (the
arming signal), |h - h_cmd| window median / p90 / max, the limiter's
arrival, the first time |h - h_cmd| <= 2 cm (K2-style), lift-off (belly risen > 2 cm held 0.5 s), standing-since,
final h_meas, max tilt, falls by cause, HOMED (p2s_stand Z2: every cam within 0.10 rad over the last 5 s = homed at the
end), settled (homed held 0.5 s), cams homed on the final step.

K3 READ (PLAN H7; :func:`k3_read`; re-scoped at the step-4 stop): per env seed, the GATED start type ``tables`` (the 6
gated cells pooled) -- a controller passes iff its STOOD rate >= R_h's - max(5 pp, 2 SE of the difference) and its
falls per 100 <= max(1, R_h's). ``ground`` (the 3 ground cells pooled) is the same line REPORTED (``gated`` False);
every cell, every top (gated poses and all poses) and every ground pose is reported. The step-4 R_h FLOOR premise
(:func:`rh_floor`): R_h STOOD >= 0.90 and falls <= 1 per 100 on each GATED top pooled over the gated poses; every
reported cell is listed beside it (rates, falls, 5 / 6-down, homed), never part of the floor or its STOP.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import math
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
#: scripts -> 2026-10-13_stand_height -> experiments -> crab_hex_omni_task
TASK = HERE.parents[2]
PRED_SCRIPTS = TASK / "experiments" / "2026-10-11_stand_network" / "scripts"
SH_CORE_PATH = HERE / "sh_core.py"
SH_KIN_PATH = HERE / "sh_kin.py"
STC_PATH = PRED_SCRIPTS / "sn_table_core.py"


def _load_by_path(path: Path, name: str):
    """A private module instance (not in ``sys.modules``; no ``__pycache__`` written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_table_core: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the campaign core (q map, R_h, limiter, h_meas, stand-net input, router, restore, warm start) -- import, never copy
SH = _load_by_path(SH_CORE_PATH, "_sht_sh_core")
#: the predecessor table test's geometry (poses, hulls, spawn arithmetic, feet counting) -- wrapped, never edited
STC = _load_by_path(STC_PATH, "_sht_sn_table_core")

# ===================================================================================== constants
LEGS = STC.LEGS
LEFT = STC.LEFT
POSES = ("knee_folded", "rod_min", "random")
FIXED_POSES = ("knee_folded", "rod_min")
CONTROLLERS = ("r_h", "stand_net", "posture_only", "s1", "amask_old")
#: controllers that run a stand checkpoint (``--stand-ckpt``)
STAND_CONTROLLERS = ("stand_net", "amask_old")
#: the one controller a residual deadband (--deadband) may run on (CHARTER 2026-10-14 probe)
DEADBAND_CONTROLLER = "stand_net"
REF_CONTROLLER = "r_h"
#: the GATED K3 tops (CHARTER-frozen, re-scoped at the step-4 stop: 0.635 / 0.6985 / 0.762)
K3_TOPS_M = tuple(float(t) for t in SH.FROZEN["k3_eval_tops_m"])
#: tops run and read but never gated (the 18-in table)
K3_REPORTED_TOPS_M = tuple(float(t) for t in SH.FROZEN["k3_reported_tops_m"])
#: the poses whose cells on the gated tops are gated (knee_folded, random); rod_min cells are reported only
K3_GATED_POSES = tuple(str(p) for p in SH.FROZEN["k3_gated_poses"])
TOP_MIN_M = float(SH.FROZEN["top_min_m"])
TOP_MAX_TRAIN_M = float(SH.FROZEN["top_max_train_m"])
if not set(K3_GATED_POSES) <= set(POSES) or set(K3_TOPS_M) & set(K3_REPORTED_TOPS_M):
    raise RuntimeError(f"sh_table_core: frozen K3 cells inconsistent (gated poses {K3_GATED_POSES} vs {POSES}; "
                       f"tops {K3_TOPS_M} / reported {K3_REPORTED_TOPS_M})")


def surface_id(top) -> str:
    """``table_0p635`` for a top of 0.635 m; ``ground`` for None."""
    return "ground" if top is None else "table_" + f"{float(top):g}".replace(".", "p")


SURFACES = tuple(surface_id(t) for t in sorted(K3_TOPS_M + K3_REPORTED_TOPS_M)) + ("ground",)
SURFACE_TOP = {surface_id(t): t for t in K3_TOPS_M + K3_REPORTED_TOPS_M}
SURFACE_TOP["ground"] = None
CELL_SEP = "__"
K3_CELLS = tuple(f"{s}{CELL_SEP}{p}" for s in SURFACES for p in POSES)
K3_GATED_SURFACES = tuple(surface_id(t) for t in K3_TOPS_M)
#: THE GATED SET (CHARTER step-4 decision): gated tops x gated poses -- every K3 line, the R_h floor and its STOP
K3_GATED_CELLS = tuple(f"{s}{CELL_SEP}{p}" for s in K3_GATED_SURFACES for p in K3_GATED_POSES)
#: run, read and reported, never gated: the 18-in table, every rod_min cell, every ground cell
K3_REPORTED_CELLS = tuple(c for c in K3_CELLS if c not in K3_GATED_CELLS)
ENVS_PER_CELL = 100


def cell_gated(cell: str) -> bool:
    """True iff ``cell`` is one of the 6 gated K3 cells."""
    return str(cell) in K3_GATED_CELLS


START_TABLE, START_GROUND = "table", "ground"
#: sh_core.START_KINDS index per start (HeightController.reset)
START_CODE = {START_TABLE: SH.START_KINDS.index("table"), START_GROUND: SH.START_KINDS.index("ground")}

# ------------------------------------------------------------------ geometry (the predecessor's, plus the tall table)
BODY_HALF_Z_M = float(STC.BODY_HALF_M[2])          # 0.15875: belly below the root
SPAWN_CLEARANCE_M = float(STC.SPAWN_CLEARANCE_M)    # 0.005
TABLE_X_M, TABLE_Y_M = float(STC.TABLE_X_M), float(STC.TABLE_Y_M)
#: one tall cuboid for every top (its base 0.24-0.54 m below the floor; kinematic: no contact with the static terrain)
TABLE_HEIGHT_M = 1.0
#: a ground env's table: centre 5 m below the floor (top at -4.5 m): out of reach and out of every raycast
PARK_CENTER_Z_M = -5.0
TABLE_MASS_KG = float(STC.TABLE_MASS_KG)
JITTER_XY_M, JITTER_YAW_DEG = float(STC.JITTER_XY_M), float(STC.JITTER_YAW_DEG)
TOE_FLOOR_CLEAR_M = 0.02                            # a table start: every leg part >= this above the floor
TABLE_TIBIA_MARGIN_M = 0.02                         # random folds on a table: tibia vs the table's worst corner
# ------------------------------------------------------------------ the random-fold family (facts.json)
RANDOM_HIP_MAX_DEG = -30.0
RANDOM_HIP_STEP_DEG = 0.5
RANDOM_MAX_TRIES = 2000
# ------------------------------------------------------------------ commands
CMD_MARGIN_EVAL_M = 0.31                            # CHARTER frozen: K3 eval commands >= top + 0.31
#: a ground start's command clears its spawn height by the table rule's rise margin
GROUND_CMD_RISE_M = CMD_MARGIN_EVAL_M - BODY_HALF_Z_M - SPAWN_CLEARANCE_M
# ------------------------------------------------------------------ episode sizing
SETTLE_S = 1.0                                      # K2's "+ 1 s" after the ramp
STOOD_WINDOW_S = 3.0
EPISODE_MARGIN_S = 0.5
EPISODE_ROUND_S = 0.5
# ------------------------------------------------------------------ K3 STOOD (module docstring)
STOOD_RISE_M = 0.10
STOOD_MIN_FEET = 4
STOOD_SHARE = 0.95
STOOD_H_TOL_M = 0.04
FOOT_LOADED_N = float(STC.FOOT_LOADED_N)            # 20 N (the predecessor table test's foot rule)
FOOT_FLOOR_Z_M = float(STC.FOOT_FLOOR_Z_M)          # 0.15 m
FALL_TILT_RAD = 0.5                                 # the env's crab_failure.limit_angle (re-read at runtime)
HIP_CONTACT_N = 500.0                               # the env's crab_failure.contact_force_threshold (re-read)
FELL_OFF_DROP_M = float(STC.FELL_OFF_DROP_M)        # 0.05
ARM_RISE_M = 0.10                                   # H5: the hip-contact clause re-arms at this belly rise
LIFT_RISE_M = 0.02
LIFT_HOLD_S = 0.5
ARRIVE_TOL_M = 0.02                                 # K2-style "within 2 cm", reported
HOMED_RAD = 0.10                                    # p2s_stand.Z2_HOMED_RAD
HOMED_LAST_S = 5.0                                  # p2s_stand.Z2_LAST_S
SETTLE_HOLD_S = 0.5                                 # p2s_stand.Z2_SETTLE_HOLD_S
_EPS = 1e-9
# ------------------------------------------------------------------ the K3 read (PLAN H7) and the step-4 R_h floor
K3_RATE_PP = 0.05
K3_SE_K = 2.0
K3_FALLS_FLOOR_PER100 = 1.0
RH_FLOOR_STOOD = 0.90
RH_FLOOR_FALLS_PER100 = 1.0
#: the GATED start type (the 6 gated cells pooled) and the REPORTED one (the ground cells pooled; never gated)
START_TYPES = ("tables",)
REPORTED_START_TYPES = ("ground",)

STOOD_DEFINITION = (
    "K3 STOOD iff, over the last 3.0 s (150 control steps): (1) the ROOT risen >= 0.10 m above its realised spawn root "
    "z on >= 95 % of the steps (>= 143 of 150; the belly centre's rise -- a tilted chassis' lowest corner is not "
    "used, it drops ~1 cm per degree of tilt); (2) >= 4 of 6 footpads loaded on the floor (net contact >= 20 N, "
    "footpad centre z <= 0.15 m) on >= 95 % of the steps (>= 143 of 150); (3) |h_meas - h_cmd| <= 0.04 m on >= 95 % "
    "of the steps (h_meas = root z - mean terrain z under the six feet; h_cmd = the env's command); and over the whole "
    "episode (4) no fall: tilt <= 0.5 rad, no ARMED hip contact (any hip link > 500 N; armed from step 0 on a table, "
    "on the ground from the first step the belly has risen >= 0.10 m above its spawn height -- PLAN H5), on a table "
    "never more than 0.05 m below the table-top plane; (5) no termination (time-out only, 2 s past the episode). "
    "Reported: 5-down / 6-down STOOD, window shares, the envs that fail ONLY the rise clause (``rise_only_fail``), the "
    "lowest-corner rise, |h - h_cmd| median / p90 / max, limiter arrival, first time within 2 cm, lift-off, "
    "standing-since, falls by cause, pre-arm hip contacts, HOMED (every cam <= 0.10 rad over the last 5 s), settled, "
    "cams homed on the final step. The hip-contact ARMING (4) keeps the lowest-corner rise (the trainer's re-arm rule, "
    "sh_starts.post_physics).")


# ===================================================================================== lazy loaders
_CACHE: dict = {}


def kin():
    """``sh_kin`` (numpy: q, the linkage, rod speeds), loaded by path on first use."""
    if "kin" not in _CACHE:
        _CACHE["kin"] = _load_by_path(SH_KIN_PATH, "_sht_sh_kin")
    return _CACHE["kin"]


def _np():
    import numpy as np  # noqa: PLC0415

    return np


# ===================================================================================== cells
def parse_cell(cell: str) -> tuple[str, str]:
    """``(surface, pose)`` of a cell id ``<surface>__<pose>`` (refuses an unknown one)."""
    parts = str(cell).split(CELL_SEP)
    if len(parts) != 2 or parts[0] not in SURFACES or parts[1] not in POSES:
        raise ValueError(f"cell {cell!r}: want <surface>{CELL_SEP}<pose>, surface in {SURFACES}, pose in {POSES}")
    return parts[0], parts[1]


def parse_cells(text: str) -> list[str]:
    """``all`` (the 15 K3 cells, gated and reported), or a comma list of cell ids / surfaces (a surface expands to its 3
    poses)."""
    t = str(text).strip()
    if t in ("", "all"):
        return list(K3_CELLS)
    out: list[str] = []
    for tok in (x.strip() for x in t.split(",") if x.strip()):
        cells = [f"{tok}{CELL_SEP}{p}" for p in POSES] if tok in SURFACES else [tok]
        for c in cells:
            parse_cell(c)
            if c not in out:
                out.append(c)
    if not out:
        raise ValueError(f"--cells {text!r}: no cells")
    return out


def start_kind(surface: str) -> str:
    return START_GROUND if surface == "ground" else START_TABLE


def table_center_z(top, floor_z: float = 0.0) -> float:
    """The tall cuboid's centre z for a top above the floor reference ``floor_z`` (None = parked, absolute)."""
    return PARK_CENTER_Z_M if top is None else float(floor_z) + float(top) - 0.5 * TABLE_HEIGHT_M


def table_spawn_root_z(top: float) -> float:
    """Root height above the floor reference on a table: top + belly 0.15875 + 0.005 (``STC.spawn_root_z``)."""
    return float(STC.spawn_root_z(float(top)))


# ===================================================================================== pose geometry
@functools.lru_cache(maxsize=8)
def fixed_pose(pose: str) -> dict:
    """``sn_table_core.retracted_pose`` for knee_folded / rod_min (cached)."""
    if pose not in FIXED_POSES:
        raise ValueError(f"fixed_pose: {pose!r} not in {FIXED_POSES}")
    return STC.retracted_pose(pose)


def lowest_parts(hip: float, knee: float) -> dict:
    """Lowest leg-plane z (body frame, root at 0) per part for a pose, with ``sn_table_core``'s hulls (the step-1 facts'
    rule, ``sh_facts._lowest_parts``): tibia + pad hull, femur stadium, hip-plate stadium, belly."""
    tib = STC.tibia_hull(hip, knee)
    lp = STC.leg_plane_points(hip, knee)
    plate_bottom = -(STC.PLATE_CORE_HALF_Z_M + STC.PLATE_RADIUS_M)
    femur_low = min(lp["pivot"][1], lp["knee"][1]) - STC.FEMUR_RADIUS_M
    parts = {"tibia_or_pad": min(p[1] for p in tib), "femur": femur_low, "hip_plate": plate_bottom,
             "belly": -BODY_HALF_Z_M}
    low = min(parts.values())
    return {"parts_z": parts, "lowest_z": low, "leg_low_z": min(v for k, v in parts.items() if k != "belly"),
            "first_contact": "+".join(sorted(k for k, v in parts.items() if v <= low + 1e-3))}


def ground_root_rel(hip: float, knee: float) -> float:
    """Root height above the highest floor point under the robot at a ground spawn: -lowest part + 0.005."""
    return -lowest_parts(hip, knee)["lowest_z"] + SPAWN_CLEARANCE_M


def table_top_min(hip: float, knee: float) -> float:
    """The lowest table top on which every leg part of the pose stays >= 2 cm above the floor at spawn."""
    return TOE_FLOOR_CLEAR_M - lowest_parts(hip, knee)["leg_low_z"] - BODY_HALF_Z_M - SPAWN_CLEARANCE_M


def _leg_mount_y() -> float:
    return float(STC.dims().LEG_MOUNT_Y_M)


def table_tibia_clear(hip: float, knee: float, top: float, *, margin: float = TABLE_TIBIA_MARGIN_M) -> bool:
    """No tibia / pad hull point below the table-top plane at spawn reaches the table's worst jittered corner: a point
    ``u`` (outboard of the yaw axis) sits at body |y| >= mount_y + min(u, 0) (cos(splay) <= 1), so the condition is
    ``u >= table_max_abs_y + margin - mount_y`` for every hull point with world z < top + margin."""
    root_z = table_spawn_root_z(top)
    u_lim = float(STC.table_max_abs_y()) + float(margin) - _leg_mount_y()
    for u, z in STC.tibia_hull(hip, knee):
        if root_z + z < float(top) + float(margin) and u < u_lim:
            return False
    return True


def fold_ok_on_table(hip: float, knee: float, top: float) -> bool:
    """A pose may start on a table of ``top``: legs >= 2 cm above the floor and the tibias clear of the table."""
    return table_top_min(hip, knee) <= float(top) + _EPS and table_tibia_clear(hip, knee, top)


@functools.lru_cache(maxsize=1)
def random_hip_grid() -> tuple:
    """The random-fold hips (rad): [-44.5 deg (stop + margin), -30 deg] on a 0.5-deg grid (30 nodes)."""
    lo = math.degrees(STC.joint_limits_rad()["hip"][0]) + STC.LIMIT_MARGIN_DEG
    n = int(round((RANDOM_HIP_MAX_DEG - lo) / RANDOM_HIP_STEP_DEG)) + 1
    return tuple(math.radians(lo + i * RANDOM_HIP_STEP_DEG) for i in range(n))


@functools.lru_cache(maxsize=64)
def random_knee_range(hip: float) -> tuple:
    """``(k_lo, k_hi)`` (left convention) of the random-fold family at ``hip``: k_lo = max(knee at the knee-rod
    minimum, knee stop + 0.5 deg), k_hi = ``sn_table_core.max_clear_knee_fold`` (the 1-mrad sweep)."""
    K = kin()
    lim = STC.joint_limits_rad()
    m = math.radians(STC.LIMIT_MARGIN_DEG)
    k_lo = max(float(K.knee_angle(K.KNEE_LEN_MIN_M, hip)), lim["knee_left"][0] + m)
    k_hi = float(STC.max_clear_knee_fold(float(hip))["knee_rad"])
    if not k_lo < k_hi:
        raise ValueError(f"random_knee_range: empty knee range at hip {hip}: [{k_lo}, {k_hi}]")
    return k_lo, k_hi


def _rng(seed: int, cell: str, k: int, stream: int):
    """A private numpy generator for (env seed, cell, env index in the cell, stream) -- never the global RNG."""
    return _np().random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(str(cell).encode()), int(k), int(stream)])


STREAM_JITTER, STREAM_FOLD, STREAM_CMD = 1, 2, 3


def draw_random_fold(seed: int, cell: str, k: int, top) -> dict:
    """A random fold for env ``k`` of ``cell`` (``top`` None = ground; on a table, rejection-sampled with
    :func:`fold_ok_on_table`)."""
    g = _rng(seed, cell, k, STREAM_FOLD)
    hips = random_hip_grid()
    for tries in range(1, RANDOM_MAX_TRIES + 1):
        hip = hips[int(g.integers(0, len(hips)))]
        k_lo, k_hi = random_knee_range(hip)
        knee = float(k_lo + (k_hi - k_lo) * float(g.random()))
        if top is None or fold_ok_on_table(hip, knee, top):
            return {"hip_rad": hip, "knee_left_rad": knee, "tries": tries}
    raise RuntimeError(f"draw_random_fold: no table-clear fold for {cell} k={k} in {RANDOM_MAX_TRIES} tries")


def jitter_offset(seed: int, cell: str, k: int, enabled: bool) -> list:
    """The table offset ``[dx, dy, dyaw]`` of env ``k`` of ``cell``: uniform +-3 cm / +-10 deg; k = 0 (and
    ``enabled=False``) nominal."""
    if not enabled or int(k) == 0:
        return [0.0, 0.0, 0.0]
    u = _rng(seed, cell, k, STREAM_JITTER).uniform(-1.0, 1.0, size=3)
    return [float(u[0]) * JITTER_XY_M, float(u[1]) * JITTER_XY_M, float(u[2]) * math.radians(JITTER_YAW_DEG)]


def cmd_quantiles(seed: int, cell: str, n: int) -> list:
    """Stratified command quantiles ``(perm[k] + 0.5) / n`` (a seeded permutation; n = 1 -> 0.5)."""
    perm = _rng(seed, cell, 0, STREAM_CMD).permutation(int(n))
    return [(float(p) + 0.5) / float(n) for p in perm]


def cmd_range(surface: str, spawn_h: float) -> tuple:
    """The K3 command range of an env: table [top + 0.31, h_max]; ground [max(h_min, spawn + 0.14625), h_max]."""
    top = SURFACE_TOP[surface]
    lo = (float(top) + CMD_MARGIN_EVAL_M) if top is not None else max(SH.H_MIN, float(spawn_h) + GROUND_CMD_RISE_M)
    if not lo <= SH.H_MAX + _EPS:
        raise ValueError(f"cmd_range: {surface} spawn {spawn_h}: lower bound {lo} above h_max {SH.H_MAX}")
    return min(lo, SH.H_MAX), SH.H_MAX


# ===================================================================================== episode sizing
def rod_lengths(hip: float, knee: float) -> tuple:
    """(hip rod, knee rod) lengths (m) of a joint pose (left convention)."""
    K = kin()
    return float(K.hip_len(hip)), float(K.knee_len(hip, knee))


def posture_rod_targets(h: float) -> tuple:
    """The action layer's rod targets for the posture q(h) (h clamped to the frozen range): hip rod at default +
    0.25 q_hip, knee rod at the DEFAULT hip from knee default + 0.25 q_knee (parkour_actions), both clamped."""
    K = kin()
    hc = min(max(float(h), SH.H_MIN), SH.H_MAX)
    qh, qk = (float(x) for x in K.q(hc))
    lh = min(max(float(K.hip_len(K.TH0 + K.ROD_SCALE * qh)), K.HIP_LEN_MIN_M), K.HIP_LEN_MAX_M)
    lk = min(max(float(K.knee_len(K.TH0, K.TK0 + K.ROD_SCALE * qk)), K.KNEE_LEN_MIN_M), K.KNEE_LEN_MAX_M)
    return lh, lk


def rod_slew_s(hip0: float, knee0: float, h: float) -> float:
    """Seconds for both rods to slew (loaded speed) from the pose to the posture q(h)'s rod targets."""
    K = kin()
    lh0, lk0 = rod_lengths(hip0, knee0)
    lh, lk = posture_rod_targets(h)
    return max(abs(lh - lh0) / float(K.HIP_SPEED_M_S), abs(lk - lk0) / float(K.KNEE_SPEED_M_S))


def env_need_s(hip0: float, knee0: float, h_init: float, h_cmd: float) -> dict:
    """The episode an env needs: max(rise to q(h_init) + ramp, rise to q(h_cmd)) + settle + window + margin."""
    ramp = abs(float(h_cmd) - float(h_init)) / SH.HEIGHT_RATE_M_S
    rise0 = rod_slew_s(hip0, knee0, h_init)
    rise1 = rod_slew_s(hip0, knee0, h_cmd)
    reach = max(rise0 + ramp, rise1)
    return {"rise_to_init_s": rise0, "ramp_s": ramp, "rise_to_cmd_s": rise1, "reach_s": reach,
            "need_s": reach + SETTLE_S + STOOD_WINDOW_S + EPISODE_MARGIN_S}


def round_episode(seconds: float) -> float:
    return math.ceil(float(seconds) / EPISODE_ROUND_S - 1e-9) * EPISODE_ROUND_S


# ===================================================================================== the per-env plan
def plan_envs(cells, n_per_cell: int, seed: int, *, jitter: bool = True, h_cmd=None) -> list:
    """One dict per env, cell-major (env = cell index x n + k): surface, start, top, pose angles, table offset, the
    spawn root height above the floor reference (``spawn_root_rel``; the reset event adds the floor), the planned spawn
    height (h_meas at spawn on a flat floor), h_init = its clamp, the command and its range, the episode need, the
    floor probe points. ``h_cmd`` (a float) fixes every env's command (it must lie in every env's range)."""
    out = []
    n = int(n_per_cell)
    if n < 1:
        raise ValueError("plan_envs: n_per_cell >= 1")
    for ci, cell in enumerate(cells):
        surface, pose = parse_cell(cell)
        top = SURFACE_TOP[surface]
        quant = cmd_quantiles(seed, cell, n)
        for k in range(n):
            if pose == "random":
                d = draw_random_fold(seed, cell, k, top)
                hip, knee, tries = d["hip_rad"], d["knee_left_rad"], d["tries"]
            else:
                p = fixed_pose(pose)
                hip, knee, tries = float(p["hip_rad"]), float(p["knee_left_rad"]), 0
                if top is not None and not fold_ok_on_table(hip, knee, top):
                    raise ValueError(f"plan_envs: {pose} cannot start on the {top} m table")
            spawn_root = ground_root_rel(hip, knee) if top is None else table_spawn_root_z(top)
            spawn_h = spawn_root
            lo, hi = cmd_range(surface, spawn_h)
            if h_cmd is None:
                cmd = lo + (hi - lo) * quant[k]
            else:
                cmd = float(h_cmd)
                if not lo - _EPS <= cmd <= hi + _EPS:
                    raise ValueError(f"plan_envs: --h-cmd {cmd} outside {cell} k={k}'s range [{lo:.4f}, {hi:.4f}]")
            h_init = min(max(spawn_h, SH.H_MIN), SH.H_MAX)
            need = env_need_s(hip, knee, h_init, cmd)
            out.append({"env": ci * n + k, "cell": cell, "k": k, "surface": surface, "start": start_kind(surface),
                        "top": top, "pose": pose, "hip_rad": float(hip), "knee_left_rad": float(knee),
                        "fold_tries": tries, "table_offset": jitter_offset(seed, cell, k, jitter and top is not None),
                        "spawn_root_rel": float(spawn_root), "spawn_h_plan": float(spawn_h),
                        "h_init_plan": float(h_init), "h_cmd": float(cmd), "h_cmd_range": [float(lo), float(hi)],
                        "need_s": need["need_s"], "sizing": need})
    return out


def plan_episode_s(plan) -> float:
    """The episode a plan needs (its slowest env), rounded up to 0.5 s."""
    return round_episode(max(e["need_s"] for e in plan))


@functools.lru_cache(maxsize=1)
def k3_episode_s() -> float:
    """THE K3 suite episode (every cell, every controller): the slowest env of the 15 cells (gated and reported) at
    the worst command (h_max), the random family's slowest fold included (the rod slew is monotone in the knee rod, so
    the knee range's ends bound it; a ground start's limiter is bounded from h_min) -- computed from the geometry, not
    from a seed."""
    worst = 0.0
    for cell in K3_CELLS:
        surface, pose = parse_cell(cell)
        top = SURFACE_TOP[surface]
        poses = ([(float(fixed_pose(pose)["hip_rad"]), float(fixed_pose(pose)["knee_left_rad"]))]
                 if pose in FIXED_POSES else
                 [(h, k) for h in random_hip_grid() for k in random_knee_range(h)])
        for hip, knee in poses:
            h_init = SH.H_MIN if top is None else min(max(table_spawn_root_z(top), SH.H_MIN), SH.H_MAX)
            worst = max(worst, env_need_s(hip, knee, h_init, SH.H_MAX)["need_s"])
    return round_episode(worst)


def joint_pos_rows(plan, joint_names) -> list:
    """(N, len(joint_names)) joint angles in the articulation order: Body_Hip / CamShaft 0, Hip_Femur = hip, Femur_Tibia
    = knee (left) / -knee (right; ``sn_table_core.retracted_pose``'s mirroring)."""
    rows = []
    for e in plan:
        row = []
        for name in joint_names:
            leg = name.split("_")[0]
            if "Hip_Femur" in name:
                row.append(e["hip_rad"])
            elif "Femur_Tibia" in name:
                row.append(e["knee_left_rad"] if leg in LEFT else -e["knee_left_rad"])
            elif "Body_Hip" in name or "CamShaft" in name:
                row.append(0.0)
            else:
                raise ValueError(f"joint_pos_rows: unknown joint {name!r}")
        rows.append(row)
    return rows


def probe_points_body(hip: float, knee: float) -> list:
    """Body-frame xy points under which a ground spawn reads the floor: every leg's pivot / knee / toe (foot FK, plant
    A15+B, Body_Hip 0) + the body box corners and centre."""
    pts = []
    for p in STC.foot_points_body(hip, knee).values():
        for key in ("pivot", "knee", "toe"):
            pts.append([float(p[key][0]), float(p[key][1])])
    hx, hy = float(STC.BODY_HALF_M[0]), float(STC.BODY_HALF_M[1])
    pts += [[sx * hx, sy * hy] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)] + [[0.0, 0.0]]
    return pts


def probe_rows(plan) -> list:
    """(N, P, 2) body-frame floor-probe points per env (:func:`probe_points_body` of its pose)."""
    return [probe_points_body(e["hip_rad"], e["knee_left_rad"]) for e in plan]


def event_params(plan, joint_names) -> dict:
    """The reset event's params for a plan (plain lists: the env cfg carries them)."""
    return {"joint_pos": joint_pos_rows(plan, joint_names), "start_code": [START_CODE[e["start"]] for e in plan],
            "root_rel": [e["spawn_root_rel"] for e in plan],
            "table_top_rel": [float("nan") if e["top"] is None else float(e["top"]) for e in plan],
            "table_offsets": [list(map(float, e["table_offset"])) for e in plan], "probe_xy_b": probe_rows(plan)}


def plan_digest(plan) -> str:
    """sha256 of the plan's JSON (the per-env starts and commands a run used)."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


# ===================================================================================== the action clip (fail closed)
def action_clip_reasons(clip, scale) -> list:
    """The env's ``actions.joint_pos`` clip / scale must be the preset-2a values the H1 residual relies on: rods +-4.8
    (``sh_core.RAW_CLIP``), cams +-1 (``sh_core.CAM_CLIP``), rod scale 0.25 rad per unit."""
    out = []
    want = {".*_Body_CamShaft_RevoluteJoint": (-SH.CAM_CLIP, SH.CAM_CLIP),
            ".*_Hip_Femur_RevoluteJoint": (-SH.RAW_CLIP, SH.RAW_CLIP),
            ".*_Femur_Tibia_RevoluteJoint": (-SH.RAW_CLIP, SH.RAW_CLIP)}
    if not isinstance(clip, dict):
        return [f"actions.joint_pos.clip is {clip!r}, want a per-joint dict {want}"]
    for k, v in want.items():
        got = clip.get(k)
        if got is None or tuple(float(x) for x in got) != v:
            out.append(f"actions.joint_pos.clip[{k!r}] = {got!r}, want {v}")
    for k in (".*_Hip_Femur_RevoluteJoint", ".*_Femur_Tibia_RevoluteJoint"):
        got = scale.get(k) if isinstance(scale, dict) else None
        if got is None or abs(float(got) - SH.ROD_SCALE) > 1e-12:
            out.append(f"actions.joint_pos.scale[{k!r}] = {got!r}, want {SH.ROD_SCALE}")
    return out


# ===================================================================================== the reset event (Kit)
#: runtime hooks: ``terrain_probe`` (a callable xy_w (M, 2) -> z (M,); the Kit script registers the terrain mesh's
#: raycast here before the construction reset, tests a fake; unset = the env's own mesh, looked up lazily) with its
#: ``probe_label``, and what the event did (``last_root_z`` (N,) device tensor, ``probe_kind``)
RUNTIME: dict = {}
PROBE_START_Z_M = 20.0


def make_terrain_probe(mesh, raycast_fn):
    """``xy_w (M, 2) -> z (M,)``: a downward raycast of the terrain mesh from z = 20 m (refuses a missing mesh and any
    miss -- a non-finite hit would spawn the robot at infinity)."""
    if mesh is None:
        raise RuntimeError("make_terrain_probe: no terrain mesh")

    def probe(xy):
        import torch  # noqa: PLC0415

        m = int(xy.shape[0])
        starts = torch.zeros((m, 3), dtype=torch.float32, device=xy.device)
        starts[:, :2] = xy.to(torch.float32)
        starts[:, 2] = PROBE_START_Z_M
        dirs = torch.zeros_like(starts)
        dirs[:, 2] = -1.0
        hits = raycast_fn(starts.contiguous(), dirs.contiguous(), mesh=mesh)[0]
        z = hits[:, 2]
        if not bool(torch.isfinite(z).all()):
            raise RuntimeError("terrain probe: a ray missed the terrain mesh")
        return z

    return probe


def terrain_probe_for(env):
    """The ground-spawn floor probe: :data:`RUNTIME` ``terrain_probe`` when set, else the env's height-scanner terrain
    mesh (``RayCaster.meshes[mesh_prim_paths[0]]``, the eval harness's lookup) with ``raycast_mesh`` (lazy Isaac
    imports). Fails closed."""
    if RUNTIME.get("terrain_probe") is not None:
        RUNTIME["probe_kind"] = RUNTIME.get("probe_label") or "registered"
        return RUNTIME["terrain_probe"]
    from isaaclab.sensors.ray_caster import RayCaster  # noqa: PLC0415
    from isaaclab.utils.warp import raycast_mesh  # noqa: PLC0415

    scanner = env.scene.sensors.get("height_scanner")
    key = scanner.cfg.mesh_prim_paths[0] if scanner is not None else "/World/ground"
    mesh = RayCaster.meshes.get(key)
    if mesh is None:
        raise RuntimeError(f"terrain_probe_for: no RayCaster mesh {key!r} (ground spawns need the floor height)")
    RUNTIME["probe_kind"] = f"raycast_mesh:{key}"
    return make_terrain_probe(mesh, raycast_mesh)


def sh_start(env, env_ids, joint_pos, start_code, root_rel, table_top_rel, table_offsets, probe_xy_b):
    """Reset event (appended LAST): per env ``i`` in ``env_ids`` -- the floor reference z_f = the HIGHEST terrain point
    under the probe points ``probe_xy_b[i]`` (body frame = world-aligned at yaw 0, + the env origin); the robot at its
    env origin xy, root z = z_f + ``root_rel[i]``, yaw 0, zero velocity, joints ``joint_pos[i]`` (articulation order)
    with matching position targets and zero velocity targets; a table start's table at the origin + its offset with its
    top at z_f + ``table_top_rel[i]`` (a ground start's, NaN there, parked). The realised root z and z_f go to
    ``RUNTIME['last_root_z']`` / ``RUNTIME['last_floor_z']``."""
    import torch  # noqa: PLC0415

    robot = env.scene["robot"]
    table = env.scene["sh_table"]
    dev = env.device
    ids = torch.arange(env.num_envs, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev).long()
    n = int(ids.numel())
    if n == 0:
        return
    origin = env.scene.env_origins[ids].clone()
    origin[:, 2] = 0.0
    dtype = origin.dtype
    code = torch.as_tensor(start_code, device=dev)[ids]
    pts = torch.as_tensor(probe_xy_b, device=dev, dtype=dtype)[ids]                  # (n, P, 2)
    xy = pts + origin[:, None, :2]
    floor = terrain_probe_for(env)(xy.reshape(-1, 2)).to(dtype).reshape(xy.shape[0], xy.shape[1]).max(dim=1).values
    rz = floor + torch.as_tensor(root_rel, device=dev, dtype=dtype)[ids]
    is_table = code == START_CODE[START_TABLE]
    top_rel = torch.as_tensor(table_top_rel, device=dev, dtype=dtype)[ids]
    if bool(torch.isnan(top_rel[is_table]).any()) or not bool(torch.isnan(top_rel[~is_table]).all()):
        raise ValueError("sh_start: table starts need a table top, ground starts NaN")
    center = torch.where(is_table, floor + torch.nan_to_num(top_rel) - 0.5 * TABLE_HEIGHT_M,
                         torch.full_like(floor, PARK_CENTER_Z_M))
    pos = origin.clone()
    pos[:, 2] = rz
    quat = torch.zeros((n, 4), device=dev, dtype=dtype)
    quat[:, 0] = 1.0
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev, dtype=dtype), env_ids=ids)
    jp = torch.as_tensor(joint_pos, device=dev, dtype=dtype)[ids]
    jv = torch.zeros_like(jp)
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    robot.set_joint_position_target(jp, env_ids=ids)
    robot.set_joint_velocity_target(jv, env_ids=ids)
    off = torch.as_tensor(table_offsets, device=dev, dtype=dtype)[ids]
    tpos = origin.clone()
    tpos[:, 0] += off[:, 0]
    tpos[:, 1] += off[:, 1]
    tpos[:, 2] = center
    tquat = torch.zeros((n, 4), device=dev, dtype=dtype)
    tquat[:, 0] = torch.cos(0.5 * off[:, 2])
    tquat[:, 3] = torch.sin(0.5 * off[:, 2])
    table.write_root_pose_to_sim(torch.cat([tpos, tquat], dim=-1), env_ids=ids)
    try:   # the same pose as the kinematic TARGET (the predecessor's rule; the Kit script checks it after step 1)
        full = table.data.root_link_pose_w.clone()
        full_xyzw = torch.cat([full[:, :3], full[:, 4:7], full[:, 3:4]], dim=-1)
        table.root_physx_view.set_kinematic_targets(full_xyzw, ids)
    except Exception as exc:  # noqa: BLE001
        if not RUNTIME.get("warned_kin"):
            RUNTIME["warned_kin"] = True
            print(f"[SHT][WARN] table kinematic target not set ({type(exc).__name__}: {exc}); relying on the pose "
                  "write (checked after the first step)", flush=True)
    for key, val in (("last_root_z", rz), ("last_floor_z", floor)):
        last = RUNTIME.get(key)
        if last is None or int(last.numel()) != int(env.num_envs) or last.device != torch.device(dev):
            last = torch.full((int(env.num_envs),), float("nan"), device=dev, dtype=dtype)
        last = last.clone()
        last[ids] = val
        RUNTIME[key] = last


# ===================================================================================== controllers (Kit; sh_core only)
def controller_actions(controller: str, obs, stand_slot, hs, *, s1, s1_est, p0: int, p1: int, num_prop: int,
                       st_pol=None, st_est=None, mask: int = 1, deadband: float = 0.0):
    """One control step's actions -> ``(actions, extra)``. ``hs`` = this step's ``sh_core.HeightController.step``
    (called BEFORE this, on the same obs). The harness's S1 path first, always: S1's own estimate on a COPY ``h`` of
    the env obs (the env obs is never written), ``a_s1 = s1.act_inference(h, hist_encoding=True)``. Then (module
    docstring): ``r_h`` / ``stand_net`` / ``posture_only`` on ``hs.stand_eff`` with ``hs.h_lim``; ``s1``;
    ``amask_old`` on ``stand_slot``. ``extra``: ``a_s1``, and per controller ``a_net``, ``q18``, ``r_target`` /
    ``r_branch`` / ``r_label_excess`` (device scalar). ``deadband``: the residual deadband of ``stand_net`` (0 = off;
    any other controller refuses one > 0)."""
    import torch  # noqa: PLC0415

    if controller not in CONTROLLERS:
        raise ValueError(f"controller must be one of {CONTROLLERS}, got {controller!r}")
    if SH.check_deadband(deadband) and controller != DEADBAND_CONTROLLER:
        raise ValueError(f"a residual deadband ({deadband}) runs on {DEADBAND_CONTROLLER} only, not {controller}")
    h = obs.clone()
    h[:, p0:p1] = s1_est.inference(h[:, :num_prop])
    a_s1 = s1.act_inference(h, hist_encoding=True)
    extra: dict = {"a_s1": a_s1}
    if controller == "s1":
        return SH.s1_clamp(a_s1), extra
    if controller == "posture_only":
        a_stand, q = SH.stand_action(torch.zeros_like(a_s1), hs.h_lim)
        extra["q18"] = q
        return SH.route(hs.stand_eff, a_stand, a_s1), extra
    if controller == "r_h":
        lab = SH.rh_label(h, hs.h_lim, check=False)
        extra.update(r_target=lab.target, r_branch=lab.branch, q18=lab.q18, r_label_excess=SH.label_excess(lab.label))
        return SH.route(hs.stand_eff, lab.target, a_s1), extra
    if st_pol is None or st_est is None:
        raise ValueError(f"controller {controller} needs st_pol and st_est")
    if controller == "stand_net":
        x = SH.stand_net_input(h, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=mask, h_cmd_limited=hs.h_lim,
                               restoring=hs.restoring)
        a_net = st_pol.act_inference(x, hist_encoding=True).to(a_s1.dtype)
        out = SH.router_actions(hs.stand_eff, a_net, a_s1, hs.h_lim, deadband=deadband)    # 0: no op
        extra.update(a_net=a_net, q18=out.q18)
        return out.actions, extra
    # amask_old: the predecessor's input (no column-15 write), clamp(a_net, +-1) with no posture, the latched slot
    x = SH.SN.stand_net_input(h, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=1)
    a_net = st_pol.act_inference(x, hist_encoding=True).to(a_s1.dtype)
    a_stand, _q = SH.stand_action(a_net, SH.H_SETTLED)
    extra["a_net"] = a_net
    return SH.route(stand_slot.bool(), a_stand, a_s1), extra


def stand_ckpt_rule(controller: str, sha256: str, infos: dict | None, *, mask: int, s1_sha256: str) -> dict:
    """How a stand checkpoint loads for ``controller`` (fail closed: raises ValueError with every reason):
    ``amask_old`` takes exactly A-mask s17; ``stand_net`` takes a campaign-trained checkpoint (``infos.sh_dagger``,
    loaded as saved) or a predecessor one (``infos.sn_dagger``, W[:, 15] zeroed: its column 15 was never a height
    input). The tag's ``stand_mask`` (an INT 0 / 1) must equal ``mask`` and its ``s1_sha256`` S1's."""
    infos = infos or {}
    reasons = []
    if controller not in STAND_CONTROLLERS:
        raise ValueError(f"stand_ckpt_rule: {controller} runs no stand checkpoint")
    if controller == "amask_old" and sha256 != SH.STAND_NET_SHA256:
        reasons.append(f"amask_old runs A-mask s17 only (sha {SH.STAND_NET_SHA256[:12]}), got {sha256[:12]}")
    tag_key = "sh_dagger" if isinstance(infos.get("sh_dagger"), dict) else (
        "sn_dagger" if isinstance(infos.get("sn_dagger"), dict) else None)
    if tag_key is None:
        reasons.append("the checkpoint carries neither infos.sh_dagger (campaign) nor infos.sn_dagger (predecessor)")
        tag = {}
    else:
        tag = infos[tag_key]
    if controller == "amask_old" and tag_key == "sh_dagger":
        reasons.append("amask_old must be the predecessor's A-mask s17 (infos.sn_dagger)")
    tm = tag.get("stand_mask")
    if tag_key is not None and (type(tm) is not int or tm != int(mask)):
        reasons.append(f"infos.{tag_key}.stand_mask {tm!r} != the run's stand mask {int(mask)}")
    if tag_key is not None and tag.get("s1_sha256") != s1_sha256:
        reasons.append(f"infos.{tag_key}.s1_sha256 is not S1's {s1_sha256[:12]} (the stand net was trained beside "
                       "another S1)")
    if reasons:
        raise ValueError("stand checkpoint refused: " + "; ".join(reasons))
    return {"tag": tag_key, "zero_w15": tag_key == "sn_dagger", "stand_mask": tm, "arm": tag.get("arm"),
            "seed": tag.get("seed"), "iterations": tag.get("iterations")}


# ===================================================================================== classification (numpy)
def first_run_start(cond, need: int):
    return STC.first_run_start(cond, need)


def feet_on_floor(foot_force_n, foot_z_m):
    """(T, 6) -> (T,) footpads loaded on the floor (>= 20 N, centre z <= 0.15 m; ``sn_table_core.feet_on_floor``)."""
    return STC.feet_on_floor(foot_force_n, foot_z_m)


def window_steps(dt: float) -> int:
    return int(round(STOOD_WINDOW_S / float(dt)))


def steps_needed(window: int) -> int:
    """ceil(0.95 x window) (143 of 150)."""
    return int(math.ceil(STOOD_SHARE * int(window) - 1e-9))


def hip_contact_armed(rise, start: str):
    """(T,) bool: the H5 arming -- every step on a table start; on a ground start from the first step whose belly rise
    is >= 0.10 m (it stays armed)."""
    np = _np()
    r = np.asarray(rise, dtype=np.float64).reshape(-1)
    if start == START_TABLE:
        return np.ones(r.size, dtype=bool)
    if start != START_GROUND:
        raise ValueError(f"hip_contact_armed: start {start!r}")
    return np.maximum.accumulate(r >= ARM_RISE_M - _EPS) if r.size else np.zeros(0, dtype=bool)


def classify(series: dict, dt: float, *, start: str, spawn_lowest_z: float, spawn_root_z: float, h_cmd: float,
             table_top=None, tilt_max: float = FALL_TILT_RAD, contact_n: float = HIP_CONTACT_N) -> dict:
    """K3 STOOD and the reported reads for ONE env (module docstring). ``series`` (all length T, entry t = the state
    AFTER control step t): ``lowest_z``, ``cos_tilt``, ``feet_on_floor`` (count), ``h_meas``, ``hip_force`` (max over
    hips of the contact-history max norm, N), ``cam_abs`` ((T, 6) wrapped |cam| or (T,) worst leg), ``root_z``;
    optional ``h_lim`` (the limiter state) and ``terminated``.

    The STOOD rise clause (1) reads the ROOT's rise above ``spawn_root_z`` (the realised post-reset root z) on >= 95 %
    of the window -- the same aggregation as the feet / height clauses. The lowest-corner rise above
    ``spawn_lowest_z`` stays the hip-contact ARMING signal (H5; the trainer's re-arm rule) and is reported. Why not the
    corner: at the lowest K3 command (top + 0.31) the nominal rise is 0.31 - 0.16375 = 0.146 m; a robot inside the
    4-cm height tolerance, on a floor whose reference (the HIGHEST point under the probe points) sits 1-3 cm above the
    mean under the feet, with a degree of tilt, reads < 0.10 at its lowest corner on some step while standing at its
    command."""
    np = _np()
    low = np.asarray(series["lowest_z"], dtype=np.float64).reshape(-1)
    T = int(low.size)
    cos_t = np.asarray(series["cos_tilt"], dtype=np.float64).reshape(-1)
    feet = np.asarray(series["feet_on_floor"]).reshape(-1)
    hm = np.asarray(series["h_meas"], dtype=np.float64).reshape(-1)
    hip_f = np.asarray(series["hip_force"], dtype=np.float64).reshape(-1)
    cam = np.asarray(series["cam_abs"], dtype=np.float64)
    cam = cam.reshape(T, -1).max(axis=-1) if cam.size else cam.reshape(-1)
    root_z = np.asarray(series["root_z"], dtype=np.float64).reshape(-1)
    term = np.asarray(series.get("terminated", np.zeros(T, dtype=bool))).reshape(-1).astype(bool)
    h_lim = series.get("h_lim")
    h_lim = None if h_lim is None else np.asarray(h_lim, dtype=np.float64).reshape(-1)
    for k, v in (("cos_tilt", cos_t), ("feet_on_floor", feet), ("h_meas", hm), ("hip_force", hip_f), ("cam_abs", cam),
                 ("root_z", root_z), ("terminated", term)) + ((("h_lim", h_lim),) if h_lim is not None else ()):
        if v.size != T:
            raise ValueError(f"classify: {k} has {v.size} entries, lowest_z {T}")
    if start not in (START_TABLE, START_GROUND):
        raise ValueError(f"classify: start {start!r}")
    if (start == START_TABLE) != (table_top is not None):
        raise ValueError("classify: a table start needs table_top, a ground start none")
    w = window_steps(dt)
    need = steps_needed(w)
    rise = low - float(spawn_lowest_z)                  # the lowest corner (arming, lift-off; reported)
    rise_r = root_z - float(spawn_root_z)               # the root (the STOOD rise clause)
    herr = np.abs(hm - float(h_cmd))
    tilt = np.arccos(np.clip(cos_t, -1.0, 1.0))
    reasons: list = []
    win = slice(max(0, T - w), T)
    shares = {"rise": None, "feet4": None, "feet5": None, "feet6": None, "h_tol": None}
    if T < w or T == 0:
        reasons.append(f"episode {T} steps < the {w}-step window")
        rise_ok = feet4_ok = feet5_ok = feet6_ok = h_ok = False
    else:
        nr = int((rise_r[win] >= STOOD_RISE_M - _EPS).sum())
        n4, n5, n6 = (int((feet[win] >= m).sum()) for m in (STOOD_MIN_FEET, 5, 6))
        nh = int((herr[win] <= STOOD_H_TOL_M + _EPS).sum())
        rise_ok = nr >= need
        feet4_ok, feet5_ok, feet6_ok, h_ok = n4 >= need, n5 >= need, n6 >= need, nh >= need
        shares = {"rise": nr / w, "feet4": n4 / w, "feet5": n5 / w, "feet6": n6 / w, "h_tol": nh / w}
        if not rise_ok:
            reasons.append(f"root rise < {STOOD_RISE_M} m on > 5 % of the window ({nr}/{w} steps risen; min "
                           f"{float(rise_r[win].min()):+.3f})")
        if not feet4_ok:
            reasons.append(f"< {STOOD_MIN_FEET} feet on the floor on > 5 % of the window ({n4}/{w} steps)")
        if not h_ok:
            reasons.append(f"|h - h_cmd| > {STOOD_H_TOL_M} m on > 5 % of the window ({nh}/{w} steps within; median "
                           f"{float(np.median(herr[win])):.3f})")
    armed = hip_contact_armed(rise, start)
    hit = hip_f > float(contact_n)
    max_tilt = float(tilt.max()) if T else 0.0
    fall_tilt = bool(max_tilt > float(tilt_max))
    fall_hip = bool((armed & hit).any())
    fell_off = bool(start == START_TABLE and T and float(low.min()) < float(table_top) - FELL_OFF_DROP_M - _EPS)
    terminated = bool(term.any())
    fall = fall_tilt or fall_hip or fell_off
    if fall_tilt:
        reasons.append(f"FALL: tilt {max_tilt:.3f} rad > {tilt_max}")
    if fall_hip:
        reasons.append(f"FALL: armed hip contact > {contact_n:g} N")
    if fell_off:
        reasons.append(f"FELL OFF THE TABLE: lowest point {float(low.min()) - float(table_top):+.3f} m vs the top")
    if terminated:
        reasons.append("terminated before the end")
    clean = not fall and not terminated
    stood = rise_ok and feet4_ok and h_ok and clean
    i_lift = first_run_start(rise > LIFT_RISE_M, max(1, int(round(LIFT_HOLD_S / dt))))
    arr = np.nonzero(herr <= ARRIVE_TOL_M + _EPS)[0]
    lim_arr = None
    if h_lim is not None:
        hit_lim = np.nonzero(np.abs(h_lim - float(h_cmd)) <= 1e-9)[0]
        lim_arr = None if hit_lim.size == 0 else int(hit_lim[0])
    arm_idx = np.nonzero(armed)[0]
    pre = ~armed
    stood_from = None
    if stood:
        good = (rise_r >= STOOD_RISE_M - _EPS) & (feet >= STOOD_MIN_FEET) & (herr <= STOOD_H_TOL_M + _EPS)
        idx = np.arange(T)
        suffix = np.cumsum(good[::-1].astype(np.int64))[::-1]
        cand = good & (suffix >= STOOD_SHARE * (T - idx) - 1e-9)
        stood_from = (int(np.nonzero(cand)[0][0]) + 1) * dt if cand.any() else None
    homed_n = max(1, int(round(HOMED_LAST_S / dt)))
    cam_last = float(cam[-homed_n:].max()) if T else None
    i_settle = first_run_start(cam <= HOMED_RAD + _EPS, max(1, int(round(SETTLE_HOLD_S / dt))))
    wv = herr[win] if T else herr
    rise_only = bool(not rise_ok and feet4_ok and h_ok and clean)
    return {
        "stood": bool(stood), "stood_5down": bool(rise_ok and feet5_ok and h_ok and clean),
        "stood_6down": bool(rise_ok and feet6_ok and h_ok and clean), "reasons": reasons,
        "clauses": {"rise": bool(rise_ok), "feet4": bool(feet4_ok), "h_tol": bool(h_ok), "no_fall": not fall,
                    "not_terminated": not terminated},
        "rise_only_fail": rise_only,
        "window_shares": shares, "fall": bool(fall), "fall_tilt": fall_tilt, "fall_hip_contact": fall_hip,
        "fell_off_table": fell_off, "terminated": terminated, "max_tilt_rad": max_tilt,
        "hip_contact_armed_s": None if arm_idx.size == 0 else float((int(arm_idx[0]) + 1) * dt),
        "hip_contact_pre_arm_steps": int((pre & hit).sum()),
        "hip_force_pre_arm_max_n": float(hip_f[pre].max()) if pre.any() else None,
        "hip_force_armed_max_n": float(hip_f[armed].max()) if armed.any() else None,
        "rise_min_window_m": float(rise_r[win].min()) if T else None,
        "rise_end_m": float(rise_r[-1]) if T else None,
        "rise_corner_min_window_m": float(rise[win].min()) if T else None,
        "rise_corner_end_m": float(rise[-1]) if T else None,
        "h_err_window_median_m": float(np.median(wv)) if wv.size else None,
        "h_err_window_p90_m": float(np.percentile(wv, 90)) if wv.size else None,
        "h_err_window_max_m": float(wv.max()) if wv.size else None,
        "h_meas_end_m": float(hm[-1]) if T else None, "h_cmd_m": float(h_cmd),
        "limiter_arrived_s": None if lim_arr is None else float((lim_arr + 1) * dt),
        "limiter_arrived_before_window": None if h_lim is None else bool(lim_arr is not None and lim_arr <= T - w),
        "within_2cm_first_s": None if arr.size == 0 else float((int(arr[0]) + 1) * dt),
        "lift_off_s": None if i_lift is None else float((i_lift + 1) * dt),
        "standing_since_s": stood_from,
        "final_root_z_m": float(root_z[-1]) if T else None,
        "final_feet_on_floor": int(feet[-1]) if T else None,
        "cam_max_last5_rad": cam_last, "homed": bool(cam_last is not None and cam_last <= HOMED_RAD + _EPS),
        "settled_s": None if i_settle is None else float(i_settle * dt), "settled": i_settle is not None,
        "cams_homed_end": bool(T and float(cam[-1]) <= HOMED_RAD + _EPS),
        "steps": T,
    }


def _stats(vals) -> dict:
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return {"n": 0, "median": None, "p90": None, "min": None, "max": None, "mean": None}
    mid = len(v) // 2
    med = v[mid] if len(v) % 2 else 0.5 * (v[mid - 1] + v[mid])
    p90 = v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)]
    return {"n": len(v), "median": med, "p90": p90, "min": v[0], "max": v[-1], "mean": sum(v) / len(v)}


def aggregate(per_env: list) -> dict:
    """One cell's block over its envs' :func:`classify` results (homed-at-end per cell = ``homed``)."""
    n = len(per_env)

    def cnt(key):
        return sum(1 for r in per_env if r.get(key))

    return {
        "n_envs": n, "n_stood": cnt("stood"), "stood_share": (cnt("stood") / n) if n else None,
        "n_stood_5down": cnt("stood_5down"), "n_stood_6down": cnt("stood_6down"),
        "falls": cnt("fall"), "falls_tilt": cnt("fall_tilt"), "falls_hip_contact": cnt("fall_hip_contact"),
        "fell_off_table": cnt("fell_off_table"), "terminated": cnt("terminated"),
        "n_fail_rise_only": cnt("rise_only_fail"),
        "falls_per100": (100.0 * cnt("fall") / n) if n else None,
        "homed": cnt("homed"), "homed_share": (cnt("homed") / n) if n else None, "settled": cnt("settled"),
        "cams_homed_end": cnt("cams_homed_end"),
        "limiter_arrived_before_window": cnt("limiter_arrived_before_window"),
        "hip_contact_pre_arm_envs": sum(1 for r in per_env if r.get("hip_contact_pre_arm_steps")),
        "h_err_window_median_m": _stats(r.get("h_err_window_median_m") for r in per_env),
        "h_err_window_p90_m": _stats(r.get("h_err_window_p90_m") for r in per_env),
        "rise_min_window_m": _stats(r.get("rise_min_window_m") for r in per_env),
        "rise_corner_min_window_m": _stats(r.get("rise_corner_min_window_m") for r in per_env),
        "rise_share": _stats((r.get("window_shares") or {}).get("rise") for r in per_env),
        "within_2cm_first_s": _stats(r.get("within_2cm_first_s") for r in per_env),
        "lift_off_s": _stats(r.get("lift_off_s") for r in per_env),
        "max_tilt_rad": _stats(r.get("max_tilt_rad") for r in per_env),
        "h_cmd_m": _stats(r.get("h_cmd_m") for r in per_env),
        "feet4_share": _stats((r.get("window_shares") or {}).get("feet4") for r in per_env),
        "feet5_share": _stats((r.get("window_shares") or {}).get("feet5") for r in per_env),
        "feet6_share": _stats((r.get("window_shares") or {}).get("feet6") for r in per_env),
        "fail_reasons": sorted({x.split(":")[0].split(" (")[0] for r in per_env for x in r.get("reasons", [])}),
    }


# ===================================================================================== merge + the K3 read
CELL_KIND = "sh_table_cell"


def cell_record(meta: dict, cell: str, plan_rows: list, per_env: list) -> dict:
    """The per-cell JSON (``<prefix>__<cell>.json``)."""
    surface, pose = parse_cell(cell)
    return {"kind": CELL_KIND, "meta": {**meta, "cell": cell, "surface": surface, "pose": pose,
                                        "top_m": SURFACE_TOP[surface], "start": start_kind(surface),
                                        "gated": cell_gated(cell)},
            "aggregate": aggregate(per_env), "plan": plan_rows, "per_env": per_env,
            "stood_definition": STOOD_DEFINITION}


def _pool(rows: list) -> dict:
    n = sum(r["n_envs"] for r in rows)
    stood = sum(r["n_stood"] for r in rows)
    falls = sum(r["falls"] for r in rows)
    return {"n_envs": n, "n_stood": stood, "stood_share": (stood / n) if n else None, "falls": falls,
            "falls_per100": (100.0 * falls / n) if n else None,
            "n_stood_5down": sum(r.get("n_stood_5down", 0) for r in rows),
            "n_stood_6down": sum(r.get("n_stood_6down", 0) for r in rows),
            "homed": sum(r.get("homed", 0) for r in rows),
            "n_fail_rise_only": sum(r.get("n_fail_rise_only", 0) for r in rows), "cells": len(rows)}


def pooled_views(cells: dict) -> dict:
    """``cells`` = {cell: aggregate} of one (seed, label) -> GATED ``tables`` (the gated cells present, pooled) and
    ``per_top`` (each gated top pooled over the gated poses); REPORTED ``ground`` (ground cells pooled),
    ``per_top_all_poses`` (every table top pooled over every pose present), ``ground_per_pose`` and ``cells`` (every
    cell present, with its ``gated`` flag); the gated / reported / missing-gated cell lists."""
    for c in cells:
        parse_cell(c)
    out = {"tables": None, "ground": None, "per_top": {}, "per_top_all_poses": {}, "ground_per_pose": {},
           "cells": {}, "gated_cells": [c for c in K3_GATED_CELLS if c in cells],
           "reported_cells": [c for c in K3_CELLS if c in cells and not cell_gated(c)],
           "missing_gated_cells": [c for c in K3_GATED_CELLS if c not in cells]}
    tab_ids = [c for c in K3_CELLS if c in cells and cell_gated(c)]
    gnd_ids = [c for c in K3_CELLS if c in cells and parse_cell(c)[0] == "ground"]
    # ``cell_ids``: which cells a pooled start type holds (k3_read compares cell SETS, not counts)
    out["tables"] = dict(_pool([cells[c] for c in tab_ids]), cell_ids=tab_ids) if tab_ids else None
    out["ground"] = dict(_pool([cells[c] for c in gnd_ids]), cell_ids=gnd_ids) if gnd_ids else None
    for s in K3_GATED_SURFACES:
        rows = [a for c, a in cells.items() if cell_gated(c) and parse_cell(c)[0] == s]
        if rows:
            out["per_top"][s] = _pool(rows)
    for s in SURFACES[:-1]:
        rows = [a for c, a in cells.items() if parse_cell(c)[0] == s]
        if rows:
            out["per_top_all_poses"][s] = _pool(rows)
    for c in K3_CELLS:
        if c not in cells:
            continue
        s, p = parse_cell(c)
        if s == "ground":
            out["ground_per_pose"][p] = _pool([cells[c]])
        out["cells"][c] = dict(_pool([cells[c]]), gated=cell_gated(c))
    return out


def k3_line(subj: dict, ref: dict) -> dict:
    """PLAN H7 K3 on one start type: rate >= ref - max(5 pp, 2 SE of the difference); falls per 100 <= max(1, ref's)."""
    ns, nr = subj["n_envs"], ref["n_envs"]
    if not ns or not nr:
        return {"pass": None, "why": "no envs"}
    ps, pr = subj["n_stood"] / ns, ref["n_stood"] / nr
    se = math.sqrt(ps * (1 - ps) / ns + pr * (1 - pr) / nr)
    line = pr - max(K3_RATE_PP, K3_SE_K * se)
    fs, fr = 100.0 * subj["falls"] / ns, 100.0 * ref["falls"] / nr
    fline = max(K3_FALLS_FLOOR_PER100, fr)
    rate_ok, falls_ok = ps >= line - _EPS, fs <= fline + _EPS
    return {"rate": ps, "ref_rate": pr, "se_diff": se, "rate_line": line, "rate_pass": rate_ok,
            "falls_per100": fs, "ref_falls_per100": fr, "falls_line_per100": fline, "falls_pass": falls_ok,
            "pass": bool(rate_ok and falls_ok), "n": ns, "ref_n": nr}


def k3_read(pooled: dict, ref_label: str = REF_CONTROLLER) -> dict:
    """``pooled`` = {seed: {label: pooled_views}} -> {seed: {label: {start type: k3_line vs ref}}}; the reference's own
    line is omitted. A start type whose cell SET (``cell_ids``) differs from the reference's is refused (``pass``
    None); the gated ``tables`` line is also refused unless both sides hold exactly :data:`K3_GATED_CELLS`. Every row
    carries ``gated``: ``tables`` (the 6 gated cells) decides; ``ground`` is the same line REPORTED, never a verdict."""
    out: dict = {}
    for seed, labels in pooled.items():
        ref = labels.get(ref_label)
        out[seed] = {}
        for label, pv in labels.items():
            if label == ref_label:
                continue
            row = {}
            for st in START_TYPES + REPORTED_START_TYPES:
                s, r = pv.get(st), (ref or {}).get(st)
                gap = ([] if s is None or r is None or st not in START_TYPES else
                       [f"{who} missing {sorted(set(K3_GATED_CELLS) - set(x.get('cell_ids') or []))}"
                        for who, x in (("subject", s), ("ref", r))
                        if sorted(x.get("cell_ids") or []) != sorted(K3_GATED_CELLS)])
                if s is None or r is None:
                    row[st] = {"pass": None, "why": f"missing {'reference' if r is None else 'subject'} cells"}
                elif gap:
                    row[st] = {"pass": None, "why": "gated cell set incomplete (" + "; ".join(gap) + ")"}
                elif s.get("cell_ids", s["cells"]) != r.get("cell_ids", r["cells"]):
                    row[st] = {"pass": None, "why": f"cell sets differ ({s.get('cell_ids', s['cells'])} vs ref "
                                                    f"{r.get('cell_ids', r['cells'])})"}
                else:
                    row[st] = k3_line(s, r)
                row[st]["gated"] = st in START_TYPES
            out[seed][label] = row
    return out


def _floor_ok(v: dict) -> bool:
    return bool(v["stood_share"] is not None and v["stood_share"] >= RH_FLOOR_STOOD - _EPS
                and v["falls_per100"] <= RH_FLOOR_FALLS_PER100 + _EPS)


def rh_floor(pv: dict) -> dict:
    """The step-4 R_h stand-up floor (PLAN step 4, re-scoped at the step-4 stop): STOOD >= 0.90 and falls <= 1 per 100
    on each GATED top pooled over the gated poses (``top <surface>``, ``gated`` True). Every reported cell present is
    listed beside it (``cell <cell>``, ``gated`` False, its floor read for information only). ``_all_gated_pass``:
    every gated cell present and every gated top passes; ``_complete`` / ``_missing_gated_cells`` say whether the
    gated set was whole. ``pv`` = R_h's pooled_views."""
    out = {}
    for s, v in (pv.get("per_top") or {}).items():
        out[f"top {s}"] = {**v, "floor_pass": _floor_ok(v), "gated": True,
                           "poses": [p for p in K3_GATED_POSES if f"{s}{CELL_SEP}{p}" in (pv.get("cells") or {})]}
    for c, v in (pv.get("cells") or {}).items():
        if not v.get("gated"):
            out[f"cell {c}"] = {**v, "floor_pass": _floor_ok(v), "gated": False}
    missing = list(pv.get("missing_gated_cells", K3_GATED_CELLS))
    out["_complete"] = not missing
    out["_missing_gated_cells"] = missing
    out["_all_gated_pass"] = bool(not missing and all(
        v["floor_pass"] for k, v in out.items() if not k.startswith("_") and v["gated"]))
    return out


def merge_cells(paths) -> dict:
    """``{"cells": {seed: {label: {cell: aggregate + run}}}, "pooled", "k3", "rh_floor", "videos", "deadbands"}`` from
    per-cell JSONs (video runs listed apart; a duplicate (seed, label, cell) is refused; a label whose cells -- any seed,
    videos included -- record different residual deadbands (``sh_core.recorded_deadband``: absent = 0) is refused: one
    controller label never pools two action paths)."""
    cells: dict = {}
    videos = []
    deadbands: dict = {}
    for p in sorted(Path(x) for x in paths):
        d = json.loads(Path(p).read_text())
        if d.get("kind") != CELL_KIND:
            continue
        m = d.get("meta") or {}
        try:
            db = SH.recorded_deadband(m)
        except ValueError as exc:
            raise ValueError(f"merge_cells: {p}: {exc}") from exc
        deadbands.setdefault(m.get("label"), {}).setdefault(db, []).append(str(p))
        if len(deadbands[m.get("label")]) > 1:
            got = {k: v[:2] for k, v in sorted(deadbands[m.get("label")].items())}
            raise ValueError(f"merge_cells: label {m.get('label')!r} mixes residual deadbands {sorted(got)} ({got}); "
                             "give each deadband its own --label")
        if m.get("video"):
            videos.append({"label": m.get("label"), "cell": m.get("cell"), "seed": m.get("seed"), "mp4": m.get("mp4"),
                           "env0": (d.get("per_env") or [{}])[0], "run": str(p), SH.RESIDUAL_DEADBAND_KEY: db})
            continue
        seed, label, cell = str(m.get("seed")), m.get("label"), m.get("cell")
        slot = cells.setdefault(seed, {}).setdefault(label, {})
        if cell in slot:
            raise ValueError(f"merge_cells: duplicate {seed} / {label} / {cell} ({slot[cell]['run']} and {p})")
        slot[cell] = {**d["aggregate"], "run": str(p), "smoke": bool(m.get("smoke")), "episode_s": m.get("episode_s"),
                      "controller": m.get("controller"), "stand_sha256": m.get("stand_sha256"),
                      SH.RESIDUAL_DEADBAND_KEY: db}
    pooled = {s: {lab: pooled_views(c) for lab, c in labs.items()} for s, labs in cells.items()}
    floor = {s: rh_floor(labs[REF_CONTROLLER]) for s, labs in pooled.items() if REF_CONTROLLER in labs}
    return {"cells": cells, "pooled": pooled, "k3": k3_read(pooled), "rh_floor": floor, "videos": videos,
            "stood_definition": STOOD_DEFINITION, "k3_episode_s": k3_episode_s(), "gating": gating_record(),
            "deadbands": {str(lab): next(iter(v)) for lab, v in deadbands.items()}}


def gating_record() -> dict:
    """Which K3 cells decide and which are reported (the merge summary and the facts record carry it)."""
    return {"gated_cells": list(K3_GATED_CELLS), "reported_cells": list(K3_REPORTED_CELLS),
            "gated_tops_m": list(K3_TOPS_M), "gated_poses": list(K3_GATED_POSES),
            "reported_tops_m": list(K3_REPORTED_TOPS_M), "gated_start_types": list(START_TYPES),
            "reported_start_types": list(REPORTED_START_TYPES),
            "rule": "CHARTER section 3, USER DECISION at the step-4 stop (narrow stand-up to tall tables): the K3 line "
                    "(tables pooled over the 6 gated cells per env seed) and the R_h floor (per gated top pooled over "
                    "the gated poses) read the gated cells only; the reported cells (the 18-in table, every rod_min "
                    "cell, every ground cell) are run and read, never gated"}


def _f(v, spec: str, width: int) -> str:
    return f"{'-':>{width}s}" if v is None else f"{v:{width}{spec}}"


def format_table(merged: dict) -> str:
    lines = [f"K3 GATED cells ({len(K3_GATED_CELLS)}; decide the K3 line, the R_h floor and its STOP): "
             + ", ".join(K3_GATED_CELLS),
             f"K3 REPORTED cells ({len(K3_REPORTED_CELLS)}; run and read, never gated): " + ", ".join(K3_REPORTED_CELLS),
             "",
             f"{'seed':4s} {'label':22s} {'cell':28s} {'gate':8s} {'envs':>4s} {'STOOD':>8s} {'5dn':>4s} {'6dn':>4s} "
             f"{'|dh| med':>8s} {'p90':>6s} {'falls':>5s} {'tilt':>4s} {'hip':>4s} {'off':>4s} {'homed':>5s} "
             f"{'rise-only':>9s}"]
    dbs = {lab: d for lab, d in (merged.get("deadbands") or {}).items() if d}
    if dbs:
        lines.insert(2, "residual deadband (sh_core.apply_deadband; every other label 0): "
                     + ", ".join(f"{lab} {d:g}" for lab, d in sorted(dbs.items())))
    for seed in sorted(merged.get("cells", {})):
        for label in sorted(merged["cells"][seed]):
            for cell in K3_CELLS:
                r = merged["cells"][seed][label].get(cell)
                if r is None:
                    continue
                lines.append(
                    f"{seed:4s} {label:22s} {cell:28s} {'GATED' if cell_gated(cell) else 'reported':8s} "
                    f"{r['n_envs']:4d} {r['n_stood']:3d}/{r['n_envs']:<4d} "
                    f"{r['n_stood_5down']:4d} {r['n_stood_6down']:4d} "
                    f"{_f(r['h_err_window_median_m']['median'], '.3f', 8)} "
                    f"{_f(r['h_err_window_p90_m']['p90'], '.3f', 6)} {r['falls']:5d} {r['falls_tilt']:4d} "
                    f"{r['falls_hip_contact']:4d} {r['fell_off_table']:4d} {r['homed']:5d} "
                    f"{r.get('n_fail_rise_only', 0):9d}")
    lines.append("")
    lines.append(f"{'seed':4s} {'label':22s} {'start':8s} {'gate':8s} {'STOOD':>12s} {'falls/100':>9s} "
                 f"{'K3 vs r_h':>10s} {'line':>6s} {'falls line':>10s}")
    for seed in sorted(merged.get("pooled", {})):
        for label, pv in sorted(merged["pooled"][seed].items()):
            for st in START_TYPES + REPORTED_START_TYPES:
                v = pv.get(st)
                if v is None:
                    continue
                k = ((merged.get("k3") or {}).get(seed) or {}).get(label, {}).get(st) or {}
                gated = st in START_TYPES
                verdict = "ref" if label == REF_CONTROLLER else {True: "PASS", False: "FAIL", None: "-"}[k.get("pass")]
                if not gated and label != REF_CONTROLLER:
                    verdict = verdict.lower()                      # a reported line is never a verdict
                lines.append(f"{seed:4s} {label:22s} {st:8s} {'GATED' if gated else 'reported':8s} "
                             f"{v['n_stood']:5d}/{v['n_envs']:<6d} "
                             f"{_f(v['falls_per100'], '.2f', 9)} {verdict:>10s} {_f(k.get('rate_line'), '.3f', 6)} "
                             f"{_f(k.get('falls_line_per100'), '.2f', 10)}")
    for seed, fl in sorted((merged.get("rh_floor") or {}).items()):
        rows = [(k, v) for k, v in fl.items() if not k.startswith("_")]
        if fl.get("_all_gated_pass"):
            end = "PASS"
        elif not fl.get("_complete", True):
            end = f"INCOMPLETE (gated cells missing: {', '.join(fl.get('_missing_gated_cells') or [])})"
        else:
            end = "STOP (step-4 premise)"
        lines.append(f"R_h floor es{seed} (GATED: each gated top pooled over {'/'.join(K3_GATED_POSES)}): " + ", ".join(
            f"{k} {v['n_stood']}/{v['n_envs']} falls {v['falls']} rise-only {v.get('n_fail_rise_only', 0)} "
            f"{'ok' if v['floor_pass'] else 'MISS'}" for k, v in rows if v["gated"]) + f" -> {end}")
        rep = [(k, v) for k, v in rows if not v["gated"]]
        if rep:
            lines.append(f"R_h reported es{seed} (never gated): " + ", ".join(
                f"{k} {v['n_stood']}/{v['n_envs']} falls {v['falls']} 5dn {v.get('n_stood_5down', 0)} "
                f"6dn {v.get('n_stood_6down', 0)} homed {v.get('homed', 0)}" for k, v in rep))
    return "\n".join(lines)


# ===================================================================================== the facts record
def build_facts(cells=None) -> dict:
    """What the test rests on (the launcher's ``facts.json`` and every run's summary)."""
    cells = list(K3_CELLS if cells is None else cells)
    poses = {}
    for p in FIXED_POSES:
        fp = fixed_pose(p)
        lp = lowest_parts(fp["hip_rad"], fp["knee_left_rad"])
        poses[p] = {"hip_deg": fp["hip_deg"], "knee_left_deg": fp["knee_left_deg"], "hip_binding": fp["hip_binding"],
                    "knee_binding": fp["knee_binding"], "lowest_parts_z_body_m": lp["parts_z"],
                    "first_contact_on_ground": lp["first_contact"],
                    "ground_spawn_root_rel_m": ground_root_rel(fp["hip_rad"], fp["knee_left_rad"]),
                    "table_top_min_m": table_top_min(fp["hip_rad"], fp["knee_left_rad"]),
                    "rods_m": rod_lengths(fp["hip_rad"], fp["knee_left_rad"])}
    hips = random_hip_grid()
    poses["random"] = {"hip_grid_deg": [math.degrees(h) for h in hips],
                       "knee_range_deg_at_grid": [[math.degrees(x) for x in random_knee_range(h)] for h in hips],
                       "rule": "hip uniform on the 0.5-deg grid, knee uniform in [k_lo, k_hi] at that hip; on a table "
                               "rejection-sampled until legs >= 2 cm above the floor and tibias clear of the table"}
    surf = {}
    for s in SURFACES:
        top = SURFACE_TOP[s]
        surf[s] = {"top_m": top, "start": start_kind(s), "gated_poses": [p for p in POSES if
                                                                           cell_gated(f"{s}{CELL_SEP}{p}")],
                   "table_center_above_floor_m": table_center_z(top),
                   "spawn_root_above_floor_m": None if top is None else table_spawn_root_z(top),
                   "cmd_rule": (f"[top + {CMD_MARGIN_EVAL_M}, h_max]" if top is not None else
                                f"[max(h_min, spawn + {GROUND_CMD_RISE_M:.5f}), h_max]"),
                   "cmd_lo_m": None if top is None else top + CMD_MARGIN_EVAL_M}
    return {"cells": cells, "k3_cells": list(K3_CELLS), "envs_per_cell": ENVS_PER_CELL, "surfaces": surf,
            "poses": poses, "controllers": list(CONTROLLERS), "h_min": SH.H_MIN, "h_max": SH.H_MAX,
            "k3_tops_m": list(K3_TOPS_M), "k3_reported_tops_m": list(K3_REPORTED_TOPS_M),
            "k3_gated_cells": list(K3_GATED_CELLS), "k3_reported_cells": list(K3_REPORTED_CELLS),
            "gating": gating_record(), "train_top_range_m": [TOP_MIN_M, TOP_MAX_TRAIN_M],
            "k3_episode_s": k3_episode_s(),
            "floor_rule": "floor reference = the highest terrain point under the pose's pivot / knee / toe points and "
                          "the body corners / centre (raycast at spawn); table top and root z are above it",
            "table": {"size_m": [TABLE_X_M, TABLE_Y_M, TABLE_HEIGHT_M],
                                                      "park_center_z_m": PARK_CENTER_Z_M,
                                                      "jitter": {"xy_m": JITTER_XY_M, "yaw_deg": JITTER_YAW_DEG}},
            "stood_definition": STOOD_DEFINITION,
            "k3_read": {"rate_pp": K3_RATE_PP, "se_k": K3_SE_K, "falls_floor_per100": K3_FALLS_FLOOR_PER100,
                        "rh_floor": {"stood": RH_FLOOR_STOOD, "falls_per100": RH_FLOOR_FALLS_PER100}},
            "clip": SH.clip_record(), "provenance": {"sh_core": SH.provenance()}}
