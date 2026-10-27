#!/usr/bin/env python3
"""Stand-network campaign -- the TABLE STAND-UP test's pure pieces (2026-10-13; ``sn_table_test.py`` runs them under
Kit, ``tests/unit/test_sn_table.py`` pins them on CPU). TEST ONLY: nothing here is read by a gate, a ledger or a reader.

TORCH- / NUMPY-FREE AT IMPORT (the campaign launcher's rule L6): the Kit script loads this file before AppLauncher; every
torch / numpy / linkage import happens inside the function that needs it. The linkage, dimensions and foot-FK modules
are the forward task's OWN files (``crab_hex_forward_task/mdp``), loaded by path (never through the task package, whose
``__init__`` pulls in Isaac).

WHAT THE TEST RESTS ON (all computed here, written into ``facts.json`` and every run's summary):

* RETRACTED START POSE (two readings; ``POSES``):
  - ``rod_min`` -- BOTH lead screws at their minimum rod length (0.450 m each). The hip screw's minimum maps to a hip
    angle of -48.7 deg, beyond the USD hard stop of -45 deg (``crab_hex_dimensions.HIP_SIM_LIMITS_DEG``; hardware "45
    deg from vertical-up = femur raised / retracted"), so the STOP binds: hip = -45 + 0.5 deg margin = -44.5 deg
    (rod 0.4572 m, 7 mm off the screw's end). The knee screw's minimum, re-solved at that hip, is a knee angle of
    -41.6 deg (left convention; right legs mirror the sign): it OPENS the knee -- the tibia swings OUTBOARD to almost
    horizontal (interior femur-tibia angle 131.6 deg; hardware "140 deg = extended"). Lead-screw-retracted legs are
    therefore splayed out sideways like wings, toes ~1.9 m lateral and ~0.11 m ABOVE the body centre (above the table
    top), not tucked.
  - ``knee_folded`` -- the hardware's own "retracted" words (``crab_hex_dimensions``: hip 45 deg from up = raised /
    retracted, knee interior 5 deg = retracted / folded): hip at the same stop, knee folded INBOARD as far as the SIM
    MODEL allows. The hardware fold (sim +85 deg, the knee stop) is NOT representable: in ``assets/crab.usda`` the hip
    plate, femur and tibia are COPLANAR (same leg-plane offset; the real parts sit side by side) and the hip-tibia pair
    is not joint-adjacent, so with self-collisions on the folded tibia would spawn inside the hip plate. The fold is
    the largest knee angle whose tibia + footpad hull keeps >= ``PLATE_CLEARANCE_M`` from the plate's collision hull
    and from the body box (:func:`max_clear_knee_fold`); the knee screw is then near its MAXIMUM (folding = extending
    the knee screw). Toes hang ~0.5 m below the body centre, i.e. below the table top but outside its footprint.
  - both: Body_Hip and CamShaft at 0 (= their defaults: ``hip_to_cam_shaft_default(0) = 0``), joint velocities 0.
* SPAWN: the root link is ``chassis/body``, a 0.7112 x 1.2192 x 0.3175 m box centred on the root frame, so the belly is
  ``0.15875`` m below the root; root z = table top 0.635 + 0.15875 + ``SPAWN_CLEARANCE_M`` (0.005) = 0.79875 m (the
  robot drops 5 mm and settles on its belly). Standing settled root is 1.064 m: the default pose clears the table top by
  ~0.27 m.
* TABLE: a kinematic cuboid per env, ``TABLE_X_M`` x ``TABLE_Y_M`` (x = body forward, y = lateral), top at 0.635 m
  (25 in), base sunk to -0.05 m (no gap over the +-3 cm ground roughness), surface material = the terrain's. The legs
  mount on the 1.2192-m-wide body's SIDE faces (yaw axes at |y| = 0.6096, the hip plates outboard of them), so a 0.70 m
  wide table (|y| <= 0.35, <= 0.45 at a jittered corner) leaves >= 0.14 m between the table and any leg part
  (:func:`table_leg_clearance`); the belly overhangs the table by ~0.26 m on each side.
* R FROM THIS START (``mdp/crab_hex_omni_restance.py``, the D3 constants): every hip reads ~ -0.80 rad below its
  default, far below ``p_down`` = -0.05 rad and ``p_up`` = -0.0625 rad, so all six legs are "UP, not DOWN"; with the cams
  at 0 (|c| <= eps_air) no leg is HOME, so every leg is SEAT -- rods 0 (the default targets), cam 0: R's action is
  EXACTLY ZERO (:func:`r_start_analysis`). A leg switches to HOLD (rods 0, cam trim -clip(k_h c, +-0.1)) once its hip is
  within 0.05 rad of the default; R steps (LIFT) only a DOWN leg with |cam| > eps_step 0.3, which nothing here drives.
  R is therefore "go to the default pose and hold" from this start -- the zero controller plus tiny cam trims.
* ACTION REACH (every controller): a rod target is ``default + 0.25 * a`` with ``a`` clipped to +-1 by the runner's
  ``clip_actions`` 1.0, converted to a rod length and slewed at the LOADED rod speed (hip 28 mm/s, knee 33 mm/s). The
  start is ~3.2 action units below the hip window, so NO controller can hold (or deepen) the retraction: every action
  pulls every rod toward the +-0.25 rad window at the rod speed (:func:`slew_timeline`). The stand-up is largely the
  actuator layer's slew; the controller only chooses where inside the window the legs go and what the cams do.

STOOD (fixed before any run; :func:`classify`): an env STOOD iff ALL of
  (1) over the LAST ``STOOD_WINDOW_S`` = 3 s (150 control steps): on EVERY step the chassis' lowest point (its oriented
      box) is >= ``STOOD_GAP_M`` = 0.10 m above the table-top plane, AND on >= ``STOOD_FEET_SHARE`` = 95 % of the steps
      (>= 143 of 150; up to 7 flicker steps) >= ``STOOD_MIN_FEET`` = 5 footpads are loaded on the floor (net contact
      force >= ``FOOT_LOADED_N`` = 20 N and footpad centre z <= ``FOOT_FLOOR_Z_M`` = 0.15 m). The feet tolerance is
      there because a standing robot shows post-settle load flickers (the campaign's "post-settle taps": sub-mm,
      ~0.02 s, median lift 0 mm), and two in one step would otherwise classify a standing robot as NOT STOOD; the gap
      clause stays per-step (a body that touches the table again did not stand);
  (2) over the WHOLE episode the body tilt (body z axis vs world z) never exceeds ``FALL_TILT_RAD`` = 0.5 rad (the
      flat-walk ``crab_failure`` limit) -- else FALL;
  (3) the chassis' lowest point never drops more than ``FELL_OFF_DROP_M`` = 0.05 m below the table-top plane -- else
      FELL OFF THE TABLE;
  (4) the env did not terminate (only a time-out remains, sized 2 s past the episode).
Reported beside it, not part of STOOD: ``stood_strict`` (the same with >= 5 feet on EVERY window step), the window
share of steps with >= 5 feet and the window's minimum feet count (so a near-miss shows in the merge table), lift-off
time (the first time from which the belly gap > 0.02 m holds for 0.5 s), final root z / gap / feet on the floor, max
tilt, cams homed at the end (worst |cam| <= 0.10 rad).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: scripts -> 2026-10-11_stand_network -> experiments -> crab_hex_omni_task
TASK = HERE.parents[2]
TASKS = TASK.parent
REPO = HERE.parents[6]
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
LINKAGE_PATH = FWD_MDP / "crab_hex_linkage.py"
DIMS_PATH = FWD_MDP / "crab_hex_dimensions.py"
FOOT_FK_PATH = FWD_MDP / "crab_hex_foot_fk.py"
USD_PATH = REPO / "assets" / "crab.usda"

IN_TO_M = 0.0254
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
LEFT = frozenset({"FL", "ML", "RL"})
CONTROLLERS = ("stand_net", "r", "s1", "zero")
POSES = ("rod_min", "knee_folded")
ROD_SCALE = 0.25            #: rad per action unit on the hip / knee channels (the action term's scale)
ACTION_CLIP = 1.0           #: the runner's clip_actions (deployment)

# ------------------------------------------------------------------ the table
TABLE_TOP_IN = 25.0
TABLE_TOP_M = TABLE_TOP_IN * IN_TO_M             # 0.635
TABLE_X_M = 0.90                                  # along body +x (forward)
TABLE_Y_M = 0.70                                  # lateral
#: the base is sunk below nominal ground (z = 0) so no gap shows over the flat tiles' roughness (<= ~+3 cm)
TABLE_BASE_Z_M = -0.05
TABLE_HEIGHT_M = TABLE_TOP_M - TABLE_BASE_Z_M
TABLE_CENTER_Z_M = 0.5 * (TABLE_TOP_M + TABLE_BASE_Z_M)
TABLE_MASS_KG = 50.0                              # kinematic: never moved by contacts; mass only for the PhysX body
SPAWN_CLEARANCE_M = 0.005
JITTER_XY_M = 0.03
JITTER_YAW_DEG = 10.0

# ------------------------------------------------------------------ the retracted pose
LIMIT_MARGIN_DEG = 0.5                            # spawn this far inside a joint hard stop
PLATE_CLEARANCE_M = 0.02                          # knee_folded: tibia / footpad hull to hip plate / body
FOOTPAD_HALF_M = (0.03, 0.03, 0.02)               # USD footpad scale 0.06 x 0.06 x 0.04 (local x, y, z)

# ------------------------------------------------------------------ STOOD (module docstring)
STOOD_WINDOW_S = 3.0
STOOD_GAP_M = 0.10
STOOD_MIN_FEET = 5
#: share of the window's steps that must carry >= STOOD_MIN_FEET loaded feet (post-settle load flickers)
STOOD_FEET_SHARE = 0.95
FOOT_LOADED_N = 20.0
FOOT_FLOOR_Z_M = 0.15
FALL_TILT_RAD = 0.5
FELL_OFF_DROP_M = 0.05
LIFT_GAP_M = 0.02
LIFT_HOLD_S = 0.5
ON_TABLE_GAP_M = 0.01
HOMED_RAD = 0.10
#: float slack on the STOOD comparisons (a gap of exactly 0.10 m read back through z - 0.635 must pass)
_EPS = 1e-9

STOOD_DEFINITION = (
    "STOOD iff (1) last 3.0 s (150 control steps): on EVERY step the chassis box's lowest point >= 0.10 m above the "
    "table-top plane (z >= 0.735 m), AND on >= 95 % of the steps (>= 143 of 150) >= 5 of 6 footpads loaded on the floor "
    "(net contact >= 20 N, footpad centre z <= 0.15 m; the tolerance absorbs post-settle load flickers); (2) whole "
    "episode: body tilt (body z vs world z) <= 0.5 rad (else FALL); (3) whole episode: the chassis' lowest point never "
    "below the table-top plane by more than 0.05 m (else FELL OFF THE TABLE); (4) no termination (only a time-out 2 s "
    "past the episode remains). Reported, not gated: stood_strict (>= 5 feet on EVERY window step), the window share of "
    "steps with >= 5 feet and the window's min feet, lift-off (belly gap > 0.02 m held 0.5 s), final root z / gap / "
    "feet, max tilt, cams homed at the end (worst |cam| <= 0.10 rad).")

# ------------------------------------------------------------------ leg-plane collision hulls (assets/crab.usda)
# Leg plane: u = outboard distance from the leg's yaw axis (the body side face), z = up from the body centre. The hip
# plate, femur and tibia all sit at the SAME leg-plane offset (pinned against the USD by the tests).
#: hip plate (``<leg>_Hip``, convexHull): a stadium around the vertical core segment u = 0.0635, z in [-0.2413, 0.2413]
#: (the plate is 0.127 wide, its rounded ends reach z = +-0.3048); the femur pivot is the lower end's centre. The USD's
#: end arcs bulge up to 1.1 mm past a true semicircle (max hull-point distance from the core 0.06462 m), so the stadium
#: radius used is 0.0647 -- every hull point inside it (conservative by <= 1.2 mm on the straight sides)
PLATE_CORE_U_M = 0.0635
PLATE_CORE_HALF_Z_M = 0.2413
PLATE_RADIUS_M = 0.0647
#: femur (``<leg>_Femur``): a stadium around the pivot -> knee segment, same 0.0647 hull radius (max 0.06461)
FEMUR_RADIUS_M = 0.0647
#: tibia (``<leg>_Tibia``) outline in its local (y, z) -- the USD mesh's 24 points of one face, verbatim; the knee is at
#: local z = TIBIA_KNEE_Z_M; the footpad centre at the outline's lowest z. Symmetrised in :func:`tibia_hull`
#: (conservative for the one-sided bulge).
TIBIA_KNEE_Z_M = 0.326009
TIBIA_OUTLINE_YZ = (
    (0.0563372, 0.19611594), (0.03439922, -0.24986742), (0.0254, -0.47287688), (0.02408174, -0.48100488),
    (0.01682242, -0.49283112), (0.011557, -0.49652936), (-0.00056896, -0.499491), (-0.01256538, -0.49653698),
    (-0.01767586, -0.49284128), (-0.02442972, -0.4810125), (-0.0254, -0.47287688), (-0.0635, 0.41487598),
    (-0.06100064, 0.43427904), (-0.05409184, 0.45015404), (-0.04365752, 0.46250098), (-0.03057398, 0.47131986),
    (-0.01572768, 0.47661322), (0.0, 0.47837598), (0.01572768, 0.47661322), (0.03057398, 0.47131986),
    (0.04365498, 0.46250098), (0.05409184, 0.45015404), (0.06100064, 0.43427904), (0.0635, 0.41487598),
)
#: body box half extents (x, y, z) from ``crab_hex_dimensions.BODY_SIZE_M`` (pinned by the tests)
BODY_HALF_M = (0.3556, 0.6096, 0.15875)


# ===================================================================================== by-path module loaders
_MODS: dict = {}


def _exec_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"sn_table_core: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def dims():
    """``crab_hex_dimensions`` (stdlib only)."""
    if "dims" not in _MODS:
        _MODS["dims"] = _exec_by_path(DIMS_PATH, "_snt_crab_hex_dimensions")
    return _MODS["dims"]


def linkage():
    """``crab_hex_linkage`` (imports torch). Its ``from . import crab_hex_dimensions`` fails outside the package and
    falls back to ``import crab_hex_dimensions``: the by-path dims instance is lent under that name for the load only."""
    if "linkage" not in _MODS:
        key = "crab_hex_dimensions"
        had, prev = key in sys.modules, sys.modules.get(key)
        sys.modules[key] = dims()
        try:
            _MODS["linkage"] = _exec_by_path(LINKAGE_PATH, "_snt_crab_hex_linkage")
        finally:
            if had:
                sys.modules[key] = prev
            else:
                sys.modules.pop(key, None)
    return _MODS["linkage"]


def foot_fk():
    """``crab_hex_foot_fk`` (numpy; loads its own dims by path)."""
    if "fk" not in _MODS:
        _MODS["fk"] = _exec_by_path(FOOT_FK_PATH, "_snt_crab_hex_foot_fk")
    return _MODS["fk"]


def _t(x):
    import torch  # noqa: PLC0415

    return torch.tensor(float(x), dtype=torch.float64)


# ===================================================================================== joint limits / defaults
def joint_limits_rad() -> dict:
    """The USD hard stops (rad): hip (all legs), knee LEFT convention, knee RIGHT (mirrored)."""
    d = dims()
    return {"hip": tuple(math.radians(v) for v in d.HIP_SIM_LIMITS_DEG),
            "knee_left": tuple(math.radians(v) for v in d.KNEE_SIM_LIMITS_LEFT_DEG),
            "knee_right": tuple(math.radians(v) for v in d.KNEE_SIM_LIMITS_RIGHT_DEG)}


def defaults_rad() -> dict:
    lk = linkage()
    return {"hip": float(lk.hip_default_rad()), "knee_left": float(lk.knee_default_left_rad())}


def spawn_root_z(table_top: float = TABLE_TOP_M, body_half_z: float = BODY_HALF_M[2],
                 clearance: float = SPAWN_CLEARANCE_M) -> float:
    """Root (``chassis/body`` centre) z that puts the belly ``clearance`` above the table top."""
    return float(table_top) + float(body_half_z) + float(clearance)


# ===================================================================================== 2-D convex geometry
def _seg_seg_dist(p, q, r, s) -> float:
    """Distance between segments pq and rs (2-D tuples); 0 if they cross."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1, o2, o3, o4 = orient(p, q, r), orient(p, q, s), orient(r, s, p), orient(r, s, q)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and o1 * o2 < 0 and o3 * o4 < 0:
        return 0.0

    def pt_seg(a, b, c):
        bx, by = b[0] - a[0], b[1] - a[1]
        n2 = bx * bx + by * by
        t = 0.0 if n2 == 0.0 else max(0.0, min(1.0, ((c[0] - a[0]) * bx + (c[1] - a[1]) * by) / n2))
        return math.hypot(a[0] + t * bx - c[0], a[1] + t * by - c[1])

    return min(pt_seg(p, q, r), pt_seg(p, q, s), pt_seg(r, s, p), pt_seg(r, s, q))


def _convex_hull(points):
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside_convex(poly, p) -> bool:
    n = len(poly)
    sign = 0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if c != 0:
            if sign == 0:
                sign = 1 if c > 0 else -1
            elif (c > 0) != (sign > 0):
                return False
    return True


def poly_seg_distance(poly, a, b) -> float:
    """Distance from a convex polygon (CCW / CW list of 2-D points) to segment ab; 0 if they touch / overlap."""
    if _inside_convex(poly, a) or _inside_convex(poly, b):
        return 0.0
    n = len(poly)
    return min(_seg_seg_dist(poly[i], poly[(i + 1) % n], a, b) for i in range(n))


def poly_poly_distance(p1, p2) -> float:
    if any(_inside_convex(p1, q) for q in p2) or any(_inside_convex(p2, q) for q in p1):
        return 0.0
    return min(_seg_seg_dist(p1[i], p1[(i + 1) % len(p1)], p2[j], p2[(j + 1) % len(p2)])
               for i in range(len(p1)) for j in range(len(p2)))


# ===================================================================================== leg-plane links
def leg_plane_points(hip: float, knee: float) -> dict:
    """Leg-plane (u outboard, z up) pivot / knee / toe for hip and knee angles in the LEFT sim convention
    (``crab_hex_foot_fk``'s chain: pivot at (0.0635, -0.2413), femur 0.5842, tibia 0.8255)."""
    d = dims()
    pivot = (float(d.FEMUR_PIVOT_OUTBOARD_M), float(d.FEMUR_PIVOT_Z_M))
    knee_pt = (pivot[0] + d.FEMUR_HINGE_TO_HINGE_M * math.cos(hip), pivot[1] - d.FEMUR_HINGE_TO_HINGE_M * math.sin(hip))
    a = hip + knee
    down = (-math.sin(a), -math.cos(a))           # knee -> toe unit vector
    toe = (knee_pt[0] + d.TIBIA_KNEE_TO_TOE_M * down[0], knee_pt[1] + d.TIBIA_KNEE_TO_TOE_M * down[1])
    return {"pivot": pivot, "knee": knee_pt, "toe": toe, "tibia_down": down}


def tibia_hull(hip: float, knee: float):
    """The tibia + footpad collision hull in the leg plane (symmetrised outline; footpad = its 0.06 x 0.04 section at
    the toe)."""
    lp = leg_plane_points(hip, knee)
    kx, kz = lp["knee"]
    dx, dz = lp["tibia_down"]
    ux, uz = -dx, -dz                     # along the tibia's local +z (toward the top)
    px, pz = dz * -1.0, dx                # perpendicular (the local y axis up to a sign; the outline is symmetrised)
    pts = []
    for y, z in TIBIA_OUTLINE_YZ:
        s = z - TIBIA_KNEE_Z_M
        for yy in (y, -y):
            pts.append((kx + s * ux + yy * px, kz + s * uz + yy * pz))
    toe_s = min(z for _, z in TIBIA_OUTLINE_YZ) - TIBIA_KNEE_Z_M
    for yy in (-FOOTPAD_HALF_M[1], FOOTPAD_HALF_M[1]):
        for zz in (-FOOTPAD_HALF_M[2], FOOTPAD_HALF_M[2]):
            s = toe_s + zz
            pts.append((kx + s * ux + yy * px, kz + s * uz + yy * pz))
    return _convex_hull(pts)


def body_rect():
    """The body box in the leg plane: u in [-2 * half_y, 0] (inboard of the side face), |z| <= half_z."""
    hz = BODY_HALF_M[2]
    return [(-2.0 * BODY_HALF_M[1], -hz), (0.0, -hz), (0.0, hz), (-2.0 * BODY_HALF_M[1], hz)]


def link_clearances(hip: float, knee: float) -> dict:
    """Leg-plane clearances (m; 0 = touching / overlapping) between the NON-adjacent link pairs the articulation's
    self-collision checks: tibia+pad vs hip plate, tibia+pad vs body, femur vs body."""
    tib = tibia_hull(hip, knee)
    a = (PLATE_CORE_U_M, -PLATE_CORE_HALF_Z_M)
    b = (PLATE_CORE_U_M, PLATE_CORE_HALF_Z_M)
    plate = max(0.0, poly_seg_distance(tib, a, b) - PLATE_RADIUS_M)
    body = poly_poly_distance(tib, body_rect())
    lp = leg_plane_points(hip, knee)
    femur_body = max(0.0, poly_seg_distance(body_rect(), lp["pivot"], lp["knee"]) - FEMUR_RADIUS_M)
    return {"tibia_plate_m": plate, "tibia_body_m": body, "femur_body_m": femur_body,
            "min_m": min(plate, body, femur_body)}


def max_clear_knee_fold(hip: float, *, clearance: float = PLATE_CLEARANCE_M, step_rad: float = 1e-3) -> dict:
    """Sweep the knee (LEFT convention) up from its default toward the fold stop (minus the margin) at ``hip``; the
    answer is the last angle before the tibia + footpad hull comes within ``clearance`` of the hip plate or the body
    (the sweep stops at the FIRST violation, so every fold up to the answer is clear)."""
    lim = joint_limits_rad()
    k = defaults_rad()["knee_left"]
    k_hi = lim["knee_left"][1] - math.radians(LIMIT_MARGIN_DEG)
    c0 = link_clearances(hip, k)
    if c0["min_m"] < clearance:
        raise ValueError(f"max_clear_knee_fold: the default knee is already within {clearance} m at hip {hip}")
    best, best_c, binding = k, c0, "joint_stop"
    while k + step_rad <= k_hi + 1e-12:
        k += step_rad
        c = link_clearances(hip, k)
        if c["min_m"] < clearance:
            binding = "plate_clearance" if c["tibia_plate_m"] <= c["tibia_body_m"] else "body_clearance"
            break
        best, best_c = k, c
    return {"knee_rad": best, "clearances": best_c, "binding": binding, "clearance_req_m": clearance,
            "knee_stop_rad": lim["knee_left"][1], "step_rad": step_rad}


# ===================================================================================== the retracted pose
def retracted_pose(pose: str) -> dict:
    """The start pose for ``pose`` in ``POSES`` (module docstring): angles, rods, bindings, the per-joint targets."""
    if pose not in POSES:
        raise ValueError(f"retracted_pose: pose must be one of {POSES}, got {pose!r}")
    lk = linkage()
    lim = joint_limits_rad()
    m = math.radians(LIMIT_MARGIN_DEG)
    dflt = defaults_rad()
    hip_rod_min_angle = float(lk.hip_angle_from_length(_t(lk.HIP_LEN_MIN_M)))
    hip_lo = lim["hip"][0] + m
    hip = max(hip_rod_min_angle, hip_lo)
    hip_binding = "joint_stop" if hip_lo > hip_rod_min_angle else "rod_min"
    knee_rod_min_angle = float(lk.knee_angle_from_length(_t(lk.KNEE_LEN_MIN_M), _t(hip)))
    k_lo, k_hi = lim["knee_left"][0] + m, lim["knee_left"][1] - m
    fold = None
    if pose == "rod_min":
        knee = min(max(knee_rod_min_angle, k_lo), k_hi)
        knee_binding = "rod_min" if k_lo <= knee_rod_min_angle <= k_hi else "joint_stop"
    else:
        fold = max_clear_knee_fold(hip)
        knee = fold["knee_rad"]
        knee_binding = fold["binding"]
    hip_rod = float(lk.hip_actuator_length(_t(hip)))
    knee_rod = float(lk.knee_actuator_length(_t(hip), _t(knee)))
    joints = {}
    for leg in LEGS:
        joints[f"{leg}_Body_Hip_RevoluteJoint"] = 0.0
        joints[f"{leg}_Body_CamShaft_RevoluteJoint"] = 0.0
        joints[f"{leg}_Hip_Femur_RevoluteJoint"] = hip
        joints[f"{leg}_Femur_Tibia_RevoluteJoint"] = knee if leg in LEFT else -knee
    # what the policy would have to command to HOLD this pose (hip: target = default + 0.25 a; knee: the rod command
    # is interpreted at the DEFAULT hip, so the knee angle that gives this knee rod at the default hip)
    knee_cmd_equiv = float(lk.knee_angle_from_length(_t(knee_rod), _t(dflt["hip"])))
    return {
        "pose": pose, "hip_rad": hip, "hip_deg": math.degrees(hip), "hip_binding": hip_binding,
        "hip_rod_min_angle_rad": hip_rod_min_angle, "hip_rod_min_angle_deg": math.degrees(hip_rod_min_angle),
        "hip_stop_rad": lim["hip"][0], "knee_left_rad": knee, "knee_left_deg": math.degrees(knee),
        "knee_right_rad": -knee, "knee_binding": knee_binding,
        "knee_rod_min_angle_rad": knee_rod_min_angle, "knee_rod_min_angle_deg": math.degrees(knee_rod_min_angle),
        "knee_interior_deg": 90.0 - math.degrees(knee),
        "hip_rod_m": hip_rod, "knee_rod_m": knee_rod,
        "hip_rod_norm": float(lk.hip_rod_normalized(_t(hip_rod))),
        "knee_rod_norm": float(lk.knee_rod_normalized(_t(knee_rod))),
        "rod_window_m": [float(lk.HIP_LEN_MIN_M), float(lk.HIP_LEN_MAX_M)],
        "limit_margin_deg": LIMIT_MARGIN_DEG, "joint_pos": joints,
        "hip_hold_action": (hip - dflt["hip"]) / ROD_SCALE,
        "knee_hold_action_left": (knee_cmd_equiv - dflt["knee_left"]) / ROD_SCALE,
        "defaults": dflt, "fold_search": fold, "link_clearances": link_clearances(hip, knee),
        "mirroring": "hip: same sign on every leg; knee: right legs (FR/MR/RR) carry the 180-deg Z frame flip -> "
                     "knee_right = -knee_left (crab_hex_scene_cfg, crab_hex_dimensions.KNEE_SIM_LIMITS_RIGHT_DEG)",
    }


# ===================================================================================== feet / table geometry
def foot_points_body(hip: float, knee_left: float) -> dict:
    """Body-frame pivot / knee / toe per leg (``crab_hex_foot_fk.leg_points``, plant A15+B, Body_Hip 0)."""
    fk = foot_fk()
    out = {}
    for leg in LEGS:
        p = fk.leg_points(leg, yaw=0.0, hip=hip, knee=knee_left)
        out[leg] = {k: [float(x) for x in v] for k, v in p.items()}
    return out


def table_corners(dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0, *, size=(TABLE_X_M, TABLE_Y_M)):
    """The table top's four corners in the robot's spawn frame for a table offset (dx, dy) and yaw dyaw."""
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    c, s = math.cos(dyaw), math.sin(dyaw)
    return [(dx + c * x - s * y, dy + s * x + c * y) for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def in_table(x: float, y: float, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0, *, margin: float = 0.0) -> bool:
    """(x, y) (spawn frame) inside the table footprint grown by ``margin``."""
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    lx, ly = c * (x - dx) - s * (y - dy), s * (x - dx) + c * (y - dy)
    return abs(lx) <= 0.5 * TABLE_X_M + margin and abs(ly) <= 0.5 * TABLE_Y_M + margin


def table_max_abs_y(jitter_xy: float = JITTER_XY_M, jitter_yaw_deg: float = JITTER_YAW_DEG) -> float:
    """The largest |y| any table-top point reaches over the jitter box (worst corner, worst offset)."""
    best = 0.0
    for yaw in (-math.radians(jitter_yaw_deg), 0.0, math.radians(jitter_yaw_deg)):
        for dy in (-jitter_xy, jitter_xy):
            best = max(best, max(abs(y) for _, y in table_corners(0.0, dy, yaw)))
    return best


def table_leg_clearance(pose: dict, root_z: float) -> dict:
    """Leg parts vs the table. The yaw axes sit on the body side faces (|y| = 0.6096, rotated hip plates keep their
    inner edge there; 1-in plate thickness at the 15-deg splay moves it in by 0.0127 * sin 15 deg), so the lateral gap
    to the worst jittered table corner bounds every leg part; plus each toe's position against the table top and
    footprint at the spawn."""
    d = dims()
    inner = float(d.LEG_MOUNT_Y_M) - 0.5 * float(d.PLY_THICKNESS_IN) * IN_TO_M * math.sin(math.radians(
        float(d.OUTER_ROW_SPLAY_DEG)))
    tmax = table_max_abs_y()
    feet = foot_points_body(pose["hip_rad"], pose["knee_left_rad"])
    toes = {}
    for leg, p in feet.items():
        tx, ty, tz = p["toe"]
        toes[leg] = {"toe_body": [tx, ty, tz], "toe_world_z": root_z + tz,
                     "pad_bottom_world_z": root_z + tz - FOOTPAD_HALF_M[2],
                     "above_table_top": bool(root_z + tz - FOOTPAD_HALF_M[2] > TABLE_TOP_M),
                     "inside_table_footprint_nominal": in_table(tx, ty),
                     "lateral_clear_of_worst_table_m": abs(ty) - tmax}
    lowest_leg_z = min(t["pad_bottom_world_z"] for t in toes.values())
    plate_bottom_world = root_z + (-PLATE_CORE_HALF_Z_M - PLATE_RADIUS_M)
    return {"leg_inner_abs_y_m": inner, "table_max_abs_y_m": tmax, "lateral_clearance_m": inner - tmax,
            "toes": toes, "lowest_toe_pad_world_z_m": lowest_leg_z,
            "hip_plate_bottom_world_z_m": plate_bottom_world,
            "any_toe_inside_table_footprint": any(t["inside_table_footprint_nominal"] for t in toes.values()),
            "toes_touch_floor_at_spawn": bool(lowest_leg_z <= 0.0)}


def jitter_draws(n: int, seed: int, *, xy: float = JITTER_XY_M, yaw_deg: float = JITTER_YAW_DEG,
                 enabled: bool = True) -> list:
    """Per-env TABLE offsets ``[dx, dy, dyaw]`` (m, m, rad): uniform in +-xy / +-yaw from ``numpy.random.default_rng(
    seed)`` (one (n, 3) draw, row-major, so env i's row does not depend on n). Env 0 is ALWAYS the nominal start
    (zeros); ``enabled=False`` -> all zeros. The robot stays at the env origin at yaw 0 (the eval's pinned spawn), so
    every env's stand command is the same face-0 row; the table moves instead."""
    import numpy as np  # noqa: PLC0415

    n = int(n)
    if n < 1:
        raise ValueError("jitter_draws: n >= 1")
    out = np.zeros((n, 3), dtype=np.float64)
    if enabled and n > 1:
        rng = np.random.default_rng(int(seed))
        u = rng.uniform(-1.0, 1.0, size=(n, 3))
        out[:, 0:2] = u[:, 0:2] * float(xy)
        out[:, 2] = u[:, 2] * math.radians(float(yaw_deg))
        out[0] = 0.0
    return out.tolist()


# ===================================================================================== kinematic slew (zero action)
def slew_timeline(pose: dict, *, root_z: float | None = None, physics_dt: float = 0.005, horizon_s: float = 10.0,
                  hip_cmd: float | None = None, knee_cmd_left: float | None = None) -> dict:
    """The action layer's rod arithmetic (``parkour_actions._apply_pitch_linkage``) run KINEMATICALLY from the start
    pose toward a constant command (default: the zero action = the default pose): hip rod -> its command at the loaded
    rod speed, knee rod -> the knee command interpreted at the DEFAULT hip, the knee angle re-solved at the hip's
    TARGET. Ignores dynamics and loads (the body is assumed to stay on the table until the toes reach the floor).
    Returns the times the rods arrive, when the mid toe first reaches the floor, and the window-entry bound for ANY
    controller (hip rod inside the +-0.25 rad window)."""
    import torch  # noqa: PLC0415

    lk = linkage()
    d = defaults_rad()
    rz = spawn_root_z() if root_z is None else float(root_z)
    h_cmd = d["hip"] if hip_cmd is None else float(hip_cmd)
    k_cmd = d["knee_left"] if knee_cmd_left is None else float(knee_cmd_left)
    lam_h = _t(pose["hip_rod_m"])
    lam_k = _t(pose["knee_rod_m"])
    lam_h_cmd = lk.hip_actuator_length(_t(h_cmd)).clamp(lk.HIP_LEN_MIN_M, lk.HIP_LEN_MAX_M)
    lam_k_cmd = lk.knee_actuator_length(_t(d["hip"]), _t(k_cmd)).clamp(lk.KNEE_LEN_MIN_M, lk.KNEE_LEN_MAX_M)
    step_h, step_k = lk.HIP_SPEED_M_S * physics_dt, lk.KNEE_SPEED_M_S * physics_dt
    win_lo = lk.hip_actuator_length(_t(d["hip"] - ROD_SCALE))
    win_hi = lk.hip_actuator_length(_t(d["hip"] + ROD_SCALE))
    lo, hi = (win_lo, win_hi) if float(win_lo) <= float(win_hi) else (win_hi, win_lo)
    floor_toe_z = -(rz - FOOTPAD_HALF_M[2])          # toe centre z (body) at which the pad bottom reaches z = 0
    t_hip = t_knee = t_floor = t_window = None
    trace = []
    n = int(round(horizon_s / physics_dt))
    for i in range(n + 1):
        th = lk.hip_angle_from_length(lam_h)
        tk = lk.knee_angle_from_length(lam_k, th)
        t = i * physics_dt
        toe_z = leg_plane_points(float(th), float(tk))["toe"][1]
        if i % 20 == 0:
            trace.append([round(t, 3), float(th), float(tk), toe_z])
        if t_window is None and float(lo) <= float(lam_h) <= float(hi):
            t_window = t
        if t_floor is None and toe_z <= floor_toe_z:
            t_floor = t
        if t_hip is None and abs(float(lam_h - lam_h_cmd)) < 1e-9:
            t_hip = t
        if t_knee is None and abs(float(lam_k - lam_k_cmd)) < 1e-9:
            t_knee = t
        if t_hip is not None and t_knee is not None and t_floor is not None and t_window is not None:
            break
        lam_h = lam_h + torch.clamp(lam_h_cmd - lam_h, -step_h, step_h)
        lam_k = lam_k + torch.clamp(lam_k_cmd - lam_k, -step_k, step_k)
    th_end = float(lk.hip_angle_from_length(lam_h_cmd))
    tk_end = float(lk.knee_angle_from_length(lam_k_cmd, lk.hip_angle_from_length(lam_h_cmd)))
    toe_end = leg_plane_points(th_end, tk_end)["toe"][1]
    rnd = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
    return {"hip_rod_arrives_s": rnd(t_hip), "knee_rod_arrives_s": rnd(t_knee), "toe_reaches_floor_s": rnd(t_floor),
            "hip_enters_action_window_s": rnd(t_window), "end_hip_rad": th_end, "end_knee_left_rad": tk_end,
            "end_toe_z_body_m": toe_end, "lift_if_body_follows_m": -(toe_end - floor_toe_z),
            "belly_gap_at_end_if_lifted_m": (-(toe_end - floor_toe_z)) + SPAWN_CLEARANCE_M,
            "floor_toe_z_body_m": floor_toe_z, "physics_dt": physics_dt,
            "trace_every_0p1s": trace,
            "note": "kinematic only: rod slew at the loaded speed, PD tracking lag, loads and the body's rise ignored"}


# ===================================================================================== R from this start
def r_start_analysis(pose: dict, SN, *, width: int = 1149) -> dict:
    """R (``SN.r_label``, the D3 constants) on a synthetic observation of the start: hip-pitch columns 29..34 = the
    pose's hip minus the default, cams 0, roll / pitch 0 -- plus the hip value at which each leg turns DOWN."""
    import torch  # noqa: PLC0415

    rst = SN.load_rst()
    p = SN.r_params()
    obs = torch.zeros((1, int(width)), dtype=torch.float32)
    dh = pose["hip_rad"] - pose["defaults"]["hip"]
    obs[0, list(rst.HIP_POS_OBS_COLS)] = float(dh)
    tgt, br = SN.r_label(obs)
    names = [rst.BRANCH_NAMES[int(b)] for b in br[0].tolist()]
    return {"hip_obs_rad": dh, "p_down_rad": p.p_down, "p_up_rad": p.p_up, "eps_air": p.eps_air,
            "eps_step": p.eps_step, "tilt_max": p.tilt_max, "branches": dict(zip(LEGS, names)),
            "target": [float(x) for x in tgt[0].tolist()], "target_is_zero": bool(torch.all(tgt == 0).item()),
            "becomes_down_when_hip_obs_ge_rad": p.p_down,
            "reading": "every leg UP and not DOWN, cam inside eps_air -> SEAT on all six: rods 0 (default targets), "
                       "cam 0 -> R's action is the zero action until a hip is within |p_down| of its default (then "
                       "HOLD: rods 0, cam trim -clip(k_h c, +-trim)); LIFT needs a DOWN leg with |cam| > eps_step"}


# ===================================================================================== the facts record
def build_facts(pose_name: str, *, SN=None) -> dict:
    """Everything the test rests on for one pose (``facts.json``, every run's summary)."""
    d = dims()
    pose = retracted_pose(pose_name)
    rz = spawn_root_z()
    out = {
        "pose": pose, "spawn_root_z_m": rz,
        "spawn_arithmetic": f"root z = table top {TABLE_TOP_M} + body half height {BODY_HALF_M[2]} + clearance "
                            f"{SPAWN_CLEARANCE_M} = {rz:.5f} m (root link chassis/body = the 0.7112 x 1.2192 x 0.3175 m "
                            "box, centred on the root frame)",
        "body_size_m": list(d.BODY_SIZE_M), "body_half_m": list(BODY_HALF_M),
        "standing_root_z_m": 1.064,
        "standing_belly_gap_m": 1.064 - BODY_HALF_M[2] - TABLE_TOP_M,
        "table": {"size_m": [TABLE_X_M, TABLE_Y_M, TABLE_HEIGHT_M], "top_z_m": TABLE_TOP_M,
                  "top_in": TABLE_TOP_IN, "base_z_m": TABLE_BASE_Z_M, "center_z_m": TABLE_CENTER_Z_M,
                  "kind": "kinematic RigidObject cuboid per env ({ENV_REGEX_NS}/SnTable), terrain material",
                  "jitter": {"xy_m": JITTER_XY_M, "yaw_deg": JITTER_YAW_DEG,
                             "rule": "per-env table offset; env 0 nominal; robot fixed at the env origin, yaw 0"}},
        "leg_mounts": {"yaw_axis_abs_y_m": float(d.LEG_MOUNT_Y_M),
                       "yaw_axis_x_m": [round(foot_fk().mount_x(leg), 4) for leg in LEGS],
                       "splay_deg": float(d.OUTER_ROW_SPLAY_DEG),
                       "hip_plate_u_m": [0.0, 2 * PLATE_CORE_U_M],
                       "hip_plate_z_body_m": [-(PLATE_CORE_HALF_Z_M + PLATE_RADIUS_M),
                                              PLATE_CORE_HALF_Z_M + PLATE_RADIUS_M]},
        "table_leg_clearance": table_leg_clearance(pose, rz),
        "slew_zero_action": slew_timeline(pose, root_z=rz),
        "height_scan": "the height scanner ray-casts /World/ground only (RayCasterCfg.mesh_prim_paths of the forward "
                       "teacher scene): the table is INVISIBLE to it; the policy's scan reads the floor ~0.80 m below "
                       "the root at the start (~1.06 m when standing), i.e. the body looks ~0.26 m low",
        "stood_definition": STOOD_DEFINITION,
    }
    if SN is not None:
        out["r_at_start"] = r_start_analysis(pose, SN)
    return out


# ===================================================================================== runtime geometry (torch or numpy)
def lowest_z_and_cos_tilt(root_z, qw, qx, qy, qz, half=BODY_HALF_M):
    """For an oriented body box (half extents ``half``) with root at height ``root_z`` and orientation (w, x, y, z):
    ``(lowest corner z, cos(tilt))`` -- the third row of the rotation matrix gives both (arithmetic only, so torch
    tensors and numpy arrays both work)."""
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    lowest = root_z - (abs(r20) * half[0] + abs(r21) * half[1] + abs(r22) * half[2])
    return lowest, r22


def quat_from_yaw(yaw: float) -> list:
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


# ===================================================================================== the reset event (Kit)
def sn_table_start(env, env_ids, joint_pos, root_z, table_center_z, table_offsets):
    """Reset event (appended LAST to the env's reset terms, so it runs after the ring spawn): the robot at its env
    origin (z 0, as ``reset_root_state_ring``) + ``root_z``, yaw 0, zero velocity, joints at ``joint_pos`` (24, the
    articulation order) with matching position targets and zero velocity targets; the table at the env origin + its
    ``table_offsets[env]`` (dx, dy, dyaw) and ``table_center_z``. The action term's own reset (which runs after the
    events) flags the rods for a re-sync from these joint angles on the first substep, so the rods start AT the
    retracted lengths and nothing snaps."""
    import torch  # noqa: PLC0415

    robot = env.scene["robot"]
    table = env.scene["sn_table"]
    dev = env.device
    ids = torch.arange(env.num_envs, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev).long()
    n = int(ids.numel())
    if n == 0:
        return
    origin = env.scene.env_origins[ids].clone()
    origin[:, 2] = 0.0
    pos = origin.clone()
    pos[:, 2] = float(root_z)
    quat = torch.zeros((n, 4), device=dev, dtype=pos.dtype)
    quat[:, 0] = 1.0
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev, dtype=pos.dtype), env_ids=ids)
    jp = torch.as_tensor(joint_pos, device=dev, dtype=pos.dtype).reshape(1, -1).repeat(n, 1)
    jv = torch.zeros_like(jp)
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    robot.set_joint_position_target(jp, env_ids=ids)
    robot.set_joint_velocity_target(jv, env_ids=ids)
    off = torch.as_tensor(table_offsets, device=dev, dtype=pos.dtype)[ids]
    tpos = origin.clone()
    tpos[:, 0] += off[:, 0]
    tpos[:, 1] += off[:, 1]
    tpos[:, 2] = float(table_center_z)
    tquat = torch.zeros((n, 4), device=dev, dtype=pos.dtype)
    tquat[:, 0] = torch.cos(0.5 * off[:, 2])
    tquat[:, 3] = torch.sin(0.5 * off[:, 2])
    # pose only: the table is KINEMATIC (PhysX refuses velocity writes on a kinematic actor; it never moves by itself)
    table.write_root_pose_to_sim(torch.cat([tpos, tquat], dim=-1), env_ids=ids)
    # ... and the same pose as its kinematic TARGET (PhysX moves a kinematic actor to its target on the next step; the
    # teleport above should already have done it -- the Kit script verifies the PhysX pose after the first step)
    try:
        full = table.data.root_link_pose_w.clone()                       # (count, 7) wxyz, just written above
        full_xyzw = torch.cat([full[:, :3], full[:, 4:7], full[:, 3:4]], dim=-1)
        table.root_physx_view.set_kinematic_targets(full_xyzw, ids)
    except Exception as exc:  # noqa: BLE001 -- a backend without kinematic targets: the teleport stands alone
        if not _WARNED.get("kin"):
            _WARNED["kin"] = True
            print(f"[SNT][WARN] table kinematic target not set ({type(exc).__name__}: {exc}); relying on the pose "
                  "write (checked after the first step)", flush=True)


_WARNED: dict = {}


# ===================================================================================== controller routing (Kit)
def controller_actions(controller: str, obs, stand, *, SN, s1, s1_est, p0: int, p1: int, num_prop: int,
                       st_pol=None, st_est=None, mask: int = 1):
    """One control step's actions for ``controller`` (``CONTROLLERS``) -> ``(actions, a_s1, extra)``.

    The harness's S1 path first, always: S1's own estimate written into the priv-explicit slice of a COPY ``h`` of
    the env's obs (the harness writes it in place on its own tensor; the env's ``obs`` is never written here),
    ``a_s1 = s1.act_inference(h, hist_encoding=True)``. Then:

    * ``s1``: ``a_s1`` on every row;
    * ``zero``: zeros;
    * ``stand_net``: ``x = SN.stand_net_input(h, st_est, p0, p1, num_prop, mask)`` (the stand checkpoint's own estimate
      on the UNMASKED prop -> a new tensor -> ``stand_mask_obs`` iff mask), ``a_st = st_pol.act_inference(x,
      hist_encoding=True)``, ``SN.route_actions(stand, a_st, a_s1)`` -- bitwise ``sn_dagger_core.launcher_actions``;
    * ``r``: ``SN.r_label(h)`` (R reads only cols 3-4 / 18-34, untouched by the priv write) routed on ``stand`` exactly
      as P2L's restance probe (R on stand rows, S1 elsewhere); ``extra`` carries R's target and branch.
    """
    import torch  # noqa: PLC0415

    if controller not in CONTROLLERS:
        raise ValueError(f"controller must be one of {CONTROLLERS}, got {controller!r}")
    h = obs.clone()
    h[:, p0:p1] = s1_est.inference(h[:, :num_prop])
    a_s1 = s1.act_inference(h, hist_encoding=True)
    extra: dict = {}
    if controller == "s1":
        return a_s1, a_s1, extra
    if controller == "zero":
        return torch.zeros_like(a_s1), a_s1, extra
    if controller == "stand_net":
        if st_pol is None or st_est is None:
            raise ValueError("controller stand_net needs st_pol and st_est")
        x = SN.stand_net_input(h, st_est, p0=p0, p1=p1, num_prop=num_prop, mask=mask)
        a_st = st_pol.act_inference(x, hist_encoding=True).to(a_s1.dtype)
        extra["a_stand"] = a_st
        return SN.route_actions(stand, a_st, a_s1), a_s1, extra
    tgt, branch = SN.r_label(h)
    extra["r_target"], extra["r_branch"] = tgt, branch
    return SN.route_actions(stand, tgt.to(a_s1.dtype), a_s1), a_s1, extra


# ===================================================================================== classification (numpy)
def first_run_start(cond, need: int):
    """The first index from which ``cond`` holds for ``need`` entries running, or None."""
    run = 0
    for t, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= need:
            return t - need + 1
    return None


def feet_on_floor(foot_force_n, foot_z_m):
    """(T, 6) arrays -> (T,) count of footpads loaded on the floor (force >= FOOT_LOADED_N, centre z <= FOOT_FLOOR_Z_M)."""
    import numpy as np  # noqa: PLC0415

    f = np.asarray(foot_force_n, dtype=np.float64)
    z = np.asarray(foot_z_m, dtype=np.float64)
    return ((f >= FOOT_LOADED_N) & (z <= FOOT_FLOOR_Z_M)).sum(axis=-1)


def stood_feet_steps_needed(window_steps: int) -> int:
    """Window steps that must carry >= STOOD_MIN_FEET loaded feet: ceil(STOOD_FEET_SHARE * w) (143 of 150)."""
    return int(math.ceil(STOOD_FEET_SHARE * int(window_steps) - 1e-9))


def classify(series: dict, dt: float) -> dict:
    """STOOD and the reported reads for ONE env. ``series`` (all length T, entry t = the state AFTER control step t,
    i.e. at time (t + 1) * dt): ``lowest_z`` (chassis lowest point, m), ``cos_tilt``, ``feet_on_floor`` (count),
    ``root_z``, ``cam_abs_max`` (rad), ``terminated`` (bool). See :data:`STOOD_DEFINITION`."""
    import numpy as np  # noqa: PLC0415

    low = np.asarray(series["lowest_z"], dtype=np.float64).reshape(-1)
    T = int(low.size)
    gap = low - TABLE_TOP_M
    tilt = np.arccos(np.clip(np.asarray(series["cos_tilt"], dtype=np.float64).reshape(-1), -1.0, 1.0))
    feet = np.asarray(series["feet_on_floor"]).reshape(-1)
    root_z = np.asarray(series["root_z"], dtype=np.float64).reshape(-1)
    cam = np.asarray(series["cam_abs_max"], dtype=np.float64).reshape(-1)
    term = np.asarray(series.get("terminated", np.zeros(T, dtype=bool))).reshape(-1).astype(bool)
    for k, v in (("cos_tilt", tilt), ("feet_on_floor", feet), ("root_z", root_z), ("cam_abs_max", cam),
                 ("terminated", term)):
        if v.size != T:
            raise ValueError(f"classify: {k} has {v.size} entries, lowest_z {T}")
    w = int(round(STOOD_WINDOW_S / dt))
    good = feet >= STOOD_MIN_FEET
    reasons = []
    feet_share = None
    if T < w or T == 0:
        reasons.append(f"episode {T} steps < the {w}-step STOOD window")
        gap_ok = feet_ok = feet_strict = False
    else:
        gap_ok = bool((gap[T - w:] >= STOOD_GAP_M - _EPS).all())
        n_good = int(good[T - w:].sum())
        feet_share = n_good / w
        feet_ok = n_good >= stood_feet_steps_needed(w)
        feet_strict = n_good == w
        if not gap_ok:
            reasons.append(f"belly gap < {STOOD_GAP_M} m in the last {STOOD_WINDOW_S} s (min "
                           f"{float(gap[T - w:].min()):.3f})")
        if not feet_ok:
            reasons.append(f"< {STOOD_MIN_FEET} feet on the floor on > {100 * (1 - STOOD_FEET_SHARE):.0f} % of the last "
                           f"{STOOD_WINDOW_S} s ({n_good}/{w} steps with >= {STOOD_MIN_FEET}; min "
                           f"{int(feet[T - w:].min())})")
    max_tilt = float(tilt.max()) if T else 0.0
    fell = max_tilt > FALL_TILT_RAD
    fell_off = bool(T and float(gap.min()) < -FELL_OFF_DROP_M - _EPS)
    terminated = bool(term.any())
    if fell:
        reasons.append(f"FALL: tilt {max_tilt:.3f} rad > {FALL_TILT_RAD}")
    if fell_off:
        reasons.append(f"FELL OFF THE TABLE: lowest point {float(gap.min()):+.3f} m vs the table top")
    if terminated:
        reasons.append("terminated before the end")
    clean = not fell and not fell_off and not terminated
    stood = gap_ok and feet_ok and clean
    stood_strict = gap_ok and feet_strict and clean
    i_lift = first_run_start(gap > LIFT_GAP_M, max(1, int(round(LIFT_HOLD_S / dt))))
    first_up = np.nonzero(gap > LIFT_GAP_M)[0]
    stood_from = None
    if stood:
        # the earliest step j from which clause (1) holds to the end: gap ok on every step of [j, T), a loaded step at
        # j, and >= STOOD_FEET_SHARE of [j, T) with >= 5 feet (exists: the first loaded step of the last window)
        gap_ok_t = gap >= STOOD_GAP_M - _EPS
        j_gap = T
        while j_gap > 0 and gap_ok_t[j_gap - 1]:
            j_gap -= 1
        idx = np.arange(T)
        suffix_good = np.cumsum(good[::-1].astype(np.int64))[::-1]
        cand = (idx >= j_gap) & good & (suffix_good >= STOOD_FEET_SHARE * (T - idx) - 1e-9)
        stood_from = (int(np.nonzero(cand)[0][0]) + 1) * dt
    return {
        "stood": bool(stood), "stood_strict": bool(stood_strict), "reasons": reasons, "fell": bool(fell),
        "fell_off_table": fell_off,
        "feet_share_last_window": feet_share,
        "terminated": terminated, "max_tilt_rad": max_tilt,
        "lift_off_s": None if i_lift is None else float((i_lift + 1) * dt),
        "first_gap_above_lift_s": None if first_up.size == 0 else float((int(first_up[0]) + 1) * dt),
        "standing_since_s": stood_from,
        "final_root_z_m": float(root_z[-1]) if T else None, "final_gap_m": float(gap[-1]) if T else None,
        "final_feet_on_floor": int(feet[-1]) if T else None,
        "min_feet_last_window": int(feet[max(0, T - w):].min()) if T else None,
        "min_gap_last_window_m": float(gap[max(0, T - w):].min()) if T else None,
        "max_gap_m": float(gap.max()) if T else None, "min_gap_m": float(gap.min()) if T else None,
        "on_table_end": bool(T and gap[-1] <= ON_TABLE_GAP_M),
        "cam_abs_max_end_rad": float(cam[-1]) if T else None,
        "cams_homed_end": bool(T and cam[-1] <= HOMED_RAD),
        "steps": T,
    }


def _stats(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None}
    s = sorted(v)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    return {"n": len(v), "median": med, "min": s[0], "max": s[-1], "mean": sum(v) / len(v)}


def aggregate(per_env: list) -> dict:
    """The per-controller JSON block over the envs' :func:`classify` results."""
    n = len(per_env)
    n_stood = sum(1 for r in per_env if r["stood"])
    feet_hist: dict = {}
    for r in per_env:
        k = str(r["final_feet_on_floor"])
        feet_hist[k] = feet_hist.get(k, 0) + 1
    n_strict = sum(1 for r in per_env if r.get("stood_strict"))
    return {
        "n_envs": n, "n_stood": n_stood, "stood_share": (n_stood / n) if n else None,
        "n_stood_strict": n_strict,
        "feet_share_last_window": _stats(r.get("feet_share_last_window") for r in per_env),
        "min_feet_last_window": _stats(r.get("min_feet_last_window") for r in per_env),
        "lift_off_s": _stats(r["lift_off_s"] for r in per_env),
        "n_lifted_off": sum(1 for r in per_env if r["lift_off_s"] is not None),
        "final_root_z_m": _stats(r["final_root_z_m"] for r in per_env),
        "final_gap_m": _stats(r["final_gap_m"] for r in per_env),
        "final_feet_on_floor": _stats(r["final_feet_on_floor"] for r in per_env),
        "final_feet_on_floor_hist": dict(sorted(feet_hist.items())),
        "max_tilt_rad": _stats(r["max_tilt_rad"] for r in per_env),
        "falls": sum(1 for r in per_env if r["fell"]),
        "fell_off_table": sum(1 for r in per_env if r["fell_off_table"]),
        "terminated": sum(1 for r in per_env if r["terminated"]),
        "on_table_end": sum(1 for r in per_env if r["on_table_end"]),
        "cams_homed_end": sum(1 for r in per_env if r["cams_homed_end"]),
        "fail_reasons": sorted({x.split(":")[0].split(" (")[0] for r in per_env for x in r["reasons"]}),
    }


def merge_summaries(paths) -> dict:
    """``{pose: {controller: aggregate + run}}`` from run summaries (``<prefix>_summary.json``); video runs (1 env)
    are listed under ``videos``."""
    table: dict = {}
    videos = []
    for p in sorted(Path(x) for x in paths):
        d = json.loads(Path(p).read_text())
        meta = d.get("meta", {})
        row = {"run": str(p), "num_envs": meta.get("num_envs"), "episode_s": meta.get("episode_s"),
               "smoke": meta.get("smoke"), **d.get("aggregate", {})}
        if meta.get("video"):
            videos.append({"pose": meta.get("pose"), "controller": meta.get("controller"), "mp4": meta.get("mp4"),
                           "env0": (d.get("per_env") or [{}])[0], "run": str(p)})
            continue
        table.setdefault(meta.get("pose"), {})[meta.get("controller")] = row
    return {"stats": table, "videos": videos, "stood_definition": STOOD_DEFINITION}


def _fmt(v, spec: str, width: int) -> str:
    return f"{'-':>{width}s}" if v is None else f"{v:{width}{spec}}"


def format_table(merged: dict) -> str:
    """The merge table. Beside STOOD: ``strict`` (>= 5 feet on EVERY window step), ``w5 share`` (the lowest env's
    window share of steps with >= 5 feet) and ``w min`` (the lowest feet count in any env's window), so a near-miss on
    the feet clause is visible."""
    lines = [f"{'pose':12s} {'controller':10s} {'envs':>4s} {'STOOD':>9s} {'strict':>6s} {'w5 share':>8s} {'w min':>5s} "
             f"{'lift-off s (med)':>16s} {'final root z':>12s} {'final gap':>9s} {'feet':>5s} {'max tilt':>8s} "
             f"{'falls':>5s} {'off':>4s} {'homed':>5s}"]
    for pose in sorted(merged.get("stats", {})):
        for ctrl in CONTROLLERS:
            r = merged["stats"][pose].get(ctrl)
            if r is None:
                continue
            lo = r["lift_off_s"]["median"]
            strict = r.get("n_stood_strict")
            w5 = (r.get("feet_share_last_window") or {}).get("min")
            wmin = (r.get("min_feet_last_window") or {}).get("min")
            lines.append(
                f"{pose:12s} {ctrl:10s} {r['n_envs']:4d} {r['n_stood']:3d}/{r['n_envs']:<3d}  "
                f"{_fmt(strict, 'd', 6)} {_fmt(w5, '.3f', 8)} {_fmt(None if wmin is None else int(wmin), 'd', 5)} "
                f"{'never' if lo is None else f'{lo:.2f}':>16s} {_fmt(r['final_root_z_m']['median'], '.3f', 12)} "
                f"{_fmt(r['final_gap_m']['median'], '.3f', 9)} {_fmt(r['final_feet_on_floor']['median'], '.1f', 5)} "
                f"{_fmt(r['max_tilt_rad']['max'], '.3f', 8)} {r['falls']:5d} {r['fell_off_table']:4d} "
                f"{r['cams_homed_end']:5d}")
    return "\n".join(lines)
