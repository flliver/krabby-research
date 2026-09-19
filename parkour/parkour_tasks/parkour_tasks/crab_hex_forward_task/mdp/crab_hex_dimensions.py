"""Measured physical-hardware dimensions for the Krabby hexapod (single source of truth).

Every number the robot model is generated from lives here, in the units it was measured
in (inches / pounds), with metric values derived. ``assets/scripts/generate_crab.py``
consumes this module to emit ``assets/crab.usda`` (the plant of record) and the variants under
``assets/variants/`` (incl. the legacy golden); sim-side config (defaults,
limits, actuator parameters) imports the derived constants so the USD and the configs can
never drift apart.

Provenance (2026-08-20 measurement session, user-confirmed against CAD):
- CAD cross-check parsed ``~/krabby/joint_specs/KrabV3-Legs.svg`` numerically (1 SVG unit
  = 1 mm). Femur hinge-to-hinge 23.0 in comes from CAD (both CAD versions agree; the
  stated 23.5 in was approximate). Tibia knee-to-toe 32.5 in and the 3.0 in knee lever are
  AS-BUILT values that supersede the SVG (31.3 in / 4.76 in hole).
- ROMs (yaw +-25 deg, hip 45-150 deg, knee 5-140 deg) supersede the 2026-08-13 set.
  Hip ROM is measured from vertical-UP: 45 deg = femur raised (retracted), 150 deg =
  femur extended down. Knee ROM is the interior femur-tibia angle.
- The hip is a VERTICAL plate door-hinged to the body side face (corrected 2026-08-20
  evening from the user's video review — the first model wrongly read it as a horizontal
  outboard beam). The yaw axis is the plate's inboard vertical edge, flush with the wall;
  the femur pivot is mid-width (2.5 in outboard), 22.334 in below the plate's top tip.
  The hip-pitch actuator hangs from the CAD small hole 2.08 in from the TOP tip (~20.3 in
  above the femur pivot); the knee actuator anchors near the pivot on the plate.
- Masses are all-inclusive: body 350 lb with everything in it, each leg 26.2 lb including
  both linear actuators. Leg parts are 1 in plywood; the per-link split below is estimated
  from part volumes + actuator bodies at their anchor positions, normalized to the
  measured leg total. Replace the estimates with real part weights if ever available.
- 2026-09 leg-mount change (splay 15 deg / outer axes 2.5 in, see the leg-mount layout block):
  chosen from the sim morphology campaigns (``.../experiments/2026-09-02_1446_leg_mount_morphology``,
  ``2026-09-04_1105_morph_x_exposure``, ``2026-09-06_2130_a15b_lineage``); it is a rigid mount
  transform, so every other measurement here is unchanged.

This module is stdlib-only (no torch, no Isaac) so the USD generator and unit tests can
import it directly (sys.path insertion of this directory, same pattern as
``test_crab_hex_cam_mapping.py``).
"""

import math

IN_TO_M = 0.0254
LB_TO_KG = 0.45359237

# --------------------------------------------------------------------------------------
# Body / chassis
# --------------------------------------------------------------------------------------
BODY_LENGTH_X_IN = 28.0  # travel-direction dimension; the leg rows run along this side
BODY_WIDTH_Y_IN = 48.0  # lateral dimension between the two leg-bearing side faces
BODY_HEIGHT_IN = 12.5
BODY_MASS_LB = 350.0  # all-inclusive (batteries, electronics, yaw gearmotors)

BODY_SIZE_M = (
    BODY_LENGTH_X_IN * IN_TO_M,  # 0.7112
    BODY_WIDTH_Y_IN * IN_TO_M,  # 1.2192
    BODY_HEIGHT_IN * IN_TO_M,  # 0.3175
)

# --------------------------------------------------------------------------------------
# Leg mount layout (3 legs per 28-inch side; yaw hinge flush with the body side face)
# --------------------------------------------------------------------------------------
# Plant of record since 2026-09-09 ("A15+B", leg-mount morphology campaigns 2026-09-02..09-07,
# user decision 2026-09-06): the outer yaw axes are re-hinged to 2.5 in from the body ends
# (axes at +-(14 - 2.5) = +-11.5 in from body centre along X; middle leg at 0) and the
# front/rear leg mounts are shimmed 15 deg outward (row F toes toward -x, row R toward +x;
# mid legs never splay). The 2026-08-20 build measured the axes at 5.5 in with no splay --
# kept below as the LEGACY values (``assets/variants/crab_simple__splay00_axis5p5in.usda``; every
# checkpoint before the a15b lineage was trained on that geometry).
OUTER_LEG_AXIS_FROM_BODY_END_IN = 2.5
OUTER_ROW_SPLAY_DEG = 15.0
LEGACY_OUTER_LEG_AXIS_FROM_BODY_END_IN = 5.5  # measured to the yaw axis (2026-08-20)
LEGACY_OUTER_ROW_SPLAY_DEG = 0.0
LEG_MOUNT_Y_M = BODY_WIDTH_Y_IN / 2.0 * IN_TO_M  # yaw axis sits on the side face plane

# --------------------------------------------------------------------------------------
# Hip plate (VERTICAL 1-in plywood plate, corrected 2026-08-20 evening from video review:
# door-hinged to the body side face along its inboard vertical edge — the yaw axis. It
# extends 5 in horizontally outboard and sticks ~6.6 in past the top and bottom of the
# 12.5 in body. The femur pivot pin is mid-width (2.5 in outboard of the wall), 3.25 in
# below the body bottom = 22.334 in below the plate's top tip. The earlier model read the
# 22.3 in dimension as a HORIZONTAL outboard beam — wrong by ~0.5 m of leg offset.)
# --------------------------------------------------------------------------------------
PLY_THICKNESS_IN = 1.0
HIP_PLATE_LENGTH_IN = 25.667  # CAD part length — runs VERTICALLY
HIP_PLATE_WIDTH_IN = 5.0  # horizontal outboard extent from the side face
FEMUR_PIVOT_FROM_PLATE_TOP_IN = 22.334  # top tip -> femur pivot, CAD ("tip to hinge")
FEMUR_PIVOT_OUTBOARD_IN = 2.5  # middle of the plate width, from the wall
# Femur pivot is 3.25 in below the body bottom face.
FEMUR_PIVOT_BELOW_BODY_IN = 3.25
# Linear-actuator rear anchors on the plate:
HIP_ACT_ANCHOR_FROM_TOP_TIP_IN = 2.08  # CAD small hole near the plate's TOP tip

FEMUR_PIVOT_OUTBOARD_M = FEMUR_PIVOT_OUTBOARD_IN * IN_TO_M  # 0.0635
# Femur pivot z relative to the body center (z-up, body center at 0):
FEMUR_PIVOT_Z_M = -(BODY_HEIGHT_IN / 2.0 + FEMUR_PIVOT_BELOW_BODY_IN) * IN_TO_M
# Derived invariant: the plate is centered on the body's mid-height EXACTLY (pivot
# 3.25 in below a 12.5 in body, 22.334 in below the top of a 25.667 in plate =>
# top tip = +half-plate-length). Assert so a future measurement change that breaks the
# symmetry is noticed rather than silently mis-centered.
HIP_PLATE_CENTER_Z_M = FEMUR_PIVOT_Z_M + (
    FEMUR_PIVOT_FROM_PLATE_TOP_IN - HIP_PLATE_LENGTH_IN / 2.0
) * IN_TO_M
assert abs(HIP_PLATE_CENTER_Z_M) < 1e-3, (
    f"hip plate no longer centered on the body ({HIP_PLATE_CENTER_Z_M:+.4f} m); "
    "update the generator's plate placement if this is a real measurement change"
)

# --------------------------------------------------------------------------------------
# Femur (1-in plywood plate, hangs from the hip beam's outboard pivot)
# --------------------------------------------------------------------------------------
FEMUR_PART_LENGTH_IN = 29.667  # CAD part length (hinges 3.33 in from each end)
FEMUR_WIDTH_IN = 5.0
FEMUR_HINGE_TO_HINGE_IN = 23.0  # CAD-confirmed (23.5 was approximate)
FEMUR_ACT_ATTACH_FROM_HIP_PIVOT_IN = 4.74  # inner hole, all legs (4.74/5.86 patterns)

FEMUR_HINGE_TO_HINGE_M = FEMUR_HINGE_TO_HINGE_IN * IN_TO_M  # 0.5842

# --------------------------------------------------------------------------------------
# Tibia (tapered 1-in plywood plank; bare plywood toe, no pad and no foot sensor)
# --------------------------------------------------------------------------------------
TIBIA_KNEE_TO_TOE_IN = 32.5  # as-built (supersedes the 31.3 in SVG value)
TIBIA_ABOVE_KNEE_IN = 6.83  # CAD: knee pivot to the top end (carries the actuator lever)
TIBIA_WIDTH_IN = 5.0  # at the knee; the plank tapers toward the toe
TIBIA_ACT_LEVER_IN = 3.0  # knee-actuator attachment above the knee pivot (as-built hole)
TIBIA_TAPER_VOLUME_FACTOR = 0.75  # tapered plank vs full rectangle, for mass estimate

TIBIA_KNEE_TO_TOE_M = TIBIA_KNEE_TO_TOE_IN * IN_TO_M  # 0.8255
TIBIA_PART_LENGTH_IN = TIBIA_ABOVE_KNEE_IN + TIBIA_KNEE_TO_TOE_IN  # 39.33

# --------------------------------------------------------------------------------------
# Joint ranges of motion (hardware conventions, 2026-08-20)
# --------------------------------------------------------------------------------------
YAW_THROW_DEG = 25.0  # +-25 deg, measured; supersedes the SVG-derived 28.54 deg
# Hip: measured from vertical-UP; 45 deg = femur raised/retracted, 150 deg = down/extended.
HIP_ROM_FROM_UP_DEG = (45.0, 150.0)
# Knee: interior femur-tibia angle; 140 deg = extended, 5 deg = retracted/folded.
KNEE_INTERIOR_ROM_DEG = (5.0, 140.0)

# Sim conventions (unchanged from the existing model):
#   hip:  0 = femur horizontal, positive = femur down  => sim = 90 - angle_from_down
#         angle_from_down = 180 - angle_from_up
#   knee: 0 = tibia perpendicular to femur (interior 90), positive folds the toe inboard
#         (left legs)                                   => sim = 90 - interior
#   Right legs (FR/MR/RR) carry a 180-deg Z frame flip (localRot0 = (0,0,1,0)), so their
#   knee limits mirror: R = (-L_upper, -L_lower).
_HIP_FROM_DOWN_DEG = (180.0 - HIP_ROM_FROM_UP_DEG[1], 180.0 - HIP_ROM_FROM_UP_DEG[0])
HIP_SIM_LIMITS_DEG = (90.0 - _HIP_FROM_DOWN_DEG[1], 90.0 - _HIP_FROM_DOWN_DEG[0])
KNEE_SIM_LIMITS_LEFT_DEG = (
    90.0 - KNEE_INTERIOR_ROM_DEG[1],
    90.0 - KNEE_INTERIOR_ROM_DEG[0],
)
KNEE_SIM_LIMITS_RIGHT_DEG = (-KNEE_SIM_LIMITS_LEFT_DEG[1], -KNEE_SIM_LIMITS_LEFT_DEG[0])
# Passive Body_Hip hard stop: wide enough that the soft limit (0.9x) clears the cam throw.
YAW_HARD_LIMIT_DEG = 28.0  # soft = 25.2 deg > 25.0 deg cam throw

# Joint defaults live in crab_hex_linkage.py (hip_default_rad / knee_default_left_rad):
# the hardware-natural neutral is both actuators at exact mid-stroke, which requires the
# linkage geometry, not just the ROM table.

# --------------------------------------------------------------------------------------
# Actuators (user-confirmed against hardware/Uno-v0.2/motor-sourcing-summary.md)
# --------------------------------------------------------------------------------------
# Hip-pitch: Sunline SLA08-24 (label: SLA08-24-100-200, 24 V, IP65, max 10 A).
HIP_ACTUATOR = {
    "force_n": 2000.0,
    "stroke_m": 0.200,
    "speed_loaded_m_s": 0.028,
    "retracted_len_m": 0.450,  # from the label's trailing -450; verify on hardware
    "mass_kg_est": 3.0,
}
# Knee: Yuhuang YH8-523D.
KNEE_ACTUATOR = {
    "force_n": 500.0,
    "stroke_m": 0.200,
    "speed_loaded_m_s": 0.033,
    "retracted_len_m": 0.450,  # assumed same class; verify on hardware
    "mass_kg_est": 1.5,
}
# Yaw: 24 V brushed gearmotor through the Whitworth quick-return crank.
YAW_MOTOR = {"speed_rpm": 30.0, "torque_nm": 20.0}
# Commanded-speed acceleration limit for the cam channels (gait-formation Phase 0):
# the real gearmotor cannot step 0 -> 30 RPM instantly; ramping full speed over ~0.5 s
# is a conservative hardware-faithful bound. Also removes the instant-full-throttle
# wheelie exploit structurally (both smoke tests) and serves gait smoothness.
CAM_ACCEL_LIMIT_RAD_S2 = math.pi / 0.5
# Whitworth ratio from the measured throw (overrides the SVG-derived K = 0.4778: the
# hardware sweep is the ground truth for the built crank/slot geometry).
YAW_K = math.sin(math.radians(YAW_THROW_DEG))  # 0.42262

# Gait clock (gait-formation-v2 Phase 1, 2026-08-22): cadence-to-command mapping for the
# clock-referenced contact-schedule reward. Endpoints anchored to hardware: the top of the
# formation command band maps to the cam motor's full speed (30 RPM = 0.5 rev/s — one gait
# cycle per cam revolution), and commands below the env's lin-vel clip are stops (clock
# frozen, all-stance schedule). Phase-0 scripted-gait probes measured ~0.07 m/s at
# 0.25 rev/s under the standard tripod phasing (setAB fixture), so the linear map is a
# specification the policy must beat with better phasing, not a kinematic identity.
CLOCK_F_MAX_REV_S = 0.5
CLOCK_V_MAX_M_S = 0.35
CLOCK_CMD_STOP_M_S = 0.2

# --------------------------------------------------------------------------------------
# Masses (per-link split estimated, normalized to the measured totals)
# --------------------------------------------------------------------------------------
LEG_MASS_LB = 26.2  # includes both linear actuators
CAMSHAFT_MASS_KG = 0.3  # virtual cam rotor (kept; carries the pinned diagonalInertia)
CAMSHAFT_DIAGONAL_INERTIA = 0.015  # placeholder pending hardware measurement (see scene cfg)
FOOTPAD_MASS_KG = 0.05  # toe-tip collision proxy (represents the bare plywood toe)

PLY_DENSITY_KG_M3 = 680.0  # birch plywood
_IN3_TO_M3 = IN_TO_M**3


def _ply_mass_kg(volume_in3: float) -> float:
    return volume_in3 * _IN3_TO_M3 * PLY_DENSITY_KG_M3


_HIP_PLY_KG = _ply_mass_kg(HIP_PLATE_LENGTH_IN * HIP_PLATE_WIDTH_IN * PLY_THICKNESS_IN)
_FEMUR_PLY_KG = _ply_mass_kg(FEMUR_PART_LENGTH_IN * FEMUR_WIDTH_IN * PLY_THICKNESS_IN)
_TIBIA_PLY_KG = _ply_mass_kg(
    TIBIA_PART_LENGTH_IN * TIBIA_WIDTH_IN * PLY_THICKNESS_IN * TIBIA_TAPER_VOLUME_FACTOR
)

# Both actuator bodies ride on the hip beam (anchored on its top edge).
_RAW_LINK_KG = {
    "hip": _HIP_PLY_KG + HIP_ACTUATOR["mass_kg_est"] + KNEE_ACTUATOR["mass_kg_est"],
    "femur": _FEMUR_PLY_KG,
    "tibia": _TIBIA_PLY_KG,
}
_LEG_BUDGET_KG = LEG_MASS_LB * LB_TO_KG - CAMSHAFT_MASS_KG - FOOTPAD_MASS_KG
_MASS_SCALE = _LEG_BUDGET_KG / sum(_RAW_LINK_KG.values())  # hardware/fasteners pro-rated

LEG_LINK_MASSES_KG = {name: kg * _MASS_SCALE for name, kg in _RAW_LINK_KG.items()}
LEG_MASS_KG = LEG_MASS_LB * LB_TO_KG
# The virtual cam rotor is budgeted INSIDE the leg's measured 26.2 lb (see _LEG_BUDGET_KG
# above), so each leg's prims -- links + footpad + camshaft -- sum to exactly the measured
# weight and the body keeps its full measured mass.
BODY_MASS_KG = BODY_MASS_LB * LB_TO_KG
TOTAL_MASS_KG = BODY_MASS_KG + 6.0 * LEG_MASS_KG  # 230.06

# Hip-link center of mass in the PLATE plane, as signed fractions of the plate's width
# (positive = outboard of the plate center) and length (positive = up from the plate
# center); used pre-scale in the USD's authored centerOfMass. Components:
# - plywood plate at its own center (mid-width, mid-length);
# - hip actuator hanging from the top-tip anchor hole, CG ~35% of the way down toward
#   the femur attachment (motor end up);
# - knee actuator near the femur pivot, CG at its rear anchor (~2.3 in outboard of the
#   pivot, i.e. near the plate's outboard edge, at pivot height).
_PLATE_HALF_LEN_IN = HIP_PLATE_LENGTH_IN / 2.0
_PIVOT_Z_FROM_CENTER_IN = _PLATE_HALF_LEN_IN - FEMUR_PIVOT_FROM_PLATE_TOP_IN  # -9.50
_HIP_ACT_ANCHOR_Z_IN = _PLATE_HALF_LEN_IN - HIP_ACT_ANCHOR_FROM_TOP_TIP_IN  # +10.75
_HIP_ACT_CG_Z_IN = _HIP_ACT_ANCHOR_Z_IN - 0.35 * (_HIP_ACT_ANCHOR_Z_IN - _PIVOT_Z_FROM_CENTER_IN)
_KNEE_ACT_CG_Y_FROM_WALL_IN = FEMUR_PIVOT_OUTBOARD_IN + 2.333  # near the outboard edge

_HIP_COM_Y_FROM_WALL_IN = (
    _HIP_PLY_KG * (HIP_PLATE_WIDTH_IN / 2.0)
    + HIP_ACTUATOR["mass_kg_est"] * FEMUR_PIVOT_OUTBOARD_IN
    + KNEE_ACTUATOR["mass_kg_est"] * _KNEE_ACT_CG_Y_FROM_WALL_IN
) / _RAW_LINK_KG["hip"]
_HIP_COM_Z_FROM_CENTER_IN = (
    _HIP_PLY_KG * 0.0
    + HIP_ACTUATOR["mass_kg_est"] * _HIP_ACT_CG_Z_IN
    + KNEE_ACTUATOR["mass_kg_est"] * _PIVOT_Z_FROM_CENTER_IN
) / _RAW_LINK_KG["hip"]

HIP_COM_WIDTH_FRACTION = (_HIP_COM_Y_FROM_WALL_IN - HIP_PLATE_WIDTH_IN / 2.0) / HIP_PLATE_WIDTH_IN
HIP_COM_LENGTH_FRACTION = _HIP_COM_Z_FROM_CENTER_IN / HIP_PLATE_LENGTH_IN

# Exports for the mesh-era generator (PLAN C): the hip link is a composite of the plywood
# plate mesh plus the two actuator point masses. Positions are in the extractor's plate
# frame: (b, z) with +b = the WALL/hinge side of the mid-width line, z up from the plate
# center. The hip actuator hangs mid-width; the knee actuator sits ~2.33 in OUTBOARD of
# mid-width (negative b) at pivot height.
HIP_ACT_POINT_MASS_KG = HIP_ACTUATOR["mass_kg_est"] * _MASS_SCALE
KNEE_ACT_POINT_MASS_KG = KNEE_ACTUATOR["mass_kg_est"] * _MASS_SCALE
HIP_ACT_CG_PLATE_IN = (0.0, _HIP_ACT_CG_Z_IN)
KNEE_ACT_CG_PLATE_IN = (-(_KNEE_ACT_CG_Y_FROM_WALL_IN - HIP_PLATE_WIDTH_IN / 2.0), _PIVOT_Z_FROM_CENTER_IN)

# --------------------------------------------------------------------------------------
# Friction (bare plywood everywhere, including the toe: no rubber feet on hardware)
# --------------------------------------------------------------------------------------
PLYWOOD_STATIC_FRICTION = 0.45
PLYWOOD_DYNAMIC_FRICTION = 0.35
PLYWOOD_RESTITUTION = 0.1
