"""Linear-actuator linkage kinematics for the crab hexapod's hip-pitch and knee joints.

The physical robot drives both pitch joints with linear actuators, not rotary motors:

- **Hip-pitch** (``*_Hip_Femur_RevoluteJoint``): a Sunline SLA08-24 hangs from the hole
  near the TOP of the vertical hip plate (2.08 in below the tip = ~20.3 in above the
  femur pivot, mid-width) and its rod pins to the femur ``a_f`` = 4.74 in from the pivot.
  This is exactly the near-vertical anchor the rod-length window (450-650 mm) + measured
  ROM had already forced before the plate geometry was corrected -- the assembly photos'
  vertical actuators are hanging from the plate top. Coordinates below remain flagged as
  estimates pending direct measurement; ``tests/unit/test_crab_hex_linkage.py`` pins the
  feasibility envelope (lengths inside the stroke window over the whole ROM, monotonicity,
  positive moment arm) so a wrong value fails loudly.

- **Knee** (``*_Femur_Tibia_RevoluteJoint``): a Yuhuang YH8-523D anchors on the hip plate
  near the femur pivot (~2.3 in outboard of it) and its rod pins to the tibia's
  above-knee lever, 3.0 in from the knee pivot. The actuator therefore spans hip -> tibia
  ACROSS the femur: knee angle is kinematically coupled to hip pitch. At constant rod
  length, pitching the femur changes the interior knee angle; the sim reproduces this in
  ``CrabHexDelayedJointPositionAction`` by treating the policy's knee command as a rod
  length and re-solving the knee target from the hip's COMMANDED trajectory each substep
  (target-side coupling -- live-state coupling pumped a rocking instability; see the
  NOTE(coupling-stability) in parkour_actions.py). A real consequence, encoded by the
  rod-length clamp: the joint-space corner "hip fully extended down + knee fully folded"
  is NOT reachable -- the box joint limits over-approximate the true reachable set.

Frames -- the leg's sagittal plane, side-agnostic (right legs' 180-deg Z frame flip is the
caller's concern, exactly as with the cam mapping):
  origin  = femur pivot;  +u = outboard along the hip beam;  +d = straight down.
  theta_h = sim hip angle: 0 = femur horizontal (outboard), positive = femur down.
  theta_k = sim knee angle: 0 = tibia perpendicular to femur, positive folds the toe
            inboard; interior femur-tibia angle = 90 deg - theta_k (left-leg convention,
            see crab_hex_dimensions).

Everything depends on the dimensions module only; regenerating hardware measurements
propagates here automatically.
"""

from __future__ import annotations

import math

import torch

try:  # package import (training) vs flat import (unit tests sys.path-insert this dir)
    from . import crab_hex_dimensions as dims
except ImportError:  # pragma: no cover
    import crab_hex_dimensions as dims

_IN = dims.IN_TO_M

# --------------------------------------------------------------------------------------
# Geometry parameters (meters, femur-pivot frame)
# --------------------------------------------------------------------------------------
FEMUR_ATTACH_R_M = dims.FEMUR_ACT_ATTACH_FROM_HIP_PIVOT_IN * _IN  # 0.1204
FEMUR_LEN_M = dims.FEMUR_HINGE_TO_HINGE_M  # 0.5842
KNEE_LEVER_M = dims.TIBIA_ACT_LEVER_IN * _IN  # 0.0762

# Hip actuator anchor (u, d): the clevis at the hip plate's TOP (vertical-plate geometry,
# corrected 2026-08-20 evening). The CAD anchor hole is 2.08 in below the top tip =
# 20.25 in above the femur pivot, at the plate's mid-width -- the same u as the pivot.
# d = -21.0 in (the clevis bracket sits just above the hole/tip) keeps the rod window
# [450, 650] mm feasible over the full ROM per the unit-test envelope checks. Still an
# estimate pending a direct hardware measurement.
HIP_ANCHOR_U_M = 0.0  # mid-width, directly above the femur pivot
HIP_ANCHOR_D_M = -21.0 * _IN

# Knee actuator anchor (u, d): near the femur pivot on the plate, ~2.3 in outboard of it
# (close to the plate's outboard edge), at pivot height +2.5 in up. Position pinned by
# rod-window feasibility (a high anchor cannot reach the tibia lever within 650 mm);
# estimate pending measurement.
KNEE_ANCHOR_U_M = 2.333 * _IN
KNEE_ANCHOR_D_M = -(dims.HIP_PLATE_WIDTH_IN / 2.0) * _IN  # -2.5 in

# Rod pin-to-pin length window (identical 200 mm stroke class on both joints).
HIP_LEN_MIN_M = dims.HIP_ACTUATOR["retracted_len_m"]
HIP_LEN_MAX_M = HIP_LEN_MIN_M + dims.HIP_ACTUATOR["stroke_m"]
KNEE_LEN_MIN_M = dims.KNEE_ACTUATOR["retracted_len_m"]
KNEE_LEN_MAX_M = KNEE_LEN_MIN_M + dims.KNEE_ACTUATOR["stroke_m"]

HIP_FORCE_N = dims.HIP_ACTUATOR["force_n"]
HIP_SPEED_M_S = dims.HIP_ACTUATOR["speed_loaded_m_s"]
KNEE_FORCE_N = dims.KNEE_ACTUATOR["force_n"]
KNEE_SPEED_M_S = dims.KNEE_ACTUATOR["speed_loaded_m_s"]


# --------------------------------------------------------------------------------------
# Hip-pitch linkage: rod from fixed anchor A to a point at radius a_f on the femur.
# --------------------------------------------------------------------------------------
def hip_actuator_length(theta_h: torch.Tensor) -> torch.Tensor:
    """Pin-to-pin rod length (m) as a function of the sim hip angle (rad)."""
    pu = FEMUR_ATTACH_R_M * torch.cos(theta_h) - HIP_ANCHOR_U_M
    pd = FEMUR_ATTACH_R_M * torch.sin(theta_h) - HIP_ANCHOR_D_M
    return torch.sqrt(pu * pu + pd * pd)


def hip_moment_arm(theta_h: torch.Tensor) -> torch.Tensor:
    """|d(length)/d(theta_h)| (m/rad): the joint-torque-per-rod-force lever, always > 0
    over the ROM (the mechanism has no dead center inside its travel)."""
    lam = hip_actuator_length(theta_h)
    pu = FEMUR_ATTACH_R_M * torch.cos(theta_h) - HIP_ANCHOR_U_M
    pd = FEMUR_ATTACH_R_M * torch.sin(theta_h) - HIP_ANCHOR_D_M
    dpu = -FEMUR_ATTACH_R_M * torch.sin(theta_h)
    dpd = FEMUR_ATTACH_R_M * torch.cos(theta_h)
    return torch.abs(pu * dpu + pd * dpd) / lam


def hip_angle_from_length(length: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`hip_actuator_length` on the operating branch.

    Law of cosines about the pivot: the anchor is at distance ``|A|`` in direction
    ``beta``; over the ROM, ``theta_h - beta`` stays inside (0, pi) where the length is
    monotonic, so the principal ``acos`` is the physical branch.
    """
    a = FEMUR_ATTACH_R_M
    mag_a = math.hypot(HIP_ANCHOR_U_M, HIP_ANCHOR_D_M)
    beta = math.atan2(HIP_ANCHOR_D_M, HIP_ANCHOR_U_M)
    cos_arg = (a * a + mag_a * mag_a - length * length) / (2.0 * a * mag_a)
    return beta + torch.acos(torch.clamp(cos_arg, -1.0, 1.0))


# --------------------------------------------------------------------------------------
# Knee linkage: rod from fixed anchor A_k to the tibia's above-knee lever; the knee point
# itself rides on the femur, which is what couples theta_k to theta_h.
# --------------------------------------------------------------------------------------
def _knee_geometry(theta_h: torch.Tensor, theta_k: torch.Tensor):
    """Rod vector (u, d) from the knee anchor to the tibia lever pin."""
    phi = theta_h + theta_k  # tibia absolute angle from vertical-down
    ku = FEMUR_LEN_M * torch.cos(theta_h)
    kd = FEMUR_LEN_M * torch.sin(theta_h)
    # Lever pin: opposite the toe, KNEE_LEVER_M above the knee along the tibia direction.
    lu = ku + KNEE_LEVER_M * torch.sin(phi)
    ld = kd - KNEE_LEVER_M * torch.cos(phi)
    return lu - KNEE_ANCHOR_U_M, ld - KNEE_ANCHOR_D_M


def knee_actuator_length(theta_h: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
    """Pin-to-pin rod length (m) as a function of BOTH pitch angles (rad)."""
    ru, rd = _knee_geometry(theta_h, theta_k)
    return torch.sqrt(ru * ru + rd * rd)


def knee_moment_arm(theta_h: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
    """|d(length)/d(theta_k)| at fixed hip (m/rad): the knee's torque lever."""
    phi = theta_h + theta_k
    ru, rd = _knee_geometry(theta_h, theta_k)
    lam = torch.sqrt(ru * ru + rd * rd)
    dlu = KNEE_LEVER_M * torch.cos(phi)
    dld = KNEE_LEVER_M * torch.sin(phi)
    return torch.abs(ru * dlu + rd * dld) / lam


def knee_hip_coupling(theta_h: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
    """d(length)/d(theta_h) at fixed knee (m/rad, SIGNED) -- the coupling term.

    At constant rod length: d(theta_k)/d(theta_h) = -coupling / d(length)/d(theta_k).
    """
    phi = theta_h + theta_k
    ru, rd = _knee_geometry(theta_h, theta_k)
    lam = torch.sqrt(ru * ru + rd * rd)
    dlu = -FEMUR_LEN_M * torch.sin(theta_h) + KNEE_LEVER_M * torch.cos(phi)
    dld = FEMUR_LEN_M * torch.cos(theta_h) + KNEE_LEVER_M * torch.sin(phi)
    return (ru * dlu + rd * dld) / lam


def knee_angle_from_length(length: torch.Tensor, theta_h: torch.Tensor) -> torch.Tensor:
    """Solve theta_k from the rod length at the given hip angle (closed form).

    With W = knee(theta_h) - anchor and r = KNEE_LEVER_M:
      length^2 = |W|^2 + r^2 + 2 r |W| sin(phi - psi),   psi = atan2(W_d, W_u)
    On the operating branch d(length)/d(theta_k) > 0, i.e. cos(phi - psi) > 0, which is
    the principal ``asin`` branch.
    """
    ku = FEMUR_LEN_M * torch.cos(theta_h) - KNEE_ANCHOR_U_M
    kd = FEMUR_LEN_M * torch.sin(theta_h) - KNEE_ANCHOR_D_M
    mag_w = torch.sqrt(ku * ku + kd * kd)
    psi = torch.atan2(kd, ku)
    r = KNEE_LEVER_M
    sin_arg = (length * length - mag_w * mag_w - r * r) / (2.0 * r * mag_w)
    phi = psi + torch.asin(torch.clamp(sin_arg, -1.0, 1.0))
    return phi - theta_h


# --------------------------------------------------------------------------------------
# Actuator capability reflected through the linkage
# --------------------------------------------------------------------------------------
def hip_torque_limit(theta_h: torch.Tensor) -> torch.Tensor:
    """Peak joint torque (N*m) available at this angle: F_max * moment_arm(theta)."""
    return HIP_FORCE_N * hip_moment_arm(theta_h)


def hip_joint_vel_limit(theta_h: torch.Tensor) -> torch.Tensor:
    """Peak joint speed (rad/s) at this angle: rod_speed / moment_arm(theta)."""
    return HIP_SPEED_M_S / hip_moment_arm(theta_h)


def knee_torque_limit(theta_h: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
    return KNEE_FORCE_N * knee_moment_arm(theta_h, theta_k)


def knee_joint_vel_limit(theta_h: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
    return KNEE_SPEED_M_S / knee_moment_arm(theta_h, theta_k)


# --------------------------------------------------------------------------------------
# Hardware-natural default pose: both actuators at mid-stroke.
# --------------------------------------------------------------------------------------
def hip_default_rad() -> float:
    """Sim hip angle with the hip actuator at mid-stroke."""
    mid = torch.tensor((HIP_LEN_MIN_M + HIP_LEN_MAX_M) / 2.0, dtype=torch.float64)
    return float(hip_angle_from_length(mid))


def knee_default_left_rad() -> float:
    """Left-leg sim knee angle with the knee actuator at mid-stroke (at the hip default)."""
    mid = torch.tensor((KNEE_LEN_MIN_M + KNEE_LEN_MAX_M) / 2.0, dtype=torch.float64)
    th = torch.tensor(hip_default_rad(), dtype=torch.float64)
    return float(knee_angle_from_length(mid, th))


def knee_default_right_rad() -> float:
    return -knee_default_left_rad()
