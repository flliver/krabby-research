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

- **Knee** (``*_Femur_Tibia_RevoluteJoint``): a Yuhuang YH8-523D whose rod pins to the
  tibia's above-knee lever, 3.0 in from the knee pivot. WHERE ITS REAR END IS ANCHORED is a
  switch (``knee_mount``, :data:`KNEE_MOUNTS`), because the two models differ sharply:

  * ``"femur"`` -- THE HARDWARE (the owner saw the knee actuator ride on the upper leg when
    the hip lifts, 2026-11-24). The anchor rides on the femur, so the rod length is a
    function of the knee angle ALONE: a held knee rod holds the knee angle while the hip
    pitches, and the folded tibia swings with the femur. This reproduces the robot's belly
    contact (knee rod at its curl end, uncurling the hip presses the folded tibia into the
    belly at 941-1003 hip hall counts, 176-188 mm at 5.33 counts/mm; the model touches at
    ~152-162 mm with the ROM table's 5 deg fold); the hip-plate model keeps the tibia ~0.1 m
    clear.
    The femur-frame anchor (:data:`KNEE_FEMUR_ANCHOR_S_M`, :data:`KNEE_FEMUR_ANCHOR_N_M`)
    is the hip-plate anchor estimate re-expressed in the femur frame at the hip default,
    so both mounts give the SAME rod length for every knee angle at the default pose (the
    policy's knee-command -> rod mapping and the mid-stroke defaults are unchanged);
    pending a direct measurement.
  * ``"hip_plate"`` -- the LEGACY model (2026-08-20) every recorded run trained on, and
    still the default so those runs replay bit-identically. The anchor sits on the hip
    plate near the femur pivot (~2.3 in outboard of it), so the actuator spans hip ->
    tibia ACROSS the femur and the knee angle is coupled to hip pitch: at constant rod
    length d(knee)/d(hip) ~ -0.9 at the default pose, i.e. the tibia keeps nearly its
    absolute orientation while the femur pitches. ``CrabHexDelayedJointPositionAction``
    re-solves the knee target from the hip's COMMANDED trajectory each substep (target-side
    coupling -- live-state coupling pumped a rocking instability; see the
    NOTE(coupling-stability) in parkour_actions.py), and its rod-length clamp makes the
    joint-space corner "hip fully extended down + knee fully folded" unreachable. Neither
    holds on the robot.

  Either way the policy's knee command is a rod-length command: the knee angle it names is
  converted to a rod length at the DEFAULT hip angle.

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

# Knee actuator anchor (u, d) of the LEGACY hip-plate mount: near the femur pivot on the
# plate, ~2.3 in outboard of it (close to the plate's outboard edge), at pivot height +2.5 in
# up. Position pinned by rod-window feasibility (a high anchor cannot reach the tibia lever
# within 650 mm); estimate pending measurement. The femur mount's anchor is derived from it
# (see "Femur-mounted knee" below).
KNEE_ANCHOR_U_M = 2.333 * _IN
KNEE_ANCHOR_D_M = -(dims.HIP_PLATE_WIDTH_IN / 2.0) * _IN  # -2.5 in

# Knee mounts (where the knee actuator's rear end is anchored; see the module docstring).
KNEE_MOUNT_HIP_PLATE = "hip_plate"  # legacy: anchored on the hip plate -> knee coupled to hip
KNEE_MOUNT_FEMUR = "femur"  # hardware: anchored on the femur -> knee independent of hip
KNEE_MOUNTS = (KNEE_MOUNT_HIP_PLATE, KNEE_MOUNT_FEMUR)
# The default keeps every recorded run bit-identical; flipping it is a plant change.
DEFAULT_KNEE_MOUNT = KNEE_MOUNT_HIP_PLATE

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
# Knee linkage: rod from the knee anchor to the tibia's above-knee lever. Every function
# takes ``mount`` (:data:`KNEE_MOUNTS`, default :data:`DEFAULT_KNEE_MOUNT`):
#   "hip_plate" -- the anchor is fixed on the hip plate while the knee point rides on the
#                  femur, which is what couples theta_k to theta_h (the legacy model; its
#                  code path below is unchanged, bit for bit);
#   "femur"     -- the anchor rides on the femur too: theta_h is ignored and the result
#                  is the femur-mounted function of theta_k (``knee_femur_*`` below),
#                  broadcast to theta_h's shape.
# --------------------------------------------------------------------------------------
def check_knee_mount(mount: str | None) -> str:
    """``mount`` validated; ``None`` = :data:`DEFAULT_KNEE_MOUNT`. Refuses anything else (a misspelled mount
    would otherwise train on the legacy plant without a word)."""
    if mount is None:
        return DEFAULT_KNEE_MOUNT
    if mount not in KNEE_MOUNTS:
        raise ValueError(f"unknown knee mount {mount!r}: known {KNEE_MOUNTS}")
    return mount


def _knee_on_femur(mount: str) -> bool:
    if mount == KNEE_MOUNT_FEMUR:
        return True
    if mount == KNEE_MOUNT_HIP_PLATE:
        return False
    raise ValueError(f"unknown knee mount {mount!r}: known {KNEE_MOUNTS}")


def _femur_theta_k(theta_h, theta_k):
    """``theta_k`` broadcast against ``theta_h``, so a femur-mount result has the hip-plate one's shape."""
    if isinstance(theta_h, torch.Tensor) and theta_h.shape != theta_k.shape:
        return torch.broadcast_tensors(theta_h, theta_k)[1]
    return theta_k


def _knee_geometry(theta_h: torch.Tensor, theta_k: torch.Tensor):
    """Rod vector (u, d) from the hip-plate knee anchor to the tibia lever pin."""
    phi = theta_h + theta_k  # tibia absolute angle from vertical-down
    ku = FEMUR_LEN_M * torch.cos(theta_h)
    kd = FEMUR_LEN_M * torch.sin(theta_h)
    # Lever pin: opposite the toe, KNEE_LEVER_M above the knee along the tibia direction.
    lu = ku + KNEE_LEVER_M * torch.sin(phi)
    ld = kd - KNEE_LEVER_M * torch.cos(phi)
    return lu - KNEE_ANCHOR_U_M, ld - KNEE_ANCHOR_D_M


def knee_actuator_length(
    theta_h: torch.Tensor, theta_k: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    """Pin-to-pin rod length (m): of BOTH pitch angles (rad) on the hip plate, of the knee alone on the femur."""
    if _knee_on_femur(mount):
        return knee_femur_actuator_length(_femur_theta_k(theta_h, theta_k))
    ru, rd = _knee_geometry(theta_h, theta_k)
    return torch.sqrt(ru * ru + rd * rd)


def knee_moment_arm(
    theta_h: torch.Tensor, theta_k: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    """|d(length)/d(theta_k)| at fixed hip (m/rad): the knee's torque lever."""
    if _knee_on_femur(mount):
        return knee_femur_moment_arm(_femur_theta_k(theta_h, theta_k))
    phi = theta_h + theta_k
    ru, rd = _knee_geometry(theta_h, theta_k)
    lam = torch.sqrt(ru * ru + rd * rd)
    dlu = KNEE_LEVER_M * torch.cos(phi)
    dld = KNEE_LEVER_M * torch.sin(phi)
    return torch.abs(ru * dlu + rd * dld) / lam


def knee_hip_coupling(
    theta_h: torch.Tensor, theta_k: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    """d(length)/d(theta_h) at fixed knee (m/rad, SIGNED) -- the coupling term.

    At constant rod length: d(theta_k)/d(theta_h) = -coupling / d(length)/d(theta_k).
    Identically zero on the femur mount (the anchor pitches with the femur).
    """
    if _knee_on_femur(mount):
        return torch.zeros_like(_femur_theta_k(theta_h, theta_k))
    phi = theta_h + theta_k
    ru, rd = _knee_geometry(theta_h, theta_k)
    lam = torch.sqrt(ru * ru + rd * rd)
    dlu = -FEMUR_LEN_M * torch.sin(theta_h) + KNEE_LEVER_M * torch.cos(phi)
    dld = FEMUR_LEN_M * torch.cos(theta_h) + KNEE_LEVER_M * torch.sin(phi)
    return (ru * dlu + rd * dld) / lam


def knee_angle_from_length(
    length: torch.Tensor, theta_h: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    """Solve theta_k from the rod length at the given hip angle (closed form).

    Hip plate -- with W = knee(theta_h) - anchor and r = KNEE_LEVER_M:
      length^2 = |W|^2 + r^2 + 2 r |W| sin(phi - psi),   psi = atan2(W_d, W_u)
    On the operating branch d(length)/d(theta_k) > 0, i.e. cos(phi - psi) > 0, which is
    the principal ``asin`` branch. Femur: :func:`knee_femur_angle_from_length` (theta_h
    ignored).
    """
    if _knee_on_femur(mount):
        if isinstance(theta_h, torch.Tensor) and theta_h.shape != length.shape:
            length = torch.broadcast_tensors(theta_h, length)[1]
        return knee_femur_angle_from_length(length)
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


def knee_torque_limit(
    theta_h: torch.Tensor, theta_k: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    return KNEE_FORCE_N * knee_moment_arm(theta_h, theta_k, mount)


def knee_joint_vel_limit(
    theta_h: torch.Tensor, theta_k: torch.Tensor, mount: str = DEFAULT_KNEE_MOUNT
) -> torch.Tensor:
    return KNEE_SPEED_M_S / knee_moment_arm(theta_h, theta_k, mount)


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


# --------------------------------------------------------------------------------------
# Femur-mounted knee (the hardware, 2026-11-24). Femur frame: origin = femur (hip) pivot,
# +s along the femur toward the knee, +n the femur normal on the toe side (= +d at
# theta_h = 0). The knee pivot is (FEMUR_LEN_M, 0) and the lever pin sits
# KNEE_LEVER_M * (sin theta_k, -cos theta_k) from it, whatever the hip angle.
# --------------------------------------------------------------------------------------
# Anchor ESTIMATE: the hip-plate anchor above, rotated into the femur frame at the hip
# default (about 2.26 in along the femur from the hip pivot and 2.56 in off its centre line
# on the lever side), so the two mounts give identical rod lengths at the default hip and
# the knee command -> rod mapping and the mid-stroke knee default do not depend on the
# mount. Replace with the measured bracket position when it is taken; the unit tests pin
# this derivation, so they will flag that the mounts then no longer agree at the default.
_KNEE_FEMUR_REF_HIP_RAD = hip_default_rad()
KNEE_FEMUR_ANCHOR_S_M = KNEE_ANCHOR_U_M * math.cos(_KNEE_FEMUR_REF_HIP_RAD) + KNEE_ANCHOR_D_M * math.sin(
    _KNEE_FEMUR_REF_HIP_RAD
)
KNEE_FEMUR_ANCHOR_N_M = -KNEE_ANCHOR_U_M * math.sin(_KNEE_FEMUR_REF_HIP_RAD) + KNEE_ANCHOR_D_M * math.cos(
    _KNEE_FEMUR_REF_HIP_RAD
)
# W = knee pivot - anchor (femur frame): the fixed side of the rod triangle.
_KNEE_FEMUR_W_S_M = FEMUR_LEN_M - KNEE_FEMUR_ANCHOR_S_M
_KNEE_FEMUR_W_N_M = -KNEE_FEMUR_ANCHOR_N_M
_KNEE_FEMUR_W_M = math.hypot(_KNEE_FEMUR_W_S_M, _KNEE_FEMUR_W_N_M)
_KNEE_FEMUR_PSI_RAD = math.atan2(_KNEE_FEMUR_W_N_M, _KNEE_FEMUR_W_S_M)


def _knee_femur_rod(theta_k: torch.Tensor):
    """Rod vector (s, n) from the femur-mounted anchor to the tibia lever pin."""
    rs = _KNEE_FEMUR_W_S_M + KNEE_LEVER_M * torch.sin(theta_k)
    rn = _KNEE_FEMUR_W_N_M - KNEE_LEVER_M * torch.cos(theta_k)
    return rs, rn


def knee_femur_actuator_length(theta_k: torch.Tensor) -> torch.Tensor:
    """Femur-mounted pin-to-pin rod length (m) as a function of the knee angle alone (rad)."""
    rs, rn = _knee_femur_rod(theta_k)
    return torch.sqrt(rs * rs + rn * rn)


def knee_femur_moment_arm(theta_k: torch.Tensor) -> torch.Tensor:
    """Femur-mounted |d(length)/d(theta_k)| (m/rad)."""
    rs, rn = _knee_femur_rod(theta_k)
    lam = torch.sqrt(rs * rs + rn * rn)
    return torch.abs(rs * KNEE_LEVER_M * torch.cos(theta_k) + rn * KNEE_LEVER_M * torch.sin(theta_k)) / lam


def knee_femur_angle_from_length(length: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`knee_femur_actuator_length` (closed form).

    length^2 = |W|^2 + r^2 + 2 r |W| sin(theta_k - psi),  psi = atan2(W_n, W_s); the knee ROM
    keeps theta_k - psi inside (-pi/2, pi/2), the principal ``asin`` branch.
    """
    r = KNEE_LEVER_M
    w = _KNEE_FEMUR_W_M
    sin_arg = (length * length - w * w - r * r) / (2.0 * r * w)
    return _KNEE_FEMUR_PSI_RAD + torch.asin(torch.clamp(sin_arg, -1.0, 1.0))


# --------------------------------------------------------------------------------------
# Optional first-order low-pass on the rod-length COMMAND (``KRABBY_ACTION_LPF_HZ``, omni
# motion-quality campaign 2026-09-22; DEFAULT OFF -- nothing calls these unless the action
# cfg carries ``lpf_hz``). Pure torch so the unit tests pin the exact arithmetic the action
# term runs.
# --------------------------------------------------------------------------------------
def lpf_alpha(cutoff_hz: float, dt: float) -> float:
    """Discrete first-order low-pass gain for a ``cutoff_hz`` corner sampled every ``dt`` s.

    ``y += alpha * (x - y)`` with ``alpha = 1 - exp(-2 pi f_c dt)`` -- the exact
    zero-order-hold discretisation of ``tau dy/dt = x - y`` with ``tau = 1 / (2 pi f_c)``, so
    the corner does not move with the physics step. ``cutoff_hz`` must be positive.
    """
    if not (cutoff_hz > 0.0 and dt > 0.0):
        raise ValueError(f"lpf_alpha needs cutoff_hz > 0 and dt > 0, got {cutoff_hz!r}, {dt!r}")
    return 1.0 - math.exp(-2.0 * math.pi * float(cutoff_hz) * float(dt))


def lpf_step(state: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """One low-pass step: ``state + alpha * (target - state)`` (returns a new tensor)."""
    return state + alpha * (target - state)


def hip_rod_normalized(length: torch.Tensor) -> torch.Tensor:
    """Hip rod length mapped to ``[-1, 1]`` over its stroke window (0 = mid-stroke)."""
    mid = (HIP_LEN_MIN_M + HIP_LEN_MAX_M) / 2.0
    half = (HIP_LEN_MAX_M - HIP_LEN_MIN_M) / 2.0
    return (length - mid) / half


def knee_rod_normalized(length: torch.Tensor) -> torch.Tensor:
    """Knee rod length mapped to ``[-1, 1]`` over its stroke window (0 = mid-stroke)."""
    mid = (KNEE_LEN_MIN_M + KNEE_LEN_MAX_M) / 2.0
    half = (KNEE_LEN_MAX_M - KNEE_LEN_MIN_M) / 2.0
    return (length - mid) / half
