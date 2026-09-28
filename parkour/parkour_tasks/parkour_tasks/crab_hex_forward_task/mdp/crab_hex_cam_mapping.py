"""Cam-shaft -> leg-yaw kinematic mapping for the crab hexapod cam mechanism.

The real cam converts continuous motor rotation into the leg's back-and-forth yaw motion;
this module defines that relationship as a plain function of the shaft's simulated joint
state, applied each physics substep in ``CrabHexDelayedJointPositionAction.apply_actions``
to kinematically slave ``*_Body_Hip_RevoluteJoint`` to ``*_Body_CamShaft_RevoluteJoint``.

Phase E (current) -- derived from CAD, not assumed. ``~/krabby/joint_specs`` (CAD exports +
an assembly photo) identifies the real mechanism as a **slotted-lever (Whitworth quick-return)
linkage**, not a plain rotating cam against a follower:

- A rigid lever pivots on the fixed chassis yaw axis (matches ``*_Body_Hip_RevoluteJoint``'s
  own axis) and carries a straight slot passing through that same pivot point.
- The motor's continuously-rotating shaft drives a crank pin, offset by radius ``r`` from the
  motor's own fixed axis, which rides inside that slot.
- The lever's angle is therefore always exactly the angle, as seen from the pivot, to wherever
  the crank pin currently is -- see the derivation below.

Let ``L`` = distance from the leg's yaw pivot to the motor's rotation axis, ``r`` = crank
radius (motor axis to pin), ``theta_shaft`` = motor rotation angle measured from the pivot's
own direction toward the motor axis. The pin position relative to the pivot is
``(L + r*cos(theta_shaft), r*sin(theta_shaft))``, and since it must lie exactly on the slot's
line through the pivot, the lever's angle is just that vector's own angle:

    theta_hip = atan2(r*sin(theta_shaft), L + r*cos(theta_shaft))

This depends on ``r`` and ``L`` **only through their ratio** ``K = r/L`` (atan2 is invariant to
scaling both arguments together). ``K`` is measured directly from the CAD
(``~/krabby/joint_specs/KrabV3-Legs.svg``, group ``YawCover/Left/YawSlot1/Main``), not
calibrated from an assumed swing limit:

- The piece has **two** slots, both on the same leg-yaw part: ``MotorSlot`` (the crank pin's
  slot) and ``HipSlot`` (at the lever's other end). The crank arm is exactly half the
  MotorSlot's length -- confirmed against the CAD, it reaches both slot ends at the two
  extremes of rotation -- so ``r = MotorSlot_length / 2``.
- The pivot sits at the exact center of ``HipSlot``, and ``L`` is the distance between the two
  slots' centers (both measured directly from the SVG path/rect geometry).

This gives ``r = 3.6807 in``, ``L = 7.7036 in``, ``K = r/L = 0.4778`` -- and correspondingly
``theta_hip_max = asin(K) = 28.54 deg``, a *prediction* from the geometry, not an input. This
is substantially tighter than the ``+/-50 deg`` previously hardcoded as the hip joint's hard
limit, which the design docs themselves flag as never having been measured against the
physical robot ("somewhat close, but not perfect"). The two front/rear legs' documented
default splay (+/-0.6 rad ~= 34.4 deg) exceeds this corrected limit and has been rescaled
accordingly -- see ``crab_hex_scene_cfg.py``.

**What this does NOT resolve**: whether the real slot's line passes through the pivot with
*exactly* zero perpendicular offset (assumed here -- the standard/simplest configuration for
this mechanism family; a nonzero offset would need a slightly more general "offset
slider-crank" form). Revisit if/when direct hardware measurement becomes available.

One real behavioral consequence of the real mechanism vs. the earlier ``sin(theta_shaft)``
placeholder it replaces: this is a genuine **quick-return** curve, not a symmetric sinusoid --
see ``tests/unit/test_crab_hex_cam_mapping.py`` for the asymmetry check. A policy trained
against the old symmetric mapping (or the earlier 50-deg-calibrated version of this same
Whitworth model) has no reason to have learned the corrected timing/amplitude, so this change
invalidates the coordination existing checkpoints learned, the same way the original
cam-mechanism migration did.
"""

from __future__ import annotations

import math

import torch

# Measured directly from KrabV3-Legs.svg (group YawCover/Left/YawSlot1/Main, all six legs share
# identical slot geometry). See module docstring for how these were read off the CAD.
_MOTOR_SLOT_LENGTH_IN = 7.3613315
"""MotorSlot's long dimension (its SVG <rect> height in the local, inch-scaled group frame)."""

_MOTOR_SLOT_CENTER_IN = (15.7529737, 8.63409305)
_HIP_SLOT_CENTER_IN = (15.752862, 16.337659)
"""Bounding-box centers of MotorSlot (a plain rect) and HipSlot (a rounded-bottom notch path)."""

_R = _MOTOR_SLOT_LENGTH_IN / 2.0
"""Crank radius: the arm is half the MotorSlot's length, reaching both ends at the extremes."""

_L = math.hypot(
    _MOTOR_SLOT_CENTER_IN[0] - _HIP_SLOT_CENTER_IN[0],
    _MOTOR_SLOT_CENTER_IN[1] - _HIP_SLOT_CENTER_IN[1],
)
"""Pivot-to-motor-axis distance: the pivot sits at HipSlot's center, so this is just the
distance between the two slots' centers."""

_K_SVG = _R / _L
"""Crank-radius / pivot-separation ratio as predicted by the SVG (~0.4778 -> 28.54 deg).
Superseded (2026-08-20) by the hardware-measured throw below; kept for provenance."""

# NOTE(hardware-measurements, 2026-08-20): the physical robot's yaw throw measures +-25 deg,
# not the SVG-predicted 28.54 deg -- the built crank/slot geometry differs from the
# KrabV3-Legs.svg layout (as-built holes elsewhere on the leg differ from that SVG too, e.g.
# the 3-in knee lever). Since the mapping depends on the geometry only through K, and
# theta_hip_max = asin(K) exactly, the measured throw IS a direct measurement of K.
try:
    from .crab_hex_dimensions import YAW_K as _K  # package import (training)
except ImportError:  # flat import (unit tests sys.path-insert this directory)
    from crab_hex_dimensions import YAW_K as _K

THETA_HIP_MAX = math.asin(_K)
"""The mechanism's own natural swing limit: asin(K) = 25.0 deg exactly, because K is now
derived from the measured throw (crab_hex_dimensions.YAW_THROW_DEG)."""


def cam_shaft_to_hip(theta_shaft: torch.Tensor, omega_shaft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Whitworth slotted-lever mapping: ``theta_hip = atan2(K*sin(theta_shaft), 1 + K*cos(theta_shaft))``.

    Velocity via the closed-form derivative (verified against a numeric derivative in the unit
    tests to ~1e-9), not the chain rule through an intermediate theta_hip computation -- this
    avoids a second inverse-trig call and stays well-conditioned everywhere, including exactly
    at the swing extrema where d(theta_hip)/d(theta_shaft) passes through zero.

    Returns (theta_hip, omega_hip), matching the shapes of the inputs.
    """
    sin_s = torch.sin(theta_shaft)
    cos_s = torch.cos(theta_shaft)
    theta_hip = torch.atan2(_K * sin_s, 1.0 + _K * cos_s)
    dtheta_hip_dtheta_shaft = (_K * (_K + cos_s)) / (1.0 + 2.0 * _K * cos_s + _K * _K)
    omega_hip = dtheta_hip_dtheta_shaft * omega_shaft
    return theta_hip, omega_hip


def hip_to_cam_shaft_default(theta_hip_default: float) -> float:
    """Principal-value inverse, for computing a self-consistent CamShaft init pose from the
    leg's Body_Hip default -- see crab_hex_scene_cfg.py.

    From the mapping's defining constraint ``r*sin(theta_shaft - theta_hip) = L*sin(theta_hip)``:
    ``theta_shaft = theta_hip + asin(sin(theta_hip) / K)``. Only defined for
    ``|theta_hip| <= THETA_HIP_MAX`` (i.e. ``|sin(theta_hip)| <= K``) -- raises ValueError
    outside that range rather than silently returning NaN, since a default pose the mechanism
    cannot physically reach is a configuration bug, not a runtime edge case.
    """
    ratio = math.sin(theta_hip_default) / _K
    if abs(ratio) > 1.0:
        raise ValueError(
            f"theta_hip_default={theta_hip_default:.4f} rad ({math.degrees(theta_hip_default):.2f} deg) "
            f"exceeds the mechanism's own swing limit of +/-{math.degrees(THETA_HIP_MAX):.2f} deg "
            f"(K={_K:.4f}); this default pose is not physically reachable."
        )
    return theta_hip_default + math.asin(ratio)
