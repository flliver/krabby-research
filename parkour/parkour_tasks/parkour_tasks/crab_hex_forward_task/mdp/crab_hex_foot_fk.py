"""Pure-numpy leg forward kinematics for the crab hex (PLAN G leg-mount morphology, 2026-09-02).

Body-frame positions (z up, body centre at the origin, +x = travel direction) of one leg's
femur pivot, knee and toe from its joint angles, for any mount variant:

* ``splay_deg`` -- outward yaw of the front/rear mounts (row F toes toward -x, row R toes
  toward +x, symmetric about the transverse mid-plane; mid legs never splay);
* ``outer_axis_in`` -- outer yaw-axis distance from the body ends along the 28-in wall.

Defaults are the plant of record (A15+B); see ``DEFAULT_SPLAY_DEG`` / ``DEFAULT_OUTER_AXIS_IN``
and the ``LEGACY_*`` constants for records made on the 2026-08-20 golden geometry.

Frame chain is transcribed from ``assets/scripts/generate_crab.py`` (``Leg.__init__``)
and ``crab_hex_dimensions.py``; angle conventions from the dimensions module:

* yaw: the ``Body_Hip`` revolute (axis +Z, right-hand) rotates the WHOLE leg about the
  vertical yaw axis at ``(x_mount, sy * wall_y)``; the mount splay is a constant offset on
  the same axis, so ``total_yaw = mount_yaw + joint_yaw``;
* hip: 0 = femur horizontal (outboard), positive = femur down;
* knee: 0 = tibia perpendicular to the femur, positive folds the toe inboard.

Angles use the LEFT-leg sim convention. Right legs (FR/MR/RR) carry a 180-deg frame flip on
their pitch joints in the USD, which mirrors their knee limits; callers feeding raw right-leg
joint values must negate the knee angle (``KNEE_SIM_LIMITS_RIGHT = -reversed(LEFT)``). The
lateral direction is handled by ``sy`` (-1 for left legs, +1 for right).

No Isaac, no torch: stdlib + numpy, importable by file path like the sibling modules.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_DIMS_PATH = Path(__file__).with_name("crab_hex_dimensions.py")
_spec = importlib.util.spec_from_file_location("crab_hex_dimensions_fk", _DIMS_PATH)
dims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dims)

LEG_NAMES = ("FL", "FR", "ML", "MR", "RL", "RR")
LEFT = frozenset({"FL", "ML", "RL"})
ROW_SIGN = {"F": -1.0, "M": 0.0, "R": 1.0}
# Defaults = the plant of record (A15+B: splay 15 deg, outer axes 2.5 in). Records made before
# 2026-09-09 ("golden"/"base") are the LEGACY geometry: pass the LEGACY_* values explicitly.
DEFAULT_OUTER_AXIS_IN = dims.OUTER_LEG_AXIS_FROM_BODY_END_IN
DEFAULT_SPLAY_DEG = dims.OUTER_ROW_SPLAY_DEG
LEGACY_OUTER_AXIS_IN = dims.LEGACY_OUTER_LEG_AXIS_FROM_BODY_END_IN
LEGACY_SPLAY_DEG = dims.LEGACY_OUTER_ROW_SPLAY_DEG
SPLAY_CAP_DEG = 20.0

FEMUR_LEN_M = dims.FEMUR_HINGE_TO_HINGE_M          # 0.5842
TIBIA_LEN_M = dims.TIBIA_KNEE_TO_TOE_M             # 0.8255
PIVOT_OUTBOARD_M = dims.FEMUR_PIVOT_OUTBOARD_M     # 0.0635
PIVOT_Z_M = dims.FEMUR_PIVOT_Z_M                   # -0.2413
WALL_Y_M = dims.LEG_MOUNT_Y_M                      # 0.6096
YAW_THROW_RAD = math.radians(dims.YAW_THROW_DEG)   # +-25 deg cam throw


def side_sign(name: str) -> float:
    """-1 for left legs (extend toward -y), +1 for right legs."""
    return -1.0 if name in LEFT else 1.0


def mount_x(name: str, outer_axis_in: float = DEFAULT_OUTER_AXIS_IN) -> float:
    """Body-frame x of the leg's yaw axis (row F at -x, R at +x, M at 0)."""
    outer = (dims.BODY_LENGTH_X_IN / 2.0 - outer_axis_in) * dims.IN_TO_M
    return ROW_SIGN[name[0]] * outer


def mount_point(name: str, outer_axis_in: float = DEFAULT_OUTER_AXIS_IN) -> np.ndarray:
    """Body-frame yaw-axis point ``(x, sy*wall_y, 0)`` of one leg."""
    return np.array([mount_x(name, outer_axis_in), side_sign(name) * WALL_Y_M, 0.0])


def mount_yaw(name: str, splay_deg: float) -> float:
    """Mount yaw offset (rad) for an OUTWARD symmetric splay: ``-row_sign * sy * splay``.

    Row R (leading, +x) toes move toward +x on both sides; row F toes toward -x; mid 0.
    """
    if not 0.0 <= splay_deg <= SPLAY_CAP_DEG:
        raise ValueError(f"splay must be within [0, {SPLAY_CAP_DEG}] deg, got {splay_deg}")
    return -ROW_SIGN[name[0]] * side_sign(name) * math.radians(splay_deg)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leg_points(
    name: str,
    yaw: float = 0.0,
    hip: float = 0.0,
    knee: float = 0.0,
    *,
    splay_deg: float = DEFAULT_SPLAY_DEG,
    outer_axis_in: float = DEFAULT_OUTER_AXIS_IN,
) -> dict[str, np.ndarray]:
    """Body-frame ``pivot``, ``knee``, ``toe`` (each shape (3,)) for one leg pose.

    ``yaw`` is the Body_Hip joint angle (rad, +Z right-hand); ``hip``/``knee`` are the pitch
    angles in the left-leg sim convention (see module docstring).
    """
    sy = side_sign(name)
    p = mount_point(name, outer_axis_in)
    # unsplayed, unyawed leg: femur pivots horizontally outboard, tibia hangs down
    pivot0 = p + np.array([0.0, sy * PIVOT_OUTBOARD_M, PIVOT_Z_M])
    knee0 = pivot0 + FEMUR_LEN_M * np.array([0.0, sy * math.cos(hip), -math.sin(hip)])
    tib_dir = np.array([0.0, -sy * math.sin(hip + knee), -math.cos(hip + knee)])
    toe0 = knee0 + TIBIA_LEN_M * tib_dir
    rot = _rot_z(mount_yaw(name, splay_deg) + yaw)
    return {
        "mount": p,
        "pivot": p + rot @ (pivot0 - p),
        "knee": p + rot @ (knee0 - p),
        "toe": p + rot @ (toe0 - p),
    }


def transform_recorded_foot(
    name: str,
    foot_body_xyz: np.ndarray,
    *,
    splay_deg: float,
    outer_axis_in: float,
    base_outer_axis_in: float = DEFAULT_OUTER_AXIS_IN,
    base_splay_deg: float = DEFAULT_SPLAY_DEG,
) -> np.ndarray:
    """Where a recorded body-frame foot position (same joint angles) lands under a variant.

    Exact for a rigid mount change: rotate about the ORIGINAL yaw axis by the change in mount
    yaw (target minus base plant), then translate by the axis move along x. Works on
    ``(..., 3)`` arrays. The base defaults to the plant of record; records made on the legacy
    golden pass ``base_outer_axis_in=LEGACY_OUTER_AXIS_IN, base_splay_deg=LEGACY_SPLAY_DEG``.
    """
    p0 = mount_point(name, base_outer_axis_in)
    dx = mount_x(name, outer_axis_in) - mount_x(name, base_outer_axis_in)
    rot = _rot_z(mount_yaw(name, splay_deg) - mount_yaw(name, base_splay_deg))
    rel = np.asarray(foot_body_xyz, dtype=np.float64) - p0
    return (rel @ rot.T) + p0 + np.array([dx, 0.0, 0.0])


def toe_x_sweep_m(yaw_rad: float = YAW_THROW_RAD, hip: float = 0.0) -> float:
    """Fore-aft toe excursion for a yaw of ``yaw_rad`` at the given hip pitch (mid legs)."""
    radius = PIVOT_OUTBOARD_M + FEMUR_LEN_M * math.cos(hip)
    return radius * math.sin(yaw_rad)


def perpendicular_reachable(splay_deg: float, throw_deg: float = dims.YAW_THROW_DEG) -> bool:
    """Sideways-walking check: can the cam throw yaw the leg back to perpendicular?"""
    return splay_deg <= throw_deg
