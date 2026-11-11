"""Simulated front RGB-D + SEN0581-spec ToF depth cameras on a robot link.

Used by ``hal.server.isaac.main`` to attach sensors to an env scene cfg **without** modifying
task packages under ``parkour/``. Imports ``quat_from_euler_xyz_tuple`` from
``parkour_tasks.default_cfg`` only (pose convention matches parkour).

Scene sensor names match catalog ids: ``front_rgb`` / ``front_camera`` → ``front_rgbd``;
``belly_left_tof`` / ``belly_right_tof`` / ``rear_tof`` → depth-only rows in
``IsaacSensorInterface``.

Depth uses Isaac Lab **pinhole** ``CameraCfg`` (Replicator / RTX), not ``RayCasterCameraCfg``.
Ray-cast depth only intersects static ``mesh_prim_paths`` (parkour uses ``/World/ground`` alone),
which does not show the robot body; ToF cameras exist to watch the legs, so they render.
"""

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from hal.server.isaac.sim_depth_ranges import (
    ISAAC_FRONT_DEPTH_CLIP_M,
    ISAAC_FRONT_RGB_CLIP_M,
    ISAAC_TOF_DEPTH_CLIP_M,
)
from hal.server.isaac.sim_hal_camera_entries import SIM_HAL_CAMERA_ENTRIES
from parkour_tasks.default_cfg import quat_from_euler_xyz_tuple

_FRONT_POS = (0.33, 0.0, 0.08)
# Must match ``parkour_tasks.default_cfg.CAMERA_CFG``: same roll/pitch/yaw and quaternion helper
# (IsaacLab ``quat_from_euler_xyz`` omits parkour's sign convention on the quaternion).
_FRONT_EULER_DEG = (180.0, 70.0, -90.0)

# Same pinhole optics for front RGB and depth (spawn + clipping differ per stream).
_FRONT_RGB_HEIGHT = 480
_FRONT_RGB_WIDTH = 640
_FRONT_RGB_FOCAL_LENGTH = 24.0
_FRONT_RGB_HORIZONTAL_APERTURE = 20.955

# SEN0581: 100x100, 20 fps, 70° H x 60° V, depth 0.15–1.5 m.
_TOF_HEIGHT = 100
_TOF_WIDTH = 100
_TOF_FOCAL_LENGTH = 24.0
# aperture = 2 * f * tan(FOV/2); set H and V explicitly (pixel grid is square, FOV is not).
_TOF_HORIZONTAL_APERTURE = 2.0 * _TOF_FOCAL_LENGTH * math.tan(math.radians(70.0) / 2.0)
_TOF_VERTICAL_APERTURE = 2.0 * _TOF_FOCAL_LENGTH * math.tan(math.radians(60.0) / 2.0)

# --- Belly / rear ToF poses (body frame; ROS base_link: +X forward, +Y left, +Z up) ---
# Hex mounts (canonical / ``sim_hal_camera_entries``): wide belly plates from crab_hex_dimensions.
_HEX_BELLY_RIGHT_POS = (-0.14605, -0.5496, -0.19375)
_HEX_BELLY_LEFT_POS = (-0.14605, 0.5496, -0.19375)
_HEX_REAR_TOF_POS = (-0.3556, 0.0, -0.15875)
# Go2 smoke mounts: hex Y/Z hang outside a narrow Go2 and sit so close to the ground that
# left FOV often clips to 1.5 m (empty / under SEN0581 near range) on parkour terrain.
_GO2_BELLY_RIGHT_POS = (-0.05, -0.12, -0.05)
_GO2_BELLY_LEFT_POS = (-0.05, 0.12, -0.05)
_GO2_REAR_TOF_POS = (-0.22, 0.0, -0.05)
# Keep aliases for callers/tests that imported the hex names.
_BELLY_RIGHT_POS = _HEX_BELLY_RIGHT_POS
_BELLY_LEFT_POS = _HEX_BELLY_LEFT_POS
_REAR_TOF_POS = _HEX_REAR_TOF_POS
# Aim: 55° down, yaw 15° toward front (+X), facing outward (±Y).
_BELLY_PITCH_DOWN_DEG = 55.0
_BELLY_YAW_TOWARD_FRONT_DEG = 15.0
_REAR_PITCH_DOWN_DEG = 45.0


def _tof_mount_positions(link_name: str) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return ``(belly_right, belly_left, rear)`` body-frame mounts for ``link_name``."""

    if link_name in ("base", "trunk"):
        return _GO2_BELLY_RIGHT_POS, _GO2_BELLY_LEFT_POS, _GO2_REAR_TOF_POS
    return _HEX_BELLY_RIGHT_POS, _HEX_BELLY_LEFT_POS, _HEX_REAR_TOF_POS


def front_raycast_pattern_matching_rgb():
    """Legacy name: pinhole intrinsics tuple used by ``zed_like_scene_cfg`` imports.

    Ray-cast pattern cfg is no longer used for HAL teleop depth; kept so callers can share FOV constants.
    """

    from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

    ha = _FRONT_RGB_HORIZONTAL_APERTURE
    return PinholeCameraPatternCfg(
        focal_length=_FRONT_RGB_FOCAL_LENGTH,
        horizontal_aperture=ha,
        vertical_aperture=ha * (_FRONT_RGB_HEIGHT / float(_FRONT_RGB_WIDTH)),
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
    )


def side_raycast_pattern_matching_rgb():
    """Legacy alias: same pinhole intrinsics/resolution as front RGB."""

    return front_raycast_pattern_matching_rgb()


def _ros_offset_rot_from_euler_deg(euler_deg: tuple[float, float, float]) -> tuple[float, ...]:
    r, p, y_deg = euler_deg
    radians = torch.deg2rad(torch.tensor([float(r), float(p), float(y_deg)]))
    return quat_from_euler_xyz_tuple(*tuple(radians))


def _rotmat_camera_from_quat_ros_wxyz(q: tuple[float, float, float, float]) -> np.ndarray:
    """rotation ``R`` with ``v_body = R @ v_cam`` (ROS camera optical: ``+Z`` forward). Matches ``offset.rot``."""

    w, x, y, z = map(float, q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_ros_wxyz_from_rotmat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Inverse of ``_rotmat_camera_from_quat_ros_wxyz`` (Shepperd, ``w,x,y,z``)."""

    m = np.asarray(R, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _yaw_about_body_z(r_cam: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ r_cam


def _belly_tof_rot(*, outward_yaw_deg: float) -> tuple[float, ...]:
    """Belly ToF: face outward (±Y), pitch down, yaw slightly toward front (+X).

    ``outward_yaw_deg`` is body yaw of the optical axis from +X: −90° for right (−Y),
    +90° for left (+Y). Then apply pitch-down via the front-like Euler base and an
    additional yaw toward +X (front) of ``_BELLY_YAW_TOWARD_FRONT_DEG``.
    """
    # Base: look along +X with pitch down (reuse front optical tilt family), then yaw.
    # Front Euler (180, pitch, −90) maps roughly forward-down; compose body yaw for outward.
    pitch = 90.0 - _BELLY_PITCH_DOWN_DEG  # 55° down → pitch ≈ 35° in front-style Euler
    q_base = _ros_offset_rot_from_euler_deg((180.0, pitch, -90.0))
    r_base = _rotmat_camera_from_quat_ros_wxyz(q_base)
    # Yaw outward (±90), then 15° toward front (+X) on both sides:
    # right −90 → −75; left +90 → +75. (Old left path used +105 = rearward.)
    yaw = outward_yaw_deg - math.copysign(
        _BELLY_YAW_TOWARD_FRONT_DEG, outward_yaw_deg
    )
    return _quat_ros_wxyz_from_rotmat(_yaw_about_body_z(r_base, yaw))


def _rear_tof_rot() -> tuple[float, ...]:
    """Rear ToF: optical axis rearward (−X), pitched ``_REAR_PITCH_DOWN_DEG`` down."""
    pitch = 90.0 - _REAR_PITCH_DOWN_DEG
    q_base = _ros_offset_rot_from_euler_deg((180.0, pitch, -90.0))
    r_base = _rotmat_camera_from_quat_ros_wxyz(q_base)
    return _quat_ros_wxyz_from_rotmat(_yaw_about_body_z(r_base, 180.0))


def _pinhole_spawn_cfg(
    *,
    clipping_range: tuple[float, float],
    focal_length: float = _FRONT_RGB_FOCAL_LENGTH,
    horizontal_aperture: float = _FRONT_RGB_HORIZONTAL_APERTURE,
    vertical_aperture: float | None = None,
) -> sim_utils.PinholeCameraCfg:
    kwargs: dict = {
        "focal_length": focal_length,
        "focus_distance": 400.0,
        "horizontal_aperture": horizontal_aperture,
        "clipping_range": clipping_range,
    }
    if vertical_aperture is not None:
        kwargs["vertical_aperture"] = vertical_aperture
    return sim_utils.PinholeCameraCfg(**kwargs)


def _pinhole_rgb_cfg(
    *,
    base: str,
    prim_suffix: str,
    pos: tuple[float, float, float],
    rot: tuple[float, ...],
    clipping_range: tuple[float, float],
) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{base}/{prim_suffix}",
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
        data_types=["rgb"],
        spawn=_pinhole_spawn_cfg(clipping_range=clipping_range),
        offset=CameraCfg.OffsetCfg(
            pos=pos,
            rot=rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )


def _pinhole_depth_cfg(
    *,
    base: str,
    prim_suffix: str,
    pos: tuple[float, float, float],
    rot: tuple[float, ...],
    clipping_range: tuple[float, float],
    height: int = _FRONT_RGB_HEIGHT,
    width: int = _FRONT_RGB_WIDTH,
    focal_length: float = _FRONT_RGB_FOCAL_LENGTH,
    horizontal_aperture: float = _FRONT_RGB_HORIZONTAL_APERTURE,
    vertical_aperture: float | None = None,
) -> CameraCfg:
    """RTX depth pinhole (front uses RGB-matched intrinsics; ToF uses SEN0581 FOV)."""

    return CameraCfg(
        prim_path=f"{base}/{prim_suffix}",
        height=height,
        width=width,
        data_types=["distance_to_camera"],
        spawn=_pinhole_spawn_cfg(
            clipping_range=clipping_range,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=pos,
            rot=rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )


def _tof_depth_cfg(
    *,
    base: str,
    prim_suffix: str,
    pos: tuple[float, float, float],
    rot: tuple[float, ...],
) -> CameraCfg:
    return _pinhole_depth_cfg(
        base=base,
        prim_suffix=prim_suffix,
        pos=pos,
        rot=rot,
        clipping_range=ISAAC_TOF_DEPTH_CLIP_M,
        height=_TOF_HEIGHT,
        width=_TOF_WIDTH,
        focal_length=_TOF_FOCAL_LENGTH,
        horizontal_aperture=_TOF_HORIZONTAL_APERTURE,
        vertical_aperture=_TOF_VERTICAL_APERTURE,
    )


def sim_hal_camera_cfgs_for_robot_link(link_name: str) -> dict[str, CameraCfg]:
    """Return scene attribute name → ``CameraCfg`` for all ``SIM_HAL_CAMERA_ENTRIES``."""

    base = f"{{ENV_REGEX_NS}}/Robot/{link_name}"
    front_rot = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)
    belly_right_pos, belly_left_pos, rear_pos = _tof_mount_positions(link_name)
    cfgs: dict[str, CameraCfg] = {
        "front_camera": _pinhole_depth_cfg(
            base=base,
            prim_suffix="front_camera",
            pos=_FRONT_POS,
            rot=front_rot,
            clipping_range=ISAAC_FRONT_DEPTH_CLIP_M,
        ),
        "front_rgb": _pinhole_rgb_cfg(
            base=base,
            prim_suffix="front_rgb",
            pos=_FRONT_POS,
            rot=front_rot,
            clipping_range=ISAAC_FRONT_RGB_CLIP_M,
        ),
        "belly_right_tof": _tof_depth_cfg(
            base=base,
            prim_suffix="belly_right_tof",
            pos=belly_right_pos,
            rot=_belly_tof_rot(outward_yaw_deg=-90.0),
        ),
        "belly_left_tof": _tof_depth_cfg(
            base=base,
            prim_suffix="belly_left_tof",
            pos=belly_left_pos,
            rot=_belly_tof_rot(outward_yaw_deg=90.0),
        ),
        "rear_tof": _tof_depth_cfg(
            base=base,
            prim_suffix="rear_tof",
            pos=rear_pos,
            rot=_rear_tof_rot(),
        ),
    }
    return cfgs


def sim_rgbd_camera_cfgs_for_robot_link(link_name: str) -> dict[str, CameraCfg]:
    """Alias for ``sim_hal_camera_cfgs_for_robot_link`` (historical name)."""

    return sim_hal_camera_cfgs_for_robot_link(link_name)


def attach_sim_rgbd_sensors_to_scene_cfg(scene_cfg, robot_link: str) -> None:
    """Mutate ``scene_cfg`` with HAL-facing sensor names matching catalog ids."""

    cfgs = sim_hal_camera_cfgs_for_robot_link(robot_link)
    for attr, cfg in cfgs.items():
        setattr(scene_cfg, attr, cfg)


def apply_sim_hal_camera_update_periods(scene_cfg, *, control_dt: float) -> None:
    """Set each HAL camera's ``update_period`` from ``SIM_HAL_CAMERA_ENTRIES`` fps."""

    for entry in SIM_HAL_CAMERA_ENTRIES:
        period = 1.0 / float(entry.fps) if entry.fps > 0.0 else control_dt
        depth_sensor = getattr(scene_cfg, entry.depth_attr, None)
        if depth_sensor is not None:
            depth_sensor.update_period = period
        if entry.rgb_attr is not None:
            rgb_sensor = getattr(scene_cfg, entry.rgb_attr, None)
            if rgb_sensor is not None:
                rgb_sensor.update_period = period
