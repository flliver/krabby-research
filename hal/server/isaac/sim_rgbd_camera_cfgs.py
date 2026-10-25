"""Simulated ZED (front) + MaixSense-style side RGB-D cameras on a robot link.

Used by ``hal.server.isaac.main`` to attach sensors to an env scene cfg **without** modifying
task packages under ``parkour/``. Imports ``quat_from_euler_xyz_tuple`` from
``parkour_tasks.default_cfg`` only (pose convention matches parkour).

Scene sensor names match Jetson catalog ids: ``side_right_*`` / ``side_left_*`` →
``side_right_rgbd`` / ``side_left_rgbd`` in ``IsaacSensorInterface``.

Depth uses Isaac Lab **pinhole** ``CameraCfg`` (Replicator / RTX), not ``RayCasterCameraCfg``.
Ray-cast depth only intersects static ``mesh_prim_paths`` (parkour uses ``/World/ground`` alone),
which does not match pinhole RGB scene content (robot, props, full terrain).
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
    ISAAC_SIDE_DEPTH_CLIP_M,
    ISAAC_SIDE_RGB_CLIP_M,
)
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
    """Same pinhole intrinsics/resolution as front RGB for side flank cameras."""

    return front_raycast_pattern_matching_rgb()

# Right flank (~ body −Y, ROS base_link: X forward, Y left). Left flank mirrors across Y at +0.08 m.
# Side optical axis matches the front camera tilt, then yaws ±90° about body +Z (right: +X → −Y;
# left: +X → +Y). Front Euler (180°, pitch ≈70°, ψ) is near gimbal lock, so ψ barely moves the
# optic axis; compose a fixed body yaw with the front rotation matrix instead of a second Euler triple.
_SIDE_RIGHT_POS = (0.0, -0.08, 0.12)
_SIDE_LEFT_POS = (0.0, 0.08, 0.12)


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


def _side_ros_offset_rot_from_front(*, yaw_deg: float) -> tuple[float, ...]:
    q_front = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)
    r_front = _rotmat_camera_from_quat_ros_wxyz(q_front)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_side = rz @ r_front
    return _quat_ros_wxyz_from_rotmat(r_side)


def _pinhole_spawn_cfg(*, clipping_range: tuple[float, float]) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        focal_length=_FRONT_RGB_FOCAL_LENGTH,
        focus_distance=400.0,
        horizontal_aperture=_FRONT_RGB_HORIZONTAL_APERTURE,
        clipping_range=clipping_range,
    )


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
) -> CameraCfg:
    """RTX depth at the same pose/intrinsics as the matching ``*_rgb`` pinhole."""

    return CameraCfg(
        prim_path=f"{base}/{prim_suffix}",
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
        data_types=["distance_to_camera"],
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


def sim_rgbd_camera_cfgs_for_robot_link(link_name: str) -> tuple[
    CameraCfg,
    CameraCfg,
    CameraCfg,
    CameraCfg,
    CameraCfg,
    CameraCfg,
]:
    """Return front + side-right + side-left RGB-D configs for HAL catalog wiring.

    Order: ``front_camera``, ``front_rgb``, ``side_right_camera``, ``side_right_rgb``,
    ``side_left_camera``, ``side_left_rgb``.

    ``*_camera`` sensors are pinhole depth; ``*_rgb`` are pinhole RGB. Pairs share pose and
    intrinsics so teleop depth matches the RGB render (same scene, same FOV).
    """

    base = f"{{ENV_REGEX_NS}}/Robot/{link_name}"

    front_rot = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)

    front_camera = _pinhole_depth_cfg(
        base=base,
        prim_suffix="front_camera",
        pos=_FRONT_POS,
        rot=front_rot,
        clipping_range=ISAAC_FRONT_DEPTH_CLIP_M,
    )

    front_rgb = _pinhole_rgb_cfg(
        base=base,
        prim_suffix="front_rgb",
        pos=_FRONT_POS,
        rot=front_rot,
        clipping_range=ISAAC_FRONT_RGB_CLIP_M,
    )

    side_right_rot = _side_ros_offset_rot_from_front(yaw_deg=-90.0)
    side_left_rot = _side_ros_offset_rot_from_front(yaw_deg=90.0)

    side_right_camera = _pinhole_depth_cfg(
        base=base,
        prim_suffix="side_right_camera",
        pos=_SIDE_RIGHT_POS,
        rot=side_right_rot,
        clipping_range=ISAAC_SIDE_DEPTH_CLIP_M,
    )
    side_right_rgb = _pinhole_rgb_cfg(
        base=base,
        prim_suffix="side_right_rgb",
        pos=_SIDE_RIGHT_POS,
        rot=side_right_rot,
        clipping_range=ISAAC_SIDE_RGB_CLIP_M,
    )
    side_left_camera = _pinhole_depth_cfg(
        base=base,
        prim_suffix="side_left_camera",
        pos=_SIDE_LEFT_POS,
        rot=side_left_rot,
        clipping_range=ISAAC_SIDE_DEPTH_CLIP_M,
    )
    side_left_rgb = _pinhole_rgb_cfg(
        base=base,
        prim_suffix="side_left_rgb",
        pos=_SIDE_LEFT_POS,
        rot=side_left_rot,
        clipping_range=ISAAC_SIDE_RGB_CLIP_M,
    )

    return (
        front_camera,
        front_rgb,
        side_right_camera,
        side_right_rgb,
        side_left_camera,
        side_left_rgb,
    )


def attach_sim_rgbd_sensors_to_scene_cfg(scene_cfg, robot_link: str) -> None:
    """Mutate ``scene_cfg`` with HAL-facing sensor names matching Jetson catalog ids."""

    (
        fc,
        fr,
        src,
        srr,
        slc,
        slr,
    ) = sim_rgbd_camera_cfgs_for_robot_link(robot_link)
    scene_cfg.front_camera = fc
    scene_cfg.front_rgb = fr
    scene_cfg.side_right_camera = src
    scene_cfg.side_right_rgb = srr
    scene_cfg.side_left_camera = slc
    scene_cfg.side_left_rgb = slr
