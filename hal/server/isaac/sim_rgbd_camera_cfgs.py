"""Simulated ZED (front) + MaixSense-style (right side) RGB-D cameras on a robot link.

Used by ``hal.server.isaac.main`` to attach sensors to an env scene cfg **without** modifying
task packages under ``parkour/``. Imports ``CAMERA_CFG`` from ``parkour_tasks.default_cfg`` only.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from parkour_tasks.default_cfg import CAMERA_CFG, quat_from_euler_xyz_tuple

_FRONT_POS = (0.33, 0.0, 0.08)
# Must match ``parkour_tasks.default_cfg.CAMERA_CFG``: same roll/pitch/yaw and quaternion helper
# (IsaacLab ``quat_from_euler_xyz`` omits parkour's sign convention on the quaternion).
_FRONT_EULER_DEG = (180.0, 70.0, -90.0)

# Same pinhole optics for front RGB USD camera and RayCaster depth pattern (`PinholeCameraPatternCfg`).
# Parkour‘s default CAMERA_CFG uses a different focal / resolution; HAL keeps pose from parkour but
# overrides rays so RGB and depth share FOV at native resolution (Isaac HAL does not resize).
_FRONT_RGB_HEIGHT = 480
_FRONT_RGB_WIDTH = 640
_FRONT_RGB_FOCAL_LENGTH = 24.0
_FRONT_RGB_HORIZONTAL_APERTURE = 20.955


def front_raycast_pattern_matching_rgb() -> PinholeCameraPatternCfg:
    """Ray-cast sampler matching ``front_rgb`` ``PinholeCameraCfg`` (focal_length + aperture, 4:3)."""

    ha = _FRONT_RGB_HORIZONTAL_APERTURE
    return PinholeCameraPatternCfg(
        focal_length=_FRONT_RGB_FOCAL_LENGTH,
        horizontal_aperture=ha,
        vertical_aperture=ha * (_FRONT_RGB_HEIGHT / float(_FRONT_RGB_WIDTH)),
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
    )


def side_raycast_pattern_matching_rgb() -> PinholeCameraPatternCfg:
    """Same as ``front_raycast_pattern_matching_rgb``: ``side_rgb`` uses identical pinhole intrinsics/resolution."""

    return front_raycast_pattern_matching_rgb()

# Right flank (~ body ``−Y``, ROS ``base_link``: ``X`` forward, ``Y`` left). Side optical axis matches
# the front camera tilt then is yawed −90° about body ``+Z`` (forward ``+X`` → right ``−Y``). Euler
# ``(180°, pitch ≈70°, ψ)`` is near gimbal lock, so ``ψ`` barely moves the optic axis; compose a
# fixed body yaw with the front rotation matrix instead of a second Euler triple.
_SIDE_POS = (0.0, -0.08, 0.12)


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


def _side_ros_offset_rot_from_front() -> tuple[float, ...]:
    q_front = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)
    r_front = _rotmat_camera_from_quat_ros_wxyz(q_front)
    yaw = math.radians(-90.0)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_side = rz @ r_front
    return _quat_ros_wxyz_from_rotmat(r_side)


def sim_rgbd_camera_cfgs_for_robot_link(link_name: str) -> tuple[
    RayCasterCameraCfg,
    CameraCfg,
    RayCasterCameraCfg,
    CameraCfg,
]:
    """Return ``(front_camera, front_rgb, side_camera, side_rgb)`` for HAL catalog wiring.

    Ray-cast depth sensors **must** use an existing articulation link prim (same pattern as
    ``parkour_tasks.default_cfg.CAMERA_CFG``), not a fictitious ``.../front_camera`` child:
    Isaac Lab resolves ``prim_path`` to a physics/RigidBody prim; ``offset`` poses the rays.

    Pinhole RGB cameras use ``spawn=`` and may live at ``.../<link>/front_rgb`` etc.
    """

    base = f"{{ENV_REGEX_NS}}/Robot/{link_name}"
    link_prim = base

    front_rot = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)

    front_camera = CAMERA_CFG.replace(
        prim_path=link_prim,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=_FRONT_POS,
            rot=front_rot,
            convention="ros",
        ),
        pattern_cfg=front_raycast_pattern_matching_rgb(),
        max_distance=2.0,
    )

    front_rgb = CameraCfg(
        prim_path=f"{base}/front_rgb",
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FRONT_RGB_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=_FRONT_RGB_HORIZONTAL_APERTURE,
            clipping_range=(0.3, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_FRONT_POS,
            rot=front_rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )

    side_rot = _side_ros_offset_rot_from_front()

    side_camera = CAMERA_CFG.replace(
        prim_path=link_prim,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=_SIDE_POS,
            rot=side_rot,
            convention="ros",
        ),
        pattern_cfg=side_raycast_pattern_matching_rgb(),
        max_distance=1.5,
    )

    side_rgb = CameraCfg(
        prim_path=f"{base}/side_rgb",
        height=_FRONT_RGB_HEIGHT,
        width=_FRONT_RGB_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FRONT_RGB_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=_FRONT_RGB_HORIZONTAL_APERTURE,
            clipping_range=(0.15, 4.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_SIDE_POS,
            rot=side_rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )

    return front_camera, front_rgb, side_camera, side_rgb


def attach_sim_rgbd_sensors_to_scene_cfg(scene_cfg, robot_link: str) -> None:
    """Mutate ``scene_cfg`` in place with HAL-facing sensor names (``front_rgb``, ``front_camera``, …)."""

    fc, fr, sc, sr = sim_rgbd_camera_cfgs_for_robot_link(robot_link)
    scene_cfg.front_camera = fc
    scene_cfg.front_rgb = fr
    scene_cfg.side_camera = sc
    scene_cfg.side_rgb = sr
