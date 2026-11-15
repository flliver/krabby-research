"""HAL sim camera wiring table (no Isaac Lab imports).

``sim_rgbd_camera_cfgs`` builds ``CameraCfg`` objects from these rows;
``hal_server.set_observation`` iterates the same rows to fill ``rgbd_by_catalog_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hal.server.isaac.sim_depth_ranges import (
    ISAAC_FRONT_DEPTH_CLIP_M,
    ISAAC_TOF_DEPTH_CLIP_M,
)

_FRONT_POS = (0.33, 0.0, 0.08)
_FRONT_FPS = 30.0
_TOF_FPS = 20.0

# Keep in sync with pose constants in ``sim_rgbd_camera_cfgs.py``.
_BELLY_RIGHT_POS = (-0.14605, -0.5496, -0.19375)
_BELLY_LEFT_POS = (-0.14605, 0.5496, -0.19375)
_REAR_TOF_POS = (-0.3556, 0.0, -0.15875)


@dataclass(frozen=True)
class SimHalCameraEntry:
    """One HAL sim camera wiring row (iterated by attach / update_period / set_observation)."""

    catalog_id: str
    depth_attr: str
    rgb_attr: Optional[str]
    channels: tuple[str, ...]
    clip_m: tuple[float, float]
    fps: float
    pos: tuple[float, float, float]
    default_pose_xyz: tuple[float, float, float]


SIM_HAL_CAMERA_ENTRIES: tuple[SimHalCameraEntry, ...] = (
    SimHalCameraEntry(
        catalog_id="front_rgbd",
        depth_attr="front_camera",
        rgb_attr="front_rgb",
        channels=("rgb", "depth"),
        clip_m=ISAAC_FRONT_DEPTH_CLIP_M,
        fps=_FRONT_FPS,
        pos=_FRONT_POS,
        default_pose_xyz=_FRONT_POS,
    ),
    SimHalCameraEntry(
        catalog_id="belly_right_tof",
        depth_attr="belly_right_tof",
        rgb_attr=None,
        channels=("depth",),
        clip_m=ISAAC_TOF_DEPTH_CLIP_M,
        fps=_TOF_FPS,
        pos=_BELLY_RIGHT_POS,
        default_pose_xyz=_BELLY_RIGHT_POS,
    ),
    SimHalCameraEntry(
        catalog_id="belly_left_tof",
        depth_attr="belly_left_tof",
        rgb_attr=None,
        channels=("depth",),
        clip_m=ISAAC_TOF_DEPTH_CLIP_M,
        fps=_TOF_FPS,
        pos=_BELLY_LEFT_POS,
        default_pose_xyz=_BELLY_LEFT_POS,
    ),
    SimHalCameraEntry(
        catalog_id="rear_tof",
        depth_attr="rear_tof",
        rgb_attr=None,
        channels=("depth",),
        clip_m=ISAAC_TOF_DEPTH_CLIP_M,
        fps=_TOF_FPS,
        pos=_REAR_TOF_POS,
        default_pose_xyz=_REAR_TOF_POS,
    ),
)
