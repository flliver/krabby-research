"""Unit tests for HAL Isaac sim RGB-D camera scene configs."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")

from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster.ray_caster_camera_cfg import RayCasterCameraCfg

from hal.server.isaac.sim_rgbd_camera_cfgs import sim_rgbd_camera_cfgs_for_robot_link


def test_hal_sim_depth_uses_pinhole_not_ground_raycast() -> None:
    fc, fr, src, srr, slc, slr = sim_rgbd_camera_cfgs_for_robot_link("crab_hex")

    for depth_cfg, rgb_cfg in ((fc, fr), (src, srr), (slc, slr)):
        assert isinstance(depth_cfg, CameraCfg)
        assert isinstance(rgb_cfg, CameraCfg)
        assert not isinstance(depth_cfg, RayCasterCameraCfg)
        assert depth_cfg.data_types == ["distance_to_camera"]
        assert rgb_cfg.data_types == ["rgb"]
        assert depth_cfg.offset.pos == rgb_cfg.offset.pos
        assert depth_cfg.offset.rot == rgb_cfg.offset.rot
        assert depth_cfg.height == rgb_cfg.height == 480
        assert depth_cfg.width == rgb_cfg.width == 640
