"""Unit tests for Isaac Sim HAL rgbd recording catalog."""

from __future__ import annotations

from hal.server.isaac.rgbd_recording_catalog import (
    isaac_hal_rgbd_catalog_ids_for_recording,
    isaac_hal_teleop_bootstrap_catalog_ids,
)
from hal.server.isaac.sensor_backend_isaac import ISAAC_HAL_RGBD_CATALOG_IDS


def test_isaac_catalog_ids_for_recording() -> None:
    assert isaac_hal_rgbd_catalog_ids_for_recording() == ISAAC_HAL_RGBD_CATALOG_IDS
    assert "front_rgbd" in ISAAC_HAL_RGBD_CATALOG_IDS


def test_isaac_teleop_bootstrap_includes_gray16_depth_rows() -> None:
    bootstrap = isaac_hal_teleop_bootstrap_catalog_ids()
    assert bootstrap == (
        "front_rgbd",
        "front_rgbd_gray16_depth",
        "side_right_rgbd",
        "side_right_rgbd_gray16_depth",
        "side_left_rgbd",
        "side_left_rgbd_gray16_depth",
    )
