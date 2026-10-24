"""Unit tests for Isaac Sim HAL rgbd recording catalog."""

from __future__ import annotations

from hal.server.isaac.rgbd_recording_catalog import isaac_hal_rgbd_catalog_ids_for_recording
from hal.server.isaac.sensor_backend_isaac import ISAAC_HAL_RGBD_CATALOG_IDS


def test_isaac_catalog_ids_for_recording() -> None:
    assert isaac_hal_rgbd_catalog_ids_for_recording() == ISAAC_HAL_RGBD_CATALOG_IDS
    assert "front_rgbd" in ISAAC_HAL_RGBD_CATALOG_IDS
