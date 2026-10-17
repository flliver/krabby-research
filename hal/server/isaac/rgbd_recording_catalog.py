"""Isaac Sim HAL RGB-D catalog ids for rosbag recording and teleop bootstrap."""

from __future__ import annotations

from hal.server.isaac.sensor_backend_isaac import ISAAC_HAL_RGBD_CATALOG_IDS


def isaac_hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Catalog ids Isaac HAL may populate in ``rgbd_by_catalog_id`` (scene-dependent subset)."""
    return ISAAC_HAL_RGBD_CATALOG_IDS
