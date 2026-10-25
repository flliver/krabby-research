"""Isaac Sim HAL RGB-D catalog ids for rosbag recording and teleop bootstrap."""

from __future__ import annotations

from hal.server.isaac.sensor_backend_isaac import ISAAC_HAL_RGBD_CATALOG_IDS


def isaac_hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Catalog ids Isaac HAL may populate in ``rgbd_by_catalog_id`` (scene-dependent subset)."""
    return ISAAC_HAL_RGBD_CATALOG_IDS


def isaac_hal_teleop_bootstrap_catalog_ids() -> tuple[str, ...]:
    """Default teleop poll / QoS catalog ids (matches ``list_sensors()`` order for a full sim scene).

    Each logical ``*_rgbd`` row is followed by its ``*_gray16_depth`` WebRTC preview catalog id.
    """
    out: list[str] = []
    for cid in ISAAC_HAL_RGBD_CATALOG_IDS:
        out.append(cid)
        out.append(f"{cid}_gray16_depth")
    return tuple(out)
