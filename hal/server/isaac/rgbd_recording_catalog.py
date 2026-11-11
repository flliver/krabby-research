"""Isaac Sim HAL RGB-D catalog ids for rosbag recording and teleop bootstrap."""

from __future__ import annotations

from hal.server.isaac.sensor_backend_isaac import (
    ISAAC_HAL_RGBD_CATALOG_CHANNELS,
    ISAAC_HAL_RGBD_CATALOG_IDS,
)


def isaac_hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Catalog ids Isaac HAL may populate in ``rgbd_by_catalog_id`` (scene-dependent subset)."""
    return ISAAC_HAL_RGBD_CATALOG_IDS


def isaac_hal_rgbd_catalog_channels_for_recording() -> dict[str, tuple[str, ...]]:
    """Catalog id → declared channel names for Isaac HAL recording topics."""
    return dict(ISAAC_HAL_RGBD_CATALOG_CHANNELS)


def isaac_hal_teleop_bootstrap_catalog_ids() -> tuple[str, ...]:
    """Default teleop poll / QoS catalog ids (matches ``list_sensors()`` order for a full sim scene).

    RGB-D rows advertise the catalog id (RGB stream) plus ``*_gray16_depth``.
    Depth-only rows advertise only ``*_gray16_depth``.
    """
    out: list[str] = []
    for cid, channels in ISAAC_HAL_RGBD_CATALOG_CHANNELS.items():
        if "rgb" in channels:
            out.append(cid)
        out.append(f"{cid}_gray16_depth")
    return tuple(out)
