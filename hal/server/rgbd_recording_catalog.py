"""RGB-D catalog ids that HAL records to rosbag (from the active HAL backend registry)."""

from __future__ import annotations

from typing import Iterable, Protocol


class _RgbdCatalogRow(Protocol):
    id: str
    type: str
    is_primary: bool
    hal_open_rgbd: bool


def rgbd_ids_from_hal_catalog(catalog: Iterable[_RgbdCatalogRow]) -> tuple[str, ...]:
    """Every ``rgbd`` row that HAL opens (primary or ``hal_open_rgbd``)."""
    ids: list[str] = []
    for entry in catalog:
        if entry.type != "rgbd":
            continue
        if entry.is_primary or entry.hal_open_rgbd:
            ids.append(entry.id)
    return tuple(ids)


def hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Logical catalog ids for ``/camera/{id}/rgb`` and ``/camera/{id}/depth`` topics.

    Jetson: derived from ``JETSON_SENSOR_CATALOG``. Isaac Sim: ``ISAAC_HAL_RGBD_CATALOG_IDS``
    (same id strings for cross-backend bags). Whichever HAL server wheel is installed wins.
    """
    try:
        from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG
    except ModuleNotFoundError:
        pass
    else:
        return rgbd_ids_from_hal_catalog(JETSON_SENSOR_CATALOG)

    from hal.server.isaac.rgbd_recording_catalog import isaac_hal_rgbd_catalog_ids_for_recording

    return isaac_hal_rgbd_catalog_ids_for_recording()
