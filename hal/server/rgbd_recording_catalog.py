"""RGB-D catalog ids that HAL records to rosbag (from the active HAL backend registry)."""

from __future__ import annotations

from typing import Iterable, Protocol


class _RgbdCatalogRow(Protocol):
    id: str
    type: str
    is_primary: bool
    hal_open_rgbd: bool


# Declared observation channels by sensor modality / type. Transport stays generic;
# recording topic pre-registration composes on these declarations.
_RGBD_CHANNELS: tuple[str, ...] = ("rgb", "depth")
_DEPTH_ONLY_CHANNELS: tuple[str, ...] = ("depth",)


def channels_for_catalog_row(entry: _RgbdCatalogRow) -> tuple[str, ...]:
    """Channel names declared for one catalog row.

    Prefer an optional ``modality`` attribute when present (``\"depth\"`` → depth-only);
    otherwise RGB-D rows declare both ``rgb`` and ``depth``.
    """
    modality = getattr(entry, "modality", None)
    if modality == "depth":
        return _DEPTH_ONLY_CHANNELS
    return _RGBD_CHANNELS


def rgbd_ids_from_hal_catalog(catalog: Iterable[_RgbdCatalogRow]) -> tuple[str, ...]:
    """Every ``rgbd`` or depth-only camera row that HAL opens (primary or ``hal_open_rgbd``)."""
    ids: list[str] = []
    for entry in catalog:
        if entry.type not in ("rgbd", "depth"):
            continue
        if entry.is_primary or entry.hal_open_rgbd:
            ids.append(entry.id)
    return tuple(ids)


def catalog_channels_from_hal_catalog(
    catalog: Iterable[_RgbdCatalogRow],
) -> dict[str, tuple[str, ...]]:
    """Map catalog id → declared channel names for opened camera rows."""
    out: dict[str, tuple[str, ...]] = {}
    for entry in catalog:
        if entry.type not in ("rgbd", "depth"):
            continue
        if entry.is_primary or entry.hal_open_rgbd:
            out[entry.id] = channels_for_catalog_row(entry)
    return out


def hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Logical catalog ids for ``/camera/{id}/{channel}`` topics.

    Jetson: derived from ``JETSON_SENSOR_CATALOG``. Isaac Sim: ``ISAAC_HAL_RGBD_CATALOG_IDS``
    (same id strings for cross-backend bags). Whichever HAL server wheel is installed wins.
    """
    return tuple(hal_rgbd_catalog_channels_for_recording().keys())


def hal_rgbd_catalog_channels_for_recording() -> dict[str, tuple[str, ...]]:
    """Catalog id → channel names for rosbag topic pre-registration."""
    try:
        from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG
    except ModuleNotFoundError:
        pass
    else:
        return catalog_channels_from_hal_catalog(JETSON_SENSOR_CATALOG)

    from hal.server.isaac.rgbd_recording_catalog import (
        isaac_hal_rgbd_catalog_channels_for_recording,
    )

    return isaac_hal_rgbd_catalog_channels_for_recording()