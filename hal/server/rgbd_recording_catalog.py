"""RGB-D catalog ids that HAL records to rosbag (from the sensor registry)."""

from __future__ import annotations

from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG


def hal_rgbd_catalog_ids_for_recording() -> tuple[str, ...]:
    """Logical catalog ids that produce ``/camera/{id}/rgb`` and ``/camera/{id}/depth``.

    Derived from ``JETSON_SENSOR_CATALOG``: every ``rgbd`` row that HAL opens
    (primary or ``hal_open_rgbd``). Isaac Sim uses the same catalog id strings
    in ``HardwareObservations.rgbd_by_catalog_id``.
    """
    ids: list[str] = []
    for entry in JETSON_SENSOR_CATALOG:
        if entry.type != "rgbd":
            continue
        if entry.is_primary or entry.hal_open_rgbd:
            ids.append(entry.id)
    return tuple(ids)
