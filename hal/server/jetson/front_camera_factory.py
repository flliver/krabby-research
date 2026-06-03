"""Pluggable factories for HAL RGB-D cameras (``RgbDepthCamera`` protocol).

Register drivers in ``FRONT_RGB_DEPTH_CAMERA_FACTORIES``. ``JetsonHalServer`` opens one
instance per ``JETSON_SENSOR_CATALOG`` rgbd row (primary always; others when ``hal_open_rgbd``).

``camera_driver`` values: ``"zed"`` (USB + pyzed; optional ``zed_serial_number``), ``"maixsense_a075v"``
(HTTP; accepts literal ``maixsense_host`` / ``maixsense_port`` on the catalog row).
"""

from __future__ import annotations

from typing import Callable, Optional

from hal.server.jetson.maixsense_rgb_depth_camera import (
    create_maixsense_a075v_rgb_depth_camera,
)
from hal.server.jetson.rgb_depth_camera import RgbDepthCamera
from hal.server.jetson.zed_camera import create_zed_camera

FrontRgbDepthCameraFactory = Callable[
    ...,
    Optional[RgbDepthCamera],
]


def _factory_zed(
    *,
    resolution: tuple[int, int],
    fps: int,
    depth_mode: str,
    zed_serial_number: Optional[int] = None,
) -> Optional[RgbDepthCamera]:
    return create_zed_camera(
        resolution=resolution,
        fps=fps,
        depth_mode=depth_mode,
        serial_number=zed_serial_number,
    )


def _factory_maixsense_a075v(
    *,
    resolution: tuple[int, int],
    fps: int,
    depth_mode: str,
    maixsense_host: Optional[str] = None,
    maixsense_port: Optional[int] = None,
) -> Optional[RgbDepthCamera]:
    return create_maixsense_a075v_rgb_depth_camera(
        resolution=resolution,
        fps=fps,
        depth_mode=depth_mode,
        maixsense_host=maixsense_host,
        maixsense_port=maixsense_port,
    )


FRONT_RGB_DEPTH_CAMERA_FACTORIES: dict[str, FrontRgbDepthCameraFactory] = {
    "zed": _factory_zed,
    "maixsense_a075v": _factory_maixsense_a075v,
}


def create_front_rgb_depth_camera(
    driver: str,
    *,
    resolution: tuple[int, int],
    fps: int,
    depth_mode: str,
    zed_serial_number: Optional[int] = None,
    maixsense_host: Optional[str] = None,
    maixsense_port: Optional[int] = None,
) -> Optional[RgbDepthCamera]:
    """Build an RGB-D camera using a registered ``driver`` name (any catalog rgbd row)."""
    try:
        factory = FRONT_RGB_DEPTH_CAMERA_FACTORIES[driver]
    except KeyError as e:
        known = ", ".join(sorted(FRONT_RGB_DEPTH_CAMERA_FACTORIES))
        raise ValueError(
            f"Unknown front RGB-D camera driver {driver!r}. Registered: {known}"
        ) from e
    if driver == "zed":
        return factory(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            zed_serial_number=zed_serial_number,
        )
    if driver == "maixsense_a075v":
        has_literal_host = bool(maixsense_host and str(maixsense_host).strip())
        if not has_literal_host:
            raise ValueError(
                "maixsense_a075v requires maixsense_host (literal) on JetsonSensorCatalogEntry"
            )
        return factory(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            maixsense_host=(maixsense_host.strip() if maixsense_host else None),
            maixsense_port=maixsense_port,
        )
    return factory(
        resolution=resolution,
        fps=fps,
        depth_mode=depth_mode,
    )
