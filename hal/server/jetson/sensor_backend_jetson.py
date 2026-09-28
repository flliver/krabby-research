"""Jetson sensor catalog: GStreamer caps + HAL RGB-D observation wiring.

Pipeline generation uses appsrc + nvenc (or software encode). ``JETSON_SENSOR_CATALOG``
rows describe every logical sensor; rgbd rows can opt into HAL capture via ``hal_open_rgbd``,
``policy_scan_slot``, ``zed_usb_serial``, and (for MaixSense) ``maixsense_host`` /
``maixsense_port`` (see ``JetsonHalServer``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from hal.server.jetson.front_camera_factory import FRONT_RGB_DEPTH_CAMERA_FACTORIES
from hal.server.sensor_interface import (
    GStreamerHandle,
    SensorInfo,
    SensorInterface,
    SensorPose,
)


@dataclass(frozen=True, slots=True)
class JetsonSensorCatalogEntry:
    """One logical sensor: all static metadata in a single row."""

    id: str
    type: str  # GStreamer branch: rgb, rgbd, radar (must match build_pipeline caps lookup)
    modality: str
    resolution: tuple[int, int]  # RGB resolution (W, H)
    fps: int
    pose: SensorPose
    appsrc_pixel_format: str  # caps: RGB or GRAY8 for appsrc (rgb/radar rows)
    # Depth map resolution (W, H). Must be set explicitly for every catalog row.
    depth_resolution: tuple[int, int]
    # Depth mode for RGB-D driver initialization (e.g. ZED: NEURAL/QUALITY/ULTRA).
    depth_mode: str
    is_primary: bool = False  # exactly one row: HAL front observation camera (resolution/fps defaults)
    # Required on ``is_primary`` rgbd rows: key in ``front_camera_factory.FRONT_RGB_DEPTH_CAMERA_FACTORIES``
    # (e.g. ``zed``, ``maixsense_a075v``). Other rows: logical capture id for listing/pipelines.
    camera_driver: str | None = None
    # HAL ``RgbDepthCamera`` capture (``JetsonHalServer``): primary row always opens when rgbd + driver
    # is registered. Non-primary rgbd rows open only when ``hal_open_rgbd`` is True.
    hal_open_rgbd: bool = False
    # Maps this row into legacy ``HardwareObservations`` policy scan slots (``scan_features`` /
    # ``side_scan_features``). Primary uses ``is_primary``; at most one other row may use ``"side"``.
    policy_scan_slot: Literal["side"] | None = None
    # For ``camera_driver=="zed"``: optional USB serial (int).
    # If unset, runtime opens the first detected ZED.
    zed_usb_serial: int | None = None
    # For ``camera_driver=="maixsense_a075v"``: literal host/port values.
    # Port must be explicit.
    maixsense_host: str | None = None
    maixsense_port: int | None = None
    # When set, ``JetsonSensorInterface.list_sensors()`` also lists ``{id}_gray16_depth`` (Gst GRAY16_LE).
    gst_depth_quant_range_m: tuple[float, float] | None = None


# ``JetsonHalServer`` uses ``front_observation_camera_catalog_entry()`` for resolution/fps and driver.
# Swap primary ``camera_driver`` between ``zed`` and ``maixsense_a075v`` for interchangeable front RGB-D.
JETSON_SENSOR_CATALOG: tuple[JetsonSensorCatalogEntry, ...] = (
    JetsonSensorCatalogEntry(
        id="front_rgbd",
        type="rgbd",
        modality="rgbd",
        # Native ZED stream observed in runtime logs for current Jetson stack.
        resolution=(672, 376),
        depth_resolution=(672, 376),
        fps=30,
        pose=SensorPose(
            x=0.33, y=0.0, z=0.08, qx=0.0, qy=0.0, qz=0.0, qw=1.0
        ),
        appsrc_pixel_format="RGB",
        is_primary=True,
        camera_driver="zed",
        depth_mode="NEURAL_PLUS",
        gst_depth_quant_range_m=(0.2, 25.0),
    ),
    # Right flank MaixSense (~ body −Y in ROS base frame; see sim ``_SIDE_RIGHT_POS``).
    JetsonSensorCatalogEntry(
        id="side_right_rgbd",
        type="rgbd",
        modality="rgbd",
        # MaixSense RGB is typically 640x480 for current config.
        resolution=(640, 480),
        # MaixSense depth payload is native 320x240 (no resize in HAL path).
        depth_resolution=(320, 240),
        fps=30,
        pose=SensorPose(
            x=0.0,
            y=-0.08,
            z=0.12,
            qx=0.0,
            qy=0.0,
            qz=-0.7071067811865476,
            qw=0.7071067811865476,
        ),
        appsrc_pixel_format="RGB",
        is_primary=False,
        camera_driver="maixsense_a075v",
        depth_mode="NEURAL",
        hal_open_rgbd=True,
        policy_scan_slot="side",
        maixsense_host="192.168.233.1",
        maixsense_port=80,
        gst_depth_quant_range_m=(0.15, 1.5),
    ),
    # Left flank MaixSense (~ body +Y). Reconfigure module to 192.168.233.2 over SSH
    # (/etc/init.d/rc.preboot); see docs/MAIXSENSE_A075V_SETUP.md.
    JetsonSensorCatalogEntry(
        id="side_left_rgbd",
        type="rgbd",
        modality="rgbd",
        resolution=(640, 480),
        depth_resolution=(320, 240),
        fps=30,
        pose=SensorPose(
            x=0.0,
            y=0.08,
            z=0.12,
            qx=0.0,
            qy=0.0,
            qz=0.7071067811865476,
            qw=0.7071067811865476,
        ),
        appsrc_pixel_format="RGB",
        is_primary=False,
        camera_driver="maixsense_a075v",
        depth_mode="NEURAL",
        hal_open_rgbd=True,
        maixsense_host="192.168.233.2",
        maixsense_port=80,
        gst_depth_quant_range_m=(0.15, 1.5),
    ),
)


_APPSRC_PIXEL_FORMAT_BY_SENSOR_ID: dict[str, str] = {
    e.id: e.appsrc_pixel_format for e in JETSON_SENSOR_CATALOG
}

JETSON_SENSOR_CATALOG_BY_ID: dict[str, JetsonSensorCatalogEntry] = {
    e.id: e for e in JETSON_SENSOR_CATALOG
}


def assert_hal_rgbd_catalog_config() -> None:
    """Validate rgbd HAL-related catalog rows (call before opening cameras)."""
    primary_rgbd = [
        e
        for e in JETSON_SENSOR_CATALOG
        if e.is_primary and e.type == "rgbd"
    ]
    if len(primary_rgbd) != 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG must contain exactly one is_primary rgbd row"
        )
    p = primary_rgbd[0]
    if not p.camera_driver or p.camera_driver not in FRONT_RGB_DEPTH_CAMERA_FACTORIES:
        raise RuntimeError(
            f"Primary rgbd row must set camera_driver registered in "
            f"FRONT_RGB_DEPTH_CAMERA_FACTORIES, got {p.camera_driver!r}"
        )
    side_rows = [e for e in JETSON_SENSOR_CATALOG if e.policy_scan_slot == "side"]
    if len(side_rows) > 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: at most one row may set policy_scan_slot='side'"
        )
    for e in side_rows:
        if e.is_primary:
            raise RuntimeError("policy_scan_slot='side' must not be set on the primary row")
        if not e.hal_open_rgbd:
            raise RuntimeError(
                f"Catalog {e.id!r}: policy_scan_slot='side' requires hal_open_rgbd=True"
            )
    for e in JETSON_SENSOR_CATALOG:
        w, h = e.resolution
        dw, dh = e.depth_resolution
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Catalog {e.id!r}: resolution must be positive, got {e.resolution!r}")
        if dw <= 0 or dh <= 0:
            raise RuntimeError(
                f"Catalog {e.id!r}: depth_resolution must be positive, got {e.depth_resolution!r}"
            )
        if e.type == "rgbd" and not str(e.depth_mode).strip():
            raise RuntimeError(f"Catalog {e.id!r}: depth_mode must be non-empty for rgbd rows")
    for e in JETSON_SENSOR_CATALOG:
        if e.type != "rgbd" or e.camera_driver != "maixsense_a075v":
            continue
        has_host_literal = bool(e.maixsense_host and str(e.maixsense_host).strip())
        if not has_host_literal:
            raise RuntimeError(
                f"Catalog {e.id!r}: camera_driver='maixsense_a075v' requires "
                f"non-empty maixsense_host (literal HTTP host)"
            )
        has_port_literal = e.maixsense_port is not None
        if not has_port_literal:
            raise RuntimeError(
                f"Catalog {e.id!r}: camera_driver='maixsense_a075v' requires "
                f"maixsense_port (literal HTTP port)"
            )


def front_observation_camera_catalog_entry() -> JetsonSensorCatalogEntry:
    """The unique catalog row with ``is_primary`` True (HAL front RGB-D defaults + driver id)."""
    primaries = [e for e in JETSON_SENSOR_CATALOG if e.is_primary]
    if len(primaries) != 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG must contain exactly one entry with is_primary=True"
        )
    entry = primaries[0]
    if not entry.camera_driver:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: the is_primary row must set camera_driver "
            "(registered key in hal.server.jetson.front_camera_factory.FRONT_RGB_DEPTH_CAMERA_FACTORIES)"
        )
    if entry.camera_driver not in FRONT_RGB_DEPTH_CAMERA_FACTORIES:
        known = ", ".join(sorted(FRONT_RGB_DEPTH_CAMERA_FACTORIES))
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: is_primary camera_driver "
            f"{entry.camera_driver!r} is not registered. Known: {known}"
        )
    return entry


def _jetson_default_sink_tail(encoding: str, use_nvenc: bool) -> str:
    """GStreamer tail after parser for encoded streams: decode + real display sink (Jetson)."""
    if use_nvenc:
        return "nvv4l2decoder ! nv3dsink"
    if encoding == "h265":
        return "avdec_h265 ! videoconvert ! autovideosink"
    return "avdec_h264 ! videoconvert ! autovideosink"


def _nvenc_pipeline(
    caps: str,
    width: int,
    height: int,
    fps: int,
    output_element: str,
    encoding: str = "h264",
    bitrate: int = 4_000_000,
    *,
    raw_input_gray16_le: bool = False,
) -> str:
    """Build Jetson nvenc pipeline: caps -> [videoconvert ->] nvvidconv -> nvv4l2h264enc -> parse -> sink."""
    # nvvidconv expects raw input; output NV12 in NVMM for encoder. GRAY16_LE needs CPU videoconvert first.
    if encoding == "h264":
        enc = f"nvv4l2h264enc bitrate={bitrate} ! h264parse"
    elif encoding == "h265":
        enc = "nvv4l2h265enc ! h265parse"
    elif encoding == "raw":
        enc = "identity"
    else:
        enc = f"nvv4l2h264enc bitrate={bitrate} ! h264parse"

    to_nvmm = (
        f"{caps} ! videoconvert ! nvvidconv"
        if raw_input_gray16_le
        else f"{caps} ! nvvidconv"
    )
    if encoding == "raw":
        return f"{caps} ! {output_element}"
    return f"{to_nvmm} ! video/x-raw(memory:NVMM),format=NV12 ! {enc} ! {output_element}"


def _sw_encode_pipeline(
    caps: str,
    output_element: str,
    encoding: str = "h264",
) -> str:
    """Software encode fallback when nvenc is not available (e.g. Orin Nano)."""
    if encoding == "h264":
        enc = "x264enc tune=zerolatency ! h264parse"
    elif encoding == "h265":
        enc = "x265enc ! h265parse"
    else:
        enc = "x264enc tune=zerolatency ! h264parse"
    return f"{caps} ! videoconvert ! {enc} ! {output_element}"


def _jetson_sensor_infos_from_catalog() -> list[SensorInfo]:
    """Catalog rows plus optional ``{id}_gray16_depth`` Gst depth streams."""
    sensors: list[SensorInfo] = []
    for entry in JETSON_SENSOR_CATALOG:
        sensors.append(
            SensorInfo(
                id=entry.id,
                type=entry.type,
                modality=entry.modality,
                resolution=entry.resolution,
                fps=entry.fps,
                pose=entry.pose,
                camera_driver=entry.camera_driver,
            )
        )
        if entry.gst_depth_quant_range_m is not None:
            lo, hi = entry.gst_depth_quant_range_m
            sensors.append(
                SensorInfo(
                    id=f"{entry.id}_gray16_depth",
                    type="depth",
                    modality="depth",
                    resolution=entry.resolution,
                    fps=entry.fps,
                    pose=entry.pose,
                    camera_driver=entry.camera_driver,
                    extra={
                        "depth_range_m": (float(lo), float(hi)),
                        "gst_depth_source_catalog_id": entry.id,
                    },
                )
            )
    return sensors


class JetsonSensorInterface(SensorInterface):
    """Jetson sensor interface: catalog rows plus optional per-row GRAY16 depth Gst entries.

    ``build_pipeline(..., output_element=None)`` defaults to a **decode + display** tail
    (``nvv4l2decoder ! nv3dsink`` with nvenc, or software decode + ``autovideosink``).
    Pass ``output_element="fakesink"`` (or ``filesink`` / ``udpsink`` / etc.) when you
    do not want a real video sink.
    """

    def __init__(self, *, use_nvenc: bool = True) -> None:
        self.use_nvenc = use_nvenc
        self._sensors = _jetson_sensor_infos_from_catalog()

    def list_sensors(self) -> list[SensorInfo]:
        return list(self._sensors)

    def get_gstreamer_handle(self, sensor: SensorInfo) -> GStreamerHandle:
        bd: dict[str, Any] = {"use_nvenc": self.use_nvenc}
        if sensor.modality == "depth" or sensor.type == "depth":
            dr = (sensor.extra or {}).get("depth_range_m")
            if dr is None or len(dr) != 2:
                raise ValueError(
                    "Depth Gst sensors require SensorInfo.extra['depth_range_m'] = (d_min, d_max) in meters"
                )
            return GStreamerHandle(
                sensor_id=sensor.id,
                sensor_type=sensor.type,
                modality=sensor.modality,
                resolution=sensor.resolution,
                fps=sensor.fps,
                camera_driver=sensor.camera_driver,
                appsrc_pixel_format="GRAY16_LE",
                depth_range_m=(float(dr[0]), float(dr[1])),
                backend_data=bd,
            )
        fmt = _APPSRC_PIXEL_FORMAT_BY_SENSOR_ID.get(sensor.id, "RGB")
        return GStreamerHandle(
            sensor_id=sensor.id,
            sensor_type=sensor.type,
            modality=sensor.modality,
            resolution=sensor.resolution,
            fps=sensor.fps,
            camera_driver=sensor.camera_driver,
            appsrc_pixel_format=fmt,
            backend_data=bd,
        )

    def build_pipeline(
        self,
        handle: GStreamerHandle,
        encoding: str = "h264",
        output_element: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        use_nvenc = kwargs.get(
            "use_nvenc",
            (handle.backend_data or {}).get("use_nvenc", self.use_nvenc)
            if isinstance(handle.backend_data, dict)
            else self.use_nvenc,
        )
        w, h = handle.resolution
        fps = handle.fps
        bitrate = kwargs.get("bitrate", 4_000_000)

        if handle.appsrc_pixel_format == "GRAY16_LE":
            if handle.depth_range_m is None:
                raise ValueError(
                    "GRAY16_LE depth pipelines require GStreamerHandle.depth_range_m = (d_min, d_max)"
                )
            caps = (
                f"appsrc name=src is-live=true format=time ! "
                f"video/x-raw,format=GRAY16_LE,width={w},height={h},framerate={fps}/1"
            )
            if encoding == "raw":
                sink = output_element if output_element is not None else "autovideosink"
                return f"{caps} ! videoconvert ! {sink}"
            sink = output_element if output_element is not None else _jetson_default_sink_tail(encoding, use_nvenc)
            if use_nvenc:
                return _nvenc_pipeline(
                    caps,
                    w,
                    h,
                    fps,
                    sink,
                    encoding=encoding,
                    bitrate=bitrate,
                    raw_input_gray16_le=True,
                )
            return _sw_encode_pipeline(caps, sink, encoding=encoding)

        fmt = handle.appsrc_pixel_format
        caps = (
            f"appsrc name=src is-live=true format=time ! "
            f"video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1"
        )

        if encoding == "raw":
            sink = output_element if output_element is not None else "autovideosink"
            return f"{caps} ! videoconvert ! {sink}"

        sink = output_element if output_element is not None else _jetson_default_sink_tail(encoding, use_nvenc)
        if use_nvenc:
            return _nvenc_pipeline(
                caps, w, h, fps, sink, encoding=encoding, bitrate=bitrate
            )
        return _sw_encode_pipeline(caps, sink, encoding=encoding)
