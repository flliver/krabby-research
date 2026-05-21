"""Isaac Sim GStreamer sensor backend.

**Live HAL**: pass ``scene_sensors`` (e.g. ``IsaacSimHalServer.camera_sensors``). When
``front_rgb`` + ``front_camera`` (and/or ``side_rgb`` + ``side_camera``) exist, each pair
adds **two** ``SensorInfo`` rows: ``front_rgbd`` / ``side_rgbd`` plus matching
``*_gray16_depth`` (``camera_driver="isaac_scene"``).

**Explicit configuration**: pass ``configured_sensors=(...)`` to supply a fixed
``SensorInfo`` list (call-site specific). If set, **scene introspection is skipped**.

**No scene and no config** → empty ``list_sensors()`` (nothing is invented).

Pipelines use appsrc; the application pushes rendered frames into GStreamer.
"""

from __future__ import annotations

from typing import Any, Optional

from hal.server.gstreamer_runtime import build_software_appsrc_encode_pipeline_string
from hal.server.sensor_interface import (
    GStreamerHandle,
    SensorInfo,
    SensorInterface,
    SensorPose,
)


def _sensor_infos_from_scene_cameras(scene_sensors: dict) -> list[SensorInfo]:
    """Map scene sensors to Jetson-matching catalog ids (``front_rgbd``, ``side_rgbd``, …).

    Full scene: **four** rows — ``front_rgbd``, ``front_rgbd_gray16_depth``, ``side_rgbd``,
    ``side_rgbd_gray16_depth``. Missing RGB or depth prim for a pair skips that pair's rows.
    """
    out: list[SensorInfo] = []

    if "front_rgb" in scene_sensors and "front_camera" in scene_sensors:
        rgb_cfg = getattr(scene_sensors["front_rgb"], "cfg", None)
        res, fps = _resolution_and_fps_from_cfg(rgb_cfg, (640, 480), 30)
        pose = _pose_from_sensor_cfg(rgb_cfg)
        out.append(
            SensorInfo(
                id="front_rgbd",
                type="rgbd",
                modality="rgbd",
                resolution=res,
                fps=fps,
                pose=pose,
                camera_driver="isaac_scene",
                extra={
                    "isaac_sensor_rgb": "front_rgb",
                    "isaac_sensor_depth": "front_camera",
                },
            )
        )
        out.append(
            SensorInfo(
                id="front_rgbd_gray16_depth",
                type="depth",
                modality="depth",
                resolution=res,
                fps=fps,
                pose=pose,
                camera_driver="isaac_scene",
                extra={
                    "isaac_sensor_depth": "front_camera",
                    # Match RayCaster ``max_distance`` (see ``hal.server.isaac.sim_rgbd_camera_cfgs``): teleop grayscale uses this band.
                    "depth_range_m": (0.2, 2.0),
                    "gst_depth_source_catalog_id": "front_rgbd",
                },
            )
        )

    if "side_rgb" in scene_sensors and "side_camera" in scene_sensors:
        rgb_cfg = getattr(scene_sensors["side_rgb"], "cfg", None)
        res, fps = _resolution_and_fps_from_cfg(rgb_cfg, (640, 480), 30)
        pose = _pose_from_sensor_cfg(rgb_cfg)
        out.append(
            SensorInfo(
                id="side_rgbd",
                type="rgbd",
                modality="rgbd",
                resolution=res,
                fps=fps,
                pose=pose,
                camera_driver="isaac_scene",
                extra={
                    "isaac_sensor_rgb": "side_rgb",
                    "isaac_sensor_depth": "side_camera",
                },
            )
        )
        out.append(
            SensorInfo(
                id="side_rgbd_gray16_depth",
                type="depth",
                modality="depth",
                resolution=res,
                fps=fps,
                pose=pose,
                camera_driver="isaac_scene",
                extra={
                    "isaac_sensor_depth": "side_camera",
                    "depth_range_m": (0.15, 1.5),
                    "gst_depth_source_catalog_id": "side_rgbd",
                },
            )
        )

    return out


def _tensor_list1d(t) -> list[float]:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return [float(x) for x in t.detach().cpu().flatten().tolist()]
    except ImportError:
        pass
    if hasattr(t, "tolist"):
        return [float(x) for x in t.tolist()]
    return [float(x) for x in t]


def _pose_from_sensor_cfg(cfg) -> Optional[SensorPose]:
    """Pose from Isaac Lab sensor ``cfg.offset`` (pos + quat), or None if unavailable."""
    if cfg is None:
        return None
    off = getattr(cfg, "offset", None)
    if off is None:
        return None
    pos = getattr(off, "pos", None)
    rot = getattr(off, "rot", None)
    if pos is None:
        return None
    p = _tensor_list1d(pos)
    if len(p) < 3:
        return None
    px, py, pz = p[0], p[1], p[2]
    if rot is None:
        return SensorPose(px, py, pz, 0.0, 0.0, 0.0, 1.0)
    r = _tensor_list1d(rot)
    if len(r) < 4:
        return SensorPose(px, py, pz, 0.0, 0.0, 0.0, 1.0)
    # Isaac Lab quat_from_euler_* returns (w, x, y, z)
    qw, qx, qy, qz = r[0], r[1], r[2], r[3]
    return SensorPose(px, py, pz, qx, qy, qz, qw)


def _resolution_and_fps_from_cfg(cfg, default_res: tuple[int, int], default_fps: int) -> tuple[tuple[int, int], int]:
    if cfg is None:
        return default_res, default_fps
    w = getattr(cfg, "width", None)
    h = getattr(cfg, "height", None)
    pc = getattr(cfg, "pattern_cfg", None)
    if (w is None or h is None) and pc is not None:
        w = getattr(pc, "width", w)
        h = getattr(pc, "height", h)
    res = (int(w), int(h)) if w is not None and h is not None else default_res
    upd = getattr(cfg, "update_period", None)
    if upd is None or float(upd) <= 0:
        return res, default_fps
    fps = max(1, int(round(1.0 / float(upd))))
    return res, fps


def _appsrc_sw_encode_pipeline(
    width: int,
    height: int,
    fps: int,
    format_caps: str = "RGB",
    encoding: str = "h264",
    output_element: str = "fakesink",
) -> str:
    """Build appsrc -> videoconvert -> software encode -> sink (for Isaac synthetic)."""
    return build_software_appsrc_encode_pipeline_string(
        width,
        height,
        fps,
        format_caps=format_caps,
        encoding=encoding,
        output_element=output_element,
    )


class IsaacSensorInterface(SensorInterface):
    """Isaac sensors: explicit ``configured_sensors``, or introspection from ``scene_sensors``."""

    def __init__(
        self,
        scene_sensors: Optional[dict] = None,
        *,
        configured_sensors: Optional[tuple[SensorInfo, ...]] = None,
    ) -> None:
        self.scene_sensors = dict(scene_sensors) if scene_sensors else {}
        if configured_sensors is not None:
            self._sensors = list(configured_sensors)
        else:
            self._sensors = _sensor_infos_from_scene_cameras(self.scene_sensors)

    def list_sensors(self) -> list[SensorInfo]:
        return list(self._sensors)

    def get_gstreamer_handle(self, sensor: SensorInfo) -> GStreamerHandle:
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
                backend_data={"extra": sensor.extra} if sensor.extra else None,
            )
        fmt = "GRAY8" if sensor.type == "radar" else "RGB"
        return GStreamerHandle(
            sensor_id=sensor.id,
            sensor_type=sensor.type,
            modality=sensor.modality,
            resolution=sensor.resolution,
            fps=sensor.fps,
            camera_driver=sensor.camera_driver,
            appsrc_pixel_format=fmt,
            backend_data={"extra": sensor.extra} if sensor.extra else None,
        )

    def build_pipeline(
        self,
        handle: GStreamerHandle,
        encoding: str = "h264",
        output_element: str = "fakesink",
        **kwargs: Any,
    ) -> str:
        w, h = handle.resolution
        fps = handle.fps
        return _appsrc_sw_encode_pipeline(
            width=w,
            height=h,
            fps=fps,
            format_caps=handle.appsrc_pixel_format,
            encoding=encoding,
            output_element=output_element,
        )
