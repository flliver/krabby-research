"""Outbound portal teleop (WebSocket signaling + WebRTC) using a dedicated HalClient for viewer frames."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any, Optional

import numpy as np
import zmq
from zmq.error import ContextTerminated

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server.robot_definition import RobotDefinition
from hal.server.sensor_interface import SensorInterface
from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.depth_preview import depth_meters_to_rgb24_u8
from teleop.edge.hal_rgb_track import HalRgbSnapshotVideoTrack
from teleop.edge.portal_client import portal_client_loop
from teleop.edge.viewer_catalog import parse_viewer_catalog_ids_from_payload
from teleop.edge.webrtc_input_controller import WebRTCInputController

logger = logging.getLogger(__name__)


class _ControlLatencyReporter:
    """Aggregate browser->robot control receive deltas from ``sent_browser_ms``."""

    _REPORT_INTERVAL_S = 5.0

    def __init__(self) -> None:
        self._samples_ms: list[float] = []
        self._total_samples = 0
        self._last_report_mono_s = time.monotonic()

    def observe_payload(self, payload: dict[str, Any]) -> None:
        raw_sent_ms = payload.get("sent_browser_ms")
        if isinstance(raw_sent_ms, bool) or not isinstance(raw_sent_ms, (int, float)):
            return
        sent_ms = float(raw_sent_ms)
        if not math.isfinite(sent_ms):
            return
        latency_ms = (time.time() * 1000.0) - sent_ms
        if not math.isfinite(latency_ms):
            return

        self._samples_ms.append(latency_ms)
        self._total_samples += 1

        now_mono_s = time.monotonic()
        if now_mono_s - self._last_report_mono_s >= self._REPORT_INTERVAL_S:
            self._report(now_mono_s)

    def _report(self, now_mono_s: float) -> None:
        if not self._samples_ms:
            self._last_report_mono_s = now_mono_s
            return
        samples = self._samples_ms
        p50_ms = self._percentile(samples, 50.0)
        p95_ms = self._percentile(samples, 95.0)
        max_ms = max(samples)
        latest_ms = samples[-1]
        logger.info(
            "teleop control latency: samples=%d total=%d p50=%.1fms p95=%.1fms max=%.1fms latest=%.1fms",
            len(samples),
            self._total_samples,
            p50_ms,
            p95_ms,
            max_ms,
            latest_ms,
        )
        self._samples_ms = []
        self._last_report_mono_s = now_mono_s

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            raise ValueError("percentile requires at least one value")
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (percentile / 100.0) * (len(ordered) - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return ordered[lo]
        frac = rank - lo
        return ordered[lo] + ((ordered[hi] - ordered[lo]) * frac)


class _OperatorOverrideGate:
    """Latest ``operator_override`` flag from portal control frames (thread-safe)."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = False

    def set(self, value: bool) -> None:
        with self._lock:
            self._value = value

    def get(self) -> bool:
        with self._lock:
            return self._value


class _ViewerCatalogIds:
    """Thread-safe catalog id list; browser updates via signaling ``catalog_ids``."""

    def __init__(self, bootstrap: list[str], teleop_settings: TeleopEdgeSettings) -> None:
        self._bootstrap = list(bootstrap)
        self._teleop_settings = teleop_settings
        self._lock = threading.Lock()
        self._active = list(bootstrap)

    def apply_from_payload(self, payload: dict[str, Any]) -> None:
        parsed = parse_viewer_catalog_ids_from_payload(
            payload, max_lines=self._teleop_settings.max_video_m_lines
        )
        if parsed is None:
            return
        new_ids = list(self._bootstrap) if not parsed else list(parsed)
        with self._lock:
            self._active = new_ids

    def snapshot_poll_ids(self) -> list[str]:
        with self._lock:
            ids = list(self._active)
        return ids if ids else list(self._bootstrap)

    def catalog_id_for_track(self, track_index: int) -> str:
        """One recvonly ``m=video`` line ⇒ one catalog id by index (**no clamping**)."""
        ids = self.snapshot_poll_ids()
        if track_index < 0 or track_index >= len(ids):
            return ""
        return ids[track_index]


def start_hal_teleop_signaling_thread(
    hal_client_config: HalClientConfig,
    transport_context: zmq.Context,
    sensor_interface: SensorInterface,
    *,
    stop_event: threading.Event,
    bootstrap_sensor_catalog_ids: list[str],
    teleop_edge_settings: TeleopEdgeSettings,
    robot_definition: RobotDefinition,
    send_hal_commands: bool = True,
) -> threading.Thread:
    """Start outbound teleop: HalClient poll thread + asyncio signaling; stop when ``stop_event`` is set.

    ``bootstrap_sensor_catalog_ids`` seeds HAL polling until the browser sends ``catalog_ids``
    on ``hello`` / ``offer`` (see portal viewer); empty list is invalid.

    ``robot_definition`` must match the HAL server's ``--robot`` topology so gamepad mapping
    produces the same joint count/order as ``apply_command``.

    When ``send_hal_commands`` is False, use ``HalClientConfig(..., command_endpoint=None)``
    so this thread only subscribes for WebRTC video (e.g. diagnostics). When inference and
    portal both PUSH to the HAL command socket, ``JointCommand.source`` (operator vs inference)
    selects precedence on the server.

    Portal ``operator_override`` (checkbox): when enabled, teleop sends operator joint commands
    (they win over autonomy when both queue); when disabled, teleop does not send commands.
    """

    def _thread_main() -> None:
        settings = teleop_edge_settings
        if not settings.agent_enabled or not settings.server_signaling_ws_url:
            logger.error(
                "Teleop requires TELEOP_EDGE_MODE=\"agent\" and SERVER_SIGNALING_WS_URL in "
                "teleop/edge/robot_settings.py; signaling thread not started",
            )
            return

        if not bootstrap_sensor_catalog_ids:
            logger.error("Teleop requires non-empty bootstrap_sensor_catalog_ids; signaling thread not started")
            return

        if send_hal_commands:
            if hal_client_config.command_endpoint is None:
                logger.error(
                    "Teleop send_hal_commands=True requires a non-null HalClientConfig.command_endpoint"
                )
                return
        elif hal_client_config.command_endpoint is not None:
            logger.error(
                "Teleop send_hal_commands=False requires HalClientConfig.command_endpoint=None "
                "(observation-only client)"
            )
            return

        catalog_state = _ViewerCatalogIds(bootstrap_sensor_catalog_ids, settings)
        available_sensors = sensor_interface.list_sensors()
        sensors_by_id = {s.id: s for s in available_sensors}
        available_catalog_ids = [s.id for s in available_sensors]

        def _validate_sensor_pipeline_binding(track_count: int) -> None:
            selected = catalog_state.snapshot_poll_ids()
            if len(selected) != track_count:
                raise ValueError(
                    "offer rejected: recvonly video m-line count "
                    f"({track_count}) must match catalog_ids length ({len(selected)}); "
                    f"snapshot={selected!r}"
                )
            for idx in range(track_count):
                cid = selected[idx]
                sensor = sensors_by_id.get(cid)
                if sensor is None:
                    raise ValueError(
                        f"offer rejected: catalog_id {cid!r} not found in SensorInterface.list_sensors()"
                    )
                if sensor.modality == "depth":
                    dr = (sensor.extra or {}).get("depth_range_m")
                    if (
                        not isinstance(dr, (list, tuple))
                        or len(dr) != 2
                        or not np.isfinite(float(dr[0]))
                        or not np.isfinite(float(dr[1]))
                        or float(dr[1]) <= float(dr[0])
                    ):
                        raise ValueError(
                            f"offer rejected: depth catalog_id {cid!r} requires "
                            f"SensorInfo.extra['depth_range_m'] = (d_min, d_max) with d_max > d_min"
                        )
                try:
                    handle = sensor_interface.get_gstreamer_handle(sensor)
                    pipeline = sensor_interface.build_pipeline(
                        handle,
                        encoding="h264",
                        output_element="fakesink",
                    )
                except Exception as e:
                    raise ValueError(
                        f"offer rejected: failed sensor pipeline preflight for catalog_id {cid!r}: {e}"
                    ) from e
                if "h264" not in pipeline.lower():
                    raise ValueError(
                        f"offer rejected: sensor pipeline for catalog_id {cid!r} is not H.264"
                    )

        latest_rgb: dict[str, np.ndarray] = {}
        latest_capture_ns: dict[str, int] = {}
        rgb_lock = threading.Lock()
        poll_stop = threading.Event()
        hal_teleop = HalClient(hal_client_config, context=transport_context)
        hal_client_lock = threading.Lock()
        hal_ready = threading.Event()
        operator_override_gate = _OperatorOverrideGate()
        webrtc_input: Optional[WebRTCInputController] = None
        if send_hal_commands:
            from controller.mappers.gamepad_to_krabby_hal_mapper import GamepadToKrabbyHALMapper

            webrtc_input = WebRTCInputController()
            gamepad_mapper = GamepadToKrabbyHALMapper(robot_definition=robot_definition)

            def _on_webrtc_state(state: Any) -> None:
                if not hal_ready.is_set():
                    return
                if not operator_override_gate.get():
                    return
                try:
                    cmd = gamepad_mapper.map(state, observation_timestamp_ns=None)
                    with hal_client_lock:
                        hal_teleop.put_joint_command(cmd)
                except Exception:
                    logger.warning("teleop control: failed to map/send command", exc_info=True)

            webrtc_input.register_callback(_on_webrtc_state)
            webrtc_input.start(update_rate_hz=50.0)

        def _poll_worker() -> None:
            try:
                while not poll_stop.is_set():
                    try:
                        with hal_client_lock:
                            obs = hal_teleop.poll(timeout_ms=15)
                    except ContextTerminated:
                        # Normal during shutdown when the shared ZMQ context is closed first.
                        logger.debug("teleop HAL client poll stopped (ZMQ context terminated)")
                        break
                    except Exception:
                        logger.warning(
                            "teleop HAL client poll failed; retrying",
                            exc_info=True,
                        )
                        continue
                    if obs is None or obs.rgbd_by_catalog_id is None:
                        continue
                    poll_ids = catalog_state.snapshot_poll_ids()
                    with rgb_lock:
                        for cid in poll_ids:
                            sensor = sensors_by_id.get(cid)
                            if (
                                sensor is not None
                                and sensor.modality == "depth"
                                and isinstance((sensor.extra or {}).get("gst_depth_source_catalog_id"), str)
                            ):
                                base = (sensor.extra or {})["gst_depth_source_catalog_id"]
                                chunk = obs.rgbd_by_catalog_id.get(base)
                                if chunk is None or chunk.depth.size == 0:
                                    continue
                                # depth_range_m: same contract as Gst depth sensors; validated in
                                # _validate_sensor_pipeline_binding before the offer is accepted.
                                dr = (sensor.extra or {})["depth_range_m"]
                                depth_range = (float(dr[0]), float(dr[1]))
                                preview = depth_meters_to_rgb24_u8(
                                    chunk.depth, depth_range_m=depth_range
                                )
                                latest_rgb[cid] = np.ascontiguousarray(preview, dtype=np.uint8)
                                latest_capture_ns[cid] = int(obs.timestamp_ns)
                                continue

                            chunk = obs.rgbd_by_catalog_id.get(cid)
                            if chunk is not None and chunk.rgb is not None and chunk.rgb.size > 0:
                                latest_rgb[cid] = np.ascontiguousarray(chunk.rgb, dtype=np.uint8)
                                latest_capture_ns[cid] = int(obs.timestamp_ns)
            finally:
                with hal_client_lock:
                    hal_teleop.close()

        with hal_client_lock:
            hal_teleop.initialize()
        hal_ready.set()
        poller = threading.Thread(target=_poll_worker, name="hal-teleop-hal-poll", daemon=True)
        poller.start()

        def _rgb_copy(catalog_id: str) -> Optional[np.ndarray]:
            with rgb_lock:
                arr = latest_rgb.get(catalog_id)
                if arr is None:
                    return None
                return np.copy(arr)

        def _make_track_factory():
            def factory(track_index: int) -> HalRgbSnapshotVideoTrack:
                cid = catalog_state.catalog_id_for_track(track_index)
                if not cid:
                    logger.error(
                        "teleop: catalog_id_for_track(%d) empty after validation; black until renegotiation",
                        track_index,
                    )
                return HalRgbSnapshotVideoTrack(frame_getter=lambda c=cid: _rgb_copy(c))

            return factory

        factory = _make_track_factory()

        def _on_signaling_json(payload: dict[str, Any]) -> None:
            catalog_state.apply_from_payload(payload)

        def _hello_ack_payload() -> dict[str, Any]:
            return {"available_catalog_ids": available_catalog_ids}

        def _pong_payload() -> dict[str, Any]:
            with rgb_lock:
                cap = dict(latest_capture_ns)
            return {"capture_timestamps_ns": cap}

        def _on_control_message(payload: dict[str, Any]) -> None:
            def _warn_rate_limited(msg: str) -> None:
                now = time.monotonic()
                if now - _last_bad_control_warn_mono[0] >= 10.0:
                    logger.warning(msg)
                    _last_bad_control_warn_mono[0] = now

            if payload.get("type") != "control":
                return
            control_latency.observe_payload(payload)
            operator_override_gate.set(bool(payload.get("operator_override", False)))
            st = payload.get("state")
            if not isinstance(st, dict):
                _warn_rate_limited("Rejected control message: missing or non-object 'state'")
                return
            if webrtc_input is None:
                return
            try:
                webrtc_input.update_from_payload(st)
            except Exception as e:
                _warn_rate_limited(f"Rejected control message: malformed controller payload ({e})")

        _last_bad_control_warn_mono = [0.0]
        control_latency = _ControlLatencyReporter()

        async def _run() -> None:
            task = asyncio.create_task(
                portal_client_loop(
                    settings.server_signaling_ws_url,
                    teleop_edge_settings=settings,
                    video_track_factory=factory,
                    on_signaling_json=_on_signaling_json,
                    pre_offer_validator=lambda _payload, n: _validate_sensor_pipeline_binding(n),
                    hello_ack_payload_builder=_hello_ack_payload,
                    pong_payload_builder=_pong_payload,
                    control_message_handler=_on_control_message,
                )
            )
            await asyncio.to_thread(stop_event.wait)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(_run())
        except Exception:
            logger.exception("Teleop signaling thread exited with error")
        finally:
            if webrtc_input is not None:
                webrtc_input.stop()
            poll_stop.set()
            poller.join(timeout=5.0)
            if poller.is_alive():
                logger.warning("Teleop HAL poll thread did not exit within timeout")

    t = threading.Thread(target=_thread_main, name="hal-teleop-signaling", daemon=True)
    t.start()
    return t
