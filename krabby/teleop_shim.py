"""MQTT ↔ local WebSocket bridge for teleop signaling.

``krabby agent`` already owns the device's single outbound MQTT connection.
This module adds the teleop half of that connection:

* Subscribe ``teleop/{thing}/signaling/in`` (cloud → robot)
* Publish ``teleop/{thing}/signaling/out`` (robot → cloud)

Message **shape** is unchanged from the existing teleop stack: the same JSON
text the portal relay forwards on ``/ws/robot`` (SDP offer/answer, ICE,
hello/ping, …).

Transport to the existing WebRTC edge agent is a localhost WebSocket server
on ``ws://127.0.0.1:9000/ws/robot`` — the URL ``build_robot_signaling_ws_url``
already builds from ``--teleop-ip 127.0.0.1``. No changes to ``teleop/edge``.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import threading
from typing import Any, Deque

# Match teleop.edge.robot_settings.build_robot_signaling_ws_url defaults so
# HAL ``--teleop-ip 127.0.0.1`` dials this shim with zero edge-side config.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_WS_PATH = "/ws/robot"

# Bound pending cloud→robot frames while the edge agent is reconnecting.
_PENDING_MAX = 64

def _signaling_trace_enabled() -> bool:
    """Per-frame MQTT/WS logs (incl. ping). Off by default — set ``KRABBY_TELEOP_SIGNALING_TRACE=1`` to debug."""
    return os.environ.get("KRABBY_TELEOP_SIGNALING_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _log_signaling_frame(*, direction: str, topic: str, text: str) -> None:
    """Log bridged signaling traffic without spamming journal on every ping."""
    nbytes = len(text.encode("utf-8"))
    if _signaling_trace_enabled():
        preview = text[:120].replace("\n", " ")
        print(
            f"[ok]  teleop signaling/{direction} topic={topic} bytes={nbytes}: {preview}",
            flush=True,
        )
        return
    msg_type: str | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            msg_type = str(parsed.get("type") or "")
    except json.JSONDecodeError:
        pass
    if msg_type in (None, "", "ping", "pong"):
        return
    print(
        f"[ok]  teleop signaling/{direction} topic={topic} type={msg_type} bytes={nbytes}",
        flush=True,
    )


def _await_crt(op: Any, *, timeout: float = 10.0) -> None:
    """Block on awscrt ``publish``/``subscribe`` (``Future`` or ``(Future, packet_id)``)."""
    if op is None:
        return
    future = op[0] if isinstance(op, tuple) else op
    future.result(timeout=timeout)


def signaling_in_topic(thing_name: str) -> str:
    return f"teleop/{thing_name}/signaling/in"


def signaling_out_topic(thing_name: str) -> str:
    return f"teleop/{thing_name}/signaling/out"


def local_signaling_ws_url(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_WS_PATH,
) -> str:
    """URL the existing WebRTC edge agent should dial (same shape as ``--teleop-ip`` builds)."""
    p = path if path.startswith("/") else f"/{path}"
    return f"ws://{host}:{port}{p}"


class TeleopSignalingShim:
    """Background localhost WS server bridged to IoT teleop signaling topics."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        path: str = DEFAULT_WS_PATH,
    ) -> None:
        self._host = host
        self._port = port
        self._path = path if path.startswith("/") else f"/{path}"
        self._connection: Any = None
        self._thing_name: str | None = None
        self._out_topic: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._runner: Any = None
        self._robot_ws: Any = None
        self._pending: Deque[str] = collections.deque(maxlen=_PENDING_MAX)
        self._lock = threading.Lock()
        self._start_error: BaseException | None = None
        self._qos_at_least_once: Any = None

    @property
    def ws_url(self) -> str:
        return local_signaling_ws_url(host=self._host, port=self._port, path=self._path)

    def robot_edge_connected(self) -> bool:
        ws = self._robot_ws
        return ws is not None and not getattr(ws, "closed", True)

    def start(self, connection: Any, thing_name: str) -> None:
        """Subscribe MQTT ``…/signaling/in`` and serve the local ``/ws/robot`` endpoint."""
        if self._thread is not None:
            raise RuntimeError("TeleopSignalingShim already started")

        self._connection = connection
        self._thing_name = thing_name
        self._out_topic = signaling_out_topic(thing_name)
        in_topic = signaling_in_topic(thing_name)

        from awscrt import mqtt

        self._qos_at_least_once = mqtt.QoS.AT_LEAST_ONCE
        try:
            _await_crt(
                connection.subscribe(
                    topic=in_topic,
                    qos=self._qos_at_least_once,
                    callback=self._on_mqtt_in,
                ),
                timeout=30.0,
            )
        except Exception as exc:
            raise RuntimeError(f"teleop signaling subscribe failed topic={in_topic}: {exc}") from exc
        print(f"[ok]  subscribed to {in_topic}")

        self._thread = threading.Thread(
            target=self._thread_main,
            name="teleop-signaling-shim",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            raise RuntimeError("teleop signaling shim failed to bind local WebSocket")
        if self._start_error is not None:
            err = self._start_error
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            raise RuntimeError(f"teleop signaling shim failed to start: {err}") from err
        print(f"[ok]  teleop signaling shim listening on {self.ws_url}")
        print(f"      point the WebRTC edge agent at it with --teleop-ip {self._host}")

    def stop(self) -> None:
        """Stop the local WS server (MQTT connection is owned by the agent)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._loop = None
        self._ready.clear()

    def _maybe_cold_start_locomotion(self) -> None:
        if self.robot_edge_connected():
            return
        from krabby._locomotion_config import request_locomotion_start

        request_locomotion_start()

    def _on_mqtt_in(self, topic: str, payload: bytes, **kwargs: Any) -> None:
        try:
            text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
        except UnicodeDecodeError:
            print(f"[err] teleop signaling/in: non-utf8 payload on {topic}", file=sys.stderr)
            return
        _log_signaling_frame(direction="in", topic=topic, text=text)
        self._maybe_cold_start_locomotion()
        loop = self._loop
        if loop is None or not loop.is_running():
            with self._lock:
                self._pending.append(text)
            return
        loop.call_soon_threadsafe(self._deliver_inbound, text)

    def _deliver_inbound(self, text: str) -> None:
        ws = self._robot_ws
        if ws is not None and not ws.closed:
            asyncio.create_task(self._safe_send(ws, text))
        else:
            self._maybe_cold_start_locomotion()
            with self._lock:
                self._pending.append(text)

    async def _safe_send(self, ws: Any, text: str) -> None:
        try:
            await ws.send_str(text)
        except Exception as exc:
            print(f"[err] teleop shim WS send failed: {exc}", file=sys.stderr)

    def _publish_out(self, text: str) -> None:
        if self._connection is None or self._out_topic is None:
            return
        try:
            _await_crt(
                self._connection.publish(
                    topic=self._out_topic,
                    payload=text,
                    qos=self._qos_at_least_once,
                ),
                timeout=10.0,
            )
            _log_signaling_frame(direction="out", topic=self._out_topic, text=text)
        except Exception as exc:
            print(
                f"[err] teleop signaling/out publish failed topic={self._out_topic}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _thread_main(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:
            self._start_error = exc
            self._ready.set()
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._async_main(web))
        except Exception as exc:
            self._start_error = exc
            print(f"[err] teleop signaling shim crashed: {exc}", file=sys.stderr)
            self._ready.set()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            if self._loop is loop:
                self._loop = None

    async def _async_main(self, web: Any) -> None:
        from aiohttp import WSMsgType

        shim = self

        async def robot_handler(request: Any) -> Any:
            ws = web.WebSocketResponse(heartbeat=30.0)
            await ws.prepare(request)

            prev = shim._robot_ws
            if prev is not None and not prev.closed:
                await prev.close()
            shim._robot_ws = ws
            print(f"[ok]  teleop edge WebSocket connected ({shim.ws_url})", flush=True)

            with shim._lock:
                pending = list(shim._pending)
                shim._pending.clear()
            for frame in pending:
                try:
                    await ws.send_str(frame)
                except Exception:
                    break

            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        data = msg.data
                        # Offload MQTT publish so the WS read loop stays responsive.
                        await asyncio.to_thread(shim._publish_out, data)
                    elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                        break
            finally:
                if shim._robot_ws is ws:
                    shim._robot_ws = None
                    print("[ok]  teleop edge WebSocket disconnected", flush=True)
            return ws

        app = web.Application()
        app.router.add_get(self._path, robot_handler)

        runner = web.AppRunner(app)
        self._runner = runner
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self._ready.set()

        while not self._stop.is_set():
            await asyncio.sleep(0.2)

        await self._async_shutdown()

    async def _async_shutdown(self) -> None:
        ws = self._robot_ws
        if ws is not None and not ws.closed:
            await ws.close()
        self._robot_ws = None
        runner = self._runner
        if runner is not None:
            await runner.cleanup()
            self._runner = None


def start_teleop_shim(connection: Any, thing_name: str, **kwargs: Any) -> TeleopSignalingShim:
    """Subscribe + bind; returns the running shim (call ``stop()`` on agent shutdown)."""
    shim = TeleopSignalingShim(**kwargs)
    shim.start(connection, thing_name)
    return shim
