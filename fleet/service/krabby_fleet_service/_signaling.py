"""Browser WebSocket ↔ IoT MQTT teleop signaling bridge.

Public URL (via Caddy ``/api`` strip): ``/devices/{thingName}/teleop/signaling``.

Direction (same topics as ``krabby.teleop_shim`` on the robot):

* Browser → WS text → publish ``teleop/{thing}/signaling/in`` (cloud → device)
* MQTT ``teleop/{thing}/signaling/out`` → WS text → browser (device → cloud)

JSON shape is unchanged from the existing teleop stack (SDP/ICE/hello/ping).
One persistent fleet MQTT connection (see ``_mqtt``) fans out
``teleop/+/signaling/out`` to active WebSocket sessions keyed by thing name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Thing names from enroll / IoT — keep path params from becoming wild topics.
_THING_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,127}$")

SIGNALING_OUT_FILTER = "teleop/+/signaling/out"


def signaling_in_topic(thing_name: str) -> str:
    return f"teleop/{thing_name}/signaling/in"


def signaling_out_topic(thing_name: str) -> str:
    return f"teleop/{thing_name}/signaling/out"


def parse_thing_from_out_topic(topic: str) -> str | None:
    """Extract thing name from ``teleop/{thing}/signaling/out``."""
    parts = topic.split("/")
    if len(parts) == 4 and parts[0] == "teleop" and parts[2] == "signaling" and parts[3] == "out":
        return parts[1]
    return None


class SignalingBridge:
    """Fan-in/fan-out between Cognito-authed browser sockets and IoT MQTT."""

    def __init__(self, mqtt: Any) -> None:
        self._mqtt = mqtt
        self._loop: asyncio.AbstractEventLoop | None = None
        # thing_name -> set of WebSocket
        self._sessions: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._mqtt_subscribed = False

    def diagnostics(self) -> dict[str, Any]:
        """Non-secret runtime state for ``GET /healthz`` (one-deploy triage)."""
        return {
            "signalingOutSubscribed": self._mqtt_subscribed,
            "activeSessionsByThing": {
                thing: len(sockets) for thing, sockets in self._sessions.items()
            },
        }

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if not self._mqtt_subscribed:
            self._mqtt.subscribe(SIGNALING_OUT_FILTER, self._on_mqtt_out)
            self._mqtt_subscribed = True
            logger.info("signaling bridge subscribed to %s", SIGNALING_OUT_FILTER)

    async def stop(self) -> None:
        if self._mqtt_subscribed:
            self._mqtt.unsubscribe(SIGNALING_OUT_FILTER, self._on_mqtt_out)
            self._mqtt_subscribed = False
        async with self._lock:
            sockets = [ws for group in self._sessions.values() for ws in group]
            self._sessions.clear()
        for ws in sockets:
            try:
                await ws.close()
            except Exception:
                pass

    def _on_mqtt_out(self, topic: str, payload: bytes) -> None:
        thing = parse_thing_from_out_topic(topic)
        if thing is None:
            logger.warning("signaling mqtt/out ignored topic=%s (unexpected shape)", topic)
            return
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("signaling/out non-utf8 on %s", topic)
            return
        msg_type = "?"
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                msg_type = str(parsed.get("type") or "?")
        except json.JSONDecodeError:
            msg_type = "non-json"
        logger.debug(
            "signaling mqtt→browser topic=%s thing=%s type=%s bytes=%d",
            topic,
            thing,
            msg_type,
            len(payload),
        )
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        def _schedule_fanout() -> None:
            asyncio.create_task(self._fanout(thing, text))

        loop.call_soon_threadsafe(_schedule_fanout)

    async def _fanout(self, thing_name: str, text: str) -> None:
        async with self._lock:
            sockets = list(self._sessions.get(thing_name, ()))
        if not sockets:
            logger.warning(
                "signaling mqtt→browser dropped thing=%s bytes=%d (no active WebSocket)",
                thing_name,
                len(text.encode("utf-8")),
            )
            return
        dead: list[WebSocket] = []
        sent = 0
        for ws in sockets:
            try:
                await ws.send_text(text)
                sent += 1
            except Exception:
                dead.append(ws)
        logger.debug(
            "signaling mqtt→browser delivered thing=%s sockets=%d sent=%d dead=%d",
            thing_name,
            len(sockets),
            sent,
            len(dead),
        )
        if dead:
            async with self._lock:
                group = self._sessions.get(thing_name)
                if group is not None:
                    for ws in dead:
                        group.discard(ws)
                    if not group:
                        del self._sessions[thing_name]

    async def attach(self, websocket: WebSocket, thing_name: str) -> None:
        """Accept ``websocket`` and bridge until the client disconnects."""
        if not _THING_NAME_RE.match(thing_name):
            await websocket.close(code=1008, reason="invalid thing name")
            return

        await websocket.accept()
        async with self._lock:
            self._sessions.setdefault(thing_name, set()).add(websocket)
        peer = websocket.client
        logger.info("teleop signaling session open thing=%s", thing_name)

        received_browser_frame = False
        try:
            while True:
                if not received_browser_frame:
                    try:
                        text = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "teleop signaling stall thing=%s peer=%s: WebSocket accepted but "
                            "no text frame in 30s — suspect Caddy/proxy or browser not sending",
                            thing_name,
                            peer,
                        )
                        text = await websocket.receive_text()
                else:
                    text = await websocket.receive_text()
                received_browser_frame = True

                topic = signaling_in_topic(thing_name)
                msg_type = "?"
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        msg_type = str(parsed.get("type") or "?")
                except json.JSONDecodeError:
                    msg_type = "non-json"
                logger.debug(
                    "signaling browser→mqtt thing=%s topic=%s type=%s bytes=%d",
                    thing_name,
                    topic,
                    msg_type,
                    len(text.encode("utf-8")),
                )
                try:
                    await asyncio.to_thread(self._mqtt.publish, topic, text)
                except Exception:
                    logger.exception(
                        "publish signaling/in failed thing=%s topic=%s", thing_name, topic
                    )
                    await websocket.close(code=1011, reason="mqtt publish failed")
                    break
                logger.debug("signaling browser→mqtt ack thing=%s topic=%s", thing_name, topic)
        except WebSocketDisconnect as disc:
            logger.debug(
                "teleop signaling disconnected thing=%s code=%s",
                thing_name,
                getattr(disc, "code", None),
            )
        except Exception:
            logger.exception(
                "teleop signaling receive failed thing=%s state=%s",
                thing_name,
                websocket.client_state,
            )
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="signaling receive error")
        finally:
            async with self._lock:
                group = self._sessions.get(thing_name)
                if group is not None:
                    group.discard(websocket)
                    if not group:
                        del self._sessions[thing_name]
            logger.info("teleop signaling session closed thing=%s", thing_name)
