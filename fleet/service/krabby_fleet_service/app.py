"""krabby-fleet-service -- Cognito-authenticated REST + teleop signaling bridge.

Bound to 127.0.0.1:8080 (see `__main__.py`); Caddy terminates TLS and
reverse-proxies `/api/*` here. Device list/detail via Fleet Indexing and
Classic Shadow; Secure Tunneling open/close for SSH; WebSocket teleop
signaling bridged to IoT MQTT ``teleop/{thing}/signaling/in|out``; ICE
server list with short-lived coturn TURN credentials.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Request, WebSocket
from pydantic import BaseModel

from krabby_fleet_service._auth import require_operator, require_operator_websocket
from krabby_fleet_service._config import Settings, get_settings
from krabby_fleet_service._devices import get_device, list_devices
from krabby_fleet_service._ice import build_ice_servers
from krabby_fleet_service._mqtt import FleetMqttClient
from krabby_fleet_service._signaling import SignalingBridge
from krabby_fleet_service._tunnels import close_ssh_tunnel, open_ssh_tunnel

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Tests may inject a pre-built bridge (and skip real IoT MQTT).
    if getattr(app.state, "signaling_bridge", None) is not None:
        bridge: SignalingBridge = app.state.signaling_bridge
        await bridge.start(asyncio.get_running_loop())
        yield
        await bridge.stop()
        return

    settings = get_settings()
    logger.info(
        "fleet startup region=%s cognito_pool=%s iot_ats=%s",
        settings.aws_region,
        settings.cognito_user_pool_id,
        settings.iot_ats_endpoint,
    )
    mqtt = FleetMqttClient()
    mqtt.connect(endpoint=settings.iot_ats_endpoint, region=settings.aws_region)
    bridge = SignalingBridge(mqtt)
    app.state.mqtt = mqtt
    app.state.signaling_bridge = bridge
    await bridge.start(asyncio.get_running_loop())
    logger.info("fleet startup complete signaling bridge ready")
    try:
        yield
    finally:
        await bridge.stop()
        mqtt.disconnect()


app = FastAPI(title="krabby-fleet-service", lifespan=lifespan)


class SshTunnelResponse(BaseModel):
    tunnelId: str
    sourceAccessToken: str
    region: str


class DeviceSummary(BaseModel):
    thingName: str
    connected: bool
    connectivityTimestamp: int | None = None
    reported: dict[str, Any] = {}


class DeviceDetail(BaseModel):
    thingName: str
    thingTypeName: str | None = None
    attributes: dict[str, str] = {}
    connected: bool
    connectivityTimestamp: int | None = None
    reported: dict[str, Any] = {}


@app.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    mqtt: FleetMqttClient | None = getattr(request.app.state, "mqtt", None)
    bridge: SignalingBridge | None = getattr(request.app.state, "signaling_bridge", None)
    body: dict[str, Any] = {"status": "ok", "mqttConnected": bool(mqtt and mqtt.connected)}
    if bridge is not None:
        body.update(bridge.diagnostics())
    return body


@app.get("/devices", response_model=list[DeviceSummary])
def get_devices(_claims: dict[str, Any] = Depends(require_operator)) -> list[dict[str, Any]]:
    return list_devices()


@app.get("/devices/{thing_name}", response_model=DeviceDetail)
def get_device_detail(
    thing_name: str, _claims: dict[str, Any] = Depends(require_operator)
) -> dict[str, Any]:
    return get_device(thing_name)


@app.post("/devices/{thing_name}/ssh-tunnel", response_model=SshTunnelResponse)
def create_ssh_tunnel(
    thing_name: str, _claims: dict[str, Any] = Depends(require_operator)
) -> dict[str, Any]:
    return open_ssh_tunnel(thing_name)


@app.delete("/devices/{thing_name}/ssh-tunnel/{tunnel_id}", status_code=204)
def delete_ssh_tunnel(
    thing_name: str, tunnel_id: str, _claims: dict[str, Any] = Depends(require_operator)
) -> None:
    close_ssh_tunnel(tunnel_id)


@app.get("/teleop/ice-servers")
def get_teleop_ice_servers(
    claims: dict[str, Any] = Depends(require_operator),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """STUN + short-lived coturn TURN credentials for browser ``RTCPeerConnection``."""
    user_id = str(claims.get("sub") or claims.get("username") or "operator")
    return build_ice_servers(settings, user_id=user_id)


@app.websocket("/devices/{thing_name}/teleop/signaling")
async def teleop_signaling(
    websocket: WebSocket,
    thing_name: str,
    claims: dict[str, Any] = Depends(require_operator_websocket),
) -> None:
    logger.debug(
        "teleop signaling auth ok thing=%s sub=%s",
        thing_name,
        claims.get("sub") or claims.get("username"),
    )
    bridge: SignalingBridge | None = getattr(websocket.app.state, "signaling_bridge", None)
    if bridge is None:
        logger.error("teleop signaling rejected thing=%s: bridge unavailable", thing_name)
        await websocket.close(code=1011, reason="signaling bridge unavailable")
        return
    await bridge.attach(websocket, thing_name)
