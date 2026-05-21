"""Robot agent: outbound WebSocket to the remote teleop server (e.g. ``/ws/robot`` on the portal)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional
from urllib.parse import quote

import aiohttp

from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.signaling_session import run_robot_signaling_loop

logger = logging.getLogger(__name__)


def server_signaling_ws_url_with_token(url: str, teleop_edge_settings: TeleopEdgeSettings) -> str:
    """Append ``teleop_edge_settings.http_auth_token`` as ``token`` query param when set."""
    tok = (teleop_edge_settings.http_auth_token or "").strip()
    if not tok:
        return url
    sep = "&" if ("?" in url) else "?"
    return f"{url}{sep}token={quote(tok, safe='')}"


async def portal_client_loop(
    url: str,
    *,
    teleop_edge_settings: TeleopEdgeSettings,
    video_track_factory: Callable[[int], Any] | None = None,
    on_signaling_json: Optional[Callable[[dict[str, Any]], None]] = None,
    pre_offer_validator: Optional[Callable[[dict[str, Any], int], None]] = None,
    hello_ack_payload_builder: Optional[Callable[[], dict[str, Any]]] = None,
    pong_payload_builder: Optional[Callable[[], dict[str, Any]]] = None,
    control_message_handler: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Reconnect forever. JSON is the same as the historical on-robot ``/ws`` signaling."""
    session = aiohttp.ClientSession()
    connect_url = server_signaling_ws_url_with_token(url, teleop_edge_settings)
    reconnect_s = teleop_edge_settings.server_reconnect_s
    last_disconnect_reason: str | None = None
    repeated_disconnect_count = 0
    try:
        while True:
            logger.info("teleop signaling: connecting to %s", connect_url)
            try:
                async with session.ws_connect(connect_url, heartbeat=30.0) as ws:
                    logger.info("teleop server signaling connected: %s", connect_url)
                    last_disconnect_reason = None
                    repeated_disconnect_count = 0
                    await run_robot_signaling_loop(
                        ws,
                        teleop_settings=teleop_edge_settings,
                        video_track_factory=video_track_factory,
                        on_signaling_json=on_signaling_json,
                        pre_offer_validator=pre_offer_validator,
                        hello_ack_payload_builder=hello_ack_payload_builder,
                        pong_payload_builder=pong_payload_builder,
                        control_message_handler=control_message_handler,
                    )
            except asyncio.CancelledError:
                break
            except aiohttp.ClientError as e:
                reason = str(e)
                if reason != last_disconnect_reason:
                    repeated_disconnect_count = 1
                    logger.warning(
                        "teleop server signaling disconnected (%s); retry in %ss",
                        e,
                        reconnect_s,
                    )
                    last_disconnect_reason = reason
                else:
                    repeated_disconnect_count += 1
                    if repeated_disconnect_count % 12 == 0:
                        logger.info(
                            "teleop server signaling still disconnected (%s); retry in %ss (attempt=%d)",
                            e,
                            reconnect_s,
                            repeated_disconnect_count,
                        )
                    else:
                        logger.debug(
                            "teleop server signaling still disconnected (%s); retry in %ss (attempt=%d)",
                            e,
                            reconnect_s,
                            repeated_disconnect_count,
                        )
            try:
                await asyncio.sleep(reconnect_s)
            except asyncio.CancelledError:
                break
    finally:
        await session.close()
