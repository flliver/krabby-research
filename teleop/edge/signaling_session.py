"""WebRTC signaling message loop for the robot agent (one WebSocket, client or server side)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from aiohttp import WSMsgType

from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.rtc_session import handle_first_offer_message
from teleop.edge.sdp_util import count_video_m_lines

logger = logging.getLogger(__name__)


async def run_robot_signaling_loop(
    ws: Any,
    *,
    teleop_settings: TeleopEdgeSettings,
    video_track_factory: Callable[[int], Any] | None = None,
    on_signaling_json: Callable[[dict[str, Any]], None] | None = None,
    pre_offer_validator: Callable[[dict[str, Any], int], None] | None = None,
    hello_ack_payload_builder: Callable[[], dict[str, Any]] | None = None,
    pong_payload_builder: Callable[[], dict[str, Any]] | None = None,
    control_message_handler: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Handle ping/hello/offer on a signaling WebSocket until close or error.

    ``video_track_factory`` is required when the browser offers recvonly video lines.
    Per-offer ``max_video_m_lines`` is read from ``teleop_settings`` (mutate the same instance
    between offers if you need changing caps in tests).

    Optional ``on_signaling_json`` is invoked for each ``hello`` and ``offer`` payload (after JSON
    decode) so the robot can apply ``catalog_ids`` from the viewer before answering offers.
    """
    pc_live: Any = None
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                payload: dict[str, Any] = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_str(json.dumps({"type": "error", "message": "invalid json"}))
                continue
            if payload.get("type") == "ping":
                extra_pong: dict[str, Any] = {}
                if pong_payload_builder is not None:
                    try:
                        pp = pong_payload_builder()
                    except Exception:
                        logger.warning("teleop: pong payload builder failed", exc_info=True)
                        pp = {}
                    if isinstance(pp, dict):
                        extra_pong.update(pp)
                if "t" in payload:
                    t = payload.get("t")
                    out = {"type": "pong", "t": t, "server_ms": time.time() * 1000.0}
                    if "t_wall_ms" in payload:
                        out["t_wall_ms"] = payload.get("t_wall_ms")
                    out.update(extra_pong)
                    await ws.send_str(json.dumps(out))
                else:
                    out = {"type": "pong"}
                    out.update(extra_pong)
                    await ws.send_str(json.dumps(out))
                continue
            if payload.get("type") == "hello":
                if on_signaling_json is not None:
                    on_signaling_json(payload)
                ver = payload.get("version", 1)
                if not isinstance(ver, int):
                    ver = 1
                ack_payload: dict[str, Any] = {"type": "hello_ack", "version": ver}
                if hello_ack_payload_builder is not None:
                    try:
                        extra = hello_ack_payload_builder()
                    except Exception:
                        logger.warning("teleop: hello_ack payload builder failed", exc_info=True)
                        extra = {}
                    if isinstance(extra, dict):
                        ack_payload.update(extra)
                await ws.send_str(json.dumps(ack_payload))
                continue
            if payload.get("type") != "offer":
                continue
            if pc_live is not None:
                await pc_live.close()
                pc_live = None
            if on_signaling_json is not None:
                on_signaling_json(payload)
            n_video = 1
            sdp_raw = payload.get("sdp")
            if isinstance(sdp_raw, str):
                n_video = count_video_m_lines(sdp_raw)
            if pre_offer_validator is not None:
                try:
                    pre_offer_validator(payload, n_video)
                except Exception as e:
                    await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
                    continue
            try:
                err_json, ans_sdp, pc = await handle_first_offer_message(
                    payload,
                    video_track_factory=video_track_factory,
                    max_video_m_lines=teleop_settings.max_video_m_lines,
                    control_message_handler=control_message_handler,
                )
            except Exception as e:
                logger.warning(
                    "teleop: failed to build WebRTC answer (offer ignored, connection stays up): %s",
                    e,
                    exc_info=True,
                )
                continue
            if err_json:
                await ws.send_str(err_json)
                if pc is not None:
                    await pc.close()
                continue
            assert ans_sdp is not None and pc is not None
            pc_live = pc
            await ws.send_str(json.dumps({"type": "answer", "sdp": ans_sdp}))
    finally:
        if pc_live is not None:
            await pc_live.close()
