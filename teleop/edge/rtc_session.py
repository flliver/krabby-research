"""Create WebRTC answers (aiortc) for a recvonly browser offer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from aiortc import RTCPeerConnection, RTCSessionDescription

from teleop.edge.sdp_util import (
    count_video_m_lines,
    offer_has_h264_video,
    video_m_line_budget_error_json,
)

logger = logging.getLogger(__name__)


async def wait_for_gathering_complete(pc: RTCPeerConnection, timeout_s: float = 10.0) -> None:
    if pc.iceGatheringState == "complete":
        return
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[None] = loop.create_future()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(None)

    try:
        await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "STUN/TURN candidate gathering did not complete within %ss; proceeding with partial SDP",
            timeout_s,
        )


async def create_answer_for_offer(
    offer_sdp: str,
    *,
    video_track_factory: Callable[[int], Any] | None = None,
    control_message_handler: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, RTCPeerConnection]:
    """Apply remote offer, attach one video track per ``m=video`` line, return (answer_sdp, pc).

    Caller must ``await pc.close()``. ``video_track_factory(i)`` is called once per recvonly video
    line (``i`` is 0 .. n-1). The factory must be provided when the offer requests video; there is
    no built-in synthetic source in production code.
    """
    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
    pc = RTCPeerConnection()
    await pc.setRemoteDescription(offer)
    n_video = count_video_m_lines(offer_sdp)
    assert video_track_factory is not None or n_video == 0
    for i in range(n_video):
        track: Any = video_track_factory(i)
        pc.addTrack(track)

    @pc.on("datachannel")
    def _on_datachannel(channel: Any) -> None:
        if channel.label != "krabby-control-v1":
            logger.info("Ignoring unexpected data channel label=%s", getattr(channel, "label", "<unknown>"))
            return

        @channel.on("message")
        def _on_message(message: Any) -> None:
            if control_message_handler is None:
                return
            if not isinstance(message, str):
                logger.warning("Rejected control message on %s: payload is not text JSON", channel.label)
                return
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Rejected control message on %s: invalid JSON", channel.label)
                return
            if not isinstance(payload, dict):
                logger.warning("Rejected control message on %s: JSON root is not an object", channel.label)
                return
            control_message_handler(payload)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_for_gathering_complete(pc)
    local = pc.localDescription
    assert local is not None
    return local.sdp, pc


async def handle_first_offer_message(
    payload: dict[str, Any],
    *,
    video_track_factory: Callable[[int], Any] | None = None,
    max_video_m_lines: int | None = None,
    control_message_handler: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str | None, str | None, RTCPeerConnection | None]:
    """Parse offer JSON. Returns (error_json, answer_sdp, pc) — only one of error or answer set."""
    if payload.get("type") != "offer":
        return (
            json.dumps({"type": "error", "message": "expected type offer with sdp"}),
            None,
            None,
        )
    sdp = payload.get("sdp")
    if not isinstance(sdp, str):
        return json.dumps({"type": "error", "message": "missing sdp"}), None, None
    budget_err = video_m_line_budget_error_json(sdp, max_video_m_lines)
    if budget_err is not None:
        n = count_video_m_lines(sdp)
        logger.warning(
            "rejected offer: %d video m-lines exceeds max_video_m_lines=%s",
            n,
            max_video_m_lines,
        )
        return budget_err, None, None
    n_vid = count_video_m_lines(sdp)
    if n_vid > 0 and not offer_has_h264_video(sdp):
        return (
            json.dumps(
                {
                    "type": "error",
                    "message": "offer rejected: H.264 is required for teleop video",
                }
            ),
            None,
            None,
        )
    if n_vid > 0 and video_track_factory is None:
        return (
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "video requires HAL camera tracks (run krabby-hal-server-jetson with "
                        "--teleop and robot teleop settings in teleop.edge.robot_settings)"
                    ),
                }
            ),
            None,
            None,
        )
    ans_sdp, pc = await create_answer_for_offer(
        sdp,
        video_track_factory=video_track_factory,
        control_message_handler=control_message_handler,
    )
    return None, ans_sdp, pc
