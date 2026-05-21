"""SDP parsing helpers (no heavy WebRTC imports)."""

from __future__ import annotations

import json
import re

_M_VIDEO = re.compile(r"^m=video ", re.MULTILINE)
_RTPMAP_H264 = re.compile(r"^a=rtpmap:\d+\s+H264(?:/|\s|$)", re.MULTILINE | re.IGNORECASE)


def count_video_m_lines(sdp: str) -> int:
    """Count ``m=video`` sections in the remote offer (one per browser ``recvonly`` transceiver)."""
    n = len(_M_VIDEO.findall(sdp))
    return n if n > 0 else 1


def video_m_line_budget_error_json(sdp: str, max_m: int | None) -> str | None:
    """If ``max_m`` is set and the offer exceeds it, return a teleop JSON **error** string; else ``None``."""
    if max_m is None:
        return None
    n = count_video_m_lines(sdp)
    if n > max_m:
        return json.dumps(
            {
                "type": "error",
                "message": (
                    f"too many recvonly video m-lines ({n}); max is {max_m} "
                    f"(teleop.edge.robot_settings.MAX_VIDEO_M_LINES)"
                ),
            }
        )
    return None


def offer_has_h264_video(sdp: str) -> bool:
    """Return True when the offer SDP declares at least one H264 payload mapping."""
    return bool(_RTPMAP_H264.search(sdp))
