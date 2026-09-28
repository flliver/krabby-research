"""Outbound **robot teleop agent** settings (dial-out WebSocket, ICE list for answers, offer caps).

Install **``krabby-teleop-edge``** on the robot only. The operator portal (**``krabby-teleop-portal``**)
is a separate package under ``teleop/portal/``; keep ``STUN_TURN_SERVERS`` here aligned with
``teleop.portal.ice_config`` on the portal host so browser and robot use the same ICE bootstrap.

Edit values here (same idea as ``data_collection/collector_settings.py``): checked-in module as
source of truth. Call :func:`build_teleop_edge_settings` at entry points and pass
:class:`teleop.edge.config.TeleopEdgeSettings` into APIs.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.turn_mint import append_env_turn_servers

logger = logging.getLogger(__name__)

# Built-in STUN entry used only when ``STUN_TURN_SERVERS`` (below) is empty or invalid after parsing.
BUILTIN_STUN_SERVERS: list[dict[str, Any]] = [
    {"urls": "stun:stun.l.google.com:19302"},
]

# --- Robot outbound signaling (edit for your deployment) ---

# ``off`` | ``agent``. ``agent`` without a non-empty ``SERVER_SIGNALING_WS_URL`` becomes ``off``.
TELEOP_EDGE_MODE: str = "off"

# WebSocket on the teleop server, e.g. ``wss://teleop.example.com/ws/robot``
SERVER_SIGNALING_WS_URL: str = ""

SERVER_RECONNECT_S: float = 5.0

# Max recvonly ``m=video`` lines per browser offer (clamped 1–32 in ``build_teleop_edge_settings``).
MAX_VIDEO_M_LINES: int = 8

# ICE list for the robot's WebRTC answers (align with ``teleop.portal.ice_config`` on the portal host).
STUN_TURN_SERVERS: list[dict[str, Any]] = copy.deepcopy(BUILTIN_STUN_SERVERS)

# If non-empty, appended as ``?token=`` on the robot's outbound signaling WebSocket URL.
HTTP_AUTH_TOKEN: str = ""

# QoS: lower fps then drop lowest-priority streams when outbound bitrate or loss exceeds budget.
QOS_ENABLED: bool = True

# Nominal per-stream bitrate budget (kbps) used by the degradation ladder.
# Calibrated for live teleop ``HalRgbSnapshotVideoTrack`` + aiortc VP8 (~100 kbps/stream),
# not full GStreamer H.264 tails (which can be 1–2 Mbps/stream).
QOS_KBPS_BUDGET_PER_STREAM: float = 120.0


def build_robot_signaling_ws_url(host_or_url: str, *, port: int = 9000) -> str:
    """Build ``ws://host:port/ws/robot`` from an IP/hostname, or pass through an existing URL."""
    s = (host_or_url or "").strip()
    if not s:
        raise ValueError("teleop host/IP must be non-empty")
    lower = s.lower()
    if lower.startswith("ws://") or lower.startswith("wss://"):
        return s
    return f"ws://{s}:{port}/ws/robot"


def build_teleop_edge_settings(
    *, host_or_url: str | None = None, control_echo_enabled: bool = False
) -> TeleopEdgeSettings:
    """Assemble :class:`TeleopEdgeSettings` from the module constants above, plus
    per-invocation overrides passed by the caller (mirrors ``host_or_url`` /
    HAL ``--teleop-ip``).

    When ``host_or_url`` is set (HAL ``--teleop-ip``), mode is forced to ``agent`` and the
    signaling URL is built from that host; module ``TELEOP_EDGE_MODE`` / ``SERVER_SIGNALING_WS_URL``
    are ignored for those fields.

    ``control_echo_enabled`` is not a module constant (unlike the settings
    above): it's a per-run instrumentation flag for the bench's E2E harness
    (HAL ``--teleop-control-echo``), not a persistent per-robot deployment
    setting, so it belongs on the invocation, not checked into this file.
    """
    reconnect = SERVER_RECONNECT_S
    if reconnect < 0.5:
        reconnect = 0.5

    max_lines = max(1, min(32, int(MAX_VIDEO_M_LINES)))

    ice: list[dict[str, Any]] = []
    for item in (STUN_TURN_SERVERS or [])[:32]:
        if isinstance(item, dict) and "urls" in item:
            ice.append(dict(item))
    if not ice:
        ice = copy.deepcopy(BUILTIN_STUN_SERVERS)
    ice = append_env_turn_servers(ice)

    qos_kbps = float(QOS_KBPS_BUDGET_PER_STREAM)
    if qos_kbps < 100.0:
        qos_kbps = 100.0

    if host_or_url is not None:
        mode = "agent"
        url = build_robot_signaling_ws_url(host_or_url)
    else:
        mode = (TELEOP_EDGE_MODE or "off").strip().lower()
        if mode not in ("off", "agent"):
            logger.warning("teleop: unknown TELEOP_EDGE_MODE %r; using off", TELEOP_EDGE_MODE)
            mode = "off"

        url = (SERVER_SIGNALING_WS_URL or "").strip() or None
        if mode == "agent" and not url:
            mode = "off"

    return TeleopEdgeSettings(
        mode=mode,
        server_signaling_ws_url=url,
        server_reconnect_s=reconnect,
        max_video_m_lines=max_lines,
        stun_turn_servers=ice,
        http_auth_token=(HTTP_AUTH_TOKEN or "").strip(),
        qos_enabled=bool(QOS_ENABLED),
        qos_kbps_budget_per_stream=qos_kbps,
        control_echo_enabled=bool(control_echo_enabled),
    )
