"""Typed teleop settings object; values come from Python modules (see ``robot_settings``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TeleopEdgeSettings:
    """Robot teleop agent: ``agent`` mode dials ``server_signaling_ws_url``; ``off`` disables.

    Construct with :func:`teleop.edge.robot_settings.build_teleop_edge_settings` from module
    literals, or build instances in tests / composition roots; pass explicitly into APIs.
    """

    mode: str  # off | agent
    server_signaling_ws_url: str | None
    server_reconnect_s: float
    max_video_m_lines: int
    stun_turn_servers: list[dict[str, Any]]
    http_auth_token: str = ""
    qos_enabled: bool = True
    qos_kbps_budget_per_stream: float = 120.0
    # Bench/test-only: echoes WebRTCInputController.get_state() onto the
    # telemetry channel as `last_control` so a test can confirm a control
    # message was actually applied. No operator-facing feature reads it;
    # off by default.
    control_echo_enabled: bool = False

    @property
    def agent_enabled(self) -> bool:
        return self.mode == "agent" and bool(self.server_signaling_ws_url)
