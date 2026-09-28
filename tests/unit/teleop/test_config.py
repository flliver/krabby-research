"""``TeleopEdgeSettings`` assembly from ``robot_settings`` module literals."""

from __future__ import annotations

import copy

from teleop.edge import robot_settings
from teleop.edge.robot_settings import build_teleop_edge_settings


def test_load_off_mode_without_url() -> None:
    robot_settings.TELEOP_EDGE_MODE = "off"
    robot_settings.SERVER_SIGNALING_WS_URL = ""
    robot_settings.HTTP_AUTH_TOKEN = ""
    s = build_teleop_edge_settings()
    assert s.mode == "off"
    assert not s.agent_enabled
    assert s.server_signaling_ws_url is None
    assert s.max_video_m_lines == 8
    assert s.http_auth_token == ""
    assert s.stun_turn_servers
    assert s.stun_turn_servers[0].get("urls") == "stun:stun.l.google.com:19302"


def test_stun_turn_servers_custom_list() -> None:
    robot_settings.STUN_TURN_SERVERS = [
        {"urls": "stun:stun1.example.com:19302"},
        {"urls": "turn:turn.example.com", "username": "u", "credential": "p"},
    ]
    s = build_teleop_edge_settings()
    assert len(s.stun_turn_servers) == 2
    assert s.stun_turn_servers[0]["urls"] == "stun:stun1.example.com:19302"
    assert s.stun_turn_servers[1]["username"] == "u"


def test_max_video_m_lines_constant() -> None:
    robot_settings.MAX_VIDEO_M_LINES = 3
    s = build_teleop_edge_settings()
    assert s.max_video_m_lines == 3


def test_agent_mode_without_url_is_off() -> None:
    robot_settings.TELEOP_EDGE_MODE = "agent"
    robot_settings.SERVER_SIGNALING_WS_URL = ""
    s = build_teleop_edge_settings()
    assert s.mode == "off"
    assert not s.agent_enabled


def test_agent_with_server_url() -> None:
    robot_settings.TELEOP_EDGE_MODE = "agent"
    robot_settings.SERVER_SIGNALING_WS_URL = "wss://teleop.example/ws/robot"
    s = build_teleop_edge_settings()
    assert s.mode == "agent"
    assert s.agent_enabled
    assert s.server_signaling_ws_url == "wss://teleop.example/ws/robot"


def test_unknown_edge_mode_treated_as_off() -> None:
    robot_settings.TELEOP_EDGE_MODE = "bogus"
    robot_settings.SERVER_SIGNALING_WS_URL = "wss://x/ws/robot"
    s = build_teleop_edge_settings()
    assert s.mode == "off"


def test_server_reconnect_s() -> None:
    robot_settings.TELEOP_EDGE_MODE = "agent"
    robot_settings.SERVER_SIGNALING_WS_URL = "wss://x/ws/robot"
    robot_settings.SERVER_RECONNECT_S = 2.5
    s = build_teleop_edge_settings()
    assert s.server_reconnect_s == 2.5


def test_empty_stun_list_falls_back_to_builtin() -> None:
    robot_settings.STUN_TURN_SERVERS = []
    s = build_teleop_edge_settings()
    assert s.stun_turn_servers == robot_settings.BUILTIN_STUN_SERVERS


def test_qos_settings_defaults() -> None:
    s = build_teleop_edge_settings()
    assert s.qos_enabled is True
    assert s.qos_kbps_budget_per_stream == 120.0


def test_qos_kbps_budget_clamped() -> None:
    robot_settings.QOS_KBPS_BUDGET_PER_STREAM = 10.0
    s = build_teleop_edge_settings()
    assert s.qos_kbps_budget_per_stream == 100.0


def test_host_or_url_overrides_module_signaling() -> None:
    robot_settings.TELEOP_EDGE_MODE = "off"
    robot_settings.SERVER_SIGNALING_WS_URL = ""
    s = build_teleop_edge_settings(host_or_url="10.0.0.130")
    assert s.mode == "agent"
    assert s.agent_enabled
    assert s.server_signaling_ws_url == "ws://10.0.0.130:9000/ws/robot"


def test_control_echo_enabled_defaults_false() -> None:
    s = build_teleop_edge_settings()
    assert s.control_echo_enabled is False


def test_control_echo_enabled_passed_through() -> None:
    """A per-run flag (HAL --teleop-control-echo), not a robot_settings.py constant."""
    s = build_teleop_edge_settings(control_echo_enabled=True)
    assert s.control_echo_enabled is True
