"""Tests for teleop robot signaling URL construction."""

import pytest

from teleop.edge.robot_settings import build_robot_signaling_ws_url


def test_builds_default_port_and_path_from_ip():
    assert build_robot_signaling_ws_url("10.0.0.130") == "ws://10.0.0.130:9000/ws/robot"


def test_builds_from_hostname():
    assert build_robot_signaling_ws_url("teleop.local") == "ws://teleop.local:9000/ws/robot"


def test_passes_through_existing_ws_url():
    url = "wss://teleop.example.com/ws/robot"
    assert build_robot_signaling_ws_url(url) == url


def test_rejects_empty_host():
    with pytest.raises(ValueError, match="non-empty"):
        build_robot_signaling_ws_url("")
