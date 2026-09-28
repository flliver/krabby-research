"""Shared fixtures for teleop unit tests."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from teleop.edge import robot_settings
from teleop.portal import settings as portal_settings


@pytest.fixture(autouse=True)
def _restore_robot_settings() -> None:
    """Snapshot and restore ``robot_settings`` module globals after each test."""
    snap = {
        "TELEOP_EDGE_MODE": robot_settings.TELEOP_EDGE_MODE,
        "SERVER_SIGNALING_WS_URL": robot_settings.SERVER_SIGNALING_WS_URL,
        "SERVER_RECONNECT_S": robot_settings.SERVER_RECONNECT_S,
        "MAX_VIDEO_M_LINES": robot_settings.MAX_VIDEO_M_LINES,
        "QOS_ENABLED": robot_settings.QOS_ENABLED,
        "QOS_KBPS_BUDGET_PER_STREAM": robot_settings.QOS_KBPS_BUDGET_PER_STREAM,
        "HTTP_AUTH_TOKEN": robot_settings.HTTP_AUTH_TOKEN,
        "STUN_TURN_SERVERS": copy.deepcopy(robot_settings.STUN_TURN_SERVERS),
    }
    yield
    for k, v in snap.items():
        setattr(robot_settings, k, copy.deepcopy(v) if k == "STUN_TURN_SERVERS" else v)


@pytest.fixture(autouse=True)
def _restore_portal_http_token() -> None:
    """Snapshot and restore ``portal_settings.HTTP_TOKEN`` after each test."""
    prev = portal_settings.HTTP_TOKEN
    yield
    portal_settings.HTTP_TOKEN = prev


@pytest.fixture
def default_portal_kwargs() -> dict:
    """Fresh ``create_portal_app`` kwargs (explicit ``build_*`` each test, no shared app singleton)."""
    from teleop.portal.ice_config import build_browser_ice_config
    from teleop.portal.settings import build_portal_auth_settings

    return {
        "browser_ice": build_browser_ice_config(),
        "portal_auth_settings": build_portal_auth_settings(),
    }
