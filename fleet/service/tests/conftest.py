"""Shared pytest fixtures for krabby-fleet-service unit tests.

Installs a no-op signaling bridge on ``app.state`` before any TestClient
enters the FastAPI lifespan, so REST tests never open a real IoT MQTT
connection.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from krabby_fleet_service._signaling import SignalingBridge
from krabby_fleet_service.app import app


@pytest.fixture(autouse=True)
def _fleet_config_env() -> None:
    """Match committed fleet.toml (same as CI github_actions export)."""
    from krabby_fleet_config import load_fleet_config

    for key, value in load_fleet_config().as_env().items():
        os.environ.setdefault(key, value)


class NoopMqtt:
    """Stand-in MQTT client for REST-only tests (subscribe/publish are no-ops)."""

    connected = True

    def subscribe(self, topic: str, callback: Any) -> None:
        pass

    def unsubscribe(self, topic: str, callback: Any) -> None:
        pass

    def publish(self, topic: str, payload: str | bytes) -> None:
        pass

    def connect(self, **kwargs: Any) -> None:
        pass

    def disconnect(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _noop_signaling_bridge() -> Any:
    mqtt = NoopMqtt()
    app.state.mqtt = mqtt
    app.state.signaling_bridge = SignalingBridge(mqtt)
    yield
    app.state.mqtt = None
    app.state.signaling_bridge = None
