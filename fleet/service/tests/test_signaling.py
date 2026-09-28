"""WebSocket teleop signaling bridge: MQTT in/out round-trip + auth gate."""
from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from krabby_fleet_service._auth import require_operator_websocket
from krabby_fleet_service._config import Settings, get_settings
from krabby_fleet_service._signaling import (
    SignalingBridge,
    parse_thing_from_out_topic,
    signaling_in_topic,
    signaling_out_topic,
)
from krabby_fleet_service.app import app

_SETTINGS = Settings(
    aws_region="us-east-1",
    cognito_user_pool_id="us-east-1_TESTPOOL",
    cognito_app_client_id="test-client-id",
    iot_ats_endpoint="example-ats.iot.us-east-1.amazonaws.com",
)
_FAKE_CLAIMS = {"sub": "test-operator", "cognito:groups": ["operator"]}


class FakeMqtt:
    def __init__(self) -> None:
        self.subscriptions: dict[str, Any] = {}
        self.published: list[tuple[str, str]] = []

    def subscribe(self, topic: str, callback: Any) -> None:
        self.subscriptions[topic] = callback

    def unsubscribe(self, topic: str, callback: Any) -> None:
        self.subscriptions.pop(topic, None)

    def publish(self, topic: str, payload: str | bytes) -> None:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        self.published.append((topic, text))

    def inject(self, topic: str, payload: str) -> None:
        # SignalingBridge registers under SIGNALING_OUT_FILTER; deliver via that cb.
        for filt, cb in list(self.subscriptions.items()):
            if filt == "teleop/+/signaling/out" or filt == topic:
                cb(topic, payload.encode("utf-8"))


class TestTopicHelpers:
    def test_topics(self) -> None:
        assert signaling_in_topic("bench-krabby-ci") == "teleop/bench-krabby-ci/signaling/in"
        assert signaling_out_topic("bench-krabby-ci") == "teleop/bench-krabby-ci/signaling/out"

    def test_parse_out_topic(self) -> None:
        assert parse_thing_from_out_topic("teleop/robot-a/signaling/out") == "robot-a"
        assert parse_thing_from_out_topic("teleop/robot-a/signaling/in") is None


@pytest.fixture
def bridge_client():
    mqtt = FakeMqtt()
    bridge = SignalingBridge(mqtt)
    app.state.mqtt = mqtt
    app.state.signaling_bridge = bridge
    app.dependency_overrides[get_settings] = lambda: _SETTINGS
    app.dependency_overrides[require_operator_websocket] = lambda: _FAKE_CLAIMS
    with TestClient(app) as client:
        yield client, mqtt
    app.dependency_overrides.clear()


@pytest.fixture
def anon_ws_client():
    mqtt = FakeMqtt()
    app.state.mqtt = mqtt
    app.state.signaling_bridge = SignalingBridge(mqtt)
    app.dependency_overrides[get_settings] = lambda: _SETTINGS
    # No auth override — real require_operator_websocket runs.
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_ws_browser_to_mqtt_in(bridge_client):
    client, mqtt = bridge_client
    thing = "bench-krabby-ci"
    with client.websocket_connect(f"/devices/{thing}/teleop/signaling") as ws:
        offer = {"type": "hello", "role": "browser", "version": 1}
        ws.send_text(json.dumps(offer))
    assert mqtt.published
    topic, payload = mqtt.published[-1]
    assert topic == signaling_in_topic(thing)
    assert json.loads(payload) == offer


def test_mqtt_out_to_ws_browser(bridge_client):
    client, mqtt = bridge_client
    thing = "bench-krabby-ci"
    with client.websocket_connect(f"/devices/{thing}/teleop/signaling") as ws:
        answer = {"type": "hello_ack", "version": 1}
        mqtt.inject(signaling_out_topic(thing), json.dumps(answer))
        got = json.loads(ws.receive_text())
        assert got == answer


def test_signaling_without_auth_rejected(anon_ws_client):
    with pytest.raises(Exception):
        with anon_ws_client.websocket_connect("/devices/bench-krabby-ci/teleop/signaling"):
            pass
