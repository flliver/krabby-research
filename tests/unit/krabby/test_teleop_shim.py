"""Unit tests for the agent teleop MQTT ↔ local WebSocket shim."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import pytest

from krabby.teleop_shim import (
    TeleopSignalingShim,
    local_signaling_ws_url,
    signaling_in_topic,
    signaling_out_topic,
)


class _FakeMqtt:
    """Minimal stand-in for awscrt MQTT connection (subscribe + publish)."""

    def __init__(self) -> None:
        self.subscriptions: dict[str, Any] = {}
        self.published: list[tuple[str, str, int]] = []

    def subscribe(self, *, topic: str, qos: int, callback: Any) -> None:
        self.subscriptions[topic] = callback

    def publish(self, *, topic: str, payload: str | bytes, qos: int) -> None:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        self.published.append((topic, text, qos))

    def inject(self, topic: str, payload: str | bytes) -> None:
        cb = self.subscriptions[topic]
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        cb(topic, raw)


class TestTopicHelpers:
    def test_signaling_topics(self) -> None:
        assert signaling_in_topic("bench-krabby-ci") == "teleop/bench-krabby-ci/signaling/in"
        assert signaling_out_topic("bench-krabby-ci") == "teleop/bench-krabby-ci/signaling/out"

    def test_local_ws_url_matches_m7_defaults(self) -> None:
        assert local_signaling_ws_url() == "ws://127.0.0.1:9000/ws/robot"


class TestTeleopSignalingShim:
    def test_mqtt_in_to_ws_and_ws_to_mqtt_out(self) -> None:
        mqtt = _FakeMqtt()
        # Ephemeral port so parallel pytest workers / busy 9000 don't collide.
        shim = TeleopSignalingShim(host="127.0.0.1", port=0)
        # Port 0 isn't supported by our TCPSite the same way — pick a free port.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        shim = TeleopSignalingShim(host="127.0.0.1", port=port)
        thing = "bench-krabby-ci"
        try:
            shim.start(mqtt, thing)
            assert signaling_in_topic(thing) in mqtt.subscriptions
            assert shim.ws_url == f"ws://127.0.0.1:{port}/ws/robot"

            async def _roundtrip() -> None:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(f"http://127.0.0.1:{port}/ws/robot") as ws:
                        # cloud → robot
                        offer = {"type": "hello", "role": "browser", "version": 1}
                        mqtt.inject(signaling_in_topic(thing), json.dumps(offer))
                        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                        assert msg.type == aiohttp.WSMsgType.TEXT
                        assert json.loads(msg.data) == offer

                        # robot → cloud
                        answer = {"type": "hello_ack", "version": 1}
                        await ws.send_str(json.dumps(answer))
                        deadline = time.monotonic() + 2.0
                        while time.monotonic() < deadline and not mqtt.published:
                            await asyncio.sleep(0.05)
                        assert mqtt.published, "expected MQTT publish on signaling/out"
                        topic, payload, qos = mqtt.published[-1]
                        assert topic == signaling_out_topic(thing)
                        assert json.loads(payload) == answer
                        assert qos == 1

            asyncio.run(_roundtrip())
        finally:
            shim.stop()

    def test_buffers_inbound_until_ws_connects(self) -> None:
        mqtt = _FakeMqtt()
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        shim = TeleopSignalingShim(host="127.0.0.1", port=port)
        thing = "robot-a"
        try:
            shim.start(mqtt, thing)
            early = {"type": "ping", "t": 42}
            mqtt.inject(signaling_in_topic(thing), json.dumps(early))

            async def _recv_buffered() -> dict[str, Any]:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(f"http://127.0.0.1:{port}/ws/robot") as ws:
                        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                        assert msg.type == aiohttp.WSMsgType.TEXT
                        return json.loads(msg.data)

            assert asyncio.run(_recv_buffered()) == early
        finally:
            shim.stop()
