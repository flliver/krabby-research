#!/usr/bin/env python3
"""One-shot MQTT signaling smoke test (same CRT path as krabby-fleet-service).

Run on the fleet EC2 host *before* or *after* deploy to separate IoT IAM/MQTT
issues from the WebSocket bridge. Uses the instance role (no static keys).

  sudo -u krabby-fleet \
    AWS_REGION=us-east-2 AWS_DEFAULT_REGION=us-east-2 \
    python3 /opt/krabby-fleet-service/src/scripts/signaling_mqtt_smoke.py \
    --thing bench-krabby-ci

Exit 0 only if connect, subscribe, and publish to ``teleop/{thing}/signaling/in``
succeed (QoS1 ack). Watch the same topic in the IoT console MQTT test client.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from krabby_fleet_service._config import get_settings
from krabby_fleet_service._mqtt import FleetMqttClient
from krabby_fleet_service._signaling import signaling_in_topic, signaling_out_topic


def main() -> int:
    parser = argparse.ArgumentParser(description="MQTT teleop signaling smoke test")
    parser.add_argument("--thing", required=True, help="IoT thing name (e.g. bench-krabby-ci)")
    parser.add_argument("--wait-secs", type=float, default=8.0, help="Listen for /out replies")
    args = parser.parse_args()

    settings = get_settings()
    in_topic = signaling_in_topic(args.thing)
    out_topic = signaling_out_topic(args.thing)

    received: list[str] = []
    lock = threading.Lock()

    def on_out(topic: str, payload: bytes) -> None:
        text = payload.decode("utf-8", errors="replace")
        with lock:
            received.append(text)
        print(f"RECV out topic={topic} bytes={len(payload)}: {text[:200]}", flush=True)

    mqtt = FleetMqttClient()
    print(
        f"CONNECT region={settings.aws_region} endpoint={settings.iot_ats_endpoint}",
        flush=True,
    )
    mqtt.connect(endpoint=settings.iot_ats_endpoint, region=settings.aws_region)
    mqtt.subscribe(out_topic, on_out)

    ping = json.dumps({"type": "ping", "t": time.time(), "smoke": True})
    print(f"PUBLISH in topic={in_topic} payload={ping}", flush=True)
    mqtt.publish(in_topic, ping)

    deadline = time.monotonic() + args.wait_secs
    while time.monotonic() < deadline:
        time.sleep(0.2)

    mqtt.disconnect()
    with lock:
        n = len(received)
    print(f"DONE out_messages={n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
