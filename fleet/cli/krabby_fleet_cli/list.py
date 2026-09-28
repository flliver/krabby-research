"""krabby-fleet list: enrolled robots with connectivity + telemetry summary."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from krabby_fleet_cli._api import list_devices
from krabby_fleet_cli._auth import get_access_token
from krabby_fleet_cli._config import load_config


def _format_last_seen(connectivity_timestamp: int | None) -> str:
    if connectivity_timestamp is None:
        return "unknown"
    # Fleet Indexing connectivity.timestamp is epoch milliseconds.
    seconds = connectivity_timestamp / 1000.0 if connectivity_timestamp > 10_000_000_000 else float(connectivity_timestamp)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _telemetry_summary(reported: dict[str, Any] | None) -> str:
    if not reported:
        return "no telemetry"
    parts: list[str] = []
    image = reported.get("reported_image")
    if image:
        parts.append(f"image={image}")
    ts = reported.get("timestamp")
    if ts is not None:
        parts.append(f"shadow={ts}")
    health = reported.get("health")
    if isinstance(health, dict):
        container = health.get("locomotion_container")
        if container:
            parts.append(f"container={container}")
    red_flags = reported.get("red_flags")
    if isinstance(red_flags, list) and red_flags:
        parts.append(f"flags={','.join(str(f) for f in red_flags)}")
    return " ".join(parts) if parts else "telemetry present"


def format_device_line(device: dict[str, Any]) -> str:
    status = "online" if device.get("connected") else "offline"
    last_seen = _format_last_seen(device.get("connectivityTimestamp"))
    summary = _telemetry_summary(device.get("reported") if isinstance(device.get("reported"), dict) else None)
    return f"{device.get('thingName', '?')}\t{status}\tlast-seen={last_seen}\t{summary}"


def cmd_list() -> None:
    config = load_config()
    access_token = get_access_token(config)
    devices = list_devices(config, access_token)

    if not devices:
        print("No devices found.")
        return

    for device in devices:
        print(format_device_line(device))
