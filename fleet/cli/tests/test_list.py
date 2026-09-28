"""Unit tests for krabby-fleet list formatting."""
from __future__ import annotations

from krabby_fleet_cli.list import _format_last_seen, _telemetry_summary, format_device_line


def test_format_last_seen_ms():
    # 2024-03-09 16:00:00 UTC
    assert _format_last_seen(1710000000000) == "2024-03-09 16:00:00Z"


def test_format_last_seen_seconds():
    assert _format_last_seen(1710000000) == "2024-03-09 16:00:00Z"


def test_format_last_seen_none():
    assert _format_last_seen(None) == "unknown"


def test_telemetry_summary_empty():
    assert _telemetry_summary(None) == "no telemetry"
    assert _telemetry_summary({}) == "no telemetry"


def test_telemetry_summary_fields():
    summary = _telemetry_summary(
        {
            "reported_image": "img:tag",
            "timestamp": 1710000000,
            "health": {"locomotion_container": "running"},
            "red_flags": ["mcu_missing"],
        }
    )
    assert "image=img:tag" in summary
    assert "shadow=1710000000" in summary
    assert "container=running" in summary
    assert "flags=mcu_missing" in summary


def test_format_device_line():
    line = format_device_line(
        {
            "thingName": "bench-krabby-ci",
            "connected": True,
            "connectivityTimestamp": 1710000000000,
            "reported": {"reported_image": "img:tag", "timestamp": 1710000000},
        }
    )
    assert line.startswith("bench-krabby-ci\tonline\t")
    assert "image=img:tag" in line
