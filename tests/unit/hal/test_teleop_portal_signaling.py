import logging
import math
import time

from hal.server.teleop_portal_signaling import _ControlLatencyReporter


def test_control_latency_reporter_logs_window_percentiles(caplog) -> None:
    reporter = _ControlLatencyReporter()
    reporter._samples_ms = [10.0, 20.0, 30.0, 40.0, 50.0]
    reporter._total_samples = 5

    with caplog.at_level(logging.INFO):
        reporter._report(now_mono_s=123.0)

    assert "teleop control latency: samples=5 total=5" in caplog.text
    assert "p50=30.0ms" in caplog.text
    assert "p95=48.0ms" in caplog.text
    assert "max=50.0ms" in caplog.text
    assert "latest=50.0ms" in caplog.text
    assert reporter._samples_ms == []
    assert reporter._last_report_mono_s == 123.0


def test_control_latency_reporter_ignores_missing_or_invalid_timestamps() -> None:
    reporter = _ControlLatencyReporter()

    reporter.observe_payload({})
    reporter.observe_payload({"sent_browser_ms": "100"})
    reporter.observe_payload({"sent_browser_ms": True})
    reporter.observe_payload({"sent_browser_ms": math.nan})

    assert reporter._samples_ms == []
    assert reporter._total_samples == 0


def test_control_latency_reporter_records_valid_timestamp() -> None:
    reporter = _ControlLatencyReporter()
    reporter._last_report_mono_s = time.monotonic() + 9999.0

    reporter.observe_payload({"sent_browser_ms": (time.time() * 1000.0) - 10.0})

    assert reporter._total_samples == 1
    assert len(reporter._samples_ms) == 1
    assert 0.0 <= reporter._samples_ms[0] < 1000.0
