import logging
import math
import time

import numpy as np
import pytest

pytest.importorskip("av")

from hal.client.data_structures.hardware import HardwareObservations
from hal.server.teleop_portal_signaling import _ControlLatencyReporter, bind_telemetry_slot
from teleop.edge.telemetry import TELEMETRY_MESSAGE_TYPE


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


def _sample_observation() -> HardwareObservations:
    return HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1_500_000_000,
        base_ang_vel_b=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        base_lin_vel_b=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
    )


def test_bind_telemetry_slot_getter_sees_hal_poll_update() -> None:
    """WebRTC getter must read telemetry written by the HAL poll path."""
    record, getter = bind_telemetry_slot()
    record(_sample_observation())
    payload = getter()
    assert payload is not None
    assert payload["type"] == TELEMETRY_MESSAGE_TYPE
    assert payload["timestamp_ns"] == 1_500_000_000
    assert payload["velocity"]["linear_m_s"] == [1.0, 0.0, 0.0]
