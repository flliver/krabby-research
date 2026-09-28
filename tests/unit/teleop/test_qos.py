"""QoS degradation policy characterization (fps + stream-count under bandwidth pressure)."""

from __future__ import annotations

import logging

import pytest

from teleop.edge.latency import (
    estimate_g2g_ms,
    g2g_target_ms,
    summarize_g2g_samples,
)
from teleop.edge.qos import (
    G2G_TARGET_MS_FOUR_STREAMS,
    G2G_TARGET_MS_SINGLE_STREAM,
    INACTIVE_STREAM_FPS,
    TeleopDegradationPolicy,
    TeleopQosController,
    diff_outbound_kbps,
)


@pytest.fixture
def policy_4_stream() -> TeleopDegradationPolicy:
    return TeleopDegradationPolicy(stream_count=4, kbps_budget_per_stream=2000.0)


def test_g2g_latency_targets() -> None:
    assert G2G_TARGET_MS_SINGLE_STREAM == 300
    assert G2G_TARGET_MS_FOUR_STREAMS == 500
    assert g2g_target_ms(1) == 300
    assert g2g_target_ms(4) == 500


def test_estimate_g2g_ms_with_offset() -> None:
    # capture at robot t=1000ms, browser now=1300ms, offset=0 => 300ms g2g
    g2g = estimate_g2g_ms(
        capture_timestamp_ns=1_000_000_000,
        browser_now_ms=1300.0,
        clock_offset_ms=0.0,
    )
    assert g2g == pytest.approx(300.0)


def test_summarize_g2g_samples_percentiles() -> None:
    summary = summarize_g2g_samples([100.0, 200.0, 300.0, 400.0, 500.0])
    assert summary["count"] == 5
    assert summary["p50_ms"] == 300.0
    assert summary["p95_ms"] == 480.0
    assert summary["max_ms"] == 500.0


def test_diff_outbound_kbps() -> None:
    kbps = diff_outbound_kbps(0, 0.0, 125_000, 1.0)  # 125kB in 1s => 1000 kbps
    assert kbps == pytest.approx(1000.0)


def test_healthy_single_stream_stays_level_0(policy_4_stream: TeleopDegradationPolicy) -> None:
    single = TeleopDegradationPolicy(stream_count=1, kbps_budget_per_stream=2000.0)
    state = single.observe(outbound_kbps=1900.0, packet_loss_fraction=0.0)
    assert state.level == 0
    assert state.target_fps == 30.0
    assert state.active_stream_count == 1


@pytest.mark.parametrize(
    ("outbound_kbps", "loss", "expected_level", "expected_fps", "expected_active"),
    [
        # 4 streams @ 2000 kbps budget each => 8000 kbps total
        (6700.0, 0.0, 1, 24.0, 4),  # <85% budget -> level 1
        (5500.0, 0.0, 2, 15.0, 4),  # <70% budget -> level 2
        (4300.0, 0.0, 3, 15.0, 3),  # <55% budget -> drop 1 stream
        (3100.0, 0.0, 4, 10.0, 2),  # <40% budget -> drop 2 streams
        (1900.0, 0.0, 5, 5.0, 1),  # <25% budget -> single stream survival
        (7600.0, 0.03, 1, 24.0, 4),  # loss-driven level 1
        (7600.0, 0.06, 2, 15.0, 4),  # loss-driven level 2
        (7600.0, 0.09, 3, 15.0, 3),  # loss-driven level 3
    ],
    ids=[
        "budget_87pct",
        "budget_70pct",
        "budget_55pct_drop1",
        "budget_40pct_drop2",
        "budget_25pct_single",
        "loss_3pct",
        "loss_6pct",
        "loss_9pct",
    ],
)
def test_degradation_ladder_characterization(
    policy_4_stream: TeleopDegradationPolicy,
    outbound_kbps: float,
    loss: float,
    expected_level: int,
    expected_fps: float,
    expected_active: int,
) -> None:
    state = policy_4_stream.observe(
        outbound_kbps=outbound_kbps,
        packet_loss_fraction=loss,
    )
    assert state.level == expected_level
    assert state.target_fps == expected_fps
    assert state.active_stream_count == expected_active
    assert state.stream_count == 4


def test_degradation_hysteresis_on_recovery(policy_4_stream: TeleopDegradationPolicy) -> None:
    policy_4_stream.observe(outbound_kbps=1900.0, packet_loss_fraction=0.0)
    assert policy_4_stream.level == 5

    # One good sample is not enough to recover.
    state = policy_4_stream.observe(outbound_kbps=8000.0, packet_loss_fraction=0.0)
    assert state.level == 5

    state = policy_4_stream.observe(outbound_kbps=8000.0, packet_loss_fraction=0.0)
    assert state.level == 0
    assert state.target_fps == 30.0
    assert state.active_stream_count == 4


def test_qos_controller_track_fps_and_active_flags() -> None:
    controller = TeleopQosController(
        TeleopDegradationPolicy(stream_count=4, kbps_budget_per_stream=2000.0)
    )
    controller.configure_streams(4)
    # configure_streams() arms a post-renegotiation grace window; clear it for unit characterization.
    with controller._lock:
        controller._grace_until_mono_s = 0.0
    controller.observe_sample(outbound_kbps=4300.0, packet_loss_fraction=0.0)

    assert controller.get_target_fps(0) == 15.0
    assert controller.get_target_fps(2) == 15.0
    assert controller.is_track_active(0) is True
    assert controller.is_track_active(2) is True
    assert controller.is_track_active(3) is False
    assert controller.get_target_fps(3) == INACTIVE_STREAM_FPS


def test_qos_controller_disabled_bypasses_degradation() -> None:
    controller = TeleopQosController(
        TeleopDegradationPolicy(stream_count=4, kbps_budget_per_stream=2000.0)
    )
    controller.set_enabled(False)
    controller.configure_streams(4)
    controller.observe_sample(outbound_kbps=1000.0, packet_loss_fraction=0.5)

    assert controller.get_target_fps(0) == 30.0
    assert controller.is_track_active(3) is True
    snap = controller.snapshot()
    assert snap["reason"] == "qos_disabled"


def test_qos_controller_logs_on_level_change(caplog) -> None:
    controller = TeleopQosController(
        TeleopDegradationPolicy(stream_count=1, kbps_budget_per_stream=2000.0)
    )
    controller.configure_streams(1)
    with controller._lock:
        controller._grace_until_mono_s = 0.0
    with caplog.at_level(logging.INFO):
        controller.observe_sample(outbound_kbps=1500.0, packet_loss_fraction=0.0)
    assert "teleop qos: level=1" in caplog.text


def test_snapshot_teleop_level5_recovers_from_low_outbound() -> None:
    """One active stream at ~50 kbps must not stay locked at level 5 forever."""
    policy = TeleopDegradationPolicy(stream_count=6, kbps_budget_per_stream=120.0)
    policy.observe(outbound_kbps=52.0, packet_loss_fraction=0.0)
    assert policy.level == 5

    state = policy.observe(outbound_kbps=52.0, packet_loss_fraction=0.0)
    assert state.level == 5

    state = policy.observe(outbound_kbps=52.0, packet_loss_fraction=0.0)
    assert state.level == 0
    assert state.active_stream_count == 6


def test_kbps_budget_scales_with_active_streams_and_fps() -> None:
    policy = TeleopDegradationPolicy(stream_count=6, kbps_budget_per_stream=120.0)
    assert policy.kbps_budget_for_level(0) == pytest.approx(720.0)
    assert policy.kbps_budget_for_level(5) == pytest.approx(20.0)
