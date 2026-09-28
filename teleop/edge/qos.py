"""Teleop QoS: bandwidth-pressure degradation (fps + stream-count adaptation).

Degradation is applied on the robot sender. Under pressure the controller lowers
per-stream frame rate first, then drops lowest-priority streams (highest track index)
while keeping at least one active stream.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Streams beyond ``active_stream_count`` are kept alive at this rate (black/held frame).
INACTIVE_STREAM_FPS: float = 1.0

# Glass-to-glass latency targets (documented in docs/TELEOP.md).
G2G_TARGET_MS_SINGLE_STREAM: int = 300
G2G_TARGET_MS_FOUR_STREAMS: int = 500


@dataclass(frozen=True)
class QosDegradationState:
    """Snapshot of the current degradation decision."""

    level: int
    target_fps: float
    active_stream_count: int
    stream_count: int
    outbound_kbps: float
    packet_loss_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "target_fps": round(self.target_fps, 2),
            "active_stream_count": self.active_stream_count,
            "stream_count": self.stream_count,
            "outbound_kbps": round(self.outbound_kbps, 1),
            "packet_loss_fraction": round(self.packet_loss_fraction, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _DegradationStep:
    level: int
    target_fps: float
    active_streams_delta: int  # subtract from stream_count (clamped to >= 1)
    min_budget_fraction: float
    min_loss_fraction: float


# Characterization ladder (see tests/unit/teleop/test_qos.py).
_DEGRADATION_LADDER: tuple[_DegradationStep, ...] = (
    _DegradationStep(0, 30.0, 0, 1.0, 0.0),
    _DegradationStep(1, 24.0, 0, 0.85, 0.02),
    _DegradationStep(2, 15.0, 0, 0.70, 0.05),
    _DegradationStep(3, 15.0, 1, 0.55, 0.08),
    _DegradationStep(4, 10.0, 2, 0.40, 0.12),
    _DegradationStep(5, 5.0, 0, 0.25, 0.20),  # active_streams_delta handled specially
)


class TeleopDegradationPolicy:
    """Pure policy: map outbound bitrate + loss to fps / active stream count."""

    def __init__(
        self,
        *,
        stream_count: int = 1,
        nominal_fps: float = 30.0,
        kbps_budget_per_stream: float = 120.0,
        recovery_hold_samples: int = 2,
    ) -> None:
        self._stream_count = max(1, int(stream_count))
        self._nominal_fps = max(1.0, float(nominal_fps))
        self._kbps_budget_per_stream = max(100.0, float(kbps_budget_per_stream))
        self._recovery_hold_samples = max(1, int(recovery_hold_samples))
        self._level = 0
        self._recovery_streak = 0
        self._last_outbound_kbps = 0.0
        self._last_loss = 0.0

    @property
    def stream_count(self) -> int:
        return self._stream_count

    @property
    def level(self) -> int:
        return self._level

    def set_stream_count(self, stream_count: int) -> None:
        self._stream_count = max(1, int(stream_count))
        self._level = 0
        self._recovery_streak = 0

    def total_kbps_budget(self) -> float:
        """Nominal budget if every requested stream were active at ``nominal_fps``."""
        return self._stream_count * self._kbps_budget_per_stream

    @staticmethod
    def active_stream_count_for_level(level: int, stream_count: int) -> int:
        step = _DEGRADATION_LADDER[level]
        if level == 5:
            return 1
        return max(1, stream_count - step.active_streams_delta)

    def kbps_budget_for_level(self, level: int) -> float:
        """Expected outbound budget at ``level`` (active streams × per-stream × fps scale)."""
        active = self.active_stream_count_for_level(level, self._stream_count)
        step = _DEGRADATION_LADDER[level]
        fps_scale = step.target_fps / self._nominal_fps
        return active * self._kbps_budget_per_stream * fps_scale

    def observe(
        self,
        *,
        outbound_kbps: float,
        packet_loss_fraction: float,
        max_level: int | None = None,
    ) -> QosDegradationState:
        outbound_kbps = max(0.0, float(outbound_kbps))
        packet_loss_fraction = min(1.0, max(0.0, float(packet_loss_fraction)))
        self._last_outbound_kbps = outbound_kbps
        self._last_loss = packet_loss_fraction

        # Compare measured outbound to the budget implied by the *current* degradation
        # level (active streams + target fps). Using the full requested stream count while
        # already degraded caused a lock at level 5: one surviving stream cannot fill a
        # six-stream nominal budget, so recovery never triggered.
        budget = self.kbps_budget_for_level(self._level)
        budget_fraction = (outbound_kbps / budget) if budget > 0.0 else 1.0

        required_level = self._required_level(budget_fraction, packet_loss_fraction)
        if required_level > self._level:
            self._level = required_level
            self._recovery_streak = 0
        elif required_level < self._level:
            self._recovery_streak += 1
            if self._recovery_streak >= self._recovery_hold_samples:
                self._level = required_level
                self._recovery_streak = 0
        else:
            self._recovery_streak = 0

        if max_level is not None and self._level > max_level:
            self._level = max_level
            self._recovery_streak = 0

        step = _DEGRADATION_LADDER[self._level]
        active = self.active_stream_count_for_level(self._level, self._stream_count)
        reason = self._build_reason(budget_fraction, packet_loss_fraction, step)
        return QosDegradationState(
            level=self._level,
            target_fps=step.target_fps,
            active_stream_count=active,
            stream_count=self._stream_count,
            outbound_kbps=outbound_kbps,
            packet_loss_fraction=packet_loss_fraction,
            reason=reason,
        )

    def _required_level(self, budget_fraction: float, loss: float) -> int:
        level = 0
        for step in _DEGRADATION_LADDER:
            over_budget = budget_fraction < step.min_budget_fraction
            over_loss = loss >= step.min_loss_fraction and step.min_loss_fraction > 0.0
            if over_budget or over_loss:
                level = step.level
        return level

    @staticmethod
    def _build_reason(budget_fraction: float, loss: float, step: _DegradationStep) -> str:
        parts: list[str] = []
        if budget_fraction < step.min_budget_fraction:
            parts.append(f"budget {budget_fraction:.0%}")
        if loss >= step.min_loss_fraction and step.min_loss_fraction > 0.0:
            parts.append(f"loss {loss:.1%}")
        if not parts:
            return "healthy"
        return "; ".join(parts)


class TeleopQosController:
    """Thread-safe holder for the active degradation state (HAL poll + aiortc recv)."""

    _REPORT_INTERVAL_S = 5.0
    _GRACE_AFTER_RECONFIGURE_S = 10.0

    def __init__(self, policy: TeleopDegradationPolicy | None = None) -> None:
        self._policy = policy or TeleopDegradationPolicy()
        self._lock = threading.Lock()
        self._grace_until_mono_s = 0.0
        self._state = QosDegradationState(
            level=0,
            target_fps=30.0,
            active_stream_count=self._policy.stream_count,
            stream_count=self._policy.stream_count,
            outbound_kbps=0.0,
            packet_loss_fraction=0.0,
            reason="init",
        )
        self._last_report_mono_s = time.monotonic()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    def configure_streams(self, stream_count: int) -> None:
        with self._lock:
            self._policy.set_stream_count(stream_count)
            self._grace_until_mono_s = time.monotonic() + self._GRACE_AFTER_RECONFIGURE_S
            self._state = QosDegradationState(
                level=0,
                target_fps=30.0,
                active_stream_count=max(1, stream_count),
                stream_count=max(1, stream_count),
                outbound_kbps=0.0,
                packet_loss_fraction=0.0,
                reason="reconfigured",
            )

    def observe_sample(
        self,
        *,
        outbound_kbps: float,
        packet_loss_fraction: float,
    ) -> QosDegradationState:
        with self._lock:
            if not self._enabled:
                sc = self._policy.stream_count
                self._state = QosDegradationState(
                    level=0,
                    target_fps=30.0,
                    active_stream_count=sc,
                    stream_count=sc,
                    outbound_kbps=outbound_kbps,
                    packet_loss_fraction=packet_loss_fraction,
                    reason="qos_disabled",
                )
                return self._state
            prev_level = self._state.level
            max_level = 0 if time.monotonic() < self._grace_until_mono_s else None
            self._state = self._policy.observe(
                outbound_kbps=outbound_kbps,
                packet_loss_fraction=packet_loss_fraction,
                max_level=max_level,
            )
            now = time.monotonic()
            if (
                self._state.level != prev_level
                or now - self._last_report_mono_s >= self._REPORT_INTERVAL_S
            ):
                self._log_state()
                self._last_report_mono_s = now
            return self._state

    def get_target_fps(self, track_index: int) -> float:
        with self._lock:
            if not self._enabled:
                return 30.0
            if track_index < 0:
                return INACTIVE_STREAM_FPS
            if track_index >= self._state.active_stream_count:
                return INACTIVE_STREAM_FPS
            return self._state.target_fps

    def is_track_active(self, track_index: int) -> bool:
        with self._lock:
            if not self._enabled:
                return True
            return 0 <= track_index < self._state.active_stream_count

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def _log_state(self) -> None:
        s = self._state
        logger.info(
            "teleop qos: level=%d fps=%.1f active_streams=%d/%d outbound=%.0fkbps loss=%.2f%% (%s)",
            s.level,
            s.target_fps,
            s.active_stream_count,
            s.stream_count,
            s.outbound_kbps,
            s.packet_loss_fraction * 100.0,
            s.reason,
        )


def parse_rtc_stats_report(report: Any) -> tuple[float, float]:
    """Extract (outbound_kbps, packet_loss_fraction) from an aiortc ``getStats()`` report."""
    if report is None:
        return 0.0, 0.0

    stats_iter: Any
    if isinstance(report, dict):
        stats_iter = report.values()
    else:
        stats_iter = report

    video_bytes = 0
    packets_lost = 0
    packets_received = 0
    for stat in stats_iter:
        stat_type = getattr(stat, "type", None) or (stat.get("type") if isinstance(stat, dict) else None)
        kind = getattr(stat, "kind", None) or (stat.get("kind") if isinstance(stat, dict) else None)
        if stat_type == "outbound-rtp" and kind == "video":
            video_bytes += int(
                getattr(stat, "bytesSent", 0)
                or (stat.get("bytesSent", 0) if isinstance(stat, dict) else 0)
            )
        if stat_type in ("remote-inbound-rtp", "inbound-rtp") and kind == "video":
            packets_lost += int(
                getattr(stat, "packetsLost", 0)
                or (stat.get("packetsLost", 0) if isinstance(stat, dict) else 0)
            )
            recv = getattr(stat, "packetsReceived", None)
            if recv is None and isinstance(stat, dict):
                recv = stat.get("packetsReceived", 0)
            packets_received += int(recv or 0)

    # ``getStats`` is caller-differenced; when only cumulative bytes are present treat as 0 kbps.
    outbound_kbps = 0.0
    denom = packets_received + packets_lost
    loss_fraction = (packets_lost / denom) if denom > 0 else 0.0
    if not math.isfinite(loss_fraction):
        loss_fraction = 0.0
    return outbound_kbps, loss_fraction


def diff_outbound_kbps(
    prev_bytes: int,
    prev_mono_s: float,
    curr_bytes: int,
    curr_mono_s: float,
) -> float:
    """Bitrate from cumulative ``bytesSent`` deltas."""
    dt = curr_mono_s - prev_mono_s
    if dt <= 0.0:
        return 0.0
    delta = max(0, curr_bytes - prev_bytes)
    return (delta * 8.0) / dt / 1000.0
