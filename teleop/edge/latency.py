"""Glass-to-glass latency helpers and documented measurement targets."""

from __future__ import annotations

import math
from typing import Any

from teleop.edge.qos import G2G_TARGET_MS_FOUR_STREAMS, G2G_TARGET_MS_SINGLE_STREAM


def estimate_g2g_ms(
    *,
    capture_timestamp_ns: int,
    browser_now_ms: float,
    clock_offset_ms: float,
) -> float | None:
    """Estimate capture-to-render latency (ms) using portal offset estimation.

    ``clock_offset_ms`` is robot wall-clock minus browser wall-clock (see portal ping/pong).
    """
    if not math.isfinite(browser_now_ms) or not math.isfinite(clock_offset_ms):
        return None
    capture_ms_robot = capture_timestamp_ns / 1e6
    capture_ms_browser = capture_ms_robot - clock_offset_ms
    g2g = browser_now_ms - capture_ms_browser
    if not math.isfinite(g2g):
        return None
    return max(0.0, g2g)


def g2g_target_ms(stream_count: int) -> int:
    """Latency budget for the given number of simultaneous viewer streams."""
    if stream_count <= 1:
        return G2G_TARGET_MS_SINGLE_STREAM
    return G2G_TARGET_MS_FOUR_STREAMS


def summarize_g2g_samples(samples_ms: list[float]) -> dict[str, Any]:
    """Return p50/p95/max for a list of glass-to-glass samples."""
    if not samples_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(samples_ms)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        rank = (p / 100.0) * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return ordered[lo]
        frac = rank - lo
        return ordered[lo] + ((ordered[hi] - ordered[lo]) * frac)

    return {
        "count": n,
        "p50_ms": round(_pct(50.0), 1),
        "p95_ms": round(_pct(95.0), 1),
        "max_ms": round(max(ordered), 1),
    }
