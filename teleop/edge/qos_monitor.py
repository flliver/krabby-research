"""Async WebRTC stats poller that feeds :class:`TeleopQosController`."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from teleop.edge.qos import TeleopQosController, diff_outbound_kbps, parse_rtc_stats_report

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0


async def run_qos_stats_loop(
    pc: Any,
    controller: TeleopQosController,
    *,
    stop_when: asyncio.Event | None = None,
    poll_interval_s: float = _POLL_INTERVAL_S,
) -> None:
    """Poll ``pc.getStats()`` until the peer connection closes or ``stop_when`` is set."""
    if not controller.enabled:
        return

    prev_bytes = 0
    prev_mono = time.monotonic()
    while pc.connectionState not in ("closed", "failed"):
        if stop_when is not None and stop_when.is_set():
            break
        try:
            report = await pc.getStats()
        except Exception:
            logger.debug("teleop qos: getStats failed", exc_info=True)
            await asyncio.sleep(poll_interval_s)
            continue

        now_mono = time.monotonic()
        _, loss_fraction = parse_rtc_stats_report(report)

        curr_bytes = 0
        stats_iter = report.values() if isinstance(report, dict) else report
        for stat in stats_iter:
            stat_type = getattr(stat, "type", None) or (
                stat.get("type") if isinstance(stat, dict) else None
            )
            kind = getattr(stat, "kind", None) or (stat.get("kind") if isinstance(stat, dict) else None)
            if stat_type == "outbound-rtp" and kind == "video":
                curr_bytes += int(
                    getattr(stat, "bytesSent", 0)
                    or (stat.get("bytesSent", 0) if isinstance(stat, dict) else 0)
                )

        outbound_kbps = diff_outbound_kbps(prev_bytes, prev_mono, curr_bytes, now_mono)
        prev_bytes = curr_bytes
        prev_mono = now_mono

        controller.observe_sample(
            outbound_kbps=outbound_kbps,
            packet_loss_fraction=loss_fraction,
        )
        await asyncio.sleep(poll_interval_s)
