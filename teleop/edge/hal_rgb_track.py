"""VideoStreamTrack driven by HAL RGB snapshots (uint8 HWC), for WebRTC answers."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import av
import numpy as np
from aiortc import VideoStreamTrack

logger = logging.getLogger(__name__)

# When HAL has no frame yet, emit this size (H, W, 3) uint8 black so the peer connection stays up.
_BLACK_HWC = (240, 320, 3)


class HalRgbSnapshotVideoTrack(VideoStreamTrack):
    """Samples ``frame_getter()`` each frame; waits briefly for the first frame then repeats last."""

    kind = "video"

    def __init__(
        self,
        *,
        frame_getter: Callable[[], np.ndarray | None],
        catalog_id: str | None = None,
    ) -> None:
        super().__init__()
        self._frame_getter = frame_getter
        self._catalog_id = (catalog_id or "").strip() or None
        self._last: np.ndarray | None = None
        self._last_no_frame_warn_mono: float = 0.0

    def _requested_sensor_label(self) -> str:
        return self._catalog_id if self._catalog_id else "(unset catalog_id)"

    def _maybe_warn_no_frames(self) -> None:
        now = time.monotonic()
        if now - self._last_no_frame_warn_mono >= 30.0:
            logger.warning(
                "teleop: no RGB from HAL for video track catalog_id=%s yet; sending black frames until data is available",
                self._requested_sensor_label(),
            )
            self._last_no_frame_warn_mono = now

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        arr = self._frame_getter()
        if arr is not None and arr.size > 0:
            self._last = np.ascontiguousarray(arr, dtype=np.uint8)
        if self._last is None:
            for _ in range(300):
                await asyncio.sleep(0.033)
                arr = self._frame_getter()
                if arr is not None and arr.size > 0:
                    self._last = np.ascontiguousarray(arr, dtype=np.uint8)
                    break
        if self._last is None:
            self._maybe_warn_no_frames()
            black = np.zeros(_BLACK_HWC, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(black, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame
        frame = av.VideoFrame.from_ndarray(self._last, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
