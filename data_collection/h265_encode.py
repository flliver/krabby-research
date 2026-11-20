"""Long-lived appsrc→H.265→appsink encoders for the data collector thread."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

PixelFormat = Literal["RGB", "GRAY8"]


def _prefer_nvenc() -> bool:
    try:
        from hal.server.gstreamer_runtime import ensure_gst_initialized, GstRuntimeUnavailable

        ensure_gst_initialized()
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        return Gst.ElementFactory.find("nvv4l2h265enc") is not None
    except Exception:
        return False


def build_h265_appsrc_appsink_pipeline(
    width: int,
    height: int,
    fps: int,
    *,
    format_caps: PixelFormat,
    use_nvenc: bool | None = None,
) -> str:
    """Pipeline string: appsrc name=src → H.265 → appsink name=sink.

    Keyframe interval is once per second (``key-int-max`` / ``iframeinterval`` = fps).

    NVENC path forces system-memory NV12 before ``nvvidconv``. Feeding GRAY8/RGB
    straight into ``nvvidconv`` often fails with ``Transform not supported`` on Jetson.
    """
    fps_i = max(1, int(fps))
    caps = (
        f"appsrc name=src is-live=true format=time do-timestamp=false ! "
        f"video/x-raw,format={format_caps},width={int(width)},height={int(height)},"
        f"framerate={fps_i}/1"
    )
    sink = (
        "appsink name=sink emit-signals=false sync=false max-buffers=4 drop=false "
        "wait-on-eos=true"
    )
    nv = _prefer_nvenc() if use_nvenc is None else bool(use_nvenc)
    if nv:
        # iframeinterval = frames between IDR; set to fps for ≥1 keyframe / second.
        enc = (
            f"videoconvert ! video/x-raw,format=NV12 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h265enc iframeinterval={fps_i} insert-sps-pps=1 ! h265parse config-interval=-1"
        )
    else:
        enc = (
            f"videoconvert ! x265enc tune=zerolatency speed-preset=ultrafast "
            f"key-int-max={fps_i} ! h265parse config-interval=-1"
        )
    return f"{caps} ! {enc} ! {sink}"


def _numpy_to_buffer(frame: np.ndarray, pts: int, duration: int, gst_mod: Any) -> Any:
    Gst = gst_mod
    data = np.ascontiguousarray(frame).tobytes()
    try:
        buf = Gst.Buffer.new_wrapped(data)
    except (TypeError, AttributeError):
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
    buf.pts = pts
    buf.duration = duration
    buf.dts = pts
    return buf


@dataclass
class _PipelineSlot:
    key: tuple[str, int, int, str, int]
    pipeline: Any
    appsrc: Any
    appsink: Any
    pts: int
    duration: int
    use_nvenc: bool
    frame_count: int = 0


class H265FrameEncoder:
    """Encode RGB uint8 or GRAY8 frames to HEVC annex-B bytes (collector thread)."""

    def __init__(self, *, use_nvenc: bool | None = None) -> None:
        # None = auto (prefer NVENC when the element exists); True/False force.
        self._use_nvenc_request = use_nvenc
        # Sticky disable after NVENC open/encode failure so we permanently use x265.
        self._nvenc_disabled = False
        # One pipeline per stream id so same-resolution ToF cameras do not share state.
        self._slots: dict[tuple[str, int, int, str, int], _PipelineSlot] = {}
        self._lock = threading.Lock()
        self._gst: Any = None
        self._gst_app: Any = None

    def close(self) -> None:
        with self._lock:
            for key in list(self._slots):
                self._close_slot_unlocked(key)

    def _ensure_gst(self) -> None:
        if self._gst is not None:
            return
        from hal.server.gstreamer_runtime import ensure_gst_initialized

        ensure_gst_initialized()
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        self._gst = Gst
        self._gst_app = GstApp

    def _effective_nvenc(self) -> bool:
        if self._use_nvenc_request is False or self._nvenc_disabled:
            return False
        if self._use_nvenc_request is True:
            return True
        return _prefer_nvenc()

    def _close_slot_unlocked(self, key: tuple[str, int, int, str, int]) -> None:
        slot = self._slots.pop(key, None)
        if slot is None:
            return
        try:
            slot.pipeline.set_state(self._gst.State.NULL)
        except Exception:
            logger.exception("Error nulling H.265 pipeline %s", slot.key)

    def _open_slot_unlocked(
        self,
        stream_id: str,
        width: int,
        height: int,
        format_caps: PixelFormat,
        fps: int,
        *,
        use_nvenc: bool,
    ) -> _PipelineSlot:
        self._ensure_gst()
        key = (str(stream_id), int(width), int(height), format_caps, int(fps))
        Gst = self._gst
        GstApp = self._gst_app
        desc = build_h265_appsrc_appsink_pipeline(
            width, height, fps, format_caps=format_caps, use_nvenc=use_nvenc
        )
        pipeline = Gst.parse_launch(desc)
        if pipeline is None:
            raise RuntimeError(f"Gst.parse_launch failed for H.265 pipeline: {desc}")
        appsrc_el = pipeline.get_by_name("src")
        appsink_el = pipeline.get_by_name("sink")
        if appsrc_el is None or appsink_el is None:
            raise RuntimeError("H.265 pipeline missing appsrc/appsink names src/sink")
        if not isinstance(appsrc_el, GstApp.AppSrc):
            raise RuntimeError(f"src is not AppSrc: {type(appsrc_el)}")
        if not isinstance(appsink_el, GstApp.AppSink):
            raise RuntimeError(f"sink is not AppSink: {type(appsink_el)}")

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            raise RuntimeError("H.265 pipeline set_state(PLAYING) failed")
        state_ret, _state, _pending = pipeline.get_state(5 * Gst.SECOND)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            raise RuntimeError("H.265 pipeline failed to reach PLAYING")

        duration = Gst.SECOND // max(1, int(fps))
        slot = _PipelineSlot(
            key=key,
            pipeline=pipeline,
            appsrc=appsrc_el,
            appsink=appsink_el,
            pts=0,
            duration=duration,
            use_nvenc=use_nvenc,
        )
        self._slots[key] = slot
        logger.info(
            "Opened H.265 encoder pipeline %s nvenc=%s",
            key,
            use_nvenc,
        )
        return slot

    def _get_slot(
        self,
        stream_id: str,
        width: int,
        height: int,
        format_caps: PixelFormat,
        fps: int,
    ) -> _PipelineSlot:
        key = (str(stream_id), int(width), int(height), format_caps, int(fps))
        slot = self._slots.get(key)
        if slot is not None:
            return slot

        want_nvenc = self._effective_nvenc()
        try:
            return self._open_slot_unlocked(
                stream_id, width, height, format_caps, fps, use_nvenc=want_nvenc
            )
        except Exception as exc:
            if not want_nvenc:
                raise
            logger.warning(
                "NVENC H.265 pipeline open failed (%s); falling back to x265enc: %s",
                key,
                exc,
            )
            self._nvenc_disabled = True
            return self._open_slot_unlocked(
                stream_id, width, height, format_caps, fps, use_nvenc=False
            )

    def _pull_encoded(
        self,
        slot: _PipelineSlot,
        frame: np.ndarray,
        *,
        timeout_s: float,
    ) -> bytes:
        Gst = self._gst
        buf = _numpy_to_buffer(frame, slot.pts, slot.duration, Gst)
        flow = slot.appsrc.push_buffer(buf)
        if flow not in (Gst.FlowReturn.OK, Gst.FlowReturn.FLUSHING):
            raise RuntimeError(f"appsrc push_buffer returned {flow!r}")
        slot.pts += slot.duration
        slot.frame_count += 1

        deadline_ns = int(max(timeout_s, 0.1) * Gst.SECOND)
        sample = slot.appsink.try_pull_sample(deadline_ns)
        if sample is None:
            raise TimeoutError(
                f"appsink timed out waiting for H.265 sample ({slot.key[0]} "
                f"{slot.key[3]} {slot.key[1]}x{slot.key[2]})"
            )
        out_buf = sample.get_buffer()
        if out_buf is None:
            raise RuntimeError("appsink sample has no buffer")
        ok, mapinfo = out_buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map H.265 buffer")
        try:
            return bytes(mapinfo.data)
        finally:
            out_buf.unmap(mapinfo)

    def encode_frame(
        self,
        frame: np.ndarray,
        *,
        stream_id: str,
        format_caps: PixelFormat,
        fps: int = 10,
        timeout_s: float = 2.0,
    ) -> bytes:
        """Push one frame and pull one encoded buffer (may include codec config)."""
        if format_caps == "RGB":
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"RGB frame must be uint8 (H,W,3); got {frame.shape}/{frame.dtype}")
            height, width = int(frame.shape[0]), int(frame.shape[1])
        elif format_caps == "GRAY8":
            if frame.dtype != np.uint8 or frame.ndim != 2:
                raise ValueError(f"GRAY8 frame must be uint8 (H,W); got {frame.shape}/{frame.dtype}")
            height, width = int(frame.shape[0]), int(frame.shape[1])
        else:
            raise ValueError(format_caps)

        with self._lock:
            slot = self._get_slot(stream_id, width, height, format_caps, fps)
            try:
                return self._pull_encoded(slot, frame, timeout_s=timeout_s)
            except TimeoutError:
                if not slot.use_nvenc:
                    raise
                logger.warning(
                    "NVENC H.265 encode timed out for %s; falling back to x265enc",
                    slot.key,
                )
                self._nvenc_disabled = True
                self._close_slot_unlocked(slot.key)
                slot = self._open_slot_unlocked(
                    stream_id, width, height, format_caps, fps, use_nvenc=False
                )
                return self._pull_encoded(slot, frame, timeout_s=timeout_s)

    def encode_rgb(self, rgb: np.ndarray, *, stream_id: str, fps: int = 10) -> bytes:
        return self.encode_frame(rgb, stream_id=stream_id, format_caps="RGB", fps=fps)

    def encode_gray8(self, gray: np.ndarray, *, stream_id: str, fps: int = 10) -> bytes:
        return self.encode_frame(gray, stream_id=stream_id, format_caps="GRAY8", fps=fps)


_SHARED: Optional[H265FrameEncoder] = None
_SHARED_LOCK = threading.Lock()


def shared_h265_encoder() -> H265FrameEncoder:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = H265FrameEncoder()
        return _SHARED
