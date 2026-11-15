"""SEN0581 (Sipeed Meta Sense Lite) USB-serial depth camera.

The module enumerates as FT2232 dual UART (``0403:6010``). Use the ``if00``
serial port only. AT commands configure ISP / binning / units / display mux;
depth frames are ``00 FF`` packets with a 16-byte meta header and a 100×100
uint8 depth payload (``UNIT=1`` → millimeters). Invalid / out-of-range samples
use ``0xFF`` and are mapped to ``0.0`` m.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hal.server.jetson.rgb_depth_camera import RgbDepthCamera

logger = logging.getLogger(__name__)

SEN0581_WIDTH = 100
SEN0581_HEIGHT = 100
SEN0581_PIXELS = SEN0581_WIDTH * SEN0581_HEIGHT
SEN0581_FRAME_HEADER = b"\x00\xff"
SEN0581_META_BYTES = 16
# rem LE after header: meta + payload (no separate trailer observed on this firmware).
SEN0581_REM_LEN = SEN0581_META_BYTES + SEN0581_PIXELS  # 10016
SEN0581_FRAME_BYTES = 2 + 2 + SEN0581_REM_LEN  # 10020
SEN0581_INVALID_DEPTH_U8 = 255
SEN0581_DEFAULT_BAUD = 115200
SEN0581_READY_TIMEOUT_S = 3.0
SEN0581_REOPEN_BACKOFF_S = 0.5
SEN0581_READ_CHUNK = 16384

_AT_SETUP = (
    b"AT+ISP=1\r",
    b"AT+BINN=1\r",
    b"AT+UNIT=1\r",
    b"AT+FPS=19\r",
    b"AT+DISP=3\r",  # LCD + USB stream
)


def sen0581_depth_u8_to_meters(depth_u8: np.ndarray) -> np.ndarray:
    """Convert UNIT=1 uint8 depth (mm) to float32 meters; ``0xFF`` → ``0.0``."""
    raw = np.asarray(depth_u8, dtype=np.uint8)
    out = raw.astype(np.float32) * 0.001
    out[raw == SEN0581_INVALID_DEPTH_U8] = 0.0
    return out


def parse_sen0581_frame(frame: bytes) -> np.ndarray:
    """Parse one complete SEN0581 frame into ``(100, 100)`` float32 meters.

    Raises:
        ValueError: if the buffer is not a well-formed frame.
    """
    if len(frame) != SEN0581_FRAME_BYTES:
        raise ValueError(
            f"SEN0581 frame length {len(frame)} != {SEN0581_FRAME_BYTES}"
        )
    if frame[:2] != SEN0581_FRAME_HEADER:
        raise ValueError(f"SEN0581 bad header {frame[:2]!r}")
    rem = struct.unpack_from("<H", frame, 2)[0]
    if rem != SEN0581_REM_LEN:
        raise ValueError(f"SEN0581 rem length {rem} != {SEN0581_REM_LEN}")
    payload = frame[4 + SEN0581_META_BYTES : 4 + SEN0581_META_BYTES + SEN0581_PIXELS]
    depth_u8 = np.frombuffer(payload, dtype=np.uint8).reshape(
        SEN0581_HEIGHT, SEN0581_WIDTH
    )
    return sen0581_depth_u8_to_meters(depth_u8)


class Sen0581FrameParser:
    """Incremental stream parser that yields complete depth frames in meters."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> list[np.ndarray]:
        if data:
            self._buf.extend(data)
        out: list[np.ndarray] = []
        while True:
            if len(self._buf) < 4:
                break
            # Resync to header.
            if self._buf[0:2] != SEN0581_FRAME_HEADER:
                idx = self._buf.find(SEN0581_FRAME_HEADER)
                if idx < 0:
                    # Keep last byte in case next chunk starts mid-header.
                    self._buf[:] = self._buf[-1:]
                    break
                del self._buf[:idx]
                if len(self._buf) < 4:
                    break
            rem = struct.unpack_from("<H", self._buf, 2)[0]
            total = 4 + rem
            if rem != SEN0581_REM_LEN:
                # Skip this false header and resync.
                del self._buf[0]
                continue
            if len(self._buf) < total:
                break
            frame = bytes(self._buf[:total])
            del self._buf[:total]
            try:
                out.append(parse_sen0581_frame(frame))
            except ValueError:
                continue
        return out


class Sen0581TofCamera:
    """Background-threaded SEN0581 depth capture as :class:`RgbDepthCamera`.

    ``get_camera_frames`` returns ``(None, depth)`` with depth ``(100, 100)`` float32
    meters from the latest parsed frame. Serial faults close and reopen with backoff
    rather than raising into the HAL grab loop.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        depth_mode: str,
        fps: int,
        *,
        sen0581_device: Optional[str] = None,
        baudrate: int = SEN0581_DEFAULT_BAUD,
        ready_timeout_s: float = SEN0581_READY_TIMEOUT_S,
    ) -> None:
        _ = depth_mode
        self.fps = int(fps)
        device = (sen0581_device or "").strip()
        if not device:
            raise RuntimeError("sen0581 requires non-empty sen0581_device (serial path)")
        self._device = device
        self._baudrate = int(baudrate)
        self._ready_timeout_s = float(ready_timeout_s)
        self._target_w, self._target_h = int(resolution[0]), int(resolution[1])
        if (self._target_w, self._target_h) != (SEN0581_WIDTH, SEN0581_HEIGHT):
            logger.warning(
                "SEN0581 catalog resolution %dx%d != native %dx%d; frames stay native",
                self._target_w,
                self._target_h,
                SEN0581_WIDTH,
                SEN0581_HEIGHT,
            )

        self._lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_mono_s = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser = None
        self._parser = Sen0581FrameParser()
        self._initialized = False

        self._open_and_configure()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"sen0581:{Path(device).name}",
            daemon=True,
        )
        self._thread.start()
        # Match other drivers: ready once the device is configured and the reader
        # is running. First USB depth frames can lag on a busy hub; HAL zero-fills
        # until get_camera_frames returns a depth map.
        self._initialized = True
        deadline = time.monotonic() + min(1.0, self._ready_timeout_s)
        while time.monotonic() < deadline:
            if self.has_recent_frame():
                break
            time.sleep(0.02)
        logger.info(
            "SEN0581 ToF ready (%s)%s",
            device,
            "" if self.has_recent_frame() else " (waiting for first depth frame)",
        )

    def _import_serial(self):
        try:
            import serial  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pyserial is required for sen0581") from e
        return serial

    def _open_and_configure(self) -> None:
        serial = self._import_serial()
        self._close_serial()
        self._parser.reset()
        ser = serial.Serial(self._device, baudrate=self._baudrate, timeout=0.05)
        time.sleep(0.05)
        ser.reset_input_buffer()
        for cmd in _AT_SETUP:
            ser.write(cmd)
            time.sleep(0.12)
            ser.read(65536)
        time.sleep(0.2)
        ser.reset_input_buffer()
        self._ser = ser

    def _close_serial(self) -> None:
        ser = self._ser
        self._ser = None
        if ser is None:
            return
        try:
            # Prefer LCD-only when stopping cleanly so the panel stays useful.
            ser.write(b"AT+DISP=1\r")
            time.sleep(0.05)
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

    def _read_loop(self) -> None:
        backoff = SEN0581_REOPEN_BACKOFF_S
        while not self._stop.is_set():
            ser = self._ser
            if ser is None:
                try:
                    self._open_and_configure()
                    backoff = SEN0581_REOPEN_BACKOFF_S
                except Exception as e:
                    logger.warning("SEN0581 reopen failed (%s): %s", self._device, e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, 5.0)
                continue
            try:
                chunk = ser.read(SEN0581_READ_CHUNK)
                if not chunk:
                    continue
                frames = self._parser.feed(chunk)
                if not frames:
                    continue
                depth = frames[-1]
                with self._lock:
                    self._latest_depth = depth
                    self._latest_mono_s = time.monotonic()
            except Exception as e:
                logger.warning("SEN0581 read error (%s): %s; reopening", self._device, e)
                self._close_serial()
                self._parser.reset()
                self._stop.wait(SEN0581_REOPEN_BACKOFF_S)

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._initialized:
            return None, None
        with self._lock:
            if self._latest_depth is None:
                return None, None
            depth = np.ascontiguousarray(self._latest_depth, dtype=np.float32)
        return None, depth

    def has_recent_frame(self) -> bool:
        with self._lock:
            if self._latest_depth is None:
                return False
            age = time.monotonic() - self._latest_mono_s
        return age <= self._ready_timeout_s

    def is_ready(self) -> bool:
        return self._initialized

    def close(self) -> None:
        self._initialized = False
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._close_serial()
        with self._lock:
            self._latest_depth = None
            self._latest_mono_s = 0.0
        logger.info("SEN0581 ToF closed (%s)", self._device)


def create_sen0581_tof_camera(
    depth_mode: str,
    resolution: tuple[int, int],
    fps: int,
    *,
    sen0581_device: Optional[str] = None,
) -> Optional[RgbDepthCamera]:
    """Build SEN0581 depth camera; returns ``None`` on configuration / open errors."""
    try:
        cam: RgbDepthCamera = Sen0581TofCamera(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            sen0581_device=sen0581_device,
        )
        return cam
    except RuntimeError as e:
        logger.error("%s", e)
        return None
    except ImportError as e:
        logger.error("SEN0581 import failed: %s", e)
        return None
    except Exception as e:
        logger.error(
            "SEN0581 open failed (%s): %s",
            sen0581_device,
            e,
        )
        return None
