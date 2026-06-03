"""MaixSense-A075V as :class:`~hal.server.jetson.rgb_depth_camera.RgbDepthCamera`.

Use ``JETSON_SENSOR_CATALOG`` with ``camera_driver="maixsense_a075v"`` on any rgbd row (primary
or ``hal_open_rgbd`` side/extra). Configure literal ``maixsense_host`` / ``maixsense_port``.

Raw depth is converted to meters with Sipeed's A075V rules (16-bit: 0.25 mm/LSB; 8-bit: nonlinear
``(u8/5.1)²`` mm then ÷1000, see their ``stream.py``/tutorial comments). Frames are emitted at
native decoded sizes (no HAL resize step in this adapter). Requires MaixSense extras (``requests``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
try:
    from requests import exceptions as requests_exceptions
except ImportError:  # pragma: no cover - optional dependency
    requests_exceptions = None

from hal.server.jetson.maixsense_a075v import (
    MaixSenseA075VClient,
    a075v_depth_raw_to_meters,
    a075v_set_cfg_bytes_hal,
)
from hal.server.jetson.rgb_depth_camera import RgbDepthCamera

logger = logging.getLogger(__name__)

class MaixSenseA075VRgbDepthCamera:
    """HTTP MaixSense RGB-D from literal host/port."""

    def __init__(
        self,
        resolution: tuple[int, int],
        depth_mode: str,
        fps: int,
        *,
        maixsense_host: Optional[str] = None,
        maixsense_port: Optional[int] = None,
    ) -> None:
        self._target_w, self._target_h = int(resolution[0]), int(resolution[1])
        self.fps = fps
        _ = depth_mode  # ZED factory parity; MaixSense HTTP path does not use ZED depth modes
        self._initialized = False
        self._client: Any = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_depth_m: Optional[np.ndarray] = None
        self._dependency_error_logged: bool = False
        self._frame_config_logged: bool = False

        host = (maixsense_host or "").strip()
        if not host:
            raise RuntimeError(
                "maixsense_a075v requires non-empty maixsense_host (literal host)"
            )
        if maixsense_port is None:
            raise RuntimeError(
                "maixsense_a075v requires explicit maixsense_port (literal port)"
            )
        port = int(maixsense_port)

        self._client = MaixSenseA075VClient(host=host, port=port)
        if not self._client.post_encode_config(a075v_set_cfg_bytes_hal()):
            raise RuntimeError(
                "MaixSense post_encode_config failed; refusing to continue with unknown device output mode"
            )
        self._initialized = True
        logger.info(
            "MaixSense A075V RGB-D ready (%s:%s → %dx%d)",
            host,
            port,
            self._target_w,
            self._target_h,
        )

    def _maix_depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        return a075v_depth_raw_to_meters(depth)

    def _fetch_and_process(self) -> bool:
        if self._client is None:
            return False
        try:
            frame = self._client.fetch_decoded()
        except ImportError as e:
            # Common configuration issue (opencv/requests missing). Log at error once to avoid loop spam.
            if not self._dependency_error_logged:
                logger.error("MaixSense dependency missing: %s", e)
                self._dependency_error_logged = True
            return False
        except Exception as e:
            logger.warning("MaixSense fetch failed: %s", e)
            return False
        if not self._frame_config_logged:
            deep_mode = frame.config[1]
            rgb_mode = frame.config[6]
            logger.info(
                "MaixSense frame config: deep_mode=%d (%s), rgb_mode=%d (%s), rgb_res=%d",
                deep_mode,
                "16-bit" if deep_mode == 0 else "8-bit",
                rgb_mode,
                "YUV" if rgb_mode == 0 else "JPG" if rgb_mode == 1 else "none",
                frame.config[7],
            )
            self._frame_config_logged = True
            if deep_mode != 0:
                logger.warning(
                    "MaixSense is not in 16-bit depth mode (deep_mode=%d); depth fidelity may be degraded",
                    deep_mode,
                )
        if frame.rgb is None:
            logger.warning("MaixSense frame has no RGB")
            return False
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        if frame.depth is None:
            logger.warning("MaixSense frame has no depth payload (enable depth in device config)")
            return False
        depth_m = self._maix_depth_to_meters(np.asarray(frame.depth))
        self._last_rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        self._last_depth_m = np.ascontiguousarray(depth_m.astype(np.float32))
        return True

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._initialized:
            return None, None
        if not self._fetch_and_process():
            return None, None
        return self._last_rgb, self._last_depth_m

    def is_ready(self) -> bool:
        return self._initialized

    def close(self) -> None:
        self._initialized = False
        self._client = None
        self._last_rgb = None
        self._last_depth_m = None
        logger.info("MaixSense A075V RGB-D camera closed")


def create_maixsense_a075v_rgb_depth_camera(
    depth_mode: str,
    resolution: tuple[int, int],
    fps: int,
    *,
    maixsense_host: Optional[str] = None,
    maixsense_port: Optional[int] = None,
) -> Optional[RgbDepthCamera]:
    """Build MaixSense RGB-D camera; returns ``None`` on configuration or import errors."""
    try:
        cam: RgbDepthCamera = MaixSenseA075VRgbDepthCamera(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            maixsense_host=maixsense_host,
            maixsense_port=maixsense_port,
        )
        return cam
    except RuntimeError as e:
        logger.error("%s", e)
        return None
    except ImportError as e:
        logger.error("MaixSense RGB-D camera import failed: %s", e)
        return None
    except Exception as e:
        if requests_exceptions is not None and isinstance(
            e,
            (
                requests_exceptions.ConnectTimeout,
                requests_exceptions.ReadTimeout,
                requests_exceptions.Timeout,
                requests_exceptions.ConnectionError,
            ),
        ):
            logger.error(
                "MaixSense RGB-D camera connect/read failed (%s). "
                "Check host/port, cable/link, and camera power; skipping this stream.",
                e,
            )
            return None
        logger.error("MaixSense RGB-D camera unexpected error: %s", e, exc_info=True)
        return None
