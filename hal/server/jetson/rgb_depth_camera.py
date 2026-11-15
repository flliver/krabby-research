"""Protocol for RGB-D cameras (RGB + depth in meters).

Policy scan vectors are derived from depth in :class:`~hal.server.jetson.hal_server.JetsonHalServer`
via :mod:`hal.server.jetson.depth_scan_features`. Implementations:
:class:`~hal.server.jetson.zed_camera.ZedCamera`,
:class:`~hal.server.jetson.maixsense_rgb_depth_camera.MaixSenseA075VRgbDepthCamera`,
:class:`~hal.server.jetson.sen0581_tof_camera.Sen0581TofCamera` (depth-only: RGB is ``None``).
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class RgbDepthCamera(Protocol):
    """Protocol for cameras that provide RGB and depth (meters).

    Implementations must provide:
    - get_camera_frames(): one capture returning (rgb, depth) in standard formats
    - is_ready(): whether the camera is initialized and can capture
    - close(): release resources
    """

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame and return (RGB, depth).

        Returns:
            (rgb, depth): rgb (H, W, 3) uint8, depth (H, W) float32 in meters;
            either may be None on capture failure.
        """
        ...

    def is_ready(self) -> bool:
        """Return True if the camera is initialized and ready to capture."""
        ...

    def close(self) -> None:
        """Release camera resources."""
        ...
