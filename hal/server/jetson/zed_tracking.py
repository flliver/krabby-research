"""ZED positional-tracking linear velocity for Jetson HAL."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hal.server.primary_rgbd_base_state import tracking_lin_vel_sensor_to_base

__all__ = [
    "positional_tracking_state_ok",
    "tracking_reference_frame_camera",
    "parse_zed_tracking_lin_vel_sensor",
    "tracking_lin_vel_sensor_to_base",
]


def positional_tracking_state_ok(state: Any, sl: Any) -> bool:
    """True when ZED reports ``POSITIONAL_TRACKING_STATE.OK``."""
    return state == sl.POSITIONAL_TRACKING_STATE.OK


def tracking_reference_frame_camera(sl: Any) -> Any:
    """``REFERENCE_FRAME.CAMERA`` — twist from ``get_position`` is in the camera frame."""
    return sl.REFERENCE_FRAME.CAMERA


def parse_zed_tracking_lin_vel_sensor(pose: Any) -> Optional[np.ndarray]:
    """Linear velocity (3,) float32 m/s in the camera frame from ``sl.Pose.twist``."""
    twist = pose.twist
    if twist is None:
        return None
    v = np.asarray([twist[0], twist[1], twist[2]], dtype=np.float32)
    if v.shape != (3,) or not np.isfinite(v).all():
        return None
    return v
