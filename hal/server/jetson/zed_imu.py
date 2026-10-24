"""ZED onboard IMU parsing and base-frame transforms for Jetson HAL."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hal.server.primary_rgbd_base_state import (
    ZedImuSample,
    apply_mount_to_imu_sample,
)
from hal.server.quat_kinematics import rotate_vector_base_to_sensor, rotate_vector_sensor_to_base

# Stereolabs SDK reports gyro rates in deg/s (see IMUData.get_angular_velocity_covariance docs).
_DEG_TO_RAD = np.pi / 180.0

__all__ = [
    "ZedImuSample",
    "apply_mount_to_imu_sample",
    "rotate_vector_base_to_sensor",
    "rotate_vector_sensor_to_base",
    "parse_zed_imu_data",
]


def parse_zed_imu_data(imu_data: Any) -> Optional[ZedImuSample]:
    """Build a sample from ``sl.SensorsData.get_imu_data()`` after ``get_sensors_data``.

    Returns None when orientation or angular velocity cannot be read.
    """
    try:
        orientation = imu_data.get_pose().get_orientation().get()
        angular_velocity_deg = imu_data.get_angular_velocity()
    except (AttributeError, TypeError, RuntimeError):
        return None

    if orientation is None or angular_velocity_deg is None:
        return None

    quat = np.asarray(orientation, dtype=np.float32).reshape(4)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        return None

    ang_deg = np.asarray(angular_velocity_deg, dtype=np.float32).reshape(3)
    if ang_deg.shape != (3,) or not np.isfinite(ang_deg).all():
        return None

    ang_rad = (ang_deg * _DEG_TO_RAD).astype(np.float32)
    return ZedImuSample(base_quat_w=quat, base_ang_vel_b=ang_rad)
