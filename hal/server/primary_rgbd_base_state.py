"""Primary RGB-D base motion fields — same contract as Jetson ZED on ``front_rgbd``.

``HardwareObservations.base_quat_w`` is world-frame orientation as **(x, y, z, w)**.
``base_ang_vel_b`` and ``base_lin_vel_b`` are in the **robot base** frame (rad/s, m/s),
after the primary catalog mount transform (see ``hal.server.jetson.zed_imu``).
"""

from __future__ import annotations

import numpy as np

from hal.server.jetson.sensor_backend_jetson import front_observation_camera_catalog_entry
from hal.server.jetson.zed_imu import ZedImuSample, apply_mount_to_imu_sample
from hal.server.jetson.zed_tracking import tracking_lin_vel_sensor_to_base


def quat_isaac_wxyz_to_hal_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Isaac Lab ``root_quat_w`` / sensor ``(w, x, y, z)`` → HAL ``(x, y, z, w)``."""
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def mount_quat_xyzw_primary() -> np.ndarray:
    """Primary ``front_rgbd`` catalog mount quaternion (base → sensor), (x, y, z, w)."""
    entry = front_observation_camera_catalog_entry()
    return np.array(
        [entry.pose.qx, entry.pose.qy, entry.pose.qz, entry.pose.qw],
        dtype=np.float32,
    )


def base_state_from_primary_zed_samples(
    *,
    sensor_quat_world_xyzw: np.ndarray,
    sensor_ang_vel_b: np.ndarray,
    sensor_lin_vel_sensor: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same mount transforms as ``JetsonHalServer`` primary ZED IMU + tracking."""
    mount = (
        np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
        if mount_quat_base_to_sensor is not None
        else mount_quat_xyzw_primary()
    )
    imu_raw = ZedImuSample(
        base_quat_w=np.asarray(sensor_quat_world_xyzw, dtype=np.float32).reshape(4),
        base_ang_vel_b=np.asarray(sensor_ang_vel_b, dtype=np.float32).reshape(3),
    )
    imu = apply_mount_to_imu_sample(imu_raw, mount)
    lin_b = tracking_lin_vel_sensor_to_base(
        np.asarray(sensor_lin_vel_sensor, dtype=np.float32).reshape(3),
        mount,
    )
    return imu.base_quat_w, imu.base_ang_vel_b, lin_b
