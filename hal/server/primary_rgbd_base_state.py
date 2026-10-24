"""Primary RGB-D base motion fields — same contract as Jetson ZED on ``front_rgbd``.

``HardwareObservations.base_quat_w`` is world-frame orientation as **(x, y, z, w)**.
``base_ang_vel_b`` and ``base_lin_vel_b`` are in the **robot base** frame (rad/s, m/s),
after the primary catalog mount transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hal.server.quat_kinematics import (
    quat_conjugate_xyzw,
    quat_multiply_xyzw,
    rotate_vector_sensor_to_base,
)

# ``front_rgbd`` primary mount (base → sensor), (x, y, z, w); matches Jetson/Isaac catalog.
_DEFAULT_FRONT_RGBD_MOUNT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ZedImuSample:
    """IMU-style sample in sensor frame (shared by Jetson ZED SDK and Isaac synthesis)."""

    base_quat_w: np.ndarray  # (4,) float32, world frame, (x, y, z, w)
    base_ang_vel_b: np.ndarray  # (3,) float32, sensor frame, rad/s


def quat_isaac_wxyz_to_hal_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Isaac Lab ``root_quat_w`` / sensor ``(w, x, y, z)`` → HAL ``(x, y, z, w)``."""
    q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def mount_quat_xyzw_primary() -> np.ndarray:
    """Primary ``front_rgbd`` catalog mount quaternion (base → sensor), (x, y, z, w)."""
    try:
        from hal.server.jetson.sensor_backend_jetson import front_observation_camera_catalog_entry
    except ModuleNotFoundError:
        return _DEFAULT_FRONT_RGBD_MOUNT_XYZW.copy()
    entry = front_observation_camera_catalog_entry()
    return np.array(
        [entry.pose.qx, entry.pose.qy, entry.pose.qz, entry.pose.qw],
        dtype=np.float32,
    )


def apply_mount_to_imu_sample(
    sample: ZedImuSample,
    mount_quat_base_to_sensor: np.ndarray,
) -> ZedImuSample:
    """Express IMU orientation and rates in the robot base frame."""
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    if np.allclose(mount, _DEFAULT_FRONT_RGBD_MOUNT_XYZW):
        return sample

    mount_inv = quat_conjugate_xyzw(mount)
    q_base = quat_multiply_xyzw(sample.base_quat_w, mount_inv)
    omega_b = rotate_vector_sensor_to_base(sample.base_ang_vel_b, mount)
    return ZedImuSample(base_quat_w=q_base, base_ang_vel_b=omega_b)


def tracking_lin_vel_sensor_to_base(
    lin_vel_sensor: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray,
) -> np.ndarray:
    """Map sensor-frame linear velocity into the robot base frame."""
    return rotate_vector_sensor_to_base(lin_vel_sensor, mount_quat_base_to_sensor)


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
