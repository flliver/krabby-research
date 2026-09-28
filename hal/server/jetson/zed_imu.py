"""ZED onboard IMU parsing and base-frame transforms for Jetson HAL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Stereolabs SDK reports gyro rates in deg/s (see IMUData.get_angular_velocity_covariance docs).
_DEG_TO_RAD = np.pi / 180.0


@dataclass(frozen=True, slots=True)
class ZedImuSample:
    """One IMU sample aligned to a camera grab (TIME_REFERENCE.IMAGE)."""

    base_quat_w: np.ndarray  # (4,) float32, world frame, (x, y, z, w)
    base_ang_vel_b: np.ndarray  # (3,) float32, robot base frame, rad/s


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.astype(np.float32, copy=True)
    out[0:3] *= -1.0
    return out


def _quat_multiply(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Hamilton product; both quaternions (x, y, z, w)."""
    x1, y1, z1, w1 = (float(q_left[i]) for i in range(4))
    x2, y2, z2, w2 = (float(q_right[i]) for i in range(4))
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _rotate_vector_by_quat(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q (x, y, z, w)."""
    q = q_xyzw.astype(np.float64)
    v = v.astype(np.float64)
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )

    return _mul(_mul(q, qv), q_conj)[0:3].astype(np.float32)


def apply_mount_to_imu_sample(
    sample: ZedImuSample,
    mount_quat_base_to_sensor: np.ndarray,
) -> ZedImuSample:
    """Express IMU orientation and rates in the robot base frame.

    ``mount_quat_base_to_sensor`` is the catalog ``SensorPose`` quaternion (base → sensor).
    ZED fused orientation is the sensor frame in world; base orientation is
    ``q_world_base = q_world_sensor * conj(q_base_sensor)``.
    Angular velocity in base frame: ``omega_base = R_base_sensor @ omega_sensor``.
    """
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    if np.allclose(mount, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)):
        return sample

    mount_inv = _quat_conjugate(mount)
    q_base = _quat_multiply(sample.base_quat_w, mount_inv)
    omega_b = rotate_vector_sensor_to_base(sample.base_ang_vel_b, mount)
    return ZedImuSample(base_quat_w=q_base, base_ang_vel_b=omega_b)


def rotate_vector_sensor_to_base(
    v_sensor: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray,
) -> np.ndarray:
    """Rotate a vector from the ZED sensor frame into the robot base frame.

    ``mount_quat_base_to_sensor`` is the catalog ``SensorPose`` quaternion (base → sensor).
    """
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    v = np.asarray(v_sensor, dtype=np.float32).reshape(3)
    if np.allclose(mount, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)):
        return v
    return _rotate_vector_by_quat(_quat_conjugate(mount), v)


def rotate_vector_base_to_sensor(
    v_base: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray,
) -> np.ndarray:
    """Rotate a vector from the robot base frame into the ZED sensor frame."""
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    v = np.asarray(v_base, dtype=np.float32).reshape(3)
    if np.allclose(mount, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)):
        return v
    return _rotate_vector_by_quat(mount, v)


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
