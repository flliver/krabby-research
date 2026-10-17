"""Quaternion helpers shared by Jetson ZED HAL and Isaac Sim (no jetson package dependency)."""

from __future__ import annotations

import numpy as np


def quat_multiply_xyzw(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
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


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.astype(np.float32, copy=True)
    out[0:3] *= -1.0
    return out


def rotate_vector_by_quat_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
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


def rotate_vector_sensor_to_base(
    v_sensor: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray,
) -> np.ndarray:
    """Rotate a vector from the sensor frame into the robot base frame."""
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    v = np.asarray(v_sensor, dtype=np.float32).reshape(3)
    if np.allclose(mount, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)):
        return v
    return rotate_vector_by_quat_xyzw(quat_conjugate_xyzw(mount), v)


def rotate_vector_base_to_sensor(
    v_base: np.ndarray,
    mount_quat_base_to_sensor: np.ndarray,
) -> np.ndarray:
    """Rotate a vector from the robot base frame into the sensor frame."""
    mount = np.asarray(mount_quat_base_to_sensor, dtype=np.float32).reshape(4)
    v = np.asarray(v_base, dtype=np.float32).reshape(3)
    if np.allclose(mount, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)):
        return v
    return rotate_vector_by_quat_xyzw(mount, v)
