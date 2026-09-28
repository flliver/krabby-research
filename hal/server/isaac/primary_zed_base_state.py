"""Isaac Sim base motion aligned with Jetson primary ZED HAL output.

Single model (no per-field fallbacks): treat the primary ``front_rgbd`` mount as fixed on the
articulation root, synthesize the same raw ZED samples Jetson would pass into
``apply_mount_to_imu_sample`` / ``tracking_lin_vel_sensor_to_base``, then run the shared mount
transform in ``hal.server.primary_rgbd_base_state``.

- **Orientation (sensor world):** ``q_world_sensor = q_world_base ⊗ q_base_sensor``
- **Gyro / tracking twist (sensor frame):** ``ω_sensor``, ``v_sensor`` from root body twist
  rotated base → sensor via the catalog mount quaternion

Jetson hardware fills those raw samples from the ZED SDK instead of root state; the HAL output
path is identical after that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hal.server.jetson.zed_imu import _quat_multiply, rotate_vector_base_to_sensor
from hal.server.primary_rgbd_base_state import (
    base_state_from_primary_zed_samples,
    mount_quat_xyzw_primary,
    quat_isaac_wxyz_to_hal_xyzw,
)

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _first_env_row(tensor: torch.Tensor) -> torch.Tensor:
    """Isaac Lab articulation buffers are ``(num_envs, …)``; HAL uses env 0."""
    return tensor[0] if tensor.ndim >= 2 else tensor


def _tensor_to_numpy_f32(row: torch.Tensor) -> np.ndarray:
    return row.detach().cpu().to(dtype=torch.float32).reshape(-1).numpy()


def _vec3_from_root(robot: Any, attr: str) -> np.ndarray:
    row = _first_env_row(getattr(robot.data, attr))
    v = _tensor_to_numpy_f32(row)
    if v.size < 3 or not np.isfinite(v[:3]).all():
        return np.zeros(3, dtype=np.float32)
    return np.asarray(v[:3], dtype=np.float32)


def _quat_wxyz_from_root(robot: Any) -> np.ndarray:
    row = _first_env_row(robot.data.root_quat_w)
    q = _tensor_to_numpy_f32(row)
    if q.size < 4 or not np.isfinite(q[:4]).all():
        return _IDENTITY_QUAT_WXYZ.copy()
    return np.asarray(q[:4], dtype=np.float32)


def isaac_primary_rgbd_base_state(robot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(base_quat_w, base_ang_vel_b, base_lin_vel_b)`` like Jetson primary ZED."""
    mount = mount_quat_xyzw_primary()
    q_world_base = quat_isaac_wxyz_to_hal_xyzw(_quat_wxyz_from_root(robot))
    q_world_sensor = _quat_multiply(q_world_base, mount)
    omega_sensor = rotate_vector_base_to_sensor(_vec3_from_root(robot, "root_ang_vel_b"), mount)
    lin_sensor = rotate_vector_base_to_sensor(_vec3_from_root(robot, "root_lin_vel_b"), mount)

    return base_state_from_primary_zed_samples(
        sensor_quat_world_xyzw=q_world_sensor,
        sensor_ang_vel_b=omega_sensor,
        sensor_lin_vel_sensor=lin_sensor,
        mount_quat_base_to_sensor=mount,
    )
