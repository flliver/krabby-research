"""Isaac Sim primary ZED-equivalent base state (root + catalog mount only)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from hal.server.isaac.primary_zed_base_state import isaac_primary_rgbd_base_state
from hal.server.primary_rgbd_base_state import mount_quat_xyzw_primary, quat_isaac_wxyz_to_hal_xyzw
from hal.server.jetson.zed_imu import _quat_multiply


def _mock_robot(
    *,
    root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ang_vel_b: tuple[float, float, float] = (0.01, 0.02, 0.03),
    lin_vel_b: tuple[float, float, float] = (0.2, 0.0, 0.0),
) -> MagicMock:
    robot = MagicMock()
    robot.data.root_quat_w = torch.tensor([list(root_quat_wxyz)], dtype=torch.float32)
    robot.data.root_ang_vel_b = torch.tensor([list(ang_vel_b)], dtype=torch.float32)
    robot.data.root_lin_vel_b = torch.tensor([list(lin_vel_b)], dtype=torch.float32)
    return robot


def test_isaac_identity_root_matches_hal_xyzw() -> None:
    q, w, v = isaac_primary_rgbd_base_state(_mock_robot())
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(w, [0.01, 0.02, 0.03], atol=1e-5)
    np.testing.assert_allclose(v, [0.2, 0.0, 0.0], atol=1e-5)


def test_isaac_sensor_world_quat_is_root_times_mount() -> None:
    root_wxyz = (0.70710678, 0.70710678, 0.0, 0.0)
    mount = mount_quat_xyzw_primary()
    q_world_base = quat_isaac_wxyz_to_hal_xyzw(np.array(root_wxyz, dtype=np.float32))
    expected_sensor = _quat_multiply(q_world_base, mount)
    q, _, _ = isaac_primary_rgbd_base_state(_mock_robot(root_quat_wxyz=root_wxyz))
    np.testing.assert_allclose(q, expected_sensor, atol=1e-4)
