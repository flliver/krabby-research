"""Tests for shared primary RGB-D base state conventions."""

from __future__ import annotations

import numpy as np

from hal.server.jetson.zed_imu import ZedImuSample, apply_mount_to_imu_sample
from hal.server.jetson.zed_tracking import tracking_lin_vel_sensor_to_base
from hal.server.primary_rgbd_base_state import (
    base_state_from_primary_zed_samples,
    mount_quat_xyzw_primary,
    quat_isaac_wxyz_to_hal_xyzw,
)


def test_quat_isaac_wxyz_to_hal_xyzw_identity() -> None:
    np.testing.assert_allclose(
        quat_isaac_wxyz_to_hal_xyzw(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        [0.0, 0.0, 0.0, 1.0],
    )


def test_base_state_matches_jetson_mount_helpers() -> None:
    mount = mount_quat_xyzw_primary()
    q_sensor = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    omega_s = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lin_s = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    expected_q, expected_w, expected_v = base_state_from_primary_zed_samples(
        sensor_quat_world_xyzw=q_sensor,
        sensor_ang_vel_b=omega_s,
        sensor_lin_vel_sensor=lin_s,
        mount_quat_base_to_sensor=mount,
    )

    imu = apply_mount_to_imu_sample(
        ZedImuSample(base_quat_w=q_sensor, base_ang_vel_b=omega_s),
        mount,
    )
    lin_b = tracking_lin_vel_sensor_to_base(lin_s, mount)

    np.testing.assert_allclose(expected_q, imu.base_quat_w)
    np.testing.assert_allclose(expected_w, imu.base_ang_vel_b)
    np.testing.assert_allclose(expected_v, lin_b)
