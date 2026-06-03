"""Unit tests for ZED IMU parsing and base-frame transforms."""

import numpy as np

from hal.server.jetson.zed_imu import (
    ZedImuSample,
    apply_mount_to_imu_sample,
    parse_zed_imu_data,
)


class _MockOrientation:
    def __init__(self, quat):
        self._quat = quat

    def get(self):
        return self._quat


class _MockPose:
    def __init__(self, quat):
        self._orientation = _MockOrientation(quat)

    def get_orientation(self):
        return self._orientation


class _MockImuData:
    def __init__(self, quat, ang_vel_deg):
        self._quat = quat
        self._ang_vel_deg = ang_vel_deg

    def get_pose(self):
        return _MockPose(self._quat)

    def get_angular_velocity(self):
        return self._ang_vel_deg


def test_parse_zed_imu_data_converts_gyro_to_rad_s():
    imu = _MockImuData([0.0, 0.0, 0.0, 1.0], [90.0, 0.0, 0.0])
    sample = parse_zed_imu_data(imu)
    assert sample is not None
    np.testing.assert_allclose(sample.base_quat_w, [0, 0, 0, 1], rtol=1e-5)
    expected = np.deg2rad([90.0, 0.0, 0.0]).astype(np.float32)
    np.testing.assert_allclose(sample.base_ang_vel_b, expected, rtol=1e-5)


def test_parse_zed_imu_data_rejects_non_finite():
    imu = _MockImuData([np.nan, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    assert parse_zed_imu_data(imu) is None


def test_apply_mount_identity_is_noop():
    sample = ZedImuSample(
        base_quat_w=np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        base_ang_vel_b=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    out = apply_mount_to_imu_sample(sample, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(out.base_quat_w, sample.base_quat_w)
    np.testing.assert_array_equal(out.base_ang_vel_b, sample.base_ang_vel_b)


def test_apply_mount_rotates_angular_velocity():
    # 90° about Z: base x -> sensor y
    mount = np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float32)
    sample = ZedImuSample(
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    out = apply_mount_to_imu_sample(sample, mount)
    np.testing.assert_allclose(out.base_ang_vel_b, [1.0, 0.0, 0.0], atol=1e-5)
