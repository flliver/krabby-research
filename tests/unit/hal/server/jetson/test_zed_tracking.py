"""Unit tests for ZED positional-tracking velocity parsing."""

import numpy as np

from hal.server.jetson.zed_tracking import (
    parse_zed_tracking_lin_vel_sensor,
    positional_tracking_state_ok,
    tracking_lin_vel_sensor_to_base,
)


class _MockTwist:
    def __init__(self, vx, vy, vz):
        self._v = [vx, vy, vz]

    def __getitem__(self, i):
        return self._v[i]


class _MockPose:
    def __init__(self, vx, vy, vz):
        self.twist = _MockTwist(vx, vy, vz)


def test_parse_zed_tracking_lin_vel_sensor() -> None:
    pose = _MockPose(1.0, 2.0, 3.0)
    lin_vel = parse_zed_tracking_lin_vel_sensor(pose)
    assert lin_vel is not None
    np.testing.assert_allclose(lin_vel, [1.0, 2.0, 3.0])


def test_parse_zed_tracking_lin_vel_sensor_rejects_non_finite() -> None:
    pose = _MockPose(np.nan, 0.0, 0.0)
    assert parse_zed_tracking_lin_vel_sensor(pose) is None


def test_tracking_lin_vel_sensor_to_base_rotates() -> None:
    mount = np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float32)
    lin_vel_sensor = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = tracking_lin_vel_sensor_to_base(lin_vel_sensor, mount)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-5)


def test_positional_tracking_state_ok() -> None:
    class _Pts:
        OK = 1
        SEARCHING = 2

    class _Sl:
        POSITIONAL_TRACKING_STATE = _Pts

    sl = _Sl()
    assert positional_tracking_state_ok(sl.POSITIONAL_TRACKING_STATE.OK, sl) is True
    assert positional_tracking_state_ok(sl.POSITIONAL_TRACKING_STATE.SEARCHING, sl) is False
