"""ZED IMU path: get_imu() unit conversion and JetsonHalServer body-frame wiring."""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from hal.client.config import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.jetson.zed_camera import ZedCamera, ZedImuSample
from hal.server.server import HalServerBase
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION


# ---------------------------------------------------------------------------
# Fake pyzed plumbing (get_imu talks only to these surfaces)
# ---------------------------------------------------------------------------

_SUCCESS = 0
_FAILURE = 1


class _FakeOrientation:
    def get(self):
        return [0.0, 0.0, 0.0, 1.0]


class _FakePose:
    def get_orientation(self):
        return _FakeOrientation()


class _FakeTimestamp:
    def get_nanoseconds(self):
        return 123_456_789


class _FakeImuData:
    def __init__(self, with_timestamp: bool = True):
        if with_timestamp:
            self.timestamp = _FakeTimestamp()

    def get_angular_velocity(self):
        return [90.0, -45.0, 180.0]  # deg/s

    def get_linear_acceleration(self):
        return [0.1, 9.8, -0.2]  # m/s²

    def get_pose(self):
        return _FakePose()


def _make_zed(status: int = _SUCCESS, initialized: bool = True, with_timestamp: bool = True) -> ZedCamera:
    imu_data = _FakeImuData(with_timestamp=with_timestamp)

    class _FakeSensorsData:
        def get_imu_data(self):
            return imu_data

    cam = ZedCamera.__new__(ZedCamera)
    cam.initialized = initialized
    cam._zed_module = SimpleNamespace(
        SensorsData=_FakeSensorsData,
        TIME_REFERENCE=SimpleNamespace(CURRENT="current"),
        ERROR_CODE=SimpleNamespace(SUCCESS=_SUCCESS),
    )
    refs_used = []

    def _get_sensors_data(sensors_data, ref):
        refs_used.append(ref)
        return status

    cam.camera = SimpleNamespace(get_sensors_data=_get_sensors_data)
    cam._test_refs_used = refs_used
    return cam


class TestGetImu:
    def test_converts_deg_s_to_rad_s(self):
        sample = _make_zed().get_imu()
        assert isinstance(sample, ZedImuSample)
        np.testing.assert_allclose(
            sample.ang_vel_rad_s, [np.pi / 2, -np.pi / 4, np.pi], rtol=1e-6
        )
        np.testing.assert_allclose(sample.lin_acc_m_s2, [0.1, 9.8, -0.2], rtol=1e-6)
        np.testing.assert_allclose(sample.orientation_quat_xyzw, [0, 0, 0, 1])
        assert sample.timestamp_ns == 123_456_789
        assert sample.ang_vel_rad_s.dtype == np.float32
        assert sample.orientation_quat_xyzw.dtype == np.float32

    def test_uses_current_time_reference(self):
        # TIME_REFERENCE.IMAGE freezes (uninitialized quat) unless grab() runs;
        # the IMU path must not depend on the video pipeline.
        cam = _make_zed()
        cam.get_imu()
        assert cam._test_refs_used == ["current"]

    def test_non_success_status_returns_none(self):
        assert _make_zed(status=_FAILURE).get_imu() is None

    def test_uninitialized_returns_none(self):
        assert _make_zed(initialized=False).get_imu() is None

    def test_missing_sensor_timestamp_falls_back_to_host_time(self):
        sample = _make_zed(with_timestamp=False).get_imu()
        assert isinstance(sample, ZedImuSample)
        assert sample.timestamp_ns > 0


# ---------------------------------------------------------------------------
# JetsonHalServer wiring (AC 5c / 5e)
# ---------------------------------------------------------------------------


@pytest.fixture
def jetson_server() -> JetsonHalServer:
    uid = uuid.uuid4().hex
    cfg = HalServerConfig(
        observation_bind=f"inproc://test_imu_obs_{uid}",
        command_bind=f"inproc://test_imu_cmd_{uid}",
    )
    model_def = PARKOUR_MODEL_OBSERVATION_DEFINITION
    rd = MagicMock()
    rd.get_total_joint_count.return_value = 12
    rd.get_joint_names.return_value = tuple(f"j{i}" for i in range(12))
    rd.get_mcu_joints.return_value = ()
    rd.get_num_prop.return_value = 48
    rd.get_observation_joint_count.return_value = 12
    obs_dims = model_def.get_observation_dimensions(rd)
    server = JetsonHalServer(
        cfg,
        observation_dimensions=obs_dims,
        action_dim=model_def.action_dim,
        robot_definition=rd,
    )
    try:
        yield server
    finally:
        server.close()


def _imu_mock(sample) -> MagicMock:
    cam = MagicMock(spec=ZedCamera)
    cam.get_imu.return_value = sample
    return cam


def _sample(ang_vel=(0.1, -0.2, 0.3), quat=(0.0, 0.0, 0.0, 1.0), timestamp_ns=1000) -> ZedImuSample:
    return ZedImuSample(
        ang_vel_rad_s=np.array(ang_vel, dtype=np.float32),
        lin_acc_m_s2=np.zeros(3, dtype=np.float32),
        orientation_quat_xyzw=np.array(quat, dtype=np.float32),
        timestamp_ns=timestamp_ns,
    )


def test_imu_sample_populates_observation(jetson_server, monkeypatch):
    # 30° roll about x: quat xyzw = (sin15°, 0, 0, cos15°)
    half = np.deg2rad(30) / 2
    quat = (np.sin(half), 0.0, 0.0, np.cos(half))
    jetson_server._imu_camera = _imu_mock(_sample(quat=quat))
    jetson_server.initialize()

    published = []
    monkeypatch.setattr(
        HalServerBase, "set_observation", lambda self, obs: published.append(obs)
    )
    jetson_server.set_observation()

    hw_obs = published[0]
    np.testing.assert_allclose(hw_obs.base_ang_vel_b, [0.1, -0.2, 0.3], rtol=1e-6)
    # Identity mount rotation: body quat == camera quat (up to sign)
    q = hw_obs.base_quat_w
    if q[3] * quat[3] < 0:
        q = -q
    np.testing.assert_allclose(q, quat, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(hw_obs.base_quat_w), 1.0, rtol=1e-6)


def test_missing_imu_sample_emits_zeros_and_rate_limited_warning(jetson_server, caplog):
    jetson_server._imu_camera = _imu_mock(None)

    with caplog.at_level(logging.WARNING, logger="hal.server.jetson.hal_server"):
        ang_vel, quat = jetson_server._imu_body_frame()
        np.testing.assert_array_equal(ang_vel, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(quat, np.array([0, 0, 0, 1], dtype=np.float32))
        assert jetson_server._imu_miss_count == 1
        assert sum("IMU sample missing" in r.message for r in caplog.records) == 1

        # Second miss: counter advances, no new warning until count hits 101
        jetson_server._imu_body_frame()
        assert jetson_server._imu_miss_count == 2
        assert sum("IMU sample missing" in r.message for r in caplog.records) == 1


def test_no_imu_source_emits_zeros_and_single_warning(jetson_server, caplog):
    assert jetson_server._imu_camera is None
    with caplog.at_level(logging.WARNING, logger="hal.server.jetson.hal_server"):
        ang_vel, quat = jetson_server._imu_body_frame()
        jetson_server._imu_body_frame()
    np.testing.assert_array_equal(ang_vel, np.zeros(3))
    np.testing.assert_array_equal(quat, [0, 0, 0, 1])
    assert sum("No IMU source" in r.message for r in caplog.records) == 1


def test_stale_timestamp_logs_info_once(jetson_server, caplog):
    jetson_server._imu_camera = _imu_mock(_sample(timestamp_ns=42))
    with caplog.at_level(logging.INFO, logger="hal.server.jetson.hal_server"):
        jetson_server._imu_body_frame()  # first sighting of ts=42, not stale
        jetson_server._imu_body_frame()  # same ts → stale, logs once
        jetson_server._imu_body_frame()  # still stale, no repeat
    assert sum("not advancing" in r.message for r in caplog.records) == 1

    # Timestamp advances → stale flag resets
    jetson_server._imu_camera = _imu_mock(_sample(timestamp_ns=43))
    jetson_server._imu_body_frame()
    assert not jetson_server._imu_stale_logged
