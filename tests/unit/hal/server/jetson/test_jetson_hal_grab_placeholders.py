"""Jetson HAL: zero tensors when catalog RGB-D grabs fail."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hal.client.config import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.jetson.zed_camera import ZedCamera
from hal.server.jetson.zed_imu import ZedImuSample
from hal.server.server import HalServerBase
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION


def _state_12dof() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ]
    ).astype(np.float32)


@pytest.fixture
def jetson_server() -> JetsonHalServer:
    uid = uuid.uuid4().hex
    cfg = HalServerConfig(
        observation_bind=f"inproc://test_ph_obs_{uid}",
        command_bind=f"inproc://test_ph_cmd_{uid}",
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


def test_grab_failure_fills_rgbd_and_primary_with_zeros(jetson_server):
    mock_cam = MagicMock(spec=ZedCamera)
    mock_cam.get_camera_frames.return_value = (None, None)
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    w, h = jetson_server.camera_resolution
    nf = jetson_server.observation_dimensions.num_scan_front

    with patch.object(HalServerBase, "set_observation") as pub:
        jetson_server.set_observation()

    assert pub.call_count == 1
    # Bound mock records only the arguments after `self`.
    hw_obs = pub.call_args.args[0]
    assert hw_obs.rgbd_by_catalog_id is not None
    assert "front_rgbd" in hw_obs.rgbd_by_catalog_id
    ch = hw_obs.rgbd_by_catalog_id["front_rgbd"]
    assert ch.rgb.shape == (h, w, 3)
    assert ch.depth.shape == (h, w)
    assert np.all(ch.rgb == 0)
    assert np.all(ch.depth == 0.0)
    assert ch.scan_features is not None
    assert ch.scan_features.shape == (nf,)
    assert np.all(ch.scan_features == 0.0)
    assert hw_obs.camera_rgb is not None
    assert hw_obs.camera_depth is not None
    assert np.all(hw_obs.camera_rgb == 0)
    assert np.all(hw_obs.camera_depth == 0.0)
    assert hw_obs.scan_features is not None
    assert np.all(hw_obs.scan_features == 0.0)


def test_grab_exception_propagates(jetson_server):
    mock_cam = MagicMock(spec=ZedCamera)
    mock_cam.get_camera_frames.side_effect = RuntimeError("usb glitch")
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    with pytest.raises(RuntimeError, match="usb glitch"):
        jetson_server.set_observation()


def test_zed_imu_overrides_base_state(jetson_server):
    h, w = 376, 672
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    mock_cam = MagicMock(spec=ZedCamera)
    mock_cam.get_camera_frames.return_value = (rgb, depth)
    mock_cam.has_imu.return_value = True
    mock_cam.has_tracking.return_value = False
    mock_cam.get_imu_sample.return_value = ZedImuSample(
        base_quat_w=np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        base_ang_vel_b=np.array([0.01, 0.02, 0.03], dtype=np.float32),
    )
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.front_camera = mock_cam
    jetson_server._zed_imu_active = True
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    with patch.object(HalServerBase, "set_observation") as pub:
        jetson_server.set_observation()

    hw_obs = pub.call_args.args[0]
    np.testing.assert_allclose(hw_obs.base_quat_w, [0.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(hw_obs.base_ang_vel_b, [0.01, 0.02, 0.03])


def test_zed_imu_missing_sample_falls_back_to_zeros(jetson_server, caplog):
    """IMU present but no sample this tick → zeros/identity + throttled warning, no crash."""
    h, w = 376, 672
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    mock_cam = MagicMock(spec=ZedCamera)
    mock_cam.get_camera_frames.return_value = (rgb, depth)
    mock_cam.has_imu.return_value = True
    mock_cam.has_tracking.return_value = False
    mock_cam.get_imu_sample.return_value = None  # sensors-data fetch failed this tick
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.front_camera = mock_cam
    jetson_server._zed_imu_active = True
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    with caplog.at_level("WARNING"), patch.object(HalServerBase, "set_observation") as pub:
        jetson_server.set_observation()

    hw_obs = pub.call_args.args[0]
    np.testing.assert_allclose(hw_obs.base_ang_vel_b, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(hw_obs.base_quat_w, [0.0, 0.0, 0.0, 1.0])
    assert jetson_server._imu_miss_count == 1
    assert any("ZED IMU sample missing" in r.message for r in caplog.records)


def test_zed_tracking_overrides_base_lin_vel(jetson_server):
    h, w = 376, 672
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    mock_cam = MagicMock(spec=ZedCamera)
    mock_cam.get_camera_frames.return_value = (rgb, depth)
    mock_cam.has_imu.return_value = False
    mock_cam.has_tracking.return_value = True
    mock_cam.get_tracking_lin_vel_sensor.return_value = np.array(
        [0.5, 0.0, 0.0], dtype=np.float32
    )
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.front_camera = mock_cam
    jetson_server._zed_tracking_active = True
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    with patch.object(HalServerBase, "set_observation") as pub:
        jetson_server.set_observation()

    hw_obs = pub.call_args.args[0]
    np.testing.assert_allclose(hw_obs.base_lin_vel_b, [0.5, 0.0, 0.0])


def _mcu_telemetry(pos, current, cal_state=2):
    return MagicMock(pos=pos, current=current, cal_state=cal_state)


_HEX_NAMES = [
    f"{leg}_{j}"
    for leg in ("FL", "FR", "ML", "MR", "RL", "RR")
    for j in ("hip_yaw", "hip_pitch", "knee")
]


def _hex_server(jetson_server, telemetry):
    """Point the fixture server at the 18-joint hex names + a mocked MCU."""
    jetson_server.robot_definition = MagicMock()
    jetson_server.robot_definition.get_joint_names.return_value = tuple(_HEX_NAMES)
    mock_mcu = MagicMock()
    mock_mcu.read_telemetry.return_value = telemetry
    jetson_server._mcusdk = mock_mcu
    return jetson_server


def test_apply_mcu_telemetry_overrides_positions_and_contacts(jetson_server):
    """6b/6e: telemetry pos lands in joint_positions; leg current → contact_forces."""
    telemetry = {n: _mcu_telemetry(0.5, 0) for n in _HEX_NAMES}
    telemetry["FL_knee"] = _mcu_telemetry(0.7, 300)  # one loaded joint on the FL leg
    server = _hex_server(jetson_server, telemetry)

    pos = np.zeros(18, dtype=np.float32)
    vel = np.zeros(18, dtype=np.float32)
    pos, vel, contacts = server._apply_mcu_telemetry(pos, vel)

    assert pos[_HEX_NAMES.index("FL_knee")] == pytest.approx(0.7)
    assert np.allclose(pos[[i for i, n in enumerate(_HEX_NAMES) if n != "FL_knee"]], 0.5)
    # FL leg summed current = 300 → slot 0 (FL) at the firm-contact ceiling.
    assert contacts[0] == pytest.approx(0.5)
    # MR is dropped (not a slot); ML/RL/RR legs carry 0 current → -0.5 (no contact).
    assert contacts[2] == pytest.approx(-0.5)


def test_apply_mcu_telemetry_velocity_zero_first_tick(jetson_server):
    """6c: first telemetry tick has no prior sample → velocity stays 0 (no crash)."""
    telemetry = {n: _mcu_telemetry(0.5, 0) for n in _HEX_NAMES}
    server = _hex_server(jetson_server, telemetry)

    _, vel, _ = server._apply_mcu_telemetry(np.zeros(18, dtype=np.float32), np.zeros(18, dtype=np.float32))
    assert np.all(vel == 0.0)


def test_apply_mcu_telemetry_no_mcu_returns_inputs(jetson_server):
    """No MCU → inputs unchanged and contact_forces None (caller keeps zeros)."""
    jetson_server._mcusdk = None
    pos = np.full(18, 0.3, dtype=np.float32)
    vel = np.full(18, 0.1, dtype=np.float32)
    out_pos, out_vel, contacts = jetson_server._apply_mcu_telemetry(pos, vel)
    assert contacts is None
    assert np.array_equal(out_pos, pos)
    assert np.array_equal(out_vel, vel)


def test_shape_mismatch_raises(jetson_server):
    mock_cam = MagicMock(spec=ZedCamera)
    h, w = jetson_server.camera_resolution[1], jetson_server.camera_resolution[0]
    wrong = np.zeros((h // 2, w, 3), dtype=np.uint8)
    mock_cam.get_camera_frames.return_value = (wrong, np.zeros((h, w), dtype=np.float32))
    jetson_server._hal_rgbd_cameras["front_rgbd"] = mock_cam
    jetson_server.initialize()
    jetson_server._build_state_vector = lambda: _state_12dof()

    with pytest.raises(RuntimeError, match="frame shape mismatch"):
        jetson_server.set_observation()
