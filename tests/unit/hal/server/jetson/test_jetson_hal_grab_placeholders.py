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
