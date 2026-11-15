"""Unit tests for Jetson HAL KrabbyStmMCUSDK."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from hal.client.data_structures.hardware import JointCommand
from hal.server.jetson.krabby_stm_mcusdk import KrabbyStmMCUSDK, _clamp_duty
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def _joint_command(values: dict[str, float]) -> JointCommand:
    names = KRABBY_HEX_DEFINITION.get_mcu_joints()
    positions = np.array([values.get(n, 0.0) for n in names], dtype=np.float32)
    return JointCommand(
        _joint_positions=positions,
        timestamp_ns=1,
        observation_timestamp_ns=1,
        joint_names=names,
    )


class TestClampDuty:
    def test_in_range(self):
        assert _clamp_duty(0.5) == 0.5
        assert _clamp_duty(-0.25) == -0.25

    def test_clamps(self):
        assert _clamp_duty(2.0) == 1.0
        assert _clamp_duty(-3.0) == -1.0


@patch("hal.server.jetson.krabby_stm_mcusdk.KrabbyMCUCANSDK")
class TestKrabbyStmMCUSDK:
    def test_init_rejects_wrong_joint_count(self, mock_sdk_cls: MagicMock):
        with pytest.raises(ValueError, match="18 names.*got 17"):
            KrabbyStmMCUSDK(mcu_joints=("A",) * 17, auto_connect=False)
        mock_sdk_cls.assert_not_called()

    def test_connect_and_close(self, mock_sdk_cls: MagicMock):
        mock_sdk = mock_sdk_cls.return_value
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        sdk = KrabbyStmMCUSDK(mcu_joints=mcu_joints, auto_connect=False)

        assert sdk.connect() is True
        mock_sdk.connect.assert_called_once()
        assert sdk._bus_up is True

        sdk.close()
        mock_sdk.close.assert_called_once()
        assert sdk._bus_up is False

    def test_is_connected_requires_telemetry(self, mock_sdk_cls: MagicMock):
        mock_sdk = mock_sdk_cls.return_value
        mock_sdk.telemetry = {i: {} for i in range(1, 7)}

        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        sdk = KrabbyStmMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        sdk.connect()
        assert sdk.is_connected() is False

        mock_sdk.telemetry[3] = {"J1_pot": 0.1}
        assert sdk.is_connected() is True

    def test_apply_command_maps_mcu_joint_order_to_set_duties(
        self, mock_sdk_cls: MagicMock
    ):
        mock_sdk = mock_sdk_cls.return_value
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        sdk = KrabbyStmMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        sdk.connect()

        values = {name: 0.0 for name in mcu_joints}
        values["FL_hip_yaw"] = 0.75
        values["RR_knee"] = -0.5
        values["ML_hip_pitch"] = 1.5
        sdk.apply_command(_joint_command(values))

        mock_sdk.set_duties.assert_called_once()
        duties = tuple(mock_sdk.set_duties.call_args[0][0])
        assert len(duties) == 18
        assert duties[0] == pytest.approx(0.75)
        assert duties[mcu_joints.index("ML_hip_pitch")] == pytest.approx(1.0)
        assert duties[mcu_joints.index("RR_knee")] == pytest.approx(-0.5)

    def test_apply_command_before_connect_raises(self, mock_sdk_cls: MagicMock):
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        sdk = KrabbyStmMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        with pytest.raises(RuntimeError, match="not connected"):
            sdk.apply_command(_joint_command({}))


def _jetson_server_config():
    import uuid

    from hal.client.config import HalServerConfig
    from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

    uid = uuid.uuid4().hex
    cfg = HalServerConfig(
        observation_bind=f"inproc://test_stm_obs_{uid}",
        command_bind=f"inproc://test_stm_cmd_{uid}",
    )
    robot_definition = KRABBY_HEX_DEFINITION
    model_def = PARKOUR_MODEL_OBSERVATION_DEFINITION
    obs_dims = model_def.get_observation_dimensions(robot_definition)
    return cfg, obs_dims, model_def.action_dim, robot_definition


class TestJetsonHalStmWiring:
    @patch("hal.server.jetson.hal_server.KrabbyMCUSDK")
    @patch("hal.server.jetson.krabby_stm_mcusdk.KrabbyStmMCUSDK")
    def test_hal_server_stm_selects_stm_sdk(
        self, mock_stm_cls: MagicMock, mock_serial_cls: MagicMock
    ):
        from hal.server.jetson.hal_server import JetsonHalServer

        cfg, obs_dims, action_dim, robot_definition = _jetson_server_config()
        server = JetsonHalServer(
            cfg,
            observation_dimensions=obs_dims,
            action_dim=action_dim,
            robot_definition=robot_definition,
            mcu_auto_connect=False,
            stm=True,
            stm_virtual=True,
        )
        try:
            mock_stm_cls.assert_called_once_with(
                mcu_joints=robot_definition.get_mcu_joints(),
                virtual=True,
                auto_connect=False,
            )
            mock_serial_cls.assert_not_called()
            assert server._mcusdk is mock_stm_cls.return_value
        finally:
            server.close()

    @patch("hal.server.jetson.hal_server.KrabbyMCUSDK")
    @patch("hal.server.jetson.krabby_stm_mcusdk.KrabbyStmMCUSDK")
    def test_hal_server_default_selects_serial_sdk(
        self, mock_stm_cls: MagicMock, mock_serial_cls: MagicMock
    ):
        from hal.server.jetson.hal_server import JetsonHalServer

        cfg, obs_dims, action_dim, robot_definition = _jetson_server_config()
        server = JetsonHalServer(
            cfg,
            observation_dimensions=obs_dims,
            action_dim=action_dim,
            robot_definition=robot_definition,
            mcu_auto_connect=False,
            stm=False,
        )
        try:
            mock_serial_cls.assert_called_once()
            mock_stm_cls.assert_not_called()
        finally:
            server.close()

    @patch("hal.server.jetson.runtime.JetsonHalServer")
    def test_create_hal_server_passes_stm_flag(self, mock_server_cls: MagicMock):
        from hal.server.jetson.runtime import create_hal_server

        mock_instance = MagicMock()
        mock_server_cls.return_value = mock_instance
        robot_definition = KRABBY_HEX_DEFINITION

        create_hal_server(
            robot_definition,
            "inproc://obs",
            "inproc://cmd",
            stm=True,
            stm_virtual=True,
        )

        _, kwargs = mock_server_cls.call_args
        assert kwargs["stm"] is True
        assert kwargs["stm_virtual"] is True
        mock_instance.initialize.assert_called_once()
        mock_instance.initialize_sensors.assert_called_once()
        mock_instance.initialize_actuators.assert_called_once()

    def test_common_args_stm_default_and_flag(self):
        import argparse

        from hal.server.jetson.runtime import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        assert parser.parse_args([]).stm is False
        assert parser.parse_args(["--stm"]).stm is True