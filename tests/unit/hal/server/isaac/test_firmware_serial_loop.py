"""Unit tests for serial_sdk fullsim transport."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_serial_loop import SerialSdkCommandLoop
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def test_serial_sdk_loop_sparse_jog_and_firmware_tick():
    hal_mcu = MagicMock()
    firmware_mcu = MagicMock()
    hal_mcu._mcu = firmware_mcu
    core = MagicMock()
    core.command_pwm_for_actuator.return_value = 0
    mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
    loop = SerialSdkCommandLoop(hal_mcu, core, mcu_joints=mcu_joints)
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    command = JointCommand(
        _joint_positions=np.zeros(18, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )

    loop.apply_hal_command(command)

    firmware_mcu.send_command_joints_hold.assert_called_once()
    assert core.tick_firmware_without_physics.call_count >= 1


def test_isaac_firmware_mcusdk_uses_apply_hal_command_when_present():
    mock_loop = MagicMock()
    mock_loop.apply_hal_command = MagicMock()
    mock_loop.signed_command_pwm.return_value = 127

    sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(
        KRABBY_HEX_DEFINITION,
        firmware_loop=mock_loop,
    )
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    values = [0.0] * 18
    values[names.index("FL_hip_pitch")] = 0.2
    command = JointCommand(
        _joint_positions=np.array(values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )

    action = sdk.apply_command(command)

    mock_loop.apply_hal_command.assert_called_once_with(command)
    mock_loop.jog_actuator.assert_not_called()
    fl_idx = names.index("FL_hip_pitch")
    assert action[0, fl_idx].item() == pytest.approx(127 / 255.0)
