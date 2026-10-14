"""Unit tests for IsaacSimFirmwareMCUSDK."""

import time

import numpy as np
import pytest
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_command_mapping import rad_to_jog_pwm
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def _make_command(joint_names: tuple[str, ...], values: list[float]) -> JointCommand:
    return JointCommand(
        _joint_positions=np.array(values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=joint_names,
    )


class TestIsaacSimFirmwareMCUSDKApplyCommand:
    def test_action_tensor_shape_matches_total_joint_count_hex(self):
        robot = KRABBY_HEX_DEFINITION
        sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(robot)
        names = robot.get_joint_names()
        command = _make_command(names, [0.1] * len(names))

        action = sdk.apply_command(command, num_envs=1)

        assert isinstance(action, torch.Tensor)
        assert action.dtype == torch.float32
        assert action.shape == (1, robot.get_total_joint_count())

    def test_action_tensor_shape_matches_total_joint_count_quad(self):
        robot = KRABBY_QUAD_DEFINITION
        sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(robot)
        names = robot.get_joint_names()
        command = _make_command(names, [0.0] * len(names))

        action = sdk.apply_command(command, num_envs=4)

        assert action.shape == (4, robot.get_total_joint_count())

    def test_values_match_thresholded_command(self):
        robot = KRABBY_QUAD_DEFINITION
        sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(robot)
        names = robot.get_joint_names()
        values = [0.2, 0.0, -0.3] * 4
        command = _make_command(names, values)

        action = sdk.apply_command(command, num_envs=1)

        for name, expected in zip(names, values):
            assert action[0, names.index(name)].item() == pytest.approx(expected)


class _MockFirmwareLoop:
    def __init__(self, pwm_by_id: dict[str, int]) -> None:
        self._pwm = pwm_by_id
        self.jog_calls: list[tuple[str, int]] = []
        self.tick_count = 0

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None:
        self.jog_calls.append((firmware_actuator_id, pwm))
        self._pwm[firmware_actuator_id] = pwm

    def tick_firmware(self) -> None:
        self.tick_count += 1

    def signed_command_pwm(self, firmware_actuator_id: str) -> int:
        return self._pwm.get(firmware_actuator_id, 0)


class TestIsaacSimFirmwareMCUSDKFirmwarePath:
    def test_apply_command_reads_pin_pwm_back_as_normalized_action(self):
        robot = KRABBY_HEX_DEFINITION
        mock = _MockFirmwareLoop({})
        sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(
            robot,
            firmware_loop=mock,
        )
        names = robot.get_joint_names()
        values = [0.0] * len(names)
        values[names.index("FL_hip_pitch")] = 0.2
        command = _make_command(names, values)

        action = sdk.apply_command(command)

        assert mock.tick_count == 1
        assert any(call[0] == "FLHL" and call[1] != 0 for call in mock.jog_calls)
        assert all(pwm == 0 for _fw_id, pwm in mock.jog_calls if _fw_id != "FLHL")
        assert len(mock.jog_calls) == 1
        fl_hl_idx = names.index("FL_hip_pitch")
        expected_pwm = rad_to_jog_pwm(0.2)
        assert action[0, fl_hl_idx].item() == pytest.approx(expected_pwm / 255.0)
