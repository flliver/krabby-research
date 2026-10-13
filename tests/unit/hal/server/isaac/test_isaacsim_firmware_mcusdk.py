"""Unit tests for IsaacSimFirmwareMCUSDK (Task 5 skeleton)."""

import time

import numpy as np
import pytest
import torch

from hal.client.data_structures.hardware import JointCommand
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
