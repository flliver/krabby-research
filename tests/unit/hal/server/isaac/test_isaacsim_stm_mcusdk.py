"""IsaacSimStmMCUSDK duty feedback (STM CAN fullsim)."""

import time
from unittest.mock import MagicMock

import numpy as np
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.isaacsim_stm_mcusdk import IsaacSimStmMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def _make_command(joint_names: tuple[str, ...], values: list[float]) -> JointCommand:
    return JointCommand(
        _joint_positions=np.array(values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=joint_names,
    )


def test_apply_command_runs_can_loop_and_returns_duties():
    loop = MagicMock()
    loop.duty_for_hal_joint.side_effect = lambda name: 0.25 if name.endswith("knee") else 0.0

    robot = KRABBY_HEX_DEFINITION
    sdk = IsaacSimStmMCUSDK.from_robot_definition(
        robot,
        device="cpu",
        firmware_loop=loop,
    )
    names = robot.get_joint_names()
    command = _make_command(names, [0.5] * len(names))
    action = sdk.apply_command(command, num_envs=1)

    loop.apply_hal_command.assert_called_once_with(command)
    loop.tick_firmware.assert_called_once()
    assert action.shape == (1, robot.get_total_joint_count())
    knee_idx = names.index("FL_knee")
    assert action[0, knee_idx].item() == 0.25


def test_passthrough_without_firmware_loop():
    robot = KRABBY_HEX_DEFINITION
    sdk = IsaacSimStmMCUSDK.from_robot_definition(robot, device="cpu")
    names = robot.get_joint_names()
    command = _make_command(names, [0.9] * len(names))
    action = sdk.apply_command(command, num_envs=1)
    assert isinstance(action, torch.Tensor)
    assert action.shape[0] == 1