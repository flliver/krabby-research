"""IsaacSimFirmwareMCUSDK: JointCommand → firmware → pins → action → joint motion."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_loop_bindings import RobotCoreCommandLoop
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"
HAL_JOINT = "FL_hip_pitch"
DRIVE_STEPS = 40


def _make_hex_command(joint_values: list[float]) -> JointCommand:
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    return JointCommand(
        _joint_positions=np.array(joint_values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )


@pytest.fixture(scope="module")
def hal_firmware_sdk(front_board_rig):
    from firmware.sim.isaac_scene import configure_open_loop_velocity_drive
    from firmware.sim.robot_core import RobotCore

    core, robot, config, indices = front_board_rig
    configure_open_loop_velocity_drive(robot, list(indices.values()))
    loop_core = RobotCore.front_board(
        env=core.env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=core.bridge,
        config=config,
        max_joint_speed=0.5,
    )
    sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(
        KRABBY_HEX_DEFINITION,
        device=robot.device,
        firmware_loop=RobotCoreCommandLoop(loop_core),
    )
    return sdk, loop_core, robot, config, indices, core.env


def test_command_updates_pin_state(hal_firmware_sdk):
    sdk, loop_core, _robot, config, _indices, _env = hal_firmware_sdk
    slot = config.slot_by_firmware_id(SLOT)
    pwm_r, pwm_l, pin_en = slot.drive_pins

    values = [0.0] * 18
    values[KRABBY_HEX_DEFINITION.get_joint_names().index(HAL_JOINT)] = 0.25
    sdk.apply_command(_make_hex_command(values))

    assert loop_core.pin_state.is_digital_high(pin_en)
    assert loop_core.pin_state.signed_pwm_command(pwm_r, pwm_l, pin_en) > 0


def test_command_action_moves_joint(hal_firmware_sdk):
    sdk, loop_core, robot, config, indices, env = hal_firmware_sdk
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]

    park_at_default_pose(loop_core, robot)
    before = robot.data.joint_pos[0, joint_index].item()

    values = [0.0] * 18
    values[KRABBY_HEX_DEFINITION.get_joint_names().index(HAL_JOINT)] = 0.25
    action = sdk.apply_command(_make_hex_command(values))

    with torch.inference_mode():
        for _ in range(DRIVE_STEPS):
            env.step(action)

    after = robot.data.joint_pos[0, joint_index].item()
    assert after - before > 0.03
