"""Pot ``analogRead`` tracks Isaac joint angle through ``RobotCore``."""

from __future__ import annotations

import pytest

from firmware.sim.pot_position import joint_angle_to_pot_adc
from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"


def _expected_adc(robot, joint_index: int) -> int:
    limits = robot.data.soft_joint_pos_limits[0, joint_index]
    position = robot.data.joint_pos[0, joint_index].item()
    return joint_angle_to_pot_adc(
        position,
        limits[0].item(),
        limits[1].item(),
    )


def test_pot_adc_matches_joint_at_step(front_board_rig):
    core, robot, config, indices = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)
    joint_index = indices[slot.isaac_joint_name]

    park_at_default_pose(core, robot)
    core.step()

    expected = _expected_adc(robot, joint_index)
    actual = core.bridge.analog_read(slot.pin_pot)
    assert abs(actual - expected) <= 1


def test_pot_adc_changes_when_joint_moves(front_board_rig):
    core, robot, config, indices = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)
    joint_index = indices[slot.isaac_joint_name]

    park_at_default_pose(core, robot)
    core.step()
    adc_before = core.bridge.analog_read(slot.pin_pot)

    core.jog(SLOT, 255)
    for _ in range(40):
        core.step()

    expected = _expected_adc(robot, joint_index)
    actual = core.bridge.analog_read(slot.pin_pot)
    # Pot is synced at the start of each step from the prior physics state; allow one step of lag.
    assert abs(actual - expected) <= 12
    assert actual > adc_before
