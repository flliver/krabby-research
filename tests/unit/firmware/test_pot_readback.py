"""Pot analogRead sync from articulation joint state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from firmware.sim.pot_position import joint_angle_to_pot_adc
from firmware.sim.pot_readback import sync_pot_analog_reads
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


class _TensorCell:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def _articulation_stub(num_joints: int, joint_index: int, position: float, lower: float, upper: float):
    joint_pos_row = [_TensorCell(0.0) for _ in range(num_joints)]
    joint_pos_row[joint_index] = _TensorCell(position)
    limits_row = [[_TensorCell(-1.0), _TensorCell(1.0)] for _ in range(num_joints)]
    limits_row[joint_index] = [_TensorCell(lower), _TensorCell(upper)]
    return SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=[joint_pos_row],
            soft_joint_pos_limits=[limits_row],
        )
    )


def test_sync_pot_analog_reads_writes_expected_adc():
    bridge = MagicMock()
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhy = actuators[0]
    articulation = _articulation_stub(len(indices), flhy.joint_index, 0.0, -1.0, 1.0)

    sync_pot_analog_reads(bridge, actuators, articulation)

    expected = joint_angle_to_pot_adc(0.0, -1.0, 1.0, band=flhy.pot_band)
    bridge.set_analog_read.assert_any_call(flhy.slot.pin_pot, expected)
