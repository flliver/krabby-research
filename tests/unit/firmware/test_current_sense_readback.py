"""Current-sense analogRead sync from wrench + EN + joint velocity."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from firmware.sim.current_sense_readback import sync_current_sense_analog_reads
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


class _TensorCell:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def test_sync_current_sense_sets_pin_is_from_wrench_and_en():
    bridge = MagicMock()
    bridge.pin_digital.return_value = 1
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhl = actuators[1]
    row = [_TensorCell(0.0)] * 6
    row[2] = _TensorCell(150.0)
    joint_vel_row = [_TensorCell(0.0) for _ in range(len(indices))]
    joint_vel_row[flhl.joint_index] = _TensorCell(0.0)
    articulation = SimpleNamespace(
        data=SimpleNamespace(
            body_incoming_joint_wrench_b=[[row]],
            joint_vel=[joint_vel_row],
        )
    )
    sync_current_sense_analog_reads(
        bridge,
        actuators,
        articulation,
        {"FLHL": 0},
    )
    bridge.set_analog_read.assert_any_call(flhl.slot.pin_is, 30)
