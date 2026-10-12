"""Hall edge sync from articulation joint state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from firmware.sim.hall_readback import sync_hall_edge_counts
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


def test_sync_hall_edge_counts_calls_host_per_slot():
    bridge = MagicMock()
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    actuators = config.actuators(indices)
    flhl = actuators[1]
    articulation = _articulation_stub(len(indices), flhl.joint_index, 0.25, -0.5, 0.75)

    sync_hall_edge_counts(bridge, actuators, articulation)

    bridge.sync_hall_joint.assert_any_call(1, 0.25, -0.5, 0.75)
