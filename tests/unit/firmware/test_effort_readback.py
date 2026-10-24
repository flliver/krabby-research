"""Applied joint torque readback for current-sense cross-check."""

from types import SimpleNamespace

from firmware.sim.effort_readback import read_applied_joint_torque_nm


class _TensorCell:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def test_read_applied_torque_from_articulation_data():
    row = [_TensorCell(0.0), _TensorCell(12.5), _TensorCell(-3.0)]
    robot = SimpleNamespace(
        data=SimpleNamespace(applied_torque=[row]),
    )
    assert read_applied_joint_torque_nm(robot, 1, env_index=0) == 12.5


def test_read_applied_torque_falls_back_to_actuator_applied_effort():
    effort_row = [_TensorCell(1.0), _TensorCell(40.0)]
    actuator = SimpleNamespace(
        applied_effort=[effort_row],
        joint_indices=(5, 7),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(applied_torque=None),
        actuators={"leg": actuator},
    )
    assert read_applied_joint_torque_nm(robot, 7, env_index=0) == 40.0
