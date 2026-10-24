"""Wrench body naming and tensor read helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from firmware.sim.wrench_readback import (
    read_body_incoming_wrench,
    resolve_wrench_body_indices,
    sample_actuator_wrenches,
    wrench_body_name_for_joint,
)


class _TensorCell:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def test_wrench_body_names_match_crab_usd_child_links():
    assert wrench_body_name_for_joint("FL_Body_Hip_RevoluteJoint") == "FL_Hip"
    assert wrench_body_name_for_joint("FL_Hip_Femur_RevoluteJoint") == "FL_Femur"
    assert wrench_body_name_for_joint("FR_Femur_Tibia_RevoluteJoint") == "FR_Tibia"


def test_resolve_wrench_body_indices_uses_find_bodies():
    config = SimulatedRobotConfig.front_board_rev3()
    robot = MagicMock()
    robot.find_bodies.return_value = ([0, 1, 2, 3, 4, 5], [f"b{i}" for i in range(6)])

    indices = resolve_wrench_body_indices(robot, config)

    assert len(indices) == 6
    assert indices["FLHY"] == 0
    assert indices["FRKL"] == 5
    expected_names = [
        "FL_Hip",
        "FL_Femur",
        "FL_Tibia",
        "FR_Hip",
        "FR_Femur",
        "FR_Tibia",
    ]
    robot.find_bodies.assert_called_once_with(expected_names, preserve_order=True)


def test_read_body_incoming_wrench_unpacks_six_components():
    row = [_TensorCell(float(i)) for i in (10, 20, 30, 0.1, 0.2, 0.3)]
    wrench = [[row]]
    robot = SimpleNamespace(
        data=SimpleNamespace(body_incoming_joint_wrench_b=wrench),
    )
    sample = read_body_incoming_wrench(robot, 0)
    assert sample.fx == 10.0 and sample.fz == 30.0
    assert sample.force_magnitude() == pytest.approx(37.4165738677, rel=1e-4)


def test_sample_actuator_wrenches_per_slot():
    row = [
        _TensorCell(0.0),
        _TensorCell(0.0),
        _TensorCell(50.0),
        _TensorCell(0.0),
        _TensorCell(0.0),
        _TensorCell(0.0),
    ]
    pad = [[_TensorCell(0.0)] * 6] * 2
    robot = SimpleNamespace(
        data=SimpleNamespace(body_incoming_joint_wrench_b=[pad + [row]]),
    )
    samples = sample_actuator_wrenches(robot, {"FLKL": 2})
    assert samples["FLKL"].fz == 50.0
