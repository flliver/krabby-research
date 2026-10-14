"""Unit tests for firmware Isaac scene helpers (no AppLauncher)."""

from unittest.mock import MagicMock

import pytest

from firmware.sim.isaac_scene import (
    DEFAULT_CRAB_HEX_PLAY_TASK,
    resolve_front_board_joint_indices,
    resolve_joint_indices_by_name,
)
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def test_default_play_task_id():
    assert DEFAULT_CRAB_HEX_PLAY_TASK == "Isaac-Crab-Hex-Flat-Walk-Play-v0"


def test_resolve_joint_indices_by_name_preserves_order():
    names = (
        "FL_Body_Hip_RevoluteJoint",
        "FR_Femur_Tibia_RevoluteJoint",
    )
    robot = MagicMock()
    robot.find_joints.return_value = ([3, 11], list(names))

    out = resolve_joint_indices_by_name(robot, names)
    assert out == {"FL_Body_Hip_RevoluteJoint": 3, "FR_Femur_Tibia_RevoluteJoint": 11}
    robot.find_joints.assert_called_once_with(list(names), preserve_order=True)


def test_resolve_joint_indices_by_name_raises_when_count_mismatch():
    robot = MagicMock()
    robot.find_joints.return_value = ([0], ["FL_Body_Hip_RevoluteJoint"])

    with pytest.raises(KeyError, match="expected 2 joints"):
        resolve_joint_indices_by_name(
            robot,
            ("FL_Body_Hip_RevoluteJoint", "FR_Body_Hip_RevoluteJoint"),
        )


def test_resolve_front_board_joint_indices_uses_config_names():
    config = SimulatedRobotConfig.front_board_rev3()
    robot = MagicMock()

    def _find(names, preserve_order=True):
        assert preserve_order is True
        indices = list(range(len(names)))
        return (indices, list(names))

    robot.find_joints.side_effect = _find
    out = resolve_front_board_joint_indices(robot, config)
    assert set(out) == set(config.isaac_joint_names())
    assert len(out) == 6
