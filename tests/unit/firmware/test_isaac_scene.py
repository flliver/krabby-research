"""Unit tests for firmware Isaac scene helpers (no AppLauncher)."""

from unittest.mock import MagicMock

import pytest

from firmware.sim.isaac_scene import (
    DEFAULT_CRAB_HEX_PLAY_TASK,
    _crab_hex_articulation_uses_ref_joint_names,
    configure_open_loop_velocity_drive,
    resolve_eighteen_joint_indices,
    resolve_front_board_joint_indices,
    resolve_joint_indices_by_name,
)
from firmware.sim.simulated_robot_config import (
    SimulatedRobotConfig,
    ThreeBoardSimConfigs,
    isaac_joint_name_for_firmware_id_crab_hex_ref,
)


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


def test_crab_hex_ref_profile_detected_by_lf_hip_yaw_probe():
    """``assets/crab_hex_ref.usd`` — no ``FL_Body_CamShaft_RevoluteJoint``, has ``lf_hip_yaw``."""
    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            raise ValueError("missing on ref USD")
        if names == ["lf_hip_yaw"]:
            return ([0], ["lf_hip_yaw"])
        raise ValueError(names)

    robot.find_joints.side_effect = find_joints
    assert _crab_hex_articulation_uses_ref_joint_names(robot) is True


def test_crab_hex_plant_profile_detected_by_cam_shaft_probe():
    """Main plant ``assets/crab.usda`` — ``FL_Body_CamShaft_RevoluteJoint`` even though ``Body_Hip`` exists."""
    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            return ([4], ["FL_Body_CamShaft_RevoluteJoint"])
        if names == ["lf_hip_yaw"]:
            raise ValueError("missing on plant USD")
        if names == ["FL_Body_Hip_RevoluteJoint"]:
            return ([3], ["FL_Body_Hip_RevoluteJoint"])
        raise ValueError(names)

    robot.find_joints.side_effect = find_joints
    assert _crab_hex_articulation_uses_ref_joint_names(robot) is False


def test_resolve_eighteen_joint_indices_plant_includes_cam_shaft_yaw():
    configs = ThreeBoardSimConfigs.rev3()
    plant_names = [slot.isaac_joint_name for slot in configs.all_slots()]
    assert "FL_Body_CamShaft_RevoluteJoint" in plant_names
    assert "FR_Body_CamShaft_RevoluteJoint" in plant_names

    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            return ([0], ["FL_Body_CamShaft_RevoluteJoint"])
        if names == ["lf_hip_yaw"]:
            raise ValueError("plant")
        return (
            list(range(len(names))),
            list(names),
        )

    robot.find_joints.side_effect = find_joints
    indices = resolve_eighteen_joint_indices(robot, configs)
    assert indices["FL_Body_CamShaft_RevoluteJoint"] == plant_names.index(
        "FL_Body_CamShaft_RevoluteJoint"
    )
    assert len(indices) == 18


def test_resolve_eighteen_joint_indices_ref_aliases_plant_slot_keys():
    configs = ThreeBoardSimConfigs.rev3()
    ref_names = [
        isaac_joint_name_for_firmware_id_crab_hex_ref(slot.firmware_actuator_id)
        for slot in configs.all_slots()
    ]

    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            raise ValueError("ref")
        if names == ["lf_hip_yaw"]:
            return ([0], ["lf_hip_yaw"])
        return (list(range(len(names))), list(names))

    robot.find_joints.side_effect = find_joints
    indices = resolve_eighteen_joint_indices(robot, configs)
    flhy_plant = configs.front.slot_by_firmware_id("FLHY").isaac_joint_name
    flhy_ref = isaac_joint_name_for_firmware_id_crab_hex_ref("FLHY")
    assert indices[flhy_plant] == indices[flhy_ref]
    assert flhy_ref == "lf_hip_yaw"
    assert len(ref_names) == 18


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


def test_front_board_open_loop_joint_names_include_cam_shaft_yaw():
    """``front_board_rig`` passes these six names to ``configure_open_loop_velocity_drive``."""
    config = SimulatedRobotConfig.front_board_rev3()
    names = set(config.isaac_joint_names())
    assert "FL_Body_CamShaft_RevoluteJoint" in names
    assert "FR_Body_CamShaft_RevoluteJoint" in names
    assert not any("Body_Hip" in n for n in names)


def test_eighteen_joint_open_loop_indices_include_six_cam_shafts():
    """Plant ``crab.usda`` rig: all 18 firmware DOFs including ``*HY`` cam shafts."""
    configs = ThreeBoardSimConfigs.rev3()
    hy_joint_names = [
        slot.isaac_joint_name
        for slot in configs.all_slots()
        if slot.firmware_actuator_id.endswith("HY")
    ]
    assert len(hy_joint_names) == 6
    assert all("Body_CamShaft" in name for name in hy_joint_names)

    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            return ([0], ["FL_Body_CamShaft_RevoluteJoint"])
        if names == ["lf_hip_yaw"]:
            raise ValueError("plant")
        return (list(range(len(names))), list(names))

    robot.find_joints.side_effect = find_joints
    indices = resolve_eighteen_joint_indices(robot, configs)
    for name in hy_joint_names:
        assert name in indices
    assert len(set(indices.values())) == 18


def test_configure_open_loop_velocity_drive_zeros_stiffness_on_every_index():
    robot = MagicMock()
    robot.actuators = {}
    joint_ids = list(range(18))

    configure_open_loop_velocity_drive(robot, joint_ids)

    robot.write_joint_stiffness_to_sim.assert_called_once_with(0.0, joint_ids=joint_ids)
