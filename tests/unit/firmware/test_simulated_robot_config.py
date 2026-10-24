"""FRONT board REV 3 pin and Isaac joint mapping."""

import pytest

from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_robot_config import (
    PIN_REVISION_LABEL,
    SimulatedRobotConfig,
)

# REV 3 drive pins (board_pins.h / ACT_LIST_FRONT)
_EXPECTED_DRIVE_PINS = (
    (2, 3, 22),
    (4, 5, 24),
    (6, 7, 26),
    (8, 9, 23),
    (10, 11, 25),
    (12, 13, 27),
)

_EXPECTED_FIRMWARE_IDS = ("FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL")

_EXPECTED_ISAAC_JOINTS = (
    "FL_Body_Hip_RevoluteJoint",
    "FL_Hip_Femur_RevoluteJoint",
    "FL_Femur_Tibia_RevoluteJoint",
    "FR_Body_Hip_RevoluteJoint",
    "FR_Hip_Femur_RevoluteJoint",
    "FR_Femur_Tibia_RevoluteJoint",
)


def test_front_board_rev3_has_six_slots_in_act_list_order():
    config = SimulatedRobotConfig.front_board_rev3()
    assert config.pin_revision_label == PIN_REVISION_LABEL
    assert config.board_role == "FRONT"
    assert len(config.slots) == 6
    assert [s.slot_index for s in config.slots] == list(range(6))


def test_drive_pins_match_board_pins_rev3():
    config = SimulatedRobotConfig.front_board_rev3()
    assert tuple(s.drive_pins for s in config.slots) == _EXPECTED_DRIVE_PINS


def test_firmware_ids_match_act_list_front():
    config = SimulatedRobotConfig.front_board_rev3()
    assert tuple(s.firmware_actuator_id for s in config.slots) == _EXPECTED_FIRMWARE_IDS


def test_isaac_joints_are_six_distinct_fl_fr_dof():
    config = SimulatedRobotConfig.front_board_rev3()
    config.validate_distinct_joints()
    assert config.isaac_joint_names() == _EXPECTED_ISAAC_JOINTS
    assert all("CamShaft" not in name for name in config.isaac_joint_names())


@pytest.mark.parametrize(
    ("slot_index", "firmware_id", "isaac_joint"),
    list(zip(range(6), _EXPECTED_FIRMWARE_IDS, _EXPECTED_ISAAC_JOINTS)),
)
def test_slot_lookup_by_index_and_firmware_id(slot_index, firmware_id, isaac_joint):
    config = SimulatedRobotConfig.front_board_rev3()
    by_index = config.slot_by_index(slot_index)
    by_id = config.slot_by_firmware_id(firmware_id)
    assert by_index is by_id
    assert by_index.isaac_joint_name == isaac_joint


def _joint_indices_for_config(config: SimulatedRobotConfig) -> dict[str, int]:
    return {name: index for index, name in enumerate(config.isaac_joint_names())}


def test_actuators_yields_six_with_distinct_drive_pins():
    config = SimulatedRobotConfig.front_board_rev3()
    indices = _joint_indices_for_config(config)
    actuators = config.actuators(indices)

    assert len(actuators) == 6
    config.validate_distinct_drive_pins()
    assert len({a.slot.drive_pins for a in actuators}) == 6
    assert tuple(a.slot.firmware_actuator_id for a in actuators) == _EXPECTED_FIRMWARE_IDS


def test_actuator_by_firmware_id_wires_isaac_joint_index():
    config = SimulatedRobotConfig.front_board_rev3()
    indices = _joint_indices_for_config(config)
    flhy = config.actuator_by_firmware_id("FLHY", indices)

    assert flhy.joint_index == indices["FL_Body_Hip_RevoluteJoint"]
    assert flhy.slot.isaac_joint_name == "FL_Body_Hip_RevoluteJoint"
    assert flhy.slot.pin_pot == 0
    assert flhy.slot.pin_is == 6


@pytest.mark.parametrize("firmware_id", _EXPECTED_FIRMWARE_IDS)
def test_drive_via_actuator_name_not_literal_pins(firmware_id):
    config = SimulatedRobotConfig.front_board_rev3()
    indices = _joint_indices_for_config(config)
    actuator = config.actuator_by_firmware_id(firmware_id, indices)
    pin_state = PinStateModel()

    pwm_r, pwm_l, pin_en = actuator.slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_r, 50)
    pin_state.analog_write(pwm_l, 0)

    assert actuator.signed_pwm_command(pin_state) == 50
