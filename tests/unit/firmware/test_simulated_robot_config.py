"""FRONT board REV 3 pin and Isaac joint mapping."""

from dataclasses import replace

import pytest

from firmware.sim.pin_state import PinStateModel
from firmware.sim.hall_encoder import DEFAULT_HALL_EDGES_PER_FULL_TRAVEL
from firmware.sim.simulated_robot_config import (
    DEFAULT_HALL_EDGES_PER_STROKE,
    DEFAULT_HALL_HIP_EDGES_PER_STROKE,
    PIN_REVISION_LABEL,
    SENSOR_HALL,
    SENSOR_POT,
    SimulatedRobotConfig,
    board_id_for_firmware_id,
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
    "FL_Body_CamShaft_RevoluteJoint",
    "FL_Hip_Femur_RevoluteJoint",
    "FL_Femur_Tibia_RevoluteJoint",
    "FR_Body_CamShaft_RevoluteJoint",
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
    hy_names = [
        s.isaac_joint_name
        for s in config.slots
        if s.firmware_actuator_id.endswith("HY")
    ]
    assert hy_names == [
        "FL_Body_CamShaft_RevoluteJoint",
        "FR_Body_CamShaft_RevoluteJoint",
    ]


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

    assert flhy.joint_index == indices["FL_Body_CamShaft_RevoluteJoint"]
    assert flhy.slot.isaac_joint_name == "FL_Body_CamShaft_RevoluteJoint"
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


def test_default_board_is_pot_without_quadrature():
    config = SimulatedRobotConfig.front_board_rev3()
    assert config.quadrature is False
    assert all(slot.sensor == SENSOR_POT for slot in config.slots)
    assert SimulatedRobotConfig.rev2_pot() == config


def test_rev3_hall_is_all_hall_on_a_quadrature_board():
    config = SimulatedRobotConfig.rev3_hall()
    assert config.quadrature is True
    assert all(slot.is_hall for slot in config.slots)
    strokes = {slot.firmware_actuator_id: slot.hall_edges_per_stroke for slot in config.slots}
    assert strokes == {
        "FLHY": DEFAULT_HALL_EDGES_PER_STROKE,
        "FLHL": DEFAULT_HALL_HIP_EDGES_PER_STROKE,
        "FLKL": DEFAULT_HALL_EDGES_PER_STROKE,
        "FRHY": DEFAULT_HALL_EDGES_PER_STROKE,
        "FRHL": DEFAULT_HALL_HIP_EDGES_PER_STROKE,
        "FRKL": DEFAULT_HALL_EDGES_PER_STROKE,
    }
    # Pins and names are unchanged.
    assert tuple(s.drive_pins for s in config.slots) == _EXPECTED_DRIVE_PINS
    assert config.firmware_actuator_ids() == _EXPECTED_FIRMWARE_IDS


def test_default_hall_strokes_follow_the_robot():
    # Knees: FLKL measured 776 edges, inside CAL_HALL_MIN/MAX_STROKE_EDGES (450..1500).
    assert DEFAULT_HALL_EDGES_PER_STROKE == 776
    assert 450 <= DEFAULT_HALL_EDGES_PER_STROKE <= 1500
    # Hips: longer than the 941-1003 edges FLHL reached before its curled tibia met the
    # body, so the fixed 700-edge sweep (CAL_HIP_SWEEP_EDGES) never reaches the end.
    assert DEFAULT_HALL_HIP_EDGES_PER_STROKE > 1003
    # Independent of the Isaac radians-to-edges scale.
    assert DEFAULT_HALL_EDGES_PER_FULL_TRAVEL == 3200


def test_one_stroke_for_every_slot_when_given():
    config = SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=900)
    assert {slot.hall_edges_per_stroke for slot in config.slots} == {900}


def test_mixed_has_a_hall_leg_and_a_pot_leg():
    config = SimulatedRobotConfig.mixed(hall_edges_per_stroke=1200)
    assert config.quadrature is True
    sensors = {slot.firmware_actuator_id: slot.sensor for slot in config.slots}
    assert sensors == {
        "FLHY": SENSOR_HALL,
        "FLHL": SENSOR_HALL,
        "FLKL": SENSOR_HALL,
        "FRHY": SENSOR_POT,
        "FRHL": SENSOR_POT,
        "FRKL": SENSOR_POT,
    }
    assert [s.firmware_actuator_id for s in config.hall_slots()] == ["FLHY", "FLHL", "FLKL"]
    assert [s.firmware_actuator_id for s in config.pot_slots()] == ["FRHY", "FRHL", "FRKL"]
    assert config.slot_by_firmware_id("FLKL").hall_edges_per_stroke == 1200


@pytest.mark.parametrize(
    ("role", "board_id", "first_id"),
    [("FRONT", 0, "FLHY"), ("LEFT", 1, "RLHY"), ("RIGHT", 2, "RRHY")],
)
def test_factories_cover_every_board(role, board_id, first_id):
    for factory in (
        SimulatedRobotConfig.rev2_pot,
        SimulatedRobotConfig.rev3_hall,
        SimulatedRobotConfig.mixed,
    ):
        config = factory(board_role=role)
        assert config.board_role == role
        assert config.board_id == board_id
        assert config.slots[0].firmware_actuator_id == first_id
        for slot in config.slots:
            assert board_id_for_firmware_id(slot.firmware_actuator_id) == board_id


def test_board_id_for_unknown_actuator_raises():
    with pytest.raises(KeyError):
        board_id_for_firmware_id("XXHY")


def test_slot_rejects_unknown_sensor_and_bad_stroke():
    slot = SimulatedRobotConfig.front_board_rev3().slots[0]
    with pytest.raises(ValueError):
        replace(slot, sensor="laser")
    with pytest.raises(ValueError):
        replace(slot, hall_edges_per_stroke=0)
