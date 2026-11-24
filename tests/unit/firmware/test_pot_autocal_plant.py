"""Unit tests for the auto-cal travel-fraction plant (pot and hall actuators)."""

from __future__ import annotations

import pytest

from firmware.sim.pot_autocal_plant import (
    AUTOCAL_PWM_DRIVE_THRESHOLD,
    AUTOCAL_QUADRATURE_TRAVEL_STEP,
    AUTOCAL_TRAVEL_STEP,
    CAL_PLANT_IS_RUN_ADC,
    CAL_PLANT_IS_STALL_ADC,
    CAL_PLANT_IS_SWITCH_OPEN_ADC,
    HALL_B_REST_HIGH_ADC,
    HALL_B_REST_LOW_ADC,
    INT32_MAX,
    INT32_MIN,
    ROBOT_STORED_KNEE,
    CalibrationTravelPlant,
    StoredKneeModel,
    advance_travel,
    advance_travel_from_pwm,
    calibration_is_adc,
    default_travel_fractions,
    hall_b_level,
    hall_count_delta,
    saturating_count,
    sync_pot_reads_from_travel,
    travel_step_for_config,
)
from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def _front_actuators(config=None):
    config = config or SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    return config, config.actuators(indices, max_joint_speed=1.0)


class FakeBoard:
    """Pins and per-board hall counts, as the host bridge exposes them."""

    def __init__(self) -> None:
        self.pots: dict[int, int] = {}
        self.counts: dict[tuple[int, int], int] = {}
        self.pwm: dict[int, int] = {}
        self.digital: dict[int, int] = {}

    def set_analog_read(self, pin: int, value: int) -> None:
        self.pots[pin] = value

    def board_hall_edge_count(self, board_id: int, slot: int) -> int:
        return self.counts.get((board_id, slot), 0)

    def board_set_hall_edge_count(self, board_id: int, slot: int, count: int) -> None:
        self.counts[(board_id, slot)] = count

    def pin_pwm(self, pin: int) -> int:
        return self.pwm.get(pin, 0)

    def pin_digital(self, pin: int) -> int:
        return self.digital.get(pin, 0)

    def drive(self, slot, signed_pwm: int) -> None:
        pwm_r, pwm_l, pin_en = slot.drive_pins
        self.digital[pin_en] = 1 if signed_pwm else 0
        self.pwm[pwm_r] = max(signed_pwm, 0)
        self.pwm[pwm_l] = max(-signed_pwm, 0)


def test_travel_advances_with_signed_pwm():
    config, actuators = _front_actuators()
    travel = default_travel_fractions(actuators)
    slot = config.slot_by_firmware_id("FLHL")
    pin_state = PinStateModel()
    pwm_r, pwm_l, pin_en = slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_r, 200)
    pin_state.analog_write(pwm_l, 0)

    flhl = next(a for a in actuators if a.slot.firmware_actuator_id == "FLHL")
    advance_travel_from_pwm([flhl], pin_state, travel)
    assert travel["FLHL"] > 0.5


def test_sync_pot_reads_from_travel_writes_band_adc():
    config, actuators = _front_actuators()

    class Bridge:
        def __init__(self) -> None:
            self.pots: dict[int, int] = {}

        def set_analog_read(self, pin: int, value: int) -> None:
            self.pots[pin] = value

    bridge = Bridge()
    band = config.pot_band_for_slot(0)
    flhy = actuators[0]
    sync_pot_reads_from_travel(bridge, [flhy], {"FLHY": 0.0})
    assert bridge.pots[flhy.slot.pin_pot] == band.min_adc


def test_travel_moves_only_at_the_drive_threshold_and_stops_at_the_ends():
    assert advance_travel(0.5, AUTOCAL_PWM_DRIVE_THRESHOLD - 1, 0.1) == 0.5
    assert advance_travel(0.5, -(AUTOCAL_PWM_DRIVE_THRESHOLD - 1), 0.1) == 0.5
    assert advance_travel(0.5, AUTOCAL_PWM_DRIVE_THRESHOLD, 0.1) == pytest.approx(0.6)
    assert advance_travel(0.5, -AUTOCAL_PWM_DRIVE_THRESHOLD, 0.1) == pytest.approx(0.4)
    assert advance_travel(0.95, 255, 0.1) == 1.0
    assert advance_travel(0.05, -255, 0.1) == 0.0


def test_step_is_slower_on_quadrature_boards_only():
    assert travel_step_for_config(None) == AUTOCAL_TRAVEL_STEP
    assert travel_step_for_config(SimulatedRobotConfig.rev2_pot()) == AUTOCAL_TRAVEL_STEP
    assert travel_step_for_config(SimulatedRobotConfig.rev3_hall()) == (
        AUTOCAL_QUADRATURE_TRAVEL_STEP
    )
    assert travel_step_for_config(SimulatedRobotConfig.mixed()) == AUTOCAL_QUADRATURE_TRAVEL_STEP


def test_extending_counts_down_retracting_counts_up_and_ends_add_no_edges():
    assert hall_count_delta(0.5, 0.75, 1000) == -250
    assert hall_count_delta(0.75, 0.5, 1000) == 250
    assert hall_count_delta(1.0, 1.0, 1000) == 0
    # Travel past an end is clamped: no edges beyond the stroke.
    assert hall_count_delta(0.9, 1.5, 1000) == -100
    assert hall_count_delta(0.1, -0.5, 1000) == 100


def test_counts_saturate_at_the_int32_range():
    assert saturating_count(INT32_MAX - 1, 5) == INT32_MAX
    assert saturating_count(INT32_MIN + 1, -5) == INT32_MIN
    assert saturating_count(10, -3) == 7


def test_hall_b_rests_on_a_rail_and_flips_with_each_edge():
    assert hall_b_level(0) == HALL_B_REST_LOW_ADC
    assert hall_b_level(1) == HALL_B_REST_HIGH_ADC
    assert hall_b_level(-1) == HALL_B_REST_HIGH_ADC
    assert hall_b_level(-2) == HALL_B_REST_LOW_ADC
    # Outside the firmware's pot band (32..990), so never read as a pot.
    assert not 32 <= HALL_B_REST_LOW_ADC <= 990
    assert not 32 <= HALL_B_REST_HIGH_ADC <= 990


def test_plant_counts_signed_edges_per_board_and_writes_hall_b():
    config = SimulatedRobotConfig.rev3_hall(board_role="LEFT", hall_edges_per_stroke=1000)
    plant = CalibrationTravelPlant(config, step=0.01)
    board = FakeBoard()
    hip = config.slot_by_firmware_id("RLHL")

    board.drive(hip, 80)  # +PWM extends
    for _ in range(10):
        plant.advance(board)
    assert plant.travel["RLHL"] == pytest.approx(0.6)
    assert board.counts[(1, hip.slot_index)] == -100
    assert (0, hip.slot_index) not in board.counts

    board.drive(hip, -80)  # retract counts back up
    for _ in range(5):
        plant.advance(board)
    assert board.counts[(1, hip.slot_index)] == -50

    plant.sync_inputs(board)
    assert board.pots[hip.pin_pot] == HALL_B_REST_LOW_ADC
    board.counts[(1, hip.slot_index)] = -51
    plant.sync_inputs(board)
    assert board.pots[hip.pin_pot] == HALL_B_REST_HIGH_ADC


def test_plant_edges_stop_at_the_stroke_end():
    config = SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=1000)
    plant = CalibrationTravelPlant(config, travel={"FLKL": 0.9}, step=0.05)
    board = FakeBoard()
    knee = config.slot_by_firmware_id("FLKL")
    board.drive(knee, 120)
    for _ in range(10):
        plant.advance(board)
    assert plant.travel["FLKL"] == 1.0
    assert board.counts[(0, knee.slot_index)] == -100


def test_plant_keeps_a_firmware_reset_of_the_counts():
    """Read-modify-write: a hallHwInit() that zeroed the board stays in force."""
    config = SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=1000)
    plant = CalibrationTravelPlant(config, step=0.01)
    board = FakeBoard()
    knee = config.slot_by_firmware_id("FRKL")
    board.drive(knee, -60)
    plant.advance(board)
    board.counts[(0, knee.slot_index)] = 0
    plant.advance(board)
    assert board.counts[(0, knee.slot_index)] == 10


def test_pot_slots_of_a_mixed_board_never_count():
    config = SimulatedRobotConfig.mixed()
    plant = CalibrationTravelPlant(config)
    board = FakeBoard()
    pot_knee = config.slot_by_firmware_id("FRKL")
    board.drive(pot_knee, 100)
    for _ in range(20):
        plant.advance(board)
    assert board.counts == {}
    plant.sync_inputs(board)
    band = config.pot_band_for_slot(pot_knee.slot_index)
    expected = round(band.min_adc + plant.travel["FRKL"] * (band.max_adc - band.min_adc))
    assert board.pots[pot_knee.pin_pot] == expected


def test_advance_travel_from_pwm_counts_hall_edges_with_a_bridge():
    config, actuators = _front_actuators(SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=1000))
    flhl = next(a for a in actuators if a.slot.firmware_actuator_id == "FLHL")
    pin_state = PinStateModel()
    pwm_r, _pwm_l, pin_en = flhl.slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_r, 100)
    travel = default_travel_fractions(actuators)
    board = FakeBoard()

    advance_travel_from_pwm([flhl], pin_state, travel, step=0.02, bridge=board, board_id=2)
    assert board.counts == {(2, flhl.slot.slot_index): -20}

    sync_pot_reads_from_travel(board, [flhl], travel, board_id=2)
    assert board.pots[flhl.slot.pin_pot] == HALL_B_REST_LOW_ADC


def test_current_sense_levels_follow_the_end_currents_of_2026_11_24():
    # 0 undriven; running; a retracted end is the limit switch (current drops);
    # an extended end that stops the rod raises the current.
    assert (CAL_PLANT_IS_SWITCH_OPEN_ADC, CAL_PLANT_IS_RUN_ADC, CAL_PLANT_IS_STALL_ADC) == (8, 12, 18)
    assert calibration_is_adc(0, 0.5, 0.5) == 0
    assert calibration_is_adc(128, 0.5, 0.6) == CAL_PLANT_IS_RUN_ADC
    assert calibration_is_adc(-128, 0.5, 0.4) == CAL_PLANT_IS_RUN_ADC
    assert calibration_is_adc(-128, 0.01, 0.0) == CAL_PLANT_IS_SWITCH_OPEN_ADC
    assert calibration_is_adc(-128, 0.0, 0.0) == CAL_PLANT_IS_SWITCH_OPEN_ADC
    assert calibration_is_adc(128, 1.0, 1.0) == CAL_PLANT_IS_STALL_ADC
    assert calibration_is_adc(128, 0.7, 0.7) == CAL_PLANT_IS_STALL_ADC  # an obstacle
    # Below the drive threshold the rod does not move: driven, not stalled.
    assert calibration_is_adc(AUTOCAL_PWM_DRIVE_THRESHOLD - 1, 0.5, 0.5) == CAL_PLANT_IS_RUN_ADC
    # A retract held at its start (a knee breaking free) is not at the switch.
    assert calibration_is_adc(-128, 1.0, 1.0) == CAL_PLANT_IS_RUN_ADC


def test_plant_writes_the_current_sense_pin_after_each_step():
    config = SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=1000)
    plant = CalibrationTravelPlant(config, travel={"FLHL": 0.02, "FLKL": 0.98}, step=0.01)
    board = FakeBoard()
    hip = config.slot_by_firmware_id("FLHL")
    knee = config.slot_by_firmware_id("FLKL")
    plant.sync_inputs(board)
    assert board.pots[hip.pin_is] == 0 and board.pots[knee.pin_is] == 0

    board.drive(hip, -128)
    plant.advance(board)
    plant.sync_inputs(board)
    assert board.pots[hip.pin_is] == CAL_PLANT_IS_RUN_ADC
    plant.advance(board)
    plant.sync_inputs(board)
    assert plant.travel["FLHL"] == 0.0
    assert board.pots[hip.pin_is] == CAL_PLANT_IS_SWITCH_OPEN_ADC

    board.drive(hip, 0)
    board.drive(knee, 128)
    for _ in range(3):
        plant.advance(board)
    plant.sync_inputs(board)
    assert board.pots[hip.pin_is] == 0
    assert plant.travel["FLKL"] == 1.0
    assert board.pots[knee.pin_is] == CAL_PLANT_IS_STALL_ADC


def test_robot_core_path_writes_the_current_sense_pin():
    config, actuators = _front_actuators(SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=1000))
    flhl = next(a for a in actuators if a.slot.firmware_actuator_id == "FLHL")
    pin_state = PinStateModel()
    pwm_r, pwm_l, pin_en = flhl.slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_l, 128)
    travel = {"FLHL": 0.01}
    currents: dict[str, int] = {}
    board = FakeBoard()

    advance_travel_from_pwm([flhl], pin_state, travel, step=0.02, bridge=board, currents=currents)
    assert currents == {"FLHL": CAL_PLANT_IS_SWITCH_OPEN_ADC}
    sync_pot_reads_from_travel(board, [flhl], travel, currents=currents)
    assert board.pots[flhl.slot.pin_is] == CAL_PLANT_IS_SWITCH_OPEN_ADC
    sync_pot_reads_from_travel(board, [flhl], travel, currents={})
    assert board.pots[flhl.slot.pin_is] == 0


def _knee_edges_out(plant, knee) -> float:
    return (1.0 - plant.travel[knee.firmware_actuator_id]) * knee.hall_edges_per_stroke


def test_without_a_knee_model_the_plant_is_unchanged():
    config = SimulatedRobotConfig.rev3_hall()
    plant = CalibrationTravelPlant(config, travel={"FLKL": 1.0}, step=0.01)
    board = FakeBoard()
    knee = config.slot_by_firmware_id("FLKL")
    board.drive(knee, -80)
    plant.advance(board)
    assert plant.travel["FLKL"] == pytest.approx(0.99)
    board.drive(knee, 0)
    plant.advance(board)
    assert plant.travel["FLKL"] == pytest.approx(0.99)  # no spring


def test_stored_knee_breaks_free_late_and_springs_out_after_a_nudge():
    config = SimulatedRobotConfig.rev3_hall()
    knee = config.slot_by_firmware_id("FLKL")
    stroke = knee.hall_edges_per_stroke
    model = ROBOT_STORED_KNEE
    plant = CalibrationTravelPlant(
        config, travel={"FLKL": 1.0}, step=1.0 / stroke, knee_model=model
    )
    board = FakeBoard()

    board.drive(knee, -80)  # uncurl (retract)
    for _ in range(model.breakaway_ms // 10):
        plant.advance(board)
    assert plant.travel["FLKL"] == 1.0  # still breaking free
    for _ in range(12):
        plant.advance(board)
    assert _knee_edges_out(plant, knee) == pytest.approx(12)
    assert board.counts[(0, knee.slot_index)] == 12  # retract counts up

    board.drive(knee, 0)
    plant.advance(board)
    assert _knee_edges_out(plant, knee) == pytest.approx(12 + model.spring_edges)
    assert board.counts[(0, knee.slot_index)] == 12 + model.spring_edges
    plant.advance(board)  # springs once
    assert _knee_edges_out(plant, knee) == pytest.approx(12 + model.spring_edges)


def test_a_weak_curl_stops_short_of_a_pwm_200_park():
    config = SimulatedRobotConfig.rev3_hall()
    knee = config.slot_by_firmware_id("FLKL")
    model = StoredKneeModel(spring_edges=0)
    plant = CalibrationTravelPlant(
        config, travel={"FLKL": 0.9}, step=0.01, knee_model=model
    )
    board = FakeBoard()
    board.drive(knee, 80)
    for _ in range(50):
        plant.advance(board)
    assert _knee_edges_out(plant, knee) == pytest.approx(model.weak_curl_end_edges)
    board.drive(knee, 200)  # the GUI jog
    for _ in range(5):
        plant.advance(board)
    assert plant.travel["FLKL"] == 1.0
    board.drive(knee, 80)  # parked deeper: a weak curl leaves it there
    plant.advance(board)
    assert plant.travel["FLKL"] == 1.0


def test_extend_stop_blocks_extension_only():
    config = SimulatedRobotConfig.rev3_hall()
    hip = config.slot_by_firmware_id("FLHL")
    plant = CalibrationTravelPlant(
        config, travel={"FLHL": 0.0}, step=0.1, extend_stops={"FLHL": 0.25}
    )
    board = FakeBoard()
    board.drive(hip, 80)
    for _ in range(5):
        plant.advance(board)
    assert plant.travel["FLHL"] == pytest.approx(0.25)
    board.drive(hip, -80)
    plant.advance(board)
    assert plant.travel["FLHL"] == pytest.approx(0.15)
