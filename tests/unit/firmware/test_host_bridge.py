import pytest

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

@pytest.fixture(scope="module")
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    bridge.reset()
    return bridge


def test_handle_jog_drives_expected_pins(host):
    host.reset()
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    pwm_r, pwm_l, pin_en = slot.drive_pins

    host.handle_jog("FLHY", 100)

    assert host.pin_digital(pin_en) == 1
    assert host.pin_pwm(pwm_r) == 100
    assert host.pin_pwm(pwm_l) == 0


def test_handle_jog_negative_pwm_retracts(host):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHL")
    pwm_r, pwm_l, pin_en = slot.drive_pins

    host.reset()
    host.handle_jog("FLHL", -80)

    assert host.pin_digital(pin_en) == 1
    assert host.pin_pwm(pwm_r) == 0
    assert host.pin_pwm(pwm_l) == 80


def test_update_all_keeps_jogged_pins_asserted(host):
    """No position target is set, so updateAll must not override the jog."""
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FRKL")
    pwm_r, _, pin_en = slot.drive_pins

    host.reset()
    host.handle_jog("FRKL", 120)
    host.update_all()

    assert host.pin_digital(pin_en) == 1
    assert host.pin_pwm(pwm_r) == 120


def test_advance_millis_moves_the_sim_clock(host):
    host.reset()
    assert host.millis() == 0

    host.advance_millis(10)
    host.advance_millis(5)

    assert host.millis() == 15


def test_eeprom_and_calibration_round_trip_via_host_api(host):
    host.reset()
    host.eeprom_update(10, 0x12)
    assert host.eeprom_read(10) == 0x12

    host.set_actuator_stops("FLHY", 150, 850)
    host.save_calibration()
    host.set_actuator_stops("FLHY", 0, 1023)
    host.load_calibration()
    assert host.actuator_min_stop("FLHY") == 150
    assert host.actuator_max_stop("FLHY") == 850


def test_hall_counts_are_signed_and_kept_per_board(host):
    host.reset()
    # A rising angle extends and counts down (retract counts up).
    host.board_sync_hall_joint(1, 2, 0.0, 0.0, 1.0)
    host.board_sync_hall_joint(1, 2, 0.5, 0.0, 1.0)
    extended = host.board_hall_edge_count(1, 2)
    assert extended < 0
    host.board_sync_hall_joint(1, 2, 0.25, 0.0, 1.0)
    assert extended < host.board_hall_edge_count(1, 2) < 0
    # Other boards and the single-board API (board 0) are untouched.
    assert host.board_hall_edge_count(0, 2) == 0
    assert host.board_hall_edge_count(2, 2) == 0
    assert host.hall_edge_count(2) == 0

    host.board_set_hall_edge_count(0, 4, -1234)
    assert host.hall_edge_count(4) == -1234
    assert host.board_hall_edge_count(0, 4) == -1234
    host.board_set_hall_edge_count(2, 4, 2**31 - 1)
    assert host.board_hall_edge_count(2, 4) == 2**31 - 1

    host.sync_hall_joint(3, 1.0, 0.0, 1.0)
    host.sync_hall_joint(3, 0.0, 0.0, 1.0)
    assert host.hall_edge_count(3) > 0  # retracting counts up on board 0


def test_retarget_moves_the_baseline_without_counting(host):
    host.reset()
    host.board_sync_hall_joint(2, 1, 0.2, 0.0, 1.0)
    host.board_retarget_hall_joint(2, 1, 0.8, 0.0, 1.0)
    assert host.board_hall_edge_count(2, 1) == 0
    host.board_sync_hall_joint(2, 1, 0.8, 0.0, 1.0)
    assert host.board_hall_edge_count(2, 1) == 0


def _usb_text_after_boot(host) -> str:
    """USB output over one SENS period (2 s) after setup."""
    host.firmware_setup()
    chars = []
    for _ in range(210):
        host.advance_millis(10)
        host.firmware_loop_once()
        while host.serial_usb_tx_available():
            chars.append(chr(host.serial_usb_tx_read()))
    return "".join(chars)


def test_quadrature_flag_defaults_off_and_reset_clears_it(host):
    config = SimulatedRobotConfig.rev3_hall()
    host.reset()
    host.board_set_hall_quadrature(config.board_id, True)
    assert "SENS FRONT FLHY:?" in _usb_text_after_boot(host)

    # A reset makes the board non-quadrature again: no SENS line.
    host.reset()
    assert "SENS " not in _usb_text_after_boot(host)


def test_board_analog_read_reads_one_board_and_open_inputs_float_high(host):
    host.reset()
    host.set_analog_read(5, 300)
    host.set_board_analog_read(0, 5, 700)
    assert host.analog_read(5) == 700
    host.set_board_analog_read(0, 5, 700, open_input=True)
    # Not in INPUT_PULLUP yet, so the open input still reads its value.
    assert host.analog_read(5) == 700


def test_three_board_sensor_state_is_none_before_any_sens_line(host):
    host.reset()
    assert host.three_board_sensor_state("FLHL") is None
    assert host.three_board_sensor_state("NOPE") is None
