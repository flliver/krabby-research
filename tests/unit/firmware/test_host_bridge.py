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
