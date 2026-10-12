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
