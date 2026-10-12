"""Pin state model for firmware-in-Isaac simulation."""

import pytest

from firmware.sim.pin_state import HIGH, INPUT, LOW, OUTPUT, PinStateModel
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def test_pin_mode_is_recorded():
    model = PinStateModel()
    model.pin_mode(22, OUTPUT)
    assert model.get_mode(22) == OUTPUT
    assert model.get_mode(99) is None


def test_digital_write_records_level():
    model = PinStateModel()
    model.digital_write(22, LOW)
    assert model.digital_read(22) == LOW
    model.digital_write(22, HIGH)
    assert model.digital_read(22) == HIGH
    assert model.is_digital_high(22)


def test_analog_write_records_pwm_and_clamps():
    model = PinStateModel()
    model.analog_write(2, 122)
    assert model.pwm_duty(2) == 122
    model.analog_write(3, 300)
    assert model.pwm_duty(3) == 255
    model.analog_write(3, -5)
    assert model.pwm_duty(3) == 0


def test_drive_actuator_extend_retract_and_coast():
    model = PinStateModel()
    en, pwm_r, pwm_l = 22, 2, 3

    model.digital_write(en, HIGH)
    model.analog_write(pwm_r, 200)
    model.analog_write(pwm_l, 0)
    assert model.signed_pwm_command(pwm_r, pwm_l, en) == 200

    model.analog_write(pwm_r, 0)
    model.analog_write(pwm_l, 150)
    assert model.signed_pwm_command(pwm_r, pwm_l, en) == -150

    model.digital_write(en, LOW)
    model.analog_write(pwm_r, 200)
    model.analog_write(pwm_l, 150)
    assert model.signed_pwm_command(pwm_r, pwm_l, en) == 0


def test_slot_signed_command_from_config():
    model = PinStateModel()
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    pwm_r, pwm_l, en = slot.drive_pins

    model.digital_write(en, HIGH)
    model.analog_write(pwm_r, 80)
    model.analog_write(pwm_l, 0)
    assert model.signed_pwm_command(*slot.drive_pins) == 80


@pytest.mark.parametrize(
    ("firmware_id", "expected_sign"),
    [("FLHL", -100), ("FRKL", 50)],
)
def test_en_gating_per_front_board_slot(firmware_id, expected_sign):
    model = PinStateModel()
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id(firmware_id)
    pwm_r, pwm_l, en = slot.drive_pins
    magnitude = abs(expected_sign)

    model.digital_write(en, HIGH)
    if expected_sign > 0:
        model.analog_write(pwm_r, magnitude)
        model.analog_write(pwm_l, 0)
    else:
        model.analog_write(pwm_r, 0)
        model.analog_write(pwm_l, magnitude)
    assert model.signed_pwm_command(pwm_r, pwm_l, en) == expected_sign

    model.digital_write(en, LOW)
    assert model.signed_pwm_command(pwm_r, pwm_l, en) == 0
