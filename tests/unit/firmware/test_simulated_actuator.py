import pytest

from firmware.sim.pin_state import HIGH, PinStateModel
from firmware.sim.simulated_actuator import (
    PWM_COMMAND_FULL_SCALE,
    RecordedArticulation,
    SimulatedActuator,
)
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def write_drive_pattern(
    pin_state: PinStateModel,
    *,
    pin_pwm_r: int,
    pin_pwm_l: int,
    pin_en: int,
    pwm_r: int,
    pwm_l: int,
    en_high: bool,
) -> None:
    pin_state.digital_write(pin_en, HIGH if en_high else 0)
    pin_state.analog_write(pin_pwm_r, pwm_r)
    pin_state.analog_write(pin_pwm_l, pwm_l)

MAX_SPEED = 2.5


@pytest.fixture
def flhy_actuator() -> SimulatedActuator:
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    return SimulatedActuator(slot=slot, joint_index=3, max_joint_speed=MAX_SPEED)


@pytest.fixture
def pin_state() -> PinStateModel:
    return PinStateModel()


def test_signed_pwm_command_cases(flhy_actuator, pin_state):
    pwm_r, pwm_l, pin_en = flhy_actuator.slot.drive_pins

    write_drive_pattern(
        pin_state,
        pin_pwm_r=pwm_r,
        pin_pwm_l=pwm_l,
        pin_en=pin_en,
        pwm_r=255,
        pwm_l=0,
        en_high=True,
    )
    assert flhy_actuator.signed_pwm_command(pin_state) == 255

    write_drive_pattern(
        pin_state,
        pin_pwm_r=pwm_r,
        pin_pwm_l=pwm_l,
        pin_en=pin_en,
        pwm_r=0,
        pwm_l=255,
        en_high=True,
    )
    assert flhy_actuator.signed_pwm_command(pin_state) == -255

    write_drive_pattern(
        pin_state,
        pin_pwm_r=pwm_r,
        pin_pwm_l=pwm_l,
        pin_en=pin_en,
        pwm_r=122,
        pwm_l=0,
        en_high=True,
    )
    assert flhy_actuator.signed_pwm_command(pin_state) == 122

    write_drive_pattern(
        pin_state,
        pin_pwm_r=pwm_r,
        pin_pwm_l=pwm_l,
        pin_en=pin_en,
        pwm_r=200,
        pwm_l=100,
        en_high=False,
    )
    assert flhy_actuator.signed_pwm_command(pin_state) == 0


@pytest.mark.parametrize(
    ("pwm_r", "pwm_l", "en_high", "expected_cmd", "expected_speed"),
    [
        (255, 0, True, 255, MAX_SPEED),
        (0, 255, True, -255, -MAX_SPEED),
        (122, 0, True, 122, MAX_SPEED * (122 / PWM_COMMAND_FULL_SCALE)),
        (200, 0, False, 0, 0.0),
    ],
)
def test_drive_scalar_and_recorded_articulation(
    flhy_actuator,
    pin_state,
    pwm_r,
    pwm_l,
    en_high,
    expected_cmd,
    expected_speed,
):
    pwm_r_pin, pwm_l_pin, pin_en = flhy_actuator.slot.drive_pins
    write_drive_pattern(
        pin_state,
        pin_pwm_r=pwm_r_pin,
        pin_pwm_l=pwm_l_pin,
        pin_en=pin_en,
        pwm_r=pwm_r,
        pwm_l=pwm_l,
        en_high=en_high,
    )

    assert flhy_actuator.signed_pwm_command(pin_state) == expected_cmd
    assert flhy_actuator.drive_scalar(pin_state) == pytest.approx(expected_speed)

    robot = RecordedArticulation()
    applied = flhy_actuator.apply_velocity_target(pin_state, robot)
    assert applied == pytest.approx(expected_speed)
    assert robot.velocity_for(flhy_actuator.joint_index) == pytest.approx(expected_speed)
