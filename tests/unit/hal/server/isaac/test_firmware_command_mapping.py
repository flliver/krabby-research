"""Unit tests for HAL → firmware jog mapping."""

import pytest

from hal.server.isaac.firmware_command_mapping import (
    joint_radians_to_jog_pwm,
    rad_to_jog_pwm,
    signed_pwm_to_normalized_action,
)


def test_rad_to_jog_pwm_neutral_and_scaled():
    assert rad_to_jog_pwm(0.0) == 0
    assert rad_to_jog_pwm(0.005) == 0
    assert rad_to_jog_pwm(0.5) == 255
    assert rad_to_jog_pwm(-0.5) == -255


def test_joint_radians_to_jog_pwm_keys():
    jog = joint_radians_to_jog_pwm(
        {"FL_hip_pitch": 0.2, "FR_knee": -0.1},
        ("FL_hip_pitch", "FR_knee"),
    )
    assert set(jog) == {"FLHL", "FRKL"}
    assert jog["FLHL"] > 0
    assert jog["FRKL"] < 0


def test_signed_pwm_to_normalized_action():
    assert signed_pwm_to_normalized_action(0) == 0.0
    assert signed_pwm_to_normalized_action(255) == 1.0
    assert signed_pwm_to_normalized_action(-128) == pytest.approx(-128 / 255.0)
