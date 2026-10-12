"""Joint angle to linear-pot ADC mapping."""

import pytest

from firmware.sim.pot_position import ADC_FULL_SCALE, joint_angle_to_pot_adc


@pytest.mark.parametrize(
    ("position", "lower", "upper", "expected"),
    [
        (-1.0, -1.0, 1.0, 0),
        (1.0, -1.0, 1.0, ADC_FULL_SCALE),
        (0.0, -1.0, 1.0, 512),
        (0.25, 0.0, 1.0, 256),
    ],
)
def test_joint_angle_to_pot_adc(position, lower, upper, expected):
    assert joint_angle_to_pot_adc(position, lower, upper) == expected


def test_joint_angle_to_pot_adc_clamps_outside_limits():
    assert joint_angle_to_pot_adc(-5.0, 0.0, 1.0) == 0
    assert joint_angle_to_pot_adc(5.0, 0.0, 1.0) == ADC_FULL_SCALE
