"""Per-motor pot ADC bands and auto-cal-style endpoint discovery."""

import pytest

from firmware.sim.pot_calibration import (
    POT_MAX_ADC_RANGE,
    POT_MIN_ADC_RANGE,
    PotCalibrationBand,
    draw_pot_calibration_bands,
    travel_fraction_to_pot_adc,
)
from firmware.sim.pot_position import joint_angle_to_pot_adc
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def test_draw_pot_calibration_bands_is_reproducible_with_seed():
    a = draw_pot_calibration_bands(6, seed=42)
    b = draw_pot_calibration_bands(6, seed=42)
    assert a == b
    assert draw_pot_calibration_bands(6, seed=99) != a


def test_draw_pot_calibration_bands_differ_across_motors():
    bands = draw_pot_calibration_bands(6, seed=1)
    assert len({(b.min_adc, b.max_adc) for b in bands}) == 6
    for band in bands:
        assert POT_MIN_ADC_RANGE[0] <= band.min_adc <= POT_MIN_ADC_RANGE[1]
        assert POT_MAX_ADC_RANGE[0] <= band.max_adc <= POT_MAX_ADC_RANGE[1]


def test_travel_fraction_to_pot_adc_interpolates_and_saturates():
    band = PotCalibrationBand(100, 900)
    assert travel_fraction_to_pot_adc(0.0, band) == 100
    assert travel_fraction_to_pot_adc(1.0, band) == 900
    assert travel_fraction_to_pot_adc(-0.2, band) == 100
    assert travel_fraction_to_pot_adc(1.5, band) == 900
    assert travel_fraction_to_pot_adc(0.5, band) == 500


def test_joint_angle_map_uses_per_motor_band():
    band = PotCalibrationBand(120, 880)
    lower, upper = -1.0, 1.0
    assert joint_angle_to_pot_adc(lower, lower, upper, band=band) == 120
    assert joint_angle_to_pot_adc(upper, lower, upper, band=band) == 880
    assert joint_angle_to_pot_adc(0.0, lower, upper, band=band) == 500


def test_front_board_config_attaches_one_band_per_slot():
    config = SimulatedRobotConfig.front_board_rev3(pot_calibration_seed=7)
    assert len(config.pot_bands) == len(config.slots) == 6


def test_auto_cal_would_persist_distinct_min_max_per_motor():
    """Stall calibration records ``getRawPos()`` at limits; bands make those differ per motor."""
    config = SimulatedRobotConfig.front_board_rev3(pot_calibration_seed=11)
    discovered = [(band.min_adc, band.max_adc) for band in config.pot_bands]
    assert len(set(discovered)) == 6
    for min_adc, max_adc in discovered:
        assert min_adc < max_adc
        normalized_mid = (min_adc + max_adc) / 2.0
        span = max_adc - min_adc
        assert abs(((normalized_mid - min_adc) / span) - 0.5) < 0.01
