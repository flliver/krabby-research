"""Map Isaac joint angle to linear-pot ADC (ideal 0-1023 or per-motor band)."""

from __future__ import annotations

from firmware.sim.pot_calibration import PotCalibrationBand, travel_fraction_to_pot_adc

ADC_FULL_SCALE = 1023


def joint_angle_to_travel_fraction(
    position_rad: float,
    lower_rad: float,
    upper_rad: float,
) -> float:
    """``lower_rad`` -> 0, ``upper_rad`` -> 1, clamped."""
    span = upper_rad - lower_rad
    if span <= 0.0:
        return 0.5
    fraction = (position_rad - lower_rad) / span
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 1.0
    return fraction


def joint_angle_to_pot_adc(
    position_rad: float,
    lower_rad: float,
    upper_rad: float,
    *,
    band: PotCalibrationBand | None = None,
) -> int:
    fraction = joint_angle_to_travel_fraction(position_rad, lower_rad, upper_rad)
    if band is None:
        return travel_fraction_to_pot_adc(fraction, PotCalibrationBand(0, ADC_FULL_SCALE))
    return travel_fraction_to_pot_adc(fraction, band)
