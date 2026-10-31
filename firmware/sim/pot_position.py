"""Map Isaac joint angle to linear-pot ADC (ideal 0-1023 or per-motor band)."""

from __future__ import annotations

import math

from firmware.sim.pot_calibration import PotCalibrationBand, travel_fraction_to_pot_adc

ADC_FULL_SCALE = 1023

# ``assets/crab.usda`` CamShaft revolute joints omit USD limits; Isaac then exposes
# non-finite soft limits. Use one full turn so pot/hall readback still scales angle.
_NOMINAL_UNLIMITED_JOINT_LOWER_RAD = -math.pi
_NOMINAL_UNLIMITED_JOINT_UPPER_RAD = math.pi


def normalize_joint_travel_limits(
    lower_rad: float,
    upper_rad: float,
) -> tuple[float, float]:
    """Return finite ``(lower, upper)`` with positive span for travel normalization."""
    if math.isfinite(lower_rad) and math.isfinite(upper_rad):
        span = upper_rad - lower_rad
        if math.isfinite(span) and span > 0.0:
            return lower_rad, upper_rad
    return _NOMINAL_UNLIMITED_JOINT_LOWER_RAD, _NOMINAL_UNLIMITED_JOINT_UPPER_RAD


def joint_angle_to_travel_fraction(
    position_rad: float,
    lower_rad: float,
    upper_rad: float,
) -> float:
    """``lower_rad`` -> 0, ``upper_rad`` -> 1, clamped."""
    if not math.isfinite(position_rad):
        return 0.5
    lower_rad, upper_rad = normalize_joint_travel_limits(lower_rad, upper_rad)
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
