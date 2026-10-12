"""Map Isaac joint angle to linear-pot ADC (0-1023) before per-motor calibration bands."""

from __future__ import annotations

ADC_FULL_SCALE = 1023


def joint_angle_to_pot_adc(position_rad: float, lower_rad: float, upper_rad: float) -> int:
    """Linear map: ``lower_rad`` -> 0, ``upper_rad`` -> 1023, clamped in between."""
    span = upper_rad - lower_rad
    if span <= 0.0:
        return ADC_FULL_SCALE // 2
    fraction = (position_rad - lower_rad) / span
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return ADC_FULL_SCALE
    return int(round(fraction * ADC_FULL_SCALE))
