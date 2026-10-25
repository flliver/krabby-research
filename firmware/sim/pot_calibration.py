"""Per-motor linear-pot ADC bands (manufacturer-style calibration spread)."""

from __future__ import annotations

import random
from dataclasses import dataclass

# Manufacturer-style bands: min stop ADC roughly 80–220, max stop roughly 800–1000.
POT_MIN_ADC_RANGE = (80, 220)
POT_MAX_ADC_RANGE = (800, 1000)

DEFAULT_POT_CALIBRATION_SEED = 0x4D21  # reproducible FRONT-board pot draw


@dataclass(frozen=True, slots=True)
class PotCalibrationBand:
    min_adc: int
    max_adc: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_adc <= 1023:
            raise ValueError(f"min_adc out of range: {self.min_adc}")
        if not 0 <= self.max_adc <= 1023:
            raise ValueError(f"max_adc out of range: {self.max_adc}")
        if self.min_adc >= self.max_adc:
            raise ValueError(
                f"min_adc must be < max_adc: {self.min_adc} >= {self.max_adc}"
            )


def draw_pot_calibration_bands(count: int, seed: int) -> tuple[PotCalibrationBand, ...]:
    """One independent band per slot; deterministic for a given ``seed``."""
    rng = random.Random(seed)
    bands: list[PotCalibrationBand] = []
    for _ in range(count):
        min_adc = rng.randint(POT_MIN_ADC_RANGE[0], POT_MIN_ADC_RANGE[1])
        max_adc = rng.randint(POT_MAX_ADC_RANGE[0], POT_MAX_ADC_RANGE[1])
        if min_adc >= max_adc:
            max_adc = min(1023, min_adc + 100)
        bands.append(PotCalibrationBand(min_adc, max_adc))
    return tuple(bands)


def travel_fraction_to_pot_adc(fraction: float, band: PotCalibrationBand) -> int:
    """Map normalized travel [0, 1] onto ``[min_adc, max_adc]``, saturate outside."""
    if fraction <= 0.0:
        return band.min_adc
    if fraction >= 1.0:
        return band.max_adc
    span = band.max_adc - band.min_adc
    return int(round(band.min_adc + fraction * span))
