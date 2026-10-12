"""Map Isaac wrench load to ``analogRead(pinIS)`` (TASK-2 §4 breakpoints)."""

from __future__ import annotations

from dataclasses import dataclass

from firmware.sim.pot_position import ADC_FULL_SCALE
from firmware.sim.wrench_readback import WrenchSample

# Tunable output levels (of 1023) — tune by eye in the wrench viewer (TASK-2 §4).
DEFAULT_IDLE_ADC = 0
DEFAULT_MOVING_NO_CONTACT_ADC = 2
DEFAULT_GROUND_CONTACT_ADC = 4
DEFAULT_LIFT_LOAD_ADC = 30

# Tunable input thresholds on ``wrench_load_newtons`` (placeholders until fleet tuning).
DEFAULT_GROUND_CONTACT_FORCE_N = 8.0
DEFAULT_LIFT_LOAD_FORCE_N = 120.0

JOINT_MOVING_VEL_RAD_S = 0.05


@dataclass(frozen=True, slots=True)
class CurrentSenseBreakpoints:
    idle_adc: int = DEFAULT_IDLE_ADC
    moving_no_contact_adc: int = DEFAULT_MOVING_NO_CONTACT_ADC
    ground_contact_adc: int = DEFAULT_GROUND_CONTACT_ADC
    lift_load_adc: int = DEFAULT_LIFT_LOAD_ADC
    ground_contact_force_n: float = DEFAULT_GROUND_CONTACT_FORCE_N
    lift_load_force_n: float = DEFAULT_LIFT_LOAD_FORCE_N
    joint_moving_vel_rad_s: float = JOINT_MOVING_VEL_RAD_S


def wrench_load_newtons(sample: WrenchSample) -> float:
    """Scalar load from incoming wrench (force magnitude; primary TASK-2 signal)."""
    return sample.force_magnitude()


def map_load_to_pin_is_adc(
    load_n: float,
    *,
    en_energized: bool,
    joint_velocity_rad_s: float = 0.0,
    breakpoints: CurrentSenseBreakpoints | None = None,
) -> int:
    """Piecewise map from load + drive state to raw current-sense ADC (firmware EMA unchanged)."""
    bp = breakpoints or CurrentSenseBreakpoints()
    moving = en_energized and abs(joint_velocity_rad_s) >= bp.joint_moving_vel_rad_s

    if load_n >= bp.lift_load_force_n:
        adc = bp.lift_load_adc
    elif load_n >= bp.ground_contact_force_n:
        adc = bp.ground_contact_adc
    elif moving:
        adc = bp.moving_no_contact_adc
    else:
        adc = bp.idle_adc

    if adc <= 0:
        return 0
    if adc >= ADC_FULL_SCALE:
        return ADC_FULL_SCALE
    return int(adc)


def map_wrench_to_pin_is_adc(
    sample: WrenchSample,
    *,
    en_energized: bool,
    joint_velocity_rad_s: float = 0.0,
    breakpoints: CurrentSenseBreakpoints | None = None,
) -> int:
    return map_load_to_pin_is_adc(
        wrench_load_newtons(sample),
        en_energized=en_energized,
        joint_velocity_rad_s=joint_velocity_rad_s,
        breakpoints=breakpoints,
    )
