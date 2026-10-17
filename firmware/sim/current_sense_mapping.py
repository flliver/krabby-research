"""Map Isaac wrench load to ``analogRead(pinIS)`` (piecewise breakpoints)."""

from __future__ import annotations

from dataclasses import dataclass

from firmware.sim.pot_position import ADC_FULL_SCALE
from firmware.sim.wrench_readback import WrenchSample

# Tunable output levels (of 1023) — tune by eye in ``wrench_tuning_viewer``.
DEFAULT_IDLE_ADC = 0
DEFAULT_MOVING_NO_CONTACT_ADC = 2
DEFAULT_GROUND_CONTACT_ADC = 4
DEFAULT_LIFT_LOAD_ADC = 30

# Tunable input thresholds on effective load (N) before breakpoint table (placeholders).
DEFAULT_GROUND_CONTACT_FORCE_N = 8.0
DEFAULT_LIFT_LOAD_FORCE_N = 120.0

# Secondary proxy: |applied_torque| (Nm) -> load (N) for cross-check with wrench (TASK-2 §4).
DEFAULT_EFFORT_NM_TO_LOAD_N = 2.5

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
    effort_nm_to_load_n: float = DEFAULT_EFFORT_NM_TO_LOAD_N
    effort_cross_check: bool = True


def wrench_load_newtons(sample: WrenchSample) -> float:
    """Scalar load from incoming wrench (force magnitude)."""
    return sample.force_magnitude()


def effort_torque_to_load_newtons(
    applied_torque_nm: float,
    *,
    nm_to_load_n: float = DEFAULT_EFFORT_NM_TO_LOAD_N,
) -> float:
    """Map clipped motor torque to the same load axis as ``wrench_load_newtons``."""
    return abs(applied_torque_nm) * nm_to_load_n


def effective_load_newtons(
    sample: WrenchSample,
    applied_torque_nm: float | None,
    *,
    breakpoints: CurrentSenseBreakpoints | None = None,
) -> tuple[float, float, float | None]:
    """Combine wrench (primary) and torque (secondary) load proxies.

    Returns ``(effective_load_n, wrench_load_n, effort_load_n_or_none)``.
    When ``effort_cross_check`` is enabled and torque is present, effective load is
    ``max(wrench, effort)`` so either signal can raise the current-sense band.
    """
    bp = breakpoints or CurrentSenseBreakpoints()
    wrench_n = wrench_load_newtons(sample)
    if applied_torque_nm is None:
        return wrench_n, wrench_n, None
    effort_n = effort_torque_to_load_newtons(
        applied_torque_nm,
        nm_to_load_n=bp.effort_nm_to_load_n,
    )
    if bp.effort_cross_check:
        return max(wrench_n, effort_n), wrench_n, effort_n
    return wrench_n, wrench_n, effort_n


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
    applied_torque_nm: float | None = None,
    breakpoints: CurrentSenseBreakpoints | None = None,
) -> int:
    load_n, _, _ = effective_load_newtons(
        sample,
        applied_torque_nm,
        breakpoints=breakpoints,
    )
    return map_load_to_pin_is_adc(
        load_n,
        en_energized=en_energized,
        joint_velocity_rad_s=joint_velocity_rad_s,
        breakpoints=breakpoints,
    )
