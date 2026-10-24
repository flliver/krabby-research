"""Wrench load -> ``pinIS`` ADC breakpoints."""

import pytest

from firmware.sim.current_sense_mapping import (
    DEFAULT_GROUND_CONTACT_ADC,
    DEFAULT_IDLE_ADC,
    DEFAULT_LIFT_LOAD_ADC,
    DEFAULT_MOVING_NO_CONTACT_ADC,
    CurrentSenseBreakpoints,
    effective_load_newtons,
    map_load_to_pin_is_adc,
    map_wrench_to_pin_is_adc,
)
from firmware.sim.wrench_readback import WrenchSample

BP = CurrentSenseBreakpoints(
    ground_contact_force_n=10.0,
    lift_load_force_n=100.0,
    joint_moving_vel_rad_s=0.05,
)


def test_idle_when_deenergized_and_low_load():
    assert map_load_to_pin_is_adc(0.0, en_energized=False, breakpoints=BP) == DEFAULT_IDLE_ADC


def test_moving_without_contact_hits_band():
    adc = map_load_to_pin_is_adc(
        0.0,
        en_energized=True,
        joint_velocity_rad_s=0.2,
        breakpoints=BP,
    )
    assert adc == DEFAULT_MOVING_NO_CONTACT_ADC


def test_ground_contact_load_hits_band():
    adc = map_load_to_pin_is_adc(25.0, en_energized=False, breakpoints=BP)
    assert adc == DEFAULT_GROUND_CONTACT_ADC


def test_lift_load_hits_band():
    adc = map_load_to_pin_is_adc(150.0, en_energized=True, breakpoints=BP)
    assert adc == DEFAULT_LIFT_LOAD_ADC


def test_effort_cross_check_raises_effective_load():
    sample = WrenchSample(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)  # 5 N wrench
    bp = CurrentSenseBreakpoints(
        effort_nm_to_load_n=10.0,
        effort_cross_check=True,
    )
    effective, wrench_n, effort_n = effective_load_newtons(sample, 15.0, breakpoints=bp)
    assert wrench_n == 5.0
    assert effort_n == 150.0
    assert effective == 150.0


def test_effort_cross_check_can_be_disabled():
    sample = WrenchSample(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    bp = CurrentSenseBreakpoints(effort_cross_check=False, effort_nm_to_load_n=10.0)
    effective, _, effort_n = effective_load_newtons(sample, 15.0, breakpoints=bp)
    assert effort_n == 150.0
    assert effective == 5.0


def test_map_wrench_uses_torque_cross_check_for_adc():
    light = WrenchSample(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    bp = CurrentSenseBreakpoints(
        ground_contact_force_n=10.0,
        lift_load_force_n=100.0,
        effort_nm_to_load_n=10.0,
    )
    adc = map_wrench_to_pin_is_adc(
        light,
        en_energized=False,
        applied_torque_nm=15.0,
        breakpoints=bp,
    )
    assert adc == DEFAULT_LIFT_LOAD_ADC


def test_wrench_wrapper_uses_force_magnitude():
    sample = WrenchSample(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)  # |F| = 5 N
    adc = map_wrench_to_pin_is_adc(
        sample,
        en_energized=False,
        joint_velocity_rad_s=0.0,
        breakpoints=BP,
    )
    assert adc == DEFAULT_IDLE_ADC
    heavy = WrenchSample(0.0, 0.0, 110.0, 0.0, 0.0, 0.0)
    assert (
        map_wrench_to_pin_is_adc(heavy, en_energized=False, breakpoints=BP)
        == DEFAULT_LIFT_LOAD_ADC
    )


@pytest.mark.parametrize(
    ("load_n", "en", "vel", "expected"),
    [
        (0.0, False, 0.0, 0),
        (0.0, True, 0.1, 2),
        (50.0, False, 0.0, 4),
        (200.0, True, 0.0, 30),
    ],
)
def test_breakpoint_table_synthetic_inputs(load_n, en, vel, expected):
    assert (
        map_load_to_pin_is_adc(load_n, en_energized=en, joint_velocity_rad_s=vel, breakpoints=BP)
        == expected
    )
