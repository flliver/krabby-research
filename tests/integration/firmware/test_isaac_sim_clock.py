"""Host ``millis()`` tracks Isaac physics step time inside ``RobotCore``."""

from __future__ import annotations

import math

import pytest

from firmware.sim.isaac_scene import physics_step_ms

pytestmark = pytest.mark.isaacsim

RAMP_INTERVAL_MS = 10
PWM_DEADBAND = 20


def test_millis_accumulates_physics_step_dt(front_board_rig):
    core, _, _, _ = front_board_rig
    dt_ms = physics_step_ms(core.env)
    core.reset()

    steps = 37
    for _ in range(steps):
        core.step()

    assert core.millis() == steps * dt_ms


def test_position_target_ramp_ticks_with_physics_millis(front_board_rig):
    """``rampIntervalMs`` is wall ms; clock must follow physics dt for ramps to step."""
    core, _, config, _ = front_board_rig
    dt_ms = physics_step_ms(core.env)
    slot = config.slot_by_firmware_id("FLHY")
    pwm_r, _, pin_en = slot.drive_pins

    core.reset()
    core.set_target("FLHY", 1.0)

    # Firmware ramps internal PWM in steps of 5 every 10 ms; drive pins stay at 0 until
    # magnitude exceeds pwmDeadband (20).
    min_steps = math.ceil((PWM_DEADBAND / 5) * RAMP_INTERVAL_MS / dt_ms) + 2
    for _ in range(min_steps):
        core.step()

    assert core.millis() == min_steps * dt_ms
    assert core.bridge.pin_digital(pin_en) == 1
    assert core.bridge.pin_pwm(pwm_r) >= PWM_DEADBAND
