"""Closed-loop ``setTarget`` / ``update()`` with pot readback from Isaac."""

from __future__ import annotations

import pytest

from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"
CLOSED_LOOP_STEPS = 80


def _normalized_travel(robot, joint_index: int) -> float:
    limits = robot.data.soft_joint_pos_limits[0, joint_index]
    pos = robot.data.joint_pos[0, joint_index].item()
    lo = limits[0].item()
    hi = limits[1].item()
    return (pos - lo) / (hi - lo)


def _firmware_normalized_pot(core, config, slot_id: str) -> float:
    """Same 0..1 space as ``setTarget`` / ``getPos()`` after stops match the pot band."""
    slot = config.slot_by_firmware_id(slot_id)
    band = config.pot_band_for_slot(slot.slot_index)
    adc = core.bridge.analog_read(slot.pin_pot)
    return (adc - band.min_adc) / (band.max_adc - band.min_adc)


def test_set_target_moves_joint_toward_command(front_board_rig):
    core, robot, config, indices = front_board_rig
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]

    park_at_default_pose(core, robot)
    core.step()
    start = _normalized_travel(robot, joint_index)
    target = min(0.92, start + 0.28)
    assert target - start > 0.12

    core.set_target(SLOT, target)
    for _ in range(CLOSED_LOOP_STEPS):
        core.step()

    end = _normalized_travel(robot, joint_index)
    assert end > start + 0.08
    assert end > start + 0.5 * (target - start)


def test_set_target_at_current_command_keeps_pwm_low(front_board_rig):
    """Velocity shim cannot lock joint pose; near-target command should idle the drive."""
    core, robot, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)
    pwm_r, pwm_l, pin_en = slot.drive_pins

    park_at_default_pose(core, robot)
    for _ in range(12):
        core.step()
    here = _firmware_normalized_pot(core, config, SLOT)

    core.set_target(SLOT, here)
    for _ in range(CLOSED_LOOP_STEPS):
        core.step()
    pwm_near = core.bridge.pin_pwm(pwm_r) + core.bridge.pin_pwm(pwm_l)

    park_at_default_pose(core, robot)
    for _ in range(12):
        core.step()
    core.set_target(SLOT, min(0.95, here + 0.35))
    for _ in range(CLOSED_LOOP_STEPS):
        core.step()
    pwm_far = core.bridge.pin_pwm(pwm_r) + core.bridge.pin_pwm(pwm_l)

    assert core.bridge.pin_digital(pin_en) == 0 or pwm_near < pwm_far
