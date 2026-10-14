"""All six FRONT slots: extend, retract, hold, and partial PWM (OVERVIEW 1f).

Complements ``test_isaac_leg_drive.py`` (one slot, same assertions) and
``test_isaac_six_slot_jog.py`` (isolation on extend only). Uses a slower
``max_joint_speed`` so partial PWM stays in the linear travel band.
"""

from __future__ import annotations

import pytest

from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

DRIVE_STEPS = 60
MAX_JOINT_SPEED = 0.5
FULL_PWM = 255
PARTIAL_PWM = 122


@pytest.fixture(scope="module")
def slow_front_rig(front_board_rig):
    from firmware.sim.robot_core import RobotCore

    core, robot, config, indices = front_board_rig
    slow = RobotCore.front_board(
        env=core.env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=core.bridge,
        config=config,
        max_joint_speed=MAX_JOINT_SPEED,
    )
    return slow, robot, config, indices


def _drive_slot_delta(rig, firmware_id: str, pwm: int, steps: int = DRIVE_STEPS) -> float:
    core, robot, config, indices = rig
    slot = config.slot_by_firmware_id(firmware_id)
    joint_index = indices[slot.isaac_joint_name]

    park_at_default_pose(core, robot)
    before = robot.data.joint_pos[0, joint_index].item()
    core.jog(firmware_id, pwm)
    for _ in range(steps):
        core.step()
    return robot.data.joint_pos[0, joint_index].item() - before


def _slot_ids(slot_index: int) -> str:
    return f"S{slot_index}"


@pytest.mark.parametrize("slot_index", range(6), ids=_slot_ids)
def test_slot_positive_pwm_extends(slow_front_rig, slot_index: int):
    _, _, config, _ = slow_front_rig
    fw_id = config.slot_by_index(slot_index).firmware_actuator_id
    delta = _drive_slot_delta(slow_front_rig, fw_id, FULL_PWM)
    assert abs(delta) > 0.05, f"{fw_id} +255 did not move joint enough: {delta}"


@pytest.mark.parametrize("slot_index", range(6), ids=_slot_ids)
def test_slot_negative_pwm_retracts(slow_front_rig, slot_index: int):
    _, _, config, _ = slow_front_rig
    fw_id = config.slot_by_index(slot_index).firmware_actuator_id
    extend_delta = _drive_slot_delta(slow_front_rig, fw_id, FULL_PWM)
    retract_delta = _drive_slot_delta(slow_front_rig, fw_id, -FULL_PWM)
    assert abs(retract_delta) > 0.05, f"{fw_id} -255 did not move joint enough: {retract_delta}"
    assert extend_delta * retract_delta < 0, (
        f"{fw_id} +255 and -255 moved same joint direction: {extend_delta}, {retract_delta}"
    )


@pytest.mark.parametrize("slot_index", range(6), ids=_slot_ids)
def test_slot_zero_pwm_holds(slow_front_rig, slot_index: int):
    core, _, config, _ = slow_front_rig
    slot = config.slot_by_index(slot_index)
    _, _, pin_en = slot.drive_pins

    delta = _drive_slot_delta(slow_front_rig, slot.firmware_actuator_id, 0)

    assert core.bridge.pin_digital(pin_en) == 0
    assert abs(delta) < 0.01


@pytest.mark.parametrize("slot_index", range(6), ids=_slot_ids)
def test_slot_partial_pwm_scales_travel(slow_front_rig, slot_index: int):
    _, _, config, _ = slow_front_rig
    fw_id = config.slot_by_index(slot_index).firmware_actuator_id
    full = _drive_slot_delta(slow_front_rig, fw_id, FULL_PWM)
    partial = _drive_slot_delta(slow_front_rig, fw_id, PARTIAL_PWM)
    ratio = partial / full
    assert 0.45 < ratio < 0.51, (
        f"{fw_id} partial/full ratio {ratio} (full={full}, partial={partial})"
    )
