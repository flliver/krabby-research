"""One FRONT slot driven through the host firmware: extend, retract, hold, partial PWM.

Assertions read ``robot.data.joint_pos`` — the simulated plant, not the potentiometer
readback, which the drive-only loop does not feed yet.
"""

from __future__ import annotations

import pytest

from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"  # FL hip pitch; +PWM extends the actuator
DRIVE_STEPS = 60
# Full scale at the rig's 2.0 rad/s saturates the joint, so travel stops scaling with PWM
# (0.59 of full travel at 122/255 instead of 0.48). 0.5 rad/s keeps both commands inside
# what the plant can follow.
MAX_JOINT_SPEED = 0.5


@pytest.fixture(scope="module")
def slow_rig(front_board_rig):
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


def drive_from_default(rig, pwm: int, steps: int = DRIVE_STEPS) -> float:
    """Park the leg, jog one slot for ``steps`` loop iterations, return the joint delta."""
    core, robot, config, indices = rig
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]

    park_at_default_pose(core, robot)
    before = robot.data.joint_pos[0, joint_index].item()
    core.jog(SLOT, pwm)
    for _ in range(steps):
        core.step()
    return robot.data.joint_pos[0, joint_index].item() - before


def test_positive_pwm_extends(slow_rig):
    assert drive_from_default(slow_rig, 255) > 0.05


def test_negative_pwm_retracts(slow_rig):
    assert drive_from_default(slow_rig, -255) < -0.05


def test_zero_pwm_pulls_enable_low_and_holds(slow_rig):
    core, _, config, _ = slow_rig
    _, _, pin_en = config.slot_by_firmware_id(SLOT).drive_pins

    delta = drive_from_default(slow_rig, 0)

    assert core.bridge.pin_digital(pin_en) == 0
    assert abs(delta) < 0.01


def test_partial_pwm_scales_travel(slow_rig):
    """122/255 of full scale should travel ~48% as far: the drive map is linear."""
    full = drive_from_default(slow_rig, 255)
    partial = drive_from_default(slow_rig, 122)

    ratio = partial / full
    assert 0.45 < ratio < 0.51, f"partial/full travel ratio {ratio} (full={full}, partial={partial})"
