"""Parametrize Mega slots S0-S5: jogging one slot moves its joint, not the other leg.

Assertions use ``robot.data.joint_pos``. Joints on the same FL/FR chain may swing with
the driven actuator; the three joints on the opposite leg stay small vs the commanded
joint.
"""

from __future__ import annotations

import pytest

from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

JOG_PWM = 255
JOG_STEPS = 40
MIN_COMMANDED_TRAVEL = 0.05
OPPOSITE_LEG_RATIO = 5


def _leg_side(joint_name: str) -> str:
    if joint_name.startswith("FL_"):
        return "FL"
    if joint_name.startswith("FR_"):
        return "FR"
    raise AssertionError(f"unexpected FRONT-board joint name: {joint_name!r}")


def _joint_deltas_after_jog(rig, firmware_id: str, pwm: int, steps: int) -> dict[str, float]:
    core, robot, config, indices = rig
    park_at_default_pose(core, robot)
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}
    core.jog(firmware_id, pwm)
    for _ in range(steps):
        core.step()
    return {
        name: robot.data.joint_pos[0, idx].item() - before[name]
        for name, idx in indices.items()
    }


@pytest.mark.parametrize("slot_index", range(6), ids=[f"S{i}" for i in range(6)])
def test_slot_jog_moves_commanded_joint_not_the_opposite_leg(front_board_rig, slot_index: int):
    core, robot, config, indices = front_board_rig
    slot = config.slot_by_index(slot_index)
    moved = _joint_deltas_after_jog(
        (core, robot, config, indices),
        slot.firmware_actuator_id,
        JOG_PWM,
        JOG_STEPS,
    )
    commanded = slot.isaac_joint_name
    cmd_travel = moved[commanded]

    assert abs(cmd_travel) > MIN_COMMANDED_TRAVEL, (
        f"{slot.firmware_actuator_id} ({commanded}) did not move enough: {moved}"
    )
    driven_leg = _leg_side(commanded)
    for name, delta in moved.items():
        if name == commanded or _leg_side(name) == driven_leg:
            continue
        assert abs(delta) * OPPOSITE_LEG_RATIO < abs(cmd_travel), (
            f"jogging {slot.firmware_actuator_id} moved opposite leg {name}: {moved}"
        )
