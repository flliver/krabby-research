"""Jog each of the 18 firmware joint names; only the matching Isaac DOF moves."""

from __future__ import annotations

import pytest

from firmware.sim.simulated_robot_config import KRABBY_HEX_FIRMWARE_ACTUATOR_IDS
from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

JOG_PWM = 255
JOG_STEPS = 40
MIN_COMMANDED_TRAVEL = 0.04
OTHER_JOINT_RATIO = 4


def _joint_deltas_after_jog(rig, firmware_id: str, pwm: int, steps: int) -> dict[str, float]:
    core, robot, _configs, indices = rig
    park_at_default_pose(core, robot)
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}
    core.jog(firmware_id, pwm)
    for _ in range(steps):
        core.step()
    return {
        name: robot.data.joint_pos[0, idx].item() - before[name]
        for name, idx in indices.items()
    }


@pytest.mark.parametrize("firmware_id", KRABBY_HEX_FIRMWARE_ACTUATOR_IDS)
def test_each_firmware_joint_jog_moves_only_its_dof(eighteen_joint_rig, firmware_id: str):
    core, robot, configs, indices = eighteen_joint_rig
    slot = configs.slot_by_firmware_id(firmware_id)
    commanded_joint = slot.isaac_joint_name

    moved = _joint_deltas_after_jog(
        (core, robot, configs, indices),
        firmware_id,
        JOG_PWM,
        JOG_STEPS,
    )
    cmd_travel = moved[commanded_joint]
    assert abs(cmd_travel) > MIN_COMMANDED_TRAVEL, (
        f"{firmware_id} ({commanded_joint}) did not move enough: {moved}"
    )

    for name, delta in moved.items():
        if name == commanded_joint:
            continue
        assert abs(delta) * OTHER_JOINT_RATIO < abs(cmd_travel), (
            f"jogging {firmware_id} also moved {name}: {moved}"
        )
