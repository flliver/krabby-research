"""``body_incoming_joint_wrench_b`` rises when a leg loads the terrain."""

from __future__ import annotations

import pytest

from firmware.sim.isaac_scene import advance_sim
from firmware.sim.wrench_readback import (
    max_actuator_wrench_force,
    resolve_wrench_body_indices,
    sample_actuator_wrenches,
)

pytestmark = pytest.mark.isaacsim

SETTLE_STEPS = 120
LOAD_JOG_STEPS = 80
MIN_LOAD_FORCE_N = 1.0


def test_incoming_wrench_nonzero_when_leg_loads_floor(gravity_front_board):
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    core, robot, _config = gravity_front_board
    config = SimulatedRobotConfig.front_board_rev3()
    body_indices = resolve_wrench_body_indices(robot, config)
    core.reset()

    for _ in range(SETTLE_STEPS):
        advance_sim(core.env)

    baseline = max_actuator_wrench_force(sample_actuator_wrenches(robot, body_indices))

    core.jog("FLKL", 200)
    for _ in range(LOAD_JOG_STEPS):
        core.step()

    loaded = max_actuator_wrench_force(
        sample_actuator_wrenches(robot, body_indices),
        actuator_ids=("FLKL", "FLHL", "FLHY"),
    )

    assert loaded >= MIN_LOAD_FORCE_N, (
        f"expected leg load wrench >= {MIN_LOAD_FORCE_N} N, got {loaded} "
        f"(baseline max force {baseline})"
    )
