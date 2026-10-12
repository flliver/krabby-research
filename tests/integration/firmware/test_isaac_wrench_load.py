"""``body_incoming_joint_wrench_b`` rises when a leg loads the terrain (TASK-2 §4)."""

from __future__ import annotations

import pytest

from firmware.sim.isaac_scene import (
    advance_sim,
    get_robot_articulation,
    make_crab_hex_env,
    resolve_front_board_joint_indices,
)
from firmware.sim.wrench_readback import (
    max_actuator_wrench_force,
    resolve_wrench_body_indices,
    sample_actuator_wrenches,
)
from tests.integration.firmware.conftest import skip_unless_firmware_isaac

pytestmark = pytest.mark.isaacsim

SETTLE_STEPS = 120
LOAD_JOG_STEPS = 80
MIN_LOAD_FORCE_N = 1.0


@pytest.fixture(scope="module")
def gravity_play_env(crab_hex_play_env):
    """Second env in the same Kit session with gravity so feet load the ground."""
    _env_fixture, _get_robot, _resolve = crab_hex_play_env
    skip_unless_firmware_isaac()
    env = make_crab_hex_env(num_envs=1, with_gravity=True, fix_base=False)
    env.reset()
    yield env
    env.close()


@pytest.fixture(scope="module")
def gravity_robot(gravity_play_env):
    return get_robot_articulation(gravity_play_env)


def test_incoming_wrench_nonzero_when_leg_loads_floor(gravity_play_env, gravity_robot):
    from firmware.sim.robot_core import RobotCore
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    config = SimulatedRobotConfig.front_board_rev3()
    indices = resolve_front_board_joint_indices(gravity_robot, config)
    body_indices = resolve_wrench_body_indices(gravity_robot, config)

    core = RobotCore.front_board(
        env=gravity_play_env,
        articulation=gravity_robot,
        joint_index_for_name=indices,
        config=config,
        max_joint_speed=1.5,
    )
    core.reset()

    for _ in range(SETTLE_STEPS):
        advance_sim(gravity_play_env)

    baseline = max_actuator_wrench_force(sample_actuator_wrenches(gravity_robot, body_indices))

    core.jog("FLKL", 200)
    for _ in range(LOAD_JOG_STEPS):
        core.step()

    loaded = max_actuator_wrench_force(
        sample_actuator_wrenches(gravity_robot, body_indices),
        actuator_ids=("FLKL", "FLHL", "FLHY"),
    )

    assert loaded >= MIN_LOAD_FORCE_N, (
        f"expected leg load wrench >= {MIN_LOAD_FORCE_N} N, got {loaded} "
        f"(baseline max force {baseline})"
    )
