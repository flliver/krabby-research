"""Opt-in Isaac Sim smoke test for firmware scene loading (single env, one step).

Preferred: run inside ``krabby-isaacsim`` (see ``images/isaacsim/README.md``)::

    ./images/isaacsim/run_firmware_isaac_pytest.sh
    make -C firmware test-firmware-isaac

Native Isaac Lab + ``/isaac-sim/python.sh`` (not plain venv python)::

    RUN_FIRMWARE_ISAAC=1 /path/to/isaac-sim/python.sh -m pytest tests/integration/firmware -v

Kit teardown is deferred to session end (``conftest.defer_isaac_teardown``) so pytest can
record outcomes before ``simulation_app.close()`` takes the process down.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.isaacsim


def test_crab_hex_scene_steps_once_and_resolves_front_joints(crab_hex_play_env):
    from firmware.sim.isaac_scene import step_env_once
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    env, get_robot, resolve_front = crab_hex_play_env
    step_env_once(env)

    config = SimulatedRobotConfig.front_board_rev3()
    robot = get_robot(env)
    indices = resolve_front(robot, config)
    assert len(indices) == 6
    assert set(indices) == set(config.isaac_joint_names())
    for name in config.isaac_joint_names():
        assert name in indices
        assert indices[name] >= 0


def test_jog_drives_one_joint_and_advances_millis(front_board_rig):
    from tests.integration.firmware.conftest import park_at_default_pose

    core, robot, config, indices = front_board_rig
    park_at_default_pose(core, robot)

    jogged_joint = config.slot_by_firmware_id("FLHL").isaac_joint_name
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}

    core.jog("FLHL", 255)
    for _ in range(40):
        core.step()

    moved = {
        name: robot.data.joint_pos[0, idx].item() - before[name]
        for name, idx in indices.items()
    }
    from firmware.sim.isaac_scene import physics_step_ms

    assert core.millis() == 40 * physics_step_ms(core.env)
    assert abs(moved[jogged_joint]) > 0.05, f"jogged joint did not move: {moved}"
    # The rest of the FL leg swings with the driven joint, and the FR leg drifts a little
    # as the body reacts, so the uncommanded joints are bounded against the driven one
    # rather than against zero.
    for name, delta in moved.items():
        if name.startswith("FR_"):
            assert abs(delta) * 5 < abs(moved[jogged_joint]), (
                f"{name} moved without a jog: {moved}"
            )
