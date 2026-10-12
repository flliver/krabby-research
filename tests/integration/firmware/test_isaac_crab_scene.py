"""Opt-in Isaac Sim smoke test for firmware scene loading (single env, one step).

Preferred: run inside ``krabby-isaacsim`` (see ``images/isaacsim/README.md``)::

    ./images/isaacsim/run_firmware_isaac_pytest.sh
    make -C firmware test-firmware-isaac

Native Isaac Lab + ``/isaac-sim/python.sh`` (not plain venv python)::

    RUN_FIRMWARE_ISAAC=1 /path/to/isaac-sim/python.sh -m pytest tests/integration/firmware/test_isaac_crab_scene.py -v

Kit teardown is deferred to session end (``conftest.defer_isaac_teardown``) so pytest can
record outcomes before ``simulation_app.close()`` takes the process down.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PARKOUR_TASKS = REPO / "parkour" / "parkour_tasks"
if str(PARKOUR_TASKS) not in sys.path:
    sys.path.insert(0, str(PARKOUR_TASKS))

pytestmark = pytest.mark.isaacsim


def _skip_unless_firmware_isaac() -> None:
    if os.environ.get("RUN_FIRMWARE_ISAAC", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Isaac firmware scene test (set RUN_FIRMWARE_ISAAC=1)")


@pytest.fixture(scope="module")
def crab_hex_play_env():
    """One headless Kit boot for the module."""
    _skip_unless_firmware_isaac()
    pytest.importorskip("isaaclab")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    device = os.environ.get("KRABBY_ISAAC_DEVICE", "cuda:0")
    app_args = parser.parse_args(["--headless", f"--device={device}"])
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    from firmware.sim.isaac_scene import (
        get_robot_articulation,
        make_crab_hex_env,
        resolve_front_board_joint_indices,
    )
    from tests.integration.firmware.conftest import defer_isaac_teardown

    env = make_crab_hex_env(num_envs=1, device=device)
    env.reset()

    def _close_sim() -> None:
        env.close()
        simulation_app.close()

    defer_isaac_teardown(_close_sim)
    yield env, get_robot_articulation, resolve_front_board_joint_indices


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


def test_jog_drives_one_joint_and_advances_millis(crab_hex_play_env):
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
    from firmware.sim.isaac_scene import configure_open_loop_velocity_drive
    from firmware.sim.robot_core import RobotCore
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(f"{exc} (build on the host, it is mounted into the container)")

    env, get_robot, resolve_front = crab_hex_play_env
    config = SimulatedRobotConfig.front_board_rev3()
    robot = get_robot(env)
    indices = resolve_front(robot, config)
    # Without this the stock position gains (16000 N·m/rad on the hip joints) hold the
    # reset pose against any speed the firmware commands.
    configure_open_loop_velocity_drive(robot, indices.values())

    core = RobotCore.front_board(
        env=env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=bridge,
        config=config,
        max_joint_speed=2.0,
    )
    core.reset()

    jogged_joint = config.slot_by_firmware_id("FLHL").isaac_joint_name
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}

    core.jog("FLHL", 255)
    for _ in range(40):
        core.step()

    moved = {
        name: robot.data.joint_pos[0, idx].item() - before[name]
        for name, idx in indices.items()
    }
    assert core.millis() == 40 * core.firmware_step_ms
    assert abs(moved[jogged_joint]) > 0.05, f"jogged joint did not move: {moved}"
    # The rest of the FL leg swings with the driven joint; the FR leg has no command.
    for name, delta in moved.items():
        if name.startswith("FR_"):
            assert abs(delta) < 0.01, f"{name} moved without a jog: {moved}"
