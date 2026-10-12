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
