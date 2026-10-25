"""Firmware Isaac integration fixtures and pytest hooks.

Kit can only be booted once per process, so ``crab_hex_play_env`` is session scoped and
shared by every test module here.

When ``FIRMWARE_ISAAC_RESULTS_DIR`` is set (``run_firmware_isaac_pytest.sh`` in Docker),
each completed test call phase is appended to ``outcomes.jsonl`` on disk. The host reads
that file for pass/fail — do not rely on the container process exit code alone; Kit teardown
can exit 0 before pytest prints its summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
PARKOUR_TASKS = REPO / "parkour" / "parkour_tasks"
if str(PARKOUR_TASKS) not in sys.path:
    sys.path.insert(0, str(PARKOUR_TASKS))

_RESULTS_DIR: Path | None = None
_ISAAC_DEFERRED_TEARDOWN: list[Callable[[], None]] = []


def skip_unless_firmware_isaac() -> None:
    if os.environ.get("RUN_FIRMWARE_ISAAC", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Isaac firmware test (set RUN_FIRMWARE_ISAAC=1)")


@pytest.fixture(scope="session")
def crab_hex_play_env():
    """One headless Kit boot for the whole session."""
    skip_unless_firmware_isaac()
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

    env = make_crab_hex_env(num_envs=1, device=device)
    env.reset()

    def _close_sim() -> None:
        env.close()
        simulation_app.close()

    defer_isaac_teardown(_close_sim)
    yield env, get_robot_articulation, resolve_front_board_joint_indices


@pytest.fixture(scope="session")
def front_board_rig(crab_hex_play_env):
    """``RobotCore`` wired to the six FRONT joints, ready to jog."""
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
    return core, robot, config, indices


@pytest.fixture(scope="session")
def eighteen_joint_rig(crab_hex_play_env):
    """Three-board ``ThreeBoardRobotCore`` on all 18 revolute DOFs."""
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
    from firmware.sim.isaac_scene import configure_open_loop_velocity_drive, resolve_eighteen_joint_indices
    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
    from firmware.sim.three_board_coordinator import ThreeBoardRobotCore

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(f"{exc} (build on the host, it is mounted into the container)")

    env, get_robot, _resolve_front = crab_hex_play_env
    configs = ThreeBoardSimConfigs.rev3()
    robot = get_robot(env)
    indices = resolve_eighteen_joint_indices(robot, configs)
    configure_open_loop_velocity_drive(robot, indices.values())

    core = ThreeBoardRobotCore.eighteen_joints(
        env=env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=bridge,
        configs=configs,
        max_joint_speed=2.0,
    )
    core.reset()
    return core, robot, configs, indices


@pytest.fixture(scope="session")
def gravity_front_board(crab_hex_play_env):
    """FRONT ``RobotCore`` with gravity enabled on the shared session env."""
    skip_unless_firmware_isaac()
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
    from firmware.sim.isaac_scene import configure_open_loop_velocity_drive, set_crab_hex_gravity_enabled
    from firmware.sim.robot_core import RobotCore
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(f"{exc} (build on the host, it is mounted into the container)")

    env, get_robot, resolve_front = crab_hex_play_env
    robot = get_robot(env)
    set_crab_hex_gravity_enabled(env, robot, enabled=True)
    config = SimulatedRobotConfig.front_board_rev3()
    indices = resolve_front(robot, config)
    configure_open_loop_velocity_drive(robot, indices.values())
    core = RobotCore.front_board(
        env=env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=bridge,
        config=config,
        max_joint_speed=1.5,
    )
    yield core, robot, config
    set_crab_hex_gravity_enabled(env, robot, enabled=False)


def park_at_default_pose(core, robot) -> None:
    """Undo a previous drive so each case starts from the same joint state."""
    import torch

    # inference_mode: the articulation buffers were created inside it by the first env
    # step, and in-place writes to inference tensors are rejected outside that context.
    with torch.inference_mode():
        pos = robot.data.default_joint_pos.clone()
        vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(pos, vel)
        robot.set_joint_velocity_target(vel)
        robot.set_joint_position_target(pos)
    core.reset()


def defer_isaac_teardown(callback: Callable[[], None]) -> None:
    """Close Kit only after pytest finishes reporting (``simulation_app.close()`` kills the process)."""
    _ISAAC_DEFERRED_TEARDOWN.append(callback)


def _results_dir() -> Path | None:
    global _RESULTS_DIR
    if _RESULTS_DIR is not None:
        return _RESULTS_DIR
    raw = os.environ.get("FIRMWARE_ISAAC_RESULTS_DIR", "").strip()
    if not raw:
        return None
    _RESULTS_DIR = Path(raw)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _append_result(record: dict[str, Any]) -> None:
    directory = _results_dir()
    if directory is None:
        return
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with (directory / "outcomes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def pytest_collection_finish(session) -> None:
    directory = _results_dir()
    if directory is None:
        return
    meta = {
        "collected": len(session.items),
        "nodeids": [item.nodeid for item in session.items],
    }
    (directory / "collected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus: int) -> None:
    directory = _results_dir()
    if directory is not None:
        payload = {
            "pytest_exitstatus": int(exitstatus),
            "collected": len(session.items),
        }
        (directory / "session.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    while _ISAAC_DEFERRED_TEARDOWN:
        callback = _ISAAC_DEFERRED_TEARDOWN.pop()
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup after tests recorded
            _append_result({"nodeid": "isaac_teardown", "outcome": "error", "longrepr": str(exc)})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    _append_result(
        {
            "nodeid": item.nodeid,
            "outcome": report.outcome,
            "duration_s": report.duration,
            "longrepr": str(report.longrepr) if report.failed or report.skipped else None,
        }
    )
