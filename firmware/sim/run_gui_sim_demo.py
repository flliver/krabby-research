"""Headless Isaac MCU host for manual ``python -m firmware.gui`` current-sense demos."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def _require_isaac_env() -> None:
    if os.environ.get("RUN_FIRMWARE_ISAAC", "").lower() not in ("1", "true", "yes"):
        print(
            "Set RUN_FIRMWARE_ISAAC=1 (Isaac Lab + GPU). "
            "See firmware/gui/CURRENT_SENSE_SIM_DEMO.md",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac sim MCU on a PTY for firmware GUI / KrabbyMCUSDK demos",
    )
    parser.add_argument(
        "--joint",
        default="FLHL",
        help="Joint name called out in the demo checklist (default FLHL)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("KRABBY_ISAAC_DEVICE", "cuda:0"),
        help="Isaac device (default KRABBY_ISAAC_DEVICE or cuda:0)",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Enable gravity (contact current-sense band ~4)",
    )
    args = parser.parse_args()
    _require_isaac_env()

    pytest = __import__("pytest")
    pytest.importorskip("isaaclab")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args(["--headless", f"--device={args.device}"])
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
    from firmware.sim.isaac_mcu_host import IsaacMcuHost
    from firmware.sim.isaac_scene import (
        configure_open_loop_velocity_drive,
        get_robot_articulation,
        make_crab_hex_env,
        resolve_front_board_joint_indices,
    )
    from firmware.sim.robot_core import RobotCore
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    env = make_crab_hex_env(
        num_envs=1,
        device=args.device,
        with_gravity=args.gravity,
        fix_base=False,
    )
    env.reset()
    robot = get_robot_articulation(env)
    config = SimulatedRobotConfig.front_board_rev3()
    indices = resolve_front_board_joint_indices(robot, config)
    configure_open_loop_velocity_drive(robot, indices.values())
    core = RobotCore.front_board(
        env=env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=bridge,
        config=config,
        max_joint_speed=2.0,
    )

    host = IsaacMcuHost(core)
    port = host.start()

    stop = False

    def _handle_sig(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print("Simulated MCU ready (FRONT board, Isaac current-sense enabled).")
    print(f"  PTY port: {port}")
    print(f"  Demo joint: {args.joint}")
    print(f"  Gravity: {args.gravity}")
    print()
    print("  python -m firmware.gui --port", port)
    print()
    print("Checklist: firmware/gui/CURRENT_SENSE_SIM_DEMO.md")
    print("Ctrl+C to stop.")

    try:
        while not stop:
            time.sleep(0.2)
    finally:
        host.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
