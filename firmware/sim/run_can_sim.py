"""Headless Isaac loop: six virtual STM boards on SocketCAN (default vcan0)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def _require_isaac_env() -> None:
    if os.environ.get("RUN_FIRMWARE_ISAAC", "").lower() not in ("1", "true", "yes"):
        print(
            "Set RUN_FIRMWARE_ISAAC=1 (Isaac Lab + GPU). See firmware/stm/docs/VIRTUAL_CAN.md",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run six krabby_stm_host leg boards on a CAN-FD bus with Isaac physics",
    )
    parser.add_argument(
        "--bus",
        default=os.environ.get("KRABBY_CAN_IFACE", "vcan0"),
        help="SocketCAN interface (default KRABBY_CAN_IFACE or vcan0)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("KRABBY_ISAAC_DEVICE", "cuda:0"),
        help="Isaac device (default KRABBY_ISAAC_DEVICE or cuda:0)",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=1.0,
        help="Duty to rad/s scale for open-loop velocity drive",
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

    from firmware.sim.can_board_bridge import StmHostLibraryNotBuiltError, load_stm_board_bridge
    from firmware.sim.can_robot_core import CanRobotCore
    from firmware.sim.isaac_scene import (
        configure_open_loop_velocity_drive,
        get_robot_articulation,
        make_crab_hex_env,
        resolve_eighteen_joint_indices,
    )
    from firmware.stm.can_transport import open_socketcan_fd

    try:
        bridge = load_stm_board_bridge()
    except StmHostLibraryNotBuiltError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    env = make_crab_hex_env(num_envs=1, device=args.device, with_gravity=False, fix_base=False)
    env.reset()
    robot = get_robot_articulation(env)
    joint_map = resolve_eighteen_joint_indices(robot)
    configure_open_loop_velocity_drive(robot, list(joint_map.values()))

    bus = open_socketcan_fd(args.bus)
    stop = False

    def _on_signal(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    core = CanRobotCore.from_isaac(
        env,
        robot,
        joint_map,
        bridge=bridge,
        bus=bus,
        max_joint_speed=args.max_joint_speed,
    )
    core.reset()
    print(f"CAN STM sim on {args.bus!r} (Ctrl+C to exit)", flush=True)
    try:
        while simulation_app.is_running() and not stop:
            core.step()
            time.sleep(0.001)
    finally:
        bus.shutdown()

    simulation_app.close()


if __name__ == "__main__":
    main()