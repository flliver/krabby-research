"""Windowed viewer: jog one FRONT actuator, print wrench + mapped ``pinIS``.

Run from repo root after ``make -C firmware host-actuator``::

    python -m firmware.sim.wrench_tuning_viewer

Keys (terminal focus): ``e`` / ``r`` extend / retract jog, space coast, ``q`` quit.
Edit breakpoints in ``firmware/sim/current_sense_mapping.py`` (``CurrentSenseBreakpoints``).
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from typing import TextIO

import os

from firmware.sim.current_sense_mapping import (
    effective_load_newtons,
    map_wrench_to_pin_is_adc,
)
from firmware.sim.effort_readback import read_applied_joint_torque_nm
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.isaac_scene import (
    configure_open_loop_velocity_drive,
    get_robot_articulation,
    make_crab_hex_env,
    resolve_front_board_joint_indices,
)
from firmware.sim.robot_core import RobotCore
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from firmware.sim.wrench_readback import (
    read_body_incoming_wrench,
    resolve_wrench_body_indices,
)


def _format_status_line(
    actuator_id: str,
    load_n: float,
    mapped_adc: int,
    shim_adc: int | None,
    joint_vel: float,
    en_high: bool,
    *,
    wrench_load_n: float | None = None,
    effort_load_n: float | None = None,
) -> str:
    en = "EN" if en_high else "en"
    shim = f" shim={shim_adc}" if shim_adc is not None else ""
    if wrench_load_n is not None and effort_load_n is not None:
        load_detail = (
            f"load={load_n:7.2f}N (wrench={wrench_load_n:.2f} effort={effort_load_n:.2f})"
        )
    else:
        load_detail = f"load={load_n:7.2f}N"
    return (
        f"{actuator_id}  {load_detail}  map_adc={mapped_adc:4d}{shim}  "
        f"dq={joint_vel:+.3f} rad/s  {en}"
    )


def _poll_jog_pwm(stdin: TextIO, jog_pwm: int) -> int | None:
    """Return jog PWM, 0 for coast, or None to quit. Non-blocking."""
    if not select.select([stdin], [], [], 0)[0]:
        return 0
    ch = stdin.read(1)
    if ch in ("q", "Q", "\x03"):
        return None
    if ch in ("e", "E"):
        return jog_pwm
    if ch in ("r", "R"):
        return -jog_pwm
    if ch == " ":
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Live wrench + mapped current-sense ADC for tuning.")
    parser.add_argument(
        "--actuator",
        type=str,
        default="FLKL",
        help="FRONT-board actuator id (FLHY, FLHL, FLKL, FRHY, FRHL, FRKL).",
    )
    parser.add_argument(
        "--jog-pwm",
        type=int,
        default=180,
        help="PWM magnitude while e/r held (signed via drive direction).",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=80,
        help="Physics steps after reset before interactive loop.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print status every N simulation steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Isaac device (default: env KRABBY_ISAAC_DEVICE or cuda:0).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    device = args.device or os.environ.get("KRABBY_ISAAC_DEVICE", "cuda:0")
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        print(exc, file=sys.stderr)
        print("Build with: make -C firmware host-actuator", file=sys.stderr)
        return 1

    config = SimulatedRobotConfig.front_board_rev3()
    config.slot_by_firmware_id(args.actuator)

    env = make_crab_hex_env(num_envs=1, device=device, with_gravity=True, fix_base=False)
    env.reset()
    robot = get_robot_articulation(env)
    indices = resolve_front_board_joint_indices(robot, config)
    configure_open_loop_velocity_drive(robot, indices.values())
    body_indices = resolve_wrench_body_indices(robot, config)

    core = RobotCore.front_board(
        env=env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=bridge,
        config=config,
        max_joint_speed=1.5,
    )
    core.reset()
    for _ in range(args.settle_steps):
        core.step()

    actuator_id = args.actuator
    body_index = body_indices[actuator_id]
    slot = config.slot_by_firmware_id(actuator_id)
    _, _, pin_en = slot.drive_pins
    joint_index = indices[slot.isaac_joint_name]

    print(
        "Wrench / current-sense tuning viewer. Focus this terminal for keys: "
        "e=extend r=retract space=coast q=quit",
        flush=True,
    )

    stdin = sys.stdin
    old_tty = termios.tcgetattr(stdin)
    step = 0
    jog_pwm = 0
    try:
        tty.setcbreak(stdin.fileno())
        while simulation_app.is_running():
            polled = _poll_jog_pwm(stdin, args.jog_pwm)
            if polled is None:
                break
            jog_pwm = polled
            core.jog(actuator_id, jog_pwm)
            core.step()
            step += 1

            if step % args.print_every != 0:
                simulation_app.update()
                continue

            wrench = read_body_incoming_wrench(robot, body_index, env_index=0)
            vel = float(robot.data.joint_vel[0, joint_index].item())
            en_high = core.bridge.pin_digital(pin_en) != 0
            torque_nm = read_applied_joint_torque_nm(robot, joint_index, env_index=0)
            load_n, wrench_n, effort_n = effective_load_newtons(wrench, torque_nm)
            mapped = map_wrench_to_pin_is_adc(
                wrench,
                en_energized=en_high,
                joint_velocity_rad_s=vel,
                applied_torque_nm=torque_nm,
            )
            shim_adc = core.bridge.analog_read(slot.pin_is)
            line = _format_status_line(
                actuator_id,
                load_n,
                mapped,
                shim_adc,
                vel,
                en_high,
                wrench_load_n=wrench_n,
                effort_load_n=effort_n,
            )
            print(line, flush=True)
            simulation_app.update()
    finally:
        termios.tcsetattr(stdin, termios.TCSADRAIN, old_tty)
        core.jog(actuator_id, 0)
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
