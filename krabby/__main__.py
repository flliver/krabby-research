"""krabby CLI entry point."""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from krabby._state import DEFAULT_TAG


def _version() -> str:
    """Version for `krabby --version`, read from the installed package metadata.

    The release tag already sets the version in pyproject at build time, which lands
    in the wheel metadata — so reading it here means there's no second place to bump
    and it can never drift from the published version. Falls back to a dev marker when
    run from an uninstalled source tree.
    """
    try:
        return _pkg_version("krabby-launcher")
    except PackageNotFoundError:
        return "0+unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="krabby",
        description="Install, update, and run the Krabby locomotion stack.",
    )
    parser.add_argument("--version", action="version", version=f"krabby {_version()}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # install
    p_install = sub.add_parser("install", help="Pull the locomotion image and set up the host")
    p_install.add_argument("--image", metavar="REF", help=f"Image ref to install (default: {DEFAULT_TAG})")
    p_install.add_argument(
        "--launch-on-startup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start `krabby run` on boot via a systemd unit; pass --no-launch-on-startup to skip",
    )

    # update
    p_update = sub.add_parser("update", help="Pull a newer image")
    p_update.add_argument("--image", metavar="REF", help="Image ref to update to")

    # run
    p_run = sub.add_parser(
        "run",
        help="Start locomotion (fleet-enrolled: HAL + teleop to agent shim; else gamepad stack)",
    )
    p_run.add_argument("--image", metavar="REF", help="Image ref to run")
    p_run.add_argument("--entrypoint", metavar="CMD", help="Override container entrypoint (inference/custom path)")
    p_run.add_argument("--gamepad-only", action="store_true", help="Explicitly launch the gamepad stack (same as the default `krabby run`)")
    p_run.add_argument("--mount", "-v", metavar="SRC:DST", action="append", dest="mounts", help="Extra volume mount (may be repeated)")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="In gamepad mode: args for krabby-uno. With `-- --checkpoint ...`: inference args for the HAL server.")

    # firmware
    p_firmware = sub.add_parser("firmware", help="Run krabby-firmware inside the container")
    p_firmware.add_argument("--image", metavar="REF", help="Image ref to use")
    p_firmware.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to krabby-firmware")

    # enroll
    p_enroll = sub.add_parser("enroll", help="One-time fleet onboarding: provision IoT identity and enable krabby-agent")
    p_enroll.add_argument("--thing-name", metavar="NAME", help="IoT thing name (default: wired MAC address)")
    p_enroll.add_argument("--endpoint", metavar="HOST", help="IoT Core ATS endpoint (default: resolved from account)")
    p_enroll.add_argument(
        "--locomotion-control-source",
        choices=("portal", "inference"),
        help="Fleet HAL mode written to /etc/krabby/locomotion.json (default: portal)",
    )
    p_enroll.add_argument(
        "--locomotion-robot",
        choices=("hex", "go2"),
        help="Robot definition for fleet locomotion (default: hex)",
    )
    p_enroll.add_argument(
        "--locomotion-checkpoint",
        metavar="PATH",
        help="Container checkpoint path (required when --locomotion-control-source inference)",
    )
    p_enroll.add_argument(
        "--locomotion-checkpoint-host-dir",
        metavar="DIR",
        help="Host directory mounted at /workspace/checkpoints",
    )
    p_enroll.add_argument(
        "--locomotion-teleop-control-echo",
        action="store_true",
        help="Set teleop_control_echo in locomotion.json (bench E2E control ack; default off)",
    )

    # agent
    sub.add_parser("agent", help="Run the always-on IoT Core MQTT client (normally started by krabby-agent.service)")

    # get telemetry
    p_get = sub.add_parser("get", help="Read local device state")
    p_get_sub = p_get.add_subparsers(dest="get_command", metavar="<resource>")
    p_get_sub.add_parser(
        "telemetry",
        help="Print the fleet telemetry snapshot (same JSON `krabby agent` publishes to the shadow)",
    )

    args = parser.parse_args()

    if args.command == "install":
        from krabby.install import cmd_install
        cmd_install(image_ref=args.image, launch_on_startup=args.launch_on_startup)

    elif args.command == "update":
        from krabby.update import cmd_update
        cmd_update(image_ref=args.image)

    elif args.command == "run":
        from krabby.run import cmd_run
        cmd_run(image_ref=args.image, extra_args=args.args, entrypoint=args.entrypoint, extra_mounts=args.mounts, gamepad_only=args.gamepad_only)

    elif args.command == "firmware":
        from krabby.firmware import cmd_firmware
        cmd_firmware(firmware_args=args.args, image_ref=args.image)

    elif args.command == "enroll":
        from krabby.enroll import cmd_enroll
        cmd_enroll(
            thing_name=args.thing_name,
            endpoint=args.endpoint,
            locomotion_control_source=args.locomotion_control_source,
            locomotion_robot=args.locomotion_robot,
            locomotion_checkpoint=args.locomotion_checkpoint,
            locomotion_checkpoint_host_dir=args.locomotion_checkpoint_host_dir,
            locomotion_teleop_control_echo=args.locomotion_teleop_control_echo,
        )

    elif args.command == "agent":
        from krabby.agent import cmd_agent
        cmd_agent()

    elif args.command == "get":
        if args.get_command == "telemetry":
            from krabby.telemetry import cmd_get_telemetry
            cmd_get_telemetry()
        else:
            parser.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
