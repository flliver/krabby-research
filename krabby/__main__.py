"""krabby CLI entry point."""
from __future__ import annotations

import argparse
import sys

_VERSION = "0.1.7"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="krabby",
        description="Install, update, and run the Krabby locomotion stack.",
    )
    parser.add_argument("--version", action="version", version=f"krabby {_VERSION}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # install
    p_install = sub.add_parser("install", help="Pull the locomotion image and set up the host")
    p_install.add_argument("--image", metavar="REF", help="Image ref to install (default: mainline-latest)")

    # update
    p_update = sub.add_parser("update", help="Pull a newer image")
    p_update.add_argument("--image", metavar="REF", help="Image ref to update to")

    # run
    p_run = sub.add_parser("run", help="Start the locomotion container")
    p_run.add_argument("--image", metavar="REF", help="Image ref to run")
    p_run.add_argument("--entrypoint", metavar="CMD", help="Override container entrypoint")
    p_run.add_argument("--gamepad-only", action="store_true", help="Run in gamepad-only mode (no inference checkpoint required)")
    p_run.add_argument("--mount", "-v", metavar="SRC:DST", action="append", dest="mounts", help="Extra volume mount (may be repeated)")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to the container")

    # firmware
    p_firmware = sub.add_parser("firmware", help="Run krabby-firmware inside the container")
    p_firmware.add_argument("--image", metavar="REF", help="Image ref to use")
    p_firmware.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to krabby-firmware")

    # uno
    p_uno = sub.add_parser("uno", help="Run the gamepad control-loop client inside the container")
    p_uno.add_argument("--image", metavar="REF", help="Image ref to use")
    p_uno.add_argument("args", nargs=argparse.REMAINDER, help="Extra args forwarded to krabby-uno")

    args = parser.parse_args()

    if args.command == "install":
        from krabby.install import cmd_install
        cmd_install(image_ref=args.image)

    elif args.command == "update":
        from krabby.update import cmd_update
        cmd_update(image_ref=args.image)

    elif args.command == "run":
        from krabby.run import cmd_run
        cmd_run(image_ref=args.image, extra_args=args.args, entrypoint=args.entrypoint, extra_mounts=args.mounts, gamepad_only=args.gamepad_only)

    elif args.command == "firmware":
        from krabby.firmware import cmd_firmware
        cmd_firmware(firmware_args=args.args, image_ref=args.image)

    elif args.command == "uno":
        from krabby.uno import cmd_uno
        cmd_uno(image_ref=args.image, extra_args=args.args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
