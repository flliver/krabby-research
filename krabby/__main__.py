"""krabby CLI entry point."""
from __future__ import annotations

import argparse
import sys

_VERSION = "0.1.2"


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
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to the container")

    # firmware
    p_firmware = sub.add_parser("firmware", help="Run krabby-firmware inside the container")
    p_firmware.add_argument("--image", metavar="REF", help="Image ref to use")
    p_firmware.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to krabby-firmware")

    args = parser.parse_args()

    if args.command == "install":
        from krabby.install import cmd_install
        cmd_install(image_ref=args.image)

    elif args.command == "update":
        from krabby.update import cmd_update
        cmd_update(image_ref=args.image)

    elif args.command == "run":
        from krabby.run import cmd_run
        cmd_run(image_ref=args.image, extra_args=args.args, entrypoint=args.entrypoint)

    elif args.command == "firmware":
        from krabby.firmware import cmd_firmware
        cmd_firmware(firmware_args=args.args, image_ref=args.image)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
