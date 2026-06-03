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
    p_run = sub.add_parser("run", help="Start the locomotion stack (HAL server + gamepad client + controller)")
    p_run.add_argument("--image", metavar="REF", help="Image ref to run")
    p_run.add_argument("--entrypoint", metavar="CMD", help="Override container entrypoint (inference/custom path)")
    p_run.add_argument("--gamepad-only", action="store_true", help="Explicitly launch the gamepad stack (same as the default `krabby run`)")
    p_run.add_argument("--mount", "-v", metavar="SRC:DST", action="append", dest="mounts", help="Extra volume mount (may be repeated)")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="In gamepad mode: args for krabby-uno. With `-- --checkpoint ...`: inference args for the HAL server.")

    # firmware
    p_firmware = sub.add_parser("firmware", help="Run krabby-firmware inside the container")
    p_firmware.add_argument("--image", metavar="REF", help="Image ref to use")
    p_firmware.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to krabby-firmware")

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

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
