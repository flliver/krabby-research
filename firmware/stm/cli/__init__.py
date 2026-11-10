"""CLI entry for optional krabby-firmware-stm console script."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="krabby-firmware-stm",
        description=(
            "Leg-board CAN CLI. Prefer krabby-firmware --stm when both transports are installed."
        ),
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="help",
        choices=("show", "provision", "jog", "telemetry", "update", "help"),
    )
    args = parser.parse_args(argv)
    if args.subcommand == "help":
        print("Use: krabby-firmware --stm <show|provision|jog|telemetry|update>")
        print("Subcommands are not implemented yet.")
        return 0
    from firmware.stm import cli_stm

    cli_stm.not_implemented(args.subcommand)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
