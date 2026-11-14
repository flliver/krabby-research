"""CLI entry for krabby-firmware-stm console script."""

from __future__ import annotations

import argparse
import sys

from firmware.stm import cli_stm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krabby-firmware-stm",
        description="Leg-board CAN-FD CLI (SET_DUTY, calibrate, telemetry).",
    )
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="Use vcan0 (sim boards) instead of can0+can1.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_duty = sub.add_parser("duty", help="Send duties for one leg (-1..1 per joint).")
    p_duty.add_argument("leg_id", type=int, help="Leg id 1..6")
    p_duty.add_argument("y", type=float, help="Hip yaw duty")
    p_duty.add_argument("h", type=float, help="Hip pitch duty")
    p_duty.add_argument("k", type=float, help="Knee duty")

    p_cal = sub.add_parser("calibrate", help="Start on-board stall cal for one joint.")
    p_cal.add_argument("leg_id", type=int)
    p_cal.add_argument("joint", type=int, help="Joint id 2 or 3")

    p_telem = sub.add_parser("telemetry", help="Print latest decoded RX after a short wait.")
    p_telem.add_argument(
        "--wait",
        type=float,
        default=2.0,
        help="Seconds to listen before printing (default 2).",
    )
    p_telem.add_argument("--json", action="store_true", help="Emit JSON snapshot.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = cli_stm._COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"unknown command {args.command!r}")
    try:
        return handler(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"CAN error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())