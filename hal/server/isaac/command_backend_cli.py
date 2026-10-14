"""Isaac HAL server CLI flags for direct vs fullsim command backends."""

from __future__ import annotations

import argparse

__all__ = ["add_command_backend_arguments", "validate_command_backend_args"]


def add_command_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        type=str,
        default="direct",
        choices=("direct", "fullsim"),
        help=(
            "Joint command path: direct (IsaacSimMCUSDK passthrough, default) or "
            "fullsim (three-board host firmware on this env)."
        ),
    )
    parser.add_argument(
        "--firmware-transport",
        type=str,
        default="in_process",
        choices=("in_process", "serial_sdk"),
        dest="firmware_transport",
        help=(
            "When --backend fullsim: in_process (host jog, default) or serial_sdk "
            "(KrabbyMCUSDK over PTY, same wire protocol as Jetson)."
        ),
    )


def validate_command_backend_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.firmware_transport != "in_process" and args.backend != "fullsim":
        parser.error("--firmware-transport requires --backend fullsim")
