"""CLI tests for Isaac HAL server command backend selection."""

import argparse

import pytest

from hal.server.isaac.command_backend_cli import (
    add_command_backend_arguments,
    validate_command_backend_args,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_command_backend_arguments(parser)
    return parser


def test_backend_defaults_to_direct():
    args = _parser().parse_args([])
    assert args.backend == "direct"
    assert args.firmware_transport == "in_process"


def test_backend_fullsim_and_firmware_transport():
    args = _parser().parse_args(
        ["--backend", "fullsim", "--firmware-transport", "serial_sdk"]
    )
    assert args.backend == "fullsim"
    assert args.firmware_transport == "serial_sdk"


def test_firmware_transport_can_choice():
    args = _parser().parse_args(
        ["--backend", "fullsim", "--firmware-transport", "can"]
    )
    assert args.firmware_transport == "can"


def test_firmware_transport_can_requires_fullsim_backend():
    parser = _parser()
    args = parser.parse_args(["--firmware-transport", "can"])
    assert args.backend == "direct"
    with pytest.raises(SystemExit):
        validate_command_backend_args(args, parser)
