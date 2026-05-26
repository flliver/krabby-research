"""Tests for the --control-source CLI interface in hal.server.jetson.main.

hal.server.jetson.main cannot be imported in the unit-test environment
(numpy/zmq/pyzed not installed). These tests build a standalone argparse
that mirrors the argument definitions in main() and verify the expected
--control-source choices, bind defaults, and robot choices.

If main.py's argparse is changed these tests act as a spec reminder.
"""
from __future__ import annotations

import argparse

import pytest


INPROC_OBSERVATION_ENDPOINT = "inproc://hal_observation"
INPROC_COMMAND_ENDPOINT = "inproc://hal_commands"


def _build_parser() -> argparse.ArgumentParser:
    """Replicate the --control-source related arguments from hal.server.jetson.main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--control-source",
        default="portal",
        choices=["portal", "inference", "gamepad"],
    )
    parser.add_argument("--observation-bind", default=None)
    parser.add_argument("--command-bind", default=None)
    parser.add_argument("--robot", default="hex", choices=["hex", "go2"])
    parser.add_argument("--teleop", action="store_true")
    return parser


def _resolve_hal_bind(args: argparse.Namespace) -> tuple[str, str]:
    """Mirror the HAL endpoint resolution in main(): gamepad → TCP, otherwise inproc."""
    if args.control_source == "gamepad":
        obs = args.observation_bind or "tcp://*:6001"
        cmd = args.command_bind or "tcp://*:6002"
    else:
        obs = args.observation_bind or INPROC_OBSERVATION_ENDPOINT
        cmd = args.command_bind or INPROC_COMMAND_ENDPOINT
    return obs, cmd


class TestControlSourceArgparse:
    def test_gamepad_is_valid(self):
        args = _build_parser().parse_args(["--control-source", "gamepad"])
        assert args.control_source == "gamepad"

    def test_portal_is_valid(self):
        args = _build_parser().parse_args(["--control-source", "portal"])
        assert args.control_source == "portal"

    def test_inference_is_valid(self):
        args = _build_parser().parse_args(["--control-source", "inference"])
        assert args.control_source == "inference"

    def test_invalid_choice_raises(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--control-source", "joystick"])

    def test_default_control_source_is_portal(self):
        args = _build_parser().parse_args([])
        assert args.control_source == "portal"

    def test_robot_default_is_hex(self):
        args = _build_parser().parse_args([])
        assert args.robot == "hex"

    def test_go2_robot_is_valid(self):
        args = _build_parser().parse_args(["--control-source", "gamepad", "--robot", "go2"])
        assert args.robot == "go2"

    def test_gamepad_does_not_require_checkpoint(self):
        args = _build_parser().parse_args(["--control-source", "gamepad"])
        assert args.checkpoint is None


class TestHalBindResolution:
    """The HAL bind URI is the only HAL config that changes between control sources."""

    def test_portal_defaults_to_inproc(self):
        args = _build_parser().parse_args(["--control-source", "portal"])
        obs, cmd = _resolve_hal_bind(args)
        assert obs == INPROC_OBSERVATION_ENDPOINT
        assert cmd == INPROC_COMMAND_ENDPOINT

    def test_inference_defaults_to_inproc(self):
        args = _build_parser().parse_args(["--control-source", "inference"])
        obs, cmd = _resolve_hal_bind(args)
        assert obs == INPROC_OBSERVATION_ENDPOINT
        assert cmd == INPROC_COMMAND_ENDPOINT

    def test_gamepad_defaults_to_tcp(self):
        args = _build_parser().parse_args(["--control-source", "gamepad"])
        obs, cmd = _resolve_hal_bind(args)
        assert obs == "tcp://*:6001"
        assert cmd == "tcp://*:6002"

    def test_explicit_observation_bind_overrides_default(self):
        args = _build_parser().parse_args(
            ["--control-source", "gamepad", "--observation-bind", "tcp://*:7001"]
        )
        obs, _ = _resolve_hal_bind(args)
        assert obs == "tcp://*:7001"

    def test_explicit_command_bind_overrides_default(self):
        args = _build_parser().parse_args(
            ["--control-source", "inference", "--command-bind", "tcp://127.0.0.1:7002"]
        )
        _, cmd = _resolve_hal_bind(args)
        assert cmd == "tcp://127.0.0.1:7002"
