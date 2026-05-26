"""Tests for the --control-source CLI interface in hal.server.jetson.main.

hal.server.jetson.main cannot be imported in the unit-test environment
(numpy/zmq/pyzed not installed).  These tests build a standalone argparse
that mirrors the argument definitions in main() and verify the expected
--control-source choices, bind defaults, and robot choices.

If main.py's argparse is changed these tests act as a spec reminder.
"""
from __future__ import annotations

import argparse

import pytest


def _build_parser() -> argparse.ArgumentParser:
    """Replicate the --control-source related arguments from hal.server.jetson.main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--control-source",
        default="portal",
        choices=["portal", "inference", "gamepad"],
    )
    parser.add_argument("--observation-bind", default="tcp://*:6001")
    parser.add_argument("--command-bind", default="tcp://*:6002")
    parser.add_argument("--robot", default="hex", choices=["hex", "go2"])
    parser.add_argument("--teleop", action="store_true")
    return parser


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

    def test_observation_bind_default(self):
        args = _build_parser().parse_args([])
        assert args.observation_bind == "tcp://*:6001"

    def test_command_bind_default(self):
        args = _build_parser().parse_args([])
        assert args.command_bind == "tcp://*:6002"

    def test_robot_default_is_hex(self):
        args = _build_parser().parse_args([])
        assert args.robot == "hex"

    def test_go2_robot_is_valid(self):
        args = _build_parser().parse_args(["--control-source", "gamepad", "--robot", "go2"])
        assert args.robot == "go2"

    def test_gamepad_does_not_require_checkpoint(self):
        args = _build_parser().parse_args(["--control-source", "gamepad"])
        assert args.checkpoint is None
