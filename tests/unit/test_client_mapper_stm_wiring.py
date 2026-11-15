"""Client wiring: HAL --stm must match mapper stm=True on command publishers."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.parkour_types import InferenceResponse
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from hal.client.config import HalClientConfig
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


class TestCliUnoStmFlag:
    def test_krabby_uno_help_lists_stm(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            [sys.executable, "-m", "controller.cli.cli_uno", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--stm" in proc.stdout

    def test_parser_accepts_stm(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--observation_endpoint", default="tcp://localhost:6001")
        parser.add_argument("--command_endpoint", default="tcp://localhost:6002")
        parser.add_argument("--device-id", default=None)
        parser.add_argument("--InputController", default=None, dest="input_controller_id")
        parser.add_argument("--rate", type=float, default=50.0)
        parser.add_argument("--robot", choices=("hex", "go2"), default="hex")
        parser.add_argument("--stm", action="store_true")
        ns = parser.parse_args(["--stm"])
        assert ns.stm is True


@patch("controller.control_loop.GamepadToKrabbyHALMapper")
@patch("controller.control_loop.HalClient")
@patch("controller.control_loop.InputController")
class TestControlLoopStmWiring:
    def test_krabby_mode_passes_stm_to_mapper(
        self,
        mock_input_cls: MagicMock,
        mock_hal_cls: MagicMock,
        mock_mapper_cls: MagicMock,
    ):
        mock_input = MagicMock()
        mock_input_cls.get_instance.return_value = mock_input

        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_KRABBY,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://127.0.0.1:6001",
                command_endpoint="tcp://127.0.0.1:6002",
            ),
            krabby_gamepad_robot_definition=KRABBY_HEX_DEFINITION,
            stm=True,
        )
        loop = ControlLoop(config)
        loop.start()
        try:
            _, kwargs = mock_mapper_cls.call_args
            assert kwargs["stm"] is True
        finally:
            loop.stop()

    def test_webrtc_mode_passes_stm_to_mapper(
        self,
        mock_input_cls: MagicMock,
        mock_hal_cls: MagicMock,
        mock_mapper_cls: MagicMock,
    ):
        from teleop.edge.webrtc_input_controller import WebRTCInputController

        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_WEBRTC,
            webrtc_input_controller=WebRTCInputController(),
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://127.0.0.1:6001",
                command_endpoint="tcp://127.0.0.1:6002",
            ),
            krabby_gamepad_robot_definition=KRABBY_HEX_DEFINITION,
            stm=True,
        )
        loop = ControlLoop(config)
        loop.start()
        try:
            _, kwargs = mock_mapper_cls.call_args
            assert kwargs["stm"] is True
        finally:
            loop.stop()


class TestParkourInferenceClientStm:
    def test_constructor_stores_stm_flag(self):
        client = ParkourInferenceClient(
            hal_client_config=HalClientConfig(
                observation_endpoint="inproc://x",
                command_endpoint="inproc://y",
            ),
            model_weights=MagicMock(),
            observation_dimensions=MagicMock(),
            robot_definition=KRABBY_HEX_DEFINITION,
            stm=True,
        )
        assert client._stm is True

    def test_stm_mapper_clamps_via_client_path_mapper(self):
        action = torch.tensor([2.0, -3.0] + [0.0] * 16, dtype=torch.float32)
        response = InferenceResponse.create_success(action=action, timing_breakdown=[])
        mapper = ParkourLocomotionToHWMapper(KRABBY_HEX_DEFINITION, stm=True)
        cmd = mapper.map(response, observation_timestamp_ns=0)
        d = cmd.to_positions_dict()
        names = KRABBY_HEX_DEFINITION.get_joint_names()
        assert d[names[0]] == pytest.approx(1.0)
        assert d[names[1]] == pytest.approx(-1.0)