"""Unit tests for ControlLoop.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/controller/test_control_loop.py -v

    # Run a specific test class
    python -m pytest tests/unit/controller/test_control_loop.py::TestControlLoopInitialization -v

    # Run a specific test
    python -m pytest tests/unit/controller/test_control_loop.py::TestControlLoopInitialization::test_initialization -v

    # Run with coverage
    python -m pytest tests/unit/controller/test_control_loop.py --cov=controller.control_loop --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/controller/test_control_loop.py -v -s


Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov

"""

import numpy as np
import pytest

from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from controller.input.state import ControllerState
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import JointCommand
from hal.server import HalServerBase, HalServerConfig
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


class TestControlLoopInitialization:
    """Test ControlLoop initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
            isaacsim_robot_definition=KRABBY_HEX_DEFINITION,
        )
        
        loop = ControlLoop(config)
        
        assert loop.config is config
        assert loop._running is False
        assert loop._thread is None
        assert loop._input_controller is None
        assert loop._hal_client is None
        assert loop._gamepad_to_isaacsim_hal_mapper is None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration values."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            input_controller_device_id=0,
            input_controller_update_rate_hz=100.0,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
            mapper_hip_up_down_scale=0.5,
            mapper_knee_out_in_scale=0.4,
            mapper_hip_yaw_scale=0.3,
            isaacsim_robot_definition=KRABBY_HEX_DEFINITION,
        )

        loop = ControlLoop(config)

        assert loop.config.input_controller_device_id == 0
        assert loop.config.input_controller_update_rate_hz == 100.0
        assert loop.config.mapper_hip_up_down_scale == 0.5
        assert loop.config.mapper_knee_out_in_scale == 0.4
        assert loop.config.mapper_hip_yaw_scale == 0.3


class TestControlLoopStartStop:
    """Test ControlLoop start/stop lifecycle."""

    def test_stop_when_not_running(self):
        """Test that stop() is safe when not running."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
            isaacsim_robot_definition=KRABBY_HEX_DEFINITION,
        )
        
        loop = ControlLoop(config)
        
        # Should not raise error
        loop.stop()
        loop.stop()  # Call again


class TestControlLoopErrorHandling:
    """Test error handling in ControlLoop."""

    def setup_method(self):
        """Set up test fixtures."""
        from controller.input.input_controller import InputController
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None

    def teardown_method(self):
        """Clean up after tests."""
        from controller.input.input_controller import InputController
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None

    def test_start_without_hal_client_config(self):
        """Test that start() raises ValueError when hal_client_config is missing."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=None,
        )
        
        loop = ControlLoop(config)
        
        with pytest.raises(ValueError, match="hal_client_config is required"):
            loop.start()

    def test_start_with_unknown_mode(self):
        """Test that start() raises ValueError for unknown mode."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
            isaacsim_robot_definition=KRABBY_HEX_DEFINITION,
        )
        # Manually set an invalid mode
        config.mode = "invalid_mode"  # type: ignore
        
        loop = ControlLoop(config)
        
        with pytest.raises(ValueError, match="Unknown control mode"):
            loop.start()

    def test_start_model_controller_krabby_not_implemented(self):
        """Test that MODEL_CONTROLLER_KRABBY mode raises NotImplementedError."""
        config = ControlLoopConfig(
            mode=ControlMode.MODEL_CONTROLLER_KRABBY,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
        )

        loop = ControlLoop(config)

        with pytest.raises(NotImplementedError, match="MODEL_CONTROLLER_KRABBY mode not yet implemented"):
            loop.start()

    def test_start_krabby_without_gamepad_robot_definition_raises(self):
        """KRABBY mode must receive an explicit RobotDefinition."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_KRABBY,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
        )

        loop = ControlLoop(config)

        with pytest.raises(ValueError, match="krabby_gamepad_robot_definition is required"):
            loop.start()

    def test_start_isaacsim_without_robot_definition_raises(self):
        """IsaacSim mode must receive an explicit RobotDefinition."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
        )

        loop = ControlLoop(config)

        with pytest.raises(ValueError, match="isaacsim_robot_definition is required"):
            loop.start()



class TestControlLoopIsRunning:
    """Test is_running() method."""

    def setup_method(self):
        """Set up test fixtures."""
        from controller.input.input_controller import InputController
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None

    def teardown_method(self):
        """Clean up after tests."""
        from controller.input.input_controller import InputController
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None

    def test_is_running_initial(self):
        """Test is_running() returns False initially."""
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(
                observation_endpoint="tcp://localhost:6001",
                command_endpoint="tcp://localhost:6002",
            ),
            isaacsim_robot_definition=KRABBY_HEX_DEFINITION,
        )
        
        loop = ControlLoop(config)
        assert loop.is_running() is False
