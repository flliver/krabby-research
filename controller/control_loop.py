"""Control loop for wiring controller components together.

This module provides a ControlLoop class that wires singleton components
(InputController, HALClient, mappers, SDKs) based on configuration.

"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from controller.input import InputController
from controller.input.state import ControllerState
from controller.mappers.gamepad_to_isaacsim_hal_mapper import GamepadToIsaacSimHALMapper
from controller.mappers.gamepad_to_krabby_hal_mapper import GamepadToKrabbyHALMapper
from hal.client import HalClient
from hal.client.config import HalClientConfig
from hal.server.robot_definition import RobotDefinition

if TYPE_CHECKING:
    from teleop.edge.webrtc_input_controller import WebRTCInputController

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """Control loop mode enumeration. 
    More modes may be added later."""
    INPUT_CONTROLLER_ISAACSIM = "input_controller_isaacsim"
    MODEL_CONTROLLER_KRABBY = "model_controller_krabby"    
    INPUT_CONTROLLER_KRABBY = "input_controller_krabby"
    INPUT_CONTROLLER_WEBRTC = "input_controller_webrtc"


@dataclass
class ControlLoopConfig:
    """Configuration for ControlLoop.
    
    Attributes:
        mode: Control mode (INPUT_CONTROLLER_ISAACSIM or MODEL_CONTROLLER_KRABBY or INPUT_CONTROLLER_KRABBY)
        input_controller_device_id: Optional device ID for InputController
        input_controller_update_rate_hz: Update rate for InputController (default: 50.0)
        hal_client_config: HAL client configuration
        mapper_hip_up_down_scale: Scaling factor for hip up/down axis
        mapper_knee_out_in_scale: Scaling factor for knee out/in axis
        mapper_hip_yaw_scale: Scaling factor for hip yaw axis
        isaacsim_robot_definition: Required for INPUT_CONTROLLER_ISAACSIM (must match IsaacSim HAL topology).
        krabby_gamepad_robot_definition: Required for INPUT_CONTROLLER_KRABBY and INPUT_CONTROLLER_WEBRTC
            (must match the HAL server topology). Unused for other modes.
    """
    mode: ControlMode
    input_controller_device_id: Optional[int] = None
    input_controller_update_rate_hz: float = 50.0
    hal_client_config: Optional[HalClientConfig] = None
    mapper_hip_up_down_scale: float = 1.0
    mapper_knee_out_in_scale: float = 1.0
    mapper_hip_yaw_scale: float = 1.0
    isaacsim_robot_definition: Optional[RobotDefinition] = None
    krabby_gamepad_robot_definition: Optional[RobotDefinition] = None
    webrtc_input_controller: Optional[WebRTCInputController] = None


class ControlLoop:
    """Control loop that wires controller components together.
    
    Manages lifecycle of InputController, HALClient, mappers, and SDKs
    based on configuration. Provides clean thread lifecycle management.
    
    Usage:
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(...),
        )
        loop = ControlLoop(config)
        loop.start()
        # ... run control loop ...
        loop.stop()
    """
    
    def __init__(self, config: ControlLoopConfig):
        """Initialize control loop.
        
        Args:
            config: Control loop configuration
        """
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Component references
        self._input_controller: Optional[InputController] = None
        self._hal_client: Optional[HalClient] = None
        self._gamepad_to_isaacsim_hal_mapper: Optional[GamepadToIsaacSimHALMapper] = None
        self._gamepad_to_krabby_hal_mapper: Optional[GamepadToKrabbyHALMapper] = None
        self._isaacsim_first_send_logged = False

    def start(self) -> None:
        """Start the control loop.
        
        Initializes and starts all components based on configuration.
        
        Raises:
            RuntimeError: If control loop is already running or configuration is invalid
        """
        if self._running:
            logger.warning("ControlLoop is already running")
            return
        
        logger.info(f"Starting ControlLoop in mode: {self.config.mode}")
        
        # Initialize components based on mode
        if self.config.mode == ControlMode.INPUT_CONTROLLER_ISAACSIM:
            self._start_input_controller_isaacsim_mode()
        elif self.config.mode == ControlMode.MODEL_CONTROLLER_KRABBY:
            # TODO: Implement MODEL_CONTROLLER_KRABBY mode. 
            raise NotImplementedError("MODEL_CONTROLLER_KRABBY mode not yet implemented")
        elif self.config.mode == ControlMode.INPUT_CONTROLLER_KRABBY:
            self._start_input_controller_krabby_mode()
        elif self.config.mode == ControlMode.INPUT_CONTROLLER_WEBRTC:
            self._start_input_controller_webrtc_mode()
        else:
            raise ValueError(f"Unknown control mode: {self.config.mode}")
        
        self._running = True
        logger.info("ControlLoop started successfully")
    
    def stop(self) -> None:
        """Stop the control loop.
        
        Stops all components and cleans up resources.
        """
        if not self._running:
            return
        
        logger.info("Stopping ControlLoop")
        self._running = False
        
        # Stop input controller
        if self._input_controller is not None:
            self._input_controller.stop()
            self._input_controller = None
        
        # Close HAL client
        if self._hal_client is not None:
            self._hal_client.close()
            self._hal_client = None
        
        logger.info("ControlLoop stopped")
    
    def _start_input_controller_isaacsim_mode(self) -> None:
        """Start components for INPUT_CONTROLLER_ISAACSIM mode."""
        # Get InputController singleton
        self._input_controller = InputController.get_instance()
        
        # Initialize HAL client
        if self.config.hal_client_config is None:
            raise ValueError("hal_client_config is required for INPUT_CONTROLLER_ISAACSIM mode")

        rd = self.config.isaacsim_robot_definition
        if rd is None:
            raise ValueError(
                "isaacsim_robot_definition is required for INPUT_CONTROLLER_ISAACSIM "
                "(e.g. krabby-uno-sim --hex|--quad, or pass RobotDefinition explicitly)"
            )

        self._hal_client = HalClient(
            config=self.config.hal_client_config,
            context=None,
        )
        self._hal_client.initialize()

        self._gamepad_to_isaacsim_hal_mapper = GamepadToIsaacSimHALMapper(
            hip_up_down_scale=self.config.mapper_hip_up_down_scale,
            knee_out_in_scale=self.config.mapper_knee_out_in_scale,
            hip_yaw_scale=self.config.mapper_hip_yaw_scale,
            robot_definition=rd,
        )
        
        # Register callback to send commands when gamepad state changes
        self._input_controller.register_callback(self._on_gamepad_state)
        
        # Start input controller
        self._input_controller.start(
            device_id=self.config.input_controller_device_id,
            update_rate_hz=self.config.input_controller_update_rate_hz,
        )
        
        logger.info("INPUT_CONTROLLER_ISAACSIM mode initialized")
    
    def _start_input_controller_krabby_mode(self) -> None:
        """Start components for INPUT_CONTROLLER_KRABBY mode."""
        # Get InputController singleton
        self._input_controller = InputController.get_instance()
        
        # Initialize HAL client
        if self.config.hal_client_config is None:
            raise ValueError("hal_client_config is required for INPUT_CONTROLLER_KRABBY mode")

        rd = self.config.krabby_gamepad_robot_definition
        if rd is None:
            raise ValueError(
                "krabby_gamepad_robot_definition is required for INPUT_CONTROLLER_KRABBY "
                "(set from CLI --robot hex|go2 or pass RobotDefinition explicitly)"
            )

        self._hal_client = HalClient(
            config=self.config.hal_client_config,
            context=None,
        )
        self._hal_client.initialize()

        self._gamepad_to_krabby_hal_mapper = GamepadToKrabbyHALMapper(
            hip_up_down_scale=self.config.mapper_hip_up_down_scale,
            knee_out_in_scale=self.config.mapper_knee_out_in_scale,
            hip_yaw_scale=self.config.mapper_hip_yaw_scale,
            robot_definition=rd,
        )

        # Register callback to send commands when gamepad state changes
        self._input_controller.register_callback(self._on_gamepad_state_krabby)
        
        # Start input controller
        self._input_controller.start(
            device_id=self.config.input_controller_device_id,
            update_rate_hz=self.config.input_controller_update_rate_hz,
        )
        
        logger.info("INPUT_CONTROLLER_KRABBY mode initialized")

    def _start_input_controller_webrtc_mode(self) -> None:
        """Start components for INPUT_CONTROLLER_WEBRTC mode."""
        if self.config.webrtc_input_controller is None:
            raise ValueError("webrtc_input_controller is required for INPUT_CONTROLLER_WEBRTC mode")
        self._input_controller = self.config.webrtc_input_controller

        if self.config.hal_client_config is None:
            raise ValueError("hal_client_config is required for INPUT_CONTROLLER_WEBRTC mode")

        rd = self.config.krabby_gamepad_robot_definition
        if rd is None:
            raise ValueError(
                "krabby_gamepad_robot_definition is required for INPUT_CONTROLLER_WEBRTC "
                "(must match Jetson HAL --robot topology)"
            )

        self._hal_client = HalClient(
            config=self.config.hal_client_config,
            context=None,
        )
        self._hal_client.initialize()

        self._gamepad_to_krabby_hal_mapper = GamepadToKrabbyHALMapper(
            hip_up_down_scale=self.config.mapper_hip_up_down_scale,
            knee_out_in_scale=self.config.mapper_knee_out_in_scale,
            hip_yaw_scale=self.config.mapper_hip_yaw_scale,
            robot_definition=rd,
        )
        self._input_controller.register_callback(self._on_gamepad_state_krabby)
        self._input_controller.start(update_rate_hz=self.config.input_controller_update_rate_hz)
        logger.info("INPUT_CONTROLLER_WEBRTC mode initialized")
    
    def _on_gamepad_state(self, state: ControllerState) -> None:
        """Callback for gamepad state updates.
        
        Maps controller state to joint command and sends via HAL client.
        
        Args:
            state: Controller state
        """
        if not self._running:
            return
        
        try:
            # Map controller state to joint command
            # TODO: For now, we don't have observation timestamp, so use None
            # In a real implementation, we'd track the last observation timestamp
            joint_cmd = self._gamepad_to_isaacsim_hal_mapper.map(state, observation_timestamp_ns=None)
            
            # Send command via HAL client
            self._hal_client.put_joint_command(joint_cmd)
            if not self._isaacsim_first_send_logged:
                self._isaacsim_first_send_logged = True
                logger.info("Pro Controller: first joint command sent to HAL server (commands now streaming)")
            d = joint_cmd.to_positions_dict()
            logger.debug(
                "Sent joint command (Pro Controller/gamepad): %d joints, range=[%.3f, %.3f]",
                len(d), min(d.values()), max(d.values()),
            )
        except Exception as e:
            logger.error(f"Error processing gamepad state: {e}", exc_info=True)
    
    def _on_gamepad_state_krabby(self, state: ControllerState) -> None:
        """Callback for gamepad state updates in Krabby mode.
        
        Maps controller state to joint command and sends via HAL client.
        
        Args:
            state: Controller state
        """
        if not self._running:
            return
        
        try:
            # Map controller state to joint command
            joint_cmd = self._gamepad_to_krabby_hal_mapper.map(state, observation_timestamp_ns=None)
            
            # Send command via HAL client
            self._hal_client.put_joint_command(joint_cmd)
            
            d = joint_cmd.to_positions_dict()
            logger.debug(
                f"Sent joint command: "
                f"joint_range=[{min(d.values()):.3f}, {max(d.values()):.3f}]"
            )
        except Exception as e:
            logger.error(f"Error processing gamepad state: {e}", exc_info=True)
    
    def is_running(self) -> bool:
        """Check if control loop is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running

    def wait_for_hal_server(self, timeout_s: float = 15.0, poll_interval_s: float = 0.1) -> bool:
        """Wait until the HAL server is reachable (at least one observation received).
        
        Only applicable when using INPUT_CONTROLLER_ISAACSIM mode. Polls the observation
        socket until data is received or timeout. Use after start() to fail fast if the
        Isaac Sim HAL server is not running.
        
        Args:
            timeout_s: Maximum time to wait in seconds.
            poll_interval_s: Time between poll attempts in seconds.
            
        Returns:
            True if an observation was received, False on timeout.
        """
        if self._hal_client is None:
            return False
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            obs = self._hal_client.poll(timeout_ms=int(poll_interval_s * 1000))
            if obs is not None:
                return True
        return False
