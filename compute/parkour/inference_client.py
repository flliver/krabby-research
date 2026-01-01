"""Parkour inference client using HAL for observations and commands.

This module provides a client that:
1. Polls observations from HAL server
2. Runs parkour policy inference
3. Sends joint commands back to HAL server

Designed to run in a separate thread from the simulation/HAL server.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import zmq

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
from compute.parkour.parkour_types import ParkourModelIO

logger = logging.getLogger(__name__)


class ParkourInferenceClient(HalClient):
    """Parkour inference client that extends HAL client.

    This is a HAL client that also runs parkour policy inference.
    It extends HalClient to inherit all HAL communication functionality,
    and adds inference capabilities on top.

    Attributes:
        model: Parkour policy model
        control_rate: Control loop rate in Hz
    """

    def __init__(
        self,
        hal_client_config: HalClientConfig,
        model_weights: ModelWeights,
        control_rate: float = 100.0,
        device: str = "cuda",
        transport_context: Optional[zmq.Context] = None,
    ):
        """Initialize Parkour inference client.

        Args:
            hal_client_config: HAL client configuration
            model_weights: Model weights configuration
            control_rate: Control loop rate in Hz
            device: Device for inference ("cuda" or "cpu")
            transport_context: ZMQ context for inproc connections (required for inproc)
        """
        # Initialize base HalClient
        super().__init__(hal_client_config, context=transport_context)
        
        self.model_weights = model_weights
        self.control_rate = control_rate
        self.device = device

        self.model: Optional[ParkourPolicyModel] = None
        self.nav_cmd: Optional[NavigationCommand] = None
        self._inference_initialized = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """Initialize HAL client and policy model."""
        # Initialize base HalClient first
        super().initialize()
        
        if self._inference_initialized:
            logger.warning("ParkourInferenceClient inference already initialized")
            return

        logger.info(f"Loading parkour policy model from {self.model_weights.checkpoint_path}")
        self.model = ParkourPolicyModel(self.model_weights, device=self.device)

        # Initialize navigation command with default values
        self.nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)

        self._inference_initialized = True
        logger.info("ParkourInferenceClient initialized")

    def _inference_step(self) -> bool:
        """Execute one inference step.

        Polls observation, runs inference, and sends command.

        Returns:
            True if step succeeded, False otherwise
        """
        if not self._initialized or not self._inference_initialized or self.model is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Poll HAL for latest hardware observation (using inherited method)
        hw_obs = self.poll(timeout_ms=1)

        if hw_obs is None:
            # No new observation available - this is normal, just continue
            return True

        # Capture current timestamp for synchronization
        current_timestamp_ns = time.time_ns()
        
        # Map hardware observation to model observation format
        # Create a new nav_cmd with current timestamp
        nav_cmd = NavigationCommand.create_now()
        nav_cmd.timestamp_ns = current_timestamp_ns
        
        mapper = HWObservationsToParkourMapper()
        model_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
        
        # Update both observation and nav_cmd timestamps to use the captured timestamp
        # This ensures synchronization between observation and nav_cmd
        model_obs.timestamp_ns = current_timestamp_ns
        nav_cmd.timestamp_ns = current_timestamp_ns

        # Build model IO (preserve timestamp from observation)
        model_io = ParkourModelIO(
            timestamp_ns=model_obs.timestamp_ns,
            schema_version=model_obs.schema_version,
            nav_cmd=nav_cmd,
            observation=model_obs,
        )

        # Run inference
        try:
            inference_result = self.model.inference(model_io)

            if not inference_result.success:
                logger.error(f"Inference failed: {inference_result.error_message}")
                return False

            # Map inference response to hardware joint positions
            hw_mapper = ParkourLocomotionToHWMapper(model_action_dim=self.model.action_dim)
            joint_cmd = hw_mapper.map(inference_result, observation_timestamp_ns=hw_obs.timestamp_ns)
            
            # Update timestamp to current time
            joint_cmd.timestamp_ns = time.time_ns()

            # Send command back to HAL server (using inherited method)
            self.put_joint_command(joint_cmd)
            return True

        except Exception as e:
            logger.error(f"Inference step failed: {e}", exc_info=True)
            return False

    def _run_loop(self, running_flag) -> None:
        """Run inference loop at control rate.

        Args:
            running_flag: Callable that returns True while loop should continue
        """
        if not self._initialized or not self._inference_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        logger.info(f"Starting inference loop at {self.control_rate} Hz")
        period_s = 1.0 / self.control_rate

        while running_flag():
            loop_start_ns = time.time_ns()

            # Execute inference step
            if not self._inference_step():
                logger.warning("Inference step failed, continuing...")

            # Timing control
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if loop_duration_s > period_s * 1.1:
                    logger.warning(
                        f"Inference unable to keep up! "
                        f"Frame time: {loop_duration_s*1000:.2f}ms exceeds "
                        f"target: {period_s*1000:.2f}ms"
                    )

        logger.info("Inference loop stopped")

    def start_thread(self, running_flag=lambda: True) -> None:
        """Start inference loop in a separate thread.

        Args:
            running_flag: Callable that returns True while loop should continue
        """
        if not self._initialized or not self._inference_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        if self._running:
            logger.warning("Inference thread already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(running_flag,),
            daemon=True,
            name="parkour-inference",
        )
        self._thread.start()
        logger.info("Inference thread started")

    def stop_thread(self, timeout: float = 5.0) -> None:
        """Stop inference thread.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        if not self._running or self._thread is None:
            logger.warning("Inference thread not running")
            return

        logger.info("Stopping inference thread...")
        self._running = False

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within timeout")

        self._thread = None
        logger.info("Inference thread stopped")

    def set_navigation_command(self, nav_cmd: NavigationCommand) -> None:
        """Set navigation command for inference.

        Args:
            nav_cmd: Navigation command (vx, vy, yaw_rate)
        """
        self.nav_cmd = nav_cmd

    def close(self) -> None:
        """Close HAL client and clean up resources."""
        if self._running:
            self.stop_thread()

        # Close base HalClient
        super().close()

        self._inference_initialized = False
        logger.info("ParkourInferenceClient closed")
