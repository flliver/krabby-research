"""Production inference runner for Jetson deployment."""

import logging
import time
from typing import Optional

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.server.jetson import JetsonHalServer

logger = logging.getLogger(__name__)


class InferenceRunner:
    """Production inference runner that coordinates HAL server and policy inference.

    Runs both HAL server (real sensors) and policy inference in the same process,
    using inproc ZMQ for communication.
    """

    def __init__(
        self,
        model: ParkourPolicyModel,
        hal_server_config: HalServerConfig,
    ):
        """Initialize inference runner.

        Args:
            model: Policy model for inference
            hal_server_config: HAL server configuration (should use inproc endpoints)
        """
        self.model = model
        self.hal_server = JetsonHalServer(hal_server_config)
        self.hal_client: Optional[HalClient] = None
        self.running = False

    def initialize(self) -> None:
        """Initialize HAL server and client."""
        # Initialize HAL server
        self.hal_server.initialize()
        self.hal_server.initialize_camera()
        self.hal_server.initialize_sensors()
        self.hal_server.initialize_actuators()

        # Create HAL client config with inproc endpoints (use unified observation endpoint)
        observation_bind = self.hal_server.config.observation_bind
        
        if "inproc://" in observation_bind:
            # For inproc, use the same endpoint strings and get transport context from server
            client_config = HalClientConfig(
                observation_endpoint=observation_bind,
                command_endpoint=self.hal_server.config.command_bind,
            )
            transport_context = self.hal_server.get_transport_context()
        else:
            # For TCP, convert bind to connect
            client_config = HalClientConfig(
                observation_endpoint=observation_bind.replace("bind", "connect").replace("*", "localhost"),
                command_endpoint=self.hal_server.config.command_bind.replace("bind", "connect").replace("*", "localhost"),
            )
            transport_context = None

        # Use shared ZMQ context when provided (for inproc or shared-context setups)
        self.hal_client = HalClient(client_config, context=transport_context)
        self.hal_client.initialize()

        logger.info("Inference runner initialized")

    def run(self, control_rate_hz: float = 100.0) -> None:
        """Run game loop (inference logic).

        The game loop is the core inference logic that:
        1. Polls HAL for latest data
        2. Builds observation tensor
        3. Runs policy inference
        4. Sends joint commands back to HAL

        Args:
            control_rate_hz: Control loop rate in Hz (default 100.0)
        """
        self.running = True
        period_s = 1.0 / control_rate_hz

        logger.info(f"Starting game loop (inference logic) at {control_rate_hz} Hz")

        try:
            while self.running:
                loop_start_ns = time.time_ns()

                # Set observation from real sensors
                self.hal_server.set_observation()

                # Poll HAL for latest data
                if self.hal_client:
                    self.hal_client.poll(timeout_ms=1)

                    # Get observation (raises RuntimeError if not available)
                    try:
                        observation = self.hal_client.get_observation()
                    except RuntimeError as e:
                        # No observation available, skip inference this frame
                        logger.debug(f"No observation available: {e}")
                        self.hal_server.move()  # Still apply any pending commands
                        continue

                    # Build ParkourModelIO for policy interface (backward compatibility)
                    model_io = self.hal_client.build_model_io()
                    if model_io is None:
                        # Navigation command not set, skip inference
                        logger.debug("Navigation command not set, skipping inference")
                        self.hal_server.move()  # Still apply any pending commands
                        continue

                    # Run inference
                    try:
                        inference_result = self.model.inference(model_io)

                        if not inference_result.success:
                            raise RuntimeError(f"Inference failed: {inference_result.error_message}")

                        # Put command back to HAL server
                        if not self.hal_client.put_joint_command(inference_result):
                            raise RuntimeError("Failed to put joint command to HAL")
                    except Exception as e:
                        logger.error(f"Critical inference failure: {e}", exc_info=True)
                        # Fail fast - re-raise to stop the loop
                        raise

                # Move robot (get command from HAL and apply)
                self.hal_server.move()

                # Timing control
                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_duration_s > period_s * 1.1:
                        logger.warning(f"Loop duration {loop_duration_s*1000:.2f}ms > period {period_s*1000:.2f}ms")

        except KeyboardInterrupt:
            logger.info("Inference runner interrupted by user")
        finally:
            self.running = False

    def stop(self) -> None:
        """Stop inference runner."""
        self.running = False

    def close(self) -> None:
        """Close all resources."""
        if self.hal_client:
            self.hal_client.close()
        self.hal_server.close()
        logger.info("Inference runner closed")

