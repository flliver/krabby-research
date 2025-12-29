"""Inference test runner for testing policy inference.

NOTE: This is a TESTING/DEVELOPMENT tool, NOT for production.

This module provides two interfaces:
1. `run_inference_test()` - Uses production ParkourInferenceClient (recommended for main tests)
2. `InferenceTestRunner` class - Works with any model interface (for integration tests with mocks)

The production path uses ParkourInferenceClient which handles:
1. Polling HAL for latest data
2. Building observation tensor
3. Running policy inference
4. Sending joint commands back to HAL

This ensures the test uses the same code path as production.
"""

import logging
import signal
import time
from typing import Optional

import zmq

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
from compute.parkour.parkour_types import ParkourModelIO
from compute.parkour.policy_interface import ModelWeights
from compute.parkour.inference_client import ParkourInferenceClient

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_running = True


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global _running
    logger.info("Received interrupt signal, stopping inference test...")
    _running = False


class InferenceTestRunner:
    """Test runner that simulates the game loop (inference logic) for testing.

    This class works with any model that has an `inference(model_io)` method.
    It's used by integration tests with mock models.

    For production-like testing, use `run_inference_test()` which uses
    `ParkourInferenceClient` instead.
    """

    def __init__(
        self,
        model,  # Any model with inference(model_io) method
        hal_client: HalClient,
        control_rate_hz: float = 100.0,
    ):
        """Initialize inference test runner.

        Args:
            model: Policy model for inference (must have inference(model_io) method)
            hal_client: HAL client for communication
            control_rate_hz: Control loop rate in Hz (default 100.0)
        """
        self.model = model
        self.hal_client = hal_client
        self.control_rate_hz = control_rate_hz
        self.period_s = 1.0 / control_rate_hz
        self.running = False
        self.nav_cmd: Optional[NavigationCommand] = None
        # Statistics for tests
        self.frame_count = 0  # Frames successfully processed (inference completed)
        self.last_inference_result = None
        self.frames_received = 0  # Observations received from HAL

    def set_navigation_command(self, nav_cmd: NavigationCommand) -> None:
        """Set navigation command for inference.
        
        Args:
            nav_cmd: Navigation command to use for inference
        """
        self.nav_cmd = nav_cmd

    def run(self) -> None:
        """Run inference test (simulates game loop).
        
        This method handles all exceptions internally and does not raise them.
        Exceptions are logged but do not propagate to callers.
        """
        self.running = True
        logger.info(f"Starting inference test runner at {self.control_rate_hz} Hz")

        # Set default navigation command if not set
        if self.nav_cmd is None:
            try:
                self.nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)
            except Exception as e:
                logger.error(f"Failed to create default navigation command: {e}", exc_info=True)
                self.running = False
                return

        try:
            while self.running:
                try:
                    loop_start_ns = time.time_ns()
                    
                    # Poll HAL for latest hardware observation (non-blocking)
                    hw_obs = self.hal_client.poll(timeout_ms=1)
                    
                    if hw_obs is None:
                        # No new data available, skip iteration
                        # Timing control sleep below will handle rate limiting
                        continue
                    
                    # Track that we received an observation
                    self.frames_received += 1

                    # Map hardware observation to model observation format
                    nav_cmd = self.nav_cmd or NavigationCommand.create_now()
                    nav_cmd.timestamp_ns = hw_obs.timestamp_ns
                    
                    mapper = HWObservationsToParkourMapper()
                    model_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
                    
                    # Build model IO
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
                            continue

                        # Track statistics for tests
                        self.frame_count += 1
                        self.last_inference_result = inference_result

                        # Map inference response to hardware joint positions
                        mapper = ParkourLocomotionToHWMapper(model_action_dim=self.model.action_dim)
                        joint_positions = mapper.map(inference_result, observation_timestamp_ns=hw_obs.timestamp_ns)

                        # Put command to HAL
                        try:
                            self.hal_client.put_joint_command(joint_positions)
                        except Exception as e:
                            logger.error(f"Failed to put joint command to HAL: {e}")
                            continue
                    except Exception as e:
                        logger.error(f"Inference failure: {e}", exc_info=True)
                        continue

                    # Timing control
                    loop_end_ns = time.time_ns()
                    loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                    sleep_time = max(0.0, self.period_s - loop_duration_s)

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    # Handle any unexpected exceptions in the loop
                    logger.error(f"Unexpected error in inference loop: {e}", exc_info=True)
                    if not self.running:
                        break
                    time.sleep(self.period_s * 0.1)
                    continue

        except KeyboardInterrupt:
            logger.info("Inference test runner interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in test runner: {e}", exc_info=True)
        finally:
            self.running = False

    def stop(self) -> None:
        """Stop inference test runner gracefully."""
        self.running = False


def run_inference_test(
    checkpoint_path: str,
    action_dim: int,
    obs_dim: int,
    hal_endpoints: dict,
    control_rate_hz: float = 100.0,
    device: str = "cuda",
    transport_context: Optional[zmq.Context] = None,
) -> None:
    """Run inference test using ParkourInferenceClient.

    This uses the production ParkourInferenceClient to ensure the test
    uses the same code path as production. Follows the sequencing pattern
    from the single latency test (without warmup).

    Args:
        checkpoint_path: Path to model checkpoint
        action_dim: Action dimension
        obs_dim: Observation dimension
        hal_endpoints: Dictionary with 'observation' and 'command' endpoints
        control_rate_hz: Control loop rate in Hz
        device: Device to run inference on
        transport_context: Optional ZMQ context for inproc connections
    """
    global _running
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create HAL client config for inference client
    logger.info("[STEP] Creating HAL client config...")
    hal_client_config = HalClientConfig(
        observation_endpoint=hal_endpoints["observation"],
        command_endpoint=hal_endpoints["command"],
    )

    # Create model weights config
    logger.info(f"[STEP] Loading model checkpoint from {checkpoint_path}...")
    model_weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )

    # Create and initialize ParkourInferenceClient
    logger.info("[STEP] Creating ParkourInferenceClient...")
    inference_client = ParkourInferenceClient(
        hal_client_config=hal_client_config,
        model_weights=model_weights,
        control_rate=control_rate_hz,
        device=device,
        transport_context=transport_context,
    )
    inference_client.initialize()
    logger.info("[STEP] ParkourInferenceClient initialized")

    # For inproc connections, poll once to establish PUB/SUB connection
    if transport_context is not None:
        inference_client.poll(timeout_ms=100)  # Poll once to establish PUB/SUB connection

    try:
        # Start inference client thread (will run for entire test)
        logger.info(f"[STEP] Starting inference client thread at {control_rate_hz} Hz...")
        inference_client.start_thread(running_flag=lambda: _running)

        # Wait until interrupted
        logger.info("[STEP] Inference test running. Press Ctrl+C to stop.")
        while _running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("[STEP] Inference test interrupted by user")
    except Exception as e:
        logger.error(f"[FAIL] Inference test failed: {e}", exc_info=True)
        raise
    finally:
        # Stop threads and cleanup
        logger.info("[STEP] Stopping inference client thread...")
        _running = False
        inference_client.stop_thread()
        inference_client.close()
        logger.info("[STEP] Inference test completed")

