"""Inference test runner for testing policy inference.

NOTE: This is a TESTING/DEVELOPMENT tool, NOT for production.
It simulates sensor messages to test the inference logic (game loop).

The "game loop" refers to the core inference logic that:
1. Polls HAL for latest data
2. Builds observation tensor
3. Runs policy inference
4. Sends joint commands back to HAL

This test runner simulates that game loop for testing purposes.
"""

import logging
import signal
import time
from typing import Optional

import numpy as np

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import ParkourModelIO
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel

logger = logging.getLogger(__name__)


class InferenceTestRunner:
    """Test runner that simulates the game loop (inference logic) for testing.

    The "game loop" is the core inference logic that polls HAL, builds observations,
    runs inference, and sends commands. This test runner simulates that loop
    for testing purposes.
    """

    def __init__(
        self,
        model: ParkourPolicyModel,
        hal_client: HalClient,
        control_rate_hz: float = 100.0,
    ):
        """Initialize inference test runner.

        Args:
            model: Policy model for inference
            hal_client: HAL client for communication
            control_rate_hz: Control loop rate in Hz (default 100.0)
        """
        self.model = model
        self.hal_client = hal_client
        self.control_rate_hz = control_rate_hz
        self.period_s = 1.0 / control_rate_hz
        self.nav_cmd: Optional[NavigationCommand] = None
        self.running = False

        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.inference_not_ready_count = 0
        self.last_inference_result: Optional[object] = None
        self.inference_in_progress = False

    def set_navigation_command(self, nav_cmd: NavigationCommand) -> None:
        """Set navigation command for inference.
        
        Args:
            nav_cmd: Navigation command to use for inference
        """
        self.nav_cmd = nav_cmd

    def run(self) -> None:
        """Run inference test (simulates game loop).
        
        This method handles all exceptions internally and does not raise them.
        Exceptions are logged but do not propagate to callers. This ensures
        production code is robust and test code can rely on clean shutdown.
        """
        self.running = True
        logger.info(f"Starting inference test runner at {self.control_rate_hz} Hz")

        # Set default navigation command
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
                        logger.debug("No new observation available")
                        time.sleep(self.period_s * 0.1)  # Small sleep to avoid busy-wait
                        continue

                    # Check if navigation command is set
                    if self.nav_cmd is None:
                        # Navigation command not set, skip iteration
                        logger.debug("Navigation command not set, skipping inference")
                        time.sleep(self.period_s * 0.1)
                        continue
                    
                    # Map hardware observation to model observation format using mapper
                    # Pass navigation command so it's included in the observation
                    from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
                    
                    mapper = KrabbyHWObservationsToParkourMapper()
                    model_obs = mapper.map(hw_obs, nav_cmd=self.nav_cmd)
                    
                    # Build model IO (preserve timestamp from observation)
                    model_io = ParkourModelIO(
                        timestamp_ns=model_obs.timestamp_ns,
                        schema_version=model_obs.schema_version,
                        nav_cmd=self.nav_cmd,
                        observation=model_obs,
                    )

                    # Run inference (may take >10ms)
                    if not self.inference_in_progress:
                        # Start inference
                        self.inference_in_progress = True
                        try:
                            inference_result = self.model.inference(model_io)
                            self.inference_in_progress = False

                            if not inference_result.success:
                                logger.error(
                                    f"Inference failed: {inference_result.error_message}. "
                                    f"Frame will be dropped."
                                )
                                self.inference_not_ready_count += 1
                                continue

                            self.last_inference_result = inference_result

                            # Check inference latency
                            if inference_result.inference_latency_ms > 10.0:
                                logger.warning(
                                    f"Inference latency {inference_result.inference_latency_ms:.2f}ms > 10ms "
                                    f"(may cause frame drops)"
                                )

                            # Map inference response to hardware joint positions
                            from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
                            mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=self.model.action_dim)
                            joint_positions = mapper.map(inference_result)

                            # Put command to HAL
                            try:
                                self.hal_client.put_joint_command(joint_positions)
                            except Exception as e:
                                logger.error(f"Failed to put joint command to HAL: {e}. Frame will be dropped.")
                                self.inference_not_ready_count += 1
                                continue
                        except Exception as e:
                            self.inference_in_progress = False
                            logger.error(f"Inference failure: {e}", exc_info=True)
                            self.inference_not_ready_count += 1
                            # Continue loop - don't raise exception
                            continue
                    else:
                        # Inference still in progress from previous call, use cached result
                        if self.last_inference_result is not None and self.last_inference_result.success:
                            # Map cached inference response to hardware joint positions
                            from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
                            mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=self.model.action_dim)
                            joint_positions = mapper.map(self.last_inference_result)
                            try:
                                self.hal_client.put_joint_command(joint_positions)
                            except Exception as e:
                                logger.error(f"Failed to put cached joint command to HAL: {e}")
                            self.dropped_frames += 1
                        else:
                            logger.warning("Inference in progress but no valid cached result available")
                            self.inference_not_ready_count += 1

                    self.frame_count += 1

                    # Timing control
                    loop_end_ns = time.time_ns()
                    loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                    sleep_time = max(0.0, self.period_s - loop_duration_s)

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # Loop took longer than period - log warning
                        if loop_duration_s > self.period_s * 1.1:  # More than 10% over
                            logger.warning(
                                f"Simulation is unable to keep up! "
                                f"Frame time: {loop_duration_s*1000:.2f}ms exceeds target: {self.period_s*1000:.2f}ms. "
                                f"This may cause control instability."
                            )

                    # Log statistics periodically
                    if self.frame_count % 1000 == 0:
                        self._log_statistics()

                except Exception as e:
                    # Handle any unexpected exceptions in the loop
                    logger.error(f"Unexpected error in inference loop: {e}", exc_info=True)
                    # Continue loop if still running, otherwise break
                    if not self.running:
                        break
                    # Small sleep to avoid tight error loop
                    time.sleep(self.period_s * 0.1)
                    continue

        except KeyboardInterrupt:
            logger.info("Inference test runner interrupted by user")
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error in test runner: {e}", exc_info=True)
        finally:
            self.running = False
            self._log_statistics()

    def stop(self) -> None:
        """Stop inference test runner gracefully.
        
        This method should not raise exceptions. If cleanup fails,
        it should log the error but not raise.
        """
        self.running = False
        # Additional cleanup can be added here if needed
        # All cleanup should be wrapped in try-except to prevent exceptions

    def _log_statistics(self) -> None:
        """Log inference test statistics."""
        logger.info(
            f"Inference test stats: frames={self.frame_count}, "
            f"dropped={self.dropped_frames}, "
            f"inference_not_ready={self.inference_not_ready_count}"
        )


def run_inference_test(
    checkpoint_path: str,
    action_dim: int,
    obs_dim: int,
    hal_endpoints: dict,
    control_rate_hz: float = 100.0,
    device: str = "cuda",
) -> None:
    """Run inference test (simulates game loop for testing).

    Args:
        checkpoint_path: Path to model checkpoint
        action_dim: Action dimension
        obs_dim: Observation dimension
        hal_endpoints: Dictionary with 'camera', 'state', 'command' endpoints
        control_rate_hz: Control loop rate in Hz
        device: Device to run inference on
    """
    # Initialize policy model
    logger.info(f"Loading model from {checkpoint_path}")
    weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )
    model = ParkourPolicyModel(weights, device=device)

    # Initialize HAL client (use unified observation endpoint)
    config = HalClientConfig(
        observation_endpoint=hal_endpoints["observation"],
        command_endpoint=hal_endpoints["command"],
    )
    hal_client = HalClient(config)
    hal_client.initialize()

    # Create and run inference test runner
    test_runner = InferenceTestRunner(model, hal_client, control_rate_hz=control_rate_hz)

    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, stopping inference test...")
        test_runner.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        test_runner.run()
    finally:
        hal_client.close()
        logger.info("Inference test completed")

