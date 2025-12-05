"""Core acceptance tests for HAL: 100-tick execution and latency.

These tests verify core runtime requirements for HAL integration:
- 100+ tick execution with proto HAL server (no stalls)
- Inference latency < 15ms (when using HAL + game loop)

Note: Model inference correctness tests are in tests/unit/test_compute_parkour_policy.py
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import ParkourObservation, OBS_DIM
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.testing.inference_test_runner import InferenceTestRunner
from tests.helpers import create_dummy_hw_obs


class ProtoHalServer(HalServerBase):
    """Proto HAL server for testing - publishes synthetic observations in training format.

    This is a minimal HAL server that publishes observations matching the training format
    exactly: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
    """

    def __init__(self, config: HalServerConfig):
        """Initialize proto HAL server.
        
        Args:
            config: HAL server configuration
        """
        super().__init__(config)
        self.tick_count = 0
        self._running = False
        self._publish_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None

    def start_publishing(self, rate_hz: float = 100.0):
        """Start publishing observations at specified rate.

        Args:
            rate_hz: Publication rate in Hz
        """
        if self._running:
            return

        self._running = True
        period = 1.0 / rate_hz

        def publish_loop():
            """Background loop that publishes synthetic observations at a fixed rate."""
            while self._running:
                # Publish synthetic observation in training format
                self.publish_observation()
                self.tick_count += 1
                time.sleep(period)

        def command_loop():
            """Handle incoming commands in a loop."""
            while self._running:
                try:
                    # Poll for commands (non-blocking with short timeout)
                    command = self.get_joint_command(timeout_ms=10)
                    if command is not None:
                        # Command received and acknowledged by get_joint_command
                        pass
                except Exception as e:
                    # Handle exceptions in command handling thread
                    # Expected exceptions during shutdown (connection errors, etc.) are logged but not raised
                    # If we need to handle specific exceptions in the future, add them here:
                    # except (zmq.ZMQError, ConnectionError) as e:
                    #     if not self._running:
                    #         # Expected during shutdown
                    #         break
                    #     raise
                    if not self._running:
                        # Expected during shutdown - exit loop gracefully
                        break
                    # Unexpected exception while running - log and continue
                    # (In production, might want to re-raise, but in test server we continue)
                    import logging
                    logging.getLogger(__name__).debug(
                        f"Exception in command loop (continuing): {e}", exc_info=True
                    )

        self._publish_thread = threading.Thread(target=publish_loop, daemon=True)
        self._publish_thread.start()
        
        self._command_thread = threading.Thread(target=command_loop, daemon=True)
        self._command_thread.start()

    def stop_publishing(self):
        """Stop publishing observations."""
        self._running = False
        if self._publish_thread:
            self._publish_thread.join(timeout=1.0)
        if self._command_thread:
            self._command_thread.join(timeout=1.0)

    def publish_observation(self):
        """Set/publish synthetic observation in training format.

        Creates observation array matching training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        
        Note: This method name is kept for backward compatibility in tests.
        It calls set_observation() on the base class.
        """
        from compute.parkour.parkour_types import OBS_DIM

        # Create synthetic observation in training format
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)

        # Fill with synthetic data (simple patterns for testing)
        num_prop = 53
        num_scan = 132
        num_priv_explicit = 9
        num_priv_latent = 29
        history_dim = 530

        obs_array[:num_prop] = np.sin(np.arange(num_prop) * 0.1).astype(np.float32)
        obs_array[num_prop : num_prop + num_scan] = np.cos(
            np.arange(num_scan) * 0.05
        ).astype(np.float32)
        obs_array[
            num_prop + num_scan : num_prop + num_scan + num_priv_explicit
        ] = np.random.randn(num_priv_explicit).astype(np.float32)
        obs_array[
            num_prop
            + num_scan
            + num_priv_explicit : num_prop
            + num_scan
            + num_priv_explicit
            + num_priv_latent
        ] = np.random.randn(num_priv_latent).astype(np.float32)
        obs_array[
            num_prop + num_scan + num_priv_explicit + num_priv_latent :
        ] = np.random.randn(history_dim).astype(np.float32)

        # Create hardware observation from the observation array
        # For testing, we'll create a dummy hardware observation
        # In production, this would come from actual sensors
        from hal.client.data_structures.hardware import KrabbyHardwareObservations
        
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        # Copy joint positions from obs_array (first 18 elements if available)
        num_joints = min(18, len(obs_array))
        hw_obs.joint_positions[:num_joints] = obs_array[:num_joints].astype(np.float32)
        
        # Publish via base class
        super().set_observation(hw_obs)

    def apply_joint_command(self, command_bytes: bytes) -> bytes:
        """Apply joint command (stub for testing).

        Args:
            command_bytes: Joint command as float32 array bytes

        Returns:
            command sent successfully
        """
        # Validate command
        if len(command_bytes) % 4 != 0:
            return b"error: invalid command size"
        action_dim = len(command_bytes) // 4
        command_array = np.frombuffer(command_bytes, dtype=np.float32)
        if len(command_array) != action_dim:
            return b"error: invalid action dimension"
        return b"ok"


@pytest.fixture
def proto_hal_setup():
    """Setup proto HAL server and client for testing."""
    import zmq
    
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_proto",
        command_bind="inproc://test_cmd_proto",
    )
    server = ProtoHalServer(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_proto",
        command_endpoint="inproc://test_cmd_proto",
    )
    # Use shared ZMQ context from server for inproc connections
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    # Publish an initial observation to establish the PUB/SUB connection
    server.publish_observation()
    client.poll(timeout_ms=100)

    yield server, client

    # Cleanup
    server.stop_publishing()
    client.close()
    server.close()


def test_100_tick_execution_with_proto_hal(proto_hal_setup):
    """Test game loop executes 100+ ticks with proto HAL server without stalls.

    This is a core acceptance test for runtime stability. It verifies:
    1. Runtime stability: System can run continuously without crashing or stalling
    2. Minimum performance: System can sustain at least 100 Hz (100 ticks per second)
    3. No stalls: System runs smoothly without significant delays or blocking

    The test runs for ~1.1 seconds at 100 Hz, expecting approximately 100-110 ticks.
    Frame count may vary due to timing variations (thread scheduling, sleep precision),
    so we use a range assertion (95-110) to verify the system runs at approximately
    the expected rate while allowing for timing variations.
    """
    server, client = proto_hal_setup

    # Create mock policy model (fast inference)
    class MockPolicyModel:
        def __init__(self):
            self.action_dim = 12
            self.inference_count = 0

        def inference(self, model_io):
            from compute.parkour.parkour_types import InferenceResponse

            self.inference_count += 1
            # Return zero action tensor
            action = torch.zeros(self.action_dim, dtype=torch.float32)
            return InferenceResponse.create_success(
                action=action,
                inference_latency_ms=2.0,  # Fast mock inference
                model_version="test",
            )

    model = MockPolicyModel()
    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    # Set navigation command on test runner
    nav_cmd = NavigationCommand.create_now(vx=1.0, vy=0.0, yaw_rate=0.0)
    test_runner.set_navigation_command(nav_cmd)

    # Start publishing observations
    server.start_publishing(rate_hz=100.0)

    # Run game loop for ~1.1 seconds at 100 Hz (expecting ~100-110 ticks)
    # Using 1.1 seconds instead of exactly 1.0 to account for timing variations
    def stop_after_time():
        time.sleep(1.1)  # Slightly more than 1 second to ensure we get at least 100 ticks
        test_runner.stop()

    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()

    # Production code (InferenceTestRunner.run()) handles all exceptions internally
    # No exception handling needed here - if run() completes, it succeeded
    # If we need to handle expected exceptions in the future, add them here:
    # try:
    #     test_runner.run()
    # except ExpectedExceptionType:
    #     # Handle expected exception
    #     pass
    test_runner.run()

    stop_thread.join(timeout=2.0)

    # Verify we got approximately 100 ticks (allowing for timing variations)
    # At 100 Hz for 1.1 seconds, we expect ~100-110 ticks
    # Allow ±5 ticks for timing variations (thread scheduling, sleep precision, etc.)
    # This verifies the system runs at approximately the expected rate
    assert (
        95 <= test_runner.frame_count <= 110
    ), (
        f"Expected approximately 100 ticks (95-110 range), "
        f"got {test_runner.frame_count}. "
        f"This indicates the system may not be running at the expected 100 Hz rate."
    )

    # Verify inference was called approximately the expected number of times
    # Inference count should match frame count (one inference per successful frame)
    assert (
        95 <= model.inference_count <= 110
    ), (
        f"Expected approximately 100 inferences (95-110 range), "
        f"got {model.inference_count}. "
        f"This should match frame_count ({test_runner.frame_count}) - "
        f"one inference per successful frame."
    )

    # Verify no significant stalls (all frames should complete in reasonable time)
    # This is a basic check - more sophisticated timing analysis could be added
    assert test_runner.frame_count > 0, "Inference test should have executed at least one frame"


def _find_checkpoint_path() -> Path:
    """Find checkpoint path.
    
    Uses PARKOUR_CHECKPOINT_PATH environment variable (should point to folder).
    Looks for unitree_go2_parkour_teacher.pt in that folder.
    
    Returns:
        Path to checkpoint file
        
    Raises:
        FileNotFoundError: If environment variable not set or checkpoint not found
    """
    checkpoint_name = "unitree_go2_parkour_teacher.pt"
    
    env_path = os.getenv("PARKOUR_CHECKPOINT_PATH")
    if not env_path:
        raise FileNotFoundError(
            "PARKOUR_CHECKPOINT_PATH environment variable is not set. "
            "Set it to the path of the checkpoint folder."
        )
    
    checkpoint_dir = Path(env_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
        )
    
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
        )
    
    return checkpoint_path


@pytest.mark.isaacsim
def test_inference_latency_requirement(proto_hal_setup):
    """Test that inference latency meets < 15ms requirement with real model.

    This test measures real inference latency over multiple runs and verifies
    that average and p99 latency meet the < 15ms requirement. It uses an actual
    checkpoint to test real inference performance.

    The test uses GPU if available (CUDA), falling back to CPU. The < 15ms requirement
    is enforced when running on GPU. CPU inference may not meet this requirement,
    which is expected.

    Checkpoint path can be configured via PARKOUR_CHECKPOINT_PATH environment variable.
    Default: searches for parkour/assets/weights/unitree_go2_parkour_teacher.pt in common locations.
    
    This test is marked with @pytest.mark.isaacsim and should be run on the
    isaacsim image using 'make test-isaacsim'.
    """
    server, client = proto_hal_setup

    # Find checkpoint path
    try:
        checkpoint_path = _find_checkpoint_path()
    except FileNotFoundError as e:
        pytest.fail(str(e))

    # Determine device: use GPU if available, fallback to CPU
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        enforce_latency_requirement = True
    else:
        device = "cpu"
        device_name = "CPU"
        enforce_latency_requirement = False

    # Create model weights config
    weights = ModelWeights(
        checkpoint_path=str(checkpoint_path),
        action_dim=12,
        obs_dim=753,  # num_prop(53) + num_scan(132) + num_priv_explicit(9) + num_priv_latent(29) + history(530)
        model_version="teacher",
    )

    # Load the model
    try:
        model = ParkourPolicyModel(weights, device=device)
    except Exception as e:
        pytest.fail(f"Failed to load checkpoint: {e}")

    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    server.start_publishing(rate_hz=100.0)

    # Collect latencies
    latencies = []

    # Monkey-patch to collect latencies (model.inference already measures latency,
    # but we want to measure end-to-end including any overhead)
    original_inference = model.inference

    def inference_with_timing(io):
        start_time = time.time_ns()
        result = original_inference(io)
        end_time = time.time_ns()
        actual_latency_ms = (end_time - start_time) / 1_000_000.0
        latencies.append(actual_latency_ms)
        return result

    model.inference = inference_with_timing

    # Run for 0.5 seconds to collect latency samples
    def stop_after_time():
        time.sleep(0.5)
        test_runner.stop()

    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()

    # Production code (InferenceTestRunner.run()) handles all exceptions internally
    # No exception handling needed here - if run() completes, it succeeded
    # If we need to handle expected exceptions in the future, add them here:
    # try:
    #     test_runner.run()
    # except ExpectedExceptionType:
    #     # Handle expected exception
    #     pass
    test_runner.run()

    stop_thread.join(timeout=1.0)

    # Analyze latencies
    if len(latencies) == 0:
        pytest.fail("No latency samples collected - inference may not have run")

    latencies_array = np.array(latencies)
    avg_latency = np.mean(latencies_array)
    p50_latency = np.percentile(latencies_array, 50)
    p95_latency = np.percentile(latencies_array, 95)
    p99_latency = np.percentile(latencies_array, 99)
    max_latency = np.max(latencies_array)

    # Log statistics
    print(f"\nInference Latency Statistics ({device_name}):")
    print(f"  Device: {device_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples: {len(latencies_array)}")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  P50: {p50_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")

    # Verify requirements (only enforce on GPU)
    if enforce_latency_requirement:
        assert (
            avg_latency < 15.0
        ), (
            f"Average latency {avg_latency:.2f}ms exceeds 15ms requirement on GPU. "
            f"This indicates inference is too slow for real-time control at 100 Hz."
        )
        assert (
            p99_latency < 15.0
        ), (
            f"P99 latency {p99_latency:.2f}ms exceeds 15ms requirement on GPU. "
            f"This indicates inference tail latency is too high for real-time control."
        )
    else:
        # On CPU, just log that requirement is not enforced
        print(f"\nNote: Running on CPU - < 15ms latency requirement not enforced.")
        print(f"      GPU inference is required to meet the < 15ms target.")
        if avg_latency >= 15.0 or p99_latency >= 15.0:
            print(f"      Current latency would not meet requirement (avg: {avg_latency:.2f}ms, p99: {p99_latency:.2f}ms)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

