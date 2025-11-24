"""Integration tests for inference logic (game loop)."""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import zmq

from HAL.ZMQ.client import HalClient
from HAL.ZMQ.server import HalServerBase
from HAL.config import HalClientConfig, HalServerConfig
from HAL.telemetry.types import NavigationCommand, OBS_DIM
from compute.testing.inference_test_runner import InferenceTestRunner


class ProtoHalServer(HalServerBase):
    """Proto HAL server for testing - publishes synthetic observations in training format.

    This is a minimal HAL server that publishes observations matching the training format
    exactly: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
    """

    def __init__(self, config, context=None):
        """Initialize proto HAL server.
        
        Args:
            config: HAL server configuration
            context: Optional shared ZMQ context (useful for inproc connections)
        """
        super().__init__(config, context=context)
        self.tick_count = 0
        self._running = False
        self._publish_thread = None
        self._command_thread = None

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
            while self._running:
                self.publish_telemetry()
                self.tick_count += 1
                time.sleep(period)

        def command_loop():
            """Handle incoming commands in a loop."""
            while self._running:
                try:
                    # Poll for commands (non-blocking with short timeout)
                    command = self.recv_joint_command(timeout_ms=10)
                    if command is not None:
                        # Command received and acknowledged by recv_joint_command
                        pass
                except Exception:
                    # Ignore errors in command handling thread
                    pass

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

    def publish_telemetry(self):
        """Publish synthetic observation in training format.

        Creates observation array matching training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        """
        # Create synthetic observation in training format
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)

        # Fill with synthetic data (simple patterns for testing)
        num_prop = 53
        num_scan = 132
        num_priv_explicit = 9
        num_priv_latent = 29
        history_dim = 530

        # Fill each section with distinct values for testing
        obs_array[:num_prop] = 0.1  # Proprioceptive
        obs_array[num_prop : num_prop + num_scan] = 0.2  # Scan
        start = num_prop + num_scan
        obs_array[start : start + num_priv_explicit] = 0.3  # Priv explicit
        start += num_priv_explicit
        obs_array[start : start + num_priv_latent] = 0.4  # Priv latent
        obs_array[-history_dim:] = 0.5  # History

        # Publish via base class
        self.publish_observation(obs_array)


class MockPolicyModel:
    """Mock policy model for testing."""

    def __init__(self, action_dim: int = 12, inference_time_ms: float = 5.0):
        """Initialize mock model.

        Args:
            action_dim: Action dimension
            inference_time_ms: Simulated inference time in milliseconds
        """
        self.action_dim = action_dim
        self.inference_time_ms = inference_time_ms
        self.inference_count = 0

    def inference(self, model_io):
        """Mock inference."""
        time.sleep(self.inference_time_ms / 1000.0)  # Simulate inference time
        self.inference_count += 1

        from HAL.commands.types import InferenceResponse

        # Return action tensor directly (matching inference output format)
        action = torch.zeros(self.action_dim, dtype=torch.float32)
        return InferenceResponse.create_success(
            action=action,
            inference_latency_ms=self.inference_time_ms,
            model_version="test",
        )


@pytest.fixture
def hal_setup():
    """Setup HAL server and client with shared ZMQ context for testing."""
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_obs",
        command_bind="inproc://test_command",
    )
    server = ProtoHalServer(server_config, context=shared_context)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_obs",
        command_endpoint="inproc://test_command",
    )
    client = HalClient(client_config, context=shared_context)
    client.initialize()

    # Wait for connection and publish a dummy observation to establish connection
    time.sleep(0.1)
    dummy_obs = np.zeros(OBS_DIM, dtype=np.float32)
    server.publish_observation(dummy_obs)
    time.sleep(0.05)
    client.poll(timeout_ms=100)

    yield server, client, shared_context

    # Cleanup
    server.stop_publishing()
    client.close()
    server.close()
    shared_context.term()


def test_game_loop_basic_functionality(hal_setup):
    """Test basic inference logic (game loop) functionality with mock HAL server."""
    server, client, shared_context = hal_setup

    # Setup mock model
    model = MockPolicyModel(action_dim=12, inference_time_ms=5.0)

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    # Set navigation command
    nav_cmd = NavigationCommand.create_now()
    client.set_navigation_command(nav_cmd)

    # Start publishing observations (handles threading internally)
    server.start_publishing(rate_hz=100.0)

    # Run inference test for a short time
    def stop_after_time():
        time.sleep(0.2)  # Run for 200ms
        test_runner.stop()

    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()

    try:
        test_runner.run()
    except Exception:
        pass  # Expected when stopped

    stop_thread.join(timeout=1.0)

    # Verify inference test ran
    assert test_runner.frame_count > 0


def test_game_loop_observation_tensor_correctness(hal_setup):
    """Test that observation tensor structure matches training format exactly.
    
    Verifies:
    - Total observation dimension (OBS_DIM = 753)
    - Component dimensions (prop=53, scan=132, priv_explicit=9, priv_latent=29, history=530)
    - View methods correctly extract each component
    - Data type is float32
    """
    from HAL.telemetry.types import (
        NUM_PROP,
        NUM_SCAN,
        NUM_PRIV_EXPLICIT,
        NUM_PRIV_LATENT,
        HISTORY_DIM,
    )
    
    server, client, shared_context = hal_setup

    # Create observation with distinct values for each component to verify structure
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    # Fill each component with distinct values
    observation[:NUM_PROP] = 1.0  # Proprioceptive
    observation[NUM_PROP : NUM_PROP + NUM_SCAN] = 2.0  # Scan
    start = NUM_PROP + NUM_SCAN
    observation[start : start + NUM_PRIV_EXPLICIT] = 3.0  # Priv explicit
    start += NUM_PRIV_EXPLICIT
    observation[start : start + NUM_PRIV_LATENT] = 4.0  # Priv latent
    observation[-HISTORY_DIM:] = 5.0  # History
    
    server.publish_observation(observation)
    time.sleep(0.1)
    client.poll(timeout_ms=1000)
    
    # Verify observation was received
    assert client._latest_observation is not None, "Observation should be received"
    obs = client._latest_observation
    
    # Verify total shape and dtype
    assert obs.observation.shape == (OBS_DIM,), \
        f"Observation shape should be ({OBS_DIM},), got {obs.observation.shape}"
    assert obs.observation.dtype == np.float32, \
        f"Observation dtype should be float32, got {obs.observation.dtype}"
    
    # Verify component dimensions using view methods
    prop = obs.get_proprioceptive()
    assert prop.shape == (NUM_PROP,), \
        f"Proprioceptive shape should be ({NUM_PROP},), got {prop.shape}"
    assert np.allclose(prop, 1.0), "Proprioceptive values should be 1.0"
    
    scan = obs.get_scan()
    assert scan.shape == (NUM_SCAN,), \
        f"Scan shape should be ({NUM_SCAN},), got {scan.shape}"
    assert np.allclose(scan, 2.0), "Scan values should be 2.0"
    
    priv_explicit = obs.get_priv_explicit()
    assert priv_explicit.shape == (NUM_PRIV_EXPLICIT,), \
        f"Privileged explicit shape should be ({NUM_PRIV_EXPLICIT},), got {priv_explicit.shape}"
    assert np.allclose(priv_explicit, 3.0), "Privileged explicit values should be 3.0"
    
    priv_latent = obs.get_priv_latent()
    assert priv_latent.shape == (NUM_PRIV_LATENT,), \
        f"Privileged latent shape should be ({NUM_PRIV_LATENT},), got {priv_latent.shape}"
    assert np.allclose(priv_latent, 4.0), "Privileged latent values should be 4.0"
    
    history = obs.get_history()
    assert history.shape == (HISTORY_DIM,), \
        f"History shape should be ({HISTORY_DIM},), got {history.shape}"
    assert np.allclose(history, 5.0), "History values should be 5.0"
    
    # Verify total dimension matches sum of components
    total_dim = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + HISTORY_DIM
    assert total_dim == OBS_DIM, \
        f"Component dimensions sum to {total_dim}, but OBS_DIM is {OBS_DIM}"


def test_game_loop_inference_latency():
    """Test inference latency is measured correctly."""
    model = MockPolicyModel(action_dim=12, inference_time_ms=8.0)

    # Create a simple inference response
    from HAL.telemetry.types import ParkourModelIO

    # Mock model_io
    model_io = MagicMock(spec=ParkourModelIO)

    start_time = time.time()
    result = model.inference(model_io)
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000.0

    assert result.success
    assert result.inference_latency_ms > 0
    assert abs(result.inference_latency_ms - elapsed_ms) < 2.0  # Within 2ms tolerance
    assert result.inference_latency_ms < 15.0  # Should be under 15ms target
    assert result.action is not None
    assert result.action.shape == (12,)



