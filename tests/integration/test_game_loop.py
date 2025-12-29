"""Integration tests for inference logic (game loop)."""

import threading
import time

import numpy as np
import pytest
import torch
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import OBS_DIM
from hal.client.data_structures.hardware import HardwareObservations
from compute.testing.inference_test_runner import InferenceTestRunner
from tests.helpers import create_dummy_hw_obs


class ProtoHalServer(HalServerBase):
    """Proto HAL server for testing - publishes synthetic observations in training format.

    This is a minimal HAL server that publishes observations matching the training format
    exactly: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
    """

    def __init__(self, config):
        """Initialize proto HAL server.
        
        Args:
            config: HAL server configuration
        """
        super().__init__(config)
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
                    if not self._running:
                        # Expected during shutdown - exit loop gracefully
                        break
                    # Unexpected exception while running - log and continue
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
        """Set/publish synthetic hardware observation.

        Creates hardware observation with synthetic data for testing.
        """
        # Create synthetic hardware observation
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        # Fill joint positions with test pattern
        hw_obs.joint_positions[:] = 0.1

        # Publish via base class
        super().set_observation(hw_obs)


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

        from compute.parkour.parkour_types import InferenceResponse

        # Return action tensor directly (matching inference output format)
        action = torch.zeros(self.action_dim, dtype=torch.float32)
        return InferenceResponse.create_success(
            action=action,
            timing_breakdown=[],
        )


@pytest.fixture
def hal_setup():
    """Setup HAL server and client with shared ZMQ context for testing."""
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs",
        command_bind="inproc://test_command",
    )
    server = ProtoHalServer(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs",
        command_endpoint="inproc://test_command",
    )
    # Use shared ZMQ context from server for inproc connections
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Wait briefly for inproc connection to be established
    # Publish and poll to establish connection
    server.publish_observation()
    client.poll(timeout_ms=100)

    yield server, client

    # Cleanup
    server.stop_publishing()
    client.close()
    server.close()


def test_game_loop_basic_functionality(hal_setup):
    """Test basic inference logic (game loop) functionality with mock HAL server."""
    server, client = hal_setup

    # Setup mock model
    model = MockPolicyModel(action_dim=12, inference_time_ms=5.0)

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    # Set navigation command on test runner (not client)
    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    # Start publishing observations (handles threading internally)
    server.start_publishing(rate_hz=100.0)

    # Run inference test for a short time
    def stop_after_time():
        time.sleep(0.2)  # Run for 200ms
        test_runner.stop()

    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()

    test_runner.run()

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
    from compute.parkour.parkour_types import (
        NUM_PROP,
        NUM_SCAN,
        NUM_PRIV_EXPLICIT,
        NUM_PRIV_LATENT,
        HISTORY_DIM,
    )
    
    server, client = hal_setup

    # Create hardware observation with test data
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    # Fill joint positions with test pattern
    hw_obs.joint_positions[:] = 1.0
    
    # Publish hardware observation
    server.set_observation(hw_obs)
    time.sleep(0.1)
    # Poll with timeout to receive observation
    received_hw_obs = client.poll(timeout_ms=1000)
    
    # Verify hardware observation was received
    assert received_hw_obs is not None, "Hardware observation should be received"
    
    # Verify hardware observation structure
    assert received_hw_obs.joint_positions.shape == (18,), \
        f"Joint positions shape should be (18,), got {received_hw_obs.joint_positions.shape}"
    assert received_hw_obs.joint_positions.dtype == np.float32, \
        f"Joint positions dtype should be float32, got {received_hw_obs.joint_positions.dtype}"
    
    # Map to ParkourObservation to verify structure
    from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
    mapper = HWObservationsToParkourMapper()
    parkour_obs = mapper.map(received_hw_obs)
    
    # Verify ParkourObservation structure
    assert parkour_obs.observation.shape == (OBS_DIM,), \
        f"ParkourObservation shape should be ({OBS_DIM},), got {parkour_obs.observation.shape}"
    assert parkour_obs.observation.dtype == np.float32, \
        f"ParkourObservation dtype should be float32, got {parkour_obs.observation.dtype}"
    
    # Verify component dimensions using view methods
    prop = parkour_obs.get_proprioceptive()
    assert prop.shape == (NUM_PROP,), \
        f"Proprioceptive shape should be ({NUM_PROP},), got {prop.shape}"
    
    scan = parkour_obs.get_scan()
    assert scan.shape == (NUM_SCAN,), \
        f"Scan shape should be ({NUM_SCAN},), got {scan.shape}"
    
    priv_explicit = parkour_obs.get_priv_explicit()
    assert priv_explicit.shape == (NUM_PRIV_EXPLICIT,), \
        f"Privileged explicit shape should be ({NUM_PRIV_EXPLICIT},), got {priv_explicit.shape}"
    
    priv_latent = parkour_obs.get_priv_latent()
    assert priv_latent.shape == (NUM_PRIV_LATENT,), \
        f"Privileged latent shape should be ({NUM_PRIV_LATENT},), got {priv_latent.shape}"
    
    history = parkour_obs.get_history()
    assert history.shape == (HISTORY_DIM,), \
        f"History shape should be ({HISTORY_DIM},), got {history.shape}"
    
    # Verify total dimension matches sum of components
    total_dim = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + HISTORY_DIM
    assert total_dim == OBS_DIM, \
        f"Component dimensions sum to {total_dim}, but OBS_DIM is {OBS_DIM}"


