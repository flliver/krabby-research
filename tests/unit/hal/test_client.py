"""Unit tests for HAL client."""

import time

import numpy as np
import pytest
import zmq

from hal.zmq.client import HalClient
from hal.zmq.server import HalServerBase
from hal.config import HalClientConfig, HalServerConfig
from hal.observation.types import NavigationCommand


def test_hal_client_initialization():
    """Test HAL client initialization with inproc endpoints."""
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation",
        command_bind="inproc://test_command",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation",
        command_endpoint="inproc://test_command",
    )
    client = HalClient(client_config)
    client.initialize()

    assert client._initialized
    assert client.context is not None
    assert client.observation_socket is not None
    assert client.command_socket is not None

    client.close()
    server.close()


def test_hal_client_poll_observation():
    """Test HAL client polling for observation messages."""
    from hal.observation.types import OBS_DIM
    
    # Use shared context for inproc connections (required for reliable inproc PUB/SUB)
    shared_context = zmq.Context()
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state2",
        command_bind="inproc://test_command2",
    )
    server = HalServerBase(server_config, context=shared_context)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_state2",
        command_endpoint="inproc://test_command2",
    )
    client = HalClient(client_config, context=shared_context)
    client.initialize()

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)
    
    # Now publish and receive
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:3] = [1.0, 2.0, 3.0]  # Set some values
    server.publish_observation(observation)
    time.sleep(0.05)  # Small delay to ensure message is sent
    
    # Poll for message
    client.poll(timeout_ms=1000)

    # Check latest observation data
    assert client._latest_observation is not None
    assert client._latest_observation.observation is not None
    np.testing.assert_array_equal(client._latest_observation.observation, observation)

    client.close()
    server.close()




def test_hal_client_build_model_io():
    """Test building ParkourModelIO from latest telemetry."""
    from hal.observation.types import OBS_DIM
    
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state4",
        command_bind="inproc://test_command4",
    )
    server = HalServerBase(server_config, context=shared_context)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_state4",
        command_endpoint="inproc://test_command4",
    )
    client = HalClient(client_config, context=shared_context)
    client.initialize()

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)

    # Set navigation command
    nav_cmd = NavigationCommand.create_now(vx=1.0, vy=0.0, yaw_rate=0.0)
    client.set_navigation_command(nav_cmd)

    # Publish observation
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:5] = [1.0, 2.0, 3.0, 4.0, 5.0]  # Set some values
    server.publish_observation(observation)

    # Poll and build model IO
    client.poll(timeout_ms=1000)
    model_io = client.build_model_io()

    assert model_io is not None
    assert model_io.nav_cmd is not None
    assert model_io.observation is not None
    np.testing.assert_array_equal(model_io.observation.observation, observation)

    client.close()
    server.close()


def test_hal_client_send_joint_command():
    """Test sending joint command via HAL client."""
    import torch
    
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state5",
        command_bind="inproc://test_command5",
    )
    server = HalServerBase(server_config, context=shared_context)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_state5",
        command_endpoint="inproc://test_command5",
    )
    client = HalClient(client_config, context=shared_context)
    client.initialize()

    time.sleep(0.1)

    # Create inference response with action tensor (new format)
    from hal.commands.types import InferenceResponse

    action_tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # Server needs to be waiting before client sends (REQ/REP pattern)
    import threading
    received_command = [None]
    
    def server_receive():
        received_command[0] = server.recv_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    time.sleep(0.05)  # Small delay to ensure server is waiting
    
    # Send command
    success = client.send_joint_command(inference_response)
    assert success
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    np.testing.assert_array_equal(received, action_tensor.numpy())

    client.close()
    server.close()


def test_hal_client_timestamp_validation():
    """Test timestamp validation for stale messages."""
    from hal.observation.types import OBS_DIM
    
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state6",
        command_bind="inproc://test_command6",
    )
    server = HalServerBase(server_config, context=shared_context)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_state6",
        command_endpoint="inproc://test_command6",
    )
    client = HalClient(client_config, context=shared_context)
    client.initialize()

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)

    # Set navigation command with old timestamp (nav_cmd can be older, it's set once and reused)
    old_nav_cmd = NavigationCommand(
        timestamp_ns=time.time_ns() - 100_000_000,  # 100ms ago
        schema_version="1.0",
    )
    client.set_navigation_command(old_nav_cmd)

    # Publish fresh observation
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    server.publish_observation(observation)

    client.poll(timeout_ms=1000)

    # Build model IO with strict max_age (should succeed - nav_cmd can be older)
    # Only observation age is checked, not nav_cmd age
    model_io = client.build_model_io(max_age_ns=10_000_000)  # 10ms max age
    assert model_io is not None  # Should succeed because observation is fresh, nav_cmd can be older
    
    # Now test with stale observation
    time.sleep(0.02)  # Wait 20ms to make observation stale
    model_io_stale = client.build_model_io(max_age_ns=10_000_000)  # 10ms max age
    assert model_io_stale is None  # Should be None because observation is too old

    client.close()
    server.close()

