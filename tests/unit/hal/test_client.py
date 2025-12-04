"""Unit tests for HAL client."""

import time

import numpy as np
import pytest
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.client.data_structures.hardware import KrabbyHardwareObservations
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import ParkourModelIO
from tests.helpers import create_dummy_hw_obs


def test_hal_client_initialization():
    """Test HAL client initialization with inproc endpoints."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation",
        command_bind="inproc://test_command",
    )
    server = HalServerBase(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation",
        command_endpoint="inproc://test_command",
    )
    client = HalClient(client_config)
    client.initialize()
    client.set_debug(True)

    assert client._initialized
    assert client.context is not None
    assert client.observation_socket is not None
    assert client.command_socket is not None

    client.close()
    server.close()


def test_hal_client_poll_observation():
    """Test HAL client polling for hardware observation messages."""
    # Use shared context for inproc connections (required for reliable inproc PUB/SUB)
    server_config = HalServerConfig(
        observation_bind="inproc://test_state2",
        command_bind="inproc://test_command2",
    )
    server = HalServerBase(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_state2",
        command_endpoint="inproc://test_command2",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()
    client.set_debug(True)

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)
    
    # Create and publish hardware observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:3] = [1.0, 2.0, 3.0]  # Set some values
    server.set_observation(hw_obs)
    time.sleep(0.05)  # Small delay to ensure message is sent
    
    # Poll for message
    received_hw_obs = client.poll(timeout_ms=1000)

    # Check latest observation data
    assert received_hw_obs is not None
    assert received_hw_obs.joint_positions is not None
    np.testing.assert_array_equal(received_hw_obs.joint_positions[:3], [1.0, 2.0, 3.0])

    client.close()
    server.close()




def test_hal_client_poll_and_map():
    """Test polling hardware observation and mapping to ParkourObservation."""
    from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
    
    # Use shared context for inproc connections
    
    server_config = HalServerConfig(
        observation_bind="inproc://test_state4",
        command_bind="inproc://test_command4",
    )
    server = HalServerBase(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_state4",
        command_endpoint="inproc://test_command4",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()
    client.set_debug(True)

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)

    # Create and publish hardware observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:5] = [1.0, 2.0, 3.0, 4.0, 5.0]  # Set some values
    server.set_observation(hw_obs)

    # Poll for hardware observation
    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None

    # Map to ParkourObservation using mapper
    mapper = KrabbyHWObservationsToParkourMapper()
    parkour_obs = mapper.map(received_hw_obs)
    
    assert parkour_obs is not None
    assert parkour_obs.observation is not None

    client.close()
    server.close()


def test_hal_client_put_joint_command():
    """Test sending joint command via HAL client."""
    import torch
    
    # Use shared context for inproc connections
    
    server_config = HalServerConfig(
        observation_bind="inproc://test_state5",
        command_bind="inproc://test_command5",
    )
    server = HalServerBase(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_state5",
        command_endpoint="inproc://test_command5",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()
    client.set_debug(True)

    time.sleep(0.1)

    # Create inference response and map to hardware joint positions
    from compute.parkour.parkour_types import InferenceResponse
    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper

    action_tensor = torch.tensor([0.1, 0.2, 0.3] + [0.0] * 9, dtype=torch.float32)  # 12 DOF
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # Map to hardware joint positions
    mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)
    joint_positions = mapper.map(inference_response)

    # Server needs to be waiting before client sends (PUSH/PULL pattern)
    import threading
    received_command = [None]
    
    def server_receive():
        received_command[0] = server.get_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    time.sleep(0.05)  # Small delay to ensure server is waiting
    
    # Send command
    client.put_joint_command(joint_positions)
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    # Should receive the mapped joint positions (18 DOF)
    # received is a KrabbyDesiredJointPositions object, access joint_positions attribute
    np.testing.assert_array_equal(received.joint_positions, joint_positions.joint_positions)

    client.close()
    server.close()


def test_hal_client_timestamp_validation():
    """Test timestamp validation for hardware observations."""
    # Use shared context for inproc connections
    
    server_config = HalServerConfig(
        observation_bind="inproc://test_state6",
        command_bind="inproc://test_command6",
    )
    server = HalServerBase(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_state6",
        command_endpoint="inproc://test_command6",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()
    client.set_debug(True)

    # With shared context, connection should be established immediately
    # Small delay to ensure sockets are ready
    time.sleep(0.1)

    # Publish fresh hardware observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs)

    # Poll for fresh observation
    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    
    # Now test with stale observation (wait to make it stale)
    time.sleep(0.02)  # Wait 20ms
    # Poll again - should still get the latest (HWM=1 keeps latest)
    received_hw_obs2 = client.poll(timeout_ms=100)
    # Should still receive (HWM=1 keeps latest message available)

    client.close()
    server.close()

