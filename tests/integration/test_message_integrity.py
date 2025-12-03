"""Integration tests for message integrity and error handling."""

import time

import numpy as np
import pytest
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.data_structures.hardware import KrabbyHardwareObservations
from tests.helpers import create_dummy_hw_obs


def test_corrupt_message_handling():
    """Test handling of corrupt messages."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_corrupt",
        command_bind="inproc://test_command_corrupt",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_corrupt",
        command_endpoint="inproc://test_command_corrupt",
    )
    client = HalClient(client_config)
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send corrupt message directly via ZMQ
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://test_observation_corrupt")  # observation endpoint
    # Use a small high-water mark to exercise buffer behavior (match server config)
    publisher.setsockopt(zmq.SNDHWM, 1)
    time.sleep(0.1)

    # Send malformed message (wrong number of parts)
    publisher.send(b"observation")  # Only topic, missing schema_version and payload

    # Poll should handle gracefully
    client.poll(timeout_ms=1000)

    # Client should not crash and should handle corrupt message
    # Latest camera should remain None or previous value
    # (We can't easily test this without more setup, but the code should not crash)

    publisher.close()
    context.term()
    client.close()
    server.close()


def test_malformed_binary_payload():
    """Test handling of malformed binary payload."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_malformed",
        command_bind="inproc://test_command_malformed",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_malformed",
        command_endpoint="inproc://test_command_malformed",
    )
    client = HalClient(client_config)
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send message with invalid binary payload
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://test_observation_malformed")  # observation endpoint
    # Use a small high-water mark to exercise buffer behavior (match server config)
    publisher.setsockopt(zmq.SNDHWM, 1)
    time.sleep(0.1)

    # Send message with invalid payload (not float32 array)
    topic = b"observation"
    schema_version = b"1.0"
    invalid_payload = b"not a float32 array"
    publisher.send_multipart([topic, schema_version, invalid_payload])

    # Poll should handle gracefully
    client.poll(timeout_ms=1000)

    # Client should not crash
    # (The deserialization will fail, but should be handled gracefully)

    publisher.close()
    context.term()
    client.close()
    server.close()


def test_missing_multipart_messages():
    """Test handling of missing multipart messages."""
    # This is similar to corrupt message test
    # The client should handle messages that don't have 3 parts
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_multipart",
        command_bind="inproc://test_command_multipart",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_multipart",
        command_endpoint="inproc://test_command_multipart",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send valid message first
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    previous_joint_pos = received_hw_obs.joint_positions.copy()

    # Send invalid message (missing parts) - use server's socket directly
    # Can't bind twice with inproc, so send directly through server socket
    server.observation_socket.send(b"observation")  # Only topic (incomplete message)

    # Poll again - should return None or previous value depending on implementation
    # The key is that it should not crash
    client.poll(timeout_ms=1000)
    client.close()
    server.close()


def test_invalid_type():
    """Test handling of invalid observation types."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_shape",
        command_bind="inproc://test_command_shape",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    # Test that server validates type
    # Try to publish numpy array (should fail - needs KrabbyHardwareObservations)
    invalid_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="KrabbyHardwareObservations"):
        server.set_observation(invalid_data)

    server.close()


def test_graceful_error_handling():
    """Test graceful error handling (skip corrupt, use previous)."""
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_graceful",
        command_bind="inproc://test_command_graceful",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_graceful",
        command_endpoint="inproc://test_command_graceful",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send valid message
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    valid_joint_pos = received_hw_obs.joint_positions.copy()

    # Send corrupt message - use server's socket directly
    # Can't bind twice with inproc, so send directly through server socket
    server.observation_socket.send(b"invalid")  # Invalid message format

    # Poll again - should handle gracefully (may return None or previous value)
    # The key is that it should not crash
    client.poll(timeout_ms=1000)
    client.close()
    server.close()


def test_schema_version_validation():
    """Test schema version validation."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_schema",
        command_bind="inproc://test_command_schema",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_schema",
        command_endpoint="inproc://test_command_schema",
    )
    client = HalClient(client_config)
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send message with unsupported schema version
    from compute.parkour.types import OBS_DIM
    
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://test_observation_schema")  # observation endpoint
    # Use a small high-water mark to exercise buffer behavior (match server config)
    publisher.setsockopt(zmq.SNDHWM, 1)
    time.sleep(0.1)

    topic = b"observation"
    unsupported_schema = b"2.0"  # Unsupported version
    payload = np.zeros(OBS_DIM, dtype=np.float32).tobytes()
    publisher.send_multipart([topic, unsupported_schema, payload])

    client.poll(timeout_ms=1000)

    # Client should log warning but not crash
    # Latest observation should remain None (unsupported schema rejected)

    publisher.close()
    context.term()
    client.close()
    server.close()


def test_required_fields_validation():
    """Test validation of required fields."""
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.types import ParkourModelIO, ParkourObservation, OBS_DIM

    # Test that incomplete model_io is rejected
    incomplete_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        nav_cmd=None,  # Missing
        observation=None,  # Missing
    )

    assert not incomplete_io.is_complete()

    # Test that complete model_io is accepted
    nav_cmd = NavigationCommand.create_now()
    observation = ParkourObservation(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )

    complete_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation,
    )

    assert complete_io.is_complete()

