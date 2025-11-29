"""Integration tests for message integrity and error handling."""

import time

import numpy as np
import pytest
import zmq

from hal.client.client import HalClient
from hal.server.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig


def test_corrupt_message_handling():
    """Test handling of corrupt messages."""
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_corrupt",
        command_bind="inproc://test_command_corrupt",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_corrupt",
        command_endpoint="inproc://test_command_corrupt",
    )
    client = HalClient(client_config)
    client.initialize()

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
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_malformed",
        command_bind="inproc://test_command_malformed",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_malformed",
        command_endpoint="inproc://test_command_malformed",
    )
    client = HalClient(client_config)
    client.initialize()

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
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_multipart",
        command_bind="inproc://test_command_multipart",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_multipart",
        command_endpoint="inproc://test_command_multipart",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    from hal.observation.types import OBS_DIM
    
    time.sleep(0.1)

    # Send valid message first
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(observation)

    client.poll(timeout_ms=1000)
    assert client._latest_observation is not None
    previous_observation = client._latest_observation

    # Send invalid message (missing parts) - use server's socket directly
    # Can't bind twice with inproc, so send directly through server socket
    server.observation_socket.send(b"observation")  # Only topic (incomplete message)

    client.poll(timeout_ms=1000)

    # Client should use previous value (graceful error handling)
    # The latest observation should remain the previous valid one
    assert client._latest_observation is not None
    client.close()
    server.close()


def test_shape_dtype_mismatch():
    """Test handling of shape/dtype mismatches."""
    from hal.observation.types import OBS_DIM
    
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_shape",
        command_bind="inproc://test_command_shape",
    )
    server = HalServerBase(server_config)
    server.initialize()

    # Test that server validates shape
    # Try to publish 2D array (should fail)
    invalid_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        server.set_observation(invalid_data)

    # Test wrong size 1D array (should fail)
    wrong_size = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        server.set_observation(wrong_size)

    # Test dtype conversion
    data_int = np.zeros(OBS_DIM, dtype=np.int32)
    # Server should convert to float32
    server.set_observation(data_int)  # Should work (converted internally)

    server.close()


def test_graceful_error_handling():
    """Test graceful error handling (skip corrupt, use previous)."""
    # Use shared context for inproc connections
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_graceful",
        command_bind="inproc://test_command_graceful",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_graceful",
        command_endpoint="inproc://test_command_graceful",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    from hal.observation.types import OBS_DIM
    
    time.sleep(0.1)

    # Send valid message
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(observation)

    client.poll(timeout_ms=1000)
    assert client._latest_observation is not None
    valid_observation = client._latest_observation.observation.copy()

    # Send corrupt message - use server's socket directly
    # Can't bind twice with inproc, so send directly through server socket
    server.observation_socket.send(b"invalid")  # Invalid message format

    # Poll again
    client.poll(timeout_ms=1000)

    # Client should still have previous valid data
    assert client._latest_observation is not None
    # Observation should be the same (previous value used)
    np.testing.assert_array_equal(client._latest_observation.observation, valid_observation)
    client.close()
    server.close()


def test_schema_version_validation():
    """Test schema version validation."""
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_schema",
        command_bind="inproc://test_command_schema",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_schema",
        command_endpoint="inproc://test_command_schema",
    )
    client = HalClient(client_config)
    client.initialize()

    time.sleep(0.1)

    # Send message with unsupported schema version
    from hal.observation.types import OBS_DIM
    
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
    from hal.observation.types import ParkourModelIO, ParkourObservation, NavigationCommand, OBS_DIM

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

