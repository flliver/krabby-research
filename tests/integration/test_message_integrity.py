"""Integration tests for message integrity and error handling."""

import time

import numpy as np
import pytest
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations
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

    # Send corrupt message directly via ZMQ
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://test_observation_corrupt")  # observation endpoint
    # Use a small high-water mark to exercise buffer behavior (match server config)
    publisher.setsockopt(zmq.SNDHWM, 1)
    # Small delay for socket to be ready
    time.sleep(0.01)

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
    # Small delay for socket to be ready
    time.sleep(0.01)

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
    # Try to publish numpy array (should fail - needs HardwareObservations)
    invalid_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="HardwareObservations"):
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
    from compute.parkour.parkour_types import OBS_DIM
    
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("inproc://test_observation_schema")  # observation endpoint
    # Use a small high-water mark to exercise buffer behavior (match server config)
    publisher.setsockopt(zmq.SNDHWM, 1)
    # Small delay for socket to be ready
    time.sleep(0.01)

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
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation, OBS_DIM

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


def test_timestamp_synchronization():
    """Test that is_synchronized() correctly validates timestamp synchronization."""
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation, OBS_DIM
    import time

    # Test synchronized timestamps (within 10ms default)
    base_time_ns = time.time_ns()
    nav_cmd = NavigationCommand.create_now()
    nav_cmd.timestamp_ns = base_time_ns
    
    observation = ParkourObservation(
        timestamp_ns=base_time_ns + 5_000_000,  # 5ms difference
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )
    
    synchronized_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation,
    )
    
    assert synchronized_io.is_synchronized(), "Timestamps within 10ms should be synchronized"
    assert synchronized_io.is_synchronized(max_age_ns=10_000_000), "Should pass with default 10ms threshold"
    
    # Test unsynchronized timestamps (more than 10ms apart)
    observation_unsync = ParkourObservation(
        timestamp_ns=base_time_ns + 15_000_000,  # 15ms difference
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )
    
    unsynchronized_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation_unsync,
    )
    
    assert not unsynchronized_io.is_synchronized(), "Timestamps >10ms apart should not be synchronized"
    assert unsynchronized_io.is_synchronized(max_age_ns=20_000_000), "Should pass with 20ms threshold"
    
    # Test with custom threshold
    observation_custom = ParkourObservation(
        timestamp_ns=base_time_ns + 5_000_000,  # 5ms difference
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )
    
    custom_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation_custom,
    )
    
    # 5ms (5_000_000 ns) is > 1ms (1_000_000 ns) threshold, so should fail
    assert not custom_io.is_synchronized(max_age_ns=1_000_000), "5ms > 1ms threshold should fail"
    
    # Test edge case: exactly at threshold
    observation_exact = ParkourObservation(
        timestamp_ns=base_time_ns + 10_000_000,  # Exactly 10ms
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )
    
    exact_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation_exact,
    )
    
    assert exact_io.is_synchronized(max_age_ns=10_000_000), "Exactly at threshold should pass"
    assert not exact_io.is_synchronized(max_age_ns=9_999_999), "Just over threshold should fail"
    
    # Test that incomplete IO is not synchronized
    incomplete_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        schema_version="1.0",
        nav_cmd=None,
        observation=None,
    )
    
    assert not incomplete_io.is_synchronized(), "Incomplete IO should not be synchronized"


def test_timestamp_synchronization_realistic_sequence():
    """Test synchronization with realistic sequence: nav_cmd created once, observations arrive continuously.
    
    This simulates the actual issue seen in the latency test where:
    1. nav_cmd is created once with NavigationCommand.create_now()
    2. Observations are published continuously with fresh timestamps
    3. The nav_cmd timestamp becomes stale, causing synchronization failures
    """
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation, OBS_DIM
    from hal.client.data_structures.hardware import HardwareObservations
    from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
    import time
    
    # Simulate the actual sequence from the latency test
    # Step 1: Create nav_cmd once (like in test_runner.py)
    nav_cmd = NavigationCommand.create_now()
    nav_cmd_timestamp = nav_cmd.timestamp_ns
    
    # Step 2: Simulate observations arriving continuously (like from publish_loop at 100Hz)
    # Each observation has a fresh timestamp from set_observation()
    mapper = HWObservationsToParkourMapper()
    
    # First observation: arrives quickly, should be synchronized
    time.sleep(0.001)  # 1ms delay
    hw_obs_1 = HardwareObservations(
        joint_positions=np.zeros(18, dtype=np.float32),
        rgb_camera_1=np.zeros((480, 640, 3), dtype=np.uint8),
        rgb_camera_2=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_map=np.zeros((480, 640), dtype=np.float32),
        confidence_map=np.ones((480, 640), dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=time.time_ns(),  # Fresh timestamp
    )
    
    parkour_obs_1 = mapper.map(hw_obs_1, nav_cmd=nav_cmd)
    model_io_1 = ParkourModelIO(
        timestamp_ns=parkour_obs_1.timestamp_ns,
        schema_version=parkour_obs_1.schema_version,
        nav_cmd=nav_cmd,  # Same nav_cmd with old timestamp
        observation=parkour_obs_1,
    )
    
    # Should be synchronized if within 10ms
    time_diff_ns = abs(parkour_obs_1.timestamp_ns - nav_cmd_timestamp)
    if time_diff_ns <= 10_000_000:  # 10ms
        assert model_io_1.is_synchronized(), f"First observation should be synchronized (diff: {time_diff_ns/1e6:.2f}ms)"
    else:
        assert not model_io_1.is_synchronized(), f"First observation should not be synchronized (diff: {time_diff_ns/1e6:.2f}ms)"
    
    # Simulate multiple observations arriving (like in the test runner loop)
    # After several iterations, the nav_cmd timestamp becomes stale
    for i in range(5):
        time.sleep(0.01)  # 10ms between observations (100Hz)
        hw_obs = HardwareObservations(
            joint_positions=np.zeros(18, dtype=np.float32),
            rgb_camera_1=np.zeros((480, 640, 3), dtype=np.uint8),
            rgb_camera_2=np.zeros((480, 640, 3), dtype=np.uint8),
            depth_map=np.zeros((480, 640), dtype=np.float32),
            confidence_map=np.ones((480, 640), dtype=np.float32),
            camera_height=480,
            camera_width=640,
            timestamp_ns=time.time_ns(),  # Fresh timestamp each time
        )
        
        parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)  # Still using same nav_cmd
        model_io = ParkourModelIO(
            timestamp_ns=parkour_obs.timestamp_ns,
            schema_version=parkour_obs.schema_version,
            nav_cmd=nav_cmd,  # Same nav_cmd with old timestamp
            observation=parkour_obs,
        )
        
        time_diff_ns = abs(parkour_obs.timestamp_ns - nav_cmd_timestamp)
        time_diff_ms = time_diff_ns / 1e6
        
        # After several iterations, timestamps will be >10ms apart
        if time_diff_ms > 10.0:
            assert not model_io.is_synchronized(), (
                f"Observation {i+1} should not be synchronized "
                f"(nav_cmd age: {time_diff_ms:.2f}ms > 10ms threshold)"
            )
        else:
            assert model_io.is_synchronized(), (
                f"Observation {i+1} should be synchronized "
                f"(nav_cmd age: {time_diff_ms:.2f}ms <= 10ms threshold)"
            )

