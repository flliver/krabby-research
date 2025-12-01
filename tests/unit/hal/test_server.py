"""Unit tests for HAL server."""

import time

import numpy as np
import pytest
import zmq

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import KrabbyHardwareObservations
from tests.helpers import create_dummy_hw_obs


def test_hal_server_initialization():
    """Test HAL server initialization with inproc endpoints."""
    config = HalServerConfig(
        observation_bind="inproc://test_observation",
        command_bind="inproc://test_command",
    )
    server = HalServerBase(config)
    server.initialize()

    assert server._initialized
    assert server.context is not None
    assert server.observation_socket is not None
    assert server.command_socket is not None

    server.close()


def test_hal_server_context_manager():
    """Test HAL server as context manager."""
    config = HalServerConfig(
        observation_bind="inproc://test_state2",
        command_bind="inproc://test_command2",
    )
    with HalServerBase(config) as server:
        assert server._initialized


def test_set_observation():
    """Test setting/publishing hardware observation."""
    config = HalServerConfig(
        observation_bind="inproc://test_state3",
        command_bind="inproc://test_command3",
    )

    with HalServerBase(config) as server:
        # Create subscriber to receive message (use server's transport context for inproc)
        transport_context = server.get_transport_context()
        subscriber = transport_context.socket(zmq.SUB)
        subscriber.connect("inproc://test_state3")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"observation")
        subscriber.setsockopt(zmq.RCVHWM, 1)
        time.sleep(0.2)  # Give subscriber time to connect

        # Set hardware observation
        from tests.helpers import create_dummy_hw_obs
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        server.set_observation(hw_obs)
        time.sleep(0.1)  # Small delay to ensure message is sent

        # Receive message with timeout
        poll_result = subscriber.poll(500, zmq.POLLIN)
        assert poll_result > 0, "No message received within timeout"
        
        parts = subscriber.recv_multipart(zmq.NOBLOCK)
        
        # Validate message format
        assert len(parts) == 8, f"Expected 8 parts (topic + schema + 6 hw_obs), got {len(parts)}"
        assert parts[0] == b"observation", f"Expected topic 'observation', got {parts[0]}"
        assert parts[1] == b"1.0", f"Expected schema '1.0', got {parts[1]}"
        
        # Extract hw_obs parts (skip topic and schema)
        # Parts: [0:topic, 1:schema, 2:metadata, 3-7:arrays]
        hw_obs_parts = parts[2:]  # Skip topic and schema
        assert len(hw_obs_parts) == 6, f"Expected 6 parts for hw_obs (metadata + 5 arrays), got {len(hw_obs_parts)}"
        
        # Deserialize and validate
        received_hw_obs = KrabbyHardwareObservations.from_bytes(hw_obs_parts)
        assert received_hw_obs.joint_positions.shape == hw_obs.joint_positions.shape
        np.testing.assert_array_equal(received_hw_obs.joint_positions, hw_obs.joint_positions)

        subscriber.close()




def test_get_joint_command():
    """Test getting joint command."""
    # Use shared context for inproc connections
    
    config = HalServerConfig(
        observation_bind="inproc://test_state5",
        command_bind="inproc://test_command5",
    )

    with HalServerBase(config) as server:
        # Create requester to send command (use server's transport context for inproc)
        transport_context = server.get_transport_context()
        requester = transport_context.socket(zmq.REQ)
        requester.connect("inproc://test_command5")
        time.sleep(0.1)  # Give requester time to connect

        # Server needs to be waiting before client sends (REQ/REP pattern)
        import threading
        received_command = [None]
        
        def server_receive():
            received_command[0] = server.get_joint_command(timeout_ms=2000)
        
        server_thread = threading.Thread(target=server_receive)
        server_thread.start()
        time.sleep(0.05)  # Small delay to ensure server is waiting

        # Send command
        command = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        requester.send(command.tobytes())

        server_thread.join(timeout=2.0)
        received = received_command[0]
        assert received is not None
        np.testing.assert_array_equal(received, command)

        # Check acknowledgement
        ack = requester.recv()
        assert ack == b"ok"

        requester.close()


def test_hwm_behavior():
    """Test observation_buffer_size=1 behavior (latest-only semantics)."""
    # Use shared context for inproc connections
    
    config = HalServerConfig(
        observation_bind="inproc://test_state6",
        command_bind="inproc://test_command6",
        observation_buffer_size=1,
    )

    with HalServerBase(config) as server:
        # Create subscriber (use server's transport context for inproc)
        transport_context = server.get_transport_context()
        subscriber = transport_context.socket(zmq.SUB)
        subscriber.connect("inproc://test_state6")  # Match server bind address
        subscriber.setsockopt(zmq.SUBSCRIBE, b"observation")
        subscriber.setsockopt(zmq.RCVHWM, 1)
        time.sleep(0.1)  # Give subscriber time to connect

        # Publish multiple messages rapidly
        for i in range(10):
            hw_obs = create_dummy_hw_obs(
                camera_height=480, camera_width=640
            )
            hw_obs.joint_positions[:] = float(i)
            server.set_observation(hw_obs)
        time.sleep(0.1)  # Small delay to ensure messages are sent

        # With observation_buffer_size=1, subscriber should receive messages (with shared context, connection is reliable)
        received_count = 0
        while subscriber.poll(100, zmq.POLLIN):
            subscriber.recv_multipart()
            received_count += 1

        # Should receive at least one message
        assert received_count >= 1

        subscriber.close()


def test_error_handling_invalid_type():
    """Test error handling for invalid observation types."""
    config = HalServerConfig(
        observation_bind="inproc://test_state7",
        command_bind="inproc://test_command7",
    )

    with HalServerBase(config) as server:
        # Try to publish wrong type (should fail)
        invalid_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="KrabbyHardwareObservations"):
            server.set_observation(invalid_data)


def test_error_handling_not_initialized():
    """Test error handling when server not initialized."""
    config = HalServerConfig(
        observation_bind="inproc://test_state8",
        command_bind="inproc://test_command8",
    )
    server = HalServerBase(config)

    # Should raise error if not initialized
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        server.set_observation(hw_obs)

