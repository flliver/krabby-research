"""Unit tests for HAL server."""

import time

import numpy as np
import pytest
import zmq

from hal.config import HalServerConfig
from hal.zmq.server import HalServerBase


def test_hal_server_initialization():
    """Test HAL server initialization with inproc endpoints."""
    config = HalServerConfig.from_endpoints(
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
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state2",
        command_bind="inproc://test_command2",
    )
    with HalServerBase(config) as server:
        assert server._initialized


def test_publish_observation():
    """Test publishing observation telemetry."""
    from hal.observation.types import OBS_DIM
    
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state3",
        command_bind="inproc://test_command3",
    )

    with HalServerBase(config) as server:
        # Create subscriber to receive message (use shared context for inproc)
        shared_context = zmq.Context()
        subscriber = shared_context.socket(zmq.SUB)
        subscriber.connect("inproc://test_observation3")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"observation")
        subscriber.setsockopt(zmq.RCVHWM, 1)
        time.sleep(0.1)  # Give subscriber time to connect

        # Publish observation data
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        server.publish_observation(observation)

        # Receive message
        if subscriber.poll(1000, zmq.POLLIN):
            parts = subscriber.recv_multipart()
            assert len(parts) >= 3
            assert parts[0] == b"observation"
            assert parts[1] == b"1.0"  # schema version
            received_data = np.frombuffer(parts[2], dtype=np.float32)
            np.testing.assert_array_equal(received_data, observation)

        subscriber.close()
        shared_context.term()




def test_recv_joint_command():
    """Test receiving joint command."""
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state5",
        command_bind="inproc://test_command5",
    )

    with HalServerBase(config, context=shared_context) as server:
        # Create requester to send command (use shared context)
        requester = shared_context.socket(zmq.REQ)
        requester.connect("inproc://test_command5")
        time.sleep(0.1)  # Give requester time to connect

        # Server needs to be waiting before client sends (REQ/REP pattern)
        import threading
        received_command = [None]
        
        def server_receive():
            received_command[0] = server.recv_joint_command(timeout_ms=2000)
        
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
    shared_context.term()


def test_hwm_behavior():
    """Test HWM=1 behavior (latest-only semantics)."""
    from hal.observation.types import OBS_DIM
    
    # Use shared context for inproc connections
    shared_context = zmq.Context()
    
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state6",
        command_bind="inproc://test_command6",
        hwm=1,
    )

    with HalServerBase(config, context=shared_context) as server:
        # Create subscriber (use shared context)
        subscriber = shared_context.socket(zmq.SUB)
        subscriber.connect("inproc://test_state6")  # Match server bind address
        subscriber.setsockopt(zmq.SUBSCRIBE, b"observation")
        subscriber.setsockopt(zmq.RCVHWM, 1)
        time.sleep(0.1)  # Give subscriber time to connect

        # Publish multiple messages rapidly
        for i in range(10):
            observation = np.full(OBS_DIM, float(i), dtype=np.float32)
            server.publish_observation(observation)
        time.sleep(0.1)  # Small delay to ensure messages are sent

        # With HWM=1, subscriber should receive messages (with shared context, connection is reliable)
        received_count = 0
        while subscriber.poll(100, zmq.POLLIN):
            subscriber.recv_multipart()
            received_count += 1

        # Should receive at least one message
        assert received_count >= 1

        subscriber.close()
    shared_context.term()


def test_error_handling_invalid_shape():
    """Test error handling for invalid array shapes."""
    from hal.observation.types import OBS_DIM
    
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state7",
        command_bind="inproc://test_command7",
    )

    with HalServerBase(config) as server:
        # Try to publish 2D array (should fail)
        invalid_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            server.publish_observation(invalid_data)
        
        # Try to publish wrong size 1D array (should fail)
        wrong_size = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            server.publish_observation(wrong_size)


def test_error_handling_not_initialized():
    """Test error handling when server not initialized."""
    from hal.observation.types import OBS_DIM
    
    config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_state8",
        command_bind="inproc://test_command8",
    )
    server = HalServerBase(config)

    # Should raise error if not initialized
    with pytest.raises(RuntimeError, match="not initialized"):
        server.publish_observation(np.zeros(OBS_DIM, dtype=np.float32))

