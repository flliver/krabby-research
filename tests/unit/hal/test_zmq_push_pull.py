"""Unit tests for ZMQ PUSH/PULL pattern."""

import json
import time
import threading

import numpy as np
import pytest
import zmq


def test_zmq_push_pull_basic():
    """Test basic PUSH/PULL communication with multipart messages."""
    context = zmq.Context()
    
    # Create PULL socket and bind
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull")
    # Small delay for socket to bind
    time.sleep(0.01)
    
    # Create PUSH socket and connect
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_push_pull")
    # Small delay for socket to connect
    time.sleep(0.01)
    
    # Send multipart message
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)
    
    # Receive message
    received_parts = puller.recv_multipart()
    
    assert len(received_parts) == 2
    assert received_parts[0] == b"metadata"
    np.testing.assert_array_equal(
        np.frombuffer(received_parts[1], dtype=np.float32),
        np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_with_threading():
    """Test PUSH/PULL with receiver in a separate thread."""
    context = zmq.Context()
    
    # Create PULL socket and bind
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull_thread")
    time.sleep(0.1)
    
    # Create PUSH socket and connect
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_push_pull_thread")
    time.sleep(0.1)
    
    # Receive in thread
    received_parts = [None]
    exception_occurred = [None]
    
    def receiver_thread():
        try:
            # Poll then receive
            if puller.poll(2000, zmq.POLLIN):
                received_parts[0] = puller.recv_multipart(zmq.NOBLOCK)
        except Exception as e:
            exception_occurred[0] = e
            import traceback
            traceback.print_exc()
    
    # Start receiver thread
    receiver = threading.Thread(target=receiver_thread)
    receiver.start()
    # Small delay for thread to start polling
    time.sleep(0.01)
    
    # Send message
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)
    
    # Wait for receiver
    receiver.join(timeout=2.0)
    
    if exception_occurred[0]:
        raise exception_occurred[0]
    
    assert received_parts[0] is not None
    assert len(received_parts[0]) == 2
    assert received_parts[0][0] == b"metadata"
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_blocking_send():
    """Test that blocking send works when PULL socket is ready."""
    context = zmq.Context()
    
    # Create PULL socket and bind
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull_blocking")
    time.sleep(0.1)
    
    # Create PUSH socket and connect
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_push_pull_blocking")
    time.sleep(0.1)
    
    # Start receiver in thread first
    received_parts = [None]
    
    def receiver_thread():
        if puller.poll(2000, zmq.POLLIN):
            received_parts[0] = puller.recv_multipart(zmq.NOBLOCK)
    
    receiver = threading.Thread(target=receiver_thread)
    receiver.start()
    # Small delay to ensure receiver is actively polling
    time.sleep(0.01)
    
    # Now send (blocking)
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)  # Blocking send
    
    receiver.join(timeout=2.0)
    
    assert received_parts[0] is not None
    assert len(received_parts[0]) == 2
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_pusher_connects_first():
    """Test blocking send when pusher connects before puller is ready."""
    context = zmq.Context()
    
    # Create PULL socket and bind
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull_pusher_first")
    time.sleep(0.1)
    
    # Create PUSH socket and connect FIRST (before receiver thread starts)
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_push_pull_pusher_first")
    time.sleep(0.1)
    
    # Now start receiver in thread
    received_parts = [None]
    exception_occurred = [None]
    
    def receiver_thread():
        try:
            if puller.poll(2000, zmq.POLLIN):
                received_parts[0] = puller.recv_multipart(zmq.NOBLOCK)
        except Exception as e:
            exception_occurred[0] = e
            import traceback
            traceback.print_exc()
    
    receiver = threading.Thread(target=receiver_thread)
    receiver.start()
    # Small delay to ensure receiver is actively polling
    time.sleep(0.01)
    
    # Now send (blocking) - pusher was already connected
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)  # Blocking send
    
    receiver.join(timeout=2.0)
    
    if exception_occurred[0]:
        raise exception_occurred[0]
    
    assert received_parts[0] is not None
    assert len(received_parts[0]) == 2
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_pusher_connects_before_puller_binds():
    """Test blocking send when pusher connects before puller binds (should fail or block)."""
    context = zmq.Context()
    
    # Create PUSH socket and try to connect BEFORE puller binds
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    # Try to connect to endpoint that doesn't exist yet
    try:
        pusher.connect("inproc://test_push_pull_pusher_first")
        # Connection might succeed even if nothing is bound yet (inproc behavior)
    except Exception as e:
        pytest.fail(f"Pusher connect failed: {e}")
    time.sleep(0.1)
    
    # Now create and bind PULL socket AFTER pusher connected
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull_pusher_first")
    time.sleep(0.1)
    
    # Now start receiver in thread
    received_parts = [None]
    exception_occurred = [None]
    
    def receiver_thread():
        try:
            if puller.poll(2000, zmq.POLLIN):
                received_parts[0] = puller.recv_multipart(zmq.NOBLOCK)
        except Exception as e:
            exception_occurred[0] = e
            import traceback
            traceback.print_exc()
    
    receiver = threading.Thread(target=receiver_thread)
    receiver.start()
    # Small delay to ensure receiver is actively polling
    time.sleep(0.01)
    
    # Now send (blocking) - pusher connected before puller was bound
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)  # Blocking send
    
    receiver.join(timeout=2.0)
    
    if exception_occurred[0]:
        raise exception_occurred[0]
    
    assert received_parts[0] is not None
    assert len(received_parts[0]) == 2
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_exception_in_receiver():
    """Test what happens when receiver thread raises an exception."""
    context = zmq.Context()
    
    # Create PULL socket and bind
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_push_pull_exception")
    time.sleep(0.1)
    
    # Create PUSH socket and connect
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_push_pull_exception")
    time.sleep(0.1)
    
    # Receive in thread, but raise exception after receiving
    received_parts = [None]
    exception_occurred = [None]
    thread_finished = [False]
    
    def receiver_thread():
        try:
            if puller.poll(2000, zmq.POLLIN):
                received_parts[0] = puller.recv_multipart(zmq.NOBLOCK)
                # Simulate exception like from_bytes() might raise
                raise ValueError("Simulated deserialization error")
        except Exception as e:
            exception_occurred[0] = e
        finally:
            thread_finished[0] = True
    
    # Start receiver thread
    receiver = threading.Thread(target=receiver_thread)
    receiver.start()
    # Small delay for thread to start polling
    time.sleep(0.01)
    
    # Send message
    message_parts = [
        b"metadata",
        np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    ]
    pusher.send_multipart(message_parts)  # Blocking send
    
    # Wait for receiver
    receiver.join(timeout=2.0)
    
    # Check if thread finished (even with exception)
    assert thread_finished[0], "Thread should have finished even with exception"
    assert exception_occurred[0] is not None, "Exception should have been raised"
    assert isinstance(exception_occurred[0], ValueError)
    assert received_parts[0] is not None, "Message should have been received before exception"
    
    pusher.close()
    puller.close()
    context.term()



def test_zmq_push_pull_mimic_hal_test_structure():
    """Test that mimics the exact structure of the HAL test_get_joint_command test."""
    context = zmq.Context()
    
    # Mimic: with HalServerBase(config) as server:
    # Server creates PULL socket and binds (like server.initialize())
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_command5")
    time.sleep(0.1)
    
    # Mimic: transport_context = server.get_transport_context()
    # pusher = transport_context.socket(zmq.PUSH)
    # pusher.connect("inproc://test_command5")
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_command5")
    time.sleep(0.1)  # Give pusher time to connect
    
    # Mimic: server_thread that calls server.get_joint_command(timeout_ms=2000)
    # which does: poll(timeout_ms) then recv_multipart(NOBLOCK) then from_bytes()
    received_command = [None]
    thread_exception = [None]
    
    def server_receive():
        try:
            # Mimic get_joint_command: poll then recv_multipart
            if puller.poll(2000, zmq.POLLIN):
                command_parts = puller.recv_multipart(zmq.NOBLOCK)
                # Mimic from_bytes() - parse and validate
                if len(command_parts) != 2:
                    raise ValueError(f"Expected 2 parts, got {len(command_parts)}")
                # Simulate deserialization
                received_command[0] = np.frombuffer(command_parts[1], dtype=np.float32)
        except Exception as e:
            thread_exception[0] = e
            import traceback
            traceback.print_exc()
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    # Small delay to ensure server thread is waiting
    time.sleep(0.01)
    
    # Mimic: Send command as JointCommand (multipart message)
    command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)  # 18 DOF
    command_parts = [
        b'{"joint_positions": {"shape": [18], "dtype": "float32"}, "timestamp_ns": 0}',
        command.tobytes()
    ]
    pusher.send_multipart(command_parts)  # Blocking send
    
    server_thread.join(timeout=2.0)
    if thread_exception[0]:
        raise thread_exception[0]
    received = received_command[0]
    assert received is not None
    np.testing.assert_array_equal(received, command)
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_mimic_hal_with_shared_context():
    """Test that mimics HAL test but uses shared context like HAL does."""
    # HAL test uses: transport_context = server.get_transport_context()
    # pusher = transport_context.socket(zmq.PUSH)
    # This means pusher and puller share the same context
    
    context = zmq.Context()
    
    # Server creates PULL socket (like server.initialize())
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_command_shared")
    time.sleep(0.1)
    
    # Client creates PUSH socket using SAME context (like server.get_transport_context())
    pusher = context.socket(zmq.PUSH)  # Same context
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_command_shared")
    time.sleep(0.1)
    
    # Thread that mimics get_joint_command
    received_command = [None]
    thread_exception = [None]
    
    def server_receive():
        try:
            if puller.poll(2000, zmq.POLLIN):
                command_parts = puller.recv_multipart(zmq.NOBLOCK)
                if len(command_parts) != 2:
                    raise ValueError(f"Expected 2 parts, got {len(command_parts)}")
                received_command[0] = np.frombuffer(command_parts[1], dtype=np.float32)
        except Exception as e:
            thread_exception[0] = e
            import traceback
            traceback.print_exc()
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    time.sleep(0.05)
    
    command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)
    command_parts = [
        b'{"joint_positions": {"shape": [18], "dtype": "float32"}, "timestamp_ns": 0}',
        command.tobytes()
    ]
    pusher.send_multipart(command_parts)  # Blocking send
    
    server_thread.join(timeout=2.0)
    if thread_exception[0]:
        raise thread_exception[0]
    assert received_command[0] is not None
    np.testing.assert_array_equal(received_command[0], command)
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_mimic_hal_exact_timing():
    """Test that mimics HAL test with exact same timing."""
    context = zmq.Context()
    
    # Create PULL and bind (like server.initialize() in context manager)
    puller = context.socket(zmq.PULL)
    puller.setsockopt(zmq.RCVHWM, 5)
    puller.bind("inproc://test_command_exact")
    time.sleep(0.1)
    
    # Create PUSH using same context
    pusher = context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_command_exact")
    time.sleep(0.1)  # Give pusher time to connect
    
    # Thread with exact same structure as HAL test
    received_command = [None]
    
    def server_receive():
        # Exact mimic of: server.get_joint_command(timeout_ms=2000)
        if puller.poll(2000, zmq.POLLIN):
            command_parts = puller.recv_multipart(zmq.NOBLOCK)
            # Mimic from_bytes() which might raise ValueError
            if len(command_parts) != 2:
                raise ValueError(f"Expected 2 parts, got {len(command_parts)}")
            # Parse metadata
            metadata = json.loads(command_parts[0].decode("utf-8"))
            # Deserialize array
            joint_pos = np.frombuffer(command_parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
            joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"]))
            received_command[0] = joint_pos
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    # Small delay to ensure server thread is waiting (same as HAL test)
    time.sleep(0.01)
    
    # Send command (exact same as HAL test)
    command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)  # 18 DOF
    command_parts = [
        json.dumps({
            "joint_positions": {"shape": list(command.shape), "dtype": str(command.dtype)},
            "timestamp_ns": time.time_ns(),
        }).encode("utf-8"),
        command.tobytes(),
    ]
    pusher.send_multipart(command_parts)  # Blocking send (same as HAL test)
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    np.testing.assert_array_equal(received, command)
    
    pusher.close()
    puller.close()
    context.term()


def test_zmq_push_pull_mimic_hal_with_class_wrapper():
    """Test that mimics HAL server class structure - socket accessed through self."""
    class MockServer:
        def __init__(self, context):
            self.context = context
            self.command_socket = context.socket(zmq.PULL)
            self.command_socket.setsockopt(zmq.RCVHWM, 5)
            self.command_socket.bind("inproc://test_command_class")
            time.sleep(0.1)
        
        def get_transport_context(self):
            return self.context
        
        def get_joint_command(self, timeout_ms=2000):
            """Mimic HAL server's get_joint_command method."""
            if self.command_socket.poll(timeout_ms, zmq.POLLIN):
                command_parts = self.command_socket.recv_multipart(zmq.NOBLOCK)
                # Mimic from_bytes()
                if len(command_parts) != 2:
                    raise ValueError(f"Expected 2 parts, got {len(command_parts)}")
                import json
                metadata = json.loads(command_parts[0].decode("utf-8"))
                joint_pos = np.frombuffer(command_parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
                joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"]))
                return joint_pos
            return None
        
        def close(self):
            if self.command_socket:
                self.command_socket.close()
            self.context.term()
    
    context = zmq.Context()
    server = MockServer(context)
    
    # Create pusher using server's context (like HAL test)
    transport_context = server.get_transport_context()
    pusher = transport_context.socket(zmq.PUSH)
    pusher.setsockopt(zmq.SNDHWM, 5)
    pusher.connect("inproc://test_command_class")
    time.sleep(0.1)
    
    # Thread that calls server method (like HAL test)
    received_command = [None]
    
    def server_receive():
        received_command[0] = server.get_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    time.sleep(0.05)
    
    # Send using actual JointCommand format
    from hal.client.data_structures.hardware import JointCommand
    command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)
    joint_cmd = JointCommand(
        joint_positions=command,
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
    )
    command_parts = joint_cmd.to_bytes()
    pusher.send_multipart(command_parts)  # Blocking send
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    np.testing.assert_array_equal(received, command)
    
    pusher.close()
    server.close()


def test_zmq_push_pull_mimic_hal_with_context_manager():
    """Test that mimics HAL server context manager pattern."""
    class MockServer:
        def __init__(self, context):
            self.context = context
            self.command_socket = None
            self._initialized = False
        
        def initialize(self):
            if self._initialized:
                return
            self.command_socket = self.context.socket(zmq.PULL)
            self.command_socket.setsockopt(zmq.RCVHWM, 5)
            self.command_socket.bind("inproc://test_command_cm")
            time.sleep(0.1)
            self._initialized = True
        
        def get_transport_context(self):
            return self.context
        
        def get_joint_command(self, timeout_ms=2000):
            if not self._initialized:
                raise RuntimeError("Server not initialized")
            if self.command_socket.poll(timeout_ms, zmq.POLLIN):
                command_parts = self.command_socket.recv_multipart(zmq.NOBLOCK)
                if len(command_parts) != 2:
                    raise ValueError(f"Expected 2 parts, got {len(command_parts)}")
                import json
                metadata = json.loads(command_parts[0].decode("utf-8"))
                joint_pos = np.frombuffer(command_parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
                joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"]))
                return joint_pos
            return None
        
        def close(self):
            if self.command_socket:
                self.command_socket.close()
            self.context.term()
            self._initialized = False
        
        def __enter__(self):
            self.initialize()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
    
    context = zmq.Context()
    
    # Use context manager like HAL test
    with MockServer(context) as server:
        # Create pusher using server's context
        transport_context = server.get_transport_context()
        pusher = transport_context.socket(zmq.PUSH)
        pusher.setsockopt(zmq.SNDHWM, 5)
        pusher.connect("inproc://test_command_cm")
        time.sleep(0.1)
        
        # Thread that calls server method
        received_command = [None]
        
        def server_receive():
            received_command[0] = server.get_joint_command(timeout_ms=2000)
        
        server_thread = threading.Thread(target=server_receive)
        server_thread.start()
        time.sleep(0.05)
        
        # Send using actual JointCommand
        from hal.client.data_structures.hardware import JointCommand
        command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)
        joint_cmd = JointCommand(
            joint_positions=command,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
        )
        command_parts = joint_cmd.to_bytes()
        pusher.send_multipart(command_parts)  # Blocking send
        
        server_thread.join(timeout=2.0)
        received = received_command[0]
        assert received is not None
        np.testing.assert_array_equal(received, command)
        
        pusher.close()


def test_zmq_push_pull_mimic_hal_exact_sequence():
    """Test that mimics the exact sequence of operations in HAL test."""
    context = zmq.Context()
    
    # Step 1: Create server-like object with context manager
    class MockServer:
        def __init__(self, context):
            self.context = context
            self.command_socket = None
            self._initialized = False
        
        def initialize(self):
            self.command_socket = self.context.socket(zmq.PULL)
            self.command_socket.setsockopt(zmq.RCVHWM, 5)
            self.command_socket.bind("inproc://test_command5")
            time.sleep(0.1)
            self._initialized = True
        
        def get_transport_context(self):
            return self.context
        
        def get_joint_command(self, timeout_ms=2000):
            if self.command_socket.poll(timeout_ms, zmq.POLLIN):
                command_parts = self.command_socket.recv_multipart(zmq.NOBLOCK)
                from hal.client.data_structures.hardware import JointCommand
                command = JointCommand.from_bytes(command_parts)
                return command.joint_positions
            return None
        
        def close(self):
            if self.command_socket:
                self.command_socket.close()
            self.context.term()
        
        def __enter__(self):
            self.initialize()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
    
    # Exact sequence from HAL test
    with MockServer(context) as server:
        server.set_debug = lambda x: None  # Mock set_debug
        
        # Create pusher to send command (use server's transport context for inproc)
        transport_context = server.get_transport_context()
        pusher = transport_context.socket(zmq.PUSH)
        pusher.connect("inproc://test_command5")
        time.sleep(0.1)  # Give pusher time to connect
        
        # Server needs to be waiting before client sends (PUSH/PULL pattern)
        received_command = [None]
        
        def server_receive():
            received_command[0] = server.get_joint_command(timeout_ms=2000)
        
        server_thread = threading.Thread(target=server_receive)
        server_thread.start()
        time.sleep(0.05)  # Small delay to ensure server is waiting
        
        # Send command as JointCommand (multipart message)
        from hal.client.data_structures.hardware import JointCommand
        command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)  # 18 DOF
        joint_cmd = JointCommand(
            joint_positions=command,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
        )
        command_parts = joint_cmd.to_bytes()
        pusher.send_multipart(command_parts)
        
        server_thread.join(timeout=2.0)
        received = received_command[0]
        assert received is not None
        np.testing.assert_array_equal(received, command)
        
        pusher.close()


def test_zmq_push_pull_mimic_hal_with_observation_socket():
    """Test that mimics HAL server creating both PUB (observation) and PULL (command) sockets."""
    context = zmq.Context()
    
    class MockServer:
        def __init__(self, context):
            self.context = context
            self.observation_socket = None
            self.command_socket = None
            self._initialized = False
        
        def initialize(self):
            if self._initialized:
                return
            # Create PUB socket for observation (like HAL server does)
            self.observation_socket = self.context.socket(zmq.PUB)
            self.observation_socket.setsockopt(zmq.SNDHWM, 1)
            self.observation_socket.bind("inproc://test_observation")
            time.sleep(0.1)
            
            # Create PULL socket for commands (like HAL server does)
            self.command_socket = self.context.socket(zmq.PULL)
            self.command_socket.setsockopt(zmq.RCVHWM, 5)
            self.command_socket.bind("inproc://test_command_obs")
            time.sleep(0.1)
            self._initialized = True
        
        def get_transport_context(self):
            return self.context
        
        def get_joint_command(self, timeout_ms=2000):
            if not self._initialized:
                raise RuntimeError("Server not initialized")
            if self.command_socket.poll(timeout_ms, zmq.POLLIN):
                command_parts = self.command_socket.recv_multipart(zmq.NOBLOCK)
                from hal.client.data_structures.hardware import JointCommand
                command = JointCommand.from_bytes(command_parts)
                return command.joint_positions
            return None
        
        def close(self):
            if self.observation_socket:
                self.observation_socket.close()
            if self.command_socket:
                self.command_socket.close()
            self.context.term()
            self._initialized = False
        
        def __enter__(self):
            self.initialize()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
    
    # Use context manager like HAL test
    with MockServer(context) as server:
        # Create pusher using server's context
        transport_context = server.get_transport_context()
        pusher = transport_context.socket(zmq.PUSH)
        pusher.setsockopt(zmq.SNDHWM, 5)
        pusher.connect("inproc://test_command_obs")
        time.sleep(0.1)
        
        # Thread that calls server method
        received_command = [None]
        
        def server_receive():
            received_command[0] = server.get_joint_command(timeout_ms=2000)
        
        server_thread = threading.Thread(target=server_receive)
        server_thread.start()
        time.sleep(0.05)
        
        # Send using actual JointCommand
        from hal.client.data_structures.hardware import JointCommand
        command = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)
        joint_cmd = JointCommand(
            joint_positions=command,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
        )
        command_parts = joint_cmd.to_bytes()
        pusher.send_multipart(command_parts)  # Blocking send
        
        server_thread.join(timeout=2.0)
        received = received_command[0]
        assert received is not None
        np.testing.assert_array_equal(received, command)
        
        pusher.close()
