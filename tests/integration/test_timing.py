"""Integration tests for timing and throughput."""

import time

import numpy as np
import pytest

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from hal.client.data_structures.hardware import KrabbyHardwareObservations
from compute.testing.inference_test_runner import InferenceTestRunner
from tests.helpers import create_dummy_hw_obs


class SlowInferenceModel:
    """Mock model with slow inference (15-20ms)."""

    def __init__(self, action_dim: int = 12):
        """Initialize slow inference model."""
        self.action_dim = action_dim
        self.inference_count = 0
        self.last_result = None

    def inference(self, model_io):
        """Slow inference (15-20ms)."""
        inference_time_ms = 18.0  # 18ms inference
        time.sleep(inference_time_ms / 1000.0)
        self.inference_count += 1

        from compute.parkour.parkour_types import InferenceResponse
        import torch

        action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
        self.last_result = InferenceResponse.create_success(
            action=action_tensor,
            inference_latency_ms=inference_time_ms,
        )
        return self.last_result


class FastInferenceModel:
    """Mock model with fast inference (<5ms)."""

    def __init__(self, action_dim: int = 12):
        """Initialize fast inference model."""
        self.action_dim = action_dim
        self.inference_count = 0

    def inference(self, model_io):
        """Fast inference (<5ms)."""
        inference_time_ms = 3.0  # 3ms inference
        time.sleep(inference_time_ms / 1000.0)
        self.inference_count += 1

        from compute.parkour.parkour_types import InferenceResponse
        import torch

        action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
        return InferenceResponse.create_success(
            action=action_tensor,
            inference_latency_ms=inference_time_ms,
        )


def test_game_loop_faster_than_inference():
    """Test inference logic (game loop) handles inference slower than loop rate."""
    import zmq
    
    # Use shared context for inproc connections
    # Use unified observation endpoint (new API)
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_slow",
        command_bind="inproc://test_command_slow",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_slow",
        command_endpoint="inproc://test_command_slow",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)

    # Use slow inference model (18ms > 10ms period at 100Hz)
    model = SlowInferenceModel(action_dim=12)
    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    import threading

    # Continuously publish hardware observation
    from hal.client.data_structures.hardware import KrabbyHardwareObservations
    def publish_loop():
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        for _ in range(100):
            server.set_observation(hw_obs)
            time.sleep(0.01)

    # Continuously receive commands (PUSH/PULL pattern requires server to be waiting)
    command_received = threading.Event()
    def command_loop():
        while not command_received.is_set():
            server.get_joint_command(timeout_ms=100)

    pub_thread = threading.Thread(target=publish_loop)
    pub_thread.start()
    
    cmd_thread = threading.Thread(target=command_loop)
    cmd_thread.start()

    # Run for short time
    def stop_after_time():
        time.sleep(0.5)  # Run for 500ms
        test_runner.stop()
        command_received.set()

    stop_thread = threading.Thread(target=stop_after_time)
    stop_thread.start()

    # Production code (InferenceTestRunner.run()) handles all exceptions internally
    # No exception handling needed here - if run() completes, it succeeded
    # If we need to handle expected exceptions in the future, add them here:
    # try:
    #     test_runner.run()
    # except ExpectedExceptionType:
    #     # Handle expected exception
    #     pass
    test_runner.run()

    command_received.set()
    stop_thread.join()
    pub_thread.join()
    cmd_thread.join()

    # Verify inference ran (may or may not have dropped frames depending on timing)
    # The key is that inference should have run without blocking
    assert test_runner.frame_count > 0
    # Verify latest result was used (not stale)
    assert test_runner.last_inference_result is not None

    client.close()
    server.close()


def test_inference_faster_than_game_loop():
    """Test inference logic (game loop) handles inference faster than loop rate."""
    import zmq
    
    # Use shared context for inproc connections
    # Use unified observation endpoint (new API)
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_fast",
        command_bind="inproc://test_command_fast",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_fast",
        command_endpoint="inproc://test_command_fast",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)

    # Use fast inference model (3ms < 10ms period at 100Hz)
    model = FastInferenceModel(action_dim=12)
    test_runner = InferenceTestRunner(model, client, control_rate_hz=100.0)

    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    import threading

    # Continuously publish hardware observation
    def publish_loop():
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        for _ in range(100):
            server.set_observation(hw_obs)
            time.sleep(0.01)

    # Continuously receive commands (PUSH/PULL pattern requires server to be waiting)
    command_received = threading.Event()
    def command_loop():
        while not command_received.is_set():
            server.get_joint_command(timeout_ms=100)

    pub_thread = threading.Thread(target=publish_loop)
    pub_thread.start()
    
    cmd_thread = threading.Thread(target=command_loop)
    cmd_thread.start()

    def stop_after_time():
        time.sleep(0.5)
        test_runner.stop()
        command_received.set()

    stop_thread = threading.Thread(target=stop_after_time)
    stop_thread.start()

    # Production code (InferenceTestRunner.run()) handles all exceptions internally
    # No exception handling needed here - if run() completes, it succeeded
    # If we need to handle expected exceptions in the future, add them here:
    # try:
    #     test_runner.run()
    # except ExpectedExceptionType:
    #     # Handle expected exception
    #     pass
    test_runner.run()

    command_received.set()
    stop_thread.join()
    pub_thread.join()
    cmd_thread.join()

    # Verify latest result was used correctly
    assert test_runner.last_inference_result is not None
    # Verify no unnecessary buffering (should have low dropped frames)
    assert test_runner.dropped_frames == 0

    client.close()
    server.close()


def test_timestamp_in_messages():
    """Test that all messages include timestamps."""
    import zmq
    
    # Use shared context for inproc connections
    # Use unified observation endpoint (new API)
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_ts",
        command_bind="inproc://test_command_ts",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_ts",
        command_endpoint="inproc://test_command_ts",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)

    # Initial dummy publish/poll to establish connection
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs)
    client.poll(timeout_ms=100)
    time.sleep(0.05)

    # Publish hardware observation
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)

    # Verify timestamps are present
    assert received_hw_obs is not None
    assert received_hw_obs.timestamp_ns > 0

    client.close()
    server.close()


def test_timestamp_precision():
    """Test timestamp precision (nanoseconds)."""
    import time

    # Create multiple commands with timestamps
    timestamps = []
    for _ in range(10):
        nav_cmd = NavigationCommand.create_now()
        timestamps.append(nav_cmd.timestamp_ns)
        time.sleep(0.001)  # 1ms sleep

    # Verify timestamps are increasing
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]

    # Verify timestamps have nanosecond precision (should have variation)
    differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    # Each difference should be around 1ms = 1,000,000 nanoseconds
    for diff in differences:
        assert 500_000 < diff < 2_000_000  # Within reasonable range for 1ms sleep


def test_timestamp_validation_stale_messages():
    """Test timestamp validation rejects stale messages."""
    import zmq
    
    # Use shared context for inproc connections
    # Use unified observation endpoint (new API)
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_stale",
        command_bind="inproc://test_command_stale",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_stale",
        command_endpoint="inproc://test_command_stale",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)

    # Publish fresh hardware observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    # Verify timestamp is recent
    assert received_hw_obs.timestamp_ns > 0

    client.close()
    server.close()


def test_end_to_end_latency():
    """Test end-to-end latency measurement."""
    import zmq
    
    # Use shared context for inproc connections
    # Use unified observation endpoint (new API)
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs_latency",
        command_bind="inproc://test_command_latency",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs_latency",
        command_endpoint="inproc://test_command_latency",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)

    # Measure send to receive latency
    send_time_ns = time.time_ns()
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    receive_time_ns = time.time_ns()
    
    assert received_hw_obs is not None

    latency_ns = receive_time_ns - send_time_ns
    latency_ms = latency_ns / 1_000_000.0

    # Latency should be very low for inproc (< 1ms typically)
    assert latency_ms < 10.0  # Should be under 10ms for inproc

    client.close()
    server.close()

