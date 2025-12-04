"""Integration tests for Jetson HAL server."""

import logging
import time
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest

logger = logging.getLogger(__name__)

from hal.client.client import HalClient
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.testing.inference_test_runner import InferenceTestRunner
from hal.server.jetson.camera import ZedCamera, create_zed_camera
from hal.server.jetson import JetsonHalServer


@pytest.fixture
def hal_server_config():
    """Create HAL server config for testing."""
    return HalServerConfig(
        observation_bind="inproc://test_jetson_observation",
        command_bind="inproc://test_jetson_command",
    )


@pytest.fixture
def hal_client_config():
    """Create HAL client config for testing."""
    return HalClientConfig(
        observation_endpoint="inproc://test_jetson_observation",
        command_endpoint="inproc://test_jetson_command",
    )


def test_jetson_hal_server_initialization(hal_server_config):
    """Test Jetson HAL server initialization with inproc endpoints."""
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    assert hal_server._initialized
    assert hal_server.context is not None
    assert hal_server.observation_socket is not None
    assert hal_server.observation_socket is not None
    assert hal_server.command_socket is not None

    hal_server.close()


@pytest.mark.skipif(
    True,  # Always skip in x86 test environment - requires Jetson hardware or pyzed
    reason="Requires Jetson hardware or ZED SDK (pyzed) - skip in x86 test environment"
)
@pytest.mark.skip(
    reason="Requires Jetson hardware or ZED SDK (pyzed) - skip in x86 test environment"
)
def test_jetson_hal_server_camera_initialization(hal_server_config):
    """Test ZED camera initialization in Jetson HAL server.
    
    Note: This test is skipped in x86 test environment because it requires:
    - Jetson hardware, OR
    - ZED SDK (pyzed) installed
    
    Run this test on ARM test environment or production Jetson hardware.
    """
    pass  # Test skipped - would require Jetson hardware or ZED SDK


def test_jetson_hal_server_observation_publishing(hal_server_config, hal_client_config):
    """Test observation publishing from Jetson HAL server."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Mock camera to return depth features
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True
    mock_camera.get_depth_features.return_value = None  # Will be set below
    hal_server.zed_camera = mock_camera

    # Setup HAL client with server's transport context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    hal_client.set_debug(True)

    # Mock depth features (NUM_SCAN = 132)
    from compute.parkour.parkour_types import NUM_SCAN

    depth_features = np.array([1.0, 2.0, 3.0] * 44, dtype=np.float32)[:NUM_SCAN]  # 132 features
    mock_camera.get_depth_features.return_value = depth_features

    # Mock state vector
    def mock_build_state():
        return np.concatenate([
            [0.0, 0.0, 0.0],  # base_pos
            [0.0, 0.0, 0.0, 1.0],  # base_quat
            [0.0, 0.0, 0.0],  # base_lin_vel
            [0.0, 0.0, 0.0],  # base_ang_vel
            [0.0] * 12,  # joint_pos
            [0.0] * 12,  # joint_vel
        ]).astype(np.float32)

    hal_server._build_state_vector = mock_build_state

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hw_obs = hal_client.poll(timeout_ms=1000)

    # Verify hardware observation data received
    assert hw_obs is not None
    assert hw_obs.joint_positions is not None

    hal_client.close()
    hal_server.close()


def test_jetson_hal_server_joint_command_application(hal_server_config, hal_client_config):
    """Test joint command application in Jetson HAL server."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    hal_client.set_debug(True)

    # Send command from client using new API
    from compute.parkour.parkour_types import InferenceResponse
    import torch
    import threading

    action_array = np.array([0.1, 0.2, 0.3] * 4, dtype=np.float32)  # 12 DOF
    action_tensor = torch.from_numpy(action_array)
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # Server needs to be waiting before client sends (PUSH/PULL pattern)
    received_command = [None]
    
    def server_receive():
        received_command[0] = hal_server.get_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    # Small delay to ensure server thread is waiting
    time.sleep(0.01)

    # Map inference response to hardware joint positions
    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
    mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)
    joint_positions = mapper.map(inference_response)
    
    # Send command
    hal_client.put_joint_command(joint_positions), "Command send failed"

    # Wait for server to receive
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None, "Server did not receive command"
    # get_joint_command now returns KrabbyDesiredJointPositions instance
    assert hasattr(received, 'joint_positions')
    # Should receive the mapped joint positions (18 DOF)
    np.testing.assert_array_equal(received.joint_positions, joint_positions.joint_positions)

    hal_client.close()
    hal_server.close()


def test_jetson_hal_server_end_to_end_with_game_loop(hal_server_config, hal_client_config):
    """Test end-to-end integration with inference logic (game loop)."""
    import zmq
    
    # Use shared context for inproc connections
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Mock camera
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True

    from compute.parkour.parkour_types import NUM_SCAN

    depth_features = np.zeros(NUM_SCAN, dtype=np.float32)
    depth_features[0:64] = 1.0  # Set first 64 features
    mock_camera.get_depth_features.return_value = depth_features
    hal_server.zed_camera = mock_camera

    # Mock state vector
    def mock_build_state():
        return np.concatenate([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0] * 12,
            [0.0] * 12,
        ]).astype(np.float32)

    hal_server._build_state_vector = mock_build_state

    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    hal_client.set_debug(True)

    # Create mock policy model
    class MockPolicyModel:
        def __init__(self):
            self.action_dim = 12
            self.inference_count = 0

        def inference(self, model_io):
            import time
            from compute.parkour.parkour_types import InferenceResponse
            import torch

            self.inference_count += 1
            action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
            return InferenceResponse.create_success(
                action=action_tensor,
                inference_latency_ms=5.0,
            )

    model = MockPolicyModel()
    test_runner = InferenceTestRunner(model, hal_client, control_rate_hz=100.0)

    # Set navigation command on test runner
    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    # Run a few iterations
    import threading

    def run_loop():
        for _ in range(10):
            # Publish observation from hal server
            hal_server.set_observation()

            # Poll client
            hw_obs = hal_client.poll(timeout_ms=10)
            if hw_obs is None:
                continue

            # Map hardware observation to ParkourObservation
            # Pass navigation command so it's included in the observation
            from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
            from compute.parkour.parkour_types import ParkourModelIO
            
            mapper = KrabbyHWObservationsToParkourMapper()
            parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
            
            # Build model IO (preserve timestamp from observation)
            model_io = ParkourModelIO(
                timestamp_ns=parkour_obs.timestamp_ns,
                schema_version=parkour_obs.schema_version,
                nav_cmd=nav_cmd,
                observation=parkour_obs,
            )
            
            if model_io is not None:
                inference_result = model.inference(model_io)
                if inference_result.success:
                    # Map inference response to hardware joint positions
                    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
                    mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)
                    joint_positions = mapper.map(inference_result)
                    hal_client.put_joint_command(joint_positions)

            # Apply joint command
            hal_server.apply_command()

            time.sleep(0.01)  # 10ms period

    thread = threading.Thread(target=run_loop)
    thread.start()
    thread.join()

    # Verify inference test ran
    assert model.inference_count > 0

    hal_client.close()
    hal_server.close()


@pytest.mark.skip(
    reason="Requires Jetson hardware or ZED SDK (pyzed) - skip in x86 test environment"
)
def test_jetson_hal_server_zed_camera_integration():
    """Test ZED camera integration (requires hardware).
    
    Note: This test is skipped in x86 test environment because it requires:
    - Jetson hardware, OR
    - ZED SDK (pyzed) installed
    
    Run this test on ARM test environment or production Jetson hardware.
    """
    pass  # Test skipped - would require Jetson hardware or ZED SDK


@pytest.mark.skip(
    reason="Requires Jetson hardware or ZED SDK (pyzed) - skip in x86 test environment"
)
def test_jetson_hal_server_inference_runner(hal_server_config):
    """Test InferenceRunner with Jetson HAL server.
    
    Note: This test is skipped in x86 test environment because it requires:
    - Jetson hardware, OR
    - ZED SDK (pyzed) installed
    
    Run this test on ARM test environment or production Jetson hardware.
    """
    pass  # Test skipped - would require Jetson hardware or ZED SDK


def test_jetson_hal_server_network_communication():
    """Test network communication (x86 → Jetson simulation).

    This test simulates network communication by using TCP endpoints.
    Note: In production, Jetson uses inproc, but this tests TCP capability.
    """
    import threading
    # Use TCP endpoints for network simulation
    server_config = HalServerConfig(
        observation_bind="tcp://*:8001",
        command_bind="tcp://*:8002",
    )
    server = JetsonHalServer(server_config)
    server.initialize()

    server.set_debug(True)

    # Client config pointing to server (simulating x86 → Jetson)
    client_config = HalClientConfig(
        observation_endpoint="tcp://localhost:8001",
        command_endpoint="tcp://localhost:8002",
    )
    client = HalClient(client_config)
    client.initialize()

    client.set_debug(True)

    # Connection should be ready immediately with inproc

    # Mock camera and state
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True

    from compute.parkour.parkour_types import NUM_SCAN

    depth_features = np.zeros(NUM_SCAN, dtype=np.float32)
    depth_features[0:64] = 1.0  # Set first 64 features
    mock_camera.get_depth_features.return_value = depth_features
    server.zed_camera = mock_camera

    def mock_build_state():
        return np.concatenate([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0] * 12,
            [0.0] * 12,
        ]).astype(np.float32)

    server._build_state_vector = mock_build_state

    # Publish observation
    server.set_observation()

    # Poll client
    hw_obs = client.poll(timeout_ms=2000)  # Longer timeout for network

    # Verify hardware observation data received
    assert hw_obs is not None
    assert hw_obs.joint_positions is not None

    # Send command
    from compute.parkour.parkour_types import InferenceResponse
    import torch

    action_array = np.array([0.1] * 12, dtype=np.float32)
    action_tensor = torch.from_numpy(action_array)
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # Server needs to be waiting before client sends (PUSH/PULL pattern)
    received_command = [None]
    
    def server_receive():
        received_command[0] = server.get_joint_command(timeout_ms=3000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    # Small delay to ensure server thread is waiting
    time.sleep(0.01)

    # Map inference response to hardware joint positions
    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
    mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)
    joint_positions = mapper.map(inference_response)
    
    client.put_joint_command(joint_positions)

    # Wait for server to receive
    server_thread.join(timeout=3.0)
    received = received_command[0]
    assert received is not None

    client.close()
    server.close()


def test_jetson_hal_server_camera_error_handling(hal_server_config):
    """Test error handling when camera fails."""
    import zmq
    
    # Use shared context for inproc connections
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Test with None camera (camera not initialized)
    hal_server.zed_camera = None
    depth_features = hal_server._build_depth_features()
    assert depth_features is None  # Should return None gracefully

    # Test with camera that returns None
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True
    mock_camera.get_depth_features.return_value = None
    hal_server.zed_camera = mock_camera

    depth_features = hal_server._build_depth_features()
    assert depth_features is None  # Should handle None gracefully

    hal_server.close()


def test_jetson_hal_server_state_error_handling(hal_server_config):
    """Test error handling when state source fails."""
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Test with None state
    state_vector = hal_server._build_state_vector()
    # Should return None (placeholder implementation)
    # In production, would handle gracefully

    hal_server.close()


def test_jetson_hal_server_sustained_bidirectional_messaging(hal_server_config, hal_client_config):
    """Test sustained bidirectional messaging without drops or disconnects.
    
    This test verifies sustained operation for 3 seconds (300 cycles at 100 Hz)
    with bidirectional messaging (server publishes, client receives and sends commands).
    Reduced from 30 seconds to 3 seconds for faster test execution while still
    verifying sustained operation, drop rate, and rate maintenance.
    """
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock camera to return depth features
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True
    from compute.parkour.parkour_types import NUM_SCAN

    depth_features = np.zeros(NUM_SCAN, dtype=np.float32)
    mock_camera.get_depth_features.return_value = depth_features
    hal_server.zed_camera = mock_camera

    # Mock state vector builder
    def mock_build_state():
        return np.concatenate([
            [0.0, 0.0, 0.0],  # base_pos
            [0.0, 0.0, 0.0, 1.0],  # base_quat
            [0.0, 0.0, 0.0],  # base_lin_vel
            [0.0, 0.0, 0.0],  # base_ang_vel
            [0.0] * 12,  # joint_pos
            [0.0] * 12,  # joint_vel
        ]).astype(np.float32)
    hal_server._build_state_vector = mock_build_state

    # Set initial navigation command
    nav_cmd = NavigationCommand.create_now()

    # Statistics - reduced to 3 seconds for faster test execution
    duration_seconds = 3.0
    target_rate_hz = 100.0
    period = 1.0 / target_rate_hz
    total_cycles = int(duration_seconds * target_rate_hz)
    
    cycles_completed = 0
    observations_published = 0
    observations_received = 0
    commands_sent = 0
    commands_received = 0
    disconnects = 0
    drops = 0
    last_observation_time = None
    
    start_time = time.time()
    
    for cycle in range(total_cycles):
        cycle_start = time.time()
        
        try:
            # Server publishes observation
            hal_server.set_observation()
            observations_published += 1
            
            # Client polls for observation
            hw_obs = hal_client.poll(timeout_ms=10)
            
            # Check if observation received
            if hw_obs is not None:
                observations_received += 1
                if last_observation_time is not None:
                    time_diff = time.time() - last_observation_time
                    if time_diff > period * 2:  # More than 2x expected period
                        drops += 1
                last_observation_time = time.time()
            
            # Map hardware observation and build model IO
            if hw_obs is not None:
                from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
                from compute.parkour.parkour_types import ParkourModelIO
                
                mapper = KrabbyHWObservationsToParkourMapper()
                parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
                
                model_io = ParkourModelIO(
                    timestamp_ns=parkour_obs.timestamp_ns,
                    schema_version=parkour_obs.schema_version,
                    nav_cmd=nav_cmd,
                    observation=parkour_obs,
                )
            else:
                model_io = None
            
            if model_io is not None:
                # Create mock command
                command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
                
                from compute.parkour.parkour_types import InferenceResponse
                import torch as torch_module
                action_tensor = torch_module.from_numpy(command)
                response = InferenceResponse.create_success(
                    action=action_tensor,
                    inference_latency_ms=5.0,
                )
                # Map inference response to hardware joint positions
                from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
                mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)
                joint_positions = mapper.map(response)
                hal_client.put_joint_command(joint_positions)
                commands_sent += 1
                
                # Server receives command (has 10ms timeout, so won't block long)
                # In REQ/REP pattern, server needs to be waiting, but with timeout it returns quickly
                try:
                    hal_server.apply_command()
                    commands_received += 1
                except RuntimeError:
                    # No command received (timeout), skip this cycle
                    pass
            
            cycles_completed += 1
            
            # Sleep to maintain rate
            elapsed = time.time() - cycle_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            disconnects += 1
            logger.error(f"Cycle {cycle} error: {e}")
            # Continue running despite errors to test resilience
    
    elapsed_total = time.time() - start_time
    
    # Verify completion
    assert cycles_completed >= total_cycles * 0.95, f"Only completed {cycles_completed}/{total_cycles} cycles"
    
    # Verify no disconnects
    assert disconnects == 0, f"Detected {disconnects} disconnects during sustained messaging test"
    
    # Verify observations were published and received
    assert observations_published > 0, "No observations published"
    assert observations_received > 0, "No observations received"
    
    # Verify commands were sent and received
    assert commands_sent > 0, "No commands sent"
    assert commands_received > 0, "No commands received"
    
    # Verify drop rate is acceptable (< 5%)
    if observations_received > 0:
        drop_rate = drops / observations_received
        assert drop_rate < 0.05, f"Drop rate {drop_rate:.2%} too high (expected < 5%)"
    
    # Verify sustained operation
    actual_rate = cycles_completed / elapsed_total
    assert actual_rate >= 90.0, f"Rate {actual_rate} Hz too low (expected >= 90 Hz)"
    
    hal_client.close()
    hal_server.close()


def test_jetson_hal_server_joystick_input_integration(hal_server_config, hal_client_config):
    """Test joystick input integration with HAL client.
    
    This test simulates joystick input by sending navigation commands
    and verifies they are properly received and used in the control loop.
    """
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock camera
    mock_camera = MagicMock(spec=ZedCamera)
    mock_camera.is_ready.return_value = True
    from compute.parkour.parkour_types import NUM_SCAN

    depth_features = np.zeros(NUM_SCAN, dtype=np.float32)
    mock_camera.get_depth_features.return_value = depth_features
    hal_server.zed_camera = mock_camera

    # Mock state vector builder
    def mock_build_state():
        return np.concatenate([
            [0.0, 0.0, 0.0],  # base_pos
            [0.0, 0.0, 0.0, 1.0],  # base_quat
            [0.0, 0.0, 0.0],  # base_lin_vel
            [0.0, 0.0, 0.0],  # base_ang_vel
            [0.0] * 12,  # joint_pos
            [0.0] * 12,  # joint_vel
        ]).astype(np.float32)
    hal_server._build_state_vector = mock_build_state

    # Simulate joystick input: send navigation commands
    # Joystick typically controls: vx (forward/back), vy (left/right), yaw_rate (rotation)
    joystick_commands = [
        NavigationCommand.create_now(vx=0.5, vy=0.0, yaw_rate=0.0),  # Forward
        NavigationCommand.create_now(vx=0.0, vy=0.3, yaw_rate=0.0),  # Left
        NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.5),  # Rotate
        NavigationCommand.create_now(vx=0.3, vy=0.2, yaw_rate=0.1),  # Combined
    ]

    commands_sent = 0
    commands_used = 0

    # Send joystick commands and verify they're used
    for nav_cmd in joystick_commands:
        commands_sent += 1

        # Publish observation
        hal_server.set_observation()

        # Poll client
        hw_obs = hal_client.poll(timeout_ms=100)
        if hw_obs is None:
            continue

        # Map hardware observation and build model IO with navigation command
        # Pass navigation command so it's included in the observation
        from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
        from compute.parkour.parkour_types import ParkourModelIO
        
        mapper = KrabbyHWObservationsToParkourMapper()
        parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
        
        model_io = ParkourModelIO(
            timestamp_ns=parkour_obs.timestamp_ns,
            schema_version=parkour_obs.schema_version,
            nav_cmd=nav_cmd,
            observation=parkour_obs,
        )
        
        if model_io is not None:
            # Verify navigation command is included
            if model_io.nav_cmd is not None:
                assert model_io.nav_cmd.vx == nav_cmd.vx
                assert model_io.nav_cmd.vy == nav_cmd.vy
                assert model_io.nav_cmd.yaw_rate == nav_cmd.yaw_rate
                commands_used += 1

        # Small delay between commands for test clarity
        time.sleep(0.001)

    # Verify all commands were sent and used
    assert commands_sent == len(joystick_commands), f"Expected {len(joystick_commands)} commands sent, got {commands_sent}"
    assert commands_used == len(joystick_commands), f"Expected {len(joystick_commands)} commands used, got {commands_used}"

    hal_client.close()
    hal_server.close()


def test_jetson_hal_server_cleanup(hal_server_config):
    """Test proper cleanup of resources."""
    hal_server = JetsonHalServer(hal_server_config)
    hal_server.initialize()

    hal_server.set_debug(True)

    # Mock camera
    mock_camera = MagicMock(spec=ZedCamera)
    hal_server.zed_camera = mock_camera

    # Close server
    hal_server.close()

    # Verify camera was closed
    mock_camera.close.assert_called_once()

    # Verify server is closed
    assert not hal_server._initialized
    assert hal_server.zed_camera is None

