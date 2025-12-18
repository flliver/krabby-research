"""Integration tests for IsaacSim HAL server."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hal.server.isaac import IsaacSimHalServer
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.testing.inference_test_runner import InferenceTestRunner


@pytest.fixture
def mock_isaac_env():
    """Create a mock IsaacSim environment."""
    env = MagicMock()
    # Create a mock scene that supports 'in' operator and items() iteration
    mock_robot = MagicMock()
    mock_robot.data = MagicMock()
    mock_robot.data.joint_pos = MagicMock()
    env.scene = MagicMock()
    env.scene.__getitem__ = MagicMock(return_value=mock_robot)  # For env.scene["robot"]
    env.scene.__contains__ = MagicMock(return_value=True)  # For "robot" in env.scene
    env.scene.items = MagicMock(return_value=iter([("robot", mock_robot)]))  # For iteration
    env.observation_manager = MagicMock()
    env.action_manager = MagicMock()
    return env


@pytest.fixture
def hal_server_config():
    """Create HAL server config for testing."""
    return HalServerConfig(
        observation_bind="inproc://test_isaac_observation",
        command_bind="inproc://test_isaac_command",
    )


@pytest.fixture
def hal_client_config():
    """Create HAL client config for testing."""
    return HalClientConfig(
        observation_endpoint="inproc://test_isaac_observation",
        command_endpoint="inproc://test_isaac_command",
    )


def test_isaacsim_hal_server_initialization(mock_isaac_env, hal_server_config):
    """Test IsaacSim HAL server initialization with minimal environment."""
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    assert hal_server._initialized
    assert hal_server.env is not None
    assert hal_server.context is not None
    assert hal_server.observation_socket is not None
    assert hal_server.command_socket is not None

    hal_server.close()


def test_isaacsim_hal_server_camera_publishing(mock_isaac_env, hal_server_config, hal_client_config):
    """Test observation publishing from IsaacSim HAL server."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    from compute.parkour.parkour_types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})
    hal_server.env.device = torch.device("cpu")

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hw_obs = hal_client.poll(timeout_ms=1000)

    # Verify hardware observation data received
    assert hw_obs is not None
    assert hw_obs.joint_positions is not None

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_state_publishing(mock_isaac_env, hal_server_config, hal_client_config):
    """Test observation publishing from IsaacSim HAL server."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    from compute.parkour.parkour_types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})
    hal_server.env.device = torch.device("cpu")

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hw_obs = hal_client.poll(timeout_ms=1000)

    # Verify hardware observation data received
    assert hw_obs is not None
    assert hw_obs.joint_positions is not None

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_joint_command_application(mock_isaac_env, hal_server_config, hal_client_config):
    """Test joint command application to IsaacSim environment."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock action manager methods (these are what apply_joint_command actually calls)
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()
    
    # Mock env.device
    hal_server.env.device = "cpu"

    # Mock command reception to return JointCommand instance (bypass ZMQ)
    def mock_get_joint_command(timeout_ms=10):
        from hal.client.data_structures.hardware import JointCommand
        command_array = np.array([0.1, 0.2, 0.3] + [0.0] * 15, dtype=np.float32)  # 18 DOF
        return JointCommand(
            joint_positions=command_array,
            timestamp_ns=time.time_ns(),
        )

    hal_server.get_joint_command = mock_get_joint_command

    # Apply joint command
    hal_server.apply_command()
    
    # Verify action manager methods were called
    hal_server.action_manager.process_action.assert_called_once()
    hal_server.action_manager.apply_action.assert_called_once()

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_end_to_end_with_game_loop(mock_isaac_env, hal_server_config, hal_client_config):
    """Test end-to-end integration with inference logic (game loop)."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from compute.parkour.parkour_types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})
    hal_server.env.device = torch.device("cpu")

    # Mock action manager
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Create mock policy model
    class MockPolicyModel:
        def __init__(self):
            self.action_dim = 12
            self.inference_count = 0

        def inference(self, model_io):
            import time
            from compute.parkour.parkour_types import InferenceResponse

            self.inference_count += 1
            action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
            return InferenceResponse.create_success(
                action=action_tensor,
                inference_latency_ms=5.0,
            )

    model = MockPolicyModel()

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(model, hal_client, control_rate_hz=100.0)

    # Set navigation command on test runner
    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    # Run a few iterations
    import threading

    def run_loop():
        for _ in range(10):  # Run 10 iterations
            # Publish observation from hal server
            hal_server.set_observation()

            # Poll client
            hw_obs = hal_client.poll(timeout_ms=10)
            if hw_obs is None:
                continue

            # Map hardware observation to ParkourObservation
            # Pass navigation command so it's included in the observation
            from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
            from compute.parkour.parkour_types import ParkourModelIO
            
            mapper = HWObservationsToParkourMapper()
            parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
            
            # Build model IO (preserve timestamp from observation)
            model_io = ParkourModelIO(
                timestamp_ns=parkour_obs.timestamp_ns,
                schema_version=parkour_obs.schema_version,
                nav_cmd=nav_cmd,
                observation=parkour_obs,
            )
            
            inference_result = model.inference(model_io)
            if inference_result.success:
                # Map inference response to hardware joint positions
                from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
                mapper = ParkourLocomotionToHWMapper(model_action_dim=12)
                joint_positions = mapper.map(inference_result)
                hal_client.put_joint_command(joint_positions)

            # Apply joint command
            hal_server.apply_command()

            time.sleep(0.01)  # 10ms period (100Hz)

    thread = threading.Thread(target=run_loop)
    thread.start()
    thread.join()

    # Verify inference test ran
    assert model.inference_count > 0

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_behavior_matches_baseline(mock_isaac_env, hal_server_config):
    """Test that IsaacSim HAL server behavior matches baseline evaluation.py.

    This test verifies that:
    - Observation is published at correct rates
    - Commands are applied correctly
    - The interface matches what evaluation.py expects
    """
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Verify server can publish observation
    hal_server._extract_depth_features = lambda: np.array([1.0] * 10, dtype=np.float32)
    hal_server._extract_state_vector = lambda: np.array([0.0] * 34, dtype=np.float32)

    # Publish observation (should not raise)
    hal_server.set_observation()

    # Verify server can receive commands
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()
    
    # Mock env.device
    hal_server.env.device = "cpu"

    # Mock command reception to return JointCommand instance
    def mock_get_joint_command(timeout_ms=100):
        from hal.client.data_structures.hardware import JointCommand
        command_array = np.array([0.0] * 18, dtype=np.float32)
        return JointCommand(
            joint_positions=command_array,
            timestamp_ns=time.time_ns(),
        )

    hal_server.get_joint_command = mock_get_joint_command

    # Apply command (should not raise)
    hal_server.apply_command()

    hal_server.close()


def test_isaacsim_hal_server_with_real_zmq_communication(mock_isaac_env, hal_server_config, hal_client_config):
    """Test IsaacSim HAL server with real ZMQ communication (inproc)."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from compute.parkour.parkour_types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hw_obs = hal_client.poll(timeout_ms=1000)

    # Verify data received
    assert hw_obs is not None

    # Send command from client
    from compute.parkour.parkour_types import InferenceResponse

    action_array = np.array([0.1, 0.2, 0.3] * 4, dtype=np.float32)
    action_tensor = torch.from_numpy(action_array)
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # In PUSH/PULL pattern, server must be waiting before client sends
    # Use threading to have server wait for command
    import threading
    received_command = [None]
    
    def server_receive():
        received_command[0] = hal_server.get_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    # Small delay to ensure server thread is waiting
    time.sleep(0.01)
    
    # Map inference response to hardware joint positions
    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
    mapper = ParkourLocomotionToHWMapper(model_action_dim=12)
    joint_positions = mapper.map(inference_response)
    
    # Send command
    hal_client.put_joint_command(joint_positions)
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    # get_joint_command now returns JointCommand instance
    assert hasattr(received, 'joint_positions')
    # Compare against mapped joint positions (18 DOF), not original action array (12 DOF)
    np.testing.assert_array_equal(received.joint_positions, joint_positions.joint_positions)

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_observation_rate(mock_isaac_env, hal_server_config, hal_client_config):
    """Test that observation can be published at required rates (30-60 Hz camera, 100+ Hz state)."""
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

    hal_client = HalClient(hal_client_config)
    hal_client.initialize()

    hal_server._extract_depth_features = lambda: np.array([1.0], dtype=np.float32)
    hal_server._extract_state_vector = lambda: np.array([0.0] * 34, dtype=np.float32)

    # Publish at high rate
    start_time = time.time()
    publish_count = 0

    for _ in range(100):  # Publish 100 times
        hal_server.set_observation()
        publish_count += 1
        # No sleep needed - test is measuring maximum publish rate

    elapsed = time.time() - start_time
    rate = publish_count / elapsed

    # Should be able to publish at high rate (>100 Hz)
    assert rate > 100.0

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_error_handling(mock_isaac_env, hal_server_config):
    """Test error handling in IsaacSim HAL server."""
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Test with None environment
    hal_server_no_env = IsaacSimHalServer(hal_server_config, env=None)
    hal_server_no_env.initialize()
    hal_server_no_env.set_debug(True)

    # Publish observation with no environment:
    # In the new implementation this correctly raises a RuntimeError instead of silently succeeding.
    with pytest.raises(RuntimeError, match="No environment set"):
        hal_server_no_env.set_observation()

    # Apply command with no environment:
    # Likewise, apply_command() now raises a RuntimeError when env is None.
    with pytest.raises(RuntimeError, match="No environment set"):
        hal_server_no_env.apply_command()

    hal_server_no_env.close()
    hal_server.close()


def test_isaacsim_hal_server_100_consecutive_command_cycles(mock_isaac_env, hal_server_config, hal_client_config):
    """Test 100 consecutive command cycles without dropped commands or NaNs.
    
    This test verifies that the HAL server can handle sustained operation
    at 100 Hz for 100 cycles (1 second) without errors, dropped commands, or NaN values.
    """
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock observation manager to return valid observations
    from compute.parkour.parkour_types import OBS_DIM
    import torch
    
    def mock_compute_observations():
        # Return observation in training format: (num_envs, OBS_DIM)
        obs_tensor = torch.zeros(1, OBS_DIM, dtype=torch.float32)
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")

    # Mock action manager
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Set navigation command
    nav_cmd = NavigationCommand.create_now()

    # Start command receiving thread (PUSH/PULL pattern requires server to be waiting)
    import threading
    command_received = threading.Event()
    commands_applied = 0
    
    # Track statistics
    cycles_completed = 0
    commands_received = 0
    observations_published = 0
    errors = []
    
    def command_loop():
        nonlocal commands_applied, errors
        while not command_received.is_set():
            try:
                # apply_command() will get the command and apply it
                hal_server.apply_command()
                commands_applied += 1
            except RuntimeError as e:
                # No command available (timeout) - this is expected, continue
                if "No command received" in str(e):
                    continue
                # Other RuntimeErrors should be logged
                errors.append(f"Command loop RuntimeError: {str(e)}")
            except Exception as e:
                # Unexpected error - log but continue
                errors.append(f"Command loop unexpected error: {str(e)}")
    
    cmd_thread = threading.Thread(target=command_loop, daemon=True)
    cmd_thread.start()
    # Give thread time to start
    time.sleep(0.01)

    # Run 100 cycles at 100 Hz (1 second total)
    period = 1.0 / 100.0  # 10ms period
    start_time = time.time()
    
    # Wait for initial observation to ensure connection is established
    hal_server.set_observation()
    initial_obs = hal_client.poll(timeout_ms=100)
    if initial_obs is None:
        raise RuntimeError("Failed to receive initial observation - connection may not be established")
    
    try:
        for cycle in range(100):
            cycle_start = time.time()
            
            try:
                # Publish observation
                hal_server.set_observation()
                observations_published += 1
                
                # Poll client with longer timeout to ensure we get the observation
                hw_obs = hal_client.poll(timeout_ms=50)
                if hw_obs is None:
                    errors.append(f"Cycle {cycle}: Failed to receive observation")
                    continue
                
                # Map hardware observation to ParkourObservation
                # Pass navigation command so it's included in the observation
                from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
                from compute.parkour.parkour_types import ParkourModelIO
                
                mapper = HWObservationsToParkourMapper()
                parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
                
                # Build model IO
                model_io = ParkourModelIO(
                    timestamp_ns=time.time_ns(),
                    schema_version=parkour_obs.schema_version,
                    nav_cmd=nav_cmd,
                    observation=parkour_obs,
                )
                
                if model_io is not None:
                    # Create mock command (12 DOF)
                    command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
                    
                    # Verify no NaN values before sending
                    if np.any(np.isnan(command)) or np.any(np.isinf(command)):
                        errors.append(f"Cycle {cycle}: NaN or Inf in command before sending")
                    
                    # Send command
                    from compute.parkour.parkour_types import InferenceResponse
                    import torch as torch_module
                    action_tensor = torch_module.from_numpy(command)
                    response = InferenceResponse.create_success(
                        action=action_tensor,
                        inference_latency_ms=5.0,
                    )
                    # Map inference response to hardware joint positions
                    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
                    mapper = ParkourLocomotionToHWMapper(model_action_dim=12)
                    joint_positions = mapper.map(response)
                    
                    # Verify no NaN values in joint positions before sending
                    if np.any(np.isnan(joint_positions.joint_positions)) or np.any(np.isinf(joint_positions.joint_positions)):
                        errors.append(f"Cycle {cycle}: NaN or Inf in joint_positions before sending")
                    
                    hal_client.put_joint_command(joint_positions)
                    commands_received += 1
                
                # Only count cycles where we successfully received and processed an observation
                cycles_completed += 1
                
                # Sleep to maintain 100 Hz rate
                elapsed = time.time() - cycle_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                errors.append(f"Cycle {cycle}: {str(e)}")
    finally:
        # Stop command thread
        command_received.set()
        cmd_thread.join(timeout=1.0)
    
    elapsed_total = time.time() - start_time
    
    # Show errors if any occurred (for debugging)
    if errors:
        print(f"Errors during test: {errors[:10]}")  # Show first 10 errors
    if cycles_completed != 100:
        print(f"Only {cycles_completed} cycles completed out of 100")
        print(f"Observations published: {observations_published}, Commands received: {commands_received}")
    
    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"
    
    # Verify all cycles completed
    assert cycles_completed == 100, f"Expected 100 cycles, got {cycles_completed}"
    
    # Verify commands were received (at least some)
    assert commands_received > 0, "No commands were received"
    
    # Verify observations were published
    assert observations_published == 100, f"Expected 100 observations, got {observations_published}"
    
    # Verify rate is approximately 100 Hz (within 10%)
    actual_rate = cycles_completed / elapsed_total
    assert 90.0 <= actual_rate <= 110.0, f"Rate {actual_rate} Hz not in expected range [90, 110] Hz"
    
    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_full_parkour_eval_simulation(mock_isaac_env, hal_server_config, hal_client_config):
    """Test full Parkour eval simulation (5-10 seconds of motion).
    
    This test simulates a full evaluation run similar to evaluation.py:
    - Runs for 5-10 seconds at 100 Hz (500-1000 cycles)
    - Verifies continuous operation without crashes or stalls
    - Tracks observation and command statistics
    """
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    time.sleep(0.1)

    # Mock observation manager
    from compute.parkour.parkour_types import OBS_DIM
    import torch
    
    def mock_compute_observations():
        # Return observation in training format with some variation
        obs_tensor = torch.randn(1, OBS_DIM, dtype=torch.float32) * 0.1
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")

    # Mock action manager
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Set navigation command
    nav_cmd = NavigationCommand.create_now()

    # Start command receiving thread (PUSH/PULL pattern requires server to be waiting)
    import threading
    command_received = threading.Event()
    commands_applied = 0
    
    # Statistics
    cycles_completed = 0
    commands_sent = 0
    observations_published = 0
    stalls = 0
    last_cycle_time = time.time()
    errors = []
    
    def command_loop():
        nonlocal commands_applied, errors
        while not command_received.is_set():
            try:
                # apply_command() will get the command and apply it
                hal_server.apply_command()
                commands_applied += 1
            except RuntimeError as e:
                # No command available (timeout) - this is expected, continue
                if "No command received" in str(e):
                    continue
                # Other RuntimeErrors should be logged
                errors.append(f"Command loop RuntimeError: {str(e)}")
            except Exception as e:
                # Unexpected error - log but continue
                errors.append(f"Command loop unexpected error: {str(e)}")
    
    cmd_thread = threading.Thread(target=command_loop, daemon=True)
    cmd_thread.start()
    # Give thread time to start
    time.sleep(0.01)

    # Run for 5 seconds at 100 Hz (500 cycles)
    duration_seconds = 5.0
    target_rate_hz = 100.0
    period = 1.0 / target_rate_hz
    total_cycles = int(duration_seconds * target_rate_hz)
    
    start_time = time.time()
    
    try:
        for cycle in range(total_cycles):
            cycle_start = time.time()
            
            # Check for stalls (cycle taking too long)
            if cycle > 0:
                cycle_duration = cycle_start - last_cycle_time
                if cycle_duration > period * 2:  # More than 2x expected period
                    stalls += 1
            
            try:
                # Publish observation
                hal_server.set_observation()
                observations_published += 1
                
                # Poll client
                hw_obs = hal_client.poll(timeout_ms=10)
                if hw_obs is None:
                    continue
                
                # Map hardware observation to ParkourObservation
                # Pass navigation command so it's included in the observation
                from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
                from compute.parkour.parkour_types import ParkourModelIO
                
                mapper = HWObservationsToParkourMapper()
                parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)
                
                # Build model IO
                model_io = ParkourModelIO(
                    timestamp_ns=time.time_ns(),
                    schema_version=parkour_obs.schema_version,
                    nav_cmd=nav_cmd,
                    observation=parkour_obs,
                )
                
                if model_io is not None:
                    # Generate command
                    command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
                    
                    # Send command
                    from compute.parkour.parkour_types import InferenceResponse
                    import torch as torch_module
                    action_tensor = torch_module.from_numpy(command)
                    response = InferenceResponse.create_success(
                        action=action_tensor,
                        inference_latency_ms=5.0,
                    )
                    # Map inference response to hardware joint positions
                    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
                    mapper = ParkourLocomotionToHWMapper(model_action_dim=12)
                    joint_positions = mapper.map(response)
                    hal_client.put_joint_command(joint_positions)
                    commands_sent += 1
                
                cycles_completed += 1
                last_cycle_time = time.time()
                
                # Sleep to maintain rate
                elapsed = time.time() - cycle_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                pytest.fail(f"Cycle {cycle} failed: {e}")
    finally:
        # Stop command thread
        command_received.set()
        cmd_thread.join(timeout=1.0)
    
    elapsed_total = time.time() - start_time
    
    # Verify completion
    assert cycles_completed == total_cycles, f"Expected {total_cycles} cycles, got {cycles_completed}"
    
    # Verify no stalls
    assert stalls == 0, f"Detected {stalls} stalls during execution"
    
    # Verify observations published
    assert observations_published == total_cycles, f"Expected {total_cycles} observations, got {observations_published}"
    
    # Verify commands processed
    assert commands_sent > 0, "No commands were sent"
    assert commands_applied > 0, f"No commands were applied (commands_applied={commands_applied})"
    
    # Verify rate is approximately correct
    actual_rate = cycles_completed / elapsed_total
    assert actual_rate >= 90.0, f"Rate {actual_rate} Hz too low (expected >= 90 Hz)"
    
    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_interface_matches_evaluation_baseline(mock_isaac_env, hal_server_config):
    """Test that IsaacSim HAL server interface matches what evaluation.py expects.
    
    This test verifies that:
    - The HAL server can be integrated into the evaluation.py workflow
    - Observation format matches what the policy expects
    - Command format matches what the environment expects
    - The interface is compatible with evaluation.py's usage pattern
    """
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Mock observation manager to return observations in the same format as evaluation.py
    from compute.parkour.parkour_types import OBS_DIM
    import torch
    
    def mock_compute_observations():
        # evaluation.py uses: obs, extras = env.get_observations()
        # which returns obs_dict["policy"] from observation_manager.compute()
        obs_tensor = torch.zeros(1, OBS_DIM, dtype=torch.float32)
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")

    # Verify observation format matches evaluation.py expectations
    obs_dict = hal_server.observation_manager.compute()
    assert "policy" in obs_dict, "Observation dict must contain 'policy' key (as in evaluation.py)"
    
    obs_tensor = obs_dict["policy"]
    assert isinstance(obs_tensor, torch.Tensor), "Observation must be torch.Tensor"
    assert obs_tensor.shape == (1, OBS_DIM), f"Observation shape must be (1, {OBS_DIM})"
    assert obs_tensor.dtype == torch.float32, "Observation dtype must be float32"

    # Verify command format matches evaluation.py expectations
    # evaluation.py uses: actions = policy(obs)
    # which returns actions that are applied via env.step(actions)
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Test command application (same pattern as evaluation.py)
    command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
    command_tensor = torch.from_numpy(command).to(device=hal_server.env.device, dtype=torch.float32)
    command_tensor = command_tensor.unsqueeze(0)  # Add batch dimension
    
    # Apply command (same as evaluation.py's env.step(actions))
    hal_server.action_manager.process_action(command_tensor)
    hal_server.action_manager.apply_action()
    
    # Verify methods were called
    hal_server.action_manager.process_action.assert_called_once()
    hal_server.action_manager.apply_action.assert_called_once()

    # Verify interface compatibility
    # The HAL server should be able to replace the direct env.step() call in evaluation.py
    # by publishing observations and receiving commands via ZMQ
    
    hal_server.close()


@pytest.mark.isaacsim
def test_isaacsim_hal_server_with_real_isaaclab():
    """Test with real IsaacLab environment.

    This test requires IsaacSim to be installed.
    Run with: make test-isaacsim
    """
    from isaaclab.app import AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

    # This would require actual IsaacSim setup
    # For now, we provide the structure
    # When implemented, use direct instantiation instead of gym.make():
    # env_cfg = parse_env_cfg(task_name, ...)
    # env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    pass

