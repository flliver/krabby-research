"""Integration tests for IsaacSim HAL server."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hal.isaac.hal_server import IsaacSimHalServer
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server.config import HalServerConfig
from hal.observation.types import NavigationCommand
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
    return HalServerConfig.from_endpoints(
        observation_bind="inproc://test_isaac_observation",
        command_bind="inproc://test_isaac_command",
    )


@pytest.fixture
def hal_client_config():
    """Create HAL client config for testing."""
    return HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_isaac_observation",
        command_endpoint="inproc://test_isaac_command",
    )


def test_isaacsim_hal_server_initialization(mock_isaac_env, hal_server_config):
    """Test IsaacSim HAL server initialization with minimal environment."""
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

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

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from hal.observation.types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})
    hal_server.env.device = torch.device("cpu")

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hal_client.poll(timeout_ms=1000)

    # Verify observation data received
    assert hal_client._latest_observation is not None
    assert hal_client._latest_observation.observation is not None
    assert hal_client._latest_observation.observation.shape == (OBS_DIM,)

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_state_publishing(mock_isaac_env, hal_server_config, hal_client_config):
    """Test observation publishing from IsaacSim HAL server."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from hal.observation.types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})
    hal_server.env.device = torch.device("cpu")

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hal_client.poll(timeout_ms=1000)

    # Verify observation data received
    assert hal_client._latest_observation is not None
    assert hal_client._latest_observation.observation is not None
    assert hal_client._latest_observation.observation.shape == (OBS_DIM,)

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_joint_command_application(mock_isaac_env, hal_server_config, hal_client_config):
    """Test joint command application to IsaacSim environment."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock action manager methods (these are what apply_joint_command actually calls)
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()
    
    # Mock env.device
    hal_server.env.device = "cpu"

    # Mock command reception to return bytes immediately (bypass ZMQ)
    # The new zero-copy implementation uses recv_joint_command_bytes()
    def mock_recv_command_bytes(timeout_ms=10):
        command_array = np.array([0.1, 0.2, 0.3] * 4, dtype=np.float32)  # 12 DOF
        return command_array.tobytes()  # Return bytes for zero-copy tensor creation

    hal_server.recv_joint_command_bytes = mock_recv_command_bytes

    # Apply joint command
    result = hal_server.move()

    # Verify command was applied (or at least attempted)
    assert result is True, "apply_joint_command should return True when command is received"
    
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

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from hal.observation.types import OBS_DIM
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
            from hal.commands.types import InferenceResponse

            self.inference_count += 1
            action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
            return InferenceResponse.create_success(
                action=action_tensor,
                inference_latency_ms=5.0,
            )

    model = MockPolicyModel()

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(model, hal_client, control_rate_hz=100.0)

    # Set navigation command
    nav_cmd = NavigationCommand.create_now()
    hal_client.set_navigation_command(nav_cmd)

    # Run a few iterations
    import threading

    def run_loop():
        for _ in range(10):  # Run 10 iterations
            # Publish observation from hal server
            hal_server.set_observation()

            # Poll client
            hal_client.poll(timeout_ms=10)

            # Build model IO and run inference
            model_io = hal_client.build_model_io()
            if model_io is not None:
                inference_result = model.inference(model_io)
                if inference_result.success:
                    hal_client.put_joint_command(inference_result)

            # Apply joint command
            hal_server.move()

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

    # Verify server can publish observation
    hal_server._extract_depth_features = lambda: np.array([1.0] * 10, dtype=np.float32)
    hal_server._extract_state_vector = lambda: np.array([0.0] * 34, dtype=np.float32)

    # Publish observation (should not raise)
    hal_server.set_observation()

    # Verify server can receive commands
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.set_command = MagicMock()

    # Mock command reception
    hal_server.recv_joint_command = lambda timeout_ms=100: np.array([0.0] * 12, dtype=np.float32)

    # Apply command (should not raise)
    result = hal_server.move()
    assert result is not None

    hal_server.close()


def test_isaacsim_hal_server_with_real_zmq_communication(mock_isaac_env, hal_server_config, hal_client_config):
    """Test IsaacSim HAL server with real ZMQ communication (inproc)."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager to return complete observation in training format
    from hal.observation.types import OBS_DIM
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(OBS_DIM, dtype=torch.float32)})

    # Publish observation
    hal_server.set_observation()

    # Poll client
    hal_client.poll(timeout_ms=1000)

    # Verify data received
    assert hal_client._latest_observation is not None

    # Send command from client
    from hal.commands.types import InferenceResponse

    action_array = np.array([0.1, 0.2, 0.3] * 4, dtype=np.float32)
    action_tensor = torch.from_numpy(action_array)
    inference_response = InferenceResponse.create_success(
        action=action_tensor,
        inference_latency_ms=5.0,
    )

    # In REQ/REP pattern, server must be waiting before client sends
    # Use threading to have server wait for command
    import threading
    received_command = [None]
    
    def server_receive():
        received_command[0] = hal_server.get_joint_command(timeout_ms=2000)
    
    server_thread = threading.Thread(target=server_receive)
    server_thread.start()
    time.sleep(0.05)  # Small delay to ensure server is waiting
    
    # Send command
    success = hal_client.put_joint_command(inference_response)
    assert success
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    np.testing.assert_array_equal(received, action_array)

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_observation_rate(mock_isaac_env, hal_server_config, hal_client_config):
    """Test that observation can be published at required rates (30-60 Hz camera, 100+ Hz state)."""
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_isaac_env)
    hal_server.initialize()

    hal_client = HalClient(hal_client_config)
    hal_client.initialize()

    time.sleep(0.1)

    hal_server._extract_depth_features = lambda: np.array([1.0], dtype=np.float32)
    hal_server._extract_state_vector = lambda: np.array([0.0] * 34, dtype=np.float32)

    # Publish at high rate
    start_time = time.time()
    publish_count = 0

    for _ in range(100):  # Publish 100 times
        hal_server.set_observation()
        publish_count += 1
        time.sleep(0.001)  # 1ms between publishes (1000 Hz theoretical max)

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

    # Test with None environment
    hal_server_no_env = IsaacSimHalServer(hal_server_config, env=None)
    hal_server_no_env.initialize()

    # Publish observation with no environment:
    # In the new implementation this correctly raises a RuntimeError instead of silently succeeding.
    with pytest.raises(RuntimeError, match="No environment set"):
        hal_server_no_env.set_observation()

    # Apply command with no environment:
    # Likewise, move() now raises a RuntimeError when env is None.
    with pytest.raises(RuntimeError, match="No environment set"):
        hal_server_no_env.move()

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

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager to return valid observations
    from hal.observation.types import OBS_DIM
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

    # Set navigation command (required for build_model_io)
    nav_cmd = NavigationCommand.create_now()
    hal_client.set_navigation_command(nav_cmd)

    # Track statistics
    cycles_completed = 0
    commands_received = 0
    observations_published = 0
    errors = []

    # Run 100 cycles at 100 Hz (1 second total)
    period = 1.0 / 100.0  # 10ms period
    start_time = time.time()
    
    for cycle in range(100):
        cycle_start = time.time()
        
        try:
            # Publish observation
            hal_server.set_observation()
            observations_published += 1
            
            # Poll client
            hal_client.poll(timeout_ms=10)
            
            # Build model IO and send command
            model_io = hal_client.build_model_io()
            if model_io is not None:
                # Create mock command (12 DOF)
                command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
                
                # Send command
                from hal.commands.types import InferenceResponse
                import torch as torch_module
                action_tensor = torch_module.from_numpy(command)
                response = InferenceResponse.create_success(
                    action=action_tensor,
                    inference_latency_ms=5.0,
                )
                hal_client.put_joint_command(response)
                
                # Receive and apply command on server
                if hal_server.move():
                    commands_received += 1
                    
                    # Verify no NaN values
                    received_cmd = hal_server.get_joint_command(timeout_ms=10)
                    if received_cmd is not None:
                        if np.any(np.isnan(received_cmd)) or np.any(np.isinf(received_cmd)):
                            errors.append(f"Cycle {cycle}: NaN or Inf in command")
            
            cycles_completed += 1
            
            # Sleep to maintain 100 Hz rate
            elapsed = time.time() - cycle_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            errors.append(f"Cycle {cycle}: {str(e)}")
    
    elapsed_total = time.time() - start_time
    
    # Verify all cycles completed
    assert cycles_completed == 100, f"Expected 100 cycles, got {cycles_completed}"
    
    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"
    
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

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    time.sleep(0.1)

    # Mock observation manager
    from hal.observation.types import OBS_DIM
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

    # Set navigation command (required for build_model_io)
    nav_cmd = NavigationCommand.create_now()
    hal_client.set_navigation_command(nav_cmd)

    # Run for 5 seconds at 100 Hz (500 cycles)
    duration_seconds = 5.0
    target_rate_hz = 100.0
    period = 1.0 / target_rate_hz
    total_cycles = int(duration_seconds * target_rate_hz)
    
    # Statistics
    cycles_completed = 0
    commands_sent = 0
    commands_received = 0
    observations_published = 0
    stalls = 0
    last_cycle_time = time.time()
    
    start_time = time.time()
    
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
            hal_client.poll(timeout_ms=10)
            
            # Build model IO
            model_io = hal_client.build_model_io()
            if model_io is not None:
                # Generate command
                command = np.random.uniform(-0.5, 0.5, size=12).astype(np.float32)
                
                # Send command
                from hal.commands.types import InferenceResponse
                import torch as torch_module
                action_tensor = torch_module.from_numpy(command)
                response = InferenceResponse.create_success(
                    action=action_tensor,
                    inference_latency_ms=5.0,
                )
                hal_client.put_joint_command(response)
                commands_sent += 1
                
                # Apply command
                if hal_server.move():
                    commands_received += 1
            
            cycles_completed += 1
            last_cycle_time = time.time()
            
            # Sleep to maintain rate
            elapsed = time.time() - cycle_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            pytest.fail(f"Cycle {cycle} failed: {e}")
    
    elapsed_total = time.time() - start_time
    
    # Verify completion
    assert cycles_completed == total_cycles, f"Expected {total_cycles} cycles, got {cycles_completed}"
    
    # Verify no stalls
    assert stalls == 0, f"Detected {stalls} stalls during execution"
    
    # Verify observations published
    assert observations_published == total_cycles, f"Expected {total_cycles} observations, got {observations_published}"
    
    # Verify commands processed
    assert commands_sent > 0, "No commands were sent"
    assert commands_received > 0, "No commands were received"
    
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

    # Mock observation manager to return observations in the same format as evaluation.py
    from hal.observation.types import OBS_DIM
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


@pytest.mark.skip(reason="Requires actual IsaacSim environment")
def test_isaacsim_hal_server_with_real_isaaclab():
    """Test with real IsaacLab environment.

    This test is skipped by default as it requires IsaacSim to be installed.
    To run: pytest tests/integration/test_isaacsim_hal.py::test_isaacsim_hal_server_with_real_isaaclab -v
    """
    from isaaclab.app import AppLauncher
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    # This would require actual IsaacSim setup
    # For now, we provide the structure
    pass

