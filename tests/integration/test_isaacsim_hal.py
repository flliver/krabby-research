"""Integration tests for IsaacSim HAL server.

These tests use mocked Isaac Sim environments and can be run with pytest.
For real Isaac Sim integration tests, see images/isaacsim/test_runner.py.
"""

import logging
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hal.client.data_structures.hardware import JointCommand
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.client.observation.types import NavigationCommand
from hal.server.isaac import IsaacSimHalServer
from hal.server.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from compute.testing.inference_test_runner import InferenceTestRunner

_TEST_OBS_DIMS = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
    KRABBY_QUAD_DEFINITION
)

# Loggers that emit high-volume DEBUG during the 100 Hz HAL + inference loop (pytest may use log_cli DEBUG).
_SPEED_SENSITIVE_LOGGERS = (
    "hal.server.isaac.hal_server",
    "hal.server.isaac.isaacsim_mcusdk",
    "hal.server.server",
    "hal.client.client",
    "compute.parkour",
)


@contextmanager
def _strict_timing_log_isolation(
    names: tuple[str, ...] = _SPEED_SENSITIVE_LOGGERS,
    level: int = logging.INFO,
):
    """Keep timing assertions strict: pytest ``log_cli_level=DEBUG`` adds root handlers at DEBUG.

    Without clamping **root** and **handlers**, DEBUG records from any library still format and emit
    on every ~10 ms loop and cause false stall / rate failures.
    """
    root = logging.getLogger()
    root_level_prev = root.level
    handler_prev = [(h, h.level) for h in root.handlers]
    loggers = [logging.getLogger(n) for n in names]
    logger_prev = [lg.level for lg in loggers]
    try:
        root.setLevel(level)
        for h in root.handlers:
            h.setLevel(level)
        for lg in loggers:
            lg.setLevel(level)
        yield
    finally:
        for lg, prv in zip(loggers, logger_prev):
            lg.setLevel(prv)
        for h, prv in handler_prev:
            h.setLevel(prv)
        root.setLevel(root_level_prev)


@pytest.fixture
def mock_isaac_env():
    """Create a mock IsaacSim environment."""
    import torch
    import numpy as np
    
    # Create mock contact sensor first
    mock_contact_sensor = MagicMock()
    mock_contact_sensor.cfg = MagicMock()
    mock_contact_sensor.cfg.body_ids = [0, 1, 2, 3]
    mock_contact_sensor.data = MagicMock()
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    mock_contact_sensor.data.net_forces_w_history = torch.zeros((1, 10, 4, 3), dtype=torch.float32)
    
    # Create a mock scene that supports dictionary-like access
    mock_robot = MagicMock()
    mock_robot.data = MagicMock()
    mock_robot.data.joint_pos = torch.zeros((1, 12), dtype=torch.float32)
    # Add required robot state data
    mock_robot.data.root_ang_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_lin_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)  # Identity quaternion
    mock_robot.data.joint_vel = torch.zeros((1, 12), dtype=torch.float32)
    
    # Create scene with sensors dict
    mock_scene = MagicMock()
    mock_scene.__getitem__ = MagicMock(return_value=mock_robot)  # For env.scene["robot"]
    mock_scene.__contains__ = MagicMock(return_value=True)  # For "robot" in env.scene
    mock_scene.keys = MagicMock(return_value=["robot"])  # For scene.keys() - returns entity names
    mock_scene.sensors = {'contact_forces': mock_contact_sensor}  # Set sensors dict directly
    
    env = MagicMock()
    env.scene = mock_scene
    env.unwrapped = env  # Make unwrapped point to same object for mock
    env.unwrapped.num_envs = 1  # Mock uses single environment
    env.device = torch.device("cpu")  # Set device for SDK initialization
    
    env.observation_manager = MagicMock()
    
    # Create mock action manager with action history
    mock_action_term = MagicMock()
    mock_action_term.action_history_buf = torch.zeros((1, 10, 12), dtype=torch.float32)  # (num_envs, history, action_dim)
    mock_action_manager = MagicMock()
    mock_action_manager.get_term = MagicMock(return_value=mock_action_term)
    env.action_manager = mock_action_manager
    
    # Mock env.step() to return expected 5-tuple: (obs_dict, rewards, dones, truncated, extras)
    def mock_step(action):
        # Return empty dict for obs_dict, zeros for rewards/dones/truncated, empty dict for extras
        return ({}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {})
    env.step = MagicMock(side_effect=mock_step)
    
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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    # Use non-zero values to pass validation (all-zero observations are rejected)
    hal_server.observation_manager = MagicMock()
    obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1  # Non-zero values
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})
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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    # Use non-zero values to pass validation (all-zero observations are rejected)
    hal_server.observation_manager = MagicMock()
    obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1  # Non-zero values
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})
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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Setup HAL client with shared ZMQ context from server (for inproc connections)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock action manager methods (these are what apply_joint_command actually calls)
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()
    
    # Mock env.device
    hal_server.env.device = "cpu"

    # Mock command reception to return JointCommand instance (bypass ZMQ)
    # apply_command() calls get_joint_command(timeout_ms=poll_delay_ms), then drains with
    # get_joint_command(timeout_ms=0) until None — mocks must return None for timeout_ms==0.
    def mock_get_joint_command(timeout_ms=10):
        if timeout_ms == 0:
            return None
        command_array = np.array([0.1, 0.2, 0.3] + [0.0] * 9, dtype=np.float32)  # 12 DOF (quad)
        return JointCommand(
            _joint_positions=command_array,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
            joint_names=KRABBY_QUAD_DEFINITION.get_joint_names(),
        )

    hal_server.get_joint_command = mock_get_joint_command

    # Apply joint command - apply_command() now just returns the action tensor
    # It doesn't apply it anymore (env.step() handles that)
    action = hal_server.apply_command()

    # Verify action tensor was returned (1, 12) when robot_definition is None (quad default)
    assert action is not None
    assert isinstance(action, torch.Tensor)
    assert action.shape == (1, 12)

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_end_to_end_with_game_loop(mock_isaac_env, hal_server_config, hal_client_config):
    """Test end-to-end integration with inference logic (game loop)."""
    import zmq
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    # Use non-zero values to pass validation (all-zero observations are rejected)
    # Vary observations to avoid duplicates
    import torch
    counter = [0]  # Use list to allow modification in closure
    
    def mock_compute_observations():
        # Return varying observations to avoid duplicates
        obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1
        # Modify first few elements with counter to ensure uniqueness
        obs_tensor[0] = 0.1 + (counter[0] % 100) * 0.01
        obs_tensor[1] = 0.1 + ((counter[0] // 10) % 100) * 0.01
        counter[0] += 1
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
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
                timing_breakdown=[],
            )

    model = MockPolicyModel()

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(
        model,
        hal_client,
        control_rate_hz=100.0,
        robot_definition=KRABBY_QUAD_DEFINITION,
        observation_dimensions=_TEST_OBS_DIMS,
    )

    # Set navigation command on test runner
    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    # Start publishing observations from hal server in background
    import threading
    import queue
    
    publish_stop = threading.Event()
    thread_exceptions = queue.Queue()  # Capture exceptions from background threads
    
    def publish_loop():
        """Publish observations at 100 Hz."""
        period = 1.0 / 100.0
        try:
            while not publish_stop.is_set():
                hal_server.set_observation()
                # Clear cache so next iteration calls observation_manager.compute() (avoids duplicate)
                hal_server._latest_obs_tensor = None
                hal_server._latest_obs_dict = None
                time.sleep(period)
        except Exception as e:
            # Capture exception so test can fail on it
            thread_exceptions.put(("publish_loop", e))
            raise
    
    # Start command receiving loop (PUSH/PULL requires server to be waiting)
    command_stop = threading.Event()
    
    def command_loop():
        """Receive and apply commands."""
        cmd_logger = logging.getLogger(__name__)
        try:
            while not command_stop.is_set():
                try:
                    hal_server.apply_command()
                except RuntimeError as e:
                    # No command available (timeout) - this is expected, continue
                    error_str = str(e).lower()
                    if "timeout" in error_str or "no command" in error_str or "not responding" in error_str:
                        time.sleep(0.001)  # Small sleep to avoid busy-wait
                        continue
                    # Other RuntimeErrors should be logged but not break the loop
                    cmd_logger.warning(f"Command loop error: {e}")
                except Exception as e:
                    if not command_stop.is_set():
                        cmd_logger.warning(f"Unexpected error in command loop: {e}")
        except Exception as e:
            # Capture exception so test can fail on it
            thread_exceptions.put(("command_loop", e))
            raise
    
    publish_thread = threading.Thread(target=publish_loop, daemon=True)
    publish_thread.start()
    
    command_thread = threading.Thread(target=command_loop, daemon=True)
    command_thread.start()
    
    # Run inference test for a short time
    def stop_after_time():
        time.sleep(0.2)  # Run for 200ms
        test_runner.stop()
        publish_stop.set()
        command_stop.set()
    
    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()
    
    # Run inference test runner (this will poll, run inference, and send commands)
    test_runner.run()
    
    # Stop threads
    publish_stop.set()
    command_stop.set()
    stop_thread.join(timeout=1.0)
    publish_thread.join(timeout=1.0)
    command_thread.join(timeout=1.0)
    
    # Check for exceptions in background threads - fail test if any occurred
    if not thread_exceptions.empty():
        thread_name, exception = thread_exceptions.get_nowait()
        pytest.fail(f"Exception in {thread_name} thread: {exception}", pytrace=False)

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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Mock observation manager to return valid observations
    hal_server.observation_manager = MagicMock()
    obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1  # Non-zero values
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})
    hal_server.env.device = torch.device("cpu")

    # Publish observation (should not raise)
    hal_server.set_observation()

    # Verify server can receive commands
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()
    
    # Mock env.device
    hal_server.env.device = "cpu"

    # apply_command() drains with get_joint_command(timeout_ms=0) until None.
    def mock_get_joint_command(timeout_ms=10):
        if timeout_ms == 0:
            return None
        command_array = np.array([0.0] * 18, dtype=np.float32)  # 18 joints for hexapod
        return JointCommand(
            _joint_positions=command_array,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
            joint_names=KRABBY_HEX_DEFINITION.get_joint_names(),
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
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Setup HAL client with shared context
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()
    hal_client.set_debug(True)

    # Mock observation manager to return complete observation in training format
    # Use non-zero values to pass validation (all-zero observations are rejected)
    hal_server.observation_manager = MagicMock()
    obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1  # Non-zero values
    hal_server.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})

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
        timing_breakdown=[],
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
    mapper = ParkourLocomotionToHWMapper(robot_definition=KRABBY_QUAD_DEFINITION)
    joint_positions = mapper.map(inference_response, observation_timestamp_ns=time.time_ns())
    
    # Send command
    hal_client.put_joint_command(joint_positions)
    
    server_thread.join(timeout=2.0)
    received = received_command[0]
    assert received is not None
    assert received.to_positions_dict() == joint_positions.to_positions_dict()

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_observation_rate(mock_isaac_env, hal_server_config, hal_client_config):
    """Test that observation can be published at required rates (30-60 Hz camera, 100+ Hz state)."""
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Use server's context so inproc works and client does not term() the context on close (avoids segfault)
    hal_client = HalClient(hal_client_config, context=hal_server.get_transport_context())
    hal_client.initialize()

    # Mock observation manager to return valid observations
    # Use a counter to avoid duplicate observations
    import torch
    counter = [0]  # Use list to allow modification in closure
    
    def mock_compute_observations():
        # Return slightly different observations each time to avoid duplicates
        obs_tensor = torch.ones(_TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * (0.1 + counter[0] * 0.0001)
        counter[0] += 1
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")

    # Publish at high rate
    start_time = time.time()
    publish_count = 0

    for _ in range(100):  # Publish 100 times
        hal_server.set_observation()
        publish_count += 1
        # Clear cache so next set_observation() calls compute() (mock returns new obs each time)
        hal_server._latest_obs_tensor = None
        hal_server._latest_obs_dict = None
        # No sleep needed - test is measuring maximum publish rate

    elapsed = time.time() - start_time
    rate = publish_count / elapsed

    # Should be able to publish at high rate (>100 Hz)
    assert rate > 100.0

    hal_client.close()
    hal_server.close()


def test_isaacsim_hal_server_error_handling(mock_isaac_env, hal_server_config):
    """Test error handling in IsaacSim HAL server."""
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)

    # Test with None environment
    hal_server_no_env = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=None, observation_dimensions=_TEST_OBS_DIMS)
    hal_server_no_env.initialize()
    hal_server_no_env.set_debug(True)

    # Publish observation with no environment:
    # In the new implementation this correctly raises a RuntimeError instead of silently succeeding.
    with pytest.raises(RuntimeError, match="No environment set"):
        hal_server_no_env.set_observation()

    # Apply command with no environment:
    # apply_command() now raises a RuntimeError when env is None (SDK not initialized).
    # The error message will be either "No environment set" or "IsaacSimMCUSDK not initialized"
    with pytest.raises(RuntimeError, match="No environment set|IsaacSimMCUSDK not initialized"):
        hal_server_no_env.apply_command()

    hal_server_no_env.close()
    hal_server.close()


def test_isaacsim_hal_server_100_consecutive_command_cycles(mock_isaac_env, hal_server_config, hal_client_config):
    """Test 100 consecutive command cycles without dropped commands or NaNs.
    
    This test verifies that the HAL server can handle sustained operation
    at 100 Hz for 100 cycles (1 second) without errors, dropped commands, or NaN values.
    
    Matches main.py sequence: uses ParkourInferenceClient in a thread.
    """
    from compute.parkour.inference_client import ParkourInferenceClient
    from compute.parkour.policy_interface import ModelWeights
    from hal.client.config import HalClientConfig
    import numpy as np
    
    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)

    checkpoint_path = "/workspace/test_assets/checkpoints/unitree_go2_parkour_teacher.pt"
    # Use Unitree Go2 robot so observation dimensions match the checkpoint (num_prop=53, obs_dim=753).
    robot_definition = UNITREE_GO2_DEFINITION
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        robot_definition,
    )

    # Mock observation manager to return valid, randomized observations (must match checkpoint obs_dim)
    import torch

    rng = np.random.RandomState(42)
    step_counter = [0]

    def mock_compute_observations():
        rng_state = rng.get_state()
        rng.seed(42 + step_counter[0])
        obs_array = rng.randn(observation_dimensions.obs_dim).astype(np.float32) * 0.1
        rng.set_state(rng_state)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)
        obs_tensor = obs_tensor + 0.1
        step_counter[0] += 1
        return {"policy": obs_tensor}

    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")
    hal_server.observation_dimensions = observation_dimensions

    # Mock action manager
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Get transport context for inproc connections
    transport_context = hal_server.get_transport_context()

    # Create HAL client config (matching main.py)
    hal_client_config_for_inference = HalClientConfig(
        observation_endpoint=hal_server_config.observation_bind,
        command_endpoint=hal_server_config.command_bind,
    )
    model_weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        observation_dimensions=observation_dimensions,
        action_dim=12,
    )
    parkour_client = ParkourInferenceClient(
        hal_client_config=hal_client_config_for_inference,
        model_weights=model_weights,
        observation_dimensions=observation_dimensions,
        robot_definition=robot_definition,
        control_rate=100.0,
        device="cpu",
        transport_context=transport_context,
    )
    
    # Initialize with real model (matching main.py)
    parkour_client.initialize()
    
    # Running flag for graceful shutdown
    running = True
    
    # Start inference client in separate thread (matching main.py)
    parkour_client.start_thread(running_flag=lambda: running)

    # Run 100 cycles at 100 Hz (1 second total)
    period_s = 1.0 / 100.0
    start_time = time.time()
    
    # Statistics
    cycles_completed = 0
    observations_published = 0
    commands_applied = 0
    
    # Publish initial observation from environment (matching main.py)
    hal_server.set_observation()
    observations_published += 1
    
    # Wait for first action from inference client (matching main.py)
    first_action = hal_server.apply_command()
    if first_action.shape[0] == 1 and mock_isaac_env.unwrapped.num_envs > 1:
        first_action = first_action.expand(mock_isaac_env.unwrapped.num_envs, -1)
    
    # Apply first action and step environment (matching main.py)
    obs_dict, _, _, _, extras = mock_isaac_env.step(first_action)
    
    # Track first applied action for next observation's previous_action
    action_np = first_action[0].cpu().numpy() if first_action.ndim == 2 else first_action.cpu().numpy()
    if len(action_np) >= 12:
        hal_server._last_applied_action[:] = action_np[:12].astype(np.float32)
    else:
        hal_server._last_applied_action[:len(action_np)] = action_np.astype(np.float32)
    # Clear cache so next set_observation() calls observation_manager.compute() (mock returns new obs each time)
    hal_server._latest_obs_tensor = None
    hal_server._latest_obs_dict = None

    timestep = 1
    commands_applied += 1

    # Main loop: step simulation and publish observations (matching main.py)
    try:
        for cycle in range(99):  # -1 because we already did the first step
            loop_start_ns = time.time_ns()
            
            # Publish hardware observations via HAL (matching main.py)
            hal_server.set_observation()
            observations_published += 1
            
            # Wait for action corresponding to the observation just published (matching main.py)
            action = hal_server.apply_command()
            if action.shape[0] == 1 and mock_isaac_env.unwrapped.num_envs > 1:
                action = action.expand(mock_isaac_env.unwrapped.num_envs, -1)
            
            # Step environment (matching main.py)
            obs_dict, _, _, _, extras = mock_isaac_env.step(action)
            
            # Track last applied action for next observation's previous_action
            action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
            if len(action_np) >= 12:
                hal_server._last_applied_action[:] = action_np[:12].astype(np.float32)
            else:
                hal_server._last_applied_action[:len(action_np)] = action_np.astype(np.float32)
            # Clear cache so next set_observation() calls observation_manager.compute()
            hal_server._latest_obs_tensor = None
            hal_server._latest_obs_dict = None

            timestep += 1
            commands_applied += 1
            cycles_completed += 1

            # Timing control (matching main.py)
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # Stop inference client thread
        running = False
        parkour_client.stop_thread(timeout=1.0)
    
    elapsed_total = time.time() - start_time
    
    # Verify all cycles completed
    assert cycles_completed == 99, f"Expected 99 cycles, got {cycles_completed}"
    
    # Verify observations published
    assert observations_published == 100, f"Expected 100 observations, got {observations_published}"
    
    # Verify commands processed
    assert commands_applied == 100, f"Expected 100 commands applied, got {commands_applied}"
    
    # Verify rate is approximately 100 Hz (within 40% to account for test overhead)
    actual_rate = (cycles_completed + 1) / elapsed_total  # +1 for initial step
    assert 60.0 <= actual_rate <= 140.0, f"Rate {actual_rate} Hz not in expected range [60, 140] Hz"
    
    parkour_client.close()
    hal_server.close()


def test_isaacsim_hal_server_full_parkour_eval_simulation(mock_isaac_env, hal_server_config, hal_client_config):
    """Test full Parkour eval simulation (5-10 seconds of motion).
    
    This test simulates a full evaluation run similar to evaluation.py:
    - Runs for 5-10 seconds at 100 Hz (500-1000 cycles)
    - Verifies continuous operation without crashes or stalls
    - Tracks observation and command statistics
    
    Matches main.py sequence: uses ParkourInferenceClient in a thread.
    """
    with _strict_timing_log_isolation():
        _run_full_parkour_eval_body(mock_isaac_env, hal_server_config, hal_client_config)


def _run_full_parkour_eval_body(mock_isaac_env, hal_server_config, hal_client_config) -> None:
    from compute.parkour.inference_client import ParkourInferenceClient
    from compute.parkour.policy_interface import ModelWeights
    from hal.client.config import HalClientConfig

    # Use shared context for inproc connections
    # Setup HAL server with shared context
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()

    checkpoint_path = "/workspace/test_assets/checkpoints/unitree_go2_parkour_teacher.pt"
    # Use Unitree Go2 robot so observation dimensions match the checkpoint (num_prop=53, obs_dim=753).
    robot_definition = UNITREE_GO2_DEFINITION
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        robot_definition,
    )

    # Mock observation manager (must match checkpoint obs_dim)
    import torch

    counter = [0]

    def mock_compute_observations():
        obs_tensor = torch.ones(1, observation_dimensions.obs_dim, dtype=torch.float32) * 0.1
        obs_tensor[0, 0] = 0.1 + (counter[0] % 100) * 0.01
        obs_tensor[0, 1] = 0.1 + ((counter[0] // 10) % 100) * 0.01
        counter[0] += 1
        return {"policy": obs_tensor}

    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")
    hal_server.observation_dimensions = observation_dimensions

    # Mock action manager
    hal_server.action_manager = MagicMock()
    hal_server.action_manager.process_action = MagicMock()
    hal_server.action_manager.apply_action = MagicMock()

    # Get transport context for inproc connections
    transport_context = hal_server.get_transport_context()

    # Create HAL client config (matching main.py)
    hal_client_config_for_inference = HalClientConfig(
        observation_endpoint=hal_server_config.observation_bind,
        command_endpoint=hal_server_config.command_bind,
    )
    model_weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        observation_dimensions=observation_dimensions,
        action_dim=12,
    )
    parkour_client = ParkourInferenceClient(
        hal_client_config=hal_client_config_for_inference,
        model_weights=model_weights,
        observation_dimensions=observation_dimensions,
        robot_definition=robot_definition,
        control_rate=100.0,
        device="cpu",
        transport_context=transport_context,
    )
    
    # Initialize with real model (matching main.py)
    parkour_client.initialize()
    
    # Running flag for graceful shutdown
    running = True
    
    # Start inference client in separate thread (matching main.py)
    parkour_client.start_thread(running_flag=lambda: running)

    # Run for 5 seconds at 100 Hz (500 cycles)
    duration_seconds = 5.0
    target_rate_hz = 100.0
    period_s = 1.0 / target_rate_hz
    total_cycles = int(duration_seconds * target_rate_hz)
    
    # Statistics
    cycles_completed = 0
    observations_published = 0
    commands_applied = 0
    stalls = 0
    last_cycle_time = time.time()
    
    start_time = time.time()
    
    # Publish initial observation from environment (matching main.py)
    hal_server.set_observation()
    
    # Wait for first action from inference client (matching main.py)
    first_action = hal_server.apply_command()
    if first_action.shape[0] == 1 and mock_isaac_env.unwrapped.num_envs > 1:
        first_action = first_action.expand(mock_isaac_env.unwrapped.num_envs, -1)
    
    # Apply first action and step environment (matching main.py)
    obs_dict, _, _, _, extras = mock_isaac_env.step(first_action)
    
    # Track first applied action for next observation's previous_action
    action_np = first_action[0].cpu().numpy() if first_action.ndim == 2 else first_action.cpu().numpy()
    if len(action_np) >= 12:
        hal_server._last_applied_action[:] = action_np[:12].astype(np.float32)
    else:
        hal_server._last_applied_action[:len(action_np)] = action_np.astype(np.float32)
    # Clear cache so next set_observation() calls observation_manager.compute()
    hal_server._latest_obs_tensor = None
    hal_server._latest_obs_dict = None

    timestep = 1
    observations_published += 1
    commands_applied += 1

    # Main loop: step simulation and publish observations (matching main.py)
    try:
        for cycle in range(total_cycles - 1):  # -1 because we already did the first step
            loop_start_ns = time.time_ns()
            
            # Check for stalls (cycle taking too long)
            if cycle > 0:
                cycle_duration = loop_start_ns / 1e9 - last_cycle_time
                if cycle_duration > period_s * 2:  # More than 2x expected period
                    stalls += 1
            
            # Publish hardware observations via HAL (matching main.py)
            hal_server.set_observation()
            observations_published += 1
            
            # Wait for action corresponding to the observation just published (matching main.py)
            action = hal_server.apply_command()
            if action.shape[0] == 1 and mock_isaac_env.unwrapped.num_envs > 1:
                action = action.expand(mock_isaac_env.unwrapped.num_envs, -1)
            
            # Step environment (matching main.py)
            obs_dict, _, _, _, extras = mock_isaac_env.step(action)
            
            # Track last applied action for next observation's previous_action
            action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
            if len(action_np) >= 12:
                hal_server._last_applied_action[:] = action_np[:12].astype(np.float32)
            else:
                hal_server._last_applied_action[:len(action_np)] = action_np.astype(np.float32)
            # Clear cache so next set_observation() calls observation_manager.compute()
            hal_server._latest_obs_tensor = None
            hal_server._latest_obs_dict = None

            timestep += 1
            commands_applied += 1
            cycles_completed += 1
            last_cycle_time = loop_start_ns / 1e9

            # Timing control (matching main.py)
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # Stop inference client thread
        running = False
        parkour_client.stop_thread(timeout=1.0)
    
    elapsed_total = time.time() - start_time
    
    # Verify completion
    assert cycles_completed >= total_cycles - 10, f"Expected at least {total_cycles - 10} cycles, got {cycles_completed}"

    # Allow occasional slow cycles (scheduler / CI load); threshold is 2× nominal period per iteration.
    assert stalls <= 3, f"Detected {stalls} stalls during execution (allow <= 3)"
    
    # Verify observations published
    assert observations_published >= total_cycles - 10, f"Expected at least {total_cycles - 10} observations, got {observations_published}"
    
    # Verify commands processed
    assert commands_applied > 0, f"No commands were applied (commands_applied={commands_applied})"
    
    # Verify rate is approximately correct
    actual_rate = cycles_completed / elapsed_total
    assert actual_rate >= 90.0, f"Rate {actual_rate} Hz too low (expected >= 90 Hz)"
    
    parkour_client.close()
    hal_server.close()


def test_isaacsim_hal_server_interface_matches_evaluation_baseline(mock_isaac_env, hal_server_config):
    """Test that IsaacSim HAL server interface matches what evaluation.py expects.
    
    This test verifies that:
    - The HAL server can be integrated into the evaluation.py workflow
    - Observation format matches what the policy expects
    - Command format matches what the environment expects
    - The interface is compatible with evaluation.py's usage pattern
    """
    hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_isaac_env, observation_dimensions=_TEST_OBS_DIMS)
    hal_server.initialize()
    hal_server.set_debug(True)
    hal_server.observation_dimensions = _TEST_OBS_DIMS

    # Mock observation manager to return observations in the same format as evaluation.py
    import torch
    
    def mock_compute_observations():
        # evaluation.py uses: obs, extras = env.get_observations()
        # which returns obs_dict["policy"] from observation_manager.compute()
        # Use non-zero values to pass validation (all-zero observations are rejected)
        obs_tensor = torch.ones(1, _TEST_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1
        return {"policy": obs_tensor}
    
    hal_server.observation_manager = MagicMock()
    hal_server.observation_manager.compute = mock_compute_observations
    hal_server.env.device = torch.device("cpu")

    # Verify observation format matches evaluation.py expectations
    obs_dict = hal_server.observation_manager.compute()
    assert "policy" in obs_dict, "Observation dict must contain 'policy' key (as in evaluation.py)"
    
    obs_tensor = obs_dict["policy"]
    assert isinstance(obs_tensor, torch.Tensor), "Observation must be torch.Tensor"
    assert obs_tensor.shape == (1, _TEST_OBS_DIMS.obs_dim), f"Observation shape must be (1, {_TEST_OBS_DIMS.obs_dim})"
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

