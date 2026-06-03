"""Unit tests for IsaacSimMCUSDK integration in IsaacSimHalServer.


"""

import threading
import time

import numpy as np
import pytest
import torch
import zmq

try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

from hal.client.data_structures.hardware import JointCommand
from hal.server import HalServerConfig
from hal.server.isaac.hal_server import IsaacSimHalServer
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION


# Minimal environment mock - only required attributes
class MockScene:
    """Minimal scene mock that supports dict-like access."""
    
    def __init__(self, robot):
        self._robot = robot
        self.sensors = {}
    
    def __getitem__(self, key):
        if key == "robot":
            return self._robot
        raise KeyError(f"Key '{key}' not found")
    
    def keys(self):
        return ["robot"]


class MockRobot:
    """Minimal robot mock with only required data attributes."""
    
    def __init__(self, num_envs=1):
        self.data = type('obj', (object,), {
            'joint_pos': torch.zeros((num_envs, 12), dtype=torch.float32),
            'root_ang_vel_b': torch.zeros((num_envs, 3), dtype=torch.float32),
            'root_lin_vel_b': torch.zeros((num_envs, 3), dtype=torch.float32),
            'root_quat_w': torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32),
            'joint_vel': torch.zeros((num_envs, 12), dtype=torch.float32),
        })()


class MockObservationManager:
    """Minimal observation manager mock."""

    def compute(self):
        obs_dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
            KRABBY_QUAD_DEFINITION
        )
        obs_tensor = torch.ones(obs_dims.obs_dim, dtype=torch.float32) * 0.1
        return {"policy": obs_tensor}


class MockEnv:
    """Minimal environment mock with only required attributes."""
    
    def __init__(self, num_envs=1, device=torch.device("cpu")):
        self.num_envs = num_envs
        self.device = device
        self.unwrapped = self
        self.scene = MockScene(MockRobot(num_envs))
        self.observation_manager = MockObservationManager()
        self.action_manager = type('obj', (object,), {})()  # Empty object


@pytest.fixture
def hal_server_config():
    """Create HAL server config for testing."""
    return HalServerConfig(
        observation_bind="inproc://test_mcusdk_observation",
        command_bind="inproc://test_mcusdk_command",
    )


class TestIsaacSimHalServerMCUSDK:
    """Test core MCUSDK functionality in IsaacSimHalServer."""
    
    def test_mcusdk_initialized_when_env_provided(self, hal_server_config):
        """Test SDK is initialized when env is provided."""
        mock_env = MockEnv()
        hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_env)
        
        assert hal_server._mcusdk is not None
        assert isinstance(hal_server._mcusdk, IsaacSimMCUSDK)
    
    def test_mcusdk_not_initialized_when_env_is_none(self, hal_server_config):
        """Test SDK is NOT initialized when env is None."""
        hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=None)
        
        assert hal_server._mcusdk is None
        
        # Calling _initialize_mcusdk() directly should not raise and should not initialize
        hal_server._initialize_mcusdk()
        assert hal_server._mcusdk is None
    
    def test_apply_command_flow(self, hal_server_config):
        """Test apply_command flow: get command via ZMQ → call SDK → convert to torch → return tensor."""
        mock_env = MockEnv(num_envs=1)
        hal_server = IsaacSimHalServer(hal_server_config, KRABBY_QUAD_DEFINITION, env=mock_env)
        hal_server.initialize()
        
        # Create command to send (12 joints for quad robot)
        command_values = np.zeros(12, dtype=np.float32)
        command = JointCommand(
            _joint_positions=command_values,
            timestamp_ns=time.time_ns(),
            observation_timestamp_ns=time.time_ns(),
            joint_names=KRABBY_QUAD_DEFINITION.get_joint_names(),
        )
        
        # Use real ZMQ sockets - create pusher to send command
        transport_context = hal_server.get_transport_context()
        pusher = transport_context.socket(zmq.PUSH)
        pusher.setsockopt(zmq.SNDHWM, 5)
        pusher.connect("inproc://test_mcusdk_command")
        time.sleep(0.01)  # Small delay for pusher to connect
        
        # Server needs to be waiting before client sends (PUSH/PULL pattern)
        received_action = [None]
        
        def server_apply():
            received_action[0] = hal_server.apply_command()
        
        server_thread = threading.Thread(target=server_apply)
        server_thread.start()
        time.sleep(0.01)  # Small delay to ensure server thread is waiting
        
        # Send command via ZMQ (single-blob)
        pusher.send(command.to_bytes())
        
        server_thread.join(timeout=2.0)
        action = received_action[0]
        
        # Verify action was received
        assert action is not None
        
        # Verify returned tensor has correct shape after transformations
        # SDK returns (12,) numpy → converted to torch (12,) → batch dim added → (1, 12)
        assert action.shape == (1, 12)
        assert action.dtype == torch.float32
        
        # Verify values match command (compare dicts)
        joint_names = KRABBY_QUAD_DEFINITION.get_joint_names()
        actual_dict = dict(zip(joint_names, action[0].tolist()))
        assert actual_dict == command.to_positions_dict()
        
        pusher.close()
