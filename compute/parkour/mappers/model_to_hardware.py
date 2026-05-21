"""Mapper from Parkour model output to hardware format.

This mapper converts model inference output (joint locomotion embedding)
into hardware joint position commands. Output length is given by the
robot definition (quad 12, hex 18).
"""

import logging

import numpy as np
import torch

from compute.parkour.parkour_types import InferenceResponse
from hal.client.data_structures.hardware import JointCommand, JointCommandSource
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)


class ParkourLocomotionToHWMapper:
    """Maps Parkour model output to hardware joint commands per robot definition.

    Command length is robot_definition.get_total_joint_count() (12 quad, 18 hex).
    If model output is shorter, pads with zeros; if longer, slices to command length.
    """

    def __init__(self, robot_definition: RobotDefinition):
        """Initialize the mapper.

        Args:
            robot_definition: Robot definition; output length = get_total_joint_count()
        """
        self.robot_definition = robot_definition
        self._command_joint_count = robot_definition.get_total_joint_count()
    
    def map(self, model_output: InferenceResponse, observation_timestamp_ns: int) -> JointCommand:
        """Map model output to hardware joint positions.
        
        Args:
            model_output: Model inference response containing action tensor
            observation_timestamp_ns: Timestamp of the observation this command responds to
            
        Returns:
            JointCommand for hardware control
            
        Raises:
            ValueError: If model output is invalid or failed
        """
        if not model_output.success:
            raise ValueError(f"Model inference failed: {model_output.error_message}")
        
        if model_output.action is None:
            raise ValueError("Model output action is None")
        
        # Get action tensor (zero-copy view)
        action_tensor = model_output.get_action()
        
        # Convert to numpy if needed (creates copy if on GPU, view if on CPU)
        if isinstance(action_tensor, torch.Tensor):
            if action_tensor.is_cuda:
                action_array = action_tensor.cpu().numpy()
            else:
                action_array = action_tensor.numpy()
        else:
            action_array = np.asarray(action_tensor, dtype=np.float32)
        
        # Handle batch dimension
        if action_array.ndim == 2:
            action_array = action_array[0]  # Take first batch element
        elif action_array.ndim != 1:
            raise ValueError(f"Action must be 1D or 2D, got shape {action_array.shape}")
        
        # Ensure float32
        if action_array.dtype != np.float32:
            action_array = action_array.astype(np.float32)
        
        # Map model output to command length per robot definition (pad if hex)
        model_joints = self._map_to_krabby_joints(action_array)
        n = self._command_joint_count
        joint_positions = np.zeros(n, dtype=np.float32)
        joint_positions[: len(model_joints)] = model_joints

        joint_names = self.robot_definition.get_joint_names()
        return JointCommand(
            _joint_positions=joint_positions,
            timestamp_ns=model_output.timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
            joint_names=joint_names,
            source=JointCommandSource.INFERENCE,
        )
    
    def _map_to_krabby_joints(self, model_action: np.ndarray) -> np.ndarray:
        """Copy model action to float32 array; length may be <= command joint count (pad in map())."""
        if len(model_action) == 0:
            raise ValueError("Model action cannot be empty")
        if len(model_action) > self._command_joint_count:
            model_action = model_action[: self._command_joint_count]
        if model_action.dtype == np.float32 and model_action.flags["C_CONTIGUOUS"]:
            return model_action
        return np.ascontiguousarray(model_action, dtype=np.float32)

