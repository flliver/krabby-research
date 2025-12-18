"""Mapper from Parkour model output to hardware format.

This mapper converts model inference output (joint locomotion embedding)
into hardware joint position commands. It uses zero-copy operations where
possible.
"""

import logging

import numpy as np
import torch

from compute.parkour.parkour_types import InferenceResponse
from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)

# Krabby has 18 DOF (joints)
KRABBY_JOINT_COUNT = 18


class ParkourLocomotionToKrabbyHWMapper:
    """Maps Parkour model output to hardware format.
    
    Converts model navigation/locomotion output to hardware joint positions.
    May be 1:1 mapping once training is complete.
    
    Zero-copy guarantees:
    - If model outputs 18 joints directly, can use view
    - If model outputs different dimension, requires transformation (copy)
    - Timestamp is always copied (scalar)
    
    Note: The model typically outputs ACTION_DIM joints (usually 12 for quadruped).
    Hardware has 18 joints, so this mapper handles the conversion. Once training
    is complete with 18-joint output, this may become a 1:1 mapping.
    """
    
    def __init__(self, model_action_dim: int = 12):
        """Initialize the mapper.
        
        Args:
            model_action_dim: Action dimension from the model (typically 12)
        """
        self.model_action_dim = model_action_dim
    
    def map(self, model_output: InferenceResponse) -> JointCommand:
        """Map model output to hardware joint positions.
        
        Args:
            model_output: Model inference response containing action tensor
            
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
        
        # Map to 18 joints
        joint_positions = self._map_to_krabby_joints(action_array)
        
        return JointCommand(
            joint_positions=joint_positions,
            timestamp_ns=model_output.timestamp_ns,
        )
    
    def _map_to_krabby_joints(self, model_action: np.ndarray) -> np.ndarray:
        """Map model action to Krabby 18-joint positions.
        
        Args:
            model_action: Model action array (shape: ACTION_DIM,)
            
        Returns:
            Joint positions array (shape: 18,)
        """
        if len(model_action) != self.model_action_dim:
            raise ValueError(
                f"Model action dimension {len(model_action)} != expected {self.model_action_dim}"
            )
        
        # If model outputs 18 joints directly, use as-is (1:1 mapping)
        if self.model_action_dim == KRABBY_JOINT_COUNT:
            # Can use view if compatible, otherwise copy
            if model_action.shape == (KRABBY_JOINT_COUNT,) and model_action.dtype == np.float32:
                # Try to use view if contiguous
                if model_action.flags["C_CONTIGUOUS"]:
                    return model_action
                else:
                    return np.ascontiguousarray(model_action, dtype=np.float32)
            else:
                return np.asarray(model_action, dtype=np.float32)
        
        # Otherwise, need to map from model action dim to 18 joints
        # Placeholder: pad or interpolate to 18 joints
        # In reality, this would depend on the model architecture and training
        joint_positions = np.zeros(KRABBY_JOINT_COUNT, dtype=np.float32)
        
        # For now, map first ACTION_DIM joints directly, pad rest with zeros
        # This is a placeholder - actual mapping will depend on model architecture
        num_to_copy = min(self.model_action_dim, KRABBY_JOINT_COUNT)
        joint_positions[:num_to_copy] = model_action[:num_to_copy]
        
        logger.debug(
            f"Mapped {self.model_action_dim} model joints to {KRABBY_JOINT_COUNT} Krabby joints"
        )
        
        return joint_positions

