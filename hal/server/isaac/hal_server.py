"""IsaacSim HAL server implementation."""

import logging
import time
from typing import Optional

import numpy as np
import torch

from hal.server import HalServerBase, HalServerConfig
from hal.client.observation.types import OBS_DIM

logger = logging.getLogger(__name__)


class IsaacSimHalServer(HalServerBase):
    """HAL server for IsaacSim environment.
    
    Extracts observations from IsaacSim environment and publishes via HAL.
    Applies joint commands received via HAL to the environment.
    """

    def __init__(self, config: HalServerConfig, env=None):
        """Initialize IsaacSim HAL server.
        
        Args:
            config: HAL server configuration
            env: IsaacSim environment (ParkourManagerBasedRLEnv or similar)
        """
        super().__init__(config)
        self.env = env
        self.scene = None
        self.robot = None
        self.observation_manager = None
        self.action_manager = None

        if env is not None:
            self._cache_references()

    def _cache_references(self) -> None:
        """Cache references to environment components."""
        if self.env is None:
            return

        try:
            # Cache scene reference
            self.scene = self.env.scene

            # Cache robot reference (assuming robot is named "robot" in scene)
            # This matches the pattern used in observations.py: env.scene[cfg.params["asset_cfg"].name]
            if "robot" in self.scene:
                self.robot = self.scene["robot"]
            else:
                # Try to find robot by iterating scene entities
                for name, entity in self.scene.items():
                    if hasattr(entity, "data") and hasattr(entity.data, "joint_pos"):
                        self.robot = entity
                        logger.info(f"Found robot entity: {name}")
                        break

            # Cache managers
            self.observation_manager = self.env.observation_manager
            self.action_manager = self.env.action_manager

            # Verify we have required references
            if self.robot is None:
                raise RuntimeError(
                    "Robot not found in scene. "
                    "IsaacSim HAL server requires a robot entity in the scene. "
                    f"Available entities: {list(self.scene.keys()) if self.scene else 'None'}"
                )
            if self.observation_manager is None:
                logger.warning("Observation manager not available")
            if self.action_manager is None:
                logger.warning("Action manager not available")

            logger.info("Cached environment references successfully")
        except Exception as e:
            logger.error(f"Could not cache all environment references: {e}", exc_info=True)

    def set_observation(self) -> None:
        """Set observation from IsaacSim environment.
        
        Extracts observation from observation_manager and publishes it via
        the transport layer. The observation format is model-specific and should
        match the training configuration. This implementation assumes the
        observation_manager produces observations in the format expected by
        the loaded model.
        
        Note: The exact observation format (dimensions, layout) depends on the
        model being used. This should be configured based on the model, not
        hardcoded.
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot set observation")

        if self.observation_manager is None:
            raise RuntimeError("Observation manager not available, cannot set observation")

        try:
            # Compute observations using observation_manager (same as training)
            obs_dict = self.observation_manager.compute()

            # Extract policy observation (matches training format)
            if "policy" not in obs_dict:
                logger.error("Observation dict does not contain 'policy' key")
                return

            obs_tensor = obs_dict["policy"]  # Shape: (num_envs, OBS_DIM)

            # For single environment, extract first environment's observation
            if obs_tensor.ndim == 2:
                if obs_tensor.shape[0] != 1:
                    logger.warning(
                        f"Expected single environment, got {obs_tensor.shape[0]} environments. "
                        "Using first environment's observation."
                    )
                obs_tensor = obs_tensor[0]  # Shape: (OBS_DIM,)

            # Validate shape
            if obs_tensor.shape != (OBS_DIM,):
                logger.error(
                    f"Observation shape {obs_tensor.shape} != expected ({OBS_DIM},). "
                    f"Observation dict keys: {list(obs_dict.keys())}"
                )
                return

            # Convert torch tensor to numpy array (zero-copy if on CPU)
            if isinstance(obs_tensor, torch.Tensor):
                if obs_tensor.is_cuda:
                    obs_array = obs_tensor.cpu().numpy().astype(np.float32)
                else:
                    obs_array = obs_tensor.numpy().astype(np.float32)
            else:
                obs_array = np.asarray(obs_tensor, dtype=np.float32)

            # Ensure contiguous array for zero-copy operations
            if not obs_array.flags["C_CONTIGUOUS"]:
                obs_array = np.ascontiguousarray(obs_array, dtype=np.float32)

            # Publish complete observation in training format
            super().set_observation(obs_array)

        except Exception as e:
            logger.error(f"Error setting observation: {e}", exc_info=True)


    def move(self, joint_positions: Optional[np.ndarray] = None) -> bool:
        """Move robot joints to specified positions.
        
        Renamed from apply_joint_command() to use verb-free naming.
        Gets command from transport layer and applies to IsaacSim environment.
        
        Args:
            joint_positions: Optional joint positions array (shape: (ACTION_DIM,)).
                If None, gets command from transport layer.
        
        Returns:
            True if command applied successfully, False otherwise
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot move robot")

        if self.action_manager is None:
            raise RuntimeError("Action manager not available, cannot move robot")

        try:
            # Get command from transport if not provided
            if joint_positions is None:
                # Zero-copy conversion: receive validated bytes and create tensor directly
                # This avoids NumPy deserialization and the read-only buffer issue
                command_bytes = self.recv_joint_command_bytes(timeout_ms=10)
                if command_bytes is None:
                    return False

                # Create tensor from bytes (copy to make writable for PyTorch)
                # The bytes are already validated (size, dtype, shape, NaN/Inf checks)
                # Note: ZMQ returns read-only bytes, so we need to copy for PyTorch
                command_tensor = torch.frombuffer(
                    bytearray(command_bytes), dtype=torch.float32
                ).to(device=self.env.device)
            else:
                # Convert provided numpy array to tensor
                if isinstance(joint_positions, np.ndarray):
                    command_tensor = torch.from_numpy(joint_positions).to(device=self.env.device, dtype=torch.float32)
                else:
                    command_tensor = joint_positions

            # Add batch dimension if needed (action_manager expects (num_envs, action_dim))
            if command_tensor.ndim == 1:
                command_tensor = command_tensor.unsqueeze(0)  # Shape: (1, ACTION_DIM)

            # Process actions through action manager (same as training)
            # This handles any preprocessing, clipping, etc.
            # Note: process_action (singular) matches the API used in env.step()
            self.action_manager.process_action(command_tensor)

            # Apply actions to set joint position targets (same as training)
            # Note: apply_action (singular) matches the API used in env.step()
            self.action_manager.apply_action()

            return True

        except Exception as e:
            logger.error(f"Error moving robot: {e}", exc_info=True)
            return False

