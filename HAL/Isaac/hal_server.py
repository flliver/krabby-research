"""IsaacSim HAL server implementation."""

import logging
import time
from typing import Optional

import numpy as np
import torch

from HAL.ZMQ.server import HalServerBase
from HAL.config import HalServerConfig
from HAL.telemetry.types import OBS_DIM

logger = logging.getLogger(__name__)


class IsaacSimHalServer(HalServerBase):
    """HAL server for IsaacSim environment.

    Extracts telemetry from IsaacSim environment and publishes via ZMQ.
    Applies joint commands received via ZMQ to the environment.
    """

    def __init__(self, config: HalServerConfig, env=None, context=None):
        """Initialize IsaacSim HAL server.

        Args:
            config: HAL server configuration
            env: IsaacSim environment (ParkourManagerBasedRLEnv or similar)
            context: Optional shared ZMQ context (useful for inproc connections)
        """
        super().__init__(config, context=context)
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
                logger.warning("Could not find robot in scene")
            if self.observation_manager is None:
                logger.warning("Observation manager not available")
            if self.action_manager is None:
                logger.warning("Action manager not available")

            logger.info("Cached environment references successfully")
        except Exception as e:
            logger.error(f"Could not cache all environment references: {e}", exc_info=True)

    def publish_telemetry(self) -> None:
        """Publish telemetry from IsaacSim environment.

        Extracts complete observation in training format from observation_manager
        and publishes via HalServerBase.publish_observation().

        The observation matches the exact training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        """
        if self.env is None:
            logger.warning("No environment set, cannot publish telemetry")
            return

        if self.observation_manager is None:
            logger.warning("Observation manager not available, cannot publish telemetry")
            return

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
            self.publish_observation(obs_array)

        except Exception as e:
            logger.error(f"Error publishing telemetry: {e}", exc_info=True)


    def apply_joint_command(self) -> bool:
        """Apply joint command received from HAL.

        Receives command via HalServerBase and applies to action manager.
        Uses the same code path as training: action_manager.process_actions() and apply_actions().

        Returns:
            True if command applied successfully, False otherwise
        """
        if self.env is None:
            logger.warning("No environment set, cannot apply joint command")
            return False

        if self.action_manager is None:
            logger.warning("Action manager not available, cannot apply joint command")
            return False

        try:
            # Zero-copy conversion: receive validated bytes and create tensor directly
            # This avoids NumPy deserialization and the read-only buffer issue
            command_bytes = self.recv_joint_command_bytes(timeout_ms=10)
            if command_bytes is None:
                return False

            # Create tensor directly from bytes (zero-copy, no NumPy intermediate)
            # The bytes are already validated (size, dtype, shape, NaN/Inf checks)
            command_tensor = torch.frombuffer(
                memoryview(command_bytes), dtype=torch.float32
            ).to(device=self.env.device)

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
            logger.error(f"Error applying joint command: {e}", exc_info=True)
            return False

