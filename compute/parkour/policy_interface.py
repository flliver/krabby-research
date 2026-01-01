"""Parkour policy inference interface using OnPolicyRunnerWithExtractor.

This module provides a minimal interface that uses OnPolicyRunnerWithExtractor
to load checkpoints and get inference policy functions. The interface converts
numpy observations (from hal) to torch tensors and calls the policy function
directly (act_inference) for zero-copy operations.

This is NOT a wrapper - it's a minimal coordinator that:
1. Loads checkpoint using OnPolicyRunnerWithExtractor
2. Gets the inference policy function (act_inference) directly
3. Converts numpy observation to torch tensor (zero-copy)
4. Calls the policy function with hist_encoding=True
5. Returns the torch.Tensor action directly (zero-copy)
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import yaml
from rsl_rl.env import VecEnv

from compute.parkour.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from compute.parkour.parkour_types import InferenceResponse, ParkourModelIO

logger = logging.getLogger(__name__)


@dataclass
class ModelWeights:
    """Model weights container.

    Attributes:
        checkpoint_path: Path to checkpoint file (.pt or .pth)
        config_path: Path to agent config file (agent.yaml) - optional, will try to find in checkpoint dir
        action_dim: Expected action dimension (typically 12)
        obs_dim: Expected observation dimension
    """

    checkpoint_path: str
    config_path: Optional[str] = None
    action_dim: int = 12
    obs_dim: int = 753  # num_prop(53) + num_scan(132) + num_priv_explicit(9) + num_priv_latent(29) + history(530)


class MinimalVecEnvStub(VecEnv):
    """Minimal VecEnv stub for OnPolicyRunnerWithExtractor initialization.

    This stub provides the minimal interface needed by OnPolicyRunnerWithExtractor
    without requiring a full environment. It's used only during initialization
    to get observation/action dimensions.
    """

    def __init__(self, num_obs: int, num_actions: int, device: str = "cpu"):
        """Initialize minimal VecEnv stub.

        Args:
            num_obs: Observation dimension
            num_actions: Action dimension
            device: Device string ("cpu" or "cuda")
        """
        self.num_envs = 1
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.max_episode_length = 1000
        self.unwrapped = self  # For compatibility with OnPolicyRunnerWithExtractor which accesses env.unwrapped.step_dt
        self.unwrapped.step_dt = 0.01  # Default step dt

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return dummy observations for initialization.

        Returns:
            Tuple of (observation tensor, extras dict)
        """
        obs = torch.zeros(1, self.num_obs, device=self.device)
        extras = {
            "observations": {
                "policy": obs,
            }
        }
        return obs, extras

    def seed(self, seed: int = -1) -> int:
        """Seed the environment (stub)."""
        return seed

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment (stub)."""
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment (stub)."""
        obs, extras = self.get_observations()
        rew = torch.zeros(1, device=self.device)
        dones = torch.zeros(1, dtype=torch.long, device=self.device)
        return obs, rew, dones, extras

    def close(self):
        """Close the environment (stub)."""
        pass


class ParkourPolicyModel:
    """Parkour policy inference interface.

    Uses OnPolicyRunnerWithExtractor to load checkpoints and get the inference policy
    function (act_inference) directly. This is NOT a wrapper - it's a minimal coordinator
    that handles numpy→torch conversion and calls the policy function directly.

    The observation normalizer is automatically applied by the policy function returned
    by get_inference_policy() if the model was trained with empirical_normalization=True.
    This is determined by the training config, not optional at inference time.
    """

    def __init__(
        self,
        weights: ModelWeights,
        train_cfg: Optional[dict] = None,
        device: str = "cuda",
    ):
        """Initialize policy model using OnPolicyRunnerWithExtractor.

        Args:
            weights: Model weights configuration
            train_cfg: Training config dict (optional, will try to load from checkpoint dir)
            device: Device to run inference on ("cuda" or "cpu")

        Raises:
            FileNotFoundError: If checkpoint file not found
            ValueError: If checkpoint loading fails
            ImportError: If OnPolicyRunnerWithExtractor cannot be imported
        """
        self.weights = weights
        self.device = torch.device(device)
        self.action_dim = weights.action_dim
        self.obs_dim = weights.obs_dim

        # Load checkpoint path
        checkpoint_path = Path(weights.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Try to load config from checkpoint directory if not provided
        if train_cfg is None:
            train_cfg = self._load_config_from_checkpoint_dir(checkpoint_path, weights.config_path)

        # Create minimal VecEnv stub for initialization
        stub_env = MinimalVecEnvStub(
            num_obs=self.obs_dim,
            num_actions=self.action_dim,
            device=str(self.device),
        )

        # Initialize OnPolicyRunnerWithExtractor
        logger.info(f"Initializing OnPolicyRunnerWithExtractor with checkpoint: {checkpoint_path}")
        try:
            self.runner = OnPolicyRunnerWithExtractor(
                env=stub_env,
                train_cfg=train_cfg,
                log_dir=None,
                device=str(self.device),
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OnPolicyRunnerWithExtractor: {e}") from e

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            self.runner.load(str(checkpoint_path), load_optimizer=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        # Get inference policy function (returns act_inference function directly)
        # IMPORTANT: The observation normalizer is NOT optional - it's automatically
        # applied based on the training config. If empirical_normalization=True in the
        # training config, get_inference_policy() returns a lambda that applies normalization.
        # If False, it returns act_inference directly. We always use whatever the model
        # was trained with - this ensures correctness.
        logger.info("Getting inference policy function")
        self.policy_fn = self.runner.get_inference_policy(device=str(self.device))

        logger.info(
            f"Policy model loaded successfully. Action dim: {self.action_dim}, "
            f"Obs dim: {self.obs_dim}"
        )

    def _load_config_from_checkpoint_dir(
        self, checkpoint_path: Path, config_path: Optional[str]
    ) -> dict:
        """Load training config from checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint file
            config_path: Optional explicit config path

        Returns:
            Training config dict

        Raises:
            FileNotFoundError: If config file not found
        """
        # Try explicit config path first
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_file}: {e}")

        # Try to find config in checkpoint directory
        checkpoint_dir = checkpoint_path.parent
        config_file = checkpoint_dir / "params" / "agent.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")

        # Fallback: create minimal config dict
        logger.warning(
            f"Config file not found, using minimal config. "
            f"Expected at: {checkpoint_dir / 'params' / 'agent.yaml'}"
        )
        return self._create_minimal_config()

    def _create_minimal_config(self) -> dict:
        """Create minimal config dict for OnPolicyRunnerWithExtractor.

        Returns:
            Minimal config dict
        """
        return {
            "algorithm": {
                "class_name": "PPOWithExtractor",
                "learning_rate": 2.0e-4,
                "dagger_update_freq": 20,
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "desired_kl": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 16,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "max_grad_norm": 1.0,
                "priv_reg_coef_schedual": [0.0, 0.1, 2000.0, 3000.0],
                "rnd_cfg": None,
                "symmetry_cfg": None,
            },
            "estimator": {
                "class_name": "DefaultEstimator",
                "num_prop": 53,
                "num_scan": 132,
                "num_priv_explicit": 9,
                "num_priv_latent": 29,
                "hidden_dims": [128, 64],
                "learning_rate": 2.0e-4,
                "train_with_estimated_states": False,
            },
            "depth_encoder": None,
            "policy": {
                "class_name": "ActorCriticRMA",
                "num_prop": 53,
                "num_scan": 132,
                "num_priv_explicit": 9,
                "num_priv_latent": 29,
                "num_hist": 10,
                "actor_hidden_dims": [512, 256, 128],
                "critic_hidden_dims": [512, 256, 128],
                "scan_encoder_dims": [128, 64, 32],
                "priv_encoder_dims": [64, 20],
                "tanh_encoder_output": False,
                "actor": {
                    "class_name": "Actor",
                    "state_history_encoder": {
                        "class_name": "StateHistoryEncoder",
                        "channel_size": 10,
                    },
                },
            },
            "empirical_normalization": False,
            "num_steps_per_env": 24,
            "save_interval": 50,
        }

    def _build_observation_tensor(self, io: ParkourModelIO) -> torch.Tensor:
        """Build observation tensor from ParkourModelIO using zero-copy operations.

        The observation is already in the correct training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]

        Args:
            io: ParkourModelIO input with observation in training format

        Returns:
            Observation tensor of shape (1, OBS_DIM) for batch inference

        Raises:
            ValueError: If input is invalid or incomplete
        """
        if not io.is_complete():
            raise ValueError("ParkourModelIO is incomplete")

        if not io.is_synchronized():
            raise ValueError("ParkourModelIO components are not synchronized")

        # Get observation array (zero-copy view)
        obs_array = io.get_observation_array()

        # Validate shape
        if len(obs_array) != self.obs_dim:
            raise ValueError(f"Observation array length {len(obs_array)} != obs_dim {self.obs_dim}")

        # Convert to tensor - torch.from_numpy shares memory if array is C-contiguous and float32
        # Optimize: combine contiguous and dtype checks into single operation
        if not obs_array.flags["C_CONTIGUOUS"] or obs_array.dtype != np.float32:
            # Make contiguous and ensure float32 in one operation
            obs_array = np.ascontiguousarray(obs_array, dtype=np.float32)

        # Create tensor (shares memory with numpy array) and add batch dimension
        return torch.from_numpy(obs_array).to(self.device).unsqueeze(0)

    def inference(self, io: ParkourModelIO) -> InferenceResponse:
        """Run inference on ParkourModelIO.

        Uses the policy function directly from get_inference_policy() which returns
        act_inference. This ensures zero-copy operations - the tensor flows directly
        from the policy function.

        Args:
            io: ParkourModelIO input

        Returns:
            InferenceResponse with action tensor and metadata
        """
        input_timestamp_ns = time.time_ns()
        timing_breakdown = []

        try:
            # Build observation tensor (zero-copy numpy -> torch conversion)
            build_start_ns = time.time_ns()
            obs_tensor = self._build_observation_tensor(io)
            build_time_ms = (time.time_ns() - build_start_ns) / 1_000_000.0
            timing_breakdown.append(("build_observation_tensor", build_time_ms))

            # Call policy function directly (act_inference with hist_encoding=True)
            # Normalization is automatically applied based on training config
            policy_start_ns = time.time_ns()
            with torch.no_grad():
                action_tensor = self.policy_fn(obs_tensor, hist_encoding=True)
            policy_time_ms = (time.time_ns() - policy_start_ns) / 1_000_000.0
            timing_breakdown.append(("policy_inference", policy_time_ms))

            # Validate action shape - simplified check
            validate_start_ns = time.time_ns()
            if action_tensor.ndim == 2:
                expected_shape = (action_tensor.shape[0], self.action_dim)
                if action_tensor.shape[1] != self.action_dim:
                    raise ValueError(f"Action shape {action_tensor.shape} != {expected_shape}")
            elif action_tensor.ndim == 1:
                if len(action_tensor) != self.action_dim:
                    raise ValueError(f"Action length {len(action_tensor)} != {self.action_dim}")
            else:
                raise ValueError(f"Action must be 1D or 2D tensor, got {action_tensor.ndim}D with shape {action_tensor.shape}")

            # Ensure float32 (convert only if necessary)
            if action_tensor.dtype != torch.float32:
                action_tensor = action_tensor.to(torch.float32)
            validate_time_ms = (time.time_ns() - validate_start_ns) / 1_000_000.0
            timing_breakdown.append(("validate_and_convert", validate_time_ms))

            # Return action tensor directly (zero-copy reference from act_inference)
            return InferenceResponse.create_success(
                action=action_tensor,
                timing_breakdown=timing_breakdown,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return InferenceResponse.create_failure(
                error_message=str(e),
            )

