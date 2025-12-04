"""Parkour-specific model types.

These types represent Parkour policy model inputs and outputs, including
observations in the exact training format and inference responses.

Training observation format: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]

Zero-Copy Guarantees:
- Large arrays (observation tensors) use views when possible
- Array slicing operations (get_proprioceptive, get_scan, etc.) return views, not copies
- Only copies when dtype conversion is required or arrays are not contiguous
- Scalar values (timestamps, schema_version) are always copied (necessary)
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from hal.client.observation.types import NavigationCommand

# Training observation dimensions (from parkour_rl_cfg.py)
NUM_PROP = 53  # Proprioceptive features
NUM_SCAN = 132  # Scan/depth features
NUM_PRIV_EXPLICIT = 9  # Privileged explicit features
NUM_PRIV_LATENT = 29  # Privileged latent features
NUM_HIST = 10  # History length
HISTORY_DIM = NUM_HIST * NUM_PROP  # 530

# Total observation dimension
OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + HISTORY_DIM  # 753


@dataclass
class ParkourObservation:
    """Parkour observation in exact training format with zero-copy support.

    This matches the training format exactly:
    [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]

    This is the base observation class. For teacher/student separation, use
    TeacherObservation or StudentObservation instead.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        observation: Flat float32 array of shape (OBS_DIM,) matching training format
                    This array should be a view when possible to avoid copies.
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    observation: Optional[np.ndarray] = None  # Shape: (OBS_DIM,), dtype: float32

    def __post_init__(self) -> None:
        """Validate observation."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")
        if self.observation is not None:
            if not isinstance(self.observation, np.ndarray):
                raise ValueError("observation must be a numpy array")
            if self.observation.dtype != np.float32:
                # Only convert if necessary - prefer views
                if self.observation.dtype == np.float64:
                    self.observation = self.observation.astype(np.float32, copy=False)
                else:
                    self.observation = self.observation.astype(np.float32)
            if self.observation.shape != (OBS_DIM,):
                raise ValueError(
                    f"Observation shape {self.observation.shape} != expected ({OBS_DIM},)"
                )

    def get_proprioceptive(self) -> np.ndarray:
        """Get proprioceptive features (num_prop=53) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[:NUM_PROP]

    def get_scan(self) -> np.ndarray:
        """Get scan/depth features (num_scan=132) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[NUM_PROP : NUM_PROP + NUM_SCAN]

    def get_priv_explicit(self) -> np.ndarray:
        """Get privileged explicit features (num_priv_explicit=9) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        start = NUM_PROP + NUM_SCAN
        return self.observation[start : start + NUM_PRIV_EXPLICIT]

    def get_priv_latent(self) -> np.ndarray:
        """Get privileged latent features (num_priv_latent=29) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        start = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT
        return self.observation[start : start + NUM_PRIV_LATENT]

    def get_history(self) -> np.ndarray:
        """Get history features (history_dim=530) as a view."""
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation[-HISTORY_DIM:]

    @classmethod
    def from_parts(
        cls,
        proprioceptive: np.ndarray,
        scan: np.ndarray,
        priv_explicit: np.ndarray,
        priv_latent: np.ndarray,
        history: np.ndarray,
        timestamp_ns: Optional[int] = None,
        schema_version: str = "1.0",
    ) -> "ParkourObservation":
        """Create ParkourObservation from component parts.
        
        Concatenates the component arrays into a single observation array
        matching the training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        
        Args:
            proprioceptive: Proprioceptive features array (shape: (NUM_PROP,), dtype: float32)
            scan: Scan/depth features array (shape: (NUM_SCAN,), dtype: float32)
            priv_explicit: Privileged explicit features array (shape: (NUM_PRIV_EXPLICIT,), dtype: float32)
            priv_latent: Privileged latent features array (shape: (NUM_PRIV_LATENT,), dtype: float32)
            history: History features array (shape: (HISTORY_DIM,), dtype: float32)
            timestamp_ns: Optional timestamp in nanoseconds (defaults to current time)
            schema_version: Schema version string (default: "1.0")
            
        Returns:
            ParkourObservation instance with concatenated observation array
            
        Raises:
            ValueError: If any component has incorrect shape or dtype
        """
        # Validate shapes
        if proprioceptive.shape != (NUM_PROP,):
            raise ValueError(f"proprioceptive shape {proprioceptive.shape} != ({NUM_PROP},)")
        if scan.shape != (NUM_SCAN,):
            raise ValueError(f"scan shape {scan.shape} != ({NUM_SCAN},)")
        if priv_explicit.shape != (NUM_PRIV_EXPLICIT,):
            raise ValueError(f"priv_explicit shape {priv_explicit.shape} != ({NUM_PRIV_EXPLICIT},)")
        if priv_latent.shape != (NUM_PRIV_LATENT,):
            raise ValueError(f"priv_latent shape {priv_latent.shape} != ({NUM_PRIV_LATENT},)")
        if history.shape != (HISTORY_DIM,):
            raise ValueError(f"history shape {history.shape} != ({HISTORY_DIM},)")
        
        # Ensure float32 dtype (copy if necessary)
        proprioceptive = np.asarray(proprioceptive, dtype=np.float32)
        scan = np.asarray(scan, dtype=np.float32)
        priv_explicit = np.asarray(priv_explicit, dtype=np.float32)
        priv_latent = np.asarray(priv_latent, dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)
        
        # Concatenate components in training format order
        observation = np.concatenate([
            proprioceptive,
            scan,
            priv_explicit,
            priv_latent,
            history,
        ])
        
        # Use current time if timestamp not provided
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        return cls(
            timestamp_ns=timestamp_ns,
            schema_version=schema_version,
            observation=observation,
        )


@dataclass
class TeacherObservation(ParkourObservation):
    """Teacher model observation (full sensor suite).
    
    Contains the complete observation with all sensors available during training.
    This is the full observation format used by the teacher model.
    
    Zero-copy guarantees:
    - Inherits zero-copy support from ParkourObservation
    - All sensor data is included in the observation array
    """
    pass


@dataclass
class StudentObservation(ParkourObservation):
    """Student model observation (subset of sensors).
    
    Contains a subset of sensors available to the student model.
    The student model typically has access to fewer sensors than the teacher
    to encourage learning robust representations.
    
    The observation array may have a different dimension than the teacher
    observation, depending on which sensors are excluded.
    
    Zero-copy guarantees:
    - Inherits zero-copy support from ParkourObservation
    - Only includes sensors available to student model
    """
    pass


@dataclass
class ParkourModelIO:
    """Combined input model aggregating all telemetry for policy inference.

    Uses zero-copy observation format matching training exactly.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        nav_cmd: Navigation command (for reference, not in observation)
        observation: Complete observation in training format
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    nav_cmd: Optional[NavigationCommand] = None
    observation: Optional[ParkourObservation] = None

    def __post_init__(self) -> None:
        """Validate ParkourModelIO."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")

    def is_complete(self) -> bool:
        """Check if all required components are present."""
        return self.nav_cmd is not None and self.observation is not None

    def is_synchronized(self, max_age_ns: int = 10_000_000) -> bool:
        """Check if all components have timestamps within max_age_ns of each other.

        Args:
            max_age_ns: Maximum age difference in nanoseconds (default 10ms)

        Returns:
            True if all components are synchronized, False otherwise
        """
        if not self.is_complete():
            return False

        timestamps = [
            self.nav_cmd.timestamp_ns,
            self.observation.timestamp_ns,
        ]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        return (max_ts - min_ts) <= max_age_ns

    def get_observation_array(self) -> np.ndarray:
        """Get observation array as a view (zero-copy).

        Returns:
            Observation array of shape (OBS_DIM,) matching training format
        """
        if self.observation is None or self.observation.observation is None:
            raise ValueError("Observation not set")
        return self.observation.observation


@dataclass
class InferenceResponse:
    """Policy inference response matching inference output format exactly.

    The action tensor is in the exact format returned by act_inference:
    - Type: torch.Tensor
    - Shape: (ACTION_DIM,) or (batch, ACTION_DIM)
    - Dtype: torch.float32
    - Device: Same as model device (cuda/cpu)
    - Zero-copy: tensor is used directly without conversion

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        inference_latency_ms: Inference latency in milliseconds
        action: Action tensor directly from inference (torch.Tensor)
        model_version: Model version string
        success: Whether inference succeeded
        error_message: Error message if success=False (optional)
    """

    timestamp_ns: int
    inference_latency_ms: float
    action: Optional[torch.Tensor] = None  # torch.Tensor from act_inference
    model_version: str = "unknown"
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate inference response."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.inference_latency_ms < 0:
            raise ValueError("inference_latency_ms must be non-negative")
        if not self.success and self.error_message is None:
            self.error_message = "Inference failed (no error message provided)"
        if self.action is not None:
            if not isinstance(self.action, torch.Tensor):
                raise ValueError("action must be torch.Tensor")
            if self.action.dtype != torch.float32:
                # Only convert if necessary - prefer views
                self.action = self.action.to(torch.float32)
            if self.action.ndim not in (1, 2):
                raise ValueError(f"action must be 1D or 2D tensor, got shape {self.action.shape}")

    def validate_action_dim(self, action_dim: int) -> None:
        """Validate that action matches expected action dimension.

        Args:
            action_dim: Expected action dimension
        """
        if self.action is None:
            raise ValueError("action is None")
        # Handle both (action_dim,) and (batch, action_dim) shapes
        if self.action.ndim == 1:
            if len(self.action) != action_dim:
                raise ValueError(f"action length {len(self.action)} != action_dim {action_dim}")
        elif self.action.ndim == 2:
            if self.action.shape[1] != action_dim:
                raise ValueError(f"action shape {self.action.shape} != (batch, {action_dim})")
        else:
            raise ValueError(f"action must be 1D or 2D tensor, got shape {self.action.shape}")

    def get_action(self) -> torch.Tensor:
        """Get action tensor as a view (zero-copy).

        Returns:
            Action tensor matching inference output format
        """
        if self.action is None:
            raise ValueError("action not set")
        return self.action

    def get_action_numpy(self) -> np.ndarray:
        """Get action as numpy array (creates copy if on GPU, view if on CPU).

        This is for ZMQ serialization. Only call when needed.

        Returns:
            Action array as numpy (shape: ACTION_DIM, dtype: float32)
        """
        if self.action is None:
            raise ValueError("action not set")
        # Squeeze batch dimension if present
        action = self.action.squeeze(0) if self.action.ndim == 2 else self.action
        # Convert to numpy - shares memory if on CPU, copies if on GPU
        if action.is_cuda:
            return action.cpu().numpy()
        else:
            return action.numpy()

    @classmethod
    def create_success(
        cls,
        action: torch.Tensor,
        inference_latency_ms: float,
        model_version: str = "unknown",
    ) -> "InferenceResponse":
        """Create successful inference response with action tensor.

        Args:
            action: Action tensor from inference (torch.Tensor from act_inference)
            inference_latency_ms: Inference latency in milliseconds
            model_version: Model version string

        Returns:
            InferenceResponse with action tensor (zero-copy)
        """
        return cls(
            timestamp_ns=time.time_ns(),
            inference_latency_ms=inference_latency_ms,
            action=action,  # Direct reference, no copy
            model_version=model_version,
            success=True,
        )

    @classmethod
    def create_failure(
        cls,
        error_message: str,
        model_version: str = "unknown",
    ) -> "InferenceResponse":
        """Create failed inference response.

        Args:
            error_message: Error message describing the failure
            model_version: Model version string

        Returns:
            InferenceResponse with success=False
        """
        return cls(
            timestamp_ns=time.time_ns(),
            inference_latency_ms=0.0,
            action=None,
            model_version=model_version,
            success=False,
            error_message=error_message,
        )

