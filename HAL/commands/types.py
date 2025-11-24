"""Command model types for HAL output data.

These models represent actuator commands and inference responses flowing from
the policy wrapper to hardware/simulation. Designed for zero-copy operations
matching inference output format exactly.

The inference methods (act_inference) return torch.Tensor directly, so command
types should match this format.
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch


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

    def get_action_numpy(self) -> "np.ndarray":
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


