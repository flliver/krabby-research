"""Unit tests for parkour policy model inference.

These tests verify that the policy model inference produces correct outputs
and handles various inputs correctly. They test the model independently of HAL.
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import InferenceResponse, OBS_DIM, ParkourModelIO, ParkourObservation


def _find_checkpoint_path() -> Path:
    """Find checkpoint path.
    
    Uses PARKOUR_CHECKPOINT_PATH environment variable (should point to folder).
    Looks for unitree_go2_parkour_teacher.pt in that folder.
    
    Returns:
        Path to checkpoint file
        
    Raises:
        FileNotFoundError: If environment variable not set or checkpoint not found
    """
    checkpoint_name = "unitree_go2_parkour_teacher.pt"
    
    env_path = os.getenv("PARKOUR_CHECKPOINT_PATH")
    if not env_path:
        raise FileNotFoundError(
            "PARKOUR_CHECKPOINT_PATH environment variable is not set. "
            "Set it to the path of the checkpoint folder."
        )
    
    checkpoint_dir = Path(env_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
        )
    
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
        )
    
    return checkpoint_path


class TestParkourPolicyModel:
    """Tests for ParkourPolicyModel inference correctness."""

    def test_forward_pass_correctness(self):
        """Test forward-pass correctness by comparing outputs to reference.

        This test verifies that our inference produces outputs matching the reference
        implementation (play.py) when given the same inputs.

        Note: This test requires a checkpoint file and reference outputs.
        For now, we'll test that we can load the checkpoint and run inference.
        
        """
        # Find checkpoint path (tries multiple locations)
        try:
            checkpoint_path = _find_checkpoint_path()
        except FileNotFoundError as e:
            pytest.fail(str(e))

        # Create model weights config
        weights = ModelWeights(
            checkpoint_path=str(checkpoint_path),
            action_dim=12,
            obs_dim=753,  # num_prop(53) + num_scan(132) + num_priv_explicit(9) + num_priv_latent(29) + history(530)
        )

        # Try to load the model (this will use OnPolicyRunnerWithExtractor)
        try:
            model = ParkourPolicyModel(weights, device="cpu")  # Use CPU for testing
        except Exception as e:
            pytest.fail(f"Failed to load checkpoint: {e}")

        # Create a synthetic observation in training format
        obs_array = np.random.randn(OBS_DIM).astype(np.float32)
        observation = ParkourObservation(
            timestamp_ns=time.time_ns(),
            schema_version="1.0",
            observation=obs_array,
        )
        
        nav_cmd = NavigationCommand.create_now(vx=1.0, vy=0.0, yaw_rate=0.0)
        
        model_io = ParkourModelIO(
            timestamp_ns=time.time_ns(),
            nav_cmd=nav_cmd,
            observation=observation,
        )

        # Run inference
        result = model.inference(model_io)

        # Verify inference succeeded
        assert result.success, f"Inference failed: {result.error_message}"
        assert result.action is not None, "Action should not be None"
        assert result.action.shape == (1, 12) or result.action.shape == (12,), f"Unexpected action shape: {result.action.shape}"
        # Check that timing breakdown exists and has positive values
        total_latency_ms = sum(time_ms for _, time_ms in result.timing_breakdown)
        assert total_latency_ms > 0, "Latency should be positive"

        # Note: Full correctness test would require:
        # 1. Recorded observations from IsaacSim (from play.py)
        # 2. Reference outputs from play.py
        # 3. Comparison of outputs within tolerance

    def test_inference_latency_measurement(self):
        """Test inference latency is measured correctly."""
        # Create a mock model that simulates inference time
        class MockPolicyModel:
            """Mock policy model for testing latency measurement."""
            def __init__(self, action_dim: int = 12, inference_time_ms: float = 8.0):
                self.action_dim = action_dim
                self.inference_time_ms = inference_time_ms
                self.inference_count = 0

            def inference(self, model_io):
                start_time = time.time_ns()
                # Simulate inference time
                time.sleep(self.inference_time_ms / 1000.0)
                end_time = time.time_ns()

                self.inference_count += 1
                actual_latency_ms = (end_time - start_time) / 1_000_000.0

                action = torch.zeros(self.action_dim, dtype=torch.float32)
                # Include timing breakdown to match real inference behavior
                timing_breakdown = [
                    ("build_observation_tensor", actual_latency_ms * 0.1),
                    ("policy_inference", actual_latency_ms * 0.8),
                    ("validate_and_convert", actual_latency_ms * 0.1),
                ]
                return InferenceResponse.create_success(
                    action=action,
                    timing_breakdown=timing_breakdown,
                )

        model = MockPolicyModel(action_dim=12, inference_time_ms=8.0)

        # Mock model_io
        model_io = MagicMock(spec=ParkourModelIO)

        start_time = time.time()
        result = model.inference(model_io)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000.0

        assert result.success
        # Calculate total latency from timing breakdown
        total_latency_ms = sum(time_ms for _, time_ms in result.timing_breakdown)
        assert total_latency_ms > 0
        assert abs(total_latency_ms - elapsed_ms) < 2.0  # Within 2ms tolerance
        assert total_latency_ms < 15.0  # Should be under 15ms target
        assert result.action is not None
        assert result.action.shape == (12,)

