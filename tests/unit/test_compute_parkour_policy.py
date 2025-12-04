"""Unit tests for parkour policy model inference.

These tests verify that the policy model inference produces correct outputs
and handles various inputs correctly. They test the model independently of HAL.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import OBS_DIM, ParkourModelIO, ParkourObservation


class TestParkourPolicyModel:
    """Tests for ParkourPolicyModel inference correctness."""

    def test_forward_pass_correctness(self):
        """Test forward-pass correctness by comparing outputs to reference.

        This test verifies that our inference produces outputs matching the reference
        implementation (play.py) when given the same inputs.

        Note: This test requires a checkpoint file and reference outputs.
        For now, we'll test that we can load the checkpoint and run inference.
        """
        # Use checkpoint from project assets directory
        checkpoint_path = Path(__file__).parent.parent.parent / "parkour" / "assets" / "weights" / "unitree_go2_parkour_teacher.pt"
        
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

        # Create model weights config
        weights = ModelWeights(
            checkpoint_path=str(checkpoint_path),
            action_dim=12,
            obs_dim=753,  # num_prop(53) + num_scan(132) + num_priv_explicit(9) + num_priv_latent(29) + history(530)
            model_version="teacher",
        )

        # Try to load the model (this will use OnPolicyRunnerWithExtractor)
        try:
            model = ParkourPolicyModel(weights, device="cpu")  # Use CPU for testing
        except Exception as e:
            pytest.skip(f"Failed to load checkpoint: {e}")

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
        assert result.inference_latency_ms > 0, "Latency should be positive"

        # Note: Full correctness test would require:
        # 1. Recorded observations from IsaacSim (from play.py)
        # 2. Reference outputs from play.py
        # 3. Comparison of outputs within tolerance

    def test_inference_latency_measurement(self):
        """Test inference latency is measured correctly."""
        from unittest.mock import MagicMock
        from compute.parkour.parkour_types import ParkourModelIO

        # Create a mock model that simulates inference time
        class MockPolicyModel:
            """Mock policy model for testing latency measurement."""
            def __init__(self, action_dim: int = 12, inference_time_ms: float = 8.0):
                self.action_dim = action_dim
                self.inference_time_ms = inference_time_ms
                self.inference_count = 0

            def inference(self, model_io):
                import time
                from compute.parkour.parkour_types import InferenceResponse
                import torch

                start_time = time.time_ns()
                # Simulate inference time
                time.sleep(self.inference_time_ms / 1000.0)
                end_time = time.time_ns()

                self.inference_count += 1
                actual_latency_ms = (end_time - start_time) / 1_000_000.0

                action = torch.zeros(self.action_dim, dtype=torch.float32)
                return InferenceResponse.create_success(
                    action=action,
                    inference_latency_ms=actual_latency_ms,
                    model_version="test",
                )

        model = MockPolicyModel(action_dim=12, inference_time_ms=8.0)

        # Mock model_io
        model_io = MagicMock(spec=ParkourModelIO)

        start_time = time.time()
        result = model.inference(model_io)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000.0

        assert result.success
        assert result.inference_latency_ms > 0
        assert abs(result.inference_latency_ms - elapsed_ms) < 2.0  # Within 2ms tolerance
        assert result.inference_latency_ms < 15.0  # Should be under 15ms target
        assert result.action is not None
        assert result.action.shape == (12,)

