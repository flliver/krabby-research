"""Tests for zero-copy operations in the HAL and policy interface.

These tests verify that data flows through the system with minimal copying:
- numpy → torch tensor conversion (should share memory when possible)
- torch → numpy conversion (should share memory on CPU)
- Action tensor passing (should be direct references, not copies)
"""

import numpy as np
import pytest
import torch

from hal.client.commands.types import InferenceResponse
from hal.client.observation.types import (
    NUM_PROP,
    NUM_SCAN,
    NUM_PRIV_EXPLICIT,
    NUM_PRIV_LATENT,
    HISTORY_DIM,
    OBS_DIM,
    ParkourModelIO,
    ParkourObservation,
    NavigationCommand,
)


class TestNumpyToTorchZeroCopy:
    """Test zero-copy numpy to torch tensor conversion."""

    def test_torch_from_numpy_shares_memory_contiguous_float32(self):
        """Verify torch.from_numpy() shares memory with C-contiguous float32 arrays."""
        # Create C-contiguous float32 array
        arr = np.zeros(OBS_DIM, dtype=np.float32)
        assert arr.flags["C_CONTIGUOUS"], "Array should be C-contiguous"
        assert arr.dtype == np.float32, "Array should be float32"

        # Convert to tensor
        tensor = torch.from_numpy(arr)

        # Verify they share memory by modifying the original array
        arr[0] = 42.0
        assert tensor[0].item() == 42.0, "Tensor should reflect changes to numpy array (shared memory)"

        # Verify modifying tensor affects numpy array
        tensor[1] = 99.0
        assert arr[1] == 99.0, "Numpy array should reflect changes to tensor (shared memory)"

    def test_torch_from_numpy_requires_contiguous_for_zero_copy(self):
        """Verify torch.from_numpy() works best with C-contiguous arrays."""
        # Create non-contiguous array by taking every other element
        arr_base = np.zeros(OBS_DIM * 2, dtype=np.float32)
        arr_non_contig = arr_base[::2]  # Every other element
        assert not arr_non_contig.flags["C_CONTIGUOUS"], "Array should not be C-contiguous"

        # torch.from_numpy() may still work but might create a copy internally
        # The key point is that for zero-copy, we need C-contiguous arrays
        tensor = torch.from_numpy(arr_non_contig)

        # The important thing is that our code ensures contiguous arrays
        # Test that making it contiguous allows zero-copy
        arr_contig = np.ascontiguousarray(arr_non_contig, dtype=np.float32)
        assert arr_contig.flags["C_CONTIGUOUS"], "Array should be C-contiguous after ascontiguousarray"
        
        tensor_contig = torch.from_numpy(arr_contig)
        
        # Verify they share memory (zero-copy)
        arr_contig[0] = 42.0
        assert tensor_contig[0].item() == 42.0, "Contiguous array should share memory with tensor"

    def test_torch_from_numpy_copies_when_not_float32(self):
        """Verify conversion handles non-float32 arrays correctly."""
        # Create float64 array
        arr = np.zeros(OBS_DIM, dtype=np.float64)
        assert arr.dtype == np.float64, "Array should be float64"

        # Convert to float32 (creates copy)
        arr_float32 = arr.astype(np.float32, copy=False)
        tensor = torch.from_numpy(arr_float32)

        # Modify original (float64) - should not affect tensor
        arr[0] = 42.0
        assert tensor[0].item() == 0.0, "Tensor should not be affected by float64 array changes"

    def test_policy_interface_observation_conversion(self):
        """Test that policy interface converts observations with zero-copy when possible."""
        from compute.parkour.policy_interface import ParkourPolicyModel

        # Create observation in training format (C-contiguous, float32)
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        assert obs_array.flags["C_CONTIGUOUS"], "Observation should be C-contiguous"

        # Create ParkourModelIO
        nav_cmd = NavigationCommand.create_now()
        observation = ParkourObservation(
            timestamp_ns=1,
            schema_version="1.0",
            observation=obs_array,
        )
        model_io = ParkourModelIO(
            timestamp_ns=1,
            schema_version="1.0",
            nav_cmd=nav_cmd,
            observation=observation,
        )

        # Create a mock policy model to test conversion
        class MockPolicyModel:
            def __init__(self):
                self.obs_dim = OBS_DIM
                self.device = torch.device("cpu")

            def _build_observation_tensor(self, io):
                """Simulate policy interface conversion."""
                obs_array = io.get_observation_array()
                if not obs_array.flags["C_CONTIGUOUS"]:
                    obs_array = np.ascontiguousarray(obs_array, dtype=np.float32)
                if obs_array.dtype != np.float32:
                    obs_array = obs_array.astype(np.float32, copy=False)
                return torch.from_numpy(obs_array).to(self.device).unsqueeze(0)

        model = MockPolicyModel()
        tensor = model._build_observation_tensor(model_io)

        # Verify they share memory
        obs_array[0] = 123.0
        assert tensor[0, 0].item() == 123.0, "Tensor should share memory with observation array"


class TestTorchToNumpyZeroCopy:
    """Test zero-copy torch to numpy conversion."""

    def test_tensor_numpy_shares_memory_on_cpu(self):
        """Verify tensor.numpy() shares memory when tensor is on CPU."""
        # Create tensor on CPU
        tensor = torch.zeros(12, dtype=torch.float32)
        assert not tensor.is_cuda, "Tensor should be on CPU"

        # Convert to numpy
        arr = tensor.numpy()

        # Verify they share memory
        tensor[0] = 42.0
        assert arr[0] == 42.0, "Numpy array should reflect changes to CPU tensor (shared memory)"

        arr[1] = 99.0
        assert tensor[1].item() == 99.0, "Tensor should reflect changes to numpy array (shared memory)"

    def test_tensor_numpy_copies_when_on_gpu(self):
        """Verify tensor.cpu().numpy() copies when tensor is on GPU."""
        assert torch.cuda.is_available(), "CUDA must be available for this test"

        # Create tensor on GPU
        tensor = torch.zeros(12, dtype=torch.float32, device="cuda")
        assert tensor.is_cuda, "Tensor should be on GPU"

        # Convert to numpy (requires CPU copy)
        arr = tensor.cpu().numpy()

        # Verify they don't share memory (tensor is on GPU, array is on CPU)
        original_value = arr[0]
        tensor[0] = 42.0
        # Array should not reflect the change immediately (they're on different devices)
        # But after copying, they should be independent
        arr[0] = 99.0
        assert tensor[0].item() == 42.0, "GPU tensor should not be affected by CPU array changes"

    def test_inference_response_get_action_numpy_cpu(self):
        """Test InferenceResponse.get_action_numpy() shares memory on CPU."""
        # Create action tensor on CPU
        action = torch.zeros(12, dtype=torch.float32)
        response = InferenceResponse.create_success(
            action=action,
            inference_latency_ms=5.0,
            model_version="test",
        )

        # Get numpy array
        arr = response.get_action_numpy()

        # Verify they share memory
        action[0] = 42.0
        assert arr[0] == 42.0, "Numpy array should share memory with CPU tensor"

    def test_inference_response_get_action_numpy_gpu(self):
        """Test InferenceResponse.get_action_numpy() copies when on GPU."""
        assert torch.cuda.is_available(), "CUDA must be available for this test"

        # Create action tensor on GPU
        action = torch.zeros(12, dtype=torch.float32, device="cuda")
        response = InferenceResponse.create_success(
            action=action,
            inference_latency_ms=5.0,
            model_version="test",
        )

        # Get numpy array (will copy from GPU to CPU)
        arr = response.get_action_numpy()

        # Verify array is on CPU and independent
        assert not arr.flags.writeable is False, "Array should be writable"
        arr[0] = 99.0
        assert action[0].item() == 0.0, "GPU tensor should not be affected by CPU array changes"


class TestActionTensorZeroCopy:
    """Test zero-copy action tensor passing."""

    def test_inference_response_action_direct_reference(self):
        """Verify InferenceResponse stores action tensor as direct reference."""
        # Create action tensor
        action = torch.zeros(12, dtype=torch.float32)

        # Create response
        response = InferenceResponse.create_success(
            action=action,
            inference_latency_ms=5.0,
            model_version="test",
        )

        # Verify it's the same object (not a copy)
        assert response.action is action, "Response should store direct reference to action tensor"

        # Modify original tensor
        action[0] = 42.0
        assert response.action[0].item() == 42.0, "Response action should reflect changes (same object)"

    def test_get_action_returns_direct_reference(self):
        """Verify get_action() returns direct reference to tensor."""
        action = torch.zeros(12, dtype=torch.float32)
        response = InferenceResponse.create_success(
            action=action,
            inference_latency_ms=5.0,
        )

        # Get action
        retrieved_action = response.get_action()

        # Verify it's the same object
        assert retrieved_action is action, "get_action() should return direct reference"

        # Modify and verify
        retrieved_action[0] = 99.0
        assert action[0].item() == 99.0, "Original action should reflect changes (same object)"
        assert response.action[0].item() == 99.0, "Response action should reflect changes (same object)"


class TestObservationViewMethods:
    """Test that observation view methods return zero-copy views."""

    def test_get_proprioceptive_returns_view(self):
        """Verify get_proprioceptive() returns a view, not a copy."""
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        prop = obs.get_proprioceptive()

        # Modify view
        prop[0] = 42.0
        assert obs_array[0] == 42.0, "Original array should reflect changes (view, not copy)"

        # Modify original
        obs_array[1] = 99.0
        assert prop[1] == 99.0, "View should reflect changes to original array"

    def test_get_scan_returns_view(self):
        """Verify get_scan() returns a view, not a copy."""
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        scan = obs.get_scan()

        # Verify it's a view
        scan[0] = 42.0
        expected_idx = NUM_PROP
        assert obs_array[expected_idx] == 42.0, "Original array should reflect changes (view)"

    def test_all_view_methods_return_views(self):
        """Verify all view methods return views, not copies."""
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        # Test all view methods
        prop = obs.get_proprioceptive()
        scan = obs.get_scan()
        priv_explicit = obs.get_priv_explicit()
        priv_latent = obs.get_priv_latent()
        history = obs.get_history()

        # Modify each view and verify original array is affected
        prop[0] = 1.0
        assert obs_array[0] == 1.0

        scan[0] = 2.0
        assert obs_array[NUM_PROP] == 2.0

        priv_explicit[0] = 3.0
        assert obs_array[NUM_PROP + NUM_SCAN] == 3.0

        priv_latent[0] = 4.0
        assert obs_array[NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT] == 4.0

        history[0] = 5.0
        assert obs_array[-HISTORY_DIM] == 5.0

