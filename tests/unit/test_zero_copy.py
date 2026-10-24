"""Unit tests for zero-copy operations in the HAL and policy interface.

These tests verify that data flows through the system with minimal copying:
- numpy → torch tensor conversion (should share memory when possible)
- torch → numpy conversion (should share memory on CPU)
- Action tensor passing (should be direct references, not copies)

These are unit tests that test data conversion utilities in isolation,
without requiring HAL integration.
"""

import numpy as np
import pytest
import torch

from hal.client.observation.types import NavigationCommand
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from compute.parkour.parkour_types import InferenceResponse, ParkourModelIO, ParkourObservation
from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)

_TEST_ROBOT = RobotDefinition(
    name="test_robot",
    legs=("FL", "FR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)


@pytest.fixture
def observation_dimensions():
    return PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(_TEST_ROBOT)


class TestNumpyToTorchZeroCopy:
    """Test zero-copy numpy to torch tensor conversion."""

    def test_torch_from_numpy_shares_memory_contiguous_float32(self, observation_dimensions):
        """Verify torch.from_numpy() shares memory with C-contiguous float32 arrays."""
        arr = np.zeros(observation_dimensions.obs_dim, dtype=np.float32)
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

    def test_torch_from_numpy_requires_contiguous_for_zero_copy(self, observation_dimensions):
        """Verify torch.from_numpy() works best with C-contiguous arrays."""
        obs_dim = observation_dimensions.obs_dim
        arr_base = np.zeros(obs_dim * 2, dtype=np.float32)
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

    def test_torch_from_numpy_copies_when_not_float32(self, observation_dimensions):
        """Verify conversion handles non-float32 arrays correctly."""
        arr = np.zeros(observation_dimensions.obs_dim, dtype=np.float64)
        assert arr.dtype == np.float64, "Array should be float64"

        # Convert to float32 (creates copy)
        arr_float32 = arr.astype(np.float32, copy=False)
        tensor = torch.from_numpy(arr_float32)

        # Modify original (float64) - should not affect tensor
        arr[0] = 42.0
        assert tensor[0].item() == 0.0, "Tensor should not be affected by float64 array changes"

    def test_policy_interface_observation_conversion(self, observation_dimensions):
        """Test that policy interface converts observations (array built at model boundary)."""
        d = observation_dimensions
        obs_array = np.zeros(d.obs_dim, dtype=np.float32)
        obs_array[0] = 42.0

        nav_cmd = NavigationCommand.create_now()
        observation = ParkourObservation.from_array(d, obs_array, timestamp_ns=1)
        model_io = ParkourModelIO(
            timestamp_ns=1,
            nav_cmd=nav_cmd,
            observation=observation,
        )

        class MockPolicyModel:
            def __init__(self, obs_dim):
                self.obs_dim = obs_dim
                self.device = torch.device("cpu")

            def _build_observation_tensor(self, io):
                """Simulate policy interface conversion (to_array at model boundary)."""
                arr = io.get_observation_array()
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                return torch.from_numpy(arr).to(self.device).unsqueeze(0)

        model = MockPolicyModel(d.obs_dim)
        tensor = model._build_observation_tensor(model_io)
        assert tensor.shape == (1, d.obs_dim)
        assert tensor[0, 0].item() == 42.0, "Tensor should contain observation values"


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
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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
            timing_breakdown=[],
        )

        # Get numpy array
        arr = response.get_action_numpy()

        # Verify they share memory
        action[0] = 42.0
        assert arr[0] == 42.0, "Numpy array should share memory with CPU tensor"

    def test_inference_response_get_action_numpy_gpu(self):
        """Test InferenceResponse.get_action_numpy() copies when on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create action tensor on GPU
        action = torch.zeros(12, dtype=torch.float32, device="cuda")
        response = InferenceResponse.create_success(
            action=action,
            timing_breakdown=[],
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
            timing_breakdown=[],
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
            timing_breakdown=[],
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

    def test_get_proprioceptive_returns_stored_part(self, observation_dimensions):
        """Verify get_proprioceptive() returns the stored part (in-place mutable)."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[],
        )
        prop = obs.get_proprioceptive()
        prop[0] = 42.0
        assert obs.to_array()[0] == 42.0
        prop[1] = 99.0
        assert obs.get_proprioceptive()[1] == 99.0

    def test_get_scan_returns_stored_part(self, observation_dimensions):
        """Verify get_scan() returns the stored part."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[],
        )
        scan = obs.get_scan()
        scan[0] = 42.0
        assert obs.to_array()[d.num_prop] == 42.0

    def test_all_view_methods_return_stored_parts(self, observation_dimensions):
        """Verify all get_* methods return the stored part arrays (mutations appear in to_array())."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[],
        )
        prop = obs.get_proprioceptive()
        scan = obs.get_scan()
        priv_explicit = obs.get_priv_explicit()
        priv_latent = obs.get_priv_latent()
        history = obs.get_history()
        # Mutate via views; to_array() concatenates stored parts so must be called after mutations
        prop[0] = 1.0
        scan[0] = 2.0
        if d.num_vision > 0:
            vision_list = obs.get_vision()
            if vision_list:
                vision_list[0][0] = 2.5
        priv_explicit[0] = 3.0
        priv_latent[0] = 4.0
        history[0] = 5.0
        arr = obs.to_array()
        assert arr[0] == 1.0
        assert arr[d.num_prop] == 2.0
        if d.num_vision > 0:
            assert arr[d.num_prop + d.num_scan] == 2.5
        assert arr[d.num_prop + d.num_scan + d.num_vision] == 3.0
        assert arr[d.num_prop + d.num_scan + d.num_vision + d.num_priv_explicit] == 4.0
        assert arr[-d.history_dim] == 5.0

