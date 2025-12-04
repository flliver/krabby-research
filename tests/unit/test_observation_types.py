"""Unit tests for unified observation format validation.

These tests verify that the observation format matches the training format exactly:
- Total dimension (OBS_DIM = 753)
- Component dimensions (prop=53, scan=132, priv_explicit=9, priv_latent=29, history=530)
- Component positioning and ordering
- Data type (float32)
- Format consistency across different sources

These are unit tests that test the observation data structures in isolation,
without requiring HAL integration.
"""

import numpy as np
import pytest

from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import (
    NUM_PROP,
    NUM_SCAN,
    NUM_PRIV_EXPLICIT,
    NUM_PRIV_LATENT,
    NUM_HIST,
    HISTORY_DIM,
    OBS_DIM,
    ParkourObservation,
    ParkourModelIO,
)


class TestObservationDimensions:
    """Test observation dimension constants match training format."""

    def test_total_observation_dimension(self):
        """Verify total observation dimension is 753."""
        assert OBS_DIM == 753, f"OBS_DIM should be 753, got {OBS_DIM}"

    def test_component_dimensions(self):
        """Verify all component dimensions are correct."""
        assert NUM_PROP == 53, f"NUM_PROP should be 53, got {NUM_PROP}"
        assert NUM_SCAN == 132, f"NUM_SCAN should be 132, got {NUM_SCAN}"
        assert NUM_PRIV_EXPLICIT == 9, f"NUM_PRIV_EXPLICIT should be 9, got {NUM_PRIV_EXPLICIT}"
        assert NUM_PRIV_LATENT == 29, f"NUM_PRIV_LATENT should be 29, got {NUM_PRIV_LATENT}"
        assert NUM_HIST == 10, f"NUM_HIST should be 10, got {NUM_HIST}"
        assert HISTORY_DIM == 530, f"HISTORY_DIM should be 530, got {HISTORY_DIM}"

    def test_dimension_sum_matches_total(self):
        """Verify sum of component dimensions equals total dimension."""
        total = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + HISTORY_DIM
        assert total == OBS_DIM, \
            f"Component dimensions sum to {total}, but OBS_DIM is {OBS_DIM}"

    def test_history_dimension_calculation(self):
        """Verify history dimension is calculated correctly."""
        assert HISTORY_DIM == NUM_HIST * NUM_PROP, \
            f"HISTORY_DIM should be {NUM_HIST * NUM_PROP}, got {HISTORY_DIM}"


class TestObservationStructure:
    """Test observation structure and component positioning."""

    def test_observation_component_positions(self):
        """Verify each component is positioned correctly in the observation array."""
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        # Fill each component with distinct values
        obs_array[:NUM_PROP] = 1.0  # Proprioceptive
        obs_array[NUM_PROP : NUM_PROP + NUM_SCAN] = 2.0  # Scan
        start = NUM_PROP + NUM_SCAN
        obs_array[start : start + NUM_PRIV_EXPLICIT] = 3.0  # Priv explicit
        start += NUM_PRIV_EXPLICIT
        obs_array[start : start + NUM_PRIV_LATENT] = 4.0  # Priv latent
        obs_array[-HISTORY_DIM:] = 5.0  # History

        # Verify each component is in the correct position
        prop = obs.get_proprioceptive()
        assert np.allclose(prop, 1.0), "Proprioceptive should be at start"
        assert prop.shape == (NUM_PROP,)

        scan = obs.get_scan()
        assert np.allclose(scan, 2.0), "Scan should be after proprioceptive"
        assert scan.shape == (NUM_SCAN,)

        priv_explicit = obs.get_priv_explicit()
        assert np.allclose(priv_explicit, 3.0), "Privileged explicit should be after scan"
        assert priv_explicit.shape == (NUM_PRIV_EXPLICIT,)

        priv_latent = obs.get_priv_latent()
        assert np.allclose(priv_latent, 4.0), "Privileged latent should be after priv explicit"
        assert priv_latent.shape == (NUM_PRIV_LATENT,)

        history = obs.get_history()
        assert np.allclose(history, 5.0), "History should be at end"
        assert history.shape == (HISTORY_DIM,)

    def test_observation_component_ordering(self):
        """Verify components are in the correct order: prop, scan, priv_explicit, priv_latent, history."""
        obs_array = np.arange(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        # Verify component boundaries
        prop = obs.get_proprioceptive()
        assert prop[0] == 0.0, "Proprioceptive should start at index 0"
        assert prop[-1] == NUM_PROP - 1, "Proprioceptive should end at NUM_PROP-1"

        scan = obs.get_scan()
        assert scan[0] == NUM_PROP, "Scan should start at NUM_PROP"
        assert scan[-1] == NUM_PROP + NUM_SCAN - 1, "Scan should end at NUM_PROP+NUM_SCAN-1"

        priv_explicit = obs.get_priv_explicit()
        expected_start = NUM_PROP + NUM_SCAN
        assert priv_explicit[0] == expected_start, "Priv explicit should start after scan"
        assert priv_explicit[-1] == expected_start + NUM_PRIV_EXPLICIT - 1

        priv_latent = obs.get_priv_latent()
        expected_start = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT
        assert priv_latent[0] == expected_start, "Priv latent should start after priv explicit"
        assert priv_latent[-1] == expected_start + NUM_PRIV_LATENT - 1

        history = obs.get_history()
        expected_start = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT
        assert history[0] == expected_start, "History should start after priv latent"
        assert history[-1] == OBS_DIM - 1, "History should end at OBS_DIM-1"

    def test_observation_no_gaps_or_overlaps(self):
        """Verify components have no gaps or overlaps."""
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        # Mark each component
        prop = obs.get_proprioceptive()
        scan = obs.get_scan()
        priv_explicit = obs.get_priv_explicit()
        priv_latent = obs.get_priv_latent()
        history = obs.get_history()

        prop[:] = 1.0
        scan[:] = 2.0
        priv_explicit[:] = 3.0
        priv_latent[:] = 4.0
        history[:] = 5.0

        # Verify no gaps (all values should be set)
        assert np.all(obs_array > 0), "All observation values should be set (no gaps)"

        # Verify no overlaps (each value should appear exactly once)
        unique_values = np.unique(obs_array)
        assert len(unique_values) == 5, "Should have exactly 5 distinct component values"


class TestObservationDataType:
    """Test observation data type requirements."""

    def test_observation_must_be_float32(self):
        """Verify observation array must be float32."""
        # Valid: float32
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)
        assert obs.observation.dtype == np.float32

        # Should convert float64 to float32
        obs_array_float64 = np.zeros(OBS_DIM, dtype=np.float64)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array_float64)
        assert obs.observation.dtype == np.float32

    def test_observation_shape_validation(self):
        """Verify observation shape is validated."""
        # Valid shape
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)
        assert obs.observation.shape == (OBS_DIM,)

        # Invalid shape should raise error
        with pytest.raises(ValueError, match="Observation shape"):
            obs_array_wrong = np.zeros(OBS_DIM + 1, dtype=np.float32)
            ParkourObservation(timestamp_ns=1, observation=obs_array_wrong)


class TestObservationFormatConsistency:
    """Test observation format consistency across different sources."""

    def test_parkour_model_io_observation_format(self):
        """Verify ParkourModelIO provides observation in correct format."""
        nav_cmd = NavigationCommand.create_now()
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        observation = ParkourObservation(timestamp_ns=1, observation=obs_array)
        model_io = ParkourModelIO(
            timestamp_ns=1,
            schema_version="1.0",
            nav_cmd=nav_cmd,
            observation=observation,
        )

        # Get observation array
        retrieved_array = model_io.get_observation_array()

        # Verify format
        assert retrieved_array.shape == (OBS_DIM,)
        assert retrieved_array.dtype == np.float32
        assert retrieved_array.flags["C_CONTIGUOUS"], "Observation should be C-contiguous"

    def test_observation_from_parts_format(self):
        """Verify from_parts() creates observation in correct format."""
        prop = np.zeros(NUM_PROP, dtype=np.float32)
        scan = np.zeros(NUM_SCAN, dtype=np.float32)
        priv_explicit = np.zeros(NUM_PRIV_EXPLICIT, dtype=np.float32)
        priv_latent = np.zeros(NUM_PRIV_LATENT, dtype=np.float32)
        history = np.zeros(HISTORY_DIM, dtype=np.float32)

        obs = ParkourObservation.from_parts(
            proprioceptive=prop,
            scan=scan,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
        )

        # Verify format
        assert obs.observation.shape == (OBS_DIM,)
        assert obs.observation.dtype == np.float32

        # Verify components are correctly positioned
        assert np.array_equal(obs.get_proprioceptive(), prop)
        assert np.array_equal(obs.get_scan(), scan)
        assert np.array_equal(obs.get_priv_explicit(), priv_explicit)
        assert np.array_equal(obs.get_priv_latent(), priv_latent)
        assert np.array_equal(obs.get_history(), history)

    def test_observation_format_matches_training_spec(self):
        """Verify observation format exactly matches training specification."""
        # Training format: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        obs_array = np.zeros(OBS_DIM, dtype=np.float32)
        obs = ParkourObservation(timestamp_ns=1, observation=obs_array)

        # Verify structure matches spec
        assert len(obs.get_proprioceptive()) == 53, "Proprioceptive should be 53 elements"
        assert len(obs.get_scan()) == 132, "Scan should be 132 elements"
        assert len(obs.get_priv_explicit()) == 9, "Privileged explicit should be 9 elements"
        assert len(obs.get_priv_latent()) == 29, "Privileged latent should be 29 elements"
        assert len(obs.get_history()) == 530, "History should be 530 elements"

        # Verify total
        total = (
            len(obs.get_proprioceptive())
            + len(obs.get_scan())
            + len(obs.get_priv_explicit())
            + len(obs.get_priv_latent())
            + len(obs.get_history())
        )
        assert total == 753, f"Total should be 753, got {total}"

