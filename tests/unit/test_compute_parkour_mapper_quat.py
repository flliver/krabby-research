"""Mapper roll/pitch extraction from base_quat_w in its documented (x, y, z, w) order.

Regression test for the quaternion-order mismatch: base_quat_w is (x, y, z, w)
on the wire but euler_xyz_from_quat expects (w, x, y, z). The mapper must
reorder, otherwise real IMU quaternions produce garbage roll/pitch.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from tests.helpers import create_dummy_hw_obs


@pytest.fixture
def mapper() -> HWObservationsToParkourMapper:
    rd = MagicMock()
    rd.get_total_joint_count.return_value = 12
    rd.get_joint_names.return_value = tuple(f"j{i}" for i in range(12))
    rd.get_num_prop.return_value = 48
    rd.get_observation_joint_count.return_value = 12
    dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(rd)
    return HWObservationsToParkourMapper(dims)


def _quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2
    return np.concatenate([axis * np.sin(half), [np.cos(half)]]).astype(np.float32)


def test_identity_quat_gives_zero_roll_pitch(mapper):
    prop = mapper._extract_proprioceptive(create_dummy_hw_obs())
    assert prop[3] == pytest.approx(0.0, abs=1e-6)  # roll
    assert prop[4] == pytest.approx(0.0, abs=1e-6)  # pitch


def test_roll_from_xyzw_quat(mapper):
    obs = create_dummy_hw_obs()
    obs.base_quat_w = _quat_xyzw(np.array([1.0, 0, 0]), np.deg2rad(30))
    prop = mapper._extract_proprioceptive(obs)
    assert prop[3] == pytest.approx(np.deg2rad(30), abs=1e-5)
    assert prop[4] == pytest.approx(0.0, abs=1e-5)


def test_pitch_from_xyzw_quat(mapper):
    obs = create_dummy_hw_obs()
    obs.base_quat_w = _quat_xyzw(np.array([0, 1.0, 0]), np.deg2rad(-45))
    prop = mapper._extract_proprioceptive(obs)
    assert prop[3] == pytest.approx(0.0, abs=1e-5)
    assert prop[4] == pytest.approx(np.deg2rad(-45), abs=1e-5)
