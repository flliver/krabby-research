"""Unit tests for firmware/observation_mapping.py (pure Python, no numpy/torch).
Run: pytest tests/unit/firmware/test_observation_mapping.py -v
"""

import pytest

from firmware.observation_mapping import (
    CONTACT_DROPPED_LEG,
    CONTACT_FULLSCALE,
    CONTACT_LEGS,
    JointVelocityEstimator,
    contact_forces_from_leg_currents,
    leg_prefix,
)


class TestLegPrefix:
    def test_firmware_name(self):
        assert leg_prefix("FLKL") == "FL"
        assert leg_prefix("RRHY") == "RR"

    def test_hal_name(self):
        assert leg_prefix("FL_knee") == "FL"
        assert leg_prefix("RR_hip_yaw") == "RR"


class TestContactForces:
    def test_option_a_five_legs_drop_mr(self):
        assert CONTACT_LEGS == ("FL", "FR", "ML", "RL", "RR")
        assert CONTACT_DROPPED_LEG == "MR" and "MR" not in CONTACT_LEGS

    def test_zero_current_is_no_contact(self):
        f = contact_forces_from_leg_currents({leg: 0.0 for leg in CONTACT_LEGS})
        assert f == [-0.5] * 5

    def test_fullscale_is_firm_contact(self):
        f = contact_forces_from_leg_currents({leg: CONTACT_FULLSCALE for leg in CONTACT_LEGS})
        assert f == [0.5] * 5

    def test_midscale_is_zero(self):
        assert contact_forces_from_leg_currents({"FL": CONTACT_FULLSCALE / 2})[0] == pytest.approx(0.0)

    def test_clips_above_fullscale(self):
        assert contact_forces_from_leg_currents({"FL": CONTACT_FULLSCALE * 99})[0] == 0.5

    def test_missing_leg_is_unknown_zero(self):
        f = contact_forces_from_leg_currents({"FL": CONTACT_FULLSCALE})
        assert f[0] == 0.5
        assert f[1:] == [0.0, 0.0, 0.0, 0.0]

    def test_slot_order(self):
        assert contact_forces_from_leg_currents({"RL": CONTACT_FULLSCALE})[CONTACT_LEGS.index("RL")] == 0.5


class TestJointVelocityEstimator:
    def test_first_sample_is_zero(self):
        est = JointVelocityEstimator()
        assert est.update("FLKL", 0.5, 100.0) == 0.0

    def test_positive_motion_gives_positive_velocity(self):
        est = JointVelocityEstimator(alpha=1.0)  # no smoothing → raw derivative
        est.update("FLKL", 0.5, 100.0)
        v = est.update("FLKL", 0.6, 100.1)  # +0.1 over 0.1s = 1.0/s
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_ema_smooths_toward_raw(self):
        est = JointVelocityEstimator(alpha=0.2)
        est.update("FLKL", 0.5, 100.0)
        v = est.update("FLKL", 0.55, 100.01)  # raw = 5.0/s; ema = 0.2*5 + 0.8*0 = 1.0
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_zero_dt_returns_prior(self):
        est = JointVelocityEstimator()
        est.update("FLKL", 0.5, 100.0)
        est.update("FLKL", 0.6, 100.1)  # establishes an ema
        same_t = est.update("FLKL", 0.9, 100.1)  # dt=0 → no update, return prior ema
        assert same_t == est._ema["FLKL"]

    def test_per_joint_isolation(self):
        est = JointVelocityEstimator(alpha=1.0)
        est.update("FLKL", 0.5, 100.0)
        est.update("FRKL", 0.1, 100.0)
        assert est.update("FLKL", 0.5, 100.1) == pytest.approx(0.0)  # FL didn't move
