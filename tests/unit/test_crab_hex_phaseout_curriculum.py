"""Unit tests for the PLAN G phase-out curriculum ramp (pure functions, no Isaac).

Contracts under test (see curriculums.py module docstring):
- cosine ramp holds w0 before t0, reaches w1 at t1, midpoint = arithmetic mean;
- epsilon floor: no output magnitude below 1e-3 (weight-0.0 terms lose telemetry);
- spec parser round-trips and fails loudly on malformed entries.
"""

import sys
from pathlib import Path

import pytest

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from curriculums import (  # noqa: E402
    EPS_WEIGHT,
    format_phaseout_spec,
    parse_phaseout_spec,
    ramp_value,
)


class TestRampValue:
    def test_holds_w0_before_and_at_t0(self):
        assert ramp_value(0, 1.0, 0.5, 100, 200) == 1.0
        assert ramp_value(100, 1.0, 0.5, 100, 200) == 1.0

    def test_reaches_w1_at_and_after_t1(self):
        assert ramp_value(200, 1.0, 0.5, 100, 200) == 0.5
        assert ramp_value(10_000_000, 1.0, 0.5, 100, 200) == 0.5

    def test_midpoint_is_arithmetic_mean(self):
        assert ramp_value(150, 1.0, 0.5, 100, 200) == pytest.approx(0.75)

    def test_monotone_decreasing_for_downward_ramp(self):
        values = [ramp_value(t, 1.0, EPS_WEIGHT, 0, 1000) for t in range(0, 1001, 50)]
        assert all(a >= b for a, b in zip(values, values[1:]))

    def test_epsilon_floor_clamps_zero_target(self):
        # A w1 of exactly 0.0 must clamp to +eps, never 0.0 (telemetry death).
        assert ramp_value(500, 1.0, 0.0, 0, 100) == EPS_WEIGHT
        assert ramp_value(50, 0.001, 0.0, 0, 100) == EPS_WEIGHT

    def test_epsilon_floor_preserves_sign_for_negative_targets(self):
        assert ramp_value(500, -0.5, 0.0, 0, 100) == -EPS_WEIGHT

    def test_no_sub_epsilon_output_anywhere_on_ramp(self):
        for t in range(0, 2001, 7):
            w = ramp_value(t, 1.0, EPS_WEIGHT, 0, 2000)
            assert abs(w) >= EPS_WEIGHT

    def test_rejects_empty_window(self):
        with pytest.raises(ValueError):
            ramp_value(0, 1.0, 0.5, 100, 100)


class TestSpecParser:
    def test_single_entry(self):
        entries = parse_phaseout_spec("reward_clock_schedule:1.0:0.001:0:100000")
        assert entries == [{
            "term_name": "reward_clock_schedule",
            "w0": 1.0, "w1": 0.001, "t0": 0, "t1": 100000,
        }]

    def test_multi_entry_round_trip(self):
        spec = "reward_clock_schedule:1.0:0.5:0:50000,reward_clock_swing_apex:1.0:0.001:0:50000"
        entries = parse_phaseout_spec(spec)
        assert len(entries) == 2
        assert parse_phaseout_spec(format_phaseout_spec(entries)) == entries

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            parse_phaseout_spec("reward_clock_schedule:1.0:0.5:0")

    def test_rejects_bad_term_name(self):
        with pytest.raises(ValueError):
            parse_phaseout_spec("bad-name:1.0:0.5:0:100")

    def test_rejects_inverted_window(self):
        with pytest.raises(ValueError):
            parse_phaseout_spec("reward_clock_schedule:1.0:0.5:100:100")

    def test_rejects_empty_spec(self):
        with pytest.raises(ValueError):
            parse_phaseout_spec("  ,  ")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
