"""The gait_eval scoring path reproduces the committed golden fixture EXACTLY.

The omni task turns ``travel_yaw`` from a constant into a per-step array throughout
``gait_eval.metrics`` / ``gait_eval.report`` (plan D10 C2). Two lineages of record are scored by
that code -- the forward task at ``travel_yaw = 0.0`` and the sideways task at ``+pi/2`` -- so the
refactor is only acceptable if both produce bit-identical numbers, not merely close ones.

``tests/unit/fixtures/gait_eval_travel_yaw_golden.json`` was generated on the clean base commit
(plan Verification 0 / step 1) by ``gen_gait_eval_travel_yaw_golden.py``. This test re-runs that
generator's ``build_raw`` / ``score_both`` against the CURRENT code and asserts float equality with
the stored JSON. Never regenerate the fixture to make this pass: a diff here means the scoring
changed for the forward or sideways rows, which is the thing the fixture exists to forbid.
"""

import json
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURES_DIR))

import gen_gait_eval_travel_yaw_golden as GEN  # noqa: E402


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GEN.FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def scored() -> dict:
    return GEN.score_both(GEN.build_raw())


def test_generator_parameters_are_the_ones_the_fixture_was_written_with(golden):
    """A changed generator would make the comparison below vacuous rather than wrong."""
    gen = golden["generator"]
    assert gen["seed"] == GEN.SEED
    assert gen["dt"] == GEN.DT
    assert gen["n_envs"] == GEN.N_ENVS
    assert gen["half_period"] == GEN.HALF_PERIOD
    assert gen["fall_env"] == GEN.FALL_ENV
    assert gen["fall_frac"] == GEN.FALL_FRAC
    assert gen["fall_grace_steps"] == GEN.FALL_GRACE_STEPS
    assert gen["travel_yaws"] == GEN.TRAVEL_YAWS
    assert gen["schedule"] == [
        {"hold_s": h.hold_s, "vx": h.vx, "vy": h.vy, "wz": h.wz, "label": h.label}
        for h in GEN._schedule()
    ]


def test_forward_and_sideways_scoring_is_exactly_the_golden_fixture(scored, golden):
    """Float EQUALITY, not approx: the refactor must not move a single ulp."""
    assert scored == golden["results"]


@pytest.mark.parametrize("key", ["yaw_0", "yaw_half_pi"])
def test_each_episode_matches_key_by_key(scored, golden, key):
    """Same assertion, reported per episode/field so a regression names itself."""
    want, got = golden["results"][key], scored[key]
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert set(g) == set(w), f"{key}[{i}]: metric keys changed"
        for field in w:
            assert g[field] == w[field], f"{key}[{i}][{field}]"


def test_the_fixture_actually_exercises_the_paths_under_test(golden):
    """Guard against a fixture that would match trivially."""
    for key in ("yaw_0", "yaw_half_pi"):
        eps = golden["results"][key]
        assert len(eps) == GEN.N_ENVS
        assert eps[0]["tripod_score"] is not None and eps[0]["tripod_score"] > 0.9
        assert eps[0]["holds"]["fwd"]["tracking"]["vx"]["ratio"] is not None
        assert eps[GEN.FALL_ENV]["fall_direction"]["fall_class"] != "none"
        assert eps[GEN.FALL_ENV]["support_polygon"]["prefall"]["n_frames"] > 0
    # The two conventions must not be the same numbers, or the pi/2 half proves nothing.
    assert golden["results"]["yaw_0"][0]["holds"]["fwd"]["tracking"]["vx"]["ratio"] != (
        golden["results"]["yaw_half_pi"][0]["holds"]["fwd"]["tracking"]["vx"]["ratio"]
    )


def test_body_rows_carry_no_omni_fields(scored):
    """The world-row additions (facing) must be absent -- not null -- on body rows."""
    for key in ("yaw_0", "yaw_half_pi"):
        for ep in scored[key]:
            assert "facing" not in ep and "facing_error_rms" not in ep
            for hold in ep["holds"].values():
                assert "facing" not in hold
