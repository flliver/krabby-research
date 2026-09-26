"""Unit tests for the omni per-category reward telemetry's bearing -> category map (pure torch).

Pins ``crab_hex_omni_command_math.telemetry_category`` (stride/posture plan, step 4):

* ``stand`` / ``turn`` / ``move_turn`` come straight from the slot and ignore ``bearing``
  entirely -- slot precedence over any bearing value;
* a ``SLOT_MOVE`` row is bucketed by ``|bearing|``: ``fwd`` <= 22.5 deg, ``side`` in
  [67.5, 112.5] deg, ``back`` >= 157.5 deg, ``diag`` otherwise -- both boundaries inclusive
  to their named bucket, and both signs (left/right, i.e. +-bearing) land in the same bucket;
* an unwrapped bearing (outside [-pi, pi]) is wrapped internally, so a caller that forgets to
  wrap gets the same category as one that did not.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

TASKS = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks" / "parkour_tasks"
COMMAND_MATH_PY = TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_command_math.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load("crab_hex_omni_command_math__telemetry_test", COMMAND_MATH_PY)

CATS = cm.TELEMETRY_CATEGORIES
STAND, FWD, SIDE, BACK, DIAG, TURN, MOVE_TURN = (CATS.index(n) for n in CATS)


def _deg(d: float) -> torch.Tensor:
    return torch.tensor([math.radians(d)])


def _cat(slot: int, bearing_deg: float) -> int:
    out = cm.telemetry_category(torch.tensor([slot], dtype=torch.long), _deg(bearing_deg))
    assert out.dtype == torch.long
    assert out.shape == (1,)
    return int(out.item())


# --------------------------------------------------------------------------------- categories
def test_categories_are_the_seven_named_buckets():
    assert cm.TELEMETRY_CATEGORIES == ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")


# ------------------------------------------------------------------------------ slot precedence
@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -90.0])
def test_stand_ignores_bearing(bearing_deg):
    assert _cat(cm.SLOT_STAND, bearing_deg) == STAND


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -90.0])
def test_turn_ignores_bearing(bearing_deg):
    assert _cat(cm.SLOT_TURN, bearing_deg) == TURN


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -90.0])
def test_move_turn_ignores_bearing(bearing_deg):
    assert _cat(cm.SLOT_MOVE_TURN, bearing_deg) == MOVE_TURN


# ------------------------------------------------------------------------------------ boundaries
def test_fwd_boundary_inclusive_both_signs():
    assert _cat(cm.SLOT_MOVE, 22.5) == FWD
    assert _cat(cm.SLOT_MOVE, -22.5) == FWD
    assert _cat(cm.SLOT_MOVE, 0.0) == FWD


def test_just_past_fwd_boundary_is_diag():
    assert _cat(cm.SLOT_MOVE, 22.5 + 1e-3) == DIAG
    assert _cat(cm.SLOT_MOVE, -(22.5 + 1e-3)) == DIAG


def test_side_band_inclusive_both_ends_and_signs():
    for d in (67.5, 90.0, 112.5):
        assert _cat(cm.SLOT_MOVE, d) == SIDE
        assert _cat(cm.SLOT_MOVE, -d) == SIDE


def test_just_outside_side_band_is_diag():
    assert _cat(cm.SLOT_MOVE, 67.5 - 1e-3) == DIAG
    assert _cat(cm.SLOT_MOVE, 112.5 + 1e-3) == DIAG


def test_back_boundary_inclusive_both_signs():
    assert _cat(cm.SLOT_MOVE, 157.5) == BACK
    assert _cat(cm.SLOT_MOVE, -157.5) == BACK
    assert _cat(cm.SLOT_MOVE, 180.0) == BACK
    assert _cat(cm.SLOT_MOVE, -180.0) == BACK


def test_just_short_of_back_boundary_is_diag():
    assert _cat(cm.SLOT_MOVE, 157.5 - 1e-3) == DIAG


def test_diag_quadrant_midpoints():
    assert _cat(cm.SLOT_MOVE, 45.0) == DIAG
    assert _cat(cm.SLOT_MOVE, -45.0) == DIAG
    assert _cat(cm.SLOT_MOVE, 135.0) == DIAG
    assert _cat(cm.SLOT_MOVE, -135.0) == DIAG


# ------------------------------------------------------------------------------------ wrap-around
def test_unwrapped_bearing_past_pi_wraps_to_back():
    # 200 deg unwrapped == -160 deg wrapped, |bearing| = 160 >= 157.5 -> back.
    assert _cat(cm.SLOT_MOVE, 200.0) == BACK


def test_unwrapped_bearing_plus_full_turn_matches_wrapped():
    for base_deg, expected in ((10.0, FWD), (90.0, SIDE), (170.0, BACK), (45.0, DIAG)):
        wrapped = _cat(cm.SLOT_MOVE, base_deg)
        unwrapped = _cat(cm.SLOT_MOVE, base_deg + 360.0)
        assert wrapped == expected
        assert unwrapped == expected


def test_negative_multiple_of_two_pi_offset_matches():
    assert _cat(cm.SLOT_MOVE, 30.0) == _cat(cm.SLOT_MOVE, 30.0 - 720.0)


# ------------------------------------------------------------------------------------------ batch
def test_batched_rows_mix_slots_and_bearings_independently():
    slot = torch.tensor(
        [cm.SLOT_STAND, cm.SLOT_MOVE, cm.SLOT_MOVE, cm.SLOT_MOVE, cm.SLOT_MOVE, cm.SLOT_TURN, cm.SLOT_MOVE_TURN],
        dtype=torch.long,
    )
    bearing = torch.deg2rad(torch.tensor([90.0, 0.0, 90.0, -90.0, 180.0, 45.0, 45.0]))
    out = cm.telemetry_category(slot, bearing)
    expected = torch.tensor([STAND, FWD, SIDE, SIDE, BACK, TURN, MOVE_TURN], dtype=torch.long)
    assert torch.equal(out, expected)


def test_output_dtype_and_device_match_slot_id():
    slot = torch.tensor([cm.SLOT_MOVE, cm.SLOT_STAND], dtype=torch.long)
    bearing = torch.zeros(2)
    out = cm.telemetry_category(slot, bearing)
    assert out.dtype == slot.dtype
    assert out.device == slot.device
    assert out.shape == slot.shape


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
