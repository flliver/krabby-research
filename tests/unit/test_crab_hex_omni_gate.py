"""The omni driver's written gates (Part H of the plan; CHARTER.md) -- twin of
``test_crab_hex_side_gate.py`` extended to 2-D holds (eight schedule holds instead of three) and
the facing criteria the omni task adds.

Three gates, all pure functions of a ``report.py::aggregate``-shaped dict:

* ``gate_soft_verdict`` -- the 1a/1b short-stage soft gate on ONE named hold (``fwd`` / ``lat_l``).
* ``gate_1c_verdict`` -- the written formation gate: walking-hold tripod (min over the holds the
  harness scores for tracking) >= 0.40, completion >= 0.70, the median SIGNED yaw-progress rate
  (``facing_by_hold[h]["yaw_progress_rate"]``, mean ``wz * sign(wz_cmd)`` over the hold's commanded-
  turn steps) >= 0.05 rad/s on BOTH ``turn_l90`` and ``turn_r90``, AND (step-16 review decision 3,
  2026-09-17) the median tracking ratio (``tracking_by_hold[h]["ratio"]["median"]``, the harness's
  ``speed_ratio``) >= ``GATE_1C_DIR_RATIO`` 0.3 on EACH of ``lat_l``, ``diag``, ``back``,
  ``move_turn`` -- coverage: a forward-only walker must not pass an omnidirectional formation gate.
  The retired ``yaw_rate_ratio x ang_vel_max`` proxy divided two magnitudes and would have passed the
  2026-09-17 yaw-capability smoke's non-turning head (ratio 0.946, net rotation -0.025 rad/s) -- the
  regression is pinned below.
* ``gate_facing_verdict`` -- the facing-precision gate (enforced from 2a): median
  ``heading_error_final_abs`` <= 0.30 rad on each turn hold.

Plus, from 2026-09-18, the TRAINING-LOG translation floor that every gate also reads
(``track_ratio_series`` / ``track_ratio_tail`` / ``dead_walker``, all pure) -- the cover for the
verified 1c failure, where tripod and survival passed a head whose speed tracking had collapsed to
0.245. Its regression fixtures are the seed-3 campaign's real ``track_ratio_mean`` prints
(``fixtures/omni_track_ratio/``).

Pure Python: the driver and the decision table are loaded by path; nothing here imports Isaac. NOTE:
importing the driver executes ``crab_hex_omni_phases.py`` by path (one task per process, same as the
forward/sideways drivers) -- until that config module exists (plan step 7, in flight alongside this
one) every test in this file errors at fixture setup with a clear ``FileNotFoundError``, not a
logic failure; it will collect and run once step 7 lands.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
DRIVER = OMNI / "experiments/tools/run_phases.py"
TABLE = OMNI / "experiments/2026-09-16_1805_omni_formation/decision_table.py"
OMNI_ENV_CFG = OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py"
OMNI_KNOBS = OMNI / "config/crab_hex/omni_knobs.py"
SCRIPTS_DIR = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M      # noqa: E402  (pure numpy; no Isaac)
from gait_eval import report as R       # noqa: E402

MOVING_HOLDS = ("fwd", "lat_l", "diag", "back", "move_turn")
TURN_HOLDS = ("turn_l90", "turn_r90")
ALL_HOLDS = ("stand",) + MOVING_HOLDS + TURN_HOLDS


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load(DRIVER, "omni_run_phases_gate_test")


@pytest.fixture(scope="module")
def table():
    return _load(TABLE, "omni_decision_table_gate_test")


def _stats(median, n):
    return {"mean": median, "median": median, "n": n, "p25": median, "p75": median}


def _agg(tripod: dict[str, float], completion: float, *, facing: dict[str, dict] | None = None,
          falls: int = 5, n: int = 100, walking_holds: tuple[str, ...] = MOVING_HOLDS,
          scored_holds: tuple[str, ...] | None = None, ratio: dict[str, float] | None = None,
          survival: dict[str, float] | None = None) -> dict:
    """A ``report.py::aggregate``-shaped dict for the omni schedule (8 holds).

    ``tripod`` maps hold -> median tripod score (``stand`` defaults to a high stepping-through-the-
    stop value like the forward reference, so the walking-only gate must actively exclude it).
    ``walking_holds`` controls which holds get a ``tracking_by_hold`` entry with ``ratio.n > 0``
    (i.e. are "walking" by the harness's own definition); ``scored_holds`` (default: all of
    ``tripod``) controls which holds get a ``tripod_score_by_hold`` entry at all (a hold absent here
    is UNMEASURED, not zero). ``ratio`` overrides the default 0.8 median tracking ratio on named
    walking holds (gate_1c's per-direction floor, step-16 review decision 3).
    """
    scored_holds = scored_holds if scored_holds is not None else tuple(tripod)
    ratio = ratio or {}
    tripod_by_hold = {h: _stats(tripod[h], n - falls) for h in scored_holds}
    tracking_by_hold = {}
    for h in ALL_HOLDS:
        is_walking = h in walking_holds
        cmd = 0.25 if is_walking else 0.0
        ratio_n = (n - falls) if is_walking else 0
        r = ratio.get(h, 0.8) if is_walking else None
        tracking_by_hold[h] = {"cmd_vx_mean": cmd, "achieved_vx": _stats(0.2 if is_walking else 0.0, ratio_n),
                                "ratio": _stats(r, ratio_n)}
    a = {
        "n_episodes": n,
        "schedule_completion_rate": completion,
        "termination_reasons": {"schedule_complete": n - falls, "fall": falls},
        "fall_classes": {"none": n - falls, "roll": falls},
        "tripod_score": _stats(sum(tripod.values()) / max(1, len(tripod)), n),
        "tripod_score_by_hold": tripod_by_hold,
        "tracking_ratio": _stats(0.7, n - falls),
        "tracking_by_hold": tracking_by_hold,
    }
    if facing:
        a["facing_by_hold"] = {h: {k: _stats(v, n - falls) for k, v in vals.items()} for h, vals in facing.items()}
    _a = a
    # survival THROUGH each hold (2026-09-17 soft-gate fix); default = the whole-schedule completion
    # for every hold so the older tests keep their semantics, and None when completion is None.
    _a["survival_by_hold"] = dict(survival) if survival is not None else {
        h: (None if completion is None else completion) for h in ALL_HOLDS}
    return _a

def _facing(final: float, settle: float, progress: float, *, net: float | None = None,
            yaw_rate_ratio: float = 0.9) -> dict:
    """One hold's facing medians, keyed exactly as ``report.py::aggregate`` writes them.

    ``progress`` is the gated quantity (``yaw_progress_rate``, rad/s toward the command);
    ``net`` (default: ``progress``) is the ``net_yaw_rate_toward`` cross-check; ``yaw_rate_ratio`` is
    deliberately left FLATTERING by default (0.9) so any test that passes the gate has to do it on
    the signed number.
    """
    return {"heading_error_rms": final, "heading_error_final_abs": final, "heading_error_max_abs": final * 1.5,
            "settle_time_s": settle, "yaw_rate_ratio": yaw_rate_ratio,
            "yaw_progress_rate": progress, "net_yaw_rate_toward": progress if net is None else net}


# A head that clears every criterion comfortably (used as the "healthy" baseline throughout).
HEALTHY_TRIPOD = {"stand": 0.90, "fwd": 0.55, "lat_l": 0.50, "diag": 0.52, "back": 0.48, "move_turn": 0.45,
                  "turn_l90": 0.30, "turn_r90": 0.30}
HEALTHY_FACING = {
    "turn_l90": _facing(final=0.15, settle=3.0, progress=0.08),   # rad/s toward the command >= 0.05
    "turn_r90": _facing(final=0.20, settle=3.5, progress=0.08),
}
#: The training-log translation number a healthy stage shows (2026-09-18 floor). 0.747 is the
#: seed-3 1a head's own its-800-1000 mean -- the only omni stage on record that walked. Every
#: eval-shaped test below passes this so it exercises the criterion IT is about; the floor has
#: its own tests further down.
HEALTHY_TRACK_RATIO = 0.747


# ------------------------------------------------------------------ walking_tripod (shared helper)
def test_walking_tripod_excludes_stand_and_turn_holds(drv):
    walk, by_hold = drv.walking_tripod(_agg(HEALTHY_TRIPOD, 0.9, facing=HEALTHY_FACING))
    assert walk == pytest.approx(min(HEALTHY_TRIPOD[h] for h in MOVING_HOLDS))
    assert by_hold["stand"] == pytest.approx(0.90)          # present, but not counted in the min
    assert set(h for h in by_hold if by_hold[h] is not None) >= set(MOVING_HOLDS)


def test_walking_tripod_none_when_a_moving_hold_is_unscored(drv):
    a = _agg(HEALTHY_TRIPOD, 0.9, facing=HEALTHY_FACING, scored_holds=tuple(h for h in HEALTHY_TRIPOD if h != "back"))
    walk, by_hold = drv.walking_tripod(a)
    assert walk is None and by_hold["back"] is None


# ------------------------------------------------------------------ gate_soft_verdict (1a/1b)
@pytest.mark.parametrize("hold, tri, comp, expect", [
    ("fwd", 0.30, 0.70, "ok"),           # both floors inclusive
    ("fwd", 0.2999, 0.90, "halt"),
    ("fwd", 0.50, 0.6999, "halt"),
    ("lat_l", 0.30, 0.70, "ok"),
    ("lat_l", 0.25, 0.90, "halt"),
])
def test_soft_gate_floor_boundaries(drv, hold, tri, comp, expect):
    tripod = dict(HEALTHY_TRIPOD); tripod[hold] = tri
    s = drv.summarize(_agg(tripod, comp, facing=HEALTHY_FACING))
    status, msg = drv.gate_soft_verdict(s, hold, drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == expect, msg
    if expect == "halt":
        assert "FAILED" in msg and "--continue" in msg


def test_soft_gate_reads_only_its_own_hold(drv):
    """1a's soft gate must not be dragged down by lat_l/diag/back being unformed (1a never samples
    those slots): a low value everywhere except 'fwd' still passes the 1a gate."""
    tripod = {"stand": 0.01, "fwd": 0.45, "lat_l": 0.02, "diag": 0.01, "back": 0.01, "move_turn": 0.0,
              "turn_l90": 0.0, "turn_r90": 0.0}
    s = drv.summarize(_agg(tripod, 0.85, facing=None, walking_holds=("fwd",), scored_holds=("stand", "fwd")))
    status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok", msg


def test_soft_gate_no_usable_number_is_a_halt(drv):
    for s in ({}, drv.summarize(None), drv.summarize(_agg(HEALTHY_TRIPOD, None, facing=HEALTHY_FACING))):
        status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION, track_ratio=HEALTHY_TRACK_RATIO)
        assert status == "halt" and "could not be evaluated" in msg


# ------------------------------------------------------------------ gate_1c_verdict
def test_gate_1c_passes_on_a_healthy_head(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok", msg
    assert "yaw-progress rate" in msg and "turn_l90 0.080" in msg and "net" in msg


@pytest.mark.parametrize("bad_hold, bad_value", [("fwd", 0.39), ("lat_l", 0.10), ("move_turn", 0.0)])
def test_gate_1c_fails_on_tripod_floor(drv, bad_hold, bad_value):
    tripod = dict(HEALTHY_TRIPOD); tripod[bad_hold] = bad_value
    s = drv.summarize(_agg(tripod, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_fails_on_completion_floor(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.6999, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_fails_a_non_turning_head(drv):
    """The plant's historical failure mode: the P-control commands a yaw rate and the body does not
    turn (or turns the wrong way). Progress 0.0 / -0.02 rad/s, both under the 0.05 floor."""
    facing = {"turn_l90": _facing(final=1.0, settle=None, progress=0.0),
              "turn_r90": _facing(final=1.0, settle=None, progress=-0.02)}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg and "yaw-progress" in msg
    assert "0.000" in msg and "-0.020" in msg


def test_gate_1c_fails_one_turn_hold_alone(drv):
    """BOTH turn holds must clear the floor -- a left turn that works does not excuse a right turn
    that does not (the 1c command band is symmetric; a one-sided head is a formation failure)."""
    for bad in TURN_HOLDS:
        facing = {h: _facing(final=0.2, settle=2.0, progress=(0.01 if h == bad else 0.12)) for h in TURN_HOLDS}
        status, msg = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing)), track_ratio=HEALTHY_TRACK_RATIO)
        assert status == "gate_1c" and "FAILED" in msg, bad


def test_gate_1c_fails_the_retired_proxys_false_pass(drv):
    """REGRESSION (2026-09-17 yaw smoke): the exact numbers the retired proxy passed.

    ``yaw_rate_ratio`` 0.946 x ``ang_vel_max`` 0.3 = 0.284 rad/s -- 5.7x the floor -- on a
    ``turn_l90`` hold whose signed progress was about -0.025 rad/s against a +0.30 command. The
    signed gate must FAIL this head.
    """
    facing = {h: _facing(final=1.61, settle=None, progress=-0.025, net=-0.02, yaw_rate_ratio=0.946)
              for h in TURN_HOLDS}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.81, facing=facing))
    assert s["facing_by_hold"]["turn_l90"]["yaw_rate_ratio"] * 0.3 > 5 * drv.GATE_1C_YAW_RATE
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg
    assert not hasattr(drv, "_turn_rate_proxy") and not hasattr(drv, "ANG_VEL_MAX_DEFAULT")


def test_gate_1c_boundary_is_inclusive(drv):
    """Progress exactly at the 0.05 rad/s floor passes."""
    facing = {h: _facing(final=0.2, settle=2.0, progress=drv.GATE_1C_YAW_RATE) for h in TURN_HOLDS}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.70, facing=facing))
    status, _ = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok"
    facing = {h: _facing(final=0.2, settle=2.0, progress=drv.GATE_1C_YAW_RATE - 1e-6) for h in TURN_HOLDS}
    status, _ = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.70, facing=facing)), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c"


@pytest.mark.parametrize("mutate", [
    lambda a: a["tripod_score_by_hold"].pop("fwd"),                      # a moving hold never scored
    lambda a: a.__setitem__("schedule_completion_rate", None),
    lambda a: a.pop("facing_by_hold"),                                    # no facing at all
    lambda a: a["facing_by_hold"]["turn_r90"].pop("yaw_progress_rate"),    # facing, but no progress key
    lambda a: a["facing_by_hold"].pop("turn_l90"),                         # one turn hold never scored
])
def test_gate_1c_no_usable_number_is_a_halt(drv, mutate):
    a = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING)
    mutate(a)
    status, msg = drv.gate_1c_verdict(drv.summarize(a), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "could not be evaluated" in msg


# ------------------------------------------------------------------ gate_1c_verdict: per-direction floor
# Step-16 review decision 3 (2026-09-17): the walking-hold tripod / completion / turning-rate
# criteria above do not require the robot to actually TRACK any particular direction -- a head that
# only ever walks forward could still clear them by stepping through the stop on the other holds.
def test_gate_1c_fails_a_forward_only_walker_naming_the_failing_hold(drv):
    """Turns fine (facing healthy) and tracks forward well (ratio 0.9), but lat_l ratio 0.05 is
    far under the 0.3 floor -- FAILS, and the message names lat_l."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio={"fwd": 0.9, "lat_l": 0.05}))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg
    assert "lat_l" in msg and "failing: lat_l" in msg


def test_gate_1c_passes_all_four_directions_above_the_floor(drv):
    ratio = {"lat_l": 0.35, "diag": 0.35, "back": 0.35, "move_turn": 0.35}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio))
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok", msg


def test_gate_1c_dir_ratio_boundary_is_inclusive(drv):
    """Exactly at the 0.3 floor passes on all four; a hair under fails."""
    ratio = {h: drv.GATE_1C_DIR_RATIO for h in drv.GATE_1C_DIR_HOLDS}
    status, _ = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok"
    ratio = {h: drv.GATE_1C_DIR_RATIO - 1e-6 for h in drv.GATE_1C_DIR_HOLDS}
    status, msg = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_missing_direction_hold_halts(drv):
    """A direction hold never scored for tracking (e.g. three failed eval attempts) halts, same as
    a missing tripod/completion/turn-rate number."""
    a = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING)
    a["tracking_by_hold"].pop("move_turn")
    status, msg = drv.gate_1c_verdict(drv.summarize(a), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "could not be evaluated" in msg


def test_decision_table_gate_1c_per_direction_floor_matches_the_driver(drv, table):
    for ratio in (
        {"lat_l": 0.05},
        {"lat_l": 0.35, "diag": 0.35, "back": 0.35, "move_turn": 0.35},
        {h: drv.GATE_1C_DIR_RATIO for h in drv.GATE_1C_DIR_HOLDS},
    ):
        a = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)
        ok, why = table.gate_1c(a)
        status, _ = drv.gate_1c_verdict(drv.summarize(a), track_ratio=HEALTHY_TRACK_RATIO)
        assert ok == (status == "ok"), (ratio, why)


# ------------------------------------------------------------------ gate_facing_verdict (2a/2b)
# 2026-09-18 (user decision after the seed-3 2a re-eval): the turning-RATE criterion is enforced,
# facing PRECISION is watched -- reported in the message, never a halt.
def test_gate_facing_passes_when_both_turn_holds_turn_fast_enough(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_facing_verdict(s)
    assert status == "ok" and "turn gate passed" in msg


@pytest.mark.parametrize("final_l,final_r", [(0.53, 0.13), (0.10, 0.95), (1.2, 1.2)])
def test_gate_facing_overshoot_is_watched_not_enforced(drv, final_l, final_r):
    facing = {h: dict(v) for h, v in HEALTHY_FACING.items()}
    facing["turn_l90"]["heading_error_final_abs"] = final_l
    facing["turn_r90"]["heading_error_final_abs"] = final_r
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    status, msg = drv.gate_facing_verdict(s)
    assert status == "ok" and "WATCH facing precision" in msg and f"{final_l:.3f}" in msg


@pytest.mark.parametrize("prog_l,prog_r", [(0.02, 0.09), (0.09, 0.0), (0.049, 0.049)])
def test_gate_facing_fails_on_the_turn_rate(drv, prog_l, prog_r):
    facing = {h: dict(v) for h, v in HEALTHY_FACING.items()}
    facing["turn_l90"]["yaw_rate_ratio"] = facing["turn_l90"].get("yaw_rate_ratio", 0.5)
    facing["turn_l90"]["yaw_progress_rate"] = prog_l
    facing["turn_r90"]["yaw_progress_rate"] = prog_r
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    status, msg = drv.gate_facing_verdict(s)
    assert status == "gate_facing" and "turn gate FAILED" in msg


def test_gate_facing_no_usable_number_is_a_halt(drv):
    for s in ({}, drv.summarize(None)):
        status, msg = drv.gate_facing_verdict(s)
        assert status == "gate_facing" and "could not be evaluated" in msg


# ------------------------------------------------------------------ summarize() carries facing + per-direction tracking
def test_summarize_exposes_facing_and_speed_ratio_by_hold(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=HEALTHY_FACING))
    assert s["facing_by_hold"]["turn_l90"]["final"] == pytest.approx(0.15)
    assert s["facing_by_hold"]["turn_l90"]["progress"] == pytest.approx(0.08)
    assert s["facing_by_hold"]["turn_l90"]["net"] == pytest.approx(0.08)
    assert s["yaw_progress_by_hold"] == {h: pytest.approx(0.08) for h in TURN_HOLDS}
    assert s["net_yaw_by_hold"] == {h: pytest.approx(0.08) for h in TURN_HOLDS}
    assert set(s["speed_ratio_by_hold"]) == set(drv.DIRECTION_HOLDS)
    assert s["speed_ratio_by_hold"]["fwd"] == pytest.approx(0.8)


# ------------------------------------------------------------------ the decision table agrees
def test_decision_table_gate_1c_matches_the_driver(drv, table):
    for tripod, comp, facing in (
        (HEALTHY_TRIPOD, 0.85, HEALTHY_FACING),
        ({**HEALTHY_TRIPOD, "back": 0.10}, 0.85, HEALTHY_FACING),
        (HEALTHY_TRIPOD, 0.85, {h: _facing(1.0, None, -0.025, net=-0.02, yaw_rate_ratio=0.946) for h in TURN_HOLDS}),
        (HEALTHY_TRIPOD, 0.85, {h: _facing(0.2, 2.0, 0.05) for h in TURN_HOLDS}),   # exactly at the floor
    ):
        a = _agg(tripod, comp, facing=facing)
        ok, why = table.gate_1c(a)
        status, _ = drv.gate_1c_verdict(drv.summarize(a), track_ratio=HEALTHY_TRACK_RATIO)
        assert ok == (status == "ok"), (tripod, comp, why)
        assert table.walking_tripod(a)[0] == pytest.approx(drv.walking_tripod(a)[0]) if drv.walking_tripod(a)[0] is not None else table.walking_tripod(a)[0] is None


def test_decision_table_gate_facing_matches_the_driver(drv, table):
    for facing in (HEALTHY_FACING, {"turn_l90": _facing(0.5, 2.0, 0.08), "turn_r90": _facing(0.1, 2.0, 0.08)}):
        a = _agg(HEALTHY_TRIPOD, 0.9, facing=facing)
        ok, _ = table.gate_facing(a)
        status, _ = drv.gate_facing_verdict(drv.summarize(a))
        assert ok == (status == "ok")


# ------------------------------------------------------------------ end-to-end: report.py -> the gate
# The gate reads ``facing_by_hold[h]["yaw_progress_rate"]["median"]``, a key written by
# ``report.py::aggregate`` from ``metrics.py::facing_metrics``. These two build the WHOLE chain out of
# synthetic per-step series -- so a rename or a shape change on either side fails here rather than
# silently making the campaign gate unevaluable (which halts) or, worse, flattering.
N_HOLD = 400          # 8 s at dt 0.02, the omni schedule's hold length
DT = 0.02
MOVING_CMD = {"fwd": 0.25, "lat_l": 0.22, "diag": 0.25, "back": 0.22, "move_turn": 0.22}


def _hold_entry(label: str, *, turning: bool) -> dict:
    """One ``score_episode`` hold entry: tripod + tracking + a real ``facing_metrics`` dict."""
    n = N_HOLD
    if label in TURN_HOLDS:
        sign = 1.0 if label == "turn_l90" else -1.0          # turn_l90 faces +pi/2, turn_r90 back to 0
        cmd_wz = np.full(n, sign * 0.30)                     # the P-control saturated at ang_vel_max
        if turning:
            n_turn = int(round((np.pi / 2) / 0.2 / DT))
            err = np.concatenate([np.linspace(sign * np.pi / 2, 0.0, n_turn), np.zeros(n - n_turn)])
            wz = np.concatenate([np.full(n_turn, sign * 0.2), np.zeros(n - n_turn)])
        else:
            # The 2026-09-17 smoke: the command sits at the clip, the body wobbles at |wz| 0.284 and
            # drifts 0.2 rad AWAY from the target over the hold.
            err = sign * np.linspace(np.pi / 2, np.pi / 2 + 0.2, n)
            wz = 0.284 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    else:
        cmd_wz = np.zeros(n)                                  # no yaw command on the travel holds
        err = np.full(n, 0.05)
        wz = 0.01 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    entry = {
        "label": label,
        "tripod": {"tripod_score": 0.55 if label in MOVING_CMD else 0.45, "low_confidence": False,
                   "valid": True, "discard_reason": None},
        "n_steps": n,
        "facing": M.facing_metrics(err, wz, DT, wz_cmd_t=cmd_wz),
    }
    cmd = MOVING_CMD.get(label, 0.0)
    if cmd:
        entry["tracking"] = {"vx": {"cmd_mean": cmd, "actual_mean": 0.8 * cmd, "ratio": 0.8}}
    else:
        entry["tracking"] = {"vx": {"cmd_mean": 0.0, "actual_mean": 0.0, "ratio": None}}
    return entry


def _episode(*, turning: bool) -> dict:
    holds = {label: _hold_entry(label, turning=turning) for label in ALL_HOLDS}
    return {
        "holds": holds,
        "tripod_score": 0.55,
        "tracking_ratio": 0.8,
        "termination_reason": "schedule_complete",
        "completed_schedule": True,
        "facing_error_rms": 0.4 if turning else 1.5,
    }


def _synthetic_aggregate(*, turning: bool, n_episodes: int = 16) -> dict:
    return R.aggregate([_episode(turning=turning) for _ in range(n_episodes)])


def test_gate_1c_on_a_report_shaped_summary_passes_a_turning_head(drv, table):
    agg = _synthetic_aggregate(turning=True)
    prog = agg["facing_by_hold"]["turn_l90"]["yaw_progress_rate"]
    assert prog["median"] == pytest.approx(0.2 * 393 / N_HOLD, rel=1e-3)          # ~0.196 rad/s
    assert agg["facing_by_hold"]["turn_l90"]["net_yaw_rate_toward"]["median"] == pytest.approx(
        (np.pi / 2) / (N_HOLD * DT), rel=1e-2)
    s = drv.summarize(agg)
    status, msg = drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok", msg
    assert table.gate_1c(agg)[0] is True
    # The harness's own summary line carries both numbers (and still the legacy ratio).
    text = R.summary_text({"scenario_id": "slow__omni", "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0",
                           "checkpoint": "model_499.pt", "terrain_source": "raycast_mesh",
                           "command_override_max_deviation": 0.0},
                          [_episode(turning=True) for _ in range(4)])
    assert "progress=0.196" in text and "net=0.196" in text and "yaw_rate_ratio=" in text


def test_gate_1c_on_a_report_shaped_summary_fails_the_wobbling_head(drv, table):
    """The yaw-capability smoke, rebuilt from per-step series: ratio ~0.95, progress ~0, net ~-0.025."""
    agg = _synthetic_aggregate(turning=False)
    fac = agg["facing_by_hold"]["turn_l90"]
    assert fac["yaw_rate_ratio"]["median"] == pytest.approx(0.284 / 0.30, rel=1e-6)   # the old proxy's input
    assert fac["yaw_progress_rate"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert fac["net_yaw_rate_toward"]["median"] == pytest.approx(-0.2 / (N_HOLD * DT), rel=1e-6)
    status, msg = drv.gate_1c_verdict(drv.summarize(agg), track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "gate_1c" and "FAILED" in msg
    assert table.gate_1c(agg)[0] is False


def test_soft_gate_reads_survival_through_its_own_hold_not_whole_schedule(drv):
    """Regression for the 2026-09-17 1a halt: 0.09 whole-schedule completion (falls on the five
    untrained holds) but 0.93 survival through 'fwd' must PASS; 0.5 through 'fwd' must FAIL even
    when the whole schedule somehow completes."""
    tripod = dict(HEALTHY_TRIPOD); tripod["fwd"] = 0.452
    surv = {h: 0.09 for h in ALL_HOLDS}; surv.update({"stand": 1.0, "fwd": 0.93})
    s = drv.summarize(_agg(tripod, 0.09, facing=None, survival=surv))
    status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION, track_ratio=HEALTHY_TRACK_RATIO)
    assert status == "ok" and "0.930" in msg, msg
    surv2 = {h: 0.95 for h in ALL_HOLDS}; surv2["fwd"] = 0.5
    s2 = drv.summarize(_agg(tripod, 0.95, facing=None, survival=surv2))
    status2, msg2 = drv.gate_soft_verdict(s2, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION, track_ratio=HEALTHY_TRACK_RATIO)
    assert status2 == "halt" and "survival through the hold 0.500" in msg2, msg2


def test_survival_by_hold_cuts_at_the_hold_ends(drv):
    labels = ["stand", "fwd", "lat_l"]
    hold_s = {"stand": 8.0, "fwd": 8.0, "lat_l": 8.0}
    # prologue 1.5 s + 8 s = 475 steps to end stand; 875 to end fwd; 1275 to end lat_l (dt 0.02)
    eps = [{"n_steps": 3000}, {"n_steps": 875}, {"n_steps": 874}, {"n_steps": 100}]
    out = drv.survival_by_hold(eps, labels, hold_s, prologue_s=1.5, dt=0.02)
    assert out == {"stand": 0.75, "fwd": 0.5, "lat_l": 0.25}
    assert drv.survival_by_hold([], labels, hold_s) == {"stand": None, "fwd": None, "lat_l": None}
    assert drv.survival_by_hold(eps, labels, {"stand": 8.0})["fwd"] is None


# ---------------------------------------------------------------- eval_env_for (2026-09-17 defect)
# The eval subprocess environment is what decides whether a gate measures the POLICY or a policy
# run through an observation/command function it never trained under. Until 2026-09-17 the driver
# built it from five reward keys + the ring geometry and nothing else: `_subenv` strips every
# KRABBY_* from the parent and the preset variable is deliberately never exported to an eval, so
# `activate_phase` never expanded OMNI_COMMON there and the Play env built on CLASS DEFAULTS --
# `facing_obs_clip=None` (raw 1.57-rad facing error on the canary's 90-degree turns) against the
# 0.6 the policy trained with. These pin the carry list, the RSI-off default and the preset-free
# invariant. Pure: `eval_env_for` touches neither os.environ nor the filesystem.


def _eval_envs(drv, phase="1c"):
    """Both eval rows for one phase, built exactly as eval_morph / eval_obst build them."""
    spec = drv.ph.PHASES[phase]
    return {
        "canary": drv.eval_env_for(spec, "A15+B", keep=drv.CANARY_KEYS, keep_source=drv.ph.FORMATION),
        "obstacle": drv.eval_env_for(spec, "A15+B", keep=drv.CANARY_KEEP + drv.TERRAIN_PASS,
                                     extra=drv.OBST_EVAL),
    }


def test_eval_carry_names_the_obs_and_command_knobs(drv):
    """The carry list is the observation / command-semantics knobs, and only those."""
    assert set(drv.EVAL_CARRY) == {
        "KRABBY_OMNI_FACING_OBS_CLIP",      # obs channels 6/7 (facing error) clip
        "KRABBY_OMNI_SPEED_ELLIPSE",        # _write_command's body command on a world-frame row
        "KRABBY_OMNI_ANG_VEL_MAX",          # heading_rate_command's yaw-rate saturation
        "KRABBY_OMNI_CLOCK_TURN_R",         # omni_clock_turn_radius -> the observed clock phase
        "KRABBY_OMNI_YAW_RANGE",            # the reset event's spawn-yaw band
        "KRABBY_OMNI_FACING_OBS_ENCODING",  # raw vs (sin, cos): the observation VECTOR itself
        "KRABBY_OMNI_ANG_VEL_CLIP",         # the yaw-rate dead band the turn metrics are scored over
        # motion-quality plan step 2 (2026-09-22), all default off:
        "KRABBY_OMNI_RELAY_HYST",           # the yaw-rate relay's hysteresis latch (turn command)
        "KRABBY_OMNI_STAND_HOLD",           # applied wz 0 on the stand slot (stand holds)
        "KRABBY_OMNI_STAND_PARK",           # the stand slot's parked clock phase (observed)
        "KRABBY_ACTION_LPF_HZ",             # the rod-command low-pass (what an action does)
        "KRABBY_ACTION_LPF_OBS",            # + 12 observation channels
        "KRABBY_OMNI_STAND_OBS",            # + the stand bit, zeroed clock
        # clock-deprecation plan step 2 (2026-09-23), default off:
        "KRABBY_OMNI_CLOCK_HIDE",           # the clock's sin/cos (dims 15-16 + history) read (0, 0)
        # side-speed-bisection plan step 1b (2026-09-28), default off:
        "KRABBY_OMNI_CMD_OBS",              # obs slots 8-9 (command xy) read 0; slot 10 and 5-7 kept
        # PPO-construction plan step 2 (2026-10-01), default off -- RUNNER side, but structural:
        "KRABBY_STD_PARAM",                 # soft std: the policy's std PARAMETER (std_raw vs std)
        "KRABBY_INIT_STD",                  # ... and the init it is built from
    }
    # The hide RAMP is keyed on the env step counter, which restarts at 0 in an eval process: carried,
    # it would show the clock on every eval episode, so it is never carried.
    assert "KRABBY_OMNI_CLOCK_HIDE_RAMP" not in drv.EVAL_CARRY
    assert "KRABBY_OMNI_CLOCK_HIDE_RAMP" not in drv.EVAL_CARRY_DEFAULTS
    # Sampling / reward / terrain knobs are NOT carried: the canary is a minimal flat eval whose
    # commands come from the manifest schedule override and whose verdict must not move when a
    # stage re-weights a reward. KRABBY_OMNI_CMD_POOL is the newest member of that list and the
    # easiest to get wrong -- it looks like a command knob and is a SAMPLER knob: it chooses which
    # command types a TRAINING window draws, and no eval row ever runs the sampler.
    # KRABBY_OMNI_MIRROR_IMAGES (side-speed-bisection plan step 1a) is the runner-only sibling of
    # KRABBY_OMNI_CMD_OBS: the symmetry loss does not run in an eval process at all, so there is
    # nothing for it to carry (see the CMD_OBS note in the block above).
    for banned in ("KRABBY_OMNI_SLOTS", "KRABBY_OMNI_DIR_BAND", "KRABBY_OMNI_FACING",
                   "KRABBY_OMNI_RSI_SOURCE", "KRABBY_YAW_W", "KRABBY_YAW_PROGRESS_W",
                   "KRABBY_STAND_FRAC", "KRABBY_RESAMPLE_S", "KRABBY_OMNI_CMD_POOL",
                   "KRABBY_OMNI_MIRROR_IMAGES"):
        assert banned not in drv.EVAL_CARRY
        assert banned not in drv.EVAL_CARRY_DEFAULTS


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_every_eval_env_carries_every_eval_carry_key_from_the_preset(drv, row):
    """EVAL_CARRY reaches BOTH eval rows: the PHASE preset's value, else the recorded default.

    A key no preset sets is one the policy trained on the config's CLASS DEFAULT, so the eval must
    build on that same default -- which it would do by omitting the key entirely. It is written out
    anyway so ``run_meta.env_vars`` records the effective value; a blank there cannot be told apart
    later from an eval that forgot the knob. Hence: every carried key reaches every row, and it is
    never absent from BOTH the preset and the defaults (that would be a stale carry list).
    """
    pe = drv.ph.phase_env("1c")
    ev = _eval_envs(drv)[row]
    for key in drv.EVAL_CARRY:
        expected = pe.get(key, drv.EVAL_CARRY_DEFAULTS.get(key))
        assert expected is not None, f"{key} is in neither the 1c preset nor EVAL_CARRY_DEFAULTS"
        assert ev.get(key) == expected, f"{row}: {key} missing or not the effective value"
    # The specific regression: the facing-error observation clip the 2026-09-17 yaw gate turned on.
    assert ev["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6"


def test_eval_carry_defaults_stand_in_only_for_keys_no_preset_sets(drv):
    """The defaults are a RECORD of the class default, so they may not shadow a preset value."""
    pe = drv.ph.phase_env("1c")
    for key, value in drv.EVAL_CARRY_DEFAULTS.items():
        assert key in drv.EVAL_CARRY, f"{key} is defaulted but never carried"
        assert key not in pe, f"{key} is set by the 1c preset; the default would be dead code"
        assert isinstance(value, str) and value, f"{key}: an env value is a non-empty string"
    # The two class-body defaults these restate (crab_hex_omni_command.py: facing_obs_encoding
    # "raw", clips.ang_vel_clip 0.2), plus the six motion-quality knobs and the clock-hide knob at
    # their OFF value (the knob-off cfg is the class default: motion_knobs / clock_knobs write
    # nothing for "0"). A drift here silently re-specifies every eval.
    assert drv.EVAL_CARRY_DEFAULTS == {"KRABBY_OMNI_FACING_OBS_ENCODING": "raw",
                                       "KRABBY_OMNI_ANG_VEL_CLIP": "0.2",
                                       "KRABBY_OMNI_RELAY_HYST": "0", "KRABBY_OMNI_STAND_HOLD": "0",
                                       "KRABBY_OMNI_STAND_PARK": "0", "KRABBY_ACTION_LPF_HZ": "0",
                                       "KRABBY_ACTION_LPF_OBS": "0", "KRABBY_OMNI_STAND_OBS": "0",
                                       "KRABBY_OMNI_CLOCK_HIDE": "0",
                                       # side-speed-bisection plan step 1b: "0" is one of
                                       # omni_knobs.CMD_OBS_OFF_SPELLINGS.
                                       "KRABBY_OMNI_CMD_OBS": "0",
                                       # PPO-construction plan step 2: each knob's off spelling
                                       "KRABBY_STD_PARAM": "0", "KRABBY_INIT_STD": "0"}
    # The five keys the presets DO set keep the preset's value, bitwise -- the defaults changed
    # nothing about the rows that already ran.
    ev = _eval_envs(drv)["canary"]
    for key in ("KRABBY_OMNI_FACING_OBS_CLIP", "KRABBY_OMNI_SPEED_ELLIPSE", "KRABBY_OMNI_ANG_VEL_MAX",
                "KRABBY_OMNI_CLOCK_TURN_R", "KRABBY_OMNI_YAW_RANGE"):
        assert ev[key] == pe[key]


def test_motion_carry_defaults_are_the_knob_off_values_and_build_nothing(drv):
    """The six motion-quality carry defaults (2026-09-22) restate each knob's OFF value, and that
    value writes NOTHING into an env cfg -- so an eval carrying the default builds exactly the
    class-default cfg a policy trained without the knob saw (the whole point of the pin)."""
    knobs = _load(OMNI_KNOBS, "omni_knobs_gate_motion_test")
    mk = knobs._mk
    # NOTE(p2r Z2, 2026-10-10): EVAL_CARRY is eval-locked, so a semantics knob added after the lock
    # (KRABBY_OMNI_STAND_PROXY) is named in mk.EVAL_CARRY_EXTERNAL and carried by the campaign's own eval env
    # instead; every other semantics knob is carried here, and an external one must really be absent here.
    external = set(getattr(mk, "EVAL_CARRY_EXTERNAL", ()))
    assert external <= set(mk.EVAL_SEMANTIC_KNOBS) and not external & set(drv.EVAL_CARRY)
    assert set(mk.EVAL_SEMANTIC_KNOBS) - external <= set(drv.EVAL_CARRY)
    for key in mk.EVAL_SEMANTIC_KNOBS:
        if key in external:
            assert key not in drv.EVAL_CARRY_DEFAULTS and mk.KNOB_OFF_VALUES[key] == "0", key
            continue
        assert drv.EVAL_CARRY_DEFAULTS[key] == mk.KNOB_OFF_VALUES[key], key
    # the reward / exploration knobs of the same plan are NOT carried (reward and sampler rule)
    for banned in ("KRABBY_ACTION_STD_MAX", "KRABBY_ACTION_STD_RESET", "KRABBY_ENTROPY_COEF",
                   "KRABBY_ACTION_RATE_SQ", "KRABBY_ACTION_SMOOTH2_W", "KRABBY_JOINT_VEL_W",
                   "KRABBY_FOOT_SLIP_W", "KRABBY_STAND_POSE_W", "KRABBY_POWER_W", "KRABBY_AIRTIME_W"):
        assert banned not in drv.EVAL_CARRY and banned not in drv.EVAL_CARRY_DEFAULTS

    class _Bag:
        pass

    def _cfg():
        cfg = _Bag()
        cfg.commands = _Bag()
        cfg.commands.base_velocity = _Bag()
        cfg.actions = _Bag()
        cfg.actions.joint_pos = _Bag()
        cfg.rewards = _Bag()
        cfg.observations = _Bag()
        return cfg

    def _snapshot(obj):
        if isinstance(obj, _Bag):
            return {k: _snapshot(v) for k, v in vars(obj).items()}
        return obj

    carried, unset = _cfg(), _cfg()
    mk.apply_motion_knobs(carried, {k: drv.EVAL_CARRY_DEFAULTS.get(k, mk.KNOB_OFF_VALUES[k])
                                    for k in mk.EVAL_SEMANTIC_KNOBS})
    mk.apply_motion_knobs(unset, {})
    assert _snapshot(carried) == _snapshot(unset) == _snapshot(_cfg())


def test_clock_hide_carry_default_is_the_knob_off_value_and_builds_nothing(drv):
    """The clock-hide carry (clock-deprecation plan step 2/3, 2026-09-23): the hide knob is carried
    with its OFF value "0" as the default -- which writes nothing into a cfg, so an eval of a
    clock-visible head builds exactly the cfg it trained on -- and the ramp is never carried."""
    knobs = _load(OMNI_KNOBS, "omni_knobs_gate_clock_test")
    ck = knobs._ck
    assert tuple(ck.EVAL_SEMANTIC_KNOBS) == ("KRABBY_OMNI_CLOCK_HIDE",)
    assert set(ck.EVAL_SEMANTIC_KNOBS) <= set(drv.EVAL_CARRY)
    for key in ck.EVAL_SEMANTIC_KNOBS:
        assert drv.EVAL_CARRY_DEFAULTS[key] == ck.KNOB_OFF_VALUES[key] == "0", key
    assert ck.CLOCK_HIDE_RAMP_VAR not in drv.EVAL_CARRY
    assert ck.CLOCK_HIDE_RAMP_VAR not in drv.EVAL_CARRY_DEFAULTS

    class _Bag:
        pass

    class _OmniObs:                      # the probe ``clock_knobs`` finds the omni term by
        OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}

    def _cfg():
        cfg = _Bag()
        cfg.observations = _Bag()
        cfg.observations.policy = _Bag()
        cfg.observations.policy.omni_obs = _Bag()
        cfg.observations.policy.omni_obs.func = _OmniObs
        cfg.observations.policy.omni_obs.params = {"history_length": 10}
        return cfg

    carried = _cfg()
    ck.apply_clock_knobs(carried, {k: drv.EVAL_CARRY_DEFAULTS[k] for k in ck.EVAL_SEMANTIC_KNOBS})
    assert carried.observations.policy.omni_obs.params == {"history_length": 10}
    armed = _cfg()                       # the same cfg IS found and armed by the ON value
    ck.apply_clock_knobs(armed, {"KRABBY_OMNI_CLOCK_HIDE": "1"})
    assert armed.observations.policy.omni_obs.params == {"history_length": 10, "clock_hide": True}
    # a ramp carried without the hide would be refused anyway -- the carry never sends one
    with pytest.raises(ValueError):
        ck.apply_clock_knobs(_cfg(), {"KRABBY_OMNI_CLOCK_HIDE": "0", ck.CLOCK_HIDE_RAMP_VAR: "0:240"})


@pytest.mark.parametrize("key,value", [("KRABBY_OMNI_ANG_VEL_CLIP", "0.05"),
                                       ("KRABBY_OMNI_FACING_OBS_ENCODING", "sincos")])
def test_a_preset_that_sets_a_defaulted_key_keeps_its_own_value(drv, monkeypatch, key, value):
    """The ``k not in pe`` guard on the defaults injection, tested against a preset that DOES set
    the key -- no shipped preset does today, so without this the guard is untested dead code.

    The day a candidate's recipe is promoted into a preset, an unguarded injection would overwrite
    it with the class default and score the policy under a command dead band (or an observation
    encoding) it never trained on, with nothing failing.
    """
    real_phase_env = drv.ph.phase_env
    monkeypatch.setattr(drv.ph, "phase_env", lambda name: {**real_phase_env(name), key: value})
    ev = drv.eval_env_for(drv.ph.PHASES["1c"], "A15+B", keep=drv.CANARY_KEYS, keep_source=drv.ph.FORMATION)
    assert ev[key] == value
    assert drv.EVAL_CARRY_DEFAULTS[key] != value        # the default would have been visible


def _declared_facing_obs_encodings(path: Path) -> list:
    """Every ``"facing_obs_encoding": <value>`` entry written in a config SOURCE file.

    Parsed rather than imported: building the omni env cfg needs Isaac, so the only test that can
    read the live object is the integration one, which does not run here. An AST walk is immune to
    re-indentation, re-wrapping and the term moving between classes -- it fails only if the
    declaration is renamed, removed or given a non-literal value, which are the three things that
    would actually break the eval.
    """
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "facing_obs_encoding":
                out.append(v.value if isinstance(v, ast.Constant) else ast.dump(v))
    return out


def test_the_shipped_obs_term_declares_the_encoding_every_eval_now_carries(drv):
    """``EVAL_CARRY_DEFAULTS`` puts ``KRABBY_OMNI_FACING_OBS_ENCODING=raw`` into EVERY eval
    subprocess, which turns ``omni_knobs._apply_facing_obs_encoding_knob`` from a path that returned
    at its first line into a live one with two ``RuntimeError`` guards.

    It passes only because the shipped omni observation term already declares the param. If that
    literal is removed, renamed or re-nested, every eval row of every phase dies at config build
    with "... do not declare a 'facing_obs_encoding' param" -- and the gate reports "could not be
    evaluated" AFTER the training window has been spent. Nothing else at unit level reads the real
    cfg (the knob's own tests drive a fake one), so this is the only cheap cover.
    """
    knobs = _load(OMNI_KNOBS, "omni_knobs_gate_test")          # Isaac-free (math import is lazy)
    declared = _declared_facing_obs_encodings(OMNI_ENV_CFG)
    assert declared, (f"{OMNI_ENV_CFG.name} declares no 'facing_obs_encoding' param; every eval "
                      "row would raise at config build")
    default = drv.EVAL_CARRY_DEFAULTS["KRABBY_OMNI_FACING_OBS_ENCODING"]
    for value in declared:
        assert value in knobs.FACING_OBS_ENCODINGS, f"{value!r} is not a parseable encoding"
        # ... and the carried default RESTATES the class default, so an eval that omits the knob and
        # one that carries it build the same observation vector.
        assert value == default


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_carry_overlay_reaches_the_eval_but_only_its_eval_carry_keys(drv, row):
    """A candidate that overrides an obs/command knob must be SCORED under that override.

    The overlay handed in is the experiment's whole TRAINING environment -- sampler and reward keys
    included -- so ``eval_env_for`` filters it to EVAL_CARRY rather than merging it: an eval that
    inherited the sampler would stop being the fixed canary every phase is compared on.
    """
    spec = drv.ph.PHASES["1c"]
    overlay = {"KRABBY_OMNI_FACING_OBS_ENCODING": "sincos",   # carried: the obs vector changes
               "KRABBY_OMNI_ANG_VEL_CLIP": "0.05",            # carried: the yaw dead band
               "KRABBY_OMNI_CMD_POOL": "fwd:1.0",             # sampler: must NOT reach the eval
               "KRABBY_OMNI_SLOTS": "0.2:0.8:0:0",            # sampler: must NOT reach the eval
               "KRABBY_YAW_PROGRESS_W": "0.5",                # reward: must NOT reach the eval
               "KRABBY_OMNI_PHASE": "1c"}                     # preset: must NOT reach the eval
    kwargs = ({"keep": drv.CANARY_KEYS, "keep_source": drv.ph.FORMATION} if row == "canary"
              else {"keep": drv.CANARY_KEEP + drv.TERRAIN_PASS, "extra": drv.OBST_EVAL})
    ev = drv.eval_env_for(spec, "A15+B", carry_overlay=overlay, **kwargs)
    assert ev["KRABBY_OMNI_FACING_OBS_ENCODING"] == "sincos"
    assert ev["KRABBY_OMNI_ANG_VEL_CLIP"] == "0.05"
    for banned in ("KRABBY_OMNI_CMD_POOL", "KRABBY_OMNI_SLOTS", "KRABBY_YAW_PROGRESS_W", "KRABBY_OMNI_PHASE"):
        assert banned not in ev
    # Without an overlay the same row falls back to preset-or-default, unchanged.
    plain = drv.eval_env_for(spec, "A15+B", **kwargs)
    assert plain["KRABBY_OMNI_FACING_OBS_ENCODING"] == "raw"
    assert plain["KRABBY_OMNI_ANG_VEL_CLIP"] == "0.2"
    assert {k: v for k, v in ev.items() if k not in drv.EVAL_CARRY} == {
        k: v for k, v in plain.items() if k not in drv.EVAL_CARRY}


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_eval_env_turns_rsi_off_explicitly(drv, row):
    """RSI_FRAC=0 is SET, not merely absent, so no eval ever starts an episode from a bank frame."""
    ev = _eval_envs(drv)[row]
    assert ev["KRABBY_RSI_FRAC"] == "0"
    assert drv.ph.phase_env("1c")["KRABBY_RSI_FRAC"] == "0.2"   # the preset would have armed it


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_eval_env_never_carries_a_preset_variable(drv, row):
    """KRABBY_OMNI_PHASE would re-expand the whole training preset inside the Play env; the other
    two raise at config import (one task per process)."""
    ev = _eval_envs(drv)[row]
    for banned in ("KRABBY_OMNI_PHASE", "KRABBY_PHASE", "KRABBY_SIDE_PHASE"):
        assert banned not in ev
    # ... and the sampler/reward keys the preset would have brought with it stay out too.
    for banned in ("KRABBY_OMNI_SLOTS", "KRABBY_OMNI_FACING", "KRABBY_YAW_PROGRESS_W",
                   "KRABBY_RSI_BANK", "KRABBY_OMNI_RSI_SOURCE"):
        assert banned not in ev


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_eval_env_names_the_plant(drv, row):
    """with_plant used to set KRABBY_HEX_USD_PATH only -- and plant_usd_path('A15+B') is None
    (A15+B IS the main asset), so the eval subprocess carried no plant variable at all and the
    harness recorded plant_requested '<default>'. The USD coincided; the request did not."""
    assert drv.ph.plant_usd_path("A15+B") is None
    assert _eval_envs(drv)[row]["KRABBY_PLANT"] == "A15+B"
    # A non-main plant still gets its USD pointed at.
    spec = drv.ph.PHASES["1c"]
    other = drv.eval_env_for(spec, "golden", keep=drv.CANARY_KEYS, keep_source=drv.ph.FORMATION)
    assert other["KRABBY_PLANT"] == "golden"
    assert other["KRABBY_HEX_USD_PATH"] == drv.ph.plant_usd_path("golden")


def test_canary_reward_frame_is_formation_not_the_phase(drv):
    """The canary's reward keys stay FORMATION's across phases (a 2a verdict must be comparable
    with a 1c one) while its EVAL_CARRY follows the phase."""
    spec_2a = drv.ph.PHASES["2a"]
    ev = drv.eval_env_for(spec_2a, "A15+B", keep=drv.CANARY_KEYS, keep_source=drv.ph.FORMATION)
    for key in drv.CANARY_KEYS:
        if key in drv.ph.FORMATION:
            assert ev[key] == drv.ph.FORMATION[key]
    assert ev["KRABBY_OMNI_FACING_OBS_CLIP"] == drv.ph.phase_env("2a")["KRABBY_OMNI_FACING_OBS_CLIP"]


def test_eval_env_keeps_the_ring_geometry_every_omni_process_requires(drv):
    """require_ring_geom raises in any omni process without it; the obstacle row's own block wins."""
    envs = _eval_envs(drv)
    assert envs["canary"]["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert envs["obstacle"]["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    for key, value in drv.OBST_EVAL.items():
        assert envs["obstacle"][key] == value
    # The canary stays a minimal flat row: no MODE/FLAT_FRAC/DIFF from the training preset.
    for key in ("KRABBY_FLAT_TERRAIN_MODE", "KRABBY_FLAT_TERRAIN_FLAT_FRAC", "KRABBY_FLAT_TERRAIN_DIFF"):
        assert key not in envs["canary"]


def test_eval_env_for_is_pure(drv, monkeypatch):
    """No os.environ read or write: a leaked parent value must not change the eval env."""
    monkeypatch.setenv("KRABBY_OMNI_FACING_OBS_CLIP", "99.0")
    monkeypatch.setenv("KRABBY_RSI_FRAC", "0.9")
    ev = _eval_envs(drv)["canary"]
    assert ev["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6" and ev["KRABBY_RSI_FRAC"] == "0"


# ------------------------------------------------- the training-log translation floor (2026-09-18)
TRACK_RATIO_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "omni_track_ratio"
#: (stage, its-800-1000 mean, last-500 mean) measured on the seed-3 campaign's own logs.
REAL_LOGS = {"1a": (0.7474, 0.6925), "1b": (0.4330, 0.4057)}


@pytest.fixture(params=sorted(REAL_LOGS), ids=sorted(REAL_LOGS))
def real_log(request):
    return request.param, TRACK_RATIO_FIXTURES / f"seed3_{request.param}_track_ratio.log"


def test_track_ratio_series_parses_a_real_training_log(drv, real_log):
    """The parser reads the trainer's own print format, not a hand-written one."""
    stage, path = real_log
    vals = drv.track_ratio_series(path)
    assert len(vals) == 1000, f"{stage}: one print per iteration"
    assert all(0.0 <= v <= 1.5 for v in vals)


def test_track_ratio_tail_reproduces_the_campaign_numbers(drv, real_log):
    stage, path = real_log
    _win, last500 = REAL_LOGS[stage]
    got = drv.track_ratio_tail(drv.track_ratio_series(path))
    assert got == pytest.approx(last500, abs=5e-4), f"{stage}: last-500 mean {got}"


def test_the_floor_would_have_halted_the_chain_at_1b(drv):
    """The regression the floor exists for: 1a PASSES, 1b FAILS.

    The seed-3 chain ran 1a -> 1b -> 1c with tripod/survival gates only and passed both short
    stages; speed tracking was 0.747 -> 0.433 -> 0.245 the whole time. With this floor the
    campaign stops after 1b, before 1c spends 2.5 GPU-h."""
    tails = {s: drv.track_ratio_tail(drv.track_ratio_series(TRACK_RATIO_FIXTURES / f"seed3_{s}_track_ratio.log"))
             for s in REAL_LOGS}
    assert tails["1a"] >= drv.GATE_TRACK_RATIO_MIN
    assert tails["1b"] < drv.GATE_TRACK_RATIO_MIN


def test_track_ratio_tail_is_none_on_a_log_without_the_metric(drv, tmp_path):
    """``None``, never ``nan``: a nan compares False against every floor, which reads as a PASS."""
    empty = tmp_path / "no_metric.log"
    empty.write_text("Mean reward: 12.3\nMean episode length: 1800\n")
    assert drv.track_ratio_series(empty) == []
    assert drv.track_ratio_tail([]) is None


def test_track_ratio_tail_averages_only_the_tail(drv):
    vals = [0.0] * 1000 + [0.8] * 500
    assert drv.track_ratio_tail(vals, n=500) == pytest.approx(0.8)
    assert drv.track_ratio_tail(vals, n=1500) == pytest.approx(0.8 * 500 / 1500)


# ---------------------------------------------------- the floor inside each gate
def test_soft_gate_fails_a_creeper_that_passes_tripod_and_survival(drv):
    """The whole point: a head can pass BOTH eval criteria and still not be walking."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.95))
    ok, _ = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION,
                                  track_ratio=0.75)
    assert ok == "ok"
    status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION,
                                        track_ratio=0.406)      # the seed-3 1b number
    assert status == "halt" and "track_ratio_mean 0.406" in msg


@pytest.mark.parametrize("tr, expect", [(0.6, "ok"), (0.5999, "halt"), (0.61, "ok")])
def test_soft_gate_track_ratio_boundary_is_inclusive(drv, tr, expect):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.95))
    status, _ = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION,
                                      track_ratio=tr)
    assert status == expect


def test_soft_gate_missing_track_ratio_is_a_halt(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.95))
    status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION,
                                        track_ratio=None)
    assert status == "halt" and "track_ratio_mean n/a" in msg


def test_gate_1c_fails_the_creeper_on_the_training_floor_alone(drv):
    """Every eval criterion healthy, training translation collapsed -> gate_1c."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    assert drv.gate_1c_verdict(s, track_ratio=0.75)[0] == "ok"
    status, msg = drv.gate_1c_verdict(s, track_ratio=0.245)     # the seed-3 1c number
    assert status == "gate_1c" and "track_ratio_mean 0.245" in msg


def test_gate_1c_missing_track_ratio_is_a_halt(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s, track_ratio=None)
    assert status == "gate_1c" and "could not be evaluated" in msg and "track_ratio_mean n/a" in msg


def test_every_gate_message_names_the_training_floor(drv):
    """A halt the user reads at 3 a.m. must say which criterion failed."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    for _st, msg in (drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION,
                                           track_ratio=HEALTHY_TRACK_RATIO),
                     drv.gate_1c_verdict(s, track_ratio=HEALTHY_TRACK_RATIO)):
        assert "track_ratio_mean" in msg and str(drv.GATE_TRACK_RATIO_MIN) in msg


# ------------------------------------------------- the from-scratch early abort (dead_walker)
def test_dead_walker_is_silent_before_the_check_point(drv):
    """The check fires ONCE, on the way past iteration 1000 -- never on a shorter run."""
    assert drv.dead_walker([0.0] * 999) is False
    assert drv.dead_walker([]) is False


def test_dead_walker_fires_on_a_creeper(drv):
    vals = [0.1] * 1000
    assert drv.dead_walker(vals) is True


def test_dead_walker_reads_the_window_not_the_whole_run(drv):
    """A run that is forming LATE must not be aborted for its noisy first 800 iterations."""
    late = [0.0] * 800 + [0.7] * 200
    assert drv.dead_walker(late) is False
    early = [0.7] * 800 + [0.1] * 200
    assert drv.dead_walker(early) is True


def test_dead_walker_would_not_have_aborted_the_1a_stage(drv):
    """1a formed a walker from scratch in exactly this budget: the floor must clear it."""
    vals = drv.track_ratio_series(TRACK_RATIO_FIXTURES / "seed3_1a_track_ratio.log")
    window = sum(vals[800:1000]) / 200
    assert window == pytest.approx(REAL_LOGS["1a"][0], abs=5e-4)
    assert window >= drv.ABORT_TRACK_RATIO_MIN
    assert drv.dead_walker(vals) is False


def test_dead_walker_boundary_is_inclusive(drv):
    assert drv.dead_walker([drv.ABORT_TRACK_RATIO_MIN] * 1000) is False
    assert drv.dead_walker([drv.ABORT_TRACK_RATIO_MIN - 1e-6] * 1000) is True


def test_dead_walker_is_continue_overridable_but_dead_plant_is_not(drv):
    """``dead_walker`` is a floor on a head that exists; ``dead_plant`` is a run with nothing."""
    assert "dead_walker" in drv.OVERRIDABLE_STATUSES
    assert drv.GATE_STATUSES <= drv.OVERRIDABLE_STATUSES
    assert "dead_plant" not in drv.OVERRIDABLE_STATUSES
    assert "nan" not in drv.OVERRIDABLE_STATUSES and "infra" not in drv.OVERRIDABLE_STATUSES


def test_abort_constants_are_consistent_with_the_gate_floor(drv):
    """The abort must be strictly laxer than the gate: aborting a run the gate would pass is a
    self-inflicted halt."""
    assert drv.ABORT_TRACK_RATIO_MIN < drv.GATE_TRACK_RATIO_MIN
    assert drv.ABORT_WINDOW < drv.ABORT_AT_ITERS


# ------------------------------------------------------------------ --from-scratch
def _cli(*args) -> "subprocess.CompletedProcess":
    import subprocess
    return subprocess.run([sys.executable, str(DRIVER), *args], capture_output=True, text=True,
                          timeout=120)


def test_from_scratch_is_advertised_in_the_cli(tmp_path):
    out = _cli("--help")
    assert out.returncode == 0
    assert "--from-scratch" in out.stdout
    assert "dead_walker" in out.stdout      # the abort it arms is documented where the flag is


def test_from_scratch_and_from_checkpoint_are_mutually_exclusive(tmp_path):
    """Both set is an ambiguous instruction about the ONE decision this run turns on."""
    out = _cli("--campaign-dir", str(tmp_path), "--phases", "1c",
               "--from-scratch", "--from-checkpoint", "/tmp/model_9.pt")
    assert out.returncode != 0
    assert "contradict" in out.stderr


def test_resume_label_spells_out_a_deliberate_cold_start(drv):
    assert drv.resume_label({"resume": "/logs/run/model_1998.pt"}) == "model_1998.pt"
    assert drv.resume_label({"resume": None}) == "scratch"
    assert drv.resume_label({"resume": None, "from_scratch": True}) == "scratch (--from-scratch)"
    # a checkpoint always wins: --from-scratch never coexists with a resume point
    assert drv.resume_label({"resume": "/a/model_9.pt", "from_scratch": True}) == "model_9.pt"


# ==================================================================== A4: the driver seams
# Everything below pins the campaign-driver seam added on 2026-09-18: the hoisted summary keys a
# decision rule reads, the exposure/telemetry collection that feeds it, and the SUBPROCESS layer
# (interrupt checks, SIGTERM-before-SIGKILL, the eval's status/return code). The subprocess tests
# drive REAL children -- a python script standing in for train.py / eval_crab_hex_gait.py -- with
# the poll interval monkeypatched down, because the whole point of the seam is what happens to a
# process, which a mocked Popen cannot show.


# ------------------------------------------------------------------ hoisted summary keys
def test_summarize_hoists_the_keys_the_candidate_ladder_scores(drv):
    """The ladder is a pure function of summarize()'s OUTPUT: every quantity it needs is hoisted
    here, so no downstream code re-derives a definition (and no two of them drift apart)."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=HEALTHY_FACING,
                           ratio={"fwd": 0.9, "lat_l": 0.5, "diag": 0.4, "back": 0.6, "move_turn": 0.3}))
    # every WALKING hold, not just the four gated directions (WATCH table)
    assert set(s["ratio_by_hold"]) == set(MOVING_HOLDS)
    assert s["ratio_by_hold"]["move_turn"] == pytest.approx(0.3)
    # achieved speed in m/s -- the absolute-units twin, reported on every hold with tracking
    assert s["achieved_speed_by_hold"]["fwd"] == pytest.approx(0.2)
    assert s["achieved_speed_by_hold"]["stand"] == pytest.approx(0.0)
    # min over the four DIRECTION holds: stricter than tripod_walking, which includes move_turn
    assert s["tripod_walking_dir"] == pytest.approx(min(HEALTHY_TRIPOD[h] for h in drv.DIRECTION_HOLDS))
    assert s["tripod_walking_dir"] >= s["tripod_walking"]
    # the worse of the two turn holds
    assert s["yaw_progress_turn_min"] == pytest.approx(0.08)
    # existing keys untouched
    assert s["tripod_walking"] is not None and s["completion"] == pytest.approx(0.9)
    assert set(s["speed_ratio_by_hold"]) == set(drv.DIRECTION_HOLDS)


def test_tripod_walking_dir_excludes_move_turn_and_is_none_when_a_direction_is_unscored(drv):
    """move_turn's tripod is a TURNING-gait number: a facing candidate moves it by design, so a
    guard that included it would reject a candidate for doing the thing it was proposed to do."""
    tripod = dict(HEALTHY_TRIPOD); tripod["move_turn"] = 0.01
    s = drv.summarize(_agg(tripod, 0.9, facing=HEALTHY_FACING))
    assert s["tripod_walking_dir"] == pytest.approx(min(HEALTHY_TRIPOD[h] for h in drv.DIRECTION_HOLDS))
    assert s["tripod_walking"] == pytest.approx(0.01)        # the laxer number DOES see it
    a = _agg(HEALTHY_TRIPOD, 0.9, facing=HEALTHY_FACING,
             scored_holds=tuple(h for h in HEALTHY_TRIPOD if h != "back"))
    assert drv.summarize(a)["tripod_walking_dir"] is None     # a floor over an incomplete set flatters


def test_yaw_progress_turn_min_is_none_when_either_turn_hold_is_unscored(drv):
    a = _agg(HEALTHY_TRIPOD, 0.9, facing={"turn_l90": _facing(0.2, 2.0, 0.08)})
    assert drv.summarize(a)["yaw_progress_turn_min"] is None
    facing = {"turn_l90": _facing(0.2, 2.0, 0.12), "turn_r90": _facing(0.2, 2.0, 0.03)}
    assert drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=facing))["yaw_progress_turn_min"] == pytest.approx(0.03)


def test_yaw_progress_turn_min_scores_the_progress_rate_not_the_net_rotation(drv):
    """The two turning quantities are DIFFERENT metrics in decision 5 -- the ordinary F block
    scores ``yaw_progress_rate``, the absolute-units F block scores whole-hold
    ``net_yaw_rate_toward`` -- so the guard may not silently read the other one.

    Every other test of this key leaves them equal (``_facing``'s ``net`` defaults to ``progress``),
    which is exactly the shape a swap survives. A policy that wobbles fast toward the command has a
    high progress rate and a near-zero net rotation; the opposite policy drifts slowly but
    monotonically. Scored on the wrong one, the ladder adopts the wrong candidate.
    """
    facing = {"turn_l90": _facing(0.2, 2.0, 0.12, net=0.01),
              "turn_r90": _facing(0.2, 2.0, 0.09, net=0.02)}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=facing))
    assert s["yaw_progress_turn_min"] == pytest.approx(0.09)       # min over PROGRESS
    assert s["yaw_progress_turn_min"] != pytest.approx(0.01)       # not min over net
    # ... and the net rate is still hoisted, separately, for the absolute-units block
    assert s["net_yaw_by_hold"]["turn_l90"] == pytest.approx(0.01)
    assert s["yaw_progress_by_hold"]["turn_l90"] == pytest.approx(0.12)


def test_turn_survivals_are_conditional_on_reaching_the_turn(drv):
    """The cumulative survivals mostly measure the four travel holds BEFORE the turns; dividing by
    the hold immediately before isolates the falls that happen during the turn itself."""
    surv = {h: 1.0 for h in ALL_HOLDS}
    surv.update({"back": 0.80, "turn_l90": 0.40, "turn_r90": 0.20})
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.2, facing=HEALTHY_FACING, survival=surv))
    assert s["turn_survival_l90"] == pytest.approx(0.40 / 0.80)
    assert s["turn_survival_r90"] == pytest.approx(0.20 / 0.40)
    # ... and they do NOT simply re-read the cumulative numbers the completion guard already covers
    assert s["turn_survival_l90"] != pytest.approx(surv["turn_l90"])


@pytest.mark.parametrize("surv, keys", [
    ({"back": 0.0, "turn_l90": 0.0, "turn_r90": 0.0}, ("turn_survival_l90", "turn_survival_r90")),
    ({"back": 0.5, "turn_l90": 0.0, "turn_r90": 0.0}, ("turn_survival_r90",)),
])
def test_turn_survival_is_missing_not_zero_on_a_zero_denominator(drv, surv, keys):
    """No episode reached the conditioning hold -> nothing is conditioned on. MISSING, and the
    decision rule treats MISSING as "cannot be scored", never as 0 (which reads as a total loss)."""
    full = {h: 1.0 for h in ALL_HOLDS}; full.update(surv)
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.2, facing=HEALTHY_FACING, survival=full))
    for key in keys:
        assert s[key] is None
    assert drv.conditional_survival(None, "turn_l90", "back") is None
    assert drv.conditional_survival({"back": 0.5}, "turn_l90", "back") is None    # numerator missing


def test_ratio_by_hold_follows_the_rows_own_schedule(drv):
    """A curriculum row with mirrored holds reports them without a code change (the gated
    speed_ratio_by_hold / dir_ratio_by_hold stay keyed by the names the gates enforce)."""
    a = {"tracking_by_hold": {
        "stand": {"cmd_vx_mean": 0.0, "achieved_vx": _stats(0.01, 80), "ratio": _stats(None, 0)},
        "fwd": {"cmd_vx_mean": 0.25, "achieved_vx": _stats(0.2, 80), "ratio": _stats(0.80, 80)},
        "lat_r": {"cmd_vx_mean": 0.22, "achieved_vx": _stats(0.1, 80), "ratio": _stats(0.45, 80)},
        "move_turn_r": {"cmd_vx_mean": 0.22, "achieved_vx": _stats(0.05, 60), "ratio": _stats(0.25, 60)},
        # commanded to walk, but every episode fell before it: named as unmeasured, never dropped
        "back": {"cmd_vx_mean": 0.22, "achieved_vx": _stats(None, 0), "ratio": _stats(None, 0)},
    }}
    s = drv.summarize(a)
    assert set(s["ratio_by_hold"]) == {"fwd", "lat_r", "move_turn_r", "back"}
    assert s["ratio_by_hold"]["lat_r"] == pytest.approx(0.45)
    assert s["ratio_by_hold"]["back"] is None          # unmeasured, and still named
    assert "stand" not in s["ratio_by_hold"]           # a commanded stop is not a walking hold
    assert s["achieved_speed_by_hold"]["stand"] == pytest.approx(0.01)   # ... but IS a speed reading


# ------------------------------------------------------------------ facing_closed (metric + proxy)
def test_facing_closed_floor_matches_the_harness(drv):
    """The driver restates the floor rather than importing the harness; a drift would make the
    proxy and the metric answer different questions on the same hold."""
    assert drv.FACING_CLOSED_MIN_INITIAL_RAD == M.FACING_CLOSED_MIN_INITIAL_RAD


def test_facing_closed_prefers_the_measured_metric_and_records_its_source(drv):
    """With the per-episode metric present, summarize reports the MEDIAN of those fractions."""
    facing = {h: dict(_facing(0.2, 2.0, 0.08), facing_closed=0.87, heading_error_initial_abs=1.57)
              for h in TURN_HOLDS}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    assert s["facing_closed_by_hold"]["turn_l90"] == pytest.approx(0.87)
    assert s["facing_closed_source"] == "metric"


def test_facing_closed_falls_back_to_the_median_proxy_on_an_older_aggregate(drv):
    """An aggregate that has the per-hold INITIAL error but not the per-episode fraction is
    re-scored from its median initial and final errors -- explicitly labelled as a proxy."""
    facing = {h: dict(_facing(0.4, 2.0, 0.08), heading_error_initial_abs=1.6) for h in TURN_HOLDS}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    assert s["facing_closed_by_hold"]["turn_r90"] == pytest.approx(1.0 - 0.4 / 1.6)
    assert s["facing_closed_source"] == "proxy"


def test_facing_closed_refuses_to_derive_a_fraction_from_the_peak_error(drv):
    """An aggregate with NO initial error (every row recorded before 2026-09-18) yields None, not a
    number built from ``heading_error_max_abs``.

    The peak error is not the starting error whenever the body first turns the wrong way, and
    ``1 - final/max`` is then flattering by construction: measured on the recorded 1a slow row, the
    max-derived fraction reads +0.30/+0.28/+0.29 on lat_l/diag/back where the per-episode metric
    reads -0.51/-0.39/-0.40 -- three sign flips. ``None`` says "re-score this run_dir from its
    raws", which is recoverable; a wrong-signed number is not.
    """
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))   # max = final * 1.5
    assert s["facing_closed_by_hold"] == {h: None for h in TURN_HOLDS}
    assert s["facing_closed_source"] is None            # nothing was derived, so nothing is sourced
    # ... and the refusal is about the MISSING initial, not about the hold: give it one and the
    # same row is scored.
    with_initial = {h: dict(v, heading_error_initial_abs=1.5) for h, v in HEALTHY_FACING.items()}
    s2 = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=with_initial))
    assert s2["facing_closed_by_hold"]["turn_l90"] == pytest.approx(1.0 - 0.15 / 1.5)
    assert s2["facing_closed_source"] == "proxy"


def test_facing_closed_is_none_where_there_was_nothing_to_close_and_is_clipped(drv):
    """A hold that starts on target measures nothing (dividing by ~0 turns wobble into a verdict);
    a body that opens the error is clipped at -1 so one spinning episode cannot own the median."""
    facing = {"turn_l90": dict(_facing(0.02, 0.0, 0.0), heading_error_initial_abs=0.05),   # under the floor
              "turn_r90": dict(_facing(3.0, None, -0.1), heading_error_initial_abs=0.5)}   # opened wide
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    assert s["facing_closed_by_hold"]["turn_l90"] is None
    assert s["facing_closed_by_hold"]["turn_r90"] == pytest.approx(-1.0)
    assert drv.facing_closed_by_hold({})[1] is None                      # no facing at all: no source
    assert drv.facing_closed_by_hold({"h": {"final": 0.1}})[0] == {"h": None}   # nothing to divide by


def test_facing_closed_source_says_mixed_when_one_row_used_both_paths(drv):
    """``"mixed"`` is a documented return value the ladder is told to act on (never pool a proxy
    number against a metric one), so it may not be a branch nobody has run.

    A single-version aggregate cannot produce it -- which is the point: if it ever appears, the row
    was assembled from two harness versions and its holds are not comparable with each other.
    """
    by_hold = {"turn_l90": {"closed": 0.9, "initial": 1.5, "final": 0.15},   # the measured metric
               "turn_r90": {"closed": None, "initial": 1.6, "final": 0.4}}   # rebuilt from medians
    out, source = drv.facing_closed_by_hold(by_hold)
    assert out["turn_l90"] == pytest.approx(0.9)
    assert out["turn_r90"] == pytest.approx(1.0 - 0.4 / 1.6)
    assert source == "mixed"
    # and a single-source row is never labelled "mixed", whichever source it is
    assert drv.facing_closed_by_hold({"a": by_hold["turn_l90"]})[1] == "metric"
    assert drv.facing_closed_by_hold({"a": by_hold["turn_r90"]})[1] == "proxy"


def test_ratio_by_hold_keeps_a_commanded_hold_that_produced_no_ratio(drv):
    """``_is_walking_hold``'s second clause and its boundary.

    A hold every episode fell before has ``ratio.n == 0`` and must appear as ``None`` rather than
    vanish: a direction hold silently missing from a WATCH table reads as "not in this schedule",
    which is the opposite of what happened. The comparison is strictly ABOVE the floor, so a hold
    commanded at exactly ``WALKING_CMD_MIN`` (i.e. at the clip, not above it) is not a walking hold.
    """
    a = _agg(HEALTHY_TRIPOD, 0.5, facing=HEALTHY_FACING, walking_holds=("fwd", "lat_l"))
    a["tracking_by_hold"]["back"]["cmd_vx_mean"] = drv.WALKING_CMD_MIN + 0.01   # commanded, never reached
    a["tracking_by_hold"]["diag"]["cmd_vx_mean"] = drv.WALKING_CMD_MIN          # exactly at the floor
    out = drv.ratio_by_hold(a)
    assert out["back"] is None                       # named, with no number -- not dropped
    assert "diag" not in out                         # at the floor is not above it
    assert set(out) == {"fwd", "lat_l", "back"}


# ------------------------------------------------------------------ telemetry collection
def test_track_ratio_series_reads_the_move_gated_key_too(drv, tmp_path):
    """The plain ratio includes the ~20 % move+turn rows, so a successful facing fix LOWERS it by
    design; the move-gated twin is the cross-check that says whether translation really fell."""
    log = tmp_path / "t.log"
    log.write_text("\n".join([
        "Metrics/base_parkour/track_ratio_mean: 0.400",
        "Metrics/base_parkour/track_ratio_move_mean: 0.700",
        "Metrics/base_parkour/track_ratio_mean: 0.500",
        "Metrics/base_parkour/track_ratio_move_mean: 0.900",
    ]))
    assert drv.track_ratio_series(log) == [0.4, 0.5]
    assert drv.track_ratio_series(log, key="track_ratio_move_mean") == [0.7, 0.9]
    assert drv.track_ratio_tail(drv.track_ratio_series(log, key="track_ratio_move_mean")) == pytest.approx(0.8)
    assert drv.track_ratio_series(log, key="not_a_metric") == []


def test_track_ratio_series_default_is_bitwise_todays_label(drv):
    """The default must not move: every recorded gate verdict was read off this exact label."""
    path = TRACK_RATIO_FIXTURES / "seed3_1a_track_ratio.log"
    assert drv.track_ratio_series(path) == drv.series(path, drv.TRACK_RATIO_LABEL)
    assert drv.TRACK_RATIO_LABEL == "Metrics/base_parkour/track_ratio_mean"


def _omni_extra_keys() -> tuple[str, ...]:
    """``OMNI_EXTRA_KEYS`` read out of the ledger's SOURCE (no torch, no Isaac import)."""
    import ast
    src = (OMNI / "mdp/omni_exposure.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OMNI_EXTRA_KEYS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("OMNI_EXTRA_KEYS is no longer a literal assignment in omni_exposure.py")


def test_every_omni_ledger_key_is_collected_by_the_driver(drv):
    """CROSS-PIN. A key the omni exposure ledger emits but EXPO_KEYS does not list is printed every
    iteration and read by nobody: it never reaches the record, and the omission is invisible."""
    extra = _omni_extra_keys()
    assert "track_ratio_move_mean" in extra                      # the key this pin was added for
    assert set(extra) <= set(drv.EXPO_KEYS), set(extra) - set(drv.EXPO_KEYS)


def test_exposure_from_log_collects_the_pool_telemetry_by_prefix(drv, tmp_path):
    """The pool's type set is a knob, so the labels are collected by prefix: a type added to a pool
    is reported without a driver change, and a log with no pool yields no pool keys at all."""
    log = tmp_path / "pool.log"
    log.write_text("\n".join([
        "Metrics/base_parkour/track_ratio_mean: 0.50",
        "Metrics/base_velocity/pool_fwd_frac: 0.40",
        "Metrics/base_velocity/pool_fwd_draws: 0.38",
        "Metrics/base_velocity/pool_side_l_turn_frac: 0.10",
        "Metrics/base_velocity/pool_side_l_turn_draws: 0.12",
        "Metrics/base_velocity/stand_frac_actual: 0.20",
        "Metrics/base_velocity/pool_fwd_frac: 0.60",
        "Metrics/base_velocity/pool_fwd_draws: 0.42",
    ]))
    ex = drv.exposure_from_log(log)
    assert ex["pool_fwd_frac"] == pytest.approx(0.5)            # tail mean of 0.40, 0.60
    assert ex["pool_fwd_draws"] == pytest.approx(0.40)
    assert ex["pool_side_l_turn_frac"] == pytest.approx(0.10)   # multi-underscore type names survive
    assert ex["stand_frac_actual"] == pytest.approx(0.20)       # the slot fractions are untouched
    assert not [k for k in ex if k.startswith("pool_stand")]     # a type the pool never declared
    plain = tmp_path / "nopool.log"
    plain.write_text("Metrics/base_parkour/track_ratio_mean: 0.50\n")
    assert not [k for k in drv.exposure_from_log(plain) if k.startswith("pool_")]


# ------------------------------------------------------------------ latest_run_ckpt by run name
def test_expo_lines_render_a_missing_move_gated_ratio_as_n_a(drv):
    """Every campaign recorded before ``track_ratio_move_mean`` existed re-renders its REPORT.md
    through this function; a missing number must read as the report's own "n/a", not as the string
    "None" (``fmt`` only knows the nan convention, and ``dict.get`` returns None)."""
    line = [ln for ln in drv.expo_lines({"track_ratio_mean": 0.763}) if "move rows only" in ln]
    assert line and "move rows only n/a" in line[0]
    assert "move rows only None" not in line[0]
    assert "move rows only 0.700" in drv.expo_lines({"track_ratio_move_mean": 0.7})[1]


def test_latest_run_ckpt_resolves_the_head_by_run_name(drv, tmp_path):
    """Two chains can share an experiment name; "newest directory" would then hand one chain the
    other's checkpoint as its own head -- a lineage error no later record could detect."""
    exp = tmp_path / "crab_hex_omni_flat_walk"
    mine, theirs = exp / "2026-01-01_00-00-00_mytag", exp / "2026-01-01_00-00-05_other"
    for d, its in ((mine, (10, 999)), (theirs, (10, 4999))):
        d.mkdir(parents=True)
        for it in its:
            (d / f"model_{it}.pt").write_text("x")
    os.utime(mine, (1_000_000, 1_000_000))
    os.utime(theirs, (2_000_000, 2_000_000))          # the sibling is NEWER
    assert drv.latest_run_ckpt(exp, 0.0, "mytag") == str(mine / "model_999.pt")
    assert drv.latest_run_ckpt(exp, 0.0) == str(theirs / "model_4999.pt")     # unchanged without a name
    assert drv.latest_run_ckpt(exp, 3_000_000.0, "mytag") is None             # nothing newer than t0
    assert drv.latest_run_ckpt(exp, 0.0, "absent") is None                    # no run of that name
    assert drv.latest_run_ckpt(tmp_path / "nope", 0.0, "mytag") is None


# ------------------------------------------------------------------ Campaign: atomic save, set_seed
def _campaign(drv, tmp_path, seed="3"):
    return drv.Campaign(tmp_path / "camp", "A15+B", seed)


def test_campaign_save_is_atomic(drv, tmp_path, monkeypatch):
    """save() runs inside the poll loop and right before every stop: a truncated state.json is the
    campaign's only record of what ran. Serialise first, write a temp file, then os.replace."""
    c = _campaign(drv, tmp_path)
    c.recs["1a"] = {"ckpt": "/logs/run/model_999.pt", "status": "ok"}
    seen: list = []

    def spy_replace(src, dst):
        seen.append((str(src), str(dst)))
        return os.replace(src, dst)

    # Patched on the DRIVER's ``os`` binding, not on the shared ``os`` module: patching
    # ``drv.os.replace`` would swap the real ``os.replace`` for every other test (and every other
    # thread) in this process for the duration of this one.
    monkeypatch.setattr(drv, "os", types.SimpleNamespace(
        replace=spy_replace, **{k: v for k, v in vars(os).items()
                                if not k.startswith("__") and k != "replace"}))
    c.save()
    good = c.state_path.read_text()
    assert json.loads((c.dir / "heads.json").read_text()) == {"seed3": {"1a": "/logs/run/model_999.pt"}}
    # Both files land through a rename, which is atomic within a directory: a reader (or a
    # relaunch after a SIGKILL) sees the previous file or the new one, never a truncated one.
    assert [Path(dst).name for _src, dst in seen] == ["state.json", "heads.json"]
    assert all(src.endswith(".tmp") for src, _dst in seen)
    assert not list(c.dir.glob("*.tmp"))
    # ... and serialisation happens BEFORE anything is opened, so an unserialisable record raises
    # with both files still intact rather than half-written.
    seen.clear()
    c.st["broken"] = object()
    with pytest.raises(TypeError):
        c.save()
    assert seen == []
    assert c.state_path.read_text() == good
    assert not list(c.dir.glob("*.tmp"))


def test_set_seed_moves_the_record_key_with_the_seed(drv, tmp_path):
    """Both or neither: a seed changed alone trains seed 2 and files the record under seed 3."""
    c = _campaign(drv, tmp_path)
    c.recs["1a"] = {"ckpt": "/a/model_1.pt"}
    assert drv.set_seed(c, 2) is c
    assert (c.seed, c.key) == ("2", "seed2")
    assert c.recs == {}                            # a different seed's records
    c.recs["1a"] = {"ckpt": "/b/model_2.pt"}
    c.save()
    assert set(json.loads(c.state_path.read_text())["keys"]) == {"seed3", "seed2"}


# ------------------------------------------------------------------ the subprocess layer
# A real child, not a mocked Popen: what is pinned here is what happens to a PROCESS -- that it is
# SIGTERMed rather than SIGKILLed (Kit corrupts its shader cache when killed mid-write, after which
# every later start balloons until the machine OOMs), that a stop reaches it within one poll tick,
# and that a child which exited on its own during a shutdown is not mistaken for an infra death and
# relaunched. The poll interval and the retry wait are monkeypatched down; each test is ~1 s.
FAKE_CHILD = textwrap.dedent('''
    import json, os, signal, sys, time
    mark = os.environ["FAKE_MARK"]

    def _term(signum, frame):
        with open(mark + ".sig", "w") as fh:
            fh.write(str(signum))
        os._exit(143)

    # installed FIRST: a test may signal immediately. FAKE_IGNORE_TERM stands in for a child that
    # does NOT die on SIGTERM (a wedged Kit), the only way to reach _stop_child's SIGKILL half.
    signal.signal(signal.SIGTERM,
                  signal.SIG_IGN if os.environ.get("FAKE_IGNORE_TERM") else _term)
    with open(mark + ".argv", "w") as fh:
        json.dump(sys.argv, fh)
    with open(mark + ".env", "w") as fh:
        json.dump({k: v for k, v in os.environ.items()
                   if k.startswith(("KRABBY_", "PYTHON", "FAKE_"))}, fh)
    with open(mark + ".launches", "a") as fh:
        fh.write("launched\\n")
    for pair in [p for p in os.environ.get("FAKE_CKPTS", "").split(",") if p]:
        d, model = pair.split("|")           # a trainer's log dir + the checkpoint it leaves there
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, model), "w") as fh:
            fh.write("ckpt")
        os.utime(d, None)                    # the parent resolves the head by dir mtime vs its t0
        time.sleep(0.02)                     # ... and by ORDER, so keep the two apart
    out = os.environ.get("FAKE_EVAL_OUT", "")
    if out:                                  # one eval run directory, as the harness leaves it
        os.makedirs(out, exist_ok=True)
        with open(os.environ["FAKE_EVAL_JSON"]) as src, \\
                open(os.path.join(out, "scenario_metrics.json"), "w") as dst:
            dst.write(src.read())
        if os.environ.get("FAKE_EVAL_COMPLETE"):   # a suite row's LAST write (plan D10's marker)
            with open(os.path.join(out, "complete.json"), "w") as fh:
                fh.write(os.environ["FAKE_EVAL_COMPLETE"])
    with open(mark + ".ready", "w") as fh:
        fh.write("ready")
    line = os.environ.get("FAKE_LINE", "")
    for _ in range(int(os.environ.get("FAKE_REPEAT", "0"))):
        print(line, flush=True)
    watch = os.environ.get("FAKE_EXIT_WHEN", "")
    if watch:                                # exit the moment the parent creates this file
        while not os.path.exists(watch):
            time.sleep(0.005)
    else:
        time.sleep(float(os.environ.get("FAKE_SLEEP", "30")))
    sys.exit(int(os.environ.get("FAKE_RC", "0")))
''')


class _Child:
    """The fake child script plus the marker files it writes."""

    def __init__(self, tmp_path: Path):
        self.script = tmp_path / "fake_child.py"
        self.script.write_text(FAKE_CHILD)
        self.mark = tmp_path / "child"

    def _p(self, suffix: str) -> Path:
        return Path(str(self.mark) + suffix)

    @property
    def argv(self) -> list:
        return json.loads(self._p(".argv").read_text())

    @property
    def env(self) -> dict:
        return json.loads(self._p(".env").read_text())

    @property
    def launches(self) -> int:
        f = self._p(".launches")
        return len(f.read_text().splitlines()) if f.exists() else 0

    @property
    def signalled(self) -> int | None:
        f = self._p(".sig")
        return int(f.read_text()) if f.exists() else None

    def ready(self) -> bool:
        return self._p(".ready").exists()


@pytest.fixture
def child(tmp_path, monkeypatch):
    ch = _Child(tmp_path)
    monkeypatch.setenv("FAKE_MARK", str(ch.mark))
    return ch


def _patch_subprocess_layer(drv, monkeypatch, tmp_path):
    """Point the driver at the fake child and shrink every wait to test scale."""
    monkeypatch.setattr(drv, "PY", sys.executable)
    monkeypatch.setattr(drv, "LOG_ROOT", tmp_path / "rsl_logs")
    monkeypatch.setattr(drv, "POLL_S", 0.05)
    monkeypatch.setattr(drv, "EVAL_POLL_S", 0.05)
    monkeypatch.setattr(drv, "TERM_GRACE_S", 5)
    monkeypatch.setattr(drv, "RETRY_WAIT_S", 0)
    monkeypatch.setattr(drv, "_wait_isaac_clear", lambda *a, **k: None)


def test_stop_child_escalates_to_sigkill_only_after_a_real_grace(drv, tmp_path, monkeypatch, child):
    """The OTHER half of the SIGTERM-before-SIGKILL rule, which every other subprocess test misses
    because the fake child dies on the SIGTERM it is sent.

    A Kit child that does NOT die on SIGTERM must still be reaped: without the escalation
    ``proc.wait(timeout=grace)`` raises ``TimeoutExpired`` out of ``_stop_child``, which inside
    ``train``'s poll loop lands in the generic handler -> ``_stop_child`` again -> raises again and
    escapes the function, leaving the driver dead and a live Kit process holding the GPU. And the
    grace must be a real, finite wait: an immediate ``kill()`` is the cache-corrupting bug this
    whole path exists to avoid.
    """
    monkeypatch.setattr(drv, "TERM_GRACE_S", 0.3)
    monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
    monkeypatch.setenv("FAKE_SLEEP", "30")
    proc = subprocess.Popen([sys.executable, str(child.script)])
    try:
        for _ in range(500):                       # wait until it is past its own signal setup
            if child.ready():
                break
            time.sleep(0.01)
        assert child.ready()
        t0 = time.time()
        drv._stop_child(proc)
        elapsed = time.time() - t0
        rc = proc.poll()                           # captured BEFORE the cleanup below reaps it
    finally:
        if proc.poll() is None:                    # never leave a child behind on a failure
            proc.kill()
            proc.wait(timeout=10)
    assert rc == -int(signal.SIGKILL)              # it was reaped, and only by the second signal
    assert child.signalled is None                 # SIG_IGN: the handler never ran
    assert 0.3 <= elapsed < 10                     # the grace was waited out, not skipped


def test_train_passes_the_overlay_the_run_name_and_an_unbuffered_child(drv, tmp_path, monkeypatch, child):
    """The seam a per-experiment campaign needs: its own env overlay on top of the two fixed keys,
    a named run directory, and a child whose prints reach the log as they happen (the poll loop
    reads that FILE -- a block-buffered child looks stalled, which the watchdog calls infra)."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0.05")
    c = _campaign(drv, tmp_path)
    spec = drv.ph.PHASES["1a"]
    overlay = {"KRABBY_OMNI_CMD_POOL": "fwd:0.8,turn:0.2", "KRABBY_OMNI_SLOTS": "0.2:0.8:0:0"}
    ckpt, status, log = drv.train(c, spec, "seed3_C1_1a", 10, None, soundness=False,
                                  extra_env=overlay, run_name="seed3_C1_1a_A15pB")
    assert (ckpt, status) == (None, "infra")            # no checkpoint under the patched LOG_ROOT
    argv = child.argv
    assert argv[argv.index("--run_name") + 1] == "seed3_C1_1a_A15pB"
    assert argv[argv.index("--task") + 1] == spec.task and argv[argv.index("--seed") + 1] == "3"
    env = child.env
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["KRABBY_OMNI_PHASE"] == "1a" and env["KRABBY_PLANT"] == "A15+B"
    assert env["KRABBY_OMNI_CMD_POOL"] == "fwd:0.8,turn:0.2"
    # the record carries the overlay, so an interrupted window can be read back with what it ran
    assert c.st["current"]["extra_env"] == overlay
    assert log == c.logs / "seed3_C1_1a_train.log"


def test_train_without_the_new_kwargs_names_no_run_and_carries_no_overlay(drv, tmp_path, monkeypatch, child):
    """Backward compatibility: the phase chain's own call sites are unchanged."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0.05")
    c = _campaign(drv, tmp_path)
    drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False)
    assert "--run_name" not in child.argv
    assert c.st["current"]["extra_env"] == {}
    assert set(child.env) - {"PYTHONUNBUFFERED"} == {"KRABBY_OMNI_PHASE", "KRABBY_PLANT",
                                                     "FAKE_MARK", "FAKE_SLEEP"}


def test_train_num_envs_comes_from_the_overlay_only_when_set(drv, tmp_path, monkeypatch, child):
    """``KRABBY_NUM_ENVS`` (PPO-construction plan step 2, DEFAULT OFF): the training command's
    ``--num_envs`` is the stage overlay's value when the knob is set, and the stock ``"256"``
    otherwise -- with no ``[KNOB]`` line in the log, so a knob-off window's log is unchanged."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0.05")
    spec = drv.ph.PHASES["1a"]
    assert drv.NUM_ENVS == {"rl": "256"} and drv.NUM_ENVS_VAR == "KRABBY_NUM_ENVS"
    c = _campaign(drv, tmp_path)
    _ck, _st, log = drv.train(c, spec, "off", 10, None, False, extra_env={"KRABBY_OMNI_SLOTS": "0.2:0.8:0:0"})
    argv = child.argv
    assert argv[argv.index("--num_envs") + 1] == "256"
    assert "[KNOB]" not in log.read_text()
    _ck, _st, log = drv.train(c, spec, "on", 10, None, False, extra_env={"KRABBY_NUM_ENVS": "1024"})
    argv = child.argv
    assert argv[argv.index("--num_envs") + 1] == "1024"
    assert log.read_text().splitlines()[0] == "[KNOB] KRABBY_NUM_ENVS=1024 consumed"
    assert child.env["KRABBY_NUM_ENVS"] == "1024"          # carried (inert) like any overlay key
    assert drv._train_num_envs(spec, {"KRABBY_NUM_ENVS": " "}) == "256"
    for bad in ("0", "-4", "1.5", "lots"):
        with pytest.raises(ValueError):
            drv._train_num_envs(spec, {"KRABBY_NUM_ENVS": bad})


def test_train_never_lets_an_overlay_unbuffer_the_child(drv, tmp_path, monkeypatch, child):
    """``PYTHONUNBUFFERED`` is set AFTER the overlay merge, so it is not an overlay's to change.

    It is deliberately not in ``TRAIN_ENV_RESERVED``: that list refuses keys whose overlay makes the
    RECORD lie (the preset, the plant). This one is about the watchdog -- a block-buffered child
    writes nothing to the log file the poll loop stats, so ``STALL_S`` calls a perfectly healthy
    window an infra death and relaunches it, and ``dead_walker`` sees no ``track_ratio`` prints.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0.05")
    c = _campaign(drv, tmp_path)
    drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False,
              extra_env={"PYTHONUNBUFFERED": "0", "KRABBY_OMNI_SLOTS": "0.2:0.8:0:0"})
    assert child.env["PYTHONUNBUFFERED"] == "1"                    # the overlay did not win
    assert child.env["KRABBY_OMNI_SLOTS"] == "0.2:0.8:0:0"         # ... and the rest of it did


@pytest.mark.parametrize("key", ["KRABBY_OMNI_PHASE", "KRABBY_PLANT", "KRABBY_PHASE", "KRABBY_SIDE_PHASE"])
def test_train_refuses_an_overlay_that_reaches_for_the_drivers_own_keys(drv, tmp_path, monkeypatch, child, key):
    """A window whose preset or plant came from an overlay is a window whose record lies about what
    it ran; the other two raise at config import (one task per process). Refused BEFORE launch."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    c = _campaign(drv, tmp_path)
    with pytest.raises(AssertionError) as exc:
        drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False, extra_env={key: "x"})
    assert key in str(exc.value)
    assert child.launches == 0


def test_train_does_not_launch_when_a_stop_is_already_pending(drv, tmp_path, monkeypatch, child):
    """A stop that arrived while the previous phase was being recorded must not commit the GPU."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    c = _campaign(drv, tmp_path)
    ckpt, status, _log = drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False,
                                   interrupt_check=lambda: "STOP requested")
    assert (ckpt, status) == (None, "interrupted")
    assert child.launches == 0


def test_train_interrupt_sigterms_the_child_within_one_tick(drv, tmp_path, monkeypatch, child):
    """SIGTERM, never SIGKILL: Kit flushes its DerivedDataCache on the way out, and a cache killed
    mid-write makes every later Kit start balloon until the machine OOMs.

    Also pins the registration itself, WHILE the child is alive: the campaign's SIGTERM handler
    reaches a running child only through ``rp.CURRENT_CHILD``, so a slot that is never filled turns
    ``systemctl --user stop`` into a signal sent to nothing -- and the only visible symptom is an
    emergency stop that hangs a live GPU window until the 120-s tick notices. Asserting the slot is
    ``None`` after the call cannot see that; asserting it during the tick can.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    c = _campaign(drv, tmp_path)
    live: list = []
    t0 = time.time()

    def check():
        if not child.ready():
            return None
        live.append(drv.CURRENT_CHILD)       # captured on the tick that stops it: the live child
        return "STOP"

    ckpt, status, _log = drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False,
                                   interrupt_check=check)
    assert (ckpt, status) == (None, "interrupted")      # never "infra": infra triggers a relaunch
    assert child.signalled == int(signal.SIGTERM)
    assert time.time() - t0 < 10                        # one tick + the child's own exit
    assert live and live[-1] is not None                # the slot held a child, not None
    assert list(live[-1].args)[:2] == [sys.executable, str(child.script)]   # ... and it was THIS one
    assert drv.CURRENT_CHILD is None                    # the slot is cleared for the next child


def test_train_reports_interrupted_when_the_child_exited_between_two_ticks(drv, tmp_path, monkeypatch, child):
    """The normal shape of a signalled shutdown: the campaign's SIGTERM handler kills the child
    itself, so the poll loop finds it already gone. Without the after-loop check that window is
    classified "no checkpoint" -> infra -> a two-hour relaunch in the middle of a shutdown.

    Sequenced so ONLY the after-loop check can fire: the flag is still clear on the tick during
    which the child exits (the check itself releases the child and waits for it to go), and turns
    truthy only on the NEXT call -- which is the one after the poll loop.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    release = tmp_path / "release"
    monkeypatch.setenv("FAKE_EXIT_WHEN", str(release))
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] == 2:            # the second call is the first in-loop tick
            release.write_text("go")   # let the child exit, and wait until it has
            time.sleep(0.5)
            return None                # ... with the flag still CLEAR on this tick
        return "STOP" if calls["n"] > 2 else None

    c = _campaign(drv, tmp_path)
    ckpt, status, _log = drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False, interrupt_check=check)
    assert (ckpt, status) == (None, "interrupted")     # NOT "infra": infra relaunches the window
    assert calls["n"] == 3                             # pre-launch, one tick, the after-loop check
    assert child.launches == 1 and child.signalled is None     # it exited on its own


def test_train_ok_returns_its_own_run_head_not_a_siblings(drv, tmp_path, monkeypatch, child):
    """The ok path, and the reason ``run_name`` exists AT THE CALL SITE.

    Two chains of one ladder share ``spec.experiment``, so "the newest directory under it" can be
    the OTHER chain's window -- and the head recorded for this experiment would then be a
    checkpoint it never produced, a lineage error no later record could detect. Every other train
    test ends with ``ckpt is None`` (the fake child writes no checkpoint), so without this one the
    third argument of ``latest_run_ckpt(..., t0, run_name)`` could be deleted with the suite green.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0.05")
    spec = drv.ph.PHASES["1a"]
    exp = tmp_path / "rsl_logs" / spec.experiment
    mine = exp / "2026-01-01_00-00-00_seed3_C1_1a_A15pB"
    theirs = exp / "2026-01-01_00-00-05_seed3_C2_1a_A15pB"       # a SIBLING chain, and newer
    monkeypatch.setenv("FAKE_CKPTS", f"{mine}|model_999.pt,{theirs}|model_4999.pt")
    c = _campaign(drv, tmp_path)
    ckpt, status, _log = drv.train(c, spec, "tag", 10, None, False, run_name="seed3_C1_1a_A15pB")
    assert status == "ok"
    assert ckpt == str(mine / "model_999.pt")
    # ... and with no run_name the old rule is untouched: the newest directory wins.
    ckpt2, status2, _log2 = drv.train(c, spec, "tag", 10, None, False)
    assert (status2, ckpt2) == ("ok", str(theirs / "model_4999.pt"))


def test_train_sigterms_the_child_on_an_unexpected_exception_and_on_a_signal(drv, tmp_path, monkeypatch, child):
    """Both non-local exits from the poll loop leave a LIVE Kit child behind, and neither may
    SIGKILL it: that is how the DerivedDataCache gets corrupted mid-write (2026-08-13, 2026-09-18).

    An ordinary exception is still this function's verdict to give (``infra``, or ``interrupted``
    if a stop is pending); a ``KeyboardInterrupt`` is not -- it stops the child and propagates,
    because the ``finally`` that clears ``CURRENT_CHILD`` is about to remove the only handle a
    signal handler would have used to reach it.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    spec = drv.ph.PHASES["1c"]

    def boom(*_a, **_kw):
        raise ValueError("an unexpected shape in the log")

    monkeypatch.setattr(drv, "dead_walker", boom)          # raises INSIDE the poll tick
    c = _campaign(drv, tmp_path)
    ckpt, status, _log = drv.train(c, spec, "tag", 5000, None, False, early_abort=True)
    assert (ckpt, status) == (None, "infra")
    assert child.signalled == int(signal.SIGTERM)          # never kill(): Kit is mid-write
    assert drv.CURRENT_CHILD is None

    (tmp_path / "second").mkdir()
    ch2 = _Child(tmp_path / "second")
    monkeypatch.setenv("FAKE_MARK", str(ch2.mark))
    monkeypatch.setattr(drv, "TRAIN", ch2.script)
    calls = {"n": 0}

    def signalled_check():
        calls["n"] += 1
        if calls["n"] > 1:                                  # not on the pre-launch look
            raise KeyboardInterrupt
        return None

    with pytest.raises(KeyboardInterrupt):
        drv.train(c, spec, "tag", 10, None, False, interrupt_check=signalled_check)
    assert ch2.signalled == int(signal.SIGTERM)
    assert drv.CURRENT_CHILD is None


def test_train_early_abort_floor_can_be_overridden_per_run(drv, tmp_path, monkeypatch, child):
    """A campaign that MEASURES the dead-walker floor off its own baseline passes the measured
    number; the module constant stays the phase chain's default."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_LINE", "Metrics/base_parkour/track_ratio_mean: 0.550")
    monkeypatch.setenv("FAKE_REPEAT", "1000")
    monkeypatch.setenv("FAKE_SLEEP", "30")
    c = _campaign(drv, tmp_path)
    ckpt, status, _log = drv.train(c, drv.ph.PHASES["1c"], "tag", 5000, None, False,
                                   early_abort=True, abort_track_ratio_min=0.6)
    assert status == "dead_walker" and ckpt is None            # 0.550 over its 800-1000 < 0.6
    orch = (c.dir / "orchestrator.log").read_text()
    assert "DEAD WALKER" in orch and "0.550" in orch and "need >= 0.6" in orch
    assert 0.550 >= drv.ABORT_TRACK_RATIO_MIN                  # the DEFAULT floor would not have fired


def test_train_early_abort_default_floor_is_unchanged(drv, tmp_path, monkeypatch, child):
    """Same log, no override: the run is left alone (and ends as an ordinary infra death here,
    since the fake child writes no checkpoint)."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setenv("FAKE_LINE", "Metrics/base_parkour/track_ratio_mean: 0.550")
    monkeypatch.setenv("FAKE_REPEAT", "1000")
    monkeypatch.setenv("FAKE_SLEEP", "0.4")
    c = _campaign(drv, tmp_path)
    _ckpt, status, _log = drv.train(c, drv.ph.PHASES["1c"], "tag", 5000, None, False, early_abort=True)
    assert status == "infra"
    assert "DEAD WALKER" not in (c.dir / "orchestrator.log").read_text()


def test_train_retry_retries_infra_once_and_never_during_a_stop(drv, tmp_path, monkeypatch):
    """Relaunching a two-hour window is the one thing a stop must never cause: an infra verdict
    reached during a shutdown is almost always the shutdown itself."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    calls: list = []

    def fake_train(c, spec, tag, *a, **kw):
        calls.append(tag)
        return None, "infra", tmp_path / "x.log"

    monkeypatch.setattr(drv, "train", fake_train)
    c = _campaign(drv, tmp_path)
    spec = drv.ph.PHASES["1a"]
    assert drv.train_retry(c, spec, "tag", 10, None, False)[1] == "infra"
    assert calls == ["tag", "tag_r2"]                      # unchanged without an interrupt check
    calls.clear()
    _ckpt, status, _lp = drv.train_retry(c, spec, "tag", 10, None, False, interrupt_check=lambda: "STOP")
    assert calls == ["tag"] and status == "infra"          # the caller reclassifies a stop
    calls.clear()
    seen = {"n": 0}

    def flips():
        seen["n"] += 1
        return "STOP" if seen["n"] > 1 else None           # clear at the verdict, set during the wait

    _ckpt, status, _lp = drv.train_retry(c, spec, "tag", 10, None, False, interrupt_check=flips)
    assert calls == ["tag"] and status == "interrupted"


def test_train_retry_does_not_sit_out_the_retry_pause_after_a_stop(drv, tmp_path, monkeypatch):
    """The pause before the relaunch is TICKED, like ``_run_eval``'s: a stop arriving inside it ends
    the call at once instead of two minutes later.

    The verdict was already right without this (the check after the wait reclassifies), so what is
    pinned is the BUDGET: the plan gives the whole shutdown 360 s (tick <= 120 + grace <= 120 +
    save), and a slept two minutes spent here on top of a stopped window is how a clean
    ``systemctl --user stop`` turns into systemd's SIGKILL -- the DerivedDataCache hazard.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "RETRY_WAIT_S", 20.0)
    calls: list = []

    def fake_train(c_, spec_, tag, *a, **kw):
        calls.append(tag)
        return None, "infra", tmp_path / "x.log"

    monkeypatch.setattr(drv, "train", fake_train)
    state = {"t": None}

    def flips():
        if state["t"] is None:                 # clear at the verdict ...
            state["t"] = time.time() + 0.25
            return None
        return "STOP" if time.time() >= state["t"] else None      # ... set INSIDE the wait

    t0 = time.time()
    _ckpt, status, _lp = drv.train_retry(_campaign(drv, tmp_path), drv.ph.PHASES["1a"], "tag", 10,
                                         None, False, interrupt_check=flips)
    assert calls == ["tag"] and status == "interrupted"
    assert time.time() - t0 < 10                                  # not the patched 20-s wait


def test_train_retry_never_relaunches_a_verdict_that_is_not_infra(drv, tmp_path, monkeypatch):
    """``infra`` is the ONLY retryable verdict, and the check is on the status, not on the flag.

    The stop test above reaches the no-retry outcome through the interrupt check, so widening the
    condition to ``status in ("infra", "interrupted")`` survives it: here the flag is CLEAR and the
    verdict is ``interrupted`` (a stop the campaign has already consumed the reason for), which must
    still not commit two more GPU hours. ``dead_walker`` and ``nan`` are not retryable either -- they
    are verdicts about the policy, and re-running them is how a floor gets averaged away.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    c, spec = _campaign(drv, tmp_path), drv.ph.PHASES["1a"]
    for verdict in ("interrupted", "nan", "dead_plant"):
        calls: list = []

        def fake_train(c_, spec_, tag, *a, _v=verdict, **kw):
            calls.append(tag)
            return None, _v, tmp_path / "x.log"

        monkeypatch.setattr(drv, "train", fake_train)
        _ckpt, status, _lp = drv.train_retry(c, spec, "tag", 10, None, False,
                                             interrupt_check=lambda: None)   # flag CLEAR
        assert calls == ["tag"], f"{verdict} was relaunched"
        assert status == verdict                       # and the verdict is handed back unchanged


def test_wait_isaac_clear_returns_before_it_polls_when_a_stop_is_pending(drv, monkeypatch):
    """The wait exists to protect the NEXT launch; during a shutdown there is no next launch, and
    ten minutes of 10-s ``pgrep`` ticks would eat the whole 360-s stop budget the driver needs to
    save its state -- handing the unit to systemd's SIGKILL, which is the DerivedDataCache
    corruption this module's SIGTERM discipline exists to avoid."""
    seen = {"n": 0}

    def fake_run(*_a, **_kw):
        seen["n"] += 1
        return types.SimpleNamespace(stdout="")        # "no other Isaac process"

    # a module-local shim, not the shared subprocess module: nothing else in the process may see it
    monkeypatch.setattr(drv, "subprocess", types.SimpleNamespace(
        run=fake_run, Popen=subprocess.Popen, TimeoutExpired=subprocess.TimeoutExpired,
        STDOUT=subprocess.STDOUT))
    t0 = time.time()
    assert drv._wait_isaac_clear(lambda: "STOP") is None
    assert seen["n"] == 0                              # it did not even ask
    assert time.time() - t0 < 5.0                      # ... let alone sit out a 10-s cycle
    # ... and the early return is about the STOP, not a general skip: with no check it still polls.
    drv._wait_isaac_clear()
    assert seen["n"] == 1


def test_train_retry_forwards_every_seam_kwarg(drv, tmp_path, monkeypatch):
    seen: list = []

    def fake_train(c, spec, tag, iters, resume, soundness, early_abort=False, **kw):
        seen.append(kw)
        return None, "infra", tmp_path / "x.log"

    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "train", fake_train)
    kwargs = {"extra_env": {"KRABBY_OMNI_CMD_POOL": "fwd:1.0"}, "run_name": "tagx",
              "abort_track_ratio_min": 0.61}
    drv.train_retry(_campaign(drv, tmp_path), drv.ph.PHASES["1a"], "tag", 10, None, False, **kwargs)
    assert len(seen) == 2                                  # the original and its retry
    for got in seen:
        assert {k: got[k] for k in kwargs} == kwargs


# ------------------------------------------------------------------ _run_eval: status, rc, stop
def _run_one_eval(drv, c, tmp_path, **kw):
    return drv._run_eval(c, "tag", "slow__omni", ["--manifest", "m"], {"KRABBY_PLANT": "A15+B"},
                         tmp_path / "out", "A15+B", kw.pop("timeout", 3600), **kw)


def _eval_payload(hold_labels, episodes, aggregate, usd="assets/crab.usda") -> dict:
    """A ``scenario_metrics.json`` of the shape the harness writes."""
    return {"aggregate": dict(aggregate), "episodes": list(episodes),
            "run_meta": {"hold_labels": list(hold_labels), "dt": 0.02, "usd_path": usd}}


def _arm_eval_output(drv, tmp_path, monkeypatch, payload, run="2026-09-18_00-00-00", sid="slow__omni"):
    """Make the fake eval child leave one fresh run directory, and return its path."""
    run_dir = tmp_path / "out" / sid / "seed001" / run
    src = tmp_path / f"metrics_{run}.json"
    src.write_text(json.dumps(payload))
    monkeypatch.setenv("FAKE_EVAL_OUT", str(run_dir))
    monkeypatch.setenv("FAKE_EVAL_JSON", str(src))
    return run_dir


def test_run_eval_files_a_fresh_row_and_injects_survival_by_hold(drv, tmp_path, monkeypatch, child):
    """The SUCCESS path -- the contract the Popen rewrite exists to preserve, and the one every
    other test in this block skips (they all end in "none" or "interrupted").

    A regression that left ``status`` at "none" on a good row would pass the rest of the suite and
    then halt the campaign on ``eval_infra`` with a perfectly healthy eval on disk; one that swapped
    the two outputs would file ``run_dir: "None"`` and no milestone could re-score the row.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    hold = next(iter(drv._manifest_hold_s("slow__omni")))       # the row's own first hold
    run_dir = _arm_eval_output(drv, tmp_path, monkeypatch, _eval_payload(
        [hold], [{"n_steps": 10 ** 6}, {"n_steps": 0}],          # one episode reached it, one fell
        {"n_episodes": 2, "schedule_completion_rate": 0.5}))
    c = _campaign(drv, tmp_path)
    latest, agg, info = _run_one_eval(drv, c, tmp_path)
    assert info == {"status": "ok", "rc": 0, "attempts": 1}      # one child, clean exit, no retry
    assert latest == run_dir and agg["schedule_completion_rate"] == 0.5
    # survival THROUGH the hold, injected in memory by the driver (it is not in the harness's file)
    assert agg["survival_by_hold"][hold] == pytest.approx(0.5)
    assert drv.LAST_EVAL_INFO == info and drv.CURRENT_CHILD is None
    orch = c.dir / "orchestrator.log"                                # a clean row logs nothing
    assert "no fresh metrics" not in (orch.read_text() if orch.exists() else "")


def test_run_eval_plant_guard_leaves_this_rows_return_code_readable(drv, tmp_path, monkeypatch, child):
    """The plant guard raises THROUGH ``_run_eval`` (a row scored on the wrong plant is not a
    number to retry), and the infra detector reading ``LAST_EVAL_INFO`` afterwards must see this
    row's child, not the previous row's."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    drv.LAST_EVAL_INFO = {"status": "none", "rc": -9, "attempts": 3}      # a previous row's
    _arm_eval_output(drv, tmp_path, monkeypatch, _eval_payload(
        ["stand"], [], {"n_episodes": 1}, usd="assets/variants/crab_simple__splay00_axis5p5in.usda"))
    with pytest.raises(RuntimeError, match="rung-iv"):
        _run_one_eval(drv, _campaign(drv, tmp_path), tmp_path)
    assert drv.LAST_EVAL_INFO == {"status": "none", "rc": 0, "attempts": 1}
    assert drv.CURRENT_CHILD is None


def test_run_eval_stop_during_the_retry_wait_is_not_a_failed_row(drv, tmp_path, monkeypatch, child):
    """The two-minute pause between attempts is TICKED, not slept: a stop arriving inside it ends
    the call as ``interrupted``.

    Slept, the flag would only be seen by the next attempt's top-of-loop check -- so the row would
    be filed as "produced nothing", which is an infra question the driver answers by retrying or
    halting, in the middle of a shutdown.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setattr(drv, "RETRY_WAIT_S", 20.0)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    monkeypatch.setenv("FAKE_RC", "3")
    c = _campaign(drv, tmp_path)
    orch, state = c.dir / "orchestrator.log", {"t_stop": None}

    def check():
        if state["t_stop"] is None:
            # flips only once the attempt has already failed, i.e. INSIDE the wait that follows
            if orch.exists() and "no fresh metrics" in orch.read_text():
                state["t_stop"] = time.time() + 0.25
            return None
        return "STOP" if time.time() >= state["t_stop"] else None

    t0 = time.time()
    latest, agg, info = _run_one_eval(drv, c, tmp_path, max_attempts=3, interrupt_check=check)
    assert (latest, agg) == (None, None)
    assert info == {"status": "interrupted", "rc": 3, "attempts": 1}   # no second child was started
    assert child.launches == 1
    assert time.time() - t0 < 10                                       # it did not sit out the wait


def test_run_eval_reports_the_childs_return_code_when_nothing_was_produced(drv, tmp_path, monkeypatch, child):
    """`(None, None)` conflated "ran and produced nothing" with "was killed": the infra detector
    needs the return code (-9 = the OOM killer / the Kit cache pathology -> stop retrying)."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    monkeypatch.setenv("FAKE_RC", "7")
    latest, agg, info = _run_one_eval(drv, _campaign(drv, tmp_path), tmp_path)
    assert (latest, agg) == (None, None)
    assert info == {"status": "none", "rc": 7, "attempts": 3}      # the unchanged 3-attempt cadence
    assert child.launches == 3
    assert drv.LAST_EVAL_INFO == info                              # readable after eval_morph -> None
    assert drv.CURRENT_CHILD is None


def test_run_eval_honours_max_attempts(drv, tmp_path, monkeypatch, child):
    """The campaign driver owns the retry (it runs its own infra check between attempts) -- and the
    call returns as soon as the verdict is in: the retry pause is for the NEXT attempt, so the last
    one does not sit out two minutes of dead time the driver is waiting on."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setattr(drv, "RETRY_WAIT_S", 30.0)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    t0 = time.time()
    _latest, _agg, info = _run_one_eval(drv, _campaign(drv, tmp_path), tmp_path, max_attempts=1)
    assert info["attempts"] == 1 and child.launches == 1
    assert time.time() - t0 < 10


def test_run_eval_does_not_launch_when_a_stop_is_already_pending(drv, tmp_path, monkeypatch, child):
    """``train``'s twin, and the same reason: a stop that landed while the previous row was being
    recorded must not boot Kit again.

    Without the before-``Popen`` check the after-loop one still reports ``interrupted``, so the
    verdict looks right -- but a 40-minute eval was launched, ran for a tick and was terminated,
    burning a Kit start (and a DerivedDataCache write, the hazard) inside the 360-s stop budget,
    invisibly.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    c = _campaign(drv, tmp_path)
    latest, agg, info = _run_one_eval(drv, c, tmp_path, interrupt_check=lambda: "STOP requested")
    assert (latest, agg) == (None, None)
    assert info == {"status": "interrupted", "rc": None, "attempts": 0}   # nothing was launched
    assert child.launches == 0
    assert not (c.logs / "tag_slow__omni_eval.log").exists()      # not even the log was opened


def test_run_eval_refuses_a_run_directory_older_than_this_attempt(drv, tmp_path, monkeypatch, child):
    """The freshness gate: a directory left by an EARLIER row is not this row's result.

    Every other success-path test hands ``_run_eval`` a directory the child has just written, so the
    ``st_mtime >= t0`` comparison is never exercised on a stale one -- and a child that dies before
    writing anything would otherwise be scored on whatever the previous eval left behind, which is a
    verdict about a different checkpoint.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    stale = tmp_path / "out" / "slow__omni" / "seed001" / "2026-09-01_00-00-00"
    stale.mkdir(parents=True)
    (stale / "scenario_metrics.json").write_text(json.dumps(_eval_payload(
        ["stand"], [], {"n_episodes": 9, "schedule_completion_rate": 1.0})))
    old = time.time() - 3600
    os.utime(stale / "scenario_metrics.json", (old, old))
    os.utime(stale, (old, old))                      # ... written an hour before this attempt
    c = _campaign(drv, tmp_path)
    latest, agg, info = _run_one_eval(drv, c, tmp_path, max_attempts=1)
    assert (latest, agg) == (None, None)             # the stale row is not adopted
    assert info["status"] == "none" and info["attempts"] == 1
    # ... and the campaign log does not promise a retry this call will not make: the line is what a
    # human reads to tell a one-attempt eval_infra halt from a three-attempt one.
    orch = (c.dir / "orchestrator.log").read_text()
    assert "no fresh metrics (rc 0) — no attempts left" in orch
    assert "retry" not in orch


def test_run_eval_stop_terminates_the_child_and_never_retries(drv, tmp_path, monkeypatch, child):
    """A stopped row is not evidence, so there is nothing to retry -- and the stop must hold even
    if the flag has cleared again by the time the attempt loop comes round (the campaign may have
    consumed the reason). The check below fires exactly ONCE for that reason.

    The eval half of the ``CURRENT_CHILD`` registration is pinned here too, WHILE the child runs:
    the campaign's SIGTERM handler has no other handle on a live eval, so an unfilled slot means a
    ``systemctl --user stop`` that signals nothing and a 40-minute eval that runs on.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    fired = {"n": 0}
    live: list = []

    def check_once():
        if child.ready() and not fired["n"]:
            fired["n"] = 1
            live.append(drv.CURRENT_CHILD)
            return "STOP"
        return None

    latest, agg, info = _run_one_eval(drv, _campaign(drv, tmp_path), tmp_path, interrupt_check=check_once)
    assert (latest, agg) == (None, None)
    assert info["status"] == "interrupted" and info["attempts"] == 1
    assert child.launches == 1 and child.signalled == int(signal.SIGTERM)
    assert live and live[-1] is not None
    assert list(live[-1].args)[:2] == [sys.executable, str(child.script)]
    assert drv.CURRENT_CHILD is None


def test_run_eval_timeout_terminates_before_it_kills(drv, tmp_path, monkeypatch, child):
    """THE deliberate behaviour change (2026-09-18): the timeout path used to SIGKILL immediately,
    which is exactly how the Kit shader cache gets corrupted."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    t0 = time.time()
    _latest, _agg, info = _run_one_eval(drv, _campaign(drv, tmp_path), tmp_path, timeout=0.5,
                                        max_attempts=1)
    assert child.signalled == int(signal.SIGTERM)
    assert info["status"] == "none" and info["rc"] == 143          # the child's own SIGTERM exit
    assert time.time() - t0 < 10


# ------------------------------------------------------------------ eval_morph / eval_obst
def _stub_run_eval(drv, monkeypatch, result, seen=None):
    def _f(*a, **kw):
        if seen is not None:
            seen.append(kw)
        return result

    monkeypatch.setattr(drv, "_run_eval", _f)


@pytest.mark.parametrize("row", ["morph", "obst"])
def test_eval_rows_still_return_dict_or_none_and_carry_the_eval_status(drv, tmp_path, monkeypatch, row):
    """run_phase assigns these straight into ``rec["evals"][...]``; the contract may not change.
    An INTERRUPTED row returns a marker dict -- distinguishable from None (the eval ran and
    produced nothing: an infra question) and from a summary (it produced metrics)."""
    c, spec = _campaign(drv, tmp_path), drv.ph.PHASES["1c"]
    call = ((lambda: drv.eval_morph(c, spec, "/a/model_1.pt", "slow", "tag")) if row == "morph"
            else (lambda: drv.eval_obst(c, spec, "/a/model_1.pt", "tag")))
    agg = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING)
    _stub_run_eval(drv, monkeypatch, (tmp_path / "rd", agg, {"status": "ok", "rc": 0, "attempts": 1}))
    out = call()
    assert out["run_dir"] == str(tmp_path / "rd")
    assert out["eval_status"] == "ok" and out["eval_rc"] == 0
    assert out["completion"] == pytest.approx(0.85)
    _stub_run_eval(drv, monkeypatch, (None, None, {"status": "none", "rc": -9, "attempts": 3}))
    assert call() is None
    _stub_run_eval(drv, monkeypatch, (None, None, {"status": "interrupted", "rc": -15, "attempts": 1}))
    assert call() == {"eval_status": "interrupted", "run_dir": None}


@pytest.mark.parametrize("row", ["morph", "obst"])
def test_eval_rows_forward_the_overlay_and_the_stop_check(drv, tmp_path, monkeypatch, row):
    seen: list = []
    _stub_run_eval(drv, monkeypatch, (None, None, {"status": "none", "rc": 1, "attempts": 1}), seen)
    captured: list = []
    real_env_for = drv.eval_env_for

    def spy_env(spec, plant, **kw):
        captured.append(kw.get("carry_overlay"))
        return real_env_for(spec, plant, **kw)

    monkeypatch.setattr(drv, "eval_env_for", spy_env)
    c, spec = _campaign(drv, tmp_path), drv.ph.PHASES["1c"]
    overlay = {"KRABBY_OMNI_FACING_OBS_ENCODING": "sincos"}
    stop = object()
    if row == "morph":
        drv.eval_morph(c, spec, "/a/model_1.pt", "slow", "tag", carry_overlay=overlay,
                       interrupt_check=stop, max_attempts=1)
    else:
        drv.eval_obst(c, spec, "/a/model_1.pt", "tag", carry_overlay=overlay,
                      interrupt_check=stop, max_attempts=1)
    assert captured == [overlay]
    assert seen[0]["interrupt_check"] is stop and seen[0]["max_attempts"] == 1


# ------------------------------------------------------------------ the ladder-directory refusal
def test_main_refuses_a_candidate_ladder_campaign_directory(tmp_path):
    """The ladder keeps its ledger in the same state.json this driver rewrites on every save: one
    run of the phase chain in that directory would drop the key and the campaign's decisions with
    it. Refused loudly, naming the driver that owns the directory, before anything is written."""
    d = tmp_path / "camp"
    d.mkdir()
    state = {"campaign": "camp", "plant": "A15+B", "phase": "run", "run_no": 0, "keys": {},
             "current": {}, "ladder": {"candidates": ["C1"], "ledger": []}}
    d.joinpath("state.json").write_text(json.dumps(state))
    out = _cli("--campaign-dir", str(d), "--phases", "1a")
    assert out.returncode != 0
    assert "ladder" in out.stderr and "run_candidates.py" in out.stderr
    assert json.loads(d.joinpath("state.json").read_text()) == state     # untouched
    assert not (d / "REPORT.md").exists()
    # "before anything is written" includes the campaign's own scaffolding: the guard reads the
    # file itself rather than loading a Campaign, whose constructor would create logs/ first.
    assert sorted(p.name for p in d.iterdir()) == ["state.json"]


def test_interruptible_sleep_returns_on_the_flag_not_on_the_deadline(drv):
    """The retry pause's primitive: it may not outlive a stop, and it may not end early without
    one (a wait that returns immediately would turn the retry cadence into a hot loop)."""
    t0 = time.time()
    assert drv._interruptible_sleep(5.0, lambda: "STOP") is True         # already set ...
    assert time.time() - t0 < 1.0                                        # ... and it cost nothing
    t0 = time.time()
    assert drv._interruptible_sleep(0.2, None, tick=0.02) is False
    assert 0.2 <= time.time() - t0 < 5.0
    calls = {"n": 0}

    def flips():
        calls["n"] += 1
        return "STOP" if calls["n"] > 2 else None

    t0 = time.time()
    assert drv._interruptible_sleep(30.0, flips, tick=0.02) is True      # set DURING the wait
    assert time.time() - t0 < 5.0


def test_run_eval_after_loop_check_catches_a_child_killed_by_the_shutdown(drv, tmp_path, monkeypatch, child):
    """The shape of a systemd stop: the campaign's own handler signals the child, so the poll loop
    finds it already gone. Without the after-loop check the attempt reads as "no fresh metrics" and
    relaunches an eval in the middle of the shutdown."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    release = tmp_path / "release"
    monkeypatch.setenv("FAKE_EXIT_WHEN", str(release))
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] == 2:              # the first in-loop tick: let the child go, flag still clear
            release.write_text("go")
            time.sleep(0.5)
            return None
        return "STOP" if calls["n"] > 2 else None

    c = _campaign(drv, tmp_path)
    _latest, _agg, info = _run_one_eval(drv, c, tmp_path, interrupt_check=check)
    assert info["status"] == "interrupted" and info["attempts"] == 1
    assert calls["n"] == 3 and child.launches == 1 and child.signalled is None
    # ... and the attempt is not FILED as a failure: a stopped row is not a row that failed, and
    # the record a user reads at 3 a.m. must not claim the eval produced nothing on its own.
    orch = c.dir / "orchestrator.log"
    assert "no fresh metrics" not in (orch.read_text() if orch.exists() else "")


# ------------------------------------------------- the Kit-clear wait after the LAST failed attempt
def _spy_isaac_wait(drv, monkeypatch) -> list:
    """Replace ``_wait_isaac_clear`` with a counter, and record the stop check each call was handed.

    Installed AFTER ``_patch_subprocess_layer``, whose own no-op stub it replaces (the real one
    pgreps for Isaac processes, which in a test run would find none and return at once -- so it
    cannot distinguish "called" from "not called").
    """
    seen: list = []

    def spy(interrupt_check=None):
        seen.append(interrupt_check)

    monkeypatch.setattr(drv, "_wait_isaac_clear", spy)
    return seen


def test_run_eval_waits_for_isaac_after_every_failed_attempt_including_the_last(
        drv, tmp_path, monkeypatch, child):
    """HEAD ran ``_wait_isaac_clear()`` after EVERY failed attempt, the third included, and the
    interrupt-aware rewrite must too.

    The wait is not for this row's retry -- it is for the NEXT ROW. ``eval_morph(slow)`` ->
    ``eval_morph(light)`` -> ``eval_obst(ring)`` run back to back inside ``run_phase`` and
    ``_run_eval`` never waits BEFORE an attempt, so a row that returns the instant its third attempt
    fails fires the next row's ``Popen`` while the timed-out Kit is still inside its SIGTERM grace,
    holding the one GPU. Only the ``RETRY_WAIT_S`` PAUSE may be skipped there.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    seen = _spy_isaac_wait(drv, monkeypatch)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setattr(drv, "RETRY_WAIT_S", 0)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    monkeypatch.setenv("FAKE_RC", "5")            # exits non-zero at once, writing no metrics
    c = _campaign(drv, tmp_path)

    def never():                                  # a stop check that never fires; its IDENTITY is
        return None                               # what the assertions below look for

    _latest, _agg, info = _run_one_eval(drv, c, tmp_path, max_attempts=3, interrupt_check=never)
    assert info == {"status": "none", "rc": 5, "attempts": 3}
    assert child.launches == 3
    # one per attempt, the LAST one included -- and each handed the stop check, so a shutdown does
    # not sit through ten minutes of pgrep (that is _wait_isaac_clear's own early return).
    assert seen == [never, never, never]
    orch = (c.dir / "orchestrator.log").read_text()
    assert "no attempts left" in orch and orch.count("no fresh metrics") == 3
    # ... and the same holds for the driver-owned single-attempt shape, where the only attempt IS
    # the last one: Part B calls _run_eval with max_attempts=1 in a loop of its own.
    seen.clear()
    _latest, _agg, info = _run_one_eval(drv, c, tmp_path, max_attempts=1, interrupt_check=never)
    assert info["attempts"] == 1 and seen == [never]


def test_run_eval_does_not_wait_for_isaac_once_the_stop_has_fired(drv, tmp_path, monkeypatch, child):
    """The interrupted path is the one place the wait is still skipped outright: there is no next
    row to protect during a shutdown, and the campaign's stop budget is what the ten minutes would
    be spent out of. Both shapes of a stop are pinned -- one that was already pending before the
    launch, and one that arrives while the child runs."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    seen = _spy_isaac_wait(drv, monkeypatch)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    c = _campaign(drv, tmp_path)
    _latest, _agg, info = _run_one_eval(drv, c, tmp_path, interrupt_check=lambda: "STOP")
    assert info == {"status": "interrupted", "rc": None, "attempts": 0}
    assert seen == [] and child.launches == 0             # nothing launched, so nothing to wait for

    fired = {"n": 0}

    def check_once():
        if child.ready() and not fired["n"]:
            fired["n"] = 1
            return "STOP"
        return None

    _latest, _agg, info = _run_one_eval(drv, c, tmp_path, interrupt_check=check_once)
    assert info["status"] == "interrupted" and info["attempts"] == 1
    assert child.launches == 1 and child.signalled == int(signal.SIGTERM)
    assert seen == []                                     # the stop short-circuits before the wait


# ------------------------------------------------- Ctrl-C on an attached driver stops the child
class _SleepInterrupt:
    """Stands in for the driver module's ``time``; its FIRST ``sleep`` is a Ctrl-C, the rest are real.

    Only ``run_phases.time`` is rebound, never the stdlib module itself: ``_stop_child`` reaps the
    child through ``Popen.wait(timeout=...)``, which busy-waits on ``subprocess``'s own ``time.sleep``
    -- patching the module object globally would make the reaper raise instead of reaping.
    """

    def __init__(self, real, on_first=None, exc=None):
        self._real = real
        self.sleeps = 0
        self._on_first = on_first
        self._exc = exc if exc is not None else KeyboardInterrupt("Ctrl-C on an attached driver")

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps == 1:
            if self._on_first is not None:
                self._on_first()
            raise self._exc
        return self._real.sleep(seconds)

    def __getattr__(self, name):             # time.time() and anything else the driver reaches for
        return getattr(self._real, name)


def test_train_ctrl_c_stops_the_training_child_before_it_propagates(drv, tmp_path, monkeypatch, child):
    """DELIBERATE CHANGE from HEAD, pinned on the DEFAULT argument path (no ``interrupt_check``): a
    Ctrl-C raised inside ``train``'s poll loop SIGTERMs the training child and then re-raises.

    HEAD's ``except Exception`` let the interrupt unwind past a live child, leaving an orphaned
    256-env Kit holding the one GPU with no supervisor left to reap it -- the worse failure, and the
    reason the handler is kept. It is now spelled ``except (KeyboardInterrupt, SystemExit)`` rather
    than ``except BaseException`` so that the narrowing is visible next to the ``except Exception``
    infra path it sits beside, and this test is what stops either one from swallowing the other: the
    exception must reach the caller, and the child must be dead when it does. Under ``systemd-run``
    this path is unreachable (a SIGTERM to the driver raises no Python exception; the stop arrives
    through ``interrupt_check``), so an attached terminal is its whole blast radius.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setattr(drv, "TERM_GRACE_S", 2)          # keeps the test under 5 s even if it hangs
    monkeypatch.setenv("FAKE_SLEEP", "30")               # a real, long-lived child
    c = _campaign(drv, tmp_path)
    live: list = []

    def grab():
        for _ in range(500):                             # let it install its SIGTERM handler first
            if child.ready():
                break
            time.sleep(0.01)
        live.append(drv.CURRENT_CHILD)

    monkeypatch.setattr(drv, "time", _SleepInterrupt(time, grab))
    try:
        with pytest.raises(KeyboardInterrupt):            # re-raised, not turned into a verdict
            drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False)
        assert live and live[0] is not None               # the loop was reached with a live child
        proc = live[0]
        deadline = time.time() + 3.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        assert proc.poll() is not None                    # ... and it is dead by the time we look
        assert child.signalled == int(signal.SIGTERM)     # never kill(): Kit is mid-cache-write
        assert drv.CURRENT_CHILD is None                  # the finally still ran on the way out
    finally:
        for p in live:                                    # never leave a child behind on a failure
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=10)


def test_train_system_exit_in_the_poll_loop_also_stops_the_child(drv, tmp_path, monkeypatch, child):
    """``SystemExit`` is the second half of the narrowed tuple, and it is not academic: the driver's
    own guards raise it (``_refuse_ladder_dir``), and ``BaseException`` used to cover it for free.

    Narrowed to ``(KeyboardInterrupt, SystemExit)``, a spelling that forgot this one would leave a
    live 256-env Kit behind on every ``sys.exit`` taken while a window is running -- while
    ``except Exception`` above, being unchanged, would not catch it either.
    """
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "TRAIN", child.script)
    monkeypatch.setattr(drv, "TERM_GRACE_S", 2)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    c = _campaign(drv, tmp_path)
    live: list = []

    def grab():
        for _ in range(500):
            if child.ready():
                break
            time.sleep(0.01)
        live.append(drv.CURRENT_CHILD)

    monkeypatch.setattr(drv, "time", _SleepInterrupt(time, grab, exc=SystemExit("bailing out")))
    try:
        with pytest.raises(SystemExit):
            drv.train(c, drv.ph.PHASES["1a"], "tag", 10, None, False)
        assert live and live[0] is not None
        proc = live[0]
        deadline = time.time() + 3.0
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        assert proc.poll() is not None and child.signalled == int(signal.SIGTERM)
        assert drv.CURRENT_CHILD is None
    finally:
        for p in live:
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=10)


# ================================================================== the training seam (plan 2026-09-21, D8)
#: run_phases.py sits on the EVAL side of the ladder's hashes (``EVAL_PATHSPECS``, D8): an edit to
#: its eval half moves ``eval_sha`` and nothing else. Its TRAINING half would then move neither hash,
#: so it is pinned here instead -- the source of every function the training command line, the
#: training env and the abort read, the ``Campaign`` record they write through, and the module
#: constants they read. A training-relevant edit must bump :data:`TRAINING_SEAM_SHA256` AND be
#: declared at relaunch with ``--accept-code-change`` (the recorded caveat 4 of the plan).
TRAINING_SEAM = ("train", "train_retry", "_subenv", "with_plant", "latest_run_ckpt", "_ckpt_iter", "set_seed",
                 "series", "tail_mean", "track_ratio_series", "dead_walker", "Campaign",
                 # PPO-construction plan step 2 (2026-10-01): the --num_envs resolver train() calls
                 "_train_num_envs")
#: ``_training_seam_pin`` over the UNEDITED run_phases.py -- ``git show 464be672:<path>`` executed
#: with ``__file__`` at its real path -- computed ONCE when the pin was written (per-category plan
#: step 3), never over the edited file: that would prove nothing.
#: BUMPED 2026-10-01 (PPO-construction plan step 2): a DELIBERATE training-seam change --
#: ``train`` takes ``--num_envs`` from the stage overlay's ``KRABBY_NUM_ENVS`` (``_train_num_envs``,
#: added to :data:`TRAINING_SEAM`) and writes a ``[KNOB]`` line when it is set. The old value
#: (17d0bdb1...) was re-verified against the unedited file before the edit; this one is computed over
#: the edited file at that change. Relaunching a campaign on it needs ``--accept-code-change``.
TRAINING_SEAM_SHA256 = "ac00c8fe1ca8210fda5550564884fe5bc2ea721cf896db7e6501f376cae90231"


def _seam_functions(obj) -> list:
    """The functions whose globals the pin reads: ``obj`` itself, or -- ``getclosurevars`` refuses a
    class -- every method, property accessor and static/class method a class defines."""
    if not inspect.isclass(obj):
        return [obj]
    out = []
    for attr in vars(obj).values():
        if isinstance(attr, (staticmethod, classmethod)):
            attr = attr.__func__
        if isinstance(attr, property):
            out += [f for f in (attr.fget, attr.fset, attr.fdel) if f is not None]
        elif inspect.isfunction(attr):
            out.append(attr)
    return out


def _global_reads(fn) -> dict:
    """``inspect.getclosurevars(fn).globals`` plus the globals read in ``fn``'s NESTED code objects:
    before Python 3.12 a comprehension is its own code object, which ``getclosurevars`` never opens
    (``train`` reads ``TRAIN_ENV_RESERVED`` inside one)."""
    out = dict(inspect.getclosurevars(fn).globals)
    stack = [c for c in fn.__code__.co_consts if inspect.iscode(c)]
    while stack:
        code = stack.pop()
        out.update({n: fn.__globals__[n] for n in code.co_names if n in fn.__globals__ and n not in out})
        stack += [c for c in code.co_consts if inspect.iscode(c)]
    return out


def _pin_value(mod, value) -> str:
    """A constant's repr with repo paths made repo-relative, so a worktree pins the same value."""
    if isinstance(value, Path):
        try:
            return f"<REPO>/{value.relative_to(mod.REPO).as_posix()}"
        except ValueError:
            return repr(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_pin_value(mod, v)}" for k, v in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}(" + ", ".join(_pin_value(mod, v) for v in value) + ")"
    return repr(value)


def _training_seam_pin(mod, source: str) -> str:
    """sha256 over the SOURCE of every :data:`TRAINING_SEAM` object (decorators included) plus a sorted
    repr of the module-level non-callable, non-module globals they read. Names any function assigns
    through ``global`` (``CURRENT_CHILD``, ``LAST_EVAL_INFO``) are runtime state and excluded."""
    tree = ast.parse(source)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    runtime_state = {name for n in ast.walk(tree) if isinstance(n, ast.Global) for name in n.names}
    lines = source.splitlines()
    parts, consts = [], {}
    for name in TRAINING_SEAM:
        node = top[name]
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        parts.append(f"## {name}\n" + "\n".join(lines[first - 1:node.end_lineno]))
        for fn in _seam_functions(getattr(mod, name)):
            for gname, value in _global_reads(fn).items():
                if gname in runtime_state or callable(value) or inspect.ismodule(value):
                    continue
                consts[gname] = _pin_value(mod, value)
    parts.append("## globals\n" + "\n".join(f"{k} = {v}" for k, v in sorted(consts.items())))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def test_run_phases_training_seam_is_unchanged(drv):
    """The per-category plan edits run_phases.py's EVAL half only (``eval_morph``'s two keywords, the
    suite-aware manifest readers, own-hold survival, ``summarize()['cat']``, the completion marker):
    the training command line, env composition, abort parsers, record and constants hash exactly
    as they did at the pre-change HEAD (``464be672``)."""
    assert _training_seam_pin(drv, DRIVER.read_text()) == TRAINING_SEAM_SHA256


def test_the_training_seam_pin_moves_with_the_source_and_with_a_constant(drv, monkeypatch):
    """The pin is not vacuous: a one-character source edit, a changed constant a seam function
    reads (including one read only inside a comprehension), and a changed ``Campaign`` method each
    move it; an eval-side edit does not."""
    src = DRIVER.read_text()
    base = _training_seam_pin(drv, src)
    assert base != _training_seam_pin(drv, src.replace('"--max_iterations", str(iters)',
                                                         '"--max_iterations", str(iters + 0)', 1))
    assert base != _training_seam_pin(drv, src.replace("def save(self) -> None:",
                                                         "def save(self) -> None:  # x", 1))
    assert base == _training_seam_pin(drv, src.replace("def eval_morph(", "def eval_morph( ", 1))
    for name, value in (("NUM_ENVS", {"rl": "512"}), ("STALL_S", 901), ("TRAIN_ENV_RESERVED", ("KRABBY_PLANT",)),
                        ("LOG_ROOT", drv.PARKOUR / "logs/other")):
        with monkeypatch.context() as m:
            m.setattr(drv, name, value)
            assert _training_seam_pin(drv, src) != base, name
    with monkeypatch.context() as m:                  # runtime state is not a constant
        m.setattr(drv, "CURRENT_CHILD", object())
        assert _training_seam_pin(drv, src) == base


# ================================================================== the per-category suite (plan 2026-09-21, B2)
# The eval half of run_phases.py on a `kind: category_suite` row: summarize()'s additive `cat`
# (D5), the manifest read through the harness's own parser (D10), own-hold survival at every reader
# (never the cumulative walk), the completion marker, and the eval_morph seam the ladder passes
# `--trained-categories` through. The synthetic suite below is built with the HARNESS's own
# `suite_run_meta` / `report.aggregate` / `report.write_run`, so every number is the harness's.
EVAL_CATEGORIES = OMNI / "eval_categories.py"
RESOLUTION = OMNI / "experiments/tools/resolution.py"
CANDIDATES_DRIVER = OMNI / "experiments/tools/run_candidates.py"
SUITE_SID = "suite__omni"
SUITE_E = 4                                   # envs per wave here (the manifest row runs 100)
SUITE_DT = 0.02
ALL_CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
#: Per-env speed ratio on every hold commanded to move. ``lat_r``'s 0.90 outlier makes the POOLED
#: side median (0.605) differ from the mean of the two hand medians (0.515).
SUITE_RATIO = {"fwd": (0.70, 0.71, 0.72, 0.73), "lat_l": (0.60, 0.61, 0.62, 0.63),
               "lat_r": (0.40, 0.41, 0.42, 0.90), "back": (0.50, 0.52, 0.54, 0.56),
               "diag": (0.65, 0.66, 0.67, 0.68), "move_turn": (0.35, 0.36, 0.37, 0.38),
               "move_turn_r": (0.45, 0.46, 0.47, 0.48)}
#: (hold, env) -> steps short of the hold's own end. 0 = a fall ON the last step, which counts as
#: survived (the legacy convention the harness keeps, D3).
SUITE_FALLS = {("lat_r", 0): 300, ("turn_l90", 1): 50, ("move_turn_r", 2): 0}


@pytest.fixture(scope="module")
def ec():
    return _load(EVAL_CATEGORIES, "omni_eval_categories_gate_test")


def _suite_scenario(drv):
    S = drv._gait_schedule()
    scenarios, _ = S.load_manifest(drv.OMNI_MANIFEST)
    return S, next(sc for sc in scenarios if sc.id == SUITE_SID)


def _suite_hold(label: str, hold, e: int, k: int, n_steps: int) -> dict:
    """One own-hold entry of the shape ``score_episode`` writes on a world row."""
    moving = hold.speed > 0
    ratio = SUITE_RATIO[label][e] if moving else None
    entry = {"label": label, "n_steps": n_steps,
             "tripod": {"tripod_score": 0.40 + 0.02 * e + 0.01 * k, "low_confidence": False},
             "tracking": {"vx": {"cmd_mean": hold.speed, "actual_mean": (ratio * hold.speed) if moving else 0.004,
                                 "ratio": ratio}},
             "motion": {"speed_abs_mean_m_s": (ratio * hold.speed) if moving else 0.01 + 0.001 * e,
                        "pos_drift_m": 0.1 * e, "path_len_m": 1.0, "yaw_drift_rad": 0.01, "foot_lift_count": 20,
                        "contact_churn_per_s": 3.0, "n_steps": n_steps}}
    turning = abs(hold.face) > 0
    entry["facing"] = {"heading_error_rms": 0.4 if turning else 0.02,
                       "heading_error_initial_abs": 1.5708 if turning else 0.01,
                       "heading_error_final_abs": (0.30 - 0.05 * e) if turning else 0.02 + 0.001 * e,
                       "heading_error_max_abs": 1.6 if turning else 0.03,
                       "facing_closed": (0.50 + 0.10 * e) if turning else None,
                       "settle_time_s": (3.0 + e) if turning else None,
                       "yaw_rate_ratio": 0.9,
                       "yaw_progress_rate": (0.08 + 0.01 * e) if turning else 0.0,
                       "net_yaw_rate_toward": (0.07 + 0.01 * e) if turning else 0.0}
    if moving:
        entry["direction"] = {"direction_error_rad": 0.05 + 0.01 * e, "cross_track_ratio": 0.10 + 0.01 * e,
                              "travel_speed_m_s": ratio * hold.speed}
    return entry


def _suite_run(drv, trained=ALL_CATEGORIES):
    """``(scenario, run_meta, episodes, aggregate)`` of a synthetic suite run at E = 4, W = 10."""
    S, sc = _suite_scenario(drv)
    suite = sc.suite
    seeds = [sc.env_seed * S.WAVE_SEED_STRIDE + k for k in range(sc.waves)]
    run_meta = S.suite_run_meta(sc, dt=SUITE_DT, num_envs=SUITE_E, wave_seeds=seeds, trained_categories=list(trained))
    cat_of, hand_of = suite.category_of_hold(), suite.hand_of_hold()
    episodes = []
    for k, hold in enumerate(suite.holds()):
        end = S.own_hold_end_step(prologue_s=sc.get_default("prologue_s"), settle_hold_s=suite.settle_hold_s,
                                  hold_s=suite.hold_s, dt=SUITE_DT)
        for e in range(SUITE_E):
            short = SUITE_FALLS.get((hold.label, e))
            n_steps = end - (short or 0)
            own = _suite_hold(hold.label, hold, e, k, n_steps)
            settle = {"label": suite.settle_label, "n_steps": 100,
                      "tripod": {"tripod_score": None, "low_confidence": True}}
            episodes.append({"env_index": k * SUITE_E + e, "wave": k, "hold": hold.label,
                             "category": cat_of[hold.label], "hand": hand_of[hold.label], "n_steps": n_steps,
                             "termination_reason": "fall" if short is not None else "schedule_complete",
                             "completed_schedule": short is None, "tripod_score": own["tripod"]["tripod_score"],
                             "tracking_ratio": own["tracking"]["vx"]["ratio"], "tippy_tap_fraction": 0.0,
                             "slip_ratio_mean": 0.0, "facing_error_rms": 0.3,
                             "holds": {suite.settle_label: settle, hold.label: own}})
    run_meta.update({"scenario_id": SUITE_SID, "task": sc.task, "checkpoint": "/ckpt/model_1.pt",
                     "checkpoint_sha256": "ab" * 32, "dt": SUITE_DT, "hold_labels": suite.hold_labels(),
                     "command_override_max_deviation": 0.0,
                     "n_steps_per_env": [ep["n_steps"] for ep in episodes]})
    return sc, run_meta, episodes, R.aggregate(episodes, suite=R.suite_aggregate_arg(run_meta))


def _suite_stamp(drv, run_dir: Path, *, complete: bool = True) -> tuple[Path, dict]:
    """A suite stamp written by the harness's own ``write_run`` (``complete.json`` LAST, suite rows
    only); ``complete=False`` removes the marker -- a stamp killed mid-way through its raws."""
    sc, run_meta, episodes, agg = _suite_run(drv)
    R.write_run(run_dir, run_meta=run_meta, episode_metrics=episodes, raw=None, scenario=sc)
    assert (run_dir / "complete.json").exists()
    if not complete:
        (run_dir / "complete.json").unlink()
    return run_dir, agg


def test_summarize_hoists_cat_from_the_suite_block_and_is_empty_without_it(drv, ec):
    _sc, _meta, _eps, agg = _suite_run(drv)
    by_cat = agg["suite"]["by_category"]
    cat = drv.summarize(agg)["cat"]
    assert list(cat) == list(ALL_CATEGORIES)
    for c, entry in cat.items():
        # the key set is D5's, cross-pinned against the categories module's contract
        assert set(entry) == set(ec.CAT_KEYS) | {"by_hand"}, c
        for key in drv.CAT_METRIC_KEYS:
            assert entry[key] == by_cat[c][key]["median"], (c, key)       # re-read, never recomputed
        paired = c in ("side", "turn", "move_turn")
        assert entry["n"] == (2 if paired else 1) * SUITE_E and entry["n_env"] == SUITE_E
        if paired:
            assert set(entry["by_hand"]) == {"l", "r"}
            for h, hand in entry["by_hand"].items():
                assert set(hand) == set(ec.CAT_KEYS), (c, h)
                assert hand["n"] == SUITE_E and hand["n_env"] == SUITE_E
                for key in drv.CAT_METRIC_KEYS:
                    assert hand[key] == by_cat[c]["by_hand"][h][key]["median"], (c, h, key)
        else:
            assert entry["by_hand"] is None
    # the pooled median over both hands (0.605), not the mean of the hand medians (0.515)
    assert cat["side"]["ratio"] == pytest.approx(0.605)
    assert cat["side"]["by_hand"]["l"]["ratio"] == pytest.approx(0.615)
    assert cat["side"]["by_hand"]["r"]["ratio"] == pytest.approx(0.415)
    # own-hold survival, n-weighted over the hands; a fall on the last step counts as survived
    assert cat["side"]["survival"] == pytest.approx(7 / 8)
    assert cat["side"]["by_hand"]["r"]["survival"] == pytest.approx(0.75)
    assert cat["side"]["by_hand"]["l"]["survival"] == pytest.approx(1.0)
    assert cat["turn"]["survival"] == pytest.approx(7 / 8)
    assert cat["move_turn"]["survival"] == pytest.approx(1.0)
    assert cat["fwd"]["survival"] == cat["stand"]["survival"] == 1.0
    # zero-speed categories carry no translation numbers; stand's drift is there (WATCH)
    assert cat["stand"]["ratio"] is None and cat["turn"]["direction_error"] is None
    assert cat["stand"]["drift_speed"] == pytest.approx(0.0115)
    assert cat["turn"]["facing_closed"] == pytest.approx(0.65) and cat["fwd"]["facing_closed"] is None
    # every other row: the key is there and empty
    assert drv.summarize(_synthetic_aggregate(turning=True))["cat"] == {}
    assert drv.cat_summary(None) == {} and drv.summarize({}) == {}


def test_summarize_legacy_keys_unchanged_on_a_suite_aggregate(drv):
    """On a suite row every pre-existing key is computed exactly as it would be without the suite
    block (the per-hold ones are the per-HAND readings); only ``cat`` and the two conditional turn
    survivals differ."""
    _sc, _meta, _eps, agg = _suite_run(drv)
    plain = copy.deepcopy(agg)
    plain.pop("suite")
    plain["survival_by_hold"] = dict(agg["suite"]["survival_by_hold"])
    s, p = drv.summarize(agg), drv.summarize(plain)
    assert set(s) == set(p) and p["cat"] == {}
    differ = {k for k in s if s[k] != p[k]}
    assert differ == {"cat", "turn_survival_l90", "turn_survival_r90"}, differ
    assert s["survival_by_hold"] == agg["suite"]["survival_by_hold"]
    assert s["ratio_by_hold"]["lat_r"] == agg["tracking_by_hold"]["lat_r"]["ratio"]["median"]
    assert s["facing_closed_by_hold"]["turn_r90"] == agg["facing_by_hold"]["turn_r90"]["facing_closed"]["median"]


def test_turn_survivals_are_none_on_a_suite_row(drv):
    """``surv[turn_l90] / surv[back]`` means "fell DURING the turn, given it reached it" only when the
    holds are nested in one episode; a suite episode runs one hold, so both are undefined (None) --
    and a cumulative table a reader might inject beside the suite block is never read."""
    _sc, _meta, _eps, agg = _suite_run(drv)
    s = drv.summarize(agg)
    assert s["turn_survival_l90"] is None and s["turn_survival_r90"] is None
    plain = copy.deepcopy(agg)
    plain.pop("suite")
    plain["survival_by_hold"] = dict(agg["suite"]["survival_by_hold"])
    assert drv.summarize(plain)["turn_survival_l90"] == pytest.approx(0.75)   # the flag, not the data
    injected = dict(agg, survival_by_hold={h: 0.123 for h in agg["suite"]["hold_labels"]})
    s2 = drv.summarize(injected)
    assert s2["survival_by_hold"] == agg["suite"]["survival_by_hold"]
    assert s2["turn_survival_l90"] is None and s2["cat"] == s["cat"]


def test_manifest_readers_derive_a_suite_rows_holds(drv):
    """The suite row has no ``schedule:`` -- the old raw-YAML read died on it. The readers go
    through the harness's parser: the union labels, a per-label ``hold_s`` of the category holds
    (``run_meta.suite.hold_s``), ``settle_hold_s``, and OWN-hold survival ends; every legacy row
    reads exactly what its ``schedule:`` says, as before."""
    import yaml
    row = drv._manifest_row(SUITE_SID)
    labels = ["stand", "fwd", "lat_l", "lat_r", "back", "diag", "turn_l90", "turn_r90", "move_turn", "move_turn_r"]
    assert row["kind"] == "category_suite" and row["waves"] == 10
    assert row["hold_labels"] == ["settle"] + labels and row["settle_label"] == "settle"
    assert row["hold_s"] == {h: 12.0 for h in labels} and row["settle_hold_s"] == 2.0
    assert row["prologue_s"] == 1.5
    assert drv._manifest_hold_s(SUITE_SID) == {h: 12.0 for h in labels}
    assert drv._is_suite_row(SUITE_SID) is True
    # own-hold ends: round((1.5 + 2 + 12) / 0.02) = 775 for every category hold, none for settle
    assert drv._survival_end_steps(SUITE_SID, 0.02) == {h: 775 for h in labels}
    doc = yaml.safe_load(drv.OMNI_MANIFEST.read_text())
    legacy = [sc for sc in doc["scenarios"] if sc.get("kind", "schedule") == "schedule"]
    assert {sc["id"] for sc in legacy} == {"slow__omni", "light__omni", "ring__omni", "curriculum__omni"}
    for sc in legacy:
        raw = {str(h["label"]): float(h["hold_s"]) for h in sc["schedule"]}
        assert drv._manifest_hold_s(sc["id"]) == raw, sc["id"]         # the pre-suite reading
        assert drv._manifest_row(sc["id"])["settle_hold_s"] is None and not drv._is_suite_row(sc["id"])
        t, walk = 1.5, {}
        for h in sc["schedule"]:
            t += float(h["hold_s"])
            walk[str(h["label"])] = int(round(t / 0.02))
        assert drv._survival_end_steps(sc["id"], 0.02) == walk, sc["id"]
    # consistent with survival_by_hold's own walk on a legacy row
    eps = [{"n_steps": n} for n in walk.values()]
    assert drv.survival_by_hold(eps, list(walk), drv._manifest_hold_s(legacy[-1]["id"])) == \
        {h: (len(walk) - i) / len(walk) for i, h in enumerate(walk)}
    with pytest.raises(KeyError):
        drv._manifest_row("nope__omni")
    assert drv._is_suite_row("nope__omni") is False


def _suite_payload(drv, tmp_path) -> tuple[dict, dict]:
    run_dir, agg = _suite_stamp(drv, tmp_path / "harness_stamp")
    return json.loads((run_dir / "scenario_metrics.json").read_text()), agg


def test_run_eval_files_a_suite_row_only_once_its_completion_marker_exists(drv, tmp_path, monkeypatch, child):
    """D10's freshness rule: ``write_run`` writes ``complete.json`` LAST on a suite row, so a stamp
    with ``scenario_metrics.json`` but no marker was killed mid-way through its ~1000 raws and is
    never ``ok`` -- the row is re-run. With the marker it is filed, with the harness's OWN-hold
    survival. A legacy row needs no marker (``test_run_eval_files_a_fresh_row...`` above)."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    payload, agg = _suite_payload(drv, tmp_path)
    c = _campaign(drv, tmp_path)

    def run():
        return drv._run_eval(c, "tag", SUITE_SID, ["--manifest", "m"], {"KRABBY_PLANT": "A15+B"},
                             tmp_path / "out", "A15+B", 3600, max_attempts=1)

    _arm_eval_output(drv, tmp_path, monkeypatch, payload, run="2026-09-21_00-00-00", sid=SUITE_SID)
    latest, a, info = run()
    assert (latest, a) == (None, None) and info["status"] == "none"
    assert "no fresh metrics" in (c.dir / "orchestrator.log").read_text()
    monkeypatch.setenv("FAKE_EVAL_COMPLETE", json.dumps({"episodes_total": 40}))
    run_dir = _arm_eval_output(drv, tmp_path, monkeypatch, payload, run="2026-09-21_00-00-01", sid=SUITE_SID)
    latest, a, info = run()
    assert info["status"] == "ok" and latest == run_dir
    assert a["survival_by_hold"] == agg["suite"]["survival_by_hold"]


def test_run_eval_refuses_a_suite_aggregate_without_the_marker_even_if_the_manifest_lookup_failed(
        drv, tmp_path, monkeypatch, child):
    """Belt and braces: the marker rule also keys off the stamp's own ``aggregate.suite``, so an
    unreadable manifest cannot let a partial suite stamp through."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    payload, _agg = _suite_payload(drv, tmp_path)
    monkeypatch.setattr(drv, "_is_suite_row", lambda sid: (_ for _ in ()).throw(ValueError("bad manifest")))
    c = _campaign(drv, tmp_path)
    _arm_eval_output(drv, tmp_path, monkeypatch, payload, run="2026-09-21_00-00-02", sid=SUITE_SID)
    latest, a, info = drv._run_eval(c, "tag", SUITE_SID, ["--manifest", "m"], {"KRABBY_PLANT": "A15+B"},
                                    tmp_path / "out", "A15+B", 3600, max_attempts=1)
    assert (latest, a) == (None, None) and info["status"] == "none"
    log = (c.dir / "orchestrator.log").read_text()
    assert "manifest unreadable" in log and "no complete.json" in log


def _spy_calls(monkeypatch, *mods) -> list:
    """Record every call of ``survival_by_hold`` (the cumulative walk) on each module's copy."""
    calls: list = []
    for mod in mods:
        real = mod.survival_by_hold

        def spy(*a, _real=real, **k):
            calls.append(a[1] if len(a) > 1 else k.get("hold_labels"))
            return _real(*a, **k)

        monkeypatch.setattr(mod, "survival_by_hold", spy)
    return calls


def _site_run_eval(drv, tmp_path, monkeypatch, child):
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    monkeypatch.setenv("FAKE_EVAL_COMPLETE", "{}")
    payload, _agg = _suite_payload(drv, tmp_path)
    _arm_eval_output(drv, tmp_path, monkeypatch, payload, run="2026-09-21_00-00-03", sid=SUITE_SID)
    calls = _spy_calls(monkeypatch, drv)
    _latest, a, info = drv._run_eval(_campaign(drv, tmp_path), "tag", SUITE_SID, ["--manifest", "m"],
                                     {"KRABBY_PLANT": "A15+B"}, tmp_path / "out", "A15+B", 3600, max_attempts=1)
    assert info["status"] == "ok"
    return drv.summarize(a), a["survival_by_hold"], calls


def _site_resolution(drv, tmp_path, monkeypatch, child):
    res = _load(RESOLUTION, "omni_resolution_gate_test")
    run_dir, _agg = _suite_stamp(drv, tmp_path / "stamp")
    calls = _spy_calls(monkeypatch, res.rp)
    summary = res._summary_with_survival(run_dir, SUITE_SID)
    return summary, summary["survival_by_hold"], calls


def _site_run_candidates(drv, tmp_path, monkeypatch, child):
    rc = _load(CANDIDATES_DRIVER, "omni_run_candidates_gate_test")
    run_dir, _agg = _suite_stamp(drv, tmp_path / "stamp")
    calls = _spy_calls(monkeypatch, rc.rp, rc.resolution_tool().rp)
    summary = rc.summarize_run_dir(run_dir, SUITE_SID)
    return summary, summary["survival_by_hold"], calls


@pytest.mark.parametrize("site", [
    "run_phases._run_eval",
    "resolution._summary_with_survival",
    "run_candidates.summarize_run_dir",
])
def test_no_reader_injects_cumulative_survival_on_a_suite_row(drv, tmp_path, monkeypatch, child, site):
    """D10's three injection sites: every reader of a suite row takes ``aggregate.suite.
    survival_by_hold`` (own-hold) and never calls the cumulative walk, whose union-label reading
    would count the settle hold and every earlier wave's label."""
    fn = {"run_phases._run_eval": _site_run_eval, "resolution._summary_with_survival": _site_resolution,
          "run_candidates.summarize_run_dir": _site_run_candidates}[site]
    _sc, _meta, _eps, agg = _suite_run(drv)
    summary, surv, calls = fn(drv, tmp_path, monkeypatch, child)
    assert calls == [], f"{site} walked the suite row cumulatively: {calls}"
    assert surv == agg["suite"]["survival_by_hold"]
    assert summary["survival_by_hold"] == agg["suite"]["survival_by_hold"]
    assert summary["turn_survival_l90"] is None and summary["turn_survival_r90"] is None
    assert set(summary["cat"]) == set(ALL_CATEGORIES)     # key order: the stamp is written sort_keys


def test_eval_morph_forwards_trained_categories_and_extra_args(drv, tmp_path, monkeypatch, child):
    """``--trained-categories`` (suite rows; a CSV or any iterable of names) and ``extra_args`` go
    AFTER the fixed harness tail, and a call with neither launches exactly the command it always
    did -- checked on the real argv of a real child, not a mock."""
    _patch_subprocess_layer(drv, monkeypatch, tmp_path)
    monkeypatch.setattr(drv, "EVAL", child.script)
    monkeypatch.setenv("FAKE_SLEEP", "0")
    monkeypatch.setattr(drv, "EVAL_ROOT", tmp_path / "evals")
    c, spec = _campaign(drv, tmp_path), drv.ph.PHASES["1c"]
    out_root = tmp_path / "evals" / c.dir.name / "tag"
    fixed = [str(child.script), "--headless", "--manifest", str(drv.OMNI_MANIFEST), "--scenario", "slow__omni",
             "--checkpoint", "/a/model_1.pt", "--save-raw", "--no-plot", "--output-root", str(out_root)]
    assert drv.eval_morph(c, spec, "/a/model_1.pt", "slow", "tag", max_attempts=1) is None
    assert child.argv == fixed                                  # byte-identical default command

    seen: list = []

    def fake(c_, tag, sid, cmd_tail, env, out_root_, plant, timeout, **kw):
        seen.append((sid, list(cmd_tail)))
        return None, None, {"status": "none", "rc": 1, "attempts": 1}

    monkeypatch.setattr(drv, "_run_eval", fake)
    head = ["--manifest", str(drv.OMNI_MANIFEST), "--scenario", SUITE_SID, "--checkpoint", "/a/m.pt", "--save-raw"]
    drv.eval_morph(c, spec, "/a/m.pt", "suite", "tag", trained_categories="stand,fwd,side,back,diag,turn,move_turn",
                   extra_args=("--episodes", "7", "--waves", 10))
    assert seen[-1] == (SUITE_SID, head + ["--trained-categories", "stand,fwd,side,back,diag,turn,move_turn",
                                           "--episodes", "7", "--waves", "10"])
    drv.eval_morph(c, spec, "/a/m.pt", "suite", "tag", trained_categories=["stand", "fwd"])
    assert seen[-1][1] == head + ["--trained-categories", "stand,fwd"]
    drv.eval_morph(c, spec, "/a/m.pt", "suite", "tag", trained_categories=frozenset({"turn", "fwd"}))
    assert seen[-1][1] == head + ["--trained-categories", "fwd,turn"]
    drv.eval_morph(c, spec, "/a/m.pt", "suite", "tag", extra_args=["--waves", "10"])
    assert seen[-1][1] == head + ["--waves", "10"]
    drv.eval_morph(c, spec, "/a/m.pt", "suite", "tag")
    assert seen[-1][1] == head
