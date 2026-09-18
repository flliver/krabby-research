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
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
DRIVER = OMNI / "experiments/tools/run_phases.py"
TABLE = OMNI / "experiments/2026-09-16_1805_omni_formation/decision_table.py"
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
def test_gate_facing_passes_when_both_turn_holds_are_precise(drv):
    facing = {"turn_l90": _facing(final=0.10, settle=2.0, progress=0.10),
              "turn_r90": _facing(final=0.30, settle=2.5, progress=0.09)}   # final exactly at the floor
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=facing))
    status, msg = drv.gate_facing_verdict(s)
    assert status == "ok", msg


@pytest.mark.parametrize("final_l, final_r", [(0.31, 0.10), (0.10, 0.45)])
def test_gate_facing_fails_when_either_turn_hold_overshoots(drv, final_l, final_r):
    facing = {"turn_l90": _facing(final=final_l, settle=2.0, progress=0.10),
              "turn_r90": _facing(final=final_r, settle=2.0, progress=0.09)}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing=facing))
    status, msg = drv.gate_facing_verdict(s)
    assert status == "gate_facing" and "FAILED" in msg


def test_gate_facing_no_usable_number_is_a_halt(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.9, facing={"turn_l90": _facing(0.1, 1.0, 0.08)}))  # turn_r90 missing
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
        "KRABBY_OMNI_FACING_OBS_CLIP",   # obs channels 6/7 (facing error) clip
        "KRABBY_OMNI_SPEED_ELLIPSE",     # _write_command's body command on a world-frame row
        "KRABBY_OMNI_ANG_VEL_MAX",       # heading_rate_command's yaw-rate saturation
        "KRABBY_OMNI_CLOCK_TURN_R",      # omni_clock_turn_radius -> the observed clock phase
        "KRABBY_OMNI_YAW_RANGE",         # the reset event's spawn-yaw band
    }
    # Sampling / reward / terrain knobs are NOT carried: the canary is a minimal flat eval whose
    # commands come from the manifest schedule override and whose verdict must not move when a
    # stage re-weights a reward.
    for banned in ("KRABBY_OMNI_SLOTS", "KRABBY_OMNI_DIR_BAND", "KRABBY_OMNI_FACING",
                   "KRABBY_OMNI_RSI_SOURCE", "KRABBY_YAW_W", "KRABBY_YAW_PROGRESS_W",
                   "KRABBY_STAND_FRAC", "KRABBY_RESAMPLE_S"):
        assert banned not in drv.EVAL_CARRY


@pytest.mark.parametrize("row", ["canary", "obstacle"])
def test_every_eval_env_carries_every_eval_carry_key_from_the_preset(drv, row):
    """EVAL_CARRY reaches BOTH eval rows, with the PHASE preset's value (not a class default)."""
    pe = drv.ph.phase_env("1c")
    ev = _eval_envs(drv)[row]
    for key in drv.EVAL_CARRY:
        assert key in pe, f"{key} is not in the 1c preset -- carry list is stale"
        assert ev.get(key) == pe[key], f"{row}: {key} missing or not the preset value"
    # The specific regression: the facing-error observation clip the 2026-09-17 yaw gate turned on.
    assert ev["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6"


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
