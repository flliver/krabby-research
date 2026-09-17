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
          scored_holds: tuple[str, ...] | None = None, ratio: dict[str, float] | None = None) -> dict:
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
    return a


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
    status, msg = drv.gate_soft_verdict(s, hold, drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION)
    assert status == expect, msg
    if expect == "halt":
        assert "FAILED" in msg and "--continue" in msg


def test_soft_gate_reads_only_its_own_hold(drv):
    """1a's soft gate must not be dragged down by lat_l/diag/back being unformed (1a never samples
    those slots): a low value everywhere except 'fwd' still passes the 1a gate."""
    tripod = {"stand": 0.01, "fwd": 0.45, "lat_l": 0.02, "diag": 0.01, "back": 0.01, "move_turn": 0.0,
              "turn_l90": 0.0, "turn_r90": 0.0}
    s = drv.summarize(_agg(tripod, 0.85, facing=None, walking_holds=("fwd",), scored_holds=("stand", "fwd")))
    status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION)
    assert status == "ok", msg


def test_soft_gate_no_usable_number_is_a_halt(drv):
    for s in ({}, drv.summarize(None), drv.summarize(_agg(HEALTHY_TRIPOD, None, facing=HEALTHY_FACING))):
        status, msg = drv.gate_soft_verdict(s, "fwd", drv.GATE_SOFT_TRIPOD, drv.GATE_SOFT_COMPLETION)
        assert status == "halt" and "could not be evaluated" in msg


# ------------------------------------------------------------------ gate_1c_verdict
def test_gate_1c_passes_on_a_healthy_head(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "ok", msg
    assert "yaw-progress rate" in msg and "turn_l90 0.080" in msg and "net" in msg


@pytest.mark.parametrize("bad_hold, bad_value", [("fwd", 0.39), ("lat_l", 0.10), ("move_turn", 0.0)])
def test_gate_1c_fails_on_tripod_floor(drv, bad_hold, bad_value):
    tripod = dict(HEALTHY_TRIPOD); tripod[bad_hold] = bad_value
    s = drv.summarize(_agg(tripod, 0.85, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_fails_on_completion_floor(drv):
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.6999, facing=HEALTHY_FACING))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_fails_a_non_turning_head(drv):
    """The plant's historical failure mode: the P-control commands a yaw rate and the body does not
    turn (or turns the wrong way). Progress 0.0 / -0.02 rad/s, both under the 0.05 floor."""
    facing = {"turn_l90": _facing(final=1.0, settle=None, progress=0.0),
              "turn_r90": _facing(final=1.0, settle=None, progress=-0.02)}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "gate_1c" and "FAILED" in msg and "yaw-progress" in msg
    assert "0.000" in msg and "-0.020" in msg


def test_gate_1c_fails_one_turn_hold_alone(drv):
    """BOTH turn holds must clear the floor -- a left turn that works does not excuse a right turn
    that does not (the 1c command band is symmetric; a one-sided head is a formation failure)."""
    for bad in TURN_HOLDS:
        facing = {h: _facing(final=0.2, settle=2.0, progress=(0.01 if h == bad else 0.12)) for h in TURN_HOLDS}
        status, msg = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=facing)))
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
    status, msg = drv.gate_1c_verdict(s)
    assert status == "gate_1c" and "FAILED" in msg
    assert not hasattr(drv, "_turn_rate_proxy") and not hasattr(drv, "ANG_VEL_MAX_DEFAULT")


def test_gate_1c_boundary_is_inclusive(drv):
    """Progress exactly at the 0.05 rad/s floor passes."""
    facing = {h: _facing(final=0.2, settle=2.0, progress=drv.GATE_1C_YAW_RATE) for h in TURN_HOLDS}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.70, facing=facing))
    status, _ = drv.gate_1c_verdict(s)
    assert status == "ok"
    facing = {h: _facing(final=0.2, settle=2.0, progress=drv.GATE_1C_YAW_RATE - 1e-6) for h in TURN_HOLDS}
    status, _ = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.70, facing=facing)))
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
    status, msg = drv.gate_1c_verdict(drv.summarize(a))
    assert status == "gate_1c" and "could not be evaluated" in msg


# ------------------------------------------------------------------ gate_1c_verdict: per-direction floor
# Step-16 review decision 3 (2026-09-17): the walking-hold tripod / completion / turning-rate
# criteria above do not require the robot to actually TRACK any particular direction -- a head that
# only ever walks forward could still clear them by stepping through the stop on the other holds.
def test_gate_1c_fails_a_forward_only_walker_naming_the_failing_hold(drv):
    """Turns fine (facing healthy) and tracks forward well (ratio 0.9), but lat_l ratio 0.05 is
    far under the 0.3 floor -- FAILS, and the message names lat_l."""
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio={"fwd": 0.9, "lat_l": 0.05}))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "gate_1c" and "FAILED" in msg
    assert "lat_l" in msg and "failing: lat_l" in msg


def test_gate_1c_passes_all_four_directions_above_the_floor(drv):
    ratio = {"lat_l": 0.35, "diag": 0.35, "back": 0.35, "move_turn": 0.35}
    s = drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio))
    status, msg = drv.gate_1c_verdict(s)
    assert status == "ok", msg


def test_gate_1c_dir_ratio_boundary_is_inclusive(drv):
    """Exactly at the 0.3 floor passes on all four; a hair under fails."""
    ratio = {h: drv.GATE_1C_DIR_RATIO for h in drv.GATE_1C_DIR_HOLDS}
    status, _ = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)))
    assert status == "ok"
    ratio = {h: drv.GATE_1C_DIR_RATIO - 1e-6 for h in drv.GATE_1C_DIR_HOLDS}
    status, msg = drv.gate_1c_verdict(drv.summarize(_agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)))
    assert status == "gate_1c" and "FAILED" in msg


def test_gate_1c_missing_direction_hold_halts(drv):
    """A direction hold never scored for tracking (e.g. three failed eval attempts) halts, same as
    a missing tripod/completion/turn-rate number."""
    a = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING)
    a["tracking_by_hold"].pop("move_turn")
    status, msg = drv.gate_1c_verdict(drv.summarize(a))
    assert status == "gate_1c" and "could not be evaluated" in msg


def test_decision_table_gate_1c_per_direction_floor_matches_the_driver(drv, table):
    for ratio in (
        {"lat_l": 0.05},
        {"lat_l": 0.35, "diag": 0.35, "back": 0.35, "move_turn": 0.35},
        {h: drv.GATE_1C_DIR_RATIO for h in drv.GATE_1C_DIR_HOLDS},
    ):
        a = _agg(HEALTHY_TRIPOD, 0.85, facing=HEALTHY_FACING, ratio=ratio)
        ok, why = table.gate_1c(a)
        status, _ = drv.gate_1c_verdict(drv.summarize(a))
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
        status, _ = drv.gate_1c_verdict(drv.summarize(a))
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
    status, msg = drv.gate_1c_verdict(s)
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
    status, msg = drv.gate_1c_verdict(drv.summarize(agg))
    assert status == "gate_1c" and "FAILED" in msg
    assert table.gate_1c(agg)[0] is False
