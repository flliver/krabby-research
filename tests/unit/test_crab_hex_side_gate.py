"""The sideways driver's written 1a gate reads the WALKING-hold tripod, not the episode roll-up.

Background (2026-09-11, first sideways 1a run): the harness's episode ``tripod_score`` is a
steady-step-weighted mean over every hold of the canary schedule, the cmd-0 stand hold included. A
head that truly stands scores ~0 there (sideways: 0.001 at 0.006 m/s) while a head that keeps
stepping through its stop does not (forward 1a reference: 0.453 at 0.066 m/s), so the sideways head
read 0.355 against a 0.40 floor although its walking holds scored 0.537 / 0.533 (forward 0.633 /
0.595). The gate now takes the minimum over the holds the harness scores for tracking
(``tracking_by_hold[h].ratio.n > 0``). Pure Python: the driver and the decision table are loaded by
path; nothing here imports Isaac.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIDE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task"
DRIVER = SIDE / "experiments/tools/run_phases.py"
TABLE = SIDE / "experiments/2026-09-11_0414_sideways_formation/decision_table.py"
REAL_1A = (REPO / "parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation"
           "/seed3_1a_001_A15pB/slow__side/seed001/2026-09-11_17-32-00/scenario_metrics.json")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load(DRIVER, "side_run_phases_gate_test")


@pytest.fixture(scope="module")
def table():
    return _load(TABLE, "side_decision_table_gate_test")


def _stats(median, n):
    return {"mean": median, "median": median, "n": n, "p25": median, "p75": median}


def _agg(stand, creep, low, completion, falls=7, n=100, with_tracking=True):
    """A ``report.py::aggregate``-shaped dict with the fields the gate reads (real 1a numbers by default)."""
    a = {
        "n_episodes": n,
        "schedule_completion_rate": completion,
        "termination_reasons": {"schedule_complete": n - falls, "fall": falls},
        "fall_classes": {"none": n - falls, "roll": falls},
        # the episode roll-up: the step-weighted mean of the three equal holds
        "tripod_score": _stats((stand + creep + low) / 3.0, n),
        "tripod_score_by_hold": {"creep": _stats(creep, 96), "low": _stats(low, 93), "stand": _stats(stand, 99)},
        "tracking_ratio": _stats(0.634, 96),
    }
    if with_tracking:
        a["tracking_by_hold"] = {
            "creep": {"cmd_vx_mean": 0.25, "achieved_vx": _stats(0.184, 96), "ratio": _stats(0.736, 96)},
            "low": {"cmd_vx_mean": 0.35, "achieved_vx": _stats(0.189, 93), "ratio": _stats(0.541, 93)},
            "stand": {"cmd_vx_mean": 0.0, "achieved_vx": _stats(0.006, 100),
                      "ratio": {"mean": None, "median": None, "n": 0, "p25": None, "p75": None}},
        }
    return a


SIDE_1A = dict(stand=0.001, creep=0.537, low=0.533, completion=0.93, falls=7)
FWD_1A = dict(stand=0.453, creep=0.633, low=0.595, completion=0.86, falls=14)


# ------------------------------------------------------------------ the gated number
def test_walking_tripod_is_the_min_over_walking_holds_only(drv):
    walk, by_hold = drv.walking_tripod(_agg(**SIDE_1A))
    assert walk == pytest.approx(0.533)
    assert by_hold == {"creep": 0.537, "low": 0.533, "stand": 0.001}
    # a stand hold with a HIGH tripod (the forward reference's stepping-through-the-stop) is still excluded
    walk_f, _ = drv.walking_tripod(_agg(**FWD_1A))
    assert walk_f == pytest.approx(0.595)
    walk_h, _ = drv.walking_tripod(_agg(stand=0.9, creep=0.537, low=0.533, completion=0.93))
    assert walk_h == pytest.approx(0.533)


def test_episode_rollup_is_kept_for_the_record_but_not_gated(drv):
    s = drv.summarize(_agg(**SIDE_1A))
    assert s["tripod"] == pytest.approx(0.357, abs=1e-3)      # what the first gate read
    assert s["tripod_walking"] == pytest.approx(0.533)         # what the gate reads now
    assert s["tripod_by_hold"]["stand"] == pytest.approx(0.001)
    assert s["completion"] == 0.93 and s["falls"] == 7 and s["n"] == 100


def test_no_walking_hold_means_no_number(drv):
    only_stand = _agg(**SIDE_1A)
    only_stand["tracking_by_hold"] = {"stand": only_stand["tracking_by_hold"]["stand"]}   # schedule of one stand hold
    only_stand["tripod_score_by_hold"] = {"stand": _stats(0.9, 99)}
    assert drv.walking_tripod(only_stand)[0] is None
    # the walking holds are enumerated from the TRACKING block: without it nothing is a walking hold
    assert drv.walking_tripod(_agg(**SIDE_1A, with_tracking=False))[0] is None
    assert drv.walking_tripod({})[0] is None and drv.walking_tripod({})[1] == {}


def test_walking_hold_without_a_gait_window_is_unmeasured_not_a_pass(drv):
    """report.py writes tripod_score_by_hold only for holds with a scorable window; a head that never
    cycles its feet at 0.35 has no 'low' entry there while the tracking block still lists the hold."""
    a = _agg(**SIDE_1A)
    del a["tripod_score_by_hold"]["low"]
    walk, by_hold = drv.walking_tripod(a)
    assert walk is None and by_hold["low"] is None and by_hold["creep"] == 0.537
    status, msg = drv.gate_1a_verdict(drv.summarize(a))
    assert status == "gate_1a" and "could not be evaluated" in msg and "low n/a" in msg
    a = _agg(**SIDE_1A)
    a["tripod_score_by_hold"]["low"] = _stats(None, 0)
    assert drv.walking_tripod(a)[0] is None


def test_classification_is_by_tracking_not_by_label(drv):
    """A hold named 'stand' that the harness scored for tracking IS a walking hold; a hold named 'creep'
    with ratio.n == 0 is not."""
    a = _agg(**SIDE_1A)
    a["tracking_by_hold"]["stand"]["ratio"] = _stats(0.5, 99)     # scored -> walking -> its 0.001 is gated
    assert drv.walking_tripod(a)[0] == pytest.approx(0.001)
    assert drv.gate_1a_verdict(drv.summarize(a))[0] == "gate_1a"
    a = _agg(**SIDE_1A)
    a["tracking_by_hold"]["creep"]["ratio"] = {"mean": None, "median": None, "n": 0, "p25": None, "p75": None}
    assert drv.walking_tripod(a)[0] == pytest.approx(0.533)         # only 'low' is a walking hold now


@pytest.mark.parametrize("kw, expect", [
    (dict(stand=0.001, creep=0.40, low=0.45, completion=0.70), "ok"),        # both floors inclusive
    (dict(stand=0.001, creep=0.3999, low=0.45, completion=0.93), "gate_1a"),
    (dict(stand=0.001, creep=0.45, low=0.45, completion=0.6999), "gate_1a"),
])
def test_floor_boundaries(drv, kw, expect):
    assert drv.gate_1a_verdict(drv.summarize(_agg(**kw)))[0] == expect


# ------------------------------------------------------------------ the verdict
def test_first_run_passes_under_the_walking_hold_gate_and_failed_under_the_episode_one(drv):
    s = drv.summarize(_agg(**SIDE_1A))
    status, msg = drv.gate_1a_verdict(s)
    assert status == "ok", msg
    assert "0.533" in msg and "creep 0.537" in msg and "stand 0.001" in msg
    # the pre-fix reading of the same run (episode tripod 0.357 < 0.40) would have halted
    assert s["tripod"] < drv.GATE_1A_TRIPOD <= s["tripod_walking"]


def test_forward_reference_passes_too(drv):
    status, _ = drv.gate_1a_verdict(drv.summarize(_agg(**FWD_1A)))
    assert status == "ok"


@pytest.mark.parametrize("kw, needle", [
    (dict(stand=0.001, creep=0.45, low=0.38, completion=0.93), "walking-hold tripod 0.380"),   # one walking hold under the floor
    (dict(stand=0.9, creep=0.39, low=0.39, completion=0.93), "walking-hold tripod 0.390"),      # a high stand hold cannot rescue it
    (dict(stand=0.001, creep=0.537, low=0.533, completion=0.69), "completion 0.690"),           # completion floor
])
def test_gate_fails_loudly(drv, kw, needle):
    status, msg = drv.gate_1a_verdict(drv.summarize(_agg(**kw)))
    assert status == "gate_1a"
    assert "FAILED" in msg and needle in msg and "--continue" in msg


def test_no_metrics_is_a_halt_not_a_pass(drv):
    for s in ({}, drv.summarize(None), drv.summarize(_agg(**SIDE_1A, with_tracking=False))):
        status, msg = drv.gate_1a_verdict(s)
        assert status == "gate_1a"
        assert "could not be evaluated" in msg


# ------------------------------------------------------------------ the decision table agrees
def test_decision_table_gate_matches_the_driver(drv, table):
    for kw in (SIDE_1A, FWD_1A, dict(stand=0.001, creep=0.45, low=0.38, completion=0.93)):
        a = _agg(**kw)
        ok, msg = table.gate_1a(a)
        status, _ = drv.gate_1a_verdict(drv.summarize(a))
        assert ok == (status == "ok"), (kw, msg)
        assert table.walking_tripod(a)[0] == pytest.approx(drv.walking_tripod(a)[0])
    assert "walk 0.53" in table.cell(_agg(**SIDE_1A))
    # the unmeasured-walking-hold case agrees too (None on both sides -> table gate False)
    a = _agg(**SIDE_1A)
    del a["tripod_score_by_hold"]["low"]
    assert table.walking_tripod(a)[0] is None and drv.walking_tripod(a)[0] is None
    assert table.gate_1a(a)[0] is False
    # cell() shows the walking number even when the episode roll-up is absent
    a = _agg(**SIDE_1A)
    del a["tripod_score"]
    assert "n/a (walk 0.53)" in table.cell(a)


# ------------------------------------------------------------------ the real record (when on disk)
def test_real_first_run_record(drv, table):
    if not REAL_1A.exists():
        pytest.skip("the 2026-09-11 1a eval is not on this machine")
    agg = json.loads(REAL_1A.read_text())["aggregate"]
    s = drv.summarize(agg)
    assert s["tripod"] == pytest.approx(0.3553, abs=1e-3)
    assert s["tripod_walking"] == pytest.approx(0.5326, abs=1e-3)
    assert s["tripod_by_hold"]["creep"] == pytest.approx(0.5374, abs=1e-3)
    assert s["completion"] == pytest.approx(0.93) and s["falls"] == 7
    assert drv.gate_1a_verdict(s)[0] == "ok"
    assert table.gate_1a(agg)[0] is True
