"""Two small ``run_candidates.py`` options for the stroke/posture screen (plan step 9, 2026-09-26).

* ``floor.park_categories`` -- a ``category_survival`` floor tripped ONLY on these categories
  parks and flags (an immediate ``floor_park`` flag) instead of rejecting (audit pass 3: the
  move_turn and diag floors sit inside W's noise). Pure pieces: :func:`floor_disposition`,
  :func:`stage_floor`'s ``detail``, :func:`floor_park_flags`, :data:`IMMEDIATE_FLAGS` via
  ``_flag_record``, and the validator.
* a candidate's ``overrides_chain`` -- the chain-overlay keys it deliberately re-sets (the arms'
  doses over the chain's eps): ``compose_recipe`` records no conflict for them; the recipe id is
  unchanged.

Both are byte-identical when absent (asserted here on the fixture, and by dry-run identity on the
clock campaign's two yamls when the options landed). Pure Python, no Isaac.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TOOLS = OMNI / "experiments/tools"
FIXTURE_YAML = REPO / "tests/unit/fixtures/candidates_v2_2026-09-22.yaml"
SCREEN_YAML = OMNI / "experiments/2026-09-26_stroke_posture/candidates.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_floor_park_test")


@pytest.fixture
def screen(rc):
    return rc.load_candidates(SCREEN_YAML)


def _refusals(rc, doc, subject=None) -> list[str]:
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    return [r for v in verdicts if subject is None or v["id"] == subject for r in v["reasons"]]


def _stage(park=None, kind="category_survival"):
    floor = {"kind": kind, "row": "suite",
             "min_by_category": {"stand": 0.8, "fwd": 0.75, "diag": 0.67, "move_turn": 0.165}}
    if park is not None:
        floor["park_categories"] = park
    return {"stage": "c6y", "categories": ["diag", "fwd", "move_turn", "stand"], "eval": ["suite"],
            "floor": floor}


def _rec(surv: dict, n: int = 100):
    return {"evals": {"suite": {"cat": {c: {"survival": p, "n_env": n} for c, p in surv.items()}}}}


# ================================================================== the screen yaml itself
def test_the_screen_yaml_validates_clean(rc, screen):
    verdicts = rc.validate_candidates(copy.deepcopy(screen), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_the_fixture_still_validates(rc):
    doc = rc.load_candidates(FIXTURE_YAML)
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


# ================================================================== floor_disposition (pure)
@pytest.mark.parametrize("kind", ["candidate", "baseline", "seed2", "twin"])
def test_disposition_without_park_categories_is_the_old_split(rc, kind):
    st = _stage()
    assert rc.floor_park_categories(st) == []
    assert rc.floor_disposition(st, True, [], kind) == "pass"
    want = "reject" if kind == "candidate" else "watch"
    assert rc.floor_disposition(st, False, ["move_turn"], kind) == want
    assert rc.floor_disposition(st, False, None, kind) == want      # e.g. a dead_walker verdict


@pytest.mark.parametrize("kind", ["candidate", "baseline", "seed2"])
def test_disposition_parks_only_when_every_failing_category_is_a_park_category(rc, kind):
    st = _stage(park=["move_turn", "diag"])
    assert rc.floor_disposition(st, False, ["move_turn"], kind) == "park"
    assert rc.floor_disposition(st, False, ["diag", "move_turn"], kind) == "park"
    want = "reject" if kind == "candidate" else "watch"
    assert rc.floor_disposition(st, False, ["move_turn", "fwd"], kind) == want
    assert rc.floor_disposition(st, False, ["fwd"], kind) == want
    assert rc.floor_disposition(st, False, [], kind) == want         # no category named: no park
    assert rc.floor_disposition(st, True, [], kind) == "pass"


# ================================================================== stage_floor's detail
def test_stage_floor_detail_names_the_failing_categories_and_changes_nothing_else(rc):
    st = _stage(park=["move_turn", "diag"])
    rec = _rec({"stand": 1.0, "fwd": 0.95, "diag": 0.87, "move_turn": 0.02})
    detail: dict = {}
    got = rc.stage_floor(st, rec, decisive=st["categories"], previous=[], detail=detail)
    assert got == rc.stage_floor(st, rec, decisive=st["categories"], previous=[])
    assert got[0] is False and detail["fails"] == ["move_turn"]
    detail = {}
    ok = rc.stage_floor(st, _rec({"stand": 1.0, "fwd": 0.95, "diag": 0.87, "move_turn": 0.4}),
                        decisive=st["categories"], previous=[], detail=detail)
    assert ok[0] is True and detail["fails"] == []


def test_stage_floor_detail_is_untouched_by_a_non_category_floor(rc):
    detail: dict = {}
    assert rc.stage_floor({"floor": {"kind": "none"}}, {}, detail=detail) == (True, "no floor at this stage")
    assert detail == {}


# ================================================================== the flag
def test_floor_park_flags_are_immediate_and_park_even_with_a_pair_queued(rc):
    assert "floor_park" in rc.IMMEDIATE_FLAGS
    items = rc.floor_park_flags(["c6y: category survival ... -- FAIL on ['move_turn']"], where="seed-3 run")
    assert len(items) == 1 and items[0]["flag"] == "floor_park" and "seed-3 run" in items[0]["text"]
    rec = rc._flag_record("s3", items, pair_queued=True)
    assert rec["park"] is True and rec["immediate"] == ["floor_park"]
    assert rc.floor_park_flags(None) == [] and rc.floor_park_flags([]) == []


def test_other_flags_still_defer_to_the_pooled_line(rc):
    rec = rc._flag_record("s3", [rc._flag("taps", "taps up")], pair_queued=True)
    assert rec["park"] is False and rec["immediate"] == []


# ================================================================== the validator
def _floor(doc):
    return doc["chains"]["c6y"]["stages"][0]["floor"]


def test_park_categories_must_be_stage_categories(rc, screen):
    _floor(screen)["park_categories"] = ["move_turn", "diagonal"]
    assert any("park_categories ['diagonal'] are not among" in r for r in _refusals(rc, screen, "chain:c6y"))


@pytest.mark.parametrize("bad", ["move_turn", [], [1]])
def test_park_categories_must_be_a_non_empty_list_of_names(rc, screen, bad):
    _floor(screen)["park_categories"] = bad
    assert any("park_categories must be a non-empty list" in r for r in _refusals(rc, screen, "chain:c6y"))


def test_park_categories_needs_a_category_survival_floor(rc, screen):
    _floor(screen)["kind"] = "none"
    assert any("park_categories needs kind category_survival" in r
               for r in _refusals(rc, screen, "chain:c6y"))


def test_park_categories_needs_the_rig_flag_machinery(rc, screen):
    screen["decision_rule"]["per_category"]["blocks"].pop("RIG")
    assert any("park_categories needs a decision rule with block RIG" in r
               for r in _refusals(rc, screen, "chain:c6y"))


# ================================================================== overrides_chain
def test_overrides_chain_silences_only_the_declared_chain_override(rc, screen):
    dose = rc.candidate_by_id(screen, "S")["overlay"]["c6y"]["KRABBY_OMNI_STROKE_W"]  # the campaign's own dose
    s = rc._recipe_with({"entries": []}, screen, "code", "S", set())
    assert s["conflicts"] == []
    assert s["overlay_by_stage"]["c6y"]["KRABBY_OMNI_STROKE_W"] == dose != "0.001"
    assert s["overlay_by_stage"]["c6y"]["KRABBY_OMNI_CENTRING_W"] == "0.001"
    undeclared = copy.deepcopy(screen)
    rc.candidate_by_id(undeclared, "S").pop("overrides_chain")
    s2 = rc._recipe_with({"entries": []}, undeclared, "code", "S", set())
    assert s2["conflicts"] == [f"stage c6y: KRABBY_OMNI_STROKE_W set to '0.001' and '{dose}' (by S)"]
    assert s2["recipe_id"] == s["recipe_id"]            # a record-only field: identity unchanged


def test_overrides_chain_does_not_silence_another_key(rc, screen):
    rc.candidate_by_id(screen, "S")["overlay"]["c6y"]["KRABBY_YAW_W"] = "0.3"
    s = rc._recipe_with({"entries": []}, screen, "code", "S", set())
    assert s["conflicts"] == ["stage c6y: KRABBY_YAW_W set to '0.2' and '0.3' (by S)"]


def test_overrides_chain_is_validated(rc, screen):
    rc.candidate_by_id(screen, "S")["overrides_chain"] = ["KRABBY_OMNI_CENTRING_W", "KRABBY_TYPO_W"]
    refusals = _refusals(rc, screen, "S")
    assert any("overrides_chain names KRABBY_OMNI_CENTRING_W, which S's overlay does not set" in r
               for r in refusals)
    assert any("overrides_chain names KRABBY_TYPO_W, which no chain stage overlay sets" in r for r in refusals)
    rc.candidate_by_id(screen, "S")["overrides_chain"] = "KRABBY_OMNI_STROKE_W"
    assert any("overrides_chain must be a list" in r for r in _refusals(rc, screen, "S"))


# ================================================================== the screen's shape (plan step 13)
def test_every_composed_run_carries_k_and_the_split_flag(rc, screen):
    base = rc.compose_recipe({"entries": []}, screen, "code", upto="S")
    for rec in (base, rc._recipe_with({"entries": []}, screen, "code", "S", set()),
                rc._recipe_with({"entries": []}, screen, "code", "P", set())):
        env = rc.stage_env(rec, "c6y")
        for k, v in (("KRABBY_CLOCK_W", "0.001"), ("KRABBY_APEX_W", "0.001"), ("KRABBY_OMNI_CLOCK_HIDE", "1"),
                     ("KRABBY_OMNI_REWARD_SPLIT", "1")):
            assert env[k] == v, (rec["recipe_id"], k)
    b, s, p = (rc.stage_env(r, "c6y") for r in (
        base, rc._recipe_with({"entries": []}, screen, "code", "S", set()),
        rc._recipe_with({"entries": []}, screen, "code", "P", set())))
    assert {k for k in s if s[k] != b.get(k)} == {"KRABBY_OMNI_STROKE_W"}
    assert {k for k in p if p[k] != b.get(k)} == {"KRABBY_OMNI_CENTRING_W"}
    assert b["KRABBY_OMNI_STROKE_W"] == b["KRABBY_OMNI_CENTRING_W"] == "0.001"


def _cmp(classes: dict, worse_guards=()):
    return {"blocks": {b: {"class": c, "z": {"BETTER": 2.0, "WORSE": -2.0}.get(c, 0.1), "R": 1.0,
                           "missing": [], "k": 1, "level": 0.95, "rows": []} for b, c in classes.items()},
            "guards": {"missing": [], "worse": list(worse_guards), "level": 0.99, "rows": []}}


@pytest.mark.parametrize("classes,worse_guards,want", [
    # plan decision 5: an all-TIE single-seed line gets its s2 pair (the driver's own all-TIE branch)
    ({"F": "TIE", "T": "TIE", "RIG": "TIE", "LEN": "TIE"}, (), ("speculative", True)),
    # pair_on [adopt]: a clear s3 ADOPT is paired too
    ({"F": "TIE", "T": "TIE", "RIG": "TIE", "LEN": "BETTER"}, (), ("speculative", True)),
    # a block or guard WORSE: a decided REJECT, no pair
    ({"F": "TIE", "T": "WORSE", "RIG": "TIE", "LEN": "BETTER"}, (), ("decided", False)),
    ({"F": "TIE", "T": "TIE", "RIG": "TIE", "LEN": "TIE"}, ("cat.side.foot_slip",), ("decided", False)),
])
def test_the_screen_rule_pairs_all_tie_and_adopt_lines(rc, screen, monkeypatch, classes, worse_guards, want):
    rule = screen["decision_rule"]
    assert rc.adopt_requires(rule) is None and rule["pair_on"] == ["adopt"]
    monkeypatch.setattr(rc, "_screen_compare", lambda *a, **k: _cmp(classes, worse_guards))
    monkeypatch.setattr(rc, "anchored_drift", lambda *a, **k: {"verdict": "OK", "why": "stub"})
    out = rc.decide({}, {}, {}, cand_spec=rc.candidate_by_id(screen, "S"), rule=rule)
    assert (out["status"], out["pair"]) == want, out["why"]
