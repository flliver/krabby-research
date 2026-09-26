"""``decision_rule.pair_on: [always]`` (stroke formation campaign, 2026-09-26) and that campaign's Stage-1 spec.

``always`` queues the seed-2 pair for EVERY decided s3 ADOPT or REJECT, whatever its class; a rule without it
decides and pairs exactly as before. The dry-run queue simulation follows the same flag. The spec tests pin the
Stage-1 ``candidates.yaml``: C1d's t3 verbatim (overlay, floor, from scratch, 3000 its), no clock-removal key,
both arms at both seeds in the order B_f s3, B_f s2, S_f s3, S_f s2, P_f s3, P_f s2.

Pure Python: the driver is loaded by path; the screens / resolution helpers are
``test_crab_hex_omni_clock_ladder``'s (loaded by path). Nothing imports Isaac or touches a campaign directory.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments"
TOOLS = EXP / "tools"
FORMATION = EXP / "2026-09-26_stroke_formation/candidates.yaml"
POSTURE = EXP / "2026-09-26_stroke_posture/candidates.yaml"
FACING = EXP / "2026-09-19_facing_tracking_ladder/candidates.yaml"
K_KEYS = ("KRABBY_CLOCK_W", "KRABBY_APEX_W", "KRABBY_OMNI_CLOCK_HIDE")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_pair_on_always_test")


@pytest.fixture(scope="module")
def CL():
    return _load(REPO / "tests/unit/test_crab_hex_omni_clock_ladder.py", "clock_ladder_helpers_for_always")


@pytest.fixture(scope="module")
def H(CL):
    return _load(REPO / "tests/unit/test_crab_hex_omni_candidates.py", "omni_candidates_helpers_for_always")


@pytest.fixture
def rule(rc, CL):
    return copy.deepcopy(rc.load_candidates(CL.FIXTURE)["decision_rule"])


def _with(rule: dict, pair_on) -> dict:
    out = copy.deepcopy(rule)
    if pair_on is None:
        out.pop("pair_on", None)
    else:
        out["pair_on"] = list(pair_on)
    return out


# ================================================================== apply_pair_on / decide
def test_always_pairs_every_decided_line_whatever_its_class(rc, CL, H, rule):
    always = _with(rule, ["always"])
    res = CL.rres(rc, rule)
    base = CL.rscreen(H)
    cases = {
        "adopt": CL.rig_better(H),
        "block_worse": CL.rig_better(H, ratio=0.50),
        "two_guards_worse": CL.rig_better(H, cat__fwd__foot_slip=0.40, cat__side__foot_slip=0.40),
        "required_tie": CL.rscreen(H, cat__fwd__foot_slip=0.40),
        "guard_missing": CL.rscreen(H, cat__stand__steps_per_s=None),
    }
    for name, cand in cases.items():
        plain = CL._decide(rc, _with(rule, None), cand, base, res)
        out = CL._decide(rc, always, cand, base, res)
        assert plain["status"] == "decided" and plain["pair"] is False, name
        # the s3 classification is kept; only the status, the pair and the evidence/why change
        assert (out["chosen"], out["confidence"]) == (plain["chosen"], plain["confidence"]), name
        assert (out["status"], out["pair"], out["evidence"]["pair_on"]) == ("speculative", True, "always"), name
        assert "pair_on always" in out["why"], name
        assert {k: v for k, v in out["evidence"].items() if k != "pair_on"} == plain["evidence"], name


def test_always_records_the_narrower_case_when_one_matches(rc, CL, H, rule):
    res = CL.rres(rc, rule)
    out = CL._decide(rc, _with(rule, ["adopt", "always"]), CL.rig_better(H), CL.rscreen(H), res)
    assert (out["status"], out["evidence"]["pair_on"]) == ("speculative", "adopt")


def test_an_all_tie_line_is_the_old_speculative_pair_with_or_without_always(rc, CL, H, rule):
    res = CL.rres(rc, rule)
    tie = CL.rscreen(H)
    no_req = {k: v for k, v in rule.items() if k != "adopt_requires"}
    plain = CL._decide(rc, _with(no_req, None), tie, CL.rscreen(H), res)
    out = CL._decide(rc, _with(no_req, ["always"]), tie, CL.rscreen(H), res)
    assert plain["status"] == "speculative" and plain["pair"] is True
    assert out == plain and "pair_on" not in out["evidence"]


def test_without_always_the_pairing_is_unchanged(rc, CL, H, rule):
    """A clear REJECT under ``pair_on: [adopt]`` (or no pair_on) stays decided and unpaired."""
    res = CL.rres(rc, rule)
    worse = CL.rig_better(H, ratio=0.50)
    for pair_on in (None, ["adopt"], ["adopt", "neutral", "lone_guard"]):
        out = CL._decide(rc, _with(rule, pair_on), worse, CL.rscreen(H), res)
        assert (out["chosen"], out["status"], out["pair"]) == ("REJECT", "decided", False)
        assert "pair_on" not in out["evidence"]


def test_apply_pair_on_without_scored_blocks(rc):
    line = {"status": "decided", "chosen": "REJECT", "pair": False, "why": "x", "evidence": {"seed3": {}}}
    assert rc.apply_pair_on(dict(line), {"pair_on": ["adopt"]}) == line
    out = rc.apply_pair_on(dict(line), {"pair_on": ["always"]})
    assert (out["status"], out["pair"], out["evidence"]["pair_on"]) == ("speculative", True, "always")
    # a SKIP (or anything but ADOPT/REJECT) and a line already speculative are never touched
    for other in (dict(line, chosen="SKIP"), dict(line, status="speculative")):
        assert rc.apply_pair_on(dict(other), {"pair_on": ["always"]}) == other


def test_the_validator_accepts_always_and_still_refuses_an_unknown_case(rc, CL):
    doc = copy.deepcopy(rc.load_candidates(CL.FIXTURE))
    doc["decision_rule"]["pair_on"] = ["always"]
    assert not [r for r in CL._refusals(rc, doc, "<campaign>") if "pair_on" in r]
    doc["decision_rule"]["pair_on"] = ["always", "tie"]
    assert any("pair_on must be a list of" in r for r in CL._refusals(rc, doc, "<campaign>"))


# ================================================================== the queue
def _queue(rc, cands, outcome):
    return [(r["label"], r["kind"]) for r in rc.simulate_queue(cands, outcome=outcome, code="C")]


@pytest.mark.parametrize("outcome", ["REJECT", "ADOPT"])
def test_the_formation_queue_runs_both_seeds_of_every_arm_in_order(rc, outcome):
    assert _queue(rc, rc.load_candidates(FORMATION), outcome) == [
        ("B_f s3", "baseline"), ("B_f s2", "baseline"),
        ("S_f s3", "candidate"), ("S_f s2", "seed2"), ("S_f s2", "pooled"),
        ("P_f s3", "candidate"), ("P_f s2", "seed2"), ("P_f s2", "pooled"),
        ("done", "done")]


def test_a_campaign_without_always_simulates_exactly_as_before(rc):
    """The stroke/posture screen (pair_on [adopt]): the simulation records a clear line, no pair."""
    assert _queue(rc, rc.load_candidates(POSTURE), "REJECT") == [
        ("B s3", "baseline"), ("B s2", "baseline"), ("S s3", "candidate"), ("P s3", "candidate"),
        ("done", "done")]


def test_the_real_next_item_queues_the_pair_after_an_always_line(rc):
    """Driven through ``next_item`` itself (not the simulation's shortcut): a speculative REJECT
    written the way ``_dispatch_candidate`` writes it is followed by its own seed-2 run."""
    cands = rc.load_candidates(FORMATION)
    ledger = {"schema": 1, "entries": []}
    state = {"ladder": {"experiments": {}, "squashed": {}, "user_choice": {}}, "keys": {}}
    for _ in range(3):                                  # B_f s3, B_f s2, S_f s3 (simulated)
        item = rc.next_item(cands, ledger, state, code="C")
        rc._simulate_apply(cands, ledger, state, item, "REJECT", "C")
    entry = ledger["entries"][-1]
    assert (entry["candidate"], entry["status"], entry["pair"]) == ("S_f", "speculative", True)
    item = rc.next_item(cands, ledger, state, code="C")
    assert (item["kind"], item["candidate"], item["seed"]) == ("seed2", "S_f", "2")


# ================================================================== the Stage-1 spec
def test_the_formation_spec_is_c1d_t3_verbatim_with_the_eps_knobs_and_no_k(rc):
    cands = rc.load_candidates(FORMATION)
    facing = rc.load_candidates(FACING)
    t3 = next(s for s in facing["chains"]["curriculum_turnfirst"]["stages"] if s["stage"] == "t3")
    (stage,) = cands["chains"]["t3f"]["stages"]
    for key in ("stage", "base", "iters", "from_scratch", "resume_from", "eval", "categories", "floor"):
        assert stage[key] == t3[key], key
    assert (stage["iters"], stage["from_scratch"], stage["base"]) == (3000, True, "1a")
    extra = {k: v for k, v in stage["overlay"].items() if k not in t3["overlay"]}
    assert {k: stage["overlay"][k] for k in t3["overlay"]} == t3["overlay"]
    assert extra == {"KRABBY_OMNI_STROKE_W": "0.001", "KRABBY_OMNI_CENTRING_W": "0.001",
                     "KRABBY_OMNI_REWARD_SPLIT": "1"}
    text = FORMATION.read_text()
    body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())   # comments may NAME K
    assert not [k for k in K_KEYS if k in body]
    assert cands["require_env"] == {"KRABBY_OMNI_REWARD_SPLIT": "1"}
    assert cands["decision_rule"]["pair_on"] == ["always"]
    assert not rc.flags_enabled(cands["decision_rule"])        # no RIG: no flag / baseline-floor park
    assert cands["screen"] == {"chain": "t3f", "eval": ["suite"], "iters": 3000}
    arms = {c["id"]: c for c in cands["candidates"]}
    assert set(arms) == {"S_f", "P_f"}
    assert arms["S_f"]["overlay"] == {"t3": {"KRABBY_OMNI_STROKE_W": arms["S_f"]["overlay"]["t3"]["KRABBY_OMNI_STROKE_W"]}}
    assert arms["P_f"]["overlay"] == {"t3": {"KRABBY_OMNI_CENTRING_W": arms["P_f"]["overlay"]["t3"]["KRABBY_OMNI_CENTRING_W"]}}
    assert all(c["control"] == "baseline" for c in arms.values())


def test_the_formation_spec_validates(rc):
    verdicts = rc.validate_candidates(rc.load_candidates(FORMATION), strict=False, check_replay=False)
    assert [v["id"] for v in verdicts if not v["ok"]] == []
