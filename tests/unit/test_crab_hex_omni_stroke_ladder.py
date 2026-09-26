"""The driver wiring for the stride/posture plan's step 3 (2026-09-25): the three stroke/posture
knobs' registration in ``run_candidates.py`` (``knob_parsers``, ``OVERLAY_DUMP_CHECKS``,
``knob_whitelist``), the new ``LEN`` decision-rule block, and the new ``candidates.yaml``
``require_env`` key -- checked against every composed run's stage env (baseline chain, and every
candidate's own composed chain, which is also what a seed-2 twin trains under).

Pure Python; ``run_candidates.py`` is loaded by path (the ``test_crab_hex_omni_motion_ladder.py`` /
``test_crab_hex_omni_clock_ladder.py`` pattern) and exercised against the existing
``tests/unit/fixtures/candidates_v2_2026-09-22.yaml`` fixture -- no Isaac, no subprocess, no real
campaign directory.
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
CANDS_YAML = REPO / "tests/unit/fixtures/candidates_v2_2026-09-22.yaml"

#: An ordinary, side-effect-free reward-weight knob (a plain float, no cross-knob rule attached),
#: used below to exercise ``require_env`` without also tripping the clock-chain rules.
INERT_VAR = "KRABBY_YAW_PROGRESS_W"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_stroke_ladder_test")


@pytest.fixture(scope="module")
def k(rc):
    return rc.knobs()


@pytest.fixture
def cands(rc):
    return rc.load_candidates(CANDS_YAML)


def _refusals(rc, doc, subject=None) -> list[str]:
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    return [r for v in verdicts if subject is None or v["id"] == subject for r in v["reasons"]]


# ================================================================== knob registration
def test_the_fixture_still_validates_before_any_of_this(rc, cands):
    """Sanity anchor: every assertion below reads a DIFFERENCE against this clean baseline."""
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_stroke_knob_names_are_visible_through_knobs_and_the_whitelist(rc, k):
    whitelist = rc.knob_whitelist()
    for var in (k.STROKE_W_VAR, k.CENTRING_W_VAR, k.REWARD_SPLIT_VAR):
        assert var in whitelist, var


def test_stroke_knob_parsers_are_registered(rc, k):
    parsers = rc.knob_parsers()
    assert parsers[k.STROKE_W_VAR]("0.02") == 0.02
    assert parsers[k.CENTRING_W_VAR]("0.0") == 0.0
    assert parsers[k.REWARD_SPLIT_VAR]("1") is True
    with pytest.raises(ValueError):
        parsers[k.STROKE_W_VAR]("-1")
    with pytest.raises(ValueError):
        parsers[k.REWARD_SPLIT_VAR]("2")


def test_len_is_a_recognised_rule_v2_block(rc):
    assert "LEN" in rc.RULE_V2_BLOCKS
    # a rule that names it validates like any other named block (no member -> N/A -> refused)
    doc = {"per_category": {"blocks": {"LEN": {"level": 0.95, "units": "ratio", "members": {}}}}}
    assert rc.rule_v2_required_blocks(doc) == ["LEN"]
    res = _load(TOOLS / "resolution.py", "resolution_stroke_ladder_test")
    assert res.BLOCK_ORDER[-1] == "LEN"                         # ordered like RIG, after it


def test_len_may_never_be_adopt_requires(rc):
    """Decision 5: S's adoption never requires LEN BETTER -- a rule naming it is refused by name."""
    rule = {"rule_version": 2, "adopt_requires": "LEN",
            "per_category": {"blocks": {"LEN": {"level": 0.95, "units": "ratio", "members": {}}}}}
    reasons: list[str] = []
    rc._check_rule_v2(rule, reasons)
    assert any("adopt_requires names 'LEN'" in r for r in reasons), reasons


def test_a_candidate_overlay_setting_the_stroke_knobs_round_trips_and_dumps(rc, cands, k):
    """The three knobs pass ``_check_knob``'s whitelist/parser gate on an ordinary chain stage, and
    each has an ``OVERLAY_DUMP_CHECKS`` entry (the loud refusal every other overlay key gets)."""
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["scratch_1c"]["stages"][0]
    stage["overlay"].update({k.STROKE_W_VAR: "0.02", k.CENTRING_W_VAR: "0.01", k.REWARD_SPLIT_VAR: "1"})
    reasons = _refusals(rc, doc, "chain:scratch_1c")
    assert not any("not a known knob" in r or "no entry in OVERLAY_DUMP_CHECKS" in r for r in reasons), reasons
    for var in (k.STROKE_W_VAR, k.CENTRING_W_VAR, k.REWARD_SPLIT_VAR):
        assert var in rc.OVERLAY_DUMP_CHECKS


# ================================================================== the dump-audit checks
STOCK_DUMP = """
rewards:
  reward_stride_length:
    func: parkour_isaaclab.envs.mdp.rewards:RewardStrideLength
    weight: 0.5
    params:
      power: 2.0
"""

STROKE_DUMP = """
rewards:
  reward_stride_length:
    func: parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_rewards:RewardFootStroke
    weight: 0.02
    params:
      rho0: 0.5
  reward_body_centring:
    func: parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_rewards:RewardBodyCentring
    weight: 0.01
    params:
      sigma: 0.2
reward_category_split: true
"""


def test_stroke_w_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_STROKE_W"]
    assert check(STOCK_DUMP, "")                              # off keeps the stock func
    assert not check(STOCK_DUMP, "0.02")                      # armed but the dump shows no swap
    assert check(STROKE_DUMP, "0.02")                         # armed and the dump shows func+weight
    assert not check(STROKE_DUMP, "0.05")                     # armed at a DIFFERENT weight than the dump
    assert not check(STROKE_DUMP, "")                         # the knob is off but the dump shows a swap


def test_centring_w_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_CENTRING_W"]
    assert check(STOCK_DUMP, "")                              # off: no term at all
    assert not check(STOCK_DUMP, "0.01")                      # armed but absent from the dump
    assert check(STROKE_DUMP, "0.01")
    assert not check(STROKE_DUMP, "0.02")


def test_reward_split_dump_check(rc):
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_REWARD_SPLIT"]
    assert check(STOCK_DUMP, "0")
    assert not check(STOCK_DUMP, "1")
    assert check(STROKE_DUMP, "1")
    assert not check(STROKE_DUMP, "0")


# ================================================================== require_env
def test_check_require_env_flags_a_missing_or_mismatched_var_and_ok_when_it_matches(rc):
    reasons: list[str] = []
    rc._check_require_env({"A": "1", "B": "x"}, "where", {"A": "1", "C": "3"}, reasons)
    assert len(reasons) == 1 and "where" in reasons[0] and "C" in reasons[0] and "must be" in reasons[0]
    reasons2: list[str] = []
    rc._check_require_env({"A": "1"}, "where", {"A": "1"}, reasons2)
    assert reasons2 == []
    reasons3: list[str] = []
    rc._check_require_env({}, "where", {}, reasons3)
    assert reasons3 == []


def test_require_env_absent_is_byte_identical_to_the_fixture(rc, cands):
    """No ``require_env`` key at all -> not a single ``require_env`` reason anywhere (the existing
    behaviour stays byte-identical when the key is absent)."""
    assert "require_env" not in cands
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert not any("require_env" in reason for v in verdicts for reason in v["reasons"])


def test_require_env_refuses_every_composed_run_missing_it(rc, cands):
    doc = copy.deepcopy(cands)
    doc["require_env"] = {INERT_VAR: "3.0"}
    bad = {v["id"]: v["reasons"] for v in rc.validate_candidates(doc, strict=False, check_replay=False)
           if not v["ok"]}
    # the bare chain (no candidate overlay) is refused
    assert any(f"require_env {INERT_VAR}" in r for r in bad.get("chain:scratch_1c", [])), bad
    # a candidate that never sets the var on its composed chain is refused too
    assert any(f"require_env {INERT_VAR}" in r for r in bad.get("C1", [])), bad
    # C2 already overlays exactly this var at exactly this value (KRABBY_YAW_PROGRESS_W: "3.0",
    # `stages: all`) -- its composed check passes even though the campaign-wide gate is new
    assert not any("require_env" in r for r in bad.get("C2", []))


def test_require_env_refuses_a_wrong_value_even_when_the_var_is_present(rc, cands):
    doc = copy.deepcopy(cands)
    doc["require_env"] = {INERT_VAR: "9.9"}                    # C2 carries "3.0", not "9.9"
    bad = {v["id"]: v["reasons"] for v in rc.validate_candidates(doc, strict=False, check_replay=False)
           if not v["ok"]}
    assert any(f"require_env {INERT_VAR}" in r and "'9.9'" in r for r in bad.get("C2", [])), bad


def test_require_env_passes_once_every_chain_and_candidate_carries_it(rc, cands):
    doc = copy.deepcopy(cands)
    doc["require_env"] = {INERT_VAR: "3.0"}
    for chain in doc["chains"].values():
        for stage in chain["stages"]:
            stage.setdefault("overlay", {})[INERT_VAR] = "3.0"
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    assert not any("require_env" in r for v in verdicts for r in v["reasons"]), [
        (v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_require_env_must_be_a_mapping(rc, cands):
    doc = copy.deepcopy(cands)
    doc["require_env"] = ["not", "a", "mapping"]
    got = _refusals(rc, doc, "<campaign>")
    assert any("require_env must map" in r for r in got), got


def test_dry_run_prints_the_require_env_refusals(rc, cands):
    """``dry_run`` prints every verdict's reasons verbatim under '## validator' (see its source),
    so a require_env refusal reaches the same place every other refusal does -- exercised here on
    the actual print loop's own formatting, not a re-derivation of it."""
    doc = copy.deepcopy(cands)
    doc["require_env"] = {INERT_VAR: "3.0"}
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    printed = [f"      - {reason}" for v in verdicts for reason in v["reasons"]]
    assert any(f"require_env {INERT_VAR}" in line for line in printed)
    src = (TOOLS / "run_candidates.py").read_text()
    body = src[src.index("def dry_run("):]
    assert 'for reason in v["reasons"]' in body
