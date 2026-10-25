"""The gait-motion-quality ladder's driver work (plan step 4, 2026-09-23): the ``checkpoint:`` resume
kind, block Q with per-metric sign, the separate guard specs, the ``adopt_requires`` rule, the
zero-spread demotion path, the per-category stage floor, the conditional ``await_if`` gate, the
promotions switch -- and the campaign's own ``candidates.yaml``.

Pure Python. ``run_candidates.py``, ``resolution.py`` and ``eval_categories.py`` are loaded by path;
the synthetic suite stamps and the fake training/eval children are the existing test modules' own
helpers (``test_crab_hex_omni_resolution.make_suite_run``, ``test_crab_hex_omni_candidates``'s
``install_fakes`` / ``make_ctx``), loaded by path so they are the same code the rest of the suite
trusts. Nothing here imports Isaac, launches a subprocess or touches a real campaign directory.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TOOLS = OMNI / "experiments/tools"
CAMPAIGN = OMNI / "experiments/2026-09-22_gait_motion_quality"
CAMPAIGN_YAML = CAMPAIGN / "candidates.yaml"
PRIOR_YAML = OMNI / "experiments/2026-09-19_facing_tracking_ladder/candidates.yaml"
GAIT_SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
CATS = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
TRANSLATION = ("fwd", "side", "back", "diag")
EVAL = "e" * 64


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_motion_ladder_test")


@pytest.fixture(scope="module")
def res(rc):
    return rc.resolution_tool()


@pytest.fixture(scope="module")
def ec(rc):
    return rc.eval_categories()


@pytest.fixture(scope="module")
def H():
    """``test_crab_hex_omni_candidates``' helpers (screens, ctx, the fake children), by path."""
    return _load(REPO / "tests/unit/test_crab_hex_omni_candidates.py", "omni_candidates_helpers_for_q")


@pytest.fixture(scope="module")
def M():
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import metrics as mod
    return mod


@pytest.fixture(scope="module")
def cands(rc):
    return rc.load_candidates(CAMPAIGN_YAML)


def _filled_text() -> str:
    """The campaign yaml with every ``TBD_replay_gate`` weight filled (penalties <= 0; the air-time
    de-weight >= 0) -- what the file looks like once the step-3 replay gate has run."""
    text = CAMPAIGN_YAML.read_text()
    text = text.replace("KRABBY_AIRTIME_W: TBD_replay_gate", 'KRABBY_AIRTIME_W: "0.4"')
    return text.replace("TBD_replay_gate", '"-0.001"')


@pytest.fixture(scope="module")
def filled(rc, tmp_path_factory):
    path = tmp_path_factory.mktemp("filled") / "candidates.yaml"
    path.write_text(_filled_text())
    return rc.load_candidates(path)


@pytest.fixture
def rule(cands):
    return copy.deepcopy(cands["decision_rule"])


# ------------------------------------------------------------------------------ fixtures: screens
def qscreen(H, M, *, quality=1.0, six_down=0.5, ratio=0.60, surv=0.95, **over) -> dict:
    """A suite screen (``H.screen_cat``) with every motion-quality key on the categories it is
    defined on (``gait_eval.metrics.MOTION_QUALITY_APPLIES``), None elsewhere. ``quality`` is the
    value of every lower-is-better key, ``six_down`` of ``six_down_frac``; ``over`` takes dotted ids
    with ``__`` for ``.``."""
    out = H.screen_cat(ratio=ratio, surv=surv)
    for c in CATS:
        entry = out["cat"][c]
        for key in M.MOTION_QUALITY_KEYS:
            applies = c in M.MOTION_QUALITY_APPLIES[key]
            val = (six_down if key == "six_down_frac" else quality) if applies else None
            entry[key] = val
            for hand in (entry.get("by_hand") or {}).values():
                hand[key] = val
    for key, value in over.items():
        H._put(out, key.replace("__", "."), value)
    return out


def qres(rc, ec, rule, *, metric_R=0.10, guard_R=0.10, block_R=0.50, guard_key_R=0.20) -> dict:
    """A synthetic D00s for the campaign rule: round R for every member and guard of the union at
    all seven categories, the dual guards (also block members) under ``<mid>@guard`` with their OWN
    R (``guard_key_R``, so a test can tell which one a guard row read)."""
    union = ec.expand_rule_union(rule, CATS, CATS)
    members = set()
    metrics = {}
    for body in union["blocks"].values():
        for mid in body["metrics"]:
            members.add(mid)
            metrics[mid] = {"R_eval": metric_R, "level": 0.95, "n": 100, "source": "cat"}
    for mid in union["guards"]["metrics"]:
        if mid in members:
            metrics[ec.guard_key(mid)] = {"R_eval": guard_key_R, "level": 0.99, "n": 100,
                                          "source": "cat", "block": "guard"}
        else:
            metrics[mid] = {"R_eval": guard_R, "level": 0.99, "n": 100, "source": "cat"}
    blocks = {b: {"R_eval": block_R, "level": 0.95, "members": []} for b in union["blocks"]}
    return {"schema": 1, "rule_version": 2, "B": 2000, "rng_seed": 1, "metrics": metrics,
            "blocks": blocks, "decisive_at": list(CATS), "process_floor": {}, "seed_terms": {},
            "inputs": [], "by_row": {}}


def _decide(rc, rule, cand, base, resolution, **kw):
    return rc.decide(cand, base, resolution, cand_spec={}, rule=rule, seed_terms={},
                     anchor_screen=None, **kw)


# ================================================================== eval_categories: sense, keys
def test_metric_sign_is_lower_is_better_for_quality_keys_except_six_down(ec):
    assert ec.metric_sign("cat.stand.action_jitter") == -1
    assert ec.metric_sign("cat.fwd.swings_per_m") == -1
    assert ec.metric_sign("cat.turn.cam_work_per_rad") == -1
    assert ec.metric_sign("cat.side.by_hand.l.action_jitter") == -1
    assert ec.metric_sign("cat.fwd.slip_per_m@guard") == -1
    assert ec.metric_sign("cat.stand.six_down_frac") == 1
    for mid in ("cat.fwd.ratio", "cat.stand.survival", "cat.turn.facing_closed", "cat.fwd.tripod",
                "cat_tripod_min", "ratio_by_hold.fwd", "completion"):
        assert ec.metric_sign(mid) == 1, mid
    # every pre-Q key keeps +1, so no recorded number of an earlier campaign can move
    assert not ({"survival", "ratio", "achieved_speed", "facing_closed", "yaw_progress", "net_yaw",
                 "tripod"} & ec.LOWER_IS_BETTER_KEYS)
    # ... plus the clock-deprecation campaign's rigidity stepping counts (2026-09-23, plan step 3)
    assert ec.LOWER_IS_BETTER_KEYS == (frozenset(ec.MOTION_QUALITY_KEYS) - {"six_down_frac"}) | {
        "steps_per_m", "steps_per_rad", "steps_per_s", "taps_per_m", "taps_per_rad"}


def test_guard_key_round_trips(ec):
    assert ec.guard_key("cat.fwd.ratio") == "cat.fwd.ratio@guard"
    assert ec.guard_key("cat.fwd.ratio@guard") == "cat.fwd.ratio@guard"
    assert ec.base_metric_id("cat.fwd.ratio@guard") == "cat.fwd.ratio"
    assert ec.base_metric_id("cat.fwd.ratio") == "cat.fwd.ratio"
    assert ec.parse_cat_id("cat.fwd.ratio@guard") is None


def test_a_demoted_member_leaves_every_block_and_guard_and_is_watched_with_its_reason(ec, rule):
    rule["per_category"]["demoted"] = {"stand": {"clock_restarts": "zero spread on X08-c6"},
                                       "fwd": {"slip_per_m": "zero spread"}}
    exp = ec.expand_rule(rule, CATS, CATS, absolute_units=False)
    assert "cat.stand.clock_restarts" not in exp["blocks"]["Q"]["metrics"]
    assert "cat.fwd.slip_per_m" not in exp["blocks"]["Q"]["metrics"]
    assert "cat.fwd.slip_per_m" not in exp["guards"]["metrics"]           # the guard goes too
    assert "cat.side.slip_per_m" in exp["guards"]["metrics"]              # only the named category
    assert exp["watch"]["cat.stand.clock_restarts"] == "demoted: zero spread on X08-c6"
    assert exp["watch"]["cat.fwd.slip_per_m"] == "demoted: zero spread"
    bad = copy.deepcopy(rule)
    bad["per_category"]["demoted"] = {"stand": ["clock_restarts"]}
    with pytest.raises(ValueError, match="must map each demoted key"):
        ec.expand_rule(bad, CATS, CATS, absolute_units=False)
    bad["per_category"]["demoted"] = {"stand": {"no_such_key": "x"}}
    with pytest.raises(ValueError, match="no_such_key"):
        ec.expand_rule(bad, CATS, CATS, absolute_units=False)


def test_tripod_is_watch_and_no_composite_guard_in_the_campaign_rule(ec, rule):
    exp = ec.expand_rule(rule, CATS, CATS, absolute_units=False)
    assert "cat_tripod_min" not in exp["guards"]["metrics"] and not exp["composite"]
    for c in TRANSLATION + ("move_turn",):
        assert f"cat.{c}.tripod" in exp["watch"], c
    # the per-second rates of the moving categories are WATCH, not Q
    for c in TRANSLATION + ("turn", "move_turn"):
        for key in ("joint_speed_mean", "rod_speed_mean", "contact_switch_hz", "action_fft_freq",
                    "wz_active_frac", "clock_restarts", "foot_slip"):
            assert f"cat.{c}.{key}" in exp["watch"] and f"cat.{c}.{key}" not in exp["blocks"]["Q"]["metrics"]
    # Q at stand: every stand-applicable quality key (12) except the campaign's pre-launch demotion --
    # settle_six_down_s, zero spread on X08:c6 (candidates.yaml decision_rule.per_category.demoted), now WATCH;
    # elsewhere per-distance / per-rad + jitter
    q = exp["blocks"]["Q"]["metrics"]
    assert set(rule["per_category"]["demoted"]) == {"stand"}
    assert set(rule["per_category"]["demoted"]["stand"]) == {"settle_six_down_s"}
    assert len([m for m in q if m.startswith("cat.stand.")]) == 11
    assert "cat.stand.settle_six_down_s" not in q
    assert exp["watch"]["cat.stand.settle_six_down_s"].startswith("demoted: zero spread on X08:c6")
    assert {m.split(".")[2] for m in q if m.startswith("cat.fwd.")} == {
        "action_jitter", "swings_per_m", "slip_per_m", "work_per_m", "cam_work_per_m"}
    assert {m.split(".")[2] for m in q if m.startswith("cat.turn.")} == {
        "action_jitter", "swings_per_rad", "slip_per_rad", "work_per_rad", "cam_work_per_rad"}
    # the speed and slip guards sit beside their block members
    for c in TRANSLATION:
        assert f"cat.{c}.ratio" in exp["guards"]["metrics"] and f"cat.{c}.slip_per_m" in exp["guards"]["metrics"]


def test_every_q_member_is_defined_on_its_category(ec, rule, M):
    """Step 1's N/A convention: a key names only the categories it is defined on."""
    for c, keys in rule["per_category"]["blocks"]["Q"]["members"].items():
        for key in keys:
            assert c in M.MOTION_QUALITY_APPLIES[key], (c, key)


# ================================================================== the validator
def test_the_campaign_yaml_validates_and_a_placeholder_is_refused_by_name(rc, cands, tmp_path):
    """The step-3 weights are filled (REPLAY_GATE.md, 2026-09-23): the campaign yaml validates. A
    `TBD_replay_gate` placeholder put back into a copy is still refused BY NAME."""
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]
    text = CAMPAIGN_YAML.read_text().replace('KRABBY_POWER_W: "-0.00023"', "KRABBY_POWER_W: TBD_replay_gate")
    assert "TBD_replay_gate" in text
    path = tmp_path / "candidates.yaml"
    path.write_text(text)
    bad = {v["id"]: v["reasons"] for v in rc.validate_candidates(rc.load_candidates(path), strict=False,
                                                                check_replay=False) if not v["ok"]}
    assert set(bad) == {"M2"} and all("PLACEHOLDER" in r and "KRABBY_POWER_W" in r for r in bad["M2"]), bad
    with pytest.raises(SystemExit, match="PLACEHOLDER"):
        rc.validate_candidates(rc.load_candidates(path), strict=True, check_replay=False)


def test_the_filled_campaign_yaml_validates(rc, filled):
    verdicts = rc.validate_candidates(copy.deepcopy(filled), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]


def test_the_facing_ladders_live_yaml_still_validates(rc):
    """The required block set is derived from each campaign's own rule, so the six-block rule
    validates exactly as before (and its F_mt-deletion refusal is pinned in the candidates tests)."""
    prior = rc.load_candidates(PRIOR_YAML)
    verdicts = rc.validate_candidates(prior, strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]
    assert rc.rule_v2_required_blocks(prior["decision_rule"]) == ["F", "F_mt", "T", "F_abs",
                                                                  "F_mt_abs", "T_abs"]


def _refusals(rc, doc, subject=None) -> list[str]:
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    return [r for v in verdicts if subject is None or v["id"] == subject for r in v["reasons"]]


def test_the_required_block_set_is_derived_from_the_rule(rc, filled):
    assert rc.rule_v2_required_blocks(filled["decision_rule"]) == ["F", "F_mt", "T", "Q"]
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["blocks"]["Q"]["members"] = {}
    assert any("blocks.Q has no member" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["blocks"]["R"] = {"level": 0.95, "units": "ratio",
                                                           "members": {"fwd": ["ratio"]}}
    assert any("outside" in r and "'R'" in r for r in _refusals(rc, doc, "<campaign>"))
    # naming ONE absolute block pulls in the whole absolute system (the facing ladder's rule)
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["blocks"]["T_abs"] = {
        "level": 0.95, "units": "absolute", "members": {"fwd": ["achieved_speed"]}}
    got = _refusals(rc, doc, "<campaign>")
    assert any("blocks.F_abs has no member" in r for r in got) and any(
        "blocks.F_mt_abs has no member" in r for r in got), got


def test_adopt_requires_and_demoted_are_validated(rc, filled):
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["adopt_requires"] = "Z"
    assert any("adopt_requires names 'Z'" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["demoted"] = {"stand": {"drift_speed": "x"}}
    assert any("demotes a member no block or guard scores" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["demoted"] = {"stand": {"clock_restarts": "zero spread"}}
    assert not _refusals(rc, doc, "<campaign>")
    # an absolute-units candidate can never satisfy adopt_requires Q (no Q_abs, no *_abs at all)
    doc = copy.deepcopy(filled)
    doc["candidates"][0]["absolute_units"] = True
    got = _refusals(rc, doc, "M1")
    assert any("carries no *_abs block" in r for r in got) and any("no Q_abs twin" in r for r in got)


def test_the_checkpoint_resume_kind_is_validated(rc, filled):
    stage = lambda d: d["chains"]["fork_c6"]["stages"][0]                  # noqa: E731
    assert rc.resume_kind(stage(filled)) == "checkpoint"
    assert rc.resume_checkpoint(stage(filled)).endswith("_X08_c6_A15pB/model_7996.pt")
    doc = copy.deepcopy(filled)
    stage(doc)["resume_from"] = {"checkpoint": "/abs/model_1.pt"}
    assert any("must be a repo-relative path" in r for r in _refusals(rc, doc, "chain:fork_c6"))
    doc = copy.deepcopy(filled)
    stage(doc)["resume_from"] = {"checkpoint": "parkour/x/model_1.pt", "seed": 2}
    assert any("not in" in r and "checkpoint" in r for r in _refusals(rc, doc, "chain:fork_c6"))
    doc = copy.deepcopy(filled)
    stage(doc)["from_scratch"] = True
    assert any("from_scratch but resume_from" in r for r in _refusals(rc, doc, "chain:fork_c6"))
    doc = copy.deepcopy(filled)
    del stage(doc)["floor"]["min_by_category"]["move_turn"]
    got = _refusals(rc, doc, "chain:fork_c6")
    assert any("min_by_category for every category" in r and "move_turn" in r for r in got), got
    doc = copy.deepcopy(filled)
    stage(doc)["resume_from"] = {"checkpoint": "parkour/not/on/this/machine/model_1.pt"}
    verdict = next(v for v in rc.validate_candidates(doc, strict=False, check_replay=False)
                   if v["id"] == "chain:fork_c6")
    assert verdict["ok"] and any("not on this machine" in n for n in verdict["notes"])


def test_await_if_and_the_promotion_switch_are_validated(rc, filled):
    doc = copy.deepcopy(filled)
    m2 = rc.candidate_by_id(doc, "M2")
    m2["depends_on"] = []
    assert any("await_if names decision(M1)" in r for r in _refusals(rc, doc, "M2"))
    doc = copy.deepcopy(filled)
    rc.candidate_by_id(doc, "M2")["await_if"] = "decision(M1).nope == ADOPT"
    assert any("await_if does not parse" in r for r in _refusals(rc, doc, "M2"))
    doc = copy.deepcopy(filled)
    doc["milestones"] = [{"id": "Z1", "after": "M1", "promote": ["2a", "2b"]}]
    assert any("promotion.enabled is false" in r for r in _refusals(rc, doc, "<campaign>"))


# ================================================================== the recipe: fork identity
def test_the_fork_path_is_hashed_into_the_recipe_id(rc, filled):
    led = {"schema": 1, "entries": []}
    a = rc.compose_recipe(led, filled, "C", upto="M1")
    assert a["resume_checkpoints"] == {"c6ft": rc.resume_checkpoint(filled["chains"]["fork_c6"]["stages"][0])}
    other = copy.deepcopy(filled)
    other["chains"]["fork_c6"]["stages"][0]["resume_from"] = {"checkpoint": "parkour/y/model_7996.pt"}
    b = rc.compose_recipe(led, other, "C", upto="M1")
    assert a["recipe_id"] != b["recipe_id"]
    assert a["overlay_by_stage"] == b["overlay_by_stage"]           # ONLY the fork differs
    want = rc.sha256_text(rc.canonical({"chain": "fork_c6", "overlay_by_stage": a["overlay_by_stage"],
                                        "resume_checkpoints": a["resume_checkpoints"]}))[:12]
    assert a["recipe_id"] == want


def test_a_recipe_without_a_fork_keeps_its_recorded_identity(rc, H):
    """The facing ladder's recipes hash exactly {chain, overlay_by_stage}, as they always did."""
    cands = rc.load_candidates(H.CANDS_YAML)
    rec = rc.compose_recipe({"schema": 1, "entries": []}, cands, "C", upto=rc.candidate_ids(cands)[0])
    assert "resume_checkpoints" not in rec
    assert rec["recipe_id"] == rc.sha256_text(rc.canonical(
        {"chain": rec["chain"], "overlay_by_stage": rec["overlay_by_stage"]}))[:12]


def test_disabled_promotions_compose_empty_teacher_overlays(rc, filled):
    led = {"schema": 1, "entries": []}
    for cid in rc.candidate_ids(filled):
        rec = rc._recipe_with(led, filled, "C", cid, set())
        assert rec["overlay_by_stage"]["2a"] == {} and rec["overlay_by_stage"]["2b"] == {}, cid
    assert rc.promotions_enabled(filled) is False
    assert rc.promotions_enabled({"promotion": {"eval": ["suite"]}}) is True


def test_both_baseline_seeds_fork_the_same_checkpoint(rc, H, M, filled, tmp_path, monkeypatch):
    """X0 s3 and X0 s2 are one recipe: both windows resume from the named head (absolute), neither
    arms the from-scratch gates, and the per-category floor reads min_by_category."""
    ckpt = tmp_path / "heads" / "model_7996.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("weights")
    text = _filled_text().replace(
        "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/model_7996.pt}",
        f"{ckpt}}}")
    yaml_path = tmp_path / "cands" / "candidates.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(text)
    cands = rc.load_candidates(yaml_path)
    ctx = H.make_ctx(rc, cands, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL,
                     resolution={"rule_version": 2})
    calls = H.install_fakes(rc, monkeypatch, tmp_path, eval_summary=qscreen(H, M))
    heads = []
    for seed in ("3", "2"):
        item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
        assert item["kind"] == "baseline" and item["seed"] == seed and item["label"] == f"X0 s{seed}"
        header = rc.run_experiment(ctx, item)
        assert header["status"] == "ok", header.get("halt")
        rc.dispatch(ctx, item, header)
        heads.append(header)
    assert [c["resume"] for c in calls["train"]] == [str(ckpt), str(ckpt)]
    assert all(not c["soundness"] and not c["early_abort"] for c in calls["train"])
    assert all(c["iters"] == 2000 for c in calls["train"])
    assert heads[0]["recipe_id"] == heads[1]["recipe_id"]
    rec = ctx.c.recs[f"{heads[-1]['xid']}:c6ft"]            # recs are keyed per seed: s2's is current
    assert rec["floor"]["passed"] and "min >= 0.79" in rec["floor"]["why"], rec["floor"]
    # the next item is M1 on the same fork (its base's screen now exists)
    nxt = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["kind"] == "candidate" and nxt["candidate"] == "M1"
    assert rc.resume_checkpoint(nxt["stages"][0]) == str(ckpt)


def test_a_missing_fork_checkpoint_halts_before_any_training(rc, H, filled, tmp_path, monkeypatch):
    text = _filled_text().replace(
        "parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/model_7996.pt}",
        "parkour/no/such/run/model_7996.pt}")
    yaml_path = tmp_path / "c" / "candidates.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(text)
    cands = rc.load_candidates(yaml_path)
    ctx = H.make_ctx(rc, cands, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL,
                     resolution={"rule_version": 2})
    calls = H.install_fakes(rc, monkeypatch, tmp_path)
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, item)
    assert not calls["train"]
    header = next(iter(rc.experiments(ctx.state).values()))
    assert header["status"] == "halt:resume_head_missing" or "resume_head_missing" in str(header.get("halt"))


# ================================================================== decide: sign, Q, guards
def test_a_lower_jitter_reads_better_and_a_higher_one_worse(rc, ec, H, M, rule):
    """The sign is applied where the delta is formed: every Q member 1 R_eval LOWER on the candidate
    gives z = +1 (> the block R 0.5) -> Q BETTER -> a clear ADOPT under adopt_requires."""
    resolution = qres(rc, ec, rule)
    base = qscreen(H, M, quality=1.0, six_down=0.5)
    better = qscreen(H, M, quality=0.9, six_down=0.6)
    out = _decide(rc, rule, better, base, resolution)
    q = out["evidence"]["seed3"]["blocks"]["Q"]
    assert q["class"] == "BETTER" and q["z"] == pytest.approx(1.0)
    jit = next(r for r in q["rows"] if r["metric"] == "cat.stand.action_jitter")
    assert jit["a"] == pytest.approx(0.9) and jit["b"] == pytest.approx(1.0)
    assert jit["delta"] == pytest.approx(0.1) and jit["lower_is_better"] is True
    six = next(r for r in q["rows"] if r["metric"] == "cat.stand.six_down_frac")
    assert six["delta"] == pytest.approx(0.1) and "lower_is_better" not in six
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("ADOPT", "clear", "decided", False)
    worse = qscreen(H, M, quality=1.1, six_down=0.4)
    out = _decide(rc, rule, worse, base, resolution)
    assert out["evidence"]["seed3"]["blocks"]["Q"]["class"] == "WORSE"
    assert (out["chosen"], out["confidence"]) == ("REJECT", "clear") and "block Q" in out["why"]


def test_q_tie_with_another_block_better_is_a_clear_reject(rc, ec, H, M, rule):
    resolution = qres(rc, ec, rule)
    base = qscreen(H, M)
    faster = qscreen(H, M, ratio=0.66)                       # T z = +0.6 > 0.5: BETTER; Q TIE
    out = _decide(rc, rule, faster, base, resolution)
    blocks = out["evidence"]["seed3"]["blocks"]
    assert blocks["T"]["class"] == "BETTER" and blocks["Q"]["class"] == "TIE"
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("REJECT", "clear", "decided", False)
    assert "ADOPT requires Q BETTER" in out["why"] and out["evidence"]["adopt_requires"] == {
        "block": "Q", "class": "TIE"}


def test_all_tie_including_q_is_a_clear_reject_with_no_pair(rc, ec, H, M, rule):
    resolution = qres(rc, ec, rule)
    out = _decide(rc, rule, qscreen(H, M, quality=0.99), qscreen(H, M), resolution)
    assert all(b["class"] == "TIE" for b in out["evidence"]["seed3"]["blocks"].values())
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("REJECT", "clear", "decided", False)
    assert "all TIE" in out["why"]
    # WITHOUT the flag the same comparison is the old speculative all-TIE with its pair queued
    legacy = {k: v for k, v in rule.items() if k != "adopt_requires"}
    out = _decide(rc, legacy, qscreen(H, M, quality=0.99), qscreen(H, M), resolution)
    assert (out["status"], out["pair"], out["confidence"]) == ("speculative", True, "uncertain")


def test_an_unmeasured_q_is_an_uncertain_reject_never_a_clear_one(rc, ec, H, M, rule):
    # Review 2026-09-23: every Q member unmeasured on the candidate is not evidence against it.
    resolution = qres(rc, ec, rule)
    blank = qscreen(H, M, quality=None, six_down=None)
    out = _decide(rc, rule, blank, qscreen(H, M), resolution)
    assert out["evidence"]["seed3"]["blocks"]["Q"]["k"] == 0     # classify() still reads it TIE (pinned)
    assert (out["chosen"], out["confidence"]) == ("REJECT", "uncertain"), out["why"]
    # the adopt_requires helper names an all-unmeasured required block MISSING, not TIE
    body = out["evidence"]["seed3"]["blocks"]["Q"]
    assert rc._required_block_class({"blocks": {"Q": body}}, "Q") == "MISSING"
    assert rc._required_block_class({"blocks": {}}, "Q") == "absent"
    assert rc._required_block_class({"blocks": {"Q": {"class": "TIE", "k": 3, "missing": []}}}, "Q") == "TIE"


def test_q_better_with_a_missing_member_is_speculative_and_the_pooled_pick_needs_pooled_q(rc, ec, H, M, rule):
    resolution = qres(rc, ec, rule)
    base = qscreen(H, M)
    cand = qscreen(H, M, quality=0.9, six_down=0.6, cat__stand__clock_restarts=None)
    out = _decide(rc, rule, cand, base, resolution)
    assert (out["chosen"], out["status"], out["pair"]) == ("ADOPT", "speculative", True)
    prior = {"chosen": "ADOPT", "status": "speculative"}
    # pooled: s2 is a TIE on Q (the s3 gain does not replicate), T BETTER on both seeds
    s3 = qscreen(H, M, quality=0.99, ratio=0.66)
    s2 = qscreen(H, M, quality=1.0, ratio=0.66)
    pooled = rc.decide_pooled(s3, s2, base, base, resolution, cand_spec={}, rule=rule, prior=prior)
    assert pooled["evidence"]["pooled"]["blocks"]["T"]["class"] == "BETTER"
    assert pooled["evidence"]["pooled"]["blocks"]["Q"]["class"] == "TIE"
    assert (pooled["chosen"], pooled["pooled_confidence"]) == ("REJECT", "clear")
    assert pooled["self_overrule"] and "ADOPT requires Q BETTER" in pooled["why"]
    # pooled Q BETTER -> a clear pooled ADOPT, the s3 pick stands
    s3 = qscreen(H, M, quality=0.93, six_down=0.57)
    s2 = qscreen(H, M, quality=0.94, six_down=0.56)
    pooled = rc.decide_pooled(s3, s2, base, base, resolution, cand_spec={}, rule=rule, prior=prior)
    assert (pooled["chosen"], pooled["pooled_confidence"], pooled["status"]) == ("ADOPT", "clear", "final")


def test_the_slip_guard_rejects_a_gait_that_swings_less_but_drags_its_feet(rc, ec, H, M, rule):
    """Q can be BETTER (fewer swings, less work everywhere) while fwd's slip per metre rises past
    its OWN one-sided guard R (the ``@guard`` key, 0.20 here -- not the block member's 0.10)."""
    resolution = qres(rc, ec, rule, guard_key_R=0.20)
    base = qscreen(H, M)
    cand = qscreen(H, M, quality=0.85, six_down=0.65, cat__fwd__slip_per_m=1.25)
    out = _decide(rc, rule, cand, base, resolution)
    assert out["evidence"]["seed3"]["blocks"]["Q"]["class"] == "BETTER"
    g = next(r for r in out["evidence"]["seed3"]["guards"]["rows"] if r["metric"] == "cat.fwd.slip_per_m")
    assert g["R_eval"] == pytest.approx(0.20) and g["r_key"] == "cat.fwd.slip_per_m@guard"
    assert g["delta"] == pytest.approx(-0.25) and g["class"] == "WORSE"
    assert out["chosen"] == "REJECT" and "guard cat.fwd.slip_per_m" in out["why"]
    # inside its own R (0.20) but outside the block member's (0.10): the guard reads TIE
    cand = qscreen(H, M, quality=0.85, six_down=0.65, cat__fwd__slip_per_m=1.15)
    g = next(r for r in _decide(rc, rule, cand, base, resolution)["evidence"]["seed3"]["guards"]["rows"]
             if r["metric"] == "cat.fwd.slip_per_m")
    assert g["class"] == "TIE"


def test_the_speed_guard_catches_a_slower_shuffle_that_t_reads_as_a_tie(rc, ec, H, M, rule):
    """D12's case: side ratio 0.521 -> 0.430 inside T's block R. The per-category speed guard is its
    own guard spec on the same metric and rejects it; the other translation ratios hold T at TIE."""
    resolution = qres(rc, ec, rule, guard_key_R=0.05)
    base = qscreen(H, M)
    cand = qscreen(H, M, quality=0.85, six_down=0.65, cat__side__ratio=0.52)
    out = _decide(rc, rule, cand, base, resolution)
    assert out["evidence"]["seed3"]["blocks"]["T"]["class"] == "TIE"
    assert out["evidence"]["seed3"]["blocks"]["Q"]["class"] == "BETTER"
    g = next(r for r in out["evidence"]["seed3"]["guards"]["rows"] if r["metric"] == "cat.side.ratio")
    assert g["class"] == "WORSE" and g["r_key"] == "cat.side.ratio@guard"
    assert out["chosen"] == "REJECT"


def test_a_lower_tripod_is_watch_and_never_rejects(rc, ec, H, M, rule):
    resolution = qres(rc, ec, rule)
    base = qscreen(H, M)
    cand = qscreen(H, M, quality=0.9, six_down=0.6)
    for c in TRANSLATION:
        cand["cat"][c]["tripod"] = 0.2
    out = _decide(rc, rule, cand, base, resolution)
    assert out["chosen"] == "ADOPT"
    watch = out["evidence"]["seed3"]["watch"]
    assert watch["cat.fwd.tripod"]["a"] == pytest.approx(0.2)
    assert all(r["metric"] != "cat_tripod_min" for r in out["evidence"]["seed3"]["guards"]["rows"])


def test_the_decision_line_names_the_adoption_rule_and_the_report_the_sign(rc, ec, H, M, rule, tmp_path, filled):
    resolution = qres(rc, ec, rule)
    decision = _decide(rc, rule, qscreen(H, M, ratio=0.66), qscreen(H, M), resolution)
    ctx = H.make_ctx(rc, filled, tmp_path, led=H.ledger(), st=H.state(), resolution=resolution,
                     eval_sha=EVAL)
    e = {"id": "D05", "experiment": "X03", "chosen": decision["chosen"], "status": decision["status"],
         "confidence": decision["confidence"], "alternative": decision["alternative"],
         "candidate": "M1"}
    item = rc._item("candidate", cid="M1", seed="3", recipe={"overlay_by_stage": {}},
                    base_recipe={"recipe_id": "b"})
    fields = rc.decision_notify_fields(ctx, e, item, decision)
    assert any(f.startswith("adopt_requires Q: TIE") for f in fields), fields
    assert any(f.startswith("Q z=") and "jitter" in f for f in fields), fields
    report = rc._decision_report(ctx, e, item, decision)
    assert "cat.stand.action_jitter: 1.000 -> 1.000 (lower is better, signed delta" in report


# ================================================================== resolution: specs, sign, D00s
def test_specs_keep_a_separate_guard_spec_for_a_metric_that_is_also_a_block_member(res, ec, rule):
    union = ec.expand_rule_union(rule, CATS, CATS)
    labels = res._manifest_scenario("suite__omni")["hold_labels"]
    specs, unresolved = res.specs_from_rule(union, labels, suite=True)
    by = {s.mid: s for s in specs}
    for c in TRANSLATION:
        blk, grd = by[f"cat.{c}.ratio"], by[f"cat.{c}.ratio@guard"]
        assert (blk.block, blk.one_sided, blk.level) == ("T", False, 0.95)
        assert (grd.block, grd.one_sided, grd.level, grd.blocks_all) == ("guard", True, 0.99, ())
        assert grd.needs == blk.needs and grd.sign == 1
        slip = by[f"cat.{c}.slip_per_m@guard"]
        assert (slip.block, slip.one_sided, slip.sign) == ("guard", True, -1)
        assert by[f"cat.{c}.slip_per_m"].block == "Q" and by[f"cat.{c}.slip_per_m"].sign == -1
    assert "cat.move_turn.ratio@guard" not in by and by["cat.move_turn.ratio"].block == "guard"
    assert by["cat.stand.six_down_frac"].sign == 1 and by["cat.stand.action_jitter"].sign == -1
    assert res.dual_guards(union) == {f"cat.{c}.{k}" for c in TRANSLATION for k in ("ratio", "slip_per_m")}


@pytest.fixture(scope="module")
def q_heads(tmp_path_factory, M):
    """Two synthetic suite stamps (the resolution tests' mini-harness), X08-c6 and X08-c4."""
    tres = _load(REPO / "tests/unit/test_crab_hex_omni_resolution.py", "omni_resolution_helpers_for_q")
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import report as RPT
    base = tmp_path_factory.mktemp("q_heads")
    rng = np.random.default_rng(5)
    out = {}
    for head, kw in (("X08-c6", dict(closes=0.7, fall_offset=0)),
                     ("X08-c4", dict(closes=0.5, ratio=0.55, fall_offset=1))):
        root = base / f"reeval_{head}"
        run = tres._stamp_dir(root)
        run.parent.mkdir(parents=True)
        tres.make_suite_run(run, head=head, rng=rng, ge_mods=(RPT, M), **kw)
        out[head] = (root, run)
    return tres, RPT, out


def _set_quality(RPT, run: Path, cat: str, key: str, fn) -> None:
    """Rewrite one quality key on every episode of ``cat`` (``fn(env_index) -> value``) and rebuild
    the aggregate through the harness's own ``report.aggregate``, so the stamp stays self-consistent
    (the reducer check still reproduces ``summarize()``)."""
    meta = json.loads((run / "run_meta.json").read_text())
    episodes = []
    for f in sorted((run / "metrics").glob("episode_*.json")):
        ep = json.loads(f.read_text())
        if ep.get("category") == cat:
            ep["holds"][ep["hold"]]["quality"][key] = fn(int(ep["env_index"]))
            f.write_text(json.dumps(ep, sort_keys=True))
        episodes.append(ep)
    episodes.sort(key=lambda e: int(e["env_index"]))
    sm = json.loads((run / "scenario_metrics.json").read_text())
    sm["aggregate"] = RPT.aggregate(episodes, suite=RPT.suite_aggregate_arg(meta))
    (run / "scenario_metrics.json").write_text(json.dumps(sm, sort_keys=True))


def test_d00s_under_the_campaign_rule_resolves_q_and_the_guard_keys(res, ec, rule, q_heads):
    tres, _RPT, heads = q_heads
    entries = [tres._pair_entry(root, head, run, preset="1a") for head, (root, run) in heads.items()]
    rule = dict(rule, bootstrap={"B": 200, "paired_on": "env_index", "rng_seed": 3},
                control_head="X08-c6")
    payload = res.resolve({"resolution_inputs": entries, "decision_rule": rule}, eval_sha=EVAL,
                          decisive_at=list(CATS), supersedes="D00")
    m = payload["metrics"]
    assert payload["blocks"]["Q"]["R_eval"] > 0
    for c in TRANSLATION:
        g, b = m[f"cat.{c}.ratio@guard"], m[f"cat.{c}.ratio"]
        assert (g["block"], g["one_sided"], g["level"]) == ("guard", True, 0.99)
        assert (b["block"], b["one_sided"]) == ("T", False)
        assert g["R_eval"] > 0 and g["R_eval"] != b["R_eval"]      # its own (99 %, one-sided) quantile
        assert m[f"cat.{c}.slip_per_m@guard"]["lower_is_better"] is True
    # the recorded point delta is SIGNED: candidate lower = positive
    jit = m["cat.stand.action_jitter"]
    assert jit["lower_is_better"] is True
    pt = next(iter(jit["points"].values()))
    assert pt["delta"] == pytest.approx(-(pt["a"] - pt["b"]))
    rat = next(iter(m["cat.fwd.ratio"]["points"].values()))
    assert rat["delta"] == pytest.approx(rat["a"] - rat["b"])
    assert "lower_is_better" not in m["cat.fwd.ratio"]
    # completeness: every Q member and every guard key carries a finite positive R
    fails = [f for f in payload["acceptance"]["failures"] if "R_eval" in f]
    assert not fails, fails
    # the null scores the dual guards under their own keys
    scored = payload["diagnostics"]["null_scored"]["guards"]
    assert "cat.fwd.ratio@guard" in scored and "cat.fwd.slip_per_m@guard" in scored
    assert "cat.fwd.ratio" not in scored


def test_the_spread_check_flags_a_zero_spread_member_and_prints_its_demotion(rc, res, rule, q_heads,
                                                                            tmp_path, filled):
    tres, RPT, heads = q_heads
    import shutil
    runs = {}
    for head, (root, run) in heads.items():
        dst = tmp_path / head / run.name
        shutil.copytree(run, dst)
        ck = tres.ckpt_of(run)
        if ck.exists():
            shutil.copytree(ck.parent, tres.ckpt_of(dst).parent)
        runs[head] = dst
    # clock restarts at stand: 0 on every episode of both heads -> no resample moves the median
    for dst in runs.values():
        _set_quality(RPT, dst, "stand", "clock_restarts", lambda i: 0.0)
    rule = dict(rule, bootstrap={"B": 200, "paired_on": "env_index", "rng_seed": 3})
    out = res.member_spread_check([runs["X08-c6"], runs["X08-c4"]], rule, heads=["X08-c6", "X08-c4"])
    assert out["zero"] == ["cat.stand.clock_restarts"], out["zero"]
    assert not out["absent"]
    assert "clock_restarts" in out["demote"]["stand"]
    assert "zero spread on X08-c6" in out["demote"]["stand"]["clock_restarts"]
    ok = out["members"]["cat.stand.action_jitter"]
    assert not ok["zero"] and ok["cross"] and all(v > 0 for v in ok["cross"].values())
    assert out["members"]["cat.fwd.ratio@guard"]["zero"] is False
    # the CLI: exit 3 and the yaml lines to paste; the demotion then validates and expands
    code, lines = rc.q_spread_report(filled, [str(runs["X08-c6"]), str(runs["X08-c4"])],
                                     ["X08-c6", "X08-c4"])
    assert code == 3 and "    demoted:" in lines and "      stand:" in lines
    doc = copy.deepcopy(filled)
    doc["decision_rule"]["per_category"]["demoted"] = out["demote"]
    assert not _refusals(rc, doc, "<campaign>")
    exp = rc.eval_categories().expand_rule(doc["decision_rule"], CATS, CATS, absolute_units=False)
    assert "cat.stand.clock_restarts" not in exp["blocks"]["Q"]["metrics"]
    # a member with NO value on a stamp is 'absent' (re-evaluate), never a demotion
    for dst in runs.values():
        _set_quality(RPT, dst, "stand", "wz_active_frac", lambda i: None)
    out = res.member_spread_check([runs["X08-c6"], runs["X08-c4"]], rule)
    assert "cat.stand.wz_active_frac" in out["absent"]
    assert "wz_active_frac" not in (out["demote"].get("stand") or {})


# ================================================================== the stage floor
def test_the_fork_floor_reads_min_by_category(rc, filled, H, M):
    stage = rc.chain_stages(filled, "fork_c6")[0]
    rec = {"evals": {"suite": qscreen(H, M, surv=0.95)}}
    passed, why = rc.stage_floor(stage, rec, decisive=list(CATS), previous=[])
    assert passed and "stand 0.950" in why and "min >= 0.79" in why and "new" not in why
    # a collapse at move_turn (0.30 on 100 envs: bound 0.407 < 0.55) fails; min_new would not
    scr = qscreen(H, M, surv=0.95)
    scr["cat"]["move_turn"]["survival"] = 0.30
    passed, why = rc.stage_floor(stage, {"evals": {"suite": scr}}, decisive=list(CATS), previous=[])
    assert not passed and "FAIL on ['move_turn']" in why
    legacy = dict(stage, floor={"kind": "category_survival", "row": "suite", "min_new": 0.10,
                                "min_prev": 0.30})
    assert rc.stage_floor(legacy, {"evals": {"suite": scr}}, decisive=list(CATS), previous=[])[0]


# ================================================================== the branch-after-M1 gate
def test_m2_to_m4_park_for_the_user_only_when_m1_did_not_adopt(rc, filled):
    rej = rc.simulate_queue(filled, outcome="REJECT", code="C")
    assert [r["label"] for r in rej] == ["X0 s3", "X0 s2", "M1 s3", "await_user:M2"]
    assert "decision(M1).chosen=REJECT != ADOPT" in rej[-1]["why"]
    assert "decide M2 --choose run" in rej[-1]["why"]
    ado = rc.simulate_queue(filled, outcome="ADOPT", code="C")
    labels = [r["label"] for r in ado]
    assert labels[:6] == ["X0 s3", "X0 s2", "M1 s3", "M2 s3", "M3 s3", "M4 s3"]
    assert labels[-1] == "await_user:M7"                      # M7 always asks (user_decision)


def test_a_user_run_releases_the_gate(rc, filled, H):
    led = H.ledger()
    st = H.state()
    for _ in range(4):
        item = rc.next_item(filled, led, st, code="C")
        if item["kind"] == "await":
            break
        rc._simulate_apply(filled, led, st, item, "REJECT", "C")
    assert item["label"] == "await_user:M2"
    st["ladder"]["user_choice"]["M2"] = {"choice": "run", "at": "2099-01-01T00:00:00"}
    item = rc.next_item(filled, led, st, code="C")
    assert item["kind"] == "candidate" and item["candidate"] == "M2"
