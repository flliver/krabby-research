"""The candidate-ladder driver: the decision rule, the recipe, the derived squash view, the queue,
the validator, the ledger transitions and the notification contract.

Pure Python. ``run_candidates.py``, ``ledger_cli.py`` and the campaign's own ``candidates.yaml`` are
loaded by path; nothing here imports Isaac, launches a subprocess (the two exceptions are the
replay script's ``--help``, which the validator reads, and a monkeypatched ``train_retry`` standing
in for a training child) or touches the real campaign directory.

The module is organised the way the plan's test paragraph is: ``decide`` / ``decide_pooled`` /
``compose_recipe`` / ``squash_set`` / ledger transitions / ``user_choice`` / recipe-head lookup /
``code_sha`` / ``stage_env`` / the validator / ``format_notify`` / ``next_item`` ordering / arming /
resume / the milestone loop. Every fixture is a dict of the shape ``summarize()`` really returns, so
a decision test fails when the DECISION changes and not when an unrelated key moves.
"""
import copy
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TOOLS = OMNI / "experiments/tools"
DRIVER = TOOLS / "run_candidates.py"
CLI = TOOLS / "ledger_cli.py"
LADDER = OMNI / "experiments/2026-09-19_facing_tracking_ladder"
CANDS_YAML = LADDER / "candidates.yaml"

F_HOLDS = ("turn_l90", "turn_r90", "move_turn")
T_HOLDS = ("fwd", "lat_l", "diag", "back")
GUARDS = ("completion", "tripod_walking_dir", "turn_survival_l90", "turn_survival_r90")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(DRIVER, "run_candidates_under_test")


@pytest.fixture(scope="module")
def cli():
    return _load(CLI, "ledger_cli_under_test")


@pytest.fixture(scope="module")
def cands(rc):
    return rc.load_candidates(CANDS_YAML)


# ---------------------------------------------------------------- fixtures: screens / resolution
def _put(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def screen(*, closed=0.50, prog=0.080, ratio=0.60, achieved=0.150, net=0.050, completion=0.50,
           tripod=0.40, surv_l=0.95, surv_r=0.95, **over) -> dict:
    """A ``summarize()``-shaped screen. ``over`` takes DOTTED metric ids (the decision rule's own)."""
    out = {
        "facing_closed_by_hold": {h: closed for h in F_HOLDS},
        "yaw_progress_by_hold": {h: prog for h in F_HOLDS},
        "net_yaw_by_hold": {h: net for h in F_HOLDS},
        "ratio_by_hold": {h: ratio for h in T_HOLDS},
        "speed_ratio_by_hold": {h: ratio for h in T_HOLDS},
        "achieved_speed_by_hold": {h: achieved for h in T_HOLDS},
        "completion": completion, "tripod_walking_dir": tripod,
        "turn_survival_l90": surv_l, "turn_survival_r90": surv_r,
        "tripod_by_hold": {"move_turn": 0.48},
        "facing_by_hold": {h: {"settle": 9.8} for h in F_HOLDS},
        "survival_by_hold": {h: 0.9 for h in ("back",) + F_HOLDS},
        "track_ratio_tail": 0.645, "track_ratio_move_tail": 0.71,
        "yaw_progress_turn_min": prog,
    }
    for key, value in over.items():
        _put(out, key.replace("__", "."), value)
    return out


def resol(*, metric_R=0.10, guard_R=0.10, block_R=0.50) -> dict:
    """A synthetic resolution record with ROUND numbers, so every z is checkable by hand.

    At ``metric_R`` 0.10 a uniform delta ``d`` on a block's members gives ``z = 10 d`` exactly, so
    ``block_R`` 0.50 means "BETTER above +0.05 on every member".
    """
    mids = ([f"facing_closed_by_hold.{h}" for h in F_HOLDS]
            + [f"yaw_progress_by_hold.{h}" for h in F_HOLDS]
            + [f"net_yaw_by_hold.{h}" for h in F_HOLDS]
            + [f"ratio_by_hold.{h}" for h in T_HOLDS]
            + [f"achieved_speed_by_hold.{h}" for h in T_HOLDS])
    metrics = {mid: {"R_eval": metric_R, "level": 0.95, "n": 100, "source": "metric"} for mid in mids}
    metrics.update({g: {"R_eval": guard_R, "level": 0.99, "n": 100, "source": "json"} for g in GUARDS})
    blocks = {b: {"R_eval": block_R, "level": 0.95, "members": []} for b in ("F", "T", "F_abs", "T_abs")}
    return {"schema": 1, "B": 2000, "rng_seed": 20260919, "metrics": metrics, "blocks": blocks,
            "process_floor": {}, "seed_terms": {}, "inputs": []}


@pytest.fixture
def rule(cands):
    return cands["decision_rule"]


def dec(rc, rule, cand_screen, base_screen, *, cand_spec=None, anchor=None, terms=None,
        resolution=None):
    return rc.decide(cand_screen, base_screen, resolution or resol(), cand_spec=cand_spec or {},
                     rule=rule, seed_terms=terms or {}, anchor_screen=anchor,
                     anchor_label="A" if anchor else None)


# ================================================================== decide
def test_clear_adopt_is_decided_with_no_pair(rc, rule):
    """F BETTER, nothing WORSE, drift n/a -> a clear ADOPT: decided, and no seed-2 pair is queued."""
    out = dec(rc, rule, screen(closed=0.72, prog=0.11), screen())
    assert out["chosen"] == "ADOPT"
    assert out["confidence"] == "clear"
    assert out["status"] == "decided"
    assert out["pair"] is False
    assert out["evidence"]["seed3"]["blocks"]["F"]["class"] == "BETTER"
    assert out["evidence"]["seed3"]["blocks"]["T"]["class"] == "TIE"


def test_clear_reject_on_a_block(rc, rule):
    out = dec(rc, rule, screen(ratio=0.40), screen())
    assert (out["chosen"], out["confidence"], out["status"]) == ("REJECT", "clear", "decided")
    assert out["evidence"]["seed3"]["blocks"]["T"]["class"] == "WORSE"
    assert out["pair"] is False


@pytest.mark.parametrize("guard,value", [("completion", 0.30), ("tripod_walking_dir", 0.20),
                                         ("turn_survival_l90", 0.70), ("turn_survival_r90", 0.60)])
def test_clear_reject_on_each_guard(rc, rule, guard, value):
    """All four guards reject on their own -- including the two CONDITIONAL turn survivals."""
    out = dec(rc, rule, screen(closed=0.72, **{guard: value}), screen())
    assert (out["chosen"], out["confidence"]) == ("REJECT", "clear")
    assert guard in out["evidence"]["seed3"]["guards"]["worse"]


def test_a_better_guard_is_never_a_reason_to_adopt(rc, rule):
    """Guards are ONE-SIDED: a guard far above its R leaves an all-TIE screen uncertain."""
    out = dec(rc, rule, screen(completion=0.95), screen())
    assert out["status"] == "speculative"
    assert out["evidence"]["seed3"]["guards"]["worse"] == []


def test_the_block_boundary_is_a_tie(rc, rule):
    """z exactly == R is a TIE; a hair above is BETTER (the classification is strict on both sides)."""
    assert rc.classify(0.5, 0.5) == "TIE"
    assert rc.classify(-0.5, 0.5) == "TIE"
    assert rc.classify(0.5 + 1e-12, 0.5) == "BETTER"
    assert rc.classify(-0.5 - 1e-12, 0.5) == "WORSE"
    assert rc.classify(None, 0.5) == "TIE" and rc.classify(1.0, None) == "TIE"
    # and at screen level: z just inside R is a TIE, so the screen is uncertain, not an adopt
    out = dec(rc, rule, screen(closed=0.54, prog=0.120), screen())
    assert out["evidence"]["seed3"]["blocks"]["F"]["z"] == pytest.approx(0.4, abs=1e-9)
    assert out["evidence"]["seed3"]["blocks"]["F"]["class"] == "TIE"
    assert out["status"] == "speculative"


def test_all_tie_uses_the_sign_of_zf_plus_zt(rc, rule):
    up = dec(rc, rule, screen(closed=0.54), screen())
    assert (up["chosen"], up["confidence"], up["status"], up["pair"]) == ("ADOPT", "uncertain",
                                                                         "speculative", True)
    assert "sign(z_F + z_T)" in up["why"]
    down = dec(rc, rule, screen(closed=0.46), screen())
    assert down["chosen"] == "REJECT"


def test_an_exact_tie_falls_to_the_incumbent(rc, rule):
    out = dec(rc, rule, screen(), screen())
    assert out["chosen"] == "REJECT"
    assert "exact tie -> incumbent" in out["why"]


def test_speculative_default_overrides_the_sign_in_both_directions(rc, rule):
    """``speculative_default`` is the all-TIE PICK, not a tiebreak: it wins against the sign."""
    neg = screen(closed=0.46)
    assert dec(rc, rule, neg, screen(), cand_spec={"speculative_default": "candidate"})["chosen"] == "ADOPT"
    pos = screen(closed=0.54)
    assert dec(rc, rule, pos, screen(), cand_spec={"speculative_default": "incumbent"})["chosen"] == "REJECT"


def test_c1c_declares_the_candidate_default_in_the_shipped_yaml(cands):
    c1c = [c for c in cands["candidates"] if c["id"] == "C1c"][0]
    assert c1c["speculative_default"] == "candidate"


def test_a_missing_guard_is_a_decided_reject_with_no_pair(rc, rule):
    """A pair cannot supply a metric the row never produced, so no pair is queued (decision 5)."""
    out = dec(rc, rule, screen(closed=0.72, turn_survival_r90=None), screen())
    assert (out["chosen"], out["confidence"], out["status"]) == ("REJECT", "uncertain", "decided")
    assert out["pair"] is False
    assert out["evidence"]["missing"]["guards"] == ["turn_survival_r90"]


def test_a_worse_block_still_rejects_when_a_guard_is_missing(rc, rule):
    out = dec(rc, rule, screen(ratio=0.40, completion=None), screen())
    assert (out["chosen"], out["confidence"]) == ("REJECT", "clear")
    assert "block T" in out["why"]


def test_a_missing_block_member_leaves_the_block_and_caps_the_outcome_at_speculative(rc, rule):
    cand = screen(closed=0.72, prog=0.11)
    cand["facing_closed_by_hold"]["move_turn"] = None
    out = dec(rc, rule, cand, screen())
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("ADOPT", "uncertain",
                                                                             "speculative", True)
    f = out["evidence"]["seed3"]["blocks"]["F"]
    assert f["k"] == 5 and f["missing"] == ["facing_closed_by_hold.move_turn"]


def test_a_metric_with_no_resolution_is_missing_not_zero(rc, rule):
    """A metric the resolution record does not carry cannot be standardised, so it LEAVES the block."""
    resolution = resol()
    resolution["metrics"].pop("ratio_by_hold.back")
    out = dec(rc, rule, screen(ratio=0.40), screen(), resolution=resolution)
    assert out["evidence"]["seed3"]["blocks"]["T"]["missing"] == ["ratio_by_hold.back"]
    assert out["evidence"]["seed3"]["blocks"]["T"]["k"] == 3


def test_absolute_units_scores_metres_per_second_and_ignores_the_ratio(rc, rule):
    """A cap candidate changes every ratio's DENOMINATOR, so C3a/C3b are scored in absolute units."""
    spec = {"absolute_units": True}
    blind = dec(rc, rule, screen(ratio=0.20), screen(), cand_spec=spec)
    assert blind["evidence"]["seed3"]["blocks"]["T"]["block"] == "T_abs"
    assert blind["evidence"]["seed3"]["blocks"]["T"]["class"] == "TIE"
    seen = dec(rc, rule, screen(achieved=0.05), screen(), cand_spec=spec)
    assert seen["evidence"]["seed3"]["blocks"]["T"]["class"] == "WORSE"
    net = dec(rc, rule, screen(net=0.30, closed=0.72), screen(), cand_spec=spec)
    assert net["evidence"]["seed3"]["blocks"]["F"]["block"] == "F_abs"
    assert net["evidence"]["seed3"]["blocks"]["F"]["class"] == "BETTER"


def test_both_absolute_blocks_are_declared_by_the_shipped_cap_candidates(cands):
    for cid in ("C3a", "C3b"):
        cand = [c for c in cands["candidates"] if c["id"] == cid][0]
        assert cand["absolute_units"] is True


def test_anchored_drift_failure_turns_a_clear_adopt_into_a_speculative_reject(rc, rule):
    """Drift is a precondition of ANY adopt: the screen is clear, the pick is REJECT, a pair runs."""
    anchor = screen(completion=0.90)
    out = dec(rc, rule, screen(closed=0.72, prog=0.11), screen(), anchor=anchor)
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("REJECT", "uncertain",
                                                                             "speculative", True)
    assert out["evidence"]["anchored_drift"]["verdict"] == "FAIL"
    assert "completion" in str(out["evidence"]["anchored_drift"]["worse"])


def test_anchored_drift_failure_also_forbids_a_speculative_adopt(rc, rule):
    out = dec(rc, rule, screen(closed=0.54, completion=0.30), screen(completion=0.30),
              anchor=screen(completion=0.90))
    assert out["chosen"] == "REJECT"
    assert "forbids an ADOPT" in out["why"]


def test_anchored_drift_is_not_applicable_without_an_anchor(rc, rule):
    out = dec(rc, rule, screen(closed=0.72), screen())
    assert out["evidence"]["anchored_drift"]["verdict"] == "n/a"


def test_the_watch_metrics_are_reported_and_never_scored(rc, rule):
    """``settle`` and the two training-log ratios are in the line and in no block or guard."""
    scored = set()
    for spec in rule["blocks"].values():
        scored |= set(spec["metrics"])
    scored |= set(rule["guards"]["metrics"])
    for mid in ("track_ratio_tail", "track_ratio_move_tail", "ratio_by_hold.move_turn",
                "tripod_by_hold.move_turn", "facing_by_hold.turn_l90.settle",
                "ratio_by_hold.move_turn_r", "ratio_by_hold.lat_r"):
        assert mid in rule["watch"]
        assert mid not in scored
    worse_watch = screen(closed=0.72, track_ratio_tail=0.10, track_ratio_move_tail=0.10)
    worse_watch["tripod_by_hold"]["move_turn"] = 0.01
    worse_watch["facing_by_hold"]["turn_l90"]["settle"] = 99.0
    out = dec(rc, rule, worse_watch, screen())
    assert out["chosen"] == "ADOPT" and out["confidence"] == "clear"
    watch = out["evidence"]["seed3"]["watch"]
    assert watch["track_ratio_tail"]["a"] == 0.10 and watch["track_ratio_tail"]["b"] == 0.645


def test_the_control_pair_submitted_as_a_candidate_is_all_tie(rc, rule):
    """``B0_s2 vs B0_s3`` must decide all-TIE BY CONSTRUCTION (decision 5's pinned property).

    The seed terms are derived from that very pair by ``seed_terms_from_pair`` -- the function the
    driver calls when it writes ``D00b`` -- rather than hand-picked, because the property the plan
    pins is that the pair standardised by its OWN spread cannot read as a signal. Hand-picked terms
    would pin nothing but the numbers in this file. Repeated over random pairs below.
    """
    s3 = screen()
    s2 = screen(closed=0.52, prog=0.081, ratio=0.61, completion=0.51, tripod=0.41)
    terms = rc.seed_terms_from_pair(s3, s2, resol(), rule)
    out = dec(rc, rule, s2, s3, terms=terms)
    assert [v["class"] for v in out["evidence"]["seed3"]["blocks"].values()] == ["TIE", "TIE"]
    assert out["evidence"]["seed3"]["guards"]["worse"] == []
    assert out["status"] == "speculative"


def test_the_control_pair_is_all_tie_over_random_pairs_in_both_directions(rc, rule):
    """The same property, driven: 200 random control pairs, each standardised by its own terms."""
    rng = random.Random(20260919)
    for _ in range(200):
        a = screen(closed=rng.uniform(0.3, 0.7), prog=rng.uniform(0.05, 0.12),
                   ratio=rng.uniform(0.5, 0.7), completion=rng.uniform(0.4, 0.6),
                   tripod=rng.uniform(0.3, 0.5), achieved=rng.uniform(0.1, 0.2),
                   net=rng.uniform(0.03, 0.07))
        b = screen(closed=rng.uniform(0.3, 0.7), prog=rng.uniform(0.05, 0.12),
                   ratio=rng.uniform(0.5, 0.7), completion=rng.uniform(0.4, 0.6),
                   tripod=rng.uniform(0.3, 0.5), achieved=rng.uniform(0.1, 0.2),
                   net=rng.uniform(0.03, 0.07))
        terms = rc.seed_terms_from_pair(a, b, resol(), rule)
        for cand, base in ((b, a), (a, b)):
            out = dec(rc, rule, cand, base, terms=terms)
            blocks = out["evidence"]["seed3"]["blocks"]
            assert [v["class"] for v in blocks.values()] == ["TIE", "TIE"], (blocks, terms)
            assert out["evidence"]["seed3"]["guards"]["worse"] == []


def test_the_block_seed_term_survives_alternating_signs(rc, rule):
    """Decision 6's reason for the rms half: six equal-magnitude ALTERNATING deltas cancel in the
    signed pair statistic, so the block term has to come from the rms of the standardised rows."""
    s3 = screen()
    alt = {}
    for i, hold in enumerate(F_HOLDS):
        sign = 1.0 if i % 2 == 0 else -1.0
        alt[f"facing_closed_by_hold.{hold}"] = 0.50 + sign * 0.10
        alt[f"yaw_progress_by_hold.{hold}"] = 0.080 - sign * 0.10
    s2 = screen(**{k.replace(".", "__"): v for k, v in alt.items()})
    terms = rc.seed_terms_from_pair(s3, s2, resol(), rule)
    rows = rc.metric_rows((rule["blocks"]["F"]["metrics"]), s2, s3, resol(), {})
    assert rc.block_stat(rows)["z"] == pytest.approx(0.0, abs=1e-12), "the signed statistic cancels"
    # rms_i(|delta_i| / R_i) / sqrt(k) with every |delta| == R_i and k == 6
    assert terms["blocks"]["F"] == pytest.approx(1.0 / math.sqrt(6.0), abs=1e-9)
    assert terms["blocks"]["F"] > 0.0


def test_the_block_seed_term_is_never_below_the_pairs_own_statistic(rc, rule):
    """``R_F^seed >= |z_F(pair)|`` -- the max of the two is what the driver writes into D00b."""
    rng = random.Random(7)
    for _ in range(200):
        over = {}
        for hold in F_HOLDS:
            over[f"facing_closed_by_hold__{hold}"] = rng.uniform(0.2, 0.8)
            over[f"yaw_progress_by_hold__{hold}"] = rng.uniform(0.0, 0.16)
        s3, s2 = screen(), screen(**over)
        terms = rc.seed_terms_from_pair(s3, s2, resol(), rule)
        z = rc.block_stat(rc.metric_rows(rule["blocks"]["F"]["metrics"], s2, s3, resol(), {}))["z"]
        assert terms["blocks"]["F"] >= abs(z) - 1e-12


def test_seed_terms_for_prefers_the_chains_own_d00c_and_takes_the_cross_chain_max(rc, cands,
                                                                                 tmp_path):
    """``D00b`` until a chain has its own ``D00c-<chain>``; a cross-chain pooled decision takes the
    element-wise MAX of the two (decision 6) -- never the narrower of two measured spreads."""
    led, st = ledger(), state()
    led["entries"].append({"id": "D00b", "kind": "resolution", "candidate": None, "status": "final",
                           "chosen": "RECORD", "options": ["RECORD"], "history": [],
                           "evidence": {"seed_terms": {"metrics": {"completion": 0.05},
                                                       "blocks": {"F": 0.20, "T": 0.10}}}})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    terms, ids = rc.seed_terms_for(ctx, "scratch_1c")
    assert ids == ["D00b"] and terms["blocks"]["F"] == 0.20
    terms, ids = rc.seed_terms_for(ctx, "curriculum")
    assert ids == ["D00b"], "a chain with no D00c of its own cites D00b"
    ctx.ledger["entries"].append({"id": "D00c-curriculum", "kind": "resolution", "candidate": None,
                                  "status": "final", "chosen": "RECORD", "options": ["RECORD"],
                                  "history": [],
                                  "evidence": {"seed_terms": {"metrics": {"completion": 0.01},
                                                              "blocks": {"F": 0.30, "T": 0.05}}}})
    terms, ids = rc.seed_terms_for(ctx, "curriculum")
    assert ids == ["D00c-curriculum"] and terms["blocks"]["F"] == 0.30
    terms, ids = rc.seed_terms_for(ctx, "curriculum", "scratch_1c")
    assert ids == ["D00c-curriculum", "D00b"]
    assert terms["blocks"] == {"F": 0.30, "T": 0.10}, "element-wise max across the two chains"
    assert terms["metrics"]["completion"] == 0.05


def test_the_seed_term_widens_a_block_and_a_guard(rc, rule):
    """``R_F = max(R_F^eval, R_F^seed)`` and ``R_i = max(R_eval,i, R_seed,i)`` -- never the narrower."""
    cand = screen(closed=0.60, prog=0.09, completion=0.35)
    tight = dec(rc, rule, cand, screen())
    assert tight["evidence"]["seed3"]["blocks"]["F"]["class"] == "BETTER"
    assert "completion" in tight["evidence"]["seed3"]["guards"]["worse"]
    wide = dec(rc, rule, cand, screen(), terms={"blocks": {"F": 5.0}, "metrics": {"completion": 0.5}})
    assert wide["evidence"]["seed3"]["blocks"]["F"]["class"] == "TIE"
    assert wide["evidence"]["seed3"]["guards"]["worse"] == []


def test_the_null_construction_lands_in_the_plans_band(rc, rule):
    """Decision 6's calibration: with eval-only R the all-TIE rate is in [0.85, 0.98].

    The generative model is the one the standardiser assumes -- each metric's delta is centred with a
    95 % half-width of its own ``R_eval,i``, each guard's with a one-sided 99 % half-width of its
    ``R_eval,i`` -- and the block resolution is then ``R_F = 1/sqrt(k)`` (because
    ``sd(z_F) = 1/(1.96 sqrt k)``). The per-block MARGINAL must land at 0.95 and the joint at
    ``0.95^2 x 0.99^4`` ~ 0.87, the independence floor the plan names. A classifier that used the
    wrong R, averaged the wrong thing, or compared a delta against a standardised bound would miss
    the band.
    """
    rng = random.Random(20260919)
    n = 1000
    blocks = {"F": 6, "T": 4}
    resolution = resol(metric_R=0.10, guard_R=0.10)
    for name, k in blocks.items():
        resolution["blocks"][name]["R_eval"] = 1.0 / math.sqrt(k)
    tie_all, tie_block = 0, {"F": 0, "T": 0}
    for _ in range(n):
        cand = screen()
        for hold in F_HOLDS:
            cand["facing_closed_by_hold"][hold] = 0.50 + rng.gauss(0, 0.10 / 1.96)
            cand["yaw_progress_by_hold"][hold] = 0.080 + rng.gauss(0, 0.10 / 1.96)
        for hold in T_HOLDS:
            cand["ratio_by_hold"][hold] = 0.60 + rng.gauss(0, 0.10 / 1.96)
        for guard, base in (("completion", 0.50), ("tripod_walking_dir", 0.40),
                            ("turn_survival_l90", 0.95), ("turn_survival_r90", 0.95)):
            cand[guard] = base + rng.gauss(0, 0.10 / 2.3263)
        out = dec(rc, rule, cand, screen(), resolution=resolution)
        classes = {k2: v["class"] for k2, v in out["evidence"]["seed3"]["blocks"].items()}
        for name in tie_block:
            tie_block[name] += int(classes[name] == "TIE")
        if all(c == "TIE" for c in classes.values()) and not out["evidence"]["seed3"]["guards"]["worse"]:
            tie_all += 1
    assert 0.85 <= tie_all / n <= 0.98, tie_all / n
    for name, count in tie_block.items():
        assert abs(count / n - 0.95) <= 0.03, (name, count / n)


def test_the_metric_alias_resolves_a_record_written_under_the_other_spelling(rc, rule):
    """``speed_ratio_by_hold.fwd`` and ``ratio_by_hold.fwd`` are one quantity under two names."""
    resolution = resol()
    for hold in T_HOLDS:
        resolution["metrics"][f"speed_ratio_by_hold.{hold}"] = resolution["metrics"].pop(
            f"ratio_by_hold.{hold}")
    out = dec(rc, rule, screen(ratio=0.40), screen(), resolution=resolution)
    assert out["evidence"]["seed3"]["blocks"]["T"]["class"] == "WORSE"
    assert out["evidence"]["seed3"]["blocks"]["T"]["missing"] == []


def test_the_decision_is_a_pure_function_of_its_inputs(rc, rule):
    a = dec(rc, rule, screen(closed=0.72), screen())
    b = dec(rc, rule, screen(closed=0.72), screen())
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ================================================================== decide_pooled
def pooled(rc, rule, cand_s3, cand_s2, base_s3, base_s2, *, prior_pick="REJECT", cand_spec=None,
           terms=None, anchor=None, floor=False):
    prior = {"id": "D05", "chosen": prior_pick, "status": "speculative", "options": ["ADOPT", "REJECT", "SKIP"]}
    return rc.decide_pooled(cand_s3, cand_s2, base_s3, base_s2, resol(), cand_spec=cand_spec or {},
                            rule=rule, prior=prior, seed_terms=terms or {}, anchor_s3=anchor,
                            floor_tripped=floor)


def test_pooled_halves_the_eval_resolution_and_can_resolve_a_tie(rc, rule):
    """``R_F^pooled = max(R_F^eval/sqrt 2, R_F^seed)``: a 0.45 z that was a TIE resolves at 0.354."""
    out = pooled(rc, rule, screen(closed=0.545, prog=0.125), screen(closed=0.545, prog=0.125),
                 screen(), screen(), prior_pick="ADOPT")
    assert out["evidence"]["pooled"]["blocks"]["F"]["z"] == pytest.approx(0.45, abs=1e-9)
    assert out["evidence"]["pooled"]["blocks"]["F"]["R"] == pytest.approx(0.5 / math.sqrt(2))
    assert out["evidence"]["pooled"]["blocks"]["F"]["class"] == "BETTER"
    assert (out["chosen"], out["pooled_confidence"], out["status"]) == ("ADOPT", "clear", "final")
    assert out["self_overrule"] is False


def test_a_pooled_verdict_that_contradicts_the_s3_pick_is_a_driver_self_overrule(rc, rule):
    out = pooled(rc, rule, screen(closed=0.545, prog=0.125), screen(closed=0.545, prog=0.125),
                 screen(), screen(), prior_pick="REJECT")
    assert out["chosen"] == "ADOPT"
    assert out["self_overrule"] is True
    assert out["status"] == "overruled"
    assert out["prior"] == "REJECT"


def test_the_pooled_delta_is_the_two_seed_mean(rc, rule):
    """One good seed and one bad seed average to nothing -- and the disagreement is reported."""
    out = pooled(rc, rule, screen(closed=0.54), screen(closed=0.46), screen(), screen())
    f = out["evidence"]["pooled"]["blocks"]["F"]
    assert f["z"] == pytest.approx(0.0, abs=1e-9)
    assert f["class"] == "TIE"
    assert out["pooled_confidence"] == "disputed"
    assert out["chosen"] == "REJECT"
    assert "disagree in sign" in out["why"]


def test_the_seed_disagreement_is_checked_before_the_missing_member_cap(rc, rule):
    """PLAN_AMENDMENTS 23: decision 7's order is pooled CLEAR -> disagree -> default -> sign.

    A block that lost a member is capped at ``uncertain``, i.e. it is NOT a clear verdict -- so it
    must not take the branch ahead of the disagreement check. With the cap first, two seeds whose
    own signs pointed opposite ways were recorded ADOPT (uncertain) instead of
    ``disputed -> incumbent``, and the pooled record showed agreement that did not exist.
    """
    cand_s3 = screen(closed=0.90, prog=0.30)                 # s3 strongly positive
    cand_s2 = screen(closed=0.30, prog=0.02, ratio=0.57)     # s2 negative on both blocks
    for row in (cand_s3, cand_s2):
        row["facing_closed_by_hold"]["move_turn"] = None     # a MISSING block member
    out = pooled(rc, rule, cand_s3, cand_s2, screen(), screen())
    assert out["evidence"]["per_seed"]["s3"]["sign"] * out["evidence"]["per_seed"]["s2"]["sign"] < 0
    assert out["evidence"]["missing"]["blocks"] == ["facing_closed_by_hold.move_turn"]
    assert (out["chosen"], out["pooled_confidence"]) == ("REJECT", "disputed"), out["why"]
    assert "disagree in sign" in out["why"] and "not clear" in out["why"]

    # and with the seeds AGREEING the cap still lands exactly where it did: uncertain ADOPT
    agree = screen(closed=0.90, prog=0.30)
    agree["facing_closed_by_hold"]["move_turn"] = None
    same = pooled(rc, rule, agree, copy.deepcopy(agree), screen(), screen(), prior_pick="ADOPT")
    assert (same["chosen"], same["pooled_confidence"]) == ("ADOPT", "uncertain")
    assert "capped at uncertain" in same["why"]


def test_a_disputed_pair_keeps_the_incumbent_even_with_a_candidate_default(rc, rule):
    out = pooled(rc, rule, screen(closed=0.54), screen(closed=0.46), screen(), screen(),
                 cand_spec={"speculative_default": "candidate"})
    assert (out["chosen"], out["pooled_confidence"]) == ("REJECT", "disputed")


def test_a_pooled_all_tie_uses_the_declared_default_then_the_pooled_sign(rc, rule):
    same_sign = (screen(closed=0.53), screen(closed=0.53))
    out = pooled(rc, rule, *same_sign, screen(), screen(), prior_pick="ADOPT")
    assert (out["chosen"], out["pooled_confidence"], out["status"]) == ("ADOPT", "uncertain", "final")
    out2 = pooled(rc, rule, *same_sign, screen(), screen(), prior_pick="ADOPT",
                  cand_spec={"speculative_default": "incumbent"})
    assert (out2["chosen"], out2["self_overrule"]) == ("REJECT", True)


def test_a_pooled_worse_is_a_clear_reject(rc, rule):
    out = pooled(rc, rule, screen(ratio=0.40), screen(ratio=0.40), screen(), screen(),
                 prior_pick="ADOPT")
    assert (out["chosen"], out["pooled_confidence"], out["self_overrule"]) == ("REJECT", "clear", True)


def test_a_floor_trip_on_the_pair_run_is_watch_and_the_pick_follows_the_pooled_outcome(rc, rule):
    """Decision 7: a FLOOR on a seed-2/twin/batched run tags the pair ``disputed`` -- it does not flip it."""
    out = pooled(rc, rule, screen(closed=0.545, prog=0.125), screen(closed=0.545, prog=0.125),
                 screen(), screen(), prior_pick="ADOPT", floor=True)
    assert out["chosen"] == "ADOPT"
    assert out["pooled_confidence"] == "disputed"
    assert "FLOOR tripped" in out["why"]


def test_the_pooled_guard_resolution_combines_both_terms(rc, rule):
    out = pooled(rc, rule, screen(), screen(), screen(), screen(),
                 terms={"metrics": {"completion": 0.2}})
    row = [r for r in out["evidence"]["pooled"]["guards"]["rows"] if r["metric"] == "completion"][0]
    assert row["R"] == pytest.approx(math.sqrt((0.10 / math.sqrt(2)) ** 2 + 0.2 ** 2))


def test_a_pooled_drift_failure_forbids_the_pooled_adopt(rc, rule):
    out = pooled(rc, rule, screen(closed=0.545, prog=0.125), screen(closed=0.545, prog=0.125),
                 screen(), screen(), prior_pick="ADOPT", anchor=screen(completion=0.95))
    assert out["chosen"] == "REJECT"
    assert out["evidence"]["anchored_drift"]["verdict"] == "FAIL"


def test_a_pooled_uncertain_final_is_not_an_anchor(rc):
    """The anchor is the last adoption that was CLEAR at its final level (or user-settled)."""
    assert rc._anchor_qualifies({"status": "final", "confidence": "uncertain",
                                 "pooled_confidence": "uncertain", "chosen": "ADOPT"}) is False
    assert rc._anchor_qualifies({"status": "final", "confidence": "uncertain",
                                 "pooled_confidence": "clear", "chosen": "ADOPT"}) is True
    assert rc._anchor_qualifies({"status": "decided", "confidence": "clear", "chosen": "ADOPT"}) is True
    assert rc._anchor_qualifies({"status": "confirmed", "confidence": "uncertain",
                                 "chosen": "ADOPT"}) is True
    assert rc._anchor_qualifies({"status": "overruled", "confidence": "uncertain", "chosen": "REJECT",
                                 "override": {"by": "user", "choice": "ADOPT"}}) is True
    assert rc._anchor_qualifies({"status": "overruled", "confidence": "clear", "chosen": "ADOPT",
                                 "override": {"by": "driver", "choice": "REJECT"}}) is False


def test_a_pooled_level_below_clear_disqualifies_an_s3_clear_adoption(rc, cands):
    """F4: once a pair has run, the POOLED level is the final level -- and it decides alone.

    ``milestone_i`` moves every ``decided`` clear ADOPT in a block to ``final`` and stamps the
    batched pair's ``pooled_confidence`` on it, which can be ``uncertain`` (an all-TIE agreement,
    or a MISSING block member) or ``disputed`` (a FLOOR-tripped batched s2). Reading the entry's
    frozen s3 ``confidence`` as well made exactly those adoptions the drift anchor for every later
    candidate.
    """
    for level in ("uncertain", "disputed"):
        e = {"status": "final", "confidence": "clear", "pooled_confidence": level, "chosen": "ADOPT"}
        assert rc._anchor_qualifies(e) is False, level
    # a user answer still outranks the pooled level, and the REASON names the right actor
    frozen = {"status": "confirmed", "confidence": "clear", "pooled_confidence": "disputed",
              "chosen": "ADOPT", "override": {"by": "user", "choice": "ADOPT"}}
    assert rc._anchor_qualifies(frozen) is True
    assert rc._anchor_reason(frozen) == "user-confirmed"
    driver = {"status": "overruled", "chosen": "REJECT", "confidence": "clear",
              "override": {"by": "driver", "choice": "ADOPT"}}
    assert rc._anchor_reason(driver) != "user-overruled to ADOPT"
    # and the recipe's own anchor follows: a pooled-clear ADOPT reads "pooled clear"
    led = ledger(entry("D05", "C2", "ADOPT", "final", pooled_confidence="clear"))
    assert rc.compose_recipe(led, cands, CODE)["anchor"]["why"] == "pooled clear"
    led = ledger(entry("D05", "C2", "ADOPT", "final", pooled_confidence="disputed"))
    assert rc.compose_recipe(led, cands, CODE)["anchor"] is None


# ================================================================== helpers: ledgers and states
CODE = "CODE0"
OTHER_CODE = "CODE1"


def entry(did, cid, chosen, status, *, kind="candidate", experiment=None, base=None, recipe=None,
          **kw) -> dict:
    out = {"id": did, "kind": kind, "candidate": cid, "chosen": chosen, "status": status,
           "options": ["ADOPT", "REJECT", "SKIP"], "experiment": experiment,
           "base_recipe_id": base, "recipe_id_if_adopted": recipe, "code_sha": CODE,
           "confidence": "clear", "history": [], "conditional_refs": []}
    out.update(kw)
    return out


def ledger(*rows) -> dict:
    return {"schema": 1, "entries": [copy.deepcopy(r) for r in rows]}


def exp(xid, kind, cid, seed, recipe_id, base_recipe_id, rank, *, status="ok", screen_=None,
        code=CODE, **kw) -> dict:
    out = {"xid": xid, "kind": kind, "candidate": cid, "seed": str(seed), "recipe_id": recipe_id,
           "base_recipe_id": base_recipe_id, "base_rank": rank, "code_sha": code, "status": status,
           "started": f"2026-09-19T00:{int(xid[1:]):02d}:00", "ckpt": f"/sim/{xid}/model_7999.pt",
           "screen": screen_ if screen_ is not None else screen(), "recs": [], "eval_dirs": [],
           "label": f"{cid} s{seed}", "chain": None, "inflight": None}
    out.update(kw)
    return out


def state(experiments=None, *, user_choice=None, squashed=None, keys=None) -> dict:
    return {"campaign": "t", "plant": "A15+B", "phase": "run", "run_no": 0, "keys": keys or {},
            "current": {},
            "ladder": {"experiments": experiments or {}, "squashed": squashed or {},
                       "user_choice": user_choice or {}, "pending": [], "resolution_ids": [],
                       "applied_overrides": [], "stage_trace": {}, "d03": None,
                       "code_sha": CODE, "candidates_sha256": None, "unit": None}}


def rid(rc, cands, led, upto=None, *, inclusive=False, code=CODE) -> str:
    return rc.compose_recipe(led, cands, code, upto=upto, inclusive=inclusive)["recipe_id"]


def first_cid(cands) -> str:
    return cands["candidates"][0]["id"]


def base_rank(cid, *, inclusive=False) -> dict:
    return {"upto": cid, "inclusive": inclusive}


def with_baseline(rc, cands, led, st, *, seeds=("3", "2"), code=CODE, screen_=None) -> dict:
    """Give the baseline recipe its ok s3 (X01) and s2 (X02) screens, plus the D02/D04 entries."""
    b0 = rc.compose_recipe(led, cands, code, upto=first_cid(cands))
    xids, dids = {"3": "X01", "2": "X02"}, {"3": "D02", "2": "D04"}
    for seed in (str(s) for s in seeds):
        xid = xids[seed]
        st["ladder"]["experiments"][xid] = exp(xid, "baseline", "B0", seed, b0["recipe_id"],
                                              b0["recipe_id"], {"upto": None, "inclusive": False},
                                              code=code, screen_=screen_, chain=b0["chain"])
        led["entries"].append(entry(dids[seed], "B0", "RECORD", "final", kind="baseline",
                                    experiment=xid, base=b0["recipe_id"], recipe=b0["recipe_id"]))
    return b0


# ================================================================== compose_recipe / stage_env
def test_the_baseline_recipe_is_the_bare_incumbent_chain(rc, cands):
    r = rc.compose_recipe(ledger(), cands, CODE)
    assert r["chain"] == "scratch_1c"
    assert r["adopted"] == []
    assert r["overlay_by_stage"] == {"1c": {}, "2a": {}, "2b": {}}
    assert r["anchor"] is None
    assert r["has_pool"] is False


def test_an_adopted_candidate_switches_the_chain_and_the_identity(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "decided"))
    r = rc.compose_recipe(led, cands, CODE)
    assert r["chain"] == "curriculum"
    assert r["adopted"] == ["C1"]
    assert [str(s["stage"]) for s in r["stages"]] == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert r["recipe_id"] != rc.compose_recipe(ledger(), cands, CODE)["recipe_id"]


def test_the_recipe_hash_is_stable_and_ignores_the_code_sha(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "decided"), entry("D06", "C2", "ADOPT", "decided"))
    a = rc.compose_recipe(led, cands, CODE)["recipe_id"]
    b = rc.compose_recipe(copy.deepcopy(led), cands, OTHER_CODE)["recipe_id"]
    assert a == b == rc.compose_recipe(led, cands, CODE)["recipe_id"]


def test_overruled_and_squashed_choices_are_excluded_and_an_override_wins(rc, cands):
    adopt = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "decided")), cands, CODE)
    assert adopt["adopted"] == ["C1"]
    overruled = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "overruled",
                                               override={"by": "user", "choice": "REJECT"})), cands, CODE)
    assert overruled["adopted"] == []
    squashed = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "squashed")), cands, CODE)
    assert squashed["adopted"] == []
    revived = rc.compose_recipe(ledger(entry("D05", "C1", "REJECT", "overruled",
                                             override={"by": "user", "choice": "ADOPT"})), cands, CODE)
    assert revived["adopted"] == ["C1"]


def test_only_the_newest_non_squashed_entry_per_candidate_counts(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "squashed"), entry("D09", "C1", "REJECT", "decided"))
    assert rc.compose_recipe(led, cands, CODE)["adopted"] == []
    led2 = ledger(entry("D05", "C1", "REJECT", "decided"), entry("D09", "C1", "ADOPT", "decided"))
    assert rc.compose_recipe(led2, cands, CODE)["adopted"] == ["C1"]


def test_upto_bounds_the_composition_by_rank(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "decided"), entry("D08", "C2", "ADOPT", "decided"))
    assert rc.compose_recipe(led, cands, CODE, upto="C2")["adopted"] == ["C1"]
    assert rc.compose_recipe(led, cands, CODE, upto="C2", inclusive=True)["adopted"] == ["C1", "C2"]
    assert rc.compose_recipe(led, cands, CODE, upto="C1")["adopted"] == []


def test_the_anchor_is_the_last_clear_adoption(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "decided", confidence="clear", experiment="X03"),
                 entry("D08", "C2", "ADOPT", "speculative", confidence="uncertain", experiment="X04"))
    r = rc.compose_recipe(led, cands, CODE)
    assert r["anchor"]["candidate"] == "C1"
    assert r["anchor"]["experiment"] == "X03"
    assert "clear" in r["anchor"]["why"]


def test_eval_carry_is_the_union_over_adopted_candidates(rc, cands):
    led = ledger(entry("D05", "C3a", "ADOPT", "decided"), entry("D06", "C5", "ADOPT", "decided"))
    carry = rc.compose_recipe(led, cands, CODE)["eval_carry"]
    assert set(carry) == {"KRABBY_OMNI_ANG_VEL_MAX", "KRABBY_OMNI_CLOCK_TURN_R",
                          "KRABBY_OMNI_FACING_OBS_ENCODING"}
    assert all(k in rc.rp.EVAL_CARRY for k in carry)


def test_c2_and_c5_reach_the_teacher_windows(rc, cands):
    """The plan's own pinned example: ``compose_recipe(B0+C2+C5).overlay_by_stage["2a"]``."""
    led = ledger(entry("D05", "C2", "ADOPT", "decided"), entry("D06", "C5", "ADOPT", "decided"))
    env = rc.compose_recipe(led, cands, CODE)["overlay_by_stage"]["2a"]
    assert env["KRABBY_YAW_PROGRESS_W"] == "3.0"
    assert env["KRABBY_OMNI_FACING_OBS_ENCODING"] == "sincos"


@pytest.mark.parametrize("cid", ["C1", "C1c"])
def test_a_chain_candidates_preset_emulation_keys_never_reach_the_teacher_windows(rc, cands, cid):
    """C1/C1c declare ``teacher_overlay: {}``: only the RESCALED pool crosses into 2a/2b."""
    r = rc.compose_recipe(ledger(entry("D05", cid, "ADOPT", "decided")), cands, CODE)
    for stage in ("2a", "2b"):
        env = r["overlay_by_stage"][stage]
        for key in ("KRABBY_OMNI_FACING", "KRABBY_OMNI_RSI_SOURCE", "KRABBY_OMNI_DIR_BAND",
                    "KRABBY_YAW_W"):
            assert key not in env, (stage, key)
        pool = rc.parse_pool(env["KRABBY_OMNI_CMD_POOL"])
        assert rc.pool_slots(pool) == pytest.approx(rc.parse_slots_str(rc.ph.SLOTS_TEACHER))
        assert rc.parse_slots_str(env["KRABBY_OMNI_SLOTS"]) == pytest.approx(
            rc.parse_slots_str(rc.ph.SLOTS_TEACHER))


def test_every_chain_stage_declares_the_projection_its_pool_produces(rc, cands):
    """The cfg refuses a pool whose SLOTS disagree with its projection, so the recipe recomputes it."""
    r = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "decided")), cands, CODE)
    for stage, env in r["overlay_by_stage"].items():
        if "KRABBY_OMNI_CMD_POOL" not in env:
            continue
        pool = rc.parse_pool(env["KRABBY_OMNI_CMD_POOL"])
        declared = rc.parse_slots_str(env["KRABBY_OMNI_SLOTS"])
        total = sum(declared)
        assert [d / total for d in declared] == pytest.approx(list(rc.pool_slots(pool)), abs=1e-6), stage


def test_c8_composes_per_chain_and_survives_the_teacher_rescale(rc, cands):
    """C8's pinned form: ``0.05:0.65:0.10:0.20`` on the pooled chain, ``0.05:0.55:0.15:0.25`` on 2a/2b."""
    curriculum = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "decided"),
                                          entry("D06", "C8", "ADOPT", "decided")), cands, CODE)
    c6 = curriculum["overlay_by_stage"]["c6"]
    assert rc.pool_slots(rc.parse_pool(c6["KRABBY_OMNI_CMD_POOL"])) == pytest.approx((0.05, 0.65, 0.10, 0.20))
    assert rc.parse_slots_str(c6["KRABBY_OMNI_SLOTS"]) == pytest.approx((0.05, 0.65, 0.10, 0.20))
    for stage in ("2a", "2b"):
        env = curriculum["overlay_by_stage"][stage]
        assert rc.pool_slots(rc.parse_pool(env["KRABBY_OMNI_CMD_POOL"])) == pytest.approx(
            (0.05, 0.55, 0.15, 0.25))
        assert rc.parse_slots_str(env["KRABBY_OMNI_SLOTS"]) == pytest.approx((0.05, 0.55, 0.15, 0.25))
    incumbent = rc.compose_recipe(ledger(entry("D05", "C8", "ADOPT", "decided")), cands, CODE)
    assert incumbent["overlay_by_stage"]["1c"]["KRABBY_OMNI_SLOTS"] == "0.05:0.65:0.1:0.2"
    assert "KRABBY_OMNI_CMD_POOL" not in incumbent["overlay_by_stage"]["1c"]
    assert rc.parse_slots_str(incumbent["overlay_by_stage"]["2b"]["KRABBY_OMNI_SLOTS"]) == \
        pytest.approx((0.05, 0.55, 0.15, 0.25))


def test_renormalise_leaves_the_turn_and_move_turn_shares_alone(rc, cands):
    pool = rc.parse_pool(cands["chains"]["curriculum"]["stages"][-1]["overlay"]["KRABBY_OMNI_CMD_POOL"])
    edited = rc.apply_pool_edit(pool, {"set_share": {"stand": 0.05}, "renormalise": True})
    before, after = rc.pool_slots(pool), rc.pool_slots(edited)
    assert before[2:] == pytest.approx(after[2:])
    assert after[0] == pytest.approx(0.05)
    assert after[1] == pytest.approx(before[1] + 0.10)
    assert sum(edited.weights) == pytest.approx(1.0)


def test_a_pool_edit_scale_composes_with_a_share_edit_in_ledger_order(rc, cands):
    led = ledger(entry("D05", "C1", "ADOPT", "decided"), entry("D06", "C7", "ADOPT", "decided"),
                 entry("D07", "C8", "ADOPT", "decided"))
    r = rc.compose_recipe(led, cands, CODE)
    pool = rc.parse_pool(r["overlay_by_stage"]["c6"]["KRABBY_OMNI_CMD_POOL"])
    scales = dict(zip(pool.names, pool.ang_scale))
    assert scales["omni_turn"] == 0.3 and scales["fwd_turn"] == 0.3
    assert scales["turn"] == 1.0, "the in-place turn type is NOT matched by *_turn"
    assert rc.pool_slots(pool) == pytest.approx((0.05, 0.65, 0.10, 0.20))


def test_the_reduced_demand_edit_is_inert_on_a_stage_whose_turn_types_do_not_exist_yet(rc, cands):
    r = rc.compose_recipe(ledger(entry("D05", "C1b", "ADOPT", "decided")), cands, CODE)
    c1 = rc.parse_pool(r["overlay_by_stage"]["c1"]["KRABBY_OMNI_CMD_POOL"])
    assert set(c1.ang_scale) == {1.0}
    c4 = rc.parse_pool(r["overlay_by_stage"]["c4"]["KRABBY_OMNI_CMD_POOL"])
    assert dict(zip(c4.names, c4.ang_scale))["side_l_turn"] == 0.3


def test_a_cap_candidate_clamps_the_teacher_facing_ramp(rc, cands):
    """The one driver-written FACING key: ``min(preset, 8 s x ang_vel_max)`` on 2a and 2b."""
    r = rc.compose_recipe(ledger(entry("D05", "C3a", "ADOPT", "decided")), cands, CODE)
    assert "KRABBY_OMNI_FACING" not in r["overlay_by_stage"]["2a"]     # preset 1.2 < 8 x 0.21
    assert r["overlay_by_stage"]["2b"]["KRABBY_OMNI_FACING"] == "0.2:1.68"
    r2 = rc.compose_recipe(ledger(entry("D05", "C3b", "ADOPT", "decided")), cands, CODE)
    assert r2["overlay_by_stage"]["2b"]["KRABBY_OMNI_FACING"] == "0.2:1.44"


def test_two_candidates_that_write_one_key_differently_are_reported_as_a_conflict(rc, cands):
    led = ledger(entry("D05", "C3a", "ADOPT", "decided"), entry("D06", "C3b", "ADOPT", "decided"))
    r = rc.compose_recipe(led, cands, CODE)
    assert any("KRABBY_OMNI_ANG_VEL_MAX" in c for c in r["conflicts"])
    assert r["overlay_by_stage"]["1c"]["KRABBY_OMNI_ANG_VEL_MAX"] == "0.18"   # ledger order wins


def test_an_equal_value_written_twice_is_not_a_conflict(rc, cands):
    conflicts = []
    rc._merge_overlay({"KRABBY_OMNI_SLOTS": "0.05:0.55:0.15:0.25"},
                      {"KRABBY_OMNI_SLOTS": "0.0500:0.55:0.15:0.25"}, "2a", "x", conflicts)
    assert conflicts == []
    rc._merge_overlay({"KRABBY_OMNI_SLOTS": "0.05:0.55:0.15:0.25"},
                      {"KRABBY_OMNI_SLOTS": "0.06:0.55:0.15:0.25"}, "2a", "x", conflicts)
    assert len(conflicts) == 1


def test_stage_env_is_strings_and_never_names_a_preset_plant_or_other_task(rc, cands):
    for cid in [c["id"] for c in cands["candidates"]]:
        r = rc._recipe_with(ledger(), cands, CODE, cid, set())
        for stage in [str(s["stage"]) for s in r["stages"]] + ["2a", "2b"]:
            env = rc.stage_env(r, stage)
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
            for banned in rc.FORBIDDEN_OVERLAY_KEYS:
                assert banned not in env, (cid, stage, banned)
            assert not any(k.startswith("KRABBY_SIDE") for k in env)


def test_stage_env_refuses_a_forbidden_extra(rc, cands):
    r = rc.compose_recipe(ledger(), cands, CODE)
    with pytest.raises(ValueError, match="KRABBY_PLANT"):
        rc.stage_env(r, "1c", {"KRABBY_PLANT": "A15+B"})


def test_the_curriculum_totals_the_screen_budget_and_ramps_move_turn(rc, cands):
    stages = cands["chains"]["curriculum"]["stages"]
    assert sum(int(s["iters"]) for s in stages) == int(cands["screen"]["iters"]) == 8000
    shares = [rc.pool_slots(rc.parse_pool(s["overlay"]["KRABBY_OMNI_CMD_POOL"]))[3] for s in stages]
    assert shares == pytest.approx([0.0, 0.0, 0.0, 0.10, 0.15, 0.20])
    assert max(shares) <= 0.20 + 1e-9
    floor = rc.pool_math().POOL_FLOOR_SHARE
    for st in stages:
        pool = rc.parse_pool(st["overlay"]["KRABBY_OMNI_CMD_POOL"])
        assert min(pool.weights) >= floor
        assert sum(pool.weights) == pytest.approx(1.0)
    assert "back" in rc.parse_pool(stages[1]["overlay"]["KRABBY_OMNI_CMD_POOL"]).names


def test_the_teacher_pool_round_trips_through_render_and_parse(rc, cands):
    c6 = cands["chains"]["curriculum"]["stages"][-1]["overlay"]["KRABBY_OMNI_CMD_POOL"]
    teacher = rc.parse_pool(c6).rescale_to(rc.parse_slots_str(rc.ph.SLOTS_TEACHER))
    text = rc.render_pool(teacher)
    assert rc.render_pool(rc.parse_pool(text)) == text
    assert rc.pool_slots(rc.parse_pool(text)) == pytest.approx(rc.parse_slots_str(rc.ph.SLOTS_TEACHER))


def test_slots_are_rendered_losslessly_enough_for_the_cfg_rule(rc):
    probs = (0.05, 0.6499999999999999, 0.1, 0.2)
    got = rc.parse_slots_str(rc.slots_str(probs))
    assert got == pytest.approx(probs, abs=1e-9)


# ---------------------------------------------------------------- canonical pool rendering (am. 27)
def _sig_digits(token: str) -> int:
    """Significant digits in one rendered ``name:weight[:ang[:speed]]`` field."""
    return max(len(part.replace("-", "").replace(".", "").lstrip("0") or "0")
               for part in token.split(":")[1:])


def _shipped_recipes(rc, cands):
    """(label, recipe) for the baseline and for every candidate's own adopted recipe."""
    led = ledger()
    out = [("B0", rc.compose_recipe(led, cands, CODE, upto=first_cid(cands)))]
    for cid in rc.candidate_ids(cands):
        out.append((cid, rc._recipe_with(led, cands, CODE, cid, set())))
    # the two chain-switching adoptions, so the pooled chains are covered on their own recipes too
    for cid in ("C1", "C1c"):
        led2 = ledger(entry("D05", cid, "ADOPT", "decided"))
        for other in ("C7", "C8"):
            out.append((f"{cid}+{other}", rc._recipe_with(led2, cands, CODE, other, set())))
    return out


def test_every_composed_pool_string_is_canonical_and_a_render_parse_fixpoint(rc, cands):
    """Amendment 27: a composed pool renders at 12 significant digits, never the ULP tail.

    ``recipe_id`` is sha256 over exactly these strings, so ``0.14`` vs ``0.14000000000000004`` is
    an IDENTITY change: ``base_recipe_id`` / ``recipe_id_if_adopted`` / ``ok_screen`` matching and
    the squash view's "the base hash matches again" resurrection all key on it, and a screen taken
    under the noisy spelling would be read as a different recipe (a 3.7-3.9 GPU-h re-screen, no
    resurrection). Checked on EVERY shipped recipe, not just the one that moved.
    """
    checked = 0
    for label, recipe in _shipped_recipes(rc, cands):
        for stage, env in recipe["overlay_by_stage"].items():
            text = env.get("KRABBY_OMNI_CMD_POOL")
            if not text:
                continue
            checked += 1
            assert rc.render_pool(rc.parse_pool(text)) == text, (label, stage, text)
            worst = max(_sig_digits(tok) for tok in text.split(","))
            assert worst <= rc.POOL_SIG_DIGITS, (label, stage, text)
            assert env["KRABBY_OMNI_SLOTS"] == rc.slots_str(rc.pool_slots(rc.parse_pool(text))), \
                (label, stage)
    assert checked >= 20, checked


def test_an_unedited_pool_rescaled_onto_the_teacher_slots_reproduces_the_yaml_literal(rc, cands):
    """The plan's own spelling: ``fwd:0.14``, ``side_l_turn:0.035`` -- not ``0.14000000000000004``.

    C8-on-``scratch_c6`` is the case that moved: its 2a pool is the c6 literal with the share edit
    re-applied after the rescale, which lands back on the yaml's own weights for the move types.
    """
    c6_raw = cands["chains"]["scratch_c6"]["stages"][-1]["overlay"]["KRABBY_OMNI_CMD_POOL"]
    assert "fwd:0.14" in c6_raw and "side_l_turn:0.035" in c6_raw

    led = ledger(entry("D05", "C1c", "ADOPT", "decided"))
    recipe = rc._recipe_with(led, cands, CODE, "C8", set())
    assert recipe["chain"] == "scratch_c6"
    # the chain stage: the yaml literal with C8's share edit, still 12 significant digits
    c6 = recipe["overlay_by_stage"]["c6"]["KRABBY_OMNI_CMD_POOL"]
    assert "side_l_turn:0.035" in c6 and "fwd_turn:0.06" in c6
    assert max(_sig_digits(tok) for tok in c6.split(",")) <= rc.POOL_SIG_DIGITS, c6
    for stage in ("2a", "2b"):
        env = recipe["overlay_by_stage"][stage]
        text = env["KRABBY_OMNI_CMD_POOL"]
        assert "fwd:0.14," in text + ",", text
        assert "0.14000000000000004" not in text
        assert max(_sig_digits(tok) for tok in text.split(",")) <= rc.POOL_SIG_DIGITS, text
        assert rc.pool_slots(rc.parse_pool(text)) == pytest.approx((0.05, 0.55, 0.15, 0.25))
        assert env["KRABBY_OMNI_SLOTS"] == "0.05:0.55:0.15:0.25"
    # and the validator's preview renders the same string the recipe does
    spec = rc._cand_spec(rc.candidate_by_id(cands, "C8"), "scratch_c6")
    preview = rc._teacher_preview(cands, spec, "scratch_c6")
    assert preview["KRABBY_OMNI_CMD_POOL"] == recipe["overlay_by_stage"]["2a"]["KRABBY_OMNI_CMD_POOL"]


def test_the_recipe_id_of_every_shipped_recipe_survives_compose_parse_compose(rc, cands):
    """compose -> parse every pool it wrote -> re-render -> re-hash: the SAME id, under NOISE.

    Two halves of the one property ``ok_screen`` / ``base_recipe_id`` / the squash view's "the base
    hash matches again" all need. First the round trip (parse the recorded strings, render and
    re-project them, re-hash). Then the part that actually moved: the same recipe composed from a
    yaml whose every pool weight is nudged by one ULP must hash IDENTICALLY -- without canonical
    rendering the tail reaches the rendered string and the recipe becomes a different recipe.
    """
    for label, recipe in _shipped_recipes(rc, cands):
        rebuilt = {}
        for stage, env in recipe["overlay_by_stage"].items():
            env = dict(env)
            text = env.get("KRABBY_OMNI_CMD_POOL")
            if text:
                pool = rc.parse_pool(text)
                env["KRABBY_OMNI_CMD_POOL"] = rc.render_pool(pool)
                env["KRABBY_OMNI_SLOTS"] = rc.slots_str(rc.pool_slots(pool))
            rebuilt[stage] = env
        again = rc.sha256_text(rc.canonical({"chain": recipe["chain"],
                                             "overlay_by_stage": rebuilt}))[:12]
        assert again == recipe["recipe_id"], label

    nudged = _nudged_cands(rc, cands)
    assert nudged != cands, "the nudge must actually change the yaml text"
    for (label, recipe), (label2, noisy) in zip(_shipped_recipes(rc, cands),
                                                _shipped_recipes(rc, nudged)):
        assert label == label2
        assert noisy["recipe_id"] == recipe["recipe_id"], (label, recipe["recipe_id"],
                                                           noisy["recipe_id"])


def _nudged_cands(rc, cands):
    """``cands`` with every chain-stage pool weight moved one ULP -- far inside 12 digits."""
    out = copy.deepcopy(cands)
    for chain in (out.get("chains") or {}).values():
        for stage in chain.get("stages") or []:
            text = (stage.get("overlay") or {}).get("KRABBY_OMNI_CMD_POOL")
            if not text:
                continue
            fields = []
            for token in text.split(","):
                parts = token.split(":")
                parts[1] = repr(math.nextafter(float(parts[1]), 1.0))
                fields.append(":".join(parts))
            stage["overlay"]["KRABBY_OMNI_CMD_POOL"] = ",".join(fields)
    return out


def test_canonicalising_a_pool_is_idempotent_and_never_moves_a_weight_materially(rc, cands):
    """12 significant digits: two orders below the tightest tolerance the pool is read with."""
    c6 = cands["chains"]["curriculum"]["stages"][-1]["overlay"]["KRABBY_OMNI_CMD_POOL"]
    noisy = rc.parse_pool(c6).rescale_to(rc.parse_slots_str(rc.ph.SLOTS_TEACHER))
    once = rc._canonical_pool(noisy)
    assert rc._canonical_pool(once) == once
    assert once.names == noisy.names
    for a, b in zip(once.weights, noisy.weights):
        assert a == pytest.approx(b, rel=1e-11, abs=1e-15)
    assert rc.pool_slots(once) == pytest.approx(rc.parse_slots_str(rc.ph.SLOTS_TEACHER), abs=1e-9)
    assert sum(once.weights) == pytest.approx(sum(noisy.weights), abs=1e-9)


# ================================================================== the derived queue
def queue_label(rc, cands, led, st, *, code=CODE):
    item = rc.next_item(cands, led, st, code=code)
    return None if item is None else item["label"]


def queue_item(rc, cands, led, st, *, code=CODE):
    return rc.next_item(cands, led, st, code=code)


def test_the_queue_starts_with_the_baselines_two_seeds(rc, cands):
    led, st = ledger(), state()
    assert queue_label(rc, cands, led, st) == "B0 s3"
    b0 = with_baseline(rc, cands, led, st, seeds=("3",))
    item = queue_item(rc, cands, led, st)
    assert item["label"] == "B0 s2" and item["kind"] == "baseline" and item["seed"] == "2"
    assert item["recipe"]["recipe_id"] == b0["recipe_id"]
    with_baseline(rc, cands, led, st, seeds=("2",))               # add the s2 screen too
    assert queue_label(rc, cands, led, st) == "C1 s3"


def test_a_speculative_decision_queues_the_candidates_own_seed_two_and_no_twin(rc, cands):
    """B0's s2 IS the twin of everything screened on B0, so the pair costs ONE window."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", c1["recipe_id"],
                                            rid(rc, cands, led, "C1"), base_rank("C1"))
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                confidence="uncertain", base=rid(rc, cands, led, "C1"),
                                recipe=c1["recipe_id"]))
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "seed2" and item["candidate"] == "C1" and item["seed"] == "2"
    assert item["decision"] == "D05"
    assert item["recipe"]["recipe_id"] == c1["recipe_id"]


def test_an_uncertain_candidate_on_an_adopted_recipe_pays_one_twin_first(rc, cands):
    """``C4 clear ADOPT -> C5 uncertain -> (R_1+C4) twin -> C5 s2``: the base needs its own s2."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c4 = rid(rc, cands, led, "C4")
    r_c4 = rc._recipe_with(led, cands, CODE, "C4", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C4", "3", r_c4["recipe_id"],
                                            base_c4, base_rank("C4"))
    led["entries"].append(entry("D05", "C4", "ADOPT", "decided", experiment="X03", base=base_c4,
                                recipe=r_c4["recipe_id"]))
    base_c5 = rid(rc, cands, led, "C5")
    assert base_c5 == r_c4["recipe_id"]
    r_c5 = rc._recipe_with(led, cands, CODE, "C5", set())
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C5", "3", r_c5["recipe_id"],
                                            base_c5, base_rank("C5"))
    led["entries"].append(entry("D06", "C5", "ADOPT", "speculative", experiment="X04",
                                confidence="uncertain", base=base_c5, recipe=r_c5["recipe_id"]))
    twin = queue_item(rc, cands, led, st)
    assert twin["kind"] == "twin" and twin["candidate"] == "C5" and twin["seed"] == "2"
    assert twin["recipe"]["recipe_id"] == base_c5
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C5", "2", base_c5, base_c5,
                                            base_rank("C5"))
    nxt = queue_item(rc, cands, led, st)
    assert nxt["kind"] == "seed2" and nxt["candidate"] == "C5"


def test_an_existing_seed_two_screen_makes_the_pooled_decision_zero_gpu(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", c1["recipe_id"],
                                            rid(rc, cands, led, "C1"), base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", c1["recipe_id"],
                                            rid(rc, cands, led, "C1"), base_rank("C1"))
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                confidence="uncertain", base=rid(rc, cands, led, "C1"),
                                recipe=c1["recipe_id"]))
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "pooled" and item["decision"] == "D05"


def test_a_clear_reject_moves_straight_on_and_a_false_predicate_writes_a_skip(rc, cands):
    """``C3a clear REJECT -> C3b SKIP -> M1 eligible`` (C3b's predicate needs a pair or a non-reject)."""
    rows = rc.simulate_queue(cands, outcome="REJECT", limit=12, code=CODE)
    labels = [r["label"] for r in rows]
    assert labels[:5] == ["B0 s3", "B0 s2", "C1 s3", "C1b s3", "C1c SKIP"]
    assert "C3b SKIP" in labels
    assert labels[labels.index("C3b SKIP") + 1] == "M1.(i)"


def test_the_all_reject_path_reaches_m1_with_no_batched_seed_two(rc, cands):
    """``R_1 = B0`` -> no ``M1.s2`` (B0's s2 exists), (i) then ``M1.promo`` of B0."""
    rows = [r["label"] for r in rc.simulate_queue(cands, outcome="REJECT", limit=40, code=CODE)]
    assert "M1.s2 s2" not in rows
    assert rows.index("M1.(i)") < rows.index("M1.promo s3")
    assert "M2.carry" in rows, "M2's recipe equals M1's, so both experiments are omitted"
    assert rows[-1] == "await_user:C12"


def test_the_all_adopt_path_batches_a_seed_two_before_every_milestone(rc, cands):
    rows = [r["label"] for r in rc.simulate_queue(cands, outcome="ADOPT", limit=40, code=CODE)]
    for mid in ("M1", "M2"):
        assert rows.index(f"{mid}.s2 s2") < rows.index(f"{mid}.(i)") < rows.index(f"{mid}.promo s3")
    assert "C1b SKIP" in rows and "C1c s3" in rows


def test_a_milestone_waits_for_every_earlier_candidate_to_be_terminal(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2", "C3a"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    # a clear s3 REJECT of C3a leaves C3b's predicate false, so the queue writes its SKIP first
    assert queue_label(rc, cands, led, st) == "C3b SKIP"
    holds, why = rc.after_holds(cands, led, "M1")
    assert holds is False and "C3b" in why
    led["entries"].append(entry("D_C3b", "C3b", "SKIP", "skipped"))
    assert rc.after_holds(cands, led, "M1")[0] is True
    assert queue_label(rc, cands, led, st) == "M1.(i)"


def test_an_overruled_candidate_is_terminal_for_the_milestone(rc, cands):
    """``C2 overruled(user) -> M1 eligible once C3b is TERMINAL``."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "overruled",
                                override={"by": "user", "choice": "REJECT"}, frozen=True))
    assert rc.after_holds(cands, led, "M1")[0] is True
    assert queue_label(rc, cands, led, st) == "M1.(i)"


def test_a_sticky_milestone_await_parks_again_on_every_relaunch(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "recipe_id": rid(rc, cands, led), "evidence": {"stage": "i"}})
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "await" and item["label"] == "await_user:M1"
    item2 = queue_item(rc, cands, led, st)
    assert item2["label"] == "await_user:M1"


def test_an_accepted_i_milestone_queues_the_promotion(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "sub_status": "accepted_i", "history": [],
                           "options": ["PASS", "WATCH", "FAIL"], "recipe_id": rid(rc, cands, led),
                           "evidence": {"stage": "i"}, "eligibility": 1})
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "milestone" and item["label"] == "M1.promo s3"
    assert [str(s["stage"]) for s in item["stages"]] == ["2a", "2b"]
    assert str(item["stages"][0]["resume_from"]) == "recipe_head"
    assert str(item["stages"][1]["resume_from"]) == "previous_stage"
    assert list(item["stages"][0]["eval"]) == list(rc.ph.EVAL_FULL)


def test_m3_final_seed_two_follows_a_closed_m3_and_then_the_campaign_is_done(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cand in cands["candidates"]:
        led["entries"].append(entry(f"D_{cand['id']}", cand["id"], "REJECT", "decided"))
    recipe_id = rid(rc, cands, led)
    for mid in ("M1", "M2", "M3"):
        led["entries"].append({"id": mid, "kind": "milestone", "candidate": mid, "chosen": "PASS",
                               "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                               "recipe_id": recipe_id, "promoted_head": {"2b": {"ckpt": "/x.pt"}}})
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "milestone_s2" and item["label"] == "M3.final_s2 s2"
    assert str(item["stages"][0]["resume_from"]) == "recipe_head_s2"
    st["ladder"]["experiments"]["X09"] = exp("X09", "milestone_s2", "M3", "2", recipe_id, recipe_id,
                                            {"upto": "C12", "inclusive": True})
    assert queue_item(rc, cands, led, st) is None


def test_a_deferred_candidate_parks_for_code_and_a_user_decision_parks_for_an_answer(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cand in cands["candidates"]:
        if cand["id"] in ("C10", "C12"):
            continue
        led["entries"].append(entry(f"D_{cand['id']}", cand["id"], "REJECT", "decided"))
    for mid in ("M1", "M2"):
        led["entries"].append({"id": mid, "kind": "milestone", "candidate": mid, "chosen": "PASS",
                               "status": "final", "options": ["PASS", "WATCH", "FAIL"],
                               "history": [], "recipe_id": rid(rc, cands, led)})
    # C10's predicate HOLDS on the recorded screen (yaw_progress_turn_min 0.080 < 0.10), so the
    # deferred candidate is reached and stops for its code
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "await" and item["label"] == "await_user:C10 code"
    assert "KRABBY_OMNI_RSI_TURN_SOURCE" in item["why"]
    # with a screen above the floor the predicate is false and it is skipped instead
    st["ladder"]["experiments"]["X01"]["screen"]["yaw_progress_turn_min"] = 0.20
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "skip" and item["candidate"] == "C10"
    led["entries"].append(entry("D_C10", "C10", "SKIP", "skipped"))
    led["entries"].append(entry("D_C11", "C11", "SKIP", "skipped"))
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "await" and item["label"] == "await_user:C12"
    # the user says run: the deferred knob is still missing, so it parks for the CODE
    st["ladder"]["user_choice"]["C12"] = {"choice": "run", "at": "2026-09-19T10:00:00"}
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "await" and item["label"] == "await_user:C12 code"
    assert "not exported yet" in item["why"]


def test_the_deferred_dispatch_check_passes_once_the_knob_exists(rc, cands, monkeypatch):
    c12 = [c for c in cands["candidates"] if c["id"] == "C12"][0]
    assert rc.dispatch_check(c12)[0] is False
    monkeypatch.setattr(rc, "knob_whitelist", lambda: {"KRABBY_OMNI_TRAVEL_FRAME"})
    assert rc.dispatch_check(c12)[0] is True


def test_a_user_skip_beats_a_true_predicate_and_a_user_run_beats_a_false_one(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    st["ladder"]["user_choice"]["C1"] = {"choice": "skip", "at": "2026-09-19T10:00:00"}
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "skip" and item["candidate"] == "C1"
    led["entries"].append(entry("D05", "C1", "SKIP", "skipped", evidence={"user": True}))
    for cid in ("C1b", "C1c"):
        st["ladder"]["user_choice"][cid] = {"choice": "run", "at": "2026-09-19T10:00:00"}
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "candidate" and item["candidate"] == "C1b"


def test_a_missing_base_screen_inserts_a_recipe_baseline_rerun(rc, cands):
    """After an overrule (or a code landing) the next item's control is missing -> ``B<k>`` first."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append(entry("D_C1c", "C1c", "SKIP", "skipped"))
    # C2 is adopted but its own screen is not recorded (an overrule squashed it, or the code moved),
    # so C3a's control does not exist and the queue inserts the recipe-baseline rerun first
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X99",
                                base=rid(rc, cands, led, "C2"),
                                recipe=rc._recipe_with(led, cands, CODE, "C2", set())["recipe_id"]))
    item = queue_item(rc, cands, led, st)
    assert item["kind"] == "baseline" and item["label"] == "B1 s3"
    assert item["recipe"]["adopted"] == ["C2"]
    assert "needs an ok s3 screen" in item["why"]


def test_a_code_change_invalidates_every_screen_and_queues_a_rerun(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    assert queue_label(rc, cands, led, st) == "C1 s3"
    item = queue_item(rc, cands, led, st, code=OTHER_CODE)
    assert item["kind"] == "baseline" and item["seed"] == "3"
    assert item["label"].startswith("B")


def test_the_queue_is_recomputed_not_stored_so_a_crash_changes_nothing(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    before = queue_label(rc, cands, led, st)
    reloaded = json.loads(json.dumps(st))
    assert queue_label(rc, cands, json.loads(json.dumps(led)), reloaded) == before


# ================================================================== the derived squash view
def test_an_overrule_squashes_the_screens_of_later_candidates_only(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c1 = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03", base=base_c1,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            base_c1, base_rank("C1"))
    base_c2 = rid(rc, cands, led, "C2")
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D06", "C2", "REJECT", "decided", experiment="X04", base=base_c2,
                                recipe=r_c2["recipe_id"]))
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C2", "3", r_c2["recipe_id"],
                                            base_c2, base_rank("C2"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert view["experiments"] == {} and view["entries"] == {}
    # now overrule C1 -> the recipe C2 was screened on no longer exists
    led["entries"][-2]["status"] = "overruled"
    led["entries"][-2]["override"] = {"by": "user", "choice": "REJECT", "why": "overrule"}
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X04" in view["experiments"] and "base recipe changed" in view["experiments"]["X04"]
    assert "X03" not in view["experiments"], "D_k's own screen is never squashed"
    assert "X01" not in view["experiments"] and "X02" not in view["experiments"]
    assert view["entries"]["D06"].startswith("its experiment X04")
    assert "C2" in view["candidates"]


def test_a_finished_seed_two_of_a_pair_resolved_decision_survives_a_later_user_action(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    base = rid(rc, cands, led, "C1")
    led["entries"].append(entry("D05", "C1", "ADOPT", "final", experiment="X03",
                                seed2_experiment="X04", base=base, recipe=r_c1["recipe_id"],
                                pooled_confidence="clear",
                                history=[{"from": "speculative", "to": "final", "by": "driver"}]))
    for xid, kind, seed in (("X03", "candidate", "3"), ("X04", "seed2", "2")):
        st["ladder"]["experiments"][xid] = exp(xid, kind, "C1", seed, r_c1["recipe_id"], base,
                                               base_rank("C1"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X04" not in view["experiments"]
    # a user confirm AFTER the pair resolved does not squash the finished s2 (cause (e) is about
    # a decision settled WHILE it was speculative)
    led["entries"][-1]["history"].append({"from": "final", "to": "confirmed", "by": "user",
                                          "why": "confirm"})
    led["entries"][-1]["status"] = "confirmed"
    assert "X04" not in rc.squash_view(led, cands, st, code=CODE)["experiments"]


def test_a_user_settling_a_speculative_decision_squashes_its_own_pair_run_forever(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    base = rid(rc, cands, led, "C1")
    led["entries"].append(entry("D05", "C1", "ADOPT", "confirmed", experiment="X03",
                                seed2_experiment="X04", base=base, recipe=r_c1["recipe_id"],
                                frozen=True, confidence="uncertain",
                                history=[{"from": "speculative", "to": "confirmed", "by": "user",
                                          "why": "confirm: looks right"}]))
    for xid, kind, seed in (("X03", "candidate", "3"), ("X04", "seed2", "2"), ("X05", "twin", "2")):
        st["ladder"]["experiments"][xid] = exp(xid, kind, "C1", seed,
                                               base if kind == "twin" else r_c1["recipe_id"], base,
                                               base_rank("C1"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X04" in view["experiments"] and "cause e" in view["experiments"]["X04"]
    assert "X05" not in view["experiments"], "the base recipe's twin is never dropped by this"


def test_a_driver_self_overrule_keeps_the_pair_run_it_was_decided_by(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    base = rid(rc, cands, led, "C1")
    led["entries"].append(entry("D05", "C1", "REJECT", "overruled", experiment="X03",
                                seed2_experiment="X04", base=base, recipe=r_c1["recipe_id"],
                                override={"by": "driver", "choice": "ADOPT", "why": "pooled"},
                                history=[{"from": "speculative", "to": "overruled", "by": "driver"}]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                            base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base,
                                            base_rank("C1"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert view["experiments"] == {}


def test_a_skip_is_squashed_when_its_referent_flips_and_transitively_by_its_refs(rc, cands):
    """The C1/C1b/C7 sequence: C1 REJECT -> C1b screened -> C1 overruled to ADOPT."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c1 = rid(rc, cands, led, "C1")
    led["entries"].append(entry("D05", "C1", "REJECT", "decided", experiment="X03", base=base_c1))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", "r_c1", base_c1,
                                            base_rank("C1"))
    r_c1b = rc._recipe_with(led, cands, CODE, "C1b", set())
    base_c1b = rid(rc, cands, led, "C1b")
    led["entries"].append(entry("D06", "C1b", "REJECT", "decided", experiment="X04", base=base_c1b,
                                recipe=r_c1b["recipe_id"]))
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C1b", "3", r_c1b["recipe_id"],
                                            base_c1b, base_rank("C1b"))
    led["entries"].append(entry("D07", "C7", "SKIP", "skipped", base=rid(rc, cands, led, "C7"),
                                conditional_refs=["C1b"],
                                evidence={"predicate": "recipe.has_pool and ...", "user": False}))
    assert rc.squash_view(led, cands, st, code=CODE)["entries"] == {}
    d05 = [e for e in led["entries"] if e["id"] == "D05"][0]        # C1 REJECT -> ADOPT
    d05["status"] = "overruled"
    d05["override"] = {"by": "user", "choice": "ADOPT", "why": "overrule"}
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X04" in view["experiments"] and "base recipe changed" in view["experiments"]["X04"]
    assert "D06" in view["entries"]
    assert "D07" in view["entries"], "C7's SKIP referenced C1b, which is undecided again"
    assert "C1b" in view["candidates"] and "C7" in view["candidates"]


def test_a_skip_whose_predicate_now_holds_is_squashed_but_a_user_skip_is_not(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "REJECT", "decided", base=rid(rc, cands, led, "C1")))
    base_c1c = rid(rc, cands, led, "C1c")
    led["entries"].append(entry("D06", "C1c", "SKIP", "skipped", base=base_c1c,
                                conditional_refs=["C1"],
                                evidence={"predicate": "decision(C1).chosen == ADOPT", "user": False}))
    assert rc.squash_view(led, cands, st, code=CODE)["entries"] == {}
    [e for e in led["entries"] if e["id"] == "D05"][0]["chosen"] = "ADOPT"   # the referent flips
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "D06" in view["entries"] and "referent flipped" in view["entries"]["D06"]
    [e for e in led["entries"] if e["id"] == "D06"][0]["evidence"]["user"] = True
    assert "D06" not in rc.squash_view(led, cands, st, code=CODE)["entries"]


def test_a_user_skip_squashes_the_running_experiment_of_that_candidate(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r = rc._recipe_with(led, cands, CODE, "C5", set())
    st["ladder"]["experiments"]["X07"] = exp("X07", "candidate", "C5", "3", r["recipe_id"],
                                            rid(rc, cands, led, "C5"), base_rank("C5"),
                                            status="running")
    st["ladder"]["user_choice"]["C5"] = {"choice": "skip", "at": "2026-09-19T23:59:59"}
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X07" in view["experiments"] and "cause d" in view["experiments"]["X07"]


def test_a_user_skip_older_than_the_header_does_not_squash_it(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r = rc._recipe_with(led, cands, CODE, "C5", set())
    st["ladder"]["experiments"]["X07"] = exp("X07", "candidate", "C5", "3", r["recipe_id"],
                                            rid(rc, cands, led, "C5"), base_rank("C5"),
                                            status="running", started="2026-09-20T00:00:00")
    st["ladder"]["user_choice"]["C5"] = {"choice": "skip", "at": "2026-09-19T00:00:00"}
    assert "X07" not in rc.squash_view(led, cands, st, code=CODE)["experiments"]


def test_a_squashed_experiment_is_resurrected_when_its_base_hash_matches_again(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c2 = rid(rc, cands, led, "C2")
    ckpt = tmp_path / "model_7999.pt"
    ckpt.write_text("x")
    header = exp("X04", "candidate", "C2", "3", "r_c2", base_c2, base_rank("C2"), ckpt=str(ckpt))
    st["ladder"]["squashed"]["X04"] = dict(header, squashed_by="derived", reason="base recipe changed")
    assert "X04" not in rc.squash_view(led, cands, st, code=CODE)["experiments"]
    ckpt.unlink()
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X04" in view["experiments"]
    assert "resurrection refused" in view["experiments"]["X04"]


def test_a_user_squash_is_never_resurrected(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    header = exp("X04", "candidate", "C2", "3", "r", rid(rc, cands, led, "C2"), base_rank("C2"))
    st["ladder"]["squashed"]["X04"] = dict(header, squashed_by="user_choice", reason="decide run")
    assert "X04" in rc.squash_view(led, cands, st, code=CODE)["experiments"]


def test_a_milestone_built_on_a_changed_recipe_is_squashed(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    c2 = entry("D_C2", "C2", "ADOPT", "decided", experiment="X03")
    led["entries"].append(c2)
    recipe_id = rid(rc, cands, led, "C3b", inclusive=True)
    st["ladder"]["experiments"]["X09"] = exp("X09", "milestone", "M1", "3", recipe_id, recipe_id,
                                            {"upto": "C3b", "inclusive": True})
    assert "X09" not in rc.squash_view(led, cands, st, code=CODE)["experiments"]
    led["entries"][-1]["status"] = "overruled"
    led["entries"][-1]["override"] = {"by": "milestone", "choice": "REJECT", "why": "M1"}
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "X09" in view["experiments"] and "milestone recipe changed" in view["experiments"]["X09"]


def test_the_last_decision_squashes_nothing(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "overruled", experiment="X03",
                                base=rid(rc, cands, led, "C1"),
                                override={"by": "user", "choice": "REJECT", "why": "overrule"}))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", "r", 
                                            rid(rc, cands, led, "C1"), base_rank("C1"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert view["experiments"] == {} and view["entries"] == {}


def test_the_view_is_identical_after_a_simulated_crash(rc, cands):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "overruled", experiment="X03",
                                base=rid(rc, cands, led, "C1"),
                                override={"by": "user", "choice": "REJECT", "why": "overrule"}))
    base_c2 = rid(rc, cands, led, "C2")
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C2", "3", "r", "stale", base_rank("C2"))
    first = rc.squash_view(led, cands, st, code=CODE)
    again = rc.squash_view(json.loads(json.dumps(led)), cands, json.loads(json.dumps(st)), code=CODE)
    assert first == again


# ================================================================== ledger transitions
def test_the_transition_table_is_the_plans_table(rc):
    legal = [("speculative", "confirmed", "user"), ("speculative", "overruled", "user"),
             ("speculative", "final", "driver"), ("speculative", "overruled", "driver"),
             ("speculative", "overruled", "milestone"), ("speculative", "squashed", "derived"),
             ("decided", "confirmed", "user"), ("decided", "overruled", "user"),
             ("decided", "final", "driver"), ("decided", "overruled", "driver"),
             ("decided", "overruled", "milestone"), ("decided", "squashed", "derived"),
             ("final", "confirmed", "user"), ("final", "overruled", "user"),
             ("final", "overruled", "milestone"), ("final", "squashed", "derived"),
             ("confirmed", "overruled", "user"), ("confirmed", "squashed", "derived"),
             ("overruled", "confirmed", "user"), ("overruled", "overruled", "user"),
             ("overruled", "squashed", "derived"), ("skipped", "squashed", "user"),
             ("skipped", "squashed", "derived"), ("open", "final", "driver"),
             ("open", "await", "driver"), ("open", "squashed", "derived"),
             ("await", "accepted_i", "user"), ("await", "final", "user"),
             ("await", "final", "driver"), ("await", "await", "driver"),
             ("await", "squashed", "derived")]
    # the RESURRECTION rows: decision 7's "reverting an overrule resurrects an experiment squashed
    # by derived-view causes (a)-(c)" restores the entry's `pre_squash_status`, and `derived` alone
    # may do it (a user squash, cause (d)/(e), is never undone)
    legal += [("squashed", pre, "derived") for pre in
              ("speculative", "decided", "final", "confirmed", "overruled", "skipped", "open",
               "await")]
    for src, dst, actor in legal:
        rc.check_transition(src, dst, actor)
    illegal = [("confirmed", "final", "driver"), ("confirmed", "overruled", "driver"),
               ("final", "final", "driver"), ("skipped", "final", "driver"),
               ("decided", "accepted_i", "user"), ("speculative", "confirmed", "driver"),
               ("await", "accepted_i", "driver"), ("squashed", "final", "driver"),
               ("squashed", "decided", "user"), ("squashed", "final", "milestone")]
    for src, dst, actor in illegal:
        with pytest.raises(ValueError):
            rc.check_transition(src, dst, actor)
    with pytest.raises(ValueError, match="unknown source"):
        rc.check_transition("nonsense", "final", "driver")
    # PLAN_AMENDMENTS 24: `overruled -> confirmed` is the revert of a DRIVER/MILESTONE override
    for by in ("driver", "milestone", None):
        rc.check_transition("overruled", "confirmed", "user", override_by=by)
    with pytest.raises(ValueError, match="nothing to revert"):
        rc.check_transition("overruled", "confirmed", "user", override_by="user")


def test_the_resurrection_write_goes_through_the_transition_guard(rc, cands, tmp_path, monkeypatch):
    """FIX_QUEUE 1: the squash and the resurrection are ledger writes like any other.

    Both used to set ``status`` directly, so the one table a reader consults to know what may move
    an entry did not describe them and nothing would have caught a resurrection into a status the
    plan forbids.
    """
    seen = []
    real = rc.check_transition
    monkeypatch.setattr(rc, "check_transition",
                        lambda *a, **k: (seen.append(tuple(a)), real(*a, **k))[1])
    ctx = _depth3_ladder(rc, cands, tmp_path)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    assert ("decided", "squashed", "derived") in seen
    seen.clear()
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "ADOPT",
                                    "why": "reverted"})
    rc.ingest_overrides(ctx)
    assert ("squashed", "decided", "derived") in seen, seen
    assert ("squashed", "skipped", "derived") in seen, "the SKIP entry too"


def test_a_frozen_decision_can_only_be_moved_by_the_user(rc):
    """A driver pair or a milestone may never touch a ``confirmed`` entry -- that is what frozen means."""
    for actor in ("driver", "milestone"):
        with pytest.raises(ValueError):
            rc.check_transition("confirmed", "overruled", actor)


# ================================================================== code_sha
def test_the_code_hash_is_order_free_and_moves_with_a_blob(rc):
    rows = ["aaa parkour/x/mdp/a.py", "bbb parkour/x/mdp/b.py"]
    assert rc.sha_rows(rows) == rc.sha_rows(list(reversed(rows)))
    assert rc.sha_rows(rows) != rc.sha_rows(["aaa parkour/x/mdp/a.py", "ccc parkour/x/mdp/b.py"])


def test_documentation_records_and_tests_are_filtered_out_of_the_code_hash(rc):
    dropped = ["parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/README.md",
               "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-19_x/CHARTER.md",
               "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-19_x/candidates.yaml",
               "tests/unit/test_x.py"]
    kept = ["parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_rewards.py",
            "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml",
            "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_phases.py"]
    for path in dropped:
        assert rc.code_drop(path) is True, path
    for path in kept:
        assert rc.code_drop(path) is False, path


def test_the_real_tree_hashes_stably_and_names_the_code_only(rc):
    a, b = rc.code_sha(), rc.code_sha()
    assert a == b and len(a) == 64
    rows = rc.code_rows()
    assert rows and all(" " in r for r in rows)
    assert rc.sha_rows(rows) == a
    paths = [r.split(" ", 1)[1] for r in rows]
    assert not any(p.endswith(".md") or "/tests/" in p for p in paths)
    assert any(p.endswith("/experiments/tools/run_phases.py") for p in paths)
    assert any(p.endswith("/experiments/eval/scenarios_omni.yaml") for p in paths)
    assert not any("/experiments/2026-" in p for p in paths)
    assert not any(p.endswith("run_candidates.py") or p.endswith("ledger_cli.py")
                   or p.endswith("/tools/resolution.py") for p in paths), \
        "the ladder's own tools are outside code_sha by design"


def test_the_tools_hash_covers_the_tools_and_the_replay_script(rc, cands, tmp_path):
    extra = tmp_path / "replay.py"
    extra.write_text("print(1)")
    a = rc.tools_sha()
    b = rc.tools_sha(extra)
    assert a != b and len(a) == 64
    extra.write_text("print(2)")
    assert rc.tools_sha(extra) != b


# ================================================================== the validator
def test_the_shipped_campaign_spec_validates(rc, cands):
    verdicts = rc.validate_candidates(cands, strict=False)
    bad = [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]
    assert bad == []
    assert {v["id"] for v in verdicts} >= {"<campaign>", "chain:curriculum", "C1", "C12", "<overlays>"}
    assert [v["id"] for v in verdicts if v["deferred"]] == ["C10", "C12"]


def _bad(cands, **patch) -> dict:
    out = copy.deepcopy(cands)
    for key, value in patch.items():
        out[key] = value
    return out


def _refusals(rc, cands, cid=None) -> list[str]:
    out = []
    for v in rc.validate_candidates(cands, strict=False, check_replay=False):
        if cid is None or v["id"] == cid:
            out += v["reasons"]
    return out


def test_a_cap_without_a_cadence_radius_is_refused(rc, cands):
    """The dry run's own case: ``ANG_VEL_MAX=0.15`` with no ``CLOCK_TURN_R`` -> a frozen gait clock."""
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "kinematics", "stages": "all", "question": "no cadence note",
                          "overlay": {"KRABBY_OMNI_ANG_VEL_MAX": "0.15"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    reasons = _refusals(rc, doc, "CX")
    assert any("below the gait clock's stop speed" in r for r in reasons), reasons


def test_a_cap_that_changes_the_cadence_without_saying_so_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "kinematics", "stages": "all", "question": "silent",
                          "overlay": {"KRABBY_OMNI_ANG_VEL_MAX": "0.21",
                                      "KRABBY_OMNI_CLOCK_TURN_R": "1.4"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    reasons = _refusals(rc, doc, "CX")
    assert any("changes the\ngait cadence" in r or "changes the gait cadence" in r for r in reasons), reasons
    doc["candidates"][0]["question"] = "cadence-preserving? no -- the cadence changes deliberately"
    assert not any("cadence" in r for r in _refusals(rc, doc, "CX"))


def test_a_cap_below_the_manifests_turn_feasibility_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "kinematics", "stages": "all",
                          "question": "cadence preserved", "absolute_units": True,
                          "overlay": {"KRABBY_OMNI_ANG_VEL_MAX": "0.14",
                                      "KRABBY_OMNI_CLOCK_TURN_R": "2.6"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    reasons = _refusals(rc, doc, "CX")
    assert any("infeasible" in r for r in reasons), reasons
    doc["candidates"][0]["overlay"]["KRABBY_OMNI_ANG_VEL_MAX"] = "0.15"
    doc["candidates"][0]["overlay"]["KRABBY_OMNI_CLOCK_TURN_R"] = "2.4"
    assert not any("infeasible" in r for r in _refusals(rc, doc, "CX"))


def test_a_stale_slots_beside_a_pool_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["chains"]["curriculum"]["stages"][0]["overlay"]["KRABBY_OMNI_SLOTS"] = "0.5:0.5:0:0"
    reasons = _refusals(rc, doc, "chain:curriculum")
    assert any("the pool projects to" in r for r in reasons), reasons


def test_a_pool_without_an_empty_dir_band_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["chains"]["curriculum"]["stages"][0]["overlay"].pop("KRABBY_OMNI_DIR_BAND")
    assert any("KRABBY_OMNI_DIR_BAND=''" in r for r in _refusals(rc, doc, "chain:curriculum"))
    doc2 = copy.deepcopy(cands)
    doc2["chains"]["curriculum"]["stages"][0]["overlay"]["KRABBY_OMNI_DIR_BAND"] = "0:0"
    assert any("together with a pool" in r for r in _refusals(rc, doc2, "chain:curriculum"))


def test_a_type_below_the_floor_share_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["curriculum"]["stages"][-1]
    stage["overlay"]["KRABBY_OMNI_CMD_POOL"] = stage["overlay"]["KRABBY_OMNI_CMD_POOL"].replace(
        "side_l_turn:0.035", "side_l_turn:0.01")
    assert any("POOL_FLOOR_SHARE" in r for r in _refusals(rc, doc, "chain:curriculum"))


def test_a_pool_written_out_of_type_order_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["chains"]["curriculum"]["stages"][0]["overlay"]["KRABBY_OMNI_CMD_POOL"] = "fwd:0.8,stand:0.2"
    assert any("POOL_TYPES order" in r for r in _refusals(rc, doc, "chain:curriculum"))


def test_an_unknown_knob_name_is_refused_and_the_yaw_weight_pins_are_kept(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "reward", "stages": "all", "question": "typo",
                          "overlay": {"KRABBY_YAW_WEIGHT": "0.2"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    assert any("not a known knob" in r for r in _refusals(rc, doc, "CX"))
    assert "KRABBY_YAW_W" in rc.knob_whitelist()
    assert "KRABBY_YAW_WEIGHT" not in rc.knob_whitelist()
    for banned in rc.FORBIDDEN_OVERLAY_KEYS:
        assert banned not in rc.knob_whitelist()


def test_a_knob_with_no_dump_check_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "reward", "stages": "all", "question": "unaudited",
                          "overlay": {"KRABBY_STAND_FRAC": "0.1"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    assert any("OVERLAY_DUMP_CHECKS" in r for r in _refusals(rc, doc, "CX"))


def test_a_value_that_does_not_round_trip_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "observation", "stages": "all", "question": "bad value",
                          "overlay": {"KRABBY_OMNI_FACING_OBS_ENCODING": "polar"}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    assert any("does not round-trip" in r for r in _refusals(rc, doc, "CX"))


def test_an_unhonourable_speed_scale_dies_at_compose_time(rc, cands):
    """The walk-floor bound from the command module, applied to a ``pool_edit`` at LOAD time."""
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "C1", "kind": "curriculum", "chain": "curriculum",
                          "question": "creeper", "teacher_overlay": {},
                          "pool_edit": {"speed_scale": {"omni": 0.5}}}]
    doc["milestones"] = [{"id": "M1", "after": "C1", "promote": ["2a", "2b"]}]
    reasons = _refusals(rc, doc, "C1")
    assert any("walk-floor bound" in r and "0.88" in r for r in reasons), reasons
    doc["candidates"][0]["pool_edit"] = {"speed_scale": {"omni": 0.95}}
    assert not any("walk-floor bound" in r for r in _refusals(rc, doc, "C1"))


def test_a_pool_edit_pattern_that_bites_nothing_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "C1", "kind": "curriculum", "chain": "curriculum", "question": "typo",
                          "teacher_overlay": {}, "pool_edit": {"ang_scale": {"*_trun": 0.3}}}]
    doc["milestones"] = [{"id": "M1", "after": "C1", "promote": ["2a", "2b"]}]
    assert any("matches no type in ANY" in r for r in _refusals(rc, doc, "C1"))


def test_a_conditional_ref_outside_depends_on_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C1b":
            cand.pop("depends_on")
    assert any("not in depends_on" in r for r in _refusals(rc, doc, "C1b"))


def test_a_forward_dependency_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C2":
            cand["depends_on"] = ["C9"]
    assert any("not ranked before" in r for r in _refusals(rc, doc, "C2"))


def test_an_unparseable_conditional_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C7":
            cand["conditional"] = "recipe.has_pool and decision(C1b).chosen !="
    assert any("does not parse" in r for r in _refusals(rc, doc, "C7"))


def test_an_eval_carry_key_outside_the_drivers_list_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C5":
            cand["eval_carry"] = ["KRABBY_OMNI_CMD_POOL"]
    assert any("not in run_phases.EVAL_CARRY" in r for r in _refusals(rc, doc, "C5"))


def test_a_replay_flag_the_script_does_not_have_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C2":
            cand["replay_args"] = ["--yaw-progress-weight", "3"]
    verdicts = rc.validate_candidates(doc, strict=False)
    reasons = [r for v in verdicts if v["id"] == "C2" for r in v["reasons"]]
    assert any("--help does not" in r for r in reasons), reasons


def test_the_shipped_replay_args_are_all_real_flags(rc, cands):
    flags = rc.replay_flags(rc.REPO / cands["replay"]["script"])
    assert {"--yaw-w", "--yaw-progress-w", "--facing-turn-w", "--coupled-sigma", "--progress-w",
            "--new-ang-turn-only", "--walker"} <= flags
    for cand in cands["candidates"]:
        for tok in cand.get("replay_args") or []:
            if str(tok).startswith("--"):
                assert tok in flags, (cand["id"], tok)


def test_a_chain_that_does_not_total_the_screen_budget_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["chains"]["curriculum"]["stages"][0]["iters"] = 999
    assert any("iterations; every screen is" in r for r in _refusals(rc, doc, "chain:curriculum"))


def test_two_candidates_writing_a_raw_pool_for_one_stage_are_refused(rc, cands):
    doc = copy.deepcopy(cands)
    pool = doc["chains"]["curriculum"]["stages"][-1]["overlay"]["KRABBY_OMNI_CMD_POOL"]
    for cid in ("C2", "C4"):
        for cand in doc["candidates"]:
            if cand["id"] == cid:
                cand["overlay"] = {"KRABBY_OMNI_CMD_POOL": pool,
                                   "KRABBY_OMNI_SLOTS": "0.15:0.55:0.1:0.2",
                                   "KRABBY_OMNI_DIR_BAND": ""}
    assert any("cannot both write the pool string" in r for r in _refusals(rc, doc, "<overlays>"))


def test_a_2a_facing_overlay_without_a_teacher_overlay_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"] = [{"id": "CX", "kind": "reward", "question": "stray facing",
                          "overlay": {"1c": {"KRABBY_OMNI_FACING": "0.2:0.8"},
                                      "2a": {"KRABBY_OMNI_FACING": "0.2:0.8"}}}]
    doc["milestones"] = [{"id": "M1", "after": "CX", "promote": ["2a", "2b"]}]
    assert any("without declaring it in" in r for r in _refusals(rc, doc, "CX"))


def test_strict_validation_raises_with_every_reason(rc, cands):
    doc = copy.deepcopy(cands)
    doc["plant"] = "NOT-A-PLANT"
    with pytest.raises(SystemExit, match="not honourable"):
        rc.validate_candidates(doc, strict=True, check_replay=False)


def test_a_floor_hold_outside_the_stages_holds_is_refused(rc, cands):
    doc = copy.deepcopy(cands)
    doc["chains"]["curriculum"]["stages"][0]["floor"]["hold"] = "diag"
    assert any("floor hold" in r for r in _refusals(rc, doc, "chain:curriculum"))


# ================================================================== the predicate language
def test_the_predicate_vocabulary_is_fixed_and_reports_its_values(rc):
    ctx = {"decision": lambda cid: {"chosen": "REJECT", "floor_stage": "c4",
                                    "pooled_confidence": "none", "status": "decided",
                                    "confidence": "clear"}[0:] if False else
           {"chosen": "REJECT", "floor_stage": "c4", "pooled_confidence": "none",
            "status": "decided", "confidence": "clear"},
           "recipe": {"chain": "curriculum", "has_pool": True},
           "screen": {"track_ratio_move_tail": 0.25, "yaw_progress_turn_min": None},
           "stage_order": ["c1", "c2", "c3", "c4", "c5", "c6"]}
    cases = [("decision(C1).chosen == REJECT", True),
             ("decision(C1).chosen != REJECT", False),
             ("decision(C1).floor_stage is None", False),
             ("decision(C1).floor_stage is not None", True),
             ("decision(C1).floor_stage >= c4", True),
             ("decision(C1).floor_stage >= c5", False),
             ("decision(C1).pooled_confidence in (uncertain, disputed)", False),
             ("decision(C1).status == decided", True),
             ("recipe.chain == curriculum", True),
             ("recipe.has_pool", True),
             ("recipe.has_pool and decision(C1).chosen != ADOPT", True),
             ("screen.track_ratio_move_tail < 0.45", True),
             ("screen.yaw_progress_turn_min < 0.10", False),
             ("decision(C1).chosen == REJECT and (decision(C1).floor_stage is None or "
              "decision(C1).floor_stage >= c4)", True),
             ("decision(C1).chosen == ADOPT or recipe.has_pool", True)]
    for text, want in cases:
        got, why = rc.compile_predicate(text).evaluate(ctx)
        assert got is want, (text, why)
        assert "=" in why or "is" in why


def test_the_predicate_refuses_anything_outside_the_vocabulary(rc):
    for text in ("__import__('os').system('x')", "decision(C1)", "screen", "1 +", "decision(C1).x ==",
                 "recipe.has_pool and"):
        with pytest.raises(ValueError):
            rc.compile_predicate(text)


def test_conditional_refs_are_derived_from_the_text(rc, cands):
    assert rc.conditional_refs("decision(C1).chosen == REJECT and decision(C3a).status == final") == \
        ["C1", "C3a"]
    assert rc.conditional_refs("screen.completion < 0.5") == []
    for cand in cands["candidates"]:
        refs = rc.conditional_refs(cand.get("conditional"))
        assert set(refs) <= set(cand.get("depends_on") or []), cand["id"]


# ================================================================== the notification contract
def test_every_tag_round_trips_through_format_and_parse(rc, cli):
    for tag in cli.NOTIFY_TAGS:
        line = cli.format_notify(tag, "D08", ["a | b", "c\nd", "  e  "])
        got = cli.parse_notify(line)
        assert got["tag"] == tag and got["id"] == "D08"
        assert got["fields"] == ["a / b", "c d", "e"]
        assert "\n" not in line
        stamped = f"2026-09-19 12:34 {line}"
        assert cli.parse_notify(stamped)["fields"] == got["fields"]
        assert cli.parse_notify(stamped)["stamp"] == "2026-09-19 12:34"


def test_an_unknown_tag_is_refused_on_both_sides(rc, cli):
    with pytest.raises(ValueError):
        cli.format_notify("NOPE", "D01", ["x"])
    with pytest.raises(ValueError):
        cli.parse_notify("NOPE D01 | x")


def test_the_skip_line_carries_the_run_command(rc, cli):
    line = cli.format_notify("DECISION", "D07", ["C3b", "chosen SKIP (predicate: ...)",
                                                 "run it: ledger_cli.py decide C3b --choose run"])
    got = cli.parse_notify(line)
    assert got["tag"] == "DECISION" and got["id"] == "D07"
    assert "chosen SKIP" in got["fields"][1] and "decide C3b --choose run" in got["fields"][2]


# ================================================================== a real campaign context
def make_ctx(rc, cands, tmp_path, *, led=None, st=None, code=CODE, name="camp", resolution=None):
    """A real :class:`Ctx` over a scratch campaign dir: state.json, ledger.json, logs/, REPORT.md."""
    cdir = Path(tmp_path) / name
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "candidates.yaml").write_text(CANDS_YAML.read_text())
    if led is not None:
        rc.lc.write_ledger(cdir, led)
    if st is not None:
        (cdir / "state.json").write_text(json.dumps(st))
    args = types.SimpleNamespace(campaign_dir=str(cdir), candidates=None, unit=None, dry_run=False,
                                 max_wall_h=None, accept_code_change=None, smoke_stage=None,
                                 accept_candidates_change=None)
    ctx = rc.Ctx(cdir, cands, args)
    ctx.code = code
    ctx.tools = "TOOLS"
    ctx.resolution = resolution if resolution is not None else resol()
    ctx.recompute()
    ctx.save()
    return ctx


def notify_lines(ctx) -> list[str]:
    path = ctx.dir / "notify.log"
    return path.read_text().splitlines() if path.exists() else []


def test_pending_is_derived_from_the_ledger_not_stored(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert [e["id"] for e in rc.pending_decisions(ctx.ledger)] == ["D05"]
    assert ctx.lad["pending"] == ["D05"]
    ctx.ledger["entries"][-1]["status"] = "confirmed"
    ctx.recompute()
    assert ctx.lad["pending"] == []


def test_a_user_confirm_freezes_the_choice_and_drops_the_candidates_own_pair(rc, cands, tmp_path):
    """Verification 7's in-flight drill: ``confirm <Did>`` while the s2 header is live."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain", base=base,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                            base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base,
                                            base_rank("C1"), status="running",
                                            inflight={"stage": "c1", "iters": 1000, "tag": "t"},
                                            decision="D05")
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C9", "2", base, base, base_rank("C9"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm", "choice": "ADOPT",
                                    "why": "the user has seen enough"})
    assert rc.ingest_overrides(ctx) is True
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "confirmed" and d05["frozen"] is True
    assert d05["override"]["by"] == "user"
    assert "X04" in ctx.lad["squashed"] and ctx.lad["squashed"]["X04"]["squashed_by"] == "user_choice"
    assert "X04" not in ctx.lad["experiments"], "the in-flight pair run is not resumed"
    assert "X05" in ctx.lad["experiments"], "the base recipe's twin is untouched"
    assert ctx.lad["pending"] == []
    assert rc.find_inflight(ctx) is None
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["candidate"] != "C1"
    assert any("CONFIRM D05" in line for line in notify_lines(ctx))


def test_a_user_overrule_squashes_the_dependents_and_reports_them(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c1 = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03", base=base_c1,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            base_c1, base_rank("C1"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    base_c2 = rc.compose_recipe(ctx.ledger, cands, CODE, upto="C2")["recipe_id"]
    r_c2 = rc._recipe_with(ctx.ledger, cands, CODE, "C2", set())
    ctx.ledger["entries"].append(entry("D06", "C2", "REJECT", "decided", experiment="X04",
                                       base=base_c2, recipe=r_c2["recipe_id"]))
    ctx.lad["experiments"]["X04"] = exp("X04", "candidate", "C2", "3", r_c2["recipe_id"], base_c2,
                                       base_rank("C2"))
    ctx.write_ledger()
    ctx.recompute()
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "the user disagrees"})
    rc.ingest_overrides(ctx)
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "overruled"
    assert "X04" in ctx.lad["squashed"] and ctx.lad["squashed"]["X04"]["squashed_by"] == "derived"
    assert rc.entry_by_id(ctx.ledger, "D06")["status"] == "squashed"
    assert "C2" in ctx.view["candidates"]
    report = (ctx.dir / "REPORT.md").read_text()
    assert "## SQUASHED" in report and "X04" in report
    assert any("SQUASH" in line for line in notify_lines(ctx))
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["candidate"] in ("C1b", "C2")


def test_a_derived_squash_with_no_pre_status_is_repaired_with_a_warn_not_a_traceback(rc, cands,
                                                                                     tmp_path):
    """R5: the agreement assertion is a guard, and a guard must not kill the launch.

    An entry stored ``squashed`` / ``squashed_by: derived`` with NO ``pre_squash_status`` is
    unreachable through any driver write (every derived squash records the status it replaced) but
    reachable by a hand-edited ledger -- and the assertion sits in ``squash_and_report``, which the
    launch calls BEFORE ``loop()``'s halt handling, so the campaign died with a traceback and no
    HALT line. Repaired to ``decided`` (terminal, unfrozen, re-decidable) with a WARN on the SQUASH
    line instead.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "ADOPT", "squashed", experiment="X03",
                                squashed_by="derived"))       # hand-edited: no pre_squash_status
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert "D05" not in ctx.view["entries"], "the view says this entry is live"

    note = rc.squash_and_report(ctx, "launch reconcile (test)")

    got = rc.entry_by_id(ctx.ledger, "D05")
    assert got["status"] == "decided" and "squashed_by" not in got
    assert got["history"][-1]["to"] == "decided" and "no pre_squash_status" in got["history"][-1]["why"]
    line = [l for l in notify_lines(ctx) if " SQUASH " in l][-1]
    assert "WARN" in line and "D05" in line, line
    assert "- WARN D05" in (ctx.dir / "REPORT.md").read_text()
    assert "repaired" in (ctx.dir / "CHANGELOG.md").read_text()
    assert note.startswith("; squashed 0"), note
    # and a second reconcile is a no-op: the record and the view now agree
    lines = len(notify_lines(ctx))
    rc.squash_and_report(ctx, "launch reconcile (test, again)")
    assert len(notify_lines(ctx)) == lines
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "decided"


def test_reverting_an_overrule_resurrects_the_squashed_experiment(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c1 = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03", base=base_c1,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            base_c1, base_rank("C1"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    base_c2 = rc.compose_recipe(ctx.ledger, cands, CODE, upto="C2")["recipe_id"]
    ckpt = Path(tmp_path) / "c2" / "model_7999.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("x")
    ctx.lad["experiments"]["X04"] = exp("X04", "candidate", "C2", "3", "r_c2", base_c2,
                                       base_rank("C2"), ckpt=str(ckpt))
    ctx.ledger["entries"].append(entry("D06", "C2", "REJECT", "decided", experiment="X04",
                                       base=base_c2))
    ctx.write_ledger()
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    assert "X04" in ctx.lad["squashed"]
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "ADOPT",
                                    "why": "reverted"})
    rc.ingest_overrides(ctx)
    assert "X04" in ctx.lad["experiments"] and "X04" not in ctx.lad["squashed"]
    assert rc.entry_by_id(ctx.ledger, "D06")["status"] == "decided"


def _depth3_ladder(rc, cands, tmp_path):
    """C1 ADOPT -> C1c/C2 on C1 -> C3a/C4 on C1+C2 -> C5: a THREE-deep dependency chain.

    Every screen has a real checkpoint on disk, because resurrection is refused without the files.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rows = [("D05", "C1", "ADOPT", "X03"), ("D06", "C1c", "REJECT", "X04"),
            ("D07", "C2", "ADOPT", "X05"), ("D08", "C3a", "REJECT", "X06"),
            ("D09", "C4", "ADOPT", "X07"), ("D10", "C5", "REJECT", "X08")]
    for did, cid, chosen, xid in rows:
        base = rc.compose_recipe(ctx.ledger, cands, CODE, upto=cid,
                                 squashed=set(ctx.view["entries"]))["recipe_id"]
        recipe = rc._recipe_with(ctx.ledger, cands, CODE, cid, set(ctx.view["entries"]))["recipe_id"]
        ckpt = Path(tmp_path) / xid / "model_7999.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("x")
        ctx.lad["experiments"][xid] = exp(xid, "candidate", cid, "3", recipe, base,
                                          base_rank(cid), ckpt=str(ckpt))
        ctx.ledger["entries"].append(entry(did, cid, chosen, "decided", experiment=xid, base=base,
                                           recipe=recipe))
        ctx.write_ledger()
        ctx.recompute()
    # C1b's SKIP: its predicate reads `decision(C1).chosen == REJECT`, so it is FALSE while C1 is
    # adopted -- and TRUE again the moment the overrule lands (derived-view cause (b)).
    rc.write_skip_entry(ctx, "C1b", "predicate", "C1 is adopted", predicate="decision(C1) != REJECT")
    ctx.save()
    return ctx


def test_reverting_an_overrule_resurrects_a_three_deep_chain_in_one_ingest(rc, cands, tmp_path):
    """The plan's "reverting an overrule resurrects an experiment squashed by causes (a)-(c)".

    Resurrection has to be TRANSITIVE, and that is a purity property of the view, not a loop count:
    while ``newest_entry`` also read an entry's STORED ``squashed`` status, ``compose_recipe``
    inside ``squash_view``'s own fixpoint saw the PREVIOUS pass's statuses, so one ingest undid
    exactly one dependency level, a fresh ``Ctx`` did not heal it, and ``next_item`` answered with a
    re-screen of an intact window (3.7-3.9 GPU-h per level). Pinned at depth three, which is where
    a one-level restore and a fixpoint differ.
    """
    ctx = _depth3_ladder(rc, cands, tmp_path)
    live_before = dict(ctx.lad["experiments"])
    statuses = {e["id"]: e["status"] for e in rc.entries(ctx.ledger)}

    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    assert set(ctx.lad["squashed"]) >= {"X04", "X05", "X06", "X07", "X08"}, ctx.lad["squashed"]

    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "ADOPT",
                                    "why": "reverted"})
    rc.ingest_overrides(ctx)                                    # ONE ingest, every level

    assert ctx.lad["squashed"] == {}, "a revert must resurrect the whole chain, not one level"
    assert set(ctx.lad["experiments"]) == set(live_before)
    # D05 itself is `overruled` (the user's own second answer stands); everything BELOW it is back
    assert {e["id"]: e["status"] for e in rc.entries(ctx.ledger) if e["id"] != "D05"} \
        == {k: v for k, v in statuses.items() if k != "D05"}
    assert rc.lc.effective_choice(rc.entry_by_id(ctx.ledger, "D05")) == "ADOPT"
    assert ctx.view["experiments"] == {} and ctx.view["entries"] == {}
    # the record is now a fixpoint: nothing is stored `squashed` that the view does not list
    assert all(e["status"] != "squashed" for e in rc.entries(ctx.ledger)
               if e["id"] not in ctx.view["entries"])

    # a fresh Ctx (a relaunch) derives the same empty view from the records on disk
    fresh = rc.Ctx(ctx.dir, cands, ctx.args)
    fresh.code, fresh.tools, fresh.resolution = CODE, "TOOLS", resol()
    fresh.recompute()
    assert fresh.view["experiments"] == {} and fresh.view["entries"] == {}

    # and the queue does not re-screen a candidate whose intact screen has just come back
    decided = {cid for cid in ("C1", "C1b", "C1c", "C2", "C3a", "C4", "C5")}
    nxt = rc.next_item(cands, fresh.ledger, fresh.state, code=CODE, view=fresh.view)
    assert nxt is None or nxt.get("candidate") not in decided, nxt


def test_the_squash_view_is_a_pure_function_of_the_effective_ledger(rc, cands, tmp_path):
    """SQUASH (6): "recomputed from the ledger on every launch" -- with no help from the record.

    The stored ``status: squashed`` an earlier pass wrote is the view's own CACHE. While
    ``newest_entry`` read it, ``compose_recipe`` INSIDE the fixpoint composed its recipes from a
    ledger that still hid the deeper levels, so a fresh ``Ctx`` over a reverted ledger derived a
    view that went on squashing them and no relaunch healed it. Driven here through the crash
    window SQUASH (6) names: the revert is in ``ledger.json`` and nothing has reconciled the record.
    """
    ctx = _depth3_ladder(rc, cands, tmp_path)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    assert [e["id"] for e in rc.entries(ctx.ledger) if e["status"] == "squashed"]
    rc.entry_by_id(ctx.ledger, "D05")["override"]["choice"] = "ADOPT"
    ctx.write_ledger()

    fresh = rc.Ctx(ctx.dir, cands, ctx.args)
    fresh.code, fresh.tools, fresh.resolution = CODE, "TOOLS", resol()
    view = fresh.recompute()                       # ONE recompute, no squash_and_report anywhere
    assert view["experiments"] == {} and view["entries"] == {} and view["candidates"] == []


def test_a_launch_reconciles_the_record_with_the_view_and_reports_the_skipped_squash(rc, cands,
                                                                                     tmp_path,
                                                                                     monkeypatch):
    """SQUASH (3)+(6) + F10(a): a crash between the ledger write and the state save.

    The override is in ``ledger.json`` and the view is right, but the headers it squashes are still
    under ``state.ladder.experiments`` and no SQUASH line, REPORT block or CHANGELOG row was ever
    written for it. The launch pass writes them once, and a launch with nothing to reconcile writes
    nothing at all.
    """
    ctx = _depth3_ladder(rc, cands, tmp_path)
    uuid = rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule",
                                           "choice": "REJECT", "why": "wrong"})
    rc.ingest_overrides(ctx)
    # rewind the RECORD to before the override (the crash), keeping ledger.json as it is
    ctx.lad["experiments"].update(ctx.lad["squashed"])
    for header in ctx.lad["experiments"].values():
        header.pop("squashed_by", None)
        header.pop("reason", None)
    ctx.lad["squashed"] = {}
    ctx.lad["applied_overrides"] = [uuid]
    ctx.c.save()

    monkeypatch.setattr(rc, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(rc, "code_sha", lambda *a, **k: CODE)
    monkeypatch.setattr(rc, "code_rows", lambda *a, **k: ["aaa parkour/x/mdp/a.py"])
    monkeypatch.setattr(rc, "ensure_resolution", lambda c: {})
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    monkeypatch.setattr(rc, "loop", lambda c: 0)
    assert rc.main(["--campaign-dir", str(ctx.dir)]) == 0

    st = json.loads((ctx.dir / "state.json").read_text())["ladder"]
    assert set(st["squashed"]) >= {"X04", "X05", "X06", "X07", "X08"}
    assert not (set(st["experiments"]) & set(st["squashed"]))
    lines = [l for l in notify_lines(ctx) if "SQUASH launch reconcile" in l]
    assert len(lines) == 1 and "X07" in lines[0], lines
    assert "## SQUASHED — launch reconcile" in (ctx.dir / "REPORT.md").read_text()
    assert "SQUASH — launch reconcile" in (ctx.dir / "CHANGELOG.md").read_text()

    # and a second launch, with nothing left to reconcile, writes no second line
    assert rc.main(["--campaign-dir", str(ctx.dir)]) == 0
    assert len([l for l in notify_lines(ctx) if "SQUASH launch reconcile" in l]) == 1


def _screened(rc, cands, led, st, tmp_path, did, cid, chosen, xid):
    """One decided candidate with a real screen and a real checkpoint (resurrection needs the file)."""
    recipe = rc._recipe_with(led, cands, CODE, cid, set())["recipe_id"]
    base = rid(rc, cands, led, cid)
    ckpt = Path(tmp_path) / xid / "model_7999.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("x")
    led["entries"].append(entry(did, cid, chosen, "decided", experiment=xid, base=base,
                                recipe=recipe))
    st["ladder"]["experiments"][xid] = exp(xid, "candidate", cid, "3", recipe, base,
                                           base_rank(cid), ckpt=str(ckpt))


def _milestone_row(rc, cands, led, mid, chosen, promoted):
    led["entries"].append({"id": mid, "kind": "milestone", "candidate": mid, "chosen": chosen,
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {}, "eligibility": 1, "promoted_head": promoted,
                           "recipe_id": rc.milestone_recipe(cands, led, mid, CODE)["recipe_id"]})


def _plan_scenario_ladder(rc, cands, tmp_path, *, c5="REJECT", name="camp"):
    """The plan's named scenario: C5, then a clear ADOPT on C6, C7, and a ``final`` M2 after them."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for i, cid in enumerate(["C1", "C1b", "C1c", "C2", "C3a", "C3b"]):
        led["entries"].append(entry(f"D{5 + i:02d}", cid, "REJECT", "decided",
                                    base=rid(rc, cands, led, cid)))
    _milestone_row(rc, cands, led, "M1", "WATCH", {})
    led["entries"].append(entry("D11", "C4", "REJECT", "decided", base=rid(rc, cands, led, "C4")))
    _screened(rc, cands, led, st, tmp_path, "D12", "C5", c5, "X07")
    _screened(rc, cands, led, st, tmp_path, "D13", "C6", "ADOPT", "X08")
    _screened(rc, cands, led, st, tmp_path, "D14", "C7", "REJECT", "X09")
    _milestone_row(rc, cands, led, "M2", "PASS", {"2a": {}, "2b": {}})
    return make_ctx(rc, cands, tmp_path, led=led, st=st, name=name)


def _re_screen(rc, cands, ctx, cid, xid):
    """Dispatch a re-run of ``cid`` whose screen is a clear ADOPT, the way ``loop`` would."""
    item, header = _candidate_header(rc, cands, ctx, cid, xid=xid,
                                     screen_=screen(closed=0.72, prog=0.11),
                                     started="2026-09-19T09:00:00")
    ctx.lad["experiments"][xid] = header
    rc._dispatch_candidate(ctx, item, header)
    return rc.entries(ctx.ledger)[-1]


def test_a_decision_that_moves_the_recipe_reconciles_the_record_like_an_overrule(rc, cands, tmp_path):
    """Amendment 28 / the plan's named scenario: C5 REJECT, C6 ADOPT, M2 final -> ``decide C5 run``
    -> the re-run lands ADOPT.

    The decision MOVES the recipe, so SQUASH (3) applies to it exactly as it does to an overrule:
    the headers move to ``state.ladder.squashed``, the dependent decisions and the milestone go to
    ``status: squashed``, and ONE SQUASH line / REPORT block / CHANGELOG row records it. With only
    ``recompute()`` the view was right and the record was not, for as long as the campaign ran
    without another override: ``_supersedes`` on the C6 re-run read ``[]``, the CLI accepted a
    ``confirm`` of a view-squashed entry, and nothing told the user 5.x GPU-h had just been voided.
    """
    ctx = _plan_scenario_ladder(rc, cands, tmp_path)
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen"})
    rc.ingest_overrides(ctx)
    assert set(ctx.view["entries"]) == {"D12"} and ctx.view["experiments"] == {}
    before = len([l for l in notify_lines(ctx) if " SQUASH " in l])

    new = _re_screen(rc, cands, ctx, "C5", "X11")
    assert (new["candidate"], new["chosen"], new["confidence"]) == ("C5", "ADOPT", "clear")

    # SQUASH (3): the headers are archived and the dependent entries are marked, at THIS write
    assert set(ctx.lad["squashed"]) == {"X08", "X09"}, ctx.lad["squashed"]
    assert not (set(ctx.lad["experiments"]) & {"X08", "X09"})
    statuses = {e["id"]: e["status"] for e in rc.entries(ctx.ledger)}
    assert statuses["D13"] == statuses["D14"] == statuses["M2"] == "squashed"
    assert rc.entry_by_id(ctx.ledger, "D13")["pre_squash_status"] == "decided"

    # exactly one SQUASH line, naming the decision, plus the REPORT block and the CHANGELOG row
    fresh = [l for l in notify_lines(ctx) if " SQUASH " in l][before:]
    assert len(fresh) == 1 and new["id"] in fresh[0] and "X08" in fresh[0], fresh
    assert f"## SQUASHED — {new['id']} C5 -> ADOPT" in (ctx.dir / "REPORT.md").read_text()
    assert f"SQUASH — {new['id']} C5 -> ADOPT" in (ctx.dir / "CHANGELOG.md").read_text()
    decision = [l for l in notify_lines(ctx) if f"DECISION {new['id']} " in l][-1]
    assert "this decision moved the recipe" in decision and "squashed 2" in decision

    # minor 9(d) is live again: the C6 re-run's header names the window it replaces
    assert rc._supersedes(ctx, rc._item("candidate", cid="C6", seed="3")) == ["X08"]
    # and the squashed C6 is what re-screens next
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert nxt["label"] == "C6 s3", nxt

    # a `confirm` of the squashed decision is now refused (squashed -> confirmed is derived-only)
    rc.lc.append_override(ctx.dir, {"decision": "D13", "action": "confirm", "why": "keep it"})
    rc.ingest_overrides(ctx)
    applied = [l for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l][-1]
    assert "REFUSED" in applied and "squashed" in applied, applied
    assert rc.entry_by_id(ctx.ledger, "D13")["status"] == "squashed"


def test_a_re_run_that_lands_the_same_choice_resurrects_what_the_decide_run_squashed(rc, cands,
                                                                                     tmp_path):
    """The other half: a re-screen that changes NOTHING must leave no stale squash behind.

    ``decide C5 run`` squashes C5's entry (cause d) and, because C5 stops being adopted, C6's and
    C7's screens with it. When the re-run lands the SAME ADOPT the view is empty again -- but the
    record only caught up at the next override or launch, and in between ``newest_entry`` read
    through a stored ``squashed(derived)`` D13 and ``next_item`` answered with a 3.7-3.9 GPU-h
    re-screen of an intact window.
    """
    ctx = _plan_scenario_ladder(rc, cands, tmp_path, c5="ADOPT")
    live_before = set(ctx.lad["experiments"])
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen"})
    rc.ingest_overrides(ctx)
    assert set(ctx.lad["squashed"]) == {"X08", "X09"}, "the decide run voids C6/C7"

    new = _re_screen(rc, cands, ctx, "C5", "X11")
    assert new["chosen"] == "ADOPT"

    assert ctx.lad["squashed"] == {}, "the same choice must bring the whole level back"
    assert set(ctx.lad["experiments"]) == live_before | {"X11"}
    for did in ("D13", "D14"):
        got = rc.entry_by_id(ctx.ledger, did)
        assert got["status"] == "decided" and "pre_squash_status" not in got and \
            "squashed_by" not in got, got
    assert set(ctx.view["entries"]) == {"D12"}, ctx.view["entries"]
    assert not any(e["status"] == "squashed" for e in rc.entries(ctx.ledger)
                   if e["id"] not in ctx.view["entries"])
    line = [l for l in notify_lines(ctx) if " SQUASH " in l][-1]
    assert "resurrected: X08, X09" in line, line
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert nxt is None or nxt.get("candidate") not in ("C6", "C7"), nxt


def test_decide_run_squashes_the_entry_and_is_never_resurrected(rc, cands, tmp_path):
    """``decide C5 --choose run`` while the driver is stopped: the old entry stays squashed."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C5")
    led["entries"].append(entry("D05", "C5", "REJECT", "decided", experiment="X03", base=base))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C5", "3", "r_c5", base,
                                            base_rank("C5"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen it"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "squashed" and d05["override"]["why"] == "decide run"
    assert "D05" in ctx.view["entries"] and "C5" in ctx.view["candidates"]
    assert ctx.lad["user_choice"]["C5"]["choice"] == "run"
    # a further recompute (a relaunch) does not revive it
    ctx.recompute()
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "squashed"
    assert "D05" in ctx.view["entries"]


def test_decide_skip_writes_a_skip_entry_and_is_refused_on_a_decided_candidate(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "skip", "why": "not now"})
    rc.ingest_overrides(ctx)
    skip = rc.newest_entry(ctx.ledger, "C5", set(ctx.view["entries"]))
    assert skip["chosen"] == "SKIP" and skip["status"] == "skipped"
    assert skip["evidence"]["user"] is True
    assert skip["base_recipe_id"] == rc.compose_recipe(ctx.ledger, cands, CODE, upto="C5")["recipe_id"]
    # and now a decision exists, so a second `skip` is refused rather than silently ignored
    ctx.ledger["entries"].append(entry("D09", "C6", "REJECT", "decided"))
    ctx.write_ledger()
    rc.lc.append_override(ctx.dir, {"candidate": "C6", "choice": "skip", "why": "too late"})
    rc.ingest_overrides(ctx)
    assert any("REFUSED" in line for line in notify_lines(ctx))
    assert rc.newest_entry(ctx.ledger, "C6", set(ctx.view["entries"]))["chosen"] == "REJECT"


def test_a_measurement_may_not_be_confirmed_or_overruled(rc, cands, tmp_path):
    led, st = ledger(), state()
    led["entries"].append({"id": "D00", "kind": "resolution", "candidate": None, "chosen": "RECORD",
                           "status": "final", "options": ["RECORD"], "history": []})
    led["entries"].append({"id": "D03", "kind": "measurement", "candidate": None, "chosen": "RECORD",
                           "status": "final", "options": ["RECORD"], "history": []})
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    for did in ("D00", "D03", "D02"):
        rc.lc.append_override(ctx.dir, {"decision": did, "action": "overrule", "choice": "REJECT",
                                        "why": "no"})
    rc.ingest_overrides(ctx)
    for did in ("D00", "D03", "D02"):
        assert rc.entry_by_id(ctx.ledger, did)["status"] == "final"
    refused = [line for line in notify_lines(ctx) if "REFUSED" in line]
    assert len(refused) == 3


def test_an_override_is_applied_once_however_often_the_file_is_read(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm", "choice": "ADOPT",
                                    "why": "yes"})
    assert rc.ingest_overrides(ctx) is True
    assert rc.ingest_overrides(ctx) is False
    assert len(rc.entry_by_id(ctx.ledger, "D05")["history"]) == 1


def test_a_torn_override_line_is_skipped_not_guessed(rc, cands, tmp_path):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "skip", "why": "ok"})
    with (ctx.dir / rc.lc.OVERRIDES_NAME).open("a") as fh:
        fh.write('{"uuid": "half-written", "candidate": "C6"')
    assert [r["choice"] for r in rc.lc.read_overrides(ctx.dir)] == ["skip"]


def test_an_append_after_a_torn_line_is_readable_whole(rc, cands, tmp_path):
    """FIX_QUEUE 4: the repeat has to survive the crash that lost the first answer.

    A torn last line ends without a newline, so an append CONCATENATED onto it -- one unparseable
    line where there should be a fragment and a record, and the user's repeated answer was lost
    exactly because the first one had been. The leading newline closes the fragment; the fragment
    is then skipped and the repeat reads whole.
    """
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    path = ctx.dir / rc.lc.OVERRIDES_NAME
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "skip", "why": "first"})
    with path.open("a") as fh:
        fh.write('{"uuid": "half-written", "candidate": "C6"')       # the crash
    uid = rc.lc.append_override(ctx.dir, {"candidate": "C6", "choice": "run", "why": "repeat"})
    got = rc.lc.read_overrides(ctx.dir)
    assert [(r["candidate"], r["choice"]) for r in got] == [("C5", "skip"), ("C6", "run")]
    assert got[-1]["uuid"] == uid
    assert path.read_text().endswith("\n")
    assert "half-written" not in json.dumps(got)
    # and appending again to the now well-formed file adds no blank line
    rc.lc.append_override(ctx.dir, {"candidate": "C7", "choice": "skip", "why": "third"})
    assert "\n\n" not in path.read_text().replace('"candidate": "C6"\n\n', '"candidate": "C6"\n')
    assert len(rc.lc.read_overrides(ctx.dir)) == 3


def test_resume_milestone_accept_freezes_every_unfrozen_block_adoption(rc, cands, tmp_path):
    """``--accept`` after a (i) FAIL: ``decided`` AND pair-resolved ``final`` adoptions are frozen."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D05", "C2", "ADOPT", "final", pooled_confidence="clear",
                                experiment="X03", recipe=r_c2["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                            rid(rc, cands, led, "C2"), base_rank("C2"))
    r_c3a = rc._recipe_with(led, cands, CODE, "C3a", set())
    led["entries"].append(entry("D06", "C3a", "ADOPT", "decided", experiment="X04",
                                recipe=r_c3a["recipe_id"]))
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C3a", "3", r_c3a["recipe_id"],
                                            r_c2["recipe_id"], base_rank("C3a"))
    # C3b ran and was rejected (a SKIP written under "C3a is a REJECT" would be squashed by the
    # derived view here, because the referent has flipped -- which is rule (b) working)
    led["entries"].append(entry("D07", "C3b", "REJECT", "decided", experiment="X05"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {"stage": "i"}, "recipe_id": r_c3a["recipe_id"],
                           "eligibility": 1})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"milestone": "M1", "accept": True, "why": "the recipe stands"})
    rc.ingest_overrides(ctx)
    for did in ("D05", "D06"):
        e = rc.entry_by_id(ctx.ledger, did)
        assert e["status"] == "confirmed" and e["frozen"] is True
        assert e["override"]["why"] == "M1 accept"
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert m1["status"] == "await" and m1["sub_status"] == "accepted_i"
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["label"] == "M1.promo s3"


def test_a_second_milestone_fail_after_an_accept_has_nothing_left_to_drop(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "ADOPT", "confirmed", frozen=True, experiment="X03",
                                override={"by": "user", "choice": "ADOPT", "why": "M1 accept"}))
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    entry_m1 = {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": None, "status": "open",
                "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {},
                "recipe_id": rc.milestone_recipe(cands, led, "M1", CODE)["recipe_id"],
                "eligibility": 1}
    led["entries"].append(entry_m1)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc._milestone_fail(ctx, "M1", entry_m1, ["M1 (ii)"], ["slow", "light"])
    assert entry_m1["chosen"] == "FAIL" and entry_m1["status"] == "await"
    assert entry_m1["evidence"]["stage"] == "ii"
    assert ctx.state["phase"] == "await_user:M1"
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "confirmed", "nothing was dropped"
    line = [line for line in notify_lines(ctx) if "MILESTONE" in line][-1]
    assert "nothing unfrozen to drop" in line


def test_resume_milestone_drop_is_refused_unless_it_names_an_adoption(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "REJECT", "decided", experiment="X03"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {"stage": "ii"},
                           "recipe_id": rc.milestone_recipe(cands, led, "M1", CODE)["recipe_id"],
                           "eligibility": 1})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"milestone": "M1", "drop": "C2", "why": "drop it"})
    rc.ingest_overrides(ctx)
    assert any("REFUSED" in line and "ADOPT" in line for line in notify_lines(ctx))
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "decided"


def test_resume_milestone_drop_overrules_the_adoption_and_requeues_the_milestone(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "ADOPT", "decided", experiment="X03"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {"stage": "ii"},
                           "recipe_id": rc.milestone_recipe(cands, led, "M1", CODE)["recipe_id"],
                           "eligibility": 1})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"milestone": "M1", "drop": "C2", "why": "drop it"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "overruled" and d05["override"]["choice"] == "REJECT"
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert m1["status"] == "squashed" and m1["eligibility"] == 2


def test_one_automatic_milestone_rerun_per_eligibility(rc, cands):
    led = ledger(entry("D05", "C2", "ADOPT", "overruled",
                       override={"by": "milestone", "choice": "REJECT", "why": "M1 batched pair"}))
    assert rc.milestone_rerun_used(led, "M1", cands) is True
    led2 = ledger(entry("D05", "C2", "ADOPT", "decided"))
    assert rc.milestone_rerun_used(led2, "M1", cands) is False
    # a USER override after the milestone's restarts the allowance
    led3 = ledger(entry("D05", "C2", "ADOPT", "overruled",
                        override={"by": "milestone", "choice": "REJECT", "why": "M1 batched pair"}),
                  entry("D06", "C3a", "ADOPT", "overruled",
                        override={"by": "user", "choice": "REJECT", "why": "user"}))
    assert rc.milestone_rerun_used(led3, "M1", cands) is False


# ================================================================== the milestone loop
def test_milestone_i_records_watch_when_no_adoption_is_pending(rc, cands, tmp_path):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "milestone_i"
    rc.milestone_i(ctx, item)
    m1 = rc.newest_milestone_entry(ctx.ledger, "M1")
    assert m1["status"] == "open" and m1["chosen"] is None
    assert "WATCH" in m1["evidence"]["pair"]
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["label"] == "M1.promo s3"


def test_milestone_i_promotes_the_blocks_decided_adoptions_when_the_pair_agrees(rc, cands, tmp_path):
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X03",
                                base=b0["recipe_id"], recipe=r_c2["recipe_id"]))
    better = screen(closed=0.72, prog=0.11)
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                            b0["recipe_id"], base_rank("C2"), screen_=better)
    st["ladder"]["experiments"]["X05"] = exp("X05", "batched_s2", "M1", "2", r_c2["recipe_id"],
                                            r_c2["recipe_id"], {"upto": "C3b", "inclusive": True},
                                            screen_=better)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "milestone_i"
    rc.milestone_i(ctx, item)
    assert rc.entry_by_id(ctx.ledger, "D_C2")["status"] == "final"
    assert rc.entry_by_id(ctx.ledger, "D_C2")["pooled_confidence"] == "clear"
    m1 = rc.newest_milestone_entry(ctx.ledger, "M1")
    assert m1["status"] == "open"
    assert any("AGREE" in line for line in notify_lines(ctx))


def test_milestone_i_contradiction_drops_the_last_unfrozen_adoption(rc, cands, tmp_path):
    """The plan's named scenario, end to end: "``M1.s2`` -> (i) contradiction -> C2 overruled
    (milestone), ``M1.s2`` + C3a/C3b squashed -> C3a/C3b re-screened -> (``M1.s2`` only if the
    resulting ``R_1`` has no ok s2) -> (i) -> ``M1.promo``".

    The drop is only the first half. What it costs is the SECOND half -- the milestone's own s2, the
    block members screened after the dropped adoption, and the milestone entry itself -- so those
    are asserted here too (3.9-7.8 GPU-hours of re-screening rides on them).
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    for cid in ("C1", "C1b", "C1c"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X03",
                                base=b0["recipe_id"], recipe=r_c2["recipe_id"]))
    led["entries"].append(entry("D_C3a", "C3a", "ADOPT", "decided", experiment="X04",
                                base=r_c2["recipe_id"]))
    # a block member screened AFTER the adoption that is about to be dropped
    r_c3a = rid(rc, cands, led, "C3b")
    led["entries"].append(entry("D_C3b", "C3b", "REJECT", "decided", experiment="X06",
                                base=r_c3a))
    worse = screen(ratio=0.30)
    recipe_id = rid(rc, cands, led, "C3b", inclusive=True)
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                            b0["recipe_id"], base_rank("C2"), screen_=worse)
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C3a", "3", recipe_id,
                                            r_c2["recipe_id"], base_rank("C3a"), screen_=worse)
    st["ladder"]["experiments"]["X05"] = exp("X05", "batched_s2", "M1", "2", recipe_id, recipe_id,
                                            {"upto": "C3b", "inclusive": True}, screen_=worse)
    st["ladder"]["experiments"]["X06"] = exp("X06", "candidate", "C3b", "3", recipe_id, r_c3a,
                                            base_rank("C3b"), screen_=worse)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "milestone_i"
    rc.milestone_i(ctx, item)
    dropped = rc.entry_by_id(ctx.ledger, "D_C3a")
    assert dropped["status"] == "overruled" and dropped["override"]["by"] == "milestone"
    assert rc.entry_by_id(ctx.ledger, "D_C2")["status"] == "decided", "only the LAST one is dropped"
    assert any("CONTRADICTION" in line for line in notify_lines(ctx))
    assert rc.milestone_rerun_used(ctx.ledger, "M1", cands) is True
    # ... and everything built on the dropped adoption goes with it
    assert "X05" in ctx.lad["squashed"], "M<k>.s2 was screened on the recipe that just changed"
    assert "X06" in ctx.lad["squashed"], "and so was the block member screened after C3a"
    assert rc.entry_by_id(ctx.ledger, "D_C3b")["status"] == "squashed"
    assert rc.newest_milestone_entry(ctx.ledger, "M1", set(ctx.view["entries"])) is None
    holds, why = rc.after_holds(cands, ctx.ledger, "M1", set(ctx.view["entries"]))
    assert holds is False and "C3b" in why, "the milestone is not eligible until the member lands"
    # the members re-decide at their ranks; C3b's predicate is now false, so it is a SKIP
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["candidate"] == "C3b"
    assert nxt["kind"] in ("skip", "candidate")


def test_a_second_contradiction_parks_for_the_user(rc, cands, tmp_path):
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X03",
                                base=b0["recipe_id"], recipe=r_c2["recipe_id"]))
    led["entries"].append(entry("D_old", "C4", "ADOPT", "overruled",
                                override={"by": "milestone", "choice": "REJECT", "why": "M1 pair"}))
    worse = screen(ratio=0.30)
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                            b0["recipe_id"], base_rank("C2"), screen_=worse)
    st["ladder"]["experiments"]["X05"] = exp("X05", "batched_s2", "M1", "2", r_c2["recipe_id"],
                                            r_c2["recipe_id"], {"upto": "C3b", "inclusive": True},
                                            screen_=worse)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    rc.milestone_i(ctx, item)
    m1 = rc.newest_milestone_entry(ctx.ledger, "M1")
    assert m1["chosen"] == "FAIL" and m1["status"] == "await"
    assert ctx.state["phase"] == "await_user:M1"
    assert rc.entry_by_id(ctx.ledger, "D_C2")["status"] == "decided"


def test_a_milestone_row_is_worse_on_a_conditional_survival_and_missing_is_not_worse(rc, cands, tmp_path):
    """The ring row: a WORSE ``turn_survival_r90`` counts; a zero denominator makes both MISSING."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    worse = rc._row_worse(ctx, "ring", screen(surv_r=0.50), screen(), {})
    assert worse["worse"] == ["guard turn_survival_r90"]
    gone = screen(surv_l=None, surv_r=None)
    missing = rc._row_worse(ctx, "ring", gone, screen(), {})
    assert missing["worse"] == []
    assert set(missing["compare"]["guards"]["missing"]) == {"turn_survival_l90", "turn_survival_r90"}


def test_ring_completion_is_watch_only_while_the_bootstrap_does_not_resolve_it(rc, cands, tmp_path):
    """Decision 6's clause is CONDITIONAL: "ring completion WATCH at milestones UNLESS the
    bootstrap resolves it". Part C (ii)'s "ring completion is WATCH (decision 6)" cites that
    sentence rather than restating it, so the escape clause governs -- otherwise a promotion that
    holds up on slow/light but collapses ring completion reaches the "< 2 rows WORSE" PASS on the
    other two rows alone. The flat row always scores it."""
    unresolved = resol()
    unresolved["by_row"] = {"ring__omni": {"metrics": {m: unresolved["metrics"][m]
                                                       for m in GUARDS if m != "completion"},
                                           "blocks": unresolved["blocks"]}}
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state(), resolution=unresolved)
    ring = rc._row_worse(ctx, "ring", screen(completion=0.10), screen(), {})
    assert ring["worse"] == []
    assert ring["ring_completion"].startswith("WATCH")
    slow = rc._row_worse(ctx, "slow", screen(completion=0.10), screen(), {})
    assert slow["worse"] == ["guard completion"]

    resolved = resol()
    resolved["by_row"] = {"ring__omni": {"metrics": dict(resolved["metrics"]),
                                        "blocks": resolved["blocks"]}}
    ctx2 = make_ctx(rc, cands, tmp_path, led=ledger(), st=state(), resolution=resolved, name="c2")
    scored = rc._row_worse(ctx2, "ring", screen(completion=0.10), screen(), {})
    assert scored["worse"] == ["guard completion"]
    assert scored["ring_completion"].startswith("scored")
    # and a collapse INSIDE the measured half-width is still not WORSE
    inside = rc._row_worse(ctx2, "ring", screen(completion=0.45), screen(), {})
    assert inside["worse"] == []


def test_the_shipped_d00_resolves_ring_completion_so_the_guard_is_scored_there(rc, cands, tmp_path):
    """The delivered ``resolution.json`` carries ``by_row[ring__omni].metrics.completion``, so on
    THIS campaign the conditional lands on "scored" -- which is what the CHARTER must say."""
    path = LADDER / "resolution.json"
    if not path.exists():                                     # pragma: no cover - record not built
        pytest.skip("resolution.json has not been computed yet")
    record = json.loads(path.read_text())
    row = ((record.get("by_row") or {}).get("ring__omni") or {}).get("metrics") or {}
    assert (row.get("completion") or {}).get("R_eval"), "D00 must resolve ring completion"
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state(), resolution=record)
    verdict = rc._row_worse(ctx, "ring", screen(), screen(), {})
    assert verdict["ring_completion"].startswith("scored")


def test_the_row_resolution_prefers_the_per_row_table(rc, cands, tmp_path):
    resolution = resol()
    resolution["by_row"] = {"ring__omni": {"metrics": {"completion": {"R_eval": 9.0, "level": 0.99}},
                                          "blocks": resolution["blocks"]}}
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state(), resolution=resolution)
    assert rc._row_resolution(ctx.resolution, "slow")["__source__"].startswith("global")
    per_row = rc._row_resolution(ctx.resolution, "ring")
    assert per_row["__source__"] == "by_row[ring__omni]"
    assert per_row["metrics"]["completion"]["R_eval"] == 9.0


# ================================================================== running an experiment
TRAIN_LOG = "\n".join(
    [f"Metrics/base_parkour/track_ratio_mean: 0.7{i % 10}" for i in range(40)]
    + [f"Metrics/base_parkour/track_ratio_move_mean: 0.6{i % 10}" for i in range(40)]
    + ["Mean reward: 90.0", "Mean episode length: 1400", "Mean value_function: 3.0",
       "Metrics/base_parkour/reach_obst_frac: 0.1", "Episode_Reward/reward_clock_schedule: 0.4",
       "Loss/symmetry: 0.01", "Episode_Reward/reward_yaw_progress: 0.1",
       "Metrics/base_velocity/yaw_progress: 0.08", "Episode_Termination/crab_failure: 0.1"]) + "\n"


def _dump_for(rc, overlay: dict) -> str:
    """A ``params/env.yaml`` that AGREES with ``overlay`` -- the shape ``overlay_checks`` reads."""
    pool = overlay.get("KRABBY_OMNI_CMD_POOL")
    slots = rc.parse_slots_str(overlay.get("KRABBY_OMNI_SLOTS", "0.15:0.55:0.1:0.2"))
    facing = rc.parse_slots_str(overlay.get("KRABBY_OMNI_FACING", "0.2:0.6"))
    lines = ["commands:", "  base_velocity:", "    slot_probs: !!python/tuple"]
    lines += [f"    - {v}" for v in slots]
    lines += ["    speed_ellipse: !!python/tuple", "    - 0.35", "    - 0.25", "    - 0.25",
              "    facing_delta: !!python/tuple", f"    - {facing[0]}", f"    - {facing[1]}",
              "    travel_dir_range: !!python/tuple", "    - -3.14", "    - 3.14",
              f"    ang_vel_max: {overlay.get('KRABBY_OMNI_ANG_VEL_MAX', '0.3')}",
              "    dir_band: null",
              f"    cmd_pool: {pool if pool else 'null'}",
              "    clips:",
              f"      ang_vel_clip: {overlay.get('KRABBY_OMNI_ANG_VEL_CLIP', '0.2')}",
              f"omni_clock_turn_radius: {overlay.get('KRABBY_OMNI_CLOCK_TURN_R', '1.2')}",
              "observations:", "  policy:", "    extreme_parkour_observations:", "      params:",
              f"        facing_obs_encoding: {overlay.get('KRABBY_OMNI_FACING_OBS_ENCODING', 'raw')}",
              "        facing_obs_clip: 0.6",
              "rewards:"]
    for term, key, default in (("reward_tracking_yaw", "KRABBY_YAW_W", "0.2"),
                               ("reward_yaw_progress", "KRABBY_YAW_PROGRESS_W", "1.0"),
                               ("reward_facing_turn", "KRABBY_OMNI_FACING_TURN_W", "0.0"),
                               ("reward_forward_progress_along_command", "KRABBY_OMNI_PROGRESS_W", "0.6")):
        lines += [f"  {term}:", "    func: x:y", f"    weight: {overlay.get(key, default)}"]
    sigma = overlay.get("KRABBY_OMNI_TRACK_FACING_SIGMA")
    lines += ["  track_lin_vel_xy_exp:",
              "    func: x:track_lin_vel_xy_exp_facing_coupled" if sigma else "    func: x:track_lin_vel_xy_exp",
              "    params:"] + ([f"      facing_sigma: {sigma}"] if sigma else ["      std: 0.2"])
    lines += ["  track_ang_vel_z_exp:",
              "    func: x:track_ang_vel_z_exp_turn_gated" if overlay.get("KRABBY_OMNI_ANG_TURN_ONLY") == "1"
              else "    func: x:track_ang_vel_z_exp",
              "    params:", f"      std: {overlay.get('KRABBY_OMNI_ANG_VEL_STD', '0.5')}",
              "events:", "  rsi_reference_reset:", "    func: x:y", "    params:",
              "      bank_path: rsi_bank_omni.npz",
              f"      source: {overlay.get('KRABBY_OMNI_RSI_SOURCE', 'all')}",
              "      yaw_range: !!python/tuple", "      - -3.14", "      - 3.14",
              "OmniParkourEvent: true", "reset_root_state_ring: true", "CrabHexOmniCommand: true",
              "CrabHexOmniParkourObservations: true", "omni_ring_sparse: true",
              "usd_path: /x/assets/crab.usda", "episode_length_s: 40.0"]
    return "\n".join(lines) + "\n"


def install_fakes(rc, monkeypatch, tmp_path, *, status="ok", eval_summary=None, eval_none=0,
                  dump_breaks=None, interrupt=None):
    """Stand in for the training and eval children: no subprocess, the same call contract."""
    calls = {"train": [], "eval": []}
    runs = Path(tmp_path) / "logroot"
    monkeypatch.setattr(rc.rp, "LOG_ROOT", runs)
    monkeypatch.setattr(rc, "INFRA_WAIT_S", 0)
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    monkeypatch.setattr(rc.rp, "_interruptible_sleep", lambda *a, **k: False)

    def train_retry(c, spec, tag, iters, resume, soundness, early_abort=False, *, extra_env=None,
                    interrupt_check=None, run_name=None, abort_track_ratio_min=None):
        calls["train"].append({"tag": tag, "iters": iters, "resume": resume, "soundness": soundness,
                               "early_abort": early_abort, "env": dict(extra_env or {}),
                               "run_name": run_name, "abort_floor": abort_track_ratio_min,
                               "preset": spec.name})
        # the real seam records this before it launches; the heartbeat reads it
        c.st["current"] = {"tag": tag, "phase": spec.name, "iters": iters, "kind": spec.kind,
                           "log": str(c.logs / f"{tag}_train.log"), "resume": resume,
                           "extra_env": dict(extra_env or {})}
        c.save()
        if interrupt is not None and interrupt_check is not None and interrupt_check():
            return None, "interrupted", c.logs / f"{tag}_train.log"
        run_dir = runs / spec.experiment / f"2026-09-19_00-00-00_{run_name or tag}"
        (run_dir / "params").mkdir(parents=True, exist_ok=True)
        overlay = dict(extra_env or {})
        if dump_breaks:
            overlay = dict(overlay, **dump_breaks)
        (run_dir / "params" / "env.yaml").write_text(_dump_for(rc, overlay))
        (run_dir / "params" / "agent.yaml").write_text(f"experiment_name: {spec.experiment}\n")
        ckpt = run_dir / f"model_{rc.rp._ckpt_iter(resume) + iters - 1}.pt"
        ckpt.write_text("x")
        log = c.logs / f"{tag}_train.log"
        log.write_text(TRAIN_LOG)
        return str(ckpt), status, log

    def _eval(c, spec, ckpt, row, tag, carry_overlay=None, max_attempts=3):
        calls["eval"].append({"row": row, "tag": tag, "carry": dict(carry_overlay or {}),
                              "max_attempts": max_attempts})
        if len(calls["eval"]) <= eval_none:
            return None
        d = Path(tmp_path) / "evals" / f"{tag}_{row}"
        d.mkdir(parents=True, exist_ok=True)
        env = rc.rp.eval_env_for(spec, c.plant, keep=rc.rp.CANARY_KEYS, keep_source=rc.ph.FORMATION,
                                 carry_overlay=carry_overlay)
        (d / "run_meta.json").write_text(json.dumps({"env_vars": {k: v for k, v in env.items()
                                                                  if k in rc.rp.EVAL_CARRY}}))
        out = dict(eval_summary if eval_summary is not None else screen())
        out.update(run_dir=str(d), eval_status="ok", eval_rc=0)
        return out

    monkeypatch.setattr(rc.rp, "train_retry", train_retry)
    monkeypatch.setattr(rc.rp, "eval_morph",
                        lambda c, spec, ckpt, row, tag, *, carry_overlay=None, interrupt_check=None,
                        max_attempts=3: _eval(c, spec, ckpt, row, tag, carry_overlay, max_attempts))
    monkeypatch.setattr(rc.rp, "eval_obst",
                        lambda c, spec, ckpt, tag, *, carry_overlay=None, interrupt_check=None,
                        max_attempts=3: _eval(c, spec, ckpt, "ring", tag, carry_overlay, max_attempts))
    return calls


def candidate_item(rc, cands, ctx, cid="C2"):
    led = ctx.ledger
    base = rc.compose_recipe(led, cands, ctx.code, upto=cid, squashed=set(ctx.view["entries"]))
    recipe = rc._recipe_with(led, cands, ctx.code, cid, set(ctx.view["entries"]))
    return rc._item("candidate", cid=cid, seed=ctx.seed_screen, recipe=recipe, base_recipe=base,
                    stages=[dict(s) for s in recipe["stages"]], needs_s3=base,
                    why="test", label=f"{cid} s3")


def test_one_experiment_trains_evaluates_records_and_audits_every_stage(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    item = candidate_item(rc, cands, ctx, "C2")
    header = rc.run_experiment(ctx, item)
    assert header["status"] == "ok"
    assert header["recs"] == ["X01:1c"] or header["recs"][-1].endswith(":1c")
    assert calls["train"][0]["preset"] == "1c"
    assert calls["train"][0]["env"] == {"KRABBY_YAW_PROGRESS_W": "3.0"}
    assert calls["train"][0]["run_name"] == calls["train"][0]["tag"]
    assert calls["train"][0]["tag"].startswith("seed3_") and "_A15pB" in calls["train"][0]["tag"]
    assert [c["row"] for c in calls["eval"]] == ["slow"]
    assert calls["eval"][0]["max_attempts"] == 1, "the driver owns the eval retry cadence"
    assert calls["eval"][0]["carry"] == {"KRABBY_YAW_PROGRESS_W": "3.0"}
    rec = ctx.c.recs[header["recs"][-1]]
    assert rec["overlay"] == {"KRABBY_YAW_PROGRESS_W": "3.0"}
    assert rec["env"] == {"KRABBY_OMNI_PHASE": "1c", "KRABBY_PLANT": "A15+B"}
    assert set(rec["evals"]) == {"slow"}
    assert rec["evals"]["slow"]["run_dir"]
    assert all(ok for _n, ok in rec["overlay_checks"]), rec["overlay_checks"]
    assert not any(name.startswith(("dir_band matches stage", "RSI source matches stage"))
                   for name, _ok in rec["checks"])
    assert header["screen"]["track_ratio_tail"] == pytest.approx(rec["track_ratio_tail"])
    assert header["screen"]["track_ratio_move_tail"] is not None
    heads = json.loads((ctx.dir / "heads.json").read_text())
    assert heads["seed3"][header["recs"][-1]] == header["ckpt"]
    assert ">>> ENTRY" in (ctx.dir / "REPORT.md").read_text()


def test_the_curriculum_runs_every_stage_in_order_and_resumes_the_previous_one(rc, cands, tmp_path,
                                                                              monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    header = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1"))
    assert header["status"] == "ok"
    assert [c["iters"] for c in calls["train"]] == [1000, 1000, 1000, 1500, 1500, 2000]
    assert calls["train"][0]["resume"] is None
    for prev, cur in zip(calls["train"], calls["train"][1:]):
        assert cur["resume"] is not None and prev["tag"] in cur["resume"]
    assert [c["row"] for c in calls["eval"]] == ["curriculum", "curriculum", "curriculum",
                                                "slow", "curriculum"]
    assert header["recs"] == [f"{header['xid']}:{s}" for s in ("c1", "c2", "c3", "c4", "c5", "c6")]
    assert header["screen"]["stage"] == "c6"


@pytest.mark.parametrize("kind,early,sound", [("candidate", True, True), ("baseline", False, True),
                                              ("twin", False, True), ("seed2", False, True),
                                              ("batched_s2", False, True)])
def test_the_dead_walker_abort_is_armed_for_candidates_only(rc, cands, tmp_path, monkeypatch, kind,
                                                            early, sound):
    """Decision 4's arming list: only a ``candidate`` may be aborted early; everything else runs whole."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    ctx.lad["d03"] = 0.42
    calls = install_fakes(rc, monkeypatch, tmp_path)
    item = candidate_item(rc, cands, ctx, "C2")
    item["kind"] = kind
    rc.run_experiment(ctx, item)
    assert calls["train"][0]["early_abort"] is early
    assert calls["train"][0]["soundness"] is sound
    assert calls["train"][0]["abort_floor"] == (0.42 if early else None)


def test_a_short_from_scratch_stage_is_neither_aborted_nor_sounded(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1"))
    first = calls["train"][0]
    assert first["iters"] == 1000 and first["early_abort"] is False and first["soundness"] is False


def test_an_eval_that_did_not_carry_the_overlay_halts_the_campaign(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)
    monkeypatch.setattr(rc.rp, "eval_morph",
                        lambda c, spec, ckpt, row, tag, **kw: {"run_dir": str(tmp_path), "eval_status": "ok"})
    (tmp_path / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_OMNI_ANG_VEL_MAX": "0.3"}}))
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C3a"))
    assert "halt:" in ctx.state["phase"] and "eval_mismatch" in ctx.state["phase"]
    assert any("HALT" in line and "eval_mismatch" in line for line in notify_lines(ctx))


def test_an_overlay_that_did_not_reach_the_config_halts_the_campaign(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path, dump_breaks={"KRABBY_YAW_PROGRESS_W": "1.0"})
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert "config_mismatch" in ctx.state["phase"]


def test_an_eval_that_produces_nothing_retries_once_then_halts_eval_infra(rc, cands, tmp_path,
                                                                         monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path, eval_none=99)
    monkeypatch.setattr(rc, "infra_signature", lambda *a, **k: (None, []))
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert len(calls["eval"]) == 2, "one attempt, the infra detector, then ONE retry"
    assert "eval_infra" in ctx.state["phase"]


def test_an_oom_signature_stops_retrying_at_once(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path, eval_none=99)
    monkeypatch.setattr(rc, "infra_signature", lambda *a, **k: ("cgroup oom_kill 0 -> 1", []))
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert len(calls["eval"]) == 1, "an OOM signature must not be retried"
    assert any("INFRA" in line and "oom_kill" in line for line in notify_lines(ctx))
    assert "eval_infra" in ctx.state["phase"]


def test_a_floor_failure_rejects_a_candidate_and_only_watches_a_baseline(rc, cands, tmp_path,
                                                                        monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    low = screen()
    low["survival_by_hold"] = {"fwd": 0.10, "back": 0.10, "move_turn_r": 0.10}
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=low)
    item = candidate_item(rc, cands, ctx, "C1")
    header = rc.run_experiment(ctx, item)
    assert header["status"] == "rejected_by_floor"
    assert header["floor_stage"] == "c1"
    assert "survival through fwd" in header["floor"]
    rc._dispatch_floor_reject(ctx, item, header)
    d = rc.newest_entry(ctx.ledger, "C1", set(ctx.view["entries"]))
    assert (d["chosen"], d["confidence"], d["status"]) == ("REJECT", "clear", "decided")
    assert d["floor_stage"] == "c1"
    # the same floor on a NON-candidate kind is WATCH: the run continues to its screen
    ctx2 = make_ctx(rc, cands, tmp_path, led=led, st=state(), name="camp2")
    item2 = candidate_item(rc, cands, ctx2, "C1")
    item2["kind"] = "twin"
    header2 = rc.run_experiment(ctx2, item2)
    assert header2["status"] == "ok"
    assert header2["floor_watch"] and "survival through fwd" in header2["floor_watch"][0]
    assert any("FLOOR tripped on a twin run" in line for line in notify_lines(ctx2))


def test_retention_seeds_the_trace_on_the_first_run_and_reports_a_drop_on_the_second(rc, cands,
                                                                                    tmp_path,
                                                                                    monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    good = screen()
    good["survival_by_hold"] = {h: 0.95 for h in ("stand", "fwd", "lat_l", "lat_r", "back")}
    good["ratio_by_hold"] = {h: 0.60 for h in ("fwd", "lat_l", "lat_r", "back")}
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=good)
    rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1"))
    trace = ctx.lad["stage_trace"]["curriculum"]
    assert trace["c1"]["survival"]["fwd"] == 0.95
    lines = [line for line in notify_lines(ctx) if "RETENTION" in line]
    assert any("seeds it" in line for line in lines)
    bad = copy.deepcopy(good)
    bad["survival_by_hold"]["fwd"] = 0.50
    bad["ratio_by_hold"]["fwd"] = 0.20
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=bad)
    rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1b"))
    dropped = [line for line in notify_lines(ctx) if "DROPPED" in line]
    assert any("survival 0.950 -> 0.500" in line for line in dropped)
    assert any("speed_ratio 0.600 -> 0.200" in line for line in dropped)
    assert all("RETENTION" in line for line in dropped), "retention is WATCH, never a verdict"


def test_an_interrupted_experiment_is_resumable_after_a_stop_and_abandoned_after_a_squash(
        rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)
    monkeypatch.setattr(rc, "interrupt_reason", lambda ctx_, xid: "stop")
    header = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1"))
    assert header["status"] == "running" and header["interrupt"] == "stop"
    monkeypatch.setattr(rc, "interrupt_reason", lambda ctx_, xid: "squashed: base recipe changed")
    header2 = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert header2["status"] == "interrupted"


def test_a_stopped_stage_resumes_the_remainder_and_keeps_the_original_record(rc, cands, tmp_path,
                                                                            monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    item = candidate_item(rc, cands, ctx, "C1")
    xid = "X03"
    recipe = item["recipe"]
    header = {"xid": xid, "kind": "candidate", "candidate": "C1", "seed": "3", "label": "C1 s3",
              "chain": recipe["chain"], "base_recipe_id": item["base_recipe"]["recipe_id"],
              "recipe_id": recipe["recipe_id"], "base_rank": base_rank("C1"), "code_sha": ctx.code,
              "overlay_by_stage": {str(s["stage"]): rc.stage_env(recipe, str(s["stage"]))
                                   for s in recipe["stages"]},
              "started": "2026-09-19T00:00:00", "status": "running", "recs": [], "eval_dirs": [],
              "stages": [str(s["stage"]) for s in recipe["stages"]],
              "inflight": {"stage": "c1", "iters": 1000, "tag": "seed3_X03_c1_A15pB", "resume": None}}
    ctx.lad["experiments"][xid] = header
    run_dir = Path(tmp_path) / "logroot" / rc.ph.PHASES["1a"].experiment / "2026_seed3_X03_c1_A15pB"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_600.pt").write_text("x")
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["remaining"] == 400
    assert header["inflight"]["ckpt"].endswith("model_600.pt")
    resumed = rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    assert resumed["status"] == "ok"
    first = calls["train"][0]
    assert first["iters"] == 400 and first["tag"].endswith("_c")
    assert first["resume"].endswith("model_600.pt")
    rec = ctx.c.recs[f"{xid}:c1"]
    assert rec["iters"] == 1000, "the record keeps the ORIGINAL window length"
    assert rec["resume"] is None, "and the ORIGINAL resume point"


def test_a_stage_that_finished_before_the_stop_skips_training_and_goes_to_its_eval(rc, cands,
                                                                                  tmp_path,
                                                                                  monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    item = candidate_item(rc, cands, ctx, "C1")
    recipe = item["recipe"]
    run_dir = Path(tmp_path) / "logroot" / rc.ph.PHASES["1a"].experiment / "2026_seed3_X03_c1_A15pB"
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    (run_dir / "params" / "env.yaml").write_text(_dump_for(rc, rc.stage_env(recipe, "c1")))
    (run_dir / "params" / "agent.yaml").write_text("experiment_name: x\n")
    (run_dir / "model_999.pt").write_text("x")
    header = {"xid": "X03", "kind": "candidate", "candidate": "C1", "seed": "3", "label": "C1 s3",
              "chain": recipe["chain"], "base_recipe_id": item["base_recipe"]["recipe_id"],
              "recipe_id": recipe["recipe_id"], "base_rank": base_rank("C1"), "code_sha": ctx.code,
              "overlay_by_stage": {str(s["stage"]): rc.stage_env(recipe, str(s["stage"]))
                                   for s in recipe["stages"]},
              "started": "2026-09-19T00:00:00", "status": "running", "recs": [], "eval_dirs": [],
              "stages": [str(s["stage"]) for s in recipe["stages"]],
              "inflight": {"stage": "c1", "iters": 1000, "tag": "seed3_X03_c1_A15pB", "resume": None}}
    ctx.lad["experiments"]["X03"] = header
    (ctx.c.logs / "seed3_X03_c1_A15pB_train.log").write_text(TRAIN_LOG)
    rc.prepare_resume(ctx, header)
    # target_final = 999 and the newest checkpoint IS 999, so the formula returns 1 == "finished"
    assert header["inflight"]["remaining"] == 1
    rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    assert [c["iters"] for c in calls["train"]] == [1000, 1000, 1500, 1500, 2000], \
        "c1 was not retrained; the remaining five stages were"
    assert calls["eval"][0]["row"] == "curriculum" and "_c1_" in calls["eval"][0]["tag"]


def test_a_stage_with_no_checkpoint_is_rerun_from_its_recorded_resume_point(rc, cands, tmp_path,
                                                                           monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)
    header = {"xid": "X03", "chain": "scratch_1c", "kind": "candidate", "candidate": "C2",
              "seed": "3", "inflight": {"stage": "1c", "iters": 8000, "tag": "nope", "resume": None},
              "stages": ["1c"], "overlay_by_stage": {"1c": {}}, "started": "x", "status": "running"}
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["remaining"] is None and header["inflight"]["ckpt"] is None


def test_the_smoke_hook_runs_one_window_and_writes_no_ledger_entry(rc, cands, tmp_path, monkeypatch):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    calls = install_fakes(rc, monkeypatch, tmp_path)
    item = rc.smoke_item(ctx, "1a:20")
    header = rc.run_experiment(ctx, item)
    assert header["status"] == "ok" and header["kind"] == "smoke"
    assert calls["train"][0]["iters"] == 20 and calls["train"][0]["preset"] == "1a"
    assert calls["eval"] == []
    rc.dispatch(ctx, item, header)
    assert rc.entries(ctx.ledger) == []
    assert rc._smoke_done(ctx) is True
    assert rc.squash_view(ctx.ledger, cands, ctx.state, code=ctx.code)["experiments"] == {}


def test_isaac_is_waited_for_before_every_train_and_every_eval_launch(rc, cands, tmp_path,
                                                                      monkeypatch):
    """Plan:534: "``_wait_isaac_clear()`` before every train/eval". It ran once per EXPERIMENT, so a
    chain's later stages, each eval row's first attempt and the experiment-level retry all launched
    a 256-env Kit on the one GPU with no check and (at the retry) no settle at all -- and
    ``_stop_child`` swallows a ``TimeoutExpired`` after ``proc.kill()``, so the previous child can
    still be alive. One ``pgrep`` when nothing is running."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    waits: list[str] = []
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: waits.append("wait"))
    item = candidate_item(rc, cands, ctx, "C1")               # the 6-stage curriculum chain
    header = rc.run_experiment(ctx, item)
    assert header["status"] == "ok"
    n_stages, n_rows = len(calls["train"]), len(calls["eval"])
    assert n_stages >= 2 and n_rows >= 1
    assert len(waits) == n_stages + n_rows, (len(waits), n_stages, n_rows)

    # R5: the eval RETRY is ONE launch and takes ONE wait. It used to take two (one before the
    # INFRA_WAIT_S settle and `_run_row`'s own), which made the invariant read
    # `n_stages + n_rows + 1 per eval retry` -- and the wait taken BEFORE a 20-minute sleep says
    # nothing about the GPU at the moment the retry actually starts.
    ctx2 = make_ctx(rc, cands, tmp_path, led=led, st=state(), name="retry")
    calls2 = install_fakes(rc, monkeypatch, tmp_path / "r", eval_none=1)
    waits2: list[str] = []
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: waits2.append("wait"))
    header2 = rc.run_experiment(ctx2, candidate_item(rc, cands, ctx2, "C2"))
    assert header2["status"] == "ok"
    assert (len(calls2["train"]), len(calls2["eval"])) == (1, 2), "one stage, one row retried once"
    assert len(waits2) == 3, (len(waits2), calls2["train"], calls2["eval"])


def test_the_experiment_level_retry_settles_before_it_relaunches(rc, cands, tmp_path, monkeypatch):
    """The retry beside ``train_retry``'s own, which does ``_wait_isaac_clear`` + ``RETRY_WAIT_S``.
    The pause is TICKED, so a stop inside it ends the experiment ``interrupted`` and resumable."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path, status="infra")
    waits, sleeps = [], []
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: waits.append("wait"))
    monkeypatch.setattr(rc.rp, "_interruptible_sleep",
                        lambda total, check=None, **k: bool(sleeps.append(total)) or False)
    item = candidate_item(rc, cands, ctx, "C2")
    with pytest.raises(SystemExit):                       # halt: infra, after the one retry
        rc.run_experiment(ctx, item)
    assert sleeps == [rc.rp.RETRY_WAIT_S], sleeps
    assert len(waits) >= 2, "one before the first train, one before the retry"

    # and a stop that lands inside that pause ends the experiment interrupted, not halted
    ctx2 = make_ctx(rc, cands, tmp_path, led=led, st=state(), name="camp2")
    install_fakes(rc, monkeypatch, tmp_path / "b", status="infra")
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    monkeypatch.setattr(rc.rp, "_interruptible_sleep", lambda *a, **k: True)
    header2 = rc.run_experiment(ctx2, candidate_item(rc, cands, ctx2, "C2"))
    assert header2["status"] == "running", "a stop leaves the window resumable"
    assert "stop:" in header2["interrupt"]


# ================================================================== the run_phases seams
def test_the_driver_seams_exist_with_the_keywords_the_ladder_passes(rc):
    import inspect
    train = inspect.signature(rc.rp.train_retry).parameters
    for key in ("extra_env", "interrupt_check", "run_name", "abort_track_ratio_min"):
        assert key in train, key
    for name in ("eval_morph", "eval_obst"):
        params = inspect.signature(getattr(rc.rp, name)).parameters
        for key in ("carry_overlay", "interrupt_check", "max_attempts"):
            assert key in params, (name, key)
    assert "name_suffix" in inspect.signature(rc.rp.latest_run_ckpt).parameters
    assert "interrupt_check" in inspect.signature(rc.rp._wait_isaac_clear).parameters
    assert isinstance(rc.rp.LAST_EVAL_INFO, dict)
    assert rc.rp.CURRENT_CHILD is None
    assert set(rc.rp.EVAL_CARRY_DEFAULTS) <= set(rc.rp.EVAL_CARRY)
    assert rc.rp.TRAIN_ENV_RESERVED == ("KRABBY_OMNI_PHASE", "KRABBY_PLANT", "KRABBY_PHASE",
                                        "KRABBY_SIDE_PHASE")


def test_the_ladder_directory_is_refused_by_the_phase_driver(rc, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"ladder": {}}))
    with pytest.raises(SystemExit, match="run_candidates.py"):
        rc.rp._refuse_ladder_dir(tmp_path)


def test_summarize_of_a_recorded_row_injects_the_survival_table(rc, tmp_path):
    """A milestone re-reads a run dir, and ``survival_by_hold`` is injected in memory only."""
    run_dir = tmp_path / "eval"
    run_dir.mkdir()
    episodes = [{"n_steps": 5000} for _ in range(10)]
    run_dir.joinpath("scenario_metrics.json").write_text(json.dumps({
        "aggregate": {"schedule_completion_rate": 0.5, "tripod_score": {"median": 0.4},
                      "tracking_by_hold": {}, "facing_by_hold": {}},
        "episodes": episodes,
        "run_meta": {"hold_labels": ["stand", "fwd", "lat_l", "diag", "back", "turn_l90",
                                     "turn_r90", "move_turn"], "dt": 0.02}}))
    out = rc.summarize_run_dir(run_dir, "slow__omni")
    assert out["survival_by_hold"]["fwd"] == 1.0
    assert out["turn_survival_l90"] is not None
    assert out["run_dir"] == str(run_dir)


# ================================================================== halts, heads, records
def test_the_halt_line_names_the_verb_for_this_kind(rc, cands, tmp_path):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    cases = {"candidate": "decide C5 --choose skip",
             "seed2": "confirm D05",
             "twin": "overrule <Did of the last ADOPT in the base>",
             "batched_s2": "resume-milestone M1 --drop",
             "milestone": "resume-milestone M1 --drop",
             "milestone_s2": "resume-milestone M1 --drop"}
    for kind, needle in cases.items():
        cid = "C5" if kind in ("candidate", "seed2", "twin") else "M1"
        verbs = " | ".join(rc._halt_verbs(ctx, {"kind": kind, "candidate": cid, "decision": "D05"}, None))
        assert needle in verbs, (kind, verbs)
    seed2 = " | ".join(rc._halt_verbs(ctx, {"kind": "seed2", "candidate": "C5", "decision": "D05"}, None))
    assert "overrule D05" in seed2 and "squashes this pair" in seed2


def test_the_measured_dead_walker_floor_is_written_once_and_cited_by_later_candidates(rc, cands,
                                                                                     tmp_path,
                                                                                     monkeypatch):
    """``D03`` is measured off B0's own its-800-1000 trace and never re-measured by a ``B<k>``."""
    led, st = ledger(), state()
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    b0 = rc.compose_recipe(ctx.ledger, cands, ctx.code, upto="C1")
    item = rc._item("baseline", cid="B0", seed="3", recipe=b0, base_recipe=b0,
                    stages=[dict(s) for s in b0["stages"]], label="B0 s3")
    header = rc.run_experiment(ctx, item)
    log = Path(ctx.c.recs[header["recs"][-1]]["log"])
    log.write_text("\n".join(f"Metrics/base_parkour/track_ratio_mean: 0.60" for _ in range(1100)))
    rc.dispatch(ctx, item, header)
    assert ctx.lad["d03"] == pytest.approx(max(0.35, 0.75 * 0.60))
    d03 = [e for e in rc.entries(ctx.ledger) if e["kind"] == "measurement"][0]
    assert d03["evidence"]["floor"] == pytest.approx(0.45)
    assert "candidate kinds only" in d03["evidence"]["armed_on"]
    # a B<k> rerun must not move it
    item2 = rc._item("baseline", cid="B1", seed="3", recipe=b0, base_recipe=b0,
                     stages=[dict(s) for s in b0["stages"]], label="B1 s3")
    header2 = rc.run_experiment(ctx, item2)
    rc.dispatch(ctx, item2, header2)
    assert len([e for e in rc.entries(ctx.ledger) if e["kind"] == "measurement"]) == 1
    assert ctx.lad["d03"] == pytest.approx(0.45)


def test_the_recipe_head_and_its_seed_two_twin_resolve_to_the_recorded_checkpoints(rc, cands):
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    assert rc.recipe_head(st, CODE, b0["recipe_id"], "3") == "/sim/X01/model_7999.pt"
    assert rc.recipe_head(st, CODE, b0["recipe_id"], "2") == "/sim/X02/model_7999.pt"
    assert rc.recipe_head(st, OTHER_CODE, b0["recipe_id"], "3") is None
    # the newest ok screen wins (a re-run's screen becomes the head)
    st["ladder"]["experiments"]["X09"] = exp("X09", "baseline", "B1", "3", b0["recipe_id"],
                                            b0["recipe_id"], {"upto": None, "inclusive": False},
                                            started="2026-09-20T00:00:00")
    assert rc.recipe_head(st, CODE, b0["recipe_id"], "3") == "/sim/X09/model_7999.pt"


def test_the_stage_record_is_the_shape_the_decision_table_reads(rc, cands, tmp_path, monkeypatch):
    """``decision_table.py --phases X01:1c`` reads ``keys.<seed>.<X:stage>.evals.<row>.run_dir``."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)
    header = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    state_json = json.loads((ctx.dir / "state.json").read_text())
    key = header["recs"][-1]
    assert ":" in key
    rec = state_json["keys"]["seed3"][key]
    assert Path(rec["evals"]["slow"]["run_dir"]).name
    for field in ("phase", "tag", "ckpt", "status", "iters", "log", "wall_h", "evals"):
        assert field in rec
    assert state_json["ladder"]["experiments"][header["xid"]]["recipe_id"]
    assert state_json["current"]["kind"] in ("rl", "eval", "replay", "bootstrap", "wait")


def test_the_heartbeat_contract_is_kept(rc, cands, tmp_path, monkeypatch):
    """``heartbeat_phases.sh`` reads ``phase`` and ``current.{tag,iters,kind,log}`` and logs/*.log."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    ctx.state["current"] = {"tag": "t", "phase": "1c", "iters": 8000, "kind": "rl", "log": "x.log"}
    ctx.state["phase"] = "run"
    ctx.save()
    got = json.loads((ctx.dir / "state.json").read_text())
    assert got["phase"] == "run"
    for field in ("tag", "iters", "kind", "log"):
        assert field in got["current"]
    assert (ctx.dir / "logs").is_dir()


def test_a_replay_is_advisory_and_a_stop_makes_it_interrupted(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = candidate_item(rc, cands, ctx, "C2")
    script = Path(tmp_path) / "replay.py"
    script.write_text("import time, sys\ntime.sleep(30)\n")
    ctx.cands = copy.deepcopy(cands)
    ctx.cands["replay"] = {"script": str(script), "mode": "advisory"}
    monkeypatch.setattr(rc, "INFRA_TICK_S", 0.05)
    monkeypatch.setattr(rc, "interrupt_reason", lambda ctx_, xid: "stop")
    out = rc.run_replay(ctx, item)
    assert out["status"] == "interrupted"
    assert rc.rp.CURRENT_CHILD is None
    assert ctx.state["current"]["kind"] == "replay"


def _replay_ctx(rc, cands, tmp_path, monkeypatch, body="import time\ntime.sleep(30)\n"):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    script = Path(tmp_path) / "replay_slow.py"
    script.write_text(body)
    ctx.cands = copy.deepcopy(cands)
    ctx.cands["replay"] = {"script": str(script), "mode": "advisory"}
    monkeypatch.setattr(rc, "INFRA_TICK_S", 0.05)
    return ctx, candidate_item(rc, cands, ctx, "C2")


def test_a_hung_replay_is_stopped_by_its_own_timeout_and_is_still_advisory(rc, cands, tmp_path,
                                                                          monkeypatch):
    """F8: the one child poll loop here with no timeout. A hung ``replay_reward_economy.py`` stalled
    the campaign indefinitely with ``current.kind == "replay"`` and only a stop to end it -- for an
    ADVISORY screen. The timeout stops the child, records ``timeout`` and lets the window run."""
    ctx, item = _replay_ctx(rc, cands, tmp_path, monkeypatch)
    monkeypatch.setattr(rc, "REPLAY_TIMEOUT_S", 0.2)
    out = rc.run_replay(ctx, item)
    assert out["status"] == "timeout"
    assert rc.rp.CURRENT_CHILD is None
    assert out["rc"] is not None, "the child was reaped, not orphaned"
    # advisory: the loop does not skip the window for a timeout the way it does for an interrupt
    assert out["status"] != "interrupted"


def test_an_exception_inside_the_replay_poll_loop_stops_the_child(rc, cands, tmp_path, monkeypatch):
    """F8: ``interrupt_reason`` does file I/O and can raise out of ``_apply_override``. Without the
    guard the escape orphaned the replay child AND cleared ``rp.CURRENT_CHILD`` -- the only handle
    the signal handler could have reached it through."""
    ctx, item = _replay_ctx(rc, cands, tmp_path, monkeypatch)
    seen = {}

    def boom(ctx_, xid):
        seen["child"] = rc.rp.CURRENT_CHILD
        raise RuntimeError("the overrides file is corrupt")

    monkeypatch.setattr(rc, "interrupt_reason", boom)
    with pytest.raises(RuntimeError, match="corrupt"):
        rc.run_replay(ctx, item)
    assert rc.rp.CURRENT_CHILD is None
    assert seen["child"].poll() is not None, "the child was stopped, not orphaned"


def test_a_missing_replay_script_is_an_error_not_a_halt(rc, cands, tmp_path):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    ctx.cands = copy.deepcopy(cands)
    ctx.cands["replay"] = {"script": "nope/replay.py", "mode": "advisory"}
    item = candidate_item(rc, cands, ctx, "C2")
    out = rc.run_replay(ctx, item)
    assert out["status"] == "error" and "does not exist" in out["why"]


def test_the_replay_is_skipped_for_a_candidate_that_declares_no_args(rc, cands, tmp_path):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    assert rc.run_replay(ctx, candidate_item(rc, cands, ctx, "C1")) is None
    assert rc.run_replay(ctx, candidate_item(rc, cands, ctx, "C5")) is None


def test_a_repo_relative_script_path_and_an_absolute_one_both_resolve(rc, cands):
    assert rc.repo_path(cands["replay"]["script"]).exists()
    assert rc.repo_path("/etc/hostname") == Path("/etc/hostname")


def test_the_walker_reference_points_at_the_baselines_own_raws_once_they_exist(rc, cands, tmp_path,
                                                                             monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc._baseline_raw_dir(ctx) is None
    raw = Path(tmp_path) / "b0eval" / "raw"
    raw.mkdir(parents=True)
    ctx.lad["experiments"]["X01"]["recs"] = ["X01:1c"]
    ctx.c.recs["X01:1c"] = {"evals": {"slow": {"run_dir": str(raw.parent)}}}
    assert rc._baseline_raw_dir(ctx) == str(raw)


# ================================================================== the ledger CLI
def cli_run(cli, argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out + out.err


def test_the_cli_shows_the_ledger_and_the_pending_pairs(rc, cli, cands, tmp_path, capsys):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "show"], capsys)
    assert code == 0 and "D05" in out and "C1" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "pending"], capsys)
    assert code == 0 and "D05" in out and "confirm D05" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "show", "--json"], capsys)
    assert json.loads(out)[0]["id"] == "D02"


def test_the_cli_refuses_to_overrule_a_measurement_and_names_the_alternative(rc, cli, cands, tmp_path,
                                                                            capsys):
    led, st = ledger(), state()
    led["entries"].append({"id": "D00", "kind": "resolution", "candidate": None, "chosen": "RECORD",
                           "status": "final", "options": ["RECORD"], "history": []})
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "overrule", "D00", "--choose", "REJECT",
                              "--why", "no"], capsys)
    assert code == 2 and "not choices" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "confirm", "D02"], capsys)
    assert code == 2 and "baseline" in out
    assert rc.lc.read_overrides(ctx.dir) == []


def test_the_cli_refuses_an_overrule_without_a_reason_or_with_an_unknown_choice(rc, cli, cands,
                                                                               tmp_path, capsys):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "overrule", "D05", "--choose", "REJECT"],
                        capsys)
    assert code == 2 and "--why" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "overrule", "D05", "--choose", "MAYBE",
                              "--why", "x"], capsys)
    assert code == 2 and "options" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "overrule", "D05", "--choose", "REJECT",
                              "--why", "the user disagrees"], capsys)
    assert code == 0 and "overrule D05 -> REJECT" in out
    assert rc.lc.read_overrides(ctx.dir)[0]["choice"] == "REJECT"


def test_the_cli_refuses_decide_skip_on_a_decided_candidate(rc, cli, cands, tmp_path, capsys):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C5", "REJECT", "decided", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "decide", "C5", "--choose", "skip"], capsys)
    assert code == 2 and "confirm D05" in out and "overrule D05" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "decide", "C5", "--choose", "run"], capsys)
    assert code == 0 and "will be squashed" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "decide", "C99", "--choose", "run"], capsys)
    assert code == 2 and "not a candidate id" in out


def test_the_cli_refuses_a_drop_that_would_not_change_the_recipe(rc, cli, cands, tmp_path, capsys):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "REJECT", "decided", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--drop", "C2"],
                        capsys)
    assert code == 2 and "ADOPT" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--drop", "C9"],
                        capsys)
    assert code == 2 and "block" in out
    ctx.ledger["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1",
                                  "chosen": "FAIL", "status": "await", "history": [],
                                  "options": ["PASS", "WATCH", "FAIL"], "evidence": {}})
    ctx.write_ledger()
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--accept"],
                        capsys)
    assert code == 0 and "frozen" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--accept",
                              "--drop", "C2"], capsys)
    assert code == 2 and "exactly one" in out


def test_accept_is_refused_unless_the_milestone_is_actually_awaiting_an_answer(rc, cli, cands,
                                                                               tmp_path, capsys):
    """F2: ``--accept`` freezes the whole block, so it is legal only FROM ``await`` (plan:552-556).

    Typed early, or re-run from an OLD milestone notification after that milestone closed, it
    permanently froze every adoption in the block -- and a frozen ADOPT can no longer be dropped by
    a milestone contradiction nor moved by a pooled self-overrule, so the campaign's own error
    correction was silently disarmed. Both halves refuse: the CLI with rc 2 and no line appended,
    the driver with a ``REFUSED:`` note and an untouched block.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "ADOPT", "decided", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)

    # (a) the CLI, with no M1 entry at all
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--accept"],
                        capsys)
    assert code == 2 and "not awaiting" in out and "not in the ledger yet" in out
    assert not (ctx.dir / "ledger_overrides.jsonl").exists(), "a refused verb appends nothing"

    # (b) the CLI, with a CLOSED M1 (the stale-notification case)
    ctx.ledger["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1",
                                  "chosen": "PASS", "status": "final", "history": [],
                                  "options": ["PASS", "WATCH", "FAIL"], "evidence": {}})
    ctx.write_ledger()
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--accept"],
                        capsys)
    assert code == 2 and "'final'" in out
    assert not (ctx.dir / "ledger_overrides.jsonl").exists()

    # (c) the driver, handed the line anyway (an old line replayed at launch)
    rc.lc.append_override(ctx.dir, {"milestone": "M1", "accept": True, "why": "stale"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "decided" and not d05.get("frozen"), "the block must not be frozen"
    assert rc.entry_by_id(ctx.ledger, "M1")["status"] == "final"
    assert any("REFUSED" in line and "not awaiting" in line for line in notify_lines(ctx))


def test_a_milestone_is_never_confirmed_or_overruled(rc, cli, cands, tmp_path, capsys):
    """F3: ``overrule M1 --choose FAIL`` re-booked the promotion and then crashed the driver.

    The CLI accepted it (milestone kinds were not in ``UNOVERRULABLE_KINDS``), ``TRANSITIONS`` let a
    ``final`` entry move to ``overruled``, ``_next_item_raw`` read "not await, not final" as the
    OPEN branch and emitted ``M1.promo`` again (4.3 h), and ``_close_entry`` then raised
    ``overruled -> final`` uncaught in ``loop()`` -- no HALT line, no phase, a traceback exit.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "PASS",
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {}, "recipe_id": rid(rc, cands, led, "C3b", inclusive=True)})
    led["entries"].append({"id": "D20", "kind": "milestone_s2", "candidate": "M3",
                           "chosen": "RECORD", "status": "final", "options": ["RECORD"],
                           "history": [], "evidence": {}})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    before = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)

    for verb, extra in (("overrule", ["--choose", "FAIL", "--why", "no"]), ("confirm", [])):
        for ident in ("M1", "D20"):
            code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), verb, ident, *extra], capsys)
            assert code == 2, (verb, ident, out)
            assert "resume-milestone" in out and "not confirmed or overruled" in out
    assert not (ctx.dir / "ledger_overrides.jsonl").exists(), "a refused verb appends nothing"

    # `show` no longer advertises the verbs for a milestone row -- it advertises the right one
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "show", "M1"], capsys)
    assert code == 0
    assert "confirm/overrule verbs available" not in out
    assert "resume-milestone M1 --accept" in out

    # and the driver refuses the line even if one reaches it, leaving the queue where it was
    rc.lc.append_override(ctx.dir, {"decision": "M1", "action": "overrule", "choice": "FAIL",
                                    "why": "from an old notification"})
    rc.ingest_overrides(ctx)
    assert rc.entry_by_id(ctx.ledger, "M1")["status"] == "final"
    assert any("REFUSED" in line and "resume-milestone" in line for line in notify_lines(ctx))
    after = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert (after or {}).get("label") == (before or {}).get("label")


def test_the_milestone_block_is_the_rank_range_between_two_milestones(rc, cli, cands):
    assert cli.milestone_block(cands, "M1") == ["C1", "C1b", "C1c", "C2", "C3a", "C3b"]
    assert cli.milestone_block(cands, "M2") == ["C4", "C5", "C6", "C7"]
    assert cli.milestone_block(cands, "M3") == ["C8", "C9", "C10", "C11", "C12"]
    with pytest.raises(KeyError):
        cli.milestone_block(cands, "M9")


def test_stop_and_unstop_write_and_remove_the_stop_file(rc, cli, cands, tmp_path, capsys):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "stop", "--why", "the user asked"], capsys)
    assert code == 0 and (ctx.dir / "STOP").exists()
    assert rc.stop_requested(ctx) == "stop"
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "unstop"], capsys)
    assert code == 0 and not (ctx.dir / "STOP").exists()
    assert rc.stop_requested(ctx) is None


def test_the_cli_says_whether_the_driver_will_see_it_now_or_at_the_next_launch(rc, cli, cands,
                                                                              tmp_path, capsys):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "confirm", "D05"], capsys)
    assert code == 0 and "driver not running" in out


def test_the_cli_refuses_a_campaign_dir_that_does_not_exist(cli, tmp_path, capsys):
    code, out = cli_run(cli, ["--campaign-dir", str(tmp_path / "nope"), "pending"], capsys)
    assert code == 2 and "no campaign directory" in out


def test_a_user_skip_moves_the_running_header_into_the_squashed_record(rc, cands, tmp_path):
    """The derived view and the RECORD must agree, or the next launch resumes an abandoned window."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r = rc._recipe_with(led, cands, CODE, "C5", set())
    st["ladder"]["experiments"]["X07"] = exp("X07", "candidate", "C5", "3", r["recipe_id"],
                                            rid(rc, cands, led, "C5"), base_rank("C5"),
                                            status="running", started="2026-09-19T01:00:00",
                                            inflight={"stage": "1c", "iters": 8000, "tag": "t"})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "skip", "why": "not now"})
    rc.ingest_overrides(ctx)
    assert "X07" in ctx.lad["squashed"]
    assert ctx.lad["squashed"]["X07"]["squashed_by"] == "user_choice"
    assert "X07" not in ctx.lad["experiments"]
    assert rc.find_inflight(ctx) is None
    assert "user skip" in str(rc.interrupt_reason(ctx, "X07")) or "user_choice" in str(
        rc.interrupt_reason(ctx, "X07"))


def test_the_decision_line_carries_every_field_the_contract_names(rc, cands, tmp_path, monkeypatch):
    """The DECISION line, end to end: blocks, guards, drift, resolution, WATCH, next, both commands."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=screen(closed=0.72, prog=0.11))
    item = candidate_item(rc, cands, ctx, "C2")
    header = rc.run_experiment(ctx, item)
    rc.dispatch(ctx, item, header)
    line = [line for line in notify_lines(ctx) if " DECISION " in line][-1]
    got = rc.lc.parse_notify(line)
    assert got["tag"] == "DECISION" and got["id"].startswith("D")
    text = " | ".join(got["fields"])
    assert "C2-reward {KRABBY_YAW_PROGRESS_W=3.0}" in text
    assert "chosen ADOPT (clear, decided) alt REJECT" in text
    assert "F z=" in text and "BETTER" in text and "T z=" in text
    assert "guards TIE" in text and "completion" in text
    assert "anchored drift" in text and "resolution D00 @0.95/0.99" in text
    assert "WATCH:" in text and "track 0.645->" in text
    assert "next:" in text
    assert "confirm: " in text and "ledger_cli.py" in text and "--campaign-dir" in text
    assert "overrule: " in text and "--choose REJECT" in text
    # the plan's own line shape: the F block names both metric FAMILIES, the two are separated by
    # "; ", and each WATCH row has a name of its own (the bare hold collided three times over)
    f_field = [f for f in got["fields"] if f.startswith("F z=")][0]
    assert f_field.split("BETTER: ")[1] == (
        "closed l90 0.500->0.720, r90 0.500->0.720, mt 0.500->0.720; "
        "progress l90 0.080->0.110, r90 0.080->0.110, mt 0.080->0.110"), f_field
    t_field = [f for f in got["fields"] if f.startswith("T z=")][0]
    assert t_field.endswith("fwd 0.600->0.600, lat_l 0.600->0.600, diag 0.600->0.600, "
                            "back 0.600->0.600"), t_field
    watch = [f for f in got["fields"] if f.startswith("WATCH: ")][0]
    names = [part.rsplit(" ", 1)[0] for part in watch[len("WATCH: "):].split(", ")]
    assert names == ["track", "track_move", "move_turn dir_ratio", "move_turn tripod",
                     "settle l90", "settle r90", "lat_r ratio", "move_turn_r ratio",
                     "closed mt_r", "surv l90", "surv r90"], names
    assert len(names) == len(set(names)), "every WATCH row must be identifiable"
    entry_ = rc.newest_entry(ctx.ledger, "C2", set(ctx.view["entries"]))
    assert entry_["chosen"] in entry_["options"]
    assert entry_["code_sha"] == ctx.code and entry_["tools_sha"] == ctx.tools
    assert entry_["control_experiment"] == "X01"
    assert ">>> ENTRY" in (ctx.dir / "REPORT.md").read_text()
    assert entry_["id"] in (ctx.dir / "CHANGELOG.md").read_text()


def test_the_ledger_ids_are_assigned_in_write_order(rc, cands, tmp_path):
    led = ledger({"id": "D00", "kind": "resolution", "chosen": "RECORD", "status": "final"},
                 {"id": "D01", "kind": "measurement", "chosen": "RECORD", "status": "final"})
    assert rc._next_did(led) == "D02"
    led["entries"].append({"id": "D02", "kind": "baseline", "chosen": "RECORD", "status": "final"})
    led["entries"].append({"id": "D00b", "kind": "resolution", "chosen": "RECORD", "status": "final"})
    assert rc._next_did(led) == "D03", "a non-numeric id does not take a number"
    led["entries"] = [e for e in led["entries"] if e["id"] != "D01"]
    assert rc._next_did(led) == "D03", "a gap is never re-used (a notification already named it)"
    assert rc._next_mid(led, "M1") == "M1"
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1"})
    assert rc._next_mid(led, "M1") == "M1#2"


# ================================================================== the startup refusals
def _main_env(rc, cands, tmp_path, monkeypatch, *, dirty=(), code=CODE, name="startup"):
    cdir = Path(tmp_path) / name
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "candidates.yaml").write_text(CANDS_YAML.read_text())
    monkeypatch.setattr(rc, "dirty_paths", lambda *a, **k: list(dirty))
    monkeypatch.setattr(rc, "code_sha", lambda *a, **k: code)
    monkeypatch.setattr(rc, "code_rows", lambda *a, **k: ["aaa parkour/x/mdp/a.py"])
    monkeypatch.setattr(rc, "ensure_resolution", lambda ctx: {})
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    monkeypatch.setattr(rc, "loop", lambda ctx: 0)
    return cdir


def test_a_dirty_code_tree_refuses_a_real_launch_but_not_a_dry_run(rc, cands, tmp_path, monkeypatch):
    cdir = _main_env(rc, cands, tmp_path, monkeypatch, dirty=["parkour/x/mdp/a.py"])
    with pytest.raises(SystemExit, match="dirty"):
        rc.main(["--campaign-dir", str(cdir)])
    assert rc.main(["--campaign-dir", str(cdir), "--dry-run"]) == 0


def test_a_changed_code_sha_refuses_a_relaunch_without_acceptance(rc, cands, tmp_path, monkeypatch):
    cdir = _main_env(rc, cands, tmp_path, monkeypatch)
    assert rc.main(["--campaign-dir", str(cdir)]) == 0
    state_json = json.loads((cdir / "state.json").read_text())
    assert state_json["ladder"]["code_sha"] == CODE
    monkeypatch.setattr(rc, "code_sha", lambda *a, **k: OTHER_CODE)
    with pytest.raises(SystemExit, match="code_sha changed"):
        rc.main(["--campaign-dir", str(cdir)])
    assert rc.main(["--campaign-dir", str(cdir), "--accept-code-change", "C10 landed"]) == 0
    line = [line for line in (cdir / "notify.log").read_text().splitlines() if "RELAUNCH" in line][-1]
    assert CODE in line and OTHER_CODE in line and "C10 landed" in line


def test_a_changed_candidates_file_refuses_a_relaunch_without_acceptance(rc, cands, tmp_path,
                                                                        monkeypatch):
    cdir = _main_env(rc, cands, tmp_path, monkeypatch)
    assert rc.main(["--campaign-dir", str(cdir)]) == 0
    (cdir / "candidates.yaml").write_text(CANDS_YAML.read_text() + "\n# an edit\n")
    with pytest.raises(SystemExit, match="candidates.yaml changed"):
        rc.main(["--campaign-dir", str(cdir)])
    assert rc.main(["--campaign-dir", str(cdir), "--accept-candidates-change", "a comment"]) == 0


def test_a_missing_candidates_file_is_named(rc, tmp_path):
    with pytest.raises(SystemExit, match="no candidates.yaml"):
        rc.main(["--campaign-dir", str(tmp_path / "empty")])


def test_the_launch_records_the_unit_and_the_header(rc, cands, tmp_path, monkeypatch):
    cdir = _main_env(rc, cands, tmp_path, monkeypatch)
    assert rc.main(["--campaign-dir", str(cdir), "--unit", "cands-x-1"]) == 0
    st = json.loads((cdir / "state.json").read_text())
    assert st["ladder"]["unit"] == "cands-x-1" and st["phase"] == "run"
    assert st["ladder"]["candidates_sha256"] == rc.sha256_file(cdir / "candidates.yaml")
    report = (cdir / "REPORT.md").read_text()
    assert "Candidate ladder" in report and "| C12 | task_definition |" in report
    assert "M3 | milestone" in report
    assert "campaign start" in (cdir / "CHANGELOG.md").read_text()


def test_the_stop_file_parks_the_loop_before_anything_is_dispatched(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    (ctx.dir / "STOP").write_text("the user asked\n")
    monkeypatch.setattr(ctx.args, "max_wall_h", None, raising=False)
    assert rc.loop(ctx) == 0
    assert ctx.state["phase"] == "await_user:stop"
    assert any("AWAIT await_user:stop" in line for line in notify_lines(ctx))


def test_a_sigterm_flag_parks_the_loop_the_same_way(rc, cands, tmp_path, monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    monkeypatch.setitem(rc.INTERRUPT, "signal", 15)
    try:
        assert rc.loop(ctx) == 0
        assert ctx.state["phase"] == "await_user:stop"
    finally:
        rc.INTERRUPT["signal"] = None


def test_the_sigterm_handler_only_flags_and_signals_the_child(rc, cands, tmp_path):
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())

    class Child:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    child = Child()
    rc.install_sigterm(ctx)
    handler = __import__("signal").getsignal(__import__("signal").SIGTERM)
    rc.rp.CURRENT_CHILD = child
    try:
        handler(15, None)
        assert rc.INTERRUPT["signal"] == 15
        assert child.terminated is True, "the handler SIGTERMs the child; it never kills or exits"
        assert rc.stop_requested(ctx) == "sigterm"
    finally:
        rc.INTERRUPT["signal"] = None
        rc.rp.CURRENT_CHILD = None
        __import__("signal").signal(__import__("signal").SIGTERM, __import__("signal").SIG_DFL)


def test_an_eval_carry_key_set_per_stage_still_reaches_the_teacher_windows(rc, cands):
    """Every ``eval_carry`` key must be in the 2a/2b env, or the promotion's rows mis-score the head."""
    doc = copy.deepcopy(cands)
    for cand in doc["candidates"]:
        if cand["id"] == "C5":
            cand.pop("stages", None)
            cand["overlay"] = {"1c": {"KRABBY_OMNI_FACING_OBS_ENCODING": "sincos"}}
    r = rc.compose_recipe(ledger(entry("D05", "C5", "ADOPT", "decided")), doc, CODE)
    assert r["eval_carry"] == ["KRABBY_OMNI_FACING_OBS_ENCODING"]
    for stage in ("2a", "2b"):
        assert r["overlay_by_stage"][stage]["KRABBY_OMNI_FACING_OBS_ENCODING"] == "sincos"


def test_the_decision_evidence_carries_the_replay_and_the_retention_lines(rc, cands, tmp_path,
                                                                        monkeypatch):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    good = screen()
    good["survival_by_hold"] = {h: 0.95 for h in ("stand", "fwd", "lat_l", "lat_r", "back")}
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=good)
    item = candidate_item(rc, cands, ctx, "C1")
    header = rc.run_experiment(ctx, item)
    header["replay"] = {"status": "ok", "rc": 0}
    rc.dispatch(ctx, item, header)
    e = rc.newest_entry(ctx.ledger, "C1", set(ctx.view["entries"]))
    assert e["evidence"]["replay"] == {"status": "ok", "rc": 0}
    assert any("RETENTION" in line for line in e["evidence"]["retention"])
    assert e["evidence"]["floor_watch"] is None


def test_the_smoke_hook_parks_instead_of_booking_a_real_window(rc, cands, tmp_path, monkeypatch):
    """Verification 8's launch: one tiny window, then ``await_user:smoke_done`` -- never B0."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    ctx.args.smoke_stage = "1a:20"
    calls = install_fakes(rc, monkeypatch, tmp_path)
    assert rc.loop(ctx) == 0
    assert ctx.state["phase"] == "await_user:smoke_done"
    assert [c["iters"] for c in calls["train"]] == [20]
    assert rc.entries(ctx.ledger) == []
    assert rc.loop(ctx) == 0 and ctx.state["phase"] == "await_user:smoke_done"


def test_squash_set_is_the_plans_name_for_the_derived_view(rc):
    assert rc.squash_set is rc.squash_view


# ============================================================ the step-5 audit's blocking findings
def _stop_after(n: int):
    """An ``interrupt_reason`` that is clear for ``n`` calls and then says "stop".

    The pre-launch check is call 1 and the training child's own check is call 2, so this stops a
    window MID-STAGE -- which is what leaves the header `running` with an ``inflight`` block, the
    state a resume actually starts from.
    """
    seen = {"n": 0}

    def _reason(ctx_, xid):
        seen["n"] += 1
        return None if seen["n"] <= n else "stop"
    return _reason

def test_the_pooled_drift_is_pooled_even_when_the_anchor_has_no_seed_two(rc, rule):
    """Plan decision 5: at the pooled level the drift is the candidate's TWO-SEED MEAN deltas vs the
    anchor's s3 -- "and its s2 ... WHEN the anchor recipe has an ok s2 screen" makes the ANCHOR's
    second seed optional, not the candidate's.

    With the anchor's s2 missing the comparison must stay pooled (the anchor's s3 standing in as its
    own second seed). Inferring "pooled" from "both second screens are present" instead collapsed
    the drift delta to the s3 delta while the threshold stayed shrunk by sqrt(2) -- which turned a
    clear ADOPT into a REJECT on evidence that had not changed, and a pooled decision is TERMINAL.
    """
    cand_s3 = screen(closed=0.62, prog=0.10)
    cand_s2 = screen(closed=0.74, prog=0.12)             # the second seed is the BETTER one
    base, anchor_s3, anchor_s2 = screen(), screen(closed=0.66, prog=0.105), screen()
    with_s2 = rc.decide_pooled(cand_s3, cand_s2, base, base, resol(), cand_spec={}, rule=rule,
                               prior={"chosen": "ADOPT", "status": "speculative",
                                      "options": ["ADOPT", "REJECT", "SKIP"]},
                               anchor_s3=anchor_s3, anchor_s2=anchor_s2, anchor_label="A")
    without = rc.decide_pooled(cand_s3, cand_s2, base, base, resol(), cand_spec={}, rule=rule,
                               prior={"chosen": "ADOPT", "status": "speculative",
                                      "options": ["ADOPT", "REJECT", "SKIP"]},
                               anchor_s3=anchor_s3, anchor_s2=None, anchor_label="A")
    assert with_s2["evidence"]["anchored_drift"]["verdict"] == "OK"
    assert without["evidence"]["anchored_drift"]["verdict"] == "OK"
    assert without["evidence"]["anchored_drift"]["anchor_seed2"].startswith("its s3 reused")
    assert with_s2["chosen"] == without["chosen"] == "ADOPT"
    # and the drift z is the TWO-SEED MEAN delta, not the s3 delta alone
    z = without["evidence"]["anchored_drift"]["z"]["F"]
    assert z == pytest.approx(((0.62 + 0.74) / 2 - 0.66) / 0.10 / 2
                              + ((0.10 + 0.12) / 2 - 0.105) / 0.10 / 2, abs=1e-9)


def test_a_pooled_comparison_that_lost_a_second_screen_is_missing_not_half_pooled(rc, rule):
    """The flag is threaded, so a pooled row with no second screen is MISSING -- never a single-seed
    delta judged against a pooled (sqrt-2 shrunk) threshold."""
    rows = rc.metric_rows(rule["blocks"]["F"]["metrics"], screen(closed=0.9), screen(), resol(), {},
                          cand2=screen(closed=0.9), base2=None, pooled=True)
    assert all(r["missing"] == "value" for r in rows)


def test_the_anchor_falls_back_to_the_baseline_until_an_adoption_qualifies(rc, cands, tmp_path):
    """Plan Vocabulary: the anchor is the last CLEAR/user-settled ADOPT, "else ``B0``".

    A speculative adoption never qualifies, so without the fallback a ladder whose adoptions are all
    speculative (the plan's own budget expects ~4) would have no anchor at all and the drift
    precondition -- the thing that stops a chain of tiny speculative adoptions drifting away from the
    baseline -- would be inert for exactly the case it exists for.
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    assert rc.compose_recipe(led, cands, CODE)["anchor"]["candidate"] == "B0"
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", confidence="uncertain",
                                experiment="X03"))
    anchor = rc.compose_recipe(led, cands, CODE)["anchor"]
    assert anchor["candidate"] == "B0" and anchor["decision"] == "D02"
    assert "baseline" in anchor["why"]
    # the anchor SCREEN resolves to B0's own seed-3 screen (X01), and its s2 to X02
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    s3, s2, label = rc._anchor_screens(ctx, rc.compose_recipe(ctx.ledger, cands, ctx.code))
    assert s3 == st["ladder"]["experiments"]["X01"]["screen"]
    assert s2 == st["ladder"]["experiments"]["X02"]["screen"]
    assert label.startswith("B0/D02")
    # a CLEAR adoption replaces it
    led["entries"].append(entry("D08", "C2", "ADOPT", "decided", confidence="clear",
                                experiment="X04"))
    assert rc.compose_recipe(led, cands, CODE)["anchor"]["candidate"] == "C2"
    assert rc.compose_recipe(ledger(), cands, CODE)["anchor"] is None, "no baseline entry, no anchor"


def test_a_candidate_that_drifted_below_the_baseline_cannot_be_a_clear_adopt(rc, cands, tmp_path,
                                                                            monkeypatch):
    """The payoff of the fallback, end to end: C2 is BETTER than the recipe it is screened on and
    WORSE than B0 on a guard, and the decision is a drift-forced speculative REJECT."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", confidence="uncertain",
                                experiment="X03", base=b0["recipe_id"], recipe=r_c1["recipe_id"]))
    drifted = screen(completion=0.20)                       # C1 already lost completion
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            b0["recipe_id"], base_rank("C1"), screen_=drifted)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path,
                  eval_summary=screen(closed=0.72, prog=0.11, completion=0.21))
    item = candidate_item(rc, cands, ctx, "C2")
    header = rc.run_experiment(ctx, item)
    rc.dispatch(ctx, item, header)
    d = rc.newest_entry(ctx.ledger, "C2", set(ctx.view["entries"]))
    assert d["chosen"] == "REJECT" and d["confidence"] == "uncertain"
    assert d["status"] == "speculative", "the pair still runs"
    assert d["evidence"]["anchored_drift"]["verdict"] == "FAIL"
    assert "completion" in str(d["evidence"]["anchored_drift"]["worse"])


def test_a_pooled_guard_missing_is_a_terminal_reject(rc, rule):
    """Decision 5's guard rule applies to the pooled level too: a guard the pair cannot measure is a
    REJECT with ``pooled_confidence: uncertain``, never a TIE that can carry an ADOPT."""
    good = screen(closed=0.72, prog=0.11)
    lost = screen(closed=0.72, prog=0.11, surv_l=None)      # surv[back] == 0 on the seed-2 run
    out = rc.decide_pooled(good, lost, screen(), screen(), resol(), cand_spec={}, rule=rule,
                           prior={"chosen": "ADOPT", "status": "speculative",
                                  "options": ["ADOPT", "REJECT", "SKIP"]})
    assert out["chosen"] == "REJECT"
    assert out["pooled_confidence"] == "uncertain"
    assert out["evidence"]["missing"]["guards"] == ["turn_survival_l90"]
    assert "turn_survival_l90" in out["why"]


def test_the_validator_refuses_a_cap_candidate_that_does_not_declare_absolute_units(rc, cands):
    """Decision 5: ANY candidate that moves the command cap/clip/ellipse is scored in absolute units
    -- the cap IS the denominator of every ratio, and a ratio against a moved denominator still
    reads like a plausible number."""
    ok = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in ok), [v for v in ok if not v["ok"]]
    for key in ("KRABBY_OMNI_ANG_VEL_MAX", "KRABBY_OMNI_ANG_VEL_CLIP", "KRABBY_OMNI_SPEED_ELLIPSE"):
        bad = copy.deepcopy(cands)
        bad["candidates"].append({"id": "CX", "kind": "kinematics", "stages": "all",
                                  "question": "moves the yardstick",
                                  "overlay": {key: {"KRABBY_OMNI_SPEED_ELLIPSE": "0.4:0.3:0.3",
                                                    "KRABBY_OMNI_ANG_VEL_MAX": "0.21",
                                                    "KRABBY_OMNI_ANG_VEL_CLIP": "0.12"}[key]}})
        if key == "KRABBY_OMNI_ANG_VEL_MAX":
            bad["candidates"][-1]["overlay"]["KRABBY_OMNI_CLOCK_TURN_R"] = "1.72"
        out = [v for v in rc.validate_candidates(bad, strict=False, check_replay=False)
               if v["id"] == "CX"][0]
        assert not out["ok"] and any("absolute_units" in r for r in out["reasons"]), (key, out)


def test_the_cap_rule_applies_the_default_clip_when_no_clip_is_overlaid(rc, cands):
    """F5: no preset sets ``KRABBY_OMNI_ANG_VEL_CLIP``, so reading it off the overlay alone left the
    ``ang_vel_max > clip`` rule unarmed for a CAP-ONLY candidate -- the one shape it is for.

    A cap in (0.15, 0.2] passed load and died at config build: no checkpoint -> ``infra`` -> two
    retries -> ``halt: infra``, which ``RESUMABLE_HALTS`` then resumes into the same failure.
    """
    bad = copy.deepcopy(cands)
    bad["candidates"].append({"id": "CX", "kind": "kinematics", "stages": "all",
                              "absolute_units": True,
                              "question": "a cadence-preserving cap below the default clip",
                              "overlay": {"KRABBY_OMNI_ANG_VEL_MAX": "0.19",
                                          "KRABBY_OMNI_CLOCK_TURN_R": "1.9"}})
    out = [v for v in rc.validate_candidates(bad, strict=False, check_replay=False)
           if v["id"] == "CX"][0]
    assert not out["ok"]
    assert any("must exceed ang_vel_clip 0.2" in r for r in out["reasons"]), out["reasons"]
    # the shipped cap candidates still validate: C3a is 0.21 > 0.2, C3b overlays clip 0.12
    ok = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in ok if v["id"] in ("C3a", "C3b")), ok
    # the >= 0.35 cadence bound's waiver: the question must RECORD the change, not merely contain
    # the substring "caden" -- and "cadence-preserving" asserts the opposite of what it waives
    assert rc._records_cadence("this changes the gait cadence deliberately") is True
    assert rc._records_cadence("a decadent overlay") is False, "a substring must not buy the waiver"
    for cid in ("C3a", "C3b"):
        q = [c for c in cands["candidates"] if c["id"] == cid][0]["question"]
        assert rc._records_cadence(q) is False, (cid, "it says cadence-PRESERVING")
    bad = copy.deepcopy(cands)
    bad["candidates"].append({"id": "CW", "kind": "kinematics", "stages": "all",
                              "absolute_units": True,
                              "question": "0.24 with a cadence-preserving radius",
                              "overlay": {"KRABBY_OMNI_ANG_VEL_MAX": "0.24",
                                          "KRABBY_OMNI_CLOCK_TURN_R": "1.3"}})
    out = [v for v in rc.validate_candidates(bad, strict=False, check_replay=False)
           if v["id"] == "CW"][0]
    assert not out["ok"] and any("gait cadence" in r for r in out["reasons"]), out["reasons"]


def test_the_cap_rule_refuses_loudly_if_its_default_clip_ever_goes_missing(rc, monkeypatch):
    """R5: the default is the rule's other OPERAND, so losing it must not unarm the rule silently.

    ``clip = _num(EVAL_CARRY_DEFAULTS.get(...))`` then ``if clip is not None and not cap > clip``
    meant a moved or renamed key put the validator straight back in the pre-F5 hole -- CX (cap 0.19,
    R 1.9) validated with ZERO reasons -- and the cap/radius branch beside it already appends a
    reason when it cannot resolve its inputs.
    """
    env = {"KRABBY_OMNI_ANG_VEL_MAX": "0.19", "KRABBY_OMNI_CLOCK_TURN_R": "1.9"}
    reasons: list = []
    rc._check_cap_rules(env, "CX", "a cadence-preserving cap below the default clip", reasons)
    assert any("must exceed ang_vel_clip 0.2" in r for r in reasons), reasons

    monkeypatch.setattr(rc.rp, "EVAL_CARRY_DEFAULTS", {"KRABBY_OMNI_FACING_OBS_ENCODING": "raw"})
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_ANG_VEL_CLIP"):
        rc._check_cap_rules(env, "CX", "a cadence-preserving cap", [])
    # an OVERLAID clip still answers the rule without touching the default
    overlaid: list = []
    rc._check_cap_rules(dict(env, KRABBY_OMNI_ANG_VEL_CLIP="0.12"), "CX", "q", overlaid)
    assert not any("ang_vel_clip" in r for r in overlaid), overlaid


def test_the_teacher_pool_is_rescaled_from_the_composed_final_stage_pool(rc, cands):
    """F6: the plan's "rescale_to(SLOTS_TEACHER) of the RENDERED final-stage pool".

    Reading the chain's RAW final-stage pool instead meant a candidate that sets a raw pool string
    on that stage (one raw setter per stage is allowed) trained its own pool on the chain while
    2a/2b promoted the OLD one -- recorded only in ``conflicts``, which nothing in a live run reads.
    """
    doc = copy.deepcopy(cands)
    final = doc["chains"]["curriculum"]["stages"][-1]
    own = "stand:0.1,fwd:0.5,turn:0.2,fwd_turn:0.2"
    doc["candidates"].append({"id": "CX", "kind": "curriculum", "chain": "curriculum",
                              "question": "a raw pool on the final stage",
                              "overlay": {str(final["stage"]): {"KRABBY_OMNI_CMD_POOL": own}},
                              "teacher_overlay": {}})
    led = ledger(entry("D05", "C1", "ADOPT", "decided"), entry("D06", "CX", "ADOPT", "decided"))
    got = rc.compose_recipe(led, doc, CODE)
    assert got["overlay_by_stage"][str(final["stage"])]["KRABBY_OMNI_CMD_POOL"] \
        == rc.render_pool(rc.parse_pool(own))
    want = rc.render_pool(rc.parse_pool(own).rescale_to(
        rc.parse_slots_str(rc.ph.SLOTS_TEACHER)))
    assert got["overlay_by_stage"]["2a"]["KRABBY_OMNI_CMD_POOL"] == want
    assert got["overlay_by_stage"]["2b"]["KRABBY_OMNI_CMD_POOL"] == want
    # the base recipe (no CX) promotes the chain's own pool, so the two really do differ
    base = rc.compose_recipe(ledger(entry("D05", "C1", "ADOPT", "decided")), doc, CODE)
    assert base["overlay_by_stage"]["2a"]["KRABBY_OMNI_CMD_POOL"] != want
    # the validator's PREVIEW of the same candidate agrees with what compose_recipe writes
    spec = rc._cand_spec(doc["candidates"][-1], "curriculum")
    assert rc._teacher_preview(doc, spec, "curriculum")["KRABBY_OMNI_CMD_POOL"] == want


def test_the_validator_refuses_a_teacher_overlay_whose_slots_contradict_its_pool(rc, cands):
    """Part B's 2a/2b rule. The failure it prevents is silent until config build -- i.e. after the
    whole 8000-iteration chain has trained and the 2.4 h promotion window has been booked."""
    bad = copy.deepcopy(cands)
    for cand in bad["candidates"]:
        if cand["id"] == "C8":
            cand["teacher_overlay"] = {"KRABBY_OMNI_SLOTS": "0.15:0.45:0.15:0.25"}
    out = [v for v in rc.validate_candidates(bad, strict=False, check_replay=False)
           if v["id"] == "C8"][0]
    assert not out["ok"]
    assert any("composed pool projects to" in r and "0.05, 0.55, 0.15, 0.25" in r
               for r in out["reasons"]), out["reasons"]
    # and a teacher mix that is neither SLOTS_TEACHER nor SLOTS_TEACHER with the share edit
    worse = copy.deepcopy(cands)
    for cand in worse["candidates"]:
        if cand["id"] == "C2":
            cand["teacher_overlay"] = {"KRABBY_OMNI_SLOTS": "0.5:0.5:0:0"}
    out = [v for v in rc.validate_candidates(worse, strict=False, check_replay=False)
           if v["id"] == "C2"][0]
    assert not out["ok"] and any("SLOTS_TEACHER" in r for r in out["reasons"]), out["reasons"]


def test_a_milestone_entry_is_squashed_when_its_recipe_moves(rc, cands):
    """SQUASH (2): "plus milestones built on a changed recipe". The ENTRY, not only the promotion
    header -- a milestone carried forward records no experiment at all, and an ``open`` entry
    outlives the header that was squashed under it."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    c2 = entry("D_C2", "C2", "ADOPT", "decided", experiment="X03")
    led["entries"].append(c2)
    recipe_id = rid(rc, cands, led, "C3b", inclusive=True)
    carried = {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "PASS",
               "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
               "evidence": {"carried_from": None}, "recipe_id": recipe_id, "eligibility": 1,
               "promoted_head": {"2a": {}, "2b": {}}}
    led["entries"].append(carried)
    assert "M1" not in rc.squash_view(led, cands, st, code=CODE)["entries"]
    led["entries"][-2]["status"] = "overruled"
    led["entries"][-2]["override"] = {"by": "user", "choice": "REJECT", "why": "overrule"}
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "M1" in view["entries"] and "milestone recipe changed" in view["entries"]["M1"]
    # an OPEN entry goes the same way, and the queue then re-runs (i) on the reduced recipe
    led["entries"][-1].update(status="open", chosen=None)
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "M1" in view["entries"]
    item = rc.next_item(cands, led, st, code=CODE, view=view)
    assert item["label"] in ("M1.(i)", "M1.s2 s2"), item
    # the rule is re-derived from the recipe on EVERY pass, stored status included -- otherwise the
    # entry would drop out of the view as soon as its status was persisted and squash_and_report
    # would resurrect it on the next override
    led["entries"][-1].update(status="squashed", pre_squash_status="open", squashed_by="derived")
    assert "M1" in rc.squash_view(led, cands, st, code=CODE)["entries"]
    # and it comes back when the overrule is reverted (the recipe matches again)
    led["entries"][-2]["override"]["choice"] = "ADOPT"
    assert "M1" not in rc.squash_view(led, cands, st, code=CODE)["entries"]


def test_a_promotion_with_an_unreadable_row_is_watch_not_pass(rc, cands, tmp_path, monkeypatch):
    """FAIL is ">= 2 of the three 2b rows WORSE"; a row NOTHING could read is not evidence that it
    is fine. Two unreadable rows plus one good one would otherwise PASS the milestone on one row."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    entry_m1 = {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": None, "status": "open",
                "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {},
                "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M1", ctx.code)["recipe_id"],
                "eligibility": 1}
    ctx.ledger["entries"].append(entry_m1)
    header = {"xid": "X09", "kind": "milestone", "candidate": "M1", "seed": "3", "chain": "scratch_1c",
              "code_sha": ctx.code, "recs": ["X09:2a", "X09:2b"], "label": "M1.promo s3"}
    for stage in ("2a", "2b"):
        ctx.c.recs[f"X09:{stage}"] = {"ckpt": f"/sim/X09/{stage}.pt",
                                      "evals": {row: {"run_dir": f"/gone/{stage}/{row}"}
                                                for row in ("slow", "light", "ring")}}
    monkeypatch.setattr(rc, "_previous_promoted",
                        lambda ctx_, mid: ({"2a": {"evals": {}}, "2b": {"evals": {
                            "slow": "/gone/prev/slow"}}}, "M0"))
    item = {"kind": "milestone", "candidate": "M1", "recipe": {"adopted": []}, "label": "M1.promo s3"}
    rc._close_milestone_promo(ctx, item, header)
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert (m1["chosen"], m1["status"]) == ("WATCH", "final")
    assert m1["evidence"]["promotion"]["missing_rows"] == ["slow", "light", "ring"]
    assert any("could not be read" in line for line in notify_lines(ctx))


def test_a_seed_two_header_links_itself_to_its_decision_the_moment_it_is_written(rc, cands, tmp_path,
                                                                                monkeypatch):
    """Decision 7 / derived-view cause (e): a user ``confirm`` drops the candidate's own seed-2 run
    "if it is in flight". The only link to that run is ``ledger[D].seed2_experiment``, so it has to
    exist while the run is IN FLIGHT -- writing it at pair resolution leaves the 3.9 h window
    unreachable, the confirm silently ineffective and the next launch resuming it."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", confidence="uncertain",
                                experiment="X03", base=b0["recipe_id"], recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            b0["recipe_id"], base_rank("C1"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.entry_by_id(ctx.ledger, "D05").get("seed2_experiment") is None
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "seed2" and item["decision"] == "D05"
    # the header is written, then the child is stopped: the link must already be in the ledger
    install_fakes(rc, monkeypatch, tmp_path, interrupt="stop")
    monkeypatch.setattr(rc, "interrupt_reason", _stop_after(1))
    header = rc.run_experiment(ctx, item)
    xid = header["xid"]
    assert rc.entry_by_id(ctx.ledger, "D05")["seed2_experiment"] == xid
    monkeypatch.undo()
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm", "choice": "ADOPT",
                                    "why": "seen enough"})
    assert rc.ingest_overrides(ctx) is True
    assert ctx.lad["squashed"][xid]["squashed_by"] == "user_choice"
    assert xid not in ctx.lad["experiments"]
    assert rc.find_inflight(ctx) is None, "the pair run is not resumed"
    assert rc.interrupt_reason(ctx, xid)


def test_an_in_flight_seed_two_header_resumes_and_resolves_its_pair(rc, cands, tmp_path, monkeypatch):
    """The plan's named resume case ("the resume branch ... with a seed-2 header"): a live pair run
    goes through find_inflight -> item_from_header -> run_experiment -> dispatch and ends in a
    pooled, TERMINAL decision rather than an illegal transition."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D05", "C2", "ADOPT", "speculative", confidence="uncertain",
                                experiment="X03", base=b0["recipe_id"], recipe=r_c2["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                            b0["recipe_id"], base_rank("C2"),
                                            screen_=screen(closed=0.545, prog=0.125))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=screen(closed=0.545, prog=0.125),
                  interrupt="stop")
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "seed2"
    monkeypatch.setattr(rc, "interrupt_reason", _stop_after(1))
    header = rc.run_experiment(ctx, item)
    assert header["status"] == "running" and header["inflight"]
    monkeypatch.undo()
    install_fakes(rc, monkeypatch, tmp_path, eval_summary=screen(closed=0.545, prog=0.125))
    live = rc.find_inflight(ctx)
    assert live is not None and live["xid"] == header["xid"]
    rc.prepare_resume(ctx, live)
    resumed = rc.item_from_header(ctx, live)
    assert resumed["kind"] == "seed2" and resumed["decision"] == "D05"
    done = rc.run_experiment(ctx, resumed)
    assert done["status"] == "ok"
    rc._after_experiment(ctx, resumed, done)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] in ("final", "overruled")
    assert d05["seed2_experiment"] == header["xid"]
    assert d05["pooled_confidence"] is not None


def test_the_pooled_line_reads_s3_plus_s2_and_cites_its_levels(rc, cands, tmp_path):
    """Q2 + F9, both in the one field the user reads before confirming a pooled outcome.

    The pooled work ITEM is the seed-2 run, so ``item['seed']`` is already the confirm seed and the
    header rendered "s2 + s2"; and ``decide_pooled``'s evidence carried no ``levels``, so the line
    printed ``@None/None`` -- decision 6 says "Levels are recorded and cited in EVERY DECISION line".
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"), decision="D05")
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain",
                                base=b0["recipe_id"], recipe=r_c1["recipe_id"],
                                control_experiment="X01"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc._item("pooled", cid="C1", seed=ctx.seed_confirm, base_recipe=b0, decision="D05",
                    why="t", label="C1 s2 pooled")
    rc._resolve_pair(ctx, item, rc.entry_by_id(ctx.ledger, "D05"),
                     ctx.lad["experiments"]["X04"]["screen"], s2_xid="X04",
                     header=ctx.lad["experiments"]["X04"])
    line = [l for l in notify_lines(ctx) if " DECISION " in l or " SELF-OVERRULE " in l][-1]
    fields = rc.lc.parse_notify(line)["fields"]
    assert fields[0].endswith("s3 + s2"), fields[0]
    assert "s2 + s2" not in fields[0]
    assert any(f.startswith("resolution ") and f.endswith(" @0.95/0.99") for f in fields), fields
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    levels = d05["evidence"]["levels"]
    assert levels["blocks"] == {"F": 0.95, "T": 0.95} and levels["guards"] == 0.99
    assert set(levels["pooled_R"]) == {"F", "T"}


def test_a_pair_that_lands_after_the_user_settled_it_is_recorded_not_applied(rc, cands, tmp_path):
    """A frozen decision is never moved by a later pair (decision 7). If a pair run finishes anyway
    the screen is a RECORD -- not the illegal ``confirmed -> final`` move."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "confirmed", frozen=True, experiment="X03",
                                base=b0["recipe_id"], recipe=r_c1["recipe_id"],
                                override={"by": "user", "choice": "ADOPT", "why": "confirm"}))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    entry_d05 = rc.entry_by_id(ctx.ledger, "D05")
    item = rc._item("seed2", cid="C1", seed="2", decision="D05")
    rc._resolve_pair(ctx, item, entry_d05, screen(), s2_xid="X07")
    assert entry_d05["status"] == "confirmed", "the user's answer stands"
    assert entry_d05["evidence"]["late_pair"]["experiment"] == "X07"
    assert any("WATCH" in line and "D05" in line for line in notify_lines(ctx))


def test_the_smoke_header_is_resumable(rc, cands, tmp_path, monkeypatch):
    """Verification 8's second half. The smoke stage is in no chain, so the work item can only be
    rebuilt from the header's own stage list -- without it the resume ran zero stages and the drill
    read as a PASS while the resume half never happened."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    install_fakes(rc, monkeypatch, tmp_path, interrupt="stop")
    monkeypatch.setattr(rc, "interrupt_reason", _stop_after(1))
    header = rc.run_experiment(ctx, rc.smoke_item(ctx, "1a:20"))
    assert header["status"] == "running" and header["inflight"]["stage"] == "1a"
    monkeypatch.undo()
    calls = install_fakes(rc, monkeypatch, tmp_path)
    live = rc.find_inflight(ctx)
    assert live is not None
    rc.prepare_resume(ctx, live)
    resumed = rc.item_from_header(ctx, live)
    assert [str(s["stage"]) for s in resumed["stages"]] == ["1a"]
    assert resumed["stages"][0]["iters"] == 20
    done = rc.run_experiment(ctx, resumed)
    assert done["status"] == "ok"
    assert [c["iters"] for c in calls["train"]] == [20], "the window really ran on the resume"
    assert rc.entries(ctx.ledger) == []


def test_a_milestone_queues_the_previous_recipes_missing_screen_before_the_batched_decision(rc, cands,
                                                                                           tmp_path):
    """next_item step (3): the promotion runs "only when (i) is resolved". (i) pairs R_k against
    R_(k-1), and R_(k-1)'s screens are not covered by the B<k> wrapper -- after a code landing they
    stop being ok screens, and (i) would record WATCH and let a 4.3 h promotion run with `decided`
    adoptions never confirmed."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b", "C5", "C6", "C7"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    # R_1 = B0 + C2 (so it is NOT the baseline recipe, whose own screens step 1/2 already covers)
    r_1 = rc._recipe_with(led, cands, CODE, "C2", set())["recipe_id"]
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X05",
                                base=b0["recipe_id"], recipe=r_1))
    st["ladder"]["experiments"]["X05"] = exp("X05", "candidate", "C2", "3", r_1, b0["recipe_id"],
                                            base_rank("C2"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "PASS",
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {}, "recipe_id": r_1, "eligibility": 1,
                           "promoted_head": {"2a": {"evals": {}}, "2b": {"evals": {}}}})
    r_c4 = rc._recipe_with(led, cands, CODE, "C4", set())
    led["entries"].append(entry("D_C4", "C4", "ADOPT", "decided", experiment="X03",
                                base=r_1, recipe=r_c4["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C4", "3", r_c4["recipe_id"], r_1,
                                            base_rank("C4"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "batched_s2", "M2", "2", r_c4["recipe_id"],
                                            r_c4["recipe_id"], {"upto": "C7", "inclusive": True})
    # R_1 has no seed-2 screen: (i) cannot form its pair, and the plan says the promotion may not
    # run while a `decided` adoption is waiting for that confirmation.
    item = rc.next_item(cands, led, st, code=CODE)
    assert item["kind"] == "baseline" and item["seed"] == "2"
    assert item["recipe"]["recipe_id"] == r_1
    assert "M2 (i) pairs against M1's recipe" in item["why"]
    # with R_1's s2 screen present, (i) runs as the zero-GPU step it is
    st["ladder"]["experiments"]["X06"] = exp("X06", "twin", "C4", "2", r_1, r_1, base_rank("C4"))
    assert rc.next_item(cands, led, st, code=CODE)["label"] == "M2.(i)"


def test_a_user_skip_is_recorded_from_the_item_not_from_its_wording(rc, cands, tmp_path):
    """``evidence.user`` decides whether cause (b) may ever squash a SKIP, so the SOURCE comes from
    the item -- a predicate whose rendering contained the word "user" must not be immune."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    st["ladder"]["user_choice"] = {"C2": {"choice": "skip", "at": "2026-09-19T01:00:00",
                                          "uuid": "u1"}}
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    while item["candidate"] != "C2":
        ctx.ledger["entries"].append(entry(f"D_{item['candidate']}", item["candidate"], "REJECT",
                                           "decided"))
        ctx.recompute()
        item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["kind"] == "skip" and item["extra"]["source"] == "user"
    predicate = rc._item("skip", cid="C11", seed="3", why="predicate: the user-facing ratio < 0.45",
                         extra={"predicate": "the user-facing ratio < 0.45"})
    assert predicate["extra"].get("source") is None, "a predicate SKIP carries no user source"


def test_the_shipped_resolution_inputs_name_every_row_d00_needs(rc, cands):
    """Verification 5's input list, pinned: the three heads on the three rows, the three
    informational re-evals, and the same-head 2b re-eval FLAGGED as the process floor.

    Nothing else pins this file's ``resolution_inputs``, which is why the process-floor row could go
    missing without a test noticing -- and a D00 without it leaves ``process_floor`` {}.
    """
    inputs = cands["resolution_inputs"]
    rows = {(i["head"], i["row"]) for i in inputs}
    for head in ("1c-004", "2b", "2a-reeval"):
        for row in ("slow__omni", "light__omni", "ring__omni"):
            assert (head, row) in rows, (head, row)
    for head in ("1a-corrected", "1b-corrected", "1c-corrected"):
        assert (head, "slow__omni") in rows
        assert [i for i in inputs if i["head"] == head][0]["provenance_ok"] is True
        assert [i for i in inputs if i["head"] == head][0]["role"] == "informational"
    floor = [i for i in inputs if i.get("process_floor_pair") or i.get("role") == "process_floor"]
    assert len(floor) == 1, "exactly one same-head process-floor row"
    assert floor[0]["row"] == "slow__omni"
    assert floor[0]["floor_of"] == "2b", "it names the head it re-evaluates"
    for item in inputs:
        assert not Path(item["run_dir"]).is_absolute(), item["run_dir"]


def test_the_shipped_milestone_reference_names_the_recorded_2b_and_2a_rows(rc, cands):
    """M1's promotion is WATCH-only "vs the seed-3 2b record and vs outputs/omni_plan/reeval_2a"
    (plan, Part C (ii)) -- the driver cannot read rows the campaign never names."""
    ref = cands["milestone_reference"]
    assert set(ref) == {"2a", "2b"}
    for stage in ("2a", "2b"):
        assert set(ref[stage]) == {"slow", "light", "ring"}
        for rel in ref[stage].values():
            assert not Path(rel).is_absolute()
    assert all("reeval_2a" in v for v in ref["2a"].values())
    assert all("2b_006" in v for v in ref["2b"].values())
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts)


def test_m1_compares_the_promoted_heads_against_the_recorded_reference(rc, cands, tmp_path,
                                                                      monkeypatch):
    """The M1 WATCH rows are really computed, through ``resolution.py``'s shared re-score (so the
    recorded side has the F block's closed fractions too) -- not just named in the line."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    seen = {}

    def fake_summary(run_dir, sid):
        seen[str(run_dir)] = sid
        return screen(closed=0.60 if "head" in str(run_dir) else 0.50)

    monkeypatch.setattr(rc, "summarize_run_dir", fake_summary)
    monkeypatch.setattr(rc, "repo_path", lambda rel: Path(tmp_path) / "repo" / str(rel))
    for stage in ("2a", "2b"):
        for row in ("slow", "light", "ring"):
            d = Path(tmp_path) / "repo" / cands["milestone_reference"][stage][row]
            d.mkdir(parents=True, exist_ok=True)
    heads = {stage: {"ckpt": f"/sim/{stage}.pt",
                     "evals": {row: f"/head/{stage}/{row}" for row in ("slow", "light", "ring")}}
             for stage in ("2a", "2b")}
    rows, lines = rc._reference_watch(ctx, heads, {})
    assert set(rows) == {f"{s}/{r}" for s in ("2a", "2b") for r in ("slow", "light", "ring")}
    assert all(not v.get("missing") for v in rows.values())
    assert any("2b vs the recorded reference" in line for line in lines)
    # both sides were summarised, and the reference rows are the ones candidates.yaml names
    assert any(cands["milestone_reference"]["2b"]["slow"] in path for path in seen)
    assert seen["/head/2b/ring"] == "ring__omni"


def test_a_block_or_guard_with_no_resolution_is_flagged_not_a_silent_tie(rc, rule, cands, tmp_path):
    """A block or guard the record carries no half-width for can never read WORSE, so a REJECT on it
    is impossible. It classifies TIE (the caller decides), but the record and the DECISION line say
    it is unresolved -- a partial resolution.json must fail loud, not open."""
    thin = resol()
    thin["blocks"]["F"]["R_eval"] = None
    thin["metrics"]["completion"]["R_eval"] = None
    out = dec(rc, rule, screen(closed=0.72, prog=0.11, completion=0.10), screen(),
              resolution=thin)
    blocks = out["evidence"]["seed3"]["blocks"]
    assert blocks["F"]["resolution_missing"] is True
    assert blocks["T"]["resolution_missing"] is False
    # a GUARD with no half-width is not a silent TIE either: metric_rows leaves its delta None, so
    # it is MISSING, and decision 5 turns a MISSING guard into a REJECT
    assert out["evidence"]["seed3"]["guards"]["missing"] == ["completion"]
    assert (out["chosen"], out["confidence"]) == ("REJECT", "uncertain")
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st, resolution=thin)
    item = rc._item("candidate", cid="C2", seed="3", recipe={"overlay_by_stage": {}},
                    base_recipe={"recipe_id": "b"})
    entry_ = {"id": "D08", "candidate": "C2", "chosen": out["chosen"], "alternative": "REJECT",
              "confidence": out["confidence"], "status": out["status"], "experiment": "X03"}
    line = " | ".join(rc.decision_notify_fields(ctx, entry_, item, out))
    assert "NO RESOLUTION for block ['F (F)']" in line
    assert "MISSING guards: ['completion']" in line


def test_the_2a_rows_are_compared_not_just_named_from_m2(rc, cands, tmp_path, monkeypatch):
    """"The three 2a rows are WATCH at every milestone (vs the previous milestone's 2a head)" --
    WATCH means reported, so the rows are compared the same way the 2b rows are and the verdict
    reads nothing from them."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    entry_m1 = {"id": "M2", "kind": "milestone", "candidate": "M2", "chosen": None, "status": "open",
                "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {},
                "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M2", ctx.code)["recipe_id"],
                "eligibility": 1}
    ctx.ledger["entries"].append(entry_m1)
    header = {"xid": "X09", "kind": "milestone", "candidate": "M2", "seed": "3",
              "chain": "scratch_1c", "code_sha": ctx.code, "label": "M2.promo s3"}
    for stage in ("2a", "2b"):
        ctx.c.recs[f"X09:{stage}"] = {"ckpt": f"/sim/X09/{stage}.pt",
                                      "evals": {row: {"run_dir": f"/head/{stage}/{row}"}
                                                for row in ("slow", "light", "ring")}}
    prev = {s: {"evals": {row: f"/prev/{s}/{row}" for row in ("slow", "light", "ring")}}
            for s in ("2a", "2b")}
    monkeypatch.setattr(rc, "_previous_promoted", lambda ctx_, mid: (prev, "M1"))
    monkeypatch.setattr(rc, "summarize_run_dir",
                        lambda run_dir, sid: screen(tripod=0.20 if "/prev/2a/" in str(run_dir)
                                                    else 0.40))
    rc._close_milestone_promo(ctx, {"kind": "milestone", "candidate": "M2",
                                    "recipe": {"adopted": []}, "label": "M2.promo s3"}, header)
    m2 = rc.entry_by_id(ctx.ledger, "M2")
    watch = m2["evidence"]["promotion"]["watch_2a"]
    assert set(watch) == {"slow", "light", "ring"}
    assert all(v["worse"] == [] for v in watch.values()), "the 2a head is ahead of M1's here"
    assert (m2["chosen"], m2["status"]) == ("PASS", "final"), "2a rows never decide the milestone"
    line = [l for l in notify_lines(ctx) if "MILESTONE" in l][-1]
    assert "2a rows are WATCH at every milestone" in line and "vs M1 ok" in line


# ================================================================== audit round 2: the four blockers
def test_a_promotion_is_never_mistaken_for_the_recipes_own_screen(rc, cands, tmp_path):
    """Vocabulary: an "ok screen" is the recipe's OWN seed-3 screen (8000 it), and the recipe head
    is "the final-stage checkpoint of the ok s3 screen".

    A milestone promotion runs the PROMOTION stage list (2a then 2b, ``EVAL_FULL`` at both) under
    the SAME ``recipe_id``, the same seed and a LATER ``started`` -- so without a kind filter it
    became the newest "ok screen" of the recipe it promotes, and every consumer moved onto a 2b row:
    the next candidate's control, the head ``M<k+1>.promo``'s 2a stage resumes from, both halves of
    a pooled pair, the ``D00b``/``D00c`` lookups and ``screen.<key>`` in C10/C11's predicates."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    promo = exp("X05", "milestone", "M1", "3", b0["recipe_id"], b0["recipe_id"],
                {"upto": "C3b", "inclusive": True}, screen_=screen(completion=0.88))
    promo["started"] = "2026-09-22T00:00:00"          # later than X01's
    promo["ckpt"] = "/sim/X05_2b/model_1899.pt"
    st["ladder"]["experiments"]["X05"] = promo
    s2_promo = exp("X06", "milestone_s2", "M3", "2", b0["recipe_id"], b0["recipe_id"],
                   {"upto": "C12", "inclusive": True})
    s2_promo["started"] = "2026-09-23T00:00:00"
    st["ladder"]["experiments"]["X06"] = s2_promo

    assert rc.ok_screen(st, CODE, b0["recipe_id"], "3")["xid"] == "X01"
    assert rc.ok_screen(st, CODE, b0["recipe_id"], "2")["xid"] == "X02"
    assert rc.recipe_head(st, CODE, b0["recipe_id"], "3") == "/sim/X01/model_7999.pt"
    assert rc.recipe_head(st, CODE, b0["recipe_id"], "2") == "/sim/X02/model_7999.pt"
    # and the predicate context reads the SCREEN's numbers, not the promotion's 2b row
    ctx = rc.predicate_context(cands, led, st, b0, CODE)
    assert ctx["screen"]["completion"] == 0.50
    # the D00c lookup (`ok_runs`) still sees every kind -- it only needs an ok RUN
    assert {h["xid"] for h in rc.ok_runs(st, CODE)} == {"X01", "X02", "X05", "X06"}
    assert set(rc.SCREEN_KINDS) == {"baseline", "candidate", "twin", "seed2", "batched_s2"}


def test_an_interrupt_during_the_experiment_level_retry_is_an_interrupt_not_a_config_halt(
        rc, cands, tmp_path, monkeypatch):
    """Main loop: "status interrupted -> exp.status=interrupted; break" -- after the retry too.

    The stop lands while the ONE experiment-level retry of an infra death runs. Without the second
    interrupt test control fell through with ``ckpt is None``, ``overlay_checks`` answered "overlay
    audit needs a checkpoint" and the driver halted ``config_mismatch``: a HALT blaming a mismatch
    that did not happen, a header left unresumable, and the stop drill unreadable after any infra
    death (which on this box is a recurring event)."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path, status="infra")
    fake = rc.rp.train_retry

    def flaky(c, spec, tag, iters, resume, soundness, early_abort=False, **kw):
        ckpt, _status, log = fake(c, spec, tag, iters, resume, soundness, early_abort, **kw)
        if len(calls["train"]) == 1:
            return ckpt, "infra", log
        (ctx.dir / rc.lc.STOP_NAME).write_text("stopped mid-retry\n")   # the user stops the unit
        return None, "interrupted", log

    monkeypatch.setattr(rc.rp, "train_retry", flaky)
    header = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert len(calls["train"]) == 2, "the stage took its one experiment-level retry"
    assert header["interrupt"] == "stop"
    assert header["status"] == "running", "a stop leaves the window RESUMABLE"
    assert header["inflight"] is not None
    assert not str(ctx.state.get("phase", "")).startswith("halt")
    assert not any("HALT" in line for line in notify_lines(ctx))
    (ctx.dir / rc.lc.STOP_NAME).unlink()
    ctx.recompute()
    assert (rc.find_inflight(ctx) or {}).get("xid") == header["xid"]


def test_a_resumed_window_disarms_the_soundness_and_dead_walker_gates(rc, cands, tmp_path,
                                                                     monkeypatch):
    """``run_phase`` arms both on ``resume_ckpt is None`` and says why: the episode-length-growth
    ratio ``dead_plant`` reads is biased on a resumed window and the verdict has no escape hatch,
    and ``dead_walker`` would measure the CONTINUATION's own its-800-1000 against a floor recorded
    for an absolute its-800-1000 -- turning a user's stop into a ``rejected_by_floor`` REJECT."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    ctx.lad["d03"] = 0.42
    calls = install_fakes(rc, monkeypatch, tmp_path)

    # the control: the same 8000-it from-scratch window, NOT resumed
    rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    fresh = calls["train"][0]
    assert (fresh["iters"], fresh["resume"]) == (8000, None)
    assert fresh["soundness"] is True and fresh["early_abort"] is True

    calls["train"].clear()
    item = candidate_item(rc, cands, ctx, "C2")
    recipe = item["recipe"]
    header = {"xid": "X07", "kind": "candidate", "candidate": "C2", "seed": "3", "label": "C2 s3",
              "chain": recipe["chain"], "base_recipe_id": item["base_recipe"]["recipe_id"],
              "recipe_id": recipe["recipe_id"], "base_rank": base_rank("C2"), "code_sha": ctx.code,
              "overlay_by_stage": {"1c": rc.stage_env(recipe, "1c")},
              "started": "2026-09-19T00:00:00", "status": "running", "recs": [], "eval_dirs": [],
              "stages": ["1c"],
              "inflight": {"stage": "1c", "iters": 8000, "tag": "seed3_X07_1c_A15pB",
                           "resume": None}}
    ctx.lad["experiments"]["X07"] = header
    run_dir = Path(tmp_path) / "logroot" / rc.ph.PHASES["1c"].experiment / "26_seed3_X07_1c_A15pB"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_1999.pt").write_text("x")
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["remaining"] == 6001
    rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    cont = calls["train"][0]
    assert cont["iters"] == 6001 and cont["resume"].endswith("model_1999.pt")
    assert cont["soundness"] is False, "a resumed window must not be judged by dead_plant"
    assert cont["early_abort"] is False, "nor aborted on the continuation's own its-800-1000"
    assert cont["abort_floor"] is None


@pytest.mark.parametrize("floor_stage,holds", [(None, True), ("c1", False), ("c2", False),
                                               ("c3", False), ("c4", True), ("c5", True),
                                               ("c6", True)])
def test_c1b_is_queued_exactly_when_the_curriculum_tripped_a_move_turn_floor(rc, cands, floor_stage,
                                                                            holds):
    """C1b's whole purpose (plan, Part C): "c4-c6 use world-held move+turn at the 0.3 rad/s cap
    (infeasible); C1b is the reduced-demand re-screen", gated on
    ``decision(C1).floor_stage is None or decision(C1).floor_stage >= c4``.

    The stage names being compared belong to the CURRICULUM chain, but the predicate is evaluated
    at C1b's rank -- where a rejected C1 leaves the base recipe on ``scratch_1c`` (stages
    ``["1c"]``). A stage order taken from the base recipe alone therefore resolved neither name and
    every ``>= c4`` read FALSE, writing C1b off as a SKIP in exactly the case it exists for."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "REJECT", "decided", experiment="X03",
                                base=rid(rc, cands, led, "C1"), floor_stage=floor_stage))
    base = rc.compose_recipe(led, cands, CODE, upto="C1b")
    assert [s["stage"] for s in base["stages"]] == ["1c"], "the base is B0 after a rejected C1"
    ctx = rc.predicate_context(cands, led, st, base, CODE)
    cand = rc.candidate_by_id(cands, "C1b")
    got, why = rc.compile_predicate(cand["conditional"]).evaluate(ctx)
    assert got is holds, why
    # the per-chain lists are published, and a flat union would order scratch_c6's `c6` first
    orders = ctx["stage_orders"]
    assert ["c1", "c2", "c3", "c4", "c5", "c6"] in orders and ["c6"] in orders


def test_a_pooled_adopt_that_lost_block_members_cannot_become_the_anchor(rc, rule, cands):
    """Decision 5: "MISSING in F/T -> that metric leaves the block mean and the best outcome is a
    SPECULATIVE ADOPT". A pooled outcome is terminal, so the cap lands on ``pooled_confidence`` --
    which matters because ``_anchor_qualifies`` makes a ``pooled_confidence: clear`` ADOPT the
    campaign's anchor, and decision 5 makes the anchor the precondition every later ADOPT is judged
    against."""
    good = screen(closed=0.72, prog=0.11)
    gone = screen(closed=0.72, prog=0.11)
    for key in ("facing_closed_by_hold", "yaw_progress_by_hold"):
        gone[key]["move_turn"] = None              # the hold fell in every episode of both seeds
    whole = pooled(rc, rule, good, good, screen(), screen(), prior_pick="ADOPT")
    assert (whole["chosen"], whole["pooled_confidence"]) == ("ADOPT", "clear")
    assert rc._anchor_qualifies({"status": "final", "chosen": "ADOPT",
                                 "pooled_confidence": "clear"}) is True

    capped = pooled(rc, rule, gone, gone, screen(), screen(), prior_pick="ADOPT")
    assert capped["chosen"] == "ADOPT", "the ADOPT stands -- there is no further pair"
    assert capped["pooled_confidence"] == "uncertain"
    assert capped["evidence"]["missing"]["blocks"] == ["facing_closed_by_hold.move_turn",
                                                       "yaw_progress_by_hold.move_turn"]
    assert "capped at uncertain" in capped["why"]
    assert rc._anchor_qualifies({"status": "final", "chosen": "ADOPT",
                                 "pooled_confidence": "uncertain"}) is False


def test_a_drop_of_an_earlier_adoption_restarts_the_milestone_rerun_allowance(rc, cands):
    """"One automatic rerun per eligibility ... counted by DECISION entries carrying
    ``override {by: milestone}`` NEWER than the last user override on any block member; eligibility
    restarts on a user ``--drop``."

    The milestone's automatic drop always takes the block's LAST unfrozen adoption, so the natural
    user answer is a ``--drop`` of an EARLIER one -- a lower ledger index. Counted positionally that
    answer did not restart the allowance and the next contradiction parked for the user instead of
    taking its rerun, making ``--drop`` silently weaker than ``--accept``."""
    def build(user_at):
        led = ledger()
        led["entries"].append(entry("D05", "C1", "ADOPT", "overruled", override={
            "by": "user", "choice": "REJECT", "why": "M1 drop C1", "at": user_at}))
        led["entries"].append(entry("D06", "C2", "ADOPT", "decided"))
        led["entries"].append(entry("D07", "C3a", "ADOPT", "overruled", override={
            "by": "milestone", "choice": "REJECT", "why": "M1 batched pair contradicts",
            "at": "2026-09-20T10:00:00"}))
        return led

    later = build("2026-09-20T12:00:00")            # the user answered AFTER the milestone's drop
    assert rc.milestone_rerun_used(later, "M1", cands) is False
    earlier = build("2026-09-20T08:00:00")          # an older user override does not restart it
    assert rc.milestone_rerun_used(earlier, "M1", cands) is True
    # ... and the ledger POSITION of the two entries is the same in both (D05 before D07)
    assert [e["id"] for e in rc.entries(later)] == [e["id"] for e in rc.entries(earlier)]


def test_an_exception_from_the_eval_seam_is_a_named_halt_not_a_traceback(rc, cands, tmp_path,
                                                                        monkeypatch):
    """Vocabulary: "halt -- the driver exits 1 with ``phase = halt:<xid> <why>`` (infra, nan,
    dead_plant, eval_infra, config/eval mismatch)".

    ``run_phases._plant_guard`` raises when an eval spawned the wrong plant (the rung-iv defect).
    Uncaught it escaped ``run_experiment`` and ``loop``: a traceback, ``phase`` still reading
    ``run``, the header still ``running`` with its ``inflight`` block -- so the next launch resumed
    the same stage and re-raised, a relaunch loop on a deterministic fault."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("seed3_X03_1c_A15pB: eval spawned plant golden but A15+B requested "
                           "(rung-iv defect)")

    monkeypatch.setattr(rc.rp, "eval_morph", boom)
    with pytest.raises(SystemExit) as exc:
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert exc.value.code == 1, "a halt exits 1, it does not propagate the RuntimeError"
    header = ctx.lad["experiments"]["X03"]
    assert header["status"] == "halt:eval_mismatch"
    assert str(ctx.state["phase"]).startswith("halt:X03 eval_mismatch")
    assert "rung-iv defect" in header["halt"]
    assert any("HALT" in line and "rung-iv" in line for line in notify_lines(ctx))


def test_the_derived_view_survives_a_real_crash_and_reload(rc, cands, tmp_path):
    """SQUASH (6): "crash-safe: the squash set is a derived view recomputed from the ledger on
    every launch". Comparing two calls on deep copies of one in-memory ledger is trivially true for
    a pure function; the property that matters is that a BRAND-NEW ``Ctx`` built off the files the
    driver wrote agrees with the one that wrote them -- including the ``why`` strings, which name a
    recipe that is itself a function of the squash set and so can be written mid-fixpoint."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base_c1, base_c2 = rid(rc, cands, led, "C1"), rid(rc, cands, led, "C2")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain", base=base_c1,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                            base_c1, base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base_c1,
                                            base_rank("C1"), status="running", decision="D05",
                                            inflight={"stage": "c1", "iters": 1000, "tag": "t"})
    st["ladder"]["experiments"]["X05"] = exp("X05", "candidate", "C2", "3", "r_c2", base_c2,
                                            base_rank("C2"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    # a real sequence, applied the way the driver applies it (files on disk change)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm", "choice": "ADOPT",
                                    "why": "user has seen enough"})
    rc.lc.append_override(ctx.dir, {"candidate": "C6", "choice": "skip", "why": "not now"})
    rc.ingest_overrides(ctx)
    before = {"view": json.loads(json.dumps(ctx.view, default=str)),
              "statuses": {e["id"]: (e.get("status"), e.get("sub_status"),
                                     rc.lc.effective_choice(e)) for e in rc.entries(ctx.ledger)},
              "live": sorted(ctx.lad["experiments"]), "dead": sorted(ctx.lad["squashed"]),
              "pending": list(ctx.lad["pending"]),
              "next": (rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code,
                                    view=ctx.view) or {}).get("label")}

    args = types.SimpleNamespace(campaign_dir=str(ctx.dir), candidates=None, unit=None,
                                 dry_run=False, max_wall_h=None, accept_code_change=None,
                                 smoke_stage=None, accept_candidates_change=None)
    reloaded = rc.Ctx(ctx.dir, cands, args)              # a fresh process, nothing in memory
    reloaded.code, reloaded.tools = CODE, "TOOLS"
    reloaded.resolution = resol()
    reloaded.recompute()
    after = {"view": json.loads(json.dumps(reloaded.view, default=str)),
             "statuses": {e["id"]: (e.get("status"), e.get("sub_status"),
                                    rc.lc.effective_choice(e)) for e in rc.entries(reloaded.ledger)},
             "live": sorted(reloaded.lad["experiments"]), "dead": sorted(reloaded.lad["squashed"]),
             "pending": list(reloaded.lad["pending"]),
             "next": (rc.next_item(cands, reloaded.ledger, reloaded.state, code=reloaded.code,
                                   view=reloaded.view) or {}).get("label")}
    assert before == after
    # X04 by the user (cause e, never resurrected); X05 because C1's confirmed ADOPT moved C2's base
    assert before["dead"] == ["X04", "X05"]
    assert ctx.lad["squashed"]["X04"]["squashed_by"] == "user_choice"
    assert ctx.lad["squashed"]["X05"]["squashed_by"] == "derived"
    assert reloaded.view["experiments"]["X05"].startswith("base recipe changed")
    assert before["pending"] == []


def test_decide_run_on_a_speculative_decision_drops_its_own_in_flight_pair(rc, cands, tmp_path):
    """Vocabulary: ``decide <Cid> --choose run`` "squashes the effective entry whatever its status
    ... (if that entry was ``speculative``, the candidate's OWN seed-2 run is removed from
    ``pending`` and, if in flight, interrupted and squashed, ``squashed_by: user_choice``; the base
    recipe's twin continues)". The ``confirm`` half of this was pinned; the ``decide run`` half was
    the untested branch of ``_apply_candidate_choice``."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain", base=base,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                            base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base,
                                            base_rank("C1"), status="running", decision="D05",
                                            inflight={"stage": "c1", "iters": 1000, "tag": "t"})
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C9", "2", base, base, base_rank("C9"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.interrupt_reason(ctx, "X04") is None
    rc.lc.append_override(ctx.dir, {"candidate": "C1", "choice": "run", "why": "re-screen it"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "squashed" and d05["override"]["why"] == "decide run"
    assert ctx.lad["squashed"]["X04"]["squashed_by"] == "user_choice"
    assert ctx.lad["squashed"]["X04"]["status"] == "interrupted", "the archived copy is not 'running'"
    assert "X04" not in ctx.lad["experiments"]
    assert "X05" in ctx.lad["experiments"], "the base recipe's twin is never dropped"
    assert "X03" in ctx.lad["experiments"], "the candidate's own s3 screen stays a record"
    assert ctx.lad["pending"] == [] and rc.find_inflight(ctx) is None
    assert "C1" in ctx.view["candidates"]
    assert rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)["label"] == "C1 s3"


def test_an_out_of_rank_rerun_is_screened_on_its_rank_recipe_and_squashes_only_what_it_moves(
        rc, cands, tmp_path):
    """The plan's named ``next_item`` scenario: "C5 decided REJECT, C6 clear ADOPT, M2 final ->
    decide C5 run -> C5 re-runs on R(<= C4) with that recipe's ok s3 screen as control and is not
    self-squashed; ADOPT -> C6's, C7's, M2's and later screens squashed by the global view and
    re-screened (M2 not eligible until they land), the re-run's screen is the recipe head;
    REJECT -> nothing squashed"."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    for i, cid in enumerate(["C1", "C1b", "C1c", "C2", "C3a", "C3b", "C4"]):
        led["entries"].append(entry(f"D{5 + i:02d}", cid, "REJECT", "decided",
                                    base=rid(rc, cands, led, cid)))
    base_c5 = rid(rc, cands, led, "C5")
    assert base_c5 == b0["recipe_id"], "every earlier candidate was rejected, so R(<= C4) is B0"
    led["entries"].append(entry("D12", "C5", "REJECT", "decided", experiment="X07", base=base_c5))
    st["ladder"]["experiments"]["X07"] = exp("X07", "candidate", "C5", "3", "r_c5", base_c5,
                                            base_rank("C5"))
    r_c6 = rc._recipe_with(led, cands, CODE, "C6", set())
    led["entries"].append(entry("D13", "C6", "ADOPT", "decided", experiment="X08",
                                base=rid(rc, cands, led, "C6"), recipe=r_c6["recipe_id"]))
    st["ladder"]["experiments"]["X08"] = exp("X08", "candidate", "C6", "3", r_c6["recipe_id"],
                                            rid(rc, cands, led, "C6"), base_rank("C6"))
    led["entries"].append(entry("D14", "C7", "SKIP", "skipped", base=rid(rc, cands, led, "C7"),
                                evidence={"predicate": "recipe.has_pool"}))
    m1_recipe = rc.milestone_recipe(cands, led, "M1", CODE)
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "WATCH",
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {}, "recipe_id": m1_recipe["recipe_id"], "eligibility": 1})
    m2_recipe = rc.milestone_recipe(cands, led, "M2", CODE)
    led["entries"].append({"id": "M2", "kind": "milestone", "candidate": "M2", "chosen": "PASS",
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {}, "recipe_id": m2_recipe["recipe_id"], "eligibility": 1})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)

    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen it"})
    rc.ingest_overrides(ctx)
    assert rc.entry_by_id(ctx.ledger, "D12")["status"] == "squashed"
    assert "X07" not in ctx.view["experiments"], "its own screen stays a record, not a squash"
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["label"] == "C5 s3" and item["kind"] == "candidate"
    assert item["base_recipe"]["recipe_id"] == b0["recipe_id"], "the RANK-composed base, not R(C6)"
    assert item["base_recipe"]["adopted"] == [], "C6's adoption is ranked after C5"
    assert rc.ok_screen(ctx.state, ctx.code, item["base_recipe"]["recipe_id"], "3")["xid"] == "X01"

    # ---- the ADOPT branch
    adopt = copy.deepcopy(ctx.ledger)
    adopt_st = copy.deepcopy(ctx.state)
    r_c5 = rc._recipe_with(adopt, cands, CODE, "C5", set(ctx.view["entries"]))
    adopt["entries"].append(entry("D15", "C5", "ADOPT", "decided", experiment="Y01",
                                  base=b0["recipe_id"], recipe=r_c5["recipe_id"]))
    adopt_st["ladder"]["experiments"]["Y01"] = exp("Y01", "candidate", "C5", "3",
                                                  r_c5["recipe_id"], b0["recipe_id"],
                                                  base_rank("C5"))
    view = rc.squash_view(adopt, cands, adopt_st, code=CODE, check_files=False)
    assert "Y01" not in view["experiments"], "the re-run is not self-squashed"
    assert {"X08"} <= set(view["experiments"]), "C6's screen was measured on a base that moved"
    assert {"D13", "D14", "M2"} <= set(view["entries"])
    assert "M1" not in view["entries"], "M1's block is ranked before C5 and did not move"
    assert {"C6", "C7"} <= set(view["candidates"])
    assert rc.after_holds(cands, adopt, "M2", set(view["entries"]))[0] is False
    assert rc.recipe_head(adopt_st, CODE, r_c5["recipe_id"], "3", view) == "/sim/Y01/model_7999.pt"
    assert rc.next_item(cands, adopt, adopt_st, code=CODE, view=view)["label"] == "C6 s3"

    # ---- the REJECT branch: the recipe never moved, so nothing is squashed
    rej = copy.deepcopy(ctx.ledger)
    rej["entries"].append(entry("D15", "C5", "REJECT", "decided", experiment="Y01",
                                base=b0["recipe_id"]))
    rej_st = copy.deepcopy(ctx.state)
    rej_st["ladder"]["experiments"]["Y01"] = exp("Y01", "candidate", "C5", "3", "r_c5b",
                                                b0["recipe_id"], base_rank("C5"))
    rej_view = rc.squash_view(rej, cands, rej_st, code=CODE, check_files=False)
    assert rej_view["experiments"] == {} and rej_view["candidates"] == []
    assert set(rej_view["entries"]) == {"D12"}, "only the entry the user squashed"


def test_a_guard_the_bootstrap_could_not_resolve_is_scored_on_its_seed_term(rc, rule):
    """Decision 6 spells a guard's resolution ``max(R_eval,i, R_seed,i)`` and a guard is compared in
    its OWN units, so a guard with only a seed term is scorable. MISSING is about a missing VALUE
    (a zero-denominator conditional survival), not about a missing half-width -- a BLOCK member
    with no ``R_eval`` does leave the mean, because the block statistic IS ``delta/R_eval,i``."""
    thin = resol()
    thin["metrics"].pop("tripod_walking_dir")
    no_terms = rc.decide(screen(tripod=0.38), screen(), thin, cand_spec={}, rule=rule,
                         seed_terms={})
    assert no_terms["evidence"]["missing"]["guards"] == ["tripod_walking_dir"]
    assert no_terms["chosen"] == "REJECT" and no_terms["confidence"] == "uncertain"

    terms = {"metrics": {"tripod_walking_dir": 0.05}}
    inside = rc.decide(screen(tripod=0.38), screen(), thin, cand_spec={}, rule=rule,
                       seed_terms=terms)
    assert inside["evidence"]["missing"]["guards"] == [], "the seed term resolves it"
    rows = {r["metric"]: r for r in inside["evidence"]["seed3"]["guards"]["rows"]}
    assert rows["tripod_walking_dir"]["R"] == pytest.approx(0.05)
    assert rows["tripod_walking_dir"]["R_from_seed_term_only"] is True
    assert rows["tripod_walking_dir"]["class"] == "TIE"          # -0.02 inside +/-0.05
    outside = rc.decide(screen(tripod=0.30), screen(), thin, cand_spec={}, rule=rule,
                        seed_terms=terms)
    assert outside["chosen"] == "REJECT" and outside["confidence"] == "clear"
    # a block member with no R_eval still leaves the mean, whatever its seed term
    thin_block = resol()
    thin_block["metrics"].pop("ratio_by_hold.back")
    blocked = rc.decide(screen(), screen(), thin_block, cand_spec={}, rule=rule,
                        seed_terms={"metrics": {"ratio_by_hold.back": 0.05}})
    assert blocked["evidence"]["missing"]["blocks"] == ["ratio_by_hold.back"]


def test_a_predicate_skip_is_squashed_when_its_base_recipe_moves_but_a_user_skip_is_not(rc, cands):
    """Derived-view cause (a) is written for "an experiment OR SKIP ENTRY", and SQUASH (2) repeats
    it ("experiments AND SKIP ENTRIES whose ``base_recipe_id`` differs from the recomputed base").
    A predicate SKIP is a statement about a recipe, so a moved base invalidates the statement even
    when the predicate still reads FALSE -- the candidate is re-evaluated at its rank and a fresh
    SKIP records the base it was actually judged on, for zero GPU. A USER skip is carved out for
    the same reason clause (b) carves it out: it answers about the CANDIDATE, not the recipe."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "REJECT", "decided", base=rid(rc, cands, led, "C1")))
    # C11's predicate reads a screen key, so it stays FALSE whatever C2 does
    led["entries"].append(entry("D06", "C11", "SKIP", "skipped", base=rid(rc, cands, led, "C11"),
                                conditional_refs=[],
                                evidence={"predicate": "screen.track_ratio_move_tail < 0.45",
                                          "user": False}))
    assert rc.squash_view(led, cands, st, code=CODE)["entries"] == {}
    led["entries"].append(entry("D07", "C2", "ADOPT", "decided", base=rid(rc, cands, led, "C2"),
                                recipe="r_c2"))
    view = rc.squash_view(led, cands, st, code=CODE)
    assert "D06" in view["entries"] and "base recipe changed" in view["entries"]["D06"]
    assert "C11" in view["candidates"], "and the candidate is undecided again"
    # the same entry, written because the USER said skip, is left alone
    [e for e in led["entries"] if e["id"] == "D06"][0]["evidence"]["user"] = True
    assert "D06" not in rc.squash_view(led, cands, st, code=CODE)["entries"]


def test_an_infra_halt_resumes_its_window_but_a_policy_verdict_does_not(rc, cands, tmp_path,
                                                                       monkeypatch):
    """Plan, Infra: "on an OOM/cache signature -> ... ``halt:<xid> eval_infra`` (the assistant
    quarantines the cache and relaunches; THE EXPERIMENT RESUMES AT THE EVAL)", and the HALT line
    for a twin / batched_s2 / milestone reads "infra only: quarantine/relaunch".

    ``_halt`` leaves ``inflight`` populated on purpose, but ``find_inflight`` only ever looked at
    ``status == "running"`` -- so a Kit DerivedDataCache corruption at the eval of an 8000-it screen
    threw ~3.7 GPU-hours away on every relaunch. ``nan`` / ``dead_plant`` / a config or eval
    mismatch are statements about the RUN and must not resume."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path, eval_none=1)
    monkeypatch.setattr(rc, "infra_signature", lambda *a, **k: ("oom_kill delta 1", ["cgroup"]))
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    header = ctx.lad["experiments"]["X03"]
    assert header["status"] == "halt:eval_infra" and header["inflight"]["stage"] == "1c"
    assert (rc.find_inflight(ctx) or {}).get("xid") == "X03"

    # ... and the relaunch continues THAT window: its training is done, so it goes to the eval
    run_dir = Path(tmp_path) / "logroot" / rc.ph.PHASES["1c"].experiment / f"26_{header['inflight']['tag']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_7999.pt").write_text("x")
    calls["train"].clear()
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["remaining"] == 1, "the window had already finished training"
    resumed = rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    assert calls["train"] == [], "the 8000-iteration window is not re-run"
    assert resumed["status"] == "ok" and resumed["xid"] == "X03"
    assert resumed["halts"][0]["status"] == "halt:eval_infra", "the halt is kept as a record"

    # a policy verdict is NOT resumable
    for verdict in ("halt:nan", "halt:dead_plant", "halt:config_mismatch", "halt:eval_mismatch"):
        resumed["status"] = verdict
        resumed["inflight"] = {"stage": "1c", "iters": 8000, "tag": "t"}
        ctx.recompute()
        assert rc.find_inflight(ctx) is None, verdict
    assert set(rc.RESUMABLE_HALTS) == {"halt:infra", "halt:eval_infra"}


# ================================================================== audit round 3
def _candidate_header(rc, cands, ctx, cid, *, xid="X07", screen_=None, started="2026-09-19T00:07:00"):
    """A finished candidate header of the shape :func:`run_experiment` writes."""
    item = candidate_item(rc, cands, ctx, cid)
    recipe = item["recipe"]
    header = {"xid": xid, "kind": "candidate", "candidate": cid, "seed": ctx.seed_screen,
              "label": f"{cid} s3", "chain": recipe["chain"],
              "base_recipe_id": item["base_recipe"]["recipe_id"],
              "recipe_id": recipe["recipe_id"], "base_rank": base_rank(cid), "code_sha": ctx.code,
              "tools_sha": ctx.tools,
              "overlay_by_stage": {str(s["stage"]): rc.stage_env(recipe, str(s["stage"]))
                                   for s in recipe["stages"]},
              "started": started, "inflight": None, "status": "ok", "ckpt": f"/sim/{xid}/m.pt",
              "eval_dirs": [], "recs": [], "decision": None, "eligibility": None,
              "stages": [str(s["stage"]) for s in recipe["stages"]],
              "eval_carry": list(recipe.get("eval_carry") or []),
              "adopted": list(recipe.get("adopted") or []),
              "anchor": recipe.get("anchor"),
              "stage_specs": json.loads(json.dumps([dict(s) for s in recipe["stages"]])),
              "screen": screen_ if screen_ is not None else screen()}
    return item, header


def test_a_refused_decide_skip_changes_nothing(rc, cands, tmp_path):
    """Vocabulary: "``skip`` is refused when ``Cid``'s newest non-squashed entry is a decision".

    The refusal has to be a NO-OP. ``ingest_overrides`` catches the raise and marks the line
    applied, so a ``user_choice`` written BEFORE the check survived it -- and a standing
    ``user_choice = skip`` is not inert: derived-view cause (d) squashes every header of that
    candidate started before its stamp with a cause that is NEVER resurrected, and ``next_item``
    step 8 then emits the very SKIP the CLI and the driver had both just refused.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain", base=base,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                             base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base,
                                             base_rank("C1"), status="running", decision="D05",
                                             inflight={"stage": "c1", "iters": 1000, "tag": "t"})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    rc.lc.append_override(ctx.dir, {"candidate": "C1", "choice": "skip", "why": "never mind"})
    rc.ingest_overrides(ctx)
    assert any("REFUSED" in line and "not decide skip" in line for line in notify_lines(ctx))
    assert ctx.lad["user_choice"] == {}, "a refused line records no standing choice"
    assert ctx.lad["squashed"] == {}, "and squashes nothing"
    assert set(ctx.lad["experiments"]) == {"X01", "X02", "X03", "X04"}
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "speculative"
    assert (rc.find_inflight(ctx) or {}).get("xid") == "X04", "the pair run is still resumable"
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["label"] != "C1 SKIP"
    # and the same refusal a second time is still a no-op (the line is idempotent)
    rc.lc.append_override(ctx.dir, {"candidate": "C1", "choice": "skip", "why": "really"})
    rc.ingest_overrides(ctx)
    assert ctx.lad["user_choice"] == {}


def test_a_resumed_candidate_still_checks_anchored_drift(rc, cands, tmp_path):
    """Decision 5: anchored drift "is a precondition of ANY ADOPT, clear or speculative".

    A window that was stopped, SIGTERMed or infra-halted is rebuilt by ``item_from_header``, whose
    recipe carries no anchor -- so reading the anchor off the ITEM made the precondition silently
    ``n/a`` on every resumed candidate, turning a drift-failing clear screen into a clear ADOPT with
    NO pair and permanently moving the recipe every later candidate is screened on.
    """
    def build(tmp, name):
        led, st = ledger(), state()
        with_baseline(rc, cands, led, st, screen_=screen(completion=0.95))   # X01 = the anchor
        led["entries"].append(entry("D05", "C1", "REJECT", "decided"))
        r_c1b = rc._recipe_with(led, cands, CODE, "C1b", set())
        led["entries"].append(entry("D06", "C1b", "ADOPT", "speculative", confidence="uncertain",
                                    experiment="X03", recipe=r_c1b["recipe_id"]))
        led["entries"].append(entry("D07", "C1c", "REJECT", "decided"))
        st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1b", "3",
                                                 r_c1b["recipe_id"], rid(rc, cands, led, "C1b"),
                                                 base_rank("C1b"), screen_=screen(completion=0.50))
        return make_ctx(rc, cands, tmp, led=led, st=st, name=name)

    better = screen(closed=0.72, prog=0.11, completion=0.50)
    ctx_fresh = build(tmp_path, "fresh")
    item, header = _candidate_header(rc, cands, ctx_fresh, "C2", screen_=better)
    assert header["anchor"]["candidate"] == "B0", "the baseline anchors until an ADOPT qualifies"
    rc._dispatch_candidate(ctx_fresh, item, header)
    fresh = rc.entries(ctx_fresh.ledger)[-1]

    ctx_res = build(tmp_path, "resumed")
    _item2, header2 = _candidate_header(rc, cands, ctx_res, "C2", screen_=better)
    rc._dispatch_candidate(ctx_res, rc.item_from_header(ctx_res, header2), header2)
    resumed = rc.entries(ctx_res.ledger)[-1]

    for got in (fresh, resumed):
        assert got["evidence"]["anchored_drift"]["verdict"] == "FAIL"
        assert (got["chosen"], got["confidence"], got["status"]) == ("REJECT", "uncertain",
                                                                     "speculative")
    assert fresh["evidence"]["anchored_drift"] == resumed["evidence"]["anchored_drift"]
    assert fresh["chosen"] == resumed["chosen"]


def _code_landing_ladder(rc, cands, tmp_path, *, rerun, name):
    """C1 a clear ADOPT under ``CODE0``; a ``B<k>`` rerun of ``rerun`` lands under ``CODE1``.

    ``rerun`` is ``"anchor"`` (the rerun is of C1's own recipe, so the anchor re-resolves onto it),
    ``"baseline"`` (only the baseline recipe was re-screened) or ``None`` (nothing under the new
    hash at all).
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03",
                                base=b0["recipe_id"], recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"), code=CODE,
                                             screen_=screen(closed=0.72))
    if rerun:
        rid_new = r_c1["recipe_id"] if rerun == "anchor" else b0["recipe_id"]
        for seed, xid, closed in (("3", "X05", 0.60), ("2", "X06", 0.61)):
            st["ladder"]["experiments"][xid] = exp(xid, "baseline", "B1", seed, rid_new, rid_new,
                                                   {"upto": None, "inclusive": False},
                                                   screen_=screen(closed=closed), code=OTHER_CODE)
            led["entries"].append(entry(f"D0{7 if seed == '3' else 8}", "B1", "RECORD", "final",
                                        kind="baseline", experiment=xid, base=rid_new,
                                        recipe=rid_new, seed=seed, code_sha=OTHER_CODE))
    return make_ctx(rc, cands, tmp_path, led=led, st=st, code=OTHER_CODE, name=name)


def test_the_drift_anchors_screens_are_resolved_under_the_current_code_sha(rc, cands, tmp_path):
    """Amendment 28: the anchor's s3 is an ``ok_screen``, never the recorded old-code header.

    Decision 9: after ``--accept-code-change`` "'ok screen' fails for every old screen ... and the
    anchor re-resolves under the new hash". ``_anchor_screens`` read ``all_headers()[xid]["screen"]``
    with no code, kind or squash filter, so every drift check after a landing compared a NEW-code
    candidate against the anchor's OLD-code screen -- a comparison across the very change the
    re-baseline exists to absorb -- even when the ``B<k>`` rerun of that same recipe was already
    recorded under the new hash.
    """
    # (a) the rerun IS the anchor's recipe: the anchor keeps its identity and gains the new screen
    ctx = _code_landing_ladder(rc, cands, tmp_path, rerun="anchor", name="onanchor")
    recipe = rc.compose_recipe(ctx.ledger, cands, ctx.code, squashed=set(ctx.view["entries"]))
    assert recipe["anchor"]["candidate"] == "C1"
    s3, s2, label = rc._anchor_screens(ctx, recipe)
    assert s3["facing_closed_by_hold"]["turn_l90"] == 0.60, "X05's screen, not X03's 0.72"
    assert s2["facing_closed_by_hold"]["turn_l90"] == 0.61
    assert label.startswith("C1/D05 (")

    # (b) only the BASELINE was re-screened: the anchor falls back to it, and says so
    ctx = _code_landing_ladder(rc, cands, tmp_path, rerun="baseline", name="onbase")
    recipe = rc.compose_recipe(ctx.ledger, cands, ctx.code, squashed=set(ctx.view["entries"]))
    assert recipe["anchor"]["candidate"] == "C1"
    s3, _s2, label = rc._anchor_screens(ctx, recipe)
    assert s3["facing_closed_by_hold"]["turn_l90"] == 0.60
    assert label.startswith("B1/D07 (") and "no ok s3 screen under this code_sha" in label, label

    # (c) nothing under the new hash: n/a, never the stale header
    ctx = _code_landing_ladder(rc, cands, tmp_path, rerun=None, name="nothing")
    recipe = rc.compose_recipe(ctx.ledger, cands, ctx.code, squashed=set(ctx.view["entries"]))
    s3, s2, label = rc._anchor_screens(ctx, recipe)
    assert s3 is None and s2 is None and label.startswith("C1/D05 (")
    assert rc.anchored_drift({}, cands["decision_rule"], screen(), s3, resol(), {})["verdict"] == "n/a"

    # and the same hash it was screened under still resolves to the anchor's own window
    same = make_ctx(rc, cands, tmp_path, led=ctx.ledger, st=ctx.state, code=CODE, name="samecode")
    recipe = rc.compose_recipe(same.ledger, cands, same.code, squashed=set(same.view["entries"]))
    s3, _s2, label = rc._anchor_screens(same, recipe)
    assert s3["facing_closed_by_hold"]["turn_l90"] == 0.72 and label.startswith("C1/D05 (")


def test_a_resumed_milestone_promotion_keeps_its_absolute_units(rc, cands, tmp_path, monkeypatch):
    """Decision 5: an absolute-units recipe is scored on F_abs/T_abs, not F/T.

    ``_recipe_cand_spec`` reads the recipe's ``adopted`` list. A resumed promotion -- the normal
    path after a Kit DerivedDataCache halt -- rebuilds its item from the header, so without
    ``adopted`` on the header it scored ``yaw_progress`` against ``blocks.F`` instead of
    ``net_yaw`` against ``blocks.F_abs``: a different statistic at a different threshold, deciding
    a PASS/FAIL that drops an adoption or lets a regression through.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2", "C3b", "C4", "C5", "C6", "C7"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append(entry("D_C3a", "C3a", "ADOPT", "decided"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    recipe = rc.milestone_recipe(cands, ctx.ledger, "M2", ctx.code, set(ctx.view["entries"]))
    assert "C3a" in recipe["adopted"]
    assert rc._recipe_cand_spec(ctx, recipe) == {"absolute_units": True}
    assert rc.block_names(rc._recipe_cand_spec(ctx, recipe)) == ("F_abs", "T_abs")

    # the header run_experiment writes carries the list, and item_from_header reads it back
    item = rc._item("milestone", cid="M2", seed="3", recipe=recipe,
                    stages=rc.promotion_stage_list(cands), why="t", label="M2.promo s3")
    header = {"xid": "X09", "kind": "milestone", "candidate": "M2", "seed": "3", "chain": None,
              "label": "M2.promo s3", "code_sha": ctx.code, "recipe_id": recipe["recipe_id"],
              "base_recipe_id": recipe["recipe_id"], "started": "x", "status": "ok",
              "overlay_by_stage": {}, "adopted": list(recipe["adopted"]), "anchor": None,
              "eval_carry": [], "recs": [], "eval_dirs": [], "inflight": None}
    rebuilt = rc.item_from_header(ctx, header)
    assert rebuilt["recipe"]["adopted"] == recipe["adopted"]
    assert rc._recipe_cand_spec(ctx, rebuilt["recipe"]) == {"absolute_units": True}

    def promote(promo_item):
        led2 = copy.deepcopy(ctx.ledger)
        st2 = copy.deepcopy(ctx.state)
        c2 = make_ctx(rc, cands, tmp_path, led=led2, st=st2, name=f"p{id(promo_item)}")
        c2.ledger["entries"].append({"id": "M2", "kind": "milestone", "candidate": "M2",
                                     "chosen": None, "status": "open", "history": [],
                                     "options": ["PASS", "WATCH", "FAIL"], "evidence": {},
                                     "recipe_id": recipe["recipe_id"], "eligibility": 1})
        for stage in ("2a", "2b"):
            c2.c.recs[f"X09:{stage}"] = {"ckpt": f"/sim/X09/{stage}.pt",
                                         "evals": {row: {"run_dir": f"/head/{stage}/{row}"}
                                                   for row in ("slow", "light", "ring")}}
        prev = {s: {"evals": {row: f"/prev/{s}/{row}" for row in ("slow", "light", "ring")}}
                for s in ("2a", "2b")}
        monkeypatch.setattr(rc, "_previous_promoted", lambda ctx_, mid: (prev, "M1"))
        monkeypatch.setattr(rc, "summarize_run_dir", lambda run_dir, sid: screen())
        rc._close_milestone_promo(c2, promo_item, dict(header, recs=["X09:2a", "X09:2b"]))
        rows = rc.entry_by_id(c2.ledger, "M2")["evidence"]["promotion"]["rows"]
        return {row["compare"]["blocks"]["F"]["block"] for row in rows if row.get("compare")}

    assert promote(rebuilt) == {"F_abs"}
    # a recipe with NO adoption list at all (a header written before the key existed) falls back to
    # a recompose rather than silently reading "no adoptions" and scoring in the wrong unit system
    header_old = {k: v for k, v in header.items() if k != "adopted"}
    old_item = rc.item_from_header(ctx, header_old)
    assert "adopted" not in old_item["recipe"]
    assert promote(old_item) == {"F_abs"}


def test_a_user_confirm_reverts_a_driver_self_overrule(rc, cands, cli, tmp_path):
    """The plan's transition table: "``overruled(driver)`` -> ``confirmed`` (user ``confirm``
    freezes the entry's ``chosen``, i.e. REVERTS the driver's override + resurrection)".

    The SELF-OVERRULE line quotes ``confirm <Did>`` and ``overrule <Did> --choose <alternative>``,
    and after a self-overrule ``alternative`` IS the driver's own pick -- so if ``confirm`` froze
    what stood, neither quoted command could undo the override and the user had no way back.
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "overruled", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain",
                                base=b0["recipe_id"], recipe=r_c1["recipe_id"],
                                alternative="REJECT", pooled_confidence="clear",
                                history=[{"from": "speculative", "to": "overruled", "by": "driver",
                                          "why": "pooled"}],
                                override={"id": "driver:X04", "by": "driver", "choice": "REJECT",
                                          "why": "its own pair disagreed",
                                          "at": "2026-09-19T01:00:00"}))
    # a later screen that ran on the REDUCED recipe: the revert has to squash it again
    reduced = rid(rc, cands, led, "C2")
    st["ladder"]["experiments"]["X06"] = exp("X06", "candidate", "C2", "3", "rid-c2", reduced,
                                             base_rank("C2"), ckpt=None, eval_dirs=[])
    led["entries"].append(entry("D06", "C2", "REJECT", "decided", experiment="X06", base=reduced))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.lc.effective_choice(rc.entry_by_id(ctx.ledger, "D05")) == "REJECT"
    assert cli.confirm_choice(rc.entry_by_id(ctx.ledger, "D05")) == "ADOPT"
    assert cli.confirm_choice(rc.entry_by_id(ctx.ledger, "D06")) == "REJECT", "an ordinary entry"

    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm",
                                    "choice": cli.confirm_choice(rc.entry_by_id(ctx.ledger, "D05")),
                                    "why": "I have seen the pair and I still want C1"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "confirmed" and d05["frozen"] is True
    assert rc.lc.effective_choice(d05) == "ADOPT", "confirm REVERTS the driver's override"
    assert d05["override"]["by"] == "user"
    assert rc.compose_recipe(ctx.ledger, cands, ctx.code)["adopted"] == ["C1"]
    assert "X06" in ctx.lad["squashed"], "the screens taken on the reduced recipe go with it"
    assert rc.entry_by_id(ctx.ledger, "D06")["status"] == "squashed"
    assert any("SQUASH" in line for line in notify_lines(ctx))


def test_a_pooled_decision_reads_the_screens_its_entry_recorded(rc, cands, tmp_path):
    """Decision 9: "a pair pending at the change keeps its recorded s3 screens under the old
    ``code_sha``, its s2 and any missing twin run under the new hash".

    ``ok_screen`` filters on the CURRENT hash, so looking the two s3 screens up that way after a
    code landing raised an UNCAUGHT RuntimeError -- and because the s2 header is ``ok`` with no
    ``inflight``, the next launch re-derived the same ``pooled`` item and crashed again.
    """
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, seeds=("3",), code=OTHER_CODE)    # the OLD hash
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"), code=OTHER_CODE)
    # the twin and the candidate's own s2 ran AFTER the landing, under the new hash
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C1", "2", b0["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"), decision="D05")
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain",
                                base=b0["recipe_id"], recipe=r_c1["recipe_id"],
                                control_experiment="X01"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.ok_screen(ctx.state, ctx.code, b0["recipe_id"], "3", ctx.view) is None, \
        "the old s3 screens are no longer ok screens -- that is the whole point"
    item = rc._item("pooled", cid="C1", seed="2", base_recipe=b0, decision="D05", why="t",
                    label="C1 s2 pooled")
    rc._resolve_pair(ctx, item, rc.entry_by_id(ctx.ledger, "D05"),
                     ctx.lad["experiments"]["X04"]["screen"], s2_xid="X04",
                     header=ctx.lad["experiments"]["X04"])
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] in ("final", "overruled")
    assert d05["evidence"]["pooled"]["pooled"] is True
    assert ctx.state["phase"] != "halt:X04 control_missing"

    # and with no recorded link and no ok screen either, it is a NAMED halt, not a traceback
    led2, st2 = copy.deepcopy(led), copy.deepcopy(st)
    [e for e in led2["entries"] if e["id"] == "D05"][0].pop("control_experiment")
    [e for e in led2["entries"] if e["id"] == "D05"][0]["status"] = "speculative"
    st2["ladder"]["experiments"]["X01"]["status"] = "interrupted"
    ctx2 = make_ctx(rc, cands, tmp_path, led=led2, st=st2, name="halted")
    with pytest.raises(SystemExit):
        rc._resolve_pair(ctx2, item, rc.entry_by_id(ctx2.ledger, "D05"),
                         ctx2.lad["experiments"]["X04"]["screen"], s2_xid="X04",
                         header=ctx2.lad["experiments"]["X04"])
    assert ctx2.state["phase"] == "halt:X04 control_missing"
    assert any("HALT X04" in line and "control_missing" in line for line in notify_lines(ctx2))


def test_a_repeat_infra_death_on_the_retry_still_notifies_INFRA(rc, cands, tmp_path, monkeypatch):
    """Infra: "on an OOM/cache signature -> stop retrying immediately, NOTIFY ``INFRA`` with the
    signature".

    The signature is what tells the assistant to quarantine the Kit cache. An eval that fails
    transiently first and dies to the OOM killer on the RETRY halted with no INFRA line at all, and
    a ``candidate``'s HALT then advised skipping the candidate -- throwing away a good 3.9-hour
    window for an infrastructure fault.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path, eval_none=2)
    seen = []

    def sig(rc_code, oom_before, log, *, since=None):
        seen.append(rc_code)
        assert since is not None, "the detector is told when this attempt started"
        return (None, []) if len(seen) == 1 else ("oom_kill delta 1 (cgroup memory.events)", ["48G"])

    monkeypatch.setattr(rc, "infra_signature", sig)
    with pytest.raises(SystemExit):
        rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert len(seen) == 2, "the retry gets its own infra read"
    lines = notify_lines(ctx)
    assert any("INFRA" in line and "(retry)" in line and "oom_kill" in line for line in lines)
    halt = [line for line in lines if "HALT" in line][-1]
    assert "eval_infra" in halt and "oom_kill" in halt
    assert "quarantine the Kit DerivedDataCache" in halt
    assert "decide C2 --choose skip" in halt, "the escape hatch is still quoted, but second"
    assert ctx.lad["experiments"]["X03"]["status"] == "halt:eval_infra"


def test_a_sigterm_during_the_replay_writes_no_experiment_header(rc, cands, tmp_path, monkeypatch):
    """The plan's named test: "SIGTERM during the replay -> replay: interrupted, NO experiment
    header, top-of-loop STOP check"."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    calls = install_fakes(rc, monkeypatch, tmp_path)
    monkeypatch.setitem(rc.INTERRUPT, "signal", None)
    item = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert item["candidate"] == "C2" and (candidate_by(cands, "C2").get("replay_args"))

    def fake_replay(ctx_, item_):
        rc.INTERRUPT["signal"] = 15
        return {"status": "interrupted", "rc": None}

    monkeypatch.setattr(rc, "run_replay", fake_replay)
    assert rc.loop(ctx) == 0
    assert ctx.state["phase"] == "await_user:stop"
    assert set(ctx.lad["experiments"]) == {"X01", "X02"}, "no experiment header was written"
    assert calls["train"] == [] and calls["eval"] == []


def candidate_by(cands, cid):
    return [c for c in cands["candidates"] if c["id"] == cid][0]


def test_a_milestone_override_never_touches_the_decisions_own_pair_run(rc, cands, tmp_path):
    """Decision 7: "a driver or milestone override never touches ``D_k``'s own seed-2 run", and
    derived-view cause (e): "the finished seed-2 run of a ``final`` decision is never squashed by
    (e), whatever the user later does to that decision"."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "final", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain",
                                pooled_confidence="uncertain", base=b0["recipe_id"],
                                recipe=r_c2["recipe_id"],
                                history=[{"from": "speculative", "to": "final", "by": "driver",
                                          "why": "pooled"}]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                             b0["recipe_id"], base_rank("C2"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C2", "2", r_c2["recipe_id"],
                                             b0["recipe_id"], base_rank("C2"), decision="D_C2")
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    m1 = {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": None, "status": "open",
          "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {},
          "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M1", ctx.code)["recipe_id"],
          "eligibility": 1}
    ctx.ledger["entries"].append(m1)
    rc._milestone_fail(ctx, "M1", m1, ["M1 (ii)"], ["slow", "light"])
    dropped = rc.entry_by_id(ctx.ledger, "D_C2")
    assert dropped["status"] == "overruled" and dropped["override"]["by"] == "milestone"
    assert "X04" in ctx.lad["experiments"], "the decision's own finished pair run is untouched"
    assert "X04" not in ctx.lad["squashed"]
    assert "X04" not in rc._user_squashed_s2(ctx.ledger, set())


def test_a_confirm_squashes_a_HALTED_seed_two_header_too(rc, cands, tmp_path):
    """Verification 7 drives ``confirm <Did>`` "against a copied in-flight seed-2 header". Since an
    infra halt became RESUMABLE, a halted header with ``inflight`` set is exactly such a header --
    the next launch would resume it -- so the confirm has to reach it as well."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "speculative", experiment="X03",
                                seed2_experiment="X04", confidence="uncertain", base=base,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                             base_rank("C1"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "seed2", "C1", "2", r_c1["recipe_id"], base,
                                             base_rank("C1"), status="halt:eval_infra",
                                             decision="D05", ckpt=None,
                                             inflight={"stage": "c1", "iters": 1000, "tag": "t"})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert (rc.find_inflight(ctx) or {}).get("xid") == "X04", "a halted infra window IS resumable"
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm", "choice": "ADOPT",
                                    "why": "enough"})
    rc.ingest_overrides(ctx)
    assert "X04" in ctx.lad["squashed"] and ctx.lad["squashed"]["X04"]["squashed_by"] == "user_choice"
    assert ctx.lad["squashed"]["X04"]["status"] == "interrupted"
    assert rc.find_inflight(ctx) is None, "and it is NOT resumed"


def test_a_stop_mid_train_goes_through_the_real_interrupt_chain(rc, cands, tmp_path, monkeypatch):
    """The stop path end to end, with the REAL ``stop_requested`` / ``interrupt_reason`` /
    ``_mark_interrupt``: a stop leaves the header ``running`` with its ``inflight`` block, a user
    ``skip`` leaves it ``interrupted``. Every other test of this path monkeypatches
    ``interrupt_reason`` to a literal string, so ``_mark_interrupt``'s classification was never
    exercised against what the reason vocabulary really returns."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    install_fakes(rc, monkeypatch, tmp_path)
    monkeypatch.setitem(rc.INTERRUPT, "signal", None)
    reasons = []

    def train_and_stop(c, spec, tag, iters, resume, sound, early=False, *, interrupt_check=None,
                       **kw):
        rc.INTERRUPT["signal"] = 15                  # the stop lands INSIDE the window
        reasons.append(interrupt_check())
        return None, "interrupted", c.logs / f"{tag}_train.log"

    monkeypatch.setattr(rc.rp, "train_retry", train_and_stop)
    header = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C1"))
    assert reasons == ["sigterm"], "the real interrupt_reason says `sigterm`"
    assert header["interrupt"] == "sigterm"
    assert header["status"] == "running", "a stop is resumable"
    assert header["inflight"] and header["inflight"]["stage"] == "c1"
    rc.INTERRUPT["signal"] = None
    assert (rc.find_inflight(ctx) or {}).get("xid") == header["xid"], "the next launch resumes it"

    # the other direction: a user skip that lands mid-window is not a stop -- it is abandoned
    def train_and_skip(c, spec, tag, iters, resume, sound, early=False, *, interrupt_check=None,
                       **kw):
        # `at` is explicit so the stamp is unambiguously NEWER than the header's `started`: cause
        # (d) and `interrupt_reason` both require "newer than", and a same-second tie is not.
        rc.lc.append_override(ctx.dir, {"candidate": "C2", "choice": "skip", "why": "not now",
                                        "at": "2099-01-01T00:00:00"})
        reasons.append(interrupt_check())
        return None, "interrupted", c.logs / f"{tag}_train.log"

    monkeypatch.setattr(rc.rp, "train_retry", train_and_skip)
    header2 = rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C2"))
    assert reasons[-1] and reasons[-1].split(":")[0] in ("squashed", "user skip of C2")
    assert header2["status"] == "interrupted", "a squash/user skip is NOT resumable"
    assert ctx.lad["squashed"][header2["xid"]]["status"] == "interrupted"
    assert (rc.find_inflight(ctx) or {}).get("xid") != header2["xid"]


def test_a_guard_above_its_R_is_reported_TIE_and_never_BETTER(rc, cands, rule, tmp_path):
    """Guards are ONE-SIDED (decision 5): "a guard may only make a candidate worse". Nothing read a
    guard's ``BETTER`` for a verdict, but the DECISION line printed it beside the guard, where it
    read as a reason to adopt."""
    out = dec(rc, rule, screen(completion=0.95, tripod=0.90), screen())
    rows = {r["metric"]: r for r in out["evidence"]["seed3"]["guards"]["rows"]}
    assert rows["completion"]["class"] == "TIE" and rows["completion"]["above_R"] is True
    assert rows["tripod_walking_dir"]["class"] == "TIE"
    assert all(r["class"] in ("TIE", "WORSE") for r in rows.values())
    assert out["evidence"]["seed3"]["guards"]["worse"] == []
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    item = candidate_item(rc, cands, ctx, "C2")
    line = " | ".join(rc.decision_notify_fields(
        ctx, entry("D08", "C2", out["chosen"], out["status"], evidence=out["evidence"],
                   alternative=out["alternative"]), item, out))
    assert "[BETTER]" not in line
    assert "guards TIE" in line


def test_d00b_reports_whether_the_seed_spread_clears_the_eval_spread(rc, cands, tmp_path):
    """Decision 6's pinned test ``R_F(D00b) >= R_F(D00)``: the seed spread is a FLOOR under the eval
    spread, so a block whose seed term comes out NARROWER says the pair is unrepresentative."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st, screen_=screen())
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    # both baseline screens identical -> every seed delta is 0 -> R_F^seed == 0 < R_F^eval
    rc._write_d00b(ctx, ctx.lad["experiments"]["X02"])
    line = [l for l in notify_lines(ctx) if "RESOLUTION D00b" in l][-1]
    assert "R_F(D00b) >= R_F(D00): VIOLATED" in line
    assert "F: R_F^seed" in line
    assert rc.entry_by_id(ctx.ledger, "D00b")["evidence"]["r_f_floor"]["violations"]
    # and the control check itself is run in BOTH unit systems
    joined = "\n".join(notify_lines(ctx))
    assert "control check (F/T)" in joined and "control check (F_abs/T_abs)" in joined
    # the per-guard halves are BOTH named, and the guards the seed term does not widen are called
    # out -- that is the window-2 read the degenerate turn_survival_r90 pair needs
    res_line = [l for l in notify_lines(ctx) if "guards @0.99 one-sided" in l][-1]
    assert "R_eval" in res_line and "R_seed" in res_line
    assert "carried by the EVAL half-width" in res_line
    assert "turn_survival_r90" in res_line


@pytest.mark.parametrize("text", ["decision(C1).nonsense == 1", "decision(C1).chose == ADOPT",
                                  "recipe.plant == x", "exp.screen.completion > 1",
                                  "screen2.completion > 1"])
def test_an_unknown_predicate_field_is_refused_at_load(rc, text):
    """The plan calls the conditional vocabulary FIXED. A typo used to COMPILE, evaluate to None and
    SKIP the candidate at its rank -- a 3.9-hour window written off, recoverable only by a
    ``decide --choose run`` nobody knew to type."""
    with pytest.raises(ValueError):
        rc.compile_predicate(text)


def test_the_validator_refuses_a_candidate_whose_conditional_names_an_unknown_field(rc, cands):
    doc = copy.deepcopy(cands)
    c7 = [c for c in doc["candidates"] if c["id"] == "C7"][0]
    c7["conditional"] = "recipe.has_pool and decision(C1b).choosen != ADOPT"
    assert any("conditional does not parse" in reason for reason in _refusals(rc, doc, "C7"))


def test_the_shipped_conditionals_all_use_the_fixed_vocabulary(rc, cands):
    for cand in cands["candidates"]:
        if cand.get("conditional"):
            rc.compile_predicate(cand["conditional"])      # raises on an unknown field
    assert rc.DECISION_FIELDS == ("chosen", "confidence", "pooled_confidence", "status",
                                  "floor_stage")
    assert rc.RECIPE_FIELDS == ("chain", "has_pool")


def test_a_stale_user_skip_does_not_re_write_itself_over_a_later_screen(rc, cands, tmp_path):
    """Vocabulary: a ``user_choice`` of ``skip`` writes a SKIP entry only while it is "newer than
    every experiment of ``Cid``"."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    st["ladder"]["user_choice"]["C1"] = {"choice": "skip", "uuid": "u1",
                                         "at": "2026-09-19T00:01:00"}
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["label"] == "C1 SKIP" and nxt["extra"]["source"] == "user"
    # a screen of C1 that STARTED after the choice makes it stale: C1 is re-decided, not re-skipped
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    ctx.lad["experiments"]["X09"] = exp("X09", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                        base_rank("C1"), started="2026-09-19T05:00:00")
    ctx.recompute()
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["label"] != "C1 SKIP"


def test_a_squashed_running_header_is_archived_interrupted_and_restored_on_resurrection(rc, cands,
                                                                                        tmp_path):
    """A squash pops the LIVE header, so the ``_mark_interrupt`` the running stage loop performs
    afterwards lands on an orphan. The archived copy must not go on claiming the window is still
    training -- and a resurrection has to hand ``find_inflight`` a resumable header again."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03",
                                base=b0["recipe_id"]))
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"],
                                             b0["recipe_id"], base_rank("C1"))
    st["ladder"]["experiments"]["X06"] = exp("X06", "candidate", "C2", "2", "rid-c2",
                                             rid(rc, cands, led, "C2"), base_rank("C2"),
                                             status="running", ckpt=None, eval_dirs=[],
                                             inflight={"stage": "1c", "iters": 8000, "tag": "t"})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert (rc.find_inflight(ctx) or {}).get("xid") == "X06"
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "not this one"})
    rc.ingest_overrides(ctx)
    assert ctx.lad["squashed"]["X06"]["status"] == "interrupted"
    assert ctx.lad["squashed"]["X06"]["pre_squash_status"] == "running"
    assert rc.find_inflight(ctx) is None
    # revert -> resurrection, and the window is resumable again
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "ADOPT",
                                    "why": "reverted"})
    rc.ingest_overrides(ctx)
    assert "X06" in ctx.lad["experiments"] and "X06" not in ctx.lad["squashed"]
    assert ctx.lad["experiments"]["X06"]["status"] == "running"
    assert "pre_squash_status" not in ctx.lad["experiments"]["X06"]
    assert (rc.find_inflight(ctx) or {}).get("xid") == "X06"


# ================================================================== audit round 5: the twin, the
# written gates as WATCH context, and the two untested plan scenarios
def test_a_user_skip_never_squashes_the_base_recipes_twin(rc, cands, tmp_path):
    """Decision 7: "the base recipe's twin is never interrupted or squashed (it is the base's s2
    screen, needed by later pairs and by ``M<k>.s2``)", and SQUASH (2) scopes the skip clause to the
    RUNNING EXPERIMENTS OF the candidate.

    A twin's ``candidate`` field names the candidate whose PAIR booked it, so derived-view cause (d)
    -- which is never resurrected -- used to squash it along with the candidate's own screens. With
    C4 adopted, R(<=C5)'s only s2 screen IS that twin: the base recipe silently lost its s2 and the
    next pair on it (and ``M<k>.s2``) re-paid a 3.9-hour window that was already on disk.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    r_c4 = rc._recipe_with(led, cands, CODE, "C4", set())
    led["entries"].append(entry("D_C4", "C4", "ADOPT", "decided", experiment="X03",
                                base=rid(rc, cands, led, "C4"), recipe=r_c4["recipe_id"]))
    base_c5 = rid(rc, cands, led, "C5")
    assert base_c5 == r_c4["recipe_id"], "C4 is adopted, so R(<=C5) is the recipe with C4 in it"
    r_c5 = rc._recipe_with(led, cands, CODE, "C5", set())
    led["entries"].append(entry("D_C5", "C5", "ADOPT", "speculative", experiment="X04",
                                confidence="uncertain", base=base_c5, recipe=r_c5["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C4", "3", r_c4["recipe_id"],
                                             rid(rc, cands, led, "C4"), base_rank("C4"))
    st["ladder"]["experiments"]["X04"] = exp("X04", "candidate", "C5", "3", r_c5["recipe_id"],
                                             base_c5, base_rank("C5"))
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C5", "2", base_c5, base_c5,
                                             base_rank("C5"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.ok_screen(ctx.state, ctx.code, base_c5, "2", ctx.view)["xid"] == "X05"

    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen it",
                                    "at": "2026-09-19T12:00:00"})
    assert rc.ingest_overrides(ctx) is True
    assert rc.entry_by_id(ctx.ledger, "D_C5")["status"] == "squashed"
    rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "skip", "why": "changed my mind",
                                    "at": "2026-09-19T12:30:00"})
    assert rc.ingest_overrides(ctx) is True

    assert "X05" in ctx.lad["experiments"] and "X05" not in (ctx.lad.get("squashed") or {})
    assert "X05" not in ctx.view["experiments"]
    kept = rc.ok_screen(ctx.state, ctx.code, base_c5, "2", ctx.view)
    assert kept is not None and kept["xid"] == "X05", "the base recipe still has its s2 screen"
    # the exemption is the twin's alone: the candidate's OWN screen still goes under cause (d)
    assert "X04" in ctx.view["experiments"] and "cause d" in ctx.view["experiments"]["X04"]
    skips = [e for e in rc.entries(ctx.ledger) if e.get("candidate") == "C5"
             and e.get("chosen") == "SKIP"]
    assert len(skips) == 1 and skips[0]["status"] == "skipped"
    # ... and a twin whose base recipe REALLY moved is still squashed (cause (a) is untouched)
    rc.lc.append_override(ctx.dir, {"decision": "D_C4", "action": "overrule", "choice": "REJECT",
                                    "why": "drop C4"})
    assert rc.ingest_overrides(ctx) is True
    assert "X05" in ctx.view["experiments"]
    assert "base recipe changed" in ctx.view["experiments"]["X05"]


def test_a_user_skip_does_not_interrupt_a_running_twin(rc, cands, tmp_path, monkeypatch):
    """The interrupt half of the same rule: a twin two hours into its window is the BASE's s2 screen,
    and killing it costs a window the next pair has to pay again."""
    led, st = ledger(), state()
    b0 = with_baseline(rc, cands, led, st)
    live = {"status": "running", "inflight": {"stage": "1c", "iters": 8000, "tag": "t"}}
    st["ladder"]["experiments"]["X05"] = exp("X05", "twin", "C5", "2", b0["recipe_id"],
                                             b0["recipe_id"], base_rank("C5"), **live)
    st["ladder"]["experiments"]["X06"] = exp("X06", "candidate", "C5", "3", "r_c5",
                                             b0["recipe_id"], base_rank("C5"), **live)
    st["ladder"]["user_choice"]["C5"] = {"choice": "skip", "uuid": "u1",
                                         "at": "2026-09-19T23:59:59"}
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    monkeypatch.setitem(rc.INTERRUPT, "signal", None)
    assert rc.interrupt_reason(ctx, "X05") is None
    reason = rc.interrupt_reason(ctx, "X06")
    assert reason and "C5" in reason


def gate_screen(**over) -> dict:
    """A ``summarize()``-shaped screen that also carries the written Part-H gates' own keys."""
    fields = {"tripod_walking": 0.55, "completion": 0.80}
    fields.update({f"dir_ratio_by_hold__{h}": 0.60
                   for h in ("lat_l", "diag", "back", "move_turn")})
    for hold in ("turn_l90", "turn_r90"):
        fields[f"facing_by_hold__{hold}__progress"] = 0.080
        fields[f"facing_by_hold__{hold}__final"] = 0.25
    fields.update(over)
    return screen(**fields)


def test_the_formation_gate_is_watch_context_and_never_moves_the_decision(rc, cands, tmp_path,
                                                                          monkeypatch):
    """The plan's reuse list names ``rp.gate_1c_verdict`` WATCH-only context. The ladder never calls
    ``run_phase``, so nothing was calling it: a screen could fail the very formation gate the rest of
    the campaign enforces and no record of the ladder would say so.

    It is CONTEXT: the same screen with a gate-failing tripod must produce the identical decision.
    """
    out = {}
    for tag, tripod, want in (("pass", 0.55, "PASS"), ("fail", 0.10, "FAIL")):
        led, st = ledger(), state()
        with_baseline(rc, cands, led, st)
        ctx = make_ctx(rc, cands, tmp_path, led=led, st=st, name=f"camp_{tag}")
        install_fakes(rc, monkeypatch, tmp_path / tag,
                      eval_summary=gate_screen(tripod_walking=tripod))
        item = candidate_item(rc, cands, ctx, "C2")
        header = rc.run_experiment(ctx, item)
        assert header["status"] == "ok"
        report = (ctx.dir / "REPORT.md").read_text()
        assert f"- WATCH gate_1c: {want} --" in report
        assert ("WATCH only: this gate never decides" in report) is (want == "FAIL")
        rc._dispatch_candidate(ctx, item, header)
        line = [l for l in notify_lines(ctx) if "DECISION D" in l][-1]
        gate = [f for f in rc.lc.parse_notify(line)["fields"] if f.startswith("WATCH gate_1c")]
        assert len(gate) == 1 and gate[0].startswith(f"WATCH gate_1c: {want} -- formation gate")
        assert "campaign HALTED" not in line, "run_phase's halt advice is false in the ladder"
        out[tag] = rc.newest_entry(ctx.ledger, "C2", set())
    a, b = out["pass"], out["fail"]
    assert (a["chosen"], a["status"], a["confidence"]) == (b["chosen"], b["status"], b["confidence"])
    assert a["evidence"]["seed3"]["blocks"] == b["evidence"]["seed3"]["blocks"]
    assert a["evidence"]["seed3"]["guards"] == b["evidence"]["seed3"]["guards"]


def test_the_turn_gate_is_watch_context_on_both_promotion_stages(rc, cands, tmp_path, monkeypatch):
    """``rp.gate_facing_verdict`` on 2a and 2b, the stages it is written for -- reported beside the
    rolling row comparison and read by nothing (a FAIL on 2b leaves the milestone PASS)."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    ctx.ledger["entries"].append(
        {"id": "M2", "kind": "milestone", "candidate": "M2", "chosen": None, "status": "open",
         "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {}, "eligibility": 1,
         "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M2", ctx.code)["recipe_id"]})
    slow = {"2a": gate_screen(), "2b": gate_screen(facing_by_hold__turn_r90__progress=0.02)}
    for stage in ("2a", "2b"):
        evals = {row: {"run_dir": f"/head/{stage}/{row}"} for row in ("slow", "light", "ring")}
        evals["slow"] = dict(slow[stage], run_dir=f"/head/{stage}/slow")
        ctx.c.recs[f"X09:{stage}"] = {"ckpt": f"/sim/X09/{stage}.pt", "evals": evals}
    prev = {s: {"evals": {row: f"/prev/{s}/{row}" for row in ("slow", "light", "ring")}}
            for s in ("2a", "2b")}
    monkeypatch.setattr(rc, "_previous_promoted", lambda ctx_, mid: (prev, "M1"))
    monkeypatch.setattr(rc, "summarize_run_dir", lambda run_dir, sid: screen())
    rc._close_milestone_promo(ctx, {"kind": "milestone", "candidate": "M2",
                                    "recipe": {"adopted": []}, "label": "M2.promo s3"},
                              {"xid": "X09", "kind": "milestone", "candidate": "M2", "seed": "3",
                               "chain": "scratch_1c", "code_sha": ctx.code, "label": "M2.promo s3"})
    fields = rc.lc.parse_notify([l for l in notify_lines(ctx) if "MILESTONE" in l][-1])["fields"]
    gates = [f for f in fields if f.startswith("WATCH gate_facing")]
    assert len(gates) == 2
    assert gates[0].startswith("WATCH gate_facing 2a: PASS -- turn gate passed")
    assert gates[1].startswith("WATCH gate_facing 2b: FAIL -- turn gate FAILED")
    m2 = rc.entry_by_id(ctx.ledger, "M2")
    assert (m2["chosen"], m2["status"]) == ("PASS", "final"), "the gate never decides a milestone"


def test_the_gate_facing_watch_clause_stays_one_notification_field(rc):
    """FIX_QUEUE 7: ``gate_facing_verdict``'s WATCH clause embeds a literal ``|``.

    ``|`` is the notification SEPARATOR, so an unsanitised message would split this one gate line
    into two fields and a parser would read the tail as an unrelated field. ``_field`` maps it to
    ``/``; this pins that the whole gate line survives a ``format_notify`` -> ``parse_notify``
    round trip as ONE field.
    """
    scr = gate_screen()
    for hold in ("turn_l90", "turn_r90"):
        scr["facing_by_hold"][hold]["final"] = 0.68       # above GATE_FACING_MAX_RAD (0.30)
    field = rc.gate_watch("gate_facing", scr, label="2b")
    assert "| WATCH facing precision" in field, "the clause with the literal | must be present"
    got = rc.lc.parse_notify(rc.lc.format_notify("MILESTONE", "M2", ["head", field, "tail"]))
    gates = [f for f in got["fields"] if f.startswith("WATCH gate_facing")]
    assert len(gates) == 1, got["fields"]
    assert "|" not in gates[0] and "/ WATCH facing precision" in gates[0]
    assert got["fields"] == ["head", gates[0], "tail"], "the tail is still its own field"


def test_only_a_final_or_promotion_stage_carries_a_gate_and_an_unreadable_one_is_na(rc):
    """A non-final curriculum stage has no gate at all (its eval is the curriculum row, and the
    formation gate is a statement about a finished 8000-iteration screen), and anything the verdict
    function cannot read is ``n/a`` with the reason -- never ``FAIL``, and never a raise."""
    chain = {"kind": "candidate", "stages": [{"stage": "c1"}, {"stage": "c6"}]}
    assert rc._stage_gate_watch(chain, {"stage": "c1"}, {"evals": {"curriculum": screen()}}) == []
    assert rc._stage_gate_watch(chain, {"stage": "c6"}, {"evals": {"curriculum": screen()}}) == \
        ["WATCH gate_1c: n/a (no slow row on this stage)"]
    # a curriculum-shaped row IS the slow row's key here: the gate's own keys are missing, which is
    # "could not be evaluated" -> n/a, not FAIL
    unreadable = rc._stage_gate_watch(chain, {"stage": "c6"},
                                      {"evals": {"slow": screen()}, "track_ratio_tail": 0.7})[0]
    assert unreadable.startswith("WATCH gate_1c: n/a -- formation gate could not be evaluated")
    promo = {"kind": "milestone", "stages": [{"stage": "2a"}, {"stage": "2b"}]}
    assert rc._stage_gate_watch(promo, {"stage": "2a"}, {"evals": {"slow": gate_screen()}})[0] \
        .startswith("WATCH gate_facing: PASS")
    # a screen shaped so the verdict function RAISES is still one line, not a traceback
    assert rc.gate_watch("gate_1c", dict(gate_screen(), tripod_walking="x"),
                         track_ratio=0.7).startswith("WATCH gate_1c: n/a (TypeError")
    assert rc.gate_watch("gate_facing", {"facing_by_hold": {"turn_l90": 3.0, "turn_r90": 3.0}}) \
        .startswith("WATCH gate_facing: n/a (AttributeError")
    assert "campaign HALTED" not in rc.gate_watch("gate_1c", gate_screen(tripod_walking=0.1),
                                                  track_ratio=0.7)


def test_a_user_run_of_a_skipped_candidate_re_queues_it_and_the_skip_never_comes_back(rc, cands,
                                                                                      tmp_path):
    """The plan's ``user_choice`` scenario: ``C3a`` clear REJECT -> ``C3b`` SKIP (conditional) ->
    ``decide C3b --choose run`` -> C3b is next and is NOT re-skipped.

    The SKIP is squashed by cause (e), which is STICKY: a fresh launch recomputes the derived view
    from the files and must not resurrect it -- that would put the candidate back to SKIP with the
    user's answer already on record.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C2"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    led["entries"].append(entry("D_C3a", "C3a", "REJECT", "decided", confidence="clear"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert nxt["label"] == "C3b SKIP" and nxt["kind"] == "skip"
    skip = rc.write_skip_entry(ctx, "C3b", "driver", "predicate",
                               predicate=nxt["extra"]["predicate"])
    assert skip["status"] == "skipped" and rc.is_terminal(skip)

    rc.lc.append_override(ctx.dir, {"candidate": "C3b", "choice": "run", "why": "screen it anyway"})
    assert rc.ingest_overrides(ctx) is True
    done = rc.entry_by_id(ctx.ledger, skip["id"])
    assert done["status"] == "squashed"
    assert done["override"]["by"] == "user" and done["override"]["why"] == "decide run"
    assert skip["id"] in ctx.view["entries"]
    assert rc.newest_entry(ctx.ledger, "C3b", set(ctx.view["entries"])) is None, "undecided again"
    nxt = rc.next_item(cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    assert (nxt["kind"], nxt["candidate"], nxt["seed"]) == ("candidate", "C3b", "3")
    assert "the user asked to run it" in nxt["why"]

    fresh = make_ctx(rc, cands, tmp_path)          # a relaunch: everything re-derived from the files
    assert rc.entry_by_id(fresh.ledger, skip["id"])["status"] == "squashed"
    assert skip["id"] in fresh.view["entries"]
    assert [e["id"] for e in rc.entries(fresh.ledger) if e.get("candidate") == "C3b"] == [skip["id"]]
    again = rc.next_item(cands, fresh.ledger, fresh.state, code=fresh.code, view=fresh.view)
    assert (again["kind"], again["candidate"]) == ("candidate", "C3b")


def test_a_batched_milestone_seed_two_is_the_base_recipes_s2_screen(rc, cands, tmp_path):
    """The plan's ``next_item`` scenario: ``C2`` clear ADOPT -> ``M1.s2`` -> ``C4`` uncertain -> NO
    twin (R_1's s2 already exists as the batched_s2 run) -> ``C4 s2``.

    ``batched_s2`` is in :data:`SCREEN_KINDS` precisely so a milestone's batched confirmation counts
    as the recipe's s2 screen; ``milestone_s2`` is NOT, because a promotion is not a screen of the
    recipe it promotes.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3a", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r1 = rc._recipe_with(led, cands, CODE, "C2", set())["recipe_id"]
    led["entries"].append(entry("D_C2", "C2", "ADOPT", "decided", experiment="X03",
                                base=rid(rc, cands, led, "C2"), recipe=r1))
    assert rc.milestone_recipe(cands, led, "M1", CODE, set())["recipe_id"] == r1
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r1,
                                             rid(rc, cands, led, "C2"), base_rank("C2"))
    st["ladder"]["experiments"]["X05"] = exp("X05", "batched_s2", "M1", "2", r1, r1,
                                             {"upto": "C3b", "inclusive": True})
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "PASS",
                           "status": "final", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "recipe_id": r1, "evidence": {}})
    r_c4 = rc._recipe_with(led, cands, CODE, "C4", set())
    led["entries"].append(entry("D_C4", "C4", "ADOPT", "speculative", experiment="X06",
                                confidence="uncertain", base=r1, recipe=r_c4["recipe_id"]))
    st["ladder"]["experiments"]["X06"] = exp("X06", "candidate", "C4", "3", r_c4["recipe_id"], r1,
                                             base_rank("C4"))

    item = queue_item(rc, cands, led, st)
    assert (item["kind"], item["candidate"], item["seed"]) == ("seed2", "C4", "2")
    assert item["base_recipe"]["recipe_id"] == r1
    assert item["recipe"]["recipe_id"] == r_c4["recipe_id"]
    assert rc.ok_screen(st, CODE, r1, "2")["xid"] == "X05"
    assert rc.recipe_head(st, CODE, r1, "2") == st["ladder"]["experiments"]["X05"]["ckpt"]

    # the negative half: M3's own seed-2 PROMOTION of the same recipe is not an s2 screen of it
    st["ladder"]["experiments"]["X07"] = exp("X07", "milestone_s2", "M3", "2", r1, r1,
                                             {"upto": "C12", "inclusive": True})
    assert st["ladder"]["experiments"]["X07"]["started"] > \
        st["ladder"]["experiments"]["X05"]["started"]
    assert rc.ok_screen(st, CODE, r1, "2")["xid"] == "X05"
    assert "milestone_s2" not in rc.SCREEN_KINDS and "batched_s2" in rc.SCREEN_KINDS


# ============================================================ F11: the seven untested functions
def test_ensure_resolution_ingests_the_record_and_freezes_the_rule_into_D00(rc, cands, tmp_path):
    """The Main loop's first step: "if resolution.json absent: resolution -> ledger D00 (final) +
    NOTIFY RESOLUTION; D01 measurement recorded", with the ids in WRITE ORDER.

    Nothing drove the function: the one launch test monkeypatched it out, and the id-order test
    hand-built the two entries. So nothing pinned that ``D00`` freezes ``decision_rule`` (the plan:
    "decision 5 verbatim, frozen into D00"), that ``D01`` is written at all, or that the stale-hash
    warning fires when ``resolution.json`` was measured against a different ``candidates.yaml``.
    """
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    record = {"schema": 1, "B": 2000, "rng_seed": 20260919, "metrics": resol()["metrics"],
              "blocks": resol()["blocks"], "seed_terms": {}, "process_floor": {}, "inputs": [],
              "candidates_sha256": rc.sha256_file(ctx.dir / "candidates.yaml")}
    (ctx.dir / "resolution.json").write_text(json.dumps(record))

    got = rc.ensure_resolution(ctx)
    assert got["B"] == 2000 and ctx.resolution["B"] == 2000
    ids = [e["id"] for e in rc.entries(ctx.ledger)]
    assert ids == ["D00", "D01"], "write order: D00 resolution, then D01 the turn-rate measurement"
    d00 = rc.entry_by_id(ctx.ledger, "D00")
    assert (d00["kind"], d00["status"], d00["chosen"]) == ("resolution", "final", "RECORD")
    assert d00["evidence"]["decision_rule"] == cands["decision_rule"], "frozen verbatim"
    assert "inputs" not in d00["evidence"]["resolution"], "the inputs are their own key"
    d01 = rc.entry_by_id(ctx.ledger, "D01")
    assert (d01["kind"], d01["status"]) == ("measurement", "final")
    assert d01["evidence"]["turn_rate"]
    assert ctx.lad["resolution_ids"] == ["D00", "D01"]
    assert any(" RESOLUTION D00 " in line for line in notify_lines(ctx))
    assert "notes" not in got or not any("STALE" in n for n in got.get("notes", []))

    # idempotent, and a record measured against a DIFFERENT candidates.yaml is called out
    rc.ensure_resolution(ctx)
    assert [e["id"] for e in rc.entries(ctx.ledger)] == ["D00", "D01"]
    record["candidates_sha256"] = "0" * 64
    (ctx.dir / "resolution.json").write_text(json.dumps(record))
    again = rc.ensure_resolution(ctx)
    assert any("STALE" in n for n in again.get("notes") or []), again.get("notes")
    assert "WARNING" in (ctx.dir / "orchestrator.log").read_text()


def _m1_promo_ctx(rc, cands, tmp_path, *, status="open"):
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    ctx.ledger["entries"].append(
        {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": None, "status": status,
         "options": ["PASS", "WATCH", "FAIL"], "history": [], "evidence": {}, "eligibility": 1,
         "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M1", ctx.code)["recipe_id"]})
    header = {"xid": "X09", "kind": "milestone", "candidate": "M1", "seed": "3",
              "chain": "scratch_1c", "code_sha": ctx.code, "recs": ["X09:2a", "X09:2b"],
              "label": "M1.promo s3"}
    for stage in ("2a", "2b"):
        ctx.c.recs[f"X09:{stage}"] = {
            "ckpt": f"/sim/X09/{stage}.pt",
            "evals": {row: {"run_dir": f"/head/{stage}/{row}"} for row in ("slow", "light", "ring")}}
    return ctx, header


def test_m1_closes_WATCH_because_there_is_no_previous_milestone_head(rc, cands, tmp_path,
                                                                     monkeypatch):
    """FIX_QUEUE 6(b) + plan pass 6: "``M1.promo`` is never omitted (M1's heads are M2's reference)".

    The WATCH branch was reached by no test -- only the QUEUE half was pinned -- so nothing asserted
    the ENTRY it writes: ``chosen WATCH``, ``final``, the promotion mode, and the heads the next
    milestone rolls against.
    """
    ctx, header = _m1_promo_ctx(rc, cands, tmp_path)
    monkeypatch.setattr(rc, "_previous_promoted", lambda ctx_, mid: (None, None))
    monkeypatch.setattr(rc, "summarize_run_dir", lambda run_dir, sid: screen())
    rc._close_milestone_promo(ctx, {"kind": "milestone", "candidate": "M1",
                                    "recipe": {"adopted": []}, "label": "M1.promo s3"}, header)
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert (m1["chosen"], m1["status"]) == ("WATCH", "final")
    assert m1["evidence"]["promotion"]["mode"] == "WATCH (no previous milestone head)"
    assert m1["promoted_head"]["2b"]["ckpt"] == "/sim/X09/2b.pt"
    assert m1["experiment"] == "X09" and m1["base_code_sha"] == ctx.code
    line = [l for l in notify_lines(ctx) if " MILESTONE " in l][-1]
    fields = rc.lc.parse_notify(line)["fields"]
    assert any("chosen WATCH (final)" in f for f in fields), fields
    assert any("no previous 2a head" in f for f in fields)
    assert "WATCH" in (ctx.dir / "CHANGELOG.md").read_text()


def test_a_carried_forward_milestone_records_the_previous_heads_and_stays_the_rolling_reference(
        rc, cands, tmp_path):
    """FIX_QUEUE 6(c): the no-GPU milestone, and the chain-walk that makes M3 find M1's real heads.

    ``milestone_close`` and ``_previous_promoted`` were executed by NO test (``_previous_promoted``
    is monkeypatched in both tests that reach it), so a regression in either would have degraded
    every later milestone row to "no previous 2a head" -- WATCH -- with the suite green.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    heads = {"2a": {"ckpt": "/a.pt", "evals": {"slow": "/e/a"}},
             "2b": {"ckpt": "/b.pt", "evals": {"slow": "/e/b"}}}
    ctx.ledger["entries"].append(
        {"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "PASS", "status": "final",
         "options": ["PASS", "WATCH", "FAIL"], "history": [], "eligibility": 1, "evidence": {},
         "recipe_id": rc.milestone_recipe(cands, ctx.ledger, "M1", ctx.code)["recipe_id"],
         "promoted_head": heads})
    ctx.write_ledger()
    ctx.recompute()

    m2_recipe = rc.milestone_recipe(cands, ctx.ledger, "M2", ctx.code, set(ctx.view["entries"]))
    rc.milestone_close(ctx, {"kind": "milestone_close", "candidate": "M2", "recipe": m2_recipe,
                             "label": "M2.carry"})
    m2 = rc.newest_milestone_entry(ctx.ledger, "M2", set(ctx.view["entries"]))
    assert (m2["chosen"], m2["status"]) == ("PASS", "final")
    assert m2["evidence"]["carried_from"] == "M1"
    assert m2["promoted_head"] == heads, "both heads carried forward, no GPU"
    line = [l for l in notify_lines(ctx) if " MILESTONE " in l][-1]
    fields = rc.lc.parse_notify(line)["fields"]
    assert any("both experiments are omitted" in f for f in fields), fields
    assert any("evidence.carried_from M1" in f for f in fields)
    assert "carried forward from M1" in (ctx.dir / "CHANGELOG.md").read_text()

    # M3's rolling reference is M1's REAL heads, reached through the carried M2
    got, prev_id = rc._previous_promoted(ctx, "M3")
    assert got["2b"]["ckpt"] == "/b.pt" and prev_id in ("M1", "M2")
    assert rc._previous_promoted(ctx, "M1") == (None, None), "M1 has no previous milestone"

    # and when the carry recorded no head of its own, the `carried_from` CHAIN-WALK is what finds
    # M1 -- without it every later milestone row degrades to "no previous 2a head" (WATCH)
    m2["promoted_head"] = {}
    ctx.write_ledger()
    ctx.recompute()
    got, prev_id = rc._previous_promoted(ctx, "M3")
    assert (prev_id, got["2b"]["ckpt"]) == ("M1", "/b.pt")
    m2["evidence"].pop("carried_from")
    ctx.write_ledger()
    assert rc._previous_promoted(ctx, "M3") == (None, None), "nothing to walk to"


def test_the_final_seed2_promotion_is_the_campaigns_last_record(rc, cands, tmp_path):
    """FIX_QUEUE/F11: ``_close_final_s2`` -- the LAST record the campaign writes, and the one the
    user names the head of record from. Only the queue item was pinned, never the entry."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    header = {"xid": "X20", "kind": "milestone_s2", "candidate": "M3", "seed": "2",
              "chain": "scratch_1c", "code_sha": ctx.code, "recipe_id": "r_final",
              "label": "M3.final_s2 s2"}
    for stage in ("2a", "2b"):
        ctx.c.recs[f"X20:{stage}"] = {
            "ckpt": f"/sim/X20/{stage}_model_2399.pt",
            "evals": {row: dict(gate_screen(), run_dir=f"/s2/{stage}/{row}")
                      for row in ("slow", "light", "ring")}}
    rc._close_final_s2(ctx, {"kind": "milestone_s2", "candidate": "M3", "label": "M3.final_s2 s2"},
                       header)
    e = [x for x in rc.entries(ctx.ledger) if x["kind"] == "milestone_s2"][-1]
    assert (e["candidate"], e["chosen"], e["status"]) == ("M3", "RECORD", "final")
    assert e["experiment"] == "X20" and e["recipe_id"] == "r_final"
    rows = e["evidence"]["promoted_head_s2"]
    assert set(rows) == {"2a", "2b"}
    assert rows["2b"]["ckpt"].endswith("2b_model_2399.pt")
    assert rows["2b"]["evals"]["slow"] == "/s2/2b/slow"
    fields = rc.lc.parse_notify([l for l in notify_lines(ctx) if " MILESTONE " in l][-1])["fields"]
    assert len([f for f in fields if f.startswith("WATCH gate_facing")]) == 2
    assert any("the user names the head of record" in f for f in fields)


def test_the_infra_signature_reads_the_cgroup_counter_and_the_kit_log(rc, tmp_path, monkeypatch):
    """Decision 10's detector. Both tests that exercise its BRANCH monkeypatch it out, so the
    reading itself -- the thing that only ever runs during an incident -- was unpinned."""
    monkeypatch.setattr(rc, "KIT_LOG_DIR", tmp_path / "kit")
    (tmp_path / "kit").mkdir()
    monkeypatch.setattr(rc, "_oom_kills", lambda: 1)
    sig, notes = rc.infra_signature(0, 0, None)
    assert sig == "cgroup oom_kill 0 -> 1" and notes == []
    assert rc.infra_signature(0, 1, None)[0] is None, "no delta, no signature"
    assert rc.infra_signature(0, None, None)[0] is None, "no BEFORE reading, no delta"
    monkeypatch.setattr(rc, "_oom_kills", lambda: None)
    assert rc.infra_signature(-9, None, None)[0].startswith("the eval child returned -9")

    log = tmp_path / "kit" / "kit_20260919.log"
    log.write_text("boot\n[omni.datastore.plugin] cache migrated in 4 s\ndone\n")
    sig, _ = rc.infra_signature(1, None, None, since=log.stat().st_mtime - 5)
    assert sig and "DerivedDataCache" in sig and log.name in sig
    # a HEALTHY log with the two words far apart on different lines is NOT the signature
    log.write_text("omni.datastore loaded\n" + "x\n" * 50 + "assets migrated to v2\n")
    assert rc.infra_signature(1, None, None, since=log.stat().st_mtime - 5)[0] is None
    # R5: within ONE line the two words are order-free. `omni\.datastore.*migrated` also pinned the
    # word ORDER, and a MISS is the expensive direction here -- the driver retries an eval into a
    # corrupted DerivedDataCache (the 2.4 GB/s hazard), while a false positive only costs a halt.
    assert rc.kit_cache_line("cache migration done by omni.datastore plugin")
    assert rc.kit_cache_line("[omni.datastore.plugin] cache migrated in 4 s")
    assert not rc.kit_cache_line("omni.datastore loaded")
    assert not rc.kit_cache_line("assets migrated to v2")
    log.write_text("boot\ncache migrated by the omni.datastore plugin\n")
    sig, _ = rc.infra_signature(1, None, None, since=log.stat().st_mtime - 5)
    assert sig and "DerivedDataCache" in sig, "order within the line must not matter"
    # and a log from BEFORE this attempt cannot explain it
    log.write_text("[omni.datastore.plugin] cache migrated in 4 s\n")
    assert rc.infra_signature(1, None, None, since=log.stat().st_mtime + 60)[0] is None

    log.unlink()
    tb = tmp_path / "eval.log"
    tb.write_text("Traceback (most recent call last):\n  ValueError\n")
    sig, notes = rc.infra_signature(1, None, tb)
    assert sig is None and any("Traceback" in n for n in notes), "a code fault is not a memory fault"


def test_the_overlay_audit_sees_a_reward_function_swap(rc, tmp_path):
    """``_func_swapped_ok`` backs the audit for the only two func-swap knobs (C6's facing-coupled
    tracking term and C11's turn-gated yaw term) and was executed by no test."""
    swapped = _dump_for(rc, {"KRABBY_OMNI_TRACK_FACING_SIGMA": "0.5"})
    plain = _dump_for(rc, {})
    assert rc._func_swapped_ok(swapped, "track_lin_vel_xy_exp", "track_lin_vel_xy_exp_facing_coupled")
    assert not rc._func_swapped_ok(plain, "track_lin_vel_xy_exp",
                                   "track_lin_vel_xy_exp_facing_coupled")
    gated = _dump_for(rc, {"KRABBY_OMNI_ANG_TURN_ONLY": "1"})
    assert rc._func_swapped_ok(gated, "track_ang_vel_z_exp", "track_ang_vel_z_exp_turn_gated")
    assert not rc._func_swapped_ok(plain, "track_ang_vel_z_exp", "track_ang_vel_z_exp_turn_gated")
    # and it fails CLOSED on a dump that does not carry the term at all
    assert not rc._func_swapped_ok("rewards: {}\n", "track_lin_vel_xy_exp", "anything")
    # R5: the match is the func PATH's last component, not a substring of the block. A needle that
    # is a PREFIX of the func actually in the dump used to pass vacuously -- the audit would report
    # a swap that never happened, and the audit is the only thing that says a knob reached the cfg.
    assert not rc._func_swapped_ok(swapped, "track_lin_vel_xy_exp", "track_lin_vel_xy_exp")
    assert rc._func_swapped_ok(plain, "track_lin_vel_xy_exp", "track_lin_vel_xy_exp")
    assert not rc._func_swapped_ok(gated, "track_ang_vel_z_exp", "track_ang_vel_z")
    assert rc._func_swapped_ok("rewards:\n  t:\n    func: pkg.mod.the_swap\n    weight: 1\n",
                               "t", "the_swap"), "a dotted func path resolves the same way"
    # the whole audit, through the registered check that calls it
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text(swapped)
    rec = {"ckpt": str(tmp_path / "model_7999.pt")}
    name = "overlay KRABBY_OMNI_TRACK_FACING_SIGMA=0.5 in params/env.yaml"
    assert dict(rc.overlay_checks(rec, {"KRABBY_OMNI_TRACK_FACING_SIGMA": "0.5"}))[name] is True
    (tmp_path / "params" / "env.yaml").write_text(plain)
    assert dict(rc.overlay_checks(rec, {"KRABBY_OMNI_TRACK_FACING_SIGMA": "0.5"}))[name] is False


def test_dirty_paths_reads_renames_and_untracked_lines(rc, monkeypatch):
    """The launch refusal's own input. Untracked lines count and a rename is read at its
    DESTINATION -- that is the path ``code_sha`` would hash."""
    porcelain = "\n".join([
        ' M parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/a.py',
        'R  parkour/x/mdp/old.py -> parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/new.py',
        '?? parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/untracked.py',
        ' M parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/README.md',
        ' M tests/unit/test_x.py',
        ' M parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-19_x/CHARTER.md',
        '',
    ])
    monkeypatch.setattr(rc, "_git_out", lambda args, repo=None: porcelain)
    assert rc.dirty_paths() == [
        "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/a.py",
        "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/new.py",
        "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/untracked.py",
    ]
    monkeypatch.setattr(rc, "_git_out", lambda args, repo=None: "")
    assert rc.dirty_paths() == []


def test_the_d00c_write_path_runs_on_a_chain_twin_and_the_next_pooled_decision_cites_it(
        rc, cands, tmp_path):
    """FIX_QUEUE 6(a): the plan's "written as a lookup step BEFORE any pooled decision that would
    cite it". Only the ORDERING was pinned; the write itself ran in no test."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    base = rid(rc, cands, led, "C1")
    r_c1 = rc._recipe_with(led, cands, CODE, "C1", set())
    led["entries"].append(entry("D05", "C1", "ADOPT", "decided", experiment="X03", base=base,
                                recipe=r_c1["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C1", "3", r_c1["recipe_id"], base,
                                             base_rank("C1"), chain="curriculum")
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    assert rc.seed_terms_for(ctx, "curriculum")[1] == ["D00b"] or \
        rc.seed_terms_for(ctx, "curriculum")[1] == [], "no D00c yet"

    header = {"xid": "X10", "kind": "twin", "candidate": "C2", "seed": "2", "chain": "curriculum",
              "code_sha": ctx.code, "recipe_id": r_c1["recipe_id"], "base_recipe_id": base,
              "screen": screen(closed=0.55, prog=0.09)}
    item = rc._item("twin", cid="C2", seed="2", recipe=r_c1, base_recipe=r_c1)
    rc._d00c_lookup(ctx, item, header)
    d = rc.entry_by_id(ctx.ledger, "D00c-curriculum")
    assert d is not None and (d["kind"], d["status"]) == ("resolution", "final")
    assert d["evidence"]["chain"] == "curriculum"
    assert d["evidence"]["pair"] == {"s3": "X03", "s2": "X10"}
    assert d["evidence"]["seed_terms"]["blocks"]["F"] > 0
    assert "D00c-curriculum" in ctx.lad["resolution_ids"]
    assert any(" RESOLUTION D00c-curriculum " in l for l in notify_lines(ctx))

    # a decision on that chain now cites it, and a second twin does not re-fit it
    terms, cited = rc.seed_terms_for(ctx, "curriculum")
    assert cited == ["D00c-curriculum"]
    assert terms["blocks"]["F"] == d["evidence"]["seed_terms"]["blocks"]["F"]
    before = json.dumps(d)
    rc._d00c_lookup(ctx, item, dict(header, xid="X11"))
    assert json.dumps(rc.entry_by_id(ctx.ledger, "D00c-curriculum")) == before
    # and the baseline chain never writes one (its spread is D00b)
    rc._d00c_lookup(ctx, rc._item("twin", cid="C2", seed="2"),
                    dict(header, xid="X12", chain="scratch_1c"))
    assert rc.entry_by_id(ctx.ledger, "D00c-scratch_1c") is None


def test_the_done_tag_is_emitted_when_the_queue_empties(rc, cands, tmp_path, monkeypatch):
    """The notification contract lists thirteen tags and the assistant's Monitor relays all of
    them; ``DONE`` -- the one that says the campaign is over -- was emitted by no test."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    monkeypatch.setattr(ctx.args, "max_wall_h", None, raising=False)
    monkeypatch.setattr(rc, "next_item", lambda *a, **k: None)
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    assert rc.loop(ctx) == 0
    assert ctx.state["phase"] == "done"
    line = [l for l in notify_lines(ctx) if " DONE " in l][-1]
    got = rc.lc.parse_notify(line)
    assert got["tag"] == "DONE" and got["id"] == ctx.dir.name
    assert any("the queue is empty" in f for f in got["fields"])
    assert any("ledger entries" in f for f in got["fields"])


# ============================================================ the minor list
def test_the_validator_bound_checks_a_raw_pools_own_speed_scale(rc, cands):
    """PLAN_AMENDMENTS 3's bound was applied to a ``pool_edit`` only. A chain stage may write the
    same scale straight into its pool string (``name:weight:ang_scale:speed_scale``), and there it
    was unchecked -- every draw of that type would sit under the gait clock's 0.2 m/s stop speed."""
    doc = copy.deepcopy(cands)
    stage = doc["chains"]["curriculum"]["stages"][-1]
    pool = rc.parse_pool(stage["overlay"]["KRABBY_OMNI_CMD_POOL"])
    scaled = ",".join(f"{n}:{w}:1.0:0.2" if n == "fwd" else f"{n}:{w}"
                      for n, w in zip(pool.names, pool.weights))
    stage["overlay"] = dict(stage["overlay"], KRABBY_OMNI_CMD_POOL=scaled)
    out = [v for v in rc.validate_candidates(doc, strict=False, check_replay=False)
           if v["id"] == "chain:curriculum"][0]
    assert not out["ok"]
    assert any("speed_scale 0.2 on 'fwd'" in r and "walk-floor bound" in r
               for r in out["reasons"]), out["reasons"]
    # the shipped chains carry no speed_scale at all, so nothing else moves
    assert all(v["ok"] for v in rc.validate_candidates(copy.deepcopy(cands), strict=False,
                                                       check_replay=False))


def test_the_validator_refuses_an_overlay_that_names_a_stage_outside_every_chain(rc, cands):
    """``compose_recipe`` only appends such an overlay to ``conflicts``, which nothing in a live run
    reads -- the lever would be silently inert for the whole window and the DECISION line would
    report it as applied."""
    doc = copy.deepcopy(cands)
    doc["candidates"].append({"id": "CX", "kind": "reward", "question": "a typo'd stage",
                              "overlay": {"c7": {"KRABBY_YAW_W": "0.3"}}})
    out = [v for v in rc.validate_candidates(doc, strict=False, check_replay=False)
           if v["id"] == "CX"][0]
    assert not out["ok"] and any("is in no chain" in r for r in out["reasons"]), out["reasons"]

    # a REAL stage of another chain, on a candidate that declares its own, is named too
    other = copy.deepcopy(cands)
    other["candidates"].append({"id": "CY", "kind": "reward", "chain": "scratch_1c",
                                "question": "a stage of the wrong chain",
                                "overlay": {"c6": {"KRABBY_YAW_W": "0.3"}}})
    out = [v for v in rc.validate_candidates(other, strict=False, check_replay=False)
           if v["id"] == "CY"][0]
    assert not out["ok"] and any("not in its own chain scratch_1c" in r for r in out["reasons"])

    # and a 2a/2b overlay on a candidate with no chain of its own is still legal
    ok = copy.deepcopy(cands)
    ok["candidates"].append({"id": "CZ", "kind": "reward", "question": "a teacher-stage overlay",
                             "overlay": {"2a": {"KRABBY_YAW_W": "0.3"}}})
    assert [v for v in rc.validate_candidates(ok, strict=False, check_replay=False)
            if v["id"] == "CZ"][0]["ok"]


def test_a_rerun_header_names_the_window_it_supersedes(rc, cands, tmp_path, monkeypatch):
    """SQUASH (5): "reruns get new ids with ``supersedes``". Nothing on the EXPERIMENT said which
    squashed window a re-screen replaces, so a reader could not follow a re-run back to the hours
    it already cost."""
    ctx = _depth3_ladder(rc, cands, tmp_path)
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    assert "X05" in ctx.lad["squashed"] and ctx.lad["squashed"]["X05"]["candidate"] == "C2"
    install_fakes(rc, monkeypatch, tmp_path / "rerun")
    item = candidate_item(rc, cands, ctx, "C2")
    header = rc.run_experiment(ctx, item)
    assert header["xid"] != "X05"
    assert header["supersedes"] == ["X05"], header["supersedes"]
    # a first-ever window supersedes nothing
    assert rc.run_experiment(ctx, candidate_item(rc, cands, ctx, "C6"))["supersedes"] == []


def test_the_await_line_quotes_a_command_that_can_be_run_verbatim(rc, cands, tmp_path, monkeypatch):
    """The AWAIT line's ``--choose run|skip`` was sanitised to ``run/skip`` by ``_field`` (``|`` is
    the SEPARATOR), so the one command the contract says to run verbatim was not runnable."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in [c["id"] for c in cands["candidates"] if c["id"] != "C12"]:
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    for mid in ("M1", "M2"):
        led["entries"].append({"id": mid, "kind": "milestone", "candidate": mid, "chosen": "PASS",
                               "status": "final", "options": ["PASS", "WATCH", "FAIL"],
                               "history": [], "evidence": {},
                               "recipe_id": rid(rc, cands, led, "C7", inclusive=True)})
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    monkeypatch.setattr(ctx.args, "max_wall_h", None, raising=False)
    monkeypatch.setattr(rc.rp, "_wait_isaac_clear", lambda *a, **k: None)
    assert rc.loop(ctx) == 0
    assert ctx.state["phase"] == "await_user:C12"
    fields = rc.lc.parse_notify([l for l in notify_lines(ctx) if " AWAIT " in l][-1])["fields"]
    text = " | ".join(fields)
    assert "run/skip" not in text, "a sanitised separator is not a command"
    assert "--choose run" in text and "--choose skip" in text


def test_unstop_says_whether_the_driver_will_see_it(rc, cli, cands, tmp_path, capsys):
    """Every other verb prints it, and ``unstop`` is the one a user types expecting the campaign to
    RESUME -- which a parked driver does not do until it is relaunched."""
    ctx = make_ctx(rc, cands, tmp_path, led=ledger(), st=state())
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "stop"], capsys)
    assert code == 0 and "driver not running" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "unstop"], capsys)
    assert code == 0 and "removed" in out
    assert "driver not running — applied at next launch" in out
    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "unstop"], capsys)
    assert code == 0 and "no " in out and "applied at next launch" in out


def test_the_override_applied_line_names_the_squashed_windows_and_their_hours(rc, cands, tmp_path):
    """The plan's contract example: ``OVERRIDE-APPLIED <uuid> D08 -> REJECT | squashed: X12 (2.1 h)
    | resume: ...``. The line counted them (``squashed 5``) and never said WHICH or what they cost,
    although ``wall_h`` is recorded on every stage record -- and the user is being asked to confirm
    an answer whose price is GPU hours already spent."""
    ctx = _depth3_ladder(rc, cands, tmp_path)
    ctx.state.setdefault("keys", {})["seed3"] = {
        "X05:c1": {"wall_h": 1.5}, "X05:c2": {"wall_h": 2.4},
    }
    ctx.lad["experiments"]["X05"]["recs"] = ["X05:c1", "X05:c2"]
    ctx.c.save()
    rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "overrule", "choice": "REJECT",
                                    "why": "wrong"})
    rc.ingest_overrides(ctx)
    line = [l for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l][-1]
    fields = rc.lc.parse_notify(line)["fields"]
    note = fields[0]
    assert "squashed 5: " in note and "X05 (3.9 h)" in note, note
    assert "X04" in note and "X08" in note, "every squashed window is named"
    assert any(f.startswith("resume: ") for f in fields)


def test_a_re_applied_override_line_is_relayed_as_already_applied_not_refused(rc, cands, tmp_path):
    """F10(b)+(c): the crash window between the ledger write and the ``applied_overrides`` save.

    The relaunch re-reads the line, the ledger already carries it, and the old code answered the
    user's standing ``confirm`` with ``REFUSED: illegal ledger transition 'confirmed' ->
    'confirmed'`` -- the Monitor would have relayed an applied answer as rejected. And the uuid is
    now recorded as applied BEFORE the apply, so the ``ctx.save()`` inside ``squash_and_report``
    carries it and the window closes.
    """
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    led["entries"].append(entry("D05", "C2", "ADOPT", "decided", experiment="X03"))
    ctx = make_ctx(rc, cands, tmp_path, led=led, st=st)
    uid = rc.lc.append_override(ctx.dir, {"decision": "D05", "action": "confirm",
                                          "choice": "ADOPT", "why": "yes"})
    rc.ingest_overrides(ctx)
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "confirmed"
    assert uid in ctx.lad["applied_overrides"]

    ctx.lad["applied_overrides"] = []          # the crash: the ledger landed, the record did not
    ctx._ovr_stamp = None
    rc.ingest_overrides(ctx)
    note = rc.lc.parse_notify([l for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l][-1])
    assert "already applied" in note["fields"][0], note["fields"]
    assert "REFUSED" not in note["fields"][0]
    assert rc.entry_by_id(ctx.ledger, "D05")["status"] == "confirmed"
    assert uid in ctx.lad["applied_overrides"]

    # the same for `decide run`, whose re-apply hit `squashed -> squashed (user)`
    uid2 = rc.lc.append_override(ctx.dir, {"candidate": "C5", "choice": "run", "why": "re-screen"})
    ctx.ledger["entries"].append(entry("D06", "C5", "REJECT", "decided", experiment="X04"))
    ctx.write_ledger()
    ctx.recompute()
    rc.ingest_overrides(ctx)
    assert rc.entry_by_id(ctx.ledger, "D06")["status"] == "squashed"
    ctx.lad["applied_overrides"] = [uid]
    ctx._ovr_stamp = None
    rc.ingest_overrides(ctx)
    notes = [rc.lc.parse_notify(l)["fields"][0] for l in notify_lines(ctx)
             if " OVERRIDE-APPLIED " in l and uid2 in l]
    assert notes and "REFUSED" not in notes[-1], notes
    d06 = rc.entry_by_id(ctx.ledger, "D06")
    assert d06["status"] == "squashed" and d06["override"]["id"] == uid2
    assert len([h for h in d06["history"] if h["to"] == "squashed"]) == 1, "squashed once, not twice"


def _milestone_await(rc, cands, tmp_path, stage, name):
    """M1 parked at ``await`` after a (i) or (ii) FAIL, with one unfrozen ADOPT (C2) in its block."""
    led, st = ledger(), state()
    with_baseline(rc, cands, led, st)
    for cid in ("C1", "C1b", "C1c", "C3b"):
        led["entries"].append(entry(f"D_{cid}", cid, "REJECT", "decided"))
    r_c2 = rc._recipe_with(led, cands, CODE, "C2", set())
    led["entries"].append(entry("D05", "C2", "ADOPT", "decided", experiment="X03",
                                recipe=r_c2["recipe_id"]))
    st["ladder"]["experiments"]["X03"] = exp("X03", "candidate", "C2", "3", r_c2["recipe_id"],
                                             rid(rc, cands, led, "C2"), base_rank("C2"))
    led["entries"].append(entry("D06", "C3a", "REJECT", "decided", experiment="X04"))
    led["entries"].append({"id": "M1", "kind": "milestone", "candidate": "M1", "chosen": "FAIL",
                           "status": "await", "options": ["PASS", "WATCH", "FAIL"], "history": [],
                           "evidence": {"stage": stage}, "eligibility": 1,
                           "recipe_id": rc.milestone_recipe(cands, led, "M1", CODE)["recipe_id"]})
    return make_ctx(rc, cands, tmp_path, led=led, st=st, name=name)


def _replay_last_line(rc, ctx):
    """The crash window: the ledger write landed, ``applied_overrides`` did not."""
    ctx.lad["applied_overrides"] = []
    ctx._ovr_stamp = None
    before = len([l for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l])
    rc.ingest_overrides(ctx)
    return [rc.lc.parse_notify(l)["fields"][0]
            for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l][before:]


@pytest.mark.parametrize("stage", ["ii", "i"])
def test_a_replayed_milestone_accept_is_relayed_as_already_applied(rc, cands, tmp_path, stage):
    """F10(b), extended to the third verb (amendment 28).

    ``--accept`` writes ``ledger.json`` and only the ingest's ``ctx.save()`` records the uuid, so a
    hard kill in between replays the line. At stage (ii) the milestone is ``final`` by then and the
    guard relayed ``REFUSED: M1 is not awaiting an answer`` for an applied answer; at stage (i) the
    status is still ``await``, so the guard PASSED and the line was applied a SECOND time -- a
    duplicate history row and an ``OVERRIDE-APPLIED`` that reads ``frozen adoptions []``.
    """
    ctx = _milestone_await(rc, cands, tmp_path, stage, f"m{stage}")
    uid = rc.lc.append_override(ctx.dir, {"milestone": "M1", "accept": True, "why": "stands"})
    rc.ingest_overrides(ctx)
    m1, d05 = rc.entry_by_id(ctx.ledger, "M1"), rc.entry_by_id(ctx.ledger, "D05")
    assert m1["accepted_uuid"] == uid
    assert m1["status"] == ("final" if stage == "ii" else "await")
    assert d05["status"] == "confirmed" and d05["frozen"]
    rows = (len(m1["history"]), len(d05["history"]))

    notes = _replay_last_line(rc, ctx)
    assert notes and "REFUSED" not in notes[-1], notes
    assert "already accepted by this line" in notes[-1], notes
    m1, d05 = rc.entry_by_id(ctx.ledger, "M1"), rc.entry_by_id(ctx.ledger, "D05")
    assert (len(m1["history"]), len(d05["history"])) == rows, "applied once, not twice"
    assert m1["status"] == ("final" if stage == "ii" else "await")
    assert uid in ctx.lad["applied_overrides"]


def test_a_replayed_milestone_drop_is_relayed_as_already_applied(rc, cands, tmp_path):
    """The same window on ``--drop``: the target is ``overruled`` by that very uuid by then.

    Re-applying hit ``--drop is refused unless C2 is an ADOPT in M1's block`` -- a REFUSED relayed
    for an answer whose squash already stands.
    """
    ctx = _milestone_await(rc, cands, tmp_path, "ii", "mdrop")
    uid = rc.lc.append_override(ctx.dir, {"milestone": "M1", "drop": "C2", "why": "drop it"})
    rc.ingest_overrides(ctx)
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "overruled" and d05["override"]["id"] == uid
    rows = len(d05["history"])

    notes = _replay_last_line(rc, ctx)
    assert notes and "REFUSED" not in notes[-1], notes
    assert "already dropped by this line" in notes[-1], notes
    d05 = rc.entry_by_id(ctx.ledger, "D05")
    assert d05["status"] == "overruled" and len(d05["history"]) == rows
    assert uid in ctx.lad["applied_overrides"]


def test_a_second_accept_on_an_accepted_milestone_is_already_applied_in_both_halves(rc, cli, cands,
                                                                                    tmp_path,
                                                                                    capsys):
    """The F2 wart: ``await -> accepted_i`` is the plan's row, and it is answered ONCE.

    ``accepted_i`` leaves the status at ``await`` (the promotion still has to run), so a re-typed
    answer -- or an old MILESTONE notification run again -- passed both guards, froze nothing new,
    added a history row and reported ``frozen adoptions []``: a line that reads like a refusal
    nobody made. The CLI now answers rc 0 with the note and appends nothing, and the driver is a
    no-op if the line reaches it anyway.
    """
    ctx = _milestone_await(rc, cands, tmp_path, "i", "twice")
    rc.lc.append_override(ctx.dir, {"milestone": "M1", "accept": True, "why": "stands"})
    rc.ingest_overrides(ctx)
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert (m1["status"], m1["sub_status"]) == ("await", "accepted_i")
    lines = len((ctx.dir / "ledger_overrides.jsonl").read_text().splitlines())
    rows = len(m1["history"])

    code, out = cli_run(cli, ["--campaign-dir", str(ctx.dir), "resume-milestone", "M1", "--accept"],
                        capsys)
    assert code == 0 and "already accepted" in out and "answered once" in out, out
    assert len((ctx.dir / "ledger_overrides.jsonl").read_text().splitlines()) == lines, \
        "an already-applied verb appends nothing"

    # and a line that reaches the driver anyway (a stale file, another terminal) is a no-op
    uid = rc.lc.append_override(ctx.dir, {"milestone": "M1", "accept": True, "why": "again"})
    rc.ingest_overrides(ctx)
    note = rc.lc.parse_notify([l for l in notify_lines(ctx) if " OVERRIDE-APPLIED " in l][-1])
    assert "already accepted" in note["fields"][0] and "REFUSED" not in note["fields"][0]
    m1 = rc.entry_by_id(ctx.ledger, "M1")
    assert (m1["status"], m1["sub_status"]) == ("await", "accepted_i")
    assert len(m1["history"]) == rows and m1["accepted_uuid"] != uid
    assert uid in ctx.lad["applied_overrides"]
