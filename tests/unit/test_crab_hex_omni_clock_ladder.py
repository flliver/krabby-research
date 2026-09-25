"""The clock-deprecation ladder's driver work (plan step 4, 2026-09-23): block RIG, the five clock
knobs' parsers and dump checks, the validator rules (i)-(v), ``control: baseline``,
``decision_rule.pair_on``, the ``arm_class`` decision field, the flag list and its driver park (with
the ``ledger_cli ack-flag`` release across a relaunch, the pilot stop and the baseline-floor park),
the ramp-stage whole-window restart, the reference-chain launcher with its ``reference`` resume, the
S3 rule in ``ledger_cli`` and a clean ``--dry-run``.

Pure Python. ``run_candidates.py`` / ``ledger_cli.py`` / ``resolution.py`` are loaded by path; the
fake training/eval children, ``make_ctx`` and the suite screens are
``test_crab_hex_omni_candidates``' own helpers (loaded by path, so they are the code the rest of the
suite trusts). The campaign spec is ``tests/unit/fixtures/candidates_clock_2026-09-23.yaml`` -- the
shape plan step 10 writes. Nothing here imports Isaac, launches a subprocess or touches a real
campaign directory.
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
FIXTURE = REPO / "tests/unit/fixtures/candidates_clock_2026-09-23.yaml"
MQ_YAML = OMNI / "experiments/2026-09-22_gait_motion_quality/candidates.yaml"
CATS = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
TRANSLATION = ("fwd", "side", "back", "diag")
CODE = "CODE0"
EVAL = "e" * 64
K = {"KRABBY_CLOCK_W": "0.001", "KRABBY_APEX_W": "0.001", "KRABBY_OMNI_CLOCK_HIDE": "1"}
PHASEOUT = ("reward_clock_schedule:1.0:0.001:0:24000,"
            "reward_clock_swing_apex:1.0:0.001:0:24000")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load(TOOLS / "run_candidates.py", "run_candidates_clock_ladder_test")


@pytest.fixture(scope="module")
def cli(rc):
    return rc.lc


@pytest.fixture(scope="module")
def H():
    """``test_crab_hex_omni_candidates``' helpers (screens, ctx, the fake children), by path."""
    return _load(REPO / "tests/unit/test_crab_hex_omni_candidates.py", "omni_candidates_helpers_for_clock")


@pytest.fixture(scope="module")
def cands(rc):
    return rc.load_candidates(FIXTURE)


@pytest.fixture
def rule(cands):
    return copy.deepcopy(cands["decision_rule"])


def _refusals(rc, doc, subject=None) -> list[str]:
    verdicts = rc.validate_candidates(doc, strict=False, check_replay=False)
    return [r for v in verdicts if subject is None or v["id"] == subject for r in v["reasons"]]


def write_doc(rc, doc: dict, tmp_path: Path, name: str = "candidates.yaml") -> dict:
    """A copy of ``doc`` on disk (``make_ctx`` / ``main`` read the FILE), reloaded."""
    import yaml
    path = Path(tmp_path) / "specs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({k: v for k, v in doc.items() if k != "__path__"}, sort_keys=False))
    return rc.load_candidates(path)


# ------------------------------------------------------------------------------ screens / resolution
RIG_ALL = ("steps_per_m", "steps_per_rad", "support_feet", "steps_per_s", "taps_per_m", "taps_per_rad",
           "swing_apex", "step_rate_hz", "foot_slip")


def rscreen(H, *, steps=11.0, steps_rad=6.0, support=3.0, taps=1.0, apex=0.05, rate=1.0, slip=0.05,
            stand_steps=0.5, ratio=0.60, surv=0.95, closed=0.50, prog=0.080, **over) -> dict:
    """A suite screen (``H.screen_cat``) with the rigidity keys and ``foot_slip`` on the categories
    they are defined on (None elsewhere); ``over`` takes dotted ids with ``__`` for ``.``."""
    out = H.screen_cat(ratio=ratio, surv=surv, closed=closed, prog=prog)
    for c in CATS:
        e = out["cat"][c]
        e.update({k: None for k in RIG_ALL})
        e["support_feet"], e["foot_slip"] = support, slip
        if c in TRANSLATION:
            e.update(steps_per_m=steps, taps_per_m=taps, swing_apex=apex, step_rate_hz=rate)
        if c in ("turn", "move_turn"):
            e.update(steps_per_rad=steps_rad, taps_per_rad=taps, swing_apex=apex, step_rate_hz=rate)
        if c == "stand":
            e["steps_per_s"] = stand_steps
        for hand in (e.get("by_hand") or {}).values():
            hand.update({k: e[k] for k in RIG_ALL})
    for key, value in over.items():
        H._put(out, key.replace("__", "."), value)
    return out


def rres(rc, rule, *, metric_R=0.10, guard_R=0.10, block_R=0.50, guard_key_R=0.20) -> dict:
    """A synthetic D00s for the fixture rule: round R per member / guard (the dual guards -- also
    block members -- under ``<mid>@guard`` with their own R)."""
    ec = rc.eval_categories()
    union = ec.expand_rule_union(rule, CATS, CATS)
    members, metrics = set(), {}
    for body in union["blocks"].values():
        for mid in body["metrics"]:
            members.add(mid)
            metrics[mid] = {"R_eval": metric_R, "level": 0.95, "n": 100, "source": "cat"}
    for mid in union["guards"]["metrics"]:
        if mid in members:
            metrics[ec.guard_key(mid)] = {"R_eval": guard_key_R, "level": 0.99, "n": 100, "source": "cat",
                                          "block": "guard"}
        else:
            metrics[mid] = {"R_eval": guard_R, "level": 0.99, "n": 100, "source": "cat"}
    blocks = {b: {"R_eval": block_R, "level": 0.95, "members": []} for b in union["blocks"]}
    return {"schema": 1, "rule_version": 2, "B": 2000, "rng_seed": 1, "metrics": metrics, "blocks": blocks,
            "decisive_at": list(CATS), "process_floor": {}, "seed_terms": {}, "inputs": [], "by_row": {}}


def _decide(rc, rule, cand, base, resolution):
    return rc.decide(cand, base, resolution, cand_spec={}, rule=rule, seed_terms={}, anchor_screen=None)


def rig_better(H, **kw):
    """Every RIG member one R_eval better (z = +1 > the block R 0.5): RIG BETTER."""
    return rscreen(H, steps=10.9, steps_rad=5.9, support=3.1, **kw)


# ------------------------------------------------------------------------------ config dumps
def clock_dump(*, clock_w=1.0, apex_w=1.0, phaseout=None, hide=False, ramp=None) -> str:
    """A ``params/env.yaml`` excerpt in the real dump's shape: the omni obs term's params, the two
    clock RewTerms, and the ``curriculum`` section (``null`` without a PHASEOUT)."""
    lines = ["observations:", "  policy:", "    extreme_parkour_observations:", "      params:",
             "        facing_obs_clip: 0.6"]
    if hide:
        lines.append("        clock_hide: true")
    if ramp:
        lines += ["        clock_hide_ramp: !!python/tuple", f"        - {ramp[0]}", f"        - {ramp[1]}"]
    lines += ["rewards:", "  reward_clock_schedule:", "    func: x:RewardClockContactSchedule",
              f"    weight: {clock_w}", "    params:", "      combine: sum",
              "  reward_clock_swing_apex:", "    func: x:RewardClockSwingApex", f"    weight: {apex_w}",
              "    params:", "      apex_m: 0.05", "terminations:", "  time_out: null"]
    if phaseout:
        lines.append("curriculum:")
        for chunk in phaseout.split(","):
            term, w0, w1, t0, t1 = chunk.split(":")
            lines += [f"  phaseout_{term}:", "    func: parkour_tasks.x.curriculums:ramp_reward_weight",
                      "    params:", f"      term_name: {term}", f"      w0: {w0}", f"      w1: {w1}",
                      f"      t0: {t0}", f"      t1: {t1}"]
    else:
        lines.append("curriculum: null")
    lines += ["commands:", "  base_velocity:", "    debug_vis: false"]
    return "\n".join(lines) + "\n"


def clock_section(env: dict) -> str:
    """What the real dump would add for a window's clock overlay (appended to ``H._dump_for``'s)."""
    lines = ["clock_terms:", "  reward_clock_schedule:", "    func: x:y",
             f"    weight: {env.get('KRABBY_CLOCK_W', '1.0')}", "  reward_clock_swing_apex:", "    func: x:y",
             f"    weight: {env.get('KRABBY_APEX_W', '1.0')}", "clock_obs:"]
    if env.get("KRABBY_OMNI_CLOCK_HIDE") == "1":
        lines.append("      clock_hide: true")
    ramp = env.get("KRABBY_OMNI_CLOCK_HIDE_RAMP")
    if ramp:
        a, b = ramp.split(":")
        lines += ["      clock_hide_ramp: !!python/tuple", f"      - {a}", f"      - {b}"]
    if env.get("KRABBY_PHASEOUT"):
        lines.append("curriculum:")
        for chunk in env["KRABBY_PHASEOUT"].split(","):
            term, w0, w1, t0, t1 = chunk.split(":")
            lines += [f"  phaseout_{term}:", "    func: x:ramp_reward_weight", "    params:",
                      f"      term_name: {term}", f"      w0: {w0}", f"      w1: {w1}", f"      t0: {t0}",
                      f"      t1: {t1}"]
    return "\n".join(lines) + "\n"


def install_clock_fakes(rc, H, monkeypatch, tmp_path, **kw):
    """``H.install_fakes`` whose training dump also carries the clock knobs (so the overlay audit of a
    K / PHASEOUT / HIDE window passes)."""
    calls = H.install_fakes(rc, monkeypatch, tmp_path, **kw)
    fake = rc.rp.train_retry

    def train_retry(c, spec, tag, iters, resume, soundness, early_abort=False, **k):
        ckpt, status, log = fake(c, spec, tag, iters, resume, soundness, early_abort, **k)
        if ckpt:
            dump = Path(ckpt).parent / "params" / "env.yaml"
            dump.write_text(dump.read_text() + clock_section(dict(k.get("extra_env") or {})))
        return ckpt, status, log

    monkeypatch.setattr(rc.rp, "train_retry", train_retry)
    return calls


# ================================================================== block RIG
def test_rig_is_a_known_block_in_both_tools_and_the_fixture_requires_it(rc, cands):
    res = rc.resolution_tool()
    assert "RIG" in rc.RULE_V2_BLOCKS and "RIG" in res.BLOCK_ORDER
    assert rc.rule_v2_required_blocks(cands["decision_rule"]) == ["F", "F_mt", "T", "RIG"]
    assert rc.flags_enabled(cands["decision_rule"]) is True
    mq = rc.load_candidates(MQ_YAML)
    assert rc.flags_enabled(mq["decision_rule"]) is False, "block Q never arms the RIG machinery"
    ec = rc.eval_categories()
    exp = ec.expand_rule(cands["decision_rule"], CATS, CATS, absolute_units=False)
    rig = exp["blocks"]["RIG"]["metrics"]
    assert len(rig) == 10 and len(exp["guards"]["metrics"]) == 23
    assert ec.metric_sign("cat.fwd.steps_per_m") == -1 and ec.metric_sign("cat.fwd.support_feet") == 1


def test_the_fixture_campaign_validates_clean(rc, cands):
    verdicts = rc.validate_candidates(copy.deepcopy(cands), strict=False, check_replay=False)
    assert all(v["ok"] for v in verdicts), [(v["id"], v["reasons"]) for v in verdicts if not v["ok"]]
    # the reference-only chains are exempt from screen.iters (Z is 8000 its, t3_s2 3000)
    assert rc.reference_only_chains(cands) == {"scratch_free", "t3_clock", "tail_s2_held"}


# ================================================================== parsers + dump checks
def test_the_five_clock_knobs_are_whitelisted_parsed_and_audited(rc):
    wl, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for key in rc.CLOCK_KNOB_KEYS:
        assert key in wl and key in parsers and key in rc.OVERLAY_DUMP_CHECKS, key
    assert parsers["KRABBY_PHASEOUT"](PHASEOUT)[1]["term_name"] == "reward_clock_swing_apex"
    reasons = []
    rc._check_knob("KRABBY_PHASEOUT", "reward_clock_schedule:1.0:0.001:10:5", "x", wl, parsers, reasons, set())
    rc._check_knob("KRABBY_OMNI_CLOCK_HIDE", "true", "x", wl, parsers, reasons, set())
    rc._check_knob("KRABBY_OMNI_CLOCK_HIDE_RAMP", "48000:24000", "x", wl, parsers, reasons, set())
    rc._check_knob("KRABBY_CLOCK_W", "eps", "x", wl, parsers, reasons, set())
    assert len(reasons) == 4 and all("does not round-trip" in r for r in reasons), reasons


def test_the_clock_weight_dump_checks(rc):
    chk = rc.OVERLAY_DUMP_CHECKS
    dump = clock_dump(clock_w=0.001, apex_w=0.001)
    assert chk["KRABBY_CLOCK_W"](dump, "0.001") and chk["KRABBY_APEX_W"](dump, "0.001")
    assert not chk["KRABBY_CLOCK_W"](dump, "1.0") and not chk["KRABBY_APEX_W"](dump, "1.0")
    held = clock_dump()
    assert chk["KRABBY_CLOCK_W"](held, "1.0") and not chk["KRABBY_CLOCK_W"](held, "0.001")
    # each reads its OWN term, not the nearest `weight:`
    mixed = clock_dump(clock_w=0.001, apex_w=1.0)
    assert chk["KRABBY_CLOCK_W"](mixed, "0.001") and not chk["KRABBY_APEX_W"](mixed, "0.001")
    assert not chk["KRABBY_CLOCK_W"]("rewards:\n  other:\n    weight: 0.001\n", "0.001")


def test_the_phaseout_dump_check_wants_exactly_the_matching_blocks(rc):
    ok = rc.OVERLAY_DUMP_CHECKS["KRABBY_PHASEOUT"]
    assert ok(clock_dump(phaseout=PHASEOUT), PHASEOUT)
    one = "reward_clock_schedule:1.0:0.001:0:24000"
    assert not ok(clock_dump(phaseout=one), PHASEOUT), "a missing block"
    assert not ok(clock_dump(phaseout=PHASEOUT), one), "an extra block"
    assert not ok(clock_dump(phaseout=PHASEOUT.replace("0.001:0:24000", "0.01:0:24000", 1)), PHASEOUT)
    assert not ok(clock_dump(phaseout=PHASEOUT.replace(":24000", ":24024", 1)), PHASEOUT)
    assert not ok(clock_dump(), PHASEOUT), "curriculum: null"
    wrong_name = clock_dump(phaseout=PHASEOUT).replace("term_name: reward_clock_schedule",
                                                       "term_name: reward_stride_length")
    assert not ok(wrong_name, PHASEOUT)


def test_the_hide_and_ramp_dump_checks(rc):
    hide, ramp = rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_CLOCK_HIDE"], rc.OVERLAY_DUMP_CHECKS["KRABBY_OMNI_CLOCK_HIDE_RAMP"]
    assert hide(clock_dump(hide=True), "1") and not hide(clock_dump(hide=True), "0")
    assert hide(clock_dump(), "0") and not hide(clock_dump(), "1"), "off writes nothing: absent"
    both = clock_dump(hide=True, ramp=(24000, 48000))
    assert hide(both, "1"), "clock_hide_ramp is never read as clock_hide"
    assert ramp(both, "24000:48000") and not ramp(both, "24000:48024")
    assert not ramp(clock_dump(hide=True), "24000:48000")
    assert ramp(clock_dump(hide=True), ""), "a blank ramp is absent"


def test_overlay_checks_audit_a_k_window_and_halt_on_a_mismatch(rc, tmp_path):
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    (run / "params" / "env.yaml").write_text(clock_dump(clock_w=0.001, apex_w=0.001, hide=True))
    rec = {"ckpt": str(run / "model_1.pt")}
    got = dict(rc.overlay_checks(rec, K))
    assert all(got.values()) and len(got) == 3, got
    (run / "params" / "env.yaml").write_text(clock_dump(clock_w=0.001, apex_w=1.0, hide=True))
    got = dict(rc.overlay_checks(rec, K))
    assert [n for n, ok in got.items() if not ok] == ["overlay KRABBY_APEX_W=0.001 in params/env.yaml"]


# ================================================================== the validator rules (i)-(v)
def _with_overlay(cands, cid, overlay, **extra) -> dict:
    doc = copy.deepcopy(cands)
    cand = next(c for c in doc["candidates"] if c["id"] == cid)
    cand["overlay"] = overlay
    cand.update(extra)
    return doc


def test_rule_i_refuses_a_clock_weight_of_zero(rc, cands):
    doc = _with_overlay(cands, "A6000", {"c6": dict(K, KRABBY_CLOCK_W="0.0")})
    got = _refusals(rc, doc, "A6000")
    assert any("(i)" in r and "KRABBY_CLOCK_W=0.0" in r for r in got), got
    doc = copy.deepcopy(cands)
    doc["chains"]["scratch_free"]["stages"][0]["overlay"]["KRABBY_APEX_W"] = "0"
    assert any("(i)" in r and "KRABBY_APEX_W=0.0" in r for r in _refusals(rc, doc, "chain:scratch_free"))


def test_rule_ii_hide_without_a_ramp_needs_both_weights_at_eps(rc, cands):
    doc = _with_overlay(cands, "A6000", {"c6": {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_CLOCK_W": "0.001"}})
    got = _refusals(rc, doc, "A6000")
    assert any("(ii)" in r and "reward_clock_swing_apex" in r for r in got), got
    # a PHASEOUT that still pays while the clock is hidden (no ramp) is the same non-Markov reward
    doc = _with_overlay(cands, "A6000", {"c6": {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_PHASEOUT": PHASEOUT}})
    assert any("(ii)" in r for r in _refusals(rc, doc, "A6000"))


def _w_doc(cands, overlay=None, *, iters=4000) -> dict:
    """Wave 2's shape: a one-stage c6x chain forked from a c6 head, candidate W with the reward
    PHASEOUT then the per-episode hide ramp (the plan's step 14 table)."""
    doc = copy.deepcopy(cands)
    c6 = copy.deepcopy(doc["chains"]["tail"]["stages"][2])
    c6.update(stage="c6x", iters=iters, resume_from={"checkpoint": "parkour/x/model_7996.pt"},
              floor={"kind": "category_survival", "row": "suite",
                     "min_by_category": {c: 0.5 for c in CATS}})
    doc["chains"] = {"c6x": {"stages": [c6]}, **{n: doc["chains"][n] for n in
                                                 ("scratch_free", "t3_clock", "tail_s2_held")}}
    doc["screen"] = {"chain": "c6x", "eval": ["suite"], "iters": iters}
    doc["baseline"] = {"id": "X0x", "chain": "c6x"}
    doc["decision_rule"]["pilot"] = "W"
    ov = {"KRABBY_PHASEOUT": PHASEOUT, "KRABBY_OMNI_CLOCK_HIDE": "1",
          "KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48000"} if overlay is None else overlay
    doc["candidates"] = [{"id": "W", "kind": "clock", "control": "baseline",
                          "eval_carry": ["KRABBY_OMNI_CLOCK_HIDE"], "question": "wave 2",
                          "overlay": {"c6x": ov}}]
    return doc


def test_rule_iii_the_wave_2_schedule_validates(rc, cands):
    doc = _w_doc(cands)
    assert not _refusals(rc, doc), _refusals(rc, doc)
    st = rc.clock_stage_state(rc._effective_stage_cfg(doc, doc["chains"]["c6x"]["stages"][0],
                                                      {**doc["chains"]["c6x"]["stages"][0]["overlay"],
                                                       **doc["candidates"][0]["overlay"]["c6x"]}), 4000)
    assert st["hide"] and st["ramp"] == (24000, 48000)
    assert st["start_w"] == {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": 1.0}
    assert st["end_w"] == {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.001}


def test_an_absent_clock_knob_is_judged_held_not_removed(rc):
    """Step-6 fix 6: a composed env silent about KRABBY_CLOCK_W / KRABBY_APEX_W never reads as 0.0
    ("removed"): a hidden clock on such a stage is refused by rule (ii), not validated."""
    st = rc.clock_stage_state({"KRABBY_OMNI_CLOCK_HIDE": "1"}, 100)
    assert st["static"] == {"reward_clock_schedule": 1.0, "reward_clock_swing_apex": 1.0}
    assert st["end_w"] == st["max_w"] == st["static"]
    assert sorted(st["defaulted"]) == ["KRABBY_APEX_W", "KRABBY_CLOCK_W"]
    reasons = []
    rc._check_clock_chain([("c6", 100, {"KRABBY_OMNI_CLOCK_HIDE": "1"})], "x", reasons)
    assert any("(ii)" in r for r in reasons), reasons
    # an explicit knob is still read as given
    st = rc.clock_stage_state({"KRABBY_CLOCK_W": "0.001", "KRABBY_APEX_W": "0.5"}, 100)
    assert st["static"] == {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.5}
    assert not st["defaulted"]


@pytest.mark.parametrize("edit, needle", [
    ({"KRABBY_OMNI_CLOCK_HIDE": "0"}, "(iii) KRABBY_OMNI_CLOCK_HIDE_RAMP needs"),
    ({"KRABBY_PHASEOUT": "reward_clock_schedule:1.0:0.001:0:24000"}, "exactly the two clock terms"),
    ({"KRABBY_PHASEOUT": PHASEOUT + ",reward_stride_length:0.5:0.001:0:24000"}, "exactly the two clock terms"),
    ({"KRABBY_PHASEOUT": PHASEOUT.replace(":24000", ":36000")}, "PHASEOUT t1 <= ramp t0"),
    ({"KRABBY_PHASEOUT": PHASEOUT.replace("0.001:0:24000", "0.01:0:24000")}, "while the clock still pays"),
    ({"KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:120000"}, "beyond the stage end 96000"),
    ({"KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48010"}, "not a multiple of 24"),
    ({"KRABBY_PHASEOUT": PHASEOUT.replace(":24000", ":24010")}, "not a multiple of 24"),
])
def test_rule_iii_refusals(rc, cands, edit, needle):
    ov = {"KRABBY_PHASEOUT": PHASEOUT, "KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48000"}
    ov.update(edit)
    got = _refusals(rc, _w_doc(cands, ov), "W")
    assert any(needle in r for r in got), got


def test_rule_iv_removal_is_monotone_along_a_chain(rc, cands):
    shown_again = _with_overlay(cands, "A4500", {"c5": dict(K), "c6": {"KRABBY_CLOCK_W": "0.001",
                                                                      "KRABBY_APEX_W": "0.001"}})
    got = _refusals(rc, shown_again, "A4500")
    assert any("(iv) shows the clock again after stage c5" in r for r in got), got
    paid_again = _with_overlay(cands, "A4500", {"c5": dict(K), "c6": dict(K, KRABBY_CLOCK_W="1.0")})
    got = _refusals(rc, paid_again, "A4500")
    assert any("(iv) reward_clock_schedule rises to 1.0 after stage c5" in r for r in got), got
    rising = _w_doc(cands, {"KRABBY_PHASEOUT": "reward_clock_schedule:0.001:1.0:0:24000,"
                                               "reward_clock_swing_apex:1.0:0.001:0:24000",
                            "KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48000"})
    assert any("(iv) KRABBY_PHASEOUT RAISES reward_clock_schedule" in r for r in _refusals(rc, rising, "W"))
    # a chain alone (Z's): K at t3 but not at c4 shows the clock again
    doc = copy.deepcopy(cands)
    for key in K:
        doc["chains"]["scratch_free"]["stages"][1]["overlay"].pop(key)
    got = _refusals(rc, doc, "chain:scratch_free")
    assert any("(iv)" in r and "stage c4" in r for r in got), got


def test_rule_v_a_clock_candidate_ends_phase_1_clock_free(rc, cands):
    early_only = _with_overlay(cands, "A4500", {"c5": dict(K)})
    got = _refusals(rc, early_only, "A4500")
    assert any("(iv)" in r for r in got) and any("(v)" in r and "final stage c6 shows the clock" in r
                                                 for r in got), got
    nothing = _with_overlay(cands, "A6000", {})
    got = _refusals(rc, nothing, "A6000")
    assert any("(v)" in r and "shows the clock" in r for r in got) and any(
        "(v)" in r and "ends at" in r for r in got), got
    # the same overlay on a non-clock kind is not held to (v)
    assert not _refusals(rc, _with_overlay(cands, "A6000", {}, kind="reward"), "A6000")
    # a PHASEOUT that has not settled by the stage end is not an effective end weight of 0.001
    unsettled = _w_doc(cands, {"KRABBY_PHASEOUT": PHASEOUT, "KRABBY_OMNI_CLOCK_HIDE": "1",
                               "KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48000"}, iters=1500)
    got = _refusals(rc, unsettled, "W")
    assert any("beyond the stage end" in r for r in got), got


def test_campaigns_without_clock_knobs_are_not_looked_at_by_the_clock_rules(rc):
    mq = rc.load_candidates(MQ_YAML)
    assert not [r for r in _refusals(rc, mq) if "(i)" in r or "(v)" in r or "(iv)" in r]


def test_control_pair_on_pilot_and_references_are_validated(rc, cands):
    doc = copy.deepcopy(cands)
    doc["candidates"][0]["control"] = "incumbent"
    assert any("control 'incumbent' unknown" in r for r in _refusals(rc, doc, "A6000"))
    doc = copy.deepcopy(cands)
    doc["decision_rule"]["pair_on"] = ["adopt", "tie"]
    assert any("pair_on must be a list of" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(cands)
    doc["decision_rule"]["pilot"] = "A9999"
    assert any("decision_rule.pilot names 'A9999'" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(cands)
    doc["references"]["t3_s2"] = {"chain": "nope", "seed": 2}
    assert any("references.t3_s2: chain 'nope'" in r for r in _refusals(rc, doc, "<campaign>"))
    doc = copy.deepcopy(cands)
    doc["chains"]["tail_s2_held"]["stages"][0]["resume_from"] = {"reference": "t3_s9"}
    assert any("resume_from.reference 't3_s9'" in r for r in _refusals(rc, doc, "chain:tail_s2_held"))
    doc = copy.deepcopy(cands)
    del doc["chains"]["tail_s2_held"]["stages"][0]["floor"]["min_by_category"]["move_turn"]
    assert any("reference fork's category_survival floor" in r
               for r in _refusals(rc, doc, "chain:tail_s2_held"))
    doc = copy.deepcopy(cands)
    doc["references"]["confirm_held"] = {"chain": "tail_s2_held", "seed": 2}
    doc["chains"]["tail_s2_held"]["stages"][0]["resume_from"] = {"reference": "confirm_held"}
    assert any("cannot fork its own head" in r for r in _refusals(rc, doc, "<campaign>"))


# ================================================================== control: baseline
def _baseline_records(rc, cands, led, st, *, screen=None, xids=("X01", "X02")) -> dict:
    """X0 s3 / s2: ok screens of the baseline recipe (header + entry, with seeds)."""
    b0 = rc.compose_recipe(led, cands, CODE, upto=rc.candidate_ids(cands)[0])
    for seed, xid, did in (("3", xids[0], "D02"), ("2", xids[1], "D04")):
        st["ladder"]["experiments"][xid] = {
            "xid": xid, "kind": "baseline", "candidate": "X0", "seed": seed, "recipe_id": b0["recipe_id"],
            "base_recipe_id": b0["recipe_id"], "base_rank": {"upto": None, "inclusive": False},
            "code_sha": CODE, "eval_sha": EVAL, "status": "ok", "started": f"2026-09-23T00:0{seed}:00",
            "ckpt": f"/sim/{xid}/model_1.pt", "screen": copy.deepcopy(screen) if screen else None,
            "recs": [], "eval_dirs": [], "label": f"X0 s{seed}", "chain": b0["chain"], "inflight": None,
            "stages": ["c4", "c5", "c6"]}
        led["entries"].append({"id": did, "kind": "baseline", "candidate": "X0", "seed": seed,
                               "experiment": xid, "chosen": "RECORD", "status": "final",
                               "options": ["RECORD"], "recipe_id_if_adopted": b0["recipe_id"],
                               "base_recipe_id": b0["recipe_id"], "code_sha": CODE, "eval_sha": EVAL,
                               "confidence": "clear", "history": []})
    led["entries"].append({"id": "D00bs", "kind": "resolution", "candidate": None, "chosen": "RECORD",
                           "options": ["RECORD"], "status": "final", "eval_sha": EVAL, "history": [],
                           "evidence": {"seed_terms": {"metrics": {}, "blocks": {}}}})
    return b0


def test_a_baseline_control_arm_reads_its_control_and_drift_against_the_baseline(rc, H, cands, rule,
                                                                                tmp_path):
    """A6000 adopted clear (pooled): A4500's base recipe, control screen and drift anchor are still
    X0's; A6000 is excluded from compose-on-adopt and is never an anchor."""
    led, st = H.ledger(), H.state()
    b0 = _baseline_records(rc, cands, led, st, screen=rscreen(H))
    a6 = rc._recipe_with(led, cands, CODE, "A6000", set())
    led["entries"].append(H.entry("D05", "A6000", "ADOPT", "final", experiment="X03", base=b0["recipe_id"],
                                  recipe=a6["recipe_id"], pooled_confidence="clear"))
    assert rc._anchor_qualifies(led["entries"][-1])
    base = rc.compose_recipe(led, cands, CODE, upto="A4500")
    assert base["recipe_id"] == b0["recipe_id"] and base["adopted"] == []
    assert base["anchor"]["decision"] == "D02" and "baseline" in base["anchor"]["why"]
    own = rc._recipe_with(led, cands, CODE, "A4500", set())
    assert own["adopted"] == ["A4500"] and own["anchor"]["decision"] == "D02"
    assert own["overlay_by_stage"]["c5"]["KRABBY_OMNI_CLOCK_HIDE"] == "1"
    # a later non-baseline-control candidate is not composed with the adopted arm either
    doc = copy.deepcopy(cands)
    doc["candidates"].append({"id": "B1", "kind": "reward", "question": "q", "stages": "all",
                              "overlay": {"KRABBY_YAW_W": "0.3"}})
    rec = rc.compose_recipe(led, doc, CODE, upto="B1")
    assert rec["adopted"] == [] and rec["recipe_id"] == b0["recipe_id"]
    # the dispatch: the control is X0 s3 and the drift anchor is X0
    st["ladder"]["user_choice"] = {"A4500": {"choice": "run", "at": "2099"}}
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=st, eval_sha=EVAL, resolution=rres(rc, rule))
    header = _cand_header(rc, cands, ctx, "X04", "A4500", screen=rig_better(H))
    item = rc._item("candidate", cid="A4500", seed="3", recipe=own, base_recipe=base,
                    stages=[dict(s) for s in own["stages"]], label="A4500 s3")
    rc._dispatch_candidate(ctx, item, header)
    entry = ctx.ledger["entries"][-1]
    assert entry["candidate"] == "A4500" and entry["control_experiment"] == "X01"
    assert entry["base_recipe_id"] == b0["recipe_id"]
    assert str(entry["evidence"]["anchor"]).startswith("X0/D02")
    assert entry["evidence"]["anchored_drift"]["verdict"] == "OK"
    # nothing A6000's adoption did squashes A4500's window
    assert "X04" not in rc.squash_view(ctx.ledger, cands, ctx.state, code=CODE)["experiments"]


def _cand_header(rc, cands, ctx, xid, cid, *, screen, seed="3", kind="candidate", stages=("c4", "c5", "c6"),
                 records=True, tmp_path=None, **kw) -> dict:
    recipe = rc._recipe_with(ctx.ledger, cands, CODE, cid, set())
    header = {"xid": xid, "kind": kind, "candidate": cid, "seed": seed, "label": f"{cid} s{seed}",
              "chain": recipe["chain"], "recipe_id": recipe["recipe_id"],
              "base_recipe_id": rc.compose_recipe(ctx.ledger, cands, CODE, upto=cid)["recipe_id"],
              "base_rank": {"upto": cid, "inclusive": False}, "code_sha": CODE, "eval_sha": EVAL,
              "status": "ok", "started": f"2026-09-23T01:{int(xid[1:]):02d}:00", "ckpt": f"/sim/{xid}/m.pt",
              "screen": screen, "stages": list(stages), "recs": [f"{xid}:{s}" for s in stages],
              "overlay_by_stage": {s: rc.stage_env(recipe, s) for s in stages}, "eval_dirs": [],
              "inflight": None}
    header.update(kw)
    ctx.lad["experiments"][xid] = header
    if records:
        k_records(ctx, header, Path(ctx.dir).parent / f"recs_{xid}")
    return header


def k_records(ctx, header, root: Path, *, income=0.0003, frac=1.0, clock_obs=0.0, weight=0.001,
              carried="1") -> None:
    """Stage recs of a finished run as the real driver leaves them: the dump, the train log and the
    suite stamp's raw npz -- every clock-free stage verifiable (removal verification)."""
    bucket = ctx.state.setdefault("keys", {}).setdefault(f"seed{header['seed']}", {})
    for sname in header["stages"]:
        env = header["overlay_by_stage"].get(sname) or {}
        removed = env.get("KRABBY_OMNI_CLOCK_HIDE") == "1"
        run = root / sname
        (run / "params").mkdir(parents=True, exist_ok=True)
        (run / "params" / "env.yaml").write_text(clock_dump(
            clock_w=weight if removed else 1.0, apex_w=weight if removed else 1.0, hide=removed))
        (run / "model_1.pt").write_text("x")
        log = root / f"{sname}_train.log"
        lines = [f"Episode_Reward/reward_clock_schedule: {income if removed else 0.32}",
                 f"Episode_Reward/reward_clock_swing_apex: {income if removed else 0.14}",
                 f"Metrics/clock_hidden_frac: {frac if removed else 0.0}"]
        log.write_text("\n".join(lines * 25) + "\n")
        ev = root / f"{sname}_suite"
        (ev / "raw").mkdir(parents=True, exist_ok=True)
        np.savez(ev / "raw" / "episode_000.npz", clock_obs=np.full((6, 2), clock_obs if removed else 0.5),
                 clock_phase=np.zeros(6))
        (ev / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_OMNI_CLOCK_HIDE":
                                                                   carried if removed else "0"}}))
        bucket[f"{header['xid']}:{sname}"] = {"ckpt": str(run / "model_1.pt"), "log": str(log),
                                              "evals": {"suite": {"run_dir": str(ev)}}}


# ================================================================== pair_on
def test_pair_on_queues_a_pair_for_an_adopt_a_neutral_and_a_lone_guard(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    out = _decide(rc, rule, rig_better(H), base, res)
    assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("ADOPT", "clear", "speculative", True)
    assert out["evidence"]["pair_on"] == "adopt" and "pair_on adopt" in out["why"]
    out = _decide(rc, rule, rscreen(H), base, res)
    assert out["evidence"]["seed3"]["blocks"]["RIG"]["class"] == "TIE"
    assert (out["chosen"], out["status"], out["pair"], out["evidence"]["pair_on"]) == (
        "REJECT", "speculative", True, "neutral")
    lone = rig_better(H, cat__fwd__foot_slip=0.40)            # slip 0.05 -> 0.40: that guard WORSE
    out = _decide(rc, rule, lone, base, res)
    assert out["evidence"]["seed3"]["guards"]["worse"] == ["cat.fwd.foot_slip"]
    assert (out["chosen"], out["status"], out["evidence"]["pair_on"]) == ("REJECT", "speculative", "lone_guard")
    # two guards WORSE, or a block WORSE, or RIG TIE beside a WORSE guard: the clear REJECT stands
    for cand in (rig_better(H, cat__fwd__foot_slip=0.40, cat__side__foot_slip=0.40),
                 rig_better(H, ratio=0.50),
                 rscreen(H, cat__fwd__foot_slip=0.40)):
        out = _decide(rc, rule, cand, base, res)
        assert (out["chosen"], out["confidence"], out["status"], out["pair"]) == ("REJECT", "clear", "decided", False)
        assert "pair_on" not in out["evidence"]
    # a MISSING guard is never paired (a pair cannot supply it)
    out = _decide(rc, rule, rscreen(H, cat__stand__steps_per_s=None), base, res)
    assert (out["status"], out["pair"]) == ("decided", False) and "guard MISSING" in out["why"]


def test_a_rig_block_with_a_missing_member_queues_no_neutral_or_lone_guard_pair(rc, H, rule):
    """Step-6 fix 7: a RIG block with ANY member MISSING is treated like a missing guard -- a pair
    cannot supply the member either, so neither the neutral nor the lone_guard pair is queued."""
    res = rres(rc, rule)
    base = rscreen(H)
    neutral = rscreen(H, cat__diag__support_feet=None)
    out = _decide(rc, rule, neutral, base, res)
    rig = out["evidence"]["seed3"]["blocks"]["RIG"]
    assert rig["missing"] and rig["class"] == "TIE" and rig["k"] > 0
    assert (out["chosen"], out["status"], out["pair"]) == ("REJECT", "decided", False)
    assert "pair_on" not in out["evidence"]
    lone = rig_better(H, cat__fwd__foot_slip=0.40, cat__diag__support_feet=None)
    out = _decide(rc, rule, lone, base, res)
    assert out["evidence"]["seed3"]["blocks"]["RIG"]["class"] == "BETTER"
    assert out["evidence"]["seed3"]["guards"]["worse"] == ["cat.fwd.foot_slip"]
    assert (out["chosen"], out["status"], out["pair"]) == ("REJECT", "decided", False)
    assert "pair_on" not in out["evidence"]


def test_without_pair_on_decide_is_exactly_the_old_rule(rc, H, rule):
    res = rres(rc, rule)
    plain = {k: v for k, v in rule.items() if k != "pair_on"}
    base = rscreen(H)
    out = _decide(rc, plain, rig_better(H), base, res)
    assert (out["chosen"], out["status"], out["pair"]) == ("ADOPT", "decided", False)
    assert out == rc._decide_s3(rig_better(H), base, res, cand_spec={}, rule=plain, seed_terms={},
                                anchor_screen=None)
    out = _decide(rc, plain, rscreen(H), base, res)
    assert (out["chosen"], out["confidence"], out["status"]) == ("REJECT", "clear", "decided")


# ================================================================== arm_class
def _pooled(rc, rule, res, cand_s3, cand_s2, base, prior="ADOPT"):
    return rc.decide_pooled(cand_s3, cand_s2, base, base, res, cand_spec={}, rule=rule,
                            prior={"chosen": prior, "status": "speculative"})


def _entry_from(rc, s3, pooled=None, *, status=None, removal=True, **kw) -> dict:
    ok = {"xid": "X", "applies": True, "ok": bool(removal), "checks": []}
    ev = dict(s3["evidence"], removal={"s3": ok, "s2": ok} if pooled else {"s3": ok})
    e = {"id": "D05", "kind": "candidate", "candidate": "A6000", "chosen": s3["chosen"],
         "status": status or s3["status"], "evidence": ev, "confidence": s3["confidence"]}
    if pooled is not None:
        ev.update(pooled["evidence"])
        e.update(status=status or pooled["status"], pooled_confidence=pooled["pooled_confidence"])
        if pooled["self_overrule"]:
            e["override"] = {"by": "driver", "choice": pooled["chosen"]}
    e.update(kw)
    return e


def test_arm_class_follows_the_step_12_table(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    s3 = _decide(rc, rule, rig_better(H), base, res)
    # speculative: the pooled line decides
    assert rc.arm_class(_entry_from(rc, s3), rule) == (None, "pending: its seed-2 pair is queued "
                                                            "(the pooled line decides)")
    win = _pooled(rc, rule, res, rig_better(H), rig_better(H), base)
    assert win["chosen"] == "ADOPT" and win["pooled_confidence"] == "clear"
    assert rc.arm_class(_entry_from(rc, s3, win), rule)[0] == "WIN"
    # balanced-RIG: a pooled ADOPT won on support alone is NEUTRAL
    lopsided = rscreen(H, steps=11.05, steps_rad=6.05, support=3.3)
    unbal = _pooled(rc, rule, res, lopsided, lopsided, base)
    assert unbal["chosen"] == "ADOPT"
    cls, why = rc.arm_class(_entry_from(rc, s3, unbal), rule)
    assert cls == "NEUTRAL" and "negative RIG part" in why and "steps -" in why
    # removal verification failed -> never a WIN
    assert rc.arm_class(_entry_from(rc, s3, win, removal=False), rule)[0] == "UNCLASSED"
    # BREAK / RIGID / NEUTRAL on decided seed-3 lines (no pair queued)
    brk = _decide(rc, rule, rig_better(H, ratio=0.50), base, res)
    assert rc.arm_class(_entry_from(rc, brk), rule)[0] == "BREAK"
    two = _decide(rc, rule, rig_better(H, cat__fwd__foot_slip=0.4, cat__side__foot_slip=0.4), base, res)
    assert rc.arm_class(_entry_from(rc, two), rule)[0] == "BREAK"
    rigid = _decide(rc, rule, rscreen(H, steps=11.1, steps_rad=6.1, support=2.9), base, res)
    assert rigid["evidence"]["seed3"]["blocks"]["RIG"]["class"] == "WORSE"
    assert rc.arm_class(_entry_from(rc, rigid), rule)[0] == "RIGID"
    tie_pooled = _pooled(rc, rule, res, rscreen(H), rscreen(H), base, prior="REJECT")
    assert rc.arm_class(_entry_from(rc, s3, tie_pooled), rule)[0] == "NEUTRAL"
    floor = {"id": "D06", "kind": "candidate", "chosen": "REJECT", "status": "decided",
             "floor_stage": "c6", "evidence": {"floor": "category survival FAIL"}}
    assert rc.arm_class(floor, rule) == ("BREAK", "floor at c6")
    s2_floor = _entry_from(rc, s3, win)
    s2_floor["evidence"]["s2_floor_watch"] = ["c5: category survival FAIL on ['diag']"]
    cls, why = rc.arm_class(s2_floor, rule)
    assert cls == "BREAK" and "seed-2 run" in why
    # UNCLASSED: SKIPped, user-overruled, RIG MISSING, a guard MISSING, an unpooled ADOPT confirmed
    assert rc.arm_class({"status": "skipped", "chosen": "SKIP"}, rule)[0] == "UNCLASSED"
    user = _entry_from(rc, s3, win, status="overruled", override={"by": "user", "choice": "REJECT"})
    assert rc.arm_class(user, rule) == ("UNCLASSED", "user-overruled to REJECT")
    blank = rscreen(H, steps=None, steps_rad=None, support=None)
    miss = _decide(rc, rule, blank, base, res)
    assert rc.arm_class(_entry_from(rc, miss), rule)[0] in ("UNCLASSED",)
    gmiss = _decide(rc, rule, rig_better(H, cat__stand__steps_per_s=None), base, res)
    cls, why = rc.arm_class(_entry_from(rc, gmiss), rule)
    assert cls == "UNCLASSED" and "guard MISSING" in why
    confirmed = _entry_from(rc, s3, status="confirmed")
    assert rc.arm_class(confirmed, rule)[0] == "UNCLASSED"
    # a rule without RIG: no class at all
    assert rc.arm_class(_entry_from(rc, s3, win), {"rule_version": 2}) == (None, "the rule carries no RIG block")


def test_a_driver_self_overrule_is_classed_on_its_pooled_line(rc, H, rule):
    """Left-for-implementation item 12: decide_pooled's own 'overruled' is not user-overruled."""
    res = rres(rc, rule)
    base = rscreen(H)
    s3 = _decide(rc, rule, rscreen(H), base, res)                 # neutral pair: REJECT speculative
    assert s3["evidence"]["pair_on"] == "neutral"
    pooled = _pooled(rc, rule, res, rig_better(H), rig_better(H), base, prior="REJECT")
    assert pooled["self_overrule"] and pooled["status"] == "overruled" and pooled["chosen"] == "ADOPT"
    assert rc.arm_class(_entry_from(rc, s3, pooled), rule)[0] == "WIN"


def test_arm_class_is_a_decision_field_await_if_can_read(rc, H, cands, rule, tmp_path):
    assert "arm_class" in rc.DECISION_FIELDS
    res = rres(rc, rule)
    base = rscreen(H)
    brk = _decide(rc, rule, rig_better(H, ratio=0.50), base, res)
    neutral = _pooled(rc, rule, res, rscreen(H), rscreen(H), base, prior="REJECT")
    s3n = _decide(rc, rule, rscreen(H), base, res)
    led, st = H.ledger(), H.state()
    b0 = _baseline_records(rc, cands, led, st, screen=base)
    e6 = _entry_from(rc, brk, id="D05", candidate="A6000", base_recipe_id=b0["recipe_id"])
    e45 = _entry_from(rc, brk, id="D06", candidate="A4500", base_recipe_id=b0["recipe_id"])
    led["entries"] += [dict(e6, options=["ADOPT", "REJECT", "SKIP"]), dict(e45, options=["ADOPT", "REJECT", "SKIP"])]
    ctx = rc.predicate_context(cands, led, st, b0, CODE)
    assert ctx["decision"]("A6000")["arm_class"] == "BREAK"
    assert ctx["decision"]("A3000")["arm_class"] is None
    item = rc.next_item(cands, led, st, code=CODE)
    assert item["label"] == "await_user:A3000" and "decision(A6000).arm_class=BREAK" in item["why"]
    # A4500 NEUTRAL instead: A3000 is sent automatically
    led["entries"][-1] = dict(_entry_from(rc, s3n, neutral, id="D06", candidate="A4500",
                                          base_recipe_id=b0["recipe_id"]), options=["ADOPT", "REJECT", "SKIP"])
    item = rc.next_item(cands, led, st, code=CODE)
    assert item["kind"] == "candidate" and item["candidate"] == "A3000"


# ================================================================== the flag list
def _flags(rc, rule, res, cand, base, **kw):
    cmp3 = _decide(rc, rule, cand, base, res)["evidence"]["seed3"]
    return {f["flag"]: f for f in rc.decision_flags(rule, cmp3, [cand], [base], **kw)}, cmp3


def test_the_margin_flags_read_each_margin_against_the_control(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    got, _ = _flags(rc, rule, res, rig_better(H), base)
    assert got == {}, "a clean RIG win raises no flag"
    got, cmp3 = _flags(rc, rule, res, rig_better(H, cat__turn__facing_closed=0.44), base)
    assert "margin" in got and "turn.facing_closed" in got["margin"]["text"]
    k_f = cmp3["blocks"]["F"]["k"]
    assert k_f > 1
    assert got["margin"]["mde"] == pytest.approx(k_f * 0.5 * 0.10), \
        "the F member's single-member MDE = k x R_F x R_i (step-6 fix 9)"
    assert got["margin"]["mde_label"] == f"single-member MDE (k={k_f} x R_F x R_i)"
    assert f"[single-member MDE (k={k_f} x R_F x R_i) {k_f * 0.05:.4g}]" in rc.flag_text(got["margin"])
    got, _ = _flags(rc, rule, res, rig_better(H, cat__turn__facing_closed=0.46), base)
    assert "margin" not in got, "-0.04 is inside the -0.05 margin"
    got, _ = _flags(rc, rule, res, rig_better(H, cat__move_turn__facing_closed=0.41), base)
    assert "margin" not in got, "move_turn's margin is -0.10"
    got, _ = _flags(rc, rule, res, rig_better(H, cat__move_turn__facing_closed=0.39), base)
    assert "move_turn.facing_closed" in got["margin"]["text"]
    got, _ = _flags(rc, rule, res, rig_better(H, cat__diag__survival=0.90), base)
    assert "diag.survival" in got["margin"]["text"] and got["margin"]["mde"] is not None
    got, _ = _flags(rc, rule, res, rig_better(H, cat__move_turn__ratio=0.54), base)
    assert "move_turn.ratio" in got["margin"]["text"]


def test_the_taps_apex_support_and_steps_guard_flags(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    got, _ = _flags(rc, rule, res, rig_better(H, cat__fwd__taps_per_m=1.30), base)
    assert "taps" in got and "fwd.taps_per_m" in got["taps"]["text"] and got["taps"]["mde"] is None
    got, _ = _flags(rc, rule, res, rig_better(H, cat__fwd__taps_per_m=1.20), base)
    assert "taps" not in got
    got, _ = _flags(rc, rule, res, rig_better(H, cat__side__swing_apex=0.035), base)
    assert "lift_loss" in got and "re-budget" in got["lift_loss"]["text"]
    got, _ = _flags(rc, rule, res, rig_better(H, cat__side__swing_apex=0.036), base)
    assert "lift_loss" not in got, "-28 % is inside the 30 % lift-loss line"
    got, _ = _flags(rc, rule, res, rig_better(H, cat__stand__support_feet=2.6), base)
    assert "support_down" in got and "stand" in got["support_down"]["text"]
    worse_steps = rig_better(H, cat__back__steps_per_m=12.8)
    got, _ = _flags(rc, rule, res, worse_steps, base)
    assert "steps_guard" not in got, "off unless decision_rule.steps_guard_flag"
    armed = dict(rule, steps_guard_flag=True)
    got, cmp3 = _flags(rc, armed, res, worse_steps, base)
    assert "steps_guard" in got and "back.steps_per_m" in got["steps_guard"]["text"]
    k_rig = cmp3["blocks"]["RIG"]["k"]
    assert k_rig == 10
    assert got["steps_guard"]["mde"] == pytest.approx(k_rig * 0.5 * 0.10), "k x R_RIG x R_i"
    assert got["steps_guard"]["mde_label"].startswith("single-member MDE (k=10 x R_RIG")


def test_the_speed_driven_flag_needs_all_three_conditions(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    faster = rig_better(H, ratio=0.66)                     # steps BETTER, step rate same, T up
    got, _ = _flags(rc, rule, res, faster, base)
    assert "speed_driven" in got
    slower_rate = rig_better(H, ratio=0.66, rate=0.9)       # dlog step_rate_hz < 0
    assert "speed_driven" not in _flags(rc, rule, res, slower_rate, base)[0]
    flat_t = rig_better(H)                                  # the T ratios did not rise
    assert "speed_driven" not in _flags(rc, rule, res, flat_t, base)[0]
    more_steps = rscreen(H, steps=11.1, steps_rad=6.1, support=3.5, ratio=0.66)
    assert "speed_driven" not in _flags(rc, rule, res, more_steps, base)[0]


def test_removal_and_pilot_flags_and_no_flags_without_rig(rc, H, rule):
    res = rres(rc, rule)
    base = rscreen(H)
    bad = {"xid": "X03", "applies": True, "ok": False,
           "checks": [{"stage": "c6", "check": "suite: clock_obs == 0", "ok": False, "detail": "max 0.4"}]}
    got, _ = _flags(rc, rule, res, rig_better(H), base, removal=[bad], pilot=True)
    assert set(got) == {"removal", "pilot"}
    assert "c6: suite: clock_obs == 0" in got["removal"]["text"]
    fine = {"xid": "X03", "applies": False, "ok": True, "checks": []}
    assert _flags(rc, rule, res, rig_better(H), base, removal=[fine])[0] == {}
    tie = _decide(rc, rule, rscreen(H), base, res)
    assert rc.pilot_stop(rule, "A6000", tie) and not rc.pilot_stop(rule, "A4500", tie)
    assert not rc.pilot_stop(rule, "A6000", _decide(rc, rule, rig_better(H), base, res))
    mq = rc.load_candidates(MQ_YAML)["decision_rule"]
    assert rc.decision_flags(mq, tie["evidence"]["seed3"], [rscreen(H)], [base], pilot=True) == []


def test_the_rig_split_is_additive(rc, H, rule):
    res = rres(rc, rule)
    cmp3 = _decide(rc, rule, rscreen(H, steps=11.05, steps_rad=6.05, support=3.3), rscreen(H), res)["evidence"]["seed3"]
    split = rc.rig_split(cmp3["blocks"]["RIG"])
    assert split["k"] == 10 and split["k_steps"] == 5 and split["k_support"] == 5
    assert split["steps"] == pytest.approx(-0.25) and split["support"] == pytest.approx(1.5)
    assert split["steps"] + split["support"] == pytest.approx(split["z"])
    assert "steps -0.250 (k 5) + support +1.500 (k 5)" in rc.rig_split_text(split)


# ================================================================== removal verification
def test_removal_verification_reads_the_dump_the_log_and_the_npz(rc, H, cands, rule, tmp_path):
    led, st = H.ledger(), H.state()
    _baseline_records(rc, cands, led, st, screen=rscreen(H))
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=st, eval_sha=EVAL, resolution=rres(rc, rule))
    header = _cand_header(rc, cands, ctx, "X03", "A4500", screen=rig_better(H))
    got = rc.removal_verification(ctx, header)
    assert got["applies"] and got["ok"], [c for c in got["checks"] if not c["ok"]]
    assert {c["stage"] for c in got["checks"]} == {"c5", "c6"}, "only the clock-free stages"
    assert len(got["checks"]) == 2 * (2 * 2 + 1 + 2)
    for kw, needle in ((dict(clock_obs=0.3), "clock_obs == 0"), (dict(frac=0.8), "clock_hidden_frac"),
                       (dict(income=0.05), "income at eps"), (dict(weight=1.0), "dumped weight"),
                       (dict(carried="0"), "run_meta carries")):
        k_records(ctx, header, tmp_path / f"bad_{needle[:5]}", **kw)
        got = rc.removal_verification(ctx, header)
        assert not got["ok"] and any(needle in c["check"] and not c["ok"] for c in got["checks"]), (kw, got)
    # a held run removes nothing: no check applies
    held = _cand_header(rc, cands, ctx, "X09", "A6000", screen=rscreen(H), stages=("c4", "c5"))
    assert rc.removal_verification(ctx, held) == {"xid": "X09", "applies": False, "ok": True, "checks": []}


def test_removal_verification_of_a_phaseout_stage_reads_w1_from_the_dump(rc, H, cands, tmp_path):
    doc = write_doc(rc, _w_doc(cands), tmp_path)
    ctx = H.make_ctx(rc, doc, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL)
    recipe = rc._recipe_with(ctx.ledger, doc, CODE, "W", set())
    header = {"xid": "X05", "kind": "candidate", "candidate": "W", "seed": "3", "chain": "c6x",
              "stages": ["c6x"], "overlay_by_stage": {"c6x": rc.stage_env(recipe, "c6x")}, "recs": ["X05:c6x"]}
    root = tmp_path / "w"
    k_records(ctx, header, root)
    (root / "c6x" / "params" / "env.yaml").write_text(clock_dump(phaseout=PHASEOUT, hide=True,
                                                                 ramp=(24000, 48000)))
    got = rc.removal_verification(ctx, header)
    assert got["ok"], [c for c in got["checks"] if not c["ok"]]
    assert any("dumped PHASEOUT w1 <= 0.001 with t1 <= 96000" in c["check"] for c in got["checks"])
    (root / "c6x" / "params" / "env.yaml").write_text(clock_dump(
        phaseout=PHASEOUT.replace("0.001:0:24000", "0.5:0:24000", 1), hide=True, ramp=(24000, 48000)))
    assert not rc.removal_verification(ctx, header)["ok"]


# ================================================================== the flag park (dispatch + queue)
def _pilot_campaign(rc, H, cands, rule, tmp_path, *, cand_screen):
    led, st = H.ledger(), H.state()
    base = rscreen(H)
    b0 = _baseline_records(rc, cands, led, st, screen=base)
    st["ladder"]["user_choice"] = {"A6000": {"choice": "run", "at": "2099"}}
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=st, eval_sha=EVAL, resolution=rres(rc, rule))
    header = _cand_header(rc, cands, ctx, "X03", "A6000", screen=cand_screen)
    own = rc._recipe_with(ctx.ledger, cands, CODE, "A6000", set())
    item = rc._item("candidate", cid="A6000", seed="3", recipe=own, base_recipe=b0,
                    stages=[dict(s) for s in own["stages"]], label="A6000 s3")
    rc._dispatch_candidate(ctx, item, header)
    return ctx, ctx.ledger["entries"][-1]


def _relaunch(rc, ctx, rule):
    """A new driver process on the same campaign dir (what the unit's restart does)."""
    ctx.save()
    new = rc.Ctx(ctx.dir, ctx.cands, ctx.args)
    new.code, new.eval_sha, new.tools, new.resolution = CODE, EVAL, "TOOLS", rres(rc, rule)
    new.recompute()
    rc.ingest_overrides(new)
    return new


def test_the_pilot_stop_parks_before_its_pair_and_before_the_next_arm(rc, H, cands, rule, tmp_path):
    ctx, entry = _pilot_campaign(rc, H, cands, rule, tmp_path, cand_screen=rscreen(H))
    assert entry["status"] == "speculative" and entry["evidence"]["pair_on"] == "neutral"
    assert entry["flags"]["park"] and entry["flags"]["immediate"] == ["pilot"]
    assert entry["flags"]["level"] == "s3" and entry["flags"]["pair_queued"]
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert item["kind"] == "await" and item["label"] == f"await_user:{entry['id']} flag"
    assert "pilot stop" in item["why"]
    # the DECISION line: the S3 rule, the park, the ack-flag release as the PRIMARY action, and
    # quoted confirm/overrule commands WITHOUT --cancel-pair (a relayed command run as-is must hit the
    # CLI's refusal); --cancel-pair is named in prose only
    line = [x for x in H.notify_lines(ctx) if x.split(" | ")[0].endswith(f"DECISION {entry['id']}")][-1]
    fields = line.split(" | ")
    assert "pair queued — wait for the pooled line; confirm cancels it" in line
    assert "PARKED at await_user:" in line and f"ack-flag {entry['id']}" in line
    primary = [f for f in fields if f.startswith("PRIMARY ACTION:")]
    assert len(primary) == 1 and f"ack-flag {entry['id']}" in primary[0]
    confirm = [f for f in fields if f.startswith("confirm: ")]
    overrule = [f for f in fields if f.startswith("overrule: ")]
    assert len(confirm) == len(overrule) == 1
    assert f"confirm {entry['id']}" in confirm[0] and "--cancel-pair" not in confirm[0]
    assert "--cancel-pair" not in overrule[0]
    assert fields.index(primary[0]) < fields.index(confirm[0])
    assert any("to answer now anyway add --cancel-pair; WIN becomes unreachable for this arm" in f
               for f in fields)
    assert "RIG z split: steps" in line
    assert "arm_class pending" in line
    # a relaunch re-derives the same park; the ack (ledger_cli) releases it across the relaunch
    again = _relaunch(rc, ctx, rule)
    assert rc.next_item(again.cands, again.ledger, again.state, code=CODE, view=again.view)["kind"] == "await"
    assert rc.lc.main(["--campaign-dir", str(ctx.dir), "ack-flag", entry["id"], "--why", "continue wave 1"]) == 0
    after = _relaunch(rc, again, rule)
    acked = rc.entry_by_id(after.ledger, entry["id"])
    assert acked["flag_acks"][0]["level"] == "s3" and acked["flag_acks"][0]["why"] == "continue wave 1"
    item = rc.next_item(after.cands, after.ledger, after.state, code=CODE, view=after.view)
    assert item["kind"] == "seed2" and item["candidate"] == "A6000", item["label"]


def test_a_queued_pair_defers_every_other_flag_to_the_pooled_line(rc, H, cands, rule, tmp_path):
    """A6000 s3 is an ADOPT with a margin flag: the pair is queued, the seed-3 line does not park;
    the POOLED line with the same flag parks, and an ack of the s3 line does not release it."""
    cand = rig_better(H, cat__turn__facing_closed=0.44)
    ctx, entry = _pilot_campaign(rc, H, cands, rule, tmp_path, cand_screen=cand)
    assert entry["status"] == "speculative" and entry["evidence"]["pair_on"] == "adopt"
    assert [f["flag"] for f in entry["flags"]["items"]] == ["margin"] and not entry["flags"]["park"]
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert item["kind"] == "seed2" and item["candidate"] == "A6000"
    # the pair lands with the same deficit: the pooled line parks
    s2 = _cand_header(rc, cands, ctx, "X04", "A6000", screen=copy.deepcopy(cand), seed="2", kind="seed2",
                      decision=entry["id"])
    rc._resolve_pair(ctx, item, entry, s2["screen"], s2_xid="X04", header=s2)
    entry = rc.entry_by_id(ctx.ledger, entry["id"])
    assert entry["status"] == "final" and entry["flags"]["level"] == "pooled" and entry["flags"]["park"]
    assert set(entry["evidence"]["removal"]) == {"s3", "s2"} and all(
        r["ok"] for r in entry["evidence"]["removal"].values())
    assert rc.arm_class(entry, rule)[0] == "WIN", "a flag is not a verdict"
    nxt = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert nxt["label"] == f"await_user:{entry['id']} flag"
    # an s3-level ack typed before the pooled line landed is refused, never applied to it
    rc.lc.append_override(ctx.dir, {"ack_flag": entry["id"], "level": "s3", "why": "old"})
    rc.ingest_overrides(ctx)
    assert not rc.entry_by_id(ctx.ledger, entry["id"]).get("flag_acks")
    assert rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)["kind"] == "await"
    assert rc.lc.main(["--campaign-dir", str(ctx.dir), "ack-flag", entry["id"]]) == 0
    rc.ingest_overrides(ctx)
    assert rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)["candidate"] == "A4500"


def test_a_decided_flagged_line_parks_and_a_removal_failure_parks_at_seed_3(rc, H, cands, rule, tmp_path):
    # a BREAK (T WORSE, no pair) with a margin flag: decided, and it parks
    ctx, entry = _pilot_campaign(rc, H, cands, rule, tmp_path / "a",
                                 cand_screen=rig_better(H, ratio=0.50, cat__turn__facing_closed=0.40))
    assert entry["status"] == "decided" and entry["flags"]["park"] and not entry["flags"]["pair_queued"]
    assert rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)["kind"] == "await"
    # an ADOPT (pair queued) whose run shows the clock: removal verification parks immediately
    led, st = H.ledger(), H.state()
    b0 = _baseline_records(rc, cands, led, st, screen=rscreen(H))
    st["ladder"]["user_choice"] = {"A6000": {"choice": "run", "at": "2099"}}
    ctx = H.make_ctx(rc, cands, tmp_path / "b", led=led, st=st, eval_sha=EVAL, resolution=rres(rc, rule))
    header = _cand_header(rc, cands, ctx, "X03", "A6000", screen=rig_better(H))
    k_records(ctx, header, tmp_path / "leak", clock_obs=0.7)
    own = rc._recipe_with(ctx.ledger, cands, CODE, "A6000", set())
    rc._dispatch_candidate(ctx, rc._item("candidate", cid="A6000", seed="3", recipe=own, base_recipe=b0,
                                         stages=[dict(s) for s in own["stages"]]), header)
    entry = ctx.ledger["entries"][-1]
    assert entry["status"] == "speculative" and entry["flags"]["immediate"] == ["removal"]
    assert entry["flags"]["park"]
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert item["kind"] == "await" and "removal verification FAILED" in item["why"]


def test_a_baseline_screen_that_trips_a_floor_parks(rc, H, cands, rule, tmp_path):
    led, st = H.ledger(), H.state()
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=st, eval_sha=EVAL, resolution=rres(rc, rule))
    b0 = rc.compose_recipe(ctx.ledger, cands, CODE, upto="A6000")
    header = {"xid": "X01", "kind": "baseline", "candidate": "X0", "seed": "3", "recipe_id": b0["recipe_id"],
              "base_recipe_id": b0["recipe_id"], "screen": rscreen(H), "chain": "tail",
              "floor_watch": ["c4: category survival FAIL on ['turn']"], "recs": []}
    ctx.lad["experiments"]["X01"] = dict(header, status="ok", code_sha=CODE, eval_sha=EVAL,
                                         base_rank={"upto": None, "inclusive": False})
    item = rc._item("baseline", cid="X0", seed="3", recipe=b0, base_recipe=b0, label="X0 s3")
    rc._dispatch_baseline(ctx, item, header)
    entry = next(e for e in ctx.ledger["entries"] if e.get("kind") == "baseline")
    assert entry["flags"]["park"] and entry["flags"]["level"] == "screen"
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert item["label"] == f"await_user:{entry['id']} flag", "parked before X0 s2"
    assert any("tripped a FLOOR" in x for x in H.notify_lines(ctx))
    assert rc.lc.main(["--campaign-dir", str(ctx.dir), "ack-flag", entry["id"]]) == 0
    rc.ingest_overrides(ctx)
    item = rc.next_item(ctx.cands, ctx.ledger, ctx.state, code=CODE, view=ctx.view)
    assert item["label"] == "X0 s2"


def test_ack_flag_refuses_an_entry_with_nothing_to_acknowledge(rc, cli, H, cands, tmp_path, capsys):
    led = H.ledger(H.entry("D05", "A6000", "ADOPT", "decided"))
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=H.state())
    assert cli.main(["--campaign-dir", str(ctx.dir), "ack-flag", "D05"]) == 2
    assert "nothing to acknowledge" in capsys.readouterr().err
    assert cli.main(["--campaign-dir", str(ctx.dir), "ack-flag", "D99"]) == 2


# ================================================================== the S3 rule (ledger_cli)
def test_the_s3_rule_refuses_confirm_or_overrule_of_a_speculative_line(rc, cli, H, cands, tmp_path, capsys):
    led = H.ledger(H.entry("D05", "A6000", "ADOPT", "speculative"))
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=H.state())
    d = str(ctx.dir)
    assert cli.main(["--campaign-dir", d, "confirm", "D05"]) == 2
    err = capsys.readouterr().err
    assert "SPECULATIVE" in err and "--cancel-pair" in err and "WIN" in err
    assert cli.main(["--campaign-dir", d, "overrule", "D05", "--choose", "REJECT", "--why", "x"]) == 2
    assert not cli.read_overrides(ctx.dir)
    assert cli.main(["--campaign-dir", d, "confirm", "D05", "--cancel-pair"]) == 0
    assert cli.read_overrides(ctx.dir)[-1]["cancel_pair"] is True
    assert cli.main(["--campaign-dir", d, "overrule", "D05", "--choose", "REJECT", "--why", "x",
                     "--cancel-pair"]) == 0
    # a decided line is not affected; nor is a campaign without pair_on
    led = H.ledger(H.entry("D05", "A6000", "ADOPT", "decided"))
    ctx = H.make_ctx(rc, cands, tmp_path, led=led, st=H.state(), name="decided")
    assert cli.main(["--campaign-dir", str(ctx.dir), "confirm", "D05"]) == 0
    mq = rc.load_candidates(MQ_YAML)
    led = H.ledger(H.entry("D05", "M1", "ADOPT", "speculative"))
    ctx = H.make_ctx(rc, mq, tmp_path, led=led, st=H.state(), name="mq")
    assert cli.main(["--campaign-dir", str(ctx.dir), "confirm", "D05"]) == 0
    assert "cancel_pair" not in cli.read_overrides(ctx.dir)[-1]


# ================================================================== the ramp-stage restart
def test_a_ramp_stage_re_runs_its_whole_window_on_continuation(rc, H, cands, tmp_path, monkeypatch):
    fork = tmp_path / "heads" / "model_7996.pt"
    fork.parent.mkdir(parents=True)
    fork.write_text("x")
    doc = _w_doc(cands)
    doc["chains"]["c6x"]["stages"][0]["resume_from"] = {"checkpoint": str(fork)}   # this machine's
    doc = write_doc(rc, doc, tmp_path)
    ctx = H.make_ctx(rc, doc, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    recipe = rc._recipe_with(ctx.ledger, doc, CODE, "W", set())
    stage = dict(recipe["stages"][0])
    tag = "seed3_X05_c6x_A15pB"
    header = {"xid": "X05", "kind": "candidate", "candidate": "W", "seed": "3", "label": "W s3",
              "chain": "c6x", "recipe_id": recipe["recipe_id"], "base_recipe_id": "b",
              "base_rank": {"upto": "W", "inclusive": False}, "code_sha": CODE, "status": "running",
              "overlay_by_stage": {"c6x": rc.stage_env(recipe, "c6x")}, "stages": ["c6x"],
              "stage_specs": [stage], "started": "2026-09-23T00:00:00", "recs": [], "eval_dirs": [],
              "inflight": {"stage": "c6x", "iters": 4000, "tag": tag, "resume": str(fork)}}
    ctx.lad["experiments"]["X05"] = header
    run = Path(tmp_path) / "logroot" / rc.ph.PHASES["1a"].experiment / f"2026-09-23_01-00-00_{tag}"
    run.mkdir(parents=True)
    (run / "model_11496.pt").write_text("x")                  # 3500 of the 4000-it window done
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["remaining"] == 500
    got = rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    assert got["status"] == "ok", got.get("halt")
    first = calls["train"][0]
    assert first["iters"] == 4000 and first["tag"] == f"{tag}_w" and first["resume"] == str(fork)
    rec = ctx.c.recs["X05:c6x"]
    assert rec["iters"] == 4000 and rec["resume"] == str(fork)
    assert all(ok for _n, ok in rec["overlay_checks"]), rec["overlay_checks"]
    # a HIDE-only stage (no step-keyed schedule) still continues its remainder
    assert not rc.ramp_stage(K) and rc.ramp_stage({"KRABBY_PHASEOUT": PHASEOUT})
    assert rc.ramp_stage({"KRABBY_OMNI_CLOCK_HIDE_RAMP": "24000:48000"}) and not rc.ramp_stage(None)


def test_a_finished_w_window_records_the_w_runs_log_not_the_dead_originals(rc, H, cands, tmp_path,
                                                                           monkeypatch):
    """Step-6 fix 5: the skip-train branch (the "_w" re-run had already reached its final iteration
    before a stop) records the log of the run whose directory holds the checkpoint, so removal
    verification never reads the tail of the original process that died mid-ramp."""
    fork = tmp_path / "heads" / "model_7996.pt"
    fork.parent.mkdir(parents=True)
    fork.write_text("x")
    doc = _w_doc(cands)
    doc["chains"]["c6x"]["stages"][0]["resume_from"] = {"checkpoint": str(fork)}
    doc = write_doc(rc, doc, tmp_path)
    ctx = H.make_ctx(rc, doc, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    recipe = rc._recipe_with(ctx.ledger, doc, CODE, "W", set())
    stage = dict(recipe["stages"][0])
    tag = "seed3_X05_c6x_A15pB"
    env = rc.stage_env(recipe, "c6x")
    header = {"xid": "X05", "kind": "candidate", "candidate": "W", "seed": "3", "label": "W s3",
              "chain": "c6x", "recipe_id": recipe["recipe_id"], "base_recipe_id": "b",
              "base_rank": {"upto": "W", "inclusive": False}, "code_sha": CODE, "status": "running",
              "overlay_by_stage": {"c6x": env}, "stages": ["c6x"],
              "stage_specs": [stage], "started": "2026-09-23T00:00:00", "recs": [], "eval_dirs": [],
              "inflight": {"stage": "c6x", "iters": 4000, "tag": tag, "resume": str(fork)}}
    ctx.lad["experiments"]["X05"] = header
    exp = Path(tmp_path) / "logroot" / rc.ph.PHASES["1a"].experiment
    dead = exp / f"2026-09-23_01-00-00_{tag}"                 # the original: died mid-ramp
    dead.mkdir(parents=True)
    (dead / "model_9496.pt").write_text("x")
    w_run = exp / f"2026-09-23_02-00-00_{tag}_w"              # the whole-window re-run: finished
    (w_run / "params").mkdir(parents=True)
    (w_run / "model_11995.pt").write_text("x")
    (w_run / "params" / "env.yaml").write_text(H._dump_for(rc, dict(env)) + clock_section(dict(env)))
    (w_run / "params" / "agent.yaml").write_text(f"experiment_name: {rc.ph.PHASES['1a'].experiment}\n")
    ctx.c.logs.mkdir(parents=True, exist_ok=True)
    (ctx.c.logs / f"{tag}_train.log").write_text("Traceback: infra death at it 1500\n")
    w_log = ctx.c.logs / f"{tag}_w_train.log"
    w_log.write_text(H.TRAIN_LOG)
    rc.prepare_resume(ctx, header)
    assert header["inflight"]["ckpt"] == str(w_run / "model_11995.pt")
    assert header["inflight"]["remaining"] == 1
    rc.run_experiment(ctx, rc.item_from_header(ctx, header))
    assert not calls["train"], "a finished window is not re-trained"
    assert ctx.c.recs["X05:c6x"]["log"] == str(w_log)


# ================================================================== the reference-chain launcher
def test_the_reference_launcher_runs_t3_s2_with_no_decision_and_no_early_abort(rc, H, cands, tmp_path,
                                                                             monkeypatch):
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "t3_s2"]) == 0
    assert len(calls["train"]) == 1
    t3 = calls["train"][0]
    assert t3["iters"] == 3000 and t3["resume"] is None and t3["tag"].startswith("seed2_")
    assert t3["early_abort"] is False and t3["abort_floor"] is None, "Z/t3_s2 are never cut at 1000 its"
    assert t3["soundness"] is False, "a reference is never judged by dead_plant either (Z s2, 2026-09-24)"
    assert [c["row"] for c in calls["eval"]] == ["suite"]
    st = json.loads((cdir / "state.json").read_text())
    # step-6 fix 10: the reference park goes to the notify line; the main ladder's phase is untouched
    assert st.get("phase") in (None, "run") and st.get("phase") != "await_user:reference_done:t3_s2"
    assert any(" AWAIT " in f" {line} " and "await_user:reference_done:t3_s2" in line
               for line in (cdir / "notify.log").read_text().splitlines())
    ref = st["ladder"]["references"]["t3_s2"][-1]
    assert ref["seed"] == "2" and ref["ckpt"].endswith("model_2999.pt") and ref["code_sha"] == CODE
    heads = json.loads((cdir / "heads.json").read_text())
    assert heads["seed2"][f"{ref['xid']}:t3"] == ref["ckpt"]
    assert not (cdir / "ledger.json").exists() or not [
        e for e in json.loads((cdir / "ledger.json").read_text())["entries"] if e.get("kind") == "candidate"]
    assert any(line.split(" ", 2)[-1].startswith("REFERENCE t3_s2 s2")
               for line in (cdir / "notify.log").read_text().splitlines())
    # the main queue never resumes a reference window
    header = st["ladder"]["experiments"][ref["xid"]]
    assert header["kind"] == "reference" and header["status"] == "ok"
    # a second launch has nothing to run
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "t3_s2"]) == 0
    assert len(calls["train"]) == 1
    # the confirm tail forks t3_s2's recorded head (resolved at launch)
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "confirm_held"]) == 0
    tail = calls["train"][-1]
    assert tail["resume"] == ref["ckpt"] and tail["iters"] == 1500 and tail["early_abort"] is False
    assert rc.reference_recipe(cands, "confirm_held")["resume_references"] == {"c4": "t3_s2"}


def test_a_reference_launch_leaves_the_main_ladders_recorded_phase_alone(rc, H, cands, tmp_path,
                                                                         monkeypatch):
    """Step-6 fix 10: a --reference-chain launch parks on its notify line and restores the main
    ladder's recorded state.phase (e.g. a flag park ``ledger_cli pending`` shows)."""
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "t3_s2"]) == 0
    st = json.loads((cdir / "state.json").read_text())
    st["phase"] = "await_user:D05 flag"
    (cdir / "state.json").write_text(json.dumps(st))
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "confirm_held"]) == 0
    st = json.loads((cdir / "state.json").read_text())
    assert st["ladder"]["references"]["confirm_held"], "the reference ran and was recorded"
    assert st["phase"] == "await_user:D05 flag"
    assert "await_user:reference_done:confirm_held" in (cdir / "notify.log").read_text()
    # and the "already finished" park restores it too
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "confirm_held"]) == 0
    assert json.loads((cdir / "state.json").read_text())["phase"] == "await_user:D05 flag"


def test_z_carries_the_hidden_clock_into_every_stage_eval(rc, H, cands, tmp_path, monkeypatch):
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    assert rc.main(["--campaign-dir", str(cdir), "--reference-chain", "scratch_free"]) == 0
    assert [c["iters"] for c in calls["train"]] == [3000, 1500, 1500, 2000]
    assert not any(c["early_abort"] for c in calls["train"])
    assert all(c["env"]["KRABBY_OMNI_CLOCK_HIDE"] == "1" for c in calls["train"])
    assert [c["row"] for c in calls["eval"]] == ["suite"] * 4
    assert all(c["carry"].get("KRABBY_OMNI_CLOCK_HIDE") == "1" for c in calls["eval"])


def test_a_reference_fork_with_no_recorded_head_halts_before_training(rc, H, cands, tmp_path, monkeypatch):
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    with pytest.raises(SystemExit):
        rc.main(["--campaign-dir", str(cdir), "--reference-chain", "confirm_held"])
    assert not calls["train"]
    st = json.loads((cdir / "state.json").read_text())
    assert "resume_head_missing" in st["phase"]


def test_the_reference_launcher_refuses_a_bad_name_or_seed(rc, H, cands, tmp_path, monkeypatch):
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    with pytest.raises(SystemExit, match="not a references: key"):
        rc.main(["--campaign-dir", str(cdir), "--reference-chain", "nope"])
    with pytest.raises(SystemExit, match="disagrees"):
        rc.main(["--campaign-dir", str(cdir), "--reference-chain", "t3_s2", "--seed", "3"])
    with pytest.raises(SystemExit, match="only goes with --reference-chain"):
        rc.main(["--campaign-dir", str(cdir), "--seed", "3"])
    doc = copy.deepcopy(cands)
    doc["references"]["t3_free"] = {"chain": "t3_clock"}
    with pytest.raises(SystemExit, match="pass --seed"):
        rc.reference_seed(doc, "t3_free")
    assert rc.reference_seed(doc, "t3_free", "5") == "5"


def test_a_reference_ramp_window_restarts_whole_and_the_queue_skips_reference_headers(rc, H, cands,
                                                                                     tmp_path, monkeypatch):
    """Left-for-implementation item 5: the reference launcher applies the whole-window restart too."""
    doc = copy.deepcopy(cands)
    t3 = copy.deepcopy(doc["chains"]["t3_clock"]["stages"][0])
    t3["overlay"].update({"KRABBY_PHASEOUT": PHASEOUT.replace("24000", "240"), "KRABBY_OMNI_CLOCK_HIDE": "1",
                          "KRABBY_OMNI_CLOCK_HIDE_RAMP": "240:480"})
    t3["iters"] = 20
    doc["chains"]["ramp_ref"] = {"stages": [t3]}
    doc["references"]["ramp_ref"] = {"chain": "ramp_ref", "seed": 3}
    doc = write_doc(rc, doc, tmp_path)
    assert not _refusals(rc, doc), _refusals(rc, doc)
    ctx = H.make_ctx(rc, doc, tmp_path, led=H.ledger(), st=H.state(), eval_sha=EVAL)
    calls = install_clock_fakes(rc, H, monkeypatch, tmp_path, eval_summary=rscreen(H))
    recipe = rc.reference_recipe(doc, "ramp_ref", CODE)
    tag = "seed3_X01_t3_A15pB"
    header = {"xid": "X01", "kind": "reference", "candidate": "ramp_ref", "seed": "3", "label": "ref:ramp_ref s3",
              "chain": "ramp_ref", "recipe_id": recipe["recipe_id"], "base_recipe_id": recipe["recipe_id"],
              "base_rank": {"upto": None, "inclusive": False}, "code_sha": CODE, "status": "running",
              "overlay_by_stage": {"t3": rc.stage_env(recipe, "t3")}, "stages": ["t3"],
              "stage_specs": [dict(recipe["stages"][0])], "started": "2026-09-23T00:00:00", "recs": [],
              "eval_dirs": [], "inflight": {"stage": "t3", "iters": 20, "tag": tag, "resume": None}}
    ctx.lad["experiments"]["X01"] = header
    ctx.save()
    monkeypatch.setattr(rc, "migration_item", lambda ctx_: None)       # no D00s migration pending
    assert rc.find_inflight(ctx) is None, "the main loop never resumes a reference window"
    run = Path(tmp_path) / "logroot" / rc.ph.PHASES["1a"].experiment / f"2026-09-23_01-00-00_{tag}"
    run.mkdir(parents=True)
    (run / "model_12.pt").write_text("x")
    assert rc.reference_loop(ctx, "ramp_ref", "3") == 0
    first = calls["train"][0]
    assert first["iters"] == 20 and first["resume"] is None and first["tag"] == f"{tag}_w"
    assert ctx.lad["references"]["ramp_ref"][-1]["xid"] == "X01"


def test_the_reference_identity_hashes_its_name_and_fork(rc, cands):
    a = rc.reference_recipe(cands, "confirm_held", CODE)
    assert a["recipe_id"] == rc.sha256_text(rc.canonical(
        {"reference": "confirm_held", "chain": "tail_s2_held", "overlay_by_stage": a["overlay_by_stage"],
         "resume_references": {"c4": "t3_s2"}}))[:12]
    doc = copy.deepcopy(cands)
    doc["references"]["other"] = {"chain": "tail_s2_held", "seed": 2}
    assert rc.reference_recipe(doc, "other", CODE)["recipe_id"] != a["recipe_id"]
    assert "resume_references" not in rc.compose_recipe({"schema": 1, "entries": []}, cands, CODE, upto="A6000")


# ================================================================== the dry run
def test_a_clean_dry_run(rc, H, cands, tmp_path, monkeypatch, capsys):
    cdir = H._main_env(rc, cands, tmp_path, monkeypatch, code=CODE)
    capsys.readouterr()
    assert rc.main(["--campaign-dir", str(cdir), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "REFUSED" not in out and "DISPATCH REFUSED" not in out
    assert "## references" in out and "scratch_free: chain scratch_free seed 3" in out
    assert "c4    1500 its from t3_s2" in out
    assert "RIG rule (plan step 4): adopt_requires RIG, pair_on ['adopt', 'neutral', 'lone_guard'], pilot A6000" in out
    assert "control: baseline ['A6000', 'A4500', 'A3000']" in out
    queue = out[out.index("## the derived queue (every decision resolved as a clear REJECT"):]
    assert "X0 s3" in queue and "await_user:A6000" in queue
    assert not (cdir / "state.json").exists() and not (cdir / "ledger.json").exists()
