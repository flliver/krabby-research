"""``experiments/tools/resolution.py`` -- the D00 eval-resolution record (plan decision 6).

Pure: the tool is loaded by path, every synthetic eval run dir is built in ``tmp_path`` by a
mini-harness that writes its aggregate through ``gait_eval.report.aggregate`` and its per-hold facing
through ``gait_eval.metrics.facing_metrics``, so a fixture is self-consistent by construction and the
tool's own replication checks run against it for real. No Isaac, no GPU, no network.

The one test that wants the recorded 1c-004 row (decision 6's null construction names it) skips when
that run dir is absent -- eval output is untracked, so a fresh checkout has none. Its property is
pinned unconditionally on synthetic columns by the test above it.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools")
RES_PY = TOOLS / "resolution.py"
GAIT_SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
RECORDED_1C = (REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-16_1805_omni_formation"
               / "seed3_1c_004_A15pB/slow__omni/seed001/2026-09-18_08-27-40")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


res = _load(RES_PY, "crab_hex_omni_resolution_under_test")


@pytest.fixture(scope="module")
def ge():
    """``gait_eval``'s ``report`` and ``metrics`` -- the harness the fixtures imitate."""
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import metrics as M
    from gait_eval import report as RPT
    return RPT, M


# --------------------------------------------------------------------------------- the mini harness
FIVE_CARRY = {"KRABBY_OMNI_FACING_OBS_CLIP": "0.6", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.25:0.25",
              "KRABBY_OMNI_ANG_VEL_MAX": "0.3", "KRABBY_OMNI_CLOCK_TURN_R": "1.2",
              "KRABBY_OMNI_YAW_RANGE": "-3.14159:3.14159"}
DT = 0.02


def _schedule(sid):
    sc = res._manifest_scenario(sid)
    return sc["hold_labels"], sc["hold_s"], sc["prologue_s"], sc["fall_exclusion_s"]


def _raw_episode(sid, rng, *, closes: float, fall: bool):
    """Per-step arrays for one episode, on the manifest's real hold boundaries.

    ``closes`` scales how much of each hold's initial heading error is closed by its end, so a
    fixture can make one head better at facing than another; ``fall`` truncates the episode inside
    the last turn hold, which is what makes the survival guards and the fall trim non-trivial.
    """
    labels, hold_s, prologue_s, _ = _schedule(sid)
    n_pro = int(round(prologue_s / DT))
    seg = [np.full(n_pro, -1, dtype=np.int16)]
    steady = [np.zeros(n_pro, dtype=bool)]
    err, wz, cmd = [np.zeros(n_pro)], [np.zeros(n_pro)], [np.zeros((n_pro, 3))]
    for hi, label in enumerate(labels):
        k = int(round(hold_s[label] / DT))
        seg.append(np.full(k, hi, dtype=np.int16))
        st = np.ones(k, dtype=bool)
        st[:int(round(1.0 / DT))] = False              # settle_s 1.0, as the manifest defaults say
        steady.append(st)
        turning = label.startswith("turn") or "turn" in label
        e0 = (1.4 + 0.2 * rng.standard_normal()) if turning else (0.05 + 0.02 * rng.random())
        t = np.arange(k) / max(1, k - 1)
        e = e0 * (1.0 - closes * t) + 0.01 * rng.standard_normal(k)
        err.append(e)
        wz.append(np.gradient(-e, DT) * 0.5 + 0.01 * rng.standard_normal(k))
        c = np.zeros((k, 3))
        c[:, 0] = 0.25 if label in ("fwd", "diag") else (0.22 if label not in ("stand",) and
                                                        not label.startswith("turn") else 0.0)
        c[:, 2] = 0.3 * np.sign(e0) if turning else 0.0
        cmd.append(c)
    out = {"segment_id": np.concatenate(seg), "steady_mask": np.concatenate(steady),
           "heading_err": np.concatenate(err).astype(np.float32),
           "root_ang_vel_b": np.stack([np.zeros(sum(len(x) for x in wz)),
                                       np.zeros(sum(len(x) for x in wz)),
                                       np.concatenate(wz)], axis=-1).astype(np.float32),
           "cmd_applied": np.concatenate(cmd).astype(np.float32)}
    total = out["segment_id"].size
    n_steps = total
    reason = "schedule_complete"
    if fall:
        n_steps = int(total * 0.8)
        reason = "fall"
    for k, v in out.items():
        out[k] = v[:n_steps]
    return out, n_steps, reason


def make_run(tmp: Path, sid, *, head, rng, n_eps=12, closes=0.6, ratio=0.6, tripod=0.5,
             fall_every=4, env_vars=None, with_raws=True, ge_mods=None):
    """One synthetic eval run dir: run_meta, metrics/episode_XX.json, scenario_metrics, raw/*.npz.

    The per-hold ``facing`` block is computed from the raws through the harness's own
    ``facing_metrics`` over the harness's own whole-hold window, so ``resolution.py``'s window
    replication check passes iff the tool cuts the same window the harness does -- which is the point
    of building the fixture this way rather than hand-writing numbers.
    """
    RPT, M = ge_mods
    labels, hold_s, prologue_s, fall_excl = _schedule(sid)
    fall_trim = int(round(fall_excl / DT))
    run = tmp
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    if with_raws:
        (run / "raw").mkdir(parents=True, exist_ok=True)
    episodes, n_steps_env = [], []
    for i in range(n_eps):
        raw, n_steps, reason = _raw_episode(sid, rng, closes=closes,
                                            fall=(fall_every and i % fall_every == 0))
        n_steps_env.append(n_steps)
        holds = {}
        for hi, label in enumerate(labels):
            widx = res.whole_hold_index(raw["segment_id"], hi, n_steps,
                                        ended_in_fall=reason in ("fall", "hard_fall"),
                                        fall_trim_steps=fall_trim)
            fac = M.facing_metrics(raw["heading_err"][widx], raw["root_ang_vel_b"][widx, 2], DT,
                                   wz_cmd_t=raw["cmd_applied"][widx, 2],
                                   min_initial=res.FACING_CLOSED_MIN_INITIAL_RAD)
            fac.pop("facing_closed")                 # a pre-2026-09-18 aggregate has neither key
            fac.pop("heading_error_initial_abs")
            fac.pop("n_uninformative")
            cmd0 = float(abs(raw["cmd_applied"][widx, 0]).mean()) if widx.size else 0.0
            speed = 0.25 if label in ("fwd", "diag") else (0.22 if cmd0 > 0.05 else 0.0)
            entry = {"label": label, "n_steps": int(widx.size), "facing": fac,
                     "tripod": {"tripod_score": float(tripod + 0.05 * rng.standard_normal()),
                                "low_confidence": False, "valid": True, "discard_reason": None}}
            if speed > 0.2 and widx.size:
                got = float(speed * (ratio + 0.05 * rng.standard_normal()))
                entry["tracking"] = {"vx": {"cmd_mean": speed, "actual_mean": got,
                                            "ratio": got / speed, "mean_abs": abs(got),
                                            "rms": abs(got), "signed_mean": got},
                                     "vy": {"actual_mean": 0.0, "cmd_mean": 0.0},
                                     "wz": {"actual_mean": 0.0, "cmd_mean": 0.0}}
            holds[label] = entry
        ep = {"env_index": i, "n_steps": int(n_steps), "duration_s": n_steps * DT,
              "termination_reason": reason, "completed_schedule": reason == "schedule_complete",
              "tripod_score": float(tripod), "tippy_tap_fraction": 0.1, "slip_ratio_mean": 0.1,
              "facing_error_rms": 0.5, "holds": holds, "discarded_windows": []}
        ep["tracking_ratio"] = float(np.mean([h["tracking"]["vx"]["ratio"] for h in holds.values()
                                              if h.get("tracking")]))
        episodes.append(ep)
        (run / "metrics" / f"episode_{i:02d}.json").write_text(json.dumps(ep, sort_keys=True))
        if with_raws:
            payload = dict(raw)
            payload["dt"] = np.asarray(DT, dtype=np.float64)
            np.savez_compressed(run / "raw" / f"episode_{i:02d}.npz", **payload)
    agg = RPT.aggregate(episodes)
    meta = {"scenario_id": sid, "hold_labels": list(labels), "dt": DT, "num_envs": n_eps,
            "n_steps_per_env": n_steps_env, "plant": "A15+B",
            "checkpoint": f"/logs/{head}/model_999.pt",
            "env_vars": dict(FIVE_CARRY if env_vars is None else env_vars)}
    (run / "run_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (run / "scenario_metrics.json").write_text(json.dumps(
        {"scenario_id": sid, "aggregate": agg, "run_meta": meta,
         "episodes": [{k: e[k] for k in ("env_index", "tripod_score", "tracking_ratio",
                                         "tippy_tap_fraction", "slip_ratio_mean",
                                         "termination_reason", "n_steps", "facing_error_rms")}
                      for e in episodes]}, sort_keys=True))
    return run


RULE = {
    "blocks": {
        "F": {"level": 0.95, "metrics": ["facing_closed_by_hold.turn_l90",
                                         "facing_closed_by_hold.turn_r90",
                                         "facing_closed_by_hold.move_turn",
                                         "yaw_progress_by_hold.turn_l90",
                                         "yaw_progress_by_hold.turn_r90",
                                         "yaw_progress_by_hold.move_turn"]},
        "T": {"level": 0.95, "metrics": ["ratio_by_hold.fwd", "ratio_by_hold.lat_l",
                                         "ratio_by_hold.diag", "ratio_by_hold.back"]},
        "F_abs": {"level": 0.95, "metrics": ["facing_closed_by_hold.turn_l90",
                                            "net_yaw_by_hold.turn_l90"]},
        "T_abs": {"level": 0.95, "metrics": ["achieved_speed_by_hold.fwd",
                                             "achieved_speed_by_hold.back"]},
    },
    "guards": {"level": 0.99, "metrics": ["completion", "tripod_walking_dir",
                                          "turn_survival_l90", "turn_survival_r90"]},
    "watch": ["ratio_by_hold.move_turn", "ratio_by_hold.lat_r", "track_ratio_tail"],
    "charter_floors": {"completion": 0.10, "tripod_walking_dir": 0.056},
    "bootstrap": {"B": 200, "paired_on": "env_index", "rng_seed": 7},
}


def _cands(*rows, rule=None):
    return {"resolution_inputs": list(rows), "decision_rule": dict(RULE if rule is None else rule)}


def _entry(run_dir, **kw):
    return {"run_dir": str(run_dir), **kw}


@pytest.fixture
def three_heads(tmp_path, ge):
    """1c-004 / 2b / 2a-reeval on ``slow__omni``, the D00 shape, with 2b worse at facing."""
    rng = np.random.default_rng(1)
    a = make_run(tmp_path / "a", "slow__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="2b", rng=rng, closes=0.4, ratio=0.5,
                 tripod=0.4, ge_mods=ge)
    c = make_run(tmp_path / "c", "slow__omni", head="2a-reeval", rng=rng, closes=0.6, ge_mods=ge)
    return a, b, c


# ------------------------------------------------------------------------------------- pure maths
def test_half_width_is_the_centred_spread_and_ignores_the_offset():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    base = res.half_width(x, 0.975)
    assert base == pytest.approx(1.96, abs=0.06)
    assert res.half_width(x + 17.0, 0.975) == pytest.approx(base, abs=1e-12)
    assert res.half_width(x, 0.99) == pytest.approx(2.326, abs=0.09)
    assert res.half_width([1.0], 0.975) is None                     # below MIN_BOOTSTRAP_N
    assert res.half_width([np.nan, np.nan, 1.0, 2.0], 0.975) is not None


def test_quantile_p_is_two_sided_95_and_one_sided_99():
    assert res.quantile_p(0.95, one_sided=False) == pytest.approx(0.975)
    assert res.quantile_p(0.99, one_sided=True) == pytest.approx(0.99)


def test_classify_is_decision_5s_rule_including_the_boundary():
    assert res.classify(1.8, 0.9) == "BETTER"
    assert res.classify(-1.8, 0.9) == "WORSE"
    assert res.classify(0.2, 0.9) == "TIE"
    assert res.classify(0.9, 0.9) == "TIE"          # boundary is a TIE on both sides
    assert res.classify(-0.9, 0.9) == "TIE"
    assert res.classify(1.8, 0.9, one_sided=True) == "TIE"   # a guard never says BETTER
    assert res.classify(-1.8, 0.9, one_sided=True) == "WORSE"
    assert res.classify(None, 0.9) == "MISSING"
    assert res.classify(1.0, None) == "MISSING"
    assert res.classify(float("nan"), 0.9) == "MISSING"


def test_single_and_pooled_R_are_decision_6s_arithmetic():
    assert res.single_block_R(0.5, 0.7) == pytest.approx(0.7)
    assert res.single_block_R(0.5, None) == pytest.approx(0.5)
    assert res.single_block_R(None, None) is None
    assert res.single_guard_R(0.08, 0.03) == pytest.approx(0.08)
    assert res.pooled_block_R(0.6, 0.2) == pytest.approx(0.6 / math.sqrt(2))
    assert res.pooled_block_R(0.6, 0.5) == pytest.approx(0.5)
    assert res.pooled_guard_R(0.08, 0.03) == pytest.approx(math.sqrt((0.08 / math.sqrt(2)) ** 2
                                                                     + 0.03 ** 2))
    assert res.pooled_guard_R(None, None) is None


def test_block_seed_term_cannot_cancel_across_alternating_signs():
    r = {f"m{i}": 1.0 for i in range(6)}
    alternating = {f"m{i}": (1.0 if i % 2 else -1.0) for i in range(6)}
    term = res.block_seed_term(alternating, r)
    assert term == pytest.approx(1.0 / math.sqrt(6), abs=1e-12)   # rms 1 / sqrt(k)
    assert term > 0.0
    # and it is never below the signed pair statistic
    aligned = {f"m{i}": 1.0 for i in range(6)}
    z_pair = abs(sum(aligned.values()) / 6)
    assert res.block_seed_term(aligned, r) >= z_pair - 1e-12
    assert res.block_seed_term({"a": None}, {"a": 1.0}) is None


def test_block_mean_drops_a_missing_member_not_the_whole_draw():
    std = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])
    got = res.block_mean(std)
    assert got[0] == pytest.approx(2.0)     # both members
    assert got[1] == pytest.approx(2.0)     # one member left the mean
    assert math.isnan(got[2])               # every member missing -> the draw drops out


def test_required_carry_keys_are_the_five_without_a_default():
    req = res.required_carry_keys()
    assert set(req) == set(res.rp.EVAL_CARRY) - set(res.rp.EVAL_CARRY_DEFAULTS)
    assert len(req) == 5
    assert "KRABBY_OMNI_FACING_OBS_ENCODING" not in req    # defaulted -> not required
    assert "KRABBY_OMNI_ANG_VEL_CLIP" not in req


# ----------------------------------------------------------------------------- the provenance rule
def test_a_defaulted_carry_key_is_taken_at_its_default_and_recorded(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(2), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.provenance_ok and row.accepted and row.reason == "ok"
    assert set(row.env_vars_defaulted) == set(res.rp.EVAL_CARRY_DEFAULTS)


def test_a_missing_non_defaulted_carry_key_is_refused_and_a_waiver_makes_it_informational(tmp_path, ge):
    thin = {k: v for k, v in FIVE_CARRY.items() if k != "KRABBY_OMNI_ANG_VEL_MAX"}
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(3),
                   env_vars=thin, ge_mods=ge)
    refused = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert refused.provenance_ok is False and refused.accepted is False
    assert "refused" in refused.reason and "KRABBY_OMNI_ANG_VEL_MAX" in refused.reason
    waived = res.load_row(_entry(run, provenance_ok=True), ring_proxy=False, rule=RULE)
    assert waived.provenance_ok is False, "the measured verdict does not change with a waiver"
    assert waived.accepted is False, "a waived row is informational, never a candidate-R source"
    assert "informational" in waived.reason and "waived" in waived.reason


def test_role_informational_is_reported_but_never_pooled(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(4), ge_mods=ge)
    row = res.load_row(_entry(run, role="informational"), ring_proxy=False, rule=RULE)
    assert row.provenance_ok is True and row.accepted is False
    assert "informational by role" in row.reason


# ----------------------------------------------------------------------------------- cross-head pairing
def test_cross_head_pairs_are_the_control_against_each_other_head(three_heads, tmp_path):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    names = {p["name"] for p in out["pairs"]}
    assert names == {"1c-004 vs 2b @slow__omni", "1c-004 vs 2a-reeval @slow__omni"}, names
    assert all(p["kind"] == "cross_head" for p in out["pairs"])
    assert out["by_row"]["slow__omni"]["control_head"] == "1c-004"
    # every scored metric names the pairs its R came from, and R is the MAX over them
    for mid, m in out["metrics"].items():
        assert m["pairs_used"], mid
        assert m["R_eval"] == pytest.approx(max(v for v in m["per_pair"].values() if v is not None))
    # the control is the FIRST accepted entry listed; reordering moves it
    flipped = res.resolve(_cands(_entry(b, head="2b"), _entry(a, head="1c-004"),
                                 _entry(c, head="2a-reeval")))
    assert flipped["by_row"]["slow__omni"]["control_head"] == "2b"
    forced = res.resolve(_cands(_entry(b, head="2b"), _entry(a, head="1c-004")),
                         control_head="1c-004")
    assert forced["by_row"]["slow__omni"]["control_head"] == "1c-004"


def test_an_informational_row_forms_no_pair(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="corrected", role="informational")))
    assert {p["name"] for p in out["pairs"]} == {"1c-004 vs 2b @slow__omni"}
    assert [i["head"] for i in out["inputs"] if not i["accepted"]] == ["corrected"]


def test_two_rows_of_one_head_are_the_process_floor_not_a_candidate_pair(tmp_path, ge):
    rng = np.random.default_rng(5)
    a = make_run(tmp_path / "a", "slow__omni", head="2b", rng=rng, ge_mods=ge)
    a2 = make_run(tmp_path / "a2", "slow__omni", head="2b", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    out = res.resolve(_cands(_entry(b, head="1c-004"), _entry(a, head="2b"),
                             _entry(a2, head="2b", process_floor_pair=True)))
    kinds = {p["name"]: p["kind"] for p in out["pairs"]}
    assert kinds["2b vs 2b @slow__omni"] == "process_floor"
    assert kinds["1c-004 vs 2b @slow__omni"] == "cross_head"
    assert out["process_floor"], "the same-head pair must be reported as the process-noise floor"
    assert "completion" in out["process_floor"]
    # and it never contributes to the candidate R
    for m in out["metrics"].values():
        assert "2b vs 2b @slow__omni" not in m["per_pair"]


def test_a_flagged_floor_row_with_its_own_label_is_a_floor_pair_not_a_third_cross_arm(tmp_path, ge,
                                                                                       three_heads):
    """``process_floor_pair`` is a FLAG, not a coincidence of labels.

    The campaign lists the 2b re-eval as ``head: 2b_floor`` (so the record reads in campaign names),
    and decision 6 says the same-head pair is "reported separately as the process-noise floor, NEVER
    as the candidate R". Grouping by head label alone made that row a third cross-head arm whose
    half-widths were folded into the candidate R by the max -- silently, since `process_floor` then
    stayed {} and the note still asked for the row that had just been added.
    """
    a, b, c = three_heads
    floor_run = make_run(tmp_path / "bf", "slow__omni", head="2b", rng=np.random.default_rng(31),
                         closes=0.4, ratio=0.5, tripod=0.4, ge_mods=ge)
    base = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                              _entry(c, head="2a-reeval")))
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval"),
                             _entry(floor_run, head="2b_floor", floor_of="2b",
                                    role="process_floor", process_floor_pair=True)))
    kinds = {p["name"]: p["kind"] for p in out["pairs"]}
    assert kinds["2b vs 2b_floor @slow__omni"] == "process_floor"
    assert set(kinds) - {"2b vs 2b_floor @slow__omni"} == {"1c-004 vs 2b @slow__omni",
                                                           "1c-004 vs 2a-reeval @slow__omni"}
    assert out["process_floor"], "the flagged row is the process floor"
    for mid, m in base["metrics"].items():                       # and the candidate R never moved
        assert out["metrics"][mid]["R_eval"] == m["R_eval"], mid
        assert all("2b_floor" not in name for name in m["per_pair"]), mid
    for block, bl in base["blocks"].items():
        assert out["blocks"][block]["R_eval"] == bl["R_eval"], block


def test_a_flagged_floor_row_that_matches_nothing_is_refused(tmp_path, ge, three_heads):
    """Better a refusal than a silent extra arm: a flag that found no target is a spec mistake."""
    a, b, _ = three_heads
    orphan = make_run(tmp_path / "orphan", "slow__omni", head="nobody",
                      rng=np.random.default_rng(32), ge_mods=ge)
    with pytest.raises(res.ResolutionError) as exc:
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                           _entry(orphan, head="ghost", floor_of="not-a-head",
                                  process_floor_pair=True)))
    assert "process_floor_pair" in str(exc.value) and "floor_of" in str(exc.value)


def test_the_block_half_width_is_standardised_by_the_MERGED_metric_R(three_heads):
    """Decision 5: "one standardiser everywhere: R_i = R_eval,i" -- and R_eval,i is the PUBLISHED
    half-width (the max over the pairs), which is what ``decide()`` divides by.

    So ``R_F^eval``, "the paired-bootstrap 95 % half-width of z_F itself", has to be bootstrapped in
    those same units. Standardising each pair by its own per-metric spread measures a slightly
    different statistic (5-7 % wide on the recorded inputs) and is only conservative by accident.
    """
    a, b, c = three_heads
    rows = [res.load_row(_entry(run, head=head), ring_proxy=False, rule=RULE)
            for run, head in ((a, "1c-004"), (b, "2b"), (c, "2a-reeval"))]
    specs, _ = res.specs_from_rule(RULE, rows[0].hold_labels)
    pairs = [res.bootstrap_pair(rows[0], other, B=200, rng_seed=7, specs=specs)
             for other in rows[1:]]
    metrics, blocks, _ = res._merge_pairs(pairs)
    for block, slot in blocks.items():
        recomputed = {}
        for pr in pairs:
            use = slot["members_used"][pr.name]
            z = res.block_mean(np.stack([pr.deltas[m] / float(metrics[m]["R_eval"]) for m in use],
                                        axis=0))
            want = res.half_width(z, res.quantile_p(0.95, one_sided=False))
            assert slot["per_pair"][pr.name] == pytest.approx(want, abs=1e-12), (block, pr.name)
            recomputed[pr.name] = want
        assert slot["R_eval"] == pytest.approx(max(recomputed.values()), abs=1e-12)
    # and it really differs from the one-pass, pair-local version on at least one pair
    local = [pr.blocks[b_]["R_eval"] for pr in pairs for b_ in pr.blocks]
    published = [blocks[b_]["per_pair"][pr.name] for pr in pairs for b_ in pr.blocks]
    assert any(abs(x - y) > 1e-12 for x, y in zip(local, published)), \
        "the merged R must differ from at least one pair's own, or this pins nothing"


def test_process_floor_is_empty_without_a_same_head_pair(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    assert out["process_floor"] == {}
    assert any("process_floor empty" in n for n in out["notes"])


# ------------------------------------------------------------------------------- facing_closed sourcing
def test_the_rescore_reproduces_every_recorded_facing_key(tmp_path, ge):
    """The window replication is exact, so ``facing_closed`` off the raws is trustworthy."""
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(6), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.mask_deviation == 0.0
    assert row.facing_closed_source == "metric"
    closed, source = res.facing_closed_by_hold_from_raws(run)
    assert source == "metric"
    assert closed["turn_l90"] is not None and closed["stand"] is None


def test_a_wrong_window_is_refused_not_absorbed(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(7), ge_mods=ge)
    ep_path = run / "metrics" / "episode_00.json"
    ep = json.loads(ep_path.read_text())
    ep["holds"]["turn_l90"]["facing"]["heading_error_final_abs"] += 0.25
    ep_path.write_text(json.dumps(ep, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="window replication"):
        res.load_row(_entry(run), ring_proxy=False, rule=RULE)


def test_a_desynchronised_aggregate_is_refused(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(8), ge_mods=ge)
    sm_path = run / "scenario_metrics.json"
    sm = json.loads(sm_path.read_text())
    sm["aggregate"]["schedule_completion_rate"] = 0.123
    sm_path.write_text(json.dumps(sm, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="does not reproduce summarize"):
        res.load_row(_entry(run), ring_proxy=False, rule=RULE)


def test_ring_proxy_scores_the_ring_row_off_heading_error_max_abs(tmp_path, ge, three_heads):
    slow_a, slow_b, _ = three_heads
    rng = np.random.default_rng(9)
    a = make_run(tmp_path / "ra", "ring__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    b = make_run(tmp_path / "rb", "ring__omni", head="2b", rng=rng, closes=0.4, ge_mods=ge)
    rows = (_entry(slow_a, head="1c-004"), _entry(slow_b, head="2b"),
            _entry(a, head="1c-004"), _entry(b, head="2b"))
    default = res.resolve(_cands(*rows))
    assert {i["facing_closed_source"] for i in default["inputs"]} == {"metric"}
    proxied = res.resolve(_cands(*rows), ring_proxy=True)
    by_row = {(i["head"], i["row"]): i["facing_closed_source"] for i in proxied["inputs"]}
    assert by_row[("1c-004", "ring__omni")] == "proxy"
    assert by_row[("1c-004", "slow__omni")] == "metric", "only the ring rows take the proxy"
    row = res.load_row(_entry(a), ring_proxy=True, rule=RULE)
    assert row.facing_closed_source == "proxy"
    # the proxy is exactly the plan's documented formula, per episode
    ep = json.loads((a / "metrics" / "episode_01.json").read_text())
    fac = ep["holds"]["turn_l90"]["facing"]
    want = 1.0 - fac["heading_error_final_abs"] / max(fac["heading_error_max_abs"],
                                                      res.FACING_CLOSED_MIN_INITIAL_RAD)
    assert row.cols["closed:turn_l90"][1] == pytest.approx(max(-1.0, min(1.0, want)))
    assert proxied["by_row"]["ring__omni"]["metrics"][
        "facing_closed_by_hold.turn_l90"]["source"] == "proxy"
    assert default["by_row"]["ring__omni"]["metrics"][
        "facing_closed_by_hold.turn_l90"]["source"] == "raw"


def test_a_row_with_no_raws_falls_back_to_the_proxy(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(10),
                   with_raws=False, ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.facing_closed_source == "proxy"
    assert row.mask_deviation is None


def test_a_proxy_row_is_never_paired_against_a_rescored_row(tmp_path, ge):
    rng = np.random.default_rng(11)
    a = make_run(tmp_path / "a", "slow__omni", head="1c-004", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="2b", rng=rng, with_raws=False, ge_mods=ge)
    with pytest.raises(res.ResolutionError, match="facing_closed sources differ"):
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b")))


# -------------------------------------------------------------------------------- blocks and record
def test_the_record_carries_the_schema_v1_interface(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    assert out["schema"] == 1
    assert out["B"] == RULE["bootstrap"]["B"], "the yaml's bootstrap block is honoured"
    assert out["rng_seed"] == RULE["bootstrap"]["rng_seed"]
    assert out["decision_rule"] == RULE, "frozen verbatim"
    assert out["seed_terms"] == {}, "D00 is eval-only; D00b/D00c are the driver's"
    for key in ("created", "inputs", "pairs", "metrics", "blocks", "process_floor", "notes"):
        assert key in out, key
    for i in out["inputs"]:
        assert set(("run_dir", "row", "head", "provenance_ok", "accepted", "reason",
                    "env_vars_defaulted")) <= set(i)
        assert Path(i["run_dir"]).is_absolute()
    for p in out["pairs"]:
        assert set(("a", "b", "row", "n")) <= set(p)
    for mid, m in out["metrics"].items():
        assert m["block"] in ("F", "T", "guard", "F_abs", "T_abs"), (mid, m["block"])
        assert m["level"] in (0.95, 0.99)
        assert m["source"] in ("raw", "proxy", "json")
        assert isinstance(m["n"], int)
        assert (m["hold"] is None) or isinstance(m["hold"], str)
    for block, bl in out["blocks"].items():
        assert block in ("F", "T", "F_abs", "T_abs")
        assert bl["members"] and set(bl["members"]) <= set(out["metrics"])
    # guards are one-sided at 99 %, block members two-sided at 95 %
    guards = {mid for mid, m in out["metrics"].items() if m["block"] == "guard"}
    assert guards == set(RULE["guards"]["metrics"])
    assert all(out["metrics"][g]["one_sided"] and out["metrics"][g]["level"] == 0.99 for g in guards)
    assert all(not out["metrics"][m]["one_sided"] for m in out["blocks"]["F"]["members"])


def test_watch_metrics_are_out_of_the_scored_dict_and_the_right_handed_holds_are_borrowed(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    assert "ratio_by_hold.move_turn" not in out["metrics"], "WATCH is never a scored member"
    assert "ratio_by_hold.move_turn" in out["watch_metrics"]
    assert "track_ratio_tail" not in out["metrics"] and "track_ratio_tail" not in out["watch_metrics"]
    assert any("track_ratio_tail" in n for n in out["notes"])
    borrowed = out["watch_metrics"]["ratio_by_hold.lat_r"]
    assert borrowed["borrowed_from"] == "ratio_by_hold.lat_l"
    assert borrowed["R_eval"] == pytest.approx(out["metrics"]["ratio_by_hold.lat_l"]["R_eval"])
    assert out["watch_metrics"]["ratio_by_hold.move_turn_r"]["borrowed_from"] == \
        "ratio_by_hold.move_turn"
    assert any("borrow" in n for n in out["notes"])
    assert out["aliases"]["speed_ratio_by_hold.fwd"] == "ratio_by_hold.fwd"


def test_the_block_half_width_is_bootstrapped_on_z_not_propagated(three_heads):
    a, b, _ = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b")))
    R_F = out["blocks"]["F"]["R_eval"]
    members = out["blocks"]["F"]["members"]
    # each standardised member has half-width ~1 by construction, so a propagated sum would be ~1
    # (perfect correlation) and an independence model 1/sqrt(k); a directly bootstrapped z lands
    # strictly between, never above 1.
    assert 1.0 / math.sqrt(len(members)) * 0.8 < R_F < 1.05
    mde = out["blocks"]["F"]["mde"]
    for mid in members:
        assert mde[mid] == pytest.approx(R_F * out["metrics"][mid]["R_eval"])


def test_a_metric_with_too_few_usable_episodes_reports_no_half_width(tmp_path, ge):
    """A spread estimated off one or two episodes is meaningless; MISSING is the loud answer."""
    rng = np.random.default_rng(12)
    rule = json.loads(json.dumps(RULE))
    rule["watch"] = ["facing_by_hold.turn_l90.settle"]
    a = make_run(tmp_path / "a", "slow__omni", head="x", rng=rng, closes=0.0, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="y", rng=rng, closes=0.0, ge_mods=ge)
    out = res.resolve(_cands(_entry(a, head="x"), _entry(b, head="y"), rule=rule))
    # closes=0.0 means the turn holds' heading error never comes inside the settle tolerance, so
    # settle_time_s is None on every episode of those holds.
    entry = out["watch_metrics"]["facing_by_hold.turn_l90.settle"]
    assert entry["n_valid"] < res.MIN_R_EVAL_N, entry
    assert entry["R_eval"] is None
    assert any("MISSING" in n and "settle" in n for n in out["notes"])
    # a hold that DOES settle keeps its half-width
    assert out["watch_metrics"]["facing_by_hold.turn_l90.settle"]["block"] == "watch"


def test_every_registered_id_reproduces_summarizes_own_number(tmp_path, ge):
    """The reducer check is the contract: a per-episode reconstruction that does not reproduce the
    recorded aggregate is resampling a different estimator than the decision rule reads."""
    rule = json.loads(json.dumps(RULE))
    rule["guards"]["metrics"] = ["completion", "tripod_walking_dir", "turn_survival_l90",
                                 "turn_survival_r90", "tripod", "tracking", "facing_error_rms",
                                 "yaw_progress_turn_min"]
    rule["watch"] = ["tripod_by_hold.move_turn", "survival_by_hold.back",
                     "facing_by_hold.turn_r90.settle", "tripod_walking", "track_ratio_tail"]
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(16), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=rule)
    specs, unresolved = res.specs_from_rule(rule, row.hold_labels)
    assert set(unresolved) == {"tripod_walking", "track_ratio_tail"}, unresolved
    assert row.reducer_check, "the check must actually have run"
    for mid, dev in row.reducer_check.items():
        assert dev == pytest.approx(0.0, abs=1e-9), (mid, dev)
    covered = {s.mid for s in specs} - {"facing_closed_by_hold.turn_l90",
                                        "facing_closed_by_hold.turn_r90",
                                        "facing_closed_by_hold.move_turn"}
    assert covered <= set(row.reducer_check), covered - set(row.reducer_check)
    assert res.why_unresolved("tripod_walking", row.hold_labels).startswith("the harness decides")
    assert "training-log" in res.why_unresolved("track_ratio_tail", row.hold_labels)


def test_the_screen_row_is_required(tmp_path, ge):
    rng = np.random.default_rng(13)
    a = make_run(tmp_path / "a", "ring__omni", head="p", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "ring__omni", head="q", rng=rng, ge_mods=ge)
    with pytest.raises(res.ResolutionError, match="screen row"):
        res.resolve(_cands(_entry(a, head="p"), _entry(b, head="q")))


def test_charter_floors_are_cited_and_superseded_never_applied(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    assert out["charter_floors"] == RULE["charter_floors"]
    for mid, floor in RULE["charter_floors"].items():
        R = out["metrics"][mid]["R_eval"]
        assert R != pytest.approx(floor), "the floor must not become the R"
        assert any(mid in n and "charter floor cited" in n for n in out["notes"])


# ------------------------------------------------------------------------------------ determinism
def test_the_record_is_deterministic_under_its_rng_seed(three_heads, tmp_path):
    a, b, c = three_heads
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a-reeval"))
    one, two = res.resolve(dict(cands)), res.resolve(dict(cands))
    one.pop("created"), two.pop("created")
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    other = res.resolve(dict(cands), rng_seed=RULE["bootstrap"]["rng_seed"] + 1)
    assert other["rng_seed"] != one["rng_seed"]
    assert any(other["metrics"][m]["R_eval"] != one["metrics"][m]["R_eval"] for m in one["metrics"])


def test_adding_a_later_input_leaves_every_existing_pair_bit_identical(tmp_path, ge, three_heads):
    """The process-noise floor row lands weeks after D00; it must not move a recorded R.

    Each pair's bootstrap is seeded from ``(rng_seed, pair name)``, not drawn from one shared stream,
    so a new input adds a pair without re-rolling the others. Decision 6 says R is never re-fitted --
    a shared stream would silently re-fit every one of them the next time the tool ran.
    """
    a, b, c = three_heads
    before = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                                _entry(c, head="2a-reeval")))
    a2 = make_run(tmp_path / "a2", "slow__omni", head="1c-004", rng=np.random.default_rng(99),
                  closes=0.7, ge_mods=ge)
    after = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                               _entry(c, head="2a-reeval"),
                               _entry(a2, head="1c-004", process_floor_pair=True)))
    assert after["process_floor"], "the new row must produce a process floor"
    for mid, m in before["metrics"].items():
        assert after["metrics"][mid]["R_eval"] == m["R_eval"], mid
    for block, bl in before["blocks"].items():
        assert after["blocks"][block]["R_eval"] == bl["R_eval"], block


def test_an_eval_paired_against_itself_has_a_process_floor_of_exactly_zero(tmp_path, ge, three_heads):
    """The paired construction cancels identical inputs exactly, so any non-zero floor from the real
    same-head re-eval is genuine process noise rather than bootstrap slack."""
    a, b, _ = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(a, head="1c-004", process_floor_pair=True)))
    assert out["process_floor"]
    assert set(out["process_floor"].values()) == {0.0}


def test_write_resolution_round_trips_and_is_sorted(three_heads, tmp_path):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    path = res.write_resolution(out, tmp_path / "camp" / "resolution.json")
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text()) == json.loads(json.dumps(out))


def test_a_missing_run_dir_is_a_refusal_not_a_traceback(tmp_path):
    """The driver reports "resolution input <dir> is not on this machine (resolution.py refuses it)",
    so a missing dir must come back as a ResolutionError rather than a FileNotFoundError."""
    with pytest.raises(res.ResolutionError, match="unusable"):
        res.load_row(_entry(tmp_path / "nope"), ring_proxy=False, rule=RULE)
    half = tmp_path / "half"
    (half / "metrics").mkdir(parents=True)
    (half / "run_meta.json").write_text("{}")
    with pytest.raises(res.ResolutionError, match="scenario_metrics.json is missing"):
        res.load_row(_entry(half), ring_proxy=False, rule=RULE)


def test_the_resolution_line_quotes_the_bootstrap_actually_used(three_heads):
    a, b, c = three_heads
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))
    out = res.resolve(dict(cands), bootstrap=311)
    assert out["B"] == 311
    assert "B=311" in out["resolution_line"]
    assert "guards @0.99 one-sided" in out["resolution_line"]
    yaml_default = res.resolve(dict(cands))
    assert yaml_default["B"] == RULE["bootstrap"]["B"]
    assert f"B={RULE['bootstrap']['B']}" in yaml_default["resolution_line"]


def test_main_writes_the_default_paths(three_heads, tmp_path):
    a, b, c = three_heads
    camp = tmp_path / "camp"
    camp.mkdir()
    import yaml
    (camp / "candidates.yaml").write_text(yaml.safe_dump(
        _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))))
    assert res.main(["--campaign-dir", str(camp)]) == 0
    assert (camp / "resolution.json").exists()
    assert res.main(["--campaign-dir", str(tmp_path / "missing")]) == 1


# ---------------------------------------------------------------------------- the curriculum hook
def test_curriculum_runs_rederives_the_right_handed_holds(tmp_path, ge, three_heads):
    a, b, c = three_heads
    rng = np.random.default_rng(14)
    cur_a = make_run(tmp_path / "ca", "curriculum__omni", head="X01", rng=rng, n_eps=10, ge_mods=ge)
    cur_b = make_run(tmp_path / "cb", "curriculum__omni", head="X05", rng=rng, n_eps=10, ratio=0.5,
                     ge_mods=ge)
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))
    borrowed = res.resolve(dict(cands))["watch_metrics"]["ratio_by_hold.lat_r"]
    assert "borrowed_from" in borrowed
    out = res.resolve(dict(cands), curriculum_runs=[cur_a, cur_b])
    got = out["watch_metrics"]["ratio_by_hold.lat_r"]
    assert "borrowed_from" not in got
    assert got["rederived_from"], "the re-derived entry names the curriculum pair it came from"
    assert got["R_eval"] != pytest.approx(borrowed["R_eval"])
    assert "curriculum__omni" in out["by_row"]
    assert any("re-derived ratio_by_hold.lat_r" in n for n in out["notes"])


def test_curriculum_runs_refuses_a_non_curriculum_row(three_heads, tmp_path):
    a, b, c = three_heads
    with pytest.raises(res.ResolutionError, match="curriculum"):
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")),
                    curriculum_runs=[a])


def test_one_curriculum_run_keeps_the_borrowed_R(tmp_path, ge, three_heads):
    a, b, c = three_heads
    cur = make_run(tmp_path / "ca", "curriculum__omni", head="X01",
                   rng=np.random.default_rng(15), n_eps=8, ge_mods=ge)
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")),
                      curriculum_runs=[cur])
    assert out["watch_metrics"]["ratio_by_hold.lat_r"]["borrowed_from"] == "ratio_by_hold.lat_l"
    assert any("fewer than two heads" in n for n in out["notes"])


# ------------------------------------------------------------------ decision 6's null construction
def _null_columns(n, rng, *, k_f=6, k_t=4):
    """Per-episode columns with the real shapes: a correlated latent plus per-metric noise, holes
    where an episode produced nothing, and a Bernoulli completion / survival pair."""
    latent = rng.standard_normal(n)
    cols, specs = {}, []
    for i, hold in enumerate(("turn_l90", "turn_r90", "move_turn")):
        for key, stem, block in (("closed", "facing_closed_by_hold", ("F", "F_abs")),
                                 ("progress", "yaw_progress_by_hold", ("F",))):
            v = 0.5 * latent + rng.standard_normal(n)
            v[rng.random(n) < 0.35] = np.nan          # only ~65 % of episodes reach a turn hold
            cols[f"{key}:{hold}"] = v
            specs.append(res._spec(f"{stem}.{hold}", block[0], block, hold, (f"{key}:{hold}",)))
    for hold in ("fwd", "lat_l", "diag", "back"):
        v = 0.4 * latent + rng.standard_normal(n)
        v[rng.random(n) < 0.2] = np.nan
        cols[f"ratio:{hold}"] = v
        specs.append(res._spec(f"ratio_by_hold.{hold}", "T", ("T",), hold, (f"ratio:{hold}",)))
        cols[f"tripod:{hold}"] = 0.5 + 0.1 * latent + 0.1 * rng.standard_normal(n)
    cols["completed"] = (latent + rng.standard_normal(n) > 0).astype(np.float64)
    cols["surv:back"] = (latent + rng.standard_normal(n) > -1.0).astype(np.float64)
    cols["surv:turn_l90"] = np.minimum(cols["surv:back"],
                                       (latent + rng.standard_normal(n) > -0.6).astype(np.float64))
    cols["surv:turn_r90"] = np.minimum(cols["surv:turn_l90"],
                                       (latent + rng.standard_normal(n) > -0.4).astype(np.float64))
    specs.append(res._spec("completion", "guard", (), None, ("completed",), "mean",
                           level=0.99, one_sided=True))
    specs.append(res._spec("tripod_walking_dir", "guard", (), None,
                           tuple(f"tripod:{h}" for h in ("fwd", "lat_l", "diag", "back")),
                           "min_of_medians", level=0.99, one_sided=True))
    specs.append(res._spec("turn_survival_l90", "guard", (), "turn_l90",
                           ("surv:turn_l90", "surv:back"), "ratio_of_means",
                           level=0.99, one_sided=True))
    specs.append(res._spec("turn_survival_r90", "guard", (), "turn_r90",
                           ("surv:turn_r90", "surv:turn_l90"), "ratio_of_means",
                           level=0.99, one_sided=True))
    del k_f, k_t
    return cols, specs


def test_the_null_candidate_is_calibrated_at_its_declared_levels():
    """Decision 6's pinned test: a run submitted against itself must almost always read all-TIE.

    Two independent episode resamples of one run, 1000 trials, thresholds re-derived by the same
    bootstrap on the same resample distribution. The per-block marginal then pins the quantile
    arithmetic itself (a 95 % spread is exceeded 5 % of the time), and all-TIE lands in the plan's
    band -- above the 0.95**2 * 0.99**4 ~ 0.87 independence bound, because blocks and guards share
    the fall events that make an episode's holds missing together.
    """
    rng = np.random.default_rng(20260919)
    cols, specs = _null_columns(100, rng)
    out = res.null_construction(cols, specs, n=100, B=2000, N=1000, rng=rng)
    assert 0.85 <= out["all_tie"] <= 0.98, out
    for block in ("F", "T"):
        assert out["blocks"][block] == pytest.approx(0.95, abs=0.03), (block, out["blocks"])
    for guard, tie in out["guards"].items():
        assert tie >= 0.96, (guard, tie)
    # widening every threshold can only add TIEs
    wide = res.null_construction(cols, specs, n=100, N=1000, rng=np.random.default_rng(1),
                                 R_metrics={k: (None if v is None else 3 * v)
                                            for k, v in out["R_metrics"].items()},
                                 R_blocks={k: (None if v is None else 3 * v)
                                           for k, v in out["R_blocks"].items()})
    assert wide["all_tie"] >= out["all_tie"]


@pytest.mark.skipif(not RECORDED_1C.exists(), reason="the recorded 1c-004 slow row is untracked")
def test_the_null_candidate_on_the_recorded_1c_004_row():
    """The plan's own dataset: the 1c-004 ``slow__omni`` row as candidate and control."""
    row = res.load_row({"run_dir": str(RECORDED_1C)}, ring_proxy=False, rule=RULE)
    assert row.mask_deviation == 0.0 and row.facing_closed_source == "metric"
    specs, _ = res.specs_from_rule(RULE, row.hold_labels)
    out = res.null_construction(row.cols, specs, n=len(row.env_index), B=2000, N=1000,
                                rng=np.random.default_rng(20260919))
    assert 0.85 <= out["all_tie"] <= 0.98, out
    for block in ("F", "T"):
        assert out["blocks"][block] == pytest.approx(0.95, abs=0.03), (block, out["blocks"])


# ---------------------------------------------------------------------------------- layout / self-pins
def test_repo_root_is_self_relative_and_the_source_carries_no_dated_path():
    assert Path(res.REPO).resolve() == REPO
    source = RES_PY.read_text()
    assert "experiments/2026-" not in source
    assert "rx." not in source
    assert "rf." not in source
    assert "STUDENT_TASK" not in source
    assert not (TOOLS / "__init__.py").exists()


def test_summary_text_names_every_input_and_the_resolution_line(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    text = res.summary_text(out)
    for head in ("1c-004", "2b", "2a-reeval"):
        assert head in text
    assert out["resolution_line"] in text
    assert "guards @0.99 one-sided" in out["resolution_line"]
    assert "B=%d" % out["B"] in out["resolution_line"]


# ------------------------------------------------------------ the null construction under D00's R
LADDER = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments"
          / "2026-09-19_facing_tracking_ladder")
SHIPPED_D00 = LADDER / "resolution.json"


def test_the_null_construction_is_run_under_D00s_own_cross_head_R():
    """Decision 6 asks for the all-TIE band with the eval-only R AND adds "under D00's cross-head R
    the marginal is asserted only >= 0.92".

    That second construction is the one the campaign actually decides in: every candidate is judged
    against the SHIPPED half-widths, not against a threshold re-derived on its own resample
    distribution. ``resolution.py`` computes it into ``diagnostics.null_under_d00_R`` at D00 time,
    but nothing asserted it, so a later re-measure of D00 (a new input, a different B or rng_seed,
    the curriculum rows landing) could drop the null candidate's all-TIE rate under the plan's 0.85
    with the suite still green.
    """
    if not (RECORDED_1C.exists() and SHIPPED_D00.exists()):
        pytest.skip("the recorded 1c-004 row or the shipped resolution.json is untracked")
    shipped = json.loads(SHIPPED_D00.read_text())
    row = res.load_row({"run_dir": str(RECORDED_1C)}, ring_proxy=False, rule=RULE)
    specs, _ = res.specs_from_rule(RULE, row.hold_labels)
    scored = shipped["by_row"][res.SCREEN_ROW]
    r_metrics = {mid: m["R_eval"] for mid, m in scored["metrics"].items()}
    r_blocks = {name: b["R_eval"] for name, b in scored["blocks"].items()}
    assert r_metrics and r_blocks, "the shipped record carries the screen row's own table"
    out = res.null_construction(row.cols, specs, n=len(row.env_index), B=2000, N=1000,
                                rng=np.random.default_rng(20260919),
                                R_metrics=r_metrics, R_blocks=r_blocks)
    assert 0.85 <= out["all_tie"] <= 0.98, out["all_tie"]
    for block in ("F", "T"):
        assert out["blocks"][block] >= 0.92, (block, out["blocks"])
    # the record's own diagnostic is the same construction, so it must agree qualitatively
    diag = (shipped.get("diagnostics") or {}).get("null_under_d00_R") or {}
    assert 0.85 <= diag["all_tie"] <= 0.98, diag
    assert all(v >= 0.92 for v in diag["blocks"].values()), diag["blocks"]
    # ... and it names the degenerate guard rather than hiding it (the CHARTER caveat)
    assert min(diag["guards"].values()) >= 0.90, diag["guards"]


def test_the_shipped_record_pins_its_control_head_rather_than_the_input_order():
    """``R_eval`` is the MAX over the pairs a control forms, so the control is load-bearing. With
    nothing in the yaml naming it, a reorder of ``resolution_inputs`` would silently re-fit D00."""
    if not SHIPPED_D00.exists():
        pytest.skip("the shipped resolution.json is untracked")
    import yaml
    doc = yaml.safe_load((LADDER / "candidates.yaml").read_text())
    assert doc["decision_rule"]["control_head"] == "1c-004"
    shipped = json.loads(SHIPPED_D00.read_text())
    for sid, table in shipped["by_row"].items():
        assert table["control_head"] == "1c-004", sid


def test_a_metric_in_two_blocks_takes_the_wider_of_their_levels():
    """``facing_closed_by_hold.*`` is in F and F_abs. It gets ONE half-width, so a rule that
    levelled the absolute-units blocks differently used to mis-level the shared members on the order
    of the yaml's keys."""
    rule = json.loads(json.dumps(RULE))
    rule["blocks"]["F"]["level"] = 0.95
    rule["blocks"]["F_abs"]["level"] = 0.99
    specs, _ = res.specs_from_rule(rule, ("turn_l90", "turn_r90", "move_turn", "fwd", "lat_l",
                                          "diag", "back"))
    shared = [s for s in specs if s.mid == "facing_closed_by_hold.turn_l90"]
    assert len(shared) == 1 and shared[0].level == 0.99
    # and the other direction: swapping which block is tighter gives the same answer
    rule["blocks"]["F"]["level"], rule["blocks"]["F_abs"]["level"] = 0.99, 0.95
    specs2, _ = res.specs_from_rule(rule, ("turn_l90", "turn_r90", "move_turn", "fwd", "lat_l",
                                           "diag", "back"))
    shared2 = [s for s in specs2 if s.mid == "facing_closed_by_hold.turn_l90"]
    assert len(shared2) == 1 and shared2[0].level == 0.99
