"""Task 4 pipeline thresholds (``crab_hex_omni_task/experiments/tools/pipeline/thresholds.py``): the math only, on CPU.

Pinned here (PLAN "Gates -> Rule -> Self-checks"):

* the n = 3 predictive simulation: ~6.7 % for a range-set line (R = 0), ~1 % at R = 2 sigma, monotone in R / sigma;
* sigma_hat^2 = (range / 1.693)^2 + (R / 1.96)^2;
* line construction: min - max(R, range, G), its lower-is-better mirror, and the |r - 1| tracking mirror;
* the fixed-line predictive t (2 df), including the degenerate zero-spread case;
* the binomial survival floor (pooled misses), its Jeffreys beta-binomial sensitivity, and the miss-count boundary;
* the Poisson count line (K4: lambda = 8/6 -> ~4.65 % per read at <= 3) and its gamma-Poisson sensitivity;
* the paired line Phi(-margin / SE), the student's two-block product, and the 0-kg margin with the drift scaled by
  its_past_p2a / 5000;
* the payload a-priori formula and resolution.binomial_guard_floor's values (0.0717 / 0.0507);
* thresholds v1.1 (CHARTER 2026-10-21): the inputs = v1 + the four SN G1 WATCH ids; a degenerate fixed-t line's rate
  is None + 'degenerate/unvalidated' and the sums count it out explicitly; v1.1 = v1 line for line otherwise and
  regenerates byte for byte (v0 / v1 still do: their maps keep the plain read);
* thresholds v1.2 (CHARTER 2026-10-22, F19 p3b -> "amend + switch now"): every trained translate category's decisive
  0-kg statistic is ``track_err <c>`` = |r_head - 1| - |r_p2a - 1| (an overshoot moving toward 1.0 passes, an
  undershoot or an overshoot growing fails, 1.06 -> 0.94 reads 0), R = the 99 % guard R, SE / drift on |r - 1|;
  move_turn is unchanged; the raw translate drop is a WATCH row; v1.2 = v1.1 line for line except the 0-kg line, its
  drift carries the track_err keys (recomputed from X12 / X08), the self-check and every golden known-bad are v1.1's,
  and it regenerates byte for byte.

The module is loaded by path (TP has no ``__init__.py``) with bytecode writing off.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
TP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/pipeline/thresholds.py"


@pytest.fixture(scope="module")
def T():
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("task4_thresholds_under_test", str(TP))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_no_init_in_tp():
    assert not (TP.parent / "__init__.py").exists()


# ---------------------------------------------------------------------- n = 3 predictive simulation
def test_three_seed_rate_range_set_line(T):
    p = T.simulate_three_seed_rate(0.0, nsim=400_000, seed=1)
    assert 0.060 < p < 0.073          # PLAN: ~6.7 %


def test_three_seed_rate_at_two_sigma(T):
    p = T.simulate_three_seed_rate(2.0, nsim=400_000, seed=2)
    assert 0.007 < p < 0.013          # PLAN: ~1 % at R ~ 2 sigma


def test_three_seed_rate_monotone_and_table(T):
    ps = [T.simulate_three_seed_rate(r, nsim=100_000, seed=3) for r in (0.0, 1.0, 2.0, 3.0)]
    assert all(a >= b for a, b in zip(ps, ps[1:]))
    tab = T.ThreeSeedTable(nsim=50_000, seed=4, r_max=3.0, step=0.5)
    assert tab(0.0) == pytest.approx(tab.p[0])
    assert tab(0.25) == pytest.approx(0.5 * (tab.p[0] + tab.p[1]))
    assert tab(10.0) == pytest.approx(tab.p[-1])      # clamped
    again = T.ThreeSeedTable(nsim=50_000, seed=4, r_max=3.0, step=0.5)
    assert np.array_equal(tab.p, again.p)            # deterministic for a seed


def test_three_seed_rate_uses_R_over_sigma_hat(T):
    tab = T.ThreeSeedTable(nsim=50_000, seed=5)
    vals, R = [0.815, 0.725, 0.8125], 0.0677
    out = T.three_seed_rate(vals, R, tab)
    s = math.sqrt((0.09 / 1.693) ** 2 + (R / 1.96) ** 2)
    assert out["sigma_hat"] == pytest.approx(s)
    assert out["r_over_sigma"] == pytest.approx(R / s)
    assert out["rate"] == pytest.approx(tab(R / s))
    assert out["r_over_sigma"] <= 1.96 + 1e-12      # sigma_hat >= R / 1.96 always


# ---------------------------------------------------------------------- sigma_hat and line construction
def test_sigma_hat(T):
    assert T.sigma_hat([1.0, 1.0, 1.0], 0.0) == 0.0
    assert T.sigma_hat([0.0, 1.693, 0.5], 0.0) == pytest.approx(1.0)
    assert T.sigma_hat([2.0, 2.0, 2.0], 1.96) == pytest.approx(1.0)
    assert T.sigma_hat([0.0, 1.693, 1.0], 1.96) == pytest.approx(math.sqrt(2.0))


def test_three_seed_line(T):
    v = [0.815, 0.725, 0.8125]
    assert T.three_seed_line(v, R=0.04, G=0.0677) == pytest.approx(0.725 - 0.09)      # range binds
    assert T.three_seed_line(v, R=0.2, G=0.0677) == pytest.approx(0.725 - 0.2)        # R binds
    assert T.three_seed_line([1, 1, 1], R=0.0, G=0.05) == pytest.approx(0.95)         # G binds
    assert T.three_seed_line([10.0, 11.0, 12.0], R=0.5, higher_is_better=False) == pytest.approx(14.0)


def test_tracking_line_mirrors_both_sides(T):
    # |r - 1| = 0.05, 0.10, 0.02 -> max 0.10 + max(R, range 0.08)
    assert T.tracking_line([1.05, 0.90, 1.02], R=0.0) == pytest.approx(0.18)
    assert T.tracking_line([1.05, 0.90, 1.02], R=0.2) == pytest.approx(0.30)


# ---------------------------------------------------------------------- fixed-line predictive t
def test_fixed_line_t_rate_matches_t2(T):
    v, R, line = [0.8781, 0.7979, 0.8675], 0.0526, 0.70
    s = math.sqrt(((0.8781 - 0.7979) / 1.693) ** 2 + (R / 1.96) ** 2)
    t = (line - np.mean(v)) / (s * math.sqrt(1 + 1 / 3))
    out = T.fixed_line_t_rate(v, R, line)
    assert out["rate"] == pytest.approx(sps.t.cdf(t, df=2))
    lo = T.fixed_line_t_rate([0.4, 0.45, 0.42], 0.0, 0.5, higher_is_better=False)
    assert lo["rate"] == pytest.approx(sps.t.cdf((np.mean([0.4, 0.45, 0.42]) - 0.5) /
                                                  (T.sigma_hat([0.4, 0.45, 0.42], 0) * math.sqrt(4 / 3)), df=2))


def test_fixed_line_t_rate_degenerate(T):
    assert T.fixed_line_t_rate([0.0, 0.0, 0.0], 0.0, 0.5, higher_is_better=False)["rate"] == 0.0
    assert T.fixed_line_t_rate([1.0, 1.0, 1.0], 0.0, 0.5, higher_is_better=False)["rate"] == 1.0
    assert T.fixed_line_t_rate([1.0, 1.0, 1.0], 0.0, 0.5)["degenerate"] is True


def test_degenerate_line_rate_is_none_and_the_sums_say_so(T):
    """CHARTER 2026-10-21 decision C: a fixed-t line with sigma_hat 0 (all recorded seeds exactly 0, R = 0 -- R18's SN
    G1 taps/s) has no modelled rate: 'unvalidated' mode reports rate None + 'degenerate/unvalidated' (not the plain
    0.0) and the sums exclude it EXPLICITLY; 'legacy_zero' (v0 / v1) leaves everything as it was."""
    import copy
    fr = T.fixed_line_t_rate([0.0, 0.0, 0.0], 0.0, 1.0, higher_is_better=False)
    deg = {"id": "stand.D1.G1.taps_per_s suite es1", "stage": "D1 (SN)", "kind": "stand", "class": "fixed_t",
           "value_set": "standard", **fr}
    flo = {"id": "s1.floor.ratio.fwd.es1", "stage": "s1", "kind": "floor", "class": "fixed_t", "value_set": "standard",
           "rate": 0.0, "sigma_hat": 0.0, "t": None}             # a floor ratio line carries sigma_hat, not the flag
    ok = {"id": "stand.D1.G1.ks_m_s suite es1", "stage": "D1 (SN)", "kind": "stand", "class": "fixed_t",
          "value_set": "standard", **T.fixed_line_t_rate([0.10, 0.11, 0.09], 0.0, 0.5, higher_is_better=False)}
    lines = [deg, flo, ok]
    assert [T.is_degenerate(x) for x in lines] == [True, True, False]
    assert T.is_degenerate({"class": "fixed_t", "per_median": {"lat_l_022": {"degenerate": True}}, "rate": 0.1})
    legacy = copy.deepcopy(lines)
    assert T.mark_degenerate(legacy, "legacy_zero") == {} and legacy == lines
    plain = T.mark_degenerate(lines, "unvalidated")
    assert plain == {"stand.D1.G1.taps_per_s suite es1|standard": 0.0, "s1.floor.ratio.fwd.es1|standard": 0.0}
    for x in (deg, flo):
        assert x["rate"] is None and x["rate_status"] == "degenerate/unvalidated" == T.DEGENERATE
        assert x["unvalidated"] is True and x["degenerate"] is True and "plain read would be 0" in x["rate_note"]
    assert ok["rate"] > 0 and "rate_status" not in ok
    res = {"lines": lines, "degenerate_mode": "unvalidated", "degenerate_plain_reads": plain}
    s = T.summarize(res, set())["standard"]
    assert s["run_sum"] == pytest.approx(ok["rate"])
    assert s["by_stage"]["D1 (SN)"] == {"sum": pytest.approx(ok["rate"]), "union": pytest.approx(ok["rate"]),
                                        "n_reads": 1, "sum_sensitivity": pytest.approx(ok["rate"]),
                                        "n_excluded_degenerate": 1}
    assert s["by_stage"]["s1"]["n_reads"] == 0 and s["by_stage"]["s1"]["n_excluded_degenerate"] == 1
    ex = s["excluded_degenerate"]
    assert ex["n"] == 2 and ex["plain_read_total"] == 0.0
    assert ex["ids"] == {"D1 (SN)": ["stand.D1.G1.taps_per_s suite es1"], "s1": ["s1.floor.ratio.fwd.es1"]}
    old = T.summarize({"lines": legacy}, set())["standard"]
    assert "excluded_degenerate" not in old and old["by_stage"]["D1 (SN)"]["n_reads"] == 2
    assert T.degenerate_mode({"version": "v0"}) == T.degenerate_mode({"version": "v1"}) == "legacy_zero"
    assert T.degenerate_mode({"version": "v1.1"}) == "unvalidated"
    assert T.degenerate_mode({"version": "v1", "degenerate_fixed_t": "unvalidated"}) == "unvalidated"
    with pytest.raises(ValueError):
        T.degenerate_mode({"version": "v1.1", "degenerate_fixed_t": "zero"})


# ---------------------------------------------------------------------- binomial survival floors
def test_max_ok_misses(T):
    assert T.max_ok_misses(100, 0.90) == 10
    assert T.max_ok_misses(100, 0.80) == 20
    assert T.max_ok_misses(200, 0.80) == 40
    assert T.max_ok_misses(100, 0.885) == 11


def test_binomial_floor_rate(T):
    assert T.binomial_floor_rate(0, 300, 100, 0.9) == 0.0
    p = 6 / 300
    assert T.binomial_floor_rate(6, 300, 100, 0.9) == pytest.approx(sps.binom.sf(10, 100, p))
    assert T.binomial_floor_rate(6, 300, 100, 0.8) < T.binomial_floor_rate(6, 300, 100, 0.9)


def test_betabinom_is_more_conservative_at_zero_misses(T):
    assert T.binomial_floor_rate(0, 300, 100, 0.95) == 0.0
    assert T.betabinom_floor_rate(0, 300, 100, 0.95) > 0.0


# ---------------------------------------------------------------------- Poisson counts
def test_poisson_count_rate_k4(T):
    lam = 8 / 6
    assert T.poisson_count_rate([1, 0, 1, 2, 1, 3], 3) == pytest.approx(sps.poisson.sf(3, lam))
    assert T.poisson_count_rate([1, 0, 1, 2, 1, 3], 3) == pytest.approx(0.0465, abs=5e-4)   # D9 line per read
    assert T.poisson_count_rate([1, 0, 1, 2, 1, 3], 1) == pytest.approx(0.385, abs=1e-3)    # S1-era line <= 1
    assert T.poisson_count_rate([0, 0, 0], 1) == 0.0


def test_nb_count_rate_zero_counts(T):
    # lambda ~ Gamma(0.5, rate 3): predictive NB(0.5, 3/4); P(X >= 1) = 1 - 0.75^0.5
    assert T.nb_count_rate([0, 0, 0], 0) == pytest.approx(1 - 0.75 ** 0.5)
    assert T.nb_count_rate([2, 1, 3], 3) > T.poisson_count_rate([2, 1, 3], 3) * 0.9


# ---------------------------------------------------------------------- paired, student, drift
def test_paired_rate(T):
    assert T.paired_rate(3.0, 1.0) == pytest.approx(sps.norm.cdf(-3.0))
    assert T.paired_rate(0.05, 0.0) == 0.0


def test_student_halt_is_product_of_two_blocks(T):
    assert T.student_halt_rate(0.1) == pytest.approx(0.01)
    assert T.student_halt_rate(0.0) == 0.0


def test_zero_kg_margin_drift_scaling(T):
    m = T.zero_kg_margin(R=0.0367, se_diff=0.0055, drift=-0.062, its_past_p2a=2500)
    assert m["drift_scaled"] == pytest.approx(0.031)
    assert m["3SE"] == pytest.approx(0.0165)
    assert m["margin"] == pytest.approx(0.0367) and m["binds"] == "R"
    m5 = T.zero_kg_margin(R=0.0367, se_diff=0.0055, drift=-0.062, its_past_p2a=5000)
    assert m5["margin"] == pytest.approx(0.062) and m5["binds"] == "drift_scaled"
    m0 = T.zero_kg_margin(R=0.0, se_diff=0.02, drift=0.0, its_past_p2a=0)
    assert m0["margin"] == pytest.approx(0.06) and m0["binds"] == "3SE"


def test_payload_apriori_rate(T):
    # variance 2 sigma^2 + SE^2
    r = T.payload_apriori_rate(0.90, 0.95, 0.02, 0.01)
    assert r == pytest.approx(sps.norm.cdf((0.90 - 0.95) / math.sqrt(2 * 0.02 ** 2 + 0.01 ** 2)))
    lo = T.payload_apriori_rate(0.25, 0.20, 0.02, 0.0, higher_is_better=False)
    assert lo == pytest.approx(sps.norm.cdf((0.20 - 0.25) / math.sqrt(2) / 0.02))


def test_binomial_guard_floor_matches_resolution(T):
    assert T.binomial_guard_floor(1.0, 100) == pytest.approx(0.0717, abs=1e-4)
    assert T.binomial_guard_floor(1.0, 200) == pytest.approx(0.0507, abs=1e-4)


def test_union_and_total(T):
    assert T.total([0.1, 0.2, None, float("nan")]) == pytest.approx(0.3)
    assert T.union([0.1, 0.2]) == pytest.approx(1 - 0.9 * 0.8)
    assert T.union([]) == 0.0


# ---------------------------------------------------------------------- the paired 0-kg check (CHARTER STOP-0)
def _write_res(tmp_path, metrics):
    import json as _json
    p = tmp_path / "res.json"
    p.write_text(_json.dumps({"metrics": metrics}))
    return p


def test_resolution_R_guard_prefers_the_99_percent_entry(T, tmp_path):
    p = _write_res(tmp_path, {
        "cat.fwd.ratio": {"R_eval": 0.041, "level": 0.95},
        "cat.fwd.ratio@guard": {"R_eval": 0.0508, "level": 0.99},
        "cat.move_turn.ratio": {"R_eval": 0.0367, "level": 0.99},     # a guard-block entry with no @guard twin
        "cat.side.ratio": {"R_eval": 0.018, "level": 0.95},            # 95 % only: no guard R
    })
    g = T.resolution_R_guard([p])
    assert g["cat.fwd.ratio"] == pytest.approx(0.0508)
    assert g["cat.move_turn.ratio"] == pytest.approx(0.0367)
    assert "cat.side.ratio" not in g
    assert T.resolution_R([p])["cat.fwd.ratio"] == pytest.approx(0.041)    # the 95 % reader is unchanged


def _eps(cat, ratios, env0=0):
    out = [{"env": env0 + i, "cat": cat, "surv": True, "ratio": r, "facing_closed": None, "yaw_progress": None}
           for i, r in enumerate(ratios)]
    if cat == "fwd":   # every synthetic suite also carries a turn category (facing_closed / yaw_progress rows)
        out += [{"env": 500 + i, "cat": "turn", "surv": True, "ratio": None, "facing_closed": 0.95,
                 "yaw_progress": 0.15} for i in range(len(ratios))]
    return out


def test_zero_kg_tracking_is_a_one_sided_ratio_drop(T):
    R = {"cat.fwd.ratio": 0.04}
    Rg = {"cat.fwd.ratio": 0.05, "cat.move_turn.ratio": 0.04}
    p2a = {1: _eps("fwd", [1.00] * 50) + _eps("move_turn", [0.8] * 50, 100)}
    faster = {1: _eps("fwd", [1.20] * 50) + _eps("move_turn", [0.8] * 50, 100)}    # |r-1| worse, ratio up
    slower = {1: _eps("fwd", [0.90] * 50) + _eps("move_turn", [0.8] * 50, 100)}    # ratio drop 0.10
    rows_f = T.zero_kg_suite_rows(faster, p2a, ["fwd"], R, Rg)
    dec = {r["statistic"]: r for r in rows_f if not r["watch"]}
    wat = {r["statistic"]: r for r in rows_f if r["watch"]}
    assert dec["ratio fwd"]["worse_by"] == pytest.approx(-0.20)        # a faster head is never failed
    assert dec["ratio fwd"]["R"] == pytest.approx(0.05)                # the 99 % guard R
    assert wat["|ratio-1| fwd"]["worse_by"] == pytest.approx(0.20)     # ... but |r-1| reports it (WATCH)
    assert wat["|ratio-1| fwd"]["R"] == pytest.approx(0.04)
    drift = {f"{r['statistic']}|{r['row']}": 0.0 for r in rows_f}
    j = T.zero_kg_judge(rows_f, drift, 2500)
    assert j["verdict"] == "PASS" and [r for r in j["rows"] if r["watch"] and r.get("exceeds")]
    rows_s = T.zero_kg_suite_rows(slower, p2a, ["fwd"], R, Rg)
    j = T.zero_kg_judge(rows_s, {f"{r['statistic']}|{r['row']}": 0.0 for r in rows_s}, 2500)
    assert j["verdict"] == "FAIL" and any("ratio fwd" in f for f in j["fails"])


def test_zero_kg_margin_uses_drift_scaled_by_iterations(T):
    R, Rg = {}, {"cat.fwd.ratio": 0.01, "cat.move_turn.ratio": 0.01}
    p2a = {1: _eps("fwd", [1.00] * 50) + _eps("move_turn", [0.8] * 50, 100)}
    head = {1: _eps("fwd", [0.97] * 50) + _eps("move_turn", [0.8] * 50, 100)}     # drop 0.03
    rows = T.zero_kg_suite_rows(head, p2a, ["fwd"], R, Rg)
    drift = {f"{r['statistic']}|{r['row']}": 0.0 for r in rows}
    drift["ratio fwd|suite es1"] = 0.06                                             # X12 - X08 over 5000 its
    assert T.zero_kg_judge(rows, drift, 1000)["verdict"] == "FAIL"                  # margin 0.012 < 0.03
    assert T.zero_kg_judge(rows, drift, 3000)["verdict"] == "PASS"                  # margin 0.036 >= 0.03


def test_zero_kg_fails_closed(T):
    p2a = {1: _eps("fwd", [1.0] * 10) + _eps("move_turn", [0.8] * 10, 100)}
    rows = T.zero_kg_suite_rows(p2a, p2a, ["fwd"], {}, {"cat.move_turn.ratio": 0.04})   # no guard R for fwd
    j = T.zero_kg_judge(rows, {f"{r['statistic']}|{r['row']}": 0.0 for r in rows}, 2500)
    assert j["verdict"] == "FAIL" and any("guard" in f for f in j["fails"])
    rows = T.zero_kg_suite_rows(p2a, p2a, ["fwd"], {}, {"cat.fwd.ratio": 0.05, "cat.move_turn.ratio": 0.04})
    assert T.zero_kg_judge(rows, {}, 2500)["verdict"] == "FAIL"                    # no recorded drift
    ok = T.zero_kg_judge(rows, {f"{r['statistic']}|{r['row']}": 0.0 for r in rows}, 2500,
                         side22_head=["FAST", "SLOW"], side22_p2a=["FAST", "FAST"])
    assert ok["verdict"] == "FAIL" and ok["side22"]["fast_fast_lost"]
    c = T.zero_kg_completion_row("field_hard", {(0, 1): True}, {(0, 2): True}, (1, 2))
    assert c["missing"]
    assert T.zero_kg_judge([c], {"field_hard completion|field_hard": 0.0}, 2500)["verdict"] == "FAIL"


def test_committed_thresholds_v0_use_the_stop0_semantics():
    import json as _json
    d = _json.loads((REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/"
                            "2026-10-16_task4_pipeline/thresholds_v0.json").read_text())
    z = next(x for x in d["lines"] if x["id"] == "p3a.capability.zero_kg_paired")
    stats = {s["statistic"] for s in z["statistics"]}
    assert "ratio fwd" in stats and not any(s.startswith("|ratio-1|") for s in stats)
    assert {s["statistic"] for s in z["watch_statistics"]} >= {"|ratio-1| fwd", "|ratio-1| move_turn"}
    fwd = next(s for s in z["statistics"] if s["statistic"] == "ratio fwd" and s["row"] == "suite es1")
    assert fwd["R"] == pytest.approx(0.0508, abs=1e-4) and "@guard" in fwd["R_source"]
    back = next(s for s in z["statistics"] if s["statistic"] == "ratio back" and s["row"] == "suite es1")
    assert back["R"] == pytest.approx(0.0617, abs=1e-4)
    assert z["known_good_self_check"] is True
    assert z["row_list"]["field_hard__omni"] == [1, 2]
    fh = next(x for x in d["lines"] if x["id"] == "p2a.capability.field_hard")
    assert fh["env_seeds"] == [1, 2, 3, 4]
    assert d["self_check_failures"] == []


# ---------------------------------------------------------------------- v1 inputs (STEPC_RULES V.2)
PL = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline"


def _sweep_stamp(root: Path, name: str, bins: dict) -> Path:
    import json as _json
    d = root / name
    d.mkdir(parents=True)
    bc = {c: {k: {"median": v} for k, v in vals.items()} for c, vals in bins.items()}
    (d / "scenario_metrics.json").write_text(_json.dumps({"aggregate": {"suite": {"by_category": bc}}}))
    (d / "run_meta.json").write_text("{}")
    (d / "summary.md").write_text("synthetic\n")
    return d


def test_sweep_anchor_lines_are_watch_only(T, tmp_path):
    """``sweep_anchors`` -> per-bin |ratio - 1| lines and the max ADJACENT-bin jump (circular) in ratio, tripod and
    steps_per_m, each <= the anchors' max + range; never decisive, never summed, never self-checked."""
    names = ["d000", "d090", "d180", "d270"]
    ratios = {16: [1.00, 0.95, 0.90, 0.97], 17: [0.98, 0.93, 0.85, 0.99], 18: [1.02, 0.96, 0.92, 0.80]}
    anchors = {}
    for s, rs in ratios.items():
        for es, off in ((1, -0.01), (2, 0.01)):          # a seed's value = the mean of its env seeds' medians
            bins = {c: {"ratio": r + off, "tripod": 0.50 + 0.01 * i, "steps_per_m": 4.0}
                    for i, (c, r) in enumerate(zip(names, rs))}
            anchors[f"s{s}_es{es}"] = str(_sweep_stamp(tmp_path, f"s{s}_es{es}", bins))
    prov = []
    lines = {x["id"]: x for x in T.sweep_watch_lines(T.Repo(tmp_path), anchors, prov)}
    assert len(prov) == 6
    d180 = lines["p2a.watch.sweep.abs_dev.d180"]
    assert d180["line"] == pytest.approx(0.15 + (0.15 - 0.08))                    # |r-1| = .10 .15 .08
    assert d180["seed_ratios"]["s17"] == pytest.approx(0.85)
    jr = lines["p2a.watch.sweep.max_adjacent_jump.ratio"]                         # s18: d270 -> d000 wrap .22
    assert jr["seed_values"]["s18"] == pytest.approx(0.22) and jr["line"] == pytest.approx(0.22 + (0.22 - 0.07))
    assert lines["p2a.watch.sweep.max_adjacent_jump.tripod"]["line"] == pytest.approx(0.03)    # the wrap again
    assert lines["p2a.watch.sweep.max_adjacent_jump.steps_per_m"]["line"] == pytest.approx(0.0)
    for x in lines.values():
        assert x["kind"] == "watch" and x["rate"] is None and x["excluded_from_sum"] and x["direction"] == "<="
    res = {"lines": list(lines.values())}
    assert T.self_check(res, set()) == [] and not any(x["decisive"] for x in res["lines"])
    assert T.summarize(res, set())["recommended"]["run_sum"] == 0.0
    with pytest.raises(ValueError):
        T.sweep_watch_lines(T.Repo(tmp_path), {"s16-es1": anchors["s16_es1"]}, [])
    odd = _sweep_stamp(tmp_path, "odd", {"d000": {"ratio": 1.0}})
    with pytest.raises(ValueError):
        T.sweep_watch_lines(T.Repo(tmp_path), {"s16_es1": anchors["s16_es1"], "s17_es1": str(odd)}, [])


def _recorded_inputs():
    yaml = pytest.importorskip("yaml")
    inputs = yaml.safe_load((PL / "threshold_inputs.yaml").read_text())
    try:
        s18 = inputs["p3a_apriori_s1"][18]
        for e in [*s18["suite"].values(), *s18["field_light"].values(), *inputs["drift_pair"]["x12"]["suite"].values()]:
            if not (REPO / e["stamp"]).is_dir():
                raise FileNotFoundError(e["stamp"])
    except (KeyError, FileNotFoundError) as exc:
        pytest.skip(f"recorded stamps absent: {exc}")
    return inputs


def _pilot_block(inputs: dict, its: float) -> dict:
    """S1 s18's own 0-kg rows as the 'pilot' L090 rows, and X12 vs X08 (the recorded drift pair) as the 0-kg pair."""
    s18 = inputs["p3a_apriori_s1"][18]
    lv = {f"suite@{es}": e for es, e in s18["suite"].items()}
    lv.update({f"field_light@{es}": e for es, e in s18["field_light"].items()})

    def side(x):
        m = {f"suite@{es}": e for es, e in x["suite"].items()}
        m["side22@1"] = x["side22"]
        for row, n in (("field_hard", 2), ("field_light", 2), ("ring_obst", 2), ("field_turn", 1)):
            m.update({f"{row}@{i + 1}": e for i, e in enumerate(x[row][:n])})
        return m

    return {"name": "test: S1 s18 as the pilot", "its_past_p2a": its, "level90": lv,
            "zero_kg": {"head": side(inputs["drift_pair"]["x12"]), "p2a": side(inputs["drift_pair"]["x08"])}}


def test_p3a_pilot_fills_the_p3a_rates(T):
    """``p3a_pilot``: every p3a.stable90.* rate = payload_apriori_rate(line, mu_pilot, sigma_hat_p2a, SE) with
    mu_pilot read as payload_read.level_reads reads it, and the paired 0-kg rate = the sum of Phi(-margin / SE_diff)
    over the pilot pair's decisive statistics. On the recorded X12-vs-X08 pair at 3000 its that sum must equal v0's
    own rate_by_Np[3000]; nothing outside the p3a terms changes."""
    import copy
    import json as _json
    inputs = _recorded_inputs()
    v0 = _json.loads((PL / "thresholds_v0.json").read_text())
    lines = copy.deepcopy(v0["lines"])
    repo = T.Repo(REPO)
    R = T.resolution_R([repo.p(p) for p in inputs["resolution_files"]])
    Rg = T.resolution_R_guard([repo.p(p) for p in inputs["resolution_files"]])
    s22 = T.load_side22_module(repo.p(inputs["side22_classifier"]))
    prov = []
    T.p3a_pilot_rates(repo, _pilot_block(inputs, 3000), lines, R, Rg, s22,
                      inputs["stages"]["p2a"]["trained_translate"], prov)
    by0 = {(x["id"], x.get("value_set")): x for x in v0["lines"]}
    n = 0
    for x in lines:
        if x["id"].startswith("p3a.stable90."):
            n += 1
            assert "pending" not in x and x["mu_pilot"] == pytest.approx(x["s1_0kg_values"]["18"], abs=1e-12)
            sig = x["sigma_hat_p2a"]
            se = sig if x["se_pilot"] is None else x["se_pilot"]
            assert x["rate"] == pytest.approx(T.payload_apriori_rate(x["line"], x["mu_pilot"], sig, se,
                                                                     x["direction"] == ">="))
            assert x["se_stand_in"] == (x["id"].split(".")[2] in ("h_meas", "roll_rms", "pitch_rms")), x["id"]
        elif x["id"] == "p3a.capability.zero_kg_paired":
            assert x["rate"] == pytest.approx(by0[(x["id"], "both")]["rate_by_Np"]["3000"], rel=1e-12)
            assert len(x["pilot_pair"]["per_statistic"]) == 26 and x["pilot_pair"]["unjudged"] == []
        else:
            assert x == by0[(x["id"], x.get("value_set"))], x["id"]
    assert n == 35 and prov and prov[0]["its_past_p2a"] == 3000


def test_absent_v1_blocks_reproduce_thresholds_v0_byte_for_byte(T, tmp_path):
    """No ``p3a_pilot`` / ``sweep_anchors`` block = thresholds_v0.json exactly (the committed file, regenerated)."""
    import subprocess
    _recorded_inputs()
    out = tmp_path / "v0.json"
    rel = (PL / "threshold_inputs.yaml").relative_to(REPO)
    r = subprocess.run([sys.executable, "-I", str(TP), "--inputs", str(rel), "--out", str(out), "--self-check"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_bytes() == (PL / "thresholds_v0.json").read_bytes()


def test_committed_thresholds_v1_is_v0_plus_the_pilot_terms():
    """The PLAN step-8 freeze: thresholds_v1.json = v0's lines with the p3a stable-at-90 and paired 0-kg rates filled
    from the payload pilot (model_22993, its_past_p2a 2999) -- equal to t4_stepC_read's cross-check (stepC_read.json
    v1.p3a_rates) -- plus the 19 WATCH sweep lines; every other line unchanged; the self-check clean."""
    import json as _json
    v0 = _json.loads((PL / "thresholds_v0.json").read_text())
    v1 = _json.loads((PL / "thresholds_v1.json").read_text())
    assert v1["version"] == "v1" and v1["self_check_failures"] == []
    sc = _json.loads((PL / "preflight/stepC_read.json").read_text())["v1"]["p3a_rates"]
    by1 = {(x["id"], x.get("value_set")): x for x in v1["lines"]}
    n = 0
    for lid, ref in sc["stable90"]["lines"].items():
        x = by1[(lid, "both")]
        assert x["rate"] == pytest.approx(ref["rate"], abs=1e-15) and x["mu_pilot"] == pytest.approx(ref["mu_pilot"])
        n += 1
    assert n == 35
    z = by1[("p3a.capability.zero_kg_paired", "both")]
    assert z["rate"] == pytest.approx(sc["zero_kg_paired"]["sum"], rel=1e-12) and z["pilot_pair"]["verdict"] == "PASS"
    assert z["pilot_pair"]["max_margin_used"] == pytest.approx(sc["zero_kg_paired"]["max_margin_used"])
    watch = [x for x in v1["lines"] if x.get("kind") == "watch"]
    assert len(watch) == 19 and all(x["rate"] is None and x["excluded_from_sum"] and not x["decisive"] for x in watch)
    for x in v0["lines"]:
        y = by1[(x["id"], x.get("value_set"))]
        if not (x["id"].startswith("p3a.stable90.") or x["id"] == "p3a.capability.zero_kg_paired"):
            assert y == x, x["id"]
    p3a = sum(x["rate"] for x in by1.values() if x["id"].startswith("p3a.stable90.")) + z["rate"]
    for w in ("standard", "recommended"):
        assert v1["sums"][w]["run_sum"] == pytest.approx(v0["sums"][w]["run_sum"] + p3a, rel=1e-9)


def test_thresholds_v1_regenerates_byte_for_byte(T, tmp_path):
    import subprocess
    _recorded_inputs()
    out = tmp_path / "v1.json"
    rel = (PL / "threshold_inputs_v1.yaml").relative_to(REPO)
    r = subprocess.run([sys.executable, "-I", str(TP), "--inputs", str(rel), "--out", str(out), "--self-check"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_bytes() == (PL / "thresholds_v1.json").read_bytes()


# ---------------------------------------------------------------------- v1.1 (CHARTER 2026-10-21, R18 SN HALT)
G1_WATCH_IDS = ["stand.D1.G1.taps_per_s suite es1", "stand.D1.G1.taps_per_s suite es2",
                "stand.D1.G1.steps_per_s suite es1", "stand.D1.G1.steps_per_s suite es2"]
DEGENERATE_DIFF = {"rate", "rate_status", "rate_note", "unvalidated", "degenerate"}


def test_threshold_inputs_v1_1_is_v1_plus_the_four_watch_ids():
    """Decision A: threshold_inputs_v1_1.yaml = v1 + the four SN G1 stillness ids on recommended.watch (a note citing
    the CHARTER decision) + degenerate_fixed_t: unvalidated (decision C); nothing else changes."""
    yaml = pytest.importorskip("yaml")
    a = yaml.safe_load((PL / "threshold_inputs_v1.yaml").read_text())
    b = yaml.safe_load((PL / "threshold_inputs_v1_1.yaml").read_text())
    assert b["version"] == "v1.1" and b["degenerate_fixed_t"] == "unvalidated"
    assert set(b) - set(a) == {"v1_1_created", "v1_1_note", "degenerate_fixed_t"} and not set(a) - set(b)
    assert all(a[k] == b[k] for k in a if k not in ("version", "recommended"))
    ra, rb = dict(a["recommended"]), dict(b["recommended"])
    assert rb.pop("watch") == ra.pop("watch") + G1_WATCH_IDS
    note = rb.pop("watch_note_v1_1")
    assert "CHARTER 2026-10-21" in note and "K5 / K6" in note and ra == rb


def test_committed_thresholds_v1_1_is_v1_plus_the_amendment():
    """thresholds_v1_1.json = v1 line for line except (A) the four SN G1 ids are WATCH in the recommended set (not
    decisive, out of the recommended sums) and (C) every degenerate fixed-t line (sigma_hat 0) reads rate None +
    'degenerate/unvalidated' instead of the plain 0.0 -- 26 per value set, all stand clauses; the run sums are v1's
    (every excluded plain read was 0.0) with the exclusions counted; the self-check is clean."""
    import json as _json
    v1 = _json.loads((PL / "thresholds_v1.json").read_text())
    v11 = _json.loads((PL / "thresholds_v1_1.json").read_text())
    assert v11["version"] == "v1.1" and v11["self_check_failures"] == []
    assert v11["recommended"]["watch"] == v1["recommended"]["watch"] + G1_WATCH_IDS
    assert [(x["id"], x.get("value_set")) for x in v11["lines"]] == [(x["id"], x.get("value_set")) for x in v1["lines"]]
    for k in v1:
        if k not in ("version", "inputs", "recommended", "lines", "sums"):
            assert v11[k] == v1[k], k
    n_deg = {"standard": 0, "recommended": 0}
    for x, y in zip(v1["lines"], v11["lines"]):
        if x == y:
            continue
        d = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
        assert x["rate"] == 0.0 and x.get("degenerate") is True and x["class"] == "fixed_t", x["id"]
        assert y["rate"] is None and y["rate_status"] == "degenerate/unvalidated" and y["unvalidated"] is True
        assert x["kind"] == "stand" and x.get("sigma_hat") == 0.0
        if x["id"] in G1_WATCH_IDS and x["value_set"] == "recommended":
            assert d == DEGENERATE_DIFF - {"degenerate"} | {"decisive", "watch"}, (x["id"], d)
            assert y["decisive"] is False and y["watch"] is True and x["decisive"] is True
        else:
            assert d <= DEGENERATE_DIFF, (x["id"], d)
        n_deg[x["value_set"]] += 1
    assert n_deg == {"standard": 26, "recommended": 26}
    taps = next(x for x in v11["lines"] if x["id"] == "stand.D1.G1.taps_per_s suite es1"
                and x["value_set"] == "standard")
    assert taps["rate"] is None and taps["seed_values"] == {"16": 0.0, "17": 0.0, "18": 0.0} and taps["decisive"]
    for w, n_ex in (("standard", 26), ("recommended", 22)):
        a, b = v1["sums"][w], v11["sums"][w]
        for k in ("run_sum", "run_union", "run_sum_sensitivity"):
            assert b[k] == a[k], (w, k)
        assert b["excluded_degenerate"]["n"] == n_ex and b["excluded_degenerate"]["plain_read_total"] == 0.0
        for st, s in a["by_stage"].items():
            t = b["by_stage"][st]
            assert t["sum"] == s["sum"] and t["n_reads"] + t["n_excluded_degenerate"] + (
                4 if (w == "recommended" and st == "D1 (SN)") else 0) == s["n_reads"], (w, st)
    assert v11["sums"]["recommended"]["by_stage"]["D1 (SN)"]["n_reads"] == 28
    assert v11["degenerate_fixed_t"]["mode"] == "unvalidated" and len(v11["degenerate_fixed_t"]["plain_reads"]) == 52


def test_thresholds_v1_1_regenerates_byte_for_byte(T, tmp_path):
    import subprocess
    _recorded_inputs()
    out = tmp_path / "v1_1.json"
    rel = (PL / "threshold_inputs_v1_1.yaml").relative_to(REPO)
    r = subprocess.run([sys.executable, "-I", str(TP), "--inputs", str(rel), "--out", str(out), "--self-check"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "degenerate lines excluded (rate None): standard 26, recommended 22" in r.stdout
    assert out.read_bytes() == (PL / "thresholds_v1_1.json").read_bytes()


# ---------------------------------------------------------------------- v1.2 (CHARTER 2026-10-22, F19 p3b -> amend)
R_T = {"cat.fwd.ratio": 0.04, "cat.move_turn.ratio": 0.03}
RG_T = {"cat.fwd.ratio": 0.05, "cat.move_turn.ratio": 0.04}


def _fwd_pair(head_fwd: float, p2a_fwd: float, n: int = 50, mt=(0.8, 0.8)):
    """A synthetic paired suite (one env seed): fwd at a constant ratio (head vs p2a), move_turn, and the turn rows."""
    head = {1: _eps("fwd", [head_fwd] * n) + _eps("move_turn", [mt[0]] * n, 100)}
    p2a = {1: _eps("fwd", [p2a_fwd] * n) + _eps("move_turn", [mt[1]] * n, 100)}
    return head, p2a


def _judge(T, head, p2a, translate, its=2500, drift=None):
    rows = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, RG_T, translate=translate)
    d = {f"{r['statistic']}|{r['row']}": 0.0 for r in rows}
    d.update(drift or {})
    return rows, T.zero_kg_judge(rows, d, its)


def test_track_err_mode_reads_and_defaults(T):
    """The input key and its defaults: the frozen v0 / v1 / v1.1 maps read ratio_drop (byte-for-byte regeneration),
    every later version track_err; an unknown mode is refused (by the map reader and by the row builder)."""
    assert T.ZERO_KG_TRANSLATE_MODES == ("ratio_drop", "track_err")
    for v in ("v0", "v1", "v1.1"):
        assert T.zero_kg_translate_mode({"version": v}) == "ratio_drop"
    assert T.zero_kg_translate_mode({"version": "v1.2"}) == "track_err"
    assert T.zero_kg_translate_mode({"version": "v1.1", "zero_kg_translate_statistic": "track_err"}) == "track_err"
    with pytest.raises(ValueError):
        T.zero_kg_translate_mode({"version": "v1.2", "zero_kg_translate_statistic": "abs"})
    h, p = _fwd_pair(1.0, 1.0)
    with pytest.raises(ValueError):
        T.zero_kg_suite_rows(h, p, ["fwd"], R_T, RG_T, translate="abs")


def test_track_err_overshoot_moving_toward_one_passes(T):
    """F19 p3b's back line in miniature: p2a overshoots (1.06), the head tracks (1.00). track_err reads -0.06 and
    passes; the raw drop (0.06 > R 0.05) is still RECORDED as a WATCH row that exceeds its margin, never decisive. The
    same pair under the v1.1 rule (ratio_drop) fails -- the defect the amendment removes."""
    head, p2a = _fwd_pair(1.00, 1.06)
    rows, j = _judge(T, head, p2a, "track_err")
    dec = {r["statistic"]: r for r in rows if not r["watch"]}
    wat = {r["statistic"]: r for r in rows if r["watch"]}
    te = dec["track_err fwd"]
    assert te["worse_by"] == pytest.approx(-0.06) and te["key"] == "track_err" and te["category"] == "fwd"
    assert te["head"] == pytest.approx(0.0) and te["p2a"] == pytest.approx(0.06)
    assert te["R"] == pytest.approx(0.05) and "@guard (99 %) on |r - 1|" in te["R_source"]
    assert te["key_path"] == "|aggregate.suite.by_category.fwd.ratio.median - 1|"
    assert te["worse_by_rule"].startswith("|head - 1| - |p2a - 1|")
    assert "ratio fwd" not in dec and "|ratio-1| fwd" not in dec and "|ratio-1| fwd" not in wat
    raw = wat["ratio fwd"]                               # the WATCH raw-drop row
    assert raw["worse_by"] == pytest.approx(0.06) and raw["R"] == pytest.approx(0.05) and raw["key"] == "ratio"
    assert "WATCH raw drop" in raw["R_source"] and raw["worse_by_rule"].startswith("p2a - head")
    assert j["verdict"] == "PASS" and j["fails"] == []
    assert next(r for r in j["rows"] if r["statistic"] == "ratio fwd")["exceeds"] is True
    _rows, j_old = _judge(T, head, p2a, "ratio_drop")
    assert j_old["verdict"] == "FAIL" and j_old["fails"][0].startswith("ratio fwd suite es1")


def test_track_err_undershoot_growing_fails(T):
    """An undershoot growing (0.98 -> 0.90: |r - 1| 0.02 -> 0.10) fails on track_err by 0.08 > 0.05, and a growth
    inside the margin (0.98 -> 0.95) passes; the margin keeps the PLAN form (drift on the new key, scaled by its)."""
    rows, j = _judge(T, *_fwd_pair(0.90, 0.98), "track_err")
    assert j["verdict"] == "FAIL" and j["fails"] == ["track_err fwd suite es1: worse by 0.0800 > margin 0.0500 (R)"]
    assert next(r for r in j["rows"] if r["statistic"] == "track_err fwd")["margin_used"] == pytest.approx(1.6)
    assert _judge(T, *_fwd_pair(0.95, 0.98), "track_err")[1]["verdict"] == "PASS"
    drift = {"track_err fwd|suite es1": 0.09}                                  # X12 - X08 on |r - 1| over 5000 its
    assert _judge(T, *_fwd_pair(0.90, 0.98), "track_err", its=5000, drift=drift)[1]["verdict"] == "PASS"
    assert _judge(T, *_fwd_pair(0.90, 0.98), "track_err", its=2000, drift=drift)[1]["verdict"] == "FAIL"


def test_track_err_overshoot_growing_fails(T):
    """An overshoot growing (1.02 -> 1.10) is an increase in tracking error: FAIL on track_err, while the v1.1 one-sided
    drop never failed a faster head."""
    _rows, j = _judge(T, *_fwd_pair(1.10, 1.02), "track_err")
    assert j["verdict"] == "FAIL" and j["fails"][0].startswith("track_err fwd suite es1: worse by 0.0800")
    assert _judge(T, *_fwd_pair(1.10, 1.02), "ratio_drop")[1]["verdict"] == "PASS"


def test_track_err_crossing_one_reads_no_change(T):
    """1.06 -> 0.94 crosses 1.0 at the same distance: track_err reads 0 (PASS); the raw drop (0.12) is the WATCH row."""
    rows, j = _judge(T, *_fwd_pair(0.94, 1.06), "track_err")
    te = next(r for r in rows if r["statistic"] == "track_err fwd")
    assert te["worse_by"] == pytest.approx(0.0, abs=1e-12)
    assert j["verdict"] == "PASS"
    raw = next(r for r in j["rows"] if r["statistic"] == "ratio fwd")
    assert raw["watch"] and raw["worse_by"] == pytest.approx(0.12) and raw["exceeds"]


def test_track_err_leaves_move_turn_survival_and_facing_unchanged(T):
    """move_turn stays EXACTLY the v1.1 one-sided ratio drop (same worse_by, R, SE_diff, key); survival and facing rows
    and the |ratio-1| move_turn WATCH row are the v1.1 rows; the decisive row count is the same in both modes. A
    move_turn drop beyond its margin fails in both."""
    head, p2a = _fwd_pair(1.0, 1.06, mt=(0.70, 0.80))
    old = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, RG_T, translate="ratio_drop")
    new = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, RG_T, translate="track_err")
    strip = lambda r: {k: v for k, v in r.items() if k != "worse_by_rule"}  # noqa: E731
    by_old = {(r["statistic"], r["watch"]): r for r in old}
    by_new = {(r["statistic"], r["watch"]): strip(r) for r in new}
    same = [("ratio move_turn", False), ("survival fwd", False), ("facing_closed turn", False),
            ("yaw_progress turn", False), ("|ratio-1| move_turn", True)]
    for k in same:
        assert by_new[k] == by_old[k], k
    assert by_new[("ratio move_turn", False)]["R"] == pytest.approx(0.04)
    assert sum(not r["watch"] for r in old) == sum(not r["watch"] for r in new) == 5
    assert {r["statistic"] for r in new if r["watch"]} == {"ratio fwd", "|ratio-1| move_turn"}
    for mode in ("ratio_drop", "track_err"):
        _rows, j = _judge(T, head, p2a, mode)
        assert any(f.startswith("ratio move_turn suite es1: worse by 0.1000") for f in j["fails"]), mode


def test_track_err_fails_closed(T):
    """A track_err row without a 99 % guard R, or read against a drift map with no track_err key (a pre-v1.2 file),
    FAILS closed; a missing category reads missing (FAIL)."""
    head, p2a = _fwd_pair(1.0, 1.0)
    rows = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, {"cat.move_turn.ratio": 0.04}, translate="track_err")
    j = T.zero_kg_judge(rows, {f"{r['statistic']}|{r['row']}": 0.0 for r in rows}, 2500)
    assert j["verdict"] == "FAIL" and any(f.startswith("track_err fwd") and "guard" in f for f in j["fails"])
    old = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, RG_T, translate="ratio_drop")
    v11_drift = {f"{r['statistic']}|{r['row']}": 0.0 for r in old}
    new = T.zero_kg_suite_rows(head, p2a, ["fwd"], R_T, RG_T, translate="track_err")
    j = T.zero_kg_judge(new, v11_drift, 2500)
    assert j["verdict"] == "FAIL" and j["fails"] == ["track_err fwd suite es1: no recorded drift for this statistic "
                                                     "(thresholds.json)"]
    rows = T.zero_kg_suite_rows({1: _eps("move_turn", [0.8] * 5, 100)}, {1: _eps("move_turn", [0.8] * 5, 100)},
                                ["fwd"], R_T, RG_T, translate="track_err")
    assert next(r for r in rows if r["statistic"] == "track_err fwd")["missing"] is True


def test_threshold_inputs_v1_2_is_v1_1_plus_the_translate_mode():
    """threshold_inputs_v1_2.yaml = v1.1 + zero_kg_translate_statistic: track_err (+ the version and a note citing the
    CHARTER 2026-10-22 decision, and a header comment); every other key is v1.1's."""
    yaml = pytest.importorskip("yaml")
    text = (PL / "threshold_inputs_v1_2.yaml").read_text()
    a = yaml.safe_load((PL / "threshold_inputs_v1_1.yaml").read_text())
    b = yaml.safe_load(text)
    assert text.startswith("# thresholds v1.2 inputs (CHARTER 2026-10-22")
    assert b["version"] == "v1.2" and b["zero_kg_translate_statistic"] == "track_err"
    assert set(b) - set(a) == {"v1_2_created", "v1_2_note", "zero_kg_translate_statistic"} and not set(a) - set(b)
    assert all(a[k] == b[k] for k in a if k != "version")
    assert "CHARTER 2026-10-22" in b["v1_2_note"] and "amend + switch now" in b["v1_2_note"]


def _zl(doc):
    return next(x for x in doc["lines"] if x["id"] == "p3a.capability.zero_kg_paired")


def test_committed_thresholds_v1_2_is_v1_1_plus_the_track_err_line():
    """thresholds_v1_2.json = v1.1 line for line except the paired 0-kg line; the top level differs only in version /
    inputs / the zero_kg_statistic text; every sum is v1.1's (the per-statistic rates are ~1e-19, so the p3a pilot
    rate is unchanged); the self-check is clean; every golden known-bad reads as in v1.1 (the failing ones still fail).
    The step-8 pilot pair (PP_p3a_s18 vs S1 s18, a recorded read, not a gate) reads FAIL under the new rule on
    track_err back only -- its back overshoot GREW (1.087 -> 1.162) -- disclosed by this test."""
    import json as _json
    a = _json.loads((PL / "thresholds_v1_1.json").read_text())
    b = _json.loads((PL / "thresholds_v1_2.json").read_text())
    assert b["version"] == "v1.2" and b["self_check_failures"] == []
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"version", "inputs", "zero_kg_statistic", "lines"}
    assert "track_err" in b["zero_kg_statistic"] and "CHARTER 2026-10-22" in b["zero_kg_statistic"]
    assert [(x["id"], x.get("value_set")) for x in a["lines"]] == [(x["id"], x.get("value_set")) for x in b["lines"]]
    diff = [x["id"] for x, y in zip(a["lines"], b["lines"]) if x != y]
    assert diff == ["p3a.capability.zero_kg_paired"]
    kb = lambda d: [(x["id"], x.get("value_set"), k, v.get("fails"))  # noqa: E731
                    for x in d["lines"] for k, v in sorted((x.get("known_bad") or {}).items())]
    assert kb(a) == kb(b) and sum(1 for *_r, f in kb(b) if f) > 0
    assert a["sums"] == b["sums"]
    za, zb = _zl(a), _zl(b)
    assert zb["translate_statistic"] == "track_err" and "translate_statistic" not in za
    assert zb["known_good_self_check"] is True and zb["rate"] == za["rate"]
    assert {k for k in set(za) | set(zb) if za.get(k) != zb.get(k)} == {
        "rule", "statistics", "watch_statistics", "drift_by_statistic", "rate_by_Np", "pilot_pair",
        "translate_statistic", "translate_statistic_note"}
    for k in ("1500", "2500", "3000"):
        assert zb["rate_by_Np"][k] == pytest.approx(za["rate_by_Np"][k], rel=1e-12)
    assert za["pilot_pair"]["verdict"] == "PASS"
    assert zb["pilot_pair"]["verdict"] == "FAIL"
    assert all(f.startswith("track_err back") for f in zb["pilot_pair"]["fails"])
    assert len(zb["pilot_pair"]["per_statistic"]) == 26 and zb["pilot_pair"]["unjudged"] == []


def test_committed_thresholds_v1_2_drift_is_recomputed_on_the_new_statistic():
    """The new decisive rows and their drift: one ``track_err <c>|suite es<k>`` key per trained translate category and
    env seed, drift = |r_X12 - 1| - |r_X08 - 1| from the same X12 / X08 stamps (equal to v1.1's |ratio-1| WATCH drift,
    the same quantity, and to the recorded medians), SE_diff on |r - 1| (v1.1's WATCH SE, the same bootstrap), R = the
    99 % guard R (v1.1's decisive ratio R). move_turn's rows are v1.1's exactly; the WATCH raw-drop rows are v1.1's
    decisive ratio rows marked WATCH."""
    import json as _json
    za = _zl(_json.loads((PL / "thresholds_v1_1.json").read_text()))
    zb = _zl(_json.loads((PL / "thresholds_v1_2.json").read_text()))
    tt = ["fwd", "side", "back", "diag"]
    da, db = za["drift_by_statistic"], zb["drift_by_statistic"]
    assert {f"track_err {c}|suite es{k}" for c in tt for k in (1, 2)} <= set(db)
    assert not any(k.startswith("|ratio-1| ") and "move_turn" not in k for k in db)
    dec_a = {(s["statistic"], s["row"]): s for s in za["statistics"]}
    wat_a = {(s["statistic"], s["row"]): s for s in za["watch_statistics"]}
    dec_b = {(s["statistic"], s["row"]): s for s in zb["statistics"]}
    wat_b = {(s["statistic"], s["row"]): s for s in zb["watch_statistics"]}
    for c in tt:
        for k in (1, 2):
            row = f"suite es{k}"
            te = dec_b[(f"track_err {c}", row)]
            abs_a, rat_a = wat_a[(f"|ratio-1| {c}", row)], dec_a[(f"ratio {c}", row)]
            assert db[f"track_err {c}|{row}"] == te["drift"] == abs_a["drift"]
            assert te["drift"] == pytest.approx(abs(rat_a["x12"] - 1.0) - abs(rat_a["x08"] - 1.0), abs=1e-15)
            assert te["se_diff"] == abs_a["se_diff"] and te["R"] == rat_a["R"] and "@guard" in te["R_source"]
            assert te["self_check_passes"] is True and te["key"] == "track_err" and te["watch"] is False
            raw = wat_b[(f"ratio {c}", row)]
            assert {kk for kk in set(raw) | set(rat_a) if raw.get(kk) != rat_a.get(kk)} == {"watch", "R_source",
                                                                                           "worse_by_rule"}
            assert db[f"ratio {c}|{row}"] == da[f"ratio {c}|{row}"]
    for k in (1, 2):
        for key, src_b, src_a in ((("ratio move_turn", f"suite es{k}"), dec_b, dec_a),
                                  (("|ratio-1| move_turn", f"suite es{k}"), wat_b, wat_a)):
            assert {kk: v for kk, v in src_b[key].items() if kk != "worse_by_rule"} == src_a[key], key
    for s in za["statistics"]:                          # survival / facing / completion rows: v1.1's
        if not s["statistic"].startswith("ratio "):
            assert {kk: v for kk, v in dec_b[(s["statistic"], s["row"])].items() if kk != "worse_by_rule"} == s


def test_thresholds_v1_2_regenerates_byte_for_byte(T, tmp_path):
    import subprocess
    _recorded_inputs()
    out = tmp_path / "v1_2.json"
    rel = (PL / "threshold_inputs_v1_2.yaml").relative_to(REPO)
    r = subprocess.run([sys.executable, "-I", str(TP), "--inputs", str(rel), "--out", str(out), "--self-check"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "self-check failures 0" in r.stdout
    assert out.read_bytes() == (PL / "thresholds_v1_2.json").read_bytes()
