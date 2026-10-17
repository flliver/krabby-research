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
* the payload a-priori formula and resolution.binomial_guard_floor's values (0.0717 / 0.0507).

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
