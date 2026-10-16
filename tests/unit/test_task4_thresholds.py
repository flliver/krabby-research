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
