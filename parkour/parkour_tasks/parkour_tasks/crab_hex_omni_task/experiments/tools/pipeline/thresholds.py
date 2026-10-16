#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 pipeline thresholds: every DECISIVE gate line, its chance-HALT rate and its self-checks (CPU).

One job: read an input map (``threshold_inputs.yaml``: stage -> seed -> recorded eval stamps, the
known-bad stamps, the stand head reads, the 0-kg drift pair) and write ``thresholds.json`` -- for
every decisive line the statistic (and the JSON key path it is read from), the three recorded seed
values, the line, the per-read chance-HALT rate under the PLAN's model for its class, the
known-bad value(s) with pass / fail, and whether every recorded seed passes. It also writes the
floor proposals (candidate values with their rates) and the run sums. Nothing here boots Kit,
trains or evaluates: it reads saved JSON / npz only and writes one JSON (plus optional markdown
tables). No dated path lives in this file; every path comes from the input map.

READ CLASSES AND THEIR MODELS (PLAN "Gates -> Rule -> Self-checks")
-------------------------------------------------------------------
* ``three_seed``  -- a line built from the three recorded seeds (``min - max(R, range, G)``, mirrored
  for lower-is-better). Rate = the n = 3 predictive probability that a fresh seed falls below the
  line, simulated as a function of ``r = R/sigma_hat`` (:func:`simulate_three_seed_rate`), with
  ``sigma_hat^2 = (range/1.693)^2 + (R/1.96)^2`` (:func:`sigma_hat`). ~6.6 % at r = 0, ~1 % at r = 2.
* ``fixed_t``     -- a fixed line (the lowered floors' ratio lines, the side22 FAST medians, the
  S1-era stand lines): ``P(T_2 < (line - mean) / (sigma_hat * sqrt(1 + 1/3)))`` (:func:`fixed_line_t_rate`).
* ``binomial``    -- survival floors and other proportions: the pooled recorded misses over the three
  seeds give ``p``; rate = ``P(Bin(n_read, p) > n_read * (1 - floor))`` (:func:`binomial_floor_rate`).
  Sensitivity (reported, not summed): the Jeffreys beta-binomial predictive (:func:`betabinom_floor_rate`).
* ``poisson``     -- counts: ``lambda`` = the mean recorded count; rate = ``P(Pois(lambda) > line)``
  (:func:`poisson_count_rate`). Sensitivity: the Jeffreys gamma-Poisson predictive (:func:`nb_count_rate`).
* ``paired``      -- the 0-kg check and the student completion: ``Phi(-margin / SE_diff)``
  (:func:`paired_rate`); the student's HALT needs two failed blocks -> ``p_block^2`` (lower bound).
* ``payload_apriori`` -- pending step 8 (no pilot yet): formula ``Phi((line - mu_pilot) /
  sqrt(2 sigma_hat_p2a^2 + SE^2))`` (:func:`payload_apriori_rate`); only illustrated here.
* ``live_kill`` / the 0-kg drift margin -- ``unvalidated``, excluded from every sum.

Usage::

    thresholds.py --inputs <PL>/threshold_inputs.yaml --out <PL>/thresholds_v0.json [--md TABLES.md]
                  [--self-check] [--sims N] [--seed S]

``--self-check`` exits 1 if a recorded seed fails a decisive line at its RECOMMENDED value, or a
decisive line has neither a failing known-bad nor the ``unvalidated`` tag. Rates are reported, not
enforced (PLAN: HALTs are prompts).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.dont_write_bytecode = True

SCHEMA = 1
DEFAULT_SEED = 20261016
RANGE_D2_N3 = 1.693          # E[range] / sigma for n = 3 normal draws
Z95 = 1.96
GUARD_LEVEL = 0.99           # resolution.binomial_guard_floor
BINOM_P_CLIP = (0.05, 0.95)

SURVIVAL_CANDIDATES = (0.90, 0.80, 0.70, 0.60)
RATIO_CANDIDATES = (0.70, 0.60, 0.50, 0.40)
STANDARD_FLOORS = {"survival": 0.90, "ratio": 0.70}
SIDE22_FAST = {"m22": 0.155, "m25": 0.175}      # side22_classify.GATE["fast"]
SIDE22_SLOW = {"m22": 0.13, "m25": 0.145}       # side22_classify.GATE["slow"]


# ============================================================================ models (pure)
def sigma_hat(values, R: float) -> float:
    """``sqrt((range/1.693)^2 + (R/1.96)^2)`` over the recorded seed values (PLAN Rule)."""
    v = np.asarray(values, float)
    rng = float(v.max() - v.min()) if v.size else 0.0
    R = 0.0 if R is None else float(R)
    return math.sqrt((rng / RANGE_D2_N3) ** 2 + (R / Z95) ** 2)


def simulate_three_seed_rate(r: float, nsim: int = 400_000, seed: int = DEFAULT_SEED) -> float:
    """P(fresh seed < min - max(r, range)) for three recorded N(0, 1) seeds (sigma = 1, R = r)."""
    g = np.random.default_rng(seed)
    x = g.standard_normal((nsim, 3))
    f = g.standard_normal(nsim)
    line = x.min(1) - np.maximum(float(r), x.max(1) - x.min(1))
    return float((f < line).mean())


class ThreeSeedTable:
    """:func:`simulate_three_seed_rate` on a grid of r, interpolated (deterministic for a seed)."""

    def __init__(self, nsim: int = 200_000, seed: int = DEFAULT_SEED, r_max: float = 4.0, step: float = 0.05):
        self.r = np.round(np.arange(0.0, r_max + 1e-9, step), 6)
        self.p = np.array([simulate_three_seed_rate(r, nsim, seed + i) for i, r in enumerate(self.r)])
        self.nsim, self.seed = nsim, seed

    def __call__(self, r: float) -> float:
        return float(np.interp(min(max(float(r), 0.0), float(self.r[-1])), self.r, self.p))


def three_seed_line(values, R: float = 0.0, G: float = 0.0, higher_is_better: bool = True) -> float:
    """``min(x) - max(R, range, G)`` (higher is better) or ``max(x) + max(R, range, G)``."""
    v = np.asarray(values, float)
    m = max(float(R or 0.0), float(v.max() - v.min()), float(G or 0.0))
    return float(v.min() - m) if higher_is_better else float(v.max() + m)


def tracking_line(ratios, R: float = 0.0) -> float:
    """``|r - 1| <= max|x - 1| + max(R, range|x - 1|)`` -- the mirrored tracking line (on |r - 1|)."""
    e = np.abs(np.asarray(ratios, float) - 1.0)
    return float(e.max() + max(float(R or 0.0), float(e.max() - e.min())))


def three_seed_rate(values, R: float, table: ThreeSeedTable) -> dict:
    """Rate of a three-seed line: the table at ``r = R / sigma_hat``."""
    s = sigma_hat(values, R)
    r = (float(R) / s) if s > 0 else float(table.r[-1])
    return {"rate": table(r), "r_over_sigma": r, "sigma_hat": s}


def fixed_line_t_rate(values, R: float, line: float, higher_is_better: bool = True) -> dict:
    """n = 3 predictive t (df 2) against a fixed line. sigma_hat == 0 -> degenerate (0 or 1)."""
    v = np.asarray(values, float)
    n = v.size
    s = sigma_hat(v, R)
    mean = float(v.mean())
    if s == 0.0:
        ok = (mean >= line) if higher_is_better else (mean <= line)
        return {"rate": 0.0 if ok else 1.0, "t": None, "sigma_hat": 0.0, "degenerate": True}
    t = ((line - mean) if higher_is_better else (mean - line)) / (s * math.sqrt(1.0 + 1.0 / n))
    return {"rate": float(sps.t.cdf(t, df=n - 1)), "t": float(-t), "sigma_hat": s, "degenerate": False}


def max_ok_misses(n: int, floor: float) -> int:
    """Largest miss count with ``(n - k) / n >= floor`` (float-safe)."""
    return int(math.floor((1.0 - float(floor)) * int(n) + 1e-9))


def binomial_floor_rate(misses_pooled: int, n_pooled: int, n_read: int, floor: float) -> float:
    """P(survival < floor) for a fresh read of n_read at the pooled recorded miss rate."""
    if n_pooled <= 0:
        return float("nan")
    p = float(misses_pooled) / float(n_pooled)
    return float(sps.binom.sf(max_ok_misses(n_read, floor), int(n_read), p))


def betabinom_floor_rate(misses_pooled: int, n_pooled: int, n_read: int, floor: float) -> float:
    """Jeffreys sensitivity: p ~ Beta(k + 0.5, n - k + 0.5), predictive beta-binomial."""
    a, b = misses_pooled + 0.5, n_pooled - misses_pooled + 0.5
    return float(sps.betabinom.sf(max_ok_misses(n_read, floor), int(n_read), a, b))


def poisson_count_rate(counts, line: float) -> float:
    """P(Pois(mean(counts)) > line); pass means count <= line."""
    lam = float(np.mean(np.asarray(counts, float)))
    return float(sps.poisson.sf(math.floor(float(line) + 1e-9), lam)) if lam > 0 else 0.0


def nb_count_rate(counts, line: float) -> float:
    """Jeffreys sensitivity: lambda ~ Gamma(sum + 0.5, rate = len), predictive negative binomial."""
    c = np.asarray(counts, float)
    a, m = float(c.sum()) + 0.5, float(c.size)
    return float(sps.nbinom.sf(math.floor(float(line) + 1e-9), a, m / (m + 1.0)))


def paired_rate(margin: float, se_diff: float) -> float:
    """Phi(-margin / SE_diff): a chance drop beyond the margin with no true difference."""
    if se_diff is None or not se_diff > 0:
        return 0.0
    return float(sps.norm.cdf(-float(margin) / float(se_diff)))


def student_halt_rate(p_block: float) -> float:
    """Two failed full blocks HALT the student branch: p_block^2 (lower bound: no distillation-seed term)."""
    return float(p_block) ** 2


def zero_kg_margin(R: float, se_diff: float, drift: float, its_past_p2a: float, ref_its: float = 5000.0) -> dict:
    """max(R, 3 SE_diff, |drift| * its_past_p2a / ref_its) and which term binds."""
    terms = {"R": float(R or 0.0), "3SE": 3.0 * float(se_diff or 0.0),
             "drift_scaled": abs(float(drift or 0.0)) * float(its_past_p2a) / float(ref_its)}
    k = max(terms, key=terms.get)
    return {"margin": terms[k], "binds": k, **terms}


def payload_apriori_rate(line: float, mu_pilot: float, sigma_p2a: float, se: float,
                         higher_is_better: bool = True) -> float:
    """Phi((line - mu) / sqrt(2 sigma_hat_p2a^2 + SE^2)) (mirrored for lower-is-better)."""
    s = math.sqrt(2.0 * float(sigma_p2a) ** 2 + float(se) ** 2)
    if s == 0:
        return 0.0 if ((mu_pilot >= line) if higher_is_better else (mu_pilot <= line)) else 1.0
    z = (line - mu_pilot) / s if higher_is_better else (mu_pilot - line) / s
    return float(sps.norm.cdf(z))


def binomial_guard_floor(p_bar: float, n: int, level: float = GUARD_LEVEL) -> float:
    """resolution.binomial_guard_floor: z_level * sqrt(2 p (1 - p) / n), p clipped to [0.05, 0.95]."""
    p = min(BINOM_P_CLIP[1], max(BINOM_P_CLIP[0], float(p_bar)))
    return float(sps.norm.ppf(level) * math.sqrt(2.0 * p * (1.0 - p) / float(n)))


def union(rates) -> float:
    r = [float(x) for x in rates if x is not None and np.isfinite(x)]
    return float(1.0 - np.prod([1.0 - x for x in r])) if r else 0.0


def total(rates) -> float:
    return float(sum(float(x) for x in rates if x is not None and np.isfinite(x)))


# ============================================================================ readers
def repo_root() -> Path:
    p = Path(__file__).resolve()
    for q in p.parents:
        if (q / "parkour").is_dir() and (q / ".git").exists():
            return q
    raise RuntimeError("repo root (a dir with parkour/ and .git) not found above " + str(p))


class Repo:
    def __init__(self, root: Path):
        self.root = root

    def p(self, rel: str) -> Path:
        return self.root / rel

    def stamp(self, entry: dict) -> Path:
        d = self.p(entry["stamp"])
        for f in ("scenario_metrics.json", "run_meta.json", "summary.md"):
            if not (d / f).exists():
                raise FileNotFoundError(f"incomplete stamp {d} (no {f})")
        cs = d / "code_state.json"
        got = json.loads(cs.read_text())["eval_sha"][:8] if cs.exists() else "none"
        if got != entry.get("eval_sha"):
            raise RuntimeError(f"{d}: eval_sha {got} != input map {entry.get('eval_sha')}")
        newer = sorted(x for x in glob.glob(str(d.parent / "*")) if Path(x).is_dir() and Path(x).name > d.name)
        if newer:
            raise RuntimeError(f"{d}: a newer stamp exists in the same slot ({newer[-1]}); regenerate the input map")
        return d


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def suite_categories(stamp: Path) -> dict:
    """Per category of a suite stamp: n, survival (n-weighted own-hold), misses, and the medians the
    harness stores (``aggregate.suite.by_category[c].<key>.median``) -- run_phases.cat_summary's reading."""
    a = load_json(stamp / "scenario_metrics.json")["aggregate"]
    s = a["suite"]
    out = {}
    for c, b in s["by_category"].items():
        holds = [h for h in s["hold_of_wave"] if s["category_of_hold"].get(h) == c]
        n = sum(int(s["n_by_hold"][h]) for h in holds)
        surv = sum(float(s["survival_by_hold"][h]) * int(s["n_by_hold"][h]) for h in holds) / n if n else None
        med = lambda k: ((b or {}).get(k) or {}).get("median")  # noqa: E731
        out[c] = {"n": n, "survival": surv, "misses": (int(round(n - surv * n)) if n else None),
                  "ratio": med("ratio"), "facing_closed": med("facing_closed"), "yaw_progress": med("yaw_progress")}
    out["_orientation"] = {k: (a.get("orientation", {}).get(k) or {}).get("median") for k in ("roll_rms", "pitch_rms")}
    out["_trained_categories"] = s.get("trained_categories")
    return out


def field_completion(stamps: list) -> dict:
    """``aggregate.schedule_completion_rate`` pooled over env seeds (n-weighted), as p2_read pools it."""
    num = den = 0.0
    per = []
    for d in stamps:
        a = load_json(d / "scenario_metrics.json")["aggregate"]
        n = int(a["n_episodes"])
        num += float(a["schedule_completion_rate"]) * n
        den += n
        per.append(float(a["schedule_completion_rate"]))
    return {"rate": num / den, "n": int(den), "per_env_seed": per}


def field_episodes(d: Path) -> dict:
    sm = load_json(d / "scenario_metrics.json")
    return {int(e["env_index"]): bool(e["termination_reason"] == "schedule_complete") for e in sm["episodes"]}


def load_side22_module(path: Path):
    spec = importlib.util.spec_from_file_location("side22_classify_t4", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def side22_read(mod, stamp: Path, B: int = 2000, seed: int = DEFAULT_SEED) -> dict:
    """side22_classify.read_stamp medians per lat_* label, their episode-bootstrap SE, and the class."""
    _meta, rows, excl = mod.read_stamp(stamp)
    g = np.random.default_rng(seed)
    med, se, n = {}, {}, {}
    for lab, rs in rows.items():
        v = np.array([r["feet_ss"] for r in rs], float)
        med[lab] = float(np.median(v))
        se[lab] = float(np.median(v[g.integers(0, len(v), (B, len(v)))], axis=1).std())
        n[lab] = int(len(v))
    cls = [mod.classify(med["lat_l_022"], med["lat_l_025"], mod.GATE),
           mod.classify(med["lat_r_022"], med["lat_r_025"], mod.GATE)]
    return {"median": med, "se": se, "n": n, "class": cls, "excluded": excl}


def h_meas_and_orientation(stamp: Path) -> dict:
    """Median over episodes of each episode's median h_meas (root z - mean terrain z under the six
    feet, sh_kin.h_meas) over alive steps after the prologue segment; plus the stamp's orientation medians.
    PROPOSED definition for payload_read.py (step 6); the harness stores no h_meas."""
    meds = []
    for f in sorted(glob.glob(str(stamp / "raw" / "episode_*.npz"))):
        z = np.load(f)
        seg, done = z["segment_id"], z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(seg)
        h = z["root_pos_w"][:S, 2].astype(float) - z["terrain_z"][:S].astype(float).mean(1)
        m = seg[:S] > 0
        if m.any():
            meds.append(float(np.median(h[m])))
    o = load_json(stamp / "scenario_metrics.json")["aggregate"].get("orientation", {})
    return {"h_meas_median": float(np.median(meds)), "h_meas_p10": float(np.percentile(meds, 10)),
            "roll_rms": (o.get("roll_rms") or {}).get("median"), "pitch_rms": (o.get("pitch_rms") or {}).get("median")}


def suite_episodes(stamp: Path) -> list:
    """Per episode of a suite stamp: within-wave env, category, own-hold survival (harness rule:
    n_steps >= round((prologue + settle + hold_s) / dt)), ratio, facing_closed, yaw_progress_rate."""
    sm = load_json(stamp / "scenario_metrics.json")
    s = sm["aggregate"]["suite"]
    meta = load_json(stamp / "run_meta.json")
    nsteps = {int(e["env_index"]): int(e["n_steps"]) for e in sm["episodes"]}
    dt = float(s["dt"])
    thr = {h: round((float(s["prologue_s"]) + float(s["settle_hold_s"]) + float(s["hold_s"][h])) / dt) for h in s["hold_s"]}
    out = []
    for m in meta["env_index_map"]:
        ei, lab = int(m["env_index"]), m["label"]
        ep = load_json(stamp / "metrics" / f"episode_{ei:02d}.json")
        h = (ep.get("holds") or {}).get(lab) or {}
        fa = h.get("facing") or {}
        out.append({"env": int(m["env"]), "cat": m["category"], "surv": nsteps[ei] >= thr[lab],
                    "ratio": (((h.get("tracking") or {}).get("vx") or {}).get("ratio")),
                    "facing_closed": fa.get("facing_closed"), "yaw_progress": fa.get("yaw_progress_rate")})
    return out


def _cat_stat(rows, key, absdev=False):
    if key == "survival":
        v = [r["surv"] for r in rows]
        return float(np.mean(v)) if v else None
    v = [r[key] for r in rows if r[key] is not None]
    if not v:
        return None
    m = float(np.median(v))
    return abs(m - 1.0) if absdev else m


def paired_suite_se(A: list, B: list, cat: str, key: str, absdev: bool = False, n_boot: int = 1000,
                    seed: int = DEFAULT_SEED) -> float:
    """Env-clustered paired bootstrap SE of (A - B) for one category statistic (same env ids both heads)."""
    ia, ib = {}, {}
    for r in A:
        if r["cat"] == cat:
            ia.setdefault(r["env"], []).append(r)
    for r in B:
        if r["cat"] == cat:
            ib.setdefault(r["env"], []).append(r)
    envs = np.array(sorted(set(ia) & set(ib)))
    g = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        pick = g.choice(envs, len(envs))
        a = _cat_stat([x for e in pick for x in ia[e]], key, absdev)
        b = _cat_stat([x for e in pick for x in ib[e]], key, absdev)
        if a is not None and b is not None:
            d.append(a - b)
    return float(np.std(d)) if d else 0.0


def paired_completion_se(a: dict, b: dict, n_boot: int = 4000, seed: int = DEFAULT_SEED) -> float:
    k = sorted(set(a) & set(b))
    x = np.array([a[i] for i in k], float)
    y = np.array([b[i] for i in k], float)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(k), (n_boot, len(k)))
    return float((x[idx].mean(1) - y[idx].mean(1)).std())


def resolution_R(paths: list) -> dict:
    """metric -> max R_eval over the committed resolution files (``metrics.<id>.R_eval``)."""
    R = {}
    for p in paths:
        for k, v in load_json(p)["metrics"].items():
            if "@" in k or not isinstance(v, dict) or v.get("R_eval") is None:
                continue
            R[k] = max(R.get(k, 0.0), float(v["R_eval"]))
    return R


# ============================================================================ builders
def _stage_suite(repo: Repo, sd: dict) -> dict:
    """{seed: {es: categories}} plus eval_sha / stamp records."""
    out, rec = {}, []
    for seed, per_es in sd["suite"].items():
        for es, ent in per_es.items():
            d = repo.stamp(ent)
            out.setdefault(int(seed), {})[int(es)] = suite_categories(d)
            rec.append({"seed": int(seed), "env_seed": int(es), "stamp": ent["stamp"], "eval_sha": ent["eval_sha"]})
    return out, rec


def floor_lines(stage: str, sd: dict, cats: dict, kb: dict, R: dict, sur_floor: float, rat_floor: float,
                t_sens_R: dict | None = None) -> list:
    """The survival and ratio floor reads of one stage (one read per trained translate category per env seed)."""
    lines = []
    for es in sd["floor_read_env_seeds"]:
        for c in sd["trained_translate"]:
            seeds = sorted(cats)
            vals = [cats[s][es][c] for s in seeds]
            n_read = vals[0]["n"]
            misses = sum(v["misses"] for v in vals)
            n_pool = sum(v["n"] for v in vals)
            sv = [v["survival"] for v in vals]
            kbv = {k: kb[k]["suite"][c]["survival"] for k in kb}
            lines.append({
                "id": f"{stage}.floor.survival.{c}.es{es}", "stage": stage, "kind": "floor", "class": "binomial",
                "statistic": f"translate survival, {c}", "key_path": f"aggregate.suite.survival_by_hold x n_by_hold over category_of_hold == '{c}' (run_phases.cat_summary 'survival')",
                "env_seed": es, "seed_values": dict(zip(seeds, sv)), "n_read": n_read, "misses_pooled": misses, "n_pooled": n_pool,
                "line": sur_floor, "direction": ">=",
                "rate": binomial_floor_rate(misses, n_pool, n_read, sur_floor),
                "rate_sensitivity": {"betabinom_jeffreys": betabinom_floor_rate(misses, n_pool, n_read, sur_floor),
                                     **({"t_model_with_guard_R": fixed_line_t_rate(sv, t_sens_R.get(c, 0.0), sur_floor)["rate"]} if t_sens_R else {})},
                "seeds_pass": all(x >= sur_floor - 1e-12 for x in sv),
                "known_bad": {k: {"value": v, "fails": v < sur_floor - 1e-12} for k, v in kbv.items()},
            })
            rv = [v["ratio"] for v in vals]
            Rr = R.get(f"cat.{c}.ratio", 0.0)
            fr = fixed_line_t_rate(rv, Rr, rat_floor)
            kbr = {k: kb[k]["suite"][c]["ratio"] for k in kb}
            lines.append({
                "id": f"{stage}.floor.ratio.{c}.es{es}", "stage": stage, "kind": "floor", "class": "fixed_t",
                "statistic": f"translate tracking ratio, {c}", "key_path": f"aggregate.suite.by_category.{c}.ratio.median",
                "env_seed": es, "seed_values": dict(zip(seeds, rv)), "R": Rr, "sigma_hat": fr["sigma_hat"], "t": fr["t"],
                "line": rat_floor, "direction": ">=", "rate": fr["rate"],
                "seeds_pass": all(x >= rat_floor for x in rv),
                "known_bad": {k: {"value": v, "fails": v < rat_floor} for k, v in kbr.items()},
            })
    return lines


def side22_line(stage: str, reads: dict, kb_reads: dict, relaxed: bool = False, proxy_of: str | None = None) -> dict:
    """FAST/FAST (coded) or not-SLOW/not-DEAD (relaxed) on side22_25lr, both hands, fixed_t per median."""
    labs = ("lat_l_022", "lat_l_025", "lat_r_022", "lat_r_025")
    thr = SIDE22_SLOW if relaxed else SIDE22_FAST
    per = {}
    for lab in labs:
        line = thr["m22"] if lab.endswith("022") else thr["m25"]
        vals = [reads[s]["median"][lab] for s in sorted(reads)]
        R = 2.0 * max(reads[s]["se"][lab] for s in reads)
        per[lab] = {"seed_values": dict(zip(sorted(reads), vals)), "R_2SE": R, "line": line,
                    **fixed_line_t_rate(vals, R, line)}
    if relaxed:   # fails only when BOTH medians of a hand are at/below SLOW: P <= min(p22, p25) per hand (upper bound)
        hand = [min(per[f"lat_{h}_022"]["rate"], per[f"lat_{h}_025"]["rate"]) for h in ("l", "r")]
    else:
        hand = [union([per[f"lat_{h}_022"]["rate"], per[f"lat_{h}_025"]["rate"]]) for h in ("l", "r")]
    fail_cls = ({"SLOW", "DEAD", "DEAD?(n=0/NaN)"} if relaxed else None)
    ok = lambda cl: (all(c not in fail_cls for c in cl) if relaxed else all(c == "FAST" for c in cl))  # noqa: E731
    return {
        "id": f"{stage}.floor.side22" + ("_notslow" if relaxed else "_fastfast"), "stage": stage, "kind": "floor",
        "class": "fixed_t", "statistic": "side22_25lr feet_ss medians, both hands" + (" (relaxed: fail on SLOW/DEAD)" if relaxed else " (FAST/FAST)"),
        "key_path": "side22_classify.read_stamp -> median feet_ss per lat_{l,r}_{022,025} (raw npz; the classifier the C0 SLOW reads came from)",
        "lines": {k: v["line"] for k, v in per.items()}, "per_median": per, "rate": union(hand),
        "rate_note": ("per hand min(p22, p25) (an upper bound on P(both <= SLOW)); hands unioned" if relaxed
                      else "per hand union(p22, p25); hands unioned (independence approximation)"),
        "seeds_class": {s: reads[s]["class"] for s in sorted(reads)},
        "seeds_pass": all(ok(reads[s]["class"]) for s in reads),
        "known_bad": {k: {"class": v["class"], "fails": not ok(v["class"])} for k, v in kb_reads.items()},
        **({"proxy_of": proxy_of} if proxy_of else {}),
    }


def stand_clause_class(gate: str, clause: str) -> str:
    c = clause.lower()
    if any(w in c for w in ("falls (count)", "stand falls es", "exit-handover falls", "onset stalls", "exit falls",
                            "miss es", "stand falls (count)")):
        return "poisson"
    if "falls/100" in c:
        return "poisson"
    if "stand-fall rate" in c:
        return "poisson"
    if any(w in c for w in ("homed", "settled", "stood")) and "settle walkstop" not in c and "median settle" not in c:
        return "binomial"
    if "completion" in c:
        return "fixed_t_relative"
    return "fixed_t"


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and np.isfinite(x) else None


def stand_clause_rate(gate: str, clause: str, vals: list, line: float, lines: list, note: str | None = None,
                      override_line: float | None = None) -> dict:
    """Rate of one stand clause (one env-seed read) from the three recorded seed values."""
    cls = stand_clause_class(gate, clause)
    L = line if override_line is None else override_line
    low_better = not ((">=" in clause) or clause.startswith(("homed", "settled")))
    out = {"class": cls, "line": L, "direction": "<=" if low_better else ">="}
    if cls == "poisson":
        if "falls/100" in clause:              # K3 falls per 100 of 600 starts -> counts per 600
            counts = [round(v * 6) for v in vals]
            Lc = math.floor(L * 6 + 1e-9)
        elif "stand-fall rate" in clause:      # head-relative SE line: passes while k <= 3 of 100 / 400
            nn = 400 if "field_hard" in clause else 100
            counts = [round(v * nn) for v in vals]
            Lc = 3
            out["line_note"] = "k/n <= HOLD(0) + 2 SE(k/n) passes while k <= 3 (n = 100 or 400)"
        else:
            counts = [int(v) for v in vals]
            Lc = L
        out.update({"counts": counts, "line_count": Lc, "rate": poisson_count_rate(counts, Lc),
                    "rate_sensitivity": {"nb_jeffreys": nb_count_rate(counts, Lc)}})
        return out
    if cls == "binomial":
        nn = 600 if "STOOD" in clause else 100 if "share" in clause else 300
        if note and "n " in note:
            try:
                nn = int(note.split("n ")[1].split()[0])
            except Exception:  # noqa: BLE001
                pass
        misses = [int(round((1.0 - v) * nn)) for v in vals]
        out.update({"n_read": nn, "misses": misses, "rate": binomial_floor_rate(sum(misses), nn * len(vals), nn, L),
                    "rate_sensitivity": {"betabinom_jeffreys": betabinom_floor_rate(sum(misses), nn * len(vals), nn, L)}})
        return out
    R = 0.0
    if cls == "fixed_t_relative":   # completion vs REFS(T*) - max(0.05, 2SE): R = 2 binomial SE of the read
        nn = 400 if "field_hard" in clause else 100
        p = float(np.mean(vals))
        R = 2.0 * math.sqrt(max(p * (1 - p), 1e-12) / nn)
        out["R_note"] = f"R = 2 sqrt(p(1-p)/{nn}) (no committed resolution entry); line = the recorded per-head line"
    else:
        out["R_note"] = "no committed resolution entry for this statistic: R = 0 (sigma_hat from the seed range only)"
    fr = fixed_line_t_rate(vals, R, L, higher_is_better=not low_better)
    out.update({"R": R, **fr})
    return out


def stand_lines(repo: Repo, sdef: dict, overrides: dict | None = None) -> list:
    """G1-G5 (SN_Amask s16/17/18) and K1-K6 (SH_T16/17/18, K5 via the SH zero-command rows)."""
    overrides = overrides or {}
    out = []
    sn = {int(s): load_json(repo.p(p)) for s, p in sdef["sn_heads"].items()}
    extra = {k: load_json(repo.p(p)) for k, p in (sdef.get("sn_extra_known_bad") or {}).items()}
    seeds = sorted(sn)
    rows = {s: sn[s]["rows"][f"SN_Amask_s{s}"] for s in seeds}
    kb_rows = {k: v for k, v in sn[seeds[0]]["rows"].items() if not k.startswith("SN_")}
    for name, j in extra.items():
        kb_rows[name] = next(v for k, v in j["rows"].items() if k.startswith("SN_"))
    known_good = {"HOLD", "R"}
    for g in ("G1", "G2", "G3", "G4", "G5"):
        for cl in rows[seeds[0]]["gates"][g]["clauses"]:
            cs = [rows[s]["gates"][g]["clauses"][cl] for s in seeds]
            vals = [c["value"] for c in cs]
            if not all(_num(v) is not None for v in vals):
                continue
            ov = overrides.get(f"SN.{g}.{cl}")
            r = stand_clause_rate(g, cl, vals, cs[0]["line"], [c["line"] for c in cs], cs[0].get("note"), ov)
            kb = {}
            for k, row in kb_rows.items():
                c = ((row.get("gates") or {}).get(g) or {}).get("clauses", {}).get(cl)
                if c is None or k in known_good:
                    continue
                v = c.get("value")
                if ov is not None and _num(v) is not None:
                    fails = (v > ov) if r["direction"] == "<=" else (v < ov)
                else:
                    fails = not c.get("ok", True)
                kb[k] = {"value": v, "fails": bool(fails), **({"fails_closed": True} if v is None else {})}
            passes = [((v <= r["line"] + 1e-12) if r["direction"] == "<=" else (v >= r["line"] - 1e-12)) for v in vals]
            if ov is None:
                passes = [bool(c.get("ok")) for c in cs]
            out.append({"id": f"stand.D1.{g}.{cl}", "stage": "D1 (SN)", "kind": "stand", "gate": g, "clause": cl,
                        "statistic": rows[seeds[0]]["gates"][g].get("statistic"), "seed_values": dict(zip(seeds, vals)),
                        "recorded_lines": dict(zip(seeds, [c["line"] for c in cs])), **r,
                        "seeds_pass": all(passes), "known_bad": kb,
                        **({"override_line": ov} if ov is not None else {})})
    sh = {int(s): load_json(repo.p(p)) for s, p in sdef["sh_heads"].items()}
    seeds = sorted(sh)
    for K in ("K1", "K2", "K3", "K4", "K6"):
        g0 = sh[seeds[0]]["gates"][K]
        for cl in g0[f"SH_T{seeds[0]}"]["clauses"]:
            cs = [sh[s]["gates"][K][f"SH_T{s}"]["clauses"][cl] for s in seeds]
            vals = [c["value"] for c in cs]
            if not all(_num(v) is not None for v in vals):
                continue
            ov = overrides.get(f"SH.{K}.{cl}")
            r = stand_clause_rate(K, cl, vals, cs[0]["line"], [c["line"] for c in cs], cs[0].get("note"), ov)
            kb = {}
            for role, row in g0.items():
                if role.startswith("SH_") or role in ("R_h",):
                    continue
                c = row["clauses"].get(cl)
                if c is None:
                    continue
                v = c.get("value")
                if ov is not None and _num(v) is not None:
                    fails = (v > ov) if r["direction"] == "<=" else (v < ov)
                else:
                    fails = not c.get("ok", True)
                kb[role] = {"value": v, "fails": bool(fails)}
            if ov is not None:
                passes = [((v <= ov + 1e-12) if r["direction"] == "<=" else (v >= ov - 1e-12)) for v in vals]
            else:
                passes = [bool(c.get("ok")) for c in cs]
            out.append({"id": f"stand.D2.{K}.{cl}", "stage": "D2 (SH)", "kind": "stand", "gate": K, "clause": cl,
                        "seed_values": dict(zip(seeds, vals)), "recorded_lines": dict(zip(seeds, [c["line"] for c in cs])),
                        **r, "seeds_pass": all(passes), "known_bad": kb, **({"override_line": ov} if ov is not None else {})})
    for g, cls_ in sdef["k5_sn_calib"]["gates"].items():
        for cl, c in cls_.items():
            vals = [c["values"][s] if s in c["values"] else c["values"][str(s)] for s in seeds]
            line = list(c["lines"].values())[0]
            if not all(_num(v) is not None for v in vals) or _num(line) is None:
                continue
            ov = overrides.get(f"SN.{g}.{cl}")
            r = stand_clause_rate(g, cl, vals, line, list(c["lines"].values()), None, ov)
            L = r["line"]
            passes = [((v <= L + 1e-12) if r["direction"] == "<=" else (v >= L - 1e-12)) for v in vals]
            if r["class"] == "poisson":
                passes = [k <= r["line_count"] for k in r["counts"]]
            out.append({"id": f"stand.D2.K5.{g}.{cl}", "stage": "D2 (SH)", "kind": "stand", "gate": f"K5/{g}", "clause": cl,
                        "seed_values": dict(zip(seeds, vals)), **r, "seeds_pass": all(passes),
                        "known_bad": {}, "known_bad_note": "K5 has no failing known-bad (A-mask s17 is a second known-good)"})
    return out


# ============================================================================ main assembly
def build(inputs: dict, repo: Repo, table: ThreeSeedTable, rec_floors: dict, stand_overrides: dict,
          stand_standard_overrides: dict | None = None) -> dict:
    R = resolution_R([repo.p(p) for p in inputs["resolution_files"]])
    s22 = load_side22_module(repo.p(inputs["side22_classifier"]))
    stages = inputs["stages"]
    provenance = []
    data = {}
    for st, sd in stages.items():
        cats, rec = _stage_suite(repo, sd)
        provenance += [{"stage": st, "row": sd["suite_row"], **x} for x in rec]
        kb = {}
        for k, v in (sd.get("known_bad") or {}).items():
            kb[k] = {"suite": suite_categories(repo.stamp(v["suite"]))}
            provenance.append({"stage": st, "known_bad": k, "row": "suite__omni", **v["suite"]})
            if v.get("side22"):
                kb[k]["side22"] = side22_read(s22, repo.stamp(v["side22"]))
                provenance.append({"stage": st, "known_bad": k, "row": "side22_25lr__omni", **v["side22"]})
        side = None
        if sd.get("side22"):
            side = {int(s): side22_read(s22, repo.stamp(e)) for s, e in sd["side22"].items()}
            provenance += [{"stage": st, "row": "side22_25lr__omni", "seed": int(s), **e} for s, e in sd["side22"].items()]
        data[st] = {"cats": cats, "kb": kb, "side": side}
    # equality checks (567d1704 vs 1d021625 re-reads)
    eq = []
    for c in stages.get("p2a", {}).get("equality_checks", []):
        a = load_json(repo.stamp(c["a"]) / "scenario_metrics.json")["aggregate"]
        b = load_json(repo.stamp(c["b"]) / "scenario_metrics.json")["aggregate"]
        for k in ("suite",):
            if k in a and isinstance(a[k], dict):
                a[k] = {kk: vv for kk, vv in a[k].items() if kk != "trained_categories"}
                b[k] = {kk: vv for kk, vv in b[k].items() if kk != "trained_categories"}
        eq.append({"a": c["a"], "b": c["b"], "aggregate_equal": a == b})

    surR = {c: R.get(f"cat.{c}.survival", 0.0) for c in ("fwd", "side", "back", "diag")}
    walker_stages = ["s1", "c4", "c5", "c6", "c6x"]
    out_lines = []
    proposals = {}
    # ---------------------------------------------------------------- floors per stage (standard + candidates)
    for st in walker_stages + ["p2a"]:
        sd, D = stages[st], data[st]
        std = floor_lines(st, sd, D["cats"], D["kb"], R, STANDARD_FLOORS["survival"], STANDARD_FLOORS["ratio"], surR)
        cand = {"survival": {}, "ratio": {}}
        for f in SURVIVAL_CANDIDATES:
            ls = [x for x in floor_lines(st, sd, D["cats"], D["kb"], R, f, 0.70, surR) if ".survival." in x["id"]]
            cand["survival"][f] = {"sum": total(x["rate"] for x in ls),
                                   "sum_t_model_sensitivity": total(x["rate_sensitivity"]["t_model_with_guard_R"] for x in ls),
                                   "sum_betabinom_sensitivity": total(x["rate_sensitivity"]["betabinom_jeffreys"] for x in ls),
                                   "per_line": {x["id"]: x["rate"] for x in ls},
                                   "seeds_pass": all(x["seeds_pass"] for x in ls),
                                   "known_bad_fails": sorted({k for x in ls for k, v in x["known_bad"].items() if v["fails"]})}
        for f in RATIO_CANDIDATES:
            ls = [x for x in floor_lines(st, sd, D["cats"], D["kb"], R, 0.90, f, surR) if ".ratio." in x["id"]]
            cand["ratio"][f] = {"sum": total(x["rate"] for x in ls), "per_line": {x["id"]: x["rate"] for x in ls},
                                "seeds_pass": all(x["seeds_pass"] for x in ls),
                                "known_bad_fails": sorted({k for x in ls for k, v in x["known_bad"].items() if v["fails"]})}
        side_std = side_rel = None
        if D["side"]:
            kbs = {k: v["side22"] for k, v in D["kb"].items() if "side22" in v}
            side_std = side22_line(st, D["side"], kbs)
            side_rel = side22_line(st, D["side"], kbs, relaxed=True)
        proposals[st] = {"candidates": cand, "side22": {"fastfast": side_std, "notslow": side_rel}}
        rf = rec_floors.get(st, STANDARD_FLOORS | {"side22": "fastfast"})
        rec = floor_lines(st, sd, D["cats"], D["kb"], R, rf["survival"], rf["ratio"], surR)
        for x in std:
            x["value_set"] = "standard"
        for x in rec:
            x["value_set"] = "recommended"
        out_lines += std + rec
        for which, ln in (("standard", side_std), ("recommended", side_std if rf.get("side22", "fastfast") == "fastfast" else side_rel)):
            if ln is not None:
                out_lines.append({**ln, "value_set": which})
    # c5: no side22 anchor -> proxy max(c4, c6) rate, flagged
    for which in ("standard", "recommended"):
        key = "fastfast" if (which == "standard" or rec_floors.get("c5", {}).get("side22", "fastfast") == "fastfast") else "notslow"
        c4l, c6l = proposals["c4"]["side22"][key], proposals["c6"]["side22"][key]
        px = c4l if c4l["rate"] >= c6l["rate"] else c6l
        out_lines.append({"id": f"c5.floor.side22_{key}", "stage": "c5", "kind": "floor", "class": "fixed_t",
                          "statistic": px["statistic"], "lines": px["lines"], "rate": px["rate"], "value_set": which,
                          "proxy_of": f"max(c4, c6) = {px['stage']}", "seeds_pass": None,
                          "seeds_pass_note": "no c5 side22 stamp exists (PLAN left-for-implementation item 4): the gate row must be added; rate is a proxy",
                          "known_bad": {}, "unvalidated": True})
    # ---------------------------------------------------------------- p2a field_hard (three-seed line)
    fdef = stages["p2a"]["field"]
    fh = {int(s): field_completion([repo.stamp(e) for e in es]) for s, es in fdef["field_hard"].items()}
    for s, es in fdef["field_hard"].items():
        provenance += [{"stage": "p2a", "row": "field_hard__omni", "seed": int(s), "env_seed": i + 1, **e} for i, e in enumerate(es)]
    v = [fh[s]["rate"] for s in sorted(fh)]
    n = fh[sorted(fh)[0]]["n"]
    Rfh = max(2.0 * math.sqrt(x * (1 - x) / n) for x in v)
    Gfh = binomial_guard_floor(float(np.mean(v)), n)
    Reff = max(Rfh, Gfh)
    line = three_seed_line(v, Rfh, Gfh)
    ts = three_seed_rate(v, Reff, table)
    kbf = {}
    for k, rows_ in stages["p2a"]["known_bad_field"].items():
        c = field_completion([repo.stamp(e) for e in rows_["field_hard"]])
        kbf[k] = {"value": c["rate"], "n": c["n"], "fails": c["rate"] < line}
    fh_line = {"id": "p2a.capability.field_hard", "stage": "p2a", "kind": "capability", "class": "three_seed",
               "statistic": "field_hard completion pooled es1-4 (n = 400)",
               "key_path": "aggregate.schedule_completion_rate, n_episodes-weighted over env seeds 1-4 (p2_read 'completion')",
               "seed_values": dict(zip(sorted(fh), v)), "R_2SE": Rfh, "G": Gfh, "line": line, "direction": ">=",
               "line_rule": "min - max(R, range, G)", **ts, "seeds_pass": all(x >= line for x in v), "known_bad": kbf,
               "value_set": "both"}
    out_lines.append(fh_line)
    # p2a other field rows (WATCH; shown for context, and their values feed the student / 0-kg reads)
    field_ctx = {row: {int(s): field_completion([repo.stamp(e) for e in es]) for s, es in fdef[row].items()}
                 for row in ("field_light", "ring_obst", "field_turn")}
    # ---------------------------------------------------------------- p3a + student floors: proxy of p2a
    for st2 in ("p3a", "student"):
        for x in [l for l in out_lines if l["stage"] == "p2a" and l["kind"] == "floor" and l["value_set"] == "standard"]:
            y = {k: vv for k, vv in x.items() if k not in ("known_bad",)}
            y.update({"id": x["id"].replace("p2a.", f"{st2}."), "stage": st2, "proxy_of": "p2a (S1 s16/17/18)",
                      "pending": "step 8" if st2 == "p3a" else "no student exists; rate is a lower bound",
                      "known_bad": x["known_bad"], "value_set": "both"})
            out_lines.append(y)
    # ---------------------------------------------------------------- p3a stable-at-90 (a priori, pending)
    apr = []
    s1 = inputs["p3a_apriori_s1"]
    per_seed = {}
    for s, rows_ in s1.items():
        s = int(s)
        su = {int(es): suite_categories(repo.stamp(e)) for es, e in rows_["suite"].items()}
        hm = {int(es): h_meas_and_orientation(repo.stamp(e)) for es, e in rows_["suite"].items()}
        flh = {int(es): h_meas_and_orientation(repo.stamp(e)) for es, e in rows_["field_light"].items()}
        flc = field_completion([repo.stamp(e) for e in rows_["field_light"].values()])
        per_seed[s] = {"suite": su, "h": hm, "fl_h": flh, "fl_c": flc}
    seeds = sorted(per_seed)
    def _apr(id_, stat, key_path, vals, line, hib, se_note):
        sig = sigma_hat(vals, 0.0)
        return {"id": f"p3a.stable90.{id_}", "stage": "p3a", "kind": "capability", "class": "payload_apriori",
                "statistic": stat, "key_path": key_path, "line": line, "direction": ">=" if hib else "<=",
                "s1_0kg_values": dict(zip(seeds, vals)),
                "s1_passes": all((x >= line) if hib else (x <= line) for x in vals),
                "sigma_hat_p2a": sig, "rate": None, "pending": "step 8 (pilot value = mean; var = 2 sigma_hat_p2a^2 + SE^2)",
                "rate_if_pilot_equalled_s1_mean": payload_apriori_rate(line, float(np.mean(vals)), sig, 0.0, hib),
                "se_note": se_note, "known_bad": {}, "unvalidated": True, "value_set": "both"}
    for es in (1, 2):
        for c in stages["p2a"]["trained_translate"]:
            apr.append(_apr(f"survival.{c}.es{es}", f"translate survival {c} @ 90 kg", f"suite es{es} cat_summary survival",
                            [per_seed[s]["suite"][es][c]["survival"] for s in seeds], 0.90, True, "binomial SE at the pilot's p"))
            apr.append(_apr(f"tracking.{c}.es{es}", f"|ratio - 1| {c} @ 90 kg", f"suite es{es} by_category.{c}.ratio.median",
                            [abs(per_seed[s]["suite"][es][c]["ratio"] - 1.0) for s in seeds], 0.25, False, "paired eval SE"))
        for c in ("turn", "move_turn"):
            apr.append(_apr(f"survival.{c}.es{es}", f"{c} survival @ 90 kg", f"suite es{es} cat_summary survival",
                            [per_seed[s]["suite"][es][c]["survival"] for s in seeds], 0.885, True, "binomial SE"))
        apr.append(_apr(f"turn_fc.es{es}", "turn facing_closed @ 90 kg", f"suite es{es} by_category.turn.facing_closed.median",
                        [per_seed[s]["suite"][es]["turn"]["facing_closed"] for s in seeds], 0.80, True, "paired eval SE"))
        apr.append(_apr(f"h_meas.suite.es{es}", "median h_meas (m) @ 90 kg, suite", "raw npz: root z - mean terrain_z (proposed payload_read)",
                        [per_seed[s]["h"][es]["h_meas_median"] for s in seeds], 0.85, True, "episode bootstrap"))
        for o in ("roll_rms", "pitch_rms"):
            apr.append(_apr(f"{o}.suite.es{es}", f"{o} (rad) @ 90 kg, suite", f"aggregate.orientation.{o}.median",
                            [per_seed[s]["h"][es][o] for s in seeds], 0.10, False, "episode bootstrap"))
            apr.append(_apr(f"{o}.field_light.es{es}", f"{o} (rad) @ 90 kg, field_light", f"aggregate.orientation.{o}.median",
                            [per_seed[s]["fl_h"][es][o] for s in seeds], 0.10, False, "episode bootstrap"))
        apr.append(_apr(f"h_meas.field_light.es{es}", "median h_meas (m) @ 90 kg, field_light", "raw npz (proposed payload_read)",
                        [per_seed[s]["fl_h"][es]["h_meas_median"] for s in seeds], 0.85, True, "episode bootstrap"))
    apr.append(_apr("field_light_completion", "field_light completion es1-2 @ 90 kg", "aggregate.schedule_completion_rate pooled es1-2",
                    [per_seed[s]["fl_c"]["rate"] for s in seeds], 0.85, True, "binomial SE"))
    out_lines += apr
    # ---------------------------------------------------------------- the paired 0-kg check (X12 vs X08)
    pr = inputs["drift_pair"]
    A = {int(es): suite_episodes(repo.stamp(e)) for es, e in pr["x12"]["suite"].items()}
    B = {int(es): suite_episodes(repo.stamp(e)) for es, e in pr["x08"]["suite"].items()}
    zk = []
    stats_ = [(c, "ratio", True) for c in stages["p2a"]["trained_translate"]] + [("move_turn", "ratio", True)] + \
             [(c, "survival", False) for c in stages["p2a"]["trained_translate"]] + \
             [("turn", "facing_closed", False), ("turn", "yaw_progress", False)]
    for es in sorted(A):
        for c, key, absdev in stats_:
            ra = [r for r in A[es] if r["cat"] == c]
            rb = [r for r in B[es] if r["cat"] == c]
            a, b = _cat_stat(ra, key, absdev), _cat_stat(rb, key, absdev)
            se = paired_suite_se(A[es], B[es], c, key, absdev)
            worse = (a - b) if absdev else (b - a)          # positive = X12 worse
            if key == "survival":
                Rk = max(R.get(f"cat.{c}.survival", 0.0), binomial_guard_floor((a + b) / 2, len(ra)))
            elif key == "ratio":
                Rk = R.get(f"cat.{c}.ratio", 0.0)
            else:
                Rk = R.get(f"cat.{c}.{key}", 0.0)
            zk.append({"statistic": f"{'|ratio-1|' if absdev else key} {c}", "row": f"suite es{es}",
                       "key_path": f"aggregate.suite.by_category.{c}.{key}.median" if key != "survival" else "own-hold survival",
                       "x12": a, "x08": b, "worse_by": worse, "se_diff": se, "R": Rk})
    for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
        ea = pr["x12"][row][:2] if row == "field_hard" else pr["x12"][row]
        eb = pr["x08"][row][:2] if row == "field_hard" else pr["x08"][row]
        ca = {}
        cb = {}
        for i, (x, y) in enumerate(zip(ea, eb)):
            ca.update({(i, k): v for k, v in field_episodes(repo.stamp(x)).items()})
            cb.update({(i, k): v for k, v in field_episodes(repo.stamp(y)).items()})
        a, b = float(np.mean(list(ca.values()))), float(np.mean(list(cb.values())))
        se = paired_completion_se(ca, cb)
        nn = len(ca)
        zk.append({"statistic": f"{row} completion", "row": f"{row} es1-{len(ea)}", "x12": a, "x08": b,
                   "worse_by": b - a, "se_diff": se, "R": max(2.0 * math.sqrt(a * (1 - a) / nn), 2.0 * math.sqrt(b * (1 - b) / nn)),
                   "key_path": "aggregate.schedule_completion_rate (paired on env_index)"})
    sa = side22_read(s22, repo.stamp(pr["x12"]["side22"]))
    sb = side22_read(s22, repo.stamp(pr["x08"]["side22"]))
    for z in zk:
        z["drift"] = z["worse_by"]
        z["self_check_its5000"] = zero_kg_margin(z["R"], z["se_diff"], z["drift"], 5000)
        z["self_check_passes"] = z["worse_by"] <= z["self_check_its5000"]["margin"] + 1e-12
        z["by_Np"] = {}
        for Np in (1500, 2500, 3000):
            m = zero_kg_margin(z["R"], z["se_diff"], z["drift"], Np)
            z["by_Np"][Np] = {**m, "rate": paired_rate(m["margin"], z["se_diff"]),
                              "x12_vs_x08_would_pass": z["worse_by"] <= m["margin"] + 1e-12}
    zero_kg = {"id": "p3a.capability.zero_kg_paired", "stage": "p3a", "kind": "capability", "class": "paired",
               "rule": "HALT if any statistic is worse than its p2a head (same rows, env seeds) by more than "
                       "max(R, 3 SE_diff, |drift| * its_past_p2a / 5000), or FAST/FAST is lost",
               "statistics": zk, "side22_fastfast": {"x12": sa["class"], "x08": sb["class"]},
               "rate_by_Np": {Np: total(z["by_Np"][Np]["rate"] for z in zk) for Np in (1500, 2500, 3000)},
               "rate_note": "illustrative: SE_diff is the X12-X08 pair's; at gate time the T* vs own-p2a SE_diff is used. "
                            "Each statistic's margin >= 3 SE_diff, so each rate <= Phi(-3) = 0.00135. FAST/FAST loss is "
                            "counted under the p3a side22 floor (proxy p2a).",
               "drift_margin": "unvalidated (one continuation run)", "known_bad": {}, "unvalidated": True,
               "known_good_self_check": all(z["self_check_passes"] for z in zk), "value_set": "both"}
    out_lines.append(zero_kg)
    # ---------------------------------------------------------------- student completion (formula only)
    stu = []
    for z in zk:
        if "completion" in z["statistic"]:
            m = max(0.05, 2 * z["se_diff"])
            stu.append({"row": z["row"], "se_diff_illustrative": z["se_diff"], "margin": m,
                        "rate_per_line": paired_rate(m, z["se_diff"])})
    p_block = union(x["rate_per_line"] for x in stu)
    out_lines.append({"id": "student.capability.completion_paired", "stage": "student", "kind": "capability",
                      "class": "paired", "rule": "completion >= T* - max(0.05, 2 SE_diff, manifest-shape split) on "
                      "field_hard, field_light, ring_obst, field_turn (whole schedule); HALT after 2 failed full blocks",
                      "illustrative_lines": stu, "p_block_completion_only": p_block,
                      "rate": student_halt_rate(p_block), "rate_note": "lower bound: SE_diff from a teacher-teacher pair "
                      "(X12-X08); no distillation-seed variance; the manifest-shape split (step 8) only widens the margin",
                      "known_bad": {}, "unvalidated": True, "value_set": "both", "seeds_pass": None})
    # ---------------------------------------------------------------- live kills (unvalidated, excluded)
    for st in walker_stages + ["p2a", "p3a", "student", "D1 (SN)", "D2 (SH)"]:
        out_lines.append({"id": f"{st}.live_collapse", "stage": st, "kind": "live_kill", "class": "live_kill",
                          "rate": None, "unvalidated": True, "excluded_from_sum": True, "value_set": "both",
                          "known_bad": {}, "seeds_pass": None})
    # ---------------------------------------------------------------- stand
    st_std = stand_lines(repo, inputs["stand"], stand_standard_overrides or {})
    st_rec = stand_lines(repo, inputs["stand"], stand_overrides)
    for x in st_std:
        x["value_set"] = "standard"
    for x in st_rec:
        x["value_set"] = "recommended"
    out_lines += st_std + st_rec
    return {"lines": out_lines, "proposals": proposals, "provenance": provenance, "equality_checks": eq,
            "R_used": R, "field_context_p2a": field_ctx}


def summarize(res: dict, watch_ids: set) -> dict:
    """Per-stage and run sums for the standard and recommended value sets (decisive reads only).
    ``run_sum`` uses each read's PLAN model; ``run_sum_sensitivity`` swaps in the Jeffreys
    predictive (beta-binomial / gamma-Poisson) where a line carries one -- zero-count and zero-miss
    lines are 0 under the plain model but not under the predictive one."""
    out = {}
    for which in ("standard", "recommended"):
        by, sens = {}, {}
        for x in res["lines"]:
            if x.get("value_set") not in (which, "both") or x.get("rate") is None or x.get("excluded_from_sum"):
                continue
            if which == "recommended" and x["id"] in watch_ids:
                continue
            by.setdefault(x["stage"], []).append(x["rate"])
            rs = x.get("rate_sensitivity") or {}
            alt = rs.get("betabinom_jeffreys", rs.get("nb_jeffreys"))
            sens.setdefault(x["stage"], []).append(x["rate"] if alt is None else alt)
        out[which] = {"by_stage": {k: {"sum": total(v), "union": union(v), "n_reads": len(v),
                                       "sum_sensitivity": total(sens[k])} for k, v in by.items()},
                      "run_sum": total(r for v in by.values() for r in v),
                      "run_union": union(r for v in by.values() for r in v),
                      "run_sum_sensitivity": total(r for v in sens.values() for r in v)}
    return out


def self_check(res: dict, watch_ids: set) -> list:
    """PLAN self-checks: every recorded seed passes each decisive line (recommended set); each decisive
    line names a failing known-bad or is tagged ``unvalidated`` (the tag is added here, every value set)."""
    fails = []
    for x in res["lines"]:
        x["decisive"] = x["id"] not in watch_ids or x.get("value_set") == "standard"
        if x["id"] in watch_ids and x.get("value_set") == "recommended":
            x["watch"] = True
        if x.get("kind") != "live_kill" and not any(v.get("fails") for v in (x.get("known_bad") or {}).values()):
            x["unvalidated"] = True
        if x.get("value_set") not in ("recommended", "both") or x["id"] in watch_ids or x.get("kind") == "live_kill":
            continue
        if x.get("seeds_pass") is False:
            fails.append(f"{x['id']}: a recorded seed fails the line")
    return fails


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recommended", type=Path, default=None,
                    help="YAML/JSON {floors: {stage: {survival, ratio, side22}}, stand_overrides: {...}, watch: [ids]}")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    import yaml  # noqa: PLC0415
    inputs = yaml.safe_load(a.inputs.read_text())
    rec = yaml.safe_load(a.recommended.read_text()) if a.recommended else (inputs.get("recommended") or {})
    repo = Repo(repo_root())
    table = ThreeSeedTable(nsim=a.sims, seed=a.seed)
    res = build(inputs, repo, table, rec.get("floors") or {}, rec.get("stand_overrides") or {},
                rec.get("stand_standard_overrides") or {})
    watch = set(rec.get("watch") or [])
    fails = self_check(res, watch)
    doc = {"schema": SCHEMA, "version": inputs.get("version", "v0"), "tool": "TP/thresholds.py",
           "inputs": str(a.inputs), "three_seed_table": {"nsim": table.nsim, "seed": table.seed,
                                                       "rate_at_r0": table(0.0), "rate_at_r2": table(2.0)},
           "recommended": rec, "lines": res["lines"], "proposals": res["proposals"],
           "sums": summarize(res, watch), "self_check_failures": fails, "provenance": res["provenance"],
           "equality_checks": res["equality_checks"], "R_used": res["R_used"],
           "floors_apply_to_D1_D2": False,
           "floors_apply_to_D1_D2_note": "D1/D2 rows route the stand net beside a FROZEN T*: the walker floors were "
                                         "already read on T* at its own stage; the stand gates are G1-G5 / K1-K6 only "
                                         "(G4 / K5-G4 carry the walls). PLAN left-for-implementation item 15.",
           "field_context_p2a": res["field_context_p2a"]}
    a.out.write_text(json.dumps(doc, indent=1, default=_json_default))
    print(f"wrote {a.out}: {len(res['lines'])} lines; run sum standard {doc['sums']['standard']['run_sum']:.3f} "
          f"recommended {doc['sums']['recommended']['run_sum']:.3f}; self-check failures {len(fails)}")
    for f in fails:
        print("  SELF-CHECK:", f)
    return 1 if (a.self_check and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
