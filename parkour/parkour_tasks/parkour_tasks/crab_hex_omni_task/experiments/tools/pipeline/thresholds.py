#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 pipeline thresholds: every DECISIVE gate line, its chance-HALT rate and its self-checks (CPU).

One job: read an input map (``threshold_inputs.yaml``: stage -> seed -> recorded eval stamps, the
known-bad stamps, the stand head reads, the 0-kg drift pair) and write ``thresholds.json`` -- for
every decisive line the statistic (and the JSON key path it is read from), the three recorded seed
values, the line, the per-read chance-HALT rate under the PLAN's model for its class, the
known-bad value(s) with pass / fail, and whether every recorded seed passes. It also writes the
floor proposals (candidate values with their rates) and the run sums. Nothing here boots Kit,
trains or evaluates: it reads saved JSON / npz only and writes one JSON. No dated path lives in
this file; every path comes from the input map.

READ CLASSES AND THEIR MODELS (PLAN "Gates -> Rule -> Self-checks")
-------------------------------------------------------------------
* ``three_seed``  -- a line built from the three recorded seeds (``min - max(R, range, G)``, mirrored
  for lower-is-better). Rate = the n = 3 predictive probability that a fresh seed falls below the
  line, simulated as a function of ``r = R/sigma_hat`` (:func:`simulate_three_seed_rate`), with
  ``sigma_hat^2 = (range/1.693)^2 + (R/1.96)^2`` (:func:`sigma_hat`). ~6.6 % at r = 0, ~1 % at r = 2.
* ``fixed_t``     -- a fixed line (the lowered floors' ratio lines, the side22 FAST medians, the
  S1-era stand lines): ``P(T_2 < (line - mean) / (sigma_hat * sqrt(1 + 1/3)))`` (:func:`fixed_line_t_rate`).
  DEGENERATE (``sigma_hat`` 0: every recorded seed equal, R = 0): the predictive t is undefined, so there is no
  modelled rate. From thresholds v1.1 (input key ``degenerate_fixed_t: unvalidated``; CHARTER 2026-10-21 decision C)
  such a line reports ``rate: None`` + ``rate_status: "degenerate/unvalidated"`` and every sum excludes it EXPLICITLY
  (``n_excluded_degenerate`` per stage, ``excluded_degenerate`` per value set) -- the plain read (0.0 when the seeds
  pass) let a HALT quote "chance 0.0" for an unmodelled line (R18 SN G1 taps/s). ``legacy_zero`` (the v0 / v1 default)
  keeps the plain read, so the frozen v0 / v1 files regenerate byte for byte.
* ``binomial``    -- survival floors and other proportions: the pooled recorded misses over the three
  seeds give ``p``; rate = ``P(Bin(n_read, p) > n_read * (1 - floor))`` (:func:`binomial_floor_rate`).
  Sensitivity (reported, not summed): the Jeffreys beta-binomial predictive (:func:`betabinom_floor_rate`).
* ``poisson``     -- counts: ``lambda`` = the mean recorded count; rate = ``P(Pois(lambda) > line)``
  (:func:`poisson_count_rate`). Sensitivity: the Jeffreys gamma-Poisson predictive (:func:`nb_count_rate`).
* ``paired``      -- the 0-kg check and the student completion: ``Phi(-margin / SE_diff)``
  (:func:`paired_rate`); the student's HALT needs two failed blocks -> ``p_block^2`` (lower bound).
  The 0-kg check's tracking statistic is a one-sided ratio DROP (p2a - head) with R at the 99 % GUARD level
  (:func:`resolution_R_guard`; CHARTER STOP-0 2026-10-16); ``|r - 1|`` is reported as WATCH only. From thresholds
  v1.2 (input key ``zero_kg_translate_statistic: track_err``; CHARTER 2026-10-22, F19 p3b -> "amend + switch now")
  every TRAINED TRANSLATE category reads ``track_err <c>`` instead: the change in tracking error
  ``|r_head - 1| - |r_p2a - 1|`` on the same per-category median ratio, failing on an increase only (the one-sided
  drop failed F19's payload heads for moving p2a's 1.05-1.07 overshoot toward 1.0), with the 99 % guard R of
  ``cat.<c>.ratio``, SE_diff and the X12 - X08 drift all on ``|r - 1|``; move_turn stays the one-sided ratio drop,
  and the raw translate drop is kept as a WATCH read (:data:`ZERO_KG_TRANSLATE_MODES`). The mode is recorded on the
  0-kg line (``translate_statistic``), so a gate applies the semantics of the file it reads. Its pure
  pieces (:func:`zero_kg_suite_rows`, :func:`zero_kg_completion_row`, :func:`zero_kg_judge`) are the ones
  ``gate_read.py`` applies at gate time, so the self-check and the gate cannot drift apart.
* ``payload_apriori`` -- pending step 8 (no pilot yet): formula ``Phi((line - mu_pilot) /
  sqrt(2 sigma_hat_p2a^2 + SE^2))`` (:func:`payload_apriori_rate`); only illustrated here.
* ``live_kill`` / the 0-kg drift margin -- ``unvalidated``, excluded from every sum.

Usage::

    thresholds.py --inputs <PL>/threshold_inputs_v1_2.yaml --out <PL>/thresholds_v1_2.json [--recommended FILE]
                  [--self-check] [--sims N] [--seed S]

``--recommended FILE`` (YAML / JSON ``{floors: {stage: {survival, ratio, side22}}, stand_overrides: {...}, watch:
[ids]}``) replaces the input map's own ``recommended:`` block, which is used when the option is absent. Run from the
repo root with the inputs path relative to it (the JSON records it as given): ``--inputs
<PL>/threshold_inputs_v1_2.yaml --self-check`` reproduces the committed ``thresholds_v1_2.json`` byte for byte
(``tests/unit/test_task4_thresholds.py``).

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
import tempfile
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

#: The paired 0-kg check's ONE row list (PLAN Payload -> 0-kg check; read at p3a, the step-8 pilot and T*):
#: suite es1-2, side22, field_hard, field_light, ring_obst, field_turn at es1-2. p2a field_hard is ALSO read at
#: es1-4 (n = 400) for its own capability line (CHARTER STOP-0); the 0-kg check uses its es1-2 rows only.
ZERO_KG_ROW_LIST = {"suite__omni": [1, 2], "side22_25lr__omni": [1], "field_hard__omni": [1, 2],
                    "field_light__omni": [1, 2], "ring_obst__omni": [1, 2], "field_turn__omni": [1, 2]}
ZERO_KG_FIELD_ROWS = ("field_hard", "field_light", "ring_obst", "field_turn")
#: the env seeds of the RECORDED drift pair the self-check reads (X12 / X08 have field_turn at es1 only)
ZERO_KG_FIELD_ENV_SEEDS = {"field_hard": (1, 2), "field_light": (1, 2), "ring_obst": (1, 2), "field_turn": (1,)}
P2A_FIELD_HARD_ENV_SEEDS = (1, 2, 3, 4)

SURVIVAL_CANDIDATES = (0.90, 0.80, 0.70, 0.60)
RATIO_CANDIDATES = (0.70, 0.60, 0.50, 0.40)
STANDARD_FLOORS = {"survival": 0.90, "ratio": 0.70}
SIDE22_FAST = {"m22": 0.155, "m25": 0.175}      # side22_classify.GATE["fast"]
SIDE22_SLOW = {"m22": 0.13, "m25": 0.145}       # side22_classify.GATE["slow"]

#: CHARTER 2026-10-21 decision C: the rate status of a degenerate fixed-t line (sigma_hat 0) -- no modelled chance rate.
#: gate_read.py carries the same string (and the same predicate, :func:`is_degenerate`; pinned equal by a unit test).
DEGENERATE = "degenerate/unvalidated"
#: ``unvalidated`` = rate None + DEGENERATE, excluded from the sums explicitly (v1.1 on); ``legacy_zero`` = the plain
#: 0.0 / 1.0 read (what v0 / v1 were written with)
DEGENERATE_MODES = ("unvalidated", "legacy_zero")
#: The DECISIVE 0-kg statistic of a trained translate category (input key ``zero_kg_translate_statistic``; recorded on
#: the 0-kg line as ``translate_statistic``). ``ratio_drop`` (v0 - v1.1, CHARTER STOP-0 2026-10-16): ``ratio <c>``, a
#: one-sided ratio DROP ``p2a - head``. ``track_err`` (v1.2 on, CHARTER 2026-10-22 decision "amend + switch now"):
#: ``track_err <c>`` = ``|r_head - 1| - |r_p2a - 1|`` (an INCREASE in tracking error fails; a head that moves an
#: overshoot toward 1.0 never does), and ``ratio <c>`` -- the same raw drop as before -- becomes a WATCH read. move_turn
#: is the one-sided ratio drop in both modes.
ZERO_KG_TRANSLATE_MODES = ("ratio_drop", "track_err")


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


def is_degenerate(x: dict) -> bool:
    """Is ``x`` (a thresholds line) a DEGENERATE fixed-t line -- sigma_hat 0, so no modelled rate?

    True for a line already marked (``rate_status`` DEGENERATE), a line carrying :func:`fixed_line_t_rate`'s own
    ``degenerate`` flag (the stand clauses), a fixed-t line whose top-level ``sigma_hat`` is 0 (the floor ratio lines),
    and a side22 line any of whose medians is degenerate. gate_read.is_degenerate is the same predicate."""
    if not isinstance(x, dict):
        return False
    if x.get("rate_status") == DEGENERATE or x.get("degenerate") is True:
        return True
    if any(isinstance(v, dict) and v.get("degenerate") is True for v in (x.get("per_median") or {}).values()):
        return True
    return x.get("class") == "fixed_t" and x.get("sigma_hat") == 0.0


def degenerate_mode(inputs: dict) -> str:
    """``inputs.degenerate_fixed_t`` (one of :data:`DEGENERATE_MODES`). Absent: ``legacy_zero`` for the frozen v0 / v1
    input maps (they regenerate byte for byte), ``unvalidated`` for every later version (CHARTER 2026-10-21 C)."""
    m = inputs.get("degenerate_fixed_t")
    if m is None:
        m = "legacy_zero" if str(inputs.get("version", "v0")) in ("v0", "v1") else "unvalidated"
    if m not in DEGENERATE_MODES:
        raise ValueError(f"degenerate_fixed_t {m!r} (want one of {DEGENERATE_MODES})")
    return m


def zero_kg_translate_mode(inputs: dict) -> str:
    """``inputs.zero_kg_translate_statistic`` (one of :data:`ZERO_KG_TRANSLATE_MODES`). Absent: ``ratio_drop`` for the
    frozen v0 / v1 / v1.1 input maps (they regenerate byte for byte), ``track_err`` for every later version (CHARTER
    2026-10-22: thresholds v1.2)."""
    m = inputs.get("zero_kg_translate_statistic")
    if m is None:
        m = "ratio_drop" if str(inputs.get("version", "v0")) in ("v0", "v1", "v1.1") else "track_err"
    if m not in ZERO_KG_TRANSLATE_MODES:
        raise ValueError(f"zero_kg_translate_statistic {m!r} (want one of {ZERO_KG_TRANSLATE_MODES})")
    return m


def mark_degenerate(lines: list, mode: str) -> dict:
    """CHARTER 2026-10-21 decision C on every degenerate fixed-t line (``mode`` 'unvalidated'): ``rate`` None,
    ``rate_status`` DEGENERATE, ``unvalidated`` True, and a ``rate_note`` naming the plain read it replaces. Returns
    ``{"<id>|<value_set>": the plain read}`` (empty in ``legacy_zero`` mode, where nothing changes)."""
    plain = {}
    if mode != "unvalidated":
        return plain
    for x in lines:
        if x.get("rate") is None or not is_degenerate(x):
            continue
        plain[f"{x['id']}|{x.get('value_set')}"] = x["rate"]
        x.update(rate=None, rate_status=DEGENERATE, degenerate=True, unvalidated=True,
                 rate_note=(f"sigma_hat = 0 (every recorded seed equal, R = 0): the n = 3 predictive t is undefined, "
                            f"so no chance rate is modelled (the plain read would be {x['rate']:g}); excluded from "
                            "every sum, counted in n_excluded_degenerate (CHARTER 2026-10-21 decision C)"))
    return plain


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


def resolution_R_guard(paths: list) -> dict:
    """metric -> max R_eval at the 99 % GUARD level over the resolution files (CHARTER STOP-0, 0-kg check).

    For ``<metric>`` the ``<metric>@guard`` entry (the one-sided 99 % guard-type read ``resolution.py`` writes
    beside a 95 % block entry), else ``<metric>`` itself when that entry IS a guard-level one (``level`` 0.99;
    e.g. ``cat.move_turn.ratio``). A metric with neither has no entry here (the caller decides, loudly)."""
    R: dict = {}
    for p in paths:
        m = load_json(p)["metrics"]
        for k, v in m.items():
            if not isinstance(v, dict) or v.get("R_eval") is None:
                continue
            if k.endswith("@guard"):
                base = k[: -len("@guard")]
            elif "@" not in k and abs(float(v.get("level") or 0.0) - GUARD_LEVEL) < 1e-9:
                base = k
            else:
                continue
            R[base] = max(R.get(base, 0.0), float(v["R_eval"]))
    return R


#: The paired 0-kg check's DECISIVE statistics per suite env seed (PLAN Payload -> 0-kg check; CHARTER STOP-0
#: 2026-10-16): tracking on every trained translate category and move_turn (R at the 99 % guard level), survival per
#: trained translate category, turn facing_closed and yaw_progress. Each is (category, key, form): form ``drop`` reads
#: the raw median, higher is better, ``worse_by = p2a - head``; form ``absdev`` reads ``|median - 1|``, lower is
#: better, ``worse_by = head - p2a``. Translate tracking per ``translate`` (:data:`ZERO_KG_TRANSLATE_MODES`):
#: ``ratio_drop`` -> ``(c, "ratio", "drop")`` (v0 - v1.1); ``track_err`` -> ``(c, "track_err", "absdev")`` (v1.2 on,
#: CHARTER 2026-10-22). move_turn is ``("move_turn", "ratio", "drop")`` in both modes.
def zero_kg_decisive_stats(trained_translate, translate: str = "ratio_drop") -> list:
    tr = ([(c, "track_err", "absdev") for c in trained_translate] if translate == "track_err"
          else [(c, "ratio", "drop") for c in trained_translate])
    return (tr + [("move_turn", "ratio", "drop")]
            + [(c, "survival", "drop") for c in trained_translate]
            + [("turn", "facing_closed", "drop"), ("turn", "yaw_progress", "drop")])


#: WATCH-only statistics of the 0-kg check (reported, never decisive). ``ratio_drop`` (CHARTER STOP-0): ``|r - 1|``
#: on every trained translate category and move_turn. ``track_err`` (CHARTER 2026-10-22): the translate ``|r - 1|``
#: is decisive now (``track_err <c>``), so the WATCH reads are the old raw translate DROP (``ratio <c>``: records still
#: show a speed drop) and ``|ratio-1| move_turn`` as before.
def zero_kg_watch_stats(trained_translate, translate: str = "ratio_drop") -> list:
    if translate == "track_err":
        return [(c, "ratio", "drop") for c in trained_translate] + [("move_turn", "ratio", "absdev")]
    return [(c, "ratio", "absdev") for c in trained_translate] + [("move_turn", "ratio", "absdev")]


def zero_kg_stat_name(cat: str, key: str, form: str) -> str:
    """``ratio <c>`` (the raw drop), ``|ratio-1| <c>`` (|r - 1|, 95 % block R; WATCH), ``track_err <c>`` (|r - 1|,
    99 % guard R; decisive from v1.2), ``<key> <c>`` otherwise. One name per quantity: a record never reuses a name
    for a different statistic (``drift_by_statistic`` is keyed ``<name>|<row>``)."""
    return f"|ratio-1| {cat}" if (key == "ratio" and form == "absdev") else f"{key} {cat}"


def zero_kg_R(cat: str, key: str, R: dict, Rg: dict, n: int, p_mean: float | None, *, watch: bool = False,
              form: str | None = None) -> dict:
    """The R term of one 0-kg statistic, and where it came from (``form`` None = the pre-v1.2 reading: a WATCH ratio
    row is the |r - 1| one).

    * ratio, form drop (decisive; from v1.2 also the WATCH raw drop): the 99 % guard R (:func:`resolution_R_guard`);
      a category without one -> ``None`` (the caller fails a decisive row closed -- an R silently read as 0 would turn
      every statistic into a 3 SE line).
    * ratio, form absdev (WATCH |r - 1|): the 95 % block R, as before the STOP-0 decision.
    * track_err (decisive from v1.2): the 99 % guard R of ``cat.<c>.ratio`` -- |r - 1| is a function of the same
      median, so it shares its resolution (CHARTER 2026-10-22); no guard entry -> ``None`` (fails closed).
    * survival: max(guard R, binomial guard G(n) at the pooled share).
    * facing_closed / yaw_progress: the 95 % block R (unchanged by the STOP-0 decision)."""
    if key in ("ratio", "track_err"):
        absdev = (form == "absdev") if form is not None else watch
        if key == "ratio" and watch and absdev:
            return {"R": R.get(f"cat.{cat}.ratio", 0.0), "R_source": f"cat.{cat}.ratio (95 % block; WATCH)"}
        if f"cat.{cat}.ratio" not in Rg:
            return {"R": None, "R_source": f"no 99 % guard entry for cat.{cat}.ratio"}
        src = f"cat.{cat}.ratio@guard (99 %)"
        if key == "track_err":
            src += " on |r - 1| (the same median's resolution; CHARTER 2026-10-22)"
        elif watch:
            src += "; WATCH raw drop (v1.2, CHARTER 2026-10-22)"
        return {"R": Rg[f"cat.{cat}.ratio"], "R_source": src}
    if key == "survival":
        g = binomial_guard_floor(1.0 if p_mean is None else p_mean, max(int(n), 1))
        r = R.get(f"cat.{cat}.survival", 0.0)
        return {"R": max(r, g), "R_source": f"max(cat.{cat}.survival {r:.4f}, binomial guard G(n={n}) {g:.4f})"}
    return {"R": R.get(f"cat.{cat}.{key}", 0.0), "R_source": f"cat.{cat}.{key} (95 % block)"}


#: ``worse_by_rule`` of a v1.2 row (recorded from v1.2 on, so a record says which difference it read)
ZERO_KG_WORSE_RULE = {"drop": "p2a - head (a one-sided drop)",
                      "absdev": "|head - 1| - |p2a - 1| (an increase in tracking error; one-sided)"}


def zero_kg_suite_rows(head_eps: dict, p2a_eps: dict, trained_translate, R: dict, Rg: dict,
                       translate: str = "ratio_drop") -> list:
    """The suite part of the paired 0-kg check, as rows (pure on :func:`suite_episodes` lists per env seed).

    ``head_eps`` / ``p2a_eps``: ``{env_seed: suite_episodes(stamp)}`` for the same env seeds. A form-``drop`` row
    reads ``worse_by = p2a - head`` on the raw statistic (a one-sided drop: a head that tracks or survives better is
    never failed); a form-``absdev`` row reads ``|r - 1|`` with ``worse_by = head - p2a`` (an increase in tracking
    error; a head whose ratio moves toward 1.0 is never failed, and 1.06 -> 0.94 reads 0). ``translate``
    (:data:`ZERO_KG_TRANSLATE_MODES`) picks the decisive translate statistic: ``ratio_drop`` (v0 - v1.1: decisive
    ``ratio <c>``, WATCH ``|ratio-1| <c>``) or ``track_err`` (v1.2, CHARTER 2026-10-22: decisive ``track_err <c>``,
    WATCH the raw ``ratio <c>`` drop); move_turn, survival and facing read the same in both. ``se_diff`` is the
    env-clustered paired bootstrap SE (:func:`paired_suite_se`) of the row's own statistic (on |r - 1| for absdev)."""
    if translate not in ZERO_KG_TRANSLATE_MODES:
        raise ValueError(f"translate {translate!r} (want one of {ZERO_KG_TRANSLATE_MODES})")
    rows = []
    for es in sorted(set(head_eps) & set(p2a_eps)):
        A, B = head_eps[es], p2a_eps[es]
        for watch, stats_ in ((False, zero_kg_decisive_stats(trained_translate, translate)),
                              (True, zero_kg_watch_stats(trained_translate, translate))):
            for c, key, form in stats_:
                src = "ratio" if key == "track_err" else key        # the episode field the statistic is read from
                absdev = form == "absdev"
                name = zero_kg_stat_name(c, key, form)
                ra = [r for r in A if r["cat"] == c]
                rb = [r for r in B if r["cat"] == c]
                a, b = _cat_stat(ra, src, absdev), _cat_stat(rb, src, absdev)
                if a is None or b is None:
                    rows.append({"statistic": name, "row": f"suite es{es}",
                                 "watch": watch, "head": a, "p2a": b, "worse_by": None, "missing": True})
                    continue
                se = paired_suite_se(A, B, c, src, absdev)
                worse = (a - b) if absdev else (b - a)
                Rk = zero_kg_R(c, key, R, Rg, len(ra), (a + b) / 2.0 if key == "survival" else None, watch=watch,
                               form=form)
                if key == "survival":
                    kp = "own-hold survival (n_steps >= prologue + settle + hold)"
                elif key == "track_err":
                    kp = f"|aggregate.suite.by_category.{c}.ratio.median - 1|"
                else:
                    kp = f"aggregate.suite.by_category.{c}.{key}.median"
                row = {"statistic": name, "row": f"suite es{es}",
                       "category": c, "key": key, "watch": watch,
                       "key_path": kp,
                       "head": a, "p2a": b, "worse_by": worse, "se_diff": se, **Rk}
                if translate != "ratio_drop":              # v1.2 on (absent from the frozen v0 - v1.1 records)
                    row["worse_by_rule"] = ZERO_KG_WORSE_RULE[form]
                rows.append(row)
    return rows


def zero_kg_completion_row(row: str, head_eps: dict, p2a_eps: dict, env_seeds) -> dict:
    """One field row of the paired 0-kg check: completion paired on (env seed index, env_index). The row label is
    the row name alone (pooled over ``env_seeds``, recorded beside it), so the drift key ``<statistic>|<row>`` is the
    same whatever the env-seed count (the recorded drift pair has field_turn at es1 only)."""
    a = float(np.mean(list(head_eps.values()))) if head_eps else None
    b = float(np.mean(list(p2a_eps.values()))) if p2a_eps else None
    if a is None or b is None or set(head_eps) != set(p2a_eps):
        return {"statistic": f"{row} completion", "row": row, "env_seeds": list(env_seeds), "watch": False,
                "head": a, "p2a": b, "worse_by": None, "missing": True,
                "why": "episode keys differ between the head and its p2a (not the same rows / env seeds)"}
    nn = len(head_eps)
    se = paired_completion_se(head_eps, p2a_eps)
    return {"statistic": f"{row} completion", "row": row, "env_seeds": list(env_seeds), "category": row,
            "key": "completion", "watch": False,
            "head": a, "p2a": b, "worse_by": b - a, "se_diff": se,
            "R": max(2.0 * math.sqrt(a * (1 - a) / nn), 2.0 * math.sqrt(b * (1 - b) / nn)),
            "R_source": "max over the pair of 2 sqrt(p(1-p)/n)",
            "key_path": "aggregate.schedule_completion_rate (paired on env_index)"}


def zero_kg_judge(rows: list, drift: dict, its_past_p2a: float, side22_head=None, side22_p2a=None) -> dict:
    """Apply the 0-kg margin to every DECISIVE row: HALT iff ``worse_by > max(R, 3 SE_diff, |drift| x its / 5000)``
    on any, or FAST/FAST is lost (the head's side22 class is not FAST on both hands while p2a's was). A decisive row
    that is missing, or whose R is unknown, FAILS closed. ``drift``: ``{statistic|row: drift}`` (the X12 - X08
    continuation drift on the same statistic and row, from thresholds.json). Returns the verdict and each row's
    margin use (the T* notification logs it)."""
    out, fails = [], []
    for r in rows:
        r = dict(r)
        if r.get("watch"):
            if r.get("worse_by") is not None and r.get("R") is not None:
                d = drift.get(f"{r['statistic']}|{r['row']}", 0.0)
                m = zero_kg_margin(r["R"], r.get("se_diff") or 0.0, d, its_past_p2a)
                r.update(margin=m, exceeds=bool(r["worse_by"] > m["margin"] + 1e-12))
            out.append(r)
            continue
        if r.get("missing") or r.get("R") is None:
            r["fails"] = True
            fails.append(f"{r['statistic']} {r['row']}: " + (r.get("why") or r.get("R_source") or "missing"))
            out.append(r)
            continue
        d = drift.get(f"{r['statistic']}|{r['row']}")
        if d is None:
            r["fails"] = True
            fails.append(f"{r['statistic']} {r['row']}: no recorded drift for this statistic (thresholds.json)")
            out.append(r)
            continue
        m = zero_kg_margin(r["R"], r.get("se_diff") or 0.0, d, its_past_p2a)
        r.update(drift=d, margin=m, fails=bool(r["worse_by"] > m["margin"] + 1e-12),
                 margin_used=(r["worse_by"] / m["margin"]) if m["margin"] > 0 else None)
        if r["fails"]:
            fails.append(f"{r['statistic']} {r['row']}: worse by {r['worse_by']:.4f} > margin {m['margin']:.4f} "
                         f"({m['binds']})")
        out.append(r)
    side = None
    if side22_head is not None or side22_p2a is not None:
        lost = (list(side22_p2a or []) == ["FAST", "FAST"]) and (list(side22_head or []) != ["FAST", "FAST"])
        side = {"head": side22_head, "p2a": side22_p2a, "fast_fast_lost": lost}
        if lost:
            fails.append(f"side22 FAST/FAST lost: head {side22_head} vs p2a {side22_p2a}")
    return {"verdict": "FAIL" if fails else "PASS", "fails": fails, "rows": out, "side22": side,
            "its_past_p2a": its_past_p2a}


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
               "statistic": "field_hard completion pooled es1-4 (n = 400)", "env_seeds": list(P2A_FIELD_HARD_ENV_SEEDS),
               "env_seeds_note": "read at es1-4 for this line (CHARTER STOP-0), in addition to the es1-2 rows of the "
                                 "0-kg row list",
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
    # CHARTER STOP-0 (2026-10-16): tracking is a one-sided ratio DROP with R at the 99 % guard level; |r - 1| is
    # WATCH only. From v1.2 (CHARTER 2026-10-22) the translate categories read track_err = |r_head - 1| - |r_p2a - 1|
    # instead (move_turn stays the drop; the raw translate drop is WATCH), so the drift below is recomputed on the
    # new statistic from the same X12 / X08 stamps (drift_by_statistic carries its "track_err <c>|<row>" keys). The
    # same pure functions (zero_kg_suite_rows / zero_kg_completion_row / zero_kg_judge) are what gate_read.py applies
    # to a T* candidate against its own p2a head at gate time, in the mode recorded on the line (translate_statistic).
    zmode = zero_kg_translate_mode(inputs)
    pr = inputs["drift_pair"]
    Rg = resolution_R_guard([repo.p(p) for p in inputs["resolution_files"]])
    A = {int(es): suite_episodes(repo.stamp(e)) for es, e in pr["x12"]["suite"].items()}
    B = {int(es): suite_episodes(repo.stamp(e)) for es, e in pr["x08"]["suite"].items()}
    zall = zero_kg_suite_rows(A, B, stages["p2a"]["trained_translate"], R, Rg, translate=zmode)
    for row in ZERO_KG_FIELD_ROWS:
        nes = ZERO_KG_FIELD_ENV_SEEDS[row]
        ea, eb = pr["x12"][row][:len(nes)], pr["x08"][row][:len(nes)]
        ca, cb = {}, {}
        for i, (x, y) in enumerate(zip(ea, eb)):
            ca.update({(i, k): v for k, v in field_episodes(repo.stamp(x)).items()})
            cb.update({(i, k): v for k, v in field_episodes(repo.stamp(y)).items()})
        zall.append(zero_kg_completion_row(row, ca, cb, nes[:len(ea)]))
    zk = [z for z in zall if not z.get("watch")]
    zw = [z for z in zall if z.get("watch")]
    sa = side22_read(s22, repo.stamp(pr["x12"]["side22"]))
    sb = side22_read(s22, repo.stamp(pr["x08"]["side22"]))
    for z in zk + zw:
        z["x12"], z["x08"] = z.pop("head"), z.pop("p2a")
        z["drift"] = z["worse_by"]
        if z.get("R") is None:
            z["self_check_passes"] = False
            continue
        z["self_check_its5000"] = zero_kg_margin(z["R"], z["se_diff"], z["drift"], 5000)
        z["self_check_passes"] = z["worse_by"] <= z["self_check_its5000"]["margin"] + 1e-12
        z["by_Np"] = {}
        for Np in (1500, 2500, 3000):
            m = zero_kg_margin(z["R"], z["se_diff"], z["drift"], Np)
            z["by_Np"][Np] = {**m, "rate": paired_rate(m["margin"], z["se_diff"]),
                              "x12_vs_x08_would_pass": z["worse_by"] <= m["margin"] + 1e-12}
    zero_kg = {"id": "p3a.capability.zero_kg_paired", "stage": "p3a", "kind": "capability", "class": "paired",
               "rule": "HALT if any DECISIVE statistic is worse than its p2a head (same rows, env seeds) by more than "
                       "max(R, 3 SE_diff, |drift| * its_past_p2a / 5000), or FAST/FAST is lost. Tracking = a one-sided "
                       "ratio DROP (p2a - head) on every trained translate category and move_turn, R at the 99 % guard "
                       "level (CHARTER STOP-0 2026-10-16); |r - 1| is WATCH (watch_statistics).",
               "row_list": ZERO_KG_ROW_LIST, "statistics": zk, "watch_statistics": zw,
               "drift_by_statistic": {f"{z['statistic']}|{z['row']}": z["drift"] for z in zk + zw},
               "side22_fastfast": {"x12": sa["class"], "x08": sb["class"]},
               "rate_by_Np": {Np: total(z["by_Np"][Np]["rate"] for z in zk if "by_Np" in z) for Np in (1500, 2500, 3000)},
               "rate_note": "illustrative: SE_diff is the X12-X08 pair's; at gate time the T* vs own-p2a SE_diff is used. "
                            "Each statistic's margin >= 3 SE_diff, so each rate <= Phi(-3) = 0.00135. FAST/FAST loss is "
                            "counted under the p3a side22 floor (proxy p2a). WATCH statistics are not summed.",
               "drift_margin": "unvalidated (one continuation run)", "known_bad": {}, "unvalidated": True,
               "known_good_self_check": all(z["self_check_passes"] for z in zk), "value_set": "both"}
    if zmode == "track_err":                           # v1.2 on (absent from the frozen v0 - v1.1 files)
        zero_kg["rule"] = (
            "HALT if any DECISIVE statistic is worse than its p2a head (same rows, env seeds) by more than "
            "max(R, 3 SE_diff, |drift| * its_past_p2a / 5000), or FAST/FAST is lost. Translate tracking (every trained "
            "translate category) = track_err <c>: worse_by = |r_head - 1| - |r_p2a - 1| on the per-category median "
            "ratio, failing on an INCREASE in tracking error only -- a head that moves p2a's overshoot toward 1.0 "
            "never fails (CHARTER 2026-10-22, F19 p3b: user decision 'amend + switch now', thresholds v1.2); "
            "R = the 99 % guard R of cat.<c>.ratio, SE_diff and the X12 - X08 drift on |r - 1|. move_turn = a "
            "one-sided ratio DROP (p2a - head), R at the 99 % guard level (CHARTER STOP-0 2026-10-16), unchanged. "
            "WATCH (watch_statistics): the raw translate ratio drop (ratio <c>, as decisive before v1.2) and "
            "|ratio-1| move_turn.")
        zero_kg["translate_statistic"] = zmode
        zero_kg["translate_statistic_note"] = (
            "gate_read.zero_kg_check applies the mode recorded here (absent = ratio_drop, the v0 - v1.1 semantics). "
            "drift_by_statistic carries the new track_err <c>|<row> keys beside the WATCH ratio <c>|<row> ones (the "
            "same raw-drop quantity and drift as v1.1's decisive ratio rows), so a track_err read of a pre-v1.2 file "
            "fails closed (no recorded drift for its statistic)")
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
    # ---------------------------------------------------------------- v1 inputs (absent = v0's output exactly)
    if inputs.get("p3a_pilot"):
        p3a_pilot_rates(repo, inputs["p3a_pilot"], out_lines, R, Rg, s22, stages["p2a"]["trained_translate"],
                        provenance, translate=zmode)
    if inputs.get("sweep_anchors"):
        out_lines += sweep_watch_lines(repo, inputs["sweep_anchors"], provenance)
    # ---------------------------------------------------------------- v1.1: degenerate fixed-t lines (decision C)
    mode = degenerate_mode(inputs)
    plain = mark_degenerate(out_lines, mode)
    return {"lines": out_lines, "proposals": proposals, "provenance": provenance, "equality_checks": eq,
            "R_used": R, "field_context_p2a": field_ctx, "degenerate_mode": mode, "degenerate_plain_reads": plain,
            "zero_kg_translate": zmode}


# ============================================================================ v1: the payload pilot and the sweep anchors
def _stamp_of(repo: Repo, entry) -> Path:
    """An input-map stamp: ``{stamp, eval_sha}`` (checked like every other input) or a bare path (completeness only)."""
    if isinstance(entry, dict):
        return repo.stamp(entry)
    d = repo.p(str(entry))
    for f in ("scenario_metrics.json", "run_meta.json", "summary.md"):
        if not (d / f).exists():
            raise FileNotFoundError(f"incomplete stamp {d} (no {f})")
    return d


def boot_se_median(vals, n_boot: int = 2000, seed: int = DEFAULT_SEED):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) < 3:
        return None
    g = np.random.default_rng(seed)
    return float(np.median(v[g.integers(0, len(v), (n_boot, len(v)))], axis=1).std())


def _row_stamps(repo: Repo, m: dict) -> dict:
    """``{"<row>@<es>": STAMP}`` (the pipeline's row keys) -> ``{"<row>@<es>": Path}``, each checked by
    :func:`_stamp_of`."""
    out = {}
    for k, e in m.items():
        row, _, es = str(k).partition("@")
        if not row or not es.isdigit():
            raise ValueError(f"p3a_pilot: row key {k!r} is not <row>@<env seed>")
        out[f"{row}@{int(es)}"] = _stamp_of(repo, e)
    return out


def pilot_stable90_values(st: dict) -> dict:
    """``{stable90 line id suffix: (value, SE or None)}`` on the pilot's L090 rows (``suite@1/2``,
    ``field_light@1/2``), read exactly as ``payload_read.level_reads`` reads them (the statistics the v0 lines were
    built on). SE (as ``t4_stepC_read.apriori_rates``): survival binomial on the category's n; tracking / turn
    facing_closed the episode-bootstrap SE of the median; field_light completion binomial on the pooled n; h_meas /
    orientation None (the caller stands sigma_hat(p2a) in, disclosed per line)."""
    out = {}
    for es in (1, 2):
        d = st.get(f"suite@{es}")
        if d is not None:
            cats, eps, ho = suite_categories(d), suite_episodes(d), h_meas_and_orientation(d)
            for c, v in cats.items():
                if c.startswith("_"):
                    continue
                sv, n = v.get("survival"), v.get("n") or 0
                if sv is not None and n:
                    out[f"survival.{c}.es{es}"] = (sv, math.sqrt(max(sv * (1 - sv), 1e-6) / n))
                if v.get("ratio") is not None:
                    out[f"tracking.{c}.es{es}"] = (abs(v["ratio"] - 1.0),
                                                   boot_se_median([e["ratio"] for e in eps if e["cat"] == c]))
            fc = (cats.get("turn") or {}).get("facing_closed")
            if fc is not None:
                out[f"turn_fc.es{es}"] = (fc, boot_se_median([e["facing_closed"] for e in eps if e["cat"] == "turn"]))
            out[f"h_meas.suite.es{es}"] = (ho["h_meas_median"], None)
            for o in ("roll_rms", "pitch_rms"):
                out[f"{o}.suite.es{es}"] = (ho[o], None)
        d = st.get(f"field_light@{es}")
        if d is not None:
            ho = h_meas_and_orientation(d)
            out[f"h_meas.field_light.es{es}"] = (ho["h_meas_median"], None)
            for o in ("roll_rms", "pitch_rms"):
                out[f"{o}.field_light.es{es}"] = (ho[o], None)
    fls = [st[f"field_light@{es}"] for es in (1, 2) if f"field_light@{es}" in st]
    if fls:
        c = field_completion(fls)
        out["field_light_completion"] = (c["rate"], math.sqrt(max(c["rate"] * (1 - c["rate"]), 1e-6) / c["n"]))
    return out


def pilot_zero_kg_rows(head: dict, p2a: dict, trained_translate, R: dict, Rg: dict,
                       translate: str = "ratio_drop") -> tuple:
    """The pilot pair's paired 0-kg rows, composed exactly as ``gate_read.zero_kg_check`` composes them (suite env
    seeds must match; a field row pairs its env seeds in order and is missing when the counts differ). ``head`` /
    ``p2a``: ``{"<row>@<es>": Path}``; ``translate``: the input map's :func:`zero_kg_translate_mode`. Returns (rows,
    side22 class of head, of p2a)."""
    def by_es(m, row):
        return [m[k] for k in sorted((k for k in m if k.split("@")[0] == row), key=lambda k: int(k.split("@")[1]))]

    A = {int(k.split("@")[1]): suite_episodes(v) for k, v in head.items() if k.startswith("suite@")}
    B = {int(k.split("@")[1]): suite_episodes(v) for k, v in p2a.items() if k.startswith("suite@")}
    if set(A) != set(B) or not A:
        raise ValueError(f"p3a_pilot zero_kg: suite env seeds differ (head {sorted(A)} vs p2a {sorted(B)})")
    rows = zero_kg_suite_rows(A, B, trained_translate, R, Rg, translate=translate)
    for row in ZERO_KG_FIELD_ROWS:
        ha, pb = by_es(head, row), by_es(p2a, row)
        ca, cb = {}, {}
        for i, (x, y) in enumerate(zip(ha, pb)):
            ca.update({(i, k): v for k, v in field_episodes(x).items()})
            cb.update({(i, k): v for k, v in field_episodes(y).items()})
        if len(ha) != len(pb) or not ha:
            rows.append({"statistic": f"{row} completion", "row": row, "watch": False, "missing": True,
                         "why": f"{row}: head {len(ha)} vs p2a {len(pb)} stamps"})
            continue
        rows.append(zero_kg_completion_row(row, ca, cb, list(range(1, len(ha) + 1))))
    return rows, by_es(head, "side22"), by_es(p2a, "side22")


def p3a_pilot_rates(repo: Repo, pp: dict, out_lines: list, R: dict, Rg: dict, s22, trained_translate,
                    provenance: list, translate: str = "ratio_drop") -> None:
    """STEPC_RULES V.2 / PLAN Rule "payload a-priori lines": recompute the p3a terms of the run sum from the pilot.

    ``pp`` (threshold_inputs v1 ``p3a_pilot``; STAMP = ``{stamp, eval_sha}`` or a stamp path)::

        name: <label, e.g. the pilot checkpoint at N_p>
        its_past_p2a: <pilot iteration - the p2a head's iteration>
        level90: {suite@1: STAMP, suite@2: STAMP, field_light@1: STAMP, field_light@2: STAMP}
        zero_kg: {head: {<row>@<es>: STAMP, ...}, p2a: {<row>@<es>: STAMP, ...}}   # the 0-kg row list, both sides

    Each ``p3a.stable90.*`` rate becomes ``payload_apriori_rate(line, mu_pilot, sigma_hat_p2a, SE)`` (variance
    2 sigma_hat(p2a)^2 + SE^2; SE stood in by sigma_hat(p2a) where the statistic has none -- disclosed per line).
    The paired 0-kg line's rate becomes the sum over its judged DECISIVE statistics of ``paired_rate(margin,
    SE_diff)`` on the PILOT pair (margin = ``zero_kg_margin`` with the recorded X12 - X08 drift scaled by
    ``its_past_p2a`` / 5000), its statistics in the map's ``translate`` mode (v1.2: track_err; CHARTER 2026-10-22).
    Both then count in the p3a stage's sum (``summarize``)."""
    vals = pilot_stable90_values(_row_stamps(repo, pp["level90"]))
    name = pp.get("name", "")
    for x in out_lines:
        if not x["id"].startswith("p3a.stable90."):
            continue
        key = x["id"][len("p3a.stable90."):]
        if key not in vals or vals[key][0] is None:
            x["rate_source"] = f"p3a_pilot {name}: no pilot value for {key} (rate stays pending)"
            continue
        mu, se = vals[key]
        sig = x.get("sigma_hat_p2a") or 0.0
        hib = x["direction"] == ">="
        x.pop("pending", None)
        x.update({"rate": payload_apriori_rate(x["line"], mu, sig, sig if se is None else se, hib),
                  "mu_pilot": mu, "se_pilot": se, "se_stand_in": se is None,
                  "pilot_passes": bool((mu >= x["line"]) if hib else (mu <= x["line"])),
                  "rate_source": f"p3a_pilot {name}: Phi((line - mu_pilot) / sqrt(2 sigma_hat_p2a^2 + SE^2))"
                                 + (" (SE stood in by sigma_hat_p2a)" if se is None else "")})
    zl = next(x for x in out_lines if x["id"] == "p3a.capability.zero_kg_paired")
    its = float(pp["its_past_p2a"])
    rows, s22h, s22p = pilot_zero_kg_rows(_row_stamps(repo, pp["zero_kg"]["head"]),
                                          _row_stamps(repo, pp["zero_kg"]["p2a"]), trained_translate, R, Rg,
                                          translate=translate)
    ch = side22_read(s22, s22h[0])["class"] if s22h else None
    cp = side22_read(s22, s22p[0])["class"] if s22p else None
    judged = zero_kg_judge(rows, zl["drift_by_statistic"], its, side22_head=ch, side22_p2a=cp)
    if ch is None or cp is None:
        judged["verdict"] = "FAIL"
        judged["fails"].append("side22 stamp missing on the head or its p2a")
    per = []
    for r in judged["rows"]:
        if r.get("watch") or r.get("se_diff") is None or not isinstance(r.get("margin"), dict):
            continue
        per.append({"statistic": r["statistic"], "row": r["row"], "se_diff": r["se_diff"],
                    "margin": r["margin"]["margin"], "binds": r["margin"]["binds"], "worse_by": r["worse_by"],
                    "margin_used": r.get("margin_used"), "rate": paired_rate(r["margin"]["margin"], r["se_diff"])})
    unjudged = [f"{r['statistic']} {r['row']}" for r in judged["rows"]
                if not r.get("watch") and not isinstance(r.get("margin"), dict)]
    used = [p["margin_used"] for p in per if p["margin_used"] is not None]
    zl["rate"] = total(p["rate"] for p in per)
    zl["rate_source"] = (f"p3a_pilot {name}: sum over the judged decisive statistics of Phi(-margin / SE_diff) on "
                         f"the pilot pair, its_past_p2a = {its:g}")
    zl["pilot_pair"] = {"verdict": judged["verdict"], "fails": judged["fails"], "per_statistic": per,
                        "unjudged": unjudged, "side22": judged["side22"],
                        "max_margin_used": max(used) if used else None}
    provenance.append({"stage": "p3a", "p3a_pilot": name, "its_past_p2a": its,
                       "rows": sorted(set(pp["level90"]) | set(pp["zero_kg"]["head"]))})


#: the sweep's statistics per bin (DESIGN 3.2 p2a row): the ratio (a world head reads the body rows through
#: ``body_hold``), tripod and the 50 N steps_per_m (``metric_class: translate``); ``by_category.<bin>.<key>.median``
SWEEP_KEYS = ("ratio", "tripod", "steps_per_m")


def sweep_watch_lines(repo: Repo, anchors: dict, provenance: list) -> list:
    """The WATCH lines of the direction sweep (``dirsweep__omni_body``; DESIGN 3.2 p2a: "per-bin |r-1| line; max
    adjacent-bin jump in ratio, tripod and steps_per_m <= the anchors' max + range"; PLAN retired the sweep HALT).

    ``anchors``: ``{"s<seed>_es<k>": STAMP}`` (``t4_stepC_read`` C5). A seed's value per bin is the mean of its env
    seeds' medians. Per bin: ``|ratio - 1| <= tracking_line(seeds)`` (R = 0: no per-bin resolution exists). Per
    statistic: the max ADJACENT-bin jump (circular, d338 -> d000 included) ``<= max over the seeds + their range``.
    Every line is ``kind: watch``: rate None, excluded from every sum, never decisive, never self-checked."""
    by_seed: dict = {}
    order = None
    for key, entry in sorted(anchors.items()):
        seed, sep, es = str(key).partition("_es")
        if not sep or not seed.startswith("s") or not es.isdigit():
            raise ValueError(f"sweep_anchors: key {key!r} is not s<seed>_es<env seed>")
        d = _stamp_of(repo, entry)
        bc = load_json(d / "scenario_metrics.json")["aggregate"]["suite"]["by_category"]
        bins = list(bc)
        order = order or bins
        if bins != order:
            raise ValueError(f"sweep anchor {key}: its bins differ from the first anchor's")
        by_seed.setdefault(seed, []).append(
            {c: {k: ((bc[c] or {}).get(k) or {}).get("median") for k in SWEEP_KEYS} for c in bins})
        provenance.append({"stage": "p2a", "row": "dirsweep__omni_body", "sweep_anchor": key,
                           "stamp": entry["stamp"] if isinstance(entry, dict) else str(entry)})

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    seeds = sorted(by_seed)
    pooled = {s: {c: {k: mean([v[c][k] for v in by_seed[s]]) for k in SWEEP_KEYS} for c in order} for s in seeds}
    common = {"stage": "p2a", "kind": "watch", "class": "watch", "row": "dirsweep__omni_body", "direction": "<=",
              "rate": None, "excluded_from_sum": True, "value_set": "both", "known_bad": {}, "seeds_pass": None,
              "watch": True, "anchor_seeds": seeds}
    out = []
    for c in order:
        rs = [pooled[s][c]["ratio"] for s in seeds]
        if any(r is None for r in rs):
            continue
        out.append({**common, "id": f"p2a.watch.sweep.abs_dev.{c}", "statistic": f"sweep bin {c} |ratio - 1|",
                    "key_path": f"aggregate.suite.by_category.{c}.ratio.median", "seed_ratios": dict(zip(seeds, rs)),
                    "line": tracking_line(rs),
                    "line_rule": "max|x - 1| + range|x - 1| over the anchor seeds (R = 0)"})
    for k in SWEEP_KEYS:
        jumps = {}
        for s in seeds:
            v = [pooled[s][c][k] for c in order]
            if v and all(x is not None for x in v):
                jumps[s] = max(abs(v[i] - v[i - 1]) for i in range(len(v)))
        if not jumps:
            continue
        j = list(jumps.values())
        out.append({**common, "id": f"p2a.watch.sweep.max_adjacent_jump.{k}",
                    "statistic": f"sweep max adjacent-bin jump in {k} (circular)",
                    "key_path": f"aggregate.suite.by_category.<bin>.{k}.median", "seed_values": jumps,
                    "line": max(j) + (max(j) - min(j)), "line_rule": "the anchor seeds' max + range"})
    return out


def summarize(res: dict, watch_ids: set) -> dict:
    """Per-stage and run sums for the standard and recommended value sets (decisive reads only).
    ``run_sum`` uses each read's PLAN model; ``run_sum_sensitivity`` swaps in the Jeffreys
    predictive (beta-binomial / gamma-Poisson) where a line carries one -- zero-count and zero-miss
    lines are 0 under the plain model but not under the predictive one.

    ``degenerate_mode`` 'unvalidated' (v1.1 on; CHARTER 2026-10-21 C): a degenerate fixed-t line (rate None,
    ``rate_status`` DEGENERATE) is NOT summed and is counted explicitly -- ``n_excluded_degenerate`` per stage (a
    stage whose every read is degenerate still gets a row) and ``excluded_degenerate`` {n, ids, plain_read_total}
    per value set, so the arithmetic says what it leaves out. A WATCH id is excluded as WATCH first (recommended
    set). In 'legacy_zero' mode the output is v0 / v1's exactly."""
    explicit = res.get("degenerate_mode") == "unvalidated"
    plain = res.get("degenerate_plain_reads") or {}
    out = {}
    for which in ("standard", "recommended"):
        by, sens, deg = {}, {}, {}
        for x in res["lines"]:
            if x.get("value_set") not in (which, "both") or x.get("excluded_from_sum"):
                continue
            if which == "recommended" and x["id"] in watch_ids:
                continue
            if x.get("rate") is None:
                if explicit and x.get("rate_status") == DEGENERATE:
                    deg.setdefault(x["stage"], []).append(x)
                continue
            by.setdefault(x["stage"], []).append(x["rate"])
            rs = x.get("rate_sensitivity") or {}
            alt = rs.get("betabinom_jeffreys", rs.get("nb_jeffreys"))
            sens.setdefault(x["stage"], []).append(x["rate"] if alt is None else alt)
        stages = list(by) + [k for k in deg if k not in by]
        by_stage = {}
        for k in stages:
            v = by.get(k, [])
            by_stage[k] = {"sum": total(v), "union": union(v), "n_reads": len(v),
                           "sum_sensitivity": total(sens.get(k, []))}
            if explicit:
                by_stage[k]["n_excluded_degenerate"] = len(deg.get(k, []))
        out[which] = {"by_stage": by_stage,
                      "run_sum": total(r for v in by.values() for r in v),
                      "run_union": union(r for v in by.values() for r in v),
                      "run_sum_sensitivity": total(r for v in sens.values() for r in v)}
        if explicit:
            ids = {k: [x["id"] for x in v] for k, v in deg.items()}
            pr = [plain.get(f"{x['id']}|{x.get('value_set')}") for v in deg.values() for x in v]
            out[which]["excluded_degenerate"] = {
                "n": sum(len(v) for v in deg.values()), "ids": ids,
                "plain_read_total": total(pr),
                "note": "degenerate fixed-t lines (sigma_hat 0): rate None, NOT in run_sum / run_union / n_reads / "
                        "run_sum_sensitivity; plain_read_total = what the plain 0.0 / 1.0 reads would have added "
                        "(CHARTER 2026-10-21 decision C)"}
    return out


def self_check(res: dict, watch_ids: set) -> list:
    """PLAN self-checks: every recorded seed passes each decisive line (recommended set); each decisive
    line names a failing known-bad or is tagged ``unvalidated`` (the tag is added here, every value set)."""
    fails = []
    for x in res["lines"]:
        if x.get("kind") == "watch":            # the v1 sweep lines: reported, never decisive, never self-checked
            x["decisive"] = False
            continue
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
           "zero_kg_statistic": ("tracking = one-sided ratio drop (p2a - head), R at the 99 % guard level; |r - 1| "
                                 "WATCH (CHARTER STOP-0 2026-10-16)" if res["zero_kg_translate"] == "ratio_drop" else
                                 "translate tracking = track_err <c>: |r_head - 1| - |r_p2a - 1| (an increase in "
                                 "tracking error fails), R = the 99 % guard R of cat.<c>.ratio, SE_diff and drift on "
                                 "|r - 1| (CHARTER 2026-10-22, thresholds v1.2); move_turn = one-sided ratio drop "
                                 "(p2a - head), R at the 99 % guard level (CHARTER STOP-0 2026-10-16); WATCH: the raw "
                                 "translate ratio drop + |r - 1| move_turn"),
           "zero_kg_row_list": ZERO_KG_ROW_LIST,
           "trained_categories_required": "every suite gate row passes --trained-categories from the stage's composed "
                                          "env; a stamp without it is INVALID (CHARTER STOP-0; gate_read.py gate 0)",
           "floors_apply_to_D1_D2": False,
           "floors_apply_to_D1_D2_note": "D1/D2 rows route the stand net beside a FROZEN T*: the walker floors were "
                                         "already read on T* at its own stage; the stand gates are G1-G5 / K1-K6 only "
                                         "(G4 / K5-G4 carry the walls). PLAN left-for-implementation item 15.",
           "field_context_p2a": res["field_context_p2a"]}
    if res["degenerate_mode"] == "unvalidated":      # v1.1 on (absent from the frozen v0 / v1 files)
        doc["degenerate_fixed_t"] = {
            "mode": res["degenerate_mode"], "rate_status": DEGENERATE,
            "rule": "a fixed-t line whose sigma_hat is 0 (every recorded seed equal, R = 0) has no modelled chance "
                    "rate: rate None + rate_status, excluded from every sum and counted "
                    "(sums.<set>.excluded_degenerate, by_stage.<stage>.n_excluded_degenerate); gate_read / the HALT "
                    "record carry the status, never a 0.0 (CHARTER 2026-10-21 decision C)",
            "plain_reads": res["degenerate_plain_reads"]}
    # atomic: a temp file in the same dir + os.replace (a running gate may read thresholds.json)
    fd, tmp = tempfile.mkstemp(dir=str(Path(a.out).resolve().parent), prefix=f".{Path(a.out).name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(doc, indent=1, default=_json_default))
        um = os.umask(0)
        os.umask(um)
        os.chmod(tmp, 0o666 & ~um)                    # what write_text gave (mkstemp is 0600)
        os.replace(tmp, a.out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f"wrote {a.out}: {len(res['lines'])} lines; run sum standard {doc['sums']['standard']['run_sum']:.3f} "
          f"recommended {doc['sums']['recommended']['run_sum']:.3f}; self-check failures {len(fails)}"
          + (f"; degenerate lines excluded (rate None): standard {doc['sums']['standard']['excluded_degenerate']['n']}"
             f", recommended {doc['sums']['recommended']['excluded_degenerate']['n']}"
             if "degenerate_fixed_t" in doc else ""))
    for f in fails:
        print("  SELF-CHECK:", f)
    return 1 if (a.self_check and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
