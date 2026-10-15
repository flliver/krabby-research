#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""``D00`` -- the measured eval resolution the candidate ladder's decision rule is scored against.

One job: turn a list of ALREADY-RECORDED eval run dirs into ``resolution.json``, the record of how
big a difference the eval can actually see. Nothing here trains, evaluates or boots Kit; it reads
saved JSON and saved ``raw/*.npz`` and writes one file.

THE CONSTRUCTION (plan decision 6)
----------------------------------
A **paired episode bootstrap**, ``B`` = 2000 draws, paired on ``env_index``. One draw is one
index list of length n drawn with replacement from the env indices both heads of a pair have; the
SAME list is applied to both heads, and each head's aggregate statistic is recomputed over its own
valid subset of those episodes -- exactly the estimator the harness reports (a median over the
episodes that produced the quantity), never a re-definition of it. ``delta`` = candidate minus
control per draw; the metric's resolution ``R_eval`` is the half-width of that bootstrap
distribution about its own centre, so a genuine head-to-head difference does not inflate it:

    half_width(x, p) = (quantile(x, p) - quantile(x, 1 - p)) / 2

with ``p`` = 0.975 for a two-sided 95 % block member and ``p`` = 0.99 for a one-sided 99 % guard.
Those two levels are what make the null candidate's all-TIE frequency 0.95**2 * 0.99**4 ~ 0.87
(decision 5): a two-sided block test against the 95 % half-width spends 2.5 % per tail, and a
one-sided guard test against the centred 99 % quantile spends 1 %.

**Pairing is CROSS-HEAD** (decision 6): the resolution a candidate-vs-control decision needs is the
spread of a difference between two DIFFERENT heads evaluated the same way -- here 1c-004 vs 2b and
1c-004 vs 2a-reeval, per row, with 1c-004 the control. ``R_eval`` per metric is the MAX over the
pairs that produced one (the resolution must cover the worse-behaved pair: understating ``R`` makes
the rule read noise as signal, which is the expensive error). Every pair's own half-width is kept
in ``per_pair`` so the max is auditable. A pair of two rows carrying the SAME checkpoint is a
**process-noise floor** pair instead (one head, two evals) and is reported under ``process_floor``,
never as the candidate ``R``.

**Blocks are bootstrapped directly.** ``z_F = mean_i(delta_i / R_eval,i)`` is formed per draw from
that draw's own deltas and the fixed ``R_eval,i``, and ``blocks.F.R_eval`` is the 95 % half-width of
``z_F`` itself -- not a propagated sum of member half-widths. ``F_abs``/``T_abs`` are the same
construction on the absolute-units members (decision 5's cap-candidate blocks).

**facing_closed does not exist in the recorded JSON.** ``heading_error_initial_abs`` and
``facing_closed`` were added to the harness on 2026-09-18, so every row recorded before that yields
``{hold: None}`` from its aggregate. The fraction is recovered by RE-SCORING each episode's saved
``raw/episode_*.npz`` through ``gait_eval.metrics.facing_metrics`` over the same whole-hold window
the harness used (``source`` "raw"). That window is replicated from ``report._window_mask``, and the
replication is VERIFIED at run time: the re-score must reproduce every facing key the recorded
per-episode JSON already carries, per episode and per hold, to ``MASK_CHECK_TOL``. If it does not,
the window is wrong and the run refuses -- a wrongly-masked ``facing_closed`` is the one number here
nothing downstream could catch. Fallbacks, in order: the per-episode ``facing_closed`` the new
harness writes ("json"), a per-episode fraction rebuilt from ``heading_error_initial_abs`` ("json"),
and -- only with ``--ring-proxy``, or for a row with no raws at all -- the ``heading_error_max_abs``
proxy ("proxy"). A "proxy" row is NEVER paired against a non-proxy row: the mixed pair raises.

AMENDMENT to decision 6's sentence "every ring row uses the documented proxy ... regardless of raw
presence" (orchestrator decision, 2026-09-19): the ring rows are re-scored too, and the proxy is a
fallback behind ``--ring-proxy``. Three reasons. (a) All twelve D00 inputs have 100 raws, so the
re-score is available. (b) The ``heading_error_max_abs`` stand-in for the initial error was MEASURED
wrong, not merely imprecise, by the 2026-09-18 review -- it flips sign against the per-episode metric
on the travel holds -- which is why ``run_phases.facing_closed_by_hold`` dropped it. (c) Every eval
the harness writes from now on carries per-episode ``facing_closed``, i.e. the metric definition, so
a milestone comparing a new ring row against the recorded 2b ring row needs BOTH sides on that one
definition; a proxy D00 would put the recorded side on the other one. On the three F holds the two
agree closely (the turn holds start near 1.57 rad, where the peak error IS the initial error:
measured 1c-004 ring turn_l90 0.7061 re-scored vs 0.7067 proxied), so this changes no verdict -- it
removes a source mismatch that would have been illegal at the first milestone.

:func:`facing_closed_by_hold_from_raws` and :func:`summarize_run_dir` export that re-score for the
DRIVER: a milestone that reads a recorded row's ``run_dir`` must go through them, or two of the four
guards (the conditional turn survivals, injected in memory only) and the whole F block (the closed
fractions, absent from a pre-2026-09-18 aggregate) come back MISSING.

WHAT IS AND IS NOT IN HERE
--------------------------
``seed_terms`` is always ``{}``. ``D00`` is the EVAL-ONLY resolution; the seed terms (``D00b`` from
B0's own seed pair, ``D00c-<chain>`` per chain) are measured later and written by the driver into the
ledger, not here. The combination rules are still defined here, once, as pure functions
(:func:`single_block_R`, :func:`single_guard_R`, :func:`pooled_block_R`, :func:`pooled_guard_R`,
:func:`block_seed_term`) so the driver imports them instead of re-deriving decision 6's arithmetic.

RULE VERSION 2 -- ``D00s``, THE SUITE RESOLUTION (per-category eval plan 2026-09-21, D5/D9/D10)
-----------------------------------------------------------------------------------------------
A ``decision_rule`` with ``rule_version: 2`` carries per-category templates, not flat blocks, and is
scored on the ``suite__omni`` row (:data:`SCREEN_ROW`; the recorded rows keep
:data:`LEGACY_SCREEN_ROW`). :func:`resolve` then

* expands the rule with the SHARED ``eval_categories.expand_rule`` (loaded by path, never
  re-implemented) at ``decisive = present = decisive_at`` -- the UNION of both unit systems, so the
  record carries an R for ``F, F_mt, T`` and ``F_abs, F_mt_abs, T_abs``, the seven ``survival``
  guards, ``cat.move_turn.ratio``, ``cat.move_turn.achieved_speed`` and ``cat_tripod_min``;
* reads each ``role: pair`` input from the eval stamp the driver recorded for it (the entry's
  ``stamp``; a missing one is a :class:`ResolutionError` naming the head) and every other
  checkpoint-shaped input through :func:`resolve_eval_dir`; every SUITE stamp must pass the
  completion-marker freshness rule (``complete.json``, a full raw set, the entry's checkpoint);
* draws the paired bootstrap CLUSTERED ON THE WITHIN-WAVE ENV: a replicate resamples env ids and
  takes every wave's episode of each drawn env, so each hold keeps its n, each paired category its
  hand split, and the same startup-randomised robot stays correlated across categories. A legacy
  row (one wave) reduces to the episode draw above, bit for bit;
* runs the null candidate on the RATIO system only (its three blocks and nine guards), clustered
  the same way, with every ``survival`` guard floored at :func:`binomial_guard_floor` and no seed
  term, and records ``acceptance = {ok, failures}``: that calibrated null inside
  ``[nominal - 0.03, 0.98]`` AND a finite positive ``R_eval`` for every non-empty block, every block
  member and every non-survival guard in both unit systems (survival guards pass on the floor);
* copies ``D00``'s ``light``/``ring`` ``by_row`` tables (``by_row_source: D00``) and its frozen
  ``decision_rule`` as ``by_row_rule`` -- milestones score those rows on the v1 rule unchanged.

``BORROW`` and ``--curriculum-runs`` are retired under rule version 2 (the suite measures both
hands). :func:`resolve` never writes; :func:`write_resolution` does.

The gait-motion-quality campaign (2026-09-23) adds three things, all inert for every earlier rule:
a block ``Q`` (:data:`BLOCK_ORDER`); a per-metric SIGN (``MetricSpec.sign`` =
``eval_categories.metric_sign``: -1 for the lower-is-better motion-quality keys), applied to every
delta the bootstrap and the null form, so a signed block ``z`` and a guard's WORSE read "positive =
better" -- the per-metric half-widths are sign-invariant, the block half-width is not; and a
SEPARATE guard spec for a guard whose metric is also a block member (:func:`specs_from_rule`),
recorded under ``<mid>@guard`` (``eval_categories.guard_key``) with its own one-sided 99 % half-width.
:func:`member_spread_check` is the pre-launch zero-spread check behind the driver's
``--q-spread-check``.

Usage::

    resolution.py --campaign-dir <ladder dir> [--candidates <candidates.yaml>] [--out resolution.json]
    resolution.py --campaign-dir <dir> --eval-sha <sha> --reeval-stamps '{"2b": "<stamp>", ...}' \\
        --supersedes D00 --d00 <dir>/resolution.json          # a rule_version 2 (D00s) manual run

``--candidates`` defaults to ``<campaign-dir>/candidates.yaml`` and supplies ``resolution_inputs``
(the run dirs) and ``decision_rule`` (frozen verbatim into the record). The driver calls
:func:`main` by import-by-path when ``resolution.json`` is absent; under rule version 2 it calls
:func:`resolve` in process instead (the CLI above is the manual mirror).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PKG = HERE.parents[1]
FWD_PKG = PKG.parent / "crab_hex_forward_task"
GAIT_SCRIPTS = FWD_PKG / "scripts"          # the dir `from gait_eval import metrics` resolves against
RUN_PHASES_PY = HERE / "run_phases.py"
#: The shared category helpers (``expand_rule``, ``composite_guards``, the category tables). Loaded
#: BY PATH like everything else here: a dotted import would run the package ``__init__``'s Isaac
#: ``import_packages``.
EVAL_CATEGORIES_PY = PKG / "eval_categories.py"


def _load(path: Path, name: str):
    """Import ``path`` as module ``name``, reusing an identical module already in ``sys.modules``.

    The reuse clause matters because the ladder driver imports this module AND ``run_phases.py`` by
    path: without it a second load would hand out a second module object with its own
    ``CURRENT_CHILD``. Nothing here touches that state (only pure functions of ``run_phases`` are
    used), but a driver that loads ``run_phases`` under a different name still gets its own copy --
    documented rather than defended against, since resolution never signals a child.
    """
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load(RUN_PHASES_PY, "crab_hex_omni_run_phases")   # EVAL_CARRY(+DEFAULTS), summarize, _manifest_hold_s


@lru_cache(maxsize=1)
def _ec():
    """``eval_categories``, loaded by path on first use (it is pure, but a legacy-only reader of this
    module has no reason to load it)."""
    return _load(EVAL_CATEGORIES_PY, "crab_hex_omni_eval_categories")

SCHEMA = 1
#: Bootstrap draws (decision 6). 2000 is the plan's number; the 95 %/99 % quantiles of a 2000-sample
#: distribution have a Monte-Carlo standard error well under a percent of the half-width.
BOOTSTRAP_B = 2000
#: Default RNG seed. Recorded in the output; a rerun with the same seed reproduces the file byte for
#: byte apart from ``created``.
RNG_SEED = 20260919
BLOCK_LEVEL = 0.95        # two-sided, per block and per block member
GUARD_LEVEL = 0.99        # one-sided (only a regression is a verdict), per guard
#: Below this many episodes with a usable value the half-width is reported with ``below_min_n`` set
#: and a note, never dropped. The plan's "n >= 50 per hold" describes why these input rows were
#: chosen; making it an exclusion rule could silently delete a GUARD, and a MISSING guard is a
#: REJECT with ``confidence: uncertain`` (decision 5) -- a loud number beats a missing one.
MIN_PAIRED_HOLD_N = 50
#: Fewer than this many usable episodes on either side and NO half-width is reported: a spread
#: estimated from one or two episodes is not a wide number, it is a meaningless one, and a guard
#: carrying a meaningless R is either permanently blind (R too wide) or a spurious REJECT (too
#: narrow). ``R_eval: null`` instead hands the caller decision 5's MISSING rules, which are loud.
MIN_R_EVAL_N = 5
#: Fewer than this many finite bootstrap draws and there is no half-width either.
MIN_BOOTSTRAP_N = 2
FACING_CLOSED_MIN_INITIAL_RAD = 0.15     # == gait_eval.metrics.FACING_CLOSED_MIN_INITIAL_RAD
#: The re-scored facing keys must reproduce the recorded ones to this tolerance (measured deviation
#: on the twelve D00 inputs: exactly 0.0 on every key, every episode, every hold).
MASK_CHECK_TOL = 1e-9
#: The row every screen is scored on under rule version 2 (per-category plan D1/D10: the category
#: suite). Its metrics and blocks are the flat ``metrics``/``blocks`` of the record; other rows land
#: in ``by_row``.
SCREEN_ROW = "suite__omni"
#: The screen row of the recorded rule-version-1 records (``D00``, ``resolution.json``): a v1 rule is
#: still resolved there, and its ``by_row`` tables (light/ring) are what milestones score on.
LEGACY_SCREEN_ROW = "slow__omni"
F_HOLDS = ("turn_l90", "turn_r90", "move_turn")
T_HOLDS = ("fwd", "lat_l", "diag", "back")
#: Right-handed holds: WATCH only (decision 5), and present only on the curriculum row. They borrow
#: their left-handed twin's R until ``--curriculum-runs`` re-derives them (decision 6). Rule
#: version 1 only: the suite measures both hands, so rule version 2 retires both mechanisms.
BORROW = {"lat_r": "lat_l", "move_turn_r": "move_turn"}
CURRICULUM_ROW = "curriculum__omni"
#: Every block name, in the order records and lines enumerate them. The set a record actually
#: carries is always DERIVED from the rule passed (a v1 rule has F/T/F_abs/T_abs, the expanded v2
#: union all six, the gait-motion-quality campaign's F/F_mt/T/Q, the clock-deprecation campaign's
#: F/F_mt/T/RIG -- plan step 4, 2026-09-23; the stride/posture screen's LEN, 2026-09-25); this
#: tuple only orders them.
BLOCK_ORDER = ("F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs", "Q", "RIG", "LEN")
#: The null candidate's scored blocks under a v1 rule (decision 5's two blocks).
LEGACY_SCORED_BLOCKS = ("F", "T")
#: Ids that exist only because the legacy holds are nested in one episode (``turn_l90`` given
#: ``back``, ``turn_r90`` given ``turn_l90``). Undefined on a suite row, where no hold follows another.
CONDITIONAL_SURVIVAL_IDS = ("turn_survival_l90", "turn_survival_r90")
#: ``summarize()['cat'][c]`` keys that are counts, not per-episode quantities (never resampled).
CAT_COUNT_KEYS = ("n", "n_env")
#: The binomial floor's clip on the pooled survival (D5): a survival saturated on both heads floors
#: at ``z_0.99 * sqrt(2 * 0.95 * 0.05 / n)`` instead of 0.
BINOM_P_CLIP = (0.05, 0.95)
#: The null band's lower slack below the nominal all-TIE rate (D9 item 4).
NULL_BAND_SLACK = 0.03
NULL_BAND_HI = 0.98
#: The completion marker a suite stamp's ``write_run`` writes LAST (D10's freshness rule).
COMPLETE_MARKER = "complete.json"
#: Cited, not applied (plan decision 6: "the bootstrap supersedes when it resolves"). Measured on the
#: sideways formation campaign's paired episode bootstraps.
CHARTER_FLOORS = {"completion": 0.10, "tripod_walking_dir": 0.056}
#: The citation is spelled descriptively, never as a dated path: a driver's source may carry no
#: campaign-directory literal (the layout test's rule), so an exact path belongs in
#: ``candidates.yaml`` (``decision_rule.charter_cite`` overrides this string).
CHARTER_CITE = ("the sideways formation campaign's CHARTER.md:227 -- paired episode bootstraps, "
                "n=100: completion +-0.06..0.10, walking tripod +-0.016..0.056")


class ResolutionError(RuntimeError):
    """A refusal: a mixed facing source, a broken window replication, a missing screen row."""


# --------------------------------------------------------------------------- pure decision-6 maths
def half_width(samples: Sequence[float] | np.ndarray, p: float) -> float | None:
    """Centred half-width of ``samples`` at quantile ``p`` (> 0.5): ``(q(p) - q(1-p)) / 2``.

    Centred, so it measures the bootstrap distribution's SPREAD and is invariant to the genuine
    head-to-head offset a cross-head pair carries. ``None`` when fewer than
    :data:`MIN_BOOTSTRAP_N` finite samples survive.
    """
    arr = np.asarray(samples, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_BOOTSTRAP_N:
        return None
    lo, hi = np.quantile(arr, [1.0 - p, p])
    return float((hi - lo) / 2.0)


def quantile_p(level: float, *, one_sided: bool) -> float:
    """The ``p`` :func:`half_width` needs for a test at ``level``.

    One-sided at 99 % -> ``p`` 0.99 (the centred 99th percentile; a regression test against it
    spends 1 %). Two-sided at 95 % -> ``p`` 0.975 (2.5 % per tail).
    """
    return float(level) if one_sided else 1.0 - (1.0 - float(level)) / 2.0


def classify(z: float | None, R: float | None, *, one_sided: bool = False) -> str:
    """Decision 5's per-block / per-guard verdict: ``BETTER`` / ``WORSE`` / ``TIE`` / ``MISSING``.

    ``BETTER`` if ``z > R``, ``WORSE`` if ``z < -R``, else ``TIE`` (the boundary is a TIE, both
    sides). ``one_sided`` suppresses ``BETTER``: a guard can only be a regression, never a reason to
    adopt. ``MISSING`` when either number is unavailable -- the caller applies decision 5's MISSING
    rules (a guard MISSING is a ``decided`` REJECT; a block member MISSING leaves the block mean).

    This is the one place the comparison lives; ``run_candidates.decide`` composes it rather than
    re-deriving it, and the null-calibration test drives it directly.
    """
    if z is None or R is None or not math.isfinite(float(z)) or not math.isfinite(float(R)):
        return "MISSING"
    z, R = float(z), float(R)
    if z < -R:
        return "WORSE"
    if z > R and not one_sided:
        return "BETTER"
    return "TIE"


def single_block_R(r_eval: float | None, r_seed: float | None) -> float | None:
    """``R_F = max(R_F^eval, R_F^seed)`` -- a single-seed block threshold (decision 6)."""
    vals = [float(v) for v in (r_eval, r_seed) if v is not None and math.isfinite(float(v))]
    return max(vals) if vals else None


def single_guard_R(r_eval: float | None, r_seed: float | None) -> float | None:
    """``max(R_eval,i, R_seed,i)`` -- a single-seed guard threshold (decision 6)."""
    return single_block_R(r_eval, r_seed)


def pooled_block_R(r_eval: float | None, r_seed: float | None) -> float | None:
    """``R_F^pooled = max(R_F^eval / sqrt(2), R_F^seed)`` (decision 6).

    Two seeds average the EVAL noise down by ``sqrt(2)``; they do not average the seed term down at
    all, because the seed term is the between-seed spread the pair is made of.
    """
    a = None if r_eval is None or not math.isfinite(float(r_eval)) else float(r_eval) / math.sqrt(2.0)
    return single_block_R(a, r_seed)


def pooled_guard_R(r_eval: float | None, r_seed: float | None,
                   r_binom: float | None = None) -> float | None:
    """``R_pooled,i = sqrt((R_eval,i / sqrt(2))**2 + R_seed,i**2)`` (decision 6).

    ``r_binom`` (per-category plan D5; ``survival`` guards only) is the guard's UNPOOLED binomial
    floor (:func:`binomial_guard_floor` at the pooled ``p_bar`` and the smallest ``n_c``); the pooled
    guard is then ``max(that, r_binom / sqrt(2))``, so a survival saturated on every screen is scored
    against a floor rather than going MISSING. ``None`` (the default) is today's rule exactly.
    """
    a = 0.0 if r_eval is None or not math.isfinite(float(r_eval)) else float(r_eval) / math.sqrt(2.0)
    b = 0.0 if r_seed is None or not math.isfinite(float(r_seed)) else float(r_seed)
    floor = (None if r_binom is None or not math.isfinite(float(r_binom))
             else float(r_binom) / math.sqrt(2.0))
    if a == 0.0 and b == 0.0:
        return floor if floor else None
    base = float(math.sqrt(a * a + b * b))
    return base if floor is None else max(base, floor)


def binomial_guard_floor(p_bar: float | None, n: int | float | None, *,
                         level: float = GUARD_LEVEL) -> float | None:
    """``R_binom = z_level * sqrt(2 p(1 - p) / n)`` with ``p = clip(p_bar, 0.05, 0.95)`` (plan D5).

    The one-sided test of a DIFFERENCE of two binomial proportions, each on ``n`` episodes, at
    ``level``: the floor a ``survival`` guard is scored against when its paired ``R_eval`` is 0 or
    missing (both heads saturated) or below the unpaired spread. ``p_bar`` is the mean of the two
    survivals compared and is clipped, so a guard at 1.00 vs 1.00 floors at 0.0717 (n = 100) /
    0.0507 (n = 200) instead of 0. ``None`` without a ``p_bar`` or a positive ``n``.
    """
    if p_bar is None or n is None or not math.isfinite(float(p_bar)) or not float(n) > 0:
        return None
    p = min(BINOM_P_CLIP[1], max(BINOM_P_CLIP[0], float(p_bar)))
    z = NormalDist().inv_cdf(float(level))
    return float(z * math.sqrt(2.0 * p * (1.0 - p) / float(n)))


def cluster_draw(rng: np.random.Generator, n_clusters: int, waves: int, B: int) -> np.ndarray:
    """``(B, waves * n_clusters)`` episode positions of an ENV-CLUSTERED bootstrap (plan D10).

    Each replicate resamples the within-wave env ids ``0..E-1`` with replacement and takes EVERY
    wave's episode of each drawn env: position ``w * E + j`` of a replicate is episode
    ``w * E + env[j]``. So every wave -- one hold -- keeps exactly its ``E`` episodes, a paired
    category keeps its hand split, and the same startup-randomised robot (env ``j`` is the same
    robot in every wave, plan D1) stays correlated across holds and categories. With one wave this
    is exactly ``rng.integers(0, n, size=(B, n))``, the legacy episode draw, bit for bit.
    """
    env = rng.integers(0, int(n_clusters), size=(int(B), int(n_clusters)))
    if int(waves) == 1:
        return env
    offsets = np.arange(int(waves), dtype=env.dtype) * int(n_clusters)
    return (offsets[None, :, None] + env[:, None, :]).reshape(int(B), int(waves) * int(n_clusters))


def _wave_support(draw: np.ndarray, cols: Sequence[np.ndarray], per_wave: int) -> np.ndarray:
    """``draw`` restricted to the waves in which any of ``cols`` has a value (suite rows only).

    A clustered draw never moves an episode out of its wave, so every drawn position in a wave
    where all of ``cols`` are NaN reads NaN: dropping those positions leaves every NaN-ignoring
    reducer's value unchanged and saves the ~5-10x of resampling a category's columns over all ten
    waves. Returns ``draw`` itself when nothing is finite (every reduction is NaN either way).
    """
    finite = np.zeros(np.asarray(cols[0]).shape[0], dtype=bool)
    for c in cols:
        finite |= np.isfinite(c)
    waves = np.unique(np.flatnonzero(finite) // int(per_wave))
    if waves.size == 0 or waves.size * int(per_wave) == draw.shape[1]:
        return draw
    keep = (waves[:, None] * int(per_wave) + np.arange(int(per_wave))[None, :]).ravel()
    return draw[:, keep]


def block_seed_term(seed_deltas: dict[str, float | None], r_eval: dict[str, float | None]) -> float | None:
    """``R_F^seed = max(|z_F(pair)|, rms_i(delta_i^seed / R_eval,i) / sqrt(k))`` (decision 6).

    ``seed_deltas`` are one control pair's per-metric ``s3 - s2`` differences, ``r_eval`` the D00
    half-widths. The first term is the SIGNED block statistic of that pair, which can cancel to
    near zero when the seed deltas alternate in sign; the second is the independence-model spread of
    a k-metric block mean, which cannot. ``None`` when no member has both numbers.
    """
    std: list[float] = []
    for mid, d in sorted(seed_deltas.items()):
        R = r_eval.get(mid)
        if d is None or R is None or not math.isfinite(float(d)) or not math.isfinite(float(R)) or float(R) == 0.0:
            continue
        std.append(float(d) / float(R))
    if not std:
        return None
    k = len(std)
    z_pair = abs(sum(std) / k)
    rms = math.sqrt(sum(v * v for v in std) / k)
    return float(max(z_pair, rms / math.sqrt(k)))


# --------------------------------------------------------------------- the harness re-score helpers
@lru_cache(maxsize=1)
def _facing_metrics():
    """``gait_eval.metrics.facing_metrics``, imported lazily.

    Lazily because importing it puts ``$FWD/scripts`` on ``sys.path``, and module import here must
    stay free of side effects (``test_experiment_layout`` execs this file).
    """
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import metrics as M  # noqa: PLC0415 -- deliberate: see the docstring
    return M.facing_metrics


@lru_cache(maxsize=1)
def _gait_schedule():
    """``gait_eval.schedule``, imported lazily for the same reason as :func:`_facing_metrics`."""
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import schedule as S  # noqa: PLC0415 -- deliberate: see _facing_metrics
    return S


@lru_cache(maxsize=None)
def _manifest_scenario(sid: str) -> dict:
    """``{hold_labels, hold_s, fall_exclusion_s, prologue_s, kind, settle_hold_s}`` for one row.

    Parsed from the manifest rather than pinned, for the same reason ``rp._manifest_hold_s`` is: a
    manifest edit must not silently desynchronise the window this module cuts from the one the
    harness cut. Read through the harness's own ``gait_eval.schedule.load_manifest`` (plan D10's
    suite-aware readers): a ``category_suite`` row has no ``schedule:`` of its own -- the loader
    DERIVES the label union ``[settle] + category holds`` -- and ``hold_s`` is then the derived
    per-hold map of the category holds (the settle hold's duration is ``settle_hold_s``; it is never
    an episode's own hold). ``kind`` is ``schedule`` for every other row, whose values are exactly
    the ones the raw yaml gave.
    """
    S = _gait_schedule()
    try:
        scenarios, _ = S.load_manifest(rp.OMNI_MANIFEST)
    except S.ManifestError as exc:
        raise ResolutionError(f"{rp.OMNI_MANIFEST} does not load: {exc}") from exc
    for sc in scenarios:
        if sc.id != sid:
            continue
        labels = [str(h.label) for h in sc.schedule]
        if sc.suite is not None:
            hold_s = {str(k): float(v) for k, v in sc.suite.hold_s_by_label().items()}
            settle_hold_s = float(sc.suite.settle_hold_s)
        else:
            hold_s = {str(h.label): float(h.hold_s) for h in sc.schedule}
            settle_hold_s = None
        return {"hold_labels": labels, "hold_s": hold_s,
                "fall_exclusion_s": float(sc.get_default("fall_exclusion_s")),
                "prologue_s": float(sc.get_default("prologue_s")),
                "kind": str(sc.kind), "settle_hold_s": settle_hold_s}
    raise ResolutionError(f"scenario {sid!r} is not in {rp.OMNI_MANIFEST}")


def whole_hold_index(segment_id: np.ndarray, hold_idx: int, n_steps: int, *,
                     ended_in_fall: bool, fall_trim_steps: int) -> np.ndarray:
    """Step indices of one hold's WHOLE window, exactly as ``report._window_mask`` cuts it.

    ``report.score_episode`` computes a hold's facing over ``_window_mask(..., steady=ones)``: every
    logged step of the hold, up to the episode's ``n_steps``, minus the last ``fall_exclusion_s``
    when the episode ended in a fall (that half-second is a fall, not a gait sample). The saved
    ``raw/episode_XX.npz`` is already trimmed to this episode's ``n_steps``, so ``upper`` collapses
    to the array length before the fall trim -- which is why this reproduces the harness's window on
    a per-episode npz without the (T, E) buffer the harness held.
    """
    total = int(np.asarray(segment_id).shape[0])
    upper = min(int(n_steps), total)
    if ended_in_fall:
        upper = max(0, upper - int(fall_trim_steps))
    return np.flatnonzero((np.arange(total) < upper) & (np.asarray(segment_id) == int(hold_idx)))


#: Facing keys the recorded per-episode JSON already carries; the re-score must reproduce every one
#: of them or the window replication is wrong (see :func:`whole_hold_index`).
MASK_CHECK_KEYS = ("n_steps", "n_turn_steps", "heading_error_rms", "heading_error_final_abs",
                   "heading_error_max_abs", "settle_time_s", "yaw_progress_rate",
                   "net_yaw_rate_toward", "yaw_rate_ratio")


def rescore_facing(run_dir: Path, episodes: Sequence[dict], hold_labels: Sequence[str], dt: float,
                   fall_exclusion_s: float) -> tuple[dict[str, np.ndarray], float]:
    """``({hold: per-episode facing_closed}, worst mask deviation)`` from the saved raws.

    Each episode's npz is re-scored hold by hold through the harness's own
    :func:`gait_eval.metrics.facing_metrics` over :func:`whole_hold_index`, and every key the
    recorded JSON carries is compared against the re-scored one. The returned deviation is the worst
    absolute disagreement seen; the caller refuses above :data:`MASK_CHECK_TOL`.
    """
    facing_metrics = _facing_metrics()
    fall_trim = int(round(float(fall_exclusion_s) / float(dt)))
    out = {h: np.full(len(episodes), np.nan, dtype=np.float64) for h in hold_labels}
    worst = 0.0
    for pos, ep in enumerate(episodes):
        idx = int(ep["env_index"])
        npz = run_dir / "raw" / f"episode_{idx:02d}.npz"
        if not npz.exists():
            # A partial raw set is a refusal naming the stamp, never a FileNotFoundError (plan D10).
            raise ResolutionError(f"{run_dir}: raw/{npz.name} is missing -- a partial raw set "
                                  "(the eval was killed during its raw writes, or the files were "
                                  "pruned); re-run the eval rather than score part of it")
        with np.load(npz) as z:
            seg = z["segment_id"]
            her = z["heading_err"]
            wz = z["root_ang_vel_b"][:, 2]
            cmd = z["cmd_applied"][:, 2]
        ended_in_fall = str(ep.get("termination_reason")) in ("fall", "hard_fall")
        for hi, hold in enumerate(hold_labels):
            widx = whole_hold_index(seg, hi, int(ep.get("n_steps", 0)),
                                    ended_in_fall=ended_in_fall, fall_trim_steps=fall_trim)
            got = facing_metrics(her[widx], wz[widx], float(dt), wz_cmd_t=cmd[widx],
                                 min_initial=FACING_CLOSED_MIN_INITIAL_RAD)
            rec = ((ep.get("holds") or {}).get(hold) or {}).get("facing")
            if rec is not None:
                for key in MASK_CHECK_KEYS:
                    a, b = got.get(key), rec.get(key)
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        worst = max(worst, float("inf"))
                        continue
                    worst = max(worst, abs(float(a) - float(b)))
            closed = got.get("facing_closed")
            out[hold][pos] = np.nan if closed is None else float(closed)
    return out, worst


def proxy_facing_from_max(episodes: Sequence[dict], hold_labels: Sequence[str]) -> dict[str, np.ndarray]:
    """The plan's documented ring proxy, PER EPISODE: ``clip(1 - final/max(max_abs, 0.15), -1, 1)``.

    A stand-in for the initial error that the 2026-09-18 review measured to be not merely imprecise
    but SIGN-UNSTABLE (on the recorded 1a slow row it reads +0.30/+0.28/+0.29 on lat_l/diag/back
    where the per-episode metric reads -0.51/-0.39/-0.40, and it overstates turn_r90 threefold),
    which is why ``run_phases.facing_closed_by_hold`` dropped it. Kept here because the plan names
    it, reachable only behind ``--ring-proxy``, applied to BOTH sides of a pair, and labelled
    ``source: "proxy"`` so it can never be compared against a re-scored row.
    """
    out = {h: np.full(len(episodes), np.nan, dtype=np.float64) for h in hold_labels}
    for pos, ep in enumerate(episodes):
        for hold in hold_labels:
            fac = ((ep.get("holds") or {}).get(hold) or {}).get("facing") or {}
            final, peak = fac.get("heading_error_final_abs"), fac.get("heading_error_max_abs")
            if final is None or peak is None:
                continue
            denom = max(float(peak), FACING_CLOSED_MIN_INITIAL_RAD)
            if float(peak) < FACING_CLOSED_MIN_INITIAL_RAD:
                continue          # nothing to close -- the same refusal facing_metrics makes
            out[hold][pos] = float(min(1.0, max(-1.0, 1.0 - float(final) / denom)))
    return out


def json_facing(episodes: Sequence[dict], hold_labels: Sequence[str]) -> dict[str, np.ndarray] | None:
    """Per-episode ``facing_closed`` straight out of a NEW-harness per-episode JSON, or ``None``.

    Two shapes count as new: the key itself, and ``heading_error_initial_abs`` (from which the
    fraction is the metric's own definition, per episode -- not the median-of-medians proxy).
    """
    have_closed = have_initial = False
    out = {h: np.full(len(episodes), np.nan, dtype=np.float64) for h in hold_labels}
    for pos, ep in enumerate(episodes):
        for hold in hold_labels:
            fac = ((ep.get("holds") or {}).get(hold) or {}).get("facing") or {}
            if fac.get("facing_closed") is not None:
                have_closed = True
                out[hold][pos] = float(fac["facing_closed"])
                continue
            initial, final = fac.get("heading_error_initial_abs"), fac.get("heading_error_final_abs")
            if initial is None or final is None:
                continue
            have_initial = True
            if float(initial) < FACING_CLOSED_MIN_INITIAL_RAD:
                continue
            closed = 1.0 - float(final) / max(float(initial), FACING_CLOSED_MIN_INITIAL_RAD)
            out[hold][pos] = float(min(1.0, max(-1.0, closed)))
    return out if (have_closed or have_initial) else None


# ------------------------------------------------------------------------------- the metric registry
@dataclass(frozen=True)
class MetricSpec:
    """One scored (or watched) quantity: where its per-episode columns come from and how they reduce.

    ``mid`` is the dotted path into :func:`run_phases.summarize`'s output, so a decision rule reads
    ``summarize()`` and this record with the same key.
    """

    mid: str
    block: str                      # a block of the rule (F | F_mt | T | *_abs) | guard | watch
    blocks_all: tuple[str, ...]     # every block it is a member of (facing_closed is in F and F_abs)
    hold: str | None
    level: float
    one_sided: bool
    needs: tuple[str, ...]          # column keys, in the reducer's order
    reducer: str                    # median | mean | min_of_medians | ratio_of_means
    #: +1 higher-is-better, -1 lower-is-better (``eval_categories.metric_sign``): every delta this
    #: module forms is ``sign * (a - b)``, so a block ``z`` and a guard's WORSE read "positive =
    #: better" (the motion-quality block Q). +1 for every id before Q, so no recorded number moves.
    sign: int = 1


def _spec(mid, block, blocks_all, hold, needs, reducer="median", *, level=BLOCK_LEVEL, one_sided=False,
          sign=1):
    return MetricSpec(mid=mid, block=block, blocks_all=tuple(blocks_all), hold=hold, level=level,
                      one_sided=one_sided, needs=tuple(needs), reducer=reducer, sign=int(sign))


def _signed(spec: "MetricSpec", delta):
    """``sign * delta`` -- the identity (the same object) for a higher-is-better spec, so every
    record measured before the lower-is-better keys existed is bit-identical."""
    return delta if spec.sign >= 0 else -delta


def suite_categories_on(hold_labels: Sequence[str]) -> list[str]:
    """The categories every one of whose holds is on this row, in ``CATEGORIES`` order."""
    have = set(str(h) for h in hold_labels)
    ec = _ec()
    return [c for c in ec.CATEGORIES if all(h in have for h in ec.CATEGORY_HOLDS[c])]


def default_suite_rule(hold_labels: Sequence[str]) -> dict:
    """The rule-less fallback on a suite row: ``eval_categories``' default v2 rule, expanded at every
    category the row carries, both unit systems (the same union :func:`resolve` scores)."""
    cats = suite_categories_on(hold_labels)
    return _ec().expand_rule_union(_ec().default_rule_v2(), cats, cats)


def metric_specs(hold_labels: Sequence[str], *, suite: bool = False) -> list[MetricSpec]:
    """Decision 5's scored surface, restricted to the holds this row actually has.

    F = the closed fraction and the signed ``yaw_progress_rate`` on the two in-place turns and
    ``move_turn``; T = the speed ratio on the four direction holds; four guards (completion, the
    direction-hold tripod floor, and the two CONDITIONAL turn survivals); ``F_abs``/``T_abs`` the
    absolute-units twins a cap candidate is scored on. Right-handed holds and everything else in
    decision 5's WATCH list are not here -- ``watch_specs`` carries the two the curriculum row adds.

    ``suite=True`` (a ``category_suite`` row, plan D10): the block set comes from the EXPANDED v2
    rule (:func:`default_suite_rule`), never from the hold-keyed tuples above -- ``cat.<c>.<key>``
    members of ``F, F_mt, T`` and their ``_abs`` twins, the per-category guards and
    ``cat_tripod_min``, and the rule's category WATCH ids. The conditional turn survivals are
    omitted: undefined on a row whose holds are not nested, and ``summarize()`` returns ``None`` for
    them there, so the reducer check would refuse every suite row. The per-hold readings are in
    :func:`watch_specs`.
    """
    if suite:
        specs, _ = specs_from_rule(default_suite_rule(hold_labels), hold_labels, suite=True)
        return specs
    have = set(str(h) for h in hold_labels)
    out: list[MetricSpec] = []
    for hold in F_HOLDS:
        if hold in have:
            out.append(_spec(f"facing_closed_by_hold.{hold}", "F", ("F", "F_abs"), hold,
                             (f"closed:{hold}",)))
            out.append(_spec(f"yaw_progress_by_hold.{hold}", "F", ("F",), hold, (f"progress:{hold}",)))
            out.append(_spec(f"net_yaw_by_hold.{hold}", "F_abs", ("F_abs",), hold, (f"net:{hold}",)))
    for hold in T_HOLDS:
        if hold in have:
            out.append(_spec(f"ratio_by_hold.{hold}", "T", ("T",), hold, (f"ratio:{hold}",)))
            out.append(_spec(f"achieved_speed_by_hold.{hold}", "T_abs", ("T_abs",), hold,
                             (f"achieved:{hold}",)))
    out.append(_spec("completion", "guard", (), None, ("completed",), "mean",
                     level=GUARD_LEVEL, one_sided=True))
    if all(h in have for h in T_HOLDS):
        out.append(_spec("tripod_walking_dir", "guard", (), None,
                         tuple(f"tripod:{h}" for h in T_HOLDS), "min_of_medians",
                         level=GUARD_LEVEL, one_sided=True))
    if "turn_l90" in have and "back" in have:
        out.append(_spec("turn_survival_l90", "guard", (), "turn_l90",
                         ("surv:turn_l90", "surv:back"), "ratio_of_means",
                         level=GUARD_LEVEL, one_sided=True))
    if "turn_r90" in have and "turn_l90" in have:
        out.append(_spec("turn_survival_r90", "guard", (), "turn_r90",
                         ("surv:turn_r90", "surv:turn_l90"), "ratio_of_means",
                         level=GUARD_LEVEL, one_sided=True))
    return out


#: ``metric_id`` -> ``(column keys, reducer)``, given the holds the id names. The dotted ids are
#: ``summarize()``'s own keys (``candidates.yaml``'s ``decision_rule`` names the same ones), and this
#: table is the ONLY place a per-episode reconstruction of one of them is written down. A ``None``
#: entry means "not an eval-row quantity" -- a training-log tail or a cross-row reading that has no
#: per-episode form and therefore no bootstrap here.
_STEM_COLUMNS = {
    "facing_closed_by_hold": ("closed", "median"),
    "yaw_progress_by_hold": ("progress", "median"),
    "net_yaw_by_hold": ("net", "median"),
    "ratio_by_hold": ("ratio", "median"),
    "speed_ratio_by_hold": ("ratio", "median"),
    "dir_ratio_by_hold": ("ratio", "median"),
    "achieved_speed_by_hold": ("achieved", "median"),
    "tripod_score_by_hold": ("tripod", "median"),
    "tripod_by_hold": ("tripod", "median"),
    "survival_by_hold": ("surv", "mean"),
}
#: Ids that are not eval-row quantities at all: the two training-log tails (read off a training log,
#: not an eval), and the ring-completion WATCH label (a cross-row reading -- decision 6 keeps ring
#: completion WATCH at milestones).
_NOT_EVAL_ROW = ("track_ratio_tail", "track_ratio_move_tail", "completion_ring")
#: Episode-level ``summarize()`` keys -- one value per episode, no hold.
_EPISODE_COLUMNS = {"tripod": "ep_tripod", "tracking": "ep_tracking",
                    "facing_error_rms": "ep_facing_rms"}
#: ``tripod_walking`` is the min over the holds the HARNESS called walking, read from the data per
#: row; reconstructing it per episode would re-derive that definition rather than read it, which is
#: exactly what this module refuses to do. Its stricter sibling ``tripod_walking_dir`` (the named
#: four direction holds) is the guard, and it resolves.
_UNRESOLVABLE = _NOT_EVAL_ROW + ("tripod_walking",)


def is_category_id(mid: str) -> bool:
    """``cat.*`` or a category composite (``cat_tripod_min``) -- defined on a suite row only."""
    return str(mid).startswith("cat.") or str(mid) in _ec().COMPOSITES


def _cat_spec(mid: str, hold_labels: Sequence[str], *, block: str, blocks_all: Sequence[str],
              level: float, one_sided: bool, composite: dict | None) -> MetricSpec | None:
    """A category id's spec on a suite row (``None`` when the row lacks its holds, or it is a count
    or a :data:`CAT_POOLED_KEYS` statistic).

    ``cat.<c>.<key>`` reads column ``cat:<c>:<key>`` and ``cat.<c>.by_hand.<h>.<key>`` column
    ``cat:<c>:hand:<h>:<key>`` (NaN outside that category / hand); ``survival`` is the mean of the
    own-hold survival indicator (= the n-weighted mean of the holds' own survivals), every other key
    the median the harness pools. ``cat_tripod_min`` is the min over its member categories'
    ``cat:<c>:tripod`` medians -- the members are the rule's (``composite`` = the expanded rule's
    ``composite`` block, the decisive translation categories), else every translation category on
    the row.
    """
    ec = _ec()
    have = set(str(h) for h in hold_labels)
    if mid in ec.COMPOSITES:
        members = list((composite or {}).get(mid) or [c for c in ec.TRANSLATION_CATEGORIES
                                                        if c in suite_categories_on(hold_labels)])
        if not members or not all(c in suite_categories_on(hold_labels) for c in members):
            return None
        return _spec(mid, block, blocks_all, None, tuple(f"cat:{c}:tripod" for c in members),
                     "min_of_medians", level=level, one_sided=one_sided)
    parsed = ec.parse_cat_id(mid)
    if parsed is None:
        return None
    cat, hand, key = parsed
    if key in CAT_COUNT_KEYS or key in CAT_POOLED_KEYS:
        return None
    holds = ec.CATEGORY_HOLDS[cat]
    if hand is not None:
        holds = tuple(h for h in holds if ec.HAND_OF_HOLD.get(h) == hand)
        if not holds:
            return None
    if not all(h in have for h in holds):
        return None
    col = f"cat:{cat}:hand:{hand}:{key}" if hand else f"cat:{cat}:{key}"
    return _spec(mid, block, blocks_all, holds[0] if len(holds) == 1 else None, (col,),
                 "mean" if key == "survival" else "median", level=level, one_sided=one_sided,
                 sign=ec.metric_sign(mid))


def spec_for(mid: str, hold_labels: Sequence[str], *, block: str, blocks_all: Sequence[str],
             level: float, one_sided: bool, suite: bool = False,
             composite: dict | None = None) -> MetricSpec | None:
    """The :class:`MetricSpec` for one ``summarize()`` id, or ``None`` when it cannot be resolved.

    ``None`` for an id this row has no holds for, and for the handful of ids with no per-episode
    form at all (:data:`_UNRESOLVABLE`). Everything else resolves through :data:`_STEM_COLUMNS` or
    the three composite rules below, so ``candidates.yaml``'s ``decision_rule`` decides MEMBERSHIP
    and this module decides only HOW each member is computed.

    ``suite`` says the row is a ``category_suite`` row (read from its ``run_meta`` /
    ``aggregate.suite``, never inferred from labels -- plan D10). Category ids (``cat.*``,
    ``cat_tripod_min``) resolve ONLY there; on any other row they are ``None`` and land in the
    caller's unresolved list. On a suite row the conditional turn survivals are ``None`` (holds are
    not nested); every other legacy id resolves as it does on a legacy row, per hand.
    """
    have = set(str(h) for h in hold_labels)
    if is_category_id(mid):
        if not suite:
            return None
        return _cat_spec(mid, hold_labels, block=block, blocks_all=blocks_all, level=level,
                         one_sided=one_sided, composite=composite)
    if suite and mid in CONDITIONAL_SURVIVAL_IDS:
        return None
    if mid in _UNRESOLVABLE:
        return None
    if mid == "completion":
        return _spec(mid, block, blocks_all, None, ("completed",), "mean", level=level,
                     one_sided=one_sided)
    if mid in _EPISODE_COLUMNS:
        return _spec(mid, block, blocks_all, None, (_EPISODE_COLUMNS[mid],), "median", level=level,
                     one_sided=one_sided)
    if mid == "yaw_progress_turn_min":
        if not all(h in have for h in ("turn_l90", "turn_r90")):
            return None
        return _spec(mid, block, blocks_all, None,
                     ("progress:turn_l90", "progress:turn_r90"), "min_of_medians", level=level,
                     one_sided=one_sided)
    if mid == "tripod_walking_dir":
        if not all(h in have for h in T_HOLDS):
            return None
        return _spec(mid, block, blocks_all, None, tuple(f"tripod:{h}" for h in T_HOLDS),
                     "min_of_medians", level=level, one_sided=one_sided)
    if mid in ("turn_survival_l90", "turn_survival_r90"):
        hold, given = (("turn_l90", "back") if mid.endswith("l90") else ("turn_r90", "turn_l90"))
        if hold not in have or given not in have:
            return None
        return _spec(mid, block, blocks_all, hold, (f"surv:{hold}", f"surv:{given}"),
                     "ratio_of_means", level=level, one_sided=one_sided)
    parts = mid.split(".")
    if len(parts) == 3 and parts[0] == "facing_by_hold" and parts[2] == "settle":
        hold = parts[1]
        return None if hold not in have else _spec(mid, block, blocks_all, hold,
                                                   (f"settle:{hold}",), "median", level=level,
                                                   one_sided=one_sided)
    if len(parts) == 2 and parts[0] in _STEM_COLUMNS:
        key, reducer = _STEM_COLUMNS[parts[0]]
        hold = parts[1]
        return None if hold not in have else _spec(mid, block, blocks_all, hold,
                                                   (f"{key}:{hold}",), reducer, level=level,
                                                   one_sided=one_sided)
    return None


def why_unresolved(mid: str, hold_labels: Sequence[str], *, suite: bool = False) -> str:
    """Why :func:`spec_for` said no -- so a note says which of the reasons it is."""
    have = set(str(h) for h in hold_labels)
    if is_category_id(mid):
        if not suite:
            return ("a category id: defined on a category_suite row only (this row is a legacy "
                    "hold-keyed row -- informational here)")
        parsed = _ec().parse_cat_id(mid)
        if parsed is not None and parsed[2] in CAT_COUNT_KEYS:
            return "a count, not a per-episode quantity (no bootstrap half-width)"
        if parsed is not None and parsed[2] in CAT_POOLED_KEYS:
            return ("one statistic across the category's episodes, not a per-episode quantity (no "
                    "bootstrap half-width; WATCH only)")
        if parsed is None and mid not in _ec().COMPOSITES:
            return "not a known category id (eval_categories.parse_cat_id refuses it)"
        return "its category's holds are not all on this row"
    if suite and mid in CONDITIONAL_SURVIVAL_IDS:
        return "undefined on a suite row: holds are not nested (each episode runs one hold)"
    if mid in _NOT_EVAL_ROW:
        return ("not an eval-row quantity (a training-log tail or a cross-row label); it is WATCH "
                "and carries no bootstrap half-width")
    if mid == "tripod_walking":
        return ("the harness decides which holds are walking per row; tripod_walking_dir (the named "
                "four direction holds) is the guard that resolves")
    if mid == "tripod_walking_dir":
        absent = [h for h in T_HOLDS if h not in have]
        return f"the direction holds {absent} are not on this row"
    parts = mid.split(".")
    if len(parts) >= 2 and parts[1] not in have:
        return f"hold {parts[1]!r} is not on this row (it is a curriculum-row hold; see watch_metrics)"
    return "no per-episode reconstruction is registered for this id"


def rule_version(rule: dict | None) -> int:
    """``decision_rule.rule_version`` (1 when absent -- every record before the suite)."""
    return int((rule or {}).get("rule_version") or 1)


def specs_from_rule(rule: dict, hold_labels: Sequence[str], *,
                    suite: bool = False) -> tuple[list[MetricSpec], list[str]]:
    """``(specs, unresolvable ids)`` from ``candidates.yaml``'s ``decision_rule`` block.

    The yaml is authoritative for what is scored and at which level; a member it names that this
    module cannot compute is returned in the second list and NOTED, never silently dropped -- a
    quietly absent block member shrinks the block mean, which is exactly the kind of quiet failure
    that costs a branch. Falls back to :func:`metric_specs` + :func:`watch_specs` when the yaml
    carries no machine-readable rule.

    A ``rule_version: 2`` rule must arrive EXPANDED (``eval_categories.expand_rule``, as
    :func:`resolve` does): the raw v2 rule has no ``blocks``/``guards`` key, and the fallback would
    otherwise score the legacy hold-keyed surface silently -- so it is refused by name. The
    expanded rule's ``composite`` block names ``cat_tripod_min``'s members; its N/A blocks
    (``metrics: []``) contribute no member, hence no block.
    """
    if rule_version(rule) == 2 and not (rule or {}).get("expanded"):
        raise ResolutionError(
            "decision_rule has rule_version 2 but was not expanded: expand it with "
            "eval_categories.expand_rule (resolve() does, at decisive_at) before specs_from_rule -- "
            "the raw per_category templates carry no blocks/guards and would fall back to the "
            "legacy hold-keyed surface")
    blocks = (rule or {}).get("blocks") or {}
    guards = (rule or {}).get("guards") or {}
    watch = list((rule or {}).get("watch") or [])
    composite = dict((rule or {}).get("composite") or {})
    if not blocks and not guards:
        fallback = metric_specs(hold_labels, suite=suite)
        seen_fb = {s.mid for s in fallback}
        return fallback + [s for s in watch_specs(hold_labels, suite=suite)
                           if s.mid not in seen_fb], []
    membership: dict[str, list[str]] = {}
    levels: dict[str, float] = {}
    for block, body in blocks.items():
        for mid in (body or {}).get("metrics") or []:
            membership.setdefault(str(mid), []).append(str(block))
            # A metric may belong to two blocks (``facing_closed_by_hold.*`` is in F and F_abs). It
            # gets ONE half-width, so when the blocks declare different levels the WIDER one is
            # taken: a member measured at a tighter level than one of its blocks asks for would make
            # that block read BETTER/WORSE on noise. Assigning per block (the last one seen won)
            # made the choice depend on yaml key order.
            lvl = float((body or {}).get("level", BLOCK_LEVEL))
            levels[str(mid)] = max(levels.get(str(mid), lvl), lvl)
    guard_level = float(guards.get("level", GUARD_LEVEL))
    guard_ids = [str(m) for m in (guards.get("metrics") or [])]
    specs: list[MetricSpec] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for mid, in_blocks in sorted(membership.items()):
        primary = in_blocks[0]
        spec = spec_for(mid, hold_labels, block=primary, blocks_all=tuple(sorted(set(in_blocks))),
                        level=levels[mid], one_sided=False, suite=suite, composite=composite)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    for mid in guard_ids:
        if mid in membership:
            # A guard on a metric that is ALSO a block member (the motion-quality campaign's speed
            # guard ``cat.<c>.ratio`` beside T, its ``slip_per_m`` guard beside Q) keeps its OWN spec:
            # one-sided at the guard level, resolved under ``eval_categories.guard_key`` so the block
            # member's two-sided half-width is never read as the guard's.
            key = _ec().guard_key(mid)
            if key in seen:
                continue
            spec = spec_for(mid, hold_labels, block="guard", blocks_all=(), level=guard_level,
                            one_sided=True, suite=suite, composite=composite)
            (specs.append(dataclasses.replace(spec, mid=key)) if spec else unresolved.append(key))
            seen.add(key)
            continue
        if mid in seen:
            continue
        spec = spec_for(mid, hold_labels, block="guard", blocks_all=(), level=guard_level,
                        one_sided=True, suite=suite, composite=composite)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    for mid in watch:
        if mid in seen:
            continue
        spec = spec_for(mid, hold_labels, block="watch", blocks_all=(), level=BLOCK_LEVEL,
                        one_sided=False, suite=suite, composite=composite)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    return specs, unresolved


def block_names(specs: Iterable[MetricSpec]) -> list[str]:
    """The blocks the specs are members of, in :data:`BLOCK_ORDER` (unknown names after, sorted).

    Every place that enumerates blocks iterates THIS, i.e. the blocks of the rule actually passed
    (F/T/F_abs/T_abs for a v1 rule, the six of the expanded v2 union) -- never a literal tuple, so a
    block the rule adds (``F_mt``) can never be dropped by a seam that forgot it (plan D5/B4).
    """
    names = {b for s in specs for b in s.blocks_all}
    return [b for b in BLOCK_ORDER if b in names] + sorted(names - set(BLOCK_ORDER))


def watch_specs(hold_labels: Sequence[str], *, suite: bool = False) -> list[MetricSpec]:
    """The right-handed holds' quantities -- WATCH only, and only where the row has them.

    ``lat_r`` is a travel hold (ratio + achieved speed); ``move_turn_r`` is a move+turn hold whose
    facing "never closes" by construction (a 90 deg change inside 8 s at ~0.15-0.2 rad/s), so its
    facing quantities are watched too. None of these is ever a block member (decision 5).

    ``suite=True``: every category hold's own per-hold reading (the legacy per-hold keys are the
    per-hand readings on a suite row) -- the speed ratio and achieved speed of each moving hold and
    the facing quantities of each turning hold, WATCH only.
    """
    have = set(str(h) for h in hold_labels)
    out: list[MetricSpec] = []
    if suite:
        ec = _ec()
        for cat in suite_categories_on(hold_labels):
            for hold in ec.CATEGORY_HOLDS[cat]:
                if cat not in ("stand", "turn"):
                    out.append(_spec(f"ratio_by_hold.{hold}", "watch", (), hold, (f"ratio:{hold}",)))
                    out.append(_spec(f"achieved_speed_by_hold.{hold}", "watch", (), hold,
                                     (f"achieved:{hold}",)))
                if cat in ("turn", "move_turn"):
                    for stem, key in (("facing_closed_by_hold", "closed"),
                                      ("yaw_progress_by_hold", "progress"),
                                      ("net_yaw_by_hold", "net")):
                        out.append(_spec(f"{stem}.{hold}", "watch", (), hold, (f"{key}:{hold}",)))
        return out
    for hold in BORROW:
        if hold not in have:
            continue
        out.append(_spec(f"ratio_by_hold.{hold}", "watch", (), hold, (f"ratio:{hold}",)))
        out.append(_spec(f"achieved_speed_by_hold.{hold}", "watch", (), hold, (f"achieved:{hold}",)))
        if hold == "move_turn_r":
            out.append(_spec(f"facing_closed_by_hold.{hold}", "watch", (), hold, (f"closed:{hold}",)))
            out.append(_spec(f"yaw_progress_by_hold.{hold}", "watch", (), hold, (f"progress:{hold}",)))
            out.append(_spec(f"net_yaw_by_hold.{hold}", "watch", (), hold, (f"net:{hold}",)))
    return out


#: ``metrics`` is keyed by the ``ratio_by_hold.*`` spelling (the ladder's data-driven key). The gates
#: read the same four numbers under ``speed_ratio_by_hold.*``; the alias map lets a consumer that
#: holds either spelling find the R without a KeyError.
def alias_map(specs: Iterable[MetricSpec]) -> dict[str, str]:
    out = {}
    for s in specs:
        if not s.mid.startswith("ratio_by_hold."):
            continue
        if s.hold in rp.DIRECTION_HOLDS:
            out[f"speed_ratio_by_hold.{s.hold}"] = s.mid
        if s.hold in rp.GATE_1C_DIR_HOLDS:
            out[f"dir_ratio_by_hold.{s.hold}"] = s.mid
    return out


def _reduce(reducer: str, cols: list[np.ndarray]) -> np.ndarray:
    """Apply one reducer to already-resampled ``(B, n)`` columns, returning ``(B,)``.

    ``median`` / ``mean`` ignore the NaNs that mark "this episode produced no value" -- the same
    filter ``report.aggregate`` applies. ``min_of_medians`` PROPAGATES a NaN (a floor over an
    incomplete set flatters: the worst hold might be the unmeasured one -- ``rp._min_or_none``'s
    rule). ``ratio_of_means`` yields NaN on a zero denominator, which drops that resample
    (decision 5: "the bootstrap drops zero-denominator resamples").
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        if reducer == "median":
            return _nanmedian(cols[0])
        if reducer == "mean":
            return _nanmean(cols[0])
        if reducer == "min_of_medians":
            return np.min(np.stack([_nanmedian(c) for c in cols], axis=0), axis=0)
        if reducer == "ratio_of_means":
            num, den = _nanmean(cols[0]), _nanmean(cols[1])
            out = np.full(num.shape, np.nan, dtype=np.float64)
            ok = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
            out[ok] = num[ok] / den[ok]
            return out
    raise ResolutionError(f"unknown reducer {reducer!r}")


def _nanmedian(x: np.ndarray) -> np.ndarray:
    """``nanmedian`` along axis 1 that returns NaN (not a warning) for an all-NaN row."""
    allnan = np.all(np.isnan(x), axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    if not np.all(allnan):
        out[~allnan] = np.nanmedian(x[~allnan], axis=1)
    return out


def _nanmean(x: np.ndarray) -> np.ndarray:
    allnan = np.all(np.isnan(x), axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    if not np.all(allnan):
        out[~allnan] = np.nanmean(x[~allnan], axis=1)
    return out


def block_mean(standardised: np.ndarray) -> np.ndarray:
    """``z = mean_i(delta_i / R_i)`` per draw over a ``(k, B)`` block of standardised deltas.

    A member that is MISSING on a draw (its resample produced no value) leaves the mean rather than
    zeroing it -- decision 5's "MISSING in F/T -> that metric leaves the block mean". A draw where
    EVERY member is missing is NaN and drops out of the half-width.
    """
    valid = np.isfinite(standardised)
    cnt = valid.sum(axis=0)
    tot = np.where(valid, standardised, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


_block_mean = block_mean


# --------------------------------------------------------------------------------- an eval run dir
@dataclass
class Row:
    """One recorded eval run dir, reduced to what the bootstrap needs."""

    run_dir: Path
    row: str
    head: str
    dt: float
    hold_labels: tuple[str, ...]
    env_vars: dict
    provenance_ok: bool
    waived: bool
    accepted: bool
    reason: str
    env_vars_defaulted: tuple[str, ...]
    facing_closed_source: str | None
    env_index: tuple[int, ...]
    cols: dict[str, np.ndarray]
    summary: dict
    n_episodes: int
    process_floor_pair: bool
    mask_deviation: float | None
    reducer_check: dict[str, float]
    checkpoint: str = ""
    floor_of: str | None = None
    #: ``role`` of the input entry (``pair`` | ``informational`` | ``process_floor`` | ``seed_pair``).
    role: str = "pair"
    #: A ``category_suite`` row (read from ``run_meta.suite`` / ``aggregate.suite``, plan D10).
    suite: bool = False
    #: Suite geometry: ``waves`` x ``episodes_per_wave`` episodes, ``env_index = wave * E + env``.
    waves: int = 1
    episodes_per_wave: int = 0
    hold_of_wave: tuple[str, ...] = ()
    #: Per episode (``env_index`` order), from ``aggregate.suite`` -- never from the hold dicts.
    hold_index: tuple[str, ...] = ()
    category_index: tuple[str, ...] = ()
    hand_index: tuple[str | None, ...] = ()
    #: The eval stamp's completion marker (``complete.json``), suite rows only.
    complete: dict | None = None
    #: The entry's own checkpoint (checkpoint-shaped inputs) and recorded stamp (``role: pair``).
    entry_checkpoint: str | None = None
    stamp: str | None = None

    @property
    def label(self) -> str:
        return f"{self.head} @{self.row}"

    @property
    def pairing_key(self) -> tuple:
        """What two rows must share to be paired on ``env_index`` (plan D2's precondition)."""
        return (self.suite, self.waves, self.episodes_per_wave, self.hold_of_wave)


def required_carry_keys() -> tuple[str, ...]:
    """The ``EVAL_CARRY`` keys that have no ``EVAL_CARRY_DEFAULTS`` entry -- the five 2026-09-17 keys.

    Read from ``run_phases`` rather than listed, so growing ``EVAL_CARRY`` (with or without a
    default) cannot leave this rule behind. The BODY-only keys (``run_phases.EVAL_CARRY_BODY_ONLY``,
    the body envelope: no default, and REFUSED in a world eval) are never required -- a world stamp
    cannot carry them.
    """
    body_only = set(getattr(rp, "EVAL_CARRY_BODY_ONLY", ()))
    return tuple(k for k in rp.EVAL_CARRY if k not in rp.EVAL_CARRY_DEFAULTS and k not in body_only)


def _episodes(run_dir: Path) -> list[dict]:
    files = sorted((run_dir / "metrics").glob("episode_*.json"))
    eps = [json.loads(f.read_text()) for f in files]
    return sorted(eps, key=lambda e: int(e["env_index"]))


def _survival_end_steps(sid: str, dt: float, *, suite: dict | None = None) -> dict[str, int]:
    """``{hold: the step index an episode must reach to have survived that hold}``.

    The same walk ``rp.survival_by_hold`` makes (prologue then each hold's duration, rounded to
    steps), hoisted so the indicator can be computed per episode and resampled.

    A suite row has no walk: every category hold's end is its OWN wave's end,
    ``round((prologue_s + settle_hold_s + hold_s[label]) / dt)`` -- the convention
    ``aggregate.suite.survival_by_hold`` uses (a fall on the wave's last step survived). ``suite`` is
    the row's RECORDED geometry (``aggregate.suite``: ``prologue_s``, ``settle_hold_s``, the per-label
    ``hold_s`` map), so an old stamp is re-read on its own durations; without it the manifest row's.
    The settle hold has no entry (it is never an episode's own hold).
    """
    if suite is None:
        sc = _manifest_scenario(sid)
        if sc.get("kind") == "category_suite":
            suite = {"prologue_s": sc["prologue_s"], "settle_hold_s": sc["settle_hold_s"],
                     "hold_s": sc["hold_s"]}
    if suite is not None:
        hold_s = suite["hold_s"]
        if not isinstance(hold_s, dict):          # a scalar: every category hold shares it
            hold_s = {str(h): float(hold_s) for h in suite["hold_of_wave"]}
        base = float(suite["prologue_s"]) + float(suite["settle_hold_s"])
        return {str(h): int(round((base + float(s)) / float(dt))) for h, s in hold_s.items()}
    out: dict[str, int] = {}
    t_end = float(sc["prologue_s"])
    for hold in sc["hold_labels"]:
        t_end += float(sc["hold_s"][hold])
        out[hold] = int(round(t_end / float(dt)))
    return out


#: ``cat:<c>:<key>`` column sources on an episode's OWN hold entry -- written out here, not imported
#: from ``report.CATEGORY_METRIC_SOURCES``, so :func:`_reducer_check` is a real cross-check of the
#: harness's ``aggregate.suite.by_category`` rather than a re-reading of its own table. ``tripod``
#: (not ``low_confidence``), ``facing_closed`` (the re-scored fraction) and ``survival`` (the
#: own-hold end) are special-cased in :func:`_category_value`.
CAT_COLUMN_SOURCES: dict[str, tuple[str, ...]] = {
    "ratio": ("tracking", "vx", "ratio"),
    "achieved_speed": ("tracking", "vx", "actual_mean"),
    "yaw_progress": ("facing", "yaw_progress_rate"),
    "net_yaw": ("facing", "net_yaw_rate_toward"),
    "heading_final": ("facing", "heading_error_final_abs"),
    "settle_s": ("facing", "settle_time_s"),
    "drift_speed": ("motion", "speed_abs_mean_m_s"),
    "direction_error": ("direction", "direction_error_rad"),
    "cross_track_ratio": ("direction", "cross_track_ratio"),
}
#: Motion-quality keys (gait-motion-quality plan step 1), each read at ``quality.<key>`` on the own
#: hold entry -- None (NaN here) on a category the key is not defined on.
CAT_QUALITY_KEYS = ("action_jitter", "action_fft_freq", "joint_speed_mean", "rod_speed_mean",
                    "contact_switch_hz", "foot_slip", "wz_active_frac", "clock_restarts",
                    "swings_per_m", "slip_per_m", "work_per_m", "cam_work_per_m",
                    "swings_per_rad", "slip_per_rad", "work_per_rad", "cam_work_per_rad",
                    "six_down_frac", "settle_six_down_s", "foot_disp_end", "work_per_s")
CAT_COLUMN_SOURCES.update({key: ("quality", key) for key in CAT_QUALITY_KEYS})
#: Rigidity keys (clock-deprecation plan step 3), each read at ``rigidity.<key>`` on the own hold
#: entry -- None (NaN here) on a category the key is not defined on. The per-EPISODE ones only:
#: :data:`CAT_POOLED_KEYS` have no per-episode value.
CAT_RIGIDITY_KEYS = ("steps_per_m", "steps_per_rad", "support_feet", "steps_per_s",
                     "taps_per_m", "taps_per_rad", "swing_apex", "step_rate_hz",
                     "gait_hz", "clock_hz", "cadence_ratio", "steps_per_rad_pre", "steps_per_rad_post")
CAT_COLUMN_SOURCES.update({key: ("rigidity", key) for key in CAT_RIGIDITY_KEYS})
#: ``summarize()['cat'][c]`` keys that are ONE statistic across the category's episodes
#: (``gait_eval.metrics.RIGIDITY_POOLED_KEYS``: the cadence-vs-speed slope and the gait-vs-clock
#: phase concentration). There is no per-episode column to resample, so -- like the counts -- they
#: never resolve (WATCH only), and :func:`why_unresolved` says so.
CAT_POOLED_KEYS = ("hz_speed_slope", "offset_R")
#: Every per-episode ``cat`` key (``summarize()['cat'][c]`` minus the two counts and the pooled keys).
CAT_COLUMN_KEYS = ("survival", "ratio", "achieved_speed", "tripod", "facing_closed", "yaw_progress",
                   "net_yaw", "settle_s", "heading_final", "drift_speed", "direction_error",
                   "cross_track_ratio") + CAT_QUALITY_KEYS + CAT_RIGIDITY_KEYS


def _category_value(entry: dict, key: str, *, closed: float, survived: float) -> float:
    """One episode's ``cat`` value from its own hold entry (NaN when the harness excludes it)."""
    if key == "survival":
        return survived
    if key == "facing_closed":
        return closed
    if key == "tripod":
        tri = entry.get("tripod") or {}
        score = tri.get("tripod_score")
        return float(score) if score is not None and not tri.get("low_confidence") else np.nan
    node: Any = entry
    for part in CAT_COLUMN_SOURCES[key]:
        if not isinstance(node, dict):
            return np.nan
        node = node.get(part)
    return np.nan if node is None else float(node)


def _columns(episodes: Sequence[dict], hold_labels: Sequence[str], closed: dict[str, np.ndarray],
             end_steps: dict[str, int], *, suite_index: dict | None = None) -> dict[str, np.ndarray]:
    """Per-episode value arrays, NaN where the harness's own aggregate would have excluded the episode.

    The exclusions are copied from ``report.aggregate``: a tripod score only when it is present and
    not ``low_confidence``; a tracking ratio only when the hold has a tracking block whose
    ``vx.actual_mean`` is not None and whose ``ratio`` is not None; a facing quantity only when the
    key is not None. ``completed`` and the survival indicators are defined for every episode.

    ``suite_index`` (a suite row: ``{hold, category, hand}`` tuples in episode order, read from
    ``aggregate.suite`` by :func:`_suite_index`) adds the plan-D10 columns: ``cat:<c>:<key>`` and
    ``cat:<c>:hand:<h>:<key>``, NaN outside that category / hand, each read on the episode's OWN
    hold entry (pooled over both hands exactly as ``aggregate.suite.by_category`` pools), with
    ``survival`` the own-hold end indicator; and each ``surv:<h>`` becomes NaN outside hold ``h``'s
    own episodes, so ``survival_by_hold.<h>`` reproduces ``aggregate.suite.survival_by_hold``.
    """
    n = len(episodes)
    cols: dict[str, np.ndarray] = {}
    for hold in hold_labels:
        for key in ("progress", "net", "ratio", "achieved", "tripod", "surv", "settle"):
            cols[f"{key}:{hold}"] = np.full(n, np.nan, dtype=np.float64)
        cols[f"closed:{hold}"] = closed.get(hold, np.full(n, np.nan, dtype=np.float64))
    for key in ("completed", "ep_tripod", "ep_tracking", "ep_facing_rms"):
        cols[key] = np.full(n, np.nan, dtype=np.float64)
    for pos, ep in enumerate(episodes):
        cols["completed"][pos] = 1.0 if ep.get("completed_schedule") else 0.0
        for key, src in (("ep_tripod", "tripod_score"), ("ep_tracking", "tracking_ratio"),
                         ("ep_facing_rms", "facing_error_rms")):
            if ep.get(src) is not None:
                cols[key][pos] = float(ep[src])
        steps = int(ep.get("n_steps", 0))
        own = suite_index["hold"][pos] if suite_index is not None else None
        for hold in hold_labels:
            end = end_steps.get(hold)
            if end is not None and (own is None or hold == own):
                cols[f"surv:{hold}"][pos] = 1.0 if steps >= end else 0.0
            entry = (ep.get("holds") or {}).get(hold) or {}
            fac = entry.get("facing") or {}
            if fac.get("yaw_progress_rate") is not None:
                cols[f"progress:{hold}"][pos] = float(fac["yaw_progress_rate"])
            if fac.get("net_yaw_rate_toward") is not None:
                cols[f"net:{hold}"][pos] = float(fac["net_yaw_rate_toward"])
            if fac.get("settle_time_s") is not None:
                cols[f"settle:{hold}"][pos] = float(fac["settle_time_s"])
            tri = entry.get("tripod") or {}
            if tri.get("tripod_score") is not None and not tri.get("low_confidence"):
                cols[f"tripod:{hold}"][pos] = float(tri["tripod_score"])
            trk = entry.get("tracking") or {}
            vx = trk.get("vx") or {}
            if vx.get("actual_mean") is not None:
                cols[f"achieved:{hold}"][pos] = float(vx["actual_mean"])
                if vx.get("ratio") is not None:
                    cols[f"ratio:{hold}"][pos] = float(vx["ratio"])
    if suite_index is not None:
        cols.update(_category_columns(episodes, closed, end_steps, suite_index))
    return cols


def _category_columns(episodes: Sequence[dict], closed: dict[str, np.ndarray],
                      end_steps: dict[str, int], suite_index: dict) -> dict[str, np.ndarray]:
    """The ``cat:<c>:<key>`` / ``cat:<c>:hand:<h>:<key>`` columns of a suite row (see :func:`_columns`)."""
    n = len(episodes)
    cats = list(dict.fromkeys(suite_index["category"]))
    out: dict[str, np.ndarray] = {}
    for cat in cats:
        hands = sorted({h for c, h in zip(suite_index["category"], suite_index["hand"])
                        if c == cat and h is not None})
        for key in CAT_COLUMN_KEYS:
            out[f"cat:{cat}:{key}"] = np.full(n, np.nan, dtype=np.float64)
            for hand in hands:
                out[f"cat:{cat}:hand:{hand}:{key}"] = np.full(n, np.nan, dtype=np.float64)
    for pos, ep in enumerate(episodes):
        own, cat, hand = (suite_index["hold"][pos], suite_index["category"][pos],
                          suite_index["hand"][pos])
        entry = (ep.get("holds") or {}).get(own) or {}
        survived = 1.0 if int(ep.get("n_steps", 0)) >= int(end_steps[own]) else 0.0
        closed_v = float(closed[own][pos]) if own in closed else np.nan
        for key in CAT_COLUMN_KEYS:
            v = _category_value(entry, key, closed=closed_v, survived=survived)
            out[f"cat:{cat}:{key}"][pos] = v
            if hand is not None:
                out[f"cat:{cat}:hand:{hand}:{key}"][pos] = v
    return out


def _summary_with_survival(run_dir: Path, sid: str) -> dict:
    """``rp.summarize`` of the row's aggregate, with ``survival_by_hold`` injected.

    ``_run_eval`` injects it IN MEMORY, so an aggregate read back off disk has no such key and both
    conditional turn survivals would come back ``None`` -- a silent guard loss (step 3's hand-off).
    """
    sm = json.loads((run_dir / "scenario_metrics.json").read_text())
    agg = dict(sm["aggregate"])
    meta = sm.get("run_meta") or {}
    eps = sm.get("episodes") or []
    if agg.get("suite"):
        # A suite row's survival is each hold's OWN (plan D10's three injection sites): the
        # cumulative walk would count every earlier wave's falls, and the manifest row has no
        # schedule to walk. The harness already wrote the right table.
        if "survival_by_hold" not in agg:
            agg["survival_by_hold"] = dict(agg["suite"].get("survival_by_hold") or {})
        summary = rp.summarize(agg)
        if not summary.get("cat"):
            # run_phases.summarize owns the plan-D5 hoist; without it every cat.* reducer check
            # would read "missing" -- refuse by name rather than as a wall of NaN disagreements.
            raise ResolutionError(f"{run_dir}: run_phases.summarize() emits no 'cat' for this "
                                  "suite aggregate (it predates the per-category hoist, plan D5)")
        return summary
    if "survival_by_hold" not in agg:
        agg["survival_by_hold"] = rp.survival_by_hold(eps, meta.get("hold_labels") or [],
                                                      rp._manifest_hold_s(sid),
                                                      dt=float(meta.get("dt", 0.02)))
    return rp.summarize(agg)


def facing_closed_by_hold_from_raws(run_dir: str | Path, *, ring_proxy: bool = False,
                                    verify: bool = True) -> tuple[dict[str, float | None], str | None]:
    """``({hold: closed fraction | None}, source)`` for a recorded run dir -- the SHARED helper.

    Same shape and meaning as ``run_phases.summarize()``'s ``facing_closed_by_hold`` /
    ``facing_closed_source``, but recovered from the saved ``raw/*.npz`` instead of read off an
    aggregate that predates the key. The ladder's milestone comparisons against the RECORDED 2b / 2a
    rows must go through this: those rows' aggregates yield ``{hold: None}`` / source ``None``, and a
    comparison of a new row's per-episode ``facing_closed`` against nothing is a silently dropped
    half of the F block. It also keeps both sides of such a comparison on ONE definition, which the
    "a proxy row is never pooled against a metric row" rule requires.
    """
    row = load_row({"run_dir": str(Path(run_dir))}, ring_proxy=ring_proxy, verify=verify)
    out: dict[str, float | None] = {}
    for hold in row.hold_labels:
        col = row.cols.get(f"closed:{hold}")
        vals = col[np.isfinite(col)] if col is not None else np.array([])
        out[hold] = float(np.median(vals)) if vals.size else None
    return out, row.facing_closed_source


def summarize_run_dir(run_dir: str | Path, *, ring_proxy: bool = False,
                      verify: bool = True) -> dict:
    """``run_phases.summarize()`` of a recorded run dir, with both recorded-row repairs applied.

    1. ``survival_by_hold`` is injected (``_run_eval`` only ever put it in memory), so
       ``turn_survival_l90`` / ``turn_survival_r90`` are numbers rather than ``None`` -- otherwise
       two of the four guards go MISSING and decision 5 turns that into a REJECT.
    2. ``facing_closed_by_hold`` / ``facing_closed_source`` are replaced by the re-score
       (:func:`facing_closed_by_hold_from_raws`), so the F block has its closed fractions.

    This is what a milestone should call on a row's ``run_dir``; ``summarize()`` alone is right only
    for a row the current harness wrote.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    summary = _summary_with_survival(run_dir, str(meta["scenario_id"]))
    closed, source = facing_closed_by_hold_from_raws(run_dir, ring_proxy=ring_proxy, verify=verify)
    summary["facing_closed_by_hold"] = closed
    summary["facing_closed_source"] = source
    return summary


def _reducer_check(specs: Sequence[MetricSpec], cols: dict[str, np.ndarray], summary: dict,
                   n: int) -> dict[str, float]:
    """Worst absolute disagreement between each reducer and ``summarize()``'s own number.

    The identity resample (every episode once, in ``env_index`` order) must reproduce the recorded
    aggregate exactly, or the per-episode reconstruction is not the estimator the decision rule
    reads. ``facing_closed_by_hold.*`` is exempt -- it exists only by re-score, and the recorded
    aggregate has nothing to compare it against (the mask check covers it instead).

    Every ``cat.<c>.<key>`` is checked against ``summarize()['cat']`` (the suite's pooled
    ``by_category`` medians and own-hold survivals); ``cat.<c>.facing_closed`` included -- on a suite
    row the harness wrote the per-episode fraction, so the re-score is checked against it. A
    COMPOSITE (``cat_tripod_min``) is checked against the SHARED ``eval_categories.composite_guards``
    applied to a copy of the summary over the spec's own member categories -- one definition for the
    harness, the driver and this module -- and a composite with no summary value is a named
    :class:`ResolutionError`, never a NaN that reads as "both missing".
    """
    out: dict[str, float] = {}
    ident = np.arange(n)[None, :]
    for spec in specs:
        if spec.mid.startswith("facing_closed_by_hold."):
            continue
        got = float(_reduce(spec.reducer, [cols[c][ident] for c in spec.needs])[0])
        if spec.mid in _ec().COMPOSITES:
            members = [c.split(":")[1] for c in spec.needs]
            want = _ec().composite_guards(copy.deepcopy(summary), members).get(spec.mid)
            if want is None:
                raise ResolutionError(
                    f"{spec.mid}: composite_guards gives no value on this row's summary over "
                    f"{members} (a member's cat.<c>.tripod is missing) -- the composite cannot be "
                    "checked, so it cannot be resolved")
            out[spec.mid] = abs(got - float(want)) if math.isfinite(got) else float("inf")
            continue
        # a guard resolved under its own key (``...@guard``) reads the plain id's summary value
        want = _summary_value(summary, _ec().base_metric_id(spec.mid))
        if want is None:
            out[spec.mid] = float("nan") if math.isfinite(got) else 0.0
            continue
        out[spec.mid] = abs(got - float(want)) if math.isfinite(got) else float("inf")
    return out


def _summary_value(summary: dict, mid: str) -> float | None:
    node: Any = summary
    for part in mid.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return None if node is None else float(node)


# ------------------------------------------------------------- suite stamps (plan D9 item 2 / D10)
def _repo_path(p: str | Path) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else REPO / p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_matches(entry_checkpoint: str | Path, meta: dict, complete: dict | None = None) -> bool:
    """Is the stamp's evaluated checkpoint the entry's? By resolved path, else by sha256.

    The harness records ABSOLUTE checkpoint paths and ``candidates.yaml`` repo-relative ones, so the
    path test resolves both against the repo; a checkpoint moved or reached through another path
    still matches when its sha256 equals the one the stamp recorded (``complete.json`` or
    ``run_meta.checkpoint_sha256``).
    """
    mine = _repo_path(entry_checkpoint)
    recorded = (meta or {}).get("checkpoint")
    if recorded and mine.resolve() == _repo_path(recorded).resolve():
        return True
    want = (complete or {}).get("checkpoint_sha256") or (meta or {}).get("checkpoint_sha256")
    if want and mine.is_file():
        return _sha256_file(mine) == str(want)
    return False


def check_suite_stamp(run_dir: str | Path, *, checkpoint: str | Path | None = None) -> dict:
    """D10's FRESHNESS RULE for one suite-row eval stamp: its ``complete.json`` on success.

    A suite row's ``write_run`` writes ``complete.json`` LAST (after the ~1000 raw npz, which follow
    the aggregate), so a stamp killed mid-write has none. Refused -- always a
    :class:`ResolutionError` naming the stamp, never a ``FileNotFoundError`` -- when the marker is
    missing, when the episode JSONs or the raw npz do not number ``episodes_total`` (a partial raw
    set), or when ``checkpoint`` is given and is not what the stamp evaluated
    (:func:`checkpoint_matches`). Legacy rows carry no marker and never come here.
    """
    run_dir = Path(run_dir)
    marker = run_dir / COMPLETE_MARKER
    if not marker.is_file():
        raise ResolutionError(
            f"{run_dir}: no {COMPLETE_MARKER} -- a suite stamp without its completion marker is "
            "partial (the eval was stopped or killed before write_run finished) and is never read")
    try:
        complete = json.loads(marker.read_text())
    except ValueError as exc:
        raise ResolutionError(f"{run_dir}: {COMPLETE_MARKER} does not parse ({exc})") from exc
    for needed in ("run_meta.json", "scenario_metrics.json", "metrics"):
        if not (run_dir / needed).exists():
            raise ResolutionError(f"{run_dir}: {needed} is missing beside a completion marker")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    total = complete.get("episodes_total", meta.get("episodes_total"))
    if total is None:
        raise ResolutionError(f"{run_dir}: neither {COMPLETE_MARKER} nor run_meta records "
                              "episodes_total")
    total = int(total)
    if meta.get("episodes_total") is not None and int(meta["episodes_total"]) != total:
        raise ResolutionError(f"{run_dir}: {COMPLETE_MARKER} says {total} episodes, run_meta "
                              f"{meta['episodes_total']}")
    n_json = sum(1 for _ in (run_dir / "metrics").glob("episode_*.json"))
    if n_json != total:
        raise ResolutionError(f"{run_dir}: {n_json} of {total} metrics/episode_*.json -- a partial "
                              "episode set")
    raw = run_dir / "raw"
    n_raw = sum(1 for _ in raw.glob("episode_*.npz")) if raw.is_dir() else 0
    if complete.get("raw_written") is False or n_raw != total:
        raise ResolutionError(f"{run_dir}: missing or partial raw set ({n_raw} of {total} "
                              "raw/episode_*.npz) -- a suite row is re-scored from its raws")
    if checkpoint is not None and not checkpoint_matches(checkpoint, meta, complete):
        raise ResolutionError(
            f"{run_dir}: evaluated {meta.get('checkpoint')!r} (sha256 "
            f"{complete.get('checkpoint_sha256') or meta.get('checkpoint_sha256')}), not the "
            f"entry's checkpoint {str(checkpoint)!r} (by resolved path or by sha256)")
    return complete


def resolve_eval_dir(root: str | Path, sid: str = SCREEN_ROW, *,
                     checkpoint: str | Path | None = None, seed_dir: str = "seed001") -> Path:
    """The NEWEST eval stamp under ``<root>/<sid>/<seed_dir>`` that passes :func:`check_suite_stamp`.

    ``root`` is repo-relative (``candidates.yaml``'s ``eval_dir``) or absolute. Stamps are the
    harness's ``%Y-%m-%d_%H-%M-%S`` directory names, so name order is time order. A newer stamp
    that fails the freshness rule (partial, or another checkpoint's) is passed over for an older
    one that passes; none passing is a :class:`ResolutionError` listing every stamp's reason. Used
    for ``role: seed_pair`` inputs and ``milestone_reference`` (the driver calls it too).
    """
    base = _repo_path(root) / str(sid) / str(seed_dir)
    if not base.is_dir():
        raise ResolutionError(f"no eval stamps under {base} (the eval has not run, or eval_dir / "
                              "row names another tree)")
    why: list[str] = []
    for stamp in sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name,
                        reverse=True):
        try:
            check_suite_stamp(stamp, checkpoint=checkpoint)
            return stamp
        except ResolutionError as exc:
            why.append(f"{stamp.name}: {exc}")
    raise ResolutionError(f"no stamp under {base} passes the freshness rule"
                          + (": " + " | ".join(why) if why else " (the directory is empty)"))


def input_run_dir(entry: dict) -> tuple[Path, str | None, str | None]:
    """``(run dir, entry checkpoint, recorded stamp)`` for one ``resolution_inputs`` entry.

    Two shapes, exactly one per entry (plan D9 item 2): ``run_dir`` (a recorded legacy row), or
    ``checkpoint + eval_dir`` (+ ``preset``, which the driver's validator requires). A checkpoint-
    shaped ``role: pair`` input is read from the stamp the driver recorded for it at this eval_sha
    (``lad.reevals``, passed in as the entry's ``stamp``: a stamp name under
    ``<eval_dir>/<row>/seed001``, or a path); a missing stamp is a :class:`ResolutionError` NAMING
    THE HEAD, never a fallback to "the newest stamp", which could be another eval's. Every other
    checkpoint-shaped input (``seed_pair``, ``informational``) takes :func:`resolve_eval_dir`. Both
    pass the freshness rule against the entry's checkpoint.
    """
    head = entry.get("head") or entry.get("checkpoint") or entry.get("run_dir") or "?"
    has_run = bool(entry.get("run_dir"))
    has_ckpt = bool(entry.get("checkpoint")) or bool(entry.get("eval_dir"))
    if has_run and has_ckpt:
        raise ResolutionError(f"resolution input {head}: carries both run_dir and checkpoint/eval_dir "
                              "-- an entry is one shape or the other")
    if not has_run and not has_ckpt:
        raise ResolutionError(f"resolution input {head}: carries neither run_dir nor "
                              "checkpoint + eval_dir")
    if has_run:
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (REPO / run_dir).resolve()
        return run_dir, None, None
    if not (entry.get("checkpoint") and entry.get("eval_dir")):
        raise ResolutionError(f"resolution input {head}: a checkpoint-shaped input needs both "
                              "checkpoint and eval_dir")
    sid = str(entry.get("row") or SCREEN_ROW)
    ckpt = str(entry["checkpoint"])
    if str(entry.get("role") or "pair") == "pair":
        stamp = entry.get("stamp")
        if not stamp:
            raise ResolutionError(
                f"resolution input {head} (role: pair): no reeval stamp recorded for it at this "
                "eval_sha (lad.reevals) -- the reeval must complete before D00s reads it")
        sp = Path(str(stamp))
        if sp.is_absolute():
            run_dir = sp
        elif len(sp.parts) == 1:
            run_dir = _repo_path(entry["eval_dir"]) / sid / "seed001" / sp
        else:
            run_dir = REPO / sp
        check_suite_stamp(run_dir, checkpoint=ckpt)
        return run_dir, ckpt, str(stamp)
    return resolve_eval_dir(entry["eval_dir"], sid, checkpoint=ckpt), ckpt, None


def _suite_index(run_dir: Path, meta: dict, agg: dict, episodes: Sequence[dict]) -> dict:
    """``{hold, category, hand}`` per episode (``env_index`` order) + the suite geometry.

    Read from ``aggregate.suite`` (``waves``, ``episodes_total``, ``hold_of_wave``,
    ``category_of_hold``, ``hand_of_hold``: episode ``i`` ran wave ``i // E``'s hold) -- never from
    which hold dicts carry steps -- and cross-checked against ``run_meta.env_index_map`` and the
    episode's own ``hold`` key where they exist. Refuses an incomplete episode set, and any episode
    with steps in a hold that is neither its own nor the settle (plan D10: every union label
    carries an entry, the unvisited ones with ``n_steps: 0``).
    """
    sb = agg.get("suite") or {}
    if not sb:
        raise ResolutionError(f"{run_dir}: run_meta marks a category_suite row but the aggregate "
                              "carries no suite block")
    waves = int(sb.get("waves") or meta.get("waves") or 0)
    total = int(sb.get("episodes_total") or meta.get("episodes_total") or 0)
    hold_of_wave = [str(h) for h in (sb.get("hold_of_wave") or [])]
    if not waves or not total or total % waves or len(hold_of_wave) != waves:
        raise ResolutionError(f"{run_dir}: inconsistent suite geometry (waves {waves}, "
                              f"episodes_total {total}, {len(hold_of_wave)} holds of wave)")
    per_wave = total // waves
    idx = [int(e["env_index"]) for e in episodes]
    if idx != list(range(total)):
        raise ResolutionError(f"{run_dir}: {len(idx)} episodes whose env_index is not 0..{total - 1} "
                              "-- a partial or re-indexed suite row cannot be paired")
    cat_of = {str(k): str(v) for k, v in (sb.get("category_of_hold") or {}).items()}
    hand_of = dict(sb.get("hand_of_hold") or {})
    labels = list(sb.get("hold_labels") or [])
    settle = str(labels[0] if labels else ((meta.get("suite") or {}).get("settle_label") or "settle"))
    emap = {int(r["env_index"]): r for r in (meta.get("env_index_map") or [])}
    holds: list[str] = []
    for ep in episodes:
        i = int(ep["env_index"])
        wave, own = i // per_wave, hold_of_wave[i // per_wave]
        rec = emap.get(i)
        if rec is not None and (str(rec.get("label")) != own or int(rec.get("wave", wave)) != wave):
            raise ResolutionError(f"{run_dir}: env_index {i} is wave {wave} / {own} by "
                                  f"aggregate.suite but {rec} by run_meta.env_index_map")
        if ep.get("hold") is not None and str(ep["hold"]) != own:
            raise ResolutionError(f"{run_dir}: episode {i} says its own hold is {ep['hold']!r}; "
                                  f"aggregate.suite says wave {wave} ran {own!r}")
        for lab, entry in (ep.get("holds") or {}).items():
            if lab in (own, settle):
                continue
            steps = int((entry or {}).get("n_steps") or 0)
            if steps > 0:
                raise ResolutionError(
                    f"{run_dir}: episode {i} (own hold {own!r}) has {steps} steps in hold {lab!r} -- "
                    "a suite episode runs the settle and ONE hold, so a populated foreign hold means "
                    "the wave did not run the schedule its label says")
        holds.append(own)
    return {"hold": tuple(holds), "category": tuple(cat_of[h] for h in holds),
            "hand": tuple(hand_of.get(h) for h in holds), "waves": waves,
            "episodes_per_wave": per_wave, "hold_of_wave": tuple(hold_of_wave), "settle": settle}


def load_row(entry: dict, *, ring_proxy: bool, rule: dict | None = None, verify: bool = True) -> Row:
    """Read one ``resolution_inputs`` entry into a :class:`Row`, applying the provenance rule.

    ``entry`` keys: ``run_dir`` (required), ``provenance_ok`` (the operator's WAIVER of the
    provenance refusal -- see below), ``role`` (``pair`` | ``informational`` | ``process_floor``;
    default ``pair``), ``process_floor_pair`` (the same thing as ``role: process_floor``),
    ``floor_of`` (the head a flagged row is the re-eval OF, when the labels differ) and the
    optional ``head`` / ``row`` labels. A missing ``head`` is derived from the checkpoint
    (``<run stamp>/<model stem>``), which is the right identity for "the same head": two rows
    carrying one checkpoint are one head's two evals, i.e. a process-floor pair, whatever they are
    called. The checkpoint is kept on the row either way, so a flagged row labelled ``2b_floor``
    still finds the ``2b`` row it is the floor of (:func:`build_pairs`).

    **Provenance.** ``provenance_ok`` is the MEASURED verdict: ``run_meta.env_vars`` carries every
    ``EVAL_CARRY`` key with no ``EVAL_CARRY_DEFAULTS`` entry. A defaulted key absent from
    ``env_vars`` is taken at its default and named in ``env_vars_defaulted``. A row that fails is
    refused; the yaml's ``provenance_ok: true`` lifts the refusal but leaves the row INFORMATIONAL
    (``accepted: false``), never a source of the candidate ``R``. That is what makes the three
    ``reeval_corrected`` rows informational, and it is not redundant with the plan's "n >= 50 per
    hold" sentence: measured on the recorded data, ``reeval_corrected/1b`` has 79/69/68 usable
    episodes on the three F holds, so an n rule alone would let it into the candidate R -- which the
    plan says it must not be. A row whose ``env_vars`` does not record the five knobs cannot be
    SHOWN to have been evaluated under the configuration of the row it would be paired with, so its
    spread is not the candidate-vs-control eval resolution.

    **Suite rows (plan D10).** An entry is ``run_dir`` or checkpoint-shaped (:func:`input_run_dir`);
    ``role: seed_pair`` rows (the two rescreened B0 screens, D00bs' inputs) are read and recorded
    but never pooled. A ``category_suite`` row -- the kind read from ``run_meta.suite`` /
    ``aggregate.suite``, never from labels -- must pass the freshness rule
    (:func:`check_suite_stamp`); each episode's hold/category/hand comes from ``aggregate.suite``
    (:func:`_suite_index`, which also refuses a populated foreign hold); survival is own-hold; and
    the ``suite`` flag reaches every spec reader, so category ids resolve and the conditional turn
    survivals do not.
    """
    run_dir, entry_ckpt, stamp = input_run_dir(entry)
    # A refusal, not a traceback: the ladder driver reports "resolution input <run_dir> is not on
    # this machine (resolution.py refuses it)" and needs this to come back as a ResolutionError.
    for needed in (run_dir / "run_meta.json", run_dir / "scenario_metrics.json", run_dir / "metrics"):
        if not needed.exists():
            raise ResolutionError(f"resolution input {run_dir} is unusable: {needed.name} is missing "
                                  "(the run dir is not on this machine, or the eval never finished)")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    sid = str(entry.get("row") or meta["scenario_id"])
    dt = float(meta.get("dt", 0.02))
    agg_rec = (json.loads((run_dir / "scenario_metrics.json").read_text()).get("aggregate") or {})
    is_suite = bool(meta.get("suite")) or bool(agg_rec.get("suite"))
    complete = None
    if is_suite:
        # Every suite-row stamp read passes the freshness rule; a checkpoint-shaped entry was
        # already checked against its checkpoint by input_run_dir.
        complete = check_suite_stamp(run_dir) if entry_ckpt is None else json.loads(
            (run_dir / COMPLETE_MARKER).read_text())
    hold_labels = tuple(str(h) for h in (meta.get("hold_labels") or _manifest_scenario(sid)["hold_labels"]))
    env_vars = dict(meta.get("env_vars") or {})
    ckpt = Path(str(meta.get("checkpoint", "unknown")))
    head = str(entry.get("head") or f"{ckpt.parent.name}/{ckpt.stem}")

    missing = [k for k in required_carry_keys() if k not in env_vars]
    defaulted = tuple(k for k in sorted(rp.EVAL_CARRY_DEFAULTS) if k not in env_vars)
    waived = bool(entry.get("provenance_ok"))
    role = str(entry.get("role") or "pair")
    provenance_ok = not missing
    if missing and not waived:
        accepted = False
        reason = ("refused: run_meta.env_vars lacks " + ", ".join(missing)
                  + " (set provenance_ok: true in resolution_inputs to keep it as an informational row)")
    elif missing:
        accepted = False
        reason = ("informational: run_meta.env_vars lacks " + ", ".join(missing)
                  + " (refusal waived by candidates.yaml provenance_ok; never a candidate-R source)")
    elif role == "informational":
        accepted, reason = False, "informational by role (reported, never pooled)"
    elif role == "seed_pair":
        accepted, reason = False, ("seed_pair by role (a rescreened B0 seed, D00bs' input; recorded, "
                                   "never a cross-head arm)")
    else:
        accepted, reason = True, "ok"

    episodes = _episodes(run_dir)
    n = len(episodes)
    closed: dict[str, np.ndarray] = {}
    source: str | None = None
    mask_dev: float | None = None
    use_proxy = ring_proxy and sid.startswith("ring")
    have_raws = (run_dir / "raw").is_dir() and any((run_dir / "raw").glob("episode_*.npz"))
    if not use_proxy and have_raws:
        closed, mask_dev = rescore_facing(run_dir, episodes, hold_labels, dt,
                                          _manifest_scenario(sid)["fall_exclusion_s"])
        if verify and mask_dev is not None and mask_dev > MASK_CHECK_TOL:
            raise ResolutionError(
                f"{run_dir}: the whole-hold window replication disagrees with the recorded facing "
                f"keys by {mask_dev:.3e} (> {MASK_CHECK_TOL:.0e}). facing_closed re-scored off a "
                "wrong window is undetectable downstream -- fix whole_hold_index before trusting it.")
        source = "metric"
    if source is None and not use_proxy:
        from_json = json_facing(episodes, hold_labels)
        if from_json is not None:
            closed, source = from_json, "metric"
    if source is None:
        closed = proxy_facing_from_max(episodes, hold_labels)
        source = "proxy" if any(np.any(np.isfinite(v)) for v in closed.values()) else None
        if source is None:
            closed = {}

    suite_idx = _suite_index(run_dir, meta, agg_rec, episodes) if is_suite else None
    end_steps = _survival_end_steps(sid, dt, suite=(agg_rec.get("suite") if is_suite else None))
    cols = _columns(episodes, hold_labels, closed, end_steps, suite_index=suite_idx)
    summary = _summary_with_survival(run_dir, sid)
    specs = (specs_from_rule(rule or {}, hold_labels, suite=is_suite)[0]
             + watch_specs(hold_labels, suite=is_suite))
    checks = _reducer_check(specs, cols, summary, n) if (verify and n) else {}
    bad = {k: v for k, v in checks.items() if not (math.isfinite(v) and v <= 1e-9)}
    if bad:
        raise ResolutionError(
            f"{run_dir}: the per-episode reconstruction does not reproduce summarize()'s own "
            f"numbers: {bad}. The bootstrap would resample a different estimator than the decision "
            "rule reads.")
    geometry: dict[str, Any] = {}
    if suite_idx is not None:
        geometry = {"suite": True, "waves": suite_idx["waves"],
                    "episodes_per_wave": suite_idx["episodes_per_wave"],
                    "hold_of_wave": suite_idx["hold_of_wave"], "hold_index": suite_idx["hold"],
                    "category_index": suite_idx["category"], "hand_index": suite_idx["hand"],
                    "complete": complete}
    return Row(run_dir=run_dir, row=sid, head=head, dt=dt, hold_labels=hold_labels, env_vars=env_vars,
               provenance_ok=provenance_ok, waived=waived, accepted=accepted, reason=reason,
               env_vars_defaulted=defaulted,
               facing_closed_source=("proxy" if source == "proxy" else source),
               env_index=tuple(int(e["env_index"]) for e in episodes), cols=cols, summary=summary,
               n_episodes=n,
               process_floor_pair=bool(entry.get("process_floor_pair")) or role == "process_floor",
               mask_deviation=mask_dev, reducer_check=checks, checkpoint=str(ckpt),
               floor_of=(str(entry["floor_of"]) if entry.get("floor_of") else None),
               role=role, entry_checkpoint=entry_ckpt, stamp=stamp,
               episodes_per_wave=(geometry.pop("episodes_per_wave", n)), **geometry)


# ------------------------------------------------------------------------------------ the bootstrap
@dataclass
class PairResult:
    a: Row
    b: Row
    name: str
    n: int
    metrics: dict[str, dict]           # mid -> {R_eval, n_valid, below_min_n, delta_point, ...}
    blocks: dict[str, dict]            # block -> {R_eval, members, n_draws} -- PAIR-LOCAL, see below
    same_head: bool
    #: ``mid -> (B,)`` bootstrap deltas. Kept because the published block half-width standardises
    #: each member by the MERGED ``R_eval,i`` (the one ``decide()`` divides by), which is only known
    #: after every pair has been drawn -- see :func:`_standardised_blocks`. ``blocks`` above is the
    #: pair-local version, standardised by this pair's own R; it is a diagnostic, not the record.
    deltas: dict[str, np.ndarray] = field(default_factory=dict)


def _pair_name(a: Row, b: Row) -> str:
    return f"{a.head} vs {b.head} @{a.row}"


def pair_rng(rng_seed: int, name: str) -> np.random.Generator:
    """A generator seeded from ``(rng_seed, name)`` -- one INDEPENDENT stream per pair.

    Not one shared generator consumed in order, deliberately. The process-noise floor pair is run
    weeks after ``D00`` (it needs 4 min of GPU), and with a shared stream its mere presence would
    shift every cross-head pair's draws and move every recorded ``R`` by the bootstrap's Monte-Carlo
    error -- after decisions had already been taken against those numbers. Decision 6 says ``R`` is
    never re-fitted; keying the stream to the pair's own name is what makes re-running the tool with
    a new input leave the existing pairs bit-identical.
    """
    tag = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng([int(rng_seed), tag])


def bootstrap_pair(a: Row, b: Row, *, B: int, rng_seed: int,
                   specs: Sequence[MetricSpec] | None = None) -> PairResult:
    """One pair's paired episode bootstrap: per-metric half-widths and the block statistics' own.

    The index list is drawn once per draw over the env indices BOTH rows have and applied to both --
    that is the whole of "paired on ``env_index``". Each head then reduces over its own valid subset
    of those episodes, so the resampled statistic is the estimator the harness reports rather than an
    intersection-of-valid-episodes variant that no recorded number corresponds to.

    On a suite row the draw is CLUSTERED on the within-wave env (:func:`cluster_draw`, plan D10) and
    both rows must share the pairing key -- ``(waves, episodes per wave, hold of each wave)`` -- and
    the full ``env_index`` set, or the pairing is refused. A legacy pair draws exactly as it always
    did. The block statistics iterate the blocks the specs name (:func:`block_names`).
    """
    if a.facing_closed_source != b.facing_closed_source:
        raise ResolutionError(
            f"{_pair_name(a, b)}: facing_closed sources differ ({a.facing_closed_source!r} vs "
            f"{b.facing_closed_source!r}). A proxy row is never pooled against a re-scored one "
            "(plan decision 6 / step-3 hand-off) -- re-score both sides or proxy both sides.")
    if (a.suite or b.suite) and a.pairing_key != b.pairing_key:
        raise ResolutionError(
            f"{_pair_name(a, b)}: the rows do not share the suite geometry (suite, waves, episodes "
            f"per wave, hold of each wave): {a.pairing_key} vs {b.pairing_key} -- identical "
            "(episodes, waves, suite) is the env_index pairing precondition (plan D2)")
    shared = sorted(set(a.env_index) & set(b.env_index))
    if len(shared) < MIN_BOOTSTRAP_N:
        raise ResolutionError(f"{_pair_name(a, b)}: only {len(shared)} shared env_index values")
    pos_a = {v: i for i, v in enumerate(a.env_index)}
    pos_b = {v: i for i, v in enumerate(b.env_index)}
    take_a = np.array([pos_a[v] for v in shared], dtype=np.int64)
    take_b = np.array([pos_b[v] for v in shared], dtype=np.int64)
    n = len(shared)
    rng = pair_rng(rng_seed, _pair_name(a, b))
    if a.suite:
        draw = cluster_draw(rng, a.episodes_per_wave, a.waves, B)
    else:
        draw = rng.integers(0, n, size=(B, n))
    ident = np.arange(n)[None, :]

    if specs is None:
        holds = tuple(h for h in a.hold_labels if h in set(b.hold_labels))
        specs = metric_specs(holds, suite=a.suite) + watch_specs(holds, suite=a.suite)

    out: dict[str, dict] = {}
    deltas: dict[str, np.ndarray] = {}
    for spec in specs:
        if any(c not in a.cols or c not in b.cols for c in spec.needs):
            continue
        cols_a = [a.cols[c][take_a] for c in spec.needs]
        cols_b = [b.cols[c][take_b] for c in spec.needs]
        n_valid = min(int(np.sum(np.isfinite(c))) for c in cols_a + cols_b)
        sub = draw if not a.suite else _wave_support(draw, cols_a + cols_b, a.episodes_per_wave)
        va = _reduce(spec.reducer, [c[sub] for c in cols_a])
        vb = _reduce(spec.reducer, [c[sub] for c in cols_b])
        delta = _signed(spec, va - vb)
        p = quantile_p(spec.level, one_sided=spec.one_sided)
        R = half_width(delta, p) if n_valid >= MIN_R_EVAL_N else None
        point_a = float(_reduce(spec.reducer, [c[ident] for c in cols_a])[0])
        point_b = float(_reduce(spec.reducer, [c[ident] for c in cols_b])[0])
        out[spec.mid] = {
            "R_eval": R, "level": spec.level, "one_sided": spec.one_sided, "block": spec.block,
            "blocks_all": list(spec.blocks_all), "hold": spec.hold,
            "n": n, "n_valid": n_valid, "below_min_n": n_valid < MIN_PAIRED_HOLD_N,
            "n_draws": int(np.sum(np.isfinite(delta))),
            "point_a": _jf(point_a), "point_b": _jf(point_b),
            "delta_point": _jf(_signed(spec, point_a - point_b)),
            "source": ("raw" if spec.mid.startswith("facing_closed_by_hold.")
                       and a.facing_closed_source == "metric" else
                       "proxy" if spec.mid.startswith("facing_closed_by_hold.") else "json"),
        }
        if spec.sign < 0:
            # recorded only where it is not the default, so every pre-Q record is unchanged
            out[spec.mid]["lower_is_better"] = True
        # Every delta is kept: which of them join a block mean is decided against the MERGED
        # ``R_eval,i`` in :func:`_standardised_blocks`, not against this pair's own.
        deltas[spec.mid] = delta

    blocks: dict[str, dict] = {}
    for block in block_names(specs):
        # PAIR-LOCAL (a diagnostic): standardised by this pair's own R. The published number is
        # re-derived against the merged R in :func:`_standardised_blocks`.
        members = [s.mid for s in specs if block in s.blocks_all and s.mid in deltas
                   and out[s.mid]["R_eval"] not in (None, 0.0)]
        if not members:
            continue
        std = np.stack([deltas[m] / out[m]["R_eval"] for m in members], axis=0)   # (k, B)
        z = _block_mean(std)
        blocks[block] = {"R_eval": half_width(z, quantile_p(BLOCK_LEVEL, one_sided=False)),
                         "level": BLOCK_LEVEL, "members": members,
                         "n_draws": int(np.sum(np.isfinite(z)))}
    return PairResult(a=a, b=b, name=_pair_name(a, b), n=n, metrics=out, blocks=blocks,
                      same_head=a.head == b.head, deltas=deltas)


def null_construction(cols: dict[str, np.ndarray], specs: Sequence[MetricSpec], *, n: int,
                      B: int = BOOTSTRAP_B, N: int = 1000,
                      rng: np.random.Generator | None = None,
                      R_metrics: dict[str, float | None] | None = None,
                      R_blocks: dict[str, float | None] | None = None,
                      waves: int = 1, scored_blocks: Sequence[str] | None = None,
                      scored_guards: Sequence[str] | None = None,
                      survival_guards: Sequence[str] = ()) -> dict:
    """Decision 6's NULL CANDIDATE: one run submitted against itself, ``N`` times.

    A trial is two INDEPENDENT episode resamples of the same run, one called candidate and one
    called control. Nothing differs between them but the resampling, so every block should read TIE
    at its 95 % level and every guard at its one-sided 99 % level, and the all-TIE frequency should
    land near ``0.95**2 * 0.99**4`` ~ 0.87 -- higher when blocks and guards share fall events, which
    they do, hence the plan's band [0.85, 0.98] rather than a target.

    ``R_metrics`` / ``R_blocks`` override the thresholds; left out, they are re-derived by the same
    construction on an independent set of ``B`` draws -- "the same paired bootstrap on the same-run
    resample distribution". That is the CALIBRATION check: a threshold taken from a distribution's
    own 95 % spread must be exceeded by fresh draws from that distribution 5 % of the time, so the
    per-block marginal TIE frequency pins the quantile arithmetic itself rather than any dataset.
    Pass D00's cross-head ``R`` instead to see how much slack the real thresholds carry.

    Returns ``{all_tie, blocks: {b: tie_freq}, guards: {g: tie_freq}, R_metrics, R_blocks, N}``.

    Rule version 2 (plan D9 item 4 / D10), all keyword-only and inert at their defaults (a legacy
    call is bit-identical): ``waves`` > 1 makes both resamples ENV-CLUSTERED (:func:`cluster_draw`
    on ``n // waves`` envs); ``scored_blocks`` / ``scored_guards`` name what enters ``all_tie`` --
    the RATIO system's three blocks and nine guards -- while every block and guard is still
    reported (default: decision 5's ``F``/``T`` and every guard spec); ``survival_guards`` are
    floored per trial at :func:`binomial_guard_floor` of that trial's two resampled survivals over
    the category's episode count (the column's non-NaN count), i.e. the guard is tested against
    ``max(R, R_binom)`` with no seed term -- matching decide() and the eval-noise nominal. A
    survival guard with no ``R`` is scored on the floor alone. The per-trial floors' median is
    reported under ``binomial_floor``.
    """
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    waves = int(waves)
    per_wave = int(n) // waves
    floor_ids = set(str(g) for g in survival_guards)

    def _deltas(rows: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if waves == 1:
            idx_a = rng.integers(0, n, size=(rows, n))
            idx_b = rng.integers(0, n, size=(rows, n))
        else:
            idx_a = cluster_draw(rng, per_wave, waves, rows)
            idx_b = cluster_draw(rng, per_wave, waves, rows)
        out: dict[str, np.ndarray] = {}
        pbar: dict[str, np.ndarray] = {}
        for spec in specs:
            if any(c not in cols for c in spec.needs):
                continue
            use = [cols[c] for c in spec.needs]
            sub_a = idx_a if waves == 1 else _wave_support(idx_a, use, per_wave)
            sub_b = idx_b if waves == 1 else _wave_support(idx_b, use, per_wave)
            va = _reduce(spec.reducer, [c[sub_a] for c in use])
            vb = _reduce(spec.reducer, [c[sub_b] for c in use])
            out[spec.mid] = _signed(spec, va - vb)
            if spec.mid in floor_ids:
                pbar[spec.mid] = (va + vb) / 2.0
        return out, pbar

    if R_metrics is None or R_blocks is None:
        ref, _ = _deltas(B)
        derived_m = {s.mid: half_width(ref[s.mid], quantile_p(s.level, one_sided=s.one_sided))
                     for s in specs if s.mid in ref}
        derived_b: dict[str, float | None] = {}
        for block in block_names(specs):
            members = [s.mid for s in specs if block in s.blocks_all
                       and s.mid in ref and derived_m.get(s.mid)]
            if not members:
                continue
            z = block_mean(np.stack([ref[m] / derived_m[m] for m in members], axis=0))
            derived_b[block] = half_width(z, quantile_p(BLOCK_LEVEL, one_sided=False))
        R_metrics = R_metrics if R_metrics is not None else derived_m
        R_blocks = R_blocks if R_blocks is not None else derived_b

    trial, trial_pbar = _deltas(N)
    tie_blocks: dict[str, np.ndarray] = {}
    for block, R_F in sorted(R_blocks.items()):
        members = [s.mid for s in specs if block in s.blocks_all
                   and s.mid in trial and R_metrics.get(s.mid)]
        if not members or R_F is None:
            continue
        z = block_mean(np.stack([trial[m] / R_metrics[m] for m in members], axis=0))
        tie_blocks[block] = ~((z > R_F) | (z < -R_F)) | ~np.isfinite(z)
    tie_guards: dict[str, np.ndarray] = {}
    floors: dict[str, float | None] = {}
    for spec in specs:
        if spec.block != "guard" or spec.mid not in trial:
            continue
        R = R_metrics.get(spec.mid)
        d = trial[spec.mid]
        if spec.mid in floor_ids:
            n_c = int(np.sum(np.isfinite(cols[spec.needs[0]])))
            pb = np.clip(trial_pbar[spec.mid], *BINOM_P_CLIP)
            z99 = NormalDist().inv_cdf(GUARD_LEVEL)
            r_binom = (z99 * np.sqrt(2.0 * pb * (1.0 - pb) / n_c) if n_c > 0
                       else np.full(d.shape, np.nan))
            base = 0.0 if R is None or not math.isfinite(float(R)) else float(R)
            thr = np.where(np.isfinite(r_binom), np.maximum(base, r_binom), base)
            floors[spec.mid] = _jf(np.nanmedian(r_binom)) if np.any(np.isfinite(r_binom)) else None
            if R is None and not np.any(thr > 0):
                continue
            tie_guards[spec.mid] = ~(d < -thr) | ~np.isfinite(d)
            continue
        if R is None:
            continue
        tie_guards[spec.mid] = ~(d < -R) | ~np.isfinite(d)
    scored_b = LEGACY_SCORED_BLOCKS if scored_blocks is None else tuple(scored_blocks)
    scored = [v for k, v in sorted(tie_blocks.items()) if k in scored_b]
    scored += [v for k, v in sorted(tie_guards.items())
               if scored_guards is None or k in set(scored_guards)]
    all_tie = np.logical_and.reduce(scored) if scored else np.ones(N, dtype=bool)
    result = {"N": int(N), "B": int(B),
              "all_tie": float(np.mean(all_tie)),
              "blocks": {k: float(np.mean(v)) for k, v in sorted(tie_blocks.items())},
              "guards": {k: float(np.mean(v)) for k, v in sorted(tie_guards.items())},
              "R_metrics": {k: _jf(v) for k, v in sorted(R_metrics.items())},
              "R_blocks": {k: _jf(v) for k, v in sorted(R_blocks.items())}}
    result["scored"] = {"blocks": sorted(k for k in tie_blocks if k in scored_b),
                        "guards": sorted(k for k in tie_guards
                                         if scored_guards is None or k in set(scored_guards))}
    if floor_ids:
        result["binomial_floor"] = {k: v for k, v in sorted(floors.items())}
    return result


def _jf(x: Any) -> float | None:
    """JSON-safe float: ``None`` for NaN / inf / None, a plain ``float`` otherwise."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _standardised_blocks(pairs: Sequence[PairResult],
                         metrics: dict[str, dict]) -> dict[str, dict]:
    """``R_F^eval`` per block: the half-width of ``z_F`` standardised by the MERGED ``R_eval,i``.

    Decision 5 says "one standardiser everywhere: ``R_i = R_eval,i``" and decision 6 defines
    ``R_F^eval`` as "the paired-bootstrap 95 % half-width of ``z_F`` ITSELF". ``z_F`` is the quantity
    ``decide()`` computes, and it divides each member by the PUBLISHED ``R_eval,i`` -- the max over
    the pairs, not the half-width of whichever pair is being drawn. So the block statistic must be
    bootstrapped in those same units, which is only possible once every pair has been drawn and
    merged. Standardising each pair by its own R (the obvious one-pass version) measures a slightly
    different statistic and lands 5-7 % wide on the recorded inputs.

    The max over pairs is kept, as for the members: the resolution has to cover the worse-behaved
    pair. ``per_pair`` records each pair's own number and ``members_used`` which members it carried.
    """
    member_of: dict[str, list[str]] = {}
    for mid, m in sorted(metrics.items()):
        R = m.get("R_eval")
        if R is None or float(R) == 0.0:
            # A zero half-width is a real answer for a same-head pair (the paired construction
            # cancels identical inputs exactly) but it cannot standardise anything.
            continue
        for block in m.get("blocks_all") or []:
            member_of.setdefault(block, []).append(mid)
    blocks: dict[str, dict] = {}
    order = [b for b in BLOCK_ORDER if b in member_of] + sorted(set(member_of) - set(BLOCK_ORDER))
    for block in order:
        members = member_of.get(block) or []
        if not members:
            continue
        slot: dict[str, Any] = {"R_eval": None, "level": BLOCK_LEVEL, "members": list(members),
                                "pairs_used": [], "per_pair": {}, "members_used": {},
                                "n_draws": None}
        for pr in pairs:
            use = [m for m in members if m in pr.deltas]
            if not use:
                continue
            z = _block_mean(np.stack([pr.deltas[m] / float(metrics[m]["R_eval"]) for m in use],
                                     axis=0))
            R = half_width(z, quantile_p(BLOCK_LEVEL, one_sided=False))
            slot["per_pair"][pr.name] = _jf(R)
            slot["members_used"][pr.name] = list(use)
            if R is None:
                continue
            slot["pairs_used"].append(pr.name)
            n_draws = int(np.sum(np.isfinite(z)))
            slot["n_draws"] = n_draws if slot["n_draws"] is None else min(slot["n_draws"], n_draws)
            if slot["R_eval"] is None or float(R) > float(slot["R_eval"]):
                slot["R_eval"] = _jf(R)
        blocks[block] = slot
    return blocks


def _merge_pairs(pairs: Sequence[PairResult]) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    """``R_eval`` per metric and per block = the MAX over the pairs that produced one.

    The resolution has to cover the worse-behaved pair: a threshold set from the tighter pair reads
    the other pair's noise as signal, and a candidate adopted on noise costs the whole branch. Every
    pair's own number stays in ``per_pair``, so the max is auditable rather than asserted.

    The blocks are a SECOND pass (:func:`_standardised_blocks`): the merged per-metric ``R_eval,i``
    has to exist before ``z_F`` can be bootstrapped in the units ``decide()`` reads.
    """
    metrics: dict[str, dict] = {}
    notes: list[str] = []
    for pr in pairs:
        for mid, m in pr.metrics.items():
            slot = metrics.setdefault(mid, {
                "R_eval": None, "level": m["level"], "one_sided": m["one_sided"],
                "block": m["block"], "blocks_all": m["blocks_all"], "hold": m["hold"],
                "source": m["source"], "pairs_used": [], "per_pair": {}, "n": m["n"],
                "n_valid": m["n_valid"], "below_min_n": bool(m["below_min_n"]), "points": {},
                "n_draws": None,
                **({"lower_is_better": True} if m.get("lower_is_better") else {}),
            })
            slot["per_pair"][pr.name] = _jf(m["R_eval"])
            slot["points"][pr.name] = {"a": m["point_a"], "b": m["point_b"],
                                       "delta": m["delta_point"]}
            if m["R_eval"] is None:
                continue
            slot["pairs_used"].append(pr.name)
            slot["n"] = min(slot["n"], m["n"])
            slot["n_valid"] = min(slot["n_valid"], m["n_valid"])
            slot["n_draws"] = (m["n_draws"] if slot["n_draws"] is None
                               else min(slot["n_draws"], m["n_draws"]))
            slot["below_min_n"] = slot["below_min_n"] or bool(m["below_min_n"])
            if slot["R_eval"] is None or float(m["R_eval"]) > float(slot["R_eval"]):
                slot["R_eval"] = _jf(m["R_eval"])
    blocks = _standardised_blocks(pairs, metrics)
    for mid, slot in sorted(metrics.items()):
        if slot["R_eval"] is None:
            notes.append(f"MISSING: {mid} produced no half-width on any pair "
                         f"(n_valid {slot['n_valid']}) -- decision 5's MISSING rules apply")
        elif slot["below_min_n"]:
            notes.append(f"thin: {mid} has {slot['n_valid']} usable episodes (< {MIN_PAIRED_HOLD_N}); "
                         "its half-width is reported, not dropped")
    return metrics, blocks, notes


def _mde(block: dict, metrics: dict[str, dict]) -> dict[str, float | None]:
    """The move each member must make for the block to read BETTER, in the member's own units.

    A block is BETTER when ``z > R_F``; if every member moves by ``c * R_i`` then ``z == c``, so the
    smallest uniform move that clears the block is ``R_F * R_i`` per member. That is the number
    "effective MDE per block" names in the RESOLUTION line.
    """
    R_F = block.get("R_eval")
    if R_F is None:
        return {}
    return {mid: _jf(float(R_F) * float(metrics[mid]["R_eval"]))
            for mid in block.get("members", []) if metrics.get(mid, {}).get("R_eval") is not None}


# ------------------------------------------------------------------------------------ the top level
def _control_head(rows: Sequence[Row], override: str | None) -> str | None:
    """The head every cross-head pair is against.

    ``--control-head`` wins. Otherwise it is the head of the FIRST accepted row listed for the
    scenario, which reproduces the plan's pairing (1c-004 vs 2b, 1c-004 vs 2a-reeval) from the order
    ``resolution_inputs`` lists the rows in, with no extra yaml key. ``candidates.yaml`` may also
    carry a ``head:`` label per entry, purely so the record and the DECISION lines read in campaign
    names rather than checkpoint stamps.
    """
    if override:
        return override
    for r in rows:
        if r.accepted:
            return r.head
    return None


def _floor_target(row: Row, candidates: Sequence[Row]) -> Row | None:
    """The row a FLAGGED process-floor row is the re-eval of: same head, or the same checkpoint.

    ``floor_of`` wins when the yaml names it; otherwise the checkpoint is the identity (two evals of
    one checkpoint are one head's two evals, whatever they are labelled), and the head label is the
    last resort for a record whose ``run_meta`` carries no checkpoint.
    """
    for pick in (lambda r: row.floor_of is not None and r.head == row.floor_of,
                 lambda r: bool(row.checkpoint) and r.checkpoint == row.checkpoint,
                 lambda r: r.head == row.head):
        for other in candidates:
            if other is row:
                continue
            if pick(other):
                return other
    return None


def build_pairs(rows: Sequence[Row], *, control: str | None, B: int, rng_seed: int,
                specs: Sequence[MetricSpec] | None = None
                ) -> tuple[list[PairResult], list[PairResult], list[str]]:
    """``(cross-head pairs, process-floor pairs, notes)`` for one scenario row.

    Cross-head pairs are ``(control, other)`` for every other accepted head -- the plan's two pairs
    (1c-004 vs 2b, 1c-004 vs 2a-reeval), not all three unordered combinations: the resolution a
    candidate-vs-control decision needs is the spread against the CONTROL. Two accepted rows that
    share a head are that head's process-noise floor instead.

    A row flagged ``process_floor_pair`` / ``role: process_floor`` is EXCLUDED from cross-head
    pairing whatever it is labelled and paired against the row it is the re-eval of
    (:func:`_floor_target`). Decision 6: "the same-head pair ... is reported separately as the
    process-noise floor, NEVER as the candidate R" -- and a flagged row that matches nothing is a
    refusal, not a silent third cross-head arm whose half-widths would be folded into the candidate
    ``R`` by the max.
    """
    notes: list[str] = []
    accepted = [r for r in rows if r.accepted]
    flagged = [r for r in accepted if r.process_floor_pair]
    pairing = [r for r in accepted if not r.process_floor_pair]
    ctrl = _control_head(pairing, control)
    by_head: dict[str, list[Row]] = {}
    for r in pairing:
        by_head.setdefault(r.head, []).append(r)
    cross: list[PairResult] = []
    floor: list[PairResult] = []
    for row in flagged:
        target = _floor_target(row, pairing)
        if target is None:
            raise ResolutionError(
                f"{row.label} is flagged process_floor_pair but no accepted row of "
                f"{row.row} is the eval it is the floor of (looked for floor_of "
                f"{row.floor_of!r}, checkpoint {row.checkpoint!r}, head {row.head!r}). A floor row "
                "that formed no pair would otherwise become a third cross-head arm and enter the "
                "candidate R, which decision 6 forbids.")
        floor.append(bootstrap_pair(target, row, B=B, rng_seed=rng_seed, specs=specs))
    for head, group in sorted(by_head.items()):
        # Two UNFLAGGED rows of one head are also a floor pair: the head label is the identity the
        # plan's "same-head pair" names. The first one listed is the cross-head arm.
        for extra in group[1:]:
            floor.append(bootstrap_pair(group[0], extra, B=B, rng_seed=rng_seed, specs=specs))
    if ctrl is None or ctrl not in by_head:
        if accepted:
            notes.append(f"no control head on {accepted[0].row}: {ctrl!r} is not among "
                         f"{sorted(by_head)} -- no cross-head pair")
        return cross, floor, notes
    base = by_head[ctrl][0]
    for head, group in sorted(by_head.items()):
        if head == ctrl:
            continue
        cross.append(bootstrap_pair(base, group[0], B=B, rng_seed=rng_seed, specs=specs))
    if not cross:
        notes.append(f"{base.row}: only one accepted head ({ctrl}) -- no cross-head pair, no R")
    return cross, floor, notes


def _borrowed_watch(*sources: dict[str, dict]) -> dict[str, dict]:
    """The right-handed holds' entries, borrowing their left-handed twin's R (decision 6).

    ``lat_r`` / ``move_turn_r`` exist only on the curriculum row, which no D00 input carries, so
    their own half-width cannot be measured yet. Until ``--curriculum-runs`` re-derives them they
    take ``lat_l`` / ``move_turn``'s, marked ``borrowed_from`` so a reader can never mistake a
    borrowed number for a measured one. WATCH only either way (decision 5).
    """
    pool: dict[str, dict] = {}
    for src_map in sources:
        pool.update(src_map)
    out: dict[str, dict] = {}
    for right, left in sorted(BORROW.items()):
        for stem in ("ratio_by_hold", "achieved_speed_by_hold", "facing_closed_by_hold",
                     "yaw_progress_by_hold", "net_yaw_by_hold"):
            src = pool.get(f"{stem}.{left}")
            if src is None or src.get("R_eval") is None:
                continue
            if right == "lat_r" and stem in ("facing_closed_by_hold", "yaw_progress_by_hold",
                                             "net_yaw_by_hold"):
                continue
            out[f"{stem}.{right}"] = {"R_eval": src["R_eval"], "level": src["level"],
                                      "one_sided": src["one_sided"], "block": "watch",
                                      "blocks_all": [], "hold": right, "source": src["source"],
                                      "borrowed_from": f"{stem}.{left}",
                                      "pairs_used": list(src.get("pairs_used") or []),
                                      "n": src.get("n"), "n_valid": src.get("n_valid")}
    return out


def _resolution_line(blocks: dict[str, dict], metrics: dict[str, dict], B: int, *,
                     label: str = "D00", tail: str = "",
                     draw: str = "paired episode bootstrap") -> str:
    """One line naming the effective MDE per block and per guard (Verification 5).

    The blocks are the record's own (in :data:`BLOCK_ORDER`), never a literal list: a v2 record
    names ``F_mt`` / ``F_mt_abs`` too. ``label`` is the record id (``D00`` / ``D00s``); ``tail`` an
    optional closing field (the v2 null all-TIE against its nominal).
    """
    parts = []
    order = [b for b in BLOCK_ORDER if b in blocks] + sorted(set(blocks) - set(BLOCK_ORDER))
    for block in order:
        b = blocks.get(block)
        if not b or b.get("R_eval") is None:
            continue
        mde = {k: v for k, v in (_mde(b, metrics) or {}).items() if v is not None}
        head = f"{block} R={float(b['R_eval']):.3f}@{b['level']:.2f}"
        if mde:
            head += f" (MDE per member {min(mde.values()):.3f}..{max(mde.values()):.3f})"
        parts.append(head)
    guards = [f"{m}+-{float(v['R_eval']):.3f}" for m, v in sorted(metrics.items())
              if v.get("block") == "guard" and v.get("R_eval") is not None]
    return ("RESOLUTION %s | %s B=%d on env_index | blocks %s | "
            "guards @%.2f one-sided %s%s" % (label, draw, int(B), "; ".join(parts) or "none",
                                             GUARD_LEVEL, ", ".join(guards) or "none", tail))


#: The legacy rows whose ``by_row`` tables ``D00s`` copies verbatim from ``D00`` (plan D7).
D00_COPIED_ROWS = ("light__omni", "ring__omni")


def _positive(v: Any) -> bool:
    """A finite, strictly positive number (the completeness test's notion of "resolved")."""
    f = _jf(v)
    return f is not None and f > 0.0


def _is_survival_guard(mid: str) -> bool:
    parsed = _ec().parse_cat_id(mid) if str(mid).startswith("cat.") else None
    return parsed is not None and parsed[1] is None and parsed[2] == "survival"


def dual_guards(rule: dict) -> set[str]:
    """The guard ids of an EXPANDED rule that are also block members (resolved under ``@guard``)."""
    members = {str(m) for body in (rule.get("blocks") or {}).values() for m in body.get("metrics") or []}
    return {str(g) for g in (rule.get("guards") or {}).get("metrics") or [] if str(g) in members}


def guard_metric_key(mid: str, metrics: dict) -> str:
    """The key a guard's half-width is recorded under: ``<mid>@guard`` when the record carries one
    (a guard that is also a block member), else ``mid`` itself (every guard before block Q)."""
    key = _ec().guard_key(mid)
    return key if key in (metrics or {}) else str(mid)


def v2_scoring_rule(rule: dict, decisive_at: Sequence[str] | None) -> tuple[dict, dict]:
    """``(union rule, systems)`` -- the v2 rule expanded for ``D00s`` (plan D5/D10).

    The UNION of ``eval_categories.expand_rule`` in both unit systems at ``decisive = present =
    decisive_at`` (B0 trains all seven categories), so every block, member and guard either system
    can score has an R. ``systems`` records, per unit system, the non-empty blocks, their members
    and the guards -- the null construction scores ``systems['ratio']`` and the completeness test
    walks both.
    """
    ec = _ec()
    if decisive_at is None:
        raise ResolutionError("decision_rule has rule_version 2: resolve() needs decisive_at "
                              "(the categories B0's final-stage composed env trains -- the driver "
                              "passes sorted(decisive(B0)); the CLI's --decisive-at)")
    dec = [str(c) for c in decisive_at]
    unknown = [c for c in dec if c not in ec.CATEGORIES]
    if unknown or not dec:
        raise ResolutionError(f"decisive_at {dec}: unknown categories {unknown} (known: "
                              f"{list(ec.CATEGORIES)})" if unknown else "decisive_at is empty")
    try:
        union = ec.expand_rule_union(rule, dec, dec)
        by_units = {"ratio": ec.expand_rule(rule, dec, dec, absolute_units=False),
                    "absolute": ec.expand_rule(rule, dec, dec, absolute_units=True)}
    except ValueError as exc:
        raise ResolutionError(f"decision_rule (rule_version 2) does not expand: {exc}") from exc
    systems = {}
    for units, exp in by_units.items():
        blocks = [b for b, body in exp["blocks"].items() if body["metrics"]]
        systems[units] = {"blocks": blocks, "na_blocks": list(exp["na_blocks"]),
                          "members": {b: list(exp["blocks"][b]["metrics"]) for b in blocks},
                          "guards": list(exp["guards"]["metrics"])}
    return union, systems


def null_nominal(system: dict) -> float:
    """``0.95 ** blocks * 0.99 ** guards`` -- the independence-model all-TIE rate of one unit system
    (plan D9 item 4: 0.95**3 * 0.99**9 ~ 0.78 in the ratio system at B0's decisive set)."""
    return float(BLOCK_LEVEL ** len(system["blocks"]) * GUARD_LEVEL ** len(system["guards"]))


def acceptance_d00s(blocks: dict[str, dict], metrics: dict[str, dict], systems: dict,
                    null: dict | None, nominal: float) -> dict:
    """``{ok, failures}`` -- D00s' acceptance (plan D9 item 4), computed BEFORE anything is written.

    (1) The CALIBRATED env-clustered null (R re-derived on the same resample distribution, survival
    guards floored at ``R_binom``) all-TIE inside ``[nominal - 0.03, 0.98]``. (2) Completeness, in
    BOTH unit systems: a finite positive ``R_eval`` for every non-empty block, every member of those
    blocks, and every non-survival guard (``cat.move_turn.ratio`` / ``achieved_speed``,
    ``cat_tripod_min``) -- without it an unresolved block is a permanent TIE the band cannot see.
    ``survival`` guards are exempt: a None or 0 ``R_eval`` is scored on the binomial floor.
    """
    failures: list[str] = []
    lo, hi = nominal - NULL_BAND_SLACK, NULL_BAND_HI
    if null is None:
        failures.append("no null construction: no accepted control row on the screen row")
    elif not (lo <= float(null["all_tie"]) <= hi):
        failures.append(f"null all-TIE {float(null['all_tie']):.3f} outside [{lo:.3f}, {hi:.2f}] "
                        f"(nominal {nominal:.3f}; env-clustered, survival guards floored)")
    for units in ("ratio", "absolute"):
        sysd = systems[units]
        for b in sysd["blocks"]:
            R = (blocks.get(b) or {}).get("R_eval")
            if not _positive(R):
                failures.append(f"block {b} ({units}): R_eval {R!r} is not finite and positive")
            for m in sysd["members"][b]:
                R = (metrics.get(m) or {}).get("R_eval")
                if not _positive(R):
                    failures.append(f"{b} member {m}: R_eval {R!r} is not finite and positive")
        for g in sysd["guards"]:
            if _is_survival_guard(g):
                continue
            key = guard_metric_key(g, metrics)
            R = (metrics.get(key) or {}).get("R_eval")
            if not _positive(R):
                failures.append(f"guard {key} ({units}): R_eval {R!r} is not finite and positive")
    failures = list(dict.fromkeys(failures))
    return {"ok": not failures, "failures": failures}


def _spread_run_dir(path: str | Path) -> Path:
    """A suite stamp dir as given, or the newest fresh stamp under an ``eval_dir`` root."""
    p = _repo_path(path)
    if (p / "run_meta.json").exists():
        return p
    return resolve_eval_dir(p, SCREEN_ROW)


def member_spread_check(run_dirs: Sequence[str | Path], rule: dict, *,
                        decisive_at: Sequence[str] | None = None, B: int | None = None,
                        rng_seed: int | None = None, heads: Sequence[str] | None = None,
                        verify: bool = True) -> dict:
    """PRE-LAUNCH: which scored members of a v2 rule would carry no finite positive ``R`` in D00s.

    The motion-quality campaign's zero-spread check (plan step 4: "D00s needs a finite R > 0 for
    every member ... any member with zero spread on X08:c6 is demoted to WATCH before launch"). CPU
    only: it reads recorded suite stamps. ``run_dirs[0]`` is the CONTROL (the incumbent's suite
    re-eval); each further run dir forms one cross-head pair with it, drawn with D00s' own
    construction (:func:`bootstrap_pair`, the rule's ``B`` / ``rng_seed``, the pair's own stream),
    so with the recorded reeval stamps it reproduces D00s' per-member ``R_eval`` exactly.

    Per scored member (every block member and every non-survival guard of the rule expanded at
    ``decisive_at``, both unit systems; survival guards are floored at ``R_binom`` and exempt):
    ``same_run`` = each row's own resample spread (two independent env-clustered resamples of that
    row, :func:`null_construction`'s derived ``R``), ``cross`` = each pair's ``R_eval``; ``zero`` when
    the control's own spread or any pair's ``R`` is None or 0 -- a median that no resample moves (a
    count that is 0 on most episodes, a censored settle time, a fraction pinned at 0). Such a member
    cannot pass D00s' acceptance; the campaign demotes it (``per_category.demoted``), never
    ``--accept-resolution``. A member with NO finite value on some row is ``absent`` instead (the
    stamp predates the metric, e.g. a pre-step-1 eval): re-evaluate, never demote on it.

    Returns ``{rows, members: {key: {...}}, zero: [key, ...], absent: [key, ...],
    demote: {category: {key: why}}}`` (``demote`` covers the ``zero`` members only).
    """
    ec = _ec()
    if rule_version(rule) != 2:
        raise ResolutionError("member_spread_check takes a rule_version 2 decision rule")
    dat = list(decisive_at) if decisive_at else list(ec.CATEGORIES)
    scoring, _systems = v2_scoring_rule(rule, dat)
    boot = dict(rule.get("bootstrap") or {})
    B = int(B if B is not None else boot.get("B", BOOTSTRAP_B))
    rng_seed = int(rng_seed if rng_seed is not None else boot.get("rng_seed", RNG_SEED))
    if not run_dirs:
        raise ResolutionError("member_spread_check needs at least the control's suite stamp")
    rows = []
    for i, d in enumerate(run_dirs):
        head = str(heads[i]) if heads and i < len(heads) and heads[i] else None
        entry = {"run_dir": str(_spread_run_dir(d))}
        if head:
            entry["head"] = head
        rows.append(load_row(entry, ring_proxy=False, rule=scoring, verify=verify))
    if not all(r.suite for r in rows):
        raise ResolutionError("member_spread_check reads category_suite stamps only: "
                              + ", ".join(r.label for r in rows if not r.suite))
    specs, unresolved = specs_from_rule(scoring, rows[0].hold_labels, suite=True)
    scored = [s for s in specs if s.block != "watch" and not _is_survival_guard(s.mid)]
    same: dict[str, dict[str, float | None]] = {}
    for row in rows:
        nul = null_construction(row.cols, scored, n=len(row.env_index), B=B, N=1,
                                rng=pair_rng(rng_seed, f"spread::{row.label}"), waves=row.waves)
        same[row.label] = dict(nul["R_metrics"])
    cross: dict[str, dict[str, float | None]] = {}
    for other in rows[1:]:
        pr = bootstrap_pair(rows[0], other, B=B, rng_seed=rng_seed, specs=scored)
        cross[pr.name] = {k: _jf(v.get("R_eval")) for k, v in pr.metrics.items()}
    members: dict[str, dict] = {}
    zero: list[str] = []
    absent: list[str] = []
    demote: dict[str, dict[str, str]] = {}
    ctrl = rows[0].label
    for spec in scored:
        key = spec.mid
        s_ctrl = _jf(same[ctrl].get(key))
        pairs = {name: tab.get(key) for name, tab in cross.items()}
        is_zero = (not _positive(s_ctrl)) or any(not _positive(v) for v in pairs.values())
        # A member with NO value on some row is not a spread question: the stamp predates the
        # metric (a pre-step-1 eval), or the key is N/A there. Re-evaluate; never demote on it.
        n_finite = {r.label: int(min(np.sum(np.isfinite(r.cols[c])) if c in r.cols else 0
                                     for c in spec.needs)) for r in rows}
        members[key] = {"block": spec.block, "blocks_all": list(spec.blocks_all),
                        "same_run": {lab: _jf(tab.get(key)) for lab, tab in same.items()},
                        "cross": pairs, "zero": is_zero, "sign": spec.sign, "n_finite": n_finite}
        if not is_zero:
            continue
        if any(v == 0 for v in n_finite.values()):
            members[key]["absent"] = True
            absent.append(key)
            continue
        zero.append(key)
        parsed = ec.parse_cat_id(ec.base_metric_id(key))
        if parsed is not None and parsed[1] is None:
            c, _h, k = parsed
            why = (f"zero spread on {ctrl} ({rows[0].run_dir.name}): same-run R {s_ctrl}"
                   + (", cross-head R " + ", ".join(f"{n} {v}" for n, v in pairs.items()) if pairs else "")
                   + " -- no finite R in D00s; demoted to WATCH before launch")
            demote.setdefault(c, {})[k] = why
    return {"rows": [{"head": r.head, "run_dir": str(r.run_dir), "n": r.n_episodes} for r in rows],
            "B": B, "rng_seed": rng_seed, "decisive_at": dat, "members": members, "zero": zero,
            "absent": absent, "demote": demote, "unresolved": list(unresolved)}


def resolve(candidates: dict, *, bootstrap: int | None = None, rng_seed: int | None = None,
            ring_proxy: bool = False, control_head: str | None = None,
            curriculum_runs: Sequence[str | Path] = (), verify: bool = True,
            candidates_sha256: str | None = None, eval_sha: str | None = None,
            decisive_at: Sequence[str] | None = None, d00: dict | None = None,
            supersedes: str | None = None) -> dict:
    """Build the ``resolution.json`` payload (schema 1) from a parsed ``candidates.yaml``.

    ``bootstrap`` / ``rng_seed`` default to ``decision_rule.bootstrap.{B, rng_seed}`` when the yaml
    declares them (the yaml is the campaign's specification) and to :data:`BOOTSTRAP_B` /
    :data:`RNG_SEED` otherwise; an explicit argument always wins.

    Deterministic under ``rng_seed``, and stable under a CHANGED INPUT LIST: every pair draws from
    its own stream, keyed to ``(rng_seed, pair name)`` by :func:`pair_rng`, so re-running the tool
    after the process-floor eval lands leaves the existing pairs' numbers bit-identical instead of
    silently re-fitting every recorded ``R``.

    **Rule version 2 (``D00s``, plan D9 item 3 / D10).** The driver calls this IN PROCESS with
    ``eval_sha`` (``lad.eval_sha``, recorded), ``decisive_at`` (``sorted(decisive(B0))``, used for
    both decisive and present), ``d00`` (the parsed ``D00`` record: its light/ring ``by_row`` tables
    and ``decision_rule``) and ``supersedes`` (``"D00"``), and ``resolution_inputs`` whose
    ``role: pair`` entries carry their recorded ``stamp``. The rule is expanded here
    (:func:`v2_scoring_rule`); the screen row is :data:`SCREEN_ROW`; ``BORROW`` and
    ``curriculum_runs`` are retired (the latter is refused). The payload gains ``rule_version``,
    ``eval_sha``, ``supersedes``, ``decisive_at``, ``scored_systems``, ``by_category_n``,
    ``null_nominal``, ``null_survival_tie`` (per survival guard, both nulls), ``by_row_rule`` /
    ``by_row_source`` (D00's copies) and ``acceptance = {ok, failures}`` (:func:`acceptance_d00s`).
    Nothing is written here -- the caller decides from ``acceptance`` whether
    :func:`write_resolution` runs. A rule-version-1 call is unchanged, bit for bit.
    """
    entries = list(candidates.get("resolution_inputs") or [])
    if not entries:
        raise ResolutionError("candidates.yaml carries no resolution_inputs")
    rule = dict(candidates.get("decision_rule") or {})
    v2 = rule_version(rule) == 2
    boot_cfg = dict(rule.get("bootstrap") or {})
    bootstrap = int(bootstrap if bootstrap is not None else boot_cfg.get("B", BOOTSTRAP_B))
    rng_seed = int(rng_seed if rng_seed is not None else boot_cfg.get("rng_seed", RNG_SEED))
    paired_on = str(boot_cfg.get("paired_on", "env_index"))
    if paired_on != "env_index":
        raise ResolutionError(f"decision_rule.bootstrap.paired_on is {paired_on!r}; this module "
                              "pairs on env_index only (plan decision 6)")
    floors = {str(k): float(v) for k, v in (rule.get("charter_floors") or CHARTER_FLOORS).items()}
    cite = str(rule.get("charter_cite") or CHARTER_CITE)
    # With no control named, the control is "the head of the first accepted row listed for that
    # row" -- which makes every recorded `R_eval` (through the max over pairs) depend on the ORDER
    # of `resolution_inputs`, with nothing in the yaml saying so. `decision_rule.control_head` pins
    # it, so a reorder of the input list can no longer re-fit D00 silently. An explicit
    # `--control-head` still wins.
    control_head = control_head or (rule.get("control_head") or None)
    notes: list[str] = []
    systems: dict | None = None
    if v2:
        if curriculum_runs:
            raise ResolutionError("--curriculum-runs is retired under rule_version 2: the suite "
                                  "measures both hands of every paired category (lat_r and "
                                  "move_turn_r included), so nothing is borrowed or re-derived")
        scoring_rule, systems = v2_scoring_rule(rule, decisive_at)
        screen_row = SCREEN_ROW
    else:
        if decisive_at is not None or d00 is not None:
            raise ResolutionError("decisive_at / d00 are rule_version 2 arguments; this "
                                  "decision_rule is rule_version 1")
        scoring_rule = rule
        screen_row = LEGACY_SCREEN_ROW

    rows = [load_row(e, ring_proxy=ring_proxy, rule=scoring_rule, verify=verify) for e in entries]
    by_scenario: dict[str, list[Row]] = {}
    for r in rows:
        by_scenario.setdefault(r.row, []).append(r)

    for r in rows:
        if not r.provenance_ok and not r.waived:
            notes.append(f"REFUSED input {r.run_dir}: {r.reason}")

    by_row: dict[str, dict] = {}
    pair_records: list[dict] = []
    floor_metrics: dict[str, dict] = {}
    unresolved_rows: dict[tuple[str, str], list[str]] = {}
    for sid in sorted(by_scenario):
        first = by_scenario[sid][0]
        specs, unresolved = specs_from_rule(scoring_rule, first.hold_labels, suite=first.suite)
        # Under v2 the legacy rows are informational: listing every category id as "not on this
        # row" for each of them would bury the notes that matter.
        if not v2 or any(r.accepted for r in by_scenario[sid]):
            for mid in unresolved:
                key = (mid, why_unresolved(mid, first.hold_labels, suite=first.suite))
                unresolved_rows.setdefault(key, []).append(sid)
        cross, floor, n2 = build_pairs(by_scenario[sid], control=control_head, B=bootstrap,
                                       rng_seed=rng_seed, specs=specs)
        notes.extend(n2)
        if cross:
            merged, blocks, n3 = _merge_pairs(cross)
            notes.extend(f"{sid}: {m}" for m in n3)
            # ``metrics`` stays exactly the rule's blocks and guards so it is a clean input to
            # decide(); everything the rule only WATCHES goes to watch_metrics.
            scored = {k: v for k, v in merged.items() if v["block"] != "watch"}
            watched = {k: v for k, v in merged.items() if v["block"] == "watch"}
            for block in blocks.values():
                block["mde"] = _mde(block, scored)
            by_row[sid] = {"metrics": scored, "watch_metrics": watched, "blocks": blocks,
                           "control_head": cross[0].a.head,
                           "heads": sorted({p.b.head for p in cross} | {cross[0].a.head})}
        for kind, group in (("cross_head", cross), ("process_floor", floor)):
            for pr in group:
                pair_records.append({"a": pr.a.head, "b": pr.b.head, "row": sid, "n": pr.n,
                                     "kind": kind, "name": pr.name})
        for pr in floor:
            for mid, m in pr.metrics.items():
                cur = floor_metrics.get(mid)
                R = _jf(m["R_eval"])
                if R is None:
                    continue
                if cur is None or R > cur["R"]:
                    floor_metrics[mid] = {"R": R, "pair": pr.name, "level": m["level"]}

    for (mid, why), sids in sorted(unresolved_rows.items()):
        notes.append(f"no half-width for {mid} on {', '.join(sorted(sids))} -- {why}")

    if screen_row not in by_row:
        raise ResolutionError(
            f"no cross-head pair on the screen row {screen_row!r}: every candidate screen is scored "
            f"there, so {'D00s' if v2 else 'D00'} without it has no R. Rows resolved: "
            f"{sorted(by_row)}")
    metrics = {k: dict(v) for k, v in by_row[screen_row]["metrics"].items()}
    blocks = {k: dict(v) for k, v in by_row[screen_row]["blocks"].items()}
    watch = {k: dict(v) for k, v in by_row[screen_row]["watch_metrics"].items()}
    if v2:
        notes.append("rule_version 2: both hands of every paired category are measured on the "
                     "suite -- BORROW and --curriculum-runs are retired")
    else:
        watch.update(_borrowed_watch(metrics, watch))

    if v2:
        pass
    elif curriculum_runs:
        watch, cnotes, crow = _rederive_curriculum(curriculum_runs, watch, ring_proxy=ring_proxy,
                                                   bootstrap=bootstrap, rng_seed=rng_seed,
                                                   verify=verify, rule=rule)
        notes.extend(cnotes)
        if crow:
            by_row[CURRICULUM_ROW] = crow
    else:
        notes.append("lat_r / move_turn_r borrow lat_l / move_turn's R (no curriculum row among the "
                     "D00 inputs); re-derive with --curriculum-runs at the first milestone")

    for mid, floor in sorted(floors.items()):
        # Under v2 the cited ids (completion, tripod_walking_dir) are WATCH: their R is in watch.
        got = ((metrics.get(mid) or (watch.get(mid) if v2 else None) or {}).get("R_eval"))
        if got is None:
            notes.append(f"charter floor cited, unsuperseded: {mid} +-{floor} ({cite}) -- the "
                         "bootstrap measured no half-width for it on this row")
            continue
        verdict = "tighter than" if float(got) < floor else "wider than"
        notes.append(f"charter floor cited, not applied: {mid} +-{floor} ({cite}); the "
                     f"bootstrap resolves it at +-{float(got):.3f} ({verdict} the floor) and "
                     "supersedes it (decision 6)")
    # Diagnostics on the screen row's CONTROL head: the same-run independent-resample spread, and
    # decision 6's null-candidate calibration. Neither changes an R; both say whether the recorded
    # thresholds are trustworthy, which is the one thing a resolution record cannot assert about
    # itself.
    ctrl_row = next((r for r in by_scenario[screen_row]
                     if r.accepted and r.head == by_row[screen_row]["control_head"]), None)
    diagnostics: dict[str, Any] = {}
    nul: dict | None = None
    cross_null: dict | None = None
    null_kw: dict[str, Any] = {}
    if v2:
        dual = dual_guards(scoring_rule)
        null_kw = {"scored_blocks": systems["ratio"]["blocks"],
                   # a guard that is also a block member is its own spec under ``@guard``
                   "scored_guards": [(_ec().guard_key(g) if g in dual else g)
                                     for g in systems["ratio"]["guards"]],
                   "survival_guards": [g for g in scoring_rule["guards"]["metrics"]
                                       if _is_survival_guard(g)]}
    if ctrl_row is not None:
        specs, _ = specs_from_rule(scoring_rule, ctrl_row.hold_labels, suite=ctrl_row.suite)
        n_ctrl = len(ctrl_row.env_index)
        if v2:
            null_kw["waves"] = ctrl_row.waves
        nul = null_construction(ctrl_row.cols, specs, n=n_ctrl, B=bootstrap, N=1000,
                                rng=pair_rng(rng_seed, f"null::{ctrl_row.label}"), **null_kw)
        cross_null = null_construction(ctrl_row.cols, specs, n=n_ctrl, B=bootstrap, N=1000,
                                       rng=pair_rng(rng_seed, f"null-d00::{ctrl_row.label}"),
                                       R_metrics={k: v["R_eval"] for k, v in metrics.items()},
                                       R_blocks={k: v["R_eval"] for k, v in blocks.items()},
                                       **null_kw)
        diagnostics = {"control_head": ctrl_row.head,
                       "same_run_spread": nul["R_metrics"], "same_run_blocks": nul["R_blocks"],
                       "null_calibrated": {k: nul[k] for k in ("N", "all_tie", "blocks", "guards")},
                       "null_under_d00_R": {k: cross_null[k] for k in ("N", "all_tie", "blocks",
                                                                      "guards")}}
        if v2:
            # What entered all_tie (the ratio system) and the per-trial survival floors' medians.
            diagnostics["null_scored"] = nul["scored"]
            diagnostics["null_calibrated"]["binomial_floor"] = nul.get("binomial_floor")
            diagnostics["null_under_d00_R"]["binomial_floor"] = cross_null.get("binomial_floor")
        shown = systems["ratio"]["blocks"] if v2 else ("F", "T")
        notes.append(f"null-candidate check on {ctrl_row.head} @{screen_row} (N=1000, two independent "
                     f"{'env-clustered ' if v2 else ''}episode resamples per trial): all-TIE "
                     f"{nul['all_tie']:.3f} with R re-derived on "
                     f"the same resample distribution (blocks "
                     + ", ".join(f"{k} {v:.3f}" for k, v in nul['blocks'].items() if k in shown)
                     + f"), {cross_null['all_tie']:.3f} under "
                     + ("this record's" if v2 else "D00's") + " cross-head R")
        floor_ratio = 1.0 / math.sqrt(2.0)
        for mid, m in sorted(metrics.items()):
            R, spread = m.get("R_eval"), nul["R_metrics"].get(mid)
            if R is None or spread is None or spread == 0.0:
                continue
            if float(R) <= 1.05 * floor_ratio * float(spread):
                kind = "guard" if m["block"] == "guard" else f"{m['block']} member"
                measured = ""
                if mid in cross_null["guards"] and mid in nul["guards"]:
                    measured = (f" (measured null TIE {cross_null['guards'][mid]:.3f} under this R "
                                f"vs {nul['guards'][mid]:.3f} under the calibrated one)")
                notes.append(
                    f"degenerate pair on {mid}: its cross-head R {float(R):.4f} is at or below the "
                    f"perfectly-correlated floor of the same-run spread "
                    f"({float(spread):.4f}/sqrt2 = {floor_ratio * float(spread):.4f}), i.e. one side "
                    "of every pair is effectively constant on it, so the paired difference carries "
                    f"only the other side's variation. A null candidate then trips this {kind} more "
                    f"often than its level says{measured}. Decision 6's max(R_eval, R_seed) from "
                    "D00b is the mitigation -- watch it at window 2.")
    dev = [f"{r.label} {r.mask_deviation:.1e}" for r in rows if r.mask_deviation is not None]
    if dev:
        notes.append("facing re-score window verified against the recorded per-episode keys "
                     "(max abs deviation per row): " + "; ".join(dev))
    if not floor_metrics:
        notes.append("process_floor empty: no same-head pair among the inputs (the 2b slow re-eval "
                     "is a 4-min GPU row; list it with process_floor_pair: true, and floor_of: "
                     "<head> when its label differs from the head it re-evaluates)")

    extra: dict[str, Any] = {}
    line_kw: dict[str, Any] = {}
    if v2:
        extra = _v2_payload(rows=rows, by_row=by_row, metrics=metrics, blocks=blocks,
                            systems=systems, nul=nul, cross_null=cross_null, ctrl_row=ctrl_row,
                            screen_row=screen_row, d00=d00, notes=notes, eval_sha=eval_sha,
                            decisive_at=decisive_at, supersedes=supersedes)
        line_kw = {"label": "D00s", "draw": "env-clustered paired bootstrap",
                   "tail": (f" | null all-TIE {nul['all_tie']:.3f} (nominal "
                            f"{extra['null_nominal']:.3f}) | acceptance "
                            + ("ok" if extra["acceptance"]["ok"] else
                               f"FAILED ({len(extra['acceptance']['failures'])})")
                            if nul is not None else " | no null (no control row)")}
    inputs = [{"run_dir": str(r.run_dir), "row": r.row, "head": r.head,
               "provenance_ok": r.provenance_ok, "accepted": r.accepted, "reason": r.reason,
               "env_vars_defaulted": list(r.env_vars_defaulted),
               "waived": r.waived, "n_episodes": r.n_episodes,
               "facing_closed_source": r.facing_closed_source,
               "process_floor_pair": r.process_floor_pair,
               "mask_deviation": _jf(r.mask_deviation)} for r in rows]
    if v2:
        for rec, r in zip(inputs, rows):
            rec.update({"role": r.role, "suite": r.suite, "stamp": r.stamp,
                        "checkpoint": r.entry_checkpoint or r.checkpoint})
    payload = {
        "schema": SCHEMA,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "B": int(bootstrap),
        "rng_seed": int(rng_seed),
        # The candidates.yaml this record was measured against. `decision_rule` below is frozen
        # verbatim into D00, and `resolution_inputs` decided every pair, so a later edit of that file
        # makes this record stale: the driver (which pins the same hash in state.ladder) can compare.
        "candidates_sha256": candidates_sha256,
        "decision_rule": candidates.get("decision_rule") or {},
        "inputs": inputs,
        "pairs": pair_records,
        "metrics": metrics,
        "blocks": blocks,
        "watch_metrics": watch,
        "aliases": alias_map(specs_from_rule(scoring_rule, by_scenario[screen_row][0].hold_labels,
                                             suite=by_scenario[screen_row][0].suite)[0]),
        "by_row": by_row,
        "screen_row": screen_row,
        "process_floor": {k: v["R"] for k, v in sorted(floor_metrics.items())},
        "process_floor_detail": {k: v for k, v in sorted(floor_metrics.items())},
        "diagnostics": diagnostics,
        "charter_floors": floors,
        "charter_cite": cite,
        "seed_terms": {},
        "levels": {"block": BLOCK_LEVEL, "guard": GUARD_LEVEL, "guard_one_sided": True},
        "resolution_line": _resolution_line(blocks, metrics, bootstrap, **line_kw),
        "notes": notes,
    }
    if not v2:
        for key, val in (("eval_sha", eval_sha), ("supersedes", supersedes)):
            if val is not None:
                payload[key] = val
    payload.update(extra)
    return payload


def _v2_payload(*, rows: Sequence[Row], by_row: dict, metrics: dict, blocks: dict, systems: dict,
                nul: dict | None, cross_null: dict | None, ctrl_row: Row | None, screen_row: str,
                d00: dict | None, notes: list[str], eval_sha: str | None,
                decisive_at: Sequence[str], supersedes: str | None) -> dict:
    """The keys a rule-version-2 (``D00s``) record adds; mutates ``by_row`` / ``metrics`` / ``notes``.

    ``by_category_n`` (per category: episode count, distinct envs, per-hold n, from the control's
    suite row); each survival guard's ``binomial_floor`` (named per cross-head pair: the floor a
    comparison of those two heads would take); ``null_survival_tie``; D00's light/ring tables and
    frozen rule; ``acceptance``.
    """
    ec = _ec()
    nominal = null_nominal(systems["ratio"])
    by_category_n: dict[str, dict] = {}
    if ctrl_row is not None and ctrl_row.suite:
        for cat in ec.CATEGORIES:
            holds = [h for h, c in zip(ctrl_row.hold_index, ctrl_row.category_index) if c == cat]
            if not holds:
                continue
            by_category_n[cat] = {"n": len(holds), "n_env": int(ctrl_row.episodes_per_wave),
                                  "holds": {h: holds.count(h) for h in dict.fromkeys(holds)}}
    floors_named: list[str] = []
    for mid, m in sorted(metrics.items()):
        if not _is_survival_guard(mid):
            continue
        per_pair: dict[str, dict] = {}
        for name, pt in (m.get("points") or {}).items():
            a, b = pt.get("a"), pt.get("b")
            if a is None or b is None:
                continue
            p_bar = (float(a) + float(b)) / 2.0
            n_c = int(m.get("n_valid") or 0)
            per_pair[name] = {"p_bar": p_bar, "n_c": n_c,
                              "R_binom": binomial_guard_floor(p_bar, n_c)}
        worst = max((v["R_binom"] for v in per_pair.values() if v["R_binom"] is not None),
                    default=None)
        m["binomial_floor"] = {"R_binom": worst, "per_pair": per_pair}
        by_row[screen_row]["metrics"][mid]["binomial_floor"] = m["binomial_floor"]
        if worst is not None:
            floors_named.append(f"{mid} {worst:.4f} (R_eval {m.get('R_eval')})")
    if floors_named:
        notes.append("binomial floors (survival guards, z_0.99 sqrt(2 p(1-p)/n_c), p clipped to "
                      "[0.05, 0.95], worst cross-head pair): " + "; ".join(floors_named))
    null_survival_tie = {}
    if nul is not None and cross_null is not None:
        for g in sorted(set(nul["guards"]) | set(cross_null["guards"])):
            if _is_survival_guard(g):
                null_survival_tie[g] = {
                    "calibrated": nul["guards"].get(g),
                    "under_cross_head_R": cross_null["guards"].get(g),
                    "binomial_floor_median": (nul.get("binomial_floor") or {}).get(g)}
    by_row_rule = None
    if d00 is not None:
        d00_rows = dict(d00.get("by_row") or {})
        missing = [r for r in D00_COPIED_ROWS if r not in d00_rows]
        if missing or not d00.get("decision_rule"):
            raise ResolutionError(
                f"d00 carries no by_row {missing} or no decision_rule: D00s copies D00's light/ring "
                "tables and its frozen rule verbatim (plan D7), so it needs the parsed D00 record")
        for r in D00_COPIED_ROWS:
            if r in by_row:
                notes.append(f"by_row[{r}] measured here is replaced by D00's copy (light/ring keep "
                             "the frozen v1 per-row rule and its R, plan D7)")
            table = copy.deepcopy(d00_rows[r])
            table["by_row_source"] = "D00"
            by_row[r] = table
        by_row_rule = copy.deepcopy(d00["decision_rule"])
        notes.append(f"by_row {', '.join(D00_COPIED_ROWS)} copied from D00 with its frozen "
                     "decision_rule as by_row_rule (milestones score those rows on the v1 rule)")
    else:
        notes.append("no D00 record given: the light/ring by_row tables are NOT copied -- a "
                     "milestone cannot score those rows against this record")
    acceptance = acceptance_d00s(blocks, metrics, systems, nul, nominal)
    return {"rule_version": 2,
            "eval_sha": eval_sha,
            "supersedes": supersedes,
            "decisive_at": list(decisive_at),
            "scored_systems": systems,
            "by_category_n": by_category_n,
            "null_nominal": nominal,
            "null_survival_tie": null_survival_tie,
            "by_row_rule": by_row_rule,
            "by_row_source": {r: "D00" for r in D00_COPIED_ROWS} if d00 is not None else {},
            "acceptance": acceptance}


def _rederive_curriculum(run_dirs: Sequence[str | Path], watch: dict[str, dict], *,
                         ring_proxy: bool, bootstrap: int, rng_seed: int,
                         verify: bool, rule: dict | None = None
                         ) -> tuple[dict[str, dict], list[str], dict | None]:
    """``--curriculum-runs``: measure ``lat_r`` / ``move_turn_r`` instead of borrowing.

    Two or more curriculum-row eval dirs (different heads) form the cross-head pair the legacy rows
    cannot: the right-handed holds exist only on ``curriculum__omni``. With one run there is no pair
    and the borrowed entries stand, with a note.
    """
    notes: list[str] = []
    rows = [load_row({"run_dir": str(d)}, ring_proxy=ring_proxy, rule=rule, verify=verify)
            for d in run_dirs]
    wrong = [r.label for r in rows if r.row != CURRICULUM_ROW]
    if wrong:
        raise ResolutionError(f"--curriculum-runs expects {CURRICULUM_ROW} rows; got {wrong}")
    specs, _ = specs_from_rule(rule or {}, rows[0].hold_labels)
    specs = list(specs) + [s for s in watch_specs(rows[0].hold_labels)
                           if s.mid not in {x.mid for x in specs}]
    cross, floor, n2 = build_pairs(rows, control=None, B=bootstrap, rng_seed=rng_seed, specs=specs)
    notes.extend(n2)
    if not cross:
        notes.append("--curriculum-runs: fewer than two heads on the curriculum row -- lat_r / "
                     "move_turn_r keep their borrowed R")
        return watch, notes, None
    metrics, blocks, n3 = _merge_pairs(cross)
    notes.extend(f"{CURRICULUM_ROW}: {m}" for m in n3)
    for block in blocks.values():
        block["mde"] = _mde(block, metrics)
    out = dict(watch)
    for mid, m in sorted(metrics.items()):
        hold = m.get("hold")
        if hold in BORROW and m.get("R_eval") is not None:
            entry = dict(m)
            entry["block"] = "watch"
            entry["blocks_all"] = []
            entry["rederived_from"] = list(m.get("pairs_used") or [])
            entry.pop("borrowed_from", None)
            out[mid] = entry
            notes.append(f"--curriculum-runs re-derived {mid}: R={float(m['R_eval']):.4f} "
                         f"(was borrowed from {BORROW[hold]})")
    crow = {"metrics": metrics, "blocks": blocks, "control_head": cross[0].a.head,
            "heads": sorted({p.b.head for p in cross} | {cross[0].a.head})}
    return out, notes, crow


def write_resolution(payload: dict, out_path: Path) -> Path:
    """Write ``payload`` to ``out_path`` atomically (temp + ``os.replace``), sorted and stable."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_path)
    return out_path


def load_candidates(path: Path) -> dict:
    import yaml  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise ResolutionError(
            f"{path} does not exist. resolution.py reads resolution_inputs and decision_rule from "
            "candidates.yaml; pass --candidates or create the campaign's candidates.yaml first.")
    return yaml.safe_load(path.read_text()) or {}


def summary_text(payload: dict) -> str:
    """A human-readable digest of the record -- what the RESOLUTION notify line quotes."""
    lines = [payload["resolution_line"], ""]
    if payload.get("rule_version") == 2:
        acc = payload.get("acceptance") or {}
        lines.append(f"rule_version 2 on {payload['screen_row']}: eval_sha {payload.get('eval_sha')}, "
                     f"supersedes {payload.get('supersedes')}, decisive_at "
                     f"{', '.join(payload.get('decisive_at') or [])}")
        lines.append(f"null nominal {payload.get('null_nominal', float('nan')):.3f}; acceptance "
                     + ("ok" if acc.get("ok") else "FAILED: " + "; ".join(acc.get("failures") or [])))
        for g, t in sorted((payload.get("null_survival_tie") or {}).items()):
            lines.append(f"  null TIE {g:28s} calibrated {t.get('calibrated')} under cross-head R "
                         f"{t.get('under_cross_head_R')}")
        lines.append("")
    lines.append(f"inputs ({len(payload['inputs'])}):")
    for i in payload["inputs"]:
        flag = "USED " if i["accepted"] else "info "
        lines.append(f"  {flag}{i['head']:28s} {i['row']:16s} n={i['n_episodes']:3d} "
                     f"prov={'ok' if i['provenance_ok'] else 'MISSING'} "
                     f"facing={i['facing_closed_source']} defaulted={','.join(i['env_vars_defaulted']) or '-'}")
        if not i["accepted"]:
            lines.append(f"        {i['reason']}")
    lines.append("")
    lines.append(f"pairs: {', '.join(p['name'] + ' [' + p['kind'] + ']' for p in payload['pairs']) or 'none'}")
    lines.append("")
    lines.append(f"metrics on {payload['screen_row']} (R_eval, level, block):")
    for mid, m in sorted(payload["metrics"].items()):
        R = m["R_eval"]
        lines.append(f"  {mid:36s} {'   n/a' if R is None else f'{R:8.4f}'} @{m['level']:.2f}"
                     f"{' 1-sided' if m['one_sided'] else '        '} {m['block']:6s} "
                     f"n={m['n_valid']}{' THIN' if m['below_min_n'] else ''} "
                     f"delta={_fmt(list(m.get('points', {}).values()))}")
    lines.append("")
    for block, b in sorted(payload["blocks"].items()):
        R = b["R_eval"]
        lines.append(f"  block {block:6s} R_eval={'n/a' if R is None else f'{R:.4f}'} "
                     f"@{b['level']:.2f} k={len(b['members'])} draws={b.get('n_draws', '-')}")
        for mid, v in sorted((b.get("mde") or {}).items()):
            lines.append(f"      MDE {mid:34s} {v:.4f}")
    lines.append("")
    lines.append("watch (never scored; borrowed where the hold is not on the screen row):")
    for mid, m in sorted(payload["watch_metrics"].items()):
        if m.get("borrowed_from"):
            src = "borrowed from " + m["borrowed_from"]
        elif m.get("rederived_from"):
            src = "re-derived from " + ", ".join(m["rederived_from"])
        else:
            src = "measured on " + ", ".join(m.get("pairs_used") or ["-"])
        R = m.get("R_eval")
        lines.append(f"  {mid:36s} {'     n/a' if R is None else f'{float(R):8.4f}'}  <- {src}"
                     + ("  THIN" if m.get("below_min_n") else ""))
    lines.append("")
    diag = payload.get("diagnostics") or {}
    if diag:
        lines.append(f"null candidate on {diag['control_head']} (N={diag['null_calibrated']['N']}):")
        lines.append(f"  calibrated R   all-TIE {diag['null_calibrated']['all_tie']:.3f}  blocks "
                     + " ".join(f"{k} {v:.3f}" for k, v in diag['null_calibrated']['blocks'].items())
                     + "  guards "
                     + " ".join(f"{k} {v:.3f}" for k, v in diag['null_calibrated']['guards'].items()))
        lines.append(f"  D00 cross-head R  all-TIE {diag['null_under_d00_R']['all_tie']:.3f}  blocks "
                     + " ".join(f"{k} {v:.3f}" for k, v in diag['null_under_d00_R']['blocks'].items())
                     + "  guards "
                     + " ".join(f"{k} {v:.3f}" for k, v in diag['null_under_d00_R']['guards'].items()))
        lines.append("")
    lines.append(f"process_floor: {payload['process_floor'] or '{} (no same-head pair)'}")
    lines.append(f"seed_terms: {payload['seed_terms']} (D00 is eval-only; D00b/D00c are the driver's)")
    lines.append("")
    lines.append("notes:")
    lines.extend("  - " + n for n in payload["notes"])
    return "\n".join(lines)


def _fmt(points: list) -> str:
    if not points:
        return "-"
    return " ".join("n/a" if p.get("delta") is None else f"{p['delta']:+.3f}" for p in points)


def parse_reeval_stamps(text: str | None, *, eval_sha: str | None = None) -> dict[str, str]:
    """``{head: stamp}`` from ``--reeval-stamps``: a JSON object ``{head: stamp}``, or a JSON list of
    ``lad.reevals`` records ``[{head, eval_sha, stamp}]`` (filtered to ``eval_sha`` when given; a
    later record wins -- the list is append-only), inline or in a file."""
    if text is None:
        return {}
    raw = text
    try:
        if Path(text).is_file():
            raw = Path(text).read_text()
    except OSError:
        pass
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ResolutionError(f"--reeval-stamps is not JSON (or a JSON file): {exc}") from exc
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        out: dict[str, str] = {}
        for rec in data:
            if eval_sha is not None and rec.get("eval_sha") != eval_sha:
                continue
            out[str(rec["head"])] = str(rec["stamp"])
        return out
    raise ResolutionError("--reeval-stamps must be a JSON object {head: stamp} or a list of "
                          "{head, eval_sha, stamp} records")


def apply_reeval_stamps(entries: Sequence[dict], stamps: dict[str, str]) -> list[dict]:
    """``resolution_inputs`` with each checkpoint-shaped ``role: pair`` entry's recorded ``stamp``
    set from ``stamps`` (by ``head``) -- what the driver does from ``lad.reevals`` in process."""
    out = []
    for e in entries or []:
        e = dict(e)
        if (str(e.get("role") or "pair") == "pair" and e.get("checkpoint")
                and str(e.get("head")) in stamps):
            e["stamp"] = stamps[str(e["head"])]
        out.append(e)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D00: the measured eval resolution (plan decision 6)")
    ap.add_argument("--campaign-dir", required=True, type=Path,
                    help="the ladder campaign dir; resolution.json is written here")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="candidates.yaml (default: <campaign-dir>/candidates.yaml)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <campaign-dir>/resolution.json)")
    ap.add_argument("--bootstrap", type=int, default=None,
                    help=f"paired bootstrap draws (default: candidates.yaml's "
                         f"decision_rule.bootstrap.B, else {BOOTSTRAP_B})")
    ap.add_argument("--rng-seed", type=int, default=None,
                    help=f"fixed RNG seed, recorded in the output (default: candidates.yaml's "
                         f"decision_rule.bootstrap.rng_seed, else {RNG_SEED})")
    ap.add_argument("--control-head", default=None,
                    help="the head every cross-head pair is against (default: the first accepted "
                         "input listed for the row)")
    ap.add_argument("--ring-proxy", action="store_true",
                    help="score facing_closed on ring rows with the plan's heading_error_max_abs "
                         "proxy instead of re-scoring the raws. The 2026-09-18 review measured that "
                         "proxy to flip sign against the per-episode metric; the default re-scores "
                         "every row that has raws and labels the source.")
    ap.add_argument("--curriculum-runs", nargs="+", default=(), type=Path,
                    help="two or more curriculum__omni eval dirs: re-derive lat_r / move_turn_r's R "
                         "instead of borrowing lat_l / move_turn's")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the window / reducer replication checks (diagnostics only)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="print the digest")
    # ---- rule version 2 (D00s): the manual mirror of the driver's in-process resolve() call
    ap.add_argument("--eval-sha", default=None,
                    help="rule v2: the campaign's lad.eval_sha, recorded in D00s; the default output "
                         "is then <campaign-dir>/resolution.<eval_sha[:12]>.json")
    ap.add_argument("--reeval-stamps", default=None,
                    help="rule v2: the role:pair inputs' recorded eval stamps -- JSON {head: stamp} "
                         "or the lad.reevals list [{head, eval_sha, stamp}] (filtered to --eval-sha), "
                         "inline or a file")
    ap.add_argument("--supersedes", default=None, help="rule v2: the record D00s supersedes (D00)")
    ap.add_argument("--d00", type=Path, default=None,
                    help="rule v2: D00's resolution.json -- its light/ring by_row tables and frozen "
                         "decision_rule are copied into D00s")
    ap.add_argument("--decisive-at", default=None,
                    help="rule v2: CSV of decisive(B0) (default: all seven categories -- B0 trains "
                         "every one)")
    args = ap.parse_args(argv)

    cand_path = args.candidates or (args.campaign_dir / "candidates.yaml")
    try:
        cands = load_candidates(cand_path)
        sha = hashlib.sha256(Path(cand_path).read_bytes()).hexdigest()
        v2 = rule_version(cands.get("decision_rule")) == 2
        kw: dict[str, Any] = {"eval_sha": args.eval_sha, "supersedes": args.supersedes}
        if v2:
            if args.out is None and not args.eval_sha:
                raise ResolutionError("rule_version 2: pass --eval-sha (names the output "
                                      "resolution.<eval_sha[:12]>.json) or --out")
            stamps = parse_reeval_stamps(args.reeval_stamps, eval_sha=args.eval_sha)
            cands = dict(cands, resolution_inputs=apply_reeval_stamps(
                cands.get("resolution_inputs") or [], stamps))
            kw["decisive_at"] = ([c.strip() for c in args.decisive_at.split(",") if c.strip()]
                                 if args.decisive_at else list(_ec().CATEGORIES))
            kw["d00"] = json.loads(args.d00.read_text()) if args.d00 else None
        elif args.reeval_stamps or args.d00 or args.decisive_at:
            raise ResolutionError("--reeval-stamps / --d00 / --decisive-at are rule_version 2 "
                                  "flags; this decision_rule is rule_version 1")
        out_path = args.out or (args.campaign_dir / (f"resolution.{args.eval_sha[:12]}.json"
                                                     if v2 else "resolution.json"))
        payload = resolve(cands, bootstrap=args.bootstrap,
                          rng_seed=args.rng_seed, ring_proxy=args.ring_proxy,
                          control_head=args.control_head,
                          curriculum_runs=args.curriculum_runs, verify=not args.no_verify,
                          candidates_sha256=sha, **kw)
    except ResolutionError as exc:
        print(f"resolution refused: {exc}", file=sys.stderr)
        return 1
    acceptance = payload.get("acceptance")
    if acceptance is not None and not acceptance.get("ok"):
        # The driver's rule: a failing D00s writes NOTHING but its diagnostic, never the record.
        rejected = out_path.with_name(f"{out_path.stem}.D00s.rejected.json")
        write_resolution(payload, rejected)
        print(f"resolution acceptance FAILED -- wrote only the diagnostic {rejected}:",
              file=sys.stderr)
        for f in acceptance.get("failures") or []:
            print(f"  - {f}", file=sys.stderr)
        print(payload["resolution_line"])
        return 2
    write_resolution(payload, out_path)
    if args.do_print:
        print(summary_text(payload))
    print(f"wrote {out_path}")
    print(payload["resolution_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
