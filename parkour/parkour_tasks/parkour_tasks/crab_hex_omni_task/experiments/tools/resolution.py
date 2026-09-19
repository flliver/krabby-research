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

Usage::

    resolution.py --campaign-dir <ladder dir> [--candidates <candidates.yaml>] [--out resolution.json]

``--candidates`` defaults to ``<campaign-dir>/candidates.yaml`` and supplies ``resolution_inputs``
(the run dirs) and ``decision_rule`` (frozen verbatim into the record). The driver calls
:func:`main` by import-by-path when ``resolution.json`` is absent.
"""
from __future__ import annotations

import argparse
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
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PKG = HERE.parents[1]
FWD_PKG = PKG.parent / "crab_hex_forward_task"
GAIT_SCRIPTS = FWD_PKG / "scripts"          # the dir `from gait_eval import metrics` resolves against
RUN_PHASES_PY = HERE / "run_phases.py"


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
#: The row every screen is scored on (decision 1 / the `screen {eval: [slow]}` block). Its metrics
#: and blocks are the flat ``metrics``/``blocks`` of the record; other rows land in ``by_row``.
SCREEN_ROW = "slow__omni"
F_HOLDS = ("turn_l90", "turn_r90", "move_turn")
T_HOLDS = ("fwd", "lat_l", "diag", "back")
#: Right-handed holds: WATCH only (decision 5), and present only on the curriculum row. They borrow
#: their left-handed twin's R until ``--curriculum-runs`` re-derives them (decision 6).
BORROW = {"lat_r": "lat_l", "move_turn_r": "move_turn"}
CURRICULUM_ROW = "curriculum__omni"
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


def pooled_guard_R(r_eval: float | None, r_seed: float | None) -> float | None:
    """``R_pooled,i = sqrt((R_eval,i / sqrt(2))**2 + R_seed,i**2)`` (decision 6)."""
    a = 0.0 if r_eval is None or not math.isfinite(float(r_eval)) else float(r_eval) / math.sqrt(2.0)
    b = 0.0 if r_seed is None or not math.isfinite(float(r_seed)) else float(r_seed)
    if a == 0.0 and b == 0.0:
        return None
    return float(math.sqrt(a * a + b * b))


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


@lru_cache(maxsize=None)
def _manifest_scenario(sid: str) -> dict:
    """``{hold_s: {...}, fall_exclusion_s: float, hold_labels: [...]}`` for one manifest row.

    Parsed from the manifest rather than pinned, for the same reason ``rp._manifest_hold_s`` is: a
    manifest edit must not silently desynchronise the window this module cuts from the one the
    harness cut.
    """
    import yaml  # noqa: PLC0415 -- the harness's own dependency, imported where it is used

    doc = yaml.safe_load(rp.OMNI_MANIFEST.read_text())
    file_defaults = dict(doc.get("defaults") or {})
    for sc in doc.get("scenarios", []):
        if sc.get("id") == sid:
            merged = dict(file_defaults)
            merged.update(sc.get("defaults") or {})
            return {"hold_labels": [str(h["label"]) for h in sc["schedule"]],
                    "hold_s": {str(h["label"]): float(h["hold_s"]) for h in sc["schedule"]},
                    "fall_exclusion_s": float(merged.get("fall_exclusion_s", 0.5)),
                    "prologue_s": float(merged.get("prologue_s", rp.PROLOGUE_S))}
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
    block: str                      # F | T | guard | F_abs | T_abs | watch
    blocks_all: tuple[str, ...]     # every block it is a member of (facing_closed is in F and F_abs)
    hold: str | None
    level: float
    one_sided: bool
    needs: tuple[str, ...]          # column keys, in the reducer's order
    reducer: str                    # median | mean | min_of_medians | ratio_of_means


def _spec(mid, block, blocks_all, hold, needs, reducer="median", *, level=BLOCK_LEVEL, one_sided=False):
    return MetricSpec(mid=mid, block=block, blocks_all=tuple(blocks_all), hold=hold, level=level,
                      one_sided=one_sided, needs=tuple(needs), reducer=reducer)


def metric_specs(hold_labels: Sequence[str]) -> list[MetricSpec]:
    """Decision 5's scored surface, restricted to the holds this row actually has.

    F = the closed fraction and the signed ``yaw_progress_rate`` on the two in-place turns and
    ``move_turn``; T = the speed ratio on the four direction holds; four guards (completion, the
    direction-hold tripod floor, and the two CONDITIONAL turn survivals); ``F_abs``/``T_abs`` the
    absolute-units twins a cap candidate is scored on. Right-handed holds and everything else in
    decision 5's WATCH list are not here -- ``watch_specs`` carries the two the curriculum row adds.
    """
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


def spec_for(mid: str, hold_labels: Sequence[str], *, block: str, blocks_all: Sequence[str],
             level: float, one_sided: bool) -> MetricSpec | None:
    """The :class:`MetricSpec` for one ``summarize()`` id, or ``None`` when it cannot be resolved.

    ``None`` for an id this row has no holds for, and for the handful of ids with no per-episode
    form at all (:data:`_UNRESOLVABLE`). Everything else resolves through :data:`_STEM_COLUMNS` or
    the three composite rules below, so ``candidates.yaml``'s ``decision_rule`` decides MEMBERSHIP
    and this module decides only HOW each member is computed.
    """
    have = set(str(h) for h in hold_labels)
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


def why_unresolved(mid: str, hold_labels: Sequence[str]) -> str:
    """Why :func:`spec_for` said no -- so a note says which of the three reasons it is."""
    have = set(str(h) for h in hold_labels)
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


def specs_from_rule(rule: dict, hold_labels: Sequence[str]) -> tuple[list[MetricSpec], list[str]]:
    """``(specs, unresolvable ids)`` from ``candidates.yaml``'s ``decision_rule`` block.

    The yaml is authoritative for what is scored and at which level; a member it names that this
    module cannot compute is returned in the second list and NOTED, never silently dropped -- a
    quietly absent block member shrinks the block mean, which is exactly the kind of quiet failure
    that costs a branch. Falls back to :func:`metric_specs` + :func:`watch_specs` when the yaml
    carries no machine-readable rule.
    """
    blocks = (rule or {}).get("blocks") or {}
    guards = (rule or {}).get("guards") or {}
    watch = list((rule or {}).get("watch") or [])
    if not blocks and not guards:
        return metric_specs(hold_labels) + watch_specs(hold_labels), []
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
                        level=levels[mid], one_sided=False)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    for mid in guard_ids:
        if mid in seen:
            continue
        spec = spec_for(mid, hold_labels, block="guard", blocks_all=(), level=guard_level,
                        one_sided=True)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    for mid in watch:
        if mid in seen:
            continue
        spec = spec_for(mid, hold_labels, block="watch", blocks_all=(), level=BLOCK_LEVEL,
                        one_sided=False)
        (specs.append(spec) if spec else unresolved.append(mid))
        seen.add(mid)
    return specs, unresolved


def watch_specs(hold_labels: Sequence[str]) -> list[MetricSpec]:
    """The right-handed holds' quantities -- WATCH only, and only where the row has them.

    ``lat_r`` is a travel hold (ratio + achieved speed); ``move_turn_r`` is a move+turn hold whose
    facing "never closes" by construction (a 90 deg change inside 8 s at ~0.15-0.2 rad/s), so its
    facing quantities are watched too. None of these is ever a block member (decision 5).
    """
    have = set(str(h) for h in hold_labels)
    out: list[MetricSpec] = []
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

    @property
    def label(self) -> str:
        return f"{self.head} @{self.row}"


def required_carry_keys() -> tuple[str, ...]:
    """The ``EVAL_CARRY`` keys that have no ``EVAL_CARRY_DEFAULTS`` entry -- the five 2026-09-17 keys.

    Read from ``run_phases`` rather than listed, so growing ``EVAL_CARRY`` (with or without a
    default) cannot leave this rule behind.
    """
    return tuple(k for k in rp.EVAL_CARRY if k not in rp.EVAL_CARRY_DEFAULTS)


def _episodes(run_dir: Path) -> list[dict]:
    files = sorted((run_dir / "metrics").glob("episode_*.json"))
    eps = [json.loads(f.read_text()) for f in files]
    return sorted(eps, key=lambda e: int(e["env_index"]))


def _survival_end_steps(sid: str, dt: float) -> dict[str, int]:
    """``{hold: the step index an episode must reach to have survived that hold}``.

    The same walk ``rp.survival_by_hold`` makes (prologue then each hold's duration, rounded to
    steps), hoisted so the indicator can be computed per episode and resampled.
    """
    sc = _manifest_scenario(sid)
    out: dict[str, int] = {}
    t_end = float(sc["prologue_s"])
    for hold in sc["hold_labels"]:
        t_end += float(sc["hold_s"][hold])
        out[hold] = int(round(t_end / float(dt)))
    return out


def _columns(episodes: Sequence[dict], hold_labels: Sequence[str], closed: dict[str, np.ndarray],
             end_steps: dict[str, int]) -> dict[str, np.ndarray]:
    """Per-episode value arrays, NaN where the harness's own aggregate would have excluded the episode.

    The exclusions are copied from ``report.aggregate``: a tripod score only when it is present and
    not ``low_confidence``; a tracking ratio only when the hold has a tracking block whose
    ``vx.actual_mean`` is not None and whose ``ratio`` is not None; a facing quantity only when the
    key is not None. ``completed`` and the survival indicators are defined for every episode.
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
        for hold in hold_labels:
            end = end_steps.get(hold)
            if end is not None:
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
    return cols


def _summary_with_survival(run_dir: Path, sid: str) -> dict:
    """``rp.summarize`` of the row's aggregate, with ``survival_by_hold`` injected.

    ``_run_eval`` injects it IN MEMORY, so an aggregate read back off disk has no such key and both
    conditional turn survivals would come back ``None`` -- a silent guard loss (step 3's hand-off).
    """
    sm = json.loads((run_dir / "scenario_metrics.json").read_text())
    agg = dict(sm["aggregate"])
    meta = sm.get("run_meta") or {}
    eps = sm.get("episodes") or []
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
    """
    out: dict[str, float] = {}
    ident = np.arange(n)[None, :]
    for spec in specs:
        if spec.mid.startswith("facing_closed_by_hold."):
            continue
        got = float(_reduce(spec.reducer, [cols[c][ident] for c in spec.needs])[0])
        want = _summary_value(summary, spec.mid)
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
    """
    run_dir = Path(entry["run_dir"])
    if not run_dir.is_absolute():
        run_dir = (REPO / run_dir).resolve()
    # A refusal, not a traceback: the ladder driver reports "resolution input <run_dir> is not on
    # this machine (resolution.py refuses it)" and needs this to come back as a ResolutionError.
    for needed in (run_dir / "run_meta.json", run_dir / "scenario_metrics.json", run_dir / "metrics"):
        if not needed.exists():
            raise ResolutionError(f"resolution input {run_dir} is unusable: {needed.name} is missing "
                                  "(the run dir is not on this machine, or the eval never finished)")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    sid = str(entry.get("row") or meta["scenario_id"])
    dt = float(meta.get("dt", 0.02))
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

    end_steps = _survival_end_steps(sid, dt)
    cols = _columns(episodes, hold_labels, closed, end_steps)
    summary = _summary_with_survival(run_dir, sid)
    specs = specs_from_rule(rule or {}, hold_labels)[0] + watch_specs(hold_labels)
    checks = _reducer_check(specs, cols, summary, n) if (verify and n) else {}
    bad = {k: v for k, v in checks.items() if not (math.isfinite(v) and v <= 1e-9)}
    if bad:
        raise ResolutionError(
            f"{run_dir}: the per-episode reconstruction does not reproduce summarize()'s own "
            f"numbers: {bad}. The bootstrap would resample a different estimator than the decision "
            "rule reads.")
    return Row(run_dir=run_dir, row=sid, head=head, dt=dt, hold_labels=hold_labels, env_vars=env_vars,
               provenance_ok=provenance_ok, waived=waived, accepted=accepted, reason=reason,
               env_vars_defaulted=defaulted,
               facing_closed_source=("proxy" if source == "proxy" else source),
               env_index=tuple(int(e["env_index"]) for e in episodes), cols=cols, summary=summary,
               n_episodes=n,
               process_floor_pair=bool(entry.get("process_floor_pair")) or role == "process_floor",
               mask_deviation=mask_dev, reducer_check=checks, checkpoint=str(ckpt),
               floor_of=(str(entry["floor_of"]) if entry.get("floor_of") else None))


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
    """
    if a.facing_closed_source != b.facing_closed_source:
        raise ResolutionError(
            f"{_pair_name(a, b)}: facing_closed sources differ ({a.facing_closed_source!r} vs "
            f"{b.facing_closed_source!r}). A proxy row is never pooled against a re-scored one "
            "(plan decision 6 / step-3 hand-off) -- re-score both sides or proxy both sides.")
    shared = sorted(set(a.env_index) & set(b.env_index))
    if len(shared) < MIN_BOOTSTRAP_N:
        raise ResolutionError(f"{_pair_name(a, b)}: only {len(shared)} shared env_index values")
    pos_a = {v: i for i, v in enumerate(a.env_index)}
    pos_b = {v: i for i, v in enumerate(b.env_index)}
    take_a = np.array([pos_a[v] for v in shared], dtype=np.int64)
    take_b = np.array([pos_b[v] for v in shared], dtype=np.int64)
    n = len(shared)
    draw = pair_rng(rng_seed, _pair_name(a, b)).integers(0, n, size=(B, n))
    ident = np.arange(n)[None, :]

    if specs is None:
        holds = tuple(h for h in a.hold_labels if h in set(b.hold_labels))
        specs = metric_specs(holds) + watch_specs(holds)

    out: dict[str, dict] = {}
    deltas: dict[str, np.ndarray] = {}
    for spec in specs:
        if any(c not in a.cols or c not in b.cols for c in spec.needs):
            continue
        cols_a = [a.cols[c][take_a] for c in spec.needs]
        cols_b = [b.cols[c][take_b] for c in spec.needs]
        n_valid = min(int(np.sum(np.isfinite(c))) for c in cols_a + cols_b)
        va = _reduce(spec.reducer, [c[draw] for c in cols_a])
        vb = _reduce(spec.reducer, [c[draw] for c in cols_b])
        delta = va - vb
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
            "delta_point": _jf(point_a - point_b),
            "source": ("raw" if spec.mid.startswith("facing_closed_by_hold.")
                       and a.facing_closed_source == "metric" else
                       "proxy" if spec.mid.startswith("facing_closed_by_hold.") else "json"),
        }
        # Every delta is kept: which of them join a block mean is decided against the MERGED
        # ``R_eval,i`` in :func:`_standardised_blocks`, not against this pair's own.
        deltas[spec.mid] = delta

    blocks: dict[str, dict] = {}
    for block in ("F", "T", "F_abs", "T_abs"):
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
                      R_blocks: dict[str, float | None] | None = None) -> dict:
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
    """
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)

    def _deltas(rows: int) -> dict[str, np.ndarray]:
        idx_a = rng.integers(0, n, size=(rows, n))
        idx_b = rng.integers(0, n, size=(rows, n))
        out: dict[str, np.ndarray] = {}
        for spec in specs:
            if any(c not in cols for c in spec.needs):
                continue
            va = _reduce(spec.reducer, [cols[c][idx_a] for c in spec.needs])
            vb = _reduce(spec.reducer, [cols[c][idx_b] for c in spec.needs])
            out[spec.mid] = va - vb
        return out

    if R_metrics is None or R_blocks is None:
        ref = _deltas(B)
        derived_m = {s.mid: half_width(ref[s.mid], quantile_p(s.level, one_sided=s.one_sided))
                     for s in specs if s.mid in ref}
        derived_b: dict[str, float | None] = {}
        for block in ("F", "T", "F_abs", "T_abs"):
            members = [s.mid for s in specs if block in s.blocks_all
                       and s.mid in ref and derived_m.get(s.mid)]
            if not members:
                continue
            z = block_mean(np.stack([ref[m] / derived_m[m] for m in members], axis=0))
            derived_b[block] = half_width(z, quantile_p(BLOCK_LEVEL, one_sided=False))
        R_metrics = R_metrics if R_metrics is not None else derived_m
        R_blocks = R_blocks if R_blocks is not None else derived_b

    trial = _deltas(N)
    tie_blocks: dict[str, np.ndarray] = {}
    for block, R_F in sorted(R_blocks.items()):
        members = [s.mid for s in specs if block in s.blocks_all
                   and s.mid in trial and R_metrics.get(s.mid)]
        if not members or R_F is None:
            continue
        z = block_mean(np.stack([trial[m] / R_metrics[m] for m in members], axis=0))
        tie_blocks[block] = ~((z > R_F) | (z < -R_F)) | ~np.isfinite(z)
    tie_guards: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.block != "guard" or spec.mid not in trial:
            continue
        R = R_metrics.get(spec.mid)
        if R is None:
            continue
        d = trial[spec.mid]
        tie_guards[spec.mid] = ~(d < -R) | ~np.isfinite(d)
    scored = [v for k, v in sorted(tie_blocks.items()) if k in ("F", "T")]
    scored += [v for _, v in sorted(tie_guards.items())]
    all_tie = np.logical_and.reduce(scored) if scored else np.ones(N, dtype=bool)
    return {"N": int(N), "B": int(B),
            "all_tie": float(np.mean(all_tie)),
            "blocks": {k: float(np.mean(v)) for k, v in sorted(tie_blocks.items())},
            "guards": {k: float(np.mean(v)) for k, v in sorted(tie_guards.items())},
            "R_metrics": {k: _jf(v) for k, v in sorted(R_metrics.items())},
            "R_blocks": {k: _jf(v) for k, v in sorted(R_blocks.items())}}


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
    for block in ("F", "T", "F_abs", "T_abs"):
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


def _resolution_line(blocks: dict[str, dict], metrics: dict[str, dict], B: int) -> str:
    """One line naming the effective MDE per block and per guard (Verification 5)."""
    parts = []
    for block in ("F", "T", "F_abs", "T_abs"):
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
    return ("RESOLUTION D00 | paired episode bootstrap B=%d on env_index | blocks %s | "
            "guards @%.2f one-sided %s" % (int(B), "; ".join(parts) or "none", GUARD_LEVEL,
                                           ", ".join(guards) or "none"))


def resolve(candidates: dict, *, bootstrap: int | None = None, rng_seed: int | None = None,
            ring_proxy: bool = False, control_head: str | None = None,
            curriculum_runs: Sequence[str | Path] = (), verify: bool = True,
            candidates_sha256: str | None = None) -> dict:
    """Build the ``resolution.json`` payload (schema 1) from a parsed ``candidates.yaml``.

    ``bootstrap`` / ``rng_seed`` default to ``decision_rule.bootstrap.{B, rng_seed}`` when the yaml
    declares them (the yaml is the campaign's specification) and to :data:`BOOTSTRAP_B` /
    :data:`RNG_SEED` otherwise; an explicit argument always wins.

    Deterministic under ``rng_seed``, and stable under a CHANGED INPUT LIST: every pair draws from
    its own stream, keyed to ``(rng_seed, pair name)`` by :func:`pair_rng`, so re-running the tool
    after the process-floor eval lands leaves the existing pairs' numbers bit-identical instead of
    silently re-fitting every recorded ``R``.
    """
    entries = list(candidates.get("resolution_inputs") or [])
    if not entries:
        raise ResolutionError("candidates.yaml carries no resolution_inputs")
    rule = dict(candidates.get("decision_rule") or {})
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

    rows = [load_row(e, ring_proxy=ring_proxy, rule=rule, verify=verify) for e in entries]
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
        specs, unresolved = specs_from_rule(rule, by_scenario[sid][0].hold_labels)
        for mid in unresolved:
            key = (mid, why_unresolved(mid, by_scenario[sid][0].hold_labels))
            unresolved_rows.setdefault(key, []).append(sid)
        cross, floor, n2 = build_pairs(by_scenario[sid], control=control_head, B=bootstrap,
                                       rng_seed=rng_seed, specs=specs)
        notes.extend(n2)
        if cross:
            merged, blocks, n3 = _merge_pairs(cross)
            notes.extend(f"{sid}: {m}" for m in n3)
            # ``metrics`` stays exactly the schema's enumerated blocks (F/T/guard/F_abs/T_abs) so it
            # is a clean input to decide(); everything the rule only WATCHES goes to watch_metrics.
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

    if SCREEN_ROW not in by_row:
        raise ResolutionError(
            f"no cross-head pair on the screen row {SCREEN_ROW!r}: every candidate screen is scored "
            f"there, so D00 without it has no R. Rows resolved: {sorted(by_row)}")
    metrics = {k: dict(v) for k, v in by_row[SCREEN_ROW]["metrics"].items()}
    blocks = {k: dict(v) for k, v in by_row[SCREEN_ROW]["blocks"].items()}
    watch = {k: dict(v) for k, v in by_row[SCREEN_ROW]["watch_metrics"].items()}
    watch.update(_borrowed_watch(metrics, watch))

    if curriculum_runs:
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
        got = (metrics.get(mid) or {}).get("R_eval")
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
    ctrl_row = next((r for r in by_scenario[SCREEN_ROW]
                     if r.accepted and r.head == by_row[SCREEN_ROW]["control_head"]), None)
    diagnostics: dict[str, Any] = {}
    if ctrl_row is not None:
        specs, _ = specs_from_rule(rule, ctrl_row.hold_labels)
        n_ctrl = len(ctrl_row.env_index)
        nul = null_construction(ctrl_row.cols, specs, n=n_ctrl, B=bootstrap, N=1000,
                                rng=pair_rng(rng_seed, f"null::{ctrl_row.label}"))
        cross_null = null_construction(ctrl_row.cols, specs, n=n_ctrl, B=bootstrap, N=1000,
                                       rng=pair_rng(rng_seed, f"null-d00::{ctrl_row.label}"),
                                       R_metrics={k: v["R_eval"] for k, v in metrics.items()},
                                       R_blocks={k: v["R_eval"] for k, v in blocks.items()})
        diagnostics = {"control_head": ctrl_row.head,
                       "same_run_spread": nul["R_metrics"], "same_run_blocks": nul["R_blocks"],
                       "null_calibrated": {k: nul[k] for k in ("N", "all_tie", "blocks", "guards")},
                       "null_under_d00_R": {k: cross_null[k] for k in ("N", "all_tie", "blocks",
                                                                      "guards")}}
        notes.append(f"null-candidate check on {ctrl_row.head} @{SCREEN_ROW} (N=1000, two independent "
                     f"episode resamples per trial): all-TIE {nul['all_tie']:.3f} with R re-derived on "
                     f"the same resample distribution (blocks "
                     + ", ".join(f"{k} {v:.3f}" for k, v in nul['blocks'].items() if k in ('F', 'T'))
                     + f"), {cross_null['all_tie']:.3f} under D00's cross-head R")
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
        "inputs": [{"run_dir": str(r.run_dir), "row": r.row, "head": r.head,
                    "provenance_ok": r.provenance_ok, "accepted": r.accepted, "reason": r.reason,
                    "env_vars_defaulted": list(r.env_vars_defaulted),
                    "waived": r.waived, "n_episodes": r.n_episodes,
                    "facing_closed_source": r.facing_closed_source,
                    "process_floor_pair": r.process_floor_pair,
                    "mask_deviation": _jf(r.mask_deviation)} for r in rows],
        "pairs": pair_records,
        "metrics": metrics,
        "blocks": blocks,
        "watch_metrics": watch,
        "aliases": alias_map(specs_from_rule(rule, by_scenario[SCREEN_ROW][0].hold_labels)[0]),
        "by_row": by_row,
        "screen_row": SCREEN_ROW,
        "process_floor": {k: v["R"] for k, v in sorted(floor_metrics.items())},
        "process_floor_detail": {k: v for k, v in sorted(floor_metrics.items())},
        "diagnostics": diagnostics,
        "charter_floors": floors,
        "charter_cite": cite,
        "seed_terms": {},
        "levels": {"block": BLOCK_LEVEL, "guard": GUARD_LEVEL, "guard_one_sided": True},
        "resolution_line": _resolution_line(blocks, metrics, bootstrap),
        "notes": notes,
    }
    return payload


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
    args = ap.parse_args(argv)

    cand_path = args.candidates or (args.campaign_dir / "candidates.yaml")
    out_path = args.out or (args.campaign_dir / "resolution.json")
    try:
        cands = load_candidates(cand_path)
        sha = hashlib.sha256(Path(cand_path).read_bytes()).hexdigest()
        payload = resolve(cands, bootstrap=args.bootstrap,
                          rng_seed=args.rng_seed, ring_proxy=args.ring_proxy,
                          control_head=args.control_head,
                          curriculum_runs=args.curriculum_runs, verify=not args.no_verify,
                          candidates_sha256=sha)
    except ResolutionError as exc:
        print(f"resolution refused: {exc}", file=sys.stderr)
        return 1
    write_resolution(payload, out_path)
    if args.do_print:
        print(summary_text(payload))
    print(f"wrote {out_path}")
    print(payload["resolution_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
