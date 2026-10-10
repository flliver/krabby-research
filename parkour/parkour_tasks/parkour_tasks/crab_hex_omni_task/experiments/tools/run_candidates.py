#!/usr/bin/env python3
"""The candidate-ladder driver: an autonomous, speculative, resumable campaign over ranked
candidates (omni task, 2026-09-19).

One long-lived process works down a RANKED list of candidate recipes. Each candidate is screened
once (seed 3, 8000 iterations of the effective recipe's chain), decided by a pure function of the
eval summary and the measured resolution, and -- when the outcome is inside the eval resolution --
adopted or rejected SPECULATIVELY so the campaign keeps moving while the confirming seed-2 pair
runs as the next windows. The user is notified of every decision and may confirm (freeze) or
overrule any of them; an overrule discards the work done under the wrong branch (SQUASH) and the
queue resumes from the new choice. That is the whole design: a branch predictor with a visible
mispredict cost.

    launch_candidates.sh --campaign-dir <dir> [--max-wall-h H] [--accept-code-change WHY]
                         [--accept-eval-change WHY] [--accept-resolution D00s|D00bs WHY]
    run_candidates.py --campaign-dir <dir> [--candidates <yaml>] --dry-run
    run_candidates.py --campaign-dir <dir> --q-spread-check <suite stamp> [<stamp> ...]  # pre-launch
    launch_candidates.sh --campaign-dir <dir> --reference-chain <name> [--seed N]  # a references: run

A relaunch under a new ``eval_sha`` first runs the eval-only MIGRATION (plan 2026-09-21 D9: rescreen
/ reeval / D00s / D00bs / rescore, derived from the records by :func:`migration_item`) before any
resume or queue item; nothing in it trains.

Everything the campaign is comes from ``candidates.yaml`` (chains, ranked candidates, milestones,
the decision rule, the resolution inputs): this file contains no dated path, no campaign identity
and no candidate. The record lives beside that yaml -- ``ledger.json`` (this driver's, atomic),
``ledger_overrides.jsonl`` (the CLI's, append-only), ``state.json`` / ``heads.json`` (the
``run_phases`` contract, so ``heartbeat_phases.sh`` works unchanged), ``REPORT.md``,
``CHANGELOG.md``, ``notify.log``, ``orchestrator.log``.

Reuse: the training/eval subprocess layer is ``tools/run_phases.py``, imported BY PATH (the
``run_morph_exposure.py`` precedent) and used through its campaign seams -- ``train_retry``
(``extra_env`` / ``interrupt_check`` / ``run_name`` / ``abort_track_ratio_min``),
``eval_morph`` / ``eval_obst`` (``carry_overlay`` / ``interrupt_check`` / ``max_attempts``),
``summarize``, ``Campaign``, ``_wait_isaac_clear``, ``CURRENT_CHILD``. Nothing in that module is
edited from here, and it refuses to run its own phase chain in a ladder directory.

Stages and candidates are env OVERLAYS on the existing presets, never new presets: the subprocess
gets ``KRABBY_OMNI_PHASE=<a real preset>`` plus ``KRABBY_PLANT`` plus the overlay, and
``params/env.yaml`` is audited against that overlay after every window (``overlay_checks``), which
is what makes "the recipe" a fact rather than a claim.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: campaign -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
REPO = HERE.parents[5]
PARKOUR = REPO / "parkour"
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"


def _load(path: Path, name: str):
    """Import a module by PATH (the campaign-driver precedent) -- no package, no Isaac import."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load(HERE / "run_phases.py", "run_phases_ladder")
lc = _load(HERE / "ledger_cli.py", "ledger_cli_ladder")
ph = rp.ph
PHASES_ALLOWED_BASES = ("1a", "1b", "1c", "2a", "2b")
PROMOTION_STAGES = ("2a", "2b")
#: ``checkpoint`` (gait-motion-quality campaign, 2026-09-23) is the only kind spelled as a MAPPING:
#: ``resume_from: {checkpoint: <repo-relative .pt>}`` forks a named head that is NOT one of this
#: campaign's own screens (``recipe_head`` resolves only those) -- the fine-tune screens all fork
#: X08:c6. The path is part of the recipe's identity (``compose_recipe`` hashes it into
#: ``recipe_id``), and every seed of the recipe forks the same file.
#: NOTE(clock-deprecation 2026-09-23, plan step 4): ``reference`` is the second mapping kind --
#: ``resume_from: {reference: <name>}`` forks the recorded head of a ``references:`` run (the
#: reference-chain launcher, :func:`reference_loop`), resolved at LAUNCH: the confirm tails fork the
#: ``t3_s2`` reference's head, which does not exist when the yaml is written.
RESUME_KINDS = ("scratch", "previous_stage", "recipe_head", "recipe_head_s2", "checkpoint", "reference")


def resume_kind(stage: dict) -> str:
    """A stage's resume kind: its ``resume_from`` string, or ``checkpoint`` / ``reference`` for the
    two mapping forms."""
    source = (stage or {}).get("resume_from")
    if isinstance(source, dict):
        if set(source) == {"checkpoint"}:
            return "checkpoint"
        if set(source) == {"reference"}:
            return "reference"
        return f"<mapping {sorted(source)}>"
    return str(source)


def resume_checkpoint(stage: dict) -> str | None:
    """The repo-relative checkpoint a ``resume_from: {checkpoint: <path>}`` stage forks, else None."""
    source = (stage or {}).get("resume_from")
    if isinstance(source, dict) and source.get("checkpoint"):
        return str(source["checkpoint"])
    return None


def resume_reference(stage: dict) -> str | None:
    """The ``references:`` name a ``resume_from: {reference: <name>}`` stage forks, else None."""
    source = (stage or {}).get("resume_from")
    if isinstance(source, dict) and source.get("reference"):
        return str(source["reference"])
    return None

# ------------------------------------------------------------------ overlay vocabulary
#: The pool and its slot projection. Written into a 2a/2b overlay ONLY in the rescaled teacher form
#: (see :func:`compose_recipe`) -- a chain stage's own pool describes a formation window's
#: distribution, and carrying it into the teacher windows verbatim would train the wrong regime.
POOL_KEYS = ("KRABBY_OMNI_CMD_POOL", "KRABBY_OMNI_SLOTS")
#: Keys a chain stage sets to EMULATE a preset it is not (the curriculum runs on the ``1a`` preset
#: and has to re-state 1c's economy). They are chain-stage-scoped and never reach 2a/2b, whose own
#: presets carry the facing ramp (0.2:1.2 / 0.2:2.4) the promotion is defined by.
PRESET_EMULATION_KEYS = ("KRABBY_OMNI_DIR_BAND", "KRABBY_OMNI_RSI_SOURCE", "KRABBY_OMNI_FACING",
                         "KRABBY_YAW_W")
#: Never in an overlay: the first two are the driver's own (a window whose preset or plant came from
#: an overlay is a window whose record lies), the next two raise at config import (one task per
#: process), and the USD path is the plant's business.
FORBIDDEN_OVERLAY_KEYS = ("KRABBY_OMNI_PHASE", "KRABBY_PLANT", "KRABBY_PHASE", "KRABBY_SIDE_PHASE",
                          "KRABBY_HEX_USD_PATH")
#: Overlaying one of these moves the DENOMINATOR every speed ratio and every facing fraction is
#: measured against, so the candidate has to be scored in absolute units (decision 5). The validator
#: refuses such a candidate unless it declares ``absolute_units: true``.
ABSOLUTE_UNITS_KEYS = ("KRABBY_OMNI_ANG_VEL_MAX", "KRABBY_OMNI_ANG_VEL_CLIP",
                       "KRABBY_OMNI_SPEED_ELLIPSE")

# ------------------------------------------------------------------ code_sha / eval_sha
#: The pre-suite ``code_sha`` pathspecs (2026-09-19 .. 2026-09-21), kept VERBATIM: every ladder that
#: predates the per-category plan recorded its ``lad.code_sha`` over this list, and the
#: ``--accept-eval-change`` alias proof (plan D8) recomputes that value at an old commit to show the
#: training tree did not move when the list was narrowed.
CODE_PATHSPECS_V1 = (
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_phases.py",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/gait_eval",
    "parkour/scripts",
    "parkour/parkour_isaaclab",
)
#: The tree the TRAINING depends on (plan D8): ``CODE_PATHSPECS_V1`` minus the eval-side files. A
#: change here invalidates every recorded screen (a relaunch under a different ``code_sha`` is
#: refused without ``--accept-code-change``, and a ``B<k>`` rerun follows); an eval-only change is
#: :data:`EVAL_PATHSPECS`' business and re-EVALUATES instead of retraining.
CODE_PATHSPECS = (
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp",
    "parkour/scripts",
    "parkour/parkour_isaaclab",
)
#: The tree the EVALUATION depends on (plan D8): the harness (``gait_eval`` + its entry point), the
#: omni manifest (every screen's holds), the eval side of ``run_phases.py`` (its TRAINING seam is
#: pinned by ``test_run_phases_training_seam_is_unchanged``, not by this hash -- the price of not
#: splitting the module) and ``eval_categories.py`` (what a stage trained, hence what is decisive).
#: A changed ``eval_sha`` is refused without ``--accept-eval-change WHY``; ``ok_screen`` then
#: requires the current value, so an old-eval screen is never a control or an anchor until it is
#: rescreened.
EVAL_PATHSPECS = (
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/gait_eval",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/eval/scenarios_omni.yaml",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_phases.py",
    "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/eval_categories.py",
)
#: How far back the alias proof looks for the commit that recorded ``lad.code_sha`` (plan D8).
ALIAS_PROOF_DEPTH = 300
#: Dropped from ``code_sha`` by a Python ``fnmatch`` (git's ``:(exclude)`` magic is deliberately not
#: used -- it interacts with ``git status`` differently and the two checks must filter identically):
#: documentation, every campaign record tree, and the test suite. A record commit must not move the
#: hash, or every milestone commit would invalidate the campaign's own screens.
#: DEVIATION, recorded: the plan names three patterns; ``tests/*`` is a fourth. It matches only a
#: REPO-ROOT ``tests/`` path, which no shipped ``CODE_PATHSPECS`` entry can produce (every positive
#: pathspec is at least two segments deep, so a test under one of them is ``.../tests/...`` and is
#: caught by ``*/tests/*``). It is therefore inert on this tree and exists so that adding a
#: top-level pathspec later cannot quietly fold the suite into the hash. Both the ``code_sha`` rows
#: and the dirty check filter through it identically, so it cannot make the two disagree.
CODE_DROP_GLOBS = ("*.md", "*/experiments/20*", "*/tests/*", "tests/*")
#: The ladder's own tools, hashed informationally per entry (``tools_sha``) and NEVER a refusal:
#: they produce no training or eval output, and every decision they make is a pure function of
#: records a later reader can re-derive.
TOOL_FILES = ("run_candidates.py", "ledger_cli.py", "resolution.py")

# ------------------------------------------------------------------ gates, floors, watch thresholds
#: Decision 4's stage FLOOR: survival through the stage's last enabled hold.
FLOOR_SURVIVAL_MIN = 0.30
#: The from-scratch ``dead_walker`` floor is RELATIVE to B0's own early trace (``D03``).
DEAD_WALKER_FLOOR_MIN = 0.35
DEAD_WALKER_FLOOR_FRAC = 0.75
#: Iteration window ``D03`` is measured over (B0's its-800-1000 ``track_ratio_mean`` mean).
D03_AT_ITERS = rp.ABORT_AT_ITERS
D03_WINDOW = rp.ABORT_WINDOW
#: RETENTION is WATCH + notify, NEVER a verdict (decision 4): these two numbers only decide whether
#: a line says "held" or "DROPPED", so they are thresholds on a report, not on a campaign.
RETENTION_SURV_DROP = 0.10
RETENTION_RATIO_DROP = 0.20
#: Walk floor arithmetic, restated from the command module's own constants at validation time.
WALK_MARGIN_M_S = 0.01
MIN_CAP_HEADROOM_M_S = 0.01
#: The manifest's turn-hold feasibility rule: a 90-degree turn needs a setup second plus the time
#: the cap itself takes, inside the hold. Recomputed with a cap candidate's overlaid cap.
TURN_SETUP_S = 1.0
QUARTER_TURN_RAD = math.pi / 2
#: The plan's explicit floor for an overlaid ``ang_vel_max`` ("refuse < 0.15"). The arithmetic above
#: refuses below ``1.0 + (pi/2)/12 = 0.1428`` on the manifest's 12-s turn hold, so this closes the
#: 0.1428-0.15 band the recomputation alone would accept. Both are checked; the stricter one bites.
TURN_CAP_MIN = 0.15
#: The gait clock's stop speed (``omni_knobs.CLOCK_TURN_MIN_SPEED``) and the stricter bound a CAP
#: candidate must clear unless its ``question`` records the cadence change it makes.
CLOCK_TURN_MIN_SPEED = 0.25
CAP_CADENCE_MIN = 0.35
#: Seconds the infra path waits (in interrupt-checked ticks) before its ONE retry.
INFRA_WAIT_S = 600
INFRA_TICK_S = 30
#: The advisory replay's own wall limit. It is a CPU-only re-score of saved npz rows (minutes), so
#: half an hour is generous; past it the child is stopped and the window runs without the screen.
REPLAY_TIMEOUT_S = 1800
#: Kit's own logs (the cache-corruption signature) and this process's cgroup (the OOM counter).
KIT_LOG_DIR = Path(PY).parents[1] / "lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1"
CGROUP_ROOT = Path("/sys/fs/cgroup")

TERMINAL_STATUSES = lc.TERMINAL_STATUSES
CANDIDATE_OPTIONS = list(lc.CANDIDATE_OPTIONS)
MILESTONE_OPTIONS = list(lc.MILESTONE_OPTIONS)
#: Legal status transitions (the plan's transition table), keyed by source status. ``actor`` is the
#: only thing that may perform the move: a user, the driver (a pooled pair), or a milestone.
#: ``user`` appears beside ``derived`` on every ``squashed`` target because the plan's table ends
#: "any effective candidate entry -> squashed (user `decide --choose run`)" -- that move is a user's.
TRANSITIONS = {
    "speculative": {"confirmed": ("user",), "overruled": ("user", "driver", "milestone"),
                    "final": ("driver",), "squashed": ("derived", "user")},
    "decided": {"confirmed": ("user",), "overruled": ("user", "driver", "milestone"),
                "final": ("driver",), "squashed": ("derived", "user")},
    # `driver` sits beside `user`/`milestone` on `final -> overruled` because the plan defines
    # `overruled(driver)` as `override.by in {driver, milestone}` (plan:545): the actor set has to
    # match that definition even though a `final` entry has no further pair to overrule it today.
    "final": {"confirmed": ("user",), "overruled": ("user", "driver", "milestone"),
              "squashed": ("derived", "user")},
    "confirmed": {"overruled": ("user",), "squashed": ("derived", "user")},
    "overruled": {"confirmed": ("user",), "overruled": ("user",),
                  "squashed": ("derived", "user")},
    "skipped": {"squashed": ("derived", "user")},
    # The two MILESTONE-only statuses. Their `squashed` target also names `driver` and `user`: the
    # plan's row is "`await` -> `squashed` (derived, after any recipe-changing override of a block
    # member)", and both the milestone's own automatic drop (driver) and the user's
    # `resume-milestone --drop` (user) ARE that override -- the entry is squashed as its
    # consequence, in the same write. Listing them keeps those writes inside `check_transition`
    # instead of bypassing it.
    "open": {"final": ("driver",), "await": ("driver",),
             "squashed": ("derived", "driver", "user")},
    "await": {"accepted_i": ("user",), "final": ("driver", "user"), "await": ("driver",),
              "squashed": ("derived", "driver", "user")},
    # RESURRECTION, the other half of the plan's "`squashed` -> re-decided by a new experiment":
    # decision 7 also says "reverting an overrule RESURRECTS an experiment squashed by derived-view
    # causes (a)-(c)", and that restores the entry's `pre_squash_status`. Those writes bypassed the
    # guard entirely, so the one table a reader consults to know what may move a ledger entry did
    # not describe them. Only `derived` may make them -- a user squash (cause d/e) is never undone.
    "squashed": {"squashed": ("derived",), "speculative": ("derived",), "decided": ("derived",),
                 "final": ("derived",), "confirmed": ("derived",), "overruled": ("derived",),
                 "skipped": ("derived",), "open": ("derived",), "await": ("derived",)},
}


def check_transition(src: str, dst: str, actor: str, *, override_by: str | None = None) -> None:
    """Raise unless ``src -> dst`` is a legal move for ``actor`` (the plan's transition table).

    Every ledger write goes through this, the derived squash and resurrection writes included. The
    table is small and the failure it prevents is not: a driver that silently moved a ``confirmed``
    (frozen) decision to ``final`` would erase a user's answer and no later reader could tell.

    ``override_by`` splits the one row the table keys on the OVERRIDE's actor rather than the
    entry's status (PLAN_AMENDMENTS 24): the plan has ``overruled(driver) -> confirmed`` and no
    ``overruled(user) -> confirmed``, because there ``confirm`` means "revert the DRIVER's override"
    -- on a user's own overrule it would silently freeze the entry's pre-override pick instead of
    the answer the user gave, which is the opposite of what the word means.
    """
    allowed = TRANSITIONS.get(src)
    if allowed is None:
        raise ValueError(f"unknown source status {src!r}")
    actors = allowed.get(dst)
    if actors is None:
        raise ValueError(f"illegal ledger transition {src!r} -> {dst!r} (legal: {sorted(allowed)})")
    if actor not in actors:
        raise ValueError(f"{actor!r} may not move {src!r} -> {dst!r} (only {actors})")
    if src == "overruled" and dst == "confirmed" and override_by == "user":
        raise ValueError("this entry is already overruled BY THE USER and frozen: `confirm` is the "
                         "revert of a driver/milestone override and has nothing to revert here. "
                         "To pick again use `overrule <Did> --choose <choice> --why ...`")


# ================================================================== small pure helpers
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def dget(d, dotted: str, default=None):
    """``dget(summary, "facing_closed_by_hold.turn_l90")`` -- ``default`` on ANY missing step.

    The decision rule names its metrics as dotted paths into ``summarize()``'s output, so this is
    the only place that turns a metric id into a number. A missing step returns the default rather
    than raising: a metric the row did not produce is MISSING, which the rule has an answer for.
    """
    cur = d
    for part in str(dotted).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _num(v):
    """``float(v)`` or ``None`` -- ``None``, non-numbers and NaN are all MISSING, not zero."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def fmt_num(v, digits: int = 3) -> str:
    f = _num(v)
    return "None" if f is None else f"{f:.{digits}f}"


def slots_str(probs) -> str:
    """``(a, b, c, d) -> "a:b:c:d"`` with ten significant digits.

    Ten digits rather than ``repr``: the value is re-parsed by ``omni_knobs.parse_slots`` and
    compared against the pool's own projection with a 1e-6 tolerance, so 1e-10 of rounding is inert
    while ``repr``'s 17-digit tails would make every config dump unreadable. The round trip is
    pinned by a test.
    """
    return ":".join(f"{float(p):.10g}" for p in probs)


def parse_slots_str(text: str) -> tuple:
    return tuple(float(x) for x in str(text).split(":"))


def numeric_equal(a, b, tol: float = 1e-9) -> bool:
    """True when two overlay VALUES mean the same thing (``0.05:0.55`` vs ``0.0500:0.55``).

    Overlay values are strings, and two candidates may compute the same colon-separated numbers by
    different routes (a teacher pool's projection and a hand-written ``teacher_overlay``). A
    conflict report has to be about MEANING, or the driver would refuse its own arithmetic.
    """
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    pa, pb = sa.split(":"), sb.split(":")
    if len(pa) != len(pb):
        return False
    try:
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(pa, pb))
    except ValueError:
        return False


# ================================================================== code_sha / tools_sha
def _git_out(args, repo: Path = REPO) -> str:
    out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def code_drop(path: str) -> bool:
    """True when ``path`` is filtered out of ``code_sha`` (documentation, records, tests)."""
    return any(fnmatch.fnmatch(path, pat) for pat in CODE_DROP_GLOBS)


def sha_rows(rows) -> str:
    """The hash of ``["<blob> <path>", ...]`` -- sorted, so ``ls-tree`` order cannot move it."""
    return sha256_text("\n".join(sorted(rows)))


def code_rows(repo: Path = REPO, *, pathspecs=None, commit: str = "HEAD") -> list[str]:
    """``["<blob id> <path>", ...]`` for every tracked file under ``pathspecs`` at ``commit``.

    Defaults: :data:`CODE_PATHSPECS` at HEAD (the ``code_sha`` rows). ``eval_rows`` and
    ``code_sha_at`` reuse this exact row format and drop filter, so a value recomputed at an old
    commit is comparable with the one a launch recorded there.
    """
    specs = CODE_PATHSPECS if pathspecs is None else tuple(pathspecs)
    rows = []
    for line in _git_out(["ls-tree", "-r", str(commit), "--", *specs], repo).splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 3 or not path:
            continue
        if code_drop(path):
            continue
        rows.append(f"{parts[2]} {path}")
    return rows


def code_sha(repo: Path = REPO) -> str:
    """sha256 over the code-bearing tree at HEAD (see :data:`CODE_PATHSPECS`).

    HEAD's own commit id is deliberately NOT the hash: the campaign commits its own records
    (``$LAD``, the README, the charter) while it runs, and a hash that moved on those would
    invalidate every recorded screen for a documentation edit.
    """
    return sha_rows(code_rows(repo))


def eval_rows(repo: Path = REPO, *, commit: str = "HEAD") -> list[str]:
    """``code_rows`` over :data:`EVAL_PATHSPECS` (the same row format and drop globs)."""
    return code_rows(repo, pathspecs=EVAL_PATHSPECS, commit=commit)


def eval_sha(repo: Path = REPO) -> str:
    """sha256 over the eval-bearing tree at HEAD (plan D8; see :data:`EVAL_PATHSPECS`)."""
    return sha_rows(eval_rows(repo))


def code_sha_at(commit: str, pathspecs, repo: Path = REPO) -> str:
    """The ``code_sha`` a launch at ``commit`` would have recorded over ``pathspecs``.

    ``git ls-tree -r <commit>`` with the SAME row format and drop globs as :func:`code_rows` -- so
    ``code_sha_at(C, CODE_PATHSPECS_V1)`` at the commit a pre-suite campaign launched on reproduces
    its recorded ``lad.code_sha`` exactly (the alias proof's first half).
    """
    return sha_rows(code_rows(repo, pathspecs=pathspecs, commit=commit))


def prove_code_alias(recorded: str, *, repo: Path = REPO, depth: int = ALIAS_PROOF_DEPTH,
                     current: str | None = None) -> dict:
    """Plan D8's alias proof: did the TRAINING tree stay put while the pathspecs were narrowed?

    Finds the newest commit ``C`` in ``git log -n depth`` with ``code_sha_at(C, CODE_PATHSPECS_V1) ==
    recorded`` (the commit the recorded ``code_sha`` was measured on, or one with the same code
    tree), then requires ``code_sha_at(C, CODE_PATHSPECS) == code_sha()`` (``current``): the narrowed
    training-side tree at that commit is the one HEAD carries now. Returns ``{ok, sha, proved_at,
    narrowed_at, current, why}``; ``ok`` False with the reason when no commit matches or the
    training tree moved (the caller then falls through to the ``code_sha`` refusal).
    """
    current = code_sha(repo) if current is None else current
    out = {"ok": False, "sha": recorded, "proved_at": None, "narrowed_at": None, "current": current,
           "why": ""}
    try:
        commits = _git_out(["log", "-n", str(int(depth)), "--format=%H"], repo).split()
    except RuntimeError as exc:
        out["why"] = f"git log failed: {exc}"
        return out
    for commit in commits:
        if code_sha_at(commit, CODE_PATHSPECS_V1, repo) != recorded:
            continue
        narrowed = code_sha_at(commit, CODE_PATHSPECS, repo)
        out.update(proved_at=commit, narrowed_at=narrowed)
        if narrowed == current:
            out.update(ok=True, why=(f"the recorded code_sha {recorded[:12]} is the pre-suite hash of "
                                     f"{commit[:12]}, whose training-side tree ({narrowed[:12]}) is "
                                     f"HEAD's ({current[:12]})"))
        else:
            out["why"] = (f"the recorded code_sha {recorded[:12]} is {commit[:12]}'s, but the training "
                          f"tree also moved since ({narrowed[:12]} -> {current[:12]}): use "
                          "--accept-code-change")
        return out
    out["why"] = (f"no commit in the last {depth} carries the recorded code_sha {recorded[:12]} under "
                  "the pre-suite pathspecs: the training tree cannot be proved unchanged -- use "
                  "--accept-code-change")
    return out


def pathspec_covers(path: str, pathspecs) -> bool:
    """Is ``path`` under one of ``pathspecs`` (a file spec, or a directory spec as a prefix)?"""
    path = str(path).strip("/")
    for spec in pathspecs:
        spec = str(spec).strip("/")
        if path == spec or path.startswith(spec + "/"):
            return True
    return False


def dirty_paths(repo: Path = REPO) -> list[str]:
    """Code- or eval-bearing paths that are modified or untracked (launch is refused while any is).

    Both :data:`CODE_PATHSPECS` and :data:`EVAL_PATHSPECS` (plan D8: "the dirty check covers both
    lists"). Filtered by exactly the same ``fnmatch`` as :func:`code_sha`, so the two can never
    disagree about what "the code" is. Untracked lines count: an un-added new reward module is as
    much a change as an edited one.
    """
    out = []
    specs = list(dict.fromkeys(CODE_PATHSPECS + EVAL_PATHSPECS))
    for line in _git_out(["status", "--porcelain", "--", *specs], repo).splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:                       # a rename: the destination is what HEAD would hash
            path = path.split(" -> ", 1)[1].strip()
        path = path.strip('"')
        if code_drop(path):
            continue
        out.append(path)
    return out


def tools_sha(extra: Path | None = None) -> str:
    """Informational hash of the ladder's own tools (+ the replay script), from the WORKING TREE.

    From the working tree rather than from git on purpose: these files are outside the code freeze
    and may legitimately be uncommitted while the campaign runs, and the honest record is what
    actually ran. Never a refusal.
    """
    rows = []
    for name in TOOL_FILES:
        path = HERE / name
        rows.append(f"{sha256_file(path) if path.exists() else 'absent'} {name}")
    if extra is not None and Path(extra).exists():
        rows.append(f"{sha256_file(extra)} {Path(extra).name}")
    return sha_rows(rows)


# ================================================================== the command pool (lazy)
_POOL_MATH = None


def pool_math():
    """The Isaac-free command-math module (``POOL_TYPES``, ``parse_cmd_pool``, ``rescale_to``).

    Loaded lazily and by path, the way ``omni_knobs`` does it: it imports torch, and this module is
    also imported by a layout test that must not pay for that. One module identity for the whole
    process, so ``CmdPool`` instances compare and render consistently.
    """
    global _POOL_MATH
    if _POOL_MATH is None:
        _POOL_MATH = _load(rp.PKG / "mdp/crab_hex_omni_command_math.py", "crab_hex_omni_command_math_ladder")
    return _POOL_MATH


_KNOBS = None


def knobs():
    """``omni_knobs`` (the parsers every overlay value must round-trip through), loaded by path."""
    global _KNOBS
    if _KNOBS is None:
        _KNOBS = _load(rp.PKG / "config/crab_hex/omni_knobs.py", "omni_knobs_ladder")
    return _KNOBS


_EVAL_CATEGORIES = None


def eval_categories():
    """``$OMNI/eval_categories.py`` (plan D4/D5), loaded by path and cached.

    The ONE home of ``trained_categories``, ``expand_rule`` and ``composite_guards``: the driver and
    ``resolution.py`` must share those definitions, never re-implement them (a second copy of the
    expansion is how a missing block becomes a silent all-TIE nobody can see).
    """
    global _EVAL_CATEGORIES
    if _EVAL_CATEGORIES is None:
        _EVAL_CATEGORIES = _load(rp.PKG / "eval_categories.py", "crab_hex_omni_eval_categories_ladder")
    return _EVAL_CATEGORIES


def rule_version(rule: dict | None) -> int:
    """``decision_rule.rule_version`` (1 when absent: the pre-suite rule carried no version)."""
    try:
        return int((rule or {}).get("rule_version") or 1)
    except (TypeError, ValueError):
        return 1


def is_rule_v2(rule: dict | None) -> bool:
    """A per-category (``rule_version: 2``) rule, raw or already expanded."""
    return rule_version(rule) == 2


class ResolutionMissing(RuntimeError):
    """A v2-scored comparison has no usable resolution: no ``D00s``, no ``D00bs`` at the current
    ``eval_sha``, or a decisive block narrower than the ``decisive_at`` its R was measured on (plan
    D5/D9). The campaign halts ``halt:resolution_missing`` -- never a header."""


def parse_pool(text: str):
    return pool_math().parse_cmd_pool(str(text))


#: Significant digits every COMPOSED pool weight/scale is rounded to before it is rendered.
#: Twelve, not a fixed number of decimals: ``render_cmd_pool`` is deliberately ``repr``-precise (a
#: ``f"{w:.3f}"`` would make the render lossy and the pool would stop matching its own ``SLOTS``
#: projection), and twelve significant digits is two orders below the tightest round-trip tolerance
#: the pool is ever read with (1e-9 in ``numeric_equal``, 1e-6 in the projection checks) while being
#: far above the ~1e-16 ULP tail a transform leaves behind.
POOL_SIG_DIGITS = 12


def _canonical_pool(pool):
    """The same pool with every weight and scale rounded to :data:`POOL_SIG_DIGITS` (amendment 27).

    The driver never writes a pool literal: it COMPOSES one from transforms (``parse_cmd_pool``,
    ``apply_pool_edit``, ``rescale_to``) and records the rendered string. A rescale that is
    numerically the identity still returns ``0.14000000000000004`` for a literal ``0.14``, and
    ``recipe_id`` is sha256 over exactly those strings -- so the ULP tail is an identity change:
    it moved four shipped recipe ids (C8-C11) the day the teacher pool started rescaling the
    COMPOSED final stage, and ``base_recipe_id`` / ``recipe_id_if_adopted`` / ``ok_screen`` matching
    and the squash view's "the base hash matches again" resurrection all key on that hash.

    Rounding here (never inside ``render_cmd_pool``, which is the mdp module's own contract and is
    shared with the launcher) makes the parse -> transform -> render path reproduce the yaml literal
    whenever the transform was a no-op, and makes it deterministic when it was not.
    """
    return pool._replace(weights=tuple(_sig(w) for w in pool.weights),
                         ang_scale=tuple(_sig(a) for a in pool.ang_scale),
                         speed_scale=tuple(_sig(s) for s in pool.speed_scale))


def _sig(x) -> float:
    return float(f"{float(x):.{POOL_SIG_DIGITS}g}")


def render_pool(pool) -> str:
    """The canonical rendering of a composed pool -- the ONE place the driver renders one.

    Canonicalised here as well as at each composition site, so a future rendering site cannot
    reintroduce the float tail by forgetting to call :func:`_canonical_pool` (it is idempotent).
    """
    return pool_math().render_cmd_pool(_canonical_pool(pool))


def pool_slots(pool) -> tuple:
    return tuple(float(p) for p in pool.projected_slot_probs())


def apply_pool_edit(pool, edit: dict, *, require_match: bool = True):
    """One ``pool_edit`` transform: ``set_share`` (+ ``renormalise``), ``ang_scale``, ``speed_scale``.

    A candidate declares a TRANSFORM rather than a pool string so that share edits and scale edits
    compose in ledger order across several adopted candidates, and so the same edit applies to
    whatever chain the recipe happens to carry.

    ``set_share`` sets a type's raw weight. With ``renormalise: true`` the freed share is
    redistributed PROPORTIONALLY over the walking (slot-1) types only, so the turn and move+turn
    shares -- the regime the facing work is measured on -- are unchanged: C8's stand 0.15 -> 0.05 on
    the c6 pool projects to ``0.05:0.65:0.10:0.20`` rather than scaling every slot.

    ``ang_scale`` / ``speed_scale`` map an ``fnmatch`` pattern to a per-type scale. ``"*_turn"``
    matches ``fwd_turn``/``side_l_turn``/``side_r_turn``/``back_turn``/``omni_turn`` and NOT the
    in-place ``turn`` type -- which is the intent (the infeasible demand is move+turn), and is
    pinned by a test rather than left to the reader of a glob.

    ``require_match`` refuses a pattern that matches no type in THIS pool. A typo ("*_trun") must be
    caught, but an early curriculum stage legitimately has no ``*_turn`` type at all -- the capability
    is not enabled yet -- so ``compose_recipe`` applies edits with ``require_match=False`` and the
    VALIDATOR instead requires the pattern to bite on at least one stage of the chain. A silently
    inert reduced-demand arm is a wasted GPU window that looks like evidence; an inert edit on a
    stage whose types do not exist yet is the curriculum working as designed.
    """
    math_mod = pool_math()
    types = math_mod.POOL_TYPES
    names = list(pool.names)
    w = [float(x) for x in pool.weights]
    a = [float(x) for x in pool.ang_scale]
    s = [float(x) for x in pool.speed_scale]
    freed = 0.0
    for name, value in (edit.get("set_share") or {}).items():
        if name not in names:
            raise ValueError(f"pool_edit set_share names {name!r}, which is not in this pool "
                             f"({names}) -- the edit would be silently inert")
        i = names.index(name)
        freed += w[i] - float(value)
        w[i] = float(value)
    if edit.get("renormalise") and abs(freed) > 0.0:
        idx = [j for j, n in enumerate(names) if types[n].slot == 1 and w[j] > 0.0]
        total = sum(w[j] for j in idx)
        if not idx or total <= 0.0:
            raise ValueError("pool_edit renormalise has no walking (slot-1) type with positive "
                             f"weight to absorb the freed share {freed} (pool {names})")
        scale = (total + freed) / total
        if scale <= 0.0:
            raise ValueError(f"pool_edit renormalise would drive the walking share to {total + freed}")
        for j in idx:
            w[j] *= scale
    for field, table in (("ang_scale", a), ("speed_scale", s)):
        for pattern, value in (edit.get(field) or {}).items():
            hit = [j for j, n in enumerate(names) if fnmatch.fnmatchcase(n, str(pattern))]
            if not hit:
                if require_match:
                    raise ValueError(f"pool_edit {field} pattern {pattern!r} matches no type in this "
                                     f"pool ({names}) -- the edit would be silently inert")
                continue
            for j in hit:
                table[j] = float(value)
    return math_mod.CmdPool(tuple(names), tuple(w), tuple(a), tuple(s))


def pool_edit_touches_shares(edit: dict) -> bool:
    return bool(edit.get("set_share")) or bool(edit.get("renormalise"))


# ================================================================== candidates.yaml
def repo_path(rel) -> Path:
    """A yaml path resolved against the repo root (an absolute one is taken as given).

    Every path in ``candidates.yaml`` is written relative to the repo root so no driver and no record
    carries a machine-specific prefix; an absolute path is still honoured, which is what lets a test
    point the replay lever at a scratch script.
    """
    path = Path(rel)
    return path if path.is_absolute() else (REPO / path)


def load_candidates(path: Path) -> dict:
    import yaml  # the harness's own dependency
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: candidates.yaml must be a mapping")
    doc["__path__"] = str(Path(path).resolve())
    return doc


def candidate_ids(cands: dict) -> list[str]:
    return [c["id"] for c in cands.get("candidates") or []]


def candidate_by_id(cands: dict, cid: str) -> dict | None:
    for c in cands.get("candidates") or []:
        if c.get("id") == cid:
            return c
    return None


def baseline_control(cand: dict | None) -> bool:
    """``control: baseline`` (clock-deprecation plan step 4, 2026-09-23).

    The candidate is one of several ALTERNATIVES judged against the baseline, never a step of a
    composition: its control is the baseline's own s3/s2 screens whatever was adopted before it,
    its drift anchor is pinned to the baseline, it is excluded from compose-on-adopt, and its
    adoption never becomes another candidate's anchor (:func:`compose_recipe`). Without it the
    driver would judge A4500 and A3000 against an adopted A6000 (the three wave-1 arms remove the
    clock at three different boundaries -- they answer "removal HERE vs clock held", not "and then
    also here").
    """
    return str((cand or {}).get("control") or "").strip() == "baseline"


def chain_by_name(cands: dict, name: str) -> dict:
    chains = cands.get("chains") or {}
    if name not in chains:
        raise KeyError(f"unknown chain {name!r} (known: {sorted(chains)})")
    return chains[name]


def chain_stages(cands: dict, name: str) -> list[dict]:
    """The chain's stages, each annotated with ``previous_holds`` / ``previous_categories``.

    RETENTION (and, since the per-category plan, the floor's ``min_new``/``min_prev`` split)
    compares what was ALREADY trained before this stage, so the comparison set is a property of the
    chain rather than of the yaml row -- derived here, once, so every consumer (the recipe, the run
    loop, the report) reads the same list.

    A per-category stage (one that declares ``categories:``, plan D6) gets ``previous_categories``
    = the union of ``decisive(stages[:i])`` -- every earlier stage, ``eval: []`` ones included --
    from each stage's composed env on this chain alone (preset + the stage's own overlay). The run
    loop recomputes it from the HEADER's ``trained_categories_by_stage``, which carries the
    recipe's adopted edits as well; this is the declaration a resume and a report fall back to.
    """
    out = []
    prev: list = []
    stages = list(chain_by_name(cands, name).get("stages") or [])
    per_category = any("categories" in s for s in stages)
    prev_cats: set = set()
    for stage in stages:
        row = dict(stage)
        row["previous_holds"] = list(prev)
        prev = list(stage.get("holds") or prev)
        if per_category:
            row["previous_categories"] = _ordered_categories(prev_cats)
            prev_cats |= set(_stage_decisive(stage))
        out.append(row)
    return out


def _ordered_categories(cats) -> list[str]:
    """Category names in the suite's wave order (unknown names kept, sorted, at the end)."""
    order = list(eval_categories().CATEGORIES)
    s = set(cats or ())
    return [c for c in order if c in s] + sorted(s - set(order))


def _stage_decisive(stage: dict, overlay: dict | None = None) -> list[str]:
    """``decisive(stage)`` from the stage's composed env (``[]`` when it cannot be derived).

    The failure case (a stage whose stand share is 0 raises ``EvalMismatch``) is the VALIDATOR's to
    refuse and the run loop's to halt on; an annotation must not raise.
    """
    env = _effective_stage_cfg({}, stage, stage.get("overlay") if overlay is None else overlay)
    try:
        return sorted(eval_categories().trained_categories(env).decisive)
    except (ValueError, KeyError):
        return []


def stage_names(cands: dict, name: str) -> list[str]:
    return [str(s["stage"]) for s in chain_stages(cands, name)]


#: The rows BOTH promotion stages evaluate (plan D7) when ``candidates.yaml`` names none under
#: ``promotion.eval``: the per-category suite in place of ``slow``; light/ring stay legacy rows,
#: scored at milestones by the frozen v1 per-row rule. ``ph.EVAL_FULL`` (inside ``code_sha``) is
#: never edited -- it is what a PRE-SUITE (rule-v1) spec's promotions evaluated, and it stays that.
PROMOTION_EVAL = ("suite", "light", "ring")


def promotions_enabled(cands: dict) -> bool:
    """``promotion.enabled`` (default true). A campaign that disables promotions (the gait-motion-
    quality fine-tune ladder: every screen is a fork of one head, nothing is promoted through 2a/2b)
    composes EMPTY 2a/2b overlays -- no teacher pool rescale, no teacher overlay, no eval-carry copy
    -- so its recipe ids carry nothing that never runs, and the validator refuses any milestone."""
    return bool((cands.get("promotion") or {}).get("enabled", True))


def promotion_eval(cands: dict) -> list[str]:
    """The promotion stages' eval rows: ``promotion.eval`` from the yaml; else :data:`PROMOTION_EVAL`
    for a rule-v2 spec; else ``ph.EVAL_FULL`` (the frozen v1 records' promotions, unchanged)."""
    rows = (cands.get("promotion") or {}).get("eval")
    if rows:
        return [str(r) for r in rows]
    if is_rule_v2(cands.get("decision_rule")):
        return list(PROMOTION_EVAL)
    return list(ph.EVAL_FULL)


def promotion_stage_list(cands: dict) -> list[dict]:
    """The milestone promotion stages: 2a from the recipe head, then 2b from 2a.

    Spelled here rather than in the yaml because it is not a campaign choice: it is what "promote
    the recipe" means for this task, and both stages' iteration counts are their presets'. BOTH
    stages evaluate :func:`promotion_eval` (plan D7: ``[suite, light, ring]`` under rule v2);
    ``final_s2_stage_list`` and ``item_from_header``'s milestone branch inherit it from here.
    """
    rows = promotion_eval(cands)
    return [
        {"stage": "2a", "base": "2a", "iters": int(ph.PHASES["2a"].iterations), "from_scratch": False,
         "resume_from": "recipe_head", "overlay": {}, "eval": list(rows), "floor": {"kind": "none"},
         "holds": []},
        {"stage": "2b", "base": "2b", "iters": int(ph.PHASES["2b"].iterations), "from_scratch": False,
         "resume_from": "previous_stage", "overlay": {}, "eval": list(rows), "floor": {"kind": "none"},
         "holds": []},
    ]


def final_s2_stage_list(cands: dict) -> list[dict]:
    stages = promotion_stage_list(cands)
    stages[0] = dict(stages[0], resume_from="recipe_head_s2")
    return stages


# ================================================================== references (reference chains)
# NOTE(clock-deprecation 2026-09-23, plan step 4): `references:` names chains the driver runs OUTSIDE
# the queue -- `run_candidates.py --reference-chain <name> [--seed N]` -- with stage evals, recorded
# in heads.json (the stage recs) and `lad.references`, and NO decision: the never-clocked Z chain
# and the formation-seed-2 confirm prefix `t3_s2` (plus its tails) are judged by hand. The yaml
# shape is `references: {<name>: {chain: <chain, default the name>, seed: <int, optional>}}`.
def reference_spec(cands: dict, name: str) -> dict:
    """``references[name]`` normalised to ``{name, chain, seed}`` (``seed`` None when unpinned)."""
    refs = cands.get("references") or {}
    if name not in refs:
        raise KeyError(f"unknown reference {name!r} (known: {sorted(refs)})")
    ref = refs.get(name) or {}
    if not isinstance(ref, dict):
        ref = {}
    seed = ref.get("seed")
    return {"name": str(name), "chain": str(ref.get("chain") or name),
            "seed": None if seed in (None, "") else str(seed)}


def reference_only_chains(cands: dict) -> set:
    """The chains ONLY a ``references:`` entry runs (exempt from ``screen.iters``): no screen, no
    baseline and no candidate runs them."""
    refs = cands.get("references") or {}
    if not isinstance(refs, dict):
        return set()
    used = {(cands.get("screen") or {}).get("chain"), (cands.get("baseline") or {}).get("chain")}
    used |= {c.get("chain") for c in cands.get("candidates") or [] if c.get("chain")}
    out = set()
    for name in refs:
        try:
            out.add(reference_spec(cands, str(name))["chain"])
        except KeyError:
            continue
    return out - used


def reference_recipe(cands: dict, name: str, code: str = "") -> dict:
    """The recipe a reference run trains: its chain's own stages and overlays, nothing adopted.

    Same shape as :func:`compose_recipe`'s (``stage_env`` reads it), with no 2a/2b overlays -- a
    reference is never promoted. The identity hashes the reference's name, its chain and overlays,
    and any checkpoint/reference fork, so two references on one chain are two recipes.
    """
    ref = reference_spec(cands, name)
    stages = chain_stages(cands, ref["chain"])
    names = [str(s["stage"]) for s in stages]
    overlay_by_stage = {n: dict(stages[i].get("overlay") or {}) for i, n in enumerate(names)}
    ident = {"reference": ref["name"], "chain": ref["chain"], "overlay_by_stage": overlay_by_stage}
    recipe = {"chain": ref["chain"], "adopted": [], "overlay_by_stage": overlay_by_stage,
              "eval_carry": [], "conflicts": [], "anchor": None, "code_sha": code, "stages": stages,
              "has_pool": any("KRABBY_OMNI_CMD_POOL" in overlay_by_stage[n] for n in names),
              "reference": ref["name"]}
    forks = {str(s["stage"]): resume_checkpoint(s) for s in stages if resume_checkpoint(s)}
    if forks:
        ident["resume_checkpoints"] = recipe["resume_checkpoints"] = forks
    refs = {str(s["stage"]): resume_reference(s) for s in stages if resume_reference(s)}
    if refs:
        ident["resume_references"] = recipe["resume_references"] = refs
    recipe["recipe_id"] = sha256_text(canonical(ident))[:12]
    return recipe


# ================================================================== the recipe
def _cand_spec(cand: dict, chain_name: str) -> dict:
    """A candidate's contribution ON THIS CHAIN (``overlay_by_chain`` applied).

    C8 is the reason: "stand share 0.05" is a pool transform on a pooled chain and a raw
    ``KRABBY_OMNI_SLOTS`` string on the pool-less incumbent chain. The candidate declares both and
    the recipe picks the one that matches the chain it is composing.
    """
    spec = dict(cand)
    per_chain = (cand.get("overlay_by_chain") or {}).get(chain_name)
    if isinstance(per_chain, dict):
        spec.update(per_chain)
        spec.pop("overlay_by_chain", None)
    return spec


def _merge_overlay(dst: dict, src: dict, where: str, who: str, conflicts: list, *,
                   quiet=frozenset()) -> None:
    """Merge ``src`` into ``dst``, recording a conflict for any key that changes MEANING.

    ``quiet`` (keys): a DECLARED override of the chain's own value (a candidate's
    ``overrides_chain``, resolved by :func:`compose_recipe` to the keys whose ``dst`` value is still
    the chain's) is not a conflict. Empty -- every campaign before 2026-09-26 -- nothing changes.
    """
    for key, value in (src or {}).items():
        if key in dst and not numeric_equal(dst[key], value) and key not in quiet:
            conflicts.append(f"{where}: {key} set to {dst[key]!r} and {value!r} (by {who})")
        dst[key] = value


def _teacher_overlay_for(spec: dict, chain_name: str, stages: list[str]) -> dict:
    """What a candidate contributes to the 2a/2b overlays.

    An explicit ``teacher_overlay`` (including an empty one -- C1 and C1c declare ``{}`` so that
    ONLY the rescaled pool reaches the teacher windows) always wins. Otherwise a ``stages: all``
    overlay is carried whole, and a per-stage overlay contributes its FINAL stage's keys minus the
    pool keys (which reach 2a/2b only in the rescaled form) and minus the preset-emulation keys
    (whose 2a/2b values belong to the teacher presets).
    """
    if "teacher_overlay" in spec:
        return dict(spec.get("teacher_overlay") or {})
    overlay = spec.get("overlay") or {}
    if spec.get("stages") == "all":
        return dict(overlay)
    if not isinstance(overlay, dict) or not overlay:
        return {}
    last = None
    for name in stages:
        if name in overlay:
            last = overlay[name]
    if not isinstance(last, dict):
        return {}
    drop = set(POOL_KEYS) | set(PRESET_EMULATION_KEYS)
    return {k: v for k, v in last.items() if k not in drop}


def compose_recipe(ledger: dict, cands: dict, code: str, *, upto: str | None = None,
                   inclusive: bool = False, squashed=None) -> dict:
    """The effective recipe: chain, per-stage overlays, eval carry, anchor and identity hash.

    ``upto`` bounds the composition by RANK -- the base recipe of candidate ``Cid`` is composed of
    the effective choices of the candidates ranked BEFORE it (``upto=Cid, inclusive=False``), which
    is what makes an out-of-rank re-run's control well defined (Vocabulary: "base recipe"). With
    ``inclusive=True`` the bound includes ``upto``, which is what a milestone's ``after`` means.

    Only EFFECTIVE choices count: the newest non-squashed entry per candidate, with a user or driver
    override applied (an ``overruled`` entry's override is its choice; a ``squashed`` entry is not a
    choice at all).

    A ``control: baseline`` candidate (:func:`baseline_control`) never composes: bounded AT one,
    the recipe is the baseline (``inclusive=False``) or the baseline plus that one candidate
    (``inclusive=True``, :func:`_recipe_with`'s form), whatever was adopted before it; bounded
    anywhere else, such candidates are skipped (excluded from compose-on-adopt, never an anchor).
    A campaign without the key composes exactly as before.
    """
    squashed = set(squashed or ())
    order = candidate_ids(cands)
    if upto is not None:
        if upto not in order:
            raise KeyError(f"unknown candidate {upto!r} (known: {order})")
        cut = order.index(upto) + (1 if inclusive else 0)
        order = order[:cut]
    # NOTE(clock-deprecation 2026-09-23, plan step 4): `control: baseline` -- the candidate's base
    # recipe is the baseline's, so its control is the baseline's own s3/s2 screens and its drift
    # anchor re-resolves to `_baseline_anchor` (nothing it composes can qualify).
    if upto is not None and baseline_control(candidate_by_id(cands, upto)):
        order = [upto] if inclusive else []
    else:
        order = [cid for cid in order if not baseline_control(candidate_by_id(cands, cid))]

    adopted, anchor = [], None
    for cid in order:
        entry = newest_entry(ledger, cid, squashed)
        # No second stored-status test here: `newest_entry` already applies
        # `hidden_by_stored_squash`, and repeating `status == "squashed"` would put the previous
        # pass's cache back into the composition that `squash_view`'s fixpoint is computing.
        if entry is None:
            continue
        if lc.effective_choice(entry) != "ADOPT":
            continue
        adopted.append(cid)
        if _anchor_qualifies(entry) and not baseline_control(candidate_by_id(cands, cid)):
            anchor = {"candidate": cid, "decision": entry.get("id"),
                      "experiment": entry.get("experiment"), "why": _anchor_reason(entry)}
    if anchor is None:
        anchor = _baseline_anchor(ledger, cands, squashed)

    chain_name = cands.get("baseline", {}).get("chain") or cands.get("screen", {}).get("chain")
    for cid in adopted:
        cand = candidate_by_id(cands, cid) or {}
        if cand.get("chain"):
            chain_name = cand["chain"]
    stages = chain_stages(cands, chain_name)
    names = [str(s["stage"]) for s in stages]
    conflicts: list[str] = []

    overlay_by_stage = {name: dict(stages[i].get("overlay") or {}) for i, name in enumerate(names)}
    for st in PROMOTION_STAGES:
        overlay_by_stage[st] = {}
    # A campaign with promotions disabled composes nothing for 2a/2b (they stay `{}`): nothing is
    # ever promoted, so no teacher rescale, teacher overlay or eval-carry copy may enter the identity.
    promote = promotions_enabled(cands)

    pool_edits, teacher = [], {}
    eval_carry: list[str] = []
    chain_overlay = {name: dict(stages[i].get("overlay") or {}) for i, name in enumerate(names)}

    def _quiet(spec: dict, name: str) -> frozenset:
        # NOTE(stride/posture 2026-09-26): `overrides_chain` -- only while the stage still holds the
        # CHAIN's value for the key (another candidate's value is still a conflict).
        keys = spec.get("overrides_chain") or ()
        return frozenset(k for k in keys if k in overlay_by_stage.get(name, {})
                         and k in chain_overlay.get(name, {})
                         and numeric_equal(overlay_by_stage[name][k], chain_overlay[name][k]))

    for cid in adopted:
        spec = _cand_spec(candidate_by_id(cands, cid) or {}, chain_name)
        overlay = spec.get("overlay") or {}
        if spec.get("stages") == "all":
            for name in names:
                _merge_overlay(overlay_by_stage[name], overlay, f"stage {name}", cid, conflicts,
                               quiet=_quiet(spec, name))
        elif isinstance(overlay, dict):
            for name, block in overlay.items():
                if str(name) not in overlay_by_stage:
                    conflicts.append(f"{cid}: overlay names stage {name!r}, not in chain {chain_name}")
                    continue
                _merge_overlay(overlay_by_stage[str(name)], block, f"stage {name}", cid, conflicts,
                               quiet=_quiet(spec, str(name)))
        if spec.get("pool_edit"):
            pool_edits.append((cid, dict(spec["pool_edit"])))
        if promote:
            _merge_overlay(teacher, _teacher_overlay_for(spec, chain_name, names), "stage 2a/2b", cid,
                           conflicts)
        for key in spec.get("eval_carry") or []:
            if key not in eval_carry:
                eval_carry.append(key)

    # ---- the pool: one rendered string + its own SLOTS projection per stage, edits in ledger order
    for name in names:
        raw = overlay_by_stage[name].get("KRABBY_OMNI_CMD_POOL")
        if not raw or not str(raw).strip():
            continue
        pool = parse_pool(raw)
        for _cid, edit in pool_edits:
            pool = apply_pool_edit(pool, edit, require_match=False)
        pool = _canonical_pool(pool)          # amendment 27: the recorded string is the identity
        overlay_by_stage[name]["KRABBY_OMNI_CMD_POOL"] = render_pool(pool)
        overlay_by_stage[name]["KRABBY_OMNI_SLOTS"] = slots_str(pool_slots(pool))

    # ---- the teacher windows: rescale the FINAL stage's pool onto SLOTS_TEACHER, then re-apply the
    # adopted share edits so they survive the rescale (C8 -> 0.05:0.55:0.15:0.25), then the
    # candidates' own teacher overlays on top.
    # The COMPOSED final-stage pool, not the chain's raw one: the plan says "rescale_to(SLOTS_TEACHER)
    # of the RENDERED final-stage pool with every pool_edit applied". Reading `stages[-1].overlay`
    # skipped any candidate overlay that SETS a raw pool string on that stage (the validator allows
    # one raw setter per stage), so such a recipe trained the candidate's pool on the chain and
    # promoted the OLD pool on 2a/2b, with the divergence recorded only in `conflicts` -- which
    # nothing in a live run reads. The stage loop above has already applied every pool_edit, so no
    # edit is re-applied here; the share edits still go on AFTER the rescale, which is what makes
    # them survive it (C8 -> 0.05:0.55:0.15:0.25).
    final_raw = overlay_by_stage.get(names[-1], {}).get("KRABBY_OMNI_CMD_POOL") if names else None
    if promote and final_raw and str(final_raw).strip():
        pool = parse_pool(final_raw)
        pool = pool.rescale_to(parse_slots_str(ph.SLOTS_TEACHER))
        for _cid, edit in pool_edits:
            if pool_edit_touches_shares(edit):
                pool = apply_pool_edit(pool, {k: v for k, v in edit.items()
                                              if k in ("set_share", "renormalise")},
                                       require_match=False)
        pool = _canonical_pool(pool)          # amendment 27, on the 2a/2b string as well
        for st in PROMOTION_STAGES:
            overlay_by_stage[st]["KRABBY_OMNI_CMD_POOL"] = render_pool(pool)
            overlay_by_stage[st]["KRABBY_OMNI_SLOTS"] = slots_str(pool_slots(pool))
    for st in (PROMOTION_STAGES if promote else ()):
        _merge_overlay(overlay_by_stage[st], teacher, f"stage {st}", "teacher_overlay", conflicts)
    # An `eval_carry` key decides what the policy SEES or what command it is scored against, so it
    # must be in the 2a/2b env too -- otherwise the promotion's EVAL_FULL rows would score the head
    # through inputs it did not train under (the 0.00-vs-0.39 defect class). A candidate with
    # `stages: all` already carries it; this covers one that sets it per stage.
    final_stage = names[-1] if names else None
    for key in (eval_carry if promote else ()):
        value = overlay_by_stage.get(final_stage, {}).get(key) if final_stage else None
        if value is None:
            continue
        for st in PROMOTION_STAGES:
            if key not in overlay_by_stage[st]:
                overlay_by_stage[st][key] = value

    # ---- the ONE driver-written FACING key: a cap candidate shortens what 8 s of turning can
    # cover, so the teacher facing ramp is clamped to what the cap can actually reach.
    cap = None
    for name in names + list(PROMOTION_STAGES):
        raw = overlay_by_stage[name].get("KRABBY_OMNI_ANG_VEL_MAX")
        if raw is not None:
            cap = float(raw)
    if cap is not None and promote:
        for st in PROMOTION_STAGES:
            preset = ph.phase_env(st).get("KRABBY_OMNI_FACING")
            lo, hi = knobs().parse_facing(overlay_by_stage[st].get("KRABBY_OMNI_FACING") or preset)
            clamp = min(float(hi), 8.0 * cap)
            if clamp < float(hi):
                overlay_by_stage[st]["KRABBY_OMNI_FACING"] = f"{lo:.10g}:{clamp:.10g}"

    recipe = {"chain": chain_name, "adopted": adopted, "overlay_by_stage": overlay_by_stage,
              "eval_carry": eval_carry, "conflicts": conflicts, "anchor": anchor,
              "code_sha": code, "stages": stages,
              "has_pool": any("KRABBY_OMNI_CMD_POOL" in overlay_by_stage[n] for n in names)}
    ident = {"chain": chain_name, "overlay_by_stage": overlay_by_stage}
    # A `resume_from: {checkpoint: <path>}` stage forks a head named in the yaml: the SAME chain and
    # overlays forking another file is another recipe, so the path is part of the identity. Only
    # when such a stage exists -- every recipe without one keeps its recorded id bit for bit.
    forks = {str(s["stage"]): resume_checkpoint(s) for s in stages if resume_checkpoint(s)}
    if forks:
        ident["resume_checkpoints"] = forks
        recipe["resume_checkpoints"] = dict(forks)
    # NOTE(clock-deprecation 2026-09-23): likewise a `resume_from: {reference: <name>}` fork -- the
    # reference's NAME is the identity (its head is resolved at launch); absent, nothing moves.
    refs = {str(s["stage"]): resume_reference(s) for s in stages if resume_reference(s)}
    if refs:
        ident["resume_references"] = refs
        recipe["resume_references"] = dict(refs)
    recipe["recipe_id"] = sha256_text(canonical(ident))[:12]
    return recipe


def _baseline_anchor(ledger: dict, cands: dict, squashed) -> dict | None:
    """The plan's ``else B0``: the baseline's own seed-3 screen anchors the drift check.

    Without this the anchored-drift precondition is INERT for exactly the case it exists for. The
    plan's budget expects several uncertain candidates, and a speculative ADOPT never qualifies as
    an anchor (``_anchor_qualifies``), so a ladder whose adoptions are all speculative would have no
    anchor at all and every later candidate would be judged only against the drifted recipe it is
    screened on -- "a chain of individually-tiny speculative adoptions drifting away from the best
    thing the campaign ever measured", which is what the check forbids.

    The NEWEST seed-3 baseline entry is taken, so after a ``code_sha`` landing the anchor re-resolves
    onto that recipe's ``B<k>`` rerun (decision 9) instead of a screen no longer comparable with
    anything. Entries written before the ``seed`` field existed fall back to write order, where the
    first baseline entry is ``B0`` s3 (``D02``) and the second is ``B0`` s2 (``D04``).
    """
    squashed = set(squashed or ())
    seed3 = str((cands.get("seeds") or {}).get("screen", 3))
    rows = [e for e in entries(ledger)
            if e.get("kind") == "baseline" and not hidden_by_stored_squash(e)
            and e.get("id") not in squashed]
    typed = [e for e in rows if e.get("seed") is not None]
    if typed:
        rows = [e for e in typed if str(e.get("seed")) == seed3]
    else:
        rows = rows[:1]
    if not rows:
        return None
    entry = rows[-1]
    return {"candidate": entry.get("candidate") or (cands.get("baseline") or {}).get("id") or "B0",
            "decision": entry.get("id"), "experiment": entry.get("experiment"),
            "why": "baseline (no adoption qualifies as an anchor yet)"}


def _anchor_qualifies(entry: dict) -> bool:
    """Anchor rule: an ADOPT that was CLEAR at its final level, or that the user settled."""
    if entry.get("synthetic"):
        return False                  # `_recipe_with`'s stand-in: a candidate is never its own anchor
    status = entry.get("status")
    if status == "confirmed":
        return True
    if status == "overruled" and (entry.get("override") or {}).get("by") == "user":
        return lc.effective_choice(entry) == "ADOPT"
    # The Vocabulary's "clear AT ITS FINAL LEVEL (s3 clear, or POOLED clear)". Once a pair has run,
    # the pooled level IS the final level, so it decides on its own -- `confidence` is the s3
    # screen's frozen classification and stays "clear" underneath it. Reading the s3 clause as well
    # made an adoption whose batched pair came back `uncertain` (an all-TIE agreement, or a
    # MISSING-member cap) or `disputed` (a FLOOR-tripped batched s2, which `decide_pooled` forces)
    # the drift anchor for every later candidate -- the one thing the anchor exists to prevent.
    if entry.get("pooled_confidence") is not None:
        return entry["pooled_confidence"] == "clear"
    return status in ("decided", "final") and entry.get("confidence") == "clear"


def _anchor_reason(entry: dict) -> str:
    if entry.get("status") == "confirmed":
        return "user-confirmed"
    # `override.by`, not the status alone: `overruled` also covers a driver self-overrule and a
    # milestone drop, neither of which `_anchor_qualifies` admits -- calling one "user-overruled"
    # in the recipe's `anchor.why` would put a claim about the user in a record nobody made.
    if entry.get("status") == "overruled" and (entry.get("override") or {}).get("by") == "user":
        return "user-overruled to ADOPT"
    if entry.get("pooled_confidence") == "clear":
        return "pooled clear"
    return "s3 clear"


def stage_env(recipe: dict, stage: str, extra: dict | None = None) -> dict:
    """The overlay ``train(extra_env=...)`` receives for one stage -- strings, and audited.

    Refuses a preset, plant, other-task or USD variable: those are the driver's own and an overlay
    that set one would make the run's record lie about what it ran. ``train`` asserts the same thing
    on its side; this is the earlier, louder half so a bad recipe never reaches a GPU window.
    """
    overlay = dict((recipe.get("overlay_by_stage") or {}).get(str(stage)) or {})
    if extra:
        overlay.update(extra)
    bad = [k for k in FORBIDDEN_OVERLAY_KEYS if k in overlay]
    if bad:
        raise ValueError(f"stage {stage}: overlay must not set {bad} (the driver owns them)")
    return {str(k): str(v) for k, v in overlay.items()}


# ================================================================== ledger queries (pure)
def entries(ledger: dict) -> list[dict]:
    return list(ledger.get("entries") or [])


def entry_by_id(ledger: dict, ident: str) -> dict | None:
    return lc.by_id(ledger, ident)


def hidden_by_stored_squash(e: dict) -> bool:
    """Does an entry's STORED ``squashed`` status hide it from the effective ledger by itself?

    A ``derived`` squash is a CACHE of the view, never a fact: the plan makes the view "global,
    self-contained and recomputed after EVERY ledger write" and SQUASH (6) recomputes it "from the
    ledger on every launch". So a stored ``squashed_by: derived`` entry is TRANSPARENT here and only
    the derived set decides -- which is what makes resurrection TRANSITIVE. Reading the stored
    status instead made ``compose_recipe`` inside ``squash_view``'s own fixpoint see the PREVIOUS
    pass's statuses, so reverting an overrule resurrected exactly one dependency level per ingest
    and re-screened intact windows (3.7-3.9 GPU-h each) at every level below it.

    Every OTHER squash is a fact and hides the entry unconditionally: ``squashed_by: user`` (cause
    (e)'s ``decide run``, a ``resume-milestone --drop``) is never resurrected, ``squashed_by:
    driver`` is a milestone's own re-run bookkeeping, and an entry a record hand-carries as
    ``squashed`` with no ``squashed_by`` predates the view.
    """
    return e.get("status") == "squashed" and e.get("squashed_by") != "derived"


def newest_entry(ledger: dict, cid: str, squashed=None) -> dict | None:
    """A candidate's EFFECTIVE entry: its newest entry that is not squashed.

    ``squashed`` is the DERIVED view's entry-id set, passed in rather than read off the entries:
    the view is recomputed after every write and a ``derived`` stored status is only its cache
    (:func:`hidden_by_stored_squash`).
    """
    squashed = set(squashed or ())
    found = None
    for e in entries(ledger):
        if e.get("candidate") != cid:
            continue
        if e.get("id") in squashed or hidden_by_stored_squash(e):
            continue
        found = e
    return found


def newest_milestone_entry(ledger: dict, mid: str, squashed=None) -> dict | None:
    squashed = set(squashed or ())
    found = None
    for e in entries(ledger):
        if e.get("kind") != "milestone" or e.get("candidate") != mid:
            continue
        if e.get("id") in squashed or hidden_by_stored_squash(e):
            continue
        found = e
    return found


def is_terminal(entry: dict | None) -> bool:
    return bool(entry) and entry.get("status") in TERMINAL_STATUSES


def experiments(state: dict) -> dict:
    return ((state.get("ladder") or {}).get("experiments") or {})


def code_aliases(state: dict) -> set[str]:
    """The ``code_sha`` values proved equal to the current training tree (``lad.code_sha_aliases``).

    Written by ``main()``'s ``--accept-eval-change`` path only after :func:`prove_code_alias`
    succeeded (plan D8): the pathspec narrowing moved the VALUE of ``code_sha`` without moving a
    training file, so a header recorded under the old value is still a run of this code.
    """
    lad = state.get("ladder") or {}
    return {str(a.get("sha")) for a in (lad.get("code_sha_aliases") or []) if a.get("sha")}


def same_code(state: dict, recorded, code: str) -> bool:
    """``recorded`` is the current ``code_sha`` or one of its proved aliases."""
    return recorded == code or (recorded is not None and str(recorded) in code_aliases(state))


def ok_runs(state: dict, code: str, view=None) -> list[dict]:
    """Every finished, non-squashed experiment under the CURRENT ``code_sha`` (an "ok run").

    "ok" is the run's own verdict (``status ok``) plus code identity -- the current ``code_sha`` or
    a proved alias of it (``lad.code_sha_aliases``, plan D8), read here so no caller can forget it.
    A screen adds recipe identity, a seed and the current ``eval_sha``; the ``D00c`` lookup needs
    only this.
    """
    squashed = set((view or {}).get("experiments") or ())
    aliases = code_aliases(state)
    out = []
    for xid, h in experiments(state).items():
        if xid in squashed:
            continue
        if h.get("status") != "ok" or not (h.get("code_sha") == code or h.get("code_sha") in aliases):
            continue
        out.append(dict(h, xid=xid))
    return out


#: The experiment kinds that run the RECIPE'S OWN CHAIN, i.e. the only kinds an "ok screen" may be
#: (Vocabulary: "the effective recipe's own seed-3 screen ... 8000 it").
#:
#: Without this filter a ``milestone`` / ``milestone_s2`` promotion satisfies every ok-screen lookup
#: for the recipe it promotes: ``run_experiment`` writes ``status ok`` + ``recipe_id`` + ``screen``
#: for EVERY kind, ``promotion_stage_list`` gives both promotion stages ``eval EVAL_FULL``, and the
#: promotion's ``started`` is later than the 8000-it screen's -- so the newest "ok screen" of that
#: recipe would silently become a **2b row**. Every consumer is affected: the candidate control in
#: ``_dispatch_candidate``, ``recipe_head`` (so ``M<k+1>.promo``'s 2a stage would resume from the
#: previous milestone's 2b checkpoint), the pooled and milestone pairs, the ``D00b``/``D00c``
#: lookups and ``predicate_context``'s ``screen.<key>`` (C10/C11's predicates).
#: ``smoke`` (the ``--smoke-stage`` drill) is never a screen either.
SCREEN_KINDS = ("baseline", "candidate", "twin", "seed2", "batched_s2")


def ok_screen(state: dict, code: str, recipe_id: str, seed, view=None) -> dict | None:
    """The NEWEST ok run of ``recipe_id`` at ``seed`` on the recipe's OWN chain -- an "ok screen".

    Newest, not first: a candidate re-run after an out-of-rank ``decide run`` leaves the older
    screen in place as a record, and the re-run's screen is the one the recipe head resolves to.
    Kind-filtered by :data:`SCREEN_KINDS`: a promotion is not a screen of the recipe it promotes.

    Eval-filtered (plan D8): once the campaign records ``lad.eval_sha``, a header whose screen was
    measured under another eval (``header.eval_sha``, written together with ``header.screen``) is
    never a control or an anchor until it is rescreened -- its numbers are not the ones a current
    screen carries. A pre-suite state (no ``lad.eval_sha`` yet) filters nothing.
    """
    want_eval = (state.get("ladder") or {}).get("eval_sha")
    best = None
    for h in ok_runs(state, code, view):
        if h.get("kind") not in SCREEN_KINDS:
            continue
        if h.get("recipe_id") != recipe_id or str(h.get("seed")) != str(seed):
            continue
        if want_eval and h.get("eval_sha") != want_eval:
            continue
        if best is None or str(h.get("started") or "") >= str(best.get("started") or ""):
            best = h
    return best


def recipe_head(state: dict, code: str, recipe_id: str, seed=3, view=None) -> str | None:
    """A pure lookup: the FINAL-stage checkpoint of the ok s3 screen of ``recipe_id``.

    Used only by milestones. ``None`` when there is no such screen, which is what makes
    ``next_item`` insert a recipe-baseline rerun instead of promoting a head nobody screened.
    """
    scr = ok_screen(state, code, recipe_id, seed, view)
    if not scr:
        return None
    return scr.get("ckpt")



# ================================================================== the overlay dump audit
def _grep_scalar(txt: str, key: str, indent: str = r"\s*") -> str | None:
    m = re.search(rf"^{indent}{re.escape(key)}:\s*(.+?)\s*$", txt, re.M)
    return m.group(1) if m else None


def _grep_tuple_n(txt: str, key: str, n: int) -> tuple | None:
    """``key: !!python/tuple`` followed by ``n`` ``- <number>`` lines, as floats."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*!!python/tuple\s*\n((?:\s*-\s*[-\d.eE+]+\s*\n){{{n}}})", txt, re.M)
    if not m:
        return None
    vals = re.findall(r"-\s*([-\d.eE+]+)", m.group(1))
    try:
        return tuple(float(v) for v in vals[:n])
    except ValueError:
        return None


def _yaml_block(txt: str, key: str, indent: int = 2) -> str:
    """The lines of a mapping entry ``<indent spaces><key>:`` up to the next sibling key.

    Used to read one reward TERM out of a config dump: a bare search for ``weight:`` would find the
    nearest one in the file, which is a different term's.
    """
    pad = " " * indent
    m = re.search(rf"^{pad}{re.escape(key)}:\s*$", txt, re.M)
    if not m:
        return ""
    rest = txt[m.end():]
    stop = re.search(rf"^{pad}\S", rest, re.M)
    return rest[:stop.start()] if stop else rest


def _weight_ok(txt: str, term: str, value) -> bool:
    block = _yaml_block(txt, term)
    got = _grep_scalar(block, "weight", indent=r"\s{4}")
    return got is not None and _num(got) is not None and abs(_num(got) - float(value)) < 1e-9


def _scalar_num_ok(txt: str, key: str, value, indent: str = r"\s*") -> bool:
    got = _grep_scalar(txt, key, indent)
    return got is not None and _num(got) is not None and abs(_num(got) - float(value)) < 1e-9


def _tuple_ok(txt: str, key: str, value, n: int) -> bool:
    got = _grep_tuple_n(txt, key, n)
    want = parse_slots_str(value)
    if got is None or len(want) != n:
        return False
    if n == 4:                       # slot_probs are compared NORMALISED, like the cfg's own rule
        tot_g, tot_w = sum(got), sum(want)
        if tot_g <= 0 or tot_w <= 0:
            return False
        got = tuple(x / tot_g for x in got)
        want = tuple(x / tot_w for x in want)
    return all(abs(a - b) < 1e-6 for a, b in zip(got, want))


def _pool_ok(txt: str, value) -> bool:
    got = _grep_scalar(txt, "cmd_pool")
    if got is None:
        return False
    got = got.strip().strip("'\"")
    if got == str(value).strip():
        return True
    try:                             # a re-rendered equal pool is the same pool
        return render_pool(parse_pool(got)) == render_pool(parse_pool(value))
    except ValueError:
        return False


def _rsi_bank_ok(txt: str, value) -> bool:
    """``KRABBY_RSI_BANK``: the omni reset event's ``bank_path`` param names the same file (absent term -> False)."""
    block = _yaml_block(txt, "rsi_reference_reset")
    got = _grep_scalar(block, "bank_path", indent=r"\s{6}")
    if got is None:
        return False
    import os as _os
    return _os.path.realpath(got.strip().strip("'\"")) == _os.path.realpath(str(value).strip())


def _rsi_source_ok(txt: str, value) -> bool:
    block = _yaml_block(txt, "rsi_reference_reset")
    got = _grep_scalar(block, "source", indent=r"\s{6}")
    return got is not None and got.strip().strip("'\"") == str(value).strip()


def _func_swapped_ok(txt: str, term: str, needle: str) -> bool:
    """The reward TERM's ``func`` is EXACTLY ``needle`` -- not merely a block that contains it.

    ``needle in _yaml_block(...)`` passed vacuously whenever one func name is a prefix of another:
    ``_func_swapped_ok(dump, "track_lin_vel_xy_exp", "track_lin_vel_xy_exp")`` returned True on a
    dump whose func really was ``track_lin_vel_xy_exp_facing_coupled``, and any needle that is a
    prefix of a sibling func would have the audit report a swap that did not happen. The audit is
    the only thing that says a candidate's knob reached the cfg, so it is matched on the func PATH's
    last component (``module:name`` / ``module.name``), never on a substring.
    """
    got = _grep_scalar(_yaml_block(txt, term), "func", indent=r"\s{4}")
    if got is None:
        return False
    return got.strip().strip("'\"").rsplit(":", 1)[-1].rsplit(".", 1)[-1] == str(needle).strip()


# ---- the side-speed-bisection knobs (plan step 1, 2026-09-28). Both DEFAULT OFF.
def _mirror_images_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_MIRROR_IMAGES``: runner side (``params/agent.yaml``).

    ``'fa'`` swaps ``symmetry_cfg.data_augmentation_func`` to the fore-aft-only, two-image
    augmentation; there is no other mode today (:data:`knobs().MIRROR_IMAGES_MODES
    <parkour_tasks.crab_hex_omni_task.config.crab_hex.omni_knobs.MIRROR_IMAGES_MODES>` is
    ``('fa',)``, and the parser raises on anything else), so this only ever checks the swap.
    """
    mode = knobs().parse_mirror_images(value)
    got = _grep_scalar(txt, "data_augmentation_func")
    if got is None:
        return False
    got = got.strip().strip("'\"").rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    want = ("crab_hex_omni_symmetry_augmentation_fa" if mode == "fa"
            else "crab_hex_omni_symmetry_augmentation")
    return got == want


def _cmd_obs_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_CMD_OBS``: env side (``params/env.yaml``).

    ``'scalar'`` means ``cmd_obs_scalar: true`` on the omni obs term; an explicit off spelling
    (:data:`knobs().CMD_OBS_OFF_SPELLINGS
    <parkour_tasks.crab_hex_omni_task.config.crab_hex.omni_knobs.CMD_OBS_OFF_SPELLINGS>`) means
    absent (or ``false``) -- a MODE string, not a plain flag, so this is its own check rather than
    :func:`_flag_dump_ok`.
    """
    mode = knobs().parse_cmd_obs(value)
    got = _grep_scalar(txt, "cmd_obs_scalar")
    return got == "true" if mode == "scalar" else got in (None, "false")


def _check_cmd_obs_mirror_composition(env: dict, where: str, reasons: list) -> None:
    """``KRABBY_OMNI_CMD_OBS=scalar`` needs ``KRABBY_OMNI_MIRROR_IMAGES=fa`` in the SAME composed
    stage env (side-speed-bisection plan step 1b): under the four-image mirror, ``M_y`` and
    ``R_z(pi) = M_x M_y`` map a +y command onto a -y command the scalar encoding cannot see (obs
    slots 8-9 read 0 either way), so the mirror loss would train the -y branch as though every
    command it ever saw were the SAME +y one -- silent, since nothing about it shows in a training
    curve. This is a TRAINING-side rule only, checked here rather than in the env/runner code: an
    eval process carries just ``run_phases.EVAL_CARRY``, which has ``KRABBY_OMNI_CMD_OBS`` but
    (deliberately) no entry for the runner-only ``KRABBY_OMNI_MIRROR_IMAGES`` (the mirror loss
    does not run in an eval), and a refusal living in the env/obs code would stop that eval
    process from building at all.
    """
    obs_raw = env.get(knobs().CMD_OBS_VAR)
    if obs_raw is None:
        return
    try:
        mode = knobs().parse_cmd_obs(obs_raw)
    except Exception:                                                   # noqa: BLE001
        return                            # a bad value is left to _check_knob's own parser check
    if mode != "scalar":
        return
    mirror_raw = env.get(knobs().MIRROR_IMAGES_VAR)
    try:
        mirror_mode = knobs().parse_mirror_images(mirror_raw) if mirror_raw else None
    except Exception:                                                   # noqa: BLE001
        mirror_mode = None
    if mirror_mode != "fa":
        reasons.append(
            f"{where}: {knobs().CMD_OBS_VAR}={obs_raw!r} is set but "
            f"{knobs().MIRROR_IMAGES_VAR}={mirror_raw!r} is not 'fa' in this composed stage -- "
            "the four-image mirror would train a -y command through a scalar observation that "
            "cannot see the sign"
        )


# ---- dump checks for EXISTING knobs the side-speed-bisection arms overlay (plan step 1c,
# 2026-09-28): the applier and parser already exist (R's four knobs were live in the pre-Revision-4
# 1b preset); only the driver-side audit entries were missing.
def _clock_combine_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_CLOCK_COMBINE``: ``combine`` in the ``reward_clock_schedule`` block."""
    mode = knobs().parse_clock_combine(value)
    got = _grep_scalar(_yaml_block(txt, "reward_clock_schedule"), "combine", indent=r"\s{6}")
    return got is not None and got.strip().strip("'\"") == mode


def _track_sigma2_ok(txt: str, value) -> bool:
    """``KRABBY_TRACK_SIGMA2``: the dump stores ``std = sqrt(sigma2)`` on ``track_lin_vel_xy_exp``.

    The TERM key stays ``track_lin_vel_xy_exp`` whatever its ``func`` is (a facing-coupled swap
    keeps the same std param), so the block lookup does not need to know which func is live.
    """
    return _scalar_num_ok(_yaml_block(txt, "track_lin_vel_xy_exp"), "std", math.sqrt(float(value)))


def _track_l1_w_ok(txt: str, value) -> bool:
    """``KRABBY_TRACK_L1_W``: the weight of ``penalty_tracking_error_l1``."""
    return _weight_ok(txt, "penalty_tracking_error_l1", value)


def _rsi_cam_stage_ok(txt: str, value, var: str) -> bool:
    """``KRABBY_RSI_CAM_STAGE`` / ``_DEBUG`` (PPO-construction plan step 4, 2026-10-01; default off):
    the omni RSI reset event's ``stage_cam`` / ``stage_cam_debug`` param. Armed: the event EXISTS and
    carries ``<param>: true`` (staging armed with RSI off has nothing to ride on -- a recipe that
    says something it does not train). Off: the param is absent (or false); the event may be absent.
    """
    param = _motion().RSI_CAM_STAGE_PARAMS[var]
    got = _grep_scalar(_yaml_block(txt, "rsi_reference_reset"), param, indent=r"\s{6}")
    return got == "true" if _motion().parse_flag(value, var) else got in (None, "false")


def _rsi_frac_ok(txt: str, value) -> bool:
    """``KRABBY_RSI_FRAC``: 0 (or unset) means ``events.rsi_reference_reset`` is ABSENT -- the
    forward knob block (``crab_hex_env_cfg.py``) only builds the term for a fraction > 0.0, and
    the omni re-point (``_repoint_rsi``) is a no-op when the term was never built. Nonzero means
    the term's ``fraction`` param carries that value.
    """
    frac = float(value)
    block = _yaml_block(txt, "rsi_reference_reset")
    if frac <= 0.0:
        return block == ""
    return _scalar_num_ok(block, "fraction", frac, indent=r"\s{6}")


# ---- the motion-quality knobs (gait-stabilisation plan step 2, 2026-09-22). Every one is DEFAULT
# OFF and writes NOTHING into the cfg when off, so "off" is audited as ABSENCE of the field/term.
def _motion():
    """``motion_knobs`` (the parsers), reached through the path-loaded ``omni_knobs``."""
    return knobs()._mk


def _flag_dump_ok(txt: str, key: str, value, indent: str = r"\s*") -> bool:
    on = _motion().parse_flag(value)
    got = _grep_scalar(txt, key, indent)
    return got == "true" if on else got in (None, "false")


def _added_term_ok(txt: str, term: str, value) -> bool:
    """A weight knob that ADDS a reward term: absent at 0 (off), else the term at that weight."""
    weight = _motion().parse_penalty_w(value)
    if weight == 0.0:
        return _yaml_block(txt, term) == ""
    return _weight_ok(txt, term, weight)


def _action_rate_sq_ok(txt: str, value) -> bool:
    got = _grep_scalar(_yaml_block(txt, "reward_action_rate"), "squared", indent=r"\s{6}")
    return got == "true" if _motion().parse_flag(value) else got is None


def _lpf_hz_ok(txt: str, value) -> bool:
    hz = _motion().parse_lpf_hz(value)
    got = _grep_scalar(txt, "lpf_hz")
    if hz is None:
        return got is None
    return got is not None and _num(got) is not None and abs(_num(got) - hz) < 1e-9


def _relay_hyst_ok(txt: str, value) -> bool:
    pair = _motion().parse_relay_hyst(value)
    if pair is None:
        return _grep_scalar(txt, "relay_hyst") is None
    got = _grep_tuple_n(txt, "relay_hyst", 2)
    return got is not None and all(abs(a - b) < 1e-9 for a, b in zip(got, pair))


def _agent_optional_ok(txt: str, key: str, parsed) -> bool:
    """An optional runner field: absent while its knob is unset, else equal to the parsed value."""
    got = _grep_scalar(txt, key)
    if parsed is None:
        return got is None
    return got is not None and _num(got) is not None and abs(_num(got) - float(parsed)) < 1e-9


# ---- the PPO-construction knobs (PPO-construction plan step 2, 2026-10-01). All DEFAULT OFF: an
# optional algorithm field is ABSENT while its knob is off; a field the runner cfg always carries
# (schedule, learning_rate, desired_kl, init_noise_std, noise_std_type) is checked for presence
# while off and for the exact value while on. Every read is scoped to its own top-level block.
def _alg_block(txt: str) -> str:
    return _yaml_block(txt, "algorithm", indent=0)


def _alg_optional_ok(txt: str, key: str, parsed) -> bool:
    """:func:`_agent_optional_ok` inside the ``algorithm:`` block (2-space keys)."""
    return _agent_optional_ok(_alg_block(txt), key, parsed)


def _stand_obs_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_STAND_OBS`` (``0`` / ``1`` / ``clock``; p2r 2026-10-09). ``1``: ``stand_obs: true`` and no
    ``stand_bit_clock``. ``clock``: ``stand_bit_clock: true``, NO ``stand_obs``, the eval slot latch armed and
    the clock hidden (``clock_hide: true``). Off: neither param (each absent or false)."""
    mode = _motion().parse_stand_obs(value)
    stand_obs = _grep_scalar(txt, "stand_obs")
    clock_bit = _grep_scalar(txt, "stand_bit_clock")
    if mode is None:
        return stand_obs in (None, "false") and clock_bit in (None, "false")
    if mode == "1":
        return stand_obs == "true" and clock_bit in (None, "false")
    return (clock_bit == "true" and stand_obs in (None, "false")
            and _grep_scalar(txt, "eval_slot_latch") == "true" and _grep_scalar(txt, "clock_hide") == "true")


def _statue_w_ok(txt: str, key: str, value, var: str) -> bool:
    """A p2r statue weight: absent while off (blank / ``0``), else equal in the algorithm block AND
    ``stand_mask_col`` equal to ``motion_knobs.STAND_BIT_OBS_COL`` there."""
    parsed = _motion().parse_statue_w(value, var)
    if not _alg_optional_ok(txt, key, parsed):
        return False
    if parsed is None:
        return True
    col = _num(_grep_scalar(_alg_block(txt), "stand_mask_col"))
    return col is not None and int(col) == int(_motion().STAND_BIT_OBS_COL)


def _stand_proxy_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_STAND_PROXY`` (p2r Z2, 2026-10-10): ``hold`` = ``stand_proxy_hold: true`` on the omni obs
    term with ``stand_bit_clock: true`` beside it (the rows it holds on); off = the param absent or false."""
    mode = _motion().parse_stand_proxy(value)
    got = _grep_scalar(txt, "stand_proxy_hold")
    if mode is None:
        return got in (None, "false")
    return got == "true" and _grep_scalar(txt, "stand_bit_clock") == "true"


def _restance_dump(txt: str) -> dict | None:
    """The ``algorithm.restance`` mapping of an agent dump as ``{name: float}`` (None when absent / null)."""
    alg = _alg_block(txt)
    inline = _grep_scalar(alg, "restance", indent=r"\s{2}")
    if inline is not None and inline.strip() in ("null", "None", "{}"):
        return None
    block = _yaml_block(alg, "restance", indent=2)
    if not block.strip():
        return None
    out = {}
    for name, raw in re.findall(r"^\s{4}(\w+):\s*(\S+)\s*$", block, re.M):
        num = _num(raw)
        if num is None:
            return {}
        out[name] = float(num)
    return out


def _stand_prior_target_ok(txt: str, value) -> bool:
    """``KRABBY_STAND_PRIOR_TARGET`` (p2r Z2): off = ``algorithm.stand_prior_target`` absent (or null); armed =
    equal to the parsed target, with ``algorithm.restance`` present beside it (the knobs travel together)."""
    want = _motion().parse_stand_prior_target(value)
    got = _grep_scalar(_alg_block(txt), "stand_prior_target", indent=r"\s{2}")
    got = None if got is None or got.strip().strip("'\"") in ("null", "None") else got.strip().strip("'\"")
    if want is None:
        return got is None
    return got == want and bool(_restance_dump(txt))


def _restance_ok(txt: str, value) -> bool:
    """``KRABBY_RESTANCE`` (p2r Z2): off = no ``algorithm.restance``; armed = the dumped mapping equals
    ``parse_restance(value).to_dict()`` key for key (every constant, no extra ones) to 1e-12."""
    params = _motion().parse_restance_knob(value)
    got = _restance_dump(txt)
    if params is None:
        return got is None
    want = params.to_dict()
    return (got is not None and set(got) == set(want)
            and all(abs(got[k] - float(want[k])) <= 1e-12 for k in want))


def _anchor_ckpt_ok(txt: str, value) -> bool:
    """``KRABBY_ANCHOR_CKPT``: absent while blank, else the algorithm block's ``anchor_ckpt`` names the same
    file (compared as real paths)."""
    want = _motion().parse_anchor_ckpt(value)
    got = _grep_scalar(_alg_block(txt), "anchor_ckpt")
    if want is None:
        return got is None or got.strip().strip("'\"") in ("null", "None")
    if got is None:
        return False
    return os.path.realpath(got.strip().strip("'\"")) == os.path.realpath(_motion().resolve_anchor_ckpt(want))


def _present_or_equal(block: str, key: str, parsed) -> bool:
    """A field the cfg always carries: present while the knob is off, else equal to its value."""
    got = _grep_scalar(block, key, indent=r"\s{2}")
    if parsed is None:
        return got is not None
    return got is not None and _num(got) is not None and abs(_num(got) - float(parsed)) < 1e-12


def _grep_list_n(txt: str, key: str, n: int) -> tuple | None:
    """``key:`` followed by ``n`` ``- <number>`` lines (a plain YAML list), as floats."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*\n((?:\s*-\s*[-\d.eE+]+\s*\n){{{n}}})", txt, re.M)
    if not m:
        return None
    try:
        return tuple(float(v) for v in re.findall(r"-\s*([-\d.eE+]+)", m.group(1))[:n])
    except ValueError:
        return None


def _bound_ramp_ok(txt: str, value) -> bool:
    """``KRABBY_ACTION_BOUND_RAMP``: ``action_bound_loss_ramp: [c0, c1, n]`` in the algorithm block."""
    ramp = _motion().parse_bound_ramp(value)
    block = _alg_block(txt)
    if ramp is None:
        return _grep_scalar(block, "action_bound_loss_ramp") is None
    got = _grep_list_n(block, "action_bound_loss_ramp", 3)
    return got is not None and all(abs(a - float(b)) < 1e-12 for a, b in zip(got, ramp))


def _std_param_ok(txt: str, value) -> bool:
    """``KRABBY_STD_PARAM``: ``noise_std_type: soft`` in the policy block; off = anything but soft."""
    got = _grep_scalar(_yaml_block(txt, "policy", indent=0), "noise_std_type", indent=r"\s{2}")
    if got is None:
        return False
    got = got.strip().strip("'\"")
    return got == "soft" if _motion().parse_std_param(value) == "soft" else got != "soft"


def _ppo_schedule_ok(txt: str, value) -> bool:
    """``KRABBY_PPO_SCHEDULE``: the algorithm block's ``schedule`` (always present)."""
    want = _motion().parse_ppo_schedule(value)
    got = _grep_scalar(_alg_block(txt), "schedule", indent=r"\s{2}")
    if got is None:
        return False
    return want is None or got.strip().strip("'\"") == want


def _num_envs_ok(txt: str, value) -> bool:
    """``KRABBY_NUM_ENVS``: ``scene.num_envs`` in params/env.yaml (the terrain block's own
    ``num_envs`` sits deeper and is not read); blank = the driver's stock count, any value."""
    want = _motion().parse_num_envs(value)
    got = _grep_scalar(_yaml_block(txt, "scene", indent=0), "num_envs", indent=r"\s{2}")
    if got is None or _num(got) is None:
        return False
    return want is None or int(_num(got)) == want


# ---- the clock-deprecation knobs (clock-deprecation plan step 4, 2026-09-23). The two clock reward
# weights are the forward cfg's presence-based overrides; KRABBY_PHASEOUT is the forward
# reward-weight curriculum (`curriculums.py`, one `phaseout_<term>` curriculum block per entry);
# the two clock-HIDE knobs are `clock_knobs.py`'s (default off: nothing in the cfg while off).
#: The two clock reward terms and the knob that sets each one's static weight.
CLOCK_TERM_KNOBS = {"reward_clock_schedule": "KRABBY_CLOCK_W", "reward_clock_swing_apex": "KRABBY_APEX_W"}
#: The removal weight (settled decision 13): 0.001, never 0.0 -- a term at exactly 0.0 is skipped by
#: the reward manager and its Episode_Reward telemetry flatlines (``curriculums.EPS_WEIGHT``).
CLOCK_EPS_W = 0.001
#: Every overlay key the clock rules (i)-(v) read; a chain or candidate that sets none of them is
#: never looked at by those rules (every earlier campaign validates exactly as before).
CLOCK_KNOB_KEYS = ("KRABBY_CLOCK_W", "KRABBY_APEX_W", "KRABBY_PHASEOUT", "KRABBY_OMNI_CLOCK_HIDE",
                   "KRABBY_OMNI_CLOCK_HIDE_RAMP")
_CURRICULUMS = None


def curriculums():
    """The forward task's ``mdp/curriculums.py`` (``parse_phaseout_spec``, ``ramp_value``), by path.

    Pure at module scope: its Isaac-Lab manager term is defined only when ``isaaclab`` imports, so a
    path-load here gets the parser and the ramp and nothing else.
    """
    global _CURRICULUMS
    if _CURRICULUMS is None:
        _CURRICULUMS = _load(rp.FWD_PKG / "mdp/curriculums.py", "crab_hex_curriculums_ladder")
    return _CURRICULUMS


def _clock():
    """``clock_knobs`` (the hide / ramp parsers), reached through the path-loaded ``omni_knobs``."""
    return knobs()._ck


def parse_phaseout_knob(text):
    """The driver's round trip for ``KRABBY_PHASEOUT``: ``None`` for the BLANK value, else the spec.

    NOTE(omni phase-2 plan step 4, 2026-10-06): an overlay may set ``KRABBY_PHASEOUT: ""`` to blank
    a preset's ramps -- the phase-2 arms run on the 2a preset, whose PHASEOUT re-arms THREE ramps
    (apex 1.0 -> 0.5, airtime 0.8 -> 0.4, stride 0.5 -> 0.25) that the side-first B recipe never
    trained with. ``activate_phase`` is ``setdefault``, so the overlay's empty string wins, and the
    forward knob block reads it with ``if _phaseout:`` (``crab_hex_env_cfg.py``): the empty string
    leaves ``cfg.curriculum = None``. Only the EXACT empty string is blank: whitespace is truthy
    there, reaches ``parse_phaseout_spec`` and kills the build ("parsed to no entries"), so it is
    refused here by name instead of at Kit start. Anything else parses exactly as before.
    """
    raw = str(text)
    if raw == "":
        return None
    if not raw.strip():
        raise ValueError(f"KRABBY_PHASEOUT={raw!r} is whitespace: the env cfg treats it as SET and its "
                         "parser refuses it at config build -- use the empty string to blank the "
                         "preset's ramps")
    return curriculums().parse_phaseout_spec(raw)


def _phaseout_ok(txt: str, value) -> bool:
    """``KRABBY_PHASEOUT``: the dump's ``curriculum`` section holds EXACTLY the ``phaseout_<term>``
    blocks the spec names -- no missing and no extra one -- each with that entry's
    ``term_name``/``w0``/``w1``/``t0``/``t1``.

    Exact, not "contains": a stray ``phaseout_*`` block (a preset's ramp the overlay was meant to
    replace) anneals a term the recipe does not claim to, and the dump is the only place it shows.

    BLANK (``""``, :func:`parse_phaseout_knob`; omni phase-2 plan step 4, 2026-10-06): the
    top-level ``curriculum: null`` -- the curriculum manager was never built -- AND no
    ``phaseout_<term>`` block anywhere in the dump. Pinned, not "no check": a blank that failed
    to reach the cfg leaves the 2a preset's three ramps in place, which is exactly the recipe the
    blank exists to remove.
    """
    spec = parse_phaseout_knob(value)
    if spec is None:
        return (_grep_scalar(txt, "curriculum", indent="") == "null"
                and re.search(r"^\s*phaseout_\w+:\s*$", txt, re.M) is None)
    want = {e["term_name"]: e for e in spec}
    section = _yaml_block(txt, "curriculum", indent=0)
    got = re.findall(r"^  (phaseout_\w+):\s*$", section, re.M)
    if sorted(got) != sorted(f"phaseout_{term}" for term in want):
        return False
    for term, entry in want.items():
        params = _yaml_block(section, f"phaseout_{term}", indent=2)
        name = _grep_scalar(params, "term_name", indent=r"\s{6}")
        if name is None or name.strip().strip("'\"") != term:
            return False
        for key in ("w0", "w1", "t0", "t1"):
            got_v = _num(_grep_scalar(params, key, indent=r"\s{6}"))
            if got_v is None or abs(got_v - float(entry[key])) > 1e-9:
                return False
    return True


def _clock_hide_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_CLOCK_HIDE``, a scalar: ``clock_hide: true`` on, ABSENT (or false) off."""
    got = _grep_scalar(txt, _clock().CLOCK_HIDE_PARAM)
    return got == "true" if _clock().parse_clock_hide(value) else got in (None, "false")


def _clock_hide_ramp_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_CLOCK_HIDE_RAMP``: the ``clock_hide_ramp`` tuple (``_tuple_ok``); blank = absent."""
    if _clock().parse_clock_hide_ramp(value) is None:
        return _grep_scalar(txt, _clock().CLOCK_HIDE_RAMP_PARAM) is None
    return _tuple_ok(txt, _clock().CLOCK_HIDE_RAMP_PARAM, value, 2)


# ---- the stroke/posture knobs (stride/posture plan step 3, 2026-09-25). All three DEFAULT OFF.
def _stroke():
    """``stroke_knobs`` (the parsers), reached through the path-loaded ``omni_knobs``."""
    return knobs()._sk


def _stroke_w_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_STROKE_W``: off keeps the stock ``RewardStrideLength`` func; on swaps the
    ``reward_stride_length`` term's func to ``RewardFootStroke`` at the knob's weight, TOGETHER."""
    w = _stroke().parse_stroke_w(value)
    if w is None:
        return _func_swapped_ok(txt, _stroke().STROKE_TERM, "RewardStrideLength")
    return (_func_swapped_ok(txt, _stroke().STROKE_TERM, "RewardFootStroke")
            and _weight_ok(txt, _stroke().STROKE_TERM, w))


def _centring_w_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_CENTRING_W``: a weight knob that ADDS a term; absent at unset, else present."""
    w = _stroke().parse_centring_w(value)
    if w is None:
        return _yaml_block(txt, _stroke().CENTRING_TERM) == ""
    return _weight_ok(txt, _stroke().CENTRING_TERM, w)


def _reward_split_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_REWARD_SPLIT``: ``reward_category_split`` on the ENV cfg (top-level, no indent)."""
    on = _stroke().parse_reward_split(value)
    got = _grep_scalar(txt, "reward_category_split", indent="")
    return got == "true" if on else got in (None, "false")


def _dict_entries(txt: str, key: str, indent: int) -> dict[str, str] | None:
    """The raw ``{sub_key: value}`` strings of a ``<indent spaces><key>:`` mapping block (one level
    deeper, ``indent + 2``), or ``None`` if ``key`` is absent at that indent. See ``_yaml_block``'s
    own docstring for why a bare search would find the wrong term's field."""
    block = _yaml_block(txt, key, indent=indent)
    if not block:
        return None
    pad = " " * (indent + 2)
    return {m.group(1): m.group(2)
            for m in re.finditer(rf"^{pad}(\w+):\s*(.+?)\s*$", block, re.M)}


def _stroke_cat_w_ok(txt: str, value) -> bool:
    """``KRABBY_OMNI_STROKE_CAT_W`` (per-category stroke plan, 2026-09-27): off has no ``cat_scale``
    key under ``reward_stride_length.params``; on, the term's func is ``RewardFootStroke`` (the SAME
    ``KRABBY_OMNI_STROKE_W`` swap) and ``params.cat_scale`` carries EXACTLY the six parsed-and-
    defaulted keys -- no ``stand`` -- each within 1e-9 of the knob's own table.

    ``params`` keys sit at indent 6 and a param's own dict entries at indent 8 in the real dump (a
    ``RewardTermCfg``'s ``params:`` block, itself at indent 4 under the term): read a ``params/
    env.yaml`` with ``RewardFootStroke`` armed to see the shape before changing this.
    """
    table = _stroke().parse_stroke_cat_w(value)
    term_block = _yaml_block(txt, _stroke().STROKE_TERM)  # indent=2: the whole term (func/params/weight)
    entries = _dict_entries(term_block, "cat_scale", indent=6)
    if table is None:
        return entries is None
    if entries is None or "stand" in entries or set(entries) != set(_stroke().STROKE_CAT_KEYS):
        return False
    if not _func_swapped_ok(txt, _stroke().STROKE_TERM, "RewardFootStroke"):
        return False
    for key, want in table.items():
        got = _num(entries.get(key))
        if got is None or abs(got - float(want)) > 1e-9:
            return False
    return True


def _check_rsi_cam_stage_composition(env: dict, where: str, reasons: list) -> None:
    """``KRABBY_RSI_CAM_STAGE`` / ``_DEBUG`` need a non-zero ``KRABBY_RSI_FRAC`` in the SAME composed
    stage env (PPO-construction plan step 4, 2026-10-01): the staging rides on the RSI reset event,
    which the forward knob block only builds for a fraction > 0. Config build tolerates it (a direct
    rung-0b FRAC-0 row runs that way on purpose) and the dump audit would only catch it after the
    window has trained; a ladder recipe that arms it on an RSI-off stage is refused here, at
    ``--dry-run``. A malformed value is left to ``_check_knob``'s parser check.
    """
    mk = _motion()
    armed = []
    for var in mk.RSI_CAM_STAGE_PARAMS:
        try:
            if mk.parse_flag(env.get(var), var):
                armed.append(var)
        except ValueError:
            continue
    if not armed:
        return
    try:
        frac = float(str(env.get("KRABBY_RSI_FRAC") or "0").strip() or "0")
    except ValueError:
        return
    if frac <= 0.0:
        reasons.append(f"{where}: {', '.join(armed)} armed but KRABBY_RSI_FRAC is unset or 0 in this "
                       "composed stage -- there is no RSI reset event to stage the cam velocity on")


def _check_stroke_cat_w_composition(env: dict, where: str, reasons: list) -> None:
    """``KRABBY_OMNI_STROKE_CAT_W`` needs a non-zero ``KRABBY_OMNI_STROKE_W`` in the SAME composed
    stage env (per-category stroke plan, 2026-09-27): the applier itself refuses this at config
    build (``stroke_knobs._apply_stroke_cat_w_knob``), but only once Kit starts -- catching it here,
    at ``--dry-run``, is the point. A malformed value is left to ``_check_knob``'s own parser check
    at the stage that actually sets it; this rule only looks at whether the two knobs compose.
    """
    cat_raw = env.get(_stroke().STROKE_CAT_W_VAR)
    if cat_raw is None or not str(cat_raw).strip():
        return
    try:
        w = _stroke().parse_stroke_w(env.get(_stroke().STROKE_W_VAR))
    except Exception:                                                   # noqa: BLE001
        return
    if w is None or w == 0.0:
        reasons.append(f"{where}: {_stroke().STROKE_CAT_W_VAR}={cat_raw!r} is set but "
                       f"{_stroke().STROKE_W_VAR} is unset, blank or 0.0 in this composed stage -- "
                       "there is no RewardFootStroke term to scale")


# ---- the omni phase-2 knobs (omni phase-2 plan step 4, 2026-10-06). The field geometry
# (``KRABBY_FLAT_TERRAIN_GEOM=field`` + ``KRABBY_FIELD_*``, mdp/omni_terrain_knobs.py), the promotion
# pair, the reused stride weight and the F1 / E1 terms (motion_knobs.py). Every check below PINS the
# dumped value: a field knob is read off EVERY field sub-terrain (at least one must exist, so a ring
# dump never passes a field check), the geometry off the sub-terrains' dumped ``function``, and the
# off spellings as the ABSENCE / default the applier leaves.
def _terrain():
    """``mdp/omni_terrain_knobs`` (the geometry / field parsers), reached through ``omni_knobs``."""
    return knobs()._tk


#: The geometry selector and the promotion pair -- forward-cfg knobs (``apply_flat_walk_knobs`` /
#: ``crab_hex_env_cfg.py``) the omni presets set in every window (``OMNI_COMMON``).
TERRAIN_GEOM_VAR = "KRABBY_FLAT_TERRAIN_GEOM"
TERRAIN_PROMOTE_VAR = "KRABBY_TERRAIN_PROMOTE"
#: The dumped ``function`` leaf of each omni sub-terrain kind (``parkour_isaaclab/terrains/omni``).
OMNI_SUB_TERRAIN_KINDS = {"omni_flat_terrain": "flat", "omni_ring_terrain": "ring",
                          "omni_field_terrain": "field"}
#: The two generators' obstacle sub-terrain KEYS, in generator order (``config/omni_ring.py`` /
#: ``config/omni_field.py``; ``tests/unit/test_omni_phase2_tooling.py`` pins them to those files).
RING_SUB_TERRAIN_KEYS = ("omni_ring_sparse", "omni_ring_dense")
FIELD_SUB_TERRAIN_KEYS = ("omni_field_sparse", "omni_field_dense")
#: ``OmniFieldTerrainCfg``'s three size RANGES (``omni_terrains_cfg.py``; pinned by the same test):
#: what ``KRABBY_FIELD_SIZE_SCALE`` multiplies.
FIELD_SIZE_RANGES = {"bar_length_range": (1.0, 2.0), "bar_thickness_range": (0.16, 0.32),
                     "slab_side_range": (0.6, 1.2)}
#: The forward cfg module that defines ``_CRAB_ACTUATED_JOINT_NAMES`` (``KRABBY_POWER_JOINTS=actuated``).
_PARKOUR_MDP_CFG = rp.FWD_PKG / "config/crab_hex/agents/parkour_mdp_cfg.py"


def parse_terrain_geom(text) -> str:
    """``KRABBY_FLAT_TERRAIN_GEOM`` -> ``ring`` / ``field`` (read as ``strip().lower()``, like the
    knob block); anything else is refused -- ``require_omni_geom`` raises on it at config build."""
    value = str(text).strip().lower()
    if value not in _terrain().OMNI_GEOMS:
        raise ValueError(f"{TERRAIN_GEOM_VAR} must be one of {_terrain().OMNI_GEOMS}, got {text!r}")
    return value


def parse_terrain_promote(text) -> tuple[float, float]:
    """``KRABBY_TERRAIN_PROMOTE`` -> ``(move_up_frac, move_down_frac)``.

    The cfg does ``up, down = (float(x) for x in text.split(":"))`` (``crab_hex_env_cfg.py``), so
    exactly two numbers or the build dies; on top of that both must be finite and non-negative and
    ``down < up`` -- an inverted pair would demote an env on the very ratio that promotes it.
    """
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f'{TERRAIN_PROMOTE_VAR} must be "up:down", got {text!r}')
    try:
        up, down = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"{TERRAIN_PROMOTE_VAR} entries must be numbers, got {text!r}") from exc
    if not all(math.isfinite(v) and v >= 0.0 for v in (up, down)):
        raise ValueError(f"{TERRAIN_PROMOTE_VAR} entries must be finite and >= 0, got {text!r}")
    if not down < up:
        raise ValueError(f"{TERRAIN_PROMOTE_VAR} needs down < up, got {text!r}")
    return up, down


_DUMP_LOADER = None
_DUMP_CACHE: dict = {}


def _dump_loader():
    """A ``yaml.SafeLoader`` that reads the trainer's python tags as PLAIN data.

    ``params/env.yaml`` is written by IsaacLab's ``dump_yaml`` (``yaml.dump`` on the cfg dict), so it
    carries ``!!python/tuple`` / ``!!python/object/apply:builtins.slice`` tags and ANCHORS for shared
    objects (``&id001`` ... ``*id001`` -- real omni teacher dumps have them). A regex reads an alias as
    the literal ``*id001``; this loader resolves it and turns every python-tagged node into the plain
    list / dict / string it holds, constructing nothing. Used by the terrain and joint-list checks
    below, whose values are tuples and lists two sub-terrains may share.
    """
    global _DUMP_LOADER
    if _DUMP_LOADER is None:
        import yaml  # the harness's own dependency

        class Loader(yaml.SafeLoader):
            pass

        def _plain(loader, _suffix, node):
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node, deep=True)
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node, deep=True)
            return loader.construct_scalar(node)

        Loader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)
        _DUMP_LOADER = Loader
    return _DUMP_LOADER


def _dump_doc(txt: str) -> dict:
    """``txt`` loaded with :func:`_dump_loader` (``{}`` when it is not a YAML mapping). Memoised on the
    text: one window's audit runs a dozen checks on the same dump."""
    key = hashlib.sha256(txt.encode()).hexdigest()
    if key not in _DUMP_CACHE:
        import yaml  # noqa: PLC0415
        try:
            doc = yaml.load(txt, Loader=_dump_loader())                 # noqa: S506 -- plain-data loader
        except yaml.YAMLError:
            doc = None
        if len(_DUMP_CACHE) > 8:
            _DUMP_CACHE.clear()
        _DUMP_CACHE[key] = doc if isinstance(doc, dict) else {}
    return _DUMP_CACHE[key]


def _dig(doc, *keys):
    for key in keys:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def _same(got, want) -> bool:
    """``got`` (a dumped number) equals ``want`` within 1e-9; a bool / missing value never does."""
    g = _num(got)
    return g is not None and abs(g - float(want)) <= 1e-9


def _sub_terrains(txt: str) -> dict[str, tuple[str | None, dict]]:
    """``{key: (kind, cfg)}`` of ``scene.terrain.terrain_generator.sub_terrains``, in DUMP order.

    ``kind`` is ``flat`` / ``ring`` / ``field`` from the sub-terrain's dumped ``function`` leaf
    (:data:`OMNI_SUB_TERRAIN_KINDS`), ``None`` for any other function. Empty without a generator.
    """
    subs = _dig(_dump_doc(txt), "scene", "terrain", "terrain_generator", "sub_terrains")
    out = {}
    for key, cfg in (subs.items() if isinstance(subs, dict) else ()):
        cfg = cfg if isinstance(cfg, dict) else {}
        func = cfg.get("function")
        leaf = str(func).strip().rsplit(":", 1)[-1].rsplit(".", 1)[-1] if func else None
        out[str(key)] = (OMNI_SUB_TERRAIN_KINDS.get(leaf), cfg)
    return out


def dump_terrain_geom(txt: str) -> str | None:
    """The omni geometry a dump's generator IS: ``ring`` (flat + ring sub-terrains, at least one
    ring), ``field`` (flat + field, at least one field), else ``None`` (no generator, an unknown
    sub-terrain function, or ring and field mixed -- neither guard would have built that)."""
    kinds = [kind for kind, _c in _sub_terrains(txt).values()]
    if not kinds or None in kinds:
        return None
    obstacle = set(kinds) - {"flat"}
    return obstacle.pop() if len(obstacle) == 1 else None


def _field_cfgs(txt: str) -> list[dict]:
    """The FIELD sub-terrains' cfgs, in generator order (empty on a ring dump)."""
    return [cfg for kind, cfg in _sub_terrains(txt).values() if kind == "field"]


def _parkour_cfg(txt: str) -> dict:
    """``parkours.base_parkour`` (``{}`` when absent)."""
    got = _dig(_dump_doc(txt), "parkours", "base_parkour")
    return got if isinstance(got, dict) else {}


def _terrain_geom_ok(txt: str, value) -> bool:
    """``KRABBY_FLAT_TERRAIN_GEOM``: :func:`dump_terrain_geom` is the knob's value, AND every omni
    sub-terrain and the parkour term carry ONE pad radius and ONE outer radius (the sync both
    knob branches perform -- the field's flat tiles share the field's 1.7 / 4.80 so the sector
    anchors agree; ``reach_edge`` / ``field_frac`` are measured on the term's copy)."""
    want = parse_terrain_geom(value)
    if dump_terrain_geom(txt) != want:
        return False
    cfgs = [cfg for _kind, cfg in _sub_terrains(txt).values()]
    parkour = _parkour_cfg(txt)
    for term_attr, sub_attr in (("clear_radius_m", "clear_radius"), ("ring_outer_radius_m", "ring_outer_radius")):
        vals = {_num(cfg.get(sub_attr)) for cfg in cfgs}
        if None in vals or len(vals) != 1 or not _same(parkour.get(term_attr), vals.pop()):
            return False
    return True


def _field_all(txt: str, ok) -> bool:
    """``ok(cfg)`` on EVERY field sub-terrain; False when there is none (a ring dump never passes)."""
    cfgs = _field_cfgs(txt)
    return bool(cfgs) and all(ok(cfg) for cfg in cfgs)


def _field_pad_r_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_PAD_R``: ``clear_radius`` on EVERY sub-terrain (field AND flat -- the flat tiles
    carry the pad so their sector anchors match) and the parkour term's ``clear_radius_m``, all equal
    to the knob; the dump must be a field dump."""
    pad = _terrain().parse_field_pad_r(value)
    if dump_terrain_geom(txt) != "field":
        return False
    return (all(_same(cfg.get("clear_radius"), pad) for _kind, cfg in _sub_terrains(txt).values())
            and _same(_parkour_cfg(txt).get("clear_radius_m"), pad))


def _field_blocks_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_BLOCKS`` ``sparse:dense``: EXACTLY two field sub-terrains, whose ``n_blocks``
    are the pair in generator order (the applier's ``zip``), as integers."""
    pair = _terrain().parse_field_blocks(value)
    cfgs = _field_cfgs(txt)
    return len(cfgs) == 2 and all(
        isinstance(cfg.get("n_blocks"), int) and not isinstance(cfg.get("n_blocks"), bool)
        and cfg["n_blocks"] == int(want) for cfg, want in zip(cfgs, pair))


def _field_kinds_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_KINDS`` ``bar:slab:pit``: ``bar_fraction`` and ``pit_fraction`` on every field
    sub-terrain (slab is the remainder and has no field of its own)."""
    bar, _slab, pit = _terrain().parse_field_kinds(value)
    return _field_all(txt, lambda cfg: _same(cfg.get("bar_fraction"), bar) and _same(cfg.get("pit_fraction"), pit))


def _field_edges_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_EDGES`` ``ramp|wall``: ``vertical_edges: false|true`` on every field sub-terrain."""
    wall = _terrain().parse_field_edges(value)
    return _field_all(txt, lambda cfg: cfg.get("vertical_edges") is wall)


def _field_min_gap_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_MIN_GAP``: ``min_gap`` on every field sub-terrain."""
    gap = _terrain().parse_field_min_gap(value)
    return _field_all(txt, lambda cfg: _same(cfg.get("min_gap"), gap))


def _field_size_scale_ok(txt: str, value) -> bool:
    """``KRABBY_FIELD_SIZE_SCALE``: each of the three size ranges on every field sub-terrain is the
    cfg default (:data:`FIELD_SIZE_RANGES`) times the knob."""
    scale = _terrain().parse_field_size_scale(value)

    def ok(cfg):
        for attr, (lo, hi) in FIELD_SIZE_RANGES.items():
            got = cfg.get(attr)
            if not (isinstance(got, list) and len(got) == 2 and _same(got[0], lo * scale)
                    and _same(got[1], hi * scale)):
                return False
        return True
    return _field_all(txt, ok)


def _terrain_promote_ok(txt: str, value) -> bool:
    """``KRABBY_TERRAIN_PROMOTE`` ``up:down``: the parkour term's ``move_up_frac`` / ``move_down_frac``."""
    up, down = parse_terrain_promote(value)
    parkour = _parkour_cfg(txt)
    return _same(parkour.get("move_up_frac"), up) and _same(parkour.get("move_down_frac"), down)


def parse_terrain_diff(value) -> tuple[float, float]:
    """``KRABBY_FLAT_TERRAIN_DIFF`` ``lo:hi`` -> ``(lo, hi)``, read exactly as the env cfg reads it
    (``float`` of each ``:`` part); refuses anything but two finite values with ``0 <= lo <= hi``."""
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f'KRABBY_FLAT_TERRAIN_DIFF must be "lo:hi", got {value!r}')
    lo, hi = (float(p) for p in parts)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi):
        raise ValueError(f"KRABBY_FLAT_TERRAIN_DIFF needs finite 0 <= lo <= hi, got {value!r}")
    return lo, hi


def _terrain_diff_ok(txt: str, value) -> bool:
    """``KRABBY_FLAT_TERRAIN_DIFF`` ``lo:hi``: the terrain generator's ``difficulty_range`` (omni phase 2 walls
    re-tune, 2026-10-06: the arms widen it past the 2a preset's 0.20:0.70)."""
    lo, hi = parse_terrain_diff(value)
    rng = _dig(_dump_doc(txt), "scene", "terrain", "terrain_generator", "difficulty_range")
    return isinstance(rng, (list, tuple)) and len(rng) == 2 and _same(rng[0], lo) and _same(rng[1], hi)


def _stand_stance_w_ok(txt: str, value) -> bool:
    """``KRABBY_STAND_STANCE_W`` (F1, an income): 0 / blank = the term ABSENT; else
    ``reward_stand_stance_quality`` present with EXACTLY that func and weight."""
    w = _motion().parse_income_w(value, _motion().STAND_STANCE_W_VAR)
    term = _motion().STAND_STANCE_TERM
    if w == 0.0:
        return _yaml_block(txt, term) == ""
    return _func_swapped_ok(txt, term, term) and _weight_ok(txt, term, w)


def foot_body_names() -> list[str]:
    """``_CRAB_FOOT_BODY_NAMES`` read from the forward cfg's SOURCE (the module itself needs Isaac)."""
    src = _PARKOUR_MDP_CFG.read_text()
    m = re.search(r"^_CRAB_FOOT_BODY_NAMES\s*=\s*\[(.*?)\]", src, re.M | re.S)
    if not m:
        raise RuntimeError(f"_CRAB_FOOT_BODY_NAMES not found in {_PARKOUR_MDP_CFG}")
    return re.findall(r"[\"']([^\"']+)[\"']", m.group(1))


def _stand_support_w_ok(txt: str, value) -> bool:
    """``KRABBY_STAND_SUPPORT_W`` (p2s T1, an income; P2S_PLAN step 3, 2026-10-08): 0 / blank = the
    term ABSENT; else ``reward_stand_support`` present with EXACTLY the motion-rewards func, that
    weight and the builder's params (the six footpads on ``contact_forces`` / ``robot``, the
    ``base_velocity`` slot, 50 N support saturation, 0.9 upright line -- and nothing else)."""
    mk = _motion()
    w = mk.parse_income_w(value, mk.STAND_SUPPORT_W_VAR)
    term = mk.STAND_SUPPORT_TERM
    if w == 0.0:
        return _yaml_block(txt, term) == ""
    doc = _dig(_dump_doc(txt), "rewards", term)
    params = _dig(doc, "params")
    if not isinstance(params, dict) or set(params) != {"sensor_cfg", "asset_cfg", "command_name", "load_ref_n",
                                                       "upright_min"}:
        return False
    feet = foot_body_names()
    return (str(doc.get("func", "")).strip() == f"{mk._REWARDS_MOD}:{term}" and _weight_ok(txt, term, w)
            and _dig(params, "sensor_cfg", "name") == "contact_forces"
            and _dig(params, "sensor_cfg", "body_names") == feet
            and _dig(params, "asset_cfg", "name") == "robot" and _dig(params, "asset_cfg", "body_names") == feet
            and params["command_name"] == "base_velocity"
            and _same(params["load_ref_n"], mk.STAND_LOAD_REF_N)
            and _same(params["upright_min"], mk.STAND_UPRIGHT_MIN))


def actuated_joint_names() -> list[str]:
    """``_CRAB_ACTUATED_JOINT_NAMES`` read from the forward cfg's SOURCE (the module itself needs Isaac)."""
    src = _PARKOUR_MDP_CFG.read_text()
    m = re.search(r"^_CRAB_ACTUATED_JOINT_NAMES\s*=\s*\[(.*?)\]", src, re.M | re.S)
    if not m:
        raise RuntimeError(f"_CRAB_ACTUATED_JOINT_NAMES not found in {_PARKOUR_MDP_CFG}")
    return re.findall(r"[\"']([^\"']+)[\"']", m.group(1))


def _power_joints_ok(txt: str, value) -> bool:
    """``KRABBY_POWER_JOINTS``: off (blank / ``all``) = ``penalty_mechanical_power``'s
    ``asset_cfg.joint_names: null`` (every joint, the stock sum); ``actuated`` = EXACTLY the forward
    cfg's ``_CRAB_ACTUATED_JOINT_NAMES`` list, on a term whose weight is non-zero (the applier refuses
    a subset it would not price)."""
    mode = _motion().parse_power_joints(value)
    term = _dig(_dump_doc(txt), "rewards", _motion().POWER_TERM)
    asset = _dig(term, "params", "asset_cfg")
    if not isinstance(asset, dict) or "joint_names" not in asset:
        return False
    if mode is None:
        return asset["joint_names"] is None
    weight = _num(term.get("weight"))
    return asset["joint_names"] == actuated_joint_names() and weight is not None and weight != 0.0


# ---- omni phase-2 stage p2b knobs (P2B_PLAN step 1, 2026-10-08). The two new flags live in
# motion_knobs (``ORIENT_ALL_TILES_VAR`` / ``FOOT_CLEAR_TERRAIN_VAR``); named by literal here so the
# audit does not depend on the import order, and a unit test pins the literals to the module's.
ORIENT_ALL_TILES_KEY = "KRABBY_ORIENT_ALL_TILES"
FOOT_CLEAR_TERRAIN_KEY = "KRABBY_FOOT_CLEAR_TERRAIN"
FOOT_CLEAR_FLAT_KEY = "KRABBY_FOOT_CLEAR_FLAT"
#: The two lift terms (CL arm) and, per term, the terrain-relative func FOOT_CLEAR_TERRAIN swaps in
#: (``crab_hex_omni_task/mdp/omni_clearance.py``).
CLEARANCE_TERRAIN_FUNCS = {"reward_foot_clearance": "reward_foot_clearance_terrain",
                           "penalty_swing_min_clearance": "penalty_swing_min_clearance_terrain"}


def _forward_weight_parser(var: str, sign: int):
    """A parser for a forward ``_overrides`` reward weight (``float(raw)`` in ``parkour_mdp_cfg``).

    Blank is REFUSED, not read as 0: the cfg's ``float("")`` raises at config build, i.e. only once
    Kit has started. ``sign`` (+1 income / -1 penalty) refuses a weight that would pay for what the
    term exists to suppress (or the reverse); 0 is always allowed (the term's default, inert).
    """
    def parse(text) -> float:
        value = float(str(text).strip())
        if not math.isfinite(value):
            raise ValueError(f"{var} must be finite, got {text!r}")
        if value * sign < 0.0:
            raise ValueError(f"{var} is {'an income' if sign > 0 else 'a penalty'} weight and must be "
                             f"{'>=' if sign > 0 else '<='} 0, got {text!r}")
        return value
    return parse


def parse_foot_clear_min(text) -> float:
    """``KRABBY_FOOT_CLEAR_MIN``: the income floor ``min_clearance_m`` (``float(raw)`` in the forward
    cfg; blank refused like :func:`_forward_weight_parser`), finite and >= 0."""
    value = float(str(text).strip())
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"KRABBY_FOOT_CLEAR_MIN must be finite and >= 0, got {text!r}")
    return value


def _foot_clear_min_ok(txt: str, value) -> bool:
    """``KRABBY_FOOT_CLEAR_MIN``: ``reward_foot_clearance``'s ``params.min_clearance_m`` (the income
    floor only; the swing-min penalty keeps its own 0.04 m floor)."""
    got = _dig(_dump_doc(txt), "rewards", "reward_foot_clearance", "params", "min_clearance_m")
    return _same(got, parse_foot_clear_min(value))


def _foot_clear_flat_ok(txt: str, value) -> bool:
    """``KRABBY_FOOT_CLEAR_FLAT``: ``1`` = ``params.parkour_name: null`` on BOTH clearance terms (the
    forward cfg lifts the parkour_flat mask, ``parkour_mdp_cfg.py``); off = a named parkour on both
    (the stock ``base_parkour``). The key must be present either way: an absent one is not a mask.
    The flag is read as the recipe's INTENT (``parse_flag``), so a value the cfg's ``== "1"`` would
    not take fails here rather than training the masked recipe under an unmasked name.
    """
    on = _motion().parse_flag(value, FOOT_CLEAR_FLAT_KEY)
    rewards = _dig(_dump_doc(txt), "rewards")
    for term in CLEARANCE_TERRAIN_FUNCS:
        params = _dig(rewards, term, "params")
        if not isinstance(params, dict) or "parkour_name" not in params:
            return False
        name = params["parkour_name"]
        if (name is not None) if on else not (isinstance(name, str) and name.strip()):
            return False
    return True


def _foot_clear_terrain_ok(txt: str, value) -> bool:
    """``KRABBY_FOOT_CLEAR_TERRAIN``: on = BOTH clearance terms' func is the terrain-relative variant;
    off = both keep the stock func (an on dump fails the off check, and the reverse)."""
    on = _motion().parse_flag(value, FOOT_CLEAR_TERRAIN_KEY)
    return all(_func_swapped_ok(txt, term, swapped if on else term)
               for term, swapped in CLEARANCE_TERRAIN_FUNCS.items())


def _orient_all_tiles_ok(txt: str, value) -> bool:
    """``KRABBY_ORIENT_ALL_TILES``: on = ``reward_orientation``'s func is ``reward_orientation_upright``
    (all tiles); off = the stock, obstacle-tile-zeroed ``reward_orientation``."""
    on = _motion().parse_flag(value, ORIENT_ALL_TILES_KEY)
    return _func_swapped_ok(txt, "reward_orientation",
                            "reward_orientation_upright" if on else "reward_orientation")


def _check_restance_composition(env: dict, where: str, reasons: list) -> None:
    """The p2r Z2 cross rules (``motion_knobs.check_restance_composition``, as a TRAINING env) on a COMPOSED
    stage env, at ``--dry-run`` (2026-10-10): the held proxy without the clock bit, the re-stance target without
    its weight / constants, or ``KRABBY_RESTANCE`` without the target would otherwise raise only at config build,
    once Kit has started (NOTE(p2r Z2 cams-only, 2026-10-10): the target no longer needs the held proxy -- R reads
    cams + hip-pitch positions). Inert while none of the three knobs is set; a malformed value is left to
    ``_check_knob``."""
    mk = _motion()
    names = (mk.STAND_PROXY_VAR, *mk.RESTANCE_KNOBS)
    if not any(str(env.get(var) or "").strip() for var in names):
        return
    for var in (*names, mk.STAND_OBS_VAR, mk.STAND_PRIOR_W_VAR):
        try:
            mk.MOTION_KNOB_PARSERS[var](env.get(var))
        except ValueError:
            return                        # _check_knob names the malformed value; one reason, not two
    try:
        mk.check_restance_composition(env, training=True)
    except ValueError as exc:
        reasons.append(f"{where}: {exc}")


def _check_field_geom_composition(env: dict, where: str, reasons: list) -> None:
    """The geometry guard (``require_omni_geom``) on a COMPOSED stage env, at ``--dry-run`` (omni
    phase-2 plan step 4, 2026-10-06): ``KRABBY_FIELD_*`` under ``ring`` (or ``KRABBY_RING_*`` under
    ``field``) would apply nothing and raises at config build, i.e. only once Kit has started. Inert
    unless the composed env names ``field`` or carries a ``KRABBY_FIELD_*`` key, so every ring
    campaign validates exactly as before; a malformed geometry is left to ``_check_knob``.
    """
    tk = _terrain()
    if str(env.get(TERRAIN_GEOM_VAR) or "").strip().lower() != tk.FIELD_GEOM and not any(
            str(env.get(var) or "").strip() for var in tk.FIELD_VARS):
        return
    try:
        tk.require_omni_geom(env)
    except RuntimeError as exc:
        reasons.append(f"{where}: {exc}")


#: Overlay keys audited against ``params/agent.yaml`` (the RUNNER cfg) instead of ``env.yaml``.
AGENT_DUMP_KEYS = frozenset({"KRABBY_ACTION_STD_MAX", "KRABBY_ACTION_STD_RESET", "KRABBY_ENTROPY_COEF",
                            # side-speed-bisection plan step 1a (2026-09-28): the symmetry loss's
                            # data_augmentation_func lives on the runner cfg.
                            "KRABBY_OMNI_MIRROR_IMAGES",
                            # PPO-construction plan step 2 (2026-10-01): every runner-side knob of
                            # motion_knobs.PPO_CONSTRUCTION_KNOBS (KRABBY_NUM_ENVS is env-side).
                            "KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND",
                            "KRABBY_STD_PARAM", "KRABBY_INIT_STD", "KRABBY_PPO_SCHEDULE", "KRABBY_PPO_LR",
                            "KRABBY_PPO_DESIRED_KL", "KRABBY_PPO_KL_STOP", "KRABBY_PPO_HIST_LR",
                            "KRABBY_PPO_HAND_DIAG",
                            # omni phase-2 stage p2r (P2R_STATUE_PLAN step 1, 2026-10-09): the statue
                            # distillation's three runner-side knobs (motion_knobs.STATUE_KNOBS).
                            "KRABBY_ANCHOR_W", "KRABBY_ANCHOR_CKPT", "KRABBY_STAND_PRIOR_W",
                            # p2r Z2 (P2R_Z2_PLAN step 1, 2026-10-10): the re-stance prior target and R's
                            # constants (motion_knobs.RESTANCE_KNOBS); KRABBY_OMNI_STAND_PROXY is env side.
                            "KRABBY_STAND_PRIOR_TARGET", "KRABBY_RESTANCE"})


#: ``KRABBY_*`` -> what the config DUMP must show for it. Every overlay key a recipe may carry needs
#: an entry: the validator refuses a candidate whose key has none, because an unaudited overlay key
#: is exactly the silent failure ``overlay_checks`` exists to catch (a knob that never reached the
#: cfg looks identical in a training log to one that did).
OVERLAY_DUMP_CHECKS = {
    "KRABBY_OMNI_CMD_POOL": lambda txt, v: (_grep_scalar(txt, "cmd_pool") == "null"
                                            if not str(v).strip() or str(v).strip().lower() in ("none", "off", "null")
                                            else _pool_ok(txt, v)),
    "KRABBY_OMNI_SLOTS": lambda txt, v: _tuple_ok(txt, "slot_probs", v, 4),
    "KRABBY_OMNI_DIR_BAND": lambda txt, v: (_grep_scalar(txt, "dir_band") == "null"
                                            if not str(v).strip() or str(v).strip().lower() in ("none", "off", "null")
                                            else _tuple_ok(txt, "dir_band", v, 2)),
    "KRABBY_OMNI_FACING": lambda txt, v: _tuple_ok(txt, "facing_delta", v, 2),
    "KRABBY_OMNI_SPEED_ELLIPSE": lambda txt, v: _tuple_ok(txt, "speed_ellipse", v, 3),
    "KRABBY_OMNI_YAW_RANGE": lambda txt, v: _tuple_ok(txt, "yaw_range", v, 2),
    "KRABBY_OMNI_RSI_SOURCE": _rsi_source_ok,
    "KRABBY_OMNI_ANG_VEL_MAX": lambda txt, v: _scalar_num_ok(txt, "ang_vel_max", v),
    "KRABBY_OMNI_ANG_VEL_CLIP": lambda txt, v: _scalar_num_ok(txt, "ang_vel_clip", v),
    "KRABBY_OMNI_CLOCK_TURN_R": lambda txt, v: _scalar_num_ok(txt, "omni_clock_turn_radius", v, indent=""),
    "KRABBY_OMNI_FACING_OBS_CLIP": lambda txt, v: _scalar_num_ok(txt, "facing_obs_clip", v),
    "KRABBY_OMNI_FACING_OBS_ENCODING": lambda txt, v: _grep_scalar(txt, "facing_obs_encoding") == str(v).strip(),
    "KRABBY_YAW_W": lambda txt, v: _weight_ok(txt, "reward_tracking_yaw", v),
    "KRABBY_YAW_PROGRESS_W": lambda txt, v: _weight_ok(txt, "reward_yaw_progress", v),
    "KRABBY_OMNI_FACING_TURN_W": lambda txt, v: _weight_ok(txt, "reward_facing_turn", v),
    "KRABBY_OMNI_PROGRESS_W": lambda txt, v: _weight_ok(txt, "reward_forward_progress_along_command", v),
    "KRABBY_OMNI_TRACK_FACING_SIGMA": lambda txt, v: (
        _func_swapped_ok(txt, "track_lin_vel_xy_exp", "track_lin_vel_xy_exp_facing_coupled")
        and _scalar_num_ok(_yaml_block(txt, "track_lin_vel_xy_exp"), "facing_sigma", v)),
    "KRABBY_OMNI_ANG_TURN_ONLY": lambda txt, v: (_func_swapped_ok(txt, "track_ang_vel_z_exp",
                                                                  "track_ang_vel_z_exp_turn_gated")
                                                 if str(v).strip() == "1" else True),
    "KRABBY_OMNI_ANG_VEL_STD": lambda txt, v: _scalar_num_ok(_yaml_block(txt, "track_ang_vel_z_exp"), "std", v),
    # ---- motion-quality knobs (step 2). Env side: params/env.yaml.
    "KRABBY_POWER_W": lambda txt, v: _weight_ok(txt, "penalty_mechanical_power", v),
    "KRABBY_ACTION_RATE_W": lambda txt, v: _weight_ok(txt, "reward_action_rate", v),
    "KRABBY_AIRTIME_W": lambda txt, v: _weight_ok(txt, "reward_feet_air_time_positive", v),
    "KRABBY_ACTION_RATE_SQ": _action_rate_sq_ok,
    "KRABBY_ACTION_SMOOTH2_W": lambda txt, v: _added_term_ok(txt, "penalty_action_smooth2", v),
    "KRABBY_JOINT_VEL_W": lambda txt, v: _added_term_ok(txt, "penalty_joint_vel", v),
    "KRABBY_FOOT_SLIP_W": lambda txt, v: _added_term_ok(txt, "penalty_foot_slip", v),
    "KRABBY_STAND_POSE_W": lambda txt, v: _added_term_ok(txt, "penalty_stand_pose", v),
    "KRABBY_ACTION_LPF_HZ": _lpf_hz_ok,
    "KRABBY_ACTION_LPF_OBS": lambda txt, v: _flag_dump_ok(txt, "lpf_obs", v),
    "KRABBY_OMNI_RELAY_HYST": _relay_hyst_ok,
    "KRABBY_OMNI_STAND_HOLD": lambda txt, v: _flag_dump_ok(txt, "stand_hold", v),
    "KRABBY_OMNI_STAND_PARK": lambda txt, v: _flag_dump_ok(txt, "omni_stand_park", v, indent=""),
    "KRABBY_OMNI_STAND_OBS": lambda txt, v: _stand_obs_ok(txt, v),
    # Runner side (AGENT_DUMP_KEYS): params/agent.yaml.
    "KRABBY_ACTION_STD_MAX": lambda txt, v: _agent_optional_ok(txt, "action_std_max",
                                                               _motion().parse_std_max(v)),
    "KRABBY_ACTION_STD_RESET": lambda txt, v: _agent_optional_ok(txt, "action_std_reset",
                                                                 _motion().parse_std_reset(v)),
    "KRABBY_ENTROPY_COEF": lambda txt, v: (_agent_optional_ok(txt, "entropy_coef", _motion().parse_entropy_coef(v))
                                           if _motion().parse_entropy_coef(v) is not None
                                           else _grep_scalar(txt, "entropy_coef") is not None),
    # ---- clock-deprecation knobs (plan step 4). The dump shows the STATIC reward weight; a
    # PHASEOUT stage's effective end weight is its curriculum block's w1 (removal verification).
    "KRABBY_CLOCK_W": lambda txt, v: _weight_ok(txt, "reward_clock_schedule", v),
    "KRABBY_APEX_W": lambda txt, v: _weight_ok(txt, "reward_clock_swing_apex", v),
    "KRABBY_PHASEOUT": _phaseout_ok,
    "KRABBY_OMNI_CLOCK_HIDE": _clock_hide_ok,
    "KRABBY_OMNI_CLOCK_HIDE_RAMP": _clock_hide_ramp_ok,
    # ---- stroke/posture knobs (stride/posture plan step 3, 2026-09-25).
    "KRABBY_OMNI_STROKE_W": _stroke_w_ok,
    "KRABBY_OMNI_CENTRING_W": _centring_w_ok,
    "KRABBY_OMNI_REWARD_SPLIT": _reward_split_ok,
    # ---- per-category stroke knob (2026-09-27).
    "KRABBY_OMNI_STROKE_CAT_W": _stroke_cat_w_ok,
    # ---- side-speed-bisection knobs (plan step 1, 2026-09-28). Both default off.
    "KRABBY_OMNI_MIRROR_IMAGES": _mirror_images_ok,
    "KRABBY_OMNI_CMD_OBS": _cmd_obs_ok,
    # ---- dump checks for EXISTING knobs the arms overlay (plan step 1c, 2026-09-28).
    "KRABBY_OMNI_CLOCK_COMBINE": _clock_combine_ok,
    "KRABBY_TRACK_SIGMA2": _track_sigma2_ok,
    "KRABBY_TRACK_L1_W": _track_l1_w_ok,
    "KRABBY_RSI_FRAC": _rsi_frac_ok,
    # ---- side-bisect reversal test (2026-09-29): "lo:hi" seconds -> commands.base_velocity.resampling_time_range
    # (crab_hex_env_cfg.py; the only resampling_time_range in an omni env.yaml).
    "KRABBY_RESAMPLE_S": lambda txt, v: _tuple_ok(txt, "resampling_time_range", v, 2),
    # ---- side-bisect short-episode test (2026-09-30): the top-level episode_length_s (crab_hex_env_cfg.py PLAN F).
    "KRABBY_EPISODE_S": lambda txt, v: _scalar_num_ok(txt, "episode_length_s", v, indent=""),
    # ---- side-bisect two-direction RSI bank (2026-09-30): the reset event's bank_path (compared as real paths).
    "KRABBY_RSI_BANK": _rsi_bank_ok,
    # ---- PPO-construction knobs (plan step 2, 2026-10-01). Runner side (AGENT_DUMP_KEYS):
    # params/agent.yaml, every field read inside its own top-level block (``learning_rate`` exists
    # in BOTH algorithm and estimator). KRABBY_NUM_ENVS is env side: params/env.yaml's scene block.
    "KRABBY_ACTION_BOUND_LOSS": lambda txt, v: _alg_optional_ok(txt, "action_bound_loss_coef",
                                                                _motion().parse_bound_loss(v)),
    "KRABBY_ACTION_BOUND_RAMP": _bound_ramp_ok,
    "KRABBY_ACTION_BOUND": lambda txt, v: _alg_optional_ok(txt, "action_bound", _motion().parse_action_bound(v)),
    "KRABBY_STD_PARAM": _std_param_ok,
    "KRABBY_INIT_STD": lambda txt, v: _present_or_equal(_yaml_block(txt, "policy", indent=0), "init_noise_std",
                                                        _motion().parse_init_std(v)),
    "KRABBY_PPO_SCHEDULE": _ppo_schedule_ok,
    "KRABBY_PPO_LR": lambda txt, v: _present_or_equal(_yaml_block(txt, "algorithm", indent=0), "learning_rate",
                                                      _motion().parse_positive(v, "KRABBY_PPO_LR")),
    "KRABBY_PPO_DESIRED_KL": lambda txt, v: _present_or_equal(_yaml_block(txt, "algorithm", indent=0),
                                                              "desired_kl",
                                                              _motion().parse_positive(v, "KRABBY_PPO_DESIRED_KL")),
    "KRABBY_PPO_KL_STOP": lambda txt, v: _alg_optional_ok(txt, "kl_stop",
                                                          _motion().parse_positive(v, "KRABBY_PPO_KL_STOP")),
    "KRABBY_PPO_HIST_LR": lambda txt, v: _alg_optional_ok(txt, "hist_encoder_learning_rate",
                                                          _motion().parse_positive(v, "KRABBY_PPO_HIST_LR")),
    "KRABBY_PPO_HAND_DIAG": lambda txt, v: _alg_optional_ok(
        txt, "hand_diag_vy_obs_col",
        _motion().HAND_DIAG_VY_OBS_COL if _motion().parse_flag(v, "KRABBY_PPO_HAND_DIAG") else None),
    "KRABBY_NUM_ENVS": _num_envs_ok,
    # ---- RSI cam staging (PPO-construction plan step 4, 2026-10-01). Env side: params/env.yaml's
    # events.rsi_reference_reset params, so every C_lr_rsi run (C0 included) proves its staging state.
    "KRABBY_RSI_CAM_STAGE": lambda txt, v: _rsi_cam_stage_ok(txt, v, "KRABBY_RSI_CAM_STAGE"),
    "KRABBY_RSI_CAM_STAGE_DEBUG": lambda txt, v: _rsi_cam_stage_ok(txt, v, "KRABBY_RSI_CAM_STAGE_DEBUG"),
    # ---- omni phase-2 knobs (omni phase-2 plan step 4, 2026-10-06). Env side: params/env.yaml.
    # The geometry and the field knobs (scene.terrain.terrain_generator.sub_terrains + the parkour
    # term's radii), the promotion pair (parkours.base_parkour), the reused stride weight, and the F1
    # / E1 terms. KRABBY_AIRTIME_W / KRABBY_POWER_W were already audited (motion-quality block above).
    TERRAIN_GEOM_VAR: _terrain_geom_ok,
    "KRABBY_FIELD_PAD_R": _field_pad_r_ok,
    "KRABBY_FIELD_BLOCKS": _field_blocks_ok,
    "KRABBY_FIELD_KINDS": _field_kinds_ok,
    "KRABBY_FIELD_EDGES": _field_edges_ok,
    "KRABBY_FIELD_MIN_GAP": _field_min_gap_ok,
    "KRABBY_FIELD_SIZE_SCALE": _field_size_scale_ok,
    TERRAIN_PROMOTE_VAR: _terrain_promote_ok,
    "KRABBY_FLAT_TERRAIN_DIFF": _terrain_diff_ok,
    "KRABBY_STRIDE_W": lambda txt, v: _weight_ok(txt, "reward_stride_length", v),
    "KRABBY_STAND_STANCE_W": _stand_stance_w_ok,
    "KRABBY_POWER_JOINTS": _power_joints_ok,
    # ---- omni phase-2 stage p2b (P2B_PLAN step 1, 2026-10-08). Env side: params/env.yaml. The CL
    # arm's four reused forward knobs + the terrain-relative swap, the TG arm's func swap + weight.
    "KRABBY_FOOT_CLEAR_W": lambda txt, v: _weight_ok(txt, "reward_foot_clearance", v),
    "KRABBY_SWING_MIN_CLEAR_W": lambda txt, v: _weight_ok(txt, "penalty_swing_min_clearance", v),
    "KRABBY_FOOT_CLEAR_MIN": _foot_clear_min_ok,
    FOOT_CLEAR_FLAT_KEY: _foot_clear_flat_ok,
    FOOT_CLEAR_TERRAIN_KEY: _foot_clear_terrain_ok,
    "KRABBY_ORIENT_W": lambda txt, v: _weight_ok(txt, "reward_orientation", v),
    ORIENT_ALL_TILES_KEY: _orient_all_tiles_ok,
    # ---- omni phase-2 stage p2s (P2S_PLAN step 3, 2026-10-08). Env side: params/env.yaml. The two
    # stand-slot terms: T1 audited like F1 plus its params, T2 like the other added penalties.
    "KRABBY_STAND_SUPPORT_W": _stand_support_w_ok,
    "KRABBY_STAND_SLIDE_W": lambda txt, v: _added_term_ok(txt, "penalty_stand_slide", v),
    # ---- omni phase-2 stage p2r (P2R_STATUE_PLAN step 1, 2026-10-09). Runner side (AGENT_DUMP_KEYS):
    # params/agent.yaml's algorithm block; KRABBY_OMNI_STAND_OBS=clock is audited above (_stand_obs_ok).
    "KRABBY_ANCHOR_W": lambda txt, v: _statue_w_ok(txt, "anchor_coef", v, "KRABBY_ANCHOR_W"),
    "KRABBY_STAND_PRIOR_W": lambda txt, v: _statue_w_ok(txt, "stand_prior_coef", v, "KRABBY_STAND_PRIOR_W"),
    "KRABBY_ANCHOR_CKPT": lambda txt, v: _anchor_ckpt_ok(txt, v),
    # ---- p2r Z2 re-stance (P2R_Z2_PLAN step 1, 2026-10-10). Env side: the held proxy on the omni obs term;
    # runner side (AGENT_DUMP_KEYS): the prior's target and R's constants in the algorithm block.
    "KRABBY_OMNI_STAND_PROXY": lambda txt, v: _stand_proxy_ok(txt, v),
    "KRABBY_STAND_PRIOR_TARGET": lambda txt, v: _stand_prior_target_ok(txt, v),
    "KRABBY_RESTANCE": lambda txt, v: _restance_ok(txt, v),
}


def overlay_checks(rec: dict, env: dict) -> list[tuple[str, bool]]:
    """``[(name, ok)]`` -- did every overlay key actually reach the config the window trained under?

    The overlay is the campaign's only description of a candidate, and ``activate_phase`` is
    ``setdefault``: a key that a preset already owns, a typo'd value, or an applier that quietly
    skipped a term all produce a run whose log looks perfect and whose recipe is a different one.
    This reads the dump the trainer wrote (``params/env.yaml``) and compares it with the overlay,
    key by key. A FAIL halts the campaign (``halt:<xid> config_mismatch``) rather than deciding.
    The runner-side keys (:data:`AGENT_DUMP_KEYS` -- the exploration knobs) are read from the
    RUNNER dump, ``params/agent.yaml``, instead: that is where ``train.py`` writes the policy's std
    clamp / reset and PPO's entropy coefficient. An overlay that arms a PPO-construction knob also
    gets :func:`construction_log_checks` against the window's TRAINING LOG (2026-10-01): the dump is
    the cfg, the log is what the algorithm actually consumed.
    """
    ckpt = rec.get("ckpt")
    if not ckpt:
        return [("overlay audit needs a checkpoint", False)]
    params_dir = Path(ckpt).parent / "params"
    envy = params_dir / "env.yaml"
    if not envy.exists():
        return [(f"params/env.yaml exists ({envy})", False)]
    txts = {"env.yaml": envy.read_text(errors="ignore")}
    if any(k in AGENT_DUMP_KEYS for k in (env or {})):
        agenty = params_dir / "agent.yaml"
        if not agenty.exists():
            return [(f"params/agent.yaml exists ({agenty})", False)]
        txts["agent.yaml"] = agenty.read_text(errors="ignore")
    out = []
    for key in sorted(env or {}):
        value = env[key]
        dump = "agent.yaml" if key in AGENT_DUMP_KEYS else "env.yaml"
        check = OVERLAY_DUMP_CHECKS.get(key)
        if check is None:
            out.append((f"overlay {key} has a dump check (none registered)", False))
            continue
        try:
            ok = bool(check(txts[dump], value))
        except Exception as exc:                                        # noqa: BLE001
            out.append((f"overlay {key}={value} audited ({exc!r})", False))
            continue
        out.append((f"overlay {key}={value} in params/{dump}", ok))
    out += construction_log_checks(rec, env)
    out += statue_log_checks(rec, env)
    return out


#: The ``PPOWithExtractor`` constructor fields the PPO-construction knobs add: a training log whose
#: "Ignoring unsupported PPO config keys" warning names one of them ran a PPO that DROPPED the knob
#: (a stale checkout's runner -- the M1b import trap), whatever the cfg dump says.
CONSTRUCTION_ALG_FIELDS = ("action_bound_loss_coef", "action_bound_loss_ramp", "action_bound", "kl_stop",
                           "hist_encoder_learning_rate", "hand_diag_vy_obs_col")


def construction_log_checks(rec: dict, env: dict) -> list[tuple[str, bool]]:
    """The TRAINING LOG half of the PPO-construction audit (plan step 2, 2026-10-01; M1b spec).

    Empty unless the overlay ARMS a PPO-construction knob (``motion_knobs.PPO_CONSTRUCTION_KNOBS``;
    off spellings arm nothing). Then the window's training log (``rec["log"]``) must show, for every
    armed knob, its exact ``[KNOB] <NAME>=<value> consumed`` line (written by ``apply_exploration_knobs`` -- or, for
    ``KRABBY_NUM_ENVS``, by ``run_phases.train``); NO "Ignoring unsupported PPO config keys" line naming
    a construction field; and, with the bound loss armed, ``Mean action_bound:`` (the loss ran).
    The dump audit above cannot see any of that: ``params/agent.yaml`` is the cfg, not the algorithm.
    """
    mk = _motion()
    armed = []
    for key in getattr(mk, "PPO_CONSTRUCTION_KNOBS", ()):
        if key not in (env or {}):
            continue
        try:
            parsed = mk.MOTION_KNOB_PARSERS[key](str(env[key]))
        except Exception:                                               # noqa: BLE001
            continue                      # a bad value is the parser check's to report, not this one's
        if parsed not in (None, False):
            armed.append(key)
    if not armed:
        return []
    log = rec.get("log")
    if not log or not Path(log).exists():
        return [(f"training log readable for the PPO-construction wiring audit ({log})", False)]
    txt = Path(log).read_text(errors="ignore")
    # the exact consumed line (never "[KNOB] <NAME>=<v> NOT consumed: ..." -- an armed knob nothing used)
    out = [(f"log: [KNOB] {key} consumed",
            re.search(rf"^\[KNOB\] {re.escape(key)}=\S+ consumed\s*$", txt, re.M) is not None)
           for key in armed]
    dropped = [ln for ln in txt.splitlines() if "Ignoring unsupported PPO config keys" in ln
               and any(re.search(rf"\b{f}\b", ln) for f in CONSTRUCTION_ALG_FIELDS)]
    out.append(("log: no PPO-construction field dropped by the runner", not dropped))
    if mk.ACTION_BOUND_LOSS_VAR in armed:
        out.append(("log: Mean action_bound printed (the bound loss ran)",
                    re.search(r"^\s*Mean action_bound:", txt, re.M) is not None))
    return out


#: The ``PPOWithExtractor`` fields the p2r statue knobs write (``algorithm.*``), and the loss the runner
#: prints (``Mean <key>:``) for each armed weight.
STATUE_ALG_FIELDS = ("anchor_coef", "anchor_ckpt", "stand_prior_coef", "stand_mask_col",
                     # p2r Z2 (2026-10-10): the re-stance prior target and R's constants
                     "stand_prior_target", "restance")
STATUE_LOSS_KEYS = {"KRABBY_ANCHOR_W": "anchor", "KRABBY_STAND_PRIOR_W": "stand_prior",
                    # p2r Z2: the share of bit-1 rows whose R target is non-zero (logged with the prior)
                    "KRABBY_STAND_PRIOR_TARGET": "restance_nonzero"}


def statue_log_checks(rec: dict, env: dict) -> list[tuple[str, bool]]:
    """The TRAINING LOG half of the p2r statue audit (P2R_STATUE_PLAN step 1, 2026-10-09).

    Empty unless the overlay arms a statue knob (``motion_knobs.STATUE_KNOBS``, or since p2r Z2 its
    ``RESTANCE_KNOBS`` / ``KRABBY_OMNI_STAND_PROXY``). Then: each armed knob's ``[KNOB] <NAME>=<value> consumed``
    line, no runner warning dropping a statue field, and ``Mean anchor:`` / ``Mean stand_prior:`` printed for each
    armed weight (the loss ran). With the re-stance target armed (p2r Z2, 2026-10-10) also the algorithm's own
    ``[PPO] construction:`` line naming ``stand_prior_target=restance`` and ``Mean restance_nonzero:`` (R was
    evaluated on the stand rows). ``KRABBY_RESTANCE`` prints its consumed line only alongside the target.
    """
    mk = _motion()
    armed = []
    for key in (*getattr(mk, "STATUE_KNOBS", ()), *getattr(mk, "RESTANCE_KNOBS", ()),
                *((mk.STAND_PROXY_VAR,) if hasattr(mk, "STAND_PROXY_VAR") else ())):
        if key not in (env or {}):
            continue
        try:
            parsed = mk.MOTION_KNOB_PARSERS[key](str(env[key]))
        except Exception:                                               # noqa: BLE001
            continue
        if parsed is not None:
            armed.append(key)
    if not armed:
        return []
    log = rec.get("log")
    if not log or not Path(log).exists():
        return [(f"training log readable for the statue wiring audit ({log})", False)]
    txt = Path(log).read_text(errors="ignore")
    # (KRABBY_RESTANCE's value is a k=v list that may carry spaces: any non-empty value there)
    value_re = {key: (r".+?" if key == "KRABBY_RESTANCE" else r"\S+") for key in armed}
    out = [(f"log: [KNOB] {key} consumed",
            re.search(rf"^\[KNOB\] {re.escape(key)}={value_re[key]} consumed\s*$", txt, re.M) is not None)
           for key in armed]
    dropped = [ln for ln in txt.splitlines() if "Ignoring unsupported PPO config keys" in ln
               and any(re.search(rf"\b{f}\b", ln) for f in STATUE_ALG_FIELDS)]
    out.append(("log: no statue field dropped by the runner", not dropped))
    for key in armed:
        loss_key = STATUE_LOSS_KEYS.get(key)
        if loss_key is not None:
            out.append((f"log: Mean {loss_key} printed (the {key} loss ran)",
                        re.search(rf"^\s*Mean {loss_key}:", txt, re.M) is not None))
    if getattr(mk, "STAND_PRIOR_TARGET_VAR", None) in armed:
        target = mk.parse_stand_prior_target(str(env[mk.STAND_PRIOR_TARGET_VAR]))
        out.append((f"log: [PPO] construction names stand_prior_target={target}",
                    re.search(rf"^\[PPO\] construction:.*\bstand_prior_target={re.escape(str(target))}\b", txt, re.M)
                    is not None))
    return out


# ================================================================== the validator
_KNOB_PARSERS_CACHE = None


def knob_parsers() -> dict:
    """``{KRABBY_*: parse_fn}`` from ``omni_knobs``: every overlay value round-trips through one.

    Built from the module's own ``*_VAR`` constants rather than a list here, so a knob that gains a
    parser gains the check for free and one that is renamed breaks loudly.
    """
    global _KNOB_PARSERS_CACHE
    if _KNOB_PARSERS_CACHE is not None:
        return _KNOB_PARSERS_CACHE
    k = knobs()
    table = {
        k.SLOTS_VAR: k.parse_slots, k.DIR_BAND_VAR: k.parse_dir_band,
        k.SPEED_ELLIPSE_VAR: k.parse_speed_ellipse, k.FACING_VAR: k.parse_facing,
        k.ANG_VEL_MAX_VAR: k.parse_ang_vel_max, k.ANG_VEL_CLIP_VAR: k.parse_ang_vel_clip,
        k.CMD_POOL_VAR: k.parse_cmd_pool, k.CLOCK_TURN_R_VAR: k.parse_clock_turn_radius,
        k.CLOCK_COMBINE_VAR: k.parse_clock_combine, k.YAW_RANGE_VAR: k.parse_yaw_range,
        k.RSI_SOURCE_VAR: k.parse_rsi_source, k.ANG_VEL_STD_VAR: k.parse_ang_vel_std,
        k.ANG_TURN_ONLY_VAR: k.parse_ang_turn_only, k.FACING_TURN_W_VAR: k.parse_facing_turn_w,
        k.TRACK_FACING_SIGMA_VAR: k.parse_track_facing_sigma, k.PROGRESS_W_VAR: k.parse_progress_w,
        k.FACING_OBS_ENCODING_VAR: k.parse_facing_obs_encoding,
        k.FACING_OBS_CLIP_VAR: float, k.YAW_PROGRESS_W_VAR: float, "KRABBY_YAW_W": float,
        # side-speed-bisection knobs (plan step 1, 2026-09-28): the two new knobs' own parsers,
        # plus the existing knobs R and C overlay that had no parser entry (KRABBY_OMNI_CLOCK_COMBINE
        # already does, above).
        k.MIRROR_IMAGES_VAR: k.parse_mirror_images, k.CMD_OBS_VAR: k.parse_cmd_obs,
        k.RSI_FRAC_VAR: float, "KRABBY_TRACK_SIGMA2": float, "KRABBY_TRACK_L1_W": float,
        # motion-quality knobs (step 2): their own parsers, plus the three reused forward weights
        **k.MOTION_KNOB_PARSERS,
        k.POWER_W_VAR: float, k.ACTION_RATE_W_VAR: float, k.AIRTIME_W_VAR: float,
        # clock-deprecation knobs (plan step 4, 2026-09-23): the two clock-hide knobs' own parsers,
        # the two forward clock weights and the forward reward-weight curriculum's spec parser
        **k.CLOCK_KNOB_PARSERS,
        "KRABBY_CLOCK_W": float, "KRABBY_APEX_W": float,
        # NOTE(omni phase-2 plan step 4, 2026-10-06): blank ("") round-trips to None.
        "KRABBY_PHASEOUT": parse_phaseout_knob,
        # stroke/posture knobs (stride/posture plan step 3, 2026-09-25): their own parsers.
        **k.STROKE_KNOB_PARSERS,
        # omni phase-2 plan step 4 (2026-10-06): the field-geometry knobs' own parsers (re-exported
        # by omni_knobs from mdp/omni_terrain_knobs.py), the geometry selector, the promotion pair
        # and the reused stride weight (the forward ``_overrides`` table's ``float``). The F1 / E1
        # knobs (KRABBY_STAND_STANCE_W / KRABBY_POWER_JOINTS) come in through MOTION_KNOB_PARSERS.
        **k.FIELD_KNOB_PARSERS,
        TERRAIN_GEOM_VAR: parse_terrain_geom,
        TERRAIN_PROMOTE_VAR: parse_terrain_promote,
        "KRABBY_STRIDE_W": float,
        # omni phase-2 walls re-tune (CHARTER 9, 2026-10-06): the arms widen the 2a preset's difficulty range.
        "KRABBY_FLAT_TERRAIN_DIFF": parse_terrain_diff,
        # omni phase-2 stage p2b (P2B_PLAN step 1, 2026-10-08): the CL / TG arms' reused forward
        # knobs (sign-checked; blank refused, as the cfg's float() would raise). The two new flags
        # (KRABBY_FOOT_CLEAR_TERRAIN / KRABBY_ORIENT_ALL_TILES) come in through MOTION_KNOB_PARSERS.
        "KRABBY_FOOT_CLEAR_W": _forward_weight_parser("KRABBY_FOOT_CLEAR_W", +1),
        "KRABBY_SWING_MIN_CLEAR_W": _forward_weight_parser("KRABBY_SWING_MIN_CLEAR_W", -1),
        "KRABBY_ORIENT_W": _forward_weight_parser("KRABBY_ORIENT_W", -1),
        "KRABBY_FOOT_CLEAR_MIN": parse_foot_clear_min,
        FOOT_CLEAR_FLAT_KEY: lambda t: _motion().parse_flag(t, FOOT_CLEAR_FLAT_KEY),
    }
    _KNOB_PARSERS_CACHE = table
    return table


_WHITELIST_CACHE = None


def knob_whitelist() -> set[str]:
    """Every ``KRABBY_*`` a recipe may set: the union of the presets, the formation frame, the omni
    knob module's own variables, the eval carry, and the forward reward/env literals.

    A union of what EXISTS rather than a hand-written list: the failure this prevents is a typo'd
    knob name in a candidate, which reaches the subprocess, is stripped by nothing, applies nothing,
    and produces a clean 3.9-hour window of the WRONG recipe. ``KRABBY_YAW_W`` is accepted because
    the forward mdp cfg names it; ``KRABBY_YAW_WEIGHT`` is refused because nothing does.
    """
    global _WHITELIST_CACHE
    if _WHITELIST_CACHE is not None:
        return set(_WHITELIST_CACHE)
    names = set()
    for spec in ph.PHASES.values():
        names.update(spec.env)
    names.update(ph.FORMATION)
    names.update(ph.OMNI_COMMON)
    names.update(rp.EVAL_CARRY)
    k = knobs()
    for attr in dir(k):
        if attr.endswith("_VAR"):
            value = getattr(k, attr)
            if isinstance(value, str) and value.startswith("KRABBY_"):
                names.add(value)
    # every knob the motion module PARSES is one it owns (omni phase-2 stage p2b, 2026-10-08): a
    # motion_knobs flag whose *_VAR omni_knobs forgot to re-export must not read as a typo.
    names.update(key for key in getattr(k, "MOTION_KNOB_PARSERS", {}) if str(key).startswith("KRABBY_"))
    mdp_cfg = rp.FWD_PKG / "config/crab_hex/agents/parkour_mdp_cfg.py"
    if mdp_cfg.exists():
        names.update(re.findall(r"KRABBY_[A-Z_]+", mdp_cfg.read_text(errors="ignore")))
    names -= set(FORBIDDEN_OVERLAY_KEYS)
    _WHITELIST_CACHE = names
    return set(names)


_REPLAY_FLAGS_CACHE: dict = {}


def replay_flags(script: Path) -> set[str]:
    """The long flags the replay screen's ``--help`` advertises (``{"--yaw-w", ...}``).

    Read from ``--help`` rather than from the plan: a candidate that declares a flag the script does
    not have would fail its advisory replay at the worst moment -- immediately before a GPU window
    -- and the advisory path swallows errors by design.
    """
    key = str(script)
    if key in _REPLAY_FLAGS_CACHE:
        return set(_REPLAY_FLAGS_CACHE[key])
    flags = set()
    try:
        out = subprocess.run([PY, str(script), "--help"], capture_output=True, text=True, timeout=120)
        flags = set(re.findall(r"--[a-zA-Z][\w-]*", out.stdout))
    except (OSError, subprocess.SubprocessError):
        flags = set()
    _REPLAY_FLAGS_CACHE[key] = flags
    return set(flags)


def _effective_stage_cfg(cands: dict, stage: dict, overlay: dict) -> dict:
    """``{knob: value}`` a stage would really build with: its preset, then the overlay on top."""
    env = dict(ph.phase_env(str(stage["base"])))
    env.update({k: str(v) for k, v in (overlay or {}).items()})
    return env


def _turn_hold_s(row: str = "suite__omni", hold: str = "turn_l90") -> float | None:
    """The scored turn hold's duration -- the SUITE's ``turn_l90`` (plan B3: every screen is scored
    there now; 12 s, as the legacy ``slow`` row's was)."""
    try:
        return float(rp._manifest_hold_s(row)[hold])
    except Exception:                                                   # noqa: BLE001
        return None


#: The words that count as "the candidate's `question` RECORDS the cadence change" (the plan's own
#: waiver). A WORD match, not the substring `caden`: any text containing "cadence" anywhere -- a
#: question that merely says "cadence-preserving", or one that mentions the word in passing -- used
#: to waive the rule, so the waiver could be bought by accident.
_CADENCE_WORDS = re.compile(r"\bcadences?\b", re.I)
#: ... and a question that says the cadence is PRESERVED asserts the OPPOSITE of the change the
#: waiver exists for, so it must not buy it. Both shipped cap candidates are spelled that way (and
#: both really do preserve it, so neither reaches the waiver); a future one whose arithmetic says
#: otherwise is now refused instead of waved through by its own reassurance.
_CADENCE_KEPT = re.compile(r"\bcadence[- ]?(preserving|preserved|unchanged|neutral)\b"
                           r"|\b(preserves?|preserving|keeps?|unchanged)\b[^.]{0,40}\bcadences?\b",
                           re.I)


def _records_cadence(question: str | None) -> bool:
    """Does the candidate's ``question`` RECORD the cadence change (the plan's waiver)?

    Clause by clause, so a question may say BOTH things ("cadence-preserving? no -- the cadence
    changes deliberately"): the waiver needs at least one clause that names the cadence and does
    NOT describe it as preserved.
    """
    text = str(question or "")
    return any(_CADENCE_WORDS.search(clause) and not _CADENCE_KEPT.search(clause)
               for clause in re.split(r"[.;:\n?]|--", text))


def _check_cap_rules(env: dict, where: str, question: str, reasons: list) -> None:
    """The clock rule, the manifest turn feasibility and the two cfg rules, on ONE effective env."""
    cap = _num(env.get("KRABBY_OMNI_ANG_VEL_MAX"))
    # No preset sets the clip, so reading it off the overlay alone left the `cap > clip` rule
    # UNARMED for exactly the candidate shape it exists for: a cap-only candidate in (0.15, 0.2]
    # passed load and died at config build (no checkpoint -> `infra` -> two retries ->
    # `halt: infra`, which `RESUMABLE_HALTS` then resumes into the same deterministic failure).
    # The default is the class default the eval carries, the way `walk_floor` handles lin_vel_clip.
    clip = _num(env.get("KRABBY_OMNI_ANG_VEL_CLIP"))
    if clip is None:
        # LOUDLY, not silently: the default is the rule's other operand, so a missing or unreadable
        # key would UNARM `cap > clip` again with no reason appended -- the validator would pass a
        # cap-only candidate in (0.15, 0.2] exactly as it did before F5, and the failure would
        # resurface 40 minutes into a window as a config-build death. The key is
        # `run_phases.EVAL_CARRY_DEFAULTS["KRABBY_OMNI_ANG_VEL_CLIP"]`; if it ever moves, the
        # validator must stop rather than shrug.
        clip = _num(rp.EVAL_CARRY_DEFAULTS.get("KRABBY_OMNI_ANG_VEL_CLIP"))
        if clip is None:
            raise RuntimeError(
                "run_phases.EVAL_CARRY_DEFAULTS['KRABBY_OMNI_ANG_VEL_CLIP'] is missing or not a "
                "number, so the validator cannot apply `ang_vel_max > ang_vel_clip` to "
                f"{where}: fix the default (or overlay KRABBY_OMNI_ANG_VEL_CLIP on every cap "
                "candidate) -- an unarmed cap rule is a config-build death mid-window")
    radius = _num(env.get("KRABBY_OMNI_CLOCK_TURN_R"))
    if cap is None or radius is None:
        reasons.append(f"{where}: cannot resolve ang_vel_max/clock_turn_radius (got {cap}, {radius})")
        return
    speed = radius * cap
    if not speed >= CLOCK_TURN_MIN_SPEED:
        reasons.append(f"{where}: KRABBY_OMNI_CLOCK_TURN_R {radius} x KRABBY_OMNI_ANG_VEL_MAX {cap} = "
                       f"{speed:.4g} m/s is below the gait clock's stop speed {CLOCK_TURN_MIN_SPEED} "
                       "-- every commanded turn would run on a frozen clock")
    elif speed < CAP_CADENCE_MIN and not _records_cadence(question):
        reasons.append(f"{where}: a cap candidate at {radius} x {cap} = {speed:.4g} m/s changes the "
                       f"gait cadence (shipped 0.36); either co-overlay a cadence-preserving radius "
                       f"(>= {CAP_CADENCE_MIN}) or record the cadence change in the candidate's question")
    if not cap > clip:                      # `clip` is never None here: the raise above saw to that
        reasons.append(f"{where}: ang_vel_max {cap} must exceed ang_vel_clip {clip} (the dead band "
                       "would zero every yaw-rate command, saturated ones included)")
    if cap < TURN_CAP_MIN:
        # The plan states the floor twice: "manifest 12-s-turn feasibility RECOMPUTED with the
        # overlaid ang_vel_max (refuse < 0.15)". The exact arithmetic below refuses below
        # 1.0 + (pi/2)/12 = 0.1428 rad/s, so a cap in [0.1428, 0.15) would pass the recomputation
        # while failing the stated floor. Both are applied; the stricter one bites.
        reasons.append(f"{where}: ang_vel_max {cap} is below the plan's floor {TURN_CAP_MIN} rad/s "
                       "for the manifest's scored turn holds")
    hold_s = _turn_hold_s()
    if hold_s is not None:
        need = TURN_SETUP_S + QUARTER_TURN_RAD / cap
        if need > hold_s:
            reasons.append(f"{where}: a 90-degree turn at cap {cap} rad/s needs {need:.2f} s "
                           f"({TURN_SETUP_S} s setup + pi/2 / cap) but the manifest's turn hold is "
                           f"{hold_s} s -- the scored turn holds become infeasible")
    ellipse = env.get("KRABBY_OMNI_SPEED_ELLIPSE")
    if ellipse:
        axes = parse_slots_str(ellipse)
        floor = walk_floor(env) + MIN_CAP_HEADROOM_M_S
        if not min(axes) > floor:
            reasons.append(f"{where}: speed_ellipse min {min(axes)} must exceed the walk floor "
                           f"+ headroom {floor}")


def walk_floor(env: dict) -> float:
    """``lin_vel_clip + WALK_MARGIN_M_S`` -- the speed every walking draw must stay above.

    ``lin_vel_clip`` is not an omni knob (no preset sets it), so the class default 0.2 is the value
    unless a future knob appears; the margin is the command module's own constant.
    """
    clip = _num(env.get("KRABBY_OMNI_LIN_VEL_CLIP"))
    return (0.2 if clip is None else clip) + WALK_MARGIN_M_S


def _check_require_env(env: dict, where: str, require_env: dict, reasons: list) -> None:
    """Every ``candidates.yaml`` ``require_env`` var reaches this composed stage env, at exactly the
    value the campaign pinned.

    ``require_env`` exists so a launcher can pin a cross-cutting knob -- the stride/posture plan's
    clock-free ``K`` is the model -- onto EVERY run a campaign composes: the baseline chain and
    every candidate's own chain (which is also what a run's seed-2 twin trains under, since a twin
    differs only in seed). A stage missing the var, or carrying a different one, is refused by
    name: a preset or an overlay that silently drops it would otherwise train the wrong recipe with
    a config dump that never says so.
    """
    for var, want in (require_env or {}).items():
        got = env.get(var)
        if got is None or str(got).strip() != str(want).strip():
            reasons.append(f"{where}: require_env {var} must be {want!r}, got {got!r}")


def _check_pool_overlay(overlay: dict, where: str, reasons: list, env: dict | None = None) -> None:
    """The pool rules replicated from ``_validate_omni_cfg``: projection equality, no ``dir_band``,
    the floor share, the ``POOL_TYPES`` render order that makes a pool a bitwise twin -- and, when
    ``env`` gives the stage's effective cfg, the walk-floor bound on the pool's OWN ``speed_scale``
    fields (PLAN_AMENDMENTS 3 spells that bound, and ``_check_pool_edit`` applied it only to a
    ``pool_edit``: a chain stage that writes ``fwd:0.2:1.0:0.5`` in its raw pool string got it for
    free, and every draw of that type would sit under the gait clock's stop speed)."""
    raw = overlay.get("KRABBY_OMNI_CMD_POOL")
    if raw is None or not str(raw).strip():
        return
    math_mod = pool_math()
    try:
        pool = parse_pool(raw)
    except ValueError as exc:
        reasons.append(f"{where}: KRABBY_OMNI_CMD_POOL does not parse ({exc})")
        return
    band = overlay.get("KRABBY_OMNI_DIR_BAND")
    if band is None:
        reasons.append(f"{where}: a pool overlay must also set KRABBY_OMNI_DIR_BAND='' -- a pool and "
                       "a global direction band are two spellings of one lever and the cfg refuses both")
    elif str(band).strip() and str(band).strip().lower() not in ("none", "off", "null"):
        reasons.append(f"{where}: KRABBY_OMNI_DIR_BAND={band!r} is set together with a pool (refused "
                       "by the command cfg: the pool gives every type its own band)")
    slots = overlay.get("KRABBY_OMNI_SLOTS")
    if slots is None:
        reasons.append(f"{where}: a pool overlay must declare KRABBY_OMNI_SLOTS = the pool's projection")
    else:
        declared = parse_slots_str(slots)
        total = sum(declared)
        projected = pool_slots(pool)
        if total <= 0 or any(abs(d / total - p) > 1e-6 for d, p in zip(declared, projected)):
            reasons.append(f"{where}: KRABBY_OMNI_SLOTS {slots} normalises to "
                           f"{tuple(round(d / total, 6) for d in declared) if total > 0 else declared} but the "
                           f"pool projects to {tuple(round(p, 6) for p in projected)} -- the cfg refuses a "
                           "mismatch (every metric, reward and eval downstream reads the SLOT)")
    order = [n for n in math_mod.POOL_TYPES if n in pool.names]
    if list(pool.names) != order:
        reasons.append(f"{where}: pool types are written out of POOL_TYPES order ({list(pool.names)} vs "
                       f"{order}) -- legal, but not a bitwise twin of a legacy window; reorder it")
    low = [(n, w) for n, w in zip(pool.names, pool.weights)
           if 0.0 < float(w) < math_mod.POOL_FLOOR_SHARE]
    if low:
        reasons.append(f"{where}: {low} are below POOL_FLOOR_SHARE {math_mod.POOL_FLOOR_SHARE} -- a type "
                       "at that share is in the dump and absent from the distribution")
    if env is not None:
        ellipse = parse_slots_str(env.get("KRABBY_OMNI_SPEED_ELLIPSE", "0.35:0.25:0.25"))
        min_speed = walk_floor(env) + MIN_CAP_HEADROOM_M_S
        for name, scale in zip(pool.names, pool.speed_scale):
            if float(scale) == 1.0:
                continue
            bound = math_mod.pool_speed_scale_floor(name, ellipse, min_speed)
            if float(scale) <= bound:
                reasons.append(f"{where}: KRABBY_OMNI_CMD_POOL sets speed_scale {scale} on {name!r}, "
                               f"at or below the walk-floor bound {bound:.4g} (ellipse {ellipse}, min "
                               f"speed {min_speed}) -- every draw of that type would sit under the "
                               "gait clock's stop speed. Widen speed_ellipse or lower lin_vel_clip "
                               "in the same overlay.")


def validate_candidates(cands: dict, *, strict: bool = True, check_replay: bool = True) -> list[dict]:
    """Load-time validation of the whole campaign spec. Returns one verdict per subject.

    A verdict is ``{"id", "ok", "deferred", "reasons": [refusals], "notes": [...]}``; the subject id
    is a candidate id, a chain name, or ``"<campaign>"`` for the global block. With ``strict`` a
    single refusal raises ``SystemExit`` naming all of them -- the campaign must not start on a spec
    that cannot be honoured, because the cost of finding out later is a booked GPU window.

    Every rule here is a rule the RUN would otherwise discover: the config validators' own
    (projection equality, dir_band exclusivity, ang_vel_max > clip, the ellipse walk floor), the
    cross-knob clock rule, the manifest's turn feasibility under an overlaid cap, the walk-floor
    bound on a ``speed_scale`` (via the exported ``pool_speed_scale_floor``, so a recipe that cannot
    be built dies at compose time), the knob-name whitelist, the value round trips, the dump-audit
    coverage, the replay flags, and the ordering rules (``depends_on`` acyclic and backwards,
    ``conditional_refs`` ⊆ ``depends_on``).
    """
    verdicts: list[dict] = []
    order = candidate_ids(cands)
    whitelist = knob_whitelist()
    parsers = knob_parsers()
    math_mod = pool_math()
    require_env = cands.get("require_env") or {}

    # ---------------- global
    g: list[str] = []
    if not isinstance(require_env, dict) or any(not isinstance(k2, str) for k2 in require_env):
        g.append(f"require_env must map KRABBY_* variable names to values, got {require_env!r}")
        require_env = {}
    notes: list[str] = []
    if int(cands.get("schema") or 0) != 1:
        g.append(f"schema must be 1, got {cands.get('schema')!r}")
    plant = cands.get("plant")
    if plant not in ph.PLANTS:
        g.append(f"plant {plant!r} is not one of {sorted(ph.PLANTS)}")
    seeds = cands.get("seeds") or {}
    if not seeds.get("screen") or not seeds.get("confirm"):
        g.append(f"seeds must name screen and confirm, got {seeds!r}")
    screen = cands.get("screen") or {}
    base = cands.get("baseline") or {}
    for name in {screen.get("chain"), base.get("chain")}:
        if name not in (cands.get("chains") or {}):
            g.append(f"chain {name!r} (screen/baseline) is not defined")
    if not base.get("id"):
        g.append("baseline needs an id (B0)")
    if len(order) != len(set(order)):
        g.append(f"duplicate candidate ids in {order}")
    if not promotions_enabled(cands) and (cands.get("milestones") or []):
        g.append("promotion.enabled is false but milestones are declared: a milestone IS a promotion "
                 "(2a -> 2b) of the recipe head")
    for mstone in cands.get("milestones") or []:
        if mstone.get("after") not in order:
            g.append(f"milestone {mstone.get('id')} after {mstone.get('after')!r} is not a candidate id")
        if list(mstone.get("promote") or []) != list(PROMOTION_STAGES):
            g.append(f"milestone {mstone.get('id')} promote must be {list(PROMOTION_STAGES)}")
    rule = cands.get("decision_rule") or {}
    _check_rule_v2(rule, g)
    pilot = rule.get("pilot")
    if pilot not in (None, "") and str(pilot) not in order:
        g.append(f"decision_rule.pilot names {pilot!r}, which is not a candidate id (the pilot stop "
                 "parks on that candidate's seed-3 line)")
    refs = cands.get("references") or {}
    if not isinstance(refs, dict):
        g.append(f"references must map a name to {{chain, seed}}, got {type(refs).__name__}")
        refs = {}
    for rname, ref in refs.items():
        _check_reference(cands, str(rname), ref, g)
    script = cands.get("replay", {}).get("script")
    script_path = repo_path(script) if script else None
    if script and not script_path.exists():
        g.append(f"replay.script {script} does not exist under the repo root")
    for item in cands.get("resolution_inputs") or []:
        _check_resolution_input(item, g, notes)
    for row in (cands.get("promotion") or {}).get("eval") or []:
        if not _known_eval_row(str(row)):
            g.append(f"promotion.eval names row {row!r}, which the omni manifest does not carry")
    for stage, table in (cands.get("milestone_reference") or {}).items():
        if str(stage) not in PROMOTION_STAGES:
            g.append(f"milestone_reference names stage {stage!r}, not one of {list(PROMOTION_STAGES)}")
        for row, rel in (table or {}).items():
            _check_milestone_reference(str(stage), str(row), rel, g, notes)
    verdicts.append({"id": "<campaign>", "ok": not g, "deferred": False, "reasons": g, "notes": notes})

    # ---------------- chains
    screen_iters = int((cands.get("screen") or {}).get("iters") or 0)
    # A chain only a `references:` run uses is exempt from `screen.iters` (it is judged by hand, never
    # against a screen): Z's from-scratch chain is 8000 iterations, the t3_s2 prefix 3000.
    ref_only = reference_only_chains(cands)
    for name, chain in (cands.get("chains") or {}).items():
        r: list[str] = []
        cn: list[str] = []
        stages = list(chain.get("stages") or [])
        if not stages:
            r.append("chain has no stages")
        total = 0
        seen = set()
        for i, stage in enumerate(stages):
            sname = str(stage.get("stage"))
            where = f"chain {name} stage {sname}"
            if sname in seen:
                r.append(f"{where}: duplicate stage name")
            seen.add(sname)
            if str(stage.get("base")) not in PHASES_ALLOWED_BASES:
                r.append(f"{where}: base {stage.get('base')!r} is not one of {PHASES_ALLOWED_BASES}")
            kind = resume_kind(stage)
            if kind not in RESUME_KINDS:
                r.append(f"{where}: resume_from {stage.get('resume_from')!r} not in {RESUME_KINDS} "
                         "(the checkpoint kind is spelled `resume_from: {checkpoint: <path>}`)")
            if kind == "previous_stage" and i == 0:
                r.append(f"{where}: the first stage cannot resume from the previous stage")
            if stage.get("from_scratch") and kind != "scratch":
                r.append(f"{where}: from_scratch but resume_from={stage.get('resume_from')!r}")
            if kind == "checkpoint":
                _check_fork_stage(stage, where, r, cn)
            if kind == "reference":
                _check_reference_stage(stage, where, r, refs)
            iters = int(stage.get("iters") or 0)
            if iters <= 0:
                r.append(f"{where}: iters must be positive")
            total += iters
            overlay = dict(stage.get("overlay") or {})
            for key, value in overlay.items():
                _check_knob(key, value, where, whitelist, parsers, r, set())
            stage_env_ = _effective_stage_cfg(cands, stage, overlay)
            _check_stage_categories(stage, stage_env_, where, r)
            if floor_park_categories(stage) and not flags_enabled(rule):
                r.append(f"{where}: floor park_categories needs a decision rule with block "
                         f"{RIG_BLOCK} (the flag/park machinery): without it a park-category floor "
                         "would be recorded and park nothing")
            _check_pool_overlay(overlay, where, r, stage_env_)
            _check_cap_rules(stage_env_, where, chain.get("notes") or "", r)
            _check_require_env(stage_env_, where, require_env, r)
            _check_stroke_cat_w_composition(stage_env_, where, r)
            _check_field_geom_composition(stage_env_, where, r)
            _check_cmd_obs_mirror_composition(stage_env_, where, r)
            _check_rsi_cam_stage_composition(stage_env_, where, r)
            _check_restance_composition(stage_env_, where, r)
        if screen_iters and total != screen_iters and name not in ref_only:
            r.append(f"chain {name} totals {total} iterations; every screen is {screen_iters} "
                     "(equal compute is what makes the screens comparable)")
        if stages and all(isinstance(s, dict) and s.get("iters") for s in stages):
            try:
                composed, touches = _composed_clock_stages(cands, name)
            except (KeyError, ValueError):
                composed, touches = [], False       # a bad base/stage: refused above by name
            if touches:
                _check_clock_chain(composed, f"chain {name}", r)
        verdicts.append({"id": f"chain:{name}", "ok": not r, "deferred": False, "reasons": r, "notes": cn})

    # ---------------- candidates
    raw_pool_setters: dict[tuple, list[str]] = {}
    key_setters: dict[str, list[str]] = {}
    for idx, cand in enumerate(cands.get("candidates") or []):
        cid = cand.get("id")
        r, n = [], []
        deferred = bool(cand.get("deferred"))
        exempt = set()
        req = cand.get("requires_knob")
        req_keys = [req] if isinstance(req, str) else list(req or [])
        if deferred:
            # Only while the knob does NOT exist: once it lands, a deferred candidate's overlay is
            # checked like every other (whitelist, parser, and an OVERLAY_DUMP_CHECKS entry so
            # overlay_checks can audit it). Exempting it unconditionally would let the knob reach a
            # booked 3.9 h window with no dump checker and halt config_mismatch after it trained.
            exempt.update(k for k in req_keys if k not in whitelist)
        if not cand.get("question"):
            r.append("no question recorded (the ledger's own explanation of the window)")
        if cand.get("chain") and cand["chain"] not in (cands.get("chains") or {}):
            r.append(f"chain {cand['chain']!r} is not defined")
        if cand.get("speculative_default") not in (None, "incumbent", "candidate"):
            r.append(f"speculative_default {cand.get('speculative_default')!r} unknown")
        deps = list(cand.get("depends_on") or [])
        for dep in deps:
            if dep not in order:
                r.append(f"depends_on {dep!r} is not a candidate id")
            elif order.index(dep) >= idx:
                r.append(f"depends_on {dep!r} is not ranked before {cid} (the queue would deadlock)")
        refs = conditional_refs(cand.get("conditional"))
        for ref in refs:
            if ref not in deps:
                r.append(f"conditional names decision({ref}) but {ref} is not in depends_on "
                         "(the predicate would be evaluated before its referent is decided)")
        if cand.get("conditional"):
            try:
                compile_predicate(cand["conditional"])
            except ValueError as exc:
                r.append(f"conditional does not parse: {exc}")
        if cand.get("await_if"):
            # The branch-and-ask gate (``await_if``): same vocabulary and the same ordering rule as
            # `conditional` -- a decision it names must be TERMINAL before it is read.
            try:
                compile_predicate(cand["await_if"])
            except ValueError as exc:
                r.append(f"await_if does not parse: {exc}")
            for ref in conditional_refs(cand.get("await_if")):
                if ref not in deps:
                    r.append(f"await_if names decision({ref}) but {ref} is not in depends_on (the "
                             "gate would be evaluated before its referent is decided)")
        for key in cand.get("eval_carry") or []:
            if key not in rp.EVAL_CARRY:
                r.append(f"eval_carry {key} is not in run_phases.EVAL_CARRY {rp.EVAL_CARRY}")
        # every overlay form: stages: all, per-stage, per-chain
        known_stages = set(PROMOTION_STAGES)
        for _cname in (cands.get("chains") or {}):
            known_stages.update(stage_names(cands, _cname))
        blocks = []
        for chain_name in (cands.get("chains") or {}):
            spec = _cand_spec(cand, chain_name)
            overlay = spec.get("overlay") or {}
            # THIS chain's blocks only. The stray-FACING check below reads `spec`, which is this
            # chain's; scanning the accumulated list would judge an earlier chain's block against a
            # later chain's spec (and is O(chains^2)).
            chain_blocks = []
            if spec.get("stages") == "all":
                chain_blocks.append((f"{cid} on {chain_name} (stages: all)", chain_name, overlay, "*"))
            elif isinstance(overlay, dict):
                for sname, block in overlay.items():
                    if isinstance(block, dict):
                        chain_blocks.append((f"{cid} on {chain_name} stage {sname}", chain_name, block,
                                             str(sname)))
                        # A per-stage overlay naming a stage that exists NOWHERE is a typo, and at
                        # compose time it is only appended to `conflicts` -- which nothing in a live
                        # run reads. The overlay would then be silently inert for the whole 3.9-hour
                        # window and the DECISION line would report a lever that never applied.
                        # Judged against every chain's stages (a candidate with no `chain:` of its
                        # own runs on whatever the recipe carries), and against its OWN chain's when
                        # it declares one.
                        if str(sname) not in known_stages:
                            msg = (f"{cid}: overlay names stage {sname!r}, which is in no chain "
                                   f"(known: {sorted(known_stages)})")
                            if msg not in r:
                                r.append(msg)
                        elif cand.get("chain") and str(sname) not in (
                                stage_names(cands, cand["chain"]) + list(PROMOTION_STAGES)):
                            msg = (f"{cid}: overlay names stage {sname!r}, which is not in its own "
                                   f"chain {cand['chain']} "
                                   f"({stage_names(cands, cand['chain'])} + {list(PROMOTION_STAGES)})")
                            if msg not in r:
                                r.append(msg)
            if promotions_enabled(cands):
                # (a campaign with promotions disabled composes no teacher overlay: nothing to check)
                t_overlay = _teacher_overlay_for(spec, chain_name, stage_names(cands, chain_name))
                chain_blocks.append((f"{cid} teacher overlay on {chain_name}", chain_name, t_overlay,
                                     "teacher"))
            blocks.extend(chain_blocks)
            for where, _c, block, label in chain_blocks:
                if label in ("teacher",) + PROMOTION_STAGES and "KRABBY_OMNI_FACING" in (block or {}) \
                        and "teacher_overlay" not in spec:
                    msg = (f"{cid}: a 2a/2b overlay sets KRABBY_OMNI_FACING without declaring it in "
                           "teacher_overlay (the teacher presets own the facing ramp)")
                    if msg not in r:
                        r.append(msg)
            if spec.get("pool_edit"):
                _check_pool_edit(cands, chain_name, dict(spec["pool_edit"]), f"{cid} on {chain_name}", r)
            if promotions_enabled(cands):
                _check_teacher_slots(cands, cand, spec, chain_name, f"{cid} on {chain_name} 2a/2b", r)
        for where, chain_name, overlay, label in blocks:
            for key, value in (overlay or {}).items():
                _check_knob(key, value, where, whitelist, parsers, r, exempt)
                if not baseline_control(cand):
                    # (a `control: baseline` arm never composes, so it can conflict with nothing)
                    key_setters.setdefault(key, []).append(cid)
                if key == "KRABBY_OMNI_CMD_POOL":
                    raw_pool_setters.setdefault((chain_name, label), []).append(cid)
            if overlay and ("KRABBY_OMNI_ANG_VEL_MAX" in overlay or "KRABBY_OMNI_ANG_VEL_CLIP" in overlay
                            or "KRABBY_OMNI_CLOCK_TURN_R" in overlay):
                # Checked once per distinct BASE PRESET rather than per stage: the overlay is the same
                # everywhere it applies and only the preset underneath it varies, so one reason per
                # preset is the whole story instead of one per stage per chain.
                stages = promotion_stage_list(cands) if label in ("teacher",) + PROMOTION_STAGES else [
                    st for st in chain_stages(cands, chain_name)
                    if label in ("*", str(st["stage"]))]
                seen_bases = set()
                for stage in stages:
                    if str(stage["base"]) in seen_bases:
                        continue
                    seen_bases.add(str(stage["base"]))
                    env = _effective_stage_cfg(cands, stage, {**(stage.get("overlay") or {}), **overlay})
                    _check_cap_rules(env, f"{where} @ preset {stage['base']}",
                                     cand.get("question") or "", r)
        # NOTE(stride/posture 2026-09-26): `overrides_chain` -- the chain-overlay keys this candidate
        # deliberately re-sets (compose_recipe records no conflict for them). Each must be set by the
        # candidate's own overlay AND by some chain stage overlay: a typo would silence nothing, or
        # would silence a real conflict with nothing to override.
        oc = cand.get("overrides_chain")
        if oc is not None:
            if not isinstance(oc, (list, tuple)) or any(not isinstance(k, str) for k in oc):
                r.append(f"overrides_chain must be a list of KRABBY_* keys, got {oc!r}")
            else:
                chain_keys = {k for cname in (cands.get("chains") or {})
                              for st in chain_stages(cands, cname) for k in (st.get("overlay") or {})}
                own_keys = {k for _w, _c, block, label in blocks if label != "teacher"
                            for k in (block or {})}
                for key in oc:
                    if key not in own_keys:
                        r.append(f"overrides_chain names {key}, which {cid}'s overlay does not set")
                    if key not in chain_keys:
                        r.append(f"overrides_chain names {key}, which no chain stage overlay sets "
                                 "(nothing to override)")
        moved = sorted({key for _w, _c, block, _l in blocks for key in (block or {})
                        if key in ABSOLUTE_UNITS_KEYS})
        if moved and not cand.get("absolute_units"):
            # Decision 5: "Absolute-units candidates (C3a, C3b -- ANY candidate that changes the
            # command cap/clip or, if ever added, the speed ellipse): F = facing_closed +
            # whole-hold net_yaw_rate_toward, T = achieved_vx m/s." The cap IS the denominator of
            # every ratio, so scoring such a candidate on `ratio_by_hold` compares it against a
            # yardstick it moved itself -- and the number still reads plausible, which is why this
            # is a refusal rather than a note.
            r.append(f"{cid} overlays {moved} but does not declare absolute_units: true -- it moves "
                     "the denominator of every speed ratio, so it must be scored in m/s and rad/s "
                     "(decision 5)")
        if cand.get("absolute_units") and not moved:
            n.append("absolute_units is declared but no cap/clip/ellipse key is overlaid: the F_abs "
                     "and T_abs blocks are scored anyway")
        if cand.get("absolute_units") and is_rule_v2(rule):
            named = list(((rule.get("per_category") or {}).get("blocks") or {}))
            if not any(str(b).endswith("_abs") for b in named):
                r.append(f"{cid} declares absolute_units but the decision rule carries no *_abs block: "
                         "every block would read N/A and the screen would be a terminal SKIP")
            req = adopt_requires(rule)
            if req is not None and f"{req}_abs" not in named:
                r.append(f"{cid} declares absolute_units but adopt_requires {req} has no {req}_abs "
                         "twin: the candidate could never be adopted")
        if cand.get("replay_args") and check_replay and script_path is not None:
            have = replay_flags(script_path)
            for tok in cand["replay_args"]:
                if str(tok).startswith("--") and str(tok) not in have:
                    r.append(f"replay_args names {tok}, which {Path(script).name} --help does not "
                             f"advertise (have: {sorted(have)})")
        if deferred:
            missing = [k for k in req_keys if k not in whitelist]
            if missing:
                n.append(f"deferred: refused at dispatch until {missing} is exported "
                         "(accepted at load, as planned)")
            else:
                n.append(f"deferred: its knob(s) {req_keys} now exist -- the dispatch check passes")
        if cand.get("user_decision"):
            n.append("user_decision: the driver stops at await_user before dispatching it")
        # NOTE(clock-deprecation 2026-09-23, plan step 4): `control: baseline` and the clock rules
        # (i)-(v) on every chain the candidate composes onto (its own, else the baseline/screen one).
        if cand.get("control") not in (None, "", "baseline"):
            r.append(f"control {cand.get('control')!r} unknown (the only per-candidate control is "
                     "`baseline`: judged against the baseline's own screens, never composed)")
        if baseline_control(cand):
            n.append("control: baseline -- its control and drift anchor are the baseline's screens, "
                     "and it is excluded from compose-on-adopt")
        clock_kind = str(cand.get("kind") or "") == "clock"
        own_chains = [cand["chain"]] if cand.get("chain") else list(dict.fromkeys(
            c for c in ((cands.get("baseline") or {}).get("chain"), (cands.get("screen") or {}).get("chain"))
            if c))
        for chain_name in own_chains:
            if chain_name not in (cands.get("chains") or {}):
                continue
            try:
                composed, touches = _composed_clock_stages(cands, chain_name, cand)
            except (KeyError, ValueError):
                continue                             # refused above by name
            if touches or clock_kind:
                _check_clock_chain(composed, f"{cid} on {chain_name}", r, clock_kind=clock_kind)
            # NOTE(stride/posture plan step 3, 2026-09-25): require_env, on every stage of every
            # chain the candidate composes onto -- unconditionally, not gated on `touches`/`clock_kind`
            # (a twin trains this same composed env under a different seed, so checking it here
            # covers the baseline, every twin and every candidate at once).
            for sname, _iters, senv in composed:
                _check_require_env(senv, f"{cid} on {chain_name} stage {sname}", require_env, r)
                _check_stroke_cat_w_composition(senv, f"{cid} on {chain_name} stage {sname}", r)
                _check_field_geom_composition(senv, f"{cid} on {chain_name} stage {sname}", r)
                _check_cmd_obs_mirror_composition(senv, f"{cid} on {chain_name} stage {sname}", r)
                _check_rsi_cam_stage_composition(senv, f"{cid} on {chain_name} stage {sname}", r)
        r = list(dict.fromkeys(r))          # one reason per distinct message, in order
        verdicts.append({"id": cid, "ok": not r, "deferred": deferred, "reasons": r, "notes": n})

    extra: list[str] = []
    for key, setters in raw_pool_setters.items():
        if len(set(setters)) > 1:
            extra.append(f"{sorted(set(setters))} all set a raw KRABBY_OMNI_CMD_POOL for {key} -- two "
                         "adopted candidates cannot both write the pool string for one stage")
    shared = {k: sorted(set(v)) for k, v in key_setters.items() if len(set(v)) > 1}
    verdicts.append({"id": "<overlays>", "ok": not extra, "deferred": False, "reasons": extra,
                     "notes": [f"{k} is set by {v} (a conflict only if both are adopted; compose_recipe "
                               f"records it)" for k, v in shared.items()]})

    if strict:
        bad = [v for v in verdicts if not v["ok"]]
        if bad:
            lines = [f"{v['id']}: {reason}" for v in bad for reason in v["reasons"]]
            raise SystemExit("candidates.yaml is not honourable:\n  - " + "\n  - ".join(lines))
    return verdicts


#: An overlay value starting with this (case-insensitive) is a placeholder the yaml has not filled in
#: yet (``TBD_replay_gate``): refused BY NAME, so the one thing a reader sees is what to fill in.
PLACEHOLDER_PREFIX = "TBD"


def _check_fork_stage(stage: dict, where: str, reasons: list, notes: list) -> None:
    """A ``resume_from: {checkpoint: <path>}`` stage (the named-fork resume kind).

    The path is repo-relative (the record must not carry a machine prefix) and names a ``.pt``; a
    file not on this machine is a NOTE -- the run would halt ``resume_head_missing`` before booking
    anything. Its ``category_survival`` floor must carry an explicit ``min_by_category`` for EVERY
    category the stage trains: the head it forks already trained them, but no earlier stage of THIS
    chain says so, so ``previous_categories`` is empty and ``min_new`` (a formation stage's 0.10)
    would pass a collapsed fine-tune.
    """
    rel = resume_checkpoint(stage)
    if not rel or not str(rel).strip():
        reasons.append(f"{where}: resume_from.checkpoint is empty")
        return
    if Path(str(rel)).is_absolute():
        reasons.append(f"{where}: resume_from.checkpoint must be a repo-relative path, got {rel!r}")
    elif not str(rel).endswith(".pt"):
        reasons.append(f"{where}: resume_from.checkpoint {rel!r} is not a .pt checkpoint")
    elif not (REPO / str(rel)).is_file():
        notes.append(f"{where}: fork checkpoint {rel} is not on this machine (the window would halt "
                     "resume_head_missing before booking a GPU)")
    floor = stage.get("floor") or {}
    if str(floor.get("kind")) == "category_survival":
        mins = floor.get("min_by_category") or {}
        missing = [c for c in (stage.get("categories") or []) if str(c) not in mins]
        if missing:
            reasons.append(f"{where}: a checkpoint fork's category_survival floor needs "
                           f"floor.min_by_category for every category it trains; missing {missing} "
                           "(no earlier stage of the chain trained them, so min_new would read them "
                           "as NEW at a formation floor)")


def _check_reference_stage(stage: dict, where: str, reasons: list, refs: dict) -> None:
    """A ``resume_from: {reference: <name>}`` stage (clock-deprecation plan step 4, 2026-09-23).

    The name must be a ``references:`` key. Its ``category_survival`` floor needs an explicit
    ``min_by_category`` for every category it trains, for the checkpoint fork's reason: the head it
    forks trained them, no earlier stage of THIS chain says so (the confirm's fork-first c4 stage
    uses X0's c4 floors).
    """
    name = resume_reference(stage)
    if not name or name not in refs:
        reasons.append(f"{where}: resume_from.reference {name!r} is not a references: key "
                       f"(known: {sorted(refs)})")
    floor = stage.get("floor") or {}
    if str(floor.get("kind")) == "category_survival":
        mins = floor.get("min_by_category") or {}
        missing = [c for c in (stage.get("categories") or []) if str(c) not in mins]
        if missing:
            reasons.append(f"{where}: a reference fork's category_survival floor needs "
                           f"floor.min_by_category for every category it trains; missing {missing}")


#: The clock weight an ABSENT KRABBY_CLOCK_W / KRABBY_APEX_W is judged at (see clock_stage_state).
CLOCK_HELD_W = 1.0


def clock_stage_state(env: dict, iters: int) -> dict:
    """ONE stage's clock state, from its composed env (preset + overlays) -- pure.

    ``hide`` / ``ramp`` (the clock-hide knobs), ``phaseout`` (the clock-term entries of
    ``KRABBY_PHASEOUT``; ``phaseout_terms`` lists every term it names), ``static`` (each clock term's
    RewTerm weight: its knob, else the HELD weight :data:`CLOCK_HELD_W` -- an absent knob must never
    read as 0.0 "removed", which would blind rules (ii)/(iv)/(v) and removal verification; every
    phase preset sets both knobs, and ``defaulted`` lists any knob this fell back for), and per term the weight
    at the stage's first step (``start_w``), at its last env step ``24 x iters`` (``end_w`` -- the
    EFFECTIVE end weight rule (v) and removal verification read) and its maximum over the window
    (``max_w``; the cosine is monotone, so the larger endpoint). A PHASEOUT entry replaces the static
    weight from the first step (the curriculum writes the term's weight every step). ``errors``
    collects values that do not parse (the knob round trip refuses them by name elsewhere).
    """
    cur = curriculums()
    out = {"hide": False, "ramp": None, "phaseout": {}, "phaseout_terms": [], "static": {},
           "start_w": {}, "end_w": {}, "max_w": {}, "errors": [], "defaulted": []}
    try:
        out["hide"] = _clock().parse_clock_hide(env.get(_clock().CLOCK_HIDE_VAR))
    except ValueError as exc:
        out["errors"].append(str(exc))
    try:
        out["ramp"] = _clock().parse_clock_hide_ramp(env.get(_clock().CLOCK_HIDE_RAMP_VAR))
    except ValueError as exc:
        out["errors"].append(str(exc))
    raw = env.get("KRABBY_PHASEOUT")
    entries_ = []
    if raw is not None and str(raw).strip():
        try:
            entries_ = cur.parse_phaseout_spec(str(raw))
        except ValueError as exc:
            out["errors"].append(str(exc))
    out["phaseout_terms"] = [e["term_name"] for e in entries_]
    out["phaseout"] = {e["term_name"]: e for e in entries_ if e["term_name"] in CLOCK_TERM_KNOBS}
    end_step = int(ph.STEPS_PER_ITER) * int(iters)
    for term, knob in CLOCK_TERM_KNOBS.items():
        static = _num(env.get(knob))
        if static is None:
            out["defaulted"].append(knob)
        out["static"][term] = CLOCK_HELD_W if static is None else static
        entry = out["phaseout"].get(term)
        if entry is None:
            w0 = w_end = out["static"][term]
        else:
            w0 = cur.ramp_value(0, entry["w0"], entry["w1"], entry["t0"], entry["t1"])
            w_end = cur.ramp_value(end_step, entry["w0"], entry["w1"], entry["t0"], entry["t1"])
        out["start_w"][term], out["end_w"][term], out["max_w"][term] = w0, w_end, max(w0, w_end)
    return out


def _touches_clock(*overlays) -> bool:
    return any(key in (ov or {}) for ov in overlays for key in CLOCK_KNOB_KEYS)


def _check_clock_chain(stages: list, where: str, reasons: list, *, clock_kind: bool = False) -> None:
    """The clock-deprecation validator rules (i)-(v) (plan step 4, 2026-09-23) on ONE composed chain.

    ``stages`` is ``[(stage name, iters, composed env)]`` in chain order. The rules:

    (i)   a clock weight of exactly 0.0 (a knob, or a clock-term PHASEOUT endpoint) is refused -- use
          0.001: a 0.0 term is skipped by the reward manager and its telemetry flatlines;
    (ii)  HIDE without a ramp needs both clock weights <= 0.001 over the whole stage: the reward
          comes off with or before the observation, never after (a clock that pays while hidden is
          non-Markov -- Siekmann 2021; Zhang 2024's unbalanced gaits);
    (iii) a ramp needs HIDE plus a PHASEOUT on exactly the two clock terms with PHASEOUT t1 <= ramp
          t0 < ramp t1 <= 24 x the stage's iterations, every value a multiple of 24 (env steps per
          iteration) -- and the PHASEOUT must reach 0.001 (w1) before the first episode is hidden,
          which is (ii) in the ramped form;
    (iv)  removal is monotone: a clock-term PHASEOUT never rises (and ends inside its stage, t
          multiples of 24), a hidden clock stays hidden in every later stage (a later ramp would
          show it again), a clock weight at <= 0.001 at a stage's end stays there;
    (v)   (``clock_kind``: a ``kind: clock`` candidate) the chain ENDS phase 1 with HIDE=1 and both
          EFFECTIVE end weights <= 0.001 (a PHASEOUT stage's w1, reached by the stage's end).
    """
    step = int(ph.STEPS_PER_ITER)
    states = []
    for sname, iters, env in stages:
        st = clock_stage_state(env, iters)
        w = f"{where} stage {sname}"
        end = step * int(iters)
        for err in st["errors"]:
            reasons.append(f"{w}: {err}")
        # (i)
        for term, knob in CLOCK_TERM_KNOBS.items():
            if knob in env and _num(env.get(knob)) == 0.0:
                reasons.append(f"{w}: (i) {knob}=0.0 is refused -- a term at exactly 0.0 is skipped by "
                               f"the reward manager and its telemetry flatlines; use {CLOCK_EPS_W}")
            entry = st["phaseout"].get(term)
            if entry and (float(entry["w0"]) == 0.0 or float(entry["w1"]) == 0.0):
                reasons.append(f"{w}: (i) KRABBY_PHASEOUT ramps {term} to/from 0.0 -- use {CLOCK_EPS_W}")
        # (iv), within the stage: every clock-term PHASEOUT falls, ends inside the stage, on 24s
        for term, entry in st["phaseout"].items():
            if float(entry["w1"]) > float(entry["w0"]):
                reasons.append(f"{w}: (iv) KRABBY_PHASEOUT RAISES {term} ({entry['w0']} -> "
                               f"{entry['w1']}) -- removal is monotone")
            if int(entry["t1"]) > end:
                reasons.append(f"{w}: KRABBY_PHASEOUT {term} ends at t1 {entry['t1']}, beyond the "
                               f"stage end {end} (24 x {iters}): its end weight is never settled")
            bad = [t for t in (entry["t0"], entry["t1"]) if int(t) % step]
            if bad:
                reasons.append(f"{w}: KRABBY_PHASEOUT {term} window {bad} is not a multiple of {step} "
                               "env steps per iteration")
        # (ii)
        if st["hide"] and st["ramp"] is None:
            high = {t: st["max_w"][t] for t in CLOCK_TERM_KNOBS if st["max_w"][t] > CLOCK_EPS_W + 1e-12}
            if high:
                reasons.append(f"{w}: (ii) KRABBY_OMNI_CLOCK_HIDE=1 without a ramp needs both clock "
                               f"weights <= {CLOCK_EPS_W} over the whole stage, got {high} -- the reward "
                               "must come off with or before the observation (a hidden clock that "
                               "still pays is a non-Markov reward)")
        # (iii)
        if st["ramp"] is not None:
            t0r, t1r = st["ramp"]
            if not st["hide"]:
                reasons.append(f"{w}: (iii) KRABBY_OMNI_CLOCK_HIDE_RAMP needs KRABBY_OMNI_CLOCK_HIDE=1")
            if sorted(st["phaseout_terms"]) != sorted(CLOCK_TERM_KNOBS):
                reasons.append(f"{w}: (iii) a ramp needs a KRABBY_PHASEOUT on exactly the two clock "
                               f"terms {sorted(CLOCK_TERM_KNOBS)}, got {st['phaseout_terms']}")
            for term, entry in st["phaseout"].items():
                if int(entry["t1"]) > t0r:
                    reasons.append(f"{w}: (iii) the {term} PHASEOUT ends at {entry['t1']}, after the "
                                   f"hide ramp starts at {t0r} (PHASEOUT t1 <= ramp t0)")
                if float(entry["w1"]) > CLOCK_EPS_W + 1e-12:
                    reasons.append(f"{w}: (iii) the {term} PHASEOUT ends at w1 {entry['w1']} > "
                                   f"{CLOCK_EPS_W}: episodes would be hidden while the clock still pays")
            if not t1r <= end:
                reasons.append(f"{w}: (iii) the hide ramp ends at {t1r}, beyond the stage end {end} "
                               f"(24 x {iters})")
            bad = [t for t in (t0r, t1r) if int(t) % step]
            if bad:
                reasons.append(f"{w}: (iii) hide ramp window {bad} is not a multiple of {step} env steps")
        states.append((sname, st))
    # (iv), along the chain
    hidden_at, removed_at = None, {t: None for t in CLOCK_TERM_KNOBS}
    for sname, st in states:
        if hidden_at is not None and (not st["hide"] or st["ramp"] is not None):
            reasons.append(f"{where} stage {sname}: (iv) shows the clock again after stage {hidden_at} "
                           "hid it -- removal is monotone along a chain")
        for term in CLOCK_TERM_KNOBS:
            if removed_at[term] is not None and st["max_w"][term] > CLOCK_EPS_W + 1e-12:
                reasons.append(f"{where} stage {sname}: (iv) {term} rises to {st['max_w'][term]} after "
                               f"stage {removed_at[term]} removed it -- removal is monotone")
        if st["hide"] and hidden_at is None:
            hidden_at = sname
        for term in CLOCK_TERM_KNOBS:
            if st["end_w"][term] <= CLOCK_EPS_W + 1e-12 and removed_at[term] is None:
                removed_at[term] = sname
    # (v)
    if clock_kind and states:
        last, st = states[-1]
        if not st["hide"]:
            reasons.append(f"{where}: (v) a kind: clock candidate ends phase 1 with "
                           f"KRABBY_OMNI_CLOCK_HIDE=1, but its final stage {last} shows the clock")
        high = {t: st["end_w"][t] for t in CLOCK_TERM_KNOBS if st["end_w"][t] > CLOCK_EPS_W + 1e-12}
        if high:
            reasons.append(f"{where}: (v) a kind: clock candidate ends phase 1 with both effective end "
                           f"weights <= {CLOCK_EPS_W}, but its final stage {last} ends at {high}")


def _composed_clock_stages(cands: dict, chain_name: str, cand: dict | None = None) -> tuple[list, bool]:
    """``([(stage, iters, composed env)], touches)`` of ``chain_name`` (+ ``cand``'s overlay on it)."""
    spec = _cand_spec(cand, chain_name) if cand else {}
    overlay = spec.get("overlay") or {}
    if isinstance(overlay, dict) and spec.get("stages") != "all":
        overlay = {str(k): v for k, v in overlay.items()}
    out, touches = [], False
    for stage in chain_stages(cands, chain_name):
        sname = str(stage["stage"])
        own = {}
        if cand:
            if spec.get("stages") == "all":
                own = dict(overlay) if isinstance(overlay, dict) else {}
            elif isinstance(overlay, dict) and isinstance(overlay.get(sname), dict):
                own = dict(overlay[sname])
        merged = {**(stage.get("overlay") or {}), **own}
        touches = touches or _touches_clock(stage.get("overlay"), own)
        out.append((sname, int(stage.get("iters") or 0), _effective_stage_cfg(cands, stage, merged)))
    return out, touches


def _check_knob(key: str, value, where: str, whitelist: set, parsers: dict, reasons: list,
                exempt: set) -> None:
    if key in FORBIDDEN_OVERLAY_KEYS:
        reasons.append(f"{where}: {key} may never appear in an overlay (the driver owns it)")
        return
    if key in exempt:
        return
    if key not in whitelist:
        reasons.append(f"{where}: {key} is not a known knob (nothing in the presets, omni_knobs, "
                       "EVAL_CARRY or the forward mdp cfg names it) -- a typo would train the wrong recipe")
        return
    if key not in OVERLAY_DUMP_CHECKS:
        reasons.append(f"{where}: {key} has no entry in OVERLAY_DUMP_CHECKS, so overlay_checks could "
                       "not audit it against params/env.yaml -- add one before using the knob")
    if str(value).strip().upper().startswith(PLACEHOLDER_PREFIX):
        # A value the campaign has not chosen yet (the motion-quality ladder's reward weights are
        # set by the offline replay gate, plan step 3): named as such, never as a parse failure.
        reasons.append(f"{where}: {key}={value!r} is a PLACEHOLDER, not a value -- fill it in "
                       "(the motion-quality reward weights come from the step-3 offline replay "
                       "gate) before the campaign can launch")
        return
    parser = parsers.get(key)
    if parser is not None:
        try:
            parser(str(value))
        except Exception as exc:                                        # noqa: BLE001
            reasons.append(f"{where}: {key}={value!r} does not round-trip ({exc})")


#: The v2 rule's block set, in both unit systems (plan D5). The validator requires every one of them
#: to carry a member at the full category set: a missing block would be N/A for every candidate --
#: a silent all-TIE no band check can see.
#: Every block name a v2 rule may carry (plan D5; ``Q`` = the gait-motion-quality campaign's motion-
#: quality block, 2026-09-23). The REQUIRED set is derived from each campaign's own rule
#: (:func:`rule_v2_required_blocks`): every block the rule names must carry a member at the full
#: category set -- a named block with no member would be N/A for every candidate, a silent all-TIE
#: no band check can see -- and a name outside this tuple is refused.
#: NOTE(clock-deprecation 2026-09-23, plan step 4): ``RIG`` = the rigidity block (stepping without
#: travel + loaded-foot count; :data:`RIG_BLOCK`), the clock campaign's ``adopt_requires``.
#: NOTE(stride/posture plan step 3, 2026-09-25): ``LEN`` = the stroke/posture screen's mechanism
#: block (side/diag ``steps_per_m``, lower is better; decision 5 -- S's adoption never requires it,
#: so LEN is decisive for the mechanism reading only and no rule may name it in ``adopt_requires``).
RULE_V2_BLOCKS = ("F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs", "Q", "RIG", "LEN")
#: The rigidity block's name. A rule that carries it arms the clock campaign's DECISION machinery
#: (the flag list and its driver park, the RIG z split, the ``arm_class`` field) -- every other
#: rule decides, prints and queues exactly as before.
RIG_BLOCK = "RIG"
#: RIG's two halves (the balanced-RIG rule, settled decision 11): the stepping counts (lower is
#: better) and the loaded-foot count (higher is better). A WIN needs each half of z >= 0.
RIG_STEPS_KEYS = ("steps_per_m", "steps_per_rad")
RIG_SUPPORT_KEYS = ("support_feet",)
#: ``decision_rule.pair_on`` (plan step 4): the s3 outcomes that queue a seed-2 pair instead of
#: deciding -- ``adopt`` (an ADOPT, which a WIN needs POOLED), ``neutral`` (the required block TIE,
#: nothing WORSE) and ``lone_guard`` (the required block BETTER, exactly one guard WORSE, no block
#: WORSE). Both of the last two are a clear REJECT without it.
#: NOTE(stroke formation 2026-09-26): ``always`` -- EVERY decided s3 ADOPT or REJECT queues its
#: seed-2 pair whatever its class (a WORSE block or guard, a MISSING guard included). For a campaign
#: whose gate is read by the orchestrator from a pooled two-seed readout, not by the driver's blocks
#: (the formation seed band equals the effect bar, so a single-seed arm is uninterpretable). A floor
#: or dead_walker REJECT (``_dispatch_floor_reject``) is not a ``decide`` outcome and stays unpaired.
#: Inert unless named: a rule without it decides and pairs exactly as before.
PAIR_ON_CASES = ("adopt", "neutral", "lone_guard", "always")
#: The facing / translation families both unit systems carry (``X`` and ``X_abs``).
RULE_V2_FAMILIES = ("F", "F_mt", "T")


def rule_v2_required_blocks(rule: dict) -> list[str]:
    """The blocks a v2 rule must carry a member in, DERIVED from the rule itself.

    Every block it names; and when it carries the absolute-units system at all (any ``*_abs``
    block), BOTH systems complete over :data:`RULE_V2_FAMILIES` -- an absolute-units candidate is
    scored on the ``_abs`` twins of exactly the ratio blocks, so a family present in one system and
    missing from the other is the silent N/A the check exists for (the facing ladder's six-block
    rule: required = all six, exactly as before). A rule with no ``*_abs`` block (the motion-quality
    ladder's F/F_mt/T/Q -- it screens no cap candidate) requires only what it names.
    """
    named = [str(b) for b in ((rule.get("per_category") or {}).get("blocks") or {})]
    need = set(named)
    if any(b.endswith("_abs") for b in named):
        need |= set(RULE_V2_FAMILIES) | {f"{b}_abs" for b in RULE_V2_FAMILIES}
    return [b for b in RULE_V2_BLOCKS if b in need] + sorted(need - set(RULE_V2_BLOCKS))
#: Stage floor kinds (plan D6). ``survival`` (the pre-suite hold-keyed prefix survival) is refused
#: for campaign use: it read contaminated survival through every earlier hold, trained or not.
FLOOR_KINDS = ("category_survival", "dead_walker", "none")


def _required_block_class(cmpres: dict, req: str) -> str:
    """The ``adopt_requires`` block's class, with an UNMEASURED block named as such: ``classify``
    maps a missing z to TIE (pinned), but a required block with no measured member (k == 0 while
    members are missing) is "MISSING", not a TIE -- the REJECT it yields is then `uncertain`."""
    body = cmpres["blocks"].get(req)
    if not body:
        return "absent"
    if not body.get("na") and body.get("k", 0) == 0 and body.get("missing"):
        return "MISSING"
    return body.get("class") or "absent"


def adopt_requires(rule: dict | None) -> str | None:
    """``decision_rule.adopt_requires``: the block (label) an ADOPT requires to read BETTER, or None.

    The gait-motion-quality campaign sets ``Q``: a win on F/F_mt/T without quieting is not this
    campaign's goal, so Q TIE (or WORSE) is a clear REJECT whatever else is BETTER, an all-TIE
    (Q included) is a clear REJECT with no seed-2 pair, and the pooled decision requires pooled Q
    BETTER too (:func:`decide`, :func:`decide_pooled`). Absent -- every earlier campaign -- the rule
    behaves exactly as before.
    """
    req = (rule or {}).get("adopt_requires")
    return None if req in (None, "") else block_label(str(req))


def _check_rule_v2(rule: dict, reasons: list) -> None:
    """The campaign's ``decision_rule`` must be the per-category rule (``rule_version: 2``, D5).

    Expanded at the FULL category set (``expand_rule_union``), every block of
    :func:`rule_v2_required_blocks` (the rule's OWN set: a campaign without the ``*_abs`` twins
    validates, one with them must be complete in both systems) must carry a member and no block may
    be named outside :data:`RULE_V2_BLOCKS`; the guards must be non-empty; the
    rule's own ``per_category`` templates are validated by ``eval_categories`` (unknown
    categories/keys/composites refused by name). ``adopt_requires`` must name one of the rule's
    ratio-system blocks, and every ``per_category.demoted`` entry must name a member some block or
    guard actually scores (a typo would be a silent no-op). The ``bootstrap`` and ``control_head``
    pins are required too -- ``resolve()`` reads them from the rule, and without ``control_head`` the
    D00s control would default to the first accepted input, now possibly a ``seed_pair`` row.
    """
    if rule_version(rule) != 2:
        reasons.append(f"decision_rule.rule_version must be 2 (the per-category rule, plan 2026-09-21 "
                       f"D5), got {rule.get('rule_version', 1)!r}: a v1 rule is a record of the "
                       "pre-suite ladder, not a campaign spec")
        return
    ec = eval_categories()
    try:
        union = ec.expand_rule_union(rule, ec.CATEGORIES, ec.CATEGORIES)
    except ValueError as exc:
        reasons.append(f"decision_rule (rule_version 2) does not expand: {exc}")
        return
    for block in rule_v2_required_blocks(rule):
        if not ((union.get("blocks") or {}).get(block) or {}).get("metrics"):
            reasons.append(f"decision_rule.per_category.blocks.{block} has no member at the full "
                           "category set (every candidate would read it N/A)")
    extra = sorted(set(union.get("blocks") or {}) - set(RULE_V2_BLOCKS))
    if extra:
        reasons.append(f"decision_rule.per_category.blocks names {extra}, outside {list(RULE_V2_BLOCKS)}")
    if not (union.get("guards") or {}).get("metrics"):
        reasons.append("decision_rule.per_category.guards has no member")
    req = rule.get("adopt_requires")
    if req not in (None, ""):
        ratio_blocks = [b for b, body in ((rule.get("per_category") or {}).get("blocks") or {}).items()
                        if (body or {}).get("units") == "ratio"]
        if str(req) not in ratio_blocks:
            reasons.append(f"decision_rule.adopt_requires names {req!r}, which is not one of the rule's "
                           f"ratio-system blocks {ratio_blocks}")
        # NOTE(stride/posture plan step 3, 2026-09-25): LEN is mechanism evidence only (decision 5).
        if str(req) == "LEN":
            reasons.append("decision_rule.adopt_requires names 'LEN', which is mechanism evidence only "
                           "(stride/posture plan decision 5: an adoption never requires LEN BETTER)")
    pc = rule.get("per_category") or {}
    scored_keys: set = set()
    for body in (pc.get("blocks") or {}).values():
        for c, keys in ((body or {}).get("members") or {}).items():
            scored_keys.update((str(c), str(k)) for k in keys or ())
    for c, keys in ((pc.get("guards") or {}).get("members") or {}).items():
        scored_keys.update((str(c), str(k)) for k in keys or ())
    for c, keys in (pc.get("demoted") or {}).items():
        for k, why in (keys or {}).items():
            if (str(c), str(k)) not in scored_keys:
                reasons.append(f"decision_rule.per_category.demoted.{c}.{k} demotes a member no block or "
                               "guard scores (a typo: the demotion would be a silent no-op)")
            if not str(why or "").strip():
                reasons.append(f"decision_rule.per_category.demoted.{c}.{k} records no reason")
    boot = rule.get("bootstrap") or {}
    if not boot.get("B") or not boot.get("paired_on"):
        reasons.append(f"decision_rule.bootstrap must name B and paired_on, got {boot!r}")
    if not rule.get("control_head"):
        reasons.append("decision_rule.control_head is not pinned (the D00s control would default to "
                       "the first accepted input)")
    # NOTE(clock-deprecation 2026-09-23, plan step 4): pair_on / steps_guard_flag.
    pair_on = rule.get("pair_on")
    if pair_on is not None:
        if not isinstance(pair_on, (list, tuple)) or not set(map(str, pair_on)) <= set(PAIR_ON_CASES):
            reasons.append(f"decision_rule.pair_on must be a list of {list(PAIR_ON_CASES)}, got {pair_on!r}")
        elif {"neutral", "lone_guard"} & set(map(str, pair_on)) and req in (None, ""):
            reasons.append("decision_rule.pair_on names neutral/lone_guard, which are defined on the "
                           "adopt_requires block -- the rule names none")
    if rule.get("steps_guard_flag") not in (None, True, False):
        reasons.append(f"decision_rule.steps_guard_flag must be true or false, got "
                       f"{rule.get('steps_guard_flag')!r}")


_EVAL_ROW_KINDS: dict = {}


def _eval_row_kind(row: str) -> str | None:
    """The manifest ``kind`` of an eval ROW short name (``suite`` -> ``category_suite``), or None."""
    sid = _sid_for(row)
    if sid not in _EVAL_ROW_KINDS:
        try:
            _EVAL_ROW_KINDS[sid] = str(rp._manifest_row(sid)["kind"])
        except Exception:                                                  # noqa: BLE001
            _EVAL_ROW_KINDS[sid] = None
    return _EVAL_ROW_KINDS[sid]


def _known_eval_row(row: str) -> bool:
    return _eval_row_kind(row) is not None


def _check_suite_rows_cover_trained(stage: dict, tc, where: str, reasons: list) -> None:
    """Every suite eval row must hold every category the stage's eval passes as
    ``--trained-categories`` (:func:`_trained_csv`), or the harness refuses the row at eval time
    (``schedule.suite_cli_args``) and the driver halts ``eval_infra`` after its retry wait -- 3000
    trained iterations with no score. Side-speed bisection review, 2026-09-29: the first
    ``side22_25`` row had no ``stand`` category and every 1b stage trains ``stand,side``."""
    csv = _trained_csv(tc.to_dict())
    if not csv:
        return
    S = rp._gait_schedule()
    for row in stage.get("eval") or []:
        if _eval_row_kind(str(row)) != "category_suite":
            continue
        scenarios, _defaults = S.load_manifest(rp.OMNI_MANIFEST)
        sc = next((x for x in scenarios if x.id == _sid_for(str(row))), None)
        if sc is None:
            continue
        try:
            S.suite_cli_args(sc, trained_categories=csv)
        except S.ManifestError as exc:
            reasons.append(f"{where}: eval row {row!r} cannot score this stage ({exc})")


def _check_resolution_input(item: dict, reasons: list, notes: list) -> None:
    """A ``resolution_inputs`` entry is ``run_dir`` XOR ``checkpoint + eval_dir + preset`` (plan D9).

    The checkpoint shape is a suite re-evaluation (``role: pair``, a D00 head) or a rescreen
    (``role: seed_pair``) whose stamped run dir does not exist when the yaml is written: the D00s
    queue item reads the stamp ``lad.reevals`` records (pair) or the newest fresh stamp under
    ``eval_dir`` (seed_pair). Neither shape, or both, is refused by name.
    """
    run_dir = item.get("run_dir")
    shape = [k for k in ("checkpoint", "eval_dir", "preset") if item.get(k) not in (None, "")]
    label = f"resolution input {item.get('head')}/{item.get('row')}"
    if run_dir and shape:
        reasons.append(f"{label}: carries run_dir AND {shape} -- one shape or the other, never both")
        return
    if not run_dir and not shape:
        reasons.append(f"{label}: carries neither run_dir nor checkpoint + eval_dir + preset")
        return
    if run_dir:
        if Path(str(run_dir)).is_absolute():
            reasons.append(f"resolution_inputs run_dir must be a repo-relative path, got {run_dir!r}")
        elif not (REPO / str(run_dir)).exists():
            notes.append(f"resolution input {run_dir} is not on this machine (resolution.py refuses it)")
        return
    missing = [k for k in ("checkpoint", "eval_dir", "preset") if k not in shape]
    if missing:
        reasons.append(f"{label}: the checkpoint shape needs checkpoint + eval_dir + preset; missing "
                       f"{missing}")
        return
    if str(item["preset"]) not in ph.PHASES:
        reasons.append(f"{label}: preset {item['preset']!r} is not one of {sorted(ph.PHASES)}")
    for key in ("checkpoint", "eval_dir"):
        if Path(str(item[key])).is_absolute():
            reasons.append(f"{label}: {key} must be a repo-relative path, got {item[key]!r}")
    if not Path(str(item["checkpoint"])).is_absolute() and not (REPO / str(item["checkpoint"])).exists():
        notes.append(f"{label}: checkpoint {item['checkpoint']} is not on this machine (its suite "
                     "eval cannot run)")
    if item.get("role") not in (None, "pair", "seed_pair", "informational"):
        reasons.append(f"{label}: role {item.get('role')!r} unknown (pair, seed_pair, informational)")


def _check_milestone_reference(stage: str, row: str, ref, reasons: list, notes: list) -> None:
    """A reference row is a recorded run dir (legacy light/ring) or ``{eval_dir, checkpoint}``.

    The suite reference names its re-evaluation ROOT and the head's checkpoint (plan D7/D10): the
    stamp is resolved with the same freshness rule ``load_row`` applies, and the checkpoint is what
    that rule compares against.
    """
    where = f"milestone_reference[{stage}][{row}]"
    if isinstance(ref, dict):
        missing = [k for k in ("eval_dir", "checkpoint") if not ref.get(k)]
        if missing:
            reasons.append(f"{where} must carry eval_dir and checkpoint, missing {missing}")
            return
        for key in ("eval_dir", "checkpoint"):
            if Path(str(ref[key])).is_absolute():
                reasons.append(f"{where}.{key} must be a repo-relative path, got {ref[key]!r}")
        if not (REPO / str(ref["checkpoint"])).exists():
            notes.append(f"{where}.checkpoint {ref['checkpoint']} is not on this machine")
        return
    if not ref or Path(str(ref)).is_absolute():
        reasons.append(f"{where} must be a repo-relative path, got {ref!r}")
    elif not (REPO / str(ref)).exists():
        # A typo here would silently turn M1's only cross-lineage read into "n/a" rather than
        # failing anything, so it is named at load.
        notes.append(f"{where} {ref} is not on this machine: M1's WATCH comparison against it will "
                     "read n/a")


def _check_reference(cands: dict, name: str, ref, reasons: list) -> None:
    """One ``references:`` entry (clock-deprecation plan step 4): ``{chain, seed}`` -- the chain
    exists (default: the reference's own name), the seed is an integer when pinned, and no stage of
    the chain forks this same reference (a run cannot resume from its own head)."""
    where = f"references.{name}"
    if ref is not None and not isinstance(ref, dict):
        reasons.append(f"{where} must be a mapping {{chain, seed}}, got {ref!r}")
        return
    spec = reference_spec(cands, name)
    if spec["chain"] not in (cands.get("chains") or {}):
        reasons.append(f"{where}: chain {spec['chain']!r} is not defined")
        return
    if spec["seed"] is not None:
        try:
            int(spec["seed"])
        except ValueError:
            reasons.append(f"{where}: seed {spec['seed']!r} is not an integer")
    for stage in (chain_by_name(cands, spec["chain"]).get("stages") or []):
        if resume_reference(stage) == name:
            reasons.append(f"{where}: its own chain's stage {stage.get('stage')} resumes from "
                           f"reference {name} -- a run cannot fork its own head")


def _check_stage_categories(stage: dict, env: dict, where: str, reasons: list) -> None:
    """Plan D6's stage fields: ``categories:`` in place of ``holds:``, and the category floor.

    ``categories`` must equal ``sorted(decisive | watch)`` of ``trained_categories`` of the stage's
    composed env (compared as sorted lists), so the yaml cannot claim a category the command
    distribution does not carry. A ``category_survival`` floor needs its row among the stage's
    eval rows and a category-suite row, and two thresholds in (0, 1].
    """
    floor = stage.get("floor") or {}
    kind = str(floor.get("kind"))
    if kind not in FLOOR_KINDS:
        reasons.append(f"{where}: floor kind {floor.get('kind')!r} unknown (one of {list(FLOOR_KINDS)}"
                       + ("; `survival` is the pre-suite hold-keyed floor, replaced by "
                          "category_survival -- plan D6)" if kind == "survival" else ")"))
    if "holds" in stage:
        reasons.append(f"{where}: `holds:` is replaced by `categories:` (plan D6)")
    for row in stage.get("eval") or []:
        if not _known_eval_row(str(row)):
            reasons.append(f"{where}: eval row {row!r} is not in the omni manifest")
    if "categories" not in stage:
        reasons.append(f"{where}: no `categories:` (= sorted(decisive | watch) of the stage's trained "
                       "categories, plan D6)")
    else:
        try:
            tc = eval_categories().trained_categories(env)
        except ValueError as exc:
            reasons.append(f"{where}: the stage's trained categories cannot be derived ({exc})")
        else:
            want = sorted(set(tc.decisive) | set(tc.watch))
            got = sorted(str(c) for c in stage.get("categories") or [])
            if got != want:
                reasons.append(f"{where}: categories {got} but the stage's composed env trains "
                               f"{want} (decisive {sorted(tc.decisive)}, watch {dict(tc.watch)})")
            _check_suite_rows_cover_trained(stage, tc, where, reasons)
    if kind == "category_survival":
        row = str(floor.get("row"))
        if row not in [str(x) for x in (stage.get("eval") or [])]:
            reasons.append(f"{where}: floor row {floor.get('row')!r} is not one of the stage's eval rows")
        elif _eval_row_kind(row) not in (None, "category_suite"):
            reasons.append(f"{where}: floor row {row!r} is not a category-suite row")
        mins = floor.get("min_by_category")
        if mins is not None:
            # An explicit per-category minimum (the checkpoint-fork stages: every category the fork
            # trains was already trained by the head it forks, and no stage before it says so).
            if not isinstance(mins, dict):
                reasons.append(f"{where}: floor min_by_category must map category -> minimum, got {mins!r}")
                mins = {}
            for c, v in mins.items():
                if str(c) not in eval_categories().CATEGORIES:
                    reasons.append(f"{where}: floor min_by_category names {c!r}, not a category")
                val = _num(v)
                if val is None or not 0.0 < val <= 1.0:
                    reasons.append(f"{where}: floor min_by_category.{c} must be a number in (0, 1], "
                                   f"got {v!r}")
        covered = mins is not None and all(str(c) in (mins or {}) for c in (stage.get("categories") or []))
        for key in ("min_new", "min_prev"):
            if covered and floor.get(key) is None:
                continue                  # every category has its own explicit minimum
            val = _num(floor.get(key))
            if val is None or not 0.0 < val <= 1.0:
                reasons.append(f"{where}: floor {key} must be a number in (0, 1], got {floor.get(key)!r}")
    # NOTE(stride/posture plan, audit pass 3, 2026-09-26): `park_categories` (park and flag).
    park = floor.get("park_categories")
    if park is not None:
        cats = [str(c) for c in stage.get("categories") or []]
        if kind != "category_survival":
            reasons.append(f"{where}: floor park_categories needs kind category_survival, got {kind!r}")
        elif (not isinstance(park, (list, tuple)) or not park
              or any(not isinstance(c, str) for c in park)):
            reasons.append(f"{where}: floor park_categories must be a non-empty list of category "
                           f"names, got {park!r}")
        else:
            bad = [c for c in park if c not in cats]
            if bad:
                reasons.append(f"{where}: floor park_categories {bad} are not among the stage's "
                               f"categories {cats} (a typo would silently turn a park into a REJECT)")


def _composed_final_pool(cands: dict, spec: dict, chain_name: str) -> str | None:
    """The final stage's pool string as ``compose_recipe`` renders it for THIS candidate alone.

    The chain's own raw pool, unless the candidate's overlay sets a raw one on that stage -- which
    is what ``compose_recipe``'s stage loop composes, and what the 2a/2b rescale must start from.
    """
    stages = chain_stages(cands, chain_name)
    if not stages:
        return None
    raw = (stages[-1].get("overlay") or {}).get("KRABBY_OMNI_CMD_POOL")
    overlay = spec.get("overlay") or {}
    if spec.get("stages") == "all":
        own = overlay.get("KRABBY_OMNI_CMD_POOL") if isinstance(overlay, dict) else None
    elif isinstance(overlay, dict):
        own = ((overlay.get(str(stages[-1]["stage"])) or {}) or {}).get("KRABBY_OMNI_CMD_POOL")
    else:
        own = None
    return own if own and str(own).strip() else raw


def _teacher_preview(cands: dict, spec: dict, chain_name: str) -> dict:
    """The 2a/2b overlay ``compose_recipe`` would build for THIS candidate alone on this chain.

    Same three steps as the recipe's teacher block: the COMPOSED final-stage pool with every
    ``pool_edit`` applied, rescaled onto ``SLOTS_TEACHER``, then the share edits re-applied so they
    survive the rescale, then the candidate's own ``teacher_overlay`` on top.
    """
    env: dict = {}
    edit = dict(spec.get("pool_edit") or {})
    final_raw = _composed_final_pool(cands, spec, chain_name)
    if final_raw and str(final_raw).strip():
        pool = parse_pool(final_raw)
        if edit:
            pool = apply_pool_edit(pool, edit, require_match=False)
        pool = pool.rescale_to(parse_slots_str(ph.SLOTS_TEACHER))
        if edit and pool_edit_touches_shares(edit):
            pool = apply_pool_edit(pool, {k: v for k, v in edit.items()
                                          if k in ("set_share", "renormalise")},
                                   require_match=False)
        pool = _canonical_pool(pool)          # the preview must render what compose_recipe renders
        env["KRABBY_OMNI_CMD_POOL"] = render_pool(pool)
        env["KRABBY_OMNI_SLOTS"] = slots_str(pool_slots(pool))
    env.update(_teacher_overlay_for(spec, chain_name, stage_names(cands, chain_name)))
    return env


def _teacher_slot_targets(cands: dict, cand: dict) -> list[tuple]:
    """``SLOTS_TEACHER``, plus ``SLOTS_TEACHER`` with this candidate's own share edits applied.

    The union is taken over the chains, because a candidate may declare its ``pool_edit`` under
    ``overlay_by_chain`` for the pool chains while its ``teacher_overlay`` (one raw SLOTS string) is
    shared with a chain that has no pool at all -- C8 is exactly that shape.
    """
    out = [tuple(parse_slots_str(ph.SLOTS_TEACHER))]
    for chain_name in (cands.get("chains") or {}):
        spec = _cand_spec(cand, chain_name)
        edit = dict(spec.get("pool_edit") or {})
        if not edit or not pool_edit_touches_shares(edit):
            continue
        raw = _composed_final_pool(cands, spec, chain_name)
        if not raw or not str(raw).strip():
            continue
        pool = apply_pool_edit(parse_pool(raw), edit, require_match=False)
        pool = pool.rescale_to(parse_slots_str(ph.SLOTS_TEACHER))
        pool = apply_pool_edit(pool, {k: v for k, v in edit.items()
                                      if k in ("set_share", "renormalise")}, require_match=False)
        got = tuple(pool_slots(_canonical_pool(pool)))   # the target is read off the SAME pool
        if not any(all(abs(a - b) <= 1e-6 for a, b in zip(got, have)) for have in out):
            out.append(got)
    return out


def _check_teacher_slots(cands: dict, cand: dict, spec: dict, chain_name: str, where: str,
                         reasons: list) -> None:
    """Part B's 2a/2b projection rule, on the overlay ``compose_recipe`` will actually write.

    Two refusals, both of which cost a 2.4 h promotion window to discover otherwise:

    1. the composed pool and the composed ``KRABBY_OMNI_SLOTS`` must project to the same thing --
       ``_validate_omni_cfg`` refuses a mismatch at config build, i.e. AFTER the whole chain has
       trained. A ``teacher_overlay`` that names SLOTS without the share edit the pool carries (or
       the other way round) is precisely that mismatch;
    2. and that projection must be ``SLOTS_TEACHER`` or ``SLOTS_TEACHER`` with the candidate's own
       ``set_share``/``renormalise`` edits applied (pinned for C8: ``0.05:0.55:0.15:0.25``) -- the
       teacher windows are the promotion's regime mix and are not a place to improvise one.
    """
    try:
        env = _teacher_preview(cands, spec, chain_name)
    except (ValueError, KeyError) as exc:
        reasons.append(f"{where}: the teacher overlay cannot be composed ({exc})")
        return
    raw, slots = env.get("KRABBY_OMNI_CMD_POOL"), env.get("KRABBY_OMNI_SLOTS")
    if not raw and not slots:
        return
    projected = None
    if raw and str(raw).strip():
        try:
            projected = tuple(pool_slots(parse_pool(raw)))
        except ValueError as exc:
            reasons.append(f"{where}: the composed teacher pool does not parse ({exc})")
            return
    declared = None
    if slots:
        try:
            vals = parse_slots_str(slots)
        except ValueError as exc:
            reasons.append(f"{where}: KRABBY_OMNI_SLOTS {slots!r} does not parse ({exc})")
            return
        total = sum(vals)
        declared = tuple(v / total for v in vals) if total > 0 else tuple(vals)
    if projected is not None and declared is not None \
            and any(abs(a - b) > 1e-6 for a, b in zip(projected, declared)):
        reasons.append(f"{where}: the composed pool projects to "
                       f"{tuple(round(p, 6) for p in projected)} but KRABBY_OMNI_SLOTS {slots} "
                       f"normalises to {tuple(round(d, 6) for d in declared)} -- the command cfg "
                       "refuses that mismatch at config build, after the chain has trained")
        return
    got = projected if projected is not None else declared
    targets = _teacher_slot_targets(cands, cand)
    if not any(all(abs(a - b) <= 1e-6 for a, b in zip(got, want)) for want in targets):
        reasons.append(f"{where}: the teacher windows would run slot mix "
                       f"{tuple(round(g, 6) for g in got)}, which is neither SLOTS_TEACHER "
                       f"{tuple(parse_slots_str(ph.SLOTS_TEACHER))} nor SLOTS_TEACHER with this "
                       f"candidate's own share edits {targets[1:]}")


def _check_pool_edit(cands: dict, chain_name: str, edit: dict, where: str, reasons: list) -> None:
    """A ``pool_edit`` must be honourable on the chain it would be applied to, at compose time."""
    math_mod = pool_math()
    stages = chain_stages(cands, chain_name)
    pools = [(str(s["stage"]), (s.get("overlay") or {}).get("KRABBY_OMNI_CMD_POOL")) for s in stages]
    pools = [(name, raw) for name, raw in pools if raw and str(raw).strip()]
    if not pools:
        return                      # a pool-less chain: the candidate's overlay_by_chain covers it
    for key in edit:
        if key not in ("set_share", "renormalise", "ang_scale", "speed_scale"):
            reasons.append(f"{where}: pool_edit key {key!r} unknown "
                           "(set_share, renormalise, ang_scale, speed_scale)")
    bites = {}
    for field in ("ang_scale", "speed_scale"):
        for pattern in (edit.get(field) or {}):
            bites[(field, pattern)] = []
    by_name = {str(s["stage"]): s for s in stages}
    for sname, raw in pools:
        pool = parse_pool(raw)
        stage = by_name[sname]
        for (field, pattern) in bites:
            if any(fnmatch.fnmatchcase(n, str(pattern)) for n in pool.names):
                bites[(field, pattern)].append(sname)
        try:
            edited = apply_pool_edit(pool, edit, require_match=False)
        except ValueError as exc:
            reasons.append(f"{where} stage {sname}: pool_edit cannot be applied ({exc})")
            continue
        if abs(sum(float(w) for w in edited.weights)) <= 0:
            reasons.append(f"{where} stage {sname}: pool_edit zeroes every weight")
        low = [(n, w) for n, w in zip(edited.names, edited.weights)
               if 0.0 < float(w) < math_mod.POOL_FLOOR_SHARE]
        if low:
            reasons.append(f"{where} stage {sname}: pool_edit leaves {low} below POOL_FLOOR_SHARE")
        # THIS stage's effective cfg: a chain whose stages sat on different presets (or whose
        # overlays set different ellipses) would otherwise be checked against the first stage's.
        env = _effective_stage_cfg(cands, stage, dict(stage.get("overlay") or {}))
        ellipse = parse_slots_str(env.get("KRABBY_OMNI_SPEED_ELLIPSE", "0.35:0.25:0.25"))
        min_speed = walk_floor(env) + MIN_CAP_HEADROOM_M_S
        for name, scale in zip(edited.names, edited.speed_scale):
            if float(scale) == 1.0:
                continue
            bound = math_mod.pool_speed_scale_floor(name, ellipse, min_speed)
            if float(scale) <= bound:
                reasons.append(f"{where} stage {sname}: speed_scale {scale} on {name!r} is at or below "
                               f"the walk-floor bound {bound:.4g} (ellipse {ellipse}, min speed "
                               f"{min_speed}) -- every draw of that type would sit under the gait "
                               "clock's stop speed. Widen speed_ellipse or lower lin_vel_clip in the "
                               "same overlay.")
    for (field, pattern), where_hit in bites.items():
        if not where_hit:
            reasons.append(f"{where}: pool_edit {field} pattern {pattern!r} matches no type in ANY "
                           f"stage of chain {chain_name} (stages {[n for n, _ in pools]}) -- a typo, or "
                           "an arm that could never bite")


# ================================================================== the conditional predicate
_TOKEN = re.compile(r"\s*(==|!=|<=|>=|<|>|\(|\)|,|\.[A-Za-z_][\w.]*|[A-Za-z_][\w.]*|"
                    r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _tokenise(text: str) -> list[str]:
    out, pos = [], 0
    text = " ".join(str(text).split())
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"cannot tokenise {text[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


#: The ONLY fields ``decision(<Cid>).<field>`` may name (the plan's fixed vocabulary). A typo here
#: used to compile, evaluate to ``None`` and SKIP the candidate at its rank -- a 3.9-hour window
#: silently written off, recoverable only by a `decide --choose run` nobody knew to type. A
#: load-time refusal costs nothing and `validate_candidates` already reports it.
#: NOTE(clock-deprecation 2026-09-23, plan step 4): ``arm_class`` -- the step-12 class of a RIG
#: rule's decision (WIN / BREAK / RIGID / NEUTRAL / UNCLASSED, :func:`arm_class`), so ``await_if``
#: can read it (``decision(A6000).arm_class == BREAK``). Named ``arm_class``, never ``class``: every
#: comparison record already carries a per-block ``class`` (BETTER/TIE/WORSE/N/A).
DECISION_FIELDS = ("chosen", "confidence", "pooled_confidence", "status", "floor_stage", "arm_class")
#: Likewise for ``recipe.<field>``. ``screen.<key>`` stays open: its keys are `summarize()`'s and a
#: new one is added by the harness, not by this vocabulary.
RECIPE_FIELDS = ("chain", "has_pool")


class _Pred:
    """A parsed conditional: a tiny fixed-vocabulary predicate, never ``eval``.

    The vocabulary is exactly the plan's -- ``decision(<Cid>).chosen|confidence|pooled_confidence|
    status|floor_stage|arm_class``, ``recipe.chain``, ``recipe.has_pool``, ``screen.<key>`` -- with ``and``,
    ``or``, parentheses, the six comparisons, ``in (a, b)`` and ``is [not] None``. It is a parser
    rather than ``eval`` because a candidate's predicate decides whether a 3.9-hour window runs:
    an expression language that can reach anything else is a hole, and an unparseable one must be a
    LOAD-time refusal, not a surprise at dispatch.
    """

    def __init__(self, tokens: list[str]):
        self.toks = tokens
        self.i = 0
        self.tree = self._expr()
        if self.i != len(self.toks):
            raise ValueError(f"trailing tokens {self.toks[self.i:]}")

    # -- parser
    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _take(self, expect=None):
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of predicate")
        if expect is not None and tok != expect:
            raise ValueError(f"expected {expect!r}, got {tok!r}")
        self.i += 1
        return tok

    def _expr(self):
        node = self._term()
        while self._peek() == "or":
            self._take()
            node = ("or", node, self._term())
        return node

    def _term(self):
        node = self._atom()
        while self._peek() == "and":
            self._take()
            node = ("and", node, self._atom())
        return node

    def _atom(self):
        if self._peek() == "(":
            self._take("(")
            node = self._expr()
            self._take(")")
            return node
        left = self._operand()
        tok = self._peek()
        if tok in ("==", "!=", "<", "<=", ">", ">="):
            self._take()
            return ("cmp", tok, left, self._operand())
        if tok == "in":
            self._take()
            self._take("(")
            items = [self._operand()]
            while self._peek() == ",":
                self._take(",")
                items.append(self._operand())
            self._take(")")
            return ("in", left, items)
        if tok == "is":
            self._take()
            negate = False
            if self._peek() == "not":
                self._take()
                negate = True
            self._take("None")
            return ("isnone", left, negate)
        if tok in (None, "and", "or", ")"):
            # A bare REFERENCE is a truth test (``recipe.has_pool``), the one form the plan's own
            # examples use without a comparison. A bare CONSTANT is not: ``screen`` or ``ADOPT`` on
            # its own is a typo, and reading it as "true" would arm a candidate silently.
            if left[0] == "const":
                raise ValueError(f"{left[1]!r} is a bare value, not a condition -- compare it "
                                 "(== / != / < / in / is None) or name a recipe/screen/decision field")
            return ("truth", left)
        raise ValueError(f"expected a comparison after {left!r}, got {tok!r}")

    def _operand(self):
        tok = self._take()
        if tok == "decision":
            self._take("(")
            cid = self._take()
            self._take(")")
            field = self._take()
            if not field.startswith("."):
                raise ValueError(f"decision({cid}) needs a .field, got {field!r}")
            if field[1:] not in DECISION_FIELDS:
                raise ValueError(f"decision({cid}){field} names no known field; the vocabulary is "
                                 f"{DECISION_FIELDS}")
            return ("decision", cid, field[1:])
        if tok in ("(", ")", ","):
            raise ValueError(f"unexpected {tok!r}")
        if tok.startswith("."):
            raise ValueError(f"unexpected {tok!r}")
        try:
            return ("const", float(tok))
        except ValueError:
            pass
        head, _, rest = tok.partition(".")
        if head in ("recipe", "screen") and rest:
            if head == "recipe" and rest not in RECIPE_FIELDS:
                raise ValueError(f"recipe.{rest} names no known field; the vocabulary is "
                                 f"{RECIPE_FIELDS}")
            return (head, rest)
        if tok in ("None", "none"):
            return ("const", None)
        if "." in tok:
            # A dotted token whose head is not `recipe`/`screen` is a typo, not a value: read as a
            # bare string constant it would compare False against everything and SKIP the candidate.
            raise ValueError(f"{tok!r} names no known field; the vocabulary is "
                             f"decision(<Cid>).<field>, recipe.<field> and screen.<summarize key>")
        return ("const", tok)

    # -- evaluation
    def refs(self) -> list[str]:
        out: list[str] = []

        def walk(node):
            if not isinstance(node, tuple):
                return
            if node[0] == "decision":
                if node[1] not in out:
                    out.append(node[1])
                return
            for part in node[1:]:
                if isinstance(part, list):
                    for sub in part:
                        walk(sub)
                else:
                    walk(part)
        walk(self.tree)
        return out

    def evaluate(self, ctx: dict) -> tuple[bool, str]:
        """``(value, why)`` -- ``why`` spells the predicate with its values substituted."""
        value, why = self._eval(self.tree, ctx)
        return bool(value), why

    def _value(self, node, ctx):
        kind = node[0]
        if kind == "const":
            return node[1], ("None" if node[1] is None else
                             (f"{node[1]:g}" if isinstance(node[1], float) else str(node[1])))
        if kind == "decision":
            got = ctx["decision"](node[1]).get(node[2], None)
            return got, f"decision({node[1]}).{node[2]}={got}"
        if kind == "recipe":
            got = (ctx.get("recipe") or {}).get(node[1])
            return got, f"recipe.{node[1]}={got}"
        if kind == "screen":
            got = dget(ctx.get("screen") or {}, node[1])
            return got, f"screen.{node[1]}={fmt_num(got) if _num(got) is not None else got}"
        raise ValueError(f"not an operand: {node!r}")

    def _eval(self, node, ctx):
        kind = node[0]
        if kind in ("and", "or"):
            a, wa = self._eval(node[1], ctx)
            b, wb = self._eval(node[2], ctx)
            return ((a and b) if kind == "and" else (a or b)), f"({wa} {kind} {wb})"
        if kind == "cmp":
            op = node[1]
            a, wa = self._value(node[2], ctx)
            b, wb = self._value(node[3], ctx)
            return _compare(op, a, b, ctx), f"{wa} {op} {wb}"
        if kind == "in":
            a, wa = self._value(node[1], ctx)
            vals, ws = [], []
            for item in node[2]:
                v, w = self._value(item, ctx)
                vals.append(v)
                ws.append(w)
            return any(_compare("==", a, v, ctx) for v in vals), f"{wa} in ({', '.join(ws)})"
        if kind == "isnone":
            a, wa = self._value(node[1], ctx)
            return ((a is not None) if node[2] else (a is None)), f"{wa} is {'not ' if node[2] else ''}None"
        if kind == "truth":
            a, wa = self._value(node[1], ctx)
            return bool(a), wa
        raise ValueError(f"cannot evaluate {node!r}")


def _compare(op: str, a, b, ctx: dict) -> bool:
    """Compare two predicate operands: numbers numerically, stage names by chain order, else as text.

    A comparison with a missing value is FALSE for every operator except ``==``/``!=`` against
    ``None`` -- a candidate whose precondition cannot be shown does not run (the safe direction: the
    cost of a wrong SKIP is one ``decide --choose run``, the cost of a wrong run is a GPU window).
    """
    if op in ("==", "!="):
        eq = _eq(a, b)
        return eq if op == "==" else not eq
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return {"<": na < nb, "<=": na <= nb, ">": na > nb, ">=": na >= nb}[op]
    if isinstance(a, str) and isinstance(b, str):
        # Stage names are ordered by the CHAIN they belong to. The base recipe's own stage list is
        # tried first, then every chain in candidates.yaml -- because the stage a predicate names is
        # usually NOT on the base recipe's chain: `decision(C1).floor_stage >= c4` is evaluated at
        # C1b's rank, where a rejected C1 leaves the base on `scratch_1c` (stages ["1c"]), and a
        # comparison that could not be resolved used to read FALSE, writing C1b off as a SKIP in
        # exactly the case (an infeasible-demand floor trip at c4-c6) the plan created C1b for.
        # Per-chain lists, never a flat union: `scratch_c6`'s single stage is also called `c6`, and
        # a union would order it before `c1..c5`.
        for order in _stage_orders(ctx):
            if a in order and b in order:
                ia, ib = order.index(a), order.index(b)
                return {"<": ia < ib, "<=": ia <= ib, ">": ia > ib, ">=": ia >= ib}[op]
    return False


def _stage_orders(ctx: dict) -> list[list[str]]:
    """Every stage-name ordering a predicate may compare within: the base recipe's, then each chain's."""
    out = []
    own = list(ctx.get("stage_order") or ())
    if own:
        out.append(own)
    for order in ctx.get("stage_orders") or ():
        order = list(order)
        if order and order not in out:
            out.append(order)
    return out


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-12
    return str(a) == str(b)


def compile_predicate(text: str) -> _Pred:
    return _Pred(_tokenise(text))


def conditional_refs(text) -> list[str]:
    """The candidate ids a conditional names (derived at load; the SKIP squash rule closes over them)."""
    if not text or not str(text).strip():
        return []
    try:
        return compile_predicate(text).refs()
    except ValueError:
        return []


def predicate_context(cands: dict, ledger: dict, state: dict, recipe: dict, code: str, view=None) -> dict:
    """The values a conditional is evaluated against: decisions, the recipe, and the screen.

    ``screen`` is the EFFECTIVE recipe's ok s3 screen (Vocabulary), so ``screen.<key>`` always means
    "the number the campaign is standing on right now", never a candidate's own pending result.
    """
    squashed = set((view or {}).get("entries") or ())
    rule = cands.get("decision_rule") or {}

    def decision(cid: str) -> dict:
        entry = newest_entry(ledger, cid, squashed)
        if entry is None:
            return {"chosen": None, "confidence": None, "pooled_confidence": "none", "status": None,
                    "floor_stage": None, "arm_class": None}
        return {"chosen": lc.effective_choice(entry),
                "confidence": entry.get("confidence"),
                # "none" rather than None: the Vocabulary says an absent pair evaluates to `none`, so
                # `pooled_confidence in (uncertain, disputed)` is FALSE rather than unevaluable.
                "pooled_confidence": entry.get("pooled_confidence") or "none",
                "status": entry.get("status"),
                "floor_stage": entry.get("floor_stage") or (entry.get("evidence") or {}).get("floor_stage"),
                # left-for-implementation item 1: the class the step-12 table gives this entry (None
                # under a rule without RIG, and while the entry is still waiting for its pair)
                "arm_class": arm_class(entry, rule)[0]}

    scr = ok_screen(state, code, recipe["recipe_id"], (cands.get("seeds") or {}).get("screen", 3), view)
    return {"decision": decision,
            "recipe": {"chain": recipe.get("chain"), "has_pool": bool(recipe.get("has_pool"))},
            "screen": (scr or {}).get("screen") or {},
            "stage_order": [str(s["stage"]) for s in recipe.get("stages") or []],
            # Every chain's own ordering, so a predicate can compare a stage name that belongs to a
            # chain the BASE recipe is not on (`decision(C1).floor_stage >= c4` at C1b's rank).
            "stage_orders": [[str(s["stage"]) for s in chain_stages(cands, name)]
                             for name in (cands.get("chains") or {})]}


# ================================================================== the decision rule (pure)
#: A metric id is a dotted path into ``summarize()``'s output. These heads are the SAME quantity
#: under two spellings, so a resolution record written against either one resolves: ``ratio_by_hold``
#: is the data-driven re-reading of ``speed_ratio_by_hold`` (identical on the four direction holds)
#: and ``net_yaw_by_hold`` is ``net_yaw_rate_toward`` hoisted. Without this a resolution and a
#: decision rule that disagreed about the spelling would report every metric MISSING -- silently.
METRIC_ALIAS_HEADS = {
    "ratio_by_hold": ("speed_ratio_by_hold", "dir_ratio_by_hold"),
    "speed_ratio_by_hold": ("ratio_by_hold", "dir_ratio_by_hold"),
    "net_yaw_by_hold": ("net_yaw_rate_toward_by_hold",),
    "net_yaw_rate_toward_by_hold": ("net_yaw_by_hold",),
    "yaw_progress_by_hold": ("yaw_progress_rate_by_hold",),
}


def _alias_ids(mid: str) -> list[str]:
    head, _, rest = str(mid).partition(".")
    out = [mid]
    for alt in METRIC_ALIAS_HEADS.get(head, ()):
        out.append(f"{alt}.{rest}" if rest else alt)
    return out


def resolution_metric(resolution: dict, mid: str) -> dict:
    """``resolution.metrics[<mid or an alias>]`` or ``{}``."""
    table = (resolution or {}).get("metrics") or {}
    for key in _alias_ids(mid):
        if key in table:
            return table[key] or {}
    return {}


def guard_resolution_key(resolution: dict, mid: str) -> str:
    """The key a GUARD's ``R_eval`` is read under: ``<mid>@guard`` when the record resolved one (the
    guard's metric is also a block member of the rule, so it has its own one-sided 99 % half-width),
    else ``mid`` -- every guard before the motion-quality campaign."""
    key = eval_categories().guard_key(mid)
    return key if key in ((resolution or {}).get("metrics") or {}) else str(mid)


def resolution_block(resolution: dict, block: str) -> dict:
    return ((resolution or {}).get("blocks") or {}).get(block) or {}


def seed_term(seed_terms: dict, key: str, *, block: bool = False) -> float | None:
    """A seed half-width from a ``D00b`` / ``D00c-<chain>`` ledger entry's ``seed_terms``."""
    table = ((seed_terms or {}).get("blocks" if block else "metrics") or {})
    if block:
        return _num(table.get(key))
    for alt in _alias_ids(key):
        if alt in table:
            return _num(table[alt])
    return None


def metric_sign(mid: str) -> int:
    """+1 higher-is-better, -1 lower-is-better (``eval_categories.metric_sign``, the ONE definition)."""
    return eval_categories().metric_sign(mid)


def _signed_delta(sign: int, a: float, b: float) -> float:
    """``sign * (a - b)`` -- exactly ``a - b`` for every higher-is-better metric (no float moves)."""
    return (a - b) if sign >= 0 else (b - a)


def metric_rows(mids, cand: dict, base: dict, resolution: dict, seed_terms: dict, *,
                cand2: dict | None = None, base2: dict | None = None,
                pooled: bool | None = None, r_key=None) -> list[dict]:
    """One row per metric: the two values, the delta, the half-widths and the standardised z.

    With ``cand2``/``base2`` the delta is the TWO-SEED MEAN delta (a pooled decision), which is why
    both seeds' screens are passed rather than pre-averaged: a metric missing on either seed makes
    the pooled delta MISSING, and averaging first would hide that.

    ``pooled`` is threaded EXPLICITLY by :func:`_screen_compare` rather than inferred from "both
    second screens are present". Inferring it silently produced a single-seed delta judged against a
    pooled (sqrt-2 shrunk) threshold whenever one side had no second screen -- which is exactly the
    anchored-drift case where the anchor recipe has no ok s2 screen. With the flag, a pooled
    comparison that is missing a second screen is MISSING, never quietly half-pooled.

    Every delta is SIGNED by :func:`metric_sign` (the motion-quality campaign, 2026-09-23): ``delta =
    sign * (a - b)``, so a positive delta, a positive block ``z`` and a BETTER always mean "better" --
    a LOWER jitter is a positive delta. ``a``/``b`` stay the raw values; a lower-is-better row carries
    ``lower_is_better: True``. ``r_key`` maps a metric id to the resolution key its ``R_eval`` is read
    under (a guard that is also a block member: ``<mid>@guard``, :func:`guard_rows`).
    """
    rows = []
    if pooled is None:
        pooled = cand2 is not None and base2 is not None
    for mid in mids:
        a, b = _num(dget(cand, mid)), _num(dget(base, mid))
        a2 = _num(dget(cand2, mid)) if (pooled and cand2 is not None) else None
        b2 = _num(dget(base2, mid)) if (pooled and base2 is not None) else None
        rkey = r_key(mid) if r_key is not None else mid
        r_eval = _num(resolution_metric(resolution, rkey).get("R_eval"))
        r_seed = seed_term(seed_terms, mid) or 0.0
        level = resolution_metric(resolution, rkey).get("level")
        sign = metric_sign(mid)
        row = {"metric": mid, "a": a, "b": b, "a2": a2, "b2": b2, "R_eval": r_eval, "R_seed": r_seed,
               "level": level, "delta": None, "z": None, "missing": None}
        if sign < 0:
            row["lower_is_better"] = True
        if rkey != mid:
            row["r_key"] = rkey
        if a is None or b is None or (pooled and (a2 is None or b2 is None)):
            row["missing"] = "value"
        elif r_eval is None or r_eval <= 0:
            row["missing"] = "resolution"
        else:
            d1 = _signed_delta(sign, a, b)
            d2 = _signed_delta(sign, a2, b2) if pooled else None
            row["delta"] = (d1 + d2) / 2.0 if pooled else d1
            row["delta_s3"], row["delta_s2"] = d1, d2
            row["z"] = row["delta"] / r_eval
        rows.append(row)
    return rows


def block_stat(rows: list[dict]) -> dict:
    """``z = mean_i(delta_i / R_eval,i)`` over the rows that HAVE a delta (decision 5)."""
    used = [r for r in rows if r.get("z") is not None]
    z = sum(r["z"] for r in used) / len(used) if used else None
    return {"z": z, "k": len(used), "used": [r["metric"] for r in used],
            "missing": [r["metric"] for r in rows if r.get("z") is None]}


def classify(z, r) -> str:
    """BETTER above +R, WORSE below -R, TIE inside. The boundary is a TIE (pinned by a test)."""
    if z is None or r is None:
        return "TIE"
    if z > r:
        return "BETTER"
    if z < -r:
        return "WORSE"
    return "TIE"


def block_resolution(resolution: dict, seed_terms: dict, block: str, *, pooled: bool = False) -> dict:
    """``R_F = max(R_F^eval, R_F^seed)``; pooled: ``max(R_F^eval/sqrt(2), R_F^seed)`` (decision 6)."""
    r_eval = _num(resolution_block(resolution, block).get("R_eval"))
    r_seed = seed_term(seed_terms, block, block=True) or 0.0
    eval_part = None if r_eval is None else (r_eval / math.sqrt(2.0) if pooled else r_eval)
    if eval_part is None:
        return {"R": r_seed or None, "R_eval": r_eval, "R_seed": r_seed, "pooled": pooled}
    return {"R": max(eval_part, r_seed), "R_eval": r_eval, "R_seed": r_seed, "pooled": pooled}


def survival_guard_category(mid: str) -> str | None:
    """The category of a per-category ``survival`` guard id (``cat.<c>.survival``), else None.

    Only these guards take the binomial floor (plan D5): a survival is a binomial proportion on
    ``n_c`` episodes, so a pair saturated on both heads has a paired ``R_eval`` of 0 and would read
    MISSING -- a REJECT for every candidate -- where the honest resolution is the binomial spread.
    """
    if not str(mid).startswith("cat."):
        return None
    parsed = eval_categories().parse_cat_id(mid)
    if parsed is None or parsed[1] is not None or parsed[2] != "survival":
        return None
    return parsed[0]


def binomial_floor(mid: str, screens, *, cats_n=None) -> dict | None:
    """``{R_binom, p_bar, n_c}`` of a survival guard over the screens it compares, or None.

    ``R_binom = z_0.99 sqrt(2 p(1-p)/n_c)`` with ``p = clip(mean of the survivals, 0.05, 0.95)`` and
    ``n_c = min(cat.<c>.n)`` over the same screens (two single-seed, four pooled) -- computed by
    ``resolution.binomial_guard_floor``, the one definition the D00s null uses too. None for a
    non-survival guard, or when a value or an ``n`` is missing (the value case is MISSING anyway).
    """
    cat = survival_guard_category(mid)
    if cat is None:
        return None
    vals, ns = [], []
    for scr in screens:
        if scr is None:
            continue
        v, n = _num(dget(scr, mid)), _num(dget(scr, f"cat.{cat}.n"))
        if v is None or n is None or n <= 0:
            return None
        vals.append(v)
        ns.append(n)
    if not vals:
        return None
    p_bar, n_c = sum(vals) / len(vals), min(ns)
    r = resolution_tool().binomial_guard_floor(p_bar, n_c)
    return None if r is None else {"R_binom": float(r), "p_bar": p_bar, "n_c": n_c}


def guard_rows(mids, cand, base, resolution, seed_terms, *, cand2=None, base2=None,
               pooled: bool = False, anchor_s2_reused: bool = False) -> list[dict]:
    """Guard rows with their own per-metric R: ``max(R_eval, R_seed, R_binom)``, or the pooled form.

    Guards are compared in their OWN units (a completion fraction, a tripod score), one-sided: a
    guard may only make a candidate worse. ``R_pooled,i = sqrt((R_eval,i/sqrt 2)^2 + R_seed,i^2)``.

    Per-category ``survival`` guards (``cat.<c>.survival``, plan D5) also take the binomial floor
    :func:`binomial_floor` -- ``R_binom`` single-seed; ``R_binom/sqrt 2`` pooled (four screens:
    ``p_bar`` over all four, ``n_c`` their min); ``R_binom * sqrt(3)/2`` for a pooled anchored
    drift whose anchor has no s2 (``anchor_s2_reused``: ``mean(a, a2) - b`` has variance
    ``1.5 sigma^2``). A survival guard whose ``R_eval`` is None or 0 is then scored against
    ``max(R_seed, floor)`` WHATEVER ``R_seed`` is (``R_from_floor: binomial``): MISSING is reserved
    for a missing VALUE. Every other guard keeps the ``R_seed > 0`` rescue only -- an unresolved
    ``cat.move_turn.ratio`` or ``cat_tripod_min`` stays MISSING and never reads as a silent TIE.
    """
    rows = metric_rows(mids, cand, base, resolution, seed_terms, cand2=cand2, base2=base2,
                       pooled=pooled, r_key=lambda m: guard_resolution_key(resolution, m))
    screens = [cand, base] + ([cand2, base2] if pooled else [])
    for row in rows:
        r_eval, r_seed = row["R_eval"], row["R_seed"] or 0.0
        binom = binomial_floor(row["metric"], screens)
        r_floor = None
        if binom is not None:
            factor = ((math.sqrt(3.0) / 2.0) if anchor_s2_reused else (1.0 / math.sqrt(2.0))) \
                if pooled else 1.0
            r_floor = binom["R_binom"] * factor
            row["R_binom"] = binom["R_binom"]
            row["binomial"] = dict(binom, factor=factor)
        survival = survival_guard_category(row["metric"]) is not None
        if row.get("missing") == "resolution" and (r_seed > 0 or (survival and r_floor)):
            # Decision 6 spells a guard's resolution `max(R_eval,i, R_seed,i)`, and a guard is
            # compared in its OWN units -- so a guard the eval bootstrap could not resolve is still
            # scorable against the seed term alone. `metric_rows` marks it MISSING because a BLOCK
            # member with no `R_eval` genuinely leaves the mean (the block statistic IS
            # `delta/R_eval,i`); a guard is not standardised, so it does not. MISSING is about a
            # missing VALUE (decision 5: a zero-denominator conditional survival), not about a
            # missing half-width. Unreachable with the shipped D00 -- all four guards carry an
            # `R_eval` on all three rows -- and kept so the code IS the stated formula.
            sign = metric_sign(row["metric"])
            d1 = None if row["a"] is None or row["b"] is None else _signed_delta(sign, row["a"], row["b"])
            d2 = (None if row["a2"] is None or row["b2"] is None
                  else _signed_delta(sign, row["a2"], row["b2"]))
            if d1 is not None and (not pooled or d2 is not None):
                row["delta"] = (d1 + d2) / 2.0 if pooled else d1
                row["delta_s3"], row["delta_s2"] = d1, (d2 if pooled else None)
                row["missing"] = None
                if survival and r_floor:
                    row["R_from_floor"] = "binomial"      # scored at max(R_seed, the floor)
                else:
                    row["R_from_seed_term_only"] = True
        if r_eval is None:
            row["R"] = r_seed or None
        elif pooled:
            row["R"] = math.sqrt((r_eval / math.sqrt(2.0)) ** 2 + r_seed ** 2)
        else:
            row["R"] = max(r_eval, r_seed)
        if r_floor:
            row["R"] = max(row["R"] or 0.0, r_floor)
        # A guard the resolution carries no half-width for is already MISSING (metric_rows leaves
        # its delta None with `missing: "resolution"`), and decision 5 turns a MISSING guard into a
        # REJECT -- so, unlike a block, it can never read as a silent TIE.
        #
        # ONE-SIDED, so the only two classes are WORSE and TIE: decision 5 makes a guard a
        # constraint ("a guard may only make a candidate worse"), never a reason to adopt. Nothing
        # consumed a guard's `BETTER` for a verdict, but the DECISION line printed it beside the
        # guard and it read as evidence FOR the candidate -- a guard above +R is simply not WORSE.
        row["class"] = ("TIE" if row["delta"] is None or row["R"] is None else
                        ("WORSE" if row["delta"] < -row["R"] else "TIE"))
        row["above_R"] = (row["delta"] is not None and row["R"] is not None
                          and row["delta"] > row["R"])
    return rows


def _rule_block_names(rule: dict) -> list[str]:
    """Every block the rule names, in its own order: an expanded or v1 rule's ``blocks``, a raw v2
    rule's ``per_category.blocks``."""
    if is_rule_v2(rule) and not rule.get("expanded"):
        return list(((rule.get("per_category") or {}).get("blocks") or {}).keys())
    return list((rule.get("blocks") or {}).keys())


def block_names(cand_spec: dict, rule: dict) -> tuple[str, ...]:
    """The blocks a candidate is scored on, DERIVED from the rule actually passed (plan D5).

    One unit system: the ``*_abs`` blocks for a candidate that declares ``absolute_units`` (it moves
    the command cap/ellipse, i.e. the denominator of every ratio), the others otherwise. So a v1
    rule gives ``(F, T)`` / ``(F_abs, T_abs)`` and a v2 rule ``(F, F_mt, T)`` / ``(F_abs, F_mt_abs,
    T_abs)`` -- never a hard-wired tuple: a two-block null reads all-TIE ~0.82, inside the v2 band,
    so a seam that dropped F_mt would be invisible to every calibration check.
    """
    absolute = bool((cand_spec or {}).get("absolute_units"))
    return tuple(b for b in _rule_block_names(rule) if str(b).endswith("_abs") == absolute)


def block_label(block: str) -> str:
    """A block's slot in a comparison record: the ratio-system name (``F_abs`` sits in slot ``F``)."""
    return block[:-len("_abs")] if str(block).endswith("_abs") else str(block)


def _tc_of(screen: dict | None):
    """A screen's recorded ``trained_categories`` (the dict ``_screen_of`` copies), or None."""
    tc = (screen or {}).get("trained_categories")
    return tc if isinstance(tc, dict) and tc.get("decisive") is not None else None


def trained_sets(cand: dict | None, decisive=None, present=None) -> tuple[frozenset, frozenset, dict]:
    """``(decisive, present, watch notes)`` of a v2 comparison -- the CANDIDATE's (plan D5).

    Explicit arguments win (a milestone row has no screen, D7); otherwise the candidate screen's
    ``trained_categories``. A v2 comparison without either is refused by name: an empty decisive
    set would be scored as the all-N/A SKIP, which is a statement about the recipe, not about a
    screen that forgot to carry its sets (``_screen_of`` halts ``eval_mismatch`` first).
    """
    notes: dict = {}
    if decisive is None:
        tc = _tc_of(cand)
        if tc is None:
            raise ValueError("a rule_version 2 comparison needs the candidate's trained categories: "
                             "the screen carries no trained_categories and none were passed")
        decisive, notes = tc.get("decisive"), dict(tc.get("watch") or {})
        if present is None:
            present = tc.get("present")
    dec = frozenset(str(c) for c in decisive or ())
    pres = frozenset(str(c) for c in (present if present is not None else dec)) | dec
    return dec, pres, notes


def _check_decisive_at(raw_rule: dict, scoring: dict, resolution: dict | None, absolute: bool) -> None:
    """Plan D5: a NON-EMPTY block (or ``cat_tripod_min``) narrower than at ``decisive_at`` halts.

    Every published block R and the ``cat_tripod_min`` R were measured on B0's members
    (``resolution.decisive_at``); a comparison whose block keeps only a strict subset of them (a T
    without ``diag``) would be judged at an R that belongs to another statistic. A memberless block
    is N/A and never halts (C1b/C7's F_mt after an adoption).
    """
    dat = (resolution or {}).get("decisive_at")
    if not dat:
        return
    ec = eval_categories()
    ref = ec.expand_rule(raw_rule, dat, dat, absolute_units=absolute)
    narrow = []
    for name, body in (scoring.get("blocks") or {}).items():
        mine = set(body.get("categories") or ())
        want = set(((ref.get("blocks") or {}).get(name) or {}).get("categories") or ())
        if mine and mine < want:
            narrow.append(f"block {name} {sorted(mine)} of {sorted(want)}")
    for name, members in (scoring.get("composite") or {}).items():
        mine, want = set(members or ()), set((ref.get("composite") or {}).get(name) or ())
        if mine and mine < want:
            narrow.append(f"{name} {sorted(mine)} of {sorted(want)}")
    if narrow:
        raise ResolutionMissing(
            "the comparison's decisive set is a strict subset of the resolution's decisive_at "
            f"{sorted(dat)} on " + "; ".join(narrow) + " -- every published R was measured on "
            "decisive_at's members, so no R exists for this narrower statistic")


def scoring_rule(rule: dict, cand_spec: dict, decisive=None, present=None, notes=None) -> dict:
    """The rule a comparison is scored on: a v1 (or already expanded) rule as it stands, a raw v2
    rule expanded at ``decisive``/``present`` in the candidate's own unit system (plan D5)."""
    if not is_rule_v2(rule) or rule.get("expanded"):
        return rule
    return eval_categories().expand_rule(rule, decisive or (), present or (),
                                         absolute_units=bool((cand_spec or {}).get("absolute_units")),
                                         watch_notes=notes)


def _screen_compare(cand_spec: dict, rule: dict, cand: dict, base: dict, resolution: dict,
                    seed_terms: dict, *, cand2=None, base2=None, pooled: bool = False,
                    decisive=None, present=None, anchor_s2_reused: bool = False,
                    watch_notes: dict | None = None) -> dict:
    """The whole standardised comparison of two screens: every block, every guard, the WATCH set.

    Under a v2 rule (plan D5) the rule is expanded at the CANDIDATE's decisive/present categories
    (``decisive``/``present`` when passed, else the candidate screen's ``trained_categories``), the
    shared ``composite_guards`` are injected into shallow copies of every screen compared (four when
    pooled), a block with no decisive member is ``N/A`` (never TIE, never MISSING), and the record
    carries ``decisive``, ``present``, ``untrained {candidate, base}`` and ``na_blocks``. The block
    set is :func:`block_names` of the rule scored -- never a hard-wired pair.
    """
    v2 = is_rule_v2(rule)
    info: dict = {}
    scoring = rule
    if v2:
        ec = eval_categories()
        if decisive is None and rule.get("expanded"):
            # an already-expanded rule carries the sets it was expanded at
            decisive, present = rule.get("decisive") or (), rule.get("present")
        dec, pres, notes = trained_sets(cand, decisive, present)
        if watch_notes:
            # explicit sets come with their own reasons (a milestone row: the promotion stage's)
            notes = dict(watch_notes)
        scoring = scoring_rule(rule, cand_spec, dec, pres, notes)
        if not rule.get("expanded"):
            _check_decisive_at(rule, scoring, resolution, bool(cand_spec.get("absolute_units")))
        cand, base = ec.with_composites(cand, dec), ec.with_composites(base, dec)
        cand2, base2 = ec.with_composites(cand2, dec), ec.with_composites(base2, dec)
        base_tc = _tc_of(base)
        all_cats = list(ec.CATEGORIES)
        info = {"decisive": _ordered_categories(dec), "present": _ordered_categories(pres),
                "untrained": {"candidate": [c for c in all_cats if c not in pres],
                              "base": (None if base_tc is None else
                                       [c for c in all_cats if c not in set(base_tc.get("present") or ())])},
                "category_notes": {c: n for c, n in notes.items() if c in pres and c not in dec}}
        if base_tc is not None:
            base_pres = set(base_tc.get("present") or ())
            for c in info["decisive"]:
                if c not in base_pres:
                    info["category_notes"][c] = "base untrained"
    out = {"blocks": {}, "guards": {}, "absolute_units": bool(cand_spec.get("absolute_units")),
           "pooled": pooled}
    for block in block_names(cand_spec, scoring):
        label = block_label(block)
        body = (scoring.get("blocks") or {}).get(block) or {}
        mids = body.get("metrics") or []
        if v2 and not mids:
            res = block_resolution(resolution, seed_terms, block, pooled=pooled)
            out["blocks"][label] = {"block": block, "rows": [], "z": None, "k": 0, "used": [],
                                    "missing": [], **res, "class": "N/A", "na": True,
                                    "resolution_missing": False, "level": body.get("level"),
                                    "why": "no decisive member (N/A: never TIE, never MISSING)"}
            continue
        rows = metric_rows(mids, cand, base, resolution, seed_terms, cand2=cand2, base2=base2,
                           pooled=pooled)
        stat = block_stat(rows)
        res = block_resolution(resolution, seed_terms, block, pooled=pooled)
        out["blocks"][label] = {"block": block, "rows": rows, **stat, **res,
                                "class": classify(stat["z"], res["R"]),
                                # A block with no R at all (no resolution entry and no seed term)
                                # can never be WORSE, so a REJECT on it is impossible: say so in the
                                # record rather than letting it read as an ordinary TIE.
                                "resolution_missing": res["R"] is None,
                                "level": body.get("level")}
        if v2:
            out["blocks"][label]["categories"] = list(body.get("categories") or [])
    gmids = (scoring.get("guards") or {}).get("metrics") or []
    grows = guard_rows(gmids, cand, base, resolution, seed_terms, cand2=cand2, base2=base2, pooled=pooled,
                       anchor_s2_reused=anchor_s2_reused)
    out["guards"] = {"rows": grows, "level": (scoring.get("guards") or {}).get("level"),
                     "missing": [r["metric"] for r in grows if r["delta"] is None],
                     "worse": [r["metric"] for r in grows if r["class"] == "WORSE"]}
    watch = scoring.get("watch") or []
    out["watch"] = {mid: {"a": _num(dget(cand, mid)), "b": _num(dget(base, mid)),
                          **({"note": watch.get(mid)} if isinstance(watch, dict) else {})}
                    for mid in watch}
    if v2:
        info["na_blocks"] = [k for k, v in out["blocks"].items() if v["class"] == "N/A"]
        out.update(info)
    return out


def _any_worse(cmpres: dict) -> list[str]:
    worse = [f"block {k}" for k, v in cmpres["blocks"].items() if v["class"] == "WORSE"]
    worse += [f"guard {m}" for m in cmpres["guards"]["worse"]]
    return worse


def _better_blocks(cmpres: dict) -> list[str]:
    return [k for k, v in cmpres["blocks"].items() if v["class"] == "BETTER"]


def _scored_blocks(cmpres: dict) -> list[str]:
    """The blocks that are not N/A (a v1 comparison has no N/A block)."""
    return [k for k, v in cmpres["blocks"].items() if v.get("class") != "N/A"]


def standardised_sum(cmpres: dict) -> tuple[float, list[str]]:
    """``sum_b z_b / R_b`` over the non-N/A blocks -- each block in units of its OWN resolution.

    The speculative sign (user decision 2026-09-21, plan D5): a raw ``sum z_b`` would follow F_mt,
    whose recorded seed spread is ~3x F's, so its noise would decide every all-TIE pick. A block
    with no ``z`` (no member measurable) or no ``R`` contributes nothing and is named.
    """
    total, skipped = 0.0, []
    for label, v in cmpres["blocks"].items():
        if v.get("class") == "N/A":
            continue
        z, r = v.get("z"), v.get("R")
        if z is None or r is None or not r > 0:
            skipped.append(label)
            continue
        total += float(z) / float(r)
    return total, skipped


def anchored_drift(cand_spec: dict, rule: dict, cand: dict, anchor: dict | None, resolution: dict,
                   seed_terms: dict, *, cand2=None, anchor2=None, pooled: bool = False,
                   decisive=None, present=None) -> dict:
    """"No worse than the anchor": a precondition of ANY ADOPT, clear or speculative.

    The anchor is the last adoption that was clear at its final level or that the user settled, and
    ``B0`` until one exists (``compose_recipe``), so this is the guard against a chain of
    individually-tiny speculative adoptions drifting away from the best thing the campaign ever
    measured. ``n/a`` only when no anchor SCREEN can be found at all (before the baseline's own
    screen exists).

    At the pooled level the plan's drift is "the candidate's two-seed mean deltas vs the anchor's s3
    (and its s2, paired seed-wise, WHEN the anchor recipe has an ok s2 screen)" -- the anchor's
    second seed is optional, the candidate's two-seed mean is not. So when the anchor has no s2 its
    own s3 stands in as its second seed: the comparison stays pooled (mean(a, a2) - b) instead of
    silently collapsing to the s3 delta against a sqrt-2 shrunk threshold.
    """
    if not anchor:
        return {"verdict": "n/a", "why": "no anchor screen (the baseline is the control)"}
    anchor_seed2 = "own s2 screen"
    reused = False
    if pooled and cand2 is not None and anchor2 is None:
        anchor2, anchor_seed2 = anchor, "its s3 reused (the anchor recipe has no ok s2 screen)"
        reused = True
    # `anchor_s2_reused`: `mean(a, a2) - b` has variance 1.5 sigma^2, so a survival guard's binomial
    # floor is R_binom * sqrt(3)/2 there, not the pooled R_binom / sqrt 2 (plan D5).
    cmpres = _screen_compare(cand_spec, rule, cand, anchor, resolution, seed_terms,
                             cand2=cand2, base2=anchor2, pooled=pooled, decisive=decisive,
                             present=present, anchor_s2_reused=reused)
    worse = _any_worse(cmpres)
    return {"verdict": "FAIL" if worse else "OK", "worse": worse, "pooled": bool(pooled),
            "anchor_seed2": anchor_seed2 if pooled else None,
            "z": {k: v["z"] for k, v in cmpres["blocks"].items()},
            "R": {k: v["R"] for k, v in cmpres["blocks"].items()},
            "why": ("drifted below the anchor on " + ", ".join(worse)) if worse else "no block or guard "
                   "below the anchor"}


def _v2_evidence(cmpres: dict) -> dict:
    """The per-category fields a v2 decision records (plan D5): what was decisive, what either side
    never trained, and every N/A block. Empty for a v1 comparison."""
    if "decisive" not in cmpres:
        return {}
    return {"decisive": cmpres.get("decisive"), "present": cmpres.get("present"),
            "untrained": cmpres.get("untrained"), "na_blocks": cmpres.get("na_blocks"),
            "category_notes": cmpres.get("category_notes")}


def decide(screen: dict, base: dict, resolution: dict, *, cand_spec: dict, rule: dict,
           seed_terms: dict | None = None, anchor_screen: dict | None = None,
           anchor_label: str | None = None, resolution_id: str = "D00",
           decisive=None, present=None) -> dict:
    """The seed-3 decision: :func:`_decide_s3`, then ``decision_rule.pair_on`` (:func:`apply_pair_on`).

    A pure function of two screens and the resolution record; without ``pair_on`` it IS
    :func:`_decide_s3`'s answer, field for field.
    """
    out = _decide_s3(screen, base, resolution, cand_spec=cand_spec, rule=rule, seed_terms=seed_terms,
                     anchor_screen=anchor_screen, anchor_label=anchor_label,
                     resolution_id=resolution_id, decisive=decisive, present=present)
    return apply_pair_on(out, rule)


def apply_pair_on(out: dict, rule: dict | None) -> dict:
    """``decision_rule.pair_on: [adopt, neutral, lone_guard]`` (clock-deprecation plan step 4).

    A DECIDED seed-3 outcome in one of the named cases queues a seed-2 pair instead: the entry is
    ``speculative`` (``pair`` True) until it is pooled, and the pooled decision is final. The cases:

    * ``adopt`` -- a clear ADOPT (a WIN needs a POOLED ADOPT);
    * ``neutral`` -- the ``adopt_requires`` block (RIG) TIE with nothing WORSE (``decide`` makes it a
      clear REJECT under ``adopt_requires``);
    * ``lone_guard`` -- the required block BETTER, exactly ONE guard WORSE and no block WORSE (a clear
      REJECT on that guard without it);
    * ``always`` (stroke formation, 2026-09-26) -- every decided ADOPT or REJECT, whatever its class
      (checked last, so a line that matches a narrower case records that case).

    Neither of the last two is paired while a guard is MISSING, or while ANY member of the required
    block (RIG) is MISSING: a pair cannot supply a metric the row never produced (``decide``'s own
    rule), so that REJECT stays decided. The s3 ``confidence``
    is kept (it is that screen's classification); ``evidence.pair_on`` names the case.
    """
    cases = set(str(c) for c in ((rule or {}).get("pair_on") or ()))
    if not cases or out.get("status") != "decided" or out.get("chosen") not in ("ADOPT", "REJECT"):
        return out
    cmpres = (out.get("evidence") or {}).get("seed3") or {}
    if not cmpres.get("blocks"):
        # NOTE(stroke formation 2026-09-26): `always` pairs even a line with no scored block.
        return _paired(out, "always") if "always" in cases else out
    worse_blocks = [k for k, v in cmpres["blocks"].items() if v.get("class") == "WORSE"]
    guards = cmpres.get("guards") or {}
    worse_guards, guard_missing = list(guards.get("worse") or []), list(guards.get("missing") or [])
    req = adopt_requires(rule)
    req_cls = _required_block_class(cmpres, req) if req is not None else None
    req_missing = bool((cmpres["blocks"].get(req) or {}).get("missing")) if req is not None else False
    case = None
    if "adopt" in cases and out["chosen"] == "ADOPT":
        case = "adopt"
    elif (out["chosen"] == "REJECT" and not worse_blocks and not guard_missing and not req_missing
          and req is not None):
        if "neutral" in cases and req_cls == "TIE" and not worse_guards:
            case = "neutral"
        elif "lone_guard" in cases and req_cls == "BETTER" and len(worse_guards) == 1:
            case = "lone_guard"
    if case is None and "always" in cases:
        case = "always"
    if case is None:
        return out
    return _paired(out, case)


def _paired(out: dict, case: str) -> dict:
    """``out`` turned into a speculative line with its seed-2 pair queued (``evidence.pair_on``)."""
    out = dict(out, status="speculative", pair=True)
    out["evidence"] = dict(out.get("evidence") or {}, pair_on=case)
    out["why"] = (f"{out.get('why')}; pair_on {case}: a seed-2 pair is queued -- the entry stays "
                  "speculative until it is pooled, and the pooled decision is final")
    return out


def _decide_s3(screen: dict, base: dict, resolution: dict, *, cand_spec: dict, rule: dict,
               seed_terms: dict | None = None, anchor_screen: dict | None = None,
               anchor_label: str | None = None, resolution_id: str = "D00",
               decisive=None, present=None) -> dict:
    """The seed-3 decision (decision 5), a pure function of two screens and the resolution record.

    Returns the ledger fields: ``chosen`` / ``alternative`` / ``confidence`` / ``status`` / ``pair``
    (whether a confirming seed-2 pair is queued) / ``evidence``. Nothing here reads the ledger, the
    clock or the disk, which is what makes a decision re-derivable from the records months later.

    Under a v2 rule ``decisive``/``present`` default to the candidate screen's trained categories
    (plan D5), and one branch is new: when EVERY scored block is N/A (the candidate trained no turn,
    move+turn or translation category at the suite's demand) the decision is a terminal ``SKIP``
    (``confidence: none``, ``evidence.source: no_decisive_block``, ``alternative: ADOPT``, no pair):
    a narrow recipe is never rejected for being narrow, and the DECISION line quotes the overrule.
    It comes after the WORSE and MISSING-guard branches -- a fall regression still rejects.
    """
    seed_terms = seed_terms or {}
    cmpres = _screen_compare(cand_spec, rule, screen, base, resolution, seed_terms,
                             decisive=decisive, present=present)
    worse = _any_worse(cmpres)
    better = _better_blocks(cmpres)
    guard_missing = cmpres["guards"]["missing"]
    block_missing = sorted({m for v in cmpres["blocks"].values() for m in v["missing"]})
    drift = anchored_drift(cand_spec, rule, screen, anchor_screen, resolution, seed_terms,
                           decisive=decisive, present=present)

    evidence = {"seed3": cmpres, "anchored_drift": drift, "missing": {"guards": guard_missing,
                                                                     "blocks": block_missing},
                "resolution_id": resolution_id, "anchor": anchor_label,
                "levels": {"blocks": {k: v.get("level") for k, v in cmpres["blocks"].items()},
                           "guards": cmpres["guards"]["level"]},
                **_v2_evidence(cmpres)}
    out = {"options": list(CANDIDATE_OPTIONS), "evidence": evidence, "pair": False}

    if worse:
        out.update(chosen="REJECT", alternative="ADOPT", confidence="clear", status="decided",
                   why="WORSE on " + ", ".join(worse))
        return out
    if guard_missing:
        # A pair cannot supply a metric the row never produced, so no pair is queued: the answer is
        # a re-run, which the user asks for with `decide <Cid> --choose run`.
        out.update(chosen="REJECT", alternative="ADOPT", confidence="uncertain", status="decided",
                   why="guard MISSING: " + ", ".join(guard_missing))
        return out
    if cmpres["blocks"] and not _scored_blocks(cmpres):
        # Plan D5: all scored blocks N/A -> a TERMINAL SKIP. Not `undecided` (that would re-screen
        # forever) and not a REJECT (the recipe is narrow, not worse); squash_view's cause (b)
        # exempts it (it answers the screen, not the predicate), causes (a) and (c) still apply.
        evidence["source"] = "no_decisive_block"
        trained = ", ".join(cmpres.get("decisive") or []) or "none"
        out.update(chosen="SKIP", alternative="ADOPT", confidence="none", status="decided",
                   why=f"no decisive block (trained: {trained})")
        return out
    req = adopt_requires(rule)
    if req is not None:
        req_cls = _required_block_class(cmpres, req)
        evidence["adopt_requires"] = {"block": req, "class": req_cls}
        if req_cls != "BETTER":
            # The campaign's adoption rule (`adopt_requires`): an ADOPT needs this block BETTER. Q TIE
            # while another block is BETTER is a clear REJECT (a win without quieting is not this
            # campaign's goal), and so is an all-TIE -- decided, no seed-2 pair: a pair could only
            # turn a TIE into a BETTER, and the orchestrator's call is that an unresolved quieting is
            # not worth a confirming window. WORSE and a MISSING guard were answered above.
            what = (f"blocks {better} BETTER but {req} {req_cls}" if better else
                    f"all TIE ({req} {req_cls})")
            # An UNMEASURED required block (N/A, MISSING, absent) is not evidence against the
            # candidate: the REJECT stands but reads `uncertain`, never `clear` (review 2026-09-23).
            conf = "clear" if req_cls in ("TIE", "WORSE") else "uncertain"
            out.update(chosen="REJECT", alternative="ADOPT", confidence=conf, status="decided",
                       why=f"{what} -- ADOPT requires {req} BETTER (adopt_requires)")
            return out
    if better:
        if drift["verdict"] == "FAIL":
            out.update(chosen="REJECT", alternative="ADOPT", confidence="uncertain",
                       status="speculative", pair=True,
                       why=f"blocks {better} BETTER but anchored drift FAIL ({drift['why']})")
            return out
        if block_missing:
            out.update(chosen="ADOPT", alternative="REJECT", confidence="uncertain",
                       status="speculative", pair=True,
                       why=f"blocks {better} BETTER with MISSING members {block_missing}")
            return out
        out.update(chosen="ADOPT", alternative="REJECT", confidence="clear", status="decided",
                   why=f"blocks {better} BETTER, nothing WORSE, drift {drift['verdict']}")
        return out

    # all TIE -> uncertain: a speculative pick, and the pair runs next.
    pick, why = _speculative_pick(cand_spec, cmpres)
    if pick == "ADOPT" and drift["verdict"] == "FAIL":
        pick, why = "REJECT", f"{why}; anchored drift FAIL ({drift['why']}) forbids an ADOPT"
    out.update(chosen=pick, alternative=("REJECT" if pick == "ADOPT" else "ADOPT"),
               confidence="uncertain", status="speculative", pair=True,
               why=f"all TIE -- {why}")
    return out


SIGN_FORMULA = "sign(Σ z_b/R_b)"


def _speculative_pick(cand_spec: dict, cmpres: dict) -> tuple[str, str]:
    """The all-TIE pick: the candidate's declared ``speculative_default``, else ``sign(Σ z_b/R_b)``.

    The sum runs over the non-N/A blocks, each in units of its OWN resolution (user decision
    2026-09-21, plan D5): a large raw ``z_F_mt`` inside its (seed-dominated) R must not outvote F
    and T. Exact tie -> the incumbent.
    """
    default = cand_spec.get("speculative_default")
    if default == "candidate":
        return "ADOPT", "speculative_default: candidate"
    if default == "incumbent":
        return "REJECT", "speculative_default: incumbent"
    total, skipped = standardised_sum(cmpres)
    tail = f" (no z or R on {skipped})" if skipped else ""
    if total > 0:
        return "ADOPT", f"{SIGN_FORMULA} = +{total:.3f}{tail}"
    if total == 0:
        return "REJECT", f"{SIGN_FORMULA} = 0.000 (exact tie -> incumbent){tail}"
    return "REJECT", f"{SIGN_FORMULA} = {total:.3f}{tail}"


def decide_pooled(screen_s3: dict, screen_s2: dict, base_s3: dict, base_s2: dict, resolution: dict, *,
                  cand_spec: dict, rule: dict, prior: dict, seed_terms: dict | None = None,
                  anchor_s3: dict | None = None, anchor_s2: dict | None = None,
                  anchor_label: str | None = None, resolution_id: str = "D00b",
                  floor_tripped: bool = False, decisive=None, present=None) -> dict:
    """The pooled (two-seed) decision -- decision 7. Every outcome is TERMINAL.

    Pick order: a pooled CLEAR verdict; a MISSING guard (decision 5's rule, terminal REJECT); else a
    DISAGREEMENT between the two seeds' own signs (pick the incumbent, ``pooled_confidence:
    disputed``); else the candidate's ``speculative_default``; else the pooled sign (exact tie ->
    incumbent). Same pick as the s3 decision -> ``final``; a different pick -> the driver overrules
    ITSELF (``overruled``, ``override.by: driver``), which is what triggers SQUASH of everything
    screened on the wrong branch.

    Every sign is ``sign(Σ z_b/R_b)`` over the non-N/A blocks (plan D5): a seed's own sign in units
    of its single-seed block R, the pooled fallback in units of the pooled R. Under a v2 rule the
    decisive/present sets are the CANDIDATE's (its s3 screen's, unless passed) for all four screens.
    """
    seed_terms = seed_terms or {}
    if is_rule_v2(rule) and decisive is None:
        decisive, present, _notes = trained_sets(screen_s3)
    cmpres = _screen_compare(cand_spec, rule, screen_s3, base_s3, resolution, seed_terms,
                             cand2=screen_s2, base2=base_s2, pooled=True, decisive=decisive,
                             present=present)
    worse = _any_worse(cmpres)
    better = _better_blocks(cmpres)
    drift = anchored_drift(cand_spec, rule, screen_s3, anchor_s3, resolution, seed_terms,
                           cand2=screen_s2, anchor2=anchor_s2, pooled=True, decisive=decisive,
                           present=present)

    per_seed = {}
    for label, cand_row, base_row in (("s3", screen_s3, base_s3), ("s2", screen_s2, base_s2)):
        single = _screen_compare(cand_spec, rule, cand_row, base_row, resolution, seed_terms,
                                 decisive=decisive, present=present)
        total, _skipped = standardised_sum(single)
        per_seed[label] = {"score": total, "formula": SIGN_FORMULA,
                           "sign": (1 if total > 0 else (-1 if total < 0 else 0)),
                           "blocks": {k: {"z": v["z"], "R": v["R"], "class": v["class"]}
                                      for k, v in single["blocks"].items()}}
    disagree = per_seed["s3"]["sign"] * per_seed["s2"]["sign"] < 0

    confidence, pick, why = "uncertain", None, ""
    guard_missing = cmpres["guards"]["missing"]
    block_missing = sorted({m for v in cmpres["blocks"].values() for m in v["missing"]})
    req = adopt_requires(rule)
    req_cls = None if req is None else _required_block_class(cmpres, req)
    if worse:
        confidence, pick, why = "clear", "REJECT", "pooled WORSE on " + ", ".join(worse)
    elif guard_missing:
        # Decision 5's guard rule, at the pooled level too: a guard the pair cannot measure (a
        # zero-denominator conditional survival on either seed) is a REJECT with confidence
        # uncertain, never a TIE that can carry an ADOPT. There is no pair left to queue -- a pooled
        # outcome is terminal -- so the answer is `decide <Cid> --choose run`.
        confidence, pick, why = "uncertain", "REJECT", ("pooled guard MISSING: "
                                                        + ", ".join(guard_missing))
    elif req is not None and req_cls != "BETTER":
        # `adopt_requires` at the pooled level: the pooled ADOPT needs the required block pooled
        # BETTER too, whatever else is BETTER and whatever the two seeds' signs say.
        what = (f"pooled blocks {better} BETTER but {req} {req_cls}" if better else
                f"pooled all TIE ({req} {req_cls})")
        confidence = "clear" if req_cls in ("TIE", "WORSE") else "uncertain"   # unmeasured != evidence
        pick, why = "REJECT", f"{what} -- ADOPT requires {req} BETTER (adopt_requires)"
    elif better and drift["verdict"] != "FAIL" and not block_missing:
        confidence, pick, why = "clear", "ADOPT", f"pooled blocks {better} BETTER, nothing WORSE"
    elif disagree:
        # Decision 7's order, exactly: pooled CLEAR -> its pick; ELSE the two seeds' own signs
        # disagree -> disputed; else speculative_default; else the pooled sign (PLAN_AMENDMENTS 23).
        # The MISSING-member cap below is NOT a clear verdict, so it does not come first: with it
        # ahead of this branch, two seeds that disagreed in sign could still be recorded ADOPT
        # (uncertain) instead of `disputed -> incumbent`. A drift FAIL only ever forbids an ADOPT,
        # and the incumbent is the pick here anyway, so the disagreement is what the record says.
        confidence, pick, why = "disputed", "REJECT", (
            f"the two seeds disagree in {SIGN_FORMULA} (s3 {per_seed['s3']['score']:+.3f}, "
            f"s2 {per_seed['s2']['score']:+.3f}) -> the incumbent stands")
        if better and block_missing:
            why += (f"; blocks {better} read BETTER but MISSING members {block_missing} left the "
                    "mean, so that verdict was not clear")
        if drift["verdict"] == "FAIL":
            why += f"; pooled anchored drift also FAIL ({drift['why']})"
    elif better and drift["verdict"] != "FAIL":
        # Decision 5's block rule applies at the pooled level too: "MISSING in F/T -> that metric
        # leaves the block mean and the best outcome is a SPECULATIVE ADOPT". A pooled outcome is
        # terminal, so the cap lands on `pooled_confidence` rather than on the status -- which
        # matters because `_anchor_qualifies` treats `pooled_confidence: clear` as an anchor, and a
        # block that lost members was never measured on the holds the campaign exists to fix.
        confidence, pick = "uncertain", "ADOPT"
        why = (f"pooled blocks {better} BETTER, nothing WORSE; MISSING block members "
               f"{block_missing} left the mean, so the outcome is capped at uncertain (it cannot "
               "become the anchor)")
    elif better and drift["verdict"] == "FAIL":
        confidence, pick, why = "uncertain", "REJECT", (f"pooled blocks {better} BETTER but pooled "
                                                       f"anchored drift FAIL ({drift['why']})")
    else:
        pick, why = _speculative_pick(cand_spec, cmpres)
        if pick == "ADOPT" and drift["verdict"] == "FAIL":
            pick, why = "REJECT", f"{why}; pooled anchored drift FAIL ({drift['why']})"
        why = f"pooled all TIE -- {why}"
    if floor_tripped and confidence != "disputed":
        # A FLOOR trip on the seed-2 / twin / batched run is WATCH: the run still finished and is
        # pooled normally, but the pair is flagged (decision 7) -- the pick follows the pooled outcome.
        confidence = "disputed"
        why += "; FLOOR tripped on a non-candidate kind (WATCH, pooled normally)"

    prior_pick = lc.effective_choice(prior) if prior else None
    same = (prior_pick == pick)
    out = {"chosen": pick, "alternative": ("REJECT" if pick == "ADOPT" else "ADOPT"),
           "pooled_confidence": confidence, "status": "final" if same else "overruled",
           "self_overrule": not same, "prior": prior_pick, "why": why,
           "options": list(CANDIDATE_OPTIONS),
           "evidence": {"pooled": cmpres, "per_seed": per_seed, "anchored_drift": drift,
                        "resolution_id": resolution_id, "anchor": anchor_label,
                        "missing": {"guards": guard_missing, "blocks": block_missing},
                        # Decision 6: "Levels are recorded and cited in EVERY DECISION line." The
                        # single-seed evidence carried them and the pooled one did not, so every
                        # pooled DECISION and SELF-OVERRULE line printed `@None/None` -- the one
                        # field that says what standard the pick was made against. `pooled_R` is
                        # the block threshold actually used (max(R_F^eval/sqrt2, R_F^seed)).
                        "levels": {"blocks": {k: v.get("level") for k, v in cmpres["blocks"].items()},
                                   "guards": cmpres["guards"]["level"],
                                   "pooled_R": {k: v.get("R") for k, v in cmpres["blocks"].items()}},
                        **({"adopt_requires": {"block": req, "class": req_cls}} if req is not None else {}),
                        **_v2_evidence(cmpres)}}
    return out


# ================================================================== the RIG rule: class, split, flags
# NOTE(clock-deprecation 2026-09-23, plan step 4). Everything in this section is armed ONLY by a rule
# that carries block RIG (:func:`flags_enabled`); for every other rule it returns None / [] and no
# record, line or queue item changes.
#: The margin flags of settled decision 7, each against the matched clock-held twin (the control):
#: ``(category key or "*walk" / "*all", key, drop)`` -- a flag when the candidate sits at least
#: ``drop`` below the control.
MARGIN_FLAGS = (("turn", "facing_closed", 0.05), ("move_turn", "facing_closed", 0.10),
                ("*all", "survival", 0.05), ("*walk", "ratio", 0.05))
#: The walking categories whose speed ratio the margin flag reads (the T categories + move_turn).
WALK_CATEGORIES = ("fwd", "side", "back", "diag", "move_turn")
#: taps up > 25 %; swing_apex down >= 30 % (lift loss); support_feet down > 10 %; and, with
#: ``decision_rule.steps_guard_flag``, steps per m / rad adverse (up) by > 15 % of the control's.
TAPS_UP_FRAC = 0.25
APEX_DOWN_FRAC = 0.30
SUPPORT_DOWN_FRAC = 0.10
STEPS_UP_FRAC = 0.15
TAP_KEYS = {"fwd": "taps_per_m", "side": "taps_per_m", "back": "taps_per_m", "diag": "taps_per_m",
            "turn": "taps_per_rad", "move_turn": "taps_per_rad"}
STEPS_GUARD_KEYS = {"fwd": "steps_per_m", "side": "steps_per_m", "back": "steps_per_m",
                    "diag": "steps_per_m", "turn": "steps_per_rad"}
#: The two flags that park at the SEED-3 line even while a pair is queued (never deferred).
#: ``floor_park`` (stride/posture 2026-09-26): a floor tripped only on ``floor.park_categories``.
IMMEDIATE_FLAGS = ("removal", "pilot", "floor_park")


def flags_enabled(rule: dict | None) -> bool:
    """Does this decision rule carry block RIG (the clock campaign's DECISION machinery)?"""
    return RIG_BLOCK in _rule_block_names(rule or {})


def _rig_key(mid: str) -> str | None:
    parsed = eval_categories().parse_cat_id(str(mid)) if str(mid).startswith("cat.") else None
    return parsed[2] if parsed else None


def rig_split(block: dict | None) -> dict | None:
    """RIG's ``z`` split into its steps part and its support_feet part (additive: ``z = steps +
    support + other``, each part = its members' ``sum(delta_i / R_i) / k``). A part with no measured
    member is None. None for an absent block."""
    if not block or block.get("class") == "N/A":
        return None
    rows = [r for r in block.get("rows") or [] if r.get("z") is not None]
    k = len(rows)
    out = {"z": block.get("z"), "k": k, "class": block.get("class")}
    for part, keys in (("steps", RIG_STEPS_KEYS), ("support", RIG_SUPPORT_KEYS)):
        mine = [r for r in rows if _rig_key(r["metric"]) in keys]
        out[f"k_{part}"] = len(mine)
        out[part] = (sum(r["z"] for r in mine) / k) if (mine and k) else None
    return out


def rig_split_text(split: dict | None) -> str:
    if not split:
        return "RIG z split: n/a (no RIG block on this line)"
    part = lambda p: ("n/a" if split.get(p) is None else f"{split[p]:+.3f}")  # noqa: E731
    return (f"RIG z split: steps {part('steps')} (k {split.get('k_steps')}) + support "
            f"{part('support')} (k {split.get('k_support')}) of z {fmt_num(split.get('z'))} "
            f"({split.get('class')}; a WIN needs both parts >= 0)")


def arm_class(entry: dict | None, rule: dict | None) -> tuple[str | None, str]:
    """The step-12 class of one decision (plan step 12; clock-deprecation step 4). ``(class, why)``.

    Read on the POOLED line when one exists, else the seed-3 line; all against the control (X0).
    Precedence BREAK > RIGID > NEUTRAL, with the user's own answer first:

    * UNCLASSED -- SKIPped, user-overruled, RIG N/A or MISSING, a guard MISSING, or anything the
      table does not name (e.g. an s3 ADOPT confirmed before its pair pooled) -> ask at S4;
    * BREAK -- F, F_mt or T (any non-RIG block) WORSE, any guard WORSE, or a floor (the seed-3
      screen's FLOOR REJECT, or a floor its seed-2 run tripped);
    * RIGID -- RIG WORSE and nothing else WORSE;
    * NEUTRAL -- nothing WORSE and RIG TIE, or a pooled ADOPT with a negative RIG part (balanced-RIG);
    * WIN -- a pooled ADOPT with both RIG parts >= 0 and removal verification passed on every run.

    ``(None, why)`` under a rule without RIG, for no entry, and while the entry is speculative (its
    pair is queued: the pooled line decides). A driver self-overrule (``decide_pooled``'s
    ``overruled``) is classed on its pooled line like any other -- it is NOT user-overruled (item 12).
    """
    if not flags_enabled(rule):
        return None, "the rule carries no RIG block"
    if not entry:
        return None, "no decision"
    status = entry.get("status")
    if status == "speculative":
        return None, "pending: its seed-2 pair is queued (the pooled line decides)"
    if status == "skipped" or entry.get("chosen") == "SKIP":
        return "UNCLASSED", "SKIPped"
    if status == "overruled" and (entry.get("override") or {}).get("by") == "user":
        return "UNCLASSED", f"user-overruled to {lc.effective_choice(entry)}"
    ev = entry.get("evidence") or {}
    if entry.get("floor_stage") or ev.get("floor"):
        return "BREAK", f"floor at {entry.get('floor_stage') or ev.get('floor_stage')}"
    if ev.get("s2_floor_watch"):
        # the pair's own run tripped a floor (WATCH for the pooled pick, a floor for the class)
        return "BREAK", f"floor on the seed-2 run: {'; '.join(map(str, ev['s2_floor_watch']))}"
    pooled = bool(ev.get("pooled"))
    cmpres = ev.get("pooled") or ev.get("seed3") or {}
    blocks = cmpres.get("blocks") or {}
    if not blocks:
        return "UNCLASSED", "no comparison recorded"
    worse = [f"block {k}" for k, v in blocks.items() if k != RIG_BLOCK and v.get("class") == "WORSE"]
    worse += [f"guard {m}" for m in (cmpres.get("guards") or {}).get("worse") or []]
    line = "pooled" if pooled else "seed-3"
    if worse:
        return "BREAK", f"WORSE on the {line} line: {', '.join(worse)}"
    rig = _required_block_class(cmpres, RIG_BLOCK)
    if rig in ("N/A", "MISSING", "absent"):
        return "UNCLASSED", f"RIG {rig} on the {line} line"
    missing = list((cmpres.get("guards") or {}).get("missing") or [])
    if missing:
        return "UNCLASSED", f"guard MISSING on the {line} line: {', '.join(missing)}"
    if rig == "WORSE":
        return "RIGID", f"RIG WORSE, nothing else WORSE ({line} line)"
    if rig == "TIE":
        return "NEUTRAL", f"RIG TIE, nothing WORSE ({line} line)"
    if pooled and lc.effective_choice(entry) == "ADOPT":
        split = rig_split(blocks.get(RIG_BLOCK))
        parts = [(p, (split or {}).get(p)) for p in ("steps", "support")]
        if any(v is None for _p, v in parts):
            return "UNCLASSED", "a pooled ADOPT whose RIG part is unmeasured (" + ", ".join(
                p for p, v in parts if v is None) + ")"
        negative = [f"{p} {v:+.3f}" for p, v in parts if v < 0]
        if negative:
            return "NEUTRAL", f"a pooled ADOPT with a negative RIG part ({', '.join(negative)}; balanced-RIG)"
        removal = ev.get("removal") or {}
        runs = [r for r in removal.values() if isinstance(r, dict)]
        if not runs:
            return "UNCLASSED", "a pooled ADOPT whose removal verification is not recorded"
        bad = [f"{k}" for k, r in removal.items() if isinstance(r, dict) and not r.get("ok")]
        if bad:
            return "UNCLASSED", f"a pooled ADOPT whose removal verification FAILED ({', '.join(bad)})"
        return "WIN", "a pooled ADOPT, both RIG parts >= 0, removal verified"
    return "UNCLASSED", (f"RIG BETTER, nothing WORSE, but no pooled ADOPT ({line} line, "
                         f"{lc.effective_choice(entry)}, {status})")


def _mean_of(screens, mid: str) -> float | None:
    """The mean of ``mid`` over ``screens`` (a pooled line averages its two seeds), None if any lacks it."""
    vals = [_num(dget(s, mid)) for s in screens or []]
    if not vals or any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def _member_mde(cmpres: dict, mid: str) -> tuple[float, str] | None:
    """The MDE a flag prints beside its member, with its label: a block member's SINGLE-MEMBER MDE
    ``k x R_b x R_i`` (the block z is the mean of ``delta_i / R_i`` over its k measured members, so
    one member alone must move k times as far to carry the block), a guard's own R."""
    for label, body in (cmpres.get("blocks") or {}).items():
        for row in body.get("rows") or []:
            if row.get("metric") == mid and _num(body.get("R")) is not None and _num(row.get("R_eval")) is not None:
                k = max(int(body.get("k") or len(body.get("rows") or []) or 1), 1)
                return (k * float(body["R"]) * float(row["R_eval"]),
                        f"single-member MDE (k={k} x R_{label} x R_i)")
    for row in (cmpres.get("guards") or {}).get("rows") or []:
        if row.get("metric") == mid and _num(row.get("R")) is not None:
            return float(row["R"]), "guard MDE (R)"
    return None


def _flag(kind: str, text: str, *, metric: str | None = None, mde=None, value=None) -> dict:
    mde, mde_label = mde if isinstance(mde, tuple) else (mde, None)
    out = {"flag": kind, "text": text, "metric": metric, "mde": _num(mde), "value": _num(value)}
    if mde_label:
        out["mde_label"] = mde_label
    return out


def decision_flags(rule: dict, cmpres: dict, cand_screens, base_screens, *, removal=(),
                   pilot: bool = False) -> list[dict]:
    """The stop-and-ask flags of one DECISION line (plan step 4 / "Decision rule"). Never a verdict.

    Values are read on the line's own screens (``cand_screens`` / ``base_screens``: one each on a
    seed-3 line, both seeds on a pooled one -- averaged), deltas are candidate - control:

    * the margin flags (settled decision 7): turn ``facing_closed`` -0.05, move_turn
      ``facing_closed`` -0.10, any category's survival -0.05, any walking speed ratio -0.05;
    * taps up > 25 % (``taps_per_m`` / ``taps_per_rad``); ``swing_apex`` down >= 30 % (lift loss);
      ``support_feet`` down > 10 % in any category;
    * with ``decision_rule.steps_guard_flag``: ``steps_per_m`` / ``steps_per_rad`` adverse by > 15 %
      of the control's value (the null-band fallback of plan step 10);
    * speed-driven -- the steps members' mean ``delta/R`` BETTER, their mean dlog ``step_rate_hz``
      >= 0 and the T block's z > 0 (fewer steps per metre only because the robot moves faster);
    * removal verification failed on any run; the pilot stop (``pilot``).

    Each flag carries its member's MDE (:func:`_member_mde`) where the member is scored.
    """
    if not flags_enabled(rule):
        return []
    out: list[dict] = []
    cats = list(cmpres.get("present") or cmpres.get("decisive") or eval_categories().CATEGORIES)

    def vals(c: str, key: str):
        mid = f"cat.{c}.{key}"
        return mid, _mean_of(cand_screens, mid), _mean_of(base_screens, mid)

    for cat, key, drop in MARGIN_FLAGS:
        targets = cats if cat == "*all" else ([c for c in WALK_CATEGORIES if c in cats]
                                             if cat == "*walk" else ([cat] if cat in cats else []))
        for c in targets:
            mid, a, b = vals(c, key)
            if a is None or b is None:
                continue
            delta = a - b
            if delta <= -drop + 1e-12:
                out.append(_flag("margin", f"margin {c}.{key} {b:.3f}->{a:.3f} (delta {delta:+.3f} <= "
                                 f"-{drop})", metric=mid, mde=_member_mde(cmpres, mid), value=delta))

    def rel(a, b):
        if a is None or b is None:
            return None
        if b > 0:
            return (a - b) / b
        return math.inf if a > b else (0.0 if a == b else -math.inf)

    for c, key in TAP_KEYS.items():
        if c not in cats:
            continue
        mid, a, b = vals(c, key)
        r = rel(a, b)
        if r is not None and r > TAPS_UP_FRAC:
            out.append(_flag("taps", f"taps up {c}.{key} {b:.3f}->{a:.3f} ({r * 100:+.0f}% > "
                             f"+{TAPS_UP_FRAC * 100:.0f}%)", metric=mid, mde=_member_mde(cmpres, mid), value=r))
    for c in cats:
        mid, a, b = vals(c, "swing_apex")
        r = rel(a, b)
        if r is not None and -r >= APEX_DOWN_FRAC - 1e-12:
            out.append(_flag("lift_loss", f"swing_apex down {c} {b:.4f}->{a:.4f} ({r * 100:+.0f}% <= "
                             f"-{APEX_DOWN_FRAC * 100:.0f}%; the lift rescue is a re-budget option)",
                             metric=mid, mde=_member_mde(cmpres, mid), value=r))
        mid, a, b = vals(c, "support_feet")
        r = rel(a, b)
        if r is not None and -r > SUPPORT_DOWN_FRAC:
            out.append(_flag("support_down", f"support_feet down {c} {b:.3f}->{a:.3f} ({r * 100:+.0f}% "
                             f"< -{SUPPORT_DOWN_FRAC * 100:.0f}%)", metric=mid,
                             mde=_member_mde(cmpres, mid), value=r))
    if (rule or {}).get("steps_guard_flag"):
        for c, key in STEPS_GUARD_KEYS.items():
            if c not in cats:
                continue
            mid, a, b = vals(c, key)
            r = rel(a, b)
            if r is not None and r > STEPS_UP_FRAC:
                out.append(_flag("steps_guard", f"{c}.{key} adverse {b:.3f}->{a:.3f} ({r * 100:+.0f}% > "
                                 f"+{STEPS_UP_FRAC * 100:.0f}% of the control's; steps_guard_flag)",
                                 metric=mid, mde=_member_mde(cmpres, mid), value=r))
    # speed-driven: all three conditions
    rig = (cmpres.get("blocks") or {}).get(RIG_BLOCK) or {}
    steps_rows = [r for r in rig.get("rows") or [] if r.get("z") is not None
                  and _rig_key(r["metric"]) in RIG_STEPS_KEYS]
    t_z = _num(((cmpres.get("blocks") or {}).get("T") or {}).get("z"))
    if steps_rows:
        mean_z = sum(r["z"] for r in steps_rows) / len(steps_rows)
        dlogs = []
        for r in steps_rows:
            c = eval_categories().parse_cat_id(r["metric"])[0]
            _mid, a, b = vals(c, "step_rate_hz")
            if a is not None and b is not None and a > 0 and b > 0:
                dlogs.append(math.log(a) - math.log(b))
        mean_dlog = (sum(dlogs) / len(dlogs)) if dlogs else None
        if mean_z > 0 and mean_dlog is not None and mean_dlog >= 0 and t_z is not None and t_z > 0:
            out.append(_flag("speed_driven", f"speed-driven: steps members' mean delta/R {mean_z:+.3f} "
                             f"(BETTER side), mean dlog step_rate_hz {mean_dlog:+.3f} >= 0, T z "
                             f"{t_z:+.3f} > 0 -- fewer steps per unit because it moves faster",
                             value=mean_z))
    for run in removal or ():
        if isinstance(run, dict) and run.get("applies") and not run.get("ok"):
            bad = [f"{c.get('stage')}: {c.get('check')}" for c in run.get("checks") or [] if not c.get("ok")]
            out.append(_flag("removal", f"removal verification FAILED on {run.get('xid')} ("
                             + "; ".join(bad) + ") -- a halt-and-fix defect"))
    if pilot:
        out.append(_flag("pilot", "pilot stop (S3p): the pilot arm's seed-3 line has RIG TIE -- the "
                         "premise (removing the clock moves steps/support beyond R) is unproven; "
                         "continue wave 1, go to wave 2, or pause for MQ's action-bound fix"))
    return out


def flag_text(fl: dict) -> str:
    mde = fl.get("mde")
    return fl.get("text", "") + (f" [{fl.get('mde_label') or 'MDE'} {mde:.4g}]" if mde is not None else
                                 (" [MDE n/a: not a scored member]" if fl.get("metric") else ""))


def pilot_stop(rule: dict | None, cid: str | None, decision: dict) -> bool:
    """``decision_rule.pilot``'s seed-3 line with RIG TIE (plan step 12, S3p) -- an immediate park."""
    pilot = (rule or {}).get("pilot")
    if not pilot or str(pilot) != str(cid) or not flags_enabled(rule):
        return False
    cmpres = (decision.get("evidence") or {}).get("seed3") or {}
    return _required_block_class(cmpres, RIG_BLOCK) == "TIE" if cmpres.get("blocks") else False


def flag_park(ledger: dict, esq=None) -> dict | None:
    """The first live ledger entry whose flags PARK the driver and carry no acknowledgement at their
    level (``ledger_cli ack-flag <Did>``, :func:`ledger_cli.flag_parked`), else None.

    Derived from the ledger on every pass, like the rest of the queue: a relaunch re-derives the same
    park, and an ack row (applied by :func:`ingest_overrides`) releases it. A squashed entry never
    parks (a user ``decide run`` voids its line with it).
    """
    esq = set(esq or ())
    for e in entries(ledger):
        if e.get("id") in esq or hidden_by_stored_squash(e):
            continue
        if lc.flag_parked(e):
            return e
    return None


# ================================================================== the derived squash view
def all_headers(state: dict) -> dict:
    """Every experiment header, live or squashed -- the squash view is computed over BOTH.

    A squashed header stays a full header in ``state.ladder.squashed`` so that a revert can
    RESURRECT it: the derived view is recomputed from the ledger on every launch, so an experiment
    whose base hash matches again simply stops being in the set.
    """
    lad = state.get("ladder") or {}
    out = {}
    for xid, h in (lad.get("squashed") or {}).items():
        out[xid] = h
    for xid, h in (lad.get("experiments") or {}).items():
        out[xid] = h
    return out


def _expected_base(ledger: dict, cands: dict, header: dict, esq: set, code: str) -> tuple:
    """``(base_recipe_id, recipe_id)`` this header WOULD have under the current effective ledger."""
    rank = header.get("base_rank") or {}
    upto, inclusive = rank.get("upto"), bool(rank.get("inclusive"))
    if upto is None:
        return (header.get("base_recipe_id"), header.get("recipe_id"))
    recipe = compose_recipe(ledger, cands, code, upto=upto, inclusive=inclusive, squashed=esq)
    return (recipe["recipe_id"], recipe["recipe_id"])


def _skip_entries(ledger: dict) -> list[dict]:
    return [e for e in entries(ledger) if e.get("chosen") == "SKIP" and e.get("kind") == "candidate"]


def _files_present(header: dict) -> bool:
    """Resurrection needs the artefacts: the checkpoint and every recorded eval run dir."""
    ckpt = header.get("ckpt")
    if ckpt and not Path(ckpt).exists():
        return False
    for d in header.get("eval_dirs") or []:
        if d and not Path(d).exists():
            return False
    return True


def _user_squashed_s2(ledger: dict, esq: set) -> dict:
    """``{xid: reason}`` for the own seed-2 run of a decision the USER settled while it was speculative.

    Derived-view cause (e), and it is STICKY: a later ``decide run`` re-screens at seed 3, it never
    revives an old seed-2 run. Recomputed from the entry's own ``history`` so the view stays
    self-contained -- the driver also stores it, and the two agree.
    """
    out = {}
    for e in entries(ledger):
        xid = e.get("seed2_experiment")
        if not xid:
            continue
        for row in e.get("history") or []:
            if row.get("from") == "speculative" and row.get("by") == "user":
                out[xid] = (f"user {row.get('why') or row.get('to')} of {e.get('id')} while it was "
                            "speculative (cause e, never resurrected)")
                break
    return out


def squash_view(ledger: dict, cands: dict, state: dict, *, code: str = "", check_files: bool = True) -> dict:
    """The GLOBAL derived squash view, recomputed after every ledger write.

    Returns ``{"experiments": {xid: why}, "entries": {id: why}, "candidates": [cid, ...]}``. An
    experiment or SKIP entry is squashed when

    (a) its ``base_recipe_id`` differs from the recipe composed of the effective choices of the
        candidates ranked before it (a milestone: when its own ``recipe_id`` moved),
    (b) it is a conditional SKIP entry whose predicate, re-evaluated against the current effective
        ledger, now HOLDS (the referent flipped; a SKIP the USER asked for, and the all-N/A SKIP of a
        screened candidate -- ``evidence.source: no_decisive_block`` -- are never squashed by this),
    (c) -- transitively -- it is a SKIP entry whose ``conditional_refs`` include a candidate squashed
        in this pass,
    (d) its candidate's newest ``user_choice`` is ``skip`` and is newer than the header's ``started``
        -- an experiment OF that candidate, which a ``twin`` is not (see below),
    (e) it is a USER squash: an entry overridden with ``decide run``, or the own seed-2 run of a
        decision the user confirmed/overruled while it was still speculative,
    (f) it is a ledger entry NAMED by some header's ``rescores[].supersedes`` (plan D9 item 5): the
        floor REJECT a rescore re-decided under the current eval. Keyed on that STATE record and
        nothing else -- so the launch reconcile squashes nothing before a rescore has run, and the
        REJECT a rescore FAIL writes (never named) stays effective.

    (d) and (e) are ``squashed_by: user_choice`` and are NEVER resurrected. (a)-(c) and (f) are
    recomputed from scratch, so reverting an overrule resurrects what it squashed -- if the files
    still exist.

    The loop is a fixpoint: squashing an entry changes the effective ledger, which changes the
    composed recipes, which can squash more. It is bounded and asserted stable.
    """
    lad = state.get("ladder") or {}
    user_choice = lad.get("user_choice") or {}
    stored = lad.get("squashed") or {}
    headers = all_headers(state)
    order = candidate_ids(cands)

    xsq: dict = {xid: (h.get("reason") or "user squash")
                 for xid, h in stored.items() if h.get("squashed_by") == "user_choice"}
    esq: dict = {}
    for xid, why in _user_squashed_s2(ledger, set()).items():
        xsq.setdefault(xid, why)
    # (f) the entries a rescore re-decided: exactly the ids named by `rescores[].supersedes`, on
    # live and squashed headers alike (a record is never rewritten, so neither is its supersession)
    for h in headers.values():
        for rec in h.get("rescores") or []:
            sid = rec.get("supersedes")
            if sid and sid not in esq:
                esq[sid] = (f"re-decided by the rescore of {h.get('xid')}:{rec.get('floor_stage') or '?'} "
                            f"under eval_sha {str(rec.get('eval_sha') or '')[:12]} ({rec.get('verdict')}; "
                            "cause f)")

    for _round in range(64):
        changed = False
        # (e) an entry the user squashed with `decide run`
        for e in entries(ledger):
            ov = e.get("override") or {}
            if ov.get("by") == "user" and str(ov.get("why", "")).startswith("decide run") \
                    and e.get("id") not in esq:
                esq[e["id"]] = "user: decide run (cause e, never resurrected)"
                changed = True
        # experiments
        for xid, h in headers.items():
            if xid in xsq:
                continue
            why = None
            cid = h.get("candidate")
            uc = user_choice.get(cid) or {}
            # Cause (d) is scoped to the running/finished experiments OF the candidate (SQUASH (2)'s
            # skip clause). A `twin`'s `candidate` field names the candidate whose PAIR booked it, not
            # the candidate it screens: a twin is the BASE recipe's s2 screen, and decision 7 says it
            # "is never interrupted or squashed (it is the base's s2 screen, needed by later pairs and
            # by M<k>.s2)". Without the exemption `decide C5 --choose run; decide C5 --choose skip`
            # squashed it under a cause that is NEVER resurrected -- and with C4 adopted that twin is
            # R(<=C5)'s only s2 screen, so the base recipe silently lost its s2 and every later pair
            # (and M<k>.s2) re-paid a 3.9-hour window. Cause (a) still applies to a twin: it falls
            # through to the `else` below, and a twin whose base recipe really moved is an experiment
            # of a recipe that no longer exists.
            if uc.get("choice") == "skip" and str(uc.get("at") or "") > str(h.get("started") or "") \
                    and h.get("kind") != "twin":
                why = f"user_choice: skip {cid} at {uc.get('at')} (cause d, never resurrected)"
            elif h.get("kind") == "baseline":
                why = None            # a baseline is a screen of a named recipe and stays a record
            else:
                want_base, want_recipe = _expected_base(ledger, cands, h, set(esq), code)
                if h.get("kind") in ("milestone", "batched_s2", "milestone_s2"):
                    if want_recipe and h.get("recipe_id") and want_recipe != h["recipe_id"]:
                        why = (f"milestone recipe changed: {h['recipe_id']} -> {want_recipe}")
                elif want_base and h.get("base_recipe_id") and want_base != h["base_recipe_id"]:
                    why = f"base recipe changed: {h['base_recipe_id']} -> {want_base}"
            if why is None and xid in stored and stored[xid].get("squashed_by") != "user_choice" \
                    and check_files and not _files_present(stored[xid]):
                why = "resurrection refused: the checkpoint or an eval run dir is gone"
            if why:
                xsq[xid] = why
                changed = True
        # entries whose experiment went
        for e in entries(ledger):
            xid = e.get("experiment")
            if xid and xid in xsq and e.get("id") not in esq:
                esq[e["id"]] = f"its experiment {xid} was squashed ({xsq[xid]})"
                changed = True
        # MILESTONE entries built on a recipe that has since changed (SQUASH (2): "plus milestones
        # built on a changed recipe"). An entry's experiment is not the link that matters here: a
        # carried-forward milestone records NO experiment at all, and an `open` entry outlives the
        # promotion header that was squashed under it. Without this rule the carried-forward entry
        # stays `final` on a recipe that no longer exists and the rolling reference hands M3 a
        # two-generations-old head, while an `open` one sends next_item straight back to the
        # promotion with (i) never re-run on the reduced recipe.
        for e in entries(ledger):
            if e.get("kind") != "milestone" or e.get("id") in esq:
                continue
            # Deliberately NOT skipped when the stored status is already `squashed`: the view is
            # derived from scratch on every launch, and squash_and_report resurrects a `derived`
            # squash the moment its cause stops being listed. An entry that dropped out here as
            # soon as its status was persisted would be resurrected on the next override.
            want = milestone_recipe(cands, ledger, e.get("candidate"), code, set(esq))
            got = e.get("recipe_id")
            if got and want.get("recipe_id") and want["recipe_id"] != got:
                esq[e["id"]] = (f"milestone recipe changed: {got} -> {want['recipe_id']} "
                                "(the milestone was decided on a recipe that no longer exists)")
                changed = True
        # The candidates that are UNDECIDED again: they have at least one entry and none of them is
        # effective any more (squashed in this pass, or already carrying status "squashed" from an
        # earlier one). Comparing "before vs after esq" instead would miss the second case, because
        # `newest_entry` already skips a stored `squashed` status.
        gone = [cid for cid in order
                if newest_entry(ledger, cid, set(esq)) is None
                and any(e.get("candidate") == cid for e in entries(ledger))]
        # SKIP entries: (b) the referent flipped, (c) transitive refs, (a) the base recipe moved
        for e in _skip_entries(ledger):
            if e.get("id") in esq:
                continue
            cid = e.get("candidate")
            cand = candidate_by_id(cands, cid) or {}
            by_user = bool((e.get("evidence") or {}).get("user"))
            # The all-N/A SKIP (plan D5: `evidence.source: no_decisive_block`) answers the SCREEN,
            # not the predicate: a conditional candidate is only screened when its predicate holds,
            # so cause (b) would squash that SKIP at once and re-screen it forever. Exempt from (b)
            # exactly as a user SKIP is; causes (a) and (c) still apply to it.
            no_block = (e.get("evidence") or {}).get("source") == "no_decisive_block"
            refs = e.get("conditional_refs") or conditional_refs(cand.get("conditional"))
            recipe = compose_recipe(ledger, cands, code, upto=cid, squashed=set(esq))
            if not by_user and not no_block and cand.get("conditional"):
                ctx = predicate_context(cands, ledger, state, recipe, code,
                                        {"entries": set(esq), "experiments": set(xsq)})
                try:
                    holds, why = compile_predicate(cand["conditional"]).evaluate(ctx)
                except ValueError:
                    holds, why = False, "unevaluable"
                if holds:
                    esq[e["id"]] = f"the SKIP referent flipped: {why} now holds (cause b)"
                    changed = True
                    continue
            if any(r in gone for r in refs):
                esq[e["id"]] = (f"transitive: conditional_refs {refs} include a candidate squashed in "
                                "this pass (cause c)")
                changed = True
                continue
            # (a) applies to a SKIP entry too -- the plan's own words: "an experiment OR SKIP ENTRY
            # is squashed when (a) its base_recipe_id differs from the recipe composed of the
            # effective choices of the candidates ranked before it", and SQUASH (2) repeats it
            # ("experiments AND SKIP ENTRIES whose base_recipe_id differs"). Tested LAST because
            # (b) and (c) are the more specific diagnoses and the SQUASH line quotes the reason. A
            # SKIP is not an ADOPT, so squashing one moves no recipe and starts no cascade; what it
            # buys is that no record claims a predicate was evaluated against a recipe that is no
            # longer the base -- the candidate is re-evaluated at its rank and the predicate (or
            # the user's standing `skip`) writes a fresh SKIP at the new base, for zero GPU.
            # A USER skip is carved out for the same reason clause (b) carves it out: it is an
            # answer about the CANDIDATE, not a function of the recipe, so a moved base is no
            # reason to re-ask it -- and re-writing it on every adoption would make a standing
            # answer read as repeatedly overturned in the record.
            if not by_user and e.get("base_recipe_id") \
                    and recipe["recipe_id"] != e["base_recipe_id"]:
                esq[e["id"]] = (f"base recipe changed: {e['base_recipe_id']} -> "
                                f"{recipe['recipe_id']} (cause a)")
                changed = True
        if not changed:
            break
    else:                                                               # pragma: no cover - bounded
        raise RuntimeError("the derived squash view did not converge in 64 rounds")

    # The `why` of a recipe-change squash names the recipe the experiment "would now have". That
    # recipe is itself a function of the squash set, so a message written mid-fixpoint can name an
    # intermediate hash while the SET is already final. Recompute those messages once, against the
    # converged set, so the SQUASH line and the stored reason are stable across recomputes.
    for xid, why in list(xsq.items()):
        header = headers.get(xid)
        if header is None:
            continue
        if str(why).startswith("base recipe changed"):
            want_base, _ = _expected_base(ledger, cands, header, set(esq), code)
            if want_base:
                xsq[xid] = f"base recipe changed: {header.get('base_recipe_id')} -> {want_base}"
        elif str(why).startswith("milestone recipe changed"):
            _, want_recipe = _expected_base(ledger, cands, header, set(esq), code)
            if want_recipe:
                xsq[xid] = f"milestone recipe changed: {header.get('recipe_id')} -> {want_recipe}"
    for e in entries(ledger):
        xid = e.get("experiment")
        if e.get("id") in esq and xid in xsq and str(esq[e["id"]]).startswith("its experiment "):
            esq[e["id"]] = f"its experiment {xid} was squashed ({xsq[xid]})"

    undecided = [cid for cid in order
                 if newest_entry(ledger, cid, set(esq)) is None
                 and any(e.get("candidate") == cid for e in entries(ledger))]
    return {"experiments": xsq, "entries": esq, "candidates": undecided}


#: The plan's own name for :func:`squash_view`. Kept as an alias because the plan, the charter and
#: the review checklist all say "squash_set"; the function returns a VIEW (three keyed sets:
#: experiments, ledger entries, and the candidates that are undecided again), which is why the
#: implementation is spelled the other way round.
squash_set = squash_view


# ================================================================== the derived queue
def _label_for(kind: str, cid: str | None, seed, extra: str = "") -> str:
    if kind == "twin":
        return f"twin({cid}) s{seed}"
    if kind == "seed2":
        return f"{cid} s{seed}"
    return f"{cid or kind}{extra} s{seed}"


def pending_decisions(ledger: dict, esq=None) -> list[dict]:
    """Every decision still waiting for its pair: DERIVED from the ledger, never a stored list.

    A decision is pending exactly while it is ``speculative``. Deriving it is what makes the
    plan's "a user confirm/overrule drops that candidate's own seed-2 run from pending" automatic
    rather than a second bookkeeping step that could disagree with the ledger.
    """
    esq = set(esq or ())
    return [e for e in entries(ledger)
            if e.get("status") == "speculative" and e.get("id") not in esq]


def _stages_for(cands: dict, chain: str) -> list[dict]:
    return [dict(s) for s in chain_stages(cands, chain)]


def _item(kind: str, *, cid=None, seed=3, recipe=None, base_recipe=None, stages=None, decision=None,
          why="", needs_s3=None, label=None, eligibility=None, extra=None) -> dict:
    return {"kind": kind, "candidate": cid, "seed": str(seed), "recipe": recipe,
            "base_recipe": base_recipe, "stages": stages or [], "decision": decision, "why": why,
            "needs_s3": needs_s3, "label": label or _label_for(kind, cid, seed),
            "eligibility": eligibility, "extra": extra or {}}


def baseline_label(cands: dict, state: dict, recipe_id: str, code: str) -> str:
    """``B0`` for the campaign's first baseline recipe, ``B<k>`` for every later recipe baseline.

    ``k`` counts the DISTINCT ``(recipe_id, code_sha)`` pairs already screened as baselines, so the
    two seeds of one baseline share a label (``B0 s3`` / ``B0 s2``) while a rerun forced by an
    overrule or by a ``code_sha`` landing gets the next number -- which is what the CHANGELOG and the
    budget line mean by "B<k>".
    """
    # Normalised through the proved aliases (plan D8): a header recorded under a pre-suite value of
    # the SAME training tree is the same baseline, not a spurious `B1`.
    aliases = code_aliases(state)
    seen = {(h.get("recipe_id"), code if h.get("code_sha") in aliases else h.get("code_sha"))
            for h in all_headers(state).values() if h.get("kind") == "baseline"}
    seen.discard((recipe_id, code))
    return (cands.get("baseline") or {}).get("id", "B0") if not seen else f"B{len(seen)}"


def after_holds(cands: dict, ledger: dict, mid: str, esq=None) -> tuple[bool, str]:
    """A milestone's ``after``: every candidate up to and including it is TERMINAL and not undecided."""
    esq = set(esq or ())
    order = candidate_ids(cands)
    after = None
    for mstone in cands.get("milestones") or []:
        if mstone.get("id") == mid:
            after = mstone.get("after")
    if after is None:
        return False, f"unknown milestone {mid}"
    for cid in order[:order.index(after) + 1]:
        entry = newest_entry(ledger, cid, esq)
        if not is_terminal(entry):
            return False, f"{cid} is {entry.get('status') if entry else 'undecided'}"
    return True, f"every candidate up to {after} is TERMINAL"


def milestone_recipe(cands: dict, ledger: dict, mid: str, code: str, esq=None) -> dict:
    after = None
    for mstone in cands.get("milestones") or []:
        if mstone.get("id") == mid:
            after = mstone.get("after")
    return compose_recipe(ledger, cands, code, upto=after, inclusive=True, squashed=set(esq or ()))


def previous_milestone(cands: dict, mid: str) -> str | None:
    prev = None
    for mstone in cands.get("milestones") or []:
        if mstone.get("id") == mid:
            return prev
        prev = mstone.get("id")
    return None


def milestone_block_entries(cands: dict, ledger: dict, mid: str, esq=None) -> list[dict]:
    """The block's effective candidate entries (what a batched confirmation confirms)."""
    esq = set(esq or ())
    out = []
    for cid in lc.milestone_block(cands, mid):
        entry = newest_entry(ledger, cid, esq)
        if entry is not None:
            out.append(entry)
    return out


def milestone_rerun_used(ledger: dict, mid: str, cands: dict, esq=None) -> bool:
    """Has this eligibility already spent its ONE automatic rerun?

    Counted by DECISION entries carrying ``override {by: milestone, why: M<k>}`` newer than the last
    USER override on any block member -- a user's answer restarts the allowance, which is what makes
    ``--drop`` a fresh eligibility rather than a second automatic attempt.

    "Newer" is the override's own ``at`` stamp, NOT the ledger position of the entry it sits on.
    The milestone's automatic drop always picks the block's LAST unfrozen adoption, so the natural
    user answer is a ``--drop`` of an EARLIER one -- whose entry sits at a LOWER ledger index. Read
    positionally, that answer did not restart the allowance and the next contradiction parked for
    the user instead of taking its rerun, making ``--drop`` silently weaker than ``--accept``.
    Ledger position is kept only as the tie-break, for records written before ``at`` existed.

    ``esq`` is accepted for call-site symmetry and deliberately NOT applied: the allowance is
    counted over the whole ledger, because the squash the automatic rerun itself performs must not
    erase the evidence that the rerun happened.
    """
    block = set(lc.milestone_block(cands, mid))
    last_user = None
    rows = list(entries(ledger))
    for i, e in enumerate(rows):
        ov = e.get("override") or {}
        if e.get("candidate") in block and ov.get("by") == "user":
            key = (str(ov.get("at") or ""), i)
            if last_user is None or key > last_user:
                last_user = key
    for i, e in enumerate(rows):
        ov = e.get("override") or {}
        if ov.get("by") != "milestone" or mid not in str(ov.get("why") or ""):
            continue
        if last_user is None or (str(ov.get("at") or ""), i) > last_user:
            return True
    return False


def _next_item_raw(cands: dict, ledger: dict, state: dict, *, code: str, view=None,
                   resolution=None) -> dict | None:
    """The next thing to do -- a DERIVED queue, recomputed from the records on every pass.

    Order (the plan's Main loop): the baseline's two screens, a recipe-baseline rerun when the next
    item's base recipe has no ok s3 screen, the pending pairs (the base twin before the candidate's
    own seed 2), a sticky milestone ``await``, the eligible milestone's three experiments, M3's
    final seed-2 promotion, then the first ranked undecided candidate whose dependencies are
    TERMINAL. ``None`` means the campaign is done.

    Nothing is stored: a crash, a relaunch or an override recomputes the same queue from the ledger,
    which is the property the whole speculative design rests on.
    """
    view = view or squash_view(ledger, cands, state, code=code)
    esq, xsq = set(view["entries"]), set(view["experiments"])
    order = candidate_ids(cands)
    seeds = cands.get("seeds") or {}
    s3, s2 = str(seeds.get("screen", 3)), str(seeds.get("confirm", 2))
    base_cfg = cands.get("baseline") or {}

    def screen_of(recipe_id, seed):
        return ok_screen(state, code, recipe_id, seed, view)

    # ---- 0: a flagged DECISION parks the driver (clock-deprecation plan step 4, 2026-09-23) --
    # before anything else, the pending-pair dispatch included (the pilot stop parks BEFORE its own
    # pair and before the next arm). Only a rule with block RIG writes flags, and only
    # `ledger_cli ack-flag <Did>` releases one; with no flagged entry this step is inert.
    parked = flag_park(ledger, esq)
    if parked is not None:
        fl = parked.get("flags") or {}
        did = parked.get("id")
        texts = "; ".join(flag_text(f) for f in fl.get("items") or []) or fl.get("why") or "flagged"
        return _item("await", cid=parked.get("candidate"), seed=s3,
                     why=(f"{did} ({parked.get('candidate')}) is FLAGGED on its {fl.get('level')} line "
                          f"-- stop and ask: {texts}"),
                     label=f"await_user:{did} flag", extra={"flag_park": did})

    # ---- 1 / 2: the baseline recipe's own two screens
    b0 = compose_recipe(ledger, cands, code, upto=(order[0] if order else None), squashed=esq)
    for seed in (s3, s2):
        if screen_of(b0["recipe_id"], seed) is None:
            label = baseline_label(cands, state, b0["recipe_id"], code)
            return _item("baseline", cid=label, seed=seed, recipe=b0, base_recipe=b0,
                         stages=_stages_for(cands, b0["chain"]),
                         why=f"the baseline recipe has no ok s{seed} screen",
                         label=f"{label} s{seed}")

    # ---- 3: pending pairs (the twin first)
    for entry in pending_decisions(ledger, esq):
        cid = entry.get("candidate")
        if cid not in order:
            continue
        base = compose_recipe(ledger, cands, code, upto=cid, squashed=esq)
        cand_recipe_id = entry.get("recipe_id_if_adopted")
        if screen_of(base["recipe_id"], s2) is None:
            return _item("twin", cid=cid, seed=s2, recipe=base, base_recipe=base,
                         stages=_stages_for(cands, base["chain"]), decision=entry.get("id"),
                         needs_s3=base,
                         why=f"{entry.get('id')} is speculative and its base recipe has no ok s{s2} screen")
        if cand_recipe_id and screen_of(cand_recipe_id, s2) is None:
            # The pair compares candidate-vs-base on BOTH seeds, so the seed-2 run is the candidate's
            # own recipe whatever the s3 pick was (a speculative REJECT still owes its pair a run).
            cand_recipe = _recipe_with(ledger, cands, code, cid, esq)
            return _item("seed2", cid=cid, seed=s2, recipe=cand_recipe, base_recipe=base,
                         stages=_stages_for(cands, cand_recipe["chain"]), decision=entry.get("id"),
                         needs_s3=base,
                         why=f"{entry.get('id')} is speculative: its own seed-{s2} run confirms the pair")
        return _item("pooled", cid=cid, seed=s2, recipe=None, base_recipe=base, decision=entry.get("id"),
                     needs_s3=base,
                     why=f"both halves of {entry.get('id')}'s pair exist -- the pooled decision is "
                         "zero-GPU (a reused seed-2 screen)")

    # ---- 4 / 5 / 6: milestones
    for mstone in cands.get("milestones") or []:
        mid = mstone.get("id")
        entry = newest_milestone_entry(ledger, mid, esq)
        status = (entry or {}).get("status")
        if status == "await" and (entry or {}).get("sub_status") != "accepted_i":
            return _item("await", cid=mid, seed=s3, why=f"{mid} is FAIL and waiting for the user "
                                                        f"(resume-milestone {mid} --accept | --drop <Cid>)",
                         label=f"await_user:{mid}")
        if status == "final":
            continue
        holds, why = after_holds(cands, ledger, mid, esq)
        if not holds:
            break                      # a later milestone cannot be eligible before an earlier one
        recipe = milestone_recipe(cands, ledger, mid, code, esq)
        stages = _stages_for(cands, recipe["chain"])
        if status == "await" and (entry or {}).get("sub_status") == "accepted_i":
            return _item("milestone", cid=mid, seed=s3, recipe=recipe,
                         stages=promotion_stage_list(cands), needs_s3=recipe,
                         eligibility=(entry or {}).get("eligibility"),
                         why=f"{mid} (i) accepted by the user -- the promotion runs",
                         label=f"{mid}.promo s{s3}")
        if screen_of(recipe["recipe_id"], s2) is None:
            return _item("batched_s2", cid=mid, seed=s2, recipe=recipe, base_recipe=recipe,
                         stages=stages, needs_s3=recipe,
                         eligibility=(entry or {}).get("eligibility"),
                         why=f"{mid}: the effective recipe has no ok s{s2} screen for the batched "
                             "confirmation", label=f"{mid}.s2 s{s2}")
        if status is None:
            # (i) pairs R_k against R_(k-1). R_k's own screens are covered above (the B<k> wrapper
            # for s3, the batched_s2 branch for s2); R_(k-1)'s are not, and a missing one would make
            # the zero-GPU (i) record WATCH and let the promotion run with `decided` adoptions never
            # confirmed -- which the plan's step (3) forbids ("only when (i) is resolved: no
            # `decided` adoption on R_k awaiting its batched confirmation"). Reachable after
            # --accept-code-change, which stops every old screen from being an ok screen.
            prev_mid = previous_milestone(cands, mid)
            pending_block = [e for e in milestone_block_entries(cands, ledger, mid, esq)
                             if lc.effective_choice(e) == "ADOPT" and e.get("status") == "decided"
                             and not e.get("frozen")]
            if prev_mid and pending_block:
                prev_recipe = milestone_recipe(cands, ledger, prev_mid, code, esq)
                for seed in (s3, s2):
                    if screen_of(prev_recipe["recipe_id"], seed) is None:
                        label = baseline_label(cands, state, prev_recipe["recipe_id"], code)
                        return _item("baseline", cid=label, seed=seed, recipe=prev_recipe,
                                     base_recipe=prev_recipe,
                                     stages=_stages_for(cands, prev_recipe["chain"]),
                                     why=f"{mid} (i) pairs against {prev_mid}'s recipe "
                                         f"{prev_recipe['recipe_id']}, which has no ok s{seed} "
                                         "screen -- the pair cannot be formed without it",
                                     label=f"{label} s{seed}")
            return _item("milestone_i", cid=mid, seed=s3, recipe=recipe, needs_s3=recipe,
                         why=f"{mid} (i): the batched seed-2 confirmation is a zero-GPU decision",
                         label=f"{mid}.(i)")
        # the entry is open: (i) is resolved, so the promotion is next (or is omitted)
        prev = previous_milestone(cands, mid)
        prev_entry = newest_milestone_entry(ledger, prev, esq) if prev else None
        omit = bool(prev and prev_entry and prev_entry.get("recipe_id") == recipe["recipe_id"])
        if omit:
            return _item("milestone_close", cid=mid, seed=s3, recipe=recipe,
                         why=f"{mid}: recipe_id equals {prev}'s -- both experiments are omitted and the "
                             "entry carries forward", label=f"{mid}.carry")
        return _item("milestone", cid=mid, seed=s3, recipe=recipe, stages=promotion_stage_list(cands),
                     needs_s3=recipe, eligibility=(entry or {}).get("eligibility"),
                     why=f"{mid} (ii): promote the recipe head through 2a then 2b",
                     label=f"{mid}.promo s{s3}")

    # ---- 7: M3's final seed-2 promotion
    final = [m for m in (cands.get("milestones") or []) if m.get("final")]
    if final:
        mid = final[-1]["id"]
        entry = newest_milestone_entry(ledger, mid, esq)
        if entry and entry.get("status") == "final":
            recipe = milestone_recipe(cands, ledger, mid, code, esq)
            have = [h for h in ok_runs(state, code, view)
                    if h.get("kind") == "milestone_s2" and h.get("recipe_id") == recipe["recipe_id"]]
            if not have:
                return _item("milestone_s2", cid=mid, seed=s2, recipe=recipe,
                             stages=final_s2_stage_list(cands), needs_s3=recipe,
                             why=f"{mid} is closed: the final recipe's seed-{s2} head runs 2a -> 2b",
                             label=f"{mid}.final_s2 s{s2}")

    # ---- 8: the first ranked undecided candidate
    lad = state.get("ladder") or {}
    user_choice = lad.get("user_choice") or {}
    for cid in order:
        entry = newest_entry(ledger, cid, esq)
        if is_terminal(entry):
            continue
        cand = candidate_by_id(cands, cid) or {}
        blocked = [d for d in (cand.get("depends_on") or [])
                   if not is_terminal(newest_entry(ledger, d, esq))]
        if blocked:
            continue
        choice = (user_choice.get(cid) or {}).get("choice")
        base = compose_recipe(ledger, cands, code, upto=cid, squashed=esq)
        if choice == "skip" and _user_skip_stands(state, cid, user_choice.get(cid)):
            return _item("skip", cid=cid, seed=s3, recipe=base, base_recipe=base,
                         why="the user asked to skip it", label=f"{cid} SKIP",
                         extra={"source": "user"})
        ctx = predicate_context(cands, ledger, state, base, code, view)
        holds, pred_why = (True, "no conditional")
        if cand.get("conditional"):
            try:
                holds, pred_why = compile_predicate(cand["conditional"]).evaluate(ctx)
            except ValueError as exc:
                holds, pred_why = False, f"unevaluable ({exc})"
        if not holds and choice != "run":
            return _item("skip", cid=cid, seed=s3, recipe=base, base_recipe=base,
                         why=f"predicate: {pred_why}", label=f"{cid} SKIP",
                         extra={"predicate": pred_why})
        if cand.get("user_decision") and choice not in ("run", "skip"):
            return _item("await", cid=cid, seed=s3,
                         why=f"{cid} redefines the task: the user decides "
                             f"(ledger_cli decide {cid} --choose run, or --choose skip)",
                         label=f"await_user:{cid}")
        if cand.get("await_if") and choice not in ("run", "skip"):
            # The branch-and-ask gate: `user_decision` made CONDITIONAL. While the predicate holds
            # the driver parks here (a user `decide <Cid> --choose run|skip`, typed now or before the
            # driver arrives, releases it); when it is false the candidate dispatches as usual.
            try:
                stop, await_why = compile_predicate(cand["await_if"]).evaluate(ctx)
            except ValueError as exc:
                stop, await_why = True, f"unevaluable ({exc}) -- parked, the safe direction"
            if stop:
                note = str(cand.get("await_why") or "the user decides").strip()
                return _item("await", cid=cid, seed=s3,
                             why=f"{cid}: {note} ({await_why}) -- ledger_cli decide {cid} --choose "
                                 "run, or --choose skip",
                             label=f"await_user:{cid}", extra={"await_if": await_why})
        ok, dispatch_why = dispatch_check(cand)
        if not ok:
            return _item("await", cid=cid, seed=s3, why=dispatch_why, label=f"await_user:{cid} code")
        # When the user forced a candidate whose predicate is FALSE, the reason this window runs is
        # the user -- quoting the predicate would make the DECISION report read as if a false
        # condition had held.
        reason = (f"the user asked to run it (decide {cid} --choose run); predicate: {pred_why}"
                  if choice == "run" and not holds else pred_why)
        held = _rescored_header(state, cid, xsq)
        if held is not None:
            # Plan D9 item 6: a floor-rejected header whose rescore PASSED under the current eval and
            # whose candidate is conditional waits here, not in `find_inflight`. Dispatched WITH ITS
            # XID -- without it this step would book a fresh experiment from the chain's first stage
            # and orphan the re-opened one -- and `needs_s3` = the composed base (item 30).
            item = item_from_header_raw(cands, held)
            item["needs_s3"] = base
            item["why"] = (f"rank {order.index(cid) + 1}: {reason}; resumes {held.get('xid')} (floor "
                           "rescored under the current eval) at its next stage")
            return item
        recipe = _recipe_with(ledger, cands, code, cid, esq)
        return _item("candidate", cid=cid, seed=s3, recipe=recipe, base_recipe=base,
                     stages=_stages_for(cands, recipe["chain"]), needs_s3=base,
                     why=f"rank {order.index(cid) + 1}: {reason}", label=f"{cid} s{s3}")
    return None


def _user_skip_stands(state: dict, cid: str, uc: dict | None) -> bool:
    """Is a standing user ``skip`` still the newest word on this candidate?

    Vocabulary: "a ``user_choice`` of ``skip`` NEWER THAN EVERY EXPERIMENT of ``Cid`` writes a SKIP
    entry at the current base". A stale one (a screen ran after it, and its entry was later
    squashed) must not silently re-write a SKIP over a candidate the campaign has since screened --
    the predicate path re-decides it instead.
    """
    at = str((uc or {}).get("at") or "")
    if not at:
        return True                      # no stamp recorded: the choice is all the evidence there is
    return not any(str(h.get("started") or "") > at
                   for h in all_headers(state).values() if h.get("candidate") == cid)


#: A floor-rejected header whose rescore PASSED under the current eval (plan D9 item 5) and whose
#: candidate declares ``depends_on`` / ``conditional``: it is re-opened, but only step 8 of the queue
#: -- after the dependency, the predicate and the user's choice -- may dispatch it (with its xid).
#: ``run_experiment`` flips it to ``running`` at that dispatch; ``find_inflight`` never resumes it.
RESCORED_AWAITING = "rescored_awaiting_predicate"


def _rescored_header(state: dict, cid: str, xsq) -> dict | None:
    """The live ``rescored_awaiting_predicate`` header of ``cid`` (oldest first), or None."""
    squashed = set(xsq or ())
    for xid, h in sorted(experiments(state).items(), key=lambda kv: _xid_key(kv[0])):
        if h.get("candidate") == cid and h.get("status") == RESCORED_AWAITING and xid not in squashed:
            return dict(h, xid=h.get("xid") or xid)
    return None


def _xid_key(xid: str) -> tuple:
    """``X03`` < ``X10`` < ``X100``: experiment ids in creation order (numeric, never lexical)."""
    m = re.fullmatch(r"X(\d+)", str(xid))
    return (0, int(m.group(1)), "") if m else (1, 0, str(xid))


def next_item(cands: dict, ledger: dict, state: dict, *, code: str, view=None,
              resolution=None) -> dict | None:
    """:func:`_next_item_raw` plus the recipe-baseline rerun rule.

    Every item except a baseline names the recipe whose ok s3 screen it needs -- a candidate needs
    its base recipe's screen as the CONTROL, a pair needs it as the pooled base, a milestone needs
    it as the head it promotes. When that screen does not exist (an overrule changed the recipe, or
    a ``code_sha`` landing invalidated every old screen) the queue inserts ``B<k>``, a rerun of that
    recipe at seed 3, FIRST. Old screens stay as records; nothing is squashed by this.
    """
    view = view or squash_view(ledger, cands, state, code=code)
    item = _next_item_raw(cands, ledger, state, code=code, view=view, resolution=resolution)
    if item is None or item["kind"] in ("baseline", "await", "skip"):
        return item
    needs = item.get("needs_s3")
    if not needs:
        return item
    seeds = cands.get("seeds") or {}
    s3 = str(seeds.get("screen", 3))
    if ok_screen(state, code, needs["recipe_id"], s3, view) is not None:
        return item
    label = baseline_label(cands, state, needs["recipe_id"], code)
    return _item("baseline", cid=label, seed=s3, recipe=needs, base_recipe=needs,
                 stages=_stages_for(cands, needs["chain"]),
                 why=f"{item['label']} needs an ok s{s3} screen of recipe {needs['recipe_id']} as its "
                     f"control and there is none (overrule or code change)", label=f"{label} s{s3}")


def _recipe_with(ledger: dict, cands: dict, code: str, cid: str, esq) -> dict:
    """The recipe a candidate's OWN window runs: its base recipe with this candidate adopted.

    Composed by pretending the candidate is an ADOPT at its rank (a synthetic ledger entry), so the
    one ``compose_recipe`` implementation decides what "with this candidate" means -- the driver
    never assembles a recipe a second way.
    """
    # `speculative`/`uncertain` on purpose: the synthetic entry must count as an ADOPT for the
    # composition and must NEVER qualify as the anchor -- a candidate that anchored the drift check
    # to itself would compare its own screen against itself and the precondition would be inert.
    synth = dict(ledger)
    synth["entries"] = entries(ledger) + [{"id": f"__synthetic_{cid}", "kind": "candidate",
                                           "candidate": cid, "chosen": "ADOPT",
                                           "status": "speculative", "confidence": "uncertain",
                                           "synthetic": True}]
    return compose_recipe(synth, cands, code, upto=cid, inclusive=True, squashed=set(esq or ()))


def dispatch_check(cand: dict) -> tuple[bool, str]:
    """Deferred candidates: accepted at LOAD, refused at DISPATCH until their knob is exported."""
    if not cand.get("deferred"):
        return True, "not deferred"
    req = cand.get("requires_knob")
    req_keys = [req] if isinstance(req, str) else list(req or [])
    missing = [k for k in req_keys if k not in knob_whitelist()]
    if missing:
        return False, (f"{cand.get('id')} is deferred: {missing} is not exported yet, so the code it "
                       f"needs has not landed. Land it, relaunch with --accept-code-change, and the "
                       f"recipe-baseline rerun plus this screen follow (billed to {cand.get('id')}).")
    return True, f"{cand.get('id')}'s knob(s) {req_keys} are exported"


def simulate_queue(cands: dict, *, outcome: str = "REJECT", limit: int = 40, code: str = "dryrun") -> list[dict]:
    """The queue the driver WOULD work through, with every decision resolved as ``outcome``.

    Used by ``--dry-run`` and by the tests: it exercises the real ``next_item`` against a synthetic
    ledger rather than re-describing the order in prose. ``REJECT`` keeps the recipe at the baseline
    (the no-adoption path, which is the ordering the plan's examples are written against);
    ``ADOPT`` walks the adoption path instead.
    """
    ledger = {"schema": 1, "entries": []}
    state = {"ladder": {"experiments": {}, "squashed": {}, "user_choice": {}}, "keys": {}}
    out, n = [], 0
    while n < limit:
        n += 1
        item = next_item(cands, ledger, state, code=code)
        if item is None:
            out.append({"label": "done", "kind": "done", "why": "no work left"})
            break
        out.append({"label": item["label"], "kind": item["kind"], "candidate": item["candidate"],
                    "seed": item["seed"], "why": item["why"],
                    "recipe_id": (item.get("recipe") or {}).get("recipe_id"),
                    "chain": (item.get("recipe") or {}).get("chain"),
                    "stages": [str(s["stage"]) for s in item.get("stages") or []]})
        if item["kind"] == "await":
            break
        _simulate_apply(cands, ledger, state, item, outcome, code)
    return out


def _simulate_apply(cands: dict, ledger: dict, state: dict, item: dict, outcome: str, code: str) -> None:
    """Record a synthetic finished experiment + entry for ``item`` (the dry run's only fiction)."""
    lad = state["ladder"]
    xid = f"S{len(lad['experiments']) + 1:02d}"
    recipe = item.get("recipe") or item.get("base_recipe") or {}
    stages = [str(s["stage"]) for s in item.get("stages") or []]
    if item["kind"] not in ("pooled", "milestone_i", "milestone_close", "skip"):
        lad["experiments"][xid] = {
            "xid": xid, "kind": ("baseline" if item["kind"] == "baseline" else item["kind"]),
            "candidate": item["candidate"], "seed": item["seed"], "chain": recipe.get("chain"),
            "base_recipe_id": (item.get("base_recipe") or recipe).get("recipe_id"),
            "recipe_id": recipe.get("recipe_id"), "code_sha": code, "status": "ok",
            "started": f"sim-{xid}", "ckpt": f"/sim/{xid}/model.pt", "stages": stages,
            "screen": {"completion": 0.5}, "base_rank": _rank_for(cands, item), "eval_dirs": []}
    kind = item["kind"]
    # NOTE(stroke formation 2026-09-26): under `pair_on: [always]` every clear s3 line is paired, so
    # the simulated entry is speculative and its `pooled` item closes it (`final`). Without `always`
    # both branches below are exactly the old simulation.
    always = "always" in set(map(str, ((cands.get("decision_rule") or {}).get("pair_on") or ())))
    if kind == "pooled" and always:
        entry = entry_by_id(ledger, item.get("decision"))
        if entry is not None:
            entry.update(status="final", pair=False)
        return
    if kind == "candidate":
        did = _next_did(ledger)
        ledger["entries"].append({"id": did, "kind": "candidate", "candidate": item["candidate"],
                                  "experiment": xid, "chosen": outcome, "options": CANDIDATE_OPTIONS,
                                  "status": "speculative" if always else "decided", "confidence": "clear",
                                  **({"pair": True} if always else {}),
                                  "base_recipe_id": (item.get("base_recipe") or {}).get("recipe_id"),
                                  "recipe_id_if_adopted": recipe.get("recipe_id"), "code_sha": code,
                                  "conditional_refs": []})
    elif kind == "skip":
        did = _next_did(ledger)
        ledger["entries"].append({"id": did, "kind": "candidate", "candidate": item["candidate"],
                                  "chosen": "SKIP", "options": CANDIDATE_OPTIONS, "status": "skipped",
                                  "evidence": {"predicate": item.get("extra", {}).get("predicate")},
                                  "base_recipe_id": (item.get("base_recipe") or {}).get("recipe_id"),
                                  "conditional_refs": conditional_refs(
                                      (candidate_by_id(cands, item["candidate"]) or {}).get("conditional")),
                                  "code_sha": code})
    elif kind == "baseline":
        did = _next_did(ledger)
        ledger["entries"].append({"id": did, "kind": "baseline", "candidate": item["candidate"],
                                  "experiment": xid, "chosen": "PASS", "options": ["PASS"],
                                  "status": "final", "recipe_id_if_adopted": recipe.get("recipe_id"),
                                  "base_recipe_id": recipe.get("recipe_id"), "code_sha": code})
    elif kind in ("milestone_i",):
        ledger["entries"].append({"id": _next_mid(ledger, item["candidate"]), "kind": "milestone",
                                  "candidate": item["candidate"], "chosen": None,
                                  "options": MILESTONE_OPTIONS, "status": "open",
                                  "recipe_id": recipe.get("recipe_id"), "code_sha": code,
                                  "eligibility": 1})
    elif kind in ("milestone", "milestone_close"):
        entry = newest_milestone_entry(ledger, item["candidate"])
        if entry is None:
            ledger["entries"].append({"id": _next_mid(ledger, item["candidate"]), "kind": "milestone",
                                      "candidate": item["candidate"], "chosen": "PASS",
                                      "options": MILESTONE_OPTIONS, "status": "final",
                                      "recipe_id": recipe.get("recipe_id"), "experiment": xid,
                                      "code_sha": code, "eligibility": 1})
        else:
            entry.update(chosen="PASS", status="final", experiment=xid,
                         recipe_id=recipe.get("recipe_id"))


def _rank_for(cands: dict, item: dict) -> dict:
    kind, cid = item["kind"], item["candidate"]
    if kind == "baseline":
        return {"upto": None, "inclusive": False}
    if kind in ("milestone", "batched_s2", "milestone_s2", "milestone_i", "milestone_close"):
        after = None
        for mstone in cands.get("milestones") or []:
            if mstone.get("id") == cid:
                after = mstone.get("after")
        return {"upto": after, "inclusive": True}
    if kind not in ("candidate", "seed2", "twin"):
        return {"upto": None, "inclusive": False}      # the smoke hook and anything else rank-free
    return {"upto": cid, "inclusive": False}


def _next_did(ledger: dict) -> str:
    """The next ``D<nn>`` id, in WRITE ORDER -- one past the highest, never filling a gap.

    The fixed five (``D00`` resolution, ``D01`` turn rate, ``D02`` B0 s3, ``D03`` the dead_walker
    floor, ``D04`` B0 s2) fall out of the order the driver writes them in, and candidates start at
    ``D05``. Monotonic rather than smallest-free because every notification quotes an id: re-using
    one would make two different lines name the same decision.
    """
    used = [e.get("id") for e in entries(ledger)]
    nums = [int(i[1:]) for i in used if isinstance(i, str) and re.fullmatch(r"D\d+", i)]
    return f"D{(max(nums) + 1) if nums else 0:02d}"


def _next_mid(ledger: dict, mid: str) -> str:
    used = {e.get("id") for e in entries(ledger)}
    if mid not in used:
        return mid
    n = 2
    while f"{mid}#{n}" in used:
        n += 1
    return f"{mid}#{n}"


# ================================================================== the run context
#: Set by the SIGTERM handler and by nothing else. The handler ONLY sets this and signals the child:
#: it never exits, never writes and never decides, so the exit path is the main loop's ordinary
#: save + park, inside the unit's 360-s stop budget (`TimeoutStopSec=360`). The WORST case is
#: ~300 s, not 240: a poll tick <= 120 s, then `_stop_child`'s SIGTERM + `TERM_GRACE_S` 120 s, then
#: -- only if the child is still alive -- SIGKILL and ANOTHER `wait(60)` for it to be reaped, then
#: the save. The post-SIGKILL wait is the term the step-3 budget note and the launcher comment
#: leave out; it still fits, with 60 s to spare.
INTERRUPT = {"signal": None}


class Ctx:
    """Everything one pass of the loop needs: the campaign record, the ledger and the derived view."""

    def __init__(self, cdir: Path, cands: dict, args):
        self.dir = Path(cdir)
        self.cands = cands
        self.args = args
        self.plant = cands["plant"]
        self.seed_screen = str((cands.get("seeds") or {}).get("screen", 3))
        self.seed_confirm = str((cands.get("seeds") or {}).get("confirm", 2))
        self.c = rp.Campaign(self.dir, self.plant, self.seed_screen)
        self.ledger = lc.read_ledger(self.dir)
        self.code = ""
        #: The CURRENT ``eval_sha`` (plan D8), set by ``main`` beside ``code``. Stamped on every stage
        #: rec that ran an eval, on ``header.eval_sha`` together with ``header.screen``, on every floor
        #: verdict (``header.floor_eval_sha``) and on every ledger entry; ``lad.eval_sha`` is the
        #: recorded value ``ok_screen`` compares against.
        self.eval_sha = ""
        self.tools = ""
        self.view = {"experiments": {}, "entries": {}, "candidates": []}
        self._ovr_stamp = None
        self.t0 = time.time()
        self.resolution = {}
        lad = self.c.st.setdefault("ladder", {})
        for key, default in (("candidates_sha256", None), ("code_sha", None), ("unit", None),
                             ("resolution_ids", []), ("pending", []), ("user_choice", {}),
                             ("experiments", {}), ("squashed", {}), ("applied_overrides", []),
                             ("stage_trace", {}), ("d03", None)):
            lad.setdefault(key, default)

    # -- convenience
    @property
    def lad(self) -> dict:
        return self.c.st["ladder"]

    @property
    def state(self) -> dict:
        return self.c.st

    def save(self) -> None:
        self.lad["pending"] = [e.get("id") for e in pending_decisions(self.ledger, set(self.view["entries"]))]
        self.c.save()

    def write_ledger(self) -> None:
        lc.write_ledger(self.dir, self.ledger)

    def recompute(self) -> dict:
        """Recompute the derived squash view (after EVERY ledger write) and the pending cache."""
        self.view = squash_view(self.ledger, self.cands, self.state, code=self.code)
        self.lad["pending"] = [e.get("id") for e in pending_decisions(self.ledger, set(self.view["entries"]))]
        return self.view

    def notify(self, tag: str, ident: str, fields) -> str:
        line = lc.format_notify(tag, ident, fields)
        self.c.notify(line)
        return line

    def cli(self, verb: str) -> str:
        return f"{PY} {HERE / 'ledger_cli.py'} --campaign-dir {self.dir} {verb}"


# ================================================================== interrupts
def install_sigterm(ctx: Ctx) -> None:
    """The stop handler. It ONLY sets the flag and terminates the child -- it never exits or writes.

    DEVIATION from the plan's "install the SIGTERM handler": SIGINT gets the same handler. A Ctrl-C
    in an attached terminal is the same intent as `systemctl --user stop`, and the default SIGINT
    (a `KeyboardInterrupt` raised at an arbitrary bytecode) would unwind through `train`'s
    `BaseException` branch, SIGKILL nothing but leave the ledger write half-done -- while a SIGKILL
    of a live Kit is itself a DerivedDataCache corruption risk (the Context's standing constraint:
    every stop path is SIGTERM first). The cost is that Ctrl-C no longer force-aborts the driver:
    the stop is honoured at the next poll tick, inside the same 360-s budget as a unit stop.
    """
    def handler(signum, _frame):
        INTERRUPT["signal"] = signum
        child = getattr(rp, "CURRENT_CHILD", None)
        if child is not None:
            try:
                child.terminate()
            except OSError:
                pass
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):                                    # pragma: no cover
            pass


def stop_requested(ctx: Ctx) -> str | None:
    if (ctx.dir / lc.STOP_NAME).exists():
        return "stop"
    if INTERRUPT["signal"] is not None:
        return "sigterm"
    return None


def interrupt_reason(ctx: Ctx, xid: str | None) -> str | None:
    """Why the running experiment should stop, or ``None``. Called in EVERY poll tick.

    Four reasons, and all four are cheap: the ``STOP`` file, the SIGTERM flag, the derived squash set
    (an override that landed while this window trained), and a user ``skip`` of this candidate. The
    overrides file is only re-read when its size or mtime moved, so a tick costs one ``stat``.
    """
    why = stop_requested(ctx)
    if why:
        return why
    if ingest_overrides(ctx, cheap=True) and xid:
        ctx.recompute()
    if xid and xid in ctx.view["experiments"]:
        return f"squashed: {ctx.view['experiments'][xid]}"
    header = (ctx.lad.get("experiments") or {}).get(xid) or {}
    cid = header.get("candidate")
    uc = (ctx.lad.get("user_choice") or {}).get(cid) or {}
    # The same exemption as derived-view cause (d), for the same reason (decision 7: "the base
    # recipe's twin is never interrupted or squashed"). A running twin whose booking candidate the
    # user has just skipped is still the base recipe's s2 screen, and killing it two hours in costs a
    # window that the next pair -- or M<k>.s2 -- has to pay again. A twin the squash view DOES list
    # (cause (a): its base recipe really moved) is still caught by the check above this one.
    if uc.get("choice") == "skip" and str(uc.get("at") or "") > str(header.get("started") or "") \
            and header.get("kind") != "twin":
        return f"user skip of {cid}"
    return None


# ================================================================== overrides -> the ledger
def _history(entry: dict, to: str, by: str, why: str) -> None:
    entry.setdefault("history", []).append({"at": now_iso(), "from": entry.get("status"), "to": to,
                                            "by": by, "why": why})


def ingest_overrides(ctx: Ctx, *, cheap: bool = False) -> bool:
    """Apply every override line not yet applied. Returns True when the ledger changed.

    Re-scans the whole file (cheap: one JSON object per line) and dedupes by uuid, so a line written
    while the driver was stopped is applied at the next launch and a line written mid-window is
    applied within one poll tick. ``cheap`` skips the work when the file has not moved -- the tick
    path calls it on every poll.
    """
    path = ctx.dir / lc.OVERRIDES_NAME
    try:
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
    except OSError:
        stamp = None
    if cheap and stamp == ctx._ovr_stamp:
        return False
    ctx._ovr_stamp = stamp
    applied = list(ctx.lad.get("applied_overrides") or [])
    changed = False
    notes: list[tuple[str, str]] = []
    for rec in lc.read_overrides(ctx.dir):
        if rec["uuid"] in applied:
            continue
        # The uuid is recorded as applied BEFORE the apply runs, so that the `ctx.save()` inside
        # `squash_and_report` (which happens mid-apply) carries it: SQUASH (6) writes `applied_at`
        # after the state is saved, and a crash in the old order left the ledger changed and the
        # line un-applied, which the next launch replayed. A refusal is still recorded as applied --
        # a line that cannot be applied must not be retried at every poll.
        applied.append(rec["uuid"])
        ctx.lad["applied_overrides"] = applied
        try:
            note = _apply_override(ctx, rec)
        except (ValueError, KeyError) as exc:
            note = f"REFUSED: {exc}"
            ctx.c.log(f"override {rec['uuid']}: {note}")
        changed = True
        notes.append((rec["uuid"], note))
    if changed:
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        # The record is saved BEFORE the line claims it, and the line names what the queue resumes
        # at -- the plan's contract example (`resume: C3a under recipe 3b9e...`), which is what the
        # Monitor relays.
        resume = _resume_field(ctx)
        for uuid, note in notes:
            ctx.notify("OVERRIDE-APPLIED", uuid, [note, resume])
    return changed


def _resume_field(ctx: Ctx) -> str:
    """``resume: <next item> under recipe <id>`` -- derived, and never allowed to raise.

    The migration comes first (plan D9 / item 30): while an eval-only migration item is pending the
    loop runs IT before any resume or queue item, so the line names it."""
    try:
        mig = migration_item(ctx)
        if mig is not None:
            return f"resume: {mig['label']} (migration, eval-only)"
        nxt = next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view,
                        resolution=ctx.resolution)
    except Exception as exc:                                            # noqa: BLE001
        return f"resume: {ctx.dir.name} (the queue could not be derived here: {exc})"
    if nxt is None:
        return "resume: nothing queued (the campaign is done)"
    rid_ = ((nxt.get("recipe") or nxt.get("base_recipe") or {}) or {}).get("recipe_id")
    return f"resume: {nxt['label']}" + (f" under recipe {rid_}" if rid_ else "")


def _apply_override(ctx: Ctx, rec: dict) -> str:
    """One override line -> the ledger. Returns the OVERRIDE-APPLIED note."""
    if rec.get("ack_flag"):
        return _apply_flag_ack(ctx, rec)
    if rec.get("decision"):
        return _apply_decision_override(ctx, rec)
    if rec.get("candidate"):
        return _apply_candidate_choice(ctx, rec)
    if rec.get("milestone"):
        return _apply_milestone_answer(ctx, rec)
    raise ValueError(f"override {rec} names neither a decision, a candidate nor a milestone")


def _apply_flag_ack(ctx: Ctx, rec: dict) -> str:
    """``ledger_cli ack-flag <Did>`` (clock-deprecation plan step 4): release a flag park.

    Recorded on the entry as ``flag_acks[] = {uuid, at, why, level}`` -- the queue re-derivation
    (:func:`flag_park`) honours an ack at the flags' CURRENT level only, so an ack typed against the
    seed-3 line never releases the pooled line that replaces it (the CLI stamps the level it saw; a
    mismatch is refused and the user acks the new line). No status moves: the decision stands, the
    user has only said "seen, go on". Idempotent by uuid (a crash between the ledger write and the
    ``applied_overrides`` save replays the line).
    """
    did = rec["ack_flag"]
    entry = entry_by_id(ctx.ledger, did)
    if entry is None:
        raise KeyError(f"no ledger entry {did}")
    if any(a.get("uuid") == rec["uuid"] for a in entry.get("flag_acks") or []):
        ctx.recompute()
        return f"{did}: flags already acknowledged by this line (ingested before the record was saved)"
    level = (entry.get("flags") or {}).get("level")
    if not (entry.get("flags") or {}).get("park"):
        raise ValueError(f"{did} carries no parked flag (nothing to acknowledge)")
    if rec.get("level") and level and str(rec["level"]) != str(level):
        raise ValueError(f"the ack names {did}'s {rec['level']} line, but its flags are now on the "
                         f"{level} line (it landed after the ack was typed): read it and ack again")
    entry.setdefault("flag_acks", []).append({"uuid": rec["uuid"], "at": rec.get("at"),
                                              "why": rec.get("why"), "level": level,
                                              "applied_at": now_iso()})
    ctx.write_ledger()
    ctx.recompute()
    return f"{did}: flags acknowledged on its {level} line -- the park is released"


def _live_window(header: dict) -> bool:
    """Would the next launch continue this window? (``find_inflight``'s own condition.)

    A header is live while it is ``running``, and also while it sits at one of the two infra halts
    with its ``inflight`` block intact -- the plan resumes those at the eval. Squashing such a
    header has to record that the window STOPPED, or the archived copy goes on claiming it trains.
    """
    status = header.get("status")
    return status == "running" or (status in RESUMABLE_HALTS and bool(header.get("inflight")))


def _drop_own_pair(ctx: Ctx, entry: dict, why: str) -> str:
    """A user settling a SPECULATIVE decision drops its own seed-2 run (never the base twin).

    The base recipe's twin is the BASE's s2 screen -- later pairs and ``M<k>.s2`` need it -- so it is
    never interrupted or squashed here. The candidate's own s2 is evidence about a question the user
    has just answered, so it is squashed ``by: user_choice`` and never resurrected (cause e).
    """
    xid = entry.get("seed2_experiment")
    if not xid:
        return ""
    header = (ctx.lad.get("experiments") or {}).get(xid) or (ctx.lad.get("squashed") or {}).get(xid)
    if not header:
        return ""
    archived = dict(header, squashed_by="user_choice", reason=f"{why} (cause e)")
    if _live_window(header):
        # The live header is popped below, so the `_mark_interrupt` the running stage loop performs
        # later lands on an orphan: record the outcome here instead of leaving the archived copy
        # claiming the window is still live.
        archived["status"] = "interrupted"
        archived["interrupt"] = why
    ctx.lad.setdefault("squashed", {})[xid] = archived
    ctx.lad.get("experiments", {}).pop(xid, None)
    return f"; its own seed-2 run {xid} squashed_by user_choice"


def _apply_decision_override(ctx: Ctx, rec: dict) -> str:
    did = rec["decision"]
    entry = entry_by_id(ctx.ledger, did)
    if entry is None:
        raise KeyError(f"no ledger entry {did}")
    refusal = lc.unoverrulable_note(entry)
    if refusal:
        raise ValueError(refusal)
    # Already applied (the uuid is the override's own id): a crash between the ledger write and the
    # `applied_overrides` save re-reads this line at the next launch, and answering the user's
    # standing `confirm` with `REFUSED: illegal ledger transition 'confirmed' -> 'confirmed'` told
    # the Monitor the answer had been rejected when it had in fact been applied.
    if (entry.get("override") or {}).get("id") == rec["uuid"]:
        ctx.recompute()
        return (f"{did} -> {lc.effective_choice(entry)} ({entry.get('status')}): already applied "
                "(this line was ingested before the record was saved)")
    action = rec.get("action") or ("confirm" if rec.get("choice") == lc.effective_choice(entry) else "overrule")
    was_speculative = entry.get("status") == "speculative"
    before = lc.effective_choice(entry)
    target = "confirmed" if action == "confirm" else "overruled"
    check_transition(entry.get("status"), target, "user",
                     override_by=(entry.get("override") or {}).get("by"))
    # The plan's transition table: "`overruled(driver)` -> `confirmed` (user `confirm` freezes the
    # entry's `chosen`, i.e. REVERTS the driver's override + resurrection)". `chosen` is the pick the
    # machine made from the entry's OWN screen; a driver self-overrule or a milestone drop replaced
    # it with the override's. So a `confirm` of such an entry restores `chosen` and lets the
    # `choice != before` branch below run SQUASH -- which resurrects the dependents the override
    # squashed. Without it `confirm` froze the DRIVER's pick and the two commands the SELF-OVERRULE
    # line quotes (`confirm <Did>` / `overrule <Did> --choose <alternative>`) both left it standing,
    # i.e. the user had no quoted way back.
    choice = rec.get("choice") if action == "overrule" else lc.confirm_choice(entry)
    if entry.get("options") and choice not in entry["options"]:
        raise ValueError(f"{choice!r} is not one of {did}'s options {entry['options']}")
    _history(entry, target, "user", f"{action}: {rec.get('why')}")
    entry["status"] = target
    entry["frozen"] = True
    entry["override"] = {"id": rec["uuid"], "by": "user", "choice": choice, "why": rec.get("why"),
                         "at": rec.get("at"), "applied_at": now_iso()}
    note = f"{did} -> {choice} ({target})"
    if was_speculative:
        note += _drop_own_pair(ctx, entry, f"user {action} of {did} while speculative")
    ctx.write_ledger()
    if choice != before:
        note += squash_and_report(ctx, f"user {action} of {did}: {before} -> {choice}")
    else:
        ctx.recompute()
    if action == "confirm":
        # The overrule case is covered by ingest's own OVERRIDE-APPLIED line; a CONFIRM is its own
        # event (the decision is now frozen and no later pair may move it).
        ctx.notify("CONFIRM", did, [f"user confirm -> {choice} (frozen)", rec.get("why") or ""])
    return note


def _apply_candidate_choice(ctx: Ctx, rec: dict) -> str:
    cid = rec["candidate"]
    if candidate_by_id(ctx.cands, cid) is None:
        raise KeyError(f"{cid} is not a candidate id")
    choice = rec.get("choice")
    if choice not in ("run", "skip"):
        raise ValueError(f"candidate choice {choice!r} must be run or skip")
    entry = newest_entry(ctx.ledger, cid, set(ctx.view["entries"]))
    # EVERY validation runs BEFORE the standing choice is recorded. `ingest_overrides` catches the
    # raise and records the line as applied, so a mutation made first would SURVIVE the refusal --
    # and a standing `user_choice[cid] = skip` is not inert: derived-view cause (d) squashes every
    # header of that candidate started before its stamp (never resurrected), and `next_item` step 8
    # then emits the very SKIP the refusal just denied. A refused line must change nothing.
    if choice == "skip" and entry is not None and entry.get("kind") in ("candidate", "seed2") \
            and lc.effective_choice(entry) in ("ADOPT", "REJECT"):
        raise ValueError(f"{cid} already carries a decision ({entry['id']} -> "
                         f"{lc.effective_choice(entry)}): use confirm/overrule, not decide skip")
    ctx.lad.setdefault("user_choice", {})[cid] = {"choice": choice, "uuid": rec["uuid"],
                                                 "at": rec.get("at"), "why": rec.get("why")}
    note = f"user_choice[{cid}] = {choice}"
    if choice == "run":
        if entry is not None and (entry.get("override") or {}).get("id") == rec["uuid"]:
            # The same already-applied case as `_apply_decision_override`'s: re-applying would hit
            # `squashed -> squashed (user)`, which no transition row allows, and relay a REFUSED for
            # a squash that stands. (`newest_entry` no longer returns it once the view lists it, so
            # this bites only while the view has not been recomputed yet.)
            ctx.recompute()
            return note + f"; {entry['id']} was already squashed by this line"
        if entry is not None:
            was_speculative = entry.get("status") == "speculative"
            before = lc.effective_choice(entry)
            check_transition(entry.get("status"), "squashed", "user")
            _history(entry, "squashed", "user", "decide run")
            entry["pre_squash_status"] = entry.get("status")
            entry["status"] = "squashed"
            entry["squashed_by"] = "user"
            entry["override"] = {"id": rec["uuid"], "by": "user", "choice": before,
                                 "why": "decide run", "at": rec.get("at"), "applied_at": now_iso()}
            if was_speculative:
                note += _drop_own_pair(ctx, entry, f"user decide run of {cid} while {entry['id']} was speculative")
            ctx.write_ledger()
            froze = ([f"frozen: {entry['id']} ({before})"]
                     if entry.get("frozen") and entry.get("pre_squash_status") in
                     ("confirmed", "overruled") else [])
            note += squash_and_report(ctx, f"user decide {cid} run (squashes {entry['id']})",
                                      extra_frozen=froze)
        else:
            ctx.write_ledger()
            ctx.recompute()
        note += f"; {cid} is undecided and re-screens at its rank"
    else:
        # Idempotent: `applied_overrides` reaches `state.json` only at the END of
        # `ingest_overrides`, and `squash_and_report` below saves state in between -- so a crash in
        # that window re-applies this line at the next launch. Confirm/overrule are idempotent
        # anyway; a SKIP entry is the one non-idempotent write, and a second one would make the
        # candidate look answered twice.
        already = next((e for e in entries(ctx.ledger)
                        if e.get("candidate") == cid and e.get("status") == "skipped"
                        and (e.get("evidence") or {}).get("override_uuid") == rec["uuid"]), None)
        if already is not None:
            return note + f"; SKIP entry {already['id']} already written by this line"
        write_skip_entry(ctx, cid, "user", f"user: {rec.get('why') or 'decide skip'}",
                         override_uuid=rec["uuid"])
        # The derived view already squashes that candidate's running experiment (cause d); apply it,
        # so the record agrees with the view and the next launch does not try to resume it.
        note += ("; a SKIP entry is written and any in-flight run of it is interrupted"
                 + squash_and_report(ctx, f"user decide {cid} skip"))
    return note


def _apply_milestone_answer(ctx: Ctx, rec: dict) -> str:
    mid = rec["milestone"]
    block = lc.milestone_block(ctx.cands, mid)
    entry = newest_milestone_entry(ctx.ledger, mid, set(ctx.view["entries"]))
    if rec.get("drop"):
        cid = rec["drop"]
        if cid not in block:
            raise ValueError(f"{cid} is not in {mid}'s block {block}")
        target = newest_entry(ctx.ledger, cid, set(ctx.view["entries"]))
        # Already applied, by uuid -- the same window the other two verbs close (F10(b)). The drop
        # writes `ledger.json` and only `squash_and_report` saves `applied_overrides`, so a crash in
        # between replays this line at the next launch. Re-applying would hit the ADOPT guard below
        # (the target is now `overruled` -> REJECT) and relay `REFUSED: --drop is refused unless C2
        # is an ADOPT` for an answer that IS applied, by that very uuid.
        if target is not None and (target.get("override") or {}).get("id") == rec["uuid"]:
            ctx.recompute()
            return (f"{mid}: {cid} was already dropped by this line "
                    "(ingested before the record was saved)")
        if target is None or lc.effective_choice(target) != "ADOPT":
            raise ValueError(f"--drop is refused unless {cid} is an ADOPT in {mid}'s block")
        check_transition(target.get("status"), "overruled", "user")
        _history(target, "overruled", "user", f"{mid} drop")
        target["status"] = "overruled"
        target["frozen"] = True
        target["override"] = {"id": rec["uuid"], "by": "user", "choice": "REJECT",
                              "why": f"{mid} drop {cid}", "at": rec.get("at"), "applied_at": now_iso()}
        if entry is not None:
            entry["eligibility"] = int(entry.get("eligibility") or 1) + 1
            entry["sub_status"] = None
            check_transition(entry.get("status") or "open", "squashed", "user")
            _history(entry, "squashed", "user", f"{mid} re-queued after --drop {cid}")
            entry["pre_squash_status"] = entry.get("status")
            entry["status"] = "squashed"
            entry["squashed_by"] = "user"
        ctx.write_ledger()
        return f"{mid}: dropped {cid} (ADOPT -> REJECT)" + squash_and_report(ctx, f"{mid} --drop {cid}")
    # --accept: freeze every unfrozen adoption in the block, then the milestone proceeds to (ii).
    # The plan's row is defined only FROM `await` (plan:552-556), and the freeze -- not the
    # milestone's own status change -- is the part that matters: a frozen ADOPT can no longer be
    # dropped by a milestone contradiction (`milestone_i`/`_milestone_fail` skip frozen entries) nor
    # moved by a pooled self-overrule. So the guard runs BEFORE the block is touched; it used to sit
    # after it, which let an early or replayed `--accept` disarm every later contradiction silently.
    # ... but AFTER the already-applied recognition, which is the same uuid test the other two verbs
    # make (F10(b)). An `--accept` that reached `ledger.json` and not `applied_overrides` is replayed
    # at the next launch: at stage (ii) the milestone is now `final`, so the guard relayed
    # `REFUSED: M1 is not awaiting an answer (it is 'final')` for an answer that IS applied; at stage
    # (i) the guard PASSED (the status is still `await`) and the line was applied a second time,
    # growing M1's history and reporting `frozen adoptions []`.
    already = _milestone_accept_applied(ctx, mid, block, entry, rec["uuid"])
    if already:
        ctx.recompute()
        return already
    if entry is None or entry.get("status") != "await":
        raise ValueError(f"{mid} is not awaiting an answer (it is "
                         f"{'absent from the ledger' if entry is None else repr(entry.get('status'))}): "
                         f"`resume-milestone --accept` freezes every unfrozen adoption in {block} and "
                         "is legal only on a (i) or (ii) FAIL")
    frozen = []
    for cid in block:
        target = newest_entry(ctx.ledger, cid, set(ctx.view["entries"]))
        if target is None or lc.effective_choice(target) != "ADOPT":
            continue
        if target.get("frozen") or target.get("status") not in ("decided", "final"):
            continue
        check_transition(target.get("status"), "confirmed", "user")
        _history(target, "confirmed", "user", f"{mid} accept")
        target["status"] = "confirmed"
        target["frozen"] = True
        target["override"] = {"id": rec["uuid"], "by": "user", "choice": "ADOPT",
                              "why": f"{mid} accept", "at": rec.get("at"), "applied_at": now_iso()}
        frozen.append(cid)
    if (entry.get("evidence") or {}).get("stage") == "ii":
        check_transition("await", "final", "user")
        _history(entry, "final", "user", f"{mid} accept after a (ii) FAIL")
        entry["status"], entry["chosen"] = "final", "PASS"
        entry["sub_status"] = None
    else:
        check_transition("await", "accepted_i", "user")
        _history(entry, "accepted_i", "user", f"{mid} accept after a (i) FAIL")
        entry["sub_status"] = "accepted_i"
    # The stamp the replay recognises. `accepted_i` leaves the status at `await` (the promotion still
    # has to run), so without it a replayed (i)-FAIL accept passes the `await` guard; and at stage
    # (ii) the block members carry the uuid but the milestone entry alone would not.
    entry["accepted_uuid"] = rec["uuid"]
    ctx.write_ledger()
    ctx.recompute()
    return f"{mid}: accepted; frozen adoptions {frozen}"


def _milestone_accept_applied(ctx: Ctx, mid: str, block: list, entry: dict | None,
                              uuid: str) -> str | None:
    """Why a ``--accept`` needs no work: the note to relay, or ``None`` to apply it.

    Three readings of "already applied", in order of how much they prove: the milestone's own
    ``accepted_uuid`` stamp; any block member frozen by THIS uuid (an older record written before
    the stamp existed); and -- whatever the uuid -- a milestone already sitting at
    ``await/accepted_i``, whose row the plan defines once (``await -> accepted_i``). The third is
    the F2 wart: a second `--accept` froze nothing new but added a history row and reported
    ``frozen adoptions []``, which reads like a refusal that was not one.
    """
    if entry is not None and entry.get("accepted_uuid") == uuid:
        return f"{mid}: already accepted by this line (ingested before the record was saved)"
    for cid in block:
        target = newest_entry(ctx.ledger, cid, set(ctx.view["entries"]))
        if target is not None and (target.get("override") or {}).get("id") == uuid:
            return (f"{mid}: already accepted by this line ({cid} is frozen by it; ingested before "
                    "the record was saved)")
    if entry is not None and entry.get("status") == "await" and entry.get("sub_status") == "accepted_i":
        return (f"{mid}: already accepted; the promotion is queued "
                "(`await -> accepted_i` is answered once)")
    return None


def write_skip_entry(ctx: Ctx, cid: str, source: str, why: str, predicate: str | None = None,
                     override_uuid: str | None = None) -> dict:
    """A SKIP entry: TERMINAL, and it records the base recipe its predicate was evaluated at.

    The base hash is what lets the derived view squash it when an earlier override changes the recipe
    the predicate was read against -- a SKIP is a decision about a recipe, not about a candidate.
    """
    base = compose_recipe(ctx.ledger, ctx.cands, ctx.code, upto=cid, squashed=set(ctx.view["entries"]))
    cand = candidate_by_id(ctx.cands, cid) or {}
    entry = {"id": _next_did(ctx.ledger), "kind": "candidate", "candidate": cid,
             "question": cand.get("question"), "options": list(CANDIDATE_OPTIONS), "chosen": "SKIP",
             "alternative": "ADOPT", "status": "skipped", "confidence": None,
             "evidence": {"predicate": predicate, "user": source == "user", "why": why,
                          "override_uuid": override_uuid},
             "depends_on": list(cand.get("depends_on") or []),
             "conditional_refs": conditional_refs(cand.get("conditional")),
             "base_recipe_id": base["recipe_id"], "code_sha": ctx.code, "eval_sha": ctx.eval_sha,
             "tools_sha": ctx.tools,
             "decided_at": now_iso(), "decided_by": source, "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.recompute()
    ctx.save()
    ctx.notify("DECISION", entry["id"],
               [f"{cid} on recipe {base['recipe_id']}", f"chosen SKIP ({'user' if source == 'user' else 'predicate'}: "
                f"{predicate or why})",
                f"run it: {ctx.cli(f'decide {cid} --choose run --why ...')}"])
    ctx.c.changelog_row(f"{entry['id']} {cid} SKIP — {why}")
    return entry


def _wall_hours(ctx: Ctx, header: dict) -> float | None:
    """The GPU wall hours an experiment cost, summed over its stage records (``None``: unrecorded).

    ``Campaign.recs`` is bucketed by the campaign's CURRENT seed, and a squashed header is usually
    another seed's, so the bucket is addressed explicitly the way ``_baseline_raw_dir`` does it.
    """
    bucket = (ctx.state.get("keys") or {}).get(f"seed{header.get('seed')}") or {}
    total, seen = 0.0, False
    for key in header.get("recs") or []:
        wall = _num((bucket.get(key) or ctx.c.recs.get(key) or {}).get("wall_h"))
        if wall is not None:
            total, seen = total + wall, True
    return total if seen else None


def _apply_squash_view(ctx: Ctx, view: dict, *, before_x: set, moved: list, moved_ids: set,
                       resurrected: list, frozen_named: list, warn: list,
                       e_moved: list, e_moved_ids: set, e_restored: list) -> bool:
    """Make the stored records agree with ONE view. Returns True when anything moved.

    Split out of :func:`squash_and_report` so the reconcile can be iterated to a FIXPOINT: moving a
    header or restoring an entry's ``pre_squash_status`` changes the inputs the view is derived
    from, and the plan's promise ("recomputed from the ledger on every launch") only holds if the
    record the next recompute reads is already the converged one.

    ``e_moved`` / ``e_restored`` are the ENTRY half of the same report. An entry-only squash (a
    SKIP entry whose referent flipped, a carried-forward milestone whose recipe moved) moves no
    header, and while only ``moved`` was collected such a squash produced no SQUASH line, no REPORT
    block and no CHANGELOG row -- the OVERRIDE-APPLIED line said "squashed 0" for an override that
    had just voided a recorded answer.
    """
    xsq, esq = view["experiments"], view["entries"]
    changed = False
    live = ctx.lad.setdefault("experiments", {})
    dead = ctx.lad.setdefault("squashed", {})
    for xid, why in xsq.items():
        was_live = xid in live
        header = live.pop(xid, None) or dead.get(xid)
        if header is None:
            continue
        by = "user_choice" if ("cause d" in why or "cause e" in why or "user" in why.split(":")[0]) else "derived"
        archived = dict(header, squashed_by=dead.get(xid, {}).get("squashed_by", by), reason=why)
        if _live_window(header):
            # The LIVE header is popped on the next line, so the `_mark_interrupt` the running
            # stage loop performs later lands on an orphan this process discards at its next save.
            # Record the outcome here instead of leaving the archived copy claiming the window is
            # still training. `pre_squash_status` is what a resurrection restores, so a squash that
            # is later undone still hands `find_inflight` a resumable header.
            archived["pre_squash_status"] = "running"
            archived["status"] = "interrupted"
            archived.setdefault("interrupt", f"squashed: {why}")
        if was_live or dead.get(xid) != archived:
            changed = True
        dead[xid] = archived
        if xid not in before_x and xid not in moved_ids:
            hours = _wall_hours(ctx, header)
            moved_ids.add(xid)
            moved.append({"xid": xid, "why": why,
                          "named": f"{xid}{'' if hours is None else f' ({hours:.1f} h)'}"})
    for xid in list(dead):
        if xid in xsq:
            continue
        header = dead.pop(xid)
        header.pop("squashed_by", None)
        header.pop("reason", None)
        if header.pop("pre_squash_status", None) == "running":
            header["status"] = "running"
            header.pop("interrupt", None)
        live[xid] = header
        resurrected.append(xid)
        changed = True
    # The plan: "A SQUASH or MILESTONE line that squashes a `confirmed` or `overruled(user)` entry
    # names it with `frozen:`". The loop below names the entries THIS call transitions; a caller
    # that had to set `status: squashed` itself (a user `decide run`) passes its own name in.
    for e in entries(ctx.ledger):
        if e.get("id") in esq:
            if e.get("status") != "squashed":
                if e.get("status") in ("confirmed", "overruled") and e.get("frozen"):
                    frozen_named.append(f"frozen: {e['id']} ({lc.effective_choice(e)})")
                check_transition(e.get("status"), "squashed", "derived")
                _history(e, "squashed", "derived", esq[e["id"]])
                e["pre_squash_status"] = e.get("status")
                e["status"] = "squashed"
                e["squashed_by"] = "derived"
                if e["id"] not in e_moved_ids:
                    e_moved_ids.add(e["id"])
                    e_moved.append({"id": e["id"], "why": esq[e["id"]],
                                    "named": f"{e['id']} ({e.get('candidate') or e.get('kind')})"})
                changed = True
        elif e.get("status") == "squashed" and e.get("squashed_by") == "derived":
            restored, why = e.get("pre_squash_status"), "resurrected: the base hash matches again"
            if not restored:
                # No driver write can produce this (every derived squash stores the status it
                # replaced), but a hand-edited ledger can -- and it used to reach the agreement
                # assertion below, which sits BEFORE `loop()`'s halt handling, so the LAUNCH died
                # with a traceback instead of a HALT line. Loud, not fatal: the repair restores the
                # kind's own conservative NON-TERMINAL-or-re-decidable status and the WARN says the
                # record was repaired.
                #
                # Kind-aware, because `decided` is a CANDIDATE status: a milestone's own statuses
                # are open/await/final, nothing reads `decided` off a milestone entry, and a
                # milestone repaired to it was a closed entry no reader could close again --
                # `next_item` skipped it as terminal while `_previous_promoted` found no head on
                # it. `open` is the milestone's start: its (i) stage is re-booked and the close
                # re-decides it.
                restored = "open" if e.get("kind") in lc.MILESTONE_KINDS else "decided"
                why = (f"resurrected as {restored!r}: stored squashed(derived) with no "
                       "pre_squash_status (a hand-edited or pre-schema ledger)")
                warn.append(f"{e.get('id')}: {why}")
            # Through the guard like every other ledger write (the table's `squashed -> <pre>` rows).
            check_transition("squashed", restored, "derived")
            _history(e, restored, "derived", why)
            e["status"] = restored
            e.pop("pre_squash_status", None)
            e.pop("squashed_by", None)
            e_restored.append(f"{e['id']} ({restored})")
            changed = True
    return changed


#: What :func:`squash_and_report` returns when the reconcile moved NOTHING -- no header, no entry.
#: A prefix test is not enough any more: an entry-only squash keeps the "; squashed 0" head and
#: adds its own clause, and reading that as "nothing moved" is exactly the P2 wart.
NOTHING_MOVED = "; squashed 0; resurrected 0"


def squash_and_report(ctx: Ctx, trigger: str, extra_frozen: list | None = None) -> str:
    """Recompute the derived view, move headers, mark entries, and report -- the SQUASH step.

    Squash is a DERIVED view: this function never decides what is squashed, it only makes the
    recorded state agree with the recomputation (and resurrects anything whose base hash matches
    again and whose files are still there). That is what makes it crash-safe -- a relaunch recomputes
    the same set.

    RECOMPUTE -> RESTORE -> RECOMPUTE, to a fixpoint, and then an ASSERTION that the record agrees
    with the view. Honest note on the loop: since :func:`hidden_by_stored_squash` made the view a
    pure function of the EFFECTIVE ledger, one round already is the fixpoint, so no test can make a
    second round bite -- the thing that fixed the one-level-per-ingest resurrection is the purity,
    not the loop. The loop and the assertion are kept as the cheap guard against any future reader
    of the stored ``squashed`` status creeping back into the derivation: they would turn a silent
    half-resurrection (which cost 3.7-3.9 GPU-h per dependency level) into a converged record, or
    into a loud failure if it could not converge.
    """
    before_x = set(ctx.lad.get("squashed") or {})
    moved, moved_ids, resurrected, warn = [], set(), [], []
    e_moved, e_moved_ids, e_restored = [], set(), []
    frozen_named = list(extra_frozen or [])
    view = None
    for _round in range(16):
        view = ctx.recompute()
        if not _apply_squash_view(ctx, view, before_x=before_x, moved=moved, moved_ids=moved_ids,
                                  resurrected=resurrected, frozen_named=frozen_named, warn=warn,
                                  e_moved=e_moved, e_moved_ids=e_moved_ids, e_restored=e_restored):
            break
    else:                                                               # pragma: no cover - bounded
        raise RuntimeError(f"the squash reconcile did not converge in 16 rounds ({trigger})")
    # The derived half of the record must now agree with the view exactly. A `user`/`driver` squash
    # is a FACT the view does not own (causes (d)/(e), a milestone's own re-run bookkeeping), so it
    # is exempt -- `hidden_by_stored_squash` is the same test the effective-ledger reader applies.
    for e in entries(ctx.ledger):
        if hidden_by_stored_squash(e):
            continue
        if (e.get("id") in view["entries"]) != (e.get("status") == "squashed"):
            raise RuntimeError(f"squash reconcile: {e.get('id')} is stored {e.get('status')!r} "
                               f"but the view says "
                               f"{'squashed' if e.get('id') in view['entries'] else 'live'}")
    ctx.write_ledger()
    ctx.save()
    long_rows = [f"{m['named']} ({m['why']})" for m in moved]
    e_long_rows = [f"{m['named']} ({m['why']})" for m in e_moved]
    if moved or resurrected or frozen_named or warn or e_moved or e_restored:
        ctx.c.report_block(f"## SQUASHED — {trigger} — {now_iso()}\n\n"
                           + "\n".join([f"- squashed {m}" for m in long_rows]
                                       + [f"- resurrected {x}" for x in resurrected]
                                       + [f"- squashed entry {m}" for m in e_long_rows]
                                       + [f"- restored entry {x}" for x in e_restored]
                                       + [f"- {f}" for f in frozen_named]
                                       + [f"- WARN {w}" for w in warn]))
        ctx.c.changelog_row(f"SQUASH — {trigger} — squashed {len(moved)}, resurrected {len(resurrected)}"
                            + (f", entries squashed {len(e_moved)}" if e_moved else "")
                            + (f", entries restored {len(e_restored)}" if e_restored else "")
                            + (f", {len(warn)} repaired" if warn else ""))
        ctx.notify("SQUASH", trigger, [f"squashed: {', '.join(long_rows) or 'none'}",
                                       f"resurrected: {', '.join(resurrected) or 'none'}"]
                                      + ([f"entries squashed: {', '.join(e_long_rows)}"] if e_moved else [])
                                      + ([f"entries restored: {', '.join(e_restored)}"] if e_restored else [])
                                      + frozen_named + [f"WARN {w}" for w in warn])
    # The plan's OVERRIDE-APPLIED contract names the squashed experiments and what they cost
    # ("squashed: X12 (2.1 h)"), not just how many: the user is being asked to confirm an answer
    # whose price is GPU hours already spent. An ENTRY-only squash costs no GPU hours but voids a
    # recorded answer, so it is named here too rather than reading as "squashed 0".
    return (f"; squashed {len(moved)}"
            + (f": {', '.join(m['named'] for m in moved)}" if moved else "")
            + f"; resurrected {len(resurrected)}"
            + (f": {', '.join(resurrected)}" if resurrected else "")
            + (f"; entries squashed: {', '.join(m['named'] for m in e_moved)}" if e_moved else "")
            + (f"; entries restored: {', '.join(e_restored)}" if e_restored else ""))


# ================================================================== stage floors / retention / infra
#: The one-sided 99 % normal quantile (the category floor's band, plan D6).
Z_99 = 2.3263478740408408


def floor_band(p_hat: float, n_env: float) -> float:
    """The one-sided 99 % upper binomial bound ``p + z_0.99 sqrt(p(1-p)/n_env)`` (plan D6).

    A Wald bound on the category's DISTINCT-ENV count (a paired category's two hands share each env's
    startup-randomised physics, D1). ~2 % false-FAIL at the threshold; a Wilson bound at the same z
    would be the refinement if that ever matters (PLAN_AMENDMENTS 30).
    """
    p = min(1.0, max(0.0, float(p_hat)))
    return p + Z_99 * math.sqrt(p * (1.0 - p) / float(n_env))


def floor_park_categories(stage: dict | None) -> list[str]:
    """``floor.park_categories`` of one stage (stride/posture plan, audit pass 3), ``[]`` when absent.

    A ``category_survival`` floor tripped ONLY on these categories parks and flags instead of
    rejecting (:func:`floor_disposition`): the run continues to its screen and the survival guard
    against the control decides. Absent -- every campaign before 2026-09-26 -- nothing changes.
    """
    return [str(c) for c in (((stage or {}).get("floor") or {}).get("park_categories") or [])]


def floor_disposition(stage: dict, passed: bool, fails, kind: str) -> str:
    """What the run loop does with one stage's floor verdict -- pure.

    ``"pass"`` (the floor held); ``"park"`` (it failed, every failing category is one of
    :func:`floor_park_categories` -- recorded as ``header.floor_park`` on ANY run kind, flagged at
    the DECISION, never a REJECT); else the pre-existing split: ``"reject"`` for a candidate (a clear
    floor REJECT, decision 4) and ``"watch"`` for every other kind (``header.floor_watch``). With no
    ``park_categories`` the answer is exactly the old ``candidate -> reject / else watch``.
    """
    if passed:
        return "pass"
    park = set(floor_park_categories(stage))
    fails = [str(c) for c in (fails or ())]
    if park and fails and set(fails) <= park:
        return "park"
    return "reject" if kind == "candidate" else "watch"


def floor_park_flags(lines, *, where: str = "") -> list[dict]:
    """The ``floor_park`` flag items of one run's ``header.floor_park`` lines (immediate: they park
    at the line they are raised on, :data:`IMMEDIATE_FLAGS`)."""
    tag = f" ({where})" if where else ""
    return [_flag("floor_park", f"FLOOR on a park category{tag} -- park and flag, never an auto-REJECT; "
                               f"the survival guard against the control decides: {w}")
            for w in (lines or [])]


def stage_floor(stage: dict, rec: dict, *, dead_walker_floor: float | None = None,
                decisive=None, previous=None, detail: dict | None = None) -> tuple[bool, str]:
    """Decision 4's stage FLOOR: ``(passed, why)``. Never a relative check -- that is RETENTION.

    ``category_survival`` (plan D6): for every ``c`` in ``decisive`` (the stage's own trained
    categories; default the stage's ``categories``), the category's own-hold survival on the stage's
    suite row is BAND-tested -- FAIL only when the one-sided 99 % upper bound (:func:`floor_band`,
    ``n_env`` = ``cat.<c>.n_env``) is below ``min_prev`` when ``c`` was decisive at an earlier stage
    (``previous``; default ``previous_categories``) and below ``min_new`` otherwise. A decisive
    category the row does not carry raises ``EvalMismatch`` (the run loop halts ``eval_mismatch``,
    never a silent pass); "not scored (no verdict)" survives only for a stage with no eval row.

    ``survival`` (the pre-suite hold-keyed floor, kept for the v1 records): the fraction of episodes
    that reached the END of the stage's last enabled hold, on the stage's own eval row.
    ``dead_walker`` is armed inside ``train`` (``early_abort``) rather than read here; this reports
    the same verdict when a NON-candidate kind trips it, which is WATCH by decision 4.

    ``detail`` (a dict, optional): the ``category_survival`` branch writes ``detail["fails"]`` (the
    failing categories) into it, which :func:`floor_disposition` reads. Omitted, nothing changes.
    """
    floor = stage.get("floor") or {}
    kind = str(floor.get("kind") or "none")
    if kind == "none":
        return True, "no floor at this stage"
    if kind == "category_survival":
        return _category_floor(stage, rec, floor, decisive, previous, detail=detail)
    if kind == "dead_walker":
        vals = []
        log = rec.get("log")
        if log and Path(log).exists():
            vals = rp.track_ratio_series(Path(log))
        got = (sum(vals[D03_AT_ITERS - D03_WINDOW:D03_AT_ITERS]) / D03_WINDOW
               if len(vals) >= D03_AT_ITERS else None)
        want = DEAD_WALKER_FLOOR_MIN if dead_walker_floor is None else float(dead_walker_floor)
        if got is None:
            return True, f"dead_walker floor {want:.3f}: the log has no its-{D03_AT_ITERS} window yet"
        return got >= want, (f"dead_walker floor: track_ratio_mean {got:.3f} over its "
                            f"{D03_AT_ITERS - D03_WINDOW}-{D03_AT_ITERS} (need >= {want:.3f})")
    hold, row = floor.get("hold"), str(floor.get("row"))
    want = float(floor.get("min", FLOOR_SURVIVAL_MIN))
    summary = (rec.get("evals") or {}).get(row) or {}
    surv = (summary.get("survival_by_hold") or {}).get(hold)
    if surv is None:
        return True, f"survival floor on {hold}: not scored by the {row} row (no verdict)"
    return float(surv) >= want, (f"survival through {hold} on {row}: {float(surv):.3f} (need >= {want})")


def _category_floor(stage: dict, rec: dict, floor: dict, decisive, previous, *,
                    detail: dict | None = None) -> tuple[bool, str]:
    """:func:`stage_floor`'s ``category_survival`` branch (plan D6)."""
    ec = eval_categories()
    row = str(floor.get("row") or "suite")
    summary = (rec.get("evals") or {}).get(row)
    if summary is None:
        if not stage.get("eval"):
            return True, f"category floor on {row}: not scored (no verdict) -- the stage has no eval row"
        raise ec.EvalMismatch(f"the category floor reads the {row} row, which this stage did not "
                              f"evaluate (eval rows {list(stage.get('eval') or [])})")
    dec = list(decisive) if decisive is not None else list(stage.get("categories") or [])
    prev = set(previous if previous is not None else (stage.get("previous_categories") or []))
    min_new = float(floor.get("min_new") if floor.get("min_new") is not None else 0.10)
    min_prev = float(floor.get("min_prev") if floor.get("min_prev") is not None else FLOOR_SURVIVAL_MIN)
    # An explicit per-category minimum wins over the prev/new split (a checkpoint fork's floor:
    # the forked head already trained every category, the chain has no earlier stage to say so).
    by_cat = {str(k): float(v) for k, v in (floor.get("min_by_category") or {}).items()}
    cat = summary.get("cat") or {}
    parts, fails = [], []
    for c in _ordered_categories(dec):
        entry = cat.get(c) or {}
        p, n = _num(entry.get("survival")), _num(entry.get("n_env"))
        if p is None or not n:
            raise ec.EvalMismatch(f"decisive category {c} is not on the {row} row (cat.{c}.survival "
                                  f"{entry.get('survival')!r}, n_env {entry.get('n_env')!r}) -- a "
                                  "category the stage trained must be scored, never passed silently")
        if c in by_cat:
            want, which = by_cat[c], "min"
        else:
            want, which = (min_prev, "prev") if c in prev else (min_new, "new")
        bound = floor_band(p, n)
        ok = not bound < want
        parts.append(f"{c} {p:.3f} (bound {bound:.3f}, n_env {int(n)}, {which} >= {want:g})")
        if not ok:
            fails.append(c)
    why = (f"category survival on {row} (one-sided 99 % band): " + ("; ".join(parts) or "no decisive "
           "category") + (f" -- FAIL on {fails}" if fails else ""))
    if detail is not None:
        detail["fails"] = list(fails)
    return not fails, why


#: RETENTION's metric per category KIND (plan D6), beside ``survival`` (every category): a
#: translation category keeps its speed ``ratio``; ``turn`` its ``facing_closed``; ``move_turn``
#: both; ``stand`` its ``drift_speed`` (WATCH in screens, tracked here -- LOWER is better).
RETENTION_METRICS = {"stand": ("drift_speed",), "fwd": ("ratio",), "side": ("ratio",),
                     "back": ("ratio",), "diag": ("ratio",), "turn": ("facing_closed",),
                     "move_turn": ("facing_closed", "ratio")}
#: ``facing_closed`` is a signed fraction that can sit at or below 0, where a relative drop flips
#: sign or fires on noise: its RETENTION drop is ABSOLUTE, at ``max(this, R_eval)``.
RETENTION_FACING_DROP = 0.10
#: ``drift_speed`` (lower is better) is flagged on a rise of at least this many m/s AND
#: :data:`RETENTION_RATIO_DROP` relative.
RETENTION_DRIFT_RISE_M_S = 0.02
RETENTION_TRACE_KEYS = ("survival", "ratio", "facing_closed", "drift_speed")


def stage_retention(stage: dict, rec: dict, trace: dict, *, previous=None, resolution: dict | None = None,
                    legacy_trace: bool = False) -> tuple[list[str], dict]:
    """RETENTION: every PREVIOUSLY trained category (or enabled hold) against the incumbent's trace.

    WATCH + notify, never a verdict (decision 4) -- the curriculum's first run SEEDS the trace that
    later runs are compared against, so the first pass has nothing to say and says so. Returns the
    WATCH lines and the trace this stage contributes.

    A per-category stage (``categories:``, plan D6) compares every ``c`` in ``previous`` (default the
    stage's ``previous_categories`` = the union of the earlier stages' decisive sets) on the stage's
    suite row against ``stage_trace_v2`` (keyed by category): ``survival`` on an absolute 0.10 drop,
    ``ratio`` on a 20 % relative drop, ``facing_closed`` on an ABSOLUTE drop of ``max(0.10,
    R_eval)`` (``resolution``'s ``cat.<c>.facing_closed``), ``drift_speed`` on a rise of >= 0.02
    m/s AND >= 20 % (lower is better -- the per-metric sign lives here). ``legacy_trace`` says the
    incumbent's only trace is the pre-suite hold-keyed one: it cannot be compared, so this run
    re-seeds it and the line says "trace predates the suite".
    """
    if "categories" in stage:
        return _category_retention(stage, rec, trace, previous=previous, resolution=resolution,
                                   legacy_trace=legacy_trace)
    row = None
    for candidate_row in (stage.get("eval") or []):
        row = str(candidate_row)
    summary = (rec.get("evals") or {}).get(row) or {}
    surv = summary.get("survival_by_hold") or {}
    ratio = summary.get("ratio_by_hold") or {}
    mine = {"survival": {h: _num(v) for h, v in surv.items()},
            "ratio": {h: _num(v) for h, v in ratio.items()}}
    prev_holds = list(stage.get("previous_holds") or [])
    lines = []
    if not prev_holds:
        # A stage with no previously-enabled holds has nothing to retain -- the promotion stages
        # (2a/2b) declare none, so without this every milestone pushed two empty RETENTION cards.
        return lines, mine
    if not trace:
        lines.append(f"RETENTION {stage['stage']}: no incumbent trace yet -- this run seeds it "
                     f"({len(mine['survival'])} holds recorded)")
        return lines, mine
    for hold in prev_holds:
        was, now = _num((trace.get("survival") or {}).get(hold)), mine["survival"].get(hold)
        if was is not None and now is not None:
            drop = was - now
            lines.append(f"RETENTION {stage['stage']} {hold}: survival {was:.3f} -> {now:.3f}"
                         + (f" DROPPED {drop:.3f} (>{RETENTION_SURV_DROP})" if drop > RETENTION_SURV_DROP else " held"))
        wr, nr = _num((trace.get("ratio") or {}).get(hold)), mine["ratio"].get(hold)
        if wr and nr is not None:
            rel = (wr - nr) / wr
            lines.append(f"RETENTION {stage['stage']} {hold}: speed_ratio {wr:.3f} -> {nr:.3f}"
                         + (f" DROPPED {rel * 100:.0f}% (>{RETENTION_RATIO_DROP * 100:.0f}%)"
                            if rel > RETENTION_RATIO_DROP else " held"))
    return lines, mine


def _category_retention(stage: dict, rec: dict, trace: dict, *, previous=None,
                        resolution: dict | None = None, legacy_trace: bool = False
                        ) -> tuple[list[str], dict]:
    """:func:`stage_retention` on a per-category stage (plan D6)."""
    row = None
    for candidate_row in (stage.get("eval") or []):
        row = str(candidate_row)
    cat = ((rec.get("evals") or {}).get(row) or {}).get("cat") or {}
    mine = {c: {k: _num((cat.get(c) or {}).get(k)) for k in RETENTION_TRACE_KEYS}
            for c in _ordered_categories(cat)}
    prev = _ordered_categories(previous if previous is not None
                               else (stage.get("previous_categories") or []))
    lines: list[str] = []
    name = stage["stage"]
    if not prev or row is None:
        # nothing previously trained, or no eval row at this stage (c3/c5): nothing to compare and
        # nothing to seed
        return lines, mine
    if not trace:
        if legacy_trace:
            lines.append(f"RETENTION {name}: the incumbent's trace predates the suite (hold-keyed) -- "
                         f"this run re-seeds it by category ({len(mine)} categories recorded)")
        else:
            lines.append(f"RETENTION {name}: no incumbent trace yet -- this run seeds it "
                         f"({len(mine)} categories recorded)")
        return lines, mine
    for c in prev:
        was_c, now_c = trace.get(c) or {}, mine.get(c) or {}
        was, now = _num(was_c.get("survival")), _num(now_c.get("survival"))
        if was is not None and now is not None:
            drop = was - now
            lines.append(f"RETENTION {name} {c}: survival {was:.3f} -> {now:.3f}"
                         + (f" DROPPED {drop:.3f} (>{RETENTION_SURV_DROP})" if drop > RETENTION_SURV_DROP
                            else " held"))
        for key in RETENTION_METRICS.get(c, ()):
            was, now = _num(was_c.get(key)), _num(now_c.get(key))
            if was is None or now is None:
                continue
            head = f"RETENTION {name} {c}: {key} {was:.3f} -> {now:.3f}"
            if key == "ratio":
                if not was:
                    continue
                rel = (was - now) / was
                lines.append(head + (f" DROPPED {rel * 100:.0f}% (>{RETENTION_RATIO_DROP * 100:.0f}%)"
                                     if rel > RETENTION_RATIO_DROP else " held"))
            elif key == "facing_closed":
                r_eval = _num(resolution_metric(resolution or {}, f"cat.{c}.facing_closed").get("R_eval"))
                thr = max(RETENTION_FACING_DROP, r_eval or 0.0)
                drop = was - now
                lines.append(head + (f" DROPPED {drop:.3f} absolute (>{thr:.3f} = max("
                                     f"{RETENTION_FACING_DROP}, R_eval))" if drop > thr else " held"))
            elif key == "drift_speed":
                rise = now - was
                rel = (rise / was) if was > 0 else (math.inf if rise > 0 else 0.0)
                bad = rise >= RETENTION_DRIFT_RISE_M_S and rel >= RETENTION_RATIO_DROP
                lines.append(head + (f" ROSE {rise:.3f} m/s (>= {RETENTION_DRIFT_RISE_M_S} and "
                                     f">= {RETENTION_RATIO_DROP * 100:.0f}%; lower is better)"
                                     if bad else " held"))
    return lines, mine


def _oom_kills() -> int | None:
    """This cgroup's ``oom_kill`` counter -- the honest answer to "was the child OOM-killed?".

    The unit runs under ``MemoryMax=48G`` with ``OOMPolicy=continue``, so a killed eval leaves the
    driver alive and the only trace is this counter and the child's ``-9``.
    """
    try:
        rel = None
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[2]:
                rel = parts[2].lstrip("/")
        if rel is None:
            return None
        events = CGROUP_ROOT / rel / "memory.events"
        for line in events.read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


#: The DerivedDataCache signature, matched PER LINE but ORDER-FREE within it. Per line, because
#: ``"omni.datastore" in tail and "migrated" in tail`` matched a healthy log that happened to
#: mention the extension anywhere and an asset migration thousands of lines later, and the
#: consequence of a false positive is a halt telling the assistant to quarantine a cache that is
#: fine. Order-free, because ``omni\.datastore.*migrated`` also required that spelling: a line
#: reading ``cache migrated by omni.datastore plugin`` did not match, and a MISS here is the
#: expensive direction -- the driver retries an eval into a corrupted cache (the 2.4 GB/s hazard).
#: No corrupted-log wording is recorded anywhere, so both words on one line is the widest test that
#: is still a signature rather than a substring.
KIT_CACHE_SIGNATURE = (re.compile(r"omni\.datastore", re.I), re.compile(r"migrat", re.I))


def kit_cache_line(line: str) -> bool:
    """True when ONE log line carries every word of the DerivedDataCache signature, in any order."""
    return all(pat.search(line) for pat in KIT_CACHE_SIGNATURE)


def infra_signature(child_rc, oom_before: int | None, eval_log: Path | None,
                    *, since: float | None = None) -> tuple[str | None, list[str]]:
    """``(stop_now_signature, notes)`` for an eval that produced nothing.

    Four readings, in the plan's order: the cgroup's ``oom_kill`` delta, the child's return code
    (``-9`` is the OOM killer or a SIGKILL), the newest Kit log for the DerivedDataCache migration
    signature, and the eval log tail for a ``Traceback``. The first three mean RETRYING MAKES IT
    WORSE -- an OOM-killed Kit that is restarted corrupts the shader cache and every later start
    balloons -- so they stop the retry loop immediately and hand the cache to the assistant.

    ``since`` is the attempt's start time: a Kit log that has not been written since then belongs to
    an EARLIER run and cannot explain this failure. Without it, one corrupted start left a matching
    log on disk that every later infra read would have blamed the next failure on.
    """
    notes = []
    oom_now = _oom_kills()
    if oom_before is not None and oom_now is not None and oom_now > oom_before:
        return f"cgroup oom_kill {oom_before} -> {oom_now}", notes
    if child_rc == -9:
        return "the eval child returned -9 (SIGKILL: the OOM killer)", notes
    try:
        logs = sorted(KIT_LOG_DIR.glob("kit_*.log"), key=lambda p: p.stat().st_mtime)
    except OSError:
        logs = []
    if logs and (since is None or logs[-1].stat().st_mtime >= float(since)):
        tail = logs[-1].read_text(errors="ignore")[-200000:]
        if any(kit_cache_line(line) for line in tail.splitlines()):
            return f"Kit cache migration signature in {logs[-1].name} (DerivedDataCache corruption)", notes
    if eval_log and Path(eval_log).exists():
        tail = Path(eval_log).read_text(errors="ignore")[-20000:]
        if "Traceback" in tail:
            notes.append("the eval log tail carries a Traceback (a code/config fault, not memory)")
    return None, notes


# ================================================================== running one experiment
def _tag_for(ctx: Ctx, item: dict, xid: str, stage: str) -> str:
    return f"seed{item['seed']}_{xid}_{stage}_{ctx.plant.replace('+', 'p')}"


def _next_xid(ctx: Ctx) -> str:
    used = set(all_headers(ctx.state))
    n = 1
    while f"X{n:02d}" in used:
        n += 1
    return f"X{n:02d}"


def resolution_tool():
    """``resolution.py``, loaded by path and cached -- the SHARED re-score helper lives there.

    The driver owns no statistics and no re-scoring: ``resolution.summarize_run_dir`` is the one
    implementation of "read a recorded run dir the way the decision rule needs it" (survival table
    injected, ``facing_closed_by_hold`` re-scored from the saved raws), and both sides of every
    milestone comparison must go through it or the F block silently loses its closed fractions on
    the recorded side (step 3's hand-off).
    """
    global _RESOLUTION_MOD
    if _RESOLUTION_MOD is None:
        _RESOLUTION_MOD = _load(HERE / "resolution.py", "resolution_ladder")
    return _RESOLUTION_MOD


_RESOLUTION_MOD = None


def summarize_run_dir(run_dir: str | Path, sid: str) -> dict | None:
    """``summarize()`` of a recorded eval row, with ``survival_by_hold`` injected.

    ``_run_eval`` injects the survival table IN MEMORY only, so a milestone or a resolution that
    re-reads a run dir must inject it itself or both conditional turn survivals come back ``None`` --
    a silent loss of two of the four guards (step 3's hand-off).

    ``facing_closed_by_hold`` needs the same treatment for the opposite reason: an aggregate written
    before the key existed carries ``{hold: None}``, so the three closed fractions of the F block
    would be MISSING on one side of a milestone comparison. ``resolution.py``'s shared helper
    re-scores them from ``raw/*.npz`` (or reads the per-episode key a new harness wrote). It is a
    hard read of real files, so a failure falls back to the plain summary with the reason recorded
    under ``facing_closed_via`` rather than taking a booked milestone down.
    """
    path = Path(run_dir) / "scenario_metrics.json"
    if not path.exists():
        return None
    try:
        out = resolution_tool().summarize_run_dir(run_dir, ring_proxy=False, verify=True)
        out["run_dir"] = str(run_dir)
        out["facing_closed_via"] = "resolution.summarize_run_dir (re-scored)"
        return out
    except Exception as exc:                                            # noqa: BLE001
        fallback = f"resolution.summarize_run_dir refused this row ({exc}); summarize() only"
    m = json.loads(path.read_text())
    a = m.get("aggregate") or {}
    if a.get("suite"):
        # Plan D10's third injection site: a category-suite row's survival is each hold's OWN
        # (`aggregate.suite.survival_by_hold`, the harness's own table). The cumulative walk below
        # would count the settle hold and every earlier wave's label, and read nonsense.
        a["survival_by_hold"] = dict(a["suite"].get("survival_by_hold") or {})
    elif not a.get("survival_by_hold"):
        try:
            a["survival_by_hold"] = rp.survival_by_hold(
                m.get("episodes") or [], (m.get("run_meta") or {}).get("hold_labels") or [],
                rp._manifest_hold_s(sid), dt=float((m.get("run_meta") or {}).get("dt") or 0.02))
        except Exception:                                               # noqa: BLE001
            pass
    out = rp.summarize(a)
    out["run_dir"] = str(run_dir)
    out["facing_closed_via"] = fallback
    return out


def _resume_ckpt_for(ctx: Ctx, item: dict, stage: dict, prev_ckpt: str | None) -> tuple[str | None, str]:
    kind = resume_kind(stage)
    if kind == "scratch":
        return None, "scratch"
    if kind == "previous_stage":
        return prev_ckpt, "previous stage"
    if kind == "checkpoint":
        # A named fork (every seed of the recipe forks the same file; the seed only moves the run).
        # Absolute, because the trainer's cwd is parkour/; None when the file is gone, which the
        # caller halts `resume_head_missing` on before any GPU is booked.
        rel = resume_checkpoint(stage)
        path = repo_path(rel)
        return (str(path) if path.is_file() else None), f"checkpoint {rel}"
    if kind == "reference":
        # NOTE(clock-deprecation 2026-09-23): the recorded head of a `references:` run, resolved NOW
        # (at launch) -- None when that reference has not finished under this code, which the caller
        # halts `resume_head_missing` on before any GPU is booked.
        name = resume_reference(stage)
        rec = reference_head(ctx, name)
        head = (rec or {}).get("ckpt")
        return ((head if head and Path(head).is_file() else None),
                f"reference {name}" + (f" ({rec.get('xid')} s{rec.get('seed')})" if rec else ""))
    seed = ctx.seed_screen if kind == "recipe_head" else ctx.seed_confirm
    rid = (item.get("recipe") or {}).get("recipe_id")
    head = recipe_head(ctx.state, ctx.code, rid, seed, ctx.view)
    return head, f"{kind} (recipe {rid} s{seed})"


def _record_stage(ctx: Ctx, item: dict, xid: str, stage: dict, spec, tag: str, ckpt, status, log,
                  env: dict, t0: float) -> dict:
    rec = {"phase": str(stage["stage"]), "tag": tag, "task": spec.task, "mode": spec.teacher_mode,
           "ckpt": ckpt, "status": status, "resume": None, "from_scratch": bool(stage.get("from_scratch")),
           "iters": int(stage["iters"]), "log": str(log), "wall_h": (time.time() - t0) / 3600.0,
           # `env` keeps the run_phases shape (the two fixed keys); `overlay` is the ladder's own
           # record of what this stage ran with -- state["current"]["extra_env"] is overwritten by
           # the next stage, so the header/record is the only durable copy.
           "env": {"KRABBY_OMNI_PHASE": spec.name, "KRABBY_PLANT": ctx.plant},
           "overlay": dict(env), "xid": xid, "kind": item["kind"], "seed": item["seed"], "evals": {}}
    if log and Path(log).exists():
        rec["smoke"] = rp.smoke_fields(Path(log))
        rec["exposure"] = rp.exposure_from_log(Path(log))
        rec["track_ratio_tail"] = rp.track_ratio_tail(rp.track_ratio_series(Path(log)))
        rec["track_ratio_move_tail"] = rp.track_ratio_tail(
            rp.track_ratio_series(Path(log), key="track_ratio_move_mean"))
        rec["pool_series_tail"] = {k: (v[-1] if v else None)
                                  for k, v in rp.pool_series_from_log(Path(log)).items()}
    return rec


def ramp_stage(overlay: dict | None) -> bool:
    """Does a stage's OVERLAY carry a step-keyed schedule (``KRABBY_PHASEOUT`` or
    ``KRABBY_OMNI_CLOCK_HIDE_RAMP``)? Such a window is never continued: it re-runs whole (plan step 4).
    The overlay only: a preset's own PHASEOUT (2a/2b promotions) resumes exactly as before."""
    return any(str((overlay or {}).get(k) or "").strip()
               for k in ("KRABBY_PHASEOUT", "KRABBY_OMNI_CLOCK_HIDE_RAMP"))


def _stage_spec_of(ctx: Ctx, header: dict, sname: str) -> dict | None:
    """The stage dict a header ran ``sname`` under: its recorded ``stage_specs`` (a LIST, looked up
    by ``stage``), else its chain in candidates.yaml, else the promotion stages."""
    for spec in header.get("stage_specs") or []:
        if str(spec.get("stage")) == str(sname):
            return spec
    chain = header.get("chain")
    if chain and chain in (ctx.cands.get("chains") or {}):
        for spec in chain_stages(ctx.cands, chain):
            if str(spec.get("stage")) == str(sname):
                return spec
    for spec in promotion_stage_list(ctx.cands):
        if str(spec.get("stage")) == str(sname):
            return spec
    return None


def header_trained_categories(ctx: Ctx, header: dict, sname: str, *, stage: dict | None = None
                              ) -> dict | None:
    """``header.trained_categories_by_stage[sname]`` -- COMPUTED AND APPENDED when absent (plan D4).

    ``trained_categories(ph.phase_env(stage.base) | header.overlay_by_stage[stage])``: the stage's
    COMPOSED training env as the window recorded it (the recipe's adopted edits included), never the
    chain overlay alone. A header that predates the record (X01-X05, every milestone header) gets
    the key appended -- a new key, nothing else rewritten; the caller saves. ``None`` when the stage
    cannot be found or its sets cannot be derived (a zero stand share): the reader halts
    ``eval_mismatch``.
    """
    table = header.get("trained_categories_by_stage") or {}
    if str(sname) in table:
        return table[str(sname)]
    stage = stage or _stage_spec_of(ctx, header, sname)
    if stage is None:
        return None
    overlay = (header.get("overlay_by_stage") or {}).get(str(sname)) or {}
    try:
        tc = eval_categories().trained_categories(_effective_stage_cfg(ctx.cands, stage, overlay))
    except ValueError as exc:
        ctx.c.log(f"{header.get('xid')}:{sname}: trained categories cannot be derived ({exc})")
        return None
    header.setdefault("trained_categories_by_stage", {})[str(sname)] = tc.to_dict()
    return header["trained_categories_by_stage"][str(sname)]


def previous_decisive(ctx: Ctx, header: dict, names, sname: str) -> list[str]:
    """``previous_categories`` from the HEADER (plan D6): the union of ``decisive`` over every stage
    before ``sname`` in ``names`` -- ``eval: []`` stages included, each from its own composed env."""
    out: set = set()
    for name in names:
        if str(name) == str(sname):
            break
        tc = header_trained_categories(ctx, header, str(name))
        out |= set((tc or {}).get("decisive") or ())
    return _ordered_categories(out)


def _filtered_phase_checks(ctx: Ctx, rec: dict, spec) -> list[tuple[str, bool]]:
    """``rp.phase_checks`` minus its two STAGE-KEYED entries, which ``overlay_checks`` replaces.

    The dropped pair (``dir_band matches stage`` / ``RSI source matches stage``) compares the dump
    against the PRESET's expectation for that stage name, which a ladder overlay deliberately
    changes: the curriculum runs on the ``1a`` preset with no band and the merged bank.

    NOTE(omni phase-2 plan step 4, 2026-10-06): ``rp.phase_checks``' ring check ("ring generator
    installed (omni_ring_sparse/dense ...)") is GEOMETRY-AWARE here (:func:`geometry_phase_check`):
    a field window's dump has no ``omni_ring_*`` key by design, so the verbatim check would record a
    false FAIL on every field record. A ring window keeps the run_phases check unchanged (same name,
    same test), so every ring record reads exactly as before.
    """
    out = []
    for name, ok in rp.phase_checks(rec, spec, ctx.c):
        if name.startswith("dir_band matches stage") or name.startswith("RSI source matches stage"):
            continue
        if name.startswith(RP_RING_CHECK_PREFIX):
            name, ok = geometry_phase_check(rec, spec, (name, ok))
        out.append((name, ok))
    return out


#: The prefix of ``run_phases.phase_checks``' ring-generator check (run_phases.py is eval-bearing and
#: frozen; the swap happens in :func:`_filtered_phase_checks`).
RP_RING_CHECK_PREFIX = "ring generator installed"


def record_terrain_geom(rec: dict, spec) -> str:
    """The geometry a window was LAUNCHED with: its recorded overlay's ``KRABBY_FLAT_TERRAIN_GEOM``,
    else its base preset's (``OMNI_COMMON`` sets ``ring`` in every omni preset), ``strip().lower()``.
    From the launch record, never from the dump the check then reads -- so a field window whose dump
    shows a ring (the install hook did not run) FAILS instead of being judged as a ring."""
    raw = (rec.get("overlay") or {}).get(TERRAIN_GEOM_VAR)
    if raw is None:
        try:
            raw = ph.phase_env(str(spec.name)).get(TERRAIN_GEOM_VAR)
        except KeyError:
            raw = None
    return str(raw or "").strip().lower()


def geometry_phase_check(rec: dict, spec, ring_check: tuple[str, bool]) -> tuple[str, bool]:
    """The generator check for the window's OWN geometry (:func:`record_terrain_geom`).

    ``field``: the dump's generator is a field (:func:`dump_terrain_geom` -- flat + ``omni_field_*``
    sub-terrains, no ring one) and both field keys (:data:`FIELD_SUB_TERRAIN_KEYS`) are present.
    Anything else (``ring``, or a value the config guard would have refused): ``ring_check``, the
    run_phases check as it came in.
    """
    if record_terrain_geom(rec, spec) != _terrain().FIELD_GEOM:
        return ring_check
    ckpt = rec.get("ckpt")
    envy = Path(ckpt).parent / "params" / "env.yaml" if ckpt else None
    txt = envy.read_text(errors="ignore") if envy is not None and envy.exists() else ""
    subs = _sub_terrains(txt)
    ok = dump_terrain_geom(txt) == "field" and all(
        subs.get(key, (None, ""))[0] == "field" for key in FIELD_SUB_TERRAIN_KEYS)
    return (f"field generator installed ({'/'.join(FIELD_SUB_TERRAIN_KEYS)} sub-terrains, no "
            "omni_ring_*) in params/env.yaml", ok)


def _link_pair_run(ctx: Ctx, item: dict, xid: str) -> None:
    """Write ``ledger[D].seed2_experiment = xid`` the moment the seed-2 HEADER is created.

    Derived-view cause (e) and decision 7's "a user confirm or overrule of a speculative decision
    drops the candidate's own seed-2 run ... and, if it is in flight, interrupts and squashes it"
    both find that run through this field. Writing it only at pair resolution (where the pooled
    decision also records it) would mean the link does not exist for the entire 3.9 h the run is in
    flight -- exactly the window the drill exercises -- so a confirm would leave the child training
    and the next launch would resume it, ending in a pooled decision on a frozen entry.
    """
    if item.get("kind") != "seed2" or not item.get("decision"):
        return
    entry = entry_by_id(ctx.ledger, item["decision"])
    if entry is None or entry.get("seed2_experiment") == xid:
        return
    entry["seed2_experiment"] = xid
    ctx.write_ledger()
    ctx.recompute()


def _supersedes(ctx: Ctx, item: dict) -> list[str]:
    """SQUASH (5): "reruns get new ids with ``supersedes``" -- the windows this one replaces.

    Every SQUASHED header of the same candidate and kind, oldest first. A record only: the ledger
    entry carries ``prior_user_choice`` for the user's side of a re-screen, and nothing on the
    EXPERIMENT said which window it supersedes, so a reader of a re-run could not follow it back to
    the squashed one (or to the GPU hours it cost) without re-deriving the whole view by hand.
    """
    dead = ctx.lad.get("squashed") or {}
    rows = sorted((h.get("started") or "", xid) for xid, h in dead.items()
                  if h.get("candidate") == item.get("candidate") and h.get("kind") == item["kind"])
    return [xid for _at, xid in rows]


_RUN_DIR_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<name>.+)$")


def _log_for_ckpt(ctx: Ctx, ckpt: str, fallback: Path) -> Path:
    """The training log of the run whose directory holds ``ckpt``. A run directory is
    ``<timestamp>_<run_name>`` and ``rp.train`` writes ``<tag>_train.log`` with ``tag == run_name``
    (or ``<run_name>_r2`` after ``train_retry``'s own infra retry); the newest existing one wins.
    ``fallback`` when the directory name does not parse or no such log exists."""
    m = _RUN_DIR_STAMP.match(Path(ckpt).parent.name)
    if not m:
        return fallback
    name = m.group("name")
    found = [p for p in (ctx.c.logs / f"{name}_train.log", ctx.c.logs / f"{name}_r2_train.log")
             if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else fallback


def run_experiment(ctx: Ctx, item: dict) -> dict:
    """One experiment: its stages in order, each trained then evaluated, recorded and audited.

    The header is written BEFORE the first launch and carries the overlay, the two recipe hashes, the
    code hash and ``inflight`` -- so a stop, a crash or a kill leaves a record a relaunch can resume
    from, and an override that lands mid-window can find and squash it.
    """
    xid = item.get("xid") or _next_xid(ctx)
    recipe = item.get("recipe") or {}
    header = (ctx.lad.get("experiments") or {}).get(xid) or {
        "xid": xid, "kind": item["kind"], "candidate": item["candidate"], "seed": item["seed"],
        "label": item["label"], "chain": recipe.get("chain"),
        "base_recipe_id": (item.get("base_recipe") or recipe).get("recipe_id"),
        "recipe_id": recipe.get("recipe_id"), "base_rank": _rank_for(ctx.cands, item),
        "code_sha": ctx.code, "tools_sha": ctx.tools,
        "overlay_by_stage": {str(s["stage"]): stage_env(recipe, str(s["stage"]))
                             for s in item.get("stages") or []},
        "started": now_iso(), "inflight": None, "status": "running", "ckpt": None,
        "eval_dirs": [], "recs": [], "decision": item.get("decision"),
        "eligibility": item.get("eligibility"), "stages": [str(s["stage"]) for s in item.get("stages") or []],
        # Read back by `item_from_header` on a resume, so a resumed row's eval-carry assertion
        # checks the RECIPE's union and not only the keys that happen to sit in the overlay.
        "eval_carry": list(recipe.get("eval_carry") or []),
        # The adoptions this window's recipe is made of. `_recipe_cand_spec` reads them to decide
        # whether the comparison runs in ABSOLUTE units (decision 5: a recipe that moved the command
        # cap/clip/ellipse is scored on F_abs/T_abs, because those knobs are `eval_carry` and the two
        # sides' ratio denominators differ). Without it on the header a RESUMED milestone promotion
        # -- the normal path after an infra halt -- scored `yaw_progress` against `blocks.F` instead
        # of `net_yaw` against `blocks.F_abs`: a different statistic at a different threshold.
        "adopted": list(recipe.get("adopted") or []),
        # Informational: the drift anchor at the moment the window was booked. The comparison itself
        # recomposes it (see `_dispatch_candidate`); this is here so the record says what it was.
        "anchor": recipe.get("anchor"),
        # The stage dicts themselves, so a resume rebuilds the work item from the HEADER even when
        # the chain is not in candidates.yaml (the --smoke-stage hook) or has since been edited.
        "stage_specs": json.loads(json.dumps(list(item.get("stages") or []))),
        "supersedes": _supersedes(ctx, item),
    }
    ctx.lad.setdefault("experiments", {})[xid] = header
    # Plan D4: the trained categories of EVERY stage (eval: [] ones included -- D6's min_prev split
    # and RETENTION's previous_categories read them), from the composed env this header records. A
    # header that predates the record gets it appended here on resume; a stage whose sets cannot be
    # derived is left absent and halts eval_mismatch where it is read.
    for stage in item.get("stages") or []:
        header_trained_categories(ctx, header, str(stage["stage"]), stage=stage)
    if header.get("status") in RESUMABLE_HALTS:
        # A relaunch after an infra halt the plan says to continue (see RESUMABLE_HALTS): the
        # window is live again, and the halt it is continuing from is kept as a record.
        header.setdefault("halts", []).append({"at": now_iso(), "status": header["status"],
                                              "why": header.get("halt")})
        header["status"] = "running"
    elif header.get("status") == RESCORED_AWAITING:
        # Plan D9 item 6: step 8 dispatched a re-opened conditional header (its predicate holds).
        # From here on it is an ordinary running window -- every later resume goes through
        # `find_inflight` / `prepare_resume`, and `_live_window` covers it.
        header.setdefault("history", []).append({"at": now_iso(), "from": RESCORED_AWAITING,
                                                "to": "running", "why": "step-8 dispatch"})
        header["status"] = "running"
    _link_pair_run(ctx, item, xid)
    ctx.save()
    rp.set_seed(ctx.c, item["seed"])
    prev_ckpt = None
    done = list(header.get("recs") or [])
    for stage in item.get("stages") or []:
        sname = str(stage["stage"])
        key = f"{xid}:{sname}"
        if key in done:                                   # resumed: this stage already landed
            prev_ckpt = (ctx.c.recs.get(key) or {}).get("ckpt")
            continue
        why = interrupt_reason(ctx, xid)
        if why:
            _mark_interrupt(ctx, header, why)
            return header
        spec = ph.PHASES[str(stage["base"])]
        env = stage_env(recipe, sname)
        tag = _tag_for(ctx, item, xid, sname)
        resume, resume_why = _resume_ckpt_for(ctx, item, stage, prev_ckpt)
        if resume is None and resume_kind(stage) != "scratch":
            return _halt(ctx, header, "resume_head_missing",
                         f"{sname} resumes from {resume_why} and there is no such checkpoint")
        iters = int(stage["iters"])
        train_tag, orig_resume = tag, resume
        skip_train, ckpt, status = False, None, "ok"
        log = ctx.c.logs / f"{tag}_train.log"
        resumed = header.get("inflight") or {}
        if str(resumed.get("stage")) == sname and resumed.get("remaining") is not None:
            # A relaunch after a stop. The RECORD keeps the original window (its length and its
            # resume point); only the training call runs the remainder, under a "_c" tag so the run
            # directory is identifiable. `remaining <= 0` means the window had already finished
            # between the last checkpoint and the stop, so there is nothing left to train.
            orig_resume = resumed.get("resume", resume)
            tag = resumed.get("tag") or tag
            remaining = int(resumed["remaining"])
            # ``remaining <= 1`` is the FINISHED case, not "one iteration to go": the formula is
            # ``target_final - iter + 1`` (the phase driver's, so a resumed window trains exactly the
            # right remainder -- ckpt 600 of a 0..999 window leaves 400), and it returns 1 exactly
            # when the checkpoint IS the window's final iteration. Re-launching Kit for two minutes
            # to re-run that one iteration would produce a checkpoint at the same number.
            if remaining <= 1 and resumed.get("ckpt"):
                skip_train, ckpt = True, resumed["ckpt"]
                # The log of the run that WROTE this checkpoint (the newest "_w"/"_c" run, found
                # by `prepare_resume`), not the original tag's: removal verification reads its
                # tail, and the original process may have died mid-window.
                log = _log_for_ckpt(ctx, ckpt, ctx.c.logs / f"{tag}_train.log")
                ctx.c.log(f"{tag}: the window had already reached its final iteration before the "
                          f"stop -- continuing at its eval")
            elif ramp_stage(env):
                # NOTE(clock-deprecation 2026-09-23, plan step 4): a KRABBY_PHASEOUT / HIDE_RAMP
                # schedule is keyed on `common_step_counter`, which restarts at 0 in EVERY process: a
                # continuation would replay the ramp from its start on a head that is already past
                # it. The WHOLE window re-runs from its original resume point instead (a "_w" run).
                iters, resume, train_tag = int(stage["iters"]), orig_resume, f"{tag}_w"
                ctx.c.log(f"{tag}: the overlay carries a step-keyed ramp (KRABBY_PHASEOUT / "
                          f"KRABBY_OMNI_CLOCK_HIDE_RAMP) and the env counter restarts per process -- "
                          f"the WHOLE {iters}-iteration window re-runs from {orig_resume or 'scratch'} "
                          f"({remaining} iterations were left)")
            else:
                iters, resume, train_tag = remaining, (resumed.get("ckpt") or resume), f"{tag}_c"
        header["inflight"] = {"stage": sname, "iters": int(stage["iters"]), "tag": tag,
                             "resume": orig_resume, "started": now_iso()}
        ctx.save()
        t0 = time.time()
        # Both gates are armed on "this CALL trains from scratch", not on the stage's static
        # `from_scratch` flag, because `resume`/`iters` have just been rewritten by the resume block
        # above. `run_phase` arms them the same way (`resume_ckpt is None`, run_phases.py) and says
        # why: the episode-length-growth ratio `dead_plant` reads is biased on a RESUMED window (the
        # episode length is already long from the checkpoint) and the verdict has no escape hatch;
        # `dead_walker` would likewise measure the continuation's own its-800-1000 against a floor
        # recorded for an absolute its-800-1000, turning a user's stop into a clear REJECT. On a
        # first run `resume is None` is exactly `stage.from_scratch` (`_resume_ckpt_for` returns
        # None only for `resume_from: scratch`), so nothing changes for an unresumed window.
        from_scratch_now = bool(stage.get("from_scratch")) and resume is None
        early = item["kind"] == "candidate" and from_scratch_now and iters > D03_AT_ITERS
        # NOTE(clock-deprecation, 2026-09-24): a REFERENCE chain is exempt from the dead_plant
        # soundness check as it is from the dead_walker abort -- its formation readout must never be
        # truncated. Z (never clocked, from scratch) forms slowly: seed 2's episode length plateaued
        # at ~650-700 steps by 2000 its and was killed twice, while seed 3 (~860) passed and recovered
        # by c6. The check guards clocked formation windows, not a reference measuring whether
        # formation happens at all.
        sound = item["kind"] != "reference" and from_scratch_now and iters >= rp.SMOKE_ITERS
        if not skip_train:
            # Plan: "`_wait_isaac_clear()` before every train/eval". The per-EXPERIMENT wait in
            # `loop` does not cover a chain's second and later stages, and the previous stage's eval
            # child may still be tearing a Kit down on the one GPU. Cheap when nothing is running
            # (one `pgrep`), and it returns at once while a stop is pending.
            rp._wait_isaac_clear(lambda: interrupt_reason(ctx, xid))
            ckpt, status, log = rp.train_retry(
                ctx.c, spec, train_tag, iters, resume, sound, early,
                extra_env=env, interrupt_check=lambda: interrupt_reason(ctx, xid), run_name=train_tag,
                abort_track_ratio_min=(ctx.lad.get("d03") if early else None))
        rec = _record_stage(ctx, item, xid, stage, spec, train_tag, ckpt, status, log, env, t0)
        rec["resume"] = orig_resume
        rec["orig_tag"] = tag
        # `interrupt_reason`, not `stop_requested`: the plan's test is "if the interrupt flag is
        # set", and a SQUASH or a user `skip` is an interrupt too. Without it such a verdict took
        # the experiment-level retry, whose own `interrupt_check` returns at once -- straight into
        # the interrupted-after-a-retry case below.
        why_int = interrupt_reason(ctx, xid)
        if status == "interrupted" or (status in ("infra", "nan", "dead_plant") and why_int):
            ctx.c.recs[f"{xid}:{sname}"] = rec
            _mark_interrupt(ctx, header, why_int or stop_requested(ctx))
            return header
        if status in ("infra", "nan", "dead_plant"):
            if not header.get("retried_stage") == sname:
                header["retried_stage"] = sname
                ctx.c.log(f"{tag}: {status} — one experiment-level retry of this stage")
                ctx.save()
                # The same settle `train_retry`'s OWN retry does (`_wait_isaac_clear` +
                # `RETRY_WAIT_S`), which this level had neither of: `_stop_child` swallows a
                # `TimeoutExpired` after `proc.kill()`, so a child in uninterruptible state is left
                # unreaped and this line relaunched a 256-env Kit on the one GPU with zero settle.
                # The pause is TICKED, so a stop inside it ends the experiment `interrupted`.
                rp._wait_isaac_clear(lambda: interrupt_reason(ctx, xid))
                if rp._interruptible_sleep(rp.RETRY_WAIT_S, lambda: interrupt_reason(ctx, xid),
                                           tick=INFRA_TICK_S):
                    ctx.c.recs[f"{xid}:{sname}"] = rec
                    _mark_interrupt(ctx, header, interrupt_reason(ctx, xid)
                                    or "stop: during the experiment-level retry wait")
                    return header
                ckpt, status, log = rp.train_retry(
                    ctx.c, spec, f"{train_tag}_e2", iters, resume, sound, early, extra_env=env,
                    interrupt_check=lambda: interrupt_reason(ctx, xid), run_name=f"{train_tag}_e2",
                    abort_track_ratio_min=(ctx.lad.get("d03") if early else None))
                rec = _record_stage(ctx, item, xid, stage, spec, f"{train_tag}_e2", ckpt, status, log,
                                    env, t0)
                rec["resume"] = orig_resume
                rec["orig_tag"] = tag
                # The SAME interrupt test as before the retry (Main loop: "status interrupted ->
                # exp.status=interrupted; break"). Without it a stop, a squash or a user `skip` that
                # lands DURING the retry falls through with `ckpt is None`, `overlay_checks` reports
                # "overlay audit needs a checkpoint" and the driver halts `config_mismatch` -- a
                # HALT that blames a config mismatch that did not happen, leaves the header
                # unresumable, and makes the stop drill unreadable after any infra death. The eval
                # path already tests after its own retry; this is the train path's half.
                why_int = interrupt_reason(ctx, xid)
                if status == "interrupted" or (status in ("infra", "nan", "dead_plant") and why_int):
                    ctx.c.recs[f"{xid}:{sname}"] = rec
                    _mark_interrupt(ctx, header, why_int or stop_requested(ctx))
                    return header
            if status in ("infra", "nan", "dead_plant"):
                ctx.c.recs[f"{xid}:{sname}"] = rec
                return _halt(ctx, header, status, f"{sname} died {status} twice", item=item)
        if status == "dead_walker":
            rec["checks"] = _filtered_phase_checks(ctx, rec, spec) if ckpt else []
            ctx.c.recs[f"{xid}:{sname}"] = rec
            # `train` already made the verdict; `stage_floor` only re-reads the log to SAY it. When
            # the record's log does not carry the its-1000 window (an aborted run's log is
            # truncated) `stage_floor` answers "no window yet", which contradicts the verdict it is
            # explaining -- so quote the abort itself instead.
            floor_passed, floor_why = stage_floor(stage, rec, dead_walker_floor=ctx.lad.get("d03"))
            if floor_passed:
                floor_why = (f"aborted by train at its {D03_AT_ITERS} (early_abort, floor "
                             f"{fmt_num(ctx.lad.get('d03') or DEAD_WALKER_FLOOR_MIN)}); the measured "
                             f"track_ratio_mean is in {rec.get('log')}")
            # `floor_eval_sha` in the same write as `floor`/`floor_stage` (plan D8): every floor
            # verdict names the eval it was measured under. A dead_walker REJECT is a training-log
            # verdict and is never rescored, but the stamp is uniform so no reader has to special-case it.
            header.update(status="rejected_by_floor", floor_stage=sname, ckpt=ckpt,
                          floor=f"dead_walker at {sname}: {floor_why}",
                          floor_eval_sha=ctx.eval_sha, inflight=None)
            ctx.save()
            return header
        # ---- the record: smoke, exposure, the filtered phase checks and the overlay audit
        rec["checks"] = _filtered_phase_checks(ctx, rec, spec)
        rec["overlay_checks"] = overlay_checks(rec, env)
        if any(not ok for _n, ok in rec["overlay_checks"]):
            ctx.c.recs[f"{xid}:{sname}"] = rec
            bad = [n for n, ok in rec["overlay_checks"] if not ok]
            return _halt(ctx, header, "config_mismatch", f"{sname}: the overlay did not reach the "
                                                         f"config: {bad}", item=item)
        # ---- evals
        # Plan D4/D7: a SUITE row is told the stage's trained categories (`--trained-categories`,
        # recorded in run_meta; scoring never reads it). Legacy rows get no flag (the harness refuses
        # it there), so their command line is exactly what it always was.
        stage_tc = header_trained_categories(ctx, header, sname, stage=stage)
        for row in (stage.get("eval") or []):
            row = str(row)
            summary, fail = _eval_row_infra(ctx, item, header, spec, ckpt, row, tag, env, xid,
                                            where=f"{sname}/{row}", ident=xid, trained=stage_tc)
            if fail is not None:
                what, detail = fail
                if what == "interrupted_wait":
                    _mark_interrupt(ctx, header, detail)
                    return header
                ctx.c.recs[f"{xid}:{sname}"] = rec
                if what == "interrupted":
                    _mark_interrupt(ctx, header, detail)
                    return header
                return _halt(ctx, header, what, detail, item=item)
            rec["evals"][row] = summary
            if summary.get("run_dir"):
                header.setdefault("eval_dirs", []).append(summary["run_dir"])
            bad = _carry_mismatch(ctx, item, summary, env, spec)
            if bad:
                ctx.c.recs[f"{xid}:{sname}"] = rec
                return _halt(ctx, header, "eval_mismatch", f"{sname}/{row}: {bad}", item=item)
        if stage.get("eval"):
            # Plan D8: the eval a stage's rows were measured under, on the rec itself.
            rec["eval_sha"] = ctx.eval_sha
        # ---- floor + retention
        per_category = "categories" in stage or str((stage.get("floor") or {}).get("kind")) == \
            "category_survival"
        decisive = previous = None
        if per_category:
            names = [str(s["stage"]) for s in item.get("stages") or []]
            tc = header_trained_categories(ctx, header, sname, stage=stage)
            if tc is None:
                ctx.c.recs[f"{xid}:{sname}"] = rec
                return _halt(ctx, header, "eval_mismatch",
                             f"{sname}: the stage's trained categories cannot be derived from its "
                             "composed env (plan D4: a zero stand share is not scorable)", item=item)
            decisive = list(tc.get("decisive") or [])
            previous = previous_decisive(ctx, header, names, sname)
            rec["categories"] = {"decisive": decisive, "previous": previous}
        floor_detail: dict = {}
        try:
            passed, floor_why = stage_floor(stage, rec, dead_walker_floor=ctx.lad.get("d03"),
                                            decisive=decisive, previous=previous, detail=floor_detail)
        except eval_categories().EvalMismatch as exc:
            ctx.c.recs[f"{xid}:{sname}"] = rec
            return _halt(ctx, header, "eval_mismatch", f"{sname}: {exc}", item=item)
        disposition = floor_disposition(stage, passed, floor_detail.get("fails"), item["kind"])
        rec["floor"] = {"passed": passed, "why": floor_why}
        if disposition == "park":
            rec["floor"]["park"] = True
        chain = recipe.get("chain")
        if per_category:
            # stage_trace_v2 is keyed by CATEGORY (plan D6); the pre-suite hold-keyed trace cannot be
            # compared with it and is re-seeded with the line "trace predates the suite".
            trace = ((ctx.lad.get("stage_trace_v2") or {}).get(chain) or {}).get(sname) or {}
            legacy = bool(((ctx.lad.get("stage_trace") or {}).get(chain) or {}).get(sname))
            lines, mine = stage_retention(stage, rec, trace, previous=previous,
                                          resolution=ctx.resolution, legacy_trace=legacy)
        else:
            trace = ((ctx.lad.get("stage_trace") or {}).get(chain) or {}).get(sname) or {}
            lines, mine = stage_retention(stage, rec, trace)
        rec["retention"] = lines
        if lines:
            ctx.notify("RETENTION", f"{xid}:{sname}", lines)
        if per_category:
            if not trace and mine:
                ctx.lad.setdefault("stage_trace_v2", {}).setdefault(chain, {})[sname] = mine
        elif not trace and mine.get("survival"):
            ctx.lad.setdefault("stage_trace", {}).setdefault(chain, {})[sname] = mine
        if disposition == "park":
            # NOTE(stride/posture plan, audit pass 3, 2026-09-26): a floor tripped ONLY on the
            # stage's `floor.park_categories` parks and flags on every run kind -- the run continues
            # to its screen, its DECISION (or baseline) line carries an immediate `floor_park` flag,
            # and the survival guard against the control decides. Never a REJECT, never floor_watch.
            header.setdefault("floor_park", []).append(f"{sname}: {floor_why}")
            ctx.notify("RETENTION", f"{xid}:{sname}",
                       [f"FLOOR tripped only on park categories {floor_park_categories(stage)} on a "
                        f"{item['kind']} run -- PARK AND FLAG: the run continues to its screen and "
                        f"its decision line parks (floor_park): {floor_why}"])
        elif not passed:
            if item["kind"] == "candidate":
                header.update(status="rejected_by_floor", floor_stage=sname, floor=floor_why, ckpt=ckpt,
                              floor_eval_sha=ctx.eval_sha, inflight=None)
                ctx.c.recs[f"{xid}:{sname}"] = rec
                ctx.save()
                return header
            header.setdefault("floor_watch", []).append(f"{sname}: {floor_why}")
            ctx.notify("RETENTION", f"{xid}:{sname}",
                       [f"FLOOR tripped on a {item['kind']} run (WATCH, the run continues to its "
                        f"screen -- decision 4): {floor_why}"])
        prev_ckpt = ckpt
        header.update(ckpt=ckpt, inflight=None)
        header.setdefault("recs", []).append(f"{xid}:{sname}")
        ctx.c.recs[f"{xid}:{sname}"] = rec
        ctx.save()
        ctx.c.report_block(_stage_report(ctx, item, xid, stage, rec, spec))
        ingest_overrides(ctx)
    header["status"] = "ok"
    header["inflight"] = None
    header["screen"] = _screen_of(ctx, item, xid)
    # Plan D8: `header.eval_sha` is (re)written TOGETHER with `header.screen` -- it names the eval
    # that produced the screen, not the header's birth, so a pre-suite header that resumes under the
    # new eval (X05, a re-opened X03) qualifies as an ok screen once this lands, and is never
    # rescreened.
    header["eval_sha"] = ctx.eval_sha
    ctx.save()
    return header


def _sid_for(row: str) -> str:
    return row if row.endswith("__omni") else f"{row}__omni"


def _run_row(ctx: Ctx, item: dict, header: dict, spec, ckpt: str, row: str, tag: str, env: dict,
             xid: str | None, *, trained_categories=None):
    """One eval row, ONE attempt: the driver owns the retry cadence (the infra detector sits between).

    ``carry_overlay=env`` is the whole point of the seam: the eval must be scored through the
    observation and command semantics the window actually trained under, and ``eval_env_for`` filters
    the overlay to ``EVAL_CARRY`` so the sampler and reward keys cannot leak into an eval.

    ``trained_categories`` (a CSV, plan D7) is forwarded as ``--trained-categories`` and is passed
    ONLY for a category-suite row -- the harness refuses the flag on every other row, so a legacy
    row's call (and its command line) is exactly what it always was.
    """
    check = lambda: interrupt_reason(ctx, xid)                            # noqa: E731
    # Plan: "`_wait_isaac_clear()` before every train/eval". `_run_eval` waits AFTER a failed
    # attempt (amendment 10), never before one -- so the row that follows a stage's training child,
    # or the second of three rows run back to back in `run_phase`, could Popen while the previous
    # Kit was still on the GPU. One `pgrep` when nothing is running, and it returns at once during
    # a stop.
    rp._wait_isaac_clear(check)
    if row == "ring":
        return rp.eval_obst(ctx.c, spec, ckpt, tag, carry_overlay=env, interrupt_check=check,
                            max_attempts=1)
    extra = {"trained_categories": trained_categories} \
        if trained_categories and _eval_row_kind(row) == "category_suite" else {}
    return rp.eval_morph(ctx.c, spec, ckpt, row, tag, carry_overlay=env, interrupt_check=check,
                         max_attempts=1, **extra)


def _row_or_error(ctx: Ctx, item: dict, header: dict, spec, ckpt: str, row: str, tag: str,
                  env: dict, xid: str | None, *, trained_categories=None
                  ) -> tuple[dict | None, str | None]:
    """:func:`_run_row`, with an exception from the eval seam turned into a NAMED halt reason.

    The live case is ``run_phases._plant_guard``, which raises when an eval spawned the wrong plant
    (the rung-iv defect) -- a config/eval mismatch, which the Vocabulary says is a ``halt``. Without
    this the exception escapes ``run_experiment`` and ``loop`` and the driver dies with a traceback:
    ``phase`` still reads ``run``, the header stays ``running`` with its ``inflight`` block, and the
    next launch RESUMES the same stage and re-raises -- a relaunch loop on a deterministic fault,
    with only the heartbeat's "unit not running while state says run" to show for it.

    ``Exception`` only: ``KeyboardInterrupt`` and ``SystemExit`` still propagate.
    """
    try:
        return _run_row(ctx, item, header, spec, ckpt, row, tag, env, xid,
                        trained_categories=trained_categories), None
    except Exception as exc:                                            # noqa: BLE001
        ctx.c.log(f"{tag}/{row}: the eval seam raised {type(exc).__name__}: {exc}")
        return None, f"{type(exc).__name__}: {exc}"


def _trained_csv(tc: dict | None) -> str | None:
    """``--trained-categories``' value: the stage's PRESENT categories (decisive + watch), in wave
    order, or None when the stage recorded none (the flag is then not passed at all)."""
    cats = list((tc or {}).get("present") or []) or list((tc or {}).get("decisive") or [])
    return ",".join(_ordered_categories(cats)) if cats else None


def _eval_row_infra(ctx: Ctx, item: dict, header: dict, spec, ckpt: str, row: str, tag: str,
                    env: dict, xid: str | None, *, where: str, ident: str, trained: dict | None = None,
                    resume_note: str = "the experiment resumes at this eval"
                    ) -> tuple[dict | None, tuple[str, str] | None]:
    """ONE eval row through ``run_experiment``'s infra path, shared with the migration (plan D9).

    The OOM / DerivedDataCache signature read, the INFRA line with the quarantine advice, the ONE
    ticked retry after :data:`INFRA_WAIT_S`, each attempt through :func:`_row_or_error` -- and NO
    verdict: the caller decides what an outcome means (``run_experiment`` halts the header; the
    migration halts the CAMPAIGN and never touches a recorded header). Returns ``(summary, None)`` on
    a scored row, else ``(None, (what, detail))`` with ``what`` one of ``eval_mismatch`` (the seam
    raised), ``eval_infra`` (nothing after the retry, or a stop-now signature), ``interrupted`` (the
    eval was stopped) or ``interrupted_wait`` (a stop during the retry wait) -- ``detail`` being the
    halt text or the interrupt reason.

    The heartbeat reads ``current.kind``; an eval row is not an ``rl`` window: a beat counts
    "episode" lines, out of 100 on a legacy row and 1000 on a suite row (ten waves of 100).
    """
    suite = _eval_row_kind(row) == "category_suite"
    csv = _trained_csv(trained) if suite else None
    log = ctx.c.logs / f"{tag}_{_sid_for(row)}_eval.log"
    ctx.state["current"] = dict(ctx.state.get("current") or {}, kind="eval",
                               tag=f"{tag}_{_sid_for(row)}", iters=1000 if suite else 100,
                               log=str(log))
    ctx.save()
    oom_before, row_t0 = _oom_kills(), time.time()
    summary, raised = _row_or_error(ctx, item, header, spec, ckpt, row, tag, env, xid,
                                    trained_categories=csv)
    if raised:
        return None, ("eval_mismatch", f"{where}: the eval seam raised {raised}")
    if summary is None:
        sig, notes = infra_signature((rp.LAST_EVAL_INFO or {}).get("rc"), oom_before, log, since=row_t0)
        if sig:
            ctx.notify("INFRA", ident, [f"{row}: {sig}", *notes,
                                        f"quarantine the Kit DerivedDataCache and relaunch; {resume_note}"])
            return None, ("eval_infra", f"{where}: {sig}")
        ctx.c.log(f"{tag}/{row}: produced nothing (no OOM/cache signature) — "
                  f"{INFRA_WAIT_S // 60} min, then ONE retry")
        # No `_wait_isaac_clear` here: `_run_row` makes it immediately before the relaunch, which is
        # the side of the INFRA_WAIT_S sleep that matters (a wait taken BEFORE a 20-minute sleep
        # proves nothing about the GPU when the retry actually starts). One per launch keeps the F7
        # invariant readable: n_stages + n_eval_rows.
        if rp._interruptible_sleep(INFRA_WAIT_S, lambda: interrupt_reason(ctx, xid), tick=INFRA_TICK_S):
            return None, ("interrupted_wait", interrupt_reason(ctx, xid) or "stop: during the infra wait")
        # The retry gets its OWN infra read. The plan makes the signature the thing that decides
        # what the assistant does ("NOTIFY INFRA with the signature ... the assistant quarantines the
        # cache and relaunches"), and an eval that fails transiently first and dies to the OOM killer
        # or the DerivedDataCache pathology on the retry used to halt with no INFRA line at all.
        oom_retry, retry_t0 = _oom_kills(), time.time()
        summary, raised = _row_or_error(ctx, item, header, spec, ckpt, row, tag, env, xid,
                                        trained_categories=csv)
        if raised:
            return None, ("eval_mismatch", f"{where}: the eval seam raised {raised}")
        if summary is None:
            sig2, notes2 = infra_signature((rp.LAST_EVAL_INFO or {}).get("rc"), oom_retry, log,
                                           since=retry_t0)
            if sig2:
                ctx.notify("INFRA", ident, [f"{row} (retry): {sig2}", *notes2,
                                            f"quarantine the Kit DerivedDataCache and relaunch; {resume_note}"])
            return None, ("eval_infra", f"{where}: no metrics after one retry"
                          + (f" ({sig2})" if sig2 else " (no OOM/cache signature)"))
    if summary.get("eval_status") == "interrupted":
        return None, ("interrupted", interrupt_reason(ctx, xid) or "stop: eval interrupted")
    return summary, None


def _carry_mismatch(ctx: Ctx, item: dict, summary: dict, env: dict, spec) -> str | None:
    """Did the eval really run with the overlay's EVAL_CARRY values (defaults included)?

    The set checked is the recipe's ``eval_carry`` union, the candidate's own, and every
    ``EVAL_CARRY`` key present in the overlay -- a mismatch means the row scored a different policy
    than the window trained, which is the 0.00-vs-0.39 defect class, so it halts.
    """
    run_dir = summary.get("run_dir")
    if not run_dir:
        return None
    meta = Path(run_dir) / "run_meta.json"
    if not meta.exists():
        return None
    got = (json.loads(meta.read_text()) or {}).get("env_vars") or {}
    recipe = item.get("recipe") or {}
    cand = candidate_by_id(ctx.cands, item.get("candidate")) or {}
    keys = set(recipe.get("eval_carry") or []) | set(cand.get("eval_carry") or [])
    keys |= {k for k in env if k in rp.EVAL_CARRY}
    bad = []
    for key in sorted(keys):
        # The expected value is the overlay's when it sets the key, else THIS STAGE's preset (the
        # eval reads EVAL_CARRY from the preset the window ran on), else the defaults injection.
        want = env.get(key, ph.phase_env(spec.name).get(key))
        if want is None:
            want = rp.EVAL_CARRY_DEFAULTS.get(key)
        if want is None:
            continue
        if str(got.get(key)) != str(want):
            bad.append(f"{key}: eval ran with {got.get(key)!r}, the window trained with {want!r}")
    return "; ".join(bad) if bad else None


def _screen_of(ctx: Ctx, item: dict, xid: str) -> dict:
    """The screen an experiment is judged by: ``summarize()`` of its FINAL stage's suite row.

    The per-category suite (``evals["suite"]``, plan D5) whenever the rec carries one -- ``slow`` is
    read ONLY on a legacy rec with no suite row, so a rescreened record (suite written beside its
    kept ``slow``) is judged on the suite and never re-files its old screen under a new
    ``eval_sha``. A suite screen carries a copy of the stage's ``trained_categories`` (from
    ``header.trained_categories_by_stage``, computed and appended when a pre-suite header lacks it,
    D4): an empty or underivable set halts ``eval_mismatch`` here rather than reaching the decision
    as an all-N/A SKIP.

    Plus the two WATCH training-log ratios, which are never scored (the plain ratio includes the
    move+turn rows a facing fix lowers by design) and are in the line so a reader sees them --
    C11's predicate reads ``screen.track_ratio_move_tail``, so they are copied on both paths.
    """
    header = all_headers(ctx.state).get(xid) or {}
    stages = [str(s["stage"]) for s in item.get("stages") or []]
    for sname in reversed(stages):
        rec = ctx.c.recs.get(f"{xid}:{sname}") or {}
        evals = rec.get("evals") or {}
        row = evals.get("suite") or evals.get("slow")
        if not row:
            continue
        out = dict(row)
        out["track_ratio_tail"] = rec.get("track_ratio_tail")
        out["track_ratio_move_tail"] = rec.get("track_ratio_move_tail")
        out["stage"] = sname
        if evals.get("suite"):
            stage = next((s for s in item.get("stages") or [] if str(s["stage"]) == sname), None)
            tc = header_trained_categories(ctx, header, sname, stage=stage) if header else None
            if not tc or not tc.get("decisive"):
                _halt(ctx, header or {"xid": xid}, "eval_mismatch",
                      f"{sname}: the suite screen has no trained categories (header."
                      f"trained_categories_by_stage[{sname}] is {'empty' if tc else 'missing'}): "
                      "nothing would be decisive -- plan D4 halts rather than letting the decision "
                      "read it as an all-N/A SKIP", item=item)
            out["trained_categories"] = json.loads(json.dumps(tc))
        return out
    return {}


def _halt(ctx: Ctx, header: dict, why: str, detail: str, item: dict | None = None):
    """Park the campaign: ``phase=halt:<xid> <why>``, a HALT notification with the verb per kind."""
    xid = header.get("xid")
    header["status"] = f"halt:{why}"
    header["halt"] = detail
    ctx.state["phase"] = f"halt:{xid} {why}"
    ctx.save()
    ctx.notify("HALT", xid, [f"{why}: {detail}", *_halt_verbs(ctx, header, item, why=why),
                             f"log: {ctx.dir / 'orchestrator.log'}"])
    ctx.c.report_block(f"## HALT {xid} — {why} — {now_iso()}\n\n- {detail}\n")
    raise SystemExit(1)


def _halt_verbs(ctx: Ctx, header: dict, item: dict | None, *, why: str = "") -> list[str]:
    """What the user can DO about this halt, per experiment kind (the plan's HALT line)."""
    kind, cid = header.get("kind"), header.get("candidate")
    if why in ("infra", "eval_infra"):
        # An infra halt is never a statement about the policy, whatever the kind: the plan's Infra
        # paragraph says "the assistant quarantines the cache and relaunches; the experiment resumes
        # at the eval". The twin/batched_s2/milestone verbs already open with that advice; the
        # `candidate` and `seed2` ones did not, so a candidate whose eval died to the Kit
        # DerivedDataCache pathology was told only to SKIP the candidate -- advice that throws away
        # a good 3.9-hour window for an infrastructure fault. The kind's own verb is kept either way.
        own = _halt_verbs(ctx, header, item)
        if any("quarantine" in v for v in own):
            return own
        return ["quarantine the Kit DerivedDataCache (mv DerivedDataCache DerivedDataCache.broken_"
                "<date>) and relaunch: the experiment resumes at this eval, nothing is re-trained",
                *own]
    if kind == "candidate":
        return [f"a repeat on the same candidate: {ctx.cli(f'decide {cid} --choose skip --why ...')}"]
    if kind == "seed2":
        did = header.get("decision")
        return [f"a repeat: {ctx.cli(f'confirm {did}')} or "
                f"{ctx.cli(f'overrule {did} --choose <the incumbent pick> --why ...')} "
                "(either drops and squashes this pair run)"]
    if kind in ("batched_s2", "milestone", "milestone_s2"):
        return ["infra only: quarantine the Kit cache and relaunch; otherwise "
                f"{ctx.cli(f'resume-milestone {cid} --drop <Cid>')} (an ADOPT in the block) changes the recipe"]
    if kind == "twin":
        return ["infra only: quarantine the Kit cache and relaunch; otherwise overrule the adoption "
                "that forms this base (overrule <Did of the last ADOPT in the base> --choose REJECT) "
                "-- a twin is the base's s2 and is not dropped by confirming the waiting decision"]
    return ["quarantine the Kit cache and relaunch"]


#: The two written Part-H gates, read by the ladder as WATCH CONTEXT only. ``run_phase`` enforces
#: them (``rp._apply_gate``) and the ladder never calls ``run_phase``, so without this the plan's
#: reuse list ("``rp.gate_1c_verdict`` / ``rp.gate_facing_verdict`` -- WATCH-only context") was
#: unrealised: a screen could fail the very formation gate the rest of the campaign enforces and no
#: record of the ladder would say so.
GATE_WATCH = {"gate_1c": "the written formation gate, on a final chain stage's suite row (its legacy "
                         "per-hand keys; slow on a pre-suite record)",
              "gate_facing": "the 2a/2b turn gate, on a promotion stage's suite row (slow on a pre-suite "
                             "record)"}


def gate_watch(name: str, screen: dict | None, *, track_ratio=None, label: str = "") -> str:
    """One ``WATCH <gate>: PASS|FAIL -- <the gate's own message>`` line. NEVER a verdict.

    Context only: nothing reads the returned string back, and no caller may let it touch ``chosen``,
    ``status``, ``confidence``, ``pooled_confidence`` or a FLOOR. The ladder decides by the decision
    rule -- a paired comparison against the incumbent at a measured half-width -- and an absolute
    written floor measured on one head cannot be allowed to override that. But a reader of a DECISION
    or MILESTONE line has to be able to see that the screen the ladder just adopted would have HALTED
    the phase pipeline, so the verdict is rendered next to the numbers.

    Two transformations of ``run_phases``' own message, both deliberate:

    * the trailing ``campaign HALTED (relaunch with --continue ...)`` clause is stripped -- it is
      ``run_phase``'s advice and is simply FALSE here (the ladder has no ``--continue`` and never
      halts on a gate), so leaving it in would read as a halt that did not happen;
    * anything the verdict function cannot read is ``n/a`` with the reason, never ``FAIL``. A stage
      with no slow row (a curriculum-only eval, an eval that produced nothing), a screen shaped
      differently (a curriculum row has no ``tripod_walking``) and a raise are all the same thing:
      no evidence. A verdict function is a foreign function inside a notification path, so it must
      not be able to raise there either.
    """
    tag = f"WATCH {name}{(' ' + label) if label else ''}"
    if not isinstance(screen, dict) or not screen:
        return f"{tag}: n/a (no slow row on this stage)"
    try:
        if name == "gate_1c":
            status, msg = rp.gate_1c_verdict(screen, track_ratio=track_ratio)
        else:
            status, msg = rp.gate_facing_verdict(screen)
    except Exception as exc:                                             # noqa: BLE001
        return f"{tag}: n/a ({type(exc).__name__}: {exc})"
    msg = re.sub(r"\s*[;—-]*\s*campaign HALTED \([^)]*\)", "", str(msg)).strip()
    verdict = "PASS" if status == "ok" else ("n/a" if "could not be evaluated" in msg else "FAIL")
    return (f"{tag}: {verdict} -- {msg}"
            + (" (WATCH only: this gate never decides in the ladder)" if verdict == "FAIL" else ""))


def _gate_row(evals: dict | None) -> dict | None:
    """The row a formation-era gate WATCH reads: the SUITE (plan D7 -- the gate verdicts stay WATCH
    context on the suite's legacy per-hand keys), the legacy ``slow`` row on a pre-suite record."""
    evals = evals or {}
    return evals.get("suite") or evals.get("slow")


def _stage_gate_watch(item: dict, stage: dict, rec: dict) -> list[str]:
    """The gate WATCH line for ONE stage: ``gate_facing`` on 2a/2b, ``gate_1c`` on a final stage.

    A non-final chain stage has no gate at all -- its eval is the curriculum row, the formation gate
    is a statement about a finished 8000-iteration screen, and a per-stage verdict would invite
    exactly the reading ("stage c3 FAILED") the campaign must not make.
    """
    sname = str(stage["stage"])
    row = _gate_row(rec.get("evals"))
    if sname in rp.FACING_GATE_PHASES:
        return [gate_watch("gate_facing", row)]
    names = [str(s["stage"]) for s in item.get("stages") or []]
    if names and sname == names[-1]:
        return [gate_watch("gate_1c", row, track_ratio=rec.get("track_ratio_tail"))]
    return []


#: What a stage report prints per category KIND (beside ``n`` and ``surv``), as ``(key, label)``.
CATEGORY_REPORT_KEYS = {
    "stand": (("drift_speed", "drift"),),
    "fwd": (("ratio", "ratio"), ("tripod", "tripod")),
    "side": (("ratio", "ratio"), ("tripod", "tripod")),
    "back": (("ratio", "ratio"), ("tripod", "tripod")),
    "diag": (("ratio", "ratio"), ("tripod", "tripod")),
    "turn": (("facing_closed", "closed"), ("yaw_progress", "progress")),
    "move_turn": (("facing_closed", "closed"), ("yaw_progress", "progress"), ("ratio", "ratio")),
}


def _watch_reason(note) -> str:
    """``watch (reduced demand)`` -> ``reduced demand``; ``untrained`` stays; None -> ``watch``."""
    text = str(note or "watch")
    m = re.fullmatch(r"watch \((.+)\)", text)
    return m.group(1) if m else text


def category_mark(c: str, tc: dict | None) -> str:
    """``decisive`` / ``WATCH: <reason>`` for one category of a stage (its trained categories)."""
    tc = tc or {}
    if c in (tc.get("decisive") or ()):
        return "decisive"
    if c in (tc.get("present") or ()):
        return f"WATCH: {_watch_reason((tc.get('watch') or {}).get(c))}"
    return "WATCH: untrained"


def category_lines(row: str, summary: dict, tc: dict | None) -> list[str]:
    """One report line per category of a suite row (plan B3's rendering):
    ``- suite fwd: n 100 surv 0.930 ratio 0.620 tripod 0.510 [decisive]`` -- the hands of a paired
    category in parentheses, every category printed (only the decisive ones decide anything)."""
    cat = (summary or {}).get("cat") or {}
    out = []
    for c in _ordered_categories(cat):
        entry = cat.get(c) or {}
        keys = CATEGORY_REPORT_KEYS.get(c, ())
        text = (f"- {row} {c}: n {entry.get('n')} surv {fmt_num(entry.get('survival'))}"
                + "".join(f" {label} {fmt_num(entry.get(key))}" for key, label in keys))
        hands = entry.get("by_hand") or {}
        if hands:
            text += " (" + "; ".join(
                f"{h}: surv {fmt_num((hv or {}).get('survival'))}"
                + "".join(f" {label} {fmt_num((hv or {}).get(key))}" for key, label in keys)
                for h, hv in hands.items()) + ")"
        out.append(f"{text} [{category_mark(c, tc)}]")
    return out


def _stage_report(ctx: Ctx, item: dict, xid: str, stage: dict, rec: dict, spec) -> str:
    head = (f"### {xid}:{stage['stage']} — {item['label']} ({item['kind']}) — {now_iso()}\n\n"
            f"- recipe `{(item.get('recipe') or {}).get('recipe_id')}` on chain "
            f"`{(item.get('recipe') or {}).get('chain')}`, base `{(item.get('base_recipe') or {}).get('recipe_id')}`, "
            f"code_sha `{ctx.code[:12]}`\n"
            f"- overlay `{rec.get('overlay')}`\n")
    lines = rp.phase_lines(rec, spec)
    header = all_headers(ctx.state).get(xid) or {}
    tc = (header.get("trained_categories_by_stage") or {}).get(str(stage["stage"]))
    # `rp.phase_lines` renders the three LEGACY rows only (slow / light / ring). The plan's reuse
    # list says the ladder adds "the driver's own curriculum-row lines", and a curriculum stage's
    # whole eval is that row -- without this its per-hold numbers are in `state.json` and in no
    # human-readable record. A category-suite row (plan B3) is one line per category instead, marked
    # decisive / WATCH from the stage's own trained categories.
    for row, summary in sorted((rec.get("evals") or {}).items()):
        if row in ("slow", "light", "ring") or not isinstance(summary, dict):
            continue
        if summary.get("cat"):
            if tc:
                watch = [f"{c} ({_watch_reason(n)})" for c, n in (tc.get("watch") or {}).items()]
                lines.append(f"- {row} trained categories: decisive "
                             f"{', '.join(tc.get('decisive') or []) or 'none'}"
                             + (f"; watch {', '.join(watch)}" if watch else ""))
            lines.extend(category_lines(row, summary, tc))
            continue
        surv = summary.get("survival_by_hold") or {}
        ratio = summary.get("ratio_by_hold") or {}
        lines.append(f"- {row}: completion {fmt_num(summary.get('completion'))}, tripod_dir "
                     f"{fmt_num(summary.get('tripod_walking_dir'))}; survival "
                     + (", ".join(f"{h} {fmt_num(v)}" for h, v in surv.items()) or "n/a")
                     + "; ratio " + (", ".join(f"{h} {fmt_num(v)}" for h, v in ratio.items()) or "n/a"))
    for line in _stage_gate_watch(item, stage, rec):
        lines.append(f"- {line}")
    for name, ok in rec.get("overlay_checks") or []:
        lines.append(f"- overlay {'PASS' if ok else 'FAIL'}: {name}")
    for line in rec.get("retention") or []:
        lines.append(f"- {line}")
    if rec.get("floor"):
        lines.append(f"- floor {'PASS' if rec['floor']['passed'] else 'FAIL'}"
                     + (" (PARK: park_categories only -- park and flag)" if rec["floor"].get("park") else "")
                     + f": {rec['floor']['why']}")
    return head + "\n".join(lines) + f"\n\n>>> ENTRY {xid}:{stage['stage']}\n"


# ================================================================== seed terms (D00b / D00c)
def seed_terms_from_pair(s3: dict, s2: dict, resolution: dict, rule: dict, *, decisive=None) -> dict:
    """Decision 6's seed terms from ONE control pair (the same recipe at seeds 3 and 2).

    Per metric ``R_seed,i = |s3 - s2|``. Per block
    ``R_F^seed = max(|z_F(s2 submitted as the candidate against s3)|, rms_i(delta_i^seed / R_eval,i) / sqrt(k))``:
    the signed pair statistic alone can cancel to near zero when the seed deltas alternate in sign,
    while the rms term is the independence-model spread of a k-metric block mean and cannot cancel.
    The max of the two is the honest floor.

    Under rule v2 (plan D5) ``rule`` arrives EXPANDED at ``decisive(B0)`` -- the union of both unit
    systems (``eval_categories.expand_rule_union``), so every block and guard either system scores
    has a term; a raw v2 rule is expanded here at ``decisive`` (required then). The shared
    ``composite_guards`` are applied to shallow copies of BOTH screens before any guard is read, so
    ``cat_tripod_min`` gets its ``R_seed`` like every other guard.
    """
    if is_rule_v2(rule):
        ec = eval_categories()
        if not rule.get("expanded"):
            if decisive is None:
                raise ValueError("seed_terms_from_pair: a raw rule_version 2 rule needs decisive "
                                 "(decisive(B0)) to be expanded at")
            rule = ec.expand_rule_union(rule, decisive, decisive)
        dec = rule.get("decisive") or decisive or ()
        s3, s2 = ec.with_composites(s3, dec), ec.with_composites(s2, dec)
    out = {"metrics": {}, "blocks": {}, "k": {}}
    for block, spec in (rule.get("blocks") or {}).items():
        mids = spec.get("metrics") or []
        rows = metric_rows(mids, s2, s3, resolution, {})
        for row in rows:
            if row["delta"] is not None:
                out["metrics"][row["metric"]] = abs(row["delta"])
        stat = block_stat(rows)
        out["k"][block] = stat["k"]
        if not stat["k"]:
            # No member of this block was measurable on the pair. The block term is UNMEASURED, and
            # writing 0.0 would make the record say "measured, and it is zero" -- which reads as a
            # seed spread of exactly nothing. Left absent, `seed_term` returns None and
            # `block_resolution` falls back to `R_F^eval` alone, which is the honest answer.
            continue
        k = stat["k"]
        rms = math.sqrt(sum((r["z"] or 0.0) ** 2 for r in rows if r["z"] is not None) / k)
        out["blocks"][block] = max(abs(stat["z"] or 0.0), rms / math.sqrt(k))
    for mid in (rule.get("guards") or {}).get("metrics") or []:
        a, b = _num(dget(s2, mid)), _num(dget(s3, mid))
        if a is not None and b is not None:
            out["metrics"][mid] = abs(a - b)
    return out


def _entry_eval_sha(entry: dict) -> str:
    return str(entry.get("eval_sha") or (entry.get("evidence") or {}).get("eval_sha") or "")


def current_resolution_entry(ctx: Ctx, stem: str, *, need_blocks: bool = False) -> dict | None:
    """The NEWEST live ``stem`` entry (``D00bs``, ``D00c-<chain>``, or one of their ``@<eval_sha>``
    successors) measured at the CURRENT ``eval_sha``; with ``need_blocks``, one whose seed terms
    carry at least one block term (an empty-terms D00c is treated as absent, plan B3)."""
    found = None
    for e in entries(ctx.ledger):
        ident = str(e.get("id") or "")
        if ident != stem and not ident.startswith(stem + "@"):
            continue
        if e.get("status") == "squashed" or _entry_eval_sha(e) != str(ctx.eval_sha or ""):
            continue
        if need_blocks and not ((e.get("evidence") or {}).get("seed_terms") or {}).get("blocks"):
            continue
        found = e
    return found


def seed_terms_for(ctx: Ctx, *chains: str, lineage: str | None = None) -> tuple[dict, list[str]]:
    """The seed terms a decision cites, and the ids for its DECISION line.

    ``D00b`` (B0's pair) is cited by every decision on the baseline chain AND by every decision on
    any other chain until that chain's own ``D00c-<chain>`` exists. A pooled decision whose candidate
    and base sit on different chains takes the element-wise MAX of the two -- the wider of two
    measured seed spreads, never the narrower.

    Rule v2 (``lineage="v2"``, the default when the campaign's rule is v2; plan B3): a chain's
    ``D00c-<chain>`` counts only when it carries the CURRENT ``eval_sha`` and non-empty block terms,
    else ``D00bs`` at the current ``eval_sha`` -- and NEVER ``D00b``, whose terms are the v1
    lineage's slow-row spread (the milestones' light/ring rows are its one sanctioned citation, which
    pass ``lineage="v1"``). No D00bs raises :class:`ResolutionMissing` (the campaign halts
    ``halt:resolution_missing``) rather than judging at eval-only R.
    """
    v2 = (lineage == "v2") if lineage else is_rule_v2(ctx.cands.get("decision_rule"))
    ids, tables = [], []
    for chain in dict.fromkeys([c for c in chains if c]):
        if v2:
            entry = current_resolution_entry(ctx, f"D00c-{chain}", need_blocks=True) \
                or current_resolution_entry(ctx, "D00bs")
            if entry is None:
                raise ResolutionMissing(f"no D00bs at eval_sha {str(ctx.eval_sha)[:12] or '(none)'} "
                                        f"(and no current D00c-{chain}): a v2-scored comparison "
                                        "never cites D00b's v1 seed terms")
        else:
            entry = entry_by_id(ctx.ledger, f"D00c-{chain}")
            if entry is None:
                entry = entry_by_id(ctx.ledger, "D00b")
        if entry is None:
            continue
        ids.append(entry["id"])
        tables.append((entry.get("evidence") or {}).get("seed_terms") or {})
    if not tables:
        return {}, []
    merged = {"metrics": {}, "blocks": {}}
    for table in tables:
        for group in ("metrics", "blocks"):
            for key, value in (table.get(group) or {}).items():
                v = _num(value)
                if v is None:
                    continue
                merged[group][key] = max(merged[group].get(key, 0.0), v)
    return merged, ids


# ================================================================== resolution bootstrap (step 0)
#: ``D01`` -- decision 8's turn-rate evidence, recorded once so every cap decision cites a measurement
#: rather than an argument. No GPU probe: these are the numbers already on record.
D01_EVIDENCE = {
    "yaw_smoke_3_training_rad_s": 0.086,
    "yaw_smoke_3_eval_rad_s": [0.063, 0.094],
    "per_head_yaw_progress_at_5000_it": [0.09, 0.10],
    "kinematic_bound_rad_s": [0.15, 0.18],
    "shipped_ang_vel_max": 0.3,
    "note": ("the commanded cap is ~2x the plant's estimated turn ceiling, and the move+turn slot is "
             "infeasible at any commandable speed (walk floor 0.21 m/s => omega <= 0.01 rad/s at "
             "r_foot 1.2 m) -- so move+turn is scored relatively and every cap candidate is screened "
             "directly instead of probed"),
}


def ensure_resolution(ctx: Ctx) -> dict:
    """``resolution.json`` -- ingested if present, computed if absent -- plus ``D00`` and ``D01``.

    The computation lives in ``resolution.py`` and is called by import-by-path so the driver owns no
    statistics: it ingests a record. The decision rule is frozen into ``D00`` at the same moment, so
    a later edit of ``candidates.yaml``'s rule is visible in the ledger rather than only in the yaml.
    """
    path = ctx.dir / "resolution.json"
    if not path.exists():
        ctx.state["current"] = {"tag": "D00", "phase": "resolution", "iters": 0, "kind": "bootstrap",
                               "log": str(ctx.dir / "orchestrator.log")}
        ctx.save()
        tool = HERE / "resolution.py"
        if not tool.exists():
            raise SystemExit(f"{path} is absent and {tool} does not exist: the campaign cannot "
                             "standardise anything. Run the resolution step first.")
        res = resolution_tool()
        argv = ["--campaign-dir", str(ctx.dir), "--candidates", str(ctx.cands["__path__"])]
        ctx.c.log(f"resolution.json absent — computing it: resolution.py {' '.join(argv)}")
        try:
            rc_code = res.main(argv)
        except TypeError:
            rc_code = res.main()
        if rc_code not in (0, None) or not path.exists():
            raise SystemExit(f"resolution.py returned {rc_code} and {path} does not exist")
    resolution = json.loads(path.read_text())
    ctx.resolution = resolution
    # The record says which candidates.yaml it was measured against. After
    # --accept-candidates-change the yaml's decision_rule (block membership, levels,
    # resolution_inputs) may have moved while resolution.json did not -- D00 would then freeze the
    # NEW rule beside half-widths measured for the OLD one. Loud, not fatal: re-running
    # resolution.py is the fix and the orchestrator has to choose when.
    want = sha256_file(ctx.cands["__path__"])
    got = resolution.get("candidates_sha256")
    if got and got != want:
        ctx.c.log(f"WARNING: {path.name} was measured against candidates.yaml sha256 {got[:16]} "
                  f"but this launch reads {want[:16]} — delete it and relaunch to re-measure D00 "
                  "if the decision_rule or resolution_inputs moved")
        resolution.setdefault("notes", []).append(
            f"STALE: measured against candidates.yaml {got[:16]}, the campaign now reads {want[:16]}")
    if entry_by_id(ctx.ledger, "D00") is None:
        entry = {"id": "D00", "kind": "resolution", "candidate": None,
                 "question": "what is the eval resolution every decision is standardised by?",
                 "options": ["RECORD"], "chosen": "RECORD", "status": "final",
                 "evidence": {"resolution": {k: v for k, v in resolution.items() if k != "inputs"},
                              "inputs": resolution.get("inputs"),
                              "decision_rule": ctx.cands.get("decision_rule")},
                 "code_sha": ctx.code, "eval_sha": ctx.eval_sha,
                 "tools_sha": ctx.tools, "decided_at": now_iso(),
                 "decided_by": "resolution.py", "history": []}
        ctx.ledger.setdefault("entries", []).append(entry)
        ctx.write_ledger()
        ctx.lad.setdefault("resolution_ids", []).append("D00")
        ctx.notify("RESOLUTION", "D00", resolution_notify_fields(ctx, resolution, "D00"))
    _ensure_d01(ctx)
    ctx.recompute()
    ctx.save()
    return resolution


def resolution_label(resolution: dict | None) -> str:
    """The id a decision cites for the loaded resolution record (plan B3): ``D00s`` for a rule-v2
    record (it carries ``rule_version: 2``), ``D00`` for the pre-suite one."""
    res = resolution or {}
    if res.get("id"):
        return str(res["id"])
    return "D00s" if rule_version(res) == 2 or res.get("decisive_at") is not None else "D00"


def _campaign_halt(ctx: Ctx, why: str, detail: str) -> None:
    """Park the CAMPAIGN, never a header: ``phase = halt:<why>`` + a HALT line, then exit.

    The migration and resolution halts (``halt:resolution_missing``, ``halt:resolution D00s|D00bs``,
    ``halt:migration``, plan D9) are statements about the campaign's records, not about any
    experiment -- so no header's ``status`` is rewritten and a relaunch re-derives the same state.
    """
    ctx.state["phase"] = f"halt:{why}"
    ctx.save()
    ctx.notify("HALT", why, [detail, "fix the records or tools and relaunch -- no experiment header "
                             "was touched", f"log: {ctx.dir / 'orchestrator.log'}"])
    ctx.c.report_block(f"## HALT — {why} — {now_iso()}\n\n- {detail}\n")
    raise SystemExit(1)


def require_resolution(ctx: Ctx, what: str) -> None:
    """Halt ``halt:resolution_missing`` when a v2 decision has no loaded resolution (plan D9 item 3:
    ``_dispatch_candidate``, ``_dispatch_pooled``, ``milestone_i``, ``_write_d00bs``)."""
    if ctx.resolution is None or (is_rule_v2(ctx.cands.get("decision_rule")) and not ctx.resolution):
        _campaign_halt(ctx, "resolution_missing",
                       f"{what}: no suite resolution (D00s) is loaded at eval_sha "
                       f"{str(ctx.eval_sha)[:12] or '(none)'} -- a v2 decision is never taken on the "
                       "old record")


def decisive_b0(ctx: Ctx) -> list[str]:
    """``decisive(B0)`` (plan D5): the loaded D00s' ``decisive_at``, else ``trained_categories`` of the
    baseline chain's final stage (its composed env on the chain alone)."""
    dat = (ctx.resolution or {}).get("decisive_at")
    if dat:
        return _ordered_categories(dat)
    chain = (ctx.cands.get("baseline") or {}).get("chain")
    stages = chain_stages(ctx.cands, chain) if chain else []
    return _ordered_categories(_stage_decisive(stages[-1])) if stages else []


def _scoring_union(ctx: Ctx, resolution: dict | None = None) -> dict:
    """The rule a v2 RESOLUTION line and a seed-term fit enumerate: the union of both unit systems at
    ``decisive(B0)`` (the record's ``decisive_at``). A v1 rule is returned as it stands."""
    rule = ctx.cands.get("decision_rule") or {}
    if not is_rule_v2(rule):
        return rule
    dat = (resolution or {}).get("decisive_at") or decisive_b0(ctx) or list(eval_categories().CATEGORIES)
    return eval_categories().expand_rule_union(rule, dat, dat)


def baseline_trained(ctx: Ctx) -> dict | None:
    """``trained_categories`` of the baseline chain's final stage (its composed env on the chain
    alone) -- B0's sets and turn annotation for the RESOLUTION line; None when underivable."""
    chain = (ctx.cands.get("baseline") or {}).get("chain")
    stages = chain_stages(ctx.cands, chain) if chain else []
    if not stages:
        return None
    try:
        return eval_categories().trained_categories(
            _effective_stage_cfg(ctx.cands, stages[-1], stages[-1].get("overlay"))).to_dict()
    except (ValueError, KeyError):
        return None


def d00s_fields(ctx: Ctx, resolution: dict) -> list[str]:
    """The D00s-specific fields of a RESOLUTION line (plan D9 item 3 / D10): what it supersedes and at
    which eval, ``decisive_at``, the null band against its nominal, the acceptance, D00's copied
    light/ring tables and B0's turn-magnitude / encoding annotation (D4: never a demotion)."""
    acc = resolution.get("acceptance") or {}
    null = ((resolution.get("diagnostics") or {}).get("null_calibrated") or {}).get("all_tie")
    nominal = _num(resolution.get("null_nominal"))
    if acc.get("ok"):
        verdict = "ok"
    elif acc.get("accepted"):
        verdict = (f"ACCEPTED with failures by --accept-resolution D00s: {acc['accepted'].get('why')} "
                   f"({'; '.join(acc.get('failures') or [])})")
    else:
        verdict = "FAILED: " + "; ".join(acc.get("failures") or [])
    out = [f"on {resolution.get('screen_row')} at eval_sha {str(resolution.get('eval_sha') or '')[:12]}, "
           f"supersedes {resolution.get('supersedes')}, decisive_at "
           f"{', '.join(resolution.get('decisive_at') or []) or 'none'}",
           f"null all-TIE {fmt_num(null)} vs nominal {fmt_num(nominal)} (band [nominal - 0.03, 0.98], "
           f"env-clustered, survival guards floored); acceptance {verdict}"]
    if resolution.get("by_row_rule"):
        out.append("light/ring by_row copied from D00 with its frozen decision_rule as by_row_rule "
                   "(milestones score those rows on the v1 rule, with D00b's seed terms)")
    note = turn_note(baseline_trained(ctx))
    if note:
        out.append(f"B0 {note}")
    return out


def resolution_notify_fields(ctx: Ctx, resolution: dict, ident: str, seed_terms: dict | None = None,
                             seed_ids=None) -> list[str]:
    """The RESOLUTION line: the effective MDE per block and per guard, with the levels cited.

    Blocks and guards are those of the rule actually scored (plan D5): a v1 rule's own, a v2 rule's
    union of both unit systems at ``decisive_at`` -- F, F_mt, T and the ``_abs`` twins, the seven
    survival guards, both move_turn guards and ``cat_tripod_min``.
    """
    raw_rule = ctx.cands.get("decision_rule") or {}
    rule = _scoring_union(ctx, resolution)
    boot = (raw_rule.get("bootstrap") or {})
    fields = [f"B={resolution.get('B', boot.get('B'))} paired on "
              f"{boot.get('paired_on', 'env_index')}, rng {resolution.get('rng_seed', boot.get('rng_seed'))}"]
    if str(ident).startswith("D00s") and rule_version(resolution) == 2:
        fields.extend(d00s_fields(ctx, resolution))
    # resolution.py publishes its own line, whose per-member MDE (`R_F x R_i`, in each member's own
    # units) is the quantity Verification 5 asks the RESOLUTION line to name. Relay it as its own
    # FIELDS rather than as one string: `ledger_cli._field` rewrites an embedded " | " to " / ".
    if resolution.get("resolution_line"):
        fields.extend(str(resolution["resolution_line"]).split(" | ")[1:])
    blocks = []
    for block in _rule_block_names(rule):
        r_eval = _num(resolution_block(resolution, block).get("R_eval"))
        r_seed = seed_term(seed_terms or {}, block, block=True)
        eff = None if r_eval is None else max(r_eval, r_seed or 0.0)
        level = ((rule.get("blocks") or {}).get(block) or {}).get("level")
        blocks.append(f"{block} R {fmt_num(eff)} (R_eval {fmt_num(r_eval)}, R_seed {fmt_num(r_seed)}) @{level}")
    # `R` and not `MDE`: this number is the threshold on the STANDARDISED block mean z_F, which is
    # unitless. The per-member MDE is `R x R_i` and is in the relayed line above -- calling both
    # "MDE" on one line was two different quantities under one name.
    fields.append("blocks (threshold on z_F; per-member MDE = R x R_i): " + "; ".join(blocks))
    guards = []
    degenerate = []
    for mid in (rule.get("guards") or {}).get("metrics") or []:
        r_eval = _num(resolution_metric(resolution, guard_resolution_key(resolution, mid)).get("R_eval"))
        r_seed = seed_term(seed_terms or {}, mid)
        eff = None if r_eval is None and r_seed is None else max(r_eval or 0.0, r_seed or 0.0)
        # BOTH halves, not only the max: decision 6's mitigation for a degenerate cross-head pair
        # is `max(R_eval,i, R_seed,i)`, and it only bites if the seed term is the wider one. A
        # reader of this line has to be able to see which of the two is carrying the guard.
        guards.append(f"{mid} +/-{fmt_num(eff)} (R_eval {fmt_num(r_eval)}, R_seed {fmt_num(r_seed)})")
        if seed_terms is not None and r_eval is not None and (r_seed or 0.0) <= r_eval:
            degenerate.append(mid)
    # A guard IS compared in its own units, so its effective R is a genuine MDE.
    fields.append(f"guards @{(rule.get('guards') or {}).get('level')} one-sided, MDE in the guard's "
                  f"own units: " + "; ".join(guards))
    if seed_terms and degenerate:
        # Read this at window 2. A cross-head pair whose two arms sit at the SAME value on a guard
        # resolves it below the perfectly-correlated floor of the same-run spread, so a null
        # candidate trips it far more often than the declared 1 %. `max(R_eval,i, R_seed,i)` is the
        # stated mitigation and it only bites when the seed term is the wider of the two.
        fields.append(f"guards still carried by the EVAL half-width (the seed term does not widen "
                      f"them): {degenerate} -- a degenerate cross-head pair stays under-resolved "
                      f"here; treat a lone WORSE on one of them as uncertain")
    floors = (raw_rule.get("charter_floors") or {})
    fields.append("sideways CHARTER floors (cited until the bootstrap supersedes them): "
                  + ", ".join(f"{k} +/-{v}" for k, v in floors.items()))
    pf = {k: v for k, v in (resolution.get("process_floor") or {}).items() if v is not None}
    if not pf:
        floor_line = "absent (the orchestrator runs that 4-min eval later)"
    else:
        # Compact on purpose: this is a push notification, and the floor is 26 numbers. The worst
        # one and where it sits against the candidate R is the whole story; the record has the rest.
        worst = max(pf, key=lambda k: float(pf[k]))
        rel = [f"{k} {fmt_num(v)}" for k, v in sorted(pf.items(), key=lambda kv: -float(kv[1]))[:3]]
        pair = ((resolution.get("process_floor_detail") or {}).get(worst) or {}).get("pair", "?")
        floor_line = (f"{len(pf)} metrics from {pair}, widest {', '.join(rel)} "
                      f"(0.0 everywhere means the two evals were episode-identical -- the harness is "
                      f"deterministic at a fixed env_seed)" if float(pf[worst]) == 0.0 else
                      f"{len(pf)} metrics from {pair}, widest {', '.join(rel)}")
    fields.append("process floor (same-head 2b re-eval): " + floor_line)
    inputs = resolution.get("inputs") or []
    floor_rows = [i for i in inputs if i.get("process_floor_pair")]
    accepted = [i for i in inputs if i.get("accepted") and i not in floor_rows]
    other = [i for i in inputs if not i.get("accepted")]
    info = [i for i in other if str(i.get("reason", "")).startswith("informational")]
    refused = [i for i in other if i not in info]
    reasons = list(dict.fromkeys(str(i.get("reason")).split(" (")[0] for i in other))
    fields.append(f"inputs: {len(accepted)} scored, {len(floor_rows)} process-floor, "
                  f"{len(info)} informational, {len(refused)} refused"
                  + (f" ({'; '.join(reasons)})" if reasons else ""))
    if seed_ids:
        fields.append(f"seed terms from {', '.join(seed_ids)}")
    fields.append(f"metrics: {len((resolution.get('metrics') or {}))} scored")
    return fields


# ================================================================== notification lines
#: A metric family's short name in a DECISION line, and the short hold names beside it. The plan's
#: own line shape is "closed l90 0.55->0.72, r90 0.90->0.91, mt 0.20->0.24; progress l90 ...": the
#: FAMILY is named once per group, the holds are abbreviated, and the two families of the F block
#: are separated by "; ". Rendering the hold ALONE printed "turn_l90 0.500->0.720" twice in one
#: field -- once for the closed fraction and once for the yaw rate -- and the reader could not tell
#: which number was which.
BLOCK_FAMILIES = {"facing_closed_by_hold": "closed", "yaw_progress_by_hold": "progress",
                  "net_yaw_by_hold": "net_yaw", "ratio_by_hold": "ratio",
                  "achieved_speed_by_hold": "speed", "survival_by_hold": "surv",
                  "tripod_by_hold": "tripod"}
#: A HOLD's short name -- and a CATEGORY's (plan B3: the per-category ids ``cat.<c>.<key>`` render
#: as ``<family> <category>``; ``move_turn`` is both a legacy hold and a category, ``mt`` either way;
#: every other category keeps its own name, and a hand renders ``side.l``).
HOLD_SHORT = {"turn_l90": "l90", "turn_r90": "r90", "move_turn": "mt", "move_turn_r": "mt_r"}
#: A ``cat.<c>.<key>`` id's metric FAMILY in a notify line (the block rows group by it).
CAT_KEY_SHORT = {"facing_closed": "closed", "yaw_progress": "progress", "net_yaw": "net_yaw",
                 "ratio": "ratio", "achieved_speed": "speed", "survival": "surv", "tripod": "tripod",
                 "settle_s": "settle", "heading_final": "heading", "drift_speed": "drift",
                 "direction_error": "dir_err", "cross_track_ratio": "xtrack", "n": "n", "n_env": "n_env",
                 # motion quality (block Q, 2026-09-23); lower is better except six_down
                 "action_jitter": "jitter", "action_fft_freq": "fft_hz", "joint_speed_mean": "jspeed",
                 "rod_speed_mean": "rodspeed", "contact_switch_hz": "switch_hz", "foot_slip": "slip",
                 "wz_active_frac": "wz_frac", "clock_restarts": "restarts",
                 "swings_per_m": "swings/m", "slip_per_m": "slip/m", "work_per_m": "work/m",
                 "cam_work_per_m": "camwork/m", "swings_per_rad": "swings/rad",
                 "slip_per_rad": "slip/rad", "work_per_rad": "work/rad",
                 "cam_work_per_rad": "camwork/rad", "six_down_frac": "six_down",
                 "settle_six_down_s": "settle6", "foot_disp_end": "disp_end", "work_per_s": "work/s",
                 # rigidity (block RIG, clock-deprecation 2026-09-23); the stepping counts are
                 # lower-is-better, support_feet / swing_apex higher
                 "steps_per_m": "steps/m", "steps_per_rad": "steps/rad", "support_feet": "support",
                 "steps_per_s": "steps/s", "taps_per_m": "taps/m", "taps_per_rad": "taps/rad",
                 "swing_apex": "apex", "step_rate_hz": "step_hz", "gait_hz": "gait_hz",
                 "clock_hz": "clock_hz", "cadence_ratio": "cadence", "steps_per_rad_pre": "steps/rad_pre",
                 "steps_per_rad_post": "steps/rad_post", "hz_speed_slope": "hz_slope",
                 "offset_R": "offset_R"}

#: The WATCH set's own labels, in the plan's spelling. Every one of these ids used to render as its
#: LAST dotted part, which collides three times over ("move_turn" for the ratio and the tripod,
#: "settle" for both holds, "move_turn_r" for the ratio and the closed fraction). A ``cat.`` id is
#: labelled ``<category> <key>`` (``mt tripod``, ``side.l surv``) by :func:`_watch_label`.
WATCH_LABELS = {
    "track_ratio_tail": "track", "track_ratio_move_tail": "track_move",
    "ratio_by_hold.move_turn": "move_turn dir_ratio", "tripod_by_hold.move_turn": "move_turn tripod",
    "facing_by_hold.turn_l90.settle": "settle l90", "facing_by_hold.turn_r90.settle": "settle r90",
    "ratio_by_hold.lat_r": "lat_r ratio", "ratio_by_hold.move_turn_r": "move_turn_r ratio",
    "facing_closed_by_hold.move_turn_r": "closed mt_r",
    "survival_by_hold.turn_l90": "surv l90", "survival_by_hold.turn_r90": "surv r90",
    "cat_tripod_min": "tripod_min",
}


def _cat_parts(mid: str) -> tuple[str, str] | None:
    """``(family, category[.hand])`` of a ``cat.<c>[.by_hand.<h>].<key>`` id, else None."""
    if not str(mid).startswith("cat."):
        return None
    parsed = eval_categories().parse_cat_id(mid)
    if parsed is None:
        return None
    c, hand, key = parsed
    return CAT_KEY_SHORT.get(key, key), HOLD_SHORT.get(c, c) + (f".{hand}" if hand else "")


def _metric_parts(mid: str) -> tuple[str, str | None]:
    """``(family, hold)`` for a block/guard metric id; ``hold`` is ``None`` for a bare one. A
    per-category id is ``(family, category)`` -- ``cat.turn.facing_closed`` -> ``(closed, turn)``."""
    cat = _cat_parts(mid)
    if cat is not None:
        return cat
    parts = str(mid).split(".")
    if len(parts) == 1:
        return BLOCK_FAMILIES.get(parts[0], parts[0]), None
    return BLOCK_FAMILIES.get(parts[0], parts[0]), HOLD_SHORT.get(parts[1], parts[1])


def _watch_label(mid: str) -> str:
    """The WATCH row's name. Unknown ids fall back to ``family hold``, which is still unique."""
    if mid in WATCH_LABELS:
        return WATCH_LABELS[mid]
    cat = _cat_parts(mid)
    if cat is not None:
        return f"{cat[1]} {cat[0]}"
    family, hold = _metric_parts(mid)
    return f"{family} {hold}" if hold else family


def _block_rows_field(rows) -> str:
    """A block's rows grouped by metric FAMILY: ``closed l90 a->b, r90 ...; progress l90 ...``.

    A block with ONE family (T's ``ratio_by_hold``, T_abs's ``achieved_speed_by_hold``) drops the
    family name -- the block label already says which it is, and the plan's T line is bare holds.
    """
    groups: list[list] = []
    for row in rows or []:
        family, hold = _metric_parts(row["metric"])
        text = f"{hold or family} {fmt_num(row['b'])}->{fmt_num(row['a'])}"
        if groups and groups[-1][0] == family:
            groups[-1][1].append(text)
        else:
            groups.append([family, [text]])
    named = len(groups) > 1
    return "; ".join((f"{family} " if named else "") + ", ".join(texts)
                     for family, texts in groups)


def _block_field(label: str, block: dict, rows_fmt) -> str:
    if block.get("class") == "N/A":
        name = block.get("block")
        shown = label if not name or name == label else f"{label}={name}"
        return f"{shown} N/A (no decisive member: never TIE, never MISSING)"
    z, r = block.get("z"), block.get("R")
    parts = [rows_fmt(block.get("rows") or [])]
    # The block NAME, not just the F/T slot: an absolute-units candidate's "F" is F_abs and its
    # numbers are rad/s and m/s, not fractions and ratios. The notify line is what the user reads
    # before confirming, so it says which.
    name = block.get("block")
    shown = label if not name or name == label else f"{label}={name}"
    missing = " [NO RESOLUTION: this block can never be WORSE]" if block.get("resolution_missing") else ""
    return (f"{shown} z={('+' if (z or 0) >= 0 else '')}{fmt_num(z)} (R {fmt_num(r)}) "
            f"{block.get('class')}{missing}: " + ", ".join(p for p in parts if p))


def _row_fmt(row: dict) -> str:
    """One GUARD row. The four guard ids carry no dots, so they render under their own names."""
    family, hold = _metric_parts(row["metric"])
    return f"{family}{'' if hold is None else ' ' + hold} {fmt_num(row['b'])}->{fmt_num(row['a'])}"


def trained_field(cmpres: dict) -> str:
    """The DECISION line's ``trained:`` field of a v2 comparison (plan B3).

    ``trained: stand, fwd, ... (decisive); watch: move_turn (reduced demand); untrained: none`` --
    a decisive category the BASE never trained is marked ``(base untrained)``, and the base's own
    untrained set follows when it differs from none.
    """
    notes = cmpres.get("category_notes") or {}
    dec = [f"{c} (base untrained)" if notes.get(c) == "base untrained" else c
           for c in cmpres.get("decisive") or []]
    watch = [f"{c} ({_watch_reason(notes.get(c))})" for c in cmpres.get("present") or []
             if c not in (cmpres.get("decisive") or [])]
    untrained = cmpres.get("untrained") or {}
    text = "trained: " + (", ".join(dec) or "none") + " (decisive)"
    if watch:
        text += "; watch: " + ", ".join(watch)
    text += "; untrained: " + (", ".join(untrained.get("candidate") or []) or "none")
    if untrained.get("base"):
        text += f" (base untrained: {', '.join(untrained['base'])})"
    if cmpres.get("na_blocks"):
        text += f"; N/A blocks: {', '.join(cmpres['na_blocks'])}"
    return text


def _watch_value(mid: str, v: dict) -> str:
    note = str((v or {}).get("note") or "")
    if note.startswith("N/A") and v.get("a") is None and v.get("b") is None:
        return f"{_watch_label(mid)} {note}"
    return f"{_watch_label(mid)} {fmt_num(v.get('b'))}->{fmt_num(v.get('a'))}"


def watch_fields(watch: dict) -> list[str]:
    """The WATCH rows of a v2 comparison, grouped by WHY they are watched (plan B3):
    ``WATCH:`` (the decisive categories' own watch keys and hands), ``WATCH <c> (<reason>):`` per
    watch category, ``untrained WATCH:`` (a category the candidate never trained -- decided on
    nothing, printed for information) and ``WATCH legacy:`` (the pooled suite keys, WATCH only)."""
    plain, legacy = [], []
    by_cat: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    untrained: dict[str, list[str]] = {}
    for mid, v in (watch or {}).items():
        note = (v or {}).get("note")
        parsed = eval_categories().parse_cat_id(mid) if str(mid).startswith("cat.") else None
        if parsed is not None and note == "untrained":
            untrained.setdefault(parsed[0], []).append(_watch_value(mid, v))
        elif parsed is not None and str(note or "").startswith("watch"):
            by_cat.setdefault(parsed[0], []).append(_watch_value(mid, v))
            reasons[parsed[0]] = _watch_reason(note)
        elif parsed is None and note is not None:
            legacy.append(_watch_value(mid, v))
        else:
            plain.append(_watch_value(mid, v))
    out = ["WATCH: " + (", ".join(plain) or "none")]
    for c in _ordered_categories(by_cat):
        out.append(f"WATCH {c} ({reasons[c]}): " + ", ".join(by_cat[c]))
    if untrained:
        out.append("untrained WATCH (decided on nothing): " + "; ".join(
            f"{c}: " + ", ".join(untrained[c]) for c in _ordered_categories(untrained)))
    if legacy:
        out.append("WATCH legacy: " + ", ".join(legacy))
    return out


def turn_note(tc: dict | None) -> str | None:
    """The trained categories' turn-magnitude / encoding annotation (``turn: demand 1.571 rad > FACING
    band 0.600; encoding raw clips at 0.6``), or None."""
    for note in (tc or {}).get("notes") or []:
        if str(note).startswith("turn: demand"):
            return str(note)
    return None


def decision_notify_fields(ctx: Ctx, entry: dict, item: dict, decision: dict, *,
                           pooled: bool = False) -> list[str]:
    """The DECISION line: what was decided, on what evidence, and the two commands that change it."""
    ev = decision.get("evidence") or {}
    cmp_key = "pooled" if pooled else "seed3"
    cmpres = ev.get(cmp_key) or {}
    cand = candidate_by_id(ctx.cands, item.get("candidate")) or {}
    overlay = {}
    for stage in (item.get("recipe") or {}).get("overlay_by_stage", {}).values():
        overlay.update({k: v for k, v in stage.items() if k not in POOL_KEYS})
    keys = list((cand.get("overlay") or {}).keys()) if isinstance(cand.get("overlay"), dict) else []
    shown = {k: overlay.get(k) for k in keys if k in overlay} or \
            ({"pool_edit": canonical(cand.get("pool_edit"))} if cand.get("pool_edit") else
             {"chain": (item.get("recipe") or {}).get("chain")})
    fields = [f"{entry.get('experiment') or ''} {item.get('candidate')}-{cand.get('kind', '')} "
              f"{{{', '.join(f'{k}={v}' for k, v in shown.items())}}} on recipe "
              f"{(item.get('base_recipe') or {}).get('recipe_id')} "
              # The pooled item IS the seed-2 work item, so `item['seed']` is already the CONFIRM
              # seed: rendering it verbatim printed "s2 + s2" for a pair whose halves are the
              # screen seed and the confirm seed.
              + (f"s{ctx.seed_screen} + s{ctx.seed_confirm}" if pooled else f"s{item.get('seed')}"),
              f"chosen {entry.get('chosen')} ({entry.get('pooled_confidence') if pooled else entry.get('confidence')}"
              f", {entry.get('status')}) alt {entry.get('alternative')}"]
    own = all_headers(ctx.state).get(entry.get("experiment") or "") or {}
    scr = own.get("screen") or {}
    v2 = "decisive" in cmpres
    if v2:
        # Plan B3: what was decisive, what was only watched (with its reason) and what the candidate
        # never trained -- the informational half of the request, on the line the user reads.
        fields.append(trained_field(cmpres))
    for label, block in (cmpres.get("blocks") or {}).items():
        if block:
            fields.append(_block_field(label, block, _block_rows_field))
    guards = cmpres.get("guards") or {}
    grows = guards.get("rows") or []
    worst = "WORSE" if guards.get("worse") else "TIE"
    fields.append(f"guards {worst}: " + ", ".join(_row_fmt(r) + f" [{r.get('class')}]" for r in grows))
    if guards.get("missing"):
        fields.append(f"MISSING guards: {guards['missing']} (a pair cannot supply them -- no pair queued)")
    unresolved = [f"{k} ({v.get('block')})" for k, v in (cmpres.get("blocks") or {}).items()
                  if v.get("resolution_missing")]
    if unresolved:
        fields.append(f"NO RESOLUTION for block {unresolved}: no half-width and no seed term, so it "
                      "can never read WORSE -- the record is incomplete, not the evidence")
    drift = ev.get("anchored_drift") or {}
    fields.append(f"anchored drift {drift.get('verdict')} ({drift.get('why')}) anchor "
                  f"{ev.get('anchor') or 'B0'}")
    req = ev.get("adopt_requires")
    if req:
        fields.append(f"adopt_requires {req.get('block')}: {req.get('class')} (an ADOPT needs "
                      f"{req.get('block')} BETTER; lower-is-better members are sign-flipped, so "
                      "a positive delta is always the better direction)")
    prior = entry.get("prior_user_choice")
    if prior:
        # The plan's SQUASH paragraph: "the re-run's DECISION line quotes the old frozen choice as
        # prior_user_choice", so a user who already answered this candidate once is reminded rather
        # than silently re-asked.
        fields.append(f"prior_user_choice: {prior.get('choice')} on {prior.get('decision')} "
                      f"({prior.get('why')}) -- re-confirm if it still stands")
    levels = ev.get("levels") or {}
    fields.append(f"resolution {ev.get('resolution_id')} @{(levels.get('blocks') or {}).get('F')}/"
                  f"{levels.get('guards')}")
    watch = cmpres.get("watch") or {}
    if v2:
        fields.extend(watch_fields(watch))
        # D4: the in-place turn's demand vs the trained FACING band and the observation encoding
        # -- an ANNOTATION, never a demotion (turn is decisive for every recipe that trains it).
        note = turn_note((scr.get("trained_categories") or {}))
        if note and "turn" in (cmpres.get("decisive") or []):
            fields.append(note)
    else:
        fields.append("WATCH: " + ", ".join(f"{_watch_label(k)} {fmt_num(v['b'])}->{fmt_num(v['a'])}"
                                           for k, v in watch.items()))
    # The written formation gate on this candidate's OWN screen -- WATCH context, never a verdict
    # (see :func:`gate_watch`). A screen the ladder is about to adopt on a relative comparison can
    # still be a head the phase pipeline would refuse to promote, and the DECISION line is where the
    # reader has to see that. The s3 screen is the one named: the s2 half of a pooled pair carries
    # its own line in its stage report.
    fields.append(gate_watch("gate_1c", scr, track_ratio=scr.get("track_ratio_tail")))
    if decision.get("why"):
        fields.append(f"why: {decision['why']}")
    rule = ctx.cands.get("decision_rule") or {}
    rig_rule = flags_enabled(rule)
    if rig_rule:
        fields.extend(rig_fields(ctx, entry, cmpres))
    nxt = next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view)
    fields.append(f"next: {nxt['label'] if nxt else 'done'}")
    did = entry.get("id")
    alt = entry.get("alternative")
    # the S3 rule (plan step 12): a speculative line's own confirm/overrule cancels its pair, so the
    # CLI refuses them without --cancel-pair. The quoted commands deliberately do NOT carry it (a
    # relayed command run as-is must hit that refusal, not silently make WIN unreachable); the flag
    # is named in prose only, after the primary action (the ack-flag release, or waiting).
    speculative = bool(rig_rule and rule.get("pair_on") and entry.get("status") == "speculative")
    if speculative:
        fl = entry.get("flags") or {}
        if fl.get("park") and not lc.flag_acked(entry):
            fields.append(f"PRIMARY ACTION: {ctx.cli(f'ack-flag {did} --why ...')} (releases the park; "
                          "the pair then runs)")
        else:
            fields.append("PRIMARY ACTION: none -- wait for the pooled line (the pair is queued)")
    reverts = lc.confirm_choice(entry)
    if reverts != lc.effective_choice(entry):
        # An `overruled(driver)` entry: the plan's table makes `confirm` the REVERT, and
        # `alternative` after a self-overrule already IS the driver's pick -- so without this line
        # both quoted commands leave the override standing and the user has no way back.
        fields.append(f"confirm REVERTS this {(entry.get('override') or {}).get('by')} override to "
                      f"{did}'s own pick {reverts} (and resurrects what it squashed); to freeze "
                      f"{lc.effective_choice(entry)} instead, overrule --choose "
                      f"{lc.effective_choice(entry)}")
    fields.append("confirm: " + ctx.cli(f"confirm {did}"))
    fields.append("overrule: " + ctx.cli(f"overrule {did} --choose {alt} --why \"...\""))
    if speculative:
        fields.append("speculative line: confirm/overrule are refused until the pooled line lands; "
                      "to answer now anyway add --cancel-pair; WIN becomes unreachable for this arm")
    return fields


def rig_fields(ctx: Ctx, entry: dict, cmpres: dict) -> list[str]:
    """A RIG rule's extra DECISION-line fields (clock-deprecation plan step 4): RIG's z split, the
    ``arm_class``, removal verification per run, the flag list with each member's MDE, and -- on a
    seed-3 line with a queued pair -- the S3 rule; on a parked line, its release verb."""
    rule = ctx.cands.get("decision_rule") or {}
    ev = entry.get("evidence") or {}
    out = [rig_split_text(rig_split((cmpres.get("blocks") or {}).get(RIG_BLOCK)))]
    cls, why = arm_class(entry, rule)
    out.append(f"arm_class {cls or 'pending'} ({why})")
    for label, run in (ev.get("removal") or {}).items():
        if isinstance(run, dict):
            out.append(f"removal verification {label}: {removal_text(run)}")
    fl = entry.get("flags") or {}
    items = fl.get("items") or []
    out.append(f"flags ({fl.get('level') or '-'} line): "
               + ("; ".join(flag_text(f) for f in items) if items else "none"))
    did = entry.get("id")
    if fl.get("pair_queued"):
        out.append("pair queued — wait for the pooled line; confirm cancels it"
                   + (f" (immediate park: {', '.join(fl.get('immediate') or [])})" if fl.get("park") else ""))
    if fl.get("park") and not lc.flag_acked(entry):
        out.append(f"PARKED at await_user:{did} flag -- release: {ctx.cli(f'ack-flag {did} --why ...')}")
    return out


# ================================================================== dispatch (per kind)
def _d00c_lookup(ctx: Ctx, item: dict, header: dict) -> None:
    """Write ``D00c-<chain>`` from the FIRST ok same-chain pair, BEFORE any decision that cites it.

    The baseline chain's seed spread is ``D00b``; every other chain borrows it until its own pair
    exists, and this is where that pair becomes a record. Written as a lookup step rather than fitted
    later, so no decision is ever standardised by a number derived after it.
    """
    if item["kind"] not in ("twin", "seed2", "batched_s2"):
        return
    chain = header.get("chain")
    baseline_chain = (ctx.cands.get("baseline") or {}).get("chain")
    if not chain or chain == baseline_chain:
        return
    raw_rule = ctx.cands.get("decision_rule") or {}
    v2 = is_rule_v2(raw_rule)
    stem = f"D00c-{chain}"
    extra: dict = {}
    if v2:
        # Plan B3: a v2 SEAM. The raw v2 rule has no `blocks`/`guards.metrics`, so passing it through
        # wrote EMPTY seed terms that `seed_terms_for` then preferred (F_mt judged at R_eval alone,
        # ~3.7x too tight). Expanded exactly as `_write_d00bs` does -- the union of both unit systems
        # at decisive(B0) -- with the composites applied to both screens (in seed_terms_from_pair),
        # stamped with eval_sha and decisive_at; the early return fires only for a D00c measured at
        # the CURRENT eval_sha, and a stale one is superseded by a new entry, never rewritten.
        if current_resolution_entry(ctx, stem) is not None:
            return
        if not ctx.resolution:
            return                      # no D00s: the pooled decision that would cite it halts first
        dat = decisive_b0(ctx)
        rule = eval_categories().expand_rule_union(raw_rule, dat, dat)
        stale = [e.get("id") for e in entries(ctx.ledger)
                 if str(e.get("id") or "") == stem or str(e.get("id") or "").startswith(stem + "@")]
        used = {e.get("id") for e in entries(ctx.ledger)}
        ident = stem if stem not in used else f"{stem}@{str(ctx.eval_sha)[:12] or 'none'}"
        extra = {"eval_sha": ctx.eval_sha, "decisive_at": dat,
                 **({"supersedes": stale[-1]} if stale else {})}
    else:
        if entry_by_id(ctx.ledger, stem) is not None:
            return
        rule, ident = raw_rule, stem
    s3 = ok_screen(ctx.state, ctx.code, header.get("recipe_id"), ctx.seed_screen, ctx.view)
    if not s3:
        return
    terms = seed_terms_from_pair(s3.get("screen") or {}, header.get("screen") or {},
                                ctx.resolution, rule)
    entry = {"id": ident, "kind": "resolution", "candidate": None,
             "question": f"what is the seed spread on chain {chain}?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"seed_terms": terms, "pair": {"s3": s3.get("xid"), "s2": header.get("xid")},
                          "chain": chain, "recipe_id": header.get("recipe_id"),
                          **({"decisive_at": extra["decisive_at"]} if extra else {})},
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "history": [], **extra}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.lad.setdefault("resolution_ids", []).append(entry["id"])
    ctx.recompute()
    ctx.notify("RESOLUTION", entry["id"],
               resolution_notify_fields(ctx, ctx.resolution, entry["id"], terms, [entry["id"]]))


def _anchor_recipe_id(ctx: Ctx, anchor: dict) -> str | None:
    """The RECIPE the anchor is -- what an "ok screen" of the anchor has to be a screen OF.

    The entry's own ``recipe_id_if_adopted`` (written by ``_dispatch_candidate`` and
    ``_dispatch_baseline`` alike), falling back to the header for a ledger written before that field
    existed.
    """
    entry = entry_by_id(ctx.ledger, anchor.get("decision"))
    rid = (entry or {}).get("recipe_id_if_adopted")
    if rid:
        return rid
    xid = (entry or {}).get("experiment") or anchor.get("experiment")
    return (all_headers(ctx.state).get(xid) or {}).get("recipe_id")


def _anchor_screens(ctx: Ctx, recipe: dict) -> tuple[dict | None, dict | None, str | None]:
    """The anchor's own s3 (and s2, when it has one) screens -- the drift comparison's other side.

    BOTH sides go through :func:`ok_screen` under the CURRENT ``code_sha`` (amendment 28). Reading
    the anchor's recorded header instead meant that after ``--accept-code-change`` every drift check
    compared a new-code candidate against an OLD-code screen -- the one comparison decision 9 forbids
    ("then 'ok screen' fails for every old screen ... and the anchor re-resolves under the new
    hash"), and the Vocabulary's anchor is "recomputed by ``compose_recipe`` under the current
    ``code_sha``" -- even when the ``B<k>`` rerun of that very recipe already existed under the new
    hash. It also returned a screen whose window the view had squashed.

    When the anchor's recipe has no ok s3 screen under this hash the BASELINE anchor stands in (the
    plan's ``else B0``, and what ``_baseline_anchor`` already does for the no-adoption case), and
    the label says so. If neither resolves the drift is ``n/a``, which ``anchored_drift`` reads as
    "no anchor screen (the baseline is the control)".
    """
    anchor = recipe.get("anchor")
    if not anchor:
        return None, None, None
    fallback = _baseline_anchor(ctx.ledger, ctx.cands, set(ctx.view["entries"]))
    tried = [anchor]
    if fallback and fallback.get("decision") != anchor.get("decision"):
        tried.append(fallback)
    for i, cur in enumerate(tried):
        rid = _anchor_recipe_id(ctx, cur)
        s3h = ok_screen(ctx.state, ctx.code, rid, ctx.seed_screen, ctx.view) if rid else None
        if not s3h:
            continue
        s2h = ok_screen(ctx.state, ctx.code, rid, ctx.seed_confirm, ctx.view)
        why = cur.get("why") if i == 0 else (
            f"baseline: {anchor.get('candidate')}'s recipe has no ok s3 screen under this code_sha")
        return s3h.get("screen"), (s2h or {}).get("screen"), \
            f"{cur.get('candidate')}/{cur.get('decision')} ({why})"
    return None, None, f"{anchor.get('candidate')}/{anchor.get('decision')} ({anchor.get('why')})"


def dispatch(ctx: Ctx, item: dict, header: dict) -> None:
    """Turn a finished experiment into the ledger entry (or record) its kind calls for."""
    kind = item["kind"]
    _d00c_lookup(ctx, item, header)
    if kind == "baseline":
        _dispatch_baseline(ctx, item, header)
    elif kind == "candidate":
        _dispatch_candidate(ctx, item, header)
    elif kind == "twin":
        ctx.c.changelog_row(f"{header['xid']} twin of recipe {header.get('recipe_id')} at seed "
                            f"{header.get('seed')} — the base's s2 screen (no decision)")
        ctx.notify("DECISION", header["xid"], [f"twin({item.get('candidate')}) recorded as the s2 screen "
                                               f"of recipe {header.get('recipe_id')}",
                                               "no decision: a twin is a control, not a candidate"])
    elif kind == "seed2":
        _dispatch_pooled(ctx, item, header)
    elif kind == "batched_s2":
        ctx.c.changelog_row(f"{header['xid']} {item['label']} — the milestone's batched s2 screen")
    elif kind == "milestone":
        _close_milestone_promo(ctx, item, header)
    elif kind == "milestone_s2":
        _close_final_s2(ctx, item, header)
    ctx.save()


def _dispatch_baseline(ctx: Ctx, item: dict, header: dict) -> None:
    entry = {"id": _next_did(ctx.ledger), "kind": "baseline", "candidate": item["candidate"],
             "experiment": header["xid"], "seed": str(item["seed"]),
             "question": f"the {item['label']} control screen",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final", "confidence": "clear",
             "base_recipe_id": header.get("base_recipe_id"), "recipe_id_if_adopted": header.get("recipe_id"),
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "evidence": {"screen": header.get("screen")}, "history": []}
    if flags_enabled(ctx.cands.get("decision_rule")) and (header.get("floor_watch")
                                                          or header.get("floor_park")):
        # NOTE(clock-deprecation 2026-09-23, plan step 4): a BASELINE screen that trips a floor parks
        # the driver (every arm is judged against it); `ack-flag <Did>` releases it.
        # NOTE(stride/posture 2026-09-26): a park-category floor (`header.floor_park`) parks it too.
        entry["flags"] = {"level": "screen", "park": True, "pair_queued": False,
                          "immediate": (["floor"] if header.get("floor_watch") else [])
                          + (["floor_park"] if header.get("floor_park") else []),
                          "items": [_flag("floor", f"baseline floor tripped: {w}")
                                    for w in header.get("floor_watch") or []]
                          + floor_park_flags(header.get("floor_park"), where="baseline")}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.recompute()
    if (entry.get("flags") or {}).get("park"):
        did = entry["id"]
        ctx.notify("DECISION", did,
                   [f"{item['label']} ({header['xid']}) tripped a FLOOR -- the driver parks here: every "
                    "arm is judged against this screen",
                    *[flag_text(f) for f in entry["flags"]["items"]],
                    f"release: {ctx.cli(f'ack-flag {did} --why ...')}"])
    is_first = item["candidate"] == (ctx.cands.get("baseline") or {}).get("id", "B0")
    if str(item["seed"]) == ctx.seed_screen and is_first and ctx.lad.get("d03") is None:
        _write_d03(ctx, header)
    if str(item["seed"]) == ctx.seed_confirm and is_first and entry_by_id(ctx.ledger, "D00b") is None \
            and not is_rule_v2(ctx.cands.get("decision_rule")):
        # Rule v1 only. Under rule v2 the seed terms are D00bs, written against D00s by the
        # migration's own item (plan D9 item 4) once both B0 screens exist at the current eval_sha
        # -- never a D00b, whose terms a v2-scored comparison must not cite.
        _write_d00b(ctx, header)
    ctx.c.changelog_row(f"{entry['id']} {item['label']} baseline screen recorded ({header['xid']})")


def _write_d03(ctx: Ctx, header: dict) -> None:
    """``D03``: the RELATIVE ``dead_walker`` floor, measured off B0's own its-800-1000 trace.

    Written ONCE, from the baseline's first seed-3 log, and never by a ``B<k>`` rerun -- the floor a
    candidate is aborted on must not drift with the control it is measured against.
    """
    rec = None
    for key in header.get("recs") or []:
        rec = ctx.c.recs.get(key) or rec
    vals = []
    if rec and rec.get("log") and Path(rec["log"]).exists():
        vals = rp.track_ratio_series(Path(rec["log"]))
    seg = vals[D03_AT_ITERS - D03_WINDOW:D03_AT_ITERS]
    mean = (sum(seg) / len(seg)) if seg else None
    floor = max(DEAD_WALKER_FLOOR_MIN, DEAD_WALKER_FLOOR_FRAC * mean) if mean is not None \
        else DEAD_WALKER_FLOOR_MIN
    ctx.lad["d03"] = float(floor)
    entry = {"id": _next_did(ctx.ledger), "kind": "measurement", "candidate": None,
             "question": "what track_ratio_mean does a forming walker reach by iteration 1000 here?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"its_800_1000_mean": mean, "floor": float(floor),
                          "rule": f"max({DEAD_WALKER_FLOOR_MIN}, {DEAD_WALKER_FLOOR_FRAC} x the "
                                  "baseline's own its-800-1000 mean)",
                          "experiment": header["xid"], "armed_on": "candidate kinds only, from_scratch, "
                                                                  f"iters > {D03_AT_ITERS}"},
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.recompute()
    ctx.notify("RESOLUTION", entry["id"],
               [f"dead_walker floor {floor:.3f} = max({DEAD_WALKER_FLOOR_MIN}, "
                f"{DEAD_WALKER_FLOOR_FRAC} x {fmt_num(mean)}) from {header['xid']}",
                "armed on candidate kinds only (baselines, twins, seed2 and milestones run their full window)"])


def _write_d00b(ctx: Ctx, header: dict) -> None:
    """``D00b``: the block- and metric-level SEED terms, from the baseline's own s3/s2 pair."""
    s3 = ok_screen(ctx.state, ctx.code, header.get("recipe_id"), ctx.seed_screen, ctx.view)
    rule = ctx.cands.get("decision_rule") or {}
    terms = seed_terms_from_pair((s3 or {}).get("screen") or {}, header.get("screen") or {},
                                ctx.resolution, rule)
    entry = {"id": "D00b", "kind": "resolution", "candidate": None,
             "question": "how much do two seeds of the SAME recipe differ?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"seed_terms": terms, "pair": {"s3": (s3 or {}).get("xid"), "s2": header["xid"]},
                          "chain": header.get("chain"), "recipe_id": header.get("recipe_id")},
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.lad.setdefault("resolution_ids", []).append("D00b")
    ctx.recompute()
    ctx.notify("RESOLUTION", "D00b", resolution_notify_fields(ctx, ctx.resolution, "D00b", terms, ["D00b"]))
    # The control pair submitted as a candidate must decide all-TIE by construction; a WORSE here
    # means the resolution is too tight and every later decision would be noise. Run it in BOTH
    # unit systems: with `{}` the comparison uses F/T, and only `absolute_units` reaches F_abs/T_abs
    # -- which C3a and C3b are scored in, so the property must be checked there too, before C3a is
    # ever screened.
    lines = []
    for spec in ({}, {"absolute_units": True}):
        # The label is the rule's OWN block set in that unit system (`block_names`), never a literal.
        label = "/".join(block_names(spec, rule))
        cmpres = _screen_compare(spec, rule, header.get("screen") or {},
                                 (s3 or {}).get("screen") or {}, ctx.resolution, terms)
        worse = _any_worse(cmpres)
        better = _better_blocks(cmpres)
        lines.append(f"control check ({label}): B0 s2 submitted against B0 s3 decides "
                     f"{'all-TIE' if not worse and not better else 'NOT all-TIE'} "
                     f"-- worse: {worse or 'none'}, better: {better or 'none'}")
    # Decision 6's other pinned property: `R_F(D00b) >= R_F(D00)`. The seed spread is a FLOOR under
    # the eval spread -- two seeds of one recipe differ by at least as much as two evals of one head
    # -- so a block whose seed term comes out NARROWER than its eval half-width says the pair is
    # unrepresentative (one lucky seed), and every later decision on that block leans on the eval
    # number alone. Not a refusal: it is a measurement, and the record has to say so.
    tight = []
    for block in _rule_block_names(rule):
        r_eval = _num(resolution_block(ctx.resolution, block).get("R_eval"))
        r_seed = seed_term(terms, block, block=True)
        if r_eval is not None and r_seed is not None and r_seed < r_eval:
            tight.append(f"{block}: R_F^seed {fmt_num(r_seed)} < R_F^eval {fmt_num(r_eval)}")
    lines.append("R_F(D00b) >= R_F(D00): " + ("holds on every block" if not tight else
                 "VIOLATED on " + "; ".join(tight) + " -- the seed pair is narrower than the eval "
                 "bootstrap, so treat its block term as a lower bound (decision 6)"))
    ctx.notify("RESOLUTION", "D00b",
               lines + ["a non-TIE here means the resolution is too tight for single-seed screens"])
    entry.setdefault("evidence", {})["r_f_floor"] = {"violations": tight}
    ctx.write_ledger()


def d00bs_evidence(s3_screen: dict, s2_screen: dict, resolution: dict, rule: dict, decisive_at) -> dict:
    """Everything ``D00bs`` records, and its acceptance -- a PURE function, computed before any write.

    Plan D9 item 4: the seed terms per block and guard from the two rescreened B0 screens, standardised
    through D00s's per-metric ``R_eval`` (so this cannot precede D00s), over the rule expanded at
    ``decisive(B0)`` -- the UNION of both unit systems, every block the expansion names (F, F_mt, T,
    F_abs, F_mt_abs, T_abs; never a literal tuple). Per unit system:

    * the CONSTRUCTION check: B0 s2 submitted against B0 s3 under the full R decides all-TIE (the
      pair's own delta is inside its own seed term, so a non-TIE is a bug) with empty
      ``block_missing`` / ``guards.missing`` (a value missing on a rescreened B0 screen);
    * the seed-dominance verdict per decisive block (``R_b^seed > R_b^eval`` -> a single-seed screen
      is judged at ``R_b^seed``) and the per-member MDE ``R_b x R_i``;
    * the informative statement: the same pair classed at D00s's EVAL-ONLY R per block;
    * ``R(D00bs) >= R(D00s)`` per block, a recorded note (not a gate).

    ``acceptance = {ok, failures}`` is the construction check and the completeness of both systems.
    """
    dat = _ordered_categories(decisive_at)
    ec = eval_categories()
    union = ec.expand_rule_union(rule, dat, dat)
    terms = seed_terms_from_pair(s3_screen or {}, s2_screen or {}, resolution, union)
    systems, failures, tight = {}, [], []
    for units, spec in (("ratio", {}), ("absolute", {"absolute_units": True})):
        full = _screen_compare(spec, rule, s2_screen or {}, s3_screen or {}, resolution, terms,
                               decisive=dat, present=dat)
        eval_only = _screen_compare(spec, rule, s2_screen or {}, s3_screen or {}, resolution, {},
                                    decisive=dat, present=dat)
        worse, better = _any_worse(full), _better_blocks(full)
        block_missing = sorted({m for v in full["blocks"].values() for m in v["missing"]})
        guard_missing = list(full["guards"]["missing"])
        blocks = {}
        for label, v in full["blocks"].items():
            if v.get("class") == "N/A":
                blocks[v["block"]] = {"class": "N/A"}
                continue
            r_eval, r_seed, r_eff = v.get("R_eval"), v.get("R_seed"), v.get("R")
            mde = {row["metric"]: (None if r_eff is None or row.get("R_eval") is None
                                   else r_eff * row["R_eval"]) for row in v.get("rows") or []}
            eo = eval_only["blocks"].get(label) or {}
            blocks[v["block"]] = {
                "R_eval": r_eval, "R_seed": r_seed, "R": r_eff,
                "seed_dominated": bool(r_eval is not None and (r_seed or 0.0) > r_eval),
                "member_mde": mde, "class_full_R": v.get("class"),
                "eval_only": {"z": eo.get("z"), "R_eval": eo.get("R"), "class": eo.get("class")}}
            if r_eval is not None and r_seed is not None and r_seed < r_eval:
                tight.append(f"{v['block']}: R^seed {fmt_num(r_seed)} < R^eval {fmt_num(r_eval)}")
        all_tie = not worse and not better
        systems[units] = {"blocks": blocks, "all_tie": all_tie, "worse": worse, "better": better,
                          "block_missing": block_missing, "guards_missing": guard_missing,
                          "guards": {r["metric"]: {"R": r.get("R"), "R_seed": r.get("R_seed"),
                                                   "R_binom": r.get("R_binom"), "class": r.get("class")}
                                     for r in full["guards"]["rows"]}}
        if not all_tie:
            failures.append(f"construction check ({units}): B0 s2 vs s3 under the full R is NOT "
                            f"all-TIE -- worse {worse or 'none'}, better {better or 'none'} (a bug: the "
                            "pair's own delta is inside its seed term)")
        if block_missing:
            failures.append(f"block members MISSING on a B0 screen ({units}): {block_missing}")
        if guard_missing:
            failures.append(f"guards MISSING on a B0 screen ({units}): {guard_missing}")
    failures = list(dict.fromkeys(failures))
    return {"seed_terms": terms, "decisive_at": dat, "systems": systems,
            "r_floor": {"violations": list(dict.fromkeys(tight)),
                        "note": "R(D00bs) >= R(D00s) per block -- recorded, not a gate"},
            "acceptance": {"ok": not failures, "failures": failures}}


def d00bs_lines(evidence: dict) -> list[str]:
    """The D00bs RESOLUTION line's own fields: seed dominance per decisive block (F and F_mt first),
    the per-member MDE, the eval-only classing, the construction check and the R-floor note."""
    lines = []
    for units in ("ratio", "absolute"):
        sysd = (evidence.get("systems") or {}).get(units) or {}
        dom, mde, eo = [], [], []
        for block, b in (sysd.get("blocks") or {}).items():
            if b.get("class") == "N/A":
                dom.append(f"{block} N/A")
                continue
            dom.append(f"{block} R^seed {fmt_num(b.get('R_seed'))} vs R^eval {fmt_num(b.get('R_eval'))} "
                       f"seed-dominated {'yes' if b.get('seed_dominated') else 'no'}")
            mde.append(f"{block}: " + ", ".join(f"{m} {fmt_num(v)}" for m, v in (b.get("member_mde")
                                                                            or {}).items()))
            e = b.get("eval_only") or {}
            eo.append(f"{block} {e.get('class')} (z {fmt_num(e.get('z'))}, "
                      f"R_eval {fmt_num(e.get('R_eval'))})")
        lines.append(f"seed dominance ({units}): " + "; ".join(dom)
                     + " -- a seed-dominated block judges a single-seed screen at R^seed")
        lines.append(f"per-member MDE = R_b x R_i ({units}): " + "; ".join(mde))
        lines.append(f"B0 s2 vs s3 at D00s's eval-only R ({units}): " + "; ".join(eo))
        lines.append(f"construction check ({units}, full R): "
                     + ("all-TIE" if sysd.get("all_tie") else "NOT all-TIE")
                     + (f", MISSING {sysd.get('block_missing') + sysd.get('guards_missing')}"
                        if (sysd.get("block_missing") or sysd.get("guards_missing")) else ""))
    tight = (evidence.get("r_floor") or {}).get("violations") or []
    lines.append("R(D00bs) >= R(D00s): " + ("holds on every block" if not tight else
                 "VIOLATED on " + "; ".join(tight) + " (recorded, not a gate)"))
    return lines


def _write_d00bs(ctx: Ctx, s3_header: dict, s2_header: dict, *,
                 accepted: dict | None = None) -> tuple[dict | None, dict]:
    """``D00bs``: the rule-v2 seed terms from the two rescreened B0 screens against ``D00s``.

    Generalises :func:`_write_d00b` (plan B3): id ``D00bs`` (``D00bs@<eval_sha[:12]>`` when that id
    is taken by an earlier eval), standardised through the loaded ``D00s``, over the expanded rule's
    blocks (:func:`d00bs_evidence`). The acceptance is computed BEFORE any write: a failure writes
    NOTHING and returns ``(None, evidence)`` -- the caller (the migration's D00bs item) writes the
    ``.rejected.json`` diagnostic and halts ``halt:resolution D00bs``, unless ``accepted`` (the
    ``lad.resolution_acceptances`` record of ``--accept-resolution D00bs WHY``) is passed, in which
    case the entry is written with ``acceptance.ok = false`` and the line cites the acceptance.
    Returns ``(entry, evidence)``.
    """
    require_resolution(ctx, "D00bs")
    rule = ctx.cands.get("decision_rule") or {}
    dat = decisive_b0(ctx)
    ev = d00bs_evidence(s3_header.get("screen") or {}, s2_header.get("screen") or {}, ctx.resolution,
                        rule, dat)
    if not ev["acceptance"]["ok"] and accepted is None:
        return None, ev
    used = {e.get("id") for e in entries(ctx.ledger)}
    ident = "D00bs" if "D00bs" not in used else f"D00bs@{str(ctx.eval_sha)[:12] or 'none'}"
    acceptance = dict(ev["acceptance"], **({"accepted": accepted} if accepted is not None else {}))
    entry = {"id": ident, "kind": "resolution", "candidate": None,
             "question": "how much do two seeds of the SAME recipe differ, on the per-category suite?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"seed_terms": ev["seed_terms"],
                          "pair": {"s3": s3_header.get("xid"), "s2": s2_header.get("xid")},
                          "chain": s3_header.get("chain"), "recipe_id": s3_header.get("recipe_id"),
                          "against": resolution_label(ctx.resolution), "decisive_at": dat,
                          "systems": ev["systems"], "r_floor": ev["r_floor"], "acceptance": acceptance},
             "eval_sha": ctx.eval_sha, "code_sha": ctx.code, "tools_sha": ctx.tools,
             "decided_at": now_iso(), "decided_by": "driver", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.lad.setdefault("resolution_ids", []).append(ident)
    ctx.recompute()
    ctx.notify("RESOLUTION", ident, resolution_notify_fields(ctx, ctx.resolution, ident, ev["seed_terms"],
                                                             [ident]))
    tail = ([f"ACCEPTED with failures by --accept-resolution D00bs: {accepted.get('why')} "
             f"({'; '.join(ev['acceptance']['failures'])})"] if accepted is not None
            and not ev["acceptance"]["ok"] else [])
    ctx.notify("RESOLUTION", ident, d00bs_lines(ev) + tail)
    return entry, ev


def _prior_user_choice(ledger: dict, cid: str) -> dict | None:
    """The FROZEN choice a user gave this candidate before its entry was squashed, if any.

    A re-screen after an override (or a ``decide run``) writes a brand-new entry; without this the
    line the user gets carries no sign that they had already answered, and the plan asks the re-run
    to quote it.
    """
    found = None
    for e in entries(ledger):
        if e.get("candidate") != cid or e.get("kind") != "candidate":
            continue
        if not e.get("frozen") or e.get("status") != "squashed":
            continue
        ov = e.get("override") or {}
        found = {"decision": e.get("id"), "choice": ov.get("choice") or e.get("chosen"),
                 "why": ov.get("why"), "at": ov.get("at")}
    return found


#: The clock/apex income a removed term may still log over the tail (Episode_Reward, per second): at
#: weight 0.001 it is ~3e-4/s (the clock-held X08:c6 logs 0.32/s and 0.14/s), so 0.005 is "at eps"
#: with ~15x headroom and is still ~30x below the held income.
CLOCK_INCOME_EPS = 0.005
#: How many of the train log's last logged values "the tail" is (the phase driver's ``tail_mean``).
REMOVAL_TAIL_N = 20
#: The train-log label of the per-reset hidden fraction (``CrabHexOmniParkourObservations.
#: CLOCK_HIDDEN_FRAC_KEY``, restated: that module imports Isaac).
CLOCK_HIDDEN_FRAC_KEY = "Metrics/clock_hidden_frac"


def _npz_clock_obs_zero(run_dir) -> tuple[bool, str]:
    """``clock_obs`` is identically 0 in every ``raw/episode_*.npz`` of an eval stamp (``--save-raw``)."""
    import numpy as np  # noqa: PLC0415 -- the harness's own dependency, only this check needs it
    raw = Path(str(run_dir)) / "raw"
    files = sorted(raw.glob("episode_*.npz")) if raw.is_dir() else []
    if not files:
        return False, f"no raw/episode_*.npz under {run_dir}"
    worst = 0.0
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                if "clock_obs" not in z.files:
                    return False, f"{f.name} carries no clock_obs"
                arr = np.asarray(z["clock_obs"], dtype=float)
        except Exception as exc:                                        # noqa: BLE001
            return False, f"{f.name} unreadable ({type(exc).__name__})"
        if arr.size:
            worst = max(worst, float(np.max(np.abs(arr))))
    return worst == 0.0, f"max |clock_obs| {worst:g} over {len(files)} episodes"


#: The hold labels whose steps carry the p2r stand bit (``KRABBY_OMNI_STAND_OBS=clock``): the suite's settle
#: (its ``run_meta.suite.settle_label``) and every ``stand`` hold.
STAND_BIT_HOLD_LABELS = ("stand", "settle")


def stand_bit_matches_labels(bit, seg, labels: list, stand_labels) -> tuple[int, int]:
    """``(mismatches, checked)`` of a per-step raw stand bit against the hold labels (pure; numpy arrays).

    ``seg`` is the episode's ``segment_id`` (``-1`` = prologue, not checked), ``labels`` the run's
    ``hold_labels``. A step whose bit differs from its own label is allowed when it equals a NEIGHBOUR's
    label (the one-step command-to-obs lag at a hold boundary; a prologue neighbour allows either value).
    """
    import numpy as np  # noqa: PLC0415
    bit = np.asarray(bit, dtype=float).reshape(-1)
    seg = np.asarray(seg).reshape(-1)[:bit.size]
    lab = np.full(bit.size, np.nan)
    for i, name in enumerate(labels):
        lab[seg == i] = 1.0 if name in stand_labels else 0.0
    bad = 0
    checked = 0
    for t in range(bit.size):
        if np.isnan(lab[t]):
            continue
        checked += 1
        if bit[t] == lab[t]:
            continue
        nbrs = [lab[u] for u in (t - 1, t + 1) if 0 <= u < bit.size]
        if any(np.isnan(x) or bit[t] == x for x in nbrs):
            continue
        bad += 1
    return bad, checked


def _npz_clock_obs_stand_bit(run_dir) -> tuple[bool, str]:
    """The NARROWED clock-removal check for chains with ``KRABBY_OMNI_STAND_OBS=clock`` (p2r, 2026-10-09):
    ``clock_obs[:, 1]`` (col 16) identically 0 and ``clock_obs[:, 0]`` (col 15, the stand bit) in {0, 1} and
    equal to the stand label of every recorded step (``STAND_BIT_HOLD_LABELS`` + the suite's settle label; the
    one-step lag at hold boundaries allowed; the prologue unchecked). A row it cannot read FAILS."""
    import numpy as np  # noqa: PLC0415
    raw = Path(str(run_dir)) / "raw"
    files = sorted(raw.glob("episode_*.npz")) if raw.is_dir() else []
    if not files:
        return False, f"no raw/episode_*.npz under {run_dir}"
    meta_path = Path(str(run_dir)) / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:                                            # noqa: BLE001
        return False, f"run_meta.json unreadable ({type(exc).__name__})"
    labels = list(meta.get("hold_labels") or [])
    stand_labels = set(STAND_BIT_HOLD_LABELS)
    settle = (meta.get("suite") or {}).get("settle_label") if isinstance(meta.get("suite"), dict) else None
    if settle:
        stand_labels.add(str(settle))
    worst16, bad_vals, bad_lab, checked = 0.0, 0, 0, 0
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                if "clock_obs" not in z.files or "segment_id" not in z.files:
                    return False, f"{f.name} carries no clock_obs / segment_id"
                arr = np.asarray(z["clock_obs"], dtype=float).reshape(-1, 2)
                seg = np.asarray(z["segment_id"])
        except Exception as exc:                                        # noqa: BLE001
            return False, f"{f.name} unreadable ({type(exc).__name__})"
        if arr.size:
            worst16 = max(worst16, float(np.max(np.abs(arr[:, 1]))))
        bad_vals += int(np.sum((arr[:, 0] != 0.0) & (arr[:, 0] != 1.0)))
        b, c = stand_bit_matches_labels(arr[:, 0], seg, labels, stand_labels)
        bad_lab += b
        checked += c
    ok = worst16 == 0.0 and bad_vals == 0 and bad_lab == 0 and checked > 0
    return ok, (f"max |col16| {worst16:g}, col15 not in {{0,1}} on {bad_vals} steps, "
                f"{bad_lab}/{checked} labelled steps off the stand label over {len(files)} episodes")


def _stand_bit_clock_stage(env: dict) -> bool:
    """The composed stage env sets ``KRABBY_OMNI_STAND_OBS=clock`` (a bad spelling reads as no)."""
    try:
        return _motion().parse_stand_obs(env.get("KRABBY_OMNI_STAND_OBS")) == "clock"
    except ValueError:
        return False


def removal_verification(ctx: Ctx, header: dict) -> dict:
    """The driver's own K-stage checks on ONE finished run (clock-deprecation plan step 4, "Removal
    verification"): ``{xid, applies, ok, checks: [{stage, check, ok, detail}]}``.

    For every stage whose overlay touches a clock knob and removes something (HIDE, or a clock term
    whose effective end weight is <= 0.001):

    * each removed term: a K stage's DUMPED static weight <= 0.001; a PHASEOUT stage's dumped
      ``phaseout_<term>`` block w1 <= 0.001 with t1 <= the stage end; either way the train log's
      ``Episode_Reward/<term>`` at eps (<= :data:`CLOCK_INCOME_EPS`) over the tail;
    * HIDE: ``Metrics/clock_hidden_frac`` 1.0 over the train log's tail, and every eval row of the
      stage shows ``clock_obs`` identically 0 (the stamp's raw npz) with the carry in ``run_meta``.

    ``applies`` False (and ok) for a run that removes nothing. A check that cannot be read FAILS: an
    unverifiable removal is not a verified one.
    """
    xid = header.get("xid")
    bucket = (ctx.state.get("keys") or {}).get(f"seed{header.get('seed')}") or {}
    checks: list[dict] = []

    def add(stage, check, ok, detail=""):
        checks.append({"stage": stage, "check": check, "ok": bool(ok), "detail": detail})

    for sname in header.get("stages") or []:
        sname = str(sname)
        overlay = (header.get("overlay_by_stage") or {}).get(sname) or {}
        stage = _stage_spec_of(ctx, header, sname)
        if stage is None or not _touches_clock(overlay):
            continue
        iters = int(stage.get("iters") or 0)
        stage_env = _effective_stage_cfg(ctx.cands, stage, overlay)
        st = clock_stage_state(stage_env, iters)
        stand_bit = _stand_bit_clock_stage(stage_env)
        removed = [t for t in CLOCK_TERM_KNOBS if st["end_w"][t] <= CLOCK_EPS_W + 1e-12]
        if not removed and not st["hide"]:
            continue
        rec = bucket.get(f"{xid}:{sname}") or ctx.c.recs.get(f"{xid}:{sname}") or {}
        dump_path = Path(rec["ckpt"]).parent / "params/env.yaml" if rec.get("ckpt") else None
        dump = dump_path.read_text(errors="ignore") if dump_path and dump_path.exists() else None
        log = Path(rec["log"]) if rec.get("log") else None
        log_ok = bool(log and log.exists())
        if not rec:
            add(sname, "stage record", False, "no stage record for this run")
            continue
        end = int(ph.STEPS_PER_ITER) * iters
        for term in removed:
            if term in st["phaseout"]:
                params = _yaml_block(_yaml_block(dump or "", "curriculum", indent=0), f"phaseout_{term}",
                                     indent=2)
                w1 = _num(_grep_scalar(params, "w1", indent=r"\s{6}"))
                t1 = _num(_grep_scalar(params, "t1", indent=r"\s{6}"))
                add(sname, f"{term}: dumped PHASEOUT w1 <= {CLOCK_EPS_W} with t1 <= {end}",
                    w1 is not None and t1 is not None and w1 <= CLOCK_EPS_W + 1e-12 and t1 <= end,
                    f"w1 {w1}, t1 {t1}" if dump else "no params/env.yaml")
            else:
                got = _num(_grep_scalar(_yaml_block(dump or "", term), "weight", indent=r"\s{4}"))
                add(sname, f"{term}: dumped weight <= {CLOCK_EPS_W}",
                    got is not None and got <= CLOCK_EPS_W + 1e-12,
                    f"weight {got}" if dump else "no params/env.yaml")
            vals = rp.series(log, f"Episode_Reward/{term}")[-REMOVAL_TAIL_N:] if log_ok else []
            tail = (sum(vals) / len(vals)) if vals else None
            add(sname, f"{term}: logged income at eps over the tail (<= {CLOCK_INCOME_EPS}/s)",
                tail is not None and abs(tail) <= CLOCK_INCOME_EPS,
                f"tail mean {fmt_num(tail, 5)} over {len(vals)}" if log_ok else "no train log")
        if st["hide"]:
            vals = rp.series(log, CLOCK_HIDDEN_FRAC_KEY)[-REMOVAL_TAIL_N:] if log_ok else []
            add(sname, "clock_hidden_frac is 1.0 over the tail",
                bool(vals) and min(vals) >= 1.0 - 1e-9,
                f"tail min {fmt_num(min(vals)) if vals else 'n/a'} over {len(vals)}" if log_ok else "no train log")
            for row, summary in sorted((rec.get("evals") or {}).items()):
                run_dir = (summary or {}).get("run_dir")
                if not run_dir:
                    add(sname, f"{row}: clock_obs == 0", False, "the row recorded no run dir")
                    continue
                if stand_bit:
                    # p2r: column 15 carries the stand bit by design (KRABBY_OMNI_STAND_OBS=clock)
                    ok, detail = _npz_clock_obs_stand_bit(run_dir)
                    add(sname, f"{row}: clock_obs col 16 == 0, col 15 == stand label", ok, detail)
                else:
                    ok, detail = _npz_clock_obs_zero(run_dir)
                    add(sname, f"{row}: clock_obs == 0", ok, detail)
                meta = Path(run_dir) / "run_meta.json"
                carried = ((json.loads(meta.read_text()) or {}).get("env_vars") or {}).get(
                    _clock().CLOCK_HIDE_VAR) if meta.exists() else None
                add(sname, f"{row}: run_meta carries {_clock().CLOCK_HIDE_VAR}=1", str(carried) == "1",
                    f"env_vars {carried!r}")
    applies = bool(checks)
    return {"xid": xid, "applies": applies, "ok": all(c["ok"] for c in checks), "checks": checks}


def removal_text(removal: dict | None) -> str:
    if not removal or not removal.get("applies"):
        return "removes nothing (no check applies)"
    bad = [f"{c['stage']}: {c['check']} ({c['detail']})" for c in removal.get("checks") or [] if not c["ok"]]
    n = len(removal.get("checks") or [])
    return (f"{removal.get('xid')} verified ({n} checks)" if not bad
            else f"{removal.get('xid')} FAILED {len(bad)}/{n}: " + "; ".join(bad))


def _flag_record(level: str, items: list, *, pair_queued: bool) -> dict:
    """``entry.flags``: the flag list of the line at ``level`` and whether it PARKS the driver -- a
    queued pair defers every flag to the pooled line except the two immediate ones (removal
    verification, the pilot stop), which park at the seed-3 line."""
    immediate = [f for f in items if f.get("flag") in IMMEDIATE_FLAGS]
    park = bool(immediate) if pair_queued else bool(items)
    return {"level": level, "items": items, "park": park, "pair_queued": bool(pair_queued),
            "immediate": [f.get("flag") for f in immediate]}


def _dispatch_candidate(ctx: Ctx, item: dict, header: dict) -> None:
    cand = candidate_by_id(ctx.cands, item["candidate"]) or {}
    rule = ctx.cands.get("decision_rule") or {}
    require_resolution(ctx, f"{header.get('xid')} {item['candidate']} decision")
    base = ok_screen(ctx.state, ctx.code, (item.get("base_recipe") or {}).get("recipe_id"),
                     ctx.seed_screen, ctx.view)
    if base is None:
        _halt(ctx, header, "control_missing", "the base recipe has no ok s3 screen at dispatch", item=item)
    try:
        terms, seed_ids = seed_terms_for(ctx, header.get("chain"), (base or {}).get("chain"))
    except ResolutionMissing as exc:
        _campaign_halt(ctx, "resolution_missing", f"{header.get('xid')} {item['candidate']}: {exc}")
    # The anchor is COMPOSED here, never read off the item: the Vocabulary makes it "recomputed by
    # `compose_recipe` under the current `code_sha`", and `_resolve_pair` already does exactly this.
    # `item_from_header` cannot carry one (a resumed window rebuilds its item from the header), so
    # reading it off the item made anchored drift read `n/a` on every RESUMED candidate -- turning a
    # drift-failing screen into a clear ADOPT with no pair, which is the one thing the precondition
    # exists to stop. `inclusive=True` is the same bound `_recipe_with` uses and the candidate's own
    # entry does not exist yet, so a fresh dispatch is unchanged.
    a3, a2, alabel = _anchor_screens(ctx, compose_recipe(ctx.ledger, ctx.cands, ctx.code,
                                                         upto=item["candidate"], inclusive=True,
                                                         squashed=set(ctx.view["entries"])))
    try:
        decision = decide(header.get("screen") or {}, (base or {}).get("screen") or {}, ctx.resolution,
                          cand_spec=cand, rule=rule, seed_terms=terms, anchor_screen=a3,
                          anchor_label=alabel,
                          resolution_id=", ".join([resolution_label(ctx.resolution)] + seed_ids))
    except ResolutionMissing as exc:
        _campaign_halt(ctx, "resolution_missing", f"{header.get('xid')} {item['candidate']}: {exc}")
    except ValueError as exc:
        # a v2 comparison of a screen that carries no trained categories (``trained_sets``) -- a
        # named halt, never a traceback that would crash-loop the relaunch
        _halt(ctx, header, "eval_mismatch", f"the decision cannot be scored: {exc}", item=item)
    entry = {"id": _next_did(ctx.ledger), "kind": "candidate", "candidate": item["candidate"],
             "experiment": header["xid"], "seed2_experiment": None,
             "question": cand.get("question"), "options": decision["options"],
             "chosen": decision["chosen"], "alternative": decision["alternative"],
             "confidence": decision["confidence"], "pooled_confidence": None,
             "status": decision["status"], "evidence": decision["evidence"],
             "depends_on": list(cand.get("depends_on") or []),
             "conditional_refs": conditional_refs(cand.get("conditional")),
             "base_recipe_id": (item.get("base_recipe") or {}).get("recipe_id"),
             "recipe_id_if_adopted": header.get("recipe_id"), "base_code_sha": header.get("code_sha"),
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools,
             "decided_at": now_iso(),
             "decided_by": "driver", "why": decision.get("why"), "history": [],
             "control_experiment": (base or {}).get("xid")}
    entry["evidence"]["replay"] = header.get("replay")
    entry["evidence"]["retention"] = _retention_lines(ctx, header)
    entry["evidence"]["floor_watch"] = header.get("floor_watch")
    entry["prior_user_choice"] = _prior_user_choice(ctx.ledger, item["candidate"])
    if flags_enabled(rule):
        # NOTE(clock-deprecation 2026-09-23, plan step 4): the seed-3 line's flags. With a pair
        # queued they are printed and deferred to the pooled line, except removal verification and
        # the pilot stop, which park HERE (before the pair and before the next arm).
        removal = removal_verification(ctx, header)
        entry["evidence"]["removal"] = {"s3": removal}
        cmp3 = decision["evidence"].get("seed3") or {}
        entry["evidence"]["rig_split"] = rig_split((cmp3.get("blocks") or {}).get(RIG_BLOCK))
        items = decision_flags(rule, cmp3, [header.get("screen") or {}], [(base or {}).get("screen") or {}],
                               removal=[removal], pilot=pilot_stop(rule, item["candidate"], decision))
        if header.get("floor_park"):
            # NOTE(stride/posture 2026-09-26): a park-category floor -- an immediate flag (parks here,
            # before any pair), recorded beside the verdict it does not change.
            entry["evidence"]["floor_park"] = list(header["floor_park"])
            items = items + floor_park_flags(header["floor_park"], where="seed-3 run")
        entry["flags"] = _flag_record("s3", items, pair_queued=decision["status"] == "speculative")
    if entry["chosen"] not in entry["options"]:
        raise ValueError(f"{entry['id']}: chosen {entry['chosen']} not in options {entry['options']}")
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    # RECONCILE, not `recompute()` alone (amendment 28). A candidate decision MOVES the recipe --
    # "an out-of-rank re-run whose decision changes the recipe squashes later screens and milestones
    # exactly like an overrule" -- and the plan has the derived view "recomputed after EVERY ledger
    # write (decision or override)". With only a recompute the VIEW was right and the RECORD was
    # not: headers stayed under `state.ladder.experiments` (so `_supersedes` on the re-run was `[]`),
    # dependent entries stayed `decided` with no SQUASH line, REPORT block or CHANGELOG row, the CLI
    # accepted a `confirm` of a view-squashed entry, and a re-run that landed the SAME choice left
    # its dependants stored `squashed(derived)` -- which `newest_entry` reads THROUGH, so `next_item`
    # answered with a 3.7-3.9 GPU-h re-screen of an intact window until the next launch healed it.
    note = _reconcile_after_decision(ctx, entry, item)
    fields = decision_notify_fields(ctx, entry, item, decision)
    if note:
        fields.append(f"this decision moved the recipe{note}")
    ctx.notify("DECISION", entry["id"], fields)
    ctx.c.changelog_row(f"{entry['id']} {item['candidate']} {entry['chosen']} "
                        f"({entry['confidence']}, {entry['status']}) — {decision.get('why')}")
    ctx.c.report_block(_decision_report(ctx, entry, item, decision))


def _reconcile_after_decision(ctx: Ctx, entry: dict, item: dict) -> str:
    """``squash_and_report`` for a decision write; ``""`` when the record did not move.

    The trigger names the decision the way an override's does, so the SQUASH line, the REPORT block
    and the CHANGELOG row all say which decision moved the recipe. Returns the note only when
    something actually moved, so an ordinary decision (the common case: nothing depends on it yet)
    adds no field to its own DECISION line and writes no SQUASH line.
    """
    note = squash_and_report(ctx, f"{entry['id']} {item.get('candidate')} -> "
                                  f"{lc.effective_choice(entry)}")
    return "" if note == NOTHING_MOVED else note


def _retention_lines(ctx: Ctx, header: dict) -> list[str]:
    """Every stage's RETENTION lines, hoisted onto the decision (WATCH evidence, never a verdict)."""
    out = []
    for key in header.get("recs") or []:
        out += list((ctx.c.recs.get(key) or {}).get("retention") or [])
    return out


def _dispatch_floor_reject(ctx: Ctx, item: dict, header: dict, *, rescore: dict | None = None) -> None:
    """A FLOOR trip on a CANDIDATE is a clear REJECT (decision 4): nothing usable was produced.

    ``rescore`` (plan D9 item 5, a rescore FAIL's step (2)): ``{eval_sha, supersedes, label}`` -- the
    entry records it and the DECISION line says the floor was re-decided eval-only and which entry
    it supersedes (squashed by cause (f) once step (3) lands)."""
    cand = candidate_by_id(ctx.cands, item["candidate"]) or {}
    entry = {"id": _next_did(ctx.ledger), "kind": "candidate", "candidate": item["candidate"],
             "experiment": header["xid"], "question": cand.get("question"),
             "options": list(CANDIDATE_OPTIONS), "chosen": "REJECT", "alternative": "ADOPT",
             "confidence": "clear", "status": "decided", "floor_stage": header.get("floor_stage"),
             "evidence": {"floor": header.get("floor"), "floor_stage": header.get("floor_stage"),
                          "retention": _retention_lines(ctx, header),
                          "replay": header.get("replay"),
                          **({"rescore": dict(rescore)} if rescore else {}),
                          "resolution_id": "D03" if "dead_walker" in str(header.get("floor")) else None},
             "depends_on": list(cand.get("depends_on") or []),
             "conditional_refs": conditional_refs(cand.get("conditional")),
             "base_recipe_id": header.get("base_recipe_id"),
             "recipe_id_if_adopted": header.get("recipe_id"), "code_sha": ctx.code, "eval_sha": ctx.eval_sha,
             "tools_sha": ctx.tools, "decided_at": now_iso(), "decided_by": "driver",
             "why": f"FLOOR: {header.get('floor')}", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    # Same reconcile as `_dispatch_candidate` (amendment 28): a FLOOR REJECT is still a decision that
    # can move the recipe -- it is what a re-run of a previously ADOPTED candidate lands on when the
    # re-run dies -- so the dependants it invalidates are archived here, not at the next launch.
    note = _reconcile_after_decision(ctx, entry, item)
    ctx.notify("DECISION", entry["id"],
               [f"{header['xid']} {item['candidate']} on recipe {header.get('base_recipe_id')} s{item['seed']}",
                f"chosen REJECT (clear, decided) alt ADOPT",
                f"FLOOR at stage {header.get('floor_stage')}: {header.get('floor')}",
                "no pair: a floor is not a resolution question",
                "re-run it: " + ctx.cli(f"decide {item['candidate']} --choose run --why \"...\"")]
               + ([f"{rescore.get('label')} (eval-only: nothing trained) under eval_sha "
                   f"{str(rescore.get('eval_sha'))[:12]}: supersedes {rescore.get('supersedes')}, whose "
                   "floor was measured under an older eval"] if rescore else [])
               + ([f"this decision moved the recipe{note}"] if note else []))
    ctx.c.changelog_row(f"{entry['id']} {item['candidate']} REJECT (floor at {header.get('floor_stage')})")


def _dispatch_pooled(ctx: Ctx, item: dict, header: dict) -> None:
    """The pooled decision on a finished seed-2 run: terminal, and possibly a driver self-overrule."""
    did = item.get("decision") or header.get("decision")
    entry = entry_by_id(ctx.ledger, did)
    if entry is None:
        raise KeyError(f"the seed-2 run {header['xid']} names decision {did}, which is not in the ledger")
    _resolve_pair(ctx, item, entry, header.get("screen") or {}, s2_xid=header["xid"],
                  floor_tripped=bool(header.get("floor_watch")), header=header)


def _recorded_screen(ctx: Ctx, xid: str | None) -> dict | None:
    """The header of an experiment the LEDGER names, whatever ``code_sha`` it ran under.

    ``ok_screen`` filters on the CURRENT hash, which is right for a control lookup and wrong for a
    pair already pending when a code change landed: decision 9 says such a pair "keeps its recorded
    s3 screens under the old ``code_sha``". The entry records both halves (``experiment`` and
    ``control_experiment``), so the pooled decision reads them from there.
    """
    if not xid:
        return None
    header = all_headers(ctx.state).get(xid) or {}
    if header.get("status") != "ok" or not header.get("screen"):
        return None
    if xid in ctx.view["experiments"]:
        return None                      # squashed: not a screen any more
    return header


def _resolve_pair(ctx: Ctx, item: dict, entry: dict, s2_screen: dict, *, s2_xid: str | None,
                  floor_tripped: bool = False, header: dict | None = None) -> None:
    if entry.get("frozen") or entry.get("status") in ("confirmed", "overruled", "squashed",
                                                      "skipped", "final"):
        # The user (or a milestone) settled this decision while the pair was in flight. Decision 7:
        # a frozen decision is never moved by a later pair. Normally the run is squashed on ingest
        # and never resumed; if one finishes anyway, its screen is a RECORD, not a transition -- and
        # certainly not the illegal `confirmed -> final` move that would kill the driver here.
        entry.setdefault("evidence", {})["late_pair"] = {
            "experiment": s2_xid, "status_when_it_landed": entry.get("status"),
            "why": "the decision was settled before this pair finished; recorded as WATCH"}
        entry["seed2_experiment"] = entry.get("seed2_experiment") or s2_xid
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("DECISION", entry["id"],
                   [f"the pair run {s2_xid} finished after {entry['id']} was already "
                    f"{entry.get('status')}", "WATCH: a frozen or resolved decision is never moved "
                    "by a later pair (decision 7); the screen is kept as a record"])
        return
    cid = entry.get("candidate")
    cand = candidate_by_id(ctx.cands, cid) or {}
    rule = ctx.cands.get("decision_rule") or {}
    require_resolution(ctx, f"{entry.get('id')} {cid} pooled decision")
    base = item.get("base_recipe") or compose_recipe(ctx.ledger, ctx.cands, ctx.code, upto=cid,
                                                    squashed=set(ctx.view["entries"]))
    # The two s3 screens are the ones the s3 DECISION was made against, so they are looked up by the
    # ids the entry recorded and only then by `ok_screen`. Decision 9: "a pair pending at the change
    # keeps its recorded s3 screens under the old `code_sha`, its s2 and any missing twin run under
    # the new hash". With `ok_screen` alone every pending pair died on the first relaunch after a
    # code landing -- and because its s2 header is `ok` with no `inflight`, `find_inflight` resumed
    # nothing and the next launch re-derived the same `pooled` item: an unrecoverable crash loop
    # after ~11 GPU-hours of B<k> + twin + s2.
    base_s3 = _recorded_screen(ctx, entry.get("control_experiment")) \
        or ok_screen(ctx.state, ctx.code, base["recipe_id"], ctx.seed_screen, ctx.view)
    base_s2 = ok_screen(ctx.state, ctx.code, base["recipe_id"], ctx.seed_confirm, ctx.view)
    cand_s3 = _recorded_screen(ctx, entry.get("experiment")) \
        or ok_screen(ctx.state, ctx.code, entry.get("recipe_id_if_adopted"), ctx.seed_screen, ctx.view)
    if not (base_s3 and base_s2 and cand_s3):
        missing = ", ".join(n for n, got in (("the base's s3", base_s3), ("the base's s2", base_s2),
                                             ("the candidate's s3", cand_s3)) if not got)
        detail = (f"{entry['id']}: the pooled decision needs the base's s3/s2 and the candidate's "
                  f"s3 -- missing {missing}")
        if header is not None:
            # A named halt, not a traceback: the user can read it, and the campaign parks instead of
            # crash-looping. `control_missing` is the halt `_dispatch_candidate` already uses for
            # the same condition.
            _halt(ctx, header, "control_missing", detail, item=item)
        raise RuntimeError(detail)
    a3, a2, alabel = _anchor_screens(ctx, compose_recipe(ctx.ledger, ctx.cands, ctx.code, upto=cid,
                                                        inclusive=True, squashed=set(ctx.view["entries"])))
    try:
        terms, seed_ids = seed_terms_for(ctx, (cand_s3 or {}).get("chain"), (base_s3 or {}).get("chain"))
        pooled = decide_pooled(cand_s3.get("screen") or {}, s2_screen, base_s3.get("screen") or {},
                               (base_s2 or {}).get("screen") or {}, ctx.resolution, cand_spec=cand,
                               rule=rule, prior=entry, seed_terms=terms, anchor_s3=a3, anchor_s2=a2,
                               anchor_label=alabel,
                               resolution_id=", ".join([resolution_label(ctx.resolution)] + seed_ids),
                               floor_tripped=floor_tripped)
    except ResolutionMissing as exc:
        _campaign_halt(ctx, "resolution_missing", f"{entry.get('id')} {cid} pooled decision: {exc}")
    except ValueError as exc:
        # a v2 pair whose candidate screen carries no trained categories: named, never a traceback
        if header is not None:
            _halt(ctx, header, "eval_mismatch", f"{entry.get('id')} pooled decision: {exc}", item=item)
        _campaign_halt(ctx, "eval_mismatch", f"{entry.get('id')} {cid} pooled decision: {exc}")
    target = "final" if not pooled["self_overrule"] else "overruled"
    check_transition(entry.get("status"), target, "driver")
    _history(entry, target, "driver", f"pooled: {pooled['why']}")
    before = lc.effective_choice(entry)
    entry["status"] = target
    entry["pooled_confidence"] = pooled["pooled_confidence"]
    entry["seed2_experiment"] = s2_xid or entry.get("seed2_experiment")
    entry.setdefault("evidence", {}).update(pooled["evidence"])
    entry["why_pooled"] = pooled["why"]
    if flags_enabled(rule):
        # NOTE(clock-deprecation 2026-09-23, plan step 4): the POOLED line's flags -- read on the
        # pair (both seeds), removal verification of both runs, and ANY flag parks the driver.
        removal = dict((entry["evidence"].get("removal") or {}))
        removal["s2"] = (removal_verification(ctx, header) if header is not None
                         else {"xid": s2_xid, "applies": True, "ok": False,
                               "checks": [{"stage": "-", "check": "the seed-2 header", "ok": False,
                                           "detail": "not found"}]})
        entry["evidence"]["removal"] = removal
        entry["evidence"]["s2_floor_watch"] = list((header or {}).get("floor_watch") or [])
        pcmp = pooled["evidence"].get("pooled") or {}
        entry["evidence"]["rig_split_pooled"] = rig_split((pcmp.get("blocks") or {}).get(RIG_BLOCK))
        items = decision_flags(rule, pcmp, [cand_s3.get("screen") or {}, s2_screen or {}],
                               [base_s3.get("screen") or {}, (base_s2 or {}).get("screen") or {}],
                               removal=[r for r in removal.values() if isinstance(r, dict)])
        s2_park = list((header or {}).get("floor_park") or [])
        if s2_park:
            # NOTE(stride/posture 2026-09-26): the seed-2 run's park-category floor (never a
            # floor_watch, so neither `floor_tripped` nor arm_class BREAK): a flag on the pooled line.
            entry["evidence"]["s2_floor_park"] = s2_park
            items = items + floor_park_flags(s2_park, where="seed-2 run")
        entry["flags"] = _flag_record("pooled", items, pair_queued=False)
    if pooled["self_overrule"]:
        entry["override"] = {"id": f"driver:{s2_xid or entry['id']}", "by": "driver",
                             "choice": pooled["chosen"], "why": pooled["why"], "at": now_iso(),
                             "applied_at": now_iso()}
    ctx.write_ledger()
    note = ""
    if pooled["self_overrule"]:
        note = squash_and_report(ctx, f"driver self-overrule of {entry['id']}: {before} -> "
                                     f"{pooled['chosen']}")
    else:
        ctx.recompute()
    ctx.save()
    tag = "SELF-OVERRULE" if pooled["self_overrule"] else "DECISION"
    fields = decision_notify_fields(ctx, entry, dict(item, recipe=item.get("recipe") or {},
                                                     base_recipe=base), pooled, pooled=True)
    if pooled["self_overrule"]:
        fields.insert(2, f"the s3 pick {before} is overruled by its own pair -> {pooled['chosen']}{note}")
    ctx.notify(tag, entry["id"], fields)
    ctx.c.changelog_row(f"{entry['id']} {cid} pooled -> {pooled['chosen']} "
                        f"({pooled['pooled_confidence']}, {target})")
    ctx.c.report_block(_decision_report(ctx, entry, item, pooled, pooled=True))


def _decision_report(ctx: Ctx, entry: dict, item: dict, decision: dict, *, pooled: bool = False) -> str:
    cmpres = (decision.get("evidence") or {}).get("pooled" if pooled else "seed3") or {}
    lines = [f"## {entry['id']} — {entry.get('candidate')} — {entry.get('chosen')} "
             f"({entry.get('pooled_confidence') if pooled else entry.get('confidence')}, "
             f"{entry.get('status')}) — {now_iso()}\n",
             f"- question: {entry.get('question')}",
             f"- experiment `{entry.get('experiment')}` recipe `{entry.get('recipe_id_if_adopted')}` "
             f"base `{entry.get('base_recipe_id')}` control `{entry.get('control_experiment')}`",
             f"- why: {decision.get('why')}"]
    for label, block in (cmpres.get("blocks") or {}).items():
        lines.append(f"- block {label} ({block.get('block')}): z {fmt_num(block.get('z'))} vs R "
                     f"{fmt_num(block.get('R'))} -> {block.get('class')} (k={block.get('k')}, "
                     f"missing {block.get('missing')})")
        for row in block.get("rows") or []:
            lines.append(f"  - {row['metric']}: {fmt_num(row['b'])} -> {fmt_num(row['a'])} "
                         f"({'lower is better, signed ' if row.get('lower_is_better') else ''}"
                         f"delta {fmt_num(row.get('delta'))}, R_eval {fmt_num(row.get('R_eval'))}, "
                         f"R_seed {fmt_num(row.get('R_seed'))}, z {fmt_num(row.get('z'))}"
                         f"{', MISSING ' + str(row.get('missing')) if row.get('missing') else ''})")
    for row in (cmpres.get("guards") or {}).get("rows") or []:
        lines.append(f"- guard {row['metric']}: {fmt_num(row['b'])} -> {fmt_num(row['a'])} "
                     f"(R {fmt_num(row.get('R'))}) -> {row.get('class')}")
    drift = (decision.get("evidence") or {}).get("anchored_drift") or {}
    lines.append(f"- anchored drift: {drift.get('verdict')} — {drift.get('why')}")
    lines.append(f"\n>>> ENTRY {entry['id']}\n")
    return "\n".join(lines)


# ================================================================== milestones
def _milestone_entry_new(ctx: Ctx, mid: str, recipe: dict, eligibility: int) -> dict:
    entry = {"id": _next_mid(ctx.ledger, mid), "kind": "milestone", "candidate": mid,
             "question": f"does the recipe adopted up to {mid} promote through 2a and 2b?",
             "options": list(MILESTONE_OPTIONS), "chosen": None, "alternative": None,
             "status": "open", "confidence": None, "pooled_confidence": None,
             "recipe_id": recipe.get("recipe_id"), "base_recipe_id": None,
             "eligibility": int(eligibility), "evidence": {}, "promoted_head": {},
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    return entry


def _previous_promoted(ctx: Ctx, mid: str) -> tuple[dict | None, str | None]:
    """The previous milestone's promoted heads, following ``carried_from`` to the last real promotion."""
    prev = previous_milestone(ctx.cands, mid)
    seen = set()
    while prev and prev not in seen:
        seen.add(prev)
        entry = newest_milestone_entry(ctx.ledger, prev, set(ctx.view["entries"]))
        if entry is None:
            return None, None
        if entry.get("promoted_head"):
            return entry["promoted_head"], entry["id"]
        prev = (entry.get("evidence") or {}).get("carried_from")
    return None, None


def milestone_i(ctx: Ctx, item: dict) -> None:
    """(i) the batched seed-2 confirmation -- a ZERO-GPU pooled decision, before any promotion.

    The pair is the effective recipe ``R_k`` (s3 and s2) against the previous milestone's recipe
    ``R_{k-1}`` (``B0`` at M1, whose s2 exists by construction). It runs only while ``decided``
    adoptions are pending since the last milestone: with nothing to confirm the pair is recorded as
    WATCH. Agreement promotes every ``decided`` adoption in the block to ``final``; a contradiction
    overrules the block's LAST unfrozen adoption (``override.by: milestone``) and the milestone
    re-runs ONCE on the reduced recipe.
    """
    mid = item["candidate"]
    recipe = item["recipe"]
    block = milestone_block_entries(ctx.cands, ctx.ledger, mid, set(ctx.view["entries"]))
    pending_adoptions = [e for e in block if lc.effective_choice(e) == "ADOPT"
                         and e.get("status") == "decided" and not e.get("frozen")]
    prev = previous_milestone(ctx.cands, mid)
    base = milestone_recipe(ctx.cands, ctx.ledger, prev, ctx.code, set(ctx.view["entries"])) if prev else \
        compose_recipe(ctx.ledger, ctx.cands, ctx.code,
                       upto=(candidate_ids(ctx.cands) or [None])[0], squashed=set(ctx.view["entries"]))
    cand_s3 = ok_screen(ctx.state, ctx.code, recipe["recipe_id"], ctx.seed_screen, ctx.view)
    cand_s2 = ok_screen(ctx.state, ctx.code, recipe["recipe_id"], ctx.seed_confirm, ctx.view)
    base_s3 = ok_screen(ctx.state, ctx.code, base["recipe_id"], ctx.seed_screen, ctx.view)
    base_s2 = ok_screen(ctx.state, ctx.code, base["recipe_id"], ctx.seed_confirm, ctx.view)
    eligibility = 1 + sum(1 for e in entries(ctx.ledger)
                          if e.get("kind") == "milestone" and e.get("candidate") == mid)
    entry = _milestone_entry_new(ctx, mid, recipe, eligibility)
    fields = [f"{mid} (i) batched seed-2 confirmation: R_k {recipe['recipe_id']} vs R_(k-1) "
              f"{base['recipe_id']}"]
    if not pending_adoptions or not (cand_s3 and cand_s2 and base_s3 and base_s2):
        why = ("no decided adoption is pending since the last milestone"
               if not pending_adoptions else
               f"a screen of the pair is missing (R_k s3 {bool(cand_s3)}/s2 {bool(cand_s2)}, "
               f"R_(k-1) s3 {bool(base_s3)}/s2 {bool(base_s2)})")
        entry["evidence"]["pair"] = f"WATCH: {why}"
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("MILESTONE", entry["id"], fields + [f"(i) WATCH -- {why}", "the promotion follows"])
        return
    rule = ctx.cands.get("decision_rule") or {}
    # Plan B3: milestone_i joins D9 item 3's resolution_missing halt list, and cites the loaded
    # resolution's own id (D00s under rule v2).
    require_resolution(ctx, f"{mid} (i)")
    try:
        terms, seed_ids = seed_terms_for(ctx, recipe.get("chain"), base.get("chain"))
        pooled = decide_pooled(cand_s3["screen"], cand_s2["screen"], base_s3["screen"],
                               base_s2["screen"], ctx.resolution,
                               cand_spec=_recipe_cand_spec(ctx, recipe), rule=rule,
                               prior={"chosen": "ADOPT", "status": "decided"}, seed_terms=terms,
                               resolution_id=", ".join([resolution_label(ctx.resolution)] + seed_ids),
                               floor_tripped=bool(cand_s2.get("floor_watch")))
    except ResolutionMissing as exc:
        _campaign_halt(ctx, "resolution_missing", f"{mid} (i): {exc}")
    except ValueError as exc:
        _campaign_halt(ctx, "eval_mismatch", f"{mid} (i): {exc}")
    entry["evidence"]["pair"] = pooled["evidence"]
    entry["evidence"]["stage"] = "i"
    entry["pooled_confidence"] = pooled["pooled_confidence"]
    if pooled["chosen"] == "ADOPT":
        promoted = []
        for e in pending_adoptions:
            check_transition(e.get("status"), "final", "driver")
            _history(e, "final", "driver", f"{mid} batched pair agrees")
            e["status"] = "final"
            e["pooled_confidence"] = pooled["pooled_confidence"]
            e.setdefault("evidence", {})["batched"] = {"milestone": mid, "why": pooled["why"]}
            promoted.append(e["id"])
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("MILESTONE", entry["id"],
                   fields + [f"(i) AGREE ({pooled['pooled_confidence']}): {pooled['why']}",
                             f"the block's decided adoptions are now final: {promoted}",
                             "the promotion follows"])
        return
    # a contradiction. `pending_adoptions` is already filtered to UNFROZEN entries, so the
    # "nothing unfrozen to drop" arm below could never fire here: an all-frozen block never reaches
    # this far (it is recorded WATCH above, with `pair: no decided adoption is pending`). Kept as
    # one name so the two FAIL arms read alike, and the notify text says the true reason.
    unfrozen = list(pending_adoptions)
    if unfrozen and not milestone_rerun_used(ctx.ledger, mid, ctx.cands, set(ctx.view["entries"])):
        target = unfrozen[-1]
        check_transition(target.get("status"), "overruled", "milestone")
        _history(target, "overruled", "milestone", f"{mid} batched pair contradicts")
        before = lc.effective_choice(target)
        target["status"] = "overruled"
        target["override"] = {"id": f"milestone:{mid}", "by": "milestone", "choice": "REJECT",
                              "why": f"{mid} batched pair contradicts the adoption", "at": now_iso(),
                              "applied_at": now_iso()}
        entry["chosen"] = None
        check_transition(entry.get("status") or "open", "squashed", "driver")
        entry["pre_squash_status"] = entry.get("status") or "open"
        entry["status"] = "squashed"
        entry["squashed_by"] = "driver"
        _history(entry, "squashed", "driver", f"{mid} (i) contradiction: re-runs once on the reduced recipe")
        ctx.write_ledger()
        note = squash_and_report(ctx, f"{mid} (i) contradiction: {target['id']} {before} -> REJECT")
        ctx.notify("MILESTONE", entry["id"],
                   fields + [f"(i) CONTRADICTION ({pooled['pooled_confidence']}): {pooled['why']}",
                             f"dropped the block's last unfrozen adoption {target['id']} "
                             f"({target.get('candidate')}) -> REJECT{note}",
                             "(i) re-runs ONCE on the reduced recipe"])
        return
    check_transition(entry.get("status") or "open", "await", "driver")
    _history(entry, "await", "driver", f"{mid} (i) FAIL")
    entry["chosen"] = "FAIL"
    entry["status"] = "await"
    entry["evidence"]["why"] = pooled["why"]
    ctx.write_ledger()
    ctx.recompute()
    ctx.state["phase"] = f"await_user:{mid}"
    ctx.save()
    ctx.notify("MILESTONE", entry["id"],
               fields + [f"(i) FAIL ({pooled['pooled_confidence']}): {pooled['why']}",
                         "the automatic rerun is spent",
                         "accept the recipe as it stands: " + ctx.cli(f"resume-milestone {mid} --accept"),
                         "or drop an adoption: " + ctx.cli(f"resume-milestone {mid} --drop <Cid>")])


def _recipe_cand_spec(ctx: Ctx, recipe: dict | None) -> dict:
    """The ``cand_spec`` a MILESTONE comparison is scored under.

    Only one field of a candidate spec matters to ``_screen_compare``: ``absolute_units``. Decision
    5 makes it a property of the CHANGE, and at a milestone the change is the whole recipe -- if any
    adopted candidate moved the command cap, clip or ellipse, then the two sides' eval rows ran
    under different denominators (those keys are ``eval_carry``) and the ratio blocks compare a
    number against a yardstick the recipe itself moved.
    """
    for cid in (recipe or {}).get("adopted") or []:
        if (candidate_by_id(ctx.cands, cid) or {}).get("absolute_units"):
            return {"absolute_units": True}
    return {}


def _row_worse(ctx: Ctx, row: str, cand_summary: dict, base_summary: dict, terms: dict,
               *, cand_spec: dict | None = None, decisive=None, present=None,
               v1_terms: dict | None = None, watch_notes: dict | None = None) -> dict:
    """One milestone ROW: is it WORSE than the previous milestone's same row?

    Blocks at ``R_F = max(R_F^eval(row), R_F^seed(chain))`` -- the standardised block seed term is
    unitless and carries across rows -- and guards at the row's own ``R_eval,i`` with the slow-row
    seed term borrowed. Ring completion is WATCH (decision 6); every other guard, the two conditional
    turn survivals included, is scored on every row; a guard that is MISSING on a row is never WORSE
    there.

    Rule v2 (plan D7) has two branches:

    * a LEGACY row -- its per-row table carries ``by_row_source: D00`` (light/ring, copied from D00
      into D00s) -- keeps the frozen v1 per-row rule byte for byte: scored with D00s'
      ``by_row_rule`` (D00's frozen ``decision_rule``; no ``expand_rule``, no decisive/present), the
      copied table's R, D00b's v1 seed terms (``v1_terms``, else read here: the one sanctioned D00b
      citation under rule v2), today's ``_any_worse`` and today's ring-completion clause;
    * the SUITE row takes ``decisive``/``present`` explicitly (a milestone row has no screen: the
      promotion stage's own sets) and is WORSE iff >= 2 decisive UNITS are WORSE -- a unit is a
      block (a member is never classed alone) or a counting guard (every ``survival`` guard and the
      move_turn translation guard; ``cat_tripod_min`` does not count). ``units_worse`` names every
      WORSE unit (one alone is recorded, not a WORSE row) and ``nominal`` the independence-model
      null rate of ">= 2 of these units" (~1 % at B0's set: 3 blocks + 8 counting guards).
    """
    rule = ctx.cands.get("decision_rule") or {}
    if is_rule_v2(rule):
        require_resolution(ctx, f"milestone row {row}")
    resolution = _row_resolution(ctx.resolution, row)
    legacy = resolution.get("by_row_source") == "D00" and is_rule_v2(rule)
    if legacy:
        rule = (ctx.resolution or {}).get("by_row_rule") or {}
        if not rule or is_rule_v2(rule):
            raise ResolutionMissing(f"row {row}: its table is D00's copy but D00s carries no v1 "
                                    "by_row_rule to score it with (plan D7)")
        terms = v1_terms if v1_terms is not None else v1_seed_terms(ctx)[0]
        decisive = present = watch_notes = None
    ring_completion = None
    # A raw v2 rule has no `guards.metrics` to strip: under rule v2 the ring row is a LEGACY row and
    # reaches this clause with D00's frozen v1 rule (the branch above), unchanged (amendment 16).
    if row == "ring" and not is_rule_v2(rule):
        # Decision 6 is conditional: "ring completion WATCH at milestones UNLESS the bootstrap
        # resolves it". Part C (ii)'s "ring completion is WATCH (decision 6)" points back at that
        # sentence rather than restating it, so the escape clause governs. `resolution.py` publishes
        # `by_row["ring__omni"]["metrics"]["completion"]` from the two cross-head ring pairs, and
        # with a half-width in hand the guard is scored like every other: stripping it regardless
        # would let a promotion that holds up on slow/light but collapses ring completion reach the
        # "< 2 rows WORSE" PASS through the other two rows alone.
        r_completion = _num(resolution_metric(resolution, "completion").get("R_eval"))
        if r_completion is None or r_completion <= 0:
            rule = json.loads(json.dumps(rule))
            rule["guards"]["metrics"] = [m for m in (rule.get("guards") or {}).get("metrics") or []
                                        if m != "completion"]
            ring_completion = "WATCH (the bootstrap does not resolve it on this row)"
        else:
            ring_completion = f"scored (the bootstrap resolves it at +/-{fmt_num(r_completion)})"
    # Plan D7: a milestone row has no SCREEN (it gets `summarize_run_dir` output), so a v2 row takes
    # the promotion stage's decisive/present sets as EXPLICIT arguments; they are passed through to
    # the expansion unchanged (None -> the summary's own trained_categories, if it carries any).
    cmpres = _screen_compare(cand_spec or {}, rule, cand_summary, base_summary, resolution, terms,
                             decisive=decisive, present=present, watch_notes=watch_notes)
    worse = _any_worse(cmpres)
    out = {"row": row, "worse": worse, "compare": cmpres,
           "ring_completion": ring_completion,
           "resolution_source": resolution.get("__source__", "global")}
    if legacy:
        out["rule"] = "v1 by_row_rule (D00's frozen rule, D00b seed terms)"
    elif "decisive" in cmpres and _eval_row_kind(row) == "category_suite":
        units, n_blocks, n_guards = worse_units(cmpres)
        out.update(units_worse=units, worse=(units if len(units) >= 2 else []),
                   units={"blocks": n_blocks, "guards": n_guards},
                   nominal=suite_row_null_rate(n_blocks, n_guards, cmpres),
                   rule=">= 2 decisive units WORSE")
    return out


def worse_units(cmpres: dict) -> tuple[list[str], int, int]:
    """``(WORSE units, n scored blocks, n counting guards)`` of a v2 comparison (plan D7).

    A unit is a scored (non-N/A) block, or a guard that COUNTS: every guard except a composite
    (``cat_tripod_min``) -- i.e. the present categories' ``survival`` guards and the move_turn
    translation guard (``ratio``, ``achieved_speed`` under absolute units)."""
    composites = set(eval_categories().COMPOSITES)
    blocks = [k for k, v in cmpres["blocks"].items() if v.get("class") != "N/A"]
    guards = [r["metric"] for r in cmpres["guards"]["rows"] if r["metric"] not in composites]
    units = [f"block {k}" for k in blocks if cmpres["blocks"][k].get("class") == "WORSE"]
    units += [f"guard {m}" for m in cmpres["guards"]["worse"] if m not in composites]
    return units, len(blocks), len(guards)


def suite_row_null_rate(n_blocks: int, n_guards: int, cmpres: dict | None = None) -> float:
    """P(>= 2 of the units WORSE) for a null comparison, independence model: a block is WORSE at
    ``(1 - level)/2`` (two-sided), a guard at ``1 - level`` (one-sided). 3 blocks + 8 guards at
    95 % / 99 % -> 0.0098 (plan D7: "~1 %")."""
    levels = [v.get("level") for v in ((cmpres or {}).get("blocks") or {}).values() if v.get("level")]
    b_level = float(levels[0]) if levels else 0.95
    g_level = float(((cmpres or {}).get("guards") or {}).get("level") or 0.99)
    rates = [(1.0 - b_level) / 2.0] * int(n_blocks) + [1.0 - g_level] * int(n_guards)
    p0 = math.prod(1.0 - p for p in rates)
    p1 = sum(p * math.prod(1.0 - q for j, q in enumerate(rates) if j != i) for i, p in enumerate(rates))
    return max(0.0, 1.0 - p0 - p1)


def v1_seed_terms(ctx: Ctx) -> tuple[dict, list[str]]:
    """D00b's seed terms -- the v1 lineage the legacy light/ring milestone rows are scored with
    under rule v2 (plan D7: the one sanctioned D00b citation). ``({}, [])`` when D00b is absent."""
    entry = entry_by_id(ctx.ledger, "D00b")
    if entry is None:
        return {}, []
    return (entry.get("evidence") or {}).get("seed_terms") or {}, ["D00b"]


def _row_resolution(resolution: dict, row: str) -> dict:
    """The per-ROW resolution table when the record carries one, else the global table.

    A milestone scores three rows (``slow``/``light``/``ring``) and the plan asks for each row's own
    ``R_F^eval(row)`` and per-guard ``R_eval,i``. ``resolution.py`` publishes them under ``by_row``
    keyed by the manifest id; ``rows`` is accepted as a synonym. The fallback to the global table is
    named in the MILESTONE line so a reader knows which resolution a row was judged against.
    """
    sid = _sid_for(row)
    for field in ("by_row", "rows"):
        table = (resolution or {}).get(field) or {}
        for key in (sid, row):
            if key in table:
                out = dict(table[key])
                out["__source__"] = f"{field}[{key}]"
                # A v2 record's per-row table (the suite's) is measured on the record's decisive_at;
                # carried along so a narrower comparison halts here too (plan D5), never on a D00 copy.
                if (resolution or {}).get("decisive_at") and out.get("by_row_source") != "D00":
                    out.setdefault("decisive_at", resolution["decisive_at"])
                return out
    out = dict(resolution or {})
    out["__source__"] = "global (the record carries no per-row table)"
    return out


def reference_run_dir(ref, row: str) -> Path:
    """The run dir a ``milestone_reference`` entry names: a recorded legacy run dir (light/ring), or
    -- the suite row, plan D7/D10 -- ``{eval_dir, checkpoint}``: the reeval ROOT resolved to its
    newest COMPLETE stamp whose checkpoint matches, by the SAME helper ``load_row`` uses
    (``resolution.resolve_eval_dir``; a ``ResolutionError`` names every refused stamp)."""
    if isinstance(ref, dict):
        return Path(resolution_tool().resolve_eval_dir(ref["eval_dir"], _sid_for(row),
                                                       checkpoint=ref.get("checkpoint")))
    return repo_path(ref)


def _reference_watch(ctx: Ctx, heads: dict, terms: dict,
                     cand_spec: dict | None = None) -> tuple[dict, list[str]]:
    """M1's WATCH-only comparison against the RECORDED 2b / 2a rows named in ``candidates.yaml``.

    Plan, Part C (ii): "at M1 WATCH-only vs the seed-3 2b record and vs ``outputs/omni_plan/
    reeval_2a`` for the 2a head". There is no previous milestone to roll against, so this is the
    campaign's only cross-lineage read -- and without it the MILESTONE line names those rows as the
    reference while comparing against nothing.

    Both sides go through :func:`summarize_run_dir`, i.e. ``resolution.py``'s shared re-score, so
    the recorded side has the F block's closed fractions too (its aggregate predates the key). The
    compute is unequal and the lineage different, so nothing here is ever a verdict: the rows are
    recorded and the line says WATCH.
    """
    ref = ctx.cands.get("milestone_reference") or {}
    if not ref:
        return {}, ["no milestone_reference in candidates.yaml -- no recorded rows to WATCH against"]
    out: dict[str, dict] = {}
    lines: list[str] = []
    for stage in ("2b", "2a"):
        table = ref.get(stage) or ref.get(str(stage)) or {}
        marks = []
        # Plan D7: the suite row is scored on the promotion STAGE's own trained categories.
        sets = (heads.get(stage) or {}).get("trained_categories") or {}
        for row in promotion_eval(ctx.cands):
            run_dir = (heads.get(stage, {}).get("evals") or {}).get(row)
            rel = table.get(row)
            if not run_dir or not rel:
                marks.append(f"{row} n/a")
                continue
            try:
                ref_dir = reference_run_dir(rel, row)
            except Exception as exc:                                    # noqa: BLE001
                out[f"{stage}/{row}"] = {"reference": str(rel), "missing": True, "why": str(exc)}
                marks.append(f"{row} MISSING ({exc})")
                continue
            a = summarize_run_dir(run_dir, _sid_for(row))
            b = summarize_run_dir(ref_dir, _sid_for(row)) if Path(ref_dir).exists() else None
            if a is None or b is None:
                out[f"{stage}/{row}"] = {"reference": str(ref_dir), "missing": True}
                marks.append(f"{row} MISSING")
                continue
            kw = ({"decisive": sets.get("decisive"), "present": sets.get("present"),
                   "watch_notes": sets.get("watch") or {}}
                  if _eval_row_kind(row) == "category_suite" and is_rule_v2(ctx.cands.get("decision_rule"))
                  else {})
            try:
                verdict = _row_worse(ctx, row, a, b, terms, cand_spec=cand_spec, **kw)
            except (ValueError, ResolutionMissing) as exc:
                # WATCH only: a row that cannot be scored is reported, never a halt of a milestone
                out[f"{stage}/{row}"] = {"reference": str(ref_dir), "missing": True, "why": str(exc)}
                marks.append(f"{row} n/a ({exc})")
                continue
            verdict["reference"] = str(ref_dir)
            verdict["facing_closed_via"] = {"head": a.get("facing_closed_via"),
                                            "reference": b.get("facing_closed_via")}
            out[f"{stage}/{row}"] = verdict
            marks.append(f"{row} {'below on ' + str(verdict['worse']) if verdict['worse'] else 'ok'}")
        lines.append(f"{stage} vs the recorded reference (WATCH, never a gate): " + "; ".join(marks))
    return out, lines


def _close_entry(entry: dict, chosen: str, why: str) -> None:
    """Close a milestone entry: ``chosen`` + ``final``, checked, with the sub-status cleared.

    ``accepted_i`` is an answer to a (i) FAIL and means "the promotion may run"; once the promotion
    has landed the entry is ``final`` and the sub-status is spent, so leaving it on would make
    ``ledger_cli show`` print ``final/accepted_i`` forever (``_milestone_fail`` already clears it).
    """
    check_transition(entry.get("status") or "open", "final", "driver")
    _history(entry, "final", "driver", why)
    entry["chosen"], entry["status"] = chosen, "final"
    entry["sub_status"] = None


def _close_milestone_promo(ctx: Ctx, item: dict, header: dict) -> None:
    """(ii) the promotion verdict: WATCH at M1, a rolling equal-compute comparison from M2."""
    mid = item["candidate"]
    entry = newest_milestone_entry(ctx.ledger, mid, set(ctx.view["entries"]))
    if entry is None:
        # The (i) entry was squashed while the promotion ran. The eligibility is DERIVED the way
        # `milestone_i` derives it (one per milestone entry ever written for this id), not reset to
        # 1 -- `--drop`'s allowance and the automatic-rerun count are both read off it.
        eligibility = 1 + sum(1 for e in entries(ctx.ledger)
                              if e.get("kind") == "milestone" and e.get("candidate") == mid)
        entry = _milestone_entry_new(ctx, mid, item["recipe"], eligibility)
    v2 = is_rule_v2(ctx.cands.get("decision_rule"))
    heads, gate_lines = {}, []
    for stage in ("2a", "2b"):
        rec = ctx.c.recs.get(f"{header['xid']}:{stage}") or {}
        heads[stage] = {"ckpt": rec.get("ckpt"),
                        "evals": {row: (summary or {}).get("run_dir")
                                  for row, summary in (rec.get("evals") or {}).items()}}
        if v2:
            # Plan D7: the suite row is scored on THIS promotion stage's trained categories (its
            # composed env; D4's compute-and-append rule covers a header that predates the record),
            # passed explicitly -- a milestone row has no screen to carry them.
            tc = header_trained_categories(ctx, header, stage)
            if tc:
                heads[stage]["trained_categories"] = {"decisive": list(tc.get("decisive") or []),
                                                      "present": list(tc.get("present") or []),
                                                      "watch": dict(tc.get("watch") or {})}
        # The written 2a/2b turn gate on each promotion stage -- WATCH context, never a verdict (see
        # :func:`gate_watch`). The promotion's verdict is the rolling row comparison below; this says
        # whether the head the milestone is about to record would clear the gate the phase pipeline
        # enforces at the same two stages. It reads the suite's legacy per-hand keys (plan D7).
        gate_lines.append(gate_watch("gate_facing", _gate_row(rec.get("evals")), label=stage))
    entry["promoted_head"] = heads
    entry["experiment"] = header["xid"]
    entry["base_code_sha"] = header.get("code_sha")
    prev_heads, prev_id = _previous_promoted(ctx, mid)
    try:
        terms, seed_ids = seed_terms_for(ctx, header.get("chain"))
    except ResolutionMissing as exc:
        _campaign_halt(ctx, "resolution_missing", f"{mid} (ii): {exc}")
    # the legacy light/ring rows' v1 lineage (plan D7) -- D00b's terms, whatever the chain
    v1_terms = v1_seed_terms(ctx)[0] if v2 else None

    def row_sets(stage: str, row: str) -> dict:
        if not v2 or _eval_row_kind(row) != "category_suite":
            return {}
        sets = heads[stage].get("trained_categories") or {}
        return {"decisive": sets.get("decisive") or [], "present": sets.get("present") or [],
                "watch_notes": sets.get("watch") or {}}
    # A recipe that adopted a cap/clip/ellipse candidate is scored in absolute units at the
    # milestone too: those knobs are `eval_carry`, so the two sides' rows ran under different ratio
    # denominators (decision 5). The header carries `adopted`; a header written before it did (or
    # any other recipe with no adoption list at all) falls back to a recompose rather than silently
    # reading "no adoptions" and scoring the promotion in the wrong unit system.
    promo_recipe = item.get("recipe") or {}
    if "adopted" not in promo_recipe:
        promo_recipe = milestone_recipe(ctx.cands, ctx.ledger, mid, ctx.code, set(ctx.view["entries"]))
    spec = _recipe_cand_spec(ctx, promo_recipe)
    fields = [f"{mid} (ii) promotion {header['xid']}: 2a {Path(heads['2a']['ckpt'] or '-').name}, "
              f"2b {Path(heads['2b']['ckpt'] or '-').name}", *gate_lines]
    watch_2a, watch_2a_rows = [], {}
    for row, run_dir in (heads["2a"]["evals"] or {}).items():
        prev_dir = ((prev_heads or {}).get("2a") or {}).get("evals", {}).get(row) if prev_heads else None
        if not prev_dir or not run_dir:
            watch_2a.append(f"2a/{row}: no previous 2a head")
            continue
        # WATCH means "reported", not "named": the rows are compared the same way the 2b rows are
        # (same standardiser, same re-score), and the verdict reads nothing from them.
        a, b = summarize_run_dir(run_dir, _sid_for(row)), summarize_run_dir(prev_dir, _sid_for(row))
        if a is None or b is None:
            watch_2a.append(f"2a/{row}: vs {prev_id} MISSING")
            continue
        try:
            verdict = _row_worse(ctx, row, a, b, terms, cand_spec=spec, v1_terms=v1_terms,
                                 **row_sets("2a", row))
        except (ValueError, ResolutionMissing) as exc:
            watch_2a.append(f"2a/{row}: vs {prev_id} n/a ({exc})")       # WATCH: reported, never a halt
            continue
        watch_2a_rows[row] = verdict
        watch_2a.append(f"2a/{row}: vs {prev_id} "
                        + (f"below on {verdict['worse']}" if verdict["worse"] else "ok"))
    fields.append("2a rows are WATCH at every milestone: " + "; ".join(watch_2a))
    if prev_heads is None:
        watch_rows, watch_lines = _reference_watch(ctx, heads, terms, spec)
        _close_entry(entry, "WATCH", f"{mid} (ii) WATCH: no previous milestone head")
        entry["evidence"]["promotion"] = {"mode": "WATCH (no previous milestone head)",
                                          "rows": list((heads["2b"]["evals"] or {})),
                                          "reference": watch_rows}
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("MILESTONE", entry["id"], fields + watch_lines + [
            "chosen WATCH (final): at M1 the comparison is unequal compute and a different lineage -- "
            "the milestone_reference rows (the recorded 2b / 2a heads; under rule v2 their suite "
            "reevals) are the reference, not a gate",
            f"these heads are the next milestone's rolling reference"])
        ctx.c.changelog_row(f"{entry['id']} {mid} promotion WATCH — heads recorded")
        return
    verdicts = []
    rows = promotion_eval(ctx.cands)
    for row in rows:
        run_dir = (heads["2b"]["evals"] or {}).get(row)
        prev_dir = ((prev_heads or {}).get("2b") or {}).get("evals", {}).get(row)
        if not run_dir or not prev_dir:
            verdicts.append({"row": row, "worse": [], "missing": True})
            continue
        a = summarize_run_dir(run_dir, _sid_for(row))
        b = summarize_run_dir(prev_dir, _sid_for(row))
        if a is None or b is None:
            verdicts.append({"row": row, "worse": [], "missing": True})
            continue
        try:
            verdict = _row_worse(ctx, row, a, b, terms, cand_spec=spec, v1_terms=v1_terms,
                                 **row_sets("2b", row))
        except ResolutionMissing as exc:
            _campaign_halt(ctx, "resolution_missing", f"{mid} (ii) row {row}: {exc}")
        except ValueError as exc:
            _campaign_halt(ctx, "eval_mismatch", f"{mid} (ii) row {row} cannot be scored: {exc}")
        # `summarize_run_dir` falls back to a plain `summarize()` when the re-score REFUSES a row
        # (the window-replication check), and the F block then runs on `yaw_progress` alone -- half
        # its members, silently. `_reference_watch` already surfaces this for M1's rows; the rolling
        # rows from M2 are where a milestone VERDICT depends on it, so the line names it there too.
        verdict["facing_closed_via"] = {"head": a.get("facing_closed_via"),
                                        "reference": b.get("facing_closed_via")}
        verdicts.append(verdict)
    worse_rows = [v["row"] for v in verdicts if v.get("worse")]
    missing_rows = [v["row"] for v in verdicts if v.get("missing")]
    entry["evidence"]["promotion"] = {"mode": f"rolling vs {prev_id}", "rows": verdicts,
                                      "worse_rows": worse_rows, "missing_rows": missing_rows,
                                      "watch_2a": watch_2a_rows}
    fields.append(f"2b rows vs {prev_id}: " + "; ".join(
        f"{v['row']} {'WORSE ' + str(v['worse']) if v.get('worse') else ('MISSING' if v.get('missing') else 'ok')}"
        for v in verdicts))
    for v in verdicts:
        if "units_worse" in v:
            # Plan D7: the suite row's own rule, with its null rate, on the line the user reads.
            n = v.get("units") or {}
            fields.append(f"{v['row']} row: WORSE iff >= 2 decisive units WORSE ({n.get('blocks')} blocks + "
                          f"{n.get('guards')} counting guards; nominal null rate {v.get('nominal', 0.0):.3f}); "
                          f"units WORSE: {v['units_worse'] or 'none'}")
        elif v.get("rule", "").startswith("v1"):
            fields.append(f"{v['row']} row: the frozen v1 per-row rule (D00's by_row_rule, D00's R, "
                          "D00b's seed terms; any WORSE block or guard)")
    ring_note = next((v.get("ring_completion") for v in verdicts
                      if v.get("row") == "ring" and v.get("ring_completion")), None)
    if ring_note:
        # Decision 6's conditional, made visible: the reader must be able to tell whether the ring
        # row's completion was a gate or a WATCH on this milestone.
        fields.append(f"ring completion: {ring_note}")
    refused = [f"{v['row']} ({side}: {via})" for v in verdicts
               for side, via in (v.get("facing_closed_via") or {}).items()
               if via and "refused" in str(via)]
    if refused:
        fields.append("facing_closed re-score REFUSED on " + "; ".join(refused)
                      + " -- that side's F block ran on yaw_progress alone (half its members)")
    if len(worse_rows) < 2 and missing_rows:
        # FAIL is ">= 2 of the three 2b rows WORSE"; PASS is what the plan calls "otherwise" -- but
        # a row nothing could read is not evidence that it is fine. Two unreadable rows plus one
        # good one would otherwise pass the milestone on one row. The plan's MISSING rule is about
        # a missing GUARD on a row, not a missing row.
        _close_entry(entry, "WATCH", f"{mid} (ii) WATCH: rows {missing_rows} unreadable")
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("MILESTONE", entry["id"], fields + [
            f"chosen WATCH (final): {missing_rows} could not be read (no eval run dir, or an "
            "aggregate this driver cannot summarise), so the rolling comparison is incomplete",
            f"{len(worse_rows)} of {len(rows) - len(missing_rows)} readable rows WORSE -- not a PASS and "
            "not the >= 2 a FAIL needs; re-run the promotion to get a verdict"])
        ctx.c.changelog_row(f"{entry['id']} {mid} promotion WATCH — rows {missing_rows} unreadable")
        return
    if len(worse_rows) < 2:
        _close_entry(entry, "PASS", f"{mid} (ii) PASS: {len(worse_rows)} of three 2b rows WORSE")
        ctx.write_ledger()
        ctx.recompute()
        ctx.save()
        ctx.notify("MILESTONE", entry["id"], fields + [f"chosen PASS (final): {len(worse_rows)} of the "
                                                       "three 2b rows WORSE (FAIL needs >= 2)"])
        ctx.c.changelog_row(f"{entry['id']} {mid} promotion PASS")
        return
    _milestone_fail(ctx, mid, entry, fields, worse_rows)


def _milestone_fail(ctx: Ctx, mid: str, entry: dict, fields: list, worse_rows: list) -> None:
    """A (ii) FAIL: drop the block's last unfrozen adoption ONCE, else park for the user."""
    block = milestone_block_entries(ctx.cands, ctx.ledger, mid, set(ctx.view["entries"]))
    unfrozen = [e for e in block if lc.effective_choice(e) == "ADOPT" and not e.get("frozen")
                and e.get("status") in ("decided", "final")]
    if unfrozen and not milestone_rerun_used(ctx.ledger, mid, ctx.cands, set(ctx.view["entries"])):
        target = unfrozen[-1]
        before = lc.effective_choice(target)
        check_transition(target.get("status"), "overruled", "milestone")
        _history(target, "overruled", "milestone", f"{mid} (ii) FAIL")
        target["status"] = "overruled"
        target["override"] = {"id": f"milestone:{mid}", "by": "milestone", "choice": "REJECT",
                              "why": f"{mid} (ii) FAIL on {worse_rows}", "at": now_iso(),
                              "applied_at": now_iso()}
        check_transition(entry.get("status") or "open", "squashed", "driver")
        entry["pre_squash_status"] = entry.get("status")
        entry["status"] = "squashed"
        entry["squashed_by"] = "driver"
        _history(entry, "squashed", "driver", f"{mid} (ii) FAIL: re-runs once on the reduced recipe")
        ctx.write_ledger()
        note = squash_and_report(ctx, f"{mid} (ii) FAIL: {target['id']} {before} -> REJECT")
        ctx.notify("MILESTONE", entry["id"], fields + [
            f"chosen FAIL: {worse_rows} WORSE (>= 2 of three 2b rows)",
            f"dropped {target['id']} ({target.get('candidate')}) -> REJECT{note}",
            "the milestone re-runs ONCE on the reduced recipe"])
        return
    check_transition(entry.get("status") or "open", "await", "driver")
    _history(entry, "await", "driver", f"{mid} (ii) FAIL on {worse_rows}")
    entry["chosen"], entry["status"] = "FAIL", "await"
    entry["evidence"]["stage"] = "ii"
    entry["sub_status"] = None
    ctx.write_ledger()
    ctx.recompute()
    ctx.state["phase"] = f"await_user:{mid}"
    ctx.save()
    ctx.notify("MILESTONE", entry["id"], fields + [
        f"chosen FAIL (await): {worse_rows} WORSE and "
        + ("nothing unfrozen to drop" if not unfrozen else "the automatic rerun is spent"),
        "accept the promotion as it stands: " + ctx.cli(f"resume-milestone {mid} --accept"),
        "or drop an adoption: " + ctx.cli(f"resume-milestone {mid} --drop <Cid>")])


def milestone_close(ctx: Ctx, item: dict) -> None:
    """A milestone whose two experiments are both omitted: carry the previous heads forward."""
    mid = item["candidate"]
    entry = newest_milestone_entry(ctx.ledger, mid, set(ctx.view["entries"]))
    if entry is None:
        entry = _milestone_entry_new(ctx, mid, item["recipe"], 1)
    prev_heads, prev_id = _previous_promoted(ctx, mid)
    check_transition(entry.get("status") or "open", "final", "driver")
    _history(entry, "final", "driver", f"carried forward from {prev_id}")
    entry.update(chosen="PASS", status="final", promoted_head=prev_heads or {})
    entry.setdefault("evidence", {})["carried_from"] = prev_id
    ctx.write_ledger()
    ctx.recompute()
    ctx.save()
    ctx.notify("MILESTONE", entry["id"],
               [f"{mid}: recipe_id {entry.get('recipe_id')} equals {prev_id}'s -- both experiments are "
                "omitted", f"chosen PASS (final), evidence.carried_from {prev_id}",
                "the rolling reference follows carried_from to the last real promotion"])
    ctx.c.changelog_row(f"{entry['id']} {mid} carried forward from {prev_id} (no GPU)")


def _close_final_s2(ctx: Ctx, item: dict, header: dict) -> None:
    """M3 (iii): the final recipe's seed-2 head through 2a -> 2b. The campaign ends after it."""
    mid = item["candidate"]
    rows, gate_lines = {}, []
    for stage in ("2a", "2b"):
        rec = ctx.c.recs.get(f"{header['xid']}:{stage}") or {}
        rows[stage] = {"ckpt": rec.get("ckpt"),
                       "evals": {row: (s or {}).get("run_dir") for row, s in (rec.get("evals") or {}).items()}}
        gate_lines.append(gate_watch("gate_facing", _gate_row(rec.get("evals")), label=stage))
    entry = {"id": _next_did(ctx.ledger), "kind": "milestone_s2", "candidate": mid,
             "experiment": header["xid"],
             "question": f"{mid}: the final recipe's seed-2 promotion (the head of record is the user's call)",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"promoted_head_s2": rows}, "recipe_id": header.get("recipe_id"),
             "code_sha": ctx.code, "eval_sha": ctx.eval_sha, "tools_sha": ctx.tools, "decided_at": now_iso(),
             "decided_by": "driver", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.recompute()
    ctx.save()
    ctx.notify("MILESTONE", entry["id"],
               [f"{mid} (iii) final seed-2 promotion recorded: 2b {Path(rows['2b']['ckpt'] or '-').name}",
                *gate_lines,
                "both seeds' 2b rows are in the REPORT -- the user names the head of record"])


# ================================================================== the advisory replay
def run_replay(ctx: Ctx, item: dict) -> dict | None:
    """The advisory replay screen, immediately before the dispatch of a candidate that declares it.

    Advisory means advisory: a failure is recorded as ``replay: error`` and the window runs anyway,
    and so is a ``replay: timeout`` (:data:`REPLAY_TIMEOUT_S`) -- neither halts and neither decides.
    The child is registered in ``rp.CURRENT_CHILD`` like any other, so a stop terminates it and the
    result is ``replay: interrupted`` rather than an error.
    """
    cand = candidate_by_id(ctx.cands, item.get("candidate")) or {}
    args = list(cand.get("replay_args") or [])
    replay = ctx.cands.get("replay") or {}
    if not args or replay.get("mode") != "advisory":
        return None
    script = repo_path(replay["script"])
    if not script.exists():
        return {"status": "error", "why": f"{script} does not exist"}
    out = ctx.dir / f"replay_{item['candidate']}.md"
    log = ctx.c.logs / f"replay_{item['candidate']}.log"
    walker = _baseline_raw_dir(ctx)
    if walker:
        args += ["--walker", walker]
    cmd = [PY, str(script), *args, "--out", str(out)]
    ctx.state["current"] = {"tag": f"replay_{item['candidate']}", "phase": "replay", "iters": 0,
                           "kind": "replay", "log": str(log)}
    ctx.save()
    ctx.c.log(f"replay (advisory): {' '.join(cmd)}")
    status, rc = "ok", None
    with log.open("a") as fh:
        proc = subprocess.Popen(cmd, cwd=str(PARKOUR), stdout=fh, stderr=subprocess.STDOUT)
        rp.CURRENT_CHILD = proc
        deadline = time.time() + REPLAY_TIMEOUT_S
        try:
            while proc.poll() is None:
                time.sleep(INFRA_TICK_S)
                if interrupt_reason(ctx, None):
                    rp._stop_child(proc)
                    status = "interrupted"
                    break
                if time.time() >= deadline:
                    # Every other child in this layer has a timeout (3600/5400 s evals, STALL_S for
                    # train, 120 s for the validator's `--help` probe). Without one a hung replay
                    # stalls the campaign indefinitely with `current.kind == "replay"` and only a
                    # stop to end it -- for an ADVISORY screen, so the answer is to drop it and get
                    # on with the window rather than to halt.
                    rp._stop_child(proc)
                    status = "timeout"
                    ctx.c.log(f"replay (advisory): no result after {REPLAY_TIMEOUT_S} s — the child "
                              f"was stopped and the window runs without it")
                    break
        except BaseException:
            # The same guard `_run_eval` and `train` carry. `interrupt_reason` does file I/O and can
            # raise out of `_apply_override`, and an escape here orphaned the replay child AND
            # cleared `rp.CURRENT_CHILD`, the only handle the signal handler could reach it through.
            rp._stop_child(proc)
            raise
        finally:
            rc = proc.poll()
            rp.CURRENT_CHILD = None
    if status not in ("interrupted", "timeout") and rc != 0:
        status = "error"
    return {"status": status, "rc": rc, "out": str(out), "args": args, "log": str(log)}


def _baseline_raw_dir(ctx: Ctx) -> str | None:
    """B0's own raw directory for the replay's walker reference -- the legacy ``slow`` row first, because
    ``replay_reward_economy.py`` reads only the 8-hold slow layout; a suite-only rec falls back to its
    ``suite`` raws (step-5 review, 2026-09-21)."""
    base_id = (ctx.cands.get("baseline") or {}).get("id", "B0")
    for xid, h in all_headers(ctx.state).items():
        if h.get("kind") != "baseline" or h.get("candidate") != base_id:
            continue
        if str(h.get("seed")) != ctx.seed_screen:
            continue
        # `ctx.c.recs` is keyed by the campaign's CURRENT seed (`Campaign.key`), which is the seed
        # of the last experiment that ran -- and `run_replay` is called before this one sets it. So
        # the screen-seed bucket is addressed explicitly; otherwise the first candidate after any
        # seed-2 window silently fell back to the default walker reference.
        bucket = (ctx.state.get("keys") or {}).get(f"seed{ctx.seed_screen}") or {}
        for key in h.get("recs") or []:
            rec = bucket.get(key) or ctx.c.recs.get(key) or {}
            evals = rec.get("evals") or {}
            run_dir = (evals.get("slow") or evals.get("suite") or {}).get("run_dir")
            if run_dir and (Path(run_dir) / "raw").exists():
                return str(Path(run_dir) / "raw")
    return None


# ================================================================== resume
def item_from_header(ctx: Ctx, header: dict) -> dict:
    """Rebuild the work item of an in-flight experiment FROM ITS HEADER.

    From the header, not from a fresh ``compose_recipe``: the overlay a running window trained under
    is frozen there, so a recipe recompute that happened while it ran (an override landing) cannot
    silently change what the resumed stages run. If the recompute means this experiment should not
    continue at all, the derived squash view says so and the resume is skipped.

    :func:`item_from_header_raw` is the ctx-free body (``_next_item_raw`` is a function of ``(cands,
    ledger, state)`` and dispatches a rescored header through it, plan D9 item 6).
    """
    return item_from_header_raw(ctx.cands, header)


#: The stage fields that describe the EVAL side, not the training window: a resumed header takes
#: them from the CURRENT chain definition (plan D9 item 6) -- the eval side is not a property of
#: the window, and a pre-suite header's frozen ``eval: [slow]`` / hold-keyed floor would otherwise
#: evaluate a row nothing reads any more.
EVAL_SIDE_STAGE_KEYS = ("eval", "floor", "categories", "previous_categories")


def item_from_header_raw(cands: dict, header: dict) -> dict:
    """:func:`item_from_header` without a ``Ctx`` -- the frozen overlay and stage list, with each
    frozen ``stage_specs`` entry's ``eval``/``floor``/``categories``/``previous_categories`` taken
    from the CURRENT chain when that chain exists in ``candidates.yaml`` (a smoke header keeps its
    frozen specs; so does a key the current stage does not carry). Carries the header's ``xid``."""
    kind = header.get("kind")
    chain = header.get("chain")
    if kind in ("milestone", "milestone_s2"):
        stages = final_s2_stage_list(cands) if kind == "milestone_s2" else promotion_stage_list(cands)
    elif header.get("stage_specs"):
        # The header's own stage list: the only source for an experiment whose chain is not in
        # candidates.yaml (--smoke-stage), and the faithful one for every other kind -- a running
        # window must resume on the stages it started, not on a re-read of a file that may have
        # been edited since.
        names = header.get("stages") or [str(s["stage"]) for s in header["stage_specs"]]
        stages = [dict(s) for s in header["stage_specs"] if str(s["stage"]) in names]
        if kind != "smoke" and chain and chain in (cands.get("chains") or {}):
            current = {str(s["stage"]): s for s in chain_stages(cands, chain)}
            for spec in stages:
                cur = current.get(str(spec["stage"]))
                for key in EVAL_SIDE_STAGE_KEYS:
                    if cur is not None and key in cur:
                        spec[key] = json.loads(json.dumps(cur[key]))
    else:
        stages = [dict(s) for s in chain_stages(cands, chain)] if chain else []
        names = header.get("stages") or [str(s["stage"]) for s in stages]
        stages = [s for s in stages if str(s["stage"]) in names]
    recipe = {"chain": chain, "overlay_by_stage": header.get("overlay_by_stage") or {},
              "recipe_id": header.get("recipe_id"), "stages": stages,
              "eval_carry": header.get("eval_carry") or [],
              # Frozen in the header at creation. `anchor` is informational (the drift comparison
              # recomposes it at dispatch); `adopted` is LOAD-BEARING -- `_recipe_cand_spec` reads
              # it to pick F/T vs F_abs/T_abs at a milestone.
              "anchor": header.get("anchor"),
              "has_pool": any("KRABBY_OMNI_CMD_POOL" in v
                              for v in (header.get("overlay_by_stage") or {}).values())}
    if "adopted" in header:
        recipe["adopted"] = list(header.get("adopted") or [])
    base = {"recipe_id": header.get("base_recipe_id"), "chain": chain}
    return _item(kind, cid=header.get("candidate"), seed=header.get("seed"), recipe=recipe,
                 base_recipe=base, stages=stages, decision=header.get("decision"),
                 why="resumed from its header", label=header.get("label") or header.get("xid"),
                 eligibility=header.get("eligibility"), extra={"xid": header.get("xid")}) | \
        {"xid": header.get("xid")}


#: Halts whose window is still WANTED, so the next launch continues it instead of re-running it.
#:
#: The plan's Infra paragraph says so in as many words: "on an OOM/cache signature -> ... halt:<xid>
#: eval_infra (the assistant quarantines the cache and relaunches; THE EXPERIMENT RESUMES AT THE
#: EVAL)", and the HALT line for a twin / batched_s2 / milestone reads "infra only: quarantine/
#: relaunch". Both are infrastructure events, not verdicts about the policy: the training is done
#: and on disk, and `inflight` is left populated on purpose. Without this a Kit DerivedDataCache
#: corruption at the eval of an 8000-it screen threw away ~3.7 GPU-hours on every relaunch.
#:
#: `nan`, `dead_plant`, `config_mismatch`, `eval_mismatch`, `control_missing` and
#: `resume_head_missing` are deliberately NOT here: they are statements about the run itself, and
#: resuming one would only reproduce it.
RESUMABLE_HALTS = ("halt:infra", "halt:eval_infra")


def find_inflight(ctx: Ctx) -> dict | None:
    """The experiment this process (or a killed predecessor) left running, if it may still continue.

    Plan D9: None while an eval-only migration item is pending (the loop runs it first), and a
    ``candidate`` header passes the RANK GATE (:func:`resume_gate`) -- it resumes only when the queue
    would dispatch that same candidate now; otherwise it stays parked and the loop falls through to
    ``next_item``. Twin, baseline, milestone, pair and smoke headers resume unconditionally.
    """
    if migration_item(ctx) is not None:
        return None
    # list(): interrupt_reason() below re-ingests overrides, which can move a header out of
    # `experiments` into `squashed` while this loop runs.
    for xid, header in list((ctx.lad.get("experiments") or {}).items()):
        if header.get("status") not in ("running",) + RESUMABLE_HALTS:
            continue
        if header.get("status") in RESUMABLE_HALTS and not header.get("inflight"):
            continue                      # nothing recorded to continue from
        if xid in ctx.view["experiments"]:
            continue
        if header.get("kind") == "reference":
            continue                      # only `--reference-chain` resumes one (reference_loop)
        why = interrupt_reason(ctx, xid)
        if why:
            ctx.c.log(f"{xid}: not resumed ({why})")
            continue
        parked = resume_gate(ctx, header)
        if parked:
            ctx.c.log(f"{xid}: {parked}")
            continue
        return header
    return None


def resume_gate(ctx: Ctx, header: dict) -> str | None:
    """Plan D9 item 6's rank gate: None when ``header`` may resume now, else ``not resumed (...)``.

    A ``candidate`` header resumes only when ``_next_item_raw`` would dispatch THAT candidate now --
    steps 1-7 return nothing and step 8's first non-terminal ranked candidate is the header's cid (a
    speculative entry never reaches step 8: step 3 returns its pair first). Without it a lower-ranked
    window recorded ``running`` (X05, C2) would resume ahead of re-opened higher-ranked candidates
    (C1, C1b) and spend ~3.3 GPU-h on a base that may still move. Every other kind passes.

    Scoped to a rule-v2 spec, the only kind a campaign can launch on (the validator refuses v1): a
    frozen rule-v1 spec is a record, and its state-machine tests keep the pre-suite resume order.
    """
    if header.get("kind") != "candidate" or not is_rule_v2(ctx.cands.get("decision_rule")):
        return None
    try:
        raw = _next_item_raw(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view,
                             resolution=ctx.resolution)
    except Exception as exc:                                            # noqa: BLE001
        # Park, never resume: a gate that cannot see the queue must not let a lower-ranked window
        # through. next_item then derives the same queue and raises the error where it is visible.
        ctx.c.log(f"{header.get('xid')}: the rank gate could not derive the queue ({exc}); parked")
        return f"not resumed (the rank gate could not derive the queue: {exc})"
    if raw is not None and raw.get("kind") == "candidate" and raw.get("candidate") == header.get("candidate"):
        return None
    return f"not resumed ({_waiting_on(ctx.cands, raw)})"


def _waiting_on(cands: dict, raw: dict | None) -> str:
    if raw is None:
        return "the queue has nothing for it"
    cid, order = raw.get("candidate"), candidate_ids(cands)
    if raw.get("kind") in ("candidate", "skip", "await") and cid in order:
        return f"waiting on {cid} at rank {order.index(cid) + 1}"
    if raw.get("kind") in ("twin", "seed2", "pooled") and cid in order:
        return f"waiting on {cid}'s pair ({raw.get('label')})"
    return f"waiting on {raw.get('label')}"


def prepare_resume(ctx: Ctx, header: dict) -> None:
    """Fill ``inflight.remaining`` / ``inflight.ckpt`` by reading the run directory off disk.

    ``remaining`` is the phase driver's own arithmetic, ``target_final - iter + 1``: a checkpoint at
    600 of a 0..999 window leaves 400 iterations, and a checkpoint AT the final iteration leaves 1,
    which the stage loop reads as "finished" (it continues at the eval instead of re-running that one
    iteration). No checkpoint at all means the stage is re-run from its recorded resume point.
    """
    fl = header.get("inflight")
    if not fl:
        return
    stage = None
    for s in list(header.get("stage_specs") or []) + \
            (chain_stages(ctx.cands, header.get("chain")) if header.get("chain") else []) + \
            promotion_stage_list(ctx.cands) + final_s2_stage_list(ctx.cands):
        if str(s["stage"]) == str(fl.get("stage")):
            stage = s
            break
    if stage is None:
        fl["remaining"] = None
        return
    spec = ph.PHASES[str(stage["base"])]
    exp_dir = rp.LOG_ROOT / spec.experiment
    runs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and f"_{fl.get('tag')}" in p.name) \
        if exp_dir.exists() else []
    # Run-dir names start with their launch timestamp, so they sort chronologically: resume from the
    # NEWEST run that saved a checkpoint. Tags are not campaign-unique (a smoke drill's X01 reuses an
    # older drill's tag), so pooling every run's checkpoints could resume from a stranger's model.
    models: list[Path] = []
    for r in reversed(runs):
        models = sorted(r.glob("model_*.pt"), key=rp._ckpt_iter)
        if models:
            break
    if not models:
        fl["remaining"] = None
        fl["ckpt"] = None
        ctx.c.log(f"{header['xid']}:{fl.get('stage')} — no checkpoint under {exp_dir} for tag "
                  f"{fl.get('tag')}; the stage re-runs from {fl.get('resume')}")
        return
    last = str(models[-1])
    target_final = rp._ckpt_iter(fl.get("resume")) + int(fl.get("iters") or 0) - 1
    fl["ckpt"] = last
    fl["remaining"] = max(0, target_final - rp._ckpt_iter(last) + 1)
    ctx.c.log(f"{header['xid']}:{fl.get('stage')} — resuming at {Path(last).name} "
              f"({fl['remaining']} iterations left of {fl.get('iters')})")


# ================================================================== the migration (plan D9)
#: The eval-only work a relaunch under a new ``eval_sha`` does before ANY resume or queue item (plan
#: D9, "zero decisions by hand"), in this order: (1) rescreen every ok screen of this code measured
#: under an older eval, (2) reeval every ``role: pair`` resolution input, (3) ``D00s``, (4)
#: ``D00bs``, (5) rescore every floor REJECT measured under an older eval. Every item is DERIVED
#: from the records on every loop pass -- nothing stores a migration queue -- and none trains.
MIGRATION_KINDS = ("rescreen", "reeval", "resolution", "rescore", "rescore_complete")
#: The two resolution kinds ``--accept-resolution <kind> WHY`` may accept (plan D9 item 4).
RESOLUTION_KINDS = ("D00s", "D00bs")


def _cur_eval(ctx: Ctx) -> str:
    """The current ``eval_sha`` the migration is keyed on (``lad.eval_sha`` once ``main`` records it,
    which is ``ctx.eval_sha``)."""
    return str(ctx.eval_sha or ctx.lad.get("eval_sha") or "")


def suite_resolution_path(ctx: Ctx, eval_sha: str | None = None) -> Path:
    """``<campaign>/resolution.<eval_sha[:12]>.json`` -- D00s' record (plan D9 item 3). Absolute (the
    driver's cwd is the repo root), and never the pre-suite ``resolution.json`` (D00's)."""
    return Path(ctx.dir).resolve() / f"resolution.{str(eval_sha or _cur_eval(ctx))[:12]}.json"


def rejected_resolution_path(ctx: Ctx, kind: str) -> Path:
    """``resolution.<eval_sha[:12]>.<kind>.rejected.json``: the ONLY write of a failing acceptance --
    a diagnostic, overwritten on each recomputation and never cited (plan D9 item 4)."""
    return Path(ctx.dir).resolve() / f"resolution.{_cur_eval(ctx)[:12]}.{kind}.rejected.json"


def _ensure_d01(ctx: Ctx) -> bool:
    """``D01`` (decision 8's turn-rate evidence) once -- True when it was written now."""
    if entry_by_id(ctx.ledger, "D01") is not None:
        return False
    entry = {"id": "D01", "kind": "measurement", "candidate": None,
             "question": "what turn rate can this plant actually deliver?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "evidence": {"turn_rate": D01_EVIDENCE}, "code_sha": ctx.code, "eval_sha": ctx.eval_sha,
             "tools_sha": ctx.tools, "decided_at": now_iso(), "decided_by": "record", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.lad.setdefault("resolution_ids", []).append("D01")
    return True


def load_resolution(ctx: Ctx) -> dict | None:
    """The resolution every decision is standardised by, loaded at STARTUP -- both branches, the
    ``--smoke-stage`` one included (plan D9 item 3). It never computes anything.

    Rule v2: ``resolution.<eval_sha[:12]>.json`` (D00s at the current eval) when it exists, else
    ``ctx.resolution = None`` -- never the pre-suite slow-row record, which no v2 decision may be
    taken on. D00s is a migration QUEUE item (:func:`migration_item`), and every v2 decision seam
    halts ``halt:resolution_missing`` while this is None. A v1 spec keeps its pre-suite behaviour:
    :func:`ensure_resolution` (``resolution.json`` + D00/D01), or -- under ``--smoke-stage``, which
    never decides -- the record as it lies (``{}`` when absent).
    """
    smoke = bool(getattr(ctx.args, "smoke_stage", None))
    if not is_rule_v2(ctx.cands.get("decision_rule")):
        if smoke:
            path = ctx.dir / "resolution.json"
            ctx.resolution = json.loads(path.read_text()) if path.exists() else {}
            return ctx.resolution
        return ensure_resolution(ctx)
    wrote = False if smoke else _ensure_d01(ctx)
    ctx.resolution = None
    cur = _cur_eval(ctx)
    path = suite_resolution_path(ctx)
    if cur and path.exists():
        payload = json.loads(path.read_text())
        if str(payload.get("eval_sha") or "") != cur:
            ctx.c.log(f"{path.name} records eval_sha {payload.get('eval_sha')}, not {cur}: not loaded")
        else:
            entry = current_resolution_entry(ctx, "D00s")
            if entry is not None:
                payload["id"] = entry["id"]
            ctx.resolution = payload
    if wrote:
        ctx.recompute()
        ctx.save()
    return ctx.resolution


def resolution_acceptance(ctx: Ctx, kind: str) -> dict | None:
    """The newest ``lad.resolution_acceptances`` record ``{kind, eval_sha, why, at}`` of ``kind`` AT
    THE CURRENT ``eval_sha`` (``--accept-resolution <kind> WHY``), or None."""
    cur = _cur_eval(ctx)
    found = None
    for rec in ctx.lad.get("resolution_acceptances") or []:
        if rec.get("kind") == kind and str(rec.get("eval_sha") or "") == cur:
            found = rec
    return found


def pair_inputs(cands: dict) -> list[dict]:
    """The checkpoint-shaped ``role: pair`` resolution inputs -- the D00 heads a migration REEVALS
    (plan D9 item 2). ``seed_pair`` entries are filled by the rescreens and never reevaluated."""
    return [dict(e) for e in cands.get("resolution_inputs") or []
            if str(e.get("role") or "pair") == "pair" and e.get("checkpoint") and not e.get("run_dir")]


def reeval_record(ctx: Ctx, head: str, eval_sha: str | None = None) -> dict | None:
    """The newest ``lad.reevals`` record ``{head, eval_sha, stamp}`` of ``head`` at ``eval_sha``."""
    want = str(eval_sha or _cur_eval(ctx))
    found = None
    for rec in ctx.lad.get("reevals") or []:
        if str(rec.get("head")) == str(head) and str(rec.get("eval_sha") or "") == want:
            found = rec
    return found


def b0_screens(ctx: Ctx) -> tuple[dict | None, dict | None]:
    """The baseline recipe's ok s3 / s2 screens at the CURRENT eval (``ok_screen``) -- D00bs' pair."""
    order = candidate_ids(ctx.cands)
    b0 = compose_recipe(ctx.ledger, ctx.cands, ctx.code, upto=(order[0] if order else None),
                        squashed=set(ctx.view["entries"]))
    return (ok_screen(ctx.state, ctx.code, b0["recipe_id"], ctx.seed_screen, ctx.view),
            ok_screen(ctx.state, ctx.code, b0["recipe_id"], ctx.seed_confirm, ctx.view))


def rescreen_due(ctx: Ctx, xid: str, header: dict, eval_sha: str) -> bool:
    """D9 item 1: a non-squashed ``ok`` screen (``SCREEN_KINDS``) of the current or an aliased
    ``code_sha`` whose ``eval_sha`` is stale. A header stamped per D8 (``eval_sha`` written WITH its
    screen) never matches again -- a pre-suite window resumed under the new eval included."""
    return (header.get("kind") in SCREEN_KINDS and header.get("status") == "ok"
            and xid not in ctx.view["experiments"]
            and same_code(ctx.state, header.get("code_sha"), ctx.code)
            and str(header.get("eval_sha") or "") != eval_sha
            and bool(header.get("recs")))


def _floor_spec(header: dict, sname) -> dict | None:
    """The header's RECORDED ``stage_specs`` entry of ``sname`` (a LIST, looked up by ``stage``)."""
    return next((s for s in header.get("stage_specs") or [] if str(s.get("stage")) == str(sname)), None)


def rescore_due(ctx: Ctx, xid: str, header: dict, eval_sha: str) -> str | None:
    """D9 item 5: the floor stage to rescore, or None.

    Only a floor MEASURED UNDER AN OLDER EVAL: ``rejected_by_floor`` with a recorded ``floor_stage``,
    an eval-row floor (the RECORDED stage spec's ``floor.kind`` in ``survival``/``category_survival``;
    a ``dead_walker`` REJECT is a training-log verdict and never rescored), ``floor_eval_sha`` !=
    the current eval (absent = pre-suite), current or aliased code, and no ``rescores[]`` record at
    the current eval -- so a floor REJECT written under this eval is never a second-chance floor.
    """
    fs = header.get("floor_stage")
    if header.get("status") != "rejected_by_floor" or not fs or xid in ctx.view["experiments"]:
        return None
    if str(header.get("floor") or "").startswith("dead_walker at"):
        return None
    spec = _floor_spec(header, fs)
    if spec is None or str((spec.get("floor") or {}).get("kind")) not in ("survival", "category_survival"):
        return None
    if str(header.get("floor_eval_sha") or "") == eval_sha:
        return None
    if not same_code(ctx.state, header.get("code_sha"), ctx.code):
        return None
    if any(str(r.get("eval_sha") or "") == eval_sha for r in header.get("rescores") or []):
        return None
    return str(fs)


def _floor_entries(ctx: Ctx, xid: str) -> list[dict]:
    """The eval-row floor REJECT entries of experiment ``xid`` (never a dead_walker one), in order."""
    return [e for e in entries(ctx.ledger)
            if e.get("kind") == "candidate" and e.get("experiment") == xid and e.get("floor_stage")
            and not str((e.get("evidence") or {}).get("floor") or "").startswith("dead_walker")]


def rescore_supersedes(ctx: Ctx, xid: str, floor_stage: str, eval_sha: str) -> str | None:
    """The entry a rescore re-decides, computed BEFORE any write (plan D9 item 5): the newest
    ``candidate`` entry of ``xid`` at ``floor_stage`` whose ``eval_sha`` is not the current one
    (absent counts as stale) -- so it can never name the replacement a FAIL writes."""
    found = None
    for e in _floor_entries(ctx, xid):
        if str(e.get("floor_stage")) == str(floor_stage) and _entry_eval_sha(e) != eval_sha:
            found = e.get("id")
    return found


def rescore_completion(ctx: Ctx, xid: str, header: dict, eval_sha: str) -> dict | None:
    """A rescore FAIL that crashed between its ledger write (2) and its state write (3), derived
    INDEPENDENTLY of ``floor_eval_sha`` (the save inside (2) already moved it): a current-eval floor
    entry AND a stale-eval floor entry for the same ``(xid, floor_stage)`` that no
    ``rescores[].supersedes`` names. ``{floor_stage, old, new}`` or None. (A re-opened header's later
    floor REJECT has no stale twin, so it never matches.)"""
    named = {r.get("supersedes") for r in header.get("rescores") or [] if r.get("supersedes")}
    rows = _floor_entries(ctx, xid)
    for fs in dict.fromkeys(str(e["floor_stage"]) for e in rows):
        new = [e for e in rows if str(e["floor_stage"]) == fs and _entry_eval_sha(e) == eval_sha]
        old = [e for e in rows if str(e["floor_stage"]) == fs and _entry_eval_sha(e) != eval_sha
               and e.get("id") not in named]
        if new and old:
            return {"floor_stage": fs, "old": old[-1]["id"], "new": new[-1]["id"]}
    return None


def migration_item(ctx: Ctx) -> dict | None:
    """The next eval-only migration item, DERIVED from the records (plan D9), or None.

    Checked by ``loop`` BEFORE ``find_inflight`` and ``next_item`` -- whose baseline-rerun rule would
    otherwise book an 8000-iteration B0 rerun the moment ``ok_screen`` goes stale -- and
    ``find_inflight`` returns None while one is pending. None under ``--smoke-stage`` (a drill, never
    a campaign), for a v1 spec (the pre-suite record), and before any ``eval_sha`` is known. Order:
    rescreens (xid order), reevals (``resolution_inputs`` order), D00s, D00bs (once both B0 screens
    exist at this eval), then per header a crashed rescore FAIL's completion or a rescore.
    """
    if getattr(ctx.args, "smoke_stage", None):
        return None
    if not is_rule_v2(ctx.cands.get("decision_rule")):
        return None
    cur = _cur_eval(ctx)
    if not cur:
        return None
    exps = sorted(experiments(ctx.state).items(), key=lambda kv: _xid_key(kv[0]))
    for xid, h in exps:
        if rescreen_due(ctx, xid, h, cur):
            return {"kind": "rescreen", "xid": xid, "label": f"rescreen {xid}", "ident": xid,
                    "why": (f"{xid} ({h.get('label')}) is an ok screen of this code measured under eval_sha "
                            f"{str(h.get('eval_sha') or 'none (pre-suite)')[:24]}; the current is {cur[:12]}")}
    for entry in pair_inputs(ctx.cands):
        head = str(entry.get("head"))
        if reeval_record(ctx, head, cur) is None:
            return {"kind": "reeval", "head": head, "entry": entry, "label": f"reeval {head}",
                    "ident": f"reeval_{head}",
                    "why": f"D00 head {head} has no suite eval at eval_sha {cur[:12]} (lad.reevals)"}
    if current_resolution_entry(ctx, "D00s") is None or not suite_resolution_path(ctx).exists():
        return {"kind": "resolution", "which": "D00s", "label": "D00s", "ident": "D00s",
                "why": f"no D00s entry and resolution file at eval_sha {cur[:12]}"}
    if current_resolution_entry(ctx, "D00bs") is None:
        s3, s2 = b0_screens(ctx)
        if s3 is not None and s2 is not None:
            return {"kind": "resolution", "which": "D00bs", "label": "D00bs", "ident": "D00bs",
                    "s3": s3.get("xid"), "s2": s2.get("xid"),
                    "why": f"no D00bs at eval_sha {cur[:12]} and both B0 screens exist ({s3.get('xid')}, "
                           f"{s2.get('xid')})"}
    for xid, h in exps:
        comp = rescore_completion(ctx, xid, h, cur)
        if comp is not None:
            fs = comp["floor_stage"]
            return {"kind": "rescore_complete", "xid": xid, "floor_stage": fs, "old": comp["old"],
                    "new": comp["new"], "label": f"rescore {xid}:{fs} (completion)", "ident": f"{xid}:{fs}",
                    "why": f"{comp['new']} (current eval) and {comp['old']} (stale) both decide "
                           f"{xid}:{fs} and no rescores[] record names {comp['old']}: step (3) never landed"}
        fs = rescore_due(ctx, xid, h, cur)
        if fs:
            return {"kind": "rescore", "xid": xid, "floor_stage": fs, "label": f"rescore {xid}:{fs}",
                    "ident": f"{xid}:{fs}",
                    "why": f"{xid}'s floor at {fs} was measured under eval_sha "
                           f"{str(h.get('floor_eval_sha') or 'none (pre-suite)')[:24]}, not {cur[:12]}"}
    return None


def run_migration_item(ctx: Ctx, mig: dict) -> None:
    """Run ONE migration item (its writes are exactly those D9 lists for it)."""
    ctx.c.log(f"migration: {mig['label']} -- {mig['why']}")
    {"rescreen": _migrate_rescreen, "reeval": _migrate_reeval, "resolution": _migrate_resolution,
     "rescore": _migrate_rescore, "rescore_complete": _migrate_rescore_complete}[mig["kind"]](ctx, mig)


def _migration_halt(ctx: Ctx, mig: dict, why: str, detail: str) -> None:
    """``phase = halt:migration <item> <why>``: a CAMPAIGN halt (plan D9) -- no recorded header's
    status is ever rewritten, and the next launch re-derives the same item."""
    _campaign_halt(ctx, f"migration {mig['label']} {why}", detail)


def _migration_eval(ctx: Ctx, mig: dict, item: dict, header: dict, spec, ckpt: str, row: str,
                    tag: str, env: dict, xid: str | None, trained: dict | None) -> dict | None:
    """One migration eval through ``run_experiment``'s infra path (:func:`_eval_row_infra`).

    The summary, or None when the eval was INTERRUPTED -- which writes nothing (never
    ``_mark_interrupt``, which would rewrite a recorded header's status: X01 would turn ``running``);
    the loop's stop check parks and the next launch re-derives the item. A seam error, no metrics
    after the retry, a stop-now infra signature or an EVAL_CARRY mismatch halts the CAMPAIGN."""
    summary, fail = _eval_row_infra(ctx, item, header, spec, ckpt, row, tag, env, xid,
                                    where=f"{mig['label']} {row}", ident=mig["ident"], trained=trained,
                                    resume_note="the migration re-derives this eval-only item at the "
                                                "next launch (nothing is re-trained)")
    if fail is not None:
        what, detail = fail
        if what.startswith("interrupted"):
            ctx.c.log(f"migration {mig['label']}: interrupted ({detail}) -- nothing written; the next "
                      "launch re-derives it")
            return None
        _migration_halt(ctx, mig, what, detail)
    bad = _carry_mismatch(ctx, item, summary, env, spec)
    if bad:
        _migration_halt(ctx, mig, "eval_mismatch", f"{row}: {bad}")
    return summary


def _header_stage(ctx: Ctx, mig: dict, header: dict, item: dict, sname: str) -> tuple[dict, dict, dict]:
    """``(stage, rec, trained categories)`` of a recorded header's stage -- the migration's common
    preamble. D4: the trained-categories record of EVERY stage is written (appended) BEFORE the
    eval; an empty or underivable set is a campaign ``halt:migration ... eval_mismatch``."""
    xid = header["xid"]
    for s in item["stages"]:
        header_trained_categories(ctx, header, str(s["stage"]), stage=s)
    stage = next((s for s in item["stages"] if str(s["stage"]) == str(sname)), None)
    rec = ctx.c.recs.get(f"{xid}:{sname}")
    if stage is None or rec is None or not rec.get("ckpt"):
        _migration_halt(ctx, mig, "record_missing",
                        f"{xid}:{sname} has no stage spec, no rec or no checkpoint in the records")
    tc = header_trained_categories(ctx, header, str(sname), stage=stage)
    if not tc or not tc.get("decisive"):
        _migration_halt(ctx, mig, "eval_mismatch",
                        f"{xid}:{sname}: the stage's trained categories are "
                        f"{'empty' if tc else 'underivable'} (plan D4) -- nothing would be decisive")
    ctx.save()
    return stage, rec, tc


def _eval_rows_of(stage: dict) -> list[str]:
    """The rows a migration eval runs on a stage: its CURRENT eval rows (``item_from_header`` takes
    them from the current chain), the suite when the stage names none."""
    return [str(r) for r in stage.get("eval") or []] or ["suite"]


def _file_rows(header: dict, rec: dict, results: dict) -> list[str]:
    """File ``results`` into ``rec.evals`` BESIDE the rows it keeps (a row it already carried moves
    to ``evals_superseded``, never overwritten silently); returns the run dirs."""
    evals = rec.setdefault("evals", {})
    dirs = []
    for row, summary in results.items():
        if row in evals:
            rec.setdefault("evals_superseded", []).append({"row": row, "eval_sha": rec.get("eval_sha"),
                                                            "summary": evals[row]})
        evals[row] = summary
        if summary.get("run_dir"):
            header.setdefault("eval_dirs", []).append(summary["run_dir"])
            dirs.append(summary["run_dir"])
    return dirs


def _cat_brief(summary: dict) -> str:
    cat = (summary or {}).get("cat") or {}
    return "survival " + ", ".join(f"{c} {fmt_num((cat.get(c) or {}).get('survival'))}"
                                   for c in _ordered_categories(cat))


def _migrate_rescreen(ctx: Ctx, mig: dict) -> None:
    """D9 item 1: run the suite on a recorded ok screen's final-stage checkpoint and file it.

    ``rp.set_seed`` first (X02 is a seed-2 record: filing it under seed 3 would leave it screenless
    and book a 3.9 h rerun); the item is ``item_from_header``'s, so the tag, out root, overlay and
    ``_carry_mismatch`` are the recorded ones. Writes ``rec.evals.suite`` beside the kept ``slow``,
    then ``header.screen`` = ``_screen_of`` of that rec (the old one under ``screens_superseded``),
    ``header.eval_sha`` and ``header.rescreens[]``. No ledger entry.
    """
    xid = mig["xid"]
    header = experiments(ctx.state)[xid]
    rp.set_seed(ctx.c, header.get("seed"))
    item = item_from_header(ctx, header)
    if not item["stages"]:
        _migration_halt(ctx, mig, "record_missing", f"{xid} records no stages")
    sname = str(item["stages"][-1]["stage"])
    stage, rec, tc = _header_stage(ctx, mig, header, item, sname)
    spec = ph.PHASES[str(stage["base"])]
    env = stage_env(item["recipe"], sname)
    tag = rec.get("orig_tag") or _tag_for(ctx, item, xid, sname)
    results = {}
    for row in _eval_rows_of(stage):
        summary = _migration_eval(ctx, mig, item, header, spec, rec["ckpt"], row, tag, env, xid, tc)
        if summary is None:
            return
        results[row] = summary
    _rescreen_write(ctx, mig, header, item, stage, results)


def _rescreen_write(ctx: Ctx, mig: dict, header: dict, item: dict, stage: dict, results: dict) -> None:
    """A rescreen's ONE state write (D8's stamping with it)."""
    xid, sname, cur = header["xid"], str(stage["stage"]), _cur_eval(ctx)
    rec = ctx.c.recs[f"{xid}:{sname}"]
    kept = sorted(r for r in (rec.get("evals") or {}) if r not in results)
    dirs = _file_rows(header, rec, results)
    rec["eval_sha"] = cur
    header.setdefault("screens_superseded", []).append({"eval_sha": header.get("eval_sha"),
                                                        "screen": header.get("screen")})
    header["screen"] = _screen_of(ctx, item, xid)
    header["eval_sha"] = cur
    header.setdefault("rescreens", []).append({"eval_sha": cur, "stage": sname, "rows": list(results),
                                               "run_dirs": dirs, "kept": kept, "at": now_iso()})
    chain = header.get("chain")
    if (chain and header.get("kind") == "baseline" and str(header.get("seed")) == ctx.seed_screen
            and not ((ctx.lad.get("stage_trace_v2") or {}).get(chain) or {}).get(sname)):
        # item 30: the incumbent's rescreen seeds its chain's category trace (otherwise the first v2
        # run of that chain would)
        _lines, mine = _category_retention(dict(stage, eval=list(results)), rec, {}, previous=[])
        if mine:
            ctx.lad.setdefault("stage_trace_v2", {}).setdefault(chain, {})[sname] = mine
    ctx.save()
    scr = header["screen"]
    ctx.notify("MIGRATION", mig["ident"],
               [f"{mig['label']} ({header.get('label')}, {header.get('kind')} s{header.get('seed')}) at "
                f"{sname} -- eval-only: nothing trains", f"{', '.join(results)} filed beside "
                f"{', '.join(kept) or 'nothing'} under eval_sha {cur[:12]}; the old screen is kept under "
                "screens_superseded", "trained: " + ", ".join((scr.get("trained_categories") or {})
                                                             .get("decisive") or []),
                _cat_brief(scr), f"run dir: {', '.join(dirs) or 'n/a'}",
                "no ledger entry (a rescreen re-files a recorded screen)"])
    ctx.c.changelog_row(f"MIGRATION {mig['label']} (eval-only) — {', '.join(results)} filed, "
                        f"eval_sha {cur[:12]}")


def _repo_rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _migrate_reeval(ctx: Ctx, mig: dict) -> None:
    """D9 item 2: the suite on one ``role: pair`` D00 head (``checkpoint + preset + eval_dir``).

    Out root ``EVAL_ROOT/<campaign>/reeval_<head>`` (``eval_morph``'s ``<campaign>/<tag>``); the
    stamp passes D10's freshness rule against the entry's checkpoint BEFORE it is recorded, and on
    completion ``{head, eval_sha, stamp}`` is APPENDED to ``lad.reevals`` (older records kept) --
    exactly the stamp D00s will read."""
    entry, head, cur = mig["entry"], mig["head"], _cur_eval(ctx)
    preset = str(entry.get("preset"))
    if preset not in ph.PHASES:
        _migration_halt(ctx, mig, "eval_mismatch", f"{head}: preset {preset!r} is not a phase")
    spec = ph.PHASES[preset]
    ckpt = str(repo_path(entry["checkpoint"]))
    if not Path(ckpt).exists():
        _migration_halt(ctx, mig, "checkpoint_missing", f"{head}: {ckpt} does not exist")
    try:
        tc = eval_categories().trained_categories(ph.phase_env(preset)).to_dict()
    except ValueError as exc:
        _migration_halt(ctx, mig, "eval_mismatch", f"{head}: preset {preset}'s trained categories: {exc}")
    row = str(entry.get("row") or "suite__omni")
    row = row[:-len("__omni")] if row.endswith("__omni") else row
    item = _item("reeval", cid=head, seed=ctx.seed_screen, label=mig["label"],
                 recipe={"chain": None, "recipe_id": None, "overlay_by_stage": {}, "eval_carry": []})
    header = {"xid": None, "kind": "reeval", "candidate": head}
    summary = _migration_eval(ctx, mig, item, header, spec, ckpt, row, f"reeval_{head}", {}, None, tc)
    if summary is None:
        return
    run_dir = summary.get("run_dir")
    try:
        resolution_tool().check_suite_stamp(run_dir, checkpoint=entry["checkpoint"])
    except Exception as exc:                                             # noqa: BLE001
        _migration_halt(ctx, mig, "stamp_refused",
                        f"{head}: the reeval stamp {run_dir} fails D10's freshness rule ({exc})")
    want = repo_path(entry["eval_dir"]).resolve()
    if want not in Path(run_dir).resolve().parents:
        ctx.c.log(f"reeval {head}: the stamp {run_dir} is not under the yaml's eval_dir {want} -- "
                  "recorded as a full path, which is what D00s reads")
    stamp = _repo_rel(run_dir)
    ctx.lad.setdefault("reevals", []).append({"head": head, "eval_sha": cur, "stamp": stamp,
                                              "checkpoint": entry["checkpoint"], "at": now_iso()})
    ctx.save()
    ctx.notify("MIGRATION", mig["ident"],
               [f"{mig['label']} (preset {preset}) -- eval-only: nothing trains",
                f"lad.reevals += {{head {head}, eval_sha {cur[:12]}, stamp {stamp}}}",
                _cat_brief(summary), "D00s reads exactly this stamp"])
    ctx.c.changelog_row(f"MIGRATION {mig['label']} (eval-only) — stamp {stamp}")


def _migrate_resolution(ctx: Ctx, mig: dict) -> None:
    if mig["which"] == "D00s":
        _migrate_d00s(ctx, mig)
    else:
        _migrate_d00bs(ctx, mig)


def _migrate_d00s(ctx: Ctx, mig: dict) -> None:
    """D9 item 3: ``D00s`` -- ``resolution.resolve()`` IN PROCESS, then the file, then the entry.

    ``res.main`` reads ``resolution_inputs`` only from the yaml on disk and has no channel for the
    recorded stamps, so the driver builds the entries itself: each ``role: pair`` input gets its
    ``lad.reevals`` stamp at the current eval (a missing one raises ``ResolutionError`` naming the
    head), and ``resolve(..., eval_sha, decisive_at=sorted(decisive(B0)), d00=<D00's record>,
    supersedes="D00", candidates_sha256=<the live yaml's>)``. The acceptance is computed BEFORE any
    write: a failure writes only ``resolution.<sha12>.D00s.rejected.json`` and halts
    ``halt:resolution D00s`` (the item stays pending; a relaunch recomputes -- deterministic, CPU
    only) unless ``--accept-resolution D00s WHY`` recorded an acceptance at this eval, which writes
    it with ``acceptance.ok = false``. A ``ResolutionError`` writes nothing and halts. A record
    already on disk at this eval (the entry write crashed) is completed, not recomputed.
    """
    res, cur = resolution_tool(), _cur_eval(ctx)
    path = suite_resolution_path(ctx)
    ctx.state["current"] = {"tag": "D00s", "phase": "resolution", "iters": 0, "kind": "bootstrap",
                            "log": str(ctx.dir / "orchestrator.log")}
    ctx.save()
    payload = None
    if path.exists():
        payload = json.loads(path.read_text())
        if str(payload.get("eval_sha") or "") != cur:
            payload = None
    if payload is None:
        stamps: dict = {}
        for rec in ctx.lad.get("reevals") or []:
            if str(rec.get("eval_sha") or "") == cur:
                stamps[str(rec.get("head"))] = str(rec.get("stamp"))
        inputs = res.apply_reeval_stamps(ctx.cands.get("resolution_inputs") or [], stamps)
        d00_path = ctx.dir / "resolution.json"
        d00 = json.loads(d00_path.read_text()) if d00_path.exists() else None
        try:
            payload = res.resolve(dict(ctx.cands, resolution_inputs=inputs), eval_sha=cur,
                                  decisive_at=sorted(decisive_b0(ctx)), d00=d00, supersedes="D00",
                                  candidates_sha256=sha256_file(ctx.cands["__path__"]))
        except res.ResolutionError as exc:
            _campaign_halt(ctx, "resolution D00s", f"D00s refused: {exc} — fix and relaunch")
        acc = payload.get("acceptance") or {"ok": False, "failures": [
            "the record carries no acceptance (not a rule-version-2 D00s)"]}
        if not acc.get("ok"):
            accepted = resolution_acceptance(ctx, "D00s")
            if accepted is None:
                rejected = res.write_resolution(payload, rejected_resolution_path(ctx, "D00s"))
                _campaign_halt(ctx, "resolution D00s",
                               f"D00s acceptance: {'; '.join(acc.get('failures') or [])} — fix the tools "
                               "and relaunch, or relaunch with --accept-resolution D00s WHY "
                               f"(diagnostic only, never cited: {rejected}); a member with ZERO "
                               "spread is demoted instead (decision_rule.per_category.demoted, "
                               "found by run_candidates.py --q-spread-check)")
            payload["acceptance"] = dict(acc, accepted=accepted)
        res.write_resolution(payload, path)
    _write_d00s_entry(ctx, payload, path)


def _write_d00s_entry(ctx: Ctx, payload: dict, path: Path) -> dict:
    """The D00s ledger entry -- AFTER its file -- then :func:`load_resolution` (so the next item sees
    it in the same process) and its RESOLUTION line."""
    used = {e.get("id") for e in entries(ctx.ledger)}
    ident = "D00s" if "D00s" not in used else f"D00s@{_cur_eval(ctx)[:12]}"
    entry = {"id": ident, "kind": "resolution", "candidate": None,
             "question": "what is the eval resolution every decision is standardised by, on the "
                         "per-category suite?",
             "options": ["RECORD"], "chosen": "RECORD", "status": "final",
             "supersedes": payload.get("supersedes") or "D00",
             "evidence": {"resolution": {k: v for k, v in payload.items() if k != "inputs"},
                          "inputs": payload.get("inputs"), "decision_rule": ctx.cands.get("decision_rule"),
                          "acceptance": payload.get("acceptance"), "path": str(path),
                          "eval_sha": payload.get("eval_sha")},
             "code_sha": ctx.code, "eval_sha": _cur_eval(ctx), "tools_sha": ctx.tools,
             "decided_at": now_iso(), "decided_by": "resolution.py (in process)", "history": []}
    ctx.ledger.setdefault("entries", []).append(entry)
    ctx.write_ledger()
    ctx.lad.setdefault("resolution_ids", []).append(ident)
    ctx.recompute()
    ctx.save()
    load_resolution(ctx)
    ctx.notify("RESOLUTION", ident, resolution_notify_fields(ctx, payload, ident))
    ctx.c.changelog_row(f"{ident} written ({path.name}) — supersedes {entry['supersedes']}")
    return entry


def _migrate_d00bs(ctx: Ctx, mig: dict) -> None:
    """D9 item 4: ``D00bs`` from the two rescreened B0 screens against D00s (``_write_d00bs``, whose
    acceptance is computed before any write). A failure writes only the rejected diagnostic and
    halts ``halt:resolution D00bs``; ``--accept-resolution D00bs WHY`` lets the next computation
    write it with ``acceptance.ok = false``."""
    s3, s2 = b0_screens(ctx)
    if s3 is None or s2 is None:
        _migration_halt(ctx, mig, "control_missing", "a B0 screen vanished between derivation and run")
    entry, ev = _write_d00bs(ctx, s3, s2, accepted=resolution_acceptance(ctx, "D00bs"))
    if entry is None:
        rejected = resolution_tool().write_resolution(
            {"kind": "D00bs", "eval_sha": _cur_eval(ctx), "against": resolution_label(ctx.resolution),
             "pair": {"s3": s3.get("xid"), "s2": s2.get("xid")}, **ev},
            rejected_resolution_path(ctx, "D00bs"))
        _campaign_halt(ctx, "resolution D00bs",
                       f"D00bs acceptance: {'; '.join(ev['acceptance']['failures'])} — fix the tools and "
                       "relaunch, or relaunch with --accept-resolution D00bs WHY "
                       f"(diagnostic only, never cited: {rejected})")


def _migrate_rescore(ctx: Ctx, mig: dict) -> None:
    """D9 item 5: re-decide a floor measured under an older eval, on the suite, at the failed stage's
    checkpoint with that stage's recorded overlay and eval_carry, under D6 on its trained categories.

    ``supersedes`` is computed BEFORE any write. PASS -> ONE state write (``rescores[]`` takes the
    header's floor fields, the rec key is appended, the header re-opens at its NEXT stage) then the
    squash reconcile (cause (f)); FAIL -> the three ordered writes of :func:`_rescore_fail`."""
    xid, fs, cur = mig["xid"], mig["floor_stage"], _cur_eval(ctx)
    header = experiments(ctx.state)[xid]
    supersedes = rescore_supersedes(ctx, xid, fs, cur)            # BEFORE any write (plan D9 item 5)
    rp.set_seed(ctx.c, header.get("seed"))
    item = item_from_header(ctx, header)
    stage, rec, tc = _header_stage(ctx, mig, header, item, fs)
    spec = ph.PHASES[str(stage["base"])]
    env = stage_env(item["recipe"], fs)
    tag = rec.get("orig_tag") or _tag_for(ctx, item, xid, fs)
    results = {}
    for row in _eval_rows_of(stage):
        summary = _migration_eval(ctx, mig, item, header, spec, rec["ckpt"], row, tag, env, xid, tc)
        if summary is None:
            return
        results[row] = summary
    names = [str(s["stage"]) for s in item["stages"]]
    previous = previous_decisive(ctx, header, names, fs)
    try:
        passed, why = stage_floor(stage, {"evals": results}, dead_walker_floor=ctx.lad.get("d03"),
                                  decisive=list(tc.get("decisive") or []), previous=previous)
    except eval_categories().EvalMismatch as exc:
        _migration_halt(ctx, mig, "eval_mismatch", f"{xid}:{fs}: {exc}")
    if passed:
        _rescore_pass(ctx, mig, header, item, fs, results, why, supersedes)
    else:
        _rescore_fail(ctx, mig, header, item, fs, results, why, supersedes)


def _file_rescore_rec(ctx: Ctx, header: dict, fs: str, results: dict, passed: bool, why: str) -> list[str]:
    """The rescored rec: the suite filed beside its old rows and ``floor_rescored`` beside its old
    ``floor`` (item 30) -- in memory; the caller's write persists it."""
    rec = ctx.c.recs[f"{header['xid']}:{fs}"]
    dirs = _file_rows(header, rec, results)
    rec["floor_rescored"] = {"passed": bool(passed), "why": why, "eval_sha": _cur_eval(ctx), "at": now_iso()}
    return dirs


def _next_stage(item: dict, fs: str) -> str | None:
    names = [str(s["stage"]) for s in item["stages"]]
    i = names.index(str(fs)) if str(fs) in names else -1
    return names[i + 1] if 0 <= i < len(names) - 1 else None


def _rescore_pass(ctx: Ctx, mig: dict, header: dict, item: dict, fs: str, results: dict, why: str,
                  supersedes: str | None) -> None:
    """PASS, in ONE write: ``rescores[]`` gains ``{eval_sha, floor, floor_stage, floor_eval_sha,
    verdict, supersedes}`` (the header's floor fields MOVE into it), ``recs`` gains ``<xid>:<fs>``
    (the rec exists with its ckpt; without it the resumed chain would re-train the stage),
    ``eval_sha``, ``inflight = None`` and ``status`` = ``running`` -- or
    ``rescored_awaiting_predicate`` for a candidate that declares ``depends_on``/``conditional``
    (step 8 dispatches it). Then ``squash_and_report``: cause (f) is keyed on STATE, and the view is
    otherwise recomputed only after ledger writes -- without it this process would still see the
    superseded entry effective and the rank gate would resume a lower-ranked window first."""
    xid, cur = header["xid"], _cur_eval(ctx)
    dirs = _file_rescore_rec(ctx, header, fs, results, True, why)
    header.setdefault("rescores", []).append({
        "eval_sha": cur, "verdict": "PASS", "why": why, "supersedes": supersedes,
        "floor": header.pop("floor", None), "floor_stage": header.pop("floor_stage", None),
        "floor_eval_sha": header.pop("floor_eval_sha", None), "run_dirs": dirs, "at": now_iso()})
    key = f"{xid}:{fs}"
    if key not in (header.get("recs") or []):
        header.setdefault("recs", []).append(key)
    cand = candidate_by_id(ctx.cands, header.get("candidate")) or {}
    status = RESCORED_AWAITING if (cand.get("depends_on") or cand.get("conditional")) else "running"
    header.setdefault("history", []).append({"at": now_iso(), "from": header.get("status"), "to": status,
                                             "why": f"rescore PASS at {fs} under eval_sha {cur[:12]}"})
    header.update(eval_sha=cur, inflight=None, status=status)
    ctx.save()
    note = squash_and_report(ctx, f"rescore {xid}:{fs} PASS")
    nxt = _next_stage(item, fs)
    ctx.notify("MIGRATION", mig["ident"],
               [f"{mig['label']} -- eval-only: nothing trains", f"floor PASS: {why}",
                f"supersedes {supersedes or 'nothing (no stale entry found)'} (its floor was measured under an "
                "older eval)", f"{xid} re-opens at {nxt or '(no stage left)'} as {status}"
                + (" (step 8 dispatches it once its predicate holds)" if status == RESCORED_AWAITING else ""),
                f"reconcile{note}"])
    ctx.c.changelog_row(f"MIGRATION {mig['label']} (eval-only) PASS — {xid} re-opens at {nxt}")


def _rescore_fail(ctx: Ctx, mig: dict, header: dict, item: dict, fs: str, results: dict, why: str,
                  supersedes: str | None) -> None:
    """FAIL, in this order (plan D9 item 5): (1) the new floor fields on the header (in memory);
    (2) the ledger REJECT with the current ``eval_sha`` -- its reconcile runs while the superseded
    entry is still effective (no dependant is archived) and its save persists (1); (3) ONE state write
    of the ``rescores[]`` record and ``header.eval_sha``, then the reconcile, where (f) squashes the
    superseded entry. A crash between (2) and (3) is completed on relaunch without a second eval."""
    cur = _cur_eval(ctx)
    _file_rescore_rec(ctx, header, fs, results, False, why)
    header.update(floor=why, floor_stage=fs, floor_eval_sha=cur)                      # (1)
    _dispatch_floor_reject(ctx, item, header, rescore={"eval_sha": cur, "supersedes": supersedes,
                                                       "label": mig["label"]})           # (2)
    old = entry_by_id(ctx.ledger, supersedes) if supersedes else None
    _rescore_fail_complete(ctx, mig, header, fs, old)                                  # (3)


def _rescore_fail_complete(ctx: Ctx, mig: dict, header: dict, fs: str, old: dict | None) -> None:
    """A rescore FAIL's step (3): the ``rescores[]`` record -- ``{eval_sha, verdict: FAIL,
    supersedes, floor, floor_stage, floor_eval_sha}`` with the OLD values from the superseded
    entry's evidence (the save inside (2) already moved the header's) -- and ``header.eval_sha``, in
    one write, then the reconcile (cause (f))."""
    cur, xid = _cur_eval(ctx), header["xid"]
    ev = (old or {}).get("evidence") or {}
    header.setdefault("rescores", []).append({
        "eval_sha": cur, "verdict": "FAIL", "supersedes": (old or {}).get("id"),
        "floor": ev.get("floor"), "floor_stage": (old or {}).get("floor_stage") or ev.get("floor_stage") or fs,
        "floor_eval_sha": _entry_eval_sha(old or {}) or None, "new_floor": header.get("floor"),
        "at": now_iso()})
    header["eval_sha"] = cur
    ctx.save()
    note = squash_and_report(ctx, f"rescore {xid}:{fs} FAIL")
    ctx.c.log(f"{mig['label']}: FAIL recorded; {(old or {}).get('id')} superseded{note}")


def _migrate_rescore_complete(ctx: Ctx, mig: dict) -> None:
    """The relaunch completion of a crashed rescore FAIL: step (3) from the two ledger entries,
    without a second eval."""
    header = experiments(ctx.state)[mig["xid"]]
    _rescore_fail_complete(ctx, mig, header, mig["floor_stage"], entry_by_id(ctx.ledger, mig["old"]))
    ctx.notify("MIGRATION", mig["ident"],
               [f"{mig['label']} -- eval-only, no eval re-run: the FAIL's ledger REJECT {mig['new']} landed "
                f"before the crash; its rescores[] record now names {mig['old']}"])


# ================================================================== the dry run
def dry_run(ctx: Ctx, verdicts: list) -> int:
    """Print everything the campaign would do, touching no GPU and needing no ``resolution.json``."""
    cands, code = ctx.cands, ctx.code
    print(f"# ladder dry run — {ctx.dir}")
    print(f"campaign-dir {ctx.dir}\ncandidates   {cands['__path__']} (sha256 "
          f"{sha256_file(cands['__path__'])[:16]})")
    print(f"plant {cands['plant']}  seeds screen={ctx.seed_screen} confirm={ctx.seed_confirm}  "
          f"baseline {(cands.get('baseline') or {}).get('id')} on chain "
          f"{(cands.get('baseline') or {}).get('chain')}")
    print(f"code_sha {code}\n  pathspecs:")
    for spec in CODE_PATHSPECS:
        print(f"    - {spec}")
    print(f"  fnmatch drops: {list(CODE_DROP_GLOBS)}")
    print(f"  tracked code files hashed: {len(code_rows())}")
    print(f"eval_sha {ctx.eval_sha} (recorded {ctx.lad.get('eval_sha') or 'none: the campaign predates it'})"
          "\n  pathspecs:")
    for spec in EVAL_PATHSPECS:
        print(f"    - {spec}")
    proof = getattr(ctx, "eval_proof", None)
    if proof is not None:
        print(f"  alias proof ({'ok' if proof.get('ok') else 'FAILED'}): recorded code_sha "
              f"{proof.get('sha')} == code_sha_at({proof.get('proved_at')}, CODE_PATHSPECS_V1); "
              f"narrowed there {proof.get('narrowed_at')} vs code_sha() {proof.get('current')} -- "
              f"{proof.get('why')}")
    dirty = dirty_paths()
    print(f"  dirty (code + eval): {dirty if dirty else 'clean'}")
    print(f"tools_sha {ctx.tools} over {list(TOOL_FILES)} + the replay script")

    print("\n## validator")
    for v in verdicts:
        mark = "ok     " if v["ok"] else "REFUSED"
        extra = " (deferred)" if v.get("deferred") else ""
        print(f"  {mark} {v['id']}{extra}")
        for reason in v["reasons"]:
            print(f"      - {reason}")
        for note in v.get("notes") or []:
            print(f"      . {note}")
    for cand in cands.get("candidates") or []:
        ok, why = dispatch_check(cand)
        if not ok:
            print(f"  DISPATCH REFUSED {cand['id']}: {why}")

    print("\n## the derived queue (every decision resolved as a clear REJECT: the no-adoption path)")
    for row in simulate_queue(cands, outcome="REJECT", limit=40, code=code):
        print(f"  {row['label']:<22} {row['kind']:<15} chain {row.get('chain')} "
              f"recipe {row.get('recipe_id')}")
        print(f"      why: {row['why']}")
    print("\n## the derived queue (every decision resolved as a clear ADOPT: the adoption path)")
    for row in simulate_queue(cands, outcome="ADOPT", limit=40, code=code):
        print(f"  {row['label']:<22} {row['kind']:<15} chain {row.get('chain')} "
              f"recipe {row.get('recipe_id')} stages {row.get('stages')}")

    print("\n## stage env of every recipe on the queue (no preset, plant or other-task variable)")
    ledger = {"schema": 1, "entries": []}
    shown = set()
    for label, recipe in _dry_recipes(ctx, ledger):
        if recipe["recipe_id"] in shown:
            continue
        shown.add(recipe["recipe_id"])
        print(f"  {label}: chain {recipe['chain']} recipe {recipe['recipe_id']} "
              f"adopted {recipe['adopted']} eval_carry {recipe['eval_carry']}")
        for stage in [str(s["stage"]) for s in recipe["stages"]] + list(PROMOTION_STAGES):
            env = stage_env(recipe, stage)
            print(f"      {stage:<4} {env}")
        if recipe["conflicts"]:
            print(f"      CONFLICTS {recipe['conflicts']}")

    refs = cands.get("references") or {}
    if refs:
        # NOTE(clock-deprecation 2026-09-23): the reference chains (run with --reference-chain)
        print("\n## references (run one with --reference-chain <name> [--seed N]; judged by hand)")
        for name in refs:
            try:
                spec = reference_spec(cands, str(name))
                recipe = reference_recipe(cands, str(name), code)
            except (KeyError, ValueError) as exc:
                print(f"  {name}: ERROR {exc}")
                continue
            head = reference_head(ctx, str(name))
            print(f"  {name}: chain {spec['chain']} seed {spec['seed'] or '(--seed)'} recipe "
                  f"{recipe['recipe_id']} -- recorded head: "
                  f"{(head or {}).get('ckpt') or 'none yet under this code_sha'}")
            for stage in recipe["stages"]:
                src = resume_reference(stage) or resume_checkpoint(stage) or stage.get("resume_from")
                print(f"      {str(stage['stage']):<5} {stage.get('iters')} its from {src} "
                      f"eval {stage.get('eval')} {stage_env(recipe, str(stage['stage']))}")

    print("\n## D00 (the decision rule the resolution is measured for)")
    rule = cands.get("decision_rule") or {}
    if is_rule_v2(rule):
        # a raw v2 rule has no flat `blocks`/`guards`: print what every comparison can score -- the
        # union of both unit systems at decisive(B0) (plan D5), never a hard-wired block tuple
        print(f"  rule_version 2 expanded at decisive(B0) {decisive_b0(ctx)} (both unit systems):")
        rule = dict(_scoring_union(ctx, ctx.resolution), bootstrap=rule.get("bootstrap"),
                    charter_floors=rule.get("charter_floors"))
    for block, spec in (rule.get("blocks") or {}).items():
        print(f"  block {block} @{spec.get('level')}: {spec.get('metrics')}")
    print(f"  guards @{(rule.get('guards') or {}).get('level')}: "
          f"{(rule.get('guards') or {}).get('metrics')}")
    print(f"  WATCH (never scored): {list(rule.get('watch') or [])}")
    raw_rule = cands.get("decision_rule") or {}
    if flags_enabled(raw_rule):
        print(f"  RIG rule (plan step 4): adopt_requires {raw_rule.get('adopt_requires')}, pair_on "
              f"{raw_rule.get('pair_on') or []}, pilot {raw_rule.get('pilot') or 'none'}, "
              f"steps_guard_flag {bool(raw_rule.get('steps_guard_flag'))}; every flagged DECISION parks "
              "(release: ledger_cli ack-flag <Did>)")
        base_ctl = [c["id"] for c in cands.get("candidates") or [] if baseline_control(c)]
        print(f"  control: baseline {base_ctl or 'none'} (their control and drift anchor are the "
              "baseline's own screens; excluded from compose-on-adopt)")
    for gate, where in GATE_WATCH.items():
        print(f"  WATCH {gate}: {where} — rendered on the stage report and on every DECISION / "
              "MILESTONE line, read by nothing")
    print(f"    gate_1c floors: tripod >= {rp.GATE_1C_TRIPOD}, completion >= {rp.GATE_1C_COMPLETION}, "
          f"yaw-progress >= {rp.GATE_1C_YAW_RATE} rad/s on each turn hold, speed_ratio >= "
          f"{rp.GATE_1C_DIR_RATIO} on {list(rp.GATE_1C_DIR_HOLDS)}, track_ratio >= {rp.GATE_TRACK_RATIO_MIN}")
    print(f"    gate_facing floors: yaw-progress >= {rp.GATE_1C_YAW_RATE} rad/s on each turn hold "
          f"(facing precision {rp.GATE_FACING_MAX_RAD} rad is WATCHED, demoted 2026-09-18)")
    print(f"  bootstrap {rule.get('bootstrap')}  charter floors {rule.get('charter_floors')}")
    path = ctx.dir / "resolution.json"
    if is_rule_v2(cands.get("decision_rule")):
        # Plan D9 item 3: D00s lives in resolution.<eval_sha[:12]>.json and is computed IN PROCESS by
        # a migration queue item; startup only loads it. The pre-suite resolution.json is D00's.
        spath = suite_resolution_path(ctx)
        if spath.exists():
            sres = json.loads(spath.read_text())
            print(f"  D00s record {spath}: present ({len(sres.get('metrics') or {})} metrics, blocks "
                  f"{sorted(sres.get('blocks') or {})})")
            for field in resolution_notify_fields(ctx, sres, "D00s"):
                print(f"    | {field}")
        else:
            print(f"  D00s record {spath}: absent -- a migration queue item computes it in process "
                  "(resolve() with the recorded reeval stamps; plan D9 item 3); startup never does")
        print(f"  D00 (pre-suite) record {path}: " + (
            "present -- its light/ring by_row tables and frozen decision_rule are copied into D00s"
            if path.exists() else "absent -- D00s then carries no light/ring by_row copy"))
    elif path.exists():
        resolution = json.loads(path.read_text())
        print(f"  resolution.json present: {len(resolution.get('metrics') or {})} metrics, "
              f"blocks {sorted((resolution.get('blocks') or {}))}")
        for mid, row in (resolution.get("metrics") or {}).items():
            print(f"    {mid}: R_eval {row.get('R_eval')} @{row.get('level')} block {row.get('block')} "
                  f"n {row.get('n')} source {row.get('source')}")
        for field in resolution_notify_fields(ctx, resolution, resolution_label(resolution)):
            print(f"    | {field}")
    else:
        print(f"  resolution.json absent ({path}) — the dry run does not need it; the driver computes "
              "it by calling resolution.py at startup")
    print(f"\n## resolution inputs ({len(cands.get('resolution_inputs') or [])})")
    for line in resolution_input_lines(ctx):
        print(line)
    print("\n## migration (the REAL migration_item derivation and find_inflight order on this campaign "
          "dir's records -- eval outcomes simulated in memory, nothing written, no GPU)")
    for line in dry_run_migration(ctx):
        print(line)
    print(f"\n## notification contract: tags {list(lc.NOTIFY_TAGS)}")
    print("  " + lc.format_notify("DECISION", "D08",
                                  ["X09 C2-reward {KRABBY_YAW_PROGRESS_W=3.0} on recipe abc s3",
                                   "chosen ADOPT (clear, decided) alt REJECT",
                                   "confirm: " + ctx.cli("confirm D08")]))
    print(f"\n## the launcher\n  {HERE / 'launch_candidates.sh'} --campaign-dir {ctx.dir}")
    return 0


def resolution_input_lines(ctx: Ctx) -> list[str]:
    """The dry run's ``resolution_inputs`` listing -- BOTH shapes (plan D9 item 2): a recorded
    ``run_dir`` (legacy, informational) or ``checkpoint + preset + eval_dir`` (a suite reeval head
    or a B0 rescreen), with the stamp D00s would read at the current eval (``lad.reevals`` for a
    ``pair``, the newest fresh stamp under ``eval_dir`` for a ``seed_pair``)."""
    out = []
    cur = _cur_eval(ctx)
    for item in ctx.cands.get("resolution_inputs") or []:
        head, row, role = item.get("head"), item.get("row"), item.get("role") or "pair"
        if item.get("run_dir"):
            exists = (REPO / item["run_dir"]).exists()
            out.append(f"  {'ok ' if exists else 'ABSENT'} {head}/{row} role={role} "
                       f"provenance_ok={item.get('provenance_ok')} {item['run_dir']}")
            continue
        ckpt = item.get("checkpoint")
        exists = bool(ckpt) and repo_path(ckpt).exists()
        out.append(f"  {'ok ' if exists else 'ABSENT'} {head}/{row} role={role} preset={item.get('preset')} "
                   f"checkpoint {ckpt} eval_dir {item.get('eval_dir')}")
        if role == "pair":
            rec = reeval_record(ctx, str(head), cur) if cur else None
            out.append(f"      stamp: {rec['stamp']} (lad.reevals at eval_sha {cur[:12]})" if rec else
                       f"      stamp: none recorded at eval_sha {cur[:12] or '(none)'} -- the migration "
                       f"reevals {head} (D9 item 2) and D00s reads exactly that stamp")
        else:
            try:
                stamp = resolution_tool().resolve_eval_dir(item["eval_dir"], _sid_for(str(row or "suite")),
                                                          checkpoint=ckpt)
                out.append(f"      newest fresh stamp: {stamp}")
            except Exception as exc:                                     # noqa: BLE001
                out.append(f"      no fresh stamp yet ({str(exc)[:320]}) -- filled by the rescreen "
                           "(D9 item 1), never reevaluated")
    return out


class _DrySim(Ctx):
    """An IN-MEMORY copy of a campaign the dry run drives the REAL derivations on (plan Verification
    4): ``migration_item``, ``find_inflight`` / ``resume_gate``, ``next_item`` and the real record
    writers (``_rescreen_write``, ``_rescore_pass``, ``squash_and_report``) run on deep copies of
    the recorded state and ledger; every write lands in memory or a throw-away temp dir (never the
    campaign dir), and no eval, training or resolution computation ever runs."""

    def __init__(self, ctx: Ctx, tmpdir: Path, sink: list):  # noqa: D107 -- deliberately no super()
        self.dir = Path(tmpdir)
        self.cands = ctx.cands
        self.args = argparse.Namespace(**{**vars(ctx.args), "smoke_stage": None, "dry_run": False})
        self.plant = ctx.plant
        self.seed_screen, self.seed_confirm = ctx.seed_screen, ctx.seed_confirm
        self.c = _SimCampaign(copy.deepcopy(ctx.c.st), ctx.plant, ctx.seed_screen, self.dir, sink)
        self.ledger = copy.deepcopy(ctx.ledger)
        self.code, self.eval_sha, self.tools = ctx.code, ctx.eval_sha, ctx.tools
        self.view = {"experiments": {}, "entries": {}, "candidates": []}
        self._ovr_stamp = None
        self.t0 = time.time()
        self.resolution = None
        self.lad.setdefault("experiments", {})
        self.lad.setdefault("squashed", {})


class _SimCampaign(rp.Campaign):
    """``rp.Campaign`` whose writes go to a sink (the dry run's in-memory campaign)."""

    def __init__(self, st: dict, plant: str, seed: str, cdir: Path, sink: list):  # noqa: D107
        super().__init__(Path(cdir), plant, seed)
        self.st = st
        self.sink = sink

    def save(self) -> None:
        pass

    def log(self, msg: str) -> None:
        self.sink.append(("log", str(msg)))

    def notify(self, msg: str) -> None:
        self.sink.append(("notify", str(msg)))

    def report_block(self, block: str) -> None:
        pass

    def changelog_row(self, row: str) -> None:
        pass


def _dry_decide(sim: Ctx, header: dict, outcome: str = "REJECT") -> None:
    """Close a simulated window: ``ok`` with a stamped screen and a clear ``outcome`` decision."""
    header.update(status="ok", inflight=None, eval_sha=_cur_eval(sim),
                  screen=header.get("screen") or {"note": "dry run"})
    if header.get("kind") == "candidate":
        sim.ledger.setdefault("entries", []).append(
            {"id": _next_did(sim.ledger), "kind": "candidate", "candidate": header.get("candidate"),
             "experiment": header.get("xid"), "chosen": outcome, "options": list(CANDIDATE_OPTIONS),
             "status": "decided", "confidence": "clear", "base_recipe_id": header.get("base_recipe_id"),
             "recipe_id_if_adopted": header.get("recipe_id"), "code_sha": sim.code,
             "eval_sha": _cur_eval(sim), "conditional_refs": [], "history": [], "evidence": {}})
    sim.recompute()


def _dry_migrate(sim: Ctx, mig: dict) -> str:
    """Apply ONE migration item's record effects with a SIMULATED eval outcome (every rescore
    PASSES); the writers are the real ones wherever the item writes state. Returns what it did."""
    cur = _cur_eval(sim)
    if mig["kind"] == "rescreen":
        header = experiments(sim.state)[mig["xid"]]
        rp.set_seed(sim.c, header.get("seed"))
        item = item_from_header(sim, header)
        sname = str(item["stages"][-1]["stage"])
        stage, _rec, _tc = _header_stage(sim, mig, header, item, sname)
        rows = _eval_rows_of(stage)
        _rescreen_write(sim, mig, header, item, stage,
                        {r: {"run_dir": None, "eval_status": "dry-run", "cat": {}} for r in rows})
        return f"{sname}: {', '.join(rows)} filed beside the kept rows; eval_sha stamped"
    if mig["kind"] == "reeval":
        sim.lad.setdefault("reevals", []).append({"head": mig["head"], "eval_sha": cur,
                                                  "stamp": "(dry run)", "at": now_iso()})
        return (f"lad.reevals += {{head {mig['head']}, eval_sha {cur[:12]}}} (dry run: no eval; the real "
                f"one runs the suite on {mig['entry'].get('checkpoint')} under tag reeval_{mig['head']})")
    if mig["kind"] == "resolution":
        ident = mig["which"]
        used = {e.get("id") for e in entries(sim.ledger)}
        if ident in used:
            ident = f"{ident}@{cur[:12]}"
        entry = {"id": ident, "kind": "resolution", "candidate": None, "chosen": "RECORD",
                 "options": ["RECORD"], "status": "final", "eval_sha": cur, "history": [],
                 "evidence": {"seed_terms": {"blocks": {}, "metrics": {}}}}
        if mig["which"] == "D00s":
            stub = {"rule_version": 2, "eval_sha": cur, "decisive_at": sorted(decisive_b0(sim)),
                    "supersedes": "D00", "id": ident}
            suite_resolution_path(sim).write_text(json.dumps(stub))
            sim.resolution = stub
            entry["supersedes"] = "D00"
        sim.ledger.setdefault("entries", []).append(entry)
        sim.recompute()
        return (f"{ident} at eval_sha {cur[:12]}" + (f" (pair {mig.get('s3')}/{mig.get('s2')})"
                                                     if mig["which"] == "D00bs" else
                                                     f", file {suite_resolution_path(sim).name}"))
    if mig["kind"] == "rescore":
        header = experiments(sim.state)[mig["xid"]]
        rp.set_seed(sim.c, header.get("seed"))
        item = item_from_header(sim, header)
        _header_stage(sim, mig, header, item, mig["floor_stage"])
        sup = rescore_supersedes(sim, mig["xid"], mig["floor_stage"], cur)
        _rescore_pass(sim, mig, header, item, mig["floor_stage"],
                      {"suite": {"run_dir": None, "eval_status": "dry-run", "cat": {}}},
                      "dry run: PASS assumed (no eval)", sup)
        return (f"PASS assumed -> {mig['xid']} re-opens at {_next_stage(item, mig['floor_stage'])} as "
                f"{header.get('status')}; rescores[] supersedes {sup}")
    header = experiments(sim.state)[mig["xid"]]
    _rescore_fail_complete(sim, mig, header, mig["floor_stage"], entry_by_id(sim.ledger, mig["old"]))
    return f"step (3) completed from {mig['new']} / {mig['old']}"


def dry_run_migration(ctx: Ctx, *, limit: int = 24) -> list[str]:
    """Plan Verification 4's dry-run section: the REAL ``migration_item`` derivation and the REAL
    ``find_inflight`` order, iterated on the campaign dir's own records.

    Simulates the relaunch (``lad.eval_sha`` = the current eval, the proved ``code_sha`` alias),
    reports what the launch reconcile would squash, walks every migration item with simulated
    outcomes (rescore PASS), then the resume/queue order with every decision a clear REJECT: each
    live candidate header's rank-gate verdict, the header ``find_inflight`` returns, or the next
    queue item. Nothing is written to the campaign dir and nothing is launched.
    """
    import tempfile                                                   # noqa: PLC0415
    lines: list[str] = []
    if not is_rule_v2(ctx.cands.get("decision_rule")):
        return ["  (rule_version 1: no migration -- the pre-suite record is scored as it stands)"]
    sink: list = []
    with tempfile.TemporaryDirectory(prefix="ladder-dry-run-") as tmp:
        sim = _DrySim(ctx, Path(tmp), sink)
        recorded_eval = sim.lad.get("eval_sha")
        sim.lad["eval_sha"] = ctx.eval_sha
        proof = getattr(ctx, "eval_proof", None)
        old_code = sim.lad.get("code_sha")
        if old_code and old_code != ctx.code:
            if not (proof or {}).get("ok"):
                lines.append(f"  WARNING: the alias proof did not succeed ({(proof or {}).get('why')}): a "
                             "relaunch falls through to the code_sha refusal; simulated as if proved")
            sim.lad.setdefault("code_sha_aliases", []).append(
                {"sha": old_code, "proved_at": (proof or {}).get("proved_at"), "why": "dry run", "at": now_iso()})
        sim.lad["code_sha"] = ctx.code
        lines.append(f"  simulated relaunch: lad.eval_sha <- {ctx.eval_sha[:12]} (recorded: "
                     f"{str(recorded_eval)[:12] if recorded_eval else 'none -- pre-suite'}); code_sha "
                     + (f"{str(old_code)[:12]} aliased to {ctx.code[:12]}" if old_code and old_code != ctx.code
                        else "unchanged"))
        view = sim.recompute()
        moved = [f"{k} ({v})" for k, v in view["entries"].items()
                 if (entry_by_id(sim.ledger, k) or {}).get("status") != "squashed"]
        moved += [f"{k} ({v})" for k, v in view["experiments"].items() if k in experiments(sim.state)]
        decided = [f"{e['id']} {e.get('status')}" for e in entries(sim.ledger)
                   if e.get("kind") == "candidate"]
        lines.append("  launch reconcile: " + ("squashes nothing" if not moved else
                                               "SQUASHES " + "; ".join(moved))
                     + f" ({', '.join(decided) or 'no candidate entry'})")
        running = [f"{x} ({h.get('label')})" for x, h in experiments(sim.state).items()
                   if h.get("status") == "running"]
        n = 0
        mig = migration_item(sim)
        if mig is not None and running:
            lines.append(f"  recorded running: {', '.join(running)} -- find_inflight returns "
                         f"{(find_inflight(sim) or {}).get('xid')} while a migration item is pending")
        while mig is not None and n < limit:
            n += 1
            try:
                did = _dry_migrate(sim, mig)
            except SystemExit:
                halt = [m for kind, m in sink if kind == "notify" and m.startswith("HALT")]
                lines.append(f"  {n:>2}. {mig['label']:<24} would HALT: {halt[-1] if halt else sim.state.get('phase')}")
                return lines
            lines.append(f"  {n:>2}. {mig['label']:<24} {did}")
            lines.append(f"      why: {mig['why']}")
            squash = [m for kind, m in sink if kind == "notify" and m.startswith("SQUASH")]
            for m in squash:
                lines.append(f"      {m}")
            sink[:] = [s for s in sink if not (s[0] == "notify" and s[1].startswith("SQUASH"))]
            mig = migration_item(sim)
        lines.append("  after the migration (simulated: every rescore PASSES, every decision is a clear "
                     "REJECT):")
        while n < limit:
            n += 1
            gates = [f"{x} ({h.get('label')}): {resume_gate(sim, h) or 'resumes'}"
                     for x, h in sorted(experiments(sim.state).items(), key=lambda kv: _xid_key(kv[0]))
                     if h.get("status") == "running" and h.get("kind") == "candidate"
                     and x not in sim.view["experiments"]]
            if gates:
                lines.append("      rank gate: " + " | ".join(gates))
            header = find_inflight(sim)
            if header is not None:
                item = item_from_header(sim, header)
                done = set(header.get("recs") or [])
                left = [str(s["stage"]) for s in item["stages"] if f"{header['xid']}:{s['stage']}" not in done]
                lines.append(f"  {n:>2}. resume {header['xid']} ({header.get('label')}) at "
                             f"{left[0] if left else '(its eval)'} -- find_inflight")
                _dry_decide(sim, header)
                continue
            item = next_item(sim.cands, sim.ledger, sim.state, code=sim.code, view=sim.view,
                             resolution=sim.resolution)
            if item is None:
                lines.append(f"  {n:>2}. done")
                break
            xid = item.get("xid")
            lines.append(f"  {n:>2}. {item['label']:<24} {item['kind']:<10} "
                         + (f"(resumes {xid} at its next stage) " if xid else "") + f"-- {item['why']}")
            if item["kind"] == "await":
                break
            if xid:
                header = experiments(sim.state)[xid]
                header.setdefault("history", []).append({"at": now_iso(), "from": header.get("status"),
                                                        "to": "running", "why": "step-8 dispatch (dry run)"})
                _dry_decide(sim, header)
                continue
            before = set(experiments(sim.state))
            _simulate_apply(sim.cands, sim.ledger, sim.state, item, "REJECT", sim.code)
            for x in set(experiments(sim.state)) - before:
                experiments(sim.state)[x]["eval_sha"] = _cur_eval(sim)
            sim.recompute()
    return lines


def _dry_recipes(ctx: Ctx, ledger: dict):
    """(label, recipe) for the baseline and for every candidate's own recipe, on a synthetic ledger."""
    out = [("baseline", compose_recipe(ledger, ctx.cands, ctx.code,
                                      upto=(candidate_ids(ctx.cands) or [None])[0]))]
    for cid in candidate_ids(ctx.cands):
        try:
            out.append((f"{cid} adopted", _recipe_with(ledger, ctx.cands, ctx.code, cid, set())))
        except (ValueError, KeyError) as exc:
            out.append((f"{cid} adopted", {"chain": "?", "recipe_id": f"ERROR {exc}", "adopted": [],
                                           "eval_carry": [], "stages": [], "conflicts": [str(exc)],
                                           "overlay_by_stage": {}}))
    return out


# ================================================================== the header / REPORT
def ensure_ladder_header(ctx: Ctx, cands: dict) -> None:
    """Write ``REPORT.md``'s header once: the campaign's own contract, in one table."""
    if ctx.c.report.exists():
        return
    rows = ["| rank | id | kind | chain | change | conditional |", "|---|---|---|---|---|---|"]
    base = cands.get("baseline") or {}
    rows.append(f"| 0 | {base.get('id')} | baseline | {base.get('chain')} | the incumbent recipe at "
                f"equal compute | — |")
    for i, cand in enumerate(cands.get("candidates") or [], start=1):
        change = canonical(cand.get("overlay") or cand.get("pool_edit") or cand.get("overlay_by_chain")
                           or {"chain": cand.get("chain")})
        rows.append(f"| {i} | {cand['id']} | {cand.get('kind')} | {cand.get('chain') or 'effective'} | "
                    f"`{change}` | `{cand.get('conditional') or '—'}` |")
    for mstone in cands.get("milestones") or []:
        rows.append(f"| — | {mstone['id']} | milestone | — | promote {mstone.get('promote')} after "
                    f"{mstone.get('after')} | — |")
    head = (f"# Candidate ladder `{ctx.dir.name}`\n\n"
            f"Plant **{ctx.plant}** (`{ph.plant_usd_path(ctx.plant) or 'assets/crab.usda (main)'}`), "
            f"screen seed {ctx.seed_screen}, confirm seed {ctx.seed_confirm}, "
            f"{(cands.get('screen') or {}).get('iters')} iterations per screen.\n\n"
            f"Driver: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/"
            f"run_candidates.py` (overlays on the `KRABBY_OMNI_PHASE` presets; the ledger is "
            f"`ledger.json`, the user's answers go through `ledger_cli.py`).\n\n"
            f"`code_sha` `{ctx.code}` · `candidates.yaml` sha256 "
            f"`{sha256_file(cands['__path__'])}`\n\n" + "\n".join(rows))
    ctx.c.report.write_text(head + "\n\n")
    ctx.c.changelog.write_text(f"# CHANGELOG — {ctx.dir.name}\n")
    ctx.c.changelog_row(f"campaign start: plant {ctx.plant}, seeds {ctx.seed_screen}/{ctx.seed_confirm}, "
                        f"code_sha {ctx.code[:12]}, candidates.yaml "
                        f"{sha256_file(cands['__path__'])[:12]}")


# ================================================================== the main loop
def _mark_interrupt(ctx: Ctx, header: dict, why: str | None) -> None:
    """Record why a window stopped -- and whether the next launch may RESUME it.

    A stop (the ``STOP`` file or a SIGTERM) leaves the header ``running`` with its ``inflight`` block
    intact, because the work is still wanted: the next launch resumes the remainder. A squash or a
    user ``skip`` leaves it ``interrupted``, because the work is not wanted any more.
    """
    reason = str(why or "interrupted")
    header["interrupt"] = reason
    header["status"] = "running" if reason.split(":")[0] in ("stop", "sigterm") else "interrupted"
    ctx.save()


def _park(ctx: Ctx, phase: str, fields: list) -> int:
    ctx.state["phase"] = phase
    ctx.save()
    ctx.notify("AWAIT", phase, fields)
    return 0


def loop(ctx: Ctx) -> int:
    """Work the derived queue until it is empty, a user is needed, or a stop arrives."""
    max_wall = float(ctx.args.max_wall_h) if ctx.args.max_wall_h else None
    while True:
        ingest_overrides(ctx)
        why = stop_requested(ctx)
        if why:
            return _park(ctx, "await_user:stop",
                         [f"stopped ({why})", f"records are complete; relaunch with "
                          f"{HERE / 'launch_candidates.sh'} --campaign-dir {ctx.dir}",
                          f"remove {ctx.dir / lc.STOP_NAME} first ({ctx.cli('unstop')})"
                          if why == "stop" else "the unit was stopped"])
        # Plan D9: the eval-only migration runs BEFORE any resume and before `next_item` (whose
        # baseline-rerun rule would book an 8000-it B0 rerun the moment `ok_screen` goes stale). An
        # interrupted migration eval writes nothing and comes back here, where the stop check parks.
        mig = migration_item(ctx)
        if mig is not None:
            run_migration_item(ctx, mig)
            continue
        header = find_inflight(ctx)
        if header is not None:
            prepare_resume(ctx, header)
            item = item_from_header(ctx, header)
            ctx.c.log(f"resuming {header['xid']} ({header.get('label')}) at "
                      f"{(header.get('inflight') or {}).get('stage')}")
            header = run_experiment(ctx, item)
            _after_experiment(ctx, item, header)
            continue
        if ctx.args.smoke_stage:
            if _smoke_done(ctx):
                # The smoke hook is a DRILL, not a campaign: it must not go on to book a real window.
                return _park(ctx, "await_user:smoke_done",
                             [f"--smoke-stage {ctx.args.smoke_stage} finished",
                              "relaunch without --smoke-stage to run the campaign"])
            item = smoke_item(ctx, ctx.args.smoke_stage)
            header = run_experiment(ctx, item)
            _after_experiment(ctx, item, header)
            continue
        item = next_item(ctx.cands, ctx.ledger, ctx.state, code=ctx.code, view=ctx.view,
                         resolution=ctx.resolution)
        if item is None:
            ctx.state["phase"] = "done"
            ctx.save()
            ctx.notify("DONE", ctx.dir.name, [f"the queue is empty: see {ctx.c.report}",
                                              f"{len(entries(ctx.ledger))} ledger entries"])
            return 0
        if item["kind"] == "await":
            phase = item["label"] if item["label"].startswith("await_user") else f"await_user:{item['candidate']}"
            flagged = (item.get("extra") or {}).get("flag_park")
            if flagged:
                # NOTE(clock-deprecation 2026-09-23): a flag park names its own release verb
                return _park(ctx, phase, [item["why"],
                                          f"release: {ctx.cli(f'ack-flag {flagged} --why ...')} (after the "
                                          "user decides; a `decide --choose run` would squash the decision)"])
            return _park(ctx, phase, [item["why"], f"answer with {ctx.cli('decide <Cid> --choose run')} "
                                                   f"(or --choose skip)"
                                                   f" or {ctx.cli('resume-milestone <Mid> --accept')}"])
        ctx.state["phase"] = "run"
        if item["kind"] == "skip":
            # The SOURCE comes from the item, never from a substring of its reason: a predicate
            # whose rendering happened to contain the word "user" would otherwise be recorded
            # `evidence.user: True` and become immune to derived-view cause (b).
            write_skip_entry(ctx, item["candidate"], item.get("extra", {}).get("source", "predicate"),
                             item["why"], predicate=item.get("extra", {}).get("predicate") or item["why"])
            continue
        if item["kind"] == "pooled":
            entry = entry_by_id(ctx.ledger, item["decision"])
            s2 = ok_screen(ctx.state, ctx.code, (entry or {}).get("recipe_id_if_adopted"),
                           ctx.seed_confirm, ctx.view)
            _resolve_pair(ctx, item, entry, (s2 or {}).get("screen") or {},
                          s2_xid=(s2 or {}).get("xid"), floor_tripped=bool((s2 or {}).get("floor_watch")),
                          header=s2)
            continue
        if item["kind"] == "milestone_i":
            ctx.state["current"] = {"tag": item["label"], "phase": item["candidate"], "iters": 0,
                                   "kind": "bootstrap", "log": str(ctx.dir / "orchestrator.log")}
            ctx.save()
            milestone_i(ctx, item)
            continue
        if item["kind"] == "milestone_close":
            milestone_close(ctx, item)
            continue
        # ---- an experiment
        rp._wait_isaac_clear(lambda: interrupt_reason(ctx, None))
        if stop_requested(ctx):
            continue
        replay = run_replay(ctx, item) if item["kind"] == "candidate" else None
        if (replay or {}).get("status") == "interrupted" or stop_requested(ctx):
            # The plan's named test: "SIGTERM during the replay -> replay: interrupted, NO
            # experiment header, top-of-loop STOP check". Without this the driver went on to write
            # an (immediately interrupted) header for a window it never launched, and the relaunch
            # resumed that header through `item_from_header` -- which never calls `run_replay`, so
            # the candidate's DECISION landed with `evidence.replay = None` although the candidate
            # declares `replay_args`.
            ctx.c.log(f"{item['label']}: the advisory replay was interrupted — no experiment header "
                      f"is written; the top-of-loop stop check follows")
            continue
        header = run_experiment(ctx, item)
        if replay is not None:
            header["replay"] = replay
            ctx.save()
        _after_experiment(ctx, item, header)
        if max_wall is not None and (time.time() - ctx.t0) / 3600.0 >= max_wall:
            return _park(ctx, "await_user:relaunch",
                         [f"--max-wall-h {max_wall} reached after {(time.time() - ctx.t0) / 3600.0:.2f} h",
                          "relaunch to continue: the queue is derived, nothing is lost"])


def _squashed_before_dispatch(ctx: Ctx, header: dict) -> bool:
    """Did this window's base move between the last poll tick and this dispatch? Record it if so.

    ``interrupt_reason`` catches an override that lands WHILE a window trains and the top of
    ``loop`` catches the ones that landed while the driver was parked, but the gap between the last
    poll tick and the dispatch belongs to neither -- and a decision written in that gap is a
    decision about a base recipe that no longer exists. The write still LOOKED right, because
    ``_dispatch_candidate``'s own reconcile then squashed the fresh entry together with the run
    that produced it; what the user read was a DECISION line saying "chosen ADOPT (clear,
    squashed) ... this decision moved the recipe; squashed 1: X11" -- an adoption that never
    applied, announced as one that did.

    So the overrides are ingested once more here and the header is checked against the view: a
    listed xid means no decision at all, the run archived as the record of a window whose base
    moved while it ran. The SQUASH line is the reconcile's own (the ingest above usually writes it
    while archiving this very header); calling it again is the cheap guarantee that the record
    agrees with the view even when the override's apply path did not reconcile.
    """
    ingest_overrides(ctx)
    xid = header.get("xid")
    why = (ctx.view.get("experiments") or {}).get(xid)
    if not why:
        return False
    ctx.c.log(f"{xid}: its base moved while it ran ({why}) — no decision is written")
    squash_and_report(ctx, f"{xid} {header.get('label')} finished on a base that moved")
    ctx.c.changelog_row(f"{xid} {header.get('label')} finished — no decision: its base moved while "
                        f"it ran ({why})")
    ctx.c.report_block(f"### experiment {xid} — {header.get('label')} — squashed — {now_iso()}\n\n"
                       f"- the window finished ({header.get('status')}) and its base moved while it "
                       f"ran: {why}\n"
                       f"- NO decision is written: the screen answers a question about a recipe "
                       f"that no longer exists\n"
                       f"- recipe `{header.get('recipe_id')}` base `{header.get('base_recipe_id')}` "
                       f"seed {header.get('seed')}\n")
    return True


def _after_experiment(ctx: Ctx, item: dict, header: dict) -> None:
    """Dispatch a finished experiment, or record why it produced no decision."""
    status = header.get("status")
    if status in ("ok", "rejected_by_floor") and _squashed_before_dispatch(ctx, header):
        return
    if status == "ok":
        dispatch(ctx, item, header)
        ctx.c.report_block(f"### experiment {header['xid']} — {header.get('label')} — ok — {now_iso()}\n\n"
                           f"- recipe `{header.get('recipe_id')}` base `{header.get('base_recipe_id')}` "
                           f"chain `{header.get('chain')}` seed {header.get('seed')}\n"
                           f"- stages {header.get('recs')}\n"
                           f"- screen: completion {fmt_num((header.get('screen') or {}).get('completion'))}, "
                           f"tripod_dir {fmt_num((header.get('screen') or {}).get('tripod_walking_dir'))}, "
                           f"track tail {fmt_num((header.get('screen') or {}).get('track_ratio_tail'))}\n"
                           f"- replay: {(header.get('replay') or {}).get('status', 'none')}\n\n"
                           f">>> ENTRY {header['xid']}\n")
        return
    if status == "rejected_by_floor":
        _dispatch_floor_reject(ctx, item, header)
        return
    ctx.c.log(f"{header.get('xid')}: {status} ({header.get('interrupt') or header.get('halt')}) — "
              "no decision from this experiment")


def _smoke_done(ctx: Ctx) -> bool:
    return any(h.get("kind") == "smoke" and h.get("status") == "ok"
               for h in all_headers(ctx.state).values())


def smoke_overlay(pairs) -> dict:
    """``--smoke-overlay KEY=VALUE`` (repeatable) -> the smoke window's overlay, validated.

    Each key goes through the same checks a candidate's overlay does (:func:`_check_knob`: the
    whitelist, the value round trip, an ``OVERLAY_DUMP_CHECKS`` entry), so a knob smoke exercises
    the real overlay audit (``overlay_checks``) against the dump its 20-iteration window writes.
    """
    overlay: dict = {}
    for pair in pairs or []:
        key, sep, value = str(pair).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--smoke-overlay wants KEY=VALUE, got {pair!r}")
        overlay[key] = value.strip()
    reasons: list[str] = []
    whitelist, parsers = knob_whitelist(), knob_parsers()
    for key, value in overlay.items():
        _check_knob(key, value, "--smoke-overlay", whitelist, parsers, reasons, set())
    if reasons:
        raise SystemExit("--smoke-overlay refused:\n  - " + "\n  - ".join(reasons))
    return overlay


def smoke_item(ctx: Ctx, spec_str: str) -> dict:
    """``--smoke-stage 1a:20[:suite]``: ONE tiny real window, for the stop drill (Verification 8).

    It runs through the same header / ``inflight`` / resume / interrupt machinery as a real
    experiment and writes no ledger entry, so the drill can stop the unit mid-child and check that
    the header is resumed by the next launch without touching the campaign's decisions. The optional
    third field names the stage's EVAL rows (comma-separated): ``1a:20:suite`` is the per-category
    plan's V6 drill, stopped during the suite's wave 3.
    """
    parts = str(spec_str).split(":")
    base = parts[0]
    iters = parts[1] if len(parts) > 1 else ""
    rows = [r.strip() for r in parts[2].split(",") if r.strip()] if len(parts) > 2 else []
    overlay = smoke_overlay(getattr(ctx.args, "smoke_overlay", None))
    stage = {"stage": base, "base": base, "iters": int(iters or 20), "from_scratch": True,
             "resume_from": "scratch", "overlay": dict(overlay), "eval": rows, "floor": {"kind": "none"},
             "holds": [], "previous_holds": []}
    recipe = {"chain": None, "overlay_by_stage": {base: dict(overlay)}, "stages": [stage], "adopted": [],
              "eval_carry": [], "conflicts": [], "anchor": None, "has_pool": False,
              "recipe_id": f"smoke-{base}-{stage['iters']}"}
    return _item("smoke", cid=f"SMOKE-{base}", seed=ctx.seed_screen, recipe=recipe, base_recipe=recipe,
                 stages=[stage], why="--smoke-stage test hook", label=f"smoke {base} {stage['iters']}it")


# ================================================================== the reference-chain launcher
# NOTE(clock-deprecation 2026-09-23, plan step 4). `--reference-chain <name> [--seed N]` runs ONE
# `references:` chain through the ordinary experiment machinery (header, inflight, resume, stage
# evals with each stage's overlay as `carry_overlay`, the overlay audit, floors as WATCH), records
# its head, and parks. It writes no DECISION, it is exempt from `screen.iters`, and its kind
# (`reference`) never arms the dead_walker early abort -- Z's formation readout must not be cut at
# 1000 its. The main queue never touches it (`find_inflight` skips its headers).
def reference_seed(cands: dict, name: str, seed=None) -> str:
    """The seed a reference runs at: ``--seed`` or the reference's pinned one (both: they agree)."""
    pinned = reference_spec(cands, name)["seed"]
    if seed is not None and pinned is not None and str(seed) != str(pinned):
        raise SystemExit(f"--seed {seed} disagrees with references.{name}.seed {pinned}")
    got = str(seed) if seed is not None else pinned
    if got is None:
        raise SystemExit(f"references.{name} pins no seed: pass --seed N")
    return got


def reference_records(state: dict, name: str) -> list[dict]:
    """``lad.references[name]``: every finished run of that reference, oldest first."""
    return list(((state.get("ladder") or {}).get("references") or {}).get(name) or [])


def reference_head(ctx: Ctx, name: str | None) -> dict | None:
    """The NEWEST finished run of reference ``name`` under the current ``code_sha`` (or a proved alias)
    at its pinned seed (any seed when unpinned) -- what ``resume_from: {reference: <name>}`` forks."""
    if not name:
        return None
    try:
        pinned = reference_spec(ctx.cands, name)["seed"]
    except KeyError:
        return None
    found = None
    for rec in reference_records(ctx.state, name):
        if not same_code(ctx.state, rec.get("code_sha"), ctx.code):
            continue
        if pinned is not None and str(rec.get("seed")) != pinned:
            continue
        found = rec
    return found


def reference_item(ctx: Ctx, name: str, seed: str) -> dict:
    recipe = reference_recipe(ctx.cands, name, ctx.code)
    return _item("reference", cid=name, seed=seed, recipe=recipe, base_recipe=recipe,
                 stages=_stages_for(ctx.cands, recipe["chain"]),
                 why=f"--reference-chain {name}: a reference run (judged by hand, no decision)",
                 label=f"ref:{name} s{seed}")


def _reference_inflight(ctx: Ctx, name: str, seed: str) -> dict | None:
    """This reference's own window left running (or at a resumable infra halt), if any."""
    for xid, header in list((ctx.lad.get("experiments") or {}).items()):
        if header.get("kind") != "reference" or header.get("candidate") != name:
            continue
        if str(header.get("seed")) != str(seed) or xid in ctx.view["experiments"]:
            continue
        if header.get("status") == "running" or (header.get("status") in RESUMABLE_HALTS
                                                 and header.get("inflight")):
            return header
    return None


def record_reference(ctx: Ctx, name: str, header: dict) -> dict:
    """Append the finished run to ``lad.references[name]`` (its stage heads are in heads.json already:
    every stage rec carries its checkpoint) and relay it."""
    bucket = (ctx.state.get("keys") or {}).get(f"seed{header.get('seed')}") or {}
    stage_ckpts = {}
    for key in header.get("recs") or []:
        rec = bucket.get(key) or ctx.c.recs.get(key) or {}
        stage_ckpts[key.split(":", 1)[-1]] = rec.get("ckpt")
    row = {"xid": header.get("xid"), "seed": str(header.get("seed")), "ckpt": header.get("ckpt"),
           "stage_ckpts": stage_ckpts, "recipe_id": header.get("recipe_id"),
           "chain": header.get("chain"), "code_sha": ctx.code, "eval_sha": ctx.eval_sha,
           "floor_watch": list(header.get("floor_watch") or []), "at": now_iso()}
    ctx.lad.setdefault("references", {}).setdefault(name, []).append(row)
    ctx.save()
    ctx.notify("REFERENCE", f"{name} s{row['seed']}",
               [f"{row['xid']} on chain {row['chain']} (recipe {row['recipe_id']}) finished -- judged by "
                "hand, no decision", f"head {row['ckpt']}",
               *[f"{s}: {c}" for s, c in stage_ckpts.items()],
               *([f"FLOOR WATCH: {w}" for w in row["floor_watch"]] or ["floors held"])])
    ctx.c.changelog_row(f"reference {name} s{row['seed']} {row['xid']} finished (head {row['ckpt']})")
    return row


def reference_loop(ctx: Ctx, name: str, seed: str, main_phase=...) -> int:
    """Run (or resume) ONE reference chain, record its head, and park. Never touches the queue.

    Nor the main ladder's recorded ``state.phase`` (e.g. a flag park ``ledger_cli pending`` shows):
    the reference's own park goes to the notify line only, and ``main_phase`` (the phase recorded
    before this launch; :func:`main` passes it, else the current one) is restored on every exit.
    """
    if main_phase is ...:
        main_phase = ctx.state.get("phase")

    def _ref_park(ctx_: Ctx, phase: str, fields: list) -> int:
        if main_phase is None:
            ctx_.state.pop("phase", None)
        else:
            ctx_.state["phase"] = main_phase
        ctx_.save()
        ctx_.notify("AWAIT", phase, fields)
        return 0

    ingest_overrides(ctx)
    why = stop_requested(ctx)
    if why:
        return _ref_park(ctx, "await_user:stop", [f"stopped ({why}) before reference {name} s{seed}"])
    header = _reference_inflight(ctx, name, seed)
    if header is not None:
        prepare_resume(ctx, header)
        item = item_from_header(ctx, header)
        ctx.c.log(f"resuming reference {header['xid']} ({header.get('label')}) at "
                  f"{(header.get('inflight') or {}).get('stage')}")
    else:
        done = [r for r in reference_records(ctx.state, name)
                if str(r.get("seed")) == str(seed) and same_code(ctx.state, r.get("code_sha"), ctx.code)]
        if done:
            return _ref_park(ctx, f"await_user:reference_done:{name}",
                         [f"reference {name} s{seed} already finished under this code_sha "
                          f"({done[-1].get('xid')}, head {done[-1].get('ckpt')}) -- nothing to run"])
        item = reference_item(ctx, name, seed)
    ctx.state["phase"] = "run"
    rp._wait_isaac_clear(lambda: interrupt_reason(ctx, None))
    header = run_experiment(ctx, item)
    if header.get("status") != "ok":
        return _ref_park(ctx, "await_user:stop" if header.get("status") == "running"
                     else f"await_user:reference_{header.get('status')}",
                     [f"reference {name} s{seed} {header.get('xid')}: {header.get('status')} "
                      f"({header.get('interrupt') or header.get('halt') or ''})",
                      f"relaunch with --reference-chain {name} --seed {seed} to resume it"])
    row = record_reference(ctx, name, header)
    return _ref_park(ctx, f"await_user:reference_done:{name}",
                 [f"reference {name} s{seed} recorded ({row['xid']}, head {row['ckpt']})",
                  "judge it by hand; forks name it as resume_from: {reference: " + name + "}"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--candidates", default=None, help="default: <campaign-dir>/candidates.yaml")
    ap.add_argument("--unit", default=None, help="the systemd unit the launcher started (recorded)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue, every stage env, the validator verdicts, D00 and "
                         "code_sha; writes nothing but the campaign dir's own logs/ directory (no "
                         "state.json, no ledger, no GPU)")
    ap.add_argument("--max-wall-h", type=float, default=None,
                    help="park at await_user:relaunch after this many hours (no default cap)")
    ap.add_argument("--accept-code-change", default=None, metavar="WHY",
                    help="accept a changed code_sha: old screens stop being ok and B<k> reruns are queued")
    ap.add_argument("--accept-eval-change", default=None, metavar="WHY",
                    help="accept a changed eval_sha (the harness, the omni manifest, run_phases.py's "
                         "eval side, eval_categories.py): old screens are re-EVALUATED, never "
                         "retrained. A pre-suite campaign (a recorded code_sha, no eval_sha) needs it "
                         "once; the narrowed code_sha is then proved an alias of the recorded one "
                         "(the training tree did not move) or the launch falls through to the "
                         "code_sha refusal")
    ap.add_argument("--accept-candidates-change", default=None, metavar="WHY",
                    help="accept a changed candidates.yaml sha256")
    ap.add_argument("--accept-resolution", nargs=2, action="append", default=None,
                    metavar=("KIND", "WHY"),
                    help="accept a D00s or D00bs whose acceptance FAILED at the current eval_sha "
                         "(plan D9 item 4): recorded in lad.resolution_acceptances; the next "
                         "computation of that kind then writes the entry with acceptance.ok = false "
                         "and cites the acceptance. Repeatable.")
    ap.add_argument("--smoke-stage", default=None, metavar="BASE:ITERS[:ROWS]",
                    help="test hook: run ONE tiny window (e.g. 1a:20, or 1a:20:suite with the "
                         "comma-separated eval rows as a third field) through the full experiment "
                         "machinery -- header, inflight, resume, interrupt, records -- and then park "
                         "at await_user:smoke_done without touching the ledger or booking a real "
                         "window. This is the stop drill's launch (Verification 8 / the per-category "
                         "plan's V6). The migration never runs under it.")
    ap.add_argument("--smoke-overlay", action="append", default=None, metavar="KEY=VALUE",
                    help="with --smoke-stage only: an overlay key for the smoke window (repeatable), "
                         "validated like a candidate overlay and audited against the window's "
                         "params/env.yaml (params/agent.yaml for the runner-side knobs) -- the "
                         "per-knob smoke of the motion-quality plan's step 2")
    ap.add_argument("--q-spread-check", nargs="+", default=None, metavar="SUITE_STAMP",
                    help="PRE-LAUNCH, CPU only, writes nothing: the zero-spread check of every scored "
                         "member (resolution.member_spread_check). The first stamp (a suite run dir, "
                         "or an eval_dir root -> its newest fresh stamp) is the control -- the "
                         "incumbent's re-eval at the current eval_sha; each further one forms a "
                         "cross-head pair with it, drawn exactly as D00s draws it. Prints every member "
                         "with no finite positive R and the decision_rule.per_category.demoted lines "
                         "that demote it to WATCH (the campaign's answer, never --accept-resolution)")
    ap.add_argument("--q-spread-heads", default=None, metavar="CSV",
                    help="with --q-spread-check: head labels for the stamps, in order")
    ap.add_argument("--reference-chain", default=None, metavar="NAME",
                    help="run ONE `references:` chain (the reference NAME; its chain defaults to that "
                         "name) with stage evals, record its head in heads.json / lad.references and "
                         "park -- no DECISION, exempt from screen.iters, no dead_walker early abort; "
                         "a stage may fork it as resume_from: {reference: NAME}")
    ap.add_argument("--seed", default=None, metavar="N",
                    help="with --reference-chain: the seed (required unless the reference pins one)")
    return ap


def q_spread_report(cands: dict, run_dirs, heads=None) -> tuple[int, list[str]]:
    """``--q-spread-check``: ``(exit code, lines)`` -- 0 when every scored member resolves, 3 when
    some member has zero spread (the lines then carry the yaml to demote it)."""
    rule = cands.get("decision_rule") or {}
    res = resolution_tool()
    chain = (cands.get("baseline") or {}).get("chain")
    stages = chain_stages(cands, chain) if chain else []
    dat = _ordered_categories(_stage_decisive(stages[-1])) if stages else None
    try:
        out = res.member_spread_check(list(run_dirs), rule, decisive_at=dat or None, heads=heads)
    except res.ResolutionError as exc:
        return 2, [f"q-spread-check refused: {exc}"]
    lines = [f"# member spread check (B={out['B']}, rng {out['rng_seed']}, decisive_at "
             f"{', '.join(out['decisive_at'])})"]
    for r in out["rows"]:
        lines.append(f"  row {r['head']}: {r['run_dir']} (n={r['n']})")
    for key, m in sorted(out["members"].items()):
        same = ", ".join(f"{h} {fmt_num(v, 4)}" for h, v in m["same_run"].items())
        cross = ", ".join(f"{n} {fmt_num(v, 4)}" for n, v in m["cross"].items()) or "-"
        mark = "ZERO " if m["zero"] else "ok   "
        sense = " (lower is better)" if m.get("sign", 1) < 0 else ""
        lines.append(f"  {mark}{key}{sense} [{m['block']}] same-run R {same}; cross-head R {cross}")
    if out["unresolved"]:
        lines.append(f"  unresolved on this row (no per-episode form): {out['unresolved']}")
    if out.get("absent"):
        lines.append(f"{len(out['absent'])} member(s) carry NO value on a stamp (it predates the metric, "
                     f"or the key is N/A there) -- re-evaluate under the current eval_sha; these are "
                     f"never demoted: {out['absent']}")
    if not out["zero"]:
        lines.append("no scored member has zero spread -- nothing to demote"
                     + ("" if not out.get("absent") else " (but see the absent members above)"))
        return (3 if out.get("absent") else 0), lines
    lines.append(f"{len(out['zero'])} member(s) with zero spread: {out['zero']}")
    lines.append("demote them before launch -- add under decision_rule.per_category (never "
                 "--accept-resolution):")
    lines.append("    demoted:")
    for c, keys in out["demote"].items():
        lines.append(f"      {c}:")
        for k, why in keys.items():
            lines.append(f"        {k}: {json.dumps(why)}")
    return 3, lines


def _eval_change_gate(ctx: Ctx, args, old_code: str | None) -> tuple[dict | None, dict | None]:
    """Plan D8's eval-side relaunch gate -- run BEFORE the ``code_sha`` comparison. ``(alias, proof)``.

    A campaign with a recorded ``code_sha`` whose ``lad.eval_sha`` is absent (every pre-suite
    campaign) or differs from ``eval_sha()`` has CHANGED its eval. Without ``--accept-eval-change`` a
    real launch is refused by a message naming that flag -- never ``--accept-code-change``, whose
    answer (a ``B<k>`` retrain) is the wrong one for an eval-only change. With it, and when the
    ``code_sha`` VALUE also moved (the pathspec narrowing moves it), :func:`prove_code_alias` must
    show the training tree did not move: a proof returns the alias record ``{sha, proved_at, why,
    at}`` (the caller appends it to ``lad.code_sha_aliases`` and the code refusal and its RELAUNCH
    "B<k> rerun is queued" text are bypassed); a FAILED proof returns ``(None, proof)`` and the
    launch falls through to the ``code_sha`` refusal. A dry run never refuses: it only computes the
    proof so ``dry_run`` can print it.
    """
    old_eval = ctx.lad.get("eval_sha")
    if not old_code or old_eval == ctx.eval_sha:
        return None, None
    if not args.accept_eval_change:
        if args.dry_run:
            return None, (prove_code_alias(old_code, current=ctx.code) if old_code != ctx.code else None)
        raise SystemExit(
            f"eval_sha changed: recorded {old_eval or 'none (this campaign predates the per-category suite)'}"
            f", now {ctx.eval_sha}.\nEvery recorded screen was measured under the old evaluation. "
            "Relaunch with --accept-eval-change \"<why>\" to accept it: old screens are "
            "RE-EVALUATED (rescreen / reeval / rescore), never retrained"
            + (f"; the changed code_sha ({old_code[:12]} -> {ctx.code[:12]}) is then proved an alias "
               "of the recorded one or refused" if old_code != ctx.code else "") + ".")
    if old_code == ctx.code:
        return None, None
    proof = prove_code_alias(old_code, current=ctx.code)
    if not proof["ok"]:
        ctx.c.log(f"--accept-eval-change: the alias proof FAILED ({proof['why']}); the code_sha "
                  "comparison decides")
        return None, proof
    alias = {"sha": old_code, "proved_at": proof["proved_at"], "narrowed": proof["narrowed_at"],
             "why": args.accept_eval_change, "at": now_iso()}
    return alias, proof


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for kind, _why in args.accept_resolution or []:
        if kind not in RESOLUTION_KINDS:
            raise SystemExit(f"--accept-resolution names {kind!r}; it accepts one of "
                             f"{list(RESOLUTION_KINDS)} (the migration's two acceptance-checked records)")
    cdir = Path(args.campaign_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cands_path = Path(args.candidates) if args.candidates else (cdir / "candidates.yaml")
    if not cands_path.exists():
        raise SystemExit(f"no candidates.yaml at {cands_path}")
    cands = load_candidates(cands_path)
    if args.q_spread_check:
        # CPU only and read-only: no validation gate (the check runs BEFORE the yaml is final --
        # its answer is what gets written into it), no Ctx (no logs/ dir), no state.
        heads = [h.strip() for h in args.q_spread_heads.split(",")] if args.q_spread_heads else None
        code, lines = q_spread_report(cands, args.q_spread_check, heads)
        print("\n".join(lines))
        return code
    verdicts = validate_candidates(cands, strict=not args.dry_run)
    ref_seed = None
    if args.reference_chain:
        # NOTE(clock-deprecation 2026-09-23): refused before anything is written, like every refusal
        if args.reference_chain not in (cands.get("references") or {}):
            raise SystemExit(f"--reference-chain {args.reference_chain!r} is not a references: key "
                             f"(known: {sorted(cands.get('references') or {})})")
        ref_seed = reference_seed(cands, args.reference_chain, args.seed)
    elif args.seed is not None:
        raise SystemExit("--seed only goes with --reference-chain (the ladder's own seeds are "
                         "candidates.yaml's)")
    # Before `Ctx`, which builds the `rp.Campaign` and so mkdirs `<campaign-dir>/logs/`: a REFUSED
    # launch must leave the tree exactly as it found it.
    dirty = dirty_paths()
    if dirty and not args.dry_run:
        raise SystemExit("the code- or eval-bearing tree is dirty; every screen would be "
                         "unreproducible:\n  - "
                         + "\n  - ".join(dirty) + "\ncommit or stash, then relaunch (the ladder's own "
                         "tools and records are outside code_sha by design)")
    ctx = Ctx(cdir, cands, args)
    ctx.code = code_sha()
    ctx.eval_sha = eval_sha()
    replay_script = (cands.get("replay") or {}).get("script")
    ctx.tools = tools_sha(repo_path(replay_script) if replay_script else None)
    sha = sha256_file(cands_path)
    old_code = ctx.lad.get("code_sha")
    alias, proof = _eval_change_gate(ctx, args, old_code)
    if old_code and old_code != ctx.code and alias is None:
        if not args.accept_code_change and not args.dry_run:
            raise SystemExit(f"code_sha changed: recorded {old_code}, now {ctx.code}.\nEvery recorded "
                             "screen was measured under the old tree. Relaunch with "
                             "--accept-code-change \"<why>\" to accept it: old screens stay as records, "
                             "nothing is squashed, and next_item inserts a recipe-baseline rerun.")
        if not args.dry_run:
            ctx.notify("RELAUNCH", ctx.dir.name,
                       [f"code_sha {old_code} -> {ctx.code}", f"accepted: {args.accept_code_change}",
                        "old screens are no longer 'ok'; a B<k> rerun is queued and later uncertain "
                        "candidates re-run their base twins"])
    old_sha = ctx.lad.get("candidates_sha256")
    if old_sha and old_sha != sha and not args.accept_candidates_change and not args.dry_run:
        raise SystemExit(f"candidates.yaml changed: recorded {old_sha}, now {sha}.\nEvery recorded "
                         "decision names the recipe this file composes. Relaunch with "
                         "--accept-candidates-change \"<why>\" to accept it.")
    if args.dry_run:
        install_sigterm(ctx)
        ctx.eval_proof = proof
        return dry_run(ctx, verdicts)

    # Only now, past every refusal, is the eval change RECORDED (a refused launch leaves the tree
    # exactly as it found it).
    old_eval = ctx.lad.get("eval_sha")
    if alias is not None:
        ctx.lad.setdefault("code_sha_aliases", []).append(alias)
        ctx.c.log(f"--accept-eval-change: {proof['why']} -- {alias['sha'][:12]} recorded as an alias "
                  f"of {ctx.code[:12]}")
    if old_eval != ctx.eval_sha:
        ctx.lad.setdefault("eval_sha_history", []).append(
            {"from": old_eval, "to": ctx.eval_sha, "why": args.accept_eval_change, "at": now_iso(),
             "alias": (alias or {}).get("sha")})
        if old_code:
            ctx.c.changelog_row(
                f"eval_sha {str(old_eval or 'none')[:12]} -> {ctx.eval_sha[:12]} accepted "
                f"({args.accept_eval_change})" + (f"; code_sha {alias['sha'][:12]} proved an alias of "
                                                  f"{ctx.code[:12]} at {str(alias['proved_at'])[:12]}"
                                                  if alias else ""))
    ctx.lad["eval_sha"] = ctx.eval_sha
    ctx.lad["code_sha"] = ctx.code
    ctx.lad["candidates_sha256"] = sha
    for kind, why in args.accept_resolution or []:
        # plan D9 item 4: an acceptance is for ONE kind at ONE eval_sha; the next computation of that
        # kind writes its entry with acceptance.ok = false and cites this record
        ctx.lad.setdefault("resolution_acceptances", []).append(
            {"kind": kind, "eval_sha": ctx.eval_sha, "why": why, "at": now_iso()})
        ctx.c.changelog_row(f"--accept-resolution {kind} at eval_sha {ctx.eval_sha[:12]}: {why}")
    if args.unit:
        ctx.lad["unit"] = args.unit
    main_phase = ctx.state.get("phase")          # restored by a --reference-chain launch's park
    ctx.state["phase"] = "run"
    ensure_ladder_header(ctx, cands)
    install_sigterm(ctx)
    ctx.recompute()
    # SQUASH (6): "the squash set is a derived view recomputed from the ledger on every launch". The
    # recompute above derives it; this makes the RECORD agree with it once, before any queue is read
    # -- which is what a crash between the ledger write and the state save leaves undone. The
    # override itself is already in ledger.json, so the view is right and `find_inflight`/`ok_screen`
    # honour it; without this pass the headers it squashes stay under `state.ladder.experiments` and
    # the SQUASH line, the REPORT block and the CHANGELOG row for that override are never written at
    # all. A launch with nothing to reconcile moves nothing and writes nothing.
    squash_and_report(ctx, f"launch reconcile ({ctx.dir.name})")
    ctx.save()
    ctx.c.log(f"ladder start: code_sha {ctx.code[:12]} tools_sha {ctx.tools[:12]} unit {args.unit} "
              f"candidates {cands_path}")
    rp._wait_isaac_clear(lambda: interrupt_reason(ctx, None))
    # Plan D9 item 3: BOTH startup branches (the smoke one included) LOAD the resolution; nothing is
    # computed here -- under rule v2 D00s is a migration queue item and `ctx.resolution` stays None
    # until it lands (every v2 decision seam halts `resolution_missing` meanwhile).
    load_resolution(ctx)
    if args.reference_chain:
        return reference_loop(ctx, args.reference_chain, ref_seed, main_phase=main_phase)
    return loop(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
