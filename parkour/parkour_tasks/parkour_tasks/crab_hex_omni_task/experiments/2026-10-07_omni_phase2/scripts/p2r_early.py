#!/usr/bin/env python3
"""Omni phase 2, stage p2r -- the Z-arm early checks (P2R_STATUE_PLAN.md step 4 "Early checks", notify-only). CPU
only, no Isaac, no CUDA, no training: reads a Z run's own TensorBoard scalars and scores its checkpoints on the
step-1 non-stand bank. It never kills a run -- every flag is a NOTIFY (exit 5); the run always continues
(P2R_STATUE_PLAN.md step 4: "On a trip, notify and ask; the run continues").

THE FOUR CHECKS, at both ``model_20500`` and ``model_21000`` (the +506 / +1006 checkpoints of a fork of
``model_19994`` -- the SAME absolute checkpoint numbers C0 and ST2 were read at, so every comparison below is a
matched-iteration pair, never a cross-iteration one):
  walking drift   D(Z) on the step-1 non-stand bank (:mod:`p2r_drift`, the anchor path) must be <= 0.5 x D(C0) at
                  the SAME checkpoint -- otherwise the walking anchor is not holding (P2R_PLAN.md's own stop rule,
                  reused here as a per-checkpoint flag rather than a one-shot stop, since Z trains past it).
  drift direction (reported, not gated) the cosine of Z's mean delta-mu (mu_Z - mu_S1, averaged over the bank) vs
                  ST2's mean delta-mu at the SAME checkpoint: how much of Z's drift points the same way ST2's
                  harmful drift did, not just how big it is.
  formation       rolling-20 Loss/stand_prior AT the fork's +200 iteration must be <= 0.1 x its FIRST-UPDATE value
                  (the exact value at the fork's first logged iteration) -- otherwise the statue never formed.
  residual        rolling-50 Loss/stand_prior (and the PER-STAND-ROW value, = value / Loss/prior_rows at the same
                  point) at +500 and +1000 vs its OWN rolling-50 MINIMUM over the +200..+500 window -- flagged if
                  either point rises more than 2x above that minimum (RL pulling the stand back toward marching).
Reported beside every checkpoint (never gated): PPO/kl_mean, PPO/lr_applied (the applied LR), PPO/grad_norm,
Loss/value_function, and -- logged every 50 iterations only, so read at the nearest available point -- the
gradient-norm split Loss/grad_norm_statue / _anchor / _rest.

Z2 MODE (``--mode z2``; NOTE(p2r Z2, 2026-10-10), P2R_Z2_PLAN.md step 4 "Early checks", notify-only as for Z1).
Z2's statue target is R(obs), not 0, so the prior does not decay to ~0 the way Z1's did:
  walking drift   unchanged (D(Z2) <= 0.5 x D(C0) at the same checkpoint, cosine vs ST2 reported).
  formation       rolling-20 Loss/stand_prior at +500 must be <= 0.25 x its first-update value (Z1's 0.1x-at-+200
                  line assumed a constant target).
  stand fidelity  :mod:`p2r_z2_fidelity` on the restance probe's ``p2r_stand_bank.npz`` (``--stand-bank``), the eval
                  path (own estimator, ``hist_encoding=True``): NOTIFY when the max over legs of the mean signed
                  HOLD-branch cam residual |mu_cam - R_cam| exceeds 0.015 (half the trim's +-0.03); a bank with no
                  HOLD-branch row cannot be read and NOTIFIES too.
  residual        computed and REPORTED only (not gated: Z2's plan lists drift / formation / fidelity; a varying target
                  makes the 2x-over-minimum line a Z1-only statistic).

Exit codes: 0 = every checkpoint read clean, 5 = at least one flag fired (NOTIFY, never a kill), 2 = not enough
telemetry yet for a checkpoint that already has a model file (retry), 1 = the run dir / bank is unreadable.

CLI:
    p2r_early.py --run-dir RUN_DIR [--bank BANK.npz] [--agent-yaml PATH] [--s1-ckpt PATH] [--c0-run DIR]
                 [--st2-run DIR] [--json OUT]
        the one-shot read: both checkpoints that already exist on disk (a missing one is reported, not an error).
    p2r_early.py --watch RUN_DIR [--poll-s N] [--timeout-s N] [... same options]
        a sentinel loop (for the orchestrator): polls every ``--poll-s`` (default 120) until BOTH model_20500.pt
        and model_21000.pt exist (or ``--timeout-s``, default unbounded), then runs the one-shot read once.
    p2r_early.py --mode z2 --run-dir RUN_DIR --bank BANK.npz --stand-bank p2r_stand_bank.npz [... same options]
        the Z2 read (see Z2 MODE above); ``--watch`` works the same way.
    p2r_early.py --self-test
        synthetic TensorBoard series (the pure formation / residual logic) + a tiny synthetic checkpoint pair (the
        drift logic, reusing p2r_graft.py's toy-checkpoint builder). Exit 0 / 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p2_live as live   # noqa: E402  (read_tb / window / point_at / first_last -- same directory, CPU-only)
import p2r_drift as drift  # noqa: E402
import p2r_z2_fidelity as z2fid  # noqa: E402  (NOTE(p2r Z2, 2026-10-10): the stand-fidelity scorer)

R = Path("/home/nickmagus/krabby/krabby-research")
#: the fixed references (P2R_STATUE_PLAN.md step 4): the graft's source S1 head, and the two matched-iteration
#: comparators C0 (ordinary continuation) and ST2 (the ST2's-term stand specialist, step 1's drift premise)
S1_CKPT_DEFAULT = R / ("parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/"
                      "model_19994.pt")
C0_RUN_DEFAULT = R / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-08_03-41-54_seed18_X12_p2b_A15pB"
ST2_RUN_DEFAULT = R / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-09_01-18-22_seed18_X15_p2s_A15pB"
#: the two checkpoints every comparison is matched at (P2R_PLAN.md's own +506 / +1006 points, named by the
#: checkpoint file itself since every run here forks the SAME model_19994)
CHECKPOINTS = ("model_20500.pt", "model_21000.pt")
D_RATIO_LINE = 0.5                 # flag if D(Z) > D_RATIO_LINE * D(C0)
FORMATION_FRAC = 0.1                # rolling-20 Loss/stand_prior at +200 <= FORMATION_FRAC x its first-update value
FORMATION_AT, FORMATION_ROLL = 200, 20
RESIDUAL_WINDOW, RESIDUAL_ROLL = (200, 500), 50
RESIDUAL_AT = (500, 1000)
RESIDUAL_RATIO_LINE = 2.0
REPORT_TAGS = ("PPO/kl_mean", "PPO/lr_applied", "PPO/grad_norm", "Loss/value_function")
GRAD_SPLIT_TAGS = ("Loss/grad_norm_statue", "Loss/grad_norm_anchor", "Loss/grad_norm_rest")
RC_OK, RC_WAIT, RC_NOTIFY, RC_ERROR = 0, 2, 5, 1
#: NOTE(p2r Z2, 2026-10-10): Z2's formation line (P2R_Z2_PLAN step 4) -- rolling-20 at +500 <= 0.25 x first update
Z2_FORMATION_FRAC, Z2_FORMATION_AT = 0.25, 500
MODES = ("z1", "z2")


# ================================================================================================ pure logic
def rolling_mean_at(series, step: int, window: int):
    """The mean of ``series`` over the trailing ``window`` points ENDING AT (and including) ``step`` -- a
    rolling-``window`` read "at" a fork-relative offset, not a window centred on it."""
    return live.window(series, step - window + 1, step)


def rolling_min_over(series, lo: int, hi: int, window: int):
    """The MINIMUM rolling-``window`` mean of ``series``, evaluated at every logged step in ``[lo, hi]`` -- the
    "+200..+500 minimum" the residual check compares against."""
    steps = sorted({s for s, _ in series or [] if lo <= s <= hi})
    vals = [v for v in (rolling_mean_at(series, s, window) for s in steps) if v is not None]
    return min(vals) if vals else None


def judge_formation(series, first: int, at: int = FORMATION_AT, frac: float = FORMATION_FRAC) -> dict:
    """P2R_STATUE_PLAN.md step 4 "formation" (pure). ``at`` / ``frac`` default to Z1's +200 / 0.1x; Z2 passes
    :data:`Z2_FORMATION_AT` / :data:`Z2_FORMATION_FRAC`."""
    first_val = live.point_at(series, first)
    r20 = rolling_mean_at(series, first + at, FORMATION_ROLL)
    out = {"first_update_value": first_val, f"rolling{FORMATION_ROLL}_at_{at}": r20}
    if first_val is None or r20 is None:
        return dict(out, rc=RC_WAIT, why="missing Loss/stand_prior at the fork start or +%d" % at)
    line = frac * first_val
    out["line"] = line
    flagged = r20 > line
    return dict(out, rc=RC_NOTIFY if flagged else RC_OK,
               why=(f"rolling-{FORMATION_ROLL} Loss/stand_prior at +{at} ({r20:.4g}) > "
                    f"{frac:g} x its first-update value ({first_val:.4g})") if flagged else "formed")


def judge_residual(series, prior_rows_series, first: int) -> dict:
    """P2R_STATUE_PLAN.md step 4 "residual" (pure): rolling-50 Loss/stand_prior (and per-stand-row) at +500 / +1000
    vs its own rolling-50 minimum over +200..+500."""
    lo, hi = RESIDUAL_WINDOW
    base = rolling_min_over(series, first + lo, first + hi, RESIDUAL_ROLL)
    out = {"baseline_min_%d_%d" % (lo, hi): base, "at": {}}
    if base is None:
        return dict(out, rc=RC_WAIT, why=f"no rolling-{RESIDUAL_ROLL} Loss/stand_prior point in +{lo}..+{hi}")
    flags = []
    for at in RESIDUAL_AT:
        v = rolling_mean_at(series, first + at, RESIDUAL_ROLL)
        pr = rolling_mean_at(prior_rows_series, first + at, RESIDUAL_ROLL) if prior_rows_series else None
        per_row = (v / pr) if (v is not None and pr not in (None, 0)) else None
        out["at"][at] = {"rolling%d" % RESIDUAL_ROLL: v, "prior_rows": pr, "per_stand_row": per_row}
        if v is None:
            continue
        if v > RESIDUAL_RATIO_LINE * base:
            flags.append(f"+{at}: rolling-{RESIDUAL_ROLL} Loss/stand_prior {v:.4g} > {RESIDUAL_RATIO_LINE:g} x "
                         f"the +{lo}..+{hi} minimum {base:.4g}")
    return dict(out, rc=RC_NOTIFY if flags else RC_OK, why=flags or "within 2x of the +200..+500 minimum")


def mean_delta_mu(obs: torch.Tensor, sd_a: dict, sd_ref: dict, w: dict) -> torch.Tensor:
    """mean(mu_a - mu_ref) over every bank row, on the anchor path (:func:`p2r_drift.actor_mu`) -- one
    ``(num_actions,)`` DIRECTION vector per checkpoint, never per-row."""
    with torch.no_grad():
        return (drift.actor_mu(obs, sd_a, w) - drift.actor_mu(obs, sd_ref, w)).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denom = float(a.norm() * b.norm())
    return float((a @ b) / denom) if denom > 0 else None


def judge_drift(d_z: float | None, d_c0: float | None) -> dict:
    """P2R_STATUE_PLAN.md step 4 "walking drift" (pure): D(Z) <= 0.5 x D(C0) at the same checkpoint."""
    if d_z is None or d_c0 is None:
        return {"rc": RC_WAIT, "why": "D(Z) or D(C0) not computed", "D_Z": d_z, "D_C0": d_c0}
    line = D_RATIO_LINE * d_c0
    flagged = d_z > line
    return {"rc": RC_NOTIFY if flagged else RC_OK, "D_Z": d_z, "D_C0": d_c0, "line": line,
           "why": (f"D(Z) {d_z:.4g} > {D_RATIO_LINE:g} x D(C0) {d_c0:.4g}" if flagged
                   else "the walking anchor is holding")}


def judge_fidelity(fid: dict | None) -> dict:
    """NOTE(p2r Z2, 2026-10-10): :func:`p2r_z2_fidelity.score`'s result -> this module's rc (pure). Over the line ->
    NOTIFY; a bank with no HOLD-branch row cannot be judged -> NOTIFY (surfaced, never silently passed)."""
    if fid is None:
        return {"rc": RC_WAIT, "why": "stand fidelity not computed"}
    out = {k: fid.get(k) for k in ("hold_cam_residual_signed_by_leg", "hold_cam_residual_max_abs", "line", "why")}
    out["by_branch_cam_abs_mean"] = {b: (v.get("cam_abs") or {}).get("mean") for b, v in (fid.get("by_branch") or {}).items()}
    out["prior_per_row"] = fid.get("prior_per_row")
    out["rc"] = RC_OK if fid.get("rc") == z2fid.RC_OK else RC_NOTIFY
    return out


def judge_checkpoint(ckpt_name: str, d_z: float | None, d_c0: float | None, cos_zst2: float | None,
                     ser: dict, first: int, mode: str = "z1", fidelity: dict | None = None) -> dict:
    """One checkpoint's full read (pure): drift + formation + residual, plus the reported-only tags. ``ser`` is
    the Z run's own TensorBoard series (:func:`p2_live.read_tb`); ``first`` its fork's first logged iteration.
    ``mode="z2"``: Z2's formation line, the stand-fidelity judge on ``fidelity`` (:func:`p2r_z2_fidelity.score`'s
    result), residual reported only."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    drift_j = judge_drift(d_z, d_c0)
    if mode == "z2":
        form_j = judge_formation(ser.get("Loss/stand_prior"), first, at=Z2_FORMATION_AT, frac=Z2_FORMATION_FRAC)
    else:
        form_j = judge_formation(ser.get("Loss/stand_prior"), first)
    resid_j = judge_residual(ser.get("Loss/stand_prior"), ser.get("Loss/prior_rows"), first)
    reported = {}
    at = int(ckpt_name.split("_")[1].split(".")[0]) - first
    for t in REPORT_TAGS:
        reported[t] = rolling_mean_at(ser.get(t), first + at, 20)
    for t in GRAD_SPLIT_TAGS:
        # logged every 50 its: the nearest point at or before fork+at
        pts = [(s, v) for s, v in ser.get(t) or [] if s <= first + at]
        reported[t] = pts[-1][1] if pts else None
    if mode == "z2":
        fid_j = judge_fidelity(fidelity)
        rc = max(j["rc"] for j in (drift_j, form_j, fid_j))
        return {"checkpoint": ckpt_name, "mode": mode, "cosine_delta_mu_vs_st2": cos_zst2, "drift": drift_j,
                "formation": form_j, "stand_fidelity": fid_j, "residual_reported": resid_j, "reported": reported,
                "rc": rc}
    rc = max(j["rc"] for j in (drift_j, form_j, resid_j))
    return {"checkpoint": ckpt_name, "cosine_delta_mu_vs_st2": cos_zst2, "drift": drift_j, "formation": form_j,
           "residual": resid_j, "reported": reported, "rc": rc}


# ================================================================================================ IO drivers
def score_checkpoint(ckpt: Path, s1_ckpt: Path, c0_run: Path, st2_run: Path, bank_path: Path,
                     agent_yaml: Path | None = None) -> dict:
    """D(Z) / D(C0) and the cosine of Z's / ST2's mean delta-mu, all at ``ckpt``'s own name resolved under the
    OTHER two runs too (IO, CPU)."""
    name = ckpt.name
    c0_ckpt, st2_ckpt = c0_run / name, st2_run / name
    aw = agent_yaml or (s1_ckpt.parent / "params" / "agent.yaml")
    w = drift.actor_widths(aw)
    bank = drift.load_bank(bank_path)
    obs = bank["obs"]
    sd_s1 = drift.load_sd(s1_ckpt)
    d_z = d_c0 = cos_zst2 = None
    if ckpt.is_file():
        d_i = drift.drift(bank, drift.load_sd(ckpt), sd_s1, w)
        d_z = d_i["D"]
    if c0_ckpt.is_file():
        d_c0 = drift.drift(bank, drift.load_sd(c0_ckpt), sd_s1, w)["D"]
    if ckpt.is_file() and st2_ckpt.is_file():
        dz = mean_delta_mu(obs, drift.load_sd(ckpt), sd_s1, w)
        dst2 = mean_delta_mu(obs, drift.load_sd(st2_ckpt), sd_s1, w)
        cos_zst2 = cosine(dz, dst2)
    return {"ckpt": str(ckpt), "exists": ckpt.is_file(), "c0_ckpt_exists": c0_ckpt.is_file(),
           "st2_ckpt_exists": st2_ckpt.is_file(), "D_Z": d_z, "D_C0": d_c0, "cosine_delta_mu_vs_st2": cos_zst2}


def run_once(run_dir: Path, s1_ckpt: Path, c0_run: Path, st2_run: Path, bank_path: Path,
            agent_yaml: Path | None = None, mode: str = "z1", stand_bank: Path | None = None) -> dict:
    """The one-shot read over both :data:`CHECKPOINTS` (IO + pure judging). ``rc`` is the worst of every
    checkpoint read (``RC_ERROR`` if the run's own TensorBoard series cannot be read at all). ``mode="z2"`` also
    scores each checkpoint's stand fidelity on ``stand_bank`` (:func:`p2r_z2_fidelity.run`, the eval path)."""
    if mode == "z2" and stand_bank is None:
        return {"rc": RC_ERROR, "why": "--mode z2 needs --stand-bank (the restance probe's p2r_stand_bank.npz)"}
    try:
        ser = live.read_tb(run_dir)
    except Exception as e:  # noqa: BLE001 -- an unreadable run dir is RC_ERROR, never silently n/a
        return {"rc": RC_ERROR, "why": f"{run_dir}: TensorBoard unreadable ({e.__class__.__name__}: {e})"}
    first, _last = live.first_last(ser)
    if first is None:
        return {"rc": RC_WAIT, "why": f"{run_dir}: no telemetry yet"}
    out = {"run_dir": str(run_dir), "fork_first_iteration": first, "checkpoints": {}}
    if mode == "z2":
        out["mode"] = mode
    worst = RC_OK
    for name in CHECKPOINTS:
        ckpt = Path(run_dir) / name
        d_score = score_checkpoint(ckpt, s1_ckpt, c0_run, st2_run, bank_path, agent_yaml)
        if not d_score["exists"]:
            out["checkpoints"][name] = {"checkpoint": name, "rc": RC_WAIT, "why": f"{ckpt} does not exist yet"}
            worst = max(worst, RC_WAIT)
            continue
        fid = None
        if mode == "z2":
            try:
                fid = z2fid.run(Path(stand_bank), ckpt, agent_yaml)
            except (OSError, KeyError, ValueError) as e:
                return {"rc": RC_ERROR, "why": f"stand fidelity unreadable ({e.__class__.__name__}: {e})"}
        j = judge_checkpoint(name, d_score["D_Z"], d_score["D_C0"], d_score["cosine_delta_mu_vs_st2"], ser, first,
                             mode=mode, fidelity=fid)
        j["score"] = d_score
        out["checkpoints"][name] = j
        worst = max(worst, j["rc"])
    out["rc"] = worst
    return out


def watch(run_dir: Path, poll_s: int, timeout_s: float | None, **kwargs) -> dict:
    """Polls every ``poll_s`` until both :data:`CHECKPOINTS` exist under ``run_dir`` (or ``timeout_s``), then
    :func:`run_once`. A sentinel loop for the orchestrator (P2R_STATUE_PLAN.md step 4's own +500 sentinel note)."""
    t0 = time.monotonic()
    while True:
        if all((Path(run_dir) / name).is_file() for name in CHECKPOINTS):
            break
        if timeout_s is not None and time.monotonic() - t0 > timeout_s:
            return {"rc": RC_ERROR, "why": f"timed out after {timeout_s}s waiting for {CHECKPOINTS} under {run_dir}"}
        time.sleep(poll_s)
    return run_once(run_dir, **kwargs)


# ================================================================================================ self-test
def _ser(first=14996, n=1100, **const):
    return {tag: [(first + i, v) for i in range(n)] for tag, v in const.items()}


def _self_test() -> int:
    import tempfile

    import numpy as np
    import p2r_graft as graft

    ok = True

    _SENTINEL = object()

    def check_(name, got, want=_SENTINEL):
        nonlocal ok
        cond = got if want is _SENTINEL else (got == want)
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}: {got if want is _SENTINEL else (got, want)}")
    F0 = 19995
    # ---- formation / residual (pure, no real TensorBoard needed) ----
    flat = _ser(first=F0, n=1100, **{"Loss/stand_prior": 2.3})
    check_("formation: a flat 2.3 series stays 10x above 0.1x its first value -> flagged",
          judge_formation(flat["Loss/stand_prior"], F0)["rc"], RC_NOTIFY)

    def ramp(vals_by_offset, n=1100):
        pts = []
        last = 2.3
        for i in range(n):
            last = vals_by_offset.get(i, last)
            pts.append((F0 + i, last))
        return pts
    formed = ramp({0: 2.3, 150: 0.2, 199: 0.2})
    check_("formation: a value that ramps down by +200 passes",
          judge_formation(formed, F0)["rc"], RC_OK)
    check_("formation: missing telemetry waits",
          judge_formation([], F0)["rc"], RC_WAIT)
    resid_ok = ramp({0: 2.3, 199: 0.2, 499: 0.2, 950: 0.3})     # +1000 at 0.3, under 2x the 0.2 baseline
    check_("residual: a mild rise (< 2x the +200..+500 minimum) passes",
          judge_residual(resid_ok, None, F0)["rc"], RC_OK)
    resid_bad = ramp({0: 2.3, 199: 0.2, 499: 0.2, 950: 0.5})     # window [+951, +1000] entirely at 0.5 > 2x 0.2
    check_("residual: a rise past 2x the minimum flags",
          judge_residual(resid_bad, None, F0)["rc"], RC_NOTIFY)
    pr = _ser(first=F0, n=1100, **{"x": 0.12})["x"]
    r = judge_residual(resid_bad, pr, F0)
    check_("residual: per-stand-row divides by prior_rows", r["at"][1000]["per_stand_row"],
          r["at"][1000]["rolling50"] / r["at"][1000]["prior_rows"])
    check_("drift: D(Z) under 0.5x D(C0) passes", judge_drift(0.5, 1.4)["rc"], RC_OK)
    check_("drift: D(Z) over 0.5x D(C0) flags", judge_drift(0.8, 1.4)["rc"], RC_NOTIFY)
    check_("drift: missing D waits", judge_drift(None, 1.4)["rc"], RC_WAIT)

    # ---- checkpoint-level aggregation ----
    ser_ok = dict(flat, **{"Loss/stand_prior": formed, "Loss/prior_rows": pr,
                           "PPO/kl_mean": _ser(first=F0, n=1100, **{"x": 0.01})["x"],
                           "PPO/lr_applied": _ser(first=F0, n=1100, **{"x": 1e-5})["x"],
                           "PPO/grad_norm": _ser(first=F0, n=1100, **{"x": 0.5})["x"],
                           "Loss/value_function": _ser(first=F0, n=1100, **{"x": 0.3})["x"],
                           "Loss/grad_norm_statue": [(F0 + i, 0.4) for i in range(0, 1100, 50)],
                           "Loss/grad_norm_anchor": [(F0 + i, 0.1) for i in range(0, 1100, 50)],
                           "Loss/grad_norm_rest": [(F0 + i, 1.2) for i in range(0, 1100, 50)]})
    j = judge_checkpoint("model_20501.pt", 0.5, 1.4, 0.9, ser_ok, F0)
    check_("judge_checkpoint: a clean checkpoint reads rc 0", j["rc"], RC_OK)
    check_("judge_checkpoint: reported tags are populated", j["reported"]["PPO/kl_mean"] is not None)
    check_("judge_checkpoint: grad-norm split is read (every-50 logging)",
          j["reported"]["Loss/grad_norm_statue"] == 0.4)
    j_bad = judge_checkpoint("model_20501.pt", 0.9, 1.4, 0.9, ser_ok, F0)
    check_("judge_checkpoint: a flagged drift propagates to rc", j_bad["rc"], RC_NOTIFY)

    # ---- Z2 mode (NOTE(p2r Z2, 2026-10-10)) ----
    z2_slow = ramp({0: 2.3, 450: 0.7})                          # 0.7 at +500 > 0.25 x 2.3 = 0.575
    check_("z2 formation: 0.7 at +500 flags against 0.25x", judge_formation(
        z2_slow, F0, at=Z2_FORMATION_AT, frac=Z2_FORMATION_FRAC)["rc"], RC_NOTIFY)
    z2_ok = ramp({0: 2.3, 450: 0.5})
    check_("z2 formation: 0.5 at +500 passes", judge_formation(
        z2_ok, F0, at=Z2_FORMATION_AT, frac=Z2_FORMATION_FRAC)["rc"], RC_OK)
    check_("z2 fidelity: within the line -> ok", judge_fidelity({"rc": z2fid.RC_OK})["rc"], RC_OK)
    check_("z2 fidelity: over the line -> notify", judge_fidelity({"rc": z2fid.RC_NOTIFY})["rc"], RC_NOTIFY)
    check_("z2 fidelity: no HOLD rows -> notify", judge_fidelity({"rc": z2fid.RC_NO_ROWS})["rc"], RC_NOTIFY)
    ser_z2 = dict(ser_ok, **{"Loss/stand_prior": z2_ok})
    jz = judge_checkpoint("model_20500.pt", 0.5, 1.4, 0.9, ser_z2, F0, mode="z2", fidelity={"rc": z2fid.RC_OK})
    check_("z2 judge_checkpoint: clean -> rc 0, residual reported only", (jz["rc"], "residual" in jz,
                                                                          "residual_reported" in jz),
           (RC_OK, False, True))
    jz_bad = judge_checkpoint("model_20500.pt", 0.5, 1.4, 0.9, ser_z2, F0, mode="z2",
                              fidelity={"rc": z2fid.RC_NOTIFY})
    check_("z2 judge_checkpoint: a fidelity flag propagates", jz_bad["rc"], RC_NOTIFY)

    # ---- end-to-end score_checkpoint on tiny synthetic checkpoints (no real Isaac/CUDA checkpoint needed) ----
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d, agent_yaml_dict, dims = graft._tiny_checkpoint()
        s1 = td / "s1" / "model_19994.pt"
        s1.parent.mkdir(parents=True)
        torch.save(d, str(s1))
        (s1.parent / "params").mkdir()
        import yaml
        yaml.safe_dump(agent_yaml_dict, open(s1.parent / "params" / "agent.yaml", "w"))
        c0 = td / "c0"
        c0.mkdir()
        d_c0, _, _ = graft._tiny_checkpoint(seed=1)
        torch.save(d_c0, str(c0 / "model_20500.pt"))
        st2 = td / "st2"
        st2.mkdir()
        d_st2, _, _ = graft._tiny_checkpoint(seed=2)
        torch.save(d_st2, str(st2 / "model_20500.pt"))
        z = td / "z"
        z.mkdir()
        d_z, _, _ = graft._tiny_checkpoint(seed=3)
        torch.save(d_z, str(z / "model_20500.pt"))
        n = 16
        bank_obs = torch.randn(n, dims["num_critic_obs"], generator=torch.Generator().manual_seed(9))
        bank_path = td / "bank.npz"
        np.savez(bank_path, obs=bank_obs.numpy(), s1_sigma=np.full(3, 0.3, dtype="float32"),
                hold_label=np.array(["fwd"] * n), category=np.array(["flat"] * n))
        sc = score_checkpoint(z / "model_20500.pt", s1, c0, st2, bank_path, agent_yaml=s1.parent / "params" / "agent.yaml")
        check_("score_checkpoint: D(Z) and D(C0) are both computed", sc["D_Z"] is not None and sc["D_C0"] is not None)
        bank = drift.load_bank(bank_path)
        w = drift.actor_widths(s1.parent / "params" / "agent.yaml")
        sd_s1 = drift.load_sd(s1)
        check_("score_checkpoint: a true self-vs-self D (same checkpoint on both sides) is exactly 0",
              drift.drift(bank, sd_s1, sd_s1, w)["D"], 0.0)
        check_("score_checkpoint: cosine is in [-1, 1]", -1.0 - 1e-6 <= sc["cosine_delta_mu_vs_st2"] <= 1.0 + 1e-6)
        sc_missing = score_checkpoint(z / "model_99999.pt", s1, c0, st2, bank_path,
                                      agent_yaml=s1.parent / "params" / "agent.yaml")
        check_("score_checkpoint: a missing checkpoint reports exists=False, no crash", sc_missing["exists"] is False)

    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--watch", type=Path, metavar="RUN_DIR")
    ap.add_argument("--poll-s", type=float, default=120)
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--bank", type=Path, default=None)
    ap.add_argument("--agent-yaml", type=Path, default=None)
    ap.add_argument("--s1-ckpt", type=Path, default=S1_CKPT_DEFAULT)
    ap.add_argument("--c0-run", type=Path, default=C0_RUN_DEFAULT)
    ap.add_argument("--st2-run", type=Path, default=ST2_RUN_DEFAULT)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--mode", choices=MODES, default="z1")
    ap.add_argument("--stand-bank", type=Path, default=None, help="z2: the restance probe's p2r_stand_bank.npz")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return _self_test()
    run_dir = a.watch or a.run_dir
    if not run_dir:
        ap.error("--run-dir or --watch is required (or --self-test)")
    if not a.bank:
        ap.error("--bank is required (the step-1 non-stand probe bank, p2r_bank.npz)")
    if a.mode == "z2" and not a.stand_bank:
        ap.error("--mode z2 needs --stand-bank (the restance probe's p2r_stand_bank.npz)")
    kwargs = dict(s1_ckpt=a.s1_ckpt, c0_run=a.c0_run, st2_run=a.st2_run, bank_path=a.bank, agent_yaml=a.agent_yaml,
                  mode=a.mode, stand_bank=a.stand_bank)
    out = watch(run_dir, a.poll_s, a.timeout_s, **kwargs) if a.watch else run_once(run_dir, **kwargs)
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(text)
    return int(out.get("rc", RC_ERROR))


if __name__ == "__main__":
    raise SystemExit(main())
