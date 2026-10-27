#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 live watch (CPU, read-only): the pre-registered LIVE rules a running stage is read against, every poll.

Pure judges on ``{tag: [(step, value)]}`` series (so they replay on recorded tfevents exactly as they run live);
iterations count FROM THE STAGE START (the run's first logged iteration; a resume of model_14995 logs from 14995 or
14996). Return codes (the queue's contract): 0 PASS / not yet applicable, 2 WAIT (not enough telemetry), 3 KILL.

DRIVER stages (walker and payload; DESIGN 2.1 "live", phase-2 ``p2_live.py`` collapse rule + ``resume_read.py``):
  resume     (a resumed stage, once >= 5 points) the fork CONTINUES its source: iteration numbering continues,
             PPO/effective_std within 0.02 of the source's last, Loss/priv_reg_coef within 5 %, Loss/action_bound_coef
             within 2 % (where the source logged it);
  dead       (s1 only) Metrics/base_parkour/track_ratio_mean over its 800-1000 < 0.35 (a dead walker);
  collapse   (a RESUMED stage, after +300) rolling-200 Train/mean_episode_length < 0.5 x the SOURCE stage's
             last-200 mean, or rolling-200 Episode_Termination/crab_failure > 0.6 (a from-scratch s1 falls > 0.6
             early by construction: replayed on X43 it would kill at +300; the dead-walker rule covers s1);
  nan        any non-finite Loss/* value.
STUDENT (distill; PLAN Gates -> Student -> live checks):
  loss       Loss_depth/depth_actor_loss rolling-50 at +250 <= 0.5 x the first logged value (``loss_rule``: kill,
             the default, or watch -- computed and reported under ``watch``, never a kill; STEPC_RULES C4.8 / V.3);
  ep_len     from +250, rolling-50 Train/mean_episode_length >= 0.8 x T*'s training tail (last-200 mean);
  crab       from +250, rolling-50 Episode_Termination/crab_failure <= T*'s tail + 0.10;
  nan        a non-finite depth loss.
Every live rule is ``unvalidated`` (PLAN: live-kill reads are excluded from the chance-HALT sum); the student's loss
rule becomes WATCH if it fails on the forward 3a tfevents or the step-8 pilots (``--replay`` reports both).

    live_watch.py check --run-dir DIR --kind driver|student [--stage s1] [--source DIR] [--tstar DIR] [--first N]
    live_watch.py replay --run-dir DIR --kind ... [--every 50]      (the polls a live run would have made)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

RC_PASS, RC_WAIT, RC_KILL = 0, 2, 3
LOSS_RULES = ("kill", "watch")
T_EPLEN = "Train/mean_episode_length"
T_CRAB = "Episode_Termination/crab_failure"
T_STD = "PPO/effective_std"
T_PRIV = "Loss/priv_reg_coef"
T_BOUND = "Loss/action_bound_coef"
T_TRACK = "Metrics/base_parkour/track_ratio_mean"
T_DEPTH = "Loss_depth/depth_actor_loss"
DRIVER_TAGS = (T_EPLEN, T_CRAB, T_STD, T_PRIV, T_BOUND, T_TRACK)
STUDENT_TAGS = (T_EPLEN, T_CRAB, T_DEPTH)

DEFAULTS = {
    "collapse": {"after": 300, "roll": 200, "ep_len_frac": 0.5, "crab_failure_max": 0.6},
    "dead_walker": {"stages": ["s1"], "window": [800, 1000], "track_min": 0.35},
    "resume": {"std_tol": 0.02, "priv_rel": 0.05, "bound_rel": 0.02, "min_points": 5},
    "student": {"loss_at": 250, "loss_roll": 50, "loss_frac": 0.5, "from": 250, "ep_len_frac": 0.8,
                "crab_failure_margin": 0.10, "roll": 50, "loss_rule": "kill"},
}


def read_series(run_dir: Path, tags=None) -> dict:
    """{tag: [(step, value), ...]} from a run dir's tfevents (every tag when ``tags`` is None)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    # "*/time" tags are logged with WALL-CLOCK seconds as their step (CHARTER step-2 note): never iterations
    have = {t for t in ea.Tags().get("scalars", []) if not t.endswith("/time")}
    want = {t for t in have if t.startswith(("Loss/", "Loss_depth/"))} | set(DRIVER_TAGS) | set(STUDENT_TAGS) \
        if tags is None else set(tags)
    return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in sorted(want & have)}


def _since(ser, first):
    return [(s, v) for s, v in (ser or []) if s >= first]


def roll_mean(ser, n):
    vals = [v for _, v in (ser or [])][-n:]
    return sum(vals) / len(vals) if vals else None


def tail_mean(ser, n=200):
    return roll_mean(ser, n)


def first_step(series: dict):
    steps = [s for v in series.values() for s, _ in (v or [])]
    return min(steps) if steps else None


def last_step(series: dict):
    steps = [s for v in series.values() for s, _ in (v or [])]
    return max(steps) if steps else None


def cut(series: dict, upto: int) -> dict:
    """The series a live poll at iteration ``upto`` would have seen (replay)."""
    return {t: [(s, v) for s, v in ser if s <= upto] for t, ser in series.items()}


# ============================================================================== driver judges
def judge_resume(fork: dict, source: dict, cfg=None) -> dict:
    """resume_read.judge (PPO-construction CHARTER 10.3), parameterised by the spec's ``live.resume``."""
    c = {**DEFAULTS["resume"], **(cfg or {})}
    out = {"rule": "resume", "fails": [], "reads": {}}
    sstd, fstd = source.get(T_STD) or [], fork.get(T_STD) or []
    if not sstd:
        return {**out, "rc": RC_KILL, "fails": ["source run has no PPO/effective_std"]}
    if len(fstd) < c["min_points"]:
        return {**out, "rc": RC_WAIT, "fails": [f"fork has {len(fstd)} PPO/effective_std points"]}
    last, first = sstd[-1][0], fstd[0][0]
    if first not in (last, last + 1):
        out["fails"].append(f"iteration numbering restarted: fork starts at {first}, source ends at {last}")
    d = abs(fstd[0][1] - sstd[-1][1])
    out["reads"]["std"] = {"source_last": sstd[-1][1], "fork_first": fstd[0][1]}
    if d > c["std_tol"]:
        out["fails"].append(f"effective_std not continued: {fstd[0][1]:.4f} vs {sstd[-1][1]:.4f}")
    for tag, rel in ((T_PRIV, c["priv_rel"]), (T_BOUND, c["bound_rel"])):
        s, f = source.get(tag) or [], fork.get(tag) or []
        if not s:
            continue
        if not f:
            out["fails"].append(f"{tag} logged by the source but absent in the fork")
            continue
        sv, fv = s[-1][1], f[0][1]
        out["reads"][tag] = {"source_last": sv, "fork_first": fv}
        if sv > 0 and abs(fv - sv) > rel * sv + 1e-9:
            out["fails"].append(f"{tag} not continued: {fv:.4g} vs {sv:.4g}")
    return {**out, "rc": RC_KILL if out["fails"] else RC_PASS}


def judge_dead_walker(ser: dict, first: int, cfg=None) -> dict:
    c = {**DEFAULTS["dead_walker"], **(cfg or {})}
    lo, hi = c["window"]
    tr = [(s, v) for s, v in ser.get(T_TRACK) or [] if first + lo <= s <= first + hi]
    last = last_step(ser)
    if last is None or last < first + hi:
        return {"rule": "dead_walker", "rc": RC_PASS, "result": f"n/a (before +{hi})"}
    if not tr:
        return {"rule": "dead_walker", "rc": RC_WAIT, "fails": [f"no {T_TRACK} in +{lo}..+{hi}"]}
    m = sum(v for _, v in tr) / len(tr)
    fails = [f"track_ratio_mean {m:.3f} over +{lo}..+{hi} < {c['track_min']}"] if m < c["track_min"] else []
    return {"rule": "dead_walker", "rc": RC_KILL if fails else RC_PASS, "fails": fails, "track": m}


def judge_collapse(ser: dict, first: int, source_tail_len, cfg=None) -> dict:
    """p2_live.judge_collapse (phase-2 CHARTER 8), parameterised."""
    c = {**DEFAULTS["collapse"], **(cfg or {})}
    last = last_step(ser)
    if last is None or last < first + c["after"]:
        return {"rule": "collapse", "rc": RC_PASS, "result": f"n/a (before +{c['after']})"}
    el, cf = _since(ser.get(T_EPLEN), first), _since(ser.get(T_CRAB), first)
    if not el or not cf:
        return {"rule": "collapse", "rc": RC_WAIT, "fails": ["episode length / crab_failure missing"]}
    r_len, r_cf = roll_mean(el, c["roll"]), roll_mean(cf, c["roll"])
    fails = []
    # a from-scratch stage (s1) has no source tail: its episode-length clause is n/a, the crab_failure one holds
    if source_tail_len and r_len < c["ep_len_frac"] * source_tail_len:
        fails.append(f"rolling-{c['roll']} episode length {r_len:.1f} < {c['ep_len_frac']} x source tail "
                     f"{source_tail_len:.1f}")
    if r_cf > c["crab_failure_max"]:
        fails.append(f"rolling-{c['roll']} crab_failure {r_cf:.3f} > {c['crab_failure_max']}")
    return {"rule": "collapse", "rc": RC_KILL if fails else RC_PASS, "fails": fails, "ep_len_roll": r_len,
            "crab_failure_roll": r_cf, "source_tail": source_tail_len}


def judge_nan(ser: dict, prefixes=("Loss/", "Loss_depth/")) -> dict:
    bad = sorted({t for t, v in ser.items() if t.startswith(prefixes) and any(not math.isfinite(x) for _, x in v)})
    return {"rule": "nan", "rc": RC_KILL if bad else RC_PASS, "fails": [f"non-finite {t}" for t in bad]}


# ============================================================================== student judges
def judge_student(ser: dict, first: int, tstar_tail: dict | None, cfg=None) -> dict:
    """PLAN Gates -> Student -> live checks (``tstar_tail``: {ep_len, crab_failure} = T*'s last-200 means)."""
    c = {**DEFAULTS["student"], **(cfg or {})}
    if c["loss_rule"] not in LOSS_RULES:
        raise ValueError(f"live_watch: student loss_rule {c['loss_rule']!r} not in {LOSS_RULES}")
    last = last_step(ser)
    out = {"rule": "student", "fails": [], "reads": {}, "watch": []}
    loss = _since(ser.get(T_DEPTH), first)
    if any(not math.isfinite(v) for _, v in loss):
        return {**out, "rc": RC_KILL, "fails": ["non-finite depth_actor_loss"]}
    if last is None or last < first + c["loss_at"]:
        return {**out, "rc": RC_PASS, "result": f"n/a (before +{c['loss_at']})"}
    if not loss:
        return {**out, "rc": RC_WAIT, "fails": [f"no {T_DEPTH}"]}
    at = [(s, v) for s, v in loss if s <= first + c["loss_at"]]
    r = roll_mean(at, c["loss_roll"])
    out["reads"]["loss"] = {"first": loss[0][1], "rolling_at": r, "loss_rule": c["loss_rule"]}
    if r is not None and r > c["loss_frac"] * loss[0][1]:
        msg = (f"depth_actor_loss rolling-{c['loss_roll']} at +{c['loss_at']} {r:.4f} > "
               f"{c['loss_frac']} x first {loss[0][1]:.4f}")
        out["reads"]["loss"]["fails"] = True
        (out["fails"] if c["loss_rule"] == "kill" else out["watch"]).append(msg)
    if tstar_tail and last >= first + c["from"]:
        el = _since(ser.get(T_EPLEN), first + c["from"])
        cf = _since(ser.get(T_CRAB), first + c["from"])
        if el and tstar_tail.get("ep_len"):
            m = roll_mean(el, c["roll"])
            out["reads"]["ep_len"] = m
            if m < c["ep_len_frac"] * tstar_tail["ep_len"]:
                out["fails"].append(f"rolling-{c['roll']} episode length {m:.1f} < {c['ep_len_frac']} x T* tail "
                                    f"{tstar_tail['ep_len']:.1f}")
        if cf and tstar_tail.get("crab_failure") is not None:
            m = roll_mean(cf, c["roll"])
            out["reads"]["crab_failure"] = m
            if m > tstar_tail["crab_failure"] + c["crab_failure_margin"]:
                out["fails"].append(f"rolling-{c['roll']} crab_failure {m:.3f} > T* tail "
                                    f"{tstar_tail['crab_failure']:.3f} + {c['crab_failure_margin']}")
    return {**out, "rc": RC_KILL if out["fails"] else RC_PASS}


# ============================================================================== combined check
def check(kind: str, ser: dict, *, stage: str | None = None, first: int | None = None, source: dict | None = None,
          tstar_tail: dict | None = None, cfg: dict | None = None, accepted: tuple = ()) -> dict:
    """Every applicable rule for one poll; the first KILL wins (``accepted``: rule names the user accepted as chance
    for this stage -- they are read and reported, never kill again)."""
    cfg = cfg or {}
    first = first_step(ser) if first is None else first
    if first is None:
        return {"rc": RC_WAIT, "rule": None, "reads": [], "why": "no telemetry yet"}
    reads = []
    if kind == "driver":
        reads.append(judge_nan(ser))
        if source:
            reads.append(judge_resume(ser, source, cfg.get("resume")))
        if stage in (cfg.get("dead_walker") or DEFAULTS["dead_walker"]).get("stages", []):
            reads.append(judge_dead_walker(ser, first, cfg.get("dead_walker")))
        # collapse is a RESUMED stage's rule (p2_live: against the source stage's tail); a from-scratch stage (s1)
        # falls at > 0.6 early by construction -- the dead-walker rule covers it
        if source:
            reads.append(judge_collapse(ser, first, tail_mean(source.get(T_EPLEN)), cfg.get("collapse")))
    elif kind == "student":
        reads.append(judge_student(ser, first, tstar_tail, cfg.get("student")))
    else:
        raise ValueError(f"live_watch: unknown kind {kind!r}")
    kills = [r for r in reads if r.get("rc") == RC_KILL and r["rule"] not in accepted]
    waits = [r for r in reads if r.get("rc") == RC_WAIT]
    rc = RC_KILL if kills else (RC_WAIT if waits else RC_PASS)
    return {"rc": rc, "rule": kills[0]["rule"] if kills else None, "fails": [f for r in kills for f in r["fails"]],
            "reads": reads, "first": first, "last": last_step(ser), "unvalidated": True,
            "accepted": list(accepted), "watch": [w for r in reads for w in (r.get("watch") or [])]}


def replay(kind: str, ser: dict, *, every: int = 50, **kw) -> dict:
    """The polls a live run would have made every ``every`` iterations: the first KILL (or none)."""
    first, last = first_step(ser), last_step(ser)
    if first is None:
        return {"killed_at": None, "polls": 0}
    n, watch = 0, None
    for at in range(first + every, last + 1, every):
        n += 1
        r = check(kind, cut(ser, at), first=first, **kw)
        if r.get("watch") and watch is None:
            watch = {"at": at, "rel": at - first, "reads": r["watch"]}
        if r["rc"] == RC_KILL:
            return {"killed_at": at, "rel": at - first, "rule": r["rule"], "fails": r["fails"], "polls": n,
                    "watch_first": watch}
    return {"killed_at": None, "polls": n, "watch_first": watch}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--kind", choices=("driver", "student"), required=True)
        p.add_argument("--stage", default=None)
        p.add_argument("--source", type=Path, default=None)
        p.add_argument("--tstar", type=Path, default=None)
        p.add_argument("--first", type=int, default=None)
        p.add_argument("--loss-rule", choices=LOSS_RULES, default=None,
                       help="student: kill (default) | watch (the loss rule is reported, never kills)")
        if name == "replay":
            p.add_argument("--every", type=int, default=50)
    a = ap.parse_args(argv)
    ser = read_series(a.run_dir)
    src = read_series(a.source) if a.source else None
    tt = None
    if a.tstar:
        ts = read_series(a.tstar, (T_EPLEN, T_CRAB))
        tt = {"ep_len": tail_mean(ts.get(T_EPLEN)), "crab_failure": tail_mean(ts.get(T_CRAB))}
    cfg = {"student": {"loss_rule": a.loss_rule}} if a.loss_rule else None
    if a.cmd == "check":
        res = check(a.kind, ser, stage=a.stage, first=a.first, source=src, tstar_tail=tt, cfg=cfg)
    else:
        res = replay(a.kind, ser, every=a.every, stage=a.stage, source=src, tstar_tail=tt, cfg=cfg)
    print(json.dumps(res, indent=1, default=str))
    return int(res.get("rc", 0)) if a.cmd == "check" else 0


if __name__ == "__main__":
    sys.exit(main())
