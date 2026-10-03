#!/usr/bin/env python3
"""x6k fork wiring read (CHARTER 10.3): did a ``resume_from: {reference}`` fork CONTINUE its source head?

CPU only. Compares the fork run's first logged iterations with the source run's last ones:

* iteration numbering continues: the fork's first step is the source's last step or the one after
  (rsl_rl re-runs the last index on resume: SMOKES.md "One A-head stage fork as in P1");
* ``PPO/effective_std`` continues: |fork first - source last| <= 0.02 (B's source sigma is about 0.748 and its
  init 0.8, so a re-initialised sigma fails; C0's source sigma is the 2.0 clamp);
* ``Loss/priv_reg_coef`` continues (a restarted ramp reads 0): within 5 % of the source's last value;
* ``Loss/action_bound_coef`` continues where the source logged it (a restarted ramp reads 1e-4, the source
  ends at 1e-3): within 2 %.

The adaptive-KL learning rate is NOT read: it moves between the floor and its ceiling within the first update
even on a fresh run, so it cannot tell a resume from a restart.

Exit codes (the run_arm.sh watcher's contract): 0 PASS, 2 not enough fork telemetry yet, 3 FAIL (a wiring
failure: the watcher kills the fork and stops the queue).

    resume_read.py --fork RUN_DIR --source RUN_DIR [--json OUT]
    resume_read.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TAGS = ("PPO/effective_std", "Loss/priv_reg_coef", "Loss/action_bound_coef")
STD_TOL = 0.02
PRIV_REL = 0.05
BOUND_REL = 0.02
MIN_FORK_POINTS = 5


def read_series(run_dir: Path) -> dict:
    """{tag: [(step, value), ...]} for TAGS (missing tags absent)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in TAGS if t in have}


def judge(fork: dict, source: dict) -> dict:
    """Pure decision on two {tag: [(step, value)]} series dicts -> {"rc", "result", "fails", "reads"}."""
    out = {"reads": {}, "fails": []}
    sstd = source.get("PPO/effective_std") or []
    fstd = fork.get("PPO/effective_std") or []
    if not sstd:
        out.update(rc=3, result="FAIL", fails=["source run has no PPO/effective_std"])
        return out
    if len(fstd) < MIN_FORK_POINTS:
        out.update(rc=2, result="WAIT", fails=[f"fork has {len(fstd)} PPO/effective_std points (< {MIN_FORK_POINTS})"])
        return out
    src_last_step = sstd[-1][0]
    first_step = fstd[0][0]
    out["reads"]["source_last_step"] = src_last_step
    out["reads"]["fork_first_step"] = first_step
    if first_step not in (src_last_step, src_last_step + 1):
        out["fails"].append(f"iteration numbering restarted: fork starts at {first_step}, source ends at {src_last_step}")
    d = abs(fstd[0][1] - sstd[-1][1])
    out["reads"]["std"] = {"source_last": sstd[-1][1], "fork_first": fstd[0][1], "abs_diff": d}
    if d > STD_TOL:
        out["fails"].append(f"effective_std not continued: {fstd[0][1]:.4f} vs source {sstd[-1][1]:.4f} (> {STD_TOL})")
    for tag, rel in (("Loss/priv_reg_coef", PRIV_REL), ("Loss/action_bound_coef", BOUND_REL)):
        s, f = source.get(tag) or [], fork.get(tag) or []
        if not s:
            continue                       # C0 has no bound coef; nothing to continue
        if not f:
            out["fails"].append(f"{tag} logged by the source but absent in the fork")
            continue
        sv, fv = s[-1][1], f[0][1]
        out["reads"][tag] = {"source_last": sv, "fork_first": fv}
        if sv > 0 and abs(fv - sv) > rel * sv + 1e-9:
            out["fails"].append(f"{tag} not continued: {fv:.4g} vs source {sv:.4g} (> {rel:.0%})")
    out.update(rc=3 if out["fails"] else 0, result="FAIL" if out["fails"] else "PASS")
    return out


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    src = {"PPO/effective_std": [(2998, 0.7484), (2999, 0.7483)],
           "Loss/priv_reg_coef": [(2998, 0.03827), (2999, 0.0383)],
           "Loss/action_bound_coef": [(2998, 1e-3), (2999, 1e-3)]}
    cont = {"PPO/effective_std": [(2999 + i, 0.7483) for i in range(6)],
            "Loss/priv_reg_coef": [(2999 + i, 0.0383) for i in range(6)],
            "Loss/action_bound_coef": [(2999 + i, 1e-3) for i in range(6)]}
    check("a continued B fork passes", judge(cont, src)["rc"], 0)
    check("too few fork points waits", judge({k: v[:3] for k, v in cont.items()}, src)["rc"], 2)
    reinit = dict(cont, **{"PPO/effective_std": [(2999 + i, 0.8) for i in range(6)]})
    check("a re-initialised sigma (0.8) fails", judge(reinit, src)["rc"], 3)
    ramp = dict(cont, **{"Loss/action_bound_coef": [(2999 + i, 1e-4) for i in range(6)]})
    check("a restarted bound ramp fails", judge(ramp, src)["rc"], 3)
    priv = dict(cont, **{"Loss/priv_reg_coef": [(2999 + i, 0.0) for i in range(6)]})
    check("a restarted priv_reg ramp fails", judge(priv, src)["rc"], 3)
    renum = {k: [(i, v[0][1]) for i in range(6)] for k, v in cont.items()}
    check("restarted iteration numbering fails", judge(renum, src)["rc"], 3)
    nobound = {k: v for k, v in cont.items() if k != "Loss/action_bound_coef"}
    check("a bound coef the source logged but the fork lacks fails", judge(nobound, src)["rc"], 3)
    c0src = {"PPO/effective_std": [(2999, 2.0)], "Loss/priv_reg_coef": [(2999, 0.0383)]}
    c0fork = {"PPO/effective_std": [(3000 + i, 2.0) for i in range(6)],
              "Loss/priv_reg_coef": [(3000 + i, 0.0383) for i in range(6)]}
    check("a continued C0 fork (no bound coef; first step = last + 1) passes", judge(c0fork, c0src)["rc"], 0)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", type=Path)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.fork or not a.source:
        ap.error("--fork and --source are required")
    res = judge(read_series(a.fork), read_series(a.source))
    res.update(fork=str(a.fork), source=str(a.source))
    line = json.dumps(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    print(line)
    return int(res["rc"])


if __name__ == "__main__":
    sys.exit(main())
