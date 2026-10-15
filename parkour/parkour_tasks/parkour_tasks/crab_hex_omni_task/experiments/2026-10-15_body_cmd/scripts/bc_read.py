#!/usr/bin/env python3
"""Shared reader for the body-command campaign (CPU): per-hold body metrics of an eval stamp.

Reads the harness's own per-episode records (``metrics/episode_*.json`` -> ``holds[label]['body']``) and the stamp's
``run_meta.json`` (``hold_body_cmds``, ``env_index_map``). Two stamp shapes:

* category suite / envelope grid (``env_index_map`` present): each episode carries ONE own hold; ``survived`` =
  the episode completed its schedule.
* single-pass timeline (8-hold field rows): every episode carries every hold; a hold is ``survived`` when the episode
  was still alive at the hold's end (``holds[label]['n_steps']`` covers the whole hold, or the schedule completed).

Every statistic of a body metric is taken over SURVIVING episodes only (CHARTER 2026-10-15: a fall inside a hold
truncates its window); survival is reported beside it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BODY_KEYS = ("net_yaw_rate", "net_yaw_ratio", "wz_ratio", "yaw_ripple", "heading_drift", "planar_drift",
             "body_ratio", "cross_body", "wz_cmd", "v_cmd", "window_s")


def _json(p: Path):
    return json.loads(Path(p).read_text())


def hold_cmds(stamp) -> dict[str, dict]:
    m = _json(Path(stamp) / "run_meta.json")
    return {h["label"]: h for h in m.get("hold_body_cmds") or []}


def load(stamp) -> dict:
    """``{label: {"n": episodes that ran it, "n_surv": .., "survival": .., "eps": [body dict of each survivor]}}``."""
    S = Path(stamp)
    m = _json(S / "run_meta.json")
    eps = sorted((S / "metrics").glob("episode_*.json"))
    out: dict[str, dict] = {}
    suite = bool(m.get("env_index_map"))
    for f in eps:
        e = _json(f)
        done = bool(e.get("completed_schedule")) or e.get("termination_reason") == "schedule_complete"
        holds = e.get("holds") or {}
        labels = [e["hold"]] if suite and e.get("hold") else [l for l in holds if l not in ("settle", "prologue")]
        for lab in labels:
            h = holds.get(lab) or {}
            body = h.get("body") or {}
            rec = out.setdefault(lab, {"n": 0, "n_surv": 0, "eps": [], "category": e.get("category"),
                                       "hand": e.get("hand")})
            if not h:
                continue
            rec["n"] += 1
            if suite:
                alive = done
            else:
                # an 8-hold timeline: alive at this hold's end iff the episode completed or outlived the hold
                alive = done or bool(h.get("survived", False))
            if alive:
                rec["n_surv"] += 1
                rec["eps"].append(body)
    for rec in out.values():
        rec["survival"] = rec["n_surv"] / rec["n"] if rec["n"] else None
    return out


def stat(rec: dict, key: str, how: str = "median"):
    vals = [b.get(key) for b in rec["eps"]]
    vals = np.array([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if not vals.size:
        return None
    if how == "median":
        return float(np.median(vals))
    if how == "mean":
        return float(vals.mean())
    if how == "p90":
        return float(np.quantile(vals, 0.9))
    if how == "absmedian":
        return float(np.median(np.abs(vals)))
    if how == "absp90":
        return float(np.quantile(np.abs(vals), 0.9))
    raise ValueError(how)


def summary(stamp, keys=BODY_KEYS) -> dict:
    rows = load(stamp)
    return {lab: {"n": r["n"], "survival": r["survival"], "category": r["category"], "hand": r["hand"],
                  **{k: stat(r, k) for k in keys},
                  "heading_drift_p90": stat(r, "heading_drift", "p90")}
            for lab, r in rows.items()}


def completion(stamp):
    a = _json(Path(stamp) / "scenario_metrics.json")["aggregate"]
    return a.get("schedule_completion_rate"), a.get("n_episodes")


if __name__ == "__main__":
    import sys
    print(json.dumps(summary(sys.argv[1]), indent=1))
