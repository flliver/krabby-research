#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 student selection (CPU): the candidate checkpoints, the prescreen rank, the full-block read, the
selection state machine (PLAN Gates -> Student; DESIGN 3.5 where PLAN does not supersede it).

* CANDIDATES: T*'s last iteration + {1000, 2000, 3000, 4000, 5000}, each mapped to the NEAREST saved checkpoint of
  the student run (it saves at absolute multiples of 100 -- 200 after 2500, 500 after 5000 -- so "+1k" is the nearest
  saved; ties go to the earlier one). "Not the last iteration" (grant 4d) is met by this screen.
* PRESCREEN (Student-Play, ``--policy-role student``): suite es1 + field_hard es1-2, AND the 90-kg level (suite es1
  + field_light es1). RANK (pre-registered in the spec): fewer prescreen floor misses (incl. the 90-kg stable lines)
  first, then field_hard completion (desc), then the mean |ratio - 1| over the trained translate categories (asc),
  then the earlier checkpoint.
* FULL BLOCK, in rank order until one passes; the branch HALTs after 2 failed full blocks. PASS = the floors (the
  student floor lines of thresholds.json: 0.90 / 0.70 + side22 FAST/FAST) AND, on field_hard / field_light / ring_obst
  / field_turn (whole schedule, every translate category -- the four cameras cover back), completion >= T* -
  max(0.05, 2 SE_diff, the manifest-shape split measured at step 8), T* read with ``--policy-role teacher`` on
  Student-Play in the same manifest shape and env seeds (so the pair is real). WATCH: the tracking lines + 0.05
  (:func:`tracking_watch`: the student vs T* on the paired tracking statistics, recorded in gates/sel.json, a miss is
  a WATCH flag, never a FAIL) and stable at min(90, T*'s max_stable).
The student-completion line's chance rate is a LOWER bound (no recorded run measures distillation-seed variance).

    select_student.py candidates --run-dir DIR --start ITER [--offsets 1000,2000,...]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OFFSETS = (1000, 2000, 3000, 4000, 5000)
FIELD_ROWS = ("field_hard", "field_light", "ring_obst", "field_turn")
TRANSLATE = ("fwd", "side", "back", "diag")


def _iter_of(p: Path) -> int | None:
    m = re.match(r"^model_(\d+)\.pt$", Path(p).name)
    return int(m.group(1)) if m else None


def candidate_checkpoints(run_dir: Path, start: int, offsets=OFFSETS) -> list[dict]:
    """``start`` = T*'s last iteration (the student run's counter continues from it)."""
    saved = sorted((i, p) for p in Path(run_dir).glob("model_*.pt") if (i := _iter_of(p)) is not None)
    out = []
    for off in offsets:
        target = int(start) + int(off)
        if not saved:
            out.append({"offset": off, "target": target, "ckpt": None})
            continue
        i, p = min(saved, key=lambda ip: (abs(ip[0] - target), ip[0]))
        out.append({"offset": off, "target": target, "iter": i, "ckpt": str(p)})
    seen, uniq = set(), []
    for c in out:                      # two offsets mapping to the same file are one candidate
        if c.get("ckpt") in seen:
            continue
        seen.add(c.get("ckpt"))
        uniq.append(c)
    return uniq


def rank_key(pre: dict) -> tuple:
    """``pre``: {misses, field_hard, mean_abs_dev, iter} of one candidate's prescreen."""
    fh = pre.get("field_hard")
    dev = pre.get("mean_abs_dev")
    return (int(pre.get("misses") or 0), -(fh if fh is not None else -1.0),
            dev if dev is not None else math.inf, int(pre.get("iter") or 0))


def rank(prescreens: list[dict]) -> list[dict]:
    return sorted(prescreens, key=rank_key)


def completion_margin(se_diff: float, cfg: dict) -> dict:
    """max(floor, se_mult x SE_diff, the manifest-shape split); the split must be a NUMBER (step 8) to gate."""
    split = cfg.get("manifest_shape_split")
    if not isinstance(split, (int, float)):
        return {"margin": None, "why": f"manifest_shape_split not set ({str(split)[:60]})"}
    terms = {"floor": float(cfg.get("floor", 0.05)), "se": float(cfg.get("se_mult", 2.0)) * float(se_diff or 0.0),
             "split": float(split)}
    k = max(terms, key=terms.get)
    return {"margin": terms[k], "binds": k, **terms}


def completion_pairs(student: dict, tstar: dict, cfg: dict) -> list[dict]:
    """Per field row: student vs T* completion on the same rows / env seeds. ``student`` / ``tstar``:
    {row: {(es_index, env_index): completed bool}}."""
    TH = _lib().thresholds_mod()
    out = []
    for row in FIELD_ROWS:
        a, b = student.get(row) or {}, tstar.get(row) or {}
        if not a or set(a) != set(b):
            out.append({"row": row, "pass": False, "missing": True,
                        "why": f"{row}: student {len(a)} vs T* {len(b)} episodes (not the same rows / env seeds)"})
            continue
        pa = sum(a.values()) / len(a)
        pb = sum(b.values()) / len(b)
        se = TH.paired_completion_se(a, b)
        m = completion_margin(se, cfg)
        if m["margin"] is None:
            out.append({"row": row, "student": pa, "tstar": pb, "se_diff": se, "pass": False, "missing": True,
                        "why": m["why"]})
            continue
        out.append({"row": row, "student": pa, "tstar": pb, "se_diff": se, **m, "line": pb - m["margin"],
                    "pass": pa >= pb - m["margin"] - 1e-12})
    return out


#: PLAN Gates -> Student "WATCH: tracking lines + 0.05": the slack on the paired tracking statistics
TRACKING_WATCH_MARGIN = 0.05
TRACKING_WATCH_RULE = ("PLAN Gates -> Student 'WATCH: tracking lines + 0.05' (CHARTER 2026-10-21 decision E): the "
                       "student vs T* (Student-Play, --policy-role teacher, same rows / env seeds) on the 0-kg check's "
                       "paired tracking statistics, per suite env seed and per trained translate category + move_turn: "
                       "ratio (one-sided drop, T* - student <= 0.05) and |ratio - 1| (student - T* <= 0.05). WATCH "
                       "only.")


def tracking_watch(student: dict, tstar: dict, trained, margin: float = TRACKING_WATCH_MARGIN) -> dict:
    """The student full block's tracking WATCH (PLAN Gates -> Student; CHARTER 2026-10-21 decision E).

    ``student`` / ``tstar``: ``{env seed: thresholds.suite_episodes(stamp)}`` on the same suite rows and env seeds.
    Per env seed and per tracking category (``trained`` translate categories + move_turn), the paired statistics of
    the 0-kg check (thresholds.zero_kg_*): ``ratio <c>`` -- the one-sided DROP, worse_by = T* - student (the decisive
    form at the 0-kg check) -- and the mirrored ``|ratio-1| <c>``, worse_by = student - T*; each passes while
    worse_by <= ``margin``. ``se_diff`` is the env-clustered paired bootstrap SE (reported). WATCH only: a miss (or
    a missing read) is a flag, never a FAIL. Returns ``{rule, margin, rows, flags}``."""
    TH = _lib().thresholds_mod()
    cats = [c for c in (trained or []) if c in TRANSLATE] + ["move_turn"]
    rows = []
    for es in sorted(set(student) | set(tstar)):
        A, B = student.get(es), tstar.get(es)
        for c in cats:
            for absdev in (False, True):
                base = {"id": f"student.watch.tracking.{'abs_dev' if absdev else 'ratio'}.{c}.es{es}",
                        "statistic": f"{'|ratio-1|' if absdev else 'ratio'} {c}", "row": f"suite es{es}",
                        "category": c, "env_seed": es, "watch": True, "margin": margin,
                        "direction": "<=" if absdev else ">=",
                        "key_path": f"aggregate.suite.by_category.{c}.ratio.median (per-episode hold ratios, paired)"}
                a = b = None
                if A is not None and B is not None:
                    a = TH._cat_stat([r for r in A if r["cat"] == c], "ratio", absdev)
                    b = TH._cat_stat([r for r in B if r["cat"] == c], "ratio", absdev)
                if a is None or b is None:
                    why = (f"no suite es{es} stamp on the {'student' if A is None else 'T*'}" if A is None or B is None
                           else f"no {c} episode with a ratio on the {'student' if a is None else 'T*'}")
                    rows.append({**base, "student": a, "tstar": b, "value": a, "line": None, "worse_by": None,
                                 "pass": False, "missing": True, "why": why})
                    continue
                worse = (a - b) if absdev else (b - a)
                rows.append({**base, "student": a, "tstar": b, "value": a,
                             "line": (b + margin) if absdev else (b - margin), "worse_by": worse,
                             "se_diff": TH.paired_suite_se(A, B, c, "ratio", absdev),
                             "pass": bool(worse <= margin + 1e-12), "missing": False})
    flags = [{"id": r["id"], "why": (f"tracking WATCH (+{margin:g} vs T*): no read ({r['why']})" if r.get("missing")
                                     else f"tracking WATCH: worse than T* by {r['worse_by']:.4f} > {margin:g}"),
              "value": r.get("value"), "line": r.get("line")} for r in rows if not r["pass"]]
    return {"rule": TRACKING_WATCH_RULE, "margin": margin, "rows": rows, "flags": flags}


def next_action(state: dict, max_failed: int = 2) -> dict:
    """The selection state machine: ``state`` = {ranked: [cand ids], blocks: {cand: 'PASS'|'FAIL'}}.
    -> {do: 'block', cand} | {do: 'selected', cand} | {do: 'halt', why}."""
    blocks = state.get("blocks") or {}
    for c, v in blocks.items():
        if v == "PASS":
            return {"do": "selected", "cand": c}
    failed = [c for c, v in blocks.items() if v == "FAIL"]
    if len(failed) >= max_failed:
        return {"do": "halt", "why": f"{len(failed)} failed full blocks ({failed})"}
    for c in state.get("ranked") or []:
        if c not in blocks:
            return {"do": "block", "cand": c}
    return {"do": "halt", "why": "every candidate's full block failed or none is left"}


def _lib():
    import importlib.util  # noqa: PLC0415

    if "tp_lib" in sys.modules:
        return sys.modules["tp_lib"]
    spec = importlib.util.spec_from_file_location("tp_lib", str(HERE / "tp_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tp_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--run-dir", required=True, type=Path)
    c.add_argument("--start", required=True, type=int)
    c.add_argument("--offsets", default=",".join(map(str, OFFSETS)))
    a = ap.parse_args(argv)
    print(json.dumps(candidate_checkpoints(a.run_dir, a.start, [int(x) for x in a.offsets.split(",")]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
