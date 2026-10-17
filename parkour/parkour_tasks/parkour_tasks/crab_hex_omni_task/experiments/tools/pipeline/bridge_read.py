#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 bridge reader (CPU): is a NEW eval stamp the RECORDED one? (PLAN step 8 C1 / C2 bridges.)

Two comparisons, both pure on the stamps' files (numpy + json, no Isaac):

* ``full``      every ``raw/episode_*.npz`` key by key (``np.array_equal``, NaNs equal; a key the new harness ADDS is
                allowed only when named by ``--allow-new-key``) plus ``scenario_metrics.json`` ``aggregate`` /
                ``episodes`` / ``scenario_id`` as parsed JSON. The same test as the body-cmd campaign's
                ``bc_bitwise.py``, with the ignore list below. Use it for the walker row families (S1 s18 rows through
                ``gate_row.sh`` against the recorded stamps).
* ``aggregate`` ``scenario_metrics.json`` ``aggregate`` only (+ ``episodes`` when both stamps carry it). Use it for the
                stand rows (SH_T17's ``hwalkstop`` / ``hraise`` es1 and ``walkstop``) re-run THROUGH THE SH LAUNCHER:
                the launcher's stamps do not carry the walker row runner's file set, so PLAN's "compare aggregates, not
                files" applies (PLAN left-for-implementation item 10).

IGNORED BY DEFAULT: ``aggregate.suite.trained_categories``. A gate row passes ``--trained-categories`` (so Gate 0 can
tell a missing list from a real one); the recorded stamps predate that flag and carry ``null`` there, and
``report.suite_aggregate_arg`` writes the list into the suite block of the aggregate. Nothing else differs, so the key is
the one metadata field the bridge excludes (PLAN left-for-implementation item 3); ``--no-default-ignore`` compares it too
and ``--ignore PATH`` adds more dotted paths (an ignore is reported in the result, never silent).

    bridge_read.py --ref STAMP --new STAMP [--mode full|aggregate] [--ignore a.b.c ...] [--no-default-ignore]
                   [--allow-new-key K ...] [--json OUT]
Exit 0 when the stamps match, 1 otherwise.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

DEFAULT_IGNORE = ("aggregate.suite.trained_categories",)
SM_KEYS_FULL = ("aggregate", "episodes", "scenario_id")


def _drop(doc, dotted: str) -> bool:
    """Remove ``dotted`` from the nested dict ``doc`` (no-op, False, when a segment is absent)."""
    cur = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    if isinstance(cur, dict) and parts[-1] in cur:
        del cur[parts[-1]]
        return True
    return False


def _episodes(stamp: Path) -> dict:
    return {p.name: p for p in sorted((Path(stamp) / "raw").glob("episode_*.npz"))}


def metrics_diff(ref: dict, new: dict, keys, ignore) -> dict:
    """Keys of ``scenario_metrics.json`` that differ after ``ignore`` (dotted paths from the document root) are dropped
    from BOTH sides; ``ignored`` lists the paths that were present on at least one side."""
    a, b = copy.deepcopy(ref), copy.deepcopy(new)
    ignored = sorted({p for p in ignore for d in (a, b) if _drop(d, p)})
    diff = [k for k in keys if (k in a or k in b) and a.get(k) != b.get(k)]
    return {"diff_keys": diff, "ignored": ignored}


def compare(ref: Path, new: Path, *, mode: str = "full", ignore=DEFAULT_IGNORE, allow_new=()) -> dict:
    if mode not in ("full", "aggregate"):
        raise ValueError(f"bridge mode {mode!r} must be full | aggregate")
    ref, new = Path(ref), Path(new)
    out = {"ref": str(ref), "new": str(new), "mode": mode, "ignore": list(ignore), "metrics_equal": None,
           "metrics_diff_keys": [], "ignored_present": []}
    ma, mb = ref / "scenario_metrics.json", new / "scenario_metrics.json"
    if ma.is_file() and mb.is_file():
        keys = SM_KEYS_FULL if mode == "full" else ("aggregate",) + (
            ("episodes",) if "episodes" in json.loads(ma.read_text()) and "episodes" in json.loads(mb.read_text())
            else ())
        md = metrics_diff(json.loads(ma.read_text()), json.loads(mb.read_text()), keys, ignore)
        out.update(metrics_equal=not md["diff_keys"], metrics_diff_keys=md["diff_keys"], ignored_present=md["ignored"])
    else:
        out["missing_metrics"] = [str(p) for p in (ma, mb) if not p.is_file()]
    ok = out["metrics_equal"] is True
    if mode == "full":
        import numpy as np  # noqa: PLC0415

        er, en = _episodes(ref), _episodes(new)
        out.update(episodes_ref=len(er), episodes_new=len(en), missing_episodes=sorted(set(er) - set(en)),
                   extra_episodes=sorted(set(en) - set(er)), key_mismatch={}, value_mismatch={})
        allow = set(allow_new)
        for name in sorted(set(er) & set(en)):
            x, y = np.load(er[name], allow_pickle=False), np.load(en[name], allow_pickle=False)
            kx, ky = set(x.files), set(y.files)
            only_ref, only_new = sorted(kx - ky), sorted(ky - kx - allow)
            if only_ref or only_new:
                out["key_mismatch"][name] = {"only_ref": only_ref, "only_new": only_new}
            bad = [k for k in sorted(kx & ky)
                   if x[k].shape != y[k].shape or x[k].dtype != y[k].dtype
                   or not np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f")]
            if bad:
                out["value_mismatch"][name] = bad
        ok = (ok and not out["missing_episodes"] and not out["extra_episodes"] and not out["key_mismatch"]
              and not out["value_mismatch"] and out["episodes_ref"] > 0)
    out["bitwise"] = bool(ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--mode", choices=("full", "aggregate"), default="full")
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--no-default-ignore", action="store_true")
    ap.add_argument("--allow-new-key", action="append", default=[])
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    ignore = tuple(a.ignore) + (() if a.no_default_ignore else DEFAULT_IGNORE)
    res = compare(a.ref, a.new, mode=a.mode, ignore=ignore, allow_new=a.allow_new_key)
    if a.json:
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    print(f"bridge {a.mode}: match={res['bitwise']} metrics_equal={res['metrics_equal']} "
          f"diff_keys={res['metrics_diff_keys']} ignored={res['ignored_present']}"
          + (f" episodes {res.get('episodes_ref')}/{res.get('episodes_new')} "
             f"value_mismatch={len(res.get('value_mismatch', {}))} key_mismatch={len(res.get('key_mismatch', {}))}"
             if a.mode == "full" else ""))
    return 0 if res["bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
