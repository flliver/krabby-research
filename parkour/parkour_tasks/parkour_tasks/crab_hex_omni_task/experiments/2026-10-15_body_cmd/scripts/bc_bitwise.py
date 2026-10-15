#!/usr/bin/env python3
"""Step 3 G0 (CPU): is a new-code eval stamp BITWISE the recorded one on every raw key?

Compares every ``raw/episode_*.npz`` of two stamps key by key with ``np.array_equal`` (NaNs equal), and
``scenario_metrics.json`` as parsed JSON. Keys present in only one stamp are listed separately: a key the new harness
ADDS (``obs_cmd``, ``adapter_face``) is allowed on the new side only when ``--allow-new-key`` names it.

    bc_bitwise.py --ref STAMP --new STAMP [--allow-new-key K ...] [--json OUT]
Exit 0 when bitwise, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _episodes(stamp: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted((stamp / "raw").glob("episode_*.npz"))}


def compare(ref: Path, new: Path, allow_new: set[str]) -> dict:
    out = {"ref": str(ref), "new": str(new), "episodes_ref": 0, "episodes_new": 0, "missing_episodes": [],
           "extra_episodes": [], "key_mismatch": {}, "value_mismatch": {}, "metrics_equal": None}
    er, en = _episodes(ref), _episodes(new)
    out["episodes_ref"], out["episodes_new"] = len(er), len(en)
    out["missing_episodes"] = sorted(set(er) - set(en))
    out["extra_episodes"] = sorted(set(en) - set(er))
    for name in sorted(set(er) & set(en)):
        a, b = np.load(er[name], allow_pickle=False), np.load(en[name], allow_pickle=False)
        ka, kb = set(a.files), set(b.files)
        only_ref, only_new = sorted(ka - kb), sorted(kb - ka - allow_new)
        if only_ref or only_new:
            out["key_mismatch"][name] = {"only_ref": only_ref, "only_new": only_new}
        bad = []
        for k in sorted(ka & kb):
            x, y = a[k], b[k]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                bad.append(k)
        if bad:
            out["value_mismatch"][name] = bad
    ma, mb = ref / "scenario_metrics.json", new / "scenario_metrics.json"
    if ma.is_file() and mb.is_file():
        # run_meta carries shas, timestamps and paths: compared by the caller, not here.
        ja, jb = json.loads(ma.read_text()), json.loads(mb.read_text())
        diff = [k for k in ("aggregate", "episodes", "scenario_id") if ja.get(k) != jb.get(k)]
        out["metrics_equal"] = not diff
        out["metrics_diff_keys"] = diff
    out["bitwise"] = (not out["missing_episodes"] and not out["extra_episodes"] and not out["key_mismatch"]
                      and not out["value_mismatch"] and out["episodes_ref"] > 0 and out["metrics_equal"] is not False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--allow-new-key", action="append", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = compare(Path(a.ref), Path(a.new), set(a.allow_new_key))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    n_bad = len(res["value_mismatch"]) + len(res["key_mismatch"])
    print(f"bitwise={res['bitwise']} episodes {res['episodes_ref']}/{res['episodes_new']} "
          f"mismatched_episodes={n_bad} metrics_equal={res['metrics_equal']}")
    for name, keys in list(res["value_mismatch"].items())[:5]:
        print(f"  {name}: {keys[:8]}")
    return 0 if res["bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
