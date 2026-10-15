#!/usr/bin/env python3
"""Step 1/3 inertness (CPU): knob-off config dumps after the body-command code == the HEAD dumps.

For every ``<name>.json`` in ``--before``, the same file in ``--after`` must hold an identical ``{task: {env, agent}}``
(``_environ`` compared separately: the process env may differ only in keys listed with ``--allow-env``). Prints the
first differing paths per target.

    bc_dump_compare.py --before DIR --after DIR [--json OUT] [--allow-env K ...]
Exit 0 when every target matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def diff_paths(a, b, path="", out=None, limit=40):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k} ({'only after' if k not in a else 'only before'})")
            else:
                diff_paths(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} -> {str(b)[:80]!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-env", action="append", default=[])
    a = ap.parse_args()
    before, after = Path(a.before), Path(a.after)
    res, ok = {}, True
    for fb in sorted(before.glob("*.json")):
        fa = after / fb.name
        if not fa.is_file():
            res[fb.stem] = ["missing after"]
            ok = False
            continue
        jb, ja = json.loads(fb.read_text()), json.loads(fa.read_text())
        eb = {k: v for k, v in jb.pop("_environ", {}).items() if k not in a.allow_env}
        ea = {k: v for k, v in ja.pop("_environ", {}).items() if k not in a.allow_env}
        d = diff_paths(jb, ja) + [f"_environ{p}" for p in diff_paths(eb, ea)]
        res[fb.stem] = d
        ok = ok and not d
        print(f"{fb.stem:24s} {'IDENTICAL' if not d else f'{len(d)} diffs'}")
        for p in d[:10]:
            print(f"    {p}")
    if a.json:
        Path(a.json).write_text(json.dumps({"identical": ok, "diffs": res}, indent=2) + "\n")
    print(f"ALL IDENTICAL: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
