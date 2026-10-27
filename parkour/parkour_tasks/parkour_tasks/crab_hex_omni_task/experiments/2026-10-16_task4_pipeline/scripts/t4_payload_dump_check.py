#!/usr/bin/env python3
"""M1a (PLAN step 3) config-dump checks for the payload knobs, CPU only, on dumps written by ``t4_dump_cfgs.py``.

Two checks, one JSON record:

1. **Knob off** (``--before DIR --after DIR``): every ``<target>.json`` in ``--before`` is byte-identical to the same
   file in ``--after`` (sha256 per file), and no knob-off dump carries an ``events.payload_reset`` term.
2. **Armed** (``--armed NAME=OFF_JSON:ARMED_JSON:KG_LO:KG_HI:ZERO_FRAC:COM_XY:COM_Z``, repeatable): the armed dump
   holds ``events.payload_reset`` with the expected func / mode / params, and differs from its knob-off twin ONLY under
   ``<task>/env/events/payload_reset`` and in the ``KRABBY_PAYLOAD_*`` keys of ``_environ``.

    t4_payload_dump_check.py --before DIR --after DIR [--armed SPEC ...] [--refused NAME=LOG ...] --json OUT

``--refused NAME=LOG`` asserts that a dump whose payload knobs are incomplete failed loudly (the log names the
refusal and no JSON was written next to it). Exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FUNC = "parkour_tasks.crab_hex_omni_task.mdp.omni_events:payload_reset"
BOX = [0.7112, 1.2192, 0.3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diff_paths(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}")
            else:
                _diff_paths(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_paths(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)
    return out


def _tasks(js: dict) -> list[str]:
    return [k for k in js if k != "_environ"]


def knob_off(before: Path, after: Path) -> dict:
    rows, ok = {}, True
    for fb in sorted(before.glob("*.json")):
        fa = after / fb.name
        row = {"before_sha256": _sha(fb), "after_sha256": _sha(fa) if fa.is_file() else None}
        row["byte_identical"] = fa.is_file() and fb.read_bytes() == fa.read_bytes()
        js = json.loads(fa.read_text()) if fa.is_file() else {}
        row["payload_term_present"] = any(
            "payload_reset" in (js[t].get("env", {}).get("events") or {}) for t in _tasks(js))
        row["ok"] = row["byte_identical"] and not row["payload_term_present"]
        ok = ok and row["ok"]
        rows[fb.stem] = row
    return {"targets": len(rows), "all_byte_identical": ok, "per_target": rows}


def armed(spec: str) -> dict:
    name, rest = spec.split("=", 1)
    off_p, arm_p, lo, hi, zf, xy, z = rest.split(":")
    off, arm = json.loads(Path(off_p).read_text()), json.loads(Path(arm_p).read_text())
    (task,) = _tasks(arm)
    term = (arm[task]["env"].get("events") or {}).get("payload_reset")
    want_params = {"kg_range": [float(lo), float(hi)], "zero_frac": float(zf), "com_xy": float(xy),
                   "com_z": float(z), "box_size": BOX}
    checks = {}
    checks["term present"] = term is not None
    term = term or {}
    checks["func"] = term.get("func") == FUNC
    checks["mode reset"] = term.get("mode") == "reset"
    params = term.get("params") or {}
    for k, v in want_params.items():
        checks[f"params.{k} == {v}"] = params.get(k) == v
    asset = params.get("asset_cfg") or {}
    checks["params.asset_cfg robot/body"] = asset.get("name") == "robot" and asset.get("body_names") == "body"
    checks["params keys"] = set(params) == {"asset_cfg", *want_params}
    diffs = _diff_paths({k: v for k, v in off.items() if k != "_environ"},
                        {k: v for k, v in arm.items() if k != "_environ"})
    stray = [d for d in diffs if not d.startswith(f"/{task}/env/events/payload_reset")]
    checks["only events.payload_reset differs"] = not stray and bool(diffs)
    env_diff = sorted(set(_diff_paths(off.get("_environ", {}), arm.get("_environ", {}))))
    checks["_environ differs only in KRABBY_PAYLOAD_*"] = all(d.startswith("/KRABBY_PAYLOAD_") for d in env_diff)
    return {"name": name, "task": task, "off": off_p, "armed": arm_p, "term": term,
            "non_payload_diffs": stray[:20], "environ_diffs": env_diff, "checks": checks,
            "ok": all(checks.values())}


def refused(spec: str) -> dict:
    name, log = spec.split("=", 1)
    text = Path(log).read_text(errors="ignore")
    js = Path(log).with_suffix(".json")
    checks = {"no dump written": not js.exists(),
              "log names the refusal": "KRABBY_PAYLOAD_KG='0:90' needs" in text and "ValueError" in text}
    return {"name": name, "log": log, "checks": checks, "ok": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--armed", action="append", default=[])
    ap.add_argument("--refused", action="append", default=[])
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rec = {"knob_off": knob_off(Path(a.before), Path(a.after)),
           "armed": [armed(s) for s in a.armed], "refused": [refused(s) for s in a.refused]}
    rec["ok"] = rec["knob_off"]["all_byte_identical"] and all(r["ok"] for r in rec["armed"] + rec["refused"])
    Path(a.json).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"knob-off: {rec['knob_off']['targets']} targets, byte-identical={rec['knob_off']['all_byte_identical']}")
    for r in rec["armed"] + rec["refused"]:
        bad = [k for k, v in r["checks"].items() if not v]
        print(f"{r['name']}: {'PASS' if r['ok'] else 'FAIL ' + str(bad)}")
    print(f"M1A DUMP CHECK ok={rec['ok']}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
