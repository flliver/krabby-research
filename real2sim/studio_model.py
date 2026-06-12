#!/usr/bin/env python3
"""studio_model.py — Studio read-model over the v4 store (HUG-SCN-005).

Post-migration (STO-SCN-080) this module is a thin compatibility layer
for the Studio UI over v4core: tasks (A) come from real2sim/tasks/
(v4 defs, transformed to the UI's form shape), graphs (D) from
real2sim/graphs/, runs (C/F-equivalent) from the v4 store scan, and
the leaderboard from scores.jsonl. The v2 reader was retired with the
store layout it read (see git history for the 57-run enumeration era).

CLI:
    python3 real2sim/studio_model.py scan [<scene>|all]
    python3 real2sim/studio_model.py leaderboard <scene>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
import v4core as v4  # noqa: E402

STORE = v4.STORE


def _ui_properties(taskdef: dict) -> dict:
    """v4 settings {class,…} -> the UI form-field shape."""
    props = {}
    for k, s in taskdef.get("settings", {}).items():
        if s.get("class") == "pin":
            continue
        d = s.get("default")
        ptype = ("integer" if isinstance(d, bool) is False and isinstance(d, int)
                 else "number" if isinstance(d, float)
                 else "array" if isinstance(d, list)
                 else "string")
        p = {"type": ptype, "description": f"[{s.get('class')}] {s.get('note', '')}".strip()}
        for src, dst in (("enum", "enum"), ("min", "minimum"), ("max", "maximum"),
                         ("default", "default")):
            if src in s:
                p[dst] = s[src]
        if s.get("class") == "frozen":
            p["frozen"] = True
        props[k] = p
    return props


def tasks() -> dict[str, dict]:
    out = {}
    for name, d in v4.tasks().items():
        out[name] = {
            "title": name,
            "description": d.get("description", ""),
            "properties": _ui_properties(d),
            "x-task": {
                "operator": any(i.get("ref") == "operator" for i in d.get("inputs", []))
                            or d.get("operator", False),
                "phase": d.get("algo"),
                "image": d.get("image"),
                "code_ref": d.get("algo"),
                "license_flag": d.get("license_flag"),
                "outputs": d.get("outputs", []),
            },
        }
    return out


def pipelines() -> dict[str, dict]:
    """Graphs (locked #4/#8) presented under the UI's pipeline shape."""
    return {g["name"]: {"name": g["name"], "description": g.get("description", ""),
                        "nodes": g["nodes"], "edges": g["edges"]}
            for g in v4.graphs().values()}


def instances() -> dict[str, dict]:
    return {d["name"]: d for d in
            (json.loads(p.read_text()) for p in sorted((REPO / "instances").glob("*.json")))}


def scenes() -> list[Path]:
    return sorted(d for d in STORE.iterdir()
                  if d.is_dir() and not d.name.startswith((".", "_")))


def runs(scene: str | None = None) -> list[dict]:
    """UI rows: one per representation, meshes/renders summarized."""
    rows = []
    for sd in ([STORE / scene] if scene else scenes()):
        if not (sd / "images").is_dir():
            continue
        sc = v4.scan_scene(sd.name)
        for rep in sc["representations"]:
            n_renders = sum(len(m["renders"]) for m in rep["meshes"]) + \
                sum(len(c["renders"]) for m in rep["meshes"] for c in m["conditioned"])
            rows.append({
                "scene": sd.name,
                "variant": rep.get("legacy_variant") or rep["identity"],
                "kind": rep["kind"],
                "identity": rep["identity"],
                "algo": rep["algo"],
                "settings": rep["settings"],
                "migrated": rep["migrated"],
                "deliverable_eligible": rep["deliverable_eligible"],
                "license_flags": rep["license_flags"],
                "meshes": [{"method": m["method"], "identity": m["identity"],
                            "conditioned": [c["identity"] for c in m["conditioned"]]}
                           for m in rep["meshes"]],
                "renders": n_renders,
                "record": True,
            })
    return rows


def leaderboard(scene: str) -> dict:
    return v4.leaderboard(scene)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "scan":
        target = None if len(args) < 2 or args[1] == "all" else args[1]
        for r in runs(target):
            el = "✓" if r["deliverable_eligible"] else "✗NC"
            print(f"{r['scene']:14s} {r['variant']:34s} {r['kind']:7s} {el:3s} "
                  f"meshes={len(r['meshes'])} renders={r['renders']}")
        return 0
    if args[0] == "leaderboard":
        for row in leaderboard(args[1])["rows"]:
            print(f"{row['mean_rank']:6.2f}  {row['label']}")
        return 0
    print(f"unknown command: {args[0]}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
