#!/usr/bin/env python3
"""task_catalog.py — load + validate the task catalog (STO-SCN-070).

The catalog (`real2sim/tasks/*.json`) is the canonical machine-readable
form of the RECIPES.md phase catalog: one JSON Schema (draft 2020-12)
per task, with settings min/max/default and the executing image + code
ref in `x-task`. Vocabulary: these are **tasks** (operator decision,
EPI-SCN-PIPELINE-STUDIO); the store keeps `transform-NN-*` paths.

CLI:
    python3 real2sim/task_catalog.py list
    python3 real2sim/task_catalog.py show <task>
    python3 real2sim/task_catalog.py validate <task> '<settings-json>'
    python3 real2sim/task_catalog.py check-spec <specification.json> [<task>]

`check-spec` validates a historical store spec's `parameters` against a
task def: only keys the def declares are validated (historical specs
mix execution pins like `image`/`model` into parameters — those are
x-task territory, reported as uncovered, never failed). Round-trip DoD
for STO-SCN-070.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

TASKS_DIR = Path(__file__).parent / "tasks"

# historical spec `transform` name fragments -> catalog task names
SPEC_NAME_HINTS = [
    ("normalize", "normalize-photos"),
    ("frame-select-sharp", "select-sharp-frames"),
    ("pool-sharp", "select-sharp-frames"),
    ("frame-select-curated", "coverage-curation"),
    ("pool-sfm", "pool-sfm"),
    ("matcha", "matcha-reconstruction"),
    ("da3", "da3-infer"),
]


def load_catalog() -> dict[str, dict]:
    cat = {}
    for p in sorted(TASKS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        cat[d["title"]] = d
    return cat


def infer_task(spec: dict) -> str | None:
    name = spec.get("transform", "") or ""
    for frag, task in SPEC_NAME_HINTS:
        if frag in name:
            return task
    return None


def defaults_of(taskdef: dict) -> dict:
    return {k: p["default"] for k, p in taskdef.get("properties", {}).items()
            if "default" in p}


def validate_settings(taskdef: dict, settings: dict) -> list[str]:
    v = jsonschema.Draft202012Validator(taskdef)
    return [e.message for e in v.iter_errors(settings)]


def check_spec(spec_path: Path, task_name: str | None) -> int:
    spec = json.loads(spec_path.read_text())
    cat = load_catalog()
    task_name = task_name or infer_task(spec)
    if not task_name or task_name not in cat:
        print(f"UNKNOWN task for spec '{spec.get('transform')}' — pass the task name explicitly")
        return 2
    taskdef = cat[task_name]
    declared = set(taskdef.get("properties", {}))
    params = dict(spec.get("parameters", {}))
    # historical field mappings (pre-catalog vocabulary)
    if task_name == "da3-infer" and "infer_gs" in params and "mode" not in params:
        params["mode"] = "gs" if params.pop("infer_gs") else "nogs"
    covered = {k: v for k, v in params.items() if k in declared}
    uncovered = sorted(set(params) - declared)
    # validate only the covered subset; drop additionalProperties/required
    # gates (a historical spec is not obliged to set every setting)
    sub = dict(taskdef)
    sub.pop("required", None)
    sub = {**sub, "additionalProperties": True}
    errs = validate_settings(sub, covered)
    print(f"spec: {spec_path}")
    print(f"task: {task_name}")
    print(f"covered settings ({len(covered)}): {covered}")
    if uncovered:
        print(f"uncovered keys (execution pins / record-only): {uncovered}")
    if errs:
        print("INVALID:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("VALID")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    cmd = args[0]
    cat = load_catalog()
    if cmd == "list":
        for name, d in cat.items():
            xt = d["x-task"]
            op = " [operator]" if xt.get("operator") else ""
            img = xt.get("image") or "host"
            print(f"{name:28s} phase {xt['phase']:>2}  {img}{op}")
        return 0
    if cmd == "show":
        print(json.dumps(cat[args[1]], indent=2))
        return 0
    if cmd == "validate":
        taskdef = cat[args[1]]
        settings = {**defaults_of(taskdef), **json.loads(args[2])}
        errs = validate_settings(taskdef, settings)
        print(f"expanded settings: {settings}")
        if errs:
            print("INVALID:")
            for e in errs:
                print(f"  - {e}")
            return 1
        print("VALID")
        return 0
    if cmd == "check-spec":
        return check_spec(Path(args[1]), args[2] if len(args) > 2 else None)
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
