#!/usr/bin/env python3
"""studio_model.py — A–F read-side adapters over the scene store (STO-SCN-071).

Presents the existing store (spec/results/run dirs, render-variant
runs, rankings.jsonl) through the Pipeline Studio taxonomy WITHOUT
moving or rewriting a single store file:

    A task              real2sim/tasks/<task>.json      (catalog, 070)
    D pipeline          real2sim/pipelines/<name>.json  (076)
    E pipeline_instance real2sim/instances/<name>.json  (076) or derived
    B task_instance     transform-NN-*/specification.json parameters
    C task_run          transform-NN-*/{specification,results}.json
    F pipeline_run      pipeline-<p>/run-<r>/ dir (+ run_record.json when present)

Unknowable fields surface as the string "unknown" — never guessed
(T-002). Strictly read-only: this module never writes into the store.

CLI:
    python3 real2sim/studio_model.py scan [<scene>|all] [--json]
    python3 real2sim/studio_model.py run <scene> <pipeline> <run> [--json]
    python3 real2sim/studio_model.py leaderboard <scene> [--json]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

STORE = Path("/var/krabby/scenes")
REPO = Path(__file__).parent
UNKNOWN = "unknown"


# ---------- A / D / E (repo-side) ----------

def tasks() -> dict[str, dict]:
    return {d["title"]: d for d in
            (json.loads(p.read_text()) for p in sorted((REPO / "tasks").glob("*.json")))}


def pipelines() -> dict[str, dict]:
    return {d["name"]: d for d in
            (json.loads(p.read_text()) for p in sorted((REPO / "pipelines").glob("*.json")))}


def instances() -> dict[str, dict]:
    return {d["name"]: d for d in
            (json.loads(p.read_text()) for p in sorted((REPO / "instances").glob("*.json")))}


# ---------- C / F (store-side, read-only) ----------

@dataclass
class TaskRun:                      # C
    transform: str
    task: str                       # catalog task name or "unknown"
    settings: dict                  # B (the instance-level settings as recorded)
    status: str
    host: str
    image: str
    image_digest: str
    duration_s: object
    outputs: int


@dataclass
class PipelineRun:                  # F
    scene: str
    pipeline: str
    run: str
    variant: str
    kind: str                       # "full" | "render-variant"
    source_run: object              # for render-variants
    task_runs: list = field(default_factory=list)
    renders: list = field(default_factory=list)
    notes: str = ""
    record: object = None           # run_record.json content when present (v3)


_TASK_HINTS = [  # spec `transform`/pipeline fragments -> catalog task
    ("matcha", "matcha-reconstruction"), ("da3", "da3-infer"),
    ("normalize", "normalize-photos"), ("sharp", "select-sharp-frames"),
    ("curated", "coverage-curation"), ("pool-sfm", "pool-sfm"),
]


def _infer_task(name: str) -> str:
    for frag, task in _TASK_HINTS:
        if frag in name:
            return task
    return UNKNOWN


def _task_run(tdir: Path) -> TaskRun:
    spec = res = {}
    if (tdir / "specification.json").exists():
        spec = json.loads((tdir / "specification.json").read_text())
    if (tdir / "results.json").exists():
        res = json.loads((tdir / "results.json").read_text())
    env = res.get("environment", {})
    cont = env.get("container", {})
    return TaskRun(
        transform=tdir.name,
        task=_infer_task(spec.get("transform", tdir.name)),
        settings=spec.get("parameters", UNKNOWN if not spec else {}),
        status=res.get("status", UNKNOWN),
        host=res.get("host", UNKNOWN),
        image=cont.get("image", spec.get("parameters", {}).get("image", UNKNOWN)
                       if isinstance(spec.get("parameters"), dict) else UNKNOWN),
        image_digest=cont.get("digest", UNKNOWN),
        duration_s=res.get("duration_s", UNKNOWN),
        outputs=len(res.get("outputs", [])),
    )


def pipeline_runs(scene_dir: Path) -> list[PipelineRun]:
    out = []
    for pdir in sorted(scene_dir.glob("pipeline-*")):
        pname = pdir.name.removeprefix("pipeline-")
        for rdir in sorted(pdir.glob("run-*")):
            rname = rdir.name.removeprefix("run-")
            run_meta = {}
            if (rdir / "run.json").exists():
                run_meta = json.loads((rdir / "run.json").read_text())
            tdirs = sorted(rdir.glob("transform-*"))
            kind = "full" if tdirs else "render-variant"
            record = None
            if (rdir / "run_record.json").exists():
                record = json.loads((rdir / "run_record.json").read_text())
            out.append(PipelineRun(
                scene=scene_dir.name, pipeline=pname, run=rname,
                variant=f"{pname}--{rname}", kind=kind,
                source_run=run_meta.get("source_run"),
                task_runs=[_task_run(t) for t in tdirs],
                renders=sorted(p.stem for p in (rdir / "renders").glob("*.png"))
                        if (rdir / "renders").is_dir() else [],
                notes=run_meta.get("notes", ""),
                record=record,
            ))
    return out


# ---------- scores (read-time join on rankings.jsonl — T-023) ----------

def rankings(scene_dir: Path) -> list[dict]:
    f = scene_dir / "rankings.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def leaderboard(scene_dir: Path) -> dict:
    """Per view: latest ranking submission wins (re-rank supersedes).
    Returns {view: {variant: rank}} + aggregate mean rank per variant."""
    per_view: dict[str, dict] = {}
    for entry in rankings(scene_dir):        # file order == submission order
        per_view[entry["view"]] = entry["ranks"]
    agg: dict[str, list] = {}
    for ranks in per_view.values():
        for variant, rank in ranks.items():
            agg.setdefault(variant, []).append(rank)
    mean = {v: round(sum(r) / len(r), 2) for v, r in agg.items()}
    return {"views": per_view,
            "aggregate_mean_rank": dict(sorted(mean.items(), key=lambda kv: kv[1]))}


# ---------- CLI ----------

def scenes() -> list[Path]:
    return sorted(d for d in STORE.iterdir()
                  if d.is_dir() and not d.name.startswith((".", "_")))


def main() -> int:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args:
        print(__doc__)
        return 2
    cmd = args[0]
    if cmd == "scan":
        target = args[1] if len(args) > 1 else "all"
        dirs = scenes() if target == "all" else [STORE / target]
        all_runs = [r for d in dirs for r in pipeline_runs(d)]
        if as_json:
            print(json.dumps([asdict(r) for r in all_runs], indent=2))
        else:
            for r in all_runs:
                tr = ",".join(f"{t.task}:{t.status}" for t in r.task_runs) or f"(<- {r.source_run})"
                rec = " v3" if r.record else ""
                print(f"{r.scene:14s} {r.variant:32s} {r.kind:14s} renders={len(r.renders)}{rec}  {tr}")
            print(f"-- {len(all_runs)} pipeline_runs")
        return 0
    if cmd == "run":
        scene, pipeline, run = args[1:4]
        for r in pipeline_runs(STORE / scene):
            if r.pipeline == pipeline and r.run == run:
                print(json.dumps(asdict(r), indent=2))
                return 0
        print("not found")
        return 1
    if cmd == "leaderboard":
        lb = leaderboard(STORE / args[1])
        if as_json:
            print(json.dumps(lb, indent=2))
        else:
            for variant, mean in lb["aggregate_mean_rank"].items():
                print(f"{mean:6.2f}  {variant}")
        return 0
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
