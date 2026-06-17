"""STO-SCN-150 — ingest-scene pipeline orchestration (EPI-SCN-SCENE-MANAGER).

Drives a canonicalized scene through the canonical pipeline (RECIPES.md /
knowledge/scene-processing) to a scouted + meshed state:

    precull(--set-primary) → solve → covis → scout(DA3 gaussian) → reconstruct-da3(DA3 mesh)

Each phase is a ``v4exec`` subprocess on the chosen ``--host``. Only ONE id
threads through (the solve), resolved from the STORE (newest ``cameras/*`` under
the primary subset) — NOT by parsing stdout. Writes ``pipeline_status.json``
(phase + pct + log tail) for the UI to poll. Phases are idempotent/NOOP where
content already exists (RECIPES), so a failed run is safe to re-run.

``dry_run`` prints the exact command plan WITHOUT executing — the pre-flight an
operator eyeballs before committing GPU. The REAL run needs ssh + GPU + docker
on the host and is operator-verified (T-020).

select (best-N view selection) is intentionally NOT in this flow — that is
EPI-SCN-AUTO-SUBSET-SELECT / the view-selection step, not the default
ingest-scene pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4EXEC = HERE / "v4exec.py"


def gpu_hosts() -> list[str]:
    """Configured GPU hosts. Default ['tbeeprz'] (the M11 fleet box); override
    with env KRABBY_GPU_HOSTS=comma,separated."""
    raw = os.environ.get("KRABBY_GPU_HOSTS", "tbeeprz")
    return [h.strip() for h in raw.split(",") if h.strip()]


def resolve_primary_subset(scene_dir: Path) -> str | None:
    """The subset id the 'primary' symlink points at (the PRIMARY subset)."""
    link = scene_dir / "images" / "subsets" / "primary"
    if link.is_symlink():
        return Path(os.readlink(link)).name
    if link.is_dir():
        return "primary"
    return None


def resolve_latest_solve(scene_dir: Path, subset: str | None = None) -> str | None:
    """Newest camera solve (dir with cameras.json) under the primary subset."""
    subset = subset or resolve_primary_subset(scene_dir)
    if not subset:
        return None
    cams = scene_dir / "images" / "subsets" / subset / "cameras"
    if not cams.is_dir():
        return None
    solves = [c for c in cams.iterdir() if c.is_dir() and (c / "cameras.json").exists()]
    if not solves:
        return None
    return max(solves, key=lambda c: c.stat().st_mtime).name


# Phase plan. `needs_solve` phases get --solve <resolved id> appended at run time.
PHASES = [
    {"key": "precull",  "label": "Pre-cull → PRIMARY subset",
     "args": ["precull", "--set-primary"], "host": False, "needs_solve": False},
    {"key": "solve",    "label": "Spine solve (FastMap poses)",
     "args": ["solve"], "host": True, "needs_solve": False},
    {"key": "covis",    "label": "Co-visibility validity gate",
     "args": ["covis"], "host": True, "needs_solve": True},
    {"key": "scout",    "label": "DA3 scout gaussian",
     "args": ["scout"], "host": True, "needs_solve": True},
    {"key": "mesh",     "label": "DA3 reconstruct (mesh)",
     "args": ["reconstruct-da3", "--sfm", "posed"], "host": True, "needs_solve": False},
]


def build_command(phase: dict, scene: str, host: str, solve: str | None) -> list[str]:
    """Assemble the v4exec command line for one phase."""
    cmd = [sys.executable, str(V4EXEC), phase["args"][0], scene]
    cmd += phase["args"][1:]
    if phase["host"]:
        cmd += ["--host", host]
    if phase["needs_solve"]:
        cmd += ["--solve", solve or "UNRESOLVED"]
    return cmd


def plan(scene: str, host: str, scene_dir: Path | None = None) -> list[dict]:
    """The full command plan (for dry-run preview). Resolves the solve id if it
    already exists; otherwise marks it <after-solve> (filled in at run time)."""
    sd = scene_dir or (Path(os.environ.get("KRABBY_SCENES_ROOT", "/var/krabby/scenes")) / scene)
    solve = resolve_latest_solve(sd) or "<after-solve>"
    return [{"key": p["key"], "label": p["label"],
             "cmd": build_command(p, scene, host, solve)} for p in PHASES]


def run_pipeline(scene_dir: Path, host: str, *, dry_run: bool = False,
                 status_cb=None, tail_lines: int = 40) -> dict:
    """Run the phases in order, stopping on the first failure. `status_cb(rec)`
    is called after every phase transition with the running status record."""
    scene = scene_dir.name
    phases = [{"key": p["key"], "label": p["label"], "status": "pending"} for p in PHASES]
    rec = {"status": "running", "host": host, "dry_run": dry_run,
           "n_phases": len(PHASES), "phase_idx": 0, "phase": None,
           "phases": phases, "log_tail": "", "solve": None}

    def emit():
        if status_cb:
            status_cb(dict(rec))

    emit()
    solve = resolve_latest_solve(scene_dir)
    for i, p in enumerate(PHASES):
        rec["phase_idx"] = i
        rec["phase"] = p["key"]
        phases[i]["status"] = "running"
        if p["needs_solve"]:
            solve = solve or resolve_latest_solve(scene_dir)
        cmd = build_command(p, scene, host, solve)
        phases[i]["cmd"] = cmd
        emit()

        if dry_run:
            phases[i]["status"] = "planned"
            emit()
            continue

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        except Exception as e:   # noqa: BLE001 — surface launch failure
            phases[i]["status"] = "error"
            rec["status"] = "error"
            rec["log_tail"] = f"{type(e).__name__}: {e}"
            emit()
            return rec
        out = (r.stdout or "") + (r.stderr or "")
        rec["log_tail"] = "\n".join(out.splitlines()[-tail_lines:])
        if r.returncode != 0:
            phases[i]["status"] = "error"
            phases[i]["rc"] = r.returncode
            rec["status"] = "error"
            emit()
            return rec
        phases[i]["status"] = "done"
        # the solve id only exists AFTER the solve phase runs
        if p["key"] == "solve":
            solve = resolve_latest_solve(scene_dir)
            rec["solve"] = solve
        emit()

    rec["status"] = "done"
    rec["phase"] = None
    emit()
    return rec
