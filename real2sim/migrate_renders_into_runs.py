#!/usr/bin/env python3
"""migrate_renders_into_runs.py — STO-SCN-058 backfill.

Moves every scene-level `comparison_renders/<view>/<variant>.png` into
the pipeline run that produced it:

    <scene>/pipeline-<p>/run-<r>/renders/<view>.png
    <scene>/pipeline-<p>/run-<r>/renders/<view>.json   (settings sidecar)

Rationale (operator, 2026-06-10): what the runoff compares is the
*pipeline configuration* that produced the image; the scene-level
comparison_renders/ layout obscured that lineage. Per-view aggregation
is rate_renders' job at read time, not a storage layout.

Backfilled sidecars carry `provenance: "backfilled"` — engine and
resolution are the matrix-script defaults that were in effect
(BLENDER_WORKBENCH 1920×1080, every render to date); `mesh_source` was
not recorded at render time and is stated as unknown (T-002 — no
fabrication). Transform parameters are snapshotted from the run's
specification.json files (stable since the render).

Usage:
    python3 migrate_renders_into_runs.py [--store /var/krabby/scenes] [--dry-run]

Idempotent: scenes without comparison_renders/ are skipped; existing
destination files are never overwritten (hard error — investigate).
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path


def sidecar(run_dir: Path, view: str, views_json: Path) -> dict:
    vcam = None
    if views_json.is_file():
        try:
            views = json.loads(views_json.read_text())
            vcam = next((v for v in views.get("views", []) if v["name"] == view), None)
        except ValueError:
            pass
    params = {}
    for tdir in sorted(run_dir.glob("transform-*")):
        sp = tdir / "specification.json"
        if sp.is_file():
            try:
                params[tdir.name] = json.loads(sp.read_text()).get("parameters", {})
            except ValueError:
                params[tdir.name] = {"error": "spec unreadable"}
    return {
        "schema_version": "1",
        "view": view,
        "view_camera": vcam,
        "render": {
            "engine": "BLENDER_WORKBENCH",
            "width": 1920,
            "height": 1080,
            "mesh_source": None,
            "mesh_relpath": None,
            "provenance": "backfilled",
            "note": ("migrated from scene-level comparison_renders/ "
                     "(STO-SCN-058); engine/resolution are the matrix-script "
                     "defaults in effect for every render to date; "
                     "mesh_source was not recorded at render time"),
            "backfilled_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "produced_by": {
            "pipeline": run_dir.parent.name.removeprefix("pipeline-"),
            "run": run_dir.name.removeprefix("run-"),
            "transform_parameters": params,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/var/krabby/scenes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    store = Path(args.store)

    moved = errors = 0
    for cr in sorted(store.glob("*/comparison_renders")):
        scene_dir = cr.parent
        views_json = scene_dir / "cameras.json"
        for view_dir in sorted(d for d in cr.iterdir() if d.is_dir()):
            view = view_dir.name
            for png in sorted(view_dir.glob("*.png")):
                variant = png.stem
                pipeline, sep, run = variant.partition("--")
                if not sep:
                    print(f"ERROR: unrecognized variant label {variant} ({png})")
                    errors += 1
                    continue
                run_dir = scene_dir / f"pipeline-{pipeline}" / f"run-{run}"
                if not run_dir.is_dir():
                    print(f"ERROR: no run dir for {variant} ({png})")
                    errors += 1
                    continue
                dst = run_dir / "renders" / f"{view}.png"
                if dst.exists():
                    print(f"ERROR: refusing to overwrite {dst}")
                    errors += 1
                    continue
                print(f"{scene_dir.name}: {view}/{variant}.png -> "
                      f"pipeline-{pipeline}/run-{run}/renders/{view}.png")
                if not args.dry_run:
                    dst.parent.mkdir(exist_ok=True)
                    shutil.move(str(png), dst)
                    (dst.parent / f"{view}.json").write_text(
                        json.dumps(sidecar(run_dir, view, views_json), indent=2) + "\n")
                moved += 1
        if not args.dry_run:
            # remove now-empty view dirs + the comparison_renders root
            for view_dir in list(cr.iterdir()):
                if view_dir.is_dir() and not any(view_dir.iterdir()):
                    view_dir.rmdir()
            if not any(cr.iterdir()):
                cr.rmdir()
            else:
                print(f"NOTE: {cr} not empty after migration — left in place")
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}moved={moved} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
