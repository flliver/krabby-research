#!/usr/bin/env python3
"""v4job.py — job runner, first slice: materialize missing RENDERS
(HUG-SCN-005 locked #4: jobs materialize; existing identities NOOP).

For every mesh artifact (meshify/*/<MID>, condition/<CID>) in a scene
and every canonical view slot, the render identity is computed
(locked #7: keyed on VIEW CONTENT, never the set). Missing → render
via Blender headless (build_blender_scene.py, the validated path);
existing → NOOP. One job record per scene invocation (locked #8).

    python3 real2sim/v4job.py render-missing <scene>|all [--dry-run]

This is exactly how `matcha--*-tetra`-class outputs appear for scenes
that never had them: represent/meshify already exist (migrated) and
NOOP; only the render node executes. Local Blender — no GPU, no fleet.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
BUILD = Path(__file__).parent / "build_blender_scene.py"
RSET = {"engine": "BLENDER_WORKBENCH", "resolution": [1920, 1080]}
ALGO = "render-workbench@0"


def rep_camera_paths(scene_dir: Path, rep_dir: Path):
    """represent dir -> (cameras.json, oriented.json) via its metadata."""
    md = json.loads((rep_dir / "metadata.json").read_text())
    subset = md.get("resolved_inputs", {}).get("subset")
    solve = md.get("resolved_inputs", {}).get("cameras")
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    cands = [base / str(solve)] if solve else sorted(base.glob("*/")) if base.is_dir() else []
    for c in cands:
        cams = c / "cameras.json"
        ods = sorted(c.glob("orient/*/oriented.json"))
        if cams.exists() and ods:
            return cams, ods[0]
    return None, None


def mesh_targets(rep_dir: Path):
    """All renderable mesh dirs under a representation."""
    for mdir in sorted(rep_dir.glob("meshify/*/*/")):
        if (mdir / "mesh.ply").exists():
            yield mdir
        for cdir in sorted(mdir.glob("condition/*/")):
            if (cdir / "mesh.ply").exists():
                yield cdir


def render_one(scene_dir: Path, mesh_dir: Path, slot: str, view_content: dict,
               cams: Path, oriented: Path, out_dir: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        views_json = Path(td) / "views.json"
        views_json.write_text(json.dumps(
            {"schema_version": 5, "views": [dict(view_content, name=slot)]}, indent=2))
        blend = Path(td) / "tmp.blend"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [BLENDER, "--background", "--python", str(BUILD), "--",
               "--mesh", str(mesh_dir / "mesh.ply"),
               "--cameras-original", str(cams),
               "--cameras-oriented", str(oriented),
               "--output", str(blend),
               "--view-camera-pose", str(views_json),
               "--view-name", slot,
               "--render-output", str(out_dir / "render.png"),
               "--render-width", "1920", "--render-height", "1080",
               "--render-engine", "BLENDER_WORKBENCH"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = (out_dir / "render.png").exists()
        if not ok:
            (out_dir / "render-error.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])
        elif (out_dir / "render-error.log").exists():
            (out_dir / "render-error.log").unlink()   # earlier attempt's log
        return ok


def run_scene(scene: str, dry: bool) -> dict:
    sdir = v4.STORE / scene
    sc = v4.Scene(scene)
    try:
        slots = sc.resolve("canonical")
    except (FileNotFoundError, KeyError):
        return {"scene": scene, "skipped": "no canonical viewset"}
    stats = {"scene": scene, "noop": 0, "rendered": 0, "failed": 0, "nodes": []}
    for rep_dir in sorted(sdir.glob("represent/*/*/")):
        if not (rep_dir / "metadata.json").exists():
            continue
        cams, oriented = rep_camera_paths(sdir, rep_dir)
        for mesh_dir in mesh_targets(rep_dir):
            mid = mesh_dir.name
            for slot in slots:
                vc = json.loads((sdir / "views" / slot / "view.json").read_text())
                vh = sc.view_content_hash(slot)
                rid = v4.identity_hash({"mesh": mid, "view_content": vh}, RSET, ALGO)
                out_dir = mesh_dir / "renders" / rid
                if (out_dir / "render.png").exists():
                    stats["noop"] += 1
                    continue
                if cams is None:
                    stats["nodes"].append(f"SKIP {mesh_dir.relative_to(sdir)}: no cameras/orient")
                    continue
                if dry:
                    stats["nodes"].append(f"WOULD-RENDER {mesh_dir.relative_to(sdir)} × {slot}")
                    stats["rendered"] += 1
                    continue
                ok = render_one(sdir, mesh_dir, slot, vc, cams, oriented, out_dir)
                if ok:
                    v4.write_metadata(out_dir, task="render", algo=ALGO, identity=rid,
                                      resolved_inputs={"mesh": mid, "view_content": vh},
                                      settings=RSET, mechanism="job",
                                      extra={"view_slot": slot})
                    stats["rendered"] += 1
                    stats["nodes"].append(f"RENDERED {mesh_dir.relative_to(sdir)} × {slot}")
                else:
                    stats["failed"] += 1
                    stats["nodes"].append(f"FAILED {mesh_dir.relative_to(sdir)} × {slot}")
    if not dry and (stats["rendered"] or stats["failed"]):
        jd = sc.job_dir()
        (jd / "job.json").write_text(json.dumps({
            "schema": 4, "graph": "render-missing", "mechanism": "job",
            "bindings": {"scene": scene, "viewset": "canonical (resolved per slot)"},
            "outcome": {k: stats[k] for k in ("noop", "rendered", "failed")},
            "nodes": stats["nodes"],
            "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2) + "\n")
    return stats


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) < 2 or args[0] != "render-missing":
        print(__doc__)
        return 2
    scenes = sorted(d.name for d in v4.STORE.iterdir()
                    if d.is_dir() and not d.name.startswith((".", "_"))) \
        if args[1] == "all" else [args[1]]
    rc = 0
    for s in scenes:
        st = run_scene(s, dry)
        if st.get("skipped"):
            print(f"{s}: {st['skipped']}")
            continue
        print(f"{s}: NOOP {st['noop']}, {'would render' if dry else 'rendered'} "
              f"{st['rendered']}, failed {st['failed']}")
        rc |= 1 if st["failed"] else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
