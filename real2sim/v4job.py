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
import os
import shutil
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


def publish_progress(scene: str, job_id: str, node: str, status: str, pct: int) -> None:
    """Best-effort retained MQTT progress to krabby/jobs/<scene>/<job_id>
    (STO-SCN-088 feedback channel; T-021 level-triggered). The job.json
    record is the source of truth — a missing broker/client is a silent
    no-op, never a job failure. Mirrors lib_progress.sh's mqtt backend."""
    host = os.environ.get("KRABBY_MQTT_HOST")
    if not host or not shutil.which("mosquitto_pub"):
        return
    payload = json.dumps({
        "node": node, "status": status, "pct": pct,
        "host": os.uname().nodename.split(".")[0],
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    try:
        subprocess.run(
            ["mosquitto_pub", "-r", "-h", host,
             "-p", os.environ.get("KRABBY_MQTT_PORT", "1883"),
             "-t", f"krabby/jobs/{scene}/{job_id}", "-m", payload],
            capture_output=True, timeout=2)
    except (subprocess.SubprocessError, OSError):
        pass


def rep_camera_paths(scene_dir: Path, rep_dir: Path):
    """represent dir -> (cameras.json, oriented.json).

    MIGRATED reps first check their own origin-data/ — each legacy run
    solved its OWN cameras (own gauge!), and the same-identity solve
    collapse (correct for the model) kept only one run's copy at the
    subset solve dir. Rendering a legacy mesh in a sibling run's gauge
    skews it (operator-caught on 009/010/003). The run's true gauge
    survives in its origin-data sweep."""
    import hashlib

    md0 = json.loads((rep_dir / "metadata.json").read_text())
    own_cams = rep_dir / "origin-data" / "mast3r_sfm" / "cameras.json"
    own_ori = rep_dir / "origin-data" / "oriented" / "oriented_cameras.json"
    # re-grounded reps (Option A, STO-SCN-089): meshes live in the rep's
    # recorded canonical gauge — the camera mapping MUST target that file
    cg = md0.get("canonical_gauge")
    if cg:
        ori = scene_dir / cg
        if ori.exists():
            if own_cams.exists():
                return own_cams, ori
            subset = md0.get("resolved_inputs", {}).get("subset")
            base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
            pool = sorted(base.glob("*/cameras.json")) if base.is_dir() else []
            want_sha = None
            orr = rep_dir / "origin-results.json"
            if orr.exists():
                for o in json.loads(orr.read_text()).get("outputs", []):
                    if isinstance(o, dict) and o.get("path", "").endswith("mast3r_sfm/cameras.json"):
                        want_sha = o.get("sha256")
            pool += sorted(base.glob("*/origin-dup-cameras.json")) if base.is_dir() else []
            cams = pool[0] if pool else None
            if want_sha:
                for p in pool:
                    if hashlib.sha256(p.read_bytes()).hexdigest() == want_sha:
                        cams = p
                        break
            if cams:
                return cams, ori
    if own_cams.exists() and own_ori.exists():
        return own_cams, own_ori
    # da3 fused meshes live in the MATCHA frame they were gauge-aligned to
    # (recorded as --matcha-run in their own run records; operator-caught
    # rotation on 006 da3--8-giant when the wrong sibling gauge was used)
    if rep_dir.parent.name == "da3":
        anchor_variant = None
        rr = rep_dir / "origin-run" / "run_record.json"
        if rr.exists():
            es = json.loads(rr.read_text()).get("instance", {}).get("expanded_settings", {})
            mr = (es.get("fuse", {}) or {}).get("matcha_run", "")
            if mr:
                anchor_variant = "matcha--" + mr.rsplit("run-", 1)[-1]
        if anchor_variant:
            for sib in sorted(rep_dir.parent.parent.glob("matcha/*/")):
                smd = sib / "metadata.json"
                if smd.exists():
                    sm = json.loads(smd.read_text())
                    variants = sm.get("legacy_variants", [sm.get("legacy_variant")])
                    if anchor_variant in variants:
                        return rep_camera_paths(scene_dir, sib)
    md = json.loads((rep_dir / "metadata.json").read_text())
    subset = md.get("resolved_inputs", {}).get("subset")
    solve = md.get("resolved_inputs", {}).get("cameras")
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    cands = [base / str(solve)] if solve else sorted(base.glob("*/")) if base.is_dir() else []
    # this rep's TRUE raw cameras sha (recorded at run time in results.json)
    want_sha = None
    orr = rep_dir / "origin-results.json"
    if orr.exists():
        for o in json.loads(orr.read_text()).get("outputs", []):
            if isinstance(o, dict) and o.get("path", "").endswith("mast3r_sfm/cameras.json"):
                want_sha = o.get("sha256")
    for c in cands:
        ods = sorted(c.glob("orient/*/oriented.json"))
        pool = [c / "cameras.json"] + sorted(c.glob("origin-dup-cameras.json"))
        pool = [p for p in pool if p.exists()]
        if not pool or not (ods or own_ori.exists()):
            continue
        cams = pool[0]
        if want_sha:
            for p in pool:
                if hashlib.sha256(p.read_bytes()).hexdigest() == want_sha:
                    cams = p          # the run's own solve, dedup-disambiguated
                    break
        return cams, (own_ori if own_ori.exists() else ods[0])
    return None, None


def mesh_targets(rep_dir: Path):
    """All renderable mesh dirs under a representation."""
    for mdir in sorted(rep_dir.glob("meshify/*/*/")):
        if (mdir / "mesh.ply").exists():
            yield mdir
        for cdir in sorted(mdir.glob("condition/*/")):
            if (cdir / "mesh.ply").exists():
                yield cdir


def scene_anchors(scene_dir: Path) -> list:
    """Migrated scenes: views were captured in ONE run's frame; sibling
    runs have their own gauges. The original cameras.json's anchor_frames
    (preserved at views/origin-cameras.json) let build_blender_scene
    Procrustes the view into EACH variant's frame — exactly v2 behavior.
    v4-native scenes (single primary solve) have no anchors -> identity."""
    oc = scene_dir / "views" / "origin-cameras.json"
    if oc.exists():
        try:
            return json.loads(oc.read_text()).get("anchor_frames", [])
        except ValueError:
            pass
    return []


def render_one(scene_dir: Path, mesh_dir: Path, slot: str, view_content: dict,
               cams: Path, oriented: Path, out_dir: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        views_json = Path(td) / "views.json"
        views_json.write_text(json.dumps(
            {"schema_version": 5,
             "anchor_frames": scene_anchors(scene_dir),
             "views": [dict(view_content, name=slot)]}, indent=2))
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

    # Job record created LAZILY on the first real render (an all-NOOP run
    # writes nothing, as before) — but once work starts it's written
    # INCREMENTALLY so the tile flips mid-job (STO-SCN-088), not only at
    # the end. Dry runs touch nothing.
    state = {"jd": None, "job_id": "(pending)"}

    def ensure_job():
        if state["jd"] is None and not dry:
            state["jd"] = sc.job_dir()
            state["job_id"] = state["jd"].name
        return state["jd"]

    def write_record(status: str) -> None:
        jd = state["jd"]
        if not jd:
            return
        (jd / "job.json").write_text(json.dumps({
            "schema": 4, "graph": "render-missing", "mechanism": "job",
            "status": status,
            "bindings": {"scene": scene, "viewset": "canonical (resolved per slot)"},
            "outcome": {k: stats[k] for k in ("noop", "rendered", "failed")},
            "nodes": stats["nodes"],
            "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2) + "\n")

    # Cheap pre-pass: count render targets for a real progress denominator.
    targets = []
    for rep_dir in sorted(sdir.glob("represent/*/*/")):
        if not (rep_dir / "metadata.json").exists():
            continue
        cams, oriented = rep_camera_paths(sdir, rep_dir)
        for mesh_dir in mesh_targets(rep_dir):
            for slot in slots:
                targets.append((rep_dir, cams, oriented, mesh_dir, slot))
    total = len(targets) or 1

    for done, (rep_dir, cams, oriented, mesh_dir, slot) in enumerate(targets, 1):
        mid = mesh_dir.name
        vc = json.loads((sdir / "views" / slot / "view.json").read_text())
        vh = sc.view_content_hash(slot)
        rid = v4.identity_hash({"mesh": mid, "view_content": vh}, RSET, ALGO)
        out_dir = mesh_dir / "renders" / rid
        node = f"{mesh_dir.relative_to(sdir)} × {slot}"
        pct = int(done * 100 / total)
        if (out_dir / "render.png").exists():
            stats["noop"] += 1
            continue
        if cams is None:
            stats["nodes"].append(f"SKIP {mesh_dir.relative_to(sdir)}: no cameras/orient")
            write_record("running")
            continue
        if dry:
            stats["nodes"].append(f"WOULD-RENDER {node}")
            stats["rendered"] += 1
            continue
        ensure_job()
        publish_progress(scene, state["job_id"], node, "running", pct)
        ok = render_one(sdir, mesh_dir, slot, vc, cams, oriented, out_dir)
        if ok:
            v4.write_metadata(out_dir, task="render", algo=ALGO, identity=rid,
                              resolved_inputs={"mesh": mid, "view_content": vh},
                              settings=RSET, mechanism="job",
                              extra={"view_slot": slot})
            stats["rendered"] += 1
            stats["nodes"].append(f"RENDERED {node}")
        else:
            stats["failed"] += 1
            stats["nodes"].append(f"FAILED {node}")
        write_record("running")            # incremental: tile flips mid-job

    if state["jd"] and (stats["rendered"] or stats["failed"]):
        write_record("done")
        publish_progress(scene, state["job_id"], "render-missing", "done", 100)
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
