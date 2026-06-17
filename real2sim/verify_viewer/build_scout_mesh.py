#!/usr/bin/env python3
"""STO-SCN-135 — "Open in Scout" for an arbitrary reconstruction MESH.

Given a scene + a mesh identity (a ranked variant), build a scout-style viewer that shows,
together and gauge-aligned:
  1. the variant's MESH (`mesh.ply`, canonical/oriented gauge)  -> served as scene.ply
  2. the full CAMERA SPINE (every posed camera of the solve)    -> frustums
  3. the UTILIZED SUBSET (the N cameras this variant was built from) -> highlighted (`proposed`)

Reuses verify_viewer's frustum primitives + viewer.html (which already has a mesh layer +
proposed-highlight + gravity grid). The frustums come out of the solve in the SOLVE gauge; the
mesh is ORIENTED, so we carry each frustum through the variant's orient (R, z_shift) — both then
sit in the oriented gauge (gravity = +z) and overlay correctly.

Usage:
  build_scout_mesh.py <scene> --mesh <mesh_identity> [--serve-dir D] [--port 8098] [--no-serve]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import posed_from_sparse as pfs                       # noqa: E402
from build_verify import frustum_from_w2c, _apply_xform  # noqa: E402

STORE = Path("/var/krabby/scenes")


def find_mesh(scene_dir: Path, mesh_id: str) -> Path | None:
    for pat in (f"represent/*/*/meshify/*/{mesh_id}/mesh.ply",
                f"represent/*/*/meshify/*/*/condition/{mesh_id}/mesh.ply"):
        for p in scene_dir.glob(pat):
            return p
    return None


def subset_stems(scene_dir: Path, rep_md: dict) -> set:
    """The frame stems this variant was built from — scout views (da3-scout) or subset members."""
    ri = rep_md.get("resolved_inputs", {})
    sub, sid, scout = ri.get("subset"), ri.get("cameras"), ri.get("scout")
    if scout and sub and sid:
        sv = scene_dir / "images" / "subsets" / sub / "cameras" / sid / "scout" / scout / "scout_views.json"
        if sv.exists():
            sd = json.loads(sv.read_text())
            names = sd.get("views") or sd.get("selected") or sd.get("names") or []
            return {n.rsplit(".", 1)[0] for n in names}
    if sub:
        sj = scene_dir / "images" / "subsets" / sub / "subset.json"
        if sj.exists():
            out = set()
            for h in json.loads(sj.read_text()).get("members", []):
                mp = scene_dir / "images" / h / "metadata.json"
                nm = json.loads(mp.read_text()).get("original_name", h) if mp.exists() else h
                out.add(nm.rsplit(".", 1)[0])
            return out
    return set()


def build(scene: str, mesh_id: str, serve_dir: str | None = None) -> Path:
    scene_dir = STORE / scene
    mesh_ply = find_mesh(scene_dir, mesh_id)
    if not mesh_ply:
        sys.exit(f"no mesh '{mesh_id}' under {scene_dir}/represent")
    parts = mesh_ply.parts
    rep_dir = Path(*parts[:parts.index("represent") + 3])      # represent/<kind>/<rid>
    rep_md = json.loads((rep_dir / "metadata.json").read_text())
    cg = rep_md.get("canonical_gauge")
    if not cg:
        sys.exit(f"rep {rep_dir.name} has no canonical_gauge — cannot orient the spine")
    cg_path = scene_dir / cg                                   # …/cameras/<sid>/orient/<oid>/oriented.json
    gj = json.loads(cg_path.read_text())
    Rg, z = np.asarray(gj["rotation"], float), float(gj["z_shift"])
    solve_dir = cg_path.parents[2]                             # …/cameras/<sid>  (the SOLVE, parent-pool aware)
    sparse = solve_dir / "sparse" / "0"
    if (sparse / "images.bin").exists():
        posed = pfs.posed_from_sparse(str(sparse))             # FastMap spine (COLMAP bins)
    else:
        # migrated/mast3r-era solves (pre-spine work) have no sparse/0, but they DO carry a
        # cameras.json (filepaths + cams2world) — build the spine frustums from it (operator
        # 2026-06-16; these older builds predate FastMap sparse/0). w2c = inv(cams2world).
        cj = solve_dir / "cameras.json"
        if not cj.exists():
            sys.exit(f"no spine: neither {sparse}/images.bin nor {cj}")
        cam = json.loads(cj.read_text())
        c2w = np.asarray(cam["cams2world"], float)
        fps = cam.get("filepaths") or [str(i) for i in range(len(c2w))]
        posed = [{"name": Path(fps[i]).name, "w2c": np.linalg.inv(c2w[i])}
                 for i in range(len(c2w))]
        print(f"[scout-mesh] spine from cameras.json ({len(posed)} cams; migrated/mast3r solve, "
              f"no sparse/0)")
    highlight = subset_stems(scene_dir, rep_md)
    xform = (1.0, Rg, np.array([0.0, 0.0, z]))                 # orient: solve gauge -> oriented gauge
    frustums = []
    for e in posed:
        rflat, c = _apply_xform(*frustum_from_w2c(e["w2c"]), xform)
        frustums.append({"R": rflat, "pos": c,
                         "proposed": e["name"].rsplit(".", 1)[0] in highlight,
                         "name": e["name"]})
    n_hi = sum(f["proposed"] for f in frustums)
    C = np.asarray([f["pos"] for f in frustums], float)
    ctr = C.mean(0).tolist()
    rad = round(float(np.linalg.norm(C - C.mean(0), axis=1).max()), 3)
    label = rep_md.get("algo") or rep_dir.parent.name
    data = {"title": f"{scene} · {label} · {n_hi}/{len(frustums)} cams (mesh {mesh_id})",
            "frustums": frustums, "up": [0.0, 0.0, 1.0],
            "scene_ctr": ctr, "scene_radius": rad, "scout_frames": []}

    serve = Path(serve_dir) if serve_dir else Path(f"/tmp/scout-mesh-{scene}-{mesh_id}")
    serve.mkdir(parents=True, exist_ok=True)
    (serve / "frustums.json").write_text(json.dumps(data))
    # Serve the mesh as-is (symlink, no copy). NOTE: meshes whose face count exceeds the WebGL
    # per-draw index limit (30M indices = 10M tris) silently won't draw in the browser — the Rank
    # UI gates the "Open in Scout" button on `scout_mesh_renderable` (server reads the PLY header
    # face count) so an un-renderable mesh never reaches here (STO-SCN-135).
    scn = serve / "scene.ply"
    if scn.exists() or scn.is_symlink():
        scn.unlink()
    scn.symlink_to(mesh_ply)
    # Reuse viewer.html in mesh-mode: mesh visible by default, splat optional (none for matcha).
    v = (HERE / "viewer.html").read_text()
    v = v.replace('id="meshop" type="range" min="0" max="1" step="0.05" value="0"',
                  'id="meshop" type="range" min="0" max="1" step="0.05" value="1"')
    # mesh-mode has NO splat: do NOT call addSplatScene — a 404 makes the GS loader *hang* the
    # top-level await, so loop() is never reached → blank page. Skip it entirely (STO-SCN-135).
    v = v.replace(
        "await viewer.addSplatScene('./scout.gs.ply' + CB, {format:GS.SceneFormat.Ply, showLoadingUI:false});",
        "document.querySelector('#s').textContent = `mesh view · ${nP} of ${FR.length} cameras "
        "(highlighted)`;")
    # no splat scene → getSplatScene(0) on the empty viewer throws "invalid scene index";
    # short it to null so the registration block skips cleanly.
    v = v.replace(
        "const sscene = viewer.getSplatScene && viewer.getSplatScene(0);",
        "const sscene = null;  // mesh-mode: no splat scene (STO-SCN-135)")
    # ...and guard the GS render so an empty-splat render() can't kill the overlay render.
    v = v.replace(
        "renderer.autoClear=true; viewer.update(); viewer.render();",
        "renderer.autoClear=true; try{ viewer.update(); viewer.render(); }catch(_e){ renderer.clear(); }")
    # ESSENTIAL: with no splat, nothing frames the camera or builds the grid (the GS splat-load
    # normally does). Call frame(up) once so the camera is positioned + the ground grid exists.
    v = v.replace("}\nloop();",
                  "}\nframe(up);  // STO-SCN-135: position camera + grid (no splat auto-frame)\nloop();")
    # accurate labels (this is a mesh + spine view, not a scout gaussian)
    v = v.replace("scout gaussian + ${nP} proposed / ${FR.length} pool frustums (solve gauge",
                  "mesh + spine · ${nP} of ${FR.length} cameras highlighted (oriented gauge")
    v = v.replace("' · DA3 mesh loaded'", "' · mesh loaded'")
    # on-page error overlay (STO-SCN-135) — surface JS/import/async errors in the status bar
    # so failures are visible without opening DevTools (T-012).
    v = v.replace(
        '<script type="module">',
        '<script>window.addEventListener("error",e=>{var s=document.getElementById("s");'
        'if(s)s.textContent="JS ERR: "+(e.message||e.error)+" @"+((e.filename||"").split("/").pop())+":"+e.lineno;});'
        'window.addEventListener("unhandledrejection",e=>{var s=document.getElementById("s");'
        'if(s)s.textContent="ASYNC ERR: "+((e.reason&&e.reason.message)||e.reason);});</script>\n'
        '<script type="module">', 1)
    (serve / "viewer.html").write_text(v)
    print(f"[scout-mesh] {n_hi} highlighted / {len(frustums)} spine cameras · mesh {mesh_ply.name} "
          f"({mesh_ply.stat().st_size // 2**20} MB) -> {serve}/viewer.html")
    return serve


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Open a reconstruction mesh in the scout viewer (STO-SCN-135).")
    ap.add_argument("scene")
    ap.add_argument("--mesh", required=True, help="mesh identity (a ranked variant)")
    ap.add_argument("--serve-dir", default=None)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--no-serve", action="store_true")
    a = ap.parse_args(argv)
    serve = build(a.scene, a.mesh, a.serve_dir)
    if a.no_serve:
        return 0
    import http.server
    import os
    os.chdir(serve)
    print(f"[scout-mesh] serving http://krabby.organl.com:{a.port}/viewer.html")
    http.server.HTTPServer(("0.0.0.0", a.port),
                           http.server.SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
