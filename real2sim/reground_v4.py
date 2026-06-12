#!/usr/bin/env python3
"""reground_v4.py — Option A: re-ground every legacy artifact into ONE
canonical gauge per representation (operator decision, 2026-06-12;
STO-SCN-089 root cause: v2 allowed per-artifact orientation ERAS to
drift — the store froze the drift; renders mixing eras roll/zoom).

Canonical gauge per rep = its final-era orientation
(origin-data/oriented/oriented_cameras.json; fallback: the subset
orient dir). Then:

  tetra meshes      <- regenerated from RAW binary-search ply (all of
                       them now — the verified-exclusion is dropped;
                       identities are inputs+settings+algo, so scores
                       survive untouched)
  conditioned       <- re-derived from the regenerated tetra
                       (decimate + Taubin + color transfer, recorded
                       settings)
  da3 fused         <- re-fused from the npz depths + canonical gauge
                       (port of da3_tsdf_mesh alignment, local CPU)
  matcha tsdf       <- no raw kept: era-resolved by trying known era
                       files and keeping the delta-transform whose
                       result best overlaps the regenerated tetra
                       (reference in canonical gauge); unresolvable ->
                       rankable:false
  ALL renders       <- invalidated; re-rendered by v4job afterwards

Run (open3d + numpy via uv):
    uv run --python 3.11 --with open3d --with numpy \
        python3 real2sim/reground_v4.py [<scene>|all] [--apply]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4
from gauge_align import align_camera_sets  # noqa: E402

APPLY = "--apply" in sys.argv


def log(*a):
    print(*a, flush=True)


def canonical_gauge(rep_dir: Path, scene_dir: Path):
    own = rep_dir / "origin-data" / "oriented" / "oriented_cameras.json"
    if own.exists():
        return own
    # merged reps (da3 studio): search origin-merged for anchor copies
    for m in sorted(rep_dir.glob("origin-merged/*/origin-data/oriented/oriented_cameras.json")):
        return m
    md = json.loads((rep_dir / "metadata.json").read_text())
    subset = md.get("resolved_inputs", {}).get("subset")
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    for c in sorted(base.glob("*/orient/*/oriented.json")) if base.is_dir() else []:
        return c
    return None


def era_candidates(rep_dir: Path, scene_dir: Path) -> list[Path]:
    """All known orientation-era files that artifacts of this rep might
    be frozen in."""
    out = []
    for p in [rep_dir / "origin-data" / "oriented" / "oriented_cameras.json",
              *sorted(rep_dir.glob("origin-merged/*/origin-data/oriented/oriented_cameras.json"))]:
        if p.exists():
            out.append(p)
    md = json.loads((rep_dir / "metadata.json").read_text())
    subset = md.get("resolved_inputs", {}).get("subset")
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    if base.is_dir():
        out += sorted(base.glob("*/orient/*/oriented.json"))
    # de-dup by content
    seen, uniq = set(), []
    for p in out:
        key = p.read_bytes()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def load_RZ(p: Path):
    import numpy as np
    d = json.loads(p.read_text())
    return np.asarray(d["rotation"], dtype=float), float(d["z_shift"])


def orient_raw(mesh, R, z):
    import numpy as np
    import open3d as o3d
    v = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(v @ R.T + np.array([0.0, 0.0, z]))
    mesh.compute_vertex_normals()
    return mesh


def era_delta(mesh, R_from, z_from, R_to, z_to):
    """mesh frozen in era FROM -> era TO (de-orient, re-orient)."""
    import numpy as np
    import open3d as o3d
    v = np.asarray(mesh.vertices)
    raw = (v - np.array([0.0, 0.0, z_from])) @ R_from   # R_from^T inverse of vRT
    out = raw @ R_to.T + np.array([0.0, 0.0, z_to])
    mesh.vertices = o3d.utility.Vector3dVector(out)
    mesh.compute_vertex_normals()
    return mesh


def overlap_score(va, vb):
    """Crude frame agreement: centroid distance + bbox IoU-ish (xy)."""
    import numpy as np
    ca, cb = va.mean(0), vb.mean(0)
    d = float(np.linalg.norm(ca - cb))
    lo = np.maximum(va.min(0), vb.min(0))
    hi = np.minimum(va.max(0), vb.max(0))
    inter = float(np.prod(np.maximum(hi - lo, 0)[:2]))
    union = float(np.prod((np.maximum(va.max(0), vb.max(0)) - np.minimum(va.min(0), vb.min(0)))[:2]))
    return inter / union - d        # higher better


def refuse_da3(rep_dir: Path, scene_dir: Path, mesh_dir: Path, gauge: Path) -> bool:
    """Re-fuse a da3 tsdf mesh from npz depths into the canonical gauge."""
    import numpy as np
    import open3d as o3d
    npz_path = None
    for cand in [rep_dir / "exports" / "npz" / "results.npz",
                 *sorted(rep_dir.glob("origin-merged/*/exports/npz/results.npz"))]:
        if cand.exists():
            npz_path = cand
            break
    if npz_path is None:
        log(f"    NO NPZ for {rep_dir.name} — cannot re-fuse")
        return False
    md = json.loads((mesh_dir / "metadata.json").read_text())
    settings = md.get("settings", {})
    voxel_frac = float(settings.get("voxel_frac", 0.004))
    conf_pct = float(settings.get("conf_percentile", 40))
    # anchor cameras: matcha sibling solve raw cams + canonical gauge
    rmd = json.loads((rep_dir / "metadata.json").read_text())
    subset = rmd.get("resolved_inputs", {}).get("subset")
    cams_path = None
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    for c in sorted(base.glob("*/cameras.json")) if base.is_dir() else []:
        cams_path = c
        break
    if cams_path is None:
        log(f"    NO anchor cameras for {rep_dir.name}")
        return False
    cams = json.loads(cams_path.read_text())
    R_o, z = load_RZ(gauge)
    order = np.argsort([fp.rsplit("/", 1)[-1] for fp in cams["filepaths"]])
    c2w = np.asarray(cams["cams2world"])[order]
    C_mat = (R_o @ c2w[:, :3, 3].T).T + np.array([0.0, 0.0, z])
    R_mat = np.einsum("ij,njk->nik", R_o, c2w[:, :3, :3])
    npz = np.load(npz_path)
    depth = npz["depth"].astype(np.float32)
    conf = npz["conf"]
    img = npz["image"]
    ext = npz["extrinsics"].astype(np.float64)
    K = npz["intrinsics"].astype(np.float64)
    n, H, W = depth.shape
    Rw, tw = ext[:, :3, :3], ext[:, :3, 3]
    C_da3 = np.einsum("nji,nj->ni", Rw, -tw)
    R_da3 = np.transpose(Rw, (0, 2, 1))
    res = align_camera_sets(C_da3, C_mat, src_rotations=R_da3, dst_rotations=R_mat)
    spread = np.linalg.norm(C_mat - C_mat.mean(0), axis=1).mean()
    frac = res["max_residual"] / spread
    log(f"    re-fuse alignment residual {frac*100:.1f}% scale {res['scale']:.4f}")
    if frac > 0.10:
        log("    REFUSED: residual > 10%")
        return False
    s, R_al, t_al = res["scale"], np.asarray(res["R"]), np.asarray(res["t"])
    thr = np.percentile(conf, conf_pct)
    span = float(np.percentile(depth[conf > thr], 95))
    voxel = span * voxel_frac
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(n):
        d = depth[i].copy()
        d[conf[i] <= thr] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(img[i])), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=float(span * 1.5), convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        w2c4 = np.eye(4)
        w2c4[:3, :4] = ext[i]
        vol.integrate(rgbd, intr, w2c4)
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    T = np.eye(4)
    T[:3, :3] = R_al
    T[:3, 3] = t_al / s
    mesh.transform(T)
    mesh.scale(s, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()
    if APPLY:
        o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.ply"), mesh)
    log(f"    re-fused: {len(mesh.vertices):,} verts -> {mesh_dir.name}/mesh.ply")
    return True


def recondition(tetra_dir: Path, cond_dir: Path) -> bool:
    import numpy as np
    import open3d as o3d
    md = json.loads((cond_dir / "metadata.json").read_text())
    s = md.get("settings", {})
    target = int(s.get("target_tris", 1_000_000))
    taubin = int(s.get("taubin_iters", 10))
    src = o3d.io.read_triangle_mesh(str(tetra_dir / "mesh.ply"))
    mesh = src.simplify_quadric_decimation(target_number_of_triangles=target)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if taubin > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin)
    if src.has_vertex_colors():
        kd = o3d.geometry.KDTreeFlann(src)
        sc = np.asarray(src.vertex_colors)
        verts = np.asarray(mesh.vertices)
        colors = np.empty_like(verts)
        for i, vtx in enumerate(verts):
            _, idx, _ = kd.search_knn_vector_3d(vtx, 1)
            colors[i] = sc[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    if APPLY:
        o3d.io.write_triangle_mesh(str(cond_dir / "mesh.ply"), mesh)
    log(f"    reconditioned {cond_dir.name}: {len(mesh.triangles):,} tris")
    return True


def reground_scene(scene: str):
    import numpy as np
    import open3d as o3d
    sdir = v4.STORE / scene
    if not (sdir / "represent").is_dir():
        return
    log(f"== {scene}")
    for rep_dir in sorted(sdir.glob("represent/*/*/")):
        if not (rep_dir / "metadata.json").exists():
            continue
        kind = rep_dir.parent.name
        gauge = canonical_gauge(rep_dir, sdir)
        if gauge is None:
            log(f"  {rep_dir.name}: NO GAUGE — skipped")
            continue
        R_can, z_can = load_RZ(gauge)
        tetra_ref_verts = None
        # --- tetra: regenerate from raw into canonical gauge
        for tdir in sorted(rep_dir.glob("meshify/tetra/*/")):
            raws = sorted(p for p in tdir.glob("tetra_mesh_binary_search_*.ply")
                          if not p.name.startswith("origin-dup"))
            if raws:
                log(f"  {rep_dir.name}/tetra/{tdir.name}: raw -> canonical")
                mesh = o3d.io.read_triangle_mesh(str(raws[-1]))
                mesh = orient_raw(mesh, R_can, z_can)
                if APPLY:
                    o3d.io.write_triangle_mesh(str(tdir / "mesh.ply"), mesh)
                tetra_ref_verts = np.asarray(mesh.vertices)
            elif (tdir / "mesh.ply").exists():
                log(f"  {rep_dir.name}/tetra/{tdir.name}: NO RAW — era-resolve")
                mesh0 = o3d.io.read_triangle_mesh(str(tdir / "mesh.ply"))
                best = None
                for era in era_candidates(rep_dir, sdir):
                    R_e, z_e = load_RZ(era)
                    m = o3d.geometry.TriangleMesh(mesh0)
                    m = era_delta(m, R_e, z_e, R_can, z_can)
                    v = np.asarray(m.vertices)
                    floor_ok = abs(float(np.percentile(v[:, 2], 2))) < 0.3
                    score = (1.0 if floor_ok else 0.0)
                    if best is None or score > best[0]:
                        best = (score, era, m)
                if best and best[0] > 0:
                    if APPLY:
                        o3d.io.write_triangle_mesh(str(tdir / "mesh.ply"), best[2])
                    tetra_ref_verts = np.asarray(best[2].vertices)
                    log(f"    era={best[1].name} delta applied")
            # conditioned under this tetra
            for cdir in sorted(tdir.glob("condition/*/")):
                if (cdir / "metadata.json").exists() and (tdir / "mesh.ply").exists():
                    recondition(tdir, cdir)
        # --- tsdf
        for mdir in sorted(rep_dir.glob("meshify/tsdf/*/")):
            if not (mdir / "metadata.json").exists() or not (mdir / "mesh.ply").exists():
                continue
            if kind == "da3":
                ok = refuse_da3(rep_dir, sdir, mdir, gauge)
                if ok and APPLY:
                    md = json.loads((mdir / "metadata.json").read_text())
                    md.pop("rankable", None)
                    md.pop("rankable_reason", None)
                    md["reground"] = {"gauge": str(gauge.relative_to(sdir)), "by": "refuse_da3"}
                    (mdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
                continue
            # matcha tsdf: era-resolve against the regenerated tetra reference
            log(f"  {rep_dir.name}/tsdf/{mdir.name}: era-resolve")
            mesh0 = o3d.io.read_triangle_mesh(str(mdir / "mesh.ply"))
            best = None
            for era in era_candidates(rep_dir, sdir):
                R_e, z_e = load_RZ(era)
                m = o3d.geometry.TriangleMesh(mesh0)
                m = era_delta(m, R_e, z_e, R_can, z_can)
                v = np.asarray(m.vertices)
                if tetra_ref_verts is not None:
                    score = overlap_score(v, tetra_ref_verts)
                else:
                    score = -abs(float(np.percentile(v[:, 2], 2)))
                if best is None or score > best[0]:
                    best = (score, era, m)
            if best:
                if APPLY:
                    o3d.io.write_triangle_mesh(str(mdir / "mesh.ply"), best[2])
                    md = json.loads((mdir / "metadata.json").read_text())
                    md["reground"] = {"gauge": str(gauge.relative_to(sdir)),
                                      "era_resolved_from": best[1].name, "score": round(best[0], 3)}
                    (mdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
                log(f"    era={best[1].name} score={best[0]:.3f}")
        # mark the rep's canonical gauge for the renderer
        if APPLY:
            md = json.loads((rep_dir / "metadata.json").read_text())
            md["canonical_gauge"] = str(gauge.relative_to(sdir))
            (rep_dir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    # invalidate ALL renders under meshify (regenerated next pass)
    n = 0
    for rmd in sdir.glob("represent/*/*/meshify/**/renders/*/metadata.json"):
        if APPLY:
            shutil.rmtree(rmd.parent)
        n += 1
    log(f"  renders invalidated: {n}")


def main():
    scenes = [a for a in sys.argv[1:] if not a.startswith("--")]
    roots = sorted(d.name for d in v4.STORE.iterdir()
                   if d.is_dir() and not d.name.startswith((".", "_"))) \
        if (not scenes or scenes[0] == "all") else scenes
    for s in roots:
        reground_scene(s)


if __name__ == "__main__":
    main()
