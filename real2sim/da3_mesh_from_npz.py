#!/usr/bin/env python3
"""da3_mesh_from_npz.py — TSDF-fuse a DA3 results.npz (depth + conf + pose + intrinsics)
into a triangle mesh, in the npz's OWN frame.

Reuses the proven Open3D ScalableTSDFVolume recipe from da3_tsdf_mesh.py, but standalone on
a single npz with NO matcha-oriented alignment — for a scout npz (STO-SCN-095/103) whose
depth is already rescaled to the input (solve) gauge and whose extrinsics are the echoed
input cameras, the fused mesh lands directly in the SOLVE gauge (aligned with the frustums /
cull box / the registered splat). The "DA3 scene" geometry from the selected-N views.

CPU-only (Open3D). Usage:
    da3_mesh_from_npz.py <results.npz> <out.ply> [--conf-percentile 40] [--voxel-frac 0.004]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def fuse_npz(npz_path, out_ply, conf_percentile=40.0, voxel_frac=0.004) -> dict:
    npz = np.load(npz_path)
    depth = npz["depth"].astype(np.float32)         # (N,H,W)
    conf = npz["conf"]
    img = npz["image"]                              # (N,H,W,3) uint8
    ext = npz["extrinsics"].astype(np.float64)      # (N,3,4) w2c
    K = npz["intrinsics"].astype(np.float64)        # (N,3,3)
    n, H, W = depth.shape

    thr = float(np.percentile(conf, conf_percentile))
    span = float(np.percentile(depth[conf > thr], 95))
    voxel = span * voxel_frac
    sdf_trunc = 4.0 * voxel
    print(f"da3-mesh: {n} views {W}x{H} · depth95 {span:.2f} · voxel {voxel:.4f} · conf>{thr:.2f}")

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(n):
        d = depth[i].copy()
        d[conf[i] <= thr] = 0.0                     # drop low-confidence depth
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(img[i])),
            o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=float(span * 1.5),
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        w2c4 = np.eye(4); w2c4[:3, :4] = ext[i]
        vol.integrate(rgbd, intr, w2c4)

    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    rec = {"tool": "da3_mesh_from_npz.py", "npz": str(npz_path), "out": str(out_ply),
           "n_views": n, "depth_res": [H, W], "voxel": round(voxel, 5),
           "sdf_trunc": round(sdf_trunc, 5), "conf_percentile": conf_percentile,
           "verts": len(mesh.vertices), "tris": len(mesh.triangles), "frame": "npz (solve gauge)"}
    print(f"da3-mesh: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris -> {out_ply}")
    return rec


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TSDF-fuse a DA3 npz into a mesh (npz frame).")
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--voxel-frac", type=float, default=0.004)
    a = ap.parse_args(argv)
    rec = fuse_npz(a.npz, a.out, a.conf_percentile, a.voxel_frac)
    Path(a.out).with_suffix(".fusion.json").write_text(json.dumps(rec, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
