#!/usr/bin/env python3
"""tetra_condition.py — de-densify + smooth a tetra mesh (STO-SCN-068).

The tetra branch wins the runoff on quality (continuous + sharp) but at
16M verts / 32M tris / 1.2 GB it is not a deliverable. This transform:

  1. quadric decimation to --target-tris
  2. Taubin smoothing (--taubin-iters; volume-preserving, unlike Laplacian)
  3. vertex-color transfer from the ORIGINAL mesh (nearest-vertex KDTree —
     decimation+smoothing degrade colors; the source mesh is the color truth)
  4. degenerate/unreferenced cleanup + fresh normals

Writes `<out-dir>/oriented_tetra_conditioned_<N>k.ply` + a
`tetra_condition_record.json` (parameters + measured counts).

Runs in-image (krabby-da3 krabby-tools carrier; open3d baked):
    python /opt/krabby-tools/tetra_condition.py \
        --in-mesh <data>/oriented/oriented_tetra.ply \
        --out-dir <data>/oriented \
        [--target-tris 1000000] [--taubin-iters 10]
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tris", type=int, default=1_000_000)
    ap.add_argument("--taubin-iters", type=int, default=10)
    args = ap.parse_args()

    src = o3d.io.read_triangle_mesh(args.in_mesh)
    n0v, n0t = len(src.vertices), len(src.triangles)
    print(f"in: {n0v:,} verts / {n0t:,} tris")

    mesh = src.simplify_quadric_decimation(target_number_of_triangles=args.target_tris)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"decimated: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    if args.taubin_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.taubin_iters)
        print(f"taubin x{args.taubin_iters} applied")

    # color transfer from source (color truth) via nearest vertex
    if src.has_vertex_colors():
        kd = o3d.geometry.KDTreeFlann(src)
        src_colors = np.asarray(src.vertex_colors)
        verts = np.asarray(mesh.vertices)
        colors = np.empty_like(verts)
        for i, v in enumerate(verts):
            _, idx, _ = kd.search_knn_vector_3d(v, 1)
            colors[i] = src_colors[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        print("colors transferred from source")

    mesh.compute_vertex_normals()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.target_tris // 1000}k"
    out = out_dir / f"oriented_tetra_conditioned_{tag}.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"wrote {out} ({out.stat().st_size/2**20:.0f} MB)")

    (out_dir / f"tetra_condition_record_{tag}.json").write_text(json.dumps({
        "tool": "real2sim/tetra_condition.py",
        "story": "STO-SCN-068",
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "in_mesh": args.in_mesh,
        "parameters": {"target_tris": args.target_tris, "taubin_iters": args.taubin_iters,
                       "color_transfer": "nearest-vertex from source"},
        "in_counts": {"verts": n0v, "tris": n0t},
        "out_counts": {"verts": len(mesh.vertices), "tris": len(mesh.triangles)},
        "out_bytes": out.stat().st_size,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    main()
