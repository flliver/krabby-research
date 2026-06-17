"""STO-SCN-142 — (A) Merge & gap-fill via screened Poisson reconstruction.

Takes a materialized (e.g. culled) mesh and produces a WATERTIGHT, manifold, hole-filled
mesh via Open3D screened Poisson over an oriented point sample of the surface, with:
  - density-trim (drop the low-density Poisson "skirt" that balloons past the real surface),
  - keep-largest-connected-component cleanup,
  - nearest-vertex colour transfer from the source mesh (Poisson output is uncoloured).

Pure CPU — a post-process conditioning step (input = a good mesh, output = watertight). Run
the same way as cull_mesh.py (via `uv run --with numpy --with open3d`). The mesh stays in its
input (canonical) gauge — Poisson is gauge-preserving.

    python merge_gapfill.py --mesh <in.ply> --output <out.ply> \
        --poisson-depth 9 --density-quantile 0.05 --samples 1000000
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", choices=["fill-holes", "poisson"], default="fill-holes",
                    help="fill-holes = LOCAL hole-fill (preserves the open scene, no balloon); "
                         "poisson = global screened Poisson (closed objects only — balloons scenes)")
    ap.add_argument("--hole-size", type=float, default=0.3,
                    help="fill-holes: max hole boundary size to fill (mesh units); leaves the big "
                         "open boundaries (sky) alone, fills walkable-surface gaps")
    ap.add_argument("--poisson-depth", type=int, default=9,
                    help="Poisson octree depth (higher = more detail, slower)")
    ap.add_argument("--density-quantile", type=float, default=0.05,
                    help="Drop Poisson verts below this density quantile (the ballooned skirt); 0=off")
    ap.add_argument("--samples", type=int, default=1_000_000,
                    help="Oriented points sampled from the surface to feed Poisson")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1] Load mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    V0 = len(mesh.vertices)
    T0 = len(mesh.triangles)
    has_colors = mesh.has_vertex_colors()
    print(f"    {V0:,} verts / {T0:,} tris (colors: {has_colors})")

    if args.method == "fill-holes":
        # LOCAL hole-fill: fills holes whose boundary is smaller than hole_size, leaving the big
        # open boundaries (sky) alone — preserves the open-scene shape (NO Poisson balloon). The
        # right operation for "no holes in walkable surfaces" on an open scene (STO-SCN-142).
        print(f"[2] Local hole-fill (hole_size={args.hole_size}) — preserves the open scene, no balloon")
        nb = len(mesh.triangles)
        tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        pmesh = tm.fill_holes(hole_size=args.hole_size).to_legacy()
        pmesh.remove_degenerate_triangles()
        pmesh.remove_duplicated_triangles()
        pmesh.remove_unreferenced_vertices()
        print(f"    {nb:,} -> {len(pmesh.triangles):,} tris / {len(pmesh.vertices):,} verts")
    else:
        print(f"[2] Build oriented point cloud (vertices-direct: robust normals, no resample)")
        if len(mesh.vertices) >= args.samples:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            pcd.normals = mesh.vertex_normals
            if has_colors:
                pcd.colors = mesh.vertex_colors
        else:
            pcd = mesh.sample_points_uniformly(number_of_points=args.samples, use_triangle_normal=True)
        print(f"    {len(pcd.points):,} oriented points (colors: {pcd.has_colors()})")

        print(f"[3] Screened Poisson (depth={args.poisson_depth}, n_threads=1)")
        t1 = time.time()
        # n_threads=1 is REQUIRED: PoissonRecon's OpenMP iso-surface extraction has a threading
        # race ("Failed to close loop", FEMTree.IsoSurface) that crashes on multi-core — especially
        # ARM64 (we're on a mac). Upstream fix (isl-org/Open3D#2027, mkazhdan/PoissonRecon#136/#139,
        # colmap/colmap#4335). NB: global Poisson balloons OPEN scenes — prefer --method fill-holes.
        pmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth, n_threads=1)
        densities = np.asarray(densities)
        print(f"    {len(pmesh.vertices):,} verts / {len(pmesh.triangles):,} tris in {time.time()-t1:.1f}s")

        if args.density_quantile > 0:
            print(f"[4] Density-trim (drop < q{args.density_quantile} of the low-density skirt)")
            thr = np.quantile(densities, args.density_quantile)
            keep = densities >= thr
            pmesh.remove_vertices_by_mask(~keep)
            print(f"    kept {int(keep.sum()):,}/{len(keep):,} verts (density >= {thr:.3f})")

        print(f"[5] Cleanup: largest component + manifold tidy")
        pmesh.remove_degenerate_triangles()
        pmesh.remove_duplicated_triangles()
        pmesh.remove_duplicated_vertices()
        pmesh.remove_unreferenced_vertices()
        tri_idx, n_tri, _areas = pmesh.cluster_connected_triangles()
        tri_idx = np.asarray(tri_idx)
        n_tri = np.asarray(n_tri)
        if len(n_tri) > 1:
            biggest = int(n_tri.argmax())
            pmesh.remove_triangles_by_mask(tri_idx != biggest)
            pmesh.remove_unreferenced_vertices()
            print(f"    {len(n_tri)} components -> kept largest ({int(n_tri[biggest]):,} tris)")

    print(f"[6] Colour transfer from source (nearest vertex)")
    if has_colors and len(pmesh.vertices):
        src = o3d.geometry.PointCloud()
        src.points = mesh.vertices
        src.colors = mesh.vertex_colors
        kd = o3d.geometry.KDTreeFlann(src)
        sc = np.asarray(mesh.vertex_colors)
        pv = np.asarray(pmesh.vertices)
        cols = np.zeros((len(pv), 3))
        for i, p in enumerate(pv):
            _, idx, _ = kd.search_knn_vector_3d(p, 1)
            cols[i] = sc[idx[0]]
        pmesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    pmesh.compute_vertex_normals()

    print(f"[7] Write {args.output}")
    o3d.io.write_triangle_mesh(args.output, pmesh)
    wt = pmesh.is_watertight()
    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f"    {sz:.1f} MB in {time.time()-t0:.1f}s total")
    print(f"    final: {len(pmesh.vertices):,} verts / {len(pmesh.triangles):,} tris  "
          f"watertight={wt}")
    print("DONE.")


if __name__ == "__main__":
    main()
