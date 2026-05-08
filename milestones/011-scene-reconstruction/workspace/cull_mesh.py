"""Phase B2 — auto-cull "out-of-scope" geometry from a MAtCha-derived mesh.

Three culling criteria, applied per-vertex; a vertex is dropped if any are true:

1. **Coverage cull** (primary): vertex was seen by < MIN_VIEWS cameras.
   Same projection logic as project_color.py — counts cameras that have
   the vertex in front of them, in image bounds, and front-facing.
   Default MIN_VIEWS = 2: a vertex needs at least two camera observations
   to be considered "real." MAtCha's chart-alignment can hallucinate
   geometry in regions only one or zero cameras saw.

2. **Below-floor cull** (secondary): vertex z < FLOOR_Z_MIN (default -0.5 m).
   The mesh has been oriented (B1) so floor is at z=0 with +z up. Anything
   more than ~50 cm below the floor is almost certainly a tetra-mesh
   floater, not a real sub-grade structure. (If we ever capture a basement
   or a step-down area this threshold should be relaxed per-scene.)

3. **Distance-from-cluster cull** (tertiary, optional): vertex more than
   MAX_DIST_FROM_CLUSTER from the camera centroid, scaled by the camera
   cluster's standard deviation. Disabled by default; enable for outdoor
   scenes where you know background pollution is the dominant noise source.

Triangles referencing any culled vertex are dropped. Vertex colors are
preserved if present in input.

Run inside the matcha-build container:
    source /opt/matcha/bin/activate
    python cull_mesh.py \
        --mesh <path>/oriented_500k_colored.ply \
        --cameras <path>/cameras.json \
        --oriented-cameras <path>/oriented_cameras.json \
        --output <path>/oriented_500k_colored_culled.ply
"""
import argparse
import json
import os
import time

import numpy as np
import open3d as o3d


def load_oriented_cameras(cameras_orig_json, cameras_oriented_json):
    """Same logic as project_color.py — kept inline to avoid cross-script import."""
    with open(cameras_orig_json) as f:
        co = json.load(f)
    with open(cameras_oriented_json) as f:
        cor = json.load(f)
    R = np.array(cor["rotation"], dtype=np.float64)
    z_shift = float(cor["z_shift"])
    R4 = np.eye(4)
    R4[:3, :3] = R
    T4 = np.eye(4)
    T4[2, 3] = z_shift
    world_orient = T4 @ R4
    cams_orig = np.array(co["cams2world"], dtype=np.float64)
    cams_oriented = np.einsum("ij,njk->nik", world_orient, cams_orig)
    world2cam = np.linalg.inv(cams_oriented)
    focals = np.array(co["focals"], dtype=np.float64)
    return cams_oriented, world2cam, focals


def compute_view_count(verts, normals, cams2world, world2cam, focals, image_size):
    """For each vertex, count how many cameras see it (front, in-bounds, front-facing).
    Returns int32 array (V,)."""
    W, H = image_size
    cx, cy = W / 2.0, H / 2.0
    V = len(verts)
    view_count = np.zeros(V, dtype=np.int32)
    N = len(cams2world)
    v_h = np.concatenate([verts, np.ones((V, 1))], axis=1)  # (V, 4)
    for ci in range(N):
        v_cam = (world2cam[ci] @ v_h.T).T
        z = v_cam[:, 2]
        front = z > 1e-6
        f = focals[ci] if ci < len(focals) else focals[0]
        u = f * v_cam[:, 0] / np.where(z != 0, z, 1) + cx
        vpix = f * v_cam[:, 1] / np.where(z != 0, z, 1) + cy
        in_bounds = (u >= 0) & (u < W) & (vpix >= 0) & (vpix < H)
        cam_center = cams2world[ci, :3, 3]
        view_ray = verts - cam_center
        dist = np.linalg.norm(view_ray, axis=1)
        view_ray_unit = view_ray / np.where(dist[:, None] > 0, dist[:, None], 1)
        cos_face = -(normals * view_ray_unit).sum(axis=1)
        front_facing = cos_face > 0.05
        view_count += (front & in_bounds & front_facing).astype(np.int32)
    return view_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--oriented-cameras", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-views", type=int, default=2,
                    help="Drop vertices seen by fewer than N cameras")
    ap.add_argument("--floor-z-min", type=float, default=-0.5,
                    help="Drop vertices with z < this (in oriented space; floor is z=0)")
    ap.add_argument("--max-dist-from-cluster", type=float, default=0.0,
                    help="Drop vertices farther than this from camera centroid (0=disabled)")
    ap.add_argument("--image-size", default="1024,576",
                    help="WxH of the source images used by MAtCha (default 1024,576)")
    args = ap.parse_args()

    image_size = tuple(int(x) for x in args.image_size.split(","))

    t0 = time.time()
    print(f"[1] Load mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    has_colors = mesh.has_vertex_colors()
    if has_colors:
        colors = np.asarray(mesh.vertex_colors)
    V0 = len(verts)
    T0 = len(triangles)
    print(f"    {V0:,} verts / {T0:,} tris (colors: {has_colors})")

    print(f"[2] Load cameras")
    cams2world, world2cam, focals = load_oriented_cameras(args.cameras, args.oriented_cameras)
    cam_centers = cams2world[:, :3, 3]
    cam_centroid = cam_centers.mean(axis=0)
    cam_spread = float(np.linalg.norm(cam_centers - cam_centroid, axis=1).std())
    print(f"    {len(cams2world)} cameras; centroid={cam_centroid}, spread σ={cam_spread:.2f}")

    print(f"[3] Compute view count per vertex")
    t1 = time.time()
    view_count = compute_view_count(verts, normals, cams2world, world2cam, focals, image_size)
    print(f"    done in {time.time()-t1:.2f}s; "
          f"distribution: min={view_count.min()}, max={view_count.max()}, "
          f"mean={view_count.mean():.2f}, median={int(np.median(view_count))}")

    # Build per-vertex masks
    print(f"[4] Apply cull criteria")
    enough_views = view_count >= args.min_views
    above_floor = verts[:, 2] >= args.floor_z_min
    if args.max_dist_from_cluster > 0:
        dist_from_centroid = np.linalg.norm(verts - cam_centroid, axis=1)
        near_cluster = dist_from_centroid <= args.max_dist_from_cluster
    else:
        near_cluster = np.ones(V0, dtype=bool)

    valid = enough_views & above_floor & near_cluster
    n_drop_views = int((~enough_views).sum())
    n_drop_floor = int((enough_views & ~above_floor).sum())
    n_drop_dist = int((enough_views & above_floor & ~near_cluster).sum())
    n_keep = int(valid.sum())
    print(f"    drop: {n_drop_views:,} (views<{args.min_views})  "
          f"+ {n_drop_floor:,} (z<{args.floor_z_min})  "
          f"+ {n_drop_dist:,} (dist>{args.max_dist_from_cluster}m)")
    print(f"    keep: {n_keep:,} verts ({100*n_keep/V0:.1f}%)")

    # Filter triangles: drop any triangle that references a culled vertex
    print(f"[5] Filter triangles")
    valid_tris = valid[triangles].all(axis=1)
    new_triangles_old_idx = triangles[valid_tris]
    n_keep_tris = int(valid_tris.sum())
    print(f"    keep: {n_keep_tris:,} tris ({100*n_keep_tris/T0:.1f}% of {T0:,})")

    # Re-index vertices (drop unused ones implicitly by remapping)
    used = np.zeros(V0, dtype=bool)
    used[new_triangles_old_idx.ravel()] = True
    new_idx_map = -np.ones(V0, dtype=np.int64)
    new_idx_map[used] = np.arange(int(used.sum()))
    new_triangles = new_idx_map[new_triangles_old_idx]
    new_verts = verts[used]
    if has_colors:
        new_colors = colors[used]

    print(f"[6] Build output mesh: {len(new_verts):,} verts / {len(new_triangles):,} tris")
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_verts)
    out.triangles = o3d.utility.Vector3iVector(new_triangles)
    if has_colors:
        out.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    out.compute_vertex_normals()
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    out.remove_duplicated_triangles()

    print(f"[7] Write {args.output}")
    o3d.io.write_triangle_mesh(args.output, out)
    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f"    {sz:.1f} MB in {time.time()-t0:.1f}s total")
    print(f"    final: {len(out.vertices):,} verts / {len(out.triangles):,} tris")
    print("DONE.")


if __name__ == "__main__":
    main()
