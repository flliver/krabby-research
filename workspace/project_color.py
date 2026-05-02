"""Phase B4 — project source frames onto an oriented MAtCha mesh as vertex colors.

Algorithm (Tier A, no raycast occlusion — relies on back-face culling):

For each vertex v in the oriented mesh:
    For each of the 12 cameras (transformed to oriented space):
        - Reject if v is behind the camera
        - Reject if v's projection falls outside the source image bounds
        - Reject if v's surface normal is back-facing relative to the camera
          (vertex normal dot view ray > 0)
        - Sample the source image at (u, v) bilinear
        - Weight by (cos angle of incidence) / (distance to camera)
    Final color = weighted average of all valid samples.
    Vertices with no valid camera get a neutral gray fallback.

Run inside the matcha-build container:
    source /opt/matcha/bin/activate
    python project_color.py \
        --mesh <path>/oriented_500k.ply \
        --cameras <path>/cameras.json \
        --oriented-cameras <path>/oriented_cameras.json \
        --images <path>/mast3r_sfm/images \
        --output <path>/oriented_500k_colored.ply

Tier B (raycast occlusion) is implementable as a follow-up:
build an Open3D RaycastingScene, cast rays from each candidate
(vertex, camera) and reject if the first hit is closer than the vertex.
"""
import argparse
import json
import os
import time

import numpy as np
import open3d as o3d
from PIL import Image


def load_oriented_cameras(cameras_orig_json, cameras_oriented_json):
    """Compose orient transform + original cams2world. Returns:
      cams2world_oriented: (N, 4, 4) numpy
      world2cam_oriented:  (N, 4, 4) numpy
      focals: (N,) numpy
    """
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


def load_images(images_dir):
    files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    imgs = []
    sizes = []
    for fn in files:
        im = np.asarray(Image.open(os.path.join(images_dir, fn)).convert("RGB"))  # (H, W, 3) uint8
        imgs.append(im)
        sizes.append(im.shape[:2][::-1])  # (W, H)
    if not imgs:
        raise SystemExit(f"No images found in {images_dir}")
    return imgs, sizes


def bilinear_sample(image, u, v):
    """Bilinear sample of an HxWx3 uint8 image at float (u, v) coords.
    u: column (x), v: row (y). Out-of-bounds clamped to nearest edge.
    Returns (R, G, B) float32 array shape matching u.
    """
    H, W = image.shape[:2]
    u = np.clip(u, 0.0, W - 1.0)
    v = np.clip(v, 0.0, H - 1.0)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    fu = (u - u0)[:, None]
    fv = (v - v0)[:, None]
    img = image.astype(np.float32)
    a = img[v0, u0] * (1 - fu) * (1 - fv)
    b = img[v0, u1] * fu * (1 - fv)
    c = img[v1, u0] * (1 - fu) * fv
    d = img[v1, u1] * fu * fv
    return a + b + c + d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True, help="MAtCha cameras.json")
    ap.add_argument("--oriented-cameras", required=True, help="orient_mesh.py output")
    ap.add_argument("--images", required=True, help="dir with the per-frame source images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fallback-rgb", default="160,160,160", help="default color for vertices no camera sees")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1] Load mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)  # (V, 3)
    normals = np.asarray(mesh.vertex_normals)  # (V, 3)
    V = len(verts)
    print(f"    {V:,} verts / {len(mesh.triangles):,} tris in {time.time()-t0:.1f}s")

    print(f"[2] Load cameras")
    cams2world, world2cam, focals = load_oriented_cameras(args.cameras, args.oriented_cameras)
    N = len(cams2world)
    print(f"    {N} cameras; focal range: {focals.min():.1f}..{focals.max():.1f}")

    print(f"[3] Load source images: {args.images}")
    images, sizes = load_images(args.images)
    if len(images) != N:
        print(f"WARNING: {len(images)} images but {N} cameras — taking first {min(len(images),N)}")
    H, W = images[0].shape[:2]
    cx, cy = W / 2.0, H / 2.0
    print(f"    {len(images)} images, size {W}x{H}, principal point ({cx:.1f}, {cy:.1f})")

    # Accumulators
    color_sum = np.zeros((V, 3), dtype=np.float64)
    weight_sum = np.zeros(V, dtype=np.float64)
    view_count = np.zeros(V, dtype=np.int32)

    # Project from each camera independently (vectorized over V)
    for ci in range(min(len(images), N)):
        t1 = time.time()
        # Transform vertices to camera space
        v_h = np.concatenate([verts, np.ones((V, 1))], axis=1)  # (V, 4)
        v_cam = (world2cam[ci] @ v_h.T).T  # (V, 4)
        z = v_cam[:, 2]
        front = z > 1e-6  # in front of camera

        # Project
        f = focals[ci] if ci < len(focals) else focals[0]
        u = f * v_cam[:, 0] / np.where(z != 0, z, 1) + cx
        vpix = f * v_cam[:, 1] / np.where(z != 0, z, 1) + cy
        in_bounds = (u >= 0) & (u < W) & (vpix >= 0) & (vpix < H)

        # View ray (world space): from cam center to vertex
        cam_center = cams2world[ci, :3, 3]
        view_ray = verts - cam_center  # (V, 3)
        dist = np.linalg.norm(view_ray, axis=1)
        view_ray_unit = view_ray / np.where(dist[:, None] > 0, dist[:, None], 1)
        # Front-facing: vertex normal must point against the view ray
        # cos_angle = -dot(normal, view_ray_unit) (positive if facing camera)
        cos_face = -(normals * view_ray_unit).sum(axis=1)
        front_facing = cos_face > 0.05  # small threshold to avoid grazing angles

        valid = front & in_bounds & front_facing
        if not valid.any():
            print(f"  cam {ci+1:2d}: 0 valid verts (in {time.time()-t1:.2f}s)")
            continue

        # Sample colors at valid projections only
        colors = bilinear_sample(images[ci], u[valid], vpix[valid])  # (n_valid, 3)
        # Weight = view-quality / distance (closer + more head-on = higher weight)
        weights = cos_face[valid] / np.maximum(dist[valid], 1e-3)

        # Accumulate
        color_sum[valid] += weights[:, None] * colors
        weight_sum[valid] += weights
        view_count[valid] += 1

        n_valid = int(valid.sum())
        print(f"  cam {ci+1:2d}: {n_valid:,} valid verts ({100*n_valid/V:.1f}%) in {time.time()-t1:.2f}s")

    print(f"[4] Resolve final vertex colors")
    has_color = weight_sum > 0
    final_color = np.zeros((V, 3), dtype=np.float64)
    final_color[has_color] = color_sum[has_color] / weight_sum[has_color, None]

    # Fallback for uncovered vertices
    fallback = np.array([float(x) for x in args.fallback_rgb.split(",")])
    final_color[~has_color] = fallback

    # Stats
    print(f"    {has_color.sum():,} / {V:,} verts colored ({100*has_color.sum()/V:.1f}%)")
    print(f"    view-count distribution: min={view_count.min()}, max={view_count.max()}, "
          f"mean={view_count.mean():.2f}, median={int(np.median(view_count))}")

    # Set vertex colors (Open3D expects float [0,1])
    mesh.vertex_colors = o3d.utility.Vector3dVector(final_color / 255.0)

    print(f"[5] Write {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"    {os.path.getsize(args.output)/1024/1024:.1f} MB in {time.time()-t0:.1f}s total")
    print("DONE.")


if __name__ == "__main__":
    main()
