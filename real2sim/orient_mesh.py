"""Phase B1 — auto-deduce ground plane and orient a MAtCha mesh.

Takes:
  - a MAtCha tetra mesh (.ply)
  - the matching cameras.json from the same MAtCha run

Produces a gravity-aligned mesh where:
  - the deduced floor plane lies on z = 0
  - the floor's "outward" normal is +z (i.e., "up" is +z)
  - the camera cluster is at positive z (they were above the floor when capturing)

Algorithm:
  1. Load the mesh's vertices and decimate to ~200K samples for plane fitting
     (full 9-21M vertices is wasteful; a uniform random subsample is fine).
  2. Run RANSAC plane segmentation to get the top-K largest planar regions.
     Re-fit on residuals to find multiple candidate planes.
  3. Extract camera centers from cameras.json's cams2world matrices.
  4. For each candidate plane:
       a. Compute signed distances from camera centers to the plane.
       b. Score the plane as "floor candidate" by:
            - Are all/most cameras on the same side? (yes → consistent)
            - Is the inlier count high? (large planes are more likely floor)
            - Is the mean abs(distance) reasonable? (cameras at consistent
              height above the plane)
       c. Reject planes where cameras are on both sides (probably a wall
          or a spurious plane through the scene).
  5. Pick the highest-scoring candidate.
  6. Construct rotation matrix R that maps the floor-plane normal n̂ to
     +z, applied so that cameras end up at +z (flip n̂ if needed).
  7. Apply R + a translation that places the plane at z=0.
  8. Write oriented_mesh.ply / .obj plus oriented_cameras.json.

Designed to be run inside the matcha-build container:
  source /opt/matcha/bin/activate
  python orient_mesh.py \
      --tetra <path>/tetra_mesh_binary_search_7.ply \
      --cameras <path>/mast3r_sfm/cameras.json \
      --output <path>/oriented/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d


def load_camera_centers(cameras_json_path):
    """Extract 3D camera positions (world-space) from MAtCha's cameras.json.

    Format:
      {"filepaths": [...], "focals": [...], "cams2world": [<list of 4x4 matrices>]}
    The translation column of each cams2world matrix is the camera position
    in world coordinates.
    """
    with open(cameras_json_path) as f:
        data = json.load(f)
    centers = []
    for c2w in data["cams2world"]:
        m = np.asarray(c2w, dtype=np.float64)
        centers.append(m[:3, 3])
    return np.array(centers)  # shape (N, 3)


def fit_candidate_planes(points, num_planes=4, distance_threshold=0.05,
                         ransac_n=3, num_iterations=2000):
    """RANSAC-fit multiple candidate planes by repeated segmentation on
    the residual cloud. Returns a list of (plane_eq, inlier_count, inlier_pts)
    sorted by inlier count descending.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    candidates = []
    remaining = pcd
    for _ in range(num_planes):
        if len(remaining.points) < ransac_n * 10:
            break
        plane_eq, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        n_in = len(inliers)
        inlier_pts = np.asarray(remaining.points)[inliers]
        candidates.append((np.asarray(plane_eq), n_in, inlier_pts))
        remaining = remaining.select_by_index(inliers, invert=True)
    candidates.sort(key=lambda x: -x[1])
    return candidates


def signed_distances(points, plane_eq):
    """Signed perpendicular distance from each point to plane.
    plane_eq is (a, b, c, d) with a*x + b*y + c*z + d = 0; returned distance
    is positive on the side of the normal (a, b, c).
    """
    a, b, c, d = plane_eq
    n = np.array([a, b, c])
    return (points @ n + d) / np.linalg.norm(n)


def score_floor_candidate(plane_eq, inlier_count, camera_centers,
                          gravity_prior=None, gravity_confidence=0.0):
    """Score a plane as a 'floor' candidate.

    Returns (score, oriented_normal, mean_camera_height) where higher score
    is better and oriented_normal points from the plane toward the cameras.
    A score of 0 means we explicitly reject this candidate.

    If `gravity_prior` is provided (a unit vector pointing toward what the
    cameras think is "up" in world space), candidates whose oriented_normal
    aligns with it are boosted; misaligned candidates are penalized. This
    discriminates between floor (parallel to gravity) and wall (perpendicular
    to gravity) when both have similar inlier counts.

    `gravity_confidence` ∈ [0, 1] scales the prior's strength. When 0 (no
    prior), scoring is identical to the legacy formula. When 1 (full
    confidence), severely misaligned candidates score near zero.
    """
    n = np.asarray(plane_eq[:3])
    n_unit = n / np.linalg.norm(n)
    distances = signed_distances(camera_centers, plane_eq)

    n_pos = int((distances > 0).sum())
    n_neg = int((distances < 0).sum())
    n_total = len(camera_centers)

    # Consistency: at least 75% of cameras on the same side
    consistency = max(n_pos, n_neg) / n_total
    if consistency < 0.75:
        return 0.0, n_unit, 0.0

    # Orient normal toward camera majority
    if n_neg > n_pos:
        oriented_normal = -n_unit
        signed_d = -distances
    else:
        oriented_normal = n_unit
        signed_d = distances

    # Use only same-side cameras for height calc
    same_side = signed_d[signed_d > 0]
    mean_height = float(same_side.mean()) if len(same_side) > 0 else 0.0
    height_std = float(same_side.std()) if len(same_side) > 0 else 1.0

    # Score factors:
    #   - inlier_count (raw mass of plane)
    #   - consistency (cameras on the same side)
    #   - height_consistency (low std relative to mean — cameras at similar height)
    height_consistency = 1.0 / (1.0 + height_std / max(mean_height, 1e-3))

    # Gravity prior (if provided): align oriented_normal with cameras' "up" axis.
    # alignment ∈ [-1, 1]; we want it close to +1 (normal points toward gravity-up).
    if gravity_prior is not None and gravity_confidence > 0:
        alignment = float(np.dot(oriented_normal, gravity_prior))
        # Map [-1, 1] to [0, 1]; square to penalize misaligned harder.
        gravity_factor = ((alignment + 1.0) / 2.0) ** 2
        # Blend by confidence: at conf=0, no effect; at conf=1, full penalty.
        gravity_multiplier = (1.0 - gravity_confidence) + gravity_confidence * gravity_factor
    else:
        gravity_multiplier = 1.0

    score = inlier_count * consistency * height_consistency * gravity_multiplier
    return float(score), oriented_normal, mean_height


def estimate_gravity_from_cameras(cams2world):
    """Infer the world-space 'up' direction from the cameras' rotations.

    MAtCha-SfM uses the OpenCV camera convention: +X right, +Y DOWN, +Z forward
    (looking direction). Therefore the camera's "up" in image-space is -Y_cam,
    and in world space it's -cams2world[:, :3, 1] for each camera.

    Returns a tuple (up, confidence). When all cameras agree on "up" (e.g.,
    handheld upright captures), confidence ≈ 1. When cameras roll across a
    wide range (e.g., a 360° around-an-object capture with bank), confidence
    drops toward 0 — the prior is unreliable and shouldn't dominate scoring.

    Confidence is the magnitude of the mean unit-up vector across cameras
    (a circular-statistics measure of agreement on direction).
    """
    cams2world = np.asarray(cams2world)
    # Each camera's world-space up = -Y column of its cams2world rotation.
    per_camera_up = -cams2world[:, :3, 1]
    # Normalize each (should already be unit-length but be safe)
    per_camera_up = per_camera_up / np.linalg.norm(per_camera_up, axis=1, keepdims=True)
    mean_up = per_camera_up.mean(axis=0)
    confidence = float(np.linalg.norm(mean_up))  # 1 = perfect agreement, 0 = canceling
    if confidence < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return (mean_up / confidence), confidence


def rotation_to_z_up(target_up):
    """Rotation matrix R that rotates target_up to +z (i.e., R @ target_up = [0,0,1]).
    Uses the Rodrigues formula via cross-product axis.
    """
    target_up = target_up / np.linalg.norm(target_up)
    z = np.array([0.0, 0.0, 1.0])
    cos_a = np.dot(target_up, z)
    if cos_a > 0.9999:
        return np.eye(3)
    if cos_a < -0.9999:
        # 180° around any axis perpendicular to z
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(target_up, z)
    axis = axis / np.linalg.norm(axis)
    sin_a = np.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R


def main():
    ap = argparse.ArgumentParser(description="Phase B1: orient a MAtCha mesh to z-up, floor at z=0.")
    ap.add_argument("--tetra", required=True, help="Path to MAtCha tetra mesh PLY")
    ap.add_argument("--cameras", required=True, help="Path to MAtCha cameras.json")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--num-candidates", type=int, default=4,
                    help="Number of RANSAC plane candidates to test")
    ap.add_argument("--ransac-distance", type=float, default=0.05,
                    help="RANSAC inlier distance threshold (in mesh units)")
    ap.add_argument("--sample-points", type=int, default=200_000,
                    help="Number of vertices to subsample for plane fitting")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[1] Loading mesh: {args.tetra}")
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(args.tetra)
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"    {n_verts:,} verts / {n_tris:,} tris in {time.time() - t0:.1f}s")

    print(f"[2] Subsampling {min(args.sample_points, n_verts):,} vertices for plane fitting")
    verts = np.asarray(mesh.vertices)
    if n_verts > args.sample_points:
        idx = np.random.RandomState(seed=0).choice(n_verts, args.sample_points, replace=False)
        sampled = verts[idx]
    else:
        sampled = verts

    print(f"[3] RANSAC plane segmentation (top {args.num_candidates} candidates)")
    candidates = fit_candidate_planes(
        sampled,
        num_planes=args.num_candidates,
        distance_threshold=args.ransac_distance,
    )
    for i, (eq, n_in, _) in enumerate(candidates):
        print(f"    cand {i}: normal=({eq[0]:+.3f}, {eq[1]:+.3f}, {eq[2]:+.3f}), "
              f"d={eq[3]:+.3f}, inliers={n_in:,}")

    print(f"[4] Loading {args.cameras}")
    with open(args.cameras) as f:
        cams_data = json.load(f)
    cams2world = np.asarray(cams_data["cams2world"], dtype=np.float64)
    cam_centers = cams2world[:, :3, 3]
    print(f"    {len(cam_centers)} cameras, "
          f"centroid={cam_centers.mean(axis=0)}, "
          f"spread={cam_centers.std(axis=0)}")

    # Estimate gravity from camera "up" axes — when the photographer holds the
    # camera roughly upright, every camera's image-up is the world-up. Average
    # over all cameras gets a robust prior for which axis is gravity.
    gravity_up, gravity_conf = estimate_gravity_from_cameras(cams2world)
    print(f"    gravity prior (avg camera-up): "
          f"({gravity_up[0]:+.3f}, {gravity_up[1]:+.3f}, {gravity_up[2]:+.3f})  "
          f"confidence={gravity_conf:.3f}")
    if gravity_conf < 0.5:
        print(f"    [warning] cameras disagree on 'up' (conf < 0.5); "
              f"prior will have weak influence")

    print("[5] Scoring candidates as floor planes (with gravity prior)")
    scored = []
    for i, (eq, n_in, _) in enumerate(candidates):
        score, oriented_n, mean_h = score_floor_candidate(
            eq, n_in, cam_centers,
            gravity_prior=gravity_up,
            gravity_confidence=gravity_conf,
        )
        # Also compute alignment for the log so we can see the prior's effect
        alignment = float(np.dot(oriented_n, gravity_up))
        scored.append((score, eq, oriented_n, mean_h, i))
        print(f"    cand {i}: score={score:.0f}  "
              f"oriented_normal=({oriented_n[0]:+.3f}, {oriented_n[1]:+.3f}, {oriented_n[2]:+.3f})  "
              f"mean_h={mean_h:+.3f}  "
              f"gravity-align={alignment:+.3f}")

    scored.sort(key=lambda x: -x[0])
    if scored[0][0] == 0:
        print("ERROR: no candidate plane has cameras consistently on one side. "
              "Is the cameras.json from the same MAtCha run as the mesh?")
        sys.exit(1)

    best_score, best_eq, oriented_n, mean_h, best_idx = scored[0]
    print(f"[6] Selected candidate {best_idx} (score={best_score:.0f}). "
          f"Floor plane normal (toward cameras) = {oriented_n}")

    print("[7] Computing rotation R that maps oriented_normal -> +z")
    R = rotation_to_z_up(oriented_n)

    # After rotation, the plane equation becomes (R*n)·x + d = 0; the new
    # normal is +z so the plane is z = -d / (R @ n)[2]. We want z = 0 at the
    # plane, so translate by -that value in the new frame.
    new_normal = R @ oriented_n
    # plane in new frame: new_normal · x' + d_new = 0
    # use the old plane offset adjusted: take any old inlier point, transform, find offset
    # Simplest: original plane passes through point p0 = -d * (a,b,c) / |(a,b,c)|^2 on ORIGINAL normal direction
    # Since we may have flipped the normal, use sign-corrected offset:
    a, b, c, d_orig = best_eq
    if np.dot([a, b, c], oriented_n) < 0:
        d_orig = -d_orig
    p0_world = -d_orig * oriented_n  # a point on the plane (closest origin point along oriented_n)
    p0_rotated = R @ p0_world
    z_shift = -p0_rotated[2]  # shift the entire scene by this in z so plane is at z=0

    print(f"    new_normal after R: {new_normal} (should be ~+z)")
    print(f"    z shift to place plane at z=0: {z_shift:+.3f}")

    print("[8] Applying transform to mesh + cameras")
    mesh_oriented = o3d.geometry.TriangleMesh(mesh)  # copy
    pts = np.asarray(mesh_oriented.vertices)
    pts_new = (R @ pts.T).T + np.array([0.0, 0.0, z_shift])
    mesh_oriented.vertices = o3d.utility.Vector3dVector(pts_new)
    mesh_oriented.compute_vertex_normals()

    cams_new = (R @ cam_centers.T).T + np.array([0.0, 0.0, z_shift])
    print(f"    cameras after transform: mean_z={cams_new[:, 2].mean():+.3f} (should be > 0)")
    print(f"    mesh z range after transform: "
          f"[{pts_new[:, 2].min():+.3f}, {pts_new[:, 2].max():+.3f}]")

    # Write outputs
    out_ply = os.path.join(args.output, "oriented_tetra.ply")
    out_obj = os.path.join(args.output, "oriented_tetra.obj")
    o3d.io.write_triangle_mesh(out_ply, mesh_oriented)
    print(f"[9] Wrote {out_ply} ({os.path.getsize(out_ply) / 1024 / 1024:.1f} MB)")
    o3d.io.write_triangle_mesh(out_obj, mesh_oriented)
    print(f"    Wrote {out_obj} ({os.path.getsize(out_obj) / 1024 / 1024:.1f} MB)")

    # Save oriented cameras + transform metadata
    out_cams = os.path.join(args.output, "oriented_cameras.json")
    cams_payload = {
        "cameras_world_oriented": cams_new.tolist(),
        "rotation": R.tolist(),
        "z_shift": z_shift,
        "floor_plane_original": list(map(float, best_eq)),
        "floor_normal_oriented_toward_cameras": oriented_n.tolist(),
        "mean_camera_height_above_floor": float(mean_h),
    }
    with open(out_cams, "w") as f:
        json.dump(cams_payload, f, indent=2)
    print(f"    Wrote {out_cams}")

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
