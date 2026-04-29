#!/bin/bash
# T2: Mesh Conditioning & USD Export
# Input:  /data/scenes/<scene>/dense/fused.ply  (from T1)
# Output: /data/scenes/<scene>/mesh/  (conditioned mesh + collision proxy)
#
# Usage: ./run_mesh_conditioning.sh <scene> [poisson_depth]
#   e.g. ./run_mesh_conditioning.sh 002-patio-dewarped 9
#
# Pipeline:
#   1. Load dense point cloud
#   2. Estimate normals
#   3. Poisson surface reconstruction
#   4. Density-based cropping (remove extrapolation artifacts)
#   5. Mesh cleanup (floaters, degenerates, holes)
#   6. Smoothing (Taubin — no shrinkage)
#   7. Decimation (quadric edge collapse)
#   8. Export conditioned mesh as OBJ
#   9. Generate collision proxy (convex decomposition)

set -euo pipefail

SCENE="${1:?Usage: $0 <scene-name> [poisson_depth]}"
POISSON_DEPTH="${2:-9}"
SCENE_DIR="/data/scenes/${SCENE}"
DENSE_PLY="${SCENE_DIR}/dense/fused.ply"
MESH_DIR="${SCENE_DIR}/mesh"

if [ ! -f "${DENSE_PLY}" ]; then
    echo "ERROR: Dense point cloud not found: ${DENSE_PLY}"
    echo "Run run_colmap_dense.sh first."
    exit 1
fi

echo "=== Mesh Conditioning ==="
echo "Scene:         ${SCENE}"
echo "Input:         ${DENSE_PLY}"
echo "Poisson depth: ${POISSON_DEPTH}"
echo "Output:        ${MESH_DIR}"
echo ""

mkdir -p "${MESH_DIR}"

python3 << PYEOF
import open3d as o3d
import numpy as np
import os
import trimesh

DENSE_PLY = "${DENSE_PLY}"
MESH_DIR = "${MESH_DIR}"
POISSON_DEPTH = ${POISSON_DEPTH}

# --- Step 1: Load dense point cloud ---
print("[1/8] Loading dense point cloud...")
pcd = o3d.io.read_point_cloud(DENSE_PLY)
print(f"  Points: {len(pcd.points):,}")

# --- Step 2: Statistical outlier removal ---
print("[2/8] Removing statistical outliers...")
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
removed = len(pcd.points) - len(pcd_clean.points)
print(f"  Removed {removed:,} outliers ({removed/len(pcd.points)*100:.1f}%)")
print(f"  Remaining: {len(pcd_clean.points):,}")

# --- Step 3: Estimate normals ---
print("[3/8] Estimating normals...")
pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd_clean.orient_normals_consistent_tangent_plane(k=15)

# --- Step 4: Poisson surface reconstruction ---
print(f"[4/8] Poisson reconstruction (depth={POISSON_DEPTH})...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_clean, depth=POISSON_DEPTH, linear_fit=True
)
print(f"  Vertices:  {len(mesh.vertices):,}")
print(f"  Triangles: {len(mesh.triangles):,}")

# --- Step 5: Density-based cropping ---
print("[5/8] Density-based cropping (removing Poisson extrapolation)...")
densities = np.asarray(densities)
density_threshold = np.quantile(densities, 0.05)
vertices_to_remove = densities < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)
print(f"  After crop — Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")

# Save raw Poisson mesh for comparison
raw_path = os.path.join(MESH_DIR, "mesh_raw.ply")
o3d.io.write_triangle_mesh(raw_path, mesh)
print(f"  Saved raw mesh: {raw_path}")

# --- Step 6: Cleanup ---
print("[6/8] Mesh cleanup...")
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
# Remove small disconnected components
triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
# Keep only clusters with > 1% of total triangles
min_cluster_size = max(100, int(len(mesh.triangles) * 0.01))
triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_size
mesh.remove_triangles_by_mask(triangles_to_remove)
mesh.remove_unreferenced_vertices()
print(f"  After cleanup — Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")

# --- Step 7: Smoothing (Taubin — no shrinkage) ---
print("[7/8] Taubin smoothing...")
mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
mesh.compute_vertex_normals()

# --- Step 8: Decimation ---
target_triangles = min(len(mesh.triangles), 200_000)
if len(mesh.triangles) > target_triangles:
    print(f"[8/8] Decimating {len(mesh.triangles):,} → {target_triangles:,} triangles...")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
else:
    print(f"[8/8] Skip decimation ({len(mesh.triangles):,} triangles already under 200k)")

print(f"  Final — Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")

# --- Export ---
conditioned_obj = os.path.join(MESH_DIR, "mesh_conditioned.obj")
conditioned_ply = os.path.join(MESH_DIR, "mesh_conditioned.ply")
o3d.io.write_triangle_mesh(conditioned_obj, mesh)
o3d.io.write_triangle_mesh(conditioned_ply, mesh)
print(f"  Saved: {conditioned_obj}")
print(f"  Saved: {conditioned_ply}")

# --- Collision proxy (convex decomposition via trimesh/V-HACD) ---
print("")
print("=== Collision Proxy ===")
try:
    tm = trimesh.load(conditioned_obj)
    if hasattr(tm, 'convex_decomposition'):
        print("Running V-HACD convex decomposition...")
        convex_parts = tm.convex_decomposition(maxhulls=32)
        if not isinstance(convex_parts, list):
            convex_parts = [convex_parts]
        collision_mesh = trimesh.util.concatenate(convex_parts)
        collision_path = os.path.join(MESH_DIR, "collision_proxy.obj")
        collision_mesh.export(collision_path)
        print(f"  Collision proxy: {len(convex_parts)} convex hulls")
        print(f"  Saved: {collision_path}")
    else:
        print("  V-HACD not available — using simplified convex hull as fallback")
        hull = tm.convex_hull
        hull_path = os.path.join(MESH_DIR, "collision_proxy.obj")
        hull.export(hull_path)
        print(f"  Saved convex hull: {hull_path}")
except Exception as e:
    print(f"  WARNING: Collision proxy generation failed: {e}")
    print(f"  Skipping — can be generated manually later")

print("")
print("=== Done ===")
print(f"Conditioned mesh: {conditioned_obj}")
print(f"Next step: USD conversion for IsaacSim")
PYEOF
