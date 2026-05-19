"""Decimate MAtCha's tetra mesh for downstream use (Blender, IsaacSim).

The raw `tetra_mesh_binary_search_7.ply` from MAtCha is ~422 MB / 21M
triangles. That's far too dense for sim or interactive viewing. Open3D's
quadric edge-collapse decimation reduces to a target triangle budget
while preserving overall geometry.

Empirical (2026-04-30, RTX 5080, scene 004 sky-house-dining):
  - Source:      10.5M verts / 21M tris
  - 200K target: 86,997 verts / 200,000 tris   →  15 MB OBJ /  6.7 MB PLY
  - 500K target: 238,488 verts / 500,000 tris  →  41 MB OBJ /  18 MB PLY
  - Decimation time: ~3.5 min per target on the matcha-build container

200K is the comparison-fair size against MASt3R-SLAM's mesh (also 200K).
500K is the higher-fidelity option for visual inspection.

Run inside the matcha container:
  source /opt/matcha/bin/activate && python decimate.py
"""
import open3d as o3d
import os
import time

SRC = "/data/matcha_output/004-sky-house/tetra_meshes/tetra_mesh_binary_search_7.ply"
OUT_DIR = "/data/matcha_output/004-sky-house/mesh"
TARGETS = (200_000, 500_000)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {SRC} ...")
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(SRC)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  vertices:  {len(mesh.vertices):,}")
    print(f"  triangles: {len(mesh.triangles):,}")

    for target in TARGETS:
        print(f"\nDecimating to {target:,} triangles...")
        t0 = time.time()
        dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        dec.remove_degenerate_triangles()
        dec.remove_unreferenced_vertices()
        dec.remove_duplicated_triangles()
        dec.compute_vertex_normals()
        print(f"  decimated in {time.time() - t0:.1f}s")
        print(f"  result: {len(dec.vertices):,} verts / {len(dec.triangles):,} tris")

        obj_path = f"{OUT_DIR}/sky_house_matcha_{target // 1000}k.obj"
        ply_path = f"{OUT_DIR}/sky_house_matcha_{target // 1000}k.ply"
        o3d.io.write_triangle_mesh(obj_path, dec)
        o3d.io.write_triangle_mesh(ply_path, dec)

        sz_obj = os.path.getsize(obj_path) / 1024 / 1024
        sz_ply = os.path.getsize(ply_path) / 1024 / 1024
        print(f"  wrote: {obj_path} ({sz_obj:.1f} MB)")
        print(f"         {ply_path} ({sz_ply:.1f} MB)")


if __name__ == "__main__":
    main()
