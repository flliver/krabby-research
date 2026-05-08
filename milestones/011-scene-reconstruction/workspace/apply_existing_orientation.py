"""Apply the rotation + z_shift from an existing oriented_cameras.json to a
new mesh. Used to bring an alternate mesh (e.g. multires_tsdf_post.ply)
into the same world frame as the tetra-derived oriented_500k_colored_culled.ply.

Args:
    --in-mesh    raw mesh PLY (un-oriented)
    --orientation oriented_cameras.json holding 'rotation' (3x3) + 'z_shift'
    --out-mesh   destination PLY

Run inside the matcha-build container (open3d available there).
"""
import argparse
import json
import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-mesh", required=True)
    p.add_argument("--orientation", required=True)
    p.add_argument("--out-mesh", required=True)
    args = p.parse_args()

    with open(args.orientation) as f:
        meta = json.load(f)
    R = np.asarray(meta["rotation"], dtype=np.float64)
    z_shift = float(meta["z_shift"])

    print(f"  reading {args.in_mesh}")
    mesh = o3d.io.read_triangle_mesh(args.in_mesh)
    n_verts = len(mesh.vertices)
    print(f"  {n_verts:,} verts / {len(mesh.triangles):,} tris")

    verts = np.asarray(mesh.vertices)
    rotated = verts @ R.T                    # apply rotation
    rotated[:, 2] += z_shift                 # then z translation
    mesh.vertices = o3d.utility.Vector3dVector(rotated)
    mesh.compute_vertex_normals()

    print(f"  writing {args.out_mesh}")
    o3d.io.write_triangle_mesh(args.out_mesh, mesh)
    print(f"  done — z range now: [{rotated[:,2].min():.2f}, {rotated[:,2].max():.2f}] "
          f"(z=0 is floor)")


if __name__ == "__main__":
    main()
