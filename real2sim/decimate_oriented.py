import open3d as o3d, os, sys, time, json
SCENES = sys.argv[1:]
for scene in SCENES:
    base = f"/data/matcha_output/{scene}/oriented"
    src = f"{base}/oriented_tetra.ply"
    if not os.path.exists(src):
        print(f"SKIP {scene}: no {src}")
        continue
    print(f"=== {scene} ===")
    mesh = o3d.io.read_triangle_mesh(src)
    print(f"  in: {len(mesh.vertices):,}v / {len(mesh.triangles):,}t")
    for target in (200_000,):
        t0 = time.time()
        dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        dec.remove_degenerate_triangles()
        dec.remove_unreferenced_vertices()
        dec.remove_duplicated_triangles()
        dec.compute_vertex_normals()
        out_obj = f"{base}/oriented_{target//1000}k.obj"
        out_ply = f"{base}/oriented_{target//1000}k.ply"
        o3d.io.write_triangle_mesh(out_obj, dec)
        o3d.io.write_triangle_mesh(out_ply, dec)
        sz_obj = os.path.getsize(out_obj)/1024/1024
        sz_ply = os.path.getsize(out_ply)/1024/1024
        print(f"  {target//1000}k: {len(dec.vertices):,}v/{len(dec.triangles):,}t in {time.time()-t0:.0f}s  obj={sz_obj:.1f}MB ply={sz_ply:.1f}MB")
