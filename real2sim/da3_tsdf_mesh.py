#!/usr/bin/env python3
"""da3_tsdf_mesh.py — fuse DA3 depth maps into a mesh (EPI-SCN-FEEDFORWARD-RECON).

The DA3 pipeline's deliverable-grade output: TSDF fusion (Open3D) over
DA3's dense per-view depths + poses → triangle mesh → similarity
transform into the scene's ORIENTED frame (via gauge_align over the
cameras both pipelines solved, same alignment as da3_render_view.py).

Writes `tsdf_meshes/multires_tsdf_post_oriented.ply` into the DA3
run's transform data dir — the standard tsdf mesh-source path, so
`render_comparison_matrix.sh --mesh-source tsdf` picks the run up as a
variant with zero special-casing.

CPU-only. Usage:
    uv run --with open3d --with numpy python3 da3_tsdf_mesh.py \
        --scene /var/krabby/scenes/006-kubota \
        --matcha-run pipeline-matcha/run-8-dense-strong \
        --da3-run pipeline-da3/run-8-giant \
        [--voxel-frac 0.004] [--conf-percentile 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))
from gauge_align import align_camera_sets  # noqa: E402


def alignment_da3_to_oriented(scene: Path, matcha_run: str, ext: np.ndarray):
    """Similarity (s, R, t): DA3 npz frame → matcha oriented frame.
    Same math as da3_render_view.py (w2c convention, verified there)."""
    mat_data = next((scene / matcha_run).glob("transform-*/data"))
    cams = json.loads((mat_data / "mast3r_sfm" / "cameras.json").read_text())
    ori = json.loads((mat_data / "oriented" / "oriented_cameras.json").read_text())
    R_o = np.asarray(ori["rotation"]); z = float(ori["z_shift"])
    order = np.argsort([fp.rsplit("/", 1)[-1] for fp in cams["filepaths"]])
    c2w = np.asarray(cams["cams2world"])[order]
    C_mat = (R_o @ c2w[:, :3, 3].T).T + np.array([0.0, 0.0, z])
    R_mat = np.einsum("ij,njk->nik", R_o, c2w[:, :3, :3])
    Rw, tw = ext[:, :3, :3], ext[:, :3, 3]
    C_da3 = np.einsum("nji,nj->ni", Rw, -tw)
    R_da3 = np.transpose(Rw, (0, 2, 1))
    res = align_camera_sets(C_da3, C_mat, src_rotations=R_da3, dst_rotations=R_mat)
    spread = np.linalg.norm(C_mat - C_mat.mean(0), axis=1).mean()
    frac = res["max_residual"] / spread
    print(f"alignment: max residual {res['max_residual']:.4f} ({frac*100:.1f}% of spread), "
          f"scale {res['scale']:.4f}")
    if frac > 0.10:
        sys.exit("ERROR: alignment residual >10% of camera spread — refusing.")
    return res["scale"], np.asarray(res["R"]), np.asarray(res["t"]), res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--matcha-run", required=True)
    ap.add_argument("--da3-run", required=True)
    ap.add_argument("--voxel-frac", type=float, default=0.004,
                    help="voxel size as fraction of median scene depth span")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    args = ap.parse_args()

    scene = Path(args.scene)
    da3_data = next((scene / args.da3_run).glob("transform-*/data"))
    npz = np.load(da3_data / "exports" / "npz" / "results.npz")
    depth = npz["depth"].astype(np.float32)        # (N,H,W)
    conf = npz["conf"]
    img = npz["image"]                              # (N,H,W,3) uint8
    ext = npz["extrinsics"].astype(np.float64)      # (N,3,4) w2c
    K = npz["intrinsics"].astype(np.float64)        # (N,3,3)
    n, H, W = depth.shape

    s, R_al, t_al, ares = alignment_da3_to_oriented(scene, args.matcha_run, ext)

    thr = np.percentile(conf, args.conf_percentile)
    d_valid = depth[conf > thr]
    span = float(np.percentile(d_valid, 95))
    voxel = span * args.voxel_frac
    sdf_trunc = 4.0 * voxel
    print(f"depth 95pct {span:.2f} → voxel {voxel:.4f}, sdf_trunc {sdf_trunc:.4f}")

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(n):
        d = depth[i].copy()
        d[conf[i] <= thr] = 0.0  # invalid → no integration
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(img[i])),
            o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=float(span * 1.5),
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        w2c4 = np.eye(4); w2c4[:3, :4] = ext[i]
        vol.integrate(rgbd, intr, w2c4)
        print(f"  integrated view {i+1}/{n}")

    mesh = vol.extract_triangle_mesh()
    print(f"fused: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # DA3 frame → oriented frame (rotate+translate via 4x4, then uniform scale)
    T = np.eye(4); T[:3, :3] = R_al; T[:3, 3] = t_al / s
    mesh.transform(T)
    mesh.scale(s, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()

    out_dir = da3_data / "tsdf_meshes"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "multires_tsdf_post_oriented.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"wrote {out} ({out.stat().st_size/2**20:.0f} MB)")

    (out_dir / "fusion_record.json").write_text(json.dumps({
        "tool": "real2sim/da3_tsdf_mesh.py",
        "voxel": voxel, "sdf_trunc": sdf_trunc,
        "conf_percentile": args.conf_percentile,
        "n_views": n, "depth_res": [H, W],
        "verts": len(mesh.vertices), "tris": len(mesh.triangles),
        "alignment": {"scale": s, "max_residual": ares["max_residual"]},
        "frame": "matcha-oriented (floor z=0 inherited via alignment)",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
