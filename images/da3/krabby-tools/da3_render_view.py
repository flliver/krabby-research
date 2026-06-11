#!/usr/bin/env python3
"""da3_render_view.py — render DA3 gaussians from a saved scene view (STO-SCN-061).

Aligns DA3's coordinate frame to the scene's ORIENTED frame via the
cameras both pipelines solved (gauge_align, orientation-augmented
Umeyama — the photo-spine machinery, T-013), maps the saved schema-5
view camera into DA3's frame, and renders the gaussians there via
DA3's own gs_video exporter.

Runs INSIDE the krabby-da3 container with the research repo's
real2sim/ mounted at /tools (for gauge_align):

    python /tools/da3_render_view.py \
        --scene /scene \
        --matcha-run pipeline-matcha/run-8-dense-strong \
        --da3-run pipeline-da3/run-8-giant \
        --view overhead-grass-quality \
        --out /scene/<da3-run>/renders

Conventions handled:
  - mast3r cameras.json: cams2world c2w, OpenCV (+Z fwd)
  - oriented_cameras.json: R (world rotation), z_shift; oriented pose =
    (R @ R_cam, R @ C + [0,0,z_shift])
  - DA3 npz extrinsics: (N,3,4) — convention VERIFIED at runtime by
    trying both w2c and c2w against the oriented set and keeping the
    one whose alignment residual is sane (printed; hard-fails if both
    are garbage, T-002)
  - schema-5 view: world_position + world_rotation_quat_wxyz (c2w,
    OpenCV) in the oriented frame; lens_mm/sensor → fx = W*f/sensor_w
    (Blender AUTO sensor fit, square pixels)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tools")
from gauge_align import align_camera_sets  # noqa: E402


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--matcha-run", required=True)
    ap.add_argument("--da3-run", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default="/opt/checkpoints/DA3NESTED-GIANT-LARGE-1.1")
    args = ap.parse_args()

    scene = Path(args.scene)
    mat_data = next((scene / args.matcha_run).glob("transform-*/data"))
    da3_data = next((scene / args.da3_run).glob("transform-*/data"))

    # --- oriented-frame poses of the matcha cameras -----------------------
    cams = json.loads((mat_data / "mast3r_sfm" / "cameras.json").read_text())
    ori = json.loads((mat_data / "oriented" / "oriented_cameras.json").read_text())
    R_o = np.asarray(ori["rotation"], dtype=np.float64)
    z_shift = float(ori["z_shift"])
    order = np.argsort([fp.rsplit("/", 1)[-1] for fp in cams["filepaths"]])
    c2w = np.asarray(cams["cams2world"], dtype=np.float64)[order]
    C_mat = (R_o @ c2w[:, :3, 3].T).T + np.array([0.0, 0.0, z_shift])
    R_mat = np.einsum("ij,njk->nik", R_o, c2w[:, :3, :3])

    # --- DA3 poses (npz; convention auto-verified) -------------------------
    npz = np.load(da3_data / "exports" / "npz" / "results.npz")
    ext = np.asarray(npz["extrinsics"], dtype=np.float64)  # (N,3,4)
    n = ext.shape[0]
    assert n == len(C_mat), f"camera count mismatch: {n} vs {len(C_mat)}"

    candidates = {}
    # hypothesis A: ext is w2c → C = -R^T t, R_c2w = R^T
    Rw, tw = ext[:, :3, :3], ext[:, :3, 3]
    candidates["w2c"] = (np.einsum("nji,nj->ni", Rw, -tw),
                         np.transpose(Rw, (0, 2, 1)))
    # hypothesis B: ext is c2w → C = t, R_c2w = R
    candidates["c2w"] = (ext[:, :3, 3].copy(), ext[:, :3, :3].copy())

    best = None
    for name, (C_da3, R_da3) in candidates.items():
        try:
            res = align_camera_sets(C_da3, C_mat, src_rotations=R_da3,
                                    dst_rotations=R_mat)
        except TypeError:
            res = align_camera_sets(C_da3, C_mat)
        spread = np.linalg.norm(C_mat - C_mat.mean(0), axis=1).mean()
        rel = res["max_residual"] / spread
        print(f"hypothesis {name}: max residual {res['max_residual']:.4f} "
              f"({rel*100:.1f}% of spread)")
        if best is None or res["max_residual"] < best[1]["max_residual"]:
            best = (name, res, C_da3, R_da3)
    name, res, C_da3, R_da3 = best
    spread = np.linalg.norm(C_mat - C_mat.mean(0), axis=1).mean()
    if res["max_residual"] > 0.10 * spread:
        sys.exit(f"ERROR: best alignment ({name}) residual "
                 f"{res['max_residual']:.4f} > 10% of camera spread "
                 f"{spread:.4f} — frames don't correspond; refusing to render.")
    print(f"using {name}; scale {res['scale']:.4f}")
    s, R_al, t_al = res["scale"], np.asarray(res["R"]), np.asarray(res["t"])

    # --- saved view → DA3 frame -------------------------------------------
    views = json.loads((scene / "cameras.json").read_text())
    v = next(x for x in views["views"] if x["name"] == args.view)
    C_v = np.asarray(v["world_position"], dtype=np.float64)
    R_v = quat_wxyz_to_R(v["world_rotation_quat_wxyz"])
    C_vd = (R_al.T @ (C_v - t_al)) / s
    R_vd = R_al.T @ R_v

    W, H = v.get("render_resolution", [1920, 1080])
    fx = W * float(v["lens_mm"]) / float(v.get("sensor_width_mm", 36.0))
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float64)

    w2c = np.eye(4)
    w2c[:3, :3] = R_vd.T
    w2c[:3, 3] = -R_vd.T @ C_vd

    # --- render via DA3's renderer directly (STO-SCN-061) ------------------
    # The gs_video exporter OOMs 16GB GPUs at 1080p (it composes layout
    # videos with the model still resident) and silently switches to a
    # "wander" trajectory for single views. We call the chunked renderer
    # ourselves: model freed first, trj_mode="original".
    import gc
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode

    model = DepthAnything3.from_pretrained(args.model_dir).to("cuda")
    images = sorted(str(p) for p in (scene / "input" / "src").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    pred = model.inference(images, infer_gs=True)
    gs_world = pred.gaussians
    # mirror export_to_gs_video: metric predictions' camera translations
    # must be divided by scale_factor to land in gaussian space
    if getattr(pred, "is_metric", False) and getattr(pred, "scale_factor", None) is not None:
        w2c_render = w2c.copy()
        # w2c translation t = -R^T C ; scaling C scales t linearly
        w2c_render[:3, 3] /= float(pred.scale_factor)
    else:
        w2c_render = w2c
    del model, pred
    gc.collect()
    torch.cuda.empty_cache()

    # renderer refuses n_views=1 with trj_mode="original" — duplicate the
    # camera and keep frame 0
    tgt_extrs = torch.tensor(w2c_render, dtype=torch.float32, device="cuda")[None, None].repeat(1, 2, 1, 1)
    tgt_intrs = torch.tensor(K, dtype=torch.float32, device="cuda")[None, None].repeat(1, 2, 1, 1)
    color, _depth = run_renderer_in_chunk_w_trj_mode(
        gaussians=gs_world,
        extrinsics=tgt_extrs,
        intrinsics=tgt_intrs,
        image_shape=(H, W),
        chunk_size=1,
        trj_mode="original",
        use_sh=True,
    )
    img = (color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.view}.png"
    from PIL import Image
    Image.fromarray(img).save(png)
    print(f"wrote {png}")
    # alignment record for the sidecar
    (out / f"{args.view}.alignment.json").write_text(json.dumps({
        "method": "gauge_align orientation-augmented Umeyama over 8 shared cameras",
        "da3_extrinsics_convention": name,
        "scale": s, "max_residual": res["max_residual"],
        "residual_frac_of_spread": res["max_residual"] / spread,
        "story": "STO-SCN-061",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
