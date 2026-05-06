"""Auto-localize a reference image's camera by extending an existing 12-frame
MAtCha-SfM run with the reference image as a 13th frame.

This is the structural answer to "where in 3D space was MAtCha's published
reference image rendered from?" Instead of hand-placing a Blender camera to
match the reference visually (the manual fallback in scene_tsdf_ref.blend),
we re-run MASt3R-SfM with the reference image included in the input pool,
then Procrustes-align the new 12-camera frame back to the original 12-camera
frame and apply the saved orient transform — landing the reference's pose
in the same world frame as the rest of the scene.

The output is a `purpose=reference-match` view written into the scene's
comparison_views.json, ready to be picked up by build_blender_scene.py on
the next .blend regeneration.

## Pipeline

  Local                          tbeeprz                    Local
  -----                          -------                    -----
  Convert ref PNG to             Build sandbox dir           Pull cameras.json
  faux-RGB JPG (greyscale  →     with 12 frame symlinks +    back, run Procrustes
  channel-replicated 3 ch)       1 ref JPG                   alignment math
  Push to remote sandbox    →    Run train.py --sfm_only     Apply world_orient
                                 with --image_idx 0..12      Convert to OpenCV
                                 Wait ~2 min for 13-frame    quat+pos, write to
                                 SfM (RTX 5080)              comparison_views.json

## Limitations & known issues

- **Shared-focal SfM averages focal across photos and the rendered reference.**
  If the paper's reference render was shot at a much wider FOV than the
  source photos, the SfM-derived focal is a compromise — not the true
  reference focal. Position can be off by tens of cm even when rotation is
  within a few degrees. Visual match is good but not perfect.
- **Greyscale-rendered reference vs RGB photos.** Most descriptors are
  greyscale-native so feature matching works, but the appearance gap may
  bias the per-frame pose. PnP localization (variant 2 in note
  2026-05-01T174651-sliding-window-sfm-and-the-keyframe-localization-alternative)
  is the principled fallback.
- **Single-shared-frame SfM.** MASt3R-SfM with --image_idx 0..12 fits the
  full reconstruction, including small adjustments to the original 12.
  Procrustes lands them back on the original frame; for our 12-shared
  bicycle case residuals were sub-cm.

## Usage

  python localize_reference_image.py \\
      --scene-variant /private/var/krabby/workspace/.../dtu-bicycle-curated-12-dense-strong \\
      --reference-image /private/var/krabby/workspace/.../dtu-bicycle/reference/tsdf_multires.png \\
      --comparison-views-out /private/var/krabby/workspace/.../dtu-bicycle/comparison_views.json \\
      --reference-name cam_ref_auto \\
      [--remote-host tbeeprz] \\
      [--remote-data-root '/home/jeremy/outposts/krabby/data/011-scene-reconstruction'] \\
      [--container matcha-build]

Output: appends a view named `--reference-name` (default `cam_ref_auto`) to
the comparison_views.json with `purpose=reference-match`,
`auto_localized=true`, `localization_method=mast3r_sfm_extend`.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image


def _run(cmd, **kw):
    """Run a subprocess; raise on non-zero. Return stdout."""
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd)}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), check=True, text=True,
        capture_output=True, **kw,
    )
    if r.stdout.strip():
        print(r.stdout)
    return r.stdout


def stage_reference_jpg(reference_image_path, out_jpg_path, quality=92):
    """Convert the reference PNG to a faux-RGB JPG. Channel-replicates
    greyscale automatically via PIL's RGB conversion."""
    im = Image.open(reference_image_path).convert("RGB")
    im.save(out_jpg_path, "JPEG", quality=quality)
    print(f"Staged reference: {out_jpg_path} ({os.path.getsize(out_jpg_path)//1024} KB, {im.size})")


def remote_stage_sandbox(host, remote_data_root, scene_variant_basename,
                         scene_dir_relative, ref_jpg_local_path):
    """On the remote host, build a sandbox dir with relative symlinks to the
    12 source frames + an uploaded reference JPG. Return the container-visible
    sandbox path (always /data/...).

    The relative symlinks let the same paths resolve on host AND in container
    (same filesystem, different mount points)."""
    sandbox_host = f"{remote_data_root}/sfm-ref-localize/{scene_variant_basename}/images"
    sandbox_container = f"/data/sfm-ref-localize/{scene_variant_basename}/images"

    # The relative symlink target: from <data-root>/sfm-ref-localize/<scene>/images/
    # back up to <data-root> = ../../.. then forward to scenes/<variant>/...
    src_relative = f"../../../{scene_dir_relative}/mast3r_sfm/images"

    _run(["ssh", host, f"""
set -e
mkdir -p {shlex.quote(sandbox_host)}
cd {shlex.quote(sandbox_host)}
find . -maxdepth 1 -type l -delete
for f in {shlex.quote(src_relative)}/*.JPG; do
  bn=$(basename "$f")
  ln -sf "{src_relative}/$bn" "$bn"
done
"""])

    # Push the reference JPG (always overwrite to pick up content changes)
    _run(["rsync", "-az", ref_jpg_local_path, f"{host}:{sandbox_host}/_DSC9999_ref.JPG"])

    return sandbox_container


def remote_run_sfm(host, container, sandbox_container_path, output_container_path,
                   n_images=13):
    """Run train.py --sfm_only on the remote container.
    Returns the container-relative path to cameras.json."""
    image_idx = " ".join(str(i) for i in range(n_images))
    cmd = f"""
trap "/usr/local/bin/nanny-progress clear 2>/dev/null" EXIT INT TERM
/usr/local/bin/nanny-progress set 1/1 0 "MAtCha SfM ref-localize ({n_images}-frame)" 2>/dev/null

# Clean previous output (root-owned from container; sudo would help but
# we'll trust the container to overwrite cleanly with --sfm_only).
docker exec {shlex.quote(container)} bash -c "rm -rf {shlex.quote(output_container_path)}/mast3r_sfm" 2>/dev/null || true

T0=$(date +%s)
docker exec {shlex.quote(container)} bash -c '
  source /opt/matcha/bin/activate
  export PYTHONPATH=/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /opt/MAtCha
  python train.py \\
    -s {shlex.quote(sandbox_container_path)} \\
    -o {shlex.quote(output_container_path)} \\
    --sfm_only \\
    --image_idx {image_idx} \\
    --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \\
    --depthanything_encoder vitl 2>&1
'
T1=$(date +%s)
echo "SfM wall time: $((T1-T0))s"
"""
    _run(["ssh", host, cmd])
    return f"{output_container_path}/mast3r_sfm/cameras.json"


def fetch_cameras_json(host, remote_path_container, remote_data_root, local_path):
    """Pull cameras.json from the container path back to the local machine.
    The container path /data/... maps to <remote_data_root>/... on the host."""
    if not remote_path_container.startswith("/data/"):
        raise ValueError(f"Expected container path under /data/, got {remote_path_container}")
    host_path = remote_data_root + remote_path_container[5:]  # strip /data
    _run(["rsync", "-az", f"{host}:{host_path}", local_path])


def umeyama(P, Q):
    """Return scale, R (3x3), t (3,) such that scale*R@P + t ≈ Q.
    P, Q are (N, 3) point clouds."""
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc
    U, S_sv, Vt = np.linalg.svd(H)
    d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d_sign])
    R = Vt.T @ D @ U.T
    var_P = float((Pc * Pc).sum())
    scale = float((np.diag(D) * S_sv).sum() / var_P) if var_P > 0 else 1.0
    t = cQ - scale * R @ cP
    return scale, R, t


def matrix_to_quat_wxyz(M):
    """Convert a 3x3 rotation matrix to (w, x, y, z) quaternion."""
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s
        y = (M[0, 2] - M[2, 0]) / s
        z = (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def align_extract_world_pose(new_cameras_json_path, orig_cameras_json_path,
                             oriented_cameras_json_path, ref_index):
    """Procrustes-align the N+1th camera back to the original frame, then
    apply world_orient. Returns dict with world_position, quat_wxyz (OpenCV),
    lens_mm derived from focal_px, and quality metrics."""
    new = json.load(open(new_cameras_json_path))
    orig = json.load(open(orig_cameras_json_path))
    orient = json.load(open(oriented_cameras_json_path))

    new_c2w = np.asarray(new["cams2world"])
    orig_c2w = np.asarray(orig["cams2world"])

    if new_c2w.shape[0] != orig_c2w.shape[0] + 1:
        raise ValueError(
            f"Expected new SfM to have one more camera than original; "
            f"got {new_c2w.shape[0]} vs {orig_c2w.shape[0]}"
        )

    new_shared = new_c2w[: orig_c2w.shape[0]]
    new_ref = new_c2w[ref_index]

    P_new = new_shared[:, :3, 3]
    P_orig = orig_c2w[:, :3, 3]
    scale, R_align, t_align = umeyama(P_new, P_orig)
    residuals = np.linalg.norm(scale * (P_new @ R_align.T) + t_align - P_orig, axis=1)
    print(f"Procrustes: scale={scale:.6f} det(R)={np.linalg.det(R_align):.6f}")
    print(f"  residuals (m): max={residuals.max():.4f} mean={residuals.mean():.4f}")

    # Apply alignment to the reference's c2w.
    ref_pos_orig = scale * (R_align @ new_ref[:3, 3]) + t_align
    ref_rot_orig = R_align @ new_ref[:3, :3]

    # Apply world_orient (rotation R_orient + translation z_shift, as
    # build_blender_scene.py does at lines 152-155).
    R_orient = np.array(orient["rotation"])
    z_shift = float(orient["z_shift"])
    T_world = np.eye(4)
    T_world[:3, :3] = R_orient
    T_world[2, 3] = z_shift

    ref_c2w_orig = np.eye(4)
    ref_c2w_orig[:3, :3] = ref_rot_orig
    ref_c2w_orig[:3, 3] = ref_pos_orig
    ref_c2w_world = T_world @ ref_c2w_orig

    # Convert to (position, quaternion) in OpenCV convention.
    pos = ref_c2w_world[:3, 3]
    quat = matrix_to_quat_wxyz(ref_c2w_world[:3, :3])

    # Derive lens_mm from MAtCha's per-camera focal_px (in 512-wide image space).
    focal_px = float(new["focals"][ref_index])
    lens_mm = focal_px / 512.0 * 36.0

    return {
        "world_position": [float(p) for p in pos],
        "world_rotation_quat_wxyz": quat,
        "lens_mm": lens_mm,
        "focal_px": focal_px,
        "procrustes_scale": float(scale),
        "procrustes_residual_max_m": float(residuals.max()),
        "procrustes_residual_mean_m": float(residuals.mean()),
        "n_anchors": len(P_new),
    }


def upsert_view(comparison_views_path, name, reference_image_relpath,
                 pose_data, render_resolution=(1920, 1080), render_engine="CYCLES"):
    """Add or update a `purpose=reference-match` view in the scene's
    comparison_views.json. Preserves all other top-level fields."""
    if os.path.exists(comparison_views_path):
        with open(comparison_views_path) as f:
            cv = json.load(f)
    else:
        cv = {"schema_version": 4, "views": [], "anchor_frames": []}

    # Drop any existing view with this name.
    cv["views"] = [v for v in cv.get("views", []) if v["name"] != name]

    captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    view = {
        "name": name,
        "captured_camera_name": name,
        "captured_at": captured_at,
        "convention": "opencv",
        "purpose": "reference-match",
        "world_position": pose_data["world_position"],
        "world_rotation_quat_wxyz": pose_data["world_rotation_quat_wxyz"],
        "lens_mm": pose_data["lens_mm"],
        "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0,
        "matches_reference_images": [reference_image_relpath],
        "render_resolution": list(render_resolution),
        "render_engine": render_engine,
        "auto_localized": True,
        "localization_method": "mast3r_sfm_extend",
    }
    cv["views"].append(view)
    cv["views"].sort(key=lambda v: v["name"])
    cv["schema_version"] = 4

    with open(comparison_views_path, "w") as f:
        json.dump(cv, f, indent=2)
    print(f"Upserted view '{name}' into {comparison_views_path}")


def main():
    description = (__doc__ or "").split("\n\n", 1)[0]
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene-variant", required=True,
                   help="Absolute local path to the scene variant directory "
                        "(must contain mast3r_sfm/cameras.json + oriented/oriented_cameras.json)")
    p.add_argument("--reference-image", required=True,
                   help="Absolute local path to the reference PNG to localize")
    p.add_argument("--comparison-views-out", required=True,
                   help="Absolute local path to the scene's comparison_views.json (will be created if missing)")
    p.add_argument("--reference-name", default="cam_ref_auto",
                   help="Name to use for the new view (default: cam_ref_auto)")
    p.add_argument("--remote-host", default="tbeeprz")
    p.add_argument("--remote-data-root", default="/home/jeremy/outposts/krabby/data/011-scene-reconstruction")
    p.add_argument("--container", default="matcha-build")
    p.add_argument("--milestone-root", default="/private/var/krabby/workspace/milestones/011-scene-reconstruction",
                   help="Used to compute the relative path of the reference image for the JSON")
    p.add_argument("--keep-sandbox", action="store_true",
                   help="Don't clean up the local /tmp staging dir")
    args = p.parse_args()

    # Validate inputs
    scene_variant = os.path.abspath(args.scene_variant)
    orig_cams_path = os.path.join(scene_variant, "mast3r_sfm/cameras.json")
    orient_path = os.path.join(scene_variant, "oriented/oriented_cameras.json")
    if not os.path.exists(orig_cams_path):
        sys.exit(f"missing {orig_cams_path}")
    if not os.path.exists(orient_path):
        sys.exit(f"missing {orient_path}")

    # Derive scene-variant basename + relative path under milestone root for the
    # remote sandbox layout.
    variant_basename = os.path.basename(scene_variant.rstrip("/"))
    milestone_data = os.path.join(args.milestone_root, "data")
    if not scene_variant.startswith(milestone_data):
        sys.exit(f"--scene-variant {scene_variant} must live under {milestone_data}")
    scene_dir_relative = os.path.relpath(scene_variant, args.milestone_root + "/data")  # e.g., scenes/dtu-bicycle-curated-12-dense-strong
    scene_dir_relative = "scenes/" + scene_dir_relative.lstrip("./").replace("scenes/", "", 1) if not scene_dir_relative.startswith("scenes/") else scene_dir_relative
    # Above: ensure the path is canonical "scenes/<variant>".

    # Reference image relative path (for the JSON's matches_reference_images field).
    ref_relative = os.path.relpath(os.path.abspath(args.reference_image), args.milestone_root)

    # Local staging
    workdir = tempfile.mkdtemp(prefix="krabby_localize_")
    print(f"workdir: {workdir}")
    ref_jpg = os.path.join(workdir, "_DSC9999_ref.JPG")
    stage_reference_jpg(args.reference_image, ref_jpg)

    # Remote stage + run
    sandbox_container = remote_stage_sandbox(
        args.remote_host, args.remote_data_root, variant_basename,
        scene_dir_relative, ref_jpg,
    )
    output_container = sandbox_container.rsplit("/", 1)[0] + "/output"  # /data/sfm-ref-localize/<variant>/output
    cameras_json_container = remote_run_sfm(
        args.remote_host, args.container, sandbox_container, output_container, n_images=13,
    )

    # Pull cameras.json back
    new_cams_local = os.path.join(workdir, "cameras_with_ref.json")
    fetch_cameras_json(args.remote_host, cameras_json_container, args.remote_data_root, new_cams_local)

    # Align + extract reference pose in world frame
    pose = align_extract_world_pose(new_cams_local, orig_cams_path, orient_path, ref_index=12)
    print(f"\nReference pose (world frame, OpenCV):")
    print(f"  position: {pose['world_position']}")
    print(f"  quaternion: {pose['world_rotation_quat_wxyz']}")
    print(f"  lens: {pose['lens_mm']:.2f} mm  (focal_px={pose['focal_px']:.2f})")

    # Write to comparison_views.json
    upsert_view(args.comparison_views_out, args.reference_name, ref_relative, pose)

    if not args.keep_sandbox:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nDone. cam_ref_auto landed in {args.comparison_views_out}.")
    print("Next: regenerate the scene's .blend via build_blender_scene.py — the new "
          "view will auto-inject alongside any existing comparison cameras.")


if __name__ == "__main__":
    main()
