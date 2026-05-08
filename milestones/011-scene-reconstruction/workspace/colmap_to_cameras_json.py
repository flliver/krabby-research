"""Convert a Mip-NeRF 360 / COLMAP `sparse/0/{cameras,images}.bin` directory
into our MASt3R-SfM-shaped cameras.json so the camera_viewer (and downstream
build_blender_scene.py) can consume it without modification.

Output schema matches what MAtCha writes for our captured scenes:
    {
        "filepaths": ["<image-dir>/IMG_0000.JPG", ...],
        "focals":    [<float>, ...],     # one per image, in pixels
        "cams2world": [<4x4 float lists>, ...]
    }

The COLMAP binary format ships extrinsics as world→camera with a rotation
quaternion and translation. We invert that pair to get cam→world, which is
what cams2world denotes in MASt3R land.

Image naming order follows COLMAP's image_id; we sort by image filename so
the output is stable across re-runs.

Usage:
    python3 colmap_to_cameras_json.py \\
        --sparse <path>/sparse/0 \\
        --images-dir <path>/images_4 \\
        --output    <path>/cameras.json
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# ---- COLMAP binary readers (extracted from Inria's 2d-gaussian-splatting repo)

def _read_next_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_intrinsics_binary(path):
    """Returns dict: camera_id → {model, width, height, params}."""
    cameras = {}
    with open(path, "rb") as f:
        n_cams = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n_cams):
            cam_id, model_id, width, height = _read_next_bytes(f, 24, "iiQQ")
            # Number of params depends on camera model; CAMERA_MODELS table
            # in colmap_loader.py. Mip-NeRF 360 uses model 4 (OPENCV) with 8
            # params, or model 1 (PINHOLE) with 4. Read the rest as floats.
            n_params_by_model = {
                0: 3,   # SIMPLE_PINHOLE: f, cx, cy
                1: 4,   # PINHOLE: fx, fy, cx, cy
                2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
                3: 5,   # RADIAL: f, cx, cy, k1, k2
                4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                5: 8,   # OPENCV_FISHEYE
                6: 12,  # FULL_OPENCV
            }
            n_params = n_params_by_model.get(model_id, 4)
            params = _read_next_bytes(f, 8 * n_params, "d" * n_params)
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_extrinsics_binary(path):
    """Returns list of dicts: {image_id, qvec, tvec, camera_id, name}."""
    out = []
    with open(path, "rb") as f:
        n_images = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read_next_bytes(
                f, 64, "idddddddi"
            )
            # Read null-terminated image name
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")
            # Skip the 2D-3D point correspondence block (we don't need it)
            n_points2d = _read_next_bytes(f, 8, "Q")[0]
            f.read(24 * n_points2d)  # x, y, point3d_id per point
            out.append({
                "image_id": image_id,
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
                "name": name,
            })
    return out


def qvec_to_rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = qvec
    # Normalize defensively
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---- Main ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sparse", required=True,
                   help="COLMAP sparse/0 directory containing cameras.bin + images.bin")
    p.add_argument("--images-dir", required=True,
                   help="Directory containing the actual image files (e.g. images_4/)")
    p.add_argument("--output", required=True, help="Output cameras.json path")
    p.add_argument("--use-fy-only", action="store_true",
                   help="When intrinsics have separate fx/fy, use fy as the focal "
                        "(default: average of the two)")
    p.add_argument("--no-auto-orient", action="store_true",
                   help="Skip the gravity-prior z-up auto-orient step. By default "
                        "we average the per-camera up-axes (= -cams2world[:,:,1] in "
                        "OpenCV convention) and rotate the world so that becomes +Z. "
                        "Without this, COLMAP datasets often display upside-down or "
                        "tilted in viewers that assume z-up.")
    args = p.parse_args()

    sparse = Path(args.sparse)
    images_dir = Path(args.images_dir).resolve()
    out_path = Path(args.output)

    cams = read_intrinsics_binary(sparse / "cameras.bin")
    print(f"  read {len(cams)} unique camera intrinsic(s)")
    imgs = read_extrinsics_binary(sparse / "images.bin")
    print(f"  read {len(imgs)} image extrinsics")

    # Sort by image name for stable output
    imgs.sort(key=lambda x: x["name"])

    # Detect resolution-rescale: the COLMAP binaries usually reference the
    # full-res camera. If the user is using images_4 / images_8, the focal in
    # px must be scaled accordingly. Detect by comparing the on-disk image
    # dimensions to what's recorded in cameras.bin.
    sample_image = images_dir / imgs[0]["name"]
    if not sample_image.exists():
        # Some datasets store images_4/00001.JPG; the COLMAP record names
        # often include subdirectories. Try the basename only.
        sample_image = images_dir / Path(imgs[0]["name"]).name
    if not sample_image.exists():
        raise SystemExit(
            f"ERROR: image referenced by COLMAP not found in {images_dir}: "
            f"{imgs[0]['name']}"
        )
    # Use PIL if available, else parse JPEG header (we have PIL via viser deps)
    try:
        from PIL import Image
        with Image.open(sample_image) as im:
            actual_w, actual_h = im.size
    except ImportError:
        raise SystemExit("PIL/Pillow required; install with `pip install pillow`.")

    # First camera in cams (assumed shared across the dataset; warn if not)
    cam_widths = {c["width"] for c in cams.values()}
    cam_heights = {c["height"] for c in cams.values()}
    if len(cam_widths) > 1 or len(cam_heights) > 1:
        print(f"  WARNING: cameras.bin lists multiple resolutions "
              f"({sorted(cam_widths)} × {sorted(cam_heights)})")
    recorded_w = next(iter(cam_widths))
    recorded_h = next(iter(cam_heights))
    scale = actual_w / recorded_w
    if abs(scale - actual_h / recorded_h) > 0.01:
        print(f"  WARNING: non-uniform rescale (W:{actual_w}/{recorded_w}={scale:.3f}, "
              f"H:{actual_h}/{recorded_h}={actual_h/recorded_h:.3f})")
    print(f"  recorded res: {recorded_w}×{recorded_h}, on-disk: {actual_w}×{actual_h}, "
          f"focal scale: {scale:.4f}")

    filepaths = []
    focals = []
    cams2world = []
    for img in imgs:
        cam = cams[img["camera_id"]]
        # Extract focal in pixels (model-dependent index)
        params = cam["params"]
        model_id = cam["model_id"]
        if model_id == 0:        # SIMPLE_PINHOLE: f, cx, cy
            fx_full = fy_full = params[0]
        elif model_id in (1, 4): # PINHOLE / OPENCV: fx, fy, cx, cy, ...
            fx_full = params[0]
            fy_full = params[1]
        elif model_id in (2, 3): # SIMPLE_RADIAL / RADIAL: f, cx, cy, k(s)
            fx_full = fy_full = params[0]
        else:
            raise SystemExit(f"Unhandled COLMAP camera model id={model_id}")
        # Apply resolution rescale to match the on-disk images
        fx = fx_full * scale
        fy = fy_full * scale
        focal = fy if args.use_fy_only else 0.5 * (fx + fy)

        # Build cams2world: COLMAP stores world→camera (R, t). Invert to
        # cam→world: R_cw = R_wc.T, t_cw = -R_wc.T @ t_wc
        R_wc = qvec_to_rotmat(img["qvec"])
        t_wc = np.array(img["tvec"], dtype=np.float64)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_cw
        c2w[:3, 3] = t_cw

        # Resolve the actual on-disk image path
        image_path = images_dir / Path(img["name"]).name
        filepaths.append(str(image_path))
        focals.append(float(focal))
        cams2world.append(c2w.tolist())

    # ---- Optional auto-orient via camera-up averaging --------------------
    if not args.no_auto_orient:
        cams2world = _auto_orient_to_z_up(cams2world)

    payload = {
        "filepaths": filepaths,
        "focals": focals,
        "cams2world": cams2world,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    f_min, f_max = min(focals), max(focals)
    print(f"\nWrote {out_path}")
    print(f"  filepaths: {len(filepaths)}")
    print(f"  focals: min={f_min:.1f} max={f_max:.1f} px (at {actual_w}×{actual_h})")


def _auto_orient_to_z_up(cams2world: list) -> list:
    """Rotate the world frame so the average camera-up axis becomes +Z.

    COLMAP / MASt3R-SfM converge to whichever world frame their SfM happens
    to land in. Most viewers (viser, Blender's default, etc.) assume +Z is
    up. When the SfM-emergent frame disagrees, scenes display upside-down
    or tilted. This function aligns them by averaging per-camera up-axes
    and rotating the world so that average becomes +Z.

    Convention: each camera in OpenCV form has +Y pointing DOWN in its
    image. The world-direction of camera-up is therefore `-cams2world[:,:,1]`
    (the negated 2nd column of the rotation block). Averaging across cams
    and normalizing yields the gravity prior; we rotate the world so this
    direction snaps to (0, 0, 1).

    This is the same lever as `orient_mesh.py:estimate_gravity_from_cameras`,
    applied at conversion time so no mesh is required. Confidence (length
    of the unnormalized average) is reported as a sanity check.
    """
    arr = np.asarray(cams2world)
    if arr.size == 0:
        return cams2world
    ups = -arr[:, :3, 1]                       # (N, 3) world-direction of camera-up
    avg = ups.mean(axis=0)
    confidence = float(np.linalg.norm(avg))    # 1.0 = perfectly clustered
    if confidence < 0.5:
        print(f"  WARN: camera-up vectors don't cluster strongly "
              f"(confidence={confidence:.3f}); skipping auto-orient")
        return cams2world
    up = avg / confidence

    # Construct a rotation R that maps `up` → (0, 0, 1) using Rodrigues' formula
    z = np.array([0.0, 0.0, 1.0])
    cos = float(np.dot(up, z))
    if cos > 0.9999:
        # Already z-up
        print(f"  auto-orient: already z-up (confidence={confidence:.3f})")
        return cams2world
    if cos < -0.9999:
        # 180° flip — pick any axis perpendicular to up; X works
        R_world = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
    else:
        axis = np.cross(up, z)
        axis = axis / np.linalg.norm(axis)
        sin = float(np.linalg.norm(np.cross(up, z)))
        K = np.array([
            [    0.0, -axis[2],  axis[1]],
            [ axis[2],     0.0, -axis[0]],
            [-axis[1],  axis[0],     0.0],
        ])
        R_world = np.eye(3) + sin * K + (1.0 - cos) * (K @ K)

    angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    print(f"  auto-orient: rotating world by {angle_deg:.1f}° "
          f"(confidence={confidence:.3f}, axis-of-rotation aligned)")

    # Apply R_world @ c2w to every cams2world matrix (rotates positions and
    # camera basis vectors simultaneously; translation rotates because the
    # world's origin doesn't move but its axes do).
    R4 = np.eye(4)
    R4[:3, :3] = R_world
    rotated = []
    for c2w in cams2world:
        new_c2w = R4 @ np.asarray(c2w)
        rotated.append(new_c2w.tolist())

    # Sanity check
    new_arr = np.asarray(rotated)
    new_avg_up = -new_arr[:, :3, 1].mean(axis=0)
    new_avg_up = new_avg_up / np.linalg.norm(new_avg_up)
    print(f"  post-rotate avg up: ({new_avg_up[0]:+.3f}, "
          f"{new_avg_up[1]:+.3f}, {new_avg_up[2]:+.3f})  ← should be ≈ (0, 0, 1)")
    return rotated


if __name__ == "__main__":
    main()
