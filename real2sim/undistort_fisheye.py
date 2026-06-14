#!/usr/bin/env python3
"""STO-SCN-093 — undistort fisheye frames -> pinhole (for FastMap).

FastMap accepts only SIMPLE_PINHOLE/SIMPLE_RADIAL (STO-SCN-101 finding), so 155-deg
DJI fisheye must be undistorted to pinhole first. This uses the per-camera
calibration from STO-SCN-102 (OpenCV fisheye K + D, resolved out of the capture
profile) to remap each frame to a pinhole projection, central-cropped so there are
no black borders (which also drops the wide-FOV background pollution our
CAPTURE-LESSONS flagged). Emits the undistorted images + an `intrinsics.json`
(the new pinhole K') the FastMap pipeline consumes.

`balance` (0..1) trades FOV vs crop: 0 = tightest crop / no black border / narrowest
FOV (default, best for SfM); 1 = keep all pixels / black corners / extreme edges.

cv2 is imported lazily so the calibration-resolve + intrinsics-write paths are
testable without OpenCV.

Usage:
  undistort_fisheye.py --images <dir> --out <dir> \
      --make DJI --model "DJI Action 3" --mode fisheye [--balance 0.0] [--profiles P]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import capture_profile as cap  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_fisheye_calibration(make, model, mode, profiles_path=None) -> dict:
    """Resolve the capture profile and return its `calibration` block.
    Fails loud if the profile has no calibration (run calibrate_camera.py first)."""
    reg = cap.load_registry(profiles_path) if profiles_path else None
    prof = cap.resolve(make, model, mode, registry=reg)
    calib = prof.get("calibration")
    if not calib:
        raise ValueError(
            f"no calibration on profile {make}/{model}/{mode} — run calibrate_camera.py "
            f"first (STO-SCN-102), or this camera+mode isn't calibrated.")
    if calib.get("model") != "OPENCV_FISHEYE":
        raise ValueError(f"calibration model is {calib.get('model')!r}; expected OPENCV_FISHEYE.")
    return calib


def pinhole_from_calibration(calib: dict, balance: float = 0.0, fov_scale: float = 1.0):
    """Build the new pinhole camera matrix K' + the undistort maps from a fisheye
    calibration. Returns (K, D, Kp, maps, size). cv2 required."""
    import cv2
    import numpy as np
    K = np.array(calib["K"], dtype=np.float64)
    D = np.array(calib["D"], dtype=np.float64).reshape(4, 1)
    w, h = calib["image_size"]
    size = (w, h)
    R = np.eye(3)
    Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, R, balance=balance, new_size=size, fov_scale=fov_scale)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, Kp, size, cv2.CV_16SC2)
    return K, D, Kp, (map1, map2), size


def intrinsics_dict(Kp, size, calib, balance) -> dict:
    w, h = size
    return {
        "model": "PINHOLE",
        "width": int(w), "height": int(h),
        "fx": float(Kp[0][0]), "fy": float(Kp[1][1]),
        "cx": float(Kp[0][2]), "cy": float(Kp[1][2]),
        "params": [float(Kp[0][0]), float(Kp[1][1]), float(Kp[0][2]), float(Kp[1][2])],
        "source": "undistort_fisheye.py (cv2.fisheye)",
        "balance": float(balance),
        "from_calibration_rms_px": calib.get("rms_reproj_px"),
    }


def undistort_dir(images_dir, out_dir, make, model, mode,
                  balance=0.0, profiles_path=None) -> dict:
    import cv2
    calib = load_fisheye_calibration(make, model, mode, profiles_path)
    _, _, Kp, (map1, map2), size = pinhole_from_calibration(calib, balance)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no images in {images_dir}")
    n = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out / p.name), und)
        n += 1
    intr = intrinsics_dict(Kp, size, calib, balance)
    (out / "intrinsics.json").write_text(json.dumps(intr, indent=2) + "\n")
    print(f"undistorted {n}/{len(paths)} frames -> {out}")
    print(f"pinhole K': fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}  (balance={balance})")
    print(f"wrote {out/'intrinsics.json'}")
    intr["_n_written"] = n
    return intr


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Undistort fisheye frames -> pinhole (STO-SCN-093).")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--make", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="fisheye")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="0=tight crop/no black border (default); 1=keep all pixels")
    ap.add_argument("--profiles", default=None)
    a = ap.parse_args(argv)
    undistort_dir(a.images, a.out, a.make, a.model, a.mode, a.balance, a.profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
