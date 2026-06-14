#!/usr/bin/env python3
"""STO-SCN-102 — per-camera fisheye calibration -> capture profile.

One-time, per-camera-MODEL fisheye calibration. Shoot a planar checkerboard with
the camera in its capture mode, run this, and it writes the intrinsics
(`K`, distortion `D`, image size, RMS reproj error) into the matching
`capture_profiles.json` entry under a `calibration` key. STO-SCN-093's undistort
step reads it; cameras without a calibration fall back to approximate-FOV
intrinsics.

Calibration is a property of the camera model, not the scene (STO-SCN-091/096 #3)
— this is the authoritative form of that. OpenCV fisheye model (Kannala-Brandt,
4 distortion params) is the standard for wide lenses.

CAPTURE RECIPE (do this once per camera+mode):
  - Print a checkerboard (e.g. 9x6 INNER corners; any size — measure the square
    edge in metres). Mount it flat/rigid.
  - Shoot ~20-40 stills with the camera in the SAME mode you reconstruct in
    (fisheye, locked focus). Vary angle + distance; CRUCIALLY push the board
    into the IMAGE CORNERS (that's where 155-degree distortion lives) and tilt it.
  - Keep the board fully visible + sharp in each frame.

Usage:
  calibrate_camera.py --images <dir> --make DJI --model "DJI Action 3" \
      --mode fisheye --board 9x6 --square 0.025 [--profiles <path>] [--dry-run]

cv2 is imported lazily so the profile-write path is testable without OpenCV.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

DEFAULT_PROFILES = Path(__file__).with_name("capture_profiles.json")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def parse_board(spec: str) -> tuple[int, int]:
    """'9x6' -> (cols, rows) inner-corner counts."""
    cols, rows = spec.lower().split("x")
    return int(cols), int(rows)


def detect_corners(gray, board: tuple[int, int]):
    """Find + refine chessboard inner corners; returns (N,1,2) float32 or None."""
    import cv2
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, board, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)


def object_grid(board: tuple[int, int], square: float):
    """Planar board points (cols*rows, 1, 3) float32, Z=0, spacing=square (m)."""
    import numpy as np
    cols, rows = board
    objp = np.zeros((cols * rows, 1, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, 0, :2] = grid * float(square)
    return objp


def calibrate_fisheye(objpoints, imgpoints, image_size):
    """cv2.fisheye.calibrate -> (rms, K(3x3 list), D(4 list))."""
    import cv2
    import numpy as np
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    n = len(objpoints)
    rvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    tvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_FIX_SKEW)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_size, K, D, rvecs, tvecs, flags, crit)
    return float(rms), K.tolist(), D.flatten().tolist()


def write_calibration(profiles_path, make, model, mode, calibration: dict) -> dict:
    """Set `calibration` on the matching {make,model,mode} capture profile entry.
    Pure JSON (no cv2). Fails loud if no matching profile exists."""
    p = Path(profiles_path)
    data = json.loads(p.read_text())
    for prof in data.get("profiles", []):
        if (_norm(prof.get("make")) == _norm(make)
                and _norm(prof.get("model")) == _norm(model)
                and _norm(prof.get("mode")) == _norm(mode)):
            prof["calibration"] = calibration
            p.write_text(json.dumps(data, indent=2) + "\n")
            return prof
    raise ValueError(
        f"no capture profile for make={make!r} model={model!r} mode={mode!r} in "
        f"{p.name} — add the profile (STO-SCN-091) before calibrating it.")


def run_calibration(images_dir, make, model, mode, board, square,
                    profiles_path=DEFAULT_PROFILES, dry_run=False) -> dict:
    import cv2
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no images in {images_dir}")
    objp = object_grid(board, square)
    objpoints, imgpoints, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]            # (w, h)
        corners = detect_corners(gray, board)
        if corners is None:
            print(f"  [skip] no board: {p.name}")
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used.append(p.name)
        print(f"  [ok]   {p.name}")
    if len(objpoints) < 5:
        raise SystemExit(f"only {len(objpoints)} usable views (need >=5 — re-shoot "
                         f"with the board covering more angles/corners)")
    print(f"calibrating fisheye from {len(objpoints)}/{len(paths)} views @ {size} ...")
    rms, K, D = calibrate_fisheye(objpoints, imgpoints, size)
    calibration = {
        "model": "OPENCV_FISHEYE",
        "image_size": list(size),
        "K": K, "D": D,
        "rms_reproj_px": round(rms, 4),
        "n_images_used": len(objpoints),
        "n_images_total": len(paths),
        "board": f"{board[0]}x{board[1]}", "square_m": float(square),
        "calibrated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "calibrate_camera.py",
    }
    print(f"RMS reproj error: {rms:.4f} px  (good: <1.0; investigate if >1.5)")
    print(f"K = {K}")
    print(f"D = {D}")
    if dry_run:
        print("[dry-run] not writing to capture profile.")
    else:
        write_calibration(profiles_path, make, model, mode, calibration)
        print(f"wrote calibration -> {Path(profiles_path).name} ({make}/{model}/{mode})")
    return calibration


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-camera fisheye calibration -> capture profile.")
    ap.add_argument("--images", required=True, help="dir of checkerboard stills")
    ap.add_argument("--make", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="fisheye")
    ap.add_argument("--board", default="9x6", help="INNER corners cols x rows (e.g. 9x6)")
    ap.add_argument("--square", type=float, required=True, help="square edge length in metres")
    ap.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    ap.add_argument("--dry-run", action="store_true", help="calibrate but don't write the profile")
    a = ap.parse_args(argv)
    run_calibration(a.images, a.make, a.model, a.mode, parse_board(a.board),
                    a.square, a.profiles, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
