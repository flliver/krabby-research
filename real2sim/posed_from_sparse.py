#!/usr/bin/env python3
"""STO-SCN-095 — COLMAP sparse/0 -> posed.json (for da3@1 pose-conditioned scout).

The scout gaussian must live in the SAME gauge as STO-SCN-094's proposed-N (the
FastMap solve), so the frustums overlay correctly. `da3_infer_posed.py` (da3@1)
puts DA3 in a supplied pose gauge, reading a `cameras/posed.json` =
`[{name, w2c(4x4), K(3x3)}]` (colmap_posed.solve_to_posed_json shape). The FastMap
solve gives a COLMAP `sparse/0`; this shim converts it to that posed.json so the
scout renders in the solve frame.

Pure stdlib (struct/math) — testable anywhere. Camera model: PINHOLE/
SIMPLE_PINHOLE (the undistorted scout images), so K is exact.

Usage:
  posed_from_sparse.py <sparse_dir> --out posed.json [--names a.jpg b.jpg ...]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from covis_graph import qvec2rotmat, _MODEL_NPARAMS  # noqa: E402


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_K(path) -> dict:
    """camera_id -> 3x3 K. Supports SIMPLE_PINHOLE(0)=[f,cx,cy], PINHOLE(1)=[fx,fy,cx,cy],
    SIMPLE_RADIAL(2)=[f,cx,cy,k]."""
    out = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            npar = _MODEL_NPARAMS.get(model_id, 4)
            p = _read(f, "d" * npar)
            if model_id in (1,):                      # PINHOLE
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            else:                                     # SIMPLE_PINHOLE / SIMPLE_RADIAL
                fx = fy = p[0]; cx, cy = p[1], p[2]
            out[cam_id] = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return out


def read_images_w2c(path) -> list:
    """-> list of {name, camera_id, w2c(4x4)} (qvec/tvec are the w2c pose)."""
    out = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "idddddddi")
            qw, qx, qy, qz = vals[1:5]
            tx, ty, tz = vals[5:8]
            cam_id = vals[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = _read(f, "Q")
            f.read(24 * n2d)
            R = qvec2rotmat(qw, qx, qy, qz)
            w2c = [[R[0][0], R[0][1], R[0][2], tx],
                   [R[1][0], R[1][1], R[1][2], ty],
                   [R[2][0], R[2][1], R[2][2], tz],
                   [0.0, 0.0, 0.0, 1.0]]
            out.append({"name": name.decode("utf-8", "replace"),
                        "camera_id": cam_id, "w2c": w2c})
    return out


def posed_from_sparse(sparse_dir, names=None) -> list:
    sp = Path(sparse_dir)
    Ks = read_cameras_K(sp / "cameras.bin")
    imgs = read_images_w2c(sp / "images.bin")
    want = set(names) if names else None
    out = []
    for im in imgs:
        if want is not None and im["name"] not in want:
            continue
        out.append({"name": im["name"], "w2c": im["w2c"],
                    "K": Ks.get(im["camera_id"], Ks.get(next(iter(Ks))))})
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COLMAP sparse/0 -> posed.json (da3@1 scout).")
    ap.add_argument("sparse_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="subset of image names (default: all)")
    a = ap.parse_args(argv)
    posed = posed_from_sparse(a.sparse_dir, a.names or None)  # empty -> no filter
    Path(a.out).write_text(json.dumps(posed, indent=2) + "\n")
    print(f"wrote {len(posed)} posed cameras -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
