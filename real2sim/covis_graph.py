#!/usr/bin/env python3
"""STO-SCN-093 — co-visibility graph from a COLMAP/FastMap sparse model.

The selection stage (STO-SCN-094) needs, for the posed pool: which images see
which 3D points, how much each image PAIR overlaps, and the triangulation angle
of that overlap. This reads a COLMAP-format `sparse/0` (cameras/images/points3D
`.bin`) **directly via struct** — FastMap's output hangs `model_analyzer` and
errors `pycolmap` (STO-SCN-093 finding), but the raw binary is clean — and emits:

  - per-image coverage (number of 3D points observed)
  - per-pair shared-point count + mean triangulation angle (deg)
  - connectivity (union-find at a min-overlap threshold): components, isolated

Pure stdlib (struct + math) — no numpy/pycolmap/cv2 — so it runs anywhere and
is fully unit-testable.

Usage:
  covis_graph.py <sparse_dir> [--out covis.json] [--min-overlap 15]
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# COLMAP camera model id -> number of params (enough of the table for our use).
_MODEL_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path) -> dict:
    out = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            npar = _MODEL_NPARAMS.get(model_id, 4)
            params = _read(f, "d" * npar)
            out[cam_id] = {"model_id": model_id, "width": w, "height": h, "params": list(params)}
    return out


def qvec2rotmat(qw, qx, qy, qz):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _cam_center(qvec, tvec):
    """C = -R^T t (world coords of the camera)."""
    R = qvec2rotmat(*qvec)
    return [-(R[0][i] * tvec[0] + R[1][i] * tvec[1] + R[2][i] * tvec[2]) for i in range(3)]


def read_images_bin(path) -> dict:
    """image_id -> {name, center, fwd}. fwd = camera optical axis in world
    (R_w2c row 2). Skips the 2D point list (not needed for covis)."""
    out = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "idddddddi")
            img_id = vals[0]
            qvec = vals[1:5]
            tvec = vals[5:8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = _read(f, "Q")
            f.read(24 * n2d)  # x(d) y(d) point3D_id(q) per 2D point — skip
            R = qvec2rotmat(*qvec)
            center = [-(R[0][i] * tvec[0] + R[1][i] * tvec[1] + R[2][i] * tvec[2])
                      for i in range(3)]
            out[img_id] = {"name": name.decode("utf-8", "replace"),
                           "center": center, "fwd": [R[2][0], R[2][1], R[2][2]]}
    return out


def read_points3D_bin(path) -> list:
    """list of {xyz, image_ids} (the track's image ids)."""
    out = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "QdddBBBd")
            xyz = list(vals[1:4])
            (tlen,) = _read(f, "Q")
            track = _read(f, "ii" * tlen) if tlen else ()
            image_ids = list(track[0::2])
            out.append({"xyz": xyz, "image_ids": image_ids})
    return out


def _angle_deg(Ca, Cb, X):
    """Triangulation angle at X between cameras Ca, Cb (degrees)."""
    ua = [Ca[i] - X[i] for i in range(3)]
    ub = [Cb[i] - X[i] for i in range(3)]
    na = math.sqrt(sum(v * v for v in ua))
    nb = math.sqrt(sum(v * v for v in ub))
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(ua[i] * ub[i] for i in range(3)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def build_covis(images: dict, points3D: list, min_overlap: int = 15) -> dict:
    """Co-visibility graph: per-image coverage, per-pair shared+mean angle,
    connectivity at `min_overlap` shared points."""
    centers = {i: img["center"] for i, img in images.items()}
    coverage = defaultdict(int)
    pair_shared = defaultdict(int)
    pair_angle = defaultdict(float)
    track_lengths = []
    for pt in points3D:
        ids = sorted({i for i in pt["image_ids"] if i in centers})
        track_lengths.append(len(ids))
        X = pt["xyz"]
        for i in ids:
            coverage[i] += 1
        for a, b in combinations(ids, 2):
            pair_shared[(a, b)] += 1
            pair_angle[(a, b)] += _angle_deg(centers[a], centers[b], X)

    pairs = [(a, b, c, round(pair_angle[(a, b)] / c, 2)) for (a, b), c in pair_shared.items()]

    # connectivity via union-find on pairs with shared >= min_overlap
    parent = {i: i for i in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c, _ in pairs:
        if c >= min_overlap:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comps = defaultdict(list)
    for i in images:
        comps[find(i)].append(i)
    comp_sizes = sorted((len(v) for v in comps.values()), reverse=True)
    isolated = [images[i]["name"] for i in images if coverage[i] == 0]
    n = len(images)
    return {
        "n_images": n,
        "n_points": len(points3D),
        "mean_track_length": round(sum(track_lengths) / max(1, len(track_lengths)), 2),
        "n_pairs": len(pairs),
        "min_overlap": min_overlap,
        "n_components": len(comp_sizes),
        "largest_component": comp_sizes[0] if comp_sizes else 0,
        "connected": len(comp_sizes) == 1,
        "n_isolated": len(isolated),
        "isolated_images": isolated[:20],
        "median_pair_overlap": sorted(c for _, _, c, _ in pairs)[len(pairs) // 2] if pairs else 0,
        "coverage": {images[i]["name"]: coverage[i] for i in images},
        "pairs": [[images[a]["name"], images[b]["name"], c, ang]
                  for a, b, c, ang in pairs if c >= 1],
    }


def covis_from_sparse(sparse_dir, min_overlap: int = 15) -> dict:
    sp = Path(sparse_dir)
    images = read_images_bin(sp / "images.bin")
    points3D = read_points3D_bin(sp / "points3D.bin")
    return build_covis(images, points3D, min_overlap)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Co-visibility graph from a COLMAP sparse model.")
    ap.add_argument("sparse_dir", help="path to sparse/0 (cameras/images/points3D.bin)")
    ap.add_argument("--out", default=None, help="write covis JSON here (default <sparse_dir>/covis.json)")
    ap.add_argument("--min-overlap", type=int, default=15,
                    help="shared-point threshold for the connectivity graph")
    a = ap.parse_args(argv)
    g = covis_from_sparse(a.sparse_dir, a.min_overlap)
    out = Path(a.out) if a.out else Path(a.sparse_dir) / "covis.json"
    out.write_text(json.dumps(g, indent=2) + "\n")
    print(f"covis: {g['n_images']} imgs, {g['n_points']} pts, "
          f"mean track {g['mean_track_length']}, {g['n_pairs']} pairs")
    print(f"connectivity @ min_overlap={g['min_overlap']}: "
          f"{'CONNECTED' if g['connected'] else str(g['n_components'])+' components'} "
          f"(largest {g['largest_component']}/{g['n_images']}), isolated={g['n_isolated']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
