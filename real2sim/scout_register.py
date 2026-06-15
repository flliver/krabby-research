#!/usr/bin/env python3
"""scout_register.py — register the DA3 scout gaussian to the FastMap solve gauge
by DIRECT point-cloud alignment (STO-SCN-105, corrected).

The earlier `scale_factor` theory was disproven on a real tbeeprz scout
(001-patio): DA3's `scale_factor` correctly maps DA3's *colmap* points to the
solve, but the **gs_ply splat we actually display is already ~solve-scale and
merely translated** (~2.9 in z on 001). So we don't trust any DA3-internal
normalization for the splat — we register the gs_ply directly against the
solve's own sparse points3D, both of which live in the store and describe the
same physical scene.

Posed-mode scouts share orientation with the solve (DA3 was handed the solve
cameras), so the registration is a **similarity without rotation**: a uniform
scale + a translation, p_solve = scale * p_gs + t. Robust to the DA3 far halo
via iterative core trimming; the scale is refined by maximizing a top-down
occupancy overlap (the metric that disproved the scale_factor approach).

Pure-stdlib (no numpy) so it runs in the build_verify environment everywhere.
"""
from __future__ import annotations

import array
import json
import math
import re
import struct
from pathlib import Path


# --------------------------------------------------------------- readers

def read_points3d_bin(path) -> list:
    """COLMAP points3D.bin -> list of (x,y,z)."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                                  # point3D_id
            x, y, z = struct.unpack("<3d", f.read(24))
            f.read(3 + 8)                              # rgb + error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)                      # track (image_id,p2d_idx)*track_len
            pts.append((x, y, z))
    return pts


def read_gs_ply_xyz(path, max_samples=60000) -> list:
    """3DGS .ply -> list of finite (x,y,z), uniformly subsampled to ~max_samples."""
    head = b""
    with open(path, "rb") as f:
        while b"end_header\n" not in head:
            chunk = f.read(256)
            if not chunk:
                break
            head += chunk
        off = head.index(b"end_header\n") + len(b"end_header\n")
        n = int(re.search(rb"element vertex (\d+)", head).group(1))
        props = head.count(b"property float")
        f.seek(off)
        buf = array.array("f")
        buf.frombytes(f.read(n * props * 4))
    step = max(1, n // max_samples)
    out = []
    for i in range(0, n, step):
        b = i * props
        x, y, z = buf[b], buf[b + 1], buf[b + 2]
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            out.append((x, y, z))
    return out


# --------------------------------------------------------------- robust core

def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def robust_center_scale(pts, keep=0.8, iters=3):
    """Median center + RMS radius of the core (iteratively trim to the inner
    `keep` fraction so the DA3 far halo / SfM outliers don't dominate)."""
    c = [_median([p[i] for p in pts]) for i in range(3)]
    core = pts
    for _ in range(iters):
        d = [math.dist(p, c) for p in core]
        thr = sorted(d)[min(len(d) - 1, int(len(d) * keep))]
        core = [p for p in core if math.dist(p, c) <= thr] or core
        c = [_median([p[i] for p in core]) for i in range(3)]
    rms = math.sqrt(sum(math.dist(p, c) ** 2 for p in core) / len(core))
    return c, rms, core


# --------------------------------------------------------------- overlap metric

def _topdown_iou(solve_core, gs_pts, cs, cg, scale, cell):
    """Top-down (XZ) occupancy IoU of gs→solve under (scale about cg, shift to cs)."""
    R2 = (cell * 30) ** 2
    occ_s = set((round((p[0] - cs[0]) / cell), round((p[2] - cs[2]) / cell))
                for p in solve_core if (p[0] - cs[0]) ** 2 + (p[2] - cs[2]) ** 2 < R2)
    occ_g = set()
    for p in gs_pts:
        x = (p[0] - cg[0]) * scale
        z = (p[2] - cg[2]) * scale
        if x * x + z * z < R2:
            occ_g.add((round(x / cell), round(z / cell)))
    inter = len(occ_s & occ_g)
    uni = len(occ_s | occ_g)
    return inter / uni if uni else 0.0


# --------------------------------------------------------------- registration

def register(gs_ply, solve_points3d, max_samples=60000) -> dict:
    """Similarity (scale + translation, no rotation) mapping the gs_ply into the
    solve gauge: p_solve = scale * p_gs + translate.

    scale seeded from the robust core-RMS ratio, then refined by maximizing the
    top-down overlap; translation re-derived at the chosen scale so the cores'
    centers coincide. Returns scale, translate[3], the achieved IoU, and counts.
    """
    G = read_gs_ply_xyz(gs_ply, max_samples)
    S = read_points3d_bin(solve_points3d)
    cg, rg, _ = robust_center_scale(G)
    cs, rs, S_core = robust_center_scale(S)
    seed = rs / rg if rg else 1.0
    cell = rs / 15.0 or 1.0

    # SCALE = robust core-RMS ratio. (We deliberately do NOT optimize scale by
    # the occupancy IoU — that grid metric is biased for scale: a smaller-scaled
    # cloud packs into fewer, denser cells and can spuriously win. Verified on
    # synthetic data where the IoU search drifted to 0.23 while the true scale
    # was 0.40 — the core-RMS ratio recovered 0.40 exactly. Core trimming has
    # already removed the DA3 halo, so the RMS ratio is the trustworthy scale.)
    scale = round(seed, 5)

    # TRANSLATION: center-match, then a small IoU refine — translation does NOT
    # change occupancy density, so the overlap metric is unbiased for it. A few
    # cells per axis recover any residual offset the core trim left.
    off = [0.0, 0.0, 0.0]
    base_iou = _topdown_iou(S_core, G, cs, cg, scale, cell)
    for ax in (0, 1, 2):
        cur = base_iou
        for d in range(-5, 6):
            cand = list(off)
            cand[ax] = d * cell
            cs2 = [cs[i] - cand[i] for i in range(3)]   # shifting solve == shifting gs the other way
            j = _topdown_iou(S_core, G, cs2, cg, scale, cell)
            if j > cur:
                cur, off[ax] = j, cand[ax]
        base_iou = cur
    # p_solve = scale*(p_gs - cg) + cs + off = scale*p_gs + (cs + off - scale*cg)
    translate = [round(cs[i] + off[i] - scale * cg[i], 5) for i in range(3)]
    best = (scale, round(base_iou, 4))
    return {
        "scale": scale,
        "translate": translate,
        "rotation": "identity (posed-mode: gs shares solve orientation)",
        "iou": round(best[1], 4),
        "seed_scale": round(seed, 4),
        "gs_center": [round(x, 4) for x in cg],
        "solve_center": [round(x, 4) for x in cs],
        "n_gs": len(G), "n_solve": len(S),
        "method": "direct gs_ply↔solve points3D core alignment (scale+translation)",
    }


def _read_manual(scout_dir) -> dict | None:
    """Operator photo-match override, if present, keyed by scout id in the repo
    (verify_viewer/gauges/<scout>.json). Wins over the automatic transform."""
    p = Path(__file__).resolve().parent / "verify_viewer" / "gauges" / (Path(scout_dir).name + ".json")
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    return {"scale": d["scale"], "quat": d.get("quat", [0, 0, 0, 1]),
            "translate": d.get("translate", [0, 0, 0]),
            "registered": True, "source": "photo-match (manual override)"}


def gauge_for(scout_dir) -> dict:
    """The gauge that maps the scout gs_ply into the solve world frame:
    p_world = scale · R(quat[xyzw]) · p_gs + translate. Precedence:
    (1) operator photo-match override (repo sidecar), (2) the automatic
    DA3-predicted-pose Umeyama written by da3_infer_posed into scout_gauge.json
    (STO-SCN-105), (3) unregistered identity (caller warns)."""
    man = _read_manual(scout_dir)
    if man:
        return man
    sg = Path(scout_dir) / "scout_gauge.json"
    if sg.exists():
        try:
            tf = (json.loads(sg.read_text()) or {}).get("transform")
        except (ValueError, OSError):
            tf = None
        if tf and tf.get("scale"):
            return {"scale": float(tf["scale"]), "quat": tf.get("quat", [0, 0, 0, 1]),
                    "translate": tf.get("translate", [0, 0, 0]),
                    "registered": True, "source": tf.get("source", "da3")}
    return {"scale": 1.0, "quat": [0, 0, 0, 1], "translate": [0, 0, 0],
            "registered": False, "source": "unregistered"}


def register_scout(scout_dir, sparse_dir) -> dict:
    """Convenience: register a scout dir's scout.gs.ply against a solve's
    sparse/0/points3D.bin. Returns the transform, or an `unregistered` stub
    when inputs are missing (caller warns instead of mis-overlaying)."""
    gs = Path(scout_dir) / "scout.gs.ply"
    p3 = Path(sparse_dir) / "points3D.bin"
    if not gs.exists() or not p3.exists():
        return {"scale": 1.0, "translate": [0.0, 0.0, 0.0],
                "registered": False, "source": "unregistered"}
    r = register(gs, p3)
    r["registered"] = True
    r["source"] = "direct"
    return r


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(register(sys.argv[1], sys.argv[2]), indent=2))
