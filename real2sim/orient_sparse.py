#!/usr/bin/env python3
"""orient_sparse.py — orient-cameras via RANSAC floor fit on the SPARSE
cloud (STO-SCN-082, method `floor-ransac-sparse` of HUG-SCN-005 locked #2).

Verification mode compares the sparse-fit gauge against the mesh-era
ground truth (the migrated `bootstrap-mesh` transform):

    python3 real2sim/orient_sparse.py verify <scene>
    python3 real2sim/orient_sparse.py run <scene>     # writes a NEW
                                                      # orient identity

The dense-mesh floor fit (STO-SCN-004) is the validated reference; this
task exists so orientation can run at ingest time, BEFORE any mesh.
Pure-python PLY reader + RANSAC — no heavy deps (sparse clouds are
small: tens of MB).
"""
from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4

SETTINGS = {"method": "floor-ransac-sparse", "ransac_dist": 0.05}
ALGO = "orient-floor@0"


def read_ply_xyz(path: Path):
    """Minimal binary/ascii PLY vertex reader (x,y,z floats)."""
    with path.open("rb") as f:
        header, fmt, n, props = [], None, 0, []
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property") and n and "list" not in line:
                props.append(line.split()[-1])
            elif line.startswith("element") and "vertex" not in line and n:
                pass
            if line == "end_header":
                break
        idx = [props.index(c) for c in ("x", "y", "z")]
        pts = []
        if fmt == "ascii":
            for _ in range(n):
                vals = f.readline().split()
                pts.append(tuple(float(vals[i]) for i in idx))
        else:
            little = "little_endian" in fmt
            esize = 4 * len(props)  # assume float32 properties
            raw = f.read(esize * n)
            fmt_s = ("<" if little else ">") + "f" * len(props)
            for i in range(n):
                vals = struct.unpack_from(fmt_s, raw, i * esize)
                pts.append(tuple(vals[j] for j in idx))
        return pts


def camera_up_prior(cameras_json: Path):
    """World-up estimate from the solve's camera poses: cameras are held
    roughly level, so up ≈ mean(-Y column of c2w rotations) (OpenCV:
    camera +Y points down). Returns a unit vector or None."""
    try:
        doc = json.loads(cameras_json.read_text())
        c2w = doc.get("cams2world") or doc.get("cams2world_list")
        if not c2w:
            return None
        acc = [0.0, 0.0, 0.0]
        for m in c2w:
            for i in range(3):
                acc[i] -= m[i][1]          # -Y column
        norm = sum(x * x for x in acc) ** 0.5
        return tuple(x / norm for x in acc) if norm > 1e-9 else None
    except Exception:
        return None


def ransac_floor(pts, dist=0.05, iters=600, seed=4, up_prior=None, max_tilt_deg=45.0):
    """Largest plane; when up_prior is given, only planes whose normal is
    within max_tilt of it qualify (kills walls — measured failure on 006:
    unconstrained fit locked onto the tractor side, 72° off)."""
    import math
    rng = random.Random(seed)
    best, best_n = None, -1
    m = len(pts)
    cos_max = math.cos(math.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = (pts[rng.randrange(m)] for _ in range(3))
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        norm = sum(x * x for x in n) ** 0.5
        if norm < 1e-9:
            continue
        n = tuple(x / norm for x in n)
        if up_prior is not None:
            dot = sum(n[i] * up_prior[i] for i in range(3))
            if dot < 0:
                n, dot = tuple(-x for x in n), -dot
            if dot < cos_max:
                continue                    # wall-ish: reject
        d = -sum(n[i] * a[i] for i in range(3))
        step = max(1, m // 4000)
        inl = sum(1 for p in pts[::step]
                  if abs(sum(n[i] * p[i] for i in range(3)) + d) < dist)
        if inl > best_n:
            best_n, best = inl, (n, d)
    return best, best_n


def rot_to_z_up(n):
    """Rotation matrix taking plane normal -> +Z (Rodrigues)."""
    import math
    z = (0.0, 0.0, 1.0)
    dot = max(-1.0, min(1.0, sum(n[i] * z[i] for i in range(3))))
    if dot < 0:                      # flip normal to point "up"
        n = tuple(-x for x in n)
        dot = -dot
    axis = (n[1] * z[2] - n[2] * z[1], n[2] * z[0] - n[0] * z[2], n[0] * z[1] - n[1] * z[0])
    s = sum(x * x for x in axis) ** 0.5
    if s < 1e-9:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis = tuple(x / s for x in axis)
    ang = math.atan2(s, dot)
    c, si = math.cos(ang), math.sin(ang)
    x, y, zx = axis
    return [
        [c + x * x * (1 - c), x * y * (1 - c) - zx * si, x * zx * (1 - c) + y * si],
        [y * x * (1 - c) + zx * si, c + y * y * (1 - c), y * zx * (1 - c) - x * si],
        [zx * x * (1 - c) - y * si, zx * y * (1 - c) + x * si, c + zx * zx * (1 - c)],
    ]


def fit(points_ply: Path, dist=0.05):
    pts = read_ply_xyz(points_ply)
    up = camera_up_prior(points_ply.parent / "cameras.json")
    (n, d), inliers = ransac_floor(pts, dist=dist, up_prior=up)
    R = rot_to_z_up(n)
    # z_shift: floor plane -> z=0 after rotation
    zs = sorted(sum(R[2][i] * p[i] for i in range(3)) for p in pts)
    step = max(1, len(zs) // 2000)
    zs = zs[::step]
    z_floor = zs[max(0, int(len(zs) * 0.02))]   # 2nd percentile ≈ floor
    return {"rotation": R, "z_shift": -z_floor,
            "inliers": inliers, "n_points": len(pts), "normal": list(n)}


def angle_between(R1, R2):
    """Angle (deg) between the two rotations' z-axes (the gauge axis)."""
    import math
    z1 = [R1[2][0], R1[2][1], R1[2][2]]
    z2 = [R2[2][0], R2[2][1], R2[2][2]]
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(z1, z2))))
    return math.degrees(math.acos(dot))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, scene = sys.argv[1], sys.argv[2]
    sdir = v4.STORE / scene
    solve = next((d for d in sdir.glob("images/subsets/primary/cameras/*/")
                  if (d / "points.ply").exists()), None)
    if solve is None:
        sys.exit(f"{scene}: no primary solve with points.ply")
    res = fit(solve / "points.ply")
    print(f"sparse fit: {res['n_points']:,} pts, inliers {res['inliers']}, "
          f"z_shift {res['z_shift']:.4f}")
    if cmd == "verify":
        gt_dirs = sorted(solve.glob("orient/*/transform.json"))
        if not gt_dirs:
            print("no ground-truth orient (mesh-era) for this solve — UNVERIFIABLE here")
            return 1
        gt = json.loads(gt_dirs[0].read_text())
        ang = angle_between(res["rotation"], gt["rotation"])
        dz = abs(res["z_shift"] - gt["z_shift"])
        print(f"vs mesh-era ground truth: z-axis angle {ang:.2f}°, |Δz_shift| {dz:.4f}")
        verdict = "PASS" if ang < 5.0 and dz < 0.15 else "FAIL"
        print(f"verdict: {verdict} (gates: <5° axis, <0.15 z)")
        return 0 if verdict == "PASS" else 1
    if cmd == "run":
        solve_id = solve.name
        oid = v4.identity_hash({"solve": solve_id}, SETTINGS, ALGO)
        odir = solve / "orient" / oid
        if (odir / "metadata.json").exists():
            print(f"NOOP: {odir} exists")
            return 0
        odir.mkdir(parents=True, exist_ok=True)
        (odir / "transform.json").write_text(json.dumps(
            {"rotation": res["rotation"], "z_shift": res["z_shift"]}, indent=2) + "\n")
        v4.write_metadata(odir, task="orient-cameras", algo=ALGO, identity=oid,
                          resolved_inputs={"solve": solve_id}, settings=SETTINGS,
                          measured={"inliers": res["inliers"], "n_points": res["n_points"]})
        print(f"wrote {odir}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
