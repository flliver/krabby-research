#!/usr/bin/env python3
"""STO-SCN-095 — build the verify surface (scout splat + proposed-N frustums) + serve.

v1 inspect-only: assemble the data the viewer needs and serve it locally. Shows
the scout gaussian with the FULL posed pool as dim frustums (coverage) and the
094 proposed-N as bright green — the human SEES the proposal + gaps in the scene.
(accept/drop/add edit controls are v2.)

Reads, from the store:
  cameras/<solve>/sparse/0       (poses -> frustums, via posed_from_sparse)
  cameras/<solve>/scout/<id>/scout.gs.ply   (the DA3 scout, solve gauge)
Selects the proposed-N (select_views) and writes a serve dir:
  <serve>/viewer.html · frustums.json · scout.gs.ply

Usage:
  build_verify.py <scene> --solve <id> --scout <id> [--subset <id>] [--n 24]
      [--serve-dir DIR] [--port 8099] [--no-serve]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # real2sim/
import posed_from_sparse as pfs                # noqa: E402
import select_views as selv                    # noqa: E402

STORE = Path("/var/krabby/scenes")


def frustum_from_w2c(w2c):
    """w2c (4x4) -> (R_c2w flattened row-major [9], camera center [3])."""
    R = [[w2c[i][j] for j in range(3)] for i in range(3)]   # R_w2c
    t = [w2c[i][3] for i in range(3)]
    center = [-(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]) for i in range(3)]
    c2w = [[R[j][i] for j in range(3)] for i in range(3)]    # R_w2c^T
    rflat = [c2w[r][c] for r in range(3) for c in range(3)]
    return rflat, center


def build_frustums(sparse_dir, n, title="095 verify", div_angle=10.0):
    posed = pfs.posed_from_sparse(str(sparse_dir))
    proposed = set(selv.select_from_sparse(str(sparse_dir), n, div_angle=div_angle)[1]["selected"])
    frustums, cs = [], []
    for e in posed:
        rflat, c = frustum_from_w2c(e["w2c"])
        frustums.append({"R": rflat, "pos": c, "proposed": e["name"] in proposed})
        cs.append(c)
    ctr = [sum(c[i] for c in cs) / max(1, len(cs)) for i in range(3)]
    mn = [min(c[i] for c in cs) for i in range(3)]
    mx = [max(c[i] for c in cs) for i in range(3)]
    diag = sum((mx[i] - mn[i]) ** 2 for i in range(3)) ** 0.5
    return {"title": title, "frustums": frustums, "gauss_ctr": ctr, "up": [0, -1, 0],
            "cam_diag": diag, "n_proposed": len(proposed), "n_pool": len(posed)}


def cull_ply(src, dst, center, radius):
    """Drop non-finite + far gaussians (DA3's sky/low-confidence halo at ~1e36
    that hijacks the viewer's auto-framing). Keep splats within `radius` of the
    camera centroid. Rewrites the binary PLY with the updated vertex count."""
    import numpy as np
    with open(src, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.read(1)
        off = f.tell()
    lines = hdr.decode("ascii", "replace").splitlines()
    n = next(int(L.split()[-1]) for L in lines if L.startswith("element vertex"))
    props = [L.split()[-1] for L in lines if L.startswith("property")]
    buf = np.fromfile(src, dtype=np.float32, offset=off, count=n * len(props)).reshape(n, len(props))
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
    xyz = buf[:, [xi, yi, zi]].astype(np.float64)
    c = np.asarray(center)
    keep = np.isfinite(xyz).all(1) & (np.linalg.norm(np.nan_to_num(xyz) - c, axis=1) <= radius)
    kept = buf[keep]
    new_hdr = []
    for L in lines:
        new_hdr.append(f"element vertex {len(kept)}" if L.startswith("element vertex") else L)
    out = ("\n".join(new_hdr) + "\n").encode("ascii")
    with open(dst, "wb") as f:
        f.write(out)
        f.write(kept.astype(np.float32).tobytes())
    print(f"  culled splat: {len(kept)}/{n} kept (radius {radius:.1f} of camera centroid)")


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build + serve the STO-SCN-095 verify surface.")
    ap.add_argument("scene")
    ap.add_argument("--solve", required=True)
    ap.add_argument("--scout", required=True, help="scout@0 identity")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--n", type=int, default=24, help="proposed-N to highlight")
    ap.add_argument("--div-angle", type=float, default=10.0,
                    help="viewpoint-diversity penalty angle (0 = off; higher = more spread)")
    ap.add_argument("--serve-dir", default=None)
    ap.add_argument("--cull-factor", type=float, default=3.0,
                    help="keep splats within this x camera-bbox-diagonal of the centroid")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--no-serve", action="store_true")
    a = ap.parse_args(argv)

    subset = a.subset
    if not subset:
        pr = STORE / a.scene / "images" / "subsets" / "primary"
        subset = pr.resolve().name if pr.exists() else sys.exit("need --subset")
    cam = STORE / a.scene / "images" / "subsets" / subset / "cameras" / a.solve
    sparse = cam / "sparse" / "0"
    scout_ply = cam / "scout" / a.scout / "scout.gs.ply"
    if not (sparse / "images.bin").exists():
        sys.exit(f"no solve sparse/0 at {sparse}")
    if not scout_ply.exists():
        sys.exit(f"no scout at {scout_ply} (run `v4exec scout` first)")

    serve = Path(a.serve_dir) if a.serve_dir else Path(f"/tmp/verify-{a.scene}-{a.solve}")
    serve.mkdir(parents=True, exist_ok=True)
    fr = build_frustums(sparse, a.n, title=f"{a.scene} · solve {a.solve} · N={a.n} · div={a.div_angle}",
                        div_angle=a.div_angle)
    (serve / "frustums.json").write_text(json.dumps(fr) + "\n")
    shutil.copy2(HERE / "viewer.html", serve / "viewer.html")
    # cull DA3's far-splat halo so the viewer frames the scene (not the 1e36 halo)
    cull_ply(scout_ply, serve / "scout.gs.ply", fr["gauss_ctr"], a.cull_factor * fr["cam_diag"])
    print(f"verify surface: {fr['n_proposed']} proposed / {fr['n_pool']} pool frustums + scout")
    print(f"  serve dir: {serve}")
    if a.no_serve:
        print(f"  serve:  python3 -m http.server {a.port} --directory {serve}")
        print(f"  open:   http://localhost:{a.port}/viewer.html")
        return 0
    print(f"  -> open http://localhost:{a.port}/viewer.html")
    subprocess.run(["python3", "-m", "http.server", str(a.port), "--directory", str(serve)])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
