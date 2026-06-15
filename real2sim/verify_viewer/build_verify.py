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
import math
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


def da3_align(npz_path, scout_posed):
    """RETIRED (STO-SCN-105): the npz `extrinsics` are the ECHOED INPUT cameras, so this
    Umeyama aligns to identity (scale≈1) and never recovered DA3's gaussian normalization —
    that was the mis-registration bug. Registration is now a single `scale_factor` applied to
    the splat (see scout_gauge.py + build_frustums). Kept only as the documented dead-end.

    Umeyama similarity (scale, R, t) mapping INPUT-gauge camera centers -> DA3's output
    (gaussian) frame, from DA3's saved output extrinsics. w2c/c2w convention auto-detected by
    residual. Returns (scale, R(3x3), t(3)) or None."""
    import numpy as np
    from gauge_align import umeyama, residuals          # noqa: E402 (real2sim on path)
    ext = np.asarray(np.load(npz_path)["extrinsics"], dtype=np.float64)   # (N,3,4)
    pin = sorted(scout_posed, key=lambda e: e["name"])   # DA3 received sorted-glob order
    P = np.array([frustum_from_w2c(e["w2c"])[1] for e in pin], dtype=np.float64)
    if len(P) != len(ext):
        print(f"  da3-align SKIPPED: {len(P)} posed vs {len(ext)} npz extrinsics")
        return None
    Rd, td = ext[:, :3, :3], ext[:, :3, 3]
    cands = {"w2c": np.einsum("nji,nj->ni", Rd, -td), "c2w": td}   # camera centers each way
    best = None
    for name, Q in cands.items():
        s, R, t = umeyama(P, Q)
        res = float(residuals(P, Q, s, R, t).mean())
        if best is None or res < best[0]:
            best = (res, s, R, t, name)
    res, s, R, t, name = best
    print(f"  da3-align: convention={name} scale={s:.3f} residual={res:.4f} "
          f"({'OK' if res < 0.1 else 'HIGH — check'}) — input gauge -> DA3 frame")
    return s, R, t


def _apply_xform(rflat, c, xform):
    """Carry one frustum (c2w-flat rotation + center) through the similarity xform."""
    import numpy as np
    s, R, t = xform
    c_new = (s * (R @ np.asarray(c)) + np.asarray(t)).tolist()
    c2w_new = R @ np.asarray(rflat, dtype=np.float64).reshape(3, 3)   # rotate orientation
    return c2w_new.flatten().tolist(), c_new


def build_frustums(sparse_dir, n, title="095 verify", selector="voxel", grid=64,
                   div_angle=10.0, scout_dir=None, cull_expand=1.0):
    posed = pfs.posed_from_sparse(str(sparse_dir))
    # STO-SCN-105 (corrected, ground-truthed): the frustums STAY in the solve
    # gauge. The scout gs_ply lives in DA3's normalized (cam-0-recentered +
    # median-dist) frame, which differs from the solve by a FULL similarity —
    # scale + ROTATION + translation (the rotation is the ~125° we saw). The
    # transform is recovered automatically by da3_infer_posed (Umeyama of DA3's
    # predicted poses → input poses) and read here via scout_register.gauge_for.
    # p_solve = scale · R(quat[xyzw]) · p_gs + translate. (Point-cloud ICP is
    # NOT used — it's ambiguous under scene symmetry; see knowledge note.)
    splat_scale, splat_translate, splat_quat = 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    if scout_dir:
        import scout_register
        g = scout_register.gauge_for(scout_dir)
        splat_scale = g["scale"]
        splat_translate = g.get("translate", [0.0, 0.0, 0.0])
        splat_quat = g.get("quat", [0.0, 0.0, 0.0, 1.0])
        if g.get("registered"):
            print(f"  scout-gauge: scale={splat_scale:.4f} "
                  f"quat={[round(x, 3) for x in splat_quat]} "
                  f"translate={[round(x, 3) for x in splat_translate]} "
                  f"(gs→solve, {g.get('source')})")
        else:
            print("  scout-gauge: NOT registered (no scout_gauge.json transform "
                  "/ no manual override) — splat at identity; re-run scout or "
                  "register in match.html")
    faces = None
    if selector == "voxel":                                   # STO-SCN-103
        import voxel_coverage as vc
        sel_names, rep, faces = vc.select_with_faces(str(sparse_dir), n, grid=grid)
        proposed = set(sel_names)
        print(f"  voxel-coverage: face-coverage {rep['face_coverage_pct']}% · "
              f"view-spread {rep['median_view_spread_deg']} deg · "
              f"{len(faces['items'])}/{faces['n_faces_total']} faces in overlay")
    else:                                                     # STO-SCN-094 (legacy track)
        proposed = set(selv.select_from_sparse(str(sparse_dir), n, div_angle=div_angle)[1]["selected"])
    frustums, cs = [], []
    for e in posed:
        rflat, c = frustum_from_w2c(e["w2c"])               # solve gauge (native)
        K = e["K"]
        vfov = math.degrees(2 * math.atan(K[1][2] / K[1][1])) if K[1][1] else 50.0
        aspect = (K[0][2] / K[1][2]) if K[1][2] else 1.5
        frustums.append({"R": rflat, "pos": c, "proposed": e["name"] in proposed,
                         "name": e["name"], "vfov": round(vfov, 2), "aspect": round(aspect, 3)})
        cs.append(c)
    import numpy as np
    ctr = [sum(c[i] for c in cs) / max(1, len(cs)) for i in range(3)]
    mn = [min(c[i] for c in cs) for i in range(3)]
    mx = [max(c[i] for c in cs) for i in range(3)]
    diag = sum((mx[i] - mn[i]) ** 2 for i in range(3)) ** 0.5
    # world up = GRAVITY recovered from the posed cameras (gauge_up: ⟂ to all camera-right
    # axes; SfM gauge has no absolute orientation). Robust to pitch; validated 1.36° vs the
    # operator's manual up on 001-patio. Replaces the hardcoded/avg-up guess.
    import gauge_up
    up = gauge_up.up_from_poses([e["w2c"] for e in posed])   # solve gauge (no xform)
    print(f"  gauge up (from poses): {[round(x,3) for x in up]} "
          f"(roll spread {gauge_up.roll_spread_deg([e['w2c'] for e in posed]):.1f}°)")

    # camera-bounded cull box, computed in the GRAVITY-ALIGNED frame (operator spec): the box
    # is axis-aligned to gravity (up) + the ground plane, NOT to the arbitrary solve axes — so
    # its vertical (up) extent is the tight ground-height and the two horizontal (ground) axes
    # are wider. Circumscribe the cameras in this frame, expand each axis by cull_expand per
    # side, emit the basis (Rg) + box (gmin/gmax) for the cull, and the 8 corners (solve gauge)
    # so the viewer draws the oriented box level with the ground grid.
    U = np.asarray(up, float); U /= (np.linalg.norm(U) or 1.0)
    ref = np.array([1.0, 0, 0]) if abs(U[0]) < 0.9 else np.array([0, 1.0, 0])
    e0 = ref - U * float(ref @ U); e0 /= (np.linalg.norm(e0) or 1.0)   # a ground-plane axis
    e1 = np.cross(U, e0)                                                # the other ground axis
    Rg = np.vstack([e0, e1, U])                                         # solve -> gravity (rows)
    Cg = np.asarray(cs, float) @ Rg.T                                   # cameras in gravity frame
    gmn = Cg.min(0); gmx = Cg.max(0); gspan = gmx - gmn
    gmin = (gmn - gspan * cull_expand)
    gmax = (gmx + gspan * cull_expand)
    corners = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                cg = np.array([gmax[0] if i else gmin[0], gmax[1] if j else gmin[1],
                               gmax[2] if k else gmin[2]])
                corners.append([round(float(x), 4) for x in (Rg.T @ cg)])   # gravity -> solve
    print(f"  cull box (gravity-aligned): ground {gspan[0]:.2f}x{gspan[1]:.2f}, "
          f"vertical {gspan[2]:.2f} (expand {cull_expand:+.0%}/side)")
    return {"title": title, "frustums": frustums, "gauss_ctr": ctr, "up": up,
            "cam_diag": diag, "n_proposed": len(proposed), "n_pool": len(posed),
            "splat_scale": splat_scale, "splat_translate": splat_translate,
            "splat_quat": splat_quat, "faces": faces,
            "cull_box": {"R": Rg.tolist(), "gmin": gmin.tolist(), "gmax": gmax.tolist(),
                         "corners": corners}}


def splat_frame(ply_path, cam_centers):
    """Robust scene center + radius from the splat CORE + cameras — read-only (never
    rewrites the .ply). DA3 flings a ~0.1% outlier tail to hundreds of units; framing on
    the median + p90-radius (clamped to cover the cameras) ignores it so the viewer frames
    the actual scene, not the sprawl. Returns (center[3], radius)."""
    import re
    import numpy as np
    head = b""
    with open(ply_path, "rb") as f:
        while b"end_header\n" not in head:
            head += f.read(256)
    off = head.index(b"end_header\n") + len(b"end_header\n")
    n = int(re.search(rb"element vertex (\d+)", head).group(1))
    props = head.count(b"property float")                     # 17 for 3DGS
    xyz = np.fromfile(ply_path, dtype=np.float32, offset=off,
                      count=n * props).reshape(n, props)[:, :3].astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(1)]
    ctr = np.median(xyz, axis=0)
    core_r = float(np.percentile(np.linalg.norm(xyz - ctr, axis=1), 90))
    cc = np.asarray(cam_centers, dtype=np.float64)
    cam_r = float(np.linalg.norm(cc - ctr, axis=1).max()) if len(cc) else core_r
    return [round(x, 4) for x in ctr.tolist()], round(max(core_r, cam_r), 3)


def cull_sphere(src, dst, center, radius, max_splats=500000):
    """Keep gaussians within `radius` of `center` (drops the rain-reflection cones), then
    DECIMATE to at most `max_splats` (strided) so the verify splat is light enough to orient
    fluidly — the density of the core, not the far cones, is what makes it heavy. CORRECT
    binary rewrite of standard 3DGS 17xfloat32 (header offset includes the trailing newline —
    the off-by-one that corrupted the first attempt). Writes the SERVE copy only; never the
    store original. Self-verifies the output before returning (T-012, post-cull-disaster)."""
    import re
    import numpy as np
    head = b""
    with open(src, "rb") as f:
        while b"end_header\n" not in head:
            head += f.read(4096)
    off = head.index(b"end_header\n") + len(b"end_header\n")
    n = int(re.search(rb"element vertex (\d+)", head).group(1))
    P = head.count(b"property float")                         # 17 for 3DGS
    buf = np.fromfile(src, dtype=np.float32, offset=off, count=n * P).reshape(n, P)
    xyz = buf[:, :3].astype(np.float64)
    c = np.asarray(center, dtype=np.float64)
    keep = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz - c, axis=1) <= radius)
    kept = np.ascontiguousarray(buf[keep])
    in_sphere = len(kept)
    if max_splats and len(kept) > max_splats:                 # strided decimation (deterministic)
        kept = np.ascontiguousarray(kept[:: (len(kept) + max_splats - 1) // max_splats])
    new_head = re.sub(rb"element vertex \d+", f"element vertex {len(kept)}".encode(), head[:off])
    with open(dst, "wb") as f:
        f.write(new_head)
        f.write(kept.tobytes())
    # self-verify: re-parse the written file exactly as a loader would
    vh = b""
    with open(dst, "rb") as f:
        while b"end_header\n" not in vh:
            vh += f.read(4096)
    voff = vh.index(b"end_header\n") + len(b"end_header\n")
    vn = int(re.search(rb"element vertex (\d+)", vh).group(1))
    vxyz = np.fromfile(dst, dtype=np.float32, offset=voff, count=vn * P).reshape(vn, P)[:, :3]
    ok = (vn == len(kept) and np.isfinite(vxyz).all()
          and (np.linalg.norm(vxyz.astype(np.float64) - c, axis=1) <= radius + 1e-3).all())
    print(f"  cull-sphere: {in_sphere}/{n} in r={radius:.2f}, decimated to {len(kept)} — "
          f"{'VERIFIED ok' if ok else 'SELF-CHECK FAILED'}")
    if not ok:
        raise RuntimeError("cull self-check failed — not serving a corrupt splat")
    return len(kept), n


def cull_box(src, dst, Rg, gmin, gmax, scale, R, t, max_splats=500000):
    """Cull splats to a camera-bounded box computed in the GRAVITY-ALIGNED frame (operator
    spec 2026-06-14). `Rg` (3x3, rows = ground-plane e0, e1, gravity-up) maps solve→gravity;
    `gmin`/`gmax` are the (expanded) camera box in that frame, so its vertical (up) axis is the
    tight ground-height extent and the horizontal ground axes are wider. The .ply is in the
    GAUSSIAN frame, so each splat is mapped gaussian→solve (scale·R·p_gs + t) → gravity (Rg·p)
    and tested. Decimates to <= max_splats. Correct 17xfloat32 rewrite; self-verifies (T-012)."""
    import re
    import numpy as np
    Rg = np.asarray(Rg, np.float64)
    gmn = np.asarray(gmin, np.float64)
    gmx = np.asarray(gmax, np.float64)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    head = b""
    with open(src, "rb") as f:
        while b"end_header\n" not in head:
            head += f.read(4096)
    off = head.index(b"end_header\n") + len(b"end_header\n")
    n = int(re.search(rb"element vertex (\d+)", head).group(1))
    P = head.count(b"property float")                         # 17 for 3DGS
    buf = np.fromfile(src, dtype=np.float32, offset=off, count=n * P).reshape(n, P)
    xyz = buf[:, :3].astype(np.float64)
    grav = (scale * (xyz @ R.T) + t) @ Rg.T                  # gaussian → solve → gravity frame
    keep = (np.isfinite(grav).all(1)
            & (grav >= gmn).all(1) & (grav <= gmx).all(1))
    kept = np.ascontiguousarray(buf[keep])
    in_box = len(kept)
    if max_splats and len(kept) > max_splats:                 # deterministic strided decimation
        kept = np.ascontiguousarray(kept[:: (len(kept) + max_splats - 1) // max_splats])
    new_head = re.sub(rb"element vertex \d+", f"element vertex {len(kept)}".encode(), head[:off])
    with open(dst, "wb") as f:
        f.write(new_head)
        f.write(kept.tobytes())
    # self-verify: re-parse + re-test the box in the gravity frame
    vh = b""
    with open(dst, "rb") as f:
        while b"end_header\n" not in vh:
            vh += f.read(4096)
    voff = vh.index(b"end_header\n") + len(b"end_header\n")
    vn = int(re.search(rb"element vertex (\d+)", vh).group(1))
    vxyz = np.fromfile(dst, dtype=np.float32, offset=voff, count=vn * P).reshape(vn, P)[:, :3]
    vg = (scale * (vxyz.astype(np.float64) @ R.T) + t) @ Rg.T
    ok = (vn == len(kept) and np.isfinite(vg).all()
          and (vg >= gmn - 1e-3).all() and (vg <= gmx + 1e-3).all())
    print(f"  cull-box (gravity-aligned): {in_box}/{n} in camera box, decimated to {len(kept)} — "
          f"{'VERIFIED ok' if ok else 'SELF-CHECK FAILED'}")
    if not ok:
        raise RuntimeError("cull-box self-check failed — not serving a corrupt splat")
    return len(kept), n


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build + serve the STO-SCN-095 verify surface.")
    ap.add_argument("scene")
    ap.add_argument("--solve", required=True)
    ap.add_argument("--scout", required=True, help="scout@0 identity")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--n", type=int, default=24, help="proposed-N to highlight")
    ap.add_argument("--selector", choices=["voxel", "track"], default="voxel",
                    help="voxel = STO-SCN-103 coverage; track = STO-SCN-094 legacy")
    ap.add_argument("--grid", type=int, default=64, help="voxel grid resolution (voxel selector)")
    ap.add_argument("--div-angle", type=float, default=10.0,
                    help="track-selector viewpoint-diversity penalty angle")
    ap.add_argument("--serve-dir", default=None)
    ap.add_argument("--cull-radius", type=float, default=2.5,
                    help="keep splats within this x scene-radius of the scene center "
                         "(drops the rain-reflection far cones; 0 = no cull)")
    ap.add_argument("--max-splats", type=int, default=500000,
                    help="decimate the verify splat to at most this many gaussians (speed)")
    ap.add_argument("--cull-expand", type=float, default=1.0,
                    help="camera-AABB cull: expand the camera box by this x per side "
                         "(1.0 = +100% = 3x camera span; lower crops the DA3 halo harder)")
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
    fr = build_frustums(sparse, a.n, selector=a.selector, grid=a.grid, div_angle=a.div_angle,
                        scout_dir=scout_ply.parent, cull_expand=a.cull_expand,
                        title=f"{a.scene} · solve {a.solve} · N={a.n} · {a.selector}")
    # splat_frame reads the .ply in the GAUSSIAN frame; the viewer maps the
    # splat into the solve gauge via p_solve = scale·p_gs + translate, so the
    # framing must be in the solve gauge too: take the gaussian-frame core (no
    # camera clamp), apply scale+translate, THEN clamp to the frustum centers.
    import numpy as np
    ss = fr.get("splat_scale", 1.0)
    st = np.asarray(fr.get("splat_translate", [0.0, 0.0, 0.0]), dtype=float)
    qx, qy, qz, qw = fr.get("splat_quat", [0.0, 0.0, 0.0, 1.0])
    Rq = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
                   [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
                   [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]], dtype=float)
    sc_g, sr_g = splat_frame(scout_ply, [])               # gaussian-frame core only
    sc = (ss * (Rq @ np.asarray(sc_g)) + st)              # → solve gauge: scale·R·p + t
    sc = [round(float(x), 4) for x in sc]
    cc = np.asarray([f["pos"] for f in fr["frustums"]], dtype=float)
    cam_r = float(np.linalg.norm(cc - np.asarray(sc), axis=1).max()) if len(cc) else sr_g * ss
    sr = round(max(sr_g * ss, cam_r), 3)
    fr["scene_ctr"], fr["scene_radius"] = sc, sr
    print(f"  scene framing: center {sc} radius {sr} (solve gauge; "
          f"splat scale {ss:.4f} quat {[round(x,3) for x in (qx,qy,qz,qw)]}; "
          f"ignores DA3 outlier tail)")

    # scout frames + their poses for the photo-match tool (match.html): copy the actual
    # frame images into the serve dir and emit name->frustum-index so the viewer can pin a
    # photo and snap the camera to its known pose.
    scout_dir = scout_ply.parent
    frames_dir = serve / "frames"
    frames_dir.mkdir(exist_ok=True)
    raw_dir = serve / "_frames_raw"
    raw_dir.mkdir(exist_ok=True)
    pj = scout_dir / "posed.json"
    scout_names = [e["name"] for e in json.loads(pj.read_text())] if pj.exists() else []
    name2idx = {f["name"]: i for i, f in enumerate(fr["frustums"])}
    want, copied = set(scout_names), set()
    for md in (STORE / a.scene).glob("images/*/metadata.json"):
        nm = json.loads(md.read_text()).get("original_name")
        if nm in want and nm not in copied:
            img = next((p for p in md.parent.glob("image.*")), None)
            if img:
                shutil.copy2(img, raw_dir / nm)
                copied.add(nm)
    # the solve/scout gauge is PINHOLE (fisheye undistorted first), so the verify photo must
    # be the DE-WARPED frame — else clicks on fisheye pixels don't match the pinhole rays.
    cap = STORE / a.scene / "capture.json"
    capd = json.loads(cap.read_text()) if cap.exists() else {}
    dewarped = False
    if capd.get("mode") == "fisheye":
        try:
            import undistort_fisheye as udf
            udf.undistort_dir(str(raw_dir), str(frames_dir), capd["make"], capd["model"],
                              capd["mode"], balance=0.0)
            dewarped = True
            print("  frames: de-warped fisheye -> pinhole (matches solve gauge)")
        except Exception as e:                       # no calibration etc. -> fall back, warn loudly
            print(f"  frames: WARN de-warp failed ({e}); serving RAW fisheye (solve will be off)")
    if not dewarped:
        for p in raw_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, frames_dir / p.name)
    fr["scout_frames"] = [{"name": nm, "idx": name2idx[nm], "file": f"frames/{nm}"}
                          for nm in scout_names if nm in name2idx and nm in copied]
    print(f"  match tool: {len(fr['scout_frames'])} scout frames -> frames/"
          f" ({'de-warped' if dewarped else 'raw'})")

    (serve / "frustums.json").write_text(json.dumps(fr) + "\n")
    shutil.copy2(HERE / "viewer.html", serve / "viewer.html")
    if (HERE / "match.html").exists():
        shutil.copy2(HERE / "match.html", serve / "match.html")
    if fr.get("cull_box"):                                     # gravity-aligned camera box cull
        bx = fr["cull_box"]
        cull_box(scout_ply, serve / "scout.gs.ply", bx["R"], bx["gmin"], bx["gmax"],
                 ss, Rq, st, a.max_splats)
    elif a.cull_radius > 0:                                    # legacy cull-sphere fallback
        cull_sphere(scout_ply, serve / "scout.gs.ply", sc_g, a.cull_radius * sr_g, a.max_splats)
    else:
        shutil.copy2(scout_ply, serve / "scout.gs.ply")
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
