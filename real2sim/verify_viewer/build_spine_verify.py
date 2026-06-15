#!/usr/bin/env python3
"""STO-SCN-100 — build the whole-spine verify surface (assembled gaussian + seams + trajectory).

Extends the per-segment verify (STO-SCN-095) to the WHOLE spine: render the fused cohesive
gaussian (STO-SCN-099) in the global gauge, overlay the global camera trajectory coloured per
segment, mark the seam (shared boundary) frames, and list per-seam 098 residuals — so the
operator can spot what per-segment QA can't: seam misalignment, accumulated drift, inter-
segment coverage gaps, and doubled geometry that survived fusion.

This is the human gate (T-020) for cohesion — it does NOT self-close. A flagged defect routes
to the responsible stage: misalignment/drift → 098 (re-register), doubled/holey seam → 099
(re-fuse), bad cut → 097 (re-segment), under-covered segment → 094 (re-select).

Reuses the STO-SCN-095 surface machinery (two-pass GaussianSplats3D + overlay, frustum/ply
helpers in build_verify) on the ASSEMBLED gaussian rather than one segment — the `100 → 095`
edge. The fused gaussian is already in the global gauge, so the splat transform is identity.

Reads from the store (by identity), or explicit paths:
  spine/<spine>/register/<reg>/global.json     (gauges + global camera centres + seam residuals)
  spine/<spine>/fuse/<fuse>/fused.gs.ply        (the assembled cohesive gaussian)
  images/subsets/<sub>/cameras/<solve>/sparse/0 (per-segment poses + intrinsics, via --solves)
Writes a serve dir: spine_viewer.html · spine.json · fused.gs.ply

Usage:
  build_spine_verify.py <scene> --spine <id> --register <id> --fuse <id> \
      --solves seg0=<sub>/<solve>,... [--serve-dir DIR] [--port 8100] [--no-serve]
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
sys.path.insert(0, str(HERE.parent))            # real2sim/
sys.path.insert(0, str(HERE))                   # verify_viewer/
import posed_from_sparse as pfs                 # noqa: E402
import spine_register as sreg                   # noqa: E402
import gauge_up                                 # noqa: E402
from build_verify import frustum_from_w2c, splat_frame, cull_sphere  # noqa: E402

STORE = Path("/var/krabby/scenes")


def _parse_manifest(spec: str) -> dict:
    out = {}
    for tok in spec.split(","):
        k, _, v = tok.partition("=")
        if not v:
            sys.exit(f"bad manifest entry '{tok}' (need seg=value)")
        out[k] = v
    return out


def build_spine_data(global_json: dict, seg_sparse: dict, scene_dir: Path,
                     title="100 whole-spine verify") -> tuple[dict, list]:
    """Assemble the viewer payload from the 098 global.json + per-segment sparse dirs.

    seg_sparse: {seg_id: sparse/0 Path}. Returns (spine.json dict, all-global-centers list).
    Frustums carry global pose + segment index + a `seam` flag (camera shared by >1 segment);
    trajectory is the global camera path in frame order coloured per segment.
    """
    gauges = global_json["gauges"]
    cam_global = {n: c["center"] for n, c in global_json["cameras"].items()}
    seg_ids = sorted(gauges)
    seg_index = {k: i for i, k in enumerate(seg_ids)}

    # which cameras each segment owns (from its solve), + a global R per camera
    owners: dict = {}
    frustum_by_name: dict = {}
    for k in seg_ids:
        Rk = global_json["gauges"][k]["R"]
        import numpy as np
        Rk = np.asarray(Rk, float)
        posed = pfs.posed_from_sparse(str(seg_sparse[k]))
        for e in posed:
            name = e["name"]
            owners.setdefault(name, []).append(seg_index[k])
            if name not in frustum_by_name:                      # first owner defines the marker
                rflat_local, _c_local = frustum_from_w2c(e["w2c"])
                Rc2w = Rk @ np.asarray(rflat_local, float).reshape(3, 3)   # local c2w -> global
                K = e["K"]
                vfov = math.degrees(2 * math.atan(K[1][2] / K[1][1])) if K[1][1] else 50.0
                aspect = (K[0][2] / K[1][2]) if K[1][2] else 1.5
                frustum_by_name[name] = {
                    "R": [float(x) for x in Rc2w.flatten()],
                    "pos": [float(x) for x in cam_global.get(name, [0, 0, 0])],
                    "name": name, "seg": seg_index[k],
                    "vfov": round(vfov, 2), "aspect": round(aspect, 3)}

    frustums, centers, w2c_global = [], [], []
    import numpy as np
    for name in sorted(frustum_by_name):
        f = frustum_by_name[name]
        f["seam"] = len(owners[name]) > 1                        # shared boundary frame
        frustums.append(f)
        centers.append(f["pos"])
        Rc2w = np.asarray(f["R"], float).reshape(3, 3)           # global c2w
        Rw2c = Rc2w.T
        t = -Rw2c @ np.asarray(f["pos"], float)
        w2c_global.append([[*Rw2c[r], t[r]] for r in range(3)] + [[0, 0, 0, 1]])

    # trajectory: frame-ordered global centres, tagged with segment (for per-seg colour)
    trajectory = [[*f["pos"], f["seg"]] for f in frustums]

    # world up from the (global) poses — gravity ⟂ camera-right axes (gauge_up)
    up = gauge_up.up_from_poses(w2c_global) if w2c_global else [0, -1, 0]

    seams = []
    for s in global_json.get("seams", []):
        seams.append({"i": seg_index.get(str(s["i"]), s["i"]),
                      "j": seg_index.get(str(s["j"]), s["j"]),
                      "residual_rel": s.get("residual_rel", 0.0),
                      "consensus_frac": s.get("consensus_frac", 1.0),
                      "n_outlier": s.get("n_outlier", 0),
                      "registrable": s.get("registrable", True)})

    data = {"title": title, "n_segments": len(seg_ids), "frustums": frustums,
            "trajectory": trajectory, "seams": seams, "up": up,
            "n_cameras": len(frustums), "n_seam_frames": sum(1 for f in frustums if f["seam"])}
    return data, centers


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build + serve the STO-SCN-100 whole-spine verify surface.")
    ap.add_argument("scene")
    ap.add_argument("--spine", required=True, help="spine@0 identity")
    ap.add_argument("--register", required=True, help="spine-register@0 identity")
    ap.add_argument("--fuse", required=True, help="spine-fuse@0 identity")
    ap.add_argument("--solves", required=True, help="comma list seg=subset/solve (per-segment poses)")
    ap.add_argument("--fused", default=None, help="override: explicit fused .ply path")
    ap.add_argument("--global-json", dest="global_json", default=None,
                    help="override: explicit register global.json path")
    ap.add_argument("--serve-dir", default=None)
    ap.add_argument("--cull-radius", type=float, default=2.5,
                    help="keep splats within this x scene-radius (0 = no cull)")
    ap.add_argument("--max-splats", type=int, default=800000)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--no-serve", action="store_true")
    a = ap.parse_args(argv)

    scene_dir = STORE / a.scene
    gj = Path(a.global_json) if a.global_json else \
        scene_dir / "spine" / a.spine / "register" / a.register / "global.json"
    fused = Path(a.fused) if a.fused else \
        scene_dir / "spine" / a.spine / "fuse" / a.fuse / "fused.gs.ply"
    if not gj.exists():
        sys.exit(f"no register global.json at {gj}")
    if not fused.exists():
        sys.exit(f"no fused gaussian at {fused}")
    solves = _parse_manifest(a.solves)
    seg_sparse = {k: scene_dir / "images" / "subsets" / sub.split("/")[0] / "cameras"
                  / sub.split("/")[1] / "sparse" / "0" for k, sub in solves.items()}
    for k, sp in seg_sparse.items():
        if not (sp / "images.bin").exists():
            sys.exit(f"no solve sparse/0 for segment {k} at {sp}")

    glob = json.loads(gj.read_text())
    data, centers = build_spine_data(glob, seg_sparse, scene_dir,
                                     title=f"{a.scene} · spine {a.spine} · {data_n(glob)} segments")

    # framing on the fused gaussian (already global) clamped to the camera extent
    sc, sr = splat_frame(fused, centers)
    data["scene_ctr"], data["scene_radius"] = sc, sr

    serve = Path(a.serve_dir) if a.serve_dir else Path(f"/tmp/spine-verify-{a.scene}-{a.spine}")
    serve.mkdir(parents=True, exist_ok=True)
    (serve / "spine.json").write_text(json.dumps(data, indent=2) + "\n")
    shutil.copy2(HERE / "spine_viewer.html", serve / "spine_viewer.html")
    if a.cull_radius and a.cull_radius > 0:
        cull_sphere(fused, serve / "fused.gs.ply", sc, sr * a.cull_radius, a.max_splats)
    else:
        shutil.copy2(fused, serve / "fused.gs.ply")

    print(f"  spine verify: {data['n_segments']} segments · {data['n_cameras']} cameras · "
          f"{data['n_seam_frames']} seam frames · {len(data['seams'])} seams")
    print(f"  serve dir: {serve}")
    if a.no_serve:
        return 0
    print(f"  serving http://localhost:{a.port}/spine_viewer.html  (Ctrl-C to stop)")
    subprocess.run([sys.executable, "-m", "http.server", str(a.port)], cwd=serve)
    return 0


def data_n(glob):
    return len(glob.get("gauges", {}))


if __name__ == "__main__":
    raise SystemExit(_main())
