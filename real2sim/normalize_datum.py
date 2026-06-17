"""STO-SCN-152 — full metric normalize: a match.html MEASURE export → datum.json.

Ties the three SHIPPED seams into one operator action:
  calibrate_datum.recompute(export)  → metric scale (+ gate flags)
  datum_frame.build_datum(centers, up, scale) → camera-derived datum frame
  calibrate_datum.apply_to_gauge(gauge, scale, datum, prov) → additive datum.json

Runs under a numpy python (the rate_renders server shells out to it). No new
math — just the glue the in-tab "Normalize Units" button needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_datum as cd          # noqa: E402
import datum_frame as df              # noqa: E402
import gauge_up                       # noqa: E402
import posed_from_sparse as pfs       # noqa: E402


def camera_centers(posed):
    """center = -R^T t for each w2c (pure python)."""
    out = []
    for e in posed:
        w = e["w2c"]
        R = [[w[i][j] for j in range(3)] for i in range(3)]
        t = [w[i][3] for i in range(3)]
        out.append([-(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]) for i in range(3)])
    return out


def normalize(scene: str, subset: str, solve: str, export: dict, *,
              store: str = "/var/krabby/scenes", gate_thresh: float = 1.5,
              force: bool = False, dry: bool = False) -> dict:
    gauge = Path(store) / scene / "images" / "subsets" / subset / "cameras" / solve
    if not (gauge / "sparse" / "0" / "images.bin").exists():
        return {"error": f"no solve sparse/0 at {gauge}"}
    rec = cd.recompute(export, gate_thresh=gate_thresh)
    scale = rec["s_meters_per_solve_unit"]
    posed = pfs.posed_from_sparse(str(gauge / "sparse" / "0"))
    centers = camera_centers(posed)
    up = gauge_up.up_from_poses([e["w2c"] for e in posed])
    up = [float(x) for x in up]
    datum = df.build_datum(centers, up, scale=scale)
    prov = {"method": "MEASURE + Normalize Units (STO-SCN-152, in-tab)",
            "status": "from match.html export",
            "n_distances": rec.get("n_distances"), "spread": rec.get("spread"),
            "final_scale": scale}
    result = {"ok": True, "scale": scale, "spread": rec.get("spread"),
              "n_distances": rec.get("n_distances"),
              "anisotropy": rec.get("anisotropy_flag"),
              "weak_triangulation": rec.get("weak_triangulation"),
              "da3_scouts_disagree": rec.get("da3_scouts_disagree"),
              "gauge": str(gauge)}
    if dry:
        result["dry"] = True
        return result
    try:
        out = cd.apply_to_gauge(gauge, scale, datum_frame=datum, provenance=prov, force=force)
    except FileExistsError as e:
        return {"error": str(e), "exists": True, **result, "ok": False}
    result["datum_json"] = out
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--solve", required=True)
    ap.add_argument("--export", required=True, help="match.html MEASURE export JSON file")
    ap.add_argument("--store", default="/var/krabby/scenes")
    ap.add_argument("--gate-thresh", type=float, default=1.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    export = json.loads(Path(a.export).read_text())
    res = normalize(a.scene, a.subset, a.solve, export, store=a.store,
                    gate_thresh=a.gate_thresh, force=a.force, dry=a.dry)
    print(json.dumps(res, indent=2))
    sys.exit(1 if res.get("error") else 0)


if __name__ == "__main__":
    main()
