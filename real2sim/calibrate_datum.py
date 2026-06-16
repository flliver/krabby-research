"""STO-SCN-016 — recompute + record the datum metric scale from a match.html MEASURE export.

Reads the control-distance JSON exported by `match.html` MEASURE mode (STO-SCN-144),
**re-triangulates** each endpoint from the stored camera rays via `metric_scale` (the tested
core — does NOT trust the browser's numbers), aggregates to one robust scale (log-median +
spread), runs the DA3 gross-error gate, and writes a **datum-scale record** (the scalar `s` +
full provenance + verdict).

Applying `s` to the v4 gauge node (cameras + meshes inherit it) is the downstream wiring — a new
calibrated gauge node, store-identity-additive (STO-SCN-136 backwards-compat rule). That step
needs a real operator measurement to test end-to-end (T-020), so this module stops at compute +
record; `apply_to_gauge()` is the documented seam.

Usage:
    calibrate_datum.py --export measure.json [--out datum_scale.json] [--gate-thresh 1.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import metric_scale as ms  # noqa: E402


def _triangulate_endpoint(picks):
    """Average closest-point-of-approach over all pick pairs (== match.html commitEndpoint)."""
    pts, gaps, pars = [], [], []
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            p, gap, par = ms.triangulate_rays(picks[i]["o"], picks[i]["d"],
                                              picks[j]["o"], picks[j]["d"])
            pts.append(p); gaps.append(gap); pars.append(par)
    if not pts:
        raise ValueError("endpoint needs >=2 picks to triangulate")
    return np.mean(pts, axis=0), max(gaps), min(pars)


def recompute(export, gate_thresh=1.5, s_monocular=None):
    """Authoritative recompute from a MEASURE export. Returns the datum-scale record dict."""
    per = []
    for k, di in enumerate(export.get("distances", [])):
        D = float(di["D"])
        if di.get("picks1") and di.get("picks2"):                  # re-triangulate (authoritative)
            P1, g1, par1 = _triangulate_endpoint(di["picks1"])
            P2, g2, par2 = _triangulate_endpoint(di["picks2"])
            gaps, parallax = [g1, g2], min(par1, par2)
        else:                                                       # fall back to stored points
            P1, P2 = np.asarray(di["P1"], float), np.asarray(di["P2"], float)
            gaps, parallax = di.get("gaps"), di.get("parallax")
        s, d_solve = ms.scale_from_distance(P1, P2, D)
        per.append({"i": k + 1, "D": D, "d_solve": d_solve, "s": s,
                    "gaps": gaps, "parallax_deg": parallax,
                    "weak": (parallax is not None and parallax < 2.0)})
    if not per:
        raise ValueError("export has no distances")

    scales = [p["s"] for p in per]
    s_median, spread = ms.aggregate_scales(scales)
    passed, ratio, mono_med = ms.da3_gate(s_median, s_monocular, gate_thresh)

    sf = [x for x in export.get("da3_scale_factors", []) if x and x > 0]
    da3_spread = (max(sf) / min(sf)) if len(sf) > 1 else 1.0

    return {
        "story": "STO-SCN-016",
        "s_meters_per_solve_unit": s_median,
        "n_distances": len(per),
        "spread": spread,
        "anisotropy_flag": spread > 1.5,                # different-axis distances disagree => distortion
        "per_distance": per,
        "weak_triangulation": any(p["weak"] for p in per),
        "da3_scale_factors": sf,
        "da3_scale_factor_spread": da3_spread,
        "da3_scouts_disagree": da3_spread > 1.5,        # the 11.22-vs-3.4 case
        "da3_gate": {"applied": passed is not None, "passed": passed,
                     "ratio": ratio, "monocular_median": mono_med, "thresh": gate_thresh,
                     "note": ("monocular prior not provided in m/unit (scale_factor->s conversion "
                              "deferred); gate inert, control distance authoritative")
                             if passed is None else None},
        "note": "apply s as ONE scalar on the solve Sim3 (datum-level); see apply_to_gauge().",
    }


def apply_to_gauge(gauge_dir, scale, datum_frame=None, provenance=None, force=False):
    """Write the metric datum as an ADDITIVE sidecar (`datum.json`) in the gauge/orient dir.

    Records the metric scale (`p_meters = scale * p_solve_gauge`) + the optional camera-derived
    datum frame (datum_frame.py) + provenance. This is store-safe by construction: it does NOT
    re-ground existing meshes or rewrite cameras.json/oriented.json, so no materialized mesh
    identity is re-keyed (STO-SCN-136 backwards-compat). Downstream consumers (USD export 017,
    primitive-cull authoring 145) read the sidecar to interpret the gauge in meters.

    Refuses to clobber an existing `datum.json` unless `force` (T-018 preserve output).
    """
    gp = Path(gauge_dir)
    out = gp / "datum.json"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force=True to overwrite")
    rec = {"scale_m_per_unit": float(scale),
           "datum_frame": datum_frame,
           "provenance": provenance,
           "note": "metric datum sidecar (additive); p_meters = scale * p_solve-gauge. STO-SCN-016."}
    gp.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="match.html MEASURE export JSON")
    ap.add_argument("--out", default=None, help="write the datum-scale record here")
    ap.add_argument("--gate-thresh", type=float, default=1.5)
    args = ap.parse_args()
    export = json.loads(Path(args.export).read_text())
    rec = recompute(export, gate_thresh=args.gate_thresh)
    txt = json.dumps(rec, indent=2)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    print(txt)
    print(f"\n  s = {rec['s_meters_per_solve_unit']:.5f} m/solve-unit "
          f"({rec['n_distances']} distance(s), spread {rec['spread']:.2f})", file=sys.stderr)
    if rec["anisotropy_flag"]:
        print("  ⚠ anisotropy: control distances on different axes disagree >1.5x (lens distortion?)", file=sys.stderr)
    if rec["weak_triangulation"]:
        print("  ⚠ weak triangulation on >=1 endpoint (near-parallel rays — use wider baseline)", file=sys.stderr)
    if rec["da3_scouts_disagree"]:
        print("  ⚠ DA3 scouts disagree (prior unreliable for this scene) — rely on the control distance", file=sys.stderr)


if __name__ == "__main__":
    main()
