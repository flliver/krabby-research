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


def apply_to_gauge(scene, solve, scale):  # pragma: no cover - store mutation, operator-gated (T-020)
    """SEAM (not yet wired): emit a calibrated gauge node = the solve gauge x `scale`, additive
    (new identity), so cameras + every downstream mesh inherit meters. Gated on a real operator
    measurement to test e2e per T-020; left as the documented next step under STO-SCN-016."""
    raise NotImplementedError(
        "apply_to_gauge is the documented next step (STO-SCN-016) — wire after operator T-020 "
        "supplies a real control measurement; must be an additive calibrated gauge node, not a re-key."
    )


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
