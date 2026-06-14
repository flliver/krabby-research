#!/usr/bin/env python3
"""STO-SCN-093 — solve-validity gate (planarity / nebula detector).

A cheap physical-prior sanity check on a pose solve BEFORE any downstream work
(STO-SCN-096 conclusion #5). A handheld ground walk produces camera centers that
are **near-coplanar** (lots of in-plane extent, little height variation). A
failed solve — e.g. the MASt3R "nebula" where a walk collapses into a
near-spherical camera blob — spreads centers in all three dimensions.

Metric: PCA on the camera centers → eigen-stddevs (σ_min ≤ σ_mid ≤ σ_max).
`out_in_ratio = σ_min / σ_max`. A flat walk ⇒ ~single-digit %; a nebula ⇒ tens
of %+. Fails loud above a threshold so a bad solve never reaches selection.

cv2-free / numpy-only (lazy import); reads camera centers via covis_graph's
binary reader. Fully unit-testable.

Usage:
  validity_gate.py <sparse_dir> [--threshold 0.30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import covis_graph as cg  # noqa: E402

# Default from evidence: a clean handheld walk lands well under this; the
# MASt3R 300-frame nebula scored ~0.6 (FAIL). 0.30 separates them with margin.
DEFAULT_THRESHOLD = 0.30


def planarity(centers: list) -> dict:
    """PCA on (N,3) camera centers -> eigen-stddevs + out-of-plane/in-plane ratio."""
    import numpy as np
    A = np.asarray(centers, dtype=float)
    A = A - A.mean(axis=0)
    cov = (A.T @ A) / max(1, len(A))
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)   # ascending
    sig = np.sqrt(eig)                                   # [σ_min, σ_mid, σ_max]
    ratio = float(sig[0] / sig[2]) if sig[2] > 0 else 0.0
    return {"eigen_stddev": [round(float(s), 4) for s in sig],
            "out_in_ratio": round(ratio, 4)}


def check_validity(centers: list, threshold: float = DEFAULT_THRESHOLD) -> dict:
    p = planarity(centers)
    p["n_cameras"] = len(centers)
    p["threshold"] = threshold
    p["verdict"] = "PASS" if p["out_in_ratio"] < threshold else "FAIL-nebula"
    return p


def check_sparse(sparse_dir, threshold: float = DEFAULT_THRESHOLD) -> dict:
    images = cg.read_images_bin(Path(sparse_dir) / "images.bin")
    centers = [img["center"] for img in images.values()]
    return check_validity(centers, threshold)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Solve-validity gate (planarity / nebula detector).")
    ap.add_argument("sparse_dir")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = ap.parse_args(argv)
    r = check_sparse(a.sparse_dir, a.threshold)
    print(f"validity: {r['verdict']}  out_in_ratio={r['out_in_ratio']} "
          f"(threshold {r['threshold']}) over {r['n_cameras']} cameras")
    print(f"  eigen-stddev (min/mid/max): {r['eigen_stddev']}")
    return 0 if r["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
