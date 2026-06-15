#!/usr/bin/env python3
"""STO-SCN-098 — global registration unit tests.

Builds a known camera track, splits it into M OVERLAPPING segments, applies a DIFFERENT
random similarity to each (simulating independent per-segment SfM gauges), and verifies the
pose graph brings them back into ONE gauge. The falsifiable bar (story 'Test'): per-seam
residual < tol on a consistent scene; a deliberately warped segment trips the residual gate;
a loop closure pulls a revisit together.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spine_register as sr  # noqa: E402


def rand_rot(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rand_sim(rng, smin=0.4, smax=2.5):
    return rng.uniform(smin, smax), rand_rot(rng), rng.standard_normal(3) * 5.0


def gt_track(n=60, rng=None, loop=False):
    """A smooth 3D camera path + per-camera orientations. If loop, the last few
    cameras spatially COINCIDE with the first few (a revisit)."""
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 1, n)
    centers = np.stack([np.sin(2 * np.pi * t) * 3, t * 6, np.cos(2 * np.pi * t) * 3], 1)
    rots = np.stack([rand_rot(np.random.default_rng(i)) for i in range(n)])
    if loop:
        for m in range(3):                       # cam (n-6+m) revisits cam m
            centers[n - 6 + m] = centers[m]
            rots[n - 6 + m] = rots[m]
    return centers, rots


def make_segments(centers, rots, ranges, rng, noise=0.0):
    """Apply a distinct random similarity (+ optional noise) to each index range
    -> nodes {seg: {names, centers, rots}} in independent local gauges."""
    nodes = {}
    for k, (lo, hi) in enumerate(ranges):
        idx = list(range(lo, hi))
        s, R, t = rand_sim(rng)
        c = s * (centers[idx] @ R.T) + t
        if noise:
            c = c + rng.standard_normal(c.shape) * noise
        rr = np.einsum("ij,njk->nik", R, rots[idx])
        nodes[k] = {"names": [f"cam{i}" for i in idx],
                    "centers": c, "rots": rr}
    return nodes


def overlapping_ranges(n, seg, ov):
    stride = seg - ov
    starts = list(range(0, n - ov, stride))
    return [(s, min(s + seg, n)) for s in starts]


# ----------------------------------------------------------------- DoD: recovery

def test_m_submaps_register_to_one_gauge():
    rng = np.random.default_rng(42)
    centers, rots = gt_track(60, rng)
    ranges = overlapping_ranges(60, 20, 8)
    nodes = make_segments(centers, rots, ranges, rng)
    assert len(ranges) >= 3, ranges

    out = sr.register(nodes, rel_tol=0.01)
    assert out["converged"], out["iters_run"]
    # seams agree to numerical precision (the cohesion invariant)
    assert out["max_seam_residual_rel"] < 1e-6, out["max_seam_residual_rel"]
    assert out["within_tol"]
    # every camera in ONE consistent gauge: recovered globals are a single similarity of GT
    names = [f"cam{i}" for i in range(60)]
    G = np.array([out["cameras"][nm]["center"] for nm in names])
    s, R, t = sr.ga.umeyama(centers, G)
    fit = sr.ga.residuals(centers, G, s, R, t)
    spread = np.linalg.norm(G - G.mean(0), axis=1).mean()
    assert fit.max() / spread < 1e-6, fit.max() / spread


def test_overlap_cameras_coincide_across_segments():
    rng = np.random.default_rng(7)
    centers, rots = gt_track(50, rng)
    ranges = overlapping_ranges(50, 18, 7)
    nodes = make_segments(centers, rots, ranges, rng)
    out = sr.register(nodes, rel_tol=0.01)
    # a camera shared by two segments resolves to a single global point
    for seam in out["seams"]:
        assert seam["residual_max"] < 1e-7, seam


# ----------------------------------------------------------------- DoD: drift gate (T-001)

def test_anisotropic_warp_trips_the_gate():
    """A segment warped by a NON-similarity (anisotropic stretch) cannot be aligned
    by any gauge -> its seams blow the residual gate. The falsifiable catch."""
    rng = np.random.default_rng(1)
    centers, rots = gt_track(60, rng)
    ranges = overlapping_ranges(60, 20, 8)

    good = make_segments(centers, rots, ranges, np.random.default_rng(1))
    out_good = sr.register(good, rel_tol=0.02)
    assert out_good["within_tol"], "control must pass"

    bad = make_segments(centers, rots, ranges, np.random.default_rng(1))
    bad[2]["centers"] = bad[2]["centers"] * np.array([1.6, 1.0, 1.0])   # anisotropic
    out_bad = sr.register(bad, rel_tol=0.02)
    assert not out_bad["within_tol"], out_bad["max_seam_residual_rel"]
    worst = max(out_bad["seams"], key=lambda e: e["residual_max"])
    assert 2 in (worst["i"], worst["j"]), worst       # the warped segment owns the worst seam


def test_sparse_outliers_surface_but_dont_fail_a_good_registration():
    """Real lesson (001-patio cross-solve, 2026-06-14): a minority of badly-solved
    boundary frames must NOT fail an otherwise-good registration — they're trimmed by
    consensus, surfaced as n_outlier, and the gate still passes. (Unlike the systematic
    warp above, which leaves too few in consensus and DOES fail.)"""
    rng = np.random.default_rng(2)
    centers, rots = gt_track(80, rng)
    ranges = overlapping_ranges(80, 24, 12)          # 12-frame seams
    nodes = make_segments(centers, rots, ranges, np.random.default_rng(2))
    # corrupt a clear MINORITY (3 of seg1's 12 seg0-overlap frames = 25%) by a MODERATE
    # amount (~40% of the segment's local spread) — matching the real 001-patio case
    # (28% of boundary frames off by ≤20% of spread, still registered). seg1's first 12
    # cams overlap seg0; gross outliers are a separate, harder problem not seen on real data.
    c = nodes[1]["centers"].copy()
    spread1 = np.linalg.norm(c - c.mean(0), axis=1).mean()
    for i, sign in zip((2, 5, 9), (1, -1, 1)):
        c[i] += sign * 0.4 * spread1 * np.array([1.0, -0.5, 0.7])
    nodes[1]["centers"] = c
    out = sr.register(nodes, rel_tol=0.02, min_consensus_frac=0.5)
    assert out["within_tol"], (out["max_seam_residual_rel"],
                               [(s["i"], s["j"], s["consensus_frac"]) for s in out["seams"]])
    assert any(s["n_outlier"] > 0 for s in out["seams"]), "outliers should be surfaced"


def test_no_overlap_refuses_to_chain():
    rng = np.random.default_rng(3)
    centers, rots = gt_track(40, rng)
    nodes = make_segments(centers, rots, [(0, 15), (20, 35)], rng)   # disjoint, no shared names
    try:
        sr.register(nodes)
        assert False, "should refuse with no edges"
    except RuntimeError:
        pass


# ----------------------------------------------------------------- DoD: loop closure

def test_loop_closure_pulls_revisit_together():
    rng = np.random.default_rng(5)
    centers, rots = gt_track(60, rng, loop=True)
    ranges = overlapping_ranges(60, 20, 8)
    nseg = len(ranges)
    # revisit: cam(54+m) (last seg) coincides with cam m (seg0). Build the loop edge.
    first, last = 0, nseg - 1
    f_names = nodes_names = None
    nodes = make_segments(centers, rots, ranges, rng, noise=0.03)

    def local_idx(seg, cam):
        return nodes[seg]["names"].index(cam)
    loop = [{"i": first, "j": last,
             "i_idx": [local_idx(first, f"cam{m}") for m in range(3)],
             "j_idx": [local_idx(last, f"cam{54 + m}") for m in range(3)]}]

    base = sr.register(nodes, rel_tol=0.05)                 # chain only
    closed = sr.register(nodes, loops=loop, rel_tol=0.05)   # + loop closure

    # distance between the revisit pairs in the global frame
    def revisit_gap(out):
        gaps = []
        for m in range(3):
            a = np.array(out["cameras"][f"cam{m}"]["center"])
            b = np.array(out["cameras"][f"cam{54 + m}"]["center"])
            gaps.append(np.linalg.norm(a - b))
        return float(np.mean(gaps))

    assert any(s["type"] == "loop" for s in closed["seams"])
    assert revisit_gap(closed) < revisit_gap(base), (revisit_gap(closed), revisit_gap(base))
    assert closed["converged"], closed["iters_run"]


# ----------------------------------------------------------------- determinism

def test_determinism():
    rng = np.random.default_rng(9)
    centers, rots = gt_track(45, rng)
    ranges = overlapping_ranges(45, 16, 6)
    nodes = make_segments(centers, rots, ranges, np.random.default_rng(9))
    a = sr.register(nodes)
    b = sr.register(nodes)
    assert a["gauges"] == b["gauges"]
    assert a["max_seam_residual"] == b["max_seam_residual"]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
