#!/usr/bin/env python3
"""STO-SCN-099 — cohesive fusion unit tests.

Synthetic 17-float 3DGS arrays (no real PLY needed for the math). The falsifiable bar
(story 'Test'): M registered submaps fuse into one cloud with NO doubled surface at the
overlap — the cross-fade halves each segment's opacity in the shared region so the two
contributions sum to single coverage, not double.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spine_fuse as sf  # noqa: E402


def make_gaussians(xs, opacity_logit=2.0):
    """Grid of gaussians along x (y=z=0), identity rotation, fixed opacity."""
    n = len(xs)
    a = np.zeros((n, sf.NPROP), np.float32)
    a[:, 0] = xs
    a[:, sf.OPA] = opacity_logit
    a[:, sf.SCL] = np.log(0.05)
    a[:, sf.IX["rot_0"]] = 1.0          # identity quat wxyz
    a[:, sf.IX["f_dc_0"]] = 0.5
    return a


# ----------------------------------------------------------------- DoD: no doubled overlap

def _two_overlapping_segments():
    seg0_x = np.arange(0, 21, dtype=float)      # 0..20
    seg1_x = np.arange(15, 36, dtype=float)     # 15..35  -> overlap 15..20
    return {
        0: {"gaussians": make_gaussians(seg0_x),
            "cameras": np.stack([seg0_x, np.zeros_like(seg0_x), np.zeros_like(seg0_x)], 1)},
        1: {"gaussians": make_gaussians(seg1_x),
            "cameras": np.stack([seg1_x, np.zeros_like(seg1_x), np.zeros_like(seg1_x)], 1)},
    }


def test_overlap_is_crossfaded_not_doubled():
    segs = _two_overlapping_segments()
    fused = sf.fuse(segs, radius=3.0)

    x = fused[:, 0]
    alpha = sf._sigmoid(fused[:, sf.OPA].astype(float))
    in_overlap = (x >= 15) & (x <= 20)
    interior0 = (x >= 2) & (x <= 13)            # only seg0 covers

    # a single segment's per-gaussian alpha in its interior (the reference density)
    single = alpha[interior0].mean()
    # summed alpha PER UNIT x in the overlap vs a single segment there
    overlap_sum = alpha[in_overlap].sum()
    n_units = 6                                  # x = 15..20 inclusive
    overlap_per_unit = overlap_sum / n_units
    # cross-fade => overlap density ~= single coverage (NOT 2x). naive concat would be ~2x.
    assert abs(overlap_per_unit - single) < 0.25 * single, (overlap_per_unit, single)


def test_coverage_weights_half_in_overlap_one_in_interior():
    segs = _two_overlapping_segments()
    cams = {k: segs[k]["cameras"] for k in segs}
    g0 = segs[0]["gaussians"]
    w0 = sf.coverage_weights(g0[:, sf.XYZ], cams, 0, radius=3.0)
    x = g0[:, 0]
    assert w0[(x >= 2) & (x <= 12)].mean() > 0.95              # interior untouched
    assert abs(w0[(x >= 16) & (x <= 19)].mean() - 0.5) < 0.1   # overlap halved


# ----------------------------------------------------------------- transform correctness

def _rand_gauge(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return {"scale": float(rng.uniform(0.3, 3.0)), "R": Q, "t": rng.standard_normal(3) * 4}


def test_transform_positions_and_scale():
    rng = np.random.default_rng(0)
    g = make_gaussians(np.arange(0, 10, dtype=float))
    G = _rand_gauge(rng)
    out = sf.transform_gaussians(g, G)
    exp_xyz = G["scale"] * (g[:, sf.XYZ].astype(float) @ np.asarray(G["R"]).T) + G["t"]
    assert np.allclose(out[:, sf.XYZ], exp_xyz, atol=1e-4)
    # log-scale shifted by log(scale)
    assert np.allclose(out[:, sf.SCL].astype(float),
                       g[:, sf.SCL].astype(float) + np.log(G["scale"]), atol=1e-5)


def test_quat_xyzw_to_R():
    assert np.allclose(sf.quat_xyzw_to_R([0, 0, 0, 1]), np.eye(3), atol=1e-9)   # identity
    # 90° about +Z: x->y, y->-x
    R = sf.quat_xyzw_to_R([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_compose_gauge_chains_105_then_098():
    """The orientation fix (operator-caught 2026-06-14): a DA3-normalized-frame gaussian
    must be carried gs->solve (105) THEN solve->global (098). compose_gauge must equal
    applying inner then outer."""
    rng = np.random.default_rng(11)
    inner = _rand_gauge(rng)        # gs -> segment solve (the 105 scout_gauge)
    outer = _rand_gauge(rng)        # segment solve -> global (the 098 gauge)
    comp = sf.compose_gauge(outer, inner)
    p = rng.standard_normal((20, 3))
    # composed applied once == inner applied, then outer
    via_comp = comp["scale"] * (p @ np.asarray(comp["R"]).T) + comp["t"]
    step1 = inner["scale"] * (p @ np.asarray(inner["R"]).T) + inner["t"]
    step2 = outer["scale"] * (step1 @ np.asarray(outer["R"]).T) + outer["t"]
    assert np.allclose(via_comp, step2, atol=1e-9)
    # and on gaussians: transform(g, comp) == transform(transform(g, inner), outer)
    g = make_gaussians(np.arange(0, 10, dtype=float))
    g[:, sf.ROT] = np.array([0.3, 0.4, 0.5, 0.7])
    one = sf.transform_gaussians(g, comp)
    two = sf.transform_gaussians(sf.transform_gaussians(g, inner), outer)
    assert np.allclose(one[:, sf.XYZ], two[:, sf.XYZ], atol=1e-3)
    assert np.allclose(one[:, sf.SCL], two[:, sf.SCL], atol=1e-4)


def test_transform_inverse_round_trips():
    rng = np.random.default_rng(1)
    g = make_gaussians(np.arange(0, 8, dtype=float))
    g[:, sf.ROT] = np.array([0.5, 0.5, 0.5, 0.5])     # a non-identity quat
    G = _rand_gauge(rng)
    s, R, t = G["scale"], np.asarray(G["R"]), G["t"]
    inv = {"scale": 1.0 / s, "R": R.T, "t": -(1.0 / s) * (R.T @ t)}
    back = sf.transform_gaussians(sf.transform_gaussians(g, G), inv)
    assert np.allclose(back[:, sf.XYZ], g[:, sf.XYZ], atol=1e-3)
    assert np.allclose(back[:, sf.SCL], g[:, sf.SCL], atol=1e-4)
    q0 = g[:, sf.ROT] / np.linalg.norm(g[:, sf.ROT], axis=1, keepdims=True)
    qb = back[:, sf.ROT] / np.linalg.norm(back[:, sf.ROT], axis=1, keepdims=True)
    assert np.allclose(qb, q0, atol=1e-3) or np.allclose(qb, -q0, atol=1e-3)


# ----------------------------------------------------------------- M=1 + io

def test_single_segment_passthrough():
    g = make_gaussians(np.arange(0, 12, dtype=float))
    out = sf.fuse({0: {"gaussians": g, "cameras": np.zeros((3, 3))}})
    assert np.array_equal(out, g)


def test_ply_round_trip():
    rng = np.random.default_rng(2)
    g = (rng.standard_normal((100, sf.NPROP)) * 2).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.ply"
        sf.write_ply(p, g)
        back = sf.read_ply(p)
    assert back.shape == g.shape
    assert np.array_equal(back, g)


def test_read_ply_rejects_wrong_prop_count():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.ply"
        p.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                      b"property float x\nproperty float y\nproperty float z\nend_header\n"
                      + np.zeros(3, "<f4").tobytes())
        try:
            sf.read_ply(p)
            assert False, "should reject non-17-prop ply"
        except ValueError:
            pass


def test_determinism():
    segs = _two_overlapping_segments()
    a = sf.fuse(segs, radius=3.0)
    b = sf.fuse(segs, radius=3.0)
    assert np.array_equal(a, b)


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
