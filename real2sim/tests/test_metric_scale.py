"""STO-SCN-144 unit tests — metric-scale recovery core (GOAL-SCN-001).

Covers the story's testing DoD: two-ray closest-approach triangulation, near-parallel weak
warning, scale recovery from a planted distance, log-median aggregation rejecting the outlier,
and the DA3 gross-error gate (trips on 3x, passes within 1.5x).

Run: uv run --quiet --python 3.11 --with numpy --with pytest python3 -m pytest real2sim/tests/test_metric_scale.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import metric_scale as ms  # noqa: E402


def test_triangulate_intersecting_rays_zero_gap():
    # ray1 along +x from origin; ray2 along -y from (2,2,0) -> intersect exactly at (2,0,0)
    p, gap, par = ms.triangulate_rays([0, 0, 0], [1, 0, 0], [2, 2, 0], [0, -1, 0])
    assert np.allclose(p, [2, 0, 0], atol=1e-9)
    assert gap < 1e-9
    assert par == pytest.approx(90.0, abs=1e-6)


def test_triangulate_skew_rays_midpoint_and_gap():
    # ray1 = x-axis through origin; ray2 = y-axis through (0,0,1). Closest pts (0,0,0)&(0,0,1).
    p, gap, par = ms.triangulate_rays([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0])
    assert np.allclose(p, [0, 0, 0.5], atol=1e-9)
    assert gap == pytest.approx(1.0, abs=1e-9)


def test_near_parallel_rays_flagged_by_small_parallax():
    # two nearly-parallel rays -> tiny parallax angle (weak triangulation signal)
    _, _, par = ms.triangulate_rays([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1e-4])
    assert par < 1.0   # caller warns below a parallax threshold


def test_pixel_ray_principal_axis():
    # clicking the principal point looks straight down the camera +z (identity pose)
    c2w = np.eye(4)
    K = np.array([[500, 0, 512], [0, 500, 288], [0, 0, 1]], float)
    o, d = ms.pixel_ray(c2w, K, 512, 288)
    assert np.allclose(o, [0, 0, 0])
    assert np.allclose(d, [0, 0, 1], atol=1e-9)


def test_scale_from_distance_recovers_planted_scale():
    # two points 2 solve-units apart that are really 1.0 m apart -> s = 0.5 m / unit
    s, d_solve = ms.scale_from_distance([0, 0, 0], [2, 0, 0], 1.0)
    assert d_solve == pytest.approx(2.0)
    assert s == pytest.approx(0.5)


def test_scale_degenerate_raises():
    with pytest.raises(ValueError):
        ms.scale_from_distance([1, 1, 1], [1, 1, 1], 1.0)


def test_aggregate_log_median_rejects_outlier():
    # the 11.22 scout outlier must die under the log-median; spread exposes it
    med, spread = ms.aggregate_scales([3.346, 3.481, 11.220])
    assert med == pytest.approx(3.481, rel=1e-6)     # middle value in log space
    assert spread == pytest.approx(11.220 / 3.346, rel=1e-6)


def test_aggregate_single_scale_unit_spread():
    med, spread = ms.aggregate_scales([0.42])
    assert med == pytest.approx(0.42)
    assert spread == pytest.approx(1.0)


def test_da3_gate_passes_within_tolerance():
    passed, ratio, med = ms.da3_gate(0.50, [0.50, 0.52, 0.49])
    assert passed is True
    assert ratio < 1.5


def test_da3_gate_trips_on_gross_discrepancy():
    # control says 0.5, monocular median ~1.6 -> 3.2x -> flag for human review
    passed, ratio, med = ms.da3_gate(0.50, [1.6, 1.55])
    assert passed is False
    assert ratio == pytest.approx(med / 0.50, rel=1e-6)
    assert ratio > 1.5


def test_da3_gate_inert_without_prior():
    passed, ratio, med = ms.da3_gate(0.50, [])
    assert passed is None and ratio is None and med is None


def test_full_pipeline_two_cameras_recovers_metric():
    # synthetic: two cameras viewing two real points 1.5 m apart in solve units, recover s.
    # cameras at (-1,0,0) and (1,0,0) both looking +z toward a wall at z=4.
    def cam(cx):
        c2w = np.eye(4); c2w[:3, 3] = [cx, 0, 0]
        return c2w
    K = np.array([[600, 0, 512], [0, 600, 288], [0, 0, 1]], float)
    # two world points (solve gauge) 1.5 units apart on x at z=4
    Pa, Pb = np.array([0.0, 0.0, 4.0]), np.array([1.5, 0.0, 4.0])

    def project(c2w, P):
        w2c = np.linalg.inv(c2w)
        pc = w2c[:3, :3] @ P + w2c[:3, 3]
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        return u, v

    cL, cR = cam(-1.0), cam(1.0)
    # triangulate Pa from L,R ; Pb from L,R
    oa1, da1 = ms.pixel_ray(cL, K, *project(cL, Pa))
    oa2, da2 = ms.pixel_ray(cR, K, *project(cR, Pa))
    Pa_hat, ga, _ = ms.triangulate_rays(oa1, da1, oa2, da2)
    ob1, db1 = ms.pixel_ray(cL, K, *project(cL, Pb))
    ob2, db2 = ms.pixel_ray(cR, K, *project(cR, Pb))
    Pb_hat, gb, _ = ms.triangulate_rays(ob1, db1, ob2, db2)
    assert ga < 1e-6 and gb < 1e-6
    assert np.allclose(Pa_hat, Pa, atol=1e-6)
    assert np.allclose(Pb_hat, Pb, atol=1e-6)
    # the real distance is 3.0 m (the 1.5 solve-units are really 3.0 m) -> s = 2.0
    s, d_solve = ms.scale_from_distance(Pa_hat, Pb_hat, 3.0)
    assert d_solve == pytest.approx(1.5, abs=1e-6)
    assert s == pytest.approx(2.0, abs=1e-6)
