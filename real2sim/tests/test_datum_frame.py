"""STO-SCN-145 — tests for the camera-derived metric datum (gauge-fixing).

Run: uv run --quiet --python 3.11 --with numpy --with pytest python3 -m pytest real2sim/tests/test_datum_frame.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import datum_frame as df  # noqa: E402


# a simple straight camera walk along +X at height 2 along world +Z
CAMS = np.array([[0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2]], float)
UP = np.array([0, 0, 1.0])


def test_up_maps_to_plus_z():
    d = df.build_datum(CAMS, UP)
    R = np.array(d["R"])
    assert np.allclose(R @ UP, [0, 0, 1], atol=1e-9)


def test_spine_maps_to_plus_x():
    d = df.build_datum(CAMS, UP)
    R = np.array(d["R"])
    # the spine tangent (+X world) should map to datum +X
    assert np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-9)


def test_right_handed_orthonormal():
    d = df.build_datum(CAMS, UP)
    R = np.array(d["R"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_origin_at_centroid_by_default():
    d = df.build_datum(CAMS, UP)
    # the camera centroid maps to the datum origin
    assert np.allclose(df.to_datum(CAMS.mean(0), d), [0, 0, 0], atol=1e-9)


def test_ground_projection_sets_height():
    # floor at world z=0; origin should sit on the ground (datum z of a ground point = 0)
    d = df.build_datum(CAMS, UP, ground_z=0.0)
    origin = np.array(d["origin"])
    assert np.isclose(origin[2], 0.0, atol=1e-9)          # origin on the floor
    # a point on the floor under the centroid -> datum height 0
    c = CAMS.mean(0); ground_pt = np.array([c[0], c[1], 0.0])
    assert np.isclose(df.to_datum(ground_pt, d)[2], 0.0, atol=1e-9)
    # the cameras (height 2) are at datum +Z = 2
    assert np.allclose(df.to_datum(CAMS, d)[:, 2], 2.0, atol=1e-9)


def test_metric_scale_applies():
    # scale=2 -> a 1-unit solve distance becomes 2 datum units (meters)
    d = df.build_datum(CAMS, UP, scale=2.0)
    p0 = df.to_datum([0, 0, 2], d)
    p1 = df.to_datum([1, 0, 2], d)
    assert np.isclose(np.linalg.norm(p1 - p0), 2.0, atol=1e-9)


def test_loop_fallback_uses_pca():
    # start == end horizontally (a loop) -> spine_azimuth falls back to PCA, still unit + in-plane
    loop = np.array([[0, 0, 1], [1, 1, 1], [2, 0, 1], [1, -1, 1], [0, 0, 1]], float)
    e0 = df.spine_azimuth(loop, UP)
    assert np.isclose(np.linalg.norm(e0), 1.0, atol=1e-9)
    assert np.isclose(e0 @ UP, 0.0, atol=1e-9)            # lies in the ground plane


def test_from_poses_recovers_up():
    # synthesize low-roll w2c (camera right ~ horizontal) -> gauge_up should give ~+Z up
    import gauge_up
    np_rng = np.array
    w2c = []
    for k, yaw in enumerate(np.linspace(-0.5, 0.5, 6)):
        # camera looking roughly -Z-ish with small yaw, right axis ~ horizontal
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # rotate about world up (y) ... build w2c
        # we want right (row0) horizontal w.r.t world +Z: construct so up recovers +Z
        Rw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])     # yaw about +Z -> right stays horizontal
        M = np.eye(4); M[:3, :3] = Rw
        w2c.append(M.tolist())
    up = np.array(gauge_up.up_from_poses(w2c))
    assert np.isclose(abs(up[2]), 1.0, atol=1e-6)            # gravity ~ world +Z
