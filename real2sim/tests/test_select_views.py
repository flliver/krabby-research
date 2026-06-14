"""STO-SCN-094 — coverage-greedy view selection. Pure stdlib; runs everywhere."""
import importlib.util
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "select_views.py"
_spec = importlib.util.spec_from_file_location("select_views", _MOD)
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)


def test_angle_quality_tent():
    assert sv.angle_quality(1) == 0.0 and sv.angle_quality(60) == 0.0
    assert sv.angle_quality(20) == 1.0 and sv.angle_quality(15) == 1.0
    assert 0 < sv.angle_quality(6) < 1 and 0 < sv.angle_quality(45) < 1


def _synthetic():
    # cluster A (z~5) seen by imgs 0,1 (+ 2 partial); cluster B (z~105) by imgs 3,4.
    names = {i: f"img{i}" for i in range(5)}
    centers = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [-1, 0, 100], 4: [1, 0, 100]}
    pt_imgs, pt_xyz, img_pts = {}, {}, {i: set() for i in range(5)}
    for k in range(10):                       # cluster A
        seen = [0, 1, 2] if k < 5 else [0, 1]
        pt_imgs[k] = seen; pt_xyz[k] = [0.0, 0.0, 5.0]
        for i in seen:
            img_pts[i].add(k)
    for k in range(10, 20):                    # cluster B (disconnected from A)
        pt_imgs[k] = [3, 4]; pt_xyz[k] = [0.0, 0.0, 105.0]
        for i in (3, 4):
            img_pts[i].add(k)
    return centers, names, img_pts, pt_imgs, pt_xyz


def test_greedy_picks_coverage_and_triangulates():
    centers, names, img_pts, pt_imgs, pt_xyz = _synthetic()
    order = sv.select(centers, names, img_pts, pt_imgs, pt_xyz, n=2, min_overlap=3)
    assert set(order) == {0, 1}              # the overlapping A-cluster pair
    r = sv.report(order, names, img_pts, pt_imgs, pt_xyz, centers)
    assert r["triangulated_points"] == 10    # all of cluster A seen by both


def test_connectivity_excludes_disjoint_view():
    centers, names, img_pts, pt_imgs, pt_xyz = _synthetic()
    # ask for 3; img3/img4 share 0 points with the A-cluster -> connectivity blocks them
    order = sv.select(centers, names, img_pts, pt_imgs, pt_xyz, n=3, min_overlap=3)
    assert 3 not in order and 4 not in order
    assert set(order) <= {0, 1, 2}


def test_n_cap_and_determinism():
    centers, names, img_pts, pt_imgs, pt_xyz = _synthetic()
    o1 = sv.select(centers, names, img_pts, pt_imgs, pt_xyz, n=2, min_overlap=3)
    o2 = sv.select(centers, names, img_pts, pt_imgs, pt_xyz, n=2, min_overlap=3)
    assert o1 == o2 and len(o1) <= 2          # deterministic + respects N
