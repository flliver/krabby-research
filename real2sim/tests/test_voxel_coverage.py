"""STO-SCN-103 — voxel-coverage view selector. Geometry + greedy unit tests."""
import importlib.util
import math
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "voxel_coverage.py"
_spec = importlib.util.spec_from_file_location("voxel_coverage", _MOD)
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def test_voxelize_counts_occupied():
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 10]], float)
    origin, vsize, occ, diag = vc.voxelize(pts, grid=10)
    assert vsize > 0 and len(occ) == 4            # four distinct corners -> four voxels
    assert math.isclose(diag, np.linalg.norm([10, 10, 10]))


def test_exposed_faces_isolated_voxel_has_six():
    # one isolated occupied voxel -> all 6 faces border empty space
    origin, vsize, occ = np.zeros(3), 1.0, {(0, 0, 0)}
    c, nrm = vc.exposed_faces(origin, vsize, occ)
    assert len(c) == 6
    # normals are the 6 axis directions, unit length
    assert {tuple(x) for x in nrm.tolist()} == {tuple(d) for d in vc._NEIGHBORS}


def test_exposed_faces_shared_face_excluded():
    # two adjacent voxels -> the shared face (between them) is interior on both sides
    origin, vsize, occ = np.zeros(3), 1.0, {(0, 0, 0), (1, 0, 0)}
    c, nrm = vc.exposed_faces(origin, vsize, occ)
    assert len(c) == 10                            # 12 total minus the 2 shared half-faces


def _cam(center, look, w=100, h=100, f=80.0):
    """Build a w2c looking from `center` toward `look` (OpenCV +Z fwd, simple up)."""
    center = np.asarray(center, float)
    fwd = np.asarray(look, float) - center
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])              # rows map world->cam axes (R_w2c)
    t = -R @ center
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
    return {"name": f"cam_{center}", "w2c": w2c,
            "intr": {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "w": w, "h": h}}


def test_flux_perpendicular_beats_grazing():
    # one face at origin, normal +Z. A camera straight on (+Z side, looking -Z) hits it
    # perpendicular (cos~1); a camera off to the side sees it grazing (lower cos).
    face_c = np.array([[0.0, 0.0, 0.0]])
    face_n = np.array([[0.0, 0.0, 1.0]])
    head_on = _cam([0, 0, 5], [0, 0, 0])
    oblique = _cam([4.5, 0, 2], [0, 0, 0])
    near, far = 0.01, 100.0
    w_head = vc.camera_weights(face_c, face_n, head_on["w2c"], head_on["intr"], near, far)[0]
    w_obl = vc.camera_weights(face_c, face_n, oblique["w2c"], oblique["intr"], near, far)[0]
    assert w_head > 0.95                           # near-perpendicular
    assert 0 < w_obl < w_head                      # grazing, but still sees it


def test_flux_behind_face_is_zero():
    # camera on the -Z side of a +Z-facing face cannot see it
    face_c = np.array([[0.0, 0.0, 0.0]])
    face_n = np.array([[0.0, 0.0, 1.0]])
    behind = _cam([0, 0, -5], [0, 0, 0])
    w = vc.camera_weights(face_c, face_n, behind["w2c"], behind["intr"], 0.01, 100.0)[0]
    assert w == 0.0


def test_greedy_prefers_new_angle_over_redundant():
    # 2 faces (a +Z face, a +X face). cam0 sees the +Z face head-on; cam1 duplicates cam0;
    # cam2 sees the +X face. Greedy's 2nd pick must be cam2 (new coverage), not cam1.
    face_c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    face_n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    cams = [_cam([0, 0, 5], [0, 0, 0]), _cam([0, 0, 5], [0, 0, 0]), _cam([5, 0, 0], [0, 0, 0])]
    W = vc.coverage_matrix(face_c, face_n, cams, 0.01, 100.0)
    order, cov, gains = vc.greedy_select(W, 2)
    assert order[0] == 0                           # tie (0,1 identical) -> first index
    assert order[1] == 2                           # complementary face, not the duplicate cam1


def test_greedy_deterministic_and_capped():
    face_c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    face_n = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    cams = [_cam([0, 0, 5], [0, 0, 0]), _cam([5, 0, 0], [0, 0, 0])]
    W = vc.coverage_matrix(face_c, face_n, cams, 0.01, 100.0)
    o1, _, _ = vc.greedy_select(W, 5)
    o2, _, _ = vc.greedy_select(W, 5)
    assert o1 == o2 and len(o1) <= 2
