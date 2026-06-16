"""STO-SCN-145 — tests for the SDF boolean cull primitives.

Run: uv run --quiet --python 3.11 --with numpy --with pytest python3 -m pytest real2sim/tests/test_sdf_primitives.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sdf_primitives as sp  # noqa: E402


def test_sphere_inside_outside():
    s = sp.sdf_sphere(np.array([[0, 0, 0], [2, 0, 0]]), [0, 0, 0], 1.0)
    assert s[0] < 0 and s[1] > 0
    assert np.isclose(s[1], 1.0)


def test_box_sdf():
    s = sp.sdf_box(np.array([[0, 0, 0], [2, 0, 0]]), [0, 0, 0], [1, 1, 1])
    assert s[0] < 0                       # center inside
    assert np.isclose(s[1], 1.0)          # 1 m outside the +x face


def test_cylinder_radial_and_caps():
    # cylinder radius 1, height 2, along z, at origin
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2]], float)
    s = sp.sdf_cylinder(pts, [0, 0, 0], 1.0, 2.0, "z")
    assert s[0] < 0                       # inside
    assert s[1] > 0                       # outside radially
    assert s[2] > 0                       # above the top cap (z=2 > height/2=1)


def test_halfspace_below_is_inside():
    # plane at z=0, normal +z -> inside = below (z<0)
    s = sp.sdf_halfspace(np.array([[0, 0, -1], [0, 0, 1]]), [0, 0, 0], [0, 0, 1])
    assert s[0] < 0 and s[1] > 0


def test_keep_only_union():
    prims = [{"type": "sphere", "op": "keep", "center": [0, 0, 0], "radius": 1},
             {"type": "sphere", "op": "keep", "center": [3, 0, 0], "radius": 1}]
    pts = np.array([[0, 0, 0], [3, 0, 0], [1.5, 0, 0]], float)
    m = sp.cull_mask(pts, prims)
    assert list(m) == [True, True, False]   # in either sphere kept; the gap dropped


def test_subtract_carves_hole():
    prims = [{"type": "box", "op": "keep", "center": [0, 0, 0], "half": [5, 5, 5]},
             {"type": "sphere", "op": "subtract", "center": [0, 0, 0], "radius": 1}]
    pts = np.array([[0, 0, 0], [3, 0, 0]], float)
    m = sp.cull_mask(pts, prims)
    assert list(m) == [False, True]         # origin carved out, outside-sphere kept


def test_subtract_only_keeps_complement():
    # no keep prims -> keep all space except the subtract
    prims = [{"type": "sphere", "op": "subtract", "center": [0, 0, 0], "radius": 1}]
    pts = np.array([[0, 0, 0], [5, 0, 0]], float)
    m = sp.cull_mask(pts, prims)
    assert list(m) == [False, True]


def test_empty_keeps_all():
    pts = np.array([[0, 0, 0], [9, 9, 9]], float)
    assert sp.cull_mask(pts, []).all()


def test_frame_transform_applied():
    # primitive sphere at datum origin; verts in a frame shifted by +10 in x via transform
    prims = [{"type": "sphere", "op": "keep", "center": [0, 0, 0], "radius": 1}]
    T = np.eye(4); T[0, 3] = -10.0          # mesh->datum: subtract 10 in x
    pts = np.array([[10, 0, 0], [12, 0, 0]], float)   # 10 maps to datum origin (kept)
    m = sp.cull_mask(pts, prims, frame_transform=T)
    assert list(m) == [True, False]


def test_camera_aabb_equivalence_box():
    # STO-137 camera-AABB cull == a keep-box SDF: verts inside the box kept, outside dropped
    prims = [{"type": "box", "op": "keep", "center": [0, 0, 0], "half": [2, 2, 0.5]}]
    pts = np.array([[0, 0, 0], [0, 0, 1.0], [1, 1, 0.2]], float)
    m = sp.cull_mask(pts, prims)
    assert list(m) == [True, False, True]   # the thin-slab (z half=0.5) drops the z=1 point
