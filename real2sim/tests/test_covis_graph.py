"""STO-SCN-093 — covis_graph: binary readers + co-visibility computation.

Pure stdlib (struct/math), so runs everywhere. Includes a COLMAP-binary
round-trip to lock the struct formats (a wrong format = garbage covis).
"""
import importlib.util
import struct
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "covis_graph.py"
_spec = importlib.util.spec_from_file_location("covis_graph", _MOD)
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)


def test_qvec2rotmat_identity():
    R = cg.qvec2rotmat(1, 0, 0, 0)
    assert R == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_angle_deg_right_angle():
    # cameras at (-1,0,0),(1,0,0), point at (0,0,1) -> 90 deg at the point
    a = cg._angle_deg([-1, 0, 0], [1, 0, 0], [0, 0, 1])
    assert abs(a - 90.0) < 1e-6


def _write_images_bin(path, imgs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, q, t, cid, name in imgs:
            f.write(struct.pack("<idddddddi", iid, *q, *t, cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D


def _write_points3D_bin(path, pts):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz, track in pts:
            f.write(struct.pack("<QdddBBBd", pid, *xyz, 0, 0, 0, 0.5))
            f.write(struct.pack("<Q", len(track)))
            for img, p2d in track:
                f.write(struct.pack("<ii", img, p2d))


def test_readers_roundtrip(tmp_path):
    imgs = [(1, (1, 0, 0, 0), (0, 0, 0), 1, "a.jpg"),
            (2, (1, 0, 0, 0), (-1, 0, 0), 1, "b.jpg")]   # tvec (0,0,0)->C0; (-1,0,0)->C(1,0,0)
    _write_images_bin(tmp_path / "images.bin", imgs)
    _write_points3D_bin(tmp_path / "points3D.bin",
                        [(10, (0.0, 0.0, 5.0), [(1, 0), (2, 0)])])
    I = cg.read_images_bin(tmp_path / "images.bin")
    P = cg.read_points3D_bin(tmp_path / "points3D.bin")
    assert I[1]["name"] == "a.jpg" and I[2]["name"] == "b.jpg"
    assert I[1]["center"] == [0.0, 0.0, 0.0]
    assert [round(v, 3) for v in I[2]["center"]] == [1.0, 0.0, 0.0]   # C=-R^T t
    assert P[0]["xyz"] == [0.0, 0.0, 5.0]
    assert sorted(P[0]["image_ids"]) == [1, 2]


def test_build_covis_connectivity():
    images = {1: {"name": "a", "center": [0, 0, 0]},
              2: {"name": "b", "center": [1, 0, 0]},
              3: {"name": "c", "center": [9, 9, 9]}}   # c shares nothing
    # 20 points seen by images 1 & 2 (well above min_overlap=15), none by 3
    points = [{"xyz": [0.0, 0.0, 5.0 + k], "image_ids": [1, 2]} for k in range(20)]
    g = cg.build_covis(images, points, min_overlap=15)
    assert g["n_images"] == 3
    assert g["coverage"]["a"] == 20 and g["coverage"]["c"] == 0
    assert g["n_isolated"] == 1 and "c" in g["isolated_images"]
    assert not g["connected"]            # c is its own component
    assert g["largest_component"] == 2   # a+b joined (shared 20 >= 15)
    pair = next(p for p in g["pairs"] if p[0] == "a" and p[1] == "b")
    assert pair[2] == 20                 # shared count
