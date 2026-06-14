"""STO-SCN-095 — sparse/0 -> posed.json shim. Pure stdlib; binary round-trip."""
import importlib.util
import struct
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "posed_from_sparse.py"
_spec = importlib.util.spec_from_file_location("posed_from_sparse", _MOD)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _write_cameras(path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        # camera_id=1, model_id=0 (SIMPLE_PINHOLE), w, h, params [f, cx, cy]
        f.write(struct.pack("<iiQQ", 1, 0, 3840, 2160))
        f.write(struct.pack("<ddd", 2000.0, 1920.0, 1080.0))


def _write_images(path, imgs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, q, t, cid, name in imgs:
            f.write(struct.pack("<idddddddi", iid, *q, *t, cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))


def test_posed_identity_pose(tmp_path):
    _write_cameras(tmp_path / "cameras.bin")
    _write_images(tmp_path / "images.bin",
                  [(1, (1, 0, 0, 0), (1, 2, 3), 1, "a.jpg")])  # identity rot, t=(1,2,3)
    posed = pf.posed_from_sparse(tmp_path)
    assert len(posed) == 1
    e = posed[0]
    assert e["name"] == "a.jpg"
    assert e["w2c"] == [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    assert e["K"] == [[2000.0, 0.0, 1920.0], [0.0, 2000.0, 1080.0], [0.0, 0.0, 1.0]]


def test_names_filter(tmp_path):
    _write_cameras(tmp_path / "cameras.bin")
    _write_images(tmp_path / "images.bin",
                  [(1, (1, 0, 0, 0), (0, 0, 0), 1, "a.jpg"),
                   (2, (1, 0, 0, 0), (0, 0, 0), 1, "b.jpg")])
    posed = pf.posed_from_sparse(tmp_path, names=["b.jpg"])
    assert [e["name"] for e in posed] == ["b.jpg"]
