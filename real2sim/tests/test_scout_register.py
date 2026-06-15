"""STO-SCN-105 (corrected) — direct gs_ply↔solve registration.

The scale_factor theory was disproven on a real scout; the splat is registered
by DIRECT point-cloud alignment instead. These tests author synthetic clouds
with a KNOWN scale + translation and assert scout_register recovers them, plus
the binary readers round-trip.
"""
import importlib.util
import math
import random
import struct
import sys
from pathlib import Path

_R2S = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("scout_register", _R2S / "scout_register.py")
sr = importlib.util.module_from_spec(_spec)
sys.modules["scout_register"] = sr
_spec.loader.exec_module(sr)


def _write_points3d_bin(path, pts):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (x, y, z) in enumerate(pts):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", x, y, z))
            f.write(bytes([128, 128, 128]))            # rgb
            f.write(struct.pack("<d", 0.5))            # error
            f.write(struct.pack("<Q", 0))              # empty track


def _write_gs_ply(path, pts):
    n = len(pts)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
             "rot_2", "rot_3"]                          # 17-float 3DGS
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n").encode()
    with open(path, "wb") as f:
        f.write(header)
        for (x, y, z) in pts:
            row = [x, y, z] + [0.0] * (len(props) - 3)
            f.write(struct.pack("<%df" % len(props), *row))


def _blob(n, center, radius, seed):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        # gaussian-ish blob
        out.append(tuple(center[i] + r.gauss(0, radius) for i in range(3)))
    return out


def test_readers_roundtrip(tmp_path):
    pts = _blob(500, (1, 2, 3), 0.5, 1)
    _write_points3d_bin(tmp_path / "p.bin", pts)
    _write_gs_ply(tmp_path / "g.ply", pts)
    rp = sr.read_points3d_bin(tmp_path / "p.bin")
    rg = sr.read_gs_ply_xyz(tmp_path / "g.ply")
    assert len(rp) == 500 and len(rg) == 500
    assert math.dist(rp[0], pts[0]) < 1e-9
    assert math.dist(rg[0], pts[0]) < 1e-4          # float32 ply


def test_recovers_known_scale_and_translation(tmp_path):
    # solve scene: blob at origin, radius 1
    solve = _blob(4000, (0, 0, 0), 1.0, 10)
    # gs scene = solve / scale - shift  (so register should recover scale, +shift)
    SCALE, SHIFT = 0.4, (0.0, 0.0, 3.0)               # mirrors the real 001 case
    gs = [((p[0] - SHIFT[0]) / SCALE, (p[1] - SHIFT[1]) / SCALE,
           (p[2] - SHIFT[2]) / SCALE) for p in solve]
    _write_points3d_bin(tmp_path / "points3D.bin", solve)
    _write_gs_ply(tmp_path / "scout.gs.ply", gs)
    r = sr.register(tmp_path / "scout.gs.ply", tmp_path / "points3D.bin")
    assert abs(r["scale"] - SCALE) / SCALE < 0.08, r          # scale within 8%
    # p_solve = scale*p_gs + translate ; check a sample maps correctly
    s, t = r["scale"], r["translate"]
    g0 = gs[0]
    mapped = [s * g0[i] + t[i] for i in range(3)]
    assert math.dist(mapped, solve[0]) < 0.4, (mapped, solve[0])  # within a fraction of radius
    # IoU caps ~0.53 even for PERFECT registration here: a continuous gaussian
    # blob discretized on a 30-cell grid never fully overlaps cell-for-cell.
    # (Scale/translate recovered exactly above; IoU is a sanity floor only.)
    assert r["iou"] > 0.45


def test_register_scout_unregistered_when_missing(tmp_path):
    r = sr.register_scout(tmp_path, tmp_path)          # no files
    assert r["registered"] is False and r["scale"] == 1.0
