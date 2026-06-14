"""STO-SCN-093 — solve-validity gate (planarity / nebula detector).

numpy-dependent (PCA), so skips where numpy is absent. Coplanar walk -> PASS;
spherical "nebula" -> FAIL.
"""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_MOD = Path(__file__).resolve().parents[1] / "validity_gate.py"
_spec = importlib.util.spec_from_file_location("validity_gate", _MOD)
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)


def test_coplanar_walk_passes():
    rng = np.random.default_rng(0)
    n = 200
    # a ground walk: meters of in-plane extent, centimeters of height noise
    x = rng.uniform(-5, 5, n)
    y = rng.uniform(-4, 4, n)
    z = rng.normal(0, 0.05, n)
    centers = np.stack([x, y, z], axis=1).tolist()
    r = vg.check_validity(centers)
    assert r["verdict"] == "PASS"
    assert r["out_in_ratio"] < 0.1            # clearly planar


def test_spherical_nebula_fails():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)   # points on a sphere
    r = vg.check_validity(v.tolist())
    assert r["verdict"] == "FAIL-nebula"
    assert r["out_in_ratio"] > 0.5            # near-isotropic


def test_ratio_monotone_with_height():
    rng = np.random.default_rng(2)
    n = 300
    flat = np.stack([rng.uniform(-5, 5, n), rng.uniform(-5, 5, n),
                     rng.normal(0, 0.02, n)], axis=1).tolist()
    tall = np.stack([rng.uniform(-5, 5, n), rng.uniform(-5, 5, n),
                     rng.normal(0, 3.0, n)], axis=1).tolist()
    assert vg.planarity(flat)["out_in_ratio"] < vg.planarity(tall)["out_in_ratio"]
