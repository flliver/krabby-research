"""STO-SCN-092 — shared dependency-free pHash (real2sim/phash.py).

Skips where numpy/PIL aren't available (e.g. the numpy-less system interpreter);
runs under any env that has them.
"""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

_MOD = Path(__file__).resolve().parents[1] / "phash.py"
_spec = importlib.util.spec_from_file_location("phash", _MOD)
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)


def _noise(seed, h=64, w=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_identical_is_zero_distance():
    a = _noise(1)
    assert ph.hamming(ph.phash(a), ph.phash(a)) == 0


def test_tiny_tweak_is_near_duplicate():
    a = _noise(1)
    b = a.copy()
    b[0, 0, 0] = (int(b[0, 0, 0]) + 1) % 256
    assert ph.hamming(ph.phash(a), ph.phash(b)) <= 2


def test_unrelated_is_far():
    a, c = _noise(1), _noise(2)
    assert ph.hamming(ph.phash(a), ph.phash(c)) > 8


def test_hash_is_64_bit_and_symmetric():
    a, c = _noise(3), _noise(4)
    ha, hc = ph.phash(a), ph.phash(c)
    assert 0 <= int(ha) < (1 << 64)
    assert ph.hamming(ha, hc) == ph.hamming(hc, ha)
