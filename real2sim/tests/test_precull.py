"""STO-SCN-092 — pose-free pre-cull engine (real2sim/precull_frames.py).

Synthetic pool exercises the four behaviors: near-dup run collapse, blur
rejection, REVISIT preservation (distant look-alikes survive), and the gap
guard. Skips where numpy/PIL aren't present.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_MOD = Path(__file__).resolve().parents[1] / "precull_frames.py"
_spec = importlib.util.spec_from_file_location("precull_frames", _MOD)
pc = importlib.util.module_from_spec(_spec)
sys.modules["precull_frames"] = pc   # dataclass + future-annotations needs this
_spec.loader.exec_module(pc)


def _noise(seed, h=64, w=64):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _smooth(h=64, w=64):
    g = np.linspace(0, 30, w, dtype=np.float64)
    return np.repeat(np.tile(g, (h, 1))[..., None], 3, axis=2).astype(np.uint8)


def _build_pool(dirp):
    """35 frames: A(8 near-dup sharp) | blur(5) | B(8) | A-revisit(6) | C(8)."""
    items = []
    idx = 0

    def add(arr):
        nonlocal idx
        name = f"{idx:04d}.png"
        p = dirp / name
        Image.fromarray(arr).save(p)
        items.append((name, p))
        idx += 1

    baseA = _noise(1)
    for _ in range(8):
        a = baseA.copy(); a[0, 0, 0] = (int(a[0, 0, 0]) + 1) % 256; add(a)
    for _ in range(5):
        add(_smooth())
    baseB = _noise(2)
    for _ in range(8):
        b = baseB.copy(); b[1, 1, 0] = (int(b[1, 1, 0]) + 1) % 256; add(b)
    for _ in range(6):  # revisit of A, temporally distant
        a = baseA.copy(); a[2, 2, 0] = (int(a[2, 2, 0]) + 1) % 256; add(a)
    baseC = _noise(3)
    for _ in range(8):
        c = baseC.copy(); c[3, 3, 0] = (int(c[3, 3, 0]) + 1) % 256; add(c)
    return items


def test_precull_culls_and_collapses_runs(tmp_path):
    items = _build_pool(tmp_path)
    res = pc.precull(items)
    r = res.report
    assert r["source_pool_n"] == 35
    assert 3 <= r["kept_n"] <= 8           # five segments, blur dropped
    assert r["dropped_near_dup"] >= 25     # the bulk were consecutive near-dups


def test_blur_frames_dropped(tmp_path):
    items = _build_pool(tmp_path)
    res = pc.precull(items)
    blur_idx = {f"{i:04d}.png" for i in range(8, 13)}
    assert not (set(res.kept) & blur_idx)


def test_revisit_is_preserved(tmp_path):
    # identical content appears early (idx 0-7) and again distant (idx 21-26);
    # global dedup would keep one — local dedup must keep both.
    items = _build_pool(tmp_path)
    res = pc.precull(items)
    kept_idx = [int(n[:4]) for n in res.kept]
    assert any(0 <= k <= 7 for k in kept_idx)
    assert any(21 <= k <= 26 for k in kept_idx)


def test_small_pool_not_thinned(tmp_path):
    items = _build_pool(tmp_path)
    res = pc.precull(items, target=300)
    assert res.report["thinned_to_target"] == 0   # conclusion #1: small pool -> use all


def test_gap_guard_bounds_spacing():
    frames = [pc.Frame(id=str(k), path=Path("."), idx=k, sharp=float(k % 7))
              for k in range(60)]
    kept, inserted = pc._gap_guard(frames, [0, 50], set(), max_gap=20)
    assert inserted >= 2
    gaps = [b - a for a, b in zip(kept, kept[1:])]
    assert max(gaps) <= 20
