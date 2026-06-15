#!/usr/bin/env python3
"""STO-SCN-097 — spine segmentation unit tests.

Operates on synthetic pHash sequences (no images): a coherent "walk" is a Gray-code
sequence (consecutive frames differ by exactly 1 bit -> registrable seams), and a
revisit is a late frame whose hash equals an early frame's (-> a loop candidate).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spine_segment as ss  # noqa: E402


def gray(i: int) -> int:
    """Gray code: consecutive values differ by exactly one bit."""
    return i ^ (i >> 1)


def walk(n: int) -> list[int]:
    return [gray(i) for i in range(n)]


def ids_for(n: int) -> list[str]:
    return [f"f{i:04d}" for i in range(n)]


# ----------------------------------------------------------------- DoD bar

def test_multisegment_walk_within_capacity_overlap_and_loop():
    """The falsifiable bar (story 'Test'): a multi-segment walk splits into M
    windows each within capacity; every adjacent pair clears the overlap
    threshold; at least one path-revisit is flagged."""
    n, cap, overlap = 50, 20, 5
    ids, h = ids_for(n), walk(n)
    h[45] = gray(2)                      # revisit: frame 45 looks like frame 2
    spec = ss.segment(ids, h, cap=cap, overlap=overlap, snap=0, loop_min_sep=2, loop_step=1)

    assert spec["n_segments"] >= 2, "should split into multiple segments"
    # every segment within solver capacity
    assert spec["within_capacity"]
    assert all(s["n_frames"] <= cap for s in spec["segments"])
    # every adjacent pair shares >= the overlap budget AND is registrable
    assert spec["seams"], "multi-segment => seams exist"
    for seam in spec["seams"]:
        assert seam["n_overlap"] >= overlap, seam
        assert seam["registrable"], seam
    assert spec["all_seams_registrable"]
    # the revisit is flagged across non-adjacent segments
    assert spec["n_loop_candidates"] >= 1
    lc = spec["loop_candidates"][0]
    assert lc["seg_b"] - lc["seg_a"] >= 2
    assert lc["hamming"] == 0
    assert {lc["idx_a"], lc["idx_b"]} == {2, 45}


# ----------------------------------------------------------------- overlap budget

def test_overlap_budget_guaranteed_by_construction():
    n, cap, overlap = 100, 25, 8
    ids, h = ids_for(n), walk(n)
    spec = ss.segment(ids, h, cap=cap, overlap=overlap, snap=0)
    for seam in spec["seams"]:
        assert seam["n_overlap"] >= overlap


def test_snapping_never_drops_overlap_below_budget():
    """Snapping only ever moves a cut EARLIER, so overlap can grow but never
    fall below the budget."""
    n, cap, overlap, snap = 120, 30, 6, 8
    ids, h = ids_for(n), walk(n)
    spec = ss.segment(ids, h, cap=cap, overlap=overlap, snap=snap)
    for seam in spec["seams"]:
        assert seam["n_overlap"] >= overlap, seam
    assert spec["within_capacity"]


# ----------------------------------------------------------------- boundary placement

def test_snap_prefers_smooth_transition():
    """A cut should snap to a low-distance (coherent) transition near the nominal
    boundary instead of landing on a rough one."""
    n, cap, overlap, snap = 60, 25, 5, 8
    stride = cap - overlap                       # nominal next start = 20
    ids, h = ids_for(n), walk(n)
    # make the nominal transition (19->20) ROUGH and an earlier one (14->15) the
    # smoothest in the search window so the snap clearly prefers 15.
    h[20] = h[19] ^ 0b11111111                   # hamming(19,20) = 8 (rough)
    for k in range(stride - snap, stride + 1):   # window [12, 20]
        if k not in (15,):
            # ensure every other in-window transition has distance >= 2
            h[k] = h[k - 1] ^ 0b11                # hamming(k-1,k) = 2
    h[15] = h[14] ^ 0b1                           # hamming(14,15) = 1 (smoothest)
    spec = ss.segment(ids, h, cap=cap, overlap=overlap, snap=snap)
    assert spec["segments"][1]["start_idx"] == 15


# ----------------------------------------------------------------- single space (M=1)

def test_small_pool_single_segment_no_seams():
    n, cap = 40, 300
    ids, h = ids_for(n), walk(n)
    spec = ss.segment(ids, h, cap=cap, overlap=30)
    assert spec["n_segments"] == 1
    assert spec["seams"] == []
    assert spec["all_seams_registrable"] is True     # vacuously
    assert spec["segments"][0]["n_frames"] == n
    assert "neighbors" not in spec["segments"][0]     # empty boundary_spec


# ----------------------------------------------------------------- validation

def test_rejects_overlap_ge_cap():
    ids, h = ids_for(10), walk(10)
    for bad in (10, 12):
        try:
            ss.segment(ids, h, cap=10, overlap=bad)
            assert False, "should reject overlap >= cap"
        except ValueError:
            pass


def test_determinism():
    n, cap, overlap = 80, 20, 6
    ids, h = ids_for(n), walk(n)
    h[70] = gray(5)
    a = ss.segment(ids, h, cap=cap, overlap=overlap, snap=4)
    b = ss.segment(ids, h, cap=cap, overlap=overlap, snap=4)
    assert a == b


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
