#!/usr/bin/env python3
"""STO-SCN-097 — spine segmentation: chunk a long trajectory into M overlapping segments.

The OUTER loop of EPI-SCN-SPINE-ASSEMBLY. A single video/trajectory is too big to pose
or reconstruct at once; the goal is ONE cohesive space. This stage partitions the ordered
frame pool into M **overlapping** windows, each individually tractable for the per-segment
pipeline (EPI-SCN-AUTO-SUBSET-SELECT), with shared boundary frames that guarantee adjacent
segments can register (STO-SCN-098). Pose-free — runs on the raw frame order, BEFORE any
per-segment solve, exactly like the pre-cull (STO-SCN-092).

It reuses the STO-SCN-092 signal — the 64-bit DCT perceptual hash (`phash`) — twice:

  1. BOUNDARY PLACEMENT. Overlapping windows on a fixed stride guarantee each segment is
     <= solver capacity AND adjacent segments share >= the overlap budget BY CONSTRUCTION.
     Each cut is then SNAPPED earlier (never later — that would drop overlap below budget)
     to the most-similar local frame transition, so the shared region sits in a *coherent*
     stretch instead of across a fast-motion gap (the "don't cut across a low-overlap gap"
     requirement). Per-seam registrability (mean consecutive pHash distance over the shared
     region) is measured and flagged when it fails a threshold.

  2. LOOP / REVISIT DETECTION. Path revisits — temporally distant frames that look alike —
     are co-visibility gold for global registration, NOT redundancy (the pre-cull preserves
     them for exactly this reason). A cheap cross-segment pHash scan over representative
     frames flags candidate loop closures between NON-adjacent segments, handed to
     STO-SCN-098 as extra pose-graph edges.

Output = the spine's per-segment `boundary_spec` (the IN contract STO-SCN-094 honors):
pinned anchor frames + the overlap region per neighbor, plus the global `camera_model`
(STO-SCN-091, identical for every segment). For a single tractable space (M=1) there are no
seams and the spec is empty — the per-segment pipeline runs unchanged.

The core `segment(ids, hashes, ...)` takes precomputed hashes so it is fully unit-testable
without images; `hashes_for(items)` decodes a pool. Pure CPU, deterministic, numpy+PIL only.

CLI:
    python spine_segment.py <src_dir> [--cap 300] [--overlap 30] [--snap 10]
        [--reg-thresh 12] [--loop-thresh 8] [--loop-min-sep 2] [--loop-step 5] [--out spine.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phash  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

DEFAULTS = dict(cap=300, overlap=30, snap=10, reg_thresh=12,
                loop_thresh=8, loop_min_sep=2, loop_step=5)


# ----------------------------------------------------------------- helpers

def _mean_consec_hamming(hashes, lo, hi) -> float:
    """Mean pHash distance between consecutive frames over the inclusive index
    range [lo, hi] — the cheap registrability proxy for a shared region: low
    distance == frames see overlapping content == the seam will register."""
    if hi <= lo:
        return 0.0
    ds = [phash.hamming(hashes[i], hashes[i + 1]) for i in range(lo, hi)]
    return sum(ds) / len(ds)


def _snap_cut(hashes, nominal, snap, lo_bound) -> int:
    """Snap a nominal segment-start index to the most-similar local transition.
    Search [nominal-snap, nominal] ONLY (never later — a later cut shrinks the
    overlap below the budget). Pick the index w whose transition INTO it
    (hamming(w-1, w)) is smallest, so the cut sits in a coherent stretch. Keeps
    w strictly > lo_bound (the previous start) for monotonic, non-empty segments."""
    lo = max(lo_bound + 1, nominal - snap, 1)
    hi = nominal
    if hi < lo:
        return nominal
    return min(range(lo, hi + 1), key=lambda w: (phash.hamming(hashes[w - 1], hashes[w]), w))


def _loop_candidates(ids, hashes, seg_of, segments, loop_thresh, loop_min_sep, loop_step):
    """Flag candidate loop closures: the single best (lowest-distance) frame pair
    between each NON-adjacent segment pair whose distance <= loop_thresh. Compares
    every `loop_step`-th frame to stay cheap on long pools. Deterministic."""
    reps = [list(range(s["start_idx"], s["end_idx"] + 1, max(1, loop_step))) for s in segments]
    out = []
    for a in range(len(segments)):
        for b in range(a + loop_min_sep, len(segments)):
            best = None
            for i in reps[a]:
                for j in reps[b]:
                    d = phash.hamming(hashes[i], hashes[j])
                    if d <= loop_thresh and (best is None or d < best[2]):
                        best = (i, j, d)
            if best is not None:
                out.append({"seg_a": a, "seg_b": b,
                            "frame_a": ids[best[0]], "frame_b": ids[best[1]],
                            "idx_a": best[0], "idx_b": best[1], "hamming": best[2]})
    return out


# ----------------------------------------------------------------- core

def segment(ids, hashes, *, cap=DEFAULTS["cap"], overlap=DEFAULTS["overlap"],
            snap=DEFAULTS["snap"], reg_thresh=DEFAULTS["reg_thresh"],
            loop_thresh=DEFAULTS["loop_thresh"], loop_min_sep=DEFAULTS["loop_min_sep"],
            loop_step=DEFAULTS["loop_step"]) -> dict:
    """Partition an ordered pool (ids + matching pHashes, temporal order) into M
    overlapping segments, each <= `cap`, adjacent pairs sharing >= `overlap`
    frames. Returns the spine spec: segments (with per-neighbor overlap +
    anchors), per-seam registrability, and cross-segment loop candidates."""
    n = len(ids)
    if n != len(hashes):
        raise ValueError("ids and hashes length mismatch")
    if not (0 < overlap < cap):
        raise ValueError(f"need 0 < overlap ({overlap}) < cap ({cap})")
    stride = cap - overlap

    # ---- starts: fixed overlapping grid, each cut snapped earlier to a coherent transition
    if n <= cap:
        starts = [0]
    else:
        starts = [0]
        while starts[-1] + cap < n:
            nominal = starts[-1] + stride
            w = _snap_cut(hashes, nominal, snap, starts[-1])
            if w <= starts[-1]:
                w = nominal                      # degenerate snap — fall back
            starts.append(w)

    segs = [{"id": k, "start_idx": s, "end_idx": min(s + cap - 1, n - 1)}
            for k, s in enumerate(starts)]

    # ---- neighbor overlap regions + per-seam registrability
    seams = []
    for k in range(len(segs) - 1):
        a, b = segs[k], segs[k + 1]
        ov_lo, ov_hi = b["start_idx"], a["end_idx"]          # shared inclusive range
        ov_ids = ids[ov_lo:ov_hi + 1]
        n_ov = len(ov_ids)
        reg = _mean_consec_hamming(hashes, ov_lo, ov_hi)
        ok = n_ov >= overlap and reg <= reg_thresh
        # anchors = the shared frames (the OUT side of the boundary contract, STO-SCN-095/098)
        a.setdefault("neighbors", {})["next"] = {
            "seg": b["id"], "n_overlap": n_ov, "overlap_ids": ov_ids,
            "anchor_ids": ov_ids, "registrability": round(reg, 2), "registrable": ok}
        b.setdefault("neighbors", {})["prev"] = {
            "seg": a["id"], "n_overlap": n_ov, "overlap_ids": ov_ids,
            "anchor_ids": ov_ids, "registrability": round(reg, 2), "registrable": ok}
        seams.append({"seg_a": a["id"], "seg_b": b["id"], "n_overlap": n_ov,
                      "registrability": round(reg, 2), "registrable": ok})

    seg_of = {}
    for s in segs:
        for i in range(s["start_idx"], s["end_idx"] + 1):
            seg_of.setdefault(i, s["id"])         # first (leftmost) owner

    # materialize frames + drop helper index off the public segment record
    for s in segs:
        s["frames"] = ids[s["start_idx"]:s["end_idx"] + 1]
        s["n_frames"] = len(s["frames"])

    loops = _loop_candidates(ids, hashes, seg_of, segs, loop_thresh, loop_min_sep, loop_step)

    return {
        "schema": 4,
        "n_frames": n,
        "n_segments": len(segs),
        "params": dict(cap=cap, overlap=overlap, snap=snap, reg_thresh=reg_thresh,
                       loop_thresh=loop_thresh, loop_min_sep=loop_min_sep, loop_step=loop_step),
        "segments": segs,
        "seams": seams,
        "all_seams_registrable": all(s["registrable"] for s in seams) if seams else True,
        "max_segment_n": max(s["n_frames"] for s in segs),
        "within_capacity": all(s["n_frames"] <= cap for s in segs),
        "loop_candidates": loops,
        "n_loop_candidates": len(loops),
    }


# ----------------------------------------------------------------- image decode

def hashes_for(items):
    """items = ordered (id, path) list -> (ids, hashes) via the pHash decode."""
    ids, hashes = [], []
    for i, p in items:
        ids.append(str(i))
        hashes.append(int(phash.phash_file(p)))
    return ids, hashes


# ----------------------------------------------------------------- CLI

def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spine segmentation: M overlapping segments (STO-SCN-097).")
    ap.add_argument("src_dir")
    ap.add_argument("--cap", type=int, default=DEFAULTS["cap"], help="max frames per segment (solver capacity)")
    ap.add_argument("--overlap", type=int, default=DEFAULTS["overlap"], help="min shared frames per seam (budget)")
    ap.add_argument("--snap", type=int, default=DEFAULTS["snap"], help="boundary snap search window")
    ap.add_argument("--reg-thresh", type=int, default=DEFAULTS["reg_thresh"], help="max mean-pHash dist for a registrable seam")
    ap.add_argument("--loop-thresh", type=int, default=DEFAULTS["loop_thresh"], help="max pHash dist for a loop candidate")
    ap.add_argument("--loop-min-sep", type=int, default=DEFAULTS["loop_min_sep"], help="min segment separation for a loop")
    ap.add_argument("--loop-step", type=int, default=DEFAULTS["loop_step"], help="frame subsample stride for loop scan")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    src = Path(a.src_dir)
    pool = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not pool:
        sys.exit(f"no images in {src}")
    ids, hashes = hashes_for([(p.name, p) for p in pool])
    spec = segment(ids, hashes, cap=a.cap, overlap=a.overlap, snap=a.snap,
                   reg_thresh=a.reg_thresh, loop_thresh=a.loop_thresh,
                   loop_min_sep=a.loop_min_sep, loop_step=a.loop_step)
    out = Path(a.out) if a.out else src / "spine.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"spine: {spec['n_frames']} frames -> {spec['n_segments']} segments "
          f"(max {spec['max_segment_n']}/{a.cap}, all-registrable={spec['all_seams_registrable']}, "
          f"{spec['n_loop_candidates']} loop candidate(s))")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
