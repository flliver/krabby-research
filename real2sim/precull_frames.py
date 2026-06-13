#!/usr/bin/env python3
"""STO-SCN-092 — pose-free pre-cull (sharpness + perceptual dedup) for large pools.

Cheaply shrink a massive frame pool (thousands of frames) to a tractable candidate
set BEFORE paying for a pose solve: drop motion-blurred frames and near-duplicates,
no poses required. Composes two existing, pose-free signals:

  - sharpness  : variance-of-Laplacian  (reuses select_sharp_frames.sharpness_of_gray)
  - similarity : 64-bit DCT perceptual hash (shared real2sim/phash.py)

Design (STO-SCN-096 #2, conclusion "cull first"):
  1. score every frame (sharpness + pHash) in a single decode pass
  2. DEDUP consecutive near-duplicate runs, keeping the sharpest per run.
     Dedup is LOCAL (a bounded temporal window) so path REVISITS — temporally
     distant frames that look alike — are PRESERVED (they're loop-closure /
     co-visibility gold for STO-SCN-098, not redundancy).
  3. BLUR GATE: drop frames below blur_rel x the median sharpness (scene-adaptive).
  4. TARGET thin (optional): windowed-sharpest down to a target count, spread
     evenly over the timeline. Default target = 300 (the measured 16 GB solve
     ceiling; see tasks/solve-cameras.json chunk_size).
  5. GAP GUARD (last): no retained temporal gap exceeds max_gap — re-insert the
     sharpest available frame to keep sequential overlap (the connectivity the
     pose solve + selection need; the 300-frame drift violated exactly this).

Pure CPU, deterministic, "runs anywhere" (numpy + PIL only). The core `precull()`
operates on an ordered list of (id, path) so it serves both the standalone CLI
(ids = filenames) and the v4 store wiring (ids = image hashes).

CLI:
    python precull_frames.py <src_dir> <out_dir> [--target N] [--phash-thresh H]
        [--sharp-pct P] [--max-gap G] [--dup-window W] [--score-edge E]
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import phash  # noqa: E402
from select_sharp_frames import sharpness_of_gray  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

DEFAULTS = dict(target=300, phash_thresh=8, blur_rel=0.2,
                max_gap=20, dup_window=12, score_edge=480)


@dataclass
class Frame:
    id: str
    path: Path
    idx: int = 0
    sharp: float = 0.0
    hash: int = 0


@dataclass
class PrecullResult:
    kept: list[str]                       # selected ids, temporal order
    report: dict = field(default_factory=dict)


def score_frames(frames: list[Frame], score_edge: int) -> None:
    """Populate sharp + hash for each frame in a single decode pass."""
    for f in frames:
        with Image.open(f.path) as im:
            gray = im.convert("L")
            f.sharp = sharpness_of_gray(gray, score_edge)
            f.hash = int(phash.phash(gray))


def _dedup_runs(frames: list[Frame], phash_thresh: int, dup_window: int) -> list[int]:
    """Collapse consecutive near-duplicate runs; keep the sharpest index per run.
    Anchor-based: a run extends while frames stay within phash_thresh of the run
    anchor AND inside dup_window — so a revisit (distant in time) starts a new run
    and is preserved."""
    n = len(frames)
    kept, i = [], 0
    while i < n:
        run = [i]
        j = i
        while (j + 1 < n and (j + 1 - i) < dup_window
               and phash.hamming(frames[i].hash, frames[j + 1].hash) <= phash_thresh):
            j += 1
            run.append(j)
        kept.append(max(run, key=lambda k: frames[k].sharp))
        i = j + 1
    return kept


def _windowed_sharpest(frames: list[Frame], candidates: list[int],
                       target: int, n: int) -> list[int]:
    """Thin `candidates` to ~target by taking the sharpest candidate in each of
    `target` even timeline windows (falls back to nothing for an empty window —
    the gap guard fills coverage holes afterwards)."""
    cand = sorted(candidates)
    bounds = np.linspace(0, n, target + 1).astype(int)
    out = []
    for w in range(target):
        lo, hi = bounds[w], bounds[w + 1]
        in_win = [k for k in cand if lo <= k < hi]
        if in_win:
            out.append(max(in_win, key=lambda k: frames[k].sharp))
    return sorted(set(out))


def _gap_guard(frames: list[Frame], kept: list[int], excluded: set[int],
               max_gap: int) -> tuple[list[int], int]:
    """Ensure no temporal gap between consecutive kept indices exceeds max_gap by
    inserting the sharpest available pool frame inside an oversized gap. Returns
    (kept, n_inserted)."""
    kept = sorted(kept)
    inserted = 0
    changed = True
    while changed:
        changed = False
        for a, b in zip(kept, kept[1:]):
            if b - a > max_gap:
                pool = [k for k in range(a + 1, b) if k not in kept]
                if not pool:
                    continue
                pick = max(pool, key=lambda k: frames[k].sharp)
                kept = sorted(kept + [pick])
                inserted += 1
                changed = True
                break
    return kept, inserted


def precull(items: list[tuple[str, "Path | str"]], *,
            target: int | None = DEFAULTS["target"],
            phash_thresh: int = DEFAULTS["phash_thresh"],
            blur_rel: float = DEFAULTS["blur_rel"],
            max_gap: int = DEFAULTS["max_gap"],
            dup_window: int = DEFAULTS["dup_window"],
            score_edge: int = DEFAULTS["score_edge"]) -> PrecullResult:
    """Pose-free pre-cull. `items` = ordered (id, path) list (temporal order).
    Returns the kept ids + a measured report. Small pools (<= target after
    dedup/gate) are kept whole (conclusion #1: small pool -> use all)."""
    frames = [Frame(id=str(i), path=Path(p), idx=k) for k, (i, p) in enumerate(items)]
    n = len(frames)
    if n == 0:
        return PrecullResult([], {"source_pool_n": 0})
    score_frames(frames, score_edge)

    deduped = _dedup_runs(frames, phash_thresh, dup_window)
    n_after_dedup = len(deduped)

    # Blur gate: scene-adaptive — drop frames below blur_rel x the median
    # sharpness (a motion-blurred frame is markedly less sharp than typical; a
    # uniformly-sharp pool drops nothing). Median-relative beats a fixed
    # percentile, which can land *inside* a minority blur cluster.
    med = float(np.median([f.sharp for f in frames]))
    thr = blur_rel * med
    gated = [k for k in deduped if frames[k].sharp >= thr]
    n_blur_dropped = n_after_dedup - len(gated)

    thinned = gated
    n_thinned = 0
    if target is not None and len(gated) > target:
        thinned = _windowed_sharpest(frames, gated, target, n)
        n_thinned = len(gated) - len(thinned)

    excluded = {f.idx for f in frames} - set(thinned)
    final, n_inserted = _gap_guard(frames, thinned, excluded, max_gap)

    kept_ids = [frames[k].id for k in sorted(final)]
    gaps = np.diff(sorted(final)) if len(final) > 1 else np.array([0])
    report = {
        "source_pool_n": n,
        "kept_n": len(final),
        "after_dedup_n": n_after_dedup,
        "dropped_near_dup": n - n_after_dedup,
        "dropped_blur": n_blur_dropped,
        "thinned_to_target": n_thinned,
        "gap_filled_inserted": n_inserted,
        "params": dict(target=target, phash_thresh=phash_thresh, blur_rel=blur_rel,
                       max_gap=max_gap, dup_window=dup_window, score_edge=score_edge),
        "sharp_median": round(med, 2),
        "blur_gate_value": round(thr, 2),
        "kept_index_gap_max": int(gaps.max()),
        "kept_index_gap_median": float(np.median(gaps)),
    }
    return PrecullResult(kept_ids, report)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pose-free pre-cull (sharpness + pHash dedup).")
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--target", type=int, default=DEFAULTS["target"],
                    help="target candidate count (0 = no thinning); default 300 = solve ceiling")
    ap.add_argument("--phash-thresh", type=int, default=DEFAULTS["phash_thresh"])
    ap.add_argument("--blur-rel", type=float, default=DEFAULTS["blur_rel"],
                    help="drop frames below this fraction of median sharpness (0 = no blur gate)")
    ap.add_argument("--max-gap", type=int, default=DEFAULTS["max_gap"])
    ap.add_argument("--dup-window", type=int, default=DEFAULTS["dup_window"])
    ap.add_argument("--score-edge", type=int, default=DEFAULTS["score_edge"])
    a = ap.parse_args(argv)

    src = Path(a.src_dir)
    pool = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not pool:
        sys.exit(f"no images in {src}")
    items = [(p.name, p) for p in pool]
    res = precull(items, target=(a.target or None), phash_thresh=a.phash_thresh,
                  blur_rel=a.blur_rel, max_gap=a.max_gap, dup_window=a.dup_window,
                  score_edge=a.score_edge)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_name = {p.name: p for p in pool}
    for name in res.kept:
        shutil.copy2(by_name[name], out / name)
    report = {"schema_version": "1", "tool": "precull_frames",
              "status": "success", "provenance": "measured",
              "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "host": socket.gethostname(), "src": str(src), "out": str(out),
              **res.report, "kept": res.kept}
    (out / "precull-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"precull: {res.report['source_pool_n']} -> {res.report['kept_n']} "
          f"(near-dup -{res.report['dropped_near_dup']}, blur -{res.report['dropped_blur']}, "
          f"thinned -{res.report['thinned_to_target']}, gap-fill +{res.report['gap_filled_inserted']})")
    print(f"wrote {out/'precull-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
