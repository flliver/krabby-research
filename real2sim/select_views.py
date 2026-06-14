#!/usr/bin/env python3
"""STO-SCN-094 — coverage-greedy best-N view selection over the SfM track graph.

Given a posed pool (a COLMAP/FastMap sparse/0, STO-SCN-093), automatically propose
the best N views for a high-quality reconstruction: maximize surface coverage and
triangulation quality, keep the selected view-graph connected, drop redundancy
(STO-SCN-096 #2). The splat (STO-SCN-095) is where a human verifies/overrides; this
is the automated proposal.

Method — greedy submodular maximization on the image↔point incidence:
  - seed with the highest-coverage view;
  - each step add the view with the largest MARGINAL gain:
      new (never-seen) points        -> weight W_NEW
      points it TRIANGULATES (a prior selected view also sees them)
                                      -> weight W_TRIANG x triangulation-angle quality,
    where angle quality is a tent peaking in ~10-30 deg (baseline wide enough for
    depth, not so wide matching fails);
  - subject to a CONNECTIVITY constraint: a kept view must share >= min_overlap
    points with the already-selected set (so the graph never fragments);
  - stop at N or when the marginal gain saturates.
Deterministic (index tie-break). Pure stdlib (struct/math via covis_graph) — runs
anywhere, fully unit-testable.

Spine note (STO-SCN-096 #7): when part of a spine, a `boundary_spec` pre-seeds the
selection with pinned anchor frames + a seam-overlap budget. Empty for M=1.

Usage:
  select_views.py <sparse_dir> --n 24 [--min-overlap 10] [--out selection.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import covis_graph as cg  # noqa: E402

W_NEW = 1.0          # marginal weight: a point this view sees for the first time
W_TRIANG = 5.0       # marginal weight: a point this view TRIANGULATES (x angle quality)


def angle_quality(deg: float) -> float:
    """Tent peaking at 1.0 in [10,30] deg; linear to 0 at 2 and 60 deg; 0 outside."""
    if deg <= 2.0 or deg >= 60.0:
        return 0.0
    if deg < 10.0:
        return (deg - 2.0) / 8.0
    if deg <= 30.0:
        return 1.0
    return (60.0 - deg) / 30.0


def load_incidence(sparse_dir):
    """-> (centers{img->xyz}, names{img->name}, img_pts{img->set(pt)}, pt_imgs{pt->[img]},
    pt_xyz{pt->xyz})."""
    images = cg.read_images_bin(Path(sparse_dir) / "images.bin")
    pts = cg.read_points3D_bin(Path(sparse_dir) / "points3D.bin")
    centers = {i: im["center"] for i, im in images.items()}
    names = {i: im["name"] for i, im in images.items()}
    img_pts = {i: set() for i in images}
    pt_imgs, pt_xyz = {}, {}
    for k, p in enumerate(pts):
        ids = [i for i in p["image_ids"] if i in centers]
        if len(ids) < 2:            # a point needs >=2 views to triangulate
            continue
        pt_imgs[k] = ids
        pt_xyz[k] = p["xyz"]
        for i in ids:
            img_pts[i].add(k)
    return centers, names, img_pts, pt_imgs, pt_xyz


def select(centers, names, img_pts, pt_imgs, pt_xyz, n, min_overlap=10,
           pinned=None):
    """Greedy coverage+triangulation+connectivity selection -> ordered list of img ids."""
    imgs = sorted(img_pts, key=lambda i: (-len(img_pts[i]), i))   # deterministic
    covered_by = {}                       # pt -> list of selected imgs seeing it
    selected, order = set(), []
    covered_pts = set()                   # pts seen by >=1 selected img

    def add(v):
        selected.add(v); order.append(v)
        for pt in img_pts[v]:
            covered_by.setdefault(pt, []).append(v)
            covered_pts.add(pt)

    # seed: pinned anchors first (spine boundary_spec), else the max-coverage view
    for v in (pinned or []):
        if v in img_pts and v not in selected:
            add(v)
    if not selected:
        add(imgs[0])

    def gain(v):
        g = 0.0
        for pt in img_pts[v]:
            cb = covered_by.get(pt)
            if not cb:
                g += W_NEW
            else:
                best = max(angle_quality(cg._angle_deg(centers[u], centers[v], pt_xyz[pt]))
                           for u in cb)
                g += W_TRIANG * best
        return g

    while len(selected) < n:
        best_v, best_g = None, 0.0
        for v in imgs:
            if v in selected:
                continue
            if min_overlap and len(img_pts[v] & covered_pts) < min_overlap:
                continue                  # connectivity: must overlap the selected set
            g = gain(v)
            if g > best_g:
                best_v, best_g = v, g
        if best_v is None or best_g <= 1e-9:
            break                          # saturated / no connected candidate left
        add(best_v)

    return order


def report(order, names, img_pts, pt_imgs, pt_xyz, centers):
    sel = set(order)
    triangulated = sum(1 for pt, ids in pt_imgs.items()
                       if sum(1 for i in ids if i in sel) >= 2)
    # triangulation angles among selected views, per point (best pair)
    angs = []
    for pt, ids in pt_imgs.items():
        s = [i for i in ids if i in sel]
        if len(s) >= 2:
            angs.append(max(cg._angle_deg(centers[s[a]], centers[s[b]], pt_xyz[pt])
                            for a in range(len(s)) for b in range(a + 1, len(s))))
    angs.sort()
    med = angs[len(angs) // 2] if angs else 0.0
    in_window = sum(1 for a in angs if 10 <= a <= 30)
    return {
        "n_selected": len(order),
        "selected": [names[i] for i in order],
        "triangulated_points": triangulated,
        "total_triangulable_points": len(pt_imgs),
        "coverage_pct": round(100 * triangulated / max(1, len(pt_imgs)), 1),
        "median_tri_angle_deg": round(med, 1),
        "pct_angles_in_10_30": round(100 * in_window / max(1, len(angs)), 1),
    }


def select_from_sparse(sparse_dir, n, min_overlap=10, pinned_names=None):
    centers, names, img_pts, pt_imgs, pt_xyz = load_incidence(sparse_dir)
    pinned = None
    if pinned_names:
        rev = {v: k for k, v in names.items()}
        pinned = [rev[x] for x in pinned_names if x in rev]
    order = select(centers, names, img_pts, pt_imgs, pt_xyz, n, min_overlap, pinned)
    r = report(order, names, img_pts, pt_imgs, pt_xyz, centers)
    return order, r


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Coverage-greedy best-N view selection (STO-SCN-094).")
    ap.add_argument("sparse_dir")
    ap.add_argument("--n", type=int, default=24, help="target view count (downstream sweet spot)")
    ap.add_argument("--min-overlap", type=int, default=10, help="connectivity: shared pts vs selected set")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    _, r = select_from_sparse(a.sparse_dir, a.n, a.min_overlap)
    out = Path(a.out) if a.out else Path(a.sparse_dir) / "selection.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"selected {r['n_selected']} views | coverage {r['coverage_pct']}% of "
          f"{r['total_triangulable_points']} triangulable pts | median tri-angle "
          f"{r['median_tri_angle_deg']} deg | {r['pct_angles_in_10_30']}% in 10-30 deg")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
