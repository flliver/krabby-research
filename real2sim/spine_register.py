#!/usr/bin/env python3
"""STO-SCN-098 — global registration of per-segment submaps into one gauge.

Each spine segment (STO-SCN-097) is solved in its own ARBITRARY SfM gauge with its own
drift. M locally-good segments in disjoint gauges are still M disjoint reconstructions —
this stage is the only place the spine becomes ONE cohesive, drift-corrected space
(EPI-SCN-SPINE-ASSEMBLY, STO-SCN-096 #7).

A SIM(3) **pose graph** over the segments:

  - NODES   = per-segment gauges (unknown similarity G_k: segment-k-local -> global).
  - EDGES   = correspondences between two segments' cameras:
      * ADJACENT (chain) edges come for free from the boundary OVERLAP — the same camera
        IDENTITIES appear in both segments (the OUT contract of STO-SCN-095/097). Direct.
      * LOOP-CLOSURE edges come from STO-SCN-097's revisit flags — the same PLACE seen
        again from a temporally distant frame. A revisit shares no camera identity (each
        video frame is captured once), so a loop edge needs an explicit correspondence
        (matched camera pairs from a feature-match expansion of the revisit neighbourhood).
        The optimiser CONSUMES loop edges to distribute cycle drift; producing the loop
        correspondence from raw revisit frames is the integration step (see Notes).

Per-edge relative similarity reuses the canonical `gauge_align` (the posed-weld gauge-sim
gate, STO-SCN-090): `consensus_align` robustly trims badly-registered overlap frames, and
the rotation-augmented solve pins orientation even for the near-collinear camera centres of
a walking path (the degenerate case `gauge_align` was hardened against on 2026-06-10).

Global solve: fix the reference segment's gauge = identity (removes the global gauge
freedom), initialise the rest along a spanning tree, then **Gauss-Seidel relaxation** —
repeatedly re-fit each segment's gauge to the current global positions of ALL its
neighbours' shared cameras. Loop edges close cycles, so the relaxation spreads the
loop-closure residual around the loop instead of dumping it at one seam. Converges to one
gauge; emits per-seam residuals (the falsifiable tolerance gate, T-001) + globally
consistent per-camera poses for fusion (STO-SCN-099).

Pure numpy. Optional global BA over merged tracks is a deliberate follow-up (Notes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gauge_align as ga  # noqa: E402

Gauge = tuple  # (scale: float, R: (3,3), t: (3,))
IDENTITY: Gauge = (1.0, np.eye(3), np.zeros(3))


# ----------------------------------------------------------------- similarity algebra

def apply_gauge(G: Gauge, pts: np.ndarray) -> np.ndarray:
    s, R, t = G
    return s * (np.asarray(pts, float) @ R.T) + t


def compose(outer: Gauge, inner: Gauge) -> Gauge:
    """outer ∘ inner: apply `inner` then `outer`."""
    s2, R2, t2 = outer
    s1, R1, t1 = inner
    return (s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2)


def _rot_global(G: Gauge, rots: np.ndarray) -> np.ndarray:
    """Camera orientations carried into the global gauge (rotation only — unitless)."""
    return np.einsum("ij,njk->nik", G[1], np.asarray(rots, float))


# ----------------------------------------------------------------- edges

def shared_id_edges(nodes: dict, min_shared: int = 3) -> list:
    """Adjacent (chain) edges: every segment pair sharing >= min_shared camera
    IDENTITIES (the boundary overlap). Non-overlapping pairs share none, so this
    yields exactly the seams. Each edge carries the matched index arrays."""
    ids = sorted(nodes)
    pos = {k: {n: i for i, n in enumerate(nodes[k]["names"])} for k in ids}
    edges = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            common = [n for n in nodes[i]["names"] if n in pos[j]]
            if len(common) >= min_shared:
                edges.append({"i": i, "j": j, "type": "adjacent",
                              "i_idx": np.array([pos[i][n] for n in common]),
                              "j_idx": np.array([pos[j][n] for n in common]),
                              "names": common})
    return edges


def _node_arrays(node, idx):
    c = np.asarray(node["centers"], float)[idx]
    r = node.get("rots")
    r = None if r is None else np.asarray(r, float)[idx]
    return c, r


def _fit(src, dst, srot, drot, rel_tol, robust) -> Gauge:
    """Per-edge / per-node similarity. Robust (default) uses consensus trimming so
    badly-registered boundary frames (real SfM noise) don't poison the fit — the
    failure that 15% of 001-patio's cross-solve overlap exhibited; a plain fit gave
    20% max residual, consensus kept 60% at 1.7%. Falls back to a plain solve only if
    consensus collapses (too few inliers) so the relaxation still completes and the
    high residual surfaces in the seam gate.

    Limitation (honest): consensus trims sequentially from a non-robust (plain
    least-squares) seed, so it handles the MODERATE outliers real SfM produces (off by
    tens of % of spread — the 001-patio cross-solve case) but can break under GROSS
    outliers (off by multiples of spread), where the seed fit is so corrupted that good
    points rank as outliers. A RANSAC seed would harden this — deferred; not exhibited by
    real data."""
    if robust and len(src) >= 6:
        try:
            a = ga.consensus_align(src, dst, rel_tol=rel_tol,
                                   src_rotations=srot, dst_rotations=drot)
            return (a["scale"], a["R"], a["t"])
        except RuntimeError:
            pass
    a = ga.align_camera_sets(src, dst, src_rotations=srot, dst_rotations=drot)
    return (a["scale"], a["R"], a["t"])


# ----------------------------------------------------------------- pose-graph solve

def _spanning_init(nodes, edges, ref, rel_tol, robust) -> dict:
    """Initialise each node's gauge by composing pairwise relative similarities
    along a spanning tree rooted at `ref` (BFS over the edge graph)."""
    adj = {k: [] for k in nodes}
    for e in edges:
        adj[e["i"]].append(e)
        adj[e["j"]].append(e)
    G = {ref: IDENTITY}
    queue = [ref]
    while queue:
        p = queue.pop(0)
        for e in adj[p]:
            k = e["j"] if e["i"] == p else e["i"]
            if k in G:
                continue
            # correspondence indices: source = k-local, dest = p-local
            if e["i"] == p:
                p_idx, k_idx = e["i_idx"], e["j_idx"]
            else:
                p_idx, k_idx = e["j_idx"], e["i_idx"]
            kc, kr = _node_arrays(nodes[k], k_idx)
            pc, pr = _node_arrays(nodes[p], p_idx)
            relG = _fit(kc, pc, kr, pr, rel_tol, robust)   # k-local -> p-local
            G[k] = compose(G[p], relG)                     # -> global
            queue.append(k)
    # any node unreachable (disconnected) -> identity (caller surfaces it)
    for k in nodes:
        G.setdefault(k, IDENTITY)
    return G


def _edges_touching(edges, k):
    for e in edges:
        if e["i"] == k:
            yield e, e["j"], e["i_idx"], e["j_idx"]
        elif e["j"] == k:
            yield e, e["i"], e["j_idx"], e["i_idx"]


def register(nodes: dict, edges: list | None = None, *, ref=None, loops: list | None = None,
             iters: int = 500, eps: float = 1e-6, rel_tol: float = 0.02,
             robust: bool = True, min_consensus_frac: float = 0.5) -> dict:
    """Register M per-segment submaps into one global gauge.

    nodes: {seg_id: {"names":[...], "centers":(N,3), "rots":(N,3,3) | None}}
    edges: explicit edges (default: derived from shared camera identities).
    loops: extra loop-closure edges, each {"i","j","i_idx","j_idx"} (matched camera
           indices in each node) — typically from STO-SCN-097 revisit flags expanded to
           correspondences. Folded into the pose graph as cycle constraints.

    Returns the per-segment gauges, per-edge seam residuals (abs + relative to scene
    spread — the scale-free gate, T-001), globally consistent per-camera poses, and
    convergence info.
    """
    ids = sorted(nodes)
    if ref is None:
        ref = ids[0]
    if edges is None:
        edges = shared_id_edges(nodes)
    if loops:
        edges = edges + [{**lp, "type": "loop",
                          "i_idx": np.asarray(lp["i_idx"]), "j_idx": np.asarray(lp["j_idx"])}
                         for lp in loops]
    if not edges:
        raise RuntimeError("no registration edges — segments share no boundary cameras "
                           "(insufficient overlap); cannot bring into one gauge.")

    G = _spanning_init(nodes, edges, ref, rel_tol, robust)

    # scale-free convergence floor: changes below eps × the reference segment's extent.
    # NB: Gauss-Seidel converges fast on a chain (<~20 iters) but only geometrically on a
    # CYCLE (loop closure) — the loop residual diffuses one node per iteration — so loopy
    # graphs need the larger `iters` budget; each iter is a handful of tiny SVDs.
    ref_c = np.asarray(nodes[ref]["centers"], float)
    scale_ref = float(np.linalg.norm(ref_c - ref_c.mean(0), axis=1).mean()) or 1.0
    chg_floor = eps * scale_ref

    # ---- Gauss-Seidel relaxation: re-fit each non-ref gauge to neighbours' global pts
    converged, it = False, 0
    for it in range(1, iters + 1):
        max_chg = 0.0
        for k in ids:
            if k == ref:
                continue
            S, D, SR, DR = [], [], [], []
            for _e, p, k_idx, p_idx in _edges_touching(edges, k):
                kc, kr = _node_arrays(nodes[k], k_idx)
                pc, pr = _node_arrays(nodes[p], p_idx)
                S.append(kc)
                D.append(apply_gauge(G[p], pc))
                if kr is not None and pr is not None:
                    SR.append(kr)
                    DR.append(_rot_global(G[p], pr))
            if not S:
                continue
            src = np.vstack(S)
            dst = np.vstack(D)
            if len(src) < 3:
                continue
            srot = np.vstack(SR) if SR and len(np.vstack(SR)) == len(src) else None
            drot = np.vstack(DR) if DR and len(np.vstack(DR)) == len(src) else None
            newG = _fit(src, dst, srot, drot, rel_tol, robust)
            # track the change at the segment's own camera centres
            before = apply_gauge(G[k], nodes[k]["centers"])
            G[k] = newG
            after = apply_gauge(G[k], nodes[k]["centers"])
            max_chg = max(max_chg, float(np.linalg.norm(after - before, axis=1).max()))
        if max_chg < chg_floor:
            converged = True
            break

    # ---- global per-camera poses (average overlap owners) + per-edge residuals
    acc: dict = {}
    for k in ids:
        gc = apply_gauge(G[k], nodes[k]["centers"])
        gr = _rot_global(G[k], nodes[k]["rots"]) if nodes[k].get("rots") is not None else None
        for n, idx in ((nm, i) for i, nm in enumerate(nodes[k]["names"])):
            rec = acc.setdefault(n, {"c": [], "r": []})
            rec["c"].append(gc[idx])
            if gr is not None:
                rec["r"].append(gr[idx])
    cameras = {}
    for n, rec in acc.items():
        cam = {"center": list(map(float, np.mean(rec["c"], axis=0)))}
        if rec["r"]:
            cam["R"] = [list(map(float, row)) for row in rec["r"][0]]  # first owner (consistent post-solve)
        cameras[n] = cam

    all_global = np.array([c["center"] for c in cameras.values()])
    spread = float(np.linalg.norm(all_global - all_global.mean(0), axis=1).mean()) or 1.0

    # ---- robust per-seam gate. A handful of badly-solved boundary frames (real SfM
    # noise) must NOT fail an otherwise-good registration, but a SYSTEMATICALLY warped /
    # mis-solved segment must be caught. So per seam: trim the worst correspondences to
    # the gate; report the inlier (consensus) residual + the consensus fraction. A seam
    # passes when a sufficient MAJORITY (>= min_consensus_frac) agrees within the gate —
    # a warp leaves too few in consensus and fails; sparse outliers are surfaced, not fatal.
    gate = rel_tol * spread
    seam_edges = []
    worst_inlier = 0.0
    worst_all = 0.0
    all_pass = True
    for e in edges:
        gi = apply_gauge(G[e["i"]], np.asarray(nodes[e["i"]]["centers"], float)[e["i_idx"]])
        gj = apply_gauge(G[e["j"]], np.asarray(nodes[e["j"]]["centers"], float)[e["j_idx"]])
        d = np.sort(np.linalg.norm(gi - gj, axis=1))
        n = len(d)
        floor = max(3, int(np.ceil(min_consensus_frac * n)))
        keep = n
        while keep > floor and d[keep - 1] > gate:
            keep -= 1
        inlier_max = float(d[keep - 1])
        cfrac = keep / n
        ok = (cfrac >= min_consensus_frac) and (inlier_max <= gate)
        all_pass = all_pass and ok
        worst_inlier = max(worst_inlier, inlier_max if ok else float(d[-1]))
        worst_all = max(worst_all, float(d[-1]))
        seam_edges.append({
            "i": e["i"], "j": e["j"], "type": e["type"], "n_shared": n,
            "residual_max": round(float(d[-1]), 6), "residual_median": round(float(d[n // 2]), 6),
            "residual_inlier_max": round(inlier_max, 6),
            "residual_rel": round(inlier_max / spread, 6),
            "consensus_frac": round(cfrac, 4), "n_outlier": n - keep,
            "registrable": ok})

    return {
        "n_segments": len(ids), "ref": ref, "n_edges": len(edges),
        "gauges": {k: {"scale": float(G[k][0]),
                       "R": [list(map(float, r)) for r in G[k][1]],
                       "t": list(map(float, G[k][2]))} for k in ids},
        "seams": seam_edges,
        "max_seam_residual": round(worst_inlier, 6),
        "max_seam_residual_rel": round(worst_inlier / spread, 6),
        "max_seam_residual_all": round(worst_all, 6),
        "gate": round(gate, 6), "rel_tol": rel_tol, "min_consensus_frac": min_consensus_frac,
        "within_tol": all_pass,
        "scene_spread": round(spread, 6),
        "cameras": cameras, "n_cameras": len(cameras),
        "converged": converged, "iters_run": it,
    }


# ----------------------------------------------------------------- colmap solve reader

def _qvec2rotmat(w, x, y, z):
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def read_solve_poses(images_bin) -> dict:
    """COLMAP/FastMap images.bin -> {name: {"center":(3,), "R":(3,3 cam->world)}}.
    R is cam->world (R_w2c transposed); center = -R_w2c^T @ t."""
    import struct
    out = {}
    with open(images_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)
            Rwc = _qvec2rotmat(qw, qx, qy, qz).T
            C = -Rwc @ np.array([tx, ty, tz])
            out[name.decode("utf-8", "replace")] = {"center": C, "R": Rwc}
    return out


def nodes_from_solves(seg_solves: dict) -> dict:
    """seg_solves: {seg_id: images_bin_path} -> nodes dict for register().
    Keys segments on the solves' shared image NAMES — adjacent segments that
    solved the same boundary frames link automatically via shared_id_edges."""
    nodes = {}
    for k, p in seg_solves.items():
        poses = read_solve_poses(p)
        names = sorted(poses)
        nodes[k] = {"names": names,
                    "centers": np.array([poses[n]["center"] for n in names]),
                    "rots": np.array([poses[n]["R"] for n in names])}
    return nodes


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Global registration of spine submaps (STO-SCN-098).")
    ap.add_argument("nodes_json", help="JSON: {seg_id: {names, centers, rots?}}")
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nodes_raw = json.loads(Path(a.nodes_json).read_text())
    nodes = {k: {"names": v["names"], "centers": np.array(v["centers"], float),
                 "rots": np.array(v["rots"], float) if v.get("rots") else None}
             for k, v in nodes_raw.items()}
    out = register(nodes, rel_tol=a.rel_tol)
    txt = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(f"registered {out['n_segments']} segments | max seam residual "
          f"{out['max_seam_residual']} ({out['max_seam_residual_rel']*100:.2f}% of spread) | "
          f"within_tol={out['within_tol']} | converged={out['converged']} ({out['iters_run']} it)")
