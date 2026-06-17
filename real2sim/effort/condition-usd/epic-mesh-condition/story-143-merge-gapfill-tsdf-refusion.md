---
xid: STO-SCN-143
parent: ./epic.md
kind: story
effort: scn
size: M
status: deferred
date: 2026-06-15
depends-on: [STO-SCN-099]
bd-id: krabby-wy7v
assignee: krabby
priority: 4
---

# (B) Merge & gap-fill via TSDF re-fusion — ScalableTSDFVolume conditioning node

# ⏸ DEFERRED — fallback approach for STO-SCN-013; revisit only after STO-SCN-142 (Poisson, A) results

## Summary

A `merge-gapfill` conditioning approach that re-integrates depth into an **Open3D
`ScalableTSDFVolume`** (the MAtCha `extract_tsdf_mesh` lineage) to produce a watertight manifold —
the surface-faithful alternative to Poisson (STO-SCN-142). **Deferred** per operator decision
(2026-06-15): do Poisson (A) first; only pick this up if A can't hold volume / fidelity on our
scenes.

## Context

STO-SCN-013's second named approach. TSDF re-fusion is more faithful to surfaces than Poisson
(no ballooning) and is the documented submap-fusion finisher, but it's **heavier**: it needs depth
maps / the gaussians as input (closer to a re-meshify than a pure post-process), so it's not a
free CPU step like the cull or Poisson. Hence deferred behind A.

## Design (sketch — to flesh out if activated)

### Approach
- Re-integrate per-view depth (from the rep's gaussians / cached depth) into an Open3D
  `ScalableTSDFVolume` with confidence-weighted integration (boundary artifacts are a confidence
  problem, not averaging — per the M11 submap-fusion notes), extract a watertight mesh.
- Likely a GPU/depth-bearing step (reuse the matcha `extract_tsdf_mesh.py` / `render_multires.py`
  path + the cached-gaussian re-extract pattern proven in `refilter`, STO-SCN-136 path A).

### Pipeline integration (v4)
- **New additive task** `merge-gapfill-tsdf` (`algo: merge-gapfill-tsdf@0`) — a distinct task from
  the Poisson `merge-gapfill@0`, so both approaches coexist + compete in the Rank UI without
  re-keying. Placement `{up_meshify_dir}/condition/{identity}` (or a meshify variant if it's a
  re-extract). Backwards-compat canonical: **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done (when activated)
- [ ] Watertight manifold via ScalableTSDF re-fusion; volume + surface fidelity ≥ Poisson (A).
- [ ] Additive node; NOOP re-run; backwards-compat preserved.
- [ ] Operator A/B vs the Poisson result in the Rank UI.

## Activation trigger
Pick this up if STO-SCN-142 (Poisson) **balloons / over-smooths / loses volume** on real scenes
(the genus/manifold + visual check fails to satisfy the operator). Until then: deferred.

## Out of scope
- Everything until activated. The Poisson approach is STO-SCN-142.
