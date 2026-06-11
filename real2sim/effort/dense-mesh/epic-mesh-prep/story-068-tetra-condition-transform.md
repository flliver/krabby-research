---
xid: STO-SCN-068
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-11
depends-on: []
bd-id: krabby-bla
---

# tetra_condition — decimate + Taubin-smooth the tetra mesh (deliverable-scale)

## Why

Runoff verdict (006, 2026-06-10): tetra mesh ranks #1 on quality but
is 32M tris / 1.2 GB — not a deliverable (IsaacSim wants ~0.5–2M tris).

## What

`real2sim/tetra_condition.py` (baked: krabby-da3:0.4 krabby-tools):
quadric decimation → Taubin smoothing (volume-preserving) →
nearest-vertex color transfer from the SOURCE mesh (decimation
degrades colors; source is color truth) → cleanup + normals. Emits
mesh + `tetra_condition_record_<N>k.json`.

## First execution (006 run-8-strong, the rank-1 variant)

- 16.1M verts / 32.2M tris (1,184 MB) → **433k verts / 1.0M tris
  (33 MB)** — 36× smaller. target_tris 1M, taubin 10.
- In runoff as `matcha--8-strong-tetra1m` (render-variant convention).
- Honest read: continuity held; mottled color speckle + small dark
  patches in thin geometry + blobbier bushes. Knobs to sweep if the
  operator wants better: target_tris 2M, taubin 0–5, smarter color
  transfer (barycentric on faces vs nearest vertex).

## Open

- [ ] Operator ranks tetra1m vs full tetra vs DA3 vs TSDF (T-020).
- [ ] Knob sweep per verdict; skirt culling is separate
      (cull_mesh.py / STO-SCN-005 lineage).
