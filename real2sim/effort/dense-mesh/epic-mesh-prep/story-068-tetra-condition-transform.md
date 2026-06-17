---
xid: STO-SCN-068
parent: ./epic.md
kind: story
effort: scn
size: M
status: deferred
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

## Resolution — deferred 2026-06-15 (reconcile dangling child of a shipped epic)

The **deliverable shipped**: `real2sim/tetra_condition.py` exists and produced the
deliverable-scale mesh documented above (433k verts / 1.0M tris, 36× smaller), in the runoff
as `matcha--8-strong-tetra1m`. The remaining open items are **not** standalone work:

- **Mesh-quality knob-sweep** (target_tris / taubin / smarter color transfer) is exactly the
  conditioning concern now owned by the **active EPI-SCN-MESH-CONDITION** — specifically
  STO-SCN-133 (mesh_res, content-addressed) and STO-SCN-136 (cull distant/sky junk). The
  standalone `tetra_condition.py` decimation role has been **absorbed into the v4 pipeline-studio
  task catalog** (STO-SCN-069/070), so this script-level story no longer has independent scope.
- **Operator ranking (T-020)** has been carried by the live runoff ranking sessions (DA3-24 /
  matcha verdicts, 2026-06-15) under the mesh-condition epic, not this script story.

Deferred (matching deferred siblings STO-SCN-011/012 under this same shipped epic) rather than
self-shipped — per T-020 I do not claim an operator sign-off specific to tetra1m. The mesh-prep
epic (EPI-SCN-MESH-PREP) is now free of dangling in-progress children.
