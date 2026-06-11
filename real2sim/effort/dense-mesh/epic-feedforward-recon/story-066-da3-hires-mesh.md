---
xid: STO-SCN-066
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-66p
---

# DA3 hi-res mesh — process_res sweep to close the fidelity gap

## What we did (2026-06-10)

The 504-default fused mesh (213k verts) is visibly coarser than the
splat render and far coarser than matcha's 30M-vert meshes. Swept
process_res upward, depths-only (`nogs` mode — the gaussian head OOMs
16 GB above 504):

| process_res | gs head | result |
|---|---|---|
| 1008 | off | OOM (14.3 GiB alloc, conv2d) — over the fleet ceiling |
| 756 | off | **fits**: 17.5 s, 14.5 GiB peak; depths 567×756 |

Fusion at voxel-frac 0.0027: **625,717 verts / 1,081,393 tris (2.9×
the 504 baseline)** → `run-8-giant-hires756`, rendered via the
standard path, in the runoff beside the 504 mesh + matcha variants.

## Where the code is

`/opt/krabby-tools/da3_infer_gs.py` (krabby-da3:0.3, baked per the
tooling-provenance policy; `nogs` arg = depths-only mode) +
`da3_tsdf_mesh.py`.

## Open

- [ ] Operator verdict: is 756-mesh fidelity acceptable, or do we need
      >1008 (requires >16 GB GPU, view-chunked inference, or
      DA3-Streaming)?
- [x] 756 run spec'd, measured, rendered, in runoff.
