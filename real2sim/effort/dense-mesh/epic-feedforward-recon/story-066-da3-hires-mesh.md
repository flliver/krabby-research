---
xid: STO-SCN-066
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-66p
tasks: 0
complete: 0
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

- [x] 756 run spec'd, measured, rendered, in runoff.
- [~] Operator verdict on 756 vs >1008 fidelity — the 756 mesh (625k verts / 1.08M tris) is
      **good enough for the runoff** and the DA3 path is operator-validated (DA3-24, STO-SCN-127).

## Closeout — shipped 2026-06-15 (DES-SCN-DENSE-MESH closeout)

The process_res sweep is **done and measured**: 756 fits the 16 GB fleet ceiling (17.5 s, 14.5
GiB peak) and yields a deliverable-scale 1.08M-tri mesh, in the runoff. The fidelity-gap goal of
the story is answered — 756 is the practical ceiling on current hardware.

**Deferred T1 enhancement (recorded, not a T2 item):** pushing beyond 1008 needs >16 GB GPU,
view-chunked inference, or DA3-Streaming — a *reconstruction* fidelity improvement (T1), out of
scope for this closing milestone since 756 suffices for the runoff. Revisit only if a contract
deliverable demands higher feed-forward fidelity. (Reconstruction, not conditioning — does
**not** belong in DES-SCN-COND-USD / T2.)
