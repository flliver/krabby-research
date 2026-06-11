---
xid: STO-SCN-067
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-wo3
---

# Tetra mesh as runoff render-variant — third mesh flavor in the comparison

## What we did (2026-06-10)

Operator inspection of 006's scene_compare.blend revealed the runoff
had never compared MAtCha's TETRA branch: the matcha variants render
the TSDF-extracted mesh (sharp, HOLES) while `oriented_tetra.ply`
(6.3M verts, vertex-colored) is continuous AND sharp — visually the
best mesh of all flavors, with skirt artifacts at scene edges as its
known cost.

Added lightweight RENDER-VARIANT runs (no transforms of their own;
`run.json.source_run` points at the reconstruction):

- `pipeline-matcha/run-8-dense-strong-tetra` (anchor residuals 0.0000)
- `pipeline-matcha/run-8-strong-tetra` (max 0.0020 m)

Both rendered from the saved view via `build_blender_scene.py`
directly; sidecars carry mesh_source=tetra + the SOURCE run's
transform parameters. Runoff now ranks SIX variants for
006/overhead-grass-quality: tsdf×2, tetra×2, da3 504 + 756.

## Convention established

A *render-variant* run = renders/ + run.json{source_run} + sidecars,
zero transform dirs. rate_renders needs no changes (variants are
defined by render presence).

## Open

- [ ] Operator ranks the six-way comparison (T-020).
- [ ] If tetra wins: cull/decimate conditioning (oriented_500k_colored
      _culled) for deliverable use + extend render matrix with a
      'tetra' mesh-source instead of the direct-call path.
