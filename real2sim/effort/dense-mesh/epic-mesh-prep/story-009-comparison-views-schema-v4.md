---
xid: STO-SCN-009
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-6ra
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.C-Schema — comparison_views.json Schema v4 + Bidirectional Injection
---

# T1.C-Schema — comparison_views.json Schema v4 + Bidirectional Injection

## Summary

Schema v3 → v4 bump on `comparison_views.json` adding fields: `purpose` (ab-comparison | reference-match), `matches_reference_images`, `render_resolution`, `render_engine`, `auto_localized`, `localization_method`.

## Context

Schema v3 → v4 bump on `comparison_views.json` adding fields: `purpose` (ab-comparison | reference-match), `matches_reference_images`, `render_resolution`, `render_engine`, `auto_localized`, `localization_method`.

Three files modified: `sync_comparison_views.py` (read custom Blender props, emit v4, fix dropped-fields bug), `build_blender_scene.py` (inject ALL views per scene, not just one), `render_comparison_matrix.sh` (--purpose filter).

Round-trip validated: 12/12 anchor cameras matched, sub-cm Procrustes residuals, byte-identical JSON modulo float precision.

Evidence: commit `4bf02c5` (M11: comparison_views.json schema v4 + auto-localize reference cameras); journal note `journal/.../matcha-quality/notes/2026-05-06T100000-auto-localized-reference-cameras.md` Phase 0 section.

## Definition of Done

- [x] Schema v4 backward-compatible with v3
- [x] Bidirectional sync (Blender ↔ JSON) round-trips losslessly
- [x] All views in JSON injected on rebuild (not just one)


## Journal Notes

`comparison_views.json` schema v3→v4 landed 2026-05-06: backward-compatible — one `views` array with an optional `purpose` discriminator (`ab-comparison`|`reference-match`, default `ab-comparison`), plus optional reference-match fields (`matches_reference_images`, `render_resolution`, `render_engine`, `auto_localized`, `localization_method`). Key call (on user pushback): one file with a purpose discriminator, not a separate `reference_cameras.json` — both are manually-positioned auxiliary cameras differing only in purpose. Round-trip persistence via Blender Camera custom properties: `sync_comparison_views.py` reads them out, `build_blender_scene.py` re-attaches on injection; bicycle round-trip byte-identical modulo ~1e-8 quaternion noise. Touched 3 scripts; included a side-fix where sync had dropped unowned top-level fields like `variant_prefix`.
_Sources: entry 2026-05-06T101958-reference-camera-auto-positioning; notes 2026-05-06T100000-…, 2026-05-06T101958-auto-positioning-…._


## Handoff Notes

Shipped in release `4bf02c5` (2026-05-06) alongside: `colmap_to_cameras_json.py` (COLMAP→our-format w/ auto-orient), `apply_existing_orientation.py` (applies orient_mesh R+z_shift to alternate meshes for TSDF), `render_comparison_matrix.sh --mesh-source {oriented|tsdf}`, `rate_renders/server.py` multi-scene discovery via `variant_prefix` + manifest-collision fix, `manifest_lib.py::variant_dir()` refusing ambiguous suffix matches. Bicycle (Mip-NeRF 360 / DTU) stood up as a second scene (`variant_prefix: "dtu-bicycle"`), its `comparison_views.json` now schema-v4 with the two reference-match views populated. (handoff-2026-05-02-2210.md)

---
_Imported from legacy beads `m11-ep4` (M11 DAG re-import, 2026-06-03)._
