---
xid: STO-SCN-010
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-0lb
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.C-Manual — Manual cam_ref Placement (bicycle TSDF)
assignee: krabby
---

# T1.C-Manual — Manual cam_ref Placement (bicycle TSDF)

## Summary

Manual placement of `cam_ref` camera in `dtu-bicycle-curated-12-dense-strong/tsdf_meshes/scene_tsdf_ref.blend` to approximate MAtCha's published TSDF-reference perspective. Hand-placed at (0.404, 2.396, 1.328); rendered to `data/scenes/dtu-bicycle/reference_actual/cam_ref_render.png`.

## Context

Manual placement of `cam_ref` camera in `dtu-bicycle-curated-12-dense-strong/tsdf_meshes/scene_tsdf_ref.blend` to approximate MAtCha's published TSDF-reference perspective. Hand-placed at (0.404, 2.396, 1.328); rendered to `data/scenes/dtu-bicycle/reference_actual/cam_ref_render.png`.

This was the human's two-day workaround while the manager agent was offline. Survives `.blend` regeneration thanks to schema v4 anchor-aligned re-injection.

Evidence: journal note `journal/.../matcha-quality/notes/2026-05-06T100000-auto-localized-reference-cameras.md` Context section.

## Definition of Done

- [x] Camera placed at vantage approximating paper TSDF reference
- [x] Render produced for visual A/B comparison
- [x] Position survives .blend regeneration via comparison_views.json round-trip


## Journal Notes

Fallback that ran on the bicycle TSDF while the agent was unreachable for two days: open `dtu-bicycle-curated-12-dense-strong/tsdf_meshes/scene_tsdf_ref.blend`, eyeball-position a `cam_ref` Camera, save, render `cam_ref_render.png`. Landed at world (0.404, 2.396, 1.328), lens 25.0 mm. Exposed three structural problems → motivated schema v4 + auto-localize: regenerating via `build_blender_scene.py` wipes manual cameras (session used a different filename to dodge overwrite — "a smell"); same broken round-trip hit the `compare_01..03` A/B cameras; hand-placement doesn't scale.
_Sources: note 2026-05-06T100000-auto-localized-reference-cameras; entry 2026-05-06T101958-reference-camera-auto-positioning._

---
_Imported from legacy beads `m11-s0h` (M11 DAG re-import, 2026-06-03)._
