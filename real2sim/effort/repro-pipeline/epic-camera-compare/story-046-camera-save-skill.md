---
xid: STO-SCN-046
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-09
depends-on: []
bd-id: krabby-2lz
assignee: krabby
shipped: 2026-06-10
tasks: 5
complete: 5
---

# /camera-save — interactive viewport→virtual-camera capture via Blender MCP

## Summary

Operator frames a shot in a live Blender viewport (run-level
scene.blend open) and runs `/camera-save <name>`; the skill captures
the viewport as a named virtual camera (pose + true lens), rewrites
scene.blend, and regenerates the scene-level `cameras.json` (schema 5)
— making the view immediately renderable by the comparison matrix.

## Context

Parent: [EPI-SCN-CAMERA-COMPARE](./epic.md). Operator tool-4 spec
(2026-06-09). This is the unlock for ranking the kubota A/B runs:
scenes without captured comparison views can't enter the runoff
(rate_renders Q&A, 2026-06-09). Schema 5 (STO-SCN-045) is the
storage; `cameras_virtual` collection (STO-SCN-044) is the in-blend
home; `sync_comparison_views.py` is the regeneration path (T-025 —
one emitter, no parallel writer).

## Problem

Adding a comparison view today means hand-adding a Camera in Blender,
positioning it numerically, setting custom properties, saving, and
running a headless sync with the right arguments — error-prone and
undocumented. The operator wants: frame it, name it, one command.

## Design

### Flow

1. Operator opens `<run-dir>/scene.blend` in Blender (MCP addon
   connected) and frames the viewport; optionally sets the viewport
   lens (N-panel).
2. `/camera-save <name> [--purpose ab-comparison]` —
   the agent, via Blender MCP `execute_blender_code`, loads
   `real2sim/viewport_capture.py` and calls `capture(name, purpose)`:
   - derives scene/pipeline/run context from `bpy.data.filepath`
     (no pipeline argument needed — the open file IS the context;
     T-006);
   - reads the active 3D viewport's `region_3d.view_matrix.inverted()`
     for pose;
   - derives the TRUE lens from `window_matrix` (the viewport at
     `space.lens=50` is wider than a 50 mm camera — copying
     `space.lens` verbatim would render tighter than what the operator
     framed; the projection matrix is the honest source);
   - creates the Camera in the `cameras_virtual` collection with v4
     custom props (`view_purpose`, render defaults), records
     `viewport_lens` for provenance;
   - saves the .blend in place.
3. The agent then runs the headless `sync_comparison_views.py` against
   the saved blend → `scenes/<scene>/cameras.json` regenerated with
   the +1 view.
4. Report: camera name, pose, lens, updated JSON path; offer a
   matrix render of the new view.

### Changes

| File | Change |
|------|--------|
| `real2sim/viewport_capture.py` | new — capture logic (exec'd in live Blender via MCP) |
| `.claude/commands/camera-save.md` | new — the skill |
| `real2sim/README.md` | document the flow |

## Definition of Done

- [x] Capture on a live scene.blend produces a camera matching the
      framed viewport — verified NUMERICALLY (pose = analytic
      expectation; captured-vs-Procrustes-injected camera: 0.000000 m /
      0.00000° / identical lens). Lens question settled empirically:
      viewport space.lens=50 → true 25 mm (projection-matrix derivation
      is mandatory; copying space.lens would halve the framed width).
      2026-06-09.
- [x] `cameras.json` regenerated with the new view (3 views incl.
      sc046_test); matrix render succeeded for matcha--12-dense-strong.
      2026-06-09 (test artifact cleaned up after verification).
- [x] Re-running with the same name updates the existing view
      (action: 'updated' observed live). 2026-06-09.
- [x] **OPERATOR (T-020):** 2026-06-10 — operator framed
      overhead-grass-quality on 006, ran /camera-save, A/B rendered,
      verified in rate_renders and ranked ("looks good").
- [x] `real2sim/README.md` updated (viewport_capture entry). 2026-06-09.

## Out of scope

- Kubota scene.blend prerequisites (orient/condition chain for the new
  dense runs) — tracked separately; needed before the kubota T-020
  pass.
- Auto-localization of captures (localize_reference_image.py exists).

## Status Notes

- 2026-06-09: Picked up by krabby per operator "proceed with
  /camera-save".
- 2026-06-09: Built + live-verified end-to-end on dtu (Blender GUI +
  MCP, server started via --python-expr timer). Two false-alarm
  "mismatches" were instrument errors (letterboxed camera view eyeball;
  pixel-diff across different meshes) — numeric comparison is the
  instrument. rv3d.update() added for programmatic-framing robustness.
  Remaining: README + operator T-020 on a kubota scene (needs the
  kubota orient/condition chain first).
- 2026-06-10: Operator ran the full loop on 006 and ranked the A/B. Shipped.
