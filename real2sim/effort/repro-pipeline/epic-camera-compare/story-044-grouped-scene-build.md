---
xid: STO-SCN-044
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
date: 2026-06-09
depends-on: []
bd-id: krabby-7kg
assignee: krabby
---

# Grouped scene build — pool/selected camera collections, selected_frames input, run-dir scene.blend

## Summary

`build_blender_scene.py` produces `<run-dir>/scene.blend` whose cameras
are split into two named, independently-toggleable Blender collections —
the full SfM pool and the curated subset — with the subset driven
directly by the viewer's `selected_frames.json`.

## Context

Parent: [EPI-SCN-CAMERA-COMPARE](./epic.md). Recovery verification
(2026-06-09, dtu-bicycle run-12-dense-strong) confirmed the builder runs
clean on Blender 4.5.2 but links every object into the single default
collection `Collection` (26 objects flat) — no pool/selected
distinction, no easy visibility toggling. The subset today reaches the
builder only implicitly (the run's `cameras.json` is already the
selected 12); the full pool never appears in the scene at all.

## Problem

Pipeline-efficacy inspection needs to see, in one `.blend`: which
cameras the SfM pool contained, which subset fed the pipeline run, and
the run's mesh output — with each camera group toggleable in the
Outliner. Today's output is a flat collection of selected-only cameras,
saved wherever `--output` points instead of canonically in the run dir.

## Design

### Approach

Extend `build_blender_scene.py` (T-013 — no rewrite):

1. **Collections.** Create `cameras_pool` and `cameras_selected`
   collections under the scene collection; `meshes` for imported
   geometry. Each camera object + its image-plane child links into its
   group's collection instead of `bpy.context.collection` (today's
   lines 202/243/425/533/550).
2. **`--selected-frames <selected_frames.json>` (optional).** When
   given with a *pool*-sized `--cameras-original`, `selected_idx`
   partitions cameras: members → `cameras_selected`, rest →
   `cameras_pool`. When absent, all cameras land in `cameras_selected`
   (today's behavior, since run-level cameras.json is already the
   subset) and `cameras_pool` is created empty.
3. **Default output.** When `--output` is omitted and the inputs live
   under a `transform-NN-*/data/` dir, default to
   `<run-dir>/scene.blend` (walk up to the `run-*` dir). Explicit
   `--output` still wins.
4. Existing virtual-camera re-injection (comparison views) keeps
   working; those cameras link into a third collection
   `cameras_virtual` (forward-compatible with STO-SCN-045/046).

### Changes

| File | Change |
|------|--------|
| `real2sim/build_blender_scene.py` | collections, `--selected-frames`, run-dir default output |
| `real2sim/README.md` | invocation docs for the new flags/output |

## Definition of Done

- [x] Headless build on dtu-bicycle run-12-dense-strong yields
      `<run-dir>/scene.blend` with collections `cameras_pool`,
      `cameras_selected` (12 cams), `cameras_virtual` (cam_ref +
      cam_ref_auto), `meshes` — verified by headless inspection script
      (`/tmp/inspect_grouping.py`), not eyeball. 2026-06-09.
- [x] With `--selected-frames` (synthetic 3-of-12 fixture), the
      partition matches `selected_idx` exactly: [0,5,11] →
      cam_001/006/012 selected, 9 in pool. 2026-06-09.
- [x] No regression: omitting the new flag reproduces today's object
      set (12 cams + 12 planes + mesh + sun) apart from collection
      structure. 2026-06-09.
- [ ] **OPERATOR (T-020):** open
      `scenes/dtu-bicycle/pipeline-matcha/run-12-dense-strong/scene.blend`
      and confirm Outliner group toggling (cameras_selected /
      cameras_virtual / meshes) works as expected.
- [x] `real2sim/README.md` updated. 2026-06-09.

## Testing

### Unit / fixture tests

- [x] Headless inspection: collection names, membership counts,
      camera→collection mapping on the dtu fixture (4 test cases, all
      PASS — see Implementation Notes).
- [x] `selected_idx` out-of-range entries fail loudly: real dtu
      `selected_frames.json` (n_pool=194 indices) against the 12-cam
      run cameras.json → `ERROR: ... out of range ... (pool mismatch?
      n_pool=194)`, no .blend written.

### Integration

- [x] Full chain on dtu-bicycle: run cameras.json + comparison_views
      v4 → grouped scene.blend auto-placed in the run dir (path
      derived via run-* ancestor walk), Procrustes 12/12 anchors,
      scale=1.0000.

## Out of scope

- Unified scene-level cameras.json emission (STO-SCN-045).
- Viewport capture (STO-SCN-046).
- Runner transform wrapper (STO-SCN-047).

## Implementation Notes

### What Changed

As designed, plus: the sun light lives in `meshes` (not loose in the
master collection) so *nothing* links to `bpy.context.collection`
anymore — the master collection holds only the four named children.
Image-plane children link into the same collection as their parent
camera, so toggling a group hides cameras and their thumbnails
together.

Test matrix (Blender 4.5.2 headless, dtu-bicycle run-12-dense-strong):

| # | Case | Result |
|---|------|--------|
| 1 | Regression (no flag, explicit --output) | PASS — 12 cams + 12 planes all in `cameras_selected`, pool empty |
| 2 | Partition (synthetic selected_idx [0,5,11] of 12) | PASS — exact partition |
| 3 | Out-of-range idx (real 194-pool file vs 12 cams) | PASS — loud SystemExit, no .blend |
| 4 | Deliverable: derived run-dir output + v4 comparison views | PASS — `<run>/scene.blend`, `cameras_virtual` = {cam_ref, cam_ref_auto} |

### Files Modified

- `real2sim/build_blender_scene.py` — `get_or_create_collection()`,
  `link_into()`, `derive_output_path()`; `--selected-frames` with
  range validation; 6 link points rerouted; docstring.
- `real2sim/README.md` — tree annotation for the new surface.

### Gotchas

- Importers auto-link meshes into the active collection; `link_into()`
  must unlink from *all* `users_collection` first or the object shows
  up twice in the Outliner.
- `derive_output_path()` resolves through the scene-store symlink
  (`/var/krabby/scenes` → `/Volumes/Archives-01/krabby/scenes`)
  because it walks `os.path.abspath` — harmless, but paths printed in
  logs show the physical volume.

## Status Notes

- 2026-06-09: Minted from operator request; picked up by krabby
  immediately (recovery verification already done this session).
- 2026-06-09: Implemented + 4/4 headless tests PASS. Deliverable
  scene.blend written to dtu run dir. Blocked only on operator T-020
  toggle check; proceeding to STO-SCN-045 meanwhile.
