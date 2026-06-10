---
xid: EPI-SCN-CAMERA-COMPARE
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-09
hugs: []
tenets: []
bd-id: krabby-xep
---

# Camera tooling for pipeline comparison (recover, unify, integrate)

## Problem Statement

To judge data-transformation pipeline efficacy (the stack-ranking
runoff: render variant × view matrices, rank, Borda-aggregate), the
operator needs the M11 camera tooling chain working against the new
scene-store layout: (1) photos → posed cameras → inspectable 3D scene,
(2) interactive subset selection for pipeline input, (3) a scene
assembler that groups pool vs selected cameras with mesh outputs, and
(4) interactive virtual-camera capture for comparison views. All four
existed pre-migration (M11 workspace era) and survive in
`real2sim/`, but: the scene assembler has no camera grouping (one flat
collection), the selector's venv was lost (rebuilt 2026-06-09), virtual
cameras live in a legacy `comparison_views.json` v4 parked in
`_unsorted/`, and none of it is wired to the config-driven runner
(HUG-KRB-002: no prototypes). Without this chain there is no
quality-judgment loop over runner output.

## Goals

- `build_blender_scene.py` writes `<run-dir>/scene.blend` with two
  toggleable collections: full SfM pool cameras + the selected subset
  (driven directly by `selected_frames.json`).
- One scene-level `scenes/<scene>/cameras.json` (schema A) holding all
  pool camera poses, the selected indices, and all captured virtual
  cameras — superseding `comparison_views.json` v4.
- `/camera-save <pipeline>` interactive skill: viewport → +1 virtual
  camera into the run's `scene.blend` + regenerated unified
  cameras.json, via the Blender MCP.
- Scene-build runs as a declared transform under `run_transform.py`
  (specification.json in, results.json out).
- Runoff tooling (`render_comparison_matrix.sh`, `rate_renders/`)
  reads the unified schema.

## Non-Goals (Out of Scope)

- camera_viewer feature work beyond environment recovery + smoke
  verification (ergonomics belong to STO-SCN-032).
- New SfM backends; pool poses keep coming from MASt3R-SfM / COLMAP
  (`colmap_to_cameras_json.py`).
- Auto-localization improvements (`localize_reference_image.py` stays
  as-is).
- Rendering-quality work in the runoff itself (engine choice, CYCLES
  settings) — existing defaults carry over.

## Context

**Source:** Operator request 2026-06-09 (this session): recover the
previously-built camera tooling (cameras.json builder, subset selector,
grouped scene assembler) + the virtual-camera capability, to feed
stack-ranking runoffs over pipeline outputs. Recovery verification ran
against `dtu-bicycle/run-12-dense-strong` (scene built clean; viewer
data-layer loads; grouping gap + venv loss confirmed).

**Dependencies:**

- Scene store layout (STO-SCN-026/033) — present.
- Runner v1 (`run_transform.py`, STO-SCN-039) — shipped; needed for
  STO-SCN-047 only.
- Blender 4.5.2 CLI + Blender MCP addon (present on the Mac author
  seat).
- Related: STO-SCN-032 (camera_viewer local-inspection ergonomics) —
  sibling work, not a blocker.

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-044` | Grouped scene build — pool/selected collections, selected_frames input, run-dir scene.blend | draft | S |
| 2 | `STO-SCN-045` | Unified scene-level cameras.json (schema A) — supersede comparison_views v4 + migrate runoff readers | draft | M |
| 3 | `STO-SCN-046` | /camera-save — interactive viewport→virtual-camera capture via Blender MCP | draft | M |
| 4 | `STO-SCN-047` | Scene-build as config-driven runner transform | draft | S |

Order: 044 → 045 → 046 → 047 (046 writes the schema 045 defines; 047
wraps the 044 tool).

## Design

### Approach

Extend, don't rewrite (T-013): all four tools exist.
044 adds named Blender collections + a `--selected-frames` input to
`build_blender_scene.py`. 045 defines unified schema A —
`{schema_version: 5, pool: {filepaths, focals, cams2world},
selected_idx: [...], virtual: [v4-style view objects]}` — emitted at
`scenes/<scene>/cameras.json`, with `sync_comparison_views.py` logic
folded in and the two runoff readers repointed. 046 is a thin
interactive skill over the Blender MCP that snapshots the viewport
(view matrix + lens) into a Camera object, saves the run's
`scene.blend`, and regenerates the unified JSON. 047 declares the 044
build as a runner transform.

### Architecture

```
photos ─SfM→ run/.../mast3r_sfm/cameras.json      (pool poses; unchanged)
        camera_viewer ──→ selected_frames.json     (subset; unchanged)
        build_blender_scene (044) ──→ <run-dir>/scene.blend
              collections: cameras_pool / cameras_selected
        /camera-save (046, Blender MCP) ──→ +virtual cam in scene.blend
        unified emitter (045) ──→ scenes/<scene>/cameras.json  [schema A]
        render_comparison_matrix.sh + rate_renders/ (045) ──→ runoff
        run_transform.py (047) ──→ scene-build as declared transform
```

### Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Schema B: keep `comparison_views.json` v4 + generate `cameras.json` as a derived view | zero churn in runoff readers | two files to keep in sync; derived-state drift | Rejected (operator, 2026-06-09) |
| Schema A: one unified scene-level `cameras.json` supersedes v4 | one source of truth (T-023); readers are 2 small changes | one-time migration of existing v4 files | Selected (operator, 2026-06-09) |

## Decisions

| XID | Decision | Status | Rationale |
|-----|----------|--------|-----------|
| — | Output `.blend` lands at `<run-dir>/scene.blend` | Adopted | Operator, 2026-06-09: scene is a product of the run |
| — | Camera groups = full pool vs selected subset | Adopted | Operator confirmed interpretation, 2026-06-09 |
| — | Unified schema A supersedes comparison_views v4 | Adopted | Operator, 2026-06-09; see Alternatives |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Viewport capture via MCP differs from a real Camera object (lens/sensor mapping) | Medium | Medium | Verify against the known `cam_ref` pose/render in dtu-bicycle |
| v4 → schema A migration loses fields the runoff readers rely on | Low | High | Schema A embeds v4 view objects verbatim; readers diffed before/after on dtu + 004-sky-house |
| opencv-vs-blender quaternion convention regression (known gotcha, STO-SCN-041 notes) | Medium | High | Reuse the frame-composition recipe recorded in STO-SCN-041 |

## Success Criteria

- [ ] dtu-bicycle end-to-end: build grouped scene.blend → capture a
      virtual camera via /camera-save → unified cameras.json → matrix
      render consumes it. Operator exercises the interactive pieces
      (T-020).
- [ ] Existing v4 data (dtu-bicycle, 004-sky-house) migrated, runoff
      output unchanged before/after reader migration.
- [ ] All stories shipped.
- [ ] `real2sim/README.md` updated for the new chain.

## Milestones

| Milestone | Target Date | Actual | Status |
|-----------|-------------|--------|--------|
| Stories defined | 2026-06-09 | 2026-06-09 | done |
| Implementation complete | | | open |
| Operator verification (T-020) | | | open |

## Retrospective

_(Fill in after epic completion.)_

### What Went Well

-

### What Could Be Improved

-

### Lessons Learned

-
