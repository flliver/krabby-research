---
xid: STO-SCN-045
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-09
depends-on: []
bd-id: krabby-auy
assignee: krabby
shipped: 2026-06-10
tasks: 12
complete: 12
---

# Unified scene-level cameras.json (schema A) — supersede comparison_views v4 + migrate runoff readers

## Summary

One scene-level `scenes/<scene>/cameras.json` (schema_version 5) holds
the full SfM pool, the selected subset, and every captured virtual
camera; the legacy `comparison_views.json` (v3/v4) is superseded and
both runoff readers work against the scene store.

## Context

Parent: [EPI-SCN-CAMERA-COMPARE](./epic.md). Operator decision
2026-06-09: schema A (one source of truth, T-023) over a derived-view
arrangement. Scope grew during investigation (T-001): both runoff
readers (`render_comparison_matrix.sh`, `rate_renders/server.py`)
hardcode the **deleted** legacy milestone layout
(`workspace/milestones/011-scene-reconstruction/data/scenes` with
`<prefix>-curated-*` variant siblings) — they are dead paths today, so
this story repoints them at the scene store
(`scenes/<scene>/pipeline-<p>/run-<r>/`). Bonus latent bug found:
`viewer.py --comparison-views` only accepts schema 3 and silently
ignores dtu's v4 file.

## Problem

Virtual cameras live in `_unsorted/comparison_views.json` (a migration
parking lot), the pool/selected information lives nowhere at scene
level, and nothing that consumes camera data agrees on a path or a
schema. The stack-ranking runoff is unrunnable.

## Design

### Schema A (`schema_version: 5`) at `scenes/<scene>/cameras.json`

```json
{
  "schema_version": 5,
  "scene": "dtu-bicycle",
  "source_run": "pipeline-matcha/run-12-dense-strong",
  "captured_from_blend": "<last harvest source>",
  "pool": {
    "filepaths":  ["<basename> × N"],
    "focals":     ["<float> × N"],
    "cams2world": ["<4x4> × N (raw SfM frame of source_run)"]
  },
  "selected_idx": ["<int> — indices into pool that fed the run"],
  "anchor_frames": ["v4-style {basename, oriented_position} — cross-run Procrustes"],
  "views": ["v4 view objects VERBATIM (virtual cameras)"],
  "variant_prefix": "carried for reader compat"
}
```

v4 `views`/`anchor_frames` carry over byte-compatible (low-risk
migration per epic Risks). New keys are additive.

### Changes

| File | Change |
|------|--------|
| `real2sim/sync_comparison_views.py` | emit schema 5: + pool (verbatim run cameras.json), + selected_idx (`--selected-frames`, default all), + scene/source_run (derived from paths), `--legacy <v4/v3 path>` seeds views when output absent |
| `real2sim/build_blender_scene.py` | accept schema 5 in `--view-camera-pose` (same path as v3/v4) |
| `real2sim/camera_viewer/viewer.py` | accept schema 3/4/5 (fixes latent v4 rejection) |
| `real2sim/render_comparison_matrix.sh` | scene-store layout: scene root `/var/krabby/scenes/<scene>`, variants auto-discovered from `pipeline-*/run-*`, views from `<scene>/cameras.json`, renders → `<scene>/comparison_renders/` |
| `real2sim/rate_renders/server.py` | `SCENES_ROOT` → scene store; scene detection = `cameras.json` schema≥5 at scene root; variants from `comparison_renders/` contents; legacy manifest read becomes optional |
| scenes repo: `dtu-bicycle/cameras.json`, `004-sky-house/cameras.json` | migrated v5 files (source runs: run-12-dense-strong / run-12-strong) |

### Migration mechanics (no new tools — dogfoods 044)

1. Build/refresh `<run>/scene.blend` with
   `--view-camera-pose _unsorted/comparison_views.json` (injects legacy
   views into `cameras_virtual`).
2. Run upgraded `sync_comparison_views.py` against that blend →
   harvests the virtual cams back + embeds pool/selected → writes
   `scenes/<scene>/cameras.json` v5.
3. Legacy `_unsorted/comparison_views.json` gets a
   `superseded_by: ../cameras.json` marker field (kept for provenance;
   T-023 single source is the new file).

## Definition of Done

- [x] `scenes/dtu-bicycle/cameras.json` v5 exists: pool=12,
      selected=12, views = {cam_ref, cam_ref_auto}; round-trip rotation
      delta 0.00000° (float32 sign-canonicalization only). 2026-06-09.
- [x] `scenes/004-sky-house/cameras.json` v5 exists: views =
      {compare_01..03} carried from legacy schema-3 file, re-anchored to
      run-12-strong frame (Procrustes 12/12, scale 0.9972). 2026-06-09.
- [x] `render_comparison_matrix.sh` rendered compare_01 × 5 variants
      on 004-sky-house from the scene store (anchor residuals ≤3 mm,
      source run exactly 0). 2026-06-09.
- [x] `rate_renders` server boots against the scene store;
      `/api/scenes` = [004-sky-house, dtu-bicycle]; run.json flows as
      manifest; legacy rankings (rater Jeremy) read from _unsorted/.
      2026-06-09.
- [x] `viewer.py --comparison-views scenes/dtu-bicycle/cameras.json`
      injects cam_ref + cam_ref_auto, zero anchor residuals (was
      silently broken for v4). 2026-06-09.
- [x] Round-trip: build → sync verified idempotent on views (dtu:
      Δpos ≤2.6e-8, rotation 0.00000°, Δlens ≤4.8e-7). 2026-06-09.
- [x] `real2sim/README.md` updated. 2026-06-09.
- [x] **OPERATOR (T-020):** verified 2026-06-09/10 — viewer approved
      ("LOOKS GOOD"); rate_renders exercised through the settings-first
      rework + the 006 A/B (operator ranked the renders).

## Testing

### Unit / fixture tests

- [x] v5 emit on dtu fixture: pool/selected/views field-level diff vs
      legacy v4 (views equivalent to float32 round-trip).
- [x] `--legacy` seed path: emit with absent output + legacy v3 input
      (004) carries all 3 views.

### Integration

- [x] Matrix render smoke: 1 view × 5 variants, workbench engine, on
      004-sky-house (dtu has only 1 oriented variant; reference-match
      views render via --purpose).
- [x] rate_renders `/api/scenes` + `/api/scene/{dtu-bicycle,004-sky-house}`
      payloads verified.

## Out of scope

- `/camera-save` interactive capture (STO-SCN-046 — consumes this
  schema).
- Runner transform (STO-SCN-047).
- Re-rendering historical comparison matrices; old PNGs under
  `_unsorted/comparison_renders/` stay as-is.

## Implementation Notes

### What Changed

As designed. Migration dogfooded 044: built each scene's run-dir
scene.blend with `--view-camera-pose <legacy file>` (injects legacy
views into cameras_virtual), then the upgraded sync harvested them back
out into schema 5. 004's views were re-anchored from the deleted legacy
workspace frame into run-12-strong's frame by the existing Procrustes
path (scale 0.9972) — no new alignment code.

### Files Modified

- `real2sim/sync_comparison_views.py` — schema 5 emitter (+pool,
  +selected_idx, +scene/source_run derivation, --legacy seed,
  --selected-frames validation; bare-filename makedirs fix).
- `real2sim/build_blender_scene.py` — schema 5 accepted in
  --view-camera-pose.
- `real2sim/camera_viewer/viewer.py` — accepts schema 3/4/5 (fixed
  latent v4 rejection).
- `real2sim/render_comparison_matrix.sh` — scene-store rewrite:
  variants = pipeline-*/run-* ("<p>--<r>" labels), views from unified
  cameras.json (legacy fallback), renders → <scene>/comparison_renders/,
  scratch matrix_render.blend cleaned per variant.
- `real2sim/rate_renders/server.py` — SCENES_ROOT env-overridable
  (KRABBY_SCENES_ROOT, default /var/krabby/scenes), scene detection by
  unified cameras.json, variants from run dirs, run.json as manifest,
  manifest_lib import dropped, legacy rankings merged from _unsorted/.
- scenes repo: dtu-bicycle/cameras.json + 004-sky-house/cameras.json
  (v5), superseded_by markers on both legacy files, 004 run-12-strong
  scene.blend (TSDF, 2.0 GB), first matrix renders.

### Gotchas

- Quaternion component diffs up to 3.4e-5 after a blend round-trip are
  float32 + sign canonicalization, NOT pose drift — compare rotation
  angle (2·acos|q1·q2|), never raw components.
- v5 keeps the `views`/`anchor_frames` keys and `variant_prefix` so
  v3/v4 consumers' code paths port with a schema-number bump only.
- The matrix script's per-variant scratch blend must not collide with
  the canonical run-dir scene.blend — it writes matrix_render.blend
  and deletes it after the variant's renders.

## Status Notes

- 2026-06-09: Picked up by krabby. Scope note: runoff readers found
  dead (legacy layout deleted) — repoint included per epic goals.
- 2026-06-09: Implemented, migrated both scenes, all technical DoD
  verified. Holding open for operator T-020 pass (viewer virtual cams +
  rate_renders UI).
- 2026-06-10: Operator exercised rate_renders end-to-end (006 A/B ranked). Shipped.
