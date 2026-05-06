---
kind: note
captured: 2026-05-06T10:00:00-07:00
consolidated: false
tags: [reference-validation, auto-localization, mast3r-sfm, comparison-views, schema-v4]
---
# Auto-localized reference cameras + comparison_views.json round-trip

Two pieces of work landed together. Phase 0 closed a long-standing gap in the
manually-placed comparison-camera workflow; Phase 1 attempted (with mixed
results) to replace the manual placement with SfM-extend auto-localization.

## Context

The 2026-05-04 planning pivot called for placing a Blender camera that
matches MAtCha's published reference image perspective, so we can render our
TSDF and adaptive-tetrahedralization meshes from that vantage and visually
A/B/C-compare against the paper. While I was unreachable for two days, Jeremy
hand-placed `cam_ref` in `dtu-bicycle-curated-12-dense-strong/tsdf_meshes/scene_tsdf_ref.blend`
and rendered a viewport image to `data/scenes/dtu-bicycle/reference_actual/cam_ref_render.png`.

Two structural problems became visible:

1. **The manual placement doesn't survive .blend regeneration.**
   `build_blender_scene.py` did support a `--view-camera-pose <json>`
   anchor-aligned re-injection (schema v3), but only for ONE view at a time
   (chosen by `--view-name`). The pre-existing `compare_01..03` cameras for
   004-sky-house had the same gap — they were one-way out (sync) and only
   single-view back in.
2. **Reference images at `reference_images/` were 30-byte placeholder
   stubs** — never actually downloaded. The real PNGs landed at
   `data/scenes/dtu-bicycle/reference/{tsdf_multires,adaptive_tetra}.png`
   under a different naming scheme. PLAN.md still references the old path.

## Phase 0: comparison_views.json schema v3 → v4 + bidirectional injection

Schema bump is small and backward-compatible:

```diff
+ "purpose": "ab-comparison" | "reference-match" | ...   (default ab-comparison)
+ "matches_reference_images": [paths]                   (only on reference-match)
+ "render_resolution": [w, h]                            (optional)
+ "render_engine": "CYCLES" | "BLENDER_WORKBENCH" | ...  (optional)
+ "auto_localized": bool                                 (optional)
+ "localization_method": "manual" | "mast3r_sfm_extend"  (optional)
```

Three files modified:

- `workspace/sync_comparison_views.py` — reads each new field from a Blender
  custom property on the camera (e.g., `cam["view_purpose"] = "reference-match"`),
  emits schema v4. Also fixed a pre-existing bug where any non-`views`
  top-level fields (`variant_prefix`) were dropped on rewrite.
- `workspace/build_blender_scene.py` — for schema v3/v4, injects ALL views
  in the JSON (not just one). `--view-name` is now "which view is the active
  scene camera," not "which view to inject." Each view becomes its own Blender
  Camera; per-view metadata is reattached as custom properties so the next
  `sync_comparison_views.py` round-trips it back.
- `workspace/render_comparison_matrix.sh` — added `--purpose <filter>`
  argument; default is `ab-comparison` so reference-match views don't get
  rendered into the per-view A/B matrix unless explicitly requested.

**Round-trip validation** (against `scene_tsdf_ref.blend`, regenerated to
`/tmp`):
- 12/12 anchors matched, Procrustes residuals all 0 (same variant).
- `cam_ref` re-injected at exactly the manual position (0.404, 2.396, 1.328).
- Round-trip back through sync produced byte-identical JSON modulo ~1e-8
  quaternion noise (rot-mat → quat → rot-mat float precision).

## Phase 1: SfM-extend auto-localization

`workspace/localize_reference_image.py` (new) implements the full pipeline:

1. Convert reference PNG → faux-RGB JPG (greyscale becomes 3-channel-replicated).
2. Stage a sandbox dir on tbeeprz with 12 relative-symlinked source frames +
   1 ref JPG named `_DSC9999_ref.JPG` (sorts last alphabetically → index 12).
3. Run `python train.py --sfm_only --image_idx 0..12` in `matcha-build`
   container — ~104 sec on RTX 5080, 13 frames, RC=0.
4. Pull resulting `cameras.json` back; Procrustes-align the new 12-camera
   centers to the original 12 (Umeyama with scale).
5. Apply that similarity transform to the 13th cam pose → reference pose in
   original SfM frame.
6. Apply `world_orient` (R, z_shift from `oriented_cameras.json`) → world frame.
7. Convert to OpenCV-convention quat + position; lens_mm derived from
   focal_px / 512 × 36.
8. Upsert into `comparison_views.json` with `purpose=reference-match`,
   `auto_localized=true`, `localization_method=mast3r_sfm_extend`.

**Result on bicycle (TSDF reference):**

| metric | value |
|---|---|
| Procrustes scale | 1.0156 |
| Procrustes residuals (12 shared cams) | max 1.3 cm, mean 0.4 cm |
| Auto-localized position (world) | (0.602, 3.112, 1.381) |
| Manual `cam_ref` position (world) | (0.404, 2.396, 1.328) |
| Position delta (auto vs manual) | **0.745 m** |
| Rotation delta (auto vs manual) | **4.55°** |
| Auto-derived lens | 34.07 mm (focal_px 484.56) |
| Manual-set lens | 25.0 mm |

A 3-up render comparison is at
`./2026-05-06T100000-cam-ref-auto-localize-three-way-compare.png`
(reference image vs manual cam_ref render vs auto cam_ref_auto render).

## Findings

- **The SfM math worked.** Sub-cm Procrustes residuals across 12 anchor
  cameras say the new SfM frame and original SfM frame are nearly identical
  (small 1.6% scale difference). The reference image converged into the
  reconstruction successfully despite being effectively greyscale (mean
  |R-G|=0.09 / 255).
- **Neither manual nor auto perfectly matches the paper's reference framing.**
  The published render appears to use a much wider FOV from a higher vantage
  than what the source-photo coverage admits. Both the manual placement
  (visual approximation) and the auto placement (SfM-derived) end up at
  similar viewpoints — close to each other (0.74 m / 4.55°) but neither is
  visibly the same vantage as the published render.
- **Shared-focal SfM is the likely reason.** MASt3R-SfM converged on
  identical focals (484.56 px) for all 13 cameras. If the paper's reference
  was rendered with a much wider FOV, SfM compromises by averaging — placing
  the auto camera at a focal/distance combo that explains the matches but
  isn't the true paper-render parameters.
- **Greyscale rendering didn't trip MASt3R.** Pairwise correspondence
  succeeded, all 13 cameras converged. The earlier worry (RGB-trained
  encoder vs greyscale input) didn't materialize — channel-replication
  was sufficient.

## Next steps (open)

1. **Decide whether 0.74 m / 4.55° is "good enough."** For visual A/B/C
   comparison purposes, the answer might be yes — the user can render the
   bicycle scene from the auto camera and visually compare to the paper.
   The mismatch is about *paper render parameters*, not *which camera best
   represents that vantage given our reconstruction*.
2. **PnP localization (variant 2 in `2026-05-01T174651-...`).** Doesn't
   share focal estimation with the existing 12 cameras — the reference's
   focal is solved independently. Should give a more faithful pose if the
   paper render really is at a different focal. Cost: ~200 lines of new
   code (descriptor extraction, matching, RANSAC).
3. **Differentiable photometric refinement.** Render TSDF mesh from
   auto-localized init, compute SSIM loss against reference image, gradient-
   descend on pose. Sub-pixel accuracy. Heaviest option; only worth it if
   tighter alignment is required for the validation use case.
4. **Repeat for adaptive-tetrahedralization reference.** The auto-localize
   script already works on any single PNG. Run it with `--reference-image
   .../adaptive_tetra.png --reference-name cam_ref_auto_tetra` to land a
   second auto camera. Compare both auto cameras; if they're near-coincident,
   the published TSDF and tetra renders share a vantage (single physical
   camera answers both A/B/C cases).

## Files added/modified

- M `workspace/sync_comparison_views.py` — schema v4, custom-property round-trip,
  preserve-prev-fields fix.
- M `workspace/build_blender_scene.py` — multi-view injection.
- M `workspace/render_comparison_matrix.sh` — `--purpose` filter.
- A `workspace/localize_reference_image.py` — Phase 1 auto-localization script.
- M `data/scenes/dtu-bicycle/comparison_views.json` — schema v4, both
  `cam_ref` (manual) and `cam_ref_auto` (auto) views.
- A `journal/.../matcha-quality/notes/2026-05-06T100000-cam-ref-auto-localize-three-way-compare.png`

The original `reference_images/` stub directory at the milestone root is
unchanged; the path in PLAN.md (`reference_images/`) is stale and should be
updated to `data/scenes/dtu-bicycle/reference/` next time the plan is touched.
