---
kind: note
captured: 2026-05-06T10:19:58-07:00
consolidated: true
tags: []
---
# Auto-positioning the reference camera: data-model + process changes

A complement to the sister results-note
(`2026-05-06T100000-auto-localized-reference-cameras`). That one captures
the *outcome* — pose deltas, render comparisons, what worked and what
didn't. This one captures the *shape* — what the data model looked like
before, what changed, and why the pipeline ended up the way it did.

## The starting friction

Phase C (validate against MAtCha's published renders) needed exactly one
new artifact: a Blender camera at the same vantage as the paper's TSDF
hero shot, so we could render our own TSDF mesh from there and visually
compare.

Manual placement was the fallback path: open
`scene_tsdf_ref.blend`, eyeball-position a `cam_ref` Camera object, save.
That worked once. Three structural problems followed:

1. **Regeneration wipes manual cameras.** `build_blender_scene.py`
   rebuilds `scene_tsdf.blend` end-to-end any time we re-orient or swap
   meshes (TSDF ↔ tetra). The OpenCode session that placed `cam_ref`
   chose a different filename (`scene_tsdf_ref.blend`) precisely because
   it knew the production `.blend` would overwrite. That's a smell.
2. **The same gap applied to the existing A/B-comparison cameras.**
   `compare_01..03` for 004-sky-house had been hand-placed and synced to
   `comparison_views.json` (one-way), but were never re-injected on
   regeneration. Nobody had hit this hard yet because `004` `.blend`s
   weren't being regenerated end-to-end.
3. **No way to scale this.** Phase C wanted reference cameras for
   eventually multiple scenes and (potentially) multiple references per
   scene. Hand-placement is the throughput ceiling.

## The data-model gap

Three camera *kinds* coexisted in our world but only one had a
round-trip:

| kind | example | source | how it gets into a regenerated .blend |
|---|---|---|---|
| MAtCha-SfM keyframes | `cam_001..cam_012` | `mast3r_sfm/cameras.json` | re-injected from `cameras.json` (always) |
| A/B-comparison cameras | `compare_01..03` | manual + `sync_comparison_views.py` | **wasn't** re-injected (round-trip broken) |
| Reference-validation cameras | `cam_ref` | manual (this session) | **wasn't** re-injected (same gap) |

`build_blender_scene.py` already had logic for "inject one view from
`comparison_views.json` via anchor-aligned Procrustes," driven by
`--view-camera-pose` + `--view-name`. The function existed; what was
missing was the call-site pattern (always inject, multi-view,
purpose-tagged). The file structure already supported the right answer;
the codepath had to be widened to use it.

## Why one file, not two

Initial instinct: introduce a third file class
(`reference_cameras.json`) alongside `cameras.json` (SfM output) and
`oriented_cameras.json` (orient metadata). Cleaner separation between
"reference comparison cameras" and "A/B comparison cameras."

User pushback (correct): the A/B and reference cameras are the *same
kind of thing* — manually-positioned auxiliary cameras. The difference
is purpose, not identity. One file with a `purpose` discriminator beats
two files. So the schema bump is small and additive:

- Schema **v3 → v4**.
- Single `views` array.
- Optional `purpose` field per view (`ab-comparison` | `reference-match`).
- Defaults preserve v3 semantics (any view without `purpose` is
  treated as `ab-comparison`).
- Optional metadata fields on `reference-match` views:
  `matches_reference_images`, `render_resolution`, `render_engine`,
  `auto_localized`, `localization_method`.

The schema bump is silent (consumers handle missing fields) which means
the existing 004-sky-house workflow keeps running without change.

## Round-trip mechanics: Blender custom properties

The harder problem was: how does
"`auto_localized: true, matches_reference_images: [...]`" survive a
sync→build→sync cycle? sync reads what's in the .blend; build writes
into the .blend. The only persistent attachment surface on a Blender
Camera object is its **custom properties dict**.

So:

- `sync_comparison_views.py` reads each non-`cam_NNN` camera's custom
  props (`view_purpose`, `matches_reference_images`,
  `render_resolution`, `render_engine`, `auto_localized`,
  `localization_method`) and emits them into the JSON view.
- `build_blender_scene.py` reads each view's metadata and re-attaches
  it as custom properties on the new Blender Camera at injection time.

Round-trip validation against the bicycle: byte-identical JSON modulo
~1e-8 floating-point noise on the rotation quaternion (rot-mat → quat
→ rot-mat). Acceptable.

A side fix: pre-existing bug where `sync_comparison_views.py` dropped
top-level fields it didn't own (e.g., `variant_prefix` written by
`render_comparison_matrix.sh`) when rewriting. Fixed in passing; would
have re-broken the bicycle's matrix render at next sync.

## Build-path widening

`build_blender_scene.py` was structured as "single view, possibly
selected by `--view-name`." The change was small:

- Schema v3/v4 case: inject **all** views in the JSON.
- `--view-name` semantics shift: it selects the **active** scene
  camera at .blend open, not which view to inject.
- Procrustes anchor alignment is computed **once** (anchors are
  shared), then applied per-view inside the loop.
- Per-view metadata reattaches as custom properties (round-trip).
- Each view's name in the JSON becomes the Blender Camera object name.

Schema v1 and v2 (legacy, single-view) paths preserved unchanged.

## render_comparison_matrix.sh: purpose filter

A consequence of "all views in the JSON inject by default": the matrix
renderer had to learn to filter. New `--purpose` arg:

- Default: `ab-comparison` (preserves existing 3×N matrix behavior).
- `any`: render every view (including `reference-match`).
- `<other>`: render only views with that purpose.

The filter is applied at view enumeration in the shell wrapper,
not inside Python — keeps the change local.

## The auto-localization pipeline

What became `localize_reference_image.py`:

```
local                                 tbeeprz                          local
─────                                 ───────                          ─────
PNG → faux-RGB JPG                    sandbox: 12 relative-symlinks    pull cameras.json
(channel-replicate                    + 1 ref JPG named so it sorts    Procrustes-align
 greyscale → 3 channels)              last (_DSC9999_ref.JPG)          new12 → orig12
push to remote sandbox       →        run train.py --sfm_only          apply same to ref
                                      --image_idx 0..12 ~104s          apply world_orient
                                      on RTX 5080                      → world frame
                                                                       quat(wxyz) + position
                                                                       upsert comparison_views.json
                                                                          purpose=reference-match
                                                                          auto_localized=true
                                                                          localization_method=mast3r_sfm_extend
```

Three design choices worth flagging:

1. **Relative symlinks, not absolute.** The host
   `/home/jeremy/outposts/krabby/data/...` mount is `/data/...` inside
   the matcha-build container. Absolute symlinks point at
   container-invisible paths and break. Relative symlinks resolve
   correctly in both contexts. (Caught after one wasted SfM run that
   bailed on `FileNotFoundError`.)
2. **Sandbox is `<data-root>/sfm-ref-localize/<scene>/`, not under
   the scene's own dir.** Keeps the original
   `<scene>/mast3r_sfm/cameras.json` immutable; the SfM-with-ref
   output lives in a separate sibling. Anything else that consumes
   `mast3r_sfm/cameras.json` (other tooling, future training runs)
   sees an unchanged file.
3. **`--image_idx 0 1 2 ... 12`, not `--n_images 13`.** Both work for
   13 input images, but `--image_idx` is explicit and order-deterministic.
   `--n_images` does constant-spacing sampling — fine when the count
   matches the pool, but a foot-gun if the pool ever has more frames
   than expected (would silently drop the reference).

## Where this sits relative to the alternatives

We picked SfM-extend (variant 1 of three documented in note
`2026-05-01T174651-sliding-window-sfm-and-the-keyframe-localization-alternative`)
because the existing wrapper supports it directly. The other two
variants stay in design space:

- **PnP localization (variant 2).** The principled answer if shared-
  focal SfM fails on rendered references. Doesn't disturb the
  original 12-camera reconstruction. The 0.74 m residual position
  delta in the bicycle test is the symptom that would push us here.
- **Differentiable photometric refinement (variant 3).** Pixel-perfect.
  Heaviest. Probably overkill for "visual A/B/C comparison."

## What changed on disk

```
M  workspace/sync_comparison_views.py        +purpose round-trip, +preserve-prev-fields
M  workspace/build_blender_scene.py          +multi-view injection, schema v4
M  workspace/render_comparison_matrix.sh     +--purpose filter
A  workspace/localize_reference_image.py     +SfM-extend pipeline (404 lines)
M  data/scenes/dtu-bicycle/comparison_views.json   v3→v4, +cam_ref +cam_ref_auto
```

Schema bump touched 3 scripts, and the round-trip mechanism touched 2.
Adding the auto-localizer was new code, not new infrastructure — by the
time it landed, the data path it needed already existed.
