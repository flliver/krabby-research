---
kind: note
captured: 2026-05-01T22:26:04-07:00
consolidated: false
tags: []
---
# Strong-alignment config eliminates garbage geometry

Result of Tier 1.a from the post-curation tuning experiment. Tested on the N=12 spatially-curated set from scene 004.

## Setup

- Same 12 picks as the curated-default run.
- Only difference: `--alignment_config strong` instead of `--alignment_config default`.
- Same source frames (1024×576), same hardware (tbeeprz, RTX 5080 / 16 GB), same downstream B1-B4 post-processing.

## Compute deltas vs default-alignment

| | default | strong |
|---|---|---|
| MAtCha wall-clock | 648 sec | 648 sec |
| MAtCha peak VRAM | 7884 MiB | 7874 MiB |
| Post-process wall | not separately tracked | 218 sec |
| Final culled mesh | 14.0 MB / 194,717 v / 343,414 polys | 14.9 MB / 200,301 v / 358,722 polys |

Both runs are essentially identical in compute profile. The strong variant ends up with ~3% more vertices and ~5% more polygons. Within noise unless the topology is meaningfully different.

## Visual verdict (per Jeremy, on inspection in Blender)

Initial glance: "roughly equal."

On closer look: **strong eliminated garbage mesh in obvious areas.** Specific failure mode improved: hallucinated/floating geometry that the default-alignment run produced in regions where SfM had noisy points. The strong config's regularization (chart-encoding norm penalty, total-variation on depth encodings, confidence-weighting on encodings) suppresses chart deformation from over-fitting to noisy SfM correspondences.

This is consistent with the Option C hypothesis from the journal — when SfM correspondences are noisy on our captures, the chart deformation MLP over-fits, producing hallucinated surfaces. Strong-alignment is essentially **"Option C lite"**: it doesn't change the multi-resolution `r` pyramid, but it adds penalty terms that suppress the same failure mode.

## Decision

**Lock `--alignment_config strong` as the operating default for all future MAtCha runs.** No measurable downside (same compute, same wall-clock, same VRAM); meaningful upside in mesh quality.

Future experiments (N=16 bracket, recapture of scenes 001/003, eventually Phase C) all proceed on strong-alignment as the new baseline.

## Implication for Option C (chart-encoding `r` truncation)

Option C is now lower-priority. The strong config already buys most of what we expected from the `r` truncation hypothesis (suppress over-fitting to noisy SfM). If we hit a quality wall later that strong-alignment alone doesn't solve, Option C is a sharper test — it changes the *architecture* of the deformation field, not just the regularization on top. But until we hit that wall, strong is enough.

## Files

- Default-alignment mesh: `data/scenes/004-sky-house-curated-12/oriented/scene_culled.blend`
- Strong-alignment mesh: `data/scenes/004-sky-house-curated-12-strong/oriented/scene_culled.blend`
- Compare side-by-side in Blender to see the garbage-elimination Jeremy observed.
