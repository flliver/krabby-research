---
kind: note
captured: 2026-05-01
consolidated: true
tags: [b5, frame-curation, mast3r-sfm, blender, workflow]
---

# Spatial frame curation via MASt3R-SfM (no mesh required)

## The reframe

Original B5 plan: build a contact-sheet picker (visual grid of candidate frames, click-to-pick UI, ~2 hr of tool-building). Cost was the main objection.

Better plan: **use MASt3R-SfM by itself to give us 3D camera poses for a candidate pool, drop those into Blender as Camera objects + textured image planes, hand-pick by spatial intuition.** No mesh, no contact-sheet UI, much smaller scope.

## Why this works

MASt3R-SfM produces two things — camera poses and a sparse point cloud. We've been thinking of the point cloud as the deliverable, but **for curation, the cameras are the deliverable**. The points can be ignored.

The cameras come out of pairwise pointmaps + global gradient-descent (matching loss in 3D, then 2D reprojection refinement). The poses end up in a coherent shared coordinate system as a byproduct of that optimization. No surface reconstruction, no dense MVS, no triangulation pass needed beyond what SfM does internally.

## The proposed workflow

```
Source video
   │
   ▼  (ffmpeg, motion-aware or even-time sampling — see "sampling" below)
~60 candidate frames at 1024×576
   │
   ▼  (standalone MASt3R-SfM call inside the krabby-matcha container)
cameras.json with ~60 poses + sparse points (ignored)
   │
   ▼  (extended build_blender_scene.py, mesh-import path skipped)
~60 Blender Camera objects + textured image planes at each cam position
   │
   ▼  (Jeremy inspects spatially in Blender, picks 12)
List of 12 chosen frame indices
   │
   ▼  (re-extract those exact 12 frames, run full MAtCha)
12-frame MAtCha mesh built from viewpoint-curated frames
```

## What we need to build (small)

- A wrapper that runs MASt3R-SfM standalone on an arbitrary frames directory and emits `cameras.json`. The MAtCha `train.py` already invokes this code path internally; we just need to invoke it without the downstream chart-alignment / refinement stages.
- A small extension to `workspace/build_blender_scene.py` to:
  - Accept N cameras (not just the 12 the current code assumes).
  - Skip the mesh-import step when no mesh is provided.
- A picker convention. Simplest possible: a JSON file listing the chosen frame indices, hand-edited in Blender. No live UI.

Total: a few hundred lines of Python at most. Most of B3's tooling already does the Blender-side work.

### Update — UI design expanded into its own note

Jeremy raised the point that "60 cameras is a slog" understates the problem in the other direction: when focusing on one area of a room, you may want *more* than 60 cameras visible there, but conditionally hide the rest. That requires real **filtering**, not a fixed pool size.

The full feasibility analysis — Blender Collections (Route A, ~0.5 day) vs a viser-based web viewer with continuous sliders, click-to-pick, time/position/direction/co-visibility filters (Route B, ~1–3 days) — lives in the sibling note `2026-05-01-camera-selection-ui-feasibility`. Recommended sequencing: Route A first, escalate to Route B if Blender's discrete-bucket Collections aren't enough.

The "small" estimate above stands for Route A only. Route B is a real but well-scoped extension.

## Sampling — open question

The candidate pool of 60 frames itself is sampled from the source video. Default: even-time-spacing (what we do today, but on more frames). Better options worth trying:

- **Motion-aware sampling.** Extract more frames where camera is moving fast (more viewpoint change per frame), fewer where it's hovering. Optical-flow magnitude gives this.
- **Blur rejection.** Skip frames with high motion blur (variance of Laplacian or similar). Cheap pre-filter.
- **Similarity-based dedupe.** ASMK-on-MASt3R-encoder gives a similarity matrix; drop frames whose nearest neighbor is above threshold.

For the first pass, just sample ~60 evenly-spaced candidates and see what MASt3R-SfM does with them. Iterate from there. Don't over-engineer the sampler before we've seen what 60 cameras-in-Blender feels like to curate.

## Pre-filtering option (purely 2D)

If we want to whittle 100+ candidates down to ~30 cheaply *before* running MASt3R-SfM, ASMK + farthest-point-sampling on the encoder features alone gives image-similarity-based diversity sampling. The MASt3R-SfM paper uses this exact method internally (§4.1) for keyframe selection in scene-graph construction. Cost: ~10 sec for 100 frames vs. ~1–2 min for SfM. Useful as a *reducer* — drop redundant candidates before paying the SfM cost — not a replacement for the SfM-and-Blender step.

A combined pipeline — ASMK to whittle 100 → 30, MASt3R-SfM on the 30, hand-pick 12 — is probably the right balance.

## Connection to the MASt3R-SfM paper's findings

§4.1 of the SfM paper: their scene-graph construction uses N_a = 20 anchor frames (FPS-sampled from the similarity matrix) plus k=10 nearest neighbors per non-anchor frame. The supplementary's kinematic-chain ablation (Table 6) showed that hierarchical clustering by *number of correspondences* on the SfM scene graph was the best parameterization — better than maximum spanning tree, better than star.

This is exactly what hand-curation tries to achieve manually: pick a small set of *viewpoint-diverse* keyframes with good pairwise correspondence structure. The paper validates that the spatial-diversity intuition is the right one — they just automate it; we'd manualize it for the first cut.

## Scaling — how many candidates is reasonable?

- **24 frames**: definitely fine. We've seen MASt3R-SfM run on 24 inside MAtCha (the OOM at 24 was *downstream* in chart-alignment, not SfM).
- **60 frames**: very likely fine. Well within the paper's tested regime, and the retrieval-based scene graph (default) is quasi-linear in N.
- **100 frames**: probably fine, would want to verify. The paper benchmarks at 200 views with 8.4 GB GPU memory using the retrieval graph.
- **200+ frames**: getting into territory where the paper used 80 GB GPUs for some operations, but the *retrieval-based* graph variant (which is what we'd use) stayed at 8.4 GB. Should be fine on RTX 5080's 16 GB.

For the curation use case, **the bottleneck is human cognitive load, not compute.** ~60 cameras is comfortable to inspect in Blender. ~30 is easy. 100 is a lot. So we should size the candidate pool to what's curatable, not to what MASt3R-SfM can handle.

## What I'd do first

1. **Measure** — run MASt3R-SfM standalone on 60 frames from scene 004, time it, eyeball VRAM. (T-017: don't extrapolate from the 12-frame timing; measure 60 directly.)
2. **Sketch the standalone wrapper** — should be a small Python entry point that reuses MAtCha's existing MASt3R-SfM invocation but stops after cameras.json is written.
3. **Extend B3** — make the Blender script work without a mesh and with arbitrary N cameras.
4. **Try it** on scene 004 with 60 evenly-spaced candidates, pick 12 by hand, run full MAtCha on those 12, compare against the existing 12-frame baseline.

That's the experiment. If the curated-12 mesh is visibly better than the evenly-time-spaced-12 baseline, B5 is validated and we apply it to 001/003. If not, we've learned the sampling strategy isn't the bottleneck on this scene type.

## Status

Folded into `entries/2026-05-01-options-on-the-table-after-b6a/entry.md` (Option B, refined). `consolidated: true`. UI design questions extracted into the sibling note `2026-05-01-camera-selection-ui-feasibility`.
