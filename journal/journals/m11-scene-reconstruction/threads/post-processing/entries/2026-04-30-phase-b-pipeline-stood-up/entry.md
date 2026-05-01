---
kind: entry
date: 2026-04-30
title: Phase B post-processing pipeline stood up — orient / cull / cameras / color
mood: shipping
consolidates_notes: []
tags: [phase-b, post-processing, blender, b1, b2, b3, b4]
---

# Phase B post-processing pipeline stood up — orient / cull / cameras / color

## What we built

Four Python tools, each a single responsibility, run end-to-end per scene:

- **B1 — `workspace/orient_mesh.py`** — RANSAC-based ground-plane detection on the raw MAtCha tetra mesh, then a rotation + z-shift so the floor sits at z=0 and is normal to +Z.
- **B2 — `workspace/cull_mesh.py`** — z-threshold + radial-distance cull. Removes the obvious below-floor outliers and the unbounded distant-background pollution. Configurable per scene.
- **B3 — `workspace/build_blender_scene.py`** — headless Blender script. Imports the oriented + culled mesh, places one Blender Camera object per recovered MAtCha pose (from `cameras.json`), creates a textured image plane at each camera position showing the source frame the camera saw, writes `scene_culled.blend`.
- **B4 — `workspace/project_color.py`** — projects vertex colors onto the mesh by sampling the source frames at each vertex's projection in each visible camera, weighted-averaged by view confidence.

End-to-end output per scene: a `scene_culled.blend` Jeremy can open in Blender and inspect. Mesh is gravity-aligned, has a usable ground plane, has the worst background noise removed, has the camera positions visualized, and has color.

## Why this was the right pivot from Phase A

Phase A retrospective (sibling thread `matcha-quality`, entry `2026-04-30-phase-a-three-meshes-and-the-post-processing-pivot`) found that the five cross-cutting issues across all three Phase A scenes — no ground plane, tilt, background noise, no cameras visible, no color — were post-processing gaps, not MAtCha failures. Phase B addresses all five.

The bet: if we fix the post-processing, the remaining quality complaints are *real* MAtCha-level issues that justify further pipeline work. Without Phase B, every quality issue is conflated.

## What's still open

- **B1 candidate-plane robustness.** The lowres-15 experiment (sibling thread, entry `2026-05-01-b6a-lowres-keyframes-negative-result`) picked a different RANSAC candidate as the floor than the baseline run on the same scene. The cull cleaned it up, but it suggests B1's scoring is sensitive to small mesh variations. Worth a regression test if we run the same scene at multiple resolutions/frame counts.
- **B2 cull thresholds.** Currently per-scene-tuned. A more principled approach (e.g., density-based or learned) might generalize better. Not blocking.
- **B4 color quality.** Vertex color is averaged across source frames — some specular surfaces look muddy. View-dependent shading or per-frame proxy textures could help, but not a milestone-blocker.

## Output state

All three Phase A scenes have a `scene_culled.blend` file in their respective `data/scenes/<id>/matcha_output/oriented/` directory. These are the artifacts Jeremy is currently inspecting against future MAtCha experiments.

## What this enables

- Apples-to-apples comparison between MAtCha runs: every run goes through the same B1–B4, so quality differences are attributable to MAtCha-level changes (frame count, resolution, curation) rather than post-processing variance.
- The scene_culled.blend is the natural input to Phase C (USD export + IsaacSim load) when we get there.
