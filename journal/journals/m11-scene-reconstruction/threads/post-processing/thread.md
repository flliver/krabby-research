---
kind: thread
name: Phase B post-processing
description: Tooling that takes raw MAtCha tetra mesh → gravity-aligned, ground-plane-deduced, background-culled, vertex-colored, camera-annotated `scene_culled.blend`. Ready-for-IsaacSim is the goal but USD export (Phase C) is owned elsewhere.
opened: 2026-04-30
closed: null
summary: null
references:
  - journals/m11-scene-reconstruction/threads/matcha-quality
tags: [post-processing, blender, mesh-conditioning]
---

# Phase B post-processing

The Phase A retrospective (after running MAtCha on 001/003/004) found that the meshes had five cross-cutting issues that were **not MAtCha's fault** — they were post-processing gaps:

1. No clear ground plane.
2. Output mesh always tilted (no consistent up direction).
3. Background-noise pollution (especially 155° fisheye outdoor).
4. No camera locations visible in the mesh.
5. No vertex color from the source frames.

Phase B is the tooling pipeline that addresses all five. Sub-phases:

- **B1** — ground-plane orient (RANSAC + z-shift). Implementation: `workspace/orient_mesh.py`.
- **B2** — background cull (z-threshold + radial). Implementation: `workspace/cull_mesh.py`.
- **B3** — camera markers + image planes in Blender. Implementation: `workspace/build_blender_scene.py` (headless Blender).
- **B4** — vertex-color projection from source frames. Implementation: `workspace/project_color.py`.

End-to-end output per scene: `scene_culled.blend` with mesh + 12 Blender Camera objects + textured image planes showing what each camera saw.

## Status

Essentially complete as of 2026-05-01. All four sub-phases working on all three Phase A scenes. Open follow-ups are minor (B1 candidate-plane robustness on the lowres-15 mesh, see entry `2026-05-01T144205-b6a-lowres-keyframes-negative-result`) rather than structural.

## Why this is a separate thread from `matcha-quality`

Post-processing improves the *delivered* mesh; it does not change the underlying MAtCha output. If MAtCha produces a hole-ridden mesh, no amount of B1–B4 will fill the holes. The two threads have different optimization targets and different time horizons.

## Cross-references

- `matcha-quality` — for the underlying-mesh improvements that have to happen *before* post-processing can do its job.
