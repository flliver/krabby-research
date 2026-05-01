---
kind: thread
name: MAtCha mesh quality
description: Open inquiry into how to get high-quality meshes out of MAtCha on our M11 captures. Covers the input-side knobs (frame count, resolution, frame selection), the MAtCha-internal knobs (chart-encoding resolution `r`, gaussian-splat iters, TSDF settings), and the open negative/positive results.
opened: 2026-04-30
closed: null
summary: null
references:
  - journals/m11-scene-reconstruction/threads/post-processing
tags: [matcha, mesh-quality, sparse-view]
---

# MAtCha mesh quality

The driving question: **given that MAtCha produces watertight meshes natively (R1 satisfied) and runs in ~11 minutes (R5 satisfied), what determines whether the mesh is *good*?**

Phase A established that the same recipe (12 evenly-spaced keyframes at 1024×576, `vitl` encoder, unposed SfM) produces a "chaotic but recognizable" result on every M11 scene we've tried (001 patio, 003 firepit, 004 sky-house). The character of that chaos is consistent: foreground is workable, distant background is polluting noise.

Phase B's post-processing pipeline (orient / cull / cameras / color) addressed the chaos that was *post-processing* in nature. What remains is the chaos that's *intrinsic to the MAtCha output* — the gaps, the geometric noise, the parts where the mesh just isn't a faithful reconstruction.

This thread tracks every move we make to improve the underlying mesh quality, regardless of where the move happens (frame extraction, MASt3R-SfM stage, MAtCha chart alignment, gaussian-splat refinement, mesh extraction).

## Live questions

These live in `experiments/DECISION-MATRIX.md` open-questions list as well; surfaced here for quick reference. Order reflects the 2026-05-01 decision in the `options-on-the-table-after-b6a` entry.

- **Q7 / Option C — `r` knob (next up).** Does a lower chart-encoding resolution `r` (the paper's per-chart deformation grid) help on our noisier captures? First step is verifying the current default. Then sweep `r ∈ {0.1, 0.2, 0.4}` on scene 004.
- **Q6 / Option B — refined manual curation (after C).** Use MASt3R-SfM standalone to compute camera poses for a candidate pool, render cameras-without-mesh in Blender, hand-pick 12. Tooling: standalone SfM wrapper + B3 extension. See note `2026-05-01-spatial-frame-curation-via-mast3r-sfm`.
- **Q5 / Option A — higher resolution (on hold).** Pending a code-read of MAtCha's internal pipeline to confirm whether photometric refinement uses the original input resolution or a 512-downscaled version. If the latter, A is testing nothing useful and should be dropped.

## Closed inquiries

- B6a — lower-res-with-more-frames (768×432 at 15) — **negative result**, see entry `2026-05-01-b6a-lowres-keyframes-negative-result`.

## Cross-references

- Phase B post-processing work lives in the `post-processing` thread (linked above), because while it improves the *delivered* mesh, it doesn't change the underlying MAtCha output.
