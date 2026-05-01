---
kind: entry
date: 2026-05-01
title: B6a lowres-keyframes — negative result locked in
mood: clarifying
consolidates_notes: []
tags: [matcha, experiment, negative-result, b6a, resolution, frame-count]
---

# B6a lowres-keyframes — negative result locked in

## The experiment

Hypothesis: lower per-frame resolution would reduce VRAM pressure during MAtCha's chart-alignment stage, letting us fit more keyframes; more keyframes would mean better viewpoint coverage; better coverage would mean a better mesh.

Concrete test: same scene 004 source video, resolution dropped from 1024×576 to **768×432** (56% the pixel count), frame count raised from 12 to 15 (the new ceiling at this resolution; 18 OOMed). All other knobs unchanged. Full per-experiment record in `experiments/004-matcha-lowres-sky-house/README.md`.

## Verdict

> "Complete garbage. More lower-quality photos is certainly worse — at least the way we did it." — Jeremy, post-inspection

The quantitative metrics on the lowres run all looked *better* than the baseline:

- Cull retention 90.5% / 87.3% (vs 78% / 65%).
- Vertex color coverage 97.7% (vs 89.4%).
- Median views per vertex 6 (vs 4) — each vertex triangulated against ~50% more cameras.
- Half the tetra-mesh size for similar topology (239 MB vs 422 MB).

**None of that translated to visually better mesh.** Per-pixel detail loss from the 768×432 input dominated the apparent gains from more frames. The watertight surface had less to bite into, and the result looked worse despite being technically tighter.

## What this rules out (and doesn't)

**Ruled out:**

- Lower-res-with-more-frames as a path to better quality on this scene type.
- By extension, almost certainly lower-res-at-same-frames (no reason to expect quality improvement from less detail at constant view count).

**Not ruled out:**

- Better frame *selection* at the 12-frame budget (B5 — manual curation of the 12 best viewpoints from a wider candidate pool).
- Higher-resolution input at the 12-frame budget (1280×720), if it fits in 16 GB VRAM.
- MAtCha-internal knobs we haven't touched (`r` chart-encoding resolution, gaussian-splat iters, TSDF settings).

The DECISION-MATRIX open-question #2 ("is the 16-frame VRAM ceiling per-resolution?") is now answered: yes, but the answer doesn't help us. Lowering resolution to fit more frames trades the wrong way.

## What I learned about MAtCha from this

Re-reading the paper after the negative result clarified why this was the predictable outcome: MAtCha's charts are initialized from a monocular depth estimator (DepthAnythingV2). The high-frequency surface detail in the final mesh comes from those depth maps, not from triangulation across views. Triangulation (via MASt3R-SfM) provides the *scale alignment* between charts, not the per-pixel detail. So degrading per-pixel resolution attacks the part of the pipeline that carries the surface detail, in exchange for a marginal improvement in the part that aligns charts to each other.

In other words: the chart density is not the bottleneck on our captures; the chart *fidelity* is. Lowres-more-frames was attacking the wrong term.

## A small floor-deduction wrinkle

Worth flagging: the lowres run picked a different RANSAC candidate as the floor (cand 2, score 12,075) than the baseline (cand 1, score 18,865). The mesh ended up with 1.26 m of below-floor geometry vs the baseline's 0.34 m. Most of it culled cleanly, but it suggests B1's candidate-plane scoring may be sensitive to small mesh changes. Not blocking; flagged in the post-processing thread for future hardening.

## Next steps that follow from this

This entry's closing question is "what direction *would* improve quality?" — covered separately in `2026-05-01-options-on-the-table-after-b6a` (sibling entry).
