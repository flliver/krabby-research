---
kind: note
captured: 2026-05-01T17:46:51-07:00
consolidated: true
tags: []
---
# Sliding-window SfM and the keyframe-localization alternatives

Surfaced during the SfM-scaling experiment. Jeremy asked: "couldn't we chain together multiple-overlapping-sliding-windows and then 'align' the camera positions based on the overlaps?"

Yes — and the answer connects to a small family of established techniques. Worth knowing about as the alternative path if we ever hit a hard ceiling on global SfM (we just did at N=500 OOM).

## Does MAtCha-SfM scale linearly?

Roughly, with a small super-linear component. From the in-progress experiment (subtracting S&box overhead from contaminated runs):

| N | edges (N_a² + (k+1)N) | sec | sec/edge | sec/N |
|---|---:|---:|---:|---:|
| 24 | 664 | 193 | 0.29 | 8.0 |
| 60 | 1060 | 415 | 0.39 | 6.9 |
| 120 | 1720 | 749 | 0.44 | 6.2 |

`sec/edge` climbs slowly (0.29 → 0.44 over a 5× N range). Between linear-in-edges and linear-in-N. The graph itself is linear in N (paper §4.1), but bundle adjustment and matching grow slightly worse than that as the cross-frame interaction matrix densifies.

VRAM scales similarly slowly — roughly +1–2 GB per doubling. **VRAM is the binding constraint, not runtime.**

## Three variants worth knowing about, in order of complexity

### (1) Sliding-window MASt3R-SfM — the user's original proposal

- Run the existing pipeline N times on N-frame windows with overlapping frames.
- Compute inter-window transforms by Procrustes on the K overlap frames (typically 7-DOF: rotation + translation + scale).
- Optionally a final pose-graph BA to redistribute drift.

**Pros:** trades wall-clock for VRAM. No 16 GB ceiling — arbitrary N.

**Cons:** drift accumulation (each window-to-window join injects small registration error). ~few hundred lines of Python wrapper. Re-runs the bundle-adjustment per window, wasting compute on within-window optimization that's mostly redundant for downstream visual selection.

### (2) PnP localization against a keyframe reconstruction — the SLAM/VL standard answer

- Run MASt3R-SfM at N=60-100 keyframes to get a high-quality reconstruction.
- For each remaining frame, run **PnP** (perspective-n-point) against the existing 3D points using MASt3R's pairwise matches.
- Each new frame's pose is computed in the existing coordinate system.

**Pros:** O(1) per additional frame. Single coordinate system. No drift. The standard "visual localization" pattern.

**Cons:** requires the keyframes to cover the scene well. Less accurate per-frame poses than full SfM (PnP is a constrained optimization, not full BA).

### (3) Pairwise pointmaps via MASt3R — lightest-weight

- Pick one "seed" frame as origin.
- For each additional frame, run MASt3R's *pairwise* forward (just the encoder + decoder, no SfM optimization) against its nearest keyframe.
- The pairwise pointmap directly gives relative pose.
- Concatenate to keyframe → world.

**Pros:** very fast per-frame, ~seconds. Reuses MASt3R machinery that's already in the container.

**Cons:** noisier than full SfM. Probably fine for visual-selection accuracy; not for precision SLAM.

## What MAtCha-SfM already does internally

Critical context: the MASt3R-SfM paper's `N_a = 20` FPS-sampled keyframes already implement keyframe-anchoring. Non-anchor frames connect to their nearest keyframe + k nearest neighbors. **This is *internal* keyframe-anchoring** — same pattern as windowed SfM, just done in one optimization pass instead of multiple.

The internal keyframe scheme is what gives us the quasi-linear scaling. If we hit a ceiling, it's *because* internally MASt3R-SfM is doing all the inter-keyframe work in one optimization pass that has VRAM cost.

So sliding-windows is *more* manual than what MAtCha already does internally. The paper's authors essentially picked "global SfM with internal keyframes" as the right point on the tradeoff curve. We'd only want to break out of that if we exceed what their global formulation can fit.

## Verdict for our use case

For **visual selection** (positioning frames so a human can pick 12), drift of cm or even meters is fine; we just need cluster/viewpoint relationships to look right.

- If global MAtCha-SfM scales to our needed N → use it. No reason to add complexity.
- If global OOMs at our needed N → **PnP localization (variant 2)** is the right answer, not sliding-window MAtCha-SfM (variant 1). Lower complexity, single coordinate system, no drift, leverages keyframe quality.
- Variant 3 (pairwise pointmaps) is the fallback if PnP infrastructure is hard to set up.

For **mesh generation at scale** (the submap-fusion architecture in the sibling note), the global pose recovery is the *first step*. Sliding-window MAtCha-SfM doesn't help for that path — what helps is one global SfM + many local MAtCha runs.

## Status

Reference note. Not actionable today (we just learned the ceiling is below 500 — we'll know whether 350 fits soon, and if so we don't need any of these alternatives for M11). Useful for the M12+ scale-up case and for handing future agents context if the global-SfM path stops being adequate.
