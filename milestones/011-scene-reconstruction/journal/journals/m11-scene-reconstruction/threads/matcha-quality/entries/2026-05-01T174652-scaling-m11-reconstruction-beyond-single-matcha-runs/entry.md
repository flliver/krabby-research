---
kind: entry
date: 2026-05-01T17:46:52-07:00
title: Scaling M11 reconstruction beyond single-MAtCha runs
mood: null
consolidates_notes:
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T153502-camera-selection-ui-feasibility
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T161229-mast3r-sfm-scaling-for-large-candidate-pools
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T164453-matcha-source-code-read
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T174650-submap-based-mesh-fusion-scaling-matcha-beyond-a-single-run
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T174651-sliding-window-sfm-and-the-keyframe-localization-alternative
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T174652-n-500-hit-the-vram-ceiling-bracketing-strategy
tags: []
---
# Scaling M11 reconstruction beyond single-MAtCha runs

The 2026-05-01 SfM-scaling experiment forced a coherent picture out of a
cluster of side questions: how big can MASt3R-SfM actually run on our 16 GB
hardware, what does the data-curation UI need to look like at that scale,
where does MAtCha's source code constrain or enable us, and what's the
right architecture when a single MAtCha run can't cover the scene any
more. This entry consolidates six notes captured across that afternoon
and evening into one synthesis.

## The single global-SfM ceiling

We measured MASt3R-SfM on RTX 5080 / 16 GB across a sweep from N=24 to
N=500. VRAM scales roughly +1–2 GB per doubling of N — driven by the
optimizer's working set, not encoder activations or the retrieval-based
scene graph (which stays quasi-linear in N at `O(N_a² + (k+1)·N)` edges
with `N_a=20`, `k=10` defaults). The empirical operating zones:

| N | status | headroom | wall-clock |
|---|---|---:|---:|
| ≤300 | comfortable | ≥2.6 GB | ~28 min |
| 300–350 | borderline | 0.3 GB | ~33 min |
| 350–500 | unmeasured | — | — |
| ≥500 | OOM (measured) | — | — |

**300 frames is the comfortable everyday operating point; 350 is the upper
bound; 500 OOMs.** That ceiling is well above what a human can curate by
hand (~60–150 frames), so global SfM is *not* the bottleneck for M11's
candidate-pool sizing — the curator's cognitive load is.

Two operational lessons from the experiment:

- **Inspect per-process GPU before measuring.** The first round of bbeeprz
  measurements were contaminated by a foreign 4.3 GB S&box process owned
  by a different user. Cleaning up mid-experiment changed our predicted
  ceiling from "OOM around N=300" to "OOM around N=400+." `nvidia-smi
  --query-compute-apps` is the diagnostic; `sudo loginctl terminate-user`
  is the fix when the foreign process is in a different user's session.
- **Watchdog the chain.** A polling watchdog that killed the queued
  N=500 run after N=300 succeeded saved ~20 minutes of futile compute.
  Captured as `experiments/004-sfm-scaling-sky-house/scripts/kill_chain_after_n300.sh`.

## Curating from a 200-frame pool: the camera-selection UI

For pools above ~60 frames, "scroll through them in a list" stops working.
The user needs to **filter** — by where the camera is, where it's
pointing, when in the video it was captured, what it sees. Two routes
were considered:

**Route A — extended Blender scene (cheap, discrete-bucket).** Reuses
`build_blender_scene.py` infrastructure. Collections give one-click
visibility; per-camera hide handles individual selection; the Outliner's
type-ahead does name filtering. Doesn't give you continuous sliders or
"pointing-toward-X" filters as continuous controls — you discretize into
collections.

**Route B — viser-based web viewer (the real version).** viser
(Nerfstudio's WebGL Python viewer) supports camera frustums, image-plane
textures, sliders, click-to-toggle on individual frustums, gizmos, and
text widgets out of the box. The mental model is "Python wires up the
filter UI; viser handles the WebGL." Filters worth wiring (each is an
independent boolean; AND them together):

| filter axis | UI control | answers |
|---|---|---|
| time range | dual-handle slider | "first walk-around segment" |
| spatial cluster | k-means checkboxes (with invert) | "cameras near the firepit" |
| view direction | look-at gizmo | "cameras pointing at the table" |
| image similarity | ASMK / pHash buckets | "drop near-duplicates" |
| co-visibility | scene-graph clusters | "cameras sharing edges with this one" |
| picked status | tri-state per camera | "what have I selected so far" |

Picked Route B end-to-end (~900 lines across `data.py`, `filters.py`,
`ui.py`, `viewer.py`, `slots.py`, `clustering.py`). All seven filters
plus selection counters, named slot save/load, and bulk visible→pick
shortcuts. Two real bugs caught during the build worth flagging:

- **`forward_axes` sign.** MASt3R-SfM emits OpenCV-convention poses
  (+Z forward), not OpenGL (-Z). The look-at filter was inverted before
  the fix.
- **PIL deprecation churn.** `Image.LANCZOS` → `Image.Resampling.LANCZOS`;
  missing `scipy` in requirements.

Output: `selected_frames.json` with the chosen frame indices. Re-extract
those frames, run MAtCha. **Plugs straight into MAtCha's `--image_idx`
flag** — no wrapper needed, which we knew because of:

## What the MAtCha source-read settled

A morning code-read of `Anttwo/MAtCha @ HEAD` resolved three open
questions that had been blocking experiment design:

1. **`r` is not a single knob.** The default config sets
   `use_multi_res_charts_encoding: True`, which engages
   `MultiResChartsEncodingParams` with `resolutions=[0.05, 0.1, 0.2, 0.4]`
   — four levels at 8 features each. The MLP sees concatenated features
   from all four levels. Option C ("lower r") is best tested as **truncate
   the multi-res list** (`[0.05, 0.1, 0.2]`, dropping the finest level)
   rather than collapsing to single-res. One-line patch in the source
   dataclass, or a new `low_r.yaml` config.

2. **Photometric supervision IS at input resolution.** Charts are pinned
   at 512 long-edge regardless of input (`dust3r.py:200-201` literally
   raises ValueError if max != 512). But the photometric refinement stage
   uses input-res images up to 1600 long-edge with
   `use_original_image_size=True`. So Option A (higher resolution)
   doesn't change chart geometry but *does* sharpen photometric loss. Not
   a no-op, but bounded by what photometric refinement can fix in our
   chaotic-distant-background failure mode (probably not the bottleneck).

3. **`train.py --sfm_only` and `--image_idx` already exist.** No B5
   wrapper needed. The viewer's `selected_frames.json` plugs directly
   into `--image_idx`.

## When a single MAtCha run isn't enough: scaling architecture

For M11's individual-room scenes (firepit, sky-house dining, patio), one
MAtCha run covers the whole scene. For M12+ when we'll want
**whole-property** captures spanning multiple rooms, the right
architecture is:

```
GLOBAL: ONE sparse MASt3R-SfM on keyframes spanning the whole property
        → unified camera frame (300-frame ceiling × multiple buildings)
LOCAL:  MANY MAtCha runs on dense subsets within that unified frame
        (each fits in 16 GB; selected via --image_idx)
FUSE:   merge local meshes using the unified camera positions as rigid
        anchors — TSDF fusion handles the volumetric merge
```

The camera path is the spine; local meshes are the flesh. **Drift only
enters if you chain SfM windows.** A single global SfM gives you a
consistent coordinate frame for free, so submaps inherit alignment
without per-pair Procrustes. ICP / fine refinement is only needed for
subtle scale drift introduced by per-MAtCha-run chart deformation.

This pattern has well-established prior art — Hierarchical 3D Gaussian
Splatting (Kerbl et al. ACM TOG 2024, cited in MAtCha as ref [27]) hits
100,000+ images / kilometer-scale areas in real-time. Same authors as
the original 3DGS paper. There's already cross-pollination between the
two works (MAtCha uses Kerbl's affine-rescaling idea per §7.1). **A
"Hierarchical MAtCha" extension is research-shaped, not novel.**

What we already have: MASt3R-SfM produces unified poses; MAtCha's
`--image_idx` selects local subsets; `extract_tsdf_mesh.py` already
implements multi-resolution TSDF fusion. What we'd need to build:
co-visibility-based submap clusterer, per-submap orchestrator, fusion
driver. ~1–2 weeks for a careful implementation.

## Alternatives if global SfM hits a hard ceiling

Three known patterns, in order of complexity, ranked for our use case:

1. **Sliding-window MASt3R-SfM** (Jeremy's original proposal). Run the
   pipeline N times on overlapping windows, Procrustes-merge on shared
   frames, optional final pose-graph BA. Trades wall-clock for VRAM. **But
   it's *more* manual than what MASt3R-SfM already does internally** —
   the paper's `N_a=20` keyframe-anchoring scheme is the same pattern,
   just done in one optimization pass. We'd only pick this if global
   SfM's optimizer hits a wall its own keyframe-anchoring can't escape.

2. **PnP localization against a keyframe reconstruction.** The
   SLAM/visual-localization standard answer. Run MASt3R-SfM at N=60–100
   keyframes, then PnP each remaining frame against the existing 3D
   points. O(1) per additional frame, single coordinate system, no drift.
   Less accurate per-frame than full SfM (PnP is constrained, not full BA).
   **The right answer if/when global SfM doesn't fit.**

3. **Pairwise MASt3R pointmaps.** For each non-keyframe, run MASt3R's
   pairwise forward against its nearest keyframe; the pointmap directly
   gives relative pose. Lightest weight, fastest per-frame. Noisier than
   full SfM. Fine for visual-selection accuracy; not for precision SLAM.

For visual-selection use cases, drift of cm or even meters is fine — we
just need cluster relationships to look right. For mesh generation at
scale, the global pose recovery is the *first step* in the
submap-fusion architecture; sliding-window doesn't help there.

## Verdict for M11 (and what's M12+)

**For M11**: global MAtCha-SfM at the curation pool size (60–200 frames)
is the right tool, and it scales to ≥300. None of the three alternatives
above are needed; they go in design backlog.

**For M12+**: when scenes span multiple rooms, the
"global-SfM-spine + local-MAtCha-flesh + TSDF-fuse" architecture is the
plan, with PnP localization as the fallback if/when the spine itself
can't fit. Building the M11 pieces (unified SfM, candidate clustering,
per-cluster MAtCha) in a way that this future stitching doesn't require
a redesign is the cheap-future-proofing move.

The submap-fusion architecture and the curation UI converge on the same
primitive: **co-visibility clustering of unified-frame cameras.**
Building that primitive once supports both directions.
