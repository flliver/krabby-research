---
kind: note
captured: 2026-05-01T16:12:29-07:00
consolidated: false
tags: []
---
# MASt3R-SfM scaling for large candidate pools

Captured in response to the question: can we *realistically* get the position of hundreds of video frames? The earlier spatial-curation note assumed a 60-frame candidate pool. Jeremy raised that 60 understates the problem when the camera moves all over a room — focusing on one area might require 30+ visible cameras *in that area*, not just 30 total. This note lays out what MASt3R-SfM can actually do at scale and how to plan for it.

## The question splits in two

1. **Does MASt3R-SfM scale?** (technical — runtime, VRAM, accuracy)
2. **Does it work *well* at hundreds of frames?** (practical — does pose accuracy hold up)

Both answers are yes, with caveats. Then there's the harder question — how do we *use* hundreds of poses without melting the human curator — which is what motivates the multi-stage pre-filter below and the camera-selection UI in the sibling note.

## Part 1 — does it scale?

**Paper claims (verified from MASt3R-SfM paper):**

- Retrieval-based scene graph is quasi-linear in N: `O(N_a² + (k+1)·N)` edges, defaults `N_a=20`, `k=10`. So 100 frames → ~1100 edges, 200 → ~2200, 500 → ~5500.
- Benchmarked on Tanks & Temples scenes from 151 to 1106 images. At 200 views with the retrieval graph, used **8.4 GB GPU memory** (Table 4) — well within RTX 5080's 16 GB.
- Goes to 1000+ images on 80 GB GPUs, but those large numbers are the *complete graph* variant we'd never use. Retrieval graph stays roughly constant in memory because per-image encoder features are cached and per-edge decoder forwards are streamed.
- Works in pure-rotation cases (paper supplementary §3) — even cameras sharing an optical center.

**Measured on bbeeprz (RTX 5080, 16 GB):**

- 24 frames: SfM stage runs cleanly inside MAtCha. We know this because the OOM at 24 frames was *downstream* in chart-alignment, not in SfM.
- 12 frames: ~1 minute SfM wall-clock.
- Anything higher: not yet measured.

**Estimates for our hardware (paper-extrapolation, T-017 — flagged as such):**

| Frames | Est. wall-clock | Est. peak VRAM | Confidence |
|--------|-----------------|----------------|------------|
| 60 | ~5–10 min | ~5–8 GB | high — well within paper's tested envelope |
| 100 | ~8–15 min | ~6–10 GB | high — paper benchmarks this directly |
| 200 | ~15–30 min | ~8–12 GB | medium — paper used 80 GB GPUs, but with retrieval graph should fit |
| 500 | ~45–90 min | ~10–14 GB | low — would want to actually measure |
| 1000 | ~2–4 hr | possibly OOM on 16 GB | unknown — paper used ≥40 GB at this scale |

None measured on bbeeprz. First step before committing to a 200+ frame workflow: **measure SfM at 60 and 200 frames** to validate the curve and pin VRAM peaks.

## Part 2 — does it work *well* at scale?

The MASt3R-SfM paper reports excellent accuracy across N=25/50/100/200/full on Tanks & Temples (Table 1). **Performance actually improves with more frames** — ATE drops from 0.034 (25 views) to 0.011 (full). More frames = more correspondences = more stable bundle-adjustment refinement.

So the answer to "does it work well at scale" is yes — *better*, in fact, than at sparse-view counts.

**Two failure modes flagged in the paper:**

1. **Symmetric-structure outlier matches** (paper §5.3 / supplementary §3). With more frames you have more chances for false correspondences between similar-looking scene parts (e.g., repeating windows, identical floor tiles). The paper's only documented failure mode. Our M11 captures (firepit, sky-house-dining) don't have obvious symmetric structure, but worth being aware of.

2. **First-order optimization slowness** (paper §5.3 limitations). The refinement uses Adam, not Levenberg-Marquardt. Convergence at large N can be slow. Paper reports minutes for N=200 but doesn't quote N=1000 timing.

## Part 3 — what's the actual planning constraint?

For curating ~12 from a candidate pool, **the binding constraint isn't compute. It's curation cognitive load and source-frame redundancy.**

For a 4-minute capture at 30 fps you have 7,200 source frames. Choices:

| Pool size | Curation feel | SfM compute |
|-----------|---------------|-------------|
| 60 (1 every ~4 sec) | comfortable for hand-pick | well within easy zone |
| 200 (1 every ~1.2 sec) | needs spatial filtering UI | comfortable |
| 500+ | needs *clustering*, not just filtering | slower, still feasible |
| All 7200 | not curation any more — "throw the video at SfM" | 4+ hours, mostly redundant |

**The interesting regime is 100–200,** where pre-filtering removes redundancy without losing diversity, SfM is fast, and curation needs a real UI (per the camera-selection-ui note).

## Recommended pipeline — multi-stage pre-filter

**Always pre-filter the source video down to a candidate pool before running SfM.** Adjacent frames in a 30 fps video are near-duplicate in pose; running SfM on them wastes compute and blows up the curation step's job for no quality gain.

```
~7200 source frames (4 min @ 30 fps)
        │
        ▼  even-time spacing to ~1 fps
~240 candidate frames
        │
        ▼  blur rejection (drop frames with low Laplacian variance)
~215 candidate frames
        │
        ▼  ASMK dedupe on MASt3R encoder features (drop near-duplicates)
~150 candidate frames
        │
        ▼  MASt3R-SfM (est. 12–20 min on RTX 5080)
~150 cameras with poses + sparse points (points ignored)
        │
        ▼  Route A or B viewer (filter by spatial / temporal / direction)
12 chosen frames
        │
        ▼  full MAtCha
mesh
```

Each pre-filter stage is independently cheap and independently optional. For the first run we can skip blur rejection and ASMK dedupe — just even-time-sample to ~150 candidates, run SfM, view in Blender (Route A). Iterate.

**Pre-filter costs:**

| Stage | What it does | Cost |
|-------|--------------|------|
| Even-time spacing | Sample 1 in N | ~free |
| Blur rejection | Drop frames with low-Laplacian-variance | ~seconds |
| Motion-aware spacing | Sample more densely where flow is high | ~minutes (compute optical flow) |
| ASMK + MASt3R encoder | Drop near-duplicates by image-feature similarity | ~10 sec / 100 frames |

The blur and ASMK filters are the most useful additions. Motion-aware spacing is conceptually appealing but adds optical-flow compute that may not be worth it.

## Verification before code-writing

T-013 (use what's already there) and T-017 (measure, don't extrapolate) before committing to any of this:

1. **Measure SfM at 60 and 150 frames on bbeeprz.** Real timing, real VRAM peaks. Quick — extract candidates, run SfM, time it. ~30 min of work.
2. **Confirm the SfM-only code path is reachable.** MAtCha's `train.py` invokes SfM internally; the same code should be callable without running chart-alignment downstream. May already be a script in the MAtCha repo, may need a small wrapper.
3. **Verify the SfM output `cameras.json` schema.** Should contain extrinsics, intrinsics, and a frame→camera mapping. MAtCha consumes it directly so it must, but worth a sanity check before we build a viewer that depends on it.

## How this fits

- Extends `2026-05-01T153523-spatial-frame-curation-via-mast3r-sfm` (which assumed 60 candidates) — the realistic regime is 100–200 candidates with multi-stage pre-filter.
- Underwrites the `2026-05-01T153502-camera-selection-ui-feasibility` note — at 100–200 cameras, Route A's discrete-bucket Blender Collections may not be enough; this is when Route B (viser viewer with continuous filters) starts paying for itself.
- The sequencing decision in `2026-05-01T152120-options-on-the-table-after-b6a` (C first, then B) is unchanged. This note specifies what B looks like at realistic scale.

Standing by for whoever picks this up next.
