---
kind: note
captured: 2026-05-01T16:44:53-07:00
consolidated: false
tags: [matcha, code-read, source, r-knob, photometric-resolution, b5, c-knob]
---

# MAtCha source code-read findings (Anttwo/MAtCha @ HEAD on JDP-Mac)

Cloned `https://github.com/Anttwo/MAtCha.git` to `~/dev/MAtCha`. Public source, no patches applied (our 8 patches are build-time fixes, not algorithmic — checked `MATCHA-NOTES.md` for confirmation).

This note settles three open questions and surfaces several smaller findings that change the plan.

## (1) `r` is not a single knob — it's a multi-resolution pyramid

**What I expected (from paper):** a single `resolution_factor r ∈ [0, 1]` per chart, paper-defaulted to `0.4` for unbounded scenes and `0.1` for sparse DTU.

**What's actually in the code:**

`matcha/dm_scene/parallel_aligner_with_cameras.py` defines two parameter classes:

```python
class ChartsEncodingParams():
    encoding_dim: int = 32
    resolution_factor: float = 0.4    # the paper's "r=0.4"
    initialization_range: float = 1e-4

class MultiResChartsEncodingParams():
    encoding_dim_per_res: int = 8
    resolutions: List[int] = [0.05, 0.1, 0.2, 0.4]  # ← four levels, not one
    initialization_range: float = 1e-4
```

The `configs/charts_alignment/default.yaml` config sets `use_multi_res_charts_encoding: True`. So **the actually-active params class is `MultiResChartsEncodingParams`**, not the single-res one.

This means MAtCha's default behavior is to maintain **four separate per-chart feature grids at resolutions 0.05, 0.1, 0.2, 0.4** of the chart size, with 8 features per level (32 total to match the single-res case). The MLP sees concatenated features from all four levels, so it can express deformations at every scale simultaneously — coarse correction at r=0.05, fine detail at r=0.4.

**Implication for Option C ("lower r"):**

The original framing — "lower `r` → smoother deformation → preserves monodepth detail" — is still directionally right, but the actual lever isn't a single value. Two ways to test the hypothesis:

- **Truncate the multi-res list** (e.g., `resolutions=[0.05, 0.1, 0.2]`, dropping the finest 0.4 level). Forces the deformation to be lower-frequency only.
- **Disable multi-res entirely** (`use_multi_res_charts_encoding: False`) and use single-res `ChartsEncodingParams(resolution_factor=0.1)`. Matches the paper's DTU 3-view setup.

Neither is exposed as a CLI flag in `train.py`. Both require either a config edit or a custom config file under `configs/charts_alignment/`. The cheapest path is **a new YAML config like `low_r.yaml`** that mirrors `default.yaml` but flips `use_multi_res_charts_encoding: False` and (somehow — TBD) overrides the single-res factor. Would need to trace whether the default.yaml even can override the dataclass defaults; if not, it's a small code patch.

**Best first experiment:** truncate the multi-res list rather than collapsing to single-res. Keeps the multi-scale architecture intact while testing whether the finest-resolution level is what's over-fitting to noisy SfM. Concretely: `resolutions=[0.05, 0.1, 0.2]` vs the default `[0.05, 0.1, 0.2, 0.4]`. One-line patch in the source dataclass, or a new config with the override.

## (2) Photometric supervision IS at input resolution (up to 1600 long edge)

**Settles the Option A premise.**

`matcha/dm_trainers/charts_alignment.py`:

```python
rendering_size = 1600
...
max_img_size = rendering_size
use_original_image_size = True
```

And `matcha/pointmap/dust3r.py` shows two distinct sizes flowing through the pipeline:

- **Charts (geometry):** capped at 512 long edge — `dust3r.py:200-201` literally raises ValueError if max != 512.
- **Photometric supervision images:** capped at 1600 long edge with `use_original_image_size=True`, so 1024×576 / 1280×720 / 1920×1080 all pass through unaltered.

**Implication for Option A:**

- Chart geometry resolution is *pinned at 512* regardless of input. Going from 1024×576 to 1280×720 does NOT change the chart resolution. Earlier caveat confirmed.
- Photometric refinement (chart-alignment stage 2 + optional free-Gaussians stage) DOES use the input-resolution image. So Option A would change the photometric supervision signal, even if it doesn't change chart geometry.
- The 2DGS rasterizer compares rendered surfels against the input-res image — so sharper gradients, finer photometric loss, potentially better fit on textured surfaces.

**Verdict on Option A:** not testing nothing. It is a meaningful experiment for the photometric refinement stage. **But** the geometry-determining stages (chart init from monodepth, chart alignment to SfM) are unchanged, so the marginal gain is bounded by what photometric refinement can actually fix in the mesh. My intuition: not the bottleneck for our scenes' "chaotic" character, which lives in distant background where photometric supervision is least effective anyway. Still on hold per the decision log; this doesn't promote it.

## (3) The pipeline has clean per-stage flags — no wrapper needed

`train.py` exposes:

| Flag | Effect |
|------|--------|
| `--sfm_only` | run only `scripts/run_sfm.py` (writes `cameras.json` + sparse points; stops) |
| `--alignment_only` | run only chart alignment (assumes SfM already done) |
| `--refinement_only` | run only free-Gaussians refinement |
| `--mesh_only` | run only mesh extraction (TSDF or tetra) |

Plus per-stage configs (`--alignment_config`, `--free_gaussians_config`, `--tsdf_config`, `--tetra_config`) that point at YAML files in `configs/<stage>/<name>.yaml`.

**This collapses several earlier "we need to build a wrapper" assumptions:**

- ❌ "Standalone MASt3R-SfM wrapper" — **not needed**. `python train.py -s <frames> -o <output> --sfm_only --n_images 60` does it.
- ❌ "Re-run full pipeline for each `r` value" — **not needed**. SfM once, then `--alignment_only` repeatedly with different `--alignment_config`. Each `r` sweep iteration = ~3 min instead of ~11 min.

## (4) Frame-curation lever exists already: `--image_idx`

`train.py` exposes `--image_idx 5 12 23 47 ...` — pass specific frame indices (zero-based). When the manual curation produces a list of 12 chosen indices, this is the direct invocation: no re-extraction of frames needed, just point at the original candidate dir and pass the indices.

This makes the B5 workflow even simpler than the spatial-curation note assumed:

```
candidate_frames/  (60 JPEGs)
        │
        ▼  python train.py -s candidate_frames -o sfm_only --sfm_only
cameras.json
        │
        ▼  Blender visualization (Route A or B), pick 12 indices
selected = [3, 7, 11, ..., 58]
        │
        ▼  python train.py -s candidate_frames --image_idx 3 7 11 ... 58
full mesh built from those 12
```

No re-extraction step. No glue script that maps "selected frames" → "new directory." Just an indices list + a flag.

There's also `--randomize_images` — if we want to shuffle before constant-spacing sampling. Useful for sensitivity testing.

## (5) Ergonomic detail: `--n_images` samples with constant spacing

From `train.py`:

> `Number of images to use for optimization, sampled with constant spacing.`

Confirms: when we pass `--n_images 12` against a 60-frame source dir, MAtCha picks every 5th frame. This is the **default sampling we've been doing** since Phase A — and it's exactly the policy that B5 is meant to replace, because it ignores camera-path geometry.

## Summary — what changes in the plan

1. **Option C is alive but the lever is different** — truncate `MultiResChartsEncodingParams.resolutions` rather than tune a single `r`. Concrete first experiment: `[0.05, 0.1, 0.2]` (drop the 0.4 finest level).
2. **Option C is even cheaper than estimated** — `--alignment_only` re-runs avoid the full SfM-and-refinement cost per iteration. Three `r` variants ≈ 10 min total compute, not 30+.
3. **Option A scope is clarified** — meaningful for photometric refinement, irrelevant to chart geometry. Still not the most promising path.
4. **B5 wrapper requirement evaporates** — `train.py --sfm_only` and `train.py --image_idx ...` are the wrapper. The only actual work for B5 is:
   - Frame extraction (existing `extract_frames.sh`).
   - Camera visualization in Blender (existing B3 with mesh-import skipped + arbitrary N support).
   - Picker UI (Route A: Blender Collections; Route B: viser).
5. **A delegation-ready task spec exists now.** When we move to compute, the bbeeprz envoy gets specific commands rather than research questions.

## Concrete next moves (in priority order)

1. **`--sfm_only` test on bbeeprz** with 60 candidate frames from scene 004. Get real timing + VRAM. Validate the workflow end-to-end.
2. **Add Blender camera-view rendering without mesh** to `build_blender_scene.py` (B3 extension). Local work on JDP-Mac.
3. **Hand-pick 12 frames** from the 60-camera Blender scene. Note the indices.
4. **`--image_idx` test on bbeeprz** with the curated 12 indices. Compare mesh vs the existing 12-evenly-spaced baseline.
5. **Option C `r` sweep** — only after we have a baseline mesh quality on curated frames. Tests "is over-fitting to noisy SfM the bottleneck?" against the best frame selection we can produce.

## Where the local clone lives

`~/dev/MAtCha/` on JDP-Mac. 122 MB, shallow clone with submodules. Read-only reference; not part of the workspace.
