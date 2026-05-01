---
kind: note
captured: 2026-05-01T14:43:03-07:00
consolidated: true
tags: [matcha, paper-read, knob, raw]
---

# Chart-encoding resolution `r` — unexplored knob

Surfaced from a fresh read of the MAtCha paper (Guédon et al., CVPR 2025).

§4.2 of the paper: each chart `i` carries a 2D grid of learnable features `E_i ∈ ℝ^(rh × rw × d)` in UV space, where `r` is a size ratio, `h, w` are depth-map dims, `d=32`. A tiny MLP turns a bilinearly interpolated feature (plus a 1D depth encoding) into a 3D deformation vector for any UV coordinate.

**The sparsity of the grid is the regularizer.** Quote: *"The sparsity of the 2D feature grid encourages the 2D deformation field to contain only low-frequency deformation, i.e., the high-frequency structures in the initial charts are preserved during optimization."*

§7.2: paper uses `r=0.1` for DTU 3-view bounded objects, `r=0.4` for unbounded 5–10 view scenes. Rule: *"the sparser the SfM point cloud or the training images, the lower the resolution of the chart encodings."*

## Why this might matter for us

Our 12 unposed views in unbounded scenes sit just past the paper's 10-view bucket — by their rule, `r=0.4` is the right ballpark. **But** our captures are messier than their benchmark scenes:

- 155° fisheye distortion on 001/003.
- Hyperlapse motion blur on 001.
- Sky and distant foliage that MASt3R-SfM is likely to match poorly (false correspondences, unstable depths).

If MASt3R-SfM is producing more outlier points on our scenes than on the paper's benchmark scenes, the deformation grid at `r=0.4` may be over-fitting to those outliers. The "chaotic but recognizable" character — and specifically the "background noise pollution" — is consistent with a per-chart deformation field that's too flexible relative to the signal-to-noise ratio of the SfM points it's chasing.

Lowering `r` would force the deformation MLP to act low-frequency, trusting DepthAnythingV2's per-image structure more and SfM-driven deformation less. SfM is least reliable on distant background; reducing its per-pixel control there is exactly the regularization we'd want.

## What I don't know

- The actual default value of `r` in our `train.py` invocation. Could be 0.4 (paper-default for our regime), could be something else.
- Whether the train.py CLI exposes `r` as a flag or whether it's a config-file edit.
- Whether the chart-alignment vs photometric-refinement stages use the same `r`.

T-002: I haven't checked the code. Don't take this as a recommendation yet — take it as a candidate hypothesis to investigate. Verify the default, then if the default is non-obvious, a quick `r ∈ {0.1, 0.2, 0.4}` sweep on scene 004 is the cheapest possible experiment.

## Status

Consolidated into `entries/2026-05-01T152120-options-on-the-table-after-b6a/entry.md` (Option C). Keeping this note around as the longer-form rationale; the entry has the decision-grade summary.
