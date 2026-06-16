# DA3 (DepthAnything3) posed-mode gs_ply lives in a NORMALIZED internal frame, not the input/COLMAP world frame

> Verified, root-caused finding. Source: STO-SCN-105 (real2sim scout-gauge
> registration). Durable reference for scene-reconstruction work involving
> DA3 posed-mode gaussian export.

## Problem
When you feed DepthAnything3 known camera poses (posed mode:
`model.inference(images, extrinsics=<Nx4x4 w2c>, intrinsics=<Nx3x3>, infer_gs=True, export_format="...gs_ply...")`),
the exported **gs_ply gaussian is NOT in your input/COLMAP world frame**.
Mapping it onto the SfM points requires a full similarity with a large
**ROTATION** (~125° on 001-patio), a **scale** (~0.14), and a
**translation**. Overlaying SfM camera frustums on the raw splat therefore
looks badly misaligned.

## Root cause (from DA3 source — ByteDance-Seed/Depth-Anything-3, `src/depth_anything_3/api.py`)
1. `_normalize_extrinsics` recenters the scene **to camera 0**
   (`transform = affine_inverse(ex_t[:, :1])`; `ex_t_norm = ex_t @ transform`)
   and **rescales translations by the median camera distance**
   (`median_dist = median(||c2w_norm translation||)`;
   `ex_t_norm[...,:3,3] /= median_dist`). The model predicts depth +
   gaussians in THIS normalized frame. The recenter-to-camera-0 is the
   source of the large rotation.
2. `inference()` then runs
   `align_poses_umeyama(prediction.extrinsics, input_extrinsics)` →
   returns `scale, aligned_extrinsics`.
   - `align_to_input_ext_scale=True` (**DEFAULT**):
     `prediction.extrinsics = input_extrinsics; prediction.depth /= scale`.
     **The gaussians are NOT transformed back.** ← this is the trap.
   - `align_to_input_ext_scale=False`:
     `prediction.extrinsics = aligned_extrinsics` (DA3's predicted poses
     Umeyama-aligned to input).
3. Consequence: with the default, the npz `extrinsics` and the `colmap`
   export cameras are the **echoed input** (byte-identical to what you
   passed) — so they CANNOT be used to recover the gaussian's frame.
   `scale_factor`/`is_metric` describe metric depth, and `scale_factor`
   maps DA3's *colmap points3D* to input scale (~3.3 ratio observed) but
   does NOT describe the gs_ply.

## The fix (camera-pose Umeyama, NOT geometry ICP)
The gaussian→world transform is the inverse of the normalization composed
with the model's residual:
- **Analytic init (from input poses alone):**
  `p_world = median_dist · R0ᵀ · p_gaussian + C0`, where (R0,t0)=camera-0
  w2c, C0 = -R0ᵀt0 = cam0 center, median_dist = median recentered camera
  distance. On 001-patio this recovered rotation 117.2°, scale 0.1996,
  t=[0.126,-0.036,0.08].
- **Exact:** the internal
  `align_poses_umeyama(DA3_predicted_poses, input_poses)` similarity (the
  one DA3 computes and discards). The analytic is ~11° / 1.4× off because
  the model's predicted poses differ slightly from the exact normalized
  input — the Umeyama on predicted poses captures that residual. To obtain
  DA3's predicted poses, run with `align_to_input_ext_scale=False`
  (returns aligned predicted poses) or capture the raw predicted
  extrinsics before the overwrite.

## Validation
Operator hand-solved the transform via a photo-match tool (aligning the
splat to a real de-warped photo from a known pose) on 001-patio scout
`OZGYMJTRXN3Z`: scale 0.1415, rotation 125.6°, t=[0.09,-0.086,0.14]. The
analytic normalization-inverse matched to ~11°/1.4×, confirming the
mechanism.

## Critical caveat — do NOT use geometry ICP
Automatic point-cloud registration (PCA-init scale-ICP of gs_ply ↔ SfM
points3D) is **UNRELIABLE**: on 001 it reached 87% inlier fit yet a 147°
**WRONG** rotation, because the scene is near-symmetric so geometry admits
high-overlap-but-wrong rotations. Only camera-pose correspondence (or the
photo) disambiguates. **Register on CAMERAS, not point clouds.**

## Metric scale: `is_metric` is true, but `scale_factor` is NOT a stable gauge calibration

> Earned 2026-06-16 (STO-SCN-016 scout audit, scene 001-patio). Owner: scout.
> **Red herring to prevent:** "DA3 says `is_metric: true`, so read `scale_factor` and that's
> meters-per-solve-unit." It is not.

Audit of three scouts of the **same scene + same solve gauge** (`001-patio`,
`cameras/62QEHJDAJZBI`), all `is_metric: true`:

| scout | n_views | `scale_factor` | `transform.scale` (gs→solve) |
|---|---|---|---|
| `3R7ZB5GAB6PC` | 29 | 3.346 | 0.1929 |
| `OZGYMJTRXN3Z` | 30 | 3.481 | — (pre-transform format) |
| `W75HYBNU37WK` | 24 | 11.220 | 1.7495 |

The metric scale of a **fixed** solve gauge must be a **constant**. The recorded `scale_factor`
spans **3.3×** (3.35 → 11.22) — so it is **not directly usable** as the solve's metric
calibration. Cause: `scale_factor` tracks DA3's per-run **median-camera-distance normalization**
(`_normalize_extrinsics`) + prediction variance, which shift with the **view subset** (the 24-view
run is the 11.2 outlier; the two ~29–30-view runs cluster at ~3.4). `transform.scale` (the gs→solve
Umeyama scale) is likewise run-dependent (0.19 vs 1.75), so no trivial `scale_factor`/`transform.scale`
combination is yet a validated meters-per-unit either — that derivation needs ground truth, not algebra alone.

**Implications for STO-SCN-016 (metric scale):**
- `is_metric` confirms DA3 emits a metric *estimate* — treat it as a **prior**, never as truth.
- Metric calibration MUST be a **hybrid**: DA3 prior **anchored/validated by one ground-control
  measurement** per scene (a hand-measured real distance, or a known-distance pick added to
  `match.html`). The control measurement also selects which DA3 runs to trust (~3.4 here; reject 11.2).
- Fix the resulting scale at the **datum/gauge level** (cameras), not as a per-mesh transform, so
  every downstream mesh + camera-relative metric cull primitive (STO-SCN-137) inherits one factor.
- Run scouts with **consistent full-spine view handling** across scenes to suppress the
  view-subset-driven `scale_factor` variance.

**Canonical generic treatment** (OLAI corpus, no project specifics):
`personal.research/3d-reconstruction/metric-scale-calibration/index.md` — the validated recipe:
control distance is primary (`s = D_measured / d_solve(p1,p2)` on the whole Sim3); monocular metric
is prior/gate/fallback only; gross-error gate at `s_control/median(s_monocular)` outside ~1.5×;
robust median in log-scale; ≥2 control distances on different axes for anisotropy. Incumbent
parallels: COLMAP `model_aligner`, Metashape scale bars, RealityCapture Define-Distance.

## Sources
- https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py
- https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md
- https://github.com/ByteDance-Seed/Depth-Anything-3/issues/62 (COLMAP format / --align-to-input-ext-scale)
- https://github.com/ByteDance-Seed/Depth-Anything-3/issues/81, /244 (metric scale)
- Internal: STO-SCN-105 (real2sim scout-gauge registration), `da3_infer_posed.py`, `da3_render_view.py`, `scout_register.py`.
