# Scene-reconstruction — durable lessons

> The point of this file: **don't re-learn what's already known.**
> Captures the root cause, the dead-ends NOT to re-chase, and the
> hard-won rules from the scout-gauge / scout-gaussian / view-selector
> effort (STO-SCN-105, -095, -103, -104, -048). Grounded against the
> stories + source as of 2026-06-14. When code and this file disagree,
> read the cited story/source and fix this file.

---

## The root cause (DA3 normalized frame)

DA3 in posed mode (`real2sim/da3_infer_posed.py`) does two things that
break naive registration of the scout gaussian against the FastMap
solve's posed frustums:

1. **It echoes the input cameras unchanged.** The `extrinsics` in
   `results.npz` (`da3_poses.npz`) are byte-identical to the FastMap
   input (Umeyama input↔npz: scale `1.0000`, residual `0.0000`) — so
   there is **nothing to align on there**.
2. **It emits gaussians/depth in its own internal normalized scale**
   (`pred.scale_factor`, ≈2.1 on 001-patio): splat core std
   `[1.54,0.93,1.48]` vs SfM points `[0.75,0.53,0.57]` ≈ 2.1× + a
   centroid offset.

The scout path saves the raw normalized `gs_ply` and never undoes
`scale_factor`, so solve-frame frustums overlay a DA3-normalized-frame
gaussian → **~2× scale + offset; cameras float above the ground, wrong
scale** (the exact reported symptoms).

**The fix:** the `/= scale_factor` + `gauge_align` (Umeyama) pattern.

- ✅ **Already correct in `real2sim/da3_render_view.py`** (lines
  ~147–152: `if pred.is_metric and pred.scale_factor: w2c_render[:3,3]
  /= float(pred.scale_factor)`, plus orientation-augmented Umeyama over
  the shared cameras ~lines 92–112). **Copy this pattern.**
- ❌ **Still needs to go into `real2sim/da3_infer_posed.py`** (capture
  `pred.scale_factor` and persist DA3's `colmap` output cameras — its
  `export_format` already requests `colmap`, but nothing downstream
  consumes it) **and `cmd_scout` in `real2sim/v4exec.py`** (~lines
  557–566: the gather rsyncs only `*.ply` + `*.npz` with `--exclude=*`,
  dropping `exports/`/`colmap` and any `scale_factor`/`gauge.json`).

STO-SCN-105 status: diagnostic + `gauge_up` are **done**; the
persist + auto-register fix is **not yet**.

---

## Red herrings — DO NOT re-chase

Every one of these is a documented dead-end. Don't re-investigate.

- **"avg-camera-up is wrong / gravity isn't in the poses" — FALSE.**
  Gravity IS recoverable: the direction ⟂ to all camera-right axes
  lands **1.36°** from hand-clicked up (avg-camera-up was 3.1° — worse
  but still usable). Poses had gravity all along
  (`gauge_up.up_from_poses()`). The viewer's hard-coded `up=[0,-1,0]` +
  a rain-corrupted splat made correct up *look* wrong.
- **"the splat PCA gives the ground/up" — FALSE.** 001-patio was shot
  in the rain; reflections make the splat's dominant plane the
  **vertical cabin wall (~80° off gravity)**. Never derive up/ground
  from splat geometry.
- **"the solve collapsed to a plane / is degenerate" — FALSE.**
  RANSAC's "99% of points in a thin plane" was an **outlier-inflated
  threshold artifact**; trimmed out-of-plane ratio is 0.166 (a normal
  flattish patio). The validity gate correctly PASSED.
- **".ply distance-cull to fix the far-halo" — BROKE THE SPLAT.** A
  naive rewrite of `scout.gs.ply` corrupted it via an off-by-one (the
  header byte offset must include the `\n` after `end_header`; 3DGS =
  17×float32). Reverted. Handle the DA3 far-halo in the **viewer**
  (frame/clip), not by rewriting the .ply. (`build_verify.py:cull_sphere`
  later did this *correctly* on a **serve-only copy** with a self-verify
  gate — but never touch the store original.)
- **"DA3's `results.npz` extrinsics can register the gaussian" —
  FALSE / a trap.** They're the echoed *input* cameras (residual
  0.0000 vs input), in the input frame, NOT the gaussian frame. The
  data you actually need is the `colmap` export + `scale_factor`, which
  the gather currently throws away.
- **Photo-match-to-one-photo is NOT how to build the correction.** DA3
  built the splat *from* that photo, so it already matches at identity
  (s=1); reorienting to upright makes it *stop* matching. `match.html`
  *confirms* a solve; it does not *derive* the correction (compute that
  from cameras / vanishing points).

---

## Hard-won rules (always / never)

- **ALWAYS set `dynamicScene: true`** on the GaussianSplats3D viewer
  when you need live splat transforms — otherwise
  `getSplatScene().{position,quaternion,scale}` +
  `getSplatMesh().updateTransforms()` are baked statically and nothing
  moves.
- **ALWAYS de-warp fisheye scout frames to pinhole** before using them
  as a verify reference — the solve gauge is pinhole, so clicks/overlays
  on raw fisheye pixels don't map to pinhole rays. `build_verify`
  de-warps via `undistort_fisheye.py` (driven by `capture.json`
  `mode == "fisheye"`); on failure it serves raw and warns loudly.
- **ALWAYS recover `up` from the cameras (`gauge_up.py`), NEVER from
  the splat** — SfM gauge has no absolute orientation, the cameras
  carry gravity (low roll → right axes ~horizontal → gravity ⟂ to all),
  and the splat's dominant plane can be a wall/reflection.
- **NEVER rewrite the store's `.ply` to cull** — a naive rewrite
  corrupts it (header off-by-one; 3DGS = 17×float32). Frame/clip in the
  viewer, or cull a **serve-only copy** with a self-verify gate that
  re-parses the output before serving (T-012).
- **NEVER trust the scout/verify surface until the gaussian is
  auto-registered to the solve gauge** (Umeyama + `scale_factor` undo)
  AND oriented (`gauge_up`) — the splat lives in DA3's normalized frame.
- **ALWAYS use OPPOSITE selection objectives for scout vs final-select**
  — scout uses `div_angle=0` (coherent/overlapping views; DA3 fuses a
  clean gaussian only from coherent views — a diversity penalty yields a
  nebula); final-select uses coverage/variety. Same machinery, inverted
  goal.
- **When validating rotations (MASt3R/DA3), use element-wise matrix
  diffs, not trace-angle formulas** (STO-048 gotcha) — those rotations
  are ~1.16e-3 non-orthonormal; trace-angle formulas fabricate ~2.5° of
  phantom error.

---

## Gauge registration mechanics

Two independent corrections compose to put a gaussian into a usable
frame:

- **`gauge_align.py` (STO-SCN-048, shipped) — intra-gauge sim(3).**
  `align_camera_sets(src, dst, max_residual, src_rotations, dst_rotations)`
  solves a similarity `(s, R, t)` between two camera sets via
  **Umeyama/Procrustes** (closed-form SVD, det-corrected), **two-pass**:
  pass 1 positions-only for scale `s`; pass 2 augments each camera
  center with synthetic points along its optical + up axes (offset
  `d_dst` = mean center spread in dst, `d_src = d_dst/s`) so camera
  **orientations pin the rotation** that coplanar/collinear centers
  leave free (walking at one height → coplanar centers → one rotation
  DOF unconstrained). A **residual hard gate raises RuntimeError**
  rather than returning a silently-bad transform. Apply via
  `apply_to_cams2world(c2w, s, R, t)` — **scale must NOT touch
  rotations.** This registers DA3's output cameras → FastMap cameras to
  get the similarity that maps the gaussian into the solve frame.
- **`gauge_up.py` (STO-SCN-105, new) — absolute orientation (gravity).**
  `up_from_poses(w2c_list)`: take each camera's right axis (row 0 of
  `R_w2c`); gravity is the direction ⟂ to all of them — the smallest
  singular vector of the stacked rights, sign-disambiguated toward mean
  camera-up. Robust to pitch (unlike averaging camera-up).
  `roll_spread_deg()` is the confidence gauge (small median ⇒ low-roll
  assumption holds). Validated 1.36° vs manual on 001-patio.
- **Composition:** SfM/FastMap fixes nothing about absolute
  up/scale/origin (gauge freedom). `gauge_align` brings the DA3
  gaussian into the solve's frame (relative scale + rotation + offset);
  `gauge_up` rotates that common frame upright. In
  `build_verify.py:build_frustums`: carry frustums through the DA3
  similarity (`_apply_xform`), then compute
  `up = gauge_up.up_from_poses(...)` and rotate `up` by the same
  `xform` R. **Metric (real-world) scale is explicitly out of scope** —
  only orientation + intra-gauge registration position frustums in the
  gaussian.

---

## View selection

- **Voxel-coverage greedy (STO-SCN-103, in-progress) — the default.**
  Voxelize the scene (`sparse/0` points; voxel size = bbox-diagonal/grid,
  **gauge-free** so it works on the non-metric solve gauge), find
  exposed faces (occupied-voxel faces bordering empty space), weight
  each (camera, face) by **incidence flux** `max(0, cos θ)` (1.0
  head-on, →0 grazing, 0 behind/out-of-frustum), then **greedy
  submodular** add the camera with the largest marginal coverage gain
  until N or saturation. Deterministic (first-max tie-break), pure-CPU
  numpy. **Replaced STO-SCN-094's track-covisibility objective** —
  track-sharing structurally over-rewards clustered same-angle cameras
  ("95% same coverage from the same angle"); marginal gain rewards a
  complementary angle, gives a redundant one ~0. **First-light scope is
  frustum + incidence only; occlusion (voxel ray-march) is the next
  increment, not yet built.** Code: `real2sim/voxel_coverage.py`,
  tests `real2sim/tests/test_voxel_coverage.py`.
- **FisherRF info-gain (STO-SCN-104, DEFERRED / reserve).** GPU,
  model-aware selector ranking views by **Expected Information Gain on
  3DGS parameters** (Fisher information) — optimizes downstream splat
  quality directly, handles occlusion for free, emits uncertainty maps.
  **Deferred (operator, 2026-06-14):** needs a trained/optimizing 3DGS
  field as input; built for sequential NBV not batch subset-ranking
  (adaptation = days of unknown effort); needs a new CUDA/gsplat
  container + GPU contention on a currently-free CPU step; less
  deterministic / harder to debug than ~200 LOC of numpy.
- **When to use which:** default to **voxel-coverage** (geometric, CPU,
  deterministic; ships the select→verify loop). Graduate to **FisherRF**
  only if coverage-greedy underperforms or you want task-aligned
  selection + uncertainty — and only after 103 ships + operator
  promotes it. Falsifiable bar (T-001): does info-gain reconstruct
  *measurably* better than geometric coverage on a real scene?

---

## Dependency map

Per `EPI-SCN-SPINE-ASSEMBLY/epic.md`, **STO-SCN-105's gauge-registration
is the prerequisite for the whole spine-assembly chain** — every spine
seam is the same problem (per-segment submaps in their own gauges must
be Umeyama-registered into one gauge):

- **STO-SCN-097** — spine segmentation (chunk trajectory into M
  overlapping segments). Open.
- **STO-SCN-098** — global registration of segment submaps (pose-graph
  + loop closure + global BA). Open. **Directly depends on 105** (same
  Umeyama/`gauge_align`, applied at seams).
- **STO-SCN-099** — cohesive fusion into one gauge. Open. Depends on 098
  registering before fusing (else double-walls/gaps at seams).
- **STO-SCN-100** — whole-spine verification in the scout gaussian.
  Open. Can't trust the scout gaussian as the verification surface
  until 105's DA3-frame fix + `gauge_up` land.

105 itself: `depends-on: [STO-SCN-095, STO-SCN-048]`. Within
EPI-SCN-AUTO-SUBSET-SELECT: 095 (verify surface) is **gated on 105**;
103 `depends-on: [STO-SCN-093]` and supersedes 094; 104
`depends-on: [STO-SCN-103]`.
