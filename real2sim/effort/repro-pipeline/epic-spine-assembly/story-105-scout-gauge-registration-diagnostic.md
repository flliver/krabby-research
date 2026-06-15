---
xid: STO-SCN-105
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-14
shipped: 2026-06-14
depends-on: [STO-SCN-095, STO-SCN-048]
bd-id: krabby-7di
assignee: krabby
tasks: 6
complete: 6
---

# Scout-gauge registration: DA3 normalized-frame root cause + photo-match diagnostic tool

## Summary

The raw DA3 scout gaussian renders **mis-registered** against the solve's posed frustums
(wrong ground/up, wrong scale, cameras floating above the ground). We built an interactive
**photo-match diagnostic tool** to find out why, and traced it to a concrete pipeline gap:
**DA3 (posed mode) emits its gaussians in its own normalized scale (`scale_factor`), echoes
the *input* cameras unchanged, and the scout pipeline never registers the gaussian back to
the FastMap solve gauge** — and the gather dropped the one artifact (DA3's output cameras /
`scale_factor`) that would let us align. This story documents the diagnostic tool, the
root-cause investigation (including the dead-ends), and the fix.

Belongs in EPI-SCN-SPINE-ASSEMBLY because **every spine seam is the same problem**: submaps
live in per-segment gauges and must be Umeyama-registered into one cohesive gauge
(STO-SCN-098/099). The DA3-frame finding + `gauge_up` are prerequisites for trusting the
scout gaussian as the whole-spine verification surface (STO-SCN-100).

## Context

- Upstream: STO-SCN-095 (scout-gaussian verification surface) renders the scout splat +
  the posed frustums for human verification. The frustums (FastMap `sparse/0`) and the
  splat (DA3) were assumed to share a frame; they do not.
- Reuses STO-SCN-048 `gauge_align` (Umeyama/Procrustes-with-scale on camera sets) — the
  same machinery `da3_render_view.py` already uses to register DA3↔matcha for rendering.
- Origin: operator verification session 2026-06-14. The operator manually recovered the
  gauge correction by hand (the photo-match tool below), which became the ground truth that
  exposed the pipeline gap.

## Problem

A reconstruction/scout is only useful if its gaussian and its cameras live in the **same,
upright, correctly-scaled** frame. Two things were broken:

1. **Orientation (gauge freedom).** SfM/FastMap reconstructs only up to an arbitrary
   similarity — the gauge has no absolute up/scale/origin. The pipeline never oriented it,
   and the viewer hard-coded `up=[0,-1,0]`.
2. **Scale/registration (DA3 normalization).** The scout gaussian is ~2× larger than the
   SfM points and offset, because DA3 normalizes its depth output by an internal
   `scale_factor` and the scout never undoes it.

## Findings (the investigation, in order — including the dead-ends, so we don't repeat them)

### What turned out to be TRUE (root cause)

- **DA3 posed-mode echoes the input cameras; it does NOT re-emit them in its own frame.**
  The scout `da3_poses.npz` `extrinsics` are **byte-identical** to the FastMap input
  (Umeyama input↔npz = scale `1.0000`, residual `0.0000`). So there is nothing to align on
  *there* — those are "the cameras we handed DA3, handed back."
- **DA3 normalizes its gaussian/depth output by `scale_factor`** (≈2.1 on 001-patio): the
  scout gaussian core std `[1.54,0.93,1.48]` vs SfM points `[0.75,0.53,0.57]` ⇒ ~2.1×, plus
  a centroid offset. `da3_render_view.py` already compensates (`w2c[:3,3] /= scale_factor`).
  **The scout path drops `scale_factor` and saves the raw, normalized `gs_ply`.** Result:
  frustums (solve frame) overlay a gaussian (DA3 normalized frame) ⇒ ~2× scale + offset ⇒
  the operator's exact symptoms (cameras too high, wrong ground/scale).
- **The gather threw away the registration data.** DA3's `export_format` includes `colmap`
  (it *does* re-emit "here's where I think the cameras are," in the gaussian frame), but
  `cmd_scout` left `exports/` empty and kept only the `gs_ply` + the (echoed) npz. The one
  similarity we needed — DA3's output cameras in the gaussian frame, or its `scale_factor` —
  was never persisted.

### What was a RED HERRING (do not chase these again)

- **"avg-camera-up is wrong" — false.** Gravity *is* recoverable from the poses: the
  direction ⟂ to all camera-right axes (cameras have ~3° median roll) lands **1.36°** from
  the operator's hand-clicked up; avg-camera-up is 3.1°. The poses had gravity all along —
  see `gauge_up.py`. The viewer's hard-coded `up` + the rain-corrupted splat PCA made the
  correct up *look* wrong.
- **"the splat PCA gives the ground" — false.** 001-patio was shot in the rain; reflections
  make the splat's dominant plane the **vertical cabin wall** (~80° off gravity). Never
  derive up/ground from the (rain-corrupted) splat geometry.
- **"the solve collapsed to a plane" — false.** RANSAC said "99% of points in a thin plane"
  but that was an **outlier-inflated threshold** artifact; trimmed point-cloud out-of-plane
  ratio is **0.166** (a normal flattish patio, not degenerate). The validity gate correctly
  PASSED (`out_in_ratio=0.0545`) — it measures *camera-track* planarity (a normal walk),
  which is fine.
- **".ply distance-cull" — broke the splat.** Rewriting `scout.gs.ply` with a naive cull
  corrupted it (off-by-one: the header byte offset must include the `\n` after
  `end_header`; 3DGS is 17×float32). Reverted. Handle the DA3 far-halo in the *viewer*, not
  by rewriting the .ply.

## The diagnostic tool (what we built to find the above)

`real2sim/verify_viewer/` — a local, browser-based gauge-alignment surface, served by
`build_verify.py`:

- **`viewer.html`** — two-pass GaussianSplats3D + three.js: scout splat + posed frustums
  (proposed-N green, pool gray) + an axis gizmo + a ground grid. Now orients up from
  `gauge_up` automatically.
- **`match.html`** — the **photo-match tool**. Camera **locked to a chosen photo** (de-warped
  to pinhole, letterboxed 1:1) as a fixed reference; the operator manipulates the gaussian
  to register it. Key mechanics earned the hard way:
  - **`dynamicScene: true`** is REQUIRED — GaussianSplats3D only applies live per-scene
    transforms (`getSplatScene().position/quaternion/scale` + `getSplatMesh().updateTransforms()`)
    when dynamic mode is on; otherwise the transform is baked statically and nothing moves.
  - **Vanishing-point axis solve** (fSpy-style): click origin + an UP point + a RIGHT point
    (seed lines), then add parallel line-pairs per axis (`X`/`Y`/`Z`); each axis direction =
    least-squares **vanishing point** (smallest eigenvector of `Σ LᵢLᵢᵀ`) → camera ray
    through the VP = world direction. VP sign is ambiguous → orient to the seed click. Reports
    an **X–Y squareness residual** ("all lines agree within tolerance"). One plane minimum.
  - **Operator-frame transform model**: the gaussian transform is `M(p)=R·s·p+t` in the
    **locked widget frame** — translate is world (widget) axes; rotate/scale pivot about the
    locked origin so "whatever is at 0,0,0 stays at 0,0,0." Widget snaps home after each drag.
  - Undo (BACKSPACE, one entry per drag), phase nav (ENTER / shift-ENTER), 3D corner
    lock-points that re-project across views, `[`/`]` to step frustums, paste-to-resume JSON.
- **`build_verify.py`** — assembles the serve dir: selects proposed-N, emits `frustums.json`
  (poses, vfov/aspect, gauge `up` from `gauge_up`, scene framing), **de-warps the fisheye
  scout frames to pinhole** (must match the solve gauge — clicks on fisheye pixels don't map
  to pinhole rays), and serves with selective caching (images cached, html/json no-store).

**Important meta-lesson:** the photo-match tool's "match the splat to one photo" goal is the
*opposite* of "reorient to upright" — DA3 built the splat from that photo, so it already
matches at identity; reorienting to upright makes it *stop* matching. The tool is good for
*confirming* a solve (at s=1), not for *building* the correction. The correction must be
computed (from cameras / VP), not hand-tuned against one photo.

## The fix — TWO attempts; the second is correct (verified on a real tbeeprz scout)

### Attempt 1 (scale_factor) — DISPROVEN, do not revive

Theory: gaussian frame = solve frame / `scale_factor` about the origin (from
`da3_render_view.py`'s `w2c[:3,3] /= scale_factor`), so scale the splat by `scale_factor`.
Implemented, then a **real scout on tbeeprz (001-patio, 2026-06-14) disproved it**:

- `scale_factor` (3.481) correctly maps DA3's **colmap points** to the solve (measured
  ratio 3.26) — but the **gs_ply we actually display is a DIFFERENT frame**: a top-down
  occupancy scan *degrades* toward 3.48; the gs_ply is ~2.4× the solve and **translated
  ~2.9 in z**, which `scale_factor` does not describe.
- Reason: `da3_render_view` uses an **unposed** inference and scales the *camera*; the scout
  is **posed** and the gs_ply exporter lands in its own re-centred/re-scaled frame.

### Attempt 2 (direct point-cloud registration) — CORRECT ✅

Register the **gs_ply directly against the solve's own `sparse/0/points3D.bin`** — both are
point clouds of the same scene, in the store; no DA3 internals trusted. Posed-mode → no
rotation, so it's a similarity `p_solve = scale·p_gs + translate`:

1. **`scout_register.py`** ✅ NEW (pure-stdlib, tested) — robust core trim (drops the DA3
   halo); `scale` = core-RMS ratio (NOT IoU-optimized: the occupancy grid is scale-biased —
   verified the IoU search drifted to 0.23 while the true scale was 0.40, core-RMS recovered
   0.40 exactly); `translate` = center-match + small IoU translation refine (unbiased).
   Synthetic recovery exact; **real 001 scout: scale 0.417, translate [−0.57,−0.01,−0.86],
   overlap IoU 0.50**.
2. **`build_verify.py`** ✅ — registers via `scout_register.register_scout(scout_dir,
   sparse_dir)`; emits `splat_scale` + `splat_translate`; frames in the solve gauge
   (`scale·core + translate`, clamped to cameras); culls in the gaussian frame; drops the
   vacuous npz `da3_align`.
3. **`viewer.html` / `match.html`** ✅ — `dynamicScene:true`; apply `scale` + `position`
   (= translate) to the splat (identity rotation). `match.html` starts/resets from the
   registered scale+translate → now **optional confirmation**, not the corrector.
4. **Orient** — `gauge_up.up_from_poses()` supplies the viewer `up` (unchanged).

Kept-but-not-the-splat-transform: `da3_infer_posed.py`/`cmd_scout` still persist
`scout_gauge.json` (`scale_factor`) + the colmap export — correct **provenance** for the
colmap frame, harmless. `scout_gauge.py` is provenance-only.

**Remaining:** operator builds the verify surface on the new scout (`OZGYMJTRXN3Z`) and
confirms the splat overlays the frustums (T-020).

## Artifacts (this session)

| File | What |
|------|------|
| `real2sim/gauge_up.py` | NEW — gravity from posed cameras (⟂ camera-right; robust to pitch; 1.36° vs manual) + roll-spread confidence |
| `real2sim/verify_viewer/build_verify.py` | verify-surface builder: VP/voxel select, de-warp frames, gauge `up`, scene framing, scout-frame export |
| `real2sim/verify_viewer/viewer.html` | scout + frustums viewer (gauge up, grid, axes) |
| `real2sim/verify_viewer/match.html` | photo-match diagnostic tool (VP solve, widget-frame transform, undo, resume) |
| `real2sim/posed_from_sparse.py` | + `read_cameras_intrinsics` (K + w/h for frustum FOV) |
| `real2sim/v4exec.py` | `cmd_scout` gathers `scout_gauge.json` + colmap export; records transform in metadata |
| `real2sim/da3_infer_posed.py` | **THE FIX** — captures DA3 predicted poses (monkeypatch) → Umeyama → gs→world similarity in `scout_gauge.json` |
| `real2sim/scout_register.py` | NEW — `gauge_for` (read transform; manual override > da3-umeyama > unregistered); `register` (point-cloud, retained as the disproven dead-end) |
| `real2sim/scout_gauge.py` | provenance-only (`scale_factor`; NOT the splat transform — kept for the colmap frame) |
| `real2sim/knowledge/da3-gsply-normalized-frame.md` | durable KB entry: DA3 normalized-frame mechanism + camera-Umeyama fix + don't-use-ICP |
| `real2sim/tests/{test_scout_gauge,test_scout_register}.py` | unit tests (readers + recovery) |

## Definition of Done

- [x] Diagnostic tool built + operator-exercised; manual gauge correction recovered.
- [x] Root cause identified: DA3 normalized-frame gaussian + dropped registration data.
- [x] `gauge_up.py` recovers gravity from poses (validated 1.36° vs manual).
- [x] Scout re-runs on tbeeprz disproved the scale_factor theory (`OZGYMJTRXN3Z`) and the
      point-cloud-ICP theory (87% fit / 147° wrong — scene symmetry), driving the camera-pose
      Umeyama solution.
- [x] **Registration solved automatically + verified**: `da3_infer_posed.py` captures DA3's
      raw predicted poses (monkeypatch of `align_poses_umeyama`) and Umeyamas them → input
      poses for the exact gs→world similarity (scale + ROTATION + translation), written to
      `scout_gauge.json["transform"]`; `scout_register.gauge_for` reads it (manual photo-match
      sidecar overrides if present); `build_verify`/`viewer.html`/`match.html` apply the full
      quat+scale+translate. Re-run scout `3R7ZB5GAB6PC`: **0.0% camera residual**; the auto
      transform fits the solve point cloud BETTER than the operator's hand photo-match
      (34.6% vs 26.3% within 3%).
- [x] **OPERATOR (T-020):** operator confirmed the registered splat overlays the frustums
      ("This looks correct", 2026-06-14).

## Status Notes

- 2026-06-14 (c): **SOLVED automatically.** Web research (DA3 source) revealed the real
  mechanism: DA3 builds the gs_ply in a normalized (camera-0-recentered + median-distance)
  frame, Umeyama-aligns its PREDICTED poses to our input poses inside `inference()`, but
  DISCARDS that transform for the gaussians (`align_to_input_ext_scale=True` only rescales
  depth + echoes input cameras). Fix wired: `da3_infer_posed.py` monkeypatches
  `align_poses_umeyama` to CAPTURE DA3's raw predicted poses, then Umeyamas them → our input
  poses to recover the exact gaussian→world similarity (scale + ROTATION + translation),
  written to `scout_gauge.json["transform"]`. `scout_register.gauge_for` reads it (manual
  photo-match sidecar overrides if present); `build_verify`/`viewer.html`/`match.html` apply
  the full similarity (quat+scale+translate). Re-run on tbeeprz (scout `3R7ZB5GAB6PC`):
  **0.0% camera residual**, and the auto transform fits the solve point cloud BETTER than the
  operator's hand photo-match (34.6% vs 26.3% of points within 3% — the manual ~9° was
  hand-eye error). Knowledge recorded in `knowledge/da3-gsply-normalized-frame.md`. Geometry
  ICP confirmed unusable (87% fit / 147° wrong — scene symmetry). Remaining: operator T-020
  visual confirm in the 3D viewer.
- 2026-06-14 (a): Implemented attempt-1 (scale_factor) code path.
- 2026-06-14 (b): **Ran a real scout on tbeeprz to verify — it DISPROVED attempt-1** (T-001
  win: caught before shipping to the operator). `scale_factor` maps DA3's colmap points but
  NOT the displayed gs_ply (which is ~2.4× + z-translated). Replaced with **direct gs_ply↔
  solve point-cloud registration** (`scout_register.py`): scale = robust core-RMS ratio,
  translation = center-match + IoU refine. Verified on synthetic (exact recovery) and the
  real scout (scale 0.417, translate [−0.57,−0.01,−0.86]). Remaining = operator T-020 visual
  confirm of the overlay.

## Out of scope

- Spine segmentation / global registration / fusion (STO-SCN-097/098/099) — this story is
  the *gauge-registration prerequisite* they all depend on.
- Metric scale from a known real-world distance (separate; only orientation + intra-gauge
  registration are needed to position frustums in the gaussian).
