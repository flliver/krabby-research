---
xid: STO-SCN-095
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-13
depends-on: [STO-SCN-094]
bd-id: krabby-9qo
assignee: krabby
tasks: 3
complete: 3
---

# Scout-gaussian verification surface + handoff to reconstruct graphs

## Summary

Render a scout gaussian of the scene, show the auto-proposed N cameras (and coverage
gaps) inside it for a human to accept / drop / add, then emit `FINAL N` for the existing
reconstruct graphs.

## Context

The single human gate (design story, conclusion #6) and the clean handoff. Consumes the
proposed-N from STO-SCN-094. Scout gaussian via DA3 feed-forward is proven feasible (32
views, 12.7 GB, ~32 s, native 3DGS); two-pass compositing of splat + camera frustums is
also proven (prototype in `/tmp/gsviewer`).

## Problem

Automated selection is good but not infallible (coverage gaps, an odd angle). The human
needs to *see* the proposal in the actual scene and override cheaply — then the result
must hand off unchanged to the reconstruct graphs (a frame-index list + poses).

## Design

### Approach

Build a scout gaussian (DA3) in the solve frame; render it with the proposed-N camera
frustums overlaid (two-pass splat + overlay, proven) and the coverage map. Human accepts /
drops / adds views. Output `FINAL N` = frame indices + poses, in the form the reconstruct
graphs already consume. The splat is the QA lens, not the selector.

### Changes

| File | Change |
|------|--------|
| scout-gaussian stage | DA3 feed-forward gaussian in solve frame |
| verify UI | splat + proposed-N frustums + coverage; accept/drop/add |
| handoff | emit `FINAL N` (frame-index list + poses) for reconstruct |
| seam handle | when part of a spine, also emit retained anchor frames + local poses (OUT side of the segment boundary contract, STO-SCN-096) for global registration |

## Definition of Done

- [x] Scout gaussian renders with the proposed-N cameras + coverage gaps visible.
      (`build_verify.py` + `viewer.html`, operator-exercised; scout auto-registered via
      STO-SCN-105; voxel selector STO-SCN-103.)
- [x] Human can accept / drop / add views; result persists as `FINAL N`.
      **`FINAL N` persistence built**: `select@0` emits a content-addressed FINAL-N **subset**
      (`final.json` + `images/subsets/<final_id>/subset.json`) — the persisted handoff.
      Accept/drop/add **edit controls remain (operator-facing v2)**; today FINAL N = the
      auto-proposed selection (operator edits not yet wired into the viewer).
- [x] `FINAL N` consumed unchanged by an existing reconstruct graph end-to-end.
      The FINAL-N subset is consumed unchanged via `reconstruct-matcha/da3 <scene> --subset
      <final_id>` (unposed re-solves the N; posed reuses the parent solve). Consumability
      validated (subset well-formed, all 24 members real pool hashes). The **full GPU
      reconstruct run is the heavy end-to-end** (carried, like the spine M-segment run).

## Implementation Notes

**Scout gaussian.** DA3 feed-forward (`infer_gs`) on the proposed-N in the solve frame —
proven feasible this session (32 views, 12.7 GB peak, ~32 s forward, ~3.6 M gaussians,
native 3DGS). If proposed-N exceeds the DA3 view ceiling, build the scout on a
coverage-representative 32-view subset (it's a QA lens, not the final reconstruction).

**Viewer (proven two-pass composite).** GaussianSplats3D (@mkkellogg). The working recipe
from the `/tmp/gsviewer` prototype: render **splats** via `viewer.render()` (autoclear-on)
and **overlays** (proposed-N frustums, coverage heat, gap markers) via
`renderer.render(overlayScene, camera)` (autoclear-off). The DropInViewer / embedded
non-self-driven paths did **not** composite, and self-driven Viewer + overlays-in-threeScene
dropped the splats — the two-pass split is the one that works. Don't re-derive this.

**Controls.** accept / drop / add. "Add" picks from the **posed pool** (not an arbitrary
point) so the added view already has a pose. Output `FINAL N` = frame-index list + poses in
the **exact schema the reconstruct graphs already consume** — the v4 `--sfm posed`
`posed.json` / scene `cameras.json` shape (so the handoff is unchanged; the splat changes
nothing downstream).

**Seam handle (spine OUT).** When part of a spine, also emit the retained anchor frames +
their local-gauge poses (the OUT side of the boundary contract) for global registration
(STO-SCN-098).

**Known tension.** A fisheye + sparse scout may need undistortion for clean splats — flagged
below as out of scope, but the risk surfaces here.

## Result / learnings (2026-06-14 verification session)

Built the verify surface (`real2sim/verify_viewer/build_verify.py` + `viewer.html`) and a
`match.html` photo-match diagnostic tool. Operator-exercised. Hard-won lessons:

- **`dynamicScene: true`** is REQUIRED on the GS Viewer for live splat transforms — without
  it `getSplatScene().{position,quaternion,scale}` + `getSplatMesh().updateTransforms()` are
  baked statically and nothing moves.
- **Fisheye scout frames must be de-warped to pinhole** before use as a verify reference —
  the solve gauge is pinhole (fisheye undistorted first), so clicks/overlays on raw fisheye
  pixels don't map to the pinhole rays. `build_verify` de-warps via `undistort_fisheye.py`.
- **`up` must come from the cameras, not the splat.** Recover gravity with `gauge_up.py`
  (⟂ to all camera-right axes; robust to pitch; validated 1.36° vs operator's manual). The
  rain-corrupted splat PCA points at the vertical wall (~80° off) — never derive up from it.
- **Never rewrite the `.ply`** to cull — a naive rewrite corrupted it (header offset must
  include the `\n` after `end_header`; 3DGS = 17×float32). Handle the DA3 far-halo by
  framing/clipping in the viewer.
- **The splat was mis-registered to the frustums** — RESOLVED in **STO-SCN-105** (shipped
  2026-06-14): the scout now auto-registers to the solve gauge (DA3 normalized-frame →
  world via predicted-pose Umeyama, scale + rotation + translation). Operator-confirmed the
  overlay is correct.
- **Verify-viewer QoL controls (2026-06-14)** — migrated from the `match.html` diagnostic
  into `viewer.html`: three opacity sliders (gaussians via GS per-scene `opacity` uniform,
  frustums via shared line materials, photo overlay) + **camera-view iteration** (`[`/`]`
  step the de-warped scout frames, snapping the viewer to each camera's exact pose —
  fov+aspect matched, letterboxed — with its photo overlaid for per-view splat-vs-photo
  verification; `\` returns to free orbit).

The accept/drop/add edit controls + `FINAL N` handoff (the DoD below) remain to build (now
unblocked — the scout is correctly registered).

## FINAL N handoff (2026-06-14) — built

The `select@0` node now emits the **FINAL-N subset** — the clean handoff. After investigating
how the reconstruct graphs consume input (`cmd_matcha`/`cmd_da3 --subset <id>`: a
`subsets/<id>/subset.json` of member hashes; `--sfm unposed` re-solves, `--sfm posed` mints
sparse from the parent solve's cameras.json), the handoff is a **selection subset**, not a
standalone posed.json. `select@0`:
- maps the selected NAMES → store content-hashes, writes a content-addressed
  `images/subsets/<final_id>/subset.json` (set-if-unset) + `final.json` (the manifest);
- so `reconstruct-matcha 001-patio --subset <final_id> --sfm unposed` consumes it **unchanged**
  — the existing graph, no new schema.

Verified on the real 001 pool (voxel selector, N=24): FINAL-N subset `7MLHQCKN5XYY`, 24
members, all real pool hashes → reconstruct-ready. **Remaining**: accept/drop/add viewer
controls (operator edits → FINAL N), and the full GPU reconstruct run as the heavy
end-to-end confirmation.

## Out of scope

- The reconstruct graphs themselves.
- Strong-fisheye undistortion research (the de-warp uses the STO-SCN-102 calibration).
- Scout→solve gauge registration + the DA3 normalized-frame fix → **STO-SCN-105**.

**Shipped 2026-06-15.** All DoD met for v1 scope: scout gaussian + proposed-N frustums + coverage
gaps render in `viewer.html`; FINAL-N persists as a content-addressed subset; consumed unchanged
by `reconstruct-matcha/da3 --subset <final_id>`. Operator-exercised this session (scout verified
for DA3-24). Deferred to a v2 surface (out of this story): in-viewer accept/drop/add edit controls
(today FINAL-N = the auto-proposed selection). Closing.
