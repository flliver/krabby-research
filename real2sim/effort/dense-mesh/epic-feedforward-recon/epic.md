---
xid: EPI-SCN-FEEDFORWARD-RECON
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-10
hugs: []
tenets: []
bd-id: krabby-amu
---

# T1: Feed-forward reconstruction (VGGT-Omega / DA3) — dense hole-free geometry

## Problem Statement

Our MASt3R-SfM + MAtCha + TSDF pipeline produces great-looking
foregrounds with HOLES: sparse correspondence-based geometry truncates
wherever curated coverage (12–17 frames) doesn't overlap. Fletcher
(2026-06-10): "Getting closer though! Foreground looking nice. Def
look at vggt omega 1B as well, it's looking really impressive. Depth
anything 3 also looking really solid. Though produced splats instead
of point cloud." A new generation of FEED-FORWARD models predicts
dense per-pixel geometry for every view in a single transformer pass
— no SfM prepass — and is plausibly hole-free at our frame budgets.

## Goals

- Evaluate the two named models as pipeline transformations in the
  scene store (settings-attached, runoff-comparable — STO-SCN-058
  layout makes this systemic).
- First runnable: DA3 image + 006-kubota pilot (STO-SCN-060).
- Honest comparison vs `pipeline-matcha` on the SAME scenes/views in
  rate_renders.

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-059` | Research: VGGT-Omega + DA3 discovery record | shipped | S |
| 2 | `STO-SCN-060` | krabby-da3 image + 006-kubota pilot | shipped | M |
| 3 | `STO-SCN-061` | Frame alignment + render from saved views | shipped | M |
| 4 | `STO-SCN-065` | TSDF mesh fusion — depths → deliverable mesh | shipped | M |
| 5 | `STO-SCN-066` | DA3 hi-res mesh — process_res sweep (756 ceiling) | shipped | M |
| 6 | `STO-SCN-127` | reconstruct-da3-scout — matcha-free, solve-gauge mesh | shipped | M |

**Transformation inventory (each is a story):** inference+export
(060), view-aligned gaussian render (061), mesh fusion (065),
process_res sweep (066), matcha-free solve-gauge reconstruct (127).

## Closeout — shipped 2026-06-15 (DES-SCN-DENSE-MESH closeout)

**Epic goal met.** DA3 was evaluated as a pipeline transformation, made runoff-comparable
(STO-SCN-058 layout), and produces dense hole-free geometry vs matcha's holes — Fletcher's
point confirmed on our own scenes. DA3 is now a first-class runoff variant (gaussian render +
TSDF mesh + matcha-free solve-gauge path), **operator-validated** on DA3-24 (001-patio, "looks
good", 2026-06-15). All six stories shipped.

**Deferred (recorded, out of this closing milestone):** higher feed-forward fidelity beyond
process_res 1008 (needs >16 GB GPU / view-chunked inference / DA3-Streaming — STO-SCN-066);
VGGT-Omega evaluation never spun up (DA3 satisfied the dense-hole-free goal). Both are T1
reconstruction enhancements, revisit only if a contract deliverable demands them.

## Risks

| Risk | Notes |
|------|-------|
| DA3 large-model license | CC BY-NC 4.0 on Giant/Large — fine for internal research evaluation, NOT for contract deliverables without resolution (Apache-2.0 Base/Metric-Large are the clean tiers) |
| Splats ≠ meshes | DA3's native quality output is 3D Gaussians; our trunk consumes meshes. Mesh path (PLY/GLB export, or TSDF over DA3 depth) needs validation |
| Repro gap | Community reports difficulty reproducing DA3's showcased splat quality (GitHub issue #44) |
