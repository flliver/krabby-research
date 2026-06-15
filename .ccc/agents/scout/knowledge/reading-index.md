# Reading index — reconstruct the whole effort

> Ordered so someone could rebuild the scene-reconstruction effort from
> scratch. Read the narratives first (they carry the *why*), then the
> source. Durable lessons distilled in [`lessons.md`](lessons.md).

## The process, documented — read this first

0. **`scene-processing/`** — `real2sim/knowledge/scene-processing/README.md`
   — *The canonical operator-facing process doc set* (T0→T4, seven
   phases). The documented shape of the whole pipeline this index
   reconstructs from source. Read the README index, then the per-phase
   docs for the tier you're working. Tracked under
   `EPI-SCN-M11-PROCESS-DOCS`. The STO stories below are the *why
   behind the how* — read them after the process docs for root-cause
   depth.

## Start here — the narrative + root cause

1. **STO-SCN-105** — `real2sim/effort/repro-pipeline/epic-spine-assembly/story-105-scout-gauge-registration-diagnostic.md`
   — *The canonical write-up.* DA3 normalized-frame root cause, the
   documented red herrings, the photo-match diagnostic tool, and the
   fix. **Read fully** — source of truth for this whole effort.
2. **STO-SCN-095** — `epic-auto-subset-select/story-095-scout-verify-handoff.md`
   — Scout-gaussian verification surface + handoff; its "Result/learnings"
   section carries the durable rules (dynamicScene, de-warp, gauge-up,
   never-rewrite-the-.ply). Verify DoD is gated on 105.

## The selector work (other half of the session)

3. **STO-SCN-103** — `epic-auto-subset-select/story-103-voxel-coverage-selector.md`
   — Voxel-coverage best-N view selector (the coverage-optimization
   greedy that replaced the track-greedy).
4. **STO-SCN-104** — `epic-auto-subset-select/story-104-fisherrf-infogain-selector.md`
   — FisherRF info-gain selector (deferred reserve) + the explicit
   "why deferred" integration-cost list.

## Context epics

5. **EPI-SCN-SPINE-ASSEMBLY** — `epic-spine-assembly/epic.md`
   — where 105 lives; 097–100 (segmentation / registration / fusion /
   verification) all depend on 105's gauge-registration.
6. **EPI-SCN-AUTO-SUBSET-SELECT** — `epic-auto-subset-select/epic.md`
   — 091–095, 103, 104 (extract → pre-cull → pose → auto-select →
   scout → human-verify → FINAL N; the splat is the QA lens, not the
   selector).
7. **STO-SCN-048** — `effort/sparse/epic-photo-spine-pipeline/story-048-gauge-align-module.md`
   — `gauge_align` (the Umeyama/Procrustes sim(3) module the fix
   reuses) + the two-pass coplanar-rotation design + the MASt3R
   non-orthonormality gotcha. Shipped.

## Source to read alongside

- `real2sim/gauge_up.py` — gravity from posed cameras (⟂ to all
  camera-right; robust to pitch). The orientation half of the fix.
- `real2sim/voxel_coverage.py` (+ `real2sim/tests/test_voxel_coverage.py`)
  — the selector; the model for a CPU, gauge-free, testable selector.
- `real2sim/verify_viewer/{build_verify.py, viewer.html, match.html}`
  — the photo-match diagnostic + serve-dir builder (de-warp, gauge-up,
  serve-only cull-sphere with self-verify). `build_verify.py` is the
  most instructive.
- `real2sim/da3_render_view.py` — **the MODEL for the fix**: already
  does `w2c[:3,3] /= scale_factor` + orientation-augmented `gauge_align`
  with a residual hard-fail. Copy this pattern.
- `real2sim/da3_infer_posed.py` — **where the fix GOES** (capture
  `scale_factor`; persist the `colmap` export — `export_format` already
  requests it, but nothing downstream consumes it).
- `real2sim/v4exec.py` `cmd_scout` — the gather; today rsyncs only
  `*.ply` + `*.npz` (`--exclude=*`), dropping the colmap export +
  `scale_factor`, and the `*.npz` it keeps is DA3's **echoed input**
  cameras, not the gaussian-frame cameras — so it persists nothing that
  can register the gaussian.

## External references (the selector research)

- **Coverage Optimization for Camera View Selection** — arXiv
  **2604.05259** (2026): sample K of N posed cameras, greedy-add
  most-diverse-baseline subject to whole-scene visibility. The published
  formulation STO-103 implements (not a local invention). Sibling:
  arXiv 2207.08434 (set-cover variant).
- **FisherRF** — ECCV 2024 (oral), *Active View Selection and
  Uncertainty Quantification for Radiance Fields using Fisher
  Information* — arXiv **2311.17874**: Fisher-information EIG view
  ranking on a 3DGS backend. The basis for STO-104; built for
  sequential NBV, hence the batch-rank adaptation cost.

> Beyond this list, for long-horizon research use the OLAI corpus:
> `/ask knowledge@olai 3d-reconstruction <question>` (see
> [`README.md`](README.md)).
