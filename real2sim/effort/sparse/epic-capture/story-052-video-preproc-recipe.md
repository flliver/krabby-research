---
xid: STO-SCN-052
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-zv4
assignee: krabby
shipped: 2026-06-10
tasks: 4
complete: 4
---

# T0 video preprocessing: recipe book + hardened sharp-select

## Summary

A recipe book (`real2sim/RECIPES.md`) cataloguing every data type we
process (video, photo stills, large/multi-session pools, benchmarks)
with the precise hardened steps for each — and the hardening of the
one step that had no tool: blur-aware sharp frame selection, the T0
preprocessing step for videos.

## Context

Parent: [EPI-SCN-CAPTURE](./epic.md). Triggered by scene 013-basement
(first new video since the 001 quality verdict). Operator direction:
"Let's begin a recipe book of types of data we might process... and
precisely how to process it. Then... let's follow our recipe."

## Problem

001-patio's sharp-select was an ad-hoc prototype (spec says
`maturity: prototype`; no Laplacian code exists anywhere in the repo)
— "precisely how" lived in nobody's head and one scene's spec JSON.
Every new video would re-freelance the step (T-025 violation waiting
to happen). And there was no single document mapping data type →
process, so each new capture restarted the "how do we do this?"
conversation.

## Design

- `real2sim/RECIPES.md` — Recipes A (video) / B (photos) / C (large
  pools → photo spine) / D (benchmarks → repro pipeline) + the common
  trunk + hard-limits table. Grounded only in scenes actually
  processed (T-010).
- `real2sim/select_sharp_frames.py` — spec-driven, results-emitting
  (HUG-KRB-002), same contract shape as `normalize_photos.py`.
  Variance-of-Laplacian @480 px, sharpest per uniform temporal
  window. Selected indices + scores go in **results.json** (measured)
  — not baked into the spec post-hoc like the 001 prototype did.
- Frame-budget guidance encoded: 12 too few for yard-scale (001
  verdict); start 24 for room-scale+.

## Definition of Done

- [x] `RECIPES.md` covers all four data types we have actually
      processed, each step naming its hardened tool.
- [x] `select_sharp_frames.py` ships with the normalize_photos
      spec/results contract.
- [x] Recipe A followed end-to-end on 013-basement preprocessing:
      probe → lossless extract → sharp-select via the new script.
- [x] README points at RECIPES.md.

## Status notes

- 2026-06-10: Recipe book + script written; following Recipe A on
  013-basement (898-frame FFV1 pool, budget 24).

### Run log (013-basement, first recipe execution)

- Input was a saved Google Drive *preview page*, not the video;
  recovered the real file via the Drive file ID embedded in the HTML
  (230 MB FFV1 lossless, 672×376 @30fps, 29.9 s).
- 898 frames extracted lossless to `input/src/` (PNG, 318 MB).
- Sharp-select 24/898: pool scores 18.7–303.4 (median 108.8);
  selected 72.3–303.4 — every winner above the pool median's
  two-thirds mark; no degenerate frames selected.
- Normalize skipped per Recipe A step 4 (672 px ≪ 2048).
- Gotcha: Mac system python3 has no numpy/PIL — ran via
  `uv run --with numpy --with pillow`.
- 2026-06-10: SHIPPED. Recipe A executed end-to-end on 013-basement
  (extract 898 → pool-sharp-200 → pool-sfm 200/200 on s → operator
  curated 17 → train 673s → TSDF (0.2.2 patch) → orient → scene.blend
  → non-ideal-dark render). Scene behaved as the pre-registered
  negative control. Recipe book later extended with the per-phase
  catalog (STO-SCN-054..057).
