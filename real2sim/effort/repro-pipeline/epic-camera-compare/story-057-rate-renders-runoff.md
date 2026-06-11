---
xid: STO-SCN-057
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-7m2
shipped: 2026-06-10
tasks: 1
complete: 1
---

# Phase: rate_renders ranking runoff app

> Retroactive phase documentation (operator directive 2026-06-10).
> Recipe section: `real2sim/RECIPES.md` § Common trunk step 7.

## What we did

Ran the operator-facing ranking runoff over the comparison-render
matrix: pairwise/ordinal ranking of variant renders per view, across
all 12 runoff scenes + 013. Output `rankings.jsonl` per scene
(committed to the store 2026-06-10) — the ground-truth preference
data the settings runoff exists to collect.

## Where the code is

- `real2sim/rate_renders/` (`server.py` + app) — serves on **:8090**.
- Input: `scenes/<scene>/comparison_renders/<view>/<variant>.png`
  (the matrix layout from `render_comparison_matrix.sh`,
  STO-SCN-045).
- Output: `scenes/<scene>/rankings.jsonl`.

## How

1. Render the matrix for the scene(s) (RECIPES.md trunk step 6).
2. `python3 real2sim/rate_renders/server.py` (or confirm it's already
   up on :8090 — it auto-discovers scenes from the store).
3. Operator ranks in the browser; rankings append to the scene's
   `rankings.jsonl`.
4. Commit the rankings into the store — they are data, not ephemera
   (T-018; nearly lost as untracked files, caught 2026-06-10).

## Notes

This phase is operator-in-the-loop BY DESIGN (T-020): render quality
judgments are not self-closeable by the agent.

## Definition of Done

- [x] Phase documented here + RECIPES.md section.
