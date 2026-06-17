---
xid: STO-SCN-162
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: []
bd-id: krabby-3vqk
priority: high
assignee: scout
---

# Preprocess gap: run SELECT (best-N FINAL-N) + repoint primary so reconstruct gets a DA3-sized subset

## Summary

The Preprocess pipeline runs the missing **`select`** step (best-N voxel-coverage
→ the **FINAL-N subset**, ≤ the DA3 view ceiling) and **points `primary` at it**,
so `scout` + `reconstruct-da3` consume a DA3-sized camera set instead of the full
spine.

## Context

Found in operator UI testing (003-firepit, 2026-06-16): the pipeline produces the
spine (`solve`, 183 cams) but **never reduces it to DA3's size**. `DA3_VIEW_CEILING
= 32` (measured, 16 GB). My STO-SCN-150 pipeline deliberately (and wrongly)
excluded `select` as "a separate concern" — but `select` is what emits the
**FINAL-N subset** that `reconstruct` consumes (RECIPES step 6 → step 12
"consumes the subset designated `primary`").

```
have:  precull → solve → covis →            scout → reconstruct(primary=full spine ✗)
need:  precull → solve → covis → SELECT(n≤32) → scout → reconstruct(primary=FINAL-N ✓)
```

## Problem

Two coupled holes in the pipeline's reconstruct handoff:
1. **No `select`** → no FINAL-N subset; `reconstruct-da3` would consume the full
   183-cam spine → over the 32 ceiling → OOM/garbage.
2. **`primary` never repointed** to the FINAL-N (repointing is a locked
   operator-only act, #1) → reconstruct reads the stale full-spine primary
   (also the cause of the earlier "pipeline solved the stale primary" symptom).

## Design

- Add a **`select`** phase to `pipeline_run.PHASES` after `covis`:
  `v4exec select <scene> --solve <id> --covis <id> --selector voxel --n 24`
  (resolve the covis id from the store like the solve id).
- **Repoint `primary`** at the emitted FINAL-N subset (the pipeline is an
  explicit operator action, so this is sanctioned within the run) — then `scout`
  + `reconstruct` consume it.
- **Long trajectories** (spine beyond one best-N's coverage): a later `--segment`
  mode runs `spine → spine-register → spine-fuse` instead; out of scope here.

### Changes

| File | Change |
|------|--------|
| `pipeline_run.py` | add `select` phase (resolve covis id); repoint primary to FINAL-N; thread the FINAL-N to scout/reconstruct |
| `rate_renders/static/scenes-pipeline.js` | render the new phase |
| `tests/test_pipeline_run.py` | plan includes select; covis-id resolution |

## Definition of Done

- [ ] Pipeline runs `select` → a FINAL-N subset (≤ DA3 ceiling) materializes.
- [ ] `primary` ends pointed at the FINAL-N; scout + reconstruct consume it.
- [ ] Preview Plan shows the select phase with the resolved covis id + n.
- [ ] **Operator-verified (T-020):** 003 reconstruct-da3 runs on the FINAL-N (~24), not 183.

## Out of scope

- The select ALGORITHM itself (exists — EPI-SCN-AUTO-SUBSET-SELECT / voxel_coverage).
- Spine segmentation/fusion for over-ceiling trajectories (`--segment`, later).
- Symmetric orient (STO-SCN-161 — sequenced FIRST).
