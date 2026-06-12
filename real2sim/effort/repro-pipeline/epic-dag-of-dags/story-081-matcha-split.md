---
xid: STO-SCN-081
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: []
hugs: [HUG-SCN-005]
bd-id: krabby-35u
shipped: 2026-06-11
tasks: 4
complete: 4
---

# Matcha monolith split: verify train-from-external-SfM + standalone tetra extraction (unwelds represent/meshify)

## Summary

Verify the matcha monolith can split into pose / represent / meshify
(unwelding locked #6's fused execution).

## Verification (2026-06-11)

**SUPPORTED at our pinned commit** (`b119fd96`, the SHA baked into
krabby-matcha — verified against the raw train.py source, not just
current docs):

- `--sfm_only` — solve standalone (already production: pool-SfM)
- `--alignment_only` / `--refinement_only` — charts+train standalone;
  each stage references the previous stage's output dir
  (`mast3r_scene_path` feeds forward) → train-from-external-SfM is
  the supported invocation, no code changes
- `--mesh_only` — tetra extraction standalone, with previously
  unexposed tunables discovered: `--tetra_downsample_ratio`
  (default 0.5), `--use_multires_tsdf`, `--no_interpolated_views`
- No auto-resume logic exists — stages run unconditionally when
  invoked. That is exactly right for us: OUR planner decides what to
  invoke (materialize-check, locked #4); the tool just runs its stage.

**Consequence:** `matcha@1` = per-stage invocation — an executor
change, not a fork. The locked-#6 weld (`@0`) and its compute-waste
corner disappear when the executor adopts stage flags.

**Honest caveat (T-002):** doc+source verified; the empirical
stage-chained run (solve → train → mesh as three dispatches on a GPU
host) is gated on the v4 executor + fleet re-clone of the migrated
store — it is the natural FIRST v4-native matcha job, and
`--tetra_downsample_ratio` should be promoted to a tunable in the
meshify-via-tetra def at that point.

## Definition of Done

- [x] Train-from-external-SfM verified (stage flags + forward-feeding
      output dirs at pinned SHA).
- [x] Standalone tetra extraction verified (`--mesh_only`).
- [x] matcha@1 path defined (executor-level, no fork).
- [x] New tunables recorded for promotion (tetra_downsample_ratio).
