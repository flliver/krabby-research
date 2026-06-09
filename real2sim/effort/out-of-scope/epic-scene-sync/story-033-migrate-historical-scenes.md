---
xid: STO-SCN-033
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-04
depends-on: []
bd-id: krabby-mqy
assignee: principal
priority: 4
title: Migrate historical scenes into the schema (git-LFS data repo)
shipped: 2026-06-04
tasks: 7
complete: 7
---

# Migrate historical scenes into the schema (git-LFS data repo)

## Summary

The ~50 GB of flat, inconsistent M11 scene data, reorganized into the canonical
schema and committed to the dedicated git-LFS data repo at `/var/krabby/scenes`
— losslessly, via APFS CoW clones, with parity verified per unit.

## Context

Unblocked by `STO-SCN-026` (schema shipped). Consumes `inventory.md` (the
21-dir→~10-scene work-list) and `SCHEMA.md` (the target shape). Reconstructing
*legacy provenance from journals* — originally in this story's title — is split
out to **`STO-SCN-036`**; the `eval/`-artifact schema home is **`STO-SCN-037`**.

## Problem

The M11 scenes had no shared shape and couldn't be consumed/shared/published.
This story moves them into the schema so the rest of the epic (sync, Docker
consume, tiering) has conformant data to operate on.

## Design / what was done

Built a reusable converter (`real2sim/scenes/migrate.py`) with the logical-scene
mapping as a reviewable table, then migrated via **APFS CoW clones** (same external
volume → instant, 0 extra space) into `/var/krabby/scenes`:

- **10 logical scenes** from 21 dirs: `001-patio`, `002-patio`, `003-firepit`,
  `004-sky-house` (5 curated MAtCha runs + dining), `005-meadow`, `006`–`012-kubota`,
  `dtu-bicycle` (external). 2 empty `vggt` staging dirs dropped.
- **8 capture videos** → each scene's `input/`.
- `manifest.json` → `scene.toml` + `run.json` + `transform-*/​{specification,results}.json`
  (manifest-bearing runs = `measured`; legacy = `deduced`; raw = input-only).
- **Git LFS** for all large binaries; metadata in plain git. The audit caught a
  124 MB `points3D.txt`, 11 MB pointmap JSONs, and 1.9 GB `.blend1` backups before
  they reached plain-git history (LFS patterns + `.gitignore` fixed).
- Originals retained (CoW-shared) for verify-before-swap.

### Changes

| File | Change |
|------|--------|
| `real2sim/scenes/migrate.py` | add — reusable CoW migrator + mapping table |
| `/var/krabby/scenes/.gitattributes` | add — LFS patterns (incl. `**/sparse/**`, `**/pointmaps/**`) |
| `/var/krabby/scenes/.gitignore` | add — cruft (`.DS_Store`, `*.blend1`) |
| `/var/krabby/scenes/**` | add — 10 migrated scenes (3 commits, 42 GB, LFS) |

## Definition of Done

- [x] All 21 source dirs accounted for (10 scenes migrated, 2 empties dropped).
- [x] Migration is **lossless** — global parity verified (only `manifest.json` →
      `manifest.legacy.json` renamed; sample sha256 matches).
- [x] Large binaries in LFS; **no plain-git file > 1 MB** (bloat guard passes).
- [x] Capture `videos/` migrated into each scene's `input/`.
- [x] Committed to the `/var/krabby/scenes` git-LFS repo.
- [x] New store verified to hold real data independent of source (pre-swap check).
- [x] `sfm-scaling-out/` identified (SfM-scaling *experiment*, not a scene) → out of scope.

## Out of scope

- **Legacy provenance reconstruction from journals** → `STO-SCN-036`.
- **`eval/` schema home + re-sorting `_unsorted/`** → `STO-SCN-037`.
- **`sfm-scaling-out/`** — a benchmark experiment (N=24→500 MASt3R-SfM VRAM/scaling
  study; 300 comfortable, 500 OOM), already on S3 as `m11-sfm-scaling`. Not a scene.
- **Removing source originals (the swap)** — operator-gated (T-018).

## Implementation Notes

### What Changed
- Added a **`run-<slug>`** level (from STO-SCN-026) — the curated sweep needed it.
- Legacy multi-tool dirs map each tool subdir → `pipeline-<tool>/run-legacy/transform-01-legacy/data/`;
  cross-run eval artifacts parked in scene-level `_unsorted/` (→ `STO-SCN-037`).

### Gotchas
- **Same-volume APFS is everything**: CoW makes migration instant + free and keeps
  source as a 0-cost safety copy. git-lfs also CoW-dedups the object store.
- **Extension-based LFS is fragile**: large `.txt`/`.json` tool outputs slipped to
  plain git until directory-level LFS rules were added. A size-based pre-commit
  guard would be more robust (candidate for `STO-SCN-034`/devex).

## Status notes

- 2026-06-04: Picked up by principal (unblocked by STO-SCN-026).
- 2026-06-04: Migrated 10 scenes + 8 videos (42 GB) via CoW; parity-verified;
  committed (scenes repo `dbf1976`, `f691e4d`). Journal-provenance → STO-SCN-036,
  eval home → STO-SCN-037. Source removal operator-gated.

## Status notes (backfill addendum)

- 2026-06-09: **Legacy-outposts backfill executed** (scenes store `ac0766c` + `14432a5`):
  (1) `004-sky-house/input/preproc-{01,02}-frame-select-{12,16}/` created from the
  surviving curated frame sets (spec/results + per-frame sha256; selection method
  manual-viewer per manifest.legacy.json; story=STO-SCN-001) and wired as `inputs`
  on all six 004 matcha runs. (2) `cfg_args` (exact tool Namespace) + `run_logs/`
  (train.log + nvidia-smi.csv, 5 s cadence) copied into 7 runs' `data/`;
  specifications enriched with `tool_args_raw` + parsed `gs_*` measured params.
  Remaining in outposts, deliberately left for STO-SCN-037 (`eval/` home):
  `sfm-scaling-out/{n350,n500}`, `sfm-ref-localize/dtu-bicycle`. Outposts scripts are
  md5-identical to repo real2sim/ — no drift, no further unique data identified.
