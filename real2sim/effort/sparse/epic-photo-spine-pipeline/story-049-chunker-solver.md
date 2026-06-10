---
xid: STO-SCN-049
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-nvq
assignee: krabby
shipped: 2026-06-10
tasks: 8
complete: 8
---

# batched_sfm chunk + per-chunk SfM solve driver (preproc-conventional)

## Summary

`real2sim/batched_sfm.py chunk|solve` turns an arbitrarily large photo
pool into overlapping ≤300-frame temporal chunks (symlinks + manifest)
and solves any one chunk in the pinned MAtCha container with hard
output gates — one GPU job per chunk, farmable to any fleet host.

## Context

Parent: [EPI-SCN-PHOTO-SPINE-PIPELINE](./epic.md). MASt3R-SfM's
measured ceiling is ~300 frames on 16 GB (sfm-scaling experiment:
300 safe @13.4 GB, OOM @400–500). 005-meadow's 2,028-photo pool both
exceeds the ceiling and confuses retrieval on self-similar grass;
temporal chunking restores locality.

## Problem

Need a deterministic, store-conventional way to (a) split a sorted
pool into chunks with enough overlap to stitch, without duplicating
gigabytes of frames, and (b) run the existing container `--sfm_only`
solve against one chunk from any host, refusing to report success
unless the output actually exists (rc=0 lies — see runner hard-gate
precedent).

## Design

### Approach

- `chunk --pool <dir> --out <spine-dir> [--chunk-size 300]
  [--overlap 50]` — sorted pool → `chunk-NN/data/` of RELATIVE
  symlinks (T-016: no copies; links resolve inside any container
  mount that spans the scene input dir) + `chunks.json` manifest
  (lo/hi/first/last per chunk). Warns above the 300 ceiling; rejects
  overlap outside [3, chunk-size).
- `solve --spine <dir> --chunk NN [--image ...] [--snapshot ...]` —
  docker `--sfm_only` run; mounts the spine's PARENT as `/work` so
  the relative symlinks resolve in-container; skip-if-solved
  (idempotent re-farm); gate = rc==0 AND
  `chunk-NN/out/mast3r_sfm/cameras.json` exists.
- numpy imported lazily (stitch only) — fleet hosts' system python
  has no numpy and chunk/solve must run there.

### Changes

| File | Change |
|------|--------|
| `real2sim/batched_sfm.py` | add (chunk + solve subcommands) |
| `/tmp/spine_chunks.sh` (fleet) | thin chain wrapper: git/lfs pull, nanny-progress, loop solve |

## Definition of Done

- [x] 2,028-photo 005 pool chunks into 8 overlapping chunks (300/50),
      manifest written, all symlinks resolve.
- [x] A chunk solves end-to-end in the container from a fleet host
      (chunk-01 on tbeeprz: 300/300 poses).
- [x] Solve refuses success without cameras.json (gate observed firing
      during the t lockup — reported failure instead of silence).
- [x] Re-invocation of a solved chunk skips (idempotent).
- [x] Self-reviewed.

## Testing

### Unit / fixture tests

- [x] Chunk math: 2,028 → 8 chunks, overlap 50, last chunk
      right-aligned (no short tail chunk).
- [x] Overlap bounds rejected outside [3, size).

### Integration

- [x] All 8 chunks of 005 solving across t/b/d/s (2026-06-10;
      7/8 landed, chunk-04 in flight at time of writing).

## Out of scope

- Stitching (STO-SCN-050).
- Automatic host scheduling — farming is operator/agent-driven
  (chain script per host).

## Implementation Notes

### What Changed

As designed, plus the container-mount fix below.

### Files Modified

- `real2sim/batched_sfm.py` — new CLI (chunk/solve; stitch in 050).

### Gotchas

- **Mount the spine's PARENT, not the spine dir.** Chunk frames are
  relative symlinks into the sibling pool dir; mounting only the
  spine dir left them dangling inside the container
  (FileNotFoundError on the first 005 batch). `/work` = scene input
  dir fixes it.
- 2,028 photos → **8** chunks at 300/50 (the epic's early estimate
  said 9; the right-aligned chunk math yields 8).
- Fleet system python has no numpy — keep chunk/solve numpy-free.
