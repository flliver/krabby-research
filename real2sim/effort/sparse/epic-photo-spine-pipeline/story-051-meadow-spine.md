---
xid: STO-SCN-051
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: [STO-SCN-049, STO-SCN-050]
bd-id: krabby-3ad
assignee: krabby
---

# 005-meadow: full 2,028-photo spine (8 chunks, fleet-farmed) + curation handoff

## Summary

005-meadow's complete 2,028-photo capture posed in one unified gauge
via the photo-spine pipeline, loaded into the curation viewer so the
operator can select the reconstruction subset — replacing the garbage
poses the single whole-pool solve produced.

## Context

Parent: [EPI-SCN-PHOTO-SPINE-PIPELINE](./epic.md). This is the
production run of STO-SCN-048/049/050 and the epic's reason to exist:
the 2026-06-10 whole-pool attempt at 005 produced unusable poses
(retrieval mismatches on self-similar meadow grass + pool 6.7× over
the solve ceiling).

## Problem

Pose all 2,028 photos accurately enough to identify critical camera
positions for 005's reconstruction runs, with per-stitch residual
evidence that the merged gauge is trustworthy.

## Design

### Approach

1. `chunk`: 2,028 photos → 8 chunks (300 frames, 50 overlap) at
   `005-meadow/input/spine-01/`.
2. `solve` × 8, farmed across the full fleet (operator: "Farm out to
   all available hosts"): t (01, 08), b (02, 03, 04), d (05, 06),
   s (07).
3. `stitch` (gate 0.10 m) → `spine_cameras.json` + `stitch_report.json`.
4. Gather chunk solves into the scene store, push, load spine into
   camera_viewer for operator curation (T-020 — the operator exercises
   the surface; this story does not self-close).

## Definition of Done

- [ ] 8/8 chunks solved (7/8 at time of writing; chunk-04 on b).
- [ ] 7/7 stitches under the 0.10 m residual gate, 2,028 unique poses
      in spine_cameras.json.
- [ ] Solves + spine pushed to the scene store.
- [ ] Operator curates from the unified spine in camera_viewer and
      confirms poses are usable (T-020 — operator-verified close).

## Testing

### Integration

- [ ] stitch_report.json residuals reviewed per stitch (the
      production validation of STO-SCN-050's gate).
- [ ] Spot-check: spine poses for chunk-01 frames match chunk-01's
      solo solve (identity transform — chunk 1 IS the reference gauge).

## Out of scope

- Running 005's reconstruction (gaussians/mesh) from the curated
  subset — next effort step after curation.
- Re-solving with different chunk parameters unless a stitch breaches
  the gate.

## Implementation Notes

### Run log

- 2026-06-10 morning: chunked 8×300/50; chunk-01 launched on t.
- t locked up mid-effort TWICE; chunk-01 had completed (10:12) before
  the second lockup. Root cause (ops@baeprz): **no swap — RAM use
  past 32 GB hard-locks the host.** Fix: 63 GiB persistent swapfile
  fleet-wide (fstab). b validated the fix within minutes (28/30 GiB
  RAM + 770 MiB swapped during chunk-02, no lockup).
- Fan-out: chains on b (02–04) and d (05–08); operator then directed
  full-fleet farming → d's queue cut after 06, 08 moved to t,
  07 raced between d and s. s won: the 40.5 GB self-contained
  0.2.1 image was shipped d→s (zstd over SSH) — s solved 07 at the
  fleet's fastest rate AND is now a permanent matcha-capable host
  (closes part of the STO-SCN-038 distribution tail).
- 11:52: 7/8 solved (01,02,03,05,06,07,08). 04 in flight on b.
- 12:15: 8/8 solved; gathered to store (8×300 poses, all valid).
- **First stitch attempt failed loudly** (by design): 01↔02 overlap
  disagreed at mean 4.3 in a span-39 gauge. Diagnosis: chunk-01's
  TAIL (frames 200–300 — exactly the 01↔02 overlap) carries 17.9-unit
  trajectory teleports; chunk-02's head is stable. Chunk-01's only
  shared frames with the spine were its bad ones.
- Stitcher redesigned (see STO-SCN-050 § Production redesign):
  consensus alignment + relative gate + chain order. Chunks 02–08
  stitched clean: 1,778 poses, 6/6 gates passed, 146 low-confidence.
- **Bridge chunk-09** (pool frames 150–450) minted + solving on s:
  overlaps chunk-01's healthy mid (100 frames) and chunk-02 (200
  frames), giving chunk-01's good frames a path into the spine.
  Final chain order: `2,3,4,5,6,7,8,9,1`.

### Gotchas

- Chunk solves land root-owned on fleet clones (docker) — gather step
  must account for ownership when syncing into the store; root-owned
  untracked outputs also make fleet `git pull` abort SILENTLY-ish
  (s sat on a stale commit; pull printed "Updating" then aborted).
  Chown via a throwaway container, clear, re-pull.
- Bridge symlinks must be minted in the SAME portable relative form
  as the chunker's (`../../../<pool>/<frame>`); deriving relpath from
  the manifest's resolved pool path on a host with a different mount
  prefix (/Volumes vs /var) produces Mac-only links that dangle on
  the fleet.
