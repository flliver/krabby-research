---
xid: STO-SCN-077
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-11
depends-on: [STO-SCN-076, STO-SCN-071]
bd-id: krabby-56b
---

# Backfill: materialize Studio records for all existing runs (provenance recovered or marked unknown)

## Summary

Every existing run in the store (001–013: matcha sweeps, DA3
branches, render-variant runs, _migrated legacy) gets materialized v3
records — pipeline_instance extracted from its specs, run_record
assembled from recoverable provenance, rankings attached as scores —
so the historical corpus competes in the leaderboard on equal footing
with new Studio-triggered runs.

## Context

The existing runoffs ARE the experimental evidence the Purpose says
we're mining (006/007/008 verdicts: tetra #1, holes disqualifying,
etc.). STO-SCN-071's adapters make them *readable*; this story makes
them *first-class records*. Precedent for the tool shape:
`migrate_renders_into_runs.py` (STO-SCN-058 backfill — idempotent,
refuses overwrites) and the legacy-provenance reconstruction work
(STO-SCN-036).

## Problem

Historical runs predate the run-record contract: image digests,
TOOLS_GIT_SHA, and input hashes were captured inconsistently (or not
at all for pre-v2/_migrated runs). Rankings reference variant labels,
not run records. Without backfill, the leaderboard starts empty and
months of operator ranking judgment is stranded.

## Design

- Idempotent backfill tool (real2sim/, run via the standard hardened
  path): scans every `pipeline-*/run-*`, derives:
  - pipeline_instance ← spec JSONs (settings as recorded)
  - run_record ← results JSONs + image labels + git history where
    recoverable
  - scores ← scene rankings.jsonl entries mapped via variant label
    `<p>--<r>`
- **T-002 hard rule:** unrecoverable provenance fields are written as
  explicit `"unknown"` with a `backfilled: true` marker — never
  inferred. The 075 harness treats `backfilled+unknown` runs as
  non-reproducible-by-record (they still rank; they can't gate M11).
- Refuses overwrites; re-run produces zero diff (verified).
- Dry-run mode prints the plan; T-007: operator sees the dry-run
  summary (counts per scene, unknowns ratio) before the real write.

## Definition of Done

- [ ] All runs in all scenes have v3 records; count reported per
      scene.
- [ ] All historical rankings appear as scores attached to runs
      (006/007/008 runoff verdicts visible in the 074 leaderboard).
- [ ] Unknown-field ratio reported honestly per run-vintage (v2 runs
      should be near-complete; _migrated mostly unknown).
- [ ] Idempotency proven: second run = zero diff.
- [ ] Store still clean for all pre-existing files (backfill is
      additive only).
