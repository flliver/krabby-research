---
xid: STO-SCN-075
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-11
depends-on: [STO-SCN-073]
bd-id: krabby-bd0
---

# Reproducibility harness: re-run from record on clean host, compare within tolerances

## Summary

Proof of the *reproducible* in "best reproducible pipeline": take a
pipeline_run's record alone, re-execute it on a different/clean host
via the 073 trigger, and compare outputs to the original within
declared tolerances. Absorbs the abandoned STO-SCN-041 verification
scope. This is the M11 gate — a winning configuration that can't be
reproduced from its record is not a deliverable.

## Context

The premise is already production-validated once: dbeeprz rebuilt 006
DA3 transients from tracked inputs+metadata in 12s and reproduced the
render with identical alignment (store-shape v2 work). The DTU
baseline repro (STO-SCN-041) established the failure chain a harness
must catch (rc-gate lies, LFS pointers, attribute drift). This story
systematizes both.

## Problem

Reproducibility is currently asserted per-anecdote. There is no tool
that takes a run record and answers "can this be reproduced, and how
close is the result?" — and GPU nondeterminism means "identical"
needs defined, per-output tolerances.

## Design

- Harness: `record → re-run on host X → compare`. Comparison is
  per-output-type with declared tolerances:
  - metadata/cameras: exact or epsilon
  - meshes: vert/tri counts, bbox, sampled surface distance
  - renders: image metrics vs original (threshold TBD from measured
    same-host variance — measure first, don't invent the tolerance,
    T-017)
- Tolerances live in the task catalog (070) next to the settings they
  govern.
- Verdict written back onto the pipeline_run as a reproducibility
  score (074 leaderboard can filter to "reproduced" runs).
- License check surfaced in the verdict: runs using CC-BY-NC weights
  (DA3 Giant/Large) are flagged not-deliverable regardless of score.

## Definition of Done

- [x] Same-host re-run variance MEASURED (tbeeprz, 8-giant-studio vs
      -var1): TSDF verts 0.0037% / tris 0.0053%, gaussian PLY verts
      bit-identical, align scale 0.005%. Tolerances = ~10× measured
      max (0.05%), recorded in the catalog (`x-task.tolerances`,
      single source — repro_check matches output patterns,
      most-specific wins). Verdict on the pair: **overall PASS**.
- [ ] **OPERATOR-GATED (second host):** reproduce the record on a
      DIFFERENT host within (cross-host) tolerance. Informative data
      already in hand: studio(t) vs historical 8-giant(d) differs
      0.44% — but that pair is uncontrolled (image 0.2 vs 0.4,
      different GPU), so the cross-host gate must be measured with a
      controlled re-run. One word names the host and I run it.
- [x] A deliberately broken record fails loudly with reasons named:
      backfilled record → "FAIL: backfilled record…", "FAIL: image
      digest not pinned", "FAIL: input hashes missing" (rc=1);
      synthetic complete record passes (rc=0).
- [ ] M11 deliverable recipe references the harness verdict as its
      evidence of reproducibility (after first PASS).

## Implementation Notes

- `real2sim/repro_check.py` — `check` (static record gate: digests
  pinned, settings snapshotted, inputs hashed, license flags =
  deliverable eligibility), `compare` (per-output deltas; mesh stats
  from PLY headers — no heavy deps; ABSTAIN on unmeasured
  tolerances), settings-differ detection ("not a reproduction pair,
  it's an A/B comparison").
- Backfilled records rank but can never gate M11 — enforced at the
  `check` level, matching STORE-SCHEMA-V3.md.
