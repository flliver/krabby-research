---
xid: STO-SCN-075
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
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

- [ ] Same-host re-run variance measured first; tolerances derived
      from data, recorded in the catalog.
- [ ] One historical winner (e.g. matcha 8-strong-tetra branch on
      006) reproduced from its record on a different host, within
      tolerance.
- [ ] A deliberately broken record (missing input hash / wrong image
      digest) fails loudly with the reason named.
- [ ] M11 deliverable recipe references the harness verdict as its
      evidence of reproducibility.
