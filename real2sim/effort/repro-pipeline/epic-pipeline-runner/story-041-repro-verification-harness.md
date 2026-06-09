---
xid: STO-SCN-041
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-09
depends-on: []
bd-id: krabby-ju5
title: Reproduction verification harness — rerun + compare within tolerances
priority: 1
assignee: krabby
---

# Reproduction verification harness — rerun a recorded spec, compare within tolerances

## Summary
`verify_reproduction`: take an existing run's spec, re-execute via the runner, and compare new vs recorded results — declared metrics, declared tolerances, pass/fail report committed alongside the run.

## Context
Bit-exactness is not achievable (driver drift 595.58.03→610.43.02; RTX 5080 nondeterminism documented in the corpus). The 2026-06-09 manual reproduction established the comparison set: duration (648 s reference), peak VRAM (7874 MiB reference), cameras.json focal + 12 poses, mesh statistics (vertex/face counts, bbox). ICA §7.2 makes reproducibility an acceptance criterion — this harness is its proof mechanism.

## Definition of Done
- [ ] Comparison metrics + tolerances defined per transform class (reconstruction vs conditioning)
- [ ] Camera comparison: focal + Procrustes pose residuals vs recorded cameras.json
- [ ] Mesh comparison: count/bbox/volume-class stats vs recorded artifacts
- [ ] Report format committed into the run dir; exit code reflects verdict
- [ ] The 2026-06-09 `004-12-strong` reproduction re-expressed as a harness run (first regression baseline)
