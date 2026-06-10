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

## Status notes

- 2026-06-09: **First baseline landed.** Manual reproduction of `004-sky-house
  run-12-strong` on tbeeprz: REPRODUCED at metric equivalence (focal 0.001%,
  Umeyama pose residuals max 0.06% of scene scale, mesh counts +3.4%, duration
  −3.5%) under driver drift 595.58.03→610.43.02. Canonical record:
  `scenes/004-sky-house/pipeline-matcha/run-12-strong-repro-20260609/`
  (+ comparison.md). The comparison.md metric table is the seed spec for this
  harness. Also surfaced: j-hub LFS transport never worked for new objects
  (ops@baeprz installing git-lfs-transfer; emergency path = rsync into
  .git/lfs/objects + push --no-verify, used once, banned henceforth).
- 2026-06-09 (dtu reproduction attempt — failure chain root-caused, T-003):
  first run "succeeded" in 11 s having done nothing. Three real defects found+fixed:
  (1) **runner trusted tool rc** — MAtCha train.py os.system-chains stages and
  exits 0 on stage crashes → runner now hard-gates on expected outputs (rc 97);
  (2) **un-smudged LFS pointers fed the tool** → runner now refuses pointer
  inputs; j's LFS store was also incomplete for pre-transport history → full
  50 GB / 4,813-object backfill pushed from the Mac (sole complete holder);
  (3) **.gitattributes case-sensitivity drift** — macOS git matches `*.jpg`
  case-insensitively, Linux doesn't → Mac-committed `.JPG` pointers could never
  smudge on the fleet; uppercase patterns added to the store's .gitattributes.
  Also noted: container writes run as root → host user can't clean run dirs
  (sudo needed once); runner should set --user or chown (follow-up, STO-SCN-040).
  Re-run is in flight with all gates green ("12 images found", GPU active).
