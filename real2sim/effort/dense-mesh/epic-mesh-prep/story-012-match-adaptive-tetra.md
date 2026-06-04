---
xid: STO-SCN-012
parent: ./epic.md
kind: story
effort: scn
size: M
status: deferred
date: 2026-06-03
depends-on: []
bd-id: krabby-36r
priority: 3
title: T1.C3 — Match Adaptive Tetrahedralization Quality (nice-to-have)
---

# T1.C3 — Match Adaptive Tetrahedralization Quality (nice-to-have)

## Summary

Reproduce the adaptive-tetrahedralization quality shown in MAtCha paper (reference image b). Likely involves revisiting alignment configs and dense regularization parameters.

## Context

Reproduce the adaptive-tetrahedralization quality shown in MAtCha paper (reference image b). Likely involves revisiting alignment configs and dense regularization parameters.

**Per Manager memo 2026-05-06: this is decoupled from T1 acceptance.** TSDF satisfies T1; tetra-match is a quality bar we set ourselves and is *not* required for milestone acceptance. Pursue only if schedule allows after T2/T3/T4 land.

## Definition of Done

- [ ] Tetrahedral mesh produced with parameter set that visually matches paper reference (b)
- [ ] Parameters documented
- [ ] Render committed to journal alongside C2 evidence


## Journal Notes

Same 2026-05-04 pivot flagged that adaptive tetrahedralization (MAtCha's default extraction) was NOT yet reproduced at the paper's fidelity — a real gap since it's the default mode. Plan: download the `adaptive_tetra.png` reference, match its camera, and tune MAtCha params (alignment configs, dense regularization) until the tetra mesh visually matches. `localize_reference_image.py` supports this via `--reference-image …/adaptive_tetra.png --reference-name cam_ref_auto_tetra`; if the two auto cameras come out near-coincident, the published TSDF and tetra renders share one vantage. Deferred/non-gating (TSDF already satisfies watertight).
_Sources: notes 2026-05-04T123000-…, 2026-05-06T100000-…._

---
_Imported from legacy beads `m11-8u6` (M11 DAG re-import, 2026-06-03)._
