---
xid: STO-SCN-016
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-0rw
priority: 0
title: T2.E1 — Scale Calibration Strategy ★ BLOCKER
assignee: principal
---

# T2.E1 — Scale Calibration Strategy ★ BLOCKER

## Summary

**Critical-path blocker for T2.** None of our captures contain reference objects of known size, so the absolute scale of every reconstruction is currently unknown.

## Context

**Critical-path blocker for T2.** None of our captures contain reference objects of known size, so the absolute scale of every reconstruction is currently unknown.

Required: a strategy that produces a uniform-scale post-hoc correction, applied before USD export.

Options to evaluate (none yet validated):
- Hand-measure a known real-world distance in each scene; apply uniform-scale correction.
- Use known camera baseline if multi-camera footage exists (it doesn't, currently).
- Use a known-scale object placed in scene for future captures (forward-only, doesn't help current scenes).

**Per Manager memo 2026-05-06 (top-3 risks #2):** this is the single biggest unknown blocking T2 acceptance.

## Definition of Done

- [ ] Strategy documented for retroactive calibration of existing scenes
- [ ] Concrete measurement performed for at least 1 scene
- [ ] Uniform-scale correction applied; verified by spawning a known-size primitive in IsaacSim and comparing
- [ ] Future-capture protocol defined (must include reference object)


## Journal Notes

No scale-calibration deliverable in the journal, but relevant scale-ambiguity notes: MAtCha's per-chart deformation MLP can re-scale geometry differently across runs (per-region depth ambiguity), so submap meshes can drift in scale even with agreed camera positions — proposed mitigations: anchor on the unified SfM sparse 3D points, a final shared-frame scale-alignment step, or skip per-chart deformation for scale-up. The reference-localization test saw a 1.6% scale difference (Procrustes 1.0156) between SfM frames. No reference-object-based metric calibration is described — confirming this story's "no fallback yet" blocker status.
_Sources: notes 2026-05-01T174650-submap-based-mesh-fusion, 2026-05-06T100000-auto-localized-reference-cameras._


## Handoff Notes

**Root cause** (manager audit 2026-05-06 + handoff-2026-04-29-1347.md): unsolved across ALL captures because no reference objects were in scene — a capture-side miss recorded in every `experiments/<scene>/CAPTURE-LESSONS.md` (and now codified in **HUG-SCN-004**). PLAN E1 calls it "unsolved across all our captures (no reference objects)." The manager audit ranks T2 the single biggest technical unknown (HIGH), with three stacked unknowns: scale calibration (no fallback), V-HACD collision proxy, and IsaacSim USD load + spawn + depth.

---
_Imported from legacy beads `m11-u3l` (M11 DAG re-import, 2026-06-03)._
