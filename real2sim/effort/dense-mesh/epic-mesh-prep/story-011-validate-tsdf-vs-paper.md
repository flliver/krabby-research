---
xid: STO-SCN-011
parent: ./epic.md
kind: story
effort: scn
size: M
status: deferred
date: 2026-06-03
depends-on: []
bd-id: krabby-ayp
priority: 3
title: T1.C2 — Validate TSDF Mesh Quality vs Paper Reference (nice-to-have)
assignee: krabby
---

# T1.C2 — Validate TSDF Mesh Quality vs Paper Reference (nice-to-have)

## Summary

Render best TSDF mesh from auto-localized reference perspective; visually compare to MAtCha published TSDF reference image. Confirm we have already met (or exceeded) reference quality.

## Context

Render best TSDF mesh from auto-localized reference perspective; visually compare to MAtCha published TSDF reference image. Confirm we have already met (or exceeded) reference quality.

**Per Manager memo 2026-05-06: this is decoupled from T1 acceptance.** TSDF satisfies the watertight grant criterion; this validation is an additional self-imposed quality check, not a milestone-acceptance requirement.

Auto-localization landed; remaining work is the actual visual comparison + evidence capture.

## Definition of Done

- [ ] 3-up render comparison committed to journal: reference image | TSDF render | tetra render (auto camera)
- [ ] Verdict captured: matches / exceeds / misses paper quality
- [ ] If misses: gap analysis for tuning


## Journal Notes

The 2026-05-04 planning pivot inserted this Phase-C validation before USD/IsaacSim: formally compare a render of the best bicycle TSDF mesh against MAtCha's published TSDF reference (`data/scenes/dtu-bicycle/reference/tsdf_multires.png`; note the milestone-root `reference_images/` path is stale 30-byte stubs and PLAN.md still points there). Team believed the TSDF meshes already meet the bar but had not formally benchmarked. The cam_ref/auto-localize machinery (STO-SCN-008/010) was built to enable this; as of the journal the A/B/C comparison was set up but not formally signed off — consistent with this being deferred/non-gating.
_Sources: notes 2026-05-04T123000-planning-pivot-to-matcha-reference-validation, 2026-05-06T100000-…._

---
_Imported from legacy beads `m11-2cd` (M11 DAG re-import, 2026-06-03)._
