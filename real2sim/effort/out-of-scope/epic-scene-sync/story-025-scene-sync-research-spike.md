---
xid: STO-SCN-025
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-04
depends-on: []
bd-id: krabby-0e1
title: Scene Synchronization — research spike (what must we build?)
assignee: principal
---

# Scene Synchronization — research spike (what must we build?)

## Summary

**Research spike — intentionally open-ended.** Investigate and scope what we would need to build to *synchronize scenes* (align/fuse multiple sub-scenes into one coherent, scale-consistent, walkable environment). Output is a findings write-up + recommended design direction + a sized backlog of follow-on stories — **not** an implementation.

## Context

This sits in the **TX — Out of Scope Work** bucket: it is **out of M11 scope**. Every M11 capture is a single room that fits one MAtCha run, so no cross-scene synchronization was needed. Forward work (M12+) and multi-submap captures will require it.

**Historical forensics — start here.** Prior design thinking already exists and should be recovered before any new design:
- Milestone journal: `threads/matcha-quality/notes/2026-05-01T174650-submap-based-mesh-fusion…` and `2026-05-04T120000-submap-fusion-strategy-detailed…` (the original submap-fusion / camera-"spine" strategy).
- **HUG-SCN-001** (TSDF >> tetra) and **STO-SCN-016** (scale-calibration ★) — the scale-drift and watertightness caveats that any synchronization scheme inherits.
- Note: the milestone journal tree is the richest source and may be archived/moved — capture what matters into this story as you go.

## Problem

We don't yet know the *shape* of the synchronization system. "Synchronize scenes" is undefined: does it mean camera-spine registration across runs? TSDF re-fusion at overlaps? a shared scale anchor? coordinate-frame reconciliation? watertightness restoration after merge? This spike exists to turn that ambiguity into a concrete, decision-ready picture.

## Approach (spike)

1. **Recover prior art** from the journal + notes above; summarize the submap-fusion strategy as previously conceived.
2. **Enumerate sub-problems** — registration/alignment, scale anchoring, overlap/gap handling, TSDF re-fusion, watertightness, coordinate frames.
3. **Survey tooling** already in hand — MAtCha `extract_tsdf_mesh.py` (multires TSDF fusion), Open3D `ScalableTSDFVolume`, MASt3R-SfM as the cross-run pose source.
4. **Produce** a recommended design direction + a sized list of follow-on stories (and any open decisions as AIQs).

## Definition of Done

- [ ] Prior submap-fusion thinking recovered from the journal and summarized here (before it's archived away)
- [ ] Sub-problems of "scene synchronization" enumerated
- [ ] Existing-tooling survey (what we can reuse vs must build)
- [ ] Recommended approach written up (design memo or AIQ for the open decisions)
- [ ] Sized backlog of follow-on stories proposed; explicit in/out-of-scope call for the next milestone

---
_Research spike — no implementation expected. Created in the TX (out-of-scope) bucket, 2026-06-04._
