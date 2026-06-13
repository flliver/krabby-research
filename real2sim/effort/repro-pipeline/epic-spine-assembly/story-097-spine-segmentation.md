---
xid: STO-SCN-097
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-13
depends-on: [STO-SCN-091, STO-SCN-092]
bd-id: krabby-duy
assignee: krabby
---

# Spine segmentation (chunk trajectory into M overlapping segments)

## Summary

Partition a long video/trajectory into M **overlapping** segments, each individually
tractable for the per-segment pipeline, with shared boundary frames that guarantee
neighbors can register.

## Context

Outer loop of EPI-SCN-SPINE-ASSEMBLY; feeds each segment into
EPI-SCN-AUTO-SUBSET-SELECT. Longer-term scaffold (STO-SCN-096 #7).

## Problem

Cut points and segment size must respect: solver capacity (segment small enough to pose),
boundary overlap (segments share enough co-visible frames to register), and natural scene
structure (don't cut across a poorly-overlapping gap).

## Design

### Approach

Overlapping-window segmentation along the temporal/trajectory spine; size to solver
capacity; place boundaries where boundary overlap is high. Loop points (path revisits) are
flagged for loop-closure constraints downstream (STO-SCN-098).

## Definition of Done

- [ ] Long pool → M overlapping segments, each within solver capacity.
- [ ] Boundary overlap between adjacent segments meets a registrability threshold.
- [ ] Loop/revisit candidates flagged for global registration.

## Implementation Notes

**Segmentation.** Overlapping temporal windows along the trajectory. Window size capped by
**solver capacity** (≤ a few-hundred frames for SfM / the DA3 view ceiling per segment).
Overlap is a tunable stride so adjacent windows share ≥ the **boundary-overlap budget**
(the registrability threshold consumed by STO-SCN-098).

**Boundary placement.** Prefer cut points where inter-frame overlap is *high* (don't cut
across a low-overlap gap, or the seam won't register). Use the pre-cull / co-visibility
signal where a coarse pass is available; otherwise fall back to trajectory speed / frame
similarity.

**Loop / revisit detection.** Cheap global-descriptor or pHash similarity across
**non-adjacent** windows (reuse the STO-SCN-092 pHash) flags candidate loop closures —
handed to STO-SCN-098 as extra pose-graph edges.

**Output = the spine's per-segment `boundary_spec`** (the IN contract that STO-SCN-094
honors): pinned anchor frames + overlap region per neighbor, plus the global `camera_model`
(STO-SCN-091, identical for every segment). This is precisely the `097 → 091,092` edge.

**Test.** A multi-segment walk splits into M windows each within capacity; every adjacent
pair clears the overlap threshold; at least one path-revisit is flagged.

## Out of scope

- Registration / fusion (later stories).
