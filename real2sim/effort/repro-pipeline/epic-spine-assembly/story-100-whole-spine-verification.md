---
xid: STO-SCN-100
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-13
depends-on: [STO-SCN-099, STO-SCN-095]
bd-id: krabby-wua
assignee: krabby
---

# Whole-spine verification (assembled space + seams in the scout gaussian)

## Summary

Verify the *assembled* multi-segment space — seams included — in the scout gaussian:
confirm segments align, no drift gaps, no doubled surfaces, full coverage end-to-end.

## Context

Extends the per-segment verification (STO-SCN-095) to the whole spine. The human QA gate
for cohesion (STO-SCN-096 #6, #7).

## Problem

Per-segment QA can pass while the *assembly* fails (seam misalignment, accumulated drift,
coverage gap between segments). Verification must operate on the whole, with seams
highlighted.

## Design

### Approach

Render the assembled cohesive gaussian; overlay segment boundaries / seams and the global
camera trajectory; let the human spot misalignment, drift, gaps, or doubled geometry and
flag the offending segment/seam for re-registration or re-selection.

## Definition of Done

- [ ] Assembled space rendered with seams + trajectory highlighted.
- [ ] Human can confirm cohesion or flag a specific seam/segment for rework.
- [ ] Pass = single drift-free space handed to condition/export.

## Implementation Notes

**Surface = same two-pass viewer as STO-SCN-095** (GaussianSplats3D splats + overlay pass),
reused on the **assembled** cohesive gaussian rather than a single segment — this is the
`100 → 095` edge. Overlays: segment-boundary / seam markers, the global camera trajectory,
and an end-to-end coverage heat so gaps *between* segments are visible.

**What the human checks** (things per-segment QA can't catch): seam misalignment,
accumulated drift along the spine, coverage gaps in the inter-segment regions, and doubled
geometry that survived fusion.

**Rework routing.** A flagged defect routes to the responsible stage rather than a blanket
re-run: misalignment/drift → STO-SCN-098 (re-register); doubled/holey seam → STO-SCN-099
(re-fuse); a structurally bad cut → STO-SCN-097 (re-segment); an under-covered segment →
STO-SCN-094 (re-select that segment). This is the loop-back named in Out of scope.

**Pass criterion.** A single drift-free cohesive space → handed to condition/export
(STO-SCN-013). This is the human gate (T-020) for the whole assembly — it does not
self-close.

**Test.** The assembled space renders with seams highlighted; a seeded inter-segment
misalignment is visibly spottable by the operator.

## Out of scope

- Fixing flagged seams (loops back to STO-SCN-097/098/099).
