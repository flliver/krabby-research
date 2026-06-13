---
xid: STO-SCN-100
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-13
depends-on: [STO-SCN-099, STO-SCN-095]
bd-id: krabby-wua
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

## Out of scope

- Fixing flagged seams (loops back to STO-SCN-097/098/099).
