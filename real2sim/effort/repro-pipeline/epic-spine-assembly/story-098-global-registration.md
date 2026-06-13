---
xid: STO-SCN-098
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-13
depends-on: [STO-SCN-097, STO-SCN-093]
bd-id: krabby-1lm
assignee: krabby
---

# Global registration of segment submaps (pose-graph + loop closure + global BA)

## Summary

Bring the M per-segment submaps into **one global gauge**, correcting drift across the
spine via pose-graph optimization, loop closure, and global bundle adjustment.

## Context

The make-or-break for cohesion (STO-SCN-096 #7). Consumes per-segment poses + boundary
overlaps (STO-SCN-097/093). Without it, locally-good segments stay disjoint and drift
compounds along the spine.

## Problem

Each segment is solved in its own arbitrary gauge with its own drift. They must be aligned
into a single, globally-consistent frame using boundary co-visibility and any loop closures
(path revisits).

## Design

### Approach

Build a pose graph over segments: relative-pose edges from shared boundary frames + loop-
closure edges from revisits; optimize globally (pose-graph optimization), then optional
global BA over the merged tracks. Output: every camera in one gauge, drift-corrected.

## Definition of Done

- [ ] M submaps → single global gauge; relative drift across seams within tolerance.
- [ ] Loop closures applied where the path revisits.
- [ ] Globally-consistent poses emitted for fusion (STO-SCN-099).

## Out of scope

- Geometry fusion (STO-SCN-099); segmentation (STO-SCN-097).
