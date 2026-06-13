---
xid: STO-SCN-094
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-13
depends-on: [STO-SCN-093]
bd-id: krabby-mft
---

# Coverage-greedy best-N selector over the co-visibility graph

## Summary

Given the posed pool + co-visibility graph, automatically propose the best-N views by
greedy coverage + connectivity maximization — the core selection algorithm.

## Context

The heart of the epic (design story conclusion #2). Consumes the track graph from
STO-SCN-093; produces the proposed-N that the scout/verify stage (STO-SCN-095) renders.

## Problem

Choose N views that maximize reconstruction quality: full surface coverage, good
triangulation angles, a connected view-graph, minimal redundancy. N targets the
downstream model's sweet spot.

## Design

### Approach

Greedy submodular coverage maximization on the track graph: iteratively add the view with
the largest marginal gain (newly-covered 3D points + improved triangulation angles),
subject to a connectivity constraint (each kept view retains sufficient pairwise overlap
with the selected set) and a baseline window (~10–30° intersection angles). Stop at N or
coverage saturation. Deterministic and testable.

### Changes

| File | Change |
|------|--------|
| selector | greedy coverage+connectivity over track graph → proposed-N |
| metrics | coverage map, triangulation-angle stats, redundancy report |

## Definition of Done

- [ ] Posed pool → ranked proposed-N maximizing coverage with connectivity preserved.
- [ ] Deterministic; emits a coverage report (covered surface, angle distribution, gaps).
- [ ] Validated on a known scene (proposed-N reconstructs at least as well as a hand pick).

## Spine note (longer-term — see STO-SCN-096 conclusion #7)

When the scene is a spine of M segments, selection is **not purely local**: each segment's
edge views must stay co-visible with its neighbors so the segments register into one
cohesive space. The greedy must therefore honor a **boundary-overlap budget** — reserve /
prefer views that maintain seam co-visibility with adjacent segments, keeping the *global*
co-visibility graph connected, not just the per-segment one. This is the **IN side of the
segment boundary contract** (STO-SCN-096): the spine passes `boundary_spec` (pinned anchor
frames + overlap to cover); the selector honors it. For a single space (M=1) the spec is
empty and the constraint is inert. The spine segmentation + global registration that consume this live in
the sibling epic (EPI-SCN-SPINE-ASSEMBLY).

## Out of scope

- Human override / rendering (STO-SCN-095).
- The pose solve (STO-SCN-093).
- Spine segmentation + cross-segment registration (sibling epic).
