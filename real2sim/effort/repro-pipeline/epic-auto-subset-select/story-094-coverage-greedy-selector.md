---
xid: STO-SCN-094
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-13
depends-on: [STO-SCN-093]
bd-id: krabby-mft
assignee: krabby
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

- [x] Posed pool → ranked proposed-N maximizing coverage with connectivity preserved.
      (`select_views.py`: greedy new+triangulated(angle-weighted), min-overlap connectivity.)
- [x] Deterministic; emits a coverage report (coverage %, triangulation-angle stats,
      pct-in-10-30°). Validated on the real 539 pool (deterministic; 7 s).
- [ ] Proposed-N reconstructs ≥ a hand pick — **deferred** (needs a reconstruct run +
      hand-pick baseline; the selector + report are delivered).

## Result (2026-06-14) — selector works; coverage is capture-limited

Coverage-vs-N on the real 539 pool (`6EHLYO3MF3QU`):

| N | triangulated-coverage | median tri-angle |
|---|---|---|
| 12 | 5.4% | 5.3° |
| 24 | 9.0% | 6.2° |
| 48 | 17.4% | 6.3° |
| 120 | 34.7% | 5.7° |

Coverage scales **~linearly with N — no small-N knee** because the hyperlapse tracks are
*thin* (~3.4 views/point), so a point only counts when *both* its observers are selected.
The selector is correct; the limit is the capture. ⇒ (a) the scout-gaussian (095) is where a
human judges "enough coverage?" / bumps N, and (b) reinforces the capture-lessons call for
**deliberate orbits / denser overlap** (thick tracks → small-N captures most coverage).
Possible refinement (noted): for thin-track pools, add a *spatial/viewpoint-diversity* term
to the objective (feed-forward reconstructors value view coverage, not just triangulated pts).

**Remaining (follow-ups):** wire as a v4 store node (`select@0` under the covis, mirroring
093); the "reconstructs ≥ hand-pick" comparison.

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

## Implementation Notes

**Algorithm (greedy submodular maximization).** Maintain a covered-point set `C`. Each
step, add the unselected view `v` with the largest **marginal gain**:
`gain(v) = Σ over points v sees but C doesn't, weighted by triangulation-angle quality`
(angle quality peaks in ~10–30°, falls off outside), **subject to a connectivity
constraint**: `v` must share ≥ K points / sufficient pairwise overlap with the current
selection (so the kept view-graph never fragments — the failure the 300-frame drift
exhibited). Stop at N, or when marginal gain < ε (coverage saturated). Deterministic via a
fixed index tie-break.

**N target.** The downstream model's sweet spot — parameterized: ~12–17 for the existing
reconstruct graphs, ~32 for a DA3 feed-forward pass. Caller supplies N.

**boundary-overlap budget (spine IN, M=1 inert).** When `boundary_spec` is non-empty:
pre-seed the selection with the pinned anchor frames and bias `gain` toward seam-overlap
coverage so adjacent segments stay co-visible. The constraint is the IN side of the
segment boundary contract (STO-SCN-096); for a single space it's empty and ignored.

**Coverage report (the selector's "eyes", T-012).** Emit covered-surface %, a
triangulation-angle histogram, an explicit **gap list** (surface regions under-covered),
and per-view marginal contribution. This report is what STO-SCN-095 renders for the human.

**Test.** On a known scene, the proposed-N must reconstruct **at least as well** as a
hand-picked set of the same size (the falsifiable bar — T-001).

## Out of scope

- Human override / rendering (STO-SCN-095).
- The pose solve (STO-SCN-093).
- Spine segmentation + cross-segment registration (sibling epic).
