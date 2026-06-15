---
xid: STO-SCN-094
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
shipped: 2026-06-14
date: 2026-06-13
depends-on: [STO-SCN-093]
bd-id: krabby-mft
assignee: krabby
tasks: 4
complete: 3
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
- [x] Wired as a v4 store node (`select@0`, gated behind a PASSing covis). `tasks/select.json`
      + `v4exec.py cmd_select` + `select` subcommand. Verified end-to-end on the real 539-pool
      solve (001-patio / `6EHLYO3MF3QU` / solve `62QEHJDAJZBI` / covis `L57FPDHY2DRG`):
      content-addressed artifact `select/OBQTTTCF6RH7` (selection.json + posed.json + metadata),
      idempotent NOOP on re-run, covis-gate rejects a missing/FAIL covis.
- [ ] Proposed-N reconstructs ≥ a hand pick — **operator-gated** (needs a reconstruct run +
      an operator hand-pick baseline + a quality judgment, T-020). The selector + report + node
      are delivered; this is the falsifiable validation bar (T-001), tracked as the close gate.
      **Operator decision 2026-06-14: close 094 with the selector + node delivered; carry this
      reconstruct-≥-handpick comparison as a tracked validation follow-up** (capture-limited
      anyway — coverage is ~linear in N with no knee). Not a blocker for the spine work that
      consumes the node.

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

**Remaining (follow-ups):** ~~wire as a v4 store node (`select@0` under the covis, mirroring
093)~~ — **DONE 2026-06-14** (`tasks/select.json` + `cmd_select`; gated behind a PASSing
covis; emits `selection.json` + `posed.json`; verified on the 539-pool solve, idempotent,
gate-enforced). Only the "reconstructs ≥ hand-pick" comparison remains — operator-gated
(needs a hand-pick baseline + a reconstruct run + a quality call).

## Node (2026-06-14) — `select@0` wired into the v4 graph

`select` runs **locally** on the store's `sparse/0` (pure-stdlib; no container/host),
**gated behind a PASSing covis** so a nebula solve never reaches the selector
(STO-SCN-093 contract). Placement mirrors covis:
`images/subsets/{subset}/cameras/{up_solve}/select/{identity}`. Settings (tunable, hashed):
`n` / `min_overlap` / `div_angle`. Outputs: `selection.json` (the coverage report STO-SCN-095
renders) + `posed.json` (the proposed-N poses in the `name`/`w2c`/`K` shape the reconstruct
graphs already consume — the clean handoff). Run:

```
v4exec.py select <scene> --solve <id> --covis <id> [--subset <s>] [--n 24] \
                 [--min-overlap 10] [--div-angle 25]
```

Verified on 001-patio / `6EHLYO3MF3QU` / solve `62QEHJDAJZBI` / covis `L57FPDHY2DRG` →
`select/OBQTTTCF6RH7` (24 views, deterministic; NOOP on re-run; rejects missing/FAIL covis).

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

## Status notes

- 2026-06-14: Closed with --force; 1/4 DoD boxes unchecked. Reason: Selector + select@0 store node delivered and verified end-to-end on the 539-pool solve (idempotent, covis-gated). Operator decision 2026-06-14: close with the reconstruct-≥-handpick comparison carried as a tracked validation follow-up (capture-limited; not a blocker for the spine work that consumes the node).
