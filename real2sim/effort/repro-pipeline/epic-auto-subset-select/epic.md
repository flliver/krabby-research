---
xid: EPI-SCN-AUTO-SUBSET-SELECT
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-13
hugs: []
tenets: []
bd-id: krabby-0qk
assignee: krabby
---

# Automated frame-subset selection: large pool → posed → best-N (human-verified in gaussian space) → reconstruct

## Problem Statement

Given a large image pool (a video / hyperlapse of hundreds–thousands of frames), the
reconstruct graphs need a *small, well-chosen* subset of views. Today that choice is
ad-hoc: naive temporal subsampling picks blurry, redundant, or drift-prone frames (the
001-patio "gaseous nebula" — a 300-frame MASt3R solve scattered cameras across many
planes), and manual `viser` picking doesn't scale and isn't automated. We need a
**fully-automated ingress** that turns any pool into a *proposed best-N* subset, lets a
human verify it inside a gaussian space, and hands a clean subset to the existing
reconstruct graphs. Small pools need none of this — they pass straight through.

## Goals

- Given `video | images`, decide and execute the camera-solve approach automatically.
- From a posed pool, **auto-propose the best-N views** for high-quality reconstruction,
  by coverage + connectivity on the co-visibility graph (not by hand, not by guessing).
- **Human verifies in gaussian space** — see the proposed cameras in a scout splat;
  accept / drop / add. The only manual touch.
- Clean, unchanged handoff (`FINAL N` = frame-index list + poses) to the existing
  reconstruct graphs.

## Non-Goals (Out of Scope)

- The downstream **reconstruct** graphs themselves (matcha/da3 → mesh/gaussian) — they
  exist and consume `FINAL N` unchanged.
- Strong-fisheye undistortion research (note it; don't solve it here).
- Mesh/gaussian *quality* tuning of the final reconstruction (separate effort).

## Context

**Source:** Operator design session 2026-06-13 (reframed from "render a splat to pick
cameras" to "given a massive pool, what are the best N for a high-quality mesh").

**Dependencies:**

- A scalable, camera-model-correct pose solver that emits a **co-visibility / track
  graph** (free byproduct of SfM).
- Scout-gaussian generation (DA3 feed-forward — proven feasible, 32 views / 12.7 GB / ~32 s).
- The existing reconstruct graphs as the downstream consumer.

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-096` | Design: automated frame-subset selection — approach & conclusions | open | M |
| 2 | `STO-SCN-091` | Camera profile at ingest (EXIF/capture-mode → camera model) | open | S |
| 3 | `STO-SCN-092` | Pose-free pre-cull (sharpness + perceptual-dedup) for large pools | open | M |
| 4 | `STO-SCN-093` | Pose the pre-culled pool → poses + co-visibility graph | open | L |
| 5 | `STO-SCN-094` | Coverage-greedy best-N selector over the co-visibility graph | open | L |
| 6 | `STO-SCN-095` | Scout-gaussian verification surface + handoff to reconstruct | open | M |

## Design

### Approach

Fully-automated left column; one human gate; clean handoff:

```
video|images → EXTRACT (EXIF→camera model) → PRE-CULL (sharp+dedup, pose-free)
   → POSE pool (camera-model-correct solver) → co-visibility graph
   → AUTO-SELECT N (greedy coverage+connectivity on the graph)
   → SCOUT GAUSSIAN → ★ HUMAN VERIFY (accept/drop/add in splat)
   → FINAL N ═══► downstream RECONSTRUCT graphs
```

Two decisions, both from reliable sources: **EXIF / known capture mode → camera model**
(not pixel inference), and **co-visibility graph → which N** (not hand-picking). The
gaussian is the *verification surface*, not the selector.

### Architecture

New ingress stages feeding the existing reconstruct graphs. Selection is a deterministic
greedy over the SfM track graph; the scout gaussian (DA3) is the human QA lens.

### Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Manual viser picking from a solve | full human control | doesn't scale, not automated | Rejected as the mechanism (kept as QA aid) |
| Naive temporal subsample → solve | trivial | picks blur/redundant frames; drift (the nebula) | Rejected |
| Per-scene lens inference from edges | no metadata needed | noisy/unstable on natural scenes | Rejected — use EXIF/capture profile |
| Pre-cull → pose → coverage-greedy select → splat verify | automated, principled, human-checkable | needs a scalable solver | **Selected** |

## Decisions

| XID | Decision | Status | Rationale |
|-----|----------|--------|-----------|
| `STO-SCN-096` | Selection driven by co-visibility graph; EXIF→camera model; splat=QA | Adopted | See the design story for full reasoning |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Posing a large pool is expensive/unstable | High | High | Pre-cull first; pick a scalable solver (STO-SCN-093) |
| Strong fisheye + feed-forward solver mismatch | Medium | High | Camera model from profile; undistort or use fisheye-correct solver |
| Coverage metric mis-tuned (gaps or redundancy) | Medium | Medium | Human verify in splat catches it; tune on known scenes |

## Success Criteria

- [ ] Given a video / large pool, the pipeline produces a human-verifiable best-N proposal.
- [ ] The verified `FINAL N` reconstructs to a quality mesh via the unchanged reconstruct graphs.
- [ ] Small pools bypass selection (pass straight through).
- [ ] All stories shipped.

## Milestones

| Milestone | Target Date | Actual | Status |
|-----------|-------------|--------|--------|
| Stories defined | 2026-06-13 | 2026-06-13 | done |
| Implementation complete | | | open |
| End-to-end on a video pool | | | open |

## Retrospective

_(Fill in after epic completion.)_
