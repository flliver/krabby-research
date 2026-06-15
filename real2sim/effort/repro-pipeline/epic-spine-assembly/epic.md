---
xid: EPI-SCN-SPINE-ASSEMBLY
parent: ../design.md
kind: epic
effort: scn
status: shipped
date: 2026-06-13
hugs: []
tenets: []
bd-id: krabby-3l9
assignee: krabby
shipped: 2026-06-14
---

# Spine assembly: segment a video into M overlapping sub-reconstructions, globally register, fuse into one cohesive space

> **Status: longer-term / lightweight scaffold.** Sibling to
> EPI-SCN-AUTO-SUBSET-SELECT. Captured now so the architecture is on record; not yet
> scheduled. See STO-SCN-096 conclusion #7 for the reasoning.

## Problem Statement

A single video is too big to pose or reconstruct at once, and the ultimate goal is **one
cohesive reconstructed space** — typically many segments along the video's trajectory
("spine"). EPI-SCN-AUTO-SUBSET-SELECT solves the *per-segment* unit (best-N for one
tractable chunk). This epic is the **composing layer**: chunk the spine into M overlapping
segments, register their submaps into one gauge, and fuse them into a single drift-free
space. Without it, M locally-good segments are M disjoint reconstructions.

## Goals

- Segment a long trajectory into M overlapping, individually-tractable windows.
- Register the per-segment submaps into **one global gauge** (drift-corrected).
- Fuse per-segment reconstructions into a single cohesive mesh/gaussian space.
- Verify the assembled whole — seams included — in the scout gaussian.

## Non-Goals (Out of Scope)

- The per-segment best-N unit (that's EPI-SCN-AUTO-SUBSET-SELECT).
- The reconstruct graphs themselves.

## Context

**Source:** Operator design session 2026-06-13 (the spine reframe).
**Relationship:** Wraps EPI-SCN-AUTO-SUBSET-SELECT — each segment runs that pipeline; this
epic adds the outer loop (segmentation), the seams (registration), and the whole (fusion +
verification). This is the DAG-of-dags composing level: per-segment sub-graphs + a spine
graph with registration/loop-closure edges.

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-097` | Spine segmentation (chunk trajectory into M overlapping segments) | shipped | M |
| 2 | `STO-SCN-098` | Global registration of segment submaps (pose-graph + loop closure + global BA) | shipped | L |
| 3 | `STO-SCN-099` | Cohesive fusion of per-segment reconstructions into one gauge | shipped | L |
| 4 | `STO-SCN-100` | Whole-spine verification (assembled space + seams in the scout gaussian) | shipped | M |
| 5 | `STO-SCN-105` | Scout-gauge registration: DA3 normalized-frame root cause + auto camera-pose Umeyama fix (prerequisite for 098/099/100) | shipped | L |

## Design

### Approach

Submap / pose-graph (SLAM-shaped): segment the spine with **overlapping windows** (shared
frames at boundaries guarantee registrability); run the per-segment pipeline; register
submaps via pose-graph optimization + loop closure + global BA; fuse into one gauge; QA the
assembled whole. Sequential/SLAM-with-submaps is the natural solver substrate; FastMap /
feed-forward are per-segment workers under the global pose graph.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Drift accumulates along the spine | High | High | Global pose-graph + loop closure; overlapping segments |
| Seams don't register (insufficient boundary overlap) | Medium | High | Boundary-overlap budget enforced in selection (STO-SCN-094) |
| Fusion produces double-walls / gaps at seams | Medium | Medium | Register before fuse; verify seams in scout (STO-SCN-100) |

## Success Criteria

- [ ] A long video → M segments → one cohesive, drift-free reconstructed space.
- [ ] Seams verifiable (no double-surfaces / gaps) in the scout gaussian.
- [ ] Reuses EPI-SCN-AUTO-SUBSET-SELECT per segment unchanged.
