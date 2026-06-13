---
xid: STO-SCN-097
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-13
depends-on: [STO-SCN-091, STO-SCN-092]
bd-id: krabby-duy
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

## Out of scope

- Registration / fusion (later stories).
