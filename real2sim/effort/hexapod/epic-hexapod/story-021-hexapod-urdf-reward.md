---
xid: STO-SCN-021
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-6pi
priority: 1
title: T4.F3 — Hexapod URDF + Reward Shaping
---

# T4.F3 — Hexapod URDF + Reward Shaping

## Summary

Per grant Task 4: convert both EP and Holosoma from quadruped to hexapod embodiment.

## Context

Per grant Task 4: convert both EP and Holosoma from quadruped to hexapod embodiment.

Work spans:
- Updated URDF / embodiment configs pointing at Krabby hexapod asset
- Action / observation space updates for higher DOF
- Reward shaping: penalize simultaneous-leg motion; encourage tripod-style alternation
- Documentation of penalty terms and tripod-bias incentives

## Definition of Done

- [ ] URDF + configs updated in both stacks
- [ ] Reward shaping documented (with rationale)
- [ ] Action/observation spaces match hexapod DOF

---
_Imported from legacy beads `m11-1ij` (M11 DAG re-import, 2026-06-03)._
