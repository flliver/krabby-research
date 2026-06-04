---
xid: STO-SCN-020
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-c5x
priority: 1
title: T3.F2 — Holosoma Dockerfile
---

# T3.F2 — Holosoma Dockerfile

## Summary

Per grant Task 3: package Holosoma quadruped locomotion model (proprioception-only, no vision) in a Docker container, parallel structure to F1.

## Context

Per grant Task 3: package Holosoma quadruped locomotion model (proprioception-only, no vision) in a Docker container, parallel structure to F1.

## Definition of Done

- [ ] Dockerfile builds and runs from a clean state
- [ ] Container launches IsaacSim and loads same USD envs as F1
- [ ] Proprio-only observation space populated
- [ ] Trajectory/metrics output in same schema as F1


## Journal Notes

No Holosoma-specific Dockerfile work in the journal. The applicable content is the shared PyTorch/CUDA/Docker infra lessons — see **STO-SCN-019 § Journal Notes** (cu128 wheel for sm_120, mandatory `--shm-size=8g`, long-run CUDA-loss restart, cross-arch wheel doc). Holosoma is proprioception-only (no vision), so the depth-observation pieces don't apply, but the wheel-selection + shm-size + fail-fast preamble all do.

---
_Imported from legacy beads `m11-x6i` (M11 DAG re-import, 2026-06-03)._
