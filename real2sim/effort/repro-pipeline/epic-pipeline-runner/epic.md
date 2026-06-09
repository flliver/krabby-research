---
xid: EPI-SCN-PIPELINE-RUNNER
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-09
hugs: []
tenets: []
bd-id: krabby-zdc
priority: 1
assignee: devex
---

# Config-Driven Pipeline Runner with Pluggable Transforms

## Goal
Replace hand-driven reconstruction scripts with a configuration-driven runner: read a `specification.json`, execute the declared transform in its pinned container, emit a `measured` `results.json` + artifacts into the scene store — with every line of tool code source-controlled.

## Stories
- STO-SCN-038 — source-pin all tool code (kills the un-versioned `~/scratch/MAtCha` class of risk)
- STO-SCN-039 — pipeline runner v1 (spec → run → results)
- STO-SCN-040 — pluggable transform interface + registry
- STO-SCN-041 — reproduction verification harness (rerun + compare)

## Relations
- Builds on scene schema (STO-SCN-026/033/036) and consume convention (STO-SCN-031); validated by schema CI (STO-SCN-034).
- Governed by **HUG-KRB-002** (full reproducibility / no prototypes).
