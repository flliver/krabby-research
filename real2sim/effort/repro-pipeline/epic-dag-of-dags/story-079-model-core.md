---
xid: STO-SCN-079
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-11
depends-on: []
hugs: [HUG-SCN-005]
bd-id: krabby-biy
shipped: 2026-06-11
tasks: 5
complete: 5
---

# v4 model core: identity recipe, ref resolution, task/graph defs, materialize-check planner

## Summary

The v4 model core implementing HUG-SCN-005's locked decisions as
code: identity recipe, ref resolution, task/graph definitions, the
materialize-check planner, and the read-side scan.

## Shipped (2026-06-11)

- `real2sim/v4core.py` — identity recipe (#3: resolved inputs +
  tunable + frozen + algo@version; pins REFUSED from the hash), HOH
  (#5, order-insensitive), content/file hashing ([0-9A-Z]{12},
  base32-sha256), Scene refs (set-if-unset, never-move — #1/#7),
  canonical view resolution, job dirs (#8), per-identity metadata
  writer, topo planner with NOOP/EXECUTE marking (#4), scan_scene +
  leaderboard (scores join), license ancestry walk (#10).
- `real2sim/tasks/*.json` — 11 v4 task defs with settings classified
  tunable/frozen/pin; DA3 carries license_flag; orient-cameras
  carries the 082 measured verdict.
- `real2sim/graphs/*.json` — ingest-scene, reconstruct-matcha,
  reconstruct-da3.
- Exercised for real by: the full-store migration (080), the Studio
  + rate_renders v4 consumers, the 082 experiment, and the license
  walk (da3 mesh blocked, matcha tetra cleared — on migrated data).

## Definition of Done

- [x] Identity recipe deterministic; settings + algo@version re-key;
      HOH order-insensitive (smoke tests).
- [x] Refs: set-if-unset true once, false after; resolution returns
      target.
- [x] Pins refused from hashable settings; frozen defaults
      participate.
- [x] License ancestry: NC parent taints nested child; clean chain
      passes.
- [x] Planner walks graphs in topo order producing identity +
      NOOP/EXECUTE rows.

## Out of scope (follow-on)

- The v4 EXECUTOR (dispatch from plan rows; run_pipeline.py is the
  v3-era precedent) — first v4-native job will drive it (see 081
  notes: matcha@1 per-stage invocation).
