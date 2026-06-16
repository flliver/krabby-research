---
xid: STO-SCN-150
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-16
depends-on: [STO-SCN-149]
bd-id: krabby-4omj
assignee: scout
---

# New Scene — run ingest-scene pipeline (host pick, spine, PRIMARY, DA3 gs+mesh) with phase progress

## Summary

Drive a canonicalized scene through the existing **ingest-scene pipeline** from the UI: pick a
processing host, run **Spine Solver** → generate **PRIMARY** subset → **DA3 gaussian** (default) →
**DA3 mesh** (default), with **phase progress** (server-side monitor, simple client display).

## Context

Creation step **9**. The pipeline exists (`v4exec` spine/scout/meshify, EPI-SCN-SPINE-ASSEMBLY);
job records + `/api/jobs/` + `/api/materialize/` already expose status. This story orchestrates the
phases and surfaces progress. Full spec: **EPI-SCN-SCENE-MANAGER § Creation flow**.

## Design / scope
- **Host picker**: choose a processing host (GPU hosts e.g. tbeeprz) for the GPU phases.
- **Phase orchestration**: Spine Solver → PRIMARY subset → DA3 gaussian (default config) → DA3 mesh
  (default config), each a `v4exec` invocation on the chosen host; chain on success.
- **Progress**: server-side monitor of each phase (job records / MQTT progress already emitted by
  `cmd_scout` etc.), simple client poll/stream showing phase + percent.
- Failure surfaces the phase + log tail; resumable per phase (re-run is NOOP where content exists).

## Definition of Done
- [ ] Operator picks a host and launches the pipeline from the tab; phases run in order.
- [ ] PRIMARY subset + DA3 gaussian + DA3 mesh materialize for the scene.
- [ ] Live phase progress (phase + percent) from the server-side monitor; failures show the phase + log.
- [ ] Reuses `v4exec` + existing job/progress plumbing (no new reconstruction logic).

## Out of scope
- Scout viewing / render views (STO-SCN-151) and MEASURE (152) — those follow once the gaussian exists.
- Non-default gaussian/mesh configs (defaults only here; tuning is the cull/condition epics).
