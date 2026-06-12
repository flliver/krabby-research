---
xid: STO-SCN-086
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: []
bd-id: krabby-793
---

# Click to Materialize: missing-render tiles trigger the job from the UI

## Summary

The missing-render placeholder tiles (STO-SCN-085) become actionable:
click → the UI triggers the materialize job for that gap → the tile
shows progress → flips to the real render when the identity exists.
This is the deliberate step where the rank UI becomes a job UI.

## Context

STO-SCN-085 surfaced the gaps read-only and explicitly deferred the
trigger ("the moment the rank UI quietly becomes a job UI — worth
deciding deliberately"). Operator decided: do it. The job machinery
exists (`v4job.py render-missing`, NOOP semantics — locked #4); this
story is the UI→job bridge.

## Design

- New endpoint: `POST /api/materialize/<scene>` with optional body
  `{mesh: <identity>, slot: "NN"}` (omit for "all missing in scene").
  Spawns `v4job render-missing <scene>` as a subprocess; returns a
  job handle (the scene job-record dir name, locked #8).
- Concurrency guard: one materialize job per scene at a time (a lock
  file beside the job record; second click returns the running
  handle instead of double-rendering).
- Tile states: missing → queued/running (spinner + job id) → done
  (reload image) or failed (tile shows the error log tail).
- Progress: poll `GET /api/materialize/<scene>/<job>` reading the
  job.json outcome (cheap; no websockets for MVP).
- Renders run LOCAL Blender (today's executor slice). GPU-task
  materialization (represent etc.) stays out of scope until the
  fleet executor exists — the endpoint must refuse non-render gaps
  honestly rather than pretend.

## Definition of Done

- [ ] Click a missing tile → render materializes → tile becomes the
      image without a manual job invocation.
- [ ] Double-click / concurrent clicks do not double-render (NOOP +
      lock verified).
- [ ] Failure surfaces on the tile (error tail), not silently.
- [ ] Job records written per invocation (locked #8).
- [ ] T-020: operator exercises the flow before close.

## Status Notes

- 2026-06-11: Minted at operator direction, immediately after
  STO-SCN-085 drew the scope line this story crosses deliberately.
