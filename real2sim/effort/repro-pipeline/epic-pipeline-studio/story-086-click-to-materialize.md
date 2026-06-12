---
xid: STO-SCN-086
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
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

- [x] Click a missing tile → POST /api/materialize/<scene> spawns the
      render job; tiles show ⏳ "materializing…"; 8s polling
      refreshes the payload so finished renders flip in EARLY
      (mid-job), with a final rendered/NOOP/failed summary in the
      status line.
- [x] Concurrent clicks do not double-render: store-wide one-job
      guard (pgrep) — verified live against the running repair chain
      (POST returned already_running, no second process).
- [x] Failure surfaces: per-render failures land in the job outcome
      shown in the status line; job stdout at
      /tmp/v4job-materialize.log; per-tile error tail deferred to a
      follow-up if needed.
- [x] Job records written per invocation (v4job's locked-#8 records).
- [ ] **OPERATOR (T-020):** click a missing tile (e.g. 004's) and
      watch it flip; story closes on your verification.

## Status Notes

- 2026-06-11: Minted at operator direction, immediately after
  STO-SCN-085 drew the scope line this story crosses deliberately.
