---
xid: STO-SCN-088
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-11
depends-on: []
bd-id: krabby-eih
---

# v4 GPU executor: materialize task gaps (represent/meshify) on an operator-chosen host; live job feedback channel

## Summary

The v4 GPU executor: make STO-SCN-087's 🧬 gap tiles clickable —
materialize represent/meshify task gaps on an **operator-chosen
host** (HUG-SCN-005 decision 3: the host is a parameter, never a
scheduler choice) — plus the live job-feedback channel the UI needs.

## Context

- The v3-era trigger (`run_pipeline.py`) proved the dispatch shape
  (SSH + docker, baked tools, digests measured from the host, LFS
  guard, expected-outputs gate) but writes the v2 store layout.
- STO-SCN-081: matcha@1 = per-stage invocation (--sfm_only /
  --alignment_only / --refinement_only / --mesh_only) — no fork.
- Fleet is on v4 (all four hosts re-cloned 2026-06-11).

## Scope

1. Port the dispatch to v4: inputs materialized from subsets (image
   hashes -> temp input dir), outputs into identity dirs, metadata
   per identity, job records per invocation.
2. Host prompt in the UI (t/b/d/s) — POST /api/materialize gains
   {task_gap, host}; refuses without host.
3. Task coverage: represent-via-da3 + meshify-via-tsdf fuse first
   (validated da3 path); matcha@1 stages next.
4. **Job feedback channel — MQTT-first** (operator direction,
   2026-06-11): the fleet ALREADY publishes progress over MQTT —
   nanny-progress wraps mosquitto_pub (host-scoped keys, beeprz
   dash), and `real2sim/lib_progress.sh` is the pluggable progress
   API (backends nanny|null, documented extension recipe). Design:
   - jobs publish progress + heartbeat to
     `krabby/jobs/<scene>/<job_id>` as **retained** messages
     ({node, status, pct, host, ts}) — retained = level-triggered
     (T-021): a late-joining UI reads current state instantly, and a
     crashed publisher leaves a stale-detectable heartbeat instead
     of a silent gap.
   - new `mqtt` backend in lib_progress.sh (per its own recipe) so
     fleet jobs feed BOTH the beeprz dash and the Studio UI from one
     emit; v4job (python) publishes via the same topic shape.
   - UI bridge: rate_renders server subscribes (or reads retained
     state on request) → `GET /api/jobs/<scene>` → per-tile
     running/failed states. Browser-direct websockets only if the
     broker already listens on ws (check, don't assume — T-002).
   - **File-level truth stays the fallback**: payload-refresh tile
     flips can't lie; MQTT is the fast path, not the source of
     truth. Incremental job.json append per node completion remains
     (locked #8 record).
   - Open verification: Mac publish credentials (~/.mqtt is
     host-scoped on fleet hosts; does the Mac hold a key?) + broker
     ws listener.

## Definition of Done

- [ ] Click a 🧬 gap → host prompt → dispatch → tile shows live
      per-node progress → flips to artifact (render tier follows
      automatically via NOOP walk).
- [ ] Job records incremental; failures surface on the tile with the
      error tail.
- [ ] NC gaps require an extra confirm ("evaluation only") before
      dispatch.
- [ ] T-020: operator exercises end-to-end on a host of their choice.
