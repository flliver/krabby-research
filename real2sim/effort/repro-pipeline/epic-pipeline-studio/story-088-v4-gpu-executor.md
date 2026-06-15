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

- [x] **Job-feedback channel built + verified** (the host-independent
      half — 2026-06-14): `mqtt` backend in `lib_progress.sh` publishes
      RETAINED progress to `krabby/jobs/<scene>/<job_id>`
      ({node,status,pct,host,ts}; T-021 level-triggered — verified
      publish + retained re-read against a live broker); `v4job`
      publishes via the same topic shape AND writes **incremental**
      job.json per node (source of truth); `GET /api/jobs/<scene>`
      serves file-truth records with an optional retained-MQTT overlay
      that degrades to {} with no broker (both halves unit/broker
      tested, `tests/test_jobs_endpoint.py`).
- [ ] Click a 🧬 gap → host prompt → dispatch → tile shows live
      per-node progress → flips to artifact (render tier follows
      automatically via NOOP walk). **(GPU SSH dispatch + UI host
      prompt — remaining engineering; needs the operator-chosen host,
      decision 3.)**
- [ ] NC gaps require an extra confirm ("evaluation only") before
      dispatch.
- [ ] **OPERATOR (T-020):** exercise end-to-end on a host of your
      choice (the dispatch leg + the live tile).

## Status Notes

- 2026-06-14: Built and verified the **feedback channel** (DoD #1) —
  the part that needs no live GPU host and no operator host-choice:
  `lib_progress.sh` `mqtt` backend, `v4job.publish_progress` +
  incremental job records, and the `/api/jobs/<scene>` endpoint
  (file-truth + MQTT overlay). All paths exercised (broker happy-path
  AND graceful no-broker degradation). **Deliberately NOT done
  autonomously:** the GPU SSH dispatch to a fleet host — HUG-SCN-005
  decision 3 makes the host an operator parameter, so dispatch waits
  on the operator naming a host (the v3 `run_pipeline.py` dispatch
  shape is the port source). The remaining work is the UI host prompt
  + the v4 dispatch leg + the T-020 end-to-end exercise.
