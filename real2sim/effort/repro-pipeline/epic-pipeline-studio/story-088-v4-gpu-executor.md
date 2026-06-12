---
xid: STO-SCN-088
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
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
4. **Job feedback channel** (operator question, 2026-06-11): current
   feedback is (a) file-level truth — polling payload refresh flips
   tiles as outputs land; (b) job.json written only at END; (c)
   stdout to a /tmp log; (d) host-side nanny-progress on the beeprz
   dash. Gaps: no incremental per-node status, no error surfacing to
   tiles. This story makes job records INCREMENTAL (append per-node
   outcome as it completes), adds GET /api/jobs/<scene> for the UI,
   and surfaces per-tile running/failed states from it. Fleet-side
   long tasks keep nanny-progress (fleet-ops rule).

## Definition of Done

- [ ] Click a 🧬 gap → host prompt → dispatch → tile shows live
      per-node progress → flips to artifact (render tier follows
      automatically via NOOP walk).
- [ ] Job records incremental; failures surface on the tile with the
      error tail.
- [ ] NC gaps require an extra confirm ("evaluation only") before
      dispatch.
- [ ] T-020: operator exercises end-to-end on a host of their choice.
