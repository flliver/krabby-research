---
xid: STO-SCN-150
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
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
- [x] Operator picks a host and launches the pipeline from the tab; phases run in order.
- [~] PRIMARY subset + DA3 gaussian + DA3 mesh materialize for the scene. — orchestration + dry-run verified; **a REAL run needs a GPU host** (operator T-020 below).
- [x] Live phase progress (phase + log tail) from the server-side monitor; failures show the phase + rc + log.
- [x] Reuses `v4exec` (exact RECIPES commands) — no new reconstruction logic.
- [ ] **Operator-verified (T-020):** Scenes → Pipeline → pick `tbeeprz` → **Preview plan** (eyeball the 5 commands), then **Run** a real scene to a scouted+meshed state; confirm phases complete + the mesh/gaussian materialize.

## Build notes (2026-06-16)
- **Scope (corrected):** the default ingest-scene pipeline is **precull(--set-primary)
  → solve → covis → scout(DA3 gaussian) → reconstruct-da3(DA3 mesh)**. `select`
  (best-N view selection) is **not** here — that's the view-selection step
  (EPI-SCN-AUTO-SUBSET-SELECT). This means only **one id threads** (the solve),
  resolved from the store, not parsed from stdout.
- **Orchestrator** `pipeline_run.py` (stdlib, numpy-free): `gpu_hosts()`
  (default `tbeeprz`, env `KRABBY_GPU_HOSTS`), `resolve_primary_subset` /
  `resolve_latest_solve` (newest `cameras/*` under primary — grounded against
  the real 001-patio layout), `PHASES` + `build_command` (exact RECIPES.md
  lines), `plan()` (dry-run preview), `run_pipeline()` (sequential subprocess
  runner, stops on first failure, writes `pipeline_status.json` with phase +
  log tail; phases idempotent/NOOP where content exists → safe re-run).
- **Endpoints** (`rate_renders/server.py`): `GET /api/hosts`;
  `GET /api/scene/<s>/pipeline-plan?host=` (preview); `POST /api/scene/<s>/pipeline`
  {host, dry_run} (threaded; one-run-per-scene guard); `GET …/pipeline-status`.
- **Frontend** `static/scenes-pipeline.js` (`window.scenesViews.pipeline`, a
  4th view-switcher tab): host dropdown, **Preview plan** (dry-run) + **Run**
  (with confirm), live phase rows (pending/running/done/error + rc) + log tail.
- **Verified (up to the GPU boundary):** `tests/test_pipeline_run.py` + HTTP
  e2e on a synthetic store — `/api/hosts`, plan resolves + threads the solve id,
  POST dry_run sequences all 5 phases to `planned`, status polls. **The real
  GPU execution (ssh+docker on the host) is intentionally NOT run here** — that
  is the operator T-020 above; dry-run is the pre-flight.

## Out of scope
- Scout viewing / render views (STO-SCN-151) and MEASURE (152) — those follow once the gaussian exists.
- Non-default gaussian/mesh configs (defaults only here; tuning is the cull/condition epics).

## Follow-up (2026-06-16): ingest phase 0 — video import in the Pipeline
Gap found in operator UI testing: "Run Pipeline" started at `precull` and assumed
a canonical image pool existed — so a video-only scene (e.g. a nuked/freshly-
imported one) had no UI path to extract frames. Filled by adding a LOCAL
**ingest** phase 0 to `pipeline_run.py`:
- If the scene has `videos/capture/video.*` and an EMPTY canonical pool, it
  extracts frames + canonicalizes (reusing `scene_ingest`), else **skips**
  (idempotent; photo scenes + already-ingested scenes skip).
- **fps is DEDUCED**: `deduce_fps(duration)` targets ~500 frames clamped to a
  1–4 fps handheld-overlap band (prevents the 003-firepit 12-frame degenerate
  failure). Preview Plan shows it: `ffmpeg -i video.mp4 -vf fps=1.51 → ~500 frames`.
- UI: the Pipeline tab renders the ingest phase + its note + a `skipped` badge.
- Tests: `deduce_fps` band, `ingest_plan` extract-vs-skip, phase-0 ordering.

## Follow-up (2026-06-16): mode-aware frame downscale in ingest
Operator UI testing: 4K frames (1.6 MB each, ~800 MB for 500) made extraction +
host-staging slow for no benefit — DA3 scout ingests at 504px, SfM is fine at
~1600px. `extract_frames` now takes `max_long_edge` (ffmpeg `scale` downscale,
aspect-preserved, never upscales). `pipeline_run.resize_target` deduces it:
**≤1920px UNLESS the scene is declared fisheye** (the fisheye undistort is
pinned to its native-res calibration, so it needs full res; an undeclared
fisheye can't pass the solve's capture-decl gate anyway). Preview Plan shows the
target (`(≤1920px)` / `(native (fisheye))`). ~4× less data for rectilinear
(iPhone) scenes.
