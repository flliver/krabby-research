---
xid: STO-SCN-058
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-m6t
shipped: 2026-06-10
tasks: 4
complete: 4
---

# Renders live in the pipeline run that produced them (settings sidecar + backfill)

## Summary

Comparison renders move from the scene-level
`comparison_renders/<view>/<variant>.png` into the producing run:
`pipeline-<p>/run-<r>/renders/<view>.png` + a `<view>.json` settings
sidecar. Per-view comparison becomes a read-time aggregation in
rate_renders. All 43 existing renders backfilled.

## Context

Operator (2026-06-10), examining the store: "What we are comparing is
the *pipeline configuration* that produced the image… By storing in
comparison_renders, the information about what is being compared is
obscured/lost. Store the rendering *in* the pipeline run and show the
settings that produced it."

## What changed

| File | Change |
|------|--------|
| `real2sim/render_comparison_matrix.sh` | output → `run-<r>/renders/<view>.png`; writes settings sidecar (engine, resolution, mesh source/relpath, view camera, run transform parameters, provenance: measured) |
| `real2sim/rate_renders/server.py` | discovery scans `pipeline-*/run-*/renders/*.png`, aggregates by view at read time; `/api/render/<scene>/<view>/<variant>.png` URL contract unchanged (no frontend change) |
| `real2sim/migrate_renders_into_runs.py` | NEW — backfill: moves PNGs into runs, mints sidecars with `provenance: backfilled` (mesh_source honestly null — not recorded at render time, T-002), removes emptied comparison_renders/ |

## Backfill record

- 2026-06-10: 43 renders across 15 scenes + dtu-bicycle, dry-run then
  live: moved=43 errors=0; every variant label mapped to an existing
  run dir (verified before any code was written). rankings.jsonl
  untouched (view/variant labels unchanged). rate server restarted —
  gotcha: a stale server process held :8090 and silently served the
  OLD code; kill by listener PID (`lsof -t -iTCP:8090`), not pgrep
  pattern.

## Definition of Done

- [x] New renders land in the producing run with a measured sidecar.
- [x] rate_renders aggregates the new layout; URL contract unchanged;
      verified 200s on old-runoff + 013 renders, all scene payloads
      list the same view/variant sets as before the move.
- [x] All 43 legacy renders backfilled with sidecars; comparison_renders/
      dirs removed store-wide.
- [x] RECIPES.md layout + phase catalog (11, 12) updated.
