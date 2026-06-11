---
xid: MSG-PROJ-004
content-path: /private/var/krabby/research/messages/2026-06-09/21-16-34-handoff-2116/001/msg-handoff-2116.md
kind: msg
effort: proj
status: in-progress
date: 2026-06-09
to: principal
from: principal
topic: handoff-2116
bd-id: krabby-0ff
assignee: principal
---

# Handoff from previous principal session

## What Was Happening

Session focused on **STO-SCN-036** (reconstruct legacy-scene provenance from
M11 journals) — **shipped and closed** (2026-06-05). Final state of the
scene store: **15 measured / 3 deduced** legacy+curated transforms (the 3
deduced are all colmap: 001-patio multi-host, 002-patio empty, dtu-bicycle
upstream — genuinely host-unrecoverable). Evidence seams used: journals,
backfill_manifests.py, run scripts, on-disk mtimes (CoW-preserved), outposts
partial trees (host attribution), dpkg.log nvidia-driver timeline, OLAI
3d-reconstruction corpus. Session histories were a confirmed dead end.

Also earlier: rewrote `EPI-SCN-SCENE-SYNC` epic.md to the as-built
**two-layer sync design** (Layer 1: fleet LAN git+rsync for experiment
load-distribution, results centralize on j; Layer 2: S3 durable cold store +
distribution, j sole gateway). Committed as `ab32881`. Operator correction
preserved there: **S3 is necessary — do not remove it again.**

Last exchange: answered operator's question about the end-of-prototyping
**comparison UI parameterization provenance**. Answer (delivered): YES on two
levels — (1) per-variant param deltas (alignment_config / dense_regul /
frames / chart_resolution) are captured + migrated into curated runs'
run.json/specification.json as *measured*; (2) comparison-set provenance
lives in `experiments/m11-scene-reconstruction/DECISION-MATRIX.md`
(pipeline-level rubric/verdicts, committed) and the
`comparison_views.json` schema-v4 tooling (`camera_viewer/viewer.py`,
`render_comparison_matrix.sh`, `sync_comparison_views.py`) — but **no
actual comparison_views.json instance was located** (may be gone or inside
a .blend). The comparison-as-artifact relationship is NOT modeled in the
scene schema.

I offered to fold that finding into **STO-SCN-037**'s scope (eval/ schema:
comparison set = member runs + diff axis + rubric + viewpoints + verdict).
**Operator has not yet answered** — that question is open.

## What Needs to Happen Next

1. Re-ask / await operator decision: scope STO-SCN-037's `eval/` schema to
   capture the comparison-set relationship (runs + diff axis +
   DECISION-MATRIX rubric + comparison_views viewpoints + verdict)?
2. If yes → pick up STO-SCN-037 (eval/ home + re-sort scene-level
   `_unsorted/`).
3. Other open backlog (do not start without direction): STO-SCN-034
   (jsonschema CI), STO-SCN-027 (tiering), STO-SCN-016 (USD scale blocker),
   epic bookkeeping for sync stories 028–030.

## Key Context

- Scene store: `/var/krabby/scenes` (SYMLINK → `/Volumes/Archives-01/krabby/scenes`;
  `find` needs trailing slash). Git-LFS repo; latest pushed hash `e2e6e77`.
- Hardened paths: always `GIT_LFS_SKIP_SMUDGE=1` + `git push --no-verify`
  on the scenes repo; rsync of `.git/lfs/objects` additive-only (never
  `--delete`). j (jbeeprz) is the hub + sole S3 gateway (`krabby` AWS profile).
- Provenance gate (T-002): write a fact only with ≥2 corroborating sources,
  else leave unknown/null. `provenance: measured` requires `host`.
- Extractor: `real2sim/scenes/reconstruct_provenance.py` (FACTS dict,
  DRIVER_TIMELINE, enrich_curated(); emits `provenance-ledger.md`). Re-runnable.
- Fleet division of labor (two independent evidence trails agree):
  mast3r→sbeeprz, vggt/slam3r→dbeeprz, matcha→tbeeprz (RTX-5080 SIFT
  non-determinism is why colmap ran on 4080s).
- Branch: `jdp/m11-real2sim`. Commit/push only when operator asks.

## Active Files

- `real2sim/scenes/reconstruct_provenance.py` — extractor (final, shipped)
- `real2sim/scenes/provenance-ledger.md` — auto-generated audit ledger
- `real2sim/effort/out-of-scope/epic-scene-sync/epic.md` — as-built two-layer design
- `real2sim/effort/out-of-scope/epic-scene-sync/story-036-…` — shipped
- `/var/krabby/scenes/**/run-legacy/transform-01-legacy/{results,specification}.json` — 24 files updated, pushed

## Beads XIDs

- `STO-SCN-036` — closed/shipped 2026-06-05; 10/10 tasks; the session's main deliverable.
- `STO-SCN-037` — open; eval/ schema home; **candidate next pickup**, pending operator answer on comparison-set scoping.
- `EPI-SCN-SCENE-SYNC` — in_progress (rollup parent); epic.md now reflects as-built design.
- `DES-SCN-REPRO`, `DES-SCN-TX` — in_progress rollup parents; no direct action.

## Status notes

- 2026-06-09: Filed.
- 2026-06-09: Picked up by principal, beginning review
