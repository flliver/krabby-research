---
xid: STO-SCN-129
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-15
depends-on: []
bd-id: krabby-iyqm
assignee: krabby
tasks: 4
complete: 4
---

# solve: emit cameras.json from sparse/0 so any FastMap-solve rep is renderable

## Summary

The `solve` step writes a `cameras.json` (`filepaths` + `cams2world` + `focals`) alongside its
`sparse/0`, so **any** representation anchored on a FastMap solve resolves cameras in
`v4job.rep_camera_paths` and renders — instead of silently rendering 0.

## Context / Problem

FastMap solves emit only `sparse/0` (COLMAP bins). The renderer (`rep_camera_paths` →
`build_blender_scene`) needs a `cameras.json` at the solve dir (or migrated `origin-data/`).
Migrated (mast3r) reps have it; FastMap-native ones don't, so a fresh variant **silently fails
to render** (counts 0, no error). Discovered building DA3-24 (STO-SCN-127): `reconstruct-da3-scout`
had to emit the solve `cameras.json` itself as a side effect. That emission belongs in the
**solve step**, once, so every downstream rep (matcha + DA3) benefits — not bolted onto each
reconstruct path.

## Design

- In `cmd_solve` (or the solve graph), after `sparse/0` lands, write
  `<solve>/cameras.json = {filepaths, cams2world, focals}` derived from `posed_from_sparse`
  (the exact logic now duplicated in `reconstruct-da3-scout`). Idempotent; content of the solve.
- Extract the emission into a shared helper (`posed_sparse_to_cameras_json`) and call it from
  both `cmd_solve` and `reconstruct-da3-scout` (T-023 — kill the duplication).

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | `cmd_solve`: emit `cameras.json`; extract shared helper; `reconstruct-da3-scout` calls it |

## Definition of Done

- [x] Every new FastMap `solve` writes `<solve>/cameras.json` with `filepaths`+`cams2world`+`focals`.
- [x] `reconstruct-da3-scout` uses the shared helper (no duplicated emission).
- [x] A rep on a FastMap solve renders without any reconstruct-path side effect.
- [x] Backfill note: existing solves (e.g. 001-patio `62QEHJDAJZBI`, already emitted) unaffected.

## Out of scope

- The matcha-on-selection wiring (STO-SCN-130).
- Re-solving anything.

## Implementation Notes

_(Earned 2026-06-15, STO-SCN-127. Render-camera contract documented in
`scene-processing/T3c-reconstruction-postprocessing.md`.)_

**Shipped 2026-06-15 — solve-side emission wired.** `cmd_solve` now calls the shared helper
`posed_sparse_to_cameras_json(sparse/0 → <solve>/cameras.json)` on **two** paths:
- **success path** — every new FastMap solve emits `cameras.json` (512-conv focals) right after
  `sparse/0` lands, before `write_metadata`. Consumers (render, da3-scout, matcha@1 posed) no
  longer backfill lazily.
- **NOOP path (self-heal)** — re-running `solve` on an older solve that predates this change
  backfills `cameras.json` if missing (idempotent; only writes when absent — does **not** touch
  the solve's identity/metadata, so existing solves are unaffected per DoD item 4).

`reconstruct-da3-scout` already calls the same helper (no duplicated emission — T-023).

**Verified end-to-end (real CLI):** `v4exec solve 001-patio --subset I2UQBLXYJE5A` recomputed the
same `sid` (`435MR3O7PMSP`), hit NOOP, and backfilled `cameras.json` — 227 cams, all three keys,
4×4 `cams2world` (`[0,0,0,1]` bottom row), focal 151.6 (512-conv, matches 001-patio). The success
path is the same helper call (unconditional), proven by the same code + da3-scout production use.
