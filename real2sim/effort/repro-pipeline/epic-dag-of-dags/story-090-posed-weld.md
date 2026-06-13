---
xid: STO-SCN-090
kind: story
effort: repro-pipeline
epic: EPI-SCN-DAG-OF-DAGS
status: shipped
created: 2026-06-12
closed: 2026-06-13
bd-id: krabby-84f
---

# STO-SCN-090 — Posed reconstruction (matcha@1 + da3@1): feed the ingest solve in, stop re-solving

## Why

Follow-up #1 from the STO-SCN-089 sweep notes. The matcha@0 weld's
`train.py` re-solves cameras internally (`--sfm_config unposed`),
minting its own arbitrary gauge per run — measured 90.7–177.1° across
the 8-scene sweep. The gauge-sim composition (089) absorbs that when
the two solves agree up to a similarity, but **003-firepit's re-solve
genuinely disagrees with the ingest solve beyond any similarity**
(3.1% and 3.9% camera residual on two independent welds vs the 2%
gate) — the scene is unbuildable under matcha@0.

The architectural fix: stop letting the weld re-solve. MAtCha natively
supports `--sfm_config posed` (COLMAP-format calibration; fixes
focal/pp/rotation/translation and aligns camera locations to the
calibrated ones), so the weld's gauge IS the ingest gauge by
construction. Kills the gauge-sim class at the root for every scene.

## Mechanism

- `colmap_posed.py` (new): mints `sparse/0/{cameras.bin,images.bin,points3D.bin}`
  from a store ingest solve.
  - Intrinsics: PINHOLE in ORIGINAL pixel space (the container loader
    re-centers pp and rescales itself). Store focals are in mast3r-512
    space → `f_orig = f_solve * max(W,H) / 512`. Image dims from a
    dependency-free JPEG-SOF/PNG-IHDR parse.
  - Extrinsics: `w2c = inv(cams2world)`, qvec via Shepperd (verified
    round-trip < 1e-9 over 200 random rotations).
  - Camera↔image matching by filename STEM (007 lesson: extension
    drift must not break identity); refuses on uncovered images.
- `v4exec.py reconstruct-matcha --sfm posed` → algo `matcha@1`:
  - stages `sparse/0` next to `images/` on the host, runs
    `train.py -s /work --sfm_config posed`.
  - `weld_to_solve_sim` flips from correction to **verification gate**:
    posed welds must come back ≈ identity (refuse at residual > 0.5%
    or rotation > 2°) — anything more means the posed path wasn't
    honored (T-003: fail loudly).
  - `matcha_reference()` prefers matcha@1 over matcha@0 (da3 reference).
- matcha@0 stays the default; nothing already-built rebuilds (identity
  hashing keeps both eras addressable).

## DA3 side: same root cause (da3@1)

003 also exposed the same disease on the DA3 branch. The unposed DA3
driver (`da3_infer_gs.py`) lets DA3 estimate its own cameras; on 003
that estimate is off by **60.7% of camera spread (scale 0.58)** vs the
ingest solve — the fuse correctly REFUSED (>10% gate). Same fix:
`DepthAnything3.inference()` natively takes `extrinsics`/`intrinsics`,
so `da3_infer_posed.py` (new, versioned in research/real2sim) feeds the
ingest solve. `reconstruct-da3 --sfm posed` → algo `da3@1`, with
`cameras` added as a resolved input (da3@0 stays subset-only /
gauge-independent). `matcha_reference` already prefers matcha@1.

The matcha@1 posed weld weld→solve gauge-sim came back **scale 1.0004,
rot 0.0°, residual 0.40%** — the posed path is honored; the gauge IS
the ingest gauge (the 089 gauge-sim is now pure verification here).

## Verification (pre-GPU, T-012)

- Minted sparse for 003 read back with the CONTAINER'S OWN reader
  (`matcha.dm_utils.dataset_readers`): PINHOLE 3840×2160,
  params [1468.73, 1468.73, 1920, 1080], w2c matches ingest solve
  (translation exact, rotation ≤ 2e-4 ≈ 0.02° orthonormalization noise).
- Focal trust: 003's unposed weld estimated 193.8 vs ingest 195.8
  (512-space) — within 1%, independent solves agree on intrinsics.

## Status Notes

- 2026-06-12: Story minted mid-implementation. colmap_posed.py +
  v4exec wiring done and verified against the container reader.
- 2026-06-12: matcha@1 posed weld (003 on t) landed rc=0 in 838s;
  gauge-sim verification scale 1.0004 / rot 0.0° / residual 0.40%
  ("Using calibrated poses" confirmed in container log). orient + tetra
  + tsdf materialized. Unposed da3 then REFUSED (60.7% / scale 0.58) →
  built da3@1 posed inference path; landed fuse 0.2% / scale 1.0008,
  ICP 2.3°/0.103m fitness 0.54 (watch: lower than 009/006 ~0.9, likely
  da3 coverage past the matcha reference — operator visual check pending).
  Scene built + loaded; view-01 captured (slot 01, lens 25mm), renders
  dispatched. cmd_views made idempotent by captured_name (ghost-slot
  guard). Awaiting operator view-02 + runoff verification (T-020).

## CLOSED: SOLVED (2026-06-13)

Operator ranked 003-firepit's runoff: "003 has been ranked and looks
correct." That is the T-020 gate — 003, the forcing scene that was
architecturally unbuildable under the re-solve path (matcha 3.1-3.9%,
DA3 60.7% disagreement with the ingest solve), now reconstructs cleanly
and ranks correctly on both branches via the posed path.

Resolves STO-SCN-089 sweep follow-up #1. The posed path
(`--sfm posed` → matcha@1 / da3@1) is now the root-cause fix for the
gauge-sim class: when the ingest solve is fed in, no tool re-solves, so
no arbitrary gauge is minted and the 089 gauge-sim becomes pure
verification. Available for the remaining-scene sweep; matcha@0/da3@0
stay the default (nothing already-built rebuilds — identity hashing
keeps both eras addressable).

## Follow-up — DONE 2026-06-13

Stale v3 guidance pointed at the retired writer
(`sync_comparison_views.py` → `cameras.json`) instead of the v4-native
graph writer (`v4exec views-from-blend` → `views/<slot>/view.json`).
Behavior was always fine (viewport_capture.py already had a v4
`_derive_run_context` branch and the capture succeeded) — this was a
guidance-only T-025/locked-#11 trap for the next caller. Fixed:
- `viewport_capture.py`: module docstring, the `next:` return hint (now
  layout-aware — emits the exact `views-from-blend` command for v4, the
  legacy command for v2 `pipeline-/run-` blends), the store-membership
  error string, and the read-back comment.
- `.claude/commands/camera-save.md`: header, intro (v4-is-the-path
  callout), preconditions (accept v4 layout + stale-blend warning),
  step 3 (the `views-from-blend` block + ghost-slot idempotency note),
  step 4, and the notes. All residual `sync_comparison_views.py`
  mentions are now explicitly tagged "legacy v2 — never v4".
- Verified: helper parses, `next` ternary branches correctly for both
  layouts.
