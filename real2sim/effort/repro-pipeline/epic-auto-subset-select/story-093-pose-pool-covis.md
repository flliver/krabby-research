---
xid: STO-SCN-093
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-13
depends-on: [STO-SCN-091, STO-SCN-092]
bd-id: krabby-0dk
assignee: krabby
---

# Pose the pre-culled pool → poses + co-visibility graph

## Summary

Solve camera poses for the pre-culled candidate set with a scalable, camera-model-correct
solver, and emit the **co-visibility / track graph** (which images see which 3D points,
pairwise overlap) that the selector consumes.

## Context

Step 2 of the pipeline. Depends on the camera model (STO-SCN-091) and the pre-cull
(STO-SCN-092). The co-visibility graph is the key output — it's what makes "best N"
a principled, automatable choice (design story conclusion #2).

## Problem

The solver must (a) use the right camera model, (b) scale to a few-hundred-frame
candidate set without the drift that killed the 300-frame MASt3R run, and (c) expose the
track graph. Solver choice depends on modality/overlap (conclusion #4): sequential/SfM
for ordered video, feed-forward for sparse — see the corpus solver landscape.

## Design

### Approach

Pick the solver by modality + the camera model from STO-SCN-091, run it on the candidate
set, and persist `poses + co-visibility graph`. Gate the result with the planarity /
sanity check (conclusion #5) before passing downstream. Candidate solvers: COLMAP-sequential
(video, needs deploy), GLOMAP/FastMap (GPU SfM), feed-forward (sparse). Exact solver
selection is the main open decision of this story.

### Changes

| File | Change |
|------|--------|
| pose stage | solver dispatch (modality + camera model) → poses + track graph |
| solve-validity gate | planarity / sanity ratio on solved cameras |

## Definition of Done

- [ ] Pre-culled pool → poses + co-visibility graph, using the profile's camera model.
- [ ] Passes the solve-validity gate (no nebula); fails loud otherwise.
- [ ] Track-graph output in a form the selector (STO-SCN-094) consumes.

## Spine note (longer-term — see STO-SCN-096 conclusion #7)

A full video is too big to pose at once. This story poses *one segment* (a tractable
window of the spine). The **global registration** of M segments into one cohesive
gauge — per-segment local solves + pose-graph optimization / loop closure / global BA —
lives in the sibling spine-assembly epic (EPI-SCN-SPINE-ASSEMBLY). Keep the per-segment
solver's output (poses + track graph) in a form that the global registration can consume
across segment boundaries.

## Implementation Notes

**Solver dispatch** keyed on `(modality, camera_model)` — modality from ingest/pre-cull
(ordered video vs sparse unordered), camera_model from STO-SCN-091:
- **ordered video** → sequential SfM (COLMAP-sequential: GPU SIFT + CPU mapper, honours the
  DJI fisheye `SIMPLE_RADIAL_FISHEYE` model) or **GPU SfM** (GLOMAP / FastMap) when CPU
  mapping is the bottleneck.
- **sparse unordered** → feed-forward (VGGT / DA3).
- The **exact pick is the open decision of this story** (T-002 — not pre-committing).
  Reference: the research-corpus `3d-reconstruction/pose-solver-landscape` entry
  (proposal `kp-20260613-88fb`).

**Posed-path reuse (free win).** Where an ingest solve already exists, `colmap_posed.py`
already mints a COLMAP `sparse/0` + `posed.json` from it with **no re-solve** (the
matcha@1 / da3@1 mechanism shipped under STO-SCN-090). The co-visibility graph then derives
directly from that COLMAP model — no new solve needed for already-posed scenes.

**Co-visibility graph output.** Persist a sidecar the selector (094) reads:
`image_id → covered 3D-point ids`, `pair(i,j) → shared-point count + mean triangulation
angle`. This is exactly what COLMAP's `points3D` + `images` tracks give for free; for
feed-forward solvers, derive overlap from projected-depth agreement.

**Validity gate (conclusion #5).** PCA on camera centers → out-of-plane / in-plane spread
ratio. Reuse the planarity check prototyped this session (sibling to
`gauge_align`): a handheld ground walk lands at a few %; the 300-frame MASt3R **nebula** hit
~60% — fail loud at a threshold (~15–20%) before any downstream work.

**Test.** A known-good scene reconstructs and passes the gate; the 300-frame MASt3R nebula
input is caught by the gate (the regression that motivated the whole epic).

## Validation findings (2026-06-13) — solver decision RESOLVED + undistort proven

The open solver decision is settled: **FastMap** (GPU SfM), deployed in STO-SCN-101,
with **fisheye undistorted to pinhole first** (FastMap takes only PINHOLE/SIMPLE_RADIAL).
Empirically validated fleet-side on 001-patio:

| Input set | Registered | Points3D |
|-----------|-----------|----------|
| Fisheye-300 (native 155° FOV) | 300/300 | 133,548 |
| Undistorted-300 (cropped pinhole, sparse) | 227/300 | 115,857 |
| **Undistorted-539 (cropped, full blur/dup-culled pool)** | **539/539** | 256,421 |

1. **Undistort step (`undistort_fisheye.py`) works** — reads the 102 calibration
   (`cv2.fisheye`, RMS 0.86 px), remaps to clean pinhole, no black borders. CPU, ~10 s,
   one-shot/cached (not on the hot path — orthogonal to the GPU-solver rule).
2. **For hyperlapse, KEEP THE FULL blur/dup-culled POOL — don't thin to 300.** The
   undistort FOV-crop drops overlap at sparse baselines (300→227), but tighter baselines
   (539) **fully recover** it (539/539). The old 300 cap was a mast3r-sfm 16 GB artifact;
   FastMap is GPU-scalable. ⇒ 092's `--target` should be high (or 0) for hyperlapse.
3. **Pre-cull MUST order by capture time** (`original_name`), not store hash order:
   hash-order dedup found 4 near-dups; capture-order found **403**. (Fix to fold into 092's
   store-path resolution.)
4. **FastMap's `sparse/0` is not cleanly pycolmap/model_analyzer-readable** (errors /
   hangs). The covis extractor must read the COLMAP `.bin` files directly (the struct
   header read worked reliably) or post-fix FastMap's output.
5. **Containers write as root** → host-side permission friction; run the krabby tools with
   `--user $(id -u):$(id -g)` (the container-as-root follow-up) rather than chown'ing.

Deliverables (built + validated on the real 539 model):
- `undistort_fisheye.py` — fisheye → pinhole via the 102 calibration.
- `run_fastmap.sh` (from 101) — GPU solve (COLMAP match + FastMap).
- `covis_graph.py` — `sparse/0` bins → coverage + pair shared-count/angle + connectivity
  (539 fully connected, 0 isolated).
- `validity_gate.py` — planarity nebula detector (good walk 5.3% PASS).
- `solve_plan.py` — solver dispatch (profile + modality → solve plan): fisheye→undistort→
  FastMap, dewarped→DA3, pinhole→FastMap; hyperlapse→keep-full-pool + exhaustive matcher.

**Remaining 093 work (store-integration; needs the v4 store mounted):** wire the dispatch
into a v4 graph node that runs the chain and writes the covis + poses as a store artifact
094 consumes (a HUG-SCN-005 store-writer change — design-gated). The CPU cores above are
all done + unit-tested + validated on real data.

## v4 store-node wiring — build spec (design approved 2026-06-13)

Turns the validated `precull → undistort → FastMap → covis → gate` chain into a
content-addressed, idempotent graph; **v4exec is the sole store writer** (HUG-SCN-005 #11).

### Graph shape (extends the repro pipeline)
```
images-subset (precull, 092, capture-order)
   → solve-cameras [algo by solve_plan: fisheye/pinhole → fastmap@0, dewarped → da3@0 (future)]
   → covis@0  → covis.json + validity.json   ──►  STO-SCN-094 selection
```

### New tasks (`real2sim/tasks/`)
**`solve-cameras` algo `fastmap@0`** (sibling to `mast3r-sfm@0`)
- inputs: the precull subset.
- settings (from `solve_plan`, enter the identity hash): `camera_model`, `undistort`,
  `balance`, `matcher`.
- placement: `images/subsets/{subset}/cameras/{identity}/`.
- outputs: `sparse/0/` (COLMAP model — covis source), `cameras.json` (poses), `intrinsics.json`.
- exec (`cmd_solve`, GPU host, krabby-fastmap container): resolve profile (091) → `solve_plan`
  → if `undistort`: undistort raw frames in the **host workdir, transient** (decision A) →
  COLMAP extract+match (per `matcher`) → FastMap → gather `sparse/0` → `write_metadata`
  (algo `fastmap@0`; measured host/duration/**registered-count**).

**`covis@0`** (new)
- inputs: the `fastmap@0` solve (`sparse/0`).  settings: `min_overlap`.
- placement: `…/cameras/{up_solve}/covis/{identity}/` (**under the solve** — decision B).
- outputs: `covis.json` (coverage, pair shared-count+angle, connectivity) + `validity.json`.
- exec (`cmd_covis`, CPU): `covis_graph.py` + `validity_gate.py`; **hard-fail on nebula**
  (decision C) — a failed solve never reaches 094.

### Identity / idempotency
- solve = `hash({subset}, solve_settings, "fastmap@0")` → NOOP on re-run; dispatch change ⇒ new artifact.
- covis = `hash({solve}, {min_overlap}, "covis@0")`.
- Undistorted images are **not persisted** (transient, decision A; multi-GB/scene avoided —
  re-undistort is ~10 s). Only `intrinsics.json` (small) is kept for provenance.

### Approved decisions
- **A** undistort transient (host workdir, not a store set).
- **B** covis is a sub-node under the solve.
- **C** validity gate hard-fails the covis node on nebula (+ records verdict).
- **D** **bake** `solve_plan/undistort_fisheye/covis_graph/validity_gate` into the
  krabby-fastmap container (operator policy: results from baked tools, not mounted `/tmp`)
  → requires a 101 container rebuild (tools currently mounted from `~/build/fastmap`).
- **E** `modality` (hyperlapse|video|photos) declared per-scene in `<scene>/capture.json`
  (alongside `mode`) — not inferable; consumed by `solve_plan`.

### Build checklist (when picked up)
1. `tasks/solve-cameras.json` algo `fastmap@0` (or `tasks/solve-fastmap.json`) + `tasks/covis.json`.
2. `cmd_solve` + `cmd_covis` in `v4exec.py` (store-writer-mediated).
3. Graph edges in the repro pipeline graph.
4. krabby-fastmap **container rebuild** baking the krabby tools (decision D).
5. `capture.json` gains `modality` (decision E); `cmd_ingest`/`cmd_solve` read it.
Already built + validated: `solve_plan`, `undistort_fisheye`, `run_fastmap`, `covis_graph`,
`validity_gate`, precull capture-order.

## Out of scope

- The selection itself (STO-SCN-094).
- Global cross-segment registration / drift management (sibling epic).
- Deploying a specific solver binary is an implementation detail decided in this story.
