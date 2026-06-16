---
xid: EPI-SCN-SCENE-MANAGER
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-16
hugs: []
tenets: []
bd-id: krabby-mnu6
assignee: scout
---

# Scene Manager — the Scenes tab (browse + create scenes; formalize the pipeline into UI)

## Problem Statement

The full capture→3D pipeline already exists as CLIs and one-off viewers (ingest, canonicalize,
spine solve, DA3 gaussian/mesh, scout, render views, MEASURE/normalize-units). But there is **no
operator UI to manage scenes** — creating a scene and driving it through the pipeline means hand-
running `v4exec` subcommands and serving `verify_viewer/` HTML by hand. This epic adds a **"Scenes"
tab** to the existing Studio app (`rate_renders/server.py` + `static/`, which already hosts the
Rank tab) that **formalizes the existing machinery into a UI** — browse a scene's structure and
drive a new scene end-to-end. **Most of the back-end is built; this epic is the UI layer + wiring.**

## Goals

- A **Scenes tab** alongside Rank: a Rank-style header **scene selector** (top-ranking image per
  scene, click/hover to select) over a large **scene-config area** filling the rest of the screen.
- **Browse** an existing scene: metadata, the camera **spine** (color-coded subsets), the **camera
  subsets** (list + primary), and a **paged photo grid** per subset (1 / 2×1 / 2×2 / 3×3 / 4×4).
- **Create** a scene end-to-end: New Scene → auto 3-digit code + kebab name → pick video / images /
  folder → MOVE-or-UPLOAD → canonicalize to content-hash → run the ingest-scene pipeline (host
  pick, spine solve, PRIMARY subset, DA3 gaussian + mesh) with **phase progress** → Scout (render
  views) → **MEASURE units + Normalize**.
- **Reuse, don't rebuild** — wire the existing CLIs/viewers behind UI; new code is the tab, the
  endpoints, and the orchestration/progress glue.

## Non-Goals (Out of Scope)

- New reconstruction algorithms — all back-end steps exist (spine STO-SCN-097/098/099, scout +
  MEASURE STO-SCN-105/144, DA3 gaussian/mesh, cull/condition). This epic surfaces them.
- The Rank tab itself (EPI-SCN-PIPELINE-STUDIO) — Scenes is a sibling tab; it reuses Rank's
  selector + grid patterns.
- N-video multi-capture (a single video / N images / one folder now; N videos noted as later).

## What already exists (the reuse map)

| UI/flow piece | Reuse |
|---|---|
| Studio app shell, tabs, scene selector, render grid | `rate_renders/server.py` (`/api/scenes`, `/api/scene/<scene>`) + `static/` (Rank tab) |
| Spine + frustums + color-coded subsets viewer | `verify_viewer/viewer.html` (+ `build_verify.py` `frustums.json`, `posed_from_sparse`, `gauge_up`) |
| Scout gaussian + Render Views + MEASURE mode | `verify_viewer/match.html` (STO-SCN-144 MEASURE) + `build_verify.py` |
| Normalize-units back-end | `metric_scale.py` + `calibrate_datum.py` + `datum_frame.py` → writes `datum.json` (DONE) |
| Ingest / canonicalize / capture metadata | capture-profile ingest (`tests/test_capture_profile_ingest.py` lineage) + `v4exec` |
| Spine solve / subsets / DA3 gaussian + mesh | `v4exec` (spine, scout, meshify) + EPI-SCN-SPINE-ASSEMBLY |
| Job/progress monitoring | `v4exec` job records + `/api/jobs/`, `/api/materialize/` (server.py) |

## Store layout (target)

```
scenes/XXX-<name>/
  videos/capture/video.<ext>            # if video ingest
  images/ingress/*                      # if image ingest (pre-canonical)
  images/<hash>/image.jpg               # canonical image (content-addressed)
  images/<hash>/metadata.json           # extracted capture metadata
  images/subsets/{primary,<id>}/...     # camera subsets (PRIMARY auto)
  .../cameras/<solve>/...               # spine solve, scout, datum.json
```

## Stories

| # | XID | Story | Area | Size |
|---|-----|-------|------|------|
| 1 | `STO-SCN-146` | Scenes tab shell + scene selector header (+ view switcher) | browse | M |
| 1a | `STO-SCN-153` | Scene metadata view | browse | S |
| 2 | `STO-SCN-147` | Spine Viewer view — camera spine + color-coded subsets | browse | M |
| 3 | `STO-SCN-148` | Camera Subsets view — list + primary + paged photo grid (1/2×1/2×2/3×3/4×4) | browse | M |
| 4 | `STO-SCN-149` | New Scene — ingest (video/images/folder, MOVE-or-UPLOAD) + canonicalize to content-hash | create | L |
| 5 | `STO-SCN-150` | New Scene — run ingest-scene pipeline (host pick, spine, PRIMARY, DA3 gs+mesh) with phase progress | create | L |
| 6 | `STO-SCN-151` | Scout in the tab — view gaussian (default) + define Render Views | create | M |
| 7 | `STO-SCN-152` | MEASURE + Normalize Units in the tab (formalize STO-SCN-144 / `datum.json`) | create | M |

**Flow:** browse — 146 (shell + view switcher) hosts the read-views 153 (metadata) / 147 (spine) /
148 (subsets) over an existing scene; create
stories (149→150→151→152) are the New-Scene pipeline in order. 152 reuses the STO-SCN-144 MEASURE
mode + the `calibrate_datum`/`datum_frame` back-end already shipped (001-patio calibrated at s=4.45).

## Creation flow (the New-Scene spec, operator 2026-06-16)

1. **New Scene** (top, right of the scene list).
2. Auto-assign 3-digit code `XXX` (001, 002, … — next free).
3. Type a name → auto **kebab-case**.
4. Pick **1 video** OR **N images** OR **a folder of homogeneous images** (N videos: later).
5. Local → choose **MOVE** (back-end move, no copy) or **UPLOAD** (copy); we operate locally → MOVE.
6. Video → `scenes/XXX-<name>/videos/capture/video.<ext>`.
7. Images → `scenes/XXX-<name>/images/ingress/*`.
8. **Canonicalize** → `images/<hash>/image.jpg` + `images/<hash>/metadata.json` (images: move +
   extract metadata; video: extract frames + extract metadata); **show progress**.
9. **Ingest-scene pipeline** (exists): pick a processing host → **Spine Solver**
   (EPI-SCN-SPINE-ASSEMBLY) → generates **PRIMARY** subset → **DA3 gaussian** (default config) →
   **DA3 mesh** (default config); **show phase progress** (server-side monitor, simple client).
10. Show in **Scout** (center view), gaussian by default.
11. Scout → **define Render Views** (1…N).
12. Scout → **MEASURE units** (press `M`): pick P1 in ≥1 photos (`[ ]` nav, click) → `E`; pick P2
    in ≥1 photos → `E`; enter the P1:P2 distance (meters); press **Normalize Units**.

## Decisions

| XID | Decision | Status | Rationale |
|-----|----------|--------|-----------|
| — | Add as a tab in the existing Studio app, not a new app | Adopted | Reuse Rank's shell/selector/grid; one operator surface |
| — | Reuse `verify_viewer/{viewer,match}.html` behind the tab vs. re-port to the framework | Open | Decide per story (embed vs. port); favor embed first |

## Success Criteria

- [ ] Operator creates a scene from a video/folder and drives it to a scouted, unit-normalized state
      entirely in the Scenes tab — no hand-run CLIs.
- [ ] Browse views (metadata / spine / subsets / grid) render for any existing scene.
- [ ] All stories shipped; back-end reused (no reconstruction logic re-implemented).

## Progress (live — `/loop finish EPI-SCN-SCENE-MANAGER`)

| Story | State | Notes |
|---|---|---|
| STO-SCN-146 shell | **built** (b3c40c3), operator-verify pending | tab bar + selector + view switcher; `scenes.js` registry |
| STO-SCN-153 metadata | **built** (c87cd4a), operator-verify pending | `/api/scene/<scene>/meta` + `scenes-meta.js`; verified vs real 001-patio |
| STO-SCN-148 subsets | **built** (58e55a5), operator-verify pending | `/api/scene/<scene>/subsets` + `/api/photo/...` + `scenes-subsets.js`; verified vs real 001-patio |
| STO-SCN-147 spine | **scoped, deferred** | 3D WebGL surface — needs `build_verify` + viewer embed + visual verify (below) |
| 149–152 create flow | **scoped, deferred** | GPU + pipeline-orchestration + operator-bound (below) |

### What shipped autonomously (the browse-data foundation)
146 + 153 + 148 are the CPU-only, unit/HTTP-testable half of "browse a scene":
the tab shell + the Metadata and Subsets views, each with a pure
`server.py` helper (`scene_meta`, `scene_subsets`) + a `window.scenesViews`
registry renderer. All three are committed with tests
(`tests/test_scene_meta.py`, `tests/test_scene_subsets.py`) and verified
end-to-end against the real 001-patio store. **Operator T-020 exercise is
staged on each — not self-closed.**

### Why 147 + 149–152 were NOT auto-built (the autonomous boundary)
These cross out of CPU/testable territory and should be done **with the
operator**, not generated on an unverified shell (T-005/T-007/T-020):
- **147 Spine** — the deliverable is a 3D color-by-subset frustum viewer.
  Reuse is clear: `build_verify.build_frustums(sparse_dir, …)` already emits
  frustums + `gauge_up` orientation + cull box; a CPU `/api/scene/<scene>/spine`
  can return that (skip the splat-cull) and tag each camera by subset
  membership, fed to an embedded `verify_viewer/viewer.html` (add per-subset
  coloring + legend). But it needs a solve with `cameras/<solve>/sparse/0`
  (001-patio: `6EHLYO3MF3QU/62QEHJDAJZBI` qualifies) and **visual verification**
  that can't be self-checked here.
- **149–152 Create flow** — ingest/canonicalize (149), pipeline orchestration
  with phase progress (150), scout + render-views (151), MEASURE + Normalize
  (152). These run real GPU pipeline steps + the MEASURE T-020 surface; they
  need compute + the operator in the loop, not autonomous generation.

**▶ Operator action (unblocks the rest):**
1. **Verify the foundation.** Open the Studio app at
   **`http://krabby.organl.com:8090/`** (rate_renders — **restart it** to load
   the new `server.py` routes; static reloads live). Click **Scenes** → confirm
   the selector lists scenes, **Metadata** shows 001-patio (s=4.45, 942 images,
   6 subsets), **Subsets** lists subsets (PRIMARY flagged, datum badge) with the
   paged photo grid (1 / 2×1 / 2×2 / 3×3 / 4×4).
2. **Decide the host.** Confirm Scenes belongs in `rate_renders/` (per the
   reuse-map) vs. the sibling **Pipeline Studio** (`studio/`, 8091). Files are
   additive + port cleanly if you want it in `studio/`.
3. Then resume the loop (or hand 147 + 149–152 to a scout↔operator session) to
   build the 3D + create-flow stories against the verified foundation.

_Loop `cron 3b007906` stopped here on purpose: the remaining stories are
operator/GPU/3D-gated, so spinning the 60s timer adds no autonomous progress
(same call as the GOAL-SCN-001 T-020 wall). Re-arm with `/loop … finish
EPI-SCN-SCENE-MANAGER` after the foundation check._

## Retrospective

_(Fill in after epic completion.)_
