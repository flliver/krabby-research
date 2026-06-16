---
xid: STO-SCN-153
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
date: 2026-06-16
depends-on: [STO-SCN-146]
bd-id: krabby-agn3
assignee: scout
---

# Scene metadata view

## Summary

A **Metadata** view in the Scenes config area: the selected scene's identity + state — code/name,
capture mode, counts (images, subsets, solves), scale/datum status (`datum.json` if present), and
pipeline state — served from `/api/scene/<scene>`.

## Context

Split out of STO-SCN-146 (which now delivers only the tab shell + selector + view switcher). This is
the first/simplest view mounted in that shell. Full spec + reuse map: **EPI-SCN-SCENE-MANAGER**.

## Design / scope
- Read-only panel mounted in the config area's view switcher (the 146 scaffold).
- Fields: scene code + name; capture mode (video/images/fisheye); counts (canonical images, subsets,
  solves, render views); **scale/datum** (`s` + provenance from `datum.json` when present, else
  "uncalibrated"); pipeline state (ingested / scouted / meshed).
- Extend `/api/scene/<scene>` to return these fields (or a sibling `/api/scene/<scene>/meta`).

## Definition of Done
- [x] Metadata view renders for any selected scene with the fields above.
- [x] Datum/scale status reflects `datum.json` (calibrated `s` + provenance, or "uncalibrated").
- [x] Served from the scene API (extend as needed); read-only.
- [ ] **Operator-verified (T-020):** open Studio → Scenes → select a scene → Metadata; confirm counts + calibrated/uncalibrated render (001-patio shows s=4.45).

## Build notes (2026-06-16)
- **Backend** `GET /api/scene/<scene>/meta` (`rate_renders/server.py`): pure
  module-level `scene_meta(scene_dir: Path)` — identity (code/name), capture
  mode (video/images/empty), counts (canonical images / subsets / solves /
  render views), datum/scale from the `datum.json` sidecar, and coarse
  pipeline-state flags (ingested/solved/scouted/meshed/calibrated). Route is
  matched **before** the generic `/api/scene/` prefix (ordering verified).
- **Frontend** `rate_renders/static/scenes-meta.js`: registers
  `window.scenesViews.meta` (the 146 registry) — read-only panel; CSS in
  `style.css`.
- **Verified:** `scene_meta` unit-checked on synthetic trees + the real
  001-patio store (942 images, 7 subsets, 5 solves, s=4.45); also
  `tests/test_scene_meta.py` (pytest, for the project runner — no pytest in
  the local env, so verified via a standalone driver). HTTP end-to-end on a
  throwaway port: `/meta` returns the payload, missing-scene → error, generic
  `/api/scene/` unaffected. Assets serve `200`.

## Out of scope
- The tab shell / selector / view switcher (STO-SCN-146). The 3D spine (147) and subset grid (148) views.
