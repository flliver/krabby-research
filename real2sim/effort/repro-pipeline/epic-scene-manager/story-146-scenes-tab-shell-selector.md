---
xid: STO-SCN-146
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-16
depends-on: []
bd-id: krabby-w424
assignee: scout
title: Scenes tab shell + scene selector header
---

# Scenes tab shell + scene selector header

## Summary

A **"Scenes" tab** in the Studio app (`rate_renders/`) with a Rank-style **header scene selector**
(top-ranking image per scene, click/hover to select) over a large **scene-config area** + a
**view switcher**, and the **New Scene** button (top, right of the list). This is the shell every
other Scene-Manager story plugs into (Metadata 153 / Spine 147 / Subsets 148 mount in the switcher).

## Context

The Studio app already has the Rank tab with a scene selector + `/api/scenes` + `/api/scene/<scene>`
(`rate_renders/server.py` + `static/`). This story adds a sibling tab reusing those patterns. Full
spec + reuse map: **EPI-SCN-SCENE-MANAGER**.

## Design / scope
- Tab routing in the existing shell (Rank | Scenes …); Scenes tab selectable.
- **Header selector**: horizontal scroll of scenes; each chip = the scene's top-ranking render/image
  (reuse Rank's aggregate/ranking to pick the top image); click or hover → select; selected scene
  drives the config area below. **New Scene** button at top-right of the list (opens STO-SCN-149).
- **Config area** = the lower majority of the screen; hosts a **view switcher** (Metadata 153 /
  Spine 147 / Subsets 148 — each its own story) and renders the selected view. This story scaffolds
  the switcher; the views land in their own stories.

## Definition of Done
- [x] Scenes tab present + selectable; header selector lists scenes with top-image chips; click/hover selects.
- [x] Config area fills the lower majority; view switcher scaffolded (views mount in 147/148/153).
- [x] New Scene button placed (wires to STO-SCN-149).
- [x] Reuses the existing Studio shell + `/api/scenes` (no new app).
- [ ] **Operator-verified (T-020):** open Studio (port 8090), click the **Scenes** tab, confirm the selector lists scenes, clicking selects, and the view switcher toggles Metadata/Spine/Subsets.

## Build notes (2026-06-16)
- `rate_renders/static/index.html`: added `#tab-bar` (Rank | Scenes), wrapped Rank in `#tab-rank`, added `#tab-scenes` (selector header + `#scenes-config` + `#scenes-viewbar` + `#scenes-view`).
- `rate_renders/static/scenes.js` (NEW, self-contained IIFE; shares no globals with `app.js`): tab switching, scene selector reusing `/api/scenes` + `.scene-card` styles, view switcher, New-Scene stub. Exposes `window.scenesViews` registry so 147/148/153 attach view renderers without editing the shell.
- `rate_renders/static/style.css`: `#tab-bar`, `#scenes-*`, `.view-placeholder` (reuses the theme vars + `.scene-card`/`.scene-arrow`).
- Verified: HTML tag-balanced, `node --check` clean on both JS files, assets serve `200` on the live 8090 server (`/static/scenes.js` 200/5817B, index emits tab-bar + scenes.js). Lazy-loads scenes only when the tab is first opened (Rank path untouched).
- **Host note for operator:** built into `rate_renders/` per the epic reuse-map. A sibling **Pipeline Studio** (`studio/`, EPI-SCN-PIPELINE-STUDIO) also runs (8091) and subclasses this app — if the Scenes tab should ultimately live there instead, the shell ports cleanly (additive files).

## Out of scope
- The Metadata (153), Spine (147), Subsets/Grid (148) views and the New-Scene flow (149–152) —
  this is the shell they mount in.
