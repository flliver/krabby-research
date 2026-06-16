---
xid: STO-SCN-146
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: []
bd-id: krabby-w424
assignee: scout
---

# Scenes tab shell + scene selector header + metadata view

## Summary

A **"Scenes" tab** in the Studio app (`rate_renders/`) with a Rank-style **header scene selector**
(top-ranking image per scene, click/hover to select) over a large **scene-config area**, plus the
first/simplest view — **metadata display** — and the **New Scene** button (top, right of the list).
This is the shell every other Scene-Manager story plugs into.

## Context

The Studio app already has the Rank tab with a scene selector + `/api/scenes` + `/api/scene/<scene>`
(`rate_renders/server.py` + `static/`). This story adds a sibling tab reusing those patterns. Full
spec + reuse map: **EPI-SCN-SCENE-MANAGER**.

## Design / scope
- Tab routing in the existing shell (Rank | Scenes …); Scenes tab selectable.
- **Header selector**: horizontal scroll of scenes; each chip = the scene's top-ranking render/image
  (reuse Rank's aggregate/ranking to pick the top image); click or hover → select; selected scene
  drives the config area below. **New Scene** button at top-right of the list (opens STO-SCN-149).
- **Config area** = the lower majority of the screen; hosts a view switcher (Metadata / Spine /
  Subsets — the other stories) and renders the selected view.
- **Metadata view** (this story): scene code/name, capture mode, counts (images, subsets, solves),
  scale/datum status (`datum.json` if present), pipeline state — from `/api/scene/<scene>` (extend
  the endpoint as needed).

## Definition of Done
- [ ] Scenes tab present + selectable; header selector lists scenes with top-image chips; click/hover selects.
- [ ] Config area fills the lower majority; view switcher scaffolded; Metadata view renders for any scene.
- [ ] New Scene button placed (wires to STO-SCN-149).
- [ ] Reuses the existing Studio shell + `/api/scenes` (no new app).

## Out of scope
- The Spine/Subsets/Grid views (147/148) and the New-Scene flow (149–152) — this is the shell they mount in.
