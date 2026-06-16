---
xid: STO-SCN-153
parent: ./epic.md
kind: story
effort: scn
size: S
status: draft
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
- [ ] Metadata view renders for any selected scene with the fields above.
- [ ] Datum/scale status reflects `datum.json` (calibrated `s` + provenance, or "uncalibrated").
- [ ] Served from the scene API (extend as needed); read-only.

## Out of scope
- The tab shell / selector / view switcher (STO-SCN-146). The 3D spine (147) and subset grid (148) views.
