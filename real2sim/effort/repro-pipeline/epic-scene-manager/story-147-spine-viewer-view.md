---
xid: STO-SCN-147
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: [STO-SCN-146]
bd-id: krabby-cq0y
assignee: scout
---

# Spine Viewer view — camera spine + color-coded subsets

## Summary

A **Spine Viewer** view in the Scenes config area: the posed camera **spine** (frustums along the
trajectory) with cameras **color-coded by subset** (PRIMARY + each subset distinct), in the
gravity-aligned frame. Reuses the existing `verify_viewer/viewer.html` lineage.

## Context

`verify_viewer/viewer.html` + `build_verify.py` already render the posed frustums + ground grid +
gravity up (`gauge_up`) from `frustums.json`. This story surfaces that as a tab view and adds
per-subset coloring. Full spec + reuse map: **EPI-SCN-SCENE-MANAGER**.

## Design / scope
- Embed the `viewer.html` frustum/spine renderer in the config area (embed first; port later if needed).
- Build the spine data from the scene's solve (`posed_from_sparse` → frustums), tagging each camera
  with its **subset membership** so frustums color by subset; a legend lists subsets with counts.
- Up/orientation from `gauge_up`; ground grid (existing).

## Definition of Done
- [ ] Spine view renders the posed frustums for the selected scene, oriented to gravity.
- [ ] Cameras are color-coded by subset (PRIMARY + others) with a legend.
- [ ] Reuses `verify_viewer/viewer.html` + `build_verify.py`/`posed_from_sparse`/`gauge_up` (no re-port of the math).

## Out of scope
- Editing subsets (read-only view); the per-subset photo grid (STO-SCN-148).
