---
xid: STO-SCN-148
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: [STO-SCN-146]
bd-id: krabby-8x8p
assignee: scout
---

# Camera Subsets view — list + primary + paged photo grid (1/2×1/2×2/3×3/4×4)

## Summary

A **Camera Subsets** view: a **list of subsets** (PRIMARY marked) for the scene; selecting one shows
the **photos of that subset in a paged grid** with layout options **1 / 2×1 / 2×2 / 3×3 / 4×4**,
mirroring the Rank viewer's grid + paging.

## Context

Subsets live at `images/subsets/{primary,<id>}/…`; canonical photos at `images/<hash>/image.jpg`.
The Rank tab already has a render grid with layout/paging to mirror. Full spec + reuse map:
**EPI-SCN-SCENE-MANAGER**.

## Design / scope
- **Subset list**: enumerate `images/subsets/*` for the scene (PRIMARY first/marked), with counts;
  select one → grid below (extend `/api/scene/<scene>` to return subset → member-image list).
- **Photo grid**: thumbnails of the selected subset's images, layouts 1 / 2×1 / 2×2 / 3×3 / 4×4,
  **paged**, reusing Rank's grid + paging components.
- Click a photo → enlarge (reuse Rank's single-image view).

## Definition of Done
- [ ] Subset list renders (PRIMARY marked) for the selected scene; selecting one loads its photos.
- [ ] Paged grid with 1 / 2×1 / 2×2 / 3×3 / 4×4 layouts, mirroring the Rank viewer.
- [ ] Reuses Rank's grid/paging; subset membership served from the scene API.

## Out of scope
- Editing/creating subsets (best-N selection is EPI-SCN-AUTO-SUBSET-SELECT); the spine 3D view (147).
