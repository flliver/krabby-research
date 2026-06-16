---
xid: STO-SCN-148
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
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
- [x] Subset list renders (PRIMARY marked) for the selected scene; selecting one loads its photos.
- [x] Paged grid with 1 / 2×1 / 2×2 / 3×3 / 4×4 layouts, mirroring the Rank viewer.
- [x] Reuses Rank's grid/paging; subset membership served from the scene API.
- [ ] **Operator-verified (T-020):** Studio → Scenes → Subsets; confirm subset list (PRIMARY flagged), datum badge, layout cycle + paging render real photos.

## Build notes (2026-06-16)
- **Backend** (`rate_renders/server.py`): pure `scene_subsets(scene_dir)` —
  lists each REAL subset (resolves the `primary` symlink to a flag, not a
  duplicate), with label/mechanism, member image hashes, camera solves, and a
  `has_datum` flag. Routes: `GET /api/scene/<scene>/subsets` and
  `GET /api/photo/<scene>/<hash>.jpg` (serves `images/<hash>/image.jpg`,
  path-clamped to the store).
- **Frontend** `rate_renders/static/scenes-subsets.js`: registers
  `window.scenesViews.subsets` — subset list (primary/datum badges, counts) +
  paged photo grid with 1 / 2×1 / 2×2 / 3×3 / 4×4 layouts; CSS in `style.css`.
- **Verified:** `tests/test_scene_subsets.py` + standalone driver against the
  real 001-patio store (6 subsets, primary→3A6MH6U5VKYP, datum on 6EHLYO3MF3QU
  with 539 members). HTTP end-to-end on a throwaway port: `/subsets` lists,
  `/api/photo/...jpg` → `200 image/jpeg 598 KB`, path traversal → `404`.

## Out of scope
- Editing/creating subsets (best-N selection is EPI-SCN-AUTO-SUBSET-SELECT); the spine 3D view (147).
