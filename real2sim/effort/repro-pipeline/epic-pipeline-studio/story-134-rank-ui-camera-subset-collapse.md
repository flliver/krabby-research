---
xid: STO-SCN-134
parent: ./epic.md
kind: story
effort: scn
size: S
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-96bb
assignee: krabby
---

# Rank UI: camera-subset section (below Manifest) + collapsible Live Results (start collapsed)

## Summary

Two info-pane refinements to the rank UI (`rate_renders` app, embedded in the studio `/rank`
tab): (1) a **"Camera Subset"** section under the Manifest panel showing **which cameras/frames
the focused rendering was built from**, and (2) make **"Live Results" collapsible, defaulting to
collapsed** so the manifest + subset are the focus.

## Context

A render's identity is partly *which frames fed it* — matcha-15 used 15 selected views, DA3-24
used 24, the historical matchas used 12. The operator can't currently see that per-variant in
the rank UI, yet it's central to judging a reconstruction (coverage vs redundancy). The Live
Results leaderboard, meanwhile, is useful but takes vertical space the operator doesn't always
want open. Operator request (2026-06-15).

## Enhancements

- [ ] **"Camera Subset" section** below the Manifest panel. For the focused variant, list the
      cameras/frames it was reconstructed from — count + the member frame names
      (e.g. *"15 cameras: frame_0010, frame_0031, … frame_0935"*). Source = the variant's
      provenance: the FINAL-N subset members (`subsets/<subset>/subset.json` → `members`, mapped
      to `original_name`), or the solve cameras for the selection. Scrollable if long.
- [ ] **Collapsible "Live Results"** — wrap the panel in a collapse toggle (e.g. `<details>` or
      a header click), **default collapsed**. Remembers the operator's open/closed choice
      (localStorage) so it doesn't re-collapse every render.

## Design

### Approach
- **Server:** extend the per-render manifest payload (`rate_renders/server.py` —
  `describe_render` / `_scene_payload`) with a `camera_subset` field: `{n, frames: [...]}`
  derived from the variant's resolved subset (members → original frame names). Read-derived
  (no new stored field), same pattern as the description (STO-SCN-106).
- **Client:** `rate_renders/static/{index.html, app.js, style.css}` — render a `#camera-subset`
  block under `#manifest-content`; wrap `#results-content` in a collapsible with a persisted
  `collapsed` state (default collapsed).

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/server.py` | add `camera_subset` (n + frame names) to the manifest payload |
| `real2sim/rate_renders/static/index.html` | "Camera Subset" section under Manifest; collapse wrapper on Live Results |
| `real2sim/rate_renders/static/app.js` | populate camera-subset from manifest; wire collapse + persist state (start collapsed) |
| `real2sim/rate_renders/static/style.css` | section + collapse styling |

## Definition of Done

- [ ] Focusing a render shows its **Camera Subset** (count + frame names) below the Manifest.
- [ ] Distinct subsets read correctly (matcha-15 → 15, DA3-24 → 24, historical → 12).
- [ ] **Live Results starts collapsed**; toggling open/closed persists across renders + reload.
- [ ] No regression to manifest / ranking / submit / descriptions / profiles / badges.
- [ ] Operator-verified in `/rank` (T-020).

## Out of scope

- Changing the leaderboard math or the ranking flow.
- Per-camera thumbnails of the subset (frame names suffice for now).

## Implementation Notes

_(Operator request 2026-06-15 — "add a section below Manifest that captures the camera subset"
+ "allow collapse of Live Results, start collapsed".)_
