---
xid: STO-SCN-085
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-11
depends-on: []
bd-id: krabby-fd0
shipped: 2026-06-11
tasks: 2
complete: 2
---

# FEATURE: rank UI shows MISSING renders (expected = meshes × canonical views)

## Summary

FEATURE (operator, 2026-06-11): the rank UI shows MISSING renders —
we know the expected set (every mesh artifact × every canonical view
slot), so a gap should be visible instead of silently absent.

## Shipped

- Server (`_scene_payload` v4): `missing: {slot: [identities]}` =
  expected (all meshes/conditioned from the scan) minus the render
  index. ~6 lines; no new scanning.
- Frontend: inert placeholder tiles in the grid for the current
  view's missing set — dashed border, "not rendered yet", label, and
  the exact materialize command (`v4job render-missing <scene>`).
  Unrankable, undraggable.
- Verified on 004: 4 missing tetra renders per slot, correctly
  labeled (runner-v1 / dtu-ref / 16-strong / r3) — exactly the gap
  the gauge-repair chain is filling right now.

## Deliberately NOT included (operator design point)

A "materialize" button on the tile would turn the rank UI into a job
UI — deferred until decided deliberately.

## Definition of Done

- [x] Missing set computed server-side from the known expected set.
- [x] Placeholder tiles render in the grid (reload to pick up).
