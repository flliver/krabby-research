---
xid: STO-SCN-084
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-11
depends-on: []
bd-id: krabby-bzd
shipped: 2026-06-11
tasks: 1
complete: 1
---

# FEATURE: mesh stats (size, verts, tris) in the rank manifest panel

## Summary

FEATURE (operator, 2026-06-11): show mesh stats in the rank manifest
panel — file size and the typically relevant counts.

## Shipped

- Server: `_ply_stats` (header-only PLY read, 64KB cap — no heavy
  deps) attaches `mesh: {verts, faces, size_mb}` to every v4 manifest
  entry (meshify + conditioned).
- Frontend: manifest panel renders `mesh: 6.3M verts · 12.6M tris ·
  462.8 MB` above the settings.
- Verified live on 013: tetra 6.3M/12.6M/463MB vs tsdf
  30.8M/58.3M/2220MB — exactly the deliverable-scale signal the
  panel was missing.

## Definition of Done

- [x] Stats visible per variant in the manifest panel (reload page).
