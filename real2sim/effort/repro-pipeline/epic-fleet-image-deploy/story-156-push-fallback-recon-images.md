---
xid: STO-SCN-156
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-16
depends-on: []
bd-id: krabby-zc3x
assignee: krabby
---

# Push mast3r/slam3r/vggt + scene-recon images to registry; rescue diverged local-only images first

## Summary

The image families that exist only as local `:latest` on individual
hosts are rescued and pushed to the registry with versioned tags, so
they are no longer one `docker image prune` away from being lost
forever.

## Context

The audit's main *nuke* risk. These exist only as un-versioned
`:latest`, registry-absent, and some diverged across hosts (same name,
different image ID per box):

| Image:tag | host(s) | image ID | note |
|---|---|---|---|
| `krabby-mast3r:latest` | b | `481571cbbb6e` (Apr-29) | local-only |
| `krabby-mast3r:latest` | s | `25592e8b33bd` (Apr-12) | **diverged from b** |
| `krabby-mast3r-base:latest` | s | `d57049231d13` | local-only |
| `krabby-slam3r:latest` | d | `d95c509577ba` | family not in registry |
| `krabby-vggt:latest` | d | `6adb513a80f3` | family not in registry |
| `krabby-011-scene-reconstruction-cuda:latest` | s / d | `97b863a4` / `a845f0ac` | **diverged across hosts** |
| `krabby-011-scene-reconstruction:latest` | s / d | `d46a38d2` / `49e6b45c` | **diverged across hosts** |

They are irreproducible (registry-absent, un-versioned, some 2 months
old). A prune or a name-colliding build erases them with no remote
copy. mast3r/slam3r/vggt as families live nowhere but these scattered
locals.

## Problem

Preserve these before any cleanup, and give them a durable home in the
registry with explicit version tags — picking the canonical image
where copies diverged.

## Design

### Approach

1. `docker save` each local-only / diverged image off its host to
   persistent storage (the rescue — do this **before** any prune).
2. For each family, pick the canonical image (newest functional, or
   operator call where diverged), tag it `j.pski.org:5000/krabby-<fam>:<ver>`,
   and push. Never `:latest`-only.
3. Record provenance (source host + original image ID + date) in the
   family's `images/<fam>/NOTES.md`.
4. Where a family lacks a committed Dockerfile, capture it (links to
   STO-SCN-155's reproducibility approach).

### Changes

| File | Change |
|------|--------|
| `images/mast3r/`, `images/slam3r/`, `images/vggt/` NOTES | record rescued provenance + registry tag |
| (registry) | versioned tags for mast3r / slam3r / vggt / 011-scene-reconstruction |

## Definition of Done

- [ ] Every local-only / diverged image is `docker save`d to persistent storage.
- [ ] mast3r, slam3r, vggt, 011-scene-reconstruction present in the registry with versioned tags.
- [ ] Diverged copies reconciled to a chosen canonical (operator-confirmed where it matters).
- [ ] Provenance (source host + image ID + date) recorded per family.

## Out of scope

- Making each family rebuildable from source — reproducibility is STO-SCN-155's pattern; this story preserves what exists now.
- Deciding whether mast3r/slam3r/vggt stay in the active pipeline (they're fallbacks; matcha is primary) — preservation is unconditional regardless.
