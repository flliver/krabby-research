---
xid: STO-SCN-155
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-16
depends-on: []
bd-id: krabby-edx8
assignee: krabby
---

# Vendor MAtCha/DA3 source customizations as committed, reproducible patches

## Summary

The krabby modifications baked into `krabby-matcha`/`krabby-da3` are
captured as version-controlled patches (or a krabby fork) so the
images are rebuildable from committed source — not from a dirty
working tree on dbeeprz.

## Context

The audit found, on dbeeprz's persistent `/home/jeremy/sc38/MAtCha`
(+ `MAtCha-v2`):

- remote = `github.com/Anttwo/MAtCha` (**upstream only — no krabby
  fork to push to**); last commit is upstream's `b119fd9` (2025-04-07).
- **28 modified files, all uncommitted** — the krabby customizations
  baked into the images: `2d-gaussian-splatting/train.py`,
  `extract_mesh*.py`, `mast3r/**`, `Depth-Anything-V2/**`,
  `matcha/pointmap/*`.

So the image-defining source changes are version-controlled **nowhere
durable** — they exist only as a dirty working tree + inside the image
layers. The images are not reproducible from committed source today.

## Problem

If that working tree is lost (disk, `git checkout`, host reset), the
exact source that produced matcha/da3 is gone and the images become
black boxes. We need the 28-file delta captured durably and a
documented path to rebuild the image from it.

## Design

### Approach

Decide the durable form (operator call): **(a)** capture as a patch
series under `images/matcha/patches/` + `images/da3/patches/` applied
at build time over a pinned upstream SHA (matches the existing
`patch_*.py` convention), or **(b)** stand up a krabby fork of MAtCha
and pin the image `git clone` to it. (a) is lighter and consistent
with how the Dockerfiles already patch upstream; (b) is cleaner if the
delta is large/structural. Recommend (a) unless the diff resists
clean patch extraction.

### Changes

| File | Change |
|------|--------|
| `images/matcha/patches/` | add extracted patch series for the 28-file MAtCha delta |
| `images/da3/patches/` | same for any DA3-side source delta |
| `images/matcha/Dockerfile`, `images/da3/Dockerfile` | pin upstream SHA + apply the patch series (or clone the fork) |
| `images/*/NOTES.md` | document upstream SHA + patch provenance |

## Definition of Done

- [ ] The 28-file MAtCha delta is captured durably (patch series committed, or fork created + pinned).
- [ ] matcha/da3 Dockerfiles pin a known upstream SHA and apply the committed delta.
- [ ] A clean rebuild from the committed recipe reproduces a functionally-equivalent image (DES-SCN-REPRO: metric equivalence, not bit-exactness).
- [ ] No image-defining source change exists only as an uncommitted working tree.

## Out of scope

- The build *recipe* (Dockerfile/patches/tools) tmpfs rescue → STO-SCN-154.
- Re-tagging/pushing rebuilt images to the fleet → handled by 157/159 once reproducibility is established.
