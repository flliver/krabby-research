---
xid: STO-SCN-043
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
date: 2026-06-09
depends-on: []
bd-id: krabby-srz
title: 007-kubota — normalize preproc + first preproc-consuming run
priority: 1
assignee: krabby
---

# 007-kubota — normalize preproc + first preproc-consuming runner execution

## Summary

Reconstruct `007-kubota` (9 iPhone MPO photos) through the full two-transform
config-driven chain: `preproc-01-normalize` (new spec-driven tool
`real2sim/normalize_photos.py`) → `transform-01-matcha` via the runner. Tests
the resolution lever surfaced by STO-SCN-042: native 5712 px cost 006 11,994 MiB
peak VRAM; this run consumes 2048 px normalized inputs.

## Steps taken

1. 2026-06-09: Authored `real2sim/normalize_photos.py` — spec-driven,
   results-emitting preproc transform (primary-image decode drops MPO aux,
   EXIF-orientation bake, LANCZOS downscale to `long_edge`, plain JPEG q95).
   Research commit; registry integration into run_transform.py = STO-SCN-040.
2. 2026-06-09: Authored + executed `007-kubota/input/preproc-01-normalize`
   (9 photos 5712×4284 MPO → 2048×1536 JPEG, 4 s, measured results.json with
   per-file sha256). Store commit 9f04cb5.
3. 2026-06-09: Authored `run-9-strong-2k` matcha spec (locked-default recipe,
   inputs = preproc data) and queued via run_transform.py on tbeeprz.

## Definition of Done

- [x] Normalize tool source-controlled + spec/results contract honored
- [x] Preproc executed with measured provenance (HUG-KRB-002 — no hand steps)
- [ ] Matcha run `status: success`; VRAM/duration delta vs 006 recorded
- [ ] Outputs pushed; renders to operator Dropbox (established pattern)
- [ ] Verdict: adopt normalize preproc as standard for photo captures? (feeds 040 registry)
