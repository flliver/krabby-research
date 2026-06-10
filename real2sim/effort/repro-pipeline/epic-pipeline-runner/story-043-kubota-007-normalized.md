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
- [x] Matcha run `status: success`; VRAM/duration delta vs 006 recorded (below)
- [x] Outputs pushed (`d37207f`, 1.6 GB); render in operator Dropbox
- [ ] Verdict: adopt normalize preproc as standard for photo captures? (feeds 040 registry)

## Results (2026-06-09)

| | 006 (native 5712px) | **007 (normalized 2048px)** | delta |
|---|---|---|---|
| status | success | success | — |
| duration | 882 s (8 photos) | 830 s (9 photos) | ~−15%/photo |
| peak VRAM | 11,994 MiB | **8,640 MiB** | **−28%** |
| mesh | 16.1 M verts | 11.2 M verts | −30% |

Mesh quality (render inspection): the 007 band is **more contiguous** than 006's
(fewer mid-band fractures), with the usual far-field floaters. Vegetation-heavy
scene again; same caveats.

**Recommendation: adopt `preproc-01-normalize` (2048 px) as the standard first
transform for photo-set captures.** −28% VRAM restores headroom for larger frame
counts (the bigger quality lever per HUG-SCN-004), with no observed quality cost
at this scene class. Feeds the STO-SCN-040 registry as the default photo-ingest
chain: normalize → matcha. Remaining open: operator verdict on adopting this
default (last DoD item).
