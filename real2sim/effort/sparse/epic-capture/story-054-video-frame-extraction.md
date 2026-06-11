---
xid: STO-SCN-054
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-1lv
shipped: 2026-06-10
tasks: 1
complete: 1
---

# Phase: video → lossless frame pool (ffmpeg passthrough)

> Retroactive phase documentation (operator directive 2026-06-10: a
> story per processing phase). Recipe section: `real2sim/RECIPES.md`
> § Phase catalog → "Video → frame pool".

## What we did

Converted source videos into complete, lossless PNG frame pools at
`input/src/` — the canonical pool every downstream selection step
scores against. Applied to 001/002-patio (3840×2160 mp4),
003-firepit, 004-sky-house, and 013-basement (672×376 FFV1 mkv,
898 frames / 318 MB).

## Where the code is

- No script — two ffmpeg/ffprobe one-liners, canonical form in
  `real2sim/RECIPES.md` Recipe A steps 1–2.
- `real2sim/extract_frames.sh` is the LEGACY form (fps-subsampled
  JPEG for the COLMAP era) — kept for history, not the recipe.

## How

1. Probe first (`ffprobe … codec,width,height,fps,duration`).
2. `ffmpeg -i <video> -fps_mode passthrough input/src/frame_%05d.png`
   — every frame, no fps subsampling, lossless PNG.

## Why these choices

- **All frames, not sampled**: selection is a separate scored phase
  (sharp-select); sampling at extraction blinds it.
- **PNG**: lossless from lossless sources (FFV1); the pool is the
  permanent record in the store (LFS).
- **`-fps_mode passthrough`**: no duplicated/dropped frames from
  VFR handling.

## Definition of Done

- [x] Phase documented here + RECIPES.md section (the deliverable —
      the phase itself shipped with the scenes that used it).
