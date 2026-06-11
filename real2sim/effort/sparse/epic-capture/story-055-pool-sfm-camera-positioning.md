---
xid: STO-SCN-055
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-45i
shipped: 2026-06-10
tasks: 1
complete: 1
---

# Phase: pool SfM — camera positioning for coverage curation

> Retroactive phase documentation (operator directive 2026-06-10).
> Recipe section: `real2sim/RECIPES.md` § Phase catalog →
> "Camera positioning (pool SfM)".

## What we did

Posed a candidate frame pool (≤300 frames) with a MASt3R-SfM
`--sfm_only` solve whose ONLY purpose is preprocessing: giving every
candidate frame a 3D position so the reconstruction subset can be
chosen by spatial coverage instead of timestamp. "The dtu flow":
dtu-bicycle (194-frame pool → 12 curated), 005-meadow
(`preproc-02-pool-sfm` over pool-sharp-200), 013-basement
(`preproc-03-pool-sfm`: 200/200 poses, ~9 GB VRAM, RTX 4080).

## Where the code is

- `real2sim/batched_sfm.py` — `solve` subcommand (single-chunk
  degenerate case of the spine machinery; STO-SCN-049). `chunk` with
  `--chunk-size ≥ pool` mints the chunk layout.
- Container: `j.pski.org:5000/krabby-matcha:*-selfcontained`,
  `train.py --sfm_config unposed --sfm_only`.

## How (013 as the worked example)

1. `batched_sfm.py chunk --pool <candidate-pool>/data --out
   input/preproc-NN-pool-sfm --chunk-size <pool-n> --overlap 3`
   (portable relative symlinks — survive any container mount).
2. On a GPU host: `batched_sfm.py solve --spine …/preproc-NN-pool-sfm
   --chunk 1 --image krabby-matcha:0.2.1-selfcontained`.
3. Output `chunk-01/out/mast3r_sfm/cameras.json` feeds camera_viewer
   (STO-SCN-001) for operator curation → `selected_frames.json`.

## Gotchas (paid for)

- Pool must be ≤300 frames (measured 16 GB solve ceiling).
- LFS-pointer guard before solving: a pointer file is ~130 bytes and
  the tool will not tell you it read garbage.
- Solve output lands root-owned on the host clone — gather-hygiene
  sequence applies (RECIPES.md § fleet execution).

## Definition of Done

- [x] Phase documented here + RECIPES.md section.
