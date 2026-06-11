---
xid: STO-SCN-060
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-q8c
---

# krabby-da3 image + 006-kubota pilot reconstruction

## Summary

A new container image `krabby-da3` (DA3 + gsplat gaussian head,
registry-distributed) and the first DA3 reconstruction in the store:
006-kubota, same input frames as the matcha runs, stored as
`pipeline-da3/run-<r>/transform-01-da3` with spec + measured results.

## Context

Parent: [EPI-SCN-FEEDFORWARD-RECON](./epic.md). Discovery record:
STO-SCN-059. 006-kubota is the designated test case (operator,
2026-06-10). New image rather than extending krabby-matcha: different
framework stack (DA3+gsplat vs MAtCha's pinned torch/pytorch3d tree),
and the 40.5 GB matcha image should not grow a second personality
(T-003: one image per toolchain).

## Design

- `images/da3/Dockerfile`: cuda base + torch>=2 + xformers + DA3
  (pinned commit) + pinned gsplat; checkpoint baked or HF-cached.
  Push to `j.pski.org:5000/krabby-da3:<tag>` (push-on-build).
- Run on d (build host; 16 GB RTX 4080).
- Inputs: 006-kubota's exact matcha-run frames (comparability).
- Outputs into the store run: depth maps, poses, gaussian PLY, GLB
  export; spec records model checkpoint + flags; results measured.

## Definition of Done

- [ ] Image builds, pushed to registry.
- [ ] 006 reconstruction completes; outputs in
      `006-kubota/pipeline-da3/run-…/transform-01-da3/data/`.
- [ ] Honest quality note vs the matcha runs (holes? splat quality?)
      + license tier recorded in the run spec.
- [ ] RECIPES.md phase catalog updated if a new phase shape emerged.
