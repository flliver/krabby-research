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

- [x] Image builds, pushed to registry (`krabby-da3:0.1`, 43.5 GB,
      checkpoint baked; digest 6a77622…).
- [x] 006 reconstruction completes; outputs in
      `006-kubota/pipeline-da3/run-8-giant/transform-01-da3/data/`.
- [x] Honest quality note (below) + license tier in the run spec.
- [x] RECIPES.md phase catalog: new pipeline section added.

## Run record (2026-06-10, dbeeprz RTX 4080)

- **21 s wall / 11.4 s inference+export / 11.0 GiB peak VRAM** for
  8 views. (MAtCha on the same scene: 673 s train alone.)
- Outputs: `gs_ply/0000.ply` (3D gaussians), `scene.glb` (1M-point
  conf-thresholded cloud), COLMAP bins (poses — directly consumable
  by our tooling), full `exports/npz`, per-view depth_vis, novel-view
  `gs_video`.
- Gotchas:
  - The `auto` CLI does NOT expose `infer_gs` — gaussian export needs
    an API driver (`model.inference(..., infer_gs=True)`); ours is
    recorded in this story and in the run spec.
  - Multi-format export = dash-joined string (`glb-npz-gs_ply-colmap`).
  - First build attempt failed CORRECTLY on the torch-version assert:
    unpinned xformers pulls torch past 2.7 (same trap as the matcha
    notes). Fix: `xformers==0.0.30` from the cu128 index.

## Honest quality read (so far)

- **Depth maps: excellent.** Smooth coherent gradients, clean
  vegetation layering, no speckle (see `data/scene.jpg`).
- **Splat quality: not yet judged.** The auto gs_video trajectory
  hugs the lawn (close-up frames, uninformative). A real verdict
  needs the splats/cloud rendered from OUR saved comparison views —
  which requires aligning DA3's frame to the scene's oriented frame
  (the COLMAP poses make this tractable). That is the natural next
  story, NOT claimed here (T-002).
- **Resolution caveat: process_res 504** (default) vs MAtCha's 1.6K
  training — geometry density comparison must account for this knob.
