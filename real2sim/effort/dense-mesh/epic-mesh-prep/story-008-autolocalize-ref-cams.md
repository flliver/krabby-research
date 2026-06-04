---
xid: STO-SCN-008
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-wzk
priority: 2
shipped: 2026-06-03
tasks: 4
complete: 4
title: T1.C-AutoLocalize — Auto-localize Reference Cameras via SfM-extend
assignee: krabby
---

# T1.C-AutoLocalize — Auto-localize Reference Cameras via SfM-extend

## Summary

`localize_reference_image.py`: full pipeline for SfM-localizing a reference image against an existing reconstruction.

## Context

`localize_reference_image.py`: full pipeline for SfM-localizing a reference image against an existing reconstruction.

1. Convert reference PNG → faux-RGB JPG (greyscale becomes 3-channel-replicated).
2. Stage sandbox dir with 12 source frames + 1 ref JPG.
3. Run `python train.py --sfm_only --image_idx 0..12` in matcha-build container — ~104s on RTX 5080.
4. Pull `cameras.json`; Procrustes-align new 12-cam centers to original 12 (Umeyama with scale).
5. Apply similarity transform to 13th cam → reference pose in original SfM frame.
6. Apply `world_orient` → world frame.
7. Convert to OpenCV-convention quat + position.
8. Upsert into `comparison_views.json` with `purpose=reference-match`, `auto_localized=true`.

Result on bicycle: scale 1.0156, sub-cm Procrustes residuals across 12 anchors. Auto camera within 0.74 m / 4.55° of manual placement. Greyscale rendering didn't trip MASt3R (channel-replication sufficient).

Evidence: commit `4bf02c5`; journal note `journal/.../matcha-quality/notes/2026-05-06T100000-auto-localized-reference-cameras.md` Phase 1 section.

## Definition of Done

- [x] Script runs end-to-end on any single PNG + existing 12-cam SfM
- [x] Procrustes residuals sub-cm on shared cameras
- [x] Result upserted to comparison_views.json with auto_localized=true
- [x] Reproducible on adaptive_tetra reference image (use case 2)


## Journal Notes

Implemented 2026-05-06 as `workspace/localize_reference_image.py` (404 lines, "SfM-extend" = variant 1 of three). Pipeline: reference PNG → faux-RGB JPG (greyscale channel-replicated), stage a tbeeprz sandbox of 12 relative-symlinked source frames + 1 ref JPG (`_DSC9999_ref.JPG`, sorts last → idx 12), run `train.py --sfm_only --image_idx 0..12` (~104 s, RC=0), pull `cameras.json`, Umeyama/Procrustes-align the 12 new centers to the original 12, apply that similarity to the 13th pose, then `world_orient`, upsert into `comparison_views.json` (`auto_localized=true, localization_method=mast3r_sfm_extend`). Bicycle TSDF result: Procrustes scale 1.0156, residuals max 1.3 cm / mean 0.4 cm — but a 0.745 m / 4.55° delta vs the manual `cam_ref`; SfM math worked, but shared-focal SfM (all 13 cams → focal 484.56 px) couldn't match the paper's wider-FOV/higher vantage. Greyscale didn't trip MASt3R. Three choices: relative (not absolute) symlinks (host mount differs in-container — caught after a FileNotFoundError run), sibling sandbox to keep the scene's `cameras.json` immutable, and `--image_idx` not `--n_images` (avoids silently dropping the ref). PnP localization (variant 2, decoupled focal) is the documented next step.
_Sources: notes 2026-05-06T100000-auto-localized-reference-cameras, 2026-05-06T101958-auto-positioning-…; entry 2026-05-06T101958-reference-camera-auto-positioning._


## Handoff Notes

Manager memo (establish-manager-role-2026-05-06.md) flags this as "nearly done": the 2026-05-04 planning pivot inserted "validate against MAtCha paper's reference quality" *before* USD export, and that reference-validation work shipped in the 2026-05-06 release. **Still open:** the A/B comparison cameras (`compare_01/02/03`) for the bicycle scene need hand-placement in Blender.

---
_Imported from legacy beads `m11-5ef` (M11 DAG re-import, 2026-06-03)._
