---
xid: STO-SCN-006
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-9uv
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.B3 — Camera-Location Markers in Blender
---

# T1.B3 — Camera-Location Markers in Blender

## Summary

Add debug camera markers at SfM-derived poses, plus textured-frame planes positioned at each camera. Provides interpretability when inspecting reconstructions in Blender — you can see exactly where each source photo was taken from.

## Context

Add debug camera markers at SfM-derived poses, plus textured-frame planes positioned at each camera. Provides interpretability when inspecting reconstructions in Blender — you can see exactly where each source photo was taken from.

Evidence: commits `27081bf` (build Blender .blend scenes with cameras auto-injected), `0c55c42` (place source frames as textured planes at each camera), `658b289` (image-plane UV flip + Z-up axis fix).

## Definition of Done

- [x] Cameras visible as Blender objects in generated .blend files
- [x] Textured planes correctly oriented and UV-mapped
- [x] Z-up axis convention applied throughout


## Journal Notes

Implemented in `workspace/build_blender_scene.py` — a headless Blender script importing the oriented+culled mesh, placing one Camera per recovered MAtCha pose (from `cameras.json`) and a textured image plane per camera, writing `scene_culled.blend` (mesh + 12 cameras + planes). Same tool was most of the camera-picker UI already, and was later widened (schema v4) to inject all comparison/reference views: `--view-name` now selects the active scene camera, with a Procrustes anchor alignment computed once and applied per-view.
_Sources: post-processing/entries 2026-05-01T144327-…; entry 2026-05-06T101958-reference-camera-auto-positioning._

---
_Imported from legacy beads `m11-8wy` (M11 DAG re-import, 2026-06-03)._
