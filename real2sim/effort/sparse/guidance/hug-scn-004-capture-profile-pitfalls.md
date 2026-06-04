---
xid: HUG-SCN-004
kind: hug
effort: scn
status: active
date: 2026-06-03
author: krabby handoff 2026-04-29
bd-id: krabby-4pm
title: Validated capture profile + COLMAP/SLAM capture pitfalls
---

# Validated capture profile + COLMAP/SLAM capture pitfalls

## Context
Capture-side findings accumulated across M11 scenes (per-scene records in `experiments/<scene>/CAPTURE-LESSONS.md`).

## Direction
- Validated profile: **2.7K @ 30fps, locked exposure/WB, stable motion**. Budget ~15–20 min MASt3R-SLAM processing per minute of 2.7K video on RTX 5080.
- DJI Action 3 native fisheye (155° FOV) needs `SIMPLE_RADIAL_FISHEYE` for COLMAP. **Dewarped** video does NOT work in COLMAP (any model).
- Hyperlapse: handled by SLAM3R / MASt3R-SLAM; COLMAP sequential matcher fails on it.
- VGGT not viable on 16 GB VRAM (needs 40+ for >50 frames).
- **Always place a known-size reference object in scene** — its absence is the root cause of the unsolved scale-calibration blocker (STO-SCN-016). Avoid 4K@60fps overcapture and fisheye background pollution.

_Source: krabby/archive/handoff-2026-04-29-1347.md._
