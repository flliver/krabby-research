---
xid: STO-SCN-001
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-urz
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T0.B5 — Frame-Selection Tooling (Camera Selection Viewer)
assignee: krabby
---

# T0.B5 — Frame-Selection Tooling (Camera Selection Viewer)

## Summary

Interactive 3D viewer (`viewer.py`) for curating the best viewpoints from a large pool of candidate frames before running SfM. Reduces the candidate pool from "every video frame" to "the N most informative", which is critical for SfM scaling.

## Context

Interactive 3D viewer (`viewer.py`) for curating the best viewpoints from a large pool of candidate frames before running SfM. Reduces the candidate pool from "every video frame" to "the N most informative", which is critical for SfM scaling.

Foundational tooling for the SfM-scaling experiment; landed alongside the closure of that experiment.

Evidence: commits `a8cde37` (Start SFM-scaling experiment + capture supporting research), `465836c` (Close SfM-scaling experiment + ship Camera Selection Viewer v0), `bfe7e59` (SfM-scaling: document local mirror layout + viewer invocation).

## Definition of Done

- [x] Viewer launches and displays candidate camera frustums in 3D
- [x] User can select a subset of frames interactively
- [x] Output integrates with the MASt3R-SfM pipeline


## Journal Notes

The Camera Selection Viewer was built end-to-end on 2026-05-01 as the "Route B" viser-based WebGL 3D viewer (~900 lines across `data.py`, `filters.py`, `ui.py`, `viewer.py`, `slots.py`, `clustering.py`), chosen over the cheaper "Route A" Blender-Collections approach because curation needs continuous filtering rather than discrete buckets. It renders camera frustums + textured image planes from MASt3R-SfM `cameras.json` plus a temporal camera-path polyline, and exposes seven AND-composing filters (time range, temporal stride, spatial-cluster k-means w/ invert, distance-from-selection, look-at gizmo, pHash dedupe, picked-status). Click-to-toggle frustum selection, bulk Select/Deselect Visible, lock-picks, coverage colorize, live counters, and named slot save/load to `cameras.slots.json`; output `selected_frames.json` plugs directly into MAtCha's `--image_idx`. Two real bugs caught: a `forward_axes` sign error (MASt3R-SfM emits OpenCV +Z-forward, not OpenGL -Z) and PIL deprecation (`Image.LANCZOS`→`Image.Resampling.LANCZOS`, missing `scipy`).
_Sources: entries 2026-05-01T174652-scaling-…, 2026-05-01T222604-day-end-…; notes 2026-05-01T153502-camera-selection-ui-feasibility, 2026-05-01T203949-accomplishments-and-next-steps._


## Handoff Notes

Direction received 2026-05-01 (handoff-2026-05-01-1324.md): pursue the chart-encoding `r` knob (Option C) **plus** manual curation backed by MASt3R-SfM-derived camera poses (no full mesh required) — chosen over plain higher-res-at-12-frames (A) and full mesh-based curation (B). MASt3R-SfM scales to 350+ frames. Two viewers ran during this work: port 8082 `camera_viewer/viewer.py` (bicycle 194-frame pool → 12-frame `dtu-bicycle/selected_frames.json`) and port 8090 `rate_renders/server.py`.

---
_Imported from legacy beads `m11-2bg` (M11 DAG re-import, 2026-06-03)._
