# Capture lessons — 003 firepit fisheye

## What we shot

- DJI Action 3, **4K @ 60 fps regular video** (not hyperlapse), 5:31 duration, 19,842 frames
- Native 155° fisheye
- Outdoor firepit area
- 4.3 GB source video file

## What worked

- **The firepit (the central object) is recognizable in the reconstruction** — the foreground object of interest got captured well
- Stable camera motion across the 5:31 duration; no obvious motion-blur artifacts in the reconstruction

## What didn't work

- **Massive overcapture for MAtCha's actual input needs.** MAtCha used 12 frames out of 19,842 — a **0.06% sampling rate**. The other 99.94% of frames were transferred over WAN and stored on bbeeprz for nothing.
- **Same background-noise problem as scene 001** — 155° FOV in an outdoor setting picks up tree-lines, distant ground, sky regions. None of that is part of the M11 deliverable.
- **No reference object in the scene for scale calibration.**
- **Camera path ambiguity** — without seeing where the operator stood, it's hard to interpret what "in scope" means for the mesh. The firepit is clearly central, but the boundary of "the captured scene" is unclear.

## Lessons for the next time we capture this scene

1. **4K @ 60 fps is overkill.** Use the validated 2.7K @ 30fps profile from scene 004 — same MAtCha output quality, 1/4 the file size.
2. **Capture duration of 5:31 is overkill.** ~3 min would have given comparable viewpoint coverage. Aim for **~2-4 min** so manual frame selection is feasible (looking through 60 fps × 5:31 = 19,842 frames is impractical even for a human).
3. **Same reference-object recommendation as scene 001.** Need a known-size object near the firepit for scale.
4. **Same QR-code boundary recommendation.** Mark "in-scope" perimeter so post-processing can cull beyond it.
5. **Camera-path pattern**: the central foreground object (firepit) deserves a deliberate orbit. We got reasonable orbit-coverage by accident, but it's worth making it intentional next time — explicit orbits give clean datasets, which is exactly what learned methods like MAtCha want.

## Suggestions for re-shoot (if we're going back)

- DJI Action 3, native fisheye, 2.7K @ 30 fps locked exposure/WB
- Capture pattern:
  - 1× orbit around the firepit at standing height
  - 1× orbit at sitting height (camera lower, looking down at firepit area)
  - 1× short walk along approach path (so the mesh includes "how to get to the firepit" — useful for hexapod path-planning)
- Total duration: 2-3 min
- Reference object near firepit
- 4 QR codes marking the "scene boundary" if QR-cull ships
