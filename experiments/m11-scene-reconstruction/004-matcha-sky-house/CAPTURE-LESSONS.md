# Capture lessons — 004 sky-house dining

## What we shot

- DJI Action 3, **2.7K @ 30 fps locked exposure / WB, stable motion** (the validated profile)
- 3:47 duration, 6,804 frames, native 155° fisheye
- Indoor / semi-outdoor sky-house dining area
- 833 MB source video (concatenated from `capture-01.mp4` + `capture-02.mp4`)

## What worked

- **The 2.7K @ 30 fps profile is the right baseline.** Smaller files than 4K, enough resolution for MAtCha's 1024px-wide downscaled input, plenty of frames for keyframe selection.
- **Locked exposure / white balance** — no per-frame brightness drift to confuse the learned point-map predictions.
- **Indoor scene = less "background noise" than the outdoor scenes 001 and 003.** The mesh stays much more contained to actual scene geometry.
- **Densest output of the three Phase A scenes** — the dining-area objects (table, chairs, walls) are clearly captured.

## What didn't work

> **"Dense in many areas, but obvious gaps in places — probably not covered."** — Jeremy, 2026-04-30 inspection

- **Coverage gaps in the mesh.** Some parts of the dining area weren't densely sampled by the operator's walk path. With 12 keyframes from 6,804 frames, we likely missed view angles needed for good triangulation in those gap regions.
- **No reference object** for scale calibration.
- **Same "no clear ground plane" / "tilted output" issue** as scenes 001 and 003 — but cross-cutting, not capture-specific.

## Lessons specific to this scene

1. **Indoor MAtCha is markedly cleaner than outdoor.** The 155° fisheye captures less "useless distance" indoors because there's no horizon / sky / distant-tree region. Keep this in mind when scoring MAtCha's quality fairly.
2. **12 keyframes from 6,804 frames means lots of capture-side opportunity is being thrown away.** This is the strongest evidence that **manual keyframe curation will improve quality** — at 6,804:12 = 567:1, even a 30-second human review of a contact sheet would surface better viewpoint coverage than even temporal sampling.
3. **Coverage gaps suggest the camera path itself missed angles.** Orbit-style or grid-walk patterns would help; "wander where it feels natural" doesn't.

## Lessons that informed Phase A

This scene was the first MAtCha success and the validation of the 2.7K @ 30 fps capture profile. The findings here drove the capture-profile recommendations in OLAI corpus `3d-reconstruction/capture-profiles`.

## Suggestions for re-shoot (if we're going back)

The capture profile is correct. **Don't change the profile** — change the path:

- Same 2.7K @ 30fps locked exposure/WB
- Capture pattern:
  - **Perimeter walk at eye level** (~30 sec)
  - **Perimeter walk at low height** (camera near hip, ~30 sec)
  - **"Crossing" walks** through the interior space (~30 sec each, multiple)
  - **Object-specific orbits** around any focal item (table, fireplace, prominent furniture)
- Total: 3-5 min
- Reference object near scene center (a placemat or coaster of known size on the dining table is fine)
- Aim for **dense overlap between successive viewpoints** — successive keyframes should share substantial common geometry to give MAtCha's chart alignment good cross-references
