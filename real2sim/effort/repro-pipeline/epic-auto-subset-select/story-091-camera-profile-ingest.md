---
xid: STO-SCN-091
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-13
depends-on: []
bd-id: krabby-y9m
assignee: krabby
---

# Camera profile at ingest (EXIF/capture-mode → camera model)

## Summary

At ingest, read the source's camera + capture mode (EXIF / known profile) and record the
camera model the solver should use — `SIMPLE_RADIAL_FISHEYE` for fisheye, `OPENCV` for
rectilinear/dewarped — instead of inferring distortion from pixels.

## Context

Conclusion #3 of the design story (STO-SCN-096): the camera model is a property of the
camera/capture mode, not the scene. EXIF on the DJI videos is identical across captures
and carries no dewarp/FOV tag, so the model comes from the *known capture mode* (a small
per-camera profile), with EXIF (encoder, resolution) as corroboration.

## Problem

The pose stage (STO-SCN-093) needs the right camera model up front. A wrong model fails
the solve outright (corpus: PINHOLE/OPENCV fail on strong DJI fisheye; dewarped footage
often fails entirely). We must decide it from reliable metadata, not noisy inference.

## Design

### Approach

A capture-profile lookup keyed on camera + mode (extend the corpus `capture-profiles`
shape): fisheye→`SIMPLE_RADIAL_FISHEYE`, dewarped→`OPENCV`. Read EXIF for the camera/mode
hints; fall back to an explicit operator-set profile when the mode isn't recorded (DJI
doesn't tag dewarp). Emit the chosen model + a reconstruction-risk flag for dewarped input.

### Changes

| File | Change |
|------|--------|
| ingest stage | read EXIF + capture profile → camera_model + risk flag |
| capture-profile registry | per-camera/mode → model mapping |

## Definition of Done

- [ ] Ingest emits `camera_model` (+ dewarp risk flag) from profile/EXIF, no pixel inference.
- [ ] Unknown/unrecorded mode → explicit profile required (fails loud, not a guess).
- [ ] Tests on the 001 (fisheye) and 002 (dewarped) captures.

## Implementation Notes

**Registry shape.** A small versioned table keyed `{make, model, capture_mode} →
{colmap_camera_model, params_hint, dewarp_risk}`. Seed from the two known DJI profiles in
the corpus: DJI Action-class **fisheye** → `SIMPLE_RADIAL_FISHEYE`; **in-camera-dewarped**
→ `OPENCV` with `dewarp_risk: high` (corpus: dewarped footage often fails reconstruction
outright, even under OPENCV). Live alongside the existing `capture-profiles` shape; this is
a per-camera property, not per-scene.

**EXIF read.** Prefer the dependency-free route already in the tree —
`colmap_posed.image_dims` parses JPEG SOF / PNG IHDR for dimensions with no deps; extend
the same spirit for make/model/encoder. Use `exiftool` (if on PATH) or Pillow `_getexif`
for richer tags. EXIF gives make/model/encoder/resolution as **corroboration** — it does
**not** carry a dewarp/FOV tag on DJI, which is exactly why the capture-mode profile (not
EXIF alone) decides the model.

**Output + failure.** Write `camera_model` + `dewarp_risk` into the ingest manifest the
pose stage (STO-SCN-093) reads. On an unknown `{make,model,mode}` combination, **fail loud
and require an explicit operator-set profile** — never silently default to PINHOLE/OPENCV
(that's the wrong-model-fails-the-solve trap from conclusion #3).

**Test data.** Scenes 001 (fisheye) and 002 (dewarped) sample frames are already in the
store. Assert 001 → `SIMPLE_RADIAL_FISHEYE`, 002 → `OPENCV` + `dewarp_risk: high`, and an
unprofiled camera → loud failure.

**Size:** S — a lookup + EXIF read + manifest field. The judgment (which model) is already
made in conclusion #3; this story just operationalizes it.

## Out of scope

- Per-scene geometric lens detection (rejected — see STO-SCN-096 conclusion #3).
