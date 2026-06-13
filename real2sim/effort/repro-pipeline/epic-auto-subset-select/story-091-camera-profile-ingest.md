---
xid: STO-SCN-091
parent: ./epic.md
kind: story
effort: scn
size: S
status: draft
date: 2026-06-13
depends-on: []
bd-id: krabby-y9m
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

## Out of scope

- Per-scene geometric lens detection (rejected — see STO-SCN-096 conclusion #3).
