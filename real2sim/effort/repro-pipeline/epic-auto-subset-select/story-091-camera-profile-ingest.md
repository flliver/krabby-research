---
xid: STO-SCN-091
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-13
depends-on: []
bd-id: krabby-y9m
assignee: krabby
tasks: 3
complete: 3
---

# Camera profile at ingest (EXIF/capture-mode → camera model)

## Summary

At ingest, read the source's camera + capture mode (EXIF / known profile) and record the
camera model the solver should use — `SIMPLE_RADIAL_FISHEYE` for fisheye; **dewarped is
COLMAP-incompatible under any model** (route to SLAM/feed-forward) — instead of inferring
distortion from pixels.

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

A capture-profile lookup keyed on camera + mode: fisheye→`SIMPLE_RADIAL_FISHEYE`,
dewarped→COLMAP-incompatible (`colmap_compatible: false`, `dewarp_dead_end: true`). Read
EXIF for camera-identity corroboration; the mode itself is declared per scene (DJI doesn't
tag dewarp). Emit the chosen model + compatibility flags. Unknown camera+mode → fail loud.

### Changes

| File | Change |
|------|--------|
| ingest stage | read EXIF + capture profile → camera_model + risk flag |
| capture-profile registry | per-camera/mode → model mapping |

## Definition of Done

- [x] Ingest emits `camera_model` (+ compatibility flags) from profile/EXIF, no pixel
      inference. (`v4exec.py cmd_ingest` `resolve-capture-profile` node → scene-store
      `images/capture-profile/<id>/capture-profile.json` + metadata.)
- [x] Unknown camera+mode / missing mode → fails loud (`ProfileError`); no guessed model.
- [x] Resolver tests on the 001 (fisheye) and 002 (dewarped) semantics + ingest wiring
      (11 tests). **Verified on real data 2026-06-13:** `cmd_ingest 003-firepit` (clean-NOOP,
      `capture.json` declared) wrote `images/capture-profile/<id>/capture-profile.json` =
      `SIMPLE_RADIAL_FISHEYE`, `colmap_compatible: true`; idempotent NOOP on re-run; dewarped
      branch resolves to `colmap_camera_model: null` / `dewarp_dead_end: true`.

## Implementation Notes (as built, 2026-06-13)

**Registry** (`real2sim/capture_profiles.json`, schema 1). Profiles keyed `{make, model,
mode}` → `{colmap_camera_model, single_camera, colmap_compatible, dewarp_dead_end, fov_deg,
notes, source}`. Seeded from HUG-SCN-004 + the 001/003/004 `CAPTURE-LESSONS.md` + the
002-dewarped dead-end:
- DJI Action 3 **fisheye** → `SIMPLE_RADIAL_FISHEYE`, `colmap_compatible: true`.
- DJI Action 3 **dewarped** → `colmap_camera_model: null`, `colmap_compatible: false`,
  `dewarp_dead_end: true` (HUG-SCN-004: dewarped does not reconstruct in COLMAP under any
  model — route to SLAM/feed-forward; this replaced the design's earlier "OPENCV").

**Resolver** (`real2sim/capture_profile.py`). Pure/importable. `resolve(make, model, mode)`
matches case-insensitively and raises `ProfileError` (fail loud) on unknown camera+mode or
missing `mode`. EXIF (`read_exif`) is best-effort make/model **corroboration only** (Pillow
→ `exiftool` → `{}`); the **mode is not in EXIF** and is declared per scene. CLI for manual
checks.

**Ingest wiring** (`v4exec.py cmd_ingest`, graph option B). New graph node
`resolve-capture-profile` (`tasks/resolve-capture-profile.json`,
`graphs/ingest-scene.json`: `pool → capture-profile → solve`). Declaration precedence:
`--capture-mode/--camera-make/--camera-model` > `<scene>/capture.json`. Identity = hash of
the declaration; written set-if-unset to `images/capture-profile/<id>/capture-profile.json`
+ metadata via the store writer (HUG-SCN-005 #11). **No declaration → SKIP** (today's
mast3r-sfm solve doesn't consume the model; only STO-SCN-093 dispatch will) so existing
scenes still re-ingest; a **present-but-unresolvable** declaration fails loud.

**Tests** (`tests/test_capture_profile.py`, `tests/test_capture_profile_ingest.py`, 11
passing): resolver semantics (001 fisheye → `SIMPLE_RADIAL_FISHEYE`; 002 dewarped →
COLMAP-dead-end; unknown/missing-mode → loud), graph topology, and the store-write
primitives.

**Boundary with STO-SCN-093.** Hyperlapse-vs-video cadence (COLMAP-sequential fails on
hyperlapse even with the right model — HUG-SCN-004) is a *solver-routing* concern, not a
camera-model one; it lives in 093's dispatch, which consumes this profile.

## Out of scope

- Per-scene geometric lens detection (rejected — see STO-SCN-096 conclusion #3).
