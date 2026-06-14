---
xid: STO-SCN-102
parent: ./epic.md
kind: story
effort: scn
status: in-progress
date: 2026-06-13
depends-on: [STO-SCN-091]
bd-id: krabby-4c5
assignee: krabby
---

# Per-camera fisheye calibration

## Summary

A one-time, per-camera-MODEL fisheye calibration (`K` + distortion `D`) baked
into the capture profile, so STO-SCN-093 can undistort fisheye → pinhole
accurately before FastMap.

## Context

101 proved FastMap accepts only `SIMPLE_PINHOLE`/`SIMPLE_RADIAL`, so our 155° DJI
fisheye scenes must be undistorted to pinhole first (web research: 155° exceeds
the Kannala-Brandt ~115° limit → undistort-with-crop is the recommended SfM
path, and the crop conveniently drops the background-pollution outer ring our
CAPTURE-LESSONS complained about). No published DJI Action 3 calibration exists,
so we calibrate the model ourselves. This is the authoritative form of
STO-SCN-091 conclusion #3 ("camera model from known metadata, not pixels");
it **feeds the undistort path in STO-SCN-093**.

## Problem

Obtain accurate, reproducible, versioned fisheye intrinsics for each supported
camera+mode, stored where 093 can read them — without per-scene guessing.

## Design

### Approach

`calibrate_camera.py` (OpenCV fisheye / Kannala-Brandt, 4 distortion params):
detect checkerboard corners across ~20–40 stills shot with the camera in its
capture mode, `cv2.fisheye.calibrate` → `K, D, rms`, write into the matching
`capture_profiles.json` entry under `calibration`. Per-MODEL (not per-unit —
SfM BA refines anyway). Cameras without a calibration fall back to
approximate-FOV intrinsics in 093 (graceful).

### Capture recipe (operator, one-time per camera+mode)

Print a checkerboard (e.g. 9×6 inner corners; measure the square edge in m).
Shoot ~20–40 stills in fisheye mode, locked focus, varied angle/distance, and
**push the board into the image corners** (where the distortion lives). Board
fully visible + sharp each frame.

### Changes

| File | Change |
|------|--------|
| `real2sim/calibrate_camera.py` | new — OpenCV fisheye calibrate → capture profile |
| `real2sim/capture_profiles.json` | gains a `calibration` block on the calibrated entry |
| `real2sim/tests/test_calibrate_camera.py` | board-parse + profile-write + fail-loud |

## Definition of Done

- [ ] `calibrate_camera.py` calibrates from a checkerboard set and writes
      `K/D/image_size/rms` into the matching capture profile (fail-loud on no profile).
- [ ] Operator shoots the DJI Action 3 fisheye set; calibration stored with a
      sane RMS (<~1 px) — **operator-verification gate (T-020)**.
- [ ] Pure paths (board parse, profile write) unit-tested.

## Out of scope

- The undistort step itself + FastMap dispatch (STO-SCN-093).
- ChArUco support (checkerboard first; ChArUco is a later robustness add).
- Per-unit calibration (per-model is sufficient; revisit only if units diverge).
