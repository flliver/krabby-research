---
xid: STO-SCN-144
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-16
depends-on: []
bd-id: krabby-mm8l
assignee: scout
---

# Metric control-distance tool — two-view triangulation MEASURE mode (match.html)

## Summary

A MEASURE mode in `verify_viewer/match.html` that recovers the **metric scale** of a solve
gauge from one or more **hand-measured control distances**: the operator clicks the same real
feature in **two different posed photos**, the tool triangulates the two camera rays to an
accurate 3D point (solve gauge), repeats for the second endpoint, takes the real measured length
`D`, and computes `s = D / |P2 − P1|`. This is the **primary** metric-calibration deliverable
for STO-SCN-016 (the datum-level scale strategy); the monocular DA3 estimate is only a prior/gate.

## Context

**Source:** operator, 2026-06-16, approving approach (A) two-view triangulation. Origin: the
STO-SCN-016 metric-scale investigation, which established that (1) SfM is a 7-DoF similarity gauge
with scale unobservable from images, and (2) DA3 `is_metric`/`scale_factor` is an unreliable prior
(measured 3.35 / 3.48 / **11.22** across three scouts of the *same* 001-patio gauge — see
`knowledge/da3-gsply-normalized-frame.md` § "Metric scale"). The robust path is the photogrammetric
standard: a hand-measured control distance, marked in ≥2 images.

`match.html` already provides every primitive this needs (STO-SCN-105): camera lock to a known
posed frame (`lockCamera`), screen-click → camera ray (`clickNdc`/`ndcDir`), 3D lock-points that
re-project across views (`updateDots`/`clicks3d`), and `[`/`]` photo navigation. MEASURE mode
composes these — it does **not** require the splat surface (the splat is visual context only).

## Design

### Approach (A) — two-view triangulation (operator-approved 2026-06-16)

Rejected (B) single-click splat raycast: a gaussian cloud has no clean surface (fuzzy depth) and
breaks when no geometry exists yet. Triangulation is gauge-native, geometry-free, and is how
COLMAP control points / Metashape scale bars / RealityCapture Define-Distance all work.

Per control distance (two endpoints P1, P2; per endpoint: mark in two photos):
1. Lock to photo A → click the feature → ray `r_A` from camera-A center (solve gauge).
2. The click re-projects as a crosshair into the other photos (reuse `updateDots`) so the operator
   re-clicks the **same** feature confidently.
3. `[`/`]` to photo B (different viewing angle) → click the same feature → ray `r_B`.
4. **Triangulate:** 3D endpoint = the midpoint of the shortest segment between the two skew rays
   `r_A`, `r_B` (closest-point-of-approach). Report the ray gap as a confidence/agreement metric;
   warn if the two rays are near-parallel (weak triangulation → pick photos with more baseline).
5. Repeat for P2. Enter real measured length `D` (meters). `d_solve = |P2 − P1|`; `s = D / d_solve`.

### Robustness (from the metric-scale guidance, STO-SCN-016)
- Support **≥2 control distances on different axes**; report each `s`, the robust median, and the
  spread (the spread = a residual scale-anisotropy / lens-distortion check).
- **DA3 gross-error gate:** load `median(scale_factor)` from the scene's scouts; flag the datum for
  human review when `s / median(s_DA3)` is outside ~1.5× (catches a mis-clicked pair or a
  fat-fingered length). The 11.22 outlier trips it; ~3.4 passes.

### Output
- Emit paste-able JSON: `s`, provenance (the 3D endpoints, the two photos per endpoint, the ray
  gaps, `D`), the per-distance values + median + spread, and the DA3-gate verdict.
- The scalar `s` is applied **at the datum/gauge level** (one factor on the solve Sim3) — NOT a
  per-mesh transform. Wiring `s` into the gauge/orient node is STO-SCN-016 (the strategy story).

### Changes
| File | Change |
|------|--------|
| `real2sim/verify_viewer/match.html` | new MEASURE mode (key toggle): triangulated endpoint picking across 2 photos, cross-photo reprojection guide, ray-gap confidence, multi-distance median + spread, DA3 gate, JSON export |
| `real2sim/verify_viewer/build_verify.py` | pass each scene's scout `median(scale_factor)` into `frustums.json` for the in-tool DA3 gate |

## Definition of Done
- [ ] MEASURE mode triangulates a control-distance endpoint from clicks in two posed photos
      (closest-point-of-approach), with a cross-photo reprojection guide + ray-gap confidence.
- [ ] Computes `s = D / d_solve`; supports ≥2 distances → reports per-distance `s`, median, spread.
- [ ] DA3 gate: flags when `s / median(s_DA3)` is outside ~1.5×.
- [ ] Exports the scalar + full provenance as paste-able JSON (consumed by STO-SCN-016 datum wiring).
- [ ] **Operator-verified (T-020):** operator measures a known distance on 001-patio and confirms
      the recovered `s` matches the real-world scale.

## Testing
- [ ] Synthetic: two rays with known closest-approach → triangulated point within tolerance.
- [ ] Near-parallel rays → weak-triangulation warning fires.
- [ ] `s` recovers a planted scale on a synthetic camera pair (exact).
- [ ] DA3 gate trips on a 3× discrepancy, passes within 1.5×.

## Out of scope
- Applying `s` to the gauge/meshes (STO-SCN-016). Absolute orientation/gravity-up datum
  (STO-SCN-105 `gauge_up` already supplies it). The camera-relative primitive cull (STO-SCN-145).

## Implementation Notes
_(Fill in during/after implementation.)_
