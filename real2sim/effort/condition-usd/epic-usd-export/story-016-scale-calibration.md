---
xid: STO-SCN-016
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-03
depends-on: [STO-SCN-144]
bd-id: krabby-0rw
priority: 1
title: T2.E1 — Metric Scale Datum (control-distance primary, DA3 prior/gate) ★ BLOCKER
assignee: scout
---

# T2.E1 — Metric Scale Datum (control-distance primary, DA3 prior/gate) ★ BLOCKER

## Summary

Fix the **absolute metric scale** of a reconstruction as **one scalar on the solve similarity
gauge (the datum)**, so every downstream mesh, USD export, and camera-relative cull primitive
inherits meters from a single source. Scale is recovered from a **hand-measured control distance**
(two-view triangulation, STO-SCN-144); the monocular **DA3 `is_metric`/`scale_factor` estimate is a
weak prior + gross-error gate only**, never the calibration. Critical-path blocker for T2.

## Context

**Source:** PLAN E1 + manager top-3-risks #2 (the single biggest T2 unknown). Reframed 2026-06-16
(operator) from a per-mesh `scale-calibrate@0` transform → a **datum-level** calibration, after the
STO-SCN-016 investigation + the `knowledge@olai` 3d-reconstruction metric-scale guidance
(corpus: `personal.research/3d-reconstruction/metric-scale-calibration/index.md`).

**Why the reframe:** scale is a gauge property (1 of the 7 SfM similarity DOF), not a mesh
property. Baking it per-mesh would (a) desync the cameras from the meshes (the camera-relative
metric cull primitives, STO-SCN-145, would not line up), and (b) re-apply it N times. One scalar on
the datum, applied to cameras + every mesh, is correct and is what every incumbent does.

## Decisions

| ID | Decision | Rationale / source |
|----|----------|--------------------|
| D1 | Scale is **unobservable from images alone** — SfM fixes only a 7-DoF similarity gauge; **external metric info is mandatory**. | Foundational (Umeyama 1991; COLMAP/Metashape/RealityCapture all require external control). |
| D2 | **DA3 `is_metric`/`scale_factor` is NOT a calibration** — a weak prior at best. `is_metric` is a mode flag with **zero confidence signal**; measured 3.35 / 3.48 / **11.22** across three scouts of the *same* 001-patio gauge. | Scout audit 2026-06-16; `knowledge/da3-gsply-normalized-frame.md` § "Metric scale". |
| D3 | **Primary calibration = hand-measured control distance**, `s = D_measured / d_solve(p1,p2)`, applied as **one scalar on the solve Sim3 (datum-level)**. | OLAI metric-scale guidance; incumbent standard (Metashape scale bars / RC Define-Distance). |
| D4 | **DA3 demoted to prior + gate + fallback:** (a) initialize, (b) gross-error gate — flag when `s_control / median(s_DA3)` is outside **~1.5×**, (c) approximate scale where no control distance exists. Aggregate DA3 as a **robust median in log-scale** across scouts. | OLAI guidance; the 11.22 outlier dies under the median + trips the gate. |
| D5 | Capture **≥2 control distances on different axes** → median + spread (residual scale-anisotropy / lens-distortion check). | Standard survey practice; OLAI guidance. |
| D6 | **Tooling = two-view triangulation** (STO-SCN-144 MEASURE mode in `match.html`). Rejected single-click splat raycast (fuzzy, geometry-dependent). | Operator-approved 2026-06-16; reuses STO-SCN-105 machinery. |
| D7 | Metric scale is the **1-DOF scale piece** of the full **gauge-fixing datum** (origin + orientation + scale). Up/gravity = `gauge_up` (STO-SCN-105); azimuth + origin + the metric frame feed the camera-relative primitive cull (STO-SCN-145). | Design session 2026-06-16. |

## Design

### Pipeline integration (v4 content-addressed store)
- **Datum scalar, not a per-mesh node.** Apply `s` where the gauge is recorded (the solve
  `cameras.json` + the orient/`oriented.json` gauge), so cameras AND every materialized mesh share
  one metric factor. (This replaces the original `scale-calibrate@0` per-mesh-transform plan.)
- **Provenance in the gauge metadata:** the measured `D`, the triangulated endpoints, the photos,
  the per-distance `s` + median + spread, and the DA3-gate verdict (`s_control` vs `median(s_DA3)`).
- **Backwards-compat:** introducing a datum scalar changes the gauge's content identity → it is a
  **new gauge node** (additive); existing uncalibrated nodes (`s=1`) are untouched and re-key
  nothing. Canonical rule: **STO-SCN-136 § "Backwards compatibility — store identity"**.
- **Future capture protocol:** include a reference object / known dimension in-scene (codified in
  HUG-SCN-004 capture lessons) so control distances are always available.

### Realization
- **STO-SCN-144** delivers the control-distance measurement (the `s`). This story consumes that `s`
  and wires it into the datum + records provenance + runs the DA3 gate.

## Definition of Done
- [ ] Strategy documented (this story's Decisions table is the canonical record). ✅ (D1–D7)
- [ ] Control distance measured for ≥1 scene via STO-SCN-144; `s` recovered.
- [ ] `s` applied as **one datum-level scalar** on the solve gauge (cameras + meshes inherit it);
      provenance + DA3-gate verdict recorded in gauge metadata.
- [ ] Verified by spawning a known-size primitive in IsaacSim and comparing (T2 acceptance).
- [ ] DA3 prior/gate wired: `median(scale_factor)` per scene; flag when outside ~1.5× of `s_control`.
- [ ] Backwards-compat: uncalibrated (`s=1`) gauge nodes unchanged (additive new node, no re-key).
- [ ] Future-capture protocol defined (reference object / known dimension in-scene).

## Implementation Notes

### Built 2026-06-16 — compute + record path (the seam to the datum)
- **`real2sim/metric_scale.py`** (tested core) + **`real2sim/calibrate_datum.py`** (new): reads a
  `match.html` MEASURE export (STO-SCN-144), **re-triangulates** each endpoint from the raw
  correspondences (does not trust the browser), aggregates to one `s` (log-median + spread),
  runs the DA3 gross-error gate, and writes a **datum-scale record** (`s` + provenance + verdict).
  CLI demoed end-to-end (synthetic: recovered `s=2.0`, flagged `da3_scouts_disagree`).
  Tests: `tests/test_calibrate_datum.py` 5/5 + `tests/test_metric_scale.py` 12/12 green.
- **Decisions D1–D7** above are the canonical record (reframe ratified).

### Built 2026-06-16 — `apply_to_gauge` (the datum-level apply, store-safe)
- **`calibrate_datum.apply_to_gauge(gauge_dir, scale, datum_frame, provenance)`** writes the metric
  scale as an **additive `datum.json` sidecar** in the gauge/orient dir: `p_meters = scale ·
  p_solve_gauge` + the camera datum frame + provenance. Store-safe by construction — it does **not**
  re-ground meshes or rewrite cameras.json/oriented.json, so **no materialized mesh identity is
  re-keyed** (STO-SCN-136 backwards-compat); refuses to clobber an existing sidecar (T-018).
  Tested (`tests/test_calibrate_datum.py`: sidecar write + clobber-refusal).

### Remaining (operator-gated — T-020 / T-026)
- **The real `s` value** needs a real measurement (a human knows a real-world distance) — the code
  path is built + tested with synthetic `s`; only the actual number is operator-supplied.
- **Consumer-side reads:** USD export (STO-SCN-017) reads `datum.json` to emit metric USD; the
  primitive-cull (STO-SCN-145) authors in those meters. (Sidecar producer done; consumers wire next.)
- **OPERATOR ACTION (calibrates scene 001):** open `match.html` MEASURE mode, measure ≥1 known real
  distance, export → `calibrate_datum.py` → `apply_to_gauge`. Everything downstream is built and waits
  only on that number + the T-020 visual confirm.

## Journal Notes
MAtCha's per-chart deformation MLP can re-scale geometry differently across runs (per-region depth
ambiguity), so submap meshes can drift in scale even with agreed camera positions — anchor on the
unified SfM sparse points / a shared-frame scale-alignment step. Reference-localization test saw a
1.6% scale difference (Procrustes 1.0156) between SfM frames. (Confirms scale must be pinned
externally + at the datum, not re-derived per mesh.)
_Sources: notes 2026-05-01T174650-submap-based-mesh-fusion, 2026-05-06T100000-auto-localized-reference-cameras._

## Handoff Notes
**Root cause** (manager audit 2026-05-06): unsolved across ALL captures because no reference objects
were in scene — a capture-side miss codified in **HUG-SCN-004**. The control-distance tool
(STO-SCN-144) is the retroactive fix for existing scenes; the future-capture protocol prevents
recurrence. **Ownership moved principal → scout (operator, 2026-06-16).**

---
_Imported from legacy beads `m11-u3l` (M11 DAG re-import, 2026-06-03). Reframed datum-level 2026-06-16._
