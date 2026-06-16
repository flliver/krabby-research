---
xid: STO-SCN-145
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-16
depends-on: [STO-SCN-016, STO-SCN-137]
bd-id: krabby-8w2r
assignee: scout
---

# Camera-relative metric datum + boolean-primitive cull coordinate system

## Summary

A **camera-derived, gravity-aligned, metric coordinate frame** (the datum) in which an operator can
describe **boolean cull primitives in meters** (sphere, box, cylinder, half-space, …) once, against
the *cameras* — which are the constant across the many meshes generated from a fixed solve — and
have those primitives applied to **every** mesh produced in that space. Generalizes the STO-SCN-137
camera-AABB cull from a single expanded box to a composable boolean primitive set, all expressed in
one fixed, isotropic, metric frame.

## Context

**Source:** design session, operator + scout, 2026-06-16. The pipeline generates **10s of meshes**
from one stable solve (no re-solve); the **entire camera spine** is the invariant input. Culling
must therefore be defined **relative to the cameras**, before any mesh exists, and reused across all
meshes. STO-SCN-137 proved the camera-box cull in the gravity frame; this story is the general,
metric, multi-primitive version.

This is a downstream consumer of **STO-SCN-016** (the metric datum gives meters) and extends
**STO-SCN-137** (the camera-AABB cull). It is **draft** — captured to track the design; not yet
greenlit for build.

## Problem

STO-137's cull is one gravity-aligned AABB scaled by a dimensionless fraction. To author real cull
regions ("keep a 2 m bubble around the toured path", "drop everything below this plane", "subtract
that pillar volume") the operator needs (a) a **fixed metric frame** to place shapes in, and (b) a
**composable primitive language**. The frame must be camera-derived (mesh-independent) and metric
(meters, for human authoring + IsaacSim physics).

## Design (from the 2026-06-16 discussion — to flesh out before build)

### The datum is gauge-fixing, not an invented coordinate system
Fixing a frame from cameras = pinning the SfM 7-DoF similarity gauge (origin 3, orientation 3,
scale 1). The cambox is **data expressed in the frame**, not the frame's definition.

| DOF | Choice | Source |
|-----|--------|--------|
| Orientation: up (2) | gravity = `gauge_up.up_from_poses` (⟂ camera-right axes) | STO-SCN-105 |
| Orientation: azimuth (1) | spine tangent (cam[0]→cam[N]) projected to ground = +X | this story |
| Origin (3) | camera centroid projected onto the gravity ground plane (z=height) | this story |
| Scale (1) | metric, 1 unit = 1 m | **STO-SCN-016** |

Result: `origin = ground-projected centroid; +Z = gravity; +X = spine azimuth; +Y = Z×X; 1 unit = 1 m`.
Fully determined by cameras + gravity + the metric datum — **no mesh required**, identical for every
mesh, **isotropic + metric** (a 2 m sphere is a 2 m sphere in every scene).

### "Where is 0,0,0 / 1,1,1" — the resolved answer
Keep the frame **rigid + metric + isotropic** (Option A from the discussion). Do **not** normalize
per-axis to a unit cube — the cambox is vertically thin (cameras at ~uniform height), so per-axis
normalization distorts shapes (a normalized sphere becomes a pancake). `1,1,1` is *one meter on each
axis*, a physical point — not a box corner. USD encodes this exactly as `upAxis` + `metersPerUnit`.

### Primitives as SDFs, booleans as min/max
Each primitive = a signed-distance function in the datum frame (Inigo Quilez catalog: sphere, box,
cylinder, capsule, half-space…). Booleans: union=`min`, intersection=`max`, difference=`max(a,−b)`.
- **Masking (keep/drop verts — robust, no deps):** evaluate the combined SDF sign per vertex → cull
  mask; drop triangles referencing a culled vertex (the existing `cull_mesh.py` machinery). Robust
  to non-manifold/garbage meshes (only classifies, never cuts). STO-137's AABB = the box-SDF case.
- **True CSG (cut/merge geometry — later, if needed):** OpenVDB level-set booleans for messy input
  (SDF→boolean→marching cubes), or Manifold (`manifold3d`) once meshes are conditioned/clean.

### Authoring + pipeline
- Author primitives as `{type, transform(TRS in datum), params(m)}` — programmatic JSON and/or
  USD `UsdGeom` prims (Cube/Sphere/Cylinder) placed in the gravity-aligned frame (round-trips to sim).
- A cull criterion on `cull-mesh@1` (the STO-137 task) consuming the primitive list + the datum;
  composes with the existing view/floor/dist/cambox culls; content-addressed, NOOP re-run.

## Open questions (resolve before build)
- **Datum stability across solves.** "Cameras are the constant" holds *within* a solve; across
  re-solves with different subsets the centroid/spine drift. We currently **do not re-solve**
  (operator, 2026-06-16) — confirm, or freeze the datum from one canonical solve + register later.
- Masking-only vs true CSG (which deliverables actually need cut geometry?).
- Authoring surface: JSON vs Blender/USD visual placement vs an in-tool primitive editor.

## Definition of Done (draft)
- [ ] A fixed camera-derived **metric, gravity-aligned datum** (origin/azimuth per table) computed
      from the solve, consuming STO-SCN-016's scale.
- [ ] Boolean primitive set (SDF-based) authored in meters in that frame; masking cull on
      `cull-mesh@1` keeps/drops verts by the combined SDF; composes with existing culls; NOOP re-run.
- [ ] Operator authors a primitive once against the cameras and it applies to multiple meshes from
      the same solve.
- [ ] **Operator-verified (T-020).**

## Out of scope
- The metric scale itself (STO-SCN-016) + the measurement tool (STO-SCN-144).
- Oriented (non-axis-aligned) hulls beyond what the primitive transforms express.
- True CSG cut geometry (separate follow-up; this story is masking-first).

## Implementation Notes

### Built 2026-06-16 — the datum FRAME foundation (DoD item 1, scale-independent part)
- **`real2sim/datum_frame.py`** (new, tested): `build_datum(cam_centers, up, scale, ground_z)` pins
  the gauge — `+Z` = gravity (`gauge_up`), `+X` = spine azimuth (cam[0]→cam[N] ground-projected,
  PCA fallback for loops), `+Y = Z×X`, origin = ground-projected centroid, metric `scale` from
  STO-SCN-016 — and emits `solve_to_datum` (4×4): `p_datum = scale·R·(p_solve − origin)`.
  `gauge_fix_from_poses` recovers up from the poses; `to_datum` maps points.
  Tests `tests/test_datum_frame.py` 8/8 (up→+Z, spine→+X, orthonormal RH, centroid→origin,
  ground projection, metric scale, loop PCA fallback, gauge_up recovery).

### Remaining (greenlit-gated + 016-gated)
- **Boolean-primitive SDF masking** (DoD item 2) in `cull_mesh.py` — author primitives in meters in
  the datum frame (sphere/box/cylinder/half-space SDFs, min/max booleans), keep/drop verts by the
  combined SDF sign. Consequential change to the cull tool → **awaits operator greenlight** (only
  the 144 triangulation approach is approved so far). The metric authoring needs STO-SCN-016
  calibrated (operator measurement).
- Datum origin's `ground_z` consumes the orient gauge's floor; wiring the datum into `cull-mesh@1`
  + the v4 store is the integration step after the above.

_(Design captured 2026-06-16; frame foundation built; primitive-cull body pending greenlight.)_
