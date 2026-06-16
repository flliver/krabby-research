---
xid: STO-SCN-145
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
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

## Definition of Done
- [x] A fixed camera-derived **metric, gravity-aligned datum** (origin/azimuth per table) computed
      from the solve, consuming STO-SCN-016's scale. **Built: `datum_frame.py` (8/8 tests).**
- [x] Boolean primitive set (SDF-based) authored in meters in that frame; masking cull keeps/drops
      verts by the combined SDF; composes with the existing culls. **Built: `sdf_primitives.py`
      (sphere/box/cylinder/halfspace + keep/subtract booleans; 10/10 tests) + `cull_mesh.py
      --primitives` (ANDs `in_prims` with the view/floor/dist/cambox masks).**
- [~] Operator authors a primitive once against the cameras and it applies to multiple meshes from
      the same solve. **Mechanism done** (primitives in the datum frame + `frame_transform` apply to
      any mesh from the same gauge); a primitive-authoring UI is a follow-up.
- [ ] **Operator-verified (T-020):** author a primitive on 001-patio, cull a real mesh, confirm in
      Rank. **← remaining; needs the operator (and a calibrated `s` for true-meters authoring).**

Note: wiring `--primitives` into a `cull-mesh@2` tunable (content identity) is the v4-store
integration step (mirrors STO-SCN-137's `cambox_expand`); the masking itself is complete + tested.

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

### Built 2026-06-16 — boolean-primitive SDF masking (DoD item 2)
- **`real2sim/sdf_primitives.py`** (new, tested 10/10): SDFs (sphere/box/cylinder/halfspace) +
  boolean combinators (keep=union via min, subtract=`max(a,−b)`); `cull_mask(verts, primitives,
  frame_transform)` keeps verts inside the combined solid. Spec is a JSON list authored in the
  datum frame (meters), optional `frame_transform` maps mesh→datum. STO-137's camera-AABB = the
  keep-box special case (test asserts the equivalence).
- **`real2sim/cull_mesh.py --primitives <json>`** — evaluates `in_prims` and ANDs it with the
  existing view/floor/dist/cambox masks (composes; drop-accounting prints the primitive count).
  Root-cause bug fixed during build (keep-union accumulator init; T-003).

### Built 2026-06-16 — v4 store integration (`cull-mesh@2`)
- **`tasks/cull-mesh.json`** bumped `@1 → @2` with a `primitives` tunable (default `null` = disabled);
  bumped (not appended to @1) because @1 has materialized nodes — prior @1 cambox/min_views nodes
  are preserved untouched (STO-SCN-136/137 pattern). **`v4exec.py cmd_cull --primitives <json>`**
  loads the spec inline → flows into the content identity → materializes `primitives.json` next to
  the node → passes it to `cull_mesh.py`. A primitive-culled mesh is a distinct, rankable,
  NOOP-on-re-run `condition/<id>` node.
- **Backwards-compat proven** (`tests/test_cull_primitives_identity.py` 7/7): primitives flow into
  identity, default-equality ({} == explicit-null) holds, distinct primitives → distinct nodes,
  and `@2` does NOT collide with / re-key the prior `@1` namespace.

### Remaining (operator-gated)
- **True-meters authoring** needs STO-SCN-016 calibrated (operator measurement) — until then
  primitives are authored in solve-gauge units (scale=1).
- **Operator T-020** (author a primitive on a real scene, cull, confirm in Rank) + an optional
  primitive-authoring UI.

_(Design captured 2026-06-16; datum frame + SDF primitive masking built + tested; store-node
tunable + operator verification remain.)_
