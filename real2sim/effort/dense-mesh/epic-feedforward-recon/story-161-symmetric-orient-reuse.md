---
xid: STO-SCN-161
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-16
depends-on: []
bd-id: krabby-qya0
priority: high
---

# Symmetric orient-floor reuse: first reconstructor (DA3 or matcha) computes the gauge, the rest reuse it

## Summary

The `orient-floor` gauge becomes **solve-scoped and model-agnostic**: whichever
reconstructor runs first (DA3 *or* matcha) computes it and stores it under the
solve; every subsequent reconstructor **reuses** it — so each model stands alone
**and** all models land in the identical frame (renders align by construction).

## Context

Operator decision (2026-06-16): the system's purpose is to pick a **single ideal
reconstruction model** (DA3 is the default — handles ~2× the cameras, fast
meshes + well-covered gaussians for web render). No model should depend on
another. Today `cmd_da3` HARD-REQUIRES matcha's orient gauge
(`matcha_reference` → "run reconstruct-matcha first (bootstrap)"), so DA3 cannot
stand alone (hit on 003-firepit's first full run). Parent: EPI-SCN-FEEDFORWARD-RECON
(see STO-SCN-061 da3-view-alignment, STO-SCN-127 reconstruct-da3-spine-gauge).

## Problem

`reconstruct-da3` exits without a matcha-derived `orient-floor@2` gauge. So:
- DA3 can't run on its own (the chosen default is coupled to the non-default).
- The 2-checkbox pipeline (DA3 / matcha, each independent) is impossible.

The gauge itself is mostly model-independent already: **"up" is pose-derived**
(`gauge_up`, identical for all models); only the **floor plane** (z-height,
maybe azimuth) is fit by RANSAC on a *mesh* — and that mesh is hardcoded to be
matcha's.

## Design

### Approach

The orient already lives in a solve-scoped location:
`images/subsets/<sub>/cameras/<solve>/orient/<oid>/oriented.json`. Make the
lookup model-agnostic and the computation a fallback:

> On reconstruct (matcha **or** DA3): if an orient gauge exists for this solve
> (from ANY prior run) → **reuse it**. Else → **compute it from this model's own
> mesh** (reusing `bootstrap_orient(mesh_verts, cam_R, cam_C)`, which is already
> mesh-agnostic) and store it for the next model.

This guarantees alignment (the second model reuses the first's gauge → identical
frame) and makes each model standalone (the first computes it). It is fully
**backward-compatible**: when matcha runs first (historical path), DA3 finds its
orient exactly as today. The gauge is a **stored sidecar**, not baked per-render,
so it is **re-alignable post-hoc** (recompute/replace `oriented.json` — same
additive pattern as the metric `datum.json` / `apply_to_gauge`).

### Changes

| File | Change |
|------|--------|
| `v4exec.py` (`cmd_da3`) | replace `matcha_reference(...)` requirement with `find_any_orient(solve)`; if none, compute from DA3's own mesh + store (don't `sys.exit`) |
| `v4exec.py` (`cmd_reconstruct_matcha`) | reuse an existing orient if present (e.g. from a prior DA3 run) instead of always recomputing from matcha's mesh |
| `v4exec.py` | `find_any_orient(solve_dir)` — glob `orient/*/oriented.json`, return the existing gauge (newest) |
| `tests/` | fixture: orient reuse (matcha-first → DA3 reuses; DA3-first → matcha reuses); standalone DA3 computes when none exists |

## Definition of Done

- [ ] `reconstruct-da3` runs to a mesh on a scene with **no matcha** (computes its own orient).
- [ ] `reconstruct-matcha` reuses an existing DA3 orient when present (no recompute).
- [ ] With both models run (either order), their meshes share the identical orient gauge → renders align.
- [ ] `find_any_orient` + the reuse/fallback are unit-tested.
- [ ] **Operator-verified (T-020):** DA3-first on 003 — confirm the RANSAC floor on a DA3 mesh comes out clean (gravity ~1–2°, floor flat). If noisy, fall back to fitting the floor on the solve's sparse points.

## Testing

### Unit / fixture tests
- [ ] `find_any_orient` returns an existing gauge regardless of which model's bootstrap produced it.
- [ ] DA3 with no orient present → computes + stores one (no exit).
- [ ] matcha with an existing orient → reuses it.

### Integration
- [ ] 003-firepit DA3-only → mesh (full video→mesh path, no matcha).
- [ ] DA3 then matcha (and vice-versa) → identical orient gauge id consumed.

## Out of scope

- The 2-model checkbox UI + conditional reconstruct phases (pipeline_run / STO-SCN-150 follow-on) — this story is the orient decoupling that makes those checkboxes independent.
- Changing the floor-fit algorithm itself (RANSAC) — only its *input mesh* becomes flexible.
- The fully model-independent "orient from solve sparse points" path — kept as the documented FALLBACK if DA3-mesh floors prove noisy, not the default.

## Implementation Notes

_(Fill in during/after implementation.)_
