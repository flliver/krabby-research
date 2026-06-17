---
xid: STO-SCN-139
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-15
depends-on: []
bd-id: krabby-uh1l
assignee: krabby
---

# Every mesh/scene modifier is a first-class task node, selectable in the studio

## Summary

Every mesh/scene **modifier** (cull, tetra-filter re-extract, condition/decimate-smooth, and the
coming merge/gap-fill, watertight-verify, smoothing, scale-calibrate, USD-export) is a **first-class
v4 task** in the catalog AND surfaced as a **selectable node in the studio** — so the operator can
see the full inventory of modifiers and pick them, not just the reconstruction front-ends.

## Context / Problem

The modifiers we've built are already first-class **catalog tasks** (`/api/tasks`):
`cull-mesh@1`, `meshify-via-tetra-filtered@0`, `condition`/`tetra-condition@0`. **But they are in
no `graphs/*.json`**, so the studio's graph view shows no node for them (same bucket as
`scout`/`select`/`spine-*` — catalog tasks invoked imperatively via `v4exec`). The operator asked
why STO-SCN-137's cull isn't a graph node: because conditioning was built operator-driven, not
graphed. This story makes the modifier inventory **visible and selectable as nodes** in the studio,
independent of any fixed graph — the prerequisite for mix-and-match (STO-SCN-140).

## Design

### Approach
- **Catalog completeness (mostly done):** assert that every mesh/scene modifier ships a
  `tasks/<name>.json` with a versioned `algo@version`, typed settings (tunable/frozen/pin),
  declared `inputs.from` (which upstream node it consumes), and `placement`. Audit the current set;
  add taskdefs for any modifier still invoked without one.
- **Studio exposure:** the studio surfaces the catalog as a palette of **modifier nodes** — each
  task rendered with its settings form (the `_fallback_errors`/catalog-constraint machinery already
  validates settings). A modifier node declares its input kind (a `meshify`/`condition` mesh) so the
  UI knows it's a mesh modifier vs a reconstruction step.
- **No fixed-graph requirement:** a modifier node is selectable on its own (operator-driven), not
  only as part of `reconstruct-matcha`. (Chaining them is STO-SCN-140; promoting a chain to a graph
  is STO-SCN-141.)

### Changes (sketch — drafting only)
| File | Change |
|------|--------|
| `real2sim/tasks/*.json` | ensure every modifier has a taskdef (cull-mesh ✓, tetra-extract-filtered ✓, condition ✓; add for merge-gapfill / verify-watertight / taubin-smooth / scale-calibrate / usd-export as they land) |
| `real2sim/studio/*` (+ `static`) | a "modifiers" palette driven by `/api/tasks`, filtered to mesh/scene modifiers (input kind = mesh), with per-task settings forms |
| `real2sim/v4core.py` (optional) | a task-level `kind: modifier` / `input_kind: mesh` hint so the studio can filter the catalog into reconstruction vs modifier tasks |

## Definition of Done
- [ ] Every mesh/scene modifier has a `tasks/*.json` (versioned algo, typed settings, declared input).
- [ ] The studio lists the modifiers as selectable nodes with settings forms (not buried as
      imperative-only commands).
- [ ] A modifier node's settings are catalog-validated (re-use the existing instance validation).
- [ ] Backwards-compat: additive only — no existing taskdef/graph re-keyed (STO-SCN-136 canonical rule).

## Out of scope
- Chaining modifiers into an experiment (STO-SCN-140) and promoting a chain to a graph (STO-SCN-141).
- New modifier *implementations* (those are their own stories, e.g. STO-SCN-142 Poisson).

## Implementation (2026-06-15)
- **`studio_model.tasks()`**: classify each task as a **modifier** — derived (T-013) from
  `inputs[].from ∈ {meshify, condition}` (consumes a materialized mesh), with an explicit
  `kind: modifier` taskdef override honored. Exposes `x-task.modifier` + `x-task.input_from`.
- **Studio Tasks tab (`index.html`)**: split into a first-class **"Mesh / scene modifiers"** group
  (badge + count + "consumes …" note) above the reconstruction/ingest tasks; settings forms unchanged.
- **Verified:** `/api/tasks` classifies `cull-mesh` + `condition` as modifiers;
  `meshify-via-tetra-filtered` correctly stays a reconstruction variant (consumes gaussians).
- **Backwards-compat:** purely additive read-side (new `x-task` fields, derived) — no taskdef/graph
  change, no identity impact.
- **Operator-verify (T-020) pending:** open the studio **Tasks** tab, confirm the modifier group.

## Notes
Current first-class modifier tasks (verified 2026-06-15): `cull-mesh@1`
(min_views/max_dist_from_cluster/cambox_expand/floor_z_min/image_size), `tetra-extract-filtered@0`
(tetra_filters), `condition`/`tetra-condition@0` (target_tris/taubin_iters). The graphs that exist
(`ingest-scene`, `solve-covis`, `reconstruct-matcha`, `reconstruct-da3`) cover ingest→solve→
represent→meshify(→condition→render); the modifier tail is catalog-only today.
