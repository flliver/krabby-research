---
xid: STO-SCN-137
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-15
depends-on: [STO-SCN-136]
bd-id: krabby-um79
assignee: krabby
---

# Cull via posed-camera bounding box — port the scout's gravity-aligned camera-AABB cull to the mesh

## Summary

A mesh cull that **keeps everything inside the posed-camera bounding box and drops everything
outside it**: take the full posed camera spine, circumscribe the camera centers with an
axis-aligned box **in the mesh's own gravity-aligned frame** (down = gravity), expand the box by a
**configurable buffer** per side, then keep mesh vertices inside the box and cull the rest. This is
the **single most reliable "drop the sky / drop the far field" lever** because the bound comes from
where the cameras actually were, not a heuristic distance or confidence threshold.

## Context

**This is already implemented — in gaussian space.** During the scout / spine verify-surface work
(STO-SCN-095), the operator specified (2026-06-14) exactly this cull and it was built into the
scout tool for **splats**: `real2sim/verify_viewer/build_verify.py` →

- recovers gravity-up from the poses (`gauge_up.up_from_poses`),
- builds a gravity-aligned basis `Rg` (rows = two ground-plane axes `e0`,`e1` + up `U`; maps
  solve→gravity) so the box is **axis-aligned to gravity + the ground plane, not the arbitrary
  solve axes** — vertical extent is the tight ground-height, the two horizontal axes are wider,
- projects the camera centers into that frame and takes the AABB (`gmn`/`gmx`),
- **expands each axis by `--cull-expand` per side** (the configurable buffer; default `1.0` =
  +100%/side = 3× camera span),
- maps each gaussian gaussian→solve→gravity and **keeps it iff inside `[gmin, gmax]`** (the
  `cull_box()` function), with a correct 17×float32 PLY rewrite + a T-012 self-verify that refuses
  to serve a corrupt splat.

This story **ports that exact, proven logic from gaussians to the mesh** — same box, same buffer
semantics, same gravity-aligned frame — as a content-addressed cull node in the v4 store.

Sits in EPI-SCN-MESH-CONDITION beside STO-SCN-136 (the `cull-mesh` task it extends). `depends-on:
STO-SCN-136` because it lands as a criterion on / sibling of that cull task.

## Problem

STO-SCN-136 exposes `min_views` (confidence), `max_dist_from_cluster` (a **spherical** radius from
the camera centroid), and `floor_z_min`. The spherical `max_dist` is a blunt approximation of "near
the cameras": it can't crop a long thin corridor without also clipping the room, and it ignores
that the *vertical* (sky/ground) extent should be much tighter than the horizontal. The camera
**bounding box** in the gravity frame is the precise version — and it's the one the operator
already validated on the scout splat. The mesh has no equivalent today.

## Design

### Approach (port `build_verify.cull_box` to mesh vertices — T-013)

The mesh in a meshify/condition node is **already grounded into the oriented (canonical) gauge**
(floor z=0, +z up), and `cull_mesh.py` already loads the cameras into that same oriented frame
(`load_oriented_cameras`). So the mesh case is *simpler* than the splat case — the gravity frame
is the mesh frame; **no `Rg` re-projection is needed**, the box is a plain AABB in mesh coordinates:

1. Oriented camera centers `C` (from `load_oriented_cameras`) → `gmn = C.min(0)`, `gmx = C.max(0)`.
2. `span = gmx - gmn`; `lo = gmn - span*buffer`, `hi = gmx + span*buffer` (per-axis buffer, same
   semantics as the splat tool's `cull_expand`).
3. **Keep** vertices with `lo <= v <= hi` (all 3 axes); **cull** the rest; drop triangles
   referencing a culled vertex; re-index (the existing `cull_mesh.py` triangle/colour machinery).
4. Self-verify the kept set lies in the box (T-012), mirroring `cull_box()`.

Because the mesh is already gravity-aligned, the box's vertical axis is automatically the tight
ground-height and the horizontal axes the wider ground extent — exactly the operator's spec.

### Pipeline integration (v4 content-addressed store)

A cull **criterion on the existing `cull-mesh` task** (STO-SCN-136, `algo: cull-mesh@0`) — a new
`tunable` `cambox_expand` (the per-side buffer; **default `-1` = disabled**, so existing/default
cull behavior is unchanged):

- **Reuse materialized outputs:** consumes the upstream meshify/condition `mesh.ply` (canonical
  gauge) + the solve `cameras.json` + `oriented.json` already resolved by the cull node; pure CPU,
  no GPU. `identity_hash({"mesh": <up id>}, {…, cambox_expand}, "cull-mesh@…")` → **NOOP** when the
  node exists; a box-culled mesh is a distinct store node from the raw + the `max_dist` cull.
- **Placement:** `{up_meshify_dir}/condition/{identity}` — auto-discovered + rendered by
  `v4job.mesh_targets`, so it is immediately rankable (no renderer change), same as STO-SCN-136.
- **Backwards-compat:** adding `cambox_expand` to `cull-mesh@0` is safe **only while that task has
  no materialized historical nodes** (it is brand-new in STO-SCN-136); if any `cull-mesh@0` nodes
  exist by the time this lands, introduce it as **`cull-mesh@1`** (a new algo version) instead of
  appending the key — never re-key materialized nodes. Canonical rule + mechanism:
  **STO-SCN-136 § "Backwards compatibility — store identity"**.

### Changes

| File | Change |
|------|--------|
| `real2sim/cull_mesh.py` | add `--cambox-expand` (per-side buffer); when set, keep verts inside the oriented camera AABB expanded by that fraction (port of `build_verify.cull_box`); compose with the existing view/floor/dist culls |
| `real2sim/tasks/cull-mesh.json` | add `cambox_expand` tunable (default `-1` = disabled) — or bump to `cull-mesh@1` if `@0` nodes already exist |
| `real2sim/v4exec.py` (`cmd_cull`) | thread `--cambox-expand` through to `cull_mesh.py`; flows into identity |
| `real2sim/knowledge/scene-processing/T3c-reconstruction-postprocessing.md` | document the camera-bbox cull + recommended buffer (cross-ref the scout splat origin) |

## Definition of Done

- [ ] `cull_mesh.py` keeps mesh verts inside the gravity-aligned posed-camera AABB expanded by a
      configurable buffer, culls the rest (ported from `build_verify.cull_box`), with a T-012
      self-check.
- [ ] Wired as a `cambox_expand` tunable on the `cull-mesh` task; flows into content identity
      (box-culled mesh ≠ raw ≠ `max_dist` cull; re-run is NOOP); default disabled → no change to
      existing cull nodes.
- [ ] Backwards-compat preserved: no materialized `cull-mesh@0` node is re-keyed (additive tunable
      while unmaterialized, else `cull-mesh@1`).
- [x] **Operator-verified (T-020):** operator exercised the cambox cull at the `overview` view on
      007-kubota's filtered tetra (`WORX76X2NUMK`), tuned `cambox_expand` 0.5 → 0.7 to keep more
      corridor context, and **declared it done** (2026-06-15). The far fragments outside the camera
      box are culled, the toured corridor intact. Clean sign-off.
- [ ] T3c doc updated (knob + recommended buffer; cross-ref the scout origin).

## Testing

### Unit / fixture tests
- [ ] Camera AABB + per-side buffer expansion matches `build_verify`'s `gmin/gmax` math on a shared
      fixture (same cameras → same box).
- [ ] `cambox_expand=-1` (disabled) → mesh byte-identical to the no-cambox cull; identity equals the
      pre-cambox node where other knobs match (default-equality).

### Integration
- [ ] matcha-15 tetra with `cambox_expand=0.25` → far-field/sky outside the camera box dropped,
      tri-count falls, interior intact.
- [ ] da3 mesh (already in oriented gauge) box-culls without re-orienting.

## Out of scope

- The other cull criteria (`min_views`/`max_dist_from_cluster`/`floor_z_min`) — STO-SCN-136.
- Oriented (non-axis-aligned) boxes beyond the gravity-aligned AABB — the gravity-aligned AABB is
  the operator's spec and what the scout validated; a tighter convex/oriented hull is a later idea.
- Re-deriving gravity for the mesh: it is already grounded into the oriented gauge upstream
  (orient node) — this cull consumes that, it does not re-orient.

## Implementation Notes

_(Fill in during / after implementation.)_

### Origin (proven prior art — T-013)
The gaussian-space implementation is `real2sim/verify_viewer/build_verify.py`:
`cull_box(src, dst, Rg, gmin, gmax, scale, R, t, …)` + the gravity-basis/box computation (the
`Rg`/`gmn`/`gmx`/`cull_expand` block) + the `--cull-expand` CLI arg. Operator spec 2026-06-14;
operator-confirmed correct on the 001-patio scout splat. This story applies the identical box to
the mesh (where the gravity frame == the mesh frame, so the per-vertex test is a plain AABB).

### Built 2026-06-15
- **`cull_mesh.py`**: `--cambox-expand E` — keep verts inside the oriented camera-center AABB
  expanded by `E·span` per side, cull outside (composes with the view/floor/dist culls);
  `<0 = disabled`.
- **`cull-mesh` taskdef + `cmd_cull`**: `cambox_expand` tunable threaded through to identity.
- **Backwards-compat:** `cull-mesh@0` already had materialized nodes, so adding `cambox_expand`
  bumped the task to **`cull-mesh@1`** (adding a key to `@0` would re-key its nodes). Verified the
  existing `@0` cull nodes recompute **unchanged**; new cambox culls are clean `@1` nodes.
- **Run:** all 3 meshes culled at `cambox_expand=0.5` (DA3 → `CYBXPPRMAN3F`, …) + rendered.
- **Note (earned):** the camera AABB is vertically THIN (cameras at ~uniform height), so the box
  is a horizontal slab unless the buffer is generous — `expand` needs to be larger to capture
  floor+walls (build_verify's gaussian-space default was 1.0). Operator can tune per scene.
- Operator-verify (T-020) at the new `overview` render view (the lone `view 01` is embedded in
  the geometry and can't show the crop).
