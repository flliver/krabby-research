---
xid: STO-SCN-136
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-3eqk
assignee: krabby
---

# Expose mesh-cull knobs (depth_trunc / num_cluster / tetra depth-gradient filter / min-views / max-dist) to kill distant + sky junk

## Summary

A meshify-time **cull stage** that removes low-confidence distant junk and sky/far-field
geometry from matcha meshes, by surfacing the MAtCha-native depth/cluster filters (already in the
container but pinned to permissive defaults) plus our existing post-process `cull_mesh.py` knobs,
as **tunable** settings on the meshify tasks — so the operator can dial "drop the sky / drop the
1-view floaters" without leaving the v4 store contract.

## Context

**Source:** operator, 2026-06-15. Opened the matcha-15 **tetra** mesh (`X6T27S2FWQQZ` /
`BOBOCDFZ2OPC`, 17.2 M tris) in Blender and called it *"complete garbage"* — spiky stretched
triangles into the sky and a halo of distant low-confidence speckle. Asked which **available
settings** address (1) low-confidence junk in the distance and (2) sky / far-away things.

**Finding (the investigation that motivated this story):** of the three catalog tunables
(`dense_regul`, `n_iters`-frozen, `mesh_res`), **none** touch distance/confidence/sky. The v4
pipeline emits the matcha mesh **raw** — the tetra path runs **zero** post-filter, and even the
TSDF path only gets the permissive defaults. The real levers exist but aren't exposed:

### Levers found in the MAtCha container (`krabby-matcha:0.2.2-selfcontained`, `/opt/MAtCha`)

**TSDF path** — `2d-gaussian-splatting/render_multires.py` CLI flags (we currently pass only
`--depth_ratio 1.0 --num_cluster 50 --mesh_res N --multires_factors 2 8 16`):
| Flag / config key | Default | Effect on junk |
|---|---|---|
| `--depth_trunc` | `-1` → auto `radius·factor` | **Max depth range for TSDF fusion.** The literal *sky / far-away* killer — cap it explicitly to stop fusing far-field. |
| `multires_factors` | `[2,8,16]` | the **×16** pass fuses depth out to 16× scene radius — that's where the distant sky comes from. **Drop the 16** (or lower) to cut the far field. |
| `--num_cluster` | `50` | keep N largest connected components. **Lower it** (e.g. 1–5) → drops disconnected distant floater/sky islands. The *low-confidence-distant-junk* killer. |
| `--voxel_size`, `--sdf_trunc` | auto | TSDF fusion granularity / truncation band. |
| `--unbounded` | off | bounded vs unbounded meshing. |

**TETRA path** — `configs/adaptive_tetrahedralization/default.yaml` (the **only** config; selected
by `train.py --tetra_config`). The tetra mesh you opened is worst-case because these are **off**:
| Config key | Default | Effect |
|---|---|---|
| `filter_with_depth_gradient` | **`false`** | **THE sky/silhouette killer** — removes stretched triangles at depth discontinuities (object→sky boundaries = the spikes). Turn **on**. |
| `filter_with_normal_consistency` | **`false`** | drops ill-oriented faces (noise speckle). Turn **on**. |
| `filter_mesh` | `true` | (already on) |
| `truncation_margin`, `gaussian_flatness`, `use_binary_opacity`, `use_dilated_depth` | — | secondary surface-quality knobs. |

**Gaussian refinement** (upstream of both meshes) — `configs/free_gaussians_refinement/default.yaml`
(`opacity_reset_interval`, `use_mip_filter`, …): coarser lever; pruning low-opacity gaussians
thins junk at the source but affects everything.

**Post-process (our own tool, NOT wired into v4)** — `real2sim/cull_mesh.py`:
| Flag | Default | Maps to |
|---|---|---|
| `--min-views N` | 2 | drop verts seen by < N cameras → **low-confidence distant junk** (low parallax = 1 view). |
| `--max-dist-from-cluster D` | 0 (off) | drop verts > D m from centroid → **sky / far away**. |
| `--floor-z-min` | −0.5 | drop sub-floor tetra spikes. |

## Problem

The matcha mesh in the v4 store is uncullable today: the only post-filter knobs (`num_cluster`,
`depth_trunc`, the tetra depth-gradient/normal filters, plus `cull_mesh.py`'s view/distance culls)
are either pinned to permissive defaults or not invoked at all. The operator cannot reduce
distant/sky garbage without forking scripts. We need these as **first-class tunables** on the
meshify tasks, flowing into the content-addressed identity so each cull setting is a distinct,
reproducible store node.

## Design

### Approach

Two complementary surfaces — prefer (A) for the TSDF path (native, no extra pass) and add (B) as
a path-agnostic finisher:

**(A) MAtCha-native cull flags (TSDF-native), as a NEW meshify variant — only if needed.** The
flags `depth_trunc` (float, −1=auto), `num_cluster` (int, default 50), `multires_factors`
(array, default `[2,8,16]`) thread into the `render_multires.py` invocation in
`v4exec.cmd_matcha` (the direct-`render_multires` path already exists from STO-SCN-133's
`--mesh-res`). **They must NOT be appended to `meshify-via-tsdf` — that re-keys all historical
TSDF meshes** (see § Backwards compatibility). If pursued, they belong on a **new task variant
with a new algo** (`meshify-via-tsdf-culled` / `tsdf-extract-culled@1`). For the **tetra** path,
the only config is baked, so a `tetra_filters` profile is delivered by **dropping a runtime YAML**
next to the staged work (override `--tetra_config <path>`) with `filter_with_depth_gradient` /
`filter_with_normal_consistency` on — **no image rebuild** (mirrors the "configs are baked, can't
drop a new yaml" workaround already in cmd_matcha). Prefer (B) first; reach for (A) only if the
post-process cull can't get there.

**(B) Wire `cull_mesh.py` as an optional post-meshify cull node** (path-agnostic: works for
matcha tetra/tsdf AND da3). Tunables `min_views`, `max_dist_from_cluster`, `floor_z_min`. This is
the one that needs the oriented cameras (it already takes `--oriented-cameras`), so it slots after
orient. Emits a culled `mesh.ply` as its own meshify-condition node (this epic's `condition/`
placement already exists — see `represent/*/meshify/*/*/condition/<id>/mesh.ply`).

All new settings are **tunable** and flow into `hashable_settings` → identity, so a culled mesh is
a distinct store node from the raw one (raw stays for comparison; no clobber). T-016: cull is cheap
CPU, runs on the gather host — no GPU re-run needed for (B); (A) re-runs only the TSDF extract.

### Changes

| File | Change |
|------|--------|
| `real2sim/tasks/cull-mesh.json` (**new task**, new `algo@version`) | `min_views`, `max_dist_from_cluster`, `floor_z_min` tunables — a conditioning node keyed on `{raw mesh identity} + {cull settings}` (NOT keys appended to an existing meshify task — see § Backwards compatibility) |
| `real2sim/tasks/meshify-via-tsdf-culled.json` (**new task variant**, optional) | TSDF-native `depth_trunc`/`num_cluster`/`multires_factors` — only if the post-process cull is insufficient; a NEW algo, never appended to `meshify-via-tsdf` |
| `real2sim/v4exec.py` (`cmd_matcha`) | new cull node invoking `cull_mesh.py` post-orient → `condition/<id>/mesh.ply`; tetra-filter YAML written+mounted under the (new) culled variant |
| `real2sim/cull_mesh.py` | confirm it reads the v4 oriented cameras path; otherwise fine as-is |
| `real2sim/knowledge/scene-processing/T3c-reconstruction-postprocessing.md` | document the cull knobs + recommended profiles |

> ⚠️ The existing `meshify-via-tsdf.json` / `meshify-via-tetra.json` taskdefs are **frozen** — do
> NOT append the new tunables to them. See § Backwards compatibility for why.

## Backwards compatibility — store identity (MUST READ before touching taskdefs)

**Goal (operator, 2026-06-15):** adding tunable parameters must NOT change the identity hash of
any already-materialized mesh, so historical store nodes keep deduping (re-run = NOOP) and the
rank UI keeps finding them.

**The mechanism is NOT "set defaults + omit them when default."** Verified against
`v4core.hashable_settings` (v4core.py:91–94): it **injects** every tunable+frozen default *into*
the hashed dict — it does **not** omit defaults. Consequences:

1. `{}` and `{mesh_res:1024}` collide **because both get `mesh_res:1024` injected**, not because
   defaults are dropped. (This is why STO-SCN-133's proof `tid({}) == tid({mesh_res:1024})` held.)
2. **Adding a NEW key to an existing taskdef re-keys every already-materialized artifact** — the
   new default is injected → a new field in the canonical JSON → a new hash. Re-running `cmd_matcha`
   would then miss the existing dir, rebuild on GPU, and write a duplicate node. **This is the
   breakage to avoid.**
3. The inverse "fix" — rewriting `hashable_settings` to omit defaults — would **also** break the
   store: every existing node was hashed *with* defaults baked in; recomputing without them
   orphans all of them. Not backwards compatible either.

**Misleading precedent:** STO-SCN-133's `mesh_res` was safe *only because `mesh_res` already
existed as a key* in `meshify-via-tsdf.json` — STO-133 added a CLI *override* for an existing key,
**no new settings key**. That is **not** a template for adding new knobs to the same task.

**The backwards-compatible rule (adopt this):**
- Leave `meshify-via-tsdf` / `meshify-via-tetra` taskdefs **unchanged** → existing mesh identities
  are preserved, re-runs stay NOOP, raw meshes stay for comparison.
- Put the cull on a **new additive conditioning node** (new `algo@version`) keyed on
  `{raw mesh identity} + {cull settings}` (the epic's existing `…/meshify/*/*/condition/<id>/`
  placement). A culled mesh is a *new* node; nothing historical moves.
- If TSDF-native extraction knobs (`depth_trunc`/`num_cluster`) are needed (they change the
  extraction, not a post-process), introduce a **new meshify task variant with a new algo** —
  never append keys to `meshify-via-tsdf`. (The post-process `cull_mesh.py` `--max-dist`/`--min-views`
  approximate them and are fully store-safe, so try that first.)
- **DoD addition:** prove it — recompute the identity of an existing matcha mesh after the change
  and assert it is unchanged (the new cull node's identity is additive, not a re-key).

## Definition of Done

- [ ] `meshify-via-tsdf` exposes `depth_trunc` + `num_cluster` (+ `multires_factors`) as tunables; they reach `render_multires.py` and change the mesh.
- [ ] A tetra cull profile turns `filter_with_depth_gradient` + `filter_with_normal_consistency` on without an image rebuild.
- [ ] An optional post-meshify `cull_mesh.py` node applies `min_views` / `max_dist_from_cluster` / `floor_z_min`, emitting a distinct `condition/<id>/mesh.ply`.
- [ ] All new settings flow into the content identity (culled mesh ≠ raw mesh node; re-run is NOOP).
- [ ] **Operator-verified (T-020):** re-build matcha-15 with a cull profile, open in Blender, confirm the sky/distant garbage is gone vs the raw `X6T27S2FWQQZ` tetra.
- [ ] T3c doc updated with the knob → effect table + recommended defaults.

## Testing

### Unit / fixture tests
- [ ] `hashable_settings` resolves new tunables; identity differs raw vs culled; default-equality holds (`{}` == explicit defaults).
- [ ] Tetra-filter YAML is written + referenced correctly.

### Integration
- [ ] matcha-15 TSDF with `num_cluster=3` + capped `depth_trunc` → far-field islands gone, tri-count drops.
- [ ] matcha-15 tetra with depth-gradient filter on → spikes gone.
- [ ] `cull_mesh.py` node with `min_views=3` drops 1-view distant speckle.

## Out of scope

- Watertighting / gap-fill / smoothing (STO-SCN-013/014/015 in this epic).
- Learned sky-segmentation masks (a heavier approach; native depth_trunc + cluster + gradient
  filters should suffice first — revisit only if they don't).
- Changing upstream gaussian pruning defaults (`free_gaussians_refinement`) — coarse, affects
  everything; deferred.

## Implementation Notes

_(Fill in during/after implementation.)_

### Investigation record (2026-06-15)
Container source read on tbeeprz (`docker run … krabby-matcha:0.2.2-selfcontained`):
- `train.py` exposes `--tetra_config` / `--tsdf_config` / `--free_gaussians_config`
  (name-selectable, but only `default.yaml` ships for tetra+tsdf → runtime-YAML override needed).
- `render_multires.py` exposes `--depth_trunc`, `--num_cluster`, `--voxel_size`, `--sdf_trunc`,
  `--unbounded`, `--mesh_res`, `--multires_factors`. The multires loop sets
  `depth_trunc = radius·factor` per factor in `[2,8,16]` — the ×16 pass is the far-field source.
- tetra `default.yaml`: `filter_with_depth_gradient: false`, `filter_with_normal_consistency: false`
  (both OFF → why raw tetra is spiky).
