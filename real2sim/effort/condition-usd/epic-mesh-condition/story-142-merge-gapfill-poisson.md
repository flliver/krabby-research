---
xid: STO-SCN-142
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-15
depends-on: [STO-SCN-099]
bd-id: krabby-tew9
assignee: krabby
priority: 1
---

# (A) Merge & gap-fill via screened Poisson — watertight manifold conditioning node

# ★ PRIORITIZED approach for STO-SCN-013 (the other approach is STO-SCN-143 TSDF re-fusion, deferred)

## Summary

A `merge-gapfill` conditioning task that makes a materialized (e.g. culled) mesh **manifold +
watertight + hole-free** via **screened Poisson reconstruction** (Open3D) over the mesh's
vertices + normals, with density-trim + keep-largest-component cleanup. Pure CPU, additive
content-addressed node — the realization of STO-SCN-013's goal, chosen first because it's a true
post-process (input = a good mesh, output = watertight) with no depth/camera inputs.

## Context

STO-SCN-013 ("Merge & Gap-Fill") names two approaches — Poisson on the output, or continuing TSDF
fusion. Operator decision (2026-06-15): **do Poisson (A) first; defer TSDF re-fusion (B,
STO-SCN-143) until we see A's results.** This story is approach (A).

## Design

### Approach (Open3D screened Poisson — mirrors the cull_mesh.py condition-node pattern)
1. Load the upstream mesh (canonical gauge); ensure vertex normals.
2. Sample an oriented point cloud (verts+normals, or Poisson-disk sample faces for even density).
3. `o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(depth=…)` → watertight manifold.
4. **Density-trim** (drop the low-density Poisson skirt that balloons past the real surface) +
   `remove_low_density_vertices` + keep-largest-connected-component.
5. Transfer color from the source mesh (nearest-vertex), recompute normals, emit `mesh.ply`.

### Pipeline integration (v4 content-addressed store)
- **Task:** `tasks/merge-gapfill.json` (**new**, `algo: merge-gapfill@0`). Input `mesh`
  `from: meshify`/`condition`. Tunables: `poisson_depth` (e.g. 9), `density_quantile` (trim),
  `samples` — each a distinct node.
- **Placement:** `{up_meshify_dir}/condition/{identity}` — auto-discovered + rendered by
  `v4job.mesh_targets`; rankable; **NOOP** on re-run. CPU on the gather host (no GPU).
- **Chains after cull/filter** (a condition node may take another condition node as input — proven).
- **Backwards-compat:** new additive task + algo; never appended to a meshify taskdef. Canonical
  rule: **STO-SCN-136 § "Backwards compatibility — store identity"**.

### Changes (drafting)
| File | Change |
|------|--------|
| `real2sim/merge_gapfill.py` (new) | Open3D Poisson + density-trim + largest-component + color transfer |
| `real2sim/tasks/merge-gapfill.json` (new) | `merge-gapfill@0`, tunables `poisson_depth`/`density_quantile`/`samples` |
| `real2sim/v4exec.py` (`cmd_mergefill` + `mergefill` parser) | resolve mesh node → run merge_gapfill.py → `condition/<id>/mesh.ply` + metadata (mirror `cmd_cull`) |

## Definition of Done
- [ ] Mesh manifold (no non-manifold edges/vertices) — verify with STO-SCN-014.
- [ ] No visible holes in walkable surfaces.
- [ ] Volume preserved (no significant shrinkage / ballooning — density-trim tuned).
- [ ] Implemented as an additive `merge-gapfill@0` condition node consuming a materialized mesh
      (NOOP re-run; no GPU; canonical gauge preserved); backwards-compat proven (existing meshify
      identity unchanged).
- [ ] **Operator-verified (T-020):** open the Poisson-filled mesh at the `overview` view, confirm
      holes filled + watertight without gross ballooning.

## Testing
- [ ] Poisson on a culled 001-patio mesh → `is_watertight() == True` (STO-SCN-014 check).
- [ ] Identity differs from raw/culled; re-run NOOP; default-equality holds.

## Out of scope
- TSDF re-fusion (STO-SCN-143, deferred). Watertight verification tool (STO-SCN-014). Smoothing (STO-SCN-015).

## Implementation + findings (2026-06-15)

Built `merge_gapfill.py` (Open3D Poisson + density-trim + largest-component + colour transfer),
`merge-gapfill@0` task (`kind: modifier`), `v4exec mergefill` (chains onto a culled mesh). Two
findings, one fixed, one fundamental:

1. **Crash — FIXED.** Open3D Poisson crashed with `"Failed to close loop"` (FEMTree.IsoSurface)
   across depths (8/9), normal sources (resampled / vertices-direct), AND versions (0.18/0.19),
   and **pymeshlab hit the identical crash**. Root cause (web search): a **PoissonRecon OpenMP
   threading race**, acute on **ARM64** (we're on a mac) — isl-org/Open3D#2027,
   mkazhdan/PoissonRecon#136/#139, colmap/colmap#4335. Fix: **`n_threads=1`** → Poisson now runs
   clean (~8–14 s).

2. **Approach mismatch — FUNDAMENTAL.** Even on the *complete* raw DA3 mesh, Poisson yields
   **161–324 disconnected components, `watertight=False`**, and the overview render shows a
   **giant ballooned closed blob**, not a clean scene. Root cause (T-003): **Poisson reconstructs
   closed OBJECTS; our reconstructions are open SCENES** (patio/pathway — ground + walls + sky, no
   closed boundary). Poisson extrapolates a balloon to "close" the open scene → garbage.

**Design question this surfaces:** what does "watertight" mean for an **open scene**? A true global
seal = the balloon (wrong). The practical physics goal is **"no holes in walkable surfaces" +
manifold** — i.e. **LOCAL hole-filling**, not global Poisson reconstruction.

**Recommended pivot:** redirect (A) from global screened-Poisson → **local hole-filling**
(Open3D `fill_holes` / tensor API) — fills the gaps a robot would fall through, **preserves the
open-scene shape, no balloon**. Keep the `merge-gapfill@0` task + `v4exec mergefill` plumbing (it
works); swap the algorithm in `merge_gapfill.py`. Pending operator confirmation of the watertight
intent. (Global Poisson stays available as a knob if a sealed volume is ever wanted.)

## Notes
Risk: Poisson balloons thin/open geometry + over-smooths — the density-trim + largest-component
cleanup is the mitigation, and STO-SCN-014's genus/manifold check is the gate. If Poisson can't hold
volume on our scenes, that's the trigger to pick up (B) TSDF re-fusion (STO-SCN-143).
