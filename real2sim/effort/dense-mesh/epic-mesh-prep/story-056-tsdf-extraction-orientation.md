---
xid: STO-SCN-056
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-3dk
shipped: 2026-06-10
tasks: 1
complete: 1
---

# Phase: TSDF mesh extraction + gravity orientation (store conditioning)

> Retroactive phase documentation (operator directive 2026-06-10).
> Recipe section: `real2sim/RECIPES.md` § Common trunk steps 3–4.

## What we did

Extracted the multi-resolution TSDF mesh from a trained run
(`mast3r_sfm` + `free_gaussians` → `tsdf_meshes/multires_tsdf_post.ply`)
and gravity-oriented it (floor → z=0, up → +z) for Blender/render
consumption. Applied to the full runoff (001–012 conditioning),
001's full-res re-extraction (320 MB lowmem → 1,128 MB default —
the quality lesson), and 013 (43.3M raw / 30.8M post verts).

## Where the code is

- TSDF: `scripts/extract_tsdf_mesh.py` **inside the matcha image**
  (`/opt/MAtCha`), config `default` = mesh_res 1024, factors [2,8,16].
- Orientation compute: `real2sim/orient_mesh.py` (RANSAC floor fit;
  STO-SCN-004) → `oriented/oriented_cameras.json` + oriented tetra.
- Orientation apply: `real2sim/apply_existing_orientation.py`
  (rigid transform of the TSDF mesh into the computed frame).
- Worked host scripts: the 013 run sequence is recorded in
  STO-SCN-052's run log; the canonical command shape is in RECIPES.md.

## How

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/extract_tsdf_mesh.py -s <mast3r_sfm> -m <free_gaussians> \
  -o <tsdf_meshes> -c default
python orient_mesh.py --tetra <tetra_meshes>/tetra_mesh_binary_search_7.ply \
  --cameras <mast3r_sfm>/cameras.json --output <data>/oriented/
python apply_existing_orientation.py --in-mesh <tsdf_meshes>/multires_tsdf_post.ply \
  --orientation <data>/oriented/oriented_cameras.json \
  --out-mesh <tsdf_meshes>/multires_tsdf_post_oriented.ply
```

## Gotchas (each cost a failed run)

- **Never the lowmem config** for deliverables — visibly degraded
  meshes (001: 320 MB vs 1,351 MB at the same scene location).
- **Freshness-gate, never existence-gate**: the tool's `os.system`
  swallows child exit codes; rc=0 with a stale mesh on disk is its
  failure mode. Gate: `[ <out>.ply -nt $MARKER ]`.
- `expandable_segments:True` required (fragmentation OOM otherwise).
- ≥17 cameras × mesh_res 1024 OOMs 16 GB GPUs in the multires merge
  → fixed in image 0.2.2 (STO-SCN-053, `multires_oom` patch).
- `mediapy` import: baked into 0.2.1+ (was a transient pip install).

## Definition of Done

- [x] Phase documented here + RECIPES.md section.
