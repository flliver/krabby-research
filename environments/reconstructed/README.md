# environments/reconstructed/

Reconstructed 3D scene assets produced by the M11 real-to-sim
pipeline. Per the M11 ICA §2:

> "Milestone 11 artifacts are merged to `main` when complete (paths
> such as `environments/reconstructed/`, `models/`, and `docker/` as
> described in the grant overview)."

## What lives here

Per-scene subdirectories containing the conditioned, watertight
meshes + collision proxies + USD scenes ready for IsaacSim load:

```
environments/reconstructed/
├── README.md                       (this file)
├── <scene-name>/
│   ├── mesh.obj                    visual mesh
│   ├── mesh_collision.obj          V-HACD convex decomposition
│   ├── scene.usd                   IsaacSim-ready USD with rigid body + collider
│   └── manifest.yaml               provenance metadata (capture date, pipeline, scale calibration)
```

## Storage convention — `>100 MB → S3`

Decided 2026-05-18. Anything **above 100 MB** lives in S3 (bucket
TBD); anything **at or below 100 MB** is committed directly to git.

This typically means:

| Artifact | Size class | Location |
|---|---|---|
| Conditioned mesh (decimated OBJ) | usually < 100 MB | **git** |
| Collision proxy (V-HACD output) | usually < 100 MB | **git** |
| USD scene file | usually < 100 MB | **git** |
| Vertex-colored meshes | sometimes > 100 MB | **S3** |
| Raw video footage | always > 100 MB | **S3** |
| Dense point clouds (fused.ply, full res) | often > 100 MB | **S3** |
| Intermediate sparse SfM outputs | typically < 100 MB | git OR S3 (judgment) |

The S3 path and full sync convention will be documented here once the
bucket is provisioned.

## Status

This directory was created 2026-05-18 as part of the M11 migration
(MIGRATION.md M-3.4). **No scenes have landed yet** — Phase E2 of the
M11 PLAN.md (Mesh-to-USD via Isaac Lab `MeshConverter`) is the open
P0 bead `m11-uy5 T2.E2` that produces the first scene's USD. Track
status via `bd ready` from anywhere in this repo.

## See also

- `real2sim/` — the SfM + mesh pipeline that produces the inputs here
- `images/matcha/` — primary T1 watertight-mesh container
- `images/scene-reconstruction-base/` — COLMAP base container
- `docs/BEADS.md` — issue-tracking convention (bd commands)
- M11 grant OVERVIEW.md (authoritative scope)
