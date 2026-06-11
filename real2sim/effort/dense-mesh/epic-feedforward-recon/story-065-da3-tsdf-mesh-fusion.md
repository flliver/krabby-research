---
xid: STO-SCN-065
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-m1u
---

# DA3 TSDF mesh fusion — depths → deliverable mesh (the transformation)

## Summary

The transformation that makes DA3 deliverable-grade: Open3D TSDF
fusion over DA3's dense per-view depths + poses → triangle mesh →
similarity transform into the scene's oriented frame → written as the
STANDARD tsdf mesh artifact so every downstream consumer (render
matrix, rate_renders, eventually USD export) treats the run as an
ordinary variant. The trunk's deliverable is a MESH (operator,
2026-06-10: "The *MESH* version is the necessary output"); splats are
an evaluation surface only.

## Where the code is

- `real2sim/da3_tsdf_mesh.py` — fusion + alignment + standard-path
  write. CPU-only, runs anywhere with open3d+numpy (Mac: `uv run`).
- Shares the camera-set alignment with `da3_render_view.py`
  (gauge_align orientation-augmented Umeyama; both gate at residual
  ≤10% of camera spread).

## How (006-kubota, first execution 2026-06-10)

1. Inputs: DA3 `exports/npz` (depth/conf/extrinsics/intrinsics/image,
   8×378×504) + the matcha run's two anchor JSONs.
2. Conf-threshold (40th pct), per-view RGBD integrate into
   `ScalableTSDFVolume` (voxel = 0.4% of 95th-pct depth, sdf_trunc 4×).
3. Mesh: 213,188 verts / 376,962 tris, 15 MB.
4. Align into matcha-oriented frame (scale 0.347, residual 2.9% of
   spread); floor z=0 inherited.
5. Write `tsdf_meshes/multires_tsdf_post_oriented.ply` +
   `fusion_record.json`; copy the two matcha anchor JSONs into the
   transform dir (+ ANCHORS-README.json) so
   `render_comparison_matrix.sh --mesh-source tsdf` needs zero
   special-casing — anchor residuals at render: 0.0000.

## Result

Mesh render in the runoff beside the matcha variants: lawn fully
CONTINUOUS (zero interior holes — the feedforward bet, demonstrated
in the deliverable format) vs coarser geometry (213k vs 30M verts —
the 504px depth ceiling). Detail lever: process_res > 504 (next
experiment).

## Definition of Done

- [x] Fusion tool versioned; alignment gated; standard-path output.
- [x] 006 mesh in the runoff via the unmodified render matrix.
- [x] Recipe phase 13 documents the mesh path with exact commands.
- [ ] Tools baked into krabby-da3 image (no /tmp scp delivery) —
      see status notes.
