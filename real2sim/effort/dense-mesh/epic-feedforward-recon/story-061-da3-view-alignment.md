---
xid: STO-SCN-061
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-e0d
---

# DA3 frame alignment — render DA3 gaussians from saved comparison views

## Summary

`real2sim/da3_render_view.py`: aligns DA3's coordinate frame to a
scene's oriented frame through the cameras both pipelines solved
(gauge_align orientation-augmented Umeyama — the photo-spine module,
T-013), maps a saved schema-5 view into DA3's frame, renders the
gaussians there with DA3's own chunked gsplat renderer, and writes
`pipeline-da3/run-<r>/renders/<view>.png` + settings sidecar — making
DA3 a first-class runoff variant (STO-SCN-058 layout).

## How it works

1. Matcha cameras → oriented frame (R, z_shift from
   oriented_cameras.json applied to mast3r cams2world).
2. DA3 poses from `exports/npz` — extrinsics convention
   **auto-verified at runtime** by aligning both hypotheses (w2c/c2w)
   and keeping the sane residual; hard refusal above 10% of camera
   spread (T-002, no silent garbage renders). 006: w2c, residual
   2.9% of spread, scale 0.347.
3. View camera (world_position + quat, oriented frame) → DA3 frame via
   the inverse similarity; intrinsics from lens/sensor (Blender AUTO
   fit: fx = W·f/sensor_w).
4. Render: `run_renderer_in_chunk_w_trj_mode(trj_mode="original",
   use_sh=True)` with the model FREED first.

## Gotchas (each one cost a failed run)

- gs_video exporter OOMs 16 GB at 1080p (layout video + model
  resident) — render directly, free the model first.
- Renderer asserts n_views>1 for trj_mode="original" — duplicate the
  camera, keep frame 0.
- Single-view gs_video silently switches to a "wander" trajectory —
  never use it for view-matched comparison renders.
- Metric predictions: camera translation must be divided by
  `prediction.scale_factor` to land in gaussian space (mirrors the
  exporter).
- render_exts must be CUDA tensors.

## Result (006-kubota, overhead-grass-quality)

**Fletcher's point confirmed on our own scene.** Same 8 photos, same
view: matcha mesh render has large holes (gray void through lawn and
edges); DA3 gaussian render has ZERO holes — complete coverage,
softer detail (process_res 504). Now rankable in rate_renders:
variant `da3--8-giant` next to `matcha--8-dense-strong/strong`.

## Definition of Done

- [x] Alignment tool version-controlled, convention-verified, with
      hard residual gate.
- [x] DA3 render from the saved view in the run's renders/ + sidecar
      (alignment record embedded).
- [x] Appears in rate_renders beside the matcha variants.
- [ ] Operator views/ranks the DA3 variant (T-020 — operator step).

## scene_compare.blend gotchas (operator-caught bugs, 2026-06-10)

Operator inspection caught two bugs in the first comparison blend
(guessed-not-measured transforms — T-012 violation, both fixed
empirically):

1. **DA3's GLB is NOT in DA3 world frame.** The exporter applies
   `A = T_center(median of cloud) @ diag(1,-1,-1) @ w2c0` (first-camera
   alignment + CV->glTF flip + median centering) — and Blender's glTF
   importer BAKES its Y-up->Z-up into the vertex data with identity
   node matrices. Correct root matrix: `[sR|t] @ A^-1 @ G^-1`
   (G = gltf->blender axis map). A was reconstructed from the exporter
   source + npz and verified against the raw GLB accessor bytes
   (bbox agreement ~1 cm).
2. **Loose-point meshes display black/uncolored.** Blender won't shade
   loose vertices: Geometry Nodes Mesh->Points (radius ~0.012) +
   Set Material with an *Attribute* node (`Color`) —
   ShaderNodeVertexColor does not work for point clouds.

`scene_compare.blend` (in the DA3 run dir) carries the fixed result:
matcha mesh + DA3 cloud + DA3 camera frustums, one oriented frame.
