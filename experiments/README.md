# M11 Experiments

A historical record of every reconstruction pipeline we ran (or attempted)
on every captured scene. Each subdirectory documents one
**(pipeline, scene)** experiment. Output artifacts (`*.ply`, `*.obj`,
`*.bin`, etc.) live under `data/scenes/<scene>/` (gitignored) and are
referenced by path from each experiment's README.

## Quick index

| # | Pipeline | Scene | Status | Best output |
|---|----------|-------|--------|-------------|
| 1 | COLMAP | 001 patio fisheye | ✅ sparse only | `data/scenes/001-patio-fisheye/sparse/0/` (21 MB, 942 frames registered) |
| 2 | MASt3R-SLAM | 001 patio fisheye | ✅ | `mast3r_output/patio/patio_720p.ply` (424 MB) |
| 3 | VGGT | 001 patio fisheye | ❌ OOM | None — frames extracted (`vggt_images/`) but pipeline never produced output |
| 4 | **MAtCha** | **001 patio fisheye** | ✅ **Phase A1** | `matcha_output/tetra_mesh_binary_search_7.ply` (185 MB, 9.2M tris) + 200K-tri decimation (16 MB OBJ) |
| 5 | COLMAP | 002 patio dewarped | ❌ failed registration | None — sparse/ and dense/ empty |
| 6 | MASt3R-SLAM | 003 firepit fisheye | ✅ | `mast3r_output/firepit/firepit_720p_10fps.ply` (72 MB) + Poisson mesh (16 MB OBJ) |
| 7 | SLAM3R | 003 firepit fisheye | ✅ | `slam3r_output/...recon.ply` (15 MB) |
| 8 | **MAtCha** | **003 firepit fisheye** | ✅ **Phase A2** | `matcha_output/tetra_mesh_binary_search_7.ply` (238 MB, 11.8M tris) + 200K-tri decimation (21 MB OBJ) |
| 9 | MASt3R-SLAM | 004 sky-house-dining | ✅ | `mast3r_output/sky_house/004-sky-house-dining.ply` (153 MB) + ball-pivoting mesh (19 MB OBJ) |
| 10 | **MAtCha** | **004 sky-house-dining** | ✅ **★ first MAtCha success** | `matcha_output/tetra_mesh_binary_search_7.ply` (422 MB, 21M tris) + 200K-tri decimation (15 MB OBJ) |
| 11 | AnyRecon | 004 sky-house-dining | ⏸ deferred — released code is video-only, not mesh | See `004-anyrecon-sky-house/README.md` |

**Phase A complete (2026-04-30):** three watertight MAtCha meshes (scenes 001, 003, 004). All Phase-A scene experiments include `CAPTURE-LESSONS.md` documenting scene-specific findings. Cross-cutting post-processing gaps identified in all three drove the Phase B reframing in `../PLAN.md`.

## What's the milestone asking for?

M11's grant deliverable is **collision-quality 3D environments for hexapod
locomotion evaluation in IsaacSim**. That implies five non-negotiables:

1. **Watertight surface mesh** (no holes — the hexapod must not be able to
   fall through gaps that don't exist in the real scene)
2. **Metric-accurate scale** (or scale-recovered after the fact via a
   known reference)
3. **2–3 distinct scenes** delivered
4. **USD-importable output** (or a clean conversion path from OBJ/PLY)
5. **Reproducible pipeline** (Docker recipe + run script, runs on the
   project's reference hardware — RTX 5080)

See `DECISION-MATRIX.md` for how each pipeline scores against these.

## Captured scenes

| Scene | Capture profile | Frames | Used by |
|-------|-----------------|--------|---------|
| 001-patio-fisheye | DJI Action 3 4K hyperlapse, native 155° fisheye | 942 | COLMAP, MASt3R-SLAM, VGGT (attempted) |
| 002-patio-dewarped | Same scene, DJI in-camera dewarp enabled | 944 | COLMAP only (failed; informs the dewarp-dead-end finding) |
| 003-firepit-fisheye | DJI Action 3 native fisheye | varies | MASt3R-SLAM (10fps subsample), SLAM3R |
| 004-sky-house-dining | DJI Action 3 2.7K @ 30fps, locked exposure/WB, stable motion (3:47 video) | 6804 | MASt3R-SLAM (full video), MAtCha (12 sparse keyframes) |

For capture-side findings, see the OLAI corpus at
`3d-reconstruction/capture-profiles`.

## How to read each experiment folder

Each `<num>-<pipeline>-<scene>/` directory contains a `README.md` with:

- **Status** — succeeded / failed / partial
- **Date** — when run
- **Pipeline + recipe** — version + path to Dockerfile/runner
- **Hardware** — which fleet host, GPU
- **Process** — commands, key params, runtime
- **Output** — file paths and sizes (gitignored, lives under `data/`)
- **Quality verdict** — subjective + measurable
- **Milestone fit** — does it satisfy M11 requirements?
- **Lessons** — what we learned that informs future work

Where useful, the folder also includes a consolidated runner script,
config snippets, or links to log files. Experiment-specific patches and
debug notes that aren't covered by the milestone-wide
`docker/MAST3R-NOTES.md` or `docker/MATCHA-NOTES.md` live here.

## Where to find...

| Concern | Location |
|---------|----------|
| Build recipes | `docker/Dockerfile.{mast3r,slam3r,vggt,matcha}`, `docker/MAST3R-NOTES.md`, `docker/MATCHA-NOTES.md` |
| Capture profiles | `REPORT-2026-04-11.md`, OLAI corpus `3d-reconstruction/capture-profiles` |
| Runner scripts | `workspace/run_*.sh` (COLMAP, MASt3R, mesh conditioning, VGGT) |
| Method comparisons | `DECISION-MATRIX.md` (this dir), OLAI corpus `3d-reconstruction/{vggt,mast3r-slam,slam3r,colmap,matcha,any-recon}` |
| Output artifacts | `data/scenes/<scene>/` (gitignored, local-only) |
| Outpost availability | `FLEET.md` |
