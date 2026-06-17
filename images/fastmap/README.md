# krabby-fastmap

GPU-accelerated SfM container (STO-SCN-101, EPI-SCN-AUTO-SUBSET-SELECT).

## Purpose

Produce camera poses **and a co-visibility / track graph** (the
STO-SCN-093 deliverable) on the GPU. [FastMap](https://github.com/pals-ttic/fastmap)
(PyTorch, TTIC) does GPU pose estimation + sparse triangulation, **replacing
COLMAP's CPU-bound incremental mapper**. COLMAP is present (built with CUDA)
but used **only** for GPU feature extraction + matching — there is no
CPU mapper anywhere (operator directive: GPU-accelerated only).

Outputs COLMAP format (`sparse/0`: `cameras`, `images` poses, `points3D`
with per-point image tracks → co-visibility derivable directly).

## Base image

`j.pski.org:5000/krabby-da3:0.4` — reuses its proven CUDA 12.8 / torch
2.7.0+cu128 / `sm_89;sm_120` (Blackwell) stack.

## Pinned upstreams

- FastMap `dafd165121036746e32a270a0a4e252fafb41ad7` (2026-03-13)
- COLMAP `4.0.4` (built CUDA, `GUI_ENABLED=OFF`, `CGAL_ENABLED=OFF`)
- pyrender fork `jiahaoli95/pyrender` (FastMap dependency)

## Build (on a GPU x86 host — not the Mac)

`krabby-tools/` is a build-time mirror of canonical `real2sim/` sources
(baked via `COPY`). **Always re-sync + gate before building** so the image
can't ship stale tools (STO-SCN-157 — the audit found `covis_graph.py` /
`lib_progress.sh` / `capture_profiles.json` drifted, so the registry image
ran old covis logic):

```bash
images/fastmap/sync-tools.sh            # real2sim/ -> krabby-tools/
images/fastmap/sync-tools.sh --check    # exit 1 on drift (also a good CI gate)

rsync -a images/fastmap/ <host>:~/build/fastmap/
ssh <host> 'cd ~/build/fastmap && docker build -t krabby-fastmap:0.3 .'
docker tag krabby-fastmap:0.3 j.pski.org:5000/krabby-fastmap:0.3
docker push j.pski.org:5000/krabby-fastmap:0.3
```

Do NOT hand-edit files under `krabby-tools/` — edit `real2sim/` and re-sync.

Build is heavy (~COLMAP CUDA compile + FastMap kernels). Risk points:
COLMAP 4.x CUDA build for `sm_120`, and FastMap's `build_ext` kernels.

## Run (GPU end-to-end)

```bash
docker run --rm --gpus all --shm-size=8g -v <data>:/data krabby-fastmap bash -c '
  colmap feature_extractor --database_path /data/database.db \
      --image_path /data/images --SiftExtraction.use_gpu 1 \
      --ImageReader.camera_model SIMPLE_RADIAL_FISHEYE   # from STO-SCN-091
  colmap exhaustive_matcher --database_path /data/database.db \
      --SiftMatching.use_gpu 1
  python /opt/fastmap/run.py --database /data/database.db \
      --image_dir /data/images --output_dir /data/out --headless
'
```

## Notes / open items for STO-SCN-093

- **Camera model:** FastMap defaults to `SIMPLE_RADIAL`. The 091 profile
  supplies `SIMPLE_RADIAL_FISHEYE` for fisheye; confirm FastMap honors the
  fisheye model from the COLMAP database (may need a config/YAML override).
- **Matcher choice:** exhaustive/vocab-tree for hyperlapse/wide-baseline;
  sequential is for dense video (HUG-SCN-004: sequential fails on hyperlapse).
- **krabby-tools baking:** the covis-extractor (093) gets baked in a later
  build so `results.json` provenance covers it (operator policy); this image
  is the solver base.
