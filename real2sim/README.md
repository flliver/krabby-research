# real2sim — Real-to-Sim Pipeline

The `real2sim/` directory holds the M11 pipeline that takes captured
real-world data (videos / photos) and produces simulation-ready 3D
assets (meshes, USD scenes) for IsaacSim evaluation. This implements
the **real-to-sim pipeline** named in the M11 ICA §1.

```
real-world capture (video) ──┐
                              │
                              ▼
                       SfM (T0)         camera poses + sparse cloud
                              │
                              ▼
                  Dense / mesh extraction (T1)   watertight OBJ / PLY
                              │
                              ▼
                  Conditioning + USD export (T2)  scale, Z-up, physics
                              │
                              ▼
                      IsaacSim env (T2)            robot spawns, depth works
                              │
                              ▼
                  Locomotion eval (T3, T4)         EP + Holosoma on hexapod
```

See:
- **`milestones/011-scene-reconstruction/PLAN.md`** in personal workspace —
  current working plan with Task ↔ Phase ↔ Beads mapping
- **`docs/BEADS.md`** — issue-tracking convention (`bd ready` for current work)
- **Patina M11 OVERVIEW.md** — authoritative grant scope at
  https://github.com/flliver/patina-foundation-grants/blob/main/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md

## Layout

```
real2sim/
├── README.md                       # this file
│
├── *.py                            # mesh-conditioning + scene-build tooling
│   ├── build_blender_scene.py      # produces .blend scenes from cameras + meshes
│   │                               #   grouped collections: cameras_pool / cameras_selected /
│   │                               #   cameras_virtual / meshes (toggle in Outliner);
│   │                               #   --selected-frames partitions pool vs selected;
│   │                               #   --output optional → <run-dir>/scene.blend (STO-SCN-044)
│   ├── localize_reference_image.py # SfM-extend a reference image into existing recon
│   ├── sync_comparison_views.py    # comparison_views.json schema-v4 round-trip
│   ├── colmap_to_cameras_json.py   # COLMAP → cameras.json (Blender consumable)
│   ├── orient_mesh.py              # T1.B1: RANSAC ground plane + Z-up
│   ├── cull_mesh.py                # T1.B2: out-of-bounds geometry cull
│   ├── project_color.py            # T1.B4: vertex-color projection
│   ├── decimate_oriented.py        # T2.D3: decimation pass
│   ├── manifest_lib.py             # scene-manifest helpers
│   ├── apply_existing_orientation.py
│   ├── backfill_manifests.py
│   └── viz_depth_maps.py
│
├── *.sh                            # pipeline runners + utilities
│   ├── run_colmap_*.sh             # T0 sparse + dense (canonical grant path)
│   ├── run_mast3r.sh               # T0 alternative (MASt3R-SLAM)
│   ├── run_vggt.sh                 # T0 alternative (VGGT, in container)
│   ├── run_mesh_conditioning.sh    # T2 mesh prep (Poisson + cleanup + collision proxy)
│   ├── extract_frames.sh           # video → JPEG frames
│   ├── render_comparison_matrix.sh # render comparison_views.json batch
│   ├── run_template.sh             # template wrapper for matcha-build container exec
│   ├── run_status.sh               # status check on running containers
│   ├── lib_progress.sh             # shared progress helpers
│   └── hello.sh                    # smoke test
│
├── camera_viewer/                  # T0.B5: interactive 3D camera-selection viewer
│   ├── README.md
│   ├── viewer.py
│   ├── data.py / clustering.py / filters.py / slots.py / ui.py
│   ├── requirements.txt
│   └── _example_run.sh
│
└── rate_renders/                   # web app for rating/curating rendered comparisons
    ├── server.py
    └── static/                      # HTML/JS/CSS for the rating UI
```

## Adjacent: Docker images

The pipeline-specific containers live at `images/<name>/`:

| Container | Purpose | Build target |
|---|---|---|
| `images/scene-reconstruction-base/` | COLMAP + Open3D base (canonical T0/T1) | `make build-scene-reconstruction-base-image` |
| `images/matcha/` | **Primary T1**: MAtCha watertight TSDF + tetra | `make build-matcha-image` |
| `images/mast3r/` | T0 alternative: MASt3R-SLAM | `make build-mast3r-image` |
| `images/slam3r/` | Alt per grant Appendix A | `make build-slam3r-image` |
| `images/vggt/` | Alt per grant Appendix A | `make build-vggt-image` |

`make build-m11-images` builds all five.

See `docs/DOCKER_DEPENDENCIES.md` for the full per-container detail.

## Quick start

Most pipelines run *inside* a container started from `images/<name>/`.
The `run_*.sh` scripts in this directory are wrappers that call into a
running container via `docker exec`, with the exception of
`run_mast3r.sh` which does its own `docker run`.

Typical T0 (sparse) flow with COLMAP:

```bash
# Once per host: install NVIDIA Container Toolkit
./scripts/setup-docker-gpu.sh

# Build the base image
make build-scene-reconstruction-base-image

# Extract frames + run sparse SfM
real2sim/extract_frames.sh data/videos/<scene>.mp4 data/scenes/<scene>/images
real2sim/run_colmap_sparse.sh <scene> SIMPLE_RADIAL_FISHEYE
```

Typical T1 with MAtCha (the primary path):

```bash
make build-matcha-image
docker run --rm --gpus all \
  --shm-size=8g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD/data":/data \
  --env-file .env \
  krabby-matcha:latest \
  bash -c '
    source /opt/matcha/bin/activate
    cd /opt/MAtCha
    python train.py \
      -s /data/frames/<scene>-matcha-24 \
      -o /data/matcha_output/<scene> \
      --sfm_config unposed --n_images 24 \
      --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
      --depthanything_encoder large
  '
```

(See `images/matcha/README.md` for the operator workflow.)
