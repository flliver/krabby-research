# Data Recipes — capture types, and precisely how to process them

> One recipe per *kind of input data*. Every step names the hardened
> tool that performs it (T-025 — no freelance re-implementation). All
> recipes converge on the same trunk after preprocessing. Grounded in
> scenes actually processed (T-010); when a new data type shows up,
> add a recipe here as part of processing it.

Scene store layout (all recipes):

```
scenes/<scene>/
├── input/                      # raw capture + preprocessing
│   ├── <original capture>     #   video file / photo dir, original names
│   ├── src/                    #   the canonical frame/photo pool
│   └── preproc-NN-<slug>/      #   spec-driven transforms (spec + results + data/)
├── pipeline-<p>/run-<r>/       # reconstruction runs (transforms 01..N)
│   └── renders/                #   <view>.png + <view>.json settings sidecar
│                               #   (the render belongs to the RUN that
│                               #   produced it — STO-SCN-058)
├── cameras.json                # scene-level unified views (schema 5)
└── rankings.jsonl              # operator runoff rankings (eval data)
```

## Hard limits (apply to every recipe)

| Limit | Value | Source |
|-------|-------|--------|
| MASt3R-SfM solve ceiling | ≈300 frames / 16 GB GPU | measured (005, OOM @400–500) |
| MASt3R-SfM internal resolution | 512 px | model property |
| MAtCha training resolution cap | 1.6K long edge | config |
| TSDF extraction (default config) | mesh_res 1024, factors [2,8,16]; needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 001 re-extraction |
| Normalize preproc | long_edge 2048, JPEG q95 → −28 % training VRAM | STO-SCN-043 |

---

## Recipe A — Video capture (continuous walkthrough / orbit)

*Scenes processed this way: 001/002-patio (4K mp4), 003-firepit,
004-sky-house, 013-basement (FFV1 mkv 672×376).*

Frame extraction + sharp selection is the **T0 preprocessing step for
videos** (EPI-SCN-CAPTURE).

1. **Probe** — know what you have before touching it:
   ```bash
   ffprobe -v error -select_streams v:0 \
     -show_entries stream=codec_name,width,height,r_frame_rate \
     -show_entries format=duration -of default=noprint_wrappers=1 input/<video>
   ```
2. **Extract ALL frames, lossless**, into the canonical pool:
   ```bash
   mkdir -p input/src
   ffmpeg -v error -i input/<video> -fps_mode passthrough input/src/frame_%05d.png
   ```
   Never fps-subsample at extraction time — selection is the next
   step's job, and it needs the full pool to score.
3. **Sharp-select a CANDIDATE pool** (blur-aware subsampling), sized
   ≤300 (the SfM solve ceiling — 200 is the proven size):
   ```bash
   mkdir -p input/preproc-01-pool-sharp-200
   # write specification.json — contract in select_sharp_frames.py docstring
   python3 real2sim/select_sharp_frames.py \
     scenes/<scene>/input/preproc-01-pool-sharp-200
   ```
   Method: variance-of-Laplacian at 480 px, sharpest frame per N
   uniform temporal windows (supersedes uniform stride after 001's
   NaN-confidence failure on degenerate frames).
4. **Camera positioning (pool SfM)** — pose the candidate pool with a
   MASt3R-SfM `--sfm_only` container solve (`preproc-02-pool-sfm`,
   GPU host). This is *preprocessing*: the poses exist so frames can
   be selected by WHERE THEY ARE, not when they were shot.
5. **Coverage curation (operator-in-the-loop, T-020)** — load the
   posed pool into the Camera Selection Viewer
   (`real2sim/camera_viewer/viewer.py`, viser :8082; STO-SCN-001):
   frustums + image planes + temporal path, filters (time, stride,
   k-means spatial clusters, distance, look-at, pHash dedupe).
   Operator selects the reconstruction subset → `selected_frames.json`
   → feeds MAtCha `--image_idx`. This is "the dtu flow" (dtu-bicycle:
   194-frame pool → 12 curated; 005: pool-sharp-200 → pool-sfm →
   viewer).
   **Why not just pick N sharp frames blind?** That was 001-patio
   (12 sharpness-only frames) — verdict: garbage from coverage gaps.
   Sharpness scoring is coverage-blind; curation over posed cameras
   is the fix. Blind sharp-N is acceptable only as a quick baseline
   run, never as the scene's final selection.
6. **Normalize** — only when source long edge > 2048 (4K video: yes;
   small sources like 013's 672 px: skip — no-op):
   see Recipe B step 2.
7. → **Common trunk** below.

## Recipe B — High-resolution photo stills

*Scenes processed this way: 006/007/008-kubota (5712×4284 JPEG/MPO).*

1. **Pool**: photos at `input/src/`, original filenames.
2. **Normalize** (standard since STO-SCN-043 — −28 % training VRAM,
   also strips MPO depth payloads + applies/strips EXIF rotation):
   ```bash
   mkdir -p input/preproc-01-normalize
   # spec: {"long_edge": 2048, "quality": 95}, inputs: ["input/src"]
   python3 real2sim/normalize_photos.py scenes/<scene>/input/preproc-01-normalize
   ```
3. **Select**: pools ≤300 can be pool-SfM'd whole; then coverage
   curation exactly as Recipe A steps 4–5 (sharp candidate pool first
   when the photo set is larger).
4. → **Common trunk** below.

## Recipe C — Large / multi-session photo pools (e.g. Polycam exports)

*Scene processed this way: 005-meadow (2,028 × 1024×768, three
capture sessions).*

Pools past the 300-frame solve ceiling cannot be solved whole — use
the **photo spine** (chunked solves + gauge stitching):

1. Split per capture session FIRST (timestamp-prefix boundaries) —
   blind chunking across session boundaries is what broke 005's
   chunk-01.
2. `real2sim/batched_sfm.py chunk` (≤300/chunk, 50 overlap) →
   `solve` per chunk (fleet-farmable) → `stitch` (consensus align,
   relative gate, `--order`).
3. Cross-session merge needs retrieval-based content matching — not
   built yet.

Full runbook + parked state: `scenes/005-meadow/FINDINGS-photo-spine-2026-06-10.md`,
epic `effort/sparse/epic-photo-spine-pipeline/`.

## Recipe D — Benchmark datasets (DTU, MipNeRF-360, …)

*Scenes processed this way: dtu-bicycle and friends.*

Go through the repro pipeline runner — `effort/repro-pipeline/`.
Gotchas already paid for: LFS pointers masquerading as inputs (runner
now guards), tool rc=0 lying (expected-outputs hard gate),
`.gitattributes` case drift.

---

## Common trunk (all recipes converge here)

Per run `scenes/<scene>/pipeline-<p>/run-<r>/`:

1. **SfM** — MASt3R-SfM container solve → `transform-…/data/mast3r_sfm/cameras.json`
2. **Train** — MAtCha gaussians (`free_gaussians`, 7k iters)
3. **Mesh** — TSDF extraction, *default* config (the lowmem config
   produces visibly degraded meshes — 001's 320 MB vs 1,351 MB lesson):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python scripts/extract_tsdf_mesh.py -s <mast3r_sfm> -m <free_gaussians> \
     -o <tsdf_meshes> -c default
   ```
   Gate completion on output **freshness** (`-nt` marker), never
   existence — `os.system` swallows the tool's exit code.
4. **Orient** — `orient_mesh.py` / `apply_existing_orientation.py`
5. **Scene build** — `build_blender_scene.py` → `scene.blend`
6. **Views & renders** — `/camera-save` (viewport → `cameras.json`),
   `render_comparison_matrix.sh` → `run-<r>/renders/<view>.png` (+
   settings sidecar)
7. **Rank** — `rate_renders` app (:8090)

Fleet execution: all four GPU hosts (t/b/d/s) pull the matcha image
from the LAN registry `j.pski.org:5000/krabby-matcha:<tag>` (latest:
0.2.2-selfcontained; build+push recipe: `images/matcha/README.md`).
Wrap long jobs with `nanny-progress`. Docker writes land root-owned on fleet clones and
silently wedge `git pull` — chown via throwaway container.

**Gather hygiene (3 wedges on 2026-06-10 alone):** after rsyncing a
host-side output into the store and pushing, **delete that output from
the host clone immediately** — the next `git pull` on that host will
otherwise abort ("untracked working tree files would be overwritten")
because the committed copy collides with the local untracked one. The
abort prints "Updating …" first and looks like success. Sequence:
chown (container) → rsync to store → commit+push → `rm -rf` on host →
host pulls clean.

---

## Phase catalog

One section per processing phase. Each phase has a STOry recording
what we did / where the code is / how (operator directive
2026-06-10). The recipes above are the *flows*; this catalog is the
per-phase *reference*.

### 1. Video → frame pool
Lossless full-frame extraction to `input/src/` (ffmpeg passthrough,
PNG). No script — the two one-liners in Recipe A steps 1–2 are
canonical (`extract_frames.sh` is the legacy COLMAP-era form).
**Story:** STO-SCN-054.

### 2. Photo normalization
`real2sim/normalize_photos.py` — long_edge 2048 JPEG q95, strips MPO
payloads, applies+strips EXIF rotation; −28 % training VRAM. Standard
for sources >2048 px. **Story:** STO-SCN-043.

### 3. Frame sampling (sharp-select)
`real2sim/select_sharp_frames.py` — variance-of-Laplacian @480 px,
sharpest per uniform temporal window; spec-driven, results-emitting.
Builds candidate pools (≤300) and quick baselines.
**Story:** STO-SCN-052 (prototype history: STO-SCN-045 era, 001).

### 4. Camera positioning (pool SfM)
`real2sim/batched_sfm.py chunk`+`solve` (single-chunk) — MASt3R-SfM
`--sfm_only` over the candidate pool so frames can be selected by
WHERE they are. "The dtu flow." **Story:** STO-SCN-055.

### 5. Coverage curation (operator, T-020)
`real2sim/camera_viewer/viewer.py` (viser :8082) — frustums + image
planes + filters (time/stride/k-means/distance/look-at/pHash);
operator picks → `selected_frames.json`. Slot naming: `curated-<N>`.
**Story:** STO-SCN-001.

### 6. Photo spine (pools >300 frames)
`real2sim/batched_sfm.py` chunk/solve/stitch + `real2sim/gauge_align.py`
(consensus align, relative gates, `--order`). Fleet-farmable.
**Stories:** STO-SCN-048/049/050/051 (epic
`sparse/epic-photo-spine-pipeline`); parked 005 state in the scene's
FINDINGS doc.

### 7. MAtCha reconstruction (runner)
`real2sim/run_transform.py` — spec-driven (`specification.json` in,
measured `results.json` out), verified flag mappings, image-label
driven mounts, digest provenance. Image:
`j.pski.org:5000/krabby-matcha:<tag>` (build recipe:
`images/matcha/README.md`). **Stories:** STO-SCN-039/040 (runner),
STO-SCN-038 (self-contained image), STO-SCN-053 (registry + 0.2.2).

### 8. TSDF extraction + gravity orientation
`scripts/extract_tsdf_mesh.py` (in-image, config `default`) →
`orient_mesh.py` (RANSAC floor, STO-SCN-004) →
`apply_existing_orientation.py`. Freshness-gate the outputs; never
lowmem for deliverables; ≥17-camera OOM fixed in image ≥0.2.2.
**Story:** STO-SCN-056.

### 9. Scene build (Blender)
`real2sim/build_blender_scene.py` (headless) — oriented tetra mesh +
vertex-color material + camera objects + image empties →
`run-<r>/scene.blend`. **Story:** STO-SCN-044.

### 10. Choosing the render camera (operator, T-020)
Open the run `scene.blend` (MCP bridge), walk (Shift+`), then
`/camera-save <view-name>` → `viewport_capture.py` captures the
viewport (TRUE lens from projection matrix),
`sync_comparison_views.py` regenerates the scene-level `cameras.json`
(schema 5). **Verify the capture result names the scene you meant —
a stale Blender instance can hold the MCP port** (2026-06-10: a 013
capture landed in 001's blend; restored from git).
**Stories:** STO-SCN-046 (capture skill), STO-SCN-045 (schema).

### 11. Comparison renders
`real2sim/render_comparison_matrix.sh --scene <s> [--views …]
[--mesh-source tsdf]` — Cartesian (variant × view) →
`pipeline-<p>/run-<r>/renders/<view>.png` + `<view>.json` settings
sidecar (engine/resolution/mesh source + the run's transform
parameters). The render is an artifact OF the run's configuration —
that's the thing being compared; per-view aggregation is read-time
(rate_renders). 1920×1080 WORKBENCH.
**Stories:** STO-SCN-045 (matrix), STO-SCN-058 (renders-in-runs +
backfill of all 43 legacy renders via `migrate_renders_into_runs.py`).

### 12. Ranking runoff (operator, T-020)
`real2sim/rate_renders/server.py` (:8090) — aggregates
`pipeline-*/run-*/renders/*.png` by view at read time (URL contract
`/api/render/<scene>/<view>/<variant>.png` unchanged); operator ranks
variants per view; output `rankings.jsonl` per scene. **Commit the rankings —
they are the data.** **Story:** STO-SCN-057.
