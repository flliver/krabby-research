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
├── cameras.json                # scene-level unified views (schema 5)
└── comparison_renders/         # <view>/<variant>.png for rate_renders
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
3. **Sharp-select** (blur-aware subsampling — the step that decides
   reconstruction coverage). Mint a preproc transform and run the
   hardened script:
   ```bash
   mkdir -p input/preproc-01-frame-select-sharp-<N>
   # write specification.json — contract in select_sharp_frames.py docstring
   python3 real2sim/select_sharp_frames.py \
     scenes/<scene>/input/preproc-01-frame-select-sharp-<N>
   ```
   Method: variance-of-Laplacian at 480 px, sharpest frame per N
   uniform temporal windows (supersedes uniform stride after 001's
   NaN-confidence failure on degenerate frames).
   **Frame budget:** 12 proved too few for a yard-scale scene
   (001-patio verdict — coverage, not extraction, was the root cause).
   Start at **24 for room-scale and larger**; sweep upward
   (`preproc-02-…-36`, …) if reconstruction coverage is poor.
4. **Normalize** — only when source long edge > 2048 (4K video: yes;
   small sources like 013's 672 px: skip — no-op):
   see Recipe B step 2.
5. → **Common trunk** below.

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
3. **Select** if pool > what the scene needs; pools ≤300 can go to
   SfM whole. Sharp-select (Recipe A step 3) works on photo pools too.
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
   `render_comparison_matrix.sh` → `comparison_renders/`
7. **Rank** — `rate_renders` app (:8090)

Fleet execution: all four GPU hosts (t/b/d/s) run the matcha image;
s + d have `krabby-matcha:0.2.1-selfcontained`. Wrap long jobs with
`nanny-progress`. Docker writes land root-owned on fleet clones and
silently wedge `git pull` — chown via throwaway container.
