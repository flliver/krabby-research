# Experiment 004 — MASt3R-SfM scaling on sky-house-dining

**Status:** 🟡 in progress (started 2026-05-01T16:55-07:00)
**Pipeline:** MASt3R-SfM only (`python train.py --sfm_only`) inside `krabby-matcha:latest`
**Hardware:** bbeeprz (RTX 5080, 16 GB VRAM), Ryzen 9800X3D
**Reference:** journal note `matcha-quality/notes/2026-05-01T161229-mast3r-sfm-scaling-for-large-candidate-pools` (the planning note this experiment validates), code-read note `matcha-quality/notes/2026-05-01T164453-matcha-source-code-read` (the source pass that confirmed `--sfm_only` exists)

## Question being answered

**How many video frames can MASt3R-SfM realistically position in 3D space on our hardware?** This bounds the candidate-pool size for downstream B5 manual frame curation. The answer is the largest N where `train.py --sfm_only --n_images N` succeeds within reasonable wall-clock and within 16 GB VRAM.

We do not run the full MAtCha pipeline here. Only the SfM stage. Only `cameras.json` matters as the output.

## Method

1. **Frame extraction (one-shot):** 500 evenly-spaced frames from `videos/004-sky-house-dining.mp4` (3:47, 227 sec) at 1024×576. ~30 MB total. Recipe: `scripts/extract_candidates.sh`.
2. **Sweep:** invoke `train.py --sfm_only --n_images N` for N ∈ {24, 60, 120, 200, 300, 500} against the 500-frame candidate dir. Each step:
   - Runs `scripts/sfm_sweep_step.sh <N>` which times the run and polls `nvidia-smi --query-gpu=memory.used` at 1 Hz to capture peak VRAM.
   - Verifies `cameras.json` was written with N camera entries (counts `len(d['filepaths'])`).
   - Stops the sweep at the first failure (OOM, divergence, no output).
3. **Chain runner:** `scripts/sfm_sweep_chain.sh` runs the remaining N values in sequence after N=24 and N=60, stopping on first failure. Records all results to `/tmp/sfm-sweep-results.tsv` on bbeeprz; pulled into `results/` here when complete.

## Configuration

The matcha-build container was already running with the right shape:

- `--shm-size=8g` — verified via `docker inspect`. Critical: per `research/docs/DOCKER_DEPENDENCIES.md` (commit 6b3f855), PyTorch containers without 8 GB shm silently deadlock at 0% GPU.
- `--gpus all` — RTX 5080 visible.
- Mounts: `/data` ← `~/outposts/krabby/data/011-scene-reconstruction/`, `/opt/MAtCha` ← `~/scratch/MAtCha` (bind-mount, so we can edit the source from the host).

MASt3R-SfM config used: `configs/mast3r/unposed.yaml` (bundled MAtCha default). Key params:

- `image_size: 512` — input downscaled to 512 long edge for SfM regardless of source resolution.
- `max_window_size: 20` — N_a (FPS-sampled keyframes for scene graph).
- `max_refid: 10` — k (k-NN per non-anchor).
- `n_coarse_iterations: 1000` and `n_refinement_iterations: 1000` — both optimizer passes.

## Results

Once the sweep completes, the results table will be saved to `results/sweep.tsv` and summarized below.

### Per-step measurements

| N | RC | Wall-clock (sec) | Peak VRAM (MiB) | Cameras returned | Notes |
|---|---|---:|---:|---:|---|
| 24 | 0 | 193 (3:13) | 8885 | 24 | smoke test ✓ |
| 60 | 0 | 415 (6:55) | 10021 | 60 | ✓ — roughly linear vs N=24 (2.5× frames → 2.15× runtime, +1136 MiB peak) |
| 120 | _in flight_ | | | | |
| 200 | pending | | | | |
| 300 | pending | | | | |
| 500 | pending | | | | |

Peak VRAM measurements are *total* GPU memory (not delta). Baseline is ~5 GB held by the envoy + container before the test starts.

### Verdict

(Pending sweep completion. Will record the largest N that succeeded with reasonable wall-clock + VRAM headroom, plus any failure mode at the next step up.)

## Files

- `scripts/extract_candidates.sh` — frame extraction recipe (one-shot per scene).
- `scripts/sfm_sweep_step.sh` — runs SfM for a single N, captures wall-clock, peak VRAM, and verifies output. Lives at `/tmp/sfm-sweep-step.sh` on bbeeprz at runtime.
- `scripts/sfm_sweep_chain.sh` — runs the sweep for the remaining N values sequentially, stopping on first failure.
- `results/` — sweep outputs (TSV table, per-N logs) once complete.

## Output layout on bbeeprz

```
~/outposts/krabby/data/011-scene-reconstruction/
├── frames/
│   └── 004-sfm-scaling-500/         500 candidate frames at 1024×576 (~30 MB)
└── sfm-scaling-out/
    ├── n024/mast3r_sfm/             SfM output for N=24
    │   ├── cameras.json             {filepaths: [24], focals: [24], ...}
    │   ├── images/                  resized inputs
    │   ├── pointmaps/               per-image pointmaps from MASt3R
    │   ├── points.ply               sparse point cloud
    │   └── sparse/0/                COLMAP-format outputs (cameras.txt, images.txt, points3D.txt)
    ├── n060/mast3r_sfm/             ... (in progress)
    └── n120/.../                    ... (pending)
```

## Lessons captured along the way

- **`/usr/bin/time` is missing inside `matcha-build`.** Used `date +%s` for wall-clock. Fixed in the script.
- **`cameras.json` is a single dict with parallel arrays** (`filepaths`, `focals`, ...), not per-camera entries. `len(d.keys())` returns 3, not N. Correct count is `len(d['filepaths'])`. Fixed in the script.
- The bind-mount `/opt/MAtCha → ~/scratch/MAtCha` on bbeeprz means we can edit MAtCha source from the host without rebuilding the container — useful for the future `r` truncation experiment (Option C).

## How to reproduce

On bbeeprz (or wherever the krabby-matcha:latest image and source video live):

```bash
# Sync the scripts dir
rsync -av experiments/004-sfm-scaling-sky-house/scripts/ bbeeprz:/tmp/

# Run frame extraction once (takes ~30 sec)
ssh bbeeprz '/tmp/extract_candidates.sh 500'

# Run the sweep (takes ~2 hours if everything succeeds)
ssh bbeeprz '/tmp/sfm-sweep-chain.sh'

# Pull results back
rsync -av bbeeprz:/tmp/sfm-sweep-results.tsv experiments/004-sfm-scaling-sky-house/results/
```

## Next steps after this experiment

Once we have the ceiling N:

1. **Validates / invalidates the scaling estimates** in the planning note. Either way, we have a measured upper bound.
2. **Determines the working candidate-pool size** for B5 manual curation. Almost certainly less than the ceiling (curation cognitive load), but the ceiling tells us what's safely available.
3. **Doesn't depend on Option C or Option A.** Pure measurement of the SfM-only path.
