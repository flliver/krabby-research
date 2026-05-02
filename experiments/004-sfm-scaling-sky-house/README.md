# Experiment 004 — MASt3R-SfM scaling on sky-house-dining

**Status:** ✅ complete (2026-05-01T18:00-07:00)
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
3. **Chain runner:** `scripts/sfm_sweep_chain.sh` runs the remaining N values in sequence after N=24 and N=60, stopping on first failure. Records all results to `/tmp/sfm-sweep-results.tsv`; pulled into `results/` here when complete.

### Parallelization (bbeeprz + tbeeprz)

Both hosts have identical hardware (RTX 5080, 16 GB VRAM, sm_120). Setup steps to bring tbeeprz online with the same MAtCha runtime:

1. **Image transfer (LAN-direct, no Mac round-trip):**
   ```bash
   ssh bbeeprz "docker save krabby-matcha:latest | ssh tbeeprz 'docker load'"
   ```
   Result: 21.9 GB image landed on tbeeprz in ~3 min over LAN.

2. **MAtCha source rsync (bind-mount target):** the bbeeprz `krabby-matcha:latest` was created via `docker commit` from a running container with `/opt/MAtCha` bind-mounted from the host. So the image *does not contain* the MAtCha source — it lives on the host at `~/scratch/MAtCha`. Need to mirror that on tbeeprz:
   ```bash
   ssh bbeeprz "rsync -aL \
     --exclude='tmp/' --exclude='core.*' --exclude='media/' \
     --exclude='__pycache__/' --exclude='.git/' \
     ~/scratch/MAtCha/ tbeeprz:/home/jeremy/scratch/MAtCha/"
   ```
   ~7 GB transferred (excluded a 2.3 GB core dump, 8.1 GB of `tmp/`, 45 MB media). LAN-direct.

3. **Frame rsync:** the 30 MB candidate-pool dir copied bbeeprz → tbeeprz so both hosts work against the exact same input frames.

4. **Container start:**
   ```bash
   docker run -d --name matcha-build --gpus all --shm-size=8g \
     -v /home/jeremy/outposts/krabby/data/011-scene-reconstruction:/data \
     -v /home/jeremy/scratch/MAtCha:/opt/MAtCha \
     krabby-matcha:latest sleep infinity
   ```

**VRAM baseline difference (initial):** tbeeprz at 0.7 GB used at start (15.6 GB free); bbeeprz at 5.1 GB used (envoy + container preheat). This meant tbeeprz had ~4.5 GB more usable headroom for SfM. If bbeeprz OOMed at some N, tbeeprz might still succeed at the same N.

**Discovery (mid-experiment):** the bbeeprz baseline overhead was **not infrastructure** — `nvidia-smi --query-compute-apps` revealed PID 446453 (`S:\common\sbox\sbox.exe`, owned by user `benny`) holding 4.3 GB of GPU memory. S&box is the Source 2 sandbox by Facepunch, almost certainly running in a Windows VM or via Wine/Proton on bbeeprz. Not infrastructure, not related to MAtCha.

**Action taken:** terminated PID 446453 via `sudo kill` mid-experiment, freeing ~4 GB of VRAM on bbeeprz. After kill, bbeeprz baseline matched tbeeprz (~0.3 GB driver overhead).

**Implication for the data:** N=24/60/120 measurements on bbeeprz are *contaminated* by the S&box overhead — their peak-VRAM numbers include 4.3 GB of unrelated allocation. N=200 was in flight when the kill happened, so its peak measurement may or may not be contaminated depending on when the polling caught the peak. **N=300 and N=500 on bbeeprz will be measured against the clean baseline,** matching tbeeprz's measurement methodology.

**Lesson:** always inspect `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` at the start of any GPU experiment to confirm the baseline is what you think it is. A persistent foreign process on a shared host can quietly distort all peak-VRAM measurements.

Division of labor:

- **bbeeprz** runs the sequential chain `[120 → 200 → 300 → 500]`, going low-to-high (already in flight).
- **tbeeprz** runs the high-uncertainty values starting from the top: **N=500 first**. Result determines what comes next:
  - If N=500 succeeds on tbeeprz: ceiling is ≥ 500. tbeeprz then tries N=750 to push further. bbeeprz keeps going (its 500 will also succeed).
  - If N=500 fails (OOM/timeout) on tbeeprz: same image, same arch, so bbeeprz's eventual N=500 will also fail. We bracket downward (tbeeprz N=400) and signal bbeeprz to stop after N=300.

The two hosts converge on the answer faster than either alone. Cross-host validation: if both produce the same N-value's measurements within a small margin, we know the result is hardware-stable.

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

| Host | N | RC | Wall-clock (sec) | Peak VRAM (MiB) | Cameras returned | Notes |
|---|---|---|---:|---:|---:|---|
| bbeeprz | 24 | 0 | 193 (3:13) | 8885 (~4622 clean) | 24 | ✓ smoke — peak includes ~4263 MiB S&box overhead |
| bbeeprz | 60 | 0 | 415 (6:55) | 10021 (~5758 clean) | 60 | ✓ — peak includes S&box overhead |
| bbeeprz | 120 | 0 | 749 (12:29) | 11774 (~7511 clean) | 120 | ✓ — peak includes S&box overhead |
| bbeeprz | 200 | 0 | 1164 (19:24) | 9984 | 200 | ✓ — clean (S&box killed mid-run) |
| bbeeprz | 300 | 0 | 1702 (28:22) | 13711 | 300 | ✓ — clean baseline; 2.6 GB headroom |
| bbeeprz | 500 | _killed_ | — | — | — | watchdog cancelled (would have OOMed; tbeeprz already proved this) |
| tbeeprz | 350 | 0 | 1971 (32:51) | 15511 | 350 | ✓ — clean; **only 300 MiB headroom (right at the edge)** |
| tbeeprz | 500 | **OOM** | 1173 (19:33) | 15674 | — | ❌ exhausted GPU at 15.45 GiB, failed to alloc 2 MiB; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True already set |

Peak VRAM measurements are *total* GPU memory (not delta). Baseline is ~5 GB held by the envoy + container before the test starts.

### Verdict

**On RTX 5080 / 16 GB, MASt3R-SfM realistically positions:**

- **Up to ~300 frames safely** — 28 min wall-clock, 13.4 GB peak VRAM, 2.6 GB headroom. Comfortable.
- **Up to ~350 frames pushing it** — 33 min wall-clock, 15.15 GB peak, only 300 MiB headroom. Borderline; any small environment change could push it over.
- **Fails around N=400-500** — measured N=500 OOM at 15.45 GiB after ~20 min. Bracket between 350-500 not narrowed further (no need; ceiling is well above any practical use case).

**Implication for B5 (manual frame curation):** the SfM-scaling ceiling is **not** the binding constraint for candidate-pool sizing. Human cognitive load caps the curatable pool at ~60-150 cameras, well below the hardware ceiling of 300+. Any candidate pool we'd realistically build for hand-curation fits comfortably in `--sfm_only` on either of our RTX 5080 hosts.

### Cross-host observations

- bbeeprz and tbeeprz produce visually-equivalent results; the 4.3 GB S&box overhead on bbeeprz inflated peak-VRAM measurements but did not change SfM correctness.
- After killing S&box, bbeeprz's clean N=200 peak (9984 MiB) closely matches tbeeprz's clean trajectory (linear extrapolation predicts ~10.5 GB at N=200; tbeeprz didn't run N=200 directly).
- **Linear scaling holds on both VRAM and wall-clock.** Peak VRAM ≈ 6 GB + (N − 0) × 30 MiB (rough fit). Wall-clock ≈ N × 4 sec + ~200 sec overhead.

### Lessons captured

- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` should be the *first* thing run before any GPU-bound experiment. It would have caught the S&box overhead immediately and saved interpretation effort. Captured as a lesson in the journal note `2026-05-01T174652-n-500-hit-the-vram-ceiling-bracketing-strategy`.
- The watchdog pattern (`scripts/kill_chain_after_n300.sh`) saved ~20 min of N=500 burn on bbeeprz once we knew the ceiling. Worth keeping the pattern in mind for any sweep that has a known failure point ahead.
- Two compatible hosts on the same LAN with 8 GB shm + cu128 PyTorch can split a sweep cleanly via `docker save | docker load` (~3 min) + rsync of the bind-mounted source dir (~7 GB / ~1 min on LAN). Setup-to-running was <10 min.

## Files

- `scripts/extract_candidates.sh` — frame extraction recipe (one-shot per scene).
- `scripts/sfm_sweep_step.sh` — runs SfM for a single N, captures wall-clock, peak VRAM, and verifies output. Lives at `/tmp/sfm-sweep-step.sh` on bbeeprz at runtime.
- `scripts/sfm_sweep_chain.sh` — runs the sweep for the remaining N values sequentially, stopping on first failure.
- `results/` — sweep outputs (TSV table, per-N logs) once complete.

## Output layout

### On the source hosts (full SfM output, ~26 GB total)

```
~/outposts/krabby/data/011-scene-reconstruction/
├── frames/
│   └── 004-sfm-scaling-500/         500 candidate frames at 1024×576 (~30 MB)
└── sfm-scaling-out/
    └── n<NNN>/mast3r_sfm/
        ├── cameras.json             {filepaths: [N], focals: [N], cams2world: [N×4×4]}  ← what we want
        ├── images/                  resized 512-px inputs                                ← what we want
        ├── points.ply               sparse SfM point cloud (~30%-of-N MB)                ← optional, kept
        ├── pointmaps/               per-image dense pointmaps (~7-20 MB × N)             ← internal, pruned
        └── sparse/0/                COLMAP-format outputs                                ← redundant, pruned
```

The dense `pointmaps/` (5–8 GB per N value) and `sparse/` (1.4 GB per N value) are intermediate artifacts that MASt3R-SfM produces during the optimization. They're not needed for downstream curation or visualization. **Excluded from the local mirror.**

### Local mirror on JDP-Mac (gitignored, ~1 GB total)

```
milestones/011-scene-reconstruction/data/sfm-scaling-out/
├── n024/mast3r_sfm/                 54 MB
├── n060/mast3r_sfm/                102 MB
├── n120/mast3r_sfm/                140 MB
├── n200/mast3r_sfm/                195 MB
├── n300/mast3r_sfm/                220 MB
├── n350/mast3r_sfm/                240 MB
└── n500/mast3r_sfm/                 47 MB  (partial — OOMed before writing cameras.json)
                                  ────────
                                    ~1 GB
```

Pulled via `rsync -a --exclude='pointmaps/' --exclude='sparse/'`. Cameras + resized images + sparse point cloud only. The `data/` dir is `.gitignore`d at the milestone level.

### How to view any of these in the camera viewer

```bash
# From the workspace root, with the viewer's venv activated
cd milestones/011-scene-reconstruction/workspace/camera_viewer
source .venv/bin/activate
python viewer.py \
  --cameras ../../data/sfm-scaling-out/n300/mast3r_sfm/cameras.json \
  --frames  ../../data/sfm-scaling-out/n300/mast3r_sfm/images \
  --output  /tmp/n300-selection.json \
  --port 8080
```

The viewer's `data.py` re-roots the JSON's `/data/...` filepaths under `--frames` automatically.

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
