# Data Recipes — capture types, and precisely how to process them

> One recipe per *kind of input data*. Every step names the hardened
> tool that performs it (T-025 — no freelance re-implementation). All
> recipes converge on the same trunk after preprocessing. Grounded in
> scenes actually processed (T-010); when a new data type shows up,
> add a recipe here as part of processing it.

> **⚠ LAYOUT MIGRATED TO v4 (2026-06-11, HUG-SCN-005 / STO-SCN-080).**
> The store is now content-addressed — see § "Storage policy —
> store-shape v4" below for the layout. **The current end-to-end
> pipeline is the v4 node-graph trunk — see § "v4 pipeline" directly
> below.** The per-data-type Recipes A–D + the MAtCha-era Phase catalog
> further down remain validated PRE-v4 know-how (their literal
> `input/src` / `pipeline-*/run-*` paths are pre-v4); translate via the
> v4 task defs (`real2sim/tasks/`) + graphs (`real2sim/graphs/`).

## Hard limits (apply to every recipe)

| Limit | Value | Source |
|-------|-------|--------|
| MASt3R-SfM solve ceiling | ≈300 frames / 16 GB GPU | measured (005, OOM @400–500) |
| MASt3R-SfM internal resolution | 512 px | model property |
| MAtCha training resolution cap | 1.6K long edge | config |
| TSDF extraction (default config) | mesh_res 1024, factors [2,8,16]; needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 001 re-extraction |
| Normalize preproc | long_edge 2048, JPEG q95 → −28 % training VRAM | STO-SCN-043 |

---

## v4 pipeline — the scene-ingestion trunk (current)

The content-addressed v4 store (HUG-SCN-005) runs as a **node graph**: each stage is a
`python3 real2sim/v4exec.py <cmd> …` that MATERIALIZES a content-addressed artifact —
identity = `hash(resolved inputs + tunable/frozen settings + algo@version)`, so re-running an
existing identity is a **NOOP**. Per-stage canonical spec (settings, ranges, algo@version):
`real2sim/tasks/<name>.json`; DAGs: `real2sim/graphs/*.json`. Numbers live in the task defs,
not here (T-023). GPU stages take `--host <gpu>` (the fleet — t/b/d/s beeprz); the rest run
locally.

End-to-end, a capture becomes a verified reconstruction:

| # | Stage | Command (representative flags) | Does | Task def · Story |
|---|-------|--------------------------------|------|------------------|
| 0 | **Capture decl** | author `scenes/<scene>/capture.json` | `{make, model, mode (fisheye\|dewarped), modality (hyperlapse\|video\|photos)}` — drives the solver & undistort | STO-SCN-091/093 |
| 1 | **Ingest** | `v4exec ingest <scene> --host <gpu> [--raw <path>] [--capture-mode …]` | video/photos → content-addressed image pool (`images/<hash>/`) + per-image metadata | (ingest graph) · STO-SCN-091 |
| 2 | **Pre-cull** | `v4exec precull <scene> [--set-primary]` | pose-free **sharpness + pHash-dedup** → curated subset ≤ solve ceiling; *preserves revisits* (loop-closure gold) | `precull-subset` · STO-SCN-092 |
| 3 | **Spine segment** ◇ | `v4exec spine <scene> [--cap 300 --overlap 30]` | long pools only: chunk the trajectory into **M overlapping segments** + loop candidates (pHash); emits `spine.json` (boundary_spec + camera_model) | `spine-segment` · STO-SCN-097 |
| 4 | **Solve** | `v4exec solve <scene> --host <gpu> [--subset <id>]` | GPU **FastMap** (camera-model-correct; fisheye→pinhole first) → `sparse/0` poses | `solve-fastmap` · STO-SCN-093 |
| 5 | **Covis** | `v4exec covis <scene> --host <gpu> --solve <id>` | co-visibility graph + **validity gate** (HARD-FAIL on a nebula — bad solve never reaches selection) | `covis` · STO-SCN-093 |
| 6 | **Select** | `v4exec select <scene> --solve <id> --covis <id> [--selector voxel --n 24]` | best-N: `voxel` (coverage-flux, default — angular variety) or `track` (covisibility). Emits the report + the **FINAL-N subset** (the handoff) | `select` · STO-SCN-094/103 |
| 7 | **Scout** | `v4exec scout <scene> --host <gpu> --solve <id> [--selector voxel\|track --n-scout N]` | DA3 `da3@1` scout gaussian **in the solve gauge** + `scout_gauge.json` (gs→solve registration, the 105 fix) | `scout` · STO-SCN-095/105 |
| 8 | **Verify** (operator) | `verify_viewer/build_verify.py <scene> --solve <id> --scout <id> [--selector voxel --cull-expand E]` → `viewer.html` (:8099) | splat + proposed-N frustums + **voxel-coverage faces** (red→green) + **gravity-aligned cull box** + WASD-fly + optional **DA3-mesh layer**; operator confirms / overrides (T-020) | STO-SCN-095/103 |
| | *— multi-segment (spine) only ◇ —* | | | |
| 9 | **Spine register** ◇ | `v4exec spine-register <scene> --spine <id> --solves seg=<sub>/<solve>,…` | SIM(3) **pose graph** over the segments → one global gauge (drift-corrected) + per-seam residuals (`global.json`) | `spine-register` · STO-SCN-098 |
| 10 | **Spine fuse** ◇ | `v4exec spine-fuse <scene> --spine <id> --register <id> --solves … --gaussians seg=<ply>,…` | **confidence-weighted** fusion of per-segment gaussians (overlap cross-fade, no doubled walls) → one cohesive `.ply` | `spine-fuse` · STO-SCN-099 |
| 11 | **Whole-spine verify** ◇ (operator) | `verify_viewer/build_spine_verify.py …` → `spine_viewer.html` (:8100) | assembled gaussian + segment-coloured frustums + seam frames + trajectory; operator confirms cohesion (T-020) | STO-SCN-100 |
| | *— reconstruct (downstream) —* | | | |
| 12 | **Reconstruct** | `v4exec reconstruct-matcha\|reconstruct-da3 <scene> --host <gpu> [--sfm posed\|unposed]` | the selection → mesh/gaussian. **Consumes the subset designated `primary`** (no `--subset` flag yet — point `primary` at the FINAL-N subset to reconstruct the selection) | `represent-via-{matcha,da3}` · STO-SCN-013 / EPI-SCN-FEEDFORWARD-RECON |
| 13 | **DA3 mesh from npz** | `da3_mesh_from_npz.py <results.npz> <out.ply>` | TSDF-fuse DA3's posed depths → mesh (scene geometry) in the npz/solve gauge — the "DA3 scene" from any scout/da3 run | (reuses `da3_tsdf_mesh.py` core) |

**◇ = spine-only.** For a **single tractable space (M=1)** skip stages 3 + 9–11 entirely (the
spine machinery no-ops) — pre-cull → solve → covis → select → scout → verify → reconstruct.

**Key invariants threaded through the trunk:**
- **Gauges.** The solve gauge is the reference frame. DA3 gaussians live in DA3's *normalized*
  frame (off by scale + ~125° rotation + translation) and are registered gs→solve via the
  `scout_gauge.json` Umeyama-of-predicted-poses (STO-SCN-105;
  `knowledge/da3-gsply-normalized-frame.md`). Spine submaps register segment-solve→global via
  the SIM(3) pose graph (STO-SCN-098). Fusion composes **105 ∘ 098** (gs→solve→global).
- **Gravity.** `gauge_up` recovers up ⟂ the camera-right axes; the verify cull box + ground
  grid are gravity-aligned (vertical crops tighter than horizontal).
- **The covis validity gate** is the quality firewall: a nebula solve HARD-FAILs and never
  reaches selection/scout.

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

> **This catalog is the MAtCha-era (pre-v4) per-phase reference.** The
> **current** end-to-end sequence is the **§ "v4 pipeline" table** near
> the top (the `v4exec` node graph). Several phases below map onto v4
> stages: sharp-select+curation (3,5) → `precull`+`select` (voxel,
> STO-SCN-103); pool SfM (4) → `solve`+`covis` (FastMap, STO-SCN-093);
> photo spine (6) → `spine`/`spine-register`/`spine-fuse`
> (STO-SCN-097/098/099); DA3 (13) → `scout`/`reconstruct-da3` +
> `da3_mesh_from_npz`. Kept for the validated detail + history.

> **Machine-readable canonical form (v4, HUG-SCN-005):**
> `real2sim/tasks/*.json` (task defs — settings classified
> tunable/frozen/pin, algo@version, license flags) +
> `real2sim/graphs/*.json` (graph defs). Read-side:
> `python3 real2sim/studio_model.py scan` / `v4core.py`.
> The prose below is the narrative; ranges and execution facts live
> in the task defs (T-023 — don't duplicate numbers here).

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

### 13. Feed-forward reconstruction (DA3) — EVALUATION
`krabby-da3` image (`j.pski.org:5000/krabby-da3:0.1`; DA3
GIANT-LARGE baked, **CC BY-NC — research evaluation only**). One
forward pass: images → poses (COLMAP bins) + dense depth + gaussian
splats + GLB cloud, ~21 s for 8 views @ 11 GB VRAM. Gaussian branch
needs the API driver (`infer_gs=True` — the auto CLI doesn't expose
it); multi-format export is dash-joined (`glb-npz-gs_ply-colmap`).
**Mesh path (the deliverable):** `real2sim/da3_tsdf_mesh.py` —
Open3D TSDF fusion over DA3's conf-thresholded depths + poses,
similarity-aligned into the scene's oriented frame
(`real2sim/da3_render_view.py` shares the alignment; both gate on
residual ≤10% of camera spread). Writes the standard
`tsdf_meshes/multires_tsdf_post_oriented.ply` so
`render_comparison_matrix.sh --mesh-source tsdf` treats the run as a
normal variant (copy the matcha run's two anchor JSONs into the
transform dir — the mesh lives in that frame by construction).
CPU-only, seconds. **Stories:** STO-SCN-059 (research), STO-SCN-060
(pilot), STO-SCN-061 (alignment + renders); epic
EPI-SCN-FEEDFORWARD-RECON.

---

## Storage policy — store-shape v4 (HUG-SCN-005, STO-SCN-080, 2026-06-11)

**The store is content-addressed.** Canonical spec: HUG-SCN-005
(`real2sim/effort/repro-pipeline/guidance/`). In brief:

```
scenes/<scene>/
├── videos/<name>/video.<ext>
├── images/<image_hash>/image.<ext> + metadata.json     # canonical pool
├── images/subsets/<HOH>/subset.json                    # subsets (content-only identity)
│   └── cameras/<solve_id>/{cameras.json, points.ply}   # solves
│       └── orient/<orient_id>/{transform,oriented}.json# THE gauge
├── images/subsets/primary -> <HOH>                     # ref (mutable)
├── views/<slot>/view.json                              # scene-global views
├── viewset/canonical/views.json                        # mutable member list
├── represent/<kind>/<RID>/…                            # representations
│   └── meshify/<method>/<MID>/mesh.ply                 # meshes (nested derivation)
│       └── condition/<CID>/mesh.ply
│           └── renders/<REND>/render.png               # keyed on VIEW, never the set
├── scores.jsonl                                        # operator judgments on identities
└── jobs/<ts>-<id>/job.json                             # what each invocation DID
```

- IDENTITY = hash(resolved inputs + tunable+frozen settings +
  algo@version); jobs MATERIALIZE (existing identity → NOOP).
- Tracked: inputs, all metadata, finals (mesh.ply, render.png,
  gs_ply). Untracked transients (free_gaussians, exports, …) live on
  the Mac archive; the store .gitignore IS the policy.
- License eligibility is DERIVED by ancestry walk (CC-BY-NC anywhere
  upstream → not deliverable).
- v2 history: see git log before 2026-06-11 (STO-SCN-062/063 era);
  legacy provenance preserved per-identity as origin-* files +
  `migrated: true` metadata.

**The store tracks lineage and deliverables, not intermediates.**

| Class | Tracked in git/LFS? | Lives where |
|-------|--------------------|-------------|
| INPUTS (video, `input/src/` frames/photos) | YES | hub (j, bare) |
| Metadata (every `*.json`: specs, results, cameras, sidecars, rankings) | YES | hub |
| FINAL outputs (`multires_tsdf_post_oriented.ply`, `gs_ply/`, `renders/`) | YES | hub |
| Transformation/preprocessor `data/` payloads | NO | Mac archive `/var/krabby/scenes` (location stanza in each `results.json`) |
| Derived blends (`scene.blend`, …) | NO | Mac archive; regenerate via these recipes |
| Fleet job scratch | NO | producing host, deleted after gather |

**Tooling provenance (operator policy, 2026-06-10):** results are
INVALID if produced by tools copied to a host ad hoc (/tmp scp). Every
in-container transform must run tools BAKED into the image
(`/opt/krabby-tools` in krabby-da3; the image digest in results.json
then covers the tool versions; `io.krabby.*.tools_git_sha` labels the
source commit). Build+push is ~1 min via the registry — bake, push,
pull, run. A `/tools` bind-mount is permissible ONLY for development
iteration, and any result it produces must be re-produced from a baked
image before it counts.

Rules of thumb:
- The `.gitignore` at the store root IS the policy — if your new
  artifact class is heavy and regenerable, add it there with the
  pattern + a negation for its metadata.
- Every untracked payload must be reachable: `results.json` →
  `transient_data.location`, or regenerable from a recipe above.
- Fleet hosts never retain transients (auto-sync retired,
  STO-SCN-030); jobs `git lfs pull --include=<paths>` what they need
  and clean up after gather.
