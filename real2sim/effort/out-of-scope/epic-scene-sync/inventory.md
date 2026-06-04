# Existing scene inventory (STO-SCN-026)

Audit of `/var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes`
on 2026-06-04 — the work-list `STO-SCN-033` (migration) consumes this. ~50 GB
across 21 directories which collapse to ~10 logical scenes once variants are
separated from scene identity.

## Key finding — the flat dirs conflate scene × pipeline × params

The biggest migration fact: directory names encode three things at once.
`004-sky-house-curated-12-dense-strong-r3` is **scene** `004-sky-house`,
**pipeline** `matcha`, **params** `{frames=12, dense, strong, r3}`. The schema
splits these: one `<scene-id>/`, one `pipeline-matcha/`, and each curated dir
becomes a **transform run** with those params in `specification.json`. Migration
must fan 21 dirs into ~10 scenes.

## Provenance is BETTER than the epic assumed — but uneven

Several curated scenes carry a `manifest.json` that already records params +
runtime env (`execution{host, gpu, duration_seconds, peak_vram_mib,
exit_status}`, `matcha{alignment_config, dense_regul, encoder, sfm_config,
image, …}`, `frames{selection_method, basenames}`). That maps near-1:1 onto
`scene.toml` + `specification.json` + `results.json` → **provenance = measured**
for those. Gaps even there: `git_sha: null`, container is a moving tag
(`krabby-matcha:latest + r-knob-sweep patch`, **no digest**), no output hashes.
Older runs (`mast3r_output/`/`matcha_output/` style) have no manifest →
**deduced** from journals. Raw captures → **n/a** (input only).

> Corrects epic Current State ("provenance for past runs is mostly unrecorded"):
> it's **mixed** — curated MAtCha runs are well-documented; older runs aren't.

## Per-scene inventory

| Dir(s) | Size | Logical scene | Pipelines present | Provenance | Migration note |
|--------|------|---------------|-------------------|-----------|----------------|
| `001-patio-fisheye` | 2.9 G | `001-patio` | colmap(sparse+dense) + mast3r + matcha + vggt + mesh | deduced | The "kitchen-sink" scene — 4 pipelines in one dir; richest fan-out |
| `001-patio-fisheye-vggt`, `…-vggt-tiny` | 0 B | `001-patio` | (images only / empty) | n/a | Empty staging dirs — drop or fold as `input/` of 001 |
| `002-patio-dewarped` | 1.4 G | `002-patio`? | colmap(sparse+dense) | deduced | Dewarped variant — confirm whether its own scene or a `preproc` of 001 |
| `003-firepit-fisheye` | 723 M | `003-firepit` | mast3r + matcha + slam3r + mesh | deduced | Only scene with `slam3r_output/` — exercises the slam3r pipeline |
| `004-sky-house-dining` | 1.2 G | `004-sky-house` | mast3r + matcha + mesh + comparison renders | deduced | The eval/comparison home (`rankings.jsonl`, `comparison_views.json`) |
| `004-sky-house-curated-12` | 5.6 G | `004-sky-house` | matcha (mast3r_sfm→tetra/tsdf→oriented) | **measured** (manifest.json) | One MAtCha run, params `{12}` |
| `004-sky-house-curated-12-strong` | 6.0 G | `004-sky-house` | matcha | **measured** | params `{12, strong}` |
| `004-sky-house-curated-12-dense-strong` | 5.1 G | `004-sky-house` | matcha | **measured** | params `{12, dense, strong}` |
| `004-sky-house-curated-12-dense-strong-r3` | 5.0 G | `004-sky-house` | matcha (+tetra_meshes) | **measured** | params `{12, dense, strong, r3}` — **reference-scene candidate** |
| `004-sky-house-curated-16-strong` | 6.9 G | `004-sky-house` | matcha (+free_gaussians) | **measured** | params `{16, strong}` |
| `005-meadow-house` | 490 M | `005-meadow` | (raw: dated folder + zip) | n/a | INPUT only — no transforms run yet |
| `006-kubota-001` … `012-kubota-007` | 62–150 M ea | `006`–`012` | (raw: `src/`) | n/a | 7 INPUT-only kubota captures — no transforms |
| `dtu-bicycle` | 232 M | `dtu-bicycle` (external) | mast3r_sfm + colmap sparse + reference | deduced | Benchmark dataset — `source = external`; has `reference`/`selected_frames.json` |
| `dtu-bicycle-curated-12-dense-strong` | 2.3 G | `dtu-bicycle` (external) | matcha | **measured** | Benchmark MAtCha run |

## Pipelines observed (→ `pipeline-<slug>`)

- **colmap** (`sparse/`, `dense/`, `images/`) — COLMAP SfM+MVS
- **mast3r** (`mast3r_output/` SLAM; `mast3r_sfm/` SfM front-end feeding matcha)
- **matcha** (`mast3r_sfm/` → `tetra_meshes/`/`tsdf_meshes/` → `oriented/`) — primary
- **vggt** (`vggt_images/`) — only on 001-patio
- **slam3r** (`slam3r_output/`) — only on 003-firepit

## Migration implications for STO-SCN-033

1. **Fan 21 dirs → ~10 scenes**; curated `004-*` (×5) and `dtu-*` (×2) collapse
   into their base scene as multiple `pipeline-matcha` transform runs.
2. **`manifest.json` → schema mapping is mechanical** for the 7 curated scenes
   (measured provenance); write a converter `manifest.json` → `scene.toml` +
   `specification.json` + `results.json`.
3. **Older scenes (001/002/003/004-dining)**: no manifest → reconstruct from
   journals (deduced) or mark `unknown`. Never fabricate (T-002).
4. **Raw-only scenes (005, 006–012)**: trivial — they're just `input/`.
5. **Empty dirs** (`001-*-vggt*` at 0 B): drop or fold.
6. **External**: `dtu-bicycle` sets `source = external` (no ordinal exemption).
7. **Scale**: no scene records a metric scale anywhere — consistent with
   `STO-SCN-016` being unsolved. `scene.toml [scale]` starts `uncalibrated` for all.

## Reference-scene pick

`004-sky-house-curated-12-dense-strong-r3` — it has the richest `manifest.json`
(full execution env + matcha params + outputs + post-processing) and the full
matcha tool-native tree (`mast3r_sfm/ tetra_meshes/ tsdf_meshes/ oriented/`), so
it best demonstrates the whole `scene.toml`/`specification.json`/`results.json`
mapping in the worked example.
