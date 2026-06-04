# Scene schema (canonical) — `EPI-SCN-SCENE-SYNC`

**Status:** normative. This is the single source of truth for what a "scene" is
on disk (T-023). `STO-SCN-026` authored it; the story body holds the rationale,
this file holds the spec. Migration (`STO-SCN-033`) targets this shape.

A scene is a **pipeline of transformations** — an auditable lineage from source
to promoted output, where every step records *what it did* and *the exact
environment it ran in*. The outer structure is ours; the inside of each
transform's `data/` is the third-party tool's native output, untouched.

## Layout (normative)

```
<scene-id>/
  scene.toml                               # scene-level manifest (human-authored)
  input/                                   # original source files (immutable once landed)
    <source files…>                        #   raw capture: video / image set
    preproc-<NN>-<slug>/                   # pipeline-AGNOSTIC preprocessing (transform-shaped)
      specification.json
      results.json
      data/
  pipeline-<slug>/                         # one reconstruction approach == an image name
    run-<slug>/                            # ONE parameterised execution of the pipeline
      run.json                             #   the variant identity + promoted flag
      transform-<NN>-<slug>/               #   one ordered step in this run
        specification.json                 #     WHAT was done
        results.json                       #     HOW/WHERE it ran
        data/                              #     this step's output (tool-native inside)
      output/                              #   this run's selected output
    output/                                # the pipeline's PROMOTED run output
  output/                                  # scene-level PROMOTED outputs (public candidate; empty today)
```

**Why the `run-` level:** a pipeline is often run many times with different
hyper-parameters (the `004-sky-house-curated-{12, 12-strong, 12-dense-strong,
…-r3, 16-strong}` sweep is five MAtCha runs of one scene on the *same* 12 input
frames). Runs are *parallel alternatives*, not sequential steps — so they get
their own level. Single-run pipelines still use one run dir (e.g. `run-default`).

### Rules

1. `input/` is immutable once a source lands; transforms read it, never write it.
2. A transform writes only its own `data/`. Steps reference each other's `data/`
   paths in `specification.json` (`inputs`). The dependency graph lives in
   pipeline **code**, not as a declared DAG in the data.
3. `preproc-*` is a pipeline-agnostic transform (carries the spec/results/data
   triple) living under `input/` because many pipelines reuse it.
4. `<scene>/output/` holds only `maturity = "promoted"` artifacts — **empty
   today** (all current work is prototype).
5. Each `data/` holds the tool's **native** output, unchanged (COLMAP
   `database.db`/`sparse/`/`dense/`; MAtCha `mast3r_sfm/`/`tetra_meshes/`/
   `tsdf_meshes/`/`oriented/`; MASt3R `--save-as`; VGGT COLMAP-format `sparse/`).

### Naming

| Token | Form | Example |
|-------|------|---------|
| `<scene-id>` | `NNN-<kebab>` (ext/benchmark may drop the ordinal but MUST set `source=external`) | `004-sky-house`, `dtu-bicycle` |
| `pipeline-<slug>` | `pipeline-` + image name | `pipeline-matcha`, `pipeline-colmap`, `pipeline-mast3r`, `pipeline-vggt`, `pipeline-slam3r` |
| `run-<slug>` | `run-` + kebab variant (`default` if single) | `run-12-dense-strong-r3` |
| `transform-<NN>-<slug>` | literal `transform` + 2-digit order + kebab | `transform-01-matcha`, `transform-02-orient` |
| `preproc-<NN>-<slug>` | `preproc-` + 2-digit + kebab | `preproc-01-frame-select` |

## `scene.toml` (human-authored metadata)

```toml
schema_version = "1"
id           = "004-sky-house"
title        = "Sky House — dining room"
source       = "capture"           # capture | external | synthetic
captured     = "2026-05-02"        # ISO date; omit if unknown/external
tier         = "research"          # research | collab | public (OVERRIDES stage default)
notes        = ""

[scale]                            # see § Scale — coordinates with STO-SCN-016
status          = "uncalibrated"   # uncalibrated | calibrated
method          = ""
meters_per_unit = 0.0

[[pipelines]]                      # one per pipeline-<slug> present
slug         = "matcha"
maturity     = "prototype"         # prototype | promoted
promoted_run = ""                  # the run-<slug> surfaced into pipeline output/ (blank until promoted)
```

Hashes are NOT hand-authored here — they live in each transform's
`results.json` (`outputs[].sha256`), including `preproc-*` (which is also where
`input/` source-file hashes are registered).

## `run.json` (run-level — the variant identity)

```json
{
  "schema_version": "1",
  "pipeline": "matcha",
  "run": "12-dense-strong-r3",
  "params": { "frames": 12, "alignment_config": "strong-r3", "dense_regul": "strong", "encoder": "vitl" },
  "promoted": false,
  "notes": "r-knob sweep — drop finest 0.4 chart level (Option C)"
}
```

This is what the **legacy `manifest.json` becomes** (it is already a run-level
document). `params` is the sweep knobs that distinguish this run; per-step detail
lives in each transform's `specification.json`.

## `specification.json` (per transform — WHAT)

```json
{
  "schema_version": "1",
  "transform": "transform-01-matcha",
  "pipeline": "matcha",
  "run": "12-dense-strong-r3",
  "kind": "reconstruction",
  "description": "MAtCha full pipeline (SfM→align→refine→tetra) on 12 curated frames",
  "inputs": ["input/preproc-01-frame-select/data"],
  "parameters": { "alignment_config": "strong-r3", "dense_regul": "strong", "encoder": "vitl",
                  "sfm_config": "unposed", "image_resolution_long_edge": 1024,
                  "chart_resolutions": [0.05, 0.1, 0.2] },
  "command": "python train.py -s <frames> -o <out> --alignment_config strong-r3 …",
  "maturity": "prototype",
  "story": "STO-SCN-0XX"
}
```

- `inputs` — scene-relative paths read (facts, not a DAG).
- `story` — the STO XID the transform ran under (dual provenance).

## `results.json` (per transform — HOW/WHERE)

```json
{
  "schema_version": "1",
  "transform": "transform-01-matcha",
  "status": "success",                 // success | failed | partial
  "provenance": "measured",            // measured | deduced | unknown
  "started": "2026-05-02T15:00:00-07:00",
  "finished": null,
  "duration_s": 605,
  "host": "tbeeprz",
  "peak_vram_mib": 7212,
  "environment": {
    "os": "Debian 13 (kernel 6.12.74)",
    "gpu": "NVIDIA RTX 5080 / 16 GB",
    "nvidia_driver": "unknown",
    "cuda": "unknown",
    "container": { "image": "krabby-matcha", "tag": "latest+r-knob-sweep", "digest": "unknown" },
    "software": { "matcha": "unknown(git_sha:null)" }
  },
  "outputs": [
    { "path": "data/tetra_meshes/tetra_mesh_binary_search_7.ply", "bytes": 0, "sha256": "unknown" }
  ]
}
```

- `provenance`: **`measured`** (runner-recorded), **`deduced`** (reconstructed
  from journals in `STO-SCN-033`), **`unknown`**. Never fabricate (T-002) — the
  example above honestly marks the gaps in the legacy manifest (`nvidia_driver`,
  `cuda`, container `digest`, `software` versions, output hashes are all unknown).
- `container.digest` is the **reproducibility anchor** (M14 tag+digest scheme);
  the legacy `image: "krabby-matcha:latest + patch"` shows the failure mode a
  moving tag without a digest creates.

## Scale (coordinates with `STO-SCN-016`)

`scene.toml [scale]` is the **single authoritative home** for a scene's metric
scale. No existing scene records one (consistent with STO-SCN-016 being the
unsolved scale-calibration blocker), so every migrated scene starts
`status = "uncalibrated"`. When STO-SCN-016 lands, it writes the result here, and
USD export (T2) **consumes** this block rather than re-deriving scale. Do not
model scale anywhere else.

## Legacy `manifest.json` → schema mapping (for STO-SCN-033)

| `manifest.json` | → schema location |
|-----------------|-------------------|
| `scene` | `scene.toml.id` (canonicalised, e.g. `004-sky-house`) |
| `captured_at` | `scene.toml.captured` |
| `variant_name` | `run.json.run` + run dir slug |
| `frames{count,basenames,selection_method}` | `input/preproc-01-frame-select/` (spec) |
| `matcha{…}` | `transform-01-matcha/specification.json.parameters` |
| `matcha.image` | `…/results.json.environment.container` (tag only; digest `unknown`) |
| `execution{host,gpu,duration_seconds,peak_vram_mib,exit_status}` | `…/results.json` (`host`, `environment.gpu`, `duration_s`, `peak_vram_mib`, `status`) |
| `outputs{…paths}` | `…/results.json.outputs[]` |
| `post_processing{orient,…}` | `transform-02-orient/` etc. (spec/results) |

Provenance for manifest-bearing runs = **measured** (with the noted gaps);
manifest-less runs = **deduced**; raw-only scenes = no transforms, just `input/`.
