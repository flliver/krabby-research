---
xid: STO-SCN-026
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-04
depends-on: []
bd-id: krabby-l10
assignee: principal
title: Define pipeline-of-transformations scene schema (input/pipeline/output + per-transform provenance) & inventory existing scenes
---

# Define the pipeline-of-transformations scene schema & inventory existing scenes

## Summary

The single documented contract that says what a "scene" is on disk: a
pipeline-of-transformations layout (`input/` → `pipeline-<slug>/transform-*/` →
`output/`), the `scene.toml` / `specification.json` / `results.json` field
definitions, machine-checkable JSON Schemas to validate them, one worked
reference scene, and an inventory of the ~21 existing scenes mapped onto the
model.

## Context

Foundational story of `EPI-SCN-SCENE-SYNC`. Everything downstream consumes this
contract: migration (`STO-SCN-033`) moves data *into* it, tiering (`STO-SCN-027`)
and S3 layout (`STO-SCN-028`) key off its stages, the sync CLI (`STO-SCN-029`)
diffs at its transform-directory grain, and the Docker runner (`STO-SCN-031`)
emits the `results.json` this story specifies. Today the ~21 scenes under
`/var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes` have no
shared shape (`004-sky-house-dining` carries `mast3r_output/ matcha_output/…`;
`009-kubota-004` has only `src/`) — this story ends that.

**This is a spec + inventory story, not a migration story.** No production data
moves here (that's `STO-SCN-033`).

## Problem

There is no definition of a scene. Without one: provenance is prose, every host
re-pulls whole scenes, tiers can't be assigned, and a collaborator can't
reproduce an artifact. The deliverable is a contract precise enough to validate a
directory mechanically and to drive every other story in the epic.

## Design

### Approach

Document the schema as `SCHEMA.md` in the epic directory plus committed JSON
Schemas for the two per-transform JSON files, lay out one real scene as a
conforming reference, and produce a migration-feeding inventory of the existing
scenes. Constrained by the epic's Decisions: dependency DAG lives in **code**
not data (so `specification.json` carries inputs but no graph format); provenance
is **dual** (`results.json` env + the STO XID in `specification.json`); maturity
is explicit (`prototype` → `promoted`) and `output/` holds only promoted.

### The layout (normative)

```
<scene-id>/
  scene.toml                          # scene-level manifest (human-authored)
  input/                              # original source files (immutable once landed)
    <source files…>                   #   raw capture: video / image set
    preproc-<NN>-<slug>/              # pipeline-AGNOSTIC preprocessing, reused by many pipelines
      specification.json              #   (a preproc step is a transform; same triple)
      results.json
      data/
  pipeline-<slug>/                    # one reconstruction approach, e.g. pipeline-mast3r-matcha
    transform-<NN>-<slug>/            # one ordered step within the pipeline
      specification.json              #   WHAT was done
      results.json                    #   HOW/WHERE it ran
      data/                           #   this step's output artifacts
    output/                           # this pipeline's final, selected output
  output/                             # scene-level PROMOTED outputs only (empty until promotion)
```

**Rules:**
1. `input/` is append-only/immutable once a source lands — transforms read it,
   never write it.
2. A transform's `data/` is the *only* place that transform writes. Steps
   communicate by referencing each other's `data/` paths in `specification.json`.
3. `preproc-*` lives under `input/` because it is pipeline-agnostic, but it is
   transform-shaped (carries the `specification.json`/`results.json`/`data/`
   triple) so its provenance is captured identically. *(Interpretation of the AID
   sketch — flag in review.)*
4. `output/` (scene level) contains only `maturity = "promoted"` artifacts.
   **It is empty today** — all current work is prototype.

### Alignment with existing code, tools & canon (non-negotiable boundaries)

This schema is **prototype-redesign-friendly on the outside, fixed on the inside.**

- **Data is not code.** The scene store holds DATA only, in a configured host data
  directory (synced via S3 cold + LAN hot), mounted into containers as
  `-v <host-data>:/data`. **No scene data — meshes, USD, point clouds — is ever
  committed to git.** This retires the prototype `environments/reconstructed/`-in-git
  and the `>100 MB→S3 / ≤100 MB→git` split (both M11-prototype, created 2026-05-18,
  superseded here).
- **`pipeline-<slug>` ↔ our image names.** Pipelines are the reconstruction images
  we already build: `colmap` (scene-reconstruction-base), `mast3r`, `matcha`,
  `vggt`, `slam3r`. Transform steps map to the existing `real2sim/run_*.sh` stage
  runners. Reuse, don't reinvent (T-013).
- **Each `transform-NN/data/` holds the third-party tool's NATIVE output,
  unchanged.** COLMAP (`database.db`/`sparse/`/`dense/`), MAtCha (`mast3r_sfm/`,
  `tetra_meshes/`, `oriented/`), MASt3R-SLAM (`--save-as …`), VGGT (COLMAP-format
  `sparse/`) define their own internal layouts — **we did not build these and do
  not reorganize them.** The schema governs only the OUTER structure; inside
  `data/`, the tool's convention rules.
- **Canon container contract.** Code baked at `/workspace`; data bind-mounted at
  `/data` — how our *delivered* locomotion/isaacsim images already work. Don't diverge.

### Naming (normative)

| Token | Form | Example | Notes |
|-------|------|---------|-------|
| `<scene-id>` | `NNN-<kebab>` | `004-sky-house` | Zero-padded ordinal + slug. External/benchmark sets (e.g. `dtu-bicycle`) may omit the ordinal but MUST set `source = "external"`. |
| `pipeline-<slug>` | `pipeline-<kebab>` | `pipeline-mast3r-matcha` | Names the approach, not a run. Param variants are transforms, not new pipelines. |
| `transform-<NN>-<slug>` | literal `transform` + 2-digit ordinal + kebab | `transform-02-dense-mast3r` | Lowercase, consistent with `pipeline-`/`preproc-`. `NN` is execution order within the pipeline. |
| `preproc-<NN>-<slug>` | `preproc-` + 2-digit + kebab | `preproc-01-frame-select` | Ordinal orders shared preprocessing. |

### `scene.toml` (scene-level manifest — human-authored)

```toml
schema_version = "1"
id           = "004-sky-house"
title        = "Sky House — dining room"
source       = "capture"          # capture | external | synthetic
captured     = "2026-04-29"        # ISO date; omit if unknown/external
tier         = "research"          # research | collab | public — OVERRIDES the stage default
notes        = ""

[scale]                            # ties to STO-SCN-016 (scale calibration)
status         = "uncalibrated"    # uncalibrated | calibrated
method         = ""                # how it was determined, when calibrated
meters_per_unit = 0.0              # 0 until calibrated

[[pipelines]]                      # one entry per pipeline-<slug> present
slug     = "mast3r-matcha"
maturity = "prototype"             # prototype | promoted
status   = "complete"              # planned | running | complete | abandoned
```

> Content hashes are **not** hand-authored in `scene.toml` (curated metadata
> only). They live per-output in each transform's `results.json`
> (`outputs[].sha256`), **including `preproc-*` steps** — and a `preproc-*` step
> is where `input/` source-file hashes get registered (input files have no
> `results.json` of their own). The sync CLI (`STO-SCN-029`) reads hashes from
> these records rather than maintaining a separate manifest.

### `specification.json` (per transform — "WHAT was done")

```json
{
  "schema_version": "1",
  "transform": "transform-02-dense-mast3r",
  "pipeline": "mast3r-matcha",
  "kind": "dense-reconstruction",
  "description": "Dense MASt3R reconstruction from curated frames",
  "inputs": [
    "input/preproc-01-frame-select/data",
    "pipeline-mast3r-matcha/transform-01-sfm/data"
  ],
  "parameters": { "resolution": 12, "strong": true },
  "command": "matcha reconstruct --strong …",
  "maturity": "prototype",
  "story": "STO-SCN-0XX"
}
```

- `inputs` — scene-relative paths this step read (the dependency *facts*, not a
  DAG format; resolution is code's job).
- `parameters` — free-form, tool-specific knobs (the recipe).
- `story` — **the STO XID this transform was run under** (dual-provenance link).

### `results.json` (per transform — "HOW/WHERE it ran")

```json
{
  "schema_version": "1",
  "transform": "transform-02-dense-mast3r",
  "status": "success",
  "provenance": "measured",
  "started": "2026-05-01T17:46:52Z",
  "finished": "2026-05-01T18:02:10Z",
  "duration_s": 918,
  "host": "tbeeprz",
  "environment": {
    "os": "Debian 13 (kernel 6.12.74)",
    "gpu": "NVIDIA RTX 5080",
    "nvidia_driver": "550.xx",
    "cuda": "12.4",
    "container": { "image": "krabby/real2sim", "tag": "2026-05-01", "digest": "sha256:…" },
    "software": { "matcha": "x.y.z", "mast3r": "…", "python": "3.11.9" }
  },
  "outputs": [
    { "path": "data/points.ply", "bytes": 51234567, "sha256": "…" }
  ]
}
```

- `provenance` — **`measured`** (recorded by the runner at run time),
  **`deduced`** (reconstructed from journals during `STO-SCN-033`), or
  **`unknown`**. This is the migration-honesty field (T-002): never fabricate a
  driver version; mark it `unknown`.
- `status` — `success | failed | partial`.
- `environment.container` is the exact contract `STO-SCN-031`'s Docker runner emits.

### Tier & maturity defaults

| Location | Default tier | Maturity |
|----------|-------------|----------|
| `input/`, `input/preproc-*` | research | n/a |
| `pipeline-<slug>/transform-*` | research | as marked (`prototype` default) |
| `pipeline-<slug>/output/` | collab | as marked |
| scene `output/` | public | `promoted` only |

`scene.toml`'s `tier` overrides the default (e.g. a sensitive capture stays
research even when promoted-quality).

### Changes

| File | Change |
|------|--------|
| `…/epic-scene-sync/SCHEMA.md` | add — normative spec (this Design, formalized) |
| `…/epic-scene-sync/schemas/specification.schema.json` | add — JSON Schema |
| `…/epic-scene-sync/schemas/results.schema.json` | add — JSON Schema |
| `…/epic-scene-sync/schemas/scene.schema.json` | add — `scene.toml` field schema |
| `…/epic-scene-sync/reference/<scene-id>/…` | add — one worked reference scene (structure only, small/sample data) |
| `…/epic-scene-sync/inventory.md` | add — existing-scene → model mapping (feeds STO-SCN-033) |

## Definition of Done

- [ ] `SCHEMA.md` committed: layout, naming, `scene.toml`, `specification.json`,
      `results.json`, tier/maturity rules — each with a field table.
- [ ] JSON Schemas committed for `specification.json`, `results.json`,
      `scene.toml`; a `valid` fixture passes and a deliberately-`invalid` one fails.
- [ ] One existing scene laid out as a conforming **reference example** (no
      production data moved — sample/links only).
- [ ] `inventory.md` maps every existing scene (~21) to its detected
      pipeline(s)/transform(s) and rates recoverable provenance
      (`measured`/`deduced`/`unknown`) — the work-list `STO-SCN-033` consumes.
- [ ] `scale` block reconciled with `STO-SCN-016` (scale calibration) — aligned or
      the gap explicitly flagged.
- [ ] Epic `Design` updated to point at `SCHEMA.md` as the normative source (this
      story's body becomes the rationale; `SCHEMA.md` becomes canonical — T-023).

## Testing

### Unit / fixture tests

- [ ] A conforming `specification.json` / `results.json` / `scene.toml` validates.
- [ ] Missing required field (e.g. `results.provenance`) fails validation.
- [ ] `provenance: "deduced"` with absent `environment.*` validates (migration case).
- [ ] A bad name (`transform-2-x`, lowercase prefix / unpadded ordinal) is rejected by the documented regex.

### Integration

- [ ] The reference scene validates end-to-end against all three schemas.

## Out of scope

- **Moving/migrating production data** — `STO-SCN-033`.
- **The sync CLI + checksum manifest** — `STO-SCN-029` (this story only *reserves*
  that the CLI owns content hashes).
- **S3 prefix layout** — `STO-SCN-028` (informed by, not defined here).
- **The Docker runner actually emitting `results.json`** — `STO-SCN-031` (this
  story defines the contract; 031 implements emission).
- **A dependency-graph file format** — decided to live in pipeline code (epic
  Decisions); `specification.json.inputs` lists paths, nothing more.

## Implementation Notes

_(Fill in during / after implementation.)_

### What Changed

_(Actual implementation. May differ from § Design above.)_

### Files Modified

- `path/to/file` — _(what changed)_

### Gotchas

_(Anything surprising or worth noting for future readers.)_
