# Store schema v3 — Studio structures (STO-SCN-076)

> Additive over store-shape v2 (STO-SCN-062/063). Nothing moves,
> nothing renames, every v2 consumer runs unchanged. **T-007 gate:
> operator reviews this doc before the first v3 write lands in the
> store** (the STO-SCN-077 backfill dry-run is the natural review
> surface).

## Where each taxonomy letter lives

| | Concept | Lives | File |
|---|---|---|---|
| A | task | **research repo** | `real2sim/tasks/<task>.json` (STO-SCN-070) |
| D | pipeline | **research repo** | `real2sim/pipelines/<name>.json` |
| E | pipeline_instance | **research repo** | `real2sim/instances/<name>.json` |
| B | task_instance | derived | E's per-task settings block (no separate file) |
| C | task_run | **store** (existing) | `transform-NN-*/{specification,results}.json` |
| F | pipeline_run | **store** | `run-<r>/run_record.json` (NEW) + existing run dir |

**Why D/E in the research repo, not the store:** pipelines and
instances are *experiment definitions* — settings without data,
scene-independent, reviewed like code. Versioning them with the code
that executes them gives provenance for free (one git SHA pins
catalog + pipeline + instance + tools). The store stays what v2 made
it: inputs, measured metadata, finals.

**Reproducibility despite the split:** `run_record.json` embeds a
full **snapshot** of the expanded instance (every task's expanded
settings + image digests + code SHA). A third party needs only the
store to re-run; the repo files are the *editable* form, the snapshot
is the *executed* form. Snapshot-vs-source drift is detectable via
the recorded SHA.

## New store file: `run-<r>/run_record.json`

Schema: `real2sim/schemas/run_record.json`. One per run dir.
Tracked automatically by the existing v2 `!**/*.json` rule —
**zero `.gitignore` change** (verified). Size: ~KBs (metadata only).

Key fields (see schema for full shape):

- `schema: 3`, `scene`, `pipeline`, `run`, `variant` (`<p>--<r>` —
  the rankings.jsonl join key)
- `instance`: name + **expanded settings snapshot** per task
  (variables already expanded; expansion captured at run time)
- `execution`: host, started/finished, trigger (`studio|manual|backfill`)
- `provenance` per task: image ref + digest, tools_git_sha, code_ref,
  input content hashes; `"unknown"` + `backfilled: true` where
  history can't answer (T-002)
- `reproducibility`: `by_record: true|false|unknown` +
  `license_flags` (e.g. DA3 CC-BY-NC → not deliverable)

## Scores: derived, not duplicated (deliberate deviation)

Operator decision 2 says rankings become scores **on** pipeline_runs.
Storage-wise we realize that as a **join, not a copy**:
`rankings.jsonl` (per scene) stays the single source of operator
judgment (T-023); `run_record.variant` is the join key; the Studio
leaderboard computes scores-on-runs at read time. A materialized
`scores.json` per run would drift the moment a ranking is re-submitted.
If score analytics ever outgrow read-time joins, revisit (069 noted
langfuse as the post-MVP candidate for that).

## Pipelines shipped with v3 (D)

- `real2sim/pipelines/matcha-trunk.json` — the common trunk
  (RECIPES.md): sfm→train→mesh implicit in matcha-reconstruction →
  tsdf-extract-orient → [tetra-condition] → build-blender-scene →
  camera-save* → render-comparison-matrix → rank-runoff*
  (* = operator tasks)
- `real2sim/pipelines/da3-eval.json` — evaluation branch:
  da3-infer → da3-tsdf-mesh → da3-render-view → rank-runoff*
  (license-flagged, never deliverable)

## Compatibility matrix

| Consumer | Effect of v3 |
|---|---|
| runoff scripts / render_comparison_matrix | none (don't read run_record) |
| rate_renders | none (scans renders/) |
| store .gitignore | none (`!**/*.json` already tracks run_record) |
| sync/gather/fleet | none (one small JSON per run) |
| run_transform.py | unchanged; 073 adds run_record emission alongside |
