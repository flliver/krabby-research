---
xid: EPI-SCN-EXPERIMENTS-TAB
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-16
hugs: []
tenets: []
bd-id: krabby-aatf
---

# Experiments tab — configure, run, and rank reconstruction experiments per scene

## Problem Statement

The **Preprocess** stage (ingest → precull → solve → covis → select → scout,
+ Measure) produces a scene's shared, once-per-scene foundation. Everything
*after* that — which reconstruction model + settings, and the optional
post-processing (cull / gap-fill / smooth) and rendering — is a **variant** the
operator wants to try several of and then **rank** to pick the single best. Today
those variants are hand-run `v4exec` reconstruct/condition/render subcommands;
there is no UI to **define, run, monitor, and manage** them. This epic adds an
**Experiments** tab where each experiment is a configured task-chain over the
preprocessed scene, run idempotently, surfaced for ranking.

## Goals

- An **Experiments** tab (Scenes app) listing every **configured experiment** for
  the selected scene + **whether it's materialized**.
- **Preconditions, FAIL LOUD** — the tab refuses to operate unless the scene has:
  - **Preprocess** done (a solve + a FINAL-N camera set),
  - **Measure** done (a metric `datum.json`),
  - **Scout** done (≥ 1 scout view).
- **An experiment = a task chain** over the preprocessed scene:
  - **REQUIRED — Reconstruct** (choose **model** + settings; e.g. DA3 / matcha),
  - **Cull** (optional),
  - **Gap-fill** (optional),
  - **Smoothing** (optional),
  - **REQUIRED (> 1 view) — Rendering** (the comparison renders that feed Rank).
- **Run + monitor** an experiment; **tasks are idempotent** (a materialized task
  is not re-run unless its inputs/settings changed).
- Operator actions: **ADD experiment** (choose a model), **ADD task** (drag/drop
  from a palette), **VIEW/EDIT task settings**, **RUN**, **DELETE** (experiment +
  its data), **PURGE** experiment data (force a clean full re-run).

## Non-Goals (Out of Scope)

- The Preprocess pipeline (EPI-SCN-SCENE-MANAGER / "Preprocess" tab) and Measure
  (STO-SCN-152) — preconditions, not part of this tab.
- The Rank tab (EPI-SCN-PIPELINE-STUDIO) — experiments *feed* it; ranking lives there.
- New reconstruction/cull/mesh algorithms — all exist as `v4exec`
  represent/meshify/condition/render nodes; this epic surfaces + orchestrates them.

## What already exists (reuse map)

| Piece | Reuse |
|---|---|
| Reconstruct (model + settings) | `v4exec reconstruct-{da3,matcha}` + the represent graph |
| Cull / gap-fill / smooth (mesh condition) | `v4exec cull` / condition nodes (EPI-SCN-MESH-CONDITION); `cull-mesh@2` + SDF primitives |
| Render | the v4 render graph + `/api/materialize/` (server.py) |
| Idempotency | v4 content-addressed identity (taskdef + settings → node hash); NOOP when present |
| Settings forms (catalog-constrained) | Pipeline Studio instance editor (`studio/`, EPI-SCN-PIPELINE-STUDIO) |
| Tab shell / scene selector / progress | the Scenes app (`rate_renders/`, EPI-SCN-SCENE-MANAGER) |

## Experiment model

```
experiment = { id, name, model, tasks: [Reconstruct(req), Cull?, GapFill?, Smooth?, Render(req>1view)] }
each task   = { kind, settings, materialized: bool, node_id? }   # node_id = v4 content hash
```
An experiment is essentially a named **represent → (condition*) → render** chain;
"materialized" = its v4 nodes exist in the store.

## Stories

| # | XID | Story | Size |
|---|-----|-------|------|
| 1 | `STO-SCN-163` | Experiments tab shell + **precondition gate (FAIL LOUD)** + experiment list (materialized status) | M |
| 2 | `STO-SCN-164` | **Add experiment** (choose model) + the experiment data model + persistence | M |
| 3 | `STO-SCN-165` | **Task palette + add task (drag/drop)** — Reconstruct/Cull/GapFill/Smooth/Render onto an experiment | M |
| 4 | `STO-SCN-166` | **View/edit task settings** (catalog-constrained forms; reuse Studio editor) | M |
| 5 | `STO-SCN-167` | **Run + monitor** an experiment; **idempotent** task execution (skip materialized) | L |
| 6 | `STO-SCN-168` | **Delete** experiment (+ data) and **Purge** experiment data (force full re-run) | M |

## Success Criteria

- [ ] On a preprocessed+measured+scouted scene, the operator builds an experiment
      (model + tasks), runs it, watches it materialize, and it appears for ranking.
- [ ] A scene missing any precondition shows a clear FAIL-LOUD block, not a broken run.
- [ ] Re-running an experiment re-does only what changed (idempotent); Purge forces a clean re-run.
- [ ] Delete removes the experiment + its materialized data.

## Retrospective

_(Fill in after epic completion.)_
