---
xid: EPI-SCN-PIPELINE-STUDIO
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-11
hugs: []
tenets: []
bd-id: krabby-ggb
---

# Pipeline Studio — task/instance/run taxonomy, pluggable transforms, DAG UI

> **PLANNED, NOT STARTED** (operator directive 2026-06-11: plan only,
> questions first). Parent: DES-SCN-REPRO.

## Problem Statement

The pipeline exists as recipes + scripts + baked images, and the UI
(rate_renders) shows only *runs* — there is no first-class distinction
between what CAN run, what is CONFIGURED to run, and what DID run.
Settings live scattered across spec JSONs with no declared
ranges/defaults; transforms are not pluggable; composing a new
pipeline means editing shell scripts. Iterating on pipelines and the
ranking system requires the structure below.

## Core taxonomy (operator spec, verbatim intent)

| | Concept | = |
|---|---|---|
| A | **task** | atomic work: code + input definition + output definition (incl. variables produced) |
| B | **task_instance** | chosen transform + 100% of settings required to run (settings + inputs) |
| C | **task_run** | instance + captured runtime data (logs, outputs) |
| D | **pipeline** | DAG of tasks, settings unspecified |
| E | **pipeline_instance** | pipeline + 100% of settings (no input/output data) |
| F | **pipeline_run** | pipeline + instance + input/output data — possibly not yet executed |

**Variables:** DAG-consistent settings are declared as VARIABLES+VALUES
at pipeline_instance level, referenced as VARIABLES at task_instance
level; expansion is captured at run time.

## Requirements

- Every transformation step declares: inputs, outputs, tunable
  settings with **min/max/default**, and the **image+code** that
  executes it (the RECIPES phase catalog + baked-tools policy are the
  seed data — 13 phases already documented).
- Pluggable transforms; draggable UI components for pipeline
  composition; UI surfaces A–F distinctly.
- **Don't reinvent**: adopt an existing DAG/orchestration library.
- **CRITICAL: MVP ASAP, and do not break what exists** (store layout,
  runoff, recipes keep working; Studio wraps, never replaces, until
  proven).

## Library landscape (candidates to VERIFY in the spike — prior
knowledge, not yet validated, T-002)

| Candidate | Maps to taxonomy | Gaps |
|---|---|---|
| **Dagster** | ops/jobs=A/D, run_config=B/E, runs=C/F; config schemas w/ defaults; Dagit web UI w/ DAG + run views | UI not draggable-authoring |
| Prefect | flows/tasks + deployments=instances | weaker config-schema story |
| Kedro (+viz) | pipeline catalog + params | viz read-only; runs tracking thin |
| **langfuse** (operator suggestion) | traces/observations≈task_runs; **Scores API ≈ ranking system** | LLM-observability focus: NO DAG authoring, scheduling, or settings schemas — likely the *evaluation/run-capture* layer, not the orchestrator |
| React Flow / Rete.js | draggable DAG UI components | UI only — pairs with any backend |

Working hypothesis to test: **orchestrator (Dagster-class) + React
Flow for drag authoring + langfuse-or-builtin for run scoring**, with
our spec/results JSONs as the persistence the adapters read/write.

## Planned stories (minted only after operator answers)

| # | Story | Size | Notes |
|---|---|---|---|
| 1 | Library spike — verify candidates, pick stack (time-boxed) | M | decision doc + throwaway demo of A–F on ONE real transform |
| 2 | Transform catalog: formalize the 13 recipe phases as task defs (A) with settings min/max/default + image/code refs | M | seed = RECIPES + baked images |
| 3 | Data-model adapters: A–F ↔ store layout (spec/results/run dirs), non-breaking | M | Studio reads/writes what exists |
| 4 | MVP UI: A–F views, read-only DAG, instance vs run distinction | L | reuse/embed rate_renders for run evaluation |
| 5 | Variable propagation (pipeline_instance vars → task_instance refs → run-time expansion capture) | M | |
| 6 | Draggable pipeline composer | L | post-MVP unless library gives it ~free |
| 7 | Ranking integration: rankings as scores attached to pipeline_runs | M | absorbs rate_renders evolution |
| 8 | Run verification/regression harness (absorbs STO-SCN-041 scope) | M | compare task_runs within tolerances |

## Open questions (operator)

1. **MVP cut line**: is read-only DAG + correct A–F distinction + run
   browsing the MVP, with draggable authoring post-MVP? (Drag-to-
   compose is the most expensive single item.)
2. **Ranking placement**: should rate_renders be absorbed INTO the
   Studio UI for MVP, or stay standalone with rankings ingested as
   scores?
3. **Execution dispatch**: MVP = Studio *records and renders* runs
   executed the current way (scripts → fleet docker), or must it also
   *launch* them (SSH/agent dispatch) to count?
4. **Hosting**: Studio web UI on the Mac (like :8090) or on jbeeprz?
5. **langfuse**: given its LLM-observability shape, OK to scope it as
   the scoring/run-capture candidate rather than the orchestrator?
6. **Vocabulary**: keep "transform/transformation" (store schema term)
   as the display name for "task," or rename store artifacts to match
   A–F?
