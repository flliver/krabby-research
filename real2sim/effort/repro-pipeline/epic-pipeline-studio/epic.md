---
xid: EPI-SCN-PIPELINE-STUDIO
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-11
hugs: []
tenets: []
bd-id: krabby-ggb
---

# Pipeline Studio — task/instance/run taxonomy, pluggable tasks, DAG UI

> Parent: DES-SCN-REPRO. Planned 2026-06-09; operator answered all six
> open questions 2026-06-11 (decisions recorded below) — epic active.

> **DIRECTION SHIFT (2026-06-11, post-MVP):** HUG-SCN-005 (DAG of
> DAGs, content-addressed store, task/graph/job) supersedes parts of
> this epic's data model: STO-SCN-076/077 closed superseded; the A–F
> taxonomy maps to the new vocabulary per locked #4. The MVP shipped
> here (catalog, adapters, trigger, harness, Studio UI) remains the
> working system until the HUG's migration lands.

## Purpose (operator, 2026-06-11 — the filter for every feature)

> The **ENTIRE purpose** of this effort is to figure out the best
> ***reproducible*** processing pipeline for converting images/videos
> to 3D scenes to meet our **M11 deliverables**.

Every MVP feature must serve that loop directly:

```
define pipeline_instance (tasks + settings)
  → run it (reproducibly: pinned image + code + captured settings)
  → render N comparison images
  → rank against competing instances
  → identify the winning configuration → M11 deliverable recipe
```

**MVP test for any feature:** does it shorten the
define→run→rank→decide cycle, or harden reproducibility (could a
third party re-run this pipeline_run from its record and get the
same mesh)? If neither — it's out of the MVP.

Non-goals for MVP (explicitly): drag-to-compose, multi-host
scheduling, jbeeprz hosting, generic-workflow features beyond what
the experiment loop needs.

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

## Operator decisions (2026-06-11)

1. **MVP cut line**: agreed — read-only DAG + A–F distinction + run
   browsing, **plus interactive *editing* of composed definitions**
   (form/field editing of instances — NOT drag-to-compose, which stays
   post-MVP).
2. **Ranking**: **absorbed into Studio.** The purpose of the system is
   to find the best pipeline definition for the best 3D scenes via
   experimental runs; each pipeline_run outputs N images that get
   compared. Ranking is core, not a bolt-on.
3. **Dispatch**: MVP runs are **triggered centrally**, executed on a
   **single, externally chosen host** (host is an operator-supplied
   parameter, not a scheduler decision).
4. **Hosting**: jbeeprz hosting is a **nice-to-have** — Mac-first is
   acceptable for MVP.
5. **langfuse**: confirmed scoped as scoring/run-capture candidate,
   not the orchestrator.
6. **Vocabulary**: **"task"** is canonical (more generic than
   "transform," which implies data manipulation). Store artifacts keep
   `transform-NN-*` paths for now (non-breaking); UI + new schemas say
   task.

## Stories — cut against the Purpose

**MVP (the experiment loop, in dependency order):**

| XID | Story | Size | Serves the purpose how |
|---|---|---|---|
| STO-SCN-069 | Library spike — verify candidates, pick stack (hard time-box; demo = ONE real task through define→run→record) | M | don't reinvent; but the spike is judged ONLY on the experiment loop, not generic-workflow features |
| STO-SCN-070 | Task catalog: the 13 recipe phases as task defs (A) — inputs, outputs, settings **min/max/default**, **image digest + code ref** per task | M | this IS the reproducibility contract: a pipeline_run record must name exactly what executed |
| STO-SCN-076 | Store schema update: additive v3 structures — instance homes, run_record, scores-on-runs — non-breaking | M | gives E/run-records/scores defined tracked locations so 072–075 don't each invent their own |
| STO-SCN-071 | Data-model adapters: A–F ↔ existing store (spec/results/run dirs), non-breaking; every historical run becomes a browsable pipeline_run | M | past runoffs are the existing experimental corpus — don't orphan them |
| STO-SCN-077 | Backfill: materialize v3 records + scores for ALL existing runs (recovered provenance or explicit unknown) | M | the leaderboard starts populated — months of ranking judgment carried forward, not stranded |
| STO-SCN-072 | Experiment UI: browse A–F, **edit composed pipeline_instances** (forms, not drag), diff two instances' settings | L | the "define" step; settings-diff is how you learn WHY one config won |
| STO-SCN-073 | Central run trigger: launch a pipeline_instance on one operator-chosen host; capture run record (settings expansion, image digests, logs, outputs) | M | the "run" step — reproducible by construction, per decision 3 |
| STO-SCN-074 | Ranking absorbed: pipeline_run → N comparison renders → rank → scores stored ON the run; leaderboard per scene/view across instances | M | the "rank→decide" step; absorbs rate_renders (decision 2) |
| STO-SCN-075 | Reproducibility harness: re-run a pipeline_run from its record on a clean host, compare outputs within tolerances (absorbs STO-SCN-041 scope) | M | proves the *reproducible* in "best reproducible pipeline" — M11 gate |

**Post-MVP (explicitly out until the loop ships):**

| Story | Why deferred |
|---|---|
| Draggable pipeline composer | composing NEW DAGs is rarer than tuning settings on the known DAG; forms cover the experiment loop |
| Variable propagation as a first-class feature | MVP captures expanded settings at run time (the reproducibility need); declared-variable sweeps come after the loop works |
| jbeeprz hosting | nice-to-have (decision 4); Mac-first |

## Open questions

All six original questions answered 2026-06-11 — see "Operator
decisions" above. Additional framing 2026-06-11: the Purpose section
(reproducible pipeline → M11) is the feature filter for the MVP.
