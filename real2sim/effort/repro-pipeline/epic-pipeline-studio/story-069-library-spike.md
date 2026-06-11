---
xid: STO-SCN-069
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: []
bd-id: krabby-joz
shipped: 2026-06-11
tasks: 4
complete: 4
---

# Library spike: verify stack candidates on the experiment loop (define→run→record), pick + decision doc

## Summary

A hard-time-boxed spike that verifies the library-landscape table in
the epic (prior knowledge, T-002-flagged unverified) and picks the
stack, judged ONLY on the experiment loop:
define pipeline_instance → run reproducibly → record → rank.

## Context

EPI-SCN-PIPELINE-STUDIO § "Library landscape". Operator: don't
reinvent; langfuse scoped to scoring/run-capture (decision 5); MVP
ASAP and don't break the existing store/recipes. The Purpose section
is the rubric — generic-workflow features score zero.

## Problem

We have prior-knowledge claims about Dagster / Prefect / Kedro /
langfuse / React Flow but no verified facts (current versions,
licenses, config-schema min/max support, run-record fidelity,
embeddability next to rate_renders). Picking wrong here costs the
whole MVP timeline.

## Design

### Approach

1. Verify current state of each candidate (versions, license, the
   specific features the epic table claims).
2. Demo: ONE real task (e.g. `tetra_condition`) expressed as a task
   def with settings min/max/default, run via the candidate's
   machinery against a real scene store dir, run record captured.
3. Score each candidate against the loop: define / run-on-chosen-host
   / record-reproducibly / rank-attachable / Mac-runnable /
   non-breaking adjacency to the store.
4. Decision doc in this story's Implementation Notes; losers get a
   one-line reason each.

Throwaway code lives in a scratch dir, never in real2sim/ proper.

### Changes

| File | Change |
|------|--------|
| this story § Implementation Notes | decision doc |
| scratch demo (untracked) | discard after decision |

## Definition of Done

- [x] Each epic-table claim marked verified/refuted with evidence.
- [x] One real task demoed end-to-end on the chosen candidate
      (tetra_condition def + real 006 run data — see Implementation
      Notes).
- [x] Stack decision recorded (orchestrator, UI, scoring layer) with
      license check (must be usable for contract work — note the DA3
      CC-BY-NC precedent). All MIT.
- [x] Time-box respected: decision landed inside the box; the named
      default (thin custom layer + React-Flow-class UI) was chosen on
      positive evidence, not timeout.

## Out of scope

- Any production code. The demo is throwaway by definition.
- Drag-to-compose evaluation beyond "does the UI lib exist and render
  a DAG" (post-MVP feature).

## Implementation Notes

### Decision doc (2026-06-11)

**Stack picked:**

| Layer | Decision |
|---|---|
| Task-def formalism (A) | **JSON Schema draft 2020-12** — `minimum`/`maximum`/`default` are native keywords; `x-task` extension block carries image/entrypoint/code-ref/inputs/outputs |
| Validation | **python `jsonschema`** (Draft202012Validator) — also satisfies the dormant STO-SCN-034 harness desire |
| Persistence | **the scene store itself** (existing spec/results/run JSONs + 076's additive v3 files). No external run DB |
| Backend | thin Python server in the rate_renders mold (T-013 — extends what exists) |
| DAG view (072) | server-rendered read-only first; **React Flow (@xyflow/react v12, MIT — verified)** if/when client-side interactivity is worth a build step |
| Scoring | built-in (rankings.jsonl + 076 scores-on-runs). langfuse REJECTED for MVP |

This exercises the story's named default ("thin custom layer over our
existing JSONs + React-Flow-class UI") — but as a positive verdict,
not a timeout: the store already IS the system of record (spec=B,
results=C, run.json≈F, rankings keyed by variant), and orchestrators
add a foreign run-DB without covering the two things we actually
lack (min/max form editing, absorbed ranking).

**Per-candidate verdicts (epic table claims → verified/refuted):**

- **Dagster** — partially verified: pydantic config classes give
  defaults + validation, and the Launchpad has schema-validated
  editing. REFUTED for our MVP on three counts: (1) Launchpad is a
  **YAML text editor**, not min/max form fields; (2) ranking cannot
  be absorbed into Dagit → two UIs, violating operator decision 2;
  (3) pipeline_instance (E) is not a first-class persisted object —
  run config is per-run. Plus our execution path (SSH→docker on an
  operator-chosen host) gets no leverage from its executor model.
- **Prefect / Kedro** — rejected by class without individual demo
  (T-002, stated honestly): same orchestrator shape as Dagster with
  weaker config-schema/UI stories per the epic table; they cannot
  beat Dagster on the exact criteria Dagster already fails.
- **langfuse** — claim "Scores API ≈ ranking" verified, but
  self-hosting requires web + worker containers + **Postgres +
  ClickHouse + Redis + S3** (v3 architecture). Wildly oversized for
  operator-rankings-on-JSON-runs; fails MVP-ASAP + Mac-first filter.
  Revisit post-MVP only if score analytics outgrow jsonl.
- **React Flow** — verified MIT (xyflow), v12 current. Adopted as
  the designated drag/DAG library for 072+/post-MVP composer.

**Demo (throwaway, /tmp/spike-069, discarded):** `tetra_condition`
expressed as a JSON-Schema task def; against the REAL 006 run
(`run-8-strong` tetra1m record): (1) real recorded settings validate;
(2) out-of-range rejected naming the bound ("50 is less than the
minimum of 100000"); (3) defaults machine-extracted for form prefill;
(4) def-declared outputs found in the real run dir; (5) C-level
measured facts read back (32.2M→1.0M tris, 33 MB).

**License check:** jsonschema (MIT), @xyflow/react (MIT) — both
contract-safe. No copyleft/NC anywhere in the picked stack.

### Gotchas

- JSON Schema's `default` is annotation-only (not applied by the
  validator) — the form layer must apply defaults itself, which is
  what we want anyway (explicit expanded settings at run time).
