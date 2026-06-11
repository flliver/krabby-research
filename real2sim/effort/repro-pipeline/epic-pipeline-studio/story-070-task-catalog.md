---
xid: STO-SCN-070
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: [STO-SCN-069]
bd-id: krabby-ffb
shipped: 2026-06-11
tasks: 4
complete: 4
---

# Task catalog: 13 recipe phases as task defs — settings min/max/default + image digest + code ref

## Summary

Every RECIPES.md phase becomes a machine-readable task definition (A):
inputs, outputs (incl. variables produced), tunable settings with
min/max/default, and the exact docker image + code path that executes
it. This catalog IS the reproducibility contract.

## Context

Seed data already exists: RECIPES.md 13-phase catalog, baked-tools
policy (`/opt/krabby-tools`), registry images with TOOLS_GIT_SHA
labels, spec/results JSONs from real runs. Format chosen by the
STO-SCN-069 spike decision.

## Problem

Settings today are scattered across spec JSONs with no declared ranges
or defaults; "what image ran this" lives in commit messages and
labels. A third party cannot enumerate what a task CAN do, only what
one run DID.

## Design

One definition file per task. Hard requirements per task def:
- settings: name, type, **min, max, default**, units, what it trades
  off (the hard-limits table — e.g. process_res 504 default / 756
  nogs ceiling / 1008 OOM — is the model)
- execution: image (registry ref + digest), entrypoint under
  `/opt/krabby-tools`, code ref (git path + the TOOLS_GIT_SHA story)
- inputs/outputs: store-relative path patterns + produced variables

Vocabulary per operator decision 6: these are **tasks**; store
`transform-NN-*` paths unchanged.

## Definition of Done

- [x] All 13 phases + the DA3-branch tasks (infer, tsdf_mesh,
      render_view, tetra_condition, gauge_align) have defs — 17 files
      in `real2sim/tasks/`.
- [x] Min/max values are **measured or sourced** (hard-limits table,
      OOM findings), not invented (T-017, T-010); unknown ranges say
      unknown (T-002) — unmeasured bounds omitted with an explicit
      "bounds unmeasured" description.
- [x] A def round-trips: catalog def + a real run's spec JSON →
      validates. Verified: matcha run-8-strong, 007 normalize, 006
      da3 hires756 (via infer_gs→mode mapping), all 013 preproc
      specs. Conditional bound works: gs@756 rejected, nogs@756
      accepted, nogs@1008 rejected — all measured ceilings.
- [x] RECIPES.md points at the catalog as the canonical phase list
      (T-023 — no second prose copy).

## Implementation Notes

- `real2sim/tasks/*.json` — 17 JSON Schema 2020-12 task defs (13
  phases; phase 8 also carries `tetra-condition`, phase 13 carries
  the DA3 branch: infer/tsdf-mesh/render-view; `gauge-align` under
  phase 6). Operator tasks (coverage-curation, camera-save,
  rank-runoff) flagged `x-task.operator: true` (T-020 surfaces).
  DA3 defs carry `license_flag: CC-BY-NC-4.0 — not deliverable`.
- `real2sim/task_catalog.py` — loader + CLI (list/show/validate/
  check-spec). `check-spec` validates only catalog-declared keys of a
  historical spec's `parameters`; execution pins reported as
  uncovered, never failed.
- Gotcha: JSON Schema `if` without `required` matches when the
  property is ABSENT — the da3-infer mode-conditional ceiling
  needed `"required": ["mode"]` inside each `if` (caught by the 756
  nogs round-trip failing wrongly).

## Out of scope

- Changing any store paths or existing spec/results schemas (071's
  adapters bridge; nothing breaks).
