---
xid: STO-SCN-070
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: [STO-SCN-069]
bd-id: krabby-ffb
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

- [ ] All 13 phases + the DA3-branch tasks (infer, tsdf_mesh,
      render_view, tetra_condition, gauge_align) have defs.
- [ ] Min/max values are **measured or sourced** (hard-limits table,
      OOM findings), not invented (T-017, T-010); unknown ranges say
      unknown (T-002).
- [ ] A def round-trips: catalog def + a real run's spec JSON →
      validates (settings within ranges, image matches).
- [ ] RECIPES.md points at the catalog as the canonical phase list
      (T-023 — no second prose copy).

## Out of scope

- Changing any store paths or existing spec/results schemas (071's
  adapters bridge; nothing breaks).
