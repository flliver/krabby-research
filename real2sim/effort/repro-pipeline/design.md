---
xid: DES-SCN-REPRO
kind: design
effort: scn
status: open
date: 2026-06-09
guidance: ./guidance.md
hugs: []
tenets: []
bd-id: krabby-61r
priority: 1
assignee: principal
---

# Reproducible, Configuration-Driven Data Pipelines

## Problem
M11's reconstruction runs were script/hand-driven prototypes: un-versioned tool source bind-mounted into containers, parameters scattered across run-scripts and journals, provenance reconstructed *after the fact* by forensics (STO-SCN-036, 15/18 measured only via journal+mtime+outposts archaeology). The 2026-06-09 reproduction attempt succeeded only because the host snapshot, image, and frame sets happened to survive — that's luck, not engineering. Per **HUG-KRB-002**: no more prototypes.

## Working assumptions
- The scene store schema (STO-SCN-026/033) is the foundation: `scenes/<scene>/pipeline-<p>/run-<r>/transform-NN-<t>/{specification.json,results.json,data/}`. The runner *produces* these records; humans author only specifications.
- Transforms are containerized, pluggable units: image + pinned source + parameter schema + consume convention (STO-SCN-031) → declared outputs.
- Validation (STO-SCN-034) gates records in CI.
- Scene distribution (S3/sync slice, P2) carries the data; this design carries the *execution*.

## Vocabulary
- **Specification** — the human-authored intent: transform, image, parameters, inputs.
- **Run** — one execution of a pipeline (ordered transforms) over a scene.
- **Transform plugin** — a registered, versioned unit that maps spec → outputs + results.
- **Reproduction** — re-executing a recorded spec and comparing results within declared tolerances (bit-exactness is NOT the bar on RTX 50xx; metric equivalence is).
