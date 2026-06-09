---
xid: HUG-KRB-002
kind: hug
effort: krb
status: active
date: 2026-06-09
author: Operator, 2026-06-09
bd-id: krabby-wnf
title: Full reproducibility or it isn't a solution — no more prototypes
---

# Full reproducibility or it isn't a solution — no more prototypes

## Quote
> "Everything we do at this point should be *fully* reproducible. If we don't have reproducibility (via source-controlled code and scripting), then we don't have a solution. … we need configuration-driven data pipelines with pluggable transformations. … No more prototypes." — Operator, 2026-06-09

## Context
Issued during the reproduction of the M11 curated MAtCha runs, immediately after two forensic findings: (1) the production MAtCha source was bind-mounted from an **un-versioned host snapshot** (`tbeeprz:~/scratch/MAtCha`, not a git repo, patches applied in place — `git_sha: None` in every provenance record); (2) the recorded runner scripts did not match the actual container mounts. STO-SCN-036 had to *forensically reconstruct* what tooling should have recorded. The scene store's `pipeline-*/run-*/transform-NN/{specification,results}.json` shape is the operator's intended foundation.

## Direction
- Every run MUST be reproducible from source-controlled code + scripting alone. Un-versioned code (host snapshots, in-place patches, hand-driven container exec) is not a solution.
- Pipelines MUST be configuration-driven: a `specification.json` is the input contract; a tool-emitted `results.json` (provenance `measured`) is the output contract. Hand-authored provenance is a migration-era artifact, never the norm going forward.
- Transformations MUST be pluggable units behind a common interface (matcha, mast3r, colmap, conditioning, …).
- `maturity: "prototype"` is no longer an acceptable end-state for new work. Prototype-grade runs may explore; anything kept/recorded must come from the runner.
- Tracked under DES-SCN-REPRO / EPI-SCN-PIPELINE-RUNNER (STO-SCN-038..041).
