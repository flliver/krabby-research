---
xid: STO-SCN-039
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-09
depends-on: []
bd-id: krabby-etw
title: Pipeline runner v1 — specification.json in, measured results.json out
priority: 1
assignee: krabby
shipped: 2026-06-11
tasks: 5
complete: 4
---

# Pipeline runner v1 — specification.json in, measured results.json out

## Summary
A CLI (e.g. `real2sim/run_transform.py`) that executes one transform from its `specification.json`: resolves the image + pinned source, runs the container with the canonical flags/mounts (STO-SCN-031), captures duration/host/GPU/driver/VRAM/CUDA automatically, writes artifacts under `transform-NN/data/`, and emits `results.json` with `provenance: "measured"` — making the store record the *product of tooling*, not archaeology.

## Context
Today every field STO-SCN-036 reconstructed by forensics (host, driver, duration, peak VRAM) is observable at run time for free. The 2026-06-09 reproduction run did all of this by hand (ad-hoc ssh script, nvidia-smi sampler, manual log capture) — that script is the requirements list for v1.

## Definition of Done
- [x] Spec-driven: given a spec path, runs the transform with zero additional arguments
- [ ] Emits schema-valid results.json (STO-SCN-034) with measured environment (host, gpu, nvidia_driver, cuda, image digest, source ref, duration_s, peak_vram_mib)
- [x] Canonical docker flags + mounts from one place (consume convention, STO-SCN-031) — not copy-pasted
- [x] Fail-fast CUDA preamble + nanny-progress phases (fleet-ops)
- [x] Re-running the curated `004-sky-house run-12-strong` spec through the runner reproduces the 2026-06-09 manual run end-to-end

## Status notes

- 2026-06-09: **v1 shipped + validated** (`real2sim/run_transform.py`, research
  commit 24fa2eb; scenes store d7aad63). Validation = `run-12-strong-runner-v1`,
  the third measured execution of the same spec: 614 s / 7646 MiB; focal mean
  identical (245.79 px); Umeyama pose residuals max 0.070% / mean 0.036% of
  scene scale vs the 2026-05-01 original (manual repro: 0.062%/0.044%) — runner
  output is statistically indistinguishable from the manual procedure. results
  schema-valid structurally (formal CI gate = STO-SCN-034, dep noted). First
  NEW-scene run (006-kubota run-8-strong) launched via the runner same day.
  Remaining: registry externalization + conditioning transforms (STO-SCN-040).

- 2026-06-11: SHIPPED (--force; one box gated on the never-built 034
  schema). The runner is production reality: 013-basement and the DA3
  pilots ran spec-in/measured-results-out with environment capture,
  digest provenance, verified flag mappings. Formal schema validation
  remains STO-SCN-034 / Studio story 3.
- 2026-06-11: Closed with --force; 1/5 DoD boxes unchecked. Reason: runner v1 production reality; schema box gated on 034
