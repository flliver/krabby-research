---
xid: STO-SCN-039
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-09
depends-on: []
bd-id: krabby-etw
title: Pipeline runner v1 — specification.json in, measured results.json out
priority: 1
assignee: devex
---

# Pipeline runner v1 — specification.json in, measured results.json out

## Summary
A CLI (e.g. `real2sim/run_transform.py`) that executes one transform from its `specification.json`: resolves the image + pinned source, runs the container with the canonical flags/mounts (STO-SCN-031), captures duration/host/GPU/driver/VRAM/CUDA automatically, writes artifacts under `transform-NN/data/`, and emits `results.json` with `provenance: "measured"` — making the store record the *product of tooling*, not archaeology.

## Context
Today every field STO-SCN-036 reconstructed by forensics (host, driver, duration, peak VRAM) is observable at run time for free. The 2026-06-09 reproduction run did all of this by hand (ad-hoc ssh script, nvidia-smi sampler, manual log capture) — that script is the requirements list for v1.

## Definition of Done
- [ ] Spec-driven: given a spec path, runs the transform with zero additional arguments
- [ ] Emits schema-valid results.json (STO-SCN-034) with measured environment (host, gpu, nvidia_driver, cuda, image digest, source ref, duration_s, peak_vram_mib)
- [ ] Canonical docker flags + mounts from one place (consume convention, STO-SCN-031) — not copy-pasted
- [ ] Fail-fast CUDA preamble + nanny-progress phases (fleet-ops)
- [ ] Re-running the curated `004-sky-house run-12-strong` spec through the runner reproduces the 2026-06-09 manual run end-to-end
