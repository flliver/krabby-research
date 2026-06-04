---
xid: STO-SCN-019
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-d13
priority: 1
title: T3.F1 — Extreme Parkour Dockerfile
assignee: devex
---

# T3.F1 — Extreme Parkour Dockerfile

## Summary

Per grant Task 3: package Extreme Parkour locomotion model in a Docker container that:

## Context

Per grant Task 3: package Extreme Parkour locomotion model in a Docker container that:
- Launches IsaacSim independently
- Consumes the standardized USD env
- Receives depth-based observations from the collision mesh
- Outputs trajectory and metric data

## Definition of Done

- [ ] Dockerfile builds and runs from a clean state
- [ ] Container launches IsaacSim and loads ≥1 USD env
- [ ] Depth observations populated correctly
- [ ] Trajectory/metrics output in agreed schema


## Journal Notes

No Extreme-Parkour-specific Dockerfile work yet, but the journal carries cross-cutting Docker/PyTorch/CUDA infra lessons any locomotion container must inherit:
1. **RTX 5080 (sm_120 Blackwell) needs the cu128 PyTorch wheel index** — cu130 stable lacks sm_120 kernels and fails at runtime with "no kernel image is available"; cu128 works even on a CUDA-Toolkit-13.0 host.
2. **`--shm-size=8g` is mandatory** — the 64 MB default `/dev/shm` causes a silent PyTorch deadlock at 0% GPU with no error. Full flag set: `--shm-size=8g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`.
3. **Long-running containers can silently lose CUDA** (NVML "Unknown Error" / "No CUDA GPUs are available" after ~5+ hrs — NVIDIA Container Toolkit cgroup/device-passthrough hiccup); fix is a ~2-min restart. Run scripts should fail-fast with a `torch.cuda.is_available()` preamble.
4. `research/docs/PYTORCH_GPU_SUPPORT.md` documents cross-arch (Jetson Orin / RTX 4080 / RTX 5080) wheel selection + PyTorch-2.6+ build patches (`.type()` removal, `weights_only` flip, `torch::linalg` namespace) overlapping MAtCha's patch list.
_Sources: notes 2026-05-01T165835-research-lessons-pytorch-rtx5080-shmsize, 2026-05-01T222605-operational-lesson-cuda-disappears, 2026-05-01T223139-beeprz-dash-…, 2026-05-01T230854-progress-reporting-…._

---
_Imported from legacy beads `m11-bz6` (M11 DAG re-import, 2026-06-03)._
