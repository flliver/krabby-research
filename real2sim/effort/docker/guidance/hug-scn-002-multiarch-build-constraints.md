---
xid: HUG-SCN-002
kind: hug
effort: scn
status: active
date: 2026-06-03
author: krabby handoffs 2026-04-29
bd-id: krabby-m74
title: Multi-arch (RTX 4080/5080) Docker build constraints
---

# Multi-arch (RTX 4080/5080) Docker build constraints for the reconstruction stack

## Context
Hard-won from building `krabby-mast3r:latest` (38.6 GB) and `krabby-matcha:latest` (33.9 GB). One multi-arch image runs on both Ada (4080) and Blackwell (5080) — no per-GPU builds.

## Direction
- Base on `nvcr.io/nvidia/pytorch:25.10-py3` (ships sm_75/80/86/90/100/120 kernels).
- `--shm-size=8g` is REQUIRED at runtime (silent deadlock without it).
- PyTorch porting: 2.6+ broke `tensor.type()` → use `.scalar_type()`; 2.9 removed `torch::linalg::*` → use `at::linalg_*`; 2.6+ flipped `torch.load` to `weights_only=True` (12 spots patched).
- CUDA 13 dropped sm_60/61/70. `faiss-gpu-cu12==1.14.1` lacks sm_120 kernels (CUDA error 209 on 5080) → use `faiss-cpu`. Do NOT install xformers (pulls torch 2.11 nightly, breaks pytorch3d ABI).
- MAtCha `--depthanything_encoder large` must be passed as `vitl` (encoder-name translation bug). 24-frame chart-alignment OOMs at 16 GB VRAM → `--n_images 12`.
- Full catalog: `docker/MAST3R-NOTES.md` + `docker/MATCHA-NOTES.md` (now under `images/<name>/`).

_Source: krabby/archive/handoff-2026-04-29-1347.md, krabby/archive/matcha-pipeline-integration.md._
