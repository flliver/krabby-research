# Experiment 001 — VGGT on patio-fisheye (attempted)

**Status:** ❌ never produced output — VRAM ceiling
**Date:** ~2026-04-12
**Pipeline:** VGGT (Meta AI / Oxford VGG, CVPR 2025)
**Hardware:** Attempted on RTX 4080 / RTX 5080 (16 GB VRAM either way)
**Reference:** `docker/Dockerfile.vggt`, `tools/vggt/` (now deleted), OLAI corpus `3d-reconstruction/vggt`

## Input

- Same scene as experiment 001-COLMAP (patio fisheye)
- Frames extracted to `data/scenes/001-patio-fisheye/vggt_images/` (48 files, 94 MB)
- A "tiny" subset attempted in `data/scenes/001-patio-fisheye/vggt_images_tiny/`

## Process

Attempted via `workspace/run_vggt.sh` and `tools/vggt/demo_colmap.py`.
Local MPS patches were applied to `demo_colmap.py` for development on Mac.

## Result

Pipeline never produced output. **Out-of-memory** during VGGT's global
self-attention pass.

## Why it failed (the structural fact)

VGGT's architecture is a **transformer with global self-attention across
all input frames simultaneously**. VRAM cost scales **roughly
quadratically** with frame count.

| GPU | VRAM | Practical frame cap |
|-----|------|---------------------|
| RTX 4080 / RTX 5080 | 16 GB | ~50 frames |
| A100 / H100 (40+ GB) | 40+ GB | hundreds of frames |

For a 942-frame scene, even 16-fold subsampling lands above the cap.
The "tiny" subset attempt confirms this — we tried smaller batches
and still couldn't reach a useful reconstruction extent.

## Milestone fit

| Req | Score |
|-----|-------|
| All | ❌ — never produced output on M11 hardware |

## Key finding (preserved in OLAI corpus)

> **VGGT is structurally incompatible with consumer GPUs in 2026.** It
> requires 40+ GB VRAM for any non-trivial scene. This is not a bug or a
> tuning issue — it's the cost of the global-attention architecture that
> gives VGGT its "one-shot global consistency" property.

For consumer-GPU 3D reconstruction, MASt3R-SLAM (sliding-window SLAM,
fits in 16 GB) and MAtCha (sparse-view, fits in 16 GB at 12 frames) are
the practical choices.

## Why we kept the failed attempt

Captured here because future M11 readers may hear "have you tried VGGT?"
and we want a documented "yes, here's exactly why it doesn't fit" rather
than rediscovering the VRAM wall by trying again.

The local clone (`tools/vggt/`) was deleted after the OLAI corpus entry
captured the technical details and the MPS-patch findings, per
`AI/agents/krabby/archive/rotate-hf-token-slam3r-dockerfile.md` (which
was actually a separate cleanup; the VGGT clone deletion happened
independently).

## When to reconsider

If the project gets access to A100/H100 hardware, VGGT becomes worth
re-evaluating — its **one-shot global consistency** is a genuine
advantage over sliding-window SLAM that we can't currently exploit.
