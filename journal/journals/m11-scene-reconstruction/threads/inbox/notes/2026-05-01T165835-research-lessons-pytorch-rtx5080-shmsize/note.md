---
captured: 2026-05-01T16:58:35-07:00
kind: note
consolidated: false
tags: [research-submodule, pytorch, rtx5080, docker, shm-size, lessons]
---

# Research-side lessons that bear on this milestone (commit 6b3f855)

Pulled in per Jeremy's explicit instruction during the SfM-scaling test setup: "make sure you pull in our lessons from krabby/research (the last commit)."

The last commit on `research/` (HEAD = 6b3f855) is "Fix PyTorch wheel index for RTX 5080 and document build constraints" — three documentation changes from real failures encountered while building MASt3R-SLAM across the project's RTX 4080 + RTX 5080 hosts in April 2026.

## The three lessons

### (1) PyTorch wheel index for RTX 5080: cu128, not cu130

**Failure mode:** cu130 stable wheels do not yet include sm_120 (Blackwell) kernels. cu130 + RTX 5080 fails at runtime with `"no kernel image is available for execution on the device"`.

**Resolution:** cu128 is the first stable PyTorch release with prebuilt sm_120 kernels. Works correctly even on a host with CUDA Toolkit 13.0 (PyTorch wheels bundle their own CUDA runtime).

**Status for our work:** the `krabby-matcha:latest` image was built with the cu128 stack per Dockerfile.matcha. Already correct.

### (2) `--shm-size=8g` is mandatory for PyTorch containers

**Failure mode:** Docker's default `/dev/shm` is **64 MB**, insufficient for PyTorch DataLoader / SLAM/pose-estimation backends. **Without `--shm-size`, the container starts, prints its config, and silently deadlocks at 0% GPU. No error message, no crash — just no progress.** Wasted multiple debugging cycles before being identified.

**Recommended flag set for any PyTorch + CUDA container** (per the new docs):

```bash
docker run --rm --gpus all \
    --shm-size=8g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ...
```

**Status for our SfM-scaling test:** verified live during the test setup. `docker inspect matcha-build` reports `ShmSize bytes: 8589934592` (= 8 GiB) and `df -h /dev/shm` inside the container shows 8.0G. Already correct. The deadlock-at-0%-GPU failure mode should not bite this test.

This is a critical lesson to keep in mind for **any future container provisioning**, not just MAtCha: if I ever spin up a fresh container for an experiment, the `--shm-size=8g` flag is mandatory or I will lose hours to a silent deadlock.

### (3) PYTORCH_GPU_SUPPORT.md exists as a cross-arch reference

A new doc at `research/docs/PYTORCH_GPU_SUPPORT.md` (157 lines) covers Jetson Orin, RTX 4080, and RTX 5080 architectures: wheel-index selection matrix, NGC container option for CUDA 13, and source patches required to build C++/CUDA extensions against PyTorch 2.6+ (`.type()` removal, `weights_only` flip, `torch::linalg` namespace removal, build-time vs runtime CUDA detection).

**Status:** worth knowing about as a reference if we ever need to rebuild MAtCha or extend to another host architecture. Not directly relevant to today's SfM-scaling test, but the patches it documents (`.type()`, `weights_only`) overlap exactly with our 8-patch list in `MATCHA-NOTES.md`. There may be consolidation opportunities later if we standardize PyTorch-ext build patterns across krabby-matcha, krabby-mast3r, and the locomotion containers.

## Implications for ongoing work

- **For the SfM-scaling test currently in progress:** matcha-build is already correctly configured. No changes needed.
- **For the future B5 tooling:** when we build a standalone-SfM wrapper container (or a viser-viewer container), it must inherit `--shm-size=8g` from the recipe.
- **For the krabby-matcha rebuild ever:** when we rebuild from Dockerfile.matcha, ensure cu128 stays. Check that cu130 hasn't been introduced.

## Status

Inbox note. To be moved to `matcha-quality` thread or kept here as cross-cutting. Neither thread is the perfect home — these lessons cut across infrastructure, not just MAtCha mesh quality. Possibly justifies a future `infrastructure` thread if such lessons accumulate.
