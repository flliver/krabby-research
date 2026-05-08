---
kind: note
captured: 2026-05-01T22:26:05-07:00
consolidated: false
tags: []
---
# Operational lesson: long-running matcha-build containers can lose CUDA

What happened during the N=16 tuning experiment on tbeeprz: the second `train.py` invocation died ~6 sec in with `RuntimeError: No CUDA GPUs are available`. The error originated inside the container's free-Gaussians stage, propagated through 2DGS training, and left the run partially-failed (no tetra mesh).

Diagnosing: `docker exec matcha-build nvidia-smi` returned `Failed to initialize NVML: Unknown Error` — the container had lost GPU visibility despite the host's `nvidia-smi` showing the GPU healthy with 15 GB free.

**Cause (suspected):** known NVIDIA Container Toolkit hiccup. After the matcha-build container had been running for ~5+ hours through multiple GPU-using runs, it lost the cgroup/device passthrough. Discord on the host kept its GPU access fine; only the container was affected.

**Fix:** restart the container. Two minutes of work:

```bash
docker stop matcha-build && docker rm matcha-build
docker run -d --name matcha-build \
    --gpus all --shm-size=8g \
    -v /home/jeremy/outposts/krabby/data/011-scene-reconstruction:/data \
    -v /home/jeremy/scratch/MAtCha:/opt/MAtCha \
    krabby-matcha:latest sleep infinity
# Verify: docker exec matcha-build nvidia-smi -L  → should list the GPU
```

Then re-copy any docker-cp'd content (`/scripts`, etc.) since those live in the writable layer that's discarded with the container.

## Lesson for future sessions

- **Long-running containers can silently lose GPU access.** Always verify `torch.cuda.is_available()` is True at the start of every run, not just at container creation. The 6-second failure mode is fast; a robust runner script should fail-fast on this rather than write empty output dirs.
- **Restart is cheap.** ~2 min including the /scripts re-copy. Don't try to debug NVML errors in-place; just restart.
- **`/scripts` should be a bind-mount, not a docker-cp.** Saves the re-copy step on container restart. Worth restructuring next session.

## Recommendation

Add a "GPU sanity check" preamble to all run scripts. Before doing anything expensive:

```bash
docker exec matcha-build bash -c '
  source /opt/matcha/bin/activate &&
  python -c "import torch; assert torch.cuda.is_available(), \"NO CUDA — restart container\""
' || {
  echo "Container has lost CUDA. Run: docker stop matcha-build && docker rm matcha-build && docker run ... krabby-matcha:latest sleep infinity"
  exit 1
}
```

This converts a 6-second silent failure (with cryptic downstream errors) into a 2-second early bail with an actionable message.
