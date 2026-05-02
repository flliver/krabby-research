---
kind: note
captured: 2026-05-01T17:46:52-07:00
consolidated: false
tags: []
---
# N=500 hit the VRAM ceiling — bracketing strategy

The SfM-scaling experiment hit its first hard failure: **N=500 OOMed on tbeeprz at 15.45 GiB used** (the entire 16 GB GPU). Failed to allocate a 2 MiB request after 19:33 of compute. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was already in effect.

## Failure trace

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 0 has a total capacity of 15.45 GiB of which 79.50 MiB is free.
Including non-PyTorch memory, this process has 14.71 GiB memory in use.
Of the allocated memory 14.13 GiB is allocated by PyTorch, and 252.41 MiB
is reserved by PyTorch but unallocated.
```

So PyTorch held 14.13 GiB allocated + 252 MiB reserved, with the GPU full. Fragmentation isn't the issue (only 252 MiB lost) — it's an absolute volume problem.

## What the measurements say

After killing the foreign S&box process that was eating 4.3 GB on bbeeprz baseline, the SfM-only peak measurements line up consistently:

| N | host | peak VRAM (clean, MiB) | result |
|---|------|---:|---|
| 24 | bbeeprz | ~4622 (peak 8885 − 4263 S&box) | ✓ |
| 60 | bbeeprz | ~5758 | ✓ |
| 120 | bbeeprz | ~7511 | ✓ |
| 200 | bbeeprz | ~9984 | ✓ (post-kill, clean) |
| 500 | tbeeprz | 15674 (whole GPU) | ❌ OOM |

VRAM scaling: roughly +1–2 GB per doubling of N. Linear extrapolation from N=200 → 500 predicts ~14–15 GB, which matches reality. **The ceiling is dictated by the SfM optimizer's working set, not by encoder activations or scene-graph size.**

## Bracketing strategy

We have certainty at N=200 (works) and N=500 (OOMs). Need to find the largest N that fits.

Started **tbeeprz N=350** (midpoint of the bracket) immediately after the N=500 OOM. Decision tree:

- **N=350 succeeds** → bracket is [350, 500). Try N=425 next; refine until we converge on the limit.
- **N=350 OOMs** → bracket is [200, 350). Try N=275 next.

A few iterations gets us within ~25 frames of the ceiling, which is plenty of precision for planning a curation pool.

## Operational decisions made in response

1. **Watchdog on bbeeprz** to kill the chain after N=300 succeeds, preventing the queued N=500 from burning ~20 minutes before failing. Watchdog polls for `n300/mast3r_sfm/cameras.json`, then `kill`s the chain PID. Captured as `experiments/004-sfm-scaling-sky-house/scripts/kill_chain_after_n300.sh`.

2. **Cross-host validation of the boundary.** Both bbeeprz and tbeeprz are RTX 5080 / 16 GB / sm_120, running the same image. Whatever ceiling we find on tbeeprz should hold for bbeeprz (with the S&box overhead, bbeeprz's effective ceiling was lower; post-kill it should match tbeeprz).

3. **Decision: don't pursue N>500 even if 350 succeeds.** The reason isn't compute — it's that exceeding the global-SfM ceiling means switching to PnP localization or submap fusion (sibling notes). For M11's curation use case, anywhere in the 200–400 range is more than enough candidates.

## Where this leaves the answer to "what's the maximum?"

**Final result (2026-05-01T18:00):**

- **N=350 succeeded** at 15511 MiB peak VRAM — only 300 MiB free at the peak. 32:51 wall-clock. Cameras returned: 350.
- **Bracket therefore [350, 500)** — and we don't refine further; the answer is well-bounded for the use case.

**Practical operating zones:**

| N | Status | Headroom | Wall-clock | Recommendation |
|---|--------|---------:|-----------:|----------------|
| ≤300 | comfortable | ≥ 2.6 GB | ~28 min | safe everyday limit |
| 300-350 | borderline | 0.3 GB | ~33 min | works but no margin for env changes |
| 350-500 | unmeasured | — | — | likely OOMs near top of range |
| ≥500 | OOM (measured) | — | — | don't try |

**Final answer to Jeremy's question** ("how many video frames can we realistically position in 3D space for the purposes of selecting a subset?"): **300 frames is the comfortable operating point on RTX 5080 / 16 GB; 350 is the upper bound for practical use.** This is well above the human-curatable pool size (60-150), so the SfM ceiling never binds for B5.

## Lesson worth remembering

The reasoning that 16 GB hardware → expect linear scaling to break around the GPU's total capacity → expect ceiling at "roughly the largest N where peak ≤ 16 GB" was correct *and* the empirical curve confirmed it. Worth noting: this is a *clean* linear extrapolation only because we removed the S&box contamination. If we'd been measuring peaks contaminated by 4.3 GB of foreign overhead, our ceiling estimates would have been ~4 GB too pessimistic — predicting OOM around N=300 instead of N=400+. **Cleaning up the baseline mid-experiment was the right call.**

## Status

Complete. Experiment closed at 2026-05-01T18:00. Full results table in
`experiments/004-sfm-scaling-sky-house/README.md` and `results/results.tsv`.
