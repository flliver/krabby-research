---
xid: STO-SCN-132
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-qx6b
assignee: scout
---

# Track peak RAM/swap per task + record failed tasks as first-class results (OOM pseudo-result)

## Summary

Every GPU/CPU task records **peak RAM + peak swap** in its `measured` block (alongside the
existing `peak_vram_mib`, STO-SCN-039), and a task that **fails** (e.g. OOM) writes a
**first-class result record** — status `failed`, a human reason ("Out of Memory"), and the peak
memory it reached — so the failure is **visible in the data** (studio/manifest), not a silent
gap. You can *see* "matcha-15 tsdf@1024 → OUT OF MEMORY (peak 77 GB)" next to the successful
variants.

## Context / Problem

Two gaps, both surfaced 2026-06-15 on matcha-15:
1. **No peak-RAM tracking.** `measured` captures host/duration/digest (+ peak VRAM per
   STO-SCN-039), but **not peak system RAM/swap** — yet RAM is what OOM-kills the TSDF extract.
2. **Failures vanish.** When the matcha weld OOM'd, it printed "matcha weld FAILED" and wrote
   **no node** — so in the store/studio it's an *absence*, indistinguishable from "never run."
   The operator can't see *why* a variant is missing.

Measured evidence (TSDF extract on tbeeprz, 31 GB RAM + 64 GB swap):

| mesh_res | result | peak RAM | peak swap | notes |
|---|---|---|---|---|
| 1024 | **OOM-killed** | ~31 GB (avail→24 MB) | ~46 GB | `Terminated`; no mesh |
| 512 | success | ~30 GB (avail→932 MB) | 9.5 GB | 91 s, 10.3 M verts |

## Design

- **Peak memory capture.** Sample host `free` (RAM avail + swap used) during the remote run
  (or read the container cgroup `memory.peak`); record `measured.peak_ram_mb` +
  `measured.peak_swap_mb`. Mirror the `solve`/`scout` sampler harness (ties to STO-SCN-131).
- **Failed-result record.** When a task fails to produce its output, write a node/job record
  with `status: failed`, `reason` (classified: `oom` when RAM/swap hit the ceiling, else the
  log tail), and the peak memory. Studio/manifest render it as a **failed variant** (greyed,
  "OUT OF MEMORY", peak shown) — first-class, like the MISSING-render tiles (STO-SCN-085).
- **Don't block the graph.** A failed result is recorded but doesn't satisfy the node's
  identity (re-run still attempts it); it's a *diagnostic* artifact.

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | peak RAM/swap capture in `run_in_matcha`/`run_in_fastmap`/da3 infer; on failure write a `status: failed` record with reason + peak |
| `real2sim/v4job.py` / studio | surface failed results (reason + peak) as a variant tile |

## Definition of Done

- [ ] Every reconstruct/render task records `peak_ram_mb` + `peak_swap_mb` in `measured`.
- [ ] An OOM failure writes a visible `status: failed` result ("Out of Memory", peak shown) —
      e.g. matcha-15 tsdf@1024 appears as a failed variant in the data, not an absence.
- [ ] A successful re-run (e.g. @512) supersedes/sits beside it with its own peak.
- [ ] No graph corruption — failed records don't masquerade as completed nodes.

## Out of scope

- Choosing mesh_res to *avoid* OOM (STO-SCN-133 — memory-aware TSDF).
- VRAM tracking (already exists, STO-SCN-039).

## Implementation Notes

_(Earned 2026-06-15 on matcha-15: the 1024 OOM was invisible in the store; peak RAM was only
knowable via an ad-hoc `free` sampler. Operator: "produce a pseudo-result … stating Out of
Memory … so we can SEE that failure mode within the data" + "track peak RAM usage for tasks.")_
