---
xid: STO-SCN-002
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-pq7
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.A — Pipeline Evaluation & MAtCha Selection
---

# T1.A — Pipeline Evaluation & MAtCha Selection

## Summary

Phase A foundational evaluation of 3D reconstruction pipelines for M11.

## Context

Phase A foundational evaluation of 3D reconstruction pipelines for M11.

Pipeline matrix evaluated: COLMAP MVS + Poisson (grant canon), MASt3R-SLAM, Spann3R, MAtCha. MAtCha selected as primary because it was the only candidate that reliably produced watertight meshes end-to-end, which is the T1 grant requirement.

Tool substitution from grant-canon COLMAP is defensible per grant Appendix A (which lists alternatives explicitly). Substitution must be disclosed in final M11 README — see bead "Risk: T0/T1 tool-substitution disclosure".

Evidence: PLAN.md Phase A; commits `12d9cba` (experiments tree + pipeline decision matrix), `09b69e1` (defer AnyRecon as not-the-right-tool), `c76153d` (MAtCha Docker recipe + patch playbook).

## Definition of Done

- [x] Pipeline matrix documented
- [x] Selection rationale captured
- [x] Tool substitution recorded for downstream disclosure


## Journal Notes

MAtCha selected after evaluating six pipelines (COLMAP, MASt3R-SLAM, SLAM3R, VGGT, MAtCha, AnyRecon) over late-April 2026: the only one producing a watertight mesh end-to-end with no separate conditioning step, ~11 min wall-clock on bbeeprz (RTX 5080) at 12 keyframes/scene. Phase A's three meshes were "chaotic but recognizable" — workable foreground, polluting distant background — consistent across very different captures, diagnosing the problem as structural not capture-specific. `--alignment_config strong` locked as default after a zero-cost test: identical 648 s wall-clock + ~7.87 GB peak VRAM vs default, ~3% more vertices, but visibly eliminated hallucinated/floating geometry (chart-encoding norm penalty + TV on depth encodings + confidence-weighting suppress over-fit to noisy SfM — "Option C lite"). SfM VRAM ceiling on RTX 5080/16 GB bracketed: N≤300 comfortable (~28 min), N=350 borderline (~33 min, 15511 MiB peak), N≥500 OOMs (15.45 GiB).
_Sources: entries 2026-05-01T144135-phase-a-…, 2026-05-01T222604-day-end-…; notes 2026-05-01T222604-strong-alignment-config-…, 2026-05-01T174652-n-500-hit-the-vram-ceiling._


## Handoff Notes

**How the MAtCha integration actually went** (krabby/archive/matcha-pipeline-integration.md): `train.py` chains four stages via `os.system` (`run_mast3r.py` → `align_charts.py` → `train_with_charts.py` → `extract_tetra_mesh.py`). Stages 2–4 initially failed (`FileNotFoundError: cameras.json`, then `AssertionError: Could not recognize scene type!`). Resolved 2026-05-01 with three fixes: pass `vitl` not `large` (encoder-name translation bug); `faiss-cpu` instead of `faiss-gpu-cu12==1.14.1` (no sm_120 kernels); `--n_images 12` (24-frame chart-alignment OOMs at 16 GB — OOM is in `cameras.py::project_points`). End-to-end ~11 min/scene (~8 min on 001/003). Runner `experiments/004-matcha-sky-house/runner.sh` (422 MB tetra), decimation `decimate.py` (21M-tri → 200K/500K).

**Negative result that redirected the work** (handoff-2026-05-01-1324.md): Phase B6a lowres-keyframes — dropping 1024×576 → 768×432 to fit 15 frames instead of 12 made meshes *worse* ("complete garbage. More lower-quality photos is certainly worse"). Per-pixel detail loss dominated view-count gain; drove the pivot to manual frame curation backed by MASt3R-SfM poses (→ STO-SCN-001).

**16-frame TSDF caveat** (handoff-2026-05-02-2210.md): scene-004's 16-frame variant uses the raw `multires_tsdf.ply` (not `_post.ply`) because the cluster-cleanup step OOMs (~18M verts, 30 GB host RAM). Rendered-region quality delta is minimal. See also **HUG-SCN-001** (TSDF >> tetra) and **AIQ-SCN-001** (rerank decision).

---
_Imported from legacy beads `m11-nk8` (M11 DAG re-import, 2026-06-03)._
