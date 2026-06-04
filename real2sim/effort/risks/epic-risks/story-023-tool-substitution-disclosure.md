---
xid: STO-SCN-023
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-r4f
priority: 2
title: R2 — T0/T1 Tool-Substitution Disclosure
---

# R2 — T0/T1 Tool-Substitution Disclosure

## Summary

**Per Manager memo 2026-05-06 (top-3 risks #3):** grant text canonicalizes COLMAP MVS + Poisson; we shipped MASt3R-SfM + MAtCha (TSDF). Substitution is defensible per grant Appendix A, but must be disclosed in the final M11 README so the Client doesn't have to discover it on review.

## Context

**Per Manager memo 2026-05-06 (top-3 risks #3):** grant text canonicalizes COLMAP MVS + Poisson; we shipped MASt3R-SfM + MAtCha (TSDF). Substitution is defensible per grant Appendix A, but must be disclosed in the final M11 README so the Client doesn't have to discover it on review.

Required: one paragraph in the final M11 README explaining the substitution, citing Appendix A, and pointing to the COLMAP-format export script for anyone who'd prefer to re-run the canonical pipeline.

Owner: Manager (drafts); krabby agent (reviews for technical accuracy).

## Definition of Done

- [ ] Paragraph drafted
- [ ] Cites Appendix A explicitly
- [ ] Documents COLMAP-export path
- [ ] Lands in final M11 README before Client review


## Journal Notes

Substitution rationale is well-grounded. Pipeline = MASt3R-SfM (unposed SfM) → MAtCha (Guédon et al., CVPR 2025), substituting for the grant's COLMAP because MAtCha was the only one of six evaluated pipelines producing a watertight mesh end-to-end with no separate conditioning, ~11 min on RTX 5080. MASt3R-SfM provides camera poses + cross-chart scale alignment; per-pixel surface detail comes from MAtCha's monocular-depth-initialized charts (DepthAnythingV2), not cross-view triangulation. MASt3R-SfM characterized to N=350 cams on our hardware and validated as scaling with more frames (paper T&T ATE 0.034→0.011 from 25→full views). Source: `Anttwo/MAtCha`, 8 build-time patches in `docker/patch_matcha_*.py` + `MATCHA-NOTES.md`, none touching chart-deformation or photometric-resolution algorithms. Defensible per grant Appendix A; must be disclosed in the final M11 README.
_Sources: entries 2026-05-01T144135-phase-a-…, 2026-05-01T144205-b6a-…; notes 2026-05-01T164453-matcha-source-code-read, 2026-05-01T161229-mast3r-sfm-scaling, 2026-05-01T163958-bbeeprz-access-path._

---
_Imported from legacy beads `m11-kr9` (M11 DAG re-import, 2026-06-03)._
