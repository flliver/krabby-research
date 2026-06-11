---
xid: STO-SCN-078
parent: ./epic.md
kind: story
effort: scn
size: S
status: draft
date: 2026-06-11
depends-on: []
bd-id: krabby-0ub
---

# License audit: every model + container in the chosen pipelines vs M11 deliverable use

## Summary

A verified license inventory for every model checkpoint and every
piece of third-party code baked into our chosen pipeline containers
(krabby-matcha, krabby-da3), with a deliverable-eligible verdict per
component — so M11 deliverable claims rest on checked licenses, not
suspicion.

## Context

Raised 2026-06-11 during the DAG-of-DAGs discussion
(EPI-SCN-PIPELINE-STUDIO): DA3's CC-BY-NC-4.0 is already tracked
(evaluation-only, flagged in the task catalog + repro_check), but the
**matcha trunk is unverified suspicion**: MASt3R is Naver (CC-BY-NC
territory) and MAtCha descends from the 3DGS/SuGaR lineage (Inria
research licenses). If both `represent` options are NC, the M11
deliverable path needs a third option or a commercial license — we
need to know NOW, not at delivery (T-002: verify, don't assert).

## Problem

Only DA3's model license is recorded. Nobody has verified, with
sources: MAtCha code license, its 3DGS/SuGaR/gaussian-rasterization
dependency licenses, MASt3R code + checkpoint licenses, MASt3R-SfM
components, or anything else baked into the two images. "Probably
non-commercial" is not a risk posture.

## Audit scope (one verdict each, with source link)

| Component | Where it lives |
|---|---|
| MAtCha source (incl. our patches) | krabby-matcha image, baked source |
| 3DGS / diff-gaussian-rasterization | MAtCha dependency (Inria license?) |
| SuGaR-derived code | MAtCha lineage |
| MASt3R / MASt3R-SfM code | krabby-matcha (also standalone pool-SfM use) |
| MASt3R checkpoints | baked weights |
| DA3 code (pinned 4173623) | krabby-da3 (verify code vs weights split) |
| DA3NESTED-GIANT-LARGE-1.1 weights | known CC-BY-NC-4.0 — confirm + record source |
| gsplat, Open3D, other libs | krabby-da3 (expected permissive — verify) |
| Our own tools (/opt/krabby-tools) | ours |

## Definition of Done

- [ ] Verdict table (component / license / source URL / deliverable
      eligible YES-NO / notes) committed to this story + referenced
      from the task catalog (`license_flag` per affected task def —
      matcha tasks currently carry NO flag; correct or confirm).
- [ ] repro_check deliverable-eligibility reflects the audited
      flags (it already gates on `license_flags`; the flags must
      now be COMPLETE).
- [ ] If the matcha trunk is NC: a one-paragraph options memo
      (commercial license terms? permissive-licensed `represent`
      alternative? customer-side research exemption per ICA terms?)
      routed to manager/contracts — this is a contract risk, not
      just a code fact.
- [ ] Disclosure language drafted if needed (this epic's purpose).

## Status Notes

- 2026-06-11: Minted from operator directive during
  EPI-SCN-PIPELINE-STUDIO DAG-of-DAGs discussion. Discussion-only
  session — audit not started.
