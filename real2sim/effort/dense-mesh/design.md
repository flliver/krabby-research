---
xid: DES-SCN-DENSE-MESH
kind: design
effort: scn
status: in-progress
date: 2026-06-03
guidance: ./guidance.md
hugs: []
tenets: []
bd-id: krabby-hot
assignee: principal
title: T1 — Dense Reconstruction
---

# T1 — Dense Reconstruction

## Overview

T1 of the real2sim arc: turn a posed scene (camera spine + frames) into **dense, deliverable
geometry**. It covers the reconstruction front-ends (MASt3R-SfM + MAtCha + TSDF/tetra, and the
feed-forward DA3 path), mesh orientation/culling to the canonical gravity gauge, and reference
localization — i.e. everything that produces a runoff-comparable mesh in a known frame.
Conditioning that mesh into a watertight, smoothed, scaled USD is the **next design**,
DES-SCN-COND-USD (T2), which depends on this one.

## Background

Authored 2026-06-03 to structure the dense-mesh work. The driving problem: MASt3R+MAtCha+TSDF
produces good-looking foregrounds with **holes** wherever sparse curated coverage doesn't
overlap. Fletcher (2026-06-10) flagged the feed-forward generation (VGGT-Omega, DA3) as
plausibly hole-free — which became EPI-SCN-FEEDFORWARD-RECON.

## Current State (at closeout, 2026-06-15)

The pipeline produces runoff-comparable meshes from both the matcha path (TSDF/tetra, oriented +
culled + color-projected) and the DA3 feed-forward path (gaussian render + TSDF mesh + a
matcha-free solve-gauge reconstruct). Meshes orient to the canonical gravity gauge (RANSAC + camera
up-vector), cull below-floor/skirt geometry, and carry source-projected color. DA3 is a
first-class runoff variant, operator-validated on DA3-24.

## Goals

- ✅ Dense reconstruction front-ends producing meshes in a known gauge (matcha TSDF/tetra + DA3).
- ✅ Mesh orientation, culling, and reference-camera localization to the canonical frame.
- ✅ Honest runoff comparison of all variants on the same scenes/views.

## Non-Goals (Out of Scope)

- Watertight merge/gap-fill, final smoothing, scale calibration, USD export → **DES-SCN-COND-USD (T2)**.
- Higher feed-forward fidelity beyond process_res 1008 (deferred T1 enhancement; STO-SCN-066).
- VGGT-Omega evaluation (DA3 satisfied the dense-hole-free goal; never required).

## Epics

| XID | Epic | Status | Implements |
|-----|------|--------|------------|
| `EPI-SCN-PIPELINE` | Pipeline eval + initial tetra/matcha meshes | shipped | HUG-SCN-001 |
| `EPI-SCN-MESH-PREP` | Mesh orientation, culling & reference localization | shipped | — |
| `EPI-SCN-FEEDFORWARD-RECON` | Feed-forward reconstruction (DA3) — dense hole-free geometry | shipped | — |

**AIQ:** `AIQ-SCN-001` (tetra-era rerank decision) — abandoned, moot/superseded (v1
rankings.jsonl/:8090 retired in favor of the v4 store + Rank UI).

## Closeout — shipped 2026-06-15

All three epics shipped; the one open AIQ was superseded by the v4 ranking store and abandoned.
Dangling in-progress children were reconciled honestly: feed-forward eval stories (061/066/127)
shipped on a met epic goal + operator-validated DA3-24; STO-SCN-068 (tetra conditioning) deferred,
its knob-sweep thread owned by T2's epic-mesh-condition (STO-SCN-015/136); STO-SCN-011/012
remain deferred. **No new DES-SCN-COND-USD stories were required** — T2 already carries every
conditioning straggler (013 merge/gap-fill, 014 watertight, 015 smoothing, 136 distant/sky cull).
Deferred T1 enhancements (DA3 >1008 fidelity, VGGT-Omega) are recorded in EPI-SCN-FEEDFORWARD-RECON
and are NOT T2 work (reconstruction, not conditioning).

## Key Concepts

_(Optional: domain-specific terminology defined for the reader. Skip
if the system uses only standard CCC concepts — link to
`docs/work-platform.md` instead.)_

## Related

- **Guidance:** [guidance.md](guidance.md)
- **Depends on:** _(other DESIGNs / external systems)_
- **Used by:** _(downstream consumers)_
