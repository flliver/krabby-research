---
xid: HUG-SCN-001
kind: hug
effort: scn
status: active
date: 2026-06-03
author: krabby agent handoffs, 2026-05-02
bd-id: krabby-o1x
title: TSDF >> adaptive tetrahedralization for visual mesh quality
---

# TSDF >> adaptive tetrahedralization for visual mesh quality

## Quote
> "TSDF looks AMAZING" — Jeremy, 2026-05-02 (after opening `scene_tsdf.blend`).

## Context
2026-05-02 finding: MAtCha's `--use_multires_tsdf` produces visibly better meshes than the default `extract_tetra_mesh.py` for scene 004 + bicycle data — **despite the MAtCha README labeling TSDF "(not recommended)"**. The website's bicycle hero shot is in fact the TSDF version. This finding invalidated every prior ranking (all collected on tetra-decimated-colored-culled meshes).

## Direction
- For **visual** mesh quality on M11 captures, SHOULD prefer MAtCha multires-TSDF (`--use_multires_tsdf`) over the default adaptive tetrahedralization.
- For **physics/collision** (Phase D/E): TSDF integration is not always watertight — MAY need V-HACD convex decomposition for collision proxies. Likely end-state: tetra (or V-HACD-on-TSDF) for collision + TSDF for visual.
- Any ranking/comparison MUST record which mesh source it used (see AIQ-SCN-001).

_Source: krabby/inbox/journal-tsdf-vs-tetra-recommendation.md, krabby/archive/handoff-2026-05-02-2210.md._
