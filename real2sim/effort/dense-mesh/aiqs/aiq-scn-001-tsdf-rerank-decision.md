---
xid: AIQ-SCN-001
kind: aiq
effort: scn
status: open
date: 2026-06-03
asks-of: operator
source: operator
bd-id: krabby-j3k
title: Handling of tetra-era rankings after TSDF swap on scene 004
---

# Handling of tetra-era rankings after TSDF swap on scene 004

## Context
`data/scenes/004-sky-house-dining/comparison_renders/` was rebuilt with TSDF meshes on 2026-05-02 (old tetra matrix archived at `comparison_renders_tetra/`); the 15 render cells are now TSDF-based. `rankings.jsonl` holds 3 rankings that are tetra-era and now semantically stale (`submitted_at < 2026-05-02 21:00`). After deciding, ~15 TSDF cells need re-ranking at http://localhost:8090 (scene `004-sky-house-dining`, ~10 min).

## Question
Pick one for the stale tetra-era rankings:
1. **Drop** them (one-line jq filter + commit).
2. **Keep** as historical (leaderboard mixes mesh sources; no code change).
3. **Tag** with `mesh_source: "tetra"` (needs `rate_renders/server.py` + frontend changes so the leaderboard can group/filter — would spawn a follow-up story).

## Answer
_(operator to fill)_

_Source: krabby/pending/tsdf-rerank-decision.md, krabby/archive/handoff-2026-05-02-2210.md._
