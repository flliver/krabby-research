---
xid: AIQ-SCN-001
kind: aiq
effort: scn
status: abandoned
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

**Moot / superseded — abandoned 2026-06-15 (DES-SCN-DENSE-MESH closeout).** The question's
entire premise no longer exists: the v1 `rate_renders/rankings.jsonl` + `localhost:8090`
ranking system and the `data/scenes/004-sky-house-dining/comparison_renders*` layout have been
**retired**. Rankings now persist in the v4 content-addressed store (STO-SCN-107, shipped) and
are served by the studio Rank UI on `krabby.organl.com:8091` — keyed on content-addressed
reconstruction identity, not a flat jsonl of mesh-source-ambiguous cells. Verified 2026-06-15:
no `rankings.jsonl` anywhere in the repo, and `data/scenes/004-sky-house-dining` is gone.

There are therefore no tetra-era jsonl entries to drop/keep/tag, and no `:8090` re-rank to run.
Option (3)'s `mesh_source` tagging concern is structurally answered by the store's identity
model. No operator action required.

_Source: krabby/pending/tsdf-rerank-decision.md, krabby/archive/handoff-2026-05-02-2210.md._
