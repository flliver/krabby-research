# T4 — View Ranking

> Phase 7 (final) of [the M11 process](README.md). The operator ranks the rendered variants
> per view in the **studio UI** to decide which reconstruction wins. The ranking *is* the data
> — it's how a pipeline/settings choice is judged on real scenes.

## Inputs → Outputs

| In | Out |
|---|---|
| the T3c renders (variant × view) | `scenes/<scene>/scores.jsonl` (per-rater rankings) + a Borda leaderboard the UI aggregates |

## The studio UI

- **Studio server** at **`:8091`** — reach it via **`krabby.organl.com:8091`** (not
  `localhost`; same machine, but the canonical name survives IP changes and keeps rankings
  origin-consistent). Standalone `rate_renders` is `:8090`; studio embeds the same rank app at
  `/rank` (one implementation — fixes flow to both).
- **Server-side rankings** (STO-SCN-107): scores persist in the scene store
  (`scores.jsonl`), not per-browser — so a ranking made anywhere is available everywhere.
- **Profiles** (STO-SCN-108): pick your rater identity from the **profile pill** (upper-right of
  the title bar) — passwordless; any user can add one. No more guessing a rater string.

## How to rank

1. **Pick the scene** from the scene strip (selected card shows a white border). The UI
   auto-focuses the highest-ranked render.
2. **Pick the view** with the View card ◀/▶ (the "View X of Y" title). Every view is ranked
   independently; the leaderboard aggregates across views too.
3. **Set your profile** (profile pill) so the submission is attributed to you.
4. **Rank**: drag render cards from the **Pool (drag to rank)** into **tiers** (1 = best). Each
   card carries a **tier-letter badge** (STO-SCN-112) — a circle in the tier's color with a
   per-render letter (A, B, C…, unique across the view) in its upper-right — so you can match a
   render across the grid, the cards, and the live results.
5. **Submit ranking.** One submission per rater per view — **the latest overwrites**
   (STO-SCN-109). Submissions are server-side (persisted immediately).
6. **Read the runoff**: the **Live Results** panel shows the per-view leaderboard + overall
   (Borda); rows are clickable (focus that render). The **Manifest** panel shows how the focused
   render was built (STO-SCN-106); **Copy MD / Copy Link** export it with a deep-link.

## Reading a render's provenance

Each variant shows an ultra-succinct **Description** (how it was built — e.g.
`matcha@1 · dense-strong · tsdf · 11.4M tris` vs `da3@1 · 24v · spine-gauge npz`), derived from
the manifest. Rank the *pipeline*, not just the pixels.

## Gotchas

- **`scores.jsonl` is the data — commit it.** The rankings are the experimental result of the
  whole pipeline; they belong in git.
- **MISSING tiles** = renders that don't exist yet (T3c not run for that variant/view) — not a
  bad result. Run `v4job.py render-missing` to fill them.
- **Discarding an invalid render:** there is no built-in retract today — excluding a known-bad
  variant from the runoff needs a sanctioned step (rank-level exclusion or a `scores.jsonl`
  retraction via a migration tool, like the `dedup_scores.py` pattern). Don't hand-edit the
  store. (Live example: STO-SCN-124 discards `PWZ4S24AZ72T`.)

## Automation status

The ranking judgment is operator-driven by design (it's the human verdict, T-020); everything
feeding it is automated. 🟡 operator-in-the-loop (intentionally — this is the point of the phase).

## End of the process

A verified, ranked best-reconstruction for the scene. Downstream (out of this doc set): the
winning fused geometry hands off to mesh-conditioning → USD export for IsaacSim
(`EPI-SCN-MESH-CONDITION` → `EPI-SCN-USD-EXPORT`).
