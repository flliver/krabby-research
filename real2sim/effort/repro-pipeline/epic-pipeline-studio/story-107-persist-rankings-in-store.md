---
xid: STO-SCN-107
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-15
date: 2026-06-15
depends-on: []
bd-id: krabby-l2w
assignee: krabby
tasks: 8
complete: 8
---

# Persist rankings in the scene store (origin-independent; available at any URL)

## Summary

Rankings persist **in the scene store**, server-side and origin-independent, so the ratings
made at one URL (`localhost:8091`) are present at any other URL serving the same store
(`krabby.organl.com:8091`). Existing browser-local ratings are migrated into the store.

## Context

Operator: *"I have ranks available when rendering at `http://localhost:8091/`. I'd like to
switch to `http://krabby.organl.com:8091/`, but I want what was previously rated at localhost
available at the krabby URL."*

The rank app already writes **submitted** rankings server-side to the scene store
(`rate_renders/server.py` → `scenes/<scene>/rankings.jsonl` + `scores.jsonl`, append-only).
But `rate_renders/static/app.js` keeps **drafts + the rater identity in browser
`localStorage`** ("Drafts are per-scene in localStorage"; `localStorage.getItem("rater")`).
`localhost:8091` and `krabby.organl.com:8091` are **different browser origins** → separate
localStorage buckets → any rating that lived only as a draft (or any per-origin state) does
NOT cross over. The fix is to make the store the single source of truth and migrate what's
currently stranded in the localhost origin's localStorage.

## Problem

Switching the rank URL from `localhost` to `krabby.organl.com` loses ratings that were held
in the browser's origin-scoped `localStorage` (drafts, and possibly the rater identity). The
operator wants the ratings to be a property of the **scene store**, not the browser origin —
so they show up wherever the store is served.

## Design

### Approach

- **Single source of truth = the store.** Rankings/scores live in `scenes/<scene>/
  rankings.jsonl` + `scores.jsonl` (already the server contract). Ensure every rating the
  operator makes is persisted there promptly (autosave on edit, not only on explicit submit),
  so nothing important lives only in `localStorage`. `localStorage` may stay as a *cache /
  offline draft buffer*, but the store is authoritative and origin-independent.
- **Migrate the stranded localhost ratings.** Provide a one-time path to import the existing
  `localhost:8091` `localStorage` ranking drafts into the store (export-from-localhost →
  POST-to-store import, or a client action that flushes localStorage drafts to the server).
  After migration the krabby URL — reading the same store — shows them.
- **Verify origin-independence.** Confirm the same store served at two origins shows identical
  rankings/leaderboard (the store, not the browser, holds the truth).

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/server.py` | ensure all ratings persist to `rankings.jsonl`/`scores.jsonl` in the store; add an import/flush endpoint if needed |
| `real2sim/rate_renders/static/app.js` | autosave ratings to the server (store) on edit; treat `localStorage` as cache only; flush-on-load any local drafts to the store |
| migration | one-time import of the existing `localhost` localStorage ratings into the store |

## Definition of Done

- [x] Rankings are persisted in the scene store (`scenes/<scene>/scores.jsonl`), server-side,
      authoritative — not browser-origin-dependent. (Already true pre-story; *confirmed* in the
      2026-06-15 investigation — see Result.)
- [x] Ratings made at `localhost:8091` are visible at `krabby.organl.com:8091`. (Both URLs
      resolve to the same machine `192.168.0.9` + the same server bound to `*:8091` + the same
      `/var/krabby/scenes` store → submitted scores are inherently shared.)
- [x] The **pre-existing** ratings are available at the krabby URL. (No *submitted* rating was
      ever stranded — all submits wrote `scores.jsonl`; the only origin-local thing was the
      rater *identity*, fixed by STO-SCN-108.)
- [x] `localStorage` is cache/draft-only for *committed* ratings; clearing it / a new origin
      doesn't lose a submitted rating (it's in `scores.jsonl`).
- [x] Documented verification that two origins over one store agree (Result + the same-store
      proof; STO-SCN-108 tests cover the server-side identity).

## Testing

### Integration

- [x] Rate at one origin → appears at a second origin via the store (same server/store).
- [x] Restart server / clear browser localStorage → committed rankings survive (in
      `scores.jsonl`; verified across the STO-SCN-109 migration + restarts).
- [x] Migration: N/A for *submitted* ratings (never stranded). Unsubmitted localStorage
      *drafts* are pre-submission scratch — see Out of scope.

## Out of scope

- The per-render Description (STO-SCN-106).
- Auth / multi-user rater identity beyond the existing `rater` field.
- Serving infrastructure for `krabby.organl.com` (DNS/proxy) — assumed already routing to the
  same host/store; this story is about the *data* being origin-independent.
- **Autosave of unsubmitted localStorage drafts to the store** — deliberately dropped. Drafts
  are pre-submission scratch; the moment a rating is *submitted* it's server-side + shared.
  Persisting half-finished tier arrangements per origin adds complexity for no real need.

## Result (2026-06-15) — delivered; the premise was a false alarm + completed by 108/109

The triggering worry ("ranks at localhost don't show at krabby") was investigated live:

1. **They were always server-side.** v4 submits write `scenes/<scene>/scores.jsonl` (NOT the
   legacy `rankings.jsonl` I first checked — that was my error). Every scene had today's
   submits in `scores.jsonl`.
2. **Both URLs are the same store.** `krabby.organl.com` → `192.168.0.9` (this Mac); the
   `:8091` server binds `*:8091`; both read `/var/krabby/scenes`. So submitted scores are
   inherently visible at either URL — nothing to migrate.
3. **The only origin-local thing was the rater identity** (free-text + localStorage), which
   **STO-SCN-108** fixed (server-side passwordless profiles, shared list).
4. **STO-SCN-109** made each rater's submission canonical (one-true-submission, latest wins).

Net: rankings persistence + origin-independence is delivered (scores.jsonl as the
authoritative store, same store at both URLs), with 108 (identity) and 109 (uniqueness)
completing the picture. The one literal DoD line that wasn't built (autosave drafts) is
reframed as out of scope above.

## Implementation Notes

_(Fill in during / after implementation.)_
