---
xid: STO-SCN-107
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-15
depends-on: []
bd-id: krabby-l2w
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

- [ ] Rankings are persisted in the scene store (`scenes/<scene>/{rankings,scores}.jsonl`),
      server-side, as the authoritative source — not dependent on a browser origin.
- [ ] Ratings made at `localhost:8091` are visible at `krabby.organl.com:8091` (same store,
      different origin) — the operator's concrete acceptance test.
- [ ] The **pre-existing** localhost ratings are migrated into the store and appear at the
      krabby URL.
- [ ] `localStorage` is cache/draft-only; losing it (new origin, cleared browser) does not
      lose any *committed* rating.
- [ ] Tests / a documented verification that two origins over one store agree.

## Testing

### Integration

- [ ] Rate at one origin → the rating appears (via the store) at a second origin without
      re-entering it.
- [ ] Restart the server / clear browser localStorage → committed rankings survive.
- [ ] Migration: existing localhost localStorage ratings land in the store and render at the
      krabby URL.

## Out of scope

- The per-render Description (STO-SCN-106).
- Auth / multi-user rater identity beyond the existing `rater` field.
- Serving infrastructure for `krabby.organl.com` (DNS/proxy) — assumed already routing to the
  same host/store; this story is about the *data* being origin-independent.

## Implementation Notes

_(Fill in during / after implementation.)_
