---
xid: STO-SCN-109
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-15
date: 2026-06-15
depends-on: [STO-SCN-107]
bd-id: krabby-exn
assignee: krabby
tasks: 7
complete: 7
---

# One submission per ranker (latest overwrites) + historical scores dedup

## Summary

A ranker has exactly **one** submission per (scene, view): re-submitting **overwrites** their
previous one. The leaderboard counts each person once. Historical `scores.jsonl` is rewritten
so the **most recent** submission per (rater, view) is the one-true-submission.

## Context

Operator: *"Disallow multiple submissions from the same person. Re-submission overwrites the
user's previous submission. Examine all historical submissions and rewrite with the most
recent being the one-true-submission."*

v4 submits append per-variant rows to `scenes/<scene>/scores.jsonl`, tagged `(ts, rater,
slot)`. The reader groups by `(ts, rater, slot)` and returns **every** group — so a person who
submitted N times for a view is counted N times, skewing the Borda aggregate (001-patio had
~5 "Jeremy" submissions for slot 01). There's no uniqueness constraint.

## Problem

Multiple submissions by the same rater for the same view all persist and all count. Re-ranking
should *replace*, not accumulate. And the existing history must be collapsed to one-per-person.

## Design

### Approach

Key a submission by **(rater, slot)** (= person × view); the latest `ts` wins.

1. **Submit = overwrite** (`_handle_post_ranking`, v4): read `scores.jsonl`, drop the rows for
   this `(rater, slot)`, append the new rows, rewrite the file. So a re-submit replaces.
2. **Read = dedup** (`_read_rankings`, v4): keep only the latest `(ts)` group per `(rater,
   slot)` — belt-and-suspenders so the aggregate is one-per-person even if stray rows linger.
3. **Historical rewrite** (one-time migration over every scene): collapse each `scores.jsonl`
   to the rows of the latest `ts` per `(rater, slot)`. The store is git-tracked, so the
   pre-dedup state is recoverable (T-018 — no separate backup needed).

Shared helper `_latest_score_rows(rows)`: keep rows whose `ts == max ts for their (rater,
slot)` (ISO8601 strings sort correctly under a single tz offset).

### Changes

| File | Change |
|------|--------|
| `real2sim/rate_renders/server.py` | `_latest_score_rows`; submit rewrites (drop prior (rater,slot) + append); read dedups to latest per (rater,slot) |
| `real2sim/rate_renders/dedup_scores.py` | one-time migration: rewrite all scenes' `scores.jsonl` to latest-per-(rater,slot) |
| `real2sim/tests/` | dedup helper + overwrite + read-dedup tests |

## Definition of Done

- [x] A ranker has one submission per (scene, view): re-submit overwrites. (`_handle_post_ranking`
      v4 now reads scores.jsonl, drops this (rater, slot)'s prior rows, writes back + appends
      the new set. Verified: double-POST `(TESTER, 01)` → only the 2nd persisted.)
- [x] The leaderboard/aggregate counts each rater once per view. (`_read_rankings` v4 collapses
      to the latest submission per (rater, slot) via `_latest_score_rows`. Verified: 001-patio
      → 1 submission group for Jeremy.)
- [x] All historical `scores.jsonl` rewritten to the latest submission per (rater, slot).
      (`rate_renders/dedup_scores.py` migration: **338 → 72 rows** across 13 scenes; `__diag__`
      test rows dropped. Store git-tracked → recoverable.)
- [x] Tests: `_latest_score_rows` dedup (latest-per-(rater,slot); distinct raters/slots
      preserved); read counts once; submit-overwrite. (`tests/test_scores_dedup.py` 3/3.)

## Testing

### Unit
- [x] `_latest_score_rows`: 3 submissions by one rater for one slot → only the latest's rows.
      (`tests/test_scores_dedup.py::test_keeps_only_latest_submission_for_a_rater_slot`.)
- [x] Distinct raters / distinct slots are preserved (not collapsed together).
      (`test_distinct_raters_and_slots_preserved`.)

### Integration
- [x] Submit twice for the same (scene, view, rater) → `scores.jsonl` holds only the 2nd;
      aggregate counts once. (Verified live: double-POST `(TESTER, 01)` → only the 2nd;
      001-patio read → 1 group for Jeremy.)

## Out of scope

- Auth (anyone can still submit as any profile — STO-SCN-108's passwordless model).
- Cross-rater dedup (different people keep their own one-true-submission).

## Implementation Notes

**Built (2026-06-15).** Key = **(rater, slot)**; latest `ts` wins.
- `server._latest_score_rows(rows)` — keep rows whose `ts == max ts for their (rater, slot)`.
- **Submit** (`_handle_post_ranking`, v4): rewrites `scores.jsonl` = (rows minus this
  (rater,slot)) + new rows → re-rank overwrites instead of accumulating.
- **Read** (`_read_rankings`, v4): runs `_latest_score_rows` before grouping → aggregate
  counts each person once per view even if stray older rows linger.
- **Historical**: `rate_renders/dedup_scores.py` (one-time; `--dry-run` supported) collapsed
  every scene to latest-per-(rater,slot) + dropped `__diag__` test rows — **338 → 72 rows**.
  The store is git-tracked, so the pre-dedup state is recoverable (T-018; no separate backup).

**Verified.** Migration 338→72; double-POST overwrite (only the 2nd persists); 001-patio read
→ 1 group for Jeremy. `tests/test_scores_dedup.py` 3/3. Studio restarted to load the new
submit/read logic. (Cleaned the `TESTER`/`__diag__` test rows via the migration — the store
guard correctly blocks hand-edits, so cleanup went through the sanctioned tool.)
