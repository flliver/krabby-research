---
xid: STO-SCN-080
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
date: 2026-06-11
depends-on: []
hugs: [HUG-SCN-005]
bd-id: krabby-9gd
shipped: 2026-06-11
tasks: 5
complete: 5
---

# Full-restructure migration: compute legacy identities (algo@0), move files, scores.jsonl, jobs-logged (locked #9)

## Summary

Full-restructure migration of the scene store to content-addressed
v4 (HUG-SCN-005 locked #9): no legacy residue, compute preserved,
identities computed — not invented.

## Shipped (2026-06-11, store commit 9d6297b)

- All 14 scenes restructured: videos/ + images/<hash> pool + subsets
  (HOH) + cameras/<solve>/orient/<orient> + views + viewset/canonical
  + represent/<kind>/<RID>/meshify/<MID>/condition/<CID>/renders/
  <REND> + scores.jsonl + jobs/.
- 5,331 renames (LFS objects reused — compute preserved), 6,181 adds
  (metadata + newly-tracked pool images), 66 deletes.
- 45 GB new LFS to the hub: the canonical image pools (v2 ignored
  preproc-data frames; v4 tracks the pool as inputs BY DESIGN —
  subsets reference image hashes, so the pool is the provenance
  anchor). Tracked set grows ~55 GiB → ~100 GiB; j has 1.2 T free.
- Identities from real specs, retroactive algo@0; refs: primary set
  per scene (013 = pool-200 with its real pool-solve migrated);
  rankings → scores.jsonl on identities (006 leaderboard reproduces
  the operator runoff verdict); migration logged as jobs.
- T-002: legacy-era runs (colmap/mast3r/vggt/slam3r) carry explicit
  UNKNOWN subsets; migrated metadata is marked, and repro_check
  fails migrated artifacts for M11 gating while letting them rank.
- Idempotent: re-run on a migrated scene = 0 actions (verified 006).

## Gotchas (paid for)

- gitignore patterns with `/` anchor at the .gitignore dir — scene
  prefix `*/` required (caught before any commit).
- NEVER mutate the store while `git add` runs (two staging races).
- Same-identity reruns collapse into one dir (sc038=8-strong repro,
  studio=8-giant) — CORRECT per the model; variant labels merged to
  `legacy_variants` lists; duplicate payloads kept as origin evidence.
- 006 needed a repair pass (mid-development re-applies minted
  placeholder-subset dirs) — merged, artifacts promoted, scan clean.

## Definition of Done

- [x] All scenes in v4 shape; zero non-spine legacy files.
- [x] Compute preserved (renames, not re-uploads).
- [x] Scores carried; leaderboard verdicts reproduce.
- [x] Store committed + pushed to hub (9d6297b).
- [x] Consumers ported and verified against the migrated store.

## Follow-on (ops)

- Fleet re-clone: t/b/d/s hold stale v2 checkouts; re-clone after
  this push (mostly renames; tracked set ~100 GiB).
