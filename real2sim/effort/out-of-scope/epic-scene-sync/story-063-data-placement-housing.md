---
xid: STO-SCN-063
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-9vp
---

# Data placement — who houses what (and keeps it OFF everywhere else)

## Summary

One copy, one home, by data class:

| Data | Home | Everyone else |
|------|------|---------------|
| git history + LFS objects (tracked set) | **j** (bare hub) | clones fetch on demand |
| transient transformation data (full) | **Mac** `/var/krabby/scenes` (Archives-01) | fleet: NEVER retained |
| fleet job scratch | producing host, deleted after gather (gather-hygiene rule) | — |
| docker images | registry `j.pski.org:5000` | hosts pull tags they run |

## Done so far (2026-06-10)

- [x] Fleet auto-sync DISABLED via its config gate (t/d/s; b pending
      — host down).
- [x] Hub bare-ified: /games 100%→87%, 238 G freed; push verified.
      Proof protocol: fsck + 8,134/8,134 LFS pointers present +
      empty diff + Mac-superset ancestry check.

## Remaining

- [x] Slim fleet clones: t/d/s 454/318/319G → **72G each** (~700G
      freed fleet-wide). Pre-flight caught the edge case the protocol
      existed for: t held 11.2G of 005 chunk-01/08 solve byproducts
      existing NOWHERE else (host locked up before cleanup this
      morning) — gathered to the Mac archive BEFORE deletion.
      `git lfs prune` retained everything despite zeroed retention
      config (opaque); used direct cache removal instead — provably
      safe (fetch-only clones, j verified complete) and refetch-on-
      demand verified per host.
- [ ] b when back up: disable sync gate + slim (same sequence).
- [x] RECIPES § Storage policy documents the placement table.
