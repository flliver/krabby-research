---
xid: STO-SCN-157
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-16
depends-on: []
bd-id: krabby-nvge
---

# De-drift fastmap: re-sync baked krabby-tools, rebuild+push, add build-time sync guard

## Summary

`krabby-fastmap` is rebuilt from `krabby-tools/` re-synced to canonical
`real2sim/`, pushed as a fresh tag, and a build-time guard fails the
build if the baked tools ever drift from `real2sim/` again.

## Context

`images/fastmap/Dockerfile` bakes pipeline code via
`COPY krabby-tools/ /opt/krabby-tools/`, with a comment instructing a
manual sync from `real2sim/` before each build. The audit found that
sync has lapsed — two baked files are **stale** vs canonical:

| File | canonical `real2sim/` | baked in image | drift |
|---|---|---|---|
| `covis_graph.py` | 06-14 07:26 — adds `fwd` (camera optical axis) to the covis graph | 06-13 22:15 | image lacks `fwd` |
| `lib_progress.sh` | 06-14 17:59 | 06-13 22:15 | 54 lines behind |

So the registry `fastmap:0.2` runs stale covis logic. (matcha is
self-contained; da3's baked tools are in sync — verified.) The manual
sync step is the root cause and will keep re-drifting unless guarded
(T-003 — fix the root cause, not the symptom).

## Problem

The deployed fastmap is behind canonical, and nothing prevents the
baked-vs-canonical drift from recurring on the next build.

## Design

### Approach

1. Re-sync `images/fastmap/krabby-tools/` from `real2sim/` (covis_graph.py,
   lib_progress.sh, any others) and commit.
2. Rebuild `krabby-fastmap` on dbeeprz; push a new tag (e.g. `0.3`)
   to `j.pski.org:5000`.
3. **Add a build-time guard** so drift can't silently recur: either a
   `RUN` step that diffs baked tools against a manifest of expected
   SHAs, or a pre-build sync script that copies from `real2sim/` and
   `git diff --exit-code`s. Consider replacing the COPY-of-a-copy with
   a single-source mechanism (symlink/sync at build) so `real2sim/` is
   the only place these tools live (DRY, T-023).

### Changes

| File | Change |
|------|--------|
| `images/fastmap/krabby-tools/covis_graph.py`, `lib_progress.sh` | re-sync from `real2sim/` |
| `images/fastmap/Dockerfile` or a pre-build script | add drift guard (diff vs canonical, fail on mismatch) |
| `images/fastmap/README.md` | document the guard + the single-source rule |

## Definition of Done

- [ ] `images/fastmap/krabby-tools/` matches `real2sim/` (no drift).
- [ ] `krabby-fastmap` rebuilt + pushed; the new tag's `covis_graph.py` has the `fwd` field.
- [ ] A build-time guard fails the build if baked tools drift from `real2sim/`.
- [ ] README documents the single-source rule.

## Out of scope

- Distributing the rebuilt fastmap to hosts — that's the fleet sync (STO-SCN-159).
- da3/matcha (da3 tools verified in sync; matcha is self-contained).
