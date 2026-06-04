---
xid: STO-SCN-024
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-cuz
title: Open upstream PRs for the three research/ doc fixes (cd6a014)
---

# Open upstream PRs for the three research/ documentation fixes

## Summary
Push the three locally-committed `research/` documentation fixes (commit `cd6a014`) to Fletcher's repo via fork-and-PR. **Non-blocking for M11.**

## Context
Commit `cd6a014` ("Fix PyTorch wheel index for RTX 5080 and document build constraints") is committed locally but never pushed: `DEVELOPER.md` (cu130→cu128 + pointer), `docs/DOCKER_DEPENDENCIES.md` (new `--shm-size=8g` subsection), and new `docs/PYTORCH_GPU_SUPPORT.md` (12-constraint reference). Permission on `flliver/krabby-research` is READ, so fork-and-PR (`jeremyprz/krabby-research`) is the only path. Also flag Brian Refsdal's 2025-11-13 `DEVELOPER.md` cu130 line as misleading outside the NGC container.

## Definition of Done
- [ ] Fork `flliver/krabby-research` (or confirm existing fork)
- [ ] Three separate PRs (DEVELOPER.md / DOCKER_DEPENDENCIES.md / PYTORCH_GPU_SUPPORT.md) — different audiences
- [ ] Note to Brian Refsdal re: the misleading cu130 line

---
_Captured from krabby/archive/handoff-2026-04-29-1431.md + handoff-2026-05-01-1324.md (agents dir, pre-deletion)._
