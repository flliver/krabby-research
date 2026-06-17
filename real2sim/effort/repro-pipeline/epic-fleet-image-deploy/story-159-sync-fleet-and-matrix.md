---
xid: STO-SCN-159
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-16
depends-on: [STO-SCN-157, STO-SCN-158]
bd-id: krabby-z49c
assignee: krabby
---

# Sync latest images onto all fleet hosts + produce drift/sync matrix

## Summary

The active scene-onboarding path (latest matcha/da3/fastmap) is pulled
onto every GPU host via the fan-out, and a per-host sync matrix
confirms the fleet is in sync — so the operator can vet scene
onboarding end-to-end across the fleet.

## Context

The terminal "ensure all the latest is on all the fleet" deliverable.
Depends on **STO-SCN-157** (a de-drifted fastmap to push — the current
registry `fastmap:0.2` is stale) and **STO-SCN-158** (the fan-out to
push it with). Audit baseline (2026-06-16):

- matcha `0.2.2-selfcontained` + da3 `0.4` — in sync on b/d/s.
- fastmap — in registry but on **zero hosts** (and 0.2 is stale until 157).
- **t** — asleep, WoL failed (s2idle); its inventory is the open hole.

## Problem

The fleet is drifted (each host has only what someone manually pulled),
and the one un-synced active-path image (fastmap) is also stale. Sync
everything and prove it.

## Design

### Approach

1. Re-check `docker ps -a` on each host immediately before sync (the
   pipeline uses ad-hoc `docker run --rm`; "no container now" ≠ idle).
2. Run the STO-SCN-158 fan-out for the active path: matcha latest, da3
   latest, fastmap (the STO-SCN-157 rebuild).
3. Sync **b / d / s** first. **HOLD tbeeprz (t)** — operator constraint
   (2026-06-16): *"don't deploy to tbeeprz until I give the go-ahead."*
   t's read-only inventory may still auto-complete via ops's watcher
   (that's allowed — it's not a deploy), but **no image pull to t** until
   the operator says go.
4. When the operator releases the hold, wake t (manual — WoL is defeated
   by s2idle) and pull onto it.
5. Capture the per-host sync matrix; attach it to this story as the
   acceptance artifact.

### Changes

| File | Change |
|------|--------|
| (this story) | the post-sync per-host matrix as the acceptance artifact |

## Definition of Done

- [ ] Pre-sync `docker ps -a` checked per host (no active job interrupted).
- [ ] matcha/da3/fastmap latest pulled onto t/b/d/s.
- [ ] t woken and included (or its absence explicitly noted with a reason).
- [ ] Per-host sync matrix captured showing the active path in sync fleet-wide.
- [ ] Operator can launch a scene-onboarding vet run on any GPU host with identical images.

## Out of scope

- mast3r/slam3r/vggt distribution (fallbacks; preserved in 156, synced only if the operator pulls them into the active path).
- The fan-out mechanism itself (STO-SCN-158).
