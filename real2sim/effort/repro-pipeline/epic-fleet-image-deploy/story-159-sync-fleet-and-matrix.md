---
xid: STO-SCN-159
parent: ./epic.md
kind: story
effort: scn
size: S
status: in-progress
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
3. Sync **all four hosts (b / d / s / t)** to the active path:
   `matcha:0.2.2-selfcontained`, `da3:0.4`, `fastmap:0.3`. Pull is
   additive (the fan-out playbook STO-SCN-158 isn't built yet, so this
   runs as per-host `docker pull`).
4. **tbeeprz hold RELEASED** by operator 2026-06-16 ("synchronize
   tbeeprz as well"). t was s2idle/WoL-resistant all session, so the
   sync gates on t being online; before pulling, run t's read-only
   inventory + preserve any t-only images (closes the audit gap).
5. Capture the per-host sync matrix; attach it to this story as the
   acceptance artifact.

### Changes

| File | Change |
|------|--------|
| (this story) | the post-sync per-host matrix as the acceptance artifact |

## Definition of Done

- [x] Pre-sync `docker ps -a` checked per host (no active job interrupted).
- [x] matcha/da3/fastmap latest pulled onto t/b/d/s.
- [x] t woken and included (physical tap; WoL defeated by s2idle).
- [x] Per-host sync matrix captured showing the active path in sync fleet-wide.
- [x] Operator can launch a scene-onboarding vet run on any GPU host with identical images.

## Implementation Notes

### DONE (2026-06-16) — fleet synchronized, preserve-first

All four GPU hosts carry **byte-identical** active-path images, verified by
authoritative **RepoDigest** (cross-driver — b/d/s use the containerd image
store, t uses legacy overlay2; comparing by driver-local image ID is invalid):

| image:tag | RepoDigest | b | d | s | t |
|---|---|---|---|---|---|
| krabby-matcha:0.2.2-selfcontained | `sha256:aa5c9ab8a77a…` | ✅ | ✅ | ✅ | ✅ |
| krabby-da3:0.4 | `sha256:5a79314657c7…` | ✅ | ✅ | ✅ | ✅ |
| krabby-fastmap:0.3 | `sha256:a388fdffae10…` | ✅ | ✅ | ✅ | ✅ |

- **t** came online via a physical tap (WoL defeated by s2idle). Preserve-first
  honored: ops saved t's 5 registry-absent/unique `:latest` images before the
  additive pull (STO-SCN-156). t was already on canonical matcha+da3 (an earlier
  "divergence" was a driver ID-scheme artifact, since corrected); it only lacked
  `fastmap:0.3`, now pulled.
- Done **without** the fan-out playbook (STO-SCN-158 not built yet) — per-host
  `docker pull`. The playbook will make this a one-command op going forward.
- Nothing pruned or retagged.

**The scene-onboarding active path is now uniform fleet-wide — ready for
end-to-end vetting on any GPU host.**

## Out of scope

- mast3r/slam3r/vggt distribution (fallbacks; preserved in 156, synced only if the operator pulls them into the active path).
- The fan-out mechanism itself (STO-SCN-158).
