---
xid: STO-SCN-029
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-04
depends-on: []
bd-id: krabby-1pa
assignee: devex
title: Scene sync — TWO layers: (a) credentialed dev sync 'krabby scenes' (LAN-first, rsync/rclone, NOT in krabby-launcher); (b) credential-free public-read client for public-tier scenes (firmware/cli.py pattern)
priority: 2
---

# LAN-first sync CLI (krabby scenes pull/push, manifest diff)

## Summary

_(One sentence: what does this story deliver? Avoid "we will add X" —
write the outcome, not the verb.)_

## Context

_(Why is this story needed? What does it depend on? Link to the parent
epic. If this is a discovered-from another story, surface the link.)_

## Problem

_(What specific problem does this story solve? Concrete; the reader
should be able to verify completion without re-reading the epic.)_

## Design

### Approach

_(How will this be implemented? Reference HUGs that constrain the
implementation choice; cite alternatives only when they shaped the
final pick.)_

### Changes

| File | Change |
|------|--------|
| `path/to/file` | _(add / modify / extract)_ |
| `path/to/test` | _(add tests for the new behavior)_ |

## Definition of Done

- [ ] _(Specific, verifiable condition — not "code works")_
- [ ] _(Specific, verifiable condition.)_
- [ ] Tests written and passing.
- [ ] Code reviewed (or self-reviewed against the engineer-knowledge
      constraints).
- [ ] `docs/work-platform.md` or other operator-facing doc updated if
      surface changed.

## Testing

### Unit / fixture tests

- [ ] _(Specific case.)_
- [ ] _(Edge case.)_

### Integration

- [ ] _(Scenario.)_

## Out of scope

- _(Things deliberately deferred to a later story. Be explicit — the
  reader should know what's *not* changing.)_

## Implementation Notes

_(Fill in during / after implementation. Capture what diverged from
the original design and why — useful for the retrospective + for
operators reading this story in a year.)_

### What Changed

_(Actual implementation. May differ from § Design above.)_

### Files Modified

- `path/to/file` — _(what changed)_

### Gotchas

_(Anything surprising or worth noting for future readers.)_

## Status notes

- 2026-06-09: **LFS transport to the j hub now works** (was never functional — only
  JSON-only pushes ever succeeded; the 39 GB store was seeded out-of-band). Installed
  `git-lfs-transfer` (charmbracelet v0.1.0, cross-compiled linux/amd64, sha256
  ac51174239c3…, provenance at j:/usr/local/share/git-lfs-transfer.provenance) to
  j:/usr/local/bin under operator fleet approval. T-020 verified both directions with
  real payloads: Mac→j push (30 MB points.ply), t→j push (2.4 GB, 5 tetra meshes,
  112 MB/s), j→t fetch (first ever). **Follow-up for devex/ops:** backfill the pinned
  install into the beeprz Ansible repo (baeprz-ops requested change-control); the ops
  delegate's sandbox blocked it from acting — its permission profile needs fixing too.
