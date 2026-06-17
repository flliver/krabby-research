---
xid: STO-SCN-158
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-16
depends-on: []
bd-id: krabby-1rco
---

# Fleet-wide pull fan-out (docker-pull.yml) — EFF-REGISTRY-001 successor

## Summary

A one-command fleet rollout — given an image+tag (or "latest of each
family"), it wakes the sleeping GPU hosts, pulls the tag onto every
one in parallel, and prints a per-host sync matrix. This is the
"easily deployed & pulled" capability the operator asked for.

## Context

The audit confirmed (two ways: playbook contents + empty grep for
`docker pull`/`docker_image`/`community.docker`) that **no fan-out
exists**. EFF-REGISTRY-001 (baeprz, shipped 2026-06-10) deliberately
scoped to the registry + per-host trust; the last mile is a manual
per-host `docker pull`. `ops` scoped the successor as a small lift:
`docker-pull.yml` (WoL → parallel pull → matrix report).

**Ownership:** the Ansible playbook lives in baeprz's
`fleet/ansible/` and is **baeprz-owned**. This story is the
**cross-project ask + acceptance**, not the krabby-side
implementation — track via the liaison / a baeprz `ops` REQ.

## Problem

Without a fan-out, "ensure the latest is on all the fleet" is an N-host
manual chore that drifts immediately. A single command that pulls +
reports is the durable fix.

## Design

### Approach (baeprz `ops` implements; krabby specifies + accepts)

- `fleet/ansible/docker-pull.yml` (or a `fleet-pull` tag on the
  existing registry playbook): inputs = image+tag, or a "latest of
  each krabby family" manifest.
- WoL any sleeping GPU hosts first (note: **t resists WoL** — s2idle
  defeats the magic packet; needs a manual wake or a power-config fix,
  tracked separately).
- `docker pull j.pski.org:5000/krabby-<img>:<tag>` across t/b/d/s in
  parallel; emit a **per-host sync matrix** (host × family × tag/ID).
- Read-only-safe: pull is additive; never prune/retag without an
  explicit flag.

### Changes

| File | Change |
|------|--------|
| (baeprz) `fleet/ansible/docker-pull.yml` | new — WoL + parallel pull + matrix |
| krabby liaison REQ to baeprz `ops` | the cross-project ask + acceptance criteria |

## Definition of Done

- [ ] Cross-project ask filed to baeprz `ops` (via liaison) with the acceptance criteria below.
- [ ] One command pulls a given tag onto every reachable GPU host.
- [ ] It prints a per-host sync matrix (host × family × tag/ID).
- [ ] Pull is additive only (no implicit prune/retag).
- [ ] krabby has verified the command against a real tag (accepts the deliverable).

## Out of scope

- The actual sync run of the active path → STO-SCN-159 (consumes this).
- Fixing t's WoL/s2idle resistance (separate fleet-host concern).
- Building krabby's images (154–157).
