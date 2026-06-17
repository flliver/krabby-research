---
xid: STO-SCN-158
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-16
depends-on: []
bd-id: krabby-1rco
assignee: krabby
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

## Ownership split (operator directive, 2026-06-16)

> *"We [krabby] should own what is being deployed and how (excluding
> ansible); ops@baeprz should own the how (ansible) and which hosts."*

- **krabby owns** the **manifest** of what's deployed →
  `images/fleet-manifest.yaml` (active path + fallback + deprecated,
  with registry + tags). This is the source of truth; krabby keeps it
  current as images are built/bumped.
- **ops@baeprz owns** the **ansible fan-out** that consumes the
  manifest and the **host inventory** — the playbook reads
  `fleet-manifest.yaml`, pulls the declared tags onto the GPU hosts,
  and reports the matrix.

## Design

### krabby side (this story) — DONE

- `images/fleet-manifest.yaml` — the deployment contract: `registry`,
  `active_path[]` (matcha 0.2.2-selfcontained, da3 0.4, fastmap 0.3),
  `fallback[]`, `deprecated[]`. The playbook's input.

### ops side (baeprz `ops` implements — GREENLIT 2026-06-16)

- `fleet/ansible/docker-pull.yml`: read `images/fleet-manifest.yaml`
  → pull the declared tags onto the hosts → emit a per-host sync
  matrix (host × image × tag/ID). Default scope = `active_path`;
  `--fallback` to include fallbacks.
- WoL sleeping hosts first (note: **t resists WoL** — s2idle defeats
  the magic packet; manual wake needed, tracked separately).
- Additive only — never prune/retag without an explicit flag.

### Changes

| File | Owner | Change |
|------|-------|--------|
| `images/fleet-manifest.yaml` | krabby | new — the deployment manifest (DONE) |
| (baeprz) `fleet/ansible/docker-pull.yml` | ops | new — manifest-driven WoL + parallel pull + matrix |
| krabby → ops dispatch | krabby | the build ask + acceptance criteria |

## Definition of Done

- [x] krabby manifest `images/fleet-manifest.yaml` authored (what's deployed).
- [ ] Build ask filed to baeprz `ops` with acceptance criteria below.
- [ ] One command (ansible) pulls the manifest's tags onto every reachable GPU host.
- [ ] It prints a per-host sync matrix (host × image × tag/ID).
- [ ] Pull is additive only (no implicit prune/retag).

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
