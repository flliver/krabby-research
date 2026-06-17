---
xid: EPI-SCN-FLEET-IMAGE-DEPLOY
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-16
hugs: []
tenets: [T-018, T-014]
bd-id: krabby-808u
assignee: krabby
---

# Reproducible image builds + fleet deployment

## Problem Statement

A 2026-06-16 read-only audit (krabby-side bake surface + a fleet
inventory by baeprz `ops`) found the M11 reconstruction images are
**neither reproducible from committed source nor easily deployed
across the fleet**. The build recipes for `krabby-matcha`/`krabby-da3`
live in **tmpfs on dbeeprz (`/tmp/{matcha,da3}-build`, wiped on
reboot)**; 28 files of MAtCha source customizations are uncommitted
against an upstream-only remote; `mast3r`/`slam3r`/`vggt` +
`011-scene-reconstruction` exist **only as local `:latest`,
registry-absent and diverged across hosts** (prune-fragile, exist
nowhere else); the baked `fastmap` tools have drifted from canonical
`real2sim/`; and there is **no fan-out** — the last mile is a manual
per-host `docker pull`. This is exactly DES-SCN-REPRO's "no more
prototypes / no luck-based reproduction" thesis applied to the
container layer, and it blocks the operator's goal of vetting the
scene-onboarding process across the whole fleet.

## Goals

- Every deployed image is **rebuildable from committed source** — no
  build input survives only in tmpfs or a dirty working tree (T-018).
- Every image family krabby uses is **present in the registry**
  (`j.pski.org:5000`) with a versioned tag — nothing lives only as a
  local `:latest`.
- A **one-command fleet rollout** pulls a given tag (or "latest of
  each family") onto every GPU host and reports a per-host sync matrix.
- The fleet is **in sync on the active scene-onboarding path** so the
  operator can vet onboarding end-to-end across hosts.

## Non-Goals (Out of Scope)

- Building the fleet-pull orchestration itself in baeprz's Ansible —
  that playbook is baeprz-owned; STO-SCN-158 is the cross-project ask,
  not the implementation.
- Migrating the production robot image (`krabby-locomotion` → AWS ECR)
  path — this epic is the LAN-registry research/M11 images only.
- Auto-distributing the host-side v4 orchestration layer (`v4exec`,
  studio, conditioning modifiers) — it is baked into no image and runs
  on `krabby.organl.com`; if that ever needs to run on GPU hosts it is
  a separate epic.

## Context

**Source:** 2026-06-16 fleet-image-deployment audit (this session),
prompted by operator: *"as we get the code stabilized I'd like the
images easily deployed to the fleet … make sure none of our code is
lost to /tmp … ensure all the latest is on all the fleet so we can
vet the scene onboarding process."* Audit ran read-only across the
registry + b/d/s (t asleep, WoL-resistant). Prior art:
**EFF-REGISTRY-001** (baeprz, shipped 2026-06-10 — registry + per-host
trust, deliberately no orchestration) and krabby **STO-SCN-101**
(deploy-fastmap-fleet).

**Dependencies:**

- baeprz `ops` owns the fleet hosts + the Ansible registry playbook
  (`fleet/ansible/docker-registry.yml`).
- Build host is **dbeeprz**; registry is **j.pski.org:5000** (LAN,
  plain-HTTP, `insecure-registries` on t/b/d/s).

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-154` | Rescue dbeeprz /tmp build recipes; make repo the build source | ✅ shipped | M |
| 2 | `STO-SCN-155` | Establish dev-loop (fast bind-mount iter + periodic re-image) | 🟡 awaiting `--dev-tools` go | L |
| 3 | `STO-SCN-156` | Push mast3r/slam3r/vggt to registry; preserve diverged locals | ✅ shipped | M |
| 4 | `STO-SCN-157` | De-drift fastmap: re-sync krabby-tools, rebuild+push, sync guard | ✅ shipped | M |
| 5 | `STO-SCN-158` | Fleet-wide pull fan-out (docker-pull.yml) — EFF-REGISTRY-001 successor | ✅ shipped | M |
| 6 | `STO-SCN-159` | Sync latest onto all fleet hosts + drift matrix | ✅ shipped | S |
| 7 | `STO-SCN-160` | Eliminate legacy 011-scene-reconstruction image | 📋 open (3 distinct builds preserved) | S |

Sequencing: **154 first** (preservation — the tmpfs recipes are one
reboot from gone). 155/156 are preservation+reproducibility and can
run in parallel. 157 de-drifts fastmap. **159 is the terminal sync**
and depends on 157 (a clean fastmap to push) + 158 (the fan-out to
push it with).

## Design

### Approach

Two phases, **preserve then deploy** (the audit's explicit ordering).
*Preserve* (154/155/156): get every build input and every
irreproducible image into durable, committed/registry storage before
anything is pulled, rebuilt, or pruned. *Deploy* (157/158/159):
de-drift fastmap, stand up the one-command fan-out, then sync the
active path across the fleet and capture the matrix.

### Architecture

Source of truth = the **krabby repo** (`images/<img>/` Dockerfiles +
patches + `krabby-tools/`) → built on **dbeeprz** → pushed to
**`j.pski.org:5000`** → pulled onto **t/b/d/s** by the fan-out. The
fan-out (158) is the missing edge between registry and hosts.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| dbeeprz reboots before 154 → tmpfs build recipes lost | Medium | High | 154 is first; immediate `docker save`/copy-off already dispatched to ops |
| `docker image prune` on a host erases the only copy of mast3r/slam3r/vggt | Medium | High | 156 rescues + pushes them before any prune is contemplated |
| Pushing a colliding tag name overwrites a diverged local image | Low | Medium | 156 saves diverged locals first; version tags, never `:latest` |

## Success Criteria

- [ ] No build input for any deployed image exists only in tmpfs or a dirty working tree.
- [ ] `mast3r`, `slam3r`, `vggt`, `011-scene-reconstruction` present in the registry with versioned tags.
- [ ] `fastmap` registry image rebuilt from synced tools (covis `fwd` present); a build-time guard fails if `krabby-tools/` drifts from `real2sim/`.
- [ ] One command pulls a tag onto every GPU host and prints a sync matrix.
- [ ] Active scene-onboarding path (matcha/da3/fastmap latest) in sync across t/b/d/s.
- [ ] All stories shipped.
