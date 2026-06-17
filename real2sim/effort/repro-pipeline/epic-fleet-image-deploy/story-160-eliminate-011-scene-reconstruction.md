---
xid: STO-SCN-160
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-16
depends-on: []
bd-id: krabby-78q0
assignee: krabby
shipped: 2026-06-16
tasks: 4
complete: 4
---

# Eliminate legacy 011-scene-reconstruction image

## Summary

The legacy `krabby-011-scene-reconstruction(-cuda)` image family is
documented, confirmed superseded, and removed from the fleet — leaving
only the durable tar preservation as its record.

## Context

Operator directive (2026-06-16): keep legacy 011 **tar-only** (don't push
to the registry) and file a story to eliminate it. The audit found it
living only as local `:latest`, **diverged across hosts** (s vs d, and a
3rd copy on t), and superseded by the v4 active path
(matcha/da3/fastmap). It is now preserved as tars under
`<host>:/home/jeremy/preserve/EPI-SCN-FLEET-IMAGE-DEPLOY/images/` (s, d,
and t copies), so removal is safe.

## Problem

`011-scene-reconstruction` is dead weight: registry-absent, diverged,
~14 GB per host across b/d/s/t, and no longer referenced by the active
onboarding pipeline. It should be documented and removed so it stops
masquerading as live infrastructure.

## Design

### Approach

1. **Document what it was** — the pre-v4 reconstruction image (M11 "011"
   milestone): what it did, its entrypoints, and where it sat in the old
   pipeline. (Pull from the preserved tar's labels + git history.)
2. **Document the replacement** — how matcha (primary) + da3 + fastmap
   cover what 011 did; confirm nothing in the current pipeline still
   invokes a `011-scene-reconstruction` image (grep the repo + run paths).
3. **Removal plan** — once (2) confirms zero live references: `docker
   rmi` the `011-scene-reconstruction(-cuda)` images on each host
   (b/d/s/t). Tars remain the sole record. (Removal is a destructive op →
   operator-confirmed before execution; preservation already done.)

### Changes

| File | Change |
|------|--------|
| `images/` or docs | a short "011 retired" note: what it was, replaced-by, removed-on |
| `images/fleet-manifest.yaml` | already lists 011 under `deprecated:` — drop once removed |
| (hosts) | `docker rmi` 011 images after the no-live-reference check |

## Definition of Done

- [x] 011-scene-reconstruction documented (what it was + replacement map — below + manifest retirement note).
- [x] Confirmed zero live references in the repo / active pipeline.
- [x] 011 images removed from b/d/s/t (tars retained as the record).
- [x] `fleet-manifest.yaml` deprecated entry removed (replaced with a retirement note).

## Implementation Notes

### Removal DONE (2026-06-16, ops — verify-before-destroy)

Every `docker rmi` was gated on the image's content (`diff_ids`) matching its
preserved tar **and** no running container — nothing deleted that wasn't provably
preserved:

| host | removed | freed |
|---|---|---|
| s | 011-cuda `97b863a4` + 011-recon `d46a38d2` | 9.5 G |
| d | 011-cuda `a845f0ac` + 011-recon `49e6b45c` | 9.5 G |
| t | 011-cuda `7c619836` | 0 G (shared layers) |

011 remains **tar-only** (never in the registry, as intended) — the 5 distinct
tars under `<host>:/home/jeremy/preserve/EPI-SCN-FLEET-IMAGE-DEPLOY/images/` are
now the *only* copies. `fleet-manifest.yaml` `deprecated:` entry replaced with a
dated retirement note. Registry catalog unchanged (7 repos).

**What 011 was / replacement:** `krabby-011-scene-reconstruction(-cuda)` was the
pre-v4 ("011" M11 milestone) reconstruction image, superseded by the v4 active
path — **matcha** (primary), **da3**, **fastmap**. No code or run-path referenced
the image (verified), so removal is clean.

### No-live-reference check — PASSED (2026-06-16)

`grep -rin krabby-011-scene-reconstruction` across `*.py/*.sh/*.json/*.yaml/Makefile`
(excluding effort/provenance/manifest) → **zero hits**. Nothing in the repo or run
paths invokes the `krabby-011-scene-reconstruction(-cuda)` **image**.

⚠️ Disambiguation: there ARE many `011-scene-reconstruction` string hits — but those
are the **M11 milestone / data-dir namespace** (`outposts/.../data/011-scene-reconstruction`,
`milestones/011-scene-reconstruction/`), which is unrelated to the docker image and
**stays**. Only the *image* is being retired.

### Preserved builds (from STO-SCN-156 finalization)

ops's RootFS-DiffID fingerprinting found the 011 images are **5 genuinely distinct
builds**, all tar-preserved (registry-absent, kept):
- `011-scene-reconstruction-cuda:latest` ×3 — s `578518`, d `dd3bd7`, t `3e7662`
- `011-scene-reconstruction:latest` ×2 — s `be5952`, d `5fd9a6`

### Remaining (operator-gated — destructive)

- `docker rmi` the 5 011 images across b/d/s/t (tars are the sole record afterward).
  Destructive → needs operator confirmation before execution.
- Then drop the `deprecated:` entry from `fleet-manifest.yaml` + add the "011 retired"
  note.

## Out of scope

- Removing any active-path or fallback image — this is 011 only.
- The tar preservation itself (done under STO-SCN-156).
