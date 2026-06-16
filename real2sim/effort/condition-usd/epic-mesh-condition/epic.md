---
xid: EPI-SCN-MESH-CONDITION
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-03
hugs: []
tenets: []
bd-id: krabby-khv
assignee: krabby
---

# Mesh Conditioning (merge, watertight, smoothing)

## Problem Statement

A raw reconstruction mesh (matcha tetra/tsdf, da3) is not yet a deliverable: it carries
sky/far-field junk and low-confidence floaters, isn't manifold/watertight, and is unsmoothed.
Mesh **conditioning** (T2.D) turns the raw materialized mesh into a clean, manifold, smoothed
mesh ready for USD export + physics — **as content-addressed v4 store nodes that reuse the
already-materialized upstream mesh (no GPU, NOOP re-runs) and never re-key historical artifacts**.

## Goals

- Every conditioning step is an **additive v4 task** (`algo@version`) consuming an upstream
  meshify/condition node's `mesh.ply`, placed at `{up_meshify_dir}/condition/{identity}`,
  auto-rendered + rankable, NOOP on re-run.
- The operator can dial cull / merge / smooth / scale knobs as tunables that flow into identity,
  with raw meshes preserved for comparison.
- **Backwards-compat is never broken** — no new key is appended to `meshify-via-tsdf`/`-tetra`;
  the canonical rule is STO-SCN-136 § "Backwards compatibility — store identity".

## Non-Goals (Out of Scope)

- Reconstruction front-ends / dense geometry (T1, DES-SCN-DENSE-MESH — shipped).
- USD export, scale calibration, IsaacSim spawn (EPI-SCN-USD-EXPORT, sibling epic in this design).
- Learned sky-segmentation masks (native cull knobs + camera-bbox should suffice first; STO-SCN-136).

## Context

**Source:** _(Where did this come from? Prior epic, user request, bug
report, AID directive — name it explicitly.)_

**Dependencies:**

- _(What must be true before we can start?)_
- _(External systems or features we need.)_

## Stories

Ordered by the operator (2026-06-15): **cull first** (136 → 137), then merge/verify/smooth.

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-136` | Cull distant/sky junk — meshify cull knobs (`cull-mesh@1`: min_views/max_dist/floor_z) | shipped | M |
| 2 | `STO-SCN-137` | Cull via posed-camera bounding box (`cull-mesh@1` cambox_expand) | shipped | M |
| 3 | `STO-SCN-013` | Merge & gap-fill (umbrella; realized via 142/143) | open | M |
| 3a | `STO-SCN-142` | └ **(A) Poisson** merge/gap-fill (`merge-gapfill@0`) — ★ PRIORITIZED | draft | M |
| 3b | `STO-SCN-143` | └ (B) TSDF re-fusion merge/gap-fill (`merge-gapfill-tsdf@0`) — ⏸ deferred | deferred | M |
| 4 | `STO-SCN-014` | Verify watertightness — genus/manifold report (`verify-watertight@0`) | open | M |
| 5 | `STO-SCN-015` | Final Taubin smoothing pass (`tetra-condition@0`/`taubin-smooth@0`) | open | S |
| 6 | `STO-SCN-145` | Camera-relative metric datum + boolean-primitive cull (extends 137; needs 016) | draft | L |

(Path-A tetra filter re-extract `tetra-extract-filtered@0` / `v4exec refilter` also shipped under STO-SCN-136.)

**Conditioning chain:** raw `meshify/{tetra,tsdf}/<id>` → cull (136/137) → merge/gap-fill (013) →
verify (014) → smooth (015) — each an additive `condition/<id>` node composing on the prior.
STO-SCN-136 § "Backwards compatibility — store identity" is the epic-canonical store rule.

## Design

### Approach

_(High-level technical approach. How will we solve the problem? Keep
to one or two paragraphs — details belong in story files.)_

### Architecture

_(Component sketch or short description of the key components touched.
For complex shapes, link to a diagram in the DESIGN.)_

### Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| _(Option A)_ | _(benefit)_ | _(drawback)_ | Rejected: _(reason)_ |
| _(Option B)_ | _(benefit)_ | _(drawback)_ | Selected |

## Decisions

| XID | Decision | Status | Rationale |
|-----|----------|--------|-----------|
| `HUG-scn-NNN` | _(major architectural decision)_ | Adopted | _(why)_ |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| _(risk)_ | Medium | High | _(how to reduce)_ |

## Success Criteria

- [ ] _(How do we know we're done?)_
- [ ] _(Measurable outcome.)_
- [ ] All stories shipped.
- [ ] Tests passing.
- [ ] `docs/work-platform.md` (or other consumer-facing doc) updated.

## Milestones

| Milestone | Target Date | Actual | Status |
|-----------|-------------|--------|--------|
| Stories defined | | | open |
| Implementation complete | | | open |
| Tests passing | | | open |

## Retrospective

_(Fill in after epic completion.)_

### What Went Well

-

### What Could Be Improved

-

### Lessons Learned

-
