---
xid: EPI-SCN-USD-EXPORT
parent: ../design.md
kind: epic
effort: scn
status: open
date: 2026-06-03
hugs: []
tenets: []
bd-id: krabby-8vd
assignee: krabby
---

# Scale Calibration & USD Export

## Problem Statement

_(What problem are we solving? Why now? What's the impact of not
solving it? One paragraph — this drives the rest of the epic.)_

## Goals

- _(Specific, measurable outcome.)_
- _(Specific, measurable outcome.)_

## Non-Goals (Out of Scope)

- _(Explicitly excluded.)_
- _(Future work — deferred to a later epic, ideally linked.)_

## Context

**Source:** _(Where did this come from? Prior epic, user request, bug
report, AID directive — name it explicitly.)_

**Dependencies:**

- _(What must be true before we can start?)_
- _(External systems or features we need.)_

## Stories

| # | XID | Story | Status | Size |
|---|-----|-------|--------|------|
| 1 | `STO-SCN-016` | T2.E1 — Metric Scale Datum (control-distance primary, DA3 prior/gate) ★ BLOCKER | open | L |
| 1a | `STO-SCN-144` | └ Metric control-distance tool — two-view triangulation MEASURE mode (match.html) | open | M |
| 2 | `STO-SCN-017` | T2.E2 — Mesh-to-USD via Isaac Lab MeshConverter | open | L |
| 3 | `STO-SCN-018` | T2.E3 — IsaacSim Load + Robot Spawn + Depth Sensor Returns | open | M |

Metric scale (016) is the **datum** the rest inherit; 144 delivers its primary calibration
(hand-measured control distance). Canonical decision record: **STO-SCN-016 § Decisions (D1–D7)**.

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
