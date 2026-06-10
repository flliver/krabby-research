---
xid: EPI-SCN-CAPTURE
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-03
hugs: []
tenets: []
bd-id: krabby-zgq
assignee: krabby
---

# Scene Capture & SfM

## Problem Statement

T0 of the real-to-sim pipeline: turn raw captures (video, photos) into
posed camera sets the reconstruction stages can consume. Capture
quality decisions (frame budget, sharpness, normalization) made here
bound everything downstream — 001-patio's final verdict traced its
garbage reconstruction to a 12-frame coverage decision at this stage.

## Goals

- Every capture data type has a precise, hardened preprocessing path
  (`real2sim/RECIPES.md` — recipe per type, tool per step).
- Frame selection is tooled, spec-driven, and results-emitting — never
  re-freelanced per scene.

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
| 1 | `STO-SCN-001` | T0.B5 Frame-Selection Tooling (Camera Selection Viewer) | shipped 2026-06-03 | M |
| 2 | `STO-SCN-052` | Video preproc: recipe book + hardened sharp-select | in-progress | S |

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
