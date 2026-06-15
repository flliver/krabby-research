---
xid: STO-SCN-128
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-ieuh
assignee: scout
---

# Scene driver — orchestrate T0→T3c end-to-end (auto-resolve ids; pause at the human gates)

> **Scoping story.** Captures the design for a push-button-ish scene driver. Implementation is
> deliberately *not* started here — the DoD is an agreed design + a breakdown into build
> stories. Promote to its own epic if the breakdown warrants it.

## Summary

A single `v4exec scene-run <scene>` driver that takes a scene through the documented M11
process (T0→T3c) by **auto-resolving the content-addressed ids** between steps and **pausing at
the four intentional human gates** (capture decl, view authoring, verify, ranking) — turning
today's ~10 hand-chained commands into one resumable run. The point is removing the *incidental*
operator toil identified in `scene-processing/README.md` § "Automation status", not removing the
human from the loop.

## Context

The process is **documented and per-step automated** (each `v4exec`/`v4job` command is a
hardened, content-addressed NOOP-on-rerun step), but **end-to-end it is operator-orchestrated**:
the operator runs the steps in order and hand-carries `solve`/`covis`/`scout`/`subset`/rep ids
between them, re-points `primary` before each reconstruct, and works around per-step gotchas
(e.g. the FastMap-solve `cameras.json` render gap, STO-SCN-127). The DA3-24 variant for
001-patio was concretely *not* push-button — manual id lookup + two code fixes.

This story is the antidote: an orchestrator. It is the end-to-end evolution of the
**central run trigger** (STO-SCN-073), which launches *one* instance on *one* host; the scene
driver chains the *whole* pipeline.

## Problem

Operator toil + footgun surface the driver should eliminate:
1. **Id-chaining.** `solve → covis → select → scout`, then rep ids into reconstruct/render —
   all copied by hand from command output. Nothing auto-resolves "the latest covis for this
   solve."
2. **No single entry point.** There is no `scene-run`; the operator must remember the order +
   flags for ~10 commands.
3. **The `primary` re-point** before every reconstruct (a deliberately locked act — must stay
   *deliberate*, but the driver can surface it as an explicit confirm rather than a silent trap).
4. **Per-step gotchas aren't self-healing** — e.g. a FastMap-solve variant silently renders 0
   until its `cameras.json` exists. The driver should pre-flight these.

## Design

### Approach (proposed)

A `v4exec scene-run <scene> [--host tbeeprz] [--from <phase>] [--to <phase>]` driver that:

- **Resolves ids forward from scene state** — "the primary subset → its solve → its covis →
  its select/FINAL-N → its scout" — using the store refs (`primary`, `canonical`) +
  latest-by-content rules, so the operator never copies a hash. Where ambiguous, it lists
  candidates and asks.
- **Runs each phase as the existing command** (T-013 — it *calls* `ingest`/`solve`/`covis`/
  `select`/`scout`/`reconstruct-*`/`render-missing`, it does not reimplement them). Content-
  addressing makes each call idempotent, so the driver is **resumable** — re-running skips
  completed nodes.
- **Pauses at the four human gates** with a clear, actionable prompt (T-026):

  | Gate | Pause prompt |
  |---|---|
  | T0 capture decl | "no `capture.json` — declare make/model/mode/modality, then re-run" |
  | T2 view authoring | "author render cameras in the `.blend` (`/camera-save`), then resume" |
  | T1 verify | "open the verify viewer; accept/override the proposed-N, then resume" |
  | T4 ranking | "open studio `:8091`; rank the variants" |

  Gates are **resume points**: `scene-run` re-entered after the operator acts continues from
  where it paused (state read from the store, not a daemon).
- **Surfaces the `primary` re-point as an explicit confirm**, never a silent default (keeps the
  locked-#1 semantics while removing the footgun).
- **Pre-flights known gotchas** before a phase (e.g. ensure the solve has a `cameras.json`
  before render; check free VRAM before a GPU dispatch).

### Open questions (resolve during the build-story breakdown)
- **State model for pause/resume** — pure store-read (preferred; no daemon) vs a small run
  ledger. Lean store-read: the driver derives "what's done" from node existence.
- **Multi-variant runs** — one `scene-run` per (model, N), or a matrix mode that fans out
  matcha-15 + da3-24 + … in one invocation?
- **Spine (M>1) orchestration** — does `scene-run` drive segment→register→fuse, or is that a
  sub-driver? (Reuses STO-SCN-097…100.)
- **Relationship to the studio "central run trigger" (STO-SCN-073)** — CLI that the studio UI
  also calls, or studio-only? (Prefer CLI-first; studio wraps it.)

### Changes (anticipated — for the breakdown, not built here)

| File | Change |
|------|--------|
| `real2sim/v4exec.py` (or new `scene_run.py`) | `scene-run` orchestrator: id-resolver + phase sequencer + gate pauses |
| per-phase commands | minor: stable machine-readable "what id did I produce" output for chaining |
| `scene-processing/README.md` | flip the "Automation status" once the driver lands |

## Definition of Done (this is a SCOPING story)

- [ ] Design above reviewed with the operator; the id-resolution rules + the four gate
      pause/resume points are agreed.
- [ ] Open questions resolved (state model, multi-variant, spine, studio relationship).
- [ ] Broken down into implementation stories (e.g. id-resolver · phase sequencer · gate
      pause/resume · pre-flight checks · studio wrap), sized; **promote to an epic if warranted**.
- [ ] No implementation in this story.

## Out of scope

- Removing any of the four human gates (T-019/T-020 — they stay; the driver *pauses*, it does
  not auto-accept verify or auto-rank).
- Reimplementing any phase command (the driver orchestrates existing hardened commands).
- The actual build (that's the stories this one spawns).

## Implementation Notes

_(Scoping only — no implementation. Earned context: the need surfaced 2026-06-15 when the
operator asked "is the process fully automated?" and the honest answer was "per step yes,
end-to-end no" — see `scene-processing/README.md` § "Automation status".)_
