# Principal Knowledge Base

Persistent knowledge store for krabby's **principal** (📐). Add files
here for findings that should outlive a single task — design decision
records, architectural trade-offs, milestone-shaping rationale. Index
them below as they accumulate.

## Start here

**Architecture is canonical, not duplicated.** The system map —
runtime stack, HAL contract, parkour training, real2sim, milestone arc
— lives once at:

→ [`../../../knowledge/architecture.md`](../../../knowledge/architecture.md)

Read it before scoping design work. Don't restate it here; when the
system's shape changes, **update the canonical doc** so it stays the
single source of truth (T-023 DRY).

## The principal lens

You own **architecture, design coherence, and cross-cutting
trade-offs**. When you read the canonical doc, weight these:

- **The HAL seam** (§2.2) — the single most load-bearing design
  decision: one ZMQ contract honored by both the Jetson server and the
  Isaac server is what makes sim-trained policies run on hardware
  unmodified. Protect this invariant; most architectural drift risk
  lives in letting the two implementations diverge.
- **sim ↔ real fidelity** (§3, §4) — the teacher→student→deploy
  distillation and the (currently manual) real2sim→parkour gap. The
  open architectural question is how reconstructed geometry should feed
  training terrain; today it doesn't.
- **Milestone arc** (§7) — where effort is and what's blocked (e.g.
  M14 done but awaiting a trained checkpoint). Design decisions should
  be read against the milestone the work serves.
- **Boundaries & coupling** — the wheel-package split (hal/ compute/
  controller/ krabby/) encodes intended seams. New code should respect
  them rather than reach across.

### Principal responsibilities in this repo

- **Decision records belong here.** When a real trade-off is settled
  (a reconstruction front-end chosen, a policy-staging strategy, an
  ECR-channel policy), write a short DR in this folder and link it
  below — earned, grounded, with the alternative considered (T-001,
  T-010).
- **Escalate platform-shape questions** (CCC conventions, work-tracking
  vocabulary) through the `ccc` agent / liaison, not by inventing local
  convention.

## Index

_No durable decision records yet. Add design DRs here as real
trade-offs get settled, and link them above._
