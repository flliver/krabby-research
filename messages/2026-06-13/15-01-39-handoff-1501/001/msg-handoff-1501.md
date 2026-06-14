---
xid: MSG-PROJ-006
content-path: /private/var/krabby/research/messages/2026-06-13/15-01-39-handoff-1501/001/msg-handoff-1501.md
kind: msg
effort: proj
status: open
date: 2026-06-13
to: manager
from: manager
topic: handoff-1501
bd-id: krabby-3ox
---

# Handoff from previous manager session

## What Was Happening

Deep M11 manager audit + a tracking-classification correction, driven by the
operator across several turns.

1. **Audited M11 vs the contract.** Read the ICA (`contracts/milestones/M11/M11.md`
   — note: contracts live at `/private/var/krabby/contracts/`, NOT the stale
   `workspace/contracts/` path in my role.md), the 5 Task acceptance files, PLAN.md,
   and extracted the live work DAG from `real2sim/effort/` (59 artifacts, now in
   `ccc-bd`/`STO-SCN-*`, migrated off the retired `research/.beads/` `m11-*` tracker).
   Scorecard: **T0 ✅ T1 ✅ (shipped) · T2 🟡 (all 6 stories open) · T3 ❌ · T4 ❌.**

2. **Operator reframed EPI-SCN-SCENE-SYNC.** I had wrongly called it out-of-scope/P4.
   Operator corrected: it's **discovered necessary work**. Final precise model (their
   words): the **scene schema + processing pipeline** half (025/026, shipped) *enabled
   T1/T2 production*; the **S3 sync/distribution** half (028/029/030/031) does NOT block
   T1/T2 production but **blocks their *delivery*** — maps to ICA §2 (merge-to-main),
   §7.2 (reproduce-on-sample-data), §7.3 (large-asset distribution). So it's
   in-scope-by-implication, a parallel delivery workstream, NOT ahead of the production
   critical path.

3. **Executed (committed `1495653`):** re-tiered the S3 slice 028/029/030/031/034 from
   P4 → P2 via `ccc-bd update --priority 2` (verified in frontmatter + Beads).

4. **Executed (uncommitted):** retitled `DES-SCN-TX` from "TX — Out of Scope Work" →
   **"TX — Discovered Work"** (Beads title + frontmatter `title:` + body H1 all updated).

## What Needs to Happen Next

Operator was working a numbered plan; (2) re-tier and title-change are done. Remaining:

- **Step (1) — re-home:** move the S3 slice out of `real2sim/effort/out-of-scope/`
  (dir still says out-of-scope though title now says "Discovered Work") into an M11
  delivery effort + add `milestone: M11`. **Pull the Lyra spike (035) OUT** into its
  own out-of-scope/M12 bucket — it's genuine scope creep, the odd one out now.
- **Step (3) — acceptance-gate edges:** wire 028/029/030/031 as `depends-on` of the
  M11 "ready-for-review" gate (mint one if absent); register the delivery risk under
  EPI-SCN-RISKS (manager-owned).
- **Step (5) — scope-discovery note to Fletcher**, framed as "delivery per §7.2 needs
  asset distribution we hadn't scoped as a task."

Standing tracker-integrity fixes still OPEN (flagged, not yet applied):
- **STO-SCN-016 (T2.E1 scale-calibration) is P3 but is THE production critical-path
  blocker** — should be P0/P1. Its dependents STO-SCN-017/018 are P0 with
  `depends-on: []` (missing edges). This priority inversion + missing edges will route
  someone to build wrongly-scaled USD. **#1 fix to recommend.**
- **R1 (Fletcher re-baseline)** vanished in the migration — re-mint as closed-w/-rationale.
- **PLAN.md + the 5 contract Task files reference the dead `m11-*` tracker** — refresh.

## Key Context

- **Production critical path is unchanged:** T2.E1 scale-cal → E2 → E3 → T3 → T4.F4
  (the deliverable, STO-SCN-022). Scale-cal is the gate; the whole back half sits behind it.
- **Scale-cal is genuinely unsolved** — no reference object in any capture; DoD untouched.
  Pragmatic path: hand-measure one distance in one scene, uniform-scale correct, validate
  vs a known-size primitive in IsaacSim. → krabby agent (pipeline work), not principal.
- **devex (krabby-devex pane) is FROZEN** on a `/receive` menu — its 7-item intake
  (EPI-SCN-DOCKER + 024 + the S3 slice 028-031/034) all carried `flag:label-shadow`
  (a cosmetic hygiene label-collision lint, NOT a mis-routing signal — assignments are
  clean). The agent over-read it. Real cause of its hesitation: the S3 work read as
  `out-of-scope/`+P4 "parking." The re-tier (done) + re-home (step 1) clears this. Tell
  devex: "yes these are yours, label-shadow is cosmetic, run `ccc-bd reconcile`."
- Scope creep is otherwise well-contained (Lyra at P4). Branch `jdp/m11-real2sim`.

## Active Files

- `real2sim/effort/out-of-scope/design.md` — retitled (UNCOMMITTED).
- `real2sim/effort/out-of-scope/epic-scene-sync/story-0{28,29,30,31,34}-*.md` — re-tiered (COMMITTED `1495653`).
- Reference: `contracts/milestones/M11/{M11.md,Task-0..4}`, `workspace/milestones/011-scene-reconstruction/PLAN.md`.

## Beads XIDs

No manager in-progress items. Relevant open artifacts:
- `DES-SCN-TX` — open; just retitled "TX — Discovered Work" (title change uncommitted).
- `STO-SCN-028/029/030/031/034` — open, now **P2** (committed), devex-owned, still in out-of-scope/ dir.
- `STO-SCN-016` — open, **P3** (SHOULD be P0 — scale-cal blocker), principal.
- `STO-SCN-023` — open, P2, manager — R2 tool-substitution disclosure (acceptance deliverable, undrafted).

## Status notes

- 2026-06-13: Parked mid-plan. Steps (2) re-tier + title-change done & (2) committed.
  Title-change uncommitted. Next: step (1) re-home + pull Lyra out.
