# Krabby — Orientation (the 10-minute tour)

> You just landed in the **krabby** project. This is the
> how-we-work-here tour: what krabby is, how the repo + work are
> organized, who the agents are, and where things go. For the
> *technical* system map (runtime stack, HAL, parkour, real2sim) read
> its companion: [`architecture.md`](architecture.md) — this doc
> points there rather than restating it (T-023 DRY).

---

## What krabby is

A locomotion stack for the **Krabby hexapod** robot — a six-legged
(18-DoF) walker on a Jetson Orin + 3× Arduino Mega. The workspace
covers the robot software (firmware → HAL → policy), the RL training
that produces its gait policies (**parkour**), and an offline
scene-reconstruction pipeline (**real2sim**, the active milestone).
It also carries the non-code side of the project: **contracts, grants
(Patina), and milestone deliverables**.

Full technical map → [`architecture.md`](architecture.md).

## Repo at a glance

Top-level dirs you'll touch most: `firmware/`, `hal/`, `controller/`,
`compute/`, `krabby/` (the CLI), `parkour/`, `real2sim/`, plus
`docs/` (krabby's own engineering docs — **not** CCC's). Authoritative
layout lives in [`../../docs/FOLDER_LAYOUT.md`](../../docs/FOLDER_LAYOUT.md);
glossary in `docs/TECHNOLOGY_AND_TERMINOLOGY.md`. Per-dir one-liners
are in [`architecture.md`](architecture.md) §5.

## How work is tracked

- **Issue tracker is Beads**, via the `ccc-bd` wrapper (translates
  CCC XIDs ↔ raw Beads BIDs — you should never see a raw BID).
  - `ccc-bd ready` — unblocked work; `ccc-bd ready --assignee=<you>`
    for your queue. `ccc-bd list` / `ccc-bd show <XID>` to inspect.
  - Raw `bd` also works for quick local queries (see `docs/BEADS.md`,
    `DEVELOPER.md`).
- **Milestones** drive the project (M11 real2sim is active; M12
  hardware + M14 bench bringup are done). Milestone artifacts live
  under `milestones/` and per-area `effort/` folders.
- **Grants/contracts** map external milestone `OVERVIEW.md` structures
  to internal tracking — that bridge is the **manager**'s job.

## Who's who (agents)

Full table + models in [`../agents/ROSTER.md`](../agents/ROSTER.md).
The short version:

| Agent | When you want it |
|---|---|
| 🦀 krabby | Primary workspace agent — research, contracts, grants. |
| 📐 principal | Architecture & design, cross-cutting trade-offs. |
| 🔧 engineer | Implementation — robot runtime, firmware, deploy. |
| 📋 manager | Milestone & contract-compliance tracking. |
| 🔗 liaison | The **only** front door for cross-project requests in/out. |
| Σ ccc | CCC-platform questions, config audits, filing platform bugs. |

## Where things go

- **Cross-project work** (something another project needs from krabby,
  or vice-versa) → always through the **liaison**, never agent-to-agent
  across a project boundary. Use `/notify <project> <summary>`.
- **"Is CCC set up right / where does this CCC artifact go / this CCC
  thing is broken"** → the **ccc** agent (Σ). Platform *bugs* file via
  `/ccc-bug`; platform *shape* questions escalate to the central
  `expert`.
- **Persistent knowledge** worth outliving a task → the relevant
  agent's `knowledge/` folder, or project-level `.ccc/knowledge/` for
  facts many agents share (this folder). Don't duplicate — link.
- **Capturing a request as tracked work** → mint via `ccc-bd new …`
  (use the hardened skills/commands, T-025; don't hand-author Beads
  records).

## Conventions that bite if you skip them

- **T-020 — shipped is what the human verified.** Robot-facing and
  operator-facing surfaces aren't "done" because tests pass; the
  physical bench is the truth. Hardware-in-the-loop > mocks.
- **T-023 — DRY.** Shared facts live once and are linked. This doc and
  `architecture.md` are the canonical pair; point at them.
- **T-025 — use the hardened path.** A skill/command/`ccc-bd`
  subcommand exists for most operations — use it instead of
  reimplementing its steps.
- **Beads-first for handoffs** inside CCC-adopting flows; the legacy
  inbox folders are transitional.

## First moves when you arrive

1. `ccc-bd ready --assignee=<you>` — see what's queued for you.
2. Skim [`architecture.md`](architecture.md) for the system you're
   touching, then the "first files to read" list in its §8.
3. Check the active milestone (M11 real2sim) and `goals/active.md`
   for what matters right now.
4. Unsure where something goes? Ask the **ccc** agent (Σ) before
   inventing a local convention.
