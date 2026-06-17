---
xid: STO-SCN-141
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-15
depends-on: [STO-SCN-140]
bd-id: krabby-eylh
assignee: krabby
---

# Promote a best-of-breed ranked mesh experiment to a reproducible graph (re-run on a new scene)

## Summary

Once a modifier chain wins on the Rank leaderboard, **promote that experiment to a reusable
`graphs/*.json`** — the recipe (ordered modifier tasks + settings), scene-agnostic — so it can be
**re-run end-to-end on a different scene** and produce the equivalent conditioned mesh. This closes
the loop: explore (mix-and-match) → rank → standardize the winner → reproduce.

## Context / Problem

`graphs/*.json` already encode reproducible DAGs (`reconstruct-matcha` chains
represent→meshify→condition→render). What's missing is the path **from an ad-hoc winning experiment
(STO-SCN-140) to a named graph** — today a good recipe lives only as a one-off chain of node ids on
one scene. Promotion makes it a first-class, scene-portable pipeline.

## Design

### Approach
- **Promote = serialize the winning chain's recipe** (ordered `[{task, settings}]`, with the
  scene-specific base swapped for a graph **input binding**) into a `graphs/<name>.json` (or a
  `pipeline_instance`) — the same shape the studio already validates + plans.
- **Scene-agnostic by construction:** the recipe references **tasks + settings**, not node ids; the
  base mesh is a graph input (`from: meshify`). Re-running on scene B resolves B's own meshify output
  + applies the identical modifier settings → B's conditioned mesh. Identity is per-scene
  (content-addressed on B's inputs), so no cross-scene collision.
- **Reproduce:** `v4exec`/`v4job` runs the promoted graph on a new scene like any other graph;
  `expected_task_gaps` shows what's missing; re-runs NOOP.
- **Provenance:** the graph records which experiment/leaderboard rank it was promoted from (audit).

### Changes (sketch — drafting only)
| File | Change |
|------|--------|
| `real2sim/studio/*` | "Promote to graph" from a ranked experiment → emit `graphs/<name>.json` (recipe, base as input binding) + record source rank |
| `real2sim/v4exec.py` / `v4job.py` | run a conditioning graph on an arbitrary scene (resolve base meshify per-scene, apply the recipe) |
| `real2sim/graphs/<name>.json` | the promoted, scene-agnostic conditioning DAG |

## Definition of Done
- [ ] A winning ranked experiment promotes to a `graphs/*.json` capturing its ordered modifier recipe + settings.
- [ ] The promoted graph is **scene-agnostic** (base = input binding, not a fixed node id).
- [ ] Re-running it on a *different* scene produces the equivalent conditioned mesh (validated on ≥2 scenes).
- [ ] Provenance: the graph records the source experiment/rank.
- [ ] Backwards-compat: additive; existing graphs/identities untouched.

## Out of scope
- The composition + ranking surface (STO-SCN-140) and the modifier nodes (STO-SCN-139).
- Auto-selecting the winner (operator picks the best-of-breed from the ranks).

## Notes
This is the payoff of the modifier-as-task model: experiments are cheap + comparable (140), and the
winner becomes a durable, portable pipeline (this story). Mirrors how `reconstruct-matcha` already
encodes a reproducible chain — promotion just lets the *conditioning* tail graduate the same way.
