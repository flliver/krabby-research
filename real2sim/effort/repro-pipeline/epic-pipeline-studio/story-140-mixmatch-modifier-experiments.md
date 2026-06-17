---
xid: STO-SCN-140
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-15
depends-on: [STO-SCN-139]
bd-id: krabby-a3n8
assignee: krabby
---

# Mix-and-match modifier chains into content-addressed experiments that feed the Rank UI

## Summary

The operator can **compose an ad-hoc chain of mesh modifiers** (e.g. `tetra → filter → cull(cambox=0.7)
→ smooth`) on a given reconstruction, **materialize** it as a content-addressed node, and have it
appear in the **Rank UI** as a comparable variant — so different conditioning recipes compete on the
leaderboard, no graph authoring required.

## Context / Problem

Today each modifier is run as a one-off `v4exec` command and the result (a `condition/<id>` node) is
auto-discovered + rendered + rankable (proven this session: cull / refilter / cambox variants all
show up in Rank). What's missing is the **composition surface**: the operator hand-runs each step and
tracks ids by memory. This story makes the **chain** a first-class, repeatable, comparable thing —
the experimentation loop that precedes promotion to a graph (STO-SCN-141).

Builds on STO-SCN-139 (modifiers are selectable nodes) and reuses the proven facts: condition nodes
chain (a modifier can take another modifier's output as input), each is content-addressed (NOOP
re-run), and `v4job.mesh_targets` + the Rank server already surface `condition/*` nodes with their
per-task settings (STO-SCN-138).

## Design

### Approach
- **Experiment = an ordered modifier chain over a base mesh.** Each step is `{task, settings}`;
  the chain resolves to a sequence of content-addressed nodes (`meshify → condition/a → condition/b
  → …`), each NOOP if it exists. The terminal node is the experiment's rankable artifact.
- **Composition surface (studio):** pick a base reconstruction → append modifier nodes from the
  STO-SCN-139 palette (each with its settings form) → "materialize" runs the chain (CPU modifiers
  local; GPU ones like refilter dispatched) → the terminal node renders + lands in Rank.
- **Identity & dedup:** the chain's terminal identity is just the last node's content hash (already
  encodes the whole upstream lineage), so two operators composing the same chain converge on the same
  node — re-runs NOOP, no duplicate experiments.
- **Rank contribution:** terminal nodes already surface as variants (STO-SCN-138 shows each modifier's
  settings); this story ensures a composed chain is labeled by its **recipe** (the ordered
  task+settings list) so the leaderboard compares recipes, not opaque hashes.

### Changes (sketch — drafting only)
| File | Change |
|------|--------|
| `real2sim/studio/*` (+ `static`) | experiment composer: base mesh + appendable modifier nodes + "materialize" trigger |
| `real2sim/v4exec.py` | a chain runner that executes an ordered `[{task,settings}]` over a base mesh node (each step NOOP-aware), reusing cmd_cull/refilter/condition |
| `real2sim/rate_renders/server.py` | label a composed terminal node by its full recipe (ordered task+settings), not just the last task |

## Definition of Done
- [ ] Operator composes a modifier chain over a base mesh and materializes it without hand-running each `v4exec`.
- [ ] Every intermediate + terminal step is a content-addressed node; re-running the chain is all-NOOP.
- [ ] The terminal node appears in the Rank UI labeled by its recipe (ordered task+settings).
- [ ] Two identical chains converge on the same node (dedup); backwards-compat preserved (additive).

## Out of scope
- Promoting a winning chain to a reusable graph + cross-scene reproduction (STO-SCN-141).
- The modifier implementations themselves (their own stories).

## Notes
The mechanism already half-exists: condition nodes chain + are rankable + show settings (138). This
story is the **operator-facing composition + recipe-labeling** layer over it.
