---
xid: STO-SCN-074
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: [STO-SCN-072, STO-SCN-073]
bd-id: krabby-3v9
---

# Ranking absorbed: renders→rank→scores on pipeline_run; leaderboard across instances

## Summary

The "rank→decide" step, absorbed into Studio (operator decision 2):
each pipeline_run yields N comparison renders; the operator ranks
them; rankings persist as scores ON the pipeline_run; a leaderboard
per scene/view compares competing pipeline_instances so the winning
configuration — and its settings delta vs the losers — is explicit.

## Context

Operator: "The purpose here is to figure out the best pipeline
definition to produce the best 3D scenes... The output of each
pipeline will be N images that will be compared." rate_renders +
rankings.jsonl + rank boards are the working precedent; renders
already live in-run with settings sidecars (STO-SCN-058). langfuse is
the candidate scoring layer per decision 5 — adopt only if the 069
spike picked it; otherwise built-in scores.

## Problem

rate_renders is standalone and view-centric; rankings name variants
but aren't attached to run records, and nothing aggregates "which
pipeline_instance wins across scenes/views" — the decision the whole
effort exists to make.

## Design

- Render step reuses render_comparison_matrix conventions
  (`run-<r>/renders/<view>.png` + settings sidecar) — unchanged.
- Ranking UI inside Studio (port/absorb rate_renders' compare flow);
  existing rankings.jsonl ingested via 071 so history counts.
- Scores attach to pipeline_run (F) and roll up to pipeline_instance
  (E) — the instance is what we're actually evaluating.
- Leaderboard: per scene + per view + aggregate; clicking two rows
  opens the 072 settings diff.
- Rankings are operator judgments — the system never auto-ranks
  (T-020; the operator's eye is the metric until a measured one is
  validated against it).

## Definition of Done

- [ ] Historical rankings (006/007/008 runoffs) appear as scores on
      their runs/instances.
- [ ] New flow end-to-end once: triggered run → renders → operator
      ranks in Studio → leaderboard updates.
- [ ] rate_renders retired ONLY after the operator confirms Studio
      covers the flow (until then both run; T-020).
