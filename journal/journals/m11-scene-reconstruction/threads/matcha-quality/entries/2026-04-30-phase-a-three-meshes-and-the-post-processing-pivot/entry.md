---
kind: entry
date: 2026-04-30
title: Phase A — three MAtCha meshes and the pivot to post-processing
mood: resolved
consolidates_notes: []
tags: [phase-a, matcha, retrospective, pivot]
---

# Phase A — three MAtCha meshes and the pivot to post-processing

## What happened

Over the back half of April, we evaluated six candidate pipelines for the M11 deliverable (COLMAP, MASt3R-SLAM, SLAM3R, VGGT, MAtCha, AnyRecon) and ended Phase A with three watertight MAtCha meshes — one per scene we'd captured (001 patio, 003 firepit, 004 sky-house-dining). MAtCha was the only pipeline that produced a watertight mesh end-to-end without a separate conditioning step, in ~11 minutes wall-clock on bbeeprz (RTX 5080), at 12 keyframes per scene.

The full per-pipeline accounting and rubric live in `experiments/DECISION-MATRIX.md`. The per-scene specifics live in:

- `experiments/001-matcha-patio-fisheye/README.md`
- `experiments/003-matcha-firepit/README.md`
- `experiments/004-matcha-sky-house/README.md`

This entry is about the *thinking* — what the three runs taught us collectively, and why we pivoted from "run more pipelines / capture more scenes" to "fix post-processing first."

## Quality verdict on the three scenes

Per Jeremy's 2026-04-30 inspection:

| Scene | Verdict |
|-------|---------|
| 001 patio (4K hyperlapse, 31 s, 155° fisheye) | "Chaotic, but obviously the filmed scene. Includes too much background noise (far things) that would ideally be culled." |
| 003 firepit (4K @ 60fps regular video, 5:31, fisheye) | "Chaotic, but obviously the filmed scene. Also includes too much background noise." |
| 004 sky-house (2.7K @ 30fps, 3:47, semi-indoor) | "Dense in many areas, but obvious gaps in places — probably not covered." |

The character of the chaos was **consistent across capture profiles**: foreground recognizable, distant background polluting. That consistency was the most informative finding of Phase A — it meant the problem was structural, not capture-specific.

## The five cross-cutting issues

Every Phase A mesh had the same five problems, regardless of which scene:

1. No clear ground plane.
2. Output mesh always tilted (no consistent up direction).
3. Background noise pollution (especially the 155° fisheye outdoor scenes).
4. No camera locations visible in the mesh — you couldn't tell where the photographer stood.
5. No vertex color from the source frames — the watertight mesh was geometry-only.

Issues 1, 2, 4, 5 are not MAtCha's job. MAtCha satisfies its T1 acceptance (watertight mesh) on every scene. Issue 3 (background pollution) is partly MAtCha's reach — its output naturally extends to the depth of the SfM points — but mostly a downstream cull problem.

## The pivot

The temptation was to keep grinding pipelines or recapture with better protocols. The actual right move was: **build a post-processing pipeline that addresses 1–5, then re-evaluate**. If the post-processed output is good enough for IsaacSim, MAtCha is the answer. If not, *then* we know whether the gap is MAtCha (more captures, more pipelines) or our tooling.

That pivot became Phase B (B1 orient, B2 cull, B3 cameras, B4 color), and is captured in the `post-processing` thread of this journal. Phase B was essentially complete by the next session.

## What I'd write differently a month from now

I'd be more skeptical earlier of the assumption that Phase A's mesh quality was a *capture* problem. Once we had two scenes (001 and 003) showing the same character under wildly different capture profiles, that should have been enough to flip the diagnosis from "improve capture" to "improve post-processing." We did get there, but it took the third scene (004) to commit.

T-001: when you have a hypothesis ("we need better captures"), look for the disconfirming case. Two scenes with the same flaw under different capture profiles is the disconfirming case.
