---
kind: journal
name: M11 Scene Reconstruction
description: Robotics-research milestone 11 — going from raw video capture to a watertight, gravity-aligned, IsaacSim-ready scene mesh that a hexapod can walk through. Tracks pipeline experiments, post-processing tooling, capture-side learnings, and the open questions still gating the milestone.
started: 2026-04-29
archived: null
tags: [robotics, m11, scene-reconstruction, matcha, mast3r-sfm]
---

# M11 Scene Reconstruction journal

This journal carries the long-running effort behind milestone 11 — turning sparse-view captures into watertight collision meshes for IsaacSim.

It is **scoped to the milestone**, not to all of krabby. When M11 ships and follow-on milestones (M12+) start, they will get their own journals. The reason for milestone-scoping rather than a single workspace-wide journal: the M11 inquiry has its own multi-month arc, its own pipeline-evaluation matrix, and its own capture/post-processing tooling that won't all carry forward to whatever M12 turns out to be. Keeping journals milestone-scoped preserves the chronology of *this* effort cleanly.

## Threads at a glance

- **inbox** — capture-fast notes that haven't been routed to a real thread yet.
- **matcha-quality** — the open inquiry into how to get good meshes out of MAtCha on our captures. Resolution sweeps, frame curation, chart-encoding-resolution knob, etc.
- **post-processing** — Phase B work on the raw MAtCha tetra mesh: gravity alignment, ground-plane deduction, background culling, vertex-color projection, camera-marker placement.

New threads are cheap to add; create one when an inquiry doesn't fit either existing track.

## How this journal relates to the rest of the milestone tree

The journal is the **chronology of how we got here**. The other artifacts in the milestone are still authoritative for their own slices:

- `milestones/011-scene-reconstruction/PLAN.md` — what we intend to do (forward-looking).
- `experiments/<id>/README.md` — what one experiment was, in detail (per-run reference).
- `experiments/DECISION-MATRIX.md` — current pipeline beliefs (snapshot).
- `experiments/<id>/CAPTURE-LESSONS.md` — capture-side findings.

Journal entries should *reference* those artifacts rather than duplicating them. The journal's job is the narrative — what we tried, what we learned, what we changed our mind about, what's still open.
