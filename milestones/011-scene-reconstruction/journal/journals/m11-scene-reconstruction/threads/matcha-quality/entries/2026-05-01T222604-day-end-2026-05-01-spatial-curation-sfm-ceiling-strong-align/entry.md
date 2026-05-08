---
kind: entry
date: 2026-05-01T22:26:04-07:00
title: Day-end 2026-05-01: spatial curation, SfM ceiling, strong-alignment win
mood: null
consolidates_notes:
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T203949-accomplishments-and-next-steps
  - journals/m11-scene-reconstruction/threads/matcha-quality/notes/2026-05-01T222604-strong-alignment-config-eliminates-garbage-geometry
tags: []
---
# Day-end 2026-05-01: spatial curation, SfM ceiling, strong-alignment win

The first concrete results day for M11 after the journal infrastructure
landed in the morning. Three tracks ran in parallel — SfM scaling
characterization, the camera-selection viewer build, and the first
curated-mesh experiment — and a late-evening alignment-config test
delivered the day's clearest quality win. This entry consolidates the
end-of-session retro and that late finding.

## What shipped on 2026-05-01

### Journal infrastructure (morning)

Manual filesystem journal at
`milestones/011-scene-reconstruction/journal/` using OLAI's locked
4-resource layout (journal/thread/entry/note). Three threads bootstrapped
(`inbox`, `matcha-quality`, `post-processing`), four entries, multiple
notes covering the Phase A retrospective and Option C planning. Slugs
migrated from date-precision to timestamp-precision after the OLAI D5
amendment. `jlib.py audit` clean. Pulled in research-side lessons:
`cu128` wheel index for RTX 5080; `--shm-size=8g` mandatory for PyTorch
containers (both verified on `matcha-build`).

### MAtCha source code-read (afternoon)

Settled three open questions; full detail consolidated separately under
*Scaling M11 reconstruction beyond single-MAtCha runs*. Headlines:

- `r` is a 4-level pyramid (`[0.05, 0.1, 0.2, 0.4]`), not a single value.
  Option C tests "truncate the list" rather than "tune one number."
- Photometric supervision uses input-resolution images (up to 1600
  long-edge); chart geometry pinned at 512.
- `train.py --sfm_only` and `--image_idx` already exist — no B5 wrapper
  needed.

### SfM-scaling experiment (afternoon)

Bracketed the MASt3R-SfM ceiling on RTX 5080 / 16 GB. **N≤300
comfortable, N=350 borderline, N≥500 OOMs.** Operational discovery:
foreign processes contaminate VRAM measurements;
`nvidia-smi --query-compute-apps` is the diagnostic. Watchdog pattern
saved ~20 min of futile compute by killing the chain after N=300
succeeded. Detailed analysis in the same scaling-architecture entry.

### Camera Selection Viewer (afternoon → evening)

Built the Route B viser-based 3D viewer end-to-end (~900 lines across
`data.py`, `filters.py`, `ui.py`, `viewer.py`, `slots.py`,
`clustering.py`). Seven filters (time range, temporal stride, spatial
cluster, distance, look-at gizmo, pHash dedupe, picked-status), bulk
visible→pick shortcuts, named slot save/load, live-updated counters.
Two real bugs caught during the build (forward-axes sign, PIL
deprecation churn). Full design rationale in the scaling-architecture
entry.

### First curated MAtCha run (evening)

Picked 12 frames from the n350 viewer scene using the seven-axis filter
UI. Ran full pipeline on tbeeprz: **648 sec wall-clock, 7.7 GB peak
VRAM** (well under the 14.8 GB free baseline). Output: watertight tetra
mesh + 2DGS gaussians + per-frame SfM at
`scenes/004-sky-house-curated-12/`. **First B5-validated mesh the
pipeline has produced** — frames spatially curated rather than evenly-
time-spaced.

### Post-processing (in-flight at session end)

B1 orient → decimate → B4 project_color → B2 cull running on tbeeprz
against the curated mesh. Background task `bfxefknnh`. Final B3 step
(Blender headless) deferred to next-session pickup on JDP-Mac.

## The strong-alignment win (late evening, separate experiment)

A Tier 1.a follow-up after the curated-default mesh landed: rerun the
same N=12 spatially-curated set with `--alignment_config strong` instead
of `default`. Same hardware, same source frames, same downstream B1-B4.

**Compute cost: zero.** MAtCha wall-clock identical (648 sec), peak VRAM
identical to within 10 MiB. Final culled mesh has ~3% more vertices /
~5% more polygons — within noise.

**Quality verdict** (per visual inspection in Blender): "roughly equal"
on first glance, but **strong eliminated garbage mesh in obvious areas.**
Specific failure mode improved: hallucinated/floating geometry that the
default-alignment run produced in regions where SfM had noisy points.
The strong config's chart-encoding norm penalty + total-variation on
depth encodings + confidence-weighting suppresses chart deformation from
over-fitting to noisy SfM.

This is consistent with the Option C hypothesis (over-fitting to noisy
SfM) — strong-alignment is essentially **"Option C lite"**: it doesn't
change the multi-resolution `r` pyramid, but it adds penalty terms that
suppress the same failure mode.

**Decision: lock `--alignment_config strong` as the operating default
for all future MAtCha runs.** Zero downside, meaningful upside. Future
experiments (N=16 bracket, scenes 001/003 recapture, Phase C) all
proceed on strong as the new baseline. Option C is now lower-priority —
strong already buys most of what we expected from `r` truncation. If we
hit a quality wall later, Option C remains the sharper architectural
test (changes the deformation field's representation, not just the
regularizer).

Files for side-by-side compare:

- Default: `data/scenes/004-sky-house-curated-12/oriented/scene_culled.blend`
- Strong:  `data/scenes/004-sky-house-curated-12-strong/oriented/scene_culled.blend`

## What this set up for the next sessions

Day-1 next-step plan (from the retro) with retrospective annotations:

1. **Finish the curated post-processing + validate B5** (next morning)
   → Done. Curated mesh produced, comparable to Phase A baseline,
   strong-alignment win banked.
2. **Iterate on curation if needed (or roll out if not)** → Pivoted
   instead toward Phase C (reference validation) when the bicycle/MAtCha
   reference images surfaced as a tighter quality benchmark than internal
   side-by-side comparison.
3. **Phase C: USD export + IsaacSim load** → Plan revised at the 5/04
   pivot to insert "validate against MAtCha paper's reference quality"
   as a Phase C prerequisite *before* USD export. See sister entry
   *Reference-camera auto-positioning: data-model and process*.
4. **Phase D: hexapod parkour validation** → Still ahead, gated on
   Phase C and Phase D mesh conditioning.

Open research tracks deferred (per retro):

- **Option C — `r` truncation.** Lowered priority by strong-alignment.
  Still the principled architectural test if/when strong isn't enough.
- **MAtCha N ceiling at 1024×576.** N=14, 16, 18, 20 picks unmeasured;
  could lift the per-mesh budget from 12 to 16-20.

## Lesson worth keeping

Late-evening single-knob experiments can deliver more than the full-day
build. The strong-alignment test was 30 minutes of compute and 10
minutes of side-by-side inspection. It produced the day's clearest
quality finding and changed the operating default for every subsequent
run. **Cheap experiments after the day's main work has shipped are
disproportionately valuable.**
