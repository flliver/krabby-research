---
xid: STO-SCN-042
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-09
depends-on: []
bd-id: krabby-b79
title: 006-kubota pilot — first new scene via the runner
priority: 1
assignee: krabby
---

# 006-kubota pilot — first NEW scene reconstructed via the runner

## Summary

Reconstruct `006-kubota` (8 iPhone photos, the construction-flavored capture
class Fletcher asked for) entirely through the config-driven path: hand-author
the spec, execute via `run_transform.py`, inspect, iterate. The pilot proves the
runner on a scene with **no prior reconstruction** and surfaces the input-
normalization gaps new captures will hit.

## Context

Follows STO-SCN-039 validation (runner output statistically indistinguishable
from the manual procedure on 004-sky-house). Operator: "then we should try a
kubota scene." The 8 kubota scenes (005–012) are input-only store entries from
the STO-SCN-033 migration.

## Known risks (recorded at scaffold time)

- Inputs are **iPhone MPO** (multi-picture JPEG, 5712×4284) — most tools read
  the primary image; if SfM degrades, add `input/preproc-01-normalize`
  (MPO→plain JPEG) and re-run as a new run.
- **8 photos is sparse** vs the ~60%-overlap capture guidance (HUG-SCN-004);
  quality expectations calibrated accordingly — this is a pipeline pilot, not a
  quality benchmark.
- Scale: uncalibrated like every scene (STO-SCN-016 ★ still open).

## Steps taken

1. 2026-06-09: Scaffolded spec-first (store `cc77253`): `run.json` +
   `specification.json` (locked-default recipe: strong alignment, vitl, unposed,
   n_images 8; inputs `input/src`) + `scene.toml` `[[pipelines]]` matcha entry.
2. 2026-06-09: Executed on tbeeprz via `run_transform.py` (zero additional
   arguments — pure spec-driven). Results pending at authoring time; will be
   appended below with the measured results.json summary + inspection verdict.

## Definition of Done

- [x] Spec authored + committed BEFORE execution (config-driven, HUG-KRB-002)
- [ ] Runner executes to `status: success` with measured results.json
- [ ] Outputs pushed to hub; fleet auto-sync propagates (STO-SCN-030 timers)
- [ ] Operator inspects mesh (visual) — T-020; verdict recorded here
- [ ] Decision recorded: MPO normalize preproc needed? more photos needed? → follow-on stories if so
