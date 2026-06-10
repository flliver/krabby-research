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
- [x] Runner executes to `status: success` with measured results.json
- [x] Outputs pushed to hub (`e40df74`, 2.2 GB LFS); fleet auto-sync propagation in progress (STO-SCN-030 timers)
- [ ] Operator inspects mesh (visual) — T-020; verdict recorded here
- [ ] Decision recorded: MPO normalize preproc needed? more photos needed? → follow-on stories if so

## Results (2026-06-09)

- **status: success** — 882 s, peak 11,994 MiB (vs ~7.6 GB for 004's 1024-px curated
  frames; the 5712-px MPO photos push VRAM near the 16 GB ceiling — a resolution-
  normalize preproc would buy headroom).
- **MPO inputs loaded cleanly**: "8 images found", zero errors/NaN in train.log —
  the MPO risk did not materialize at the SfM/ingest level.
- Mesh: `tetra_mesh_binary_search_7.ply` — **16.1 M verts / 32.2 M faces** (677 MB);
  SfM cloud 1.03 M points, cameras recovered in a coherent arc; cloud structure
  matches the capture (lawn plane + raised vegetation masses).
- Scene content: landscaped garden (lawn/shrubs/trees) — vegetation-heavy, the
  hardest class; 8 photos is sparse (HUG-SCN-004). Pipeline-pilot verdict: the
  config-driven path works end-to-end on a brand-new scene with zero manual steps.
- Inspection aids: `/tmp/kubota-sfm-views.png` (3-view colored cloud + cameras);
  mesh at `data/tetra_meshes/` for MeshLab/Blender A/B vs source photos in `input/src/`.
