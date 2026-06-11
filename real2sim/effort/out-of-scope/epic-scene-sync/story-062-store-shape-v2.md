---
xid: STO-SCN-062
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: []
bd-id: krabby-mb8
---

# Store shape v2 — track inputs+metadata+finals; transformation data untracked

## Summary

The scene store stops being "everything everywhere." Tracked set
(operator spec, 2026-06-10):

- **INPUTS** — input video OR input still-frames (`input/src/`,
  `input/*.mp4|mkv`, pool archives)
- **PREPROCESSOR / TRANSFORMATIONS** — metadata only:
  `specification.json`, `results.json` (with a `transient_data`
  location stanza), selection/camera JSONs. The bulky `data/`
  payloads are UNTRACKED.
- **OUTPUTS** — metadata + FINAL output(s): the deliverable mesh
  (`multires_tsdf_post_oriented.ply`), DA3 `gs_ply`, run `renders/`
  (+sidecars), scene `cameras.json`, `rankings.jsonl`, NOTES/FINDINGS.

## Context

2026-06-10: /games hit 100% (0 bytes free). Audit: 509G working tree
×(hub non-bare ×2)×(4 fleet full mirrors) ≈ 3 TB disk for ~250 GB
unique current data — and ~85 G of it format-duplicates and
intermediates. Operator: "We need to *not* store everything
everywhere."

## Design

- `.gitignore` (store root): transform/preproc `data/` payload
  patterns + `scene.blend`/`matrix_render.blend`, with explicit
  NEGATIONS for the tracked whitelist (all `*.json`, final mesh,
  `gs_ply/`, `renders/`).
- One untrack commit: `git rm -r --cached` the newly-ignored paths
  (files STAY on disk on the Mac — it is the designated transient
  archive, STO-SCN-063).
- Every affected `results.json` gains:
  `"transient_data": {"location": "mac:/var/krabby/scenes/<path>/data", "policy": "store-shape-v2 untracked"}`
- RECIPES.md gains a § Storage policy; tools stop writing the
  obviated duplicates (orient_mesh.py dual .obj write).

## Definition of Done

- [ ] .gitignore + untrack commit pushed; new clones materialize
      tracked-set only.
- [ ] Location stanzas in all affected results.json.
- [ ] RECIPES.md storage policy section.
- [ ] orient_mesh.py stops writing oriented_tetra.obj.
