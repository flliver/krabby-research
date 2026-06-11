---
xid: STO-SCN-062
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-mb8
shipped: 2026-06-11
tasks: 4
complete: 4
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

- [x] .gitignore + untrack commit pushed (106651b; verified over all
      11,506 tracked files, zero misclassifications). Pointmaps leak
      (bulk data serialized as JSON, 11.2 GiB) caught + fixed (08eb8a5).
- [x] Location stanzas in all 58 affected results.json.
- [x] RECIPES.md § Storage policy (incl. tooling-provenance rule).
- [x] orient_mesh.py stops writing oriented_tetra.obj (49 GB class).

## Status notes

- 2026-06-11: SHIPPED. Tracked set 6.1k files / ~55 GiB (was 11.5k /
  240 GiB). Validated end-to-end the same day: d regenerated 006's DA3
  transients from tracked inputs+metadata in 12 s and reproduced the
  render bit-comparably — the v2 premise (inputs+metadata+recipes =
  artifacts) demonstrated in production.
