---
xid: STO-SCN-127
parent: ./epic.md
kind: story
effort: scn
status: shipped
date: 2026-06-15
depends-on: []
bd-id: krabby-x2s3
assignee: krabby
tasks: 8
complete: 8
---

# reconstruct-da3: accept the solve/spine gauge directly — make matcha-reference optional

## Summary

A `v4exec reconstruct-da3-scout` command builds a **DA3 scene mesh in the solve/spine gauge
from an existing scout npz — matcha-free, no GPU** — and writes it as a content-addressed
`represent/da3` + `orient` + `meshify` node that `v4job render-missing` renders like any other
variant. The spine already poses the cameras, so DA3 needs no matcha reference for placement.

## Context

The α decision (2026-06-15): the EPI-SCN-AUTO-SUBSET-SELECT spine produces a full posed
sequence, so DA3 is posed from the solve — matcha is unnecessary for DA3 placement. But the
existing `reconstruct-da3` **hard-requires** a matcha reference (`matcha_reference()` gates the
command before inference, for the gravity gauge + an ICP reference mesh). That gate is the gap
this story closes. Discovered while building the DA3-24 variant for scene 001-patio
(EPI-SCN-SCENE-001-PATIO / STO-SCN-122).

## Problem

- `reconstruct-da3` exits with *"run reconstruct-matcha first"* if no matcha mesh exists on the
  subset — even though the spine already provides poses + a gravity prior. On a fresh
  FastMap-solved scene with no matcha, DA3 can't be reconstructed at all.
- The DA3 dense depths for the best-24 already exist (the n=24 voxel scout's `da3_poses.npz`,
  in the solve gauge) — there should be a matcha-free path that uses them, no GPU re-infer.

## Design

### Approach

New subcommand `reconstruct-da3-scout` (matcha-free; no `--host`/GPU):
1. **Fuse** the scout's `da3_poses.npz` (depth+conf+posed extrinsics, already in the solve
   gauge) → raw mesh via `da3_mesh_from_npz.fuse_npz` (RECIPES step 13 — no matcha alignment).
2. **Gravity orient** from the **solve cameras** (the npz's own extrinsics, inverted to c2w) via
   `bootstrap_orient` — the *same* orient matcha uses, but with no matcha mesh. Writes an
   `orient` node (`oriented.json`) under the solve.
3. **Ground** the mesh into the canonical gauge (`ground_mesh`, no weld-sim) → a `meshify/tsdf`
   node under a `represent/da3` rep; set the rep's `canonical_gauge`.
4. **Render-path fix:** emit the solve's `cameras.json` (`filepaths`+`cams2world`+`focals`) from
   `sparse/0` so `v4job.rep_camera_paths` can place the T2 cameras (see § Gotchas).

### Changes

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | new `cmd_da3_scout` + `reconstruct-da3-scout` parser; reuses `fuse_npz`, `bootstrap_orient`, `ground_mesh`, `posed_from_sparse` |
| `real2sim/knowledge/scene-processing/T3c-reconstruction-postprocessing.md` | "Render camera contract" subsection (the cameras.json/focals/oriented.json requirement + the FastMap-solve gap) |

## Definition of Done

- [x] `reconstruct-da3-scout` builds a DA3 mesh in the solve gauge with **no matcha reference**.
- [x] No GPU — reuses the existing scout `da3_poses.npz`.
- [x] The mesh is a content-addressed `represent/da3 … meshify/tsdf` node with a gravity gauge.
- [x] `v4job render-missing` discovers + renders it (the solve `cameras.json` emission closes
      the camera-resolution gap).
- [x] Operator-verifiable render produced (001-patio DA3-24, viewed 2026-06-15 — correct
      orientation, recognizable patio).
- [x] Operator sign-off (T-020) on the rendered variant in context — operator opened DA3-24
      in the Rank UI / Scout and confirmed it "looks good" (2026-06-15). Clean T-020.
- [x] T3c doc updated with the render-camera contract (the earned plumbing).

**Shipped 2026-06-15** (DES-SCN-DENSE-MESH closeout). Operator exercised the exact variant
(DA3-24, 001-patio) and signed off; the matcha-free `reconstruct-da3-scout` path is the
documented, repeatable α command.

## Testing

### Integration
- [x] 001-patio: `reconstruct-da3-scout --solve 62QEHJDAJZBI --scout W75HYBNU37WK` →
      `represent/da3/42UOWRFIK6SB`, mesh `BO6XP5CFKDA5` (2.67M verts), orient `EP6MPQFJCAG4`
      (horizon residual 4.7°). `render-missing` → rendered 1; render visually correct.

## Out of scope

- A fresh higher-res DA3 infer (the existing scout npz is res 504 — good enough for the runoff;
  a `--host` higher-res path can be a follow-on).
- Wiring this into `reconstruct-da3` proper as a `--gauge solve` flag (the dedicated
  `reconstruct-da3-scout` command is the minimal matcha-free path; folding it into the main
  command — with fresh infer — is deferred).

## Implementation Notes

**Built + validated end-to-end 2026-06-15.** The α path is now a documented, repeatable command.

### What Changed
`cmd_da3_scout` in `v4exec.py`: fuse scout npz → bootstrap_orient (solve cameras, matcha-free)
→ ground → `represent/da3/<rid>` + `orient/<oid>` + `meshify/tsdf/<tid>` (algo `da3-mesh@0`),
with `canonical_gauge` set. Emits the solve `cameras.json` from `sparse/0`.

### Gotchas (earned — see T3c "Render camera contract")
1. **The matcha gate fires for posed AND unposed** `reconstruct-da3` — it's before the sfm
   branch, so it's about gauge + ICP reference, not poses. The poses come from the spine.
2. **FastMap solves emit only `sparse/0`, no `cameras.json`** — so `render-missing` skipped the
   DA3 mesh *silently* until the command emitted one from `sparse/0`.
3. **`cameras.json` must include `focals`** — `build_blender_scene` raises `KeyError: 'focals'`
   without it (focal = `K[0][0]` from `posed_from_sparse`).
4. The scout retains the depths as **`da3_poses.npz`** (not `results.npz`) — it *does* carry
   `depth/conf/extrinsics/intrinsics/image`, so it's a usable fusion input.

### Files Modified
- `real2sim/v4exec.py` — `cmd_da3_scout` + parser.
- `real2sim/knowledge/scene-processing/T3c-reconstruction-postprocessing.md` — render-camera contract.
