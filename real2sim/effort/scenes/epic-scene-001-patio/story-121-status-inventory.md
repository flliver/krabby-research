---
xid: STO-SCN-121
parent: ./epic.md
kind: story
effort: scn
status: in-progress
date: 2026-06-15
depends-on: []
bd-id: krabby-2gmy
assignee: krabby
---

# Scene 001-patio status inventory — what's done / what's not

## Summary

The authoritative, store-grounded snapshot of **scene `001-patio`** ("the Patio at the
Packwood Cabin") — every artifact that exists today and the gaps that remain — so the
remaining-work stories (STO-SCN-122…126) start from fact, not memory. Verified against
`/var/krabby/scenes/001-patio/` on 2026-06-15.

## Scene identity

- **Store id:** `001-patio` (this is the canonical id; do not rename).
- **Subject:** the patio at the Packwood Cabin.
- **Capture/processing:** posed via FastMap; the active solve is the M11 "spine" for this scene.

## What's DONE (verified in the store)

| Stage | Artifact | Id / detail |
|---|---|---|
| T1 solve | FastMap solve (the spine) | **`62QEHJDAJZBI`** (subset `6EHLYO3MF3QU`) |
| T1 covis | co-visibility graph + validity gate | **`L57FPDHY2DRG`** |
| T3a select | voxel best-N (n=24) → FINAL-24 subset | select **`72CTZUDLZB3M`** → subset **`7MLHQCKN5XYY`** |
| T1 scout | DA3 `scout@0` splats (verify surface, solve gauge) | 4 nodes — `W75HYBNU37WK` (24v), `3R7ZB5GAB6PC` (29v), `OZGYMJTRXN3Z` (30v), `VZXYUNOO7DPG` (32v); each has `scout.gs.ply`, **no `results.npz` retained** |
| T3 reconstruct (historical) | matcha reconstructions — **12-frame** sets | `LQLIS7O67GHX`, `EV3YPPJL7SWV` (subset `VPBP7W4PYCCJ`), `IYI4BXFH327F`, `U6TN5SNVGQJA` (subset `3A6MH6U5VKYP`), `KRHALSWG3HV4` (legacy) |
| T3c renders / T4 | comparison renders + scores | `scores.jsonl` present; historical variants ranked |

## What's NOT done (the gaps → the remaining stories)

1. **No reconstruction on the FINAL-24 spine subset** (`7MLHQCKN5XYY`). All matcha reconstructions
   are the old 12-frame sets; none use the 24-view selection. → **STO-SCN-122 / 123**.
2. **No DA3 *scene* (mesh) at N=24** — only DA3 *scouts* (gaussians) exist, and they did not
   retain the depth `npz`, so `da3_mesh_from_npz.py` has no input yet. → **STO-SCN-122**
   (matcha-free spine-gauge path, host `tbeeprz`).
3. **No matcha-15** — the deliberately-sub-OOM (≤17-cam) matcha reconstruction on the
   FINAL-15 subset (which doesn't exist yet — needs `select --n 15`). → **STO-SCN-123**.
4. **`PWZ4S24AZ72T` is invalid** — a TSDF mesh of the `12sharp-strong` matcha
   (`represent/matcha/LQLIS7O67GHX/meshify/tsdf/PWZ4S24AZ72T`), referenced once in
   `scores.jsonl`; the operator marked it invalid and it must be excluded from ranking
   (no built-in discard mechanism exists today). → **STO-SCN-124**.
5. **No comparison renders** for the new DA3-24 / matcha-15 meshes. → **STO-SCN-125**.
6. **The runoff comparison** (DA3-24 vs matcha-15 vs the surviving historical variants) has
   not been ranked. → **STO-SCN-126**.

## Decisions locked (this session, 2026-06-15)

- **DA3-24 path = α (matcha-free, spine-gauge).** DA3 is posed from the spine solve
  `62QEHJDAJZBI`; the scene mesh comes from `da3_mesh_from_npz.py` in the solve gauge — **no
  matcha reference**. (The `reconstruct-da3` command currently hard-requires a matcha
  reference for gauge + ICP; that gap is **STO-SCN-127**.)
- **matcha-15** stays at N=15 to sit safely under the ≥17-camera TSDF OOM cliff (unless on
  matcha image ≥0.2.2).
- **Host = `tbeeprz`** for all GPU work.

## Definition of Done

- [x] Every existing artifact and gap recorded against the live store (above).
- [ ] Remaining-work stories (122–126) reference this inventory as their starting state.
- [ ] Inventory re-verified true at the time the scene runoff (126) closes.

## Out of scope

- Doing the reconstructions/ranking (those are 122–126).
- The process documentation (that's `EPI-SCN-M11-PROCESS-DOCS`).
