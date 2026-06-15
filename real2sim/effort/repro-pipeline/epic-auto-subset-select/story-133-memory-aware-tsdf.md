---
xid: STO-SCN-133
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-15
depends-on: []
bd-id: krabby-i7qz
assignee: scout
---

# Memory-aware TSDF: mesh_res tunable + RAM pre-flight + local-fuse (B) strategy

## Summary

Make TSDF mesh extraction **fit the host it runs on** instead of OOM-roulette: expose
`mesh_res` as a tunable, **pre-flight the RAM need** (coarse probe + square-law) to auto-pick a
`mesh_res` that fits, and add a **local-fuse strategy (B)** — render gaussian depths on the GPU
host (cheap RAM) and run the RAM-heavy Open3D fusion on the **128 GB local box** at full
`mesh_res 1024`.

## Context / Problem

`extract_tsdf_mesh.py` reads `mesh_res` from a baked YAML config (`default` = **1024**); it's
not operator-tunable per run, and there's no memory check. On the 31 GB tbeeprz host, the
TSDF extract is **CPU + system-RAM bound** (Open3D `ScalableTSDFVolume` + marching cubes — the
gaussian *render* before it is GPU/CUDA). Measured 2026-06-15:

| mesh_res | peak RAM | peak swap | outcome |
|---|---|---|---|
| 1024 | ~31 GB | ~46 GB | **OOM-killed** (~77 GB needed) |
| 512 | ~30 GB | 9.5 GB | ✅ 91 s, 10.3 M verts |

The blowup is `mesh_res`-vs-**scale**: the spine/solve gauge gives a small bounding radius
(0.31) → at `mesh_res 1024` the voxel grid is absurdly fine → memory explodes.

## Design

### 1. `mesh_res` tunable (unblocks matcha-15 today)
Expose `mesh_res` through `reconstruct-matcha` (and the `meshify-via-tsdf` task — it's already
declared `tunable`, "1024 validated; others unmeasured"). Since the config is a baked YAML,
either mount a generated config or call `render_multires.py --mesh_res <N>` directly (it
produces the same `multires_tsdf_post.ply`). Record `mesh_res` in identity so 512/1024 are
distinct content-addressed nodes.

### 2. RAM pre-flight (avoid OOM by construction)
A coarse probe (e.g. `mesh_res 128`, seconds) measures occupied voxel blocks / vert count;
extrapolate peak RAM by the **square law** (`∝ (mesh_res/extent)²`, `∝ 1/factor²`); pick the
largest `mesh_res` whose predicted peak ≤ (host RAM + safe swap margin). Ties to the peak-RAM
tracking (STO-SCN-132) — the probe + the law are validated against measured peaks (1024≈77 GB,
512≈40 GB).

### 3. Strategy B — local-fuse on 128 GB (keeps full res)
The RAM-bound phase is host-agnostic Open3D CPU work; the **local box has 128 GB unified RAM**
(≫ the ~77 GB that 1024 needs). So: **GPU-render depths on tbeeprz → Open3D-fuse + extract
locally at mesh_res 1024**. Precedent: `da3_mesh_from_npz.py` already fuses DA3 depths locally
on this Mac.

**Prerequisite for B (matcha):** matcha depths are **NOT broken out** — `render_multires.py`
renders gaussian depths *inline* and fuses them in-process (only `multires_tsdf.ply` is saved,
no depth artifact). So B-for-matcha needs a **depth-export step**: split `render_multires` into
(GPU) render-depths→`depths.npz` + (local) Open3D fuse — a matcha analog of DA3's
`da3_poses.npz` → `da3_mesh_from_npz.py`. **B-for-DA3 is already done** (the npz path).

| File | Change |
|------|--------|
| `real2sim/v4exec.py` | `reconstruct-matcha --mesh-res`; pre-flight probe→auto mesh_res; optional `--fuse local` |
| matcha image / a new tool | break depth-render out of fusion → export `depths.npz`; local Open3D fuse (matcha analog of `da3_mesh_from_npz`) |
| `real2sim/tasks/meshify-via-tsdf.json` | `mesh_res` honored per-run; record in identity |

## Definition of Done

- [ ] `reconstruct-matcha` accepts `mesh_res`; 512 materializes matcha-15 as a store node + renders.
- [ ] Pre-flight predicts peak RAM within ~1 GB (validated vs the measured 1024/512 peaks) and
      auto-selects a fitting `mesh_res` (or warns + offers local-fuse).
- [ ] Strategy B available: depths exported from the GPU render; local Open3D fuse at 1024 on
      the 128 GB box produces a mesh without OOM.
- [ ] Operator can choose: A (fit-on-host mesh_res) or B (full-res local-fuse).

## Out of scope

- Peak-RAM tracking / failed-result records (STO-SCN-132 — consumed here).
- The select→matcha posed wiring (STO-SCN-130, done).

## Implementation Notes

_(Earned 2026-06-15. `mesh_res 512` validated on tbeeprz (91 s, fits); 1024 OOMs (~77 GB).
B reframes the wall: put the RAM-bound fusion where the RAM is — 128 GB local. Matcha depths
must be broken out first; DA3's npz path is the template.)_
