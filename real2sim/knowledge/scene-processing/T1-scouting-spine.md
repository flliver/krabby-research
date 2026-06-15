# T1 — Scouting & Spine

> Phase 2 of [the M11 process](README.md). Pose the image pool into a **spine** (the full
> camera trajectory in one gauge), then produce a **scout splat** the operator verifies.
> The spine + a verified scout is the foundation every reconstruction stands on.

## Inputs → Outputs

| In | Out |
|---|---|
| the T0 image pool + `capture.json` | a posed **solve** (`sparse/0`), a **covis** graph (validity-gated), [a registered+fused **spine** for long pools], a **scout** gaussian in the solve gauge + `scout_gauge.json`, and an operator **verify** verdict |

## The steps (all `v4exec`)

| # | Step | Command | Produces |
|---|---|---|---|
| 1 | **Pre-cull** | `v4exec precull <scene> [--set-primary]` | pose-free **sharpness + pHash-dedup** → curated subset ≤ solve ceiling. **Preserves revisits** (loop-closure gold) and **orders by capture time** (STO-SCN-092/096). |
| 2 | **Solve** | `v4exec solve <scene> --host tbeeprz [--subset <id>]` | GPU **FastMap** SfM (fisheye→pinhole first via per-camera calib, STO-SCN-102) → `sparse/0` poses = **the spine** |
| 3 | **Covis** | `v4exec covis <scene> --host tbeeprz --solve <id>` | co-visibility graph + **validity gate** — HARD-FAILs a "nebula"/drift solve so a bad solve never reaches selection (STO-SCN-093) |
| 4◇ | **Spine segment** | `v4exec spine <scene> [--cap 300 --overlap 30]` | *long pools only:* chunk the trajectory into **M overlapping segments** + loop candidates (`spine.json`) |
| 5◇ | **Spine register** | `v4exec spine-register <scene> --spine <id> --solves seg=<sub>/<solve>,…` | SIM(3) **pose graph** over segments → one global gauge (drift-corrected) + per-seam residuals (`global.json`) |
| 6◇ | **Spine fuse** | `v4exec spine-fuse <scene> --spine <id> --register <id> --solves … --gaussians seg=<ply>,…` | confidence-weighted fusion of per-segment gaussians (overlap cross-fade, no doubled walls) → one cohesive `.ply` |
| 7 | **Scout** | `v4exec scout <scene> --host tbeeprz --solve <id> [--selector voxel\|track --n-scout N]` | DA3 `da3@1` **scout gaussian in the solve gauge** + `scout_gauge.json` (gs→solve registration, the STO-SCN-105 fix). `--n-scout` default 32 (~DA3 ceiling). |
| 8★ | **Verify** | `verify_viewer/build_verify.py <scene> --solve <id> --scout <id> [--selector voxel --cull-expand E]` → `viewer.html` | splat + proposed-N frustums + voxel-coverage faces (red→green) + gravity-aligned cull box + WASD-fly + optional DA3-mesh layer. **Operator confirms / overrides (T-020).** |

◇ = spine steps; **no-op at M=1** (a short scene *is* one segment — the segment is the whole
scene). ★ = the human-in-the-loop gate.

## The spine concept (why this phase exists)

EPI-SCN-AUTO-SUBSET-SELECT's deliverable is a **FULL posed sequence of all cameras** — the
spine. A single long video is too big to pose/reconstruct at once, so it splits into M
overlapping segments, each locally solved, then **globally registered** (pose graph + loop
closure) into one gauge and fused. At M=1 it degenerates to a single solve. Crucially, the
spine gives downstream phases **posed camera frustums for free** — DA3 reconstruction (T3) can
be posed straight from the spine with no matcha needed for poses.

## The scout / verify surface

The scout is a fast **DA3 gaussian splat** placed in the solve gauge — it is the surface where
a human *sees* the proposed cameras + coverage gaps and accepts/overrides. The splat is the
**verification surface, not the selector** (selection is automated from covis, T3a). Viewer
features: coverage faces, gravity cull box, WASD fly. Served locally (reach it via
`krabby.organl.com`, not `localhost`).

## Gotchas

- **Solver must match modality** — sparse-view methods (MASt3R/DUSt3R) drift on dense video
  (no temporal prior → the "nebula"). FastMap (GPU) is the proven solver here.
- The covis gate is **load-bearing** — never bypass it; a drift solve scores a high
  out-of-plane/in-plane ratio and is rejected before any GPU reconstruct is wasted.
- Scouts retain `scout.gs.ply` but **not** the depth `results.npz` — see T3a if you need a DA3
  mesh from a scout's depths.

## Automation status

Steps 1–7 are single `v4exec` commands; step 8 is the one operator gate. ✅ automated up to verify.

## Next

→ [T2 — View Selection](T2-view-selection.md)
