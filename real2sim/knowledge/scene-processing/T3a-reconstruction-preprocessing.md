# T3a — Reconstruction Pre-Processing

> Phase 4 of [the M11 process](README.md). Before any GPU reconstruction: **choose the model**,
> **choose its settings**, and produce the **best-N "best cameras" subset** the model will
> consume. This is where the matcha-vs-DA3 and how-many-frames decisions are made and recorded.

## Inputs → Outputs

| In | Out |
|---|---|
| the spine solve (`<solve>`) + covis (`<covis>`) | a content-addressed **FINAL-N subset** (the reconstruction handoff) + a recorded model + settings choice |

## 1. Choose the model

| Model | Task / algo | Use it for | License |
|---|---|---|---|
| **MAtCha** | `represent-via-matcha` (`matcha@1` posed / `@0`) | **primary** — watertight TSDF + tetra meshes from sparse views | OK |
| **DA3** | `represent-via-da3` (`da3@1` posed / `@0`) | denser scene from more frames; the **spine-gauge npz mesh** path | **CC BY-NC — evaluation only** |
| COLMAP / VGGT / MASt3R | `represent/<model>` | alternative SfM/feed-forward fronts (research) | varies |

- **MAtCha** is the deliverable-grade mesher but is a **sparse-view** method (target band
  **24–30 frames**) and has a TSDF OOM cliff (below).
- **DA3 can consume more frames** → more scene, but it's the **evaluation branch** (NC) — never
  a shippable deliverable, only a comparison/quality probe.

## 2. Choose the settings

**MAtCha** (`represent-via-matcha.json`):

| Setting | Class | Default | Notes |
|---|---|---|---|
| `dense_regul` | tunable | `default` (or `dense`) | densification regularization |
| `n_iters` | frozen | 7000 | |
| `encoder` | frozen | `vitl` | |
| `alignment_config` | frozen | `strong` | |
| `--sfm` | — | `posed` (`matcha@1`) | feed the spine solve as `sparse/0` (no re-solve, no arbitrary gauge) — the validated path |

**DA3** (`represent-via-da3.json`): `--sfm posed` = `da3@1` (spine extrinsics/intrinsics fed
to inference); `--sfm unposed` = `da3@0` (DA3 estimates its own cameras; gauge enters at fuse).

### The hard constraint you must respect (matcha N vs VRAM)

| Limit | Value | Source |
|---|---|---|
| **TSDF multires-merge OOM** | **≥17 cameras × `mesh_res` 1024 OOMs a 16 GB GPU** — **fixed in matcha image ≥0.2.2** (`multires_oom`, STO-SCN-053/056) | story-056 |
| Required env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation OOM otherwise) | story-056 |
| MAtCha target band | **24–30 frames** (sparse-view) | `images/matcha/NOTES.md` |
| Normalize preproc | long-edge 2048 → **−28 % training VRAM** (headroom for larger N) | STO-SCN-043 |

**Rule of thumb:** on matcha image `<0.2.2`, keep matcha **N ≤ 15** (sub-cliff). On `≥0.2.2`
+ normalized inputs, matcha can run its full 24–30 band. DA3 is not bound by this cliff.

## 3. Auto-select the best-N cameras (`select`)

```bash
v4exec select <scene> --solve <solve> --covis <covis> [--selector voxel --n 24]
```

| Setting | Default | Notes |
|---|---|---|
| `selector` | **`voxel`** (STO-SCN-103) | voxelize the scene, reward each camera for exposed voxel-faces it sees (incidence-angle flux), **greedy-add largest marginal coverage** — rewards angular variety; beats `track` on the real 001 pool |
| | `track` (STO-SCN-094) | greedy submodular over shared SfM track points (connectivity + triangulation angle) |
| `n` | 24 | target view count (downstream sweet spot) |
| `grid` | 64 | voxel grid resolution (voxel selector) |
| `min_overlap` / `div_angle` | 10 / 25° | connectivity / viewpoint-diversity (track selector) |

Emits `selection.json` (coverage report the verify surface renders) + `posed.json` + the
**content-addressed FINAL-N subset** (member hashes). Greedy is **prefix-stable**: `--n 15`
yields the first 15 of the `--n 24` order, so N-subsets nest.

## Gotchas

- **`reconstruct` reads `primary`, not `--subset`** — selecting the FINAL-N subset is only half
  the handoff; you re-point `primary` at it in [T3b](T3b-reconstruction-processing.md).
- **DA3 mesh without matcha:** `reconstruct-da3` currently hard-requires a matcha reference
  (gauge + ICP). For a spine-native, matcha-free DA3 mesh, fuse a scout/da3 **posed `npz`** with
  `da3_mesh_from_npz.py` (solve gauge). Scouts don't retain the npz → run a DA3 infer that does.
  Tracked: **STO-SCN-127**.

## Automation status

`select` is one command; model/settings are recorded choices. ✅ automated (the choice is yours,
the execution is deterministic + content-addressed).

## Next

→ [T3b — Reconstruction Processing](T3b-reconstruction-processing.md)
