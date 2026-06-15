# T3c — Reconstruction Post-Processing

> Phase 6 of [the M11 process](README.md). Turn a reconstructed **mesh** into the **N
> renderings** (one per T2 view) that the runoff ranks. This is what produces the images the
> operator actually compares in T4.

## Inputs → Outputs

| In | Out |
|---|---|
| a reconstruction mesh (T3b) + the T2 virtual views | `…/renders/<view>.png` per (mesh × view) + a settings sidecar; surfaced as a **variant** in the rank UI |

## What gets rendered

The comparison is **Cartesian: variant × view**. Each reconstruction (a "variant") is rendered
from every T2 camera, so the operator compares the *same* viewpoints across *different*
pipelines. The render is an artifact **of the run's configuration** — that configuration is the
thing being judged.

## Step — render the missing views (graph-native)

```bash
python3 real2sim/v4job.py render-missing <scene>|all [--dry-run]
```

- Walks the **expected set** = (meshes × canonical T2 views) and renders any that don't exist
  yet — content-addressed, so already-rendered (variant, view) pairs are NOOPs.
- Engine **`BLENDER_WORKBENCH`**, **1920×1080** (fast, consistent shading for comparison).
- Writes `…/renders/<view>.png` next to the mesh node, with a per-view settings sidecar
  (engine/resolution/mesh-source + the run's transform parameters). Emits a `jobs/` record +
  MQTT progress (`render-missing … done 100`).
- `--dry-run` lists what *would* render (use it to sanity-check the expected set first).

## Render camera contract — why a variant renders (or silently doesn't)

`render-missing` doesn't render a mesh in isolation — it frames it with the T2 cameras via
`build_blender_scene.py`. For that to work, the renderer must resolve **two files per
representation** (via `v4job.rep_camera_paths`):

| File | What it must contain | Where it comes from |
|---|---|---|
| the rep's **`cameras.json`** | `{"filepaths": [...], "cams2world": [4×4, …], "focals": [fx, …]}` — the solve's posed cameras + per-camera focal (pixels) | the solve |
| the rep's **`oriented.json`** | `{"rotation": 3×3, "z_shift": float}` — the gravity gauge (`canonical_gauge` in the rep metadata) | the orient node |

If `rep_camera_paths` can't find a `cameras.json`, **`render-missing` skips the mesh
silently** (it counts NOOPs/renders for the meshes it *can* place; the unresolvable one just
never appears). That's the trap.

### The FastMap / spine-gauge gap (earned 2026-06-15, STO-SCN-127)

Migrated (mast3r-era) reps carry their own cameras at
`represent/<…>/origin-data/mast3r_sfm/cameras.json`, so they render. **FastMap solves emit
only `sparse/0` (COLMAP bins) — no `cameras.json`** — so a spine-gauge variant (e.g. the DA3-24
from `reconstruct-da3-scout`) renders **0** until a `cameras.json` is generated from `sparse/0`:

```python
# what reconstruct-da3-scout now does once per solve (posed_from_sparse → cameras.json):
fps, c2ws, focals = [], [], []
for e in posed_from_sparse(f"{solve}/sparse/0"):      # [{name, w2c(4x4), K(3x3)}]
    w2c = to4x4(e["w2c"]);  c2ws.append(inv(w2c).tolist())
    fps.append(e["name"]);  focals.append(e["K"][0][0])
write(f"{solve}/cameras.json", {"filepaths": fps, "cams2world": c2ws, "focals": focals})
```

`focals` is **required** — omitting it raises `KeyError: 'focals'` in `build_blender_scene`
(line ~221). The emitted `cameras.json` benefits every rep on that solve (matcha-15 too).

### The `.blend` — build a persistent, openable scene

The render path uses a throwaway `.blend`. To produce a **persistent** one you can open and
orbit (operator inspection, `/camera-save` re-framing, manual renders), call
`build_blender_scene.py` with the same inputs `render_one` uses:

```bash
# views.json: schema-5 wrapper around the view slot + the scene's anchor_frames
#   {"schema_version":5, "anchor_frames": <views/origin-cameras.json>.anchor_frames,
#    "views":[ {<views/<slot>/view.json>, "name":"<slot>"} ]}
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python real2sim/build_blender_scene.py -- \
  --mesh        <rep>/meshify/<m>/<id>/mesh.ply \
  --cameras-original <solve>/cameras.json \
  --cameras-oriented <solve>/orient/<oid>/oriented.json \
  --output      <out>.blend \
  --view-camera-pose /tmp/views.json --view-name <slot>
```

Then open `<out>.blend` in Blender (it ships with the mesh + the comparison camera as the
active camera). Omit nothing — `build_blender_scene` reads `cams2world`, `focals`, and
`filepaths` from `--cameras-original` and `rotation`/`z_shift` from `--cameras-oriented`.

## Settings / choices

- **Mesh source** — which mesh of a reconstruction to render (e.g. matcha **tsdf** vs **tetra**;
  DA3 **fuse**). The expected-set planner enumerates the rankable meshes per variant.
- **Views** — the full T2 set by default; you can target a subset while iterating.
- **Never `lowmem` TSDF for deliverables** — it's visibly degraded; lowmem is a dev shortcut
  only (story-056).

## Gotchas

- **Missing renders are first-class in the UI** — the rank surface shows MISSING tiles
  (expected − existing) and can trigger materialization (STO-SCN-085/086). A blank in the
  runoff means "not rendered yet," not "bad."
- A render only appears as a rank variant once it exists under `…/renders/` — run
  `render-missing` after every new reconstruction (T3b) before ranking (T4).
- Re-grounding: if a mesh's gauge changed (re-orient), its renders are stale — the
  content-addressed identity changes, so `render-missing` re-renders them.

## Automation status

`v4job.py render-missing` is one command and graph-native (renders exactly the gaps). ✅ automated.

## Next

→ [T4 — View Ranking](T4-view-ranking.md)
