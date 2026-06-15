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

### The `.blend` (if you need the scene assembled)

`real2sim/build_blender_scene.py` (headless) assembles an oriented mesh + vertex-color material
+ camera objects + image empties → a `scene.blend`. The render path above renders meshes
directly; the `.blend` is for operator inspection / `/camera-save` (T2) / manual renders.

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
