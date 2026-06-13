---
description: Capture the current Blender viewport as a named virtual comparison camera (+1 in scene.blend, materialize the view via the v4 graph writer)
argument-hint: <view-name> [purpose]
allowed-tools: mcp__blender__execute_blender_code, mcp__blender__get_scene_info, mcp__blender__get_viewport_screenshot, Bash, Read
---

# /camera-save — viewport → virtual camera (STO-SCN-046)

Capture the operator's current Blender viewport framing as a named
virtual camera in the open store `scene.blend`, then materialize the
view into the v4 scene store via the graph-native writer
(`v4exec.py views-from-blend`, which writes `views/<slot>/view.json`)
so it is immediately renderable (`v4job.py render-missing`) and
rankable in `rate_renders`.

> **v4 is the path.** The scene store is content-addressed v4
> (HUG-SCN-005 locked #11: `v4exec.py` is the ONLY store writer). The
> legacy v2 path (`sync_comparison_views.py` → `cameras.json`) is
> retired — do not use it for v4 scenes.

**Arguments:** `$1` = view name (required; e.g. `front_left_low`).
`$2` = purpose (optional, default `ab-comparison`; `reference-match`
for whitepaper-matching views).

## Preconditions (check, don't assume)

1. `mcp__blender__get_scene_info` succeeds — a live Blender with the
   MCP addon is connected. If not: tell the operator to open the store
   `scene.blend` in Blender and start the MCP connection; stop.
2. The open file is in the scene store — the capture helper accepts
   both the v4 layout (`scenes/<scene>/scene.blend`) and the legacy v2
   layout (`scenes/<scene>/pipeline-<p>/run-<r>/scene.blend`), and
   errors otherwise. **Confirm the right scene is loaded** (a stale
   blend from a prior scene is a real footgun — check
   `bpy.data.filepath`).

## Steps

### 1. Capture (live session, via MCP)

Execute via `mcp__blender__execute_blender_code`:

```python
exec(open("/private/var/krabby/research/real2sim/viewport_capture.py").read())
result = capture("$1", purpose="${2:-ab-comparison}")
print(result)
```

- On `{"error": ...}` → report the error verbatim and stop (the helper
  rejects camera-view/ortho viewports, bad names, non-store blends).
- On success the helper has: created/updated the camera in
  `cameras_virtual`, derived the TRUE lens from the projection matrix
  (NOT `space.lens` — see helper docstring), set the v4/v5 custom
  props, made it the active scene camera, and **saved the .blend**.

### 2. Visual verification (T-012)

Take `mcp__blender__get_viewport_screenshot`. The viewport is now in
the captured camera's view (active camera was switched) — confirm the
framing matches what the operator set. If it looks wrong, say so
honestly and investigate the lens derivation before proceeding.

### 3. Materialize the view into the v4 store (headless, graph-native)

The helper saved the blend with the viewport-capture camera; now run
the ONLY store writer to read it back into `views/<slot>/view.json`.
From the `result` fields (`scene`, `blend`) — the helper's `next`
field also prints this exact command:

```bash
cd /var/krabby/research/real2sim && \
  .venv/bin/python -u v4exec.py views-from-blend <scene> <blend>
```

`cmd_views` is idempotent by `captured_name` — a camera already in a
slot NOOPs, so re-running after framing the next view never duplicates
an earlier one (the ghost-slot guard). It also updates the canonical
viewset. Confirm the printed slot (e.g. `view 02: 'view-02' lens
25.0mm`, `canonical viewset: ['01', '02']`).

(Legacy v2 scenes only — never v4: the `sync_comparison_views.py` →
`cameras.json` path. The helper's `next` field auto-selects this when
the blend is a `pipeline-/run-` layout.)

### 4. Report

- camera name + created/updated, position, derived `lens_mm` (and
  `viewport_lens` provenance), purpose
- view slot written + canonical viewset
- Offer: render the new view now
  (`.venv/bin/python v4job.py render-missing <scene>`) so its A/B
  matrix appears in rate_renders.

## Notes

- Re-running with the same name **updates** the existing camera
  (re-frame and re-save is the expected iteration loop).
- The capture helper is version-controlled at
  `real2sim/viewport_capture.py` — fix logic there, never inline
  divergent copies (T-023/T-025).
- The blend save + `views-from-blend` are two writes to the scene
  store; the store's auto-sync propagates them to the fleet.
