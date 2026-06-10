---
description: Capture the current Blender viewport as a named virtual comparison camera (+1 in scene.blend, regenerate scene cameras.json)
argument-hint: <view-name> [purpose]
allowed-tools: mcp__blender__execute_blender_code, mcp__blender__get_scene_info, mcp__blender__get_viewport_screenshot, Bash, Read
---

# /camera-save — viewport → virtual camera (STO-SCN-046)

Capture the operator's current Blender viewport framing as a named
virtual camera in the open run-level `scene.blend`, then regenerate the
scene-level unified `cameras.json` (schema 5) so the view is
immediately renderable by `render_comparison_matrix.sh` and rankable in
`rate_renders`.

**Arguments:** `$1` = view name (required; e.g. `front_left_low`).
`$2` = purpose (optional, default `ab-comparison`; `reference-match`
for whitepaper-matching views).

## Preconditions (check, don't assume)

1. `mcp__blender__get_scene_info` succeeds — a live Blender with the
   MCP addon is connected. If not: tell the operator to open the
   run-level `scene.blend` in Blender and start the MCP connection;
   stop.
2. The open file is a run-level store blend — the capture helper
   validates `scenes/<scene>/pipeline-<p>/run-<r>/scene.blend` and
   errors otherwise.

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

### 3. Regenerate the unified cameras.json (headless, hardened path)

From the `result` fields (`scene`, `source_run`, `blend`):

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python /private/var/krabby/research/real2sim/sync_comparison_views.py -- \
  <blend> \
  /var/krabby/scenes/<scene>/<source_run>/transform-*/data/mast3r_sfm/cameras.json \
  /var/krabby/scenes/<scene>/cameras.json
```

(Resolve the `transform-*` glob to the actual transform dir first.
If `/var/krabby/scenes/<scene>/cameras.json` doesn't exist yet, add
`--legacy <scene>/_unsorted/comparison_views.json` when that file
exists.)

### 4. Report

- camera name + created/updated, position, derived `lens_mm` (and
  `viewport_lens` provenance), purpose
- `cameras.json` path + new view count
- Offer: matrix-render the new view now
  (`render_comparison_matrix.sh --scene <scene> --views "$1"`) so it
  appears in rate_renders.

## Notes

- Re-running with the same name **updates** the existing camera
  (re-frame and re-save is the expected iteration loop).
- The capture helper is version-controlled at
  `real2sim/viewport_capture.py` — fix logic there, never inline
  divergent copies (T-023/T-025).
- The blend save + JSON regen are two writes to the scene store; the
  store's auto-sync propagates them to the fleet.
