---
kind: note
captured: 2026-05-01
consolidated: false
tags: [b5, frame-curation, ui, viser, blender, feasibility]
---

# Camera selection UI — feasibility and routes

## The problem

When you have 60–200 candidate camera frusta in 3D and need to hand-pick ~12, "scroll through them in a list" doesn't work. You need to **filter** — by where the camera is, where it's pointing, when in the video it was captured, what it sees. The original spatial-curation note assumed Blender Collections would be enough; this note revisits that and lays out a more complete UI plan.

## What Jeremy described as the ideal

1. Get all 3D positions and frustums from MASt3R-SfM.
2. Render a web page that views the 3D space (frustums + images, no mesh).
3. A connecting line showing the camera path.
4. Inputs to easily select and de-select different cameras.
5. Sliders/groupings/whatever to conditionally show/hide subsets — because if you're focusing on one area of a room, you might want 30+ cameras in that area, not the rest.

This is a **continuous filtering** problem, not a discrete-bucket problem. Continuous controls (sliders, look-at gizmos) compose better than checkboxes when the user's question of the day is unpredictable.

## Two routes

### Route A — extended Blender scene (cheap, fast, discrete-bucket)

B3's existing `build_blender_scene.py` already builds Blender scenes with Camera objects + textured image planes. Three Blender features cover most of the requirement for free:

- **Collections** with one-click visibility toggles → groupings.
- **Per-camera hide (`H`)** → individual selection control.
- **Outliner type-ahead search** → filter by name pattern.

What Blender doesn't give you natively:
- Continuous sliders. You discretize (e.g., 5 time-buckets as 5 collections) instead of a smooth slider.
- "Show cameras pointing toward X" as a continuous filter. Precompute view-direction clusters into collections.
- Web-shareable. It's a `.blend` file.

**Scope of work:**
1. Extend `build_blender_scene.py` to accept N cameras (not 12 hardcoded).
2. Make the mesh-import step optional.
3. Compute clusterings (k-means on 3D position, k-means on view direction, time-buckets) and assign each camera to one collection per axis.
4. Build a polyline curve connecting cameras in temporal order.
5. Emit a `selection.json` template that the export step can fill in.

**Effort:** ~0.5 day. Pure Python, all libraries already on hand (numpy, scikit-learn, bpy).

### Route B — viser-based web viewer (the real version)

[viser](https://github.com/nerfstudio-project/viser) is a Python WebGL viewer built by the Nerfstudio team for exactly this category of problem (interactive 3D visualization of camera-rich scenes). MIT-licensed, pip-installable.

What viser provides out of the box:

- `scene.add_camera_frustum(...)` — camera frustums with image-plane support.
- `scene.add_spline_catmull_rom(...)` — connecting curves for the camera path.
- `gui.add_slider(...)`, `gui.add_checkbox(...)`, `gui.add_multi_slider(...)` — sliders/checkboxes/dropdowns wired to Python callbacks.
- `frustum.on_click(callback)` — click-to-toggle on individual frustums.
- `gui.add_text(...)` for selection counter.
- Local web server; Jeremy opens `http://bbeeprz:8080` in his browser.

The mental model: write Python that says "for each camera in `cameras.json`, add a frustum with this image-plane texture; here's a slider for time-range; on slider change, set `frustum.visible = (start_t <= cam.t <= end_t)`." viser handles the WebGL.

**Filters worth wiring up (each is an independent boolean; visible iff all active filters agree):**

| Filter axis | UI control | What it answers |
|-------------|------------|-----------------|
| Time range | dual-handle slider | "Show cameras from the first walk-around segment" |
| Spatial cluster | checkboxes (k-means on 3D position) | "Show only cameras near the firepit" |
| View direction | look-at gizmo or 6 face buckets | "Show only cameras pointing at the table" |
| Image similarity | ASMK cluster checkboxes | "Show only cameras whose visual content is in this cluster" (catches near-duplicates) |
| Co-visibility | scene-graph clusters from MASt3R-SfM | "Show only cameras that share scene-graph edges with the selected one" |
| Picked status | tri-state per camera (unpicked / picked / hidden) | "What have I selected so far?" |
| Selection counter | text widget | "12 / 60 selected" |

**Click semantics:**
- Single-click frustum → toggle picked status.
- Double-click → fly to that frustum's view.
- Hover → show frame index + thumbnail in a side panel.

**Output:** `selected_frames.json` with the chosen frame indices. Re-extract those exact frames, run MAtCha.

**Effort:**
- v1 (frustums + image planes + temporal slider + click-to-pick + JSON export): ~1 day.
- + spatial clustering: +0.5 day.
- + view-direction filtering: +0.5–1 day (look-at gizmo is the harder bit).
- + ASMK similarity grouping: +0.5 day.
- + co-visibility clustering: +0.5 day.

Full ideal: **2–3 days**. v1 working version: **1 day**.

## Performance considerations

For ~200 frustums-with-image-planes, each JPEG at 1024×576 decodes to ~2 MB raw RGB in browser memory. 200 × 2 MB = 400 MB texture memory. Modern GPUs handle that, but it's not trivial. Mitigation: downscale image-plane textures to 512×288 (~1.5 MB *total* for 200 cams after compression) for the viewer; originals remain on disk for the actual MAtCha run.

For 60 cameras (the realistic curation pool today), memory is a non-issue.

## Grouping strategies — beyond the obvious ones

The SfM literature suggests two clusterings that map well to "where are they positioned and pointing":

1. **Co-visibility clustering.** From the SfM scene graph: "these N cameras share enough correspondences that they're seeing the same physical region." Already computed by MASt3R-SfM as part of its pipeline — we get it for free. Probably the **most useful clustering** for "show me cameras that see the same area," because it's grounded in actual feature-match overlap rather than a heuristic on positions.

2. **Frustum-overlap clustering.** Two cameras "overlap" if their viewing frusta intersect a common scene region. Computed from frustum geometry (no learned features needed). More principled than spatial-position clustering for "show cameras looking at this object."

Both are worth adding to Route B if it gets built.

## Recommendation

**Phase 1: Route A first.** Extend `build_blender_scene.py` for 60+ cameras with discrete-bucket collections (spatial / time / direction). ~0.5 day. Use for the first scene 004 curation experiment. See whether Blender Collections + visibility toggles feel sufficient or whether lack of continuous filtering is a real friction.

**Phase 2: Route B if needed.** If Phase 1 reveals that curation is bottlenecked by "I keep wanting to filter by something I don't have a collection for," build the viser viewer. ~1 day for v1. **The Phase 1 clustering code carries forward** — both routes share the cluster-computation step, so the work isn't wasted.

**Pivot criterion:** if curation in Blender takes >20 min per scene, or if Jeremy keeps creating ad-hoc collections during a session, escalate to viser.

## Where this fits in the larger plan

This note expands the "B3 extension" line item from `2026-05-01-spatial-frame-curation-via-mast3r-sfm`. That note assumed Blender alone would suffice; this note doesn't disagree, but lays out the upgrade path so we have it queued if/when needed.

The order remains: C (`r` knob) first, then B (this UI work) second.

## Status

To be folded into the `options-on-the-table-after-b6a` entry under Option B, alongside the spatial-curation note. Once that update lands, flip `consolidated: true`.
