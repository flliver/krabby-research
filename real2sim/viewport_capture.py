"""Capture the current 3D viewport as a virtual comparison camera.

STO-SCN-046 (/camera-save). Designed to be exec'd INSIDE a live Blender
session via the Blender MCP (`execute_blender_code`):

    exec(open("/private/var/krabby/research/real2sim/viewport_capture.py").read())
    result = capture("front_left_low")          # or capture(name, purpose=...)
    print(result)

What it does:
  1. Validates the open .blend lives in the scene store and derives the
     scene context from bpy.data.filepath. Accepts both the v2 run-level
     layout (scenes/<scene>/pipeline-<p>/run-<r>/scene.blend) and the v4
     layout (any .blend under scenes/<scene>/, e.g.
     scenes/<scene>/scene.blend).
  2. Reads the active 3D viewport's pose (region_3d.view_matrix.inverted())
     and TRUE field of view from the projection (window_matrix).

     Why not space.lens? Blender's viewport at lens=50 shows a WIDER view
     than a 50 mm camera (the viewport projection is not the camera
     projection). Copying space.lens verbatim makes the saved camera
     render tighter than what the operator framed. Deriving the lens from
     window_matrix — lens = (sensor/2) / tan(fov/2) with fov taken from
     the projection's diagonal terms — reproduces the framed view exactly,
     regardless of how Blender maps viewport lens internally.
  3. Creates/updates a Camera object named <name> in the cameras_virtual
     collection, tagged localization_method="viewport-capture", with the
     schema-v4/v5 custom properties read back by the store writer.
     Re-capture with the same name = update.
  4. Saves the .blend in place.

The caller (the /camera-save skill) then runs the store writer headless
to materialize the view. For v4 scenes this is the graph-native
`v4exec.py views-from-blend <scene> <blend>` (writes
scenes/<scene>/views/<slot>/view.json — the ONLY store writer, locked
#11). The legacy v3 path was sync_comparison_views.py →
scenes/<scene>/cameras.json; do not use it for v4 scenes.
"""
import math

import bpy  # type: ignore


def _find_view3d():
    """Return (space, region_3d) of the largest visible VIEW_3D area."""
    best = None
    best_size = -1
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            size = area.width * area.height
            if size > best_size:
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        best = (space, space.region_3d)
                        best_size = size
    return best


def _derive_run_context(filepath):
    """v2: scenes/<scene>/pipeline-<p>/run-<r>/scene.blend → (scene, pipeline/run).
    v4 (HUG-SCN-005): any .blend under scenes/<scene>/ → (scene, relative path).
    Returns (None, None) when the file isn't in the scene store at all."""
    parts = filepath.split("/")
    for i, p in enumerate(parts):
        if p.startswith("run-") and i >= 2 and parts[i - 1].startswith("pipeline-"):
            return parts[i - 2], f"{parts[i-1]}/{p}"
    # v4 layout: scene dir directly under the store root
    for i, p in enumerate(parts):
        if p == "scenes" and i + 1 < len(parts) - 1:
            return parts[i + 1], "/".join(parts[i + 2:-1]) or "."
    return None, None


def capture(name, purpose="ab-comparison", sensor_width=36.0,
            render_resolution=(1920, 1080), render_engine="BLENDER_WORKBENCH"):
    """Capture the viewport as virtual camera <name>. Returns a dict report."""
    if not name or "/" in name or name.startswith("cam_"):
        return {"error": f"bad name {name!r} — must be non-empty, no '/', and "
                         f"not the cam_NNN preset pattern"}

    filepath = bpy.data.filepath
    if not filepath:
        return {"error": "no .blend open"}
    scene_name, source_run = _derive_run_context(filepath)
    if scene_name is None:
        return {"error": f"open file {filepath} is not in the scene store "
                         f"(expected scenes/<scene>/... — v4 scene.blend or "
                         f"v2 pipeline-<p>/run-<r>/scene.blend)"}

    v3d = _find_view3d()
    if v3d is None:
        return {"error": "no 3D viewport found"}
    space, rv3d = v3d
    if rv3d.view_perspective == "CAMERA":
        return {"error": "viewport is in camera view — frame in free "
                         "perspective (the capture would just duplicate the "
                         "active camera)"}
    if rv3d.view_perspective == "ORTHO":
        return {"error": "viewport is orthographic — switch to perspective "
                         "(numpad 5) to frame a camera shot"}

    # Pose: viewport view matrix is world→view; invert for cam→world.
    # update() first: view_matrix is recomputed on draw, so a viewport
    # framed programmatically (or any not-yet-redrawn change) would
    # otherwise yield the previous pose (observed in the 2026-06-09
    # live test — capture returned the startup framing).
    rv3d.update()
    cam2world = rv3d.view_matrix.inverted()

    # True FOV from the projection matrix. window_matrix[0][0] = 1/tan(fov_x/2),
    # [1][1] = 1/tan(fov_y/2). Use the horizontal term against a HORIZONTAL
    # sensor fit so the saved camera reproduces the framed width exactly.
    p00 = rv3d.window_matrix[0][0]
    p11 = rv3d.window_matrix[1][1]
    fov_x = 2.0 * math.atan(1.0 / p00)
    lens_mm = (sensor_width / 2.0) / math.tan(fov_x / 2.0)

    existing = bpy.data.objects.get(name)
    if existing is not None and existing.type != "CAMERA":
        return {"error": f"object {name!r} exists and is not a camera"}
    if existing is None:
        cam_data = bpy.data.cameras.new(name=name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        coll = bpy.data.collections.get("cameras_virtual")
        if coll is None:
            coll = bpy.data.collections.new("cameras_virtual")
            bpy.context.scene.collection.children.link(coll)
        coll.objects.link(cam_obj)
        action = "created"
    else:
        cam_obj = existing
        action = "updated"

    cam_obj.matrix_world = cam2world
    cam_obj.data.lens = lens_mm
    cam_obj.data.sensor_width = sensor_width
    cam_obj.data.sensor_height = sensor_width * 24.0 / 36.0
    cam_obj.data.sensor_fit = "HORIZONTAL"
    cam_obj.hide_render = False

    # v4/v5 round-trip metadata (read back by the store writer:
    # v4exec.py views-from-blend for v4, sync_comparison_views.py for v2).
    cam_obj["view_purpose"] = purpose
    cam_obj["render_resolution"] = list(render_resolution)
    cam_obj["render_engine"] = render_engine
    cam_obj["auto_localized"] = False
    cam_obj["localization_method"] = "viewport-capture"
    cam_obj["viewport_lens"] = float(space.lens)  # provenance only

    # Make it the active scene camera so a camera-view toggle (numpad 0)
    # immediately shows the operator what was captured (T-012 for humans).
    bpy.context.scene.camera = cam_obj

    bpy.ops.wm.save_mainfile()

    loc = cam_obj.matrix_world.to_translation()
    return {
        "action": action,
        "name": name,
        "scene": scene_name,
        "source_run": source_run,
        "blend": filepath,
        "position": [round(v, 4) for v in loc],
        "lens_mm": round(lens_mm, 2),
        "viewport_lens": float(space.lens),
        "fov_x_deg": round(math.degrees(fov_x), 2),
        "fov_y_check_deg": round(math.degrees(2.0 * math.atan(1.0 / p11)), 2),
        "purpose": purpose,
        "next": (
            f"v4exec.py views-from-blend {scene_name} {filepath}  "
            f"(graph-native writer → views/<slot>/view.json)"
            if "run-" not in (source_run or "") else
            "run sync_comparison_views.py headless against this blend "
            "to regenerate scenes/<scene>/cameras.json (legacy v2)"),
    }
