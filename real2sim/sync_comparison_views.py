"""Read all "comparison" cameras from a .blend and emit the unified
scene-level cameras.json (schema_version 5, STO-SCN-045).

A "comparison" camera is any Camera object whose name is NOT in the cam_NNN
pattern (those are the MAtCha preset cameras we placed). Typically the user
added these by adding Cameras in Blender (Add → Camera, or the default
"Camera" that ships with new scenes), positioned them, and saved.

Each found camera becomes a named view in the multi-view JSON. If a view
with the same name already exists, it's overwritten (so re-running this
after editing the Camera updates its pose). Anchors are recomputed from
the source variant's cameras.json on each run.

Schema 5 = the v4 shape (anchor_frames + views, byte-compatible) PLUS:
  scene          scene name (derived from the output path)
  source_run     "pipeline-<p>/run-<r>" whose SfM frame pool/anchors/views
                 live in (derived from the cameras.json path)
  pool           {filepaths (basenames), focals, cams2world} — verbatim
                 from the run's mast3r cameras.json (raw SfM frame)
  selected_idx   indices into pool that fed the pipeline run
                 (--selected-frames, default: all of pool)

Run via Blender headless:

  Blender --background --python sync_comparison_views.py -- \\
      <source-run-blend> <source-run-cameras.json> <output-cameras.json> \\
      [--selected-frames <selected_frames.json>] \\
      [--legacy <old comparison_views.json>]   # seeds views when output absent

Multiple "Camera*"-style names land as separate views. Rename them in
Blender's Outliner (e.g., "front_door_view", "kitchen_corner") for
meaningful view names.

## Purpose round-trip (schema v4)

Each comparison camera can carry optional Blender custom properties that
get persisted into the JSON and re-applied when build_blender_scene.py
regenerates the .blend:

  view_purpose            "ab-comparison" (default) | "reference-match" | ...
  matches_reference_images JSON array of paths (relative to milestone root)
  render_resolution        JSON [w, h] — applied to scene.render when this
                           view is the active camera
  render_engine            "CYCLES" | "BLENDER_WORKBENCH" | ...
  auto_localized           true | false (was the pose computed by an algorithm
                           or hand-placed?)
  localization_method      "manual" | "mast3r_sfm_extend" | "pnp_..." | ...

To set a custom property in Blender on a Camera object:

  bpy.data.objects["cam_ref"]["view_purpose"] = "reference-match"
  bpy.data.objects["cam_ref"]["matches_reference_images"] = [
      "data/scenes/dtu-bicycle/reference/tsdf_multires.png",
      "data/scenes/dtu-bicycle/reference/adaptive_tetra.png",
  ]

These properties are read on sync and written back on build, so once a
camera's metadata is set (manually or by a localization tool), it persists
across .blend regeneration.
"""
import bpy  # type: ignore
import json
import sys
import os
from datetime import datetime
from mathutils import Matrix  # type: ignore


# Convention conversion: Blender camera (X right, Y up, Z back) → OpenCV
# camera (X right, Y down, Z forward). 180° rotation around local X axis.
# This flip is applied to the camera's BASIS — i.e., M_opencv = M_blender @ flip.
# We store OpenCV in the JSON because both MAtCha-SfM and viser use OpenCV.
_FLIP_BLENDER_TO_OPENCV = Matrix((
    (1, 0, 0, 0),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, 1),
))

argv = sys.argv[sys.argv.index("--") + 1:]
blend_path, cams_json_path, out_json_path = argv[:3]

# Optional flags after the 3 positionals (STO-SCN-045).
_flags = {}
_i = 3
while _i < len(argv):
    if argv[_i].startswith("--") and _i + 1 < len(argv):
        _flags[argv[_i][2:]] = argv[_i + 1]
        _i += 2
    else:
        _i += 1
selected_frames_path = _flags.get("selected-frames")
legacy_views_path = _flags.get("legacy")

bpy.ops.wm.open_mainfile(filepath=blend_path)

# Read source-variant's cameras.json — filepaths in cam_NNN order
with open(cams_json_path) as f:
    cd = json.load(f)
filepaths = cd["filepaths"]
basenames = [p.rsplit("/", 1)[-1] for p in filepaths]
n = len(filepaths)

# Get cam_NNN positions (already in oriented frame)
cam_positions = {}  # basename → [x, y, z]
for i in range(n):
    obj = bpy.data.objects.get(f"cam_{i+1:03d}")
    if obj is None:
        continue
    cam_positions[basenames[i]] = list(obj.location)

# Record ALL preset cameras as candidate anchors. Consumers iterate over
# whichever ones they find in their own dataset and use the intersection.
# More anchors → tighter Procrustes fit, especially when aligning across
# different SfM runs (n350 viewer vs per-variant oriented frames).
anchors = [
    {"basename": basenames[i], "oriented_position": cam_positions[basenames[i]]}
    for i in range(n)
    if basenames[i] in cam_positions
]
print(f"Recording {len(anchors)} candidate anchors (all preset cameras)")

# Find all comparison cameras (anything NOT cam_NNN)
def is_preset(name: str) -> bool:
    if not name.startswith("cam_"):
        return False
    suffix = name[4:]
    return suffix.isdigit()

comparison_cams = [
    o for o in bpy.data.objects
    if o.type == "CAMERA" and not is_preset(o.name)
]
print(f"Found {len(comparison_cams)} comparison camera(s):")
for c in comparison_cams:
    print(f"  - {c.name}")

# Load existing JSON (if any) so we update-in-place rather than overwrite.
# We preserve top-level scene-wide fields (like variant_prefix) that
# aren't owned by this script — they get added by other tooling and must
# survive a sync. When the output doesn't exist yet, --legacy seeds the
# views from the old comparison_views.json (v3/v4 migration, STO-SCN-045).
existing_views = []
prev = {}
if os.path.exists(out_json_path):
    with open(out_json_path) as f:
        prev = json.load(f)
    existing_views = prev.get("views", [])
elif legacy_views_path and os.path.exists(legacy_views_path):
    with open(legacy_views_path) as f:
        prev = json.load(f)
    existing_views = prev.get("views", [])
    print(f"Seeding {len(existing_views)} view(s) from legacy "
          f"{legacy_views_path} (schema {prev.get('schema_version')})")

views_by_name = {v["name"]: v for v in existing_views}

# Optional custom-property keys on Camera objects that round-trip through
# the JSON (set on the camera in Blender, persisted on sync, re-applied on
# build). All are OPTIONAL — sensible defaults if absent.
#   view_purpose             str, default "ab-comparison"
#   matches_reference_images list[str]
#   render_resolution        [w, h]
#   render_engine            str (e.g., "CYCLES", "BLENDER_WORKBENCH")
#   auto_localized           bool
#   localization_method      str

def _read_custom_prop(obj, key, default=None):
    """Read a Blender custom property, normalizing list-like values."""
    if key not in obj:
        return default
    val = obj[key]
    # IDPropertyArray (Blender's wrapper for list custom props) doesn't
    # JSON-serialize directly; convert to plain list.
    if hasattr(val, "to_list"):
        return val.to_list()
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


# Update / add each comparison camera as a view
captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
for cam in comparison_cams:
    # Convert Blender's matrix_world → OpenCV form for storage.
    # Position is unchanged (it's the camera's world location); only the
    # local-axis interpretation changes.
    mat_blender = cam.matrix_world
    mat_opencv = mat_blender @ _FLIP_BLENDER_TO_OPENCV
    loc = mat_opencv.to_translation()
    quat = mat_opencv.to_quaternion()
    cd_ = cam.data
    view = {
        "name": cam.name,
        "captured_camera_name": cam.name,
        "captured_at": captured_at,
        "convention": "opencv",  # +X right, +Y down, +Z forward (looking direction)
        "purpose": _read_custom_prop(cam, "view_purpose", default="ab-comparison"),
        "world_position": [loc.x, loc.y, loc.z],
        "world_rotation_quat_wxyz": [quat.w, quat.x, quat.y, quat.z],
        "lens_mm": cd_.lens,
        "sensor_width_mm": cd_.sensor_width,
        "sensor_height_mm": cd_.sensor_height,
    }
    # Optional metadata fields: only emit if present so JSON stays minimal
    # for plain ab-comparison views.
    for k in ("matches_reference_images", "render_resolution",
              "render_engine", "auto_localized", "localization_method"):
        v = _read_custom_prop(cam, k)
        if v is not None:
            view[k] = v
    views_by_name[cam.name] = view

# Sort views by name for stable output
out_views = [views_by_name[k] for k in sorted(views_by_name.keys())]

# ---- Schema 5 additions (STO-SCN-045): pool + selected_idx + provenance.
# Pool is the run's cameras.json verbatim, except filepaths are reduced to
# basenames — absolute container paths (/data/scenes/...) are meaningless
# outside the run's container and basenames are what every consumer
# (anchor matching, viewer --frames override) actually keys on.
pool = {
    "filepaths": basenames,
    "focals": cd["focals"],
    "cams2world": cd["cams2world"],
}

if selected_frames_path:
    with open(selected_frames_path) as f:
        _sel = json.load(f)
    selected_idx = sorted(_sel["selected_idx"])
    _bad = [i for i in selected_idx if not (0 <= i < n)]
    if _bad:
        raise SystemExit(
            f"ERROR: --selected-frames indices out of range for pool of "
            f"{n}: {_bad[:10]} (n_pool={_sel.get('n_pool')})")
else:
    selected_idx = list(range(n))

# Derive scene + source_run from the cameras.json path within the scene
# store: scenes/<scene>/pipeline-<p>/run-<r>/transform-NN-*/data/...
def _derive_scene_and_run(path):
    parts = os.path.abspath(path).split(os.sep)
    for i, p in enumerate(parts):
        if p.startswith("run-") and i >= 2 and parts[i - 1].startswith("pipeline-"):
            return parts[i - 2], f"{parts[i-1]}/{p}"
    return None, None

scene_name, source_run = _derive_scene_and_run(cams_json_path)

payload = {
    "schema_version": 5,
    "scene": scene_name,
    "source_run": source_run,
    "captured_from_blend": blend_path,
    "pool": pool,
    "selected_idx": selected_idx,
    "anchor_frames": anchors,
    "views": out_views,
}
# Preserve scene-wide fields that other tooling owns (e.g., variant_prefix
# is read by render_comparison_matrix.sh). Anything in the previous JSON
# that we don't explicitly write here is carried forward.
for k, v in prev.items():
    if k not in payload:
        payload[k] = v
# variant_prefix default: the scene name (new-store scenes don't need the
# legacy "<prefix>-curated-" stem distinction).
if not payload.get("variant_prefix") and scene_name:
    payload["variant_prefix"] = scene_name

_out_dir = os.path.dirname(out_json_path)
if _out_dir:
    os.makedirs(_out_dir, exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nWrote {out_json_path} (schema 5)")
print(f"  scene: {scene_name}  source_run: {source_run}")
print(f"  pool: {len(basenames)} cameras, selected: {len(selected_idx)}")
print(f"  anchors: {len(anchors)} candidates")
print(f"  views: {[v['name'] for v in out_views]}")
