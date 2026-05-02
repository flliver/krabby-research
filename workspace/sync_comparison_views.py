"""Read all "comparison" cameras from a .blend and merge into comparison_views.json.

A "comparison" camera is any Camera object whose name is NOT in the cam_NNN
pattern (those are the MAtCha preset cameras we placed). Typically the user
added these by adding Cameras in Blender (Add → Camera, or the default
"Camera" that ships with new scenes), positioned them, and saved.

Each found camera becomes a named view in the multi-view JSON. If a view
with the same name already exists, it's overwritten (so re-running this
after editing the Camera updates its pose). Anchors are recomputed from
the source variant's cameras.json on each run.

Run via Blender headless:

  Blender --background --python sync_comparison_views.py -- \\
      <source-variant-blend> <source-variant-cameras.json> <output-views.json>

Multiple "Camera*"-style names land as separate views. Rename them in
Blender's Outliner (e.g., "front_door_view", "kitchen_corner") for
meaningful view names.
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

# Load existing JSON (if any) so we update-in-place rather than overwrite
existing_views = []
if os.path.exists(out_json_path):
    with open(out_json_path) as f:
        prev = json.load(f)
    existing_views = prev.get("views", [])

views_by_name = {v["name"]: v for v in existing_views}

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
    views_by_name[cam.name] = {
        "name": cam.name,
        "captured_camera_name": cam.name,
        "captured_at": captured_at,
        "convention": "opencv",  # +X right, +Y down, +Z forward (looking direction)
        "world_position": [loc.x, loc.y, loc.z],
        "world_rotation_quat_wxyz": [quat.w, quat.x, quat.y, quat.z],
        "lens_mm": cd_.lens,
        "sensor_width_mm": cd_.sensor_width,
        "sensor_height_mm": cd_.sensor_height,
    }

# Sort views by name for stable output
out_views = [views_by_name[k] for k in sorted(views_by_name.keys())]

payload = {
    "schema_version": 3,
    "captured_from_blend": blend_path,
    "anchor_frames": anchors,
    "views": out_views,
}

os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
with open(out_json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"\nWrote {out_json_path}")
print(f"  anchors: {len(anchors)} candidates")
print(f"  views: {[v['name'] for v in out_views]}")
