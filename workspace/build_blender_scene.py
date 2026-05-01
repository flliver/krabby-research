"""Build a Blender .blend file containing the oriented mesh + 12 camera objects.

Run via Blender headless:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python build_blender_scene.py -- \
      --mesh <path>/oriented_500k.obj \
      --cameras-original <path>/cameras.json \
      --cameras-oriented <path>/oriented_cameras.json \
      --output <path>/scene.blend

Inputs:
  - mesh: 500K-tri oriented OBJ
  - cameras-original: MAtCha's cameras.json with cams2world 4x4 matrices and focals
  - cameras-oriented: orient_mesh.py output with rotation R + z_shift

Output:
  - scene.blend with the mesh imported and 12 Blender Camera objects positioned
    at the transformed camera poses, with focal length from the original
    cameras.json. Cameras named cam_001 through cam_012.
"""
import bpy  # type: ignore  # only resolves inside Blender
import json
import sys
import os
import math
from mathutils import Matrix, Vector


def parse_args():
    # Blender passes args after `--` to the script
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            args[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)


def opencv_cam_to_blender(cam2world_4x4):
    """Convert an OpenCV/COLMAP-style cams2world (X right, Y down, Z forward)
    to a Blender camera transform (X right, Y up, Z back).

    The conversion is a 180° rotation around the camera's local X axis,
    applied to the camera's basis. Equivalent to flipping Y and Z columns.
    """
    M = Matrix(cam2world_4x4)
    flip = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return M @ flip


def main():
    args = parse_args()
    mesh_path = args["mesh"]
    cams_orig_path = args["cameras-original"]
    cams_oriented_path = args["cameras-oriented"]
    output_path = args["output"]

    print(f"Building Blender scene")
    print(f"  mesh:       {mesh_path}")
    print(f"  cams orig:  {cams_orig_path}")
    print(f"  cams or'd:  {cams_oriented_path}")
    print(f"  output:     {output_path}")

    clear_scene()

    # Import the mesh
    print("Importing mesh...")
    if mesh_path.lower().endswith(".obj"):
        # Blender 4.x: bpy.ops.wm.obj_import (not import_scene.obj)
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif mesh_path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        raise SystemExit(f"Unsupported mesh format: {mesh_path}")

    # Rename the imported mesh
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.name = "scene_mesh"
            print(f"  imported as: {o.name}, {len(o.data.vertices)} verts, {len(o.data.polygons)} polys")
            break

    # Load camera transforms
    with open(cams_orig_path) as f:
        cams_orig = json.load(f)
    with open(cams_oriented_path) as f:
        cams_oriented_meta = json.load(f)

    R = Matrix([row + [0.0] for row in cams_oriented_meta["rotation"]] + [[0, 0, 0, 1]])
    z_shift = cams_oriented_meta["z_shift"]
    T = Matrix.Translation(Vector((0, 0, z_shift)))
    world_orient = T @ R  # apply R then translate

    cams_world = cams_orig["cams2world"]
    focals = cams_orig["focals"]
    n_cams = len(cams_world)
    print(f"Adding {n_cams} cameras...")

    # MAtCha's downscaled image size — we extracted at 1024×576 (16:9)
    # but MAtCha internally resized to 512×288-ish. The focal in cameras.json is
    # in MAtCha's internal pixel space; we'll set Blender's sensor dimensions
    # to match so framing looks right.
    # Use a reasonable approximation: assume focal is for a 512-pixel-wide image.
    image_width_px = 512.0
    sensor_width_mm = 36.0  # standard 35mm sensor

    for i, c2w_orig in enumerate(cams_world):
        # Apply the orientation transform to the camera pose
        c2w_orig_m = Matrix(c2w_orig)
        c2w_oriented = world_orient @ c2w_orig_m

        # Convert to Blender's camera convention
        c2w_blender = opencv_cam_to_blender(c2w_oriented)

        # Create camera
        cam_data = bpy.data.cameras.new(name=f"cam_{i+1:03d}")
        focal_px = focals[i] if i < len(focals) else focals[0]
        focal_mm = focal_px / image_width_px * sensor_width_mm
        cam_data.lens = focal_mm
        cam_data.sensor_width = sensor_width_mm

        cam_obj = bpy.data.objects.new(name=f"cam_{i+1:03d}", object_data=cam_data)
        cam_obj.matrix_world = c2w_blender
        bpy.context.collection.objects.link(cam_obj)

    # Set the first camera as active
    if bpy.data.objects.get("cam_001"):
        bpy.context.scene.camera = bpy.data.objects["cam_001"]

    # Add a default world light so the mesh is visible
    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.location = (5, -5, 10)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    bpy.context.collection.objects.link(sun_obj)

    # Set viewport so the user sees something useful when opening the .blend
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Wrote {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
    print("DONE.")


if __name__ == "__main__":
    main()
