"""Build a Blender .blend file containing the oriented mesh + camera objects.

Run via Blender headless:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python build_blender_scene.py -- \
      --mesh <path>/oriented_500k.obj \
      --cameras-original <path>/cameras.json \
      --cameras-oriented <path>/oriented_cameras.json \
      [--frames-dir <path>/mast3r_sfm/images] \
      [--selected-frames <path>/selected_frames.json] \
      [--output <path>/scene.blend]

Inputs:
  - mesh: 500K-tri oriented OBJ or PLY (PLY preserves vertex colors → auto-wired
    Principled BSDF material with Attribute-Color base color)
  - cameras-original: MAtCha's cameras.json with cams2world 4x4 matrices and focals
  - cameras-oriented: orient_mesh.py output with rotation R + z_shift
  - frames-dir (optional): the dir of source images (e.g., mast3r_sfm/images/).
    When provided, an Image Empty is created at each camera's position
    showing what that camera actually saw. Visible in any viewport shading
    mode; oriented to match the camera's pose; parented to the camera so
    moving the camera moves the image with it.
  - selected-frames (optional): camera_viewer's selected_frames.json
    ({"selected_idx": [...]}, zero-based indices into cameras-original).
    Cameras in selected_idx land in the `cameras_selected` collection,
    the rest in `cameras_pool`. Without this flag ALL cameras go to
    `cameras_selected` (run-level cameras.json is already the curated
    subset) and `cameras_pool` stays empty.
  - output (optional): where to write the .blend. When omitted, derived
    from the input paths: the nearest enclosing `run-*` directory of
    --cameras-original gets `<run-dir>/scene.blend` (STO-SCN-044). Fails
    loudly if neither --output nor a run-* ancestor exists.

Output (STO-SCN-044 grouping):
  - scene.blend organized into named collections, each toggleable in the
    Outliner:
      meshes            — imported geometry
      cameras_pool      — SfM pool cameras NOT fed to the pipeline run
      cameras_selected  — the curated subset (cam_NNN + cam_NNN_view planes)
      cameras_virtual   — comparison/reference views (--view-camera-pose)
    Cameras named cam_001..cam_NNN by pool index; focal length from
    cameras.json. If frames-dir is provided, textured planes named
    cam_NNN_view are parented to each camera.
"""
import bpy  # type: ignore  # only resolves inside Blender
import json
import sys
import os
import math
import numpy as _np  # for Kabsch in view-camera-pose schema_v2 alignment
from mathutils import Matrix, Vector  # type: ignore  # only resolves inside Blender


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


def get_or_create_collection(name):
    """Return the named collection, creating + linking it under the scene
    collection if absent (STO-SCN-044 camera grouping)."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def link_into(obj, coll):
    """Link obj into coll exclusively (unlink from any other collection —
    importers/operators auto-link into the active collection)."""
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def derive_output_path(cams_orig_path):
    """Default output: <run-dir>/scene.blend, where run-dir is the nearest
    `run-*` ancestor of --cameras-original (scene-store layout
    scenes/<scene>/pipeline-<p>/run-<r>/transform-NN-<t>/data/...).
    Returns None when no run-* ancestor exists (caller fails loudly)."""
    d = os.path.dirname(os.path.abspath(cams_orig_path))
    while d and d != os.path.dirname(d):
        if os.path.basename(d).startswith("run-"):
            return os.path.join(d, "scene.blend")
        d = os.path.dirname(d)
    return None


def main():
    args = parse_args()
    mesh_path = args["mesh"]
    cams_orig_path = args["cameras-original"]
    cams_oriented_path = args["cameras-oriented"]
    output_path = args.get("output")
    if not output_path:
        output_path = derive_output_path(cams_orig_path)
        if not output_path:
            raise SystemExit(
                "ERROR: --output omitted and --cameras-original has no run-* "
                "ancestor directory to derive <run-dir>/scene.blend from. "
                "Pass --output explicitly."
            )
        print(f"  (output derived from run dir: {output_path})")

    print(f"Building Blender scene")
    print(f"  mesh:       {mesh_path}")
    print(f"  cams orig:  {cams_orig_path}")
    print(f"  cams or'd:  {cams_oriented_path}")
    print(f"  output:     {output_path}")

    clear_scene()

    # STO-SCN-044: named, toggleable collections for camera grouping.
    coll_meshes = get_or_create_collection("meshes")
    coll_pool = get_or_create_collection("cameras_pool")
    coll_selected = get_or_create_collection("cameras_selected")
    coll_virtual = get_or_create_collection("cameras_virtual")

    # Import the mesh.
    # IMPORTANT: our orient_mesh.py output is Z-up (floor at z=0, normal +Z).
    # Force the importer to treat the file as Z-up — otherwise the OBJ
    # importer's default `up_axis='Y'` rotates the scene 90° (or worse,
    # appears upside-down because of the chained X/Y flip implied by the
    # default forward axis '-Z'). The PLY importer defaults to Z-up but
    # we set it explicitly anyway for safety.
    print("Importing mesh...")
    if mesh_path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
    elif mesh_path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
    else:
        raise SystemExit(f"Unsupported mesh format: {mesh_path}")

    # Rename the imported mesh + check for vertex colors
    mesh_obj = None
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.name = "scene_mesh"
            mesh_obj = o
            link_into(o, coll_meshes)
            print(f"  imported as: {o.name}, {len(o.data.vertices)} verts, {len(o.data.polygons)} polys")
            break

    # Detect vertex colors and wire up a material that displays them
    if mesh_obj is not None:
        has_vcolors = bool(mesh_obj.data.color_attributes) or bool(mesh_obj.data.vertex_colors)
        if has_vcolors:
            attr_name = (
                mesh_obj.data.color_attributes.active_color_name
                if mesh_obj.data.color_attributes
                else mesh_obj.data.vertex_colors.active.name
            )
            print(f"  found vertex colors: '{attr_name}', wiring up material...")
            mat = bpy.data.materials.new(name="VertexColor")
            mat.use_nodes = True
            nt = mat.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            attr = nt.nodes.new("ShaderNodeAttribute")
            attr.attribute_name = attr_name
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs["Roughness"].default_value = 0.9
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            attr.location = (-400, 0)
            bsdf.location = (-100, 0)
            out.location = (200, 0)
            mesh_obj.data.materials.append(mat)
            # Set viewport shading to Material Preview so colors show
            for area in bpy.context.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "MATERIAL"
        else:
            print("  no vertex colors detected; mesh stays untextured")

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
    filepaths = cams_orig.get("filepaths", [])  # for view-camera-pose schema_v2 anchors
    n_cams = len(cams_world)

    # STO-SCN-044: optional pool/selected partition from camera_viewer's
    # selected_frames.json. Without it, every camera is "selected" (the
    # run-level cameras.json is already the curated subset).
    selected_frames_path = args.get("selected-frames")
    if selected_frames_path:
        with open(selected_frames_path) as f:
            sel = json.load(f)
        selected_idx = set(sel["selected_idx"])
        bad = [i for i in selected_idx if not (0 <= i < n_cams)]
        if bad:
            raise SystemExit(
                f"ERROR: selected_frames.json indices out of range for "
                f"{n_cams} cameras: {sorted(bad)[:10]} "
                f"(pool mismatch? n_pool={sel.get('n_pool')})"
            )
        print(f"Adding {n_cams} cameras "
              f"({len(selected_idx)} selected / {n_cams - len(selected_idx)} pool, "
              f"from {selected_frames_path})...")
    else:
        selected_idx = set(range(n_cams))
        print(f"Adding {n_cams} cameras (all → cameras_selected; no --selected-frames)...")

    # Optional: load source frames for image-empty placement
    frames_dir = args.get("frames-dir")
    frame_paths = []
    if frames_dir and os.path.isdir(frames_dir):
        frame_paths = sorted(
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  + Image empties from {frames_dir}: {len(frame_paths)} frames available")
    else:
        if frames_dir:
            print(f"  WARNING: --frames-dir {frames_dir} not found, skipping image empties")

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
        cam_coll = coll_selected if i in selected_idx else coll_pool
        cam_coll.objects.link(cam_obj)
        # Hide preset cameras from final renders — they're authoring aids only.
        # Toggle off in Blender's Outliner if you want to see frustums in a render.
        cam_obj.hide_render = True

        # Optional: textured-plane mesh showing what this camera saw.
        # We use an actual plane mesh + emission-shader material rather than
        # an Image Empty because empty_image_add requires a 3D-View context
        # which doesn't exist in --background mode.
        if i < len(frame_paths):
            img = bpy.data.images.load(frame_paths[i])
            iw, ih = img.size
            aspect = ih / iw if iw > 0 else 9 / 16
            plane_w = 0.4
            plane_h = plane_w * aspect

            # 4-vertex quad in XY plane, normal +Z, centered on origin
            mesh_data = bpy.data.meshes.new(name=f"cam_{i+1:03d}_view_mesh")
            mesh_data.from_pydata(
                [
                    (-plane_w / 2, -plane_h / 2, 0),
                    (plane_w / 2, -plane_h / 2, 0),
                    (plane_w / 2, plane_h / 2, 0),
                    (-plane_w / 2, plane_h / 2, 0),
                ],
                [],
                [(0, 1, 2, 3)],
            )
            mesh_data.update()
            # Set UVs to match Blender's image convention: UV (0,0) is the
            # bottom-left of the displayed image, (1,1) is top-right.
            # Map plane vertices (in CCW order around +Z normal) to image
            # corners so the image appears right-side up when viewed from
            # the camera (i.e., looking down -Z onto the plane's +Z face).
            uv = mesh_data.uv_layers.new(name="UVMap")
            uv.data[0].uv = (0.0, 0.0)  # vertex 0 (-w/2, -h/2) → image bottom-left
            uv.data[1].uv = (1.0, 0.0)  # vertex 1 (+w/2, -h/2) → image bottom-right
            uv.data[2].uv = (1.0, 1.0)  # vertex 2 (+w/2, +h/2) → image top-right
            uv.data[3].uv = (0.0, 1.0)  # vertex 3 (-w/2, +h/2) → image top-left

            plane = bpy.data.objects.new(name=f"cam_{i+1:03d}_view", object_data=mesh_data)
            cam_coll.objects.link(plane)  # same group as its parent camera
            # Source-frame thumbnails are authoring aids — hide from renders.
            plane.hide_render = True

            # Material: image texture into Emission shader (bright in viewport
            # without scene lighting).
            mat = bpy.data.materials.new(name=f"cam_{i+1:03d}_view_mat")
            mat.use_nodes = True
            nt = mat.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            emission = nt.nodes.new("ShaderNodeEmission")
            emission.inputs["Strength"].default_value = 1.0
            outm = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
            nt.links.new(emission.outputs["Emission"], outm.inputs["Surface"])
            tex.location = (-400, 0)
            emission.location = (-100, 0)
            outm.location = (200, 0)
            mesh_data.materials.append(mat)

            # Parent to camera; 0.5 m in front along camera's -Z (forward),
            # rotation matches camera so plane is perpendicular to view axis.
            plane.parent = cam_obj
            plane.matrix_parent_inverse = Matrix.Identity(4)
            plane.location = (0.0, 0.0, -0.5)
            plane.rotation_euler = (0.0, 0.0, 0.0)

    # Optional: persistent comparison view camera(s) (--view-camera-pose <json>).
    # Reads a comparison_view.json. Schema versions:
    #   v1 = single view, absolute coords (legacy; same SfM frame required)
    #   v2 = single view, anchor-based Procrustes alignment
    #   v3 = list of views sharing anchors, anchor-based alignment, no purpose
    #   v4 = v3 + per-view `purpose` and optional metadata (matches_reference_images,
    #        render_resolution, render_engine, auto_localized, localization_method)
    #
    # For v3+v4: ALL views are injected by default. --view-name selects which
    # one is the active scene camera at .blend open (defaults to the first view).
    # Each view becomes its own Blender Camera object named after view['name'].
    # Optional per-view metadata is round-tripped via Blender custom properties
    # (e.g., bpy.data.objects[view_name]["view_purpose"]) so sync_comparison_views.py
    # can re-emit it on the next harvest.
    #
    # For schema v2: a single view (legacy single-view JSON), 3 anchor frames
    # identified by basename are looked up in this variant's cam_NNN poses, Kabsch
    # solves the rigid+scale transform between the source variant's oriented frame
    # and this variant's oriented frame, and the view_cam pose is transformed
    # accordingly.
    view_cam_pose_path = args.get("view-camera-pose")
    view_cam_obj = None  # the active scene camera (one of the injected views)
    # Default Blender object name (used only for legacy schema v1/v2 single-view).
    view_cam_blender_name = "view_cam"
    if view_cam_pose_path and os.path.exists(view_cam_pose_path):
        print(f"Adding view cameras from {view_cam_pose_path}...")
        with open(view_cam_pose_path) as f:
            vc = json.load(f)

        schema = vc.get("schema_version", 1)
        _multi_view_meta = None  # populated when schema >= 3 (set below)

        # Schema v3 / v4 / v5: list of views with shared anchors. Inject ALL
        # views. (v5 = unified scene-level cameras.json, STO-SCN-045 — same
        # anchor_frames/views shape plus pool/selected_idx which this path
        # ignores.) --view-name optionally selects which one becomes the
        # active scene camera (default: first view). Optional per-view
        # metadata is stored as Blender custom properties for round-trip via
        # sync_comparison_views.py.
        if schema in (3, 4, 5):
            views = vc.get("views", [])
            if not views:
                raise SystemExit(f"schema_v{schema} file has no 'views' list")
            requested_view_name = args.get("view-name")
            if requested_view_name:
                matching = [v for v in views if v["name"] == requested_view_name]
                if not matching:
                    raise SystemExit(
                        f"  ERROR: view '{requested_view_name}' not in {view_cam_pose_path}. "
                        f"Available: {[v['name'] for v in views]}"
                    )
                active_view_name = requested_view_name
            else:
                active_view_name = views[0]["name"]
            print(f"  injecting {len(views)} view(s); active = '{active_view_name}'")

            # Anchor-based Procrustes is computed once per build (anchors are
            # shared across views). We compute it once here, then apply to
            # each view inline below.
            _multi_view_meta = {
                "anchor_frames": vc["anchor_frames"],
                "views": views,
                "active_view_name": active_view_name,
            }
            # Use a sentinel schema value (-2) below to indicate "iterate views";
            # the existing schema=2 block does the per-view math.
            schema = -2

        if schema == -2 and _multi_view_meta is not None:
            # Multi-view path (schema v3 / v4): Procrustes anchor alignment is
            # computed once (anchors are shared across views), then applied per
            # view. Each view becomes its own Blender Camera object named after
            # view['name']. Optional per-view metadata (purpose, matches_reference_images,
            # render_resolution, render_engine, auto_localized, localization_method)
            # is attached as Blender custom properties for round-trip via
            # sync_comparison_views.py.
            anchors = _multi_view_meta["anchor_frames"]
            views = _multi_view_meta["views"]
            active_view_name = _multi_view_meta["active_view_name"]

            # Collect anchor correspondences: source-frame oriented_position
            # ↔ this-variant cam_NNN location.
            source_anchor_pos = []
            target_anchor_pos = []
            target_basenames = [fp.rsplit("/", 1)[-1] for fp in filepaths]
            missing = []
            for a in anchors:
                bn = a["basename"]
                if bn not in target_basenames:
                    missing.append(bn)
                    continue
                target_idx = target_basenames.index(bn)
                tgt_obj = bpy.data.objects.get(f"cam_{target_idx+1:03d}")
                if tgt_obj is None:
                    continue
                source_anchor_pos.append(a["oriented_position"])
                target_anchor_pos.append(list(tgt_obj.location))
            if len(source_anchor_pos) < 3:
                raise SystemExit(
                    f"  ERROR: need ≥3 matching anchors; got {len(source_anchor_pos)}. "
                    f"Missing: {missing[:5]}{'...' if len(missing)>5 else ''}"
                )
            print(f"  matched {len(source_anchor_pos)}/{len(anchors)} anchors "
                  f"({len(missing)} missing in this variant)")

            # Umeyama Procrustes: solve scale s, rotation R_mat, translation t_vec.
            P = _np.asarray(source_anchor_pos)
            Q = _np.asarray(target_anchor_pos)
            cP = P.mean(axis=0)
            cQ = Q.mean(axis=0)
            Pc = P - cP
            Qc = Q - cQ
            H = Pc.T @ Qc
            U, S_sv, Vt = _np.linalg.svd(H)
            d_sign = _np.sign(_np.linalg.det(Vt.T @ U.T))
            D = _np.diag([1, 1, d_sign])
            R_mat = Vt.T @ D @ U.T
            var_P = float((Pc * Pc).sum())
            scale = float((_np.diag(D) * S_sv).sum() / var_P) if var_P > 0 else 1.0
            t_vec = cQ - scale * R_mat @ cP
            residuals = _np.linalg.norm((scale * (P @ R_mat.T) + t_vec) - Q, axis=1)
            print(f"  Procrustes: scale={scale:.4f} det(R)={_np.linalg.det(R_mat):.4f}")
            print(f"  anchor residuals: max={residuals.max():.4f}  "
                  f"mean={residuals.mean():.4f}  median={_np.median(residuals):.4f} m")

            # Per-view: apply Procrustes, build camera, attach metadata.
            from mathutils import Quaternion, Matrix as MM  # type: ignore
            R4 = MM(((R_mat[0,0], R_mat[0,1], R_mat[0,2], 0),
                     (R_mat[1,0], R_mat[1,1], R_mat[1,2], 0),
                     (R_mat[2,0], R_mat[2,1], R_mat[2,2], 0),
                     (0, 0, 0, 1)))
            _FLIP_OPENCV_TO_BLENDER = MM(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))

            for view in views:
                vname = view["name"]
                src_pos = _np.asarray(view["world_position"])
                src_quat = view["world_rotation_quat_wxyz"]  # w, x, y, z
                convention = view.get("convention", "opencv")
                tgt_pos = scale * (R_mat @ src_pos) + t_vec

                src_quat_obj = Quaternion(src_quat)
                src_rot_mat = src_quat_obj.to_matrix().to_4x4()
                tgt_rot_mat = R4 @ src_rot_mat
                if convention == "opencv":
                    tgt_rot_mat = tgt_rot_mat @ _FLIP_OPENCV_TO_BLENDER
                tgt_quat = tgt_rot_mat.to_quaternion()

                v_cam_data = bpy.data.cameras.new(name=vname)
                v_cam_data.lens = float(view.get("lens_mm", 50.0))
                v_cam_data.sensor_width = float(view.get("sensor_width_mm", 36.0))
                v_cam_data.sensor_height = float(view.get("sensor_height_mm", 24.0))
                v_cam_obj = bpy.data.objects.new(vname, v_cam_data)
                v_cam_obj.location = (float(tgt_pos[0]), float(tgt_pos[1]), float(tgt_pos[2]))
                v_cam_obj.rotation_mode = "QUATERNION"
                v_cam_obj.rotation_quaternion = (tgt_quat.w, tgt_quat.x, tgt_quat.y, tgt_quat.z)
                coll_virtual.objects.link(v_cam_obj)

                # Round-trip metadata as custom properties. sync_comparison_views.py
                # reads these via _read_custom_prop().
                purpose = view.get("purpose", "ab-comparison")
                v_cam_obj["view_purpose"] = purpose
                if "matches_reference_images" in view:
                    v_cam_obj["matches_reference_images"] = list(view["matches_reference_images"])
                if "render_resolution" in view:
                    v_cam_obj["render_resolution"] = list(view["render_resolution"])
                if "render_engine" in view:
                    v_cam_obj["render_engine"] = view["render_engine"]
                if "auto_localized" in view:
                    v_cam_obj["auto_localized"] = bool(view["auto_localized"])
                if "localization_method" in view:
                    v_cam_obj["localization_method"] = view["localization_method"]

                tag = "★" if vname == active_view_name else " "
                print(f"  {tag} '{vname}' [{purpose}] pos=({tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f}) lens={v_cam_data.lens:.0f}mm")

                if vname == active_view_name:
                    view_cam_obj = v_cam_obj
                    view_cam_blender_name = vname

        elif schema == 2:
            # Anchor-based registration. Match anchors by source-image basename
            # (same source frames across variants → byte-identical files →
            # reliable hash). Use whichever anchors are present in THIS
            # variant; skip missing ones.
            anchors = vc["anchor_frames"]
            source_anchor_pos = []
            target_anchor_pos = []
            target_basenames = [fp.rsplit("/", 1)[-1] for fp in filepaths]
            missing = []
            for a in anchors:
                bn = a["basename"]
                if bn not in target_basenames:
                    missing.append(bn)
                    continue
                target_idx = target_basenames.index(bn)
                tgt_obj = bpy.data.objects.get(f"cam_{target_idx+1:03d}")
                if tgt_obj is None:
                    continue
                source_anchor_pos.append(a["oriented_position"])
                target_anchor_pos.append(list(tgt_obj.location))
            if len(source_anchor_pos) < 3:
                raise SystemExit(
                    f"  ERROR: need ≥3 matching anchors; got {len(source_anchor_pos)}. "
                    f"Missing: {missing[:5]}{'...' if len(missing)>5 else ''}"
                )
            print(f"  matched {len(source_anchor_pos)}/{len(anchors)} anchors "
                  f"({len(missing)} missing in this variant)")

            # Umeyama (Procrustes with scale): find scale s, rotation R, translation t
            # such that s · R @ source_pts + t ≈ target_pts. Scale-aware because
            # different SfM runs can converge to different scale conventions.
            P = _np.asarray(source_anchor_pos)
            Q = _np.asarray(target_anchor_pos)
            cP = P.mean(axis=0)
            cQ = Q.mean(axis=0)
            Pc = P - cP
            Qc = Q - cQ
            H = Pc.T @ Qc
            U, S_sv, Vt = _np.linalg.svd(H)
            d_sign = _np.sign(_np.linalg.det(Vt.T @ U.T))
            D = _np.diag([1, 1, d_sign])
            R_mat = Vt.T @ D @ U.T
            var_P = float((Pc * Pc).sum())
            scale = float((_np.diag(D) * S_sv).sum() / var_P) if var_P > 0 else 1.0
            t_vec = cQ - scale * R_mat @ cP
            residuals = _np.linalg.norm((scale * (P @ R_mat.T) + t_vec) - Q, axis=1)
            print(f"  Procrustes: scale={scale:.4f} det(R)={_np.linalg.det(R_mat):.4f}")
            print(f"  anchor residuals: max={residuals.max():.4f}  "
                  f"mean={residuals.mean():.4f}  median={_np.median(residuals):.4f} m")

            # Apply transform to view_cam pose
            vcam_src = vc["view_camera_in_source_frame"]
            src_pos = _np.asarray(vcam_src["world_position"])
            src_quat = vcam_src["world_rotation_quat_wxyz"]  # w, x, y, z
            convention = vcam_src.get("convention", "opencv")
            tgt_pos = scale * (R_mat @ src_pos) + t_vec

            # Compose source rotation (from quat) with R for the new orientation.
            # JSON stores OpenCV convention (+X right, +Y down, +Z forward), so
            # after composing with R we flip back to Blender convention before
            # assigning to the Blender Camera object.
            from mathutils import Quaternion, Matrix as MM  # type: ignore
            src_quat_blender_or_opencv = Quaternion(src_quat)  # raw (w,x,y,z)
            src_rot_mat = src_quat_blender_or_opencv.to_matrix().to_4x4()
            R4 = MM(((R_mat[0,0], R_mat[0,1], R_mat[0,2], 0),
                     (R_mat[1,0], R_mat[1,1], R_mat[1,2], 0),
                     (R_mat[2,0], R_mat[2,1], R_mat[2,2], 0),
                     (0, 0, 0, 1)))
            tgt_rot_mat = R4 @ src_rot_mat
            if convention == "opencv":
                # OpenCV → Blender: rotate 180° around camera's local X axis
                _FLIP = MM(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
                tgt_rot_mat = tgt_rot_mat @ _FLIP
            tgt_quat = tgt_rot_mat.to_quaternion()

            vc_data = bpy.data.cameras.new(name=view_cam_blender_name)
            vc_data.lens = float(vcam_src.get("lens_mm", 50.0))
            vc_data.sensor_width = float(vcam_src.get("sensor_width_mm", 36.0))
            vc_data.sensor_height = float(vcam_src.get("sensor_height_mm", 24.0))
            view_cam_obj = bpy.data.objects.new(view_cam_blender_name, vc_data)
            view_cam_obj.location = (float(tgt_pos[0]), float(tgt_pos[1]), float(tgt_pos[2]))
            view_cam_obj.rotation_mode = "QUATERNION"
            view_cam_obj.rotation_quaternion = (tgt_quat.w, tgt_quat.x, tgt_quat.y, tgt_quat.z)
            coll_virtual.objects.link(view_cam_obj)
            print(f"  view_cam (anchor-aligned) pos=({tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f})  lens={vc_data.lens:.0f}mm")

        else:
            # Schema v1: legacy absolute coords. Same frame across variants
            # is NOT guaranteed; use only when you know all variants share the
            # same SfM frame (e.g., re-runs of the same data).
            vc_data = bpy.data.cameras.new(name=view_cam_blender_name)
            vc_data.lens = float(vc.get("lens_mm", 50.0))
            vc_data.sensor_width = float(vc.get("sensor_width_mm", 36.0))
            vc_data.sensor_height = float(vc.get("sensor_height_mm", 24.0))
            view_cam_obj = bpy.data.objects.new(view_cam_blender_name, vc_data)
            pos = vc["world_position"]
            quat = vc["world_rotation_quat_wxyz"]
            view_cam_obj.location = (pos[0], pos[1], pos[2])
            view_cam_obj.rotation_mode = "QUATERNION"
            view_cam_obj.rotation_quaternion = (quat[0], quat[1], quat[2], quat[3])
            coll_virtual.objects.link(view_cam_obj)
            print(f"  view_cam (schema v1, absolute) pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  lens={vc_data.lens:.0f}mm")

    # Set the active scene camera: prefer view_cam, fall back to cam_001
    if view_cam_obj:
        bpy.context.scene.camera = view_cam_obj
        print(f"  active scene camera: {view_cam_blender_name} (default view on .blend open)")
    elif bpy.data.objects.get("cam_001"):
        bpy.context.scene.camera = bpy.data.objects["cam_001"]

    # Add a default world light so the mesh is visible
    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.location = (5, -5, 10)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    coll_meshes.objects.link(sun_obj)

    # Set viewport so the user sees something useful when opening the .blend
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"Wrote {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

    # Optional: render the active camera's view to PNG for A/B comparison.
    # Uses Workbench engine for speed — geometry + vertex colors are visible
    # without scene-world setup. Override with --render-engine for a different
    # look (e.g., BLENDER_EEVEE_NEXT for proper PBR).
    render_output = args.get("render-output")
    if render_output:
        print(f"Rendering to {render_output}...")
        scn = bpy.context.scene
        scn.render.filepath = os.path.abspath(render_output)
        scn.render.image_settings.file_format = "PNG"
        scn.render.resolution_x = int(args.get("render-width", 1920))
        scn.render.resolution_y = int(args.get("render-height", 1080))
        scn.render.resolution_percentage = 100
        scn.render.engine = args.get("render-engine", "BLENDER_WORKBENCH")
        # Workbench needs explicit shading config to show vertex colors
        if scn.render.engine == "BLENDER_WORKBENCH":
            scn.display.shading.color_type = "VERTEX"
            scn.display.shading.light = "STUDIO"
        os.makedirs(os.path.dirname(scn.render.filepath), exist_ok=True)
        bpy.ops.render.render(write_still=True)
        print(f"  rendered ({scn.render.resolution_x}×{scn.render.resolution_y}, "
              f"engine={scn.render.engine}) → {scn.render.filepath}")

    print("DONE.")


if __name__ == "__main__":
    main()
