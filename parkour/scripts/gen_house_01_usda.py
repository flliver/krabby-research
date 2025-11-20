#!/usr/bin/env python3
"""
Programmatically generates the house + environment USD scene equivalent to `house_model_reference.usd`.

Contents:
  /House (walls, roof, door, windows, materials)
  /Site  (ground, walkway, trees, hedge, materials)
  /Fence (panels + gate)

Uses Usd, UsdGeom, UsdShade APIs to define meshes and bind materials with MaterialBindingAPI.

Example:
    python3 gen_house_01_usda.py --output ../assets/scenes/house_01_generated.usd --overwrite

Notes:
  Dimensions approximate meters. Fence encloses 14m x 14m area. Ground plane is 40m square.
"""
from __future__ import annotations

import argparse
import os
from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Gf, Sdf


def create_material(stage: Usd.Stage, path: str, diffuse=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, opacity=None):
    """Create a UsdPreviewSurface material at path and return the material prim."""
    mat = UsdShade.Material.Define(stage, path)
    # Use 'PreviewSurface' to match authored USD shader prim naming
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_attr = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_attr.Set(Gf.Vec3f(*diffuse))
    rough_attr = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    rough_attr.Set(roughness)
    metal_attr = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    metal_attr.Set(metallic)
    if opacity is not None:
        op_attr = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        op_attr.Set(opacity)
    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat_surface = mat.CreateSurfaceOutput()
    mat_surface.ConnectToSource(surface_output)
    return mat


def bind_material(prim: Usd.Prim, material: UsdShade.Material):
    """Bind a material to a prim using MaterialBindingAPI.

    Parameters:
      prim: Geometry or Xform prim to receive the binding.
      material: The `UsdShade.Material` to bind.
    Returns: None."""
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_mesh(stage: Usd.Stage, path: str, points, face_counts, face_indices, uvs=None, double_sided=True):
    """Create a Mesh prim, author topology, optional UVs, and double-sided flag.

    Parameters:
      stage: Target stage.
      path: Prim path for the mesh.
      points: Sequence of (x, y, z) floats.
      face_counts: Per-face vertex counts.
      face_indices: Flattened indices referencing `points`.
      uvs: Optional sequence of (u, v) tuples.
      double_sided: If True, set `doubleSided` attr for viewport convenience.
    Returns: `UsdGeom.Mesh` instance."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    if uvs:
        # Use PrimvarsAPI for texture coordinates
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(uvs)
    if double_sided:
        mesh.CreateDoubleSidedAttr(True)
    return mesh


def add_house(stage: Usd.Stage):
    """Author the `/House` hierarchy (walls, roof, door, windows + materials).

    Parameters:
      stage: Target stage.
    Returns: Root Xform prim for the house."""
    house = UsdGeom.Xform.Define(stage, "/House")
    # Explicit sub-Xforms to match authored structure
    walls_xf = UsdGeom.Xform.Define(stage, "/House/Walls")
    roof_xf = UsdGeom.Xform.Define(stage, "/House/Roof")
    # Materials
    wall_mat = create_material(stage, "/House/Materials/WallMaterial", diffuse=(0.75, 0.75, 0.82), roughness=0.5)
    roof_mat = create_material(stage, "/House/Materials/RoofMaterial", diffuse=(0.6, 0.1, 0.1), roughness=0.4)
    glass_mat = create_material(stage, "/House/Materials/GlassMaterial", diffuse=(0.3, 0.4, 0.6), roughness=0.1, opacity=0.35)
    door_mat = create_material(stage, "/House/Materials/DoorMaterial", diffuse=(0.35, 0.2, 0.05), roughness=0.6)

    # Front wall segments (simplified representation of openings as separate quads)
    wall_front = create_mesh(
        stage,
        "/House/Walls/Wall_Front",
        points=[
            (-2, 3, 0), (2, 3, 0), (2, 3, 3), (-2, 3, 3),  # main quad
            (-0.5, 3, 0), (0.5, 3, 0), (0.5, 3, 2), (-0.5, 3, 2),  # door opening quad
            (-1.8, 3, 1.2), (-0.6, 3, 1.2), (-0.6, 3, 2.2), (-1.8, 3, 2.2),  # window opening quad
        ],
        face_counts=[4, 4, 4],
        face_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        uvs=[
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
            (0, 0), (1, 0), (1, 1), (0, 1),
        ],
    )
    bind_material(wall_front.GetPrim(), wall_mat)

    wall_back = create_mesh(
        stage,
        "/House/Walls/Wall_Back",
        points=[(-2, -3, 0), (2, -3, 0), (2, -3, 3), (-2, -3, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(wall_back.GetPrim(), wall_mat)

    wall_left = create_mesh(
        stage,
        "/House/Walls/Wall_Left",
        points=[
            (-2, -3, 0), (-2, 3, 0), (-2, 3, 3), (-2, -3, 3),
            (-2, -1, 1), (-2, 1, 1), (-2, 1, 2), (-2, -1, 2),
        ],
        face_counts=[4, 4],
        face_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(wall_left.GetPrim(), wall_mat)

    wall_right = create_mesh(
        stage,
        "/House/Walls/Wall_Right",
        points=[(2, -3, 0), (2, 3, 0), (2, 3, 3), (2, -3, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(wall_right.GetPrim(), wall_mat)

    # Roof
    roof = create_mesh(
        stage,
        "/House/Roof/RoofMesh",
        points=[(-2, 3, 3), (2, 3, 3), (-2, -3, 3), (2, -3, 3), (0, 0, 4)],
        face_counts=[3, 3, 3, 3],
        face_indices=[0, 1, 4, 1, 3, 4, 3, 2, 4, 2, 0, 4],
        uvs=[
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1),
        ],
    )
    bind_material(roof.GetPrim(), roof_mat)

    # Door
    door = create_mesh(
        stage,
        "/House/Door",
        points=[(-0.5, 3.01, 0), (0.5, 3.01, 0), (0.5, 3.01, 2), (-0.5, 3.01, 2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(door.GetPrim(), door_mat)

    # Windows
    win_front = create_mesh(
        stage,
        "/House/Window_Front",
        points=[(-1.8, 3.01, 1.2), (-0.6, 3.01, 1.2), (-0.6, 3.01, 2.2), (-1.8, 3.01, 2.2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(win_front.GetPrim(), glass_mat)

    win_left = create_mesh(
        stage,
        "/House/Window_Left",
        points=[(-2.01, -1, 1), (-2.01, 1, 1), (-2.01, 1, 2), (-2.01, -1, 2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(win_left.GetPrim(), glass_mat)

    return house


def add_site(stage: Usd.Stage):
    """Author the `/Site` environment (ground, walkway, trees, hedge, fence).

    Creates and binds materials, places simple geometric proxies for foliage
    and structural elements to approximate a small property footprint.

    Parameters:
      stage: Target stage.
    Returns: Root Xform prim for the site."""
    site = UsdGeom.Xform.Define(stage, "/Site")
    # Site materials: use standard create_material; verifier tolerates shader name differences.
    def create_site_material(path, diffuse, roughness, metallic=0.0):
        return create_material(stage, path, diffuse=diffuse, roughness=roughness, metallic=metallic)

    grass_mat = create_site_material("/Site/Materials/GrassMaterial", (0.10, 0.50, 0.10), 0.9)
    concrete_mat = create_site_material("/Site/Materials/ConcreteMaterial", (0.55, 0.55, 0.55), 0.6)
    trunk_mat = create_site_material("/Site/Materials/TrunkMaterial", (0.35, 0.20, 0.05), 0.8)
    foliage_mat = create_site_material("/Site/Materials/FoliageMaterial", (0.05, 0.35, 0.12), 0.9)
    fence_mat = create_site_material("/Site/Materials/FenceMaterial", (0.55, 0.45, 0.35), 0.7)

    ground = create_mesh(
        stage,
        "/Site/Ground",
        points=[(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (4, 0), (4, 4), (0, 4)],
    )
    bind_material(ground.GetPrim(), grass_mat)

    walkway = create_mesh(
        stage,
        "/Site/Walkway",
        points=[(-0.75, 3.0, 0.01), (0.75, 3.0, 0.01), (0.75, 9.0, 0.01), (-0.75, 9.0, 0.01)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 3), (0, 3)],
    )
    bind_material(walkway.GetPrim(), concrete_mat)

    # Trees
    tree1 = UsdGeom.Xform.Define(stage, "/Site/Tree_1")
    tree1.AddTranslateOp().Set(Gf.Vec3d(-5, 5, 0))
    trunk1 = UsdGeom.Capsule.Define(stage, "/Site/Tree_1/Trunk")
    trunk1.CreateRadiusAttr(0.15)
    trunk1.CreateHeightAttr(2.0)
    bind_material(trunk1.GetPrim(), trunk_mat)
    foliage1 = UsdGeom.Cone.Define(stage, "/Site/Tree_1/Foliage")
    foliage1.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.2))
    foliage1.CreateHeightAttr(2.5)
    foliage1.CreateRadiusAttr(1.5)
    bind_material(foliage1.GetPrim(), foliage_mat)

    tree2 = UsdGeom.Xform.Define(stage, "/Site/Tree_2")
    tree2.AddTranslateOp().Set(Gf.Vec3d(6, -4, 0))
    trunk2 = UsdGeom.Capsule.Define(stage, "/Site/Tree_2/Trunk")
    trunk2.CreateRadiusAttr(0.18)
    trunk2.CreateHeightAttr(2.2)
    bind_material(trunk2.GetPrim(), trunk_mat)
    foliage2 = UsdGeom.Cone.Define(stage, "/Site/Tree_2/Foliage")
    foliage2.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2.4))
    foliage2.CreateHeightAttr(2.8)
    foliage2.CreateRadiusAttr(1.6)
    bind_material(foliage2.GetPrim(), foliage_mat)

    # Hedge (3D box)
    hedge = create_mesh(
        stage,
        "/Site/FrontHedge",
        points=[
            (-3.5, 2.35, 0), (3.5, 2.35, 0), (3.5, 2.35, 0.6), (-3.5, 2.35, 0.6),
            (-3.5, 2.65, 0), (3.5, 2.65, 0), (3.5, 2.65, 0.6), (-3.5, 2.65, 0.6),
        ],
        face_counts=[4, 4, 4, 4, 4, 4],
        face_indices=[
            0, 1, 2, 3,  # front
            4, 5, 6, 7,  # back
            3, 2, 6, 7,  # top
            0, 1, 5, 4,  # bottom
            0, 4, 7, 3,  # left
            1, 5, 6, 2,  # right
        ],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1)],
    )
    bind_material(hedge.GetPrim(), foliage_mat)

    # Fence panels + gate
    # Fence root Xform
    fence_root = UsdGeom.Xform.Define(stage, "/Fence")

    def fence_panel(name, pts, uvs=None):
        mesh = create_mesh(stage, f"/Fence/{name}", points=pts, face_counts=[4], face_indices=[0, 1, 2, 3], uvs=uvs or [(0, 0), (1, 0), (1, 1), (0, 1)])
        bind_material(mesh.GetPrim(), fence_mat)

    fence_panel("Fence_South", [(-7, -7, 0), (7, -7, 0), (7, -7, 1.2), (-7, -7, 1.2)])
    fence_panel("Fence_North_Left", [(-7, 7, 0), (-0.9, 7, 0), (-0.9, 7, 1.2), (-7, 7, 1.2)], uvs=[(0, 0), (0.8, 0), (0.8, 1), (0, 1)])
    fence_panel("Fence_North_Right", [(0.9, 7, 0), (7, 7, 0), (7, 7, 1.2), (0.9, 7, 1.2)], uvs=[(0, 0), (0.8, 0), (0.8, 1), (0, 1)])
    fence_panel("Fence_East", [(7, -7, 0), (7, 7, 0), (7, 7, 1.2), (7, -7, 1.2)])
    fence_panel("Fence_West", [(-7, -7, 0), (-7, 7, 0), (-7, 7, 1.2), (-7, -7, 1.2)])
    fence_panel("Gate", [(-0.9, 7.05, 0), (0.9, 7.05, 0), (0.9, 7.05, 1.2), (-0.9, 7.05, 1.2)])

    return site


def build_stage(output_path: str):
    """Create a new stage, set metadata, add all scene components, and save.

    Parameters:
      output_path: Destination USD/USDA file path (overwritten if exists).
    Returns: Populated `Usd.Stage` (already saved)."""
    stage = Usd.Stage.CreateNew(output_path)
    # Ensure the scene uses Z-up so the ground (XY plane at Z=0) aligns with the viewer grid
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # Optional but common: set meters per unit to 1.0 (meters)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(0)
    add_house(stage)
    add_site(stage)
    # Set default prim for convenience
    stage.SetDefaultPrim(stage.GetPrimAtPath("/House"))
    stage.Save()
    return stage


def main():
    """CLI entrypoint for generating the house + environment USD scene.

    Parses command line arguments to determine output path and format:
        --output     Target file name (.usd or .usda). If --ascii is supplied or
                                    the name ends with .usda, an ASCII layer is written.
        --overwrite  Allow replacing an existing file (otherwise aborts safely).
        --ascii      Force ASCII USDA output (overrides provided extension).

    Steps:
        1. Parse arguments and resolve final output path (respecting --ascii).
        2. Guard against accidental overwrite unless --overwrite given.
        3. Build stage (geometry, materials, bindings, metadata, up-axis).
        4. Save stage and emit a status message with format (ASCII/Binary).

    Returns:
        int exit status code:
            0 = success
            1 = refused overwrite
            2 = generation failure (file not found after build)
    """
    parser = argparse.ArgumentParser(description="Generate house + environment USD scene")
    parser.add_argument("--output", default="../assets/scenes/house_01_generated.usd", help="Output USD filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if file exists")
    parser.add_argument("--ascii", action="store_true", help="Write ASCII USDA instead of binary USD (forces .usda extension)")
    args = parser.parse_args()

    # Decide on ASCII vs binary based on flag or extension
    requested_path = args.output
    is_ascii = args.ascii or requested_path.lower().endswith(".usda")
    if is_ascii and not requested_path.lower().endswith(".usda"):
        # Force .usda extension
        base, _ext = os.path.splitext(requested_path)
        output_path = base + ".usda"
    else:
        output_path = requested_path

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Refusing to overwrite existing file: {output_path}. Use --overwrite to force.")
        return 1

    build_stage(output_path)
    if os.path.exists(output_path):
        fmt = "ASCII" if is_ascii else "Binary"
        print(f"Generated {fmt} USD: {os.path.abspath(output_path)}")
        return 0
    print("Failed to generate USD file.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())