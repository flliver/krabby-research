#!/usr/bin/env python3
"""
Programmatically generates the Wyatt House USD scene - a simple house asset.

This script creates a basic house structure with:
  - Four walls forming a 5m x 4m footprint
  - A pyramidal roof
  - A front door
  - Two side windows
  - Basic materials for walls, roof, door, and windows

The house is 3m tall with a roof peak at 4.5m.

Usage:
    python3 gen_wyatt_house.py --output ../assets/scenes/wyattHouse_generated.usda --overwrite

Notes:
  - Dimensions are in meters
  - Z-up coordinate system
  - ASCII USDA format by default
"""
from __future__ import annotations

import argparse
import os
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf


def create_material(stage: Usd.Stage, path: str, diffuse=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, opacity=None):
    """Create a UsdPreviewSurface material at the specified path.
    
    Parameters:
        stage: USD stage to create the material in
        path: Path for the material prim
        diffuse: RGB color tuple for diffuse color
        roughness: Material roughness (0.0 to 1.0)
        metallic: Material metallic value (0.0 to 1.0)
        opacity: Optional opacity value (0.0 to 1.0)
    
    Returns:
        UsdShade.Material: The created material
    """
    mat = UsdShade.Material.Define(stage, path)
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
        prim: The prim to bind the material to
        material: The material to bind
    """
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_mesh(stage: Usd.Stage, path: str, points, face_counts, face_indices, uvs=None, double_sided=True):
    """Create a mesh prim with the specified geometry.
    
    Parameters:
        stage: USD stage to create the mesh in
        path: Path for the mesh prim
        points: List of (x, y, z) point tuples
        face_counts: List of vertex counts per face
        face_indices: Flattened list of indices into points array
        uvs: Optional list of (u, v) texture coordinate tuples
        double_sided: Whether the mesh should be double-sided
    
    Returns:
        UsdGeom.Mesh: The created mesh
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    
    if uvs:
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(uvs)
    
    if double_sided:
        mesh.CreateDoubleSidedAttr(True)
    
    return mesh


def create_wyatt_house(stage: Usd.Stage):
    """Create the Wyatt House structure with all components.
    
    Creates a complete house with walls, roof, door, windows, and materials.
    
    Parameters:
        stage: USD stage to create the house in
    
    Returns:
        UsdGeom.Xform: The root transform for the house
    """
    # Root transform
    house = UsdGeom.Xform.Define(stage, "/WyattHouse")
    
    # Create materials
    wall_mat = create_material(
        stage, "/WyattHouse/Materials/WallMaterial",
        diffuse=(0.85, 0.85, 0.90), roughness=0.6
    )
    roof_mat = create_material(
        stage, "/WyattHouse/Materials/RoofMaterial",
        diffuse=(0.4, 0.2, 0.15), roughness=0.5
    )
    door_mat = create_material(
        stage, "/WyattHouse/Materials/DoorMaterial",
        diffuse=(0.5, 0.3, 0.1), roughness=0.7
    )
    window_mat = create_material(
        stage, "/WyattHouse/Materials/WindowMaterial",
        diffuse=(0.2, 0.3, 0.5), roughness=0.1, opacity=0.4
    )
    
    # Create walls container
    walls_xf = UsdGeom.Xform.Define(stage, "/WyattHouse/Walls")
    
    # Front Wall
    front_wall = create_mesh(
        stage,
        "/WyattHouse/Walls/FrontWall",
        points=[(-2.5, 2, 0), (2.5, 2, 0), (2.5, 2, 3), (-2.5, 2, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(front_wall.GetPrim(), wall_mat)
    
    # Back Wall
    back_wall = create_mesh(
        stage,
        "/WyattHouse/Walls/BackWall",
        points=[(-2.5, -2, 0), (2.5, -2, 0), (2.5, -2, 3), (-2.5, -2, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(back_wall.GetPrim(), wall_mat)
    
    # Left Wall
    left_wall = create_mesh(
        stage,
        "/WyattHouse/Walls/LeftWall",
        points=[(-2.5, -2, 0), (-2.5, 2, 0), (-2.5, 2, 3), (-2.5, -2, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(left_wall.GetPrim(), wall_mat)
    
    # Right Wall
    right_wall = create_mesh(
        stage,
        "/WyattHouse/Walls/RightWall",
        points=[(2.5, -2, 0), (2.5, 2, 0), (2.5, 2, 3), (2.5, -2, 3)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(right_wall.GetPrim(), wall_mat)
    
    # Roof (pyramid)
    roof = create_mesh(
        stage,
        "/WyattHouse/Roof",
        points=[
            (-2.5, 2, 3), (2.5, 2, 3),
            (2.5, -2, 3), (-2.5, -2, 3),
            (0, 0, 4.5)
        ],
        face_counts=[3, 3, 3, 3],
        face_indices=[
            0, 1, 4,
            1, 2, 4,
            2, 3, 4,
            3, 0, 4
        ],
        uvs=[
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1),
            (0, 0), (1, 0), (0.5, 1)
        ]
    )
    bind_material(roof.GetPrim(), roof_mat)
    
    # Door
    door = create_mesh(
        stage,
        "/WyattHouse/Door",
        points=[(-0.6, 2.01, 0), (0.6, 2.01, 0), (0.6, 2.01, 2.2), (-0.6, 2.01, 2.2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(door.GetPrim(), door_mat)
    
    # Left Window
    window_left = create_mesh(
        stage,
        "/WyattHouse/WindowLeft",
        points=[(-2.51, -0.5, 1.2), (-2.51, 0.5, 1.2), (-2.51, 0.5, 2.2), (-2.51, -0.5, 2.2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(window_left.GetPrim(), window_mat)
    
    # Right Window
    window_right = create_mesh(
        stage,
        "/WyattHouse/WindowRight",
        points=[(2.51, -0.5, 1.2), (2.51, 0.5, 1.2), (2.51, 0.5, 2.2), (2.51, -0.5, 2.2)],
        face_counts=[4],
        face_indices=[0, 1, 2, 3],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    bind_material(window_right.GetPrim(), window_mat)
    
    return house


def build_stage(output_path: str):
    """Create a new stage with the Wyatt House and save it.
    
    Parameters:
        output_path: Path to save the USD file
    
    Returns:
        Usd.Stage: The created and saved stage
    """
    stage = Usd.Stage.CreateNew(output_path)
    
    # Set stage metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(0)
    
    # Create the house
    create_wyatt_house(stage)
    
    # Set default prim
    stage.SetDefaultPrim(stage.GetPrimAtPath("/WyattHouse"))
    
    # Save the stage
    stage.Save()
    
    return stage


def main():
    """CLI entrypoint for generating the Wyatt House USD scene.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Generate Wyatt House - a simple house asset in USD format"
    )
    parser.add_argument(
        "--output",
        default="../assets/scenes/wyattHouse_generated.usda",
        help="Output USD filename"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if file exists"
    )
    args = parser.parse_args()
    
    output_path = args.output
    
    # Check if file exists and we're not allowed to overwrite
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Error: File {output_path} already exists. Use --overwrite to replace it.")
        return 1
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate the house
    try:
        build_stage(output_path)
        print(f"Successfully generated Wyatt House: {os.path.abspath(output_path)}")
        return 0
    except Exception as e:
        print(f"Error generating Wyatt House: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
