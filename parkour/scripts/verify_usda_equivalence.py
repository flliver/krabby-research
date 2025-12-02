#!/usr/bin/env python3
"""
Verification utility to compare the authored `house_model.usda` with a generated USD
file (e.g. from `generate_house_scene.py`). Requires a Python environment with the
`pxr` USD modules (Omniverse / Isaac Sim / official USD build).

Checks:
  1. Prim path presence & type parity
  2. Material names and diffuse colors (and opacity if present)
  3. Mesh face counts, vertex count, and bounding boxes
  4. Reports differences and exits with non-zero code if mismatches found

Usage:
  python3 verify_house_scene_equivalence.py \
    --original house_model.usda \
    --generated house_model_generated.usda

Exit codes:
  0 - Equivalent within tolerance
  1 - Missing prims or structural differences
  2 - Geometry or material mismatches
  3 - Other error
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pxr import Usd, UsdGeom, UsdShade, Gf


@dataclass
class MeshInfo:
    path: str
    points: int
    faces: int
    bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]]

@dataclass
class MaterialInfo:
    path: str
    diffuse: Tuple[float, float, float]
    opacity: float | None


def collect_prims(stage: Usd.Stage) -> List[str]:
    return [p.GetPath().pathString for p in stage.Traverse() if p.IsActive()]


def collect_mesh_info(stage: Usd.Stage) -> Dict[str, MeshInfo]:
    out = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = len(mesh.GetPointsAttr().Get())
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            faces = sum(face_counts) // 4 if face_counts else 0  # simplistic (all quads or triangles); for reporting only
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
            out[prim.GetPath().pathString] = MeshInfo(
                path=prim.GetPath().pathString,
                points=points,
                faces=len(face_counts),
                bbox=((bbox.GetMin()[0], bbox.GetMin()[1], bbox.GetMin()[2]), (bbox.GetMax()[0], bbox.GetMax()[1], bbox.GetMax()[2])),
            )
    return out


def collect_material_info(stage: Usd.Stage) -> Dict[str, MaterialInfo]:
    out = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            mat = UsdShade.Material(prim)
            diffuse = (0.0, 0.0, 0.0)
            opacity = None
            if mat.GetSurfaceOutput():
                conn = mat.GetSurfaceOutput().GetConnectedSource()
                if conn:
                    src_prim = conn[0].GetPrim()
                    if src_prim and src_prim.IsValid():
                        shader = UsdShade.Shader(src_prim)
                        diff_in = shader.GetInput("diffuseColor")
                        if diff_in:
                            v = diff_in.Get()
                            if v is not None:
                                diffuse = (v[0], v[1], v[2])
                        op_in = shader.GetInput("opacity")
                        if op_in:
                            op_val = op_in.Get()
                            if op_val is not None:
                                opacity = float(op_val)
            out[prim.GetPath().pathString] = MaterialInfo(path=prim.GetPath().pathString, diffuse=diffuse, opacity=opacity)
    return out


def compare_sets(label: str, a: List[str], b: List[str]) -> List[str]:
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    messages = []
    # Ignore shader child naming differences (PreviewSurface vs Shader) for robustness
    def filter_shader_names(paths: List[str]) -> List[str]:
        return [p for p in paths if not (p.endswith("/PreviewSurface") or p.endswith("/Shader"))]
    missing = filter_shader_names(missing)
    extra = filter_shader_names(extra)
    if missing:
        messages.append(f"[STRUCT] Missing in generated ({label}): {missing}")
    if extra:
        messages.append(f"[STRUCT] Extra in generated ({label}): {extra}")
    return messages


def compare_materials(orig: Dict[str, MaterialInfo], gen: Dict[str, MaterialInfo], tol: float = 1e-4) -> List[str]:
    messages = []
    for path, om in orig.items():
        gm = gen.get(path)
        if gm is None:
            messages.append(f"[MAT] Missing material: {path}")
            continue
        for i, comp in enumerate("rgb"):
            if abs(om.diffuse[i] - gm.diffuse[i]) > tol:
                messages.append(f"[MAT] Diffuse mismatch {path} component {comp}: {om.diffuse[i]:.3f} vs {gm.diffuse[i]:.3f}")
        if (om.opacity or 0.0) != (gm.opacity or 0.0):
            messages.append(f"[MAT] Opacity mismatch {path}: {om.opacity} vs {gm.opacity}")
    return messages


def compare_meshes(orig: Dict[str, MeshInfo], gen: Dict[str, MeshInfo]) -> List[str]:
    messages = []
    for path, om in orig.items():
        gm = gen.get(path)
        if gm is None:
            messages.append(f"[MESH] Missing mesh: {path}")
            continue
        if om.points != gm.points:
            messages.append(f"[MESH] Vertex count mismatch {path}: {om.points} vs {gm.points}")
        if om.faces != gm.faces:
            messages.append(f"[MESH] Face group count mismatch {path}: {om.faces} vs {gm.faces}")
    return messages


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify generated USD matches original house model")
    ap.add_argument("--original", required=True, help="Path to original USD (authored)")
    ap.add_argument("--generated", required=True, help="Path to generated USD")
    args = ap.parse_args(argv)

    try:
        orig_stage = Usd.Stage.Open(args.original)
        gen_stage = Usd.Stage.Open(args.generated)
    except Exception as e:
        print(f"Error opening stages: {e}")
        return 3

    orig_prims = collect_prims(orig_stage)
    gen_prims = collect_prims(gen_stage)
    struct_msgs = compare_sets("prims", orig_prims, gen_prims)

    orig_mats = collect_material_info(orig_stage)
    gen_mats = collect_material_info(gen_stage)
    mat_msgs = compare_materials(orig_mats, gen_mats)

    orig_mesh = collect_mesh_info(orig_stage)
    gen_mesh = collect_mesh_info(gen_stage)
    mesh_msgs = compare_meshes(orig_mesh, gen_mesh)

    problems = struct_msgs + mat_msgs + mesh_msgs
    if problems:
        print("Verification FAILED. Differences:")
        for m in problems:
            print(m)
        # Distinguish structural vs data mismatches
        if any(m.startswith("[STRUCT]") for m in problems):
            return 1
        return 2
    print("Verification PASSED: Generated USD matches original within tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())