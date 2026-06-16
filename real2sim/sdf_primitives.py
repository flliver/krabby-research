"""STO-SCN-145 / GOAL-SCN-001 — boolean cull primitives as signed-distance functions (SDFs).

Operators author cull regions as primitives **in meters in the datum frame** (STO-SCN-016 scale +
datum_frame.py); booleans compose via min/max (Inigo Quilez catalog). A vertex is kept iff it lies
inside the resulting solid (sdf <= 0). This is masking (keep/drop verts) — robust to non-manifold
meshes, never cuts geometry. The STO-SCN-137 camera-AABB is the box-SDF special case.

Primitive spec (JSON, authored in the datum frame, meters):
  {"frame_transform": <optional 4x4 mesh->datum>, "primitives": [
     {"type":"sphere",    "op":"keep|subtract", "center":[x,y,z], "radius":r},
     {"type":"box",       "op":..., "center":[x,y,z], "half":[hx,hy,hz]},
     {"type":"cylinder",  "op":..., "center":[x,y,z], "radius":r, "height":h, "axis":"z"},
     {"type":"halfspace", "op":..., "point":[x,y,z], "normal":[nx,ny,nz]}   # inside = below the plane
  ]}
Combined solid = (union of all `keep` prims) minus (union of all `subtract` prims). With no `keep`
prims the keep-region is all space (subtract-only). Pure numpy.
"""
from __future__ import annotations

import numpy as np

_BIG = 1e18


def sdf_sphere(p, center, radius):
    return np.linalg.norm(p - np.asarray(center, float), axis=-1) - float(radius)


def sdf_box(p, center, half):
    q = np.abs(p - np.asarray(center, float)) - np.asarray(half, float)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_cylinder(p, center, radius, height, axis="z"):
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    d = p - np.asarray(center, float)
    along = d[..., ax]
    radial_axes = [i for i in range(3) if i != ax]
    radial = np.linalg.norm(d[..., radial_axes], axis=-1) - float(radius)
    cap = np.abs(along) - 0.5 * float(height)
    outside = np.linalg.norm(np.maximum(np.stack([radial, cap], axis=-1), 0.0), axis=-1)
    inside = np.minimum(np.maximum(radial, cap), 0.0)
    return outside + inside


def sdf_halfspace(p, point, normal):
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) or 1.0)
    return (p - np.asarray(point, float)) @ n     # inside (kept) = negative = below the plane


def _sdf_one(prim, p):
    t = prim["type"]
    if t == "sphere":
        return sdf_sphere(p, prim["center"], prim["radius"])
    if t == "box":
        return sdf_box(p, prim["center"], prim["half"])
    if t == "cylinder":
        return sdf_cylinder(p, prim["center"], prim["radius"], prim["height"], prim.get("axis", "z"))
    if t == "halfspace":
        return sdf_halfspace(p, prim["point"], prim["normal"])
    raise ValueError(f"unknown primitive type: {t}")


def evaluate(primitives, p):
    """Combined SDF of the primitive list at points p (..,3). <=0 inside the kept solid."""
    p = np.asarray(p, float)
    n = p.shape[0] if p.ndim > 1 else 1
    keep = np.full(n, _BIG)         # union of keeps (min); starts +BIG so the min accumulates
    sub = np.full(n, _BIG)          # union of subtracts (min); +BIG => nothing subtracted
    any_keep = False
    for prim in primitives:
        s = _sdf_one(prim, p)
        if prim.get("op", "keep") == "subtract":
            sub = np.minimum(sub, s)
        else:
            keep = np.minimum(keep, s); any_keep = True
    if not any_keep:
        keep = np.full(n, -_BIG)     # subtract-only: keep all space, then carve
    return np.maximum(keep, -sub)    # intersect keep-region with complement of subtracts


def cull_mask(verts, primitives, frame_transform=None):
    """Keep mask for verts (N,3): True iff inside the primitive solid.

    `frame_transform` (4x4) maps verts INTO the primitive/datum frame before evaluation; None =
    primitives already in the verts' frame.
    """
    v = np.asarray(verts, float)
    if frame_transform is not None:
        T = np.asarray(frame_transform, float)
        v = v @ T[:3, :3].T + T[:3, 3]
    if not primitives:
        return np.ones(len(v), dtype=bool)
    return evaluate(primitives, v) <= 0.0


def load_spec(spec):
    """Normalize a primitives spec dict: returns (primitives_list, frame_transform_or_None)."""
    if isinstance(spec, list):
        return spec, None
    return spec.get("primitives", []), spec.get("frame_transform")
