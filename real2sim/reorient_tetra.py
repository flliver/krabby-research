#!/usr/bin/env python3
"""reorient_tetra.py — regenerate legacy tetra mesh.ply from the RAW
binary-search PLY + the rep's CURRENT orientation (post-migration
gauge repair; operator-caught on 003/004).

Why: some runs' oriented_tetra.ply predates that run's final
re-orientation (stale gauge), and several runs only ever had the raw
unoriented tetra. The raw mesh + the rep's own oriented_cameras
(rotation, z_shift) is the truth.

    uv run --with open3d --with numpy --python 3.11 \
        python3 real2sim/reorient_tetra.py [--scene <s>] [--apply]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4


def rep_orientation(rep_dir: Path, scene_dir: Path):
    own = rep_dir / "origin-data" / "oriented" / "oriented_cameras.json"
    if own.exists():
        return own
    md = json.loads((rep_dir / "metadata.json").read_text())
    subset = md.get("resolved_inputs", {}).get("subset")
    base = scene_dir / "images" / "subsets" / str(subset) / "cameras"
    for c in sorted(base.glob("*/orient/*/oriented.json")) if base.is_dir() else []:
        return c
    return None


def main() -> int:
    apply = "--apply" in sys.argv
    scenes = [a for a in sys.argv[1:] if not a.startswith("--")]
    roots = [v4.STORE / s for s in scenes] if scenes else \
        sorted(d for d in v4.STORE.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    todo = []
    for sdir in roots:
        for tetra in sorted(sdir.glob("represent/matcha/*/meshify/tetra/*/")):
            raws = sorted(p for p in tetra.glob("tetra_mesh_binary_search_*.ply")
                          if not p.name.startswith("origin-dup"))
            if not raws:
                continue        # no raw source: keep existing mesh.ply (runoff-era, verified good)
            # never touch a mesh whose renders the operator verified in the
            # v2 runoffs (006/007/008 winners) — migrated renders mark those
            has_verified = any(
                json.loads(rm.read_text()).get("mechanism") in ("migrate", "migrate-repair")
                for rm in tetra.glob("renders/*/metadata.json"))
            if has_verified:
                continue
            ori = rep_orientation(tetra.parent.parent.parent, sdir)
            if ori is None:
                print(f"SKIP {tetra}: no orientation found")
                continue
            todo.append((sdir.name, tetra, raws[-1], ori))
    print(f"{len(todo)} tetra dirs to regenerate from raw")
    if not apply:
        for s, t, r, o in todo:
            print(f"  {s}: {t.relative_to(v4.STORE / s)} <- {r.name}")
        return 0
    import numpy as np
    import open3d as o3d
    for s, tetra, raw, ori in todo:
        meta = json.loads(ori.read_text())
        R = np.asarray(meta["rotation"], dtype=np.float64)
        z = float(meta["z_shift"])
        print(f"{s}: {raw.name} -> mesh.ply (gauge: {ori.relative_to(v4.STORE / s)})")
        mesh = o3d.io.read_triangle_mesh(str(raw))
        verts = np.asarray(mesh.vertices)
        mesh.vertices = o3d.utility.Vector3dVector(verts @ R.T + np.array([0.0, 0.0, z]))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(tetra / "mesh.ply"), mesh)
        md_file = tetra / "metadata.json"
        md = json.loads(md_file.read_text())
        md["mesh_regenerated"] = {"from": raw.name, "orientation": str(ori),
                                  "reason": "stale/missing oriented tetra (003/004 operator catch)"}
        md_file.write_text(json.dumps(md, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
