#!/usr/bin/env python3
"""Patch 2d-gaussian-splatting/render_multires.py: chunk the multires-merge
vertex visibility test.

Upstream computes one monolithic (n_cameras, n_verts, ...) projection
tensor to decide which lower-resolution faces to drop. At mesh_res 1024 a
real scene produces ~43M raw vertices; with 17 cameras the projection is a
single >3 GiB allocation on top of ~13 GiB already resident — OOM on every
16 GB GPU in the fleet (first hit: 013-basement, 2026-06-10, RTX 4080).

Fix: iterate the test over 2M-vertex slices. Semantically identical — the
per-vertex `any(dim=0)` reduction is order-independent — and transient
memory is bounded by the slice size (~0.5 GiB). Validated 2026-06-10 on
013-basement via runtime bind-mount overlay on sbeeprz: 43.3M raw /
30.8M post vertices, mesh byte-identical semantics, no OOM.

Story: STO-SCN-053 (EPI-SCN-DOCKER).
"""
from pathlib import Path

TARGET = Path("/opt/MAtCha/2d-gaussian-splatting/render_multires.py")

OLD = """                # Check which vertices are in the field of view...
                projections = cameras_wrapper.project_points(verts.view(1, -1, 3))  # (n_cameras, n_verts, 2)
                height, width = cameras_wrapper.gs_cameras[0].image_height, cameras_wrapper.gs_cameras[0].image_width
                factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=projections.device)  # (1, 1, 2)
                projections = projections / factors  # (n_cameras, n_verts, 2)
                visible_mask = (projections[..., 0] > -1.0) & (projections[..., 0] < 1.0) & (projections[..., 1] > -1.0) & (projections[..., 1] < 1.0)  # (n_cameras, n_verts)
                
                # ... and which are close to the camera
                depths = cameras_wrapper.transform_points_world_to_view(verts.view(1, -1, 3))[..., 2]  # (n_cameras, n_verts)
                close_verts = (depths < depth_truncs[i_mesh - 1])
                
                non_valid_verts = (visible_mask & close_verts).any(dim=0)  # (n_verts)"""

NEW = """                # Check which vertices are in the field of view...
                # [krabby patch: multires_oom] chunked over verts: the
                # monolithic (n_cameras, n_verts, ...) projection OOMs 16GB
                # GPUs on mesh_res-1024 meshes (~43M verts x 17 cams).
                # Semantically identical; transient memory bounded by the
                # chunk size. See patch_matcha_multires_oom.py.
                height, width = cameras_wrapper.gs_cameras[0].image_height, cameras_wrapper.gs_cameras[0].image_width
                factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=verts.device)  # (1, 1, 2)
                _chunk = 2_000_000
                _nv_parts = []
                for _lo in range(0, verts.shape[0], _chunk):
                    _v = verts[_lo:_lo + _chunk].view(1, -1, 3)
                    projections = cameras_wrapper.project_points(_v) / factors  # (n_cameras, chunk, 2)
                    visible_mask = (projections[..., 0] > -1.0) & (projections[..., 0] < 1.0) & (projections[..., 1] > -1.0) & (projections[..., 1] < 1.0)
                    depths = cameras_wrapper.transform_points_world_to_view(_v)[..., 2]  # (n_cameras, chunk)
                    close_verts = (depths < depth_truncs[i_mesh - 1])
                    _nv_parts.append((visible_mask & close_verts).any(dim=0))
                non_valid_verts = torch.cat(_nv_parts)  # (n_verts)"""


def main() -> None:
    text = TARGET.read_text()
    if "[krabby patch: multires_oom]" in text:
        print("already patched — no-op")
        return
    if OLD not in text:
        raise SystemExit(
            f"ERROR: anchor not found in {TARGET} — upstream changed; "
            f"re-derive the patch (do NOT build with a silently unpatched tree)")
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
