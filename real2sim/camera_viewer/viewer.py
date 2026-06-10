"""Camera Selection Viewer — main entry point.

Loads a MASt3R-SfM cameras.json + frame images, composes a 3D scene of
camera frustums + textured image planes + a temporal polyline, and serves
it via viser. Click frustums to pick frames; "Save" writes the chosen
indices to a selected_frames.json that MAtCha consumes via --image_idx.

Usage:
    python viewer.py \
        --cameras /path/to/cameras.json \
        --frames /path/to/frames/ \
        --output selected_frames.json \
        --port 8080

The viewer holds the browser open until interrupted (Ctrl-C). Selection
is in-memory; saving is explicit (button click).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import viser
from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]

import data
import ui
from clustering import compute_phashes, kmeans_position_clusters
from filters import (
    DistanceFromSelectionFilter,
    FilterStack,
    ImageSimilarityFilter,
    LookAtTargetFilter,
    PickedStatusFilter,
    SelectionState,
    SpatialClusterFilter,
    TemporalStrideFilter,
    TimeRangeFilter,
)
from slots import SlotsManager


# Color constants
COLOR_UNPICKED = (0.35, 0.55, 0.75)         # muted blue (so picked stand out)
COLOR_PICKED = (1.0, 0.15, 0.15)            # vivid red — high contrast on photo textures
COLOR_PICKED_MARKER = (1.0, 0.85, 0.0)      # gold marker sphere at picked positions
MARKER_RADIUS = 0.04                        # in scene units
LOOKAT_GIZMO_NAME = "/lookat_target"
PICKED_MARKERS_NAMESPACE = "/picked_markers"


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def add_camera_frustums(
    server: viser.ViserServer,
    cams: data.CameraSet,
    on_click: Callable[[int], None],
) -> list[viser.CameraFrustumHandle]:
    """Add one viser CameraFrustum per camera. Returns handles in index order."""
    handles: list[viser.CameraFrustumHandle] = []
    for i in range(cams.n):
        thumb = cams.thumbnails[i]
        h, w = thumb.shape[:2]
        aspect = w / h
        fov = 2 * np.arctan2(h / 2, float(cams.focals[i]))

        rot = cams.cams2world[i, :3, :3]
        wxyz = R.from_matrix(rot).as_quat(scalar_first=True)
        position = cams.cams2world[i, :3, 3]

        frustum = server.scene.add_camera_frustum(
            f"/cams/{i:04d}",
            fov=float(fov),
            aspect=float(aspect),
            scale=0.15,
            color=COLOR_UNPICKED,
            image=thumb,
            wxyz=tuple(wxyz),
            position=tuple(position),
        )

        def make_cb(idx: int):
            def _click(_) -> None:
                on_click(idx)
            return _click

        frustum.on_click(make_cb(i))
        handles.append(frustum)
    return handles


def add_camera_path(server: viser.ViserServer, cams: data.CameraSet) -> None:
    """Add a temporal polyline through the camera centers."""
    server.scene.add_spline_catmull_rom(
        "/path",
        positions=cams.positions.astype(np.float32),
        color=(0.3, 0.3, 0.3),
        line_width=2.0,
        tension=0.5,
    )


# ---------------------------------------------------------------------------
# Coverage colorizer
# ---------------------------------------------------------------------------

def coverage_color(picked_positions: np.ndarray, idx_in_picked: int) -> tuple[float, float, float]:
    """Color a picked frustum by its distance to the nearest other picked camera.

    Small distance (redundant) → red. Large distance (well-spread) → green.
    Uses a magenta/yellow/green gradient so it's distinct from the default
    yellow used in non-coverage mode.
    """
    if len(picked_positions) <= 1:
        return COLOR_PICKED
    me = picked_positions[idx_in_picked]
    others = np.delete(picked_positions, idx_in_picked, axis=0)
    nearest = float(np.linalg.norm(others - me, axis=1).min())
    # Normalize against the median pairwise distance among all picks (rough)
    import scipy.spatial.distance as ssd
    pairs = ssd.pdist(picked_positions)
    median = float(np.median(pairs)) if len(pairs) else 1.0
    if median <= 0:
        return COLOR_PICKED
    ratio = min(nearest / median, 1.0)  # 0 = on top of neighbor; 1 = far
    # Magenta (0,1) → yellow (0.5) → green (1)
    if ratio < 0.5:
        # Magenta → yellow
        t = ratio * 2
        return (1.0, t, 1.0 - t)
    else:
        # Yellow → green
        t = (ratio - 0.5) * 2
        return (1.0 - t, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera selection viewer (Route B from the camera-selection-ui-feasibility note)",
    )
    p.add_argument("--cameras", required=True, type=Path,
                   help="Path to cameras.json from MASt3R-SfM (--sfm_only output)")
    p.add_argument("--frames", type=Path, default=None,
                   help="Override frame directory (re-roots cameras.json filepaths). "
                        "Useful when JSON was produced on a different host.")
    p.add_argument("--output", type=Path, default=Path("selected_frames.json"),
                   help="Where to write the selection (default: ./selected_frames.json)")
    p.add_argument("--port", type=int, default=8080, help="viser HTTP port")
    p.add_argument("--clusters", type=int, default=None,
                   help="K for the spatial-cluster filter (default: auto, max(2, min(8, N//8)))")
    p.add_argument("--thumbnail-edge", type=int, default=512,
                   help="Longest edge in pixels for the in-viewer image planes")
    p.add_argument("--no-phash", action="store_true",
                   help="Skip pHash precompute (disables image-similarity filter; faster startup)")
    p.add_argument("--comparison-views", type=Path, default=None,
                   help="Optional comparison_views.json (schema_v3) — display the "
                        "virtual comparison cameras alongside the candidate cameras. "
                        "Aligns to this dataset's frame via Kabsch on shared anchor frames.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Virtual comparison cameras
# ---------------------------------------------------------------------------

COLOR_VIRTUAL = (0.2, 1.0, 0.6)  # mint green — distinct from blue (unpicked) / red (picked)
VIRTUAL_NAMESPACE = "/virtual_cameras"


def add_virtual_cameras(
    server: viser.ViserServer,
    cams: data.CameraSet,
    comparison_views_json: Path,
) -> int:
    """Load comparison_views.json, Kabsch-align to this dataset's frame, render frustums.

    Returns the number of virtual cameras added.
    """
    with open(comparison_views_json) as f:
        cv = json.load(f)
    if cv.get("schema_version") not in (3, 4, 5):
        print(f"[viewer] WARNING: comparison views file has schema "
              f"{cv.get('schema_version')}, expected 3/4/5 — ignoring")
        return 0

    anchors = cv.get("anchor_frames", [])
    views = cv.get("views", [])
    if not anchors or not views:
        return 0

    # Build P (source anchors from JSON) and Q (this-dataset anchors by basename match)
    src_pts, tgt_pts = [], []
    missing = []
    cam_basenames = [Path(p).name for p in cams.filepaths]
    for a in anchors:
        bn = a["basename"]
        if bn not in cam_basenames:
            missing.append(bn)
            continue
        idx = cam_basenames.index(bn)
        src_pts.append(a["oriented_position"])
        tgt_pts.append(cams.positions[idx])

    if missing:
        print(f"[viewer] WARNING: {len(missing)} anchor frame(s) not in this dataset: "
              f"{missing}. Skipping virtual cameras.")
        return 0
    if len(src_pts) < 3:
        print(f"[viewer] WARNING: need ≥3 matching anchors; got {len(src_pts)}. "
              f"Skipping virtual cameras.")
        return 0

    # Procrustes WITH scaling (Umeyama 1991): aligning across two SfM runs
    # requires solving for scale too — different MASt3R-SfM runs converge
    # to different scale conventions even on the same scene. Variants of
    # the same SfM share scale (residuals ~mm); n350-vs-variant doesn't.
    P = np.asarray(src_pts)
    Q = np.asarray(tgt_pts)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d_sign])
    R_mat = Vt.T @ D @ U.T
    var_P = float((Pc * Pc).sum())
    if var_P <= 0:
        scale = 1.0
    else:
        scale = float((np.diag(D) * S).sum() / var_P)
    t_vec = cQ - scale * R_mat @ cP
    residuals = np.linalg.norm((scale * (P @ R_mat.T) + t_vec) - Q, axis=1)
    print(f"[viewer] virtual-camera alignment: scale={scale:.4f}  "
          f"anchor residuals = {residuals.round(3).tolist()} m")

    for view in views:
        name = view["name"]
        src_pos = np.asarray(view["world_position"])
        tgt_pos = scale * (R_mat @ src_pos) + t_vec

        # Compose source rotation with R for display orientation
        src_quat = view["world_rotation_quat_wxyz"]  # (w, x, y, z)
        # Convert quat → rotmat
        w, x, y, z = src_quat
        src_rot = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])
        tgt_rot = R_mat @ src_rot
        tgt_quat = R.from_matrix(tgt_rot).as_quat(scalar_first=True)

        # Field-of-view from lens + sensor (default 50mm + 36mm sensor)
        lens = float(view.get("lens_mm", 50.0))
        sensor_w = float(view.get("sensor_width_mm", 36.0))
        sensor_h = float(view.get("sensor_height_mm", 24.0))
        fov_v = 2 * np.arctan2(sensor_h / 2, lens)
        aspect = sensor_w / sensor_h

        server.scene.add_camera_frustum(
            f"{VIRTUAL_NAMESPACE}/{name}",
            fov=float(fov_v),
            aspect=float(aspect),
            scale=0.25,
            color=COLOR_VIRTUAL,
            wxyz=tuple(tgt_quat),
            position=tuple(tgt_pos),
        )
    print(f"[viewer] added {len(views)} virtual comparison camera(s) "
          f"(color: mint green): {[v['name'] for v in views]}")
    return len(views)


def main() -> int:
    args = parse_args()

    print(f"[viewer] loading cameras from {args.cameras}")
    cams = data.load(
        cameras_json=args.cameras,
        frames_dir=args.frames,
        thumbnail_long_edge=args.thumbnail_edge,
    )
    print(f"[viewer] loaded {cams.n} cameras")

    # --- State ---
    selection = SelectionState(cams.n)
    cluster_labels = kmeans_position_clusters(cams.positions, k=args.clusters)
    n_clusters = len(np.unique(cluster_labels))
    print(f"[viewer] computed {n_clusters} spatial clusters")

    if args.no_phash:
        # Provide zero-hashes so the filter exists but always passes
        phashes = np.zeros(cams.n, dtype=np.uint64)
    else:
        print(f"[viewer] computing pHashes for {cams.n} thumbnails...")
        phashes = compute_phashes(cams.thumbnails)
        print(f"[viewer] pHash done")

    filters = FilterStack([
        TimeRangeFilter(cams.n),
        TemporalStrideFilter(),
        SpatialClusterFilter(cluster_labels),
        DistanceFromSelectionFilter(cams.positions, selection),
        LookAtTargetFilter(cams.positions, cams.forward_axes),
        ImageSimilarityFilter(phashes),
        PickedStatusFilter(selection),
    ])

    # --- Server ---
    server = viser.ViserServer(port=args.port)
    print(f"[viewer] viser listening on http://localhost:{args.port}")

    # Gravity-align the navigation: SfM's world frame is arbitrary, so set
    # viser's up direction from the cameras' average up-vector (same prior
    # as orient_mesh.py::estimate_gravity_from_cameras uses to disambiguate
    # floor candidates). Low confidence (e.g. banked orbit captures) →
    # leave viser's default +Z alone.
    up, up_conf = cams.estimate_up()
    if up_conf >= 0.5:
        server.scene.set_up_direction(tuple(float(v) for v in up))
        print(f"[viewer] up direction from camera average: "
              f"({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}), "
              f"confidence {up_conf:.2f}")
    else:
        print(f"[viewer] camera-up confidence too low ({up_conf:.2f}); "
              f"keeping default +Z up")

    add_camera_path(server, cams)

    # --- Frustums + per-pick marker spheres ---
    frustums: list[viser.CameraFrustumHandle] = []
    picked_markers: dict[int, Any] = {}  # cam_idx → IcosphereHandle

    def add_picked_marker(i: int) -> None:
        """Add a bright sphere just above the camera position to mark it picked."""
        pos = cams.positions[i].astype(float)
        marker = server.scene.add_icosphere(
            f"{PICKED_MARKERS_NAMESPACE}/{i:04d}",
            radius=MARKER_RADIUS,
            color=COLOR_PICKED_MARKER,
            position=tuple(pos),
        )
        picked_markers[i] = marker

    def remove_picked_marker(i: int) -> None:
        marker = picked_markers.pop(i, None)
        if marker is not None:
            try:
                marker.remove()
            except Exception:
                pass

    def update_frustum_appearance(i: int) -> None:
        """Apply current filter visibility + selection color to frustum i."""
        f = frustums[i]
        f.visible = filters.visible(i)
        if selection.is_picked(i):
            if getattr(server, "coverage_mode", False):
                # Look up this camera's index within the picked set, then colorize
                picked = selection.picked_indices()
                if i in picked:
                    pos = cams.positions[picked]
                    pidx = picked.index(i)
                    f.color = coverage_color(pos, pidx)
                else:
                    f.color = COLOR_PICKED
            else:
                f.color = COLOR_PICKED
            # Marker — add if missing
            if i not in picked_markers:
                add_picked_marker(i)
        else:
            f.color = COLOR_UNPICKED
            # Marker — remove if present
            if i in picked_markers:
                remove_picked_marker(i)

    def refresh_all() -> None:
        for i in range(cams.n):
            update_frustum_appearance(i)
        # Visible count depends on filter state; refresh it whenever any
        # filter or selection-dependent filter input changes.
        try:
            server.refresh_visible_count()  # type: ignore[attr-defined]
        except AttributeError:
            pass  # GUI not yet built — refresh_all may run before build_gui

    def on_click(idx: int) -> None:
        was_picked = selection.is_picked(idx)
        new_state = selection.toggle(idx)
        if was_picked == new_state:
            # No-op (locked); brief log line then bail
            print(f"[viewer] frame {idx} click ignored — picks locked")
            return
        # Selection-dependent filters mean we likely need to refresh ALL
        # frustums, not just the one clicked. Coverage mode also requires
        # updating every picked frustum's color when the picked set changes.
        if filters.has_selection_dependent_filter() or getattr(server, "coverage_mode", False):
            refresh_all()
        else:
            update_frustum_appearance(idx)
        server.refresh_counter()  # type: ignore[attr-defined]
        print(f"[viewer] frame {idx} {'picked' if new_state else 'unpicked'} "
              f"(total: {selection.count()})")

    def save_selection(out: Path) -> None:
        path = ui.write_selection(
            selection=selection,
            source_pool=args.frames or args.cameras.parent,
            output_path=out,
        )
        print(f"[viewer] wrote selection ({selection.count()} frames) → {path}")

    # --- Look-at gizmo ---
    lookat_filter = filters.get("look_at_target")
    assert isinstance(lookat_filter, LookAtTargetFilter)
    # Pre-position the gizmo at the centroid of all cameras
    centroid = cams.positions.mean(axis=0).astype(np.float32)
    lookat_gizmo: viser.TransformControlsHandle | None = None

    def show_lookat_gizmo(visible: bool) -> None:
        nonlocal lookat_gizmo
        if visible and lookat_gizmo is None:
            gizmo = server.scene.add_transform_controls(
                LOOKAT_GIZMO_NAME,
                position=tuple(centroid),
                disable_axes=False,
                disable_sliders=True,
                disable_rotations=True,
                scale=0.5,
            )
            lookat_gizmo = gizmo
            lookat_filter.set_target(np.array(gizmo.position))

            @gizmo.on_update
            def _(_) -> None:
                lookat_filter.set_target(np.array(gizmo.position))
                refresh_all()
        elif not visible and lookat_gizmo is not None:
            lookat_gizmo.remove()
            lookat_gizmo = None
            lookat_filter.set_target(None)

    frustums = add_camera_frustums(server, cams, on_click)
    print(f"[viewer] added {len(frustums)} camera frustums")

    # --- Optional: virtual comparison cameras (mint green) ---
    if args.comparison_views and args.comparison_views.exists():
        add_virtual_cameras(server, cams, args.comparison_views)

    # --- Slots manager: parallel to cameras.json with .slots.json suffix ---
    slots_mgr = SlotsManager(args.cameras)
    print(f"[viewer] slots file: {slots_mgr.path}  ({len(slots_mgr.names())} existing slot(s))")

    # --- GUI ---
    ui.build_gui(
        server=server,
        n=cams.n,
        filters=filters,
        selection=selection,
        slots_mgr=slots_mgr,
        on_change=refresh_all,
        on_save=save_selection,
        on_lookat_toggle=show_lookat_gizmo,
        output_path=args.output,
    )

    print("[viewer] ready — open the URL above in a browser")
    print("[viewer] Ctrl-C to quit (selection is lost unless saved first)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[viewer] shutting down")
        return 0


if __name__ == "__main__":
    sys.exit(main())
