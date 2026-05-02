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
is in-memory; saving is explicit (button click). If you Ctrl-C without
saving, the selection is lost.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import viser
from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]

import data
import ui
from clustering import kmeans_position_clusters
from filters import (
    FilterStack,
    PickedStatusFilter,
    SelectionState,
    SpatialClusterFilter,
    TimeRangeFilter,
)


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def add_camera_frustums(
    server: viser.ViserServer,
    cams: data.CameraSet,
    on_click: Callable[[int], None],
) -> list[viser.CameraFrustumHandle]:
    """Add one viser CameraFrustum per camera. Returns the handles in index order.

    The selection state is consulted only at click time (via the on_click
    callback), so it doesn't need to be passed here. Visibility/color
    refresh after a click is the caller's job.
    """
    handles: list[viser.CameraFrustumHandle] = []
    # Assume MASt3R-SfM downscaled images to 512 long-edge for SfM, so the
    # focal-to-fov computation is referenced against that. We use the
    # thumbnail aspect ratio (which preserves the source).
    for i in range(cams.n):
        thumb = cams.thumbnails[i]
        h, w = thumb.shape[:2]
        aspect = w / h
        # MASt3R focals are in pixels at the SfM-internal 512-edge image.
        # Vertical fov = 2 * atan( h_internal / (2*f) ).
        # h_internal ≈ 512 * (h/max(h,w))  approximately. For simplicity here
        # we use the thumbnail h directly and treat focal as in same pixel space.
        fov = 2 * np.arctan2(h / 2, float(cams.focals[i]))

        # cams2world rotation as quaternion (wxyz)
        rot = cams.cams2world[i, :3, :3]
        wxyz = R.from_matrix(rot).as_quat(scalar_first=True)
        position = cams.cams2world[i, :3, 3]

        frustum = server.scene.add_camera_frustum(
            f"/cams/{i:04d}",
            fov=float(fov),
            aspect=float(aspect),
            scale=0.15,
            color=(0.4, 0.7, 1.0),  # default light blue
            image=thumb,
            wxyz=tuple(wxyz),
            position=tuple(position),
        )

        # closure-capture the index for the click callback
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
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera selection viewer (Route B from the camera-selection-ui-feasibility note)",
    )
    p.add_argument("--cameras", required=True, type=Path,
                   help="Path to cameras.json from MASt3R-SfM (--sfm_only output)")
    p.add_argument("--frames", type=Path, default=None,
                   help="Override frame directory (re-roots the cameras.json filepaths). "
                        "Useful when the JSON was produced on a different host.")
    p.add_argument("--output", type=Path, default=Path("selected_frames.json"),
                   help="Where to write the selection (default: ./selected_frames.json)")
    p.add_argument("--port", type=int, default=8080,
                   help="viser HTTP port (default 8080)")
    p.add_argument("--clusters", type=int, default=None,
                   help="K for the spatial-cluster filter (default: auto, max(2, min(8, N//8)))")
    p.add_argument("--thumbnail-edge", type=int, default=512,
                   help="Longest edge in pixels for the in-viewer image planes")
    return p.parse_args()


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

    filters = FilterStack([
        TimeRangeFilter(cams.n),
        PickedStatusFilter(selection),
        SpatialClusterFilter(cluster_labels),
    ])

    # --- Server ---
    server = viser.ViserServer(port=args.port)
    print(f"[viewer] viser listening on http://localhost:{args.port}")

    add_camera_path(server, cams)

    # --- Frustums + click handler ---
    frustums: list[viser.CameraFrustumHandle] = []  # populated below

    def update_frustum_appearance(i: int) -> None:
        """Update the i-th frustum's color/visibility based on filter + selection state."""
        f = frustums[i]
        f.visible = filters.visible(i)
        # Picked = bright yellow; unpicked = light blue
        f.color = (1.0, 0.85, 0.2) if selection.is_picked(i) else (0.4, 0.7, 1.0)

    def on_click(idx: int) -> None:
        new_state = selection.toggle(idx)
        update_frustum_appearance(idx)
        server.refresh_counter()  # type: ignore[attr-defined]
        print(f"[viewer] frame {idx} {'picked' if new_state else 'unpicked'} "
              f"(total: {selection.count()})")

    def refresh_all() -> None:
        for i in range(cams.n):
            update_frustum_appearance(i)

    def save_selection(out: Path) -> None:
        path = ui.write_selection(
            selection=selection,
            source_pool=args.frames or args.cameras.parent,
            output_path=out,
        )
        print(f"[viewer] wrote selection ({selection.count()} frames) → {path}")

    frustums = add_camera_frustums(server, cams, on_click)
    print(f"[viewer] added {len(frustums)} camera frustums")

    # --- GUI ---
    ui.build_gui(
        server=server,
        n=cams.n,
        filters=filters,
        selection=selection,
        on_change=refresh_all,
        on_save=save_selection,
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
