"""GUI panel composition for the viewer.

Wires viser GUI primitives to the filter / selection state. Callbacks
update the filter state and trigger a visibility refresh on the scene.

Kept separate from `viewer.py` so the scene-composition logic doesn't
get tangled with widget plumbing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import viser

from filters import (
    FilterStack,
    PickedStatusFilter,
    SelectionState,
    SpatialClusterFilter,
    TimeRangeFilter,
)


def build_gui(
    server: viser.ViserServer,
    n: int,
    filters: FilterStack,
    selection: SelectionState,
    on_change: Callable[[], None],
    on_save: Callable[[Path], None],
    output_path: Path,
) -> None:
    """Create the side-panel widgets and wire them into the filter stack.

    `on_change` is called whenever any filter state changes; the caller
    is expected to refresh frustum visibility based on `filters.visible(i)`.
    `on_save` is called when the user clicks "Save selection".
    """
    # --- Selection counter (always visible at top) ---
    counter = server.gui.add_text("Selected", initial_value=f"0 / {n}")

    def refresh_counter() -> None:
        counter.value = f"{selection.count()} / {n}"

    # --- Filters folder ---
    with server.gui.add_folder("Filters"):
        # Time range
        time_filter = next(f for f in filters.filters if f.name == "time_range")
        assert isinstance(time_filter, TimeRangeFilter)
        time_slider = server.gui.add_multi_slider(
            "Frame range",
            min=0, max=n - 1, step=1,
            initial_value=(0, n - 1),
        )

        @time_slider.on_update
        def _(_) -> None:
            time_filter.start, time_filter.end = time_slider.value
            on_change()

        # Picked status
        picked_filter = next(f for f in filters.filters if f.name == "picked_status")
        assert isinstance(picked_filter, PickedStatusFilter)
        picked_dropdown = server.gui.add_dropdown(
            "Show",
            options=("all", "picked only", "unpicked only"),
            initial_value="all",
        )
        _mode_map = {
            "all": PickedStatusFilter.SHOW_ALL,
            "picked only": PickedStatusFilter.SHOW_PICKED,
            "unpicked only": PickedStatusFilter.SHOW_UNPICKED,
        }

        @picked_dropdown.on_update
        def _(_) -> None:
            picked_filter.mode = _mode_map[picked_dropdown.value]
            on_change()

        # Spatial clusters
        cluster_filter = next(
            (f for f in filters.filters if f.name == "spatial_cluster"), None,
        )
        if cluster_filter is not None:
            assert isinstance(cluster_filter, SpatialClusterFilter)
            with server.gui.add_folder("Spatial clusters"):
                for cluster_id in sorted(cluster_filter.enabled):
                    box = server.gui.add_checkbox(
                        f"Cluster {cluster_id}", initial_value=True,
                    )

                    def make_cb(cid: int, b: viser.GuiCheckboxHandle) -> Callable[[object], None]:
                        def _on(_: object) -> None:
                            cluster_filter.enabled[cid] = b.value
                            on_change()
                        return _on

                    box.on_update(make_cb(cluster_id, box))

    # --- Selection actions ---
    with server.gui.add_folder("Selection"):
        save_btn = server.gui.add_button("💾 Save selection")
        clear_btn = server.gui.add_button("🗑️ Clear selection")
        reset_btn = server.gui.add_button("Reset filters")

        @save_btn.on_click
        def _(_) -> None:
            on_save(output_path)

        @clear_btn.on_click
        def _(_) -> None:
            selection.clear()
            refresh_counter()
            on_change()

        @reset_btn.on_click
        def _(_) -> None:
            filters.reset()
            time_slider.value = (0, n - 1)
            picked_dropdown.value = "all"
            on_change()

    # Expose the counter refresher for the click handler in viewer.py
    server.refresh_counter = refresh_counter  # type: ignore[attr-defined]


def write_selection(
    selection: SelectionState,
    source_pool: Path,
    output_path: Path,
) -> Path:
    """Serialize the current selection to JSON in the format MAtCha consumes."""
    payload = {
        "source_pool": str(source_pool),
        "n_pool": selection.n,
        "selected_idx": selection.picked_indices(),
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path
