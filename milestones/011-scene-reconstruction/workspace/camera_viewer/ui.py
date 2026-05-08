"""GUI panel composition for the viewer.

Wires viser GUI primitives to the filter / selection state. Panel
organization (v1):

    📁 Temporal       time range, stride
    📁 Spatial        cluster checkboxes + invert, distance-from-selection
    📁 Content        look-at gizmo + radius, image-similarity threshold
    📁 Picked         show: all/picked/unpicked
    📁 Slots          named save/load of filter+selection state
    📁 Selection      bulk select/deselect, lock toggle, save/clear/reset

Selection counter at the top is always visible.

The widget handles are stashed in a dict so that loading a slot can
programmatically sync widget values to the loaded filter state.

Kept separate from `viewer.py` so the scene-composition logic doesn't
get tangled with widget plumbing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import viser

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


# Map between PickedStatusFilter modes and dropdown labels
_PICKED_LABEL_TO_MODE = {
    "all": PickedStatusFilter.SHOW_ALL,
    "picked only": PickedStatusFilter.SHOW_PICKED,
    "unpicked only": PickedStatusFilter.SHOW_UNPICKED,
}
_PICKED_MODE_TO_LABEL = {v: k for k, v in _PICKED_LABEL_TO_MODE.items()}


def build_gui(
    server: viser.ViserServer,
    n: int,
    filters: FilterStack,
    selection: SelectionState,
    slots_mgr: SlotsManager,
    on_change: Callable[[], None],
    on_save: Callable[[Path], None],
    on_lookat_toggle: Callable[[bool], None],
    output_path: Path,
) -> None:
    """Create the side-panel widgets and wire them into the filter stack.

    Holds widget handles internally so slot-loading can sync them.
    """
    # Suppression flag: set to True while applying a slot, so widget on_update
    # callbacks don't double-mutate state and re-fire on_change for every widget.
    _suspend = {"on": False}

    def fire_on_change() -> None:
        if not _suspend["on"]:
            on_change()

    # --- Counters ---
    visible_text = server.gui.add_text("Visible", initial_value=f"{n} / {n}")
    counter = server.gui.add_text("Selected", initial_value=f"0 / {n}")

    def refresh_counter() -> None:
        counter.value = f"{selection.count()} / {n}"

    def refresh_visible_count() -> None:
        visible_count = sum(1 for i in range(n) if filters.visible(i))
        visible_text.value = f"{visible_count} / {n}"

    # ----------------------------------------------------------------
    # Temporal
    # ----------------------------------------------------------------
    with server.gui.add_folder("Temporal"):
        time_filter = filters.get("time_range")
        assert isinstance(time_filter, TimeRangeFilter)
        time_slider = server.gui.add_multi_slider(
            "Frame range",
            min=0, max=n - 1, step=1, initial_value=(0, n - 1),
        )

        @time_slider.on_update
        def _(_) -> None:
            time_filter.start, time_filter.end = time_slider.value
            fire_on_change()

        stride_filter = filters.get("temporal_stride")
        assert isinstance(stride_filter, TemporalStrideFilter)
        stride_slider = server.gui.add_slider(
            "Stride (every Nth)",
            min=1, max=20, step=1, initial_value=1,
        )

        @stride_slider.on_update
        def _(_) -> None:
            stride_filter.stride = int(stride_slider.value)
            fire_on_change()

    # ----------------------------------------------------------------
    # Spatial
    # ----------------------------------------------------------------
    with server.gui.add_folder("Spatial"):
        cluster_filter = filters.get("spatial_cluster")
        assert isinstance(cluster_filter, SpatialClusterFilter)
        cluster_boxes: dict[int, Any] = {}
        with server.gui.add_folder("Clusters"):
            for cluster_id in sorted(cluster_filter.cluster_enabled):
                box = server.gui.add_checkbox(
                    f"Cluster {cluster_id}", initial_value=True,
                )
                cluster_boxes[cluster_id] = box

                def make_cb(cid: int, b: viser.GuiCheckboxHandle) -> Callable[[object], None]:
                    def _on(_: object) -> None:
                        cluster_filter.cluster_enabled[cid] = b.value
                        fire_on_change()
                    return _on

                box.on_update(make_cb(cluster_id, box))

            invert_box = server.gui.add_checkbox("Invert clusters", initial_value=False)

            @invert_box.on_update
            def _(_) -> None:
                cluster_filter.invert = invert_box.value
                fire_on_change()

        dist_filter = filters.get("distance_from_selection")
        assert isinstance(dist_filter, DistanceFromSelectionFilter)
        max_dist = float(_estimate_scene_radius(dist_filter.positions))
        dist_slider = server.gui.add_slider(
            "Min distance from picked (m)",
            min=0.0, max=max_dist, step=max_dist / 100, initial_value=0.0,
        )

        @dist_slider.on_update
        def _(_) -> None:
            dist_filter.min_dist = float(dist_slider.value)
            fire_on_change()

    # ----------------------------------------------------------------
    # Content
    # ----------------------------------------------------------------
    with server.gui.add_folder("Content"):
        lookat_filter = filters.get("look_at_target")
        assert isinstance(lookat_filter, LookAtTargetFilter)
        lookat_toggle = server.gui.add_checkbox("Look-at filter", initial_value=False)
        lookat_radius = server.gui.add_slider(
            "Radius (m)",
            min=0.05, max=max_dist, step=max_dist / 200,
            initial_value=lookat_filter.radius,
        )

        @lookat_toggle.on_update
        def _(_) -> None:
            lookat_filter.enabled = lookat_toggle.value
            on_lookat_toggle(lookat_toggle.value)
            fire_on_change()

        @lookat_radius.on_update
        def _(_) -> None:
            lookat_filter.radius = float(lookat_radius.value)
            fire_on_change()

        sim_filter = filters.get("image_similarity")
        assert isinstance(sim_filter, ImageSimilarityFilter)
        sim_slider = server.gui.add_slider(
            "pHash dedupe threshold",
            min=0, max=20, step=1, initial_value=0,
        )

        @sim_slider.on_update
        def _(_) -> None:
            sim_filter.threshold = int(sim_slider.value)
            fire_on_change()

    # ----------------------------------------------------------------
    # Picked status
    # ----------------------------------------------------------------
    with server.gui.add_folder("Picked"):
        picked_filter = filters.get("picked_status")
        assert isinstance(picked_filter, PickedStatusFilter)
        picked_dropdown = server.gui.add_dropdown(
            "Show",
            options=tuple(_PICKED_LABEL_TO_MODE.keys()),
            initial_value="all",
        )

        @picked_dropdown.on_update
        def _(_) -> None:
            picked_filter.mode = _PICKED_LABEL_TO_MODE[picked_dropdown.value]
            fire_on_change()

    # ----------------------------------------------------------------
    # Slots — named save/load
    # ----------------------------------------------------------------
    with server.gui.add_folder("Slots"):
        slots_status = server.gui.add_text(
            "Slots file", initial_value=str(slots_mgr.path),
        )
        slot_name_input = server.gui.add_text(
            "Name", initial_value="",
        )
        save_slot_btn = server.gui.add_button("💾 Save slot")

        existing = list(slots_mgr.names()) or ["(no slots)"]
        load_dropdown = server.gui.add_dropdown(
            "Load slot",
            options=tuple(existing),
            initial_value=existing[0],
        )
        load_slot_btn = server.gui.add_button("📥 Load")
        delete_slot_btn = server.gui.add_button("🗑️ Delete slot")
        slot_action_status = server.gui.add_text(
            "Last slot action", initial_value="(none)",
        )

        def refresh_slot_dropdown(prefer_select: str | None = None) -> None:
            names = list(slots_mgr.names()) or ["(no slots)"]
            load_dropdown.options = tuple(names)
            if prefer_select and prefer_select in names:
                load_dropdown.value = prefer_select
            else:
                load_dropdown.value = names[0]

        @save_slot_btn.on_click
        def _(_) -> None:
            name = slot_name_input.value.strip()
            if not name:
                slot_action_status.value = "name is empty — slot not saved"
                return
            slots_mgr.save(name, filters, selection)
            refresh_slot_dropdown(prefer_select=name)
            slot_action_status.value = (
                f"saved '{name}' at {datetime.now().strftime('%H:%M:%S')} "
                f"({len(slots_mgr.names())} total)"
            )
            print(f"[viewer] saved slot '{name}'")

        @load_slot_btn.on_click
        def _(_) -> None:
            name = load_dropdown.value
            if name == "(no slots)" or not slots_mgr.names():
                slot_action_status.value = "no slot to load"
                return
            try:
                slot = slots_mgr.load(name, filters, selection)
            except KeyError:
                slot_action_status.value = f"slot '{name}' not found"
                return
            # Sync widgets to match loaded filter+selection state
            _suspend["on"] = True
            try:
                _sync_widgets_to_state(
                    widgets=dict(
                        time_slider=time_slider,
                        stride_slider=stride_slider,
                        cluster_boxes=cluster_boxes,
                        invert_box=invert_box,
                        dist_slider=dist_slider,
                        lookat_toggle=lookat_toggle,
                        lookat_radius=lookat_radius,
                        sim_slider=sim_slider,
                        picked_dropdown=picked_dropdown,
                        lock_toggle=lock_toggle,
                    ),
                    filters=filters,
                    selection=selection,
                )
            finally:
                _suspend["on"] = False
            on_lookat_toggle(lookat_filter.enabled)
            refresh_counter()
            on_change()
            slot_action_status.value = (
                f"loaded '{name}' (saved {slot.get('saved_at', '?')}, "
                f"{selection.count()} picks)"
            )
            print(f"[viewer] loaded slot '{name}' — {selection.count()} picks")

        @delete_slot_btn.on_click
        def _(_) -> None:
            name = load_dropdown.value
            if name == "(no slots)" or not slots_mgr.names():
                slot_action_status.value = "nothing to delete"
                return
            ok = slots_mgr.delete(name)
            if ok:
                refresh_slot_dropdown()
                slot_action_status.value = f"deleted '{name}'"
                print(f"[viewer] deleted slot '{name}'")

    # ----------------------------------------------------------------
    # Selection actions
    # ----------------------------------------------------------------
    with server.gui.add_folder("Selection"):
        select_vis_btn = server.gui.add_button("✅ Select Visible")
        deselect_vis_btn = server.gui.add_button("❌ Deselect Visible")

        @select_vis_btn.on_click
        def _(_) -> None:
            visible = filters.visible_indices(n)
            added = selection.select_indices(visible)
            refresh_counter()
            on_change()
            print(f"[viewer] Select Visible: +{added} (total: {selection.count()})")

        @deselect_vis_btn.on_click
        def _(_) -> None:
            visible = filters.visible_indices(n)
            removed = selection.deselect_indices(visible)
            if removed == 0 and selection.lock_picks:
                print("[viewer] Deselect Visible: ignored — picks are locked")
            else:
                print(f"[viewer] Deselect Visible: -{removed} (total: {selection.count()})")
            refresh_counter()
            on_change()

        lock_toggle = server.gui.add_checkbox("🔒 Lock picks", initial_value=False)

        @lock_toggle.on_update
        def _(_) -> None:
            selection.lock_picks = lock_toggle.value

        coverage_toggle = server.gui.add_checkbox("Coverage colorize", initial_value=False)

        @coverage_toggle.on_update
        def _(_) -> None:
            server.coverage_mode = coverage_toggle.value  # type: ignore[attr-defined]
            fire_on_change()

        save_btn = server.gui.add_button("💾 Save selection")
        save_status = server.gui.add_text(
            "Last save", initial_value=f"(not saved) → {output_path}",
        )
        clear_btn = server.gui.add_button("🗑️ Clear selection")
        reset_btn = server.gui.add_button("Reset filters")

        @save_btn.on_click
        def _(_) -> None:
            on_save(output_path)
            save_status.value = (
                f"{datetime.now().strftime('%H:%M:%S')} "
                f"({selection.count()} picks) → {output_path}"
            )

        @clear_btn.on_click
        def _(_) -> None:
            selection.clear()
            refresh_counter()
            on_change()

        @reset_btn.on_click
        def _(_) -> None:
            filters.reset()
            time_slider.value = (0, n - 1)
            stride_slider.value = 1
            invert_box.value = False
            for box in cluster_boxes.values():
                box.value = True
            dist_slider.value = 0.0
            lookat_toggle.value = False
            lookat_radius.value = lookat_filter.radius
            sim_slider.value = 0
            picked_dropdown.value = "all"
            on_lookat_toggle(False)
            on_change()

    # Expose state hooks for the click handler in viewer.py
    server.refresh_counter = refresh_counter            # type: ignore[attr-defined]
    server.refresh_visible_count = refresh_visible_count  # type: ignore[attr-defined]
    server.coverage_mode = False                        # type: ignore[attr-defined]


def _sync_widgets_to_state(
    widgets: dict,
    filters: FilterStack,
    selection: SelectionState,
) -> None:
    """Push current filter+selection state into the widgets.

    Used after loading a slot — the underlying filter objects already
    have the loaded state; the widgets need to reflect it visually.
    """
    time_filter = filters.get("time_range")
    assert isinstance(time_filter, TimeRangeFilter)
    widgets["time_slider"].value = (time_filter.start, time_filter.end)

    stride_filter = filters.get("temporal_stride")
    assert isinstance(stride_filter, TemporalStrideFilter)
    widgets["stride_slider"].value = stride_filter.stride

    cluster_filter = filters.get("spatial_cluster")
    assert isinstance(cluster_filter, SpatialClusterFilter)
    for cid, box in widgets["cluster_boxes"].items():
        box.value = cluster_filter.cluster_enabled.get(cid, True)
    widgets["invert_box"].value = cluster_filter.invert

    dist_filter = filters.get("distance_from_selection")
    assert isinstance(dist_filter, DistanceFromSelectionFilter)
    widgets["dist_slider"].value = dist_filter.min_dist

    lookat_filter = filters.get("look_at_target")
    assert isinstance(lookat_filter, LookAtTargetFilter)
    widgets["lookat_toggle"].value = lookat_filter.enabled
    widgets["lookat_radius"].value = lookat_filter.radius

    sim_filter = filters.get("image_similarity")
    assert isinstance(sim_filter, ImageSimilarityFilter)
    widgets["sim_slider"].value = sim_filter.threshold

    picked_filter = filters.get("picked_status")
    assert isinstance(picked_filter, PickedStatusFilter)
    widgets["picked_dropdown"].value = _PICKED_MODE_TO_LABEL[picked_filter.mode]

    widgets["lock_toggle"].value = selection.lock_picks


def _estimate_scene_radius(positions) -> float:
    """Rough scene-radius estimate for slider upper bounds."""
    import numpy as np
    extent = positions.max(axis=0) - positions.min(axis=0)
    return float(np.linalg.norm(extent))


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
