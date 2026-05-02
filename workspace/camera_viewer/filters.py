"""Filter composition + selection state.

A camera is visible iff `all(f.passes(i) for f in active_filters)`. Filters
are independent: each one decides per-camera whether the camera passes,
without consulting the others. The viewer recomputes visibility whenever
a filter's state changes.

Selection (the picked/unpicked flag) is separate from visibility. A camera
can be hidden by a filter but still selected. Selection persists across
filter changes; the user builds it iteratively across multiple filter
configurations.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Filter(Protocol):
    """A filter is anything that can be queried per-camera."""
    name: str
    def passes(self, cam_idx: int) -> bool: ...
    def reset(self) -> None: ...


class TimeRangeFilter:
    """Frame-index range; cam_idx must be in [start, end]."""

    def __init__(self, n: int):
        self.name = "time_range"
        self.n = n
        self.start = 0
        self.end = n - 1

    def passes(self, cam_idx: int) -> bool:
        return self.start <= cam_idx <= self.end

    def reset(self) -> None:
        self.start = 0
        self.end = self.n - 1


class PickedStatusFilter:
    """Tri-state: show all (default) / picked only / unpicked only.

    Reads selection state from the shared SelectionState passed at construction.
    """

    SHOW_ALL = "all"
    SHOW_PICKED = "picked"
    SHOW_UNPICKED = "unpicked"

    def __init__(self, selection: "SelectionState"):
        self.name = "picked_status"
        self.selection = selection
        self.mode = self.SHOW_ALL

    def passes(self, cam_idx: int) -> bool:
        if self.mode == self.SHOW_ALL:
            return True
        is_picked = self.selection.is_picked(cam_idx)
        if self.mode == self.SHOW_PICKED:
            return is_picked
        return not is_picked

    def reset(self) -> None:
        self.mode = self.SHOW_ALL


class SpatialClusterFilter:
    """Per-cluster checkboxes; camera passes iff its cluster is enabled.

    Cluster labels come from clustering.kmeans_position_clusters().
    """

    def __init__(self, cluster_labels: np.ndarray):
        self.name = "spatial_cluster"
        self.labels = cluster_labels
        self.enabled = {int(c): True for c in np.unique(cluster_labels)}

    def passes(self, cam_idx: int) -> bool:
        return self.enabled[int(self.labels[cam_idx])]

    def reset(self) -> None:
        for c in self.enabled:
            self.enabled[c] = True


class FilterStack:
    """Composes a list of filters. A camera is visible iff every filter passes it."""

    def __init__(self, filters: list[Filter]):
        self.filters = filters

    def visible(self, cam_idx: int) -> bool:
        return all(f.passes(cam_idx) for f in self.filters)

    def visible_indices(self, n: int) -> list[int]:
        return [i for i in range(n) if self.visible(i)]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()


class SelectionState:
    """The set of camera indices the user has picked.

    Persistent across filter changes. The viewer's "Save" button serializes
    this to selected_frames.json.
    """

    def __init__(self, n: int):
        self.n = n
        self._picked: set[int] = set()

    def toggle(self, cam_idx: int) -> bool:
        """Flip pick state for one camera. Returns the new state."""
        if cam_idx in self._picked:
            self._picked.remove(cam_idx)
            return False
        self._picked.add(cam_idx)
        return True

    def is_picked(self, cam_idx: int) -> bool:
        return cam_idx in self._picked

    def picked_indices(self) -> list[int]:
        return sorted(self._picked)

    def count(self) -> int:
        return len(self._picked)

    def clear(self) -> None:
        self._picked.clear()
