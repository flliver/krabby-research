"""Filter composition + selection state.

A camera is visible iff `all(f.passes(i) for f in active_filters)`. Filters
are independent: each one decides per-camera whether the camera passes,
without consulting the others. The viewer recomputes visibility whenever
a filter's state changes OR the selection state changes (some filters
depend on selection).

Selection (the picked/unpicked flag) is separate from visibility. A camera
can be hidden by a filter but still selected. Selection persists across
filter changes; the user builds it iteratively across multiple filter
configurations.

Filter classes (v1):
    TimeRangeFilter            — frame index range
    TemporalStrideFilter       — every Nth frame within the time range
    SpatialClusterFilter       — k-means cluster checkboxes (with invert toggle)
    DistanceFromSelectionFilter — hide cameras near already-picked ones
    LookAtTargetFilter         — only show cameras whose forward ray hits a target
    ImageSimilarityFilter      — pHash dedupe (cheap ASMK stand-in)
    PickedStatusFilter         — all / picked-only / unpicked-only
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Filter(Protocol):
    """A filter is anything that can be queried per-camera + serialized."""
    name: str
    enabled: bool
    def passes(self, cam_idx: int) -> bool: ...
    def reset(self) -> None: ...
    def to_state(self) -> dict: ...
    def from_state(self, state: dict) -> None: ...


# ---------------------------------------------------------------------------
# Temporal filters
# ---------------------------------------------------------------------------

class TimeRangeFilter:
    """Frame-index range; cam_idx must be in [start, end]."""

    def __init__(self, n: int):
        self.name = "time_range"
        self.enabled = True
        self.n = n
        self.start = 0
        self.end = n - 1

    def passes(self, cam_idx: int) -> bool:
        return self.start <= cam_idx <= self.end

    def reset(self) -> None:
        self.start = 0
        self.end = self.n - 1

    def to_state(self) -> dict:
        return {"start": int(self.start), "end": int(self.end)}

    def from_state(self, state: dict) -> None:
        self.start = int(state.get("start", 0))
        self.end = int(state.get("end", self.n - 1))


class TemporalStrideFilter:
    """Show every Nth frame (relative to the visible time-range start).

    stride=1 → all frames pass.
    stride=5 → every 5th frame passes.

    Combined with TimeRangeFilter this gives "every 5th frame within
    the picked range" — fast coverage scans across long videos.
    """

    def __init__(self):
        self.name = "temporal_stride"
        self.enabled = True
        self.stride = 1
        self.anchor = 0  # the index that stride is measured against (default 0)

    def passes(self, cam_idx: int) -> bool:
        if self.stride <= 1:
            return True
        return (cam_idx - self.anchor) % self.stride == 0

    def reset(self) -> None:
        self.stride = 1

    def to_state(self) -> dict:
        return {"stride": int(self.stride), "anchor": int(self.anchor)}

    def from_state(self, state: dict) -> None:
        self.stride = int(state.get("stride", 1))
        self.anchor = int(state.get("anchor", 0))


# ---------------------------------------------------------------------------
# Spatial filters
# ---------------------------------------------------------------------------

class SpatialClusterFilter:
    """Per-cluster checkboxes; pass iff camera's cluster is enabled.

    `invert` toggle flips the meaning: when invert=True, pass iff
    camera's cluster is NOT enabled. Saves clicks for "show everything
    except this big cluster" cases.
    """

    def __init__(self, cluster_labels: np.ndarray):
        self.name = "spatial_cluster"
        self.enabled = True
        self.labels = cluster_labels
        self.cluster_enabled = {int(c): True for c in np.unique(cluster_labels)}
        self.invert = False

    def passes(self, cam_idx: int) -> bool:
        in_enabled_cluster = self.cluster_enabled[int(self.labels[cam_idx])]
        return (not in_enabled_cluster) if self.invert else in_enabled_cluster

    def reset(self) -> None:
        for c in self.cluster_enabled:
            self.cluster_enabled[c] = True
        self.invert = False

    def to_state(self) -> dict:
        # JSON keys must be strings
        return {
            "cluster_enabled": {str(k): bool(v) for k, v in self.cluster_enabled.items()},
            "invert": bool(self.invert),
        }

    def from_state(self, state: dict) -> None:
        ce = state.get("cluster_enabled", {})
        for k, v in ce.items():
            self.cluster_enabled[int(k)] = bool(v)
        self.invert = bool(state.get("invert", False))


class DistanceFromSelectionFilter:
    """Hide cameras within `min_dist` meters of any already-picked camera.

    The picked cameras themselves always pass (you want to see your
    selection). Unpicked cameras within min_dist of a picked one are hidden,
    surfacing the underexplored regions.

    min_dist=0 disables the filter (everything passes).

    Reads selection state from the shared SelectionState passed at construction.
    """

    def __init__(self, positions: np.ndarray, selection: "SelectionState"):
        self.name = "distance_from_selection"
        self.enabled = True
        self.positions = positions       # (N, 3)
        self.selection = selection
        self.min_dist = 0.0

    def passes(self, cam_idx: int) -> bool:
        if self.min_dist <= 0.0:
            return True
        if self.selection.is_picked(cam_idx):
            return True
        picked = self.selection.picked_indices()
        if not picked:
            return True
        dists = np.linalg.norm(
            self.positions[picked] - self.positions[cam_idx], axis=1,
        )
        return float(dists.min()) >= self.min_dist

    def reset(self) -> None:
        self.min_dist = 0.0

    def to_state(self) -> dict:
        return {"min_dist": float(self.min_dist)}

    def from_state(self, state: dict) -> None:
        self.min_dist = float(state.get("min_dist", 0.0))


class LookAtTargetFilter:
    """Show only cameras whose forward axis passes within `radius` of `target`.

    Geometric test: closest distance from `target` to the ray defined by
    (camera_position, forward_axis). If less than radius, the camera is
    "looking at" the target.

    target=None disables the filter (everything passes); set via the
    set_target() method when the user moves the gizmo.
    """

    def __init__(self, positions: np.ndarray, forward_axes: np.ndarray):
        self.name = "look_at_target"
        self.enabled = False  # off by default; toggled when user enables gizmo
        self.positions = positions
        self.forward_axes = forward_axes
        self.target: np.ndarray | None = None
        self.radius = 1.0  # in scene meters (whatever SfM's scale is)

    def set_target(self, target: np.ndarray | None) -> None:
        self.target = target

    def passes(self, cam_idx: int) -> bool:
        if not self.enabled or self.target is None:
            return True
        cam_pos = self.positions[cam_idx]
        fwd = self.forward_axes[cam_idx]
        # closest distance from `target` to the ray cam_pos + t*fwd, t>=0
        to_target = self.target - cam_pos
        t = float(np.dot(to_target, fwd))
        if t < 0:
            # target is behind the camera
            return False
        closest = cam_pos + t * fwd
        return bool(np.linalg.norm(self.target - closest) <= self.radius)

    def reset(self) -> None:
        self.enabled = False
        self.target = None
        self.radius = 1.0

    def to_state(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "target": None if self.target is None else [float(x) for x in self.target],
            "radius": float(self.radius),
        }

    def from_state(self, state: dict) -> None:
        self.enabled = bool(state.get("enabled", False))
        t = state.get("target")
        self.target = None if t is None else np.array(t, dtype=float)
        self.radius = float(state.get("radius", 1.0))


# ---------------------------------------------------------------------------
# Content filters
# ---------------------------------------------------------------------------

class ImageSimilarityFilter:
    """Cheap ASMK stand-in: pHash-based dedupe of near-identical frames.

    For each pair of cameras, compute Hamming distance between their
    perceptual hashes. A camera is "redundant" if any visible-and-passing
    earlier camera (lower index) has phash distance < threshold to it.

    threshold=0 disables the filter.

    Note: this is a v1 stand-in. The MASt3R-encoder + ASMK approach the
    SfM paper uses gives content-aware similarity; pHash is purely
    image-pixel-level and miss content-aware dupes (e.g., camera moved
    slightly but still sees the same scene). Defer to ASMK if this isn't
    enough.
    """

    def __init__(self, phashes: np.ndarray):
        self.name = "image_similarity"
        self.enabled = True
        self.phashes = phashes  # (N,) uint64 hash values
        self.threshold = 0      # Hamming distance; 0 = disabled

    def passes(self, cam_idx: int) -> bool:
        if self.threshold <= 0:
            return True
        # cam passes if no earlier-indexed camera is within threshold
        for j in range(cam_idx):
            if int(_hamming_distance(self.phashes[cam_idx], self.phashes[j])) < self.threshold:
                return False
        return True

    def reset(self) -> None:
        self.threshold = 0

    def to_state(self) -> dict:
        return {"threshold": int(self.threshold)}

    def from_state(self, state: dict) -> None:
        self.threshold = int(state.get("threshold", 0))


def _hamming_distance(a: np.uint64, b: np.uint64) -> int:
    """Hamming distance between two 64-bit integers."""
    return int(bin(int(a) ^ int(b)).count("1"))


# ---------------------------------------------------------------------------
# Picked-status filter
# ---------------------------------------------------------------------------

class PickedStatusFilter:
    """Tri-state: show all (default) / picked only / unpicked only."""

    SHOW_ALL = "all"
    SHOW_PICKED = "picked"
    SHOW_UNPICKED = "unpicked"

    def __init__(self, selection: "SelectionState"):
        self.name = "picked_status"
        self.enabled = True
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

    def to_state(self) -> dict:
        return {"mode": str(self.mode)}

    def from_state(self, state: dict) -> None:
        mode = state.get("mode", self.SHOW_ALL)
        if mode in (self.SHOW_ALL, self.SHOW_PICKED, self.SHOW_UNPICKED):
            self.mode = mode


# ---------------------------------------------------------------------------
# Filter stack
# ---------------------------------------------------------------------------

class FilterStack:
    """Composes a list of filters. A camera is visible iff every filter passes it.

    Filters with `enabled=False` are skipped. Filter ordering is preserved
    so callers can ask for them by name.
    """

    def __init__(self, filters: Sequence[Filter]):
        self.filters = list(filters)

    def visible(self, cam_idx: int) -> bool:
        return all(f.passes(cam_idx) for f in self.filters if f.enabled)

    def visible_indices(self, n: int) -> list[int]:
        return [i for i in range(n) if self.visible(i)]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def get(self, name: str) -> Filter:
        for f in self.filters:
            if f.name == name:
                return f
        raise KeyError(f"no filter named {name!r}")

    def has_selection_dependent_filter(self) -> bool:
        """True if any active filter's passes() depends on selection state.

        Used by the click handler to decide whether to refresh ALL frustums
        after a selection toggle (because their visibility may have changed)
        or just the clicked one.
        """
        return any(
            f.enabled and isinstance(f, (PickedStatusFilter, DistanceFromSelectionFilter))
            for f in self.filters
        )

    def to_state(self) -> dict:
        return {f.name: f.to_state() for f in self.filters}

    def from_state(self, state: dict) -> None:
        for f in self.filters:
            if f.name in state:
                f.from_state(state[f.name])


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------

class SelectionState:
    """The set of camera indices the user has picked.

    Persistent across filter changes. The viewer's "Save" button serializes
    this to selected_frames.json.

    `lock_picks` mode: when True, toggle() is a no-op for already-picked
    cameras. Prevents accidental unpicking during dense interaction.
    """

    def __init__(self, n: int):
        self.n = n
        self._picked: set[int] = set()
        self.lock_picks = False

    def toggle(self, cam_idx: int) -> bool:
        """Flip pick state. Returns the new state.

        With lock_picks=True, cannot unpick already-picked cameras.
        Returns the (possibly unchanged) state.
        """
        if cam_idx in self._picked:
            if self.lock_picks:
                return True  # locked; pick remains
            self._picked.remove(cam_idx)
            return False
        self._picked.add(cam_idx)
        return True

    def select_indices(self, indices) -> int:
        """Add multiple indices to the picked set. Idempotent on already-picked.

        Returns the number of *newly* picked cameras (excludes those that
        were already picked).
        """
        n_added = 0
        for i in indices:
            i = int(i)
            if i not in self._picked:
                self._picked.add(i)
                n_added += 1
        return n_added

    def deselect_indices(self, indices) -> int:
        """Remove multiple indices from the picked set.

        Respects `lock_picks` — when locked, this is a no-op (returns 0).
        If you want to bulk-deselect while a lock is active, unlock first.

        Returns the number of cameras actually unpicked.
        """
        if self.lock_picks:
            return 0
        n_removed = 0
        for i in indices:
            i = int(i)
            if i in self._picked:
                self._picked.discard(i)
                n_removed += 1
        return n_removed

    def is_picked(self, cam_idx: int) -> bool:
        return cam_idx in self._picked

    def picked_indices(self) -> list[int]:
        return sorted(self._picked)

    def count(self) -> int:
        return len(self._picked)

    def clear(self) -> None:
        self._picked.clear()

    def to_state(self) -> dict:
        return {
            "picked_idx": self.picked_indices(),
            "lock_picks": bool(self.lock_picks),
        }

    def from_state(self, state: dict) -> None:
        self._picked = {int(i) for i in state.get("picked_idx", [])}
        self.lock_picks = bool(state.get("lock_picks", False))
