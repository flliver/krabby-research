"""Spatial clustering + content-similarity hashing for filter inputs.

v0: k-means on 3D camera centers, six-bucket view-direction grouping.
v1: pHash perceptual hashes for the cheap image-similarity filter.

Future (v2): co-visibility clustering using the SfM scene graph. The graph
edges from MASt3R-SfM tell us which camera pairs share enough correspondences
that they're seeing the same physical region. That's a more principled
"these cameras see the same place" signal than position-only k-means.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

# Shared, dependency-free pHash lives one level up (real2sim/phash.py),
# replacing the former `imagehash` dependency (STO-SCN-092).
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from phash import phash as _phash  # noqa: E402


def kmeans_position_clusters(positions: np.ndarray, k: int | None = None) -> np.ndarray:
    """Cluster camera centers by position via k-means.

    Args:
        positions: (N, 3) camera centers in world coords.
        k: number of clusters. If None, picks `max(2, min(8, N // 8))`.

    Returns:
        (N,) integer cluster labels in [0, k).
    """
    n = len(positions)
    if k is None:
        k = max(2, min(8, n // 8))
    if k >= n:
        k = max(2, n - 1)

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    return km.fit_predict(positions)


def view_direction_buckets(forward_axes: np.ndarray) -> np.ndarray:
    """Bucket cameras into 6 axis-aligned view-direction groups.

    Used by the (v1) ViewDirectionFilter. Buckets by which world axis the
    camera's forward direction is most aligned with: ±X / ±Y / ±Z.

    Args:
        forward_axes: (N, 3) unit vectors pointing where each camera looks.

    Returns:
        (N,) integer bucket labels in [0, 6):
        0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z
    """
    abs_axes = np.abs(forward_axes)
    primary = abs_axes.argmax(axis=1)               # 0/1/2 = x/y/z
    sign_pos = forward_axes[np.arange(len(forward_axes)), primary] > 0
    return primary * 2 + (~sign_pos).astype(int)


def compute_phashes(thumbnails: Sequence[np.ndarray]) -> np.ndarray:
    """Compute 64-bit perceptual hashes (pHash) for each thumbnail.

    Used by the ImageSimilarityFilter as a cheap stand-in for ASMK.
    pHash represents an image as a 64-bit fingerprint; two images are
    "similar" if their hashes have low Hamming distance.

    Args:
        thumbnails: list of (H, W, 3) uint8 arrays.

    Returns:
        (N,) uint64 array of hash values.

    Uses the shared dependency-free DCT pHash (real2sim/phash.py) — same
    32x32 -> DCT -> 8x8 -> median-threshold fingerprint the former imagehash
    path produced, with no external dependency (STO-SCN-092).
    """
    hashes = np.empty(len(thumbnails), dtype=np.uint64)
    for i, thumb in enumerate(thumbnails):
        hashes[i] = _phash(thumb)
    return hashes
