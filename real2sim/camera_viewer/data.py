"""Load MASt3R-SfM cameras.json + source frames, expose as numpy arrays.

The cameras.json schema is the bare minimum produced by `train.py --sfm_only`:

    {
      "filepaths":  [str × N],     # absolute paths to source frames
      "focals":     [float × N],   # one focal per frame (in pixels)
      "cams2world": [4×4 × N]      # cam-to-world transform per frame
    }

This module decouples the rest of the viewer from the JSON schema.
If the schema ever grows fields (principal points, distortion, etc.),
this is the only file that needs to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass
class CameraSet:
    """All the per-camera data the viewer needs.

    Index `i` consistently refers to the i-th camera across every array.
    Camera positions and rotations come straight from cams2world.
    """

    filepaths: Sequence[Path]
    focals: np.ndarray            # (N,) float64, focal length in pixels (assumed 512-px image)
    cams2world: np.ndarray        # (N, 4, 4) float64
    image_sizes: np.ndarray       # (N, 2) int   width, height of each source image
    thumbnails: list[np.ndarray]  # (N,) list of (H, W, 3) uint8 — downscaled for the viewer

    @property
    def n(self) -> int:
        return len(self.filepaths)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) camera centers in world coords."""
        return self.cams2world[:, :3, 3]

    @property
    def rotations(self) -> np.ndarray:
        """(N, 3, 3) camera rotations (cam-to-world)."""
        return self.cams2world[:, :3, :3]

    @property
    def forward_axes(self) -> np.ndarray:
        """(N, 3) world-space forward axis of each camera.

        **MASt3R-SfM uses the OpenCV camera convention: cameras look along
        +Z in their own frame.** So world-space forward is the +Z column
        of the camera-to-world rotation (= rotations[:, :, 2]).

        This matters for the LookAtTargetFilter and any view-direction
        clustering. Don't flip the sign — viser's frustum renderer also
        uses OpenCV, which is why frustums look right but earlier
        forward-axis logic (when sign-flipped) had the look-at filter
        inverted (cameras facing AWAY from the target were considered
        as looking AT it).
        """
        return self.rotations[:, :, 2]

    def estimate_up(self) -> tuple[np.ndarray, float]:
        """World-space 'up' from the cameras' average up-vector.

        Same gravity prior as orient_mesh.py::estimate_gravity_from_cameras
        (canonical math + rationale live there — STO-SCN-044 follow-up):
        OpenCV convention has +Y pointing DOWN in the image, so each
        camera's world up is -Y column of its cam-to-world rotation.

        Returns (unit_up, confidence). Confidence is the magnitude of the
        mean per-camera unit-up (circular statistics): 1.0 = all cameras
        agree, →0 = rolls cancel out (e.g. banked orbit captures) and the
        prior shouldn't be trusted.
        """
        per_camera_up = -self.rotations[:, :, 1]
        per_camera_up = per_camera_up / np.linalg.norm(
            per_camera_up, axis=1, keepdims=True
        )
        mean_up = per_camera_up.mean(axis=0)
        confidence = float(np.linalg.norm(mean_up))
        if confidence < 1e-6:
            return np.array([0.0, 0.0, 1.0]), 0.0
        return mean_up / confidence, confidence


def load(cameras_json: Path, frames_dir: Path | None = None,
         thumbnail_long_edge: int = 512) -> CameraSet:
    """Load a cameras.json + the referenced images.

    Args:
        cameras_json: path to the JSON file written by MAtCha's --sfm_only.
        frames_dir: optional override. If provided, filepaths in the JSON
            are re-rooted under this directory (useful when the JSON was
            produced on a different host with different mount paths).
        thumbnail_long_edge: longest edge in pixels for the in-viewer image
            planes. 512 keeps texture memory manageable for ~400 cameras.

    Returns:
        CameraSet ready to feed into the viser scene composition.
    """
    with cameras_json.open() as f:
        d = json.load(f)

    filepaths_raw = d["filepaths"]
    focals = np.asarray(d["focals"], dtype=np.float64)
    cams2world = np.asarray(d["cams2world"], dtype=np.float64)

    # Re-root paths if requested (the typical bbeeprz → JDP-Mac case)
    filepaths: list[Path]
    if frames_dir is not None:
        filepaths = [frames_dir / Path(p).name for p in filepaths_raw]
    else:
        filepaths = [Path(p) for p in filepaths_raw]

    # Sanity check: all images exist
    missing = [p for p in filepaths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(filepaths)} frames not found. "
            f"First missing: {missing[0]}. "
            f"Pass --frames to point at the right directory."
        )

    # Load + downscale each image once
    thumbnails: list[np.ndarray] = []
    image_sizes = np.zeros((len(filepaths), 2), dtype=np.int32)
    for i, p in enumerate(filepaths):
        img = Image.open(p).convert("RGB")
        image_sizes[i] = img.size  # (w, h) — note PIL convention
        long_edge = max(img.size)
        if long_edge > thumbnail_long_edge:
            scale = thumbnail_long_edge / long_edge
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        thumbnails.append(np.asarray(img))

    return CameraSet(
        filepaths=filepaths,
        focals=focals,
        cams2world=cams2world,
        image_sizes=image_sizes,
        thumbnails=thumbnails,
    )
