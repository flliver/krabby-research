"""Camera→body mount rotation for the ZED IMU path.

The ZED 2i's IMU reports in the *camera's* coordinate frame. The camera is
mounted on the robot at some fixed pose, so a constant rotation matrix
``R_camera_to_body`` maps camera-frame vectors into the robot body frame:

    x_body = R_camera_to_body @ x_camera

The matrix is loaded from a YAML config (``config/zed_mount.yaml`` next to
this module, overridable via the ``KRABBY_ZED_MOUNT_YAML`` env var). Default
is identity — correct only for a camera rigidly axis-aligned with the body.
The assumed mount pose is documented in the YAML; update the matrix there
when the physical mount changes (M15 Task 2 refines it against the V0.2
chassis).

Quaternion convention throughout: (x, y, z, w), matching
``HardwareObservations.base_quat_w``.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ZED_MOUNT_YAML_ENV = "KRABBY_ZED_MOUNT_YAML"
DEFAULT_MOUNT_YAML = Path(__file__).parent / "config" / "zed_mount.yaml"

IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def load_camera_to_body_rotation(path: Optional[Path] = None) -> np.ndarray:
    """Load R_camera_to_body from YAML; identity if no config exists.

    Resolution order: explicit ``path`` arg, ``KRABBY_ZED_MOUNT_YAML`` env var,
    then the packaged ``config/zed_mount.yaml``. A missing file yields identity
    (flat axis-aligned mount); a file that exists but is invalid raises — a
    wrong rotation silently corrupts every IMU observation, so fail at startup.

    Returns:
        (3, 3) float32 rotation matrix.
    """
    if path is None:
        env_path = os.environ.get(ZED_MOUNT_YAML_ENV, "").strip()
        path = Path(env_path) if env_path else DEFAULT_MOUNT_YAML
    if not path.exists():
        logger.info(
            "ZED mount config %s not found; using identity camera→body rotation "
            "(assumes camera axes aligned with body axes)",
            path,
        )
        return np.eye(3, dtype=np.float32)

    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "r_camera_to_body" not in data:
        raise RuntimeError(f"ZED mount config {path}: missing 'r_camera_to_body' key")
    rot = np.asarray(data["r_camera_to_body"], dtype=np.float32)
    _validate_rotation_matrix(rot, source=str(path))
    logger.info("Loaded ZED camera→body rotation from %s:\n%s", path, rot)
    return rot


def _validate_rotation_matrix(rot: np.ndarray, source: str) -> None:
    """Raise RuntimeError unless rot is a proper 3×3 rotation (orthonormal, det +1)."""
    if rot.shape != (3, 3):
        raise RuntimeError(f"ZED mount config {source}: r_camera_to_body must be 3x3, got {rot.shape}")
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-5):
        raise RuntimeError(f"ZED mount config {source}: r_camera_to_body is not orthonormal")
    if not np.isclose(np.linalg.det(rot), 1.0, atol=1e-5):
        raise RuntimeError(
            f"ZED mount config {source}: r_camera_to_body has det {np.linalg.det(rot):.4f}, "
            "expected +1 (a reflection is not a valid mount rotation)"
        )


def quat_xyzw_to_rot_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (x, y, z, w) to a (3, 3) rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert a (3, 3) rotation matrix to a unit quaternion (x, y, z, w).

    Shepperd's method: branch on the largest of trace/diagonal terms for
    numerical stability.
    """
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def apply_camera_to_body(
    r_camera_to_body: np.ndarray,
    ang_vel_camera: np.ndarray,
    orientation_quat_xyzw_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an IMU sample from the camera frame into the robot body frame.

    Angular velocity is a plain vector rotation. The orientation quaternion is
    the camera's attitude in the world (R_world_camera); the body's attitude is
    R_world_body = R_world_camera @ R_camera_to_body.T, since
    x_camera = R_camera_to_body.T @ x_body.

    Returns:
        (ang_vel_body (3,) float32, orientation_quat_xyzw_body (4,) float32)
    """
    ang_vel_body = (r_camera_to_body @ np.asarray(ang_vel_camera, dtype=np.float64)).astype(np.float32)
    r_world_camera = quat_xyzw_to_rot_matrix(orientation_quat_xyzw_camera)
    r_world_body = r_world_camera @ np.asarray(r_camera_to_body, dtype=np.float64).T
    return ang_vel_body, rot_matrix_to_quat_xyzw(r_world_body)
