"""HAL observation → browser telemetry payloads (cockpit HUD)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TELEMETRY_CHANNEL_LABEL = "krabby-telemetry-v1"
TELEMETRY_MESSAGE_TYPE = "telemetry"


def quat_xyzw_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw in degrees from quaternion (x, y, z, w), ROS right-handed."""
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def build_telemetry_payload(
    *,
    timestamp_ns: int,
    base_quat_w: np.ndarray,
    base_ang_vel_b: np.ndarray,
    base_lin_vel_b: np.ndarray,
) -> dict[str, Any]:
    """JSON-serializable robot state for the operator cockpit."""
    q = np.asarray(base_quat_w, dtype=np.float64).reshape(4)
    lin = np.asarray(base_lin_vel_b, dtype=np.float64).reshape(3)
    ang = np.asarray(base_ang_vel_b, dtype=np.float64).reshape(3)
    roll_deg, pitch_deg, yaw_deg = quat_xyzw_to_euler_deg(q)
    speed = float(np.linalg.norm(lin))
    horizontal = float(math.hypot(float(lin[0]), float(lin[1])))

    return {
        "type": TELEMETRY_MESSAGE_TYPE,
        "timestamp_ns": int(timestamp_ns),
        "quaternion": {
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "w": float(q[3]),
        },
        "orientation_deg": {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "yaw": yaw_deg,
        },
        "velocity": {
            "linear_m_s": [float(lin[0]), float(lin[1]), float(lin[2])],
            "angular_rad_s": [float(ang[0]), float(ang[1]), float(ang[2])],
            "speed_m_s": speed,
            "horizontal_speed_m_s": horizontal,
        },
    }
