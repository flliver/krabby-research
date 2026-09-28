"""Unit tests for teleop cockpit telemetry payloads."""

from __future__ import annotations

import math

import numpy as np

from teleop.edge.telemetry import (
    TELEMETRY_MESSAGE_TYPE,
    build_telemetry_payload,
    quat_xyzw_to_euler_deg,
)


def test_quat_identity_is_zero_euler() -> None:
    roll, pitch, yaw = quat_xyzw_to_euler_deg(np.array([0.0, 0.0, 0.0, 1.0]))
    assert abs(roll) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6


def test_build_telemetry_payload_speed_and_orientation() -> None:
    q = np.array([0.0, 0.0, 0.0, 1.0])
    lin = np.array([3.0, 4.0, 0.0])
    ang = np.array([0.1, 0.2, 0.3])
    payload = build_telemetry_payload(
        timestamp_ns=1_000_000_000,
        base_quat_w=q,
        base_ang_vel_b=ang,
        base_lin_vel_b=lin,
    )
    assert payload["type"] == TELEMETRY_MESSAGE_TYPE
    assert payload["timestamp_ns"] == 1_000_000_000
    assert payload["velocity"]["speed_m_s"] == 5.0
    assert payload["velocity"]["horizontal_speed_m_s"] == 5.0
    assert payload["velocity"]["linear_m_s"] == [3.0, 4.0, 0.0]
    assert payload["velocity"]["angular_rad_s"] == [0.1, 0.2, 0.3]
    orient = payload["orientation_deg"]
    assert abs(orient["roll"]) < 1e-6
    assert abs(orient["pitch"]) < 1e-6
    assert abs(orient["yaw"]) < 1e-6


def test_build_telemetry_payload_horizontal_speed_ignores_z() -> None:
    lin = np.array([3.0, 4.0, 12.0])
    payload = build_telemetry_payload(
        timestamp_ns=0,
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0]),
        base_ang_vel_b=np.zeros(3),
        base_lin_vel_b=lin,
    )
    assert payload["velocity"]["speed_m_s"] == math.sqrt(3.0**2 + 4.0**2 + 12.0**2)
    assert payload["velocity"]["horizontal_speed_m_s"] == 5.0
