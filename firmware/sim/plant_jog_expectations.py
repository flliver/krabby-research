"""Jog isolation expectations for main plant ``crab.usda`` vs ``crab_hex_ref.usd``."""

from __future__ import annotations

from typing import Any

from firmware.sim.isaac_scene import _crab_hex_articulation_uses_ref_joint_names


def _leg_prefix(isaac_joint_name: str) -> str:
    return isaac_joint_name.split("_", 1)[0]


def skip_plant_jog_coupling_check(
    articulation: Any,
    commanded_joint: str,
    other_joint: str,
) -> bool:
    """Return True when ``other_joint`` may move with ``commanded_joint`` on the plant USD.

    Ref USD keeps strict one-DOF isolation. The plant couples cam shaft, passive Body_Hip,
    and the other joints on the same leg (HL/KL/HY share a kinematic tree).
    """
    if other_joint == commanded_joint:
        return True
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        return False
    if "Body_Hip_RevoluteJoint" in other_joint:
        return True
    return _leg_prefix(other_joint) == _leg_prefix(commanded_joint)


def cross_leg_jog_coupling_ratio(articulation: Any) -> float:
    """Scale factor for opposite-leg drift vs commanded travel (front-board jog smoke)."""
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        return 5.0
    return 2.0


def min_commanded_jog_travel_rad(articulation: Any) -> float:
    """Minimum |Δq| on the commanded DOF after a full-PWM jog burst."""
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        return 0.04
    return 0.015


PARTIAL_PWM = 122
FULL_PWM = 255
NOMINAL_PARTIAL_PWM_RATIO = PARTIAL_PWM / FULL_PWM


def max_zero_pwm_joint_drift_rad(
    articulation: Any,
    firmware_actuator_id: str | None = None,
) -> float:
    """|Δq| allowed after a zero-PWM jog burst (plant coupling is looser than ref USD)."""
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        return 0.01
    if firmware_actuator_id is not None and firmware_actuator_id.endswith("HY"):
        return 0.04
    return 0.03


def partial_pwm_ratio_bounds(
    articulation: Any,
    firmware_actuator_id: str,
) -> tuple[float, float]:
    """Acceptable partial/full travel ratio for ``PARTIAL_PWM`` vs ``FULL_PWM``."""
    expected = NOMINAL_PARTIAL_PWM_RATIO
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        margin = 0.03
        return expected - margin, expected + margin
    if firmware_actuator_id.endswith("HY"):
        return 0.15, 0.75
    margin = 0.10
    return expected - margin, expected + margin
