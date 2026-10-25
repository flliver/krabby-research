"""Map HAL ``JointCommand`` radians to firmware jog PWM (Jetson-compatible)."""

from __future__ import annotations

# Keep in sync with hal/server/jetson/krabby_mcusdk.py (avoid importing Jetson HAL here).
JOINT_LIMIT_RAD = 0.5
NEUTRAL_RAD_THRESHOLD = 0.01
PWM_SCALE = 255 / JOINT_LIMIT_RAD
PWM_COMMAND_FULL_SCALE = 255

_HAL_TO_FW_SUFFIX = {"hip_yaw": "HY", "hip_pitch": "HL", "knee": "KL"}


def hal_joint_to_firmware_id(hal_name: str) -> str:
    leg, _, suffix = hal_name.partition("_")
    return leg + _HAL_TO_FW_SUFFIX.get(
        suffix, suffix[:2].upper() if len(suffix) >= 2 else "??"
    )


def rad_to_jog_pwm(rad: float) -> int:
    """Neutral band → hold (0 PWM); otherwise scale radians like the Jetson HAL SDK."""
    if abs(rad) <= NEUTRAL_RAD_THRESHOLD:
        return 0
    return max(-255, min(255, int(round(rad * PWM_SCALE))))


def joint_radians_to_jog_pwm(
    cmd_dict: dict[str, float],
    mcu_joint_names: tuple[str, ...],
) -> dict[str, int]:
    """Build firmware-id → PWM map for batch jog."""
    jog: dict[str, int] = {}
    for name in mcu_joint_names:
        rad = cmd_dict.get(name, 0.0)
        jog[hal_joint_to_firmware_id(name)] = rad_to_jog_pwm(rad)
    return jog


def signed_pwm_to_normalized_action(pwm: int) -> float:
    """Map firmware drive command to Isaac HAL action scale (−1..1)."""
    clamped = max(-PWM_COMMAND_FULL_SCALE, min(PWM_COMMAND_FULL_SCALE, int(pwm)))
    return clamped / float(PWM_COMMAND_FULL_SCALE)
