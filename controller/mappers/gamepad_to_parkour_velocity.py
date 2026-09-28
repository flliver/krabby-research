"""Map ``ControllerState`` (pygame SDL2) to parkour ``base_velocity`` commands.

Uses the same stick conventions as ``controller.input.InputController`` (Pro Controller
via SDL game-controller DB). Intended for Isaac Sim play/demo scripts that drive
``UniformParkourCommand.vel_command_b`` and ``heading_target``.
"""

from __future__ import annotations

from controller.input.state import ControllerState


def _deadzone_axis(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - deadzone) / max(1.0 - deadzone, 1e-6)


def controller_state_to_velocity_command(
    state: ControllerState,
    lin_vel_x_range: tuple[float, float],
    *,
    lin_vel_scale: float = 1.0,
    yaw_heading_scale: float = 1.6,
    deadzone: float = 0.05,
    lin_vel_clip: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(vx, vy, heading_target)`` for parkour velocity command.

    - **Forward:** left stick Y — SDL ``LY`` is negative when pushed up → ``forward = max(0, -LY)``.
      Centered stick → ``vx = 0`` (idle). Full forward → ``lin_vel_x`` max.
    - **Yaw:** right stick X → ``heading_target`` (parkour ``_update_command`` sets ``omega_z``).
    """
    ly = _deadzone_axis(state.LY, deadzone)
    rx = _deadzone_axis(state.RX, deadzone)
    lo, hi = lin_vel_x_range
    forward = max(0.0, min(1.0, -ly))
    if forward <= lin_vel_clip:
        vx = 0.0
    else:
        vx = lo + lin_vel_scale * forward * (hi - lo)
    heading_target = yaw_heading_scale * max(-1.0, min(1.0, rx))
    return vx, 0.0, heading_target


def controller_state_to_teleop_yaw(
    state: ControllerState,
    *,
    yaw_heading_scale: float = 1.6,
    max_yaw_rate: float = 1.0,
    deadzone: float = 0.05,
    ang_vel_clip: float = 0.0,
) -> tuple[float, float]:
    """Return ``(delta_yaw_obs, omega_z)`` for student teleop (rate + proprio turn hint).

  - **omega_z:** body-frame yaw rate command (``vel_command_b[:, 2]``), clipped by ``ang_vel_clip``.
  - **delta_yaw:** injected into proprio index 7 (same scale as parkour goal-yaw error).
    """
    rx = _deadzone_axis(state.RX, deadzone)
    rx = max(-1.0, min(1.0, rx))
    delta_yaw = yaw_heading_scale * rx
    omega_z = max_yaw_rate * rx
    if abs(omega_z) <= ang_vel_clip:
        omega_z = 0.0
    return delta_yaw, omega_z


def apply_controller_state_to_parkour_command(
    cmd_term,
    state: ControllerState,
    lin_vel_x_range: tuple[float, float],
    *,
    lin_vel_scale: float = 1.0,
    yaw_heading_scale: float = 1.6,
    deadzone: float = 0.05,
    lin_vel_clip: float = 0.0,
) -> None:
    """Write gamepad state into ``UniformParkourCommand`` buffers."""
    vx, vy, heading_target = controller_state_to_velocity_command(
        state,
        lin_vel_x_range,
        lin_vel_scale=lin_vel_scale,
        yaw_heading_scale=yaw_heading_scale,
        deadzone=deadzone,
        lin_vel_clip=lin_vel_clip,
    )
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.heading_target[:] = heading_target


def apply_teleop_state_to_parkour_command(
    cmd_term,
    state: ControllerState,
    lin_vel_x_range: tuple[float, float],
    *,
    lin_vel_scale: float = 1.0,
    yaw_heading_scale: float = 1.6,
    max_yaw_rate: float = 1.0,
    deadzone: float = 0.05,
    lin_vel_clip: float = 0.0,
    ang_vel_clip: float = 0.0,
) -> tuple[float, float]:
    """Write forward speed + yaw **rate** (not absolute heading) for student play teleop."""
    vx, vy, _ = controller_state_to_velocity_command(
        state,
        lin_vel_x_range,
        lin_vel_scale=lin_vel_scale,
        yaw_heading_scale=yaw_heading_scale,
        deadzone=deadzone,
        lin_vel_clip=lin_vel_clip,
    )
    delta_yaw, omega_z = controller_state_to_teleop_yaw(
        state,
        yaw_heading_scale=yaw_heading_scale,
        max_yaw_rate=max_yaw_rate,
        deadzone=deadzone,
        ang_vel_clip=ang_vel_clip,
    )
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.vel_command_b[:, 2] = omega_z
    return delta_yaw, omega_z
