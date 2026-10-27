"""The "travel frame": the body frame rotated by a constant yaw ``travel_yaw``.

``travel_yaw`` is 0 for the forward-walk task (the travel frame is the body frame
itself) and +pi/2 for the sideways task, where the intended travel axis is the
robot's body +y axis rather than body +x. Every helper below returns the SAME
tensor object when ``travel_yaw`` (or the equivalent angle/offset argument) is
exactly ``0.0``, so the forward task -- which always calls these with angle 0.0 --
stays byte-identical to its pre-existing behaviour.
"""

import math

import torch


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wrap input angles (in radians) to the range :math:`[-\pi, \pi]`.

    Copied verbatim (semantics preserved) from ``wrap_to_pi`` in
    ``isaaclab/utils/math.py`` (IsaacLab source, function at line 101):
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


def rotate_xy(v: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate the planar vectors in ``v[..., :2]`` counter-clockwise by ``angle`` radians.

    Only the first two components of the trailing dimension are used.

    Args:
        v: Tensor whose trailing dimension holds at least an (x, y) pair.
        angle: Rotation angle in radians (counter-clockwise, standard math convention).

    Returns:
        A new tensor of shape ``v.shape[:-1] + (2,)`` with the rotated (x, y) pair.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    x = v[..., 0]
    y = v[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def vel_travel_xy(vel_b_xy: torch.Tensor, travel_yaw: float) -> torch.Tensor:
    """Express a body-frame xy velocity in the travel frame.

    At ``travel_yaw == 0.0`` this is the identity (returns ``vel_b_xy`` itself).
    At ``travel_yaw == +pi/2`` this maps ``(vx, vy) -> (vy, -vx)``, i.e. travel
    speed equals body +y velocity.

    Args:
        vel_b_xy: Body-frame xy velocity, shape ``(..., 2)`` (or more, only the
            first two components are used).
        travel_yaw: Constant yaw offset of the travel frame from the body frame.

    Returns:
        The xy velocity expressed in the travel frame.
    """
    if travel_yaw == 0.0:
        return vel_b_xy
    return rotate_xy(vel_b_xy, -travel_yaw)


def yaw_error(target_yaw: torch.Tensor, yaw_wrapped: torch.Tensor, travel_yaw: float) -> torch.Tensor:
    """Compute the heading error against a target yaw, accounting for travel-frame offset.

    At ``travel_yaw == 0.0`` this returns exactly ``target_yaw - yaw_wrapped`` --
    today's forward-task expression, unwrapped, unchanged. Otherwise the travel-yaw
    offset is subtracted and the result is wrapped to :math:`[-\\pi, \\pi]`.

    Args:
        target_yaw: Target yaw (radians).
        yaw_wrapped: Current (already wrapped) yaw (radians).
        travel_yaw: Constant yaw offset of the travel frame from the body frame.

    Returns:
        The yaw error.
    """
    if travel_yaw == 0.0:
        return target_yaw - yaw_wrapped
    return wrap_to_pi(target_yaw - yaw_wrapped - travel_yaw)


def heading_with_offset(heading: torch.Tensor, offset: float) -> torch.Tensor:
    """Add a constant offset to a heading tensor.

    At ``offset == 0.0`` this is the identity (returns ``heading`` itself).

    Args:
        heading: Heading tensor (radians).
        offset: Constant offset to add (radians).

    Returns:
        ``heading + offset``, or ``heading`` itself when ``offset == 0.0``.
    """
    if offset == 0.0:
        return heading
    return heading + offset


def track_lin_vel_exp(cmd_xy: torch.Tensor, vel_b_xy: torch.Tensor, std: float, travel_yaw: float) -> torch.Tensor:
    """Exponential-kernel linear-velocity tracking reward, evaluated in the travel frame.

    Reproduces the IsaacLab ``track_lin_vel_xy_exp`` formula (IsaacLab source,
    ``isaaclab/envs/mdp/rewards.py`` line 304-315), with the body-frame velocity
    first rotated into the travel frame via :func:`vel_travel_xy`:
    ``lin_vel_error = sum((cmd - vel) ** 2, dim=1); exp(-lin_vel_error / std**2)``.

    Args:
        cmd_xy: Commanded xy velocity in the travel frame, shape ``(N, 2)``.
        vel_b_xy: Body-frame xy velocity, shape ``(N, 2)``.
        std: Standard deviation of the exponential kernel.
        travel_yaw: Constant yaw offset of the travel frame from the body frame.

    Returns:
        The per-environment tracking reward, shape ``(N,)``.
    """
    vel = vel_travel_xy(vel_b_xy, travel_yaw)
    lin_vel_error = torch.sum(torch.square(cmd_xy - vel), dim=1)
    return torch.exp(-lin_vel_error / std**2)
