# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch math for the omnidirectional command (omni plan D1).

No isaaclab imports: every function here runs on CPU in
``tests/unit/test_crab_hex_omni_command.py``, and the shipped command term
(:mod:`crab_hex_omni_command`) calls exactly these functions -- the math the tests pin is
the math that trains.

FRAMES AND SIGNS
----------------
Everything below lives in one of two planar frames, and mixing them up is silent, so they
are spelled out once:

* **World frame** -- right-handed, ``z`` up. A *world direction* is an angle measured
  counter-clockwise from world ``+x``. ``theta_w`` (the travel direction) and ``yaw`` /
  ``heading_target`` (facing) are world angles.
* **Body frame** -- body ``+x`` is forward, body ``+y`` is the robot's **left**, ``+z`` up.
  ``yaw`` is the world angle of body ``+x`` (``robot.data.heading_w``, already wrapped to
  ``[-pi, pi]``).

A world direction ``theta_w`` seen from the body is

    ``phi = wrap_to_pi(theta_w - yaw)``

so ``phi = 0`` is straight ahead, ``phi = +pi/2`` is the robot's left (body ``+y``),
``phi = pi`` is straight back and ``phi = -pi/2`` is the robot's right. Rotating the body
counter-clockwise (increasing ``yaw``) *decreases* ``phi`` for a held world direction --
that is the whole content of the "clamp-only re-cap" in
:func:`body_command_from_world`.

A positive yaw-rate command turns the body counter-clockwise (towards its left), i.e. it
drives ``yaw`` towards larger values -- hence :func:`heading_rate_command` is a P-control
on ``wrap(heading_target - yaw)`` with **no** sign flip.

ANGLES ARE NOT RE-WRAPPED
-------------------------
:func:`sample_travel_dir` and :func:`sample_facing` return world angles that may leave
``[-pi, pi]`` (``yaw + delta``). That is the house convention already used by the forward
command's ``heading_target`` (see ``travel_frame.heading_with_offset``): every consumer
wraps the *difference*, and not wrapping here keeps the body-relative bands exact
(``dir_band = (0, 0)`` gives ``phi == 0.0`` bitwise, not ``1e-16``).
"""

from __future__ import annotations

import math

import torch

# ------------------------------------------------------------------ command-regime slots
#
# The four slots of the omni command regime. The encoding is deliberately a 2-bit mask --
# bit 0 = "moving", bit 1 = "turning" -- so ``slot = moving + 2 * turning`` and the two
# predicates below are the inverse. Every consumer (metrics, telemetry, the eval harness)
# uses these names rather than bare integers.
SLOT_STAND = 0
SLOT_MOVE = 1
SLOT_TURN = 2
SLOT_MOVE_TURN = 3
SLOT_NAMES = ("stand", "move", "turn", "move_turn")
N_SLOTS = 4


def slot_is_moving(slot: torch.Tensor) -> torch.Tensor:
    """True where the slot commands a non-zero travel speed (slots 1 and 3)."""
    return (slot & 1) != 0


def slot_is_turning(slot: torch.Tensor) -> torch.Tensor:
    """True where the slot commands a facing change (slots 2 and 3)."""
    return (slot & 2) != 0


# ------------------------------------------------------------------------------- angles


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wrap input angles (in radians) to the range :math:`[-\pi, \pi]`.

    Byte-identical to ``parkour_isaaclab.envs.mdp.travel_frame.wrap_to_pi`` (itself copied
    from IsaacLab's ``isaaclab/utils/math.py``): :math:`\pi` maps to :math:`\pi` and
    :math:`-\pi` maps to :math:`-\pi`. Copied rather than imported so this module stays
    free of the ``parkour_isaaclab.envs`` import chain (which pulls in Isaac); a unit test
    pins the two implementations against each other on a dense grid.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


# ------------------------------------------------------------------------- the speed cap


def _check_ellipse(a: float, b: float, c: float) -> None:
    if not (a > 0.0 and b > 0.0 and c > 0.0):
        raise ValueError(f"speed_ellipse semi-axes must be positive, got (a={a}, b={b}, c={c})")
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        raise ValueError(f"speed_ellipse semi-axes must be finite, got (a={a}, b={b}, c={c})")


def ellipse_cap(phi: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    r"""Direction-scaled speed cap: the radius of a piecewise ellipse in the body frame.

    The cap is the polar radius of the closed curve

    .. math::
        \frac{x^2}{A^2} + \frac{y^2}{b^2} = 1,
        \qquad A = a \ (x \ge 0), \quad A = c \ (x < 0)

    i.e. two half-ellipses glued along the body ``y`` axis: semi-axis ``a`` forwards
    (``+x``), ``c`` backwards (``-x``) and ``b`` sideways (both ``+y`` and ``-y``). In
    polar form, with :math:`A = a` when :math:`\cos\phi \ge 0` and :math:`A = c` otherwise,

    .. math::
        \mathrm{cap}(\phi) = \frac{A\,b}{\sqrt{(b\cos\phi)^2 + (A\sin\phi)^2}} .

    Properties (all of them pinned by the unit test):

    * ``cap(0) == a``, ``cap(+-pi/2) == b``, ``cap(pi) == cap(-pi) == c``;
    * **continuous everywhere**: at the seam :math:`\cos\phi = 0` both branches evaluate to
      :math:`A b / A = b` regardless of which semi-axis the ``where`` picked, so the float
      sign of ``cos(pi/2)`` is irrelevant;
    * ``min(a, b, c) <= cap(phi) <= max(a, b, c)`` for every ``phi`` (an ellipse radius is
      between its semi-axes) -- this is what makes the walk-floor assertion in
      ``CrabHexOmniCommand.__init__`` sufficient to guarantee a non-empty speed band in
      every direction.

    Args:
        phi: Body-frame direction of travel, radians (any shape). Need not be wrapped.
        a: Fore-aft (body ``+x``) semi-axis, m/s.
        b: Lateral (body ``+-y``) semi-axis, m/s.
        c: Backward (body ``-x``) semi-axis, m/s.

    Returns:
        The maximum commandable speed along ``phi``, same shape/dtype/device as ``phi``.
    """
    _check_ellipse(a, b, c)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    semi_x = torch.where(
        cos_phi >= 0.0,
        torch.full_like(cos_phi, a),
        torch.full_like(cos_phi, c),
    )
    denom = torch.sqrt(torch.square(b * cos_phi) + torch.square(semi_x * sin_phi))
    return semi_x * b / denom


# ------------------------------------------------------ world command -> body command


def body_command_details(
    theta_w: torch.Tensor,
    speed: torch.Tensor,
    yaw: torch.Tensor,
    a: float,
    b: float,
    c: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`body_command_from_world` plus the intermediates the term logs.

    Returns ``(cmd_xy, phi, cap)`` where ``cmd_xy`` is the ``(N, 2)`` body-frame command,
    ``phi`` the body-frame travel direction and ``cap`` the ellipse cap at ``phi``. The
    command term uses this form so the ``recap_frac`` metric costs no second
    :func:`ellipse_cap` evaluation.
    """
    phi = wrap_to_pi(theta_w - yaw)
    cap = ellipse_cap(phi, a, b, c)
    v = torch.minimum(speed, cap)
    cmd_xy = torch.stack((v * torch.cos(phi), v * torch.sin(phi)), dim=-1)
    return cmd_xy, phi, cap


def body_command_from_world(
    theta_w: torch.Tensor,
    speed: torch.Tensor,
    yaw: torch.Tensor,
    a: float,
    b: float,
    c: float,
) -> torch.Tensor:
    """Express a held WORLD travel direction + speed as a body-frame velocity command.

    ``phi = wrap(theta_w - yaw)`` and

        ``(vx_b, vy_b) = min(speed, ellipse_cap(phi)) * (cos phi, sin phi)``.

    **Clamp-only re-cap.** The commanded *world* direction ``theta_w`` is held while the
    body turns; only the magnitude is re-capped, every step, against the cap of the new
    body-frame direction. A command sampled at the forward cap ``a`` and then rotated
    90 deg therefore becomes a pure lateral command of magnitude ``b`` (not ``a``, which the
    plant cannot do sideways) pointing at the SAME world direction -- the speed is
    clamped down, never rotated and never restored when the body turns back.

    Args:
        theta_w: Commanded world travel direction, radians, shape ``(N,)``. Need not be
            wrapped.
        speed: Commanded speed along ``theta_w``, m/s, shape ``(N,)``. ``0`` on standing
            slots (which then produce an exactly zero command, whatever ``theta_w`` is).
        yaw: Current world yaw of body ``+x``, radians, shape ``(N,)``.
        a, b, c: Ellipse semi-axes, see :func:`ellipse_cap`.

    Returns:
        Body-frame ``(vx_b, vy_b)`` command, shape ``(N, 2)``.
    """
    return body_command_details(theta_w, speed, yaw, a, b, c)[0]


def heading_rate_command(
    heading_target: torch.Tensor,
    yaw: torch.Tensor,
    stiffness: float,
    ang_vel_max: float,
    ang_vel_clip: float,
) -> torch.Tensor:
    """Yaw-rate command from an absolute world facing target (P-control + dead band).

    This is ``UniformParkourCommand._update_command``'s expression with ONE change: the
    saturation is ``+-ang_vel_max`` instead of the hard-coded ``+-1`` rad/s. The order is
    preserved exactly -- scale, then saturate, then apply the dead band to the *saturated*
    value:

        ``wz = clip(wrap_to_pi(heading_target - yaw) * stiffness, +-ang_vel_max)``
        ``wz = wz * (|wz| > ang_vel_clip)``

    The dead band is a strict ``>`` (a value exactly at the clip is zeroed), matching
    ``uniform_parkour_command.py:107``. With the omni defaults (``stiffness`` 1.5,
    ``ang_vel_clip`` 0.2) the band covers facing errors below 0.1333 rad; with
    ``ang_vel_max`` 0.3 > ``ang_vel_clip`` a saturated command always survives the band.

    Sign: positive ``wz`` turns the body counter-clockwise, increasing ``yaw`` towards
    ``heading_target``.

    Args:
        heading_target: Target world yaw, radians, shape ``(N,)``. Need not be wrapped.
        yaw: Current world yaw, radians, shape ``(N,)``.
        stiffness: ``cfg.heading_control_stiffness`` (rad/s per rad).
        ang_vel_max: Saturation of the yaw-rate command, rad/s (``cfg.ang_vel_max``).
        ang_vel_clip: Dead band on the yaw-rate command, rad/s
            (``cfg.clips.ang_vel_clip``).

    Returns:
        The yaw-rate command, shape ``(N,)``.
    """
    if not (ang_vel_max > 0.0):
        raise ValueError(f"ang_vel_max must be positive, got {ang_vel_max}")
    if ang_vel_clip < 0.0:
        raise ValueError(f"ang_vel_clip must be non-negative, got {ang_vel_clip}")
    wz = torch.clip(
        wrap_to_pi(heading_target - yaw) * stiffness,
        min=-ang_vel_max,
        max=ang_vel_max,
    )
    return wz * (torch.abs(wz) > ang_vel_clip)


# ------------------------------------------------------------------------- the samplers


def sample_slots(
    generator: torch.Generator | None,
    probs,
    n: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Draw ``n`` command-regime slots from a categorical distribution.

    Consumes exactly ONE uniform draw per environment (the command term relies on the
    draw count being fixed so its RNG stream does not depend on the slot mix).

    Args:
        generator: Optional ``torch.Generator`` (tests pass a seeded one; the term passes
            ``None`` and uses the global stream, as the forward command does).
        probs: Non-negative weights, one per slot -- ``cfg.slot_probs``
            ``(stand, move, turn, move_turn)``. Normalised here, so they need not sum to 1.
        n: Number of draws.
        device: Device for the result (and the uniform draw).

    Returns:
        ``int64`` slot ids in ``[0, len(probs))``, shape ``(n,)``.
    """
    p = torch.as_tensor(probs, dtype=torch.float64)
    if p.ndim != 1 or p.numel() == 0:
        raise ValueError(f"slot probs must be a non-empty 1-D sequence, got {probs!r}")
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError(f"slot probs must be finite and non-negative, got {probs!r}")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError(f"slot probs must not be all zero, got {probs!r}")
    cum = torch.cumsum(p / total, dim=0)
    cum[-1] = 1.0  # exact right edge: no draw can fall past the last slot
    cum = cum.to(device=device, dtype=torch.float32).contiguous()
    u = torch.rand(n, generator=generator, device=device)
    slot = torch.searchsorted(cum, u.contiguous(), right=True)
    return slot.clamp_(max=p.numel() - 1).to(torch.long)


def sample_facing(
    yaw: torch.Tensor,
    delta_lo: float,
    delta_hi: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample an absolute world facing target RELATIVE to the current yaw.

    ``psi = yaw + sign * U(delta_lo, delta_hi)`` with ``sign`` uniform in ``{-1, +1}``:
    the band is on the *magnitude* of the facing change, so a phase's
    ``facing_delta = (0.2, 0.6)`` means "turn by 0.2 to 0.6 rad, either way", never "turn
    to an absolute world heading in that band". Consumes two uniform draws per
    environment (magnitude, then sign).

    The result is NOT wrapped (see the module docstring); ``wrap_to_pi(psi - yaw)`` is
    exactly ``+-U(delta_lo, delta_hi)`` for any band within ``+-pi``.

    Args:
        yaw: Current world yaw, radians, shape ``(N,)``.
        delta_lo, delta_hi: Facing-change magnitude band, radians, ``0 <= lo <= hi``.
        generator: Optional ``torch.Generator``.

    Returns:
        The world facing target ``psi``, shape/dtype/device of ``yaw``.
    """
    if not (0.0 <= delta_lo <= delta_hi):
        raise ValueError(f"facing_delta must satisfy 0 <= lo <= hi, got ({delta_lo}, {delta_hi})")
    shape = tuple(yaw.shape)
    u_mag = torch.rand(shape, generator=generator, device=yaw.device, dtype=yaw.dtype)
    u_sign = torch.rand(shape, generator=generator, device=yaw.device, dtype=yaw.dtype)
    mag = delta_lo + u_mag * (delta_hi - delta_lo)
    sign = torch.where(u_sign < 0.5, -torch.ones_like(yaw), torch.ones_like(yaw))
    return yaw + sign * mag


def sample_travel_dir(
    yaw: torch.Tensor,
    dir_band: tuple[float, float] | None,
    travel_dir_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the WORLD travel direction a slot will hold.

    Two modes, one uniform draw per environment either way:

    * ``dir_band is None`` (the omni default) -- world-uniform:
      ``theta = U(travel_dir_range)``. The robot's own attitude does not enter.
    * ``dir_band = (lo, hi)`` -- **body-relative at resample time**:
      ``theta = yaw + U(lo, hi)``, snapshotted against the yaw the robot has at the
      resample instant and then held as a WORLD direction for the rest of the slot (so a
      body-relative band still trains "hold your line while the body turns"). This is how
      the phase-1 stages pin travel to one body axis: ``dir_band = (0, 0)`` is stage 1a
      (body ``+x``, forward walking), ``dir_band = (pi/2, pi/2)`` is stage 1b (body ``+y``,
      left-leading sideways walking, the sideways task's convention).

    The result is NOT wrapped (see the module docstring), so ``dir_band = (0, 0)`` gives
    ``phi == 0.0`` bitwise.

    Args:
        yaw: Current world yaw, radians, shape ``(N,)``. Ignored when ``dir_band`` is None.
        dir_band: ``None``, or a ``(lo, hi)`` body-relative band in radians.
        travel_dir_range: ``(lo, hi)`` world band used when ``dir_band`` is None.
        generator: Optional ``torch.Generator``.

    Returns:
        The world travel direction ``theta``, shape/dtype/device of ``yaw``.
    """
    u = torch.rand(tuple(yaw.shape), generator=generator, device=yaw.device, dtype=yaw.dtype)
    if dir_band is None:
        lo, hi = float(travel_dir_range[0]), float(travel_dir_range[1])
        if hi < lo:
            raise ValueError(f"travel_dir_range must satisfy lo <= hi, got ({lo}, {hi})")
        return lo + u * (hi - lo)
    lo, hi = float(dir_band[0]), float(dir_band[1])
    if hi < lo:
        raise ValueError(f"dir_band must satisfy lo <= hi, got ({lo}, {hi})")
    return yaw + (lo + u * (hi - lo))
