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
from types import MappingProxyType
from typing import NamedTuple

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


# ------------------------------------------------------------------- telemetry categories
#
# The seven buckets the per-category reward telemetry (stride/posture plan, step 4) reports
# against. ``stand`` / ``turn`` / ``move_turn`` come straight off the slot; the four
# ``SLOT_MOVE`` rows are split by the INTENDED bearing so a heading-tracking error can never
# move a row between categories (see :func:`telemetry_category`).
TELEMETRY_CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")

_CAT_STAND = TELEMETRY_CATEGORIES.index("stand")
_CAT_FWD = TELEMETRY_CATEGORIES.index("fwd")
_CAT_SIDE = TELEMETRY_CATEGORIES.index("side")
_CAT_BACK = TELEMETRY_CATEGORIES.index("back")
_CAT_DIAG = TELEMETRY_CATEGORIES.index("diag")
_CAT_TURN = TELEMETRY_CATEGORIES.index("turn")
_CAT_MOVE_TURN = TELEMETRY_CATEGORIES.index("move_turn")

_FWD_MAX_DEG = 22.5
_SIDE_LO_DEG = 67.5
_SIDE_HI_DEG = 112.5
_BACK_MIN_DEG = 157.5

_BEARING_TOL_DEG = 1e-4
"""Slack on the boundary comparisons below, degrees. A bearing built from a wrap + ``rad2deg``
of an exact boundary value (a caller's ``22.5``, ``67.5``, ...) rounds to within ~5e-6 deg of
it even in float64 -- direction unpredictable -- so the four bucket edges stay INCLUSIVE (per
the spec) only with a tolerance comfortably above that noise floor and comfortably below the
narrowest gap between two edges (45 deg)."""


def telemetry_category(slot_id: torch.Tensor, bearing: torch.Tensor) -> torch.Tensor:
    """``(slot_id, bearing)`` -> index into :data:`TELEMETRY_CATEGORIES`.

    ``stand``, ``turn`` and ``move_turn`` come straight from the slot. A ``SLOT_MOVE`` row is
    bucketed by ``|bearing|``: ``fwd`` <= 22.5 deg, ``side`` in [67.5, 112.5] deg, ``back`` >=
    157.5 deg, ``diag`` otherwise.

    Args:
        slot_id: ``(N,)`` int64, ``SLOT_STAND`` .. ``SLOT_MOVE_TURN``.
        bearing: ``(N,)`` the commanded WORLD travel bearing relative to the WORLD facing
            target -- i.e. ``travel_dir_w - heading_target``, in radians. This is the
            INTENDED bearing, not ``phi_b`` (the instantaneous body-frame command): using the
            commanded pair means a facing-tracking error never moves a row between
            categories. Need not be pre-wrapped; this function wraps it.

    Returns:
        ``(N,)`` int64, dtype/device of ``slot_id``. Rows outside ``SLOT_MOVE`` ignore
        ``bearing`` entirely.
    """
    # Promoted to float64 for the wrap + degree conversion only: a bearing that lands exactly
    # on a boundary (67.5 deg from the caller's ``travel_dir_w - heading_target`` in float32)
    # must not fall the wrong side of it from wrap/rad2deg rounding alone. The output stays
    # ``slot_id``'s own dtype (int64) regardless.
    abs_deg = torch.rad2deg(wrap_to_pi(bearing.double()).abs())
    move_cat = torch.full_like(slot_id, _CAT_DIAG)
    move_cat = torch.where(abs_deg <= _FWD_MAX_DEG + _BEARING_TOL_DEG, torch.full_like(slot_id, _CAT_FWD), move_cat)
    move_cat = torch.where(abs_deg >= _BACK_MIN_DEG - _BEARING_TOL_DEG, torch.full_like(slot_id, _CAT_BACK), move_cat)
    move_cat = torch.where(
        (abs_deg >= _SIDE_LO_DEG - _BEARING_TOL_DEG) & (abs_deg <= _SIDE_HI_DEG + _BEARING_TOL_DEG),
        torch.full_like(slot_id, _CAT_SIDE),
        move_cat,
    )

    category = torch.full_like(slot_id, _CAT_STAND)
    category = torch.where(slot_id == SLOT_MOVE, move_cat, category)
    category = torch.where(slot_id == SLOT_TURN, torch.full_like(slot_id, _CAT_TURN), category)
    category = torch.where(slot_id == SLOT_MOVE_TURN, torch.full_like(slot_id, _CAT_MOVE_TURN), category)
    return category


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


def check_relay_hyst(engage: float, release: float) -> None:
    """``0 <= release < engage``, both finite -- anything else is a relay that never turns off
    (release >= engage chatters exactly like the dead band it replaces) or never turns on."""
    if not (math.isfinite(engage) and math.isfinite(release)):
        raise ValueError(f"relay hysteresis thresholds must be finite, got ({engage}, {release})")
    if not (0.0 <= release < engage):
        raise ValueError(
            f"relay hysteresis needs 0 <= release < engage (rad), got engage={engage}, "
            f"release={release}"
        )


def heading_rate_command_hyst(
    heading_target: torch.Tensor,
    yaw: torch.Tensor,
    stiffness: float,
    ang_vel_max,
    latch: torch.Tensor,
    engage: float,
    release: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`heading_rate_command` with a HYSTERESIS latch in place of the dead band.

    ``KRABBY_OMNI_RELAY_HYST`` (motion-quality campaign 2026-09-22; default OFF -- the shipped
    path calls :func:`heading_rate_command` / :func:`heading_rate_command_rows` and never this).
    The stock relay zeroes ``|wz| <= ang_vel_clip`` and passes everything above it, so a facing
    error that wanders across the ~0.133 rad band edge toggles a >= 0.2 rad/s command on and off
    -- and with it the gait clock. Here, per env, with ``err = wrap(heading_target - yaw)``:

        ``latch' = |err| > release  if latch  else  |err| > engage``
        ``wz = clip(err * stiffness, +-ang_vel_max) * latch'``

    i.e. the turn command switches ON only past ``engage`` and stays on (a plain saturated
    P-control, NO dead band) until the error is below ``release``. The dead band is replaced,
    not stacked: with it, an engaged command would be cut again at the old band edge.

    Args:
        heading_target, yaw, stiffness: as :func:`heading_rate_command`.
        ang_vel_max: saturation, a float or an ``(N,)`` per-row tensor (the pool path).
        latch: ``(N,)`` bool, the previous step's latch.
        engage, release: thresholds on ``|err|``, rad (validated by :func:`check_relay_hyst`
            once, where the cfg is built -- not per step).

    Returns:
        ``(wz, latch')`` -- the yaw-rate command and the updated latch, both ``(N,)``.
    """
    err = wrap_to_pi(heading_target - yaw)
    abs_err = torch.abs(err)
    new_latch = torch.where(latch, abs_err > release, abs_err > engage)
    if isinstance(ang_vel_max, torch.Tensor):
        wz = torch.clip(err * stiffness, min=-ang_vel_max, max=ang_vel_max)
    else:
        wz = torch.clip(err * stiffness, min=-float(ang_vel_max), max=float(ang_vel_max))
    return wz * new_latch, new_latch


def check_row_ang_limits(ang_vel_max: torch.Tensor, ang_vel_clip: torch.Tensor) -> None:
    """Validate the per-row turn limits :func:`heading_rate_command_rows` will be handed.

    Call this ONCE, where the limit tables are built (``CrabHexOmniCommand.__init__``), never
    per step. Both checks are whole-tensor reductions followed by a ``bool()``, and on CUDA a
    ``bool(tensor)`` is a host-device SYNCHRONISATION: inside the per-step yaw-rate expression
    they would cost two syncs per physics step for the whole run, on a condition that is
    constant by construction (the tables are ``cfg.ang_vel_max * ang_scale`` and
    ``cfg.clips.ang_vel_clip * ang_scale`` with ``ang_scale > 0`` already enforced by
    :func:`parse_cmd_pool` and the two cfg values by ``_validate_omni_cfg``). A pooled stage
    would then run measurably slower than the legacy stage it is being compared against --
    a wall-clock confound bought with a check that can never fire at runtime.

    Args:
        ang_vel_max: ``(N,)`` or ``(K+1,)`` per-row saturation, rad/s; every entry positive.
        ang_vel_clip: same shape, per-row dead band, rad/s; every entry non-negative.
    """
    if not bool((ang_vel_max > 0.0).all()):
        raise ValueError("ang_vel_max must be positive on every row")
    if bool((ang_vel_clip < 0.0).any()):
        raise ValueError("ang_vel_clip must be non-negative on every row")


def heading_rate_command_rows(
    heading_target: torch.Tensor,
    yaw: torch.Tensor,
    stiffness: float,
    ang_vel_max: torch.Tensor,
    ang_vel_clip: torch.Tensor,
) -> torch.Tensor:
    """:func:`heading_rate_command` with PER-ROW saturation and dead band.

    Same expression, same order (scale -> saturate -> dead-band the *saturated* value, strict
    ``>``); only the two limits become ``(N,)`` tensors so a command pool can give a type its
    own turn demand (``ang_scale`` scales BOTH, which is what keeps the dead band the same
    fraction of the cap -- scaling the cap alone would silently swallow a whole type's turns).

    With unit scales this is bitwise :func:`heading_rate_command`: the shipped legacy path
    still calls the scalar version, so the two never have to be compared at runtime, but a
    unit test pins them against each other.

    NO VALIDATION HERE, deliberately: this runs every physics step of a pooled run, and the
    two limits come from tables built once at construction. Their validation lives in
    :func:`check_row_ang_limits`, which the command term calls once -- see that docstring for
    why a per-step ``bool(tensor)`` is not free.

    Args:
        heading_target, yaw, stiffness: as :func:`heading_rate_command`.
        ang_vel_max: ``(N,)`` per-row saturation, rad/s. Every entry must be positive
            (checked once by :func:`check_row_ang_limits`).
        ang_vel_clip: ``(N,)`` per-row dead band, rad/s. Every entry must be non-negative.
    """
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


# ============================================================== the per-type command pool
#
# The four slots above are a REGIME taxonomy (does this row travel? does it turn?) and they
# are all the shipped sampler has: direction and turn-ness are drawn independently from one
# global ``dir_band`` and one global ``slot_probs``, so "forward while turning, but never
# sideways while turning" is inexpressible -- the two levers are orthogonal.
#
# A COMMAND POOL is a list of named TYPES, each carrying its own probability, its own travel
# direction set and its own turn flag. The pool does not replace the slots: every type
# PROJECTS onto one of them (``POOL_TYPES[name].slot``), so every metric, reward, eval hold
# and piece of telemetry downstream keeps reading ``slot_id`` and never learns that a pool
# exists. That projection is the whole reason a capability curriculum can be added without
# touching ``_update_metrics``, ``reward_yaw_progress``, ``schedule_hook`` or the eval
# harness.
#
# WHAT A TYPE'S BAND MEANS. Exactly what ``dir_band`` means today: a BODY-RELATIVE band
# snapshotted against the yaw at the resample instant and then HELD as a world direction
# ("clamp-only re-cap", see :func:`body_command_from_world`). ``fwd`` is the zero-width band
# ``(0, 0)`` -- bitwise ``theta == yaw`` -- and ``side_l`` is ``(+pi/2, +pi/2)``, the same
# numbers stages 1a and 1b pass through ``dir_band`` today. The two ``omni`` types are the
# other mode: WORLD-uniform over ``travel_dir_range``, the robot's attitude not entering.
#
# NON-TRAVELLING TYPES (``stand``, ``turn``) carry the zero-width body band ``(0, 0)``: their
# speed is exactly 0, so the held direction is unobservable in ``vel_command_b``, and pinning
# it to the body's own nose is both cheaper and tidier than a random world ray (the parkour
# event's ground bead trail reads ``travel_dir_w`` and is gated on the commanded speed).
#
# THE ONE VISIBLE CONSEQUENCE, and the ruling on it (2026-09-18). A legacy window draws ONE
# direction band for every row, whatever its slot; a pool draws each row's band from its TYPE.
# So a pool twin of a legacy window is bitwise on ``slot_id / speed_cmd / heading_target /
# vel_command_b`` for every row and on ``travel_dir_w`` for every MOVING row, but its
# NON-MOVING rows hold ``yaw + 0`` where the legacy window held whatever its global band said:
# a random world ray when ``dir_band is None`` (the 1c-style windows), and ``yaw + pi/2`` when
# ``dir_band = (pi/2, pi/2)`` (a 1b-style window with a ``stand`` share). Both divergences are
# the same fact and neither is fixable while a type has ONE band: making ``stand``/``turn``
# world types would merely move the mismatch onto the body-band windows. The plan's twin
# bullet ("bitwise on ... travel_dir_w") is therefore read as "on the moving rows", and the
# difference is unobservable downstream -- ``travel_dir_w`` on a speed-0 row reaches only
# ``OmniParkourEvent.target_pos_rel``, whose one consumer (the PLAY bead trail) is gated on
# ``speed_cmd > 0``; no omni reward or observation reads it. Pinned by
# ``test_pool_1c_twin_matches_the_legacy_world_uniform_window`` and
# ``test_pool_standing_rows_hold_the_body_axis_not_the_windows_band``.


class PoolType(NamedTuple):
    """One entry of :data:`POOL_TYPES`: where a named command type lands and how it travels.

    ``slot``
        The 2-bit regime slot the type projects onto (``SLOT_STAND`` ... ``SLOT_MOVE_TURN``).
    ``world``
        ``True`` -> the travel direction is drawn WORLD-uniform over ``travel_dir_range``
        (the ``omni`` types); ``False`` -> from ``band``, body-relative at resample time.
    ``band``
        ``(lo, hi)`` body-relative radians, or ``None`` for a world type. Zero-width for
        every axis-pinned type, which is what makes ``theta == yaw + <exact literal>``.
    """

    slot: int
    world: bool
    band: tuple[float, float] | None


_HALF_PI = math.pi / 2.0

POOL_TYPES: MappingProxyType[str, PoolType] = MappingProxyType({
    # -- stand: no travel, no turn. The band is unobservable (speed 0); see the note above.
    "stand": PoolType(SLOT_STAND, False, (0.0, 0.0)),
    # -- the five pure-travel types (slot 1)
    "fwd": PoolType(SLOT_MOVE, False, (0.0, 0.0)),
    "side_l": PoolType(SLOT_MOVE, False, (_HALF_PI, _HALF_PI)),
    "side_r": PoolType(SLOT_MOVE, False, (-_HALF_PI, -_HALF_PI)),
    "back": PoolType(SLOT_MOVE, False, (math.pi, math.pi)),
    "omni": PoolType(SLOT_MOVE, True, None),
    # -- turn in place (slot 2)
    "turn": PoolType(SLOT_TURN, False, (0.0, 0.0)),
    # -- the five travel-while-turning types (slot 3), one per travel type
    "fwd_turn": PoolType(SLOT_MOVE_TURN, False, (0.0, 0.0)),
    "side_l_turn": PoolType(SLOT_MOVE_TURN, False, (_HALF_PI, _HALF_PI)),
    "side_r_turn": PoolType(SLOT_MOVE_TURN, False, (-_HALF_PI, -_HALF_PI)),
    "back_turn": PoolType(SLOT_MOVE_TURN, False, (math.pi, math.pi)),
    "omni_turn": PoolType(SLOT_MOVE_TURN, True, None),
})
"""Every command type a pool may name, in the curriculum's own order.

``side_l`` / ``side_r`` are SEPARATE types (the shipped ``dir_band`` can only express one
lateral direction at a time, and the RSI bank's source-1 frames are left-leading only), and
``back`` is its own type rather than a widening of ``omni`` so a stage can enable backward
travel without enabling every bearing at once."""

POOL_FLOOR_SHARE = 0.025
"""Smallest share a curriculum stage may give a type it has ENABLED.

Below this a type is present in the config dump and absent from the training distribution:
at 64 envs x 10 s resampling a 1 % type is drawn a few times per minute, which neither
teaches the capability nor shows up in ``pool_<name>_draws`` as anything but noise. The
campaign driver's recipe validator imports this constant rather than re-spelling it."""


class CmdPool(NamedTuple):
    """A parsed command pool: K named types with weights and per-type scales.

    All four tuples are K long and index-aligned. ``weights`` are raw, non-negative and not
    all zero (the sampler normalises, exactly as it does for ``slot_probs``, so the config
    dump keeps saying what the launcher asked for). A weight of ``0.0`` is legal and KEPT:
    :meth:`rescale_to` never drops a type, so the type list -- and therefore the
    ``pool_<name>_*`` metric names and the render -> parse round trip -- is stable along a
    chain of stages.

    ``ang_scale``
        Multiplies the row's yaw-rate P-control cap AND its dead band (``ang_vel_max`` and
        ``clips.ang_vel_clip``), so a type can be given a gentler turn demand without
        changing the dead band's share of the cap. Default 1.0.
    ``speed_scale``
        Multiplies the ellipse cap AT DRAW TIME, so a type can be given a slower speed band.
        The per-step re-cap in ``_write_command`` stays the GLOBAL ellipse: the scale picks a
        smaller speed, it does not narrow the clamp the body's own rotation runs into.
        Default 1.0.
    """

    names: tuple[str, ...]
    weights: tuple[float, ...]
    ang_scale: tuple[float, ...]
    speed_scale: tuple[float, ...]

    def slot_of(self, name: str) -> int:
        """The 2-bit regime slot a type name projects onto."""
        try:
            return POOL_TYPES[str(name)].slot
        except KeyError:
            raise ValueError(
                f"unknown command-pool type {name!r}; known: {tuple(POOL_TYPES)}"
            ) from None

    def projected_slot_probs(self) -> tuple[float, float, float, float]:
        """The pool's weights summed per slot and NORMALISED, in ``SLOT_NAMES`` order.

        This is the 4-tuple ``KRABBY_OMNI_SLOTS`` has to carry for the same distribution --
        the command cfg's validator refuses a pool whose projection disagrees with
        ``slot_probs``, so a launcher can never describe two different regimes at once.
        """
        acc = [0.0] * N_SLOTS
        for name, weight in zip(self.names, self.weights):
            acc[POOL_TYPES[name].slot] += float(weight)
        total = sum(acc)
        if total <= 0.0:
            raise ValueError("command pool weights must not be all zero")
        return tuple(x / total for x in acc)  # type: ignore[return-value]

    def rescale_to(self, slot_probs) -> CmdPool:
        """A new pool with the given slot projection, keeping the WITHIN-slot proportions.

        The teacher windows (2a/2b) are described by a slot mix, not by a pool; this is how a
        curriculum's final pool is carried into them unchanged in shape and re-weighted in
        regime. Every type keeps its ``ang_scale`` / ``speed_scale`` and its position.

        Rules:

        * a target slot with positive probability but NO type of that slot in the pool is a
          ``ValueError`` naming the slot (silently dropping the mass would train a regime
          nobody asked for);
        * a target slot with probability 0 keeps its types at weight ``0.0`` -- never
          dropped, so ``render_cmd_pool`` -> :func:`parse_cmd_pool` still round-trips and the
          metric names do not move;
        * a slot whose types are all at weight 0 but which is given positive mass splits it
          UNIFORMLY (there are no within-slot proportions to keep).
        """
        target = tuple(float(p) for p in slot_probs)
        if len(target) != N_SLOTS:
            raise ValueError(f"rescale_to needs {N_SLOTS} slot probs {SLOT_NAMES}, got {slot_probs!r}")
        if any((not math.isfinite(p)) or p < 0.0 for p in target):
            raise ValueError(f"rescale_to slot probs must be finite and non-negative, got {slot_probs!r}")
        total = sum(target)
        if total <= 0.0:
            raise ValueError(f"rescale_to slot probs must not be all zero, got {slot_probs!r}")
        target = tuple(p / total for p in target)

        slots = tuple(POOL_TYPES[n].slot for n in self.names)
        per_slot = [0.0] * N_SLOTS
        count = [0] * N_SLOTS
        for slot, weight in zip(slots, self.weights):
            per_slot[slot] += float(weight)
            count[slot] += 1
        for slot in range(N_SLOTS):
            if target[slot] > 0.0 and count[slot] == 0:
                raise ValueError(
                    f"rescale_to: slot {SLOT_NAMES[slot]!r} is given probability {target[slot]} "
                    f"but the pool has no type in it (types: {self.names})"
                )
        out = []
        for slot, weight in zip(slots, self.weights):
            if target[slot] <= 0.0:
                out.append(0.0)
            elif per_slot[slot] > 0.0:
                out.append(float(weight) / per_slot[slot] * target[slot])
            else:
                out.append(target[slot] / count[slot])
        return CmdPool(self.names, tuple(out), self.ang_scale, self.speed_scale)


def parse_cmd_pool(text: str) -> CmdPool:
    """``name:weight[:ang_scale[:speed_scale]],...`` -> :class:`CmdPool`.

    The canonical spelling of a curriculum stage's training distribution, and the string the
    command cfg stores VERBATIM (``cmd_pool``) so a config dump says exactly what the
    launcher asked for. Whitespace around tokens and around fields is tolerated; everything
    else is refused by name, because every failure mode is silent in a training log:

    * an unknown type name (a typo would otherwise drop a whole capability);
    * a duplicate name (which of the two weights wins is invisible);
    * a negative weight, a non-finite weight, or all weights zero;
    * an ``ang_scale`` / ``speed_scale`` that is not strictly positive (0 would make a type's
      turns or steps inert while the dump still listed it);
    * a token with no ``:``, with more than four fields, or empty (a stray comma).

    A weight of exactly ``0.0`` IS accepted: a stage may keep a type in the list at zero mass
    so the type list, the metric names and the render round trip stay stable.
    """
    raw = str(text).strip()
    if raw == "":
        raise ValueError("command pool must not be empty, got ''")
    names: list[str] = []
    weights: list[float] = []
    ang: list[float] = []
    spd: list[float] = []
    for token in raw.split(","):
        fields = [f.strip() for f in token.strip().split(":")]
        if len(fields) < 2 or len(fields) > 4 or any(f == "" for f in fields):
            raise ValueError(
                f"command pool entry must be 'name:weight[:ang_scale[:speed_scale]]', got {token!r}"
            )
        name = fields[0]
        if name not in POOL_TYPES:
            raise ValueError(f"unknown command-pool type {name!r}; known: {tuple(POOL_TYPES)}")
        if name in names:
            raise ValueError(f"command-pool type {name!r} appears twice in {text!r}")
        numbers = []
        for field in fields[1:]:
            try:
                value = float(field)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"command pool entry {token!r} has a non-numeric field {field!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"command pool entry {token!r} has a non-finite field {field!r}")
            numbers.append(value)
        weight = numbers[0]
        if weight < 0.0:
            raise ValueError(f"command-pool weight must be non-negative, got {token!r}")
        a_scale = numbers[1] if len(numbers) > 1 else 1.0
        s_scale = numbers[2] if len(numbers) > 2 else 1.0
        if not a_scale > 0.0:
            raise ValueError(f"command-pool ang_scale must be positive, got {token!r}")
        if not s_scale > 0.0:
            raise ValueError(f"command-pool speed_scale must be positive, got {token!r}")
        names.append(name)
        weights.append(weight)
        ang.append(a_scale)
        spd.append(s_scale)
    if sum(weights) <= 0.0:
        raise ValueError(f"command pool weights must not all be zero, got {text!r}")
    return CmdPool(tuple(names), tuple(weights), tuple(ang), tuple(spd))


def render_cmd_pool(pool: CmdPool) -> str:
    """:class:`CmdPool` -> the canonical knob string, in the pool's own type order.

    ``repr`` precision on every float, NEVER a fixed number of decimals: the campaign driver
    composes a stage's pool from TRANSFORMS (a share edit, then a scale edit), renders it to
    one string per stage and records that string as the truth. A ``f"{w:.3f}"`` would make
    the render lossy, so a rendered-then-parsed pool would no longer match the ``SLOTS``
    projection the same recipe computed, and the command cfg's validator would refuse it --
    or, worse, accept it 1e-4 off.

    Shortest faithful form: ``name:weight`` when both scales are 1.0, ``name:weight:ang``
    when only ``speed_scale`` is 1.0, and all four fields otherwise.
    """
    parts = []
    for name, weight, a_scale, s_scale in zip(pool.names, pool.weights, pool.ang_scale, pool.speed_scale):
        if float(s_scale) != 1.0:
            parts.append(f"{name}:{float(weight)!r}:{float(a_scale)!r}:{float(s_scale)!r}")
        elif float(a_scale) != 1.0:
            parts.append(f"{name}:{float(weight)!r}:{float(a_scale)!r}")
        else:
            parts.append(f"{name}:{float(weight)!r}")
    return ",".join(parts)


def pool_type_min_cap(name: str, a: float, b: float, c: float) -> float:
    """The smallest ellipse cap the pool type ``name`` can ever draw at, m/s.

    A zero-width body band is one exact direction, so its cap is :func:`ellipse_cap` there --
    which is why ``fwd`` may take a ``speed_scale`` the lateral axis could not. Anything wider
    (a world type, or a band that is genuinely a range) is judged against the smallest
    semi-axis, because it really can draw that direction.

    Non-moving types (``stand``, ``turn``) never draw a speed at all; this still answers for
    them (their band is ``(0, 0)``), and callers gate on the slot.
    """
    spec = POOL_TYPES[name]
    band = spec.band
    if spec.world or band is None or float(band[0]) != float(band[1]):
        return float(min(a, b, c))
    return float(ellipse_cap(torch.tensor([float(band[0])], dtype=torch.float64), a, b, c))


def pool_speed_scale_floor(name: str, speed_ellipse, min_speed: float) -> float:
    """The strict lower bound on a legal ``speed_scale`` for pool type ``name``.

    A type's speed band is ``floor + u * (cap * speed_scale - floor)``: once
    ``cap * speed_scale`` sinks to ``min_speed`` (the walk floor plus its headroom) the band
    is empty or INVERTED and every draw sits under the gait clock's stop speed -- a type that
    is still in the config dump, still drawn at its weight, and training a creeper. So a
    ``speed_scale`` is legal iff ``speed_scale > pool_speed_scale_floor(...)`` (or is exactly
    1.0, which is the unscaled case the ellipse check already covers).

    THIS IS THE BOUND, exported rather than re-spelled: ``_validate_omni_cfg`` refuses with it
    at config build, and the campaign driver's recipe validator is meant to call it at COMPOSE
    time so a ``pool_edit {speed_scale: ...}`` that cannot be honoured is caught before a GPU
    window is booked, not after. At the shipped ellipse (0.35, 0.25, 0.25) with
    ``lin_vel_clip`` 0.2 the bound is ~0.88 for every world / lateral / backward type and
    ~0.63 for ``fwd``: the shipped ellipse simply has no room under it, and the way to buy
    room is a wider ``speed_ellipse`` or a smaller ``lin_vel_clip`` in the same overlay, not a
    smaller scale.

    Args:
        name: A key of :data:`POOL_TYPES`.
        speed_ellipse: ``(a, b, c)`` semi-axes, m/s.
        min_speed: The speed a draw must stay strictly above -- the walk floor plus the
            construction headroom (``omni_walk_floor(cfg) + MIN_CAP_HEADROOM_M_S``).

    Returns:
        ``min_speed / min_cap``, a float. ``speed_scale`` values at or below it are refused.
    """
    a, b, c = (float(x) for x in speed_ellipse)
    return float(min_speed) / pool_type_min_cap(name, a, b, c)


class PoolTables(NamedTuple):
    """The per-type lookup tables :class:`CrabHexOmniCommand` gathers with ``type_id``.

    Every table has **K + 1** entries. The trailing one is the ``type_id = -1`` SENTINEL --
    "this row was not drawn from the pool" (an eval ``schedule_hook`` row, an RSI-staged row,
    and every row at all when no pool is configured). Python's negative indexing makes
    ``table[-1]`` land on it for free, so the gather needs no ``where``: the sentinel's scales
    are exactly 1.0 (the GLOBAL cap and dead band), its band is zero-width body-relative, and
    its slot is ``SLOT_MOVE`` (what a staged row is forced to).

    Getting this wrong is silent and expensive: a K-entry table indexed with ``-1`` would hand
    every eval and every RSI row the LAST type's scales.
    """

    slot: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    world: torch.Tensor
    ang_scale: torch.Tensor
    speed_scale: torch.Tensor


def pool_tables(
    pool: CmdPool,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PoolTables:
    """Build the K+1-entry gather tables for ``pool`` (see :class:`PoolTables`)."""
    slot = [POOL_TYPES[n].slot for n in pool.names] + [SLOT_MOVE]
    lo, hi, world = [], [], []
    for name in pool.names:
        spec = POOL_TYPES[name]
        band = (0.0, 0.0) if spec.band is None else spec.band
        lo.append(float(band[0]))
        hi.append(float(band[1]))
        world.append(bool(spec.world))
    lo.append(0.0)
    hi.append(0.0)
    world.append(False)
    return PoolTables(
        slot=torch.tensor(slot, dtype=torch.long, device=device),
        lo=torch.tensor(lo, dtype=dtype, device=device),
        hi=torch.tensor(hi, dtype=dtype, device=device),
        world=torch.tensor(world, dtype=torch.bool, device=device),
        ang_scale=torch.tensor([float(s) for s in pool.ang_scale] + [1.0], dtype=dtype, device=device),
        speed_scale=torch.tensor([float(s) for s in pool.speed_scale] + [1.0], dtype=dtype, device=device),
    )


def sample_pool_travel_dir(
    yaw: torch.Tensor,
    type_id: torch.Tensor,
    lo_t: torch.Tensor,
    hi_t: torch.Tensor,
    world_t: torch.Tensor,
    travel_dir_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """:func:`sample_travel_dir` with a PER-ROW band, gathered by ``type_id``.

    ONE uniform draw per environment, like both legacy modes -- the resample RNG stream must
    not depend on the type mix, or no pooled run is comparable with a legacy one (and no
    seed-3 record reproduces). Both branches are evaluated and selected with ``where`` for
    exactly that reason.

    The two expressions are the legacy ones, character for character:

    * world types: ``theta = lo + u * (hi - lo)`` over ``travel_dir_range``;
    * body types: ``theta = yaw + (lo_row + u * (hi_row - lo_row))``, so a zero-width band
      gives ``yaw + 0.0`` BITWISE (``fwd`` is ``theta == yaw``, ``side_l`` is exactly
      ``yaw + math.pi/2``) -- the property the legacy twins are pinned on.

    Args:
        yaw: Current world yaw, radians, shape ``(N,)``.
        type_id: ``(N,)`` int64 pool type ids; ``-1`` lands on the tables' sentinel entry.
        lo_t, hi_t: ``(K+1,)`` body-relative band tables (``pool_tables``).
        world_t: ``(K+1,)`` bool "this type draws world-uniform" table.
        travel_dir_range: ``(lo, hi)`` world band for the world types.
        generator: Optional ``torch.Generator``.

    Returns:
        The world travel direction ``theta``, shape/dtype/device of ``yaw``. NOT wrapped.
    """
    u = torch.rand(tuple(yaw.shape), generator=generator, device=yaw.device, dtype=yaw.dtype)
    lo, hi = float(travel_dir_range[0]), float(travel_dir_range[1])
    if hi < lo:
        raise ValueError(f"travel_dir_range must satisfy lo <= hi, got ({lo}, {hi})")
    world = lo + u * (hi - lo)
    lo_row = lo_t[type_id]
    hi_row = hi_t[type_id]
    body = yaw + (lo_row + u * (hi_row - lo_row))
    return torch.where(world_t[type_id], world, body)


# ============================================================ the body-frame command mode
#
# Robot-frame velocity commands campaign (``experiments/2026-10-15_body_cmd``, PLAN D1 / D3 / D6;
# DEFAULT OFF behind ``KRABBY_OMNI_COMMAND_FRAME=body``). The world command above is a held world
# travel direction + an absolute world facing, turned into ``(vx, vy, wz)`` every step by the
# clamp-only re-cap and the facing P-relay -- so the policy has only ever seen
# ``wz in {0} U (0.2, 0.3]``. The body mode commands ``(vx, vy, wz)`` DIRECTLY, held in the body
# frame for the slot: the deployment language (HAL teleop sends body-frame ``vx, vy, yaw_rate``).
#
# Everything a body command may be is ONE envelope, defined ONCE here
# (:func:`body_cmd_violation_masks`) and used by the sampler's tests, the step-2 eval validator,
# the eval hook (``CrabHexOmniBodyCommand.schedule_hook_body``) and the deployment / ``world_rate``
# mapper (:func:`shape_body_command`):
#
# * speed ``|v| = hypot(vx, vy)`` in ``{0} U [walk_floor, ellipse_cap(atan2(vy, vx))]``;
# * yaw rate ``|wz|`` in ``{0} U [w_min, w_max]``;
# * moving AND turning: ``|wz| <= body_cap_wz(|v|)`` -- the COUPLED move+turn cap, a piecewise-
#   linear frontier through measured ``(|v|, wz)`` points (PLAN D3: the robot cannot turn as fast
#   while it walks as it can on the spot).
#
# The numbers are a KNOB / manifest parameter (``KRABBY_OMNI_BODY_ENVELOPE`` /
# ``KRABBY_OMNI_BODY_CAP``, frozen in the campaign CHARTER at step 4), never a code constant: this
# module owns their GRAMMAR (:func:`parse_body_envelope`).

#: The two knob names whose grammar this module owns (``config/crab_hex/body_knobs.py`` re-states
#: them as its ``*_VAR`` constants; a unit test pins the two spellings against each other).
BODY_ENVELOPE_KNOB = "KRABBY_OMNI_BODY_ENVELOPE"
BODY_CAP_KNOB = "KRABBY_OMNI_BODY_CAP"


class BodyEnvelope(NamedTuple):
    """The measured command envelope of a body-frame run (PLAN D3), all rad/s except ``cap``'s v.

    ``w_min``
        The smallest non-zero yaw-rate command (the slow-turn floor; 0.05 rad/s pre-registered).
    ``ceiling``
        The plant's sustained pure-turn rate (S1's measured 0.167 rad/s at the plan's writing) --
        recorded with the envelope because the gates are read at it; the sampler draws above it,
        up to ``w_max`` (the 1.2x headroom).
    ``w_max``
        The largest yaw-rate command any row may carry; a pure turn draws up to it.
    ``cap``
        The coupled move+turn frontier: ``((v0, wz0), (v1, wz1), ...)`` with ``v0 == 0``, ``v``
        strictly increasing and ``wz`` non-increasing -- see :func:`body_cap_wz`.
    """

    w_min: float
    ceiling: float
    w_max: float
    cap: tuple[tuple[float, float], ...]


def _finite_float(knob: str, text: str, token: str) -> float:
    try:
        value = float(str(token).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{knob}: {token!r} is not a number (in {text!r})") from exc
    if not math.isfinite(value):
        raise ValueError(f"{knob}: {token!r} is not finite (in {text!r})")
    return value


def parse_body_envelope_triple(text: str) -> tuple[float, float, float]:
    """``KRABBY_OMNI_BODY_ENVELOPE`` = ``w_min:ceiling:w_max`` (rad/s), ``0 < w_min < ceiling <= w_max``."""
    raw = str(text).strip()
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"{BODY_ENVELOPE_KNOB} must be 'w_min:ceiling:w_max' (rad/s), got {text!r}")
    w_min, ceiling, w_max = (_finite_float(BODY_ENVELOPE_KNOB, text, p) for p in parts)
    if not (0.0 < w_min < ceiling <= w_max):
        raise ValueError(
            f"{BODY_ENVELOPE_KNOB} needs 0 < w_min < ceiling <= w_max, got w_min={w_min}, "
            f"ceiling={ceiling}, w_max={w_max} ({text!r})"
        )
    return (w_min, ceiling, w_max)


def parse_body_cap_points(text: str) -> tuple[tuple[float, float], ...]:
    """``KRABBY_OMNI_BODY_CAP`` = ``v0@wz0,v1@wz1,...`` -> ``((v0, wz0), ...)`` (m/s, rad/s).

    The SHAPE rules, which need no other knob: ``v0 == 0`` exactly, ``v`` strictly increasing,
    every ``wz`` finite and ``> 0``, ``wz`` non-increasing (a turn never gets EASIER as the robot
    walks faster). The cross-knob rule ``wz <= w_max`` is :func:`check_body_envelope`'s. Whitespace
    around tokens is tolerated; an empty entry (a stray comma), a missing ``@`` or a third field is
    refused by name.
    """
    raw = str(text).strip()
    if raw == "":
        raise ValueError(f"{BODY_CAP_KNOB} must not be empty")
    points = []
    for token in raw.split(","):
        fields = token.strip().split("@")
        if len(fields) != 2 or any(f.strip() == "" for f in fields):
            raise ValueError(f"{BODY_CAP_KNOB} entries must be 'v@wz' (m/s @ rad/s), got {token!r} in {text!r}")
        v, wz = (_finite_float(BODY_CAP_KNOB, text, f) for f in fields)
        points.append((v, wz))
    if points[0][0] != 0.0:
        raise ValueError(f"{BODY_CAP_KNOB}: the first point must be at v = 0 (the pure-turn end), got {text!r}")
    for (v0, w0), (v1, w1) in zip(points, points[1:]):
        if not v1 > v0:
            raise ValueError(f"{BODY_CAP_KNOB}: v must be strictly increasing, got {v0} then {v1} in {text!r}")
        if w1 > w0:
            raise ValueError(f"{BODY_CAP_KNOB}: wz must be non-increasing in v, got {w0} then {w1} in {text!r}")
    for v, wz in points:
        if v < 0.0:
            raise ValueError(f"{BODY_CAP_KNOB}: v must be >= 0, got {v} in {text!r}")
        if not wz > 0.0:
            raise ValueError(f"{BODY_CAP_KNOB}: every wz must be > 0 rad/s, got {wz} in {text!r}")
    return tuple(points)


def check_body_envelope(env: BodyEnvelope) -> BodyEnvelope:
    """Re-validate a :class:`BodyEnvelope` from any source (a knob, a cfg, a hand-built tuple).

    The triple rule, the cap's shape rules and the cross rule ``every cap wz in (0, w_max]``.
    Returns the envelope with every number a Python float and ``cap`` a tuple of float pairs.
    """
    try:
        w_min, ceiling, w_max = (float(env.w_min), float(env.ceiling), float(env.w_max))
        cap = tuple((float(v), float(wz)) for v, wz in env.cap)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"body envelope is not (w_min, ceiling, w_max, ((v, wz), ...)): {env!r}") from exc
    parse_body_envelope_triple(f"{w_min!r}:{ceiling!r}:{w_max!r}")
    if not cap:
        raise ValueError(f"{BODY_CAP_KNOB}: the coupled cap needs at least the v = 0 point, got {env!r}")
    parse_body_cap_points(",".join(f"{v!r}@{wz!r}" for v, wz in cap))
    over = [(v, wz) for v, wz in cap if wz > w_max]
    if over:
        raise ValueError(
            f"{BODY_CAP_KNOB}: every cap wz must be <= w_max ({BODY_ENVELOPE_KNOB} w_max = {w_max}); "
            f"{over} exceed it"
        )
    return BodyEnvelope(w_min, ceiling, w_max, cap)


def parse_body_envelope(env_text: str, cap_text: str) -> BodyEnvelope:
    """The two body-envelope knobs -> one validated :class:`BodyEnvelope` (the knob grammar)."""
    w_min, ceiling, w_max = parse_body_envelope_triple(env_text)
    return check_body_envelope(BodyEnvelope(w_min, ceiling, w_max, parse_body_cap_points(cap_text)))


def body_cap_tables(env: BodyEnvelope, device=None, dtype: torch.dtype = torch.float32):
    """``(v_points, wz_points)`` as two 1-D tensors -- built ONCE by a caller that evaluates the cap
    every resample (the command term), so no host->device copy happens in the hot path."""
    v = torch.tensor([float(p[0]) for p in env.cap], device=device, dtype=dtype)
    w = torch.tensor([float(p[1]) for p in env.cap], device=device, dtype=dtype)
    return v, w


def body_cap_wz_from_tables(speed: torch.Tensor, v_pts: torch.Tensor, w_pts: torch.Tensor) -> torch.Tensor:
    """:func:`body_cap_wz` on prebuilt tables (``speed`` already a magnitude, ``>= 0``)."""
    k = int(v_pts.numel())
    if k == 1:
        return torch.full_like(speed, 0.0) + w_pts[0]
    a = speed.contiguous()
    i = (torch.searchsorted(v_pts, a, right=True) - 1).clamp(0, k - 2)
    v0, v1 = v_pts[i], v_pts[i + 1]
    w0, w1 = w_pts[i], w_pts[i + 1]
    t = ((a - v0) / (v1 - v0)).clamp(0.0, 1.0)
    inside = w0 + t * (w1 - w0)
    return torch.where(a >= v_pts[-1], torch.zeros_like(inside) + w_pts[-1], inside)


def body_cap_wz(v, env: BodyEnvelope):
    """The coupled cap's yaw rate at speed ``|v|``: piecewise-linear through ``env.cap`` in ``|v|``.

    ``v = 0`` -> ``cap[0].wz``; between two points the straight line through them; CONSTANT at the
    last point's ``wz`` beyond it (the frontier is not extrapolated past what was measured). Tensor
    in -> tensor out (same shape / dtype / device); a Python number in -> a float out, computed in
    float64 through the same expression.
    """
    if isinstance(v, torch.Tensor):
        v_pts, w_pts = body_cap_tables(env, device=v.device, dtype=v.dtype)
        return body_cap_wz_from_tables(v.abs(), v_pts, w_pts)
    a = torch.tensor([abs(float(v))], dtype=torch.float64)
    v_pts, w_pts = body_cap_tables(env, dtype=torch.float64)
    return float(body_cap_wz_from_tables(a, v_pts, w_pts)[0])


def body_v_turn_max(env: BodyEnvelope) -> float:
    """The largest ``|v|`` at which a moving row may still turn: ``body_cap_wz(v) >= w_min``.

    ``inf`` when the cap never drops below ``w_min`` (the last point's ``wz >= w_min``); ``0.0`` when
    even ``v = 0`` sits below it (no moving turn is commandable -- the command term refuses such an
    envelope, because a move+turn row could then not keep its walk-floor speed). Otherwise the
    crossing on the first segment that leaves ``[w_min, inf)``, by the same linear interpolation
    :func:`body_cap_wz` uses (so ``body_cap_wz(body_v_turn_max(env)) == w_min`` up to rounding;
    callers clamp the band, never re-derive it).
    """
    w_min = float(env.w_min)
    cap = tuple((float(v), float(wz)) for v, wz in env.cap)
    if cap[-1][1] >= w_min:
        return math.inf
    last = None
    for j, (_v, wz) in enumerate(cap):
        if wz >= w_min:
            last = j
    if last is None:
        return 0.0
    (v0, w0), (v1, w1) = cap[last], cap[last + 1]
    return v0 + (w_min - w0) * (v1 - v0) / (w1 - w0)


def sample_yaw_rate(u_mag: torch.Tensor, u_sign: torch.Tensor, lo, hi) -> torch.Tensor:
    """A signed yaw-rate command from two uniforms: ``|wz| = lo + u_mag (hi - lo)``.

    The sign is :func:`sample_facing`'s convention EXACTLY -- ``u_sign < 0.5`` -> ``-1``, else
    ``+1`` -- so the body sampler, which feeds this the same two "facing" uniforms the world sampler
    draws, turns the same rows the same way the world sampler's facing change would. ``lo`` / ``hi``
    are floats or per-row tensors (the coupled cap gives every moving row its own ``hi``).
    """
    mag = lo + u_mag * (hi - lo)
    sign = torch.where(u_sign < 0.5, -torch.ones_like(u_mag), torch.ones_like(u_mag))
    return sign * mag


def slot_from_body_cmd(vx: torch.Tensor, vy: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
    """``slot = moving + 2 turning`` of a body command: ``moving = (vx, vy) != 0``, ``turning = wz != 0``.

    The 2-bit encoding of :data:`SLOT_STAND` ... :data:`SLOT_MOVE_TURN`; ``int64``. In body mode the
    slot is a pure function of the held command (stand IS ``(0, 0, 0)``, no latch), which is why the
    sampler's type-table slot must equal this on every draw (pinned by the unit tests).
    """
    moving = (vx != 0.0) | (vy != 0.0)
    turning = wz != 0.0
    return moving.to(torch.long) + 2 * turning.to(torch.long)


def body_cmd_violation_masks(vx, vy, wz, env: BodyEnvelope, ellipse, walk_floor: float, tol: float = 1e-6):
    """``{kind: (N,) bool}`` -- the ONE envelope definition, per violation kind (tensor path).

    Kinds: ``nonfinite`` (any component), ``speed`` (``|v|`` not in ``{0} U [floor, cap(atan2(vy,
    vx))]``), ``wz`` (``|wz|`` not in ``{0} U [w_min, w_max]``), ``move_turn_cap`` (moving AND
    turning with ``|wz| > body_cap_wz(|v|)``). Zero is EXACT on both axes (the slot rule's own
    ``!= 0``); every bound is widened by ``tol``.
    """
    a, b, c = (float(x) for x in ellipse)
    floor = float(walk_floor)
    finite = torch.isfinite(vx) & torch.isfinite(vy) & torch.isfinite(wz)
    speed = torch.hypot(vx, vy)
    cap = ellipse_cap(torch.atan2(vy, vx), a, b, c)
    moving = (vx != 0.0) | (vy != 0.0)
    turning = wz != 0.0
    awz = torch.abs(wz)
    speed_bad = moving & ((speed < floor - tol) | (speed > cap + tol))
    wz_bad = turning & ((awz < float(env.w_min) - tol) | (awz > float(env.w_max) + tol))
    coupled_bad = moving & turning & (awz > body_cap_wz(speed, env) + tol)
    return {
        "nonfinite": ~finite,
        "speed": speed_bad & finite,
        "wz": wz_bad & finite,
        "move_turn_cap": coupled_bad & finite,
    }


def body_cmd_violation_mask(vx, vy, wz, env: BodyEnvelope, ellipse, walk_floor: float, tol: float = 1e-6):
    """``(N,)`` bool: the rows that violate the envelope in ANY way (the batch helper)."""
    masks = body_cmd_violation_masks(vx, vy, wz, env, ellipse, walk_floor, tol)
    out = torch.zeros_like(masks["nonfinite"])
    for m in masks.values():
        out = out | m
    return out


def body_cmd_violations(vx: float, vy: float, wz: float, env: BodyEnvelope, ellipse, walk_floor: float,
                        tol: float = 1e-6) -> tuple[str, ...]:
    """The float path: one body command -> a tuple of short strings, ``()`` when it is inside.

    Computed in float64 through :func:`body_cmd_violation_masks` -- the same definition, never a
    re-spelling -- so the eval hook / validator and the sampler's tests cannot disagree.
    """
    t = [torch.tensor([float(x)], dtype=torch.float64) for x in (vx, vy, wz)]
    masks = body_cmd_violation_masks(*t, env, ellipse, walk_floor, tol)
    if bool(masks["nonfinite"][0]):
        return (f"non-finite command ({vx}, {vy}, {wz})",)
    a, b, c = (float(x) for x in ellipse)
    speed = math.hypot(float(vx), float(vy))
    out = []
    if bool(masks["speed"][0]):
        cap = float(ellipse_cap(torch.atan2(t[1], t[0]), a, b, c)[0])
        out.append(f"speed {speed:.6g} m/s not in {{0}} U [{float(walk_floor):.6g}, {cap:.6g}] "
                   f"(bearing {math.degrees(math.atan2(float(vy), float(vx))):.1f} deg)")
    if bool(masks["wz"][0]):
        out.append(f"|wz| {abs(float(wz)):.6g} rad/s not in {{0}} U [{float(env.w_min):.6g}, "
                   f"{float(env.w_max):.6g}]")
    if bool(masks["move_turn_cap"][0]):
        out.append(f"|wz| {abs(float(wz)):.6g} rad/s above the move+turn cap "
                   f"{body_cap_wz(speed, env):.6g} at |v| {speed:.6g} m/s")
    return tuple(out)


def shape_body_command(vx, vy, wz, env: BodyEnvelope, ellipse, walk_floor: float, turns: bool = True):
    """Map ANY finite ``(vx, vy, wz)`` into the envelope -> ``(vx, vy, wz, clipped)`` (PLAN D6).

    The deployment mapper and the eval's ``world_rate`` adapter (a body head driven on the legacy
    world rows: the relay's output passed through here, for the head's trained envelope). In order:

    1. ``turns=False`` (a stage with no turn slots) -> ``wz = 0``;
    2. the yaw dead zone: ``0 < |wz| < w_min`` -> 0;
    3. speed: ``0 < |v| < floor`` -> ``floor``; above the ellipse cap at the command's bearing ->
       the cap; the BEARING is kept either way;
    4. moving AND turning: if ``body_cap_wz(|v|) < w_min`` the speed drops to
       :func:`body_v_turn_max`, then ``|wz|`` is clamped to ``[w_min, body_cap_wz(|v|)]``; a pure
       turn clamps ``|wz|`` to ``[w_min, w_max]``. The sign of ``wz`` is kept.

    ``clipped`` is True where any component moved by more than 1e-9. A component the rules did not
    touch is returned BITWISE (the speed is only re-composed from its bearing when it changed), and
    the result always satisfies ``body_cmd_violations == ()`` (pinned by the unit tests).

    Tensor inputs (any one shape) -> tensors + a bool tensor; Python numbers -> floats + a bool.

    Raises:
        ValueError: an ellipse whose smallest semi-axis is below the walk floor, or an envelope whose
            :func:`body_v_turn_max` is below it (no moving turn could keep the floor speed) -- the
            same two refusals the command term makes at construction.
    """
    scalar = not isinstance(vx, torch.Tensor)
    if scalar:
        vx, vy, wz = (torch.tensor([float(x)], dtype=torch.float64) for x in (vx, vy, wz))
    a, b, c = (float(x) for x in ellipse)
    floor = float(walk_floor)
    if min(a, b, c) < floor:
        raise ValueError(f"shape_body_command: ellipse {(a, b, c)} dips below the walk floor {floor}")
    v_turn = body_v_turn_max(env)
    if v_turn < floor:
        raise ValueError(
            f"shape_body_command: body_v_turn_max {v_turn} m/s is below the walk floor {floor} m/s -- "
            "this envelope admits no moving turn"
        )
    w_min, w_max = float(env.w_min), float(env.w_max)
    zero = torch.zeros_like(wz)
    w = wz if turns else zero
    w = torch.where((w != 0.0) & (torch.abs(w) < w_min), zero, w)

    speed = torch.hypot(vx, vy)
    bearing = torch.atan2(vy, vx)
    moving = (vx != 0.0) | (vy != 0.0)
    v = torch.where(moving & (speed < floor), torch.full_like(speed, floor), speed)
    v = torch.minimum(v, ellipse_cap(bearing, a, b, c))
    turning = w != 0.0
    move_turn = moving & turning
    reduce = move_turn & (body_cap_wz(v, env) < w_min)
    v = torch.where(reduce, torch.full_like(v, v_turn), v)
    cap_w = body_cap_wz(v, env)
    hi = torch.where(moving, torch.maximum(cap_w, torch.full_like(cap_w, w_min)), torch.full_like(cap_w, w_max))
    aw = torch.minimum(torch.maximum(torch.abs(w), torch.full_like(w, w_min)), hi)
    w = torch.where(turning, torch.sign(w) * aw, zero)

    changed = v != speed
    nvx = torch.where(changed, v * torch.cos(bearing), vx)
    nvy = torch.where(changed, v * torch.sin(bearing), vy)
    clipped = (torch.abs(nvx - vx) > 1e-9) | (torch.abs(nvy - vy) > 1e-9) | (torch.abs(w - wz) > 1e-9)
    if scalar:
        return float(nvx[0]), float(nvy[0]), float(w[0]), bool(clipped[0])
    return nvx, nvy, w, clipped


def body_hold_step(face, yaw, wz, dt: float, windup: float = 0.6):
    """The ``body_hold`` adapter's facing integrator (PLAN D6): ``yaw + clamp(wrap(face + wz dt - yaw), +-windup)``.

    Drives a WORLD-trained head (which follows an absolute facing target) on a body row ``wz``: the
    target advances at the commanded rate, and the anti-windup bound keeps it at most ``windup`` rad
    ahead of (or behind) the robot's actual yaw, so a head that turns slower than asked is pulled
    along at its relay's saturation instead of accumulating a target it can never reach. Tensors or
    Python numbers (floats in, float out).
    """
    if isinstance(face, torch.Tensor) or isinstance(yaw, torch.Tensor) or isinstance(wz, torch.Tensor):
        face_t = torch.as_tensor(face)
        lead = wrap_to_pi(face_t + wz * dt - yaw)
        return yaw + torch.clamp(lead, -float(windup), float(windup))
    lead = float(wrap_to_pi(torch.tensor([float(face) + float(wz) * float(dt) - float(yaw)],
                                         dtype=torch.float64))[0])
    return float(yaw) + max(-float(windup), min(float(windup), lead))


class BodyHoldRow(NamedTuple):
    """A WORLD hold row (``CrabHexOmniCommand.schedule_hook``'s ``speed`` / ``dir`` / ``face``)."""

    speed: float
    dir: float
    face: float


def body_hold_world_row(vx, vy, wz, face, yaw) -> BodyHoldRow:
    """A body row -> the world row the ``body_hold`` adapter hands a world head this step.

    ``speed = |v|``, ``dir = yaw + atan2(vy, vx)`` (the body bearing laid onto the CURRENT yaw, so
    the world direction rotates with the robot, as a body command does), ``face`` = the integrator's
    target (:func:`body_hold_step`). ``wz`` is accepted for the call's symmetry and is consumed by
    :func:`body_hold_step`, not here. Floats or tensors.
    """
    del wz
    if isinstance(vx, torch.Tensor):
        return BodyHoldRow(torch.hypot(vx, vy), yaw + torch.atan2(vy, vx), face)
    return BodyHoldRow(math.hypot(float(vx), float(vy)), float(yaw) + math.atan2(float(vy), float(vx)), float(face))
