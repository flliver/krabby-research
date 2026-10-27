# SPDX-License-Identifier: BSD-3-Clause
"""The omnidirectional command term: a world travel direction + speed, and a world facing.

Omni plan D1. The forward and sideways tasks each train ONE travel axis fixed in the body
frame (``vel_command_b[:, 0]`` is "speed along the travel axis", steered by a heading
P-control that keeps the chassis square to the course). This term instead samples

* a **world** travel direction ``theta`` with a speed ``v`` capped per direction by an
  ellipse (fore-aft ``a``, lateral ``b``, backward ``c``), and
* a **world** facing target ``psi`` (``heading_target``, the inherited name -- the
  observation and the facing reward read it),

independently, in one of four command-regime slots: stand / move / turn / move+turn.

WHAT THE POLICY SEES. ``vel_command_b`` keeps its shape and meaning: slots 0 and 1 are the
body-frame xy velocity command, slot 2 the yaw-rate command. The 2-D xy command is what
makes the stock ``track_lin_vel_xy_exp``, ``penalty_tracking_error_l1``,
``reward_forward_progress_along_command`` and ``RewardStrideLength`` correct without a
travel frame, which is why the omni cfg carries no ``travel_yaw`` (plan, "The omni cfg
carries no travel_yaw").

THE TWO SIGN-BEARING CONVENTIONS (see ``crab_hex_omni_command_math`` for the full frame
note, which this module deliberately does not repeat):

* ``phi = wrap(theta - yaw)`` -- the world direction seen from the body; ``+pi/2`` is the
  robot's left (body ``+y``).
* the world direction is HELD while the body turns and only the magnitude is re-capped
  ("clamp-only re-cap"), so a forward command at ``a`` becomes a lateral command at ``b``
  in the same world direction after a 90 deg turn. ``recap_frac`` logs how often that bites.

THE BODY-FRAME MODE (robot-frame velocity commands campaign 2026-10-15, ``experiments/
2026-10-15_body_cmd``; DEFAULT OFF). :class:`CrabHexOmniBodyCommand` / :class:`CrabHexOmniBodyCommandCfg`
(bottom of this file) command a HELD robot-frame ``(vx, vy, wz)`` per slot instead -- drawn from the
same uniforms in the same order, inside one measured envelope -- and are reached only through
``KRABBY_OMNI_COMMAND_FRAME=body`` (``omni_knobs._apply_command_frame``). Both classes answer
:meth:`CrabHexOmniCommand.command_signature` / :meth:`CrabHexOmniCommand.intended_bearing_b`, the
two identity questions the stroke / centring / telemetry consumers ask (world: bitwise the old
expressions).

IMPORTS. The pure math lives in the sibling module and is reached by a RELATIVE import, so
this module loads under any package name -- the real
``parkour_tasks.crab_hex_omni_task.mdp`` in Isaac, a synthetic one in the CPU unit test
(the pattern of ``uniform_parkour_command.py``'s ``from ..travel_frame import ...``).
``CrabHexOmniCommand`` is defined BEFORE ``CrabHexOmniCommandCfg`` because the cfg's
``class_type`` default is the class object itself.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp.parkour_commands import ParkourCommandCfg, UniformParkourCommand
from parkour_isaaclab.envs.mdp.parkour_commands.command_sampling import WALK_MARGIN_M_S

from .crab_hex_omni_command_math import (
    N_SLOTS,
    POOL_TYPES,
    SLOT_MOVE,
    SLOT_MOVE_TURN,
    SLOT_NAMES,
    SLOT_STAND,
    SLOT_TURN,
    TELEMETRY_CATEGORIES,
    BodyEnvelope,
    body_cap_tables,
    body_cap_wz_from_tables,
    body_cmd_violations,
    body_command_details,
    body_v_turn_max,
    check_body_envelope,
    check_relay_hyst,
    check_row_ang_limits,
    ellipse_cap,
    heading_rate_command,
    heading_rate_command_hyst,
    heading_rate_command_rows,
    parse_cmd_pool,
    pool_speed_scale_floor,
    pool_tables,
    pool_type_min_cap,
    sample_facing,
    sample_pool_travel_dir,
    sample_slots,
    sample_travel_dir,
    sample_yaw_rate,
    slot_from_body_cmd,
    slot_is_moving,
    slot_is_turning,
    telemetry_category as _telemetry_category,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedEnv


MIN_CAP_HEADROOM_M_S = 0.01
"""Minimum width of the ``[walk_floor, cap]`` speed band, in every direction.

``__init__`` refuses an ellipse whose smallest semi-axis is not at least this far above the
walk floor: below it a "moving" slot would draw a speed indistinguishable from the floor
(and, at or under the floor, from a stop -- the gait clock freezes at
``CLOCK_CMD_STOP_M_S``). Separate constant from ``WALK_MARGIN_M_S`` even though both are
0.01 m/s today: one is the gap the sampler leaves ABOVE the clip, this one is the width the
band must still have."""

POOL_SLOT_MATCH_TOL = 1e-6
"""How far a pool's slot projection may sit from ``slot_probs`` before construction refuses.

The two are the same distribution written twice -- once per TYPE (``cmd_pool``, what the
sampler draws) and once per SLOT (``slot_probs``, what every metric, reward and eval hold
downstream reads). A mismatch is invisible: the run trains one regime and reports another.
1e-6 is the round-trip error of ``render_cmd_pool`` -> ``parse_cmd_pool`` -> projection with
``repr``-precision floats, i.e. exactly zero in practice and a real tolerance only against
the hand-written decimals a launcher types into ``KRABBY_OMNI_SLOTS``. The campaign driver's
recipe validator applies the SAME comparison at load, so a bad recipe fails before a GPU is
booked."""


# ----------------------------------------------------------------- debug-vis constants
#
# MODULE-LEVEL, not cfg fields, DELIBERATELY. ``CrabHexOmniCommandCfg``'s docstring promises
# that "the config-dump delta is exactly this list" and ``test_crab_hex_omni_phases.py``
# pins the omni-vs-forward env key delta; three new visualiser cfg fields would widen that
# delta for something that is never read during training (debug vis is off in the train cfg
# -- ``crab_hex_omni_env_cfg._apply_play`` is the only place that turns it on). Keeping the
# markers here leaves every config dump bitwise identical.

ARROW_Z_ACHIEVED_M = 0.45
ARROW_Z_TRAVEL_M = 0.62
ARROW_Z_FACING_M = 0.79
"""Heights above the base origin, metres, of the three arrows -- achieved below travel
below facing. They share an xy position every frame, so a shared height would z-fight and
read as one bent arrow; the order makes the stack say, top-down, "face this way / go this
way / actually going that way". 0.45 clears the chassis top in the PLAY chase camera."""

ARROW_SHAFT_SCALE = 0.17
"""y/z scale of the speed arrows -- their THICKNESS, ~0.04 m. NOT the stock 0.5 of
``parkour_command_cfg``: at the crab's 0.2-0.35 m/s the stock length multiplier leaves an
arrow WIDER than it is long, and the first render of this viewer was three coloured blobs.
Paired with the gains below this is roughly 11:1 and reads as an arrow at a glance."""

FACING_SHAFT_SCALE = 0.13
"""Thickness of the facing arrow -- thinner still, so "long and thin = facing" and "shorter
and fatter = speed" are separable even before the colour registers."""

TRAVEL_ARROW_GAIN = 25.0
"""Marker x-scale per m/s of commanded speed, for BOTH speed-proportional arrows.

A SCALE, not a length: ``arrow_x.usd`` is about 0.07 m long at unit scale (measured off the
2026-09-18 headless renders -- the asset is streamed from Nucleus, so there is nothing local
to read it from), which makes this about 1.7 m of arrow per m/s and puts the phase-default
band 0.21-0.35 m/s at roughly 0.37-0.6 m -- a chassis length or two, visible in the PLAY
chase camera without swamping the robot. The stock ``_resolve_xy_velocity_to_arrow`` factor
of 3.0 was picked against quadrupeds moving at 1 m/s and is invisible here. A stopped
command is zero-length, i.e. invisible, which is the "hidden when the slot commands no
travel" requirement."""

FACING_ARROW_LEN_FACTOR = 10.0
"""FIXED marker x-scale of the facing arrow (~0.7 m). Longer than any travel arrow the
ellipse can produce, so "the long orange one is the facing" survives a glance."""

TRAVEL_ARROW_COLOR = (0.0, 1.0, 0.0)
FACING_ARROW_COLOR = (1.0, 0.5, 0.0)
ACHIEVED_ARROW_COLOR = (0.0, 0.35, 1.0)


def _omni_arrow_cfg(prim_path: str, color: tuple[float, float, float], scale: tuple[float, float, float]):
    """A recoloured private copy of the stock x-arrow marker.

    DEEP copy, not ``.replace()``: ``configclass``'s ``replace`` is ``dataclasses.replace``,
    which is SHALLOW -- the returned cfg shares the ``markers`` dict (and therefore the one
    ``arrow`` prototype object) with the module-level ``GREEN_ARROW_X_MARKER_CFG``. That is
    how ``parkour_command_cfg.py`` ends up mutating the global arrow's scale at import; this
    module must not add a second such edit, because the colour would then leak into every
    other task's green arrow.
    """
    cfg = copy.deepcopy(GREEN_ARROW_X_MARKER_CFG)
    cfg.prim_path = prim_path
    arrow = cfg.markers["arrow"]
    arrow.scale = scale
    arrow.visual_material.diffuse_color = color
    return cfg


# x scale 1.0 on every prototype, so the per-frame ``length_factor`` IS the arrow's marker
# x-scale and the constants above are the only place a length is decided.
OMNI_TRAVEL_CMD_MARKER_CFG = _omni_arrow_cfg(
    "/Visuals/Command/omni_travel_cmd", TRAVEL_ARROW_COLOR, (1.0, ARROW_SHAFT_SCALE, ARROW_SHAFT_SCALE)
)
OMNI_FACING_CMD_MARKER_CFG = _omni_arrow_cfg(
    "/Visuals/Command/omni_facing_cmd", FACING_ARROW_COLOR, (1.0, FACING_SHAFT_SCALE, FACING_SHAFT_SCALE)
)
OMNI_ACHIEVED_VEL_MARKER_CFG = _omni_arrow_cfg(
    "/Visuals/Command/omni_achieved_vel", ACHIEVED_ARROW_COLOR, (1.0, ARROW_SHAFT_SCALE, ARROW_SHAFT_SCALE)
)


# --------------------------------------------------------------------- debug-vis helpers
# Pure (tensor in, tensor out), module-level and unit-tested on CPU: the marker call itself
# is untestable without Kit, and a silently wrong arrow is exactly the failure this viewer
# exists to prevent.


def arrow_quat_from_world_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """``(N,)`` WORLD yaw -> ``(N, 4)`` wxyz quaternion aiming the arrow's ``+x`` along it.

    Yaw-only ON PURPOSE. The stock helper post-multiplies by the full ``root_quat_w``, so
    its arrow tilts with the chassis roll/pitch; all three omni arrows are world-horizontal
    instead, which is what makes "is the orange arrow left or right of the nose?" a
    reliable read on rough ring terrain.
    """
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)


def arrow_scale_from_length(default_scale, length_factor: torch.Tensor) -> torch.Tensor:
    """``(N, 3)`` marker scale: the prototype's scale with ``x`` times ``length_factor``.

    Same shape of arithmetic as ``_resolve_xy_velocity_to_arrow`` (only the ``x`` axis --
    the arrow's long axis -- is stretched, so the shaft stays the same thickness).
    """
    scale = torch.as_tensor(default_scale, device=length_factor.device, dtype=length_factor.dtype)
    scale = scale.unsqueeze(0).repeat(length_factor.shape[0], 1)
    scale[:, 0] = scale[:, 0] * length_factor
    return scale


def marker_pos_above_base(root_pos_w: torch.Tensor, z_offset: float) -> torch.Tensor:
    """``(N, 3)`` world marker position: the base origin raised by ``z_offset`` metres."""
    pos = root_pos_w.clone()
    pos[:, 2] = pos[:, 2] + z_offset
    return pos


def omni_walk_floor(cfg: ParkourCommandCfg) -> float:
    """The lowest speed a moving slot may command, m/s.

    ``clips.lin_vel_clip + WALK_MARGIN_M_S`` -- derived from the shared constants, not
    hard-coded: it is exactly the ``walking_band`` floor of the forward task's explicit
    Bernoulli sampler (``command_sampling.walking_band``), i.e. the lowest command that
    still runs the gait clock (the clock stops at ``CLOCK_CMD_STOP_M_S`` = the stock clip
    0.2). With the stock clip this is 0.21 m/s.
    """
    return float(cfg.clips.lin_vel_clip) + WALK_MARGIN_M_S


def _row_field(row, name: str):
    """Read ``name`` off an eval-harness hold row, which may be a mapping or an object."""
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _validate_omni_cfg(cfg: CrabHexOmniCommandCfg) -> None:
    """Fail loudly at construction on any cfg the sampler cannot honour.

    Every check here is silent-failure insurance: a too-small ellipse would give an empty
    or inverted speed band (negative speeds along some directions), a bad slot mix would
    train the wrong regime, and ``ang_vel_max <= ang_vel_clip`` would zero every yaw-rate
    command the P-control ever produces (the facing half of the task would be inert with no
    error anywhere).
    """
    ellipse = tuple(float(x) for x in cfg.speed_ellipse)
    if len(ellipse) != 3:
        raise ValueError(f"speed_ellipse must be (a, b, c), got {cfg.speed_ellipse!r}")
    a, b, c = ellipse
    if not all(math.isfinite(x) and x > 0.0 for x in ellipse):
        raise ValueError(f"speed_ellipse semi-axes must be finite and positive, got {ellipse}")
    floor = omni_walk_floor(cfg)
    if not min(ellipse) > floor + MIN_CAP_HEADROOM_M_S:
        raise ValueError(
            "speed_ellipse is too small for the walk floor: "
            f"min(a={a}, b={b}, c={c}) = {min(ellipse)} must exceed "
            f"lin_vel_clip({cfg.clips.lin_vel_clip}) + WALK_MARGIN_M_S({WALK_MARGIN_M_S}) "
            f"+ MIN_CAP_HEADROOM_M_S({MIN_CAP_HEADROOM_M_S}) = {floor + MIN_CAP_HEADROOM_M_S}"
        )

    probs = tuple(float(p) for p in cfg.slot_probs)
    if len(probs) != N_SLOTS:
        raise ValueError(f"slot_probs must have {N_SLOTS} entries {SLOT_NAMES}, got {cfg.slot_probs!r}")
    if any(p < 0.0 or not math.isfinite(p) for p in probs) or sum(probs) <= 0.0:
        raise ValueError(f"slot_probs must be finite, non-negative and not all zero, got {probs}")

    lo, hi = (float(x) for x in cfg.facing_delta)
    if not (0.0 <= lo <= hi):
        raise ValueError(f"facing_delta must satisfy 0 <= lo <= hi, got ({lo}, {hi})")

    tlo, thi = (float(x) for x in cfg.travel_dir_range)
    if not tlo <= thi:
        raise ValueError(f"travel_dir_range must satisfy lo <= hi, got ({tlo}, {thi})")

    if cfg.dir_band is not None:
        dlo, dhi = (float(x) for x in cfg.dir_band)
        if not dlo <= dhi:
            raise ValueError(f"dir_band must satisfy lo <= hi, got ({dlo}, {dhi})")

    ang_vel_max = float(cfg.ang_vel_max)
    ang_vel_clip = float(cfg.clips.ang_vel_clip)
    if not ang_vel_max > 0.0:
        raise ValueError(f"ang_vel_max must be positive, got {ang_vel_max}")
    if not ang_vel_max > ang_vel_clip:
        raise ValueError(
            f"ang_vel_max ({ang_vel_max}) must exceed clips.ang_vel_clip ({ang_vel_clip}): "
            "otherwise the dead band zeroes every yaw-rate command, including saturated ones"
        )

    if cfg.cmd_pool is not None:
        pool = parse_cmd_pool(cfg.cmd_pool)  # grammar / names / weights, by name
        if cfg.dir_band is not None:
            raise ValueError(
                f"cmd_pool ({cfg.cmd_pool!r}) and dir_band ({tuple(cfg.dir_band)}) are two "
                "spellings of the same lever: the pool gives every TYPE its own direction "
                "band, so a global one would silently win or lose depending on the order the "
                "knobs were applied. Set KRABBY_OMNI_DIR_BAND='' when a pool is used."
            )
        projected = pool.projected_slot_probs()
        total = sum(float(p) for p in probs)
        declared = tuple(float(p) / total for p in probs)
        if any(abs(d - p) > POOL_SLOT_MATCH_TOL for d, p in zip(declared, projected)):
            raise ValueError(
                "cmd_pool and slot_probs describe different regimes: the pool projects onto "
                f"{projected} but slot_probs normalises to {declared} (tolerance "
                f"{POOL_SLOT_MATCH_TOL}). The two must agree -- every metric, reward and eval "
                "downstream reads the SLOT, so a mismatch trains one distribution and reports "
                "another. Set KRABBY_OMNI_SLOTS to the pool's projection."
            )
        # A ``speed_scale`` that pulls a MOVING type's cap under the walk floor inverts its
        # speed band (``floor + u * (cap - floor)`` with ``cap < floor`` draws BELOW the floor,
        # where the gait clock stops) -- and does it silently: the type is still in the dump
        # and still drawn at its weight. Same bound as the ellipse check above, per type, and
        # the SAME bound the driver's recipe validator is meant to apply at compose time --
        # see ``pool_speed_scale_floor``, which is exported so it is spelled in one place.
        min_speed = floor + MIN_CAP_HEADROOM_M_S
        for name, scale in zip(pool.names, pool.speed_scale):
            spec = POOL_TYPES[name]
            if not spec.slot & 1 or float(scale) == 1.0:
                continue  # a non-moving type never draws a speed; an unscaled one is covered
            scaled = pool_type_min_cap(name, a, b, c) * float(scale)
            if not scaled > min_speed:
                bound = pool_speed_scale_floor(name, (a, b, c), min_speed)
                raise ValueError(
                    f"cmd_pool type {name!r} has speed_scale {float(scale)}, which puts its "
                    f"smallest speed cap at {scaled} m/s -- at or below the walk floor "
                    f"({floor} + MIN_CAP_HEADROOM_M_S {MIN_CAP_HEADROOM_M_S} = {min_speed}). "
                    "Its speed band would be inverted and every draw would sit under the gait "
                    "clock's stop speed, so the type would silently train a creeper. The "
                    f"largest cut this type can take at speed_ellipse {(a, b, c)} and "
                    f"lin_vel_clip {float(cfg.clips.lin_vel_clip)} is speed_scale > {bound} "
                    "(crab_hex_omni_command_math.pool_speed_scale_floor); to go lower, widen "
                    "speed_ellipse or lower lin_vel_clip in the same overlay -- there is no "
                    "scale that both keeps this ellipse and stays above the clock's stop speed."
                )


class CrabHexOmniCommand(UniformParkourCommand):
    """Four-slot omnidirectional command: world travel direction + speed, world facing.

    Per-environment state (all world-frame unless said otherwise, all shape ``(N,)``):

    ``travel_dir_w``
        ``theta`` -- the world direction of travel this slot holds, radians. NOT wrapped
        to ``[-pi, pi]`` (see the math module); consumers wrap the difference.
    ``speed_cmd``
        ``v`` -- the commanded speed along ``theta``, m/s, already inside the ellipse at
        the resample instant. Exactly ``0.0`` on standing slots.
    ``heading_target``
        ``psi`` -- the world yaw to face (INHERITED name: the observation's steering
        channels, ``reward_facing_exp`` and the yaw-rate P-control all read it). Set to
        the current yaw on non-turning slots, i.e. "hold what you have".
    ``slot_id``
        ``int64`` command regime, ``SLOT_STAND | SLOT_MOVE | SLOT_TURN | SLOT_MOVE_TURN``.
    ``type_id``
        ``int64`` index into ``cfg.cmd_pool``'s type list, or ``-1`` for "not drawn from the
        pool" -- an eval ``schedule_hook`` row, an RSI-staged row, and EVERY row when no pool
        is configured. Always present, so a consumer never has to ask whether a pool is set;
        ``-1`` gathers the trailing UNIT entry of every per-type table
        (``crab_hex_omni_command_math.pool_tables``), i.e. the global speed cap and the global
        yaw-rate cap / dead band.
    ``rsi_cmd_staged``
        ``(N, 3)`` NaN-filled staging buffer written by the RSI reset event
        (``reset_from_reference_states_omni``, plan D6) with ``(theta, v, psi)`` in world
        frame. Reset events run BEFORE the command manager's resample, so a staged row's
        first command continues the reference frame's own momentum with zero facing error.
        Consumed EXACTLY ONCE: ``_resample_command`` reads it and writes NaN back.
    ``phi_b`` / ``recap_mask``
        Telemetry from the last ``_update_command``: the body-frame travel direction, and
        whether the ellipse re-cap clamped the speed this step.

    Metrics (published by the command manager as ``Metrics/base_velocity/<key>``):
    the inherited ``error_vel_xy`` (recomputed here on the full 2-D command -- there is no
    travel frame) and ``error_vel_yaw``; ``stand_frac_actual`` (redefined: slot-based, not
    ``vel_command_b[:, 0] == 0``, which would count every pure-lateral command as standing)
    plus ``move_frac_actual`` / ``turn_frac_actual`` / ``move_turn_frac_actual``;
    ``facing_err_abs`` (mean ``|wrap(psi - yaw)|`` over the episode), ``recap_frac`` and
    ``yaw_progress`` (SIGNED mean ``wz_b * sign(cmd_wz)`` over the turning rows -- see
    :meth:`_yaw_progress_step`; the one metric that distinguishes "turns the wrong way" from
    "does not turn", which is what the 2026-09-17 yaw smoke had to be re-derived offline to
    learn). WITH A POOL, and only then, two more families appear -- ``pool_<name>_frac``
    (step-weighted, the slot fracs' per-type twin) and ``pool_<name>_draws`` (the realised
    draw rates, to be read against the pool's own weights); a shipped pool-less run logs
    exactly the nine names above.

    DEBUG VISUALISATION (PLAY only -- ``_apply_play`` in ``crab_hex_omni_env_cfg`` is the
    only place that sets ``debug_vis``; a training run never builds a marker). Three arrows
    stacked over the base origin, ALL in the WORLD frame and ALL horizontal (yaw-only
    quaternions -- see :func:`arrow_quat_from_world_yaw`):

    ========  ======================  =============================  ========================
    colour    meaning                 direction (world)              length
    ========  ======================  =============================  ========================
    GREEN     commanded TRAVEL        ``travel_dir_w`` (theta)       ~1.7 m per m/s of the
              ("go this way")                                        commanded speed, so
                                                                     0.35-0.6 m across the
                                                                     phase speed band; ZERO
                                                                     (invisible) on a
                                                                     standing slot
    ORANGE    commanded FACING        ``heading_target`` (psi), an   fixed ~0.7 m -- longer
              ("face this way")       ABSOLUTE world yaw; on a       than any travel arrow,
                                      non-turning slot it is the     on a thinner shaft
                                      yaw held at resample, so it
                                      lies on the nose
    BLUE      ACHIEVED base velocity  ``atan2(root_lin_vel_w)``      ~1.7 m per m/s
    ========  ======================  =============================  ========================

    Read it as: orange against the robot's nose = the facing error the yaw-rate command is
    working on; green against blue = the travel-tracking error; no green = standing. The
    four slots are readable off the pair without a fourth marker -- no green means
    ``stand`` or ``turn``, orange off the nose means ``turn`` or ``move_turn``.

    The base class's GREEN "goal" arrow is REPLACED, not kept (``_set_debug_vis_impl``
    deliberately does not call ``super()``). It draws ``vel_command_b[:, :2]`` rotated by
    the FULL base quaternion -- the same world travel direction this term already holds, so
    keeping it would put a second green arrow a few degrees off the first (the roll/pitch
    tilt) carrying no extra information. Nothing is lost: the green arrow above IS that
    command, resolved yaw-only and drawn at the length of the EFFECTIVE commanded speed
    (``|vel_command_b[:, :2]|``, i.e. after the per-step ellipse re-cap, so a clamped
    command visibly shortens -- that is what ``recap_frac`` counts). The base's BLUE
    "current" arrow keeps its meaning and is likewise redrawn world-horizontal, off
    ``root_lin_vel_w`` directly. ``cfg.goal_vel_visualizer_cfg`` and
    ``cfg.current_vel_visualizer_cfg`` are therefore unread by this term; they stay on the
    cfg (inherited) so no config dump moves.
    """

    cfg: CrabHexOmniCommandCfg

    def __init__(self, cfg: CrabHexOmniCommandCfg, env: ParkourManagerBasedEnv):
        # Before anything is allocated: a cfg the sampler cannot honour must not reach a
        # training run (every failure mode it guards is silent -- see _validate_omni_cfg).
        _validate_omni_cfg(cfg)
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.travel_dir_w = torch.zeros(n, device=dev)
        self.speed_cmd = torch.zeros(n, device=dev)
        self.slot_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.rsi_cmd_staged = torch.full((n, 3), float("nan"), device=dev)
        self.phi_b = torch.zeros(n, device=dev)
        self.recap_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # ``-1`` everywhere until a pool draws: "this row did not come from the pool". With no
        # pool configured it STAYS -1 for the whole run, so any consumer may read it
        # unconditionally, and every table gather lands on the sentinel (unit) entry.
        self.type_id = torch.full((n,), -1, dtype=torch.long, device=dev)
        # -- two OPTIONAL, DEFAULT-OFF command knobs (motion-quality campaign, 2026-09-22), read
        # off cfg attributes no shipped cfg carries (``getattr`` -> off), so the recorded path is
        # untouched. ``relay_hyst`` = ``(engage, release)`` rad (``KRABBY_OMNI_RELAY_HYST``): the
        # yaw-rate relay gets a per-env hysteresis latch (``heading_rate_command_hyst``) in place
        # of its dead band. ``stand_hold`` (``KRABBY_OMNI_STAND_HOLD``): the APPLIED yaw-rate
        # command is zero on every row whose slot is STAND -- keyed on ``slot_id``, never on
        # ``type_id`` (an eval hold sets ``type_id = -1``), so it is live under the eval's
        # schedule override too. Zero wz plus the stand slot's exactly-zero planar command is a
        # zero clock speed, i.e. the gait clock stops (``clock_command_speed``).
        hyst = getattr(cfg, "relay_hyst", None)
        self._relay_hyst = None if hyst is None else (float(hyst[0]), float(hyst[1]))
        if self._relay_hyst is not None:
            check_relay_hyst(*self._relay_hyst)
            self.relay_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._stand_hold = bool(getattr(cfg, "stand_hold", False))
        # ``eval_slot_latch``: see :meth:`schedule_hook` (eval only; the sampler latches anyway).
        self._slot_latch = bool(getattr(cfg, "eval_slot_latch", False))
        if self._slot_latch:
            self._hook_turning = torch.zeros(n, dtype=torch.bool, device=dev)
            # NOTE(p2r statue, 2026-10-09): rows reset since the last hook call -- see :meth:`reset`.
            self._latch_rearm = torch.zeros(n, dtype=torch.bool, device=dev)
        # ``stand_frac_actual`` already exists (base class); the rest are new.
        for key in ("move_frac_actual", "turn_frac_actual", "move_turn_frac_actual",
                    "facing_err_abs", "recap_frac", "yaw_progress"):
            self.metrics[key] = torch.zeros(n, device=dev)

        # -- the optional per-type command pool (plan: the capability curriculum).
        self._pool = None if cfg.cmd_pool is None else parse_cmd_pool(cfg.cmd_pool)
        self._pool_tables = (
            None if self._pool is None
            else pool_tables(self._pool, device=dev, dtype=self.travel_dir_w.dtype)
        )
        if self._pool_tables is not None:
            # The per-row turn limits ``_write_command`` will gather, validated ONCE here: the
            # per-step expression does not re-check them, because a ``bool(tensor)`` per step
            # is a host-device sync on CUDA (see ``check_row_ang_limits``).
            check_row_ang_limits(
                float(cfg.ang_vel_max) * self._pool_tables.ang_scale,
                float(cfg.clips.ang_vel_clip) * self._pool_tables.ang_scale,
            )
        # Lifetime draw counters, NOT per-episode: ``pool_<name>_draws`` is "does the sampler
        # really draw at the asked rates", and a running total converges to the weights
        # instead of jittering with a handful of per-episode draws. They are deliberately not
        # reset with the episode (see _update_metrics).
        if self._pool is not None:
            k = len(self._pool.names)
            self._pool_draws = torch.zeros(n, k, device=dev)
            self._pool_draw_n = torch.zeros(n, device=dev)
            # Per-type metrics exist ONLY with a pool, so the nine-name metric set of a
            # shipped (pool-less) run is untouched.
            for name in self._pool.names:
                self.metrics[f"pool_{name}_frac"] = torch.zeros(n, device=dev)
                self.metrics[f"pool_{name}_draws"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        msg = "CrabHexOmniCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}\n"
        msg += f"\tSlot probabilities {SLOT_NAMES}: {tuple(self.cfg.slot_probs)}\n"
        msg += f"\tSpeed ellipse (fore-aft, lateral, backward): {tuple(self.cfg.speed_ellipse)}\n"
        msg += f"\tWalk floor: {omni_walk_floor(self.cfg)} m/s\n"
        msg += f"\tFacing delta band: {tuple(self.cfg.facing_delta)} rad\n"
        msg += f"\tTravel direction: {'world ' + str(tuple(self.cfg.travel_dir_range)) if self.cfg.dir_band is None else 'body-relative ' + str(tuple(self.cfg.dir_band))}\n"
        msg += f"\tYaw-rate cap / dead band: {self.cfg.ang_vel_max} / {self.cfg.clips.ang_vel_clip} rad/s"
        if self._pool is not None:
            msg += f"\n\tCommand pool: {self.cfg.cmd_pool}"
        return msg

    # ------------------------------------------------------------------ implementation

    def _resample_command(self, env_ids: Sequence[int]):
        """Draw a new slot (or consume an RSI-staged command) for ``env_ids``.

        Draw order is FIXED -- slot, direction, speed, facing -- and every draw is taken
        for every row regardless of slot or staging, so the RNG stream a run consumes does
        not depend on the slot mix, on ``dir_band`` or on which rows the RSI event happened
        to stage. Only the *use* of a draw is conditional.

        TWO BRANCHES, ONE STREAM. With ``cfg.cmd_pool`` unset the block below is the shipped
        legacy sampler, character for character -- that is the invariant every recorded
        seed-3 run depends on, so it is kept whole under one ``if`` rather than threaded with
        per-row tensors that happen to be unit-valued. The pool branch draws the SAME five
        uniforms in the SAME order (one categorical, one direction, one speed, two facing),
        so a pool that describes a legacy distribution reproduces it bitwise.
        """
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        a, b, c = (float(x) for x in cfg.speed_ellipse)
        floor = omni_walk_floor(cfg)
        yaw = self.robot.data.heading_w[env_ids]

        if cfg.cmd_pool is None:
            slot = sample_slots(None, cfg.slot_probs, n, device=self.device)
            theta = sample_travel_dir(yaw, cfg.dir_band, cfg.travel_dir_range)
            u_speed = torch.rand(n, device=self.device)
            psi = sample_facing(yaw, float(cfg.facing_delta[0]), float(cfg.facing_delta[1]))

            # Speed band is direction-scaled: the cap is evaluated at the body-frame direction
            # the slot STARTS at (phi0). The per-step re-cap in ``_update_command`` then keeps
            # the command inside the ellipse as the body turns.
            phi0 = wrap_to_pi(theta - yaw)
            cap0 = ellipse_cap(phi0, a, b, c)
            speed = torch.where(
                slot_is_moving(slot),
                floor + u_speed * (cap0 - floor),
                torch.zeros_like(u_speed),
            )
            # Non-turning slots hold the facing they have (zero commanded facing change).
            psi = torch.where(slot_is_turning(slot), psi, yaw)

            # No pool: every row keeps the -1 sentinel, so ``_write_command`` gathers the unit
            # scales and downstream code can read ``type_id`` without asking whether one is set.
            type_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        else:
            tables = self._pool_tables
            # One uniform, K categories instead of 4 -- the SAME draw, read against a longer
            # cumulative vector; the slot each type projects onto comes from the table.
            type_id = sample_slots(None, self._pool.weights, n, device=self.device)
            slot = tables.slot[type_id]
            theta = sample_pool_travel_dir(
                yaw, type_id, tables.lo, tables.hi, tables.world, cfg.travel_dir_range
            )
            u_speed = torch.rand(n, device=self.device)
            psi = sample_facing(yaw, float(cfg.facing_delta[0]), float(cfg.facing_delta[1]))

            # As above, times the type's own ``speed_scale`` (1.0 for every type by default,
            # so this is bitwise the legacy cap unless a stage asked for a slower type).
            phi0 = wrap_to_pi(theta - yaw)
            cap0 = ellipse_cap(phi0, a, b, c) * tables.speed_scale[type_id]
            speed = torch.where(
                slot_is_moving(slot),
                floor + u_speed * (cap0 - floor),
                torch.zeros_like(u_speed),
            )
            psi = torch.where(slot_is_turning(slot), psi, yaw)

        # RSI-staged rows override all three and become a MOVE slot: the staged speed is
        # already clamped into [floor, cap] by the reset event and the staged psi is the
        # spawn yaw, so the frame starts on-command with zero facing error.
        staged = self.rsi_cmd_staged[env_ids]
        has_staged = torch.isfinite(staged).all(dim=1)
        theta = torch.where(has_staged, staged[:, 0], theta)
        speed = torch.where(has_staged, staged[:, 1], speed)
        psi = torch.where(has_staged, staged[:, 2], psi)
        slot = torch.where(has_staged, torch.full_like(slot, SLOT_MOVE), slot)
        # A staged row did not come from the pool: back to the sentinel, so it gets the global
        # turn cap in ``_write_command`` and counts in no ``pool_<name>_*`` metric.
        type_id = torch.where(has_staged, torch.full_like(type_id, -1), type_id)

        self.travel_dir_w[env_ids] = theta
        self.speed_cmd[env_ids] = speed
        self.heading_target[env_ids] = psi
        self.slot_id[env_ids] = slot
        self.type_id[env_ids] = type_id
        if self._relay_hyst is not None:
            # A FRESH command re-arms the latch: a commanded turn starts engaged whenever there is
            # more than ``release`` of it to do (so a small commanded turn below ``engage`` is not
            # swallowed); a held facing (psi = yaw) starts disengaged.
            self.relay_latch[env_ids] = torch.abs(wrap_to_pi(psi - yaw)) > self._relay_hyst[1]
        # Consumed exactly once: whatever was staged is gone even if this row was not
        # staged (writing NaN over NaN), so a stale row can never be replayed.
        self.rsi_cmd_staged[env_ids] = float("nan")

        if self._pool is not None:
            self._count_pool_draws(env_ids, type_id, has_staged)

        # Refresh the published command for these rows now. ``compute()`` calls
        # ``_update_command`` right after this, but ``reset()`` resamples WITHOUT it, and
        # the observation manager may run before the next ``compute``.
        self._write_command(env_ids)

    def _count_pool_draws(self, env_ids, type_id: torch.Tensor, has_staged: torch.Tensor) -> None:
        """Tally this resample's pool draws (staged rows excluded) into the lifetime counters.

        No ``if not drawn.any(): return`` short circuit: ``bool(tensor)`` is a host-device
        SYNCHRONISATION on CUDA, and the all-staged case it would skip is already a no-op
        (``scatter_`` writes 0.0 and both ``+=`` add zero). Same no-sync-in-the-hot-path rule
        as ``check_row_ang_limits`` -- this runs once per resample rather than once per step,
        but the guard bought nothing to pay a sync for.
        """
        drawn = ~has_staged
        one_hot = torch.zeros_like(self._pool_draws[env_ids])
        # ``clamp_min(0)`` only matters for the staged rows, whose contribution the ``drawn``
        # gate zeroes anyway -- it keeps ``scatter_`` from seeing the -1 sentinel as an index.
        one_hot.scatter_(1, type_id.clamp_min(0).unsqueeze(1), drawn.to(one_hot.dtype).unsqueeze(1))
        self._pool_draws[env_ids] += one_hot
        self._pool_draw_n[env_ids] += drawn.to(self._pool_draw_n.dtype)

    def _update_command(self):
        """Re-project the held world command into the CURRENT body frame (every step)."""
        self._write_command(slice(None))

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """The base reset; with the eval slot latch armed, the reset rows' latch is re-derived.

        NOTE(p2r statue, 2026-10-09; ``P2R_PLAN.md`` "Left for implementation"): an episode reset is
        a FRESH command, but :meth:`schedule_hook` re-latches the turning bit only where the facing
        target changed -- and a suite wave's reset re-spawns at the pinned yaw, where a non-turning
        resample's ``psi = yaw`` equals the wave's first face, so ~62 % of the move_turn wave's
        settle inherited the turn wave's latch. Here the reset rows' ``_hook_turning`` is cleared and
        they are flagged, so the next :meth:`schedule_hook` re-derives their turning bit from the raw
        relay output exactly as for a fresh target (the step-1 launcher's rule). Rebinding, never an
        in-place write: the latch buffers may be inference tensors after an eval step. Latch off (every
        training run without a stand-keyed knob): the base reset, bitwise.
        """
        extras = super().reset(env_ids)
        if getattr(self, "_slot_latch", False) and getattr(self, "_hook_turning", None) is not None:
            rows = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            rows[slice(None) if env_ids is None else env_ids] = True
            self._hook_turning = torch.where(rows, torch.zeros_like(self._hook_turning), self._hook_turning)
            rearm = getattr(self, "_latch_rearm", None)
            self._latch_rearm = rows if rearm is None else (rearm | rows)
        return extras

    def stand_mask(self) -> torch.Tensor:
        """``(N,)`` bool: the rows whose command regime is the STAND slot (by ``slot_id``).

        The one predicate every stand-keyed knob reads (``KRABBY_OMNI_STAND_HOLD`` here, the
        action term's ``KRABBY_OMNI_STAND_PARK`` clock park, the stand observation bit and the
        stand-pose penalty) -- keyed on the SLOT, never on ``type_id``: an eval hold carries
        ``type_id = -1`` while its ``slot_id`` is re-derived by :meth:`schedule_hook`.
        """
        return self.slot_id == SLOT_STAND

    TELEMETRY_CATEGORIES = TELEMETRY_CATEGORIES
    """Re-exported so a consumer holding the TERM (not this module) can name the indices
    :meth:`telemetry_category` returns, without importing ``parkour_tasks`` -- the per-category
    reward manager (``parkour_isaaclab``, which must not depend on any task package) reads it
    this way, duck-typed, off whatever command term ``reward_category_command`` names."""

    def telemetry_category(self) -> torch.Tensor:
        """``(N,)`` int64: this step's :data:`TELEMETRY_CATEGORIES` index, per env.

        Calls the pure :func:`crab_hex_omni_command_math.telemetry_category` on the COMMANDED
        pair -- ``slot_id`` and the intended bearing ``travel_dir_w - heading_target`` -- never
        on ``phi_b`` (the instantaneous body-frame command): a facing-tracking error must not
        move a row between categories mid-episode. The bearing is :meth:`intended_bearing_b` (the
        body-frame command mode's own held bearing in :class:`CrabHexOmniBodyCommand`; here
        bitwise the expression above).
        """
        return _telemetry_category(self.slot_id, self.intended_bearing_b())

    # ------------------------------------------------------------------ command identity
    # NOTE(body-frame command mode, 2026-10-15; STEP1_SPEC §2): the two questions every command
    # consumer asks -- "did a NEW command arrive?" and "which body bearing was commanded?" -- answered
    # by the term, so a consumer never spells them off the world fields (in the body mode those
    # follow the robot's yaw every step and would read as a new command on every step). The world
    # answers are BITWISE the expressions the consumers used before.

    def command_signature(self) -> torch.Tensor:
        """``(N, 3)``: changes exactly when a new command is drawn or hooked.

        World: ``(slot_id, travel_dir_w, heading_target)`` -- the tuple the stroke / centring terms
        compared before this method existed (``crab_hex_omni_stroke_rewards._OncePerStep``), bitwise.
        """
        return torch.stack((self.slot_id.float(), self.travel_dir_w, self.heading_target), dim=-1)

    def intended_bearing_b(self) -> torch.Tensor:
        """``(N,)``: the commanded body bearing, NOT wrapped (consumers wrap).

        World: ``travel_dir_w - heading_target`` -- the commanded travel direction relative to the
        commanded facing, exactly as :meth:`telemetry_category` and the centring term used it.
        """
        return self.travel_dir_w - self.heading_target

    def _write_command(self, env_ids, apply_hold: bool = True):
        """World command -> ``vel_command_b`` for ``env_ids`` (a tensor/list or a slice).

        The xy half is the global ellipse on BOTH paths: a pool type's ``speed_scale`` picks a
        slower speed at draw time, it does not narrow the clamp the body's own rotation runs
        into. Only the yaw-rate half is per-row, and only when a pool is configured.

        ``apply_hold=False`` skips the ``stand_hold`` zeroing (a no-op without that knob): the eval
        hook re-derives ``slot_id`` from the RAW relay output first and applies the hold after.
        """
        cfg = self.cfg
        a, b, c = (float(x) for x in cfg.speed_ellipse)
        yaw = self.robot.data.heading_w[env_ids]
        speed = self.speed_cmd[env_ids]
        cmd_xy, phi, cap = body_command_details(self.travel_dir_w[env_ids], speed, yaw, a, b, c)
        self.vel_command_b[env_ids, :2] = cmd_xy
        if self._relay_hyst is not None:
            # KRABBY_OMNI_RELAY_HYST (default off): the latch replaces the dead band on both
            # paths; a pool row keeps its own saturation (``ang_scale``), as below.
            ang_max = (float(cfg.ang_vel_max) if self._pool is None
                       else float(cfg.ang_vel_max) * self._pool_tables.ang_scale[self.type_id[env_ids]])
            wz, latch = heading_rate_command_hyst(
                self.heading_target[env_ids],
                yaw,
                float(cfg.heading_control_stiffness),
                ang_max,
                self.relay_latch[env_ids],
                self._relay_hyst[0],
                self._relay_hyst[1],
            )
            self.relay_latch[env_ids] = latch
            self.vel_command_b[env_ids, 2] = wz
        elif self._pool is None:
            self.vel_command_b[env_ids, 2] = heading_rate_command(
                self.heading_target[env_ids],
                yaw,
                float(cfg.heading_control_stiffness),
                float(cfg.ang_vel_max),
                float(cfg.clips.ang_vel_clip),
            )
        else:
            # ``ang_scale`` per row, gathered by ``type_id``; ``-1`` (eval holds, RSI-staged
            # rows) lands on the tables' trailing UNIT entry, i.e. exactly the global cap.
            scale = self._pool_tables.ang_scale[self.type_id[env_ids]]
            self.vel_command_b[env_ids, 2] = heading_rate_command_rows(
                self.heading_target[env_ids],
                yaw,
                float(cfg.heading_control_stiffness),
                float(cfg.ang_vel_max) * scale,
                float(cfg.clips.ang_vel_clip) * scale,
            )
        if self._stand_hold and apply_hold:
            self._apply_stand_hold(env_ids)
        self.phi_b[env_ids] = phi
        self.recap_mask[env_ids] = speed > cap

    def _apply_stand_hold(self, env_ids) -> None:
        """``KRABBY_OMNI_STAND_HOLD``: zero the applied yaw-rate command on the STAND-slot rows."""
        stand = self.slot_id[env_ids] == SLOT_STAND
        wz = self.vel_command_b[env_ids, 2]
        self.vel_command_b[env_ids, 2] = torch.where(stand, torch.zeros_like(wz), wz)

    def _update_metrics(self):
        """Override: the 2-D command has no travel frame, and standing is a SLOT."""
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        episode_steps = float(self._env.max_episode_length)
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        self.metrics["stand_frac_actual"] += (self.slot_id == SLOT_STAND).float() / episode_steps
        self.metrics["move_frac_actual"] += (self.slot_id == SLOT_MOVE).float() / episode_steps
        self.metrics["turn_frac_actual"] += (self.slot_id == SLOT_TURN).float() / episode_steps
        self.metrics["move_turn_frac_actual"] += (self.slot_id == SLOT_MOVE_TURN).float() / episode_steps
        self.metrics["facing_err_abs"] += (
            torch.abs(wrap_to_pi(self.heading_target - self.robot.data.heading_w)) / episode_steps
        )
        self.metrics["recap_frac"] += self.recap_mask.float() / episode_steps
        self.metrics["yaw_progress"] += self._yaw_progress_step() / episode_steps
        if self._pool is not None:
            self._update_pool_metrics(episode_steps)

    def _update_pool_metrics(self, episode_steps: float) -> None:
        """The two per-type metric families -- only ever called when a pool is configured.

        ``pool_<name>_frac``
            STEP-weighted, exactly like the four ``*_frac_actual`` siblings: the share of the
            episode's steps this env spent under that type. RSI-staged rows carry the -1
            sentinel and therefore count towards no type, so the pool fracs sum to the
            non-staged share of the episode while ``stand/move/turn/move_turn_frac_actual``
            still sum to 1 (a staged row IS in a slot -- ``SLOT_MOVE`` -- it just did not come
            from the pool).
        ``pool_<name>_draws``
            The share of this env's pool DRAWS (one per resample, staged rows excluded) that
            picked that type -- a LIFETIME running fraction, OVERWRITTEN each step rather than
            accumulated. It answers "is the sampler really drawing at the asked rates", which
            a per-episode count could not: an env sees only a handful of resamples per
            episode. Overwriting a metric is the parkour event's own ledger-mean pattern; the
            manager's reset zeroes it and the next step writes it back.

        ONE-HOT, not K comparisons: this runs every physics step of a pooled run, and a pooled
        stage is compared against a legacy stage on wall clock as well as on reward. The
        ``(type_id == index)`` scan and the ``_pool_draws[:, index] / drawn`` divide are
        hoisted out of the loop, which leaves only the K accumulator writes -- the same
        wall-clock hygiene ``check_row_ang_limits`` bought for the yaw-rate expression.
        Bitwise identical to the per-index form: a one-hot column IS ``type_id == index`` as a
        float, and the division is elementwise either way.
        """
        drawn = self._pool_draw_n.clamp_min(1.0)
        # ``clamp_min(0)`` folds the -1 sentinel onto column 0; ``active`` then zeroes that
        # whole row, so a staged / eval row counts towards NO type (never towards the first).
        active = (self.type_id >= 0).to(self._pool_draws.dtype).unsqueeze(1)
        one_hot = torch.nn.functional.one_hot(
            self.type_id.clamp_min(0), len(self._pool.names)
        ).to(self._pool_draws.dtype) * active
        shares = self._pool_draws / drawn.unsqueeze(1)
        for index, name in enumerate(self._pool.names):
            self.metrics[f"pool_{name}_frac"] += one_hot[:, index] / episode_steps
            self.metrics[f"pool_{name}_draws"][:] = shares[:, index]

    def _yaw_progress_step(self) -> torch.Tensor:
        """Mean ``wz_b * sign(cmd_wz)`` over the envs commanding a turn this step (0 if none).

        The SIGNED rotation telemetry the yaw-capability smoke did not have: its log showed
        ``facing_err_abs`` stuck at 0.27-0.43 and ``error_vel_yaw`` large, both of which are
        consistent with "turns the wrong way", with "does not turn at all" and with "turns
        correctly but too slowly". This is the one number that separates them -- it is exactly
        the quantity ``crab_hex_omni_rewards.reward_yaw_progress`` pays for (before its clip),
        so ``Metrics/base_velocity/yaw_progress`` and
        ``Episode_Reward/reward_yaw_progress`` must move together; positive means rotating
        TOWARDS the commanded facing. The eval harness computes the same quantity per hold as
        ``yaw_progress_rate`` (``gait_eval/metrics.py``), so a training print and an eval row
        are directly comparable.

        A SCALAR (broadcast into every row of the accumulator, then divided by the episode
        length like every sibling metric, so the logged value is the episode mean of the
        per-step mean). Averaging only over the turning rows, rather than over all envs, keeps
        the number comparable across slot mixes: a window with 10% turning slots would
        otherwise report a tenth of the same behaviour. Standing / pure-travel / settled rows
        are excluded by the same ``cmd_wz != 0`` gate the reward uses -- the P-control's dead
        band, so "a turn is actually commanded".
        """
        cmd_wz = self.vel_command_b[:, 2]
        turning = cmd_wz != 0.0
        progress = self.robot.data.root_ang_vel_b[:, 2] * torch.sign(cmd_wz)
        n_turning = turning.sum()
        total = torch.where(turning, progress, torch.zeros_like(progress)).sum()
        return total / n_turning.clamp_min(1).to(total.dtype)

    # --------------------------------------------------------------- debug visualisation

    _VIS_ATTRS = ("travel_cmd_visualizer", "facing_cmd_visualizer", "achieved_vel_visualizer")

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/show the three omni arrows. Does NOT call ``super()`` -- see the class
        docstring for why the base's green goal arrow is replaced rather than kept."""
        if debug_vis:
            if not hasattr(self, "travel_cmd_visualizer"):
                self.travel_cmd_visualizer = VisualizationMarkers(OMNI_TRAVEL_CMD_MARKER_CFG)
                self.facing_cmd_visualizer = VisualizationMarkers(OMNI_FACING_CMD_MARKER_CFG)
                self.achieved_vel_visualizer = VisualizationMarkers(OMNI_ACHIEVED_VEL_MARKER_CFG)
            for name in self._VIS_ATTRS:
                getattr(self, name).set_visibility(True)
        elif hasattr(self, "travel_cmd_visualizer"):
            for name in self._VIS_ATTRS:
                getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        """Redraw the three arrows for every env (vectorised -- ``num_envs`` rows each)."""
        if not self.robot.is_initialized:
            return
        root_pos_w = self.robot.data.root_pos_w
        # (1) commanded travel: the HELD world direction, at the length of the effective
        # commanded speed. ``body_command_details`` only ever clamps the MAGNITUDE, so the
        # world direction of ``vel_command_b`` is ``travel_dir_w`` exactly and taking the
        # angle from the state rather than from ``atan2`` keeps a zero-speed command
        # pointing somewhere defined instead of at atan2(0, 0).
        speed_eff = torch.linalg.norm(self.vel_command_b[:, :2], dim=1)
        self.travel_cmd_visualizer.visualize(
            marker_pos_above_base(root_pos_w, ARROW_Z_TRAVEL_M),
            arrow_quat_from_world_yaw(self.travel_dir_w),
            arrow_scale_from_length(
                self.travel_cmd_visualizer.cfg.markers["arrow"].scale, speed_eff * TRAVEL_ARROW_GAIN
            ),
        )

        # (2) commanded facing: an absolute world yaw, fixed length (it is an angle, not a
        # rate -- a length would invite reading it as "how hard to turn").
        self.facing_cmd_visualizer.visualize(
            marker_pos_above_base(root_pos_w, ARROW_Z_FACING_M),
            arrow_quat_from_world_yaw(self.heading_target),
            arrow_scale_from_length(
                self.facing_cmd_visualizer.cfg.markers["arrow"].scale,
                torch.full_like(self.heading_target, FACING_ARROW_LEN_FACTOR),
            ),
        )

        # (3) achieved base velocity, straight off the WORLD root velocity (no body->world
        # round trip, so this arrow cannot disagree with the command arrows about frames).
        vel_w = self.robot.data.root_lin_vel_w[:, :2]
        self.achieved_vel_visualizer.visualize(
            marker_pos_above_base(root_pos_w, ARROW_Z_ACHIEVED_M),
            arrow_quat_from_world_yaw(torch.atan2(vel_w[:, 1], vel_w[:, 0])),
            arrow_scale_from_length(
                self.achieved_vel_visualizer.cfg.markers["arrow"].scale,
                torch.linalg.norm(vel_w, dim=1) * TRAVEL_ARROW_GAIN,
            ),
        )

    # ------------------------------------------------------------------- eval harness

    def schedule_hook(self, row) -> None:
        """Pin the whole batch to one world-frame hold row (eval harness, plan D10).

        ``row`` carries ``speed`` (m/s), ``dir`` and ``face`` (WORLD radians) -- as a
        mapping or as attributes. The hook writes the same three world quantities the
        sampler writes and then runs the real ``_update_command``, so an eval hold goes
        through the identical world->body conversion and the identical closed-loop facing
        P-control that training uses; nothing about the command path is special-cased for
        eval. ``rsi_cmd_staged`` is deliberately untouched (RSI is off during evals).

        ``slot_id`` is re-derived from what the hook actually produced -- moving iff the
        commanded speed is non-zero, turning iff the yaw-rate command survived the dead
        band -- using the 2-bit slot encoding (``slot = moving + 2 * turning``). So a turn
        hold reports ``turn`` while it is still turning and ``stand`` once the facing error
        has settled inside the dead band.

        With ``eval_slot_latch`` armed (set by the motion-quality knobs whenever a STAND-keyed
        knob is -- ``KRABBY_OMNI_STAND_HOLD`` / ``_STAND_PARK`` / ``_STAND_OBS``; default OFF) the
        TURNING bit is instead LATCHED when a hold's facing target arrives (a fresh target: it
        differs from the held one, which is also true after any reset's resample) and kept for the
        rest of the hold -- the training sampler's rule, where a slot is drawn once and holds for
        its whole duration. Without it a stand hold that drifted past the dead band would re-derive
        as ``turn`` and escape the stand hold/park/bit, and a settled turn hold would re-derive as
        ``stand`` and receive them: the legacy re-derivation only ever calls a hold "stand" where
        the relay already outputs zero, so a stand-keyed knob would be a no-op in every eval.
        """
        speed = float(_row_field(row, "speed"))
        direction = float(_row_field(row, "dir"))
        face = float(_row_field(row, "face"))
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError(f"schedule_hook: speed must be finite and >= 0, got {speed}")
        if not (math.isfinite(direction) and math.isfinite(face)):
            raise ValueError(f"schedule_hook: dir/face must be finite, got ({direction}, {face})")
        # A NEW facing target is a fresh command (the sampler's re-arm point). Only the two
        # default-off latches below read it.
        fresh = self.heading_target != face
        if self._relay_hyst is not None:
            # Engaged if there is more than ``release`` of turn to do; an unchanged target keeps
            # its latch.
            armed = torch.abs(wrap_to_pi(face - self.robot.data.heading_w)) > self._relay_hyst[1]
            self.relay_latch[:] = torch.where(fresh, armed, self.relay_latch)
        self.travel_dir_w[:] = direction
        self.speed_cmd[:] = speed
        self.heading_target[:] = face
        # An eval hold did not come from the pool: clear the type BEFORE ``_write_command``,
        # so the yaw-rate cap it uses is the global one and a hold is never quietly run at
        # some type's reduced turn demand. (The hold's own validator sizes it against
        # ``ang_vel_max``, so a scaled cap would make a legal row unachievable.)
        self.type_id[:] = -1
        # The slot is re-derived from the RAW relay output, so the hold below cannot latch a
        # turn hold into "stand" (a zeroed wz would otherwise read as "not turning" next step).
        self._write_command(slice(None), apply_hold=False)
        moving = self.speed_cmd > 0.0
        turning = self.vel_command_b[:, 2] != 0.0
        if self._slot_latch:
            # A reset since the last hook call is a fresh command too (:meth:`reset`). ``getattr``: the
            # step-1 launcher arms the latch at runtime without the rearm buffer.
            rearm = getattr(self, "_latch_rearm", None)
            relatch = fresh if rearm is None else (fresh | rearm)
            self._hook_turning = torch.where(relatch, turning, self._hook_turning)
            if rearm is not None:
                self._latch_rearm = torch.zeros_like(rearm)
            turning = self._hook_turning
        self.slot_id[:] = moving.long() + 2 * turning.long()
        if self._stand_hold:
            # KRABBY_OMNI_STAND_HOLD (default off): a suite stand hold IS a stand slot here, so it
            # receives the hold exactly as a training stand slot does.
            self._apply_stand_hold(slice(None))

    def schedule_hook_body(self, vx, vy=None, wz=None) -> None:
        """REFUSED on a world term: a body row ``(vx, vy, wz)`` here is a route mismatch.

        NOTE(body-frame command mode, 2026-10-15): the eval harness must route body rows to a
        :class:`CrabHexOmniBodyCommand`, or drive this world term through a NAMED adapter
        (``body_hold``: ``crab_hex_omni_command_math.body_hold_step`` / ``body_hold_world_row`` into
        :meth:`schedule_hook`) -- an unnamed conversion would score something nobody chose.
        """
        raise RuntimeError(
            "schedule_hook_body (a BODY row: vx/vy/wz) called on the WORLD CrabHexOmniCommand: route "
            "mismatch -- use schedule_hook(row) or the named body_hold adapter"
        )


@configclass
class CrabHexOmniCommandCfg(ParkourCommandCfg):
    """Cfg for :class:`CrabHexOmniCommand` (plan D1 defaults).

    Fields the forward command cfg does NOT have -- so nothing the forward task builds can
    accidentally reach the omni behaviour, and the config-dump delta is exactly this list.

    Inherited-but-IGNORED by this term, kept so the shared knob block
    (``KRABBY_LIN_VEL_X``, ``KRABBY_STAND_FRAC``) still writes the same dump paths and the
    forward/omni dumps stay comparable: ``ranges.lin_vel_x``, ``ranges.heading``,
    ``stand_frac``, ``small_commands_to_zero``. A unit test pins that changing any of them
    changes nothing about the omni sampler.

    NOT declared here, deliberately: ``relay_hyst`` (``(engage, release)`` rad), ``stand_hold``
    and ``eval_slot_latch`` (bools) -- the motion-quality knobs (``KRABBY_OMNI_RELAY_HYST`` /
    ``KRABBY_OMNI_STAND_HOLD``; the latch is armed with any stand-keyed knob), see
    ``config/crab_hex/motion_knobs.py``. They are written onto the INSTANCE only when armed and
    read with ``getattr`` defaults, so a knob-off config dump is byte-identical to one built
    before they existed.
    """

    class_type: type = CrabHexOmniCommand

    slot_probs: tuple[float, float, float, float] = (0.15, 0.55, 0.10, 0.20)
    """Categorical weights over ``(stand, move, turn, move_turn)``. Normalised at sampling
    time. Phase-1 stages 1a/1b use ``(0.2, 0.8, 0, 0)``; 2a+ uses
    ``(0.15, 0.45, 0.15, 0.25)`` (knob ``KRABBY_OMNI_SLOTS``, parsed in step 7)."""

    speed_ellipse: tuple[float, float, float] = (0.35, 0.25, 0.25)
    """Direction-scaled speed cap ``(a, b, c)`` in m/s: ``a`` fore-aft (body ``+x``), ``b``
    lateral (body ``+-y``), ``c`` backward (body ``-x``). See
    ``crab_hex_omni_command_math.ellipse_cap``. A sampling cap, not a plant capability
    claim (measured achieved speeds are ~0.19-0.22 m/s)."""

    facing_delta: tuple[float, float] = (0.2, 0.6)
    """``(lo, hi)`` band on the |facing change| a turning slot commands, radians, RELATIVE
    to the yaw at the resample instant, sign random. Per-phase ramp 0.2:0.6 (1c) ->
    0.2:1.2 (2a) -> 0.2:2.4 (2b+)."""

    travel_dir_range: tuple[float, float] = (-math.pi, math.pi)
    """``(lo, hi)`` WORLD band the travel direction is drawn from when ``dir_band`` is
    None."""

    ang_vel_max: float = 0.3
    """Saturation of the yaw-rate command, rad/s -- replaces the stock hard-coded +-1. The
    plant's kinematic turn rate is ~0.15-0.2 rad/s, so +-1 demanded 5x what it can do and
    left ``track_ang_vel_z_exp`` with no gradient. Must exceed
    ``clips.ang_vel_clip``."""

    dir_band: tuple[float, float] | None = None
    """None -> world-uniform travel direction over ``travel_dir_range``. ``(lo, hi)`` ->
    body-relative at resample time (``theta = yaw + U(lo, hi)``), then held as a world
    direction. Phase-1 stages: 1a ``(0, 0)`` = body ``+x``, 1b ``(pi/2, pi/2)`` = body
    ``+y``. Mutually exclusive with ``cmd_pool``."""

    cmd_pool: str | None = None
    """None (the default and EVERY shipped preset) -> the legacy sampler above, bitwise.

    Otherwise the canonical pool STRING, stored verbatim so the config dump says exactly what
    the launcher asked for (``KRABBY_OMNI_CMD_POOL``, grammar
    ``name:weight[:ang_scale[:speed_scale]],...``; see
    ``crab_hex_omni_command_math.parse_cmd_pool`` and :data:`POOL_TYPES` for the names). A
    pool replaces BOTH ``slot_probs`` and ``dir_band`` as the thing that is drawn: every type
    projects onto one of the four slots, ``dir_band`` is refused outright, and ``slot_probs``
    must equal the pool's projection to within :data:`POOL_SLOT_MATCH_TOL` -- it stays as the
    dump path and the slot-level truth every metric, reward and eval hold downstream reads.

    A STRING rather than a parsed object because it is a config field: it has to survive a
    YAML dump and an ``env.yaml`` audit unchanged, and the campaign driver composes it from
    transforms and records the rendered result as the run's truth."""

    clips: ParkourCommandCfg.Clips = ParkourCommandCfg.Clips(lin_vel_clip=0.2, ang_vel_clip=0.2)
    """Re-declared with a DEFAULT (the parent's is MISSING) so the term's contract holds
    for any construction: ``lin_vel_clip`` is the stock 0.2 (it sets the walk floor), and
    ``ang_vel_clip`` is 0.2 rather than the forward task's 0.4 -- at stiffness 1.5 that is
    a facing dead band of 0.133 rad instead of 0.267 rad, and it sits below
    ``ang_vel_max`` so a saturated turn command survives it."""


# ======================================================================= the body-frame command
#
# Robot-frame velocity commands campaign (``experiments/2026-10-15_body_cmd``, PLAN D1; step-1 spec
# §2). DEFAULT OFF: the class below and its cfg exist only so ``KRABBY_OMNI_COMMAND_FRAME=body``
# (``omni_knobs._apply_command_frame``) can SWAP ``commands.base_velocity`` for a
# :class:`CrabHexOmniBodyCommandCfg` carrying every field of the world cfg plus the envelope. No
# shipped cfg names them, so a world run's config dump, RNG stream and every value are untouched.


def _validate_body_cfg(cfg: CrabHexOmniBodyCommandCfg, env=None):
    """The body cfg's own refusals (the world ones run too, in the parent's ``__init__``).

    Returns the validated :class:`BodyEnvelope`. Every refusal is a silent failure otherwise: a
    missing envelope would sample nothing sensible, a pool type with a scaled TURN demand has no
    meaning when the yaw rate is drawn from the envelope (the world relay's ``ang_scale`` would be
    silently ignored), the relay hysteresis drives a relay this mode does not run, and an RSI event
    stages a WORLD command (``rsi_cmd_staged`` = theta, v, psi) the body sampler cannot honour.
    """
    fields = (cfg.body_w_min, cfg.body_ceiling, cfg.body_w_max, cfg.body_cap)
    if any(f is None for f in fields):
        raise ValueError(
            "CrabHexOmniBodyCommandCfg needs body_w_min / body_ceiling / body_w_max / body_cap "
            "(KRABBY_OMNI_BODY_ENVELOPE + KRABBY_OMNI_BODY_CAP); got "
            f"{fields}"
        )
    envelope = check_body_envelope(BodyEnvelope(*fields))
    floor = omni_walk_floor(cfg)
    if body_v_turn_max(envelope) < floor:
        raise ValueError(
            f"body envelope {envelope} admits no moving turn: body_v_turn_max = "
            f"{body_v_turn_max(envelope)} m/s is below the walk floor {floor} m/s (the coupled cap "
            "drops under w_min before the slowest legal walk) -- PLAN D3's STOP condition"
        )
    if getattr(cfg, "relay_hyst", None) is not None:
        raise ValueError(
            f"relay_hyst {cfg.relay_hyst} (KRABBY_OMNI_RELAY_HYST) is set on a body-frame command: "
            "there is no facing relay in body mode"
        )
    if cfg.cmd_pool is not None:
        pool = parse_cmd_pool(cfg.cmd_pool)
        scaled = [(n, s) for n, s in zip(pool.names, pool.ang_scale) if float(s) != 1.0]
        if scaled:
            raise ValueError(
                f"cmd_pool {cfg.cmd_pool!r} gives {scaled} a non-unit ang_scale: in body mode the "
                "yaw rate is drawn from the envelope, so a per-type turn scale would be silently "
                "ignored"
            )
    events = getattr(getattr(env, "cfg", None), "events", None)
    if getattr(events, "rsi_reference_reset", None) is not None:
        raise ValueError(
            "events.rsi_reference_reset is installed in a body-frame env: the RSI event stages a "
            "WORLD command (theta, v, psi) the body sampler cannot honour (set KRABBY_RSI_FRAC=0)"
        )
    return envelope


class CrabHexOmniBodyCommand(CrabHexOmniCommand):
    """The omni command in the ROBOT frame: a held ``(vx, vy, wz)`` per slot (PLAN D1).

    Per resample, the SAME uniforms in the SAME order as the world sampler of the same branch
    (pool / pool-less) -- categorical, direction, speed, then the two "facing" uniforms, which feed
    :func:`~crab_hex_omni_command_math.sample_yaw_rate` instead of a facing change -- so a body run
    and a world run consume one RNG stream (pinned by the unit tests). Per row:

    * ``phi = wrap(theta - yaw)`` -- the body bearing the unchanged direction draw lands on, HELD;
    * the speed exactly as the world sampler draws it at ``phi`` (the ellipse cap x the type's
      ``speed_scale``, above the walk floor), 0 on non-moving rows;
    * turning rows: a pure turn draws ``|wz|`` in ``[w_min, w_max]``; a moving turn in
      ``[w_min, min(w_max, body_cap_wz(v))]``, its speed first reduced to ``body_v_turn_max`` where
      the coupled cap would otherwise sit below ``w_min``; the sign from the second uniform;
      non-turning rows ``wz = 0``;
    * stand rows are EXACTLY ``(0, 0, 0)``; ``slot_id`` comes from the type table as today and equals
      ``slot_from_body_cmd`` of the drawn command by construction.

    Held state ``body_phi`` / ``body_speed`` / ``body_wz`` (``(N,)``); ``vel_command_b`` is written at
    the resample (or by :meth:`schedule_hook_body`) and then NEVER re-projected: no re-cap, no relay.
    Every step the legacy world fields are refreshed so their readers see defined values --
    ``heading_target = yaw`` (zero facing error, by construction), ``travel_dir_w = yaw + body_phi``
    (the body bearing laid onto the current yaw), ``speed_cmd = body_speed``, ``phi_b = body_phi``,
    ``recap_mask = False``. The metrics keep their names and meanings (``facing_err_abs`` reads 0;
    ``yaw_progress`` keeps its ``cmd_wz != 0`` gate). No new metric, no new cfg field on the world
    class.

    The RSI staging channel is refused (the cfg validator refuses an installed RSI event, and a row
    staged anyway raises at the resample that would consume it).
    """

    cfg: CrabHexOmniBodyCommandCfg

    def __init__(self, cfg: CrabHexOmniBodyCommandCfg, env: ParkourManagerBasedEnv):
        envelope = _validate_body_cfg(cfg, env)
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.body_envelope = envelope
        self.body_v_turn_max = body_v_turn_max(envelope)
        self.body_phi = torch.zeros(n, device=dev)
        self.body_speed = torch.zeros(n, device=dev)
        self.body_wz = torch.zeros(n, device=dev)
        # The cap tables, built ONCE: the resample path evaluates the coupled cap per row and must
        # not copy two small host lists to the device on every call.
        self._cap_v, self._cap_w = body_cap_tables(envelope, device=dev, dtype=self.body_phi.dtype)

    def __str__(self) -> str:
        env = self.body_envelope
        msg = "CrabHexOmniBodyCommand (robot-frame vx, vy, wz):\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}\n"
        msg += f"\tSlot probabilities {SLOT_NAMES}: {tuple(self.cfg.slot_probs)}\n"
        msg += f"\tSpeed ellipse (fore-aft, lateral, backward): {tuple(self.cfg.speed_ellipse)}\n"
        msg += f"\tWalk floor: {omni_walk_floor(self.cfg)} m/s\n"
        msg += f"\tYaw rate: w_min {env.w_min} / ceiling {env.ceiling} / w_max {env.w_max} rad/s\n"
        msg += f"\tMove+turn cap (v, wz): {env.cap}; v_turn_max {self.body_v_turn_max} m/s"
        if self._pool is not None:
            msg += f"\n\tCommand pool: {self.cfg.cmd_pool}"
        return msg

    # ------------------------------------------------------------------ the sampler
    def _resample_command(self, env_ids: Sequence[int]):
        """Draw a held body command for ``env_ids`` (see the class docstring).

        The world sampler's two branches, uniform for uniform: the four draw calls below are the
        world ones with ``sample_facing``'s two internal ``torch.rand`` calls written out (same
        shape, device and dtype, same order), because the body mode consumes those two uniforms
        itself rather than as a facing target.
        """
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        env_ = self.body_envelope
        a, b, c = (float(x) for x in cfg.speed_ellipse)
        floor = omni_walk_floor(cfg)
        yaw = self.robot.data.heading_w[env_ids]

        if cfg.cmd_pool is None:
            slot = sample_slots(None, cfg.slot_probs, n, device=self.device)
            theta = sample_travel_dir(yaw, cfg.dir_band, cfg.travel_dir_range)
            u_speed = torch.rand(n, device=self.device)
            u_mag = torch.rand(tuple(yaw.shape), device=yaw.device, dtype=yaw.dtype)
            u_sign = torch.rand(tuple(yaw.shape), device=yaw.device, dtype=yaw.dtype)
            phi = wrap_to_pi(theta - yaw)
            cap0 = ellipse_cap(phi, a, b, c)
            type_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        else:
            tables = self._pool_tables
            type_id = sample_slots(None, self._pool.weights, n, device=self.device)
            slot = tables.slot[type_id]
            theta = sample_pool_travel_dir(
                yaw, type_id, tables.lo, tables.hi, tables.world, cfg.travel_dir_range
            )
            u_speed = torch.rand(n, device=self.device)
            u_mag = torch.rand(tuple(yaw.shape), device=yaw.device, dtype=yaw.dtype)
            u_sign = torch.rand(tuple(yaw.shape), device=yaw.device, dtype=yaw.dtype)
            phi = wrap_to_pi(theta - yaw)
            cap0 = ellipse_cap(phi, a, b, c) * tables.speed_scale[type_id]

        moving = slot_is_moving(slot)
        turning = slot_is_turning(slot)
        speed = torch.where(moving, floor + u_speed * (cap0 - floor), torch.zeros_like(u_speed))
        # The coupled move+turn cap: a moving turn whose speed puts the cap under w_min walks at
        # body_v_turn_max instead (always below the drawn speed there, so still inside the ellipse,
        # and >= the walk floor -- the cfg validator refused any envelope where it is not).
        w_min, w_max = float(env_.w_min), float(env_.w_max)
        reduce = moving & turning & (body_cap_wz_from_tables(speed, self._cap_v, self._cap_w) < w_min)
        if math.isfinite(self.body_v_turn_max):
            speed = torch.where(reduce, torch.full_like(speed, self.body_v_turn_max), speed)
        cap_w = body_cap_wz_from_tables(speed, self._cap_v, self._cap_w)
        # ``max(.., w_min)``: at exactly body_v_turn_max the interpolated cap is w_min up to one
        # rounding -- the band must never invert.
        hi = torch.where(
            moving,
            torch.clamp(torch.minimum(cap_w, torch.full_like(cap_w, w_max)), min=w_min),
            torch.full_like(cap_w, w_max),
        )
        wz = sample_yaw_rate(u_mag, u_sign, w_min, hi)
        wz = torch.where(turning, wz, torch.zeros_like(wz))

        # RSI staging is refused in body mode; a staged row here is a mis-wired env (one sync per
        # resample call, body mode only -- the world path never reaches this line).
        staged = self.rsi_cmd_staged[env_ids]
        if bool(torch.isfinite(staged).all(dim=1).any()):
            raise RuntimeError(
                "CrabHexOmniBodyCommand: an RSI-staged world command reached the body sampler; "
                "RSI staging is refused in body mode (KRABBY_RSI_FRAC must be 0)"
            )

        self.body_phi[env_ids] = phi
        self.body_speed[env_ids] = speed
        self.body_wz[env_ids] = wz
        self.slot_id[env_ids] = slot
        self.type_id[env_ids] = type_id
        if self._pool is not None:
            self._count_pool_draws(env_ids, type_id, torch.zeros_like(moving))
        self._write_command(env_ids)

    def _write_command(self, env_ids, apply_hold: bool = True):
        """The held body command -> ``vel_command_b`` (and the legacy world fields) for ``env_ids``.

        Non-moving rows are written as exact zeros (``0 * cos(phi)`` would be ``-0.0`` for a bearing
        behind the robot). ``apply_hold`` is accepted for the parent's signature and ignored: a body
        stand row is already ``(0, 0, 0)``, which is all ``KRABBY_OMNI_STAND_HOLD`` would write.
        """
        del apply_hold
        speed = self.body_speed[env_ids]
        phi = self.body_phi[env_ids]
        zero = torch.zeros_like(speed)
        moving = speed != 0.0
        self.vel_command_b[env_ids, 0] = torch.where(moving, speed * torch.cos(phi), zero)
        self.vel_command_b[env_ids, 1] = torch.where(moving, speed * torch.sin(phi), zero)
        self.vel_command_b[env_ids, 2] = self.body_wz[env_ids]
        self._write_legacy(env_ids)

    def _write_legacy(self, env_ids) -> None:
        """The legacy world fields, refreshed from the held body command and the CURRENT yaw."""
        yaw = self.robot.data.heading_w[env_ids]
        self.heading_target[env_ids] = yaw
        self.travel_dir_w[env_ids] = yaw + self.body_phi[env_ids]
        self.speed_cmd[env_ids] = self.body_speed[env_ids]
        self.phi_b[env_ids] = self.body_phi[env_ids]
        self.recap_mask[env_ids] = False

    def _update_command(self):
        """Every step: NO re-projection and no relay -- ``vel_command_b`` stays held; only the legacy
        world fields follow the robot's yaw."""
        self._write_legacy(slice(None))

    def _update_metrics(self):
        """The parent's metrics, after refreshing the legacy fields to THIS step's yaw.

        The command manager runs ``_update_metrics`` before ``_update_command``, i.e. after physics
        moved the yaw; without the refresh ``facing_err_abs`` would read the step's own rotation
        ``|wz| dt`` instead of the 0 a body command's facing error is by definition.
        """
        self._write_legacy(slice(None))
        super()._update_metrics()

    # ------------------------------------------------------------------ identity
    def command_signature(self) -> torch.Tensor:
        """``(N, 3)`` = ``(body_phi, body_speed, body_wz)``: changes exactly when a new command is held."""
        return torch.stack((self.body_phi, self.body_speed, self.body_wz), dim=-1)

    def intended_bearing_b(self) -> torch.Tensor:
        """``(N,)`` the commanded body bearing -- the held ``body_phi`` itself."""
        return self.body_phi

    # ------------------------------------------------------------------ eval harness
    def schedule_hook(self, row) -> None:
        """REFUSED in body mode: a world hold row (speed, dir, face) on a body term is a route
        mismatch -- the harness must route body rows through :meth:`schedule_hook_body`, or a world
        row through a NAMED adapter (``world_rate``)."""
        raise RuntimeError(
            "schedule_hook (a WORLD row: speed/dir/face) called on CrabHexOmniBodyCommand: route "
            "mismatch -- use schedule_hook_body(vx, vy, wz), or a named adapter"
        )

    def schedule_hook_body(self, vx, vy=None, wz=None) -> None:
        """Pin every env to one held body command (eval harness, PLAN D5 / spec §2).

        ``vx, vy, wz`` as numbers, or ONE row argument (a mapping or an object with ``vx`` / ``vy`` /
        ``wz``). Validated -- finite AND inside the envelope (``body_cmd_violations == ()``), else
        ``ValueError`` naming every violation; the hook never shapes a row (that is the adapter's
        job, :func:`~crab_hex_omni_command_math.shape_body_command`). Then: the held state, the
        command (``vel_command_b`` = the row's three components VERBATIM, so obs cols 8/9/5 read the
        row exactly), ``slot_id = slot_from_body_cmd``, ``type_id = -1`` and the legacy fields.
        """
        if vy is None and wz is None:
            row = vx
            vx, vy, wz = (_row_field(row, k) for k in ("vx", "vy", "wz"))
        vx, vy, wz = float(vx), float(vy), float(wz)
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(wz)):
            raise ValueError(f"schedule_hook_body: vx/vy/wz must be finite, got ({vx}, {vy}, {wz})")
        bad = body_cmd_violations(vx, vy, wz, self.body_envelope, tuple(self.cfg.speed_ellipse),
                                  omni_walk_floor(self.cfg))
        if bad:
            raise ValueError(f"schedule_hook_body: ({vx}, {vy}, {wz}) is outside the body envelope: {bad}")
        speed = math.hypot(vx, vy)
        self.body_phi[:] = math.atan2(vy, vx) if speed > 0.0 else 0.0
        self.body_speed[:] = speed
        self.body_wz[:] = wz
        slot = slot_from_body_cmd(torch.tensor([vx]), torch.tensor([vy]), torch.tensor([wz]))
        self.slot_id[:] = int(slot[0])
        self.type_id[:] = -1
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = vy
        self.vel_command_b[:, 2] = wz
        self._write_legacy(slice(None))


@configclass
class CrabHexOmniBodyCommandCfg(CrabHexOmniCommandCfg):
    """Cfg for :class:`CrabHexOmniBodyCommand` -- the world cfg's fields plus the measured envelope.

    NEVER built by a shipped cfg: ``omni_knobs._apply_command_frame`` constructs one only under
    ``KRABBY_OMNI_COMMAND_FRAME=body``, carrying EVERY field of the realised world cfg unchanged
    (so the slot mix, the pool, the ellipse, the resampling time and every motion-knob instance
    attribute carry over) plus the four fields below -- which exist ONLY on this subclass, so a
    world config dump cannot grow them. The world relay fields (``facing_delta``,
    ``ang_vel_max``, ``clips.ang_vel_clip``, ``heading_control_stiffness``) are carried and inert:
    the yaw rate is drawn from the envelope, and nothing re-projects or relays the command.

    The envelope is the knob grammar's (``crab_hex_omni_command_math.parse_body_envelope``) and is
    re-validated by the term at construction (``_validate_body_cfg``).
    """

    class_type: type = CrabHexOmniBodyCommand

    body_w_min: float | None = None
    """Smallest non-zero yaw-rate command, rad/s (``KRABBY_OMNI_BODY_ENVELOPE``'s first field)."""

    body_ceiling: float | None = None
    """The plant's sustained pure-turn rate, rad/s (recorded with the envelope; gates read at it)."""

    body_w_max: float | None = None
    """Largest yaw-rate command, rad/s; a pure turn draws up to it."""

    body_cap: tuple | None = None
    """The coupled move+turn cap ``((v, wz), ...)`` (``KRABBY_OMNI_BODY_CAP``), see
    ``crab_hex_omni_command_math.body_cap_wz``."""


@configclass
class CrabHexOmniCommandsCfg:
    """The forward task's ``CommandsCfg`` literals, on the omni command class.

    Standalone rather than a subclass of the Go2 ``CommandsCfg`` (the sideways precedent):
    the only member is ``base_velocity`` and it has to be rebuilt anyway -- different
    class, and the KNOB-TIMING pillar (the forward class bodies are evaluated before the
    omni activation exports its knobs, so an inherited cfg default would be a deep copy of
    the UN-expanded forward object).

    Copied verbatim from ``extreme_parkour_task/config/go2/parkour_mdp_cfg.py::CommandsCfg``
    (the cfg the forward env cfg instantiates) with ONE deliberate departure:
    ``clips.ang_vel_clip`` is 0.2, not the Go2 0.4 -- the tighter facing dead band is part
    of the turning fix and is listed in the expected omni-vs-forward delta (Appendix A).
    ``heading_control_stiffness`` stays the Go2 0.8 here; the per-phase knob block raises it
    to 1.5 exactly as it does for the forward task.
    """

    base_velocity = CrabHexOmniCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        heading_control_stiffness=0.8,
        ranges=ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8),
            heading=(-1.6, 1.6),
        ),
        clips=ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.2,
        ),
    )
