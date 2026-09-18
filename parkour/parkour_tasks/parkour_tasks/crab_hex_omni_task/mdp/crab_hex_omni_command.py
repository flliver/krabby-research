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
    SLOT_MOVE,
    SLOT_MOVE_TURN,
    SLOT_NAMES,
    SLOT_STAND,
    SLOT_TURN,
    body_command_details,
    ellipse_cap,
    heading_rate_command,
    sample_facing,
    sample_slots,
    sample_travel_dir,
    slot_is_moving,
    slot_is_turning,
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
    learn).

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
        # ``stand_frac_actual`` already exists (base class); the rest are new.
        for key in ("move_frac_actual", "turn_frac_actual", "move_turn_frac_actual",
                    "facing_err_abs", "recap_frac", "yaw_progress"):
            self.metrics[key] = torch.zeros(n, device=dev)

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
        return msg

    # ------------------------------------------------------------------ implementation

    def _resample_command(self, env_ids: Sequence[int]):
        """Draw a new slot (or consume an RSI-staged command) for ``env_ids``.

        Draw order is FIXED -- slot, direction, speed, facing -- and every draw is taken
        for every row regardless of slot or staging, so the RNG stream a run consumes does
        not depend on the slot mix, on ``dir_band`` or on which rows the RSI event happened
        to stage. Only the *use* of a draw is conditional.
        """
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        a, b, c = (float(x) for x in cfg.speed_ellipse)
        floor = omni_walk_floor(cfg)
        yaw = self.robot.data.heading_w[env_ids]

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

        # RSI-staged rows override all three and become a MOVE slot: the staged speed is
        # already clamped into [floor, cap] by the reset event and the staged psi is the
        # spawn yaw, so the frame starts on-command with zero facing error.
        staged = self.rsi_cmd_staged[env_ids]
        has_staged = torch.isfinite(staged).all(dim=1)
        theta = torch.where(has_staged, staged[:, 0], theta)
        speed = torch.where(has_staged, staged[:, 1], speed)
        psi = torch.where(has_staged, staged[:, 2], psi)
        slot = torch.where(has_staged, torch.full_like(slot, SLOT_MOVE), slot)

        self.travel_dir_w[env_ids] = theta
        self.speed_cmd[env_ids] = speed
        self.heading_target[env_ids] = psi
        self.slot_id[env_ids] = slot
        # Consumed exactly once: whatever was staged is gone even if this row was not
        # staged (writing NaN over NaN), so a stale row can never be replayed.
        self.rsi_cmd_staged[env_ids] = float("nan")

        # Refresh the published command for these rows now. ``compute()`` calls
        # ``_update_command`` right after this, but ``reset()`` resamples WITHOUT it, and
        # the observation manager may run before the next ``compute``.
        self._write_command(env_ids)

    def _update_command(self):
        """Re-project the held world command into the CURRENT body frame (every step)."""
        self._write_command(slice(None))

    def _write_command(self, env_ids):
        """World command -> ``vel_command_b`` for ``env_ids`` (a tensor/list or a slice)."""
        cfg = self.cfg
        a, b, c = (float(x) for x in cfg.speed_ellipse)
        yaw = self.robot.data.heading_w[env_ids]
        speed = self.speed_cmd[env_ids]
        cmd_xy, phi, cap = body_command_details(self.travel_dir_w[env_ids], speed, yaw, a, b, c)
        self.vel_command_b[env_ids, :2] = cmd_xy
        self.vel_command_b[env_ids, 2] = heading_rate_command(
            self.heading_target[env_ids],
            yaw,
            float(cfg.heading_control_stiffness),
            float(cfg.ang_vel_max),
            float(cfg.clips.ang_vel_clip),
        )
        self.phi_b[env_ids] = phi
        self.recap_mask[env_ids] = speed > cap

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
        """
        speed = float(_row_field(row, "speed"))
        direction = float(_row_field(row, "dir"))
        face = float(_row_field(row, "face"))
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError(f"schedule_hook: speed must be finite and >= 0, got {speed}")
        if not (math.isfinite(direction) and math.isfinite(face)):
            raise ValueError(f"schedule_hook: dir/face must be finite, got ({direction}, {face})")
        self.travel_dir_w[:] = direction
        self.speed_cmd[:] = speed
        self.heading_target[:] = face
        self._write_command(slice(None))
        moving = self.speed_cmd > 0.0
        turning = self.vel_command_b[:, 2] != 0.0
        self.slot_id[:] = moving.long() + 2 * turning.long()


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
    ``+y``."""

    clips: ParkourCommandCfg.Clips = ParkourCommandCfg.Clips(lin_vel_clip=0.2, ang_vel_clip=0.2)
    """Re-declared with a DEFAULT (the parent's is MISSING) so the term's contract holds
    for any construction: ``lin_vel_clip`` is the stock 0.2 (it sets the walk floor), and
    ``ang_vel_clip`` is 0.2 rather than the forward task's 0.4 -- at stiffness 1.5 that is
    a facing dead band of 0.133 rad instead of 0.267 rad, and it sits below
    ``ang_vel_max`` so a saturated turn command survives it."""


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
