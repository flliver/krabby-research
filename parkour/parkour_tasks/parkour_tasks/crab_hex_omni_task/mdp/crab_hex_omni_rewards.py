# SPDX-License-Identifier: BSD-3-Clause
"""Omnidirectional reward twins, their cfg, and the goal-free termination (plan D4/A6).

FOUR reward FUNCTIONS are swapped under the SAME term names, one term's ``std`` is retuned,
and ONE term is ADDED (:func:`reward_yaw_progress`, the 36th name -- see "THE ADDED TERM"
below). Everything else -- the forward 35 names, every weight, every ramp, every
``KRABBY_*_W`` knob in the parent's ``__post_init__`` -- is inherited untouched, because the
whole telemetry/driver/phase-out machinery keys on the NAMES (``Episode_Reward/<name>``,
``KRABBY_PHASEOUT``'s term list, the driver's ``exposure_from_log``). This is the sideways
task's func-swap precedent (``CrabHexSideFlatWalkRewardsCfg``), four terms wide.

WHY ONLY FOUR. The omni command is a 2-D body-frame velocity plus a yaw rate, so the stock
``track_lin_vel_xy_exp``, ``penalty_tracking_error_l1``,
``reward_forward_progress_along_command`` and ``RewardStrideLength`` -- all of which already
read ``cmd[:, :2]`` as a vector -- are correct as they stand (this is the plan's "the omni
cfg carries no ``travel_yaw``"). What breaks is every term that reads the command as ONE
number:

``penalty_lin_vel_y`` -> :func:`penalty_cross_command_vel_l2`
    "don't drift sideways" becomes "don't drift ACROSS the commanded direction". For a pure
    +-x command the two are the same number, bitwise (pinned by the unit test).
``reward_tracking_goal_vel`` -> :func:`reward_tracking_command_vel_on_obstacle`
    The forward term pays progress toward the next COURSE GOAL, at ``cmd[:, 0]``'s speed.
    The ring course never advances a goal (there is no route), so the target direction is
    the commanded one and the denominator is ``||cmd_xy||``.
``reward_tracking_yaw`` -> :func:`reward_facing_exp`
    Facing is now an independent command, not a by-product of following a route: the term
    pays ``exp(-|wrap(psi - yaw)|)`` on EVERY tile with no command gate, so a standing robot
    is still paid to face its target (the forward twin was masked to parkour tiles and gated
    on a forward command, which would make the facing half of the task unpayable on the
    flat disc where most of phase 1 happens).
``reward_obstacle_clearance`` -> :func:`reward_obstacle_clearance_omni`
    Same five-factor gate; "goal progress" is the command-velocity term above and "forward
    speed" is the body speed ALONG the commanded direction.

``track_ang_vel_z_exp`` keeps its function and weight and gets ``std = 0.15`` (was
``sqrt(0.25)`` = 0.5). The adversarial review behind this plan measured |wz| ~ 0.003 rad/s
against 0.6-1.0 rad/s commanded on every head of record: at std 0.5 the exponential is flat
(0.99 -> 0.94) across the entire 0-0.3 rad/s band this plant can actually turn in, i.e. no
gradient where the behaviour lives. 0.15 puts the well's edge at the yaw-rate cap.

**REVISION 4 (2026-09-18) REVERSES THAT DOSE FROM THE PRESET SIDE.** The class-body literal
stays 0.15 (the knob-timing pillar: a cfg class body carries its declared value and knobs move
it), but every omni preset now sets ``KRABBY_OMNI_ANG_VEL_STD=0.5`` -- back to the forward
kernel -- so no shipped omni run trains at 0.15. What 0.15 costs, measured on the seed-3 1c
head: the tight well also applies on the ~88 % of steps that command NO turn, where ordinary
gait yaw wobble of 0.10-0.15 rad/s is taxed 0.36-0.63 per second, more than the entire
translation income a walking policy can earn -- one of the four terms that made standing still
the optimum. The turning gradient this docstring was worried about is supplied instead by
``reward_yaw_progress`` below (weight 1.0, signed, dense), which did not exist when 0.15 was
chosen. See ``README.md`` "Revision 4 -- the reward economy".

THE ADDED TERM (:func:`reward_yaw_progress`, the 36th name; 2026-09-17). The plan's
500-iteration yaw-capability smoke (``KRABBY_OMNI_SLOTS=0:0:1:0``, flat, from scratch; run
``crab_hex_omni_flat_walk/2026-09-16_20-35-23``) FAILED its gate: the policy learned to stand
(flat failure 0.50 -> 0.12) and to step in place under the clock (tripod 0.47 on the turn
holds) but not to rotate. On the harness's ``turn_l90`` hold the yaw-rate command sat
saturated at +0.30 rad/s for 8 s while the median net body rotation was -0.025 rad/s -- the
WRONG SIGN -- inside a mean |wz| of 0.28 rad/s of pure gait wobble; training
``facing_err_abs`` never trended down (0.27-0.43 throughout). Both facing terms are
wobble-blind at that amplitude: ``track_ang_vel_z_exp`` (std 0.15) and ``reward_facing_exp``
(weight 0.2) are even functions of a signed quantity evaluated through +-0.28 rad/s of
noise, so their gradient through one step of gait wobble swamps the drift they are meant to
price. Nothing in the 35 rewarded ROTATING the way
``reward_forward_progress_along_command`` rewards TRANSLATING. This term is exactly that
rotational twin: ``wz_b * sign(cmd_wz)``, clipped, gated on a non-zero yaw command.

FRAME CONVENTIONS (the silent-failure surface):

* ``cmd[:, :2]`` is a BODY-frame velocity command; its direction ``dir_b`` is a unit vector
  in the body frame. ``||cmd_xy||`` is the commanded speed.
* the unit direction divides by ``clamp_min(speed, _DIR_EPS)``, NOT by ``speed + eps``: for
  an axis-aligned command the division is then exact (``vx / |vx| == 1.0``), which is what
  makes the ``penalty_lin_vel_y`` identity bitwise rather than approximate.
* the facing error is measured against ``asset.data.heading_w`` -- the same yaw
  ``CrabHexOmniCommand``'s P-control and the omni observation term use, so reward,
  observation and command all agree. (The forward ``reward_tracking_yaw`` decomposes the
  quaternion itself; the two differ under roll/pitch, and the one that matters here is the
  one the command is steering.)
* the world-frame command direction is ``R_z(heading_w) @ dir_b``, which reconstructs
  exactly the world travel direction the command term sampled.
* THE YAW SIGN CHAIN, end to end (pinned by ``test_crab_hex_omni_rewards.py`` against the
  real ``heading_rate_command``): a positive facing error
  ``heading_err = wrap(heading_target - heading_w) > 0`` means "turn counter-clockwise, yaw
  must increase"; the command term's P-control makes ``cmd_wz = clip(heading_err * k,
  +-ang_vel_max)`` (dead-banded), so ``cmd_wz`` carries the SAME sign as ``heading_err``; a
  positive body yaw rate ``root_ang_vel_b[:, 2]`` turns the body counter-clockwise, i.e.
  increases ``heading_w`` and shrinks the error. So ``wz_b * sign(cmd_wz) > 0`` IS progress,
  in both directions. Get any one link wrong and the term trains the policy to spin away
  from its facing target while every other number in the log looks healthy.
* ``root_ang_vel_b[:, 2]`` (the BODY z rate), not a world yaw rate: it is what the stock
  ``track_ang_vel_z_exp`` and the command term's own ``error_vel_yaw`` metric compare
  ``cmd[:, 2]`` against, so the reward, the tracking term and the telemetry all price one
  quantity. Near-upright (the only attitude that survives termination) the two agree.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CrabHexFlatWalkRewardsCfg,
    CrabHexFlatWalkTerminationsCfg,
)

#: Floor of the divisor that turns ``cmd[:, :2]`` into a unit direction. ``clamp_min`` (not
#: ``+ eps``) keeps ``dir_b`` EXACTLY ``(+-1, 0)`` / ``(0, +-1)`` for an axis-aligned
#: command; see the module docstring.
_DIR_EPS = 1e-6

#: The ``v_min`` of the forward ``reward_tracking_goal_vel`` divisor guard, kept verbatim.
_TRACK_V_MIN = 0.05

#: ``track_ang_vel_z_exp``'s retuned std (plan D4). NOTE (Revision 4, 2026-09-18): every shipped
#: omni preset overrides this with ``KRABBY_OMNI_ANG_VEL_STD=0.5``, the forward kernel -- see the
#: module docstring. This literal is what an omni env built with NO knobs uses; it is kept at the
#: plan's value because the class body is not where doses are set.
OMNI_ANG_VEL_STD = 0.15


def _command_dir_and_speed(cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dir_b, speed)``: the unit body-frame travel direction and ``||cmd_xy||``.

    At zero command ``dir_b`` is the zero vector (0 / eps), so every term built on it pays
    exactly 0 rather than picking an arbitrary direction.
    """
    cmd_xy = cmd[:, :2]
    speed = torch.linalg.norm(cmd_xy, dim=-1)
    dir_b = cmd_xy / speed.clamp_min(_DIR_EPS).unsqueeze(-1)
    return dir_b, speed


def _facing_error(
    env: ParkourManagerBasedRLEnv, asset: Articulation, command_name: str
) -> torch.Tensor:
    """``wrap(heading_target - heading_w)`` -- the signed facing error, radians."""
    cmd_term = env.command_manager.get_term(command_name)
    return wrap_to_pi(cmd_term.heading_target - asset.data.heading_w)


# --------------------------------------------------------------------------- reward twins
def penalty_cross_command_vel_l2(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_cmd_norm: float = 0.12,
) -> torch.Tensor:
    """Penalize the body velocity ACROSS the commanded direction, squared.

    The omni twin of ``penalty_lin_vel_y_l2``. The cross component is
    ``v_b . n_b`` with ``n_b`` the left normal of the unit command direction
    (``n_b = (-dir_y, dir_x)``), so for a pure +-x command it is exactly ``+-v_y`` and the
    squared value is bitwise the forward term's. Gated on ``||cmd_xy|| > min_cmd_norm``:
    with no travel commanded there is no "across" to speak of (and a standing or
    turning-in-place robot must not be taxed for the sway a turn produces).

    The forward term additionally required ``|cmd_y| < 0.05`` ("only travel-axis motion is
    commanded"); under an omnidirectional command that condition is vacuous -- every
    direction is commandable -- and it is the CROSS component, not the y component, that is
    unwanted.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dir_b, speed = _command_dir_and_speed(cmd)
    v_b = asset.data.root_lin_vel_b[:, :2]
    cross = v_b[:, 0] * (-dir_b[:, 1]) + v_b[:, 1] * dir_b[:, 0]
    moving_cmd = speed > min_cmd_norm
    return torch.square(cross) * moving_cmd.float()


def reward_tracking_command_vel(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """World velocity projected on the commanded direction, per unit commanded speed.

    The forward ``reward_tracking_goal_vel`` with its two course-specific inputs replaced:
    the target direction is the commanded one (the ring course has no route to progress
    along) and the denominator is ``||cmd_xy||`` instead of ``cmd[:, 0]``. Everything else
    -- the world-frame projection, the ``v_min`` divisor guard, ``minimum(proj, cmd)``, the
    NaN scrub and the ``+-10`` clamp -- is the forward function's shape, unchanged.

    The forward divisor also carried a ``sign(command_vel)`` branch so that a NEGATIVE
    command still produced a positive reward for moving backwards. A speed is non-negative,
    so that branch is identically ``+1`` here and is dropped rather than carried dead;
    backward travel is expressed as a backward DIRECTION, not a negative speed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dir_b, speed = _command_dir_and_speed(cmd)
    yaw = asset.data.heading_w
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    dir_w = torch.stack(
        (
            cos_yaw * dir_b[:, 0] - sin_yaw * dir_b[:, 1],
            sin_yaw * dir_b[:, 0] + cos_yaw * dir_b[:, 1],
        ),
        dim=-1,
    )
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(dir_w * cur_vel, dim=-1)
    denom = torch.where(speed < _TRACK_V_MIN, torch.full_like(speed, _TRACK_V_MIN), speed)
    rew_move = torch.minimum(proj_vel, speed) / denom
    rew_move = torch.nan_to_num(rew_move, nan=0.0, posinf=0.0, neginf=0.0)
    rew_move = torch.clamp(rew_move, -10.0, 10.0)
    return rew_move


def reward_tracking_command_vel_on_obstacle(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """:func:`reward_tracking_command_vel`, zero on ``parkour_flat`` tiles.

    The omni twin of ``reward_tracking_goal_vel_on_parkour``: the same mask, for the same
    reason (on flat ground the stock ``track_lin_vel_xy_exp`` already prices the command;
    this term is the obstacle-tile bonus that the unmasked version's 2b1 collapse taught us
    to keep masked).
    """
    rew = reward_tracking_command_vel(env, asset_cfg, command_name)
    return rew * (1.0 - mdp_rewards._parkour_flat_mask(env, parkour_name))


def reward_facing_exp(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    parkour_name: str | None = None,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """``exp(-|wrap(heading_target - heading_w)|)`` -- on every tile, at every command.

    The omni twin of ``reward_tracking_yaw_on_parkour``. No tile mask and no command gate:
    facing is an independent command in this task (the "turn only" slot commands nothing
    else), and phase 1 is mostly flat, so a masked or movement-gated facing term would be
    unpayable exactly where facing has to be learned.

    ``parkour_name`` is accepted and IGNORED -- it is in the forward term's params and
    keeping the key makes the omni-vs-forward config delta a pure func swap.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.abs(_facing_error(env, asset, command_name)))


def reward_obstacle_clearance_omni(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    min_goal_progress: float = 0.15,
    min_forward_speed: float = 0.25,
    min_forward_speed_cmd: float = 0.12,
    max_tilt_gravity_xy_sq: float = 0.02,
) -> torch.Tensor:
    """Lift-and-cross bonus on obstacle tiles, read along the commanded direction.

    Factor for factor the forward ``reward_obstacle_clearance``; the three command-shaped
    inputs are the omni ones:

    * goal progress -> :func:`reward_tracking_command_vel` (unmasked, exactly as the forward
      term uses the unmasked ``reward_tracking_goal_vel``; the ``on_parkour`` factor below
      already carries the mask);
    * the command gate -> ``||cmd_xy|| > min_forward_speed_cmd`` (the forward
      ``cmd[:, 0] > ...`` is <= 0 for every lateral and backward command);
    * the achieved speed -> ``v_b . dir_b``, the body speed along the commanded direction
      (the forward ``_vel_travel_xy(...)[:, 0]``, whose travel frame is the identity here).
    """
    on_parkour = 1.0 - mdp_rewards._parkour_flat_mask(env, parkour_name)
    stumble = mdp_rewards.reward_feet_stumble(env, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_prog = torch.clamp(
        reward_tracking_command_vel(env, asset_cfg, command_name),
        min=0.0,
    )
    cmd = env.command_manager.get_command(command_name)
    dir_b, speed = _command_dir_and_speed(cmd)
    travel_cmd = speed > min_forward_speed_cmd
    vx = torch.sum(asset.data.root_lin_vel_b[:, :2] * dir_b, dim=-1)
    tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    progressing = (goal_prog >= min_goal_progress).float()
    moving = ((vx >= min_forward_speed) & travel_cmd).float()
    landed = (tilt_sq <= max_tilt_gravity_xy_sq).float()
    no_stumble = 1.0 - stumble
    return on_parkour * no_stumble * progressing * moving * landed


# ------------------------------------------------------------------- the added term (36th)
def _yaw_progress_clip(env: ParkourManagerBasedRLEnv, command_name: str) -> float:
    """The commanded yaw-rate saturation, read off the COMMAND TERM'S CFG at call time.

    ``cfg.ang_vel_max`` exists only on ``CrabHexOmniCommandCfg`` and is written per window by
    ``KRABBY_OMNI_ANG_VEL_MAX``. Reading it here (rather than copying 0.3 into the reward's
    params) keeps the cap in ONE place: retuning the command's cap -- the plan's first
    revision candidate -- can never leave the reward paying for rotation faster than the
    policy is ever asked to produce. Same pattern as ``_facing_error`` reaching for the
    command term's ``heading_target``: the term is the authority on its own command.

    Raises loudly if the command term has no such field -- a forward/sideways command cfg in
    an omni env would otherwise give this reward an arbitrary silent default.
    """
    cmd_term = env.command_manager.get_term(command_name)
    cap = getattr(getattr(cmd_term, "cfg", None), "ang_vel_max", None)
    if cap is None:
        raise RuntimeError(
            f"reward_yaw_progress: the command term {command_name!r} "
            f"({type(cmd_term).__name__}) has no cfg.ang_vel_max; it is not a "
            "CrabHexOmniCommand, so the yaw-rate cap this reward clips at is unknown. Pass "
            "an explicit `clip` or launch the omni command term."
        )
    return float(cap)


def reward_yaw_progress(
    env: ParkourManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float | None = None,
) -> torch.Tensor:
    """Dense progress toward the commanded turn: ``wz_b * sign(cmd_wz)``, rad/s.

    The ROTATIONAL TWIN of ``reward_forward_progress_along_command`` (``$LIB/envs/mdp/
    rewards.py``), factor for factor: that term projects the achieved body velocity on the
    commanded travel direction, clips the projection at a ceiling and gates on "a travel
    command is active"; this one projects the achieved body yaw rate on the commanded turn
    DIRECTION (its sign), clips at the yaw-rate cap and gates on "a turn is commanded". It
    exists because nothing in the inherited 35 paid for rotating -- see the module
    docstring's THE ADDED TERM for the smoke that proved it.

    Three deliberate differences from the linear twin, each load-bearing:

    SIGNED, not ``clamp(min=0)``. The forward term pays only non-negative progress. The
    measured failure here is a net rotation of the WRONG SIGN (-0.025 rad/s against a +0.30
    rad/s command) hidden inside 0.28 rad/s of gait wobble -- which a non-negative term
    cannot see at all: it pays the wobble's positive halves and ignores its negative ones,
    so pure symmetric wobble earns income and yields no gradient toward net rotation. Paying
    ``wz_b * sign(cmd_wz)`` signed makes the wobble's own halves cancel and leaves exactly
    the net drift, which is the quantity that has to move. The lower clip at ``-cap`` bounds
    the damage: a fall-induced spin costs at most one cap's worth per step, not an unbounded
    negative that would make termination the cheaper option.

    CLIPPED AT THE COMMANDED CAP, not at a multiple of it. ``cap`` is the command term's own
    ``ang_vel_max`` (0.3 rad/s under ``KRABBY_OMNI_ANG_VEL_MAX``; :func:`_yaw_progress_clip`),
    i.e. the largest yaw rate the policy is ever ASKED for. The reward therefore saturates at
    the achievable rate and cannot be farmed by spinning faster than commanded -- the
    rotational form of the wheelie exploit ``max_speed_scale`` 1.05 closed on the linear term.
    Over-rotation is still discouraged elsewhere: ``track_ang_vel_z_exp``'s income falls as
    |wz - cmd_wz| grows (that is what ``std`` 0.15 is for), and ``reward_facing_exp`` falls as
    soon as the robot swings past its target.

    GATED ON ``cmd_wz != 0``, not on a magnitude threshold. The command's P-control already
    zeroes the yaw-rate command inside the dead band (``clips.ang_vel_clip`` 0.2 rad/s ->
    |facing error| < 0.133 rad at stiffness 1.5), so "the command is non-zero" IS
    "|error| > dead band": there is no second threshold to tune and no window in which this
    term pushes against the dead band the command term just imposed. Every standing,
    pure-travel or already-settled row pays exactly 0.0.

    That gate is also why the presets can arm the term in EVERY window, 1a/1b included. Those
    stages draw no turning slot (``slot_probs[2] == slot_probs[3] == 0``), so no COMMANDED turn
    exists to be paid for; the only thing that can open the gate there is the robot's own yaw
    drifting more than 0.133 rad off the facing the slot held, and what the term then pays for
    is turning back -- a heading hold, in the same direction as everything else in the stage.
    (The exactly-zero claim therefore holds per row while the facing error stays inside the
    dead band, NOT unconditionally for the whole stage. Setting ``KRABBY_YAW_PROGRESS_W=0`` in
    1a/1b would make it unconditional, at the cost of the dose living in three places.)

    Args:
        env: the omni env.
        command_name: the command term to read ``cmd[:, 2]`` and ``cfg.ang_vel_max`` from.
        asset_cfg: the articulation whose ``root_ang_vel_b`` is the achieved yaw rate.
        clip: override for the symmetric clip, rad/s. ``None`` (the cfg's value) reads the
            command term's cap, which is what every training window wants.

    Returns:
        ``(N,)`` rad/s in ``[-cap, +cap]``, zero wherever no turn is commanded.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_wz = cmd[:, 2]
    cap = _yaw_progress_clip(env, command_name) if clip is None else float(clip)
    if not cap > 0.0:
        raise ValueError(f"reward_yaw_progress: clip must be positive, got {cap}")
    turning_cmd = cmd_wz != 0.0
    # sign(cmd_wz) is already 0 on the gated rows; the explicit mask documents the gate and
    # keeps the term exactly 0.0 (not -0.0) there.
    progress = asset.data.root_ang_vel_b[:, 2] * torch.sign(cmd_wz)
    progress = torch.clamp(progress, min=-cap, max=cap)
    return progress * turning_cmd.float()


# --------------------------------------------------------------------------- termination
def terminate_episode_no_goals(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """``parkour_isaaclab...terminations.terminate_episode`` without the goal cutoff.

    The forward function ends an episode when ``cur_goal_idx >= num_goals`` -- the robot
    finished the corridor. The ring course's eight sector anchors are never advanced
    (``OmniParkourEvent`` holds ``cur_goal_idx``), so that branch would be dead code at best;
    what it would actually do is fire on any future bookkeeping change and silently cut
    every episode short. Roll / pitch / height / timeout are the forward literals.
    """
    reset_buf = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:, 3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > 1.5
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > 1.5
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    height_cutoff = asset.data.root_state_w[:, 2] < -0.25
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    return reset_buf


# --------------------------------------------------------------------------- cfg classes
@configclass
class CrabHexOmniFlatWalkRewardsCfg(CrabHexFlatWalkRewardsCfg):
    """The 35 flat-walk reward terms (four functions swapped, one ``std`` retuned) + 1 added.

    36 names: the forward 35 with their attribute names, weights and knobs, plus
    ``reward_yaw_progress``. Re-declaring an inherited field keeps its position in the
    parent's field order; the added one lands after them all. Weights stay at the forward
    class-body literals on purpose (the KNOB-TIMING pillar): ``KRABBY_GOAL_VEL_W`` /
    ``KRABBY_YAW_W`` / ``KRABBY_CLEARANCE_W`` are applied by the parent's ``__post_init__``,
    which this class inherits unchanged, so the per-phase doses come from the presets and
    live in exactly one place. The ADDED term's weight follows the same rule from the other
    side: the class body carries the plan's default and ``KRABBY_YAW_PROGRESS_W`` (applied in
    ``omni_knobs.apply_omni_knobs``, which runs after the forward knob block) is the only
    thing that moves it -- the parent's ``_overrides`` table cannot know a name that does not
    exist on it.
    """

    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": OMNI_ANG_VEL_STD},
    )
    penalty_lin_vel_y = RewTerm(
        func=penalty_cross_command_vel_l2,
        weight=-3.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    reward_tracking_goal_vel = RewTerm(
        func=reward_tracking_command_vel_on_obstacle,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
        },
    )
    reward_tracking_yaw = RewTerm(
        func=reward_facing_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
        },
    )
    reward_obstacle_clearance = RewTerm(
        func=reward_obstacle_clearance_omni,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Footpad"),
            "parkour_name": "base_parkour",
            "command_name": "base_velocity",
            "min_goal_progress": 0.05,
            "min_forward_speed": 0.10,
            "min_forward_speed_cmd": 0.12,
            "max_tilt_gravity_xy_sq": 0.02,
        },
    )
    # The 36th name: the ONLY term that is added rather than swapped (2026-09-17, after the
    # yaw-capability smoke). Weight 1.0 x cap 0.3 rad/s caps its per-step income at 0.3 --
    # about half the linear progress term's ceiling (0.60 x 1.05), so translation still
    # dominates a move+turn slot. ``clip: None`` = "read the command's own ang_vel_max", so
    # KRABBY_OMNI_ANG_VEL_MAX retunes command and reward together.
    reward_yaw_progress = RewTerm(
        func=reward_yaw_progress,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "clip": None,
        },
    )


@configclass
class CrabHexOmniFlatWalkTerminationsCfg(CrabHexFlatWalkTerminationsCfg):
    """The flat-walk terminations with the goal cutoff removed from ``total_terminates``.

    ``crab_failure`` is inherited verbatim (a plain literal with no knob reads).
    """

    total_terminates = DoneTerm(
        func=terminate_episode_no_goals,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
