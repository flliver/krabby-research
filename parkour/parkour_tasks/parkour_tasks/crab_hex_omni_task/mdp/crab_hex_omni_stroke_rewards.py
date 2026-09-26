# SPDX-License-Identifier: BSD-3-Clause
"""``ManagerTermBase`` wrappers around ``crab_hex_omni_stroke_math`` (stride/posture plan, step 3).

:class:`RewardFootStroke` and :class:`RewardBodyCentring` drive the pure per-foot stroke income
and body-centring bonus from real env/sensor data -- the ``RewardStrideLength`` /
``RewardTripodSchedule`` pattern (``parkour_isaaclab/envs/mdp/rewards.py``): the math module stays
free of any ``isaaclab`` import, and these classes hold the per-env, per-foot state the math needs
between steps and reset it on ``ManagerTermBase.reset``.

Neither term is wired into any shipped rewards cfg here -- ``config/crab_hex/stroke_knobs.py``
(default OFF) is what reaches them: :data:`crab_hex_omni_task.config.crab_hex.stroke_knobs.STROKE_W_VAR`
swaps ``RewardFootStroke`` in under the existing ``reward_stride_length`` name, and
:data:`...stroke_knobs.CENTRING_W_VAR` adds ``reward_body_centring`` with :class:`RewardBodyCentring`.
With both knobs unset, a built cfg is unchanged: ``rewards.py``, ``RewardStrideLength`` and
``crab_hex_stride_reward.py`` are never imported by this module and stay untouched.

The stroke gate's clock speed is computed inside the math (``clock_speed``) from the command with
the facing relay's yaw rate removed on MOVE slots, using the env's own ``omni_clock_turn_radius``. On
TURN / MOVE_TURN slots it equals ``parkour_isaaclab.envs.mdp.rewards._clock_cmd_speed`` exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_clock_reward import FOOT_ORDER
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_command_math import SLOT_MOVE, SLOT_TURN, wrap_to_pi
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_math import (
    C_MAX,
    CENTRING_MIN_CMD_M_S,
    CENTRING_PHI_LO,
    CENTRING_PHI_SAT,
    CENTRING_SIGMA_M,
    CENTRING_TAU_S,
    DRHO,
    POWER,
    PROG_LO,
    PROG_TAU_S,
    PROG_WIDTH,
    RHO0,
    body_centring_step,
    foot_stroke_step,
    init_centring_state,
    init_stroke_state,
    reset_centring_state,
    reset_stroke_state,
)

if TYPE_CHECKING:
    from isaaclab.managers import RewardTermCfg

    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


def _resolved_body_names(entity, body_ids) -> list[str]:
    """The entity's own body names at ``body_ids`` (a slice(None) means "every body")."""
    names = entity.body_names
    if isinstance(body_ids, slice):
        return list(names)
    return [names[i] for i in body_ids]


def _assert_foot_order(asset, asset_cfg: SceneEntityCfg, sensor, sensor_cfg: SceneEntityCfg) -> None:
    """The articulation's and contact sensor's resolved foot bodies must agree, element for element.

    ``preserve_order=True`` on both ``SceneEntityCfg``\\ s is what makes this meaningful -- without
    it either resolver would sort its ids into scene order and the two lists could still match by
    accident. A mismatch here would price one foot's own stroke against a DIFFERENT foot's contact
    force -- a silent wrong-leg pairing, not a crash -- so it is checked once, loudly, at term
    construction rather than trusted from the two ``SceneEntityCfg``\\ s' params.
    """
    asset_names = _resolved_body_names(asset, asset_cfg.body_ids)
    sensor_names = _resolved_body_names(sensor, sensor_cfg.body_ids)
    # The legs themselves, in FOOT_ORDER (``FL_Footpad`` -> ``FL``): the per-foot ``pay`` the
    # telemetry and the replay parity check read is indexed by FOOT_ORDER.
    in_order = len(asset_names) == len(FOOT_ORDER) and all(
        name.startswith(f"{leg}_") for name, leg in zip(asset_names, FOOT_ORDER))
    if asset_names != sensor_names or not in_order:
        raise RuntimeError(
            "RewardFootStroke: the articulation's resolved foot bodies "
            f"{asset_names} and the contact sensor's {sensor_names} must match element for "
            f"element and number {len(FOOT_ORDER)} ({FOOT_ORDER}); a mismatch would price one "
            "foot's own sweep against another foot's contact force with no error at all. Check "
            "that both SceneEntityCfg.body_names name the six foot bodies in FOOT_ORDER with "
            "preserve_order=True."
        )


def _omni_command_term(env: ParkourManagerBasedRLEnv, command_name: str):
    """The omni command term; raises by name if it is not one (no ``slot_id``)."""
    term = env.command_manager.get_term(command_name)
    if getattr(term, "slot_id", None) is None:
        raise RuntimeError(
            f"crab_hex_omni_stroke_rewards: the command term {command_name!r} ({type(term).__name__}) has no "
            "slot_id; the stroke and centring terms need the omni command (CrabHexOmniCommand)."
        )
    return term


class _OncePerStep:
    """Stateful terms must advance exactly once per env step: raise if called twice in the same step.

    Also detects a command resample: the slot, the travel target or the facing target changed since this
    term's previous call. NaN after a reset compares unequal, so a fresh env counts as resampled.
    """

    def _init_resample(self, num_envs: int) -> None:
        self._prev_cmd = torch.full((num_envs, 3), float("nan"), device=self.device)

    def _reset_resample(self, idx) -> None:
        self._prev_cmd[idx] = float("nan")

    def _resampled(self, cmd_term) -> torch.Tensor:
        now = torch.stack((cmd_term.slot_id.float(), cmd_term.travel_dir_w, cmd_term.heading_target), dim=-1)
        resampled = (now != self._prev_cmd).any(dim=-1)
        self._prev_cmd.copy_(now)
        return resampled

    def _check_once(self, env: ParkourManagerBasedRLEnv) -> None:
        step = int(env.common_step_counter)
        if getattr(self, "_last_step", None) == step:
            raise RuntimeError(
                f"{type(self).__name__} was called twice in env step {step}; its per-foot state would advance "
                "twice. Read telemetry from the reward manager's values, never by re-calling the term."
            )
        self._last_step = step


class RewardFootStroke(ManagerTermBase, _OncePerStep):
    """Per-foot loaded stance-stroke income (stride/posture plan, replaces ``RewardStrideLength``).

    See ``crab_hex_omni_stroke_math.foot_stroke_step`` for the math, including why
    ``RewardStrideLength``'s ``compute_first_contact`` / ``compute_first_air`` edges were dropped (the
    float32 sensor timestamp drift that kills them about 2 s into every episode). This wrapper owns the
    per-env, per-foot :class:`~crab_hex_omni_stroke_math.StrokeState` between steps, and keeps the last
    per-foot ``pay`` for telemetry and the offline replay tool's parity check.

    Only TURN / MOVE_TURN slots keep the command's yaw rate in the projection. On MOVE slots it is only
    the facing relay's P-control, so the math's ``turning`` mask drops it.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.body_ids = asset_cfg.body_ids
        self.sensor_body_ids = sensor_cfg.body_ids
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        _assert_foot_order(asset, asset_cfg, sensor, sensor_cfg)
        num_feet = len(FOOT_ORDER)
        self.turn_radius = float(getattr(env.cfg, "omni_clock_turn_radius", 0.0))
        self.state = init_stroke_state(env.num_envs, num_feet, device=self.device)
        self._init_resample(env.num_envs)
        #: Last per-foot stance pay (0 except on a stance's close step) -- telemetry / replay parity.
        self.pay = torch.zeros(env.num_envs, num_feet, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_stroke_state(self.state, env_ids)
        self._reset_resample(idx)
        self.pay[idx] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        rho0: float = RHO0,
        drho: float = DRHO,
        power: float = POWER,
        c_max: float = C_MAX,
        prog_tau: float = PROG_TAU_S,
        prog_lo: float = PROG_LO,
        prog_width: float = PROG_WIDTH,
    ) -> torch.Tensor:
        self._check_once(env)
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        cmd_term = _omni_command_term(env, command_name)
        reward, self.state, self.pay = foot_stroke_step(
            contact_sensor.data.net_forces_w[:, self.sensor_body_ids].norm(dim=-1),
            asset.data.body_pos_w[:, self.body_ids],
            asset.data.root_pos_w,
            asset.data.heading_w,
            env.command_manager.get_command(command_name),
            cmd_term.stand_mask(),
            asset.data.root_lin_vel_b[:, :2],
            asset.data.root_ang_vel_b[:, 2],
            self.state,
            env.step_dt,
            self.turn_radius,
            cmd_term.slot_id >= SLOT_TURN,
            self._resampled(cmd_term),
            rho0, drho, power, c_max, prog_tau, prog_lo, prog_width,
        )
        return reward


class RewardBodyCentring(ManagerTermBase, _OncePerStep):
    """Body-centring bonus for a commanded lateral hold (stride/posture plan, the P arm).

    See ``crab_hex_omni_stroke_math.body_centring_step`` for the math: an EMA'd six-foot centroid and
    the body's lead over it along the command, paid on the lateral share of the command with an EMA'd
    statue guard. The wrapper supplies two inputs:

    * ``lat_share``: ``|sin(travel - facing)|`` of the hold's INTENDED bearing, on MOVE slots only (0
      elsewhere), so a heading error cannot steer it.
    * ``resampled``: a command resample (slot, travel target or facing target changed) re-seeds the
      EMAs, so a new hold never inherits the last one's posture.

    It keeps the last body lead ``o`` for telemetry.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.body_ids = asset_cfg.body_ids
        asset: Articulation = env.scene[asset_cfg.name]
        names = _resolved_body_names(asset, asset_cfg.body_ids)
        if len(names) != len(FOOT_ORDER) or not all(n.startswith(f"{leg}_") for n, leg in zip(names, FOOT_ORDER)):
            raise RuntimeError(
                f"RewardBodyCentring: asset_cfg must resolve to the six foot bodies in FOOT_ORDER {FOOT_ORDER}, "
                f"got {names}; the centroid would be taken over the wrong bodies with no error at all."
            )
        self.state = init_centring_state(env.num_envs, device=self.device)
        self._init_resample(env.num_envs)
        #: Last body lead over the six-foot centroid (m), :func:`crab_hex_omni_stroke_math.body_lead`
        #: -- the readout's ``body_offset`` -- kept for telemetry.
        self.o = torch.zeros(env.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_centring_state(self.state, env_ids)
        self._reset_resample(idx)
        self.o[idx] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        sigma: float = CENTRING_SIGMA_M,
        tau: float = CENTRING_TAU_S,
        min_cmd: float = CENTRING_MIN_CMD_M_S,
        phi_lo: float = CENTRING_PHI_LO,
        phi_sat: float = CENTRING_PHI_SAT,
    ) -> torch.Tensor:
        self._check_once(env)
        asset: Articulation = env.scene[asset_cfg.name]
        cmd_term = _omni_command_term(env, command_name)
        slot = cmd_term.slot_id
        resampled = self._resampled(cmd_term)
        bearing = wrap_to_pi(cmd_term.travel_dir_w - cmd_term.heading_target)
        lat_share = torch.where(slot == SLOT_MOVE, torch.sin(bearing).abs(), torch.zeros_like(bearing))
        b, self.state, self.o = body_centring_step(
            asset.data.body_pos_w[:, self.body_ids],
            asset.data.root_pos_w,
            asset.data.heading_w,
            env.command_manager.get_command(command_name),
            cmd_term.stand_mask(),
            asset.data.root_lin_vel_b[:, :2],
            self.state,
            env.step_dt,
            lat_share,
            resampled,
            sigma, tau, min_cmd, phi_lo, phi_sat,
        )
        return b
