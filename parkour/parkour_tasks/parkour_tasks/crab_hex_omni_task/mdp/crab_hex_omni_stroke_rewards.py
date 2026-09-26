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

``_clock_cmd_speed`` is read the way ``crab_hex_omni_rewards.py`` reads the forward mdp helpers --
through the ``parkour_isaaclab.envs.mdp.rewards`` module object, never re-implemented here, so the
gate this term uses can never drift from the gait clock's own definition of "commanded speed".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_clock_reward import FOOT_ORDER
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_math import (
    C_MAX,
    CENTRING_MIN_CMD_M_S,
    CENTRING_PHI_SAT,
    CENTRING_SIGMA_M,
    CENTRING_TAU_S,
    DRHO,
    POWER,
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


class RewardFootStroke(ManagerTermBase):
    """Per-foot loaded stance-stroke income (stride/posture plan, replaces ``RewardStrideLength``).

    See ``crab_hex_omni_stroke_math.foot_stroke_step`` for the math and the full rationale for
    dropping ``RewardStrideLength``'s ``compute_first_contact`` / ``compute_first_air`` edges (the
    float32 sensor timestamp drift that kills them about 2 s into every episode). This wrapper
    owns the per-env, per-foot :class:`~crab_hex_omni_stroke_math.StrokeState` between steps and
    the last per-foot ``pay`` for telemetry / the offline replay tool's parity check.
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
        self.state = init_stroke_state(env.num_envs, num_feet, device=self.device)
        #: Last per-foot stance pay (0 except on a stance's close step) -- telemetry / replay parity.
        self.pay = torch.zeros(env.num_envs, num_feet, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_stroke_state(self.state, env_ids)
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
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force = contact_sensor.data.net_forces_w[:, self.sensor_body_ids].norm(dim=-1)
        foot_pos_w = asset.data.body_pos_w[:, self.body_ids]
        root_pos_w = asset.data.root_pos_w
        heading = asset.data.heading_w
        cmd = env.command_manager.get_command(command_name)
        stand = env.command_manager.get_term(command_name).stand_mask()
        v_clk = mdp_rewards._clock_cmd_speed(env, cmd)
        reward, self.state, self.pay = foot_stroke_step(
            force, foot_pos_w, root_pos_w, heading, cmd, stand, v_clk, self.state,
            env.step_dt, rho0, drho, power, c_max,
        )
        return reward


class RewardBodyCentring(ManagerTermBase):
    """Body-centring bonus for a commanded lateral hold (stride/posture plan, the P arm).

    See ``crab_hex_omni_stroke_math.body_centring_step`` for the math: an EMA'd six-foot centroid
    and the body's lead over it along the command, paid on the lateral SHARE of the command with a
    statue guard. This wrapper owns the per-env
    :class:`~crab_hex_omni_stroke_math.CentringState` and the last body lead ``o`` for telemetry.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.body_ids = asset_cfg.body_ids
        self.state = init_centring_state(env.num_envs, device=self.device)
        #: Last body lead over the six-foot centroid (m), :func:`crab_hex_omni_stroke_math.body_lead`
        #: -- the readout's ``body_offset`` -- kept for telemetry.
        self.o = torch.zeros(env.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_centring_state(self.state, env_ids)
        self.o[idx] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        sigma: float = CENTRING_SIGMA_M,
        tau: float = CENTRING_TAU_S,
        min_cmd: float = CENTRING_MIN_CMD_M_S,
        phi_sat: float = CENTRING_PHI_SAT,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        foot_pos_w = asset.data.body_pos_w[:, self.body_ids]
        root_pos_w = asset.data.root_pos_w
        heading = asset.data.heading_w
        cmd = env.command_manager.get_command(command_name)
        stand = env.command_manager.get_term(command_name).stand_mask()
        root_lin_vel_b_xy = asset.data.root_lin_vel_b[:, :2]
        b, self.state, self.o = body_centring_step(
            foot_pos_w, root_pos_w, heading, cmd, stand, root_lin_vel_b_xy, self.state,
            env.step_dt, sigma, tau, min_cmd, phi_sat,
        )
        return b
