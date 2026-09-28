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
from parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_stroke_parity import get_recorder as _get_parity_recorder

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


def _resolve_cat_scale(cat_scale: dict, categories: Sequence[str], device=None) -> torch.Tensor:
    """``{category_name: multiplier}`` -> the resolved ``[len(categories)]`` float tensor (per-
    category stroke plan, 2026-09-27). Unlisted names -- always including ``stand`` -- default to
    ``1.0`` (irrelevant for ``stand``, which pays 0). Pure: only ``torch``, no Isaac, so it is
    unit-tested by lifting it out of this module (``test_crab_hex_omni_stroke_knobs.py``).

    Refuses a name outside ``categories``, an explicit ``stand`` key, and a ``fwd``/``back``
    asymmetry -- defence in depth: the knob that builds ``cat_scale``
    (``stroke_knobs.parse_stroke_cat_w``) already refuses all three, so only a hand-edited cfg
    reaches any check here. ``stand`` needs its OWN check rather than falling out of the unknown-name
    one: unlike a typo, ``"stand"`` IS one of ``categories`` (``TELEMETRY_CATEGORIES``), so it would
    otherwise resolve silently -- into a multiplier on a term that pays exactly 0 on a stand step.
    """
    if "stand" in cat_scale:
        raise RuntimeError(
            "RewardFootStroke: cat_scale refuses the key 'stand': it pays exactly 0 (a stance that "
            "closes on a STAND step scores nothing), so a stand multiplier would look armed and do "
            f"nothing, got {cat_scale!r}."
        )
    unknown = sorted(set(cat_scale) - set(categories))
    if unknown:
        raise RuntimeError(
            f"RewardFootStroke: cat_scale names unknown categor{'y' if len(unknown) == 1 else 'ies'} "
            f"{unknown} (allowed: {tuple(categories)})."
        )
    if float(cat_scale.get("fwd", 1.0)) != float(cat_scale.get("back", 1.0)):
        raise RuntimeError(
            f"RewardFootStroke: cat_scale requires fwd == back (the mirror tie), got "
            f"fwd={cat_scale.get('fwd', 1.0)!r} back={cat_scale.get('back', 1.0)!r}."
        )
    vec = torch.ones(len(categories), device=device)
    for name, value in cat_scale.items():
        vec[categories.index(name)] = float(value)
    return vec


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

    ``cat_scale`` (per-category stroke plan, 2026-09-27): an optional ``{category_name: multiplier}``
    dict -- ``config/crab_hex/stroke_knobs.py``'s ``KRABBY_OMNI_STROKE_CAT_W`` is what writes it, never
    a shipped cfg by hand. When present it is resolved ONCE, in :meth:`__init__`, into a
    ``[len(TELEMETRY_CATEGORIES)]`` float tensor (unlisted names, always including ``stand``, default
    to ``1.0``) and applied to the RETURNED reward only, by close-step category, in :meth:`__call__`.
    ``self.pay`` (the per-foot telemetry / replay-parity pay) stays unscaled either way.
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
        self._parity = _get_parity_recorder()
        #: ``[len(TELEMETRY_CATEGORIES)]`` float tensor or ``None`` (knob off) -- see the ``cat_scale``
        #: paragraph above. Resolved here, once, from ``cfg.params["cat_scale"]`` (written only when
        #: ``KRABBY_OMNI_STROKE_CAT_W`` is armed), so ``__call__`` never re-parses the dict or
        #: re-asserts names.
        cat_scale = cfg.params.get("cat_scale")
        self._cat_scale = None
        if cat_scale is not None:
            command_name: str = cfg.params.get("command_name", "base_velocity")
            cmd_term = _omni_command_term(env, command_name)
            self._cat_scale = _resolve_cat_scale(cat_scale, cmd_term.TELEMETRY_CATEGORIES, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_stroke_state(self.state, env_ids)
        self._reset_resample(idx)
        self.pay[idx] = 0.0
        self._parity.note_reset(env_ids)

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
        cat_scale: dict[str, float] | None = None,  # noqa: ARG002 -- resolved once in __init__ (self._cat_scale);
        # accepted here only so IsaacLab's cfg.params signature check passes when the knob writes it.
    ) -> torch.Tensor:
        self._check_once(env)
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        cmd_term = _omni_command_term(env, command_name)
        force = contact_sensor.data.net_forces_w[:, self.sensor_body_ids].norm(dim=-1)
        foot_pos_w = asset.data.body_pos_w[:, self.body_ids]
        root_pos_w = asset.data.root_pos_w
        heading = asset.data.heading_w
        cmd = env.command_manager.get_command(command_name)
        stand = cmd_term.stand_mask()
        root_lin_vel_b_xy = asset.data.root_lin_vel_b[:, :2]
        root_ang_vel_b_z = asset.data.root_ang_vel_b[:, 2]
        turning = cmd_term.slot_id >= SLOT_TURN
        resampled = self._resampled(cmd_term)
        reward, self.state, self.pay = foot_stroke_step(
            force, foot_pos_w, root_pos_w, heading, cmd, stand, root_lin_vel_b_xy, root_ang_vel_b_z,
            self.state, env.step_dt, self.turn_radius, turning, resampled,
            rho0, drho, power, c_max, prog_tau, prog_lo, prog_width,
        )
        # Per-category stroke plan (2026-09-27): the ONLY place cat_scale touches the term -- the
        # RETURNED reward, by the CLOSE-step category (the same step foot_stroke_step just paid at).
        # self.pay stays unscaled (per-foot pay=1 telemetry/replay-parity meaning). Knob off
        # (self._cat_scale is None) never calls telemetry_category(), so the op sequence is identical.
        cat = None
        if self._cat_scale is not None:
            cat = cmd_term.telemetry_category()
            reward = reward * self._cat_scale[cat]
        self._parity.record_stroke(
            int(env.common_step_counter), force=force, foot_pos_w=foot_pos_w, root_pos_w=root_pos_w,
            heading=heading, cmd=cmd, stand=stand, root_lin_vel_b_xy=root_lin_vel_b_xy,
            root_ang_vel_b_z=root_ang_vel_b_z, turning=turning, resampled=resampled, reward=reward,
            pay=self.pay, dt=env.step_dt, turn_radius=self.turn_radius,
            cat=cat, cat_scale=self._cat_scale,
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
        self._parity = _get_parity_recorder()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        reset_centring_state(self.state, env_ids)
        self._reset_resample(idx)
        self.o[idx] = 0.0
        self._parity.note_reset(env_ids)

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
        foot_pos_w = asset.data.body_pos_w[:, self.body_ids]
        root_pos_w = asset.data.root_pos_w
        heading = asset.data.heading_w
        cmd = env.command_manager.get_command(command_name)
        stand = cmd_term.stand_mask()
        root_lin_vel_b_xy = asset.data.root_lin_vel_b[:, :2]
        b, self.state, self.o = body_centring_step(
            foot_pos_w, root_pos_w, heading, cmd, stand, root_lin_vel_b_xy, self.state, env.step_dt,
            lat_share, resampled, sigma, tau, min_cmd, phi_lo, phi_sat,
        )
        self._parity.record_centring(
            int(env.common_step_counter), foot_pos_w=foot_pos_w, root_pos_w=root_pos_w, heading=heading,
            cmd=cmd, stand=stand, lat_share=lat_share, resampled=resampled, b=b, o=self.o, dt=env.step_dt,
        )
        return b
