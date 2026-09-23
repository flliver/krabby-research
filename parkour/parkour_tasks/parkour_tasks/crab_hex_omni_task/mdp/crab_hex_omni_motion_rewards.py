# SPDX-License-Identifier: BSD-3-Clause
"""Motion-quality penalty terms (gait-stabilisation plan, step 2, 2026-09-22). DEFAULT OFF.

No term here is in any shipped rewards cfg: ``config/crab_hex/motion_knobs.apply_motion_knobs``
ADDS each one to ``cfg.rewards`` only when its weight knob is set to a non-zero (negative)
value, so every recorded run's reward manager -- and its config dump -- is untouched.

==========================  ===============================  ==========================================
term (``rewards.<name>``)   knob                             per-env value (weight <= 0 multiplies it)
==========================  ===============================  ==========================================
``penalty_action_smooth2``  ``KRABBY_ACTION_SMOOTH2_W``      ``||a_t - 2 a_{t-1} + a_{t-2}||^2`` from the
                                                             action term's history (the policy's
                                                             clipped output), 0 for the first two
                                                             steps of an episode
``penalty_joint_vel``       ``KRABBY_JOINT_VEL_W``           ``sum(qdot^2)`` over the rod joints
``penalty_foot_slip``       ``KRABBY_FOOT_SLIP_W``           ``sum ||v_foot_xy||`` over feet whose
                                                             contact force exceeds 50 N (UNSQUARED,
                                                             phase-independent -- the ``feet_slide``
                                                             form with a real load gate)
``penalty_stand_pose``      ``KRABBY_STAND_POSE_W``          ``sum |q - q_default|`` over the rod
                                                             joints, on STAND-slot rows only
==========================  ===============================  ==========================================

Pure torch at module scope (the functions duck-type ``env``), so the unit tests drive them with
stand-in envs and pin exactly the arithmetic that trains.
"""

from __future__ import annotations

import torch


def penalty_action_second_diff(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared L2 of the second difference of the policy's last three actions, ``(N,)``.

    Reads ``action_history_buf`` (``(N, H, A)``, newest last; the term's clipped policy output,
    before the delay) -- the 8-step history every flat-walk / phase-2 action cfg carries. The
    history is zero-filled at a reset, so the first two steps of an episode would price the jump
    from zeros; they are masked (``episode_length_buf >= 3``).
    """
    buf = env.action_manager.get_term(action_name).action_history_buf
    if buf.shape[1] < 3:
        raise RuntimeError(
            f"penalty_action_second_diff needs an action history of >= 3 steps, got {buf.shape[1]}"
        )
    d2 = buf[:, -1] - 2.0 * buf[:, -2] + buf[:, -3]
    valid = (env.episode_length_buf >= 3).to(d2.dtype)
    return torch.sum(torch.square(d2), dim=1) * valid


def penalty_joint_vel_l2(env, asset_cfg) -> torch.Tensor:
    """``sum(qdot^2)`` over ``asset_cfg.joint_ids`` (Isaac Lab's ``joint_vel_l2``, restated)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def penalty_loaded_foot_slip(env, sensor_cfg, asset_cfg, force_threshold: float = 50.0) -> torch.Tensor:
    """Summed planar speed of the LOADED feet (contact force > ``force_threshold`` N), ``(N,)``.

    The ``feet_slide`` form (Isaac Lab locomotion ``mdp.feet_slide``: speed times a contact
    mask, unsquared) with two changes: the mask is a real load (50 N, the clock reward's
    ``TOUCH_REF_N``; ``feet_slide`` uses the max over the force history > 1 N, which a
    brushing swing foot trips) and it reads the CURRENT force. Independent of the gait clock:
    a planted foot should not slide whatever the schedule says.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    loaded = (force > force_threshold).to(force.dtype)
    speed = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(speed * loaded, dim=1)


def penalty_stand_pose_l1(env, asset_cfg, command_name: str = "base_velocity") -> torch.Tensor:
    """L1 deviation of ``asset_cfg.joint_ids`` from the default pose, on STAND-slot rows only.

    The stand gate is the omni command term's ``stand_mask()`` (``slot_id == SLOT_STAND``, never
    ``type_id``), the same predicate the stand hold / park / observation knobs key on.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    dev = torch.sum(torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    stand = env.command_manager.get_term(command_name).stand_mask().to(dev.dtype)
    return dev * stand
