"""Pure-torch stride-length reward math for the crab hexapod (Milestone 18 follow-on).

Rewards each leg's **stance-phase** contribution to real body progress along the commanded
direction. Earlier versions of this term rewarded hip-yaw (theta_hip, see
``crab_hex_cam_mapping.py``) movement across *both* swing (liftoff-to-touchdown) and stance
(touchdown-to-liftoff) phases -- but only a planted foot can actually push the robot forward; a
foot repositioning in the air doesn't move the body at all. Rewarding swing-phase joint movement
turned out to be actively exploitable: real training data showed one leg "snapping" through ~195mm
of horizontal foot travel within a single 20ms physics step (~9.9 m/s, contact force cleanly
zero -- a real, fast motion, not sensor noise) to bank the same reward as a slow, deliberate
stride, since nothing measured *how* that displacement related to the robot actually moving.

This version measures something physically meaningful instead: while a leg is planted, it
integrates the robot's body-frame linear velocity projected onto the commanded direction (clipped
to non-negative -- motion in the *wrong* direction earns nothing, not a penalty) over the whole
stance phase, and grants the accumulated total as a reward exactly at liftoff. Because the reward
now comes from genuine body displacement rather than a joint-space proxy, the original swing-phase
snap exploit is eliminated structurally (swing is no longer rewarded at all), with no separate
velocity-cost term needed to counter it.

``power`` keeps the reward convex in stance-phase progress (same rationale as before): splitting a
given amount of forward push into several short stance phases scores less than delivering it in
one sustained stance, which is what keeps the term from preferring "tippy-tap" over confident
striding. ``min_phase_duration`` excludes stance phases too brief to be a trustworthy signal
(guards against a spurious one-step contact reading, not against exploit-by-speed the way it did
in the swing-phase version -- fast *body* progress during a real stance is exactly the desired
outcome, not something to penalize).

**v4 addendum (reverted, see `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/CHANGELOG.md`).** A later attempt
redefined this term as each foot's own signed touchdown-to-touchdown world-frame displacement.
That version did finally beat `motor_reversal_on`'s measured 2b2 stride length (the metric this
whole redesign chain has been chasing), but only by producing the *worst* tippy-tap fraction in
the entire comparison series at both the flat and 2b2 stages, alongside worse training-time
`crab_failure`, `slip_ratio`, and forward-progress than this (v3) design at every stage. Reverted
back to this stance-only body-progress design, which remains the strongest run in the series on
every training-stability measure.

See ``RewardStrideLength`` in ``parkour_isaaclab/envs/mdp/rewards.py`` for the stateful
``ManagerTermBase`` wrapper that drives this from real env/sensor data; this module stays free of
any ``isaaclab`` import so it can be unit-tested without Isaac Sim, matching
``crab_hex_cam_mapping.py``'s own isolation pattern.
"""

from __future__ import annotations

import torch


def stride_length_reward_step(
    root_lin_vel_b_xy: torch.Tensor,
    command_xy: torch.Tensor,
    in_contact: torch.Tensor,
    first_contact: torch.Tensor,
    first_air: torch.Tensor,
    last_contact_time: torch.Tensor,
    stance_progress: torch.Tensor,
    dt: float,
    power: float = 2.0,
    min_phase_duration: float = 0.1,
    min_cmd_norm: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the stride-length reward's state machine.

    Args:
        root_lin_vel_b_xy: ``[N, 2]`` robot body-frame planar linear velocity (m/s).
        command_xy: ``[N, 2]`` commanded planar velocity (body frame, same convention as
            ``reward_forward_progress_along_command``).
        in_contact: ``[N, 6]`` bool, whether each leg is touching the ground *this* step.
        first_contact: ``[N, 6]`` bool, this leg touched down this step (starts a new stance).
        first_air: ``[N, 6]`` bool, this leg lifted off this step (ends a stance -> reward).
        last_contact_time: ``[N, 6]`` duration (s) of the stance phase that just ended -- only
            meaningful where ``first_air`` is true.
        stance_progress: ``[N, 6]`` state in -- accumulated forward-progress (m) for each leg's
            current, in-progress stance phase.
        dt: physics step duration (s), for integrating velocity into distance.
        power: exponent applied to the accumulated stance progress; >1 is strictly convex, so one
            long productive stance outscores several short ones covering the same net progress.
        min_phase_duration: stance phases lasting this long or less are not rewarded at liftoff
            (their accumulated progress is simply not paid out); the accumulator itself is reset
            at the *next* touchdown regardless, so a rejected phase can't leak into a later one.
        min_cmd_norm: below this commanded planar speed there's no defined "desired direction",
            so no progress is accumulated (matches ``reward_forward_progress_along_command``).

    Returns:
        ``(reward[N], new_stance_progress[N, 6])``.
    """
    cmd_norm = torch.norm(command_xy, dim=1, keepdim=True)
    command_dir_xy = command_xy / (cmd_norm + 1e-8)
    cmd_active = (cmd_norm.squeeze(-1) > min_cmd_norm).float()

    step_progress = torch.clamp((root_lin_vel_b_xy * command_dir_xy).sum(dim=1), min=0.0) * dt
    step_progress = step_progress * cmd_active
    step_progress_per_leg = step_progress.unsqueeze(-1).expand_as(stance_progress)

    # Fresh accumulator at touchdown, then add this step's contribution while planted. In_contact
    # is False at the exact step first_air fires (the foot has already left), so this adds nothing
    # new at liftoff -- the reward below sees exactly the completed stance's total.
    accum = torch.where(first_contact, torch.zeros_like(stance_progress), stance_progress)
    accum = accum + step_progress_per_leg * in_contact.float()

    valid_liftoff = first_air & (last_contact_time > min_phase_duration)
    reward = (accum.pow(power) * valid_liftoff.float()).sum(dim=1)

    return reward, accum
