"""Unit tests for the crab-hex stride-length reward's pure step function (Milestone 18 follow-on).

Pure torch -- no Isaac Sim needed. These pin down ``crab_hex_stride_reward.py``'s design goals:

1. Only stance-phase (foot planted) body progress along the commanded direction is rewarded --
   swing-phase movement contributes nothing, since a foot in the air can't push the robot. This
   is what structurally eliminates the earlier "snap" exploit (a leg covering its whole joint
   range in a single physics step to bank reward without the robot actually moving).
2. Motion in the wrong direction during stance earns nothing (not a penalty).
3. Accumulated stance progress is convex in ``power``, so one long productive stance outscores
   several short ones covering the same net progress (same anti-tippy-tap rationale as the
   earlier hip-yaw-diff version).
4. ``min_phase_duration`` excludes stance phases too brief to trust, without letting a rejected
   phase's accumulator leak into the next one.

(v4, a later per-foot touchdown-to-touchdown redesign, was tried and reverted -- see
``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0106_stride_length_v4/CHANGELOG.md``. This file tests the reinstated v3 design.)
"""

import sys
from pathlib import Path

import pytest
import torch

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from crab_hex_stride_reward import stride_length_reward_step  # noqa: E402

DT = 0.02


def _run(
    vel_seq,
    cmd,
    contact_seq,
    power=2.0,
    min_phase_duration=0.1,
    min_cmd_norm=0.12,
    n_legs=1,
):
    """Drive the pure function across a sequence of steps for a single env.

    Args:
        vel_seq: list of ``(vx, vy)`` body-frame velocities, one per step.
        cmd: ``(cx, cy)`` commanded planar velocity, held constant across the sequence.
        contact_seq: list of bool, contact state per step (single leg, broadcast to n_legs).
        n_legs: number of (identical) legs to simulate in parallel, for batching checks.

    Returns:
        list of per-step reward tensors (shape ``[1]``), one per step.
    """
    stance_progress = torch.zeros(1, n_legs)
    command_xy = torch.tensor([[cmd[0], cmd[1]]], dtype=torch.float32)
    rewards = []
    prev_contact = False
    for contact in contact_seq:
        first_contact = torch.tensor([[contact and not prev_contact] * n_legs])
        first_air = torch.tensor([[(not contact) and prev_contact] * n_legs])
        in_contact = torch.tensor([[contact] * n_legs])
        # last_contact_time is only consulted where first_air is True; give it a value that
        # would clear the default gate unless a test overrides min_phase_duration to probe it.
        last_contact_time = torch.full((1, n_legs), 999.0)
        vel = vel_seq[len(rewards) % len(vel_seq)]
        root_lin_vel_b_xy = torch.tensor([[vel[0], vel[1]]], dtype=torch.float32)
        reward, stance_progress = stride_length_reward_step(
            root_lin_vel_b_xy, command_xy, in_contact, first_contact, first_air,
            last_contact_time, stance_progress, DT, power, min_phase_duration, min_cmd_norm,
        )
        rewards.append(reward)
        prev_contact = contact
    return rewards


def test_stance_phase_forward_progress_is_rewarded_at_liftoff():
    """A leg planted for several steps while the body moves in the commanded direction should
    be rewarded, exactly once, at the step it lifts off."""
    # 5 steps of stance at 1.0 m/s along +x, command is also +x -> progress = 5 * 1.0 * DT
    contact_seq = [True, True, True, True, True, False]
    rewards = _run(vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, min_phase_duration=0.0)
    expected_progress = 5 * 1.0 * DT
    assert [r.item() for r in rewards[:5]] == [0.0] * 5  # nothing paid out mid-stance
    assert rewards[5].item() == pytest.approx(expected_progress**2, abs=1e-6)  # paid at liftoff


def test_swing_phase_contributes_nothing_even_with_fast_body_motion():
    """The whole point of dropping swing-phase reward: rapid body motion while every leg is
    airborne (no stance at all) must never be rewarded, however fast."""
    contact_seq = [False] * 10
    rewards = _run(vel_seq=[(10.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, min_phase_duration=0.0)
    assert all(r.item() == pytest.approx(0.0, abs=1e-9) for r in rewards)


def test_wrong_direction_motion_earns_nothing_not_a_penalty():
    """Stance-phase motion opposite the commanded direction must score exactly zero, not a
    negative reward -- "only rewards the part that contributes", not an active penalty."""
    contact_seq = [True, True, True, False]
    rewards = _run(vel_seq=[(-1.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, min_phase_duration=0.0)
    assert rewards[-1].item() == pytest.approx(0.0, abs=1e-9)
    assert all(r.item() >= 0.0 for r in rewards)


def test_sideways_motion_is_projected_not_ignored():
    """A diagonal command should reward only the aligned component of body velocity."""
    contact_seq = [True, True, False]
    # command is pure +x; body moves at 45 degrees -- only the x-component should count
    rewards = _run(vel_seq=[(1.0, 1.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, min_phase_duration=0.0)
    expected_progress = 2 * 1.0 * DT  # only vx=1.0 contributes, vy=1.0 is orthogonal to command
    assert rewards[-1].item() == pytest.approx(expected_progress**2, abs=1e-6)


def test_single_long_stance_beats_split_into_two_short_ones():
    """Core design goal carried over from the hip-yaw version: covering a given amount of
    forward progress in one sustained stance must outscore splitting it into two shorter
    stances (same total steps in contact, same net progress) -- otherwise the term would be
    blind to tippy-tapping through stance as much as through swing."""
    # scenario A: one continuous 4-step stance
    rewards_a = _run(
        vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0),
        contact_seq=[True, True, True, True, False], min_phase_duration=0.0,
    )
    total_a = sum(r.item() for r in rewards_a)

    # scenario B: the same 4 total contact-steps split into two 2-step stances
    rewards_b = _run(
        vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0),
        contact_seq=[True, True, False, True, True, False], min_phase_duration=0.0,
    )
    total_b = sum(r.item() for r in rewards_b)

    R = 4 * 1.0 * DT
    assert total_a == pytest.approx(R**2, abs=1e-6)
    assert total_b == pytest.approx(2 * (R / 2) ** 2, abs=1e-6)
    assert total_a > total_b


def test_short_stance_is_not_rewarded_and_does_not_leak_into_next_stance():
    """A stance shorter than min_phase_duration pays out nothing, and its (rejected) progress
    must not carry over into a later, genuine stance for the same leg."""
    vel = torch.tensor([[1.0, 0.0]])
    cmd = torch.tensor([[1.0, 0.0]])

    # step 0: touchdown of a brief stance
    _, accum = stride_length_reward_step(
        vel, cmd, torch.tensor([[True]]), torch.tensor([[True]]), torch.tensor([[False]]),
        torch.tensor([[999.0]]), torch.zeros(1, 1), DT, 2.0, 0.1,
    )
    # step 1: liftoff after only 0.02s of stance -- under the default min_phase_duration=0.1 gate
    reward_rejected, accum = stride_length_reward_step(
        vel, cmd, torch.tensor([[False]]), torch.tensor([[False]]), torch.tensor([[True]]),
        torch.tensor([[0.02]]), accum, DT, 2.0, 0.1,
    )
    assert reward_rejected.item() == pytest.approx(0.0, abs=1e-9)

    # step 2: a genuine, later touchdown for the same leg must start from zero, not leak the
    # rejected phase's progress
    _, accum = stride_length_reward_step(
        vel, cmd, torch.tensor([[True]]), torch.tensor([[True]]), torch.tensor([[False]]),
        torch.tensor([[999.0]]), accum, DT, 2.0, 0.1,
    )
    assert accum.item() == pytest.approx(1.0 * DT, abs=1e-6)  # only this step's contribution


def test_min_cmd_norm_gates_near_zero_commands():
    """No defined "desired direction" when the command is ~stopped -- no progress accumulated
    even if the body happens to be moving."""
    contact_seq = [True, True, True, False]
    rewards = _run(
        vel_seq=[(1.0, 0.0)], cmd=(0.01, 0.0), contact_seq=contact_seq,
        min_cmd_norm=0.12, min_phase_duration=0.0,
    )
    assert rewards[-1].item() == pytest.approx(0.0, abs=1e-9)


def test_power_parameter_is_respected():
    contact_seq = [True, True, False]
    progress = 2 * 1.0 * DT
    rewards_linear = _run(vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, power=1.0, min_phase_duration=0.0)
    rewards_square = _run(vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, power=2.0, min_phase_duration=0.0)
    assert rewards_linear[-1].item() == pytest.approx(progress, abs=1e-6)
    assert rewards_square[-1].item() == pytest.approx(progress**2, abs=1e-6)


def test_multiple_legs_batch_independently():
    """Each leg accumulates its own stance progress independently within the same env."""
    reward, new_accum = stride_length_reward_step(
        root_lin_vel_b_xy=torch.tensor([[1.0, 0.0]]),
        command_xy=torch.tensor([[1.0, 0.0]]),
        in_contact=torch.tensor([[True, False, False]]),  # leg 2 already airborne -- first_air fires
        first_contact=torch.tensor([[False, False, False]]),
        first_air=torch.tensor([[False, False, True]]),
        last_contact_time=torch.tensor([[999.0, 999.0, 999.0]]),
        stance_progress=torch.tensor([[0.05, 0.0, 0.03]]),
        dt=DT,
        power=2.0,
        min_phase_duration=0.0,
    )
    # leg 0: still in stance, not paid out yet, but accumulates this step
    # leg 1: not in contact, untouched
    # leg 2: lifts off this step -- paid out at its pre-step accumulated value (0.03)
    assert reward.item() == pytest.approx(0.03**2, abs=1e-6)
    assert new_accum[0, 0].item() == pytest.approx(0.05 + 1.0 * DT, abs=1e-6)
    assert new_accum[0, 1].item() == pytest.approx(0.0, abs=1e-9)


def test_reset_at_next_touchdown_after_a_completed_stance():
    """After a stance is rewarded at liftoff, the very next touchdown must start counting from
    zero, not from the just-paid-out total."""
    contact_seq = [True, True, False, False, True, True, True, False]
    rewards = _run(vel_seq=[(1.0, 0.0)], cmd=(1.0, 0.0), contact_seq=contact_seq, min_phase_duration=0.0)
    first_stance_reward = (2 * 1.0 * DT) ** 2
    second_stance_reward = (3 * 1.0 * DT) ** 2
    assert rewards[2].item() == pytest.approx(first_stance_reward, abs=1e-6)  # liftoff is step index 2
    assert rewards[-1].item() == pytest.approx(second_stance_reward, abs=1e-6)


def test_batched_and_broadcastable():
    """Matches how RewardStrideLength calls this every step across all envs/legs at once."""
    n_envs, n_legs = 256, 6
    root_lin_vel_b_xy = torch.randn(n_envs, 2, dtype=torch.float64)
    command_xy = torch.randn(n_envs, 2, dtype=torch.float64)
    stance_progress = torch.rand(n_envs, n_legs, dtype=torch.float64)
    in_contact = torch.randint(0, 2, (n_envs, n_legs)).bool()
    first_contact = torch.randint(0, 2, (n_envs, n_legs)).bool() & in_contact
    first_air = torch.randint(0, 2, (n_envs, n_legs)).bool() & ~in_contact
    last_contact_time = torch.rand(n_envs, n_legs, dtype=torch.float64)

    reward, new_accum = stride_length_reward_step(
        root_lin_vel_b_xy, command_xy, in_contact, first_contact, first_air,
        last_contact_time, stance_progress, DT, power=2.0, min_phase_duration=0.1, min_cmd_norm=0.12,
    )
    assert reward.shape == (n_envs,)
    assert new_accum.shape == (n_envs, n_legs)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(new_accum).all()
    assert (reward >= 0).all()  # squared, non-negative accumulated progress -- never negative
