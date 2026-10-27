"""``clock_command_speed``: the one seam the gait clock's cadence is read through.

The gait clock (``CrabHexDelayedJointPositionAction.process_actions``) and the two clock
rewards (``RewardClockContactSchedule``, ``RewardClockSwingApex``) used to each spell
``cmd[:, 0].abs()`` inline. They now go through this function so an omnidirectional sibling
task can drive the same clock from a planar command plus a yaw rate -- which is only safe
while the DEFAULT (``planar=False``) return is the old expression bit for bit. That is what
the first test pins, on random tensors including negatives (the ``.abs()`` is load-bearing:
a backward command must still run the clock).

Pure torch -- no Isaac. Loaded from the module file the same way
``test_crab_hex_clock_reward.py`` loads it.
"""

import sys
from pathlib import Path

import torch

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from crab_hex_clock_reward import clock_command_speed  # noqa: E402

SEED = 20260916
N = 257


def _random_cmds() -> list[torch.Tensor]:
    """Random (vx, vy, wz) batches, plus the edge rows the gate cares about."""
    g = torch.Generator().manual_seed(SEED)
    out = [(torch.rand(N, 3, generator=g) - 0.5) * 2.0 for _ in range(4)]
    out.append(torch.rand(N, 3, generator=g) * 4.0 - 2.0)  # wider band
    out.append(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.30, 0.0, 0.0],  # backward: must still read 0.30
                [0.30, 0.0, 0.0],
                [0.0, 0.25, 0.0],  # pure lateral: 0 on the non-planar branch
                [0.0, 0.0, -0.30],  # pure turn
                [1e-9, -1e-9, 1e-9],
            ]
        )
    )
    return out


def test_non_planar_return_is_exactly_the_forward_expression():
    for cmd in _random_cmds():
        got = clock_command_speed(cmd)
        assert torch.equal(got, cmd[:, 0].abs())
        # the default argument, spelled out, and the explicit False: same thing
        assert torch.equal(clock_command_speed(cmd, False), cmd[:, 0].abs())
        # turn_radius is inert on the forward branch, whatever it is
        assert torch.equal(clock_command_speed(cmd, False, 1.2), cmd[:, 0].abs())
        assert got.shape == (cmd.shape[0],)
        assert got.dtype == cmd.dtype


def test_non_planar_ignores_the_lateral_and_yaw_channels():
    cmd = torch.tensor([[0.22, 0.0, 0.0], [0.22, 5.0, -7.0]])
    speed = clock_command_speed(cmd)
    assert torch.equal(speed, torch.tensor([0.22, 0.22]))


def test_planar_is_the_max_of_the_command_norm_and_the_turn_speed():
    for cmd in _random_cmds():
        for r in (0.0, 1.2, 2.0):
            got = clock_command_speed(cmd, True, r)
            want = torch.maximum(
                torch.linalg.norm(cmd[:, :2], dim=-1), r * cmd[:, 2].abs()
            )
            assert torch.equal(got, want)


def test_planar_with_zero_turn_radius_is_the_planar_norm():
    for cmd in _random_cmds():
        got = clock_command_speed(cmd, True, 0.0)
        assert torch.equal(got, torch.linalg.norm(cmd[:, :2], dim=-1))
        # the default turn_radius is 0.0
        assert torch.equal(clock_command_speed(cmd, True), got)


def test_planar_keeps_the_clock_running_on_a_pure_turn():
    """The design number: r_turn 1.2 m x the smallest non-zero yaw command 0.2 rad/s =
    0.24 m/s, above CLOCK_CMD_STOP_M_S (0.2). A pure turn must not read as a stop."""
    from crab_hex_dimensions import CLOCK_CMD_STOP_M_S  # noqa: E402

    cmd = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]])
    speed = clock_command_speed(cmd, True, 1.2)
    assert torch.allclose(speed, torch.tensor([0.24, 0.36, 0.0]))
    assert bool((speed[:2] > CLOCK_CMD_STOP_M_S).all())
    assert not bool(speed[2] > CLOCK_CMD_STOP_M_S)
    # ... and the forward branch would call that same pure turn a stop, as it always has.
    assert torch.equal(clock_command_speed(cmd), torch.zeros(3))


def test_planar_lateral_command_reads_as_travel():
    cmd = torch.tensor([[0.0, 0.25, 0.0], [-0.25, 0.0, 0.0]])
    speed = clock_command_speed(cmd, True, 1.2)
    assert torch.allclose(speed, torch.tensor([0.25, 0.25]))
