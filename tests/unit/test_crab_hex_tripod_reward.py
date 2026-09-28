"""Unit tests for the crab-hex tripod-alternation reward's pure step function (Milestone 18
Task 1 follow-on; see ``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md`` for the
v1-v5 design history these tests pin down).

Pure torch -- no Isaac Sim needed. v5 design goals under test:

1. Income is paid ONLY at zero-crossings of the smoothed support difference ``x = s_A - s_B``:
   no static contact configuration earns anything, ever (the v1-v3b state-farming lesson).
2. Credit requires opposition: sets that move *together* (unison, tip-rock) keep ``q_anti = 0``
   even when ``x`` noise crosses zero (the v3-skate lesson from the offline replay gate).
3. Credit requires real swings on BOTH sides of the crossing (``min_amp``, min of the two
   peaks) -- shallow support wiggles earn nothing, deeper alternation earns more, maxing at
   full-set exchange.
4. Credit is band-passed on crossing period: contact chatter (< ``min_period``) and slow
   weight-shift oscillations (> ``max_period``, the v3b-drag signature) earn nothing.
5. The first crossing after a reset is never paid (``RESET_T_SINCE`` puts it out of band).
6. No reward below ``min_cmd_norm``, matching ``reward_forward_progress_along_command``.
7. The tripod-A/tripod-B foot groupings are pinned to the gait-eval convention
   (``gait_eval/metrics.py``'s ``FOOT_ORDER``/``TRIPOD_A``/``TRIPOD_B``).
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

from crab_hex_tripod_reward import (  # noqa: E402
    RESET_T_SINCE,
    S_T_SINCE,
    STATE_DIM,
    TRIPOD_A_IDX,
    TRIPOD_B_IDX,
    tripod_swap_crossing_reward_step,
)

DT = 0.02

# FOOT_ORDER = (FL, FR, ML, MR, RL, RR); tripod A = {FL, MR, RL} -> indices (0, 3, 4);
# tripod B = {FR, ML, RR} -> indices (1, 2, 5).
A_PLANTED = [True, False, False, True, True, False]
B_PLANTED = [False, True, True, False, False, True]
ALL_PLANTED = [True] * 6
ALL_AIR = [False] * 6
MIXED_STATIC = [True, True, False, False, True, False]  # 2 of A, 1 of B down, held forever


def fresh_state(n: int = 1) -> torch.Tensor:
    state = torch.zeros(n, STATE_DIM)
    state[:, S_T_SINCE] = RESET_T_SINCE
    return state


def ideal_alternation(n_cycles: int = 20, period_s: float = 0.30, overlap_s: float = 0.03):
    """Full-set tripod alternation: A stance then B stance with a brief double-support overlap
    (mirrors the synthetic ideal trace the offline replay gate scores at ~20x healthy income)."""
    seq = []
    steps = round(period_s / DT)
    a_end = round((period_s / 2 + overlap_s) / DT)
    b_start = round((period_s / 2) / DT)
    overlap = round(overlap_s / DT)
    for _ in range(n_cycles):
        for i in range(steps):
            in_a = i < a_end
            in_b = i >= b_start or i < overlap
            seq.append([
                in_a if j in TRIPOD_A_IDX else in_b for j in range(6)
            ])
    return seq


def _run(contact_seq, cmd=(0.5, 0.0), state=None, **params):
    """Drive the pure function across a sequence of steps. Returns (rewards list, final state)."""
    if state is None:
        state = fresh_state()
    command_xy = torch.tensor([[cmd[0], cmd[1]]], dtype=torch.float32)
    rewards = []
    for contact in contact_seq:
        contact_t = torch.tensor([contact], dtype=torch.bool)
        r, state = tripod_swap_crossing_reward_step(contact_t, command_xy, state, DT, **params)
        rewards.append(r)
    return rewards, state


def test_tripod_groupings_pinned():
    assert TRIPOD_A_IDX == (0, 3, 4)
    assert TRIPOD_B_IDX == (1, 2, 5)


@pytest.mark.parametrize(
    "pattern", [A_PLANTED, B_PLANTED, ALL_PLANTED, ALL_AIR, MIXED_STATIC],
    ids=["hold-A", "hold-B", "statue-all-down", "flight-all-up", "mixed-static"],
)
def test_static_states_earn_nothing(pattern):
    rewards, _ = _run([pattern] * 200)
    assert sum(float(r) for r in rewards) == 0.0


def test_ideal_alternation_earns_on_crossings_only():
    rewards, _ = _run(ideal_alternation(n_cycles=20))
    total = sum(float(r) for r in rewards)
    assert total > 0.0
    paid = [float(r) for r in rewards if float(r) > 0.0]
    # 2 crossings per 0.30s cycle; the first crossing after reset is unpaid and the
    # correlation moments need ~corr_tau to warm up, so expect most-but-not-all paid.
    assert 20 <= len(paid) <= 40
    # per-crossing credit is min(prev_peak, peak) * q_anti^2 <= 1.0 by construction
    assert max(paid) <= 1.0
    # after warmup, full-set alternation should earn substantial per-crossing credit
    assert max(paid[10:]) > 0.4


def test_unison_motion_earns_nothing():
    # all six feet down/up together: x stays ~0 -> no meaningful crossings, no amplitude
    seq = (([ALL_PLANTED] * 8) + ([ALL_AIR] * 8)) * 20
    rewards, _ = _run(seq)
    assert sum(float(r) for r in rewards) == 0.0


def test_positively_correlated_sets_earn_nearly_nothing():
    # tip-rock analog: the sets move together but A leads B by one step, so x wiggles through
    # zero -- q_anti must kill the credit (offline gate: v3-skate earned 0.013/min this way).
    lead = [
        [True, False, False, True, True, False],  # A down first
    ]
    both = [ALL_PLANTED]
    trail = [
        [False, True, True, False, False, True],  # A up first, B still down
    ]
    none = [ALL_AIR]
    seq = (lead + both * 7 + trail + none * 7) * 20
    rewards, _ = _run(seq)
    ideal_total = sum(float(r) for r in _run(ideal_alternation(n_cycles=20))[0])
    assert sum(float(r) for r in rewards) < 0.02 * ideal_total


def test_slow_alternation_out_of_band():
    # v3b-drag analog: genuine anti-phase but at 1.6s period (0.8s per crossing > max_period)
    seq = (([A_PLANTED] * 40) + ([B_PLANTED] * 40)) * 8
    rewards, _ = _run(seq)
    assert sum(float(r) for r in rewards) == 0.0


def test_single_step_chatter_earns_nothing():
    # alternating every step: crossings land below min_period and the 0.06s EMA keeps the
    # amplitude below min_amp -- both mechanisms must hold this at exactly zero
    seq = [A_PLANTED, B_PLANTED] * 100
    rewards, _ = _run(seq)
    assert sum(float(r) for r in rewards) == 0.0


def test_shallow_swings_below_min_amp_earn_nothing():
    # only one foot per set alternates (the other two of each set stay planted): support
    # difference swings by ~1/3 * EMA transmission, well under a min_amp that full sets clear
    shallow_a = [True, True, True, False, False, True]   # FL down, MR/RL up... A count 1
    shallow_b = [False, True, True, True, True, True]
    seq = (([shallow_a] * 8) + ([shallow_b] * 8)) * 20
    rewards, _ = _run(seq, min_amp=0.5)
    assert sum(float(r) for r in rewards) == 0.0


def test_deeper_alternation_pays_more_per_crossing():
    def partial(n_feet):
        a_on = [j for j in TRIPOD_A_IDX][:n_feet]
        b_on = [j for j in TRIPOD_B_IDX][:n_feet]
        pa = [j in a_on for j in range(6)]
        pb = [j in b_on for j in range(6)]
        return (([pa] * 8) + ([pb] * 8)) * 20

    full_paid = [float(r) for r in _run(partial(3))[0] if float(r) > 0]
    two_paid = [float(r) for r in _run(partial(2))[0] if float(r) > 0]
    assert full_paid and two_paid
    assert max(full_paid) > max(two_paid)


def test_first_crossing_after_reset_is_unpaid():
    # half a cycle of A-stance then swap to B: the crossing happens with t_since at
    # RESET_T_SINCE-ish -- way out of band -- so it must not pay
    seq = ([A_PLANTED] * 8) + ([B_PLANTED] * 8)
    rewards, _ = _run(seq)
    assert sum(float(r) for r in rewards) == 0.0


def test_wrapper_style_reset_reproduces_fresh_behavior():
    seq = ideal_alternation(n_cycles=6)
    rewards_fresh, state = _run(seq)
    # wrapper reset(): zero the row, then set S_T_SINCE = RESET_T_SINCE
    state[:] = 0.0
    state[:, S_T_SINCE] = RESET_T_SINCE
    rewards_again, _ = _run(seq, state=state)
    assert torch.allclose(
        torch.stack([r for r in rewards_fresh]), torch.stack([r for r in rewards_again])
    )


def test_no_reward_when_command_inactive():
    rewards, _ = _run(ideal_alternation(n_cycles=20), cmd=(0.0, 0.0))
    assert sum(float(r) for r in rewards) == 0.0


def test_batched_envs_are_independent():
    seq = ideal_alternation(n_cycles=20)
    single_rewards, _ = _run(seq)

    state = torch.zeros(2, STATE_DIM)
    state[:, S_T_SINCE] = RESET_T_SINCE
    cmd = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    batched = []
    for contact in seq:
        c0 = torch.tensor([contact], dtype=torch.bool)
        c1 = torch.tensor([ALL_PLANTED], dtype=torch.bool)
        r, state = tripod_swap_crossing_reward_step(
            torch.cat([c0, c1]), cmd, state, DT
        )
        batched.append(r)
    env0 = torch.stack([r[0] for r in batched])
    env1 = torch.stack([r[1] for r in batched])
    assert torch.allclose(env0, torch.stack(single_rewards).squeeze(1))
    assert float(env1.sum()) == 0.0
