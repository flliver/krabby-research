"""Unit tests for the clock-referenced contact-schedule reward (gait-formation-v2 Phase 1).

Pure numpy/torch -- no Isaac Sim. The design contract under test: the schedule pays the
alternating tripod from step 0, no holdable state earns full income while the clock runs,
and the stop-command mode rewards exactly quiet standing.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from crab_hex_clock_reward import (  # noqa: E402
    FOOT_OFFSETS,
    FOOT_ORDER,
    TRIPOD_A,
    clock_schedule_income,
    clock_swing_apex_income,
)

RUN = torch.tensor([True])
STOP = torch.tensor([False])
STANCE_F = 700.0  # typical per-foot stance load


def _ideal_forces(phase: float) -> torch.Tensor:
    """Force pattern of a perfect tripod at the given clock phase: swing set unloaded."""
    f = torch.zeros(1, 6)
    for i, off in enumerate(FOOT_OFFSETS):
        swinging = math.sin(phase + off) > 0.0
        f[0, i] = 0.0 if swinging else STANCE_F
    return f


def test_perfect_tripod_earns_near_one_all_cycle():
    # 0.85 not 1.0: near window boundaries the smooth (probabilistic) edges hold back a
    # little income against a binary contact pattern -- Siekmann-style smoothing, intended.
    for phase in (0.3, 1.2, 2.0, 3.5, 4.4, 5.9):
        income = clock_schedule_income(
            torch.tensor([phase]), _ideal_forces(phase), torch.zeros(1, 6), RUN
        )
        assert income.item() > 0.85, f"phase {phase}: {income.item()}"
    # dead-center of the windows the income is essentially full
    mid = clock_schedule_income(
        torch.tensor([math.pi / 2]), _ideal_forces(math.pi / 2), torch.zeros(1, 6), RUN
    )
    assert mid.item() > 0.97


def test_standing_earns_half_while_clock_runs():
    """All six feet planted and still: full stance income, zero swing income -> ~0.5.
    This is the anti-creep property -- standing can never match a walking gait."""
    forces = torch.full((1, 6), STANCE_F)
    incomes = [
        clock_schedule_income(torch.tensor([p]), forces, torch.zeros(1, 6), RUN).item()
        for p in (0.5, 1.5, 2.5, 4.0, 5.5)
    ]
    assert all(0.4 < v < 0.6 for v in incomes), incomes


def test_wrong_handed_tripod_earns_near_zero():
    """The anti-phase error: planted exactly when told to swing."""
    for phase in (0.4, 2.2, 3.9):
        wrong = _ideal_forces(phase + math.pi)
        # feet that are planted (wrongly) also slide -- give them speed
        speeds = torch.where(wrong > 0, torch.full_like(wrong, 0.5), torch.zeros_like(wrong))
        income = clock_schedule_income(torch.tensor([phase]), wrong, speeds, RUN)
        assert income.item() < 0.15, f"phase {phase}: {income.item()}"


def test_stop_command_pays_quiet_standing_fully():
    income = clock_schedule_income(
        torch.tensor([1.0]), torch.full((1, 6), STANCE_F), torch.zeros(1, 6), STOP
    )
    assert income.item() > 0.99
    # ...and pays a drifting stand less
    drifting = clock_schedule_income(
        torch.tensor([1.0]), torch.full((1, 6), STANCE_F), torch.full((1, 6), 0.15), STOP
    )
    assert drifting.item() < income.item() - 0.3


def test_sliding_stance_is_penalized():
    for phase in (0.8, 4.0):
        still = clock_schedule_income(
            torch.tensor([phase]), _ideal_forces(phase), torch.zeros(1, 6), RUN
        )
        slide = torch.full((1, 6), 0.3)
        sliding = clock_schedule_income(
            torch.tensor([phase]), _ideal_forces(phase), slide, RUN
        )
        assert sliding.item() < still.item() - 0.3


def test_hovering_still_feet_earn_no_stance_income():
    """The farm the load gate kills: unloaded feet held motionless during their stance
    window must earn ~nothing."""
    phase = 1.0
    income = clock_schedule_income(
        torch.tensor([phase]), torch.zeros(1, 6), torch.zeros(1, 6), RUN
    )
    # swing-window feet legitimately earn (unloaded is correct there) -> about half.
    assert 0.4 < income.item() < 0.6
    stop = clock_schedule_income(
        torch.tensor([phase]), torch.zeros(1, 6), torch.zeros(1, 6), STOP
    )
    assert stop.item() < 0.05  # all-stance mode: hovering earns nothing at all


def test_offsets_encode_alternating_tripod():
    assert len(FOOT_OFFSETS) == 6
    for leg, off in zip(FOOT_ORDER, FOOT_OFFSETS):
        expected = 0.0 if leg in TRIPOD_A else math.pi
        assert off == pytest.approx(expected)


def test_product_mode_zeroes_the_creep_ceiling():
    """The 2026-08-22 basin-lottery fix: additive pays a creeper its stance half (~0.5);
    product multiplies by the ~0 swing quality, zeroing the creep income while walking
    keeps ~0.8+ -- the differential that makes the gait basin dominant."""
    phase = torch.tensor([1.0])
    creep_forces = torch.full((1, 6), STANCE_F)  # all planted, never swings
    creep_sum = clock_schedule_income(phase, creep_forces, torch.zeros(1, 6), RUN)
    creep_prod = clock_schedule_income(
        phase, creep_forces, torch.zeros(1, 6), RUN, combine="product"
    )
    assert 0.4 < creep_sum.item() < 0.6      # the additive ceiling
    assert creep_prod.item() < 0.05          # the product kills it
    walk_prod = clock_schedule_income(
        phase, _ideal_forces(1.0), torch.zeros(1, 6), RUN, combine="product"
    )
    assert walk_prod.item() > 0.75
    # stop-command semantics unchanged: quiet loaded standing earns fully in both modes
    stop_prod = clock_schedule_income(
        phase, creep_forces, torch.zeros(1, 6), STOP, combine="product"
    )
    assert stop_prod.item() > 0.99


def test_product_mode_gradient_alive_at_creep():
    """Unloading one foot inside its swing window must raise product income from ~0."""
    phase = torch.tensor([math.pi / 2])  # FL/MR/RL mid-swing-window
    creep = torch.full((1, 6), STANCE_F)
    before = clock_schedule_income(phase, creep, torch.zeros(1, 6), RUN, combine="product")
    lifted = creep.clone()
    lifted[0, 0] = 0.0  # FL unloads during its swing window
    after = clock_schedule_income(phase, lifted, torch.zeros(1, 6), RUN, combine="product")
    assert after.item() > before.item() + 0.15


def test_apex_income_pays_commanded_height_at_mid_swing():
    """Specified swing apex: full pay at the commanded height, partial below with the
    gradient pointing up (the property income-priced clearance lacked), nothing on stops."""
    phase = torch.tensor([math.pi / 2])  # set A at dead mid-swing
    h = torch.zeros(1, 6)
    swing_feet = [i for i, leg in enumerate(FOOT_ORDER) if leg in TRIPOD_A]
    # at the commanded apex (0.08): ~full income
    h[0, swing_feet] = 0.08
    at_apex = clock_swing_apex_income(phase, h, torch.tensor([True]))
    assert at_apex.item() > 0.95
    # at the plant's old ceiling (0.044): partial, and raising height raises income
    h[0, swing_feet] = 0.044
    low = clock_swing_apex_income(phase, h, torch.tensor([True]))
    h[0, swing_feet] = 0.06
    mid = clock_swing_apex_income(phase, h, torch.tensor([True]))
    assert 0.1 < low.item() < mid.item() < at_apex.item()
    # stops schedule no swings
    stop = clock_swing_apex_income(phase, h, torch.tensor([False]))
    assert stop.item() == pytest.approx(0.0, abs=1e-6)
    # outside the mid-swing window (phase where set A is in stance) the term is silent
    # for those feet; with set B at ITS mid-swing the pay follows set B instead.
    phase_b = torch.tensor([3 * math.pi / 2])
    h2 = torch.zeros(1, 6)
    b_feet = [i for i, leg in enumerate(FOOT_ORDER) if leg not in TRIPOD_A]
    h2[0, b_feet] = 0.08
    assert clock_swing_apex_income(phase_b, h2, torch.tensor([True])).item() > 0.95


def test_mirror_consistency_pi_shift_swaps_sets():
    """The L/R mirrored gait = the same schedule advanced by pi: income must be identical
    for (phase, pattern) and (phase+pi, set-swapped pattern) -- the property that justifies
    the clock obs dims' mirror sign of -1."""
    phase = 0.7
    forces = _ideal_forces(phase)
    swap = [FOOT_ORDER.index({"FL": "FR", "FR": "FL", "ML": "MR", "MR": "ML",
                              "RL": "RR", "RR": "RL"}[leg]) for leg in FOOT_ORDER]
    a = clock_schedule_income(torch.tensor([phase]), forces, torch.zeros(1, 6), RUN)
    b = clock_schedule_income(
        torch.tensor([phase + math.pi]), forces[:, swap], torch.zeros(1, 6), RUN
    )
    assert a.item() == pytest.approx(b.item(), abs=1e-5)
