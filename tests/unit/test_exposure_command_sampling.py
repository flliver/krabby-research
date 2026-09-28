"""PLAN H B1: pure command-slot sampler contracts (no Isaac).

- stand_frac=None reproduces today's small_commands_to_zero rule exactly on the same draws;
- armed: standing rate ~= p, walking slots never sub-clip and never at the clock stop
  threshold, walking band = [max(clip, lo) + 0.01, hi];
- invalid p / empty walking band fail loudly.
"""
import sys
from pathlib import Path

import pytest
import torch

MOD_DIR = Path(__file__).resolve().parents[2] / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkour_commands"
if str(MOD_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_DIR))

from command_sampling import WALK_MARGIN_M_S, sample_slot, walking_band  # noqa: E402

LO, HI, CLIP = 0.0, 0.35, 0.2


def _today_rule(u_vel):
    vx = LO + u_vel * (HI - LO)
    return vx * (vx.abs() > CLIP)


class TestUnarmedReference:
    def test_reproduces_clip_rule_exactly(self):
        g = torch.Generator().manual_seed(0)
        u = torch.rand(20000, generator=g)
        vx, stand = sample_slot(u, torch.zeros_like(u), None, LO, HI, CLIP)
        assert torch.equal(vx, _today_rule(u))
        assert torch.equal(stand, vx == 0.0)

    def test_implicit_stand_fraction_on_training_band(self):
        u = torch.linspace(0, 1, 100001)[:-1]
        _, stand = sample_slot(u, torch.zeros_like(u), None, LO, HI, CLIP)
        # |vx| <= 0.2 on U(0, 0.35) -> 0.2/0.35 = 57%
        assert stand.float().mean().item() == pytest.approx(0.2 / 0.35, abs=0.002)


class TestArmed:
    def test_stand_rate_matches_p(self):
        g = torch.Generator().manual_seed(1)
        u_vel, u_stand = torch.rand(200000, generator=g), torch.rand(200000, generator=g)
        vx, stand = sample_slot(u_vel, u_stand, 0.2, LO, HI, CLIP)
        assert stand.float().mean().item() == pytest.approx(0.2, abs=0.005)
        assert torch.equal(stand, vx == 0.0)

    def test_walking_slots_never_sub_clip(self):
        g = torch.Generator().manual_seed(2)
        u_vel, u_stand = torch.rand(100000, generator=g), torch.rand(100000, generator=g)
        vx, stand = sample_slot(u_vel, u_stand, 0.2, LO, HI, CLIP)
        walk = vx[~stand]
        assert walk.min().item() >= CLIP + WALK_MARGIN_M_S - 1e-6
        assert walk.max().item() <= HI + 1e-6
        assert (walk > CLIP).all()  # strictly above the gait-clock stop threshold

    def test_walking_band_uses_lo_when_above_clip(self):
        assert walking_band(0.2, 0.30, 0.65, 0.2) == pytest.approx((0.31, 0.65))
        assert walking_band(0.2, 0.0, 0.35, 0.2) == pytest.approx((0.21, 0.35))

    def test_p_zero_never_stands(self):
        u = torch.rand(1000)
        vx, stand = sample_slot(u, torch.rand(1000), 0.0, LO, HI, CLIP)
        assert not stand.any()
        assert (vx > CLIP).all()

    def test_invalid_p_raises(self):
        u = torch.rand(4)
        for p in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError):
                sample_slot(u, u, p, LO, HI, CLIP)

    def test_empty_walking_band_raises(self):
        u = torch.rand(4)
        with pytest.raises(ValueError):
            sample_slot(u, u, 0.2, 0.0, 0.2, 0.2)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            sample_slot(torch.rand(3), torch.rand(4), 0.2, LO, HI, CLIP)
