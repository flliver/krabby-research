"""PLAN H: env-var knob parsing, corridor bounds, and the widened preset application."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MDP_DIR = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

import exposure_knobs as xk  # noqa: E402


def _sub_terrains():
    return {
        "parkour_flat": SimpleNamespace(),
        "parkour_gap": SimpleNamespace(half_valid_width=(0.85, 1.15)),
        "parkour_hurdle": SimpleNamespace(half_valid_width=(0.4, 0.8)),
        "parkour_step": SimpleNamespace(half_valid_width=(0.5, 1.0)),
        "parkour": SimpleNamespace(stone_width=1.0),
    }


class TestCorridor:
    def test_widened_preset_clears_half_stance_and_keeps_trench(self):
        lo, hi = xk.RECAL2B2W_HALF_VALID_WIDTH
        max_off = max(abs(v) for v in xk.RECAL2B2W_Y_RANGE)
        # blind to the corridor offset: narrowest corridor must cover half-stance + |offset|
        assert lo >= xk.CRAB_HALF_STANCE_M + max_off
        # at least one side trench survives the widest corridor at the largest offset
        usable_half = 3.92 / 2
        assert usable_half - (hi - max_off) >= 0.2
        assert lo > xk.CRAB_HALF_STANCE_M + 0.2
        assert hi <= xk.CORRIDOR_HALF_WIDTH_MAX_M
        assert xk.RECAL2B2W_STONE_WIDTH <= xk.STONE_WIDTH_MAX_M
        assert xk.RECAL2B2W_STONE_WIDTH > 2 * xk.CRAB_HALF_STANCE_M

    def test_apply_sets_every_corridor_terrain_and_stones(self):
        st = _sub_terrains()
        applied = xk.apply_corridor_widths(st, xk.RECAL2B2W_HALF_VALID_WIDTH, xk.RECAL2B2W_STONE_WIDTH)
        for key in xk.CORRIDOR_TERRAINS:
            assert st[key].half_valid_width == xk.RECAL2B2W_HALF_VALID_WIDTH
            assert applied[key] == xk.RECAL2B2W_HALF_VALID_WIDTH
        assert st["parkour"].stone_width == xk.RECAL2B2W_STONE_WIDTH
        assert not hasattr(st["parkour_flat"], "half_valid_width")

    def test_preset_offsets_apply_and_are_bounded(self):
        st = _sub_terrains()
        applied = xk.apply_corridor_widths(st, xk.RECAL2B2W_HALF_VALID_WIDTH, xk.RECAL2B2W_STONE_WIDTH,
                                           y_range=xk.RECAL2B2W_Y_RANGE, stone_y_range=xk.RECAL2B2W_STONE_Y_RANGE)
        for key in xk.CORRIDOR_TERRAINS:
            assert st[key].y_range == xk.RECAL2B2W_Y_RANGE
        assert st["parkour"].y_range == xk.RECAL2B2W_STONE_Y_RANGE
        assert applied["parkour_gap.y_range"] == xk.RECAL2B2W_Y_RANGE
        with pytest.raises(ValueError):
            xk.apply_corridor_widths(_sub_terrains(), None, None, y_range=(-0.5, 0.5))
        # env overrides never touch offsets
        st2 = _sub_terrains()
        xk.apply_corridor_widths(st2, (1.5, 1.8), None)
        assert not hasattr(st2["parkour_gap"], "y_range")

    def test_apply_none_is_noop(self):
        st = _sub_terrains()
        assert xk.apply_corridor_widths(st, None, None) == {}
        assert st["parkour_gap"].half_valid_width == (0.85, 1.15)
        assert st["parkour"].stone_width == 1.0

    def test_tile_width_bound_asserted(self):
        with pytest.raises(ValueError):
            xk.apply_corridor_widths(_sub_terrains(), (1.5, 1.9), None)
        with pytest.raises(ValueError):
            xk.apply_corridor_widths(_sub_terrains(), None, 3.8)
        with pytest.raises(ValueError):
            xk.check_corridor_half_width(1.7, 1.4)

    def test_env_overrides(self):
        half, stone = xk.corridor_overrides_from_env({"KRABBY_CORRIDOR_HALF_WIDTH": "1.5:1.8", "KRABBY_STONE_WIDTH": "3.0"})
        assert half == (1.5, 1.8) and stone == 3.0
        assert xk.corridor_overrides_from_env({}) == (None, None)
        with pytest.raises(ValueError):
            xk.corridor_overrides_from_env({"KRABBY_CORRIDOR_HALF_WIDTH": "1.5:2.5"})


class TestKnobParsers:
    def test_stand_frac(self):
        assert xk.parse_stand_frac("0.2") == 0.2
        for bad in ("1.0", "-0.1"):
            with pytest.raises(ValueError):
                xk.parse_stand_frac(bad)

    def test_spawn_offset_bounds_keep_spawn_on_platform(self):
        assert xk.parse_spawn_offset("2.0") == 2.0
        # tile-local x = 4.0 - offset must stay <= 2.48 (platform edge) -> offset >= 1.6
        assert 4.0 - xk.SPAWN_OFFSET_MIN_M <= 2.48
        with pytest.raises(ValueError):
            xk.parse_spawn_offset("1.0")
        with pytest.raises(ValueError):
            xk.parse_spawn_offset("4.0")

    def test_spawn_spread(self):
        assert xk.parse_spawn_spread("1.0:11.0:0.5") == (1.0, 11.0, 0.5)
        assert xk.parse_spawn_spread("2.0:9.0") == (2.0, 9.0, xk.SPREAD_FRAC_DEFAULT)
        for bad in ("0.5:11.0", "11.0:1.0", "1.0:16.0", "1.0:11.0:0.0", "1.0:11.0:1.5", "1.0"):
            with pytest.raises(ValueError):
                xk.parse_spawn_spread(bad)

    def test_truthy(self):
        assert xk.truthy("1") and xk.truthy("true") and xk.truthy(" YES ")
        assert not xk.truthy("") and not xk.truthy(None) and not xk.truthy("0")
