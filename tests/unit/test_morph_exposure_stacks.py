"""Morphology x exposure campaign: env stacks, promotion scaling, plant resolution (stdlib + the
two campaign modules loaded by path; no Isaac)."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ORCH = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments" / "2026-09-04_1105_morph_x_exposure" / "run_morph_exposure.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("run_morph_exposure", ORCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPromotionScaling:
    def test_identity_at_reference_horizon(self, m):
        assert m.promote_fracs_for_horizon(0.45, 0.25, 20.0) == "0.45:0.25"

    def test_scales_by_ref_over_T(self, m):
        assert m.promote_fracs_for_horizon(0.45, 0.25, 40.0) == "0.225:0.125"
        assert m.promote_fracs_for_horizon(0.45, 0.25, 70.0).startswith("0.1286:")

    def test_rejects_bad_horizons(self, m):
        with pytest.raises(ValueError):
            m.promote_fracs_for_horizon(0.45, 0.25, 0.0)


class TestStacks:
    def test_p1_is_rung_v_config_plus_stand_frac(self, m):
        for cfg in m.ORDER:
            ev = m.p1_stack(cfg)
            assert ev["KRABBY_STAND_FRAC"] == "0.2"
            core = {k: v for k, v in ev.items() if k not in ("KRABBY_STAND_FRAC", "KRABBY_HEX_USD_PATH")}
            assert core == m.FORMATION  # exactly the rung-v formation config of record
            assert "KRABBY_EPISODE_S" not in ev and "KRABBY_FLAT_TERRAIN_CURRICULUM" not in ev

    def test_p2_stage1_adds_only_the_horizon(self, m):
        ev1, ev2 = m.p1_stack("A15"), m.p2_stage1_stack("A15")
        assert {k: v for k, v in ev2.items() if k not in m.LONG} == ev1
        assert ev2["KRABBY_EPISODE_S"] == "40" and ev2["KRABBY_RESAMPLE_S"] == "10:10"

    def test_p2_stage2_holds_the_equilibrium_on_widened_geometry(self, m):
        ev = m.p2_stage2_stack("A20+B")
        assert ev["KRABBY_FLAT_TERRAIN_GEOM"] == "recal2b2w"
        assert ev["KRABBY_TERRAIN_PROMOTE"] == "0.225:0.125"
        assert ev["KRABBY_FLAT_TERRAIN_CURRICULUM"] == "1"
        assert ev["KRABBY_EPISODE_S"] == "40" and ev["KRABBY_STAND_FRAC"] == "0.2"
        # window-1 elements present, gait weights at their window-1 values
        for k in ("KRABBY_YAW_W", "KRABBY_DR_PUSH", "KRABBY_EDGE_W", "KRABBY_STUMBLE_W", "KRABBY_COLLISION_W"):
            assert k in ev
        assert ev["KRABBY_CLOCK_W"] == "1.0" and ev["KRABBY_APEX_W"] == "1.0"
        assert ev["KRABBY_RSI_BANK"].endswith("rsi_bank_P0_null.npz")


class TestPlants:
    def test_base_is_the_explicit_legacy_golden_and_variants_resolve(self, m):
        # since 2026-09-09 the config default is the A15+B main asset, so "base" must be explicit
        base = m.p1_stack("base")["KRABBY_HEX_USD_PATH"]
        assert base.endswith("variants/crab_simple__splay00_axis5p5in.usda") and Path(base).exists()
        for cfg in m.ORDER:
            if cfg == "base":
                continue
            p = m.p1_stack(cfg)["KRABBY_HEX_USD_PATH"]
            assert Path(p).exists(), p
            assert Path(p).name == f"crab_simple__{m.CONFIGS[cfg]}.usda"

    def test_with_plant_replaces_a_stale_override(self, m):
        ev = m.with_plant({"KRABBY_HEX_USD_PATH": "/stale.usda", "X": "1"}, "A10")
        assert ev["KRABBY_HEX_USD_PATH"].endswith("splay10_axis5p5in.usda") and ev["X"] == "1"
        assert m.with_plant({"KRABBY_HEX_USD_PATH": "/stale.usda"}, "base")["KRABBY_HEX_USD_PATH"].endswith("variants/crab_simple__splay00_axis5p5in.usda")


class TestRanking:
    def test_default_top_uses_completion_times_survival_and_prefers_splay(self, m):
        p1 = {
            "base": {"ckpt": "x", "evals": {"slow": {"completion": 0.9}, "step": {"falls": 30, "n": 100}}},
            "A15": {"ckpt": "x", "evals": {"slow": {"completion": 0.8}, "step": {"falls": 10, "n": 100}}},   # 0.72
            "A20+B": {"ckpt": "x", "evals": {"slow": {"completion": 0.8}, "step": {"falls": 10, "n": 100}}}, # 0.72 tie -> splay first
            "B": {"ckpt": "x", "evals": {"slow": {"completion": 0.9}, "step": {"falls": 40, "n": 100}}},     # 0.54
            "A10": {"ckpt": None, "status": "dead_plant", "evals": {}},                                     # unsound
            "A20": {"ckpt": "x", "evals": {"slow": {"completion": 0.5}, "step": {"falls": 5, "n": 100}}},   # 0.475
        }
        assert m.default_top(p1) == ["A15", "A20+B", "B"]

    def test_corrected_stand_frac(self, m):
        assert m.corrected_stand_frac(0.444, 700.0, 20.0) == pytest.approx(0.634, abs=0.01)
        assert m.corrected_stand_frac(0.257, 1534.0, 70.0) == pytest.approx(0.586, abs=0.01)
