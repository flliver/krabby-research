"""Phase presets of record (crab_hex_phases.py) must equal the recorded campaign stacks bit for bit,
and activation must never override an explicitly set variable. Pure Python (no Isaac)."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PHASES_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/config/crab_hex/crab_hex_phases.py"
LINEAGE_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage/run_lineage.py"
EXPOSURE_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/run_exposure.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ph():
    return _load(PHASES_PY, "crab_hex_phases")


@pytest.fixture(scope="module")
def lineage():
    return _load(LINEAGE_PY, "run_lineage")


@pytest.fixture(scope="module")
def exposure():
    return _load(EXPOSURE_PY, "run_exposure")


def _strip_plant(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k != "KRABBY_HEX_USD_PATH"}


class TestScheduleOfRecord:
    @pytest.mark.parametrize("name,window", [("1a", 0), ("2a", 1), ("2b", 2), ("2c", 3)])
    def test_phase_equals_lineage_window(self, ph, lineage, name, window):
        # run_lineage.window_stack encodes the baked A15+B lineage (plant added separately)
        assert ph.PHASES[name].env == _strip_plant(lineage.window_stack(window))

    def test_phase_chain_and_tasks(self, ph):
        assert ph.PHASES["1a"].task == "Isaac-Crab-Hex-Flat-Walk-v0" and ph.PHASES["1a"].resume_from is None
        for n, prev, mode in (("2a", "1a", "2a"), ("2b", "2a", "2b"), ("2c", "2b", "2c")):
            s = ph.PHASES[n]
            assert s.task == "Isaac-Crab-Hex-Teacher-v0" and s.resume_from == prev and s.teacher_mode == mode
        assert ph.PHASES["3a"].resume_from == "2c" and ph.PHASES["3b"].resume_from == "3a"
        assert all(ph.PHASES[n].task == "Isaac-Crab-Hex-Student-v0" for n in ("3a", "3b"))

    def test_2c_holds_clock_at_half_and_no_later_anneal_in_paradigm(self, ph):
        assert ph.PHASES["2c"].env["KRABBY_PHASEOUT"].startswith("reward_clock_schedule:1.0:0.5:")
        assert not any(n.startswith("2d") or n.startswith("2e") for n in ph.PHASES if not n.startswith("legacy"))


class TestStudentPhases:
    def test_student_env_is_the_2c_mdp_without_rewards(self, ph):
        s3a, s2c = ph.PHASES["3a"].env, ph.PHASES["2c"].env
        for k in ph._MDP_KEYS:
            assert s3a.get(k) == s2c.get(k)
        for k in s3a:
            assert not k.endswith("_W") and k != "KRABBY_PHASEOUT", k
        assert "KRABBY_CLOCK_W" not in s3a and "KRABBY_PHASEOUT" not in s3a

    def test_3b_only_changes_the_difficulty_band(self, ph):
        a, b = dict(ph.PHASES["3a"].env), dict(ph.PHASES["3b"].env)
        assert b.pop("KRABBY_FLAT_TERRAIN_DIFF") == "0.70:0.90"
        a.pop("KRABBY_FLAT_TERRAIN_DIFF")
        assert a == b
        assert ph.is_student_phase({"KRABBY_PHASE": "3a"}) and not ph.is_student_phase({"KRABBY_PHASE": "2c"})


class TestLegacyGolden:
    @pytest.mark.parametrize("name,window,weights", [("legacy_golden_1a", 0, None), ("legacy_golden_2a", 1, None),
                                                     ("legacy_golden_2b", 2, None), ("legacy_golden_2c", 3, None),
                                                     ("legacy_golden_2d", 4, None), ("legacy_golden_2e", 5, None)])
    def test_legacy_equals_exposure_lineage_stack(self, ph, exposure, name, window, weights):
        if window == 0:
            expect = exposure.lineage_stack(0, {"KRABBY_CLOCK_W": "1.0", "KRABBY_APEX_W": "1.0"}, exposure.BANKS[1])
        else:
            w, ramps = exposure.REPLAY_SCHEDULE[window]
            expect = exposure.lineage_stack(window, w, exposure.BANKS[window])
            expect["KRABBY_PHASEOUT"] = exposure.phaseout_spec(ramps)
        assert ph.PHASES[name].env == expect


class TestActivation:
    def test_setdefault_never_overrides_explicit_values(self, ph):
        env = {"KRABBY_PHASE": "2b", "KRABBY_PLANT": "A15+B", "KRABBY_STAND_FRAC": "0.35"}
        applied = ph.activate_phase(env)
        assert env["KRABBY_STAND_FRAC"] == "0.35" and "KRABBY_STAND_FRAC" not in applied
        assert env["KRABBY_HEX_TEACHER_MODE"] == "2b"
        assert "KRABBY_HEX_USD_PATH" not in env  # A15+B is the main asset: nothing to export
        assert env["KRABBY_FLAT_TERRAIN_GEOM"] == "recal2b2w" and env["KRABBY_EPISODE_S"] == "40"

    def test_main_plant_exports_nothing_and_legacy_golden_is_explicit(self, ph):
        assert ph.MAIN_PLANT == "A15+B" and ph.PLANTS["A15+B"] is None and ph.PLANTS["main"] is None
        env = {"KRABBY_PLANT": "A15+B"}
        ph.activate_phase(env)
        assert "KRABBY_HEX_USD_PATH" not in env and "KRABBY_HEX_TEACHER_MODE" not in env
        for name in ("legacy_golden", "golden"):
            env = {"KRABBY_PHASE": "1a", "KRABBY_PLANT": name}
            ph.activate_phase(env)
            assert env["KRABBY_HEX_USD_PATH"].endswith("variants/crab_simple__splay00_axis5p5in.usda")
            assert "KRABBY_HEX_TEACHER_MODE" not in env
        env = {"KRABBY_PLANT": "B"}
        ph.activate_phase(env)
        assert env["KRABBY_HEX_USD_PATH"].endswith("variants/crab_simple__splay00_axis2p5in.usda")

    def test_plant_name_for_path(self, ph):
        assert ph.plant_name_for_path("/x/assets/crab.usda") == "A15+B"
        assert ph.plant_name_for_path("/x/assets/variants/crab_simple__splay15_axis2p5in.usda") == "A15+B"
        assert ph.plant_name_for_path("/x/assets/variants/crab_simple__splay00_axis5p5in.usda") == "legacy_golden"
        assert ph.plant_name_for_path("/x/assets/crab_simple.usda") is None  # hand-authored historical model, not a plant
        assert ph.plant_name_for_path("/x/assets/variants/crab_simple__splay20_axis2p5in.usda") == "A20+B"
        assert ph.plant_name_for_path("/x/other.usda") is None and ph.plant_name_for_path(None) is None

    def test_plant_table_matches_generator_variants(self, ph):
        gen = _load(REPO / "assets/scripts/generate_crab.py", "generate_crab_for_test")
        table = {n: Path(p).name for n, p in ph.PLANTS.items() if p is not None and n != "golden"}
        expected = {n: gen.variant_asset_path(n).name for n in gen.VARIANTS if n != "A15+B"}
        assert table == expected
        assert gen.MAIN_ASSET.name == ph.MAIN_ASSET.name == "crab.usda"

    def test_no_phase_is_a_noop(self, ph):
        env = {"KRABBY_STAND_FRAC": "0.2"}
        assert ph.activate_phase(env) == {} and env == {"KRABBY_STAND_FRAC": "0.2"}

    def test_unknown_phase_or_plant_fails_loudly(self, ph):
        with pytest.raises(KeyError):
            ph.activate_phase({"KRABBY_PHASE": "9z"})
        with pytest.raises(KeyError):
            ph.plant_usd_path("nope")

    def test_plant_table_resolves_to_existing_assets(self, ph):
        for name in ph.PLANTS:
            p = ph.plant_usd_path(name)
            assert p is None or Path(p).exists(), p
        assert Path(ph.FORMATION["KRABBY_RSI_BANK"]).exists()
        for b in ph.LEGACY_GOLDEN_BANKS.values():
            assert Path(b).exists(), b


class TestStudentMdpFlag:
    def test_flag_selects_the_phase3_structure_without_a_preset(self, ph):
        assert ph.is_student_phase({"KRABBY_STUDENT_MDP": "1"})
        assert ph.is_student_phase({"KRABBY_STUDENT_MDP": "phase"})
        assert not ph.is_student_phase({"KRABBY_STUDENT_MDP": "0"})
        assert not ph.is_student_phase({})
        assert ph.is_student_phase({"KRABBY_PHASE": "3b"})
        assert not ph.is_student_phase({"KRABBY_PHASE": "2c"})
