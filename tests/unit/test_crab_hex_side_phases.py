"""The sideways presets (``crab_hex_side_phases.py``) must be the forward schedule of record with
exactly three deltas -- task ids, experiment names, RSI off (+ the phase-3 student flag) -- and the
two preset variables must never collide.

Both preset modules are loaded BY FILE PATH under distinct ``sys.modules`` names, so nothing here
imports ``parkour_tasks`` (pinned by ``test_side_module_is_pure``). Pure Python, no Isaac.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task"
SIDE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task"
FWD_PHASES_PY = PKG / "config/crab_hex/crab_hex_phases.py"
SIDE_PHASES_PY = SIDE / "config/crab_hex/crab_hex_side_phases.py"

PHASE_NAMES = ("1a", "2a", "2b", "2c", "3a", "3b")
STUDENT_NAMES = ("3a", "3b")
RSI_KEYS = {"KRABBY_RSI_FRAC", "KRABBY_RSI_BANK"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fwd():
    return _load(FWD_PHASES_PY, "crab_hex_phases__fwd_test")


@pytest.fixture(scope="module")
def side():
    return _load(SIDE_PHASES_PY, "crab_hex_side_phases")


# ------------------------------------------------------------------------------ schedule shape
def test_phase_chain_and_tasks(side):
    assert set(side.PHASES) == set(PHASE_NAMES)
    assert not any("legacy" in n for n in side.PHASES)

    s = side.PHASES["1a"]
    assert s.task == "Isaac-Crab-Hex-Side-Walk-v0"
    assert s.experiment == "crab_hex_side_flat_walk"
    assert s.resume_from is None and s.teacher_mode is None and s.kind == "rl"

    for n, prev in (("2a", "1a"), ("2b", "2a"), ("2c", "2b")):
        s = side.PHASES[n]
        assert s.task == "Isaac-Crab-Hex-Side-Teacher-v0"
        assert s.experiment == "crab_hex_side_teacher"
        assert s.resume_from == prev and s.teacher_mode == n and s.kind == "rl"

    for n, prev in (("3a", "2c"), ("3b", "3a")):
        s = side.PHASES[n]
        assert s.task == "Isaac-Crab-Hex-Side-Student-v0"
        assert s.experiment == "crab_hex_side_student"
        assert s.resume_from == prev and s.teacher_mode is None and s.kind == "distill"

    assert side.PHASE_MODES == {"2a": "2a", "2b": "2b", "2c": "2c"}
    assert side.STUDENT_PHASES == STUDENT_NAMES


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_env_is_forward_env_minus_rsi(fwd, side, name):
    f, s = fwd.PHASES[name].env, side.PHASES[name].env
    assert set(f) - set(s) == RSI_KEYS
    extra = {"KRABBY_STUDENT_MDP"} if name in STUDENT_NAMES else set()
    assert set(s) - set(f) == extra
    for k in set(f) & set(s):
        assert s[k] == f[k], k
    assert side.PHASES[name].iterations == fwd.PHASES[name].iterations


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_phase_env_key_delta(fwd, side, name):
    f = fwd.phase_env(name, "A15+B")
    s = side.phase_env(name, "A15+B")
    assert set(f) - set(s) == {"KRABBY_PHASE"} | RSI_KEYS
    extra = {"KRABBY_STUDENT_MDP"} if name in STUDENT_NAMES else set()
    assert set(s) - set(f) == {"KRABBY_SIDE_PHASE"} | extra
    assert s["KRABBY_SIDE_PHASE"] == name
    assert "KRABBY_PHASE" not in s
    if name in ("2a", "2b", "2c"):
        assert s["KRABBY_HEX_TEACHER_MODE"] == f["KRABBY_HEX_TEACHER_MODE"] == name


def test_no_rsi_key_anywhere(side):
    for name in PHASE_NAMES:
        assert not RSI_KEYS & set(side.PHASES[name].env), name
        assert not RSI_KEYS & set(side.phase_env(name)), name
        assert not RSI_KEYS & set(side.phase_env(name, "A15+B")), name
        env = {"KRABBY_SIDE_PHASE": name, "KRABBY_PLANT": "A15+B"}
        applied = side.activate_phase(env)
        assert not RSI_KEYS & set(applied), name
        assert not RSI_KEYS & set(env), name


def test_student_phases_export_the_student_flag(side):
    for name in STUDENT_NAMES:
        assert side.PHASES[name].env["KRABBY_STUDENT_MDP"] == "1"
    assert "KRABBY_STUDENT_MDP" not in side.PHASES["2c"].env
    assert side.is_student_phase({"KRABBY_SIDE_PHASE": "3a"})
    assert side.is_student_phase({"KRABBY_STUDENT_MDP": "1"})
    assert not side.is_student_phase({"KRABBY_SIDE_PHASE": "2c"})
    assert not side.is_student_phase({})


# ------------------------------------------------------------------------------ activation
def test_activation_from_side_variable(side):
    env = {"KRABBY_SIDE_PHASE": "2b", "KRABBY_PLANT": "A15+B", "KRABBY_STAND_FRAC": "0.35"}
    applied = side.activate_phase(env)
    assert env["KRABBY_STAND_FRAC"] == "0.35" and "KRABBY_STAND_FRAC" not in applied
    assert env["KRABBY_HEX_TEACHER_MODE"] == "2b"
    assert "KRABBY_HEX_USD_PATH" not in env          # A15+B is the main asset: nothing to export
    assert env["KRABBY_FLAT_TERRAIN_GEOM"] == "recal2b2w" and env["KRABBY_EPISODE_S"] == "40"
    assert "KRABBY_PHASE" not in env                  # never written by the sideways package
    assert "KRABBY_RSI_FRAC" not in env


def test_forward_variable_alone_is_a_noop_for_side(side):
    env = {"KRABBY_PHASE": "2b"}
    assert side.activate_phase(env) == {}
    assert env == {"KRABBY_PHASE": "2b"}


def test_side_variable_alone_is_a_noop_for_forward(fwd):
    env = {"KRABBY_SIDE_PHASE": "2b"}
    assert fwd.activate_phase(env) == {}
    assert env == {"KRABBY_SIDE_PHASE": "2b"}


def test_both_variables_raise(side):
    with pytest.raises(RuntimeError, match="one task per process"):
        side.activate_phase({"KRABBY_PHASE": "2b", "KRABBY_SIDE_PHASE": "2b"})


def test_plant_only_exports_like_forward(side):
    env = {"KRABBY_PLANT": "A15+B"}
    assert side.activate_phase(env) == {}
    assert env == {"KRABBY_PLANT": "A15+B"}
    env = {"KRABBY_PLANT": "B"}
    side.activate_phase(env)
    assert env["KRABBY_HEX_USD_PATH"].endswith("crab_simple__splay00_axis2p5in.usda")


def test_unknown_side_phase_fails_loudly(side):
    for name in ("s9z", "legacy_golden_1a"):
        with pytest.raises(KeyError):
            side.phase_env(name)
        with pytest.raises(KeyError):
            side.activate_phase({"KRABBY_SIDE_PHASE": name})


# ------------------------------------------------------------------------------ shared tables
def test_side_module_is_pure():
    """Loading the side presets by path must not import ``parkour_tasks`` or ``isaaclab``.

    Checked in a FRESH interpreter: other unit tests in the same pytest process legitimately stub
    ``isaaclab.*`` into ``sys.modules`` (``test_crab_hex_rsi_spawn_fix``, the heading-offset test),
    so asserting on the shared ``sys.modules`` would report their stubs, not this module's imports.
    """
    import os
    import subprocess

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('crab_hex_side_phases', {str(SIDE_PHASES_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['crab_hex_side_phases'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.PHASES\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'isaaclab'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_plant_table_and_blocks_are_the_forward_ones(fwd, side):
    assert side.PLANTS == fwd.PLANTS
    assert side.MAIN_PLANT == "A15+B"
    assert side.FORMATION == fwd.FORMATION
    assert side.ELEMENTS_5K == fwd.ELEMENTS_5K and side.ELEMENTS_10K == fwd.ELEMENTS_10K
    assert side.WALKING_SLOTS == fwd.WALKING_SLOTS and side.LONG_EPISODES == fwd.LONG_EPISODES
    assert side.WIDENED_CORRIDORS == fwd.WIDENED_CORRIDORS
    assert side.GAIT_WEIGHTS == fwd.GAIT_WEIGHTS and side.GAIT_RAMPS == fwd.GAIT_RAMPS
    assert side.WINDOW_ITERS == fwd.WINDOW_ITERS and side.RAMP_ITERS == fwd.RAMP_ITERS
    assert side.STEPS_PER_ITER == fwd.STEPS_PER_ITER and side.EPS == fwd.EPS
    assert side._MDP_KEYS == fwd._MDP_KEYS
    assert side.MAIN_ASSET.name == fwd.MAIN_ASSET.name == "crab.usda"
    assert side.MAIN_ASSET_NAMES == fwd.MAIN_ASSET_NAMES
    assert side.plant_usd_path("B") == fwd.plant_usd_path("B")
    assert side.plant_name_for_path("/x/assets/crab.usda") == "A15+B"
    assert side.REPO_ROOT == REPO and side.EXPERIMENTS == SIDE / "experiments"
