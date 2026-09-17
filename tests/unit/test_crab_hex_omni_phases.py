"""The omni presets (``crab_hex_omni_phases.py``) must be the forward windows with exactly the
plan's key deltas, and the three preset variables must never collide.

Twin of ``test_crab_hex_side_phases.py``. All three preset modules are loaded BY FILE PATH under
distinct ``sys.modules`` names, so nothing here imports ``parkour_tasks`` (pinned by
``test_omni_module_is_pure``). Pure Python, no Isaac.

The second half pins the ``KRABBY_OMNI_*`` knob PARSERS (``config/crab_hex/omni_knobs.py``),
which are the other pure half of this step: they run at config build inside Isaac, where a
wrong band or a silently truncated tuple is invisible until a campaign has burned hours.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task"
SIDE = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task"
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
FWD_PHASES_PY = PKG / "config/crab_hex/crab_hex_phases.py"
SIDE_PHASES_PY = SIDE / "config/crab_hex/crab_hex_side_phases.py"
OMNI_PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"

PHASE_NAMES = ("1a", "1b", "1c", "2a", "2b", "2c", "2c_clockheld")
#: omni stage -> the forward phase whose window env it is built from
FORWARD_OF = {"1a": "1a", "1b": "1a", "1c": "1a", "2a": "2a", "2b": "2b", "2c": "2c",
              "2c_clockheld": "2c"}

WALK = "Isaac-Crab-Hex-Omni-Walk-v0"
TEACHER = "Isaac-Crab-Hex-Omni-Teacher-v0"

# ---------------------------------------------------------------- the expected env key delta
#: Re-pointed in every window (same key, different value).
REPOINTED = {"KRABBY_RSI_BANK", "KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_TERRAIN_PROMOTE"}
#: The phase-independent omni block (added to every window). ``KRABBY_YAW_PROGRESS_W`` is a
#: reward dose rather than a sampler field, but it is armed in EVERY window (1a/1b draw no
#: turning slot, so nothing there commands a turn for it to pay), so it belongs to the common
#: block. ``KRABBY_OMNI_FACING_OBS_CLIP`` (Revision 3, 2026-09-17) is likewise armed everywhere
#: at the same 0.6 rad -- the facing band ramps per phase, but the observation clip does not.
OMNI_COMMON_KEYS = {"KRABBY_OMNI_SPEED_ELLIPSE", "KRABBY_OMNI_ANG_VEL_MAX",
                    "KRABBY_OMNI_CLOCK_TURN_R", "KRABBY_OMNI_YAW_RANGE",
                    "KRABBY_YAW_PROGRESS_W", "KRABBY_OMNI_FACING_OBS_CLIP"}

#: phase -> (forward-only keys, omni-only keys, keys present in both with a different value).
#: Window 0 carries neither ``KRABBY_FLAT_TERRAIN_GEOM`` nor ``KRABBY_TERRAIN_PROMOTE``, so for
#: the three phase-1 stages those two are ADDITIONS, not re-points.
_STAGE1_ADDED = OMNI_COMMON_KEYS | {"KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_TERRAIN_PROMOTE",
                                    "KRABBY_OMNI_SLOTS", "KRABBY_OMNI_RSI_SOURCE"}
_TEACHER_ADDED = OMNI_COMMON_KEYS | {"KRABBY_OMNI_SLOTS", "KRABBY_OMNI_RSI_SOURCE",
                                     "KRABBY_OMNI_FACING"}
EXPECTED_DELTA: dict[str, tuple[set, set, set]] = {
    "1a": (set(), _STAGE1_ADDED | {"KRABBY_OMNI_DIR_BAND"}, {"KRABBY_RSI_BANK"}),
    "1b": (set(), _STAGE1_ADDED | {"KRABBY_OMNI_DIR_BAND"}, {"KRABBY_RSI_BANK"}),
    "1c": (set(), _STAGE1_ADDED | {"KRABBY_OMNI_FACING", "KRABBY_YAW_W"}, {"KRABBY_RSI_BANK"}),
    "2a": (set(), _TEACHER_ADDED, REPOINTED),
    "2b": ({"KRABBY_HEADING"}, _TEACHER_ADDED, REPOINTED),
    "2c": ({"KRABBY_HEADING"}, _TEACHER_ADDED, REPOINTED),
    "2c_clockheld": ({"KRABBY_HEADING", "KRABBY_PHASEOUT"}, _TEACHER_ADDED, REPOINTED),
}

#: phase -> the per-stage sampler values (``None`` = the key must be absent).
EXPECTED_STAGE: dict[str, dict[str, str | None]] = {
    "1a": {"KRABBY_OMNI_DIR_BAND": "0:0", "KRABBY_OMNI_SLOTS": "0.2:0.8:0:0",
           "KRABBY_OMNI_RSI_SOURCE": "0", "KRABBY_OMNI_FACING": None, "KRABBY_YAW_W": None},
    "1b": {"KRABBY_OMNI_DIR_BAND": "1.5708:1.5708", "KRABBY_OMNI_SLOTS": "0.2:0.8:0:0",
           "KRABBY_OMNI_RSI_SOURCE": "1", "KRABBY_OMNI_FACING": None, "KRABBY_YAW_W": None},
    "1c": {"KRABBY_OMNI_DIR_BAND": None, "KRABBY_OMNI_SLOTS": "0.15:0.55:0.10:0.20",
           "KRABBY_OMNI_RSI_SOURCE": "all", "KRABBY_OMNI_FACING": "0.2:0.6",
           "KRABBY_YAW_W": "0.2"},
    "2a": {"KRABBY_OMNI_DIR_BAND": None, "KRABBY_OMNI_SLOTS": "0.15:0.45:0.15:0.25",
           "KRABBY_OMNI_RSI_SOURCE": "all", "KRABBY_OMNI_FACING": "0.2:1.2",
           "KRABBY_YAW_W": "0.2"},
    "2b": {"KRABBY_OMNI_DIR_BAND": None, "KRABBY_OMNI_SLOTS": "0.15:0.45:0.15:0.25",
           "KRABBY_OMNI_RSI_SOURCE": "all", "KRABBY_OMNI_FACING": "0.2:2.4",
           "KRABBY_YAW_W": "0.2"},
    "2c": {"KRABBY_OMNI_DIR_BAND": None, "KRABBY_OMNI_SLOTS": "0.15:0.45:0.15:0.25",
           "KRABBY_OMNI_RSI_SOURCE": "all", "KRABBY_OMNI_FACING": "0.2:2.4",
           "KRABBY_YAW_W": "0.2"},
    "2c_clockheld": {"KRABBY_OMNI_DIR_BAND": None, "KRABBY_OMNI_SLOTS": "0.15:0.45:0.15:0.25",
                     "KRABBY_OMNI_RSI_SOURCE": "all", "KRABBY_OMNI_FACING": "0.2:2.4",
                     "KRABBY_YAW_W": "0.2"},
}

#: The omni block that is identical in every window (plan D11).
EXPECTED_COMMON = {
    "KRABBY_FLAT_TERRAIN_GEOM": "ring",
    "KRABBY_TERRAIN_PROMOTE": "0.45:0.25",
    "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.25:0.25",
    "KRABBY_OMNI_ANG_VEL_MAX": "0.3",
    "KRABBY_OMNI_CLOCK_TURN_R": "1.2",
    "KRABBY_OMNI_YAW_RANGE": "-3.14159:3.14159",
    # the added rotational-progress term, armed in every window (1a/1b command no turn)
    "KRABBY_YAW_PROGRESS_W": "1.0",
    # Revision 3 (2026-09-17): the facing-error observation clip, fixed at the 1c band's edge
    # in every window regardless of the facing band's own per-phase ramp (0.6 -> 1.2 -> 2.4).
    "KRABBY_OMNI_FACING_OBS_CLIP": "0.6",
}


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


@pytest.fixture(scope="module")
def omni():
    return _load(OMNI_PHASES_PY, "crab_hex_omni_phases")


@pytest.fixture(scope="module")
def knobs():
    return _load(OMNI_KNOBS_PY, "crab_hex_omni_knobs")


# ------------------------------------------------------------------------------ schedule shape
def test_phase_chain_and_tasks(omni):
    assert set(omni.PHASES) == set(PHASE_NAMES)
    assert not any("legacy" in n for n in omni.PHASES)
    assert omni.STUDENT_PHASES == ()
    assert not any(s.kind == "distill" for s in omni.PHASES.values()), "the omni plan has no student"
    assert "3a" not in omni.PHASES and "3b" not in omni.PHASES

    # 1a -> 1b -> 1c: the three phase-1 stages, all in the Omni-Walk task, off window 0
    for name, prev, iters in (("1a", None, 1000), ("1b", "1a", 1000), ("1c", "1b", 5000)):
        s = omni.PHASES[name]
        assert s.name == name
        assert s.task == WALK, name
        assert s.experiment == "crab_hex_omni_flat_walk", name
        assert s.resume_from == prev and s.teacher_mode is None and s.kind == "rl", name
        assert s.iterations == iters, name
        assert s.window == 0, name

    # 1c -> 2a -> 2b -> 2c, and the clock-held control off the 2b head
    for name, prev, mode in (("2a", "1c", "2a"), ("2b", "2a", "2b"), ("2c", "2b", "2c"),
                             ("2c_clockheld", "2b", "2c")):
        s = omni.PHASES[name]
        assert s.task == TEACHER, name
        assert s.experiment == "crab_hex_omni_teacher", name
        assert s.resume_from == prev and s.teacher_mode == mode and s.kind == "rl", name
        assert s.iterations == 5000, name

    assert omni.PHASE_MODES == {"2a": "2a", "2b": "2b", "2c": "2c", "2c_clockheld": "2c"}
    # the short stages are canary-only; every 5000-iteration window runs the full row set
    assert omni.PHASES["1a"].eval_scenarios == omni.PHASES["1b"].eval_scenarios == ("slow",)
    for name in ("1c", "2a", "2b", "2c", "2c_clockheld"):
        assert omni.PHASES[name].eval_scenarios == ("slow", "light", "ring"), name


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_env_is_the_forward_window_with_the_documented_deltas(fwd, omni, name):
    """The omni env is the forward window env, key for key, apart from the plan's delta."""
    f = fwd.PHASES[FORWARD_OF[name]].env
    o = omni.PHASES[name].env
    exp_fwd_only, exp_omni_only, exp_differ = EXPECTED_DELTA[name]

    assert set(f) - set(o) == exp_fwd_only, f"{name}: forward-only keys {sorted(set(f) - set(o))}"
    assert set(o) - set(f) == exp_omni_only, f"{name}: omni-only keys {sorted(set(o) - set(f))}"
    differ = {k for k in set(f) & set(o) if f[k] != o[k]}
    assert differ == exp_differ, f"{name}: differing values {sorted(differ)}"
    # everything else -- gait weights, phase-out ramps, tracking knobs, episode length,
    # resample, RSI fraction, DR, the element doses, terrain mode/fraction/difficulty -- is
    # carried through unchanged
    for k in sorted((set(f) & set(o)) - exp_differ):
        assert o[k] == f[k], f"{name}: {k} = {o[k]!r} (omni) != {f[k]!r} (forward)"


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_omni_common_block_in_every_window(omni, name):
    env = omni.PHASES[name].env
    for k, v in EXPECTED_COMMON.items():
        assert env[k] == v, f"{name}: {k}"
    assert env["KRABBY_RSI_BANK"] == str(omni.OMNI_BANK)
    assert env["KRABBY_RSI_BANK"].endswith("rsi_bank_omni.npz")
    assert "KRABBY_HEADING" not in env, "the facing sampler replaces KRABBY_HEADING"
    for dropped in omni.DROPPED_KEYS:
        assert dropped not in env, f"{name}: {dropped} must be dropped from every omni window"


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_per_stage_sampler_values(omni, name):
    env = omni.PHASES[name].env
    for key, want in EXPECTED_STAGE[name].items():
        if want is None:
            assert key not in env, f"{name}: {key} must be absent, got {env.get(key)!r}"
        else:
            assert env[key] == want, f"{name}: {key}"


def test_phase1_stages_differ_only_in_the_sampler(omni):
    """1a / 1b / 1c share one env except for the direction band, the slots, the RSI source and
    (1c only) the facing band and the facing reward weight."""
    a, b, c = (omni.PHASES[n].env for n in ("1a", "1b", "1c"))
    expected_ab = {"KRABBY_OMNI_DIR_BAND", "KRABBY_OMNI_RSI_SOURCE"}
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == expected_ab
    expected_ac = {"KRABBY_OMNI_DIR_BAND", "KRABBY_OMNI_SLOTS", "KRABBY_OMNI_RSI_SOURCE",
                   "KRABBY_OMNI_FACING", "KRABBY_YAW_W"}
    assert {k for k in set(a) | set(c) if a.get(k) != c.get(k)} == expected_ac


def test_clockheld_is_2c_minus_the_phaseout(omni):
    ch, c = omni.PHASES["2c_clockheld"].env, omni.PHASES["2c"].env
    assert "KRABBY_PHASEOUT" in c and "KRABBY_PHASEOUT" not in ch
    assert ch["KRABBY_CLOCK_W"] == "1.0"
    assert {k: v for k, v in c.items() if k != "KRABBY_PHASEOUT"} == ch
    assert omni.PHASES["2c_clockheld"].task == omni.PHASES["2c"].task


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_phase_env_key_delta(fwd, omni, name):
    f = fwd.phase_env(FORWARD_OF[name], "A15+B")
    o = omni.phase_env(name, "A15+B")
    exp_fwd_only, exp_omni_only, _ = EXPECTED_DELTA[name]
    assert set(f) - set(o) == exp_fwd_only | {"KRABBY_PHASE"}
    assert set(o) - set(f) == exp_omni_only | {"KRABBY_OMNI_PHASE"}
    assert o["KRABBY_OMNI_PHASE"] == name
    assert "KRABBY_PHASE" not in o and "KRABBY_SIDE_PHASE" not in o
    if name.startswith("2"):
        assert o["KRABBY_HEX_TEACHER_MODE"] == omni.PHASES[name].teacher_mode


# ------------------------------------------------------------------------------ activation
def test_activation_from_omni_variable(omni):
    env = {"KRABBY_OMNI_PHASE": "2b", "KRABBY_PLANT": "A15+B", "KRABBY_STAND_FRAC": "0.35"}
    applied = omni.activate_phase(env)
    assert env["KRABBY_STAND_FRAC"] == "0.35" and "KRABBY_STAND_FRAC" not in applied
    assert env["KRABBY_HEX_TEACHER_MODE"] == "2b"
    assert "KRABBY_HEX_USD_PATH" not in env          # A15+B is the main asset: nothing to export
    assert env["KRABBY_FLAT_TERRAIN_GEOM"] == "ring" and env["KRABBY_EPISODE_S"] == "40"
    assert env["KRABBY_OMNI_SLOTS"] == "0.15:0.45:0.15:0.25"
    assert "KRABBY_PHASE" not in env and "KRABBY_SIDE_PHASE" not in env
    assert "KRABBY_HEADING" not in env


@pytest.mark.parametrize("foreign", ["KRABBY_PHASE", "KRABBY_SIDE_PHASE"])
def test_foreign_variable_alone_is_a_noop_for_omni(omni, foreign):
    """import_packages imports forward -> omni -> sideways: a forward or sideways process must
    pass through the omni activation without one byte moving."""
    env = {foreign: "2b", "KRABBY_PLANT": "A15+B"}
    before = dict(env)
    assert omni.activate_phase(env) == {}
    assert env == before


def test_nothing_set_is_a_noop(omni):
    env: dict[str, str] = {}
    assert omni.activate_phase(env) == {}
    assert env == {}
    env = {"KRABBY_PLANT": "B"}
    assert omni.activate_phase(env) == {}
    assert env == {"KRABBY_PLANT": "B"}, "the omni activation must not expand a bare plant"


def test_omni_variable_alone_is_a_noop_for_the_other_two(fwd, side):
    for mod in (fwd, side):
        env = {"KRABBY_OMNI_PHASE": "1c"}
        assert mod.activate_phase(env) == {}
        assert env == {"KRABBY_OMNI_PHASE": "1c"}


@pytest.mark.parametrize("foreign", ["KRABBY_PHASE", "KRABBY_SIDE_PHASE"])
def test_pairwise_collision_raises(omni, foreign):
    with pytest.raises(RuntimeError, match="one task per process"):
        omni.activate_phase({"KRABBY_OMNI_PHASE": "1a", foreign: "2b"})


def test_all_three_variables_raise(omni, side):
    both = {"KRABBY_OMNI_PHASE": "1a", "KRABBY_PHASE": "2b", "KRABBY_SIDE_PHASE": "2b"}
    with pytest.raises(RuntimeError, match="one task per process"):
        omni.activate_phase(dict(both))
    with pytest.raises(RuntimeError, match="one task per process"):
        side.activate_phase(dict(both))


def test_unknown_omni_phase_fails_loudly(omni):
    for name in ("o9z", "1d", "3a", "legacy_golden_1a"):
        with pytest.raises(KeyError):
            omni.phase_env(name)
        with pytest.raises(KeyError):
            omni.activate_phase({"KRABBY_OMNI_PHASE": name})


def test_plant_export(omni):
    env = {"KRABBY_OMNI_PHASE": "1a", "KRABBY_PLANT": "B"}
    omni.activate_phase(env)
    assert env["KRABBY_HEX_USD_PATH"].endswith("crab_simple__splay00_axis2p5in.usda")


# ------------------------------------------------------------------------------ shared tables
def test_omni_module_is_pure():
    """Loading the omni presets by path must not import ``parkour_tasks`` or ``isaaclab``.

    Checked in a FRESH interpreter: other unit tests in the same pytest process legitimately
    stub ``isaaclab.*`` into ``sys.modules``, so asserting on the shared ``sys.modules`` would
    report their stubs, not this module's imports.
    """
    import os
    import subprocess

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('crab_hex_omni_phases', {str(OMNI_PHASES_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['crab_hex_omni_phases'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.PHASES\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'isaaclab'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_knobs_module_is_pure():
    """Same for the knob parsers: they must be loadable (and testable) without Isaac."""
    import os
    import subprocess

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('omni_knobs', {str(OMNI_KNOBS_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['omni_knobs'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.parse_slots('0.2:0.8:0:0')\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'isaaclab'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_plant_table_and_blocks_are_the_forward_ones(fwd, omni):
    assert omni.PLANTS == fwd.PLANTS
    assert omni.MAIN_PLANT == "A15+B"
    assert omni.FORMATION == fwd.FORMATION
    assert omni.ELEMENTS_5K == fwd.ELEMENTS_5K and omni.ELEMENTS_10K == fwd.ELEMENTS_10K
    assert omni.WALKING_SLOTS == fwd.WALKING_SLOTS and omni.LONG_EPISODES == fwd.LONG_EPISODES
    assert omni.GAIT_WEIGHTS == fwd.GAIT_WEIGHTS and omni.GAIT_RAMPS == fwd.GAIT_RAMPS
    assert omni.WINDOW_ITERS == fwd.WINDOW_ITERS and omni.RAMP_ITERS == fwd.RAMP_ITERS
    assert omni.STEPS_PER_ITER == fwd.STEPS_PER_ITER and omni.EPS == fwd.EPS
    assert omni.MAIN_ASSET_NAMES == fwd.MAIN_ASSET_NAMES
    assert omni.plant_usd_path("B") == fwd.plant_usd_path("B")
    assert omni.REPO_ROOT == REPO and omni.EXPERIMENTS == OMNI / "experiments"
    assert omni.PHASE_VAR == "KRABBY_OMNI_PHASE"


# ------------------------------------------------------------------------------ knob parsers
def test_parse_slots(knobs):
    assert knobs.parse_slots("0.2:0.8:0:0") == (0.2, 0.8, 0.0, 0.0)
    assert knobs.parse_slots(" 0.15 : 0.45 : 0.15 : 0.25 ") == (0.15, 0.45, 0.15, 0.25)
    for bad in ("0.2:0.8:0", "0.2:0.8:0:0:0", "0:0:0:0", "0.2:0.8:-0.1:0.1", "a:b:c:d", ""):
        with pytest.raises(ValueError):
            knobs.parse_slots(bad)


def test_parse_dir_band(knobs):
    assert knobs.parse_dir_band("0:0") == (0.0, 0.0)
    assert knobs.parse_dir_band("1.5708:1.5708") == (1.5708, 1.5708)
    for empty in ("", "  ", "none", "OFF"):
        assert knobs.parse_dir_band(empty) is None, empty
    for bad in ("1:0", "0", "0:1:2", "x:y"):
        with pytest.raises(ValueError):
            knobs.parse_dir_band(bad)


def test_parse_speed_ellipse_and_scalars(knobs):
    assert knobs.parse_speed_ellipse("0.35:0.25:0.25") == (0.35, 0.25, 0.25)
    assert knobs.parse_ang_vel_max("0.3") == 0.3
    assert knobs.parse_clock_turn_radius("1.2") == 1.2
    assert knobs.parse_clock_turn_radius("0") == 0.0
    for bad in ("0.35:0.25", "0:0.25:0.25", "-0.1:0.25:0.25"):
        with pytest.raises(ValueError):
            knobs.parse_speed_ellipse(bad)
    for bad in ("0", "-0.3", "nan", "0.3:0.4"):
        with pytest.raises(ValueError):
            knobs.parse_ang_vel_max(bad)
    with pytest.raises(ValueError):
        knobs.parse_clock_turn_radius("-1")


def test_parse_facing_and_yaw_range(knobs):
    assert knobs.parse_facing("0.2:0.6") == (0.2, 0.6)
    assert knobs.parse_facing("0.2:2.4") == (0.2, 2.4)
    for bad in ("0.6:0.2", "-0.1:0.6", "0.2"):
        with pytest.raises(ValueError):
            knobs.parse_facing(bad)
    lo, hi = knobs.parse_yaw_range("-3.14159:3.14159")
    assert math.isclose(lo, -3.14159) and math.isclose(hi, 3.14159)
    assert knobs.parse_yaw_range("0.5:0.5") == (0.5, 0.5)   # a pinned spawn yaw is legal
    with pytest.raises(ValueError):
        knobs.parse_yaw_range("1:0")


def test_parse_rsi_source(knobs):
    assert knobs.parse_rsi_source("0") == 0
    assert knobs.parse_rsi_source("1") == 1
    assert knobs.parse_rsi_source("all") == "all"
    assert knobs.parse_rsi_source(" ALL ") == "all"
    for bad in ("2", "", "forward", "0.0"):
        with pytest.raises(ValueError):
            knobs.parse_rsi_source(bad)


@pytest.mark.parametrize("name", PHASE_NAMES)
def test_every_preset_value_parses(omni, knobs, name):
    """Round-trip: each window's knob strings must be accepted by the parsers that read them."""
    env = omni.PHASES[name].env
    knobs.parse_slots(env["KRABBY_OMNI_SLOTS"])
    knobs.parse_speed_ellipse(env["KRABBY_OMNI_SPEED_ELLIPSE"])
    knobs.parse_ang_vel_max(env["KRABBY_OMNI_ANG_VEL_MAX"])
    knobs.parse_clock_turn_radius(env["KRABBY_OMNI_CLOCK_TURN_R"])
    knobs.parse_yaw_range(env["KRABBY_OMNI_YAW_RANGE"])
    knobs.parse_rsi_source(env["KRABBY_OMNI_RSI_SOURCE"])
    if "KRABBY_OMNI_DIR_BAND" in env:
        assert knobs.parse_dir_band(env["KRABBY_OMNI_DIR_BAND"]) is not None
    if "KRABBY_OMNI_FACING" in env:
        knobs.parse_facing(env["KRABBY_OMNI_FACING"])
    # the sampler cap must clear the walk floor with headroom (the command cfg's own check)
    a, b, c = knobs.parse_speed_ellipse(env["KRABBY_OMNI_SPEED_ELLIPSE"])
    assert min(a, b, c) > 0.21 + 0.01
    # and the yaw-rate saturation must sit above the dead band (ang_vel_clip 0.2)
    assert knobs.parse_ang_vel_max(env["KRABBY_OMNI_ANG_VEL_MAX"]) > 0.2
    # the added term's weight is a plain float (applied by _apply_reward_weight_knobs)
    assert float(env["KRABBY_YAW_PROGRESS_W"]) == 1.0
    # the facing-error observation clip is a plain float > 0 (applied by
    # _apply_facing_obs_clip_knob) and must round-trip to the 1c band's edge, 0.6 rad, in
    # EVERY window regardless of that window's own facing-band ramp (Revision 3, 2026-09-17)
    assert float(env["KRABBY_OMNI_FACING_OBS_CLIP"]) == 0.6


# ------------------------------------------------------------------ leaked-command-cfg guard
class _FakeOmniCommandCfg:
    """Stand-in for ``CrabHexOmniCommandCfg`` -- just the fields ``_apply_command_knobs``
    writes, so the guard can be exercised without the real Isaac-backed class (the
    ``omni_terrain_knobs.apply_omni_ring_knobs(..., classes=...)`` precedent)."""

    def __init__(self):
        self.slot_probs = None
        self.speed_ellipse = None
        self.facing_delta = None
        self.ang_vel_max = None
        self.dir_band = None


class _FakeForwardCommandCfg:
    """A DIFFERENT class -- stands in for a leaked forward/sideways command cfg."""


def test_apply_command_knobs_applies_onto_the_omni_cfg(knobs):
    cfg = _FakeOmniCommandCfg()
    knobs._apply_command_knobs(
        cfg, {"KRABBY_OMNI_SLOTS": "0.2:0.8:0:0"}, cmd_cls=_FakeOmniCommandCfg
    )
    assert cfg.slot_probs == (0.2, 0.8, 0.0, 0.0)


def test_apply_command_knobs_rejects_a_leaked_forward_cfg(knobs):
    """A forward/sideways command cfg leaking into the omni env must raise loudly by name --
    NOT silently grow an orphan ``slot_probs`` attribute, which is what plain ``setattr`` would
    do (the bug the module docstring's item 1 now documents, replacing the old false
    ``AttributeError`` claim)."""
    cfg = _FakeForwardCommandCfg()
    with pytest.raises(RuntimeError, match="_FakeForwardCommandCfg"):
        knobs._apply_command_knobs(
            cfg, {"KRABBY_OMNI_SLOTS": "0.2:0.8:0:0"}, cmd_cls=_FakeOmniCommandCfg
        )
    assert not hasattr(cfg, "slot_probs")


# ------------------------------------------------------- the added term's weight knob
class _FakeTerm:
    def __init__(self, weight=1.0):
        self.weight = weight


class _FakeEnvCfg:
    """Just the one path ``_apply_reward_weight_knobs`` writes."""

    def __init__(self, with_term=True):
        self.rewards = types.SimpleNamespace()
        if with_term:
            self.rewards.reward_yaw_progress = _FakeTerm()


def test_yaw_progress_weight_knob_is_applied(knobs):
    assert knobs.YAW_PROGRESS_W_VAR == "KRABBY_YAW_PROGRESS_W"
    cfg = _FakeEnvCfg()
    knobs._apply_reward_weight_knobs(cfg, {"KRABBY_YAW_PROGRESS_W": "0.6"})
    assert cfg.rewards.reward_yaw_progress.weight == 0.6
    # presence-based, like the forward _overrides table: 0.0 is a legitimate ABLATION and must
    # win over the class-body default, so it may not be skipped as "falsy"
    knobs._apply_reward_weight_knobs(cfg, {"KRABBY_YAW_PROGRESS_W": "0"})
    assert cfg.rewards.reward_yaw_progress.weight == 0.0
    # unset / empty -> the class-body literal survives untouched
    cfg2 = _FakeEnvCfg()
    knobs._apply_reward_weight_knobs(cfg2, {})
    knobs._apply_reward_weight_knobs(cfg2, {"KRABBY_YAW_PROGRESS_W": "  "})
    assert cfg2.rewards.reward_yaw_progress.weight == 1.0


def test_yaw_progress_weight_knob_refuses_a_cfg_without_the_term(knobs):
    """A leaked forward rewards cfg must raise, not grow an orphan weight nobody reads."""
    cfg = _FakeEnvCfg(with_term=False)
    with pytest.raises(RuntimeError, match="reward_yaw_progress"):
        knobs._apply_reward_weight_knobs(cfg, {"KRABBY_YAW_PROGRESS_W": "1.0"})
    assert not hasattr(cfg.rewards, "reward_yaw_progress")
    # ... and with the knob unset it is a no-op even then (RSI-off style configurations)
    knobs._apply_reward_weight_knobs(cfg, {})


# ------------------------------------------------- the facing-error observation clip knob
class _FakeObsFunc:
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}


class _FakeForwardObsFunc:
    pass


class _FakeObsTerm:
    def __init__(self, func):
        self.func = func
        self.params = {"asset_cfg": "robot"}


class _FakeObsCfg:
    def __init__(self, func):
        class _Group:  # noqa: D401 - a bag of terms
            pass
        self.observations = types.SimpleNamespace(policy=_Group())
        self.observations.policy.proprio = _FakeObsTerm(func)


def test_facing_obs_clip_knob_absent_is_a_no_op(knobs):
    cfg = _FakeObsCfg(_FakeObsFunc)
    knobs._apply_facing_obs_clip_knob(cfg, {})
    assert "facing_obs_clip" not in cfg.observations.policy.proprio.params


def test_facing_obs_clip_knob_writes_the_term_param(knobs):
    cfg = _FakeObsCfg(_FakeObsFunc)
    knobs._apply_facing_obs_clip_knob(cfg, {"KRABBY_OMNI_FACING_OBS_CLIP": "0.6"})
    assert cfg.observations.policy.proprio.params["facing_obs_clip"] == 0.6
    assert cfg.observations.policy.proprio.params["asset_cfg"] == "robot"


def test_facing_obs_clip_knob_refuses_a_leaked_forward_obs(knobs):
    cfg = _FakeObsCfg(_FakeForwardObsFunc)
    with pytest.raises(RuntimeError, match="no omni observation term"):
        knobs._apply_facing_obs_clip_knob(cfg, {"KRABBY_OMNI_FACING_OBS_CLIP": "0.6"})


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_facing_obs_clip_knob_rejects_non_positive(knobs, bad):
    cfg = _FakeObsCfg(_FakeObsFunc)
    with pytest.raises(RuntimeError, match="KRABBY_OMNI_FACING_OBS_CLIP"):
        knobs._apply_facing_obs_clip_knob(cfg, {"KRABBY_OMNI_FACING_OBS_CLIP": bad})
