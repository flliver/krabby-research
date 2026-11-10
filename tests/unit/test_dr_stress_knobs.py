"""DR stress sweep (omni rung 1, 2026-11-06): the single-parameter stress knobs and their hooks, on CPU.

``config/crab_hex/stress_knobs.py`` (the ``KRABBY_STRESS_*`` knobs, default off, set by no preset) moves ONE plant or
sensor parameter of the frozen F19 student per sweep cell: it rescales actuator fields on the env cfg, pins the action
delay and the friction, adds a startup leg-mass event, writes the attributes the action term reads (the rod / cam
scales and the push spec ``mdp/stress_push.py`` turns into per-env timers), writes ``cfg.stress_obs`` for the sensor
side (``mdp/stress_obs.py``) and re-points the four depth terms (``image_features_batched`` ``lag``). Every failure mode
here is silent in a real run -- a knob that is armed but inert is read by the sweep as "ROBUST" -- so each is pinned:

* parsers: every variable has one; blank is off; range edges; non-finite, non-numeric and fractional-integer values are
  refused with the variable's name; ``VXY_ZERO=1`` with ``VXY_NOISE`` is refused; any OTHER ``KRABBY_STRESS_*`` name is
  refused (a typo would run the cell at nominal); every pre-registered CHARTER level parses; a typo anywhere refuses
  the whole cell before anything is written;
* unset (or blank) writes nothing -- no attribute, no event, no print, no rebound container; armed ALONE, a knob changes
  exactly its own targets (diffed against a deep copy), the scaled dicts are NEW objects (``crab_hex_scene_cfg`` passes
  module-level dicts into the actuator cfgs by reference, so an in-place scale could leak into another cfg), and arming
  every knob at once is the union of the singles; a cfg that lacks a knob's target raises instead of passing silently;
* the sensor side (``StressObs``): bounded uniform noise at the reference half-widths, the other channels untouched,
  inputs never modified, a private seeded generator (same seed same draws, the global torch stream untouched), the
  current proxy's gain / clip / offset exactly, ``cur_scale = 0`` honoured, unknown spec keys refused;
* the push (``StressPush``, stepped by the action term): the interval in env steps, timers that fire at exactly that
  many steps after construction / reset, per-env restarts, |dv| exactly, headings that are a pure splitmix64 hash of
  (seed, epoch, env, k) -- independent of what other envs did, a new epoch and a restarted k at every reset of ALL envs,
  no torch RNG consumed -- and refusals; the action term's three hooks (build in ``__init__``, ``reset``, ``step`` in
  ``process_actions``) are exec'd from their own source and driven on a fake term; the leg-mass builder's contents
  against IsaacLab's static argument check; the leg-link regex against the plant USD;
* the depth term: ``lag`` absent is the old ``buffer[:, -2]``; ``lag = n`` returns the image n refreshes old (a model over
  41 steps); a reset row never shows a pre-reset frame; bad lags refused; ``__call__`` takes ``lag`` and keeps it
  defaulted, so the unarmed configs still pass the static check;
* wiring: ``omni_knobs`` re-exports the tuple, the parsers and the 14 ``STRESS_<NAME>_VAR`` names (the ``*_VAR`` scan the
  driver's whitelist runs) and applies the stress knobs between the payload and the motion knobs; the attribute and key
  names agree between the writer and its consumers (source-level for the two Kit-only files).

Nothing real is registered under ``isaaclab.*`` / ``parkour_tasks.*``: the four stress modules are path-loaded under
private names, the Isaac leg-mass builder runs against ``monkeypatch``-ed stand-ins, the depth term and the action
term's push hooks are exec'd from their source segments (``test_task4_student.py``'s technique).
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import os
import re
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour" / "parkour_tasks" / "parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
STRESS_KNOBS_PY = OMNI / "config" / "crab_hex" / "stress_knobs.py"
OMNI_KNOBS_PY = OMNI / "config" / "crab_hex" / "omni_knobs.py"
OMNI_OBS_PY = OMNI / "mdp" / "crab_hex_omni_observations.py"
STRESS_OBS_PY = FWD_MDP / "stress_obs.py"
STRESS_PUSH_PY = FWD_MDP / "stress_push.py"
FWD_OBS_PY = FWD_MDP / "observations.py"
FWD_ACTIONS_PY = FWD_MDP / "parkour_actions.py"
PLANT_USD = REPO / "assets" / "crab.usda"

def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


sk = _load(STRESS_KNOBS_PY, "stress_knobs__dr_stress_test")
so = _load(STRESS_OBS_PY, "stress_obs__dr_stress_test")
sp = _load(STRESS_PUSH_PY, "stress_push__dr_stress_test")


# =============================================================================== a fake env cfg
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
PITCH = tuple(f"{leg}_Hip_Femur_RevoluteJoint" for leg in LEGS)
KNEE = tuple(f"{leg}_Femur_Tibia_RevoluteJoint" for leg in LEGS)
CAMS = tuple(f"{leg}_Body_CamShaft_RevoluteJoint" for leg in LEGS)
HIP_TRACKING = tuple(f"{leg}_Body_Hip_RevoluteJoint" for leg in LEGS)
ABSENT = "<absent>"


def _per(names, value):
    return {name: value for name in names}


def _fake_cfg():
    """What the stress knobs read and write of a built omni student env cfg, at the F19 nominal values.

    Mirrors ``crab_hex_scene_cfg._crab_robot_cfg`` (the joint-keyed dicts), ``crab_hex_student_cfg`` (delay on, history 8),
    the parkour ``physics_material`` event and the four student depth groups. ``body_hip_tracking`` and the velocity
    limits are bystanders: no knob may touch them.
    """
    def material():
        return SimpleNamespace(static_friction=1.0, dynamic_friction=1.0)

    def depth_group():
        return SimpleNamespace(depth_image=SimpleNamespace(
            params={"sensor_cfg": "cam", "resize": (58, 87), "buffer_len": 2}))

    return SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(actuators={
                "body_hip_yaw": SimpleNamespace(
                    effort_limit=_per(CAMS, 20.0), saturation_effort=_per(CAMS, 24.0)),
                "body_hip_tracking": SimpleNamespace(
                    effort_limit=_per(HIP_TRACKING, 600.0), saturation_effort=_per(HIP_TRACKING, 738.5),
                    stiffness=_per(HIP_TRACKING, 4000.0), damping=_per(HIP_TRACKING, 40.0)),
                "hip_femur": SimpleNamespace(
                    effort_limit_sim=_per(PITCH, 1000.0), velocity_limit_sim=2.0,
                    stiffness=_per(PITCH, 16000.0), damping=_per(PITCH, 600.0)),
                "femur_tibia": SimpleNamespace(
                    effort_limit_sim=_per(KNEE, 400.0), velocity_limit_sim=3.0,
                    stiffness=_per(KNEE, 6000.0), damping=_per(KNEE, 200.0)),
            }),
            terrain=SimpleNamespace(physics_material=material()),
        ),
        sim=SimpleNamespace(physics_material=material()),
        actions=SimpleNamespace(joint_pos=SimpleNamespace(use_delay=True, history_length=8, action_delay_steps=[1, 1])),
        events=SimpleNamespace(
            physics_material=SimpleNamespace(params={"friction_range": (0.6, 2.0), "num_buckets": 64}),
            push_by_setting_velocity=None, reset_root_state="spawn"),
        observations=SimpleNamespace(**{group: depth_group() for group in sk.DEPTH_GROUPS}),
    )


def _leaves(obj, path=""):
    """``{dotted path: leaf}`` of a cfg tree: SimpleNamespaces and dicts are walked, everything else (an EMPTY container
    too, so a stray ``stress_obs = {}`` shows up in a diff) is a leaf."""
    if isinstance(obj, SimpleNamespace):
        items = list(vars(obj).items())
    elif isinstance(obj, dict):
        items = list(obj.items())
    else:
        return {path: obj}
    if not items:
        return {path: obj}
    out = {}
    for key, value in items:
        out.update(_leaves(value, f"{path}.{key}" if path else str(key)))
    return out


def _containers(obj, path=""):
    """``{dotted path: id}`` of every SimpleNamespace / dict in the tree (a rebound container changes its id)."""
    out = {}
    if isinstance(obj, (SimpleNamespace, dict)):
        out[path] = id(obj)
        items = vars(obj).items() if isinstance(obj, SimpleNamespace) else obj.items()
        for key, value in items:
            out.update(_containers(value, f"{path}.{key}" if path else str(key)))
    return out


def _diff(before, after):
    """``{path: (old, new)}`` for every leaf that differs; ``ABSENT`` marks a leaf one side lacks."""
    b, a = _leaves(before), _leaves(after)
    return {p: (b.get(p, ABSENT), a.get(p, ABSENT)) for p in sorted(set(b) | set(a))
            if b.get(p, ABSENT) != a.get(p, ABSENT)}


def _leg_mass(scale):
    return ("leg", scale)


def _boom(_value):
    raise AssertionError("the leg-mass event builder ran for a knob that is off")


def _alone(environ, cfg=None):
    """Apply ``environ`` to a fresh fake cfg; ``(cfg, before, diff, armed)``."""
    cfg = _fake_cfg() if cfg is None else cfg
    before = copy.deepcopy(cfg)
    armed = sk.apply_stress_knobs(cfg, environ, leg_mass_event=_leg_mass)
    return cfg, before, _diff(before, cfg), armed


# =============================================================================== A. the parsers
#: (variable, lo, hi, lo_open): the range of every float knob. Typo guards, not physics -- the CHARTER levels sit inside.
FLOAT_KNOBS = (
    (sk.ROD_SPEED_VAR, 0.0, 4.0, True),
    (sk.ROD_EFFORT_VAR, 0.0, 4.0, True),
    (sk.CAM_VAR, 0.0, 4.0, True),
    (sk.KPKD_VAR, 0.0, 4.0, True),
    (sk.LEG_MASS_VAR, 0.0, 4.0, True),
    (sk.MU_VAR, 0.0, 3.0, True),
    (sk.OBS_NOISE_VAR, 0.0, 10.0, True),
    (sk.CUR_SCALE_VAR, 0.0, 4.0, False),      # a gain of exactly 0 is a legal "dead current sensor"
    (sk.CUR_NOISE_VAR, 0.0, 1.0, True),
    (sk.VXY_NOISE_VAR, 0.0, 2.0, True),
    (sk.PUSH_DV_VAR, 0.0, 3.0, True),
)
INT_KNOBS = ((sk.ACT_DELAY_VAR, 0, 7), (sk.DEPTH_LAG_VAR, 1, 10))
BAD_TEXT = ("abc", "nan", "NaN", "inf", "-inf", "1,5", "0x1", "1 2", "--1")

#: The pre-registered levels (CHARTER 3.2); each is a cell the queue will launch, so each must parse.
CHARTER_LEVELS = {
    sk.ROD_SPEED_VAR: ("0.5", "0.7", "0.8", "1.15", "1.35"),
    sk.ROD_EFFORT_VAR: ("0.5", "0.7", "0.85"),
    sk.CAM_VAR: ("0.5", "0.8", "1.2"),
    sk.KPKD_VAR: ("0.5", "0.75", "1.25"),
    sk.LEG_MASS_VAR: ("0.9", "1.15", "1.5"),
    sk.MU_VAR: ("0.3", "0.4", "0.8", "1.2"),
    sk.ACT_DELAY_VAR: ("2", "3", "4"),
    sk.OBS_NOISE_VAR: ("0.5", "1", "2", "4"),
    sk.CUR_SCALE_VAR: ("0.85", "1.15"),
    sk.CUR_NOISE_VAR: ("0.1", "0.25"),
    sk.VXY_NOISE_VAR: ("0.05", "0.1", "0.2"),
    sk.VXY_ZERO_VAR: ("1",),
    sk.DEPTH_LAG_VAR: ("2", "3"),
    sk.PUSH_DV_VAR: ("0.1", "0.2", "0.35", "0.5"),
}


def test_every_variable_has_a_parser_and_is_registered():
    consts = {getattr(sk, a) for a in dir(sk) if a.endswith("_VAR")}
    assert consts == set(sk.STRESS_VARS)                    # a *_VAR constant left out of STRESS_VARS would be inert
    assert len(sk.STRESS_VARS) == len(set(sk.STRESS_VARS))
    assert all(v.startswith("KRABBY_STRESS_") for v in sk.STRESS_VARS)
    assert set(sk.STRESS_KNOB_PARSERS) == set(sk.STRESS_VARS)
    assert all(callable(p) for p in sk.STRESS_KNOB_PARSERS.values())
    assert set(CHARTER_LEVELS) == set(sk.STRESS_VARS)       # this file's tables cover every knob
    assert {v for v, *_ in FLOAT_KNOBS} | {v for v, *_ in INT_KNOBS} | {sk.VXY_ZERO_VAR} == set(sk.STRESS_VARS)
    for name in sk.__all__:
        assert hasattr(sk, name), name


@pytest.mark.parametrize("var, lo, hi, lo_open", FLOAT_KNOBS, ids=[k[0] for k in FLOAT_KNOBS])
def test_float_knob_range_edges(var, lo, hi, lo_open):
    parse = sk.STRESS_KNOB_PARSERS[var]
    for blank in (None, "", "   "):
        assert parse(blank) is None                          # blank = off
    assert parse("1e-9") == 1e-9                             # just inside the lower edge, open or closed
    assert parse(repr(hi)) == hi and parse(f"  {hi}  ") == hi    # the top edge is inclusive; whitespace is stripped
    assert type(parse(repr(hi))) is float
    refused = [repr(hi + 1e-6), repr(hi * 2), "-1", "-1e-9"]
    if lo_open:
        refused += ["0", "0.0", "-0"]
    else:
        assert parse("0") == 0.0 and math.copysign(1.0, parse("-0")) == 1.0       # +0.0, never -0.0
    for text in refused + list(BAD_TEXT):
        with pytest.raises(ValueError, match=var):
            parse(text)


@pytest.mark.parametrize("var, lo, hi", INT_KNOBS, ids=[k[0] for k in INT_KNOBS])
def test_integer_knob_edges(var, lo, hi):
    parse = sk.STRESS_KNOB_PARSERS[var]
    for blank in (None, "", "   "):
        assert parse(blank) is None
    assert parse(str(lo)) == lo and parse(str(hi)) == hi and parse(f" {hi} ") == hi
    assert type(parse(str(hi))) is int                       # a count, not a float (it indexes a history buffer)
    for text in (str(lo - 1), str(hi + 1), "2.5", "1.5", f"{lo}.5", "-1.0", "0.5") + BAD_TEXT:
        with pytest.raises(ValueError, match=var):
            parse(text)


def test_vxy_zero_is_a_flag_whose_zero_means_unarmed():
    parse = sk.STRESS_KNOB_PARSERS[sk.VXY_ZERO_VAR]
    for on in ("1", "true", "True", " TRUE "):
        assert parse(on) is True
    for off in (None, "", "  ", "0", "false", "False", " 0 "):
        assert parse(off) is None                            # an explicit off arms nothing
    for bad in ("2", "-1", "maybe", "nan", "1.0", "01", "yes please"):
        with pytest.raises(ValueError, match=sk.VXY_ZERO_VAR):
            parse(bad)
    assert sk.parse_flag("1", "V") is True and sk.parse_flag("0", "V") is False and sk.parse_flag("", "V") is False
    with pytest.raises(ValueError, match="V must be 1 or 0"):
        sk.parse_flag("2", "V")


def test_every_preregistered_charter_level_parses():
    for var, levels in CHARTER_LEVELS.items():
        parse = sk.STRESS_KNOB_PARSERS[var]
        for text in levels:
            value = parse(text)
            assert value is not None, (var, text)
            if var != sk.VXY_ZERO_VAR:
                assert value == float(text), (var, text)
            assert sk.parse_stress_knobs({var: text}) == {var: value}


def _clean_stress_environ(monkeypatch):
    """No ``KRABBY_STRESS_*`` in the process environment (an unknown one would be refused, a known one would arm)."""
    for key in [k for k in os.environ if k.startswith("KRABBY_STRESS_")]:
        monkeypatch.delenv(key)


def test_parse_stress_knobs_arms_only_what_is_set(monkeypatch):
    _clean_stress_environ(monkeypatch)
    assert sk.parse_stress_knobs({}) == {}
    assert sk.parse_stress_knobs({v: "" for v in sk.STRESS_VARS}) == {}
    assert sk.parse_stress_knobs({v: "  " for v in sk.STRESS_VARS}) == {}
    assert sk.parse_stress_knobs({sk.VXY_ZERO_VAR: "0"}) == {}
    environ = {sk.ROD_SPEED_VAR: "0.7", sk.MU_VAR: " 0.4 ", sk.ACT_DELAY_VAR: "3", sk.VXY_ZERO_VAR: "1",
               sk.PUSH_DV_VAR: "", "PATH": "/bin", "KRABBY_PAYLOAD_KG": "0:90"}
    assert sk.parse_stress_knobs(environ) == {
        sk.ROD_SPEED_VAR: 0.7, sk.MU_VAR: 0.4, sk.ACT_DELAY_VAR: 3, sk.VXY_ZERO_VAR: True}
    monkeypatch.setenv(sk.KPKD_VAR, "1.25")                  # None / default: the process environment
    assert sk.parse_stress_knobs() == sk.parse_stress_knobs(None) == {sk.KPKD_VAR: 1.25}
    cfg = _fake_cfg()
    assert sk.apply_stress_knobs(cfg) == {sk.KPKD_VAR: 1.25}
    assert cfg.scene.robot.actuators["hip_femur"].stiffness[PITCH[0]] == 16000.0 * 1.25


@pytest.mark.parametrize("name", ["KRABBY_STRESS_ROD_SPED", "KRABBY_STRESS_", "KRABBY_STRESS_PUSH", "KRABBY_STRESS_MU2",
                                  "KRABBY_STRESS_rod_speed"])
@pytest.mark.parametrize("value", ["1", "", "  "], ids=["valued", "blank", "whitespace"])
def test_an_unknown_stress_name_is_refused_whatever_its_value(name, value):
    """A misspelled knob would otherwise run its cell at nominal and read as ROBUST."""
    with pytest.raises(ValueError, match=name):
        sk.parse_stress_knobs({name: value})
    cfg = _fake_cfg()
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match=name):              # and the valid knob beside it is not applied either
        sk.apply_stress_knobs(cfg, {sk.ROD_SPEED_VAR: "0.7", name: value}, leg_mass_event=_boom)
    assert cfg == before


def test_unknown_stress_names_are_refused_in_the_process_environment_and_look_alikes_are_not_ours(monkeypatch):
    _clean_stress_environ(monkeypatch)
    monkeypatch.setenv("KRABBY_STRESS_TYPO", "1")
    with pytest.raises(ValueError, match="KRABBY_STRESS_TYPO"):
        sk.parse_stress_knobs()                              # environ=None reads os.environ
    with pytest.raises(ValueError, match="KRABBY_STRESS_TYPO"):
        sk.apply_stress_knobs(_fake_cfg())
    monkeypatch.delenv("KRABBY_STRESS_TYPO")
    # the prefix is exactly ``KRABBY_STRESS_``: other families and look-alikes are left to their own modules
    look_alikes = {"KRABBY_STRESS": "1", "KRABBY_STRESSFUL": "1", "krabby_stress_x": "1", "KRABBY_PAYLOAD_KG": "0:90",
                   "XKRABBY_STRESS_X": "1"}
    assert sk.parse_stress_knobs(look_alikes) == {}


def test_vxy_zero_and_vxy_noise_are_exclusive():
    both = {sk.VXY_ZERO_VAR: "1", sk.VXY_NOISE_VAR: "0.1"}
    with pytest.raises(ValueError, match="exclusive"):
        sk.parse_stress_knobs(both)
    cfg = _fake_cfg()
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match="exclusive"):       # apply refuses it too, before writing anything
        sk.apply_stress_knobs(cfg, both)
    assert cfg == before
    assert sk.parse_stress_knobs({sk.VXY_ZERO_VAR: "0", sk.VXY_NOISE_VAR: "0.1"}) == {sk.VXY_NOISE_VAR: 0.1}
    assert sk.parse_stress_knobs({sk.VXY_ZERO_VAR: "1"}) == {sk.VXY_ZERO_VAR: True}


def test_a_malformed_knob_refuses_the_whole_cell_before_anything_is_written():
    cfg = _fake_cfg()
    before = copy.deepcopy(cfg)
    environ = {sk.ROD_SPEED_VAR: "0.7", sk.KPKD_VAR: "1.25", sk.OBS_NOISE_VAR: "2", sk.MU_VAR: "0,4"}
    with pytest.raises(ValueError, match=sk.MU_VAR):
        sk.apply_stress_knobs(cfg, environ, leg_mass_event=_boom)
    assert cfg == before                                     # not even the knobs listed before the typo
    assert not hasattr(cfg, "stress_obs")


# =============================================================================== B. unset writes nothing
UNSET_ENVIRONS = [
    pytest.param({}, id="empty"),
    pytest.param({v: "" for v in sk.STRESS_VARS}, id="all-blank"),
    pytest.param({v: "   " for v in sk.STRESS_VARS}, id="all-whitespace"),
    pytest.param({sk.VXY_ZERO_VAR: "0"}, id="explicit-off-flag"),
    pytest.param({"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_PAYLOAD_KG": "0:90", "KRABBY_STD_PARAM": "soft"},
                 id="other-families"),
]


@pytest.mark.parametrize("environ", UNSET_ENVIRONS)
def test_unset_or_blank_writes_nothing(environ, capsys):
    cfg = _fake_cfg()
    before = copy.deepcopy(cfg)
    shape = _containers(cfg)
    assert sk.apply_stress_knobs(cfg, environ, leg_mass_event=_boom) == {}
    assert cfg == before                                     # structurally identical, no new attribute anywhere
    assert _containers(cfg) == shape                         # and no container was even rebound to an equal copy
    for name in (sk.STRESS_OBS_ATTR, "stress_rod_speed_scale", "stress_cam_scale", sk.LEG_MASS_EVENT,
                 sk.STRESS_PUSH_ATTR):
        assert not any(name in path for path in _leaves(cfg)), name
    assert capsys.readouterr().out == ""                     # and says nothing
    # off never needs a block (nothing is read, nothing is written)
    bare = SimpleNamespace()
    assert sk.apply_stress_knobs(bare, environ) == {} and vars(bare) == {}


def test_armed_knobs_print_one_knob_line_per_group(capsys):
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.ROD_SPEED_VAR: "0.7", sk.KPKD_VAR: "1.25"})
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and all(line.startswith("[KNOB] ") for line in lines)
    assert f"{sk.ROD_SPEED_VAR}=0.7" in lines[0] and f"{sk.KPKD_VAR}=1.25" in lines[1]


# =============================================================================== C. each knob alone
def _rod_fields(fields, scale):
    """Expected diff entries: ``fields`` of both rod actuators scaled by ``scale``."""
    want = {}
    for actuator, names, nominal in (("hip_femur", PITCH, {"effort_limit_sim": 1000.0, "stiffness": 16000.0,
                                                            "damping": 600.0}),
                                     ("femur_tibia", KNEE, {"effort_limit_sim": 400.0, "stiffness": 6000.0,
                                                            "damping": 200.0})):
        for field in fields:
            for joint in names:
                want[f"scene.robot.actuators.{actuator}.{field}.{joint}"] = (nominal[field], nominal[field] * scale)
    return want


def _shared_dicts(cfg):
    """``{(actuator, field): (dict object, snapshot)}`` (the real dicts are module-level constants passed in by reference)."""
    out = {}
    for name, act in cfg.scene.robot.actuators.items():
        for field, value in vars(act).items():
            if isinstance(value, dict):
                out[(name, field)] = (value, dict(value))
    return out


def _assert_shared_dicts_untouched(cfg, shared, scaled):
    """The original dict objects hold their original values; the fields in ``scaled`` are fresh dict objects."""
    for (name, field), (obj, snapshot) in shared.items():
        assert obj == snapshot, (name, field)                                  # never scaled in place
        if (name, field) in scaled:
            assert getattr(cfg.scene.robot.actuators[name], field) is not obj, (name, field)


def test_rod_speed_writes_only_the_action_attribute():
    cfg, _, diff, armed = _alone({sk.ROD_SPEED_VAR: "0.7"})
    assert armed == {sk.ROD_SPEED_VAR: 0.7}
    assert diff == {"actions.joint_pos.stress_rod_speed_scale": (ABSENT, 0.7)}
    assert cfg.actions.joint_pos.stress_rod_speed_scale == 0.7


def test_rod_effort_scales_effort_limit_sim_of_both_rods_into_new_dicts():
    cfg = _fake_cfg()
    shared = _shared_dicts(cfg)
    cfg, _, diff, armed = _alone({sk.ROD_EFFORT_VAR: "0.85"}, cfg)
    assert armed == {sk.ROD_EFFORT_VAR: 0.85}
    assert diff == _rod_fields(("effort_limit_sim",), 0.85)      # stiffness, damping, velocity limits, cam untouched
    assert len(diff) == 12
    _assert_shared_dicts_untouched(cfg, shared, {("hip_femur", "effort_limit_sim"), ("femur_tibia", "effort_limit_sim")})
    assert cfg.scene.robot.actuators["hip_femur"].effort_limit_sim[PITCH[3]] == pytest.approx(850.0)
    assert cfg.scene.robot.actuators["femur_tibia"].effort_limit_sim[KNEE[3]] == pytest.approx(340.0)


@pytest.mark.parametrize("text", ["0.8", "1.2"])
def test_cam_scales_the_cam_motor_and_tells_the_action_term(text):
    s = float(text)
    cfg = _fake_cfg()
    shared = _shared_dicts(cfg)
    cfg, _, diff, armed = _alone({sk.CAM_VAR: text}, cfg)
    assert armed == {sk.CAM_VAR: s}
    want = {f"scene.robot.actuators.body_hip_yaw.effort_limit.{j}": (20.0, 20.0 * s) for j in CAMS}
    want.update({f"scene.robot.actuators.body_hip_yaw.saturation_effort.{j}": (24.0, 24.0 * s) for j in CAMS})
    want["actions.joint_pos.stress_cam_scale"] = (ABSENT, s)
    assert diff == want                                          # body_hip_tracking (also effort_limit) untouched
    _assert_shared_dicts_untouched(cfg, shared, {("body_hip_yaw", "effort_limit"), ("body_hip_yaw", "saturation_effort")})


def test_kpkd_scales_stiffness_and_damping_of_both_rods_only():
    cfg = _fake_cfg()
    shared = _shared_dicts(cfg)
    cfg, _, diff, armed = _alone({sk.KPKD_VAR: "1.25"}, cfg)
    assert armed == {sk.KPKD_VAR: 1.25}
    assert diff == _rod_fields(("stiffness", "damping"), 1.25)   # effort untouched; so is body_hip_tracking's Kp / Kd
    assert len(diff) == 24
    _assert_shared_dicts_untouched(cfg, shared, {(a, f) for a in ("hip_femur", "femur_tibia")
                                                 for f in ("stiffness", "damping")})


def test_a_scalar_actuator_field_is_scaled_too():
    cfg = _fake_cfg()
    cfg.scene.robot.actuators["hip_femur"].effort_limit_sim = 1000.0         # a plant variant may carry a float, not a dict
    sk.apply_stress_knobs(cfg, {sk.ROD_EFFORT_VAR: "0.5"})
    assert cfg.scene.robot.actuators["hip_femur"].effort_limit_sim == 500.0
    assert cfg.scene.robot.actuators["femur_tibia"].effort_limit_sim == _per(KNEE, 200.0)


def test_leg_mass_installs_the_injected_startup_event_once():
    calls = []
    cfg = _fake_cfg()
    before = copy.deepcopy(cfg)
    sk.apply_stress_knobs(cfg, {sk.LEG_MASS_VAR: "1.15"}, leg_mass_event=lambda s: calls.append(s) or ("leg", s))
    assert calls == [1.15]
    assert _diff(before, cfg) == {"events.stress_leg_mass": (ABSENT, ("leg", 1.15))}
    assert sk.LEG_MASS_EVENT == "stress_leg_mass"


def test_leg_body_expr_selects_exactly_the_plant_leg_links():
    pattern = sk.LEG_BODY_EXPR
    links = [f"{leg}_{part}" for leg in LEGS for part in ("Hip", "Femur", "Tibia", "Footpad")]
    assert all(re.fullmatch(pattern, name) for name in links) and len(links) == 24      # IsaacLab matches with fullmatch
    for name in ("FL_CamShaft", "RR_CamShaft", "body", "chassis", "Body", "FL_Body_Hip_RevoluteJoint",
                 "FL_Hip_Femur_RevoluteJoint", "FL_HipX", "pre_FL_Hip", "fl_hip", "XX_Hip", "FL_", "", "FL_Hip_Femur"):
        assert re.fullmatch(pattern, name) is None, name
    if not PLANT_USD.exists():
        pytest.skip("assets/crab.usda not in this checkout")
    names = set(re.findall(r'def \w+ "([A-Za-z_]+)"', PLANT_USD.read_text()))
    assert "body" in names and "FL_CamShaft" in names                       # the regex is tested against the real link names
    assert {n for n in names if re.fullmatch(pattern, n)} == set(links)


def test_mu_pins_the_robot_event_and_both_default_materials():
    cfg, _, diff, armed = _alone({sk.MU_VAR: "0.4"})
    assert armed == {sk.MU_VAR: 0.4}
    assert diff == {
        "events.physics_material.params.friction_range": ((0.6, 2.0), (0.4, 0.4)),
        "scene.terrain.physics_material.static_friction": (1.0, 0.4),
        "scene.terrain.physics_material.dynamic_friction": (1.0, 0.4),
        "sim.physics_material.static_friction": (1.0, 0.4),
        "sim.physics_material.dynamic_friction": (1.0, 0.4),
    }                                                           # num_buckets and the rest of the params stay
    assert cfg.events.physics_material.params["num_buckets"] == 64
    assert all(type(v) is float for v in cfg.events.physics_material.params["friction_range"])


def test_mu_tolerates_the_student_cfgs_aliased_and_a_missing_sim_material():
    cfg = _fake_cfg()
    cfg.sim.physics_material = cfg.scene.terrain.physics_material        # crab_hex_student_cfg: the sim's IS the terrain's
    sk.apply_stress_knobs(cfg, {sk.MU_VAR: "1.2"})
    assert (cfg.sim.physics_material.static_friction, cfg.sim.physics_material.dynamic_friction) == (1.2, 1.2)
    cfg = _fake_cfg()
    cfg.sim.physics_material = None                                       # optional: only the terrain side is then written
    sk.apply_stress_knobs(cfg, {sk.MU_VAR: "1.2"})
    assert cfg.scene.terrain.physics_material.static_friction == 1.2 and cfg.sim.physics_material is None


@pytest.mark.parametrize("text, hist", [("3", 8), ("0", 8), ("7", 8), ("4", 5)])
def test_act_delay_pins_the_single_delay(text, hist):
    cfg = _fake_cfg()
    cfg.actions.joint_pos.history_length = hist
    cfg, _, diff, armed = _alone({sk.ACT_DELAY_VAR: text}, cfg)
    assert armed == {sk.ACT_DELAY_VAR: int(text)}
    assert diff == {"actions.joint_pos.action_delay_steps": ([1, 1], [int(text)])}      # the action term pops element 0
    assert type(cfg.actions.joint_pos.action_delay_steps[0]) is int


@pytest.mark.parametrize("environ, spec", [
    ({sk.OBS_NOISE_VAR: "2"}, {"noise_k": 2.0}),
    ({sk.CUR_SCALE_VAR: "0.85", sk.CUR_NOISE_VAR: "0.1"}, {"cur_scale": 0.85, "cur_noise": 0.1}),
    ({sk.CUR_SCALE_VAR: "0"}, {"cur_scale": 0.0}),
    ({sk.CUR_NOISE_VAR: "0.25"}, {"cur_noise": 0.25}),
    ({sk.VXY_ZERO_VAR: "1"}, {"vxy_zero": True}),
    ({sk.VXY_NOISE_VAR: "0.1"}, {"vxy_noise": 0.1}),
    ({sk.OBS_NOISE_VAR: "1", sk.VXY_NOISE_VAR: "0.05", sk.CUR_SCALE_VAR: "1.15"},
     {"noise_k": 1.0, "vxy_noise": 0.05, "cur_scale": 1.15}),
])
def test_sensor_knobs_write_only_the_stress_obs_spec(environ, spec):
    cfg, _, diff, _armed = _alone(environ)
    assert diff == {f"stress_obs.{k}": (ABSENT, v) for k, v in spec.items()}
    assert cfg.stress_obs == spec and type(cfg.stress_obs) is dict       # a plain dict: it survives a cfg dump / deepcopy
    assert sk.STRESS_OBS_ATTR == "stress_obs"


def test_depth_lag_resizes_the_buffer_and_sets_lag_on_all_four_groups():
    cfg, _, diff, armed = _alone({sk.DEPTH_LAG_VAR: "3"})
    assert armed == {sk.DEPTH_LAG_VAR: 3}
    want = {}
    for group in ("depth_front", "depth_left", "depth_right", "depth_rear"):
        want[f"observations.{group}.depth_image.params.buffer_len"] = (2, 4)
        want[f"observations.{group}.depth_image.params.lag"] = (ABSENT, 3)
    assert diff == want                                          # resize / sensor_cfg untouched, buffer_len = lag + 1
    assert tuple(sk.DEPTH_GROUPS) == ("depth_front", "depth_left", "depth_right", "depth_rear")
    assert sk.DEPTH_TERM == "depth_image"
    for group in sk.DEPTH_GROUPS:
        params = getattr(cfg.observations, group).depth_image.params
        assert type(params["lag"]) is int and type(params["buffer_len"]) is int


def test_push_writes_its_spec_on_the_action_cfg_and_creates_no_event():
    cfg, _, diff, armed = _alone({sk.PUSH_DV_VAR: "0.35"})
    assert armed == {sk.PUSH_DV_VAR: 0.35}
    assert diff == {"actions.joint_pos.stress_push.dv": (ABSENT, 0.35),
                    "actions.joint_pos.stress_push.interval_s": (ABSENT, 10.0)}
    spec = cfg.actions.joint_pos.stress_push
    assert spec == {"dv": 0.35, "interval_s": 10.0} and type(spec) is dict         # a plain dict: it survives a cfg dump
    assert type(spec["dv"]) is float and type(spec["interval_s"]) is float
    assert sk.STRESS_PUSH_ATTR == "stress_push" and sk.STRESS_PUSH_INTERVAL_S == 10.0
    assert not hasattr(cfg.events, "stress_push")            # an event-manager term would draw from the global CUDA stream
    cfg = _fake_cfg()
    cfg.events = None                                        # the push no longer touches the event manager at all
    sk.apply_stress_knobs(cfg, {sk.PUSH_DV_VAR: "0.5"})
    assert cfg.actions.joint_pos.stress_push["dv"] == 0.5


def test_arming_every_knob_is_the_union_of_the_singles():
    levels = {var: texts[0] for var, texts in CHARTER_LEVELS.items() if var != sk.VXY_ZERO_VAR}     # VXY_NOISE, not _ZERO
    merged = {}
    for var, text in levels.items():
        diff = _alone({var: text})[2]
        assert diff and merged.keys().isdisjoint(diff), var     # every knob writes something, and nothing twice
        merged.update(diff)
    assert _alone(levels)[2] == merged                         # no knob changes what another one writes
    assert _alone({sk.VXY_ZERO_VAR: "1"})[2] == {"stress_obs.vxy_zero": (ABSENT, True)}


def _walk(cfg, dotted):
    """``(parent, key)`` of the node at ``dotted``, walking attributes and dict items."""
    *head, key = dotted.split(".")
    node = cfg
    for part in head:
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    return node, key


def _set(dotted, value):
    """A cfg mutator: set the attribute (or dict item) at ``dotted``."""
    def mutate(cfg):
        node, key = _walk(cfg, dotted)
        if isinstance(node, dict):
            node[key] = value
        else:
            setattr(node, key, value)
    return mutate


def _del(dotted):
    """A cfg mutator: delete the attribute (or dict item) at ``dotted``."""
    def mutate(cfg):
        node, key = _walk(cfg, dotted)
        if isinstance(node, dict):
            del node[key]
        else:
            delattr(node, key)
    return mutate


ACTUATORS = "scene.robot.actuators"
BROKEN = [
    pytest.param({sk.ROD_SPEED_VAR: "0.7"}, _set("actions.joint_pos", None), "actions.joint_pos", id="rod-speed-no-action"),
    pytest.param({sk.CAM_VAR: "0.8"}, _set("actions", None), "actions.joint_pos", id="cam-no-actions"),
    pytest.param({sk.ACT_DELAY_VAR: "3"}, _set("actions", None), "actions.joint_pos", id="delay-no-actions"),
    pytest.param({sk.ROD_EFFORT_VAR: "0.85"}, _del(f"{ACTUATORS}.femur_tibia"), "femur_tibia", id="effort-no-knee"),
    pytest.param({sk.KPKD_VAR: "1.25"}, _del(f"{ACTUATORS}.hip_femur"), "hip_femur", id="kpkd-no-hip"),
    pytest.param({sk.CAM_VAR: "0.8"}, _del(f"{ACTUATORS}.body_hip_yaw"), "body_hip_yaw", id="cam-no-motor"),
    pytest.param({sk.ROD_EFFORT_VAR: "0.85"}, _set("scene.robot", None), "actuators", id="effort-no-robot"),
    pytest.param({sk.KPKD_VAR: "1.25"}, _set(f"{ACTUATORS}.hip_femur.stiffness", None), "nothing to scale",
                 id="kpkd-field-none"),
    pytest.param({sk.LEG_MASS_VAR: "1.15"}, _set("events", None), "no events block", id="leg-mass-no-events"),
    pytest.param({sk.PUSH_DV_VAR: "0.35"}, _set("actions.joint_pos", None), "actions.joint_pos", id="push-no-action"),
    pytest.param({sk.MU_VAR: "0.4"}, _set("events", None), "no events block", id="mu-no-events"),
    pytest.param({sk.MU_VAR: "0.4"}, _set("events.physics_material", None), "physics_material is off", id="mu-event-off"),
    pytest.param({sk.MU_VAR: "0.4"}, _del("events.physics_material"), "physics_material is off", id="mu-event-absent"),
    pytest.param({sk.MU_VAR: "0.4"}, _set("scene.terrain.physics_material", None), "terrain.physics_material",
                 id="mu-no-ground-material"),
    pytest.param({sk.ACT_DELAY_VAR: "3"}, _set("actions.joint_pos.use_delay", False), "use_delay", id="delay-off"),
    pytest.param({sk.ACT_DELAY_VAR: "7"}, _set("actions.joint_pos.history_length", 7), "history_length",
                 id="delay-7-hist-7"),
    pytest.param({sk.ACT_DELAY_VAR: "4"}, _set("actions.joint_pos.history_length", 1), "history_length",
                 id="delay-4-hist-1"),
    pytest.param({sk.DEPTH_LAG_VAR: "3"}, _del("observations.depth_rear"), "depth_rear", id="lag-no-rear-group"),
    pytest.param({sk.DEPTH_LAG_VAR: "3"}, _set("observations.depth_left.depth_image", None), "depth_left",
                 id="lag-no-left-term"),
    pytest.param({sk.DEPTH_LAG_VAR: "2"}, _set("observations", None), "depth_front", id="lag-no-observations"),
]


@pytest.mark.parametrize("environ, mutate, match", BROKEN)
def test_a_cfg_without_the_knobs_target_raises(environ, mutate, match):
    cfg = _fake_cfg()
    mutate(cfg)
    with pytest.raises(RuntimeError, match=match):
        sk.apply_stress_knobs(cfg, environ)                    # real builders: the target check precedes any Isaac import


def test_depth_lag_error_names_every_missing_group_and_the_student_env():
    cfg = _fake_cfg()
    del cfg.observations.depth_left
    del cfg.observations.depth_rear
    with pytest.raises(RuntimeError) as exc:
        sk.apply_stress_knobs(cfg, {sk.DEPTH_LAG_VAR: "2"})
    assert "depth_left" in str(exc.value) and "depth_rear" in str(exc.value) and "student" in str(exc.value)
    assert "depth_front" not in str(exc.value)


def test_act_delay_boundary_is_history_length_minus_one():
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.ACT_DELAY_VAR: "7"})          # history 8: index -1 - 7 = -8, the oldest slot
    assert cfg.actions.joint_pos.action_delay_steps == [7]
    cfg = _fake_cfg()
    cfg.actions.joint_pos.history_length = 7
    with pytest.raises(RuntimeError, match=r"d < actions.joint_pos.history_length \(7\)"):
        sk.apply_stress_knobs(cfg, {sk.ACT_DELAY_VAR: "7"})


# =============================================================================== the real event builders
class _TermCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Entity:
    def __init__(self, name, **kw):
        self.name, self.kw = name, kw

    def __eq__(self, other):
        return isinstance(other, _Entity) and (self.name, self.kw) == (other.name, other.kw)

    __hash__ = None


@pytest.fixture
def isaac_stand_ins(monkeypatch):
    """Stand-ins for the Isaac names the lazy leg-mass builder imports; returns the ``randomize_rigid_body_mass`` sentinel."""
    mass_fn = object()
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(EventTermCfg=_TermCfg, SceneEntityCfg=_Entity))
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp.events", SimpleNamespace(randomize_rigid_body_mass=mass_fn))
    return mass_fn


def _static_check_names(without, with_defaults, params, min_argc):
    """IsaacLab ``ManagerBase._resolve_common_term_cfg``'s static check (managers/manager_base.py): the arguments after
    ``min_argc`` (``self`` + env [+ env_ids] for a class term) must be exactly the cfg params plus the defaulted ones."""
    args = list(without) + list(with_defaults)
    if len(args) <= min_argc:
        return True
    return set(args[min_argc:]) == set(list(params) + list(with_defaults))


def _static_check_ok(func, params, min_argc):
    sig = inspect.signature(func).parameters
    return _static_check_names([a for a in sig if sig[a].default is inspect.Parameter.empty],
                               [a for a in sig if sig[a].default is not inspect.Parameter.empty], params, min_argc)


def test_the_real_leg_mass_builder_scales_the_leg_links_by_a_degenerate_range(isaac_stand_ins):
    term = sk.leg_mass_event_cfg(1.15)
    assert term.func is isaac_stand_ins and term.mode == "startup"
    assert term.params == {"asset_cfg": _Entity("robot", body_names=sk.LEG_BODY_EXPR),
                           "mass_distribution_params": (1.15, 1.15),        # lo == hi: every env gets exactly the scale
                           "operation": "scale",                            # 'add' would add 1.15 kg
                           "recompute_inertia": True}
    # and the whole path: parse -> apply with the real builder -> cfg.events.stress_leg_mass
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.LEG_MASS_VAR: "1.5"})
    assert cfg.events.stress_leg_mass.params["mass_distribution_params"] == (1.5, 1.5)


#: Fallback if ``find_spec`` cannot locate the package without importing it (as ``test_isaaclab_contact_abs_tol_pin``).
FALLBACK_ISAACLAB_DIR = Path("/home/nickmagus/krabby/IsaacLab/source/isaaclab/isaaclab")


def _isaaclab_events_py():
    """``isaaclab/envs/mdp/events.py`` on disk, found without importing isaaclab (which needs Kit); None if absent."""
    try:
        spec = importlib.util.find_spec("isaaclab")
    except Exception:  # noqa: BLE001 -- a stand-in with __spec__ None in sys.modules raises ValueError here
        spec = None
    roots = list(getattr(spec, "submodule_search_locations", None) or []) + [str(FALLBACK_ISAACLAB_DIR)]
    for root in roots:
        candidate = Path(root) / "envs" / "mdp" / "events.py"
        if candidate.exists():
            return candidate
    return None


def test_the_leg_mass_params_pass_isaaclabs_static_check_for_randomize_rigid_body_mass(isaac_stand_ins):
    """A param renamed upstream would fail the env build (not silently) -- but only on the GPU box, after the queue
    launched. Parsed with ``ast``, never imported: skipped where there is no IsaacLab checkout."""
    path = _isaaclab_events_py()
    if path is None:
        pytest.skip("no IsaacLab source checkout on this machine")
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "randomize_rigid_body_mass")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    args = [a.arg for a in call.args.args]
    n_default = len(call.args.defaults)
    without, with_defaults = args[:len(args) - n_default], args[len(args) - n_default:]
    params = sk.leg_mass_event_cfg(1.15).params
    assert _static_check_names(without, with_defaults, params, min_argc=3)               # class event: self, env, env_ids
    assert not _static_check_names(without, with_defaults, dict(params, bogus=1), min_argc=3)
    assert not _static_check_names(without, with_defaults, {k: v for k, v in params.items() if k != "operation"}, 3)


# =============================================================================== D. the sensor side
N = 7
BLOCK_NAMES = ("ang_vel", "roll_pitch", "lin_vel_xy", "joint_pos_delta", "joint_vel", "contact_fill")
#: the four blocks ``noise_k`` perturbs, with the OBS_NOISE_REF row each uses
NOISE_BLOCKS = {"ang_vel": "ang_vel", "roll_pitch": "roll_pitch", "joint_pos_delta": "joint_pos", "joint_vel": "joint_vel"}


def _blocks(n=N, seed=5):
    g = torch.Generator().manual_seed(seed)
    return {
        "ang_vel": torch.randn(n, 3, generator=g),
        "roll_pitch": torch.randn(n, 2, generator=g) * 0.1,
        "lin_vel_xy": torch.randn(n, 2, generator=g),
        "joint_pos_delta": torch.randn(n, 24, generator=g) * 0.3,
        "joint_vel": torch.randn(n, 24, generator=g),
        "contact_fill": torch.rand(n, 6, generator=g) * 1.5 - 0.5,
    }


def _stress(spec, blocks=None, seed=3):
    """``(outputs by block name, the inputs)``; asserts the inputs are still equal to their clones afterwards."""
    blocks = _blocks() if blocks is None else blocks
    clones = {k: v.clone() for k, v in blocks.items()}
    out = dict(zip(BLOCK_NAMES, so.StressObs(dict(spec), "cpu", seed)(**blocks)))
    for k in blocks:
        assert torch.equal(blocks[k], clones[k]), f"{k}: the input was modified"
    return out, blocks


def test_obs_noise_is_bounded_by_the_reference_table_and_touches_only_four_blocks():
    out, blocks = _stress({"noise_k": 2.0})
    for block, ref in NOISE_BLOCKS.items():
        bound = 2.0 * so.OBS_NOISE_REF[ref]
        change = (out[block] - blocks[block]).abs().max()
        assert out[block].shape == blocks[block].shape and out[block] is not blocks[block]
        assert float(change) <= bound + 1e-6, block              # |noise| <= k * reference half-width, per element
        assert float(change) > 0.25 * bound, block               # and it is actually drawn (not all zero)
    for block in ("lin_vel_xy", "contact_fill"):
        assert torch.equal(out[block], blocks[block]), block


def test_obs_noise_is_uniform_at_the_reference_half_widths():
    blocks = _blocks(n=4000)
    out, _ = _stress({"noise_k": 1.0}, blocks)
    for block, ref in NOISE_BLOCKS.items():
        hw = so.OBS_NOISE_REF[ref]
        noise = out[block] - blocks[block]
        assert hw * 0.98 < float(noise.abs().max()) <= hw + 1e-6, block              # fills [-hw, hw], never beyond
        assert abs(float(noise.mean())) < 0.02 * hw, block
        assert float(noise.std()) == pytest.approx(hw / math.sqrt(3.0), rel=0.03), block    # U[-hw, hw]


def test_the_reference_table_is_the_charter_table():
    assert so.OBS_NOISE_REF == {"ang_vel": 0.2, "roll_pitch": 0.05, "joint_pos": 0.01, "joint_vel": 0.1}


SENSOR_SPECS = [
    pytest.param({"noise_k": 2.0, "vxy_noise": 0.1, "cur_noise": 0.1}, id="noise+vxy+cur"),
    pytest.param({"noise_k": 1.0}, id="noise"),
    pytest.param({"cur_noise": 0.25, "cur_scale": 0.85}, id="cur"),
]


@pytest.mark.parametrize("spec", SENSOR_SPECS)
def test_same_env_seed_same_draws_other_seed_other_draws(spec):
    a, _ = _stress(spec, seed=3)
    b, _ = _stress(spec, seed=3)
    c, _ = _stress(spec, seed=4)
    assert all(torch.equal(a[k], b[k]) for k in BLOCK_NAMES)
    noisy = [k for k in BLOCK_NAMES if not torch.equal(a[k], _blocks()[k])]
    assert noisy and all(not torch.equal(a[k], c[k]) for k in noisy)
    # the stream advances call to call, and a fresh object with the same seed replays the first call
    obj = so.StressObs(dict(spec), "cpu", 3)
    first = obj(**_blocks())
    second = obj(**_blocks())
    assert any(not torch.equal(x, y) for x, y in zip(first, second))
    assert all(torch.equal(x, a[k]) for k, x in zip(BLOCK_NAMES, first))


def test_the_global_torch_stream_is_untouched():
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    spec = {"noise_k": 4.0, "vxy_noise": 0.2, "cur_noise": 0.25, "cur_scale": 0.85}
    obj = so.StressObs(spec, "cpu", 9)                           # construction too
    for _ in range(3):
        obj(**_blocks())
    assert torch.equal(torch.get_rng_state(), state)             # arming a sensor knob shifts no other random stream
    after = torch.rand(4)
    torch.set_rng_state(state)
    assert torch.equal(after, torch.rand(4))


def test_the_private_seed_is_salted_and_in_range():
    assert so.stress_obs_seed(None) == so.stress_obs_seed(0)
    assert so.stress_obs_seed(18) != 18 and 0 <= so.stress_obs_seed(18) < (1 << 63)
    assert 0 <= so.stress_obs_seed(-1) < (1 << 63) and 0 <= so.stress_obs_seed((1 << 63) + 5) < (1 << 63)
    torch.Generator().manual_seed(so.stress_obs_seed(-7))        # every seed it returns is one torch accepts
    assert sp.stress_push_seed(None) == sp.stress_push_seed(0) and sp.stress_push_seed(18) != 18
    assert 0 <= sp.stress_push_seed(-1) < (1 << 63)
    torch.Generator().manual_seed(sp.stress_push_seed(-7))
    assert so.stress_obs_seed(18) != sp.stress_push_seed(18)     # the sensor and the push streams are not the same stream
    g = torch.Generator().manual_seed(18)
    h = torch.Generator().manual_seed(so.stress_obs_seed(18))
    assert not torch.equal(torch.rand(8, generator=g), torch.rand(8, generator=h))


def test_vxy_zero_zeroes_the_slot_and_nothing_else():
    out, blocks = _stress({"vxy_zero": True})
    assert out["lin_vel_xy"].shape == (N, 2) and torch.equal(out["lin_vel_xy"], torch.zeros(N, 2))
    for block in BLOCK_NAMES:
        if block != "lin_vel_xy":
            assert torch.equal(out[block], blocks[block]), block
    both, _ = _stress({"vxy_zero": True, "vxy_noise": 0.5})      # the parser refuses this pair; here the zero still wins
    assert torch.equal(both["lin_vel_xy"], torch.zeros(N, 2))


def test_vxy_noise_is_gaussian_on_the_slot_only():
    out, blocks = _stress({"vxy_noise": 0.1})
    assert not torch.equal(out["lin_vel_xy"], blocks["lin_vel_xy"])
    for block in BLOCK_NAMES:
        if block != "lin_vel_xy":
            assert torch.equal(out[block], blocks[block]), block
    big = _blocks(n=20000)
    out, _ = _stress({"vxy_noise": 0.1}, big)
    noise = out["lin_vel_xy"] - big["lin_vel_xy"]
    assert float(noise.std()) == pytest.approx(0.1, rel=0.03) and abs(float(noise.mean())) < 0.003
    assert float(noise.abs().max()) > 0.3                        # Gaussian tails (a uniform of this std tops out at 0.17)


def test_cur_scale_is_the_gain_clip_and_offset_of_the_current_proxy():
    out, blocks = _stress({"cur_scale": 1.2})
    c = blocks["contact_fill"]
    assert torch.equal(out["contact_fill"], ((c + 0.5) * 1.2).clamp(0.0, 1.5) - 0.5)
    for block in BLOCK_NAMES:
        if block != "contact_fill":
            assert torch.equal(out[block], blocks[block]), block
    anchors = dict(blocks, contact_fill=torch.tensor([[-0.5, 0.0, 0.5, 1.0, 0.25, -0.25]] * N))
    got, _ = _stress({"cur_scale": 1.2}, anchors)
    # idle (-0.5) stays idle; 0.0 -> 0.6 - 0.5; 0.5 -> 1.2 - 0.5; a saturated 1.0 (current 1.5) stays at the proxy's clip
    assert got["contact_fill"][0].tolist() == pytest.approx([-0.5, 0.1, 0.7, 1.0, 0.4, -0.2], abs=1e-6)
    low, _ = _stress({"cur_scale": 0.85}, anchors)
    assert low["contact_fill"][0].tolist() == pytest.approx([-0.5, -0.075, 0.35, 0.775, 0.1375, -0.2875], abs=1e-6)


def test_cur_scale_zero_is_honoured_not_skipped():
    out, blocks = _stress({"cur_scale": 0.0})                    # parser-legal: a dead current sensor reads idle everywhere
    assert torch.equal(out["contact_fill"], torch.full_like(blocks["contact_fill"], -0.5))
    cfg = SimpleNamespace(stress_obs={"cur_scale": 0.0}, seed=1)
    assert so.build_stress_obs(cfg, "cpu") is not None           # {"cur_scale": 0.0} is a non-empty spec


def test_cur_noise_is_gaussian_and_reclipped_to_the_proxy_range():
    n = 20000
    mid = dict(_blocks(n), contact_fill=torch.full((n, 6), 0.25))           # current 0.75: far from both clips
    out, _ = _stress({"cur_noise": 0.1}, mid)
    noise = out["contact_fill"] - 0.25
    assert float(noise.std()) == pytest.approx(0.1, rel=0.03) and abs(float(noise.mean())) < 0.003
    wide = dict(_blocks(n), contact_fill=torch.linspace(-0.5, 1.0, n * 6).reshape(n, 6))
    out, _ = _stress({"cur_noise": 0.25}, wide)                              # the CHARTER's top sigma
    lo, hi = float(out["contact_fill"].min()), float(out["contact_fill"].max())
    assert lo == -0.5 and hi == 1.0                                          # clamp(0, 1.5) - 0.5; both clips reached
    assert int((out["contact_fill"] == -0.5).sum()) > 0 and int((out["contact_fill"] == 1.0).sum()) > 0
    assert not torch.equal(out["contact_fill"], wide["contact_fill"])
    small, _ = _stress({"cur_noise": 0.1})                                   # N = 7
    assert float(small["contact_fill"].min()) >= -0.5 and float(small["contact_fill"].max()) <= 1.0


def test_cur_noise_is_added_after_the_gain():
    n = 2000
    blocks = dict(_blocks(n), contact_fill=torch.full((n, 6), 0.25))
    out, _ = _stress({"cur_scale": 0.85, "cur_noise": 0.1}, blocks)
    noise = torch.randn((n, 6), generator=torch.Generator().manual_seed(so.stress_obs_seed(3))) * 0.1   # the only draw
    gain_then_noise = ((torch.full((n, 6), 0.25) + 0.5) * 0.85 + noise).clamp(0.0, 1.5) - 0.5
    noise_then_gain = ((torch.full((n, 6), 0.25) + 0.5 + noise) * 0.85).clamp(0.0, 1.5) - 0.5
    assert torch.allclose(out["contact_fill"], gain_then_noise, rtol=0, atol=1e-6)
    assert not torch.allclose(out["contact_fill"], noise_then_gain, rtol=0, atol=1e-3)


def test_unknown_spec_keys_are_refused():
    with pytest.raises(ValueError, match="bogus"):
        so.StressObs({"bogus": 1.0}, "cpu")
    with pytest.raises(ValueError, match="noise_kk"):
        so.StressObs({"noise_k": 1.0, "noise_kk": 2.0}, "cpu")
    assert tuple(so._KEYS) == ("noise_k", "cur_scale", "cur_noise", "vxy_noise", "vxy_zero")


def test_build_stress_obs_is_none_unless_a_spec_is_present():
    assert so.build_stress_obs(SimpleNamespace(), "cpu") is None
    assert so.build_stress_obs(SimpleNamespace(stress_obs={}), "cpu") is None
    assert so.build_stress_obs(SimpleNamespace(stress_obs=None, seed=3), "cpu") is None
    obj = so.build_stress_obs(SimpleNamespace(stress_obs={"noise_k": 1.0, "vxy_zero": True}, seed=5), "cpu")
    assert isinstance(obj, so.StressObs) and obj.noise_k == 1.0 and obj.vxy_zero is True and obj.cur_scale is None
    assert obj.generator.initial_seed() == so.stress_obs_seed(5)
    unseeded = so.build_stress_obs(SimpleNamespace(stress_obs={"noise_k": 1.0}), "cpu")
    assert unseeded.generator.initial_seed() == so.stress_obs_seed(None)


def test_every_spec_the_knobs_write_builds_a_stress_obs():
    environ = {sk.OBS_NOISE_VAR: "2", sk.CUR_SCALE_VAR: "0.85", sk.CUR_NOISE_VAR: "0.1", sk.VXY_NOISE_VAR: "0.1"}
    cfg = _fake_cfg()
    cfg.seed = 4
    sk.apply_stress_knobs(cfg, environ)
    obj = so.build_stress_obs(cfg, "cpu")
    assert (obj.noise_k, obj.cur_scale, obj.cur_noise, obj.vxy_noise, obj.vxy_zero) == (2.0, 0.85, 0.1, 0.1, False)
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.VXY_ZERO_VAR: "1"})
    assert so.build_stress_obs(cfg, "cpu").vxy_zero is True
    assert sk.STRESS_OBS_ATTR == so.STRESS_OBS_ATTR


# =============================================================================== E. the additive push
STEP_DT = 0.02                    # the policy step: sim.dt 0.005 x decimation 4 (``env.step_dt``)


def _push(n=7, *, interval_s=0.1, dv=0.35, step_dt=STEP_DT, seed=3, device="cpu"):
    """A ``StressPush``; the default 0.1 s interval is 5 policy steps."""
    return sp.StressPush({"dv": dv, "interval_s": interval_s}, n, step_dt, device, seed)


@pytest.mark.parametrize("interval_s, step_dt, want", [
    (10.0, 0.02, 500), (10.0, 0.005, 2000), (0.1, 0.02, 5), (0.031, 0.02, 2), (0.029, 0.02, 1), (0.02, 0.02, 1),
    (0.001, 0.02, 1), (1.0, 1.0, 1),
])
def test_interval_steps_is_interval_over_step_dt_rounded_and_never_below_one(interval_s, step_dt, want):
    push = sp.StressPush({"dv": 0.3, "interval_s": interval_s}, 4, step_dt, "cpu", 0)
    assert push.interval_steps == want and type(push.interval_steps) is int
    assert push.timer.tolist() == [want] * 4 and push.timer.dtype == torch.long
    assert push.pushes.tolist() == [0] * 4 and push.pushes.dtype == torch.long
    assert push.dv == 0.3


def test_the_timers_fire_at_exactly_interval_steps_after_construction():
    push = _push(n=5)
    assert push.interval_steps == 5
    for step in range(1, 5):
        assert push.step() is None, step                         # nothing before the 5th step
    ids, dvel = push.step()
    assert ids.tolist() == [0, 1, 2, 3, 4] and dvel.shape == (5, 2)
    assert push.pushes.tolist() == [1] * 5
    for step in range(6, 10):
        assert push.step() is None, step
    ids, _ = push.step()                                         # and every 5 steps after
    assert ids.tolist() == [0, 1, 2, 3, 4] and push.pushes.tolist() == [2] * 5


def test_a_one_step_interval_pushes_every_step():
    push = _push(n=3, interval_s=STEP_DT)
    assert push.interval_steps == 1
    for k in range(1, 4):
        ids, dvel = push.step()
        assert ids.tolist() == [0, 1, 2] and dvel.shape == (3, 2) and push.pushes.tolist() == [k] * 3


def test_reset_restarts_only_those_envs():
    push = _push(n=6)
    for _ in range(3):
        assert push.step() is None
    push.reset(torch.tensor([1, 4]))                             # envs 1 and 4 get a full interval again
    assert push.timer.tolist() == [2, 5, 2, 2, 5, 2]
    assert push.step() is None                                   # step 4
    ids, _ = push.step()                                         # step 5: the others are due
    assert ids.tolist() == [0, 2, 3, 5] and push.pushes.tolist() == [1, 0, 1, 1, 0, 1]
    assert push.step() is None and push.step() is None           # steps 6, 7
    ids, _ = push.step()                                         # step 8 = 3 + 5: the restarted ones
    assert ids.tolist() == [1, 4] and push.pushes.tolist() == [1, 1, 1, 1, 1, 1]
    assert push.step() is None                                   # step 9
    ids, _ = push.step()                                         # step 10 = 5 + 5
    assert ids.tolist() == [0, 2, 3, 5]


def test_reset_none_restarts_every_env_and_a_late_reset_postpones_the_push_by_a_full_interval():
    push = _push(n=4)
    for _ in range(3):
        push.step()
    push.reset(None)
    assert push.timer.tolist() == [5] * 4
    for step in range(4, 8):
        assert push.step() is None, step
    ids, _ = push.step()                                         # 3 + 5 = step 8
    assert ids.tolist() == [0, 1, 2, 3]
    late = _push(n=2)
    for _ in range(4):
        late.step()                                              # both envs are one step from due
    late.reset([0])
    ids, _ = late.step()                                         # step 5: only env 1 is due
    assert ids.tolist() == [1]


@pytest.mark.parametrize("make_ids", [
    lambda: [1, 4], lambda: torch.tensor([1, 4]), lambda: torch.tensor([1, 4], dtype=torch.int32),
    lambda: (1, 4)], ids=["list", "int64", "int32", "tuple"])
def test_reset_accepts_the_env_id_forms_and_an_empty_reset_changes_nothing(make_ids):
    push = _push(n=6)
    for _ in range(3):
        push.step()
    push.reset(make_ids())
    assert push.timer.tolist() == [2, 5, 2, 2, 5, 2]
    push.reset([])
    push.reset(torch.tensor([], dtype=torch.long))
    assert push.timer.tolist() == [2, 5, 2, 2, 5, 2]


def test_pushes_counts_per_env_and_a_reset_never_clears_it():
    push = _push(n=3)
    for _ in range(5):
        push.step()
    assert push.pushes.tolist() == [1, 1, 1]
    push.reset([1])
    for _ in range(5):
        push.step()
    assert push.pushes.tolist() == [2, 2, 2]
    push.reset(None)
    assert push.pushes.tolist() == [2, 2, 2]


def test_dvel_has_norm_dv_per_row_and_uniform_headings():
    n = 4000
    push = _push(n=n, interval_s=STEP_DT, dv=0.5)
    ids, dvel = push.step()
    assert ids.dtype == torch.long and ids.tolist() == list(range(n))
    assert dvel.shape == (n, 2) and dvel.dtype == torch.float32 and str(dvel.device) == "cpu"
    assert torch.allclose(dvel.norm(dim=1), torch.full((n,), 0.5), atol=1e-6)
    angle = torch.atan2(dvel[:, 1], dvel[:, 0])
    edges = (-math.pi, -math.pi / 2, 0.0, math.pi / 2)
    quadrants = [int(((angle >= lo) & (angle < lo + math.pi / 2)).sum()) for lo in edges]
    assert all(800 < q < 1200 for q in quadrants), quadrants     # a random heading, not a fixed direction
    assert abs(float(dvel[:, 0].mean())) < 0.025 and abs(float(dvel[:, 1].mean())) < 0.025


def test_splitmix64_is_the_reference_algorithm():
    golden = 0x9E3779B97F4A7C15
    assert sp._splitmix64(0) == 0xE220A8397B1DCDAF                                       # the first three outputs of the
    assert sp._splitmix64(golden) == 0x6E789E6AA1B965F4                                  # reference splitmix64 stream (seed 0)
    assert sp._splitmix64((2 * golden) & ((1 << 64) - 1)) == 0x06C45D188009454F
    assert sp.STRESS_PUSH_SEED_SALT == int.from_bytes(b"push", "big")


def test_push_heading_is_a_pure_function_of_seed_epoch_env_and_k():
    h = sp.push_heading(3, 1, 5, 2)
    assert h == sp.push_heading(3, 1, 5, 2) and 0.0 <= h < 2.0 * math.pi
    others = [sp.push_heading(4, 1, 5, 2), sp.push_heading(3, 2, 5, 2), sp.push_heading(3, 1, 6, 2),
              sp.push_heading(3, 1, 5, 3), sp.push_heading(3, 1, 2, 5)]
    assert len({h, *others}) == 6                                # every argument matters, and (env, k) is not symmetric
    for args in ((0, 0, 0, 0), (2 ** 63, 2 ** 40, 10 ** 6, 10 ** 9), (2 ** 64 + 5, 0, 0, 0), (-1, -1, -1, -1)):
        assert 0.0 <= sp.push_heading(*args) < 2.0 * math.pi, args


def test_headings_are_uniform_along_every_argument():
    n = 4000
    axes = {"env": lambda i: (3, 1, i, 0), "k": lambda i: (3, 1, 0, i), "epoch": lambda i: (3, i, 0, 0),
            "seed": lambda i: (i, 0, 0, 0), "diagonal": lambda i: (3, 1, i, i)}
    for name, args in axes.items():
        u = sorted(sp.push_heading(*args(i)) / (2.0 * math.pi) for i in range(n))
        ks = max(max(abs(v - (i + 1) / n), abs(v - i / n)) for i, v in enumerate(u))     # Kolmogorov-Smirnov distance
        assert ks < 0.04, (name, ks)


def test_step_uses_the_hash_of_the_envs_own_seed_epoch_and_push_count():
    push = _push(n=4, interval_s=STEP_DT, dv=0.5, seed=7)        # fires every step
    assert push.seed == sp.stress_push_seed(7) and push.epoch == 0 and push.epoch_pushes == [0, 0, 0, 0]
    for k in range(3):
        ids, dvel = push.step()
        for row, e in enumerate(ids.tolist()):
            h = sp.push_heading(sp.stress_push_seed(7), 0, e, k)
            assert dvel[row].tolist() == pytest.approx([0.5 * math.cos(h), 0.5 * math.sin(h)], abs=1e-6), (k, e)
    assert push.epoch_pushes == [3, 3, 3, 3] and push.pushes.tolist() == [3, 3, 3, 3] and push.epoch == 0


def test_the_headings_come_from_no_torch_generator():
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    push = _push(n=7, interval_s=STEP_DT, seed=3)
    for _ in range(4):
        push.step()
        push.reset([2, 4])
    push.reset(None)
    push.step()
    assert torch.equal(torch.get_rng_state(), state)             # construction, steps and both kinds of reset: untouched
    assert not hasattr(push, "generator")                        # and there is no private generator either


def test_the_same_seed_gives_the_same_pushes_and_another_seed_other_ones():
    def run(seed):
        push = _push(n=7, interval_s=STEP_DT, seed=seed)
        out = []
        for _ in range(3):
            out.append(push.step()[1])
            push.reset([2, 4])
        return torch.cat(out)
    a, b, c = run(3), run(3), run(4)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(run(None), run(0)) and _push(seed=None).seed == _push(seed=0).seed == sp.stress_push_seed(0)


def test_headings_are_independent_of_what_other_envs_did():
    def fires(reset_env2_before_step):
        push = _push(n=4, seed=3)                                # a 5-step interval
        out = {}
        for step in range(1, 16):
            if step == reset_env2_before_step:
                push.reset([2])
            hit = push.step()
            if hit is not None:
                for row, e in enumerate(hit[0].tolist()):
                    out.setdefault(e, []).append((step, hit[1][row].tolist()))
        return out
    plain, shifted = fires(None), fires(8)
    for e in (0, 1, 3):
        assert plain[e] == shifted[e], e                         # a partial reset of env 2 changes nobody else's pushes
    assert [s for s, _ in plain[2]] == [5, 10, 15] and [s for s, _ in shifted[2]] == [5, 12]     # env 2's timer restarted
    assert [v for _, v in plain[2][:2]] == [v for _, v in shifted[2]]       # its k-th push is the same push, only later
    assert [s for s, _ in plain[0]] == [5, 10, 15]


@pytest.mark.parametrize("make_ids", [
    lambda: None, lambda: torch.arange(6), lambda: list(range(6)), lambda: [5, 3, 1, 0, 2, 4],
    lambda: [0, 1, 2, 3, 4, 5, 5, 0], lambda: torch.arange(6, dtype=torch.int32), lambda: tuple(range(6)),
], ids=["none", "arange", "list", "permutation", "duplicates", "int32", "tuple"])
def test_a_reset_of_every_env_opens_a_new_epoch_and_restarts_k(make_ids):
    push = _push(n=6, interval_s=STEP_DT)
    push.step()
    push.step()
    assert push.epoch == 0 and push.epoch_pushes == [2] * 6
    push.reset(make_ids())
    assert push.epoch == 1 and push.epoch_pushes == [0] * 6 and push.timer.tolist() == [1] * 6
    assert push.pushes.tolist() == [2] * 6                       # the all-time counter keeps counting
    ids, dvel = push.step()
    h0 = [sp.push_heading(push.seed, 1, e, 0) for e in range(6)]
    assert torch.allclose(dvel, torch.tensor([[0.35 * math.cos(h), 0.35 * math.sin(h)] for h in h0]), atol=1e-6)
    assert push.epoch_pushes == [1] * 6 and push.pushes.tolist() == [3] * 6


@pytest.mark.parametrize("make_ids", [
    lambda: [], lambda: [1, 4], lambda: list(range(5)), lambda: [0, 0, 1, 2, 3, 4], lambda: torch.tensor([5]),
    lambda: torch.tensor([], dtype=torch.long)], ids=["empty", "two", "five", "five-duplicated", "one", "empty-tensor"])
def test_a_partial_reset_keeps_the_epoch_and_the_push_counts(make_ids):
    push = _push(n=6, interval_s=STEP_DT)
    push.step()
    push.reset(make_ids())
    assert push.epoch == 0 and push.epoch_pushes == [1] * 6


def test_the_epoch_counts_full_resets_and_a_single_env_run_resets_fully():
    push = _push(n=5)
    for want in (1, 2, 3):
        push.reset(None)
        assert push.epoch == want
    one = _push(n=1)
    one.reset([0])
    assert one.epoch == 1 and one.epoch_pushes == [0]            # one env: any reset covers every env


def test_after_a_full_reset_the_pushes_no_longer_depend_on_the_history_before_it():
    def after_reset(history_steps):
        push = _push(n=3, interval_s=STEP_DT, seed=2)
        for _ in range(history_steps):
            push.step()
        push.reset(None)
        return push.step()[1]
    assert torch.equal(after_reset(0), after_reset(1)) and torch.equal(after_reset(1), after_reset(5))
    before = _push(n=3, interval_s=STEP_DT, seed=2).step()[1]    # epoch 0 / k = 0 is a different push from epoch 1 / k = 0
    assert not torch.equal(before, after_reset(0))


@pytest.mark.parametrize("spec, match", [
    ({"dv": 0.3, "interval_s": 10.0, "bogus": 1}, "bogus"),
    ({"dv": 0.3, "interval_s": 10.0, "interval": 5}, "interval"),
    ({"dv": 0.0, "interval_s": 10.0}, "dv"), ({"dv": -0.1, "interval_s": 10.0}, "dv"),
    ({"dv": float("nan"), "interval_s": 10.0}, "dv"), ({"dv": float("inf"), "interval_s": 10.0}, "dv"),
    ({"dv": float("-inf"), "interval_s": 10.0}, "dv"),
    ({"dv": 0.3, "interval_s": 0.0}, "interval_s"), ({"dv": 0.3, "interval_s": -1.0}, "interval_s"),
    ({"dv": 0.3, "interval_s": float("nan")}, "interval_s"), ({"dv": 0.3, "interval_s": float("inf")}, "interval_s"),
])
def test_a_bad_push_spec_is_refused_at_construction(spec, match):
    with pytest.raises(ValueError, match=match):
        sp.StressPush(spec, 4, STEP_DT, "cpu", 0)


@pytest.mark.parametrize("spec", [{"dv": 0.3}, {"interval_s": 10.0}, {"dv": 0.3, "bogus": 1}])
def test_an_incomplete_push_spec_is_refused(spec):
    with pytest.raises((KeyError, ValueError)):
        sp.StressPush(spec, 4, STEP_DT, "cpu", 0)


def test_build_stress_push_is_none_unless_the_cfg_carries_a_spec():
    assert sp.build_stress_push(SimpleNamespace(), 4, STEP_DT, "cpu") is None
    assert sp.build_stress_push(SimpleNamespace(stress_push=None), 4, STEP_DT, "cpu") is None
    assert sp.build_stress_push(SimpleNamespace(stress_push={}), 4, STEP_DT, "cpu", 3) is None
    spec = {"dv": 0.35, "interval_s": 10.0}
    push = sp.build_stress_push(SimpleNamespace(stress_push=spec), 7, STEP_DT, "cpu", 5)
    assert isinstance(push, sp.StressPush) and push.interval_steps == 500 and push.dv == 0.35
    assert push.timer.shape == (7,) and push.pushes.shape == (7,)
    assert push.seed == sp.stress_push_seed(5) and push.num_envs == 7 and push.epoch == 0
    assert sp.build_stress_push(SimpleNamespace(stress_push=spec), 7, STEP_DT, "cpu").seed == sp.stress_push_seed(None)
    spec["dv"] = 9.0                                             # the spec was copied: later edits change nothing
    assert push.dv == 0.35
    with pytest.raises(ValueError, match="dv"):
        sp.build_stress_push(SimpleNamespace(stress_push={"dv": 0.0, "interval_s": 10.0}), 4, STEP_DT, "cpu")
    assert sp.STRESS_PUSH_ATTR == sk.STRESS_PUSH_ATTR == "stress_push"


def test_the_spec_the_knob_writes_builds_a_push_with_the_documented_interval():
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.PUSH_DV_VAR: "0.35"})
    push = sp.build_stress_push(cfg.actions.joint_pos, 5, STEP_DT, "cpu", 1)
    assert push.interval_steps == round(sk.STRESS_PUSH_INTERVAL_S / STEP_DT) == 500 and push.dv == 0.35


# ---- the action term's three hooks, exec'd from parkour_actions.py itself (it imports Isaac, so it cannot be imported)
GUARD = ast.unparse(ast.parse('stress_push = getattr(self, "_stress_push", None)'))


def _action_term_methods():
    src = FWD_ACTIONS_PY.read_text()
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "CrabHexDelayedJointPositionAction")
    return src, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def _segment(src, node):
    return textwrap.dedent(ast.get_source_segment(src, node, padded=True))


def _compile_hook(src, stmts, params, namespace=None):
    code = f"def hook({', '.join(params)}):\n" + textwrap.indent("\n".join(_segment(src, s) for s in stmts), "    ") + "\n"
    ns = dict(namespace or {})
    exec(compile(code, str(FWD_ACTIONS_PY), "exec"), ns)
    return ns["hook"]


def _guarded_hook(method_name, params):
    """The method's ``stress_push`` statements -- exactly the ``getattr`` guard and the ``if stress_push is not None``
    block, nothing else in the method mentions the push -- compiled into ``hook(self, ...)``."""
    src, methods = _action_term_methods()
    stmts = [s for s in methods[method_name].body if "stress_push" in ast.unparse(s)]
    assert len(stmts) == 2, f"{method_name}: expected the guard + one if-block, found {[ast.unparse(s)[:60] for s in stmts]}"
    assign, guard = stmts
    assert ast.unparse(assign) == GUARD
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "stress_push is not None" and not guard.orelse
    return _compile_hook(src, stmts, params)


class _FakeAsset:
    """The robot as the action term's push block touches it: ``data.root_vel_w`` and ``write_root_velocity_to_sim``."""

    def __init__(self, vel):
        self.data = SimpleNamespace(root_vel_w=vel)
        self.writes = []                                         # (velocity rows, env ids) per call

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        self.writes.append((velocity.clone(), None if env_ids is None else env_ids.clone()))


def _fake_asset(n=6):
    return _FakeAsset(torch.randn(n, 6, generator=torch.Generator().manual_seed(11)))


def test_process_actions_steps_the_push_and_writes_the_xy_change_of_the_due_envs_only():
    hook = _guarded_hook("process_actions", ["self"])
    n = 6
    asset = _fake_asset(n)
    push = _push(n=n)
    term = SimpleNamespace(_stress_push=push, _asset=asset)
    original = asset.data.root_vel_w.clone()
    for step in range(1, 5):
        hook(term)
        assert asset.writes == [], step                          # one step() per process_actions call
    hook(term)
    ((rows, ids),) = asset.writes
    assert ids.tolist() == list(range(n)) and rows.shape == (n, 6)
    delta = rows - original
    assert torch.allclose(delta[:, :2].norm(dim=1), torch.full((n,), 0.35), atol=1e-6)
    assert torch.equal(delta[:, 2:], torch.zeros(n, 4))          # z and angular velocity untouched, bit for bit
    assert torch.equal(asset.data.root_vel_w, original)          # the sim's own buffer is never edited in place
    assert push.pushes.tolist() == [1] * n
    # a reset mid-interval: only the due envs are written, and the rows follow the ids
    for _ in range(2):
        hook(term)                                               # steps 6, 7
    push.reset([1, 4])                                           # envs 1 and 4 now trail the others by 2 steps
    for _ in range(2):
        hook(term)                                               # steps 8, 9
    assert len(asset.writes) == 1                                # quiet steps write nothing
    hook(term)                                                   # step 10: the others are due, 1 and 4 are not
    rows, ids = asset.writes[1]
    assert ids.tolist() == [0, 2, 3, 5]
    assert torch.allclose((rows - original[ids])[:, :2].norm(dim=1), torch.full((4,), 0.35), atol=1e-6)
    assert torch.equal((rows - original[ids])[:, 2:], torch.zeros(4, 4))


def test_process_actions_without_a_push_is_the_untouched_path():
    hook = _guarded_hook("process_actions", ["self"])
    asset = _fake_asset()
    for term in (SimpleNamespace(_asset=asset), SimpleNamespace(_stress_push=None, _asset=asset)):
        for _ in range(20):
            hook(term)
    assert asset.writes == []                                    # getattr(self, "_stress_push", None) -> None -> nothing


def test_reset_restarts_the_push_timers_of_the_reset_envs():
    hook = _guarded_hook("reset", ["self", "env_ids"])
    push = _push(n=4)
    for _ in range(3):
        push.step()
    hook(SimpleNamespace(_stress_push=push), torch.tensor([0, 3]))
    assert push.timer.tolist() == [5, 2, 2, 5]
    hook(SimpleNamespace(), torch.tensor([0]))                   # unarmed: attribute absent / None -> nothing, no error
    hook(SimpleNamespace(_stress_push=None), torch.arange(4))
    assert push.timer.tolist() == [5, 2, 2, 5]


def test_init_builds_the_push_from_the_action_cfg_with_the_policy_step_and_the_env_seed():
    src, methods = _action_term_methods()
    stmts = [s for s in ast.walk(methods["__init__"])
             if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "self._stress_push"]
    assert len(stmts) == 1
    hook = _compile_hook(src, stmts, ["self", "cfg", "env"], {"build_stress_push": sp.build_stress_push})
    env = SimpleNamespace(step_dt=STEP_DT, physics_dt=0.005, cfg=SimpleNamespace(seed=3))
    term = SimpleNamespace(num_envs=7, device="cpu")
    hook(term, SimpleNamespace(stress_push={"dv": 0.35, "interval_s": 10.0}), env)
    push = term._stress_push
    assert isinstance(push, sp.StressPush) and push.dv == 0.35
    assert push.interval_steps == 500                            # env.step_dt, the POLICY step (the physics dt would give 2000)
    assert push.timer.shape == (7,) and push.seed == sp.stress_push_seed(3)
    seedless = SimpleNamespace(step_dt=STEP_DT, physics_dt=0.005, cfg=SimpleNamespace())
    hook(term, SimpleNamespace(stress_push={"dv": 0.35, "interval_s": 10.0}), seedless)
    assert term._stress_push.seed == sp.stress_push_seed(None)
    hook(term, SimpleNamespace(), env)                           # an unarmed cfg builds nothing
    assert term._stress_push is None


def test_the_action_term_imports_the_push_from_its_module():
    src = FWD_ACTIONS_PY.read_text()
    assert "from parkour_tasks.crab_hex_forward_task.mdp.stress_push import build_stress_push" in src
    assert sp.STRESS_PUSH_ATTR in {"stress_push"} and 'getattr(self, "_stress_push", None)' in src


# =============================================================================== F. the depth term's lag
def _image_term_ns():
    """Exec the depth term's source segment (``test_task4_student.py``'s helper): the module imports Isaac at the top."""
    src = OMNI_OBS_PY.read_text()
    tree = ast.parse(src)

    def seg(name):
        node = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))][0]
        return textwrap.dedent(ast.get_source_segment(src, node))

    class _TermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.num_envs, self.device = cfg, env, env.num_envs, env.device

    ns = {"torch": torch, "F": torch.nn.functional, "ManagerTermBase": _TermBase,
          "SceneEntityCfg": object, "ParkourManagerBasedRLEnv": object}
    code = "from __future__ import annotations\n" + "\n\n".join(
        seg(n) for n in ("DEPTH_TERM_PERIOD", "_check_depth_crop", "process_depth_batch", "image_features_batched"))
    exec(compile(code, str(OMNI_OBS_PY), "exec"), ns)
    return ns


class _Sensor:
    """Every pixel of every env reads ``frame``, so an image's age can be read off its value."""

    cfg = SimpleNamespace(max_distance=2.0)

    def __init__(self, n):
        self.n, self.frame = n, 0

    @property
    def data(self):
        return SimpleNamespace(output={"distance_to_camera": torch.full((self.n, 58, 87, 1), float(self.frame))})


def _norm(frame):
    return frame / 2.0 - 0.5                                     # process_depth_batch: d / clip - 0.5 (clip 2.0)


def _image_term(ns, n=3, **params):
    """``(term, env, sensor, read)``; ``read(counter)`` calls the term the way the observation manager does."""
    sensor = _Sensor(n)
    env = SimpleNamespace(num_envs=n, device="cpu", scene={"cam": sensor}, common_step_counter=0)
    cfg = SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name="cam"), "resize": (58, 87), "buffer_len": 2, **params})
    term = ns["image_features_batched"](cfg, env)
    term.reset(None)                                             # frame 0 in every slot

    def read(counter):
        env.common_step_counter = counter
        sensor.frame = counter                                   # frame f is the image the ray-caster shows at counter f
        return term(env, **cfg.params)
    return term, env, sensor, read


def test_lag_absent_is_the_old_previous_refresh_term():
    ns = _image_term_ns()
    term, _env, _sensor, read = _image_term(ns)
    seen = []
    for counter in range(12):
        out = read(counter)
        assert torch.equal(out, term.depth_buffer[:, -2])         # exactly the pre-change return value
        seen.append(float(out[0, 0, 0]))
    assert term.lag == 1
    assert seen == [_norm(0)] * 10 + [_norm(5)] * 2              # test_task4_student's expectation, refreshes at 0, 5, 10
    _t, _e, _s, read1 = _image_term(ns, buffer_len=3, lag=1)      # an explicit lag 1 is the same term (a deeper buffer)
    _t, _e, _s, read_absent = _image_term(ns, buffer_len=3)
    assert all(torch.equal(read1(c), read_absent(c)) for c in range(30))


@pytest.mark.parametrize("lag", [1, 2, 3, 4])
@pytest.mark.parametrize("extra", [0, 2], ids=["exact-buffer", "deeper-buffer"])
def test_lag_n_returns_the_image_n_refreshes_old(lag, extra):
    ns = _image_term_ns()
    term, _env, _sensor, read = _image_term(ns, buffer_len=lag + 1 + extra, lag=lag)
    assert term.lag == lag and term.buffer_len == lag + 1 + extra
    for counter in range(41):
        out = read(counter)
        newest = 5 * (counter // 5)                              # the last refresh (frames 0, 5, 10, ...)
        want = _norm(max(0, newest - 5 * lag))                   # lag refreshes before it; the reset filled frame 0 below
        assert torch.equal(out, torch.full_like(out, want)), (counter, lag)
        assert torch.equal(out, term.depth_buffer[:, -1 - lag])
    # lag 2 by hand: frame 0 until counter 14, frame 5 from counter 15 (buffer [5, 10, 15])
    _t, _e, _s, read2 = _image_term(ns, buffer_len=3, lag=2)
    assert [float(read2(c)[0, 0, 0]) for c in range(25)] == [_norm(0)] * 15 + [_norm(5)] * 5 + [_norm(10)] * 5


def test_a_reset_row_never_shows_a_pre_reset_frame():
    ns = _image_term_ns()
    term, env, sensor, read = _image_term(ns, buffer_len=3, lag=2)
    for counter in range(15):
        read(counter)
    sensor.frame = 7
    term.reset([1])                                              # env 1 re-posed at frame 7 (counter stays 14: no refresh)
    out = read(14)                                               # same counter: still no refresh
    assert torch.all(out[1] == _norm(7)) and torch.all(out[0] == _norm(0))
    assert torch.all(term.depth_buffer[1] == _norm(7))           # every slot refilled, not just the newest
    later = []
    for counter in range(15, 41):
        later.append(float(read(counter)[1, 0, 0]))
    assert set(later) <= {_norm(7)} | {_norm(f) for f in range(15, 41, 5)}     # frames 0, 5, 10 (pre-reset) never appear
    assert later[0] == _norm(7) and later[-1] == _norm(30)


def test_lag_outside_the_buffer_is_refused():
    ns = _image_term_ns()
    for lag, buffer_len in ((0, 3), (-1, 3), (3, 3), (4, 3), (2, 2), (1, 1)):
        with pytest.raises(ValueError, match="lag|buffer_len"):
            _image_term(ns, buffer_len=buffer_len, lag=lag)
    for lag, buffer_len in ((0, 3), (3, 3), (2, 2)):
        with pytest.raises(ValueError, match=r"lag must be in \[1, buffer_len - 1\]"):
            _image_term(ns, buffer_len=buffer_len, lag=lag)
    with pytest.raises(ValueError, match="buffer_len"):           # the existing guard is still the first one for 1
        _image_term(ns, buffer_len=1)
    _image_term(ns, buffer_len=2, lag=1)                         # the shipped configuration builds


def test_the_depth_call_takes_lag_defaulted_so_the_unarmed_configs_pass_the_static_check():
    ns = _image_term_ns()
    cls = ns["image_features_batched"]
    sig = inspect.signature(cls.__call__).parameters
    assert sig["lag"].default is None and sig["crop"].default is None
    for required in ("sensor_cfg", "resize", "buffer_len"):
        assert sig[required].default is inspect.Parameter.empty
    unarmed = {"sensor_cfg": 1, "resize": (58, 87), "buffer_len": 2}                 # the shipped student depth term
    armed = dict(unarmed, buffer_len=4, lag=3)                                       # what DEPTH_LAG=3 writes
    assert _static_check_ok(cls.__call__, unarmed, min_argc=2) and _static_check_ok(cls.__call__, armed, min_argc=2)
    assert not _static_check_ok(cls.__call__, dict(unarmed, bogus=1), min_argc=2)
    assert not _static_check_ok(cls.__call__, {"sensor_cfg": 1, "resize": (58, 87)}, min_argc=2)
    # and the knob writes exactly the params the call accepts
    cfg = _fake_cfg()
    sk.apply_stress_knobs(cfg, {sk.DEPTH_LAG_VAR: "3"})
    written = cfg.observations.depth_front.depth_image.params
    assert set(written) - {"sensor_cfg", "resize"} == {"buffer_len", "lag"} and set(written) <= set(sig)


# =============================================================================== G. wiring
def test_omni_knobs_reexports_and_applies_between_the_payload_and_the_motion_knobs():
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__dr_stress_test")
    assert tuple(knobs.STRESS_VARS) == sk.STRESS_VARS
    assert set(knobs.STRESS_KNOB_PARSERS) == set(sk.STRESS_VARS)
    for attr in ("STRESS_VARS", "STRESS_KNOB_PARSERS"):
        assert attr in knobs.__all__, attr
    # the 14 individual names, STRESS_<NAME>_VAR: ``dir(knobs)`` ending in ``_VAR`` is what the driver's knob whitelist scans
    singles = sorted(a for a in dir(sk) if a.endswith("_VAR"))
    assert [getattr(sk, a) for a in singles] != [] and set(getattr(sk, a) for a in singles) == set(sk.STRESS_VARS)
    for name in singles:
        exported = f"STRESS_{name}"                             # ROD_SPEED_VAR -> STRESS_ROD_SPEED_VAR
        assert getattr(knobs, exported) == getattr(sk, name), exported
        assert exported in knobs.__all__, exported
    scan = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR") and isinstance(getattr(knobs, a), str)}
    assert set(sk.STRESS_VARS) <= scan                           # run_candidates.knob_whitelist() now sees every stress knob
    assert knobs.STRESS_PUSH_DV_VAR == "KRABBY_STRESS_PUSH_DV" and knobs.STRESS_ROD_SPEED_VAR == "KRABBY_STRESS_ROD_SPEED"
    assert knobs.STRESS_VARS is knobs._srk.STRESS_VARS and knobs.STRESS_KNOB_PARSERS is knobs._srk.STRESS_KNOB_PARSERS
    assert knobs._stress_knobs() is knobs._srk                   # one module object for every loader
    for var, parse in knobs.STRESS_KNOB_PARSERS.items():          # the driver's round-trip table parses the same way
        assert parse("") is None and parse(None) is None, var
        assert parse(CHARTER_LEVELS[var][0]) == sk.STRESS_KNOB_PARSERS[var](CHARTER_LEVELS[var][0]), var
    src = inspect.getsource(knobs.apply_omni_knobs)
    assert src.count("apply_stress_knobs") == 1
    hook = src.index("_srk.apply_stress_knobs(cfg, env)")        # the caller's environ, not os.environ
    # after the ring/field and payload knobs, before the motion knobs; the last three calls stay motion -> stroke -> frame
    assert src.index("apply_omni_ring_knobs(cfg, env)") < src.index("_pk.apply_payload_knobs(cfg, env)") < hook
    assert hook < src.index("_mk.apply_motion_knobs(cfg, env)") < src.index("_sk.apply_stroke_knobs(cfg, env)")
    assert src.index("_sk.apply_stroke_knobs(cfg, env)") < src.index("_apply_command_frame(cfg, env)")


def _top_level_imports(path: Path) -> set[str]:
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            out |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add((node.module or "").split(".")[0])
    return out


def test_the_knob_and_obs_modules_import_without_isaac():
    assert _top_level_imports(STRESS_KNOBS_PY) <= {"__future__", "math", "os"}      # omni_knobs path-loads it Isaac-free
    assert _top_level_imports(STRESS_OBS_PY) <= {"__future__", "torch"}
    assert _top_level_imports(STRESS_PUSH_PY) <= {"__future__", "math", "torch"}                  # no event manager any more
    tree = ast.parse(STRESS_KNOBS_PY.read_text())
    lazy = {}
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in ("isaaclab", "parkour_tasks"):
                lazy.setdefault(fn.name, []).append(node.module)
    assert set(lazy) == {"leg_mass_event_cfg"}, lazy                         # Isaac imports only inside the leg-mass builder


def test_no_preset_names_a_stress_knob():
    """``stress_knobs.py``: "EVERY KNOB IS DEFAULT OFF AND SET BY NO PRESET". A preset / env literal in a config module
    that spelled a stress variable would arm it in every run of that phase (docstrings and comments may mention them)."""
    names = set(sk.STRESS_VARS)
    hits = []
    for root in (OMNI / "config", TASKS / "crab_hex_forward_task" / "config"):
        for path in root.rglob("*.py"):
            if path == STRESS_KNOBS_PY:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                is_name = isinstance(node, ast.Constant) and isinstance(node.value, str)
                if is_name and node.value.split("=")[0].strip() in names:
                    hits.append(f"{path.relative_to(REPO)}: {node.value!r}")
    assert hits == []


def test_the_consumers_read_the_names_the_knobs_write():
    actions = FWD_ACTIONS_PY.read_text()
    for attr in ("stress_rod_speed_scale", "stress_cam_scale"):
        assert f'"{attr}"' in actions, attr                      # getattr(cfg, "<attr>", None) in the action term
    cfg, _, diff, _ = _alone({sk.ROD_SPEED_VAR: "0.7", sk.CAM_VAR: "0.8"})
    assert {"actions.joint_pos.stress_rod_speed_scale", "actions.joint_pos.stress_cam_scale"} <= set(diff)
    obs = FWD_OBS_PY.read_text()
    assert re.search(r"self\._stress_obs = build_stress_obs\(", obs) and "stress_obs import build_stress_obs" in obs


def test_the_observation_hook_passes_the_six_channels_in_order_and_scales_the_stressed_values():
    """The only CPU-visible evidence of the Kit-only hook: a swapped pair of channels, or a raw read left inside the
    ``torch.cat``, would make a sensor knob move the wrong slot (or nothing) with no error."""
    channels = ["ang_vel", "imu_obs", "lin_vel_xy", "joint_pos_delta", "joint_vel", "contact_fill"]
    assert list(inspect.signature(so.StressObs.__call__).parameters) == [
        "self", "ang_vel", "roll_pitch", "lin_vel_xy", "joint_pos_delta", "joint_vel", "contact_fill"]
    tree = ast.parse(FWD_OBS_PY.read_text())
    hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and isinstance(n.targets[0], ast.Tuple) and len(n.targets[0].elts) == 6
             and ast.unparse(n.value.func).endswith("stress_obs")]
    assert len(hooks) == 1
    assert [ast.unparse(t) for t in hooks[0].targets[0].elts] == channels
    assert [ast.unparse(a) for a in hooks[0].value.args] == channels
    cat = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "obs_buf"
           and ast.unparse(n.value).startswith("torch.cat(")]
    assert len(cat) == 1
    text = ast.unparse(cat[0].value)
    for stressed in ("ang_vel * 0.25", "lin_vel_xy * 2.0", "joint_vel * 0.05", "joint_pos_delta", "contact_fill",
                     "imu_obs"):
        assert stressed in text, stressed                        # the original scales, applied to the stressed values
    for raw in ("root_ang_vel_b", "root_lin_vel_b", "data.joint_vel", "_get_contact_fill"):
        assert raw not in text, raw                              # no raw sensor read bypasses the hook


def test_the_current_proxy_constants_match_the_observation_term():
    src = FWD_OBS_PY.read_text()
    assert float(re.search(r"^_CURRENT_SENSE_MAX = ([0-9.]+)", src, re.M).group(1)) == so.CURRENT_MAX == 1.5
    assert "torch.maximum(current_h, current_k) - 0.5" in src and so.CURRENT_OFFSET == 0.5   # the proxy's own clip / offset
