"""Rod-speed domain randomisation for training (rod-speed DR campaign, 2026-11-07), on CPU.

``KRABBY_DR_ROD_SPEED=lo:hi[:ramp_steps]`` (``config/crab_hex/dr_knobs.py``, default off, set by no preset) writes
``actions.joint_pos.rod_speed_dr``; the crab action term builds one ``RodSpeedDR`` (``mdp/rod_speed_dr.py``) and ALIASES its
``(N, 1)`` ``scale`` as ``_rod_speed_scale`` -- the factor the hip and knee slew limits already multiply -- and redraws it
for the reset envs only, from a private CPU generator, at each reset. Every failure mode here is silent in a real run (a
knob that is armed but inert trains at nominal and the campaign reads it as "DR did nothing"; a draw from the global stream
shifts the same-seed run's reset states), so each is pinned:

* the knob module: pure, blank is off, the forms and their types, every refusal names the variable, look-alike names are
  refused while the forward task's ``KRABBY_DR_MASS`` / ``_PUSH`` / ``_COM`` are left alone, unset writes nothing (no
  attribute, no print, no rebound container), armed writes exactly one attribute and one ``[KNOB]`` line, exclusive with the
  stress rod-speed knob (environment and attribute), the attribute name agrees between writer, builder and action term,
  ``omni_knobs`` re-exports it and applies it between the stress and motion knobs, no preset names it, the student runner
  and the eval carry do not know it;
* ``RodSpeedDR``: imports without Isaac; unarmed builds no generator and leaves the global RNG alone; spec refusals; a seed
  salted apart from the other private streams; ``scale`` is ones, never rebound; reset redraws the reset envs only, in
  env-id order, from a stream that replicates; the global RNG is untouched; the distribution is uniform on
  ``[lo_eff, hi]``; the ramp bounds; the log dict; exactly one ``[DR]`` line whose sha16 is the sha of the draws;
* the action term, exec'd from ``parkour_actions.py`` itself (it imports Isaac): ``__init__`` builds and aliases and
  refuses DR + stress; ``reset`` redraws the reset ids only, logs, draws nothing from the global stream and, knob off, is
  bitwise the reset of commit 44c8165a; ``_apply_pitch_linkage`` / ``process_actions`` are unchanged from that commit and
  slew each env at ``s_e`` x the rated speed; a scale of ones is bitwise the off path; the current-sense proxy keeps the
  rated constants.

The helpers of ``test_dr_stress_knobs.py``, ``test_rsi_cam_stage.py`` and ``test_crab_hex_omni_motion_knobs.py`` are
path-loaded under private names (those modules are plain Python; nothing real is registered under ``isaaclab.*``).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "unit"
TASKS = REPO / "parkour" / "parkour_tasks" / "parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task" / "mdp"
DR_KNOBS_PY = OMNI / "config" / "crab_hex" / "dr_knobs.py"
OMNI_KNOBS_PY = OMNI / "config" / "crab_hex" / "omni_knobs.py"
STUDENT_SPEC_PY = OMNI / "config" / "crab_hex" / "omni_student_spec.py"
RUN_PHASES_PY = OMNI / "experiments" / "tools" / "run_phases.py"
RUN_CANDIDATES_PY = OMNI / "experiments" / "tools" / "run_candidates.py"
OMNI_EVENTS_PY = OMNI / "mdp" / "omni_events.py"
RSD_PY = FWD_MDP / "rod_speed_dr.py"
ACTIONS_PY = FWD_MDP / "parkour_actions.py"
OBSERVATIONS_PY = FWD_MDP / "observations.py"

#: The commit whose ``parkour_actions.py`` predates the DR hooks (the knob-off identity reference).
PRE_CHANGE_COMMIT = "44c8165a"
VAR = "KRABBY_DR_ROD_SPEED"


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


# the existing helper modules, path-loaded under private names
dsk = _load(TESTS / "test_dr_stress_knobs.py", "test_dr_stress_knobs__rod_speed_dr")
rcs = _load(TESTS / "test_rsi_cam_stage.py", "test_rsi_cam_stage__rod_speed_dr")
mkt = _load(TESTS / "test_crab_hex_omni_motion_knobs.py", "test_crab_hex_omni_motion_knobs__rod_speed_dr")

rsd = _load(RSD_PY, "rod_speed_dr__rod_speed_dr_test")
drk = _load(DR_KNOBS_PY, "dr_knobs__rod_speed_dr_test")
knobs = _load(OMNI_KNOBS_PY, "omni_knobs__rod_speed_dr_test")
linkage = mkt.linkage
STRESS_ROD_VAR = dsk.sk.ROD_SPEED_VAR            # the name dr_knobs.py may not spell
ABSENT = dsk.ABSENT

SPEC = {"lo": 0.6, "hi": 1.35, "ramp_steps": 0}
RAMPED = {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}


def _rd(spec=None, n=8, seed=19):
    return rsd.RodSpeedDR(dict(SPEC if spec is None else spec), n, "cpu", seed)


def _replica(seed):
    return torch.Generator(device="cpu").manual_seed(rsd.rod_speed_dr_seed(seed))


def _strings_assigned(path: Path, name: str) -> set[str]:
    """Every string constant in the module-level assignment ``name = ...`` (no import: the file may need Isaac)."""
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(getattr(t, "id", None) == name for t in targets):
            return {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    raise AssertionError(f"{path.name} has no module-level {name}")


# =============================================================================== A. the knob module
def test_the_knob_module_is_pure_and_does_not_spell_the_stress_variable():
    assert dsk._top_level_imports(DR_KNOBS_PY) <= {"__future__", "math", "os"}       # omni_knobs path-loads it Isaac-free
    assert "KRABBY_STRESS_ROD_SPEED" not in DR_KNOBS_PY.read_text()                  # test_dr_stress_knobs scans every config
    assert STRESS_ROD_VAR == "KRABBY_STRESS_ROD_SPEED"                               # ... so the caller passes it in


def test_variable_names_are_the_contract():
    assert drk.ROD_SPEED_VAR == VAR and drk.DR_VARS == (VAR,)
    assert set(drk.DR_KNOB_PARSERS) == set(drk.DR_VARS) and drk.DR_KNOB_PARSERS[VAR] is drk.parse_rod_speed_dr
    assert drk.ROD_SPEED_DR_ATTR == rsd.ROD_SPEED_DR_ATTR == "rod_speed_dr"
    assert (drk.ROD_SPEED_LO_MIN, drk.ROD_SPEED_HI_MAX) == (0.3, 2.0)


def test_the_lo_floor_is_the_current_proxys_gate_fraction():
    """CHARTER 3.1: lo is floored at 0.3, where the current-proxy drive gate saturates (rod speed / rated speed)."""
    frac = float(re.search(r"^_CURRENT_SENSE_GATE_FRACTION = ([0-9.]+)", OBSERVATIONS_PY.read_text(), re.M).group(1))
    assert drk.ROD_SPEED_LO_MIN == frac


@pytest.mark.parametrize("text", [None, "", " ", "\t", "  \n"])
def test_blank_is_off(text):
    assert drk.parse_rod_speed_dr(text) is None


@pytest.mark.parametrize("text, want", [
    ("0.6:1.35:12000", (0.6, 1.35, 12000)),
    ("0.6:1.35", (0.6, 1.35, 0)),
    (" 0.6 : 1.35 : 12000 ", (0.6, 1.35, 12000)),
    ("0.6:1.35:12000.0", (0.6, 1.35, 12000)),
    ("0.6:1.35:0", (0.6, 1.35, 0)),
    ("1:1", (1.0, 1.0, 0)),
    ("1:1:5", (1.0, 1.0, 5)),
    ("0.3:2", (0.3, 2.0, 0)),
    ("0.3:2.0:1", (0.3, 2.0, 1)),
    ("1.2:1.5", (1.2, 1.5, 0)),                      # no ramp: lo above 1 is a legal (one-sided) band
    ("1.2:1.5:0", (1.2, 1.5, 0)),
])
def test_forms_parse_with_plain_types(text, want):
    spec = drk.parse_rod_speed_dr(text)
    assert spec == {"lo": want[0], "hi": want[1], "ramp_steps": want[2]}
    assert type(spec["lo"]) is float and type(spec["hi"]) is float and type(spec["ramp_steps"]) is int
    assert list(spec) == ["lo", "hi", "ramp_steps"]


@pytest.mark.parametrize("text", [
    "nan:1", "0.6:nan", "inf:inf", "0.6:inf", "-inf:1", "0.6:1.35:nan", "0.6:1.35:inf",     # not finite
    "0.6", "0.6:1:2:3", ":", "0.6:", ":1.35", "0.6:1.35:", "abc:1", "0.6:xyz", "0.6:1.35:abc", "1,5:2",   # shape / text
    "1.4:1.0",                                                                              # lo > hi
    "0.29:1", "0.0:1", "-0.5:1", "0.1:1.35:100",                                            # lo < 0.3
    "0.6:2.01", "0.6:3", "1:10",                                                            # hi > 2
    "0.6:1.35:-1", "0.6:1.35:1.5", "0.6:1.35:0.5",                                           # ramp not an integer >= 0
    "1.1:1.35:100", "1.01:2:1",                                                             # a ramp with lo > 1
    "0.6:0.9:100", "0.3:0.99:7",                                                            # a ramp with hi < 1
])
def test_refusals_name_the_variable(text):
    with pytest.raises(ValueError, match=VAR):
        drk.parse_rod_speed_dr(text)
    with pytest.raises(ValueError, match=VAR):
        drk.parse_dr_knobs({VAR: text})
    with pytest.raises(ValueError, match=VAR):
        drk.apply_dr_knobs(dsk._fake_cfg(), {VAR: text})


@pytest.mark.parametrize("name", ["KRABBY_DR_ROD_SPEEDS", "KRABBY_DR_ROD", "KRABBY_DR_RODSPEED", "KRABBY_DR_ROD_SPEED_X",
                                  "KRABBY_DR_ROD_SPED", "KRABBY_DR_ROD_SPEED2"])
@pytest.mark.parametrize("value", ["", "0.6:1.35", "junk"])
def test_look_alike_names_are_refused_whatever_their_value(name, value):
    for environ in ({name: value}, {name: value, VAR: "0.6:1.35"}):
        with pytest.raises(ValueError, match=name):
            drk.parse_dr_knobs(environ)
        with pytest.raises(ValueError, match=name):
            drk.apply_dr_knobs(dsk._fake_cfg(), environ)


def test_the_forward_tasks_dr_names_are_not_ours_and_are_untouched(capsys):
    environ = {"KRABBY_DR_MASS": "-0.5:1.5", "KRABBY_DR_PUSH": "0.5", "KRABBY_DR_COM": "0.01",
               "KRABBY_STRESS_ROD_SPEED": "0.7", "KRABBY_OMNI_SLOTS": "x"}
    assert drk.parse_dr_knobs(environ) == {}
    cfg = dsk._fake_cfg()
    ids = dsk._containers(cfg)
    assert drk.apply_dr_knobs(cfg, environ) == {}
    assert dsk._containers(cfg) == ids and capsys.readouterr().out == ""
    assert drk.parse_dr_knobs({**environ, VAR: "0.6:1.35"}) == {VAR: {"lo": 0.6, "hi": 1.35, "ramp_steps": 0}}


def test_parse_dr_knobs_arms_only_what_is_set(monkeypatch):
    assert drk.parse_dr_knobs({}) == {} and drk.parse_dr_knobs({VAR: ""}) == {} and drk.parse_dr_knobs({VAR: "  "}) == {}
    assert drk.parse_dr_knobs({VAR: "0.6:1.35:12000"}) == {VAR: {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}}
    monkeypatch.setenv(VAR, "0.7:1.2")                                # environ=None reads os.environ
    assert drk.parse_dr_knobs() == {VAR: {"lo": 0.7, "hi": 1.2, "ramp_steps": 0}}
    monkeypatch.delenv(VAR)
    assert drk.parse_dr_knobs() == {}


@pytest.mark.parametrize("environ", [{}, {VAR: ""}, {VAR: "   "}, {STRESS_ROD_VAR: "0.7"}])
def test_unset_or_blank_writes_nothing(environ, capsys):
    cfg = dsk._fake_cfg()
    before, ids = copy.deepcopy(cfg), dsk._containers(cfg)
    assert drk.apply_dr_knobs(cfg, environ, exclusive_with=(STRESS_ROD_VAR,)) == {}
    assert dsk._diff(before, cfg) == {} and dsk._containers(cfg) == ids          # no leaf, no rebound container
    assert "rod_speed_dr" not in vars(cfg.actions.joint_pos) and capsys.readouterr().out == ""
    bare = SimpleNamespace()                                                       # a cfg without the target is not touched
    assert drk.apply_dr_knobs(bare, environ) == {} and vars(bare) == {}


@pytest.mark.parametrize("text, want", [
    ("0.6:1.35:12000", {"lo": 0.6, "hi": 1.35, "ramp_steps": 12000}),
    ("0.6:1.35", {"lo": 0.6, "hi": 1.35, "ramp_steps": 0}),
])
def test_armed_writes_only_the_action_attribute_and_one_knob_line(text, want, capsys):
    cfg = dsk._fake_cfg()
    before, ids = copy.deepcopy(cfg), dsk._containers(cfg)
    armed = drk.apply_dr_knobs(cfg, {VAR: text}, exclusive_with=(STRESS_ROD_VAR,))
    assert armed == {VAR: want}
    assert set(dsk._diff(before, cfg)) == {f"actions.joint_pos.rod_speed_dr.{k}" for k in want}
    assert vars(cfg.actions.joint_pos)["rod_speed_dr"] == want
    value = vars(cfg.actions.joint_pos)["rod_speed_dr"]
    assert type(value) is dict and [type(v) for v in value.values()] == [float, float, int]    # plain scalars
    assert all(dsk._containers(cfg)[p] == i for p, i in ids.items())             # nothing existing was rebound
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.startswith("[KNOB] ") and VAR in out and "actions.joint_pos.rod_speed_dr" in out


def test_exclusive_with_the_stress_rod_speed_knob(capsys):
    for stress_value in ("0.7", "1", "0.0001"):                                    # any non-blank value, via the environment
        cfg = dsk._fake_cfg()
        with pytest.raises(ValueError) as err:
            drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35", STRESS_ROD_VAR: stress_value}, exclusive_with=(STRESS_ROD_VAR,))
        assert VAR in str(err.value) and STRESS_ROD_VAR in str(err.value)
        assert "rod_speed_dr" not in vars(cfg.actions.joint_pos)                   # refused BEFORE anything is written
    assert capsys.readouterr().out == ""
    for blank in ("", "  "):                                                       # a blank stress variable is no stress knob
        cfg = dsk._fake_cfg()
        drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35", STRESS_ROD_VAR: blank}, exclusive_with=(STRESS_ROD_VAR,))
        assert "rod_speed_dr" in vars(cfg.actions.joint_pos)
    cfg = dsk._fake_cfg()                                                          # via the attribute the stress knob wrote
    cfg.actions.joint_pos.stress_rod_speed_scale = 0.7
    with pytest.raises(ValueError, match="stress_rod_speed_scale"):
        drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35"})
    assert "rod_speed_dr" not in vars(cfg.actions.joint_pos)
    cfg = dsk._fake_cfg()                                                          # the real stress applier's output
    dsk.sk.apply_stress_knobs(cfg, {STRESS_ROD_VAR: "0.7"})
    with pytest.raises(ValueError, match="stress_rod_speed_scale"):
        drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35"})
    cfg = dsk._fake_cfg()
    dsk.sk.apply_stress_knobs(cfg, {dsk.sk.CAM_VAR: "0.8"})                        # another stress knob is no clash
    drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35"})
    assert vars(cfg.actions.joint_pos)["rod_speed_dr"]["lo"] == 0.6
    plain = dsk._fake_cfg()                                                        # DR off: nothing to refuse
    assert drk.apply_dr_knobs(plain, {STRESS_ROD_VAR: "0.7"}, exclusive_with=(STRESS_ROD_VAR,)) == {}


def test_a_cfg_without_the_target_raises_when_armed():
    with pytest.raises(RuntimeError, match="actions.joint_pos"):
        drk.apply_dr_knobs(SimpleNamespace(), {VAR: "0.6:1.35"})
    with pytest.raises(RuntimeError, match="actions.joint_pos"):
        drk.apply_dr_knobs(SimpleNamespace(actions=SimpleNamespace()), {VAR: "0.6:1.35"})


def test_the_attribute_name_agrees_between_writer_builder_and_action_term(capsys):
    cfg = dsk._fake_cfg()
    drk.apply_dr_knobs(cfg, {VAR: "0.6:1.35:12000"})
    capsys.readouterr()
    assert rsd.ROD_SPEED_DR_ATTR in vars(cfg.actions.joint_pos)                    # the writer's key is the builder's
    built = rsd.build_rod_speed_dr(cfg.actions.joint_pos, 5, "cpu", 3)             # ... end to end, knob -> builder
    assert isinstance(built, rsd.RodSpeedDR) and (built.lo, built.hi, built.ramp_steps) == (0.6, 1.35, 12000)
    src = ACTIONS_PY.read_text()
    assert "from parkour_tasks.crab_hex_forward_task.mdp.rod_speed_dr import build_rod_speed_dr" in src
    assert 'getattr(action_cfg, ROD_SPEED_DR_ATTR, None)' in RSD_PY.read_text()
    assert re.search(r"self\._rod_speed_dr = build_rod_speed_dr\(cfg, ", src)       # the consumer builds from the action cfg


def test_omni_knobs_reexports_the_names_and_the_one_module_object():
    assert tuple(knobs.DR_VARS) == drk.DR_VARS and set(knobs.DR_KNOB_PARSERS) == set(drk.DR_VARS)
    assert knobs.DR_ROD_SPEED_VAR == VAR
    for attr in ("DR_VARS", "DR_KNOB_PARSERS", "DR_ROD_SPEED_VAR"):
        assert attr in knobs.__all__, attr
    assert knobs.DR_VARS is knobs._drk.DR_VARS and knobs.DR_KNOB_PARSERS is knobs._drk.DR_KNOB_PARSERS
    assert knobs._dr_knobs() is knobs._drk                                          # one module object for every loader
    assert knobs.DR_KNOB_PARSERS[VAR]("") is None and knobs.DR_KNOB_PARSERS[VAR]("0.6:1.35") == SPEC
    scan = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR") and isinstance(getattr(knobs, a), str)}
    assert VAR in scan                                                              # what run_candidates.knob_whitelist scans


def test_the_ladder_whitelist_knows_the_knob():
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__rod_speed_dr_test")
    assert VAR in rc.knob_whitelist()
    assert "KRABBY_DR_ROD_SPED" not in rc.knob_whitelist()                          # a typo is still a typo


def test_omni_knobs_applies_the_dr_knobs_between_the_stress_and_motion_knobs():
    fn = next(n for n in ast.parse(OMNI_KNOBS_PY.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "apply_omni_knobs")
    calls = [ast.unparse(s) for s in fn.body if isinstance(s, ast.Expr)]
    stress = calls.index("_srk.apply_stress_knobs(cfg, env)")
    # the call passes the stress variable's NAME in (config modules may not spell it) and the caller's environ
    assert calls[stress + 1] == "_drk.apply_dr_knobs(cfg, env, exclusive_with=(_srk.ROD_SPEED_VAR,))"
    assert calls[stress + 2] == "_mk.apply_motion_knobs(cfg, env)"
    assert calls[-3:] == ["_mk.apply_motion_knobs(cfg, env)", "_sk.apply_stroke_knobs(cfg, env)",
                          "_apply_command_frame(cfg, env)"]                        # the last three stay motion, stroke, frame
    assert calls.index("_pk.apply_payload_knobs(cfg, env)") < stress
    assert "".join(calls).count("apply_dr_knobs") == 1
    assert knobs._srk.ROD_SPEED_VAR == STRESS_ROD_VAR


def test_no_preset_names_the_dr_knob():
    """Default off AND set by no preset: a preset literal in a config module would arm the DR in every run of that phase
    (docstrings and comments may mention it)."""
    hits = []
    for root in (OMNI / "config", TASKS / "crab_hex_forward_task" / "config"):
        for path in root.rglob("*.py"):
            if path == DR_KNOBS_PY:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value.split("=")[0].strip().startswith("KRABBY_DR_ROD"):
                    hits.append(f"{path.relative_to(REPO)}: {node.value!r}")
    assert hits == []


def test_the_student_runner_and_the_eval_carry_do_not_know_the_knob():
    """Training-only: it must not reach an eval row (CHARTER 2) and the student runner has nothing to refuse or apply."""
    refused = _strings_assigned(STUDENT_SPEC_PY, "STUDENT_RUNNER_KNOBS_REFUSED")
    applied = _strings_assigned(STUDENT_SPEC_PY, "STUDENT_RUNNER_KNOBS_APPLIED") | {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"}
    carry = _strings_assigned(RUN_PHASES_PY, "EVAL_CARRY")
    assert refused and carry, "the AST reads found nothing"
    for table in (refused, applied, carry):
        assert VAR not in table and not any(v.startswith("KRABBY_DR_ROD") for v in table)


# =============================================================================== B. RodSpeedDR
def test_the_module_imports_without_isaac():
    assert dsk._top_level_imports(RSD_PY) <= {"__future__", "hashlib", "math", "torch"}


def test_unarmed_builds_no_generator_and_leaves_the_global_rng_alone(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a torch.Generator was built for an unarmed cfg")
    monkeypatch.setattr(torch, "Generator", boom)
    torch.manual_seed(5)
    state = torch.get_rng_state().clone()
    for cfg in (SimpleNamespace(), SimpleNamespace(rod_speed_dr=None), SimpleNamespace(rod_speed_dr={}),
                SimpleNamespace(rod_speed_dr=False), SimpleNamespace(rod_speed_dr=0)):
        assert rsd.build_rod_speed_dr(cfg, 8, "cpu", 3) is None
    assert torch.equal(torch.get_rng_state(), state)


def test_armed_builds_one_private_cpu_generator(monkeypatch):
    made = []
    real = torch.Generator

    def counting(*a, **k):
        made.append((a, k))
        return real(*a, **k)
    monkeypatch.setattr(torch, "Generator", counting)
    rd = rsd.build_rod_speed_dr(SimpleNamespace(rod_speed_dr=dict(SPEC)), 8, "cpu", 3)
    assert isinstance(rd, rsd.RodSpeedDR) and len(made) == 1
    assert rd.gen.device.type == "cpu" and rd.seed == rsd.rod_speed_dr_seed(3) and rd.num_envs == 8


def test_build_copies_the_spec():
    spec = dict(SPEC)
    rd = rsd.build_rod_speed_dr(SimpleNamespace(rod_speed_dr=spec), 4, "cpu", 0)
    spec["lo"] = 0.9
    assert rd.lo == 0.6


@pytest.mark.parametrize("spec", [
    {"lo": 0.6, "hi": 1.35, "ramp_steps": 0, "extra": 1},                           # unknown key
    {"lo": 0.6, "hi": 1.35},                                                        # missing key
    {"lo": 0.6, "ramp_steps": 0},
    {},
    {"lo": 0.6, "hi": 1.35, "ramp_steps": -1},
    {"lo": 0.6, "hi": 1.35, "ramp_steps": 1.5},
    {"lo": 0.6, "hi": 1.35, "ramp_steps": True},
    {"lo": 1.4, "hi": 1.0, "ramp_steps": 0},                                        # lo > hi
    {"lo": 0.0, "hi": 1.0, "ramp_steps": 0},
    {"lo": -0.1, "hi": 1.0, "ramp_steps": 0},
    {"lo": float("nan"), "hi": 1.0, "ramp_steps": 0},
    {"lo": 0.6, "hi": float("inf"), "ramp_steps": 0},
    {"lo": 1.1, "hi": 1.35, "ramp_steps": 100},                                     # a ramp starts at 1.0
    {"lo": 0.6, "hi": 0.9, "ramp_steps": 100},
])
def test_spec_refusals(spec):
    with pytest.raises(ValueError):
        rsd.RodSpeedDR(spec, 4, "cpu", 0)


def test_the_seed_is_salted_and_distinct_from_the_other_private_streams():
    assert rsd.ROD_SPEED_DR_SEED_SALT == int.from_bytes(b"rodspd", "big")
    assert rsd.rod_speed_dr_seed(None) == rsd.rod_speed_dr_seed(0) == rsd.ROD_SPEED_DR_SEED_SALT
    assert rsd.rod_speed_dr_seed(19) == 19 + rsd.ROD_SPEED_DR_SEED_SALT
    assert all(0 <= rsd.rod_speed_dr_seed(s) < (1 << 63) for s in (None, 0, 1, 19, 2 ** 40, 2 ** 63 + 5))
    payload_salt = int(re.search(r"^PAYLOAD_SEED_SALT = (0x[0-9A-Fa-f]+)", OMNI_EVENTS_PY.read_text(), re.M).group(1), 16)
    salts = {"rod": rsd.ROD_SPEED_DR_SEED_SALT, "push": dsk.sp.STRESS_PUSH_SEED_SALT,
             "obs": dsk.so.STRESS_OBS_SEED_SALT, "payload": payload_salt}
    assert len(set(salts.values())) == 4
    for env_seed in (None, 0, 1, 3, 19, 20, 1000):
        seeds = {rsd.rod_speed_dr_seed(env_seed), dsk.sp.stress_push_seed(env_seed), dsk.so.stress_obs_seed(env_seed),
                 (0 if env_seed is None else env_seed) + payload_salt}
        assert len(seeds) == 4, env_seed                                            # no two streams share a seed
    assert len({rsd.rod_speed_dr_seed(s) for s in range(64)}) == 64                  # distinct env seeds, distinct streams
    assert _rd(seed=3).seed == rsd.rod_speed_dr_seed(3) and _rd(seed=None).seed == rsd.rod_speed_dr_seed(0)


def test_scale_is_ones_and_never_rebound():
    rd = _rd(n=16)
    scale, ptr = rd.scale, rd.scale.data_ptr()
    assert scale.shape == (16, 1) and scale.dtype == torch.float32 and bool((scale == 1.0).all())
    for ids in ([3, 4], torch.tensor([0, 15]), None, [], torch.tensor([7])):
        rd.reset(ids, step=0)
        assert rd.scale is scale and rd.scale.data_ptr() == ptr
    assert scale.shape == (16, 1)
    assert not bool((scale == 1.0).all())                                           # ... and it did change in place


def test_reset_redraws_only_the_reset_envs():
    rd = _rd(n=32)
    rd.reset(None, 0)
    first = rd.scale.clone()
    rd.reset([1, 4, 7], 0)
    keep = [i for i in range(32) if i not in (1, 4, 7)]
    assert torch.equal(rd.scale[keep], first[keep])                                 # bit-identical
    assert not bool((rd.scale[[1, 4, 7]] == first[[1, 4, 7]]).any())                # all three moved
    again = rd.scale.clone()
    rd.reset(torch.tensor([31]), 0)
    assert torch.equal(rd.scale[:31], again[:31]) and rd.scale[31, 0] != again[31, 0]
    snap = rd.scale.clone()
    rd.reset([], 0)                                                                 # an empty reset changes nothing
    assert torch.equal(rd.scale, snap)


@pytest.mark.parametrize("make_ids", [
    lambda: [5, 2, 9],
    lambda: torch.tensor([5, 2, 9]),
    lambda: [9, 5, 2],
    lambda: [2, 5, 9],
    lambda: (9, 2),
    lambda: None,
], ids=["list", "tensor", "reversed", "ascending", "tuple", "all"])
def test_the_private_stream_matches_a_replica_in_env_id_order(make_ids):
    n, seed = 12, 19
    rd = _rd(n=n, seed=seed)
    g = _replica(seed)
    for step_ids in (make_ids(), [11, 0], make_ids()):                              # the stream CONTINUES across resets
        ids = list(range(n)) if step_ids is None else [int(i) for i in step_ids]
        u = torch.rand(len(ids), generator=g)
        want = SPEC["lo"] + (SPEC["hi"] - SPEC["lo"]) * u
        rd.reset(step_ids, 0)
        assert torch.equal(rd.scale[ids, 0], want)                                  # the i-th draw goes to ids[i]
        assert torch.equal(rd.scale[:, 0][ids], want)
    if make_ids() is not None:                                                      # the order matters: ids[0] gets the 1st draw
        a, b = _rd(n=n, seed=seed), _rd(n=n, seed=seed)
        a.reset([2, 9], 0)
        b.reset([9, 2], 0)
        assert a.scale[2, 0] == b.scale[9, 0] and a.scale[9, 0] == b.scale[2, 0] and a.scale[2, 0] != a.scale[9, 0]


def test_same_seed_same_draws_other_seed_other_draws():
    a, b, c = _rd(seed=4), _rd(seed=4), _rd(seed=5)
    for rd in (a, b, c):
        rd.reset(None, 0)
    assert torch.equal(a.scale, b.scale) and not torch.equal(a.scale, c.scale)


def test_the_global_rng_is_untouched():
    torch.manual_seed(1)
    state = torch.get_rng_state().clone()
    rd = _rd(n=64, spec=RAMPED)
    for step, ids in ((0, None), (6000, [1, 2, 3]), (12000, torch.arange(10)), (99999, [])):
        rd.reset(ids, step)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(1)
    expect = torch.rand(5)
    torch.manual_seed(1)
    rd.reset(None, 0)
    assert torch.equal(torch.rand(5), expect)


def test_the_draws_are_uniform_on_the_band():
    n = 65536
    rd = _rd(n=n, seed=19)
    rd.reset(None, 0)
    s = rd.scale[:, 0].double()
    lo, hi = SPEC["lo"], SPEC["hi"]
    assert float(s.min()) >= lo and float(s.max()) <= hi
    assert float(s.min()) < lo + 1e-3 and float(s.max()) > hi - 1e-3                # it uses the whole band
    sigma = (hi - lo) / math.sqrt(12.0) / math.sqrt(n)
    assert abs(float(s.mean()) - (lo + hi) / 2.0) < 4.0 * sigma
    cdf = ((s.sort().values - lo) / (hi - lo)).numpy()
    i = np.arange(1, n + 1)
    d = max(float(np.max(i / n - cdf)), float(np.max(cdf - (i - 1) / n)))
    assert d < 1.63 / math.sqrt(n)                                                  # Kolmogorov-Smirnov, 1 % level


def test_ramp_bounds():
    r = 12000
    assert rsd.rod_speed_bounds(0, 0.6, 1.35, r) == (1.0, 1.35)
    assert rsd.rod_speed_bounds(r // 2, 0.6, 1.35, r) == pytest.approx((0.8, 1.35))
    assert rsd.rod_speed_bounds(r, 0.6, 1.35, r) == pytest.approx((0.6, 1.35))
    assert rsd.rod_speed_bounds(2 * r, 0.6, 1.35, r) == pytest.approx((0.6, 1.35))     # stays at lo
    assert rsd.rod_speed_bounds(10 ** 9, 0.6, 1.35, r) == pytest.approx((0.6, 1.35))
    assert rsd.rod_speed_bounds(-5, 0.6, 1.35, r) == (1.0, 1.35)                    # a negative step is step 0
    lows = [rsd.rod_speed_bounds(k, 0.6, 1.35, r)[0] for k in range(0, 2 * r + 1, 500)]
    assert all(b <= a + 1e-12 for a, b in zip(lows, lows[1:]))                      # linear, monotone, only the low edge moves
    assert {rsd.rod_speed_bounds(k, 0.6, 1.35, r)[1] for k in range(0, 2 * r, 977)} == {1.35}
    for step in (0, 1, r // 2, r, 10 ** 9):                                         # ramp 0 = no ramp
        assert rsd.rod_speed_bounds(step, 0.6, 1.35, 0) == (0.6, 1.35)
    assert rsd.rod_speed_bounds(5, 0.6, 1.35, 10) == pytest.approx((0.8, 1.35))
    assert _rd(spec=RAMPED).bounds(r // 2) == pytest.approx((0.8, 1.35))
    assert _rd().bounds(7) == (0.6, 1.35)


def test_the_ramp_moves_the_drawn_band():
    n, r = 4096, 12000
    for step, (lo_eff, hi) in ((0, (1.0, 1.35)), (r // 2, (0.8, 1.35)), (r, (0.6, 1.35)), (2 * r, (0.6, 1.35))):
        rd = _rd(spec=RAMPED, n=n)
        rd.reset(None, step)
        s = rd.scale[:, 0]
        assert float(s.min()) >= lo_eff - 1e-6 and float(s.max()) <= hi + 1e-6, step
        assert float(s.min()) < lo_eff + 0.01 and float(s.max()) > hi - 0.01, step      # ... and fills the band
    rd = _rd(spec=SPEC, n=n)                                                        # ramp 0: lo from the very first reset
    rd.reset(None, 0)
    assert float(rd.scale.min()) < 0.62


def test_the_log_dict():
    rd = _rd(spec=RAMPED, n=8)
    rd.reset(None, 0)
    log = rd.reset([1, 3], 6000)
    assert tuple(log) == rsd.LOG_KEYS == ("DR/rod_speed_lo", "DR/rod_speed_mean", "DR/rod_speed_min", "DR/rod_speed_max")
    for key, value in log.items():
        assert isinstance(value, torch.Tensor) and value.ndim == 0 and value.dtype == torch.float32, key
    col = rd.scale[:, 0]
    assert float(log["DR/rod_speed_lo"]) == pytest.approx(0.8)                      # the low bound in force at that step
    # mean / min / max are over ALL N envs (the dose in force), not just the ones this reset redrew
    assert torch.equal(log["DR/rod_speed_mean"], col.mean())
    assert torch.equal(log["DR/rod_speed_min"], col.min()) and torch.equal(log["DR/rod_speed_max"], col.max())
    assert float(log["DR/rod_speed_min"]) <= float(rd.scale[[1, 3]].min())
    unreset = torch.tensor([i for i in range(8) if i not in (1, 3)])
    assert float(log["DR/rod_speed_max"]) >= float(rd.scale[unreset].max()) - 1e-12
    empty = rd.reset([], 12000)                                                     # an empty reset still logs the state in force
    assert float(empty["DR/rod_speed_lo"]) == pytest.approx(0.6) and torch.equal(empty["DR/rod_speed_min"], col.min())


def test_exactly_one_dr_line_on_the_first_reset_whose_sha_is_the_sha_of_the_draws(capsys):
    rd = _rd(n=8, seed=19)
    assert capsys.readouterr().out == ""                                            # construction prints nothing
    ids = [6, 1, 4]
    rd.reset(ids, 0)
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.count("[DR]") == 1 and out.startswith("[DR] rod_speed ")
    kv = dict(re.findall(r"(\w+)=(\S+)", out))
    drawn = rd.scale[ids, 0].numpy().astype("<f4")
    assert kv["sha16"] == hashlib.sha256(drawn.tobytes()).hexdigest()[:16]          # the sha of the draws, in id order
    assert kv["n"] == "3" and kv["step"] == "0" and kv["seed"] == str(rsd.rod_speed_dr_seed(19))
    assert kv["module"] == str(RSD_PY) and f"bounds=[{SPEC['lo']:g}, {SPEC['hi']:g}]" in out
    assert re.search(r"min/mean/max [0-9.]+/[0-9.]+/[0-9.]+", out)
    for _ in range(3):
        rd.reset(None, 100)
        rd.reset([], 100)
    assert capsys.readouterr().out == ""                                            # later resets are silent
    other = _rd(n=8, seed=19)                                                       # ... and each instance prints its own
    other.reset(ids, 0)
    again = capsys.readouterr().out
    assert again == out                                                             # same seed, same ids: the same line
    third = _rd(n=8, seed=20)
    third.reset(ids, 0)
    assert re.search(r"sha16=(\w+)", capsys.readouterr().out).group(1) != kv["sha16"]


# =============================================================================== C. the action term's hooks
def _init_hook():
    """The ``rod_scale`` / ``_rod_speed_scale`` / ``_rod_speed_dr`` statements of ``__init__``, compiled into
    ``hook(self, cfg, env)`` (the class imports Isaac, so it cannot be built)."""
    src, methods = dsk._action_term_methods()
    body = methods["__init__"].body
    start = next(i for i, s in enumerate(body) if ast.unparse(s).startswith("rod_scale = getattr(cfg"))
    end = next(i for i, s in enumerate(body) if isinstance(s, ast.If) and "_rod_speed_dr" in ast.unparse(s.test))
    stmts = body[start:end + 1]
    assert [ast.unparse(s).split("=")[0].strip().split(" ")[0] for s in stmts[:3]] == [
        "rod_scale", "self._rod_speed_scale", "self._rod_speed_dr"], "the init hook moved"
    assert len(stmts) == 4
    return dsk._compile_hook(src, stmts, ["self", "cfg", "env"], {"build_rod_speed_dr": rsd.build_rod_speed_dr})


def _init(cfg_attrs, seed=3, n=7, **env_extra):
    hook = _init_hook()
    env_cfg = SimpleNamespace() if seed is ABSENT else SimpleNamespace(seed=seed)
    term = SimpleNamespace(num_envs=n, device="cpu")
    hook(term, SimpleNamespace(**cfg_attrs), SimpleNamespace(cfg=env_cfg, **env_extra))
    return term


def test_init_builds_and_aliases_the_scale():
    term = _init({"rod_speed_dr": dict(RAMPED)}, seed=3, n=7)
    assert isinstance(term._rod_speed_dr, rsd.RodSpeedDR)
    assert term._rod_speed_scale is term._rod_speed_dr.scale                        # the SAME tensor, not a copy
    assert term._rod_speed_scale.shape == (7, 1)
    assert term._rod_speed_dr.seed == rsd.rod_speed_dr_seed(3)                      # the env seed (env.cfg.seed)
    assert (term._rod_speed_dr.lo, term._rod_speed_dr.hi, term._rod_speed_dr.ramp_steps) == (0.6, 1.35, 12000)
    assert term._rod_speed_dr.num_envs == 7 and term._rod_speed_dr.device == "cpu"
    ptr = term._rod_speed_scale.data_ptr()
    term._rod_speed_dr.reset([0, 3], 0)                                             # the alias stays live through resets
    assert term._rod_speed_scale is term._rod_speed_dr.scale and term._rod_speed_scale.data_ptr() == ptr
    assert float(term._rod_speed_scale[0, 0]) != 1.0 and float(term._rod_speed_scale[1, 0]) == 1.0
    seedless = _init({"rod_speed_dr": dict(SPEC)}, seed=ABSENT)
    assert seedless._rod_speed_dr.seed == rsd.rod_speed_dr_seed(None)
    assert _init({"rod_speed_dr": dict(SPEC)}, seed=None)._rod_speed_dr.seed == rsd.rod_speed_dr_seed(0)


def test_init_unarmed_leaves_none_or_the_stress_float():
    for attrs in ({}, {"rod_speed_dr": None}, {"rod_speed_dr": {}}):
        term = _init(attrs)
        assert term._rod_speed_dr is None and term._rod_speed_scale is None
    term = _init({"stress_rod_speed_scale": 0.7})
    assert term._rod_speed_dr is None and term._rod_speed_scale == 0.7 and type(term._rod_speed_scale) is float


def test_init_refuses_dr_together_with_the_stress_scale():
    with pytest.raises(ValueError, match="stress_rod_speed_scale"):
        _init({"rod_speed_dr": dict(SPEC), "stress_rod_speed_scale": 0.7})
    with pytest.raises(ValueError, match="rod_speed_dr"):
        _init({"rod_speed_dr": dict(SPEC), "stress_rod_speed_scale": 1.0})          # even the nominal stress value


def _reset_term(spec=None, *, n=6, env=ABSENT, attach=True, cls=None):
    cls = cls or rcs._term_class(ACTIONS_PY.read_text(), ("reset",))
    t = rcs._make_term(cls=cls, n=n)
    if attach:
        t._rod_speed_dr = _rd(spec, n=n, seed=19)
        t._rod_speed_scale = t._rod_speed_dr.scale
    if env is not ABSENT:
        t._env = env
    return t


def _env(step=0, extras=None):
    return SimpleNamespace(common_step_counter=step, extras={"log": {}} if extras is None else extras)


def test_reset_redraws_only_the_reset_ids_and_logs():
    t = _reset_term(RAMPED, env=_env(6000))
    t.reset(None)                                                                   # all envs, ascending
    first = t._rod_speed_dr.scale.clone()
    t._env.extras["log"].clear()
    t.reset(torch.tensor([1, 4]))
    scale = t._rod_speed_dr.scale
    keep = [0, 2, 3, 5]
    assert torch.equal(scale[keep], first[keep]) and not bool((scale[[1, 4]] == first[[1, 4]]).any())
    assert t._rod_speed_scale is scale                                              # the slew factor is this very tensor
    log = t._env.extras["log"]
    assert set(log) == set(rsd.LOG_KEYS) and float(log["DR/rod_speed_lo"]) == pytest.approx(0.8)   # step 6000 of 12000
    assert float(log["DR/rod_speed_min"]) == float(scale.min()) and float(log["DR/rod_speed_max"]) == float(scale.max())
    t.reset([5])                                                                    # a plain list works too
    assert torch.equal(t._rod_speed_dr.scale[:5], scale[:5])


def test_reset_draws_replicate_the_private_stream_in_env_id_order():
    t = _reset_term(SPEC, env=_env())
    g = _replica(19)
    for ids in ([4, 0, 2], None, [3]):
        n_ids = 6 if ids is None else len(ids)
        want = SPEC["lo"] + (SPEC["hi"] - SPEC["lo"]) * torch.rand(n_ids, generator=g)
        t.reset(torch.tensor(ids) if ids is not None else None)
        assert torch.equal(t._rod_speed_dr.scale[list(range(6)) if ids is None else ids, 0], want)


def test_reset_draws_nothing_from_the_global_stream():
    armed, plain = _reset_term(RAMPED, env=_env(100)), _reset_term(attach=False)
    torch.manual_seed(21)
    armed.reset(torch.tensor([0, 2, 3]))
    after_armed = torch.rand(4)
    torch.manual_seed(21)
    plain.reset(torch.tensor([0, 2, 3]))
    assert torch.equal(after_armed, torch.rand(4)), "arming the DR changed the global-stream consumption of reset"
    # the rest of the reset state is the same too: the clock phases come from the same global draws
    assert torch.equal(armed.clock_phase[[0, 2, 3]], plain.clock_phase[[0, 2, 3]])


def test_reset_without_a_log_dict_still_redraws():
    for env in (SimpleNamespace(common_step_counter=0), SimpleNamespace(common_step_counter=0, extras=None),
                SimpleNamespace(common_step_counter=0, extras={}), SimpleNamespace(common_step_counter=0, extras={"log": None}),
                SimpleNamespace(common_step_counter=0, extras=[1]), SimpleNamespace()):
        t = _reset_term(SPEC, env=env)
        t.reset(torch.tensor([0, 1]))
        assert float(t._rod_speed_dr.scale[0, 0]) != 1.0 and float(t._rod_speed_dr.scale[2, 0]) == 1.0
    t = _reset_term(RAMPED, env=SimpleNamespace())                                  # no counter: step 0, lo_eff = 1.0
    t.reset(None)
    assert float(t._rod_speed_dr.scale.min()) >= 1.0
    t = _reset_term(RAMPED, env=_env(0, extras={"other": 1, "log": {"keep": 2.0}}))
    t.reset(None)
    assert t._env.extras["other"] == 1 and t._env.extras["log"]["keep"] == 2.0       # merged, not replaced
    assert set(t._env.extras["log"]) == set(rsd.LOG_KEYS) | {"keep"}


def test_an_unarmed_reset_never_touches_the_env():
    for attrs in ({}, {"_rod_speed_dr": None}):
        t = _reset_term(attach=False)                                               # no ``_env`` attribute at all
        for k, v in attrs.items():
            setattr(t, k, v)
        assert not hasattr(t, "_env")
        t.reset(torch.tensor([0, 1]))
        t.reset(None)


def test_the_reset_hook_is_one_guard_and_leaves_the_push_hook_alone():
    src, methods = dsk._action_term_methods()
    reset = methods["reset"]
    mine = [s for s in reset.body if "rod_speed_dr" in ast.unparse(s)]
    assert [type(s).__name__ for s in mine] == ["Assign", "If"]
    assert ast.unparse(mine[0]) == ast.unparse(ast.parse('rod_speed_dr = getattr(self, "_rod_speed_dr", None)'))
    assert ast.unparse(mine[1].test) == "rod_speed_dr is not None" and not mine[1].orelse
    text = "\n".join(dsk._segment(src, s) for s in mine)
    assert "stress_push" not in text and "torch.rand" not in text and "torch.randn" not in text
    # ``self._env`` is touched ONLY inside the guard (an unarmed reset must not need it)
    assert "self._env" not in ast.unparse(mine[0]) and "self._env" in ast.unparse(mine[1])
    assert all("self._env" not in ast.unparse(s) for s in reset.body if s not in mine)
    dsk._guarded_hook("reset", ["self", "env_ids"])                                 # still exactly two ``stress_push`` statements
    dsk._guarded_hook("process_actions", ["self"])


def _git_source(rel: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_CHANGE_COMMIT}:{rel.relative_to(REPO).as_posix()}"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"pre-change commit {PRE_CHANGE_COMMIT} not readable here")
    return out.stdout


def test_knob_off_reset_is_bitwise_the_pre_change_reset():
    old_src = _git_source(ACTIONS_PY)
    assert "rod_speed_dr" not in old_src, "the reference must predate the DR hooks"
    old_cls = rcs._term_class(old_src, ("reset",))
    new_cls = rcs._term_class(ACTIONS_PY.read_text(), ("reset",))
    for variant in ("absent", "none"):
        for ids in (torch.tensor([0, 2, 4]), None, [1, 3]):
            torch.manual_seed(7)
            new = rcs._make_term(cls=new_cls)
            torch.manual_seed(7)
            old = rcs._make_term(cls=old_cls)
            if variant == "none":
                new._rod_speed_dr = None
            torch.manual_seed(11)
            new.reset(ids)
            torch.manual_seed(11)
            old.reset(ids)
            a, b = rcs._clone_state(new), rcs._clone_state(old)
            assert set(a) - set(b) <= {"_rod_speed_dr"}
            for key in b:
                if isinstance(b[key], torch.Tensor):
                    assert torch.equal(a[key], b[key]) or (torch.isnan(a[key]).all() and torch.isnan(b[key]).all()), key
        torch.manual_seed(5)
        t_new = rcs._make_term(cls=new_cls)
        torch.manual_seed(5)
        t_old = rcs._make_term(cls=old_cls)
        torch.manual_seed(9)
        t_new.reset(torch.tensor([0, 1]))
        after_new = torch.rand(4)
        torch.manual_seed(9)
        t_old.reset(torch.tensor([0, 1]))
        assert torch.equal(after_new, torch.rand(4)), "the knob-off reset consumed a different number of draws"


def _method_dump(src: str, name: str) -> str:
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "CrabHexDelayedJointPositionAction")
    return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))


@pytest.mark.parametrize("name", ["_apply_pitch_linkage", "process_actions", "apply_actions", "_clip_raw_actions"])
def test_the_slew_and_hot_path_methods_are_unchanged_from_the_pre_change_commit(name):
    assert _method_dump(ACTIONS_PY.read_text(), name) == _method_dump(_git_source(ACTIONS_PY), name)


def test_only_init_and_reset_mention_the_dr():
    src, methods = dsk._action_term_methods()
    mentions = {name for name, node in methods.items() if "rod_speed_dr" in ast.get_source_segment(src, node)}
    assert mentions == {"__init__", "reset"}
    for name in ("process_actions", "_apply_pitch_linkage", "apply_actions"):
        assert "rod_speed_dr" not in ast.get_source_segment(src, methods[name]), name
    outside = [ln for ln in src.splitlines() if "rod_speed_dr" in ln and not ln.startswith((" ", "\t"))]
    assert outside == ["from parkour_tasks.crab_hex_forward_task.mdp.rod_speed_dr import build_rod_speed_dr"]


# ---- the slew, lifted from the real _apply_pitch_linkage (test_crab_hex_omni_motion_knobs' technique)
DT = 0.005
SCALES = [[0.5], [0.8], [1.0], [1.35]]


def _slew_term(scale, n=4):
    t, _ = mkt._action_term(n)
    if scale is not None:
        t._rod_speed_scale = scale
    return t


def _far(n=4, hip=0.4, knee=0.4):
    out = torch.zeros(n, 18)
    out[:, 6:12] = hip
    out[:, 12:18] = knee
    return out


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_far_target_slews_each_env_at_its_own_scale_times_the_rated_speed(sign):
    scale = torch.tensor(SCALES)
    plain, dr = _slew_term(None), _slew_term(scale)
    for t in (plain, dr):
        t._processed_actions = _far(hip=0.4 * sign, knee=0.4 * sign)
    synced_h = linkage.hip_actuator_length(torch.zeros(4, 6))
    synced_k = linkage.knee_actuator_length(torch.zeros(4, 6), torch.zeros(4, 6))
    plain._apply_pitch_linkage()
    dr._apply_pitch_linkage()
    # premise: the target is far enough that the slew limit, not the window clamp or the target, sets the step
    assert torch.allclose(plain.hip_rod_speed.abs(), torch.full((4, 6), linkage.HIP_SPEED_M_S), rtol=1e-6)
    assert torch.allclose(plain.knee_rod_speed.abs(), torch.full((4, 6), linkage.KNEE_SPEED_M_S), rtol=1e-6)
    s = scale.expand(4, 6)
    assert torch.allclose(dr.hip_rod_speed.abs(), s * linkage.HIP_SPEED_M_S, rtol=1e-6, atol=0)
    assert torch.allclose(dr.knee_rod_speed.abs(), s * linkage.KNEE_SPEED_M_S, rtol=1e-6, atol=0)
    assert torch.equal(torch.sign(dr.hip_rod_speed), torch.sign(plain.hip_rod_speed))        # same direction
    dlam_h = (dr._hip_rod_len - synced_h).abs().double()
    dlam_k = (dr._knee_rod_len - synced_k).abs().double()
    assert torch.allclose(dlam_h, s.double() * linkage.HIP_SPEED_M_S * DT, rtol=2e-3)       # |dlam| = speed * dt * s_e
    assert torch.allclose(dlam_k, s.double() * linkage.KNEE_SPEED_M_S * DT, rtol=2e-3)
    assert float(dlam_h[0, 0]) < float(dlam_h[1, 0]) < float(dlam_h[2, 0]) < float(dlam_h[3, 0])   # per env, not one value
    for _ in range(2):                                                              # it keeps going at s_e x speed
        dr._apply_pitch_linkage()
    assert torch.allclose((dr._hip_rod_len - synced_h).abs().double(), 3 * s.double() * linkage.HIP_SPEED_M_S * DT, rtol=2e-3)
    assert torch.allclose(dr.hip_rod_speed.abs(), s * linkage.HIP_SPEED_M_S, rtol=1e-6, atol=0)


def test_a_scale_of_one_is_nominal_and_the_unscaled_env_sees_no_dr_factor():
    scale = torch.ones(4, 1)
    plain, dr = _slew_term(None), _slew_term(scale)
    for t in (plain, dr):
        t._processed_actions = _far()
        t._apply_pitch_linkage()
    assert torch.equal(plain.hip_rod_speed, dr.hip_rod_speed) and torch.equal(plain.knee_rod_speed, dr.knee_rod_speed)


def test_a_scale_of_ones_is_bitwise_the_off_path_over_many_substeps():
    n = 4
    plain, dr = _slew_term(None, n), _slew_term(torch.ones(n, 1), n)
    gen = torch.Generator().manual_seed(2)
    slewed = 0
    for step in range(50):
        actions = torch.randn(n, 18, generator=gen) * 0.6                           # far and near targets, both signs
        for t in (plain, dr):
            t._processed_actions = actions.clone()
            t._apply_pitch_linkage()
        for attr in ("_hip_rod_len", "_knee_rod_len", "hip_rod_speed", "knee_rod_speed"):
            assert torch.equal(getattr(plain, attr), getattr(dr, attr)), (step, attr)
        assert plain._asset.pos_targets.keys() == dr._asset.pos_targets.keys()
        for key in plain._asset.pos_targets:
            assert torch.equal(plain._asset.pos_targets[key], dr._asset.pos_targets[key]), (step, key)
        slewed += int(bool((plain.hip_rod_speed.abs() > 0.99 * linkage.HIP_SPEED_M_S).any()))
    assert slewed >= 5, "the drive never reached the slew limit, so the comparison proved nothing"


def test_the_scale_is_read_every_substep_so_a_reset_redraw_takes_effect():
    scale = torch.ones(4, 1)
    t = _slew_term(scale)
    t._processed_actions = _far()
    t._apply_pitch_linkage()
    assert torch.allclose(t.hip_rod_speed.abs(), torch.full((4, 6), linkage.HIP_SPEED_M_S), rtol=1e-6)
    scale[2, 0] = 0.4                                                               # written in place, as RodSpeedDR.reset does
    t._apply_pitch_linkage()
    want = torch.tensor([1.0, 1.0, 0.4, 1.0]).reshape(4, 1).expand(4, 6) * linkage.HIP_SPEED_M_S
    assert torch.allclose(t.hip_rod_speed.abs(), want, rtol=1e-6)


def test_the_current_proxy_keeps_the_rated_constants():
    src = OBSERVATIONS_PY.read_text()
    assert "action_term.hip_rod_speed.abs()" in src and "action_term.knee_rod_speed.abs()" in src
    assert "(_CURRENT_SENSE_GATE_FRACTION * linkage.HIP_SPEED_M_S)" in src
    assert "(_CURRENT_SENSE_GATE_FRACTION * linkage.KNEE_SPEED_M_S)" in src
    for name in ("rod_speed_dr", "_rod_speed_scale", "rod_speed_scale"):
        assert name not in src, name                                                # the sensor never sees the plant's scale
