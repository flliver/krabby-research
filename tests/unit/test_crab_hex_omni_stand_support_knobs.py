"""Omni phase-2 stage p2s (``P2S_PLAN.md`` step 3, "The lever", 2026-10-08): the stand-slot support
income (T1) and loaded-foot slide penalty (T2).

Both are DEFAULT OFF and set by no preset. Pinned here, without Isaac:

* the knob module (``config/crab_hex/motion_knobs.py``): ``KRABBY_STAND_SUPPORT_W`` (an INCOME weight,
  >= 0) and ``KRABBY_STAND_SLIDE_W`` (a PENALTY weight, <= 0) -- parsers and signs, registration in
  ``MOTION_KNOB_PARSERS`` (and so the ladder's whitelist / parser table), env side (not runner-side,
  not eval-carried, no stand slot latch), and the applier on a stand-in cfg: unset / blank / ``0``
  writes NOTHING (the knob-off cfg is identical), armed adds exactly its one term with the builder's
  func / weight / params, a wrong sign or a duplicate term is refused before anything is written;
* the real builder table, built against stub Isaac modules: the exact params both terms carry;
* the kernels (``mdp/crab_hex_omni_motion_rewards.py``, pure torch) on a stand-in env: six loaded feet
  +3, a statue with one foot at 0 N +2, a tripod hover 0, a sliding loaded foot priced by its 3-D
  speed, an airborne moving foot NOT priced by T2, exactly zero off the stand slot and when tipped;
* the ``run_candidates.py`` ``OVERLAY_DUMP_CHECKS`` entries, on the REAL S1 s18 p2a dump (off passes,
  armed fails) and on that dump with the terms written in (armed passes, off / a wrong param fails).
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
MOTION_REWARDS_PY = OMNI / "mdp/crab_hex_omni_motion_rewards.py"
OMNI_PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
OMNI_REWARDS_PY = OMNI / "mdp/crab_hex_omni_rewards.py"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
#: The S1 s18 p2a head's run (the p2s fork source), relative to a checkout that holds the logs.
S1_DUMP_REL = "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/params/env.yaml"

SUPPORT_W, SLIDE_W = "KRABBY_STAND_SUPPORT_W", "KRABBY_STAND_SLIDE_W"
SUPPORT, SLIDE = "reward_stand_support", "penalty_stand_slide"
#: P2S_PLAN's planning dose for ST2 (step 4 sets the final values; any in-sign value works here).
ST2 = {SUPPORT_W: "1.12", SLIDE_W: "-1.25"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _s1_dump() -> Path | None:
    """The S1 dump in this checkout or, from a git worktree, in the main checkout (logs are untracked)."""
    for root in (REPO, *(p for p in REPO.parents if (p / ".git").is_dir())):
        if (root / S1_DUMP_REL).exists():
            return root / S1_DUMP_REL
    return None


S1_DUMP = _s1_dump()
mk = _load(MOTION_KNOBS_PY, "motion_knobs__stand_support_test")
mrew = _load(MOTION_REWARDS_PY, "motion_rewards__stand_support_test")
#: The forward cfg's six footpads, read from its SOURCE (the module itself needs Isaac).
FEET = next(ast.literal_eval(n.value) for n in ast.parse(FWD_MDP_CFG_PY.read_text()).body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_CRAB_FOOT_BODY_NAMES" for t in n.targets))


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# ======================================================================= parsers / registration
def test_parsers_and_signs():
    sup, sli = mk.MOTION_KNOB_PARSERS[SUPPORT_W], mk.MOTION_KNOB_PARSERS[SLIDE_W]
    for off in (None, "", "  ", "0", "0.0"):
        assert sup(off) == 0.0 and sli(off) == 0.0, off
    assert sup("1.12") == 1.12 and sup(" 0.38 ") == 0.38
    assert sli("-1.25") == -1.25 and sli(" -0.5 ") == -0.5
    with pytest.raises(ValueError, match=f"{SUPPORT_W} is an income weight and must be >= 0"):
        sup("-0.5")
    with pytest.raises(ValueError, match=f"{SLIDE_W} is a penalty weight and must be <= 0"):
        sli("0.5")
    for bad in ("x", "nan", "inf", "-inf"):
        for parse, var in ((sup, SUPPORT_W), (sli, SLIDE_W)):
            with pytest.raises(ValueError, match=var):
                parse(bad)


def test_variables_terms_and_params_are_the_contract():
    assert mk.STAND_SUPPORT_W_VAR == SUPPORT_W and mk.STAND_SLIDE_W_VAR == SLIDE_W
    assert mk.STAND_SUPPORT_TERM == SUPPORT and mk.STAND_SLIDE_TERM == SLIDE
    # 50 N = the eval's "loaded" line (and the clock's TOUCH_REF_N / FOOT_SLIP's); 0.9 = F1's upright line
    assert mk.STAND_LOAD_REF_N == mk.FOOT_SLIP_FORCE_N == 50.0 and mk.STAND_UPRIGHT_MIN == 0.9
    assert {"STAND_SUPPORT_W_VAR", "STAND_SLIDE_W_VAR", "STAND_SUPPORT_TERM", "STAND_SLIDE_TERM",
            "STAND_LOAD_REF_N", "STAND_UPRIGHT_MIN"} <= set(mk.__all__)
    for var in (SUPPORT_W, SLIDE_W):
        assert var in mk.MOTION_KNOB_PARSERS
        # env side (params/env.yaml); a reward weight is not an eval semantic -- an eval never carries it
        assert var not in mk.AGENT_KNOBS and var not in mk.EVAL_SEMANTIC_KNOBS
        assert var not in mk.EVAL_STRUCTURE_KNOBS and var not in mk.KNOB_OFF_VALUES
        assert var not in mk.PPO_CONSTRUCTION_KNOBS and var not in mk.REUSED_WEIGHT_KNOBS


def test_no_preset_arms_them_and_no_shipped_cfg_declares_them():
    presets = (OMNI_PHASES_PY.read_text()
               + (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_phases.py").read_text())
    for var in (SUPPORT_W, SLIDE_W):
        assert var not in presets, var
    for path in (FWD_MDP_CFG_PY, OMNI_REWARDS_PY, OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py"):
        for term in (SUPPORT, SLIDE):
            assert term not in path.read_text(), (path, term)


@pytest.mark.parametrize("path, allowed", [(MOTION_KNOBS_PY, ()), (MOTION_REWARDS_PY, ("torch",))])
def test_modules_stay_pure(path, allowed):
    """The knob module imports no Isaac / torch / package code; the rewards module only torch (the
    replay gate imports it directly)."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['m'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'parkour_tasks', 'parkour_isaaclab', 'isaaclab',\n"
        "                                                      'isaacsim', 'omni', 'torch'}))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == str(sorted(allowed)), out.stdout


# ======================================================================= the applier (stand-ins)
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


class _OmniObsFunc:
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}


def _cfg():
    cfg = _Bag()
    cfg.commands = _Bag()
    cfg.commands.base_velocity = _Bag()
    cfg.actions = _Bag()
    cfg.actions.joint_pos = _Bag()
    cfg.rewards = _Bag()
    cfg.rewards.reward_action_rate = _Term(func="rate", weight=-0.3, params={"asset_cfg": "robot"})
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    cfg.observations.policy.extreme_parkour_observations = _Term(func=_OmniObsFunc, params={"history_length": 10})
    return cfg


def _snap(obj):
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


BUILDERS = {name: (f"func:{name}", {"p": name}) for name in
            (mk.SMOOTH2_TERM, mk.JOINT_VEL_TERM, mk.FOOT_SLIP_TERM, mk.STAND_POSE_TERM, mk.STAND_STANCE_TERM,
             SUPPORT, SLIDE)}


def _apply(cfg, env):
    mk.apply_motion_knobs(cfg, env, term_cls=_Term, builders=BUILDERS)


@pytest.mark.parametrize("env", [{}, {SUPPORT_W: "", SLIDE_W: ""}, {SUPPORT_W: "0", SLIDE_W: "0"},
                                 {SUPPORT_W: " 0.0 ", SLIDE_W: "-0"}, {SUPPORT_W: "-0", SLIDE_W: "-0.0"}])
def test_unset_blank_or_zero_writes_nothing(env, capsys):
    pristine = _snap(_cfg())
    cfg = _cfg()
    _apply(cfg, env)
    assert _snap(cfg) == pristine, env
    assert not hasattr(cfg.rewards, SUPPORT) and not hasattr(cfg.rewards, SLIDE)
    assert "[KNOB]" not in capsys.readouterr().out


@pytest.mark.parametrize("var, term, raw, weight", [(SUPPORT_W, SUPPORT, "1.12", 1.12),
                                                    (SLIDE_W, SLIDE, "-1.25", -1.25)])
def test_each_knob_adds_exactly_its_one_term(var, term, raw, weight, capsys):
    cfg = _cfg()
    before = _snap(cfg)
    _apply(cfg, {var: raw})
    added = getattr(cfg.rewards, term)
    assert (added.func, added.weight, added.params) == (f"func:{term}", weight, {"p": term})
    assert added.params is not BUILDERS[term][1]                     # a copy, never the builder's dict
    after = _snap(cfg)
    del after["rewards"][term]
    assert after == before                                           # nothing else written ...
    # ... in particular no stand slot latch: the terms are rewards only, never an eval semantic
    assert not hasattr(cfg.commands.base_velocity, "eval_slot_latch")
    assert not hasattr(cfg.commands.base_velocity, "stand_hold") and not hasattr(cfg, "omni_stand_park")
    assert "stand_obs" not in cfg.observations.policy.extreme_parkour_observations.params
    assert capsys.readouterr().out.strip() == f"[KNOB] {var}={raw} consumed"


def test_both_compose_after_f1_in_order(capsys):
    cfg = _cfg()
    _apply(cfg, {**ST2, "KRABBY_STAND_STANCE_W": "0.9", "KRABBY_STAND_POSE_W": "-0.1"})
    names = list(vars(cfg.rewards))
    assert names == ["reward_action_rate", mk.STAND_POSE_TERM, mk.STAND_STANCE_TERM, SUPPORT, SLIDE]
    assert (cfg.rewards.reward_stand_support.weight, cfg.rewards.penalty_stand_slide.weight) == (1.12, -1.25)
    out = capsys.readouterr().out.splitlines()
    assert out == ["[KNOB] KRABBY_STAND_STANCE_W=0.9 consumed", f"[KNOB] {SUPPORT_W}=1.12 consumed",
                   f"[KNOB] {SLIDE_W}=-1.25 consumed"]


@pytest.mark.parametrize("env, match", [({SUPPORT_W: "-1.12"}, ">= 0"), ({SLIDE_W: "1.25"}, "<= 0"),
                                        ({SUPPORT_W: "1.12", SLIDE_W: "1.25"}, "<= 0"),
                                        ({SUPPORT_W: "-1.12", SLIDE_W: "-1.25"}, ">= 0")])
def test_a_wrong_sign_is_refused_before_anything_is_written(env, match):
    cfg = _cfg()
    pristine = _snap(cfg)
    with pytest.raises(ValueError, match=match):
        _apply(cfg, env)
    assert _snap(cfg) == pristine


@pytest.mark.parametrize("var, term", [(SUPPORT_W, SUPPORT), (SLIDE_W, SLIDE)])
def test_a_term_that_already_exists_is_refused(var, term):
    cfg = _cfg()
    setattr(cfg.rewards, term, _Term())
    with pytest.raises(RuntimeError, match=f"{var}: the rewards cfg already carries a {term!r} term"):
        _apply(cfg, {var: ST2[var]})


def test_the_ladder_whitelists_parses_and_audits_them():
    """The driver's whitelist / parser table come from MOTION_KNOB_PARSERS (no driver edit), and the
    validator's dump-check requirement is met (every new key has an OVERLAY_DUMP_CHECKS entry)."""
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__stand_support_reg_test")
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in (SUPPORT_W, SLIDE_W):
        assert var in whitelist, var
        assert parsers[var] is rc.knobs()._mk.MOTION_KNOB_PARSERS[var], var
        assert var in rc.OVERLAY_DUMP_CHECKS and var not in rc.AGENT_DUMP_KEYS, var
        for value, ok in ((ST2[var], True), ("0", True), ("", True), ("x", False),
                          (ST2[var].lstrip("-") if var == SLIDE_W else "-" + ST2[var], False)):
            reasons = []
            rc._check_knob(var, value, "x", whitelist, parsers, reasons, set())
            assert (reasons == []) is ok, (var, value, reasons)
            if not ok:
                assert any("does not round-trip" in r for r in reasons), reasons


# ======================================================================= the real builder table
class _Entity:
    """Stands in for ``SceneEntityCfg``: records its constructor arguments."""

    def __init__(self, name, **kw):
        self.name, self.kw = name, kw

    def __eq__(self, other):
        return isinstance(other, _Entity) and (self.name, self.kw) == (other.name, other.kw)

    def __repr__(self):
        return f"SceneEntityCfg({self.name!r}, {self.kw!r})"


def test_the_real_builders_wire_both_terms(monkeypatch):
    """``_default_term_builders`` run against stub Isaac modules: the exact func and params each p2s
    term is added with -- every param a keyword the kernel takes, the six footpads on both entities."""
    monkeypatch.setitem(sys.modules, "isaaclab.managers", _ns(SceneEntityCfg=_Entity))
    monkeypatch.setitem(sys.modules, "parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg",
                        _ns(_CRAB_FOOT_BODY_NAMES=list(FEET), _CRAB_POSITION_ACTUATED_JOINT_NAMES=["rod"]))
    monkeypatch.setitem(sys.modules, mk._REWARDS_MOD, mrew)
    monkeypatch.setitem(sys.modules, mk._LIB_REWARDS_MOD,
                        _ns(reward_stand_stance_quality=lambda *a, **k: None, CLOCK_VEL_REF_M_S=0.1))
    table = mk._default_term_builders()
    expect = {
        "sensor_cfg": _Entity("contact_forces", body_names=FEET, preserve_order=True),
        "asset_cfg": _Entity("robot", body_names=FEET, preserve_order=True),
        "command_name": "base_velocity",
        "load_ref_n": 50.0,
        "upright_min": 0.9,
    }
    assert len(FEET) == 6 and all(f.endswith("_Footpad") for f in FEET)
    for term, func in ((SUPPORT, mrew.reward_stand_support), (SLIDE, mrew.penalty_stand_slide)):
        got_func, params = table[term]
        assert got_func is func and params == expect, term
        sig = inspect.signature(func)
        assert set(params) == set(sig.parameters) - {"env"}, term     # every kernel param, no stray one
        assert sig.parameters["load_ref_n"].default == mk.STAND_LOAD_REF_N
        assert sig.parameters["upright_min"].default == mk.STAND_UPRIGHT_MIN
        assert sig.parameters["command_name"].default == "base_velocity"
    # each term gets its own entity cfgs (Isaac resolves body_ids on them in place)
    assert table[SUPPORT][1]["sensor_cfg"] is not table[SLIDE][1]["sensor_cfg"]
    assert table[SUPPORT][1]["asset_cfg"] is not table[SLIDE][1]["asset_cfg"]


# ======================================================================= the kernels
class _Scene(dict):
    def __init__(self, robot, sensors):
        super().__init__(robot=robot)
        self.sensors = sensors


SENSOR_FEET = [1, 2, 3, 4, 5, 6]          # 8 sensor bodies
ROBOT_FEET = [3, 4, 5, 6, 7, 8]           # 9 robot bodies


def _env(force, vel=None, stand=None, gz=None, command_name="base_velocity"):
    """``force`` ``(N, 6)`` N (along z; or ``(N, 6, 3)``), ``vel`` ``(N, 6, 3)`` m/s of the six feet."""
    force = torch.as_tensor(force, dtype=torch.float32)
    n = force.shape[0]
    net = torch.full((n, 8, 3), 999.0)                  # non-foot bodies carry junk: never read
    net[:, SENSOR_FEET] = force if force.dim() == 3 else torch.stack(
        [torch.zeros_like(force), torch.zeros_like(force), force], dim=-1)
    lin = torch.full((n, 9, 3), 9.0)
    lin[:, ROBOT_FEET] = torch.zeros(n, 6, 3) if vel is None else torch.as_tensor(vel, dtype=torch.float32)
    grav = torch.zeros(n, 3)
    grav[:, 2] = -(torch.ones(n) if gz is None else torch.as_tensor(gz, dtype=torch.float32))
    stand = torch.ones(n, dtype=torch.bool) if stand is None else torch.as_tensor(stand)
    asked = []

    def get_term(name):
        asked.append(name)
        assert name == command_name
        return _ns(stand_mask=lambda: stand)

    robot = _ns(data=_ns(body_lin_vel_w=lin, projected_gravity_b=grav))
    env = _ns(scene=_Scene(robot, {"contact_forces": _ns(data=_ns(net_forces_w=net))}),
              command_manager=_ns(get_term=get_term))
    env.asked = asked
    return env


SENSOR = _ns(name="contact_forces", body_ids=SENSOR_FEET)
ASSET = _ns(name="robot", body_ids=ROBOT_FEET)


def _t1(env, **kw):
    return mrew.reward_stand_support(env, SENSOR, ASSET, **kw)


def _t2(env, **kw):
    return mrew.penalty_stand_slide(env, SENSOR, ASSET, **kw)


LOADED = 700.0                            # a 230 kg robot's stance foot carries 200-800 N


def test_support_counts_supported_feet_beyond_a_tripod():
    rows = torch.tensor([
        [LOADED] * 6,                                        # six down             -> +3
        [LOADED] * 5 + [0.0],                                # statue, one at 0 N   -> +2
        [LOADED, 0.0, LOADED, 0.0, LOADED, 0.0],             # tripod hover         -> 0
        [0.0] * 6,                                           # all airborne         -> -3
        [2000.0, 50.0, 25.0, 0.0, 300.0, 49.0],              # saturation + partial load
    ])
    got = _t1(_env(rows))
    want = torch.tensor([3.0, 2.0, 0.0, -3.0, 1.0 + 1.0 + 0.5 + 0.0 + 1.0 + 0.98 - 3.0])
    assert torch.allclose(got, want, atol=1e-6), got
    assert got.dtype == torch.float32 and got.shape == (5,)
    # pressing sideways cannot inflate a foot past 1: |F| is the 3-D norm, clamped
    side = torch.zeros(1, 6, 3)
    side[0, :, 0] = 30.0
    side[0, :, 2] = 40.0                                     # |F| = 50 N exactly -> s = 1
    assert torch.allclose(_t1(_env(side)), torch.tensor([3.0]))
    # load_ref_n is honoured
    assert torch.allclose(_t1(_env(torch.full((1, 6), 50.0)), load_ref_n=100.0), torch.tensor([0.0]))


def test_slide_prices_loaded_moving_feet_by_their_3d_speed():
    vel = torch.zeros(4, 6, 3)
    vel[0, 0] = torch.tensor([0.06, 0.08, 0.0])               # one loaded foot sliding at 0.1 m/s
    vel[1, 0] = torch.tensor([0.0, 0.0, 0.1])                 # ... or moving vertically (scuff / press)
    vel[2, :] = torch.tensor([0.1, 0.0, 0.0])                 # six-down shuffle at 0.1 m/s per foot
    vel[3, 5] = torch.tensor([0.0, 0.3, 0.4])                 # a half-loaded (25 N) foot at 0.5 m/s
    force = torch.full((4, 6), LOADED)
    force[3, 5] = 25.0
    got = _t2(_env(force, vel))
    assert torch.allclose(got, torch.tensor([0.1, 0.1, 0.6, 0.25]), atol=1e-6), got
    assert (got >= 0).all()                                   # positive: the weight carries the sign
    # unsquared: double the speed, double the price
    assert torch.allclose(_t2(_env(force, 2 * vel)), 2 * got, atol=1e-6)
    # a still stance costs nothing
    assert torch.equal(_t2(_env(force)), torch.zeros(4))


def test_an_airborne_moving_foot_is_not_priced_by_the_slide_term():
    force = torch.tensor([[LOADED, 0.0, LOADED, 0.0, LOADED, 0.0]])
    vel = torch.zeros(1, 6, 3)
    vel[0, [1, 3, 5]] = torch.tensor([0.5, 0.0, 1.5])         # the swing tripod marching at speed
    assert torch.equal(_t2(_env(force, vel)), torch.zeros(1))
    assert torch.equal(_t1(_env(force, vel)), torch.zeros(1))  # and the tripod earns no support
    # a still hover is priced by neither -- the support income is what separates it from the statue
    assert _t1(_env(torch.tensor([[LOADED] * 5 + [0.0]]))) - _t1(_env(force)) == 2.0


def test_exactly_zero_off_the_stand_slot_and_when_tipped():
    n = 6
    force = torch.full((n, 6), LOADED)
    force[:, 0] = 0.0                                         # T1 = +2 where gated in
    vel = torch.zeros(n, 6, 3)
    vel[:, 2] = torch.tensor([0.3, 0.0, 0.0])                 # T2 = 0.3 where gated in
    stand = torch.tensor([True, False, True, True, False, True])
    gz = torch.tensor([1.0, 1.0, 0.5, 0.85, 0.5, 0.95])       # rows 2-4 tipped past F1's 0.9 line
    env = _env(force, vel, stand, gz)
    t1, t2 = _t1(env), _t2(env)
    assert torch.equal(t1, torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 2.0]))
    assert torch.allclose(t2, torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.3]))
    # the all-airborne income (-3) is gated to zero too: a fall is never priced through these terms
    air = _env(torch.zeros(2, 6), stand=torch.tensor([False, True]), gz=torch.tensor([1.0, 0.2]))
    assert torch.equal(_t1(air), torch.zeros(2))
    # upright_min is honoured
    assert torch.equal(_t1(env, upright_min=0.4), torch.tensor([2.0, 0.0, 2.0, 2.0, 0.0, 2.0]))


def test_the_gate_reads_the_command_slot_only():
    """``stand_mask()`` of the named command term, nothing else: no command value, no turn gate."""
    env = _env(torch.full((2, 6), LOADED), stand=torch.tensor([True, False]), command_name="cmd")
    assert torch.equal(_t1(env, command_name="cmd"), torch.tensor([3.0, 0.0]))
    assert env.asked == ["cmd"]
    src = MOTION_REWARDS_PY.read_text()
    assert "get_command" not in src and "stand_mask()" in src


# ======================================================================= the dump checks
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__stand_support_test")


def _entity(name: str, feet=FEET) -> dict:
    """A ``SceneEntityCfg`` as IsaacLab's ``dump_yaml`` writes it."""
    s = slice(None)
    return {"name": name, "joint_names": None, "joint_ids": s, "fixed_tendon_names": None, "fixed_tendon_ids": s,
            "body_names": list(feet), "body_ids": s, "object_collection_names": None, "object_collection_ids": s,
            "preserve_order": True}


def _term(func: str, weight: float, **changes) -> dict:
    params = {"sensor_cfg": _entity("contact_forces"), "asset_cfg": _entity("robot"),
              "command_name": "base_velocity", "load_ref_n": 50.0, "upright_min": 0.9}
    for key, value in changes.items():
        if value is None:
            params.pop(key)
        else:
            params[key] = value
    return {"func": f"parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards:{func}",
            "params": params, "weight": weight}


class _NoAliasDumper(yaml.Dumper):
    """No ``&idNNN`` anchors: the real dump already defines its own, and a duplicate is a YAML error."""

    def ignore_aliases(self, data):
        return True


def _armed(base: str, terms: dict) -> str:
    """``base`` (a real dump) with ``terms`` written at the top of its ``rewards:`` block."""
    block = yaml.dump({"rewards": terms}, Dumper=_NoAliasDumper, default_flow_style=False, sort_keys=False)
    assert block.startswith("rewards:\n")
    assert base.count("\nrewards:\n") == 1
    return base.replace("\nrewards:\n", "\n" + block, 1)


def _ok(rc, key):
    return rc.OVERLAY_DUMP_CHECKS[key]


needs_s1 = pytest.mark.skipif(S1_DUMP is None, reason="the S1 s18 p2a run directory is not on this machine")


@needs_s1
def test_real_s1_dump_passes_off_and_fails_armed(rc):
    txt = S1_DUMP.read_text()
    assert SUPPORT not in txt and SLIDE not in txt
    for key in (SUPPORT_W, SLIDE_W):
        for off in ("", "0", "0.0"):
            assert _ok(rc, key)(txt, off), (key, off)
        assert not _ok(rc, key)(txt, ST2[key]), key


@needs_s1
def test_armed_dump_passes_armed_and_fails_off(rc):
    txt = _armed(S1_DUMP.read_text(), {SUPPORT: _term(SUPPORT, 1.12), SLIDE: _term(SLIDE, -1.25)})
    assert _ok(rc, SUPPORT_W)(txt, "1.12") and _ok(rc, SLIDE_W)(txt, "-1.25")
    for key, other in ((SUPPORT_W, "1.0"), (SLIDE_W, "-1.0")):
        assert not _ok(rc, key)(txt, "0") and not _ok(rc, key)(txt, ""), key
        assert not _ok(rc, key)(txt, other), key                      # its own weight, exactly
    # each check reads its own term: the neighbour's weight never passes
    assert not _ok(rc, SUPPORT_W)(txt, "1.25") and not _ok(rc, SLIDE_W)(txt, "-1.12")
    only_slide = _armed(S1_DUMP.read_text(), {SLIDE: _term(SLIDE, -1.25)})
    assert _ok(rc, SUPPORT_W)(only_slide, "0") and not _ok(rc, SUPPORT_W)(only_slide, "1.25")


@needs_s1
@pytest.mark.parametrize("changes", [
    {"load_ref_n": 40.0}, {"upright_min": 0.8}, {"command_name": "other"}, {"command_name": None},
    {"upright_min": None}, {"min_upright_gz": 0.9}, {"sensor_cfg": _entity("contact_sensor")},
    {"asset_cfg": _entity("robot", FEET[:5])}, {"sensor_cfg": _entity("contact_forces", FEET[::-1])},
    {"func": "parkour_isaaclab.envs.mdp.rewards:reward_stand_support"},
    {"func": "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards:reward_stand_stance_quality"},
])
def test_support_check_pins_func_and_params(rc, changes):
    """The support check is F1's (absent at 0, else exact func + weight) plus the builder's params:
    any other func, a moved / missing / stray param or other feet fail."""
    changes = dict(changes)
    term = _term(SUPPORT, 1.12, **{k: v for k, v in changes.items() if k != "func"})
    if "func" in changes:
        term["func"] = changes["func"]
    txt = _armed(S1_DUMP.read_text(), {SUPPORT: term})
    assert not _ok(rc, SUPPORT_W)(txt, "1.12") and not _ok(rc, SUPPORT_W)(txt, "0")
    good = _armed(S1_DUMP.read_text(), {SUPPORT: _term(SUPPORT, 1.12)})
    assert _ok(rc, SUPPORT_W)(good, "1.12")


@pytest.mark.parametrize("key, bad", [(SUPPORT_W, "-1"), (SLIDE_W, "1"), (SUPPORT_W, "x"), (SLIDE_W, "nan")])
def test_a_malformed_value_raises_instead_of_picking_a_side(rc, key, bad):
    with pytest.raises(ValueError):
        _ok(rc, key)("rewards:\n", bad)


def _overlay(rc, tmp_path, text, env):
    params = tmp_path / "run" / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "env.yaml").write_text(text)
    return rc.overlay_checks({"ckpt": str(tmp_path / "run" / "model_24994.pt")}, env)


@needs_s1
def test_overlay_checks_end_to_end_on_the_real_dump(rc, tmp_path):
    """Every new overlay key has an env-side check: the ST2 overlay fails on the S1 dump (the
    continuation) and passes once both terms are in it; the OFF overlay passes on S1."""
    s1 = S1_DUMP.read_text()
    bad = _overlay(rc, tmp_path / "s1", s1, ST2)
    assert len(bad) == 2 and not any(ok for _n, ok in bad), bad
    assert {n for n, _ok_ in bad} == {f"overlay {k}={v} in params/env.yaml" for k, v in ST2.items()}
    good = _overlay(rc, tmp_path / "arm", _armed(s1, {SUPPORT: _term(SUPPORT, 1.12), SLIDE: _term(SLIDE, -1.25)}),
                    ST2)
    assert len(good) == 2 and all(ok for _n, ok in good), good
    off = _overlay(rc, tmp_path / "off", s1, {SUPPORT_W: "0", SLIDE_W: "0"})
    assert len(off) == 2 and all(ok for _n, ok in off), off


def test_the_synthetic_term_matches_the_builder():
    """The dump fixture above is the builder's params as IsaacLab would dump them (kept in step)."""
    term = copy.deepcopy(_term(SUPPORT, 1.0))
    assert set(term["params"]) == set(inspect.signature(mrew.reward_stand_support).parameters) - {"env"}
    assert term["params"]["load_ref_n"] == mk.STAND_LOAD_REF_N and term["params"]["upright_min"] == mk.STAND_UPRIGHT_MIN
    assert term["func"].split(":")[0] == mk._REWARDS_MOD
