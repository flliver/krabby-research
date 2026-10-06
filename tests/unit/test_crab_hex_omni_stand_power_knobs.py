"""Omni phase-2 plan step 3 (2026-10-06): the F1 stand term and the E1 actuated-joint power subset.

Both are DEFAULT OFF and set by no preset. Pinned here, without Isaac:

* the knob module (``config/crab_hex/motion_knobs.py``): ``KRABBY_STAND_STANCE_W`` (an INCOME
  weight, >= 0) and ``KRABBY_POWER_JOINTS`` (``actuated`` | off) -- parsers, registration in
  ``MOTION_KNOB_PARSERS`` / the ``omni_knobs`` re-export / the ladder whitelist, env-side (not
  runner-side, not eval-carried), and the applier on a stand-in cfg: unset / blank / off writes
  NOTHING (the knob-off rewards block is byte-identical), armed writes exactly one term or one
  param, mis-use is refused;
* the pure math: ``crab_hex_clock_reward.stand_stance_income`` IS the clock term's stopped
  branch (both combine modes);
* the Isaac-side functions, lifted out of ``parkour_isaaclab/envs/mdp/rewards.py`` with ``ast``
  and exec'd against stubs: ``reward_stand_stance_quality`` is 0 on non-stand rows and equals the
  stance-quality income (times the upright gate) on stand rows of a synthetic fixture;
  ``penalty_mechanical_power`` with the default (all-joint) ``asset_cfg`` is bitwise the original
  expression and with a joint subset sums only the selected joints.

The OVERLAY_DUMP_CHECKS entries for both knobs belong to plan step 4 (``run_candidates.py``) and
are not pinned here.
"""

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task/mdp"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
OMNI_PHASES_PY = OMNI / "config/crab_hex/crab_hex_omni_phases.py"
OMNI_REWARDS_PY = OMNI / "mdp/crab_hex_omni_rewards.py"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"

STAND_W, POWER_J = "KRABBY_STAND_STANCE_W", "KRABBY_POWER_JOINTS"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_of(path: Path, name: str) -> str:
    src = path.read_text()
    node = [n for n in ast.parse(src, str(path)).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


mk = _load(MOTION_KNOBS_PY, "motion_knobs__stand_power_test")
if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as clock  # noqa: E402


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# ======================================================================= parsers / registration
def test_parsers():
    p = mk.parse_income_w
    assert p(None) == 0.0 and p("") == 0.0 and p("  ") == 0.0 and p("0") == 0.0
    assert p("0.75") == 0.75 and p(" 1 ") == 1.0
    with pytest.raises(ValueError, match=">= 0"):
        p("-0.5", STAND_W)
    for bad in ("x", "nan", "inf"):
        with pytest.raises(ValueError):
            p(bad, STAND_W)
    q = mk.parse_power_joints
    for off in (None, "", " ", "all", "ALL"):
        assert q(off) is None, off
    assert q("actuated") == "actuated" and q(" Actuated ") == "actuated"
    for bad in ("rods", "1", "0", "actuated,rods"):
        with pytest.raises(ValueError, match=POWER_J):
            q(bad)
    assert mk.MOTION_KNOB_PARSERS[STAND_W]("0.6") == 0.6
    assert mk.MOTION_KNOB_PARSERS[POWER_J]("actuated") == "actuated"
    with pytest.raises(ValueError, match=STAND_W):
        mk.MOTION_KNOB_PARSERS[STAND_W]("-1")


def test_variables_are_env_side_and_registered():
    assert mk.STAND_STANCE_W_VAR == STAND_W and mk.POWER_JOINTS_VAR == POWER_J
    assert {STAND_W, POWER_J} <= set(mk.MOTION_KNOB_PARSERS)
    assert {"STAND_STANCE_W_VAR", "POWER_JOINTS_VAR", "parse_income_w", "parse_power_joints"} <= set(mk.__all__)
    # env side (params/env.yaml): not runner knobs, and a reward weight / reward scope is not an
    # eval semantic -- an eval never needs to carry them
    for var in (STAND_W, POWER_J):
        assert var not in mk.AGENT_KNOBS and var not in mk.EVAL_SEMANTIC_KNOBS
        assert var not in mk.EVAL_STRUCTURE_KNOBS and var not in mk.KNOB_OFF_VALUES
    assert mk.STAND_STANCE_TERM == "reward_stand_stance_quality"
    assert mk.POWER_TERM == mk.REUSED_WEIGHT_KNOBS[mk.POWER_W_VAR] == "penalty_mechanical_power"


def test_omni_knobs_reexports_them():
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__stand_power_test")
    names = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}
    assert {STAND_W, POWER_J} <= names
    assert {"STAND_STANCE_W_VAR", "POWER_JOINTS_VAR"} <= set(knobs.__all__)
    assert knobs.MOTION_KNOB_PARSERS is knobs._mk.MOTION_KNOB_PARSERS


def test_the_ladder_whitelists_and_parses_them():
    """The driver's whitelist / parser table are built from omni_knobs, so both knobs reach them
    without a driver edit (their OVERLAY_DUMP_CHECKS entries are plan step 4)."""
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__stand_power_test")
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in (STAND_W, POWER_J):
        assert var in whitelist and var in parsers, var


def test_no_preset_arms_them():
    src = OMNI_PHASES_PY.read_text() + (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_phases.py").read_text()
    for var in (STAND_W, POWER_J):
        assert var not in src, var


def test_knob_module_stays_pure():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('mk', {str(MOTION_KNOBS_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['mk'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.parse_power_joints('actuated') == 'actuated'\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'parkour_isaaclab', 'isaaclab', 'torch'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ======================================================================= the applier (stand-ins)
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


ROBOT_ALL = ("SceneEntityCfg", "robot")          # stands in for SceneEntityCfg("robot")
ACTUATED = ("SceneEntityCfg", "robot", "actuated")


def _cfg(power_w=0.0):
    cfg = _Bag()
    cfg.commands = _Bag()
    cfg.commands.base_velocity = _Bag()
    cfg.actions = _Bag()
    cfg.actions.joint_pos = _Bag()
    cfg.rewards = _Bag()
    cfg.rewards.reward_action_rate = _Term(func="rate", weight=-0.3, params={"asset_cfg": "robot"})
    cfg.rewards.penalty_mechanical_power = _Term(func="power", weight=power_w,
                                                 params={"asset_cfg": ROBOT_ALL})
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    return cfg


def _snap(obj):
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


BUILDERS = {name: (f"func:{name}", {"p": name}) for name in
            (mk.SMOOTH2_TERM, mk.JOINT_VEL_TERM, mk.FOOT_SLIP_TERM, mk.STAND_POSE_TERM, mk.STAND_STANCE_TERM)}


def _apply(cfg, env):
    mk.apply_motion_knobs(cfg, env, term_cls=_Term, builders=BUILDERS,
                          power_joint_cfgs={"actuated": ACTUATED})


@pytest.mark.parametrize("power_w", [0.0, -2.3e-4])
def test_unset_blank_or_off_writes_nothing(power_w):
    """The knob-off rewards block is byte-identical: no term added, the power term's params and
    weight untouched (whether or not KRABBY_POWER_W itself is armed)."""
    pristine = _snap(_cfg(power_w))
    for env in ({}, {STAND_W: "", POWER_J: ""}, {STAND_W: "0", POWER_J: "all"}, {STAND_W: "0.0", POWER_J: " "}):
        cfg = _cfg(power_w)
        _apply(cfg, env)
        assert _snap(cfg) == pristine, env
        assert not hasattr(cfg.rewards, mk.STAND_STANCE_TERM)


def test_stand_stance_w_adds_the_one_income_term(capsys):
    cfg = _cfg()
    before = _snap(cfg)
    _apply(cfg, {STAND_W: "0.75"})
    term = getattr(cfg.rewards, mk.STAND_STANCE_TERM)
    assert (term.func, term.weight, term.params) == (f"func:{mk.STAND_STANCE_TERM}", 0.75,
                                                     {"p": mk.STAND_STANCE_TERM})
    after = _snap(cfg)
    del after["rewards"][mk.STAND_STANCE_TERM]
    assert after == before                                    # nothing else written
    assert f"[KNOB] {STAND_W}=0.75 consumed" in capsys.readouterr().out
    # not a stand-keyed EVAL knob: the eval slot latch stays unarmed
    assert not hasattr(cfg.commands.base_velocity, "eval_slot_latch")
    with pytest.raises(ValueError, match=">= 0"):
        _apply(_cfg(), {STAND_W: "-0.5"})
    dup = _cfg()
    dup.rewards.reward_stand_stance_quality = _Term()
    with pytest.raises(RuntimeError, match="already carries"):
        _apply(dup, {STAND_W: "0.5"})
    # composes with the penalty knobs
    both = _cfg()
    _apply(both, {STAND_W: "0.5", "KRABBY_STAND_POSE_W": "-0.1"})
    assert both.rewards.penalty_stand_pose.weight == -0.1 and both.rewards.reward_stand_stance_quality.weight == 0.5


def test_power_joints_rescopes_the_reused_power_term(capsys):
    cfg = _cfg(power_w=-2.3e-4)
    before = _snap(cfg)
    _apply(cfg, {POWER_J: "actuated"})
    term = cfg.rewards.penalty_mechanical_power
    assert term.params == {"asset_cfg": ACTUATED} and term.weight == -2.3e-4 and term.func == "power"
    after = _snap(cfg)
    after["rewards"]["penalty_mechanical_power"]["params"]["asset_cfg"] = ROBOT_ALL
    assert after == before                                    # only the asset_cfg param changed
    assert f"[KNOB] {POWER_J}=actuated consumed" in capsys.readouterr().out
    # refused: a subset of a term that prices nothing, a missing term, a bad spelling
    with pytest.raises(RuntimeError, match="KRABBY_POWER_W"):
        _apply(_cfg(power_w=0.0), {POWER_J: "actuated"})
    no_term = _cfg(power_w=-1e-4)
    del no_term.rewards.penalty_mechanical_power
    with pytest.raises(RuntimeError, match="penalty_mechanical_power"):
        _apply(no_term, {POWER_J: "actuated"})
    with pytest.raises(ValueError, match=POWER_J):
        _apply(_cfg(power_w=-1e-4), {POWER_J: "rods"})


def test_real_builders_by_source():
    """The Isaac-side builders, checked by SOURCE (building them needs Isaac)."""
    src = MOTION_KNOBS_PY.read_text()
    assert "lib_rew.reward_stand_stance_quality" in src
    assert mk._LIB_REWARDS_MOD == "parkour_isaaclab.envs.mdp.rewards"
    lib_src = LIB_REWARDS_PY.read_text()
    assert "def reward_stand_stance_quality(" in lib_src
    assert "stand_stance_income," in lib_src                  # imported from the clock module
    # the actuated subset is the forward cfg's 18-joint list: cam shafts + the two rod joints,
    # NOT the passive Body_Hip emulator joint
    tree = ast.parse(FWD_MDP_CFG_PY.read_text())
    lists = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and t.id.startswith("_CRAB_")
             and isinstance(n.value, ast.List)}
    assert lists["_CRAB_ACTUATED_JOINT_NAMES"] == [".*_Body_CamShaft_RevoluteJoint", ".*_Hip_Femur_RevoluteJoint",
                                                   ".*_Femur_Tibia_RevoluteJoint"]
    assert "_CRAB_ACTUATED_JOINT_NAMES" in src and "Body_Hip" not in "".join(lists["_CRAB_ACTUATED_JOINT_NAMES"])


def test_actuated_regexes_select_18_of_the_24_asset_joints():
    """Against the shipped asset's joint names (``assets/crab.usda``): the three regexes pick the
    6 cam shafts + 12 rods and leave exactly the six Body_Hip emulator joints out."""
    import re

    usda = REPO / "assets/crab.usda"
    if not usda.exists():
        pytest.skip("assets/crab.usda not present")
    joints = sorted(set(re.findall(r'"([A-Z]{2}_[A-Za-z]+_[A-Za-z]+_RevoluteJoint)"', usda.read_text())))
    assert len(joints) == 24
    pats = [".*_Body_CamShaft_RevoluteJoint", ".*_Hip_Femur_RevoluteJoint", ".*_Femur_Tibia_RevoluteJoint"]
    picked = [j for j in joints if any(re.fullmatch(p, j) for p in pats)]
    assert len(picked) == 18
    assert sorted(set(joints) - set(picked)) == sorted(f"{leg}_Body_Hip_RevoluteJoint"
                                                       for leg in ("FL", "FR", "ML", "MR", "RL", "RR"))


def test_shipped_rewards_cfgs_are_untouched():
    """Default byte-identity at the SOURCE: the forward class body still declares the power term
    over every joint (``SceneEntityCfg("robot")``, weight 0.0, KRABBY_POWER_W override), and no
    rewards class body declares the F1 term -- it exists only when the knob adds it."""
    fwd = FWD_MDP_CFG_PY.read_text()
    assert fwd.count('func=mdp_rewards.penalty_mechanical_power,\n        weight=0.0,\n'
                     '        params={"asset_cfg": SceneEntityCfg("robot")},') >= 1
    assert '"KRABBY_POWER_W": "penalty_mechanical_power"' in fwd
    for path in (FWD_MDP_CFG_PY, OMNI_REWARDS_PY, OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py"):
        assert "reward_stand_stance_quality" not in path.read_text(), path
        assert "POWER_JOINTS" not in path.read_text(), path


# ======================================================================= pure math
def _feet(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    force = torch.rand(n, 6, generator=g) * 300.0
    force[0] = 0.0                                           # all feet unloaded
    force[1] = 700.0                                         # all loaded
    speed = torch.rand(n, 6, generator=g) * 0.3
    speed[1] = 0.0                                           # ... and still
    return force, speed


def test_stand_stance_income_is_the_stopped_clock_branch():
    force, speed = _feet()
    n = force.shape[0]
    got = clock.stand_stance_income(force, speed)
    stopped = torch.zeros(n, dtype=torch.bool)
    for phase in (torch.zeros(n), torch.rand(n) * 6.28):     # phase is irrelevant when stopped
        prod = clock.clock_schedule_income(phase, force, speed, stopped, combine="product")
        summ = clock.clock_schedule_income(phase, force, speed, stopped, combine="sum")
        assert torch.equal(got, prod)
        assert torch.allclose(got, summ, rtol=0, atol=1e-6)
    # restated by hand: mean over feet of load x planted-still
    loaded = 1.0 - torch.exp(-((force / clock.TOUCH_REF_N) ** 2))
    still = torch.exp(-((speed / clock.VEL_REF_M_S) ** 2))
    assert torch.allclose(got, (loaded * still).mean(dim=1), atol=1e-6)
    assert got[0] == 0.0 and torch.isclose(got[1], torch.tensor(1.0))
    assert ((got >= 0) & (got <= 1)).all()
    # vel_ref is honoured
    assert (clock.stand_stance_income(force, speed, vel_ref=1.0) >= got - 1e-7).all()


# ======================================================================= Isaac-side functions
class _Scene(dict):
    def __init__(self, robot, sensors=None):
        super().__init__(robot=robot)
        self.sensors = sensors or {}


def _lift(*names):
    ns = {"torch": torch, "SceneEntityCfg": lambda name="robot", **kw: _ns(name=name, joint_ids=slice(None)),
          "Articulation": object, "ParkourManagerBasedRLEnv": object,
          "stand_stance_income": clock.stand_stance_income,
          "CLOCK_VEL_REF_M_S": clock.VEL_REF_M_S}
    for name in names:
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, name),
                     str(LIB_REWARDS_PY), "exec"), ns)
    return ns


def _stand_env(force, speed, stand, gz):
    n = force.shape[0]
    net = torch.zeros(n, 8, 3)                       # 8 sensor bodies; feet are bodies 1..6
    net[:, 1:7, 2] = force
    vel = torch.zeros(n, 9, 3)                       # 9 robot bodies; feet are bodies 3..8
    vel[:, 3:9, 0] = speed * 0.6
    vel[:, 3:9, 1] = speed * 0.8                     # |v_xy| = speed
    vel[:, 3:9, 2] = 5.0                             # vertical speed is not slip
    grav = torch.zeros(n, 3)
    grav[:, 2] = -gz
    robot = _ns(data=_ns(body_lin_vel_w=vel, projected_gravity_b=grav))
    sensor = _ns(data=_ns(net_forces_w=net))
    cmd_term = _ns(stand_mask=lambda: stand)
    return _ns(scene=_Scene(robot, {"contact_forces": sensor}),
               command_manager=_ns(get_term=lambda name: cmd_term))


def test_f1_is_zero_off_stand_and_the_stance_income_on_stand():
    fn = _lift("reward_stand_stance_quality")["reward_stand_stance_quality"]
    force, speed = _feet(n=6, seed=3)
    stand = torch.tensor([True, True, False, False, True, True])
    gz = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.95])          # row 4 fallen past the 0.9 gate
    env = _stand_env(force, speed, stand, gz)
    asset_cfg = _ns(name="robot", body_ids=[3, 4, 5, 6, 7, 8])
    sensor_cfg = _ns(name="contact_forces", body_ids=[1, 2, 3, 4, 5, 6])
    got = fn(env, asset_cfg, sensor_cfg)
    income = clock.stand_stance_income(force, speed)
    expect = torch.where(stand & (gz > 0.9), income, torch.zeros_like(income))
    assert torch.equal(got, expect)
    assert (got[~stand] == 0).all()                              # non-stand rows: exactly zero
    assert torch.equal(got[[0, 1, 5]], income[[0, 1, 5]])      # stand + upright: the income itself
    assert got[1] > 0.99                                         # six still, loaded feet ~ 1
    # the stand gate reads the SLOT, whatever the command or clock: no command input at all
    assert "get_command" not in _source_of(LIB_REWARDS_PY, "reward_stand_stance_quality")
    # a marching stand row (half the feet in the air and moving) earns visibly less than a still one
    march_f = torch.tensor([[700.0, 0.0, 700.0, 0.0, 700.0, 0.0]])
    march_v = torch.tensor([[0.02, 0.3, 0.02, 0.3, 0.02, 0.3]])
    env1 = _stand_env(march_f, march_v, torch.tensor([True]), torch.tensor([1.0]))
    m = fn(env1, asset_cfg, sensor_cfg)
    assert 0.4 < m.item() < 0.5


def test_power_default_is_bitwise_the_original_and_subset_sums_only_selected():
    fn = _lift("penalty_mechanical_power")["penalty_mechanical_power"]
    g = torch.Generator().manual_seed(7)
    tau = torch.randn(4, 24, generator=g) * 30.0
    qd = torch.randn(4, 24, generator=g) * 3.0
    env = _ns(scene=_Scene(_ns(data=_ns(applied_torque=tau, joint_vel=qd))))
    original = torch.sum(torch.abs(tau * qd), dim=1)              # the pre-2026-10-06 expression
    assert torch.equal(fn(env, _ns(name="robot", joint_ids=slice(None))), original)
    assert torch.equal(fn(env), original)                         # the signature default too
    # the Isaac joint layout is breadth-first; any 18-of-24 index list works for the sum
    ids = [i for i in range(24) if i % 4 != 1]                    # drop 6 "Body_Hip" columns
    got = fn(env, _ns(name="robot", joint_ids=ids))
    assert torch.allclose(got, torch.abs(tau[:, ids] * qd[:, ids]).sum(dim=1))
    dropped = [i for i in range(24) if i % 4 == 1]
    assert torch.allclose(got + torch.abs(tau[:, dropped] * qd[:, dropped]).sum(dim=1), original, rtol=1e-6)
    # a selected-joint-only change moves the cost; an excluded-joint change does not
    tau2 = tau.clone()
    tau2[:, dropped] *= 100.0
    env2 = _ns(scene=_Scene(_ns(data=_ns(applied_torque=tau2, joint_vel=qd))))
    assert torch.equal(fn(env2, _ns(name="robot", joint_ids=ids)), got)
    # only the FULL slice means "every joint": a partial slice prices just its columns
    # (review 2026-10-06; before, any slice took the all-joint branch)
    part = fn(env, _ns(name="robot", joint_ids=slice(0, 18)))
    assert torch.allclose(part, torch.abs(tau[:, :18] * qd[:, :18]).sum(dim=1))
    assert not torch.allclose(part, original)
