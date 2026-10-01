"""The motion-quality knobs (gait-stabilisation plan, step 2, 2026-09-22), without Isaac.

Every knob is DEFAULT OFF, and "off" must mean "writes nothing" -- that is what keeps the knob-off
config dumps byte-identical to ``dumps_ladder_base`` and an eval carrying a knob's off value
identical to one that omits it. So each knob is pinned twice here: what it writes when armed, and
that it writes NOTHING (no field, no term, no param) when unset, blank or at its off value.

Covered, one block per knob group:

* the pure knob module (``config/crab_hex/motion_knobs.py``): parsers, the env-side applier on a
  stand-in cfg, the runner-side applier, its re-export through ``omni_knobs`` and no preset arming
  any knob;
* the four ADDED penalty terms (``mdp/crab_hex_omni_motion_rewards.py``, pure torch) on stand-in
  envs;
* ``reward_action_rate``'s squared form and the clock rewards' stand-park gate, lifted out of
  ``parkour_isaaclab/envs/mdp/rewards.py`` with ``ast`` and exec'd against stubs;
* the gait-clock park (``crab_hex_clock_reward.park_clock_phase``) and the rod low-pass helpers
  (``crab_hex_linkage``), plus the action term's ``process_actions`` / ``_apply_pitch_linkage``
  lifted the same way, knob-off bitwise and knob-on;
* ``CrabHexActorCriticRMA``'s std clamp / std reset (loaded as a synthetic package: no Isaac);
* the campaign ladder's audit of every knob (``run_candidates.OVERLAY_DUMP_CHECKS`` against
  ``params/env.yaml`` / ``params/agent.yaml``, the parser table, the whitelist, ``--smoke-overlay``).

The command-side knobs (relay hysteresis, stand hold) are pinned in
``test_crab_hex_omni_command.py``, the observation blocks in ``test_crab_hex_omni_observations.py``,
their mirror rows in ``test_crab_hex_omni_mirror.py`` and the eval carry in
``test_crab_hex_omni_gate.py``.
"""

import ast
import importlib
import importlib.util
import math
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
MOTION_REWARDS_PY = OMNI / "mdp/crab_hex_omni_motion_rewards.py"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"
ACTIONS_PY = FWD_MDP / "parkour_actions.py"
RSL_MODULES = REPO / "parkour/scripts/rsl_rl/modules"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"

ALL_KNOBS = ("KRABBY_ACTION_STD_MAX", "KRABBY_ACTION_STD_RESET", "KRABBY_ENTROPY_COEF",
             "KRABBY_ACTION_RATE_SQ", "KRABBY_ACTION_SMOOTH2_W", "KRABBY_JOINT_VEL_W",
             "KRABBY_FOOT_SLIP_W", "KRABBY_STAND_POSE_W", "KRABBY_ACTION_LPF_HZ",
             "KRABBY_ACTION_LPF_OBS", "KRABBY_OMNI_RELAY_HYST", "KRABBY_OMNI_STAND_HOLD",
             "KRABBY_OMNI_STAND_PARK", "KRABBY_OMNI_STAND_OBS",
             # PPO-construction plan step 2 (2026-10-01): pinned in detail by
             # tests/unit/test_ppo_construction_knobs.py
             "KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND",
             "KRABBY_STD_PARAM", "KRABBY_INIT_STD", "KRABBY_PPO_SCHEDULE", "KRABBY_PPO_LR",
             "KRABBY_PPO_DESIRED_KL", "KRABBY_PPO_KL_STOP", "KRABBY_PPO_HIST_LR", "KRABBY_PPO_HAND_DIAG",
             "KRABBY_NUM_ENVS",
             # PPO-construction plan step 4 (2026-10-01): RSI cam staging, env side (event params);
             # pinned in detail by tests/unit/test_rsi_cam_stage.py
             "KRABBY_RSI_CAM_STAGE", "KRABBY_RSI_CAM_STAGE_DEBUG")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_of(path: Path, *, cls: str | None, name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src, str(path))
    body = tree.body
    if cls is not None:
        body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    node = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


mk = _load(MOTION_KNOBS_PY, "motion_knobs__test")
mrew = _load(MOTION_REWARDS_PY, "motion_rewards__test")
if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as clock  # noqa: E402
import crab_hex_linkage as linkage  # noqa: E402


# ================================================================= the knob module (pure)
def test_motion_knobs_module_is_pure():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('mk', {str(MOTION_KNOBS_PY)!r})\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['mk'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert mod.parse_relay_hyst('1') == (0.25, 0.05)\n"
        "print(sorted(m for m in sys.modules if m.startswith(('parkour_tasks', 'isaaclab', 'torch'))))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_variable_names_are_the_contract():
    got = {getattr(mk, a) for a in dir(mk) if a.endswith("_VAR")}
    assert set(ALL_KNOBS) <= got
    assert set(mk.MOTION_KNOB_PARSERS) == set(ALL_KNOBS)
    assert set(mk.AGENT_KNOBS) == {"KRABBY_ACTION_STD_MAX", "KRABBY_ACTION_STD_RESET", "KRABBY_ENTROPY_COEF",
                                   # PPO construction (2026-10-01): every one but the driver's NUM_ENVS
                                   *(set(mk.PPO_CONSTRUCTION_KNOBS) - {"KRABBY_NUM_ENVS"})}
    assert mk.REUSED_WEIGHT_KNOBS == {"KRABBY_POWER_W": "penalty_mechanical_power",
                                      "KRABBY_ACTION_RATE_W": "reward_action_rate",
                                      "KRABBY_AIRTIME_W": "reward_feet_air_time_positive"}
    # the reused three really exist in the forward rewards cfg's override table (not re-read here)
    mdp_src = (TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py").read_text()
    for var, term in mk.REUSED_WEIGHT_KNOBS.items():
        assert f'"{var}": "{term}"' in mdp_src, var


def test_parsers():
    assert mk.parse_flag("1") is True and mk.parse_flag("0") is False and mk.parse_flag(" ") is False
    for bad in ("2", "true", "yes"):
        with pytest.raises(ValueError):
            mk.parse_flag(bad)
    assert mk.parse_std_max("0.5") == 0.5 and mk.parse_std_max("") is None
    for bad in ("0.05", "0", "-1", "nan", "x"):
        with pytest.raises(ValueError):
            mk.parse_std_max(bad)
    assert mk.parse_std_reset("0.05") == 0.05 and mk.parse_std_reset(None) is None
    with pytest.raises(ValueError):
        mk.parse_std_reset("0.01")
    assert mk.parse_entropy_coef("0") == 0.0 and mk.parse_entropy_coef("0.002") == 0.002
    with pytest.raises(ValueError):
        mk.parse_entropy_coef("-0.01")
    assert mk.parse_penalty_w("-0.01") == -0.01 and mk.parse_penalty_w("0") == 0.0
    assert mk.parse_penalty_w("") == 0.0
    with pytest.raises(ValueError, match="<= 0"):
        mk.parse_penalty_w("0.01")
    assert mk.parse_lpf_hz("4") == 4.0 and mk.parse_lpf_hz("0") is None and mk.parse_lpf_hz("") is None
    for bad in ("-1", "100", "inf"):
        with pytest.raises(ValueError):
            mk.parse_lpf_hz(bad)
    assert mk.parse_relay_hyst("1") == (0.25, 0.05) == mk.RELAY_HYST_DEFAULT
    assert mk.parse_relay_hyst("0.3:0.1") == (0.3, 0.1)
    for off in ("", "0", "off", "OFF", None):
        assert mk.parse_relay_hyst(off) is None
    for bad in ("0.1:0.1", "0.1:0.2", "0.2:-0.1", "0.3", "a:b", "0.3:0.1:0"):
        with pytest.raises(ValueError):
            mk.parse_relay_hyst(bad)


def test_every_off_value_parses_to_off():
    for var, off in mk.KNOB_OFF_VALUES.items():
        value = mk.MOTION_KNOB_PARSERS[var](off)
        assert value in (None, False, 0.0), (var, value)
    assert set(mk.KNOB_OFF_VALUES) == set(mk.EVAL_SEMANTIC_KNOBS) | set(mk.EVAL_STRUCTURE_KNOBS)


# ------------------------------------------------------------------- stand-in env cfg
class _Bag:
    pass


class _Term:
    def __init__(self, func=None, weight=0.0, params=None):
        self.func, self.weight, self.params = func, weight, params


class _OmniObsFunc:
    OBSERVED_COMMAND_SLOTS = {8: 0, 9: 1, 5: 2}


def _cfg(with_obs=True):
    cfg = _Bag()
    cfg.commands = _Bag()
    cfg.commands.base_velocity = _Bag()
    cfg.actions = _Bag()
    cfg.actions.joint_pos = _Bag()
    cfg.rewards = _Bag()
    cfg.rewards.reward_action_rate = _Term(func="rate", weight=-0.3, params={"asset_cfg": "robot"})
    cfg.observations = _Bag()
    cfg.observations.policy = _Bag()
    if with_obs:
        cfg.observations.policy.extreme_parkour_observations = _Term(
            func=_OmniObsFunc, params={"history_length": 10, "facing_obs_encoding": "raw"})
    return cfg


def _snap(obj):
    if isinstance(obj, (_Bag, _Term)):
        return {k: _snap(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _snap(v) for k, v in obj.items()}
    return obj


BUILDERS = {name: (f"func:{name}", {"p": name}) for name in
            (mk.SMOOTH2_TERM, mk.JOINT_VEL_TERM, mk.FOOT_SLIP_TERM, mk.STAND_POSE_TERM)}


def _apply(cfg, env):
    mk.apply_motion_knobs(cfg, env, term_cls=_Term, builders=BUILDERS)


def test_every_env_knob_unset_blank_or_off_writes_nothing():
    pristine = _snap(_cfg())
    for env in ({}, {k: "" for k in ALL_KNOBS},
                {**mk.KNOB_OFF_VALUES, "KRABBY_ACTION_RATE_SQ": "0", "KRABBY_ACTION_SMOOTH2_W": "0",
                 "KRABBY_JOINT_VEL_W": "0", "KRABBY_FOOT_SLIP_W": "0", "KRABBY_STAND_POSE_W": "0"}):
        cfg = _cfg()
        _apply(cfg, env)
        assert _snap(cfg) == pristine, env


def test_regulariser_knobs():
    cfg = _cfg()
    _apply(cfg, {"KRABBY_ACTION_RATE_SQ": "1", "KRABBY_ACTION_SMOOTH2_W": "-0.02",
                 "KRABBY_JOINT_VEL_W": "-1e-4", "KRABBY_FOOT_SLIP_W": "-0.5", "KRABBY_STAND_POSE_W": "-0.3"})
    assert cfg.rewards.reward_action_rate.params == {"asset_cfg": "robot", "squared": True}
    assert cfg.rewards.reward_action_rate.weight == -0.3            # the weight stays ACTION_RATE_W's
    for name, w in ((mk.SMOOTH2_TERM, -0.02), (mk.JOINT_VEL_TERM, -1e-4), (mk.FOOT_SLIP_TERM, -0.5),
                    (mk.STAND_POSE_TERM, -0.3)):
        term = getattr(cfg.rewards, name)
        assert (term.func, term.weight, term.params) == (f"func:{name}", w, {"p": name})
    # only the armed ones are added
    cfg2 = _cfg()
    _apply(cfg2, {"KRABBY_FOOT_SLIP_W": "-0.5"})
    assert hasattr(cfg2.rewards, mk.FOOT_SLIP_TERM) and not hasattr(cfg2.rewards, mk.SMOOTH2_TERM)
    with pytest.raises(ValueError, match="<= 0"):
        _apply(_cfg(), {"KRABBY_JOINT_VEL_W": "0.1"})
    no_rate = _cfg()
    del no_rate.rewards.reward_action_rate
    with pytest.raises(RuntimeError, match="reward_action_rate"):
        _apply(no_rate, {"KRABBY_ACTION_RATE_SQ": "1"})


def test_the_real_term_builders_name_real_functions_and_joint_sets():
    """The Isaac-side builder table, checked by SOURCE (building it needs Isaac): every added term
    points at a function the rewards module defines, with the params that function takes."""
    src = MOTION_KNOBS_PY.read_text()
    for fn in ("penalty_action_second_diff", "penalty_joint_vel_l2", "penalty_loaded_foot_slip",
               "penalty_stand_pose_l1"):
        assert f"rew.{fn}" in src and callable(getattr(mrew, fn))
    assert mk._REWARDS_MOD.endswith("crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards")
    assert mk.FOOT_SLIP_FORCE_N == clock.TOUCH_REF_N == 50.0


def test_action_path_knobs():
    cfg = _cfg()
    _apply(cfg, {"KRABBY_ACTION_LPF_HZ": "4"})
    assert cfg.actions.joint_pos.lpf_hz == 4.0
    assert "lpf_obs" not in cfg.observations.policy.extreme_parkour_observations.params
    _apply(cfg, {"KRABBY_ACTION_LPF_HZ": "4", "KRABBY_ACTION_LPF_OBS": "1"})
    assert cfg.observations.policy.extreme_parkour_observations.params["lpf_obs"] is True
    with pytest.raises(RuntimeError, match="KRABBY_ACTION_LPF_HZ"):
        _apply(_cfg(), {"KRABBY_ACTION_LPF_OBS": "1"})
    with pytest.raises(RuntimeError, match="no omni observation term"):
        _apply(_cfg(with_obs=False), {"KRABBY_ACTION_LPF_HZ": "4", "KRABBY_ACTION_LPF_OBS": "1"})


def test_command_and_stand_knobs():
    cfg = _cfg()
    _apply(cfg, {"KRABBY_OMNI_RELAY_HYST": "1"})
    assert cfg.commands.base_velocity.relay_hyst == (0.25, 0.05)
    assert not hasattr(cfg.commands.base_velocity, "eval_slot_latch")   # not stand-keyed
    for var, check in (("KRABBY_OMNI_STAND_HOLD", lambda c: c.commands.base_velocity.stand_hold is True),
                       ("KRABBY_OMNI_STAND_PARK", lambda c: c.omni_stand_park is True),
                       ("KRABBY_OMNI_STAND_OBS",
                        lambda c: c.observations.policy.extreme_parkour_observations.params["stand_obs"] is True)):
        cfg = _cfg()
        _apply(cfg, {var: "1"})
        assert check(cfg), var
        # every stand-keyed knob latches the eval hook's slot per hold
        assert cfg.commands.base_velocity.eval_slot_latch is True, var


def test_exploration_knobs():
    agent = _Bag()
    agent.policy, agent.algorithm = _Bag(), _Bag()
    agent.algorithm.entropy_coef = 0.01
    mk.apply_exploration_knobs(agent, {})
    assert vars(agent.policy) == {} and agent.algorithm.entropy_coef == 0.01
    mk.apply_exploration_knobs(agent, {"KRABBY_ACTION_STD_MAX": "0.5", "KRABBY_ACTION_STD_RESET": "0.5",
                                       "KRABBY_ENTROPY_COEF": "0.0"})
    assert agent.policy.action_std_max == 0.5 and agent.policy.action_std_reset == 0.5
    assert agent.algorithm.entropy_coef == 0.0
    with pytest.raises(ValueError, match="clamp ceiling"):
        mk.apply_exploration_knobs(agent, {"KRABBY_ACTION_STD_MAX": "0.5", "KRABBY_ACTION_STD_RESET": "0.6"})
    with pytest.raises(ValueError, match="clamp ceiling"):
        mk.apply_exploration_knobs(agent, {"KRABBY_ACTION_STD_RESET": "2.5"})   # over the stock 2.0


def test_omni_knobs_reexports_and_applies_the_motion_knobs_before_stroke_posture():
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__motion_test")
    for var in ALL_KNOBS + ("KRABBY_POWER_W", "KRABBY_ACTION_RATE_W", "KRABBY_AIRTIME_W"):
        assert var in {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}, var
    assert knobs.MOTION_KNOB_PARSERS is knobs._mk.MOTION_KNOB_PARSERS
    fn = [n for n in ast.parse(OMNI_KNOBS_PY.read_text()).body
          if isinstance(n, ast.FunctionDef) and n.name == "apply_omni_knobs"][0]
    calls = [ast.unparse(s) for s in fn.body if isinstance(s, ast.Expr)]
    # NOTE(stride/posture plan step 3, 2026-09-25): the stroke/posture knobs (stroke_knobs.py) now
    # apply LAST of all, so a term they add sees the motion knobs' own additions too.
    assert calls[-1] == "_sk.apply_stroke_knobs(cfg, env)"
    assert calls[-2] == "_mk.apply_motion_knobs(cfg, env)"
    # the runner side: both omni runner cfgs call the exploration applier
    runner_src = (OMNI / "config/crab_hex/agents/rsl_rl_ppo_cfg.py").read_text()
    assert runner_src.count("_apply_exploration_knobs(self)") == 2


def test_no_preset_arms_a_motion_knob():
    src = OMNI_PHASES_PY.read_text() + (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_phases.py").read_text()
    for var in ALL_KNOBS:
        assert var not in src, var


# ======================================================== the added penalty terms (pure)
class _Scene(dict):
    def __init__(self, robot, sensors=None):
        super().__init__(robot=robot)
        self.sensors = sensors or {}


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_second_difference_penalty():
    buf = torch.zeros(3, 8, 2)
    buf[:, -3] = torch.tensor([0.0, 1.0])
    buf[:, -2] = torch.tensor([1.0, 1.0])
    buf[:, -1] = torch.tensor([1.0, 3.0])
    term = _ns(action_history_buf=buf)
    env = _ns(action_manager=_ns(get_term=lambda name: term), episode_length_buf=torch.tensor([1, 2, 3]))
    # d2 = a_t - 2 a_{t-1} + a_{t-2} = (1 - 2 + 0, 3 - 2 + 1) = (-1, 2) -> 5; first two steps masked
    assert torch.equal(mrew.penalty_action_second_diff(env), torch.tensor([0.0, 0.0, 5.0]))
    # a constant or linear action sequence costs nothing
    lin = torch.arange(8.0).view(1, 8, 1).repeat(2, 1, 3)
    env.action_manager = _ns(get_term=lambda name: _ns(action_history_buf=lin))
    env.episode_length_buf = torch.tensor([50, 50])
    assert torch.equal(mrew.penalty_action_second_diff(env), torch.zeros(2))
    env.action_manager = _ns(get_term=lambda name: _ns(action_history_buf=torch.zeros(2, 2, 3)))
    with pytest.raises(RuntimeError, match=">= 3"):
        mrew.penalty_action_second_diff(env)


def test_joint_vel_and_stand_pose_penalties():
    data = _ns(joint_vel=torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]]),
               joint_pos=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.5, 0.0]]),
               default_joint_pos=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    env = _ns(scene=_Scene(_ns(data=data)))
    cfg = _ns(name="robot", joint_ids=[1, 2])
    assert torch.allclose(mrew.penalty_joint_vel_l2(env, cfg), torch.tensor([13.0, 1.25]))
    cmd_term = _ns(stand_mask=lambda: torch.tensor([True, False]))
    env.command_manager = _ns(get_term=lambda name: cmd_term)
    assert torch.allclose(mrew.penalty_stand_pose_l1(env, cfg), torch.tensor([0.2 + 0.7, 0.0]))


def test_loaded_foot_slip_penalty_is_unsquared_and_load_gated():
    forces = torch.zeros(2, 3, 3)
    forces[0, :, 2] = torch.tensor([10.0, 60.0, 200.0])          # only feet 1 and 2 loaded
    forces[1, :, 2] = torch.tensor([51.0, 49.0, 0.0])
    sensor = _ns(data=_ns(net_forces_w=forces))
    vel = torch.zeros(2, 3, 3)
    vel[:, :, 0] = 0.3
    vel[:, :, 1] = 0.4                                            # |v_xy| = 0.5 per foot
    vel[:, :, 2] = 9.0                                            # vertical speed is not slip
    env = _ns(scene=_Scene(_ns(data=_ns(body_lin_vel_w=vel)), {"contact_forces": sensor}))
    got = mrew.penalty_loaded_foot_slip(env, _ns(name="contact_forces", body_ids=[0, 1, 2]),
                                        _ns(name="robot", body_ids=[0, 1, 2]))
    assert torch.allclose(got, torch.tensor([1.0, 0.5]))


# ================================================= rewards.py: squared rate, clock park gate
class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self.device = cfg, "cpu"


def _lift_rewards():
    ns = {"torch": torch, "ManagerTermBase": _ManagerTermBase, "Sequence": object,
          "Articulation": object, "SceneEntityCfg": object, "ParkourManagerBasedRLEnv": object,
          "RewardTermCfg": object}
    for name in ("reward_action_rate", "_clock_running"):
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, cls=None, name=name),
                     str(LIB_REWARDS_PY), "exec"), ns)
    return ns


def test_action_rate_squared_is_opt_in_and_the_default_is_the_norm():
    ns = _lift_rewards()
    raw = [torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([[3.0, 4.0], [1.0, 2.0]])]
    term_holder = _ns(raw_actions=None, _num_joints=2)
    env = _ns(num_envs=2, action_manager=_ns(get_term=lambda name: term_holder))
    plain = ns["reward_action_rate"](_ns(params={"asset_cfg": _ns(name="robot")}), env)
    sq = ns["reward_action_rate"](_ns(params={"asset_cfg": _ns(name="robot")}), env)
    for step, a in enumerate(raw):
        term_holder.raw_actions = a
        p = plain(env, None)
        s = sq(env, None, squared=True)
    assert torch.allclose(p, torch.tensor([5.0, 1.0]))          # ||(3,4)-(0,0)||, ||(0,1)||
    assert torch.allclose(s, torch.tensor([25.0, 1.0]))


def test_clock_running_park_gate():
    ns = _lift_rewards()
    ns["_clock_cmd_speed"] = lambda env, cmd: cmd[:, 0]
    parked = torch.tensor([True, False, True, False])
    env = _ns(action_manager=_ns(get_term=lambda name: _ns(clock_parked=parked)))
    cmd = torch.tensor([[0.3], [0.3], [0.1], [0.1]])
    assert torch.equal(ns["_clock_running"](env, cmd, 0.2, False), torch.tensor([True, True, False, False]))
    assert torch.equal(ns["_clock_running"](env, cmd, 0.2, True), torch.tensor([False, True, False, False]))


# ========================================================= clock park and the rod low-pass
def test_park_clock_phase_goes_to_the_next_touchdown_and_holds():
    pi = math.pi
    phase = torch.tensor([0.1, pi - 0.05, pi, pi + 0.3, 2 * pi - 0.01, 0.0])
    got = clock.park_clock_phase(phase, 0.2)
    assert torch.allclose(got, torch.tensor([0.3, pi, pi, pi + 0.5, 0.0, 0.0]), atol=1e-6)
    # iterating reaches the boundary and stays: set A touches down at pi, set B at 0 (== 2 pi)
    p = torch.tensor([0.1, 3.5])
    for _ in range(40):
        p = clock.park_clock_phase(p, 0.2)
    assert torch.allclose(p, torch.tensor([pi, 0.0]), atol=1e-6)
    # at a touchdown every foot is at the edge of its swing window: sin(phase + offset) == 0
    offsets = torch.tensor(clock.FOOT_OFFSETS)
    assert torch.allclose(torch.sin(p[:, None] + offsets[None, :]), torch.zeros(2, 6), atol=1e-5)
    # float32 rounding a hair past a boundary never sends the clock round another half cycle
    near = torch.tensor([pi], dtype=torch.float32) + 1e-6
    assert float(clock.park_clock_phase(near, 0.2)) == pytest.approx(pi, abs=1e-5)


def test_rod_low_pass_helpers():
    dt = 0.005
    a = linkage.lpf_alpha(4.0, dt)
    assert a == pytest.approx(1.0 - math.exp(-2 * math.pi * 4.0 * dt))
    for bad in ((0.0, dt), (-1.0, dt), (4.0, 0.0)):
        with pytest.raises(ValueError):
            linkage.lpf_alpha(*bad)
    # a step settles to 63.2% in one time constant tau = 1 / (2 pi f)
    y = torch.zeros(1)
    steps = round(1.0 / (2 * math.pi * 4.0) / dt)
    for _ in range(steps):
        y = linkage.lpf_step(y, torch.ones(1), a)
    assert float(y) == pytest.approx(1.0 - math.exp(-steps * 2 * math.pi * 4.0 * dt), rel=1e-5)
    mid_h = (linkage.HIP_LEN_MIN_M + linkage.HIP_LEN_MAX_M) / 2
    assert float(linkage.hip_rod_normalized(torch.tensor(mid_h))) == pytest.approx(0.0, abs=1e-7)
    assert float(linkage.hip_rod_normalized(torch.tensor(linkage.HIP_LEN_MAX_M))) == pytest.approx(1.0)
    assert float(linkage.knee_rod_normalized(torch.tensor(linkage.KNEE_LEN_MIN_M))) == pytest.approx(-1.0)


# ------------------------------------- the action term's clock and pitch linkage, lifted
class _Dims:
    CLOCK_CMD_STOP_M_S = 0.2
    CLOCK_V_MAX_M_S = 0.35
    CLOCK_F_MAX_REV_S = 0.5


def _action_ns():
    ns = {"torch": torch, "math": math, "linkage": linkage, "park_clock_phase": clock.park_clock_phase,
          "clock_command_speed": clock.clock_command_speed}
    for name in ("process_actions", "_apply_pitch_linkage", "rod_cmd_filtered"):
        exec(compile("from __future__ import annotations\n"
                     + _source_of(ACTIONS_PY, cls="CrabHexDelayedJointPositionAction", name=name),
                     str(ACTIONS_PY), "exec"), ns)
    return ns


class _Asset:
    def __init__(self, n):
        self.data = _ns(joint_pos=torch.zeros(n, 12), default_joint_pos=torch.zeros(n, 12))
        self.pos_targets = {}

    def set_joint_position_target(self, t, joint_ids):
        self.pos_targets[tuple(joint_ids)] = t.clone()

    def set_joint_velocity_target(self, t, joint_ids):
        pass


def _action_term(n, *, park=False, lpf_alpha=None, slot=None):
    ns = _action_ns()
    cmd = torch.zeros(n, 3)
    cmd_term = _ns(stand_mask=lambda: slot == 0)
    env = _ns(command_manager=_ns(get_command=lambda name: cmd, get_term=lambda name: cmd_term),
              cfg=_ns(omni_command=True, omni_clock_turn_radius=1.2), step_dt=0.02, physics_dt=0.005,
              common_step_counter=1)
    t = _ns()
    t.process_actions = types.MethodType(ns["process_actions"], t)
    t._apply_pitch_linkage = types.MethodType(ns["_apply_pitch_linkage"], t)
    t.rod_cmd_filtered_fn = types.MethodType(ns["rod_cmd_filtered"], t)   # the property's getter
    t._env = t.env = env
    t.clock_phase = torch.linspace(0.1, 6.0, n)
    t._stand_park = park
    t._delay_update_global_steps = 1000
    t._action_delay_steps = []
    t._clip_raw_actions = lambda a: a
    t._action_history_buf = torch.zeros(n, 8, 18)
    t._use_delay = False
    t.delay = torch.tensor(1.0)
    t._raw_actions = torch.zeros(n, 18)
    t._scale, t._offset = 0.25, torch.zeros(n, 18)
    t._lpf_alpha = lpf_alpha
    t._pitch_joint_ids, t._knee_joint_ids = list(range(6)), list(range(6, 12))
    t._pitch_action_cols, t._knee_action_cols = list(range(6, 12)), list(range(12, 18))
    t._knee_sign = torch.ones(1, 6)
    t._asset = _Asset(n)
    t._hip_rod_len = torch.full((n, 6), (linkage.HIP_LEN_MIN_M + linkage.HIP_LEN_MAX_M) / 2)
    t._knee_rod_len = torch.full((n, 6), (linkage.KNEE_LEN_MIN_M + linkage.KNEE_LEN_MAX_M) / 2)
    t._rod_sync_needed = torch.ones(n, dtype=torch.bool)
    if lpf_alpha is not None:
        t._hip_rod_cmd_lpf = t._hip_rod_len.clone()
        t._knee_rod_cmd_lpf = t._knee_rod_len.clone()
    return t, cmd


def test_stand_park_parks_only_stand_rows_and_is_off_by_default():
    n = 4
    slot = torch.tensor([0, 1, 0, 1])
    plain, cmd_p = _action_term(n, slot=slot)
    parked, cmd_k = _action_term(n, park=True, slot=slot)
    for c in (cmd_p, cmd_k):
        c[:, 2] = 0.3                                   # a relay yaw rate keeps the clock running
    before = plain.clock_phase.clone()
    plain.process_actions(torch.zeros(n, 18))
    parked.process_actions(torch.zeros(n, 18))
    step = 2 * math.pi * min(1.2 * 0.3 / 0.35, 1.0) * 0.5 * 0.02
    assert torch.allclose(plain.clock_phase, torch.remainder(before + step, 2 * math.pi))
    moving = slot != 0
    assert torch.equal(parked.clock_phase[moving], plain.clock_phase[moving])
    park_step = 2 * math.pi * 0.5 * 0.02
    assert torch.allclose(parked.clock_phase[~moving], clock.park_clock_phase(before[~moving], park_step))
    assert torch.equal(parked.clock_parked, slot == 0)
    assert not hasattr(plain, "clock_parked")


def test_rod_low_pass_is_bitwise_off_by_default_and_filters_when_armed():
    n = 3
    plain, _ = _action_term(n)
    alpha = linkage.lpf_alpha(4.0, 0.005)
    filt, _ = _action_term(n, lpf_alpha=alpha)
    step_cmd = torch.zeros(n, 18)
    step_cmd[:, 6:12] = 0.4                               # a hip-pitch step command (rad)
    for t in (plain, filt):
        t._processed_actions = step_cmd.clone()
        t._apply_pitch_linkage()
    lam_cmd = linkage.hip_actuator_length(step_cmd[:, 6:12]).clamp(linkage.HIP_LEN_MIN_M, linkage.HIP_LEN_MAX_M)
    mid = (linkage.HIP_LEN_MIN_M + linkage.HIP_LEN_MAX_M) / 2
    synced = linkage.hip_actuator_length(torch.zeros(n, 6))  # the rod sync on the first substep
    want_lpf = synced + alpha * (lam_cmd - synced)
    assert torch.allclose(filt._hip_rod_cmd_lpf, want_lpf)
    # the filtered command moves the rod LESS than the raw one on the first substep
    assert bool((torch.abs(filt._hip_rod_len - synced) <= torch.abs(plain._hip_rod_len - synced) + 1e-12).all())
    # the normalised observation block: [hip x6 | knee x6], mid-stroke = 0
    obs = filt.rod_cmd_filtered_fn()
    assert obs.shape == (n, 12)
    assert torch.allclose(obs[:, :6], (want_lpf - mid) / ((linkage.HIP_LEN_MAX_M - linkage.HIP_LEN_MIN_M) / 2))
    with pytest.raises(RuntimeError, match="lpf_hz"):
        plain.rod_cmd_filtered_fn()


# ============================================================ the policy's std clamp / reset
def _rsl_modules():
    if "_rslmods_motion" not in sys.modules:
        pkg = types.ModuleType("_rslmods_motion")
        pkg.__path__ = [str(RSL_MODULES)]
        sys.modules["_rslmods_motion"] = pkg
    return importlib.import_module("_rslmods_motion.crab_actor_critic_with_encoder")


OBS_W = 5 + 4 + 2 + 3 + 10 * 5          # prop + scan + priv_explicit + priv_latent + history


def _policy_kwargs(**extra):
    base = dict(num_prop=5, num_scan=4, num_hist=10, num_priv_latent=3, num_priv_explicit=2)
    kw = dict(
        init_noise_std=1.5, noise_std_type="scalar", actor_hidden_dims=[8], critic_hidden_dims=[8],
        activation="elu", tanh_encoder_output=False, scan_encoder_dims=[4, 2], priv_encoder_dims=[4],
        actor=dict(**base, class_name="Actor",
                   state_history_encoder=dict(**base, class_name="StateHistoryEncoder", channel_size=2)),
    )
    kw.update(extra)
    return kw


def test_std_clamp_and_reset_default_off():
    mod = _rsl_modules()
    pol = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs())
    assert pol.action_std_max == mod.ACTION_STD_MAX == 2.0 and pol.action_std_reset is None
    obs = torch.zeros(2, OBS_W)
    with torch.no_grad():
        pol.std.fill_(2.0003)                              # the X08:c6 reading
    pol.update_distribution(obs, False)
    assert torch.allclose(pol.distribution.stddev, torch.full((2, 3), 2.0))
    state = pol.state_dict()
    pol2 = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs())
    assert pol2.load_state_dict(state) is True
    assert torch.allclose(pol2.std, torch.full((3,), 2.0003))   # untouched without the knob


def test_std_clamp_and_reset_when_armed():
    mod = _rsl_modules()
    src = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs())
    with torch.no_grad():
        src.std.fill_(2.0003)
    pol = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs(action_std_max=0.5, action_std_reset=0.5))
    std_param = pol.std
    assert pol.load_state_dict(src.state_dict()) is True
    assert pol.std is std_param, "the reset is in place (the optimizer keeps its reference)"
    assert torch.equal(pol.std.detach(), torch.full((3,), 0.5))
    with torch.no_grad():
        pol.std.fill_(1.7)
    pol.update_distribution(torch.zeros(2, OBS_W), False)
    assert torch.allclose(pol.distribution.stddev, torch.full((2, 3), 0.5))
    log_pol = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs(noise_std_type="log", action_std_reset=0.3))
    log_pol.reset_action_std(0.3)
    assert torch.allclose(torch.exp(log_pol.log_std.detach()), torch.full((3,), 0.3))
    with pytest.raises(ValueError):
        mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs(action_std_max=0.01))


def test_std_reset_is_one_shot_a_load_within_the_ceiling_keeps_its_std():
    # A later stage / crash resume of an already-forked run loads a std within the ceiling: keep it.
    mod = _rsl_modules()
    src = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs())
    with torch.no_grad():
        src.std.fill_(0.37)
    pol = mod.CrabHexActorCriticRMA(OBS_W, 3, **_policy_kwargs(action_std_max=0.5, action_std_reset=0.4))
    pol.load_state_dict(src.state_dict())
    assert torch.allclose(pol.std.detach(), torch.full((3,), 0.37))


# ================================================================ the ladder's audit of every knob
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__motion_test")


ENV_YAML_OFF = textwrap.dedent("""\
    actions:
      joint_pos:
        class_type: x
        history_length: 8
    commands:
      base_velocity:
        ang_vel_max: 0.3
    observations:
      policy:
        extreme_parkour_observations:
          params:
            history_length: 10
            facing_obs_encoding: raw
    rewards:
      reward_action_rate:
        func: parkour_isaaclab.envs.mdp.rewards:reward_action_rate
        params:
          asset_cfg:
            name: robot
        weight: -0.3
      penalty_mechanical_power:
        func: x
        weight: -0.001
      reward_feet_air_time_positive:
        func: y
        weight: 0.4
    omni_command: true
    """)

ENV_YAML_ON = textwrap.dedent("""\
    actions:
      joint_pos:
        class_type: x
        history_length: 8
        lpf_hz: 4.0
    commands:
      base_velocity:
        ang_vel_max: 0.3
        relay_hyst: !!python/tuple
        - 0.25
        - 0.05
        stand_hold: true
        eval_slot_latch: true
    observations:
      policy:
        extreme_parkour_observations:
          params:
            history_length: 10
            facing_obs_encoding: raw
            lpf_obs: true
            stand_obs: true
    rewards:
      reward_action_rate:
        func: parkour_isaaclab.envs.mdp.rewards:reward_action_rate
        params:
          asset_cfg:
            name: robot
          squared: true
        weight: -0.3
      penalty_action_smooth2:
        func: z
        weight: -0.02
      penalty_joint_vel:
        func: z
        weight: -0.0001
      penalty_foot_slip:
        func: z
        weight: -0.5
      penalty_stand_pose:
        func: z
        weight: -0.3
    omni_command: true
    omni_stand_park: true
    """)

AGENT_YAML_OFF = "policy:\n  class_name: CrabHexActorCriticRMA\n  init_noise_std: 1.5\nalgorithm:\n  entropy_coef: 0.01\n"
AGENT_YAML_ON = ("policy:\n  class_name: CrabHexActorCriticRMA\n  init_noise_std: 1.5\n  action_std_max: 0.5\n"
                 "  action_std_reset: 0.5\nalgorithm:\n  entropy_coef: 0.0\n")

ON_OVERLAY = {"KRABBY_ACTION_RATE_SQ": "1", "KRABBY_ACTION_SMOOTH2_W": "-0.02", "KRABBY_JOINT_VEL_W": "-1e-4",
              "KRABBY_FOOT_SLIP_W": "-0.5", "KRABBY_STAND_POSE_W": "-0.3", "KRABBY_ACTION_LPF_HZ": "4",
              "KRABBY_ACTION_LPF_OBS": "1", "KRABBY_OMNI_RELAY_HYST": "1", "KRABBY_OMNI_STAND_HOLD": "1",
              "KRABBY_OMNI_STAND_PARK": "1", "KRABBY_OMNI_STAND_OBS": "1", "KRABBY_ACTION_STD_MAX": "0.5",
              "KRABBY_ACTION_STD_RESET": "0.5", "KRABBY_ENTROPY_COEF": "0"}
OFF_OVERLAY = {"KRABBY_ACTION_RATE_SQ": "0", "KRABBY_ACTION_SMOOTH2_W": "0", "KRABBY_JOINT_VEL_W": "0",
               "KRABBY_FOOT_SLIP_W": "0", "KRABBY_STAND_POSE_W": "0", "KRABBY_ACTION_LPF_HZ": "0",
               "KRABBY_ACTION_LPF_OBS": "0", "KRABBY_OMNI_RELAY_HYST": "0", "KRABBY_OMNI_STAND_HOLD": "0",
               "KRABBY_OMNI_STAND_PARK": "0", "KRABBY_OMNI_STAND_OBS": "0", "KRABBY_ENTROPY_COEF": "0.01",
               "KRABBY_POWER_W": "-0.001", "KRABBY_ACTION_RATE_W": "-0.3", "KRABBY_AIRTIME_W": "0.4"}


def _run_dir(tmp_path, env_yaml, agent_yaml):
    params = tmp_path / "params"
    params.mkdir(parents=True)
    (params / "env.yaml").write_text(env_yaml)
    (params / "agent.yaml").write_text(agent_yaml)
    ckpt = tmp_path / "model_19.pt"
    ckpt.write_text("")
    return {"ckpt": str(ckpt)}


def test_every_motion_knob_is_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in ALL_KNOBS + ("KRABBY_POWER_W", "KRABBY_ACTION_RATE_W", "KRABBY_AIRTIME_W"):
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
    # KRABBY_OMNI_MIRROR_IMAGES (side-speed-bisection plan step 1a, 2026-09-28) is the first
    # AGENT_DUMP_KEYS member that is NOT a motion-quality knob: the symmetry loss's
    # data_augmentation_func lives on the runner cfg for a reason unrelated to motion_knobs.py, so
    # it is asserted here explicitly rather than folded into mk.AGENT_KNOBS.
    assert rc.AGENT_DUMP_KEYS == set(mk.AGENT_KNOBS) | {"KRABBY_OMNI_MIRROR_IMAGES"}
    reasons = []
    rc._check_knob("KRABBY_JOINT_VEL_W", "0.1", "t", whitelist, parsers, reasons, set())
    assert reasons and "round-trip" in reasons[0]


def test_overlay_audit_passes_armed_knobs_against_both_dumps(rc, tmp_path):
    rec = _run_dir(tmp_path, ENV_YAML_ON, AGENT_YAML_ON)
    checks = rc.overlay_checks(rec, ON_OVERLAY)
    assert checks and all(ok for _n, ok in checks), [n for n, ok in checks if not ok]
    names = dict(checks)
    assert "overlay KRABBY_ACTION_STD_MAX=0.5 in params/agent.yaml" in names
    assert "overlay KRABBY_OMNI_STAND_PARK=1 in params/env.yaml" in names


def test_overlay_audit_passes_off_values_on_a_knob_free_dump(rc, tmp_path):
    rec = _run_dir(tmp_path, ENV_YAML_OFF, AGENT_YAML_OFF)
    checks = rc.overlay_checks(rec, OFF_OVERLAY)
    assert all(ok for _n, ok in checks), [n for n, ok in checks if not ok]


def test_overlay_audit_catches_a_knob_that_did_not_reach_the_cfg(rc, tmp_path):
    rec = _run_dir(tmp_path, ENV_YAML_OFF, AGENT_YAML_OFF)
    checks = dict(rc.overlay_checks(rec, ON_OVERLAY))
    assert not any(checks.values()), [n for n, ok in checks.items() if ok]
    # ... and an armed dump fails every OFF claim that it contradicts
    rec_on = _run_dir(tmp_path / "on", ENV_YAML_ON, AGENT_YAML_ON)
    off = {k: v for k, v in OFF_OVERLAY.items() if k not in ("KRABBY_POWER_W", "KRABBY_AIRTIME_W")}
    bad = [n for n, ok in rc.overlay_checks(rec_on, off) if not ok]
    assert len(bad) == len(off) - 1, bad          # ACTION_RATE_W -0.3 is true in both dumps


def test_agent_keys_need_the_agent_dump(rc, tmp_path):
    rec = _run_dir(tmp_path, ENV_YAML_ON, AGENT_YAML_ON)
    (Path(rec["ckpt"]).parent / "params" / "agent.yaml").unlink()
    assert rc.overlay_checks(rec, {"KRABBY_ACTION_STD_MAX": "0.5"})[0][1] is False
    assert all(ok for _n, ok in rc.overlay_checks(rec, {"KRABBY_OMNI_STAND_HOLD": "1"}))


def test_smoke_overlay_is_validated(rc):
    assert rc.smoke_overlay(["KRABBY_OMNI_STAND_HOLD=1", "KRABBY_ACTION_LPF_HZ = 4"]) == {
        "KRABBY_OMNI_STAND_HOLD": "1", "KRABBY_ACTION_LPF_HZ": "4"}
    assert rc.smoke_overlay(None) == {}
    for bad in (["KRABBY_OMNI_STAND_HOLD"], ["KRABBY_NOT_A_KNOB=1"], ["KRABBY_OMNI_STAND_HOLD=2"],
                ["KRABBY_PLANT=A15+B"]):
        with pytest.raises(SystemExit):
            rc.smoke_overlay(bad)
    ap = rc.build_parser()
    args = ap.parse_args(["--campaign-dir", "x", "--smoke-stage", "1a:20",
                          "--smoke-overlay", "KRABBY_OMNI_STAND_PARK=1", "--smoke-overlay", "KRABBY_ENTROPY_COEF=0"])
    assert args.smoke_overlay == ["KRABBY_OMNI_STAND_PARK=1", "KRABBY_ENTROPY_COEF=0"]
    item = rc.smoke_item(types.SimpleNamespace(args=args, seed_screen=3), "1a:20")
    stage = item["stages"][0]
    assert stage["overlay"] == item["recipe"]["overlay_by_stage"]["1a"] == {
        "KRABBY_OMNI_STAND_PARK": "1", "KRABBY_ENTROPY_COEF": "0"}
    assert rc.stage_env(item["recipe"], "1a") == {"KRABBY_OMNI_STAND_PARK": "1", "KRABBY_ENTROPY_COEF": "0"}
