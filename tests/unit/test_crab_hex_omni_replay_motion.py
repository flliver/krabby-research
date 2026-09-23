"""The gait-motion-quality replay gate (plan step 3, 2026-09-23): the replay math against the
training code, the gates' statistics, and the tool end to end on a synthetic suite stamp.

``experiments/tools/replay_motion_terms.py`` is loaded by path (pure numpy at module scope). Each
replayed term is pinned against the function that TRAINS:

* the four added penalty terms (``mdp/crab_hex_omni_motion_rewards.py``, pure torch) on stand-in
  envs driven step by step;
* ``reward_action_rate`` (plain and squared), ``penalty_mechanical_power`` and
  ``PenaltyFootIdleWhenForward`` (with the omni ``_cmd_gate`` / ``_clock_cmd_speed``) lifted out of
  ``parkour_isaaclab/envs/mdp/rewards.py`` with ``ast`` and exec'd against stubs -- the pattern of
  ``test_crab_hex_omni_motion_knobs.py``;
* the contact sensor's float32 clock re-derived in torch float32 (the runtime training uses) and
  compared bit for bit with the numpy emulation.

Nothing here imports Isaac, reads a real eval stamp or touches the campaign directory.
"""

import ast
import importlib.util
import json
import math
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
FWD_MDP = TASKS / "crab_hex_forward_task/mdp"
GAIT_SCRIPTS = TASKS / "crab_hex_forward_task/scripts"
REPLAY_PY = OMNI / "experiments/tools/replay_motion_terms.py"
MOTION_REWARDS_PY = OMNI / "mdp/crab_hex_omni_motion_rewards.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_of(path: Path, *, name: str) -> str:
    src = path.read_text()
    tree = ast.parse(src, str(path))
    node = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


R = _load(REPLAY_PY, "replay_motion_terms__test")
mrew = _load(MOTION_REWARDS_PY, "motion_rewards__replay_test")
mk = _load(MOTION_KNOBS_PY, "motion_knobs__replay_test")
if str(FWD_MDP) not in sys.path:
    sys.path.insert(0, str(FWD_MDP))
import crab_hex_clock_reward as clock  # noqa: E402


def _ns(**kw):
    return types.SimpleNamespace(**kw)


class _Scene(dict):
    def __init__(self, robot, sensors=None):
        super().__init__(robot=robot)
        self.sensors = sensors or {}


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self.device = cfg, "cpu"


def _lift(*names):
    """Exec the named functions / classes of rewards.py against stubs (no Isaac)."""
    ns = {"torch": torch, "ManagerTermBase": _ManagerTermBase, "Sequence": object,
          "Articulation": object, "ParkourManagerBasedRLEnv": object,
          "SceneEntityCfg": lambda name="robot", **kw: _ns(name=name, **kw),
          "RewardTermCfg": object, "clock_command_speed": clock.clock_command_speed}
    for name in names:
        exec(compile("from __future__ import annotations\n" + _source_of(LIB_REWARDS_PY, name=name),
                     str(LIB_REWARDS_PY), "exec"), ns)
    return ns


RNG = np.random.default_rng(7)


# ======================================================================= action terms
@pytest.mark.parametrize("squared", [False, True])
def test_action_rate_matches_the_training_term_with_the_one_step_delay(squared):
    """``reward_action_rate`` reads the action term's DELAYED ``raw_actions`` (history[-2] under
    action_delay_steps [1, 1]); the replay must reproduce every step, reset spike included."""
    ns = _lift("reward_action_rate")
    T, A = 30, 5
    mu = RNG.normal(0.0, 1.5, (T, A))
    a = R.clip_actions(mu, 1.0)
    holder = _ns(raw_actions=None, _num_joints=A)
    env = _ns(num_envs=1, action_manager=_ns(get_term=lambda name: holder))
    term = ns["reward_action_rate"](_ns(params={"asset_cfg": _ns(name="robot")}), env)
    hist = torch.zeros(1, 8, A)
    got = []
    for s in range(T):
        hist = torch.cat([hist[:, 1:], torch.tensor(a[s], dtype=torch.float32)[None, None]], dim=1)
        holder.raw_actions = hist[:, -2].clone()          # delay 1
        got.append(float(term(env, None, squared=squared)[0]))
    want = R.action_rate(a, delay=1, squared=squared)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert want[0] == 0.0 and want[1] == pytest.approx(np.linalg.norm(a[0]) ** (2 if squared else 1))


def test_second_difference_matches_the_training_term_step_by_step():
    T, A = 25, 4
    a = R.clip_actions(RNG.normal(0.0, 1.0, (T, A)), 1.0)
    hist = torch.zeros(1, 8, A)
    got = []
    for s in range(T):
        hist = torch.cat([hist[:, 1:], torch.tensor(a[s], dtype=torch.float32)[None, None]], dim=1)
        env = _ns(action_manager=_ns(get_term=lambda name, h=hist: _ns(action_history_buf=h)),
                  episode_length_buf=torch.tensor([s + 1]))    # incremented before the rewards
        got.append(float(mrew.penalty_action_second_diff(env)[0]))
    np.testing.assert_allclose(got, R.action_second_diff(a), rtol=1e-5, atol=1e-6)
    assert R.action_second_diff(a)[:2].tolist() == [0.0, 0.0]
    # a ramp costs nothing
    assert np.allclose(R.action_second_diff(np.arange(10.0)[:, None] * np.ones((1, 3))), 0.0)


def test_clip_is_the_runner_clip_actions():
    a = np.array([[2.0, -3.0, 0.5]])
    assert R.clip_actions(a, 1.0).tolist() == [[1.0, -1.0, 0.5]]
    assert R.clip_actions(a, None).tolist() == a.tolist()


# ======================================================================= state terms
def test_state_penalties_match_the_training_functions():
    N, J, F = 6, 24, 6
    qd = torch.tensor(RNG.normal(0, 2, (N, J)), dtype=torch.float64)
    q = torch.tensor(RNG.normal(0, 0.3, (N, J)), dtype=torch.float64)
    q0 = torch.tensor(RNG.normal(0, 0.3, (J,)), dtype=torch.float64)
    tau = torch.tensor(RNG.normal(0, 5, (N, J)), dtype=torch.float64)
    rods = list(range(12, 24))
    data = _ns(joint_vel=qd, joint_pos=q, default_joint_pos=q0.expand(N, J), applied_torque=tau)
    env = _ns(scene=_Scene(_ns(data=data)))
    cfg = _ns(name="robot", joint_ids=rods)
    np.testing.assert_allclose(mrew.penalty_joint_vel_l2(env, cfg).numpy(), R.joint_vel_l2(qd.numpy(), rods))
    env.command_manager = _ns(get_term=lambda name: _ns(stand_mask=lambda: torch.ones(N, dtype=torch.bool)))
    np.testing.assert_allclose(mrew.penalty_stand_pose_l1(env, cfg).numpy(),
                               R.stand_pose_l1(q.numpy(), q0.numpy(), rods))
    ns = _lift("penalty_mechanical_power")
    np.testing.assert_allclose(ns["penalty_mechanical_power"](env, _ns(name="robot")).numpy(),
                               R.mechanical_power(tau.numpy(), qd.numpy()))
    force = torch.tensor(RNG.uniform(0, 120, (N, F)), dtype=torch.float64)
    fvec = torch.zeros(N, F, 3, dtype=torch.float64)
    fvec[..., 2] = force
    vel = torch.tensor(RNG.normal(0, 0.3, (N, F, 3)), dtype=torch.float64)
    env2 = _ns(scene=_Scene(_ns(data=_ns(body_lin_vel_w=vel)), {"contact_forces": _ns(data=_ns(net_forces_w=fvec))}))
    got = mrew.penalty_loaded_foot_slip(env2, _ns(name="contact_forces", body_ids=list(range(F))),
                                        _ns(name="robot", body_ids=list(range(F))))
    np.testing.assert_allclose(got.numpy(), R.loaded_foot_slip(force.numpy(), vel.numpy(), 50.0))


def test_foot_idle_matches_the_training_term_under_the_omni_gate():
    ns = _lift("_cmd_gate", "_clock_cmd_speed", "PenaltyFootIdleWhenForward")
    S, F = 200, 6
    force = np.zeros((S, F))
    force[0] = 5.0                          # every foot touches once at the start, then idles
    force[60, 2] = 0.05                     # below the 0.1-N contact threshold: still idle
    cmd = np.zeros((S, 3))
    cmd[:, 0] = np.where(np.arange(S) < 120, 0.3, 0.0)
    cmd[150:, 2] = 0.2                     # a turn command opens the omni gate (1.2 * 0.2 > 0.12)
    sensor = _ns(data=_ns(net_forces_w_history=None))
    env = _ns(scene=_ns(sensors={"contact_forces": sensor}), num_envs=1,
              cfg=_ns(omni_command=True, omni_clock_turn_radius=1.2))
    params = {"command_name": "base_velocity", "sensor_cfg": _ns(name="contact_forces", body_ids=list(range(F))),
              "max_idle_steps": 90, "contact_force_threshold": 0.1, "min_forward_speed_cmd": 0.12}
    term = ns["PenaltyFootIdleWhenForward"](_ns(params=params), env)
    got = []
    for s in range(S):
        hist = torch.zeros(1, 2, F, 3)
        hist[0, 0, :, 2] = torch.tensor(force[s])
        sensor.data.net_forces_w_history = hist
        env.command_manager = _ns(get_command=lambda name, c=cmd[s]: torch.tensor(c, dtype=torch.float32)[None])
        got.append(float(term(env, **params)[0]))
    want = R.foot_idle(force, cmd, max_idle_steps=90, force_threshold=0.1, min_cmd=0.12, turn_r=1.2)
    np.testing.assert_allclose(got, want)
    assert want[120:150].sum() == 0.0 and want[150:].sum() > 0.0 and want[:120].sum() > 0.0


# ======================================================================= air time and the sensor clock
def _torch_sensor_elapsed(n, physics_dt=0.005, decimation=4, period=0.02):
    """``SensorBase.update`` + ``_update_outdated_buffers`` in torch float32 (the training runtime)."""
    t = torch.zeros(1)
    last = torch.zeros(1)
    outdated = torch.zeros(1, dtype=torch.bool)
    out = []
    for _ in range(n):
        el = torch.zeros(1)
        for _ in range(decimation):
            t += physics_dt
            outdated |= t - last + 1e-6 >= period
            if bool(outdated):
                el = t - last
                last = t.clone()
                outdated[:] = False
        out.append(float(el))
    return np.asarray(out, dtype=np.float32)


def test_sensor_clock_emulation_is_torch_float32_bit_for_bit():
    np.testing.assert_array_equal(R.sensor_elapsed_f32(1200), _torch_sensor_elapsed(1200))


def test_first_contact_stops_registering_after_two_seconds():
    el = torch.tensor(R.sensor_elapsed_f32(2000))
    ok = el < (0.02 + 1e-8)                      # compute_first_contact's test, float32 like training
    refreshed = el > 0                           # past ~16 s a step can carry no refresh at all
    assert ok[:95].float().mean() > 0.9
    assert not bool((ok & refreshed)[110:].any())


def _swing_trace(S, lift, land, F=6):
    force = np.full((S, F), 200.0)
    force[lift:land, 0] = 0.0                    # foot 0 airborne on rows [lift, land)
    return force


def test_air_time_exact_timing_pays_the_swing_at_touchdown():
    S = 60
    force = _swing_trace(S, 10, 40)              # 30 air steps -> last_air_time = 31 * 0.02
    cmd = np.tile([0.3, 0.0, 0.0], (S, 1))
    r = R.feet_air_time_positive(force, cmd, dt=0.02, sensor_threshold=1.0, threshold=0.38)
    assert np.flatnonzero(r).tolist() == [40]
    assert r[40] == pytest.approx(31 * 0.02 - 0.38)
    # the command gate: ||cmd_xy|| > 0.1 (a pure turn pays nothing)
    turn = np.tile([0.0, 0.0, 0.3], (S, 1))
    assert R.feet_air_time_positive(force, turn, dt=0.02, sensor_threshold=1.0, threshold=0.38).sum() == 0.0
    # a swing shorter than the threshold pays nothing
    short = _swing_trace(S, 10, 25)
    assert R.feet_air_time_positive(short, cmd, dt=0.02, sensor_threshold=1.0, threshold=0.38).sum() == 0.0


def test_air_time_training_clock_pays_early_swings_only():
    S = 400
    cmd = np.tile([0.3, 0.0, 0.0], (S, 1))
    el = R.sensor_elapsed_f32(S)
    early = R.feet_air_time_positive(_swing_trace(S, 20, 60), cmd, dt=0.02, sensor_threshold=1.0,
                                     threshold=0.38, elapsed=el)
    late = R.feet_air_time_positive(_swing_trace(S, 250, 290), cmd, dt=0.02, sensor_threshold=1.0,
                                    threshold=0.38, elapsed=el)
    assert early.sum() == pytest.approx(41 * 0.02 - 0.38, abs=1e-5)
    assert late.sum() == 0.0
    exact_late = R.feet_air_time_positive(_swing_trace(S, 250, 290), cmd, dt=0.02, sensor_threshold=1.0,
                                          threshold=0.38)
    assert exact_late.sum() == pytest.approx(41 * 0.02 - 0.38)


# ======================================================================= tracking income
def test_tracking_kernels_and_facing_terms():
    cmd = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]])
    v = np.array([[0.1, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    w = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.1], [0.0, 0.0, 0.4]])
    assert R.track_lin_vel_xy_exp(cmd, v, 0.2)[0] == pytest.approx(math.exp(-0.02 / 0.04))
    assert R.track_ang_vel_z_exp(cmd, w, 0.5)[1] == pytest.approx(math.exp(-0.04 / 0.25))
    yp = R.yaw_progress(cmd, w, 0.3)
    assert yp.tolist() == pytest.approx([0.0, 0.3, -0.1, 0.0])     # clipped, signed, gated on cmd_wz != 0
    assert R.facing_exp(np.array([0.0, -1.0])).tolist() == pytest.approx([1.0, math.exp(-1.0)])


def test_clock_command_speed_matches_the_training_seam():
    cmd = RNG.normal(0, 0.3, (20, 3))
    want = clock.clock_command_speed(torch.tensor(cmd), True, 1.2).numpy()
    np.testing.assert_allclose(R.clock_command_speed(cmd, 1.2), want)


def test_step_series_reads_the_post_step_state_row():
    """Row t is recorded BEFORE env.step(actions[t]): the reward of step s reads the state of row
    s + 1 and the command of row s."""
    T, J = 12, 24
    z = {k: np.zeros(s) for k, s in {
        "actions": (T, 18), "cmd_applied": (T, 3), "foot_force_norm": (T, 6), "joint_vel": (T, J),
        "applied_torque": (T, J), "joint_pos": (T, J), "foot_lin_vel_w": (T, 6, 3),
        "root_lin_vel_b": (T, 3), "root_ang_vel_b": (T, 3), "heading_err": (T,)}.items()}
    z["joint_vel"][5, 12] = 2.0
    z["cmd_applied"][3, 0] = 0.2
    p = R.TrainParams()
    s = R.step_series(z, p, rod_idx=list(range(12, 24)), q0=np.zeros(J))
    assert len(s["joint_vel"]) == T - 1
    assert np.flatnonzero(s["joint_vel"]).tolist() == [4]
    lin = s["inc_lin"]
    assert lin[3] == pytest.approx(math.exp(-0.04 / 0.04)) and lin[2] == pytest.approx(1.0)


# ======================================================================= statistics
def test_spearman_ranks_and_ties():
    assert R.rankdata([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    x = list(range(12))
    assert R.spearman(x, [v ** 3 for v in x])[0] == pytest.approx(1.0)
    assert R.spearman(x, [-v for v in x])[0] == pytest.approx(-1.0)
    assert R.spearman(x[:5], x[:5]) == (None, 5)                   # below MIN_RHO_N
    assert R.spearman(x, [1.0] * 12)[0] is None                     # constant side
    assert R.spearman(x + [None], x + [3.0])[1] == 12               # None pairs dropped
    scipy_stats = pytest.importorskip("scipy.stats")
    a, b = RNG.normal(size=40), RNG.normal(size=40)
    b[:10] = a[:10]
    assert R.spearman(a, b)[0] == pytest.approx(scipy_stats.spearmanr(a, b).statistic)


def test_sig2_and_quartile():
    assert R.sig2(0.000234) == 0.00023
    assert R.sig2(0.0987) == 0.099
    assert R.sig2(-0.2749) == -0.27
    assert R.sig2(0.0) == 0.0
    assert sorted(R.best_quartile([1, 5, 3, 9, 7, 2, 8, 4]).tolist()) == [3, 6]


def test_constants_are_the_training_ones():
    M = R.gait_metrics()
    assert R.MOTION_TOUCH_REF_N == M.MOTION_TOUCH_REF_N
    assert R.TrainParams().foot_slip_force == mk.FOOT_SLIP_FORCE_N
    assert R.TrainParams().sensor_force_threshold == 1.0


def test_train_params_from_a_config_dump():
    env = {
        "decimation": 4, "sim": {"dt": 0.005}, "episode_length_s": 40.0, "omni_clock_turn_radius": 1.2,
        "commands": {"base_velocity": {"ang_vel_max": 0.25}},
        "rewards": {
            "track_lin_vel_xy_exp": {"weight": 1.25, "params": {"std": 0.3},
                                     "func": "isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp"},
            "track_ang_vel_z_exp": {"weight": 1.0, "params": {"std": 0.5}, "func": "x:track_ang_vel_z_exp_turn_gated"},
            "reward_yaw_progress": {"weight": 0.5, "params": {"clip": None}},
            "reward_action_rate": {"weight": -0.2, "params": {"squared": True}},
            "reward_feet_air_time_positive": {"weight": 0.4, "params": {"threshold": 0.3}},
            "penalty_foot_slip": {"weight": -0.3, "params": {}},
        },
        "scene": {"contact_forces": {"force_threshold": 2.0, "update_period": 0.02},
                  "robot": {"init_state": {"joint_pos": {".*_Hip_Femur_RevoluteJoint": 0.1,
                                                         "FL_Femur_Tibia_RevoluteJoint": 0.3}}}},
        "actions": {"joint_pos": {"action_delay_steps": [1, 1], "use_delay": True}},
    }
    p = R.train_params_from_dumps(env, {"clip_actions": 1.0}, "test")
    assert (p.track_lin_std, p.yaw_progress_cap, p.action_rate_w, p.action_rate_squared) == (0.3, 0.25, -0.2, True)
    assert (p.airtime_w, p.airtime_threshold, p.sensor_force_threshold, p.action_delay) == (0.4, 0.3, 2.0, 1)
    assert any("turn_gated" in n for n in p.notes) and any("penalty_foot_slip" in n for n in p.notes)
    q0 = R.resolve_default_pos(p.default_joint_pos, ["FL_Hip_Femur_RevoluteJoint", "FL_Femur_Tibia_RevoluteJoint",
                                                     "FL_Body_Hip_RevoluteJoint"])
    assert q0.tolist() == [0.1, 0.3, 0.0]


# ======================================================================= the gates
def _ep(cat, sums, *, income=10.0, dur=10.0, jitter=0.1, switch=1.0, dist=2.0, yaw=1.0, swings=20.0):
    base = {k: 0.0 for k in ("power", "smooth2", "joint_vel", "action_rate_sq", "foot_slip", "airtime",
                             "stand_pose", "action_rate", "airtime_exact", "foot_idle")}
    base.update(sums)
    return {"category": cat, "sums": base, "income": income, "income_kernels": income * 0.9,
            "duration_s": dur, "quality": {"action_jitter": jitter, "contact_switch_hz": switch,
                                           "distance_m": dist, "net_yaw_rad": yaw, "duration_s": dur,
                                           "swings_per_m": swings / dist, "swings_per_rad": swings / yaw}}


def test_gate1_pass_fail_inactive():
    eps = []
    for i in range(12):
        # fwd: the joint-speed cost per metre rises with jitter -> PASS; power is noise -> FAIL
        eps.append(_ep("fwd", {"joint_vel": 2.0 * (i + 1), "power": float(RNG.uniform())},
                       jitter=0.01 * (i + 1), switch=float(RNG.uniform())))
    g = R.gate1(eps)
    assert g["joint_vel"]["rows"]["fwd"]["pass"] and g["joint_vel"]["rows"]["fwd"]["rho_jitter"] == pytest.approx(1.0)
    assert g["joint_vel"]["verdict"] == "PASS"
    assert g["foot_slip"]["verdict"] == "INACTIVE"                  # identically zero
    assert not g["stand_pose"]["rows"]["fwd"]["active"]             # stand-only term
    assert g["power"]["rows"]["fwd"]["rho_jitter"] is not None


def test_gate2_proposal_is_bound_by_the_costliest_category():
    eps = [_ep("fwd", {"power": 20.0}, income=10.0) for _ in range(8)]      # ratio 2 -> w10 0.05
    eps += [_ep("diag", {"power": 50.0}, income=10.0) for _ in range(8)]    # ratio 5 -> w10 0.02
    g2 = R.gate2(eps, R.TrainParams())
    t = g2["terms"]["power"]
    assert t["proposed_weight"] == -0.02 and t["binding_category"] == "diag"
    assert t["rows"]["fwd"]["share_at_weight"] == pytest.approx(0.04)
    assert t["rows"]["diag"]["share_at_weight"] == pytest.approx(0.10) and t["verdict"] == "PASS"
    # a fixed-weight term is only checked: action_rate_sq at -0.3 with ratio 1 costs 30 % -> FAIL
    eps2 = [_ep("fwd", {"action_rate_sq": 10.0, "action_rate": 5.0}) for _ in range(8)]
    g2b = R.gate2(eps2, R.TrainParams())
    assert g2b["terms"]["action_rate_sq"]["verdict"] == "FAIL"
    alt = R.m3_alternative(g2b, R.TrainParams())
    assert alt["KRABBY_ACTION_RATE_W"] == -0.15                      # cost-neutral with the stock -0.3


def test_gate2_quartile_is_the_best_tracking_one():
    eps = [_ep("side", {"foot_slip": float(i)}, income=float(10 + i)) for i in range(8)]
    g2 = R.gate2(eps, R.TrainParams())
    assert g2["quartile"]["side"]["n"] == 2
    # the two best-tracking episodes are i = 6, 7: ratios 6/16 and 7/17
    assert g2["terms"]["foot_slip"]["rows"]["side"]["median_ratio_q"] == pytest.approx((6 / 16 + 7 / 17) / 2)


# ======================================================================= end to end
JOINTS = ([f"{l}_Body_Hip_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Body_CamShaft_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Hip_Femur_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
          + [f"{l}_Femur_Tibia_RevoluteJoint" for l in ("FL", "FR", "ML", "MR", "RL", "RR")])
HOLDS = [("stand", "stand", 0.0, 0.0), ("fwd", "fwd", 0.25, 0.0), ("side", "lat_l", 0.22, 0.0),
         ("back", "back", 0.22, 0.0), ("diag", "diag", 0.25, 0.0), ("turn", "turn_l90", 0.0, 0.3),
         ("move_turn", "move_turn", 0.22, 0.3)]


def _synthetic_stamp(root: Path, per_cat: int = 9, T: int = 160, settle: int = 40) -> Path:
    run = root / "suite__omni/seed001/2026-01-01_00-00-00"
    (run / "raw").mkdir(parents=True)
    (run / "metrics").mkdir()
    labels = ["settle"] + [h[1] for h in HOLDS]
    emap, n_steps = [], []
    idx = 0
    for cat, label, speed, wz in HOLDS:
        for k in range(per_cat):
            rng = np.random.default_rng(1000 + idx)
            seg = np.zeros(T, dtype=np.int16)
            seg[settle:] = labels.index(label)
            cmd = np.zeros((T, 3))
            cmd[settle:, 0] = speed
            cmd[settle:, 2] = wz
            v = cmd + rng.normal(0, 0.05, (T, 3))
            yaw = np.cumsum(v[:, 2]) * 0.02
            pos = np.zeros((T, 3))
            pos[:, 0] = np.cumsum(v[:, 0]) * 0.02
            quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=1)
            force = rng.uniform(0, 150, (T, 6)) * (rng.uniform(0, 1, (T, 6)) > 0.3 - 0.02 * k)
            z = {"actions": rng.normal(0, 0.5 + 0.1 * k, (T, 18)), "cmd_applied": cmd,
                 "foot_force_norm": force, "foot_pos_w": rng.normal(0, 0.1, (T, 6, 3)),
                 "foot_lin_vel_w": rng.normal(0, 0.1, (T, 6, 3)), "root_pos_w": pos, "root_quat_w": quat,
                 "root_lin_vel_b": v, "root_ang_vel_b": np.stack([0 * yaw, 0 * yaw, v[:, 2]], axis=1),
                 "joint_vel": rng.normal(0, 1, (T, 24)), "applied_torque": rng.normal(0, 3, (T, 24)),
                 "joint_pos": rng.normal(0, 0.2, (T, 24)), "heading_err": rng.normal(0, 0.1, T),
                 "segment_id": seg, "steady_mask": np.ones(T, dtype=bool), "done": np.zeros(T, dtype=bool),
                 "dt": np.float64(0.02)}
            np.savez_compressed(run / "raw" / f"episode_{idx:02d}.npz", **z)
            (run / "metrics" / f"episode_{idx:02d}.json").write_text(json.dumps(
                {"category": cat, "hold": label, "termination_reason": "schedule_complete", "holds": {}}))
            emap.append({"category": cat, "env": idx, "env_index": idx, "hand": None, "label": label, "wave": 0})
            n_steps.append(T)
            idx += 1
    meta = {"checkpoint": None, "checkpoint_sha256": "0" * 64, "scenario_id": "suite__omni", "dt": 0.02,
            "joint_names": JOINTS, "hold_labels": labels, "env_index_map": emap, "episodes_total": idx,
            "n_steps_per_env": n_steps, "termination_reason": ["schedule_complete"] * idx,
            "suite": {"hold_s": {h[1]: (T - settle) * 0.02 for h in HOLDS}}}
    (run / "run_meta.json").write_text(json.dumps(meta))
    (run / "complete.json").write_text(json.dumps({"episodes_total": idx}))
    return run


def test_the_tool_end_to_end_on_a_synthetic_stamp(tmp_path):
    run = _synthetic_stamp(tmp_path)
    out = tmp_path / "out"
    assert R.main(["--run-dir", str(run), "--out", str(out), "--noise-sigmas", "0.5",
                   "--noise-samples", "1"]) == 0
    md = (out / "REPLAY_GATE.md").read_text()
    js = json.loads((out / "replay_gate.json").read_text())
    for knob in ("KRABBY_POWER_W", "KRABBY_ACTION_SMOOTH2_W", "KRABBY_JOINT_VEL_W", "KRABBY_FOOT_SLIP_W",
                 "KRABBY_AIRTIME_W", "KRABBY_STAND_POSE_W"):
        assert knob in md and js["proposed_weights"][knob] is not None
    for key in ("KRABBY_POWER_W", "KRABBY_ACTION_SMOOTH2_W", "KRABBY_JOINT_VEL_W", "KRABBY_FOOT_SLIP_W",
                "KRABBY_STAND_POSE_W"):
        assert js["proposed_weights"][key] < 0.0
    assert js["proposed_weights"]["KRABBY_AIRTIME_W"] == 0.4
    assert set(js["gate1"]["power"]["rows"]) == set(R.CATEGORIES)
    assert "Gate 1" in md and "Gate 2" in md and "What a FAIL means" in md
    # the stand-pose term only exists on the stand hold
    assert list(js["gate2"]["terms"]["stand_pose"]["rows"]) == ["stand"]


def test_the_tool_refuses_a_partial_stamp(tmp_path):
    run = _synthetic_stamp(tmp_path, per_cat=1)
    (run / "complete.json").unlink()
    with pytest.raises(SystemExit, match="complete.json"):
        R.main(["--run-dir", str(run), "--out", str(tmp_path / "o")])
