"""Stand-height campaign (``2026-10-13_stand_height``) step 2: the height DAgger trainer END TO END on CPU -- the REAL S1
and A-mask s17 networks (CPU copies) driving a FAKE env whose reset / command / termination / push paths run the
campaign's REAL env composition (``sh_starts.StartState`` through ``make_reset_event`` / ``make_command_class`` /
``wrap_termination`` / ``wrap_push``), with no Isaac / Kit / GPU.

Pinned:

* routing on one step: R episodes at sigma 0 execute R_h's target bitwise; R episodes with sigma execute
  q + clamp(residual + sigma noise); student episodes q + clamp(mean); S1 rows clamp(S1, +-1); restore rows are stand-net
  rows; every effective stand row enters the reservoir with R_h's residual label; the env wrapper clips at 4.8;
* a short run: one ``get_observations`` (T6), the global RNG untouched (T5), files, the forced first stands, the limiter
  init per start kind, the table park, the push suppression, the hip-contact re-arm, a same-seed rerun replays bitwise;
* the warm start (W[:, 15] zeroed, everything else A-mask s17, estimator == S1's) and the checkpoint round trip
  (infos: stand_mask 1, the height channel, the h range);
* the step-5 smoke list: every check present and boolean, the dose checks pass on the fake, forward parity bitwise;
  guards that must be able to fail (a router that drops the S1 clamp, a stand-net input that ignores the restore rows,
  a wrong label).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
SCRIPTS = TASK / "experiments/2026-10-13_stand_height/scripts"
W = 1149
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
JOINTS = ([f"{l}_Body_Hip_RevoluteJoint" for l in LEGS] + [f"{l}_Hip_Femur_RevoluteJoint" for l in LEGS]
          + [f"{l}_Body_CamShaft_RevoluteJoint" for l in LEGS] + [f"{l}_Femur_Tibia_RevoluteJoint" for l in LEGS])
HIP_J = [JOINTS.index(f"{l}_Hip_Femur_RevoluteJoint") for l in LEGS]
KNEE_J = [JOINTS.index(f"{l}_Femur_Tibia_RevoluteJoint") for l in LEGS]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


SC = _load(UNIT / "test_sn_common.py", "_sh_loop_t_sn_common")


@pytest.fixture(scope="module")
def CORE():
    return _load(SCRIPTS / "sh_dagger_core.py", "_t_sh_dagger_loop_core")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((SC.S1_DIR / "params/agent.yaml").read_text())


@pytest.fixture(scope="module")
def classes():
    if not SC.S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return SC.rsl_modules()
    finally:
        sys.dont_write_bytecode = old


@pytest.fixture(scope="module")
def bank(CORE):
    return CORE.STARTS.pose_bank_rows()


@pytest.fixture(scope="module")
def stance(CORE):
    return CORE.STARTS.stance_table()


@pytest.fixture(scope="module")
def nets_proto(CORE, classes, cfg):
    if not CORE.SH.STAND_NET_CKPT.is_file():
        pytest.skip("the A-mask s17 stand net is not on this machine")
    return CORE.build_networks(classes[0], classes[1], cfg, SC.S1_CKPT, "cpu")


def make_nets(nets_proto):
    return copy.deepcopy(nets_proto)


# ===================================================================================== the fake env
class FakeData:
    def __init__(self, n):
        self.root_link_pose_w = torch.zeros(n, 7)
        self.root_link_pose_w[:, 3] = 1.0
        self.root_vel_w = torch.zeros(n, 6)
        self.joint_pos = torch.zeros(n, len(JOINTS))

    @property
    def root_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_quat_w(self):
        return self.root_link_pose_w[:, 3:7]

    @property
    def heading_w(self):
        q = self.root_link_pose_w[:, 3:7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _rows(env_ids, n):
    return torch.arange(n) if env_ids is None else torch.as_tensor(env_ids).long()


class FakeRobot:
    def __init__(self, n):
        self.n = n
        self.joint_names = list(JOINTS)
        self.data = FakeData(n)
        self.targets = torch.zeros(n, len(JOINTS))
        self.writes = {"root": 0, "joints": 0}

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.data.root_link_pose_w[_rows(env_ids, self.n)] = pose.to(torch.float32)
        self.writes["root"] += 1

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        self.data.root_vel_w[_rows(env_ids, self.n)] = vel.reshape(-1, 6).to(torch.float32)

    def write_joint_state_to_sim(self, q, v, env_ids=None):
        self.data.joint_pos[_rows(env_ids, self.n)] = q.to(torch.float32)
        self.writes["joints"] += 1

    def set_joint_position_target(self, q, env_ids=None):
        self.targets[_rows(env_ids, self.n)] = q.to(torch.float32)

    def set_joint_velocity_target(self, v, env_ids=None):
        pass


class FakeTable:
    def __init__(self, n):
        self.n = n
        self.data = types.SimpleNamespace(root_link_pose_w=torch.zeros(n, 7))
        self.data.root_link_pose_w[:, 2] = -20.0
        self.data.root_link_pose_w[:, 3] = 1.0

    @property
    def root_pos(self):
        return self.data.root_link_pose_w[:, :3]

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.data.root_link_pose_w[_rows(env_ids, self.n)] = pose.to(torch.float32)


class FakeScene(dict):
    pass


class FakeCmdBase:
    """The command-term interface the wrapper and the trainer use (CommandTerm._resample semantics: time_left first,
    then _resample_command, then the counter)."""

    def __init__(self, env, n, g, p_stand=0.5, rs=(3.0, 12.0)):
        self._env, self.n, self.g, self.p_stand, self.rs = env, n, g, p_stand, rs
        self.robot = env.scene["robot"]
        self.time_left = torch.zeros(n)
        self.command_counter = torch.zeros(n, dtype=torch.long)
        self.slot_id = torch.ones(n, dtype=torch.long)
        self.type_id = torch.full((n,), -1, dtype=torch.long)
        self.speed_cmd = torch.zeros(n)
        self.heading_target = torch.zeros(n)
        self.vel_command_b = torch.zeros(n, 3)
        self._pool = types.SimpleNamespace(names=("fwd", "stand", "side_l"))
        self.write_calls = 0

    def _resample_command(self, ids):
        u = torch.rand(len(ids), generator=self.g)
        stand = u < self.p_stand
        self.slot_id[ids] = torch.where(stand, 0, 1)
        self.type_id[ids] = torch.where(stand, 1, 0)
        self.speed_cmd[ids] = torch.where(stand, 0.0, 0.3)
        self.heading_target[ids] = self.robot.data.heading_w[ids] + torch.where(stand, 0.0, 0.2)
        self._write_command(ids)

    def _write_command(self, ids):
        self.write_calls += 1
        self.vel_command_b[ids, 0] = self.speed_cmd[ids]
        self.vel_command_b[ids, 2] = 0.1 * (self.heading_target[ids] - self.robot.data.heading_w[ids])

    def _resample(self, ids):
        if len(ids):
            self.time_left[ids] = self.rs[0] + torch.rand(len(ids), generator=self.g) * (self.rs[1] - self.rs[0])
            self._resample_command(ids)
            self.command_counter[ids] += 1

    def reset(self, ids):
        self.command_counter[ids] = 0
        self._resample(ids)

    def compute(self, dt):
        self.time_left -= dt
        self._resample(torch.nonzero(self.time_left <= 0).flatten())

    def stand_mask(self):
        return self.slot_id == 0


def flat_raycast(starts, dirs, mesh=None):
    hits = starts.clone()
    hits[:, 2] = 0.0
    return (hits,)


class FakeEnv:
    """Wrapper-shaped: clips at 4.8; hips follow 0.25 x the rod action, the root height follows the hips' posture
    (1.064 + a_hip / 7), a table under the belly / the floor bounds it, cams integrate; rare tilt falls; a hip clause that
    fires below 0.40 m; a global push every ``push_every`` steps. Its own generator (never the global RNG); the start
    event, the command subclass, the termination and push wrappers are the campaign's."""

    def __init__(self, STARTS, gens, *, bank, stance, n=48, seed=0, dt=0.1, max_ep=400, p_fall=0.0005,
                 push_every=40, mesh=True):
        self.g = torch.Generator().manual_seed(1000 + seed)
        self.n, self.dt, self.max_ep, self.p_fall, self.push_every = n, dt, max_ep, p_fall, push_every
        self.num_envs, self.num_obs, self.num_actions, self.clip_actions = n, W, 18, 4.8
        self.robot, self.table = FakeRobot(n), FakeTable(n)
        self.scene = FakeScene(robot=self.robot, sh_table=self.table)
        self.scene.env_origins = torch.stack([torch.arange(n) * 10.0, torch.zeros(n), torch.zeros(n)], -1)
        self.state = STARTS.StartState(n, "cpu", gen_start=gens["starts"], gen_hcmd=gens["hcmd"],
                                       mesh_fn=(lambda: "mesh") if mesh else (lambda: None), raycast_fn=flat_raycast,
                                       bank=bank, stance=stance)
        self.reset_event = STARTS.make_reset_event(self.state)
        self.crab_failure = STARTS.wrap_termination(self._crab_failure_orig, self.state)
        self.push = STARTS.wrap_push(self._push_orig, self.state)
        self._fail = torch.zeros(n, dtype=torch.bool)
        self.applied = torch.zeros(n, 18)
        self.unwrapped = types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_term=lambda name: self.cmd),
            termination_manager=types.SimpleNamespace(get_term=lambda name: self._fail),
            episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=max_ep, step_dt=dt,
            scene=self.scene, action_manager=types.SimpleNamespace(action=self.applied))
        cls = STARTS.make_command_class(FakeCmdBase, self.state)
        self.cmd = cls(self.unwrapped, n, self.g)
        self.c = torch.zeros(n, 6)
        self.p = torch.zeros(n, 6)
        self.last_a = torch.zeros(n, 18)
        self.hist = torch.zeros(n, 10, 89)
        self.get_calls = 0
        self.steps = 0
        self.seen = []
        self.terrain_levels = torch.arange(n) % 5
        self._tilt = torch.zeros(n, dtype=torch.bool)
        self._reset(torch.arange(n))

    # -- the env's own pieces the wrappers wrap
    def _crab_failure_orig(self, env, asset_cfg=None, limit_angle=0.5, minimum_root_height_z=None,
                           contact_force_threshold=500.0, hip_contact_sensor_cfg=None):
        out = self._tilt.clone()
        if contact_force_threshold is not None:
            out = out | (self.robot.data.root_pos_w[:, 2] < 0.40)
        return out

    def _push_orig(self, env, env_ids, velocity_range=None):
        rows = _rows(env_ids, self.n)
        v = self.robot.data.root_vel_w[rows].clone()
        v[:, :2] = (torch.rand(len(rows), 2, generator=self.g) - 0.5)
        self.robot.write_root_velocity_to_sim(v, env_ids=env_ids)

    # -- reset: ring spawn, joint reset, the start event (LAST), the command reset, ep length 0
    def _reset(self, ids):
        k = len(ids)
        if k == 0:
            return
        yaw = (torch.rand(k, generator=self.g) * 2 - 1) * math.pi
        pose = torch.zeros(k, 7)
        pose[:, 0:2] = self.scene.env_origins[ids, :2]
        pose[:, 2] = 1.085
        pose[:, 3] = torch.cos(0.5 * yaw)
        pose[:, 6] = torch.sin(0.5 * yaw)
        self.robot.write_root_pose_to_sim(pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(k, 6), env_ids=ids)
        self.applied[ids] = 0.0               # IsaacLab ActionManager.reset: _action[env_ids] = 0 on every reset row
        jp = torch.zeros(k, len(JOINTS))
        th0, tk0 = float(KIN.TH0), float(KIN.TK0)
        scale = 0.95 + 0.1 * torch.rand(k, 1, generator=self.g)
        jp[:, HIP_J] = th0 * scale
        jp[:, KNEE_J] = tk0 * scale * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.robot.data.joint_pos[ids] = jp
        self.reset_event(self.unwrapped, ids)
        self.cmd.reset(ids)
        self.p[ids] = self.robot.data.joint_pos[ids][:, HIP_J] - th0
        self.c[ids] = 0.0
        self.hist[ids] = 0.0
        self.last_a[ids] = 0.0
        self.unwrapped.episode_length_buf[ids] = 0

    def _obs_buf(self):
        ob = torch.zeros(self.n, 89)
        ob[:, 0:3] = 0.01
        ob[:, 5] = self.cmd.vel_command_b[:, 2]
        ob[:, 8:10] = self.cmd.vel_command_b[:, :2]
        ob[:, 10] = self.cmd.vel_command_b[:, :2].norm(dim=1)
        ob[:, [18 + 2 * i for i in range(6)]] = self.c
        ob[:, [29 + i for i in range(6)]] = self.p
        ob[:, 65:83] = self.last_a
        ob[:, 83:89] = 0.2
        return ob

    def _obs(self):
        ob = self._obs_buf()
        o = torch.cat([ob, torch.full((self.n, 108), 0.05), torch.zeros(self.n, 62), self.hist.reshape(self.n, -1)], 1)
        self.hist = torch.cat([self.hist[:, 1:], ob[:, None]], 1)
        return o

    def get_observations(self):
        self.get_calls += 1
        o = self._obs()
        return o, {"observations": {"policy": o}}

    def step(self, actions):
        self.seen.append((actions.detach().clone(), self.cmd.stand_mask().clone()))
        a = actions.clamp(-self.clip_actions, self.clip_actions)
        self.applied.copy_(a)
        self.last_a = a.clone()
        up = self.p <= -0.06
        c = self.c + math.pi * self.dt * a[:, :6].clamp(-1, 1) * torch.where(up, 1.0, 0.3)
        self.c = torch.atan2(torch.sin(c), torch.cos(c))
        self.p = self.p + 0.6 * (0.25 * a[:, 6:12] - self.p)
        z = self.robot.data.root_link_pose_w[:, 2]
        h_t = 1.064 + (self.p.mean(dim=1) / 0.25) / 7.0
        z = z + 0.3 * (h_t - z)
        st = self.state
        floor = torch.where(st.table_active, (st.top_w + 0.15875).float(), torch.full_like(z, 0.30))
        self.robot.data.root_link_pose_w[:, 2] = torch.maximum(z, floor)
        u = self.unwrapped
        u.episode_length_buf = u.episode_length_buf + 1
        self._tilt = torch.rand(self.n, generator=self.g) < self.p_fall
        crab = self.crab_failure(self.unwrapped, limit_angle=0.5, contact_force_threshold=500.0)
        timeouts = u.episode_length_buf >= self.max_ep
        dones = crab | timeouts
        self._fail = crab
        self._reset(torch.nonzero(dones).flatten())
        self.cmd.compute(self.dt)
        self.steps += 1
        if self.steps % self.push_every == 0:
            self.push(self.unwrapped, None, velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5)})
        o = self._obs()
        return o, torch.zeros(self.n), dones.long(), {"time_outs": timeouts & ~crab, "observations": {"policy": o}}

    def probe(self):
        st = self.state
        z = self.robot.data.root_pos_w[:, 2].double()
        low = z - 0.15875
        base = torch.where(st.table_active, st.top_w, torch.zeros_like(z))
        feet = torch.where(low > base + 0.05, 6, 3)
        return {"root_z": z, "lowest_z": low, "h_meas": z.clone(), "feet_loaded": feet.long(),
                "table_top_w": self.table.data.root_link_pose_w[:, 2].double() + 1.0,
                "applied_action": self.applied.clone()}


KIN = _load(SCRIPTS / "sh_kin.py", "_t_sh_loop_kin")


def run_dagger(CORE, classes, cfg, nets_proto, bank, stance, out, *, seed=18, stop_after=6, iterations=60,
               smoke=True, n=48, terrain=True, fake_seed=None, **opts):
    nets = make_nets(nets_proto)
    gens = CORE.make_generators(seed, "cpu")
    env = FakeEnv(CORE.STARTS, gens, bank=bank, stance=stance, n=n, seed=seed if fake_seed is None else fake_seed)
    kw = dict(minibatches=2, minibatch=64, buffer_rows=4000, ckpt_every=3, log_every=0)
    kw.update(opts)
    o = CORE.RunOpts(seed=seed, iterations=iterations, stop_after=stop_after, num_envs=n, smoke=smoke, **kw)
    tr = CORE.Trainer(env, nets, o, out, state=env.state, probe=env.probe, gens=gens, train_cfg=cfg,
                      policy_cls=classes[0], est_cls=classes[1],
                      infos={"sh_dagger": {"test": True}}, terrain_fn=(lambda: env.terrain_levels) if terrain else None,
                      config_clip=1.0, log=lambda s: None)
    return tr, env, tr.run()


# ===================================================================================== warm start / networks
def test_warm_start_zeroes_only_w15_and_keeps_s1_estimator(CORE, nets_proto):
    n = nets_proto
    am = torch.load(CORE.SH.STAND_NET_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = n.student.state_dict()
    for k, v in am.items():
        if k == CORE.SH.WARM_ZERO_KEY:
            want = v.clone()
            want[:, 15] = 0
            assert torch.equal(sd[k], want)
            assert float(v[:, 15].norm()) > 1.0
        else:
            assert torch.equal(sd[k], v), k
    assert CORE.w15_is_zero(n.student)
    s1 = torch.load(SC.S1_CKPT, map_location="cpu", weights_only=False)
    ok, why = CORE.SN.state_dicts_equal(n.s1_estimator_state, s1["estimator_state_dict"])
    assert ok, why
    assert {name for name, p in n.student.named_parameters() if p.requires_grad} == {
        name for name, _ in n.student.named_parameters() if name.startswith("actor.actor_backbone.")}
    assert n.warm["sha256"] == CORE.SH.STAND_NET_SHA256 and n.warm["w15_norm_before"] == pytest.approx(1.7404, abs=1e-3)


def test_warm_start_refuses_a_wrong_sha(CORE, classes, cfg):
    with pytest.raises(RuntimeError, match="sha256"):
        CORE.build_networks(classes[0], classes[1], cfg, SC.S1_CKPT, "cpu", warm_sha="0" * 64)


# ===================================================================================== routing on one step
def test_one_step_routing_and_labels(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    nets = make_nets(nets_proto)
    gens = CORE.make_generators(5, "cpu")
    env = FakeEnv(CORE.STARTS, gens, bank=bank, stance=stance, n=8, seed=1)
    o = CORE.RunOpts(seed=5, iterations=30, num_envs=8, buffer_rows=100, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "r", state=env.state, probe=env.probe, gens=gens, train_cfg=cfg,
                      policy_cls=classes[0], est_cls=classes[1])
    obs, _ = env.get_observations()
    obs[:, [5, 8, 9, 10]] = torch.tensor([0.05, 0.2, -0.1, 0.22])      # live command columns on every row
    tr.meas = env.probe()
    # rows 0-5 in a stand slot at assorted heights, 6 walking, 7 restoring (left a stand at 0.85 one step ago)
    env.cmd.slot_id[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    env.state.h_cmd[:] = torch.tensor([0.80, 1.064, 1.25, 0.90, 1.10, 1.30, 1.064, 1.064], dtype=torch.float64)
    tr.ctrl.limiter.state[:] = torch.tensor([0.80, 1.064, 1.25, 0.90, 1.10, 1.30, 1.064, 0.85], dtype=torch.float64)
    tr.ctrl.prev_slot[:] = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool)
    tr.ep_kind = torch.tensor([2, 1, 0, 0, 1, 0, 0, 2])           # control, beta-R, student, student, beta-R(sig), ...
    tr.ep_sigma = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0])
    dart = torch.Generator().set_state(tr.gens["dart"].get_state())
    slot = env.cmd.stand_mask().clone()
    with torch.inference_mode():
        h_prev = tr.ctrl.h_lim.clone()
        ctrl_copy = copy.deepcopy(tr.ctrl)
        hs = ctrl_copy.step(slot, env.state.h_cmd, obs)
        tr._step(obs, slot, torch.zeros(8, dtype=torch.bool), 0)
        h_lim, eff, rest = hs.h_lim, hs.stand_eff, hs.restoring
        lab = CORE.SH.rh_label(obs, h_lim, check=True)
        x = CORE.stand_input(obs, nets, h_lim, rest)
        mu = CORE.student_mean(nets.student, x)
        a_s1 = CORE.s1_actions(obs, nets)
    noise = torch.randn(8, 18, generator=dart)
    a, st = env.seen[-1]
    assert torch.equal(st, slot)
    assert eff.tolist() == [True] * 6 + [False, True] and rest.tolist() == [False] * 7 + [True]
    assert torch.equal(h_lim, tr.ctrl.h_lim) and not torch.equal(h_prev, h_lim)
    # R at sigma 0 executes R_h's target bitwise (control row 0, beta-R row 1, restore row 7 on a control episode)
    for r in (0, 1, 7):
        assert torch.equal(a[r], lab.target[r]), r
    # student rows: q + clamp(mean), rods +-4.8
    for r in (2, 5):
        want, _ = CORE.SH.stand_action(mu[r:r + 1], h_lim[r:r + 1])
        assert torch.equal(a[r], want[0]), r
    # DART on the residual: student sigma 0.3 (row 3), beta-R sigma 0.1 (row 4)
    w3, _ = CORE.SH.stand_action(mu[3:4] + 0.3 * noise[3:4], h_lim[3:4])
    w4, _ = CORE.SH.stand_action(lab.label[4:5] + 0.1 * noise[4:5], h_lim[4:5])
    assert torch.equal(a[3], w3[0]) and torch.equal(a[4], w4[0])
    # S1 row: clamp(S1, +-1); the wrapper clipped at 4.8 (a stand row beyond +-1 survived)
    assert torch.equal(a[6], a_s1[6].clamp(-1, 1))
    assert float(env.applied[0, 6:12].min()) < -1.0 and torch.equal(env.applied, a.clamp(-4.8, 4.8))
    # the restore row's input: command columns zeroed, col 15 = its limited command
    assert torch.all(x[7, [5, 8, 9, 10]] == 0) and float(x[7, 15]) == pytest.approx((float(h_lim[7]) - 1.064) / 0.4)
    assert torch.all(x[:7, [5, 8, 9, 10]] != 0) and torch.all(obs[:, [5, 8, 9, 10]] != 0)   # only the clone, only row 7
    # every effective stand row entered the reservoir, labelled with R_h's residual
    assert tr.buf.seen == 7
    assert torch.equal(tr.buf.tgt[:7], lab.label[eff].float())
    assert torch.equal(tr.buf.expand(torch.arange(7)), tr.buf.expand_rows(tr.buf.compact(x[eff])))
    assert env.get_calls == 1


# ===================================================================================== a short run
def test_run_end_to_end(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    torch.manual_seed(77)
    np.random.seed(3)
    before = CORE.save_rng()
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "a", stop_after=4,
                              smoke=False)
    assert CORE.rng_equal(before, CORE.save_rng())                     # T5
    assert env.get_calls == 1                                          # T6
    assert res["status"] == "done" and res["iterations_run"] == 4 and res["opt_steps"] == 8
    assert [Path(p).name for p in res["checkpoints"]] == ["model_3.pt", "model_4.pt"]
    log = [json.loads(x) for x in (tmp_path / "a/train_log.jsonl").read_text().splitlines()]
    assert [r["it"] for r in log] == list(range(4))
    assert set(log[0]["restore"]) == {"student", "beta_r", "control"}
    assert set(log[0]["restore"]["control"]) == {"sigma0", "sigma_pos"}
    assert set(log[0]["restore"]["control"]["sigma0"]) == {"walk", "table", "ground", "random"}
    recs = CORE.SPEC.read_stands(tmp_path / "a/stands.jsonl")
    assert recs and all(set(CORE.SPEC.RECORD_KEYS) <= set(r) for r in recs)
    assert {r["start_kind"] for r in recs} <= set(CORE.SPEC.START_KINDS)
    eps = CORE.SPEC.read_jsonl(tmp_path / "a/episodes.jsonl")
    assert all(set(CORE.SPEC.EPISODE_KEYS) <= set(e) for e in eps)
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "done"
    # every executed action went through the 4.8 wrapper clip, S1 rows within +-1 by the router
    assert all(a.shape == (48, 18) for a, _ in env.seen) and len(env.seen) == 4 * 24


def test_same_seed_replays_bitwise(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    _, _, r1 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "x", stop_after=3, smoke=False)
    _, _, r2 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "y", stop_after=3, smoke=False)
    for f in ("stands.jsonl", "episodes.jsonl"):
        assert (tmp_path / "x" / f).read_text() == (tmp_path / "y" / f).read_text()
    a = torch.load(r1["final_checkpoint"], weights_only=False)
    b = torch.load(r2["final_checkpoint"], weights_only=False)
    ok, why = CORE.SN.state_dicts_equal(a["model_state_dict"], b["model_state_dict"])
    assert ok, why


def test_start_modes_forced_stands_and_limiter_init(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    """Long enough on the fake (dt 0.1) that table / ground / random-height episodes start, stand up, hand back and
    park their tables: the env-side state and the trainer agree per reset row."""
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "s", stop_after=12,
                              iterations=36, smoke=False, n=64)
    st = env.state
    c = st.counts()
    total = sum(c[f"episodes_{k}"] for k in CORE.SPEC.START_KINDS)
    assert total >= 64 and all(c[f"episodes_{k}"] > 0 for k in CORE.SPEC.START_KINDS)
    # one forced first stand per non-walk episode, none on walk episodes
    assert c["forced_walk"] == 0
    for k in ("table", "ground", "random"):
        assert c[f"forced_{k}"] == c[f"episodes_{k}"]
    a = tr.smoke_measures()
    assert a["init_violations"] == 0 and a["init_n"][0] > 0 and sum(a["init_n"][1:]) > 0
    assert a["hlim_step_max"] <= 0.08 * 0.1 + 1e-12
    recs = list(tr.tracker.done)
    first = [r for r in recs if r["first_stand"] and r["entry_kind"] != "run_start"]
    assert first and all(r["start_kind"] != "walk" and r["entry_ep"] == 0 for r in first)
    for r in first:
        assert CORE.first_stand_ok(r)[0], r
        if r["start_kind"] == "table":
            assert r["h_cmd0"] >= r["table_top"] + 0.264 - 1e-9
    assert c["parks"] > 0 and c["parks_before_change"] == 0 and c["parks_not_clear"] == 0
    assert c["push_suppressed"] > 0 and c["push_restore_mismatch"] == 0
    assert c["hip_term_disarmed"] == 0


# ===================================================================================== checkpoints
def test_checkpoint_round_trip_and_infos(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "c", stop_after=3,
                              smoke=False)
    path = res["final_checkpoint"]
    ck = torch.load(path, weights_only=False)
    assert set(CORE.REQUIRED_CKPT_KEYS) <= set(ck) and ck["iter"] == 3
    assert all(v.device.type == "cpu" for v in ck["model_state_dict"].values())
    sn, sh = ck["infos"]["sn_dagger"], ck["infos"]["sh_dagger"]
    assert type(sn["stand_mask"]) is int and sn["stand_mask"] == 1 and sn["s1_sha256"] == SC_S1_SHA()
    hc = sh["height_command"]
    assert hc["column"] == 15 and hc["h_range"] == [0.719, 1.346] and hc["h_settled"] == 1.064
    assert hc["encoding"] == "(h_cmd_limited - 1.064) / 0.4" and hc["restore"]["timeout_steps"] == 54
    assert sh["warm_start"]["sha256"] == CORE.SH.STAND_NET_SHA256 and sh["test"] is True
    assert CORE.checkpoint_reasons(path, tr.nets, classes[0], classes[1], cfg) == []
    am = torch.load(CORE.SH.STAND_NET_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    moved = {k for k in ck["model_state_dict"] if not torch.equal(ck["model_state_dict"][k], am[k].cpu())}
    assert moved and all(k.startswith("actor.actor_backbone.") for k in moved)
    fresh = classes[0](W, 18, **{k: v for k, v in copy.deepcopy(cfg["policy"]).items() if k != "class_name"})
    ppo_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    ppo_opt.load_state_dict(ck["optimizer_state_dict"])
    # a checkpoint whose infos lost the height channel / carries a bool mask is refused
    bad = copy.deepcopy(ck)
    bad["infos"]["sn_dagger"]["stand_mask"] = True
    del bad["infos"]["sh_dagger"]["height_command"]
    torch.save(bad, tmp_path / "bad.pt")
    r = CORE.checkpoint_reasons(tmp_path / "bad.pt", tr.nets, classes[0], classes[1], cfg)
    assert any("stand_mask" in x for x in r) and any("height_command" in x for x in r)


def SC_S1_SHA():
    return "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"


# ===================================================================================== the step-5 smoke
@pytest.fixture(scope="module")
def smoke_run(CORE, classes, cfg, nets_proto, bank, stance, tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "run"
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, out, stop_after=14, iterations=60,
                              smoke=True, n=96)
    checks = CORE.smoke_checks(tr, env_parity=[], ckpt=res["final_checkpoint"], env_clip=env.clip_actions)
    return tr, env, res, checks, out


#: dose / plumbing checks that must pass on the fake (the rest depend on the untrained policy's behaviour)
MUST_PASS = ("start_mode_shares", "retracted_pose_shares", "table_tops", "ground_spawns", "h_cmd_distribution",
             "limiter", "residual_labels_within_1", "restores_complete", "table_first_stand",
             "forced_first_stand_ground_random",
             "table_parked_at_slot_change", "pushes_off_until_up", "ground_episodes_hip_rearm",
             "restore_rows_stand_net_inputs", "restore_rows_in_reservoir_with_rh_labels", "s1_rows_pm1",
             "stand_hip_targets_below_minus1", "harness_clip_4p8", "forward_parity", "warm_start_zeroed",
             "handovers", "beta_schedule", "obs_parity_env", "obs_parity_cols", "s1_deterministic",
             "buffer_projection", "ckpt_loads", "terrain_level_recorded", "dart_sigma_realized")


def test_smoke_checks_complete_and_doses_pass(CORE, smoke_run):
    tr, env, res, checks, out = smoke_run
    v = CORE.SPEC.smoke_verdict(checks)
    assert v["problems"] == [], v
    by = {c["name"]: c for c in checks}
    assert set(by) == {n for n, _ in CORE.SPEC.SMOKE_DAGGER}
    for name in MUST_PASS:
        assert by[name]["pass"] is True, (name, by[name]["value"])
    fp = by["forward_parity"]["value"]
    assert fp["equal"] is True and fp["stand_rows"] > 0
    assert (out / "smoke_capture.pt").is_file()
    json.dumps(checks, default=str)                                   # the smoke.json is writable


def test_premise_read_on_the_run_files(CORE, smoke_run):
    tr, env, res, checks, out = smoke_run
    recs = CORE.SPEC.read_stands(out / "stands.jsonl")
    log = CORE.SPEC.read_jsonl(out / "train_log.jsonl")
    pr = CORE.SPEC.premise_read(recs, log)
    assert set(pr["by_kind"]) == set(CORE.SPEC.START_KINDS)
    pooled = pr["restore_pooled_all_sigma"]
    assert sum(pooled[ex]["completed"] for ex in pooled) > 0
    # the per-iteration counts sum to the run's accumulator (no event lost between the log and the smoke)
    acc = tr.smoke_measures()["restore"]
    tot = CORE.SPEC.restore_totals(log)
    for i, ex in enumerate(CORE.SPEC.EXECUTORS):
        for g, sg in enumerate(CORE.SPEC.RESTORE_SIGMA):
            for b, k in enumerate(CORE.SPEC.START_KINDS):
                for j, e in enumerate(CORE.SPEC.RESTORE_EVENTS):
                    assert tot[ex][sg][k][e] == acc[i][g][b][j]
    assert pr["stop"] in (True, False, None) and len(pr["checks"]) == 6


def test_fake_env_zeroes_the_applied_action_on_reset_rows(CORE, bank, stance):
    """The fake mirrors IsaacLab's ActionManager.reset (``_action[env_ids] = 0``): after an env.step whose rows reset,
    those rows' applied action is 0 while the routed action was not -- the unmasked clip check would count them."""
    env = FakeEnv(CORE.STARTS, CORE.make_generators(3, "cpu"), bank=bank, stance=stance, n=16, seed=3, max_ep=5)
    a = torch.full((16, 18), 0.5)
    seen_reset = False
    for _ in range(7):
        _, _, dones, _ = env.step(a)
        d = dones.bool()
        if d.any():
            seen_reset = True
            assert (env.applied[d] == 0).all()
            assert torch.equal(env.applied[~d], a.clamp(-4.8, 4.8)[~d])
            assert torch.equal((env.applied != a.clamp(-4.8, 4.8)).any(dim=1), d)   # the old rule's miscount
    assert seen_reset


def test_train_time_stood_reads_the_root_rise_on_95pct_of_the_window(CORE):
    """Review fix: the train-time STOOD (step-6 premise) reads the ROOT's rise (the trainer passes root_z - spawn
    root z) on >= 95 % of the first draw's last 3 s, like K3 -- 7 un-risen window rows of 150 still STOOD, 8 do not."""
    SP = CORE.SPEC
    trk = CORE.StandTracker(3, "cpu", dt=0.02, max_ep=100000, stood_window_steps=150)
    n, L = 3, 200
    obs = torch.zeros(n, 89)
    low = torch.tensor([7, 8, 0])
    for t in range(L):
        inwin = t >= L - 150
        k = t - (L - 150)
        rise = torch.full((n,), 0.25, dtype=torch.float64)
        for e in range(n):
            if inwin and k % 2 == 0 and k // 2 < int(low[e]):
                rise[e] = 0.05
        trk.pre(torch.ones(n, dtype=torch.bool), obs, torch.full((n,), (L - t) * 0.02),
                torch.zeros(n, dtype=torch.long), torch.full((n,), t, dtype=torch.long), 0,
                torch.zeros(n, dtype=torch.long), torch.zeros(n),
                h_meas=torch.full((n,), 1.0, dtype=torch.float64), feet=torch.full((n,), 6, dtype=torch.long),
                rise=rise, h_cmd=torch.full((n,), 1.0, dtype=torch.float64),
                h_init=torch.full((n,), 0.9, dtype=torch.float64), start_kind=torch.full((n,), SP.SK_TABLE),
                top=torch.full((n,), 0.6, dtype=torch.float64), pose=torch.zeros(n, dtype=torch.long))
    z = torch.zeros(n, dtype=torch.bool)
    trk.post(z, z, z, 0)                                              # every env exits its stand: records pend
    recs = [trk.pending[e] for e in range(n)]
    assert [r["win_n0"] for r in recs] == [150] * 3
    assert [r["win_rise0"] for r in recs] == [143, 142, 150]
    assert [r["stood0"] for r in recs] == [True, False, True]
    assert recs[0]["rise_min_win0"] == pytest.approx(0.05) and set(SP.RECORD_KEYS) <= set(recs[0])


def test_restores_complete_gates_the_rh_slices_and_reports_the_student(CORE, smoke_run):
    """Review fix: the smoke's restore check gates the mechanics on the R_h-executed slices (control sigma-0 restores
    complete; beta-R sigma-0 within the control line) and only REPORTS the student's timeout share (the warm-start
    student is a policy outcome; step 6's premise gates it). Student timeouts far over the line leave it passing;
    beta-R timeouts or a control slice that never completes fail it."""
    tr, env, res, checks, out = smoke_run
    kw = dict(env_parity=[], ckpt=res["final_checkpoint"], env_clip=env.clip_actions)
    S = CORE.SPEC
    i_c, i_t = S.RESTORE_EVENTS.index("completed"), S.RESTORE_EVENTS.index("timed_out")
    a = tr.acc
    keep = a["restore"]
    try:
        r = keep.clone()
        r[S.EX_STUDENT, 0, :, i_t] += 50.0                               # the student times out a lot (sigma 0)
        a["restore"] = r
        c = {x["name"]: x for x in CORE.smoke_checks(tr, **kw)}["restores_complete"]
        assert c["pass"] is True and c["value"]["student_reported_not_gated"]["over_line"] is True
        r = keep.clone()
        r[S.EX_BETA_R, 0, :, i_t] += 50.0                                # the SAME executor as control times out
        a["restore"] = r
        assert {x["name"]: x for x in CORE.smoke_checks(tr, **kw)}["restores_complete"]["pass"] is False
        r = keep.clone()
        r[S.EX_CONTROL, 0, :, i_c] = 0.0                                 # the control slice never completed
        a["restore"] = r
        assert {x["name"]: x for x in CORE.smoke_checks(tr, **kw)}["restores_complete"]["pass"] is False
        r = keep.clone()
        r[S.EX_CONTROL, 0, :, i_t] += 50.0                               # R_h itself times out (re-review fix):
        r[S.EX_BETA_R, 0, :, i_t] += 50.0                                # the relative line alone would pass
        a["restore"] = r
        c = {x["name"]: x for x in CORE.smoke_checks(tr, **kw)}["restores_complete"]
        assert c["value"]["over_line"]["beta_r"] is not True and c["pass"] is False
    finally:
        a["restore"] = keep


def test_estimate_range_on_off_height_stand_rows_is_reported(CORE, smoke_run):
    """Review item (accepted, reported): the frozen estimator's input carries q + residual beyond +-1 in its
    last-action block on stand rows at h != 1.064; the smoke reports the per-column |estimate| range there vs on the
    S1 rows (not gated)."""
    tr, env, res, checks, out = smoke_run
    by = {c["name"]: c for c in checks}
    er = by["restore_rows_stand_net_inputs"]["value"]["estimate_range_reported"]
    n = tr.nets.p1 - tr.nets.p0
    assert er["rows_off_height"] > 0 and len(er["absmax_off_height"]) == len(er["absmax_s1_rows"]) == n
    assert max(er["absmax_s1_rows"]) > 0 and er["max_ratio"] is not None
    assert by["restore_rows_stand_net_inputs"]["pass"] is True


def test_applied_action_checks_skip_rows_that_reset(CORE, smoke_run):
    """Blocking-fix pin: rows that reset on a step carry a zeroed applied action; the clip / S1 / stand-hip
    accumulators read only the rows that did not reset, so the smoke's clip check passes WITH resets in the run."""
    tr, env, res, checks, out = smoke_run
    a = tr.smoke_measures()
    assert a["applied_rows_reset_skipped"] > 0                       # resets happened while the checks ran
    assert a["applied_rows_live"] > 0 and a["clip_mismatch"] == 0
    by = {c["name"]: c for c in checks}
    assert by["harness_clip_4p8"]["pass"] is True and by["s1_rows_pm1"]["pass"] is True
    assert by["harness_clip_4p8"]["value"]["applied_rows_reset_skipped"] == a["applied_rows_reset_skipped"]


def test_smoke_guards_can_fail(CORE, smoke_run):
    """The dose checks are not tautologies: corrupt what each one reads and it fails."""
    tr, env, res, checks, out = smoke_run
    kw = dict(env_parity=[], ckpt=res["final_checkpoint"], env_clip=env.clip_actions)

    def run_checks():
        return {c["name"]: c for c in CORE.smoke_checks(tr, **kw)}

    a = tr.acc
    saved = {k: v.clone() for k, v in a.items()}
    try:
        f64 = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731  (fresh tensors: the run's are inference
        a["restore_zero_viol"] = a["restore_zero_viol"] + 1    # tensors, not writable in place out here)
        a["col15_viol"] = a["col15_viol"] + 1
        a["s1_applied_absmax"] = f64(1.5)
        a["clip_mismatch"] = a["clip_mismatch"] + 1
        a["init_violations"] = a["init_violations"] + 1
        a["hlim_step_max"] = f64(0.05)
        a["label_absmax"] = f64(1.1)
        by = run_checks()
        for name in ("restore_rows_stand_net_inputs", "s1_rows_pm1", "harness_clip_4p8", "limiter",
                     "residual_labels_within_1"):
            assert by[name]["pass"] is False, name
    finally:
        for k, v in saved.items():
            a[k] = v
    rc = tr.restore_capture
    keep = rc["label"]
    bad = keep.clone()
    bad[0, 0] += 0.01
    rc["label"] = bad
    assert run_checks()["restore_rows_in_reservoir_with_rh_labels"]["pass"] is False
    rc["label"] = keep
    c = env.state.c
    old = {k: c[k].clone() for k in ("hip_term_disarmed", "parks_before_change", "push_restore_mismatch",
                                     "hcmd_table_violations")}
    try:
        for k in old:
            c[k] = c[k] + 1
        by = run_checks()
        for name in ("ground_episodes_hip_rearm", "table_parked_at_slot_change", "pushes_off_until_up",
                     "h_cmd_distribution"):
            assert by[name]["pass"] is False, name
    finally:
        for k, v in old.items():
            c[k] = v
    assert all(run_checks()[n]["pass"] is True for n in MUST_PASS if n != "forward_parity")


def test_a_router_without_the_s1_clamp_is_caught(CORE, classes, cfg, nets_proto, bank, stance, tmp_path,
                                                   monkeypatch):
    real = CORE.SH.route

    def no_clamp(stand_eff, a_stand, a_s1):
        return torch.where(stand_eff[:, None], a_stand.to(a_s1.dtype), a_s1 * 3.0)

    monkeypatch.setattr(CORE.SH, "route", no_clamp)
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "g", stop_after=3, smoke=True)
    monkeypatch.setattr(CORE.SH, "route", real)
    by = {c["name"]: c for c in CORE.smoke_checks(tr, env_parity=[], ckpt=res["final_checkpoint"],
                                                 env_clip=env.clip_actions)}
    assert by["s1_rows_pm1"]["pass"] is False and by["s1_rows_pm1"]["value"]["applied_absmax_non_stand_rows"] > 1.0


def test_an_input_that_ignores_restore_rows_is_caught(CORE, classes, cfg, nets_proto, bank, stance, tmp_path,
                                                       monkeypatch):
    real = CORE.SH.stand_net_input

    def no_zeroing(obs, estimator, *, p0, p1, num_prop, mask, h_cmd_limited, restoring=None):
        return real(obs, estimator, p0=p0, p1=p1, num_prop=num_prop, mask=mask, h_cmd_limited=h_cmd_limited,
                    restoring=None)

    monkeypatch.setattr(CORE.SH, "stand_net_input", no_zeroing)
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "h", stop_after=6,
                              smoke=False, n=64)
    monkeypatch.setattr(CORE.SH, "stand_net_input", real)
    a = tr.smoke_measures()
    assert a["restore_rows"] > 0 and a["restore_zero_viol"] > 0


def test_forward_parity_trainer_vs_launcher_with_restore_rows(CORE, nets_proto):
    """Bitwise on CPU on a batch with stand, restore and walking rows at assorted limited heights."""
    n = make_nets(nets_proto)
    obs = SC.obs_batch(24, seed=11)
    eff = torch.tensor([True, False] * 12)
    rest = torch.tensor([True, False, False, False] * 6) & eff
    h = torch.linspace(0.719, 1.346, 24, dtype=torch.float64)
    with torch.inference_mode():
        a = CORE.trainer_actions(obs, eff, rest, h, n)
        b = CORE.launcher_actions(obs, eff, rest, h, n.s1, n.est, n.student, n.est, p0=n.p0, p1=n.p1,
                                  num_prop=n.num_prop)
        a_s1 = CORE.s1_actions(obs, n)
    assert torch.equal(a, b)
    assert torch.equal(a[~eff], a_s1[~eff].clamp(-1, 1))
    assert float(a[eff][:, 6:12].min()) < -1.0


def test_compaction_guard_fails_closed(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    nets = make_nets(nets_proto)
    gens = CORE.make_generators(6, "cpu")
    env = FakeEnv(CORE.STARTS, gens, bank=bank, stance=stance, n=8, seed=6)
    o = CORE.RunOpts(seed=6, iterations=2, num_envs=8, buffer_rows=100, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "g", state=env.state, probe=env.probe, gens=gens, train_cfg=cfg,
                      policy_cls=classes[0], est_cls=classes[1])
    obs = SC.obs_batch(8, seed=3)
    tr.compaction_guard(obs)
    tr.buf.hist_start = 400
    tr.buf.n_keep = tr.buf.p1 + (W - 400)
    with pytest.raises(RuntimeError, match="compaction"):
        tr.compaction_guard(obs)


def test_trainer_refuses_a_state_on_other_streams(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    nets = make_nets(nets_proto)
    env = FakeEnv(CORE.STARTS, CORE.make_generators(1, "cpu"), bank=bank, stance=stance, n=8, seed=1)
    o = CORE.RunOpts(seed=1, iterations=2, num_envs=8, buffer_rows=100, log_every=0)
    with pytest.raises(RuntimeError, match="generators"):
        CORE.Trainer(env, nets, o, tmp_path / "z", state=env.state, probe=env.probe,
                     gens=CORE.make_generators(1, "cpu"), train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1])


def test_runopts_one_arm_one_mode(CORE):
    with pytest.raises(ValueError):
        CORE.RunOpts(mode="recover_probe", seed=1, iterations=5)
    with pytest.raises(ValueError):
        CORE.RunOpts(arm="full", seed=1, iterations=5)
    with pytest.raises(ValueError):
        CORE.RunOpts(seed=1, iterations=5, stop_after=6)
    assert CORE.RunOpts(seed=1, iterations=5).stand_mask == 1
