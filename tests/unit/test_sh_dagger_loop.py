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
  a wrong label);
* the residual deadband (CHARTER 2026-10-14): at 0 the step is the composition without the option (bitwise) and a run
  replays the default run; at 0.25 only student-executed rows (restore rows included) are deadbanded, labels and the R
  executors untouched; records (status, checkpoint infos), forward parity, the realised smoke check (and a trainer that
  ignores its deadband is caught);
* the trim branch (CHARTER 2026-10-14, "retrain the pilot with a trim weight"): off = today's weights (a verbatim copy
  of the pre-option estimator) on a synthetic batch and a run that replays the default run bitwise; on = HOLD legs with
  a non-zero R_h cam label split from HOLD into a 6th category, the inverse-frequency weights recomputed with the cap, a
  HOLD leg with a zero cam label stays HOLD, nothing else (labels, onsets, actions) changes; records; the
  ``trim_branch_weights`` smoke check passes / fails on hand-built reservoirs, reads "not applied" off, and runs beside
  the deadband check; the trim DIAGNOSTIC's per-iteration histograms on a hand-set step.
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
        foot = torch.arange(6)[None, :] < feet[:, None]                  # per-foot flags: the first `feet` legs
        return {"root_z": z, "lowest_z": low, "h_meas": z.clone(), "feet_loaded": feet.long(), "foot_loaded": foot,
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
    assert total >= 64 and all(c[f"episodes_{k}"] > 0 for k in CORE.SPEC.DRAWN_START_KINDS)
    assert c["episodes_ground"] == 0 and c["poses_table"][1] == 0          # share 0 since the step-4 decision
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
    assert set(pr["by_kind"]) == set(CORE.SPEC.DRAWN_START_KINDS) and pr["absent_kinds"]["ground"]["records"] == 0
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
    assert pr["stop"] in (True, False, None) and len(pr["checks"]) == 5         # no ground start type (step 4)


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
    # a share-0 kind / pose that shows up (one ground episode, one rod_min pose) fails its ABSENT assertion
    old = {k: c[k].clone() for k in ("episodes_ground", "forced_ground", "poses_table")}
    try:
        c["episodes_ground"] = c["episodes_ground"] + 1
        c["forced_ground"] = c["forced_ground"] + 1
        pt = c["poses_table"].clone()
        pt[1] += 1
        c["poses_table"] = pt
        by = run_checks()
        for name in ("start_mode_shares", "retracted_pose_shares", "forced_first_stand_ground_random",
                     "ground_episodes_hip_rearm"):
            assert by[name]["pass"] is False, name
        assert by["start_mode_shares"]["value"]["ground"]["asserted_absent"] is True
    finally:
        for k, v in old.items():
            c[k] = v
    saved = {k: v.clone() for k, v in a.items()}
    try:
        sn = a["spawn_h_n"].clone()
        sn[CORE.SPEC.SK_GROUND] += 1
        a["spawn_h_n"] = sn
        assert run_checks()["ground_spawns"]["pass"] is False
    finally:
        for k, v in saved.items():
            a[k] = v
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


# ===================================================================================== the residual deadband
# CHARTER section 3, 2026-10-14: "train it in" executes the deadbanded STUDENT action (labels unchanged = R_h's
# residuals, the R_h control and beta-R executors untouched); --deadband 0 is the trainer without the option.
def _today_stand_action(SH, a_net, h):
    """The stand composition as it stood before the deadband option (verbatim, sh_core at e629b579)."""
    n = int(a_net.shape[0])
    a = torch.clamp(a_net, -SH.RESIDUAL_CLIP, SH.RESIDUAL_CLIP)
    q = SH.q18(SH._vec(h, n, a.device, name="h_cmd_limited"), dtype=a.dtype, device=a.device)
    s = torch.where(q != 0, a + q, a)
    return torch.cat([s[:, SH.CAM_ACT].clamp(-SH.CAM_CLIP, SH.CAM_CLIP),
                      s[:, SH.HIP_ACT.start:].clamp(-SH.RAW_CLIP, SH.RAW_CLIP)], dim=1)


def _one_step(CORE, classes, cfg, nets_proto, bank, stance, out, pre=None, **opts):
    """One trainer step on 8 hand-set rows: 0 control sigma 0, 1 beta-R sigma 0, 2 student, 3 student sigma 0.3,
    4 beta-R sigma 0.1, 5 student, 6 walking (S1), 7 a STUDENT restore row (left a 0.85 stand one step ago).
    ``pre(env)`` (optional) edits the fake's state before the observation is read."""
    nets = make_nets(nets_proto)
    gens = CORE.make_generators(5, "cpu")
    env = FakeEnv(CORE.STARTS, gens, bank=bank, stance=stance, n=8, seed=1)
    o = CORE.RunOpts(seed=5, iterations=30, num_envs=8, buffer_rows=100, log_every=0, **opts)
    tr = CORE.Trainer(env, nets, o, out, state=env.state, probe=env.probe, gens=gens, train_cfg=cfg,
                      policy_cls=classes[0], est_cls=classes[1])
    if pre is not None:
        pre(env)
    obs, _ = env.get_observations()
    tr.meas = env.probe()
    env.cmd.slot_id[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    env.state.h_cmd[:] = torch.tensor([0.80, 1.064, 1.25, 0.90, 1.10, 1.30, 1.064, 1.064], dtype=torch.float64)
    tr.ctrl.limiter.state[:] = torch.tensor([0.80, 1.064, 1.25, 0.90, 1.10, 1.30, 1.064, 0.85], dtype=torch.float64)
    tr.ctrl.prev_slot[:] = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool)
    tr.ep_kind = torch.tensor([2, 1, 0, 0, 1, 0, 0, 0])
    tr.ep_sigma = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0])
    dart = torch.Generator().set_state(tr.gens["dart"].get_state())
    slot = env.cmd.stand_mask().clone()
    with torch.inference_mode():
        hs = copy.deepcopy(tr.ctrl).step(slot, env.state.h_cmd, obs)
        tr._step(obs, slot, torch.zeros(8, dtype=torch.bool), 0)
        lab = CORE.SH.rh_label(obs, hs.h_lim, check=True)
        mu = CORE.student_mean(nets.student, CORE.stand_input(obs, nets, hs.h_lim, hs.restoring))
        a_s1 = CORE.s1_actions(obs, nets)
    noise = torch.randn(8, 18, generator=dart)
    return tr, env, hs, lab, mu, a_s1, noise


def test_deadband_zero_trainer_step_is_the_path_without_the_option(CORE, classes, cfg, nets_proto, bank, stance,
                                                                    tmp_path):
    SH = CORE.SH
    for kw in ({}, {"deadband": 0.0}):
        tr, env, hs, lab, mu, a_s1, noise = _one_step(CORE, classes, cfg, nets_proto, bank, stance,
                                                      tmp_path / f"z{len(kw)}", **kw)
        a, _ = env.seen[-1]
        h = hs.h_lim
        assert tr.o.deadband == 0.0 and hs.restoring.tolist() == [False] * 7 + [True]
        for r in (2, 5, 7):
            assert torch.equal(a[r], _today_stand_action(SH, mu[r:r + 1], h[r:r + 1])[0]), r
        assert torch.equal(a[3], _today_stand_action(SH, mu[3:4] + 0.3 * noise[3:4], h[3:4])[0])
        assert torch.equal(a[4], _today_stand_action(SH, lab.label[4:5] + 0.1 * noise[4:5], h[4:5])[0])
        for r in (0, 1):
            assert torch.equal(a[r], lab.target[r]), r
        assert torch.equal(a[6], a_s1[6].clamp(-1, 1))
        assert tr.acc["db_rows"] == 0 and tr.acc["db_entries"] == 0          # the realised check never runs at 0


def test_deadband_trainer_step_student_rows_only_labels_unchanged(CORE, classes, cfg, nets_proto, bank, stance,
                                                                   tmp_path):
    SH = CORE.SH
    d = SH.RESIDUAL_DEADBAND_PROBE
    tr, env, hs, lab, mu, a_s1, noise = _one_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "d",
                                                  deadband=d)
    a, _ = env.seen[-1]
    h = hs.h_lim
    assert tr.o.deadband == d and tr.o.to_dict()["deadband"] == d
    # student rows (2, 5, the restore row 7, and the DART row 3): q + deadband(clamp(residual))
    for r in (2, 5, 7):
        want, _ = SH.stand_action(mu[r:r + 1], h[r:r + 1], deadband=d)
        assert torch.equal(a[r], want[0]), r
    w3, _ = SH.stand_action(mu[3:4] + 0.3 * noise[3:4], h[3:4], deadband=d)
    assert torch.equal(a[3], w3[0])
    small = (mu[[2, 3, 5, 7]][:, 6:].clamp(-1, 1).abs() < d)
    assert small.any()                                                    # non-vacuous: the deadband acted
    assert not torch.equal(a[[2, 5, 7]], _today_stand_action(SH, mu[[2, 5, 7]], h[[2, 5, 7]]))
    # R executors untouched: control / beta-R at sigma 0 = R_h's target, beta-R with sigma = q + clamp(label + noise)
    for r in (0, 1):
        assert torch.equal(a[r], lab.target[r]), r
    assert torch.equal(a[4], _today_stand_action(SH, lab.label[4:5] + 0.1 * noise[4:5], h[4:5])[0])
    assert torch.equal(a[6], a_s1[6].clamp(-1, 1))                         # S1 row
    # labels unchanged: every effective stand row enters the reservoir with R_h's residual
    eff = hs.stand_eff
    assert torch.equal(tr.buf.tgt[:int(eff.sum())], lab.label[eff].float())
    # the realised student residual (applied - q) on hip / knee: 0 or >= d
    assert tr.acc["db_rows"] > 0 and tr.acc["db_viol_rows"] == 0 and tr.acc["db_zero_entries"] > 0


def test_deadband_run_records_parity_and_the_realised_smoke_check(CORE, classes, cfg, nets_proto, bank, stance,
                                                                   tmp_path):
    d = CORE.SH.RESIDUAL_DEADBAND_PROBE
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "db", stop_after=4,
                              smoke=True, n=64, deadband=d)
    assert res[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert json.loads((tmp_path / "db/status.json").read_text())[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    ck = torch.load(res["final_checkpoint"], weights_only=False)
    assert ck["infos"]["sh_dagger"][CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert CORE.SH.recorded_deadband(ck["infos"]["sh_dagger"]) == d
    by = {c["name"]: c for c in CORE.smoke_checks(tr, env_parity=[], ckpt=res["final_checkpoint"],
                                                 env_clip=env.clip_actions)}
    assert set(by) == {n for n, _ in CORE.SPEC.SMOKE_DAGGER}
    c = by["student_deadband_realised"]
    assert c["pass"] is True, c
    assert c["value"]["student_rows_live"] > 0 and c["value"]["rows_with_residual_inside_0_d"] == 0
    assert c["value"]["entries_zero"] > 0 and c["value"]["entries_kept_ge_d"] > 0
    fp = by["forward_parity"]["value"]
    assert by["forward_parity"]["pass"] is True and fp["equal"] is True and fp[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert torch.load(fp["capture"], weights_only=False)[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert by["harness_clip_4p8"]["value"]["record"][CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    # a plain run reports the check as not applied (and passes): the default smoke list is unchanged in meaning
    tr0, env0, res0 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "plain", stop_after=3,
                                 smoke=True)
    by0 = {c["name"]: c for c in CORE.smoke_checks(tr0, env_parity=[], ckpt=res0["final_checkpoint"],
                                                  env_clip=env0.clip_actions)}
    assert by0["student_deadband_realised"]["pass"] is True and res0[CORE.SH.RESIDUAL_DEADBAND_KEY] == 0.0
    assert torch.load(res0["final_checkpoint"], weights_only=False)["infos"]["sh_dagger"][
        CORE.SH.RESIDUAL_DEADBAND_KEY] == 0.0


def test_deadband_zero_run_replays_the_default_run_bitwise(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    # iterations 3: beta reaches 0 at it 1, so the episodes drawn from then on are student-executed
    kw = dict(stop_after=3, iterations=3, smoke=False, n=64)
    _, _, r1 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "x", **kw)
    _, _, r2 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "y", deadband=0.0, **kw)
    _, _, r3 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "z", deadband=0.25, **kw)
    for f in ("stands.jsonl", "episodes.jsonl"):
        assert (tmp_path / "x" / f).read_text() == (tmp_path / "y" / f).read_text()
    a, b, c = (torch.load(r["final_checkpoint"], weights_only=False)["model_state_dict"] for r in (r1, r2, r3))
    assert CORE.SN.state_dicts_equal(a, b)[0]
    assert not CORE.SN.state_dicts_equal(a, c)[0]                         # the deadband changes what is visited


def test_a_trainer_that_ignores_its_deadband_is_caught(CORE, classes, cfg, nets_proto, bank, stance, tmp_path,
                                                       monkeypatch):
    real = CORE.SH.stand_action
    monkeypatch.setattr(CORE.SH, "stand_action", lambda a_net, h, *, deadband=0.0: real(a_net, h))
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "ign", stop_after=3,
                              iterations=3, smoke=True, n=64, deadband=0.25)
    monkeypatch.setattr(CORE.SH, "stand_action", real)
    by = {c["name"]: c for c in CORE.smoke_checks(tr, env_parity=[], ckpt=res["final_checkpoint"],
                                                 env_clip=env.clip_actions)}
    assert by["student_deadband_realised"]["pass"] is False
    assert by["student_deadband_realised"]["value"]["rows_with_residual_inside_0_d"] > 0


def test_forward_parity_with_the_deadband(CORE, nets_proto):
    n = make_nets(nets_proto)
    obs = SC.obs_batch(24, seed=12)
    eff = torch.tensor([True, False] * 12)
    rest = torch.tensor([True, False, False, False] * 6) & eff
    h = torch.linspace(0.719, 1.346, 24, dtype=torch.float64)
    with torch.inference_mode():
        a = CORE.trainer_actions(obs, eff, rest, h, n, deadband=0.25)
        b = CORE.launcher_actions(obs, eff, rest, h, n.s1, n.est, n.student, n.est, p0=n.p0, p1=n.p1,
                                  num_prop=n.num_prop, deadband=0.25)
        a0 = CORE.trainer_actions(obs, eff, rest, h, n)
        b0 = CORE.launcher_actions(obs, eff, rest, h, n.s1, n.est, n.student, n.est, p0=n.p0, p1=n.p1,
                                   num_prop=n.num_prop, deadband=0.0)
        a_s1 = CORE.s1_actions(obs, n)
    assert torch.equal(a, b) and torch.equal(a0, b0) and not torch.equal(a, a0)
    assert torch.equal(a[~eff], a_s1[~eff].clamp(-1, 1)) and torch.equal(a[~eff], a0[~eff])
    with pytest.raises(ValueError):
        CORE.RunOpts(seed=1, iterations=5, deadband=1.0)
    assert CORE.RunOpts(seed=1, iterations=5).deadband == 0.0


# ===================================================================================== the trim branch
# CHARTER section 3, 2026-10-14 "retrain the pilot with a trim weight + the deadband in the loop": a stand-row leg whose
# R_h branch is HOLD and whose R_h cam label is non-zero is its own category ("trim") in the per-leg inverse-branch-
# frequency weights (cap W_MAX unchanged); off by default (--trim-branch), the run without it bitwise.
def _today_branch_weight_table(branch, w_max=20.0):
    """The per-leg weight estimator as it stood before the trim option (verbatim, sh_dagger_core at 7cbdd37e)."""
    b = branch.long()
    n = int(b.shape[0])
    counts = torch.stack([torch.bincount(b[:, leg], minlength=5)[:5] for leg in range(6)])
    counts = counts.to(torch.float64)
    if n == 0:
        return torch.zeros(6, 5, dtype=torch.float32, device=branch.device)
    f = counts / n
    present = counts > 0
    k = present.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float64)
    w = torch.where(present, 1.0 / (k * f.clamp_min(1e-300)), torch.zeros_like(f))
    return w.clamp(max=float(w_max)).to(torch.float32)


def _synthetic_batch(n=4000, seed=3):
    g = torch.Generator().manual_seed(seed)
    p = torch.tensor([0.80, 0.07, 0.02, 0.05, 0.06])
    branch = torch.multinomial(p, n * 6, replacement=True, generator=g).reshape(n, 6)
    branch[:, 5] = torch.where(branch[:, 5] == 2, torch.zeros_like(branch[:, 5]), branch[:, 5])   # leg 5: no LIFT
    label = 0.05 * torch.randn(n, 18, generator=g)
    label[: n // 10, :6] = 0.0                                                       # some zero cam labels
    return branch, label


def test_trim_off_weights_are_todays_on_a_synthetic_batch(CORE):
    branch, label = _synthetic_batch()
    assert CORE.weight_branch(branch, label, None) is branch                       # off: R_h's code, the same tensor
    want = _today_branch_weight_table(branch)
    for got in (CORE.branch_weight_table(branch), CORE.branch_weight_table(branch, 20.0),
                CORE.branch_weight_table(branch, CORE.SPEC.W_MAX, CORE.N_BRANCH)):
        assert got.shape == (6, 5) and torch.equal(got, want)
    assert torch.equal(CORE.branch_weight_table(branch[:0]), _today_branch_weight_table(branch[:0]))
    assert torch.equal(CORE.element_weights(want, branch), CORE.element_weights(CORE.branch_weight_table(branch),
                                                                                branch))
    o = CORE.RunOpts(seed=1, iterations=5)
    assert o.trim_branch is False and o.trim_min is None and o.n_branch == 5 == CORE.N_BRANCH
    assert o.to_dict()["trim_branch"] is False and o.to_dict()["trim_min"] is None
    assert CORE.SPEC.W_MAX == 20 and inspect_default(CORE.train_iteration, "n_branch") == CORE.N_BRANCH


def inspect_default(fn, name):
    import inspect  # noqa: PLC0415

    return inspect.signature(fn).parameters[name].default


def test_trim_on_splits_hold_and_recomputes_capped_weights(CORE):
    n = 100
    branch = torch.zeros(n, 6, dtype=torch.long)
    label = torch.zeros(n, 18)
    # leg 0: 30 HOLD (cam label 0), 60 HOLD with a trim (alternating sign), 10 LIFT (cam label non-zero: not trim)
    label[30:90, 0] = torch.tensor([0.1, -0.1] * 30)
    branch[90:, 0] = CORE.BRANCH_LIFT
    label[90:, 0] = 0.3
    # leg 1: 1 HOLD, 99 trim -> the rare non-trim HOLD hits the cap
    label[1:, 1] = -0.02
    # leg 2: every entry HOLD with a zero CAM label (its hip / knee labels non-zero): stays HOLD
    label[:, 8] = 0.5
    label[:, 14] = -0.5
    # leg 3: WAIT with a non-zero cam label: stays WAIT
    branch[:, 3] = 1
    label[:, 3] = 0.07
    # leg 4: |label| exactly at the threshold stays HOLD, just above is trim (50 / 50)
    label[:50, 4] = 1e-6
    label[50:, 4] = -1.5e-6
    # leg 5: SEAT, zero cam label
    branch[:, 5] = 4
    wb = CORE.weight_branch(branch, label, CORE.SPEC.TRIM_LABEL_MIN)
    T, H = CORE.BRANCH_TRIM, CORE.BRANCH_HOLD
    assert wb.dtype == branch.dtype and wb.shape == branch.shape
    assert (wb[:30, 0] == H).all() and (wb[30:90, 0] == T).all() and (wb[90:, 0] == CORE.BRANCH_LIFT).all()
    assert wb[0, 1] == H and (wb[1:, 1] == T).all()
    assert (wb[:, 2] == H).all() and (wb[:, 3] == 1).all() and (wb[:, 5] == 4).all()
    assert (wb[:50, 4] == H).all() and (wb[50:, 4] == T).all()
    assert torch.equal(CORE.weight_branch(branch, label, None), branch)
    t = CORE.branch_weight_table(wb, 20.0, CORE.N_BRANCH_TRIM).double()
    assert t.shape == (6, 6)
    # leg 0: K = 3 (hold 0.3, lift 0.1, trim 0.6): 1 / (K f)
    assert t[0].tolist() == pytest.approx([1 / 0.9, 0, 1 / 0.3, 0, 0, 1 / 1.8], rel=1e-6)
    # leg 1: K = 2; HOLD 1 / (2 x 0.01) = 50 capped at W_MAX = 20; trim 1 / (2 x 0.99)
    assert t[1, H] == 20.0 and t[1, T].item() == pytest.approx(1 / 1.98, rel=1e-6)
    assert t[2].tolist() == [1.0, 0, 0, 0, 0, 0] and t[3].tolist() == [0, 1.0, 0, 0, 0, 0]
    assert t[4].tolist() == pytest.approx([1.0, 0, 0, 0, 0, 1.0]) and t[5].tolist() == [0, 0, 0, 0, 1.0, 0]
    # HOLD's frequency counts only the non-trim HOLD entries: the off table weighs leg 0's 90 HOLD at 1 / (2 x 0.9)
    off = CORE.branch_weight_table(branch)
    assert off[0, H].item() == pytest.approx(1 / 1.8, rel=1e-6) and off.shape == (6, 5)
    # the same cap, the same element layout (cams / hips / knees share the leg's weight)
    e = CORE.element_weights(t.float(), wb)
    assert e.shape == (n, 18) and torch.equal(e[:, 0:6], e[:, 6:12]) and torch.equal(e[:, 0:6], e[:, 12:18])
    assert e[35, 0].item() == pytest.approx(1 / 1.8, rel=1e-6) and e[0, 1].item() == 20.0
    # the threshold override
    wb2 = CORE.weight_branch(branch, label, 0.05)
    assert (wb2[30:90, 0] == T).all() and (wb2[1:, 1] == H).all() and (wb2[:, 4] == H).all()
    o = CORE.RunOpts(seed=1, iterations=5, trim_branch=True)
    assert o.n_branch == 6 and o.trim_min == CORE.SPEC.TRIM_LABEL_MIN
    assert CORE.RunOpts(seed=1, iterations=5, trim_branch=True, trim_min=0.0999).trim_min == 0.0999
    for bad in (dict(trim_min=0.1), dict(trim_branch=True, trim_min=1.0), dict(trim_branch=1)):
        with pytest.raises(ValueError):
            CORE.RunOpts(seed=1, iterations=5, **bad)


def test_hold_leg_with_a_zero_cam_label_stays_hold(CORE):
    branch = torch.zeros(5, 6, dtype=torch.long)
    label = torch.zeros(5, 18)
    label[:, 6:] = 0.5                                     # hip / knee labels never make a trim leg
    assert torch.equal(CORE.weight_branch(branch, label, CORE.SPEC.TRIM_LABEL_MIN), branch)
    label[2, 3] = -0.1                                     # one leg's cam trim acting
    wb = CORE.weight_branch(branch, label, CORE.SPEC.TRIM_LABEL_MIN)
    assert wb[2, 3] == CORE.BRANCH_TRIM and int((wb == CORE.BRANCH_TRIM).sum()) == 1


def _stub_trainer(CORE, codes, trim=True):
    t = torch.tensor(codes, dtype=torch.int8)
    buf = types.SimpleNamespace(branch=t.reshape(-1, 1).expand(-1, 6).clone(), size=len(codes))
    return types.SimpleNamespace(o=CORE.RunOpts(seed=1, iterations=5, trim_branch=trim), buf=buf)


def test_trim_branch_smoke_check_passes_and_fails(CORE):
    H, T, L_ = CORE.BRANCH_HOLD, CORE.BRANCH_TRIM, CORE.BRANCH_LIFT
    v, ok = CORE.trim_branch_check(_stub_trainer(CORE, [H] * 85 + [T] * 5 + [L_] * 10))
    assert ok is True and v["entries"]["trim"] == 30 and v["entries"]["hold"] == 510
    assert v["mean_weight_trim"] == pytest.approx(1 / 0.15) and v["mean_weight_hold_non_trim"] == pytest.approx(
        1 / 2.55)
    assert v["weight_table_columns"][-1] == "trim" and set(v["weight_table"]["leg0"]) == set(CORE.BRANCH_NAMES_TRIM)
    assert v["shares"]["trim"] == pytest.approx(0.05) and v["w_max"] == 20
    # a threshold that puts almost all of HOLD in "trim" (the rare non-trim HOLD capped at 20): trim's mean weight is
    # BELOW HOLD's -> FAIL (the check exists to catch exactly this)
    v, ok = CORE.trim_branch_check(_stub_trainer(CORE, [H] * 1 + [T] * 89 + [L_] * 10))
    assert ok is False and v["mean_weight_hold_non_trim"] == 20.0 and v["mean_weight_trim"] < 1
    assert CORE.trim_branch_check(_stub_trainer(CORE, [H] * 90 + [L_] * 10))[1] is False       # no trim entries
    assert CORE.trim_branch_check(_stub_trainer(CORE, [T] * 90 + [L_] * 10))[1] is False       # no non-trim HOLD
    v, ok = CORE.trim_branch_check(_stub_trainer(CORE, [H] * 90 + [L_] * 10, trim=False))
    assert ok is True and v["trim_branch"] is False and "not applied" in v["note"]


def test_trim_branch_trainer_step_changes_only_the_weight_code(CORE, classes, cfg, nets_proto, bank, stance,
                                                                tmp_path):
    def pre(env):                                   # cams off home on four legs (two in-band trims), two at home
        env.c[:] = torch.tensor([0.05, -0.2, 0.0, 0.12, 0.0, -0.01])

    tr0, env0, hs0, lab0, *_ = _one_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "off", pre=pre)
    tr1, env1, hs1, lab1, *_ = _one_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "on", pre=pre,
                                         trim_branch=True, deadband=0.25)
    tr2, env2, *_ = _one_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "db", pre=pre, deadband=0.25)
    eff = hs1.stand_eff
    k = int(eff.sum())
    assert torch.equal(env1.seen[-1][0], env2.seen[-1][0])                 # the executed actions: the trim never acts
    assert torch.equal(tr0.buf.tgt[:k], tr1.buf.tgt[:k]) and torch.equal(tr0.buf.x[:k], tr1.buf.x[:k])
    assert torch.equal(tr0.buf.onset[:k], tr1.buf.onset[:k]) and torch.equal(tr0.prev_branch, tr1.prev_branch)
    assert torch.equal(tr0.buf.branch[:k].long(), lab0.branch[eff])                         # off: R_h's code
    want = CORE.weight_branch(lab1.branch, lab1.label.to(torch.float32), CORE.SPEC.TRIM_LABEL_MIN)[eff]
    assert torch.equal(tr1.buf.branch[:k].long(), want)
    assert int((want == CORE.BRANCH_TRIM).sum()) > 0                                       # non-vacuous
    hold0 = lab1.branch[eff] == CORE.BRANCH_HOLD
    assert torch.equal(want == CORE.BRANCH_TRIM, hold0 & (lab1.label[eff][:, :6].abs() > 1e-6))
    assert bool((hold0 & (want == CORE.BRANCH_HOLD)).any())                  # zero-cam HOLD legs stay HOLD
    assert tr0.acc["branch_student"].shape == (6, 5) and tr1.acc["branch_student"].shape == (6, 6)
    assert tr1.acc["branch_student"].sum() == tr0.acc["branch_student"].sum()


def test_trim_off_run_replays_the_default_run_bitwise(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    kw = dict(stop_after=3, iterations=3, smoke=False, n=64)
    _, _, r1 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "x", **kw)
    _, _, r2 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "y", trim_branch=False, **kw)
    _, _, r3 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "z", trim_branch=True, **kw)
    for f in ("stands.jsonl", "episodes.jsonl"):
        assert (tmp_path / "x" / f).read_text() == (tmp_path / "y" / f).read_text()
    a, b, c = (torch.load(r["final_checkpoint"], weights_only=False)["model_state_dict"] for r in (r1, r2, r3))
    assert CORE.SN.state_dicts_equal(a, b)[0]
    assert not CORE.SN.state_dicts_equal(a, c)[0]                         # the trim weights change the gradient
    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("t_")} for r in rows]   # noqa: E731
    la, lb = (CORE.SPEC.read_jsonl(tmp_path / d / "train_log.jsonl") for d in ("x", "y"))
    assert strip(la) == strip(lb) and all("trim_diag" in r for r in la)
    assert r1["trim_branch"] is False and r1["trim_min"] is None and r3["trim_branch"] is True


def test_trim_branch_run_records_and_smoke_beside_the_deadband(CORE, classes, cfg, nets_proto, bank, stance,
                                                               tmp_path):
    d = CORE.SH.RESIDUAL_DEADBAND_PROBE
    tr, env, res = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "t", stop_after=4,
                              smoke=True, n=64, deadband=d, trim_branch=True)
    assert res["trim_branch"] is True and res["trim_min"] == CORE.SPEC.TRIM_LABEL_MIN
    st = json.loads((tmp_path / "t/status.json").read_text())
    assert st["trim_branch"] is True and st[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert st["trim_diag_loaded"].startswith("probe foot_loaded")          # the fake's probe gives per-foot flags
    ck = torch.load(res["final_checkpoint"], weights_only=False)["infos"]["sh_dagger"]
    assert ck["trim_branch"] is True and ck["trim_min"] == CORE.SPEC.TRIM_LABEL_MIN
    assert ck[CORE.SH.RESIDUAL_DEADBAND_KEY] == d
    assert tr.o.to_dict()["trim_branch"] is True
    checks = CORE.smoke_checks(tr, env_parity=[], ckpt=res["final_checkpoint"], env_clip=env.clip_actions)
    by = {c["name"]: c for c in checks}
    assert set(by) == {n for n, _ in CORE.SPEC.SMOKE_DAGGER} and CORE.SPEC.smoke_verdict(checks)["problems"] == []
    assert by["student_deadband_realised"]["pass"] is True                  # the deadband check works beside it
    c = by["trim_branch_weights"]
    v = c["value"]
    assert v["trim_branch"] is True and v["weight_table_columns"] == list(CORE.BRANCH_NAMES_TRIM)
    assert v["entries"]["trim"] > 0 and sum(v["entries"].values()) == v["leg_entries"] == 6 * tr.buf.size
    assert c["pass"] is (v["entries"]["hold"] > 0 and v["mean_weight_hold_non_trim"] is not None
                         and v["mean_weight_trim"] > v["mean_weight_hold_non_trim"])
    sh = by["branch_shares_student_rows"]
    assert set(sh["value"]) == set(CORE.BRANCH_NAMES_TRIM) and sh["trim_branch"] is True
    assert sum(sh["value"].values()) == pytest.approx(1.0)
    log = CORE.SPEC.read_jsonl(tmp_path / "t/train_log.jsonl")
    assert len(log) == 4 and all(set(r["trim_diag"]) == {"n", "exec", "rh"} for r in log)
    # a plain run: the check reads "not applied" (pass) and the branch shares are R_h's five
    tr0, env0, res0 = run_dagger(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "p", stop_after=3,
                                 iterations=3, smoke=True, n=64)
    by0 = {c["name"]: c for c in CORE.smoke_checks(tr0, env_parity=[], ckpt=res0["final_checkpoint"],
                                                  env_clip=env0.clip_actions)}
    assert by0["trim_branch_weights"]["pass"] is True and "not applied" in by0["trim_branch_weights"]["value"]["note"]
    assert set(by0["branch_shares_student_rows"]["value"]) == set(CORE.BRANCH_NAMES)
    ck0 = torch.load(res0["final_checkpoint"], weights_only=False)["infos"]["sh_dagger"]
    assert ck0["trim_branch"] is False and ck0["trim_min"] is None


def test_trim_diagnostic_histograms_on_a_hand_set_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path):
    """Student-executed sigma-0 stand rows (2, 5, the restore row 7; not the sigma-0.3 row 3, not the R rows), legs
    R_h HOLDs with |c| in (0.1, 0.3] and the foot loaded: the executed cam command and R_h's cam label toward home."""
    cams = torch.tensor([0.15, -0.20, 0.25, -0.12, 0.05, 0.30])          # leg 4 below the band; leg 5 on its edge
    seen = {}

    def pre(env):
        env.c[:] = cams
        env.p[:] = 0.0                                                     # every leg down (R's HOLD band)
        seen["m"] = env.probe()

    tr, env, hs, lab, mu, a_s1, noise = _one_step(CORE, classes, cfg, nets_proto, bank, stance, tmp_path / "td",
                                                  pre=pre, deadband=0.25, trim_branch=True)
    a, _ = env.seen[-1]
    rows = torch.zeros(8, dtype=torch.bool)
    rows[[2, 5, 7]] = True
    rows &= hs.stand_eff
    ac = cams.double().abs()                       # the band in float64 as the trainer (and the stand tracker) read it:
    band = (ac > 0.1) & (ac <= 0.3)                # float32 0.30 = 0.3000000119 is OUTSIDE (0.1, 0.3]
    assert band.tolist() == [True, True, True, True, False, False]
    sel = rows[:, None] & (lab.branch == 0) & band[None, :] & seen["m"]["foot_loaded"]
    assert int(sel.sum()) > 0                                              # non-vacuous
    toward = -torch.sign(cams.double())[None, :]
    nb = CORE.SPEC.TRIM_DIAG_NB
    for j, v in enumerate((a[:, :6], lab.label[:, :6])):
        u = (toward * v.double()).clamp(-1, 1)
        k = torch.round((u + 1.0) / 0.001).long()
        want = torch.bincount(k[sel], minlength=nb).double()
        assert torch.equal(tr.trim_it[j], want), j
    # R_h's in-band trim is saturated: +0.1 toward home on every selected leg
    assert torch.equal(tr.trim_it[1].nonzero().flatten(), torch.tensor([1100]))
    assert CORE.SPEC.trim_diag_center(1100) == pytest.approx(0.1)
    # the log row carries the histograms (sparse, keyed by bin) and the accumulator is zeroed
    row = tr._log_row(0, [], [], {"steps": 0}, 0.0, 0.0)
    assert row["trim_diag"]["n"] == int(sel.sum()) and row["trim_diag"]["rh"] == {"1100": int(sel.sum())}
    assert sum(row["trim_diag"]["exec"].values()) == int(sel.sum()) and float(tr.trim_it.abs().sum()) == 0.0
    r = CORE.SPEC.trim_diag_pool([dict(row, it=0)])
    assert r["rh_toward_home"]["median"] == pytest.approx(0.1) and r["ratio"] == pytest.approx(
        r["exec_toward_home"]["median"] / 0.1)
    assert tr.trim_loaded.startswith("probe foot_loaded")


# ===================================================================================== the warm start (Task 4, DESIGN B6)
def test_explicit_default_warm_start_is_the_default_build(CORE, classes, cfg, nets_proto):
    """sh_dagger.py passes (STAND_NET_CKPT, STAND_NET_SHA256) when --warm-start is unset: the build is the default's."""
    same = CORE.build_networks(classes[0], classes[1], cfg, SC.S1_CKPT, "cpu",
                               warm_path=CORE.SH.STAND_NET_CKPT, warm_sha=CORE.SH.STAND_NET_SHA256)
    a, b = nets_proto.student.state_dict(), same.student.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert same.warm == nets_proto.warm


def test_warm_start_with_a_wrong_sha_is_refused(CORE, classes, cfg, nets_proto):
    with pytest.raises(Exception, match="sha"):
        CORE.build_networks(classes[0], classes[1], cfg, SC.S1_CKPT, "cpu",
                            warm_path=CORE.SH.STAND_NET_CKPT, warm_sha="0" * 64)
