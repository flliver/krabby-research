"""Stand-network campaign (``2026-10-11_stand_network``) step 4: the DAgger trainer END TO END on CPU -- the real S1
networks (a CPU copy of model_19994) driving a FAKE env with the wrapper interface (no Isaac / Kit / GPU) -- plus the
entry point ``sn_dagger.py`` (pre-Kit path) and ``sn_dagger.sh`` (dry run).

Pinned:

* routing on one step (map 3.4): stand rows execute R (control / beta-R) or the student's mean (+ DART sigma noise
  from the private generator) and non-stand rows S1's action; every executed action is what the env wrapper clips;
* a short DAgger run: one ``get_observations`` (T6), the global torch / numpy / python RNG untouched (T5), the
  reservoir fed, fixed gradient steps, ``model_<it>.pt`` every ``ckpt_every`` + final, ``train_log.jsonl`` /
  ``stands.jsonl`` / ``status.json``; a same-seed rerun replays bitwise;
* the checkpoint round trip (map 3.7, T11): strict load into fresh networks (the launcher's ``runner.load``), through
  the phase-2 launcher's ``_load_extra`` with a runner whose ``load`` is OPR's, S1's estimator verbatim, the
  optimizer state loading into PPO's single-group Adam;
* the step-6 smoke checks (every name of the list; the forward parity / ckpt / determinism checks pass) and the
  offline replay ``sn_dagger_parity.py`` through the launcher's real ``StandNetMode``;
* recover_probe (A1): no student forward, no optimizer, no checkpoint; spoiler routing; the A1-mode smoke list;
* ``sn_dagger.py``: torch-free import, ``--emit-env`` composes an env the pre-Kit check accepts, every pre-Kit refusal
  fires before Kit with exit code 2, ``--read``; ``sn_dagger.sh --dry-run`` (incl. the ``timeout --signal=TERM``
  bound) and its refusals; the busy patterns (``ps -ww``; the Kit-free ``--read`` / ``--emit-env`` calls excluded;
  long command lines seen whatever ``$COLUMNS``); ``--detach`` returns only once the trainer is visible;
* review 2026-10-12 smoke guards that must be able to fail: the LIFT-onset dose (student rows, loss-weight share),
  the A1 last-row read (schedule + first-draw clauses), terrain levels recorded.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
ENTRY = SCRIPTS / "sn_dagger.py"
SH = SCRIPTS / "sn_dagger.sh"
P2L_PATH = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
W, P0, P1, NP = 1149, 207, 216, 75


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


SC = _load(UNIT / "test_sn_common.py", "_sn_dagger_loop_t_common")


@pytest.fixture(scope="module")
def CORE():
    return _load(SCRIPTS / "sn_dagger_core.py", "_t_sn_dagger_loop_core")


@pytest.fixture(scope="module")
def SPEC(CORE):
    return CORE.SPEC


@pytest.fixture(scope="module")
def ENT():
    return _load(ENTRY, "_t_sn_dagger_entry")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((SC.S1_DIR / "params/agent.yaml").read_text())


@pytest.fixture(scope="module")
def classes():
    if not SC.S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    return SC.rsl_modules()


def make_nets(CORE, classes, cfg):
    return CORE.build_networks(classes[0], classes[1], cfg, SC.S1_CKPT, "cpu")


# ===================================================================================== the fake env
class FakeCmd:
    """The omni command term's training-time interface: ``stand_mask`` / ``time_left`` / ``command_counter``,
    resampled on reset and when ``time_left`` runs out (reset -> resample -> compute: a reset entry holds draw - dt)."""

    def __init__(self, n, g, p_stand=0.5, rs=(3.0, 12.0)):
        self.n, self.g, self.p_stand, self.rs = n, g, p_stand, rs
        self.time_left = torch.zeros(n)
        self.command_counter = torch.zeros(n, dtype=torch.long)
        self.slot_id = torch.ones(n, dtype=torch.long)

    def stand_mask(self):
        return self.slot_id == 0

    def resample(self, ids):
        if ids.numel() == 0:
            return
        u = torch.rand(ids.numel(), 2, generator=self.g)
        self.time_left[ids] = self.rs[0] + u[:, 0] * (self.rs[1] - self.rs[0])
        self.slot_id[ids] = torch.where(u[:, 1] < self.p_stand, 0, 1)
        self.command_counter[ids] += 1

    def compute(self, dt):
        self.time_left -= dt
        self.resample(torch.nonzero(self.time_left <= 0).flatten())


class FakeEnv:
    """A wrapper-shaped env: clips actions to +-1, cams integrate their velocity (faster with the leg up), hips follow
    0.25 x the rod action, rare falls, 30-s episodes at dt 0.1; its own generator (never the global RNG)."""

    def __init__(self, n=16, seed=0, dt=0.1, max_ep=300, p_fall=0.002):
        self.g = torch.Generator().manual_seed(1000 + seed)
        self.n, self.dt, self.max_ep, self.p_fall = n, dt, max_ep, p_fall
        self.num_envs, self.num_obs, self.num_actions, self.clip_actions = n, W, 18, 1.0
        self.cmd = FakeCmd(n, self.g)
        self._fail = torch.zeros(n, dtype=torch.bool)
        self.unwrapped = types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_term=lambda name: self.cmd),
            termination_manager=types.SimpleNamespace(get_term=lambda name: self._fail),
            episode_length_buf=torch.zeros(n, dtype=torch.long), max_episode_length=max_ep, step_dt=dt)
        self.c = (torch.rand(n, 6, generator=self.g) - 0.5) * 0.8
        self.p = torch.zeros(n, 6)
        self.last_a = torch.zeros(n, 18)
        self.hist = torch.zeros(n, 10, 89)
        self.get_calls = 0
        self.seen = []                 # (pre-clip actions, stand mask at the step) per step
        self.terrain_levels = torch.arange(n) % 5    # the parkour manager's per-env levels (stamped on each stand)
        self.cmd.resample(torch.arange(n))
        self.cmd.compute(dt)

    def _obs_buf(self):
        ob = torch.zeros(self.n, 89)
        ob[:, 0:3] = 0.01
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
        a = actions.clamp(-1.0, 1.0)
        self.last_a = a.clone()
        up = self.p <= -0.06
        c = self.c + math.pi * self.dt * a[:, :6] * torch.where(up, 1.0, 0.3)
        self.c = torch.atan2(torch.sin(c), torch.cos(c))
        self.p = self.p + 0.6 * (0.25 * a[:, 6:12] - self.p)
        elb = self.unwrapped.episode_length_buf + 1
        timeouts = elb >= self.max_ep
        fails = torch.rand(self.n, generator=self.g) < self.p_fall
        dones = timeouts | fails
        ids = torch.nonzero(dones).flatten()
        elb[ids] = 0
        self.c[ids] = (torch.rand(ids.numel(), 6, generator=self.g) - 0.5) * 0.8
        self.p[ids] = 0.0
        self.hist[ids] = 0.0
        self.last_a[ids] = 0.0
        self.unwrapped.episode_length_buf = elb
        self.cmd.resample(ids)
        self.cmd.compute(self.dt)
        self._fail = fails
        o = self._obs()
        return o, torch.zeros(self.n), dones.long(), {"time_outs": timeouts & ~fails, "observations": {"policy": o}}


def run_dagger(CORE, classes, cfg, out, *, seed=18, arm="mask", stop_after=6, iterations=60, smoke=True, n=16,
               terrain=True):
    nets = make_nets(CORE, classes, cfg)
    env = FakeEnv(n=n, seed=seed)
    o = CORE.RunOpts(mode="dagger", arm=arm, seed=seed, iterations=iterations, stop_after=stop_after, num_envs=n,
                     minibatches=2, minibatch=64, buffer_rows=3000, ckpt_every=3, smoke=smoke, log_every=0)
    tr = CORE.Trainer(env, nets, o, out, train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1],
                      infos={"sn_dagger": {"test": True, "stand_mask": {"mask": 1, "full": 0}[arm]}},
                      terrain_fn=(lambda: env.terrain_levels) if terrain else None, log=lambda s: None)
    return tr, env, tr.run()


# ===================================================================================== routing on one step
def test_one_step_routing(CORE, classes, cfg, tmp_path):
    nets = make_nets(CORE, classes, cfg)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in nets.student.named_parameters():
            if name.startswith("actor.actor_backbone.6"):
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    env = FakeEnv(n=8, seed=1)
    env.cmd.slot_id = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    o = CORE.RunOpts(mode="dagger", arm="mask", seed=5, iterations=30, num_envs=8, buffer_rows=100, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "r", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1])
    obs, _ = env.get_observations()
    tr.ep_kind = torch.tensor([2, 1, 0, 0, 1, 0, 0, 0])      # control, beta-R, student, student, beta-R(sig), ...
    tr.ep_sigma = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0])
    dart = torch.Generator().set_state(tr.gens["dart"].get_state())
    stand = env.cmd.stand_mask().clone()
    with torch.inference_mode():
        tr._step(obs, stand, torch.zeros(8, dtype=torch.bool), 0)
        tgt, _ = CORE.SN.r_label(obs)
        mu = CORE.student_mean(nets.student, CORE.stand_input(obs, nets, 1))
        a_s1 = CORE.s1_actions(obs, nets)
    noise = torch.randn(8, 18, generator=dart)
    a, st = env.seen[-1]
    assert torch.equal(st, stand)
    assert torch.equal(a[0], tgt[0]) and torch.equal(a[1], tgt[1])                     # control, beta-R (sigma 0)
    assert torch.equal(a[2], mu[2]) and torch.equal(a[5], mu[5])                       # student mean
    assert torch.equal(a[3], mu[3] + 0.3 * noise[3]) and torch.equal(a[4], tgt[4] + 0.1 * noise[4])   # DART
    assert torch.equal(a[6:], a_s1[6:])                                                 # S1 off stand
    assert tr.buf.seen == 6 and tr.student_forwards == 1
    assert env.get_calls == 1


# ===================================================================================== a short DAgger run
def test_dagger_run_end_to_end(CORE, classes, cfg, tmp_path, SPEC):
    torch.manual_seed(77)
    np.random.seed(3)
    before = CORE.save_rng()
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "a")
    assert CORE.rng_equal(before, CORE.save_rng())                     # T5: no trainer draw on the global streams
    assert env.get_calls == 1                                          # T6
    assert res["status"] == "done" and res["iterations_run"] == 6 and res["opt_steps"] == 12
    assert [Path(p).name for p in res["checkpoints"]] == ["model_3.pt", "model_6.pt"]
    assert res["final_checkpoint"].endswith("model_6.pt")
    assert tr.buf.seen > 0 and tr.buf.size == min(3000, tr.buf.seen)
    log = [json.loads(x) for x in (tmp_path / "a/train_log.jsonl").read_text().splitlines()]
    assert [r["it"] for r in log] == list(range(6)) and all(r["train_steps"] == 2 for r in log)
    assert log[0]["beta"] == 1.0 and log[5]["beta"] == pytest.approx(1 - 5 / 20)
    recs = SPEC.read_stands(tmp_path / "a/stands.jsonl")
    assert recs and all(set(SPEC.RECORD_KEYS) <= set(r) for r in recs)
    assert all(r["mode"] == "dagger" and r["kind"] in SPEC.EXECUTORS for r in recs)
    assert all(r["last_row_ep"] == r["end_ep"] - 1 and r["entry_ep"] <= r["last_row_ep"] for r in recs)
    assert all(r["sched_end0_ep"] <= r["sched_end_ep"] and (r["n_merges"] > 0 or r["sched_end0_ep"] == r["sched_end_ep"])
               for r in recs)                                           # the first draw is never re-extended
    assert all(r["terrain_level"] == r["env"] % 5 for r in recs)
    assert json.loads((tmp_path / "a/status.json").read_text())["status"] == "done"
    # every executed action went through the wrapper's clip (the fake clips; the trainer hands the unclipped value)
    assert all(a.shape == (16, 18) for a, _ in env.seen) and len(env.seen) == 6 * 24
    # premise read runs on the file
    pr = SPEC.premise_read(recs)
    assert set(pr) >= {"student", "control", "beta_r", "ratio", "stop"}


def test_same_seed_replays_bitwise(CORE, classes, cfg, tmp_path):
    _, _, r1 = run_dagger(CORE, classes, cfg, tmp_path / "x", stop_after=3, smoke=False)
    _, _, r2 = run_dagger(CORE, classes, cfg, tmp_path / "y", stop_after=3, smoke=False)
    assert (tmp_path / "x/stands.jsonl").read_text() == (tmp_path / "y/stands.jsonl").read_text()
    a = torch.load(r1["final_checkpoint"], weights_only=False)
    b = torch.load(r2["final_checkpoint"], weights_only=False)
    ok, why = CORE.SN.state_dicts_equal(a["model_state_dict"], b["model_state_dict"])
    assert ok, why


# ===================================================================================== checkpoints
def test_checkpoint_round_trip(CORE, classes, cfg, tmp_path):
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "c", stop_after=3, smoke=False)
    path = res["final_checkpoint"]
    ck = torch.load(path, weights_only=False)
    assert set(CORE.REQUIRED_CKPT_KEYS) <= set(ck) and "alg_state" not in ck and ck["iter"] == 3
    assert all(v.device.type == "cpu" for v in ck["model_state_dict"].values())
    assert set(ck["model_state_dict"]) == set(tr.nets.s1.state_dict())                 # critic + std_raw included
    assert not any(k.startswith(("_orig_mod.", "module.")) for k in ck["model_state_dict"])
    s1 = torch.load(SC.S1_CKPT, map_location="cpu", weights_only=False)
    ok, why = CORE.SN.state_dicts_equal(ck["estimator_state_dict"], s1["estimator_state_dict"])
    assert ok, why                                                                       # S1's estimator, verbatim
    assert ck["infos"]["sn_dagger"]["iterations_done"] == 3 and ck["infos"]["sn_dagger"]["test"] is True
    assert CORE.checkpoint_reasons(path, tr.nets, classes[0], classes[1], cfg) == []
    pol, est, _ = CORE.load_strict(path, classes[0], classes[1], cfg)
    obs = SC.obs_batch(24, seed=4)
    with torch.inference_mode():
        x = CORE.stand_input(obs, tr.nets, 1)
        assert torch.equal(pol.act_inference(x, hist_encoding=True), CORE.student_mean(tr.nets.student, x))
    # trained: the backbone moved, nothing else did
    moved = {k for k in ck["model_state_dict"] if not torch.equal(ck["model_state_dict"][k],
                                                                    s1["model_state_dict"][k].cpu())}
    assert moved and all(k.startswith("actor.actor_backbone.") for k in moved)
    # T11: PPO's single-group Adam over policy.parameters() takes the optimizer state (play.py / evaluation.py)
    fresh = classes[0](W, 18, **{k: v for k, v in copy.deepcopy(cfg["policy"]).items() if k != "class_name"})
    ppo_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    ppo_opt.load_state_dict(ck["optimizer_state_dict"])
    assert ppo_opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    # the phase-2 launcher's _load_extra with a runner whose load() is OPR's (torch.load, strict policy + estimator)
    p2l = _load(P2L_PATH, "_t_sn_dagger_p2l")

    class OprLikeRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            pc, ec = train_cfg["policy"], train_cfg["estimator"]
            pc.pop("class_name")
            ec.pop("class_name")
            self.alg = types.SimpleNamespace(policy=classes[0](W, 18, **pc), estimator=classes[1](**ec))

        def load(self, path, load_optimizer=True):
            d = torch.load(path, weights_only=False)
            self.alg.policy.load_state_dict(d["model_state_dict"])
            self.alg.estimator.load_state_dict(d["estimator_state_dict"])
            return d["infos"]

        def get_inference_policy(self, device=None):
            self.alg.policy.eval()
            return self.alg.policy.act_inference

        def get_estimator_inference_policy(self, device=None):
            self.alg.estimator.eval()
            return self.alg.estimator

    runner, st_pol, st_est = p2l._load_extra(OprLikeRunner, None, copy.deepcopy(cfg), path, "cpu")
    ok, why = CORE.SN.state_dicts_equal(st_est.state_dict(), tr.nets.est.state_dict())
    assert ok, why                                                     # the launcher's L5 identity check passes
    with torch.inference_mode():
        assert torch.equal(st_pol(x, hist_encoding=True), CORE.student_mean(tr.nets.student, x))


def test_checkpoint_reasons_refuse(CORE, classes, cfg, tmp_path):
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "d", stop_after=3, smoke=False)
    ck = torch.load(res["final_checkpoint"], weights_only=False)
    bad = copy.deepcopy(ck)
    bad["estimator_state_dict"]["estimator.4.bias"] = bad["estimator_state_dict"]["estimator.4.bias"] + 1e-3
    torch.save(bad, tmp_path / "bad_est.pt")
    r = CORE.checkpoint_reasons(tmp_path / "bad_est.pt", tr.nets, classes[0], classes[1], cfg)
    assert any("estimator" in x for x in r)
    nopt = copy.deepcopy(ck)
    nopt.pop("optimizer_state_dict")
    torch.save(nopt, tmp_path / "nopt.pt")
    assert any("optimizer_state_dict" in x for x in CORE.checkpoint_reasons(tmp_path / "nopt.pt", tr.nets,
                                                                             classes[0], classes[1], cfg))


# ===================================================================================== the step-6 smoke + offline parity
def test_dagger_smoke_checks_and_offline_parity(CORE, classes, cfg, tmp_path, SPEC):
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "s")
    checks = CORE.smoke_checks_dagger(tr, env_parity=[], ckpt=res["final_checkpoint"])
    v = SPEC.smoke_verdict("dagger", checks)
    assert v["problems"] == []                                         # every listed check is present, boolean
    by = {c["name"]: c for c in checks}
    for name in ("forward_parity", "ckpt_loads", "s1_deterministic", "obs_parity_env", "obs_parity_cols",
                 "buffer_projection", "dart_sigma_realized", "terrain_level_recorded"):
        assert by[name]["pass"] is True, (name, by[name])
    assert by["forward_parity"]["value"]["equal"] is True and by["forward_parity"]["value"]["stand_rows"] > 0
    assert by["handovers"]["value"]["handover"] >= 0
    bad = CORE.smoke_checks_dagger(tr, env_parity=["KRABBY_X: trainer '1' vs eval '0'"], ckpt=res["final_checkpoint"])
    assert {c["name"]: c for c in bad}["obs_parity_env"]["pass"] is False
    cap = tmp_path / "s/smoke_capture.pt"
    assert cap.is_file()
    par = _load(SCRIPTS / "sn_dagger_parity.py", "_t_sn_dagger_loop_parity")
    r = par.replay(cap)
    assert r["verdict"] == "PASS" and r["cpu_trainer_vs_standnetmode_bitwise"] is True, r
    assert r["cpu_vs_captured_max_abs"] == 0.0                         # captured on CPU here: bitwise
    # a capture whose checkpoint changed is refused
    c = torch.load(cap, weights_only=False)
    c["ckpt_sha256"] = "0" * 64
    torch.save(c, tmp_path / "cap_bad.pt")
    assert par.replay(tmp_path / "cap_bad.pt")["verdict"] == "REFUSED"


# ===================================================================================== A1 recover_probe
def test_recover_probe_run(CORE, classes, cfg, tmp_path, SPEC):
    nets = make_nets(CORE, classes, cfg)
    env = FakeEnv(n=24, seed=4)
    o = CORE.RunOpts(mode="recover_probe", seed=18, iterations=40, num_envs=24, smoke=True, log_every=0)
    torch.manual_seed(9)
    before = CORE.save_rng()
    tr = CORE.Trainer(env, nets, o, tmp_path / "p", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1],
                      terrain_fn=lambda: env.terrain_levels)
    res = tr.run()
    assert CORE.rng_equal(before, CORE.save_rng())
    assert res["status"] == "done" and res["opt_steps"] == 0 and res["student_forwards"] == 0
    assert res["checkpoints"] == [] and tr.opt is None and tr.buf is None
    assert not list((tmp_path / "p").glob("model_*.pt"))
    recs = SPEC.read_stands(tmp_path / "p/stands.jsonl")
    assert recs and all(r["mode"] == "recover_probe" and r["kind"] in SPEC.ASSIGNS and r["sigma"] == 0.0
                        for r in recs)
    checks = CORE.smoke_checks_a1(tr)
    v = SPEC.smoke_verdict("a1", checks)
    assert v["problems"] == []
    by = {c["name"]: c for c in checks}
    for name in ("a1_executed_r_or_spoiler", "a1_no_optimizer", "a1_dart_zero", "a1_statue_zero",
                 "a1_chatter_sigma", "a1_last_row_read", "terrain_level_recorded"):
        assert by[name]["pass"] is True, (name, by[name])
    lr = by["a1_last_row_read"]["value"]
    assert lr["exits"] > 0 and lr["exit_schedule_off_by_more_than_1"] == 0 and lr["unmerged_exits"] > 0
    # the schedule clause can fail: an exit whose last row is 2 steps off its scheduled end
    recs_bad = list(tr.tracker.done)
    j = next(i for i, r in enumerate(recs_bad) if r["end"] == "exit")
    tr.tracker.done[j] = dict(recs_bad[j], sched_end_ep=recs_bad[j]["sched_end_ep"] + 3)
    assert {c["name"]: c for c in CORE.smoke_checks_a1(tr)}["a1_last_row_read"]["pass"] is False
    # ... and so can the first-draw read (abs_c0 != abs_c on an unmerged exit inside its first draw)
    tr.tracker.done[j] = recs_bad[j]
    k = next(i for i, r in enumerate(recs_bad) if r["end"] == "exit" and r["n_merges"] == 0
             and r["last_row_ep"] <= r["sched_end0_ep"] - 1)
    tr.tracker.done[k] = dict(recs_bad[k], abs_c0=[x + 0.01 for x in recs_bad[k]["abs_c0"]])
    assert {c["name"]: c for c in CORE.smoke_checks_a1(tr)}["a1_last_row_read"]["pass"] is False
    tr.tracker.done[k] = recs_bad[k]
    a1 = SPEC.a1_read(recs)
    assert a1["verdict"] in ("PASS", "STOP", "INVALID") and set(a1["groups"]) == set(SPEC.ASSIGNS)
    # routing: control stands execute R from entry; a spoiled stand's window rows execute its spoiler
    for r in recs:
        assert r["take_ep"] >= r["entry_ep"]
        if r["kind"] == "control":
            assert r["take_ep"] == r["entry_ep"]
        else:
            assert 10 <= r["take_ep"] - r["entry_ep"] <= 40                # U[1, 4] s at dt 0.1


def test_recover_probe_refuses_an_arm_and_dagger_needs_one(CORE):
    with pytest.raises(ValueError):
        CORE.RunOpts(mode="recover_probe", arm="mask", seed=1, iterations=5)
    with pytest.raises(ValueError):
        CORE.RunOpts(mode="dagger", arm=None, seed=1, iterations=5)
    with pytest.raises(ValueError):
        CORE.RunOpts(mode="dagger", arm="mask", seed=1, iterations=5, stop_after=6)
    assert CORE.RunOpts(mode="dagger", arm="full", seed=1, iterations=5).stand_mask == 0
    assert CORE.RunOpts(mode="dagger", arm="mask", seed=1, iterations=5).stand_mask == 1


# ===================================================================================== sn_dagger.py (pre-Kit)
def test_entry_import_is_torch_free():
    code = textwrap.dedent(f"""\
        import importlib.util, sys
        s = importlib.util.spec_from_file_location("_x", {str(ENTRY)!r})
        m = importlib.util.module_from_spec(s); sys.dont_write_bytecode = True; s.loader.exec_module(m)
        bad = [k for k in ("torch", "numpy", "yaml", "isaaclab", "_sn_dagger_core") if k in sys.modules]
        print("BAD", bad); sys.exit(1 if bad else 0)
        """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr


def test_emit_env_composes_an_accepted_env(SPEC):
    dirty = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_STAND_"))}
    dirty.update(KRABBY_FOO="1", KRABBY_OMNI_PHASE="2c", P2R_MODE="st2", SN_STAND_CKPT="/x", CUDA_VISIBLE_DEVICES="")
    script = f'eval "$({sys.executable} -B {ENTRY} --emit-env)" && env -0'
    r = subprocess.run(["bash", "-c", script], capture_output=True, timeout=120, env=dirty)
    assert r.returncode == 0, r.stderr.decode()
    env = dict(x.split("=", 1) for x in r.stdout.decode().split("\0") if "=" in x)
    assert SPEC.env_reasons(env) == [], SPEC.env_reasons(env)
    assert "KRABBY_FOO" not in env and "P2R_MODE" not in env and "SN_STAND_CKPT" not in env
    assert env["KRABBY_OMNI_PHASE"] == "2a" and env["KRABBY_PHASEOUT"] == "" and env["KRABBY_OMNI_DIR_BAND"] == ""
    assert env["OMNI_KIT_ACCEPT_EULA"] == "yes" and env["PYTHONUNBUFFERED"] == "1"


def _good_args(out):
    return ["--mode", "dagger", "--arm", "mask", "--seed", "18", "--iterations", "300", "--out", str(out)]


def _skip_without_the_s1_walker(ENT):
    """The entry's pre-Kit checks need S1's machine-local checkpoint, its params/agent.yaml and the REFS env stamp
    (gitignored parkour/logs + parkour/outputs); a checkout without them (CI) never reaches the accepting path."""
    need = [ENT.SPEC.S1_CKPT, ENT.SPEC.S1_AGENT_YAML, ENT.SPEC.REFS_ENV_STAMP / "run_meta.json"]
    missing = [str(p) for p in need if not Path(p).is_file()]
    if missing:
        pytest.skip(f"the S1 walker's machine-local files are not on this machine: {', '.join(missing)}")


def test_entry_refusals_fire_before_kit(ENT, SPEC, tmp_path, monkeypatch, capsys):
    _skip_without_the_s1_walker(ENT)
    called = []
    monkeypatch.setattr(ENT, "run_kit", lambda o, rest: called.append((o, rest)) or 0)
    good_env = SPEC.train_env({"PATH": os.environ.get("PATH", "")})
    out = tmp_path / "run"
    assert ENT.main([str(ENTRY)] + _good_args(out) + ["--headless"], environ=good_env) == 0
    assert len(called) == 1 and called[0][1] == ["--headless"]
    called.clear()
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "model_3.pt").write_bytes(b"x")
    cases = [
        (_good_args(out)[:4] + _good_args(out)[6:], good_env, "required"),                       # no --seed
        (["--mode", "dagger", "--seed", "1", "--iterations", "3", "--out", str(out)], good_env, "--arm"),
        (["--mode", "recover_probe", "--arm", "mask", "--seed", "1", "--iterations", "3", "--out", str(out)],
         good_env, "no --arm"),
        (_good_args("rel/run"), good_env, "not absolute"),
        (_good_args(tmp_path / "a b"), good_env, "whitespace"),
        (_good_args(out) + ["--smoke", "--smoke-iters", "500"], good_env, "smoke-iters"),
        (_good_args(tmp_path / "busy"), good_env, "already holds"),
        (_good_args(out), dict(good_env, KRABBY_OMNI_STAND_OBS="clock"), "STAND_OBS"),
        (_good_args(out), {k: v for k, v in good_env.items() if k != "KRABBY_RSI_FRAC"}, "RSI_FRAC"),
        (_good_args(out), dict(good_env, P2R_BANK_MAX="5"), "P2R_"),
        (_good_args(out), dict(good_env, KRABBY_RESTANCE="default"), "KRABBY_RESTANCE"),
        (_good_args(out) + ["--iterationsx", "3"], good_env, "malformed"),
    ]
    capsys.readouterr()
    for argv, env, needle in cases:
        # review 2026-10-12: every pre-Kit refusal is exit code 2 (the documented contract), message on stderr
        assert ENT.main([str(ENTRY)] + argv, environ=env) == ENT.REFUSED_RC == 2, argv
        err = capsys.readouterr().err
        assert needle in err, (argv, err)
    assert called == []
    # argparse's own errors keep their exit code 2 (SystemExit with an int)
    with pytest.raises(SystemExit) as ei:
        ENT.main([str(ENTRY), "--seed", "x"], environ=good_env)
    assert ei.value.code == 2


def test_entry_refusal_exit_code_in_a_process(tmp_path):
    """The process exit code (what sn_dagger.sh / the orchestrator see) of a pre-Kit refusal is 2, not 1."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_"))}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-B", str(ENTRY)] + _good_args(tmp_path / "run"), capture_output=True,
                       text=True, timeout=120, env=env)
    assert r.returncode == 2 and "refused before Kit" in r.stderr, (r.returncode, r.stderr)
    assert not (tmp_path / "run").exists()


def test_entry_read_mode(ENT, CORE, classes, cfg, tmp_path, capsys):
    run_dagger(CORE, classes, cfg, tmp_path / "rr", stop_after=3, smoke=False)
    assert ENT.main([str(ENTRY), "--read", str(tmp_path / "rr"), "--premise", "0:3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["premise"]["window"] == [0, 3] and out["n_records"] >= 0
    assert ENT.main([str(ENTRY), "--read", str(tmp_path / "rr"), "--a1"]) == 0
    assert "a1" in json.loads(capsys.readouterr().out)


def test_entry_provenance_names_every_file(ENT):
    o = types.SimpleNamespace(mode="dagger", arm="mask", seed=18, iterations=3000, smoke=False, num_envs=1024,
                              buffer_gb=4.0)
    p = ENT.provenance(o, "ab" * 32)["sn_dagger"]
    for k in ("sn_dagger_sha256", "sn_dagger_core_sha256", "sn_dagger_spec_sha256", "sn_common_sha256",
              "restance_module_sha256"):
        assert len(p[k]) == 64
    assert p["stand_mask"] == 1 and p["r_knob"] == ENT.SN.RESTANCE_D3 and p["s1_sha256"] == ENT.SPEC.S1_SHA256


# ===================================================================================== sn_dagger.sh
def _sh(args, tmp_path, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2R_", "SN_"))}
    env.update({"SN_OUT": str(tmp_path / "snout"), "SN_DDC": str(tmp_path / "ddc"), "CUDA_VISIBLE_DEVICES": "",
                "SN_PY": sys.executable, **(extra_env or {})})
    (tmp_path / "ddc").mkdir(exist_ok=True)
    return subprocess.run(["bash", str(SH)] + args, capture_output=True, text=True, timeout=120, env=env)


def test_sh_dry_run_composes_the_command(tmp_path, SPEC):
    r = _sh(["--dry-run", "SN_SMOKE_mask_s18", "--", "--mode", "dagger", "--arm", "mask", "--seed", "18",
             "--iterations", "3000", "--smoke"], tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "systemd-run --user --scope -q --unit=sn-dagger-SN_SMOKE_mask_s18-" in out
    assert "-p MemoryMax=48G -p MemorySwapMax=0" in out
    # review 2026-10-12: a wall-clock bound (SIGTERM, the signal the trainer handles; never SIGKILL)
    assert f"MemorySwapMax=0 timeout --signal=TERM 3600 {sys.executable} {ENTRY} " in out
    assert "--kill-after" not in out and " -k " not in out
    assert f"{ENTRY} --headless --mode dagger --arm mask --seed 18 --iterations 3000 --smoke --out " in out
    assert str(tmp_path / "snout/runs/SN_SMOKE_mask_s18") in out
    assert "export KRABBY_OMNI_STAND_OBS=0" in out and "export OMNI_KIT_ACCEPT_EULA=yes" in out
    assert "export KRABBY_PHASEOUT=''" in out
    assert not (tmp_path / "snout/runs").exists()                   # a dry run creates nothing
    full = _sh(["--dry-run", "SN_A1_s18", "--", "--mode", "recover_probe", "--seed", "18", "--iterations", "900"],
               tmp_path)
    assert full.returncode == 0 and "timeout --signal=TERM 14400 " in full.stdout, full.stderr
    own = _sh(["--dry-run", "SN_A1_s18", "--", "--mode", "recover_probe", "--seed", "18", "--iterations", "900"],
              tmp_path, {"SN_TIMEOUT": "2400"})
    assert own.returncode == 0 and "timeout --signal=TERM 2400 " in own.stdout
    for bad in ("0", "1h", "-5", "3.5"):
        r = _sh(["--dry-run", "SN_A1_s18", "--", "--mode", "recover_probe", "--seed", "18", "--iterations", "9"],
                tmp_path, {"SN_TIMEOUT": bad})
        assert r.returncode == 5 and "SN_TIMEOUT" in r.stderr, (bad, r.stderr)


@pytest.mark.parametrize("args, needle", [
    (["bad-name", "--", "--mode", "dagger"], "NAME"),
    (["N1", "--mode", "dagger"], "expected '--'"),
    (["N1", "--", "--out", "/x"], "not allowed"),
    (["N1", "--", "--emit-env"], "not allowed"),
])
def test_sh_refusals(tmp_path, args, needle):
    r = _sh(["--dry-run"] + args, tmp_path)
    assert r.returncode == 5 and needle in r.stderr, r.stderr


def test_sh_refuses_a_non_empty_run_dir_and_stop_queue(tmp_path):
    (tmp_path / "snout/runs/N2").mkdir(parents=True)
    (tmp_path / "snout/runs/N2/model_1.pt").write_bytes(b"x")
    r = _sh(["--dry-run", "N2", "--", "--mode", "recover_probe", "--seed", "1", "--iterations", "3"], tmp_path)
    assert r.returncode == 5 and "not empty" in r.stderr
    (tmp_path / "snout/STOP_QUEUE").write_text("")
    r = _sh(["--dry-run", "N3", "--", "--mode", "recover_probe", "--seed", "1", "--iterations", "3"], tmp_path)
    assert r.returncode == 4


def _var(src: str, name: str) -> str:
    return src.split(f"{name}='", 1)[1].split("'", 1)[0]


def _busy(pat: str, nonkit: str, lines: list[str]) -> bool:
    """The scripts' busy pipeline (``ps -eo args | grep -E PAT | grep -qvE NONKIT``) on ``lines``."""
    r = subprocess.run(["bash", "-c", 'printf "%s\\n" "$@" | grep -E "$P" | grep -qvE "$N"', "_", *lines],
                       env={"P": pat, "N": nonkit, "PATH": os.environ.get("PATH", "")}, text=True)
    return r.returncode == 0


def test_sh_busy_pattern():
    src = SH.read_text()
    pat, nonkit = _var(src, "ISAAC_PAT"), _var(src, "NONKIT_PAT")
    assert 'busy() { ps -ww -eo args | grep -E "$ISAAC_PAT" | grep -qvE "$NONKIT_PAT"; }' in src
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    yes = [f"{py} {ENTRY} --headless --mode dagger --out /r/N",
           f"{py} /x/crab_hex_forward_task/scripts/eval_crab_hex_gait.py --headless",
           f"{py} {SCRIPTS}/sn_stand_net_eval_crab_hex_gait.py --headless",
           f"{py} /home/x/parkour/scripts/rsl_rl/train.py --task T",
           f"{py} /x/tools/run_candidates.py --campaign-dir c"]
    # review 2026-10-12: the Kit-free --read / --emit-env calls never block a launch
    no = [f"{py} -B {ENTRY} --emit-env", f"{py} {ENTRY} --emit-env", f"{py} {ENTRY} --read /r/N --a1",
          f"{py} {ENTRY} --read=/r/N", f"grep -qE {pat}", f"bash {SH} N -- --mode dagger",
          f"systemd-run --user --scope -q {py} {ENTRY}",
          f"timeout --signal=TERM 3600 {py} {ENTRY} --headless --out /r/N"]
    for line in yes:
        assert _busy(pat, nonkit, [line]), line
    for line in no:
        assert not _busy(pat, nonkit, [line]), line
    assert _busy(pat, nonkit, [no[2], yes[0]])                       # a read beside a real trainer: still busy


# ===================================================================================== guards that must be able to fail
def test_a1_routing_check_catches_a_broken_route(CORE, classes, cfg, tmp_path, monkeypatch):
    """The A1 smoke's 'executed stand rows are R or spoiler only' compares the ROUTED actions with R / the spoiler /
    S1 -- a router that hands S1's action to stand rows must fail it (the check is not a tautology)."""
    nets = make_nets(CORE, classes, cfg)
    env = FakeEnv(n=8, seed=2)
    o = CORE.RunOpts(mode="recover_probe", seed=3, iterations=3, num_envs=8, smoke=True, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "b", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1])
    monkeypatch.setattr(CORE.SN, "route_actions", lambda stand, a_stand, a_s1: a_s1)
    tr.run()
    by = {c["name"]: c for c in CORE.smoke_checks_a1(tr)}
    assert by["a1_executed_r_or_spoiler"]["pass"] is False and by["a1_executed_r_or_spoiler"]["value"][
        "mismatched_rows"] > 0


def test_run_start_entries_and_stop_request(CORE, classes, cfg, tmp_path, SPEC):
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "e", stop_after=4, smoke=False)
    recs = SPEC.read_stands(tmp_path / "e/stands.jsonl")
    assert {r["entry_kind"] for r in recs} <= {"spawn", "handover", "run_start"}
    assert any(r["entry_kind"] == "run_start" for r in recs)
    # a stop request (SIGTERM handler) ends the run at the iteration boundary: status interrupted, no final save
    nets = make_nets(CORE, classes, cfg)
    env2 = FakeEnv(n=8, seed=5)
    o = CORE.RunOpts(mode="dagger", arm="full", seed=5, iterations=50, num_envs=8, buffer_rows=500, minibatch=32,
                     minibatches=1, ckpt_every=100, log_every=0)
    tr2 = CORE.Trainer(env2, nets, o, tmp_path / "f", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1])
    orig = tr2._log_row

    def stop_after_two(it, *a, **k):
        if it == 1:
            tr2.stop_requested = True
        return orig(it, *a, **k)

    tr2._log_row = stop_after_two
    r2 = tr2.run()
    assert r2["status"] == "interrupted" and r2["iterations_run"] == 2 and r2["checkpoints"] == []
    assert json.loads((tmp_path / "f/status.json").read_text())["status"] == "interrupted"


def test_compaction_guard_fails_closed(CORE, classes, cfg, tmp_path, monkeypatch):
    nets = make_nets(CORE, classes, cfg)
    env = FakeEnv(n=8, seed=6)
    o = CORE.RunOpts(mode="dagger", arm="mask", seed=6, iterations=2, num_envs=8, buffer_rows=100, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "g", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1])
    obs = SC.obs_batch(8, seed=3)                                       # every column non-zero
    tr.compaction_guard(obs)                                            # the real layout passes
    tr.buf.hist_start = 400                                             # a wrong compaction (drops read column 399)
    tr.buf.n_keep = tr.buf.p1 + (W - 400)
    with pytest.raises(RuntimeError, match="compaction"):
        tr.compaction_guard(obs)


def test_episode_draws_shares_and_private_stream(CORE, SPEC):
    """Per-episode draws (OPEN-4 / OPEN-5 / PLAN D3) on a large batch: control ~10 %, beta-R ~beta of the rest,
    sigma ~ {0: 1/2, 0.1: 1/4, 0.3: 1/4} with sigma 0 on every control episode; only new-episode rows change; the
    draw comes from the private 'episode' generator (the global RNG never moves)."""
    n = 200000
    fake = types.SimpleNamespace(gens=CORE.make_generators(7, "cpu"), n=n, device="cpu", dagger=True,
                                 o=types.SimpleNamespace(iterations=300, control_share=SPEC.CONTROL_SHARE),
                                 ep_kind=torch.full((n,), 9, dtype=torch.long), ep_sigma=torch.full((n,), -1.0))
    torch.manual_seed(1)
    before = CORE.save_rng()
    new = torch.arange(n) % 2 == 0
    CORE.Trainer._episode_draws(fake, new, 50)                     # beta(50, 300) = 0.5
    assert CORE.rng_equal(before, CORE.save_rng())
    k, s = fake.ep_kind[new], fake.ep_sigma[new]
    assert bool((fake.ep_kind[~new] == 9).all()) and bool((fake.ep_sigma[~new] == -1.0).all())
    ctrl = k == SPEC.EX_CONTROL
    assert abs(float(ctrl.float().mean()) - 0.10) < 0.005
    nc = ~ctrl
    assert abs(float((k[nc] == SPEC.EX_BETA_R).float().mean()) - 0.5) < 0.01
    assert bool((s[ctrl] == 0).all())
    for val, p in SPEC.SIGMA_TABLE:
        assert abs(float((s[nc] == val).float().mean()) - p) < 0.01
    CORE.Trainer._episode_draws(fake, torch.ones(n, dtype=torch.bool), 200)          # beta 0: no beta-R
    assert not bool((fake.ep_kind == SPEC.EX_BETA_R).any())


def test_eval_rows_wait_for_the_trainer():
    """T14 from the eval side: sn_eval_rows.sh waits before every row while an sn_dagger.py TRAINER runs (run_arm.sh's
    drivers_idle only sees evals); its Kit-free --read / --emit-env calls do not count (review 2026-10-12)."""
    src = (SCRIPTS / "sn_eval_rows.sh").read_text()
    pat, nonkit = _var(src, "TRAINER_PAT"), _var(src, "NONKIT_PAT")
    assert 'trainer_busy() { ps -ww -eo args | grep -E "$TRAINER_PAT" | grep -qvE "$NONKIT_PAT"; }' in src
    assert "drivers_busy() { ps -ww -eo args | grep -qE \"$DRIVER_PAT\"; }" in src
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    assert _busy(_var(src, "DRIVER_PAT"), "^$", [f"{py} /x/crab_hex_forward_task/scripts/eval_crab_hex_gait.py --a"])
    assert not _busy(_var(src, "DRIVER_PAT"), "^$", [f"{py} {ENTRY} --headless"])     # the trainer: trainer_busy's
    assert _busy(pat, nonkit, [f"{py} {ENTRY} --headless --mode dagger --out /r/N"])
    for line in (f"{py} -B {ENTRY} --emit-env", f"{py} {ENTRY} --emit-env", f"{py} {ENTRY} --read /r/N --premise 0:9",
                 f"{py} /x/eval_crab_hex_gait.py --headless"):
        assert not _busy(pat, nonkit, [line]), line
    loop = src.split('for spec in "${ROWS[@]}"; do\n  row=${spec%@*}; es=${spec##*@}; man=', 1)[1]
    assert loop.index("trainer_busy") < loop.index('"${cmd[@]}"; rc=$?')
    assert loop.index("drivers_busy") < loop.index('"${cmd[@]}"; rc=$?')


def _isaac_running() -> bool:
    src = SH.read_text()
    out = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return _busy(_var(src, "ISAAC_PAT"), _var(src, "NONKIT_PAT"), out) if out else False


def _defs(src: str, names: tuple) -> str:
    """The scripts' own one-line variable / function definitions, verbatim, for ``bash -c``."""
    out = []
    for line in src.splitlines():
        t = line.strip()
        if any(t.startswith(f"{n}=") or t.startswith(f"{n}() {{") for n in names):
            out.append(t)
    assert len(out) == len(names), (names, out)
    return "\n".join(out)


@pytest.mark.skipif(sys.platform != "linux", reason="ps / exec -a")
def test_busy_checks_see_long_command_lines_whatever_COLUMNS(tmp_path):
    """Review 2026-10-12 follow-up: plain ``ps -eo args`` truncates every line to an exported $COLUMNS (80 under
    some Python parents -- pytest's children get COLUMNS=80), so an anchored pattern never matched a real Isaac
    command line (the sn_dagger.py path alone is > 150 chars): the busy checks said "idle" with a trainer running.
    The scripts' REAL definitions, run with COLUMNS=80 against a live long-named process, must say busy."""
    ev = (SCRIPTS / "sn_eval_rows.sh").read_text()
    sh = SH.read_text()
    line = f"{tmp_path}/isaac_venv/bin/python {ENTRY} --headless --mode dagger --arm mask --out {tmp_path}/runs/SN_W"
    assert len(line) > 200
    proc = subprocess.Popen(["bash", "-c", f'exec -a "{line}" sleep 3'])
    try:
        time.sleep(0.3)
        env = {"PATH": os.environ.get("PATH", ""), "COLUMNS": "80", "LINES": "24"}
        plain = subprocess.run(["bash", "-c", "ps -eo args | grep -c 'runs/SN_[W]'"], capture_output=True, text=True, env=env)
        assert plain.stdout.strip() == "0"                      # the hazard itself: plain ps loses the line
        for src, names, fn in ((ev, ("TRAINER_PAT", "NONKIT_PAT", "trainer_busy"), "trainer_busy"),
                               (sh, ("ISAAC_PAT", "NONKIT_PAT", "busy"), "busy")):
            r = subprocess.run(["bash", "-c", _defs(src, names) + f"\n{fn}"], env=env)
            assert r.returncode == 0, fn
    finally:
        proc.kill()
        proc.wait()


FAKE_SYSTEMD_RUN = textwrap.dedent('''\
    #!/bin/bash
    # a fake systemd-run for the detach test: skip its options and the timeout prefix, then show the trainer's
    # command line under a harmless short sleep (never runs the trainer, never starts Kit)
    while [ $# -gt 0 ]; do case "$1" in -p) shift 2 ;; -*) shift ;; *) break ;; esac; done
    [ "$1" = timeout ] && shift 3
    echo "$*" >> "$FAKE_SDR_LOG"
    [ "${FAKE_SDR_FAIL:-0}" = 1 ] && exit 9
    exec -a "$*" sleep "${FAKE_SDR_SLEEP:-3}"
    ''')


@pytest.mark.skipif(sys.platform != "linux", reason="ps / setsid / exec -a")
def test_sh_detach_returns_only_once_the_trainer_is_visible(tmp_path):
    """Review 2026-10-12 (T14 race): a detached launch returns only once the trainer's python is visible to the busy
    checks (an eval started right after cannot overlap it); a run that ends first is exit 7. A FAKE systemd-run on
    PATH shows the trainer's command line under ``sleep`` -- the trainer itself never runs."""
    if _isaac_running():
        pytest.skip("an Isaac process is running on this machine (sn_dagger.sh would refuse: busy)")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sdr = bindir / "systemd-run"
    sdr.write_text(FAKE_SYSTEMD_RUN)
    sdr.chmod(0o755)
    venv = tmp_path / "isaac_venv/bin"
    venv.mkdir(parents=True)
    py = venv / "python"                                         # --emit-env runs through it (Kit-free); the trainer never
    py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    log = tmp_path / "sdr.log"
    extra = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "SN_PY": str(py), "FAKE_SDR_LOG": str(log),
             "SN_UP_WAIT": "20", "FAKE_SDR_SLEEP": "2"}
    args = ["--detach", "SN_DET", "--", "--mode", "recover_probe", "--seed", "1", "--iterations", "3"]
    r = _sh(args, tmp_path, extra)
    assert r.returncode == 0 and "trainer visible after" in r.stdout, (r.returncode, r.stdout, r.stderr)
    shown = log.read_text()
    assert f"{py} {ENTRY} --headless --mode recover_probe" in shown
    assert f"--out {tmp_path / 'snout/runs/SN_DET'}" in shown
    for _ in range(100):                                         # the fake trainer (sleep 2) exits: idle again
        if not _isaac_running():
            break
        time.sleep(0.1)
    assert not _isaac_running()
    # a launch that dies before it is visible: exit 7, with the END sentinel quoted
    r2 = _sh(["--detach", "SN_DET2", "--", "--mode", "recover_probe", "--seed", "1", "--iterations", "3"], tmp_path,
             {**extra, "FAKE_SDR_FAIL": "1"})
    assert r2.returncode == 7 and "ended before it became visible" in r2.stderr and "rc=9" in r2.stderr, r2.stderr


def test_lift_onset_dose_check_can_fail_and_weights_are_measured(CORE, classes, cfg, tmp_path):
    """Review 2026-10-12: the minibatch ROW share of LIFT-onset rows is >= ONSET_SHARE by construction whenever any
    exists (reported only); the check binds on onsets on STUDENT-visited stand rows (pre-oversampling dose) and on
    their share of the LOSS WEIGHT after both balancing mechanisms -- each clause can fail on its own."""
    tr, env, res = run_dagger(CORE, classes, cfg, tmp_path / "o", stop_after=12, iterations=24)
    by = {c["name"]: c for c in CORE.smoke_checks_dagger(tr, env_parity=[], ckpt=res["final_checkpoint"])}
    v = by["lift_onset_after_balancing"]["value"]
    assert set(v) >= {"onset_rows_student", "student_stand_rows", "pre_oversampling_share_student_rows",
                      "minibatch_row_share_reported", "loss_weight_share_after_balancing"}
    assert len(tr.mb_onset_w) == len(tr.mb_onset) > 0 and all(0.0 <= x <= 1.0 for x in tr.mb_onset_w)
    want = v["onset_rows_student"] > 0 and v["loss_weight_share_after_balancing"] >= 0.01
    assert by["lift_onset_after_balancing"]["pass"] is want
    # (i) no onset on student rows -> FAIL, whatever the minibatch shares
    tr.smoke_acc["onset_rows_student"].zero_()
    tr.mb_onset_w[:] = [0.5] * len(tr.mb_onset_w)
    c = {x["name"]: x for x in CORE.smoke_checks_dagger(tr, env_parity=[], ckpt=res["final_checkpoint"])}
    assert c["lift_onset_after_balancing"]["pass"] is False
    # (ii) onsets present but under 1 % of the loss weight after balancing -> FAIL (the row share alone would pass)
    tr.smoke_acc["onset_rows_student"].fill_(10)
    tr.mb_onset[:] = [0.2] * len(tr.mb_onset)
    tr.mb_onset_w[:] = [0.004] * len(tr.mb_onset_w)
    c = {x["name"]: x for x in CORE.smoke_checks_dagger(tr, env_parity=[], ckpt=res["final_checkpoint"])}
    assert c["lift_onset_after_balancing"]["pass"] is False
    tr.mb_onset_w[:] = [0.05] * len(tr.mb_onset_w)
    c = {x["name"]: x for x in CORE.smoke_checks_dagger(tr, env_parity=[], ckpt=res["final_checkpoint"])}
    assert c["lift_onset_after_balancing"]["pass"] is True


def test_train_iteration_onset_weight_share_is_the_weighted_minibatch_share(CORE, classes, cfg):
    """``onset_w`` = sum of the onset rows' element weights / sum of all element weights, per minibatch (replayed with
    a copy of the minibatch generator)."""
    nets = make_nets(CORE, classes, cfg)
    g = torch.Generator().manual_seed(4)
    buf = CORE.Reservoir(200, p1=nets.p1, hist_start=nets.hist_start, device="cpu", gen=g)
    x = SC.obs_batch(200, seed=8)
    branch = torch.randint(0, 5, (200, 6), generator=g)
    onset = torch.zeros(200, dtype=torch.bool)
    onset[::7] = True
    buf.add(x, torch.zeros(200, 18), branch, onset)
    gens = CORE.make_generators(3, "cpu")
    mb = torch.Generator().set_state(gens["minibatch"].get_state())
    opt = CORE.make_optimizer(nets.student, 1e-4)
    out = CORE.train_iteration(nets.student, opt, buf, gens, minibatches=1, minibatch=64, onset_share=0.1,
                               w_max=20.0, max_grad_norm=1.0)
    table = CORE.branch_weight_table(buf.branch[:buf.size], 20.0)
    idx = buf.sample(64, 0.1, mb, buf.onset_index())
    w = CORE.element_weights(table, buf.branch[idx]).double().sum(1)
    want = float((w * buf.onset[idx].double()).sum() / w.sum())
    assert out["onset_w"] == pytest.approx(want, rel=1e-12)
    assert out["onset"] == pytest.approx(float(buf.onset[idx].double().mean()), rel=1e-12)


@pytest.mark.parametrize("how", ["none", "raises"])
def test_terrain_check_fails_when_the_accessor_fails(CORE, classes, cfg, tmp_path, how):
    """Review 2026-10-12: a missing / failing terrain accessor leaves terrain_level -1 on every record (the run does
    not stop) -- the smoke's terrain_level_recorded check must fail, in both the DAgger and the A1 lists."""
    nets = make_nets(CORE, classes, cfg)
    env = FakeEnv(n=8, seed=7)

    def boom():
        raise AttributeError("parkour_manager has no term base_parkour")

    fn = None if how == "none" else boom
    logs = []
    o = CORE.RunOpts(mode="recover_probe", seed=7, iterations=12, num_envs=8, smoke=True, log_every=0)
    tr = CORE.Trainer(env, nets, o, tmp_path / "t", train_cfg=cfg, policy_cls=classes[0], est_cls=classes[1],
                      terrain_fn=fn, log=logs.append)
    tr.run()
    recs = list(tr.tracker.done)
    assert recs and all(r["terrain_level"] == -1 for r in recs)
    by = {c["name"]: c for c in CORE.smoke_checks_a1(tr)}
    t = by["terrain_level_recorded"]
    assert t["pass"] is False and t["value"]["terrain_level_missing"] == len(recs) + len(tr.tracker.pending)
    assert t["value"]["accessor_failures"] > 0 and t["value"]["first_error"]
    assert sum("terrain levels unavailable" in x for x in logs) == 1          # warned once, not per step
    if how == "raises":
        assert "AttributeError" in t["value"]["first_error"]
    tr2, _, res2 = run_dagger(CORE, classes, cfg, tmp_path / "t2", stop_after=3, smoke=True, terrain=False)
    d = {c["name"]: c for c in CORE.smoke_checks_dagger(tr2, env_parity=[], ckpt=res2["final_checkpoint"])}
    assert d["terrain_level_recorded"]["pass"] is False
