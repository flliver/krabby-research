"""Stand-network campaign (``2026-10-11_stand_network``) step 4: the DAgger trainer's spec + core units, CPU only
(nothing here starts Isaac / Kit or touches a GPU).

Pinned (PLAN D1-D3 / Amendment 1, CHARTER section 3 OPEN-1..9, IMPLEMENTATION_MAP section 3-4 risks T1-T14):

* the process environment: ``S1_OVERLAY`` == candidates.yaml's ``P2__S1_s18`` overlay VERBATIM (empty strings
  included, T2); the stand-0.5 pool projects onto the SLOTS it ships with (T3); the composed env; the pre-Kit refusal
  of any drift; D5 obs parity against the REFS eval stamp (EVAL_CARRY pinned to run_phases');
* the cfg asserts before gym.make and the network-cfg comparison with S1's own agent.yaml (duck-typed fakes);
* beta (1 -> 0 over the first third), sigma / assignment tables, private generators (T5: the global torch / numpy /
  python RNG never moves);
* R labels == ``restance_target`` with the D3 constants (T4); the D1 forward path (T1): the trainer's routed action ==
  the launcher's REAL ``StandNetMode`` bitwise on CPU (mask 0 and 1), the stand policy's input == the trainer's;
  the MSE forward (on the reservoir's compacted rows) == the rollout forward bitwise;
* spoiler formulas; LIFT-onset detection; per-leg branch weights; the reservoir (cap, dedupe, compaction, sampling);
* the stand tracker on synthetic traces: entries / merges / exits / falls / timeouts, the last-row read (never a
  post-reset row, T9), the scheduled length (OPEN-7), the exit-fall window, the carry-over field;
* the step-7 premise read and the A1 PASS rule on synthetic records; the smoke-list verdict;
* review 2026-10-12: scoring on the stand's FIRST draw (amendment to OPEN-7: inclusion never depends on the outcome;
  homed / |c| on the first draw's last row), run-start stands excluded; the parity replay's trained-arm check.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
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
SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
CANDIDATES = TASK / "experiments/2026-10-07_omni_phase2/candidates.yaml"
RUN_PHASES = TASK / "experiments/tools/run_phases.py"
CMD_MATH = TASK / "mdp/crab_hex_omni_command_math.py"
D3 = ("lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,"
      "tilt_max=0.1,eps_step=0.3")
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


SC = _load(UNIT / "test_sn_common.py", "_sn_dagger_t_common")       # the real-S1 builders


@pytest.fixture(scope="module")
def SPEC():
    return _load(SCRIPTS / "sn_dagger_spec.py", "_t_sn_dagger_spec")


@pytest.fixture(scope="module")
def CORE():
    return _load(SCRIPTS / "sn_dagger_core.py", "_t_sn_dagger_core")


@pytest.fixture(scope="module")
def PAR():
    return _load(SCRIPTS / "sn_dagger_parity.py", "_t_sn_dagger_parity")


@pytest.fixture(scope="module")
def RST():
    return _load(TASK / "mdp/crab_hex_omni_restance.py", "_t_sn_dagger_rst")


@pytest.fixture(scope="module")
def s1_cfg():
    path = SC.S1_DIR / "params/agent.yaml"
    if not path.is_file():
        pytest.skip(f"S1's training-run agent.yaml is not on this machine ({path.relative_to(REPO)}; parkour/logs is gitignored)")
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def nets_factory(CORE, s1_cfg):
    if not SC.S1_CKPT.is_file():
        pytest.skip("S1 model_19994.pt is not on this machine")
    crab, est = SC.rsl_modules()

    def make():
        return CORE.build_networks(crab, est, s1_cfg, SC.S1_CKPT, "cpu")
    return make


def obs_batch(n=32, seed=0):
    """Plausible rows: small values, level IMU, cams / hips spread over R's branches."""
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, W, generator=g) * 0.3
    o[:, 3:5] *= 0.05
    o[:, [18 + 2 * i for i in range(6)]] = (torch.rand(n, 6, generator=g) - 0.5) * 1.2
    o[:, [29 + i for i in range(6)]] = torch.where(torch.rand(n, 6, generator=g) < 0.6, torch.zeros(n, 6),
                                                   -torch.rand(n, 6, generator=g) * 0.2)
    return o


# ===================================================================================== the process environment
def test_overlay_is_candidates_verbatim(SPEC):
    d = yaml.safe_load(CANDIDATES.read_text())
    st = d["chains"]["P2__S1_s18"]["stages"]
    assert len(st) == 1 and st[0]["base"] == "2a"
    got = {k: ("" if v is None else str(v)) for k, v in st[0]["overlay"].items()}
    assert got == SPEC.S1_OVERLAY
    assert SPEC.S1_OVERLAY["KRABBY_PHASEOUT"] == "" and SPEC.S1_OVERLAY["KRABBY_OMNI_DIR_BAND"] == ""


def test_pool_projects_onto_its_slots(SPEC):
    cm = _load(CMD_MATH, "_t_sn_dagger_cmd_math")
    pool = cm.parse_cmd_pool(SPEC.POOL)
    slots = [float(x) for x in SPEC.SLOTS.split(":")]
    want = [x / sum(slots) for x in slots]
    got = pool.projected_slot_probs()
    assert max(abs(a - b) for a, b in zip(got, want)) < 1e-6
    assert abs(got[0] - 0.5) < 1e-12                                        # PLAN D2: stand share 0.5
    s1 = cm.parse_cmd_pool(SPEC.S1_OVERLAY["KRABBY_OMNI_CMD_POOL"])
    # within-slot type proportions of the non-stand slots are S1's
    assert pool.names == s1.names and pool.weights[1:] == s1.weights[1:]


def test_expected_env_and_train_env(SPEC):
    want = SPEC.expected_krabby_env()
    assert want["KRABBY_OMNI_PHASE"] == "2a" and want["KRABBY_PLANT"] == "A15+B"
    assert want["KRABBY_OMNI_STAND_OBS"] == "0" and want["KRABBY_RESAMPLE_S"] == "3:12"
    assert want["KRABBY_OMNI_CMD_POOL"] == SPEC.POOL and want["KRABBY_OMNI_SLOTS"] == SPEC.SLOTS
    for k in SPEC.FORBIDDEN_ENV + ("KRABBY_PHASE", "KRABBY_SIDE_PHASE"):
        assert k not in want
    base = {"PATH": "/usr/bin", "KRABBY_PHASE": "x", "KRABBY_RESTANCE": D3, "P2R_MODE": "st2", "SN_STAND_CKPT": "/a",
            "HOME": "/h"}
    env = SPEC.train_env(base)
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/h" and env["OMNI_KIT_ACCEPT_EULA"] == "yes"
    assert env["PYTHONUNBUFFERED"] == "1" and env["TERM"] == "xterm"
    assert {k: v for k, v in env.items() if k.startswith("KRABBY_")} == want
    assert not any(k.startswith(("P2R_", "SN_STAND_")) for k in env)
    assert SPEC.env_reasons(env) == []


@pytest.mark.parametrize("patch, needle", [
    ({"KRABBY_OMNI_STAND_OBS": "clock"}, "STAND_OBS"),
    ({"KRABBY_OMNI_STAND_OBS": None}, "STAND_OBS"),
    ({"KRABBY_RSI_FRAC": None}, "RSI_FRAC"),                                  # T2: the preset 0.2 would apply
    ({"KRABBY_PHASEOUT": None}, "PHASEOUT"),                                  # T2: the empty string matters
    ({"KRABBY_OMNI_SLOTS": None}, "SLOTS"),                                   # T3
    ({"KRABBY_OMNI_CMD_POOL": "stand:0.15,fwd:0.85"}, "CMD_POOL"),
    ({"KRABBY_RESTANCE": D3}, "KRABBY_RESTANCE"),
    ({"KRABBY_OMNI_STAND_MASK": "1"}, "STAND_MASK"),
    ({"KRABBY_HEX_USD_PATH": "/x.usda"}, "USD_PATH"),
    ({"KRABBY_PHASE": "2a"}, "KRABBY_PHASE"),
    ({"P2R_MODE": "st2"}, "P2R_MODE"),
    ({"SN_STAND_MASK": "1"}, "SN_STAND_MASK"),
    ({"OMNI_KIT_ACCEPT_EULA": None}, "EULA"),
])
def test_env_reasons_refuse(SPEC, patch, needle):
    env = SPEC.train_env({"PATH": "/usr/bin"})
    for k, v in patch.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    r = SPEC.env_reasons(env)
    assert r and any(needle in x for x in r), r


def test_eval_carry_matches_run_phases(SPEC):
    src = RUN_PHASES.read_text()
    ns: dict = {}
    for name in ("EVAL_CARRY", "EVAL_CARRY_DEFAULTS"):
        start = src.index(f"\n{name} = ") + 1
        depth, i = 0, src.index("=", start) + 1
        while True:                                               # the literal up to its balanced closing bracket
            ch = src[i]
            depth += ch in "({[" and 1 or 0
            depth -= ch in ")}]" and 1 or 0
            i += 1
            if depth == 0 and ch in ")}]":
                break
        ns[name] = eval(src[src.index("=", start) + 1:i], {})     # noqa: S307 -- a literal from the repo's own file
    # NOTE(robot-frame velocity commands campaign step 2, 2026-10-15): run_phases grew exactly the three
    # command-frame keys after this (closed, world-frame) campaign froze its copy; the frame's off value
    # ("world") is what an absent key builds, so the frozen copy is run_phases' carry minus those three.
    frame = ("KRABBY_OMNI_COMMAND_FRAME", "KRABBY_OMNI_BODY_ENVELOPE", "KRABBY_OMNI_BODY_CAP")
    assert tuple(ns["EVAL_CARRY"][-3:]) == frame
    assert tuple(k for k in ns["EVAL_CARRY"] if k not in frame) == SPEC.EVAL_CARRY
    assert {k: v for k, v in ns["EVAL_CARRY_DEFAULTS"].items() if k not in frame} == SPEC.EVAL_CARRY_DEFAULTS
    assert ns["EVAL_CARRY_DEFAULTS"]["KRABBY_OMNI_COMMAND_FRAME"] == "world"


def _effective_trainer_env(SPEC):
    """The trainer's env after the preset's import-time ``setdefault`` (phase_env, loaded by path: pure python)."""
    ph = _load(TASK / "config/crab_hex/crab_hex_omni_phases.py", "_t_sn_dagger_phases")
    env = SPEC.train_env({})
    for k, v in ph.phase_env("2a", "A15+B").items():
        env.setdefault(k, v)
    return env


@pytest.mark.skipif(not (SC.REFS_SUITE1 / "run_meta.json").is_file(), reason="REFS env stamp not on this machine")
def test_obs_parity_with_the_refs_stamp(SPEC):
    refs = json.loads((SC.REFS_SUITE1 / "run_meta.json").read_text())["env_vars"]
    env = _effective_trainer_env(SPEC)
    assert "KRABBY_HEX_USD_PATH" not in env                       # A15+B is the main asset (never exported)
    assert env["KRABBY_RESAMPLE_S"] == "3:12" and env["KRABBY_OMNI_FACING_OBS_CLIP"] == "0.6"
    assert SPEC.obs_parity_reasons(env, refs) == []
    sn = _load(SCRIPTS / "sn_common.py", "_t_sn_dagger_sn")
    assert sn.env_reasons(env, env4=True) == []                   # ENV4 after the expansion
    for k, v in (("KRABBY_OMNI_CLOCK_HIDE", "0"), ("KRABBY_OMNI_SPEED_ELLIPSE", "0.35:0.25:0.25"),
                 ("KRABBY_OMNI_FACING_OBS_ENCODING", "sincos"), ("KRABBY_OMNI_STAND_PROXY", "hold"),
                 ("KRABBY_OMNI_STAND_OBS", "clock"), ("KRABBY_STD_PARAM", "scalar")):
        bad = dict(env, **{k: v})
        assert SPEC.obs_parity_reasons(bad, refs), k


# ===================================================================================== cfg asserts (duck-typed)
def _ns(**kw):
    return types.SimpleNamespace(**kw)


def good_cfgs(SPEC):
    env_cfg = _ns(
        commands=_ns(base_velocity=_ns(resampling_time_range=(3.0, 12.0), cmd_pool=SPEC.POOL,
                                       slot_probs=(0.85, 0.55, 0.1, 0.2), dir_band=None,
                                       speed_ellipse=(0.35, 0.35, 0.25))),
        episode_length_s=40.0,
        parkours=_ns(base_parkour=_ns(move_up_frac=0.7, move_down_frac=0.4, freeze_terrain_levels=False)),
        terminations=_ns(crab_failure=_ns(params={"limit_angle": 0.5, "contact_force_threshold": 500.0,
                                                  "minimum_root_height_z": None})),
        observations=_ns(policy=_ns(enable_corruption=False, extreme_parkour_observations=_ns(
            params={"clock_hide": True, "facing_obs_clip": 0.6, "facing_obs_encoding": "raw"}))),
        actions=_ns(joint_pos=_ns(action_delay_steps=[1, 1], use_delay=True)),
        events=_ns())
    agent = _ns(clip_actions=1.0, policy=_ns(noise_std_type="soft", init_noise_std=0.8))
    return env_cfg, agent


def test_cfg_reasons_pass_and_refuse(SPEC):
    env_cfg, agent = good_cfgs(SPEC)
    assert SPEC.cfg_reasons(env_cfg, agent) == []
    cases = [
        (lambda e, a: setattr(e.commands.base_velocity, "resampling_time_range", (10.0, 10.0)), "resampling"),
        (lambda e, a: setattr(e.commands.base_velocity, "slot_probs", (0.15, 0.55, 0.1, 0.2)), "slot_probs"),
        (lambda e, a: setattr(e.commands.base_velocity, "eval_slot_latch", True), "eval_slot_latch"),
        (lambda e, a: setattr(e, "episode_length_s", 20.0), "episode_length_s"),
        (lambda e, a: e.terminations.crab_failure.params.update(limit_angle=0.7), "limit_angle"),
        (lambda e, a: e.observations.policy.extreme_parkour_observations.params.update(clock_hide=False),
         "clock_hide"),
        (lambda e, a: e.observations.policy.extreme_parkour_observations.params.update(stand_mask=True),
         "stand_mask"),
        (lambda e, a: e.observations.policy.extreme_parkour_observations.params.update(stand_bit_clock=True),
         "stand_bit_clock"),
        (lambda e, a: setattr(e.observations.policy, "enable_corruption", True), "corruption"),
        (lambda e, a: setattr(e.actions.joint_pos, "action_delay_steps", [1, 2]), "delay"),
        (lambda e, a: setattr(e.events, "rsi_reference_reset", object()), "rsi"),
        (lambda e, a: setattr(a, "clip_actions", 4.8), "clip_actions"),
        (lambda e, a: setattr(a.policy, "noise_std_type", "scalar"), "noise_std_type"),
        (lambda e, a: setattr(e.parkours.base_parkour, "freeze_terrain_levels", True), "freeze"),
    ]
    for fn, needle in cases:
        e, a = good_cfgs(SPEC)
        fn(e, a)
        r = SPEC.cfg_reasons(e, a)
        assert r and any(needle in x for x in r), (needle, r)


def test_network_cfg_reasons(SPEC, s1_cfg):
    assert SPEC.network_cfg_reasons(copy.deepcopy(s1_cfg), s1_cfg) == []
    bad = copy.deepcopy(s1_cfg)
    bad["policy"]["actor_hidden_dims"] = [256, 256, 128]
    assert SPEC.network_cfg_reasons(bad, s1_cfg)
    bad = copy.deepcopy(s1_cfg)
    bad["estimator"]["hidden_dims"] = [256, 64]
    assert SPEC.network_cfg_reasons(bad, s1_cfg)
    tup = copy.deepcopy(s1_cfg)
    tup["policy"]["actor_hidden_dims"] = tuple(tup["policy"]["actor_hidden_dims"])      # tuples == lists
    assert SPEC.network_cfg_reasons(tup, s1_cfg) == []


# ===================================================================================== schedules + generators
def test_beta_schedule(SPEC):
    T = 3000
    assert SPEC.beta(0, T) == 1.0 and SPEC.beta(500, T) == pytest.approx(0.5) and SPEC.beta(999, T) > 0
    assert SPEC.beta(1000, T) == 0.0 and SPEC.beta(2999, T) == 0.0
    g = [SPEC.beta(i, T) for i in range(T)]
    assert all(a >= b for a, b in zip(g, g[1:]))
    assert SPEC.beta(0, 1) == 1.0
    with pytest.raises(ValueError):
        SPEC.beta(0, 0)


def test_sigma_and_assignment_tables(SPEC):
    assert sum(p for _, p in SPEC.SIGMA_TABLE) == pytest.approx(1.0)
    assert [s for s, _ in SPEC.SIGMA_TABLE] == [0.0, 0.1, 0.3]
    us = [i / 1000 for i in range(1000)]
    sig = [SPEC.sigma_of(u) for u in us]
    assert sig.count(0.0) == 500 and sig.count(0.1) == 250 and sig.count(0.3) == 250
    ks = [SPEC.assign_of(u) for u in us]
    assert ks.count(0) == 500 and all(abs(ks.count(k) - 1000 / 6) <= 1 for k in (1, 2, 3))
    assert SPEC.CONTROL_SHARE == 0.10 and SPEC.ASSIGNS == ("control", "statue", "chatter", "stuck_lift")


def test_private_generators_never_touch_the_global_rng(CORE):
    torch.manual_seed(123)
    np.random.seed(5)
    random.seed(7)
    before = CORE.save_rng()
    g = CORE.make_generators(18, "cpu")
    a = torch.rand(5, generator=g["episode"])
    b = torch.randn(3, 18, generator=g["dart"])
    buf = CORE.Reservoir(4, p1=P1, hist_start=399, device="cpu", gen=g["reservoir"])
    buf.add(torch.randn(9, W, generator=g["dart"]), torch.zeros(9, 18), torch.zeros(9, 6, dtype=torch.long),
            torch.zeros(9, dtype=torch.bool))
    buf.sample(16, 0.1, g["minibatch"])
    tr = CORE.StandTracker(4, "cpu", dt=0.02, max_ep=2000, mode="recover_probe", gen=g["a1"])
    tr.pre(torch.ones(4, dtype=torch.bool), torch.zeros(4, W), torch.full((4,), 5.0), torch.ones(4),
           torch.zeros(4, dtype=torch.long), 0)
    assert CORE.rng_equal(before, CORE.save_rng())
    g2 = CORE.make_generators(18, "cpu")
    assert torch.equal(torch.rand(5, generator=g2["episode"]), a)               # seeded: replays
    assert not torch.equal(torch.rand(5, generator=CORE.make_generators(19, "cpu")["episode"]), a)
    assert not torch.equal(torch.rand(5, generator=g2["dart"]), torch.rand(5, generator=g2["episode"]))
    assert set(g) == set(CORE.GEN_NAMES)


def test_build_networks_restores_rng_and_freezes(CORE, nets_factory):
    torch.manual_seed(4)
    before = CORE.save_rng()
    nets = nets_factory()
    assert CORE.rng_equal(before, CORE.save_rng())
    assert (nets.p0, nets.p1, nets.num_prop, nets.num_hist, nets.hist_start) == (P0, P1, NP, 10, 399)
    train = [n for n, p in nets.student.named_parameters() if p.requires_grad]
    assert train and all(n.startswith("actor.actor_backbone.") for n in train)              # OPEN-1
    assert not any(p.requires_grad for p in nets.s1.parameters())
    assert not any(p.requires_grad for p in nets.est.parameters())
    assert nets.s1_sha.startswith(SC.S1_SHA)
    ok, why = CORE.SN.state_dicts_equal(nets.student.state_dict(), nets.s1.state_dict())
    assert ok, why                                                           # warm start == S1
    opt = CORE.make_optimizer(nets.student, 3e-4)
    assert len(opt.param_groups) == 1 and len(opt.param_groups[0]["params"]) == len(list(nets.s1.parameters()))


# ===================================================================================== labels + the D1 forward path
def test_r_labels_are_restance_target_with_d3(CORE, RST):
    obs = obs_batch(64, seed=3)
    t, b = CORE.SN.r_label(obs)
    t2, b2 = RST.restance_target(obs, RST.parse_restance(D3))
    assert torch.equal(t, t2) and torch.equal(b, b2)
    td, _ = RST.restance_target(obs, RST.RestanceParams())
    assert not torch.equal(td, t)                                            # T4: "default" is another R
    assert CORE.SN.RESTANCE_D3 == D3


class Spy:
    """A stand policy that records the input it is handed (and delegates to the real network)."""

    def __init__(self, pol):
        self.pol, self.seen = pol, []

    def act_inference(self, x, hist_encoding=False, **kw):
        self.seen.append((bool(hist_encoding), x.clone()))
        return self.pol.act_inference(x, hist_encoding=hist_encoding, **kw)

    def __getattr__(self, k):
        return getattr(self.pol, k)


def perturbed_student(nets, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in nets.student.named_parameters():
            if name.startswith("actor.actor_backbone.6"):
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    return nets


def trained_ckpt(path, mask: int) -> str:
    """A file carrying only what ``StandNetMode.trained_arm`` reads: a trainer checkpoint's ``infos.sn_dagger``."""
    torch.save({"infos": {"sn_dagger": {"mode": "dagger", "arm": {1: "mask", 0: "full"}[mask], "stand_mask": mask}}},
               str(path))
    return str(path)


@pytest.mark.parametrize("mask", [0, 1])
def test_forward_path_equals_the_launchers_standnetmode(CORE, PAR, nets_factory, mask, tmp_path):
    """T1 / map 2.7 item 3 / step-6 forward parity on CPU: the trainer's routed action == the stand_net launcher's
    REAL StandNetMode (private P2L instance) on the same batch, bitwise; the stand policy's input == the trainer's
    ``stand_input`` (the mask applied after the estimate)."""
    nets = perturbed_student(nets_factory())
    obs = obs_batch(48, seed=11)
    stand = torch.arange(48) % 3 != 1
    with torch.inference_mode():
        a_tr = CORE.trainer_actions(obs, stand, nets, mask)
        x_tr = CORE.stand_input(obs, nets, mask)
    spy = Spy(nets.student)
    ck = trained_ckpt(tmp_path / "ck.pt", mask)
    a_mode = PAR.launcher_mode_actions(obs, stand, nets.s1, nets.est, spy, nets.est, p0=P0, p1=P1, num_prop=NP,
                                       mask=mask, ckpt=ck)
    assert torch.equal(a_tr, a_mode)
    assert len(spy.seen) == 1 and spy.seen[0][0] is True and torch.equal(spy.seen[0][1], x_tr)
    with torch.inference_mode():
        a_l = CORE.launcher_actions(obs, stand, nets.s1, nets.est, nets.student, nets.est, p0=P0, p1=P1,
                                    num_prop=NP, mask=mask)
        a_s1 = CORE.s1_actions(obs, nets)
    assert torch.equal(a_l, a_tr)
    assert torch.equal(a_tr[~stand], a_s1[~stand]) and not torch.equal(a_tr[stand], a_s1[stand])
    if mask:
        rst = CORE.SN.load_rst()
        h = obs.clone()
        h[:, P0:P1] = nets.est.inference(h[:, :NP])
        assert torch.equal(x_tr, rst.stand_mask_obs(h))                     # estimate first, then mask
    # the parity replay applies the eval's trained-arm check: a checkpoint of the OTHER arm is refused
    with pytest.raises(SystemExit, match="trained with stand_mask"):
        PAR.launcher_mode_actions(obs, stand, nets.s1, nets.est, spy, nets.est, p0=P0, p1=P1, num_prop=NP,
                                  mask=mask, ckpt=trained_ckpt(tmp_path / "other.pt", 1 - mask))


def test_mse_forward_on_compacted_rows_is_the_rollout_forward(CORE, nets_factory):
    """T1 (the fit half): the reservoir stores the actor input compacted to the columns the hist path reads; the
    training forward on the re-expanded rows == the rollout forward on the full rows, bitwise (same batch)."""
    nets = perturbed_student(nets_factory())
    obs = obs_batch(40, seed=5)
    buf = CORE.Reservoir(40, p1=nets.p1, hist_start=nets.hist_start, device="cpu",
                         gen=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        x = CORE.stand_input(obs, nets, 1)
        mu_roll = CORE.student_mean(nets.student, x)
        tgt, br = CORE.SN.r_label(obs)
        buf.add(x, tgt, br, torch.zeros(40, dtype=torch.bool))
    assert buf.size == 40                                                    # fill phase: slot == position
    xe = buf.expand(torch.arange(40))
    assert torch.equal(xe[:, :P1], x[:, :P1]) and torch.equal(xe[:, 399:], x[:, 399:])
    assert not xe[:, P1:399].any()
    mu_fit = CORE.student_mean(nets.student, xe)                             # grad mode, as in training
    assert torch.equal(mu_fit.detach(), mu_roll)
    assert torch.equal(buf.tgt[:40], tgt) and torch.equal(buf.branch[:40].long(), br)
    loss, _ = CORE.dagger_loss(nets.student, xe, buf.tgt[:40], torch.ones(40, 18))
    loss.backward()
    grads = {n for n, p in nets.student.named_parameters() if p.grad is not None}
    assert grads and all(n.startswith("actor.actor_backbone.") for n in grads)


# ===================================================================================== spoilers
def test_spoiler_actions(CORE):
    p = CORE.SN.r_params()
    n = 8
    obs = obs_batch(n, seed=9)
    tgt, _ = CORE.SN.r_label(obs)
    noise = torch.randn(n, 18, generator=torch.Generator().manual_seed(2))
    kind = torch.tensor([0, 1, 2, 3, 3, 3, 1, 0])
    leg = torch.tensor([0, 1, 2, 0, 3, 5, 4, 2])
    sign = torch.tensor([1., -1., 1., 1., -1., 1., 1., -1.])
    a = CORE.spoiler_actions(kind, tgt, obs, leg, sign, noise, p)
    assert torch.equal(a[0], tgt[0]) and torch.equal(a[7], tgt[7])           # control: R
    assert not a[1].any() and not a[6].any()                                  # statue: exactly 0
    assert torch.equal(a[2], tgt[2] + 0.3 * noise[2])                         # chatter
    side = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    for r in (3, 4, 5):
        l, s = int(leg[r]), float(sign[r])
        c = float(obs[r, 18 + 2 * l])
        assert float(a[r, 6 + l]) == -0.5                                     # hip at R's lift
        assert float(a[r, 12 + l]) == pytest.approx(-1.0 * side[l] * -0.5)  # knee = -rho * side * hip
        assert float(a[r, l]) == pytest.approx(max(-1.0, min(1.0, -2.0 * (c - s * 0.4))), abs=1e-6)
        others = [j for j in range(18) if j not in (l, 6 + l, 12 + l)]
        assert torch.equal(a[r, others], tgt[r, others])                     # the other legs per R
    # the saturated P-law drives the cam toward s * 0.4 (velocity sign)
    o2 = obs[3:4].clone()
    o2[0, 18] = 0.0
    a2 = CORE.spoiler_actions(torch.tensor([3]), tgt[3:4], o2, torch.tensor([0]), torch.tensor([1.0]), noise[3:4], p)
    assert float(a2[0, 0]) == pytest.approx(0.8)


# ===================================================================================== onsets, weights, reservoir
def test_lift_onset(CORE):
    L, H, Wt = 2, 0, 1
    prev = torch.tensor([[H] * 6, [L] + [H] * 5, [Wt] * 6, [H] * 6])
    cur = torch.tensor([[L] + [H] * 5, [L] + [H] * 5, [Wt, L, Wt, Wt, Wt, Wt], [L] * 6])
    valid = torch.tensor([True, True, True, False])
    assert CORE.lift_onset(cur, prev, valid).tolist() == [True, False, True, False]


def test_branch_weight_table(CORE):
    # leg 0: 90 % HOLD, 10 % LIFT; leg 1: 4 branches; legs 2..5: HOLD only
    n = 1000
    br = torch.zeros(n, 6, dtype=torch.int8)
    br[:100, 0] = 2
    br[:, 1] = torch.tensor([0, 1, 3, 4] * 250, dtype=torch.int8)
    br[:3, 2] = 4                                                            # very rare: capped
    t = CORE.branch_weight_table(br, w_max=20.0)
    assert t.shape == (6, 5)
    assert float(t[0, 0]) == pytest.approx(1 / (2 * 0.9)) and float(t[0, 2]) == pytest.approx(1 / (2 * 0.1))
    assert float(t[0, 1]) == 0.0 and float(t[0, 3]) == 0.0                     # absent branches
    f0 = torch.tensor([0.9, 0, 0.1, 0, 0])
    assert float((f0 * t[0]).sum()) == pytest.approx(1.0)                    # sum_b f w = 1 (uncapped)
    assert torch.allclose(t[1], torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert float(t[2, 4]) == 20.0                                             # 1 / (2 * 0.003) = 167 -> cap
    assert float(t[3, 0]) == 1.0
    w = CORE.element_weights(t, br[:5].long())
    assert w.shape == (5, 18)
    for col in range(18):
        leg = CORE.LEG_OF_COL[col]
        assert torch.equal(w[:, col], t[leg, br[:5, leg].long()])
    assert CORE.LEG_OF_COL[:6] == (0, 1, 2, 3, 4, 5) and CORE.LEG_OF_COL[6] == 0 and CORE.LEG_OF_COL[17] == 5


def test_reservoir_cap_dedupe_and_sampling(CORE):
    g = torch.Generator().manual_seed(1)
    buf = CORE.Reservoir(50, p1=P1, hist_start=399, device="cpu", gen=g)
    assert buf.n_keep == 216 + 750 and CORE.Reservoir.row_bytes(966) == 4 * 966 + 72 + 6 + 1
    assert buf.nbytes == 50 * CORE.Reservoir.row_bytes(966)
    seen_ids = []
    for step in range(40):
        k = 7
        ids = torch.arange(step * k, step * k + k, dtype=torch.float32)
        x = ids[:, None].expand(k, W).clone()
        onset = (ids.long() % 10) == 0
        buf.add(x, torch.zeros(k, 18), torch.zeros(k, 6, dtype=torch.long), onset)
        seen_ids += ids.long().tolist()
        assert buf.size == min(50, buf.seen)
    assert buf.seen == 280 and buf.size == 50
    stored = buf.x[:, 0].long().tolist()
    assert len(set(stored)) == 50                                             # no slot holds a duplicate row
    assert all(0 <= s < 280 for s in stored)
    assert torch.equal(buf.onset, (buf.x[:, 0].long() % 10) == 0)             # flags stay with their rows
    assert sum(1 for s in stored if s >= 50) > 25                            # the replacement phase ran
    idx = buf.sample(1000, 0.25, torch.Generator().manual_seed(2))
    share = float(buf.onset[idx].float().mean())
    assert share >= 0.25 and idx.max() < 50
    none = CORE.Reservoir(10, p1=P1, hist_start=399, device="cpu", gen=torch.Generator().manual_seed(3))
    none.add(torch.zeros(5, W), torch.zeros(5, 18), torch.zeros(5, 6, dtype=torch.long), torch.zeros(5, dtype=torch.bool))
    assert float(none.onset[none.sample(100, 0.5, torch.Generator().manual_seed(4))].float().mean()) == 0.0
    # deterministic given the generator seed (same-seed runs replay)
    b2 = CORE.Reservoir(50, p1=P1, hist_start=399, device="cpu", gen=torch.Generator().manual_seed(1))
    for step in range(40):
        ids = torch.arange(step * 7, step * 7 + 7, dtype=torch.float32)
        b2.add(ids[:, None].expand(7, W).clone(), torch.zeros(7, 18), torch.zeros(7, 6, dtype=torch.long),
               (ids.long() % 10) == 0)
    assert torch.equal(b2.x, buf.x)
    with pytest.raises(ValueError):
        CORE.Reservoir(10, p1=400, hist_start=399, device="cpu", gen=g)
    assert CORE.Reservoir.rows_for(4.0, 966) * CORE.Reservoir.row_bytes(966) <= 4e9


def test_reservoir_in_inference_mode_trains_outside(CORE):
    buf = CORE.Reservoir(8, p1=P1, hist_start=399, device="cpu", gen=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        x = torch.randn(6, W)
        buf.add(x, torch.randn(6, 18), torch.zeros(6, 6, dtype=torch.long), torch.zeros(6, dtype=torch.bool))
    lin = torch.nn.Linear(W, 18)
    xe = buf.expand(torch.arange(6))
    ((lin(xe) - buf.tgt[:6]) ** 2).mean().backward()
    assert lin.weight.grad is not None


# ===================================================================================== the stand tracker (synthetic)
class Trace:
    """Drives a StandTracker with hand-built rows (one env per column)."""

    def __init__(self, CORE, n=3, mode="dagger", dt=0.02, max_ep=2000, gen=None, xf=None):
        self.CORE = CORE
        self.t = CORE.StandTracker(n, "cpu", dt=dt, max_ep=max_ep, mode=mode, gen=gen, r_params=CORE.SN.r_params(),
                                   exit_window_steps=xf)
        self.n = n
        self.final = []

    def obs(self, c):
        o = torch.zeros(self.n, W)
        o[:, [18 + 2 * i for i in range(6)]] = torch.as_tensor(c, dtype=torch.float32)
        return o

    def step(self, stand, c, time_left, counter, ep_len, it, *, dones, fell, stand_next, kind=None, sigma=None):
        n = self.n
        kind = torch.zeros(n, dtype=torch.long) if kind is None else torch.as_tensor(kind)
        sigma = torch.zeros(n) if sigma is None else torch.as_tensor(sigma, dtype=torch.float32)
        e, v = self.t.pre(torch.as_tensor(stand), self.obs(c), torch.as_tensor(time_left, dtype=torch.float32),
                          torch.as_tensor(counter), torch.as_tensor(ep_len), it, kind, sigma)
        self.final += self.t.post(torch.as_tensor(dones), torch.as_tensor(fell), torch.as_tensor(stand_next), it)
        return e, v


def test_tracker_exit_reads_the_last_stand_row_and_schedules(CORE):
    tr = Trace(CORE, n=1, dt=0.02, max_ep=2000)
    c_home = [[0.05] * 6]
    c_off = [[0.25] + [0.05] * 5]
    # entry at ep 100 with time_left 3.0 s (150 rows: 100..249); rows 100..249 stand, exit row 250
    for k in range(150):
        ep = 100 + k
        last = k == 149
        tr.step([True], c_off if k < 149 else c_home, [3.0 - 0.02 * k], [7], [ep], it=k // 24,
                 dones=[False], fell=[False], stand_next=[not last], kind=[0], sigma=[0.0])
    assert tr.final == [] and len(tr.t.pending) == 1                           # the exit waits for its window
    rec = tr.t.pending[0]
    assert rec["entry_ep"] == 100 and rec["last_row_ep"] == 249 and rec["end_ep"] == 250
    assert rec["sched_end_ep"] == 250 and rec["sched_len_s"] == pytest.approx(3.0) and not rec["cut"]
    assert rec["end"] == "exit" and rec["homed"] is True and rec["max_abs_c"] == pytest.approx(0.05)
    assert rec["kind"] == "student" and rec["entry_kind"] == "handover" and rec["prev_kind"] is None
    # 100 clean steps close the window
    for k in range(100):
        tr.step([False], c_off, [5.0], [8], [251 + k], it=10, dones=[False], fell=[False], stand_next=[False])
    assert len(tr.final) == 1 and tr.final[0]["exit_fall"] is False and tr.final[0]["exit_censored"] is False


def test_tracker_never_reads_a_post_reset_row_and_scores_falls(CORE):
    tr = Trace(CORE, n=2, max_ep=2000)
    # env 0 falls on its 3rd stand row; env 1 times out (cut). The obs AFTER the done (a fresh episode) has cams 0.
    c_bad = [[0.5] * 6, [0.5] * 6]
    for k in range(3):
        last = k == 2
        tr.step([True, True], c_bad, [9.0, 9.0], [1, 1], [10 + k, 1990 + k], it=0,
                dones=[last, last], fell=[last, False], stand_next=[True, True])
    # the next rows are post-reset rows (ep 0) with homed cams: they belong to a NEW stand
    tr.step([True, True], [[0.0] * 6, [0.0] * 6], [9.0, 9.0], [2, 2], [0, 0], it=0,
            dones=[False, False], fell=[False, False], stand_next=[True, True])
    recs = {r["env"]: r for r in tr.final}
    assert recs[0]["end"] == "fall" and recs[0]["homed"] is False and recs[0]["max_abs_c"] == pytest.approx(0.5)
    assert recs[0]["last_row_ep"] == 12 and recs[0]["sched_end_ep"] == 10 + 450
    assert recs[1]["end"] == "timeout" and recs[1]["cut"] is True              # 1990 + 450 >= 2000
    assert all(r["exit_fall"] is None for r in tr.final)
    assert bool(tr.t.in_stand.all()) and tr.t.entry_ep.tolist() == [0, 0]     # the new stands started at the reset
    assert tr.t.entry_kind.tolist() == [0, 0]                                 # spawn entries
    sc = [r for r in tr.final if CORE.SPEC.scored(r, 8.0)]
    assert [r["env"] for r in sc] == [0]                                       # the fall scores; the cut one does not
    assert recs[0]["fall_cause"] == "crab_failure" and recs[1]["fall_cause"] is None


def test_tracker_other_termination_is_a_fall_not_a_cut(CORE):
    """total_terminates also ends episodes at |roll| / |pitch| > 1.5 rad or a height floor (time_out=True in the cfg):
    only the episode-length timeout is a cut; any other done counts as a fall (conservative)."""
    t = CORE.StandTracker(2, "cpu", dt=0.02, max_ep=2000, mode="dagger", r_params=CORE.SN.r_params())
    obs = torch.zeros(2, W)
    t.pre(torch.ones(2, dtype=torch.bool), obs, torch.full((2,), 9.0), torch.ones(2), torch.tensor([50, 1999]), 0,
          torch.zeros(2, dtype=torch.long), torch.zeros(2))
    out = t.post(torch.tensor([True, True]), torch.tensor([False, False]), torch.tensor([True, True]), 0,
                 timeout=torch.tensor([False, True]))
    recs = {r["env"]: r for r in out}
    assert recs[0]["end"] == "fall" and recs[0]["fall_cause"] == "termination" and recs[0]["homed"] is False
    assert recs[1]["end"] == "timeout"


def test_tracker_merge_extends_and_exit_fall_window(CORE):
    tr = Trace(CORE, n=1, dt=0.1, max_ep=300, xf=20)
    c = [[0.0] * 6]
    # entry ep 5, time_left 1.0 (10 rows) ; at ep 15 a stand->stand resample (counter 3 -> 4) with time_left 2.0
    for k in range(10):
        tr.step([True], c, [1.0 - 0.1 * k], [3], [5 + k], it=0, dones=[False], fell=[False], stand_next=[True])
    for k in range(20):
        last = k == 19
        tr.step([True], c, [2.0 - 0.1 * k], [4], [15 + k], it=1, dones=[False], fell=[False], stand_next=[not last])
    rec = tr.t.pending[0]
    assert rec["sched_end_ep"] == 35 and rec["sched_len_s"] == pytest.approx(3.0) and rec["n_merges"] == 1
    assert rec["end_ep"] == 35 and rec["length_s"] == pytest.approx(3.0)
    # a fall 15 steps after the exit (inside the 20-step window) -> exit_fall True
    for k in range(15):
        f = k == 14
        tr.step([False], c, [5.0], [5], [35 + k], it=2, dones=[f], fell=[f], stand_next=[False])
    assert len(tr.final) == 1 and tr.final[0]["exit_fall"] is True and tr.final[0]["exit_censored"] is False
    # a timeout inside the window -> censored
    tr2 = Trace(CORE, n=1, dt=0.1, max_ep=300, xf=20)
    for k in range(5):
        tr2.step([True], c, [0.5 - 0.1 * k], [1], [k + 1], it=0, dones=[False], fell=[False], stand_next=[k < 4])
    for k in range(3):
        tr2.step([False], c, [5.0], [2], [6 + k], it=0, dones=[k == 2], fell=[False], stand_next=[False])
    assert tr2.final[0]["exit_fall"] is False and tr2.final[0]["exit_censored"] is True


def test_tracker_reset_entry_schedule_and_prev_kind(CORE):
    tr = Trace(CORE, n=1, dt=0.02, max_ep=2000)
    c = [[0.0] * 6]
    # a reset entry: time_left = draw - dt = 2.98 -> 149 rows (0..148)
    for k in range(149):
        tr.step([True], c, [2.98 - 0.02 * k], [1], [k], it=0, dones=[False], fell=[False], stand_next=[k < 148],
                kind=[2])
    r1 = tr.t.pending[0]
    assert r1["sched_end_ep"] == 149 and r1["end_ep"] == 149 and r1["kind"] == "control"
    assert r1["entry_kind"] == "spawn"
    for k in range(200):                                                     # walk (closes the window clean)
        tr.step([False], c, [5.0], [2], [149 + k], it=1, dones=[False], fell=[False], stand_next=[False])
    for k in range(3):
        tr.step([True], c, [3.0], [3], [349 + k], it=2, dones=[k == 2], fell=[k == 2], stand_next=[True], kind=[0])
    r2 = [r for r in tr.final if r["entry_ep"] == 349][0]
    assert r2["prev_kind"] == "control" and r2["kind"] == "student" and r2["end"] == "fall"
    assert int(tr.t.prev_kind[0]) == -1                                       # a reset forgets the episode's stands


def test_tracker_a1_assignment_window_and_takeover_capture(CORE):
    g = torch.Generator().manual_seed(0)
    n = 4000
    t = CORE.StandTracker(n, "cpu", dt=0.02, max_ep=2000, mode="recover_probe", gen=g, r_params=CORE.SN.r_params())
    stand = torch.ones(n, dtype=torch.bool)
    obs = torch.zeros(n, W)
    obs[:, 18] = 0.42
    obs[:, 29] = -0.1
    e, _ = t.pre(stand, obs, torch.full((n,), 10.0), torch.ones(n), torch.full((n,), 100), 0)
    assert bool(e.all())
    shares = [float((t.kind == k).float().mean()) for k in range(4)]
    assert abs(shares[0] - 0.5) < 0.03 and all(abs(s - 1 / 6) < 0.03 for s in shares[1:])
    win = t.take_ep - 100
    ctrl = t.kind == 0
    assert bool((win[ctrl] == 0).all()) and int(win[~ctrl].min()) >= 50 and int(win[~ctrl].max()) <= 200
    assert set(t.leg.tolist()) == set(range(6)) and set(t.sign.tolist()) == {-1.0, 1.0}
    assert bool(t.in_window(stand, torch.full((n,), 100))[~ctrl].all())
    assert not bool(t.in_window(stand, torch.full((n,), 100))[ctrl].any())
    # the takeover row records the held leg's hip / cam
    j = int(torch.nonzero(~ctrl & (t.leg == 0))[0])
    ep = torch.full((n,), 100)
    ep[j] = int(t.take_ep[j])
    t.pre(stand, obs, torch.full((n,), 9.0), torch.ones(n), ep, 1)
    assert float(t.take_c[j]) == pytest.approx(0.42) and float(t.take_hip[j]) == pytest.approx(-0.1)
    assert bool(t.reached_take[j])


def _drive_stand(CORE, segments, *, fall_at=None, exit_after=True, entry=100, dt=0.1, mode="dagger"):
    """One env through a stand made of ``segments`` = [(draw_s, cam value), ...] (the 2nd+ are stand->stand merges);
    ``fall_at`` = the row index (from entry) whose step falls; else the stand exits after its last segment. Returns
    the stand's record."""
    tr = Trace(CORE, n=1, dt=dt, max_ep=2000, mode=mode, xf=5)
    k = 0
    for seg, (draw, cam) in enumerate(segments):
        rows = int(round(draw / dt))
        for j in range(rows):
            last_row = seg == len(segments) - 1 and j == rows - 1
            fell = fall_at is not None and k == fall_at
            tr.step([True], [[cam] * 6], [draw - dt * j], [1 + seg], [entry + k], it=0, dones=[fell], fell=[fell],
                    stand_next=[not (last_row and exit_after) or fell], kind=[2])
            k += 1
            if fell:
                break
        if fall_at is not None and k > fall_at:
            break
    recs = list(tr.t.done) + list(tr.t.pending.values())
    assert len(recs) == 1, recs
    return recs[0]


def test_first_draw_scoring_is_outcome_independent(CORE, SPEC):
    """Review 2026-10-12 (blocking; amendment to OPEN-7): a stand's merge extension happens only if it SURVIVED its
    first draw, so scoring on the merged length admitted a short-first-draw stand only when it did not fall. Scoring
    is now on the FIRST draw: two stands with the same first draw are both scored or both excluded, whatever happens
    to them; homed / |c| are read on the first draw's last row."""
    a = _drive_stand(CORE, [(5.0, 0.05)], fall_at=19)                        # 5-s draw, falls at 2 s
    b = _drive_stand(CORE, [(5.0, 0.05), (7.0, 0.3)])                        # 5-s draw, survives, merges, exits
    c = _drive_stand(CORE, [(9.0, 0.05)], fall_at=19)                        # 9-s draw, falls at 2 s
    d = _drive_stand(CORE, [(9.0, 0.05), (3.0, 0.3)])                        # 9-s draw, survives, merges, exits
    assert (a["sched_len0_s"], b["sched_len0_s"], c["sched_len0_s"], d["sched_len0_s"]) == pytest.approx(
        (5.0, 5.0, 9.0, 9.0))
    assert b["sched_len_s"] == pytest.approx(12.0) and b["n_merges"] == 1 and d["sched_len_s"] == pytest.approx(12.0)
    assert a["sched_len_s"] == pytest.approx(5.0)                            # a fall is never extended
    # the OLD rule (merged length) -- the outcome decides inclusion: the 5-s survivor in, the 5-s faller out
    old = [r for r in (a, b, c, d) if r["sched_len_s"] >= 8.0 and not r["cut"] and r["end"] in ("exit", "fall")]
    assert [r is b for r in old].count(True) == 1 and a not in old
    assert SPEC.group_stats(old)["fall"]["p"] == pytest.approx(1 / 3)        # the fall share shrinks
    # the first-draw rule: both 5-s stands out, both 9-s stands in
    new = [r for r in (a, b, c, d) if SPEC.scored(r, SPEC.A1_MIN_SCHED_S)]
    assert len(new) == 2 and new[0] is c and new[1] is d
    st = SPEC.group_stats(new)
    assert st["fall"]["p"] == pytest.approx(0.5) and st["homed"]["p"] == pytest.approx(0.5)
    # outcomes on the first draw: the merged stand is homed at its first draw's end (cams 0.05), not at its own end
    assert c["end0"] == "fall" and c["fall0"] is True and c["homed0"] is False and c["end"] == "fall"
    assert c["r_time0_s"] == pytest.approx(2.0)
    assert d["end0"] == "reached" and d["homed0"] is True and d["max_abs_c0"] == pytest.approx(0.05)
    assert d["homed"] is False and d["max_abs_c"] == pytest.approx(0.3)      # the merged end: reported only
    assert d["sched_end0_ep"] == 190 and d["r_time0_s"] == pytest.approx(9.0) and d["fall0"] is False
    assert SPEC.homed_scored(d) and SPEC.fall_scored(c) and not SPEC.fall_scored(d)
    # the step-7 premise read (4-s floor) scores the 5-s first draws too, on the same footing
    assert SPEC.premise_scored(a) and SPEC.premise_scored(b) and b["homed0"] is True and b["end0"] == "reached"
    # a fall AFTER the first draw's end is not a first-draw fall (the stand reached its scored end)
    e = _drive_stand(CORE, [(9.0, 0.05), (3.0, 0.3)], fall_at=95)
    assert e["end"] == "fall" and e["fall0"] is False and e["end0"] == "reached" and e["homed0"] is True
    assert SPEC.scored(e, SPEC.A1_MIN_SCHED_S) and not SPEC.fall_scored(e)


def test_first_draw_cut_and_run_start_exclusion(CORE, SPEC):
    """``cut0`` = the first draw's scheduled end at / past the episode timeout; a stand in progress at the run's first
    step (``entry_kind`` run_start: spawned at reset, scheduled from the random init episode clock -- review
    2026-10-12) is never scored; A1's excluded counts name both."""
    t = CORE.StandTracker(2, "cpu", dt=0.1, max_ep=300, mode="dagger", r_params=CORE.SN.r_params())
    obs = torch.zeros(2, W)
    t.pre(torch.ones(2, dtype=torch.bool), obs, torch.tensor([9.0, 9.0]), torch.ones(2), torch.tensor([250, 10]), 0,
          torch.zeros(2, dtype=torch.long), torch.zeros(2), first=True)
    out = t.post(torch.tensor([True, True]), torch.tensor([True, True]), torch.tensor([True, True]), 0)
    recs = {r["env"]: r for r in out}
    assert recs[0]["cut0"] is True and recs[1]["cut0"] is False                 # 250 + 90 >= 300
    assert all(r["entry_kind"] == "run_start" and r["end0"] == "fall" for r in out)
    assert not any(SPEC.scored(r, 4.0) for r in out)                           # both excluded (cut0 / run_start)
    ok = rec("control")
    assert SPEC.scored(ok, 8.0) and not SPEC.scored(dict(ok, entry_kind="run_start"), 8.0)
    assert not SPEC.premise_scored(dict(rec("student", mode="dagger", sched=5.0), entry_kind="run_start"))
    r = SPEC.a1_read(a1_records() + [rec("statue", entry_kind="run_start"), rec("control", end0="timeout")])
    assert r["excluded"] == {"short": 0, "cut": 0, "run_start": 1, "timeout_or_other": 1}
    assert r["groups"]["statue"]["n"] == 100 and r["groups"]["control"]["n"] == 100


# ===================================================================================== reads (synthetic records)
def rec(kind, end="exit", homed=True, sched=10.0, cut=False, mode="recover_probe", sigma=0.0, it=5, **kw):
    """A synthetic stand record of an UNMERGED stand (first draw == the whole stand: the first-draw fields mirror the
    merged ones; override either set through ``kw``)."""
    h = homed and end == "exit"
    r = {"mode": mode, "kind": kind, "end": end, "homed": h, "sched_len_s": sched, "cut": cut,
         "sigma": sigma, "it_end": it, "exit_fall": False if end == "exit" else None, "exit_censored": False,
         "r_time_s": 6.0, "fell_before_takeover": False, "n_down": 6, "trim_legs": 0, "prev_kind": None,
         "entry_kind": "handover", "n_merges": 0,
         "sched_len0_s": sched, "cut0": cut, "end0": {"exit": "reached"}.get(end, end), "fall0": end == "fall",
         "homed0": h, "r_time0_s": 6.0, "n_down0": 6, "trim_legs0": 0}
    r.update(kw)
    return r


def a1_records(ctrl_homed=0.9, spoil_homed=(0.85, 0.85, 0.85), spoil_falls=(0, 0, 0), n=100, ctrl_falls=0):
    out = []
    out += [rec("control", homed=i < ctrl_homed * n) for i in range(n - ctrl_falls)]
    out += [rec("control", end="fall") for _ in range(ctrl_falls)]
    for name, h, f in zip(("statue", "chatter", "stuck_lift"), spoil_homed, spoil_falls):
        out += [rec(name, homed=i < h * n) for i in range(n - f)]
        out += [rec(name, end="fall", fell_before_takeover=True) for _ in range(f)]
    return out


def test_a1_pass_rule(SPEC):
    r = SPEC.a1_read(a1_records())
    assert r["verdict"] == "PASS" and all(c["pass"] for c in r["checks"]) and len(r["checks"]) == 7
    assert SPEC.a1_read(a1_records(ctrl_homed=0.70, spoil_homed=(0.7, 0.7, 0.7)))["verdict"] == "STOP"   # floor
    r = SPEC.a1_read(a1_records(ctrl_homed=0.9, spoil_homed=(0.85, 0.71, 0.85)))                         # 0.8 x
    assert r["verdict"] == "STOP" and [c["check"] for c in r["checks"] if not c["pass"]] == [
        "chatter homed >= 0.8 x control"]
    r = SPEC.a1_read(a1_records(spoil_falls=(0, 0, 2)))                                                    # +0.01
    assert r["verdict"] == "STOP" and not [c for c in r["checks"] if "stuck_lift stand-fall" in c["check"]][0]["pass"]
    assert SPEC.a1_read(a1_records(spoil_falls=(1, 0, 0), ctrl_falls=0))["verdict"] == "PASS"            # 0.01 ok
    assert r["groups"]["stuck_lift"]["falls_before_takeover"] == 2
    bad = [x for x in a1_records() if x["kind"] != "chatter"]
    r = SPEC.a1_read(bad)
    assert r["verdict"] == "INVALID" and "chatter: no scored stand" in r["invalid_reasons"]


def test_a1_scoring_filter_and_extras(SPEC):
    recs = a1_records()
    extra = [rec("statue", sched=7.9, homed=False), rec("statue", cut=True, homed=False),
             rec("statue", end="timeout", homed=False), rec("control", mode="dagger", homed=False)]
    r = SPEC.a1_read(recs + extra)
    assert r["verdict"] == "PASS" and r["groups"]["statue"]["n"] == 100
    assert r["excluded"] == {"short": 1, "cut": 1, "run_start": 0, "timeout_or_other": 1}
    recs2 = a1_records()
    for x in recs2[:10]:
        x["prev_kind"] = "statue"
    r2 = SPEC.a1_read(recs2)
    assert r2["control_by_previous_stand"]["statue"]["n"] == 10
    assert set(r2["groups"]["control"]["homed_by_r_time"]) == {"[6,8)"}
    assert SPEC.r_time_bin(0.0) == "[0,2)" and SPEC.r_time_bin(9.0) == "[8,inf)"


def test_premise_read(SPEC):
    recs = ([rec("student", mode="dagger", homed=i < 30, sched=5.0, it=i) for i in range(100)]
            + [rec("student", mode="dagger", homed=True, sched=5.0, sigma=0.3) for _ in range(50)]     # sigma > 0
            + [rec("student", mode="dagger", homed=True, sched=3.0) for _ in range(50)]                # short
            + [rec("control", mode="dagger", homed=i < 90, sched=5.0, it=i) for i in range(100)])
    r = SPEC.premise_read(recs)
    assert r["student"]["n"] == 100 and r["student"]["homed"]["p"] == pytest.approx(0.30)
    assert r["control"]["homed"]["p"] == pytest.approx(0.90)
    assert r["ratio"] == pytest.approx(1 / 3) and r["stop"] is True
    w = SPEC.premise_read(recs, 50, 100)
    assert w["student"]["n"] == 50 and w["student"]["homed"]["p"] == 0.0
    edge = SPEC.premise_read([rec("student", mode="dagger", sched=4.0)] * 5 + [rec("student", mode="dagger",
                                                                                   sched=4.02)] * 3)
    assert edge["student"]["n"] == 3                                         # "longer than 4 s": strict
    ok = SPEC.premise_read([rec("student", mode="dagger", sched=5.0)] * 10 + [rec("control", mode="dagger",
                                                                                  sched=5.0)] * 10)
    assert ok["stop"] is False and ok["ratio"] == 1.0
    assert SPEC.premise_read([])["stop"] is None


def test_smoke_verdict(SPEC):
    good = [{"name": n, "pass": True} for n, _ in SPEC.SMOKE_DAGGER]
    assert SPEC.smoke_verdict("dagger", good)["verdict"] == "PASS"
    assert SPEC.smoke_verdict("dagger", good[1:])["verdict"] == "FAIL"
    assert SPEC.smoke_verdict("dagger", good + [{"name": "x", "pass": True}])["verdict"] == "FAIL"
    assert SPEC.smoke_verdict("dagger", good[:-1] + [{"name": good[-1]["name"], "pass": None}])["verdict"] == "FAIL"
    bad = [dict(c, **{"pass": c["name"] != "forward_parity"}) for c in good]
    v = SPEC.smoke_verdict("dagger", bad)
    assert v["verdict"] == "FAIL" and v["failed"] == ["forward_parity"]
    a1 = [{"name": n, "pass": True} for n, _ in SPEC.SMOKE_A1]
    assert SPEC.smoke_verdict("a1", a1)["verdict"] == "PASS"
    assert len(SPEC.SMOKE_DAGGER) == 13 and len(SPEC.SMOKE_A1) == 10
    assert "terrain_level_recorded" in dict(SPEC.SMOKE_DAGGER) and "terrain_level_recorded" in dict(SPEC.SMOKE_A1)


def test_path_reasons(SPEC):
    assert SPEC.path_reasons("/abs/run_1") == []
    for bad in ("rel/run", "/a b/run", "/a/r*n", "/a/$x", "/a/r;n"):
        assert SPEC.path_reasons(bad), bad


# ===================================================================================== the walker override (Task 4, DESIGN B6)
def _override(**kw):
    base = {"S1_CKPT": "/abs/walker/model_22494.pt", "S1_SHA256": "ab" * 32, "REFS_ENV_STAMP": "/abs/refs/stamp",
            "CANDIDATES": "/abs/run/candidates.yaml", "OVERLAY_CHAIN": "R18_p2a", "REFS_NAME": "REFS_R18_tstar",
            "S1_OVERLAY": dict(_SN_DEFAULT_OVERLAY)}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


_SN_DEFAULT_OVERLAY = yaml.safe_load(CANDIDATES.read_text())["chains"]["P2__S1_s18"]["stages"][0]["overlay"]
_SN_DEFAULT_OVERLAY = {k: ("" if v is None else str(v)) for k, v in _SN_DEFAULT_OVERLAY.items()}


def test_walker_override_unset_is_today(SPEC, monkeypatch):
    monkeypatch.delenv("SN_WALKER_SPEC", raising=False)
    assert SPEC.WALKER_OVERRIDE is None
    assert SPEC.S1_SHA256 == "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"
    assert SPEC.S1_CKPT.name == "model_19994.pt" and "X08_p2a" in str(SPEC.S1_CKPT)
    assert SPEC.S1_AGENT_YAML == SPEC.S1_CKPT.parent / "params" / "agent.yaml"
    assert SPEC.REFS_ENV_STAMP.name == "2026-10-09_03-20-24" and SPEC.REFS_NAME == "REFS_S1_s18"
    assert SPEC.OVERLAY_CHAIN == "P2__S1_s18" and SPEC.CANDIDATES.parent.name == "2026-10-07_omni_phase2"
    assert SPEC.walker_spec_override({}) is None and SPEC.walker_spec_override({"SN_WALKER_SPEC": "  "}) is None


def test_walker_override_set_replaces_the_walker(monkeypatch):
    monkeypatch.setenv("SN_WALKER_SPEC", json.dumps(_override()))
    sp = _load(SCRIPTS / "sn_dagger_spec.py", "_t_sn_dagger_spec_override")
    assert sp.WALKER_OVERRIDE is not None
    assert str(sp.S1_CKPT) == "/abs/walker/model_22494.pt" and sp.S1_SHA256 == "ab" * 32
    assert sp.S1_AGENT_YAML == sp.S1_CKPT.parent / "params" / "agent.yaml"          # the optional key's default
    assert str(sp.REFS_ENV_STAMP) == "/abs/refs/stamp" and sp.REFS_NAME == "REFS_R18_tstar"
    assert sp.OVERLAY_CHAIN == "R18_p2a" and str(sp.CANDIDATES) == "/abs/run/candidates.yaml"
    assert sp.S1_OVERLAY == _SN_DEFAULT_OVERLAY
    # the trainer env is composed from the override's overlay, unchanged otherwise (pool / slots derived from S1's)
    assert sp.expected_krabby_env()["KRABBY_OMNI_CMD_POOL"] == sp.POOL
    # the SH spec inherits it (PRED is loaded fresh, under the same environment)
    sh = _load(TASK / "experiments/2026-10-13_stand_height/scripts/sh_dagger_spec.py", "_t_sh_spec_override")
    assert sh.S1_SHA256 == "ab" * 32 and sh.REFS_NAME == "REFS_R18_tstar" and sh.WALKER_OVERRIDE is not None


@pytest.mark.parametrize("patch,needle", [
    ({"S1_CKPT": "rel/model.pt"}, "absolute"),
    ({"S1_SHA256": "xyz"}, "64 hex"),
    ({"REFS_NAME": None}, "missing"),
    ({"EXTRA": 1}, "unknown"),
    ({"OVERLAY_CHAIN": "a b"}, "OVERLAY_CHAIN"),
    ({"S1_OVERLAY": dict(_SN_DEFAULT_OVERLAY, KRABBY_PAYLOAD_KG="0:90")}, "0 kg"),
    ({"S1_OVERLAY": dict(_SN_DEFAULT_OVERLAY, KRABBY_OMNI_CMD_POOL="stand:1")}, "POOL"),
])
def test_walker_override_refusals(SPEC, patch, needle):
    with pytest.raises(ValueError, match=needle):
        SPEC.walker_spec_override({"SN_WALKER_SPEC": json.dumps(_override(**patch))})
    with pytest.raises(ValueError, match="JSON"):
        SPEC.walker_spec_override({"SN_WALKER_SPEC": "{not json"})


def test_sn_calib_refs_name_follows_the_spec(monkeypatch):
    monkeypatch.delenv("SN_WALKER_SPEC", raising=False)
    old = sys.dont_write_bytecode
    try:
        calib = _load(SCRIPTS / "sn_calib.py", "_t_sn_calib_refs_default")
        assert calib.NAMES["REFS"] == "REFS_S1_s18" and calib.WALKER.WALKER_OVERRIDE is None
        monkeypatch.setenv("SN_WALKER_SPEC", json.dumps(_override()))
        calib2 = _load(SCRIPTS / "sn_calib.py", "_t_sn_calib_refs_override")
        assert calib2.NAMES["REFS"] == "REFS_R18_tstar"
        assert {k: v for k, v in calib2.NAMES.items() if k != "REFS"} == \
            {k: v for k, v in calib.NAMES.items() if k != "REFS"}
    finally:
        sys.dont_write_bytecode = old
