"""Task 4 pipeline step 4 (M1b): the omni depth student, on CPU, without Isaac.

What is pinned here, and why each failure would be silent otherwise:

* **student column invariants** -- the shared step (``scripts/rsl_rl/modules/omni_student_step.py``) never
  writes the facing command (cols 6-7) or the privileged velocity (13-14), writes the constant field pair
  into cols 11-12 AND their ten history copies, and the estimator into 207-215 from the student copy; the
  caller's observation is never modified;
* **GRU reset on done rows** only (and autograd-safe), the 5-step encode cadence, the K = 4 camera order;
* **shared-step call identity** -- the trainer, ``play.py`` and ``play_stream_video.py`` all run the ONE
  ``OmniStudentStep`` (runtime for the trainer, AST for the two Kit scripts), and the deploy export wraps the
  same observation prep (the exported graphs reproduce the sim step bit for bit);
* **forward call identity** -- the forward student paths are unchanged: the forward ``learn_vision`` of this
  tree and of the pre-M1b baseline commit produce bitwise-identical action streams and weights on a fake
  env, and the forward branches of play.py / play_stream_video.py are AST-identical to the baseline's;
* **the camera tripwire** -- composed quaternions point every camera where cameras.yaml says (all three
  camera axes), and the tripwire catches the Euler-yaw edit the sideways task warns about;
* **the runner knob policy** -- only ``KRABBY_STD_PARAM`` / ``KRABBY_INIT_STD`` apply, every other runner knob
  ``apply_exploration_knobs`` reads is refused;
* **the export round trip** -- TorchScript and ONNX (onnxruntime when installed, else the ONNX reference
  evaluator) match torch on a dummy ``(1, 4, 58, 87)`` depth + proprio input, GRU state fed back.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")
try:  # the training stack these modules import (rsl-rl-lib 5.x); CI pins 3.1.2 for compute/parkour, which has neither
    import rsl_rl.extensions  # noqa: E402,F401
    from rsl_rl.modules import EmpiricalNormalization  # noqa: E402,F401
except ImportError:
    pytest.skip("needs the Isaac training stack (rsl-rl-lib 5.x: rsl_rl.extensions, rsl_rl.modules.EmpiricalNormalization);"
                " CI pins rsl-rl-lib 3.1.2 for compute/parkour", allow_module_level=True)
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RSL = REPO / "parkour/scripts/rsl_rl"
MODULES = RSL / "modules"
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
SPEC_PY = OMNI / "config/crab_hex/omni_student_spec.py"
SCENE_PY = OMNI / "config/crab_hex/crab_hex_omni_scene_cfg.py"
ENV_PY = OMNI / "config/crab_hex/crab_hex_omni_env_cfg.py"
INIT_PY = OMNI / "config/crab_hex/__init__.py"
RUNNER_CFG_PY = OMNI / "config/crab_hex/agents/rsl_rl_ppo_cfg.py"
FWD_RUNNER_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/rsl_rl_ppo_cfg.py"
OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
RESTANCE_PY = OMNI / "mdp/crab_hex_omni_restance.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
PLAY_PY = RSL / "play.py"
PSV_PY = RSL / "play_stream_video.py"
RUNNER_PY = MODULES / "on_policy_runner_with_extractor.py"
#: HEAD before the M1b code (2026-10-16): the forward-identity baseline.
BASELINE = "5f0b45b8"

if str(REPO / "parkour") not in sys.path:
    sys.path.insert(0, str(REPO / "parkour"))  # parkour_isaaclab.utils.nonfinite_logging


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pkg(name: str, directory: Path):
    """Import a package directory under a private name (its own ``__init__`` runs)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[name]


PKG = "_t4_rsl_modules"
_load_pkg(PKG, MODULES)
step_mod = importlib.import_module(f"{PKG}.omni_student_step")
export_mod = importlib.import_module(f"{PKG}.omni_student_export")
fe = importlib.import_module(f"{PKG}.feature_extractors")
acm = importlib.import_module(f"{PKG}.actor_critic_with_encoder")
runner_mod = importlib.import_module(f"{PKG}.on_policy_runner_with_extractor")
spec = _load(SPEC_PY, "omni_student_spec__t4_test")

NUM_PROP, NUM_SCAN, NUM_PRIV_EXPLICIT, NUM_PRIV_LATENT, NUM_HIST = 75, 132, 9, 41, 10
EST0, EST1 = NUM_PROP + NUM_SCAN, NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT  # 207, 216
W = 1149


# =============================================================================================== builders
def _actor_cfg() -> dict:
    return dict(class_name="Actor", num_prop=NUM_PROP, num_scan=NUM_SCAN, num_hist=NUM_HIST,
                num_priv_latent=NUM_PRIV_LATENT, num_priv_explicit=NUM_PRIV_EXPLICIT,
                state_history_encoder=dict(class_name="StateHistoryEncoder", channel_size=10))


def _policy_cfg() -> dict:
    return dict(class_name="ActorCriticRMA", init_noise_std=0.8, actor_hidden_dims=[64, 32],
                critic_hidden_dims=[32], activation="elu", scan_encoder_dims=[128, 64, 32],
                priv_encoder_dims=[64, 20], tanh_encoder_output=False, actor=_actor_cfg())


def _nets(seed: int = 0):
    """``(teacher policy, depth actor, estimator, omni encoder)`` -- the real classes, small hidden dims."""
    torch.manual_seed(seed)
    cfg = _policy_cfg()
    cfg.pop("class_name")
    policy = acm.ActorCriticRMA(W, 18, **cfg)
    estimator = fe.DefaultEstimator(num_prop=NUM_PROP, num_priv_explicit=NUM_PRIV_EXPLICIT, hidden_dims=[128, 64])
    encoder = fe.RecurrentMultiDepthBackbone(
        fe.DepthOnlyFCBackbone58x87(32, 512), dict(num_prop=89, num_cameras=4, latent_dim=32, grad_checkpoint=True)
    )
    depth_actor = copy.deepcopy(policy.actor)
    for p in depth_actor.parameters():  # a student that differs from its teacher
        p.data.add_(0.01 * torch.randn_like(p))
    return policy, depth_actor, estimator, encoder


def _step(policy, depth_actor, estimator, encoder):
    return step_mod.OmniStudentStep(encoder, depth_actor, estimator, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                    num_priv_explicit=NUM_PRIV_EXPLICIT, teacher=policy)


def _depth(n: int, gen=None) -> dict:
    return {g: torch.rand(n, 58, 87, generator=gen) - 0.5 for g in step_mod.DEPTH_GROUPS}


class _SpyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(W, 18)
        self.calls = []

    def forward(self, obs, hist_encoding, scandots_latent=None):
        self.calls.append((obs.clone(), hist_encoding, None if scandots_latent is None else scandots_latent.clone()))
        return self.lin(obs)


class _SpyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(W, 18)
        self.calls = []

    def act_inference(self, obs, hist_encoding=False, scandots_latent=None):
        self.calls.append((obs.clone(), hist_encoding))
        return self.lin(obs)


class _SpyEncoder(nn.Module):
    num_prop = 89
    num_cameras = 4

    def __init__(self):
        super().__init__()
        self.calls = []
        self.resets = []

    def forward(self, depth, prop):
        self.calls.append((depth.clone(), prop.clone()))
        return torch.full((depth.shape[0], 32), float(len(self.calls)))

    def reset(self, dones=None):
        self.resets.append(None if dones is None else dones.clone())

    def detach_hidden_states(self):
        pass


# =============================================================================================== layout
def test_layout_constants_match_the_omni_observation():
    restance = _load(RESTANCE_PY, "crab_hex_omni_restance__t4_test")
    assert step_mod.OBS_BUF_WIDTH == restance.OBS_BUF_WIDTH == 89
    assert step_mod.HISTORY_OFFSET == restance.HISTORY_OFFSET == 89 + 108 + 9 + 53
    assert step_mod.HISTORY_FRAMES == restance.HISTORY_FRAMES == 10
    assert step_mod.OBS_WIDTH == step_mod.HISTORY_OFFSET + 10 * 89 == W
    assert step_mod.DEPTH_NUM_PROP == 89  # D4
    assert (EST0, EST1) == (207, 216)
    cols = step_mod.tile_flag_columns(W)
    assert cols[:2] == (11, 12) and len(cols) == 22
    assert cols[2:] == tuple(259 + 89 * j + c for j in range(10) for c in (11, 12))
    assert step_mod.tile_flag_columns(89) == (11, 12)
    assert step_mod.FIELD_TILE_FLAGS == (1.0, 0.0)  # (~on_flat, on_flat) on a field tile


def test_constants_agree_across_the_spec_the_obs_term_and_the_step():
    assert step_mod.DEPTH_GROUPS == spec.DEPTH_GROUPS == tuple(f"depth_{n}" for n in spec.CAMERA_NAMES)
    assert spec.CAMERA_NAMES == ("front", "left", "right", "rear")
    src = OBS_PY.read_text()
    period = [n for n in ast.parse(src).body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "DEPTH_TERM_PERIOD" for t in n.targets)]
    assert len(period) == 1 and ast.literal_eval(period[0].value) == step_mod.DEPTH_PERIOD == spec.DEPTH_PERIOD == 5


def test_is_omni_student_flag():
    fwd_3a = {"depth_encoder": {"num_prop": 75, "backbone_class_name": "DepthOnlyFCBackbone58x87",
                                "encoder_class_name": "RecurrentDepthBackbone", "depth_shape": (87, 58),
                                "hidden_dims": 512, "learning_rate": 0.001, "num_steps_per_env": 120}}
    assert not step_mod.is_omni_student(fwd_3a)
    assert not step_mod.is_omni_student({"depth_encoder": None})
    assert not step_mod.is_omni_student({})
    assert not step_mod.is_omni_student(None)
    assert step_mod.is_omni_student({"depth_encoder": {"omni_student": True}})
    assert step_mod.is_omni_student(SimpleNamespace(depth_encoder=SimpleNamespace(omni_student=True)))
    assert not step_mod.is_omni_student(SimpleNamespace(depth_encoder=SimpleNamespace(num_prop=75)))
    assert not step_mod.is_omni_student(SimpleNamespace(depth_encoder=None))


# ==================================================================================== column invariants
def _obs(n=5, seed=1):
    return torch.randn(n, W, generator=torch.Generator().manual_seed(seed))


def test_student_copy_column_invariants():
    _p, _a, est, _e = _nets()
    prep = step_mod.OmniStudentObsPrep(est, num_prop=NUM_PROP, num_scan=NUM_SCAN, num_priv_explicit=NUM_PRIV_EXPLICIT)
    obs = _obs()
    before = obs.clone()
    out = prep(obs)
    assert torch.equal(obs, before), "the caller's observation was modified"
    flag_cols = list(step_mod.tile_flag_columns(W))
    est_cols = list(range(EST0, EST1))
    untouched = [c for c in range(W) if c not in flag_cols and c not in est_cols]
    assert torch.equal(out[:, untouched], obs[:, untouched])
    assert torch.equal(out[:, 6:8], obs[:, 6:8]), "facing command written"
    assert torch.equal(out[:, 13:15], obs[:, 13:15]), "privileged velocity written"
    for j, col in enumerate(flag_cols):
        assert torch.all(out[:, col] == (1.0 if j % 2 == 0 else 0.0)), col
    # the estimator reads the STUDENT copy (field pair in cols 11-12), never the true flags
    flagged = obs.clone()
    flagged[:, flag_cols] = torch.tensor([1.0, 0.0] * 11)
    with torch.no_grad():
        assert torch.equal(out[:, EST0:EST1], est(flagged[:, :NUM_PROP]))
        assert not torch.equal(out[:, EST0:EST1], est(obs[:, :NUM_PROP]))


def test_act_feeds_the_student_copy_k4_images_and_89_prop():
    _p, _a, est, _e = _nets()
    actor, enc, teacher = _SpyActor(), _SpyEncoder(), _SpyTeacher()
    st = step_mod.OmniStudentStep(enc, actor, est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                  num_priv_explicit=NUM_PRIV_EXPLICIT, teacher=teacher)
    obs = _obs(3)
    before = obs.clone()
    depth = _depth(3, torch.Generator().manual_seed(2))
    st.act(obs, depth, 0)
    assert torch.equal(obs, before)
    student = st.student_obs(obs)
    (a_obs, hist, latent), = actor.calls
    assert hist is True and torch.equal(a_obs, student) and torch.equal(latent, torch.full((3, 32), 1.0))
    (d, prop), = enc.calls
    assert d.shape == (3, 4, 58, 87)
    for k, g in enumerate(step_mod.DEPTH_GROUPS):
        assert torch.equal(d[:, k], depth[g]), g
    assert torch.equal(prop, student[:, :89])
    # the teacher label: the TRUE obs (true tile flags) with its own estimator write
    st.teacher_actions(obs)
    (t_obs, t_hist), = teacher.calls
    assert t_hist is True
    assert torch.equal(t_obs[:, 11:13], obs[:, 11:13])
    with torch.no_grad():
        assert torch.equal(t_obs[:, EST0:EST1], est(obs[:, :NUM_PROP]))
    keep = [c for c in range(W) if not EST0 <= c < EST1]
    assert torch.equal(t_obs[:, keep], obs[:, keep])


def test_encode_cadence_holds_the_latent_between_refreshes():
    _p, _a, est, _e = _nets()
    enc = _SpyEncoder()
    st = step_mod.OmniStudentStep(enc, _SpyActor(), est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                  num_priv_explicit=NUM_PRIV_EXPLICIT)
    obs, depth = _obs(2), _depth(2)
    for counter in range(3, 13):  # starts off-phase: the first call encodes (nothing held)
        st.act(obs, depth, counter)
    assert len(enc.calls) == 3  # counters 3 (none held), 5, 10
    st.reset()
    assert st.depth_latent is None and enc.resets[-1] is None
    st.act(obs, depth, 13)
    assert len(enc.calls) == 4


def test_zero_depth_feeds_zero_images_on_the_same_path():
    _p, _a, est, _e = _nets()
    enc = _SpyEncoder()
    st = step_mod.OmniStudentStep(enc, _SpyActor(), est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                  num_priv_explicit=NUM_PRIV_EXPLICIT, zero_depth=True)
    obs, depth = _obs(2), _depth(2)
    st.act(obs, depth, 0)
    st.act(obs, torch.stack([depth[g] for g in step_mod.DEPTH_GROUPS], dim=1), 5)
    assert len(enc.calls) == 2 and all(torch.all(d == 0) and d.shape == (2, 4, 58, 87) for d, _ in enc.calls)
    assert torch.equal(enc.calls[0][1], st.student_obs(obs)[:, :89])  # proprio unchanged
    cfg = _train_cfg(omni=True, steps=10)
    assert "zero_depth" not in cfg["depth_encoder"]  # absent -> off
    p, a, e, en = _nets()
    alg = SimpleNamespace(depth_encoder=en, depth_actor=a, estimator=e, policy=p)
    assert not step_mod.OmniStudentStep.from_runner(SimpleNamespace(cfg=cfg, alg=alg, device="cpu")).zero_depth
    cfg["depth_encoder"]["zero_depth"] = True
    assert step_mod.OmniStudentStep.from_runner(SimpleNamespace(cfg=cfg, alg=alg, device="cpu")).zero_depth


def test_step_refuses_a_mismatched_encoder():
    _p, actor, est, _e = _nets()
    bad = _SpyEncoder()
    bad.num_cameras = 3
    with pytest.raises(ValueError, match="K=3"):
        step_mod.OmniStudentStep(bad, actor, est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                 num_priv_explicit=NUM_PRIV_EXPLICIT)


# ======================================================================================== GRU reset
def test_gru_reset_touches_only_done_rows():
    _p, _a, _est, enc = _nets()
    enc.eval()
    n = 4
    depth = torch.stack([_depth(n)[g] for g in step_mod.DEPTH_GROUPS], dim=1)
    prop = torch.randn(n, 89)
    with torch.no_grad():
        enc(depth, prop)
        enc(depth, prop)
    h = enc.hidden_states.clone()
    assert torch.all(h.abs().sum(-1) > 0)
    enc.reset(torch.tensor([0, 1, 0, 1]))
    assert torch.equal(enc.hidden_states[:, [0, 2]], h[:, [0, 2]])
    assert torch.all(enc.hidden_states[:, [1, 3]] == 0)
    # a reset row then encodes exactly as a fresh encoder does
    fresh = copy.deepcopy(enc)
    fresh.reset()
    with torch.no_grad():
        out = enc(depth, prop)
        ref = fresh(depth, prop)
    assert torch.equal(out[[1, 3]], ref[[1, 3]]) and not torch.equal(out[[0, 2]], ref[[0, 2]])
    with pytest.raises(ValueError, match="done flags"):
        enc.reset(torch.tensor([1, 0]))


def test_gru_reset_inside_a_recorded_graph_backpropagates():
    _p, _a, _est, enc = _nets()
    enc.train()
    depth = torch.stack([_depth(3)[g] for g in step_mod.DEPTH_GROUPS], dim=1)
    prop = torch.randn(3, 89)
    a = enc(depth, prop)
    enc.reset(torch.tensor([1, 0, 0]))
    b = enc(depth, prop)
    (a.sum() + b.sum()).backward()  # an in-place reset would raise here
    assert enc.rnn.weight_hh_l0.grad is not None


def test_grad_checkpoint_gives_the_same_gradient():
    _p, _a, _est, enc = _nets()
    plain = copy.deepcopy(enc)
    plain.grad_checkpoint = False
    enc.train()
    plain.train()
    depth = torch.stack([_depth(3, torch.Generator().manual_seed(5))[g] for g in step_mod.DEPTH_GROUPS], dim=1)
    prop = torch.randn(3, 89, generator=torch.Generator().manual_seed(6))
    for m in (enc, plain):
        m(depth, prop).pow(2).sum().backward()
    for (name, p), q in zip(enc.named_parameters(), plain.parameters()):
        assert torch.allclose(p.grad, q.grad, rtol=0, atol=1e-6), name


# ======================================================================== runner: fake env, both paths
class FakeVecEnv:
    """A deterministic stand-in for ``ParkourRslRlVecEnvWrapper``: 1149-wide obs, image groups, done rows."""

    def __init__(self, n: int, groups, *, seed: int, delta_yaw: bool):
        self.num_envs, self.num_actions, self.device, self.cfg = n, 18, "cpu", None
        self.unwrapped = SimpleNamespace(common_step_counter=0, step_dt=0.02)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.g = torch.Generator().manual_seed(seed)
        self.groups, self.delta_yaw = tuple(groups), delta_yaw
        self.received, self.emitted, self.dones = [], [], []
        self._obs = self._draw()

    def _draw(self):
        return torch.randn(self.num_envs, W, generator=self.g) * 0.5

    def _extras(self, obs):
        d = {"policy": obs}
        for gname in self.groups:
            d[gname] = torch.rand(self.num_envs, 58, 87, generator=self.g) - 0.5
        if self.delta_yaw:
            d["delta_yaw_ok"] = torch.rand(self.num_envs, 1, generator=self.g) > 0.3
        return d

    def get_observations(self):
        self.emitted.append((self._obs, self._obs.clone()))
        return self._obs, {"observations": self._extras(self._obs)}

    def step(self, actions):
        a = actions.detach().clone()
        self.received.append(a)
        self.unwrapped.common_step_counter += 1
        obs = self._draw()
        obs[:, :18] += 0.1 * a
        dones = (torch.rand(self.num_envs, generator=self.g) < 0.25).long()
        self.dones.append(dones.clone())
        self.emitted.append((obs, obs.clone()))
        self._obs = obs
        return obs, torch.zeros(self.num_envs), dones, {"observations": self._extras(obs), "log": {}}


def _train_cfg(*, omni: bool, steps: int) -> dict:
    de = dict(backbone_class_name="DepthOnlyFCBackbone58x87", encoder_class_name="RecurrentDepthBackbone",
              num_prop=NUM_PROP, num_scan=NUM_SCAN, num_hist=NUM_HIST, num_priv_explicit=NUM_PRIV_EXPLICIT,
              num_priv_latent=NUM_PRIV_LATENT, depth_shape=(87, 58), hidden_dims=512, learning_rate=1e-3,
              num_steps_per_env=steps)
    if omni:
        de.update(encoder_class_name="RecurrentMultiDepthBackbone", num_prop=89, num_cameras=4,
                  depth_shape=(58, 87), latent_dim=32, grad_checkpoint=True, omni_student=True)
    return dict(
        algorithm=dict(class_name="DistillationWithExtractor", learning_rate=2e-4, max_grad_norm=1.0),
        estimator=dict(class_name="DefaultEstimator", num_prop=NUM_PROP, num_scan=NUM_SCAN, num_hist=NUM_HIST,
                       num_priv_explicit=NUM_PRIV_EXPLICIT, num_priv_latent=NUM_PRIV_LATENT, hidden_dims=[128, 64],
                       learning_rate=1e-4, train_with_estimated_states=True),
        depth_encoder=de, policy=_policy_cfg(), num_steps_per_env=24, save_interval=100,
        empirical_normalization=False, logger="tensorboard", resume_alg_state=True,
    )


def _run(rmod, cfg: dict, env: FakeVecEnv, log_dir: Path, iters: int, monkeypatch):
    """Build the real ``OnPolicyRunnerWithExtractor`` on the fake env and run ``learn`` (seeded)."""
    monkeypatch.setattr(rmod, "store_code_state", lambda *a, **k: [])
    torch.manual_seed(0)
    runner = rmod.OnPolicyRunnerWithExtractor(env, copy.deepcopy(cfg), log_dir=str(log_dir), device="cpu")
    losses = []
    orig = runner.alg.update_depth_actor

    def rec(actions_buffer, yaws_buffer):
        out = orig(actions_buffer, yaws_buffer)
        losses.append(dict(out))
        return out

    runner.alg.update_depth_actor = rec
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
    return runner, losses


def _baseline_modules(tmp_path: Path):
    """The pre-M1b ``parkour/scripts/rsl_rl/modules`` package from git, loaded under a private name."""
    name = "_t4_rsl_modules_baseline"
    if name in sys.modules:
        return sys.modules[name]
    try:
        files = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", BASELINE,
                                "parkour/scripts/rsl_rl/modules"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"baseline commit {BASELINE} unavailable: {exc}")
    root = tmp_path / "baseline_modules"
    for rel in files.stdout.split():
        if not rel.endswith(".py"):
            continue
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASELINE}:{rel}"], check=True,
                             capture_output=True, text=True).stdout
        dst = root / Path(rel).relative_to("parkour/scripts/rsl_rl/modules")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src)
    return _load_pkg(name, root)


def test_forward_learn_vision_is_bitwise_unchanged(tmp_path, monkeypatch):
    """Forward call identity: the forward student (no omni flag) trains bit for bit as before M1b."""
    base_pkg = _baseline_modules(tmp_path)
    base_rmod = importlib.import_module(f"{base_pkg.__name__}.on_policy_runner_with_extractor")
    cfg = _train_cfg(omni=False, steps=10)
    runs = []
    for rmod, tag in ((base_rmod, "base"), (runner_mod, "cur")):
        env = FakeVecEnv(4, ("depth_camera",), seed=11, delta_yaw=True)
        runner, losses = _run(rmod, cfg, env, tmp_path / tag, 2, monkeypatch)
        runs.append((runner, losses, env))
    (rb, lb, eb), (rc, lc, ec) = runs
    assert not hasattr(rc, "omni_student_step"), "the forward run reached the omni path"
    assert len(eb.received) == len(ec.received) == 20
    for a, b in zip(eb.received, ec.received):
        assert torch.equal(a, b)
    assert lb == lc and all(d["yaw_loss"] > 0 for d in lc)
    for mod_b, mod_c in ((rb.alg.depth_actor, rc.alg.depth_actor), (rb.alg.depth_encoder, rc.alg.depth_encoder)):
        sb, sc = mod_b.state_dict(), mod_c.state_dict()
        assert sb.keys() == sc.keys()
        for k in sb:
            assert torch.equal(sb[k], sc[k]), k


def test_omni_trainer_runs_the_shared_step(tmp_path, monkeypatch):
    """The trainer's omni branch: every env action is ``OmniStudentStep.act``'s, GRU reset on done rows,
    the env's observations are never written, zero-weight yaw loss."""
    acts, resets = [], []
    orig_act, orig_reset = step_mod.OmniStudentStep.act, step_mod.OmniStudentStep.reset

    def act(self, obs, depth, counter):
        out = orig_act(self, obs, depth, counter)
        acts.append(out.detach().clone())
        return out

    def reset(self, dones=None):
        resets.append(None if dones is None else dones.clone())
        return orig_reset(self, dones)

    monkeypatch.setattr(step_mod.OmniStudentStep, "act", act)
    monkeypatch.setattr(step_mod.OmniStudentStep, "reset", reset)
    env = FakeVecEnv(4, step_mod.DEPTH_GROUPS, seed=12, delta_yaw=False)
    runner, losses = _run(runner_mod, _train_cfg(omni=True, steps=10), env, tmp_path / "omni", 2, monkeypatch)
    step = runner.omni_student_step
    assert isinstance(step, step_mod.OmniStudentStep)
    assert step.depth_encoder is runner.alg.depth_encoder and step.depth_actor is runner.alg.depth_actor
    assert step.teacher is runner.alg.policy and step.prep.estimator is runner.alg.estimator
    assert len(acts) == len(env.received) == 20
    for a, b in zip(acts, env.received):
        assert torch.equal(a, b)
    assert resets[0] is None and len(resets) == 21
    for r, d in zip(resets[1:], env.dones):
        assert torch.equal(r, d)
    for obs, snap in env.emitted:
        assert torch.equal(obs, snap), "the omni trainer wrote into an env observation"
    assert len(losses) == 2 and all(d["yaw_loss"] == 0.0 and d["total_loss"] == d["depth_actor_loss"] for d in losses)
    assert all(math.isfinite(d["depth_actor_loss"]) for d in losses)
    # from_runner (what play.py / play_stream_video.py / the harness build) == the trainer's step
    other = step_mod.OmniStudentStep.from_runner(runner)
    obs = _obs(4, seed=3)
    depth = _depth(4, torch.Generator().manual_seed(4))
    runner.alg.depth_encoder.eval()
    with torch.no_grad():
        step.reset()
        x = orig_act(step, obs, depth, 0)
        other.reset()
        y = orig_act(other, obs, depth, 0)
    assert torch.equal(x, y)


def test_from_runner_refuses_a_forward_cfg():
    fake = SimpleNamespace(cfg=_train_cfg(omni=False, steps=10), alg=None, device="cpu")
    with pytest.raises(ValueError, match="omni student"):
        step_mod.OmniStudentStep.from_runner(fake)


# ======================================================================= the entry scripts (AST)
def _git_show(rel: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{BASELINE}:{rel}"], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"baseline commit {BASELINE} unavailable: {exc}")


def _main(src: str) -> ast.FunctionDef:
    return [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]


def _dl(x) -> str:
    """``ast.dump`` of a node or of a statement list."""
    return repr([ast.dump(n) for n in x]) if isinstance(x, list) else ast.dump(x)


def _mentions(node, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _first(stmts, kind, pred=lambda n: True):
    return [i for i, n in enumerate(stmts) if isinstance(n, kind) and pred(n)][0]


def _is_distill_if(n) -> bool:
    return "DistillationWithExtractor" in ast.dump(n.test)


@pytest.mark.parametrize("script", [PLAY_PY, PSV_PY], ids=lambda p: p.name)
def test_entry_script_forward_branches_unchanged(script):
    """Forward call identity for play.py / play_stream_video.py: the teacher and the forward-student branches
    (export block and loop) are AST-identical to the baseline; the only other changes are the omni lines."""
    rel = str(script.relative_to(REPO))
    base, cur = _main(_git_show(rel)), _main(script.read_text())
    cur_body = [s for s in cur.body if not (isinstance(s, ast.Assign) and _mentions(s, "omni_student"))]
    assert len(cur_body) == len(base.body)
    i_exp = _first(base.body, ast.If, _is_distill_if)
    i_loop = _first(base.body, ast.While)
    for i, (b, c) in enumerate(zip(base.body, cur_body)):
        if i not in (i_exp, i_loop):
            assert ast.dump(b) == ast.dump(c), f"main() statement {i} changed"
    # export block: omni first, then the baseline if/else verbatim
    b_exp, c_exp = base.body[i_exp], cur_body[i_exp]
    assert isinstance(c_exp.test, ast.BoolOp) and _mentions(c_exp.test, "is_omni_student")
    (c_fwd,) = c_exp.orelse
    assert ast.dump(c_fwd) == ast.dump(b_exp)
    # loop: the baseline teacher body, then omni, then the baseline forward-student block verbatim
    b_loop, c_loop = base.body[i_loop], cur_body[i_loop]
    c_stmts = [s for s in c_loop.body if not (isinstance(s, ast.If) and _mentions(s.test, "omni_student"))]
    assert len(c_stmts) == len(b_loop.body)
    j_if = _first(b_loop.body, ast.If, _is_distill_if)
    for j, (b, c) in enumerate(zip(b_loop.body, c_stmts)):
        if j == j_if:
            assert _dl(b.test) == _dl(c.test) and _dl(b.body) == _dl(c.body)
            (omni_if,) = c.orelse
            assert _mentions(omni_if.test, "omni_student")
            assert _dl(omni_if.orelse) == _dl(b.orelse), "forward student loop branch changed"
        elif isinstance(b, ast.Assign) and "env.step" in ast.unparse(b.value):
            assert ast.unparse(b.value) == ast.unparse(c.value)
            assert ast.unparse(c.targets[0]) == "(obs, _, dones, extras)"
        else:
            assert ast.dump(b) == ast.dump(c), f"loop statement {j} changed"


def _calls(tree, attr: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


@pytest.mark.parametrize("script", [PLAY_PY, PSV_PY], ids=lambda p: p.name)
def test_entry_scripts_call_the_shared_step(script):
    tree = ast.parse(script.read_text())
    (fr,) = [c for c in _calls(tree, "from_runner") if ast.unparse(c.func.value) == "OmniStudentStep"]
    assert ast.unparse(fr.args[0]) == "ppo_runner"
    (act,) = [c for c in _calls(tree, "act") if ast.unparse(c.func.value) == "omni_student"]
    assert [ast.unparse(a) for a in act.args] == ["obs", "extras['observations']", "env.unwrapped.common_step_counter"]
    (rst,) = [c for c in _calls(tree, "reset") if ast.unparse(c.func.value) == "omni_student"]
    assert [ast.unparse(a) for a in rst.args] == ["dones"]
    assert [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "export_omni_student_deploy"]
    # the omni branches never write obs cols 6:8 (that write lives only in the forward branch)
    main = _main(script.read_text())
    omni_blocks = [n for n in ast.walk(main) if isinstance(n, ast.If) and _mentions(n.test, "omni_student")
                   or isinstance(n, ast.If) and _mentions(n.test, "is_omni_student")]
    for blk in omni_blocks:
        for n in ast.walk(ast.Module(body=blk.body, type_ignores=[])):
            if isinstance(n, ast.Assign):
                assert "6:8" not in ast.unparse(n.targets[0])


def test_trainer_routes_to_the_shared_step():
    tree = ast.parse(RUNNER_PY.read_text())
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyRunnerWithExtractor"][0]
    meth = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    first = meth["learn_vision"].body[0]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "is_omni_student(self.cfg)"
    assert ast.unparse(first.body[0]) == "return self._learn_vision_omni(num_learning_iterations, init_at_random_ep_len)"
    omni = meth["_learn_vision_omni"]
    (fr,) = [c for c in _calls(omni, "from_runner") if ast.unparse(c.func.value) == "OmniStudentStep"]
    assert ast.unparse(fr.args[0]) == "self"
    assert [ast.unparse(c) for c in _calls(omni, "act")] == ["step.act(obs, depth, self.env.unwrapped.common_step_counter)"]
    assert [ast.unparse(c) for c in _calls(omni, "teacher_actions")] == ["step.teacher_actions(obs)"]
    assert "step.reset(dones)" in [ast.unparse(c) for c in _calls(omni, "reset")]
    assert "self.alg.update_depth_actor(actions_buffer, None)" in [ast.unparse(c) for c in _calls(omni, "update_depth_actor")]
    for n in ast.walk(omni):  # no write into an observation in the omni loop
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript):
            assert "obs" not in ast.unparse(n.targets[0].value), ast.unparse(n)
    # the forward body of learn_vision is the baseline's, after the routing statement
    base = ast.parse(_git_show(str(RUNNER_PY.relative_to(REPO))))
    bcls = [n for n in base.body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyRunnerWithExtractor"][0]
    blv = [n for n in bcls.body if isinstance(n, ast.FunctionDef) and n.name == "learn_vision"][0]
    assert [ast.dump(s) for s in meth["learn_vision"].body[1:]] == [ast.dump(s) for s in blv.body]


# =========================================================================================== cameras
def test_composed_quaternions_point_every_camera_as_specified():
    specs = spec.camera_specs()
    assert tuple(specs) == spec.CAMERA_NAMES
    for name, c in specs.items():
        assert spec.camera_rotation_errors(name, c["rot"]) == []
        axes = spec.body_axes(c["rot"])
        assert all(abs(a - b) < 1e-3 for a, b in zip(axes["forward"], spec.YAML_OPTICAL_AXES[name])), name
        assert abs(sum(v * v for v in c["rot"]) - 1.0) < 1e-12
        # datasheet FOV recovered from the apertures
        f = spec.FOCAL_LENGTH
        assert abs(math.degrees(2 * math.atan(c["horizontal_aperture"] / (2 * f))) - c["hfov_deg"]) < 1e-9
        assert abs(math.degrees(2 * math.atan(c["vertical_aperture"] / (2 * f))) - c["vfov_deg"]) < 1e-9
        assert (c["height"], c["width"]) == (58, 87)


def test_quaternion_recipe_reproduces_the_forward_and_sideways_cameras():
    # forward CAMERA_CFG: euler (180, 70, -90) -> optical axis body (0.94, 0, -0.342), 20 deg down
    fwd = spec.forward_euler_quat(180.0, 70.0, -90.0)
    assert spec.camera_quat(0.0, 20.0) == pytest.approx(fwd, abs=1e-12)
    axis = spec.body_axes(fwd)["forward"]
    assert axis == pytest.approx((math.cos(math.radians(20)), 0.0, -math.sin(math.radians(20))), abs=1e-9)
    # crab_hex_side_scene_cfg's tripwire value for q_z(+90) (x) the forward rotation
    assert spec.camera_quat(90.0, 20.0) == pytest.approx((-0.5735764, 0.8191520, 0.0, 0.0), abs=1e-6)


def _fake_scene():
    cams = {}
    for name, c in spec.camera_specs().items():
        cams[c["attr"]] = SimpleNamespace(
            prim_path=spec.CAMERA_PRIM_PATH, data_types=["distance_to_camera"],
            offset=SimpleNamespace(pos=tuple(c["pos"]), rot=tuple(c["rot"]), convention="ros"),
            pattern_cfg=SimpleNamespace(height=58, width=87, focal_length=spec.FOCAL_LENGTH,
                                        horizontal_aperture=c["horizontal_aperture"],
                                        vertical_aperture=c["vertical_aperture"]),
            max_distance=2.0, mesh_prim_paths=["/World/ground"],
        )
    return SimpleNamespace(**cams)


def test_camera_tripwire_passes_the_spec_and_catches_drift():
    spec.check_scene_cameras(_fake_scene())
    p = spec.camera_specs()["left"]["pitch_down_deg"]
    edits = {
        # the forbidden path: turning a camera through the Euler yaw term (rolls the image instead)
        "euler yaw edit": lambda s: setattr(s.depth_camera_left.offset, "rot", spec.forward_euler_quat(180.0, 90.0 - p, 0.0)),
        "swapped sides": lambda s: setattr(s.depth_camera_left.offset, "rot", tuple(s.depth_camera_right.offset.rot)),
        "pitch +2 deg": lambda s: setattr(s.depth_camera_front.offset, "rot", spec.camera_quat(0.0, 57.0)),
        "roll about the axis": lambda s: setattr(s.depth_camera_rear.offset, "rot", spec.quat_mul(
            s.depth_camera_rear.offset.rot, (math.cos(0.05), 0.0, 0.0, math.sin(0.05)))),
        "grid": lambda s: setattr(s.depth_camera_front.pattern_cfg, "width", 106),
        "fov": lambda s: setattr(s.depth_camera_right.pattern_cfg, "horizontal_aperture", 20.955),
        "targets": lambda s: setattr(s.depth_camera_front, "mesh_prim_paths", ["/World/ground", "/World/tables"]),
        "clip": lambda s: setattr(s.depth_camera_rear, "max_distance", 3.0),
        "pos": lambda s: setattr(s.depth_camera_left.offset, "pos", (0.0, 0.61, 0.10)),
        "convention": lambda s: setattr(s.depth_camera_left.offset, "convention", "world"),
        "missing": lambda s: delattr(s, "depth_camera_rear"),
    }
    for label, edit in edits.items():
        scene = _fake_scene()
        edit(scene)
        with pytest.raises(ValueError, match="drifted"):
            spec.check_scene_cameras(scene)
        assert label  # (named for the failure message)


def test_scene_and_env_cfgs_wire_the_spec():
    scene = ast.parse(SCENE_PY.read_text())
    cls = [n for n in scene.body if isinstance(n, ast.ClassDef) and n.name == "CrabHexOmniStudentSceneCfg"][0]
    assert [ast.unparse(b) for b in cls.bases] == ["CrabHexOmniTeacherSceneCfg"]
    assigns = [(n.targets[0].id, ast.unparse(n.value)) for n in cls.body if isinstance(n, ast.Assign)]
    assert assigns == [(a, f"_student_camera('{n}')") for a, n in zip(spec.CAMERA_ATTRS, spec.CAMERA_NAMES)]
    env = ast.parse(ENV_PY.read_text())
    classes = {n.name: n for n in env.body if isinstance(n, ast.ClassDef)}
    obs_cls = classes["CrabHexOmniStudentObservationsCfg"]
    assert [ast.unparse(b) for b in obs_cls.bases] == ["CrabHexOmniObservationsCfg"]
    groups = [n.target.id for n in obs_cls.body if isinstance(n, ast.AnnAssign)]
    assert tuple(groups) == spec.DEPTH_GROUPS
    nested = [n for n in obs_cls.body if isinstance(n, ast.ClassDef)]
    assert [ast.unparse(n.body[0].value) for n in nested] == [f"_depth_term('{a}')" for a in spec.CAMERA_ATTRS]
    names = [n.target.id for n in obs_cls.body if isinstance(n, ast.AnnAssign)] + [n.name for n in nested]
    assert not [n for n in names if "yaw" in n.lower()], names  # no delta_yaw_ok group
    for name, base in (("CrabHexOmniStudentEnvCfg", "CrabHexOmniTeacherEnvCfg"),
                       ("CrabHexOmniStudentEnvCfgPLAY", "CrabHexOmniFlatWalkEnvCfgPLAY")):
        c = classes[name]
        assert [ast.unparse(b) for b in c.bases] == [base]
        fields = {n.target.id: ast.unparse(n.value) for n in c.body if isinstance(n, ast.AnnAssign)}
        assert fields == {"scene": "CrabHexOmniStudentSceneCfg(num_envs=192, env_spacing=1.0)",
                          "observations": "CrabHexOmniStudentObservationsCfg()"}
        post = [n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"][0]
        assert [ast.unparse(s) for s in post.body] == ["super().__post_init__()", "_apply_student_sensors(self)"]
    sensors = [n for n in env.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_student_sensors"][0]
    text = ast.unparse(sensors)
    assert "_student_spec.check_scene_cameras(cfg.scene)" in text and "random_camera_position" in text
    assert "cfg.sim.dt * cfg.decimation" in text


def test_gym_ids_and_runner_cfg():
    tree = ast.parse(INIT_PY.read_text())
    regs = {}
    for c in _calls(tree, "register"):
        kw = {k.arg: k.value for k in c.keywords}
        regs[ast.literal_eval(kw["id"])] = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(kw["kwargs"].keys, kw["kwargs"].values)}
    assert regs["Isaac-Crab-Hex-Omni-Student-v0"] == {
        "env_cfg_entry_point": "CrabHexOmniStudentEnvCfg",
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg"}
    assert regs["Isaac-Crab-Hex-Omni-Student-Play-v0"] == {
        "env_cfg_entry_point": "CrabHexOmniStudentEnvCfgPLAY",
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg"}
    rc = {n.name: n for n in ast.parse(RUNNER_CFG_PY.read_text()).body if isinstance(n, ast.ClassDef)}
    enc = {n.target.id: ast.unparse(n.value) for n in rc["CrabHexOmniRslRlDepthEncoderCfg"].body if isinstance(n, ast.AnnAssign)}
    assert enc["encoder_class_name"] == "'RecurrentMultiDepthBackbone'" and enc["num_prop"] == "89"
    assert enc["omni_student"] == "True" and enc["grad_checkpoint"] == "True" and enc["zero_depth"] == "False"
    run = rc["CrabHexOmniStudentPPORunnerCfg"]
    assert [ast.unparse(b) for b in run.bases] == ["CrabHexStudentPPORunnerCfg"]
    body = ast.unparse(run)
    assigned = [ast.unparse(t) for n in run.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])]
    assert "experiment_name = 'crab_hex_omni_student'" in body and "clip_actions" not in assigned
    assert "_student_spec.apply_student_runner_knobs(self)" in body
    fwd = [n for n in ast.parse(FWD_RUNNER_CFG_PY.read_text()).body
           if isinstance(n, ast.ClassDef) and n.name == "CrabHexStudentPPORunnerCfg"][0]
    assert "clip_actions = 1.0" in ast.unparse(fwd)  # inherited, load-bearing


# ===================================================================================== runner knobs
def _agent():
    return SimpleNamespace(policy=SimpleNamespace(noise_std_type="scalar", init_noise_std=0.65))


def test_student_runner_applies_only_the_std_knobs():
    a = _agent()
    spec.apply_student_runner_knobs(a, {})
    assert (a.policy.noise_std_type, a.policy.init_noise_std) == ("scalar", 0.65)
    spec.apply_student_runner_knobs(a, {"KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8",
                                        "KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_ENTROPY_COEF": " "})
    assert (a.policy.noise_std_type, a.policy.init_noise_std) == ("soft", 0.8)
    with pytest.raises(ValueError, match="soft band"):
        spec.apply_student_runner_knobs(_agent(), {"KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "2.5"})
    for var in spec.STUDENT_RUNNER_KNOBS_REFUSED:
        with pytest.raises(ValueError, match=var):
            spec.apply_student_runner_knobs(_agent(), {var: "1"})
    # DESIGN stage S1: the runner-only keys of the T* overlay (P2__S1_s18) are exactly refused ones
    for var in ("KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND",
                "KRABBY_ENTROPY_COEF", "KRABBY_PPO_HAND_DIAG", "KRABBY_OMNI_MIRROR_IMAGES"):
        assert var in spec.STUDENT_RUNNER_KNOBS_REFUSED


def test_refused_list_covers_every_runner_knob_apply_exploration_knobs_reads():
    mk = _load(MOTION_KNOBS_PY, "motion_knobs__t4_test")
    src = MOTION_KNOBS_PY.read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "apply_exploration_knobs"][0]
    read = set()
    for c in _calls(fn, "get"):
        if ast.unparse(c.func.value) == "env" and c.args and isinstance(c.args[0], ast.Name):
            read.add(getattr(mk, c.args[0].id))
    assert len(read) >= 20
    # KRABBY_OMNI_STAND_MASK is an ENV-side observation knob (read there only to mirror the mask onto an armed
    # BC bank, which the student refuses anyway): T*'s env overlay may carry it, so it is not refused.
    covered = set(spec.STUDENT_RUNNER_KNOBS_REFUSED) | set(spec.STUDENT_RUNNER_KNOBS_APPLIED) | {mk.STAND_MASK_VAR}
    assert read <= covered, sorted(read - covered)
    assert not set(spec.STUDENT_RUNNER_KNOBS_REFUSED) & set(spec.STUDENT_RUNNER_KNOBS_APPLIED)
    ok = _load(OMNI_KNOBS_PY, "omni_knobs__t4_test")
    assert ok.MIRROR_IMAGES_VAR in spec.STUDENT_RUNNER_KNOBS_REFUSED
    assert (mk.STD_PARAM_VAR, mk.INIT_STD_VAR) == spec.STUDENT_RUNNER_KNOBS_APPLIED


# ======================================================================================= image term
def _image_term_ns():
    src = OBS_PY.read_text()
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
    exec(compile(code, str(OBS_PY), "exec"), ns)
    return ns


def test_batched_preprocessing_equals_the_forward_per_env_path():
    tv = pytest.importorskip("torchvision")
    ns = _image_term_ns()
    g = torch.Generator().manual_seed(0)
    raw = torch.rand(5, 60, 106, generator=g) * 2.5
    resize = tv.transforms.Resize((58, 87), interpolation=tv.transforms.InterpolationMode.BICUBIC)
    fwd = torch.stack([resize(d[:-2, 4:-4][None]).squeeze() / 2.0 - 0.5 for d in raw])  # image_features
    out = ns["process_depth_batch"](raw, clip=2.0, resize=(58, 87), crop=(0, 2, 4, 4))
    assert torch.equal(out, fwd)
    direct = torch.rand(3, 58, 87, generator=g) * 2.0
    assert torch.equal(ns["process_depth_batch"](direct, clip=2.0, resize=(58, 87)), direct / 2.0 - 0.5)
    with pytest.raises(ValueError):
        ns["_check_depth_crop"]((1, 2, 3))


def test_image_term_cadence_latency_and_reset_refill():
    ns = _image_term_ns()
    n = 3

    class _Sensor:
        cfg = SimpleNamespace(max_distance=2.0)
        frame = 0

        @property
        def data(self):
            return SimpleNamespace(output={"distance_to_camera": torch.full((n, 58, 87, 1), float(self.frame))})

    sensor = _Sensor()
    env = SimpleNamespace(num_envs=n, device="cpu", scene={"cam": sensor}, common_step_counter=0)
    cfg = SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name="cam"), "resize": (58, 87), "buffer_len": 2})
    term = ns["image_features_batched"](cfg, env)
    term.reset(None)  # frame 0 everywhere
    seen = []
    for counter in range(0, 12):
        env.common_step_counter = counter
        sensor.frame = counter
        seen.append(float(term(env, cfg.params["sensor_cfg"], (58, 87), 2)[0, 0, 0]))
    norm = lambda f: f / 2.0 - 0.5  # noqa: E731
    # refreshed at 0, 5, 10; returns the PREVIOUS refresh (buffer[-2])
    assert seen == [norm(0)] * 5 + [norm(0)] * 5 + [norm(5)] * 2
    sensor.frame = 7
    term.reset([1])
    out = term(env, cfg.params["sensor_cfg"], (58, 87), 2)
    assert torch.all(out[1] == norm(7)) and torch.all(out[0] == norm(5))
    with pytest.raises(ValueError, match="buffer_len"):
        ns["image_features_batched"](SimpleNamespace(params=dict(cfg.params, buffer_len=1)), env)


# ============================================================================================ export
def _onnx_runner(path: str):
    """``(feeds -> outputs, label)``: onnxruntime when installed, else the ONNX reference evaluator."""
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(onnx.load(path))
    try:
        import onnxruntime as ort  # noqa: PLC0415

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return (lambda feeds: sess.run(None, feeds)), "onnxruntime"
    except ImportError:
        from onnx.reference import ReferenceEvaluator  # noqa: PLC0415

        ref = ReferenceEvaluator(path)
        return (lambda feeds: ref.run(None, feeds)), "onnx.reference.ReferenceEvaluator"


def test_export_round_trip_matches_the_sim_step(tmp_path):
    policy, depth_actor, est, enc = _nets(seed=3)
    enc.eval()
    depth_actor.eval()
    files = export_mod.export_omni_student(depth_actor, est, enc, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                           num_priv_explicit=NUM_PRIV_EXPLICIT, path=str(tmp_path))
    io = json.loads(Path(files["io"]).read_text())
    assert io["depth_image"] == [1, 4, 58, 87] and io["proprioception"] == [1, 89] and io["gru_hidden"] == [1, 1, 512]
    assert io["camera_groups"] == list(step_mod.DEPTH_GROUPS) and io["encode_every_policy_steps"] == 5
    assert io["estimator_cols"] == [207, 216] and io["field_tile_flags"] == [1.0, 0.0]
    E = export_mod.OmniDepthEncoderExport(enc)
    P = export_mod.OmniDeployPolicyExport(depth_actor, est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                          num_priv_explicit=NUM_PRIV_EXPLICIT)
    jit_e, jit_p = torch.jit.load(files["depth_jit"]), torch.jit.load(files["policy_jit"])
    st = _step(policy, depth_actor, est, enc)
    g = torch.Generator().manual_seed(9)
    h = torch.zeros(1, 1, 512)
    st.reset()
    for counter in (0, 5):  # two encodes: the exported GRU state is fed back
        obs = torch.randn(1, W, generator=g)
        depth = {k: torch.rand(1, 58, 87, generator=g) - 0.5 for k in step_mod.DEPTH_GROUPS}
        dimg = torch.stack([depth[k] for k in step_mod.DEPTH_GROUPS], dim=1)
        with torch.no_grad():
            a_sim = st.act(obs, depth, counter)
            latent, h_new = E(dimg, obs[:, :89], h)
            a_exp = P(obs, latent)
            jl, jh = jit_e(dimg, obs[:, :89], h)
            ja = jit_p(obs, latent)
        assert torch.equal(a_sim, a_exp), "the exported graphs differ from the sim step"
        assert torch.equal(latent, st.depth_latent) and torch.equal(h_new, enc.hidden_states)
        assert torch.equal(jl, latent) and torch.equal(jh, h_new) and torch.equal(ja, a_exp)
        h = h_new


def test_onnx_matches_torch_on_a_dummy_k4_input(tmp_path):
    _p, depth_actor, est, enc = _nets(seed=4)
    files = export_mod.export_omni_student(depth_actor, est, enc, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                           num_priv_explicit=NUM_PRIV_EXPLICIT, path=str(tmp_path))
    E = export_mod.OmniDepthEncoderExport(enc)
    P = export_mod.OmniDeployPolicyExport(depth_actor, est, num_prop=NUM_PROP, num_scan=NUM_SCAN,
                                          num_priv_explicit=NUM_PRIV_EXPLICIT)
    run_e, label = _onnx_runner(files["depth_onnx"])
    run_p, _ = _onnx_runner(files["policy_onnx"])
    print(f"[t4] ONNX runtime used: {label}")
    (d, prop, h), (obs, _lat) = export_mod.dummy_inputs(E, seed=1)
    assert d.shape == (1, 4, 58, 87) and prop.shape == (1, 89)
    for _ in range(2):
        with torch.no_grad():
            t_lat, t_h = E(d, prop, h)
            t_act = P(obs, t_lat)
        o_lat, o_h = run_e({"depth_image": d.numpy(), "proprioception": prop.numpy(), "h_in": h.numpy()})
        (o_act,) = run_p({"obs": obs.numpy(), "depth_latent": t_lat.numpy()})
        assert abs(o_lat - t_lat.numpy()).max() < 1e-5
        assert abs(o_h - t_h.numpy()).max() < 1e-5
        assert abs(o_act - t_act.numpy()).max() < 1e-5
        h = t_h


def test_exporter_entry_point_delegates(tmp_path, monkeypatch):
    """``scripts/rsl_rl/exporter.export_omni_student_deploy`` (play.py's call) writes the omni files."""
    stub_exp = types.ModuleType("isaaclab_rl.rsl_rl.exporter")
    stub_exp._TorchPolicyExporter = stub_exp._OnnxPolicyExporter = nn.Module
    monkeypatch.setitem(sys.modules, "isaaclab_rl", types.ModuleType("isaaclab_rl"))
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", types.ModuleType("isaaclab_rl.rsl_rl"))
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl.exporter", stub_exp)
    pkg = types.ModuleType("_t4_rsl_scripts")
    pkg.__path__ = [str(RSL)]
    monkeypatch.setitem(sys.modules, "_t4_rsl_scripts", pkg)
    exporter = importlib.import_module("_t4_rsl_scripts.exporter")
    _p, depth_actor, est, enc = _nets(seed=5)
    runner = SimpleNamespace(cfg=_train_cfg(omni=True, steps=10),
                             alg=SimpleNamespace(depth_actor=depth_actor, estimator=est, depth_encoder=enc))
    files = exporter.export_omni_student_deploy(runner, str(tmp_path), onnx=False)
    assert sorted(Path(f).name for f in files.values()) == ["depth_latest.pt", "omni_student_io.json", "policy.pt"]
    with pytest.raises(ValueError, match="omni student"):
        exporter.export_omni_student_deploy(SimpleNamespace(cfg=_train_cfg(omni=False, steps=10)), str(tmp_path))
