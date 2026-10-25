"""PPO-construction knobs and telemetry (PPO-construction plan step 2, 2026-10-01), CPU only.

Plan: ``~/.claude/plans/foamy-wondering-ullman.md`` steps 2-3 and 5. Pinned here:

* the action-bound loss (value, gradient, the coefficient-0 gate is bitwise the old update, the ramp
  keyed on ``alg.counter`` and continued across a resume);
* the ``soft`` std type (init / inverse / load round trip, finite gradient at both caps, the effective
  sigma in ``Policy/mean_noise_std`` and ``PPO/effective_std``);
* the always-on ``PPO/`` telemetry and the per-hand diagnostics leave parameters, ``.grad`` buffers,
  optimizer state and the RNG state bit-identical -- for the update they run in AND the next one;
* the step-size schedules (``fixed`` LR in every param group and after any load, ``epoch_kl`` at most
  one LR change per epoch, the KL early stops) and the history-encoder LR;
* fail-loud algorithm / policy keys and their allow-lists;
* the knob parsers, the runner-side applier and the campaign driver's dump audit.

A REAL ``PPOWithExtractor`` around a tiny real ``CrabHexActorCriticRMA`` (18 actions: 6 cams then 12
rods, like the robot) and a real ``RolloutStorage``, filled by a synthetic rollout under
``torch.inference_mode`` exactly as ``learn_rl`` fills it. No Isaac, no GPU.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
import math
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
tensordict = pytest.importorskip("tensordict")
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODULES = REPO / "parkour" / "scripts" / "rsl_rl" / "modules"
PKG = "_krabby_rsl_rl_modules"          # the same private package name test_ppo_resume_state.py uses
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
OMNI_OBS_PY = OMNI / "mdp/crab_hex_omni_observations.py"
RUN_PHASES_PY = OMNI / "experiments/tools/run_phases.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
EVAL_PY = TASKS / "crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
#: The MAIN checkout's training logs (read-only): real ``params/agent.yaml`` dumps and eval
#: ``run_meta.json`` files, used when present (a fresh worktree has no ``parkour/logs``).
MAIN_LOGS = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl")

if str(REPO / "parkour") not in sys.path:
    sys.path.insert(0, str(REPO / "parkour"))  # parkour_isaaclab.utils.nonfinite_logging


def _load_modules():
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PKG, MODULES / "__init__.py", submodule_search_locations=[str(MODULES)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = pkg
        spec.loader.exec_module(pkg)
    return (
        importlib.import_module(f"{PKG}.ppo_with_extractor"),
        importlib.import_module(f"{PKG}.on_policy_runner_with_extractor"),
        importlib.import_module(f"{PKG}.crab_actor_critic_with_encoder"),
        importlib.import_module(f"{PKG}.actor_critic_with_encoder"),
    )


ppo_mod, runner_mod, crab_mod, ac_mod = _load_modules()
PPO = ppo_mod.PPOWithExtractor
Runner = runner_mod.OnPolicyRunnerWithExtractor
CrabPolicy = crab_mod.CrabHexActorCriticRMA


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mk = _load(MOTION_KNOBS_PY, "motion_knobs__ppo_construction_test")

# ------------------------------------------------------------------ the tiny real policy + PPO
NUM_PROP, NUM_SCAN, NUM_PRIV_EXPLICIT, NUM_PRIV_LATENT, NUM_HIST = 12, 4, 2, 3, 10   # history encoder: 10/20/50
OBS_W = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + NUM_PROP * NUM_HIST
N_ACT = 18
VY_COL = 9                      # the omni lateral-command column (pinned against the obs term below)
N_ENVS, N_STEPS = 16, 6


def _policy_kwargs(**extra):
    base = dict(num_prop=NUM_PROP, num_scan=NUM_SCAN, num_hist=NUM_HIST, num_priv_latent=NUM_PRIV_LATENT,
                num_priv_explicit=NUM_PRIV_EXPLICIT)
    kw = dict(
        init_noise_std=1.5, noise_std_type="scalar", actor_hidden_dims=[16], critic_hidden_dims=[16],
        activation="elu", tanh_encoder_output=False, scan_encoder_dims=[4, 2], priv_encoder_dims=[4],
        actor=dict(**base, class_name="Actor",
                   state_history_encoder=dict(**base, class_name="StateHistoryEncoder", channel_size=2)),
    )
    kw.update(extra)
    return kw


def _mirror(obs=None, actions=None, env=None, obs_type="policy"):
    """A stand-in two-image augmentation (negation) so the update runs its mirror-loss path."""
    obs_aug = None if obs is None else torch.cat([obs, -obs], dim=0)
    act_aug = None if actions is None else torch.cat([actions, -actions], dim=0)
    return obs_aug, act_aug


def _alg(seed: int = 0, *, policy_kw: dict | None = None, symmetry: bool = True, mu_bias: float | None = None,
         **alg_kw) -> PPO:
    torch.manual_seed(seed)
    policy = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(**(policy_kw or {})))
    if mu_bias is not None:      # park the mean outside the bound so the hinge is live
        with torch.no_grad():
            policy.actor.actor_backbone[-1].bias.fill_(mu_bias)
    estimator = nn.Linear(NUM_PROP, NUM_PRIV_EXPLICIT)
    kw = dict(num_learning_epochs=2, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
              desired_kl=0.01, entropy_coef=0.01, max_grad_norm=1.0)
    if symmetry:
        kw["symmetry_cfg"] = {"use_data_augmentation": False, "use_mirror_loss": True,
                              "data_augmentation_func": _mirror, "mirror_loss_coeff": 0.5, "_env": None}
    kw.update(alg_kw)
    alg = PPO(policy, estimator,
              {"num_priv_explicit": NUM_PRIV_EXPLICIT, "num_prop": NUM_PROP, "num_scan": NUM_SCAN,
               "learning_rate": 1.0e-3, "train_with_estimated_states": True},
              **kw)
    alg.init_storage("rl", N_ENVS, N_STEPS, [OBS_W], [OBS_W], [N_ACT])
    return alg


def _obs(gen: torch.Generator) -> torch.Tensor:
    obs = torch.randn(N_ENVS, OBS_W, generator=gen)
    # the lateral command: 6 left (+y), 6 right (-y), 4 inside the dead zone
    obs[:, VY_COL] = torch.tensor([0.22] * 6 + [-0.22] * 6 + [0.01] * 4)
    return obs


def _rollout(alg: PPO, gen: torch.Generator) -> None:
    """One ``learn_rl`` rollout + ``compute_returns`` on synthetic data from ``gen`` (the policy's
    own sampling draws from the GLOBAL RNG, exactly as in training)."""
    with torch.inference_mode():
        obs = _obs(gen)
        for _ in range(N_STEPS):
            alg.act(obs, obs, hist_encoding=False)
            obs = _obs(gen)
            rewards = torch.randn(N_ENVS, generator=gen)
            dones = (torch.rand(N_ENVS, generator=gen) < 0.15).long()
            td = tensordict.TensorDict({"policy": obs, "critic": obs}, batch_size=[N_ENVS])
            alg.process_env_step(td, rewards, dones, {})
        alg.compute_returns(obs)


def _snapshot(alg: PPO) -> dict:
    out = {"rng": torch.get_rng_state().clone(), "lr": alg.learning_rate, "counter": alg.counter}
    for tag, module in (("policy", alg.policy), ("estimator", alg.estimator)):
        for name, p in module.named_parameters():
            out[f"{tag}.{name}"] = p.detach().clone()
            out[f"{tag}.{name}.grad"] = None if p.grad is None else p.grad.detach().clone()
    for oname in ("optimizer", "estimator_optimizer", "hist_encoder_optimizer"):
        sd = getattr(alg, oname).state_dict()
        for idx, st in sd["state"].items():
            for k, v in st.items():
                out[f"{oname}.{idx}.{k}"] = v.clone() if torch.is_tensor(v) else v
        out[f"{oname}.lrs"] = [g["lr"] for g in sd["param_groups"]]
    return out


def _assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if torch.is_tensor(va) or torch.is_tensor(vb):
            assert va is not None and vb is not None and torch.equal(va, vb), k
        else:
            assert va == vb, k


def _runner(alg, **cfg):
    r = Runner.__new__(Runner)
    r.alg = alg
    r.cfg = dict(cfg)
    r.training_type = "rl"
    r.dagger_update_freq = 20
    r.depth_encoder_cfg = None
    r.empirical_normalization = False
    r.logger_type = "tensorboard"
    r.disable_logs = False
    r.current_learning_iteration = 0
    return r


# ================================================================== the action-bound loss
def test_action_bound_loss_value_and_gradient():
    b = 1.1
    inside = torch.tensor([[0.0, 1.1, -1.1], [0.5, -0.3, 1.0]], requires_grad=True)
    loss = ppo_mod.action_bound_loss(inside, b)
    loss.backward()
    assert loss.item() == 0.0 and torch.equal(inside.grad, torch.zeros_like(inside))
    mu = torch.tensor([[2.1, -3.1, 0.0], [1.6, 0.0, -1.1]], requires_grad=True)
    loss = ppo_mod.action_bound_loss(mu, b)
    want = ((1.0 ** 2 + 2.0 ** 2) + 0.5 ** 2) / 2          # summed over dims, averaged over rows
    assert loss.item() == pytest.approx(want)
    loss.backward()
    excess = (mu.detach().abs() - b).clamp(min=0) * mu.detach().sign()
    assert torch.allclose(mu.grad, 2.0 * excess / 2)        # d/dmu = 2 (|mu| - b) sign(mu) / rows


def test_bound_loss_at_coefficient_zero_is_bitwise_the_old_update():
    """Armed with a ramp from c0 = 0, the first update's effective coefficient is 0: the update must
    be the knob-off update bit for bit (the term is computed and logged, never added)."""
    gen_a, gen_b = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    off = _alg(mu_bias=3.0)
    _rollout(off, gen_a)
    off_losses = off.update()
    snap_off = _snapshot(off)
    on = _alg(mu_bias=3.0, action_bound_loss_coef=1e-3, action_bound_loss_ramp=[0.0, 1e-3, 500])
    _rollout(on, gen_b)
    on_losses = on.update()
    _assert_same(snap_off, _snapshot(on))
    assert on_losses["action_bound_coef"] == 0.0 and on_losses["action_bound"] > 0.0   # mu ~ 3 > 1.1
    assert "action_bound" not in off_losses and "action_bound_coef" not in off_losses
    assert {k: v for k, v in on_losses.items() if not k.startswith("action_bound")} == off_losses
    # ... and the term is LIVE once the ramp leaves 0: the next update differs and pulls mu in
    _rollout(off, gen_a)
    _rollout(on, gen_b)
    off.update()
    on_losses2 = on.update()
    assert on_losses2["action_bound_coef"] == pytest.approx(1e-3 * 1 / 500)
    assert not torch.equal(on.policy.actor.actor_backbone[-1].bias, off.policy.actor.actor_backbone[-1].bias)


def test_bound_loss_constant_coefficient_pulls_the_mean_toward_the_bound():
    gen_a, gen_b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    off = _alg(mu_bias=3.0, symmetry=False, entropy_coef=0.0)
    on = _alg(mu_bias=3.0, symmetry=False, entropy_coef=0.0, action_bound_loss_coef=1.0)
    for alg, gen in ((off, gen_a), (on, gen_b)):
        _rollout(alg, gen)
        alg.update()
    b_off = off.policy.actor.actor_backbone[-1].bias.detach()
    b_on = on.policy.actor.actor_backbone[-1].bias.detach()
    assert bool((b_on < 3.0).all())                        # every dim pulled down toward 1.1
    assert float(b_on.mean()) < float(b_off.mean())


def test_bound_ramp_is_keyed_on_the_counter_and_survives_a_resume(tmp_path):
    alg = _alg(action_bound_loss_coef=1e-3, action_bound_loss_ramp=(1e-4, 1e-3, 500))
    for counter, want in ((0, 1e-4), (250, 5.5e-4), (500, 1e-3), (5000, 1e-3)):
        alg.counter = counter
        assert alg.action_bound_coef() == pytest.approx(want)
    const = _alg(action_bound_loss_coef=2e-3)
    const.counter = 17
    assert const.action_bound_coef() == 2e-3
    # a resume continues the ramp (counter restored), it never restarts it
    alg.counter = 300
    ckpt = tmp_path / "m.pt"
    r = _runner(alg)
    r.current_learning_iteration = 299
    r.save(str(ckpt))
    fresh = _alg(action_bound_loss_coef=1e-3, action_bound_loss_ramp=(1e-4, 1e-3, 500))
    assert fresh.action_bound_coef() == pytest.approx(1e-4)
    _runner(fresh).load(str(ckpt))
    assert fresh.counter == 300 and fresh.action_bound_coef() == pytest.approx(1e-4 + 0.9e-3 * 300 / 500)


@pytest.mark.parametrize("kw", [
    dict(action_bound_loss_coef=-1e-3),
    dict(action_bound_loss_coef=1e-3, action_bound=0.0),
    dict(action_bound_loss_coef=1e-3, action_bound_loss_ramp=(1e-4, 2e-3, 500)),     # c1 != coef
    dict(action_bound_loss_coef=0.0, action_bound_loss_ramp=(1e-4, 1e-3, 500)),
    dict(action_bound_loss_coef=1e-3, action_bound_loss_ramp=(1e-4, 1e-3, 0)),
    dict(action_bound_loss_coef=1e-3, action_bound_loss_ramp=(1e-4, 1e-3)),
    dict(schedule="bogus"),
    dict(schedule="epoch_kl", desired_kl=None),
    dict(schedule="adaptive", kl_stop=0.02),
    dict(schedule="fixed", kl_stop=0.0),
    dict(hist_encoder_learning_rate=0.0),
])
def test_ppo_constructor_refuses_inconsistent_construction_knobs(kw):
    with pytest.raises(ValueError):
        _alg(**kw)


# ================================================================== the soft std
@pytest.mark.parametrize("sigma", [0.051, 0.1, 0.4, 0.8, 1.0, 1.5, 1.99])
def test_soft_std_init_is_the_exact_inverse(sigma):
    x = crab_mod.soft_std_inverse(sigma)
    assert float(crab_mod.soft_std(torch.tensor(x, dtype=torch.float64))) == pytest.approx(sigma, rel=1e-12)
    pol = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(noise_std_type="soft", init_noise_std=sigma))
    assert torch.allclose(pol.effective_std(), torch.full((N_ACT,), sigma), rtol=1e-6)
    assert not hasattr(pol, "std") and not hasattr(pol, "log_std")
    # the same registration slot as the scalar ``std`` (so the optimizer's param order matches)
    scalar = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs())
    assert [n for n, _ in pol.named_parameters()] == [
        "std_raw" if n == "std" else n for n, _ in scalar.named_parameters()]


def test_soft_std_matches_the_tanh_formula_and_refuses_the_caps():
    x = torch.linspace(-6.0, 6.0, 101, dtype=torch.float64)
    lo, hi = math.log(0.05), math.log(2.0)
    want = lo + 0.5 * (hi - lo) * (torch.tanh(x) + 1.0)
    assert torch.allclose(crab_mod.soft_log_std(x), want, atol=1e-12)
    for bad in (0.05, 2.0, 2.5, 0.01, float("nan")):
        with pytest.raises(ValueError):
            crab_mod.soft_std_inverse(bad)
    with pytest.raises(ValueError):
        CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(noise_std_type="soft", init_noise_std=2.0))


@pytest.mark.parametrize("x0", [-30.0, -12.0, -3.0, 0.0, 3.0, 12.0, 30.0])
def test_soft_std_gradient_is_finite_and_nonzero_everywhere(x0):
    x = torch.tensor([x0], requires_grad=True)          # float32, as trained
    crab_mod.soft_std(x).sum().backward()
    g = float(x.grad)
    assert math.isfinite(g) and g > 0.0, (x0, g)
    sigma = float(crab_mod.soft_std(x.detach()))
    assert 0.05 - 1e-6 <= sigma <= 2.0 + 1e-6           # exp(ln 0.05) rounds in float32
    if abs(x0) >= 12.0:
        # the literal tanh form is EXACTLY flat there in float32 -- why soft_log_std is split
        y = torch.tensor([x0], requires_grad=True)
        torch.exp(math.log(0.05) + 0.5 * (math.log(2.0) - math.log(0.05)) * (torch.tanh(y) + 1.0)).sum().backward()
        assert float(y.grad) == 0.0


def test_soft_std_has_no_clamp_and_its_load_round_trips():
    pol = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(noise_std_type="soft", init_noise_std=0.8))
    with torch.no_grad():
        pol.std_raw.copy_(torch.linspace(-4.0, 4.0, N_ACT))
    pol.update_distribution(torch.zeros(3, OBS_W), False)
    assert torch.equal(pol.distribution.stddev[0], crab_mod.soft_std(pol.std_raw).detach())
    other = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(noise_std_type="soft", init_noise_std=1.2))
    assert other.load_state_dict(pol.state_dict()) is True
    assert torch.equal(other.std_raw, pol.std_raw) and torch.equal(other.effective_std(), pol.effective_std())
    raw = other.std_raw
    other.reset_action_std(0.4)                          # written through the inverse, in place
    assert other.std_raw is raw and torch.allclose(other.effective_std(), torch.full((N_ACT,), 0.4), rtol=1e-6)
    # structure is not interchangeable: a scalar-std checkpoint does not load into a soft policy
    scalar = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs())
    with pytest.raises(RuntimeError):
        other.load_state_dict(scalar.state_dict())
    with pytest.raises(RuntimeError):
        scalar.load_state_dict(pol.state_dict())
    for kw in (dict(action_std_max=0.5), dict(action_std_reset=0.4)):
        with pytest.raises(ValueError, match="soft"):
            CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(noise_std_type="soft", init_noise_std=0.8, **kw))


def test_soft_std_trains_through_ppo_and_is_logged_as_the_effective_sigma():
    gen = torch.Generator().manual_seed(5)
    alg = _alg(policy_kw=dict(noise_std_type="soft", init_noise_std=0.8))
    raw0 = alg.policy.std_raw.detach().clone()
    _rollout(alg, gen)
    alg.update()
    assert not torch.equal(alg.policy.std_raw.detach(), raw0)           # the gradient reached it
    assert alg.telemetry["effective_std"] == pytest.approx(0.8, rel=1e-5)  # the rollout's sigma
    writer = _Writer()
    _log(alg, writer)
    assert writer.scalars["Policy/mean_noise_std"] == pytest.approx(float(alg.policy.effective_std().mean()))
    assert writer.scalars["PPO/effective_std"] == pytest.approx(0.8, rel=1e-5)


def test_scalar_effective_std_is_the_clamped_std():
    pol = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(action_std_max=0.5))
    with torch.no_grad():
        pol.std.fill_(1.7)
    assert torch.equal(pol.effective_std(), torch.full((N_ACT,), 0.5))


# ================================================================== knob-off == the pre-change code
#: The last commit that touched ``parkour/scripts/rsl_rl/modules`` before the PPO-construction work.
PRE_CONSTRUCTION_SHA = "eabe73245223031958f2c7addd7c97f94701ecae"


@pytest.fixture(scope="module")
def pre_construction(tmp_path_factory):
    """``parkour/scripts/rsl_rl/modules`` as of :data:`PRE_CONSTRUCTION_SHA`, under a private name."""
    import io
    import shutil
    import subprocess
    import tarfile

    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    proc = subprocess.run([git, "-C", str(REPO), "archive", PRE_CONSTRUCTION_SHA, "parkour/scripts/rsl_rl/modules"],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"commit {PRE_CONSTRUCTION_SHA[:8]} is not in this clone")
    out = tmp_path_factory.mktemp("pre_construction")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        try:
            tar.extractall(out, filter="data")
        except TypeError:                                # a Python without extraction filters
            tar.extractall(out)
    mods = out / "parkour/scripts/rsl_rl/modules"
    name = "_krabby_pre_construction_modules"
    spec = importlib.util.spec_from_file_location(name, mods / "__init__.py", submodule_search_locations=[str(mods)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return (importlib.import_module(f"{name}.ppo_with_extractor").PPOWithExtractor,
            importlib.import_module(f"{name}.crab_actor_critic_with_encoder").CrabHexActorCriticRMA)


@pytest.mark.parametrize("schedule", ["adaptive", "fixed"])
def test_knob_off_update_is_bitwise_the_pre_construction_update(pre_construction, schedule):
    """The plan's knob-off identity, on CPU: the new code -- telemetry ON, every knob at its default --
    against the pre-change PPO + policy, same seeds, three updates each followed by a DAgger update
    (mirror loss on): parameters, ``.grad`` buffers, every optimizer state, LR, counter, RNG state and
    the loss dicts are bit-identical."""
    old_ppo, old_policy = pre_construction
    runs = []
    for ppo_cls, policy_cls in ((old_ppo, old_policy), (PPO, CrabPolicy)):
        torch.manual_seed(0)
        policy = policy_cls(OBS_W, N_ACT, **_policy_kwargs())
        alg = ppo_cls(policy, nn.Linear(NUM_PROP, NUM_PRIV_EXPLICIT),
                      {"num_priv_explicit": NUM_PRIV_EXPLICIT, "num_prop": NUM_PROP, "num_scan": NUM_SCAN,
                       "learning_rate": 1.0e-3, "train_with_estimated_states": True},
                      num_learning_epochs=3, num_mini_batches=4, learning_rate=3.0e-4, schedule=schedule,
                      desired_kl=0.01, entropy_coef=0.01, max_grad_norm=1.0,
                      symmetry_cfg={"use_data_augmentation": False, "use_mirror_loss": True,
                                    "data_augmentation_func": _mirror, "mirror_loss_coeff": 0.5, "_env": None})
        alg.init_storage("rl", N_ENVS, N_STEPS, [OBS_W], [OBS_W], [N_ACT])
        gen = torch.Generator().manual_seed(11)
        run = []
        for _ in range(3):
            _rollout(alg, gen)
            losses = alg.update()
            dagger = alg.update_dagger()
            run.append((_snapshot(alg), losses, dagger))
        runs.append(run)
    assert runs[1][0][0] and getattr(alg, "telemetry", None)          # the new code ran its telemetry
    for (a, la, da), (b, lb, db) in zip(*runs):
        _assert_same(a, b)
        assert la == lb and da == db


# ================================================================== telemetry + hand diagnostics
EXPECTED_TELEMETRY = {"sat1_cam", "sat15_cam", "mu_med_cam", "lock_cam", "sat1_rod", "sat15_rod", "mu_med_rod",
                      "lock_rod", "effective_std", "exec_action_rate", "explained_var", "exec_change", "kl_mean",
                      "kl_mb0", "clip_frac", "lr_applied", "grad_norm", "grad_clip_share", "skipped_updates"}
HAND_TAGS = {"hand_frac_l", "hand_frac_r", "hand_adv_mean_l", "hand_adv_mean_r", "hand_grad_cos"}


def _two_updates(*, telemetry: bool, hand_diag: bool, **alg_kw) -> tuple[list[dict], list[dict]]:
    gen = torch.Generator().manual_seed(11)
    alg = _alg(mu_bias=1.3, hand_diag_vy_obs_col=VY_COL if hand_diag else None, **alg_kw)
    alg.telemetry_enabled = telemetry
    snaps, tels = [], []
    for _ in range(2):
        _rollout(alg, gen)
        alg.update()
        snaps.append(_snapshot(alg))
        tels.append(dict(alg.telemetry))
    return snaps, tels


@pytest.mark.parametrize("alg_kw", [{}, dict(schedule="fixed"), dict(schedule="epoch_kl"),
                                    dict(action_bound_loss_coef=1e-3)])
def test_telemetry_and_hand_diag_leave_training_bit_identical(alg_kw):
    """Same seed, same data: with telemetry + hand diagnostics vs with neither, the parameters,
    the ``.grad`` buffers, every optimizer state and the global RNG state are bit-identical after
    the update they ran in AND after the next one (which samples its rollout from that RNG)."""
    snaps_on, tels_on = _two_updates(telemetry=True, hand_diag=True, **alg_kw)
    snaps_off, tels_off = _two_updates(telemetry=False, hand_diag=False, **alg_kw)
    for a, b in zip(snaps_on, snaps_off):
        _assert_same(a, b)
    assert tels_off == [{}, {}]
    assert EXPECTED_TELEMETRY <= set(tels_on[0]) and EXPECTED_TELEMETRY <= set(tels_on[1])
    assert HAND_TAGS <= set(tels_on[0])                 # counter 0: fires at the first update
    assert not HAND_TAGS & set(tels_on[1])              # counter 1: not a multiple of 25
    for tel in tels_on:
        assert all(math.isfinite(v) for v in tel.values()), tel
        assert not any("symmetry" in k or "mirror" in k for k in tel), tel


def test_hand_diag_is_rng_free_and_touches_no_grad_or_optimizer_state():
    gen = torch.Generator().manual_seed(2)
    alg = _alg(hand_diag_vy_obs_col=VY_COL)
    _rollout(alg, gen)
    for p in alg.policy.parameters():                    # pre-existing .grad buffers must survive
        p.grad = torch.full_like(p, 0.25)
    before = _snapshot(alg)
    dist = alg.policy.distribution
    out = {}
    alg._hand_diag(out)
    _assert_same(before, _snapshot(alg))
    assert alg.policy.distribution is dist               # the rollout's distribution object is restored
    assert set(out) == HAND_TAGS and -1.0 <= out["hand_grad_cos"] <= 1.0
    obs = alg._slice_obs()
    adv = alg._stored_rows(alg.storage.advantages)
    left, right = obs[:, VY_COL] > 0.05, obs[:, VY_COL] < -0.05
    assert out["hand_frac_l"] == pytest.approx(float(left.float().mean()))
    assert out["hand_frac_r"] == pytest.approx(float(right.float().mean()))
    assert out["hand_adv_mean_l"] == pytest.approx(float(adv[left].mean()))
    assert out["hand_adv_mean_r"] == pytest.approx(float(adv[right].mean()))
    # the cosine is the surrogate gradients' on the shared backbone, recomputed independently
    params = list(alg.policy.actor.actor_backbone.parameters())
    acts = alg._stored_rows(alg.storage.actions)
    logp = alg._stored_rows(alg.storage.actions_log_prob)

    def grad(mask):
        alg.policy.update_distribution(obs[mask], False)
        ratio = torch.exp(alg.policy.get_actions_log_prob(acts[mask]) - logp[mask].squeeze(-1))
        a = adv[mask].squeeze(-1)
        s = torch.max(-a * ratio, -a * ratio.clamp(0.8, 1.2)).mean()
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(s, params)])

    gl, gr = grad(left), grad(right)
    assert out["hand_grad_cos"] == pytest.approx(float(gl @ gr / (gl.norm() * gr.norm())), rel=1e-5)


def test_hand_diag_fires_on_the_runners_iteration_not_the_counter():
    """F7: ``update_dagger`` advances ``counter`` too, so ``counter % 25`` drifts off the 25-iteration
    cadence; the runner's iteration keys it instead (the counter only without a runner)."""
    gen = torch.Generator().manual_seed(4)
    fired = []
    for it, counter in ((0, 0), (25, 27), (26, 25), (50, 52), (49, 50)):
        alg = _alg(hand_diag_vy_obs_col=VY_COL)
        alg.counter = counter                            # a counter ahead of the iteration (DAgger updates)
        alg.iteration = it
        _rollout(alg, gen)
        alg.update()
        fired.append((it, bool(HAND_TAGS & set(alg.telemetry))))
    assert fired == [(0, True), (25, True), (26, False), (50, True), (49, False)]
    src = (MODULES / "on_policy_runner_with_extractor.py").read_text()
    assert src.index("self.alg.iteration = it") < src.index("loss_dict = self.alg.update()")


def test_hist_lr_tag_reads_the_history_encoder_optimizer_only_when_pinned():
    gen = torch.Generator().manual_seed(5)
    pinned = _alg(learning_rate=1e-4, schedule="fixed", hist_encoder_learning_rate=3e-4)
    _rollout(pinned, gen)
    pinned.update()
    assert pinned.telemetry["hist_lr"] == pytest.approx(3e-4) and pinned.telemetry["lr_applied"] == pytest.approx(1e-4)
    unset = _alg(learning_rate=1e-4)
    _rollout(unset, gen)
    unset.update()
    assert "hist_lr" not in unset.telemetry


def test_hand_diag_vy_column_is_the_omni_lateral_command_slot():
    tree = ast.parse(OMNI_OBS_PY.read_text())
    slots = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OBSERVED_COMMAND_SLOTS"
                                                for t in node.targets):
            slots = ast.literal_eval(node.value)
    assert slots is not None
    vy_slot = [slot for slot, cmd_idx in slots.items() if cmd_idx == 1]
    assert vy_slot == [mk.HAND_DIAG_VY_OBS_COL] == [VY_COL]
    assert VY_COL < NUM_PROP                            # inside the current step's proprio head


def test_action_layout_is_six_cams_then_twelve_rods():
    assert (ppo_mod.CRAB_ACTION_DIM, ppo_mod.CRAB_CAM_DIMS) == (18, 6)
    metas = sorted(MAIN_LOGS.glob("gait_eval/omni_phases/*/*/suite__omni/seed001/*/run_meta.json"))
    if not metas:
        pytest.skip("no omni eval run_meta.json on this machine")
    names = json.loads(metas[-1].read_text())["action_joint_names"]
    assert len(names) == 18
    assert all("_Body_CamShaft_" in n for n in names[:6])
    assert not any("_Body_CamShaft_" in n for n in names[6:])


def test_rollout_telemetry_values_on_a_hand_built_storage():
    alg = _alg()
    st = alg.storage
    T, N = st.num_transitions_per_env, st.num_envs
    mu = torch.zeros(T, N, N_ACT)
    mu[..., :6] = 2.0                                    # every cam saturated past 1.5
    mu[..., 6:] = 0.5                                    # no rod saturated
    mu[0, 0, 6] = 1.2                                    # one rod past 1 but not 1.5
    sigma = torch.full((T, N, N_ACT), 0.25)
    st.distribution_params = (mu, sigma)
    st.actions.zero_()
    st.actions[1:, :, 0] = 3.0                           # executed +1 from step 1 on (clip)
    st.dones.zero_()
    st.dones[0, :N // 2] = 1                             # half the envs reset after step 0
    out = {}
    alg._rollout_telemetry(out)
    assert out["sat1_cam"] == out["sat15_cam"] == 1.0 and out["mu_med_cam"] == 2.0
    assert out["lock_cam"] == 1.0                        # 2.0 > 1 + 2 * 0.25
    assert out["sat1_rod"] == pytest.approx(1 / (T * N * 12)) and out["sat15_rod"] == 0.0
    assert out["mu_med_rod"] == 0.5 and out["lock_rod"] == 0.0
    assert out["effective_std"] == pytest.approx(0.25)
    # pairs (t-1, t): only (0, 1) moves (by 1.0, executed), and it is excluded for the reset half
    n_pairs = (T - 1) * N - N // 2
    assert out["exec_action_rate"] == pytest.approx((N - N // 2) * 1.0 / n_pairs)


def test_explained_variance_and_exec_change():
    r = torch.randn(64, 1)
    assert ppo_mod._explained_variance(r, torch.zeros_like(r)) == pytest.approx(1.0)   # values == returns
    assert ppo_mod._explained_variance(r, r) == pytest.approx(0.0)                     # values == 0
    assert math.isnan(ppo_mod._explained_variance(torch.ones(8, 1), torch.zeros(8, 1)))
    gen = torch.Generator().manual_seed(4)
    frozen = _alg(schedule="fixed", learning_rate=0.0)   # Adam at LR 0: the policy cannot move
    _rollout(frozen, gen)
    frozen.update()
    assert frozen.telemetry["exec_change"] == 0.0 and frozen.telemetry["lr_applied"] == 0.0
    live = _alg(mu_bias=0.2)
    _rollout(live, torch.Generator().manual_seed(4))
    live.update()
    assert live.telemetry["exec_change"] > 0.0
    assert 0.0 <= live.telemetry["clip_frac"] <= 1.0 and 0.0 <= live.telemetry["grad_clip_share"] <= 1.0
    assert live.telemetry["skipped_updates"] == 0.0 and live.telemetry["kl_mean"] >= 0.0


# ================================================================== runner logging
class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = float(value)


def _log(alg, writer):
    r = _runner(alg)
    r.writer = writer
    r.num_steps_per_env = N_STEPS
    r.env = types.SimpleNamespace(num_envs=N_ENVS)
    r.gpu_world_size = 1
    r.tot_timesteps = 0
    r.tot_time = 0.0
    locs = {"collection_time": 1.0, "learn_time": 1.0, "loss_dict": {"surrogate": 0.0}, "it": 0,
            "tot_iter": 10, "start_iter": 0, "ep_infos": [], "rewbuffer": [], "lenbuffer": []}
    r.log(locs)
    return r


def test_runner_logs_ppo_telemetry_tags_and_prints_them(capsys):
    gen = torch.Generator().manual_seed(9)
    alg = _alg(hand_diag_vy_obs_col=VY_COL)
    _rollout(alg, gen)
    alg.update()
    writer = _Writer()
    _log(alg, writer)
    printed = capsys.readouterr().out
    ppo_tags = {k for k in writer.scalars if k.startswith("PPO/")}
    assert ppo_tags == {f"PPO/{k}" for k in alg.telemetry}
    assert {f"PPO/{k}" for k in EXPECTED_TELEMETRY | HAND_TAGS} <= ppo_tags
    for k in alg.telemetry:
        assert f"PPO/{k}:" in printed
    assert writer.scalars["Policy/mean_noise_std"] == pytest.approx(float(alg.policy.std.mean()))  # raw, as before
    assert not any(("symmetry" in k or "mirror" in k) for k in ppo_tags)
    assert not any(("symmetry" in line or "mirror" in line) for line in printed.splitlines() if "PPO/" in line)


# ================================================================== step-size schedules
def test_fixed_schedule_writes_its_lr_into_every_param_group_and_survives_resume(tmp_path):
    src = _alg(learning_rate=3e-4)                       # an adaptive head at its LR floor
    src.learning_rate = 1e-5
    src._write_policy_lr()
    ckpt = tmp_path / "head.pt"
    _runner(src).save(str(ckpt))
    for resume_alg_state in (True, False):
        alg = _alg(schedule="fixed", learning_rate=1e-4)
        assert all(g["lr"] == 1e-4 for g in alg.optimizer.param_groups)
        _runner(alg, resume_alg_state=resume_alg_state).load(str(ckpt))
        assert alg.learning_rate == 1e-4
        assert all(g["lr"] == 1e-4 for g in alg.optimizer.param_groups), resume_alg_state
    # ... and it stays there through an update (logged as the applied LR)
    gen = torch.Generator().manual_seed(1)
    _rollout(alg, gen)
    alg.update()
    assert alg.telemetry["lr_applied"] == pytest.approx(1e-4, rel=1e-12)
    assert all(g["lr"] == 1e-4 for g in alg.optimizer.param_groups)


def test_adaptive_resume_is_unchanged_and_epoch_kl_restores_its_lr(tmp_path):
    src = _alg()
    src.learning_rate = 2.5e-5
    src._write_policy_lr()
    ckpt = tmp_path / "head.pt"
    _runner(src).save(str(ckpt))
    adaptive = _alg()
    _runner(adaptive).load(str(ckpt))
    assert adaptive.learning_rate == 2.5e-5
    ekl = _alg(schedule="epoch_kl", learning_rate=3e-4)
    _runner(ekl).load(str(ckpt))
    assert ekl.learning_rate == 2.5e-5 and all(g["lr"] == 2.5e-5 for g in ekl.optimizer.param_groups)
    fresh = _alg(schedule="epoch_kl", learning_rate=3e-4)
    _runner(fresh, resume_alg_state=False).load(str(ckpt))
    assert fresh.learning_rate == 3e-4 and all(g["lr"] == 3e-4 for g in fresh.optimizer.param_groups)


def test_hist_encoder_lr_knob_and_its_unset_default(tmp_path):
    unset = _alg(learning_rate=1e-4)
    assert [g["lr"] for g in unset.hist_encoder_optimizer.param_groups] == [1e-4]
    pinned = _alg(learning_rate=1e-4, schedule="fixed", hist_encoder_learning_rate=3e-4)
    assert [g["lr"] for g in pinned.hist_encoder_optimizer.param_groups] == [3e-4]
    assert [g["lr"] for g in pinned.optimizer.param_groups] == [1e-4]
    ckpt = tmp_path / "m.pt"
    _runner(unset).save(str(ckpt))                       # its history encoder saved at 1e-4
    _runner(pinned).load(str(ckpt))
    assert [g["lr"] for g in pinned.hist_encoder_optimizer.param_groups] == [3e-4]   # re-written
    other = _alg(learning_rate=5e-4)
    _runner(other).load(str(ckpt))
    assert [g["lr"] for g in other.hist_encoder_optimizer.param_groups] == [1e-4]    # old behaviour: loaded


def _scripted_kl(monkeypatch, values):
    """Replace the minibatch KL with a script (one value per minibatch, in order)."""
    seq = iter(values)
    monkeypatch.setattr(ppo_mod, "_gaussian_kl", lambda *a, **k: torch.tensor(float(next(seq))))


def _step_lrs(alg):
    lrs = []
    alg.optimizer.register_step_pre_hook(lambda opt, args, kwargs: lrs.append(opt.param_groups[0]["lr"]))
    return lrs


def test_epoch_kl_changes_the_lr_at_most_once_per_epoch(monkeypatch):
    alg = _alg(schedule="epoch_kl", desired_kl=0.01, learning_rate=3e-4, num_learning_epochs=3, num_mini_batches=4)
    _rollout(alg, torch.Generator().manual_seed(6))
    lrs = _step_lrs(alg)
    _scripted_kl(monkeypatch, [0.05] * 12)              # every epoch mean > 2 * desired -> /1.5
    alg.update()
    assert lrs == [3e-4] * 4 + [3e-4 / 1.5] * 4 + [3e-4 / 1.5 ** 2] * 4
    assert alg.learning_rate == pytest.approx(3e-4 / 1.5 ** 3)
    assert all(g["lr"] == alg.learning_rate for g in alg.optimizer.param_groups)
    # a quiet epoch raises it once; an in-band epoch leaves it; the floor and cap hold
    alg2 = _alg(schedule="epoch_kl", desired_kl=0.01, learning_rate=1e-4, num_learning_epochs=3, num_mini_batches=4)
    _rollout(alg2, torch.Generator().manual_seed(6))
    lrs2 = _step_lrs(alg2)
    _scripted_kl(monkeypatch, [0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.001, 0.03, 0.03, 0.03])
    alg2.update()
    assert lrs2 == [1e-4] * 4 + [pytest.approx(1.5e-4)] * 8
    assert alg2.learning_rate == pytest.approx(1e-4)    # epoch 3's mean 0.02275 > 2 * 0.01: back down, once
    floor = _alg(schedule="epoch_kl", desired_kl=0.01, learning_rate=1.2e-5)
    _rollout(floor, torch.Generator().manual_seed(6))
    _scripted_kl(monkeypatch, [1.0] * 8)
    floor.update()
    assert floor.learning_rate == ppo_mod.LR_FLOOR


def test_adaptive_still_changes_the_lr_per_minibatch(monkeypatch):
    alg = _alg(schedule="adaptive", desired_kl=0.01, learning_rate=3e-4)
    _rollout(alg, torch.Generator().manual_seed(6))
    lrs = _step_lrs(alg)
    _scripted_kl(monkeypatch, [0.05] * 8)
    alg.update()
    assert lrs == [pytest.approx(3e-4 / 1.5 ** (i + 1)) for i in range(8)]


def test_fixed_kl_stop_hard_stop_epoch_stop_and_no_stop(monkeypatch):
    def run(values, **kw):
        alg = _alg(schedule="fixed", learning_rate=1e-4, kl_stop=0.02, num_learning_epochs=3, num_mini_batches=4, **kw)
        _rollout(alg, torch.Generator().manual_seed(8))
        lrs = _step_lrs(alg)
        _scripted_kl(monkeypatch, values)
        losses = alg.update()
        return alg, lrs, losses

    # hard stop: minibatch 3 (KL 0.05 > 2 * 0.02) takes no step and ends the update
    alg, lrs, losses = run([0.01, 0.01, 0.05] + [0.0] * 9)
    assert len(lrs) == 2 and alg.telemetry["kl_stop"] == 1.0 and alg.telemetry["epochs_done"] == 0.0
    assert all(math.isfinite(v) for v in losses.values())
    # per-epoch stop: epoch 1's mean 0.025 > 0.02 (no minibatch above 0.04) -> epochs 2-3 skipped
    alg, lrs, _ = run([0.01, 0.03, 0.03, 0.03] + [0.0] * 8)
    assert len(lrs) == 4 and alg.telemetry["kl_stop"] == 1.0 and alg.telemetry["epochs_done"] == 1.0
    # no stop: every epoch in band, all 12 steps at the fixed LR
    alg, lrs, _ = run([0.015] * 12)
    assert lrs == [1e-4] * 12 and alg.telemetry["kl_stop"] == 0.0 and alg.telemetry["epochs_done"] == 3.0
    # a LAST epoch over the target has nothing left to skip: not a stop
    alg, lrs, _ = run([0.0] * 8 + [0.03] * 4)
    assert len(lrs) == 12 and alg.telemetry["kl_stop"] == 0.0
    # without kl_stop the fixed schedule logs neither tag
    monkeypatch.undo()
    plain = _alg(schedule="fixed")
    _rollout(plain, torch.Generator().manual_seed(8))
    plain.update()
    assert "kl_stop" not in plain.telemetry and "epochs_done" not in plain.telemetry


# ================================================================== fail-loud keys
REAL_ALG_KEYS = ("class_name", "num_learning_epochs", "num_mini_batches", "learning_rate", "schedule", "gamma",
                 "lam", "entropy_coef", "desired_kl", "max_grad_norm", "optimizer", "value_loss_coef",
                 "use_clipped_value_loss", "clip_param", "normalize_advantage_per_mini_batch",
                 "share_cnn_encoders", "rnd_cfg", "symmetry_cfg", "dagger_update_freq", "priv_reg_coef_schedual")
NEW_ALG_KEYS = ("action_bound_loss_coef", "action_bound", "action_bound_loss_ramp", "kl_stop",
                "hist_encoder_learning_rate", "hand_diag_vy_obs_col")


def _real_agent_yamls(limit: int = 40) -> list:
    files = sorted(MAIN_LOGS.glob("crab_hex_*/*/params/agent.yaml"), key=lambda p: p.stat().st_mtime)
    return files[-limit:]


def test_unknown_algorithm_keys_raise_and_the_allow_list_is_dropped_with_a_warning(caplog):
    cfg = {k: None for k in REAL_ALG_KEYS if k not in ("class_name", "dagger_update_freq")}
    cfg.update({k: None for k in NEW_ALG_KEYS})
    with caplog.at_level(logging.WARNING):
        out = runner_mod.filter_ppo_alg_cfg(PPO, cfg)
    assert set(out) == set(cfg) - {"share_cnn_encoders"}
    assert "Ignoring unsupported PPO config keys for PPOWithExtractor: share_cnn_encoders" in caplog.text
    for typo in ("action_bound_loss_coeff", "kl_stopp", "hand_diag"):
        with pytest.raises(ValueError, match=typo):
            runner_mod.filter_ppo_alg_cfg(PPO, {**cfg, typo: 1})
    assert runner_mod.ALG_CFG_IGNORED_KEYS == ("share_cnn_encoders",)


def test_every_recent_real_algorithm_and_policy_block_is_accepted():
    files = _real_agent_yamls()
    if not files:
        pytest.skip("no training dumps on this machine")
    yaml = pytest.importorskip("yaml")
    for f in files:
        d = yaml.unsafe_load(f.read_text())
        alg = dict(d["algorithm"])
        name = alg.pop("class_name")
        if name == "PPOWithExtractor":
            alg.pop("dagger_update_freq")
            runner_mod.filter_ppo_alg_cfg(PPO, alg)             # raises on an unknown key
        else:
            assert runner_mod.warn_unused_distillation_alg_cfg(types.SimpleNamespace(__name__=name), alg)
        pol = dict(d["policy"])
        pol.pop("class_name")
        leftover = set(pol) - {"actor", "init_noise_std", "noise_std_type", "actor_hidden_dims",
                               "critic_hidden_dims", "activation", "action_std_max", "action_std_reset"}
        assert leftover <= set(ac_mod.POLICY_KWARGS_ALLOWED), (f, leftover)
    # ... and the newest three policy blocks really construct (the fail-loud check runs in __init__)
    for f in files[-3:]:
        pol = dict(yaml.unsafe_load(f.read_text())["policy"])
        assert pol.pop("class_name") == "CrabHexActorCriticRMA"
        CrabPolicy(1149, 18, **pol)


def test_distillation_warns_about_the_keys_it_drops(caplog):
    cfg = {"learning_rate": 1e-3, "max_grad_norm": 1.0, "clip_param": 0.2, "schedule": "adaptive"}
    with caplog.at_level(logging.WARNING):
        unused = runner_mod.warn_unused_distillation_alg_cfg(types.SimpleNamespace(__name__="Distill"), cfg)
    assert unused == ["clip_param", "schedule"]
    assert "Ignoring algorithm config keys Distill does not use: clip_param, schedule" in caplog.text
    assert runner_mod.warn_unused_distillation_alg_cfg(
        types.SimpleNamespace(__name__="Distill"), {"learning_rate": 1e-3, "max_grad_norm": 1.0}) == []


def test_unknown_policy_kwargs_raise_and_the_isaaclab_fields_are_accepted():
    pol = CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(state_dependent_std=False, actor_obs_normalization={},
                                                    critic_obs_normalization={}))
    assert pol.noise_std_type == "scalar"
    for bad in ("noise_std", "init_std", "action_bound"):
        with pytest.raises(ValueError, match=bad):
            CrabPolicy(OBS_W, N_ACT, **_policy_kwargs(**{bad: 1.0}))
    with pytest.raises(ValueError, match="bogus"):
        ac_mod.ActorCriticRMA(OBS_W, N_ACT, **_policy_kwargs(bogus=True))
    assert set(ac_mod.POLICY_KWARGS_ALLOWED) == {"priv_encoder_dims", "scan_encoder_dims", "tanh_encoder_output",
                                                 "state_dependent_std", "actor_obs_normalization",
                                                 "critic_obs_normalization"}


def test_every_runner_cfg_policy_block_passes_the_allow_list():
    """The forward / sideways / omni / student runner cfg SOURCES name no policy field outside the
    allow-list (the dumps above cover what actually trained; this covers the cfg classes)."""
    fields = set()
    for path in TASKS.glob("*/config/**/agents/*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "").endswith("ActorCriticCfg"):
                fields.update(kw.arg for kw in node.keywords if kw.arg)
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "policy" and isinstance(node.ctx, ast.Store)):
                fields.add(node.attr)
    known = {"actor", "init_noise_std", "noise_std_type", "actor_hidden_dims", "critic_hidden_dims", "activation",
             "action_std_max", "action_std_reset"}
    assert fields - known <= set(ac_mod.POLICY_KWARGS_ALLOWED), fields - known


# ================================================================== knob parsers + applier
class _Bag:
    pass


def _agent(schedule="adaptive"):
    agent = _Bag()
    agent.policy, agent.algorithm = _Bag(), _Bag()
    agent.algorithm.schedule = schedule
    return agent


def test_knob_names_and_registration():
    assert mk.PPO_CONSTRUCTION_KNOBS == (
        "KRABBY_ACTION_BOUND_LOSS", "KRABBY_ACTION_BOUND_RAMP", "KRABBY_ACTION_BOUND", "KRABBY_STD_PARAM",
        "KRABBY_INIT_STD", "KRABBY_PPO_SCHEDULE", "KRABBY_PPO_LR", "KRABBY_PPO_DESIRED_KL", "KRABBY_PPO_KL_STOP",
        "KRABBY_PPO_HIST_LR", "KRABBY_PPO_HAND_DIAG", "KRABBY_NUM_ENVS")
    assert set(mk.PPO_CONSTRUCTION_KNOBS) <= set(mk.MOTION_KNOB_PARSERS)
    assert set(mk.PPO_CONSTRUCTION_KNOBS) - {"KRABBY_NUM_ENVS"} <= set(mk.AGENT_KNOBS)
    assert "KRABBY_NUM_ENVS" not in mk.AGENT_KNOBS
    for name in ("parse_bound_loss", "parse_bound_ramp", "parse_action_bound", "parse_std_param", "parse_init_std",
                 "parse_ppo_schedule", "parse_positive", "parse_num_envs", "PPO_CONSTRUCTION_KNOBS"):
        assert name in mk.__all__
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__ppo_construction_test")
    exported = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}
    assert set(mk.PPO_CONSTRUCTION_KNOBS) <= exported
    assert knobs.MOTION_KNOB_PARSERS is knobs._mk.MOTION_KNOB_PARSERS


def test_parsers():
    assert mk.parse_bound_loss("") is None and mk.parse_bound_loss("0") is None and mk.parse_bound_loss("1e-3") == 1e-3
    with pytest.raises(ValueError):
        mk.parse_bound_loss("-1e-3")
    assert mk.parse_bound_ramp("1e-4:1e-3:500") == (1e-4, 1e-3, 500) and mk.parse_bound_ramp(" ") is None
    assert mk.parse_bound_ramp("0:1e-3:1") == (0.0, 1e-3, 1)
    for bad in ("1e-4:1e-3", "1e-4:0:500", "-1:1e-3:500", "1e-4:1e-3:0", "1e-4:1e-3:5.5", "a:b:c"):
        with pytest.raises(ValueError):
            mk.parse_bound_ramp(bad)
    assert mk.parse_action_bound("1.1") == 1.1 and mk.parse_action_bound("") is None
    for bad in ("0", "-1", "nan"):
        with pytest.raises(ValueError):
            mk.parse_action_bound(bad)
    assert mk.parse_std_param("soft") == "soft" and mk.parse_std_param(" SOFT ") == "soft"
    for off in ("", "0", "off", None):
        assert mk.parse_std_param(off) is None
    for bad in ("scalar", "log", "1"):
        with pytest.raises(ValueError):
            mk.parse_std_param(bad)
    assert mk.parse_init_std("0.8") == 0.8 and mk.parse_init_std("0") is None and mk.parse_init_std("") is None
    for bad in ("0.05", "-1", "inf"):
        with pytest.raises(ValueError):
            mk.parse_init_std(bad)
    for s in ("adaptive", "fixed", "epoch_kl"):
        assert mk.parse_ppo_schedule(s) == s
    assert mk.parse_ppo_schedule("") is None
    with pytest.raises(ValueError):
        mk.parse_ppo_schedule("Fixed")
    assert mk.parse_positive("1e-4", "X") == 1e-4 and mk.parse_positive("", "X") is None
    with pytest.raises(ValueError):
        mk.parse_positive("0", "X")
    assert mk.parse_num_envs("1024") == 1024 and mk.parse_num_envs("") is None
    for bad in ("0", "1.5", "-2", "many"):
        with pytest.raises(ValueError):
            mk.parse_num_envs(bad)
    assert mk.MOTION_KNOB_PARSERS["KRABBY_PPO_HAND_DIAG"]("1") is True
    with pytest.raises(ValueError):
        mk.MOTION_KNOB_PARSERS["KRABBY_PPO_HAND_DIAG"]("yes")


OFF_ENVS = (
    {},
    {k: "" for k in mk.PPO_CONSTRUCTION_KNOBS},
    {"KRABBY_ACTION_BOUND_LOSS": "0", "KRABBY_STD_PARAM": "0", "KRABBY_INIT_STD": "0", "KRABBY_PPO_HAND_DIAG": "0"},
    {"KRABBY_STD_PARAM": "off", "KRABBY_ACTION_BOUND_RAMP": "  ", "KRABBY_NUM_ENVS": "1024"},   # NUM_ENVS: driver-only
)


@pytest.mark.parametrize("env", OFF_ENVS)
def test_every_ppo_knob_off_writes_nothing_and_prints_nothing(env, capsys):
    agent = _agent()
    mk.apply_exploration_knobs(agent, env)
    assert vars(agent.policy) == {} and vars(agent.algorithm) == {"schedule": "adaptive"}
    assert capsys.readouterr().out == ""


def test_armed_knobs_write_exact_values_and_print_consumed_lines(capsys):
    env = {"KRABBY_ACTION_BOUND_LOSS": "1e-3", "KRABBY_ACTION_BOUND_RAMP": "1e-4:1e-3:500", "KRABBY_ACTION_BOUND": "1.1",
           "KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8", "KRABBY_ENTROPY_COEF": "0.0025",
           "KRABBY_PPO_HAND_DIAG": "1"}
    agent = _agent()
    mk.apply_exploration_knobs(agent, env)
    assert vars(agent.policy) == {"noise_std_type": "soft", "init_noise_std": 0.8}
    assert vars(agent.algorithm) == {"schedule": "adaptive", "entropy_coef": 0.0025, "action_bound_loss_coef": 1e-3,
                                     "action_bound_loss_ramp": [1e-4, 1e-3, 500], "action_bound": 1.1,
                                     "hand_diag_vy_obs_col": 9}
    lines = capsys.readouterr().out.splitlines()
    assert sorted(lines) == sorted(f"[KNOB] {k}={v} consumed" for k, v in env.items())
    fixed = _agent()
    mk.apply_exploration_knobs(fixed, {"KRABBY_PPO_SCHEDULE": "fixed", "KRABBY_PPO_LR": "1e-4",
                                       "KRABBY_PPO_KL_STOP": "0.02", "KRABBY_PPO_HIST_LR": "3e-4"})
    assert vars(fixed.algorithm) == {"schedule": "fixed", "learning_rate": 1e-4, "kl_stop": 0.02,
                                     "hist_encoder_learning_rate": 3e-4}
    ekl = _agent()
    mk.apply_exploration_knobs(ekl, {"KRABBY_PPO_SCHEDULE": "epoch_kl", "KRABBY_PPO_DESIRED_KL": "0.0025"})
    assert vars(ekl.algorithm) == {"schedule": "epoch_kl", "desired_kl": 0.0025}


@pytest.mark.parametrize("env,match", [
    ({"KRABBY_ACTION_BOUND_RAMP": "1e-4:1e-3:500"}, "RAMP"),                                   # no loss
    ({"KRABBY_ACTION_BOUND_LOSS": "2e-3", "KRABBY_ACTION_BOUND_RAMP": "1e-4:1e-3:500"}, "RAMP"),  # c1 mismatch
    ({"KRABBY_ACTION_BOUND": "1.2"}, "BOUND"),
    ({"KRABBY_STD_PARAM": "soft", "KRABBY_ACTION_STD_MAX": "0.5"}, "soft"),
    ({"KRABBY_STD_PARAM": "soft", "KRABBY_ACTION_STD_RESET": "0.4"}, "soft"),
    ({"KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "2.0"}, "soft band"),
    ({"KRABBY_INIT_STD": "2.5"}, "ceiling"),
    ({"KRABBY_ACTION_STD_MAX": "0.5", "KRABBY_INIT_STD": "0.8"}, "ceiling"),
    ({"KRABBY_PPO_KL_STOP": "0.02"}, "fixed"),                                                 # cfg is adaptive
    ({"KRABBY_PPO_SCHEDULE": "epoch_kl", "KRABBY_PPO_KL_STOP": "0.02"}, "fixed"),
    ({"KRABBY_PPO_SCHEDULE": "fixed", "KRABBY_PPO_DESIRED_KL": "0.01"}, "no effect"),
])
def test_cross_knob_refusals(env, match):
    agent = _agent()
    with pytest.raises(ValueError, match=match):
        mk.apply_exploration_knobs(agent, env)
    assert vars(agent.policy) == {} and vars(agent.algorithm) == {"schedule": "adaptive"}   # nothing written


def test_kl_stop_reads_the_cfg_schedule_when_the_schedule_knob_is_unset():
    agent = _agent(schedule="fixed")
    mk.apply_exploration_knobs(agent, {"KRABBY_PPO_KL_STOP": "0.02"})
    assert agent.algorithm.kl_stop == 0.02


def test_the_applied_cfg_builds_the_algorithm_and_policy(tmp_path):
    """What the applier writes is exactly what the constructors accept (the fail-loud runner would
    refuse anything else): the armed A-arm recipe builds and runs one update."""
    agent = _agent()
    mk.apply_exploration_knobs(agent, {"KRABBY_ACTION_BOUND_LOSS": "1e-3", "KRABBY_ACTION_BOUND_RAMP": "1e-4:1e-3:500",
                                       "KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8",
                                       "KRABBY_PPO_HAND_DIAG": "1"})
    alg_cfg = {k: v for k, v in vars(agent.algorithm).items()}
    filtered = runner_mod.filter_ppo_alg_cfg(PPO, alg_cfg)
    alg = _alg(policy_kw=vars(agent.policy), **filtered)
    assert alg.action_bound_loss_ramp == (1e-4, 1e-3, 500) and alg.hand_diag_vy_obs_col == 9
    assert torch.allclose(alg.policy.effective_std(), torch.full((N_ACT,), 0.8), rtol=1e-6)
    _rollout(alg, torch.Generator().manual_seed(1))
    losses = alg.update()
    assert losses["action_bound_coef"] == pytest.approx(1e-4) and "hand_grad_cos" in alg.telemetry


# ================================================================== eval carry + dump audit
@pytest.fixture(scope="module")
def rp():
    return _load(RUN_PHASES_PY, "run_phases__ppo_construction_test")


@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__ppo_construction_test")


def test_eval_carries_the_policy_structure_knobs_at_their_off_values(rp):
    assert set(mk.EVAL_STRUCTURE_KNOBS) == {"KRABBY_STD_PARAM", "KRABBY_INIT_STD"}
    assert set(mk.EVAL_STRUCTURE_KNOBS) <= set(rp.EVAL_CARRY)
    for key in mk.EVAL_STRUCTURE_KNOBS:
        assert rp.EVAL_CARRY_DEFAULTS[key] == mk.KNOB_OFF_VALUES[key] == "0"
    agent = _agent()
    mk.apply_exploration_knobs(agent, {k: rp.EVAL_CARRY_DEFAULTS[k] for k in mk.EVAL_STRUCTURE_KNOBS})
    assert vars(agent.policy) == {} and vars(agent.algorithm) == {"schedule": "adaptive"}
    # every OTHER construction knob shapes the trainer only and is never carried
    for key in set(mk.PPO_CONSTRUCTION_KNOBS) - set(mk.EVAL_STRUCTURE_KNOBS):
        assert key not in rp.EVAL_CARRY and key not in rp.EVAL_CARRY_DEFAULTS
    ev = rp.eval_env_for(rp.ph.PHASES["1c"], "A15+B", carry_overlay={"KRABBY_STD_PARAM": "soft",
                                                                    "KRABBY_INIT_STD": "0.8",
                                                                    "KRABBY_ACTION_BOUND_LOSS": "1e-3"})
    assert ev["KRABBY_STD_PARAM"] == "soft" and ev["KRABBY_INIT_STD"] == "0.8"
    assert "KRABBY_ACTION_BOUND_LOSS" not in ev


def test_eval_sample_actions_reads_the_effective_std_for_soft():
    src = EVAL_PY.read_text()
    i = src.index("if args_cli.sample_actions:\n            if is_student")
    block = src[i:i + 2000]
    assert 'getattr(_pol_module, "noise_std_type", None) == "soft"' in block
    assert "_pol_module.effective_std()" in block


AGENT_YAML_OFF = textwrap.dedent("""\
    algorithm:
      class_name: PPOWithExtractor
      num_learning_epochs: 5
      learning_rate: 0.0003
      schedule: adaptive
      entropy_coef: 0.01
      desired_kl: 0.01
      symmetry_cfg:
        mirror_loss_coeff: 0.5
      dagger_update_freq: 20
    policy:
      class_name: CrabHexActorCriticRMA
      init_noise_std: 1.5
      noise_std_type: scalar
      state_dependent_std: false
    estimator:
      num_prop: 75
      learning_rate: 0.0001
    """)

AGENT_YAML_ON = textwrap.dedent("""\
    algorithm:
      class_name: PPOWithExtractor
      num_learning_epochs: 5
      learning_rate: 0.0001
      schedule: fixed
      entropy_coef: 0.0025
      desired_kl: 0.0025
      symmetry_cfg:
        mirror_loss_coeff: 0.5
      dagger_update_freq: 20
      action_bound_loss_coef: 0.001
      action_bound_loss_ramp:
      - 0.0001
      - 0.001
      - 500
      action_bound: 1.1
      kl_stop: 0.02
      hist_encoder_learning_rate: 0.0003
      hand_diag_vy_obs_col: 9
    policy:
      class_name: CrabHexActorCriticRMA
      init_noise_std: 0.8
      noise_std_type: soft
      state_dependent_std: false
    estimator:
      num_prop: 75
      learning_rate: 0.0001
    """)

ENV_YAML = textwrap.dedent("""\
    scene:
      num_envs: {n}
      env_spacing: 1.0
      terrain:
        prim_path: /World/ground
        num_envs: 256
    rewards:
      reward_action_rate:
        weight: -0.3
    """)

ON_OVERLAY = {"KRABBY_ACTION_BOUND_LOSS": "0.001", "KRABBY_ACTION_BOUND_RAMP": "1e-4:1e-3:500",
              "KRABBY_ACTION_BOUND": "1.1", "KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8",
              "KRABBY_PPO_SCHEDULE": "fixed", "KRABBY_PPO_LR": "1e-4", "KRABBY_PPO_KL_STOP": "0.02",
              "KRABBY_PPO_DESIRED_KL": "0.0025",
              "KRABBY_PPO_HIST_LR": "3e-4", "KRABBY_PPO_HAND_DIAG": "1", "KRABBY_NUM_ENVS": "1024"}
OFF_OVERLAY = {"KRABBY_ACTION_BOUND_LOSS": "0", "KRABBY_ACTION_BOUND_RAMP": "", "KRABBY_ACTION_BOUND": "",
               "KRABBY_STD_PARAM": "0", "KRABBY_INIT_STD": "0", "KRABBY_PPO_SCHEDULE": "",
               "KRABBY_PPO_LR": "", "KRABBY_PPO_KL_STOP": "", "KRABBY_PPO_HIST_LR": "", "KRABBY_PPO_DESIRED_KL": "",
               "KRABBY_PPO_HAND_DIAG": "0", "KRABBY_NUM_ENVS": ""}


def _train_log(overlay: dict, *, bound_line: bool = True, extra: str = "") -> str:
    """A training log as the armed window writes it: consumed lines (the applier's, the driver's),
    the algorithm's construction line, and an iteration block."""
    lines = [f"[KNOB] {k}={str(v).strip()} consumed" for k, v in overlay.items()
             if mk.MOTION_KNOB_PARSERS[k](v) not in (None, False)]
    lines.append("[PPO] construction: schedule=fixed, action_bound_loss_coef=0.001")
    lines.append("                      Mean surrogate: -0.0012")
    if bound_line:
        lines.append("                  Mean action_bound: 3.2100")
        lines.append("             Mean action_bound_coef: 0.0001")
    return "\n".join(lines) + "\n" + extra


def _run_dir(tmp_path, agent_yaml, n_envs, log: str | None = None):
    params = tmp_path / "params"
    params.mkdir(parents=True)
    (params / "env.yaml").write_text(ENV_YAML.format(n=n_envs))
    (params / "agent.yaml").write_text(agent_yaml)
    ckpt = tmp_path / "model_19.pt"
    ckpt.write_text("")
    rec = {"ckpt": str(ckpt)}
    if log is not None:
        (tmp_path / "train.log").write_text(log)
        rec["log"] = str(tmp_path / "train.log")
    return rec


def test_every_construction_knob_is_registered_with_the_ladder(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in mk.PPO_CONSTRUCTION_KNOBS:
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
        reasons = []
        rc._check_knob(var, ON_OVERLAY[var], "t", whitelist, parsers, reasons, set())
        assert reasons == [], reasons
    assert set(mk.PPO_CONSTRUCTION_KNOBS) - {"KRABBY_NUM_ENVS"} <= rc.AGENT_DUMP_KEYS
    assert "KRABBY_NUM_ENVS" not in rc.AGENT_DUMP_KEYS


def test_dump_audit_on_values_against_an_armed_dump(rc, tmp_path):
    checks = rc.overlay_checks(_run_dir(tmp_path, AGENT_YAML_ON, 1024, _train_log(ON_OVERLAY)), ON_OVERLAY)
    # one dump check per key, one consumed-line check per armed key, the dropped-key check, the loss line
    assert len(checks) == 2 * len(ON_OVERLAY) + 2
    assert all(ok for _n, ok in checks), [n for n, ok in checks if not ok]
    assert "overlay KRABBY_NUM_ENVS=1024 in params/env.yaml" in dict(checks)
    assert "overlay KRABBY_PPO_LR=1e-4 in params/agent.yaml" in dict(checks)


def test_dump_audit_off_values_against_a_knob_free_dump(rc, tmp_path):
    checks = rc.overlay_checks(_run_dir(tmp_path, AGENT_YAML_OFF, 256), OFF_OVERLAY)
    assert all(ok for _n, ok in checks), [n for n, ok in checks if not ok]


def test_dump_audit_catches_every_knob_that_did_not_arrive(rc, tmp_path):
    checks = dict(rc.overlay_checks(_run_dir(tmp_path, AGENT_YAML_OFF, 256), ON_OVERLAY))
    assert checks.pop("training log readable for the PPO-construction wiring audit (None)") is False
    assert not any(checks.values()), [n for n, ok in checks.items() if ok]
    on_dir = _run_dir(tmp_path / "on", AGENT_YAML_ON, 1024)
    off_vs_on = dict(rc.overlay_checks(on_dir, OFF_OVERLAY))
    # the presence-only off checks (schedule / LR / init std / env count always exist) are true in
    # both dumps; every ABSENCE claim fails against the armed dump
    assert {n for n, ok in off_vs_on.items() if ok} == {
        "overlay KRABBY_PPO_SCHEDULE= in params/agent.yaml", "overlay KRABBY_PPO_LR= in params/agent.yaml",
        "overlay KRABBY_PPO_DESIRED_KL= in params/agent.yaml",
        "overlay KRABBY_INIT_STD=0 in params/agent.yaml", "overlay KRABBY_NUM_ENVS= in params/env.yaml"}


def test_off_overlay_needs_no_training_log(rc, tmp_path):
    assert rc.construction_log_checks({}, OFF_OVERLAY) == []
    assert rc.construction_log_checks({}, {"KRABBY_ENTROPY_COEF": "0.0025"}) == []   # not a construction knob


def test_training_log_audit_catches_a_dropped_or_unconsumed_knob(rc, tmp_path):
    overlay = {"KRABBY_ACTION_BOUND_LOSS": "1e-3", "KRABBY_STD_PARAM": "soft", "KRABBY_NUM_ENVS": "1024"}
    good = dict(rc.construction_log_checks({"log": str(_write(tmp_path / "a.log", _train_log(overlay)))}, overlay))
    assert all(good.values()) and len(good) == 5, good
    # a consumed line missing (the knob never reached the runner cfg in that process)
    log = _train_log(overlay).replace("[KNOB] KRABBY_STD_PARAM=soft consumed\n", "")
    bad = dict(rc.construction_log_checks({"log": str(_write(tmp_path / "b.log", log))}, overlay))
    assert bad == {**good, "log: [KNOB] KRABBY_STD_PARAM consumed": False}
    # F6: a "NOT consumed" line for the knob is not its consumed line
    log = _train_log(overlay).replace("[KNOB] KRABBY_STD_PARAM=soft consumed\n",
                                      "[KNOB] KRABBY_STD_PARAM=soft NOT consumed: some reason\n")
    bad = dict(rc.construction_log_checks({"log": str(_write(tmp_path / "b2.log", log))}, overlay))
    assert bad == {**good, "log: [KNOB] KRABBY_STD_PARAM consumed": False}
    # the M1b import trap: a stale runner dropped the field behind its warning
    log = _train_log(overlay, extra="WARNING Ignoring unsupported PPO config keys for PPOWithExtractor: "
                                    "action_bound_loss_coef, share_cnn_encoders\n")
    bad = dict(rc.construction_log_checks({"log": str(_write(tmp_path / "c.log", log))}, overlay))
    assert bad["log: no PPO-construction field dropped by the runner"] is False
    # the routine share_cnn_encoders warning alone is fine
    log = _train_log(overlay, extra="WARNING Ignoring unsupported PPO config keys for PPOWithExtractor: "
                                    "share_cnn_encoders\n")
    assert all(dict(rc.construction_log_checks({"log": str(_write(tmp_path / "d.log", log))}, overlay)).values())
    # the bound loss never printed (it did not run)
    log = _train_log(overlay, bound_line=False)
    bad = dict(rc.construction_log_checks({"log": str(_write(tmp_path / "e.log", log))}, overlay))
    assert bad["log: Mean action_bound printed (the bound loss ran)"] is False
    assert sorted(rc.CONSTRUCTION_ALG_FIELDS) == sorted(NEW_ALG_KEYS)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_dump_audit_scopes_learning_rate_and_num_envs_to_their_own_blocks(rc):
    check_lr = rc.OVERLAY_DUMP_CHECKS["KRABBY_PPO_LR"]
    assert check_lr(AGENT_YAML_ON, "1e-4") and not check_lr(AGENT_YAML_ON, "0.0001001")
    # the estimator's learning_rate (0.0001) must not satisfy an algorithm-LR claim on the OFF dump
    assert not check_lr(AGENT_YAML_OFF, "1e-4")
    check_n = rc.OVERLAY_DUMP_CHECKS["KRABBY_NUM_ENVS"]
    assert check_n(ENV_YAML.format(n=1024), "1024") and not check_n(ENV_YAML.format(n=1024), "256")
    assert not check_n(ENV_YAML.format(n=512), "256")    # the terrain block's 256 is not read
    ramp = rc.OVERLAY_DUMP_CHECKS["KRABBY_ACTION_BOUND_RAMP"]
    assert ramp(AGENT_YAML_ON, "1e-4:1e-3:500") and not ramp(AGENT_YAML_ON, "1e-4:1e-3:400")
    bound = rc.OVERLAY_DUMP_CHECKS["KRABBY_ACTION_BOUND"]
    assert bound(AGENT_YAML_ON, "1.1") and not bound(AGENT_YAML_ON, "1.2")      # not action_bound_loss_coef


#: the construction knobs were built and smoked on 2026-10-01 (ppoc_smk_* runs that morning; committed in 60532fb9 at
#: 10:30), so a run dir named before that day is knob-off by construction. Since the side-first B recipe was baked
#: (2026-10-06) every newer omni run trains WITH knobs (STD_PARAM soft, ACTION_BOUND*, PPO_HAND_DIAG), so "the newest
#: dump" is no longer a knob-off dump -- and the audit rightly fails the OFF overlay on it.
KNOBS_INTRODUCED_RUN_PREFIX = "2026-10-01"


def test_dump_audit_off_values_pass_on_a_real_dump(rc, tmp_path):
    files = [f for f in sorted(MAIN_LOGS.glob("crab_hex_omni*/*/params/agent.yaml"), key=lambda p: p.stat().st_mtime)
             if f.parent.parent.name < KNOBS_INTRODUCED_RUN_PREFIX]
    if not files:
        pytest.skip("no pre-knob (knob-off) omni training dumps on this machine")
    agent_txt = files[-1].read_text()
    env_txt = (files[-1].parent / "env.yaml").read_text()
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text(agent_txt)
    (params / "env.yaml").write_text(env_txt)
    (tmp_path / "model_1.pt").write_text("")
    checks = rc.overlay_checks({"ckpt": str(tmp_path / "model_1.pt")}, OFF_OVERLAY)
    assert all(ok for _n, ok in checks), [n for n, ok in checks if not ok]
