"""A resumed PPO run continues its algorithm state instead of restarting it (2026-09-22).

``PPOWithExtractor.counter`` drives the ``priv_reg`` coefficient (schedule [0, 0.1, 2000, 3000]:
0 until update 2000, 0.1 from update 5000). The runner used to save and restore only ``iter``, so
every resume -- every curriculum-chain stage, every ladder relaunch -- restarted the counter at 0 and
the adaptive learning rate at its configured value, and reset the estimator / history-encoder
(and, for the depth student, the depth-actor) Adam states. The omni ladder's X07 resumed at
model_6100 and logged ``priv_reg_coef`` 0.0 for its last 1900 iterations.

Pure torch on CPU, no Isaac: the runner is built without ``__init__`` (``save`` / ``load`` need
only the algorithm and a few attributes) around a stand-in policy with the attributes
``PPOWithExtractor.__init__`` reads.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("tensordict")
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODULES = REPO / "parkour" / "scripts" / "rsl_rl" / "modules"
PKG = "_krabby_rsl_rl_modules"

if str(REPO / "parkour") not in sys.path:
    sys.path.insert(0, str(REPO / "parkour"))  # parkour_isaaclab.utils.nonfinite_logging


def _load_modules():
    """Import ``parkour/scripts/rsl_rl/modules`` from this checkout under a private name."""
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
        importlib.import_module(f"{PKG}.distillation_with_extractor"),
    )


ppo_mod, runner_mod, dist_mod = _load_modules()
PPOWithExtractor = ppo_mod.PPOWithExtractor
updates_through_iteration = ppo_mod.updates_through_iteration
Runner = runner_mod.OnPolicyRunnerWithExtractor

SCHED = [0.0, 0.1, 2000.0, 3000.0]
DAGGER = 20
LR = 3.0e-4


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 2)
        self.history_encoder = nn.Linear(4, 3)


class _Policy(nn.Module):
    """The attributes ``PPOWithExtractor.__init__`` and the runner's save/load touch."""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = _Actor()
        self.critic = nn.Linear(4, 1)


def _alg(schedule: str = "adaptive") -> PPOWithExtractor:
    torch.manual_seed(0)
    return PPOWithExtractor(
        _Policy(),
        nn.Linear(4, 2),
        {
            "num_priv_explicit": 2,
            "num_prop": 4,
            "num_scan": 0,
            "learning_rate": 1.0e-4,
            "train_with_estimated_states": True,
        },
        learning_rate=LR,
        schedule=schedule,
        priv_reg_coef_schedual=SCHED,
    )


def _runner(alg, **cfg) -> Runner:
    r = Runner.__new__(Runner)
    r.alg = alg
    r.cfg = dict(cfg)
    r.training_type = "rl"
    r.dagger_update_freq = DAGGER
    r.depth_encoder_cfg = None
    r.empirical_normalization = False
    r.logger_type = "tensorboard"
    r.disable_logs = False
    r.current_learning_iteration = 0
    return r


def _train(alg: PPOWithExtractor, start: int, stop: int) -> int:
    """Advance the counter as ``learn_rl`` does for iterations ``start..stop-1`` (``update`` every
    iteration, ``update_dagger`` every ``DAGGER``-th) and dirty the auxiliary optimizers so their
    state is non-trivial. Returns the number of counter increments."""
    before = alg.counter
    for it in range(start, stop):
        alg.update_counter()
        if it % DAGGER == 0:
            alg.update_counter()
    for opt, params in (
        (alg.estimator_optimizer, alg.estimator.parameters()),
        (alg.hist_encoder_optimizer, alg.policy.actor.history_encoder.parameters()),
    ):
        opt.zero_grad()
        sum(p.sum() for p in params).backward()
        opt.step()
    return alg.counter - before


def _save(runner: Runner, it: int, path: Path) -> None:
    runner.current_learning_iteration = it
    runner.save(str(path))


def test_updates_through_iteration_matches_the_learn_loop():
    alg = _alg()
    for it in range(0, 6200):
        alg.update_counter()
        if it % DAGGER == 0:
            alg.update_counter()
        assert alg.counter == updates_through_iteration(it, DAGGER)


def test_priv_reg_coef_schedule_endpoints():
    alg = _alg()
    for counter, want in ((0, 0.0), (2000, 0.0), (3500, 0.05), (5000, 0.1), (9000, 0.1)):
        alg.counter = counter
        assert alg.priv_reg_coef() == pytest.approx(want)


def test_resume_restores_counter_lr_and_aux_optimizers(tmp_path):
    first = _alg()
    _train(first, 0, 6101)  # an uninterrupted run through model_6100 (X07)
    first.learning_rate = 1.0e-5  # the adaptive LR where these runs sit
    for g in first.optimizer.param_groups:
        g["lr"] = first.learning_rate
    ckpt = tmp_path / "model_6100.pt"
    _save(_runner(first), 6100, ckpt)

    resumed = _alg()
    runner = _runner(resumed)
    runner.load(str(ckpt))

    assert runner.current_learning_iteration == 6100
    assert resumed.counter == first.counter == updates_through_iteration(6100, DAGGER)
    assert resumed.priv_reg_coef() == pytest.approx(0.1)
    assert resumed.learning_rate == pytest.approx(1.0e-5)
    for name in ("estimator_optimizer", "hist_encoder_optimizer"):
        got = getattr(resumed, name).state_dict()["state"]
        want = getattr(first, name).state_dict()["state"]
        assert got.keys() == want.keys() and got
        for k in want:
            assert torch.equal(got[k]["exp_avg"], want[k]["exp_avg"])
            assert int(got[k]["step"]) == int(want[k]["step"])


def test_chain_of_resumes_carries_the_counter(tmp_path):
    """Six stages (the ladder's c1..c6 lengths), each a fresh runner resuming the previous
    stage's last checkpoint exactly as ``learn_rl`` + ``train.py --resume`` do: the counter
    carries across all of them, so the later stages run the scheduled coefficient (the old
    code ran every stage after c1 at 0.0)."""
    lengths = (1000, 1000, 1000, 1500, 1500, 2000)
    total = 0
    ckpt = None
    start = 0
    for n in lengths:
        alg = _alg()
        runner = _runner(alg)
        if ckpt is not None:
            runner.load(str(ckpt))
            assert alg.counter == total  # nothing restarted
        start = runner.current_learning_iteration
        total += _train(alg, start, start + n)  # a resume repeats the saved iteration
        ckpt = tmp_path / f"model_{start + n - 1}.pt"
        _save(runner, start + n - 1, ckpt)
    assert alg.counter == total
    assert total >= updates_through_iteration(start + lengths[-1] - 1, DAGGER)
    assert alg.priv_reg_coef() == pytest.approx(0.1)


def test_legacy_checkpoint_rebuilds_counter_from_iter_and_lr_from_optimizer(tmp_path):
    first = _alg()
    _train(first, 0, 1000)
    for g in first.optimizer.param_groups:
        g["lr"] = 1.5e-5
    ckpt = tmp_path / "legacy.pt"
    _save(_runner(first), 6100, ckpt)
    d = torch.load(ckpt, weights_only=False)
    del d["alg_state"]  # as written before 2026-09-22
    torch.save(d, ckpt)

    resumed = _alg()
    _runner(resumed).load(str(ckpt))
    assert resumed.counter == updates_through_iteration(6100, DAGGER)
    assert resumed.priv_reg_coef() == pytest.approx(0.1)
    assert resumed.learning_rate == pytest.approx(1.5e-5)
    assert resumed.estimator_optimizer.state_dict()["state"] == {}  # not in the file: fresh


def test_resume_alg_state_false_keeps_the_old_restart(tmp_path):
    first = _alg()
    _train(first, 0, 6101)
    first.learning_rate = 1.0e-5
    ckpt = tmp_path / "model_6100.pt"
    _save(_runner(first), 6100, ckpt)

    resumed = _alg()
    _runner(resumed, resume_alg_state=False).load(str(ckpt))
    assert resumed.counter == 0
    assert resumed.priv_reg_coef() == 0.0
    assert resumed.learning_rate == LR
    assert resumed.estimator_optimizer.state_dict()["state"] == {}


def test_fixed_schedule_keeps_the_configured_learning_rate(tmp_path):
    first = _alg(schedule="fixed")
    _train(first, 0, 10)
    first.learning_rate = 9.9e-4  # never happens under "fixed"; proves it is not restored
    ckpt = tmp_path / "m.pt"
    _save(_runner(first), 9, ckpt)
    resumed = _alg(schedule="fixed")
    _runner(resumed).load(str(ckpt))
    assert resumed.learning_rate == LR
    assert resumed.counter == first.counter


def test_distillation_ignores_teacher_state_and_restores_its_own():
    alg = dist_mod.DistillationWithExtractor.__new__(dist_mod.DistillationWithExtractor)
    net = nn.Linear(3, 2)
    alg.depth_actor_optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    alg.load_resume_state(_alg().resume_state())  # a teacher checkpoint's alg_state
    assert alg.depth_actor_optimizer.state_dict()["state"] == {}

    alg.depth_actor_optimizer.zero_grad()
    net.weight.sum().backward()
    alg.depth_actor_optimizer.step()
    state = alg.resume_state()

    fresh = dist_mod.DistillationWithExtractor.__new__(dist_mod.DistillationWithExtractor)
    fresh.depth_actor_optimizer = torch.optim.Adam(nn.Linear(3, 2).parameters(), lr=1e-3)
    fresh.load_resume_state(state)
    assert int(fresh.depth_actor_optimizer.state_dict()["state"][0]["step"]) == 1
