"""Unit tests pinning ParkourRewardManager.compute() behavior that the gait-income
phase-out campaign depends on (PLAN G, 2026-08-31):

1. A term at weight exactly 0.0 is skipped entirely -- its func is never called and
   its Episode_Reward telemetry stays flat. (This is why annealed terms must floor
   at epsilon, never 0.0: the campaign's gait gate metric IS the term's episode sum.)
2. A term at the epsilon floor (1e-3) still computes, contributes to the reward
   buffer, and accumulates episodic telemetry.
3. The total step reward is clipped at >= 0 (legged-gym style). Removing positive
   income can therefore park the net at the clip floor where all gradients die --
   the campaign's income-budget tables exist to keep a margin above this floor.

Pure torch -- no Isaac Sim. The real isaaclab/omni modules are used when importable
(isaac venv); otherwise minimal stubs let the production module import so compute()
runs verbatim.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_isaaclab" / "managers" / "parkour_reward_manager.py"
)


def _install_stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_manager_class():
    try:  # pragma: no cover - exercised only inside the isaac venv
        import isaaclab.managers  # noqa: F401
        import omni.kit.app  # noqa: F401
    except ImportError:
        if "omni.kit.app" not in sys.modules:
            omni = _install_stub("omni")
            kit = _install_stub("omni.kit")
            app = _install_stub("omni.kit.app")
            omni.kit = kit
            kit.app = app
        if "isaaclab.managers" not in sys.modules:
            isaaclab = _install_stub("isaaclab")
            managers = _install_stub("isaaclab.managers", RewardManager=object)
            isaaclab.managers = managers

    import importlib.util

    spec = importlib.util.spec_from_file_location("parkour_reward_manager_under_test", MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ParkourRewardManager


ParkourRewardManager = _load_manager_class()

NUM_ENVS = 4
DT = 0.02


class _TermCfg:
    def __init__(self, weight, func, params=None):
        self.weight = weight
        self.func = func
        self.params = params or {}


class _CountingFunc:
    """Reward func stub returning a constant per-env value and counting calls."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, env, **params):
        self.calls += 1
        return torch.full((NUM_ENVS,), self.value)


def _make_manager(term_specs):
    """Build a ParkourRewardManager without RewardManager.__init__ (no sim needed).

    compute() only touches the private buffers set up here, so the production code
    path at parkour_reward_manager.py:18-39 runs verbatim.
    """
    manager = ParkourRewardManager.__new__(ParkourRewardManager)
    manager._env = None
    manager._term_names = [name for name, _ in term_specs]
    manager._term_cfgs = [cfg for _, cfg in term_specs]
    manager._reward_buf = torch.zeros(NUM_ENVS)
    manager._step_reward = torch.zeros(NUM_ENVS, len(term_specs))
    manager._episode_sums = {name: torch.zeros(NUM_ENVS) for name, _ in term_specs}
    return manager


def test_zero_weight_term_is_skipped_and_telemetry_stays_flat():
    zero_func = _CountingFunc(5.0)
    live_func = _CountingFunc(1.0)
    manager = _make_manager([
        ("zeroed_gait_term", _TermCfg(0.0, zero_func)),
        ("live_term", _TermCfg(1.0, live_func)),
    ])

    for _ in range(3):
        manager.compute(DT)

    assert zero_func.calls == 0, "weight==0.0 term must be skipped entirely"
    assert torch.all(manager._episode_sums["zeroed_gait_term"] == 0.0), (
        "Episode_Reward telemetry for a weight-0 term stays flat -- the phase-out "
        "must floor at epsilon to keep its gate metric alive"
    )
    assert torch.all(manager._step_reward[:, 0] == 0.0)
    assert live_func.calls == 3


def test_epsilon_weight_term_still_computes_and_logs():
    eps = 1e-3
    eps_func = _CountingFunc(0.5)
    manager = _make_manager([
        ("annealed_clock_term", _TermCfg(eps, eps_func)),
    ])

    reward = manager.compute(DT)

    assert eps_func.calls == 1, "epsilon-floored term must still be evaluated"
    expected = 0.5 * eps * DT
    assert torch.allclose(manager._episode_sums["annealed_clock_term"],
                          torch.full((NUM_ENVS,), expected))
    assert torch.allclose(manager._step_reward[:, 0], torch.full((NUM_ENVS,), 0.5 * eps))
    assert torch.allclose(reward, torch.full((NUM_ENVS,), expected))


def test_total_reward_clips_at_zero_when_penalties_exceed_income():
    income = _CountingFunc(0.3)
    penalty = _CountingFunc(-2.0)
    manager = _make_manager([
        ("income", _TermCfg(1.0, income)),
        ("penalty", _TermCfg(1.0, penalty)),
    ])

    reward = manager.compute(DT)

    assert torch.all(reward == 0.0), "net-negative step must clip to the >=0 floor"
    # Telemetry is NOT clipped -- episodic sums keep the true signed values.
    assert torch.all(manager._episode_sums["penalty"] < 0.0)


def test_total_reward_unclipped_when_income_covers_penalties():
    income = _CountingFunc(1.0)
    penalty = _CountingFunc(-0.4)
    manager = _make_manager([
        ("income", _TermCfg(1.0, income)),
        ("penalty", _TermCfg(1.0, penalty)),
    ])

    reward = manager.compute(DT)

    expected = (1.0 - 0.4) * DT
    assert torch.allclose(reward, torch.full((NUM_ENVS,), expected))


def test_episode_sums_scale_linearly_with_weight_for_unit_weight_normalization():
    """Unit-weight normalization (income / w) is valid only because the episodic sum
    is linear in the weight at constant func output."""
    func_hi = _CountingFunc(0.5)
    func_lo = _CountingFunc(0.5)
    manager = _make_manager([
        ("term_full", _TermCfg(1.0, func_hi)),
        ("term_half", _TermCfg(0.5, func_lo)),
    ])

    manager.compute(DT)

    full = manager._episode_sums["term_full"]
    half = manager._episode_sums["term_half"]
    assert torch.allclose(full / 1.0, half / 0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
