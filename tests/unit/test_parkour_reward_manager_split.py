"""Unit tests for ``ParkourRewardManager``'s opt-in ``reward_category_split`` telemetry
(stride/posture plan, step 4).

Pins:

1. Split-on vs split-off bitwise identity of ``_reward_buf``, ``_episode_sums``,
   ``_step_reward`` and the base ``Episode_Reward/*`` extras -- the split only READS.
   Split-off never touches ``env.command_manager`` at all.
2. Per-category term sums and dwell-step counts on a toy with hand-computed values,
   including a step whose total goes negative (so the category split of it, and the whole
   category's ``pre_clip_total``/``post_clip_total``, cross the manager's own zero clip).
3. ``clip_bind_frac`` and ``clip_absorbed_event`` on that same hand-computed case.
4. ``reset(env_ids)`` clears and reports only the reset envs' accumulators.
5. The per-category buffers grow (without losing earlier totals) when a higher category
   index is seen for the first time.

Uses the stub pattern of ``test_parkour_reward_manager_epsilon.py``: the production module is
exec'd with minimal ``isaaclab``/``omni`` stubs so ``compute()``/``reset()`` run verbatim, and
the manager is built with ``__new__`` (no ``RewardManager.__init__``, no sim).
"""

from __future__ import annotations

import importlib.util
import math
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


class _StubRewardManager:
    """Stand-in for ``isaaclab.managers.RewardManager`` with a real ``reset()`` -- unlike the
    ``RewardManager=object`` placeholder in ``test_parkour_reward_manager_epsilon.py`` (which
    never calls ``reset()``), this test's ``super().reset(env_ids)`` needs the base class's
    actual episodic-sum-extras behavior to run underneath the override. Copied from
    ``isaaclab/managers/reward_manager.py`` (``RewardManager.reset``).
    """

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras


def _using_real_isaac() -> bool:
    """True only if the REAL ``omni``/``isaaclab`` packages are importable -- never true just
    because some other test module already left a stub in ``sys.modules`` (``find_spec`` on an
    already-imported name returns that module's own ``__spec__``, which a bare
    ``types.ModuleType()`` stub leaves ``None``)."""
    try:
        return (
            importlib.util.find_spec("omni.kit.app") is not None
            and importlib.util.find_spec("isaaclab.managers") is not None
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_manager_class():
    if not _using_real_isaac():  # pragma: no branch - exercised only inside the isaac venv otherwise
        if "omni.kit.app" not in sys.modules:
            omni = _install_stub("omni")
            kit = _install_stub("omni.kit")
            app = _install_stub("omni.kit.app")
            omni.kit = kit
            kit.app = app
        isaaclab = sys.modules.get("isaaclab") or _install_stub("isaaclab")
        managers = sys.modules.get("isaaclab.managers") or _install_stub("isaaclab.managers")
        # Always ours, even if another test module already cached a bare ``RewardManager=object``
        # stub here first: that one cannot ``reset()``, and this file's tests need it to.
        managers.RewardManager = _StubRewardManager
        isaaclab.managers = managers
        sys.modules["isaaclab"] = isaaclab
        sys.modules["isaaclab.managers"] = managers

    spec = importlib.util.spec_from_file_location("parkour_reward_manager_split_under_test", MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ParkourRewardManager


ParkourRewardManager = _load_manager_class()

NUM_ENVS = 4
DT = 0.02


# --------------------------------------------------------------------------------- test doubles
class _TermCfg:
    def __init__(self, weight, func, params=None):
        self.weight = weight
        self.func = func
        self.params = params or {}


class _SequenceFunc:
    """Reward func stub returning the next row of a fixed ``[T, N]`` table each call."""

    def __init__(self, rows):
        self.rows = [torch.as_tensor(row, dtype=torch.float32) for row in rows]
        self.calls = 0

    def __call__(self, env, **params):
        row = self.rows[self.calls]
        self.calls += 1
        return row.clone()


class _FakeCommandTerm:
    """Stub command term: a fixed ``[T, N]`` category table, one row consumed per call."""

    def __init__(self, category_rows, category_names):
        self.TELEMETRY_CATEGORIES = category_names
        self._rows = [torch.as_tensor(row, dtype=torch.long) for row in category_rows]
        self.calls = 0

    def telemetry_category(self) -> torch.Tensor:
        row = self._rows[self.calls]
        self.calls += 1
        return row.clone()


class _AssertingCommandManager:
    """Raises if ``get_term`` is ever called -- proves the split-off path never reads it."""

    def get_term(self, name):
        raise AssertionError(f"command_manager.get_term({name!r}) must not be called when split is off")


class _FakeCommandManager:
    def __init__(self, term, expected_name=None):
        self._term = term
        self._expected_name = expected_name

    def get_term(self, name):
        if self._expected_name is not None:
            assert name == self._expected_name, f"expected command term {self._expected_name!r}, got {name!r}"
        return self._term


class _FakeCfg(types.SimpleNamespace):
    pass


class _FakeEnv:
    def __init__(self, cfg, command_manager=None, num_envs=NUM_ENVS, dt=DT):
        self.cfg = cfg
        self.command_manager = command_manager
        self.step_dt = dt
        self.max_episode_length_s = dt * 1000.0


def _make_manager(term_specs, env, num_envs=NUM_ENVS):
    """Build a ParkourRewardManager without ``RewardManager.__init__`` (no sim needed).

    ``compute()``/``reset()`` only touch the private buffers set up here, so the production
    code paths run verbatim (parkour_reward_manager.py).
    """
    manager = ParkourRewardManager.__new__(ParkourRewardManager)
    manager._env = env
    manager._term_names = [name for name, _ in term_specs]
    manager._term_cfgs = [cfg for _, cfg in term_specs]
    manager._class_term_cfgs = []  # base reset() iterates this; empty since no class-based terms here
    manager._reward_buf = torch.zeros(num_envs)
    manager._step_reward = torch.zeros(num_envs, len(term_specs))
    manager._episode_sums = {name: torch.zeros(num_envs) for name, _ in term_specs}
    return manager


# ------------------------------------------------------------------------- 1. bitwise identity
def test_split_on_is_bitwise_identical_to_split_off_and_never_reads_command_manager_when_off():
    n_envs = 3
    n_steps = 5
    income_rows = [[1.0, -0.5, 2.0], [0.5, 0.2, -3.0], [2.0, 2.0, 2.0], [0.1, 0.1, 0.1], [-1.0, 0.5, 0.5]]
    penalty_rows = [[-0.2, -0.1, -0.3], [-0.1, 0.0, 0.0], [-5.0, 0.0, -0.1], [0.0, -0.2, -0.2], [-0.3, -0.1, -0.1]]
    cat_rows = [[0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]

    def _managers():
        specs = [
            ("income", _TermCfg(1.5, _SequenceFunc(income_rows))),
            ("penalty", _TermCfg(1.0, _SequenceFunc(penalty_rows))),
        ]
        return specs

    cfg_on = _FakeCfg(reward_category_split=True)  # reward_category_command/_event_terms left at defaults
    term_on = _FakeCommandTerm(cat_rows, ("fwd", "side"))
    env_on = _FakeEnv(cfg_on, _FakeCommandManager(term_on, expected_name="base_velocity"), num_envs=n_envs)
    mgr_on = _make_manager(_managers(), env_on, num_envs=n_envs)

    cfg_off = _FakeCfg()  # reward_category_split absent entirely, like every shipped cfg
    env_off = _FakeEnv(cfg_off, _AssertingCommandManager(), num_envs=n_envs)
    mgr_off = _make_manager(_managers(), env_off, num_envs=n_envs)

    for _ in range(n_steps):
        r_on = mgr_on.compute(DT)
        r_off = mgr_off.compute(DT)
        assert torch.equal(r_on, r_off)

    assert torch.equal(mgr_on._reward_buf, mgr_off._reward_buf)
    assert torch.equal(mgr_on._step_reward, mgr_off._step_reward)
    for name in ("income", "penalty"):
        assert torch.equal(mgr_on._episode_sums[name], mgr_off._episode_sums[name])

    extras_on = mgr_on.reset(None)
    extras_off = mgr_off.reset(None)
    for name in ("income", "penalty"):
        key = f"Episode_Reward/{name}"
        assert torch.equal(extras_on[key], extras_off[key])

    assert not any(k.startswith("Episode_RewardCat/") for k in extras_off)
    assert any(k.startswith("Episode_RewardCat/") for k in extras_on)


# ---------------------------------------------------------------- 2/3. hand-computed toy case
def _toy_manager():
    """3 steps, 2 envs, category table:

        step0: (fwd, fwd)   step1: (fwd, side)   step2: (side, side)

    ``reward_stride_length`` is the (default) event term; ``big_penalty`` drives env0's step-1
    total negative, so that step both clips (post != pre) and absorbs stride's own pay.
    """
    stride_rows = [[0.5, 0.2], [0.4, 0.3], [0.6, 0.25]]
    fwd_rows = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    penalty_rows = [[0.0, 0.0], [-2.0, 0.0], [0.0, 0.0]]
    cat_rows = [[0, 0], [0, 1], [1, 1]]

    specs = [
        ("reward_stride_length", _TermCfg(1.0, _SequenceFunc(stride_rows))),
        ("forward_progress", _TermCfg(1.0, _SequenceFunc(fwd_rows))),
        ("big_penalty", _TermCfg(1.0, _SequenceFunc(penalty_rows))),
    ]
    cfg = _FakeCfg(reward_category_split=True)  # event terms default to ("reward_stride_length",)
    term = _FakeCommandTerm(cat_rows, ("fwd", "side"))
    env = _FakeEnv(cfg, _FakeCommandManager(term), num_envs=2)
    return _make_manager(specs, env, num_envs=2)


def test_toy_per_category_term_sums_and_dwell():
    mgr = _toy_manager()
    for _ in range(3):
        mgr.compute(DT)
    extras = mgr.reset(None)

    assert extras["Episode_RewardCat/fwd/dwell_s"].item() == pytest.approx(3 * DT)
    assert extras["Episode_RewardCat/side/dwell_s"].item() == pytest.approx(3 * DT)

    assert extras["Episode_RewardCat/fwd/forward_progress"].item() == pytest.approx(0.1)
    assert extras["Episode_RewardCat/side/forward_progress"].item() == pytest.approx(0.1)

    assert extras["Episode_RewardCat/fwd/reward_stride_length"].item() == pytest.approx(0.022 / (3 * DT))
    assert extras["Episode_RewardCat/side/reward_stride_length"].item() == pytest.approx(0.023 / (3 * DT))

    assert extras["Episode_RewardCat/fwd/big_penalty"].item() == pytest.approx(-0.04 / (3 * DT))
    assert extras["Episode_RewardCat/side/big_penalty"].item() == pytest.approx(0.0)


def test_toy_clip_totals_and_bind_frac():
    mgr = _toy_manager()
    for _ in range(3):
        mgr.compute(DT)
    extras = mgr.reset(None)

    assert extras["Episode_RewardCat/fwd/pre_clip_total"].item() == pytest.approx(-0.012 / (3 * DT))
    assert extras["Episode_RewardCat/fwd/post_clip_total"].item() == pytest.approx(0.018 / (3 * DT))
    assert extras["Episode_RewardCat/fwd/clip_bind_frac"].item() == pytest.approx(1.0 / 3.0)

    assert extras["Episode_RewardCat/side/pre_clip_total"].item() == pytest.approx(0.029 / (3 * DT))
    assert extras["Episode_RewardCat/side/post_clip_total"].item() == pytest.approx(0.029 / (3 * DT))
    assert extras["Episode_RewardCat/side/clip_bind_frac"].item() == pytest.approx(0.0)


def test_toy_clip_absorbed_event():
    mgr = _toy_manager()
    for _ in range(3):
        mgr.compute(DT)
    extras = mgr.reset(None)

    # fwd: env0's step-1 stride pay (0.008) is fully absorbed by the total-floor clip.
    assert extras["Episode_RewardCat/fwd/clip_absorbed_event"].item() == pytest.approx(0.008 / 0.022)
    # side: every dwell step there has headroom, so nothing is absorbed.
    assert extras["Episode_RewardCat/side/clip_absorbed_event"].item() == pytest.approx(0.0)


def test_toy_reward_buf_and_episode_sums_are_the_plain_manager_values():
    """Sanity: the split bookkeeping did not perturb the values it is reading."""
    mgr = _toy_manager()
    totals = []
    for _ in range(3):
        totals.append(mgr.compute(DT).clone())
    # step1 env0 clips to 0 (pre-clip total was -0.03); every other entry is unclipped.
    assert totals[0].tolist() == pytest.approx([0.012, 0.006])
    assert totals[1].tolist() == pytest.approx([0.0, 0.008])
    assert totals[2].tolist() == pytest.approx([0.014, 0.007])


# ------------------------------------------------------------------ 4. partial-envs reset
def test_reset_clears_and_reports_only_the_reset_envs():
    rows = [[1.0, 5.0], [1.0, 5.0]]
    cat_rows = [[0, 0], [0, 0]]
    specs = [("t1", _TermCfg(1.0, _SequenceFunc(rows)))]
    cfg = _FakeCfg(reward_category_split=True, reward_category_command="cmd")
    term = _FakeCommandTerm(cat_rows, ("only_cat",))
    env = _FakeEnv(cfg, _FakeCommandManager(term, expected_name="cmd"), num_envs=2)
    mgr = _make_manager(specs, env, num_envs=2)

    mgr.compute(DT)
    mgr.compute(DT)

    extras_env0 = mgr.reset([0])
    assert extras_env0["Episode_RewardCat/only_cat/t1"].item() == pytest.approx(1.0)
    assert extras_env0["Episode_RewardCat/only_cat/dwell_s"].item() == pytest.approx(2 * DT)

    # env0's accumulators are zeroed; env1's are untouched.
    assert mgr._cat_dwell[0, 0].item() == 0.0
    assert mgr._cat_dwell[1, 0].item() == 2.0
    assert mgr._cat_term_sum[1, 0, 0].item() == pytest.approx(2 * (5.0))

    extras_env1 = mgr.reset([1])
    assert extras_env1["Episode_RewardCat/only_cat/t1"].item() == pytest.approx(5.0)
    assert mgr._cat_dwell[1, 0].item() == 0.0

    # Nothing left dwelling anywhere -> a further reset logs no Episode_RewardCat key.
    extras_none_left = mgr.reset(None)
    # every known category is still logged (the runner logs only the first step's keys), with NaN
    # rates the runner's nanmean ignores and zero dwell
    assert math.isnan(extras_none_left["Episode_RewardCat/only_cat/t1"].item())
    assert extras_none_left["Episode_RewardCat/only_cat/dwell_s"].item() == 0.0


# ------------------------------------------------------------------------ 5. buffer growth
def test_category_buffer_grows_without_losing_earlier_totals():
    rows = [[1.0], [1.0], [1.0]]
    cat_rows = [[0], [0], [2]]  # category index 2 appears only on the third call
    specs = [("t1", _TermCfg(1.0, _SequenceFunc(rows)))]
    cfg = _FakeCfg(reward_category_split=True)
    term = _FakeCommandTerm(cat_rows, ("a", "b", "c"))
    env = _FakeEnv(cfg, _FakeCommandManager(term), num_envs=1)
    mgr = _make_manager(specs, env, num_envs=1)

    mgr.compute(DT)
    mgr.compute(DT)
    assert mgr._cat_dwell.shape[1] == 1  # only category 0 seen so far

    mgr.compute(DT)  # category 2 seen -> buffers must grow to 3 columns
    assert mgr._cat_dwell.shape[1] == 3
    # category 0's dwell/sum from before the growth must survive it.
    assert mgr._cat_dwell[0, 0].item() == 2.0
    assert mgr._cat_term_sum[0, 0, 0].item() == pytest.approx(2.0)

    extras = mgr.reset(None)
    assert extras["Episode_RewardCat/a/t1"].item() == pytest.approx(1.0)
    assert extras["Episode_RewardCat/c/t1"].item() == pytest.approx(1.0)
    assert math.isnan(extras["Episode_RewardCat/b/t1"].item())  # zero dwell -> NaN, key kept
    assert extras["Episode_RewardCat/b/dwell_s"].item() == 0.0



# ------------------------------------------------- 6. pooling across envs, zero event pay
def test_reset_pools_over_envs_by_dwell_and_zero_event_pay_logs_zero():
    """Rates pool over the reset envs as sum(value) / sum(dwell) (NOT a mean of per-env
    rates), a weight-0 event term (never computed) logs clip_absorbed_event 0, and every
    logged value is a 0-dim tensor like the base ``Episode_Reward/*`` extras."""
    rows = [[1.0, 4.0], [1.0, 4.0], [1.0, 4.0]]
    cat_rows = [[0, 0], [0, 1], [0, 1]]  # env0: 3 steps in cat0; env1: 1 step cat0, 2 cat1
    stride_rows = [[9.0, 9.0]] * 3
    specs = [
        ("t1", _TermCfg(1.0, _SequenceFunc(rows))),
        ("reward_stride_length", _TermCfg(0.0, _SequenceFunc(stride_rows))),
    ]
    cfg = _FakeCfg(reward_category_split=True)
    term = _FakeCommandTerm(cat_rows, ("a", "b"))
    env = _FakeEnv(cfg, _FakeCommandManager(term), num_envs=2)
    mgr = _make_manager(specs, env, num_envs=2)
    for _ in range(3):
        mgr.compute(DT)
    extras = mgr.reset(torch.tensor([0, 1]))

    # cat a: env0 3 x 1.0 + env1 1 x 4.0 over 4 dwell steps -> 7/4 (a per-env mean would be 2.5).
    assert extras["Episode_RewardCat/a/t1"].item() == pytest.approx(7.0 / 4.0)
    assert extras["Episode_RewardCat/a/dwell_s"].item() == pytest.approx(4 * DT)
    assert extras["Episode_RewardCat/b/t1"].item() == pytest.approx(4.0)
    assert extras["Episode_RewardCat/b/dwell_s"].item() == pytest.approx(2 * DT)
    for c in ("a", "b"):
        assert extras[f"Episode_RewardCat/{c}/reward_stride_length"].item() == 0.0
        assert extras[f"Episode_RewardCat/{c}/clip_absorbed_event"].item() == 0.0
        assert extras[f"Episode_RewardCat/{c}/clip_bind_frac"].item() == 0.0
    assert all(v.dim() == 0 for k, v in extras.items() if k.startswith("Episode_RewardCat/"))
    assert all(torch.isfinite(v) for k, v in extras.items() if k.startswith("Episode_RewardCat/"))

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
